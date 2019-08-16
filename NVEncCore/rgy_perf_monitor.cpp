// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#include <chrono>
#include <cassert>
#include <thread>
#include <memory>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <string>
#include "rgy_status.h"
#include "rgy_perf_monitor.h"
#include "cpu_info.h"
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_pipe.h"
#include "gpuz_info.h"
#if defined(_WIN32) || defined(_WIN64)
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

extern "C" {
extern char _binary_PerfMonitor_perf_monitor_pyw_start[];
extern char _binary_PerfMonitor_perf_monitor_pyw_end[];
extern char _binary_PerfMonitor_perf_monitor_pyw_size[];
}

#endif //#if defined(_WIN32) || defined(_WIN64)

#if ENABLE_METRIC_FRAMEWORK
#pragma comment(lib, "gmframework.lib")
#pragma comment(lib, "building_blocks.lib")

static const char *METRIC_NAMES[] = {
    "com.intel.media.mfx_usage",
    "com.intel.media.eu_usage",
    //"com.intel.gpu.avg_gpu_core_frequency_mhz",
};

void CQSVConsumer::SetValue(const std::string& metricName, double value) {
    if (metricName == METRIC_NAMES[0]) {
        m_QSVInfo.dMFXLoad = value;
    } else if (metricName == METRIC_NAMES[1]) {
        m_QSVInfo.dEULoad = value;
    } else if (metricName == METRIC_NAMES[2]) {
        m_QSVInfo.dGPUFreq = value;
    }
}

#pragma warning(push)
#pragma warning(disable: 4100)
void CQSVConsumer::OnMetricUpdated(uint32_t count, MetricHandle * metrics, const uint64_t * types, const void ** buffers, uint64_t * sizes) {
    m_bInfoValid = true;
    for (uint32_t i = 0; i < count; i++) {
        const auto& metricName = m_MetricsUsed[metrics[i]];
        switch (types[i]) {
        case DM_OPCODE_TIME_STAMPED_DOUBLE:
        {
            DM_UINT64 tsc;
            DM_DOUBLE value;
            if (Decode_TIME_STAMPED_DOUBLE(&buffers[i], &tsc, &value)) {
                SetValue(metricName, value);
            }
        }
        break;
        case DM_OPCODE_TIME_STAMPED_FLOAT:
        {
            DM_UINT64 tsc;
            DM_FLOAT value;
            if (Decode_TIME_STAMPED_FLOAT(&buffers[i], &tsc, &value)) {
                SetValue(metricName, value);
            }
        }
        break;
        case DM_OPCODE_TIME_STAMPED_UINT64:
        {
            DM_UINT64 tsc;
            DM_UINT64 value;
            if (Decode_TIME_STAMPED_UINT64(&buffers[i], &tsc, &value)) {
                SetValue(metricName, (double)value);
            }
        }
        break;
        case DM_OPCODE_TIME_STAMPED_UINT32:
        {
            DM_UINT64 tsc;
            DM_UINT32 value;
            if (Decode_TIME_STAMPED_UINT32(&buffers[i], &tsc, &value)) {
                SetValue(metricName, value);
            }
        }
        break;
        default:
            break;
        }
    }
};
#pragma warning(pop)

void CQSVConsumer::AddMetrics(const std::map<MetricHandle, std::string>& metrics) {
    m_MetricsUsed.insert(metrics.cbegin(), metrics.cend());
}

#endif //#if ENABLE_METRIC_FRAMEWORK

#if ENCODER_NVENC
#if ENABLE_NVML
const TCHAR *nvmlErrStr(nvmlReturn_t ret) {
    switch (ret) {
    case NVML_SUCCESS:                   return _T("The operation was successful");
    case NVML_ERROR_UNINITIALIZED:       return _T("case NVML was not first initialized with case NVMLInit()");
    case NVML_ERROR_INVALID_ARGUMENT:    return _T("A supplied argument is invalid");
    case NVML_ERROR_NOT_SUPPORTED:       return _T("The requested operation is not available on target device");
    case NVML_ERROR_NO_PERMISSION:       return _T("The current user does not have permission for operation");
    case NVML_ERROR_ALREADY_INITIALIZED: return _T("Deprecated: Multiple initializations are now allowed through ref counting");
    case NVML_ERROR_NOT_FOUND:           return _T("A query to find an object was unsuccessful");
    case NVML_ERROR_INSUFFICIENT_SIZE:   return _T("An input argument is not large enough");
    case NVML_ERROR_INSUFFICIENT_POWER:  return _T("A device's external power cables are not properly attached");
    case NVML_ERROR_DRIVER_NOT_LOADED:   return _T("NVIDIA driver is not loaded");
    case NVML_ERROR_TIMEOUT:            return _T("User provided timeout passed");
    case NVML_ERROR_IRQ_ISSUE:          return _T("NVIDIA Kernel detected an interrupt issue with a GPU");
    case NVML_ERROR_LIBRARY_NOT_FOUND:  return _T("case NVML Shared Library couldn't be found or loaded");
    case NVML_ERROR_FUNCTION_NOT_FOUND: return _T("Local version of case NVML doesn't implement this function");
    case NVML_ERROR_CORRUPTED_INFOROM:  return _T("infoROM is corrupted");
    case NVML_ERROR_GPU_IS_LOST:        return _T("The GPU has fallen off the bus or has otherwise become inaccessible");
    case NVML_ERROR_RESET_REQUIRED:     return _T("The GPU requires a reset before it can be used again");
    case NVML_ERROR_OPERATING_SYSTEM:   return _T("The GPU control device has been blocked by the operating system/cgroups");
    case NVML_ERROR_LIB_RM_VERSION_MISMATCH:   return _T("RM detects a driver/library version mismatch");
    case NVML_ERROR_IN_USE:             return _T("An operation cannot be performed because the GPU is currently in use");
    case NVML_ERROR_UNKNOWN:
    default:                            return _T("An internal driver error occurred");
    }
}


nvmlReturn_t NVMLMonitor::LoadDll() {
    if (m_hDll) {
        CloseHandle(m_hDll);
    }
    m_hDll = LoadLibrary(NVML_DLL_PATH);
    if (m_hDll == NULL) {
        m_hDll = LoadLibrary(_T("nvml.dll"));
        if (m_hDll == NULL) {
            return NVML_ERROR_NOT_FOUND;
        }
    }
#define LOAD_NVML_FUNC(x) { \
    if ( NULL == (m_func.f_ ## x = (pf ## x)GetProcAddress(m_hDll, #x )) ) { \
        memset(&m_func, 0, sizeof(m_func)); \
        return NVML_ERROR_NOT_FOUND; \
    } \
}
    LOAD_NVML_FUNC(nvmlInit);
    LOAD_NVML_FUNC(nvmlShutdown);
    LOAD_NVML_FUNC(nvmlErrorString);
    LOAD_NVML_FUNC(nvmlDeviceGetCount);
    LOAD_NVML_FUNC(nvmlDeviceGetHandleByPciBusId);
    LOAD_NVML_FUNC(nvmlDeviceGetUtilizationRates);
    LOAD_NVML_FUNC(nvmlDeviceGetEncoderUtilization);
    LOAD_NVML_FUNC(nvmlDeviceGetDecoderUtilization);
    LOAD_NVML_FUNC(nvmlDeviceGetMemoryInfo);
    LOAD_NVML_FUNC(nvmlDeviceGetClockInfo);
    LOAD_NVML_FUNC(nvmlDeviceGetPcieThroughput);
    LOAD_NVML_FUNC(nvmlDeviceGetCurrPcieLinkGeneration);
    LOAD_NVML_FUNC(nvmlDeviceGetCurrPcieLinkWidth);
    LOAD_NVML_FUNC(nvmlDeviceGetMaxPcieLinkGeneration);
    LOAD_NVML_FUNC(nvmlDeviceGetMaxPcieLinkWidth);
    LOAD_NVML_FUNC(nvmlSystemGetDriverVersion);
    LOAD_NVML_FUNC(nvmlSystemGetNVMLVersion);

    return NVML_SUCCESS;

#undef LOAD_NVML_FUNC
}

nvmlReturn_t NVMLMonitor::Init(const std::string& pciBusId) {
    auto ret = LoadDll();
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    ret = m_func.f_nvmlInit();
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    ret = m_func.f_nvmlDeviceGetHandleByPciBusId(pciBusId.c_str(), &m_device);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NVMLMonitor::getData(NVMLMonitorInfo *info) {
    info->dataValid = false;
    if (m_hDll == NULL || m_func.f_nvmlInit == NULL || m_device == NULL) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    nvmlUtilization_t utilData;
    auto ret = m_func.f_nvmlDeviceGetUtilizationRates(m_device, &utilData);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->GPULoad = utilData.gpu;
    uint32_t value, sample;
    ret = m_func.f_nvmlDeviceGetEncoderUtilization(m_device, &value, &sample);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->VEELoad = value;
    ret = m_func.f_nvmlDeviceGetDecoderUtilization(m_device, &value, &sample);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->VEDLoad = value;
    ret = m_func.f_nvmlDeviceGetClockInfo(m_device, NVML_CLOCK_GRAPHICS, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->GPUFreq = value;
    ret = m_func.f_nvmlDeviceGetClockInfo(m_device, NVML_CLOCK_VIDEO, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->VEFreq = value;
    ret = m_func.f_nvmlDeviceGetCurrPcieLinkGeneration(m_device, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->pcieGen = value;
    ret = m_func.f_nvmlDeviceGetCurrPcieLinkWidth(m_device, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->pcieLink = value;
    ret = m_func.f_nvmlDeviceGetPcieThroughput(m_device, NVML_PCIE_UTIL_TX_BYTES, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->pcieLoadTX = value;
    ret = m_func.f_nvmlDeviceGetPcieThroughput(m_device, NVML_PCIE_UTIL_RX_BYTES, &value);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->pcieLoadRX = value;
    nvmlMemory_t mem;
    ret = m_func.f_nvmlDeviceGetMemoryInfo(m_device, &mem);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    info->memFree = mem.free;
    info->memUsage = mem.used;
    info->memMax = mem.total;
    info->dataValid = true;
    return ret;
}

nvmlReturn_t NVMLMonitor::getDriverVersionx1000(int& ver) {
    ver = 0;

    char buffer[1024];
    auto ret = m_func.f_nvmlSystemGetDriverVersion(buffer, _countof(buffer));
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    try {
        double d = std::stod(buffer);
        ver = (int)(d * 1000.0 + 0.5);
    } catch (...) {
        return NVML_ERROR_UNKNOWN;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NVMLMonitor::getMaxPCIeLink(int& gen, int& width) {
    uint32_t val = 0;
    auto ret = m_func.f_nvmlDeviceGetMaxPcieLinkGeneration(m_device, &val);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    gen = (int)val;
    ret = m_func.f_nvmlDeviceGetMaxPcieLinkWidth(m_device, &val);
    if (ret != NVML_SUCCESS) {
        return ret;
    }
    width = (int)val;
    return ret;
}

void NVMLMonitor::Close() {
    if (m_func.f_nvmlShutdown) {
        m_func.f_nvmlShutdown();
    }
    if (m_hDll) {
        FreeLibrary(m_hDll);
    }
    memset(&m_func, 0, sizeof(m_func));
}
#endif //#if ENABLE_NVML

int NVSMIInfo::getData(NVMLMonitorInfo *info, const std::string& gpu_pcibusid) {
    memset(info, 0, sizeof(info[0]));

    RGYPipeProcessWin process;
    ProcessPipe pipes = { 0 };
    pipes.stdOut.mode = PIPE_MODE_ENABLE;
    std::vector<const TCHAR *> args;
    args.push_back(NVSMI_PATH);
    args.push_back(_T("-q"));
    if (process.run(args, nullptr, &pipes, NORMAL_PRIORITY_CLASS, true, true)) {
        return 1;
    }
    if (m_NVSMIOut.length() == 0) {
        auto read_from_pipe = [&]() {
            DWORD pipe_read = 0;
            if (!PeekNamedPipe(pipes.stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
                return -1;
            if (pipe_read) {
                char read_buf[1024] = { 0 };
                ReadFile(pipes.stdOut.h_read, read_buf, sizeof(read_buf) - 1, &pipe_read, NULL);
                m_NVSMIOut += read_buf;
            }
            return (int)pipe_read;
        };

        while (WAIT_TIMEOUT == WaitForSingleObject(process.getProcessInfo().hProcess, 10)) {
            read_from_pipe();
        }
        for (;;) {
            if (read_from_pipe() <= 0) {
                break;
            }
        }
        m_NVSMIOut = tolowercase(m_NVSMIOut);
    }
    if (m_NVSMIOut.length() == 0) {
        return 1;
    }
    const auto gpu_info_list = split(m_NVSMIOut, "\ngpu ");
    for (const auto& gpu_str : gpu_info_list) {
        if (gpu_str.substr(0, gpu_str.find("\n")).find(gpu_pcibusid) != std::string::npos) {
            //対象のGPU
            auto pos_utilization = gpu_str.find("utilization");
            if (pos_utilization != std::string::npos) {
                auto pos_gpu = gpu_str.find("gpu", pos_utilization);
                if (pos_gpu != std::string::npos) {
                    auto str_gpu = trim(gpu_str.substr(pos_gpu, gpu_str.find("\n", pos_gpu) - pos_gpu));
                    if (str_gpu.length() > 0) {
                        sscanf_s(str_gpu.c_str(), "gpu : %lf %%", &info->GPULoad);
                    }
                }
                auto pos_enc = gpu_str.find("encoder", pos_utilization);
                if (pos_enc != std::string::npos) {
                    auto str_enc = trim(gpu_str.substr(pos_enc, gpu_str.find("\n", pos_enc) - pos_enc));
                    if (str_enc.length() > 0) {
                        sscanf_s(str_enc.c_str(), "encoder : %lf %%", &info->VEELoad);
                    }
                }
                auto pos_dec = gpu_str.find("decoder", pos_utilization);
                if (pos_dec != std::string::npos) {
                    auto str_dec = trim(gpu_str.substr(pos_dec, gpu_str.find("\n", pos_dec) - pos_dec));
                    if (str_dec.length() > 0) {
                        sscanf_s(str_dec.c_str(), "decoder : %lf %%", &info->VEDLoad);
                    }
                }
            }
            auto pos_mem_usage = gpu_str.find("fb memory usage");
            if (pos_mem_usage != std::string::npos) {
                auto pos_total = gpu_str.find("total", pos_mem_usage);
                if (pos_total != std::string::npos) {
                    auto str_total = trim(gpu_str.substr(pos_total, gpu_str.find("\n", pos_total) - pos_total));
                    if (str_total.length() > 0) {
                        int value = 0;
                        if (       1 == sscanf_s(str_total.c_str(), "total : %d k", &value)) {
                            info->memMax = value * (int64_t)1024;
                        } else if (1 == sscanf_s(str_total.c_str(), "total : %d m", &value)) {
                            info->memMax = value * (int64_t)(1024 * 1024);
                        } else if (1 == sscanf_s(str_total.c_str(), "total : %d g", &value)) {
                            info->memMax = value * (int64_t)(1024 * 1024 * 1024);
                        } else {
                            info->memMax = 0;
                        }
                    }
                }
                auto pos_used = gpu_str.find("used", pos_mem_usage);
                if (pos_used != std::string::npos) {
                    auto str_used = trim(gpu_str.substr(pos_used, gpu_str.find("\n", pos_used) - pos_used));
                    if (str_used.length() > 0) {
                        int value = 0;
                        if (       1 == sscanf_s(str_used.c_str(), "used : %d k", &value)) {
                            info->memUsage = value * (int64_t)1024;
                        } else if (1 == sscanf_s(str_used.c_str(), "used : %d m", &value)) {
                            info->memUsage = value * (int64_t)(1024 * 1024);
                        } else if (1 == sscanf_s(str_used.c_str(), "used : %d g", &value)) {
                            info->memUsage = value * (int64_t)(1024 * 1024 * 1024);
                        } else {
                            info->memUsage = 0;
                        }
                    }
                }
                auto pos_free = gpu_str.find("free", pos_mem_usage);
                if (pos_free != std::string::npos) {
                    auto str_free = trim(gpu_str.substr(pos_free, gpu_str.find("\n", pos_free) - pos_free));
                    if (str_free.length() > 0) {
                        int value = 0;
                        if (       1 == sscanf_s(str_free.c_str(), "free : %df k", &value)) {
                            info->memFree = value * (int64_t)1024;
                        } else if (1 == sscanf_s(str_free.c_str(), "free : %d m", &value)) {
                            info->memFree = value * (int64_t)(1024 * 1024);
                        } else if (1 == sscanf_s(str_free.c_str(), "free : %d g", &value)) {
                            info->memFree = value * (int64_t)(1024 * 1024 * 1024);
                        } else {
                            info->memFree = 0;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
#endif //#if ENCODER_NVENC

tstring CPerfMonitor::SelectedCounters(int select) {
    if (select == 0) {
        return _T("none");
    }
    tstring str;
    for (uint32_t i = 0; i < _countof(list_pref_monitor); i++) {
        if (list_pref_monitor[i].desc &&
            (select & list_pref_monitor[i].value) == list_pref_monitor[i].value) {
            if (str.length()) {
                str += _T(",");
            }
            str += list_pref_monitor[i].desc;
            select &= ~(list_pref_monitor[i].value);
        }
    }
    return str;
}

CPerfMonitor::CPerfMonitor() :
    m_nStep(0),
    m_sPywPath(),
    m_info(),
    m_thCheck(),
    m_thMainThread(),
    m_pProcess(),
    m_pipes(),
    m_thEncThread(NULL),
    m_thInThread(NULL),
    m_thOutThread(NULL),
    m_thAudProcThread(NULL),
    m_thAudEncThread(NULL),
    m_nLogicalCPU(get_cpu_info().logical_cores),
    m_pEncStatus(),
    m_nEncStartTime(0),
    m_nOutputFPSRate(0),
    m_nOutputFPSScale(0),
    m_nCreateTime100ns(0),
    m_bAbort(false),
    m_bEncStarted(false),
    m_nInterval(500),
    m_sMonitorFilename(),
    m_fpLog(),
    m_nSelectCheck(0),
    m_nSelectOutputLog(0),
    m_nSelectOutputPlot(0),
    m_QueueInfo(),
    m_pRGYLog(),
#if ENABLE_METRIC_FRAMEWORK
    m_pLoader(nullptr),
    m_pManager(),
    m_Consumer(),
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    m_nvmlMonitor(),
    m_nvmlInfo(),
#endif //#if ENABLE_NVML
#if ENABLE_GPUZ_INFO
    m_GPUZInfo(),
#endif //#if ENABLE_GPUZ_INFO
    m_bGPUZInfoValid(false)
{
    memset(m_info, 0, sizeof(m_info));
    memset(&m_pipes, 0, sizeof(m_pipes));
    memset(&m_QueueInfo, 0, sizeof(m_QueueInfo));
#if ENABLE_METRIC_FRAMEWORK
    m_pManager = nullptr;
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    memset(&m_nvmlInfo, 0, sizeof(m_nvmlInfo));
#endif
#if ENABLE_GPUZ_INFO
    memset(&m_GPUZInfo, 0, sizeof(m_GPUZInfo));
#endif //#if ENABLE_GPUZ_INFO
}

CPerfMonitor::~CPerfMonitor() {
    clear();
}

void CPerfMonitor::clear() {
    if (m_thCheck.joinable()) {
        m_bAbort = true;
        m_thCheck.join();
    }
    memset(m_info, 0, sizeof(m_info));
    memset(&m_QueueInfo, 0, sizeof(m_QueueInfo));
#if ENABLE_METRIC_FRAMEWORK
    if (m_pManager) {
        const auto metricsUsed = m_Consumer.getMetricUsed();
        for (auto metric = metricsUsed.cbegin(); metric != metricsUsed.cend(); metric++) {
            m_pManager->UnsubscribeMetric(m_Consumer, metric->first);
        }
    }
    m_pManager.reset();
#endif //#if ENABLE_METRIC_FRAMEWORK

    m_nStep = 0;
    m_thMainThread.reset();
    m_thAudProcThread = NULL;
    m_thEncThread = NULL;
    m_thOutThread = NULL;
    m_bAbort = false;
    m_bEncStarted = false;
    if (m_fpLog) {
        fprintf(m_fpLog.get(), "\n\n");
    }
    m_fpLog.reset();
    if (m_pipes.f_stdin) {
        fclose(m_pipes.f_stdin);
        m_pipes.f_stdin = NULL;
    }
    m_pProcess.reset();
    m_pRGYLog.reset();
}

int CPerfMonitor::createPerfMpnitorPyw(const TCHAR *pywPath) {
    //リソースを取り出し
    int ret = 0;
    uint32_t resourceSize = 0;
    FILE *fp = NULL;
    const char *pDataPtr = NULL;
#if defined(_WIN32) || defined(_WIN64)
    HRSRC hResource = NULL;
    HGLOBAL hResourceData = NULL;
#if BUILD_AUO
    HMODULE hModule = GetModuleHandleA(AUO_NAME);
#else
    HMODULE hModule = GetModuleHandleA(NULL);
#endif
    if (   NULL == hModule
        || NULL == (hResource = FindResource(hModule, _T("PERF_MONITOR_PYW"), _T("PERF_MONITOR_SRC")))
        || NULL == (hResourceData = LoadResource(hModule, hResource))
        || NULL == (pDataPtr = (const char *)LockResource(hResourceData))
        || 0    == (resourceSize = SizeofResource(hModule, hResource))) {
        ret = 1;
    } else
#else
    pDataPtr = _binary_PerfMonitor_perf_monitor_pyw_start;
    resourceSize = (uint32_t)(size_t)_binary_PerfMonitor_perf_monitor_pyw_size;
#endif //#if defined(_WIN32) || defined(_WIN64)
    if (_tfopen_s(&fp, pywPath, _T("wb")) || NULL == fp) {
        ret = 1;
    } else if (resourceSize != fwrite(pDataPtr, 1, resourceSize, fp)) {
        ret = 1;
    }
    if (fp)
        fclose(fp);
    return ret;
}

void CPerfMonitor::write_header(FILE *fp, int nSelect) {
    if (fp == NULL || nSelect == 0) {
        return;
    }
    std::string str;
    if (nSelect & PERF_MONITOR_CPU) {
        str += ",cpu (%)";
    }
    if (nSelect & PERF_MONITOR_CPU_KERNEL) {
        str += ",cpu kernel (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_MAIN) {
        str += ",cpu main thread (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_ENC) {
        str += ",cpu enc thread (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_AUDP) {
        str += ",cpu aud proc thread (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_AUDE) {
        str += ",cpu aud enc thread (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_IN) {
        str += ",cpu in thread (%)";
    }
    if (nSelect & PERF_MONITOR_THREAD_OUT) {
        str += ",cpu out thread (%)";
    }
    if (nSelect & PERF_MONITOR_GPU_LOAD) {
        str += ",gpu load (%)";
    }
    if (nSelect & PERF_MONITOR_GPU_CLOCK) {
        str += ",gpu clock (MHz)";
    }
    if (nSelect & PERF_MONITOR_MFX_LOAD) {
        str += ",mfx load (%)";
    }
    if (nSelect & PERF_MONITOR_VEE_LOAD) {
        str += ",video encoder load (%)";
    }
    if (nSelect & PERF_MONITOR_VED_LOAD) {
        str += ",video decoder load (%)";
    }
    if (nSelect & PERF_MONITOR_VE_CLOCK) {
        str += ",video engine clock (MHz)";
    }
    if (nSelect & PERF_MONITOR_PCIE_LOAD) {
        str += ",pcie link,pcie tx, pci rx";
    }
    if (nSelect & PERF_MONITOR_QUEUE_VID_IN) {
        str += ",queue vid in";
    }
    if (nSelect & PERF_MONITOR_QUEUE_AUD_IN) {
        str += ",queue aud in";
    }
    if (nSelect & PERF_MONITOR_QUEUE_VID_OUT) {
        str += ",queue vid out";
    }
    if (nSelect & PERF_MONITOR_QUEUE_AUD_OUT) {
        str += ",queue aud out";
    }
    if (nSelect & PERF_MONITOR_MEM_PRIVATE) {
        str += ",mem private (MB)";
    }
    if (nSelect & PERF_MONITOR_MEM_VIRTUAL) {
        str += ",mem virtual (MB)";
    }
    if (nSelect & PERF_MONITOR_FRAME_IN) {
        str += ",frame in";
    }
    if (nSelect & PERF_MONITOR_FRAME_OUT) {
        str += ",frame out";
    }
    if (nSelect & PERF_MONITOR_FPS) {
        str += ",enc speed (fps)";
    }
    if (nSelect & PERF_MONITOR_FPS_AVG) {
        str += ",enc speed avg (fps)";
    }
    if (nSelect & PERF_MONITOR_BITRATE) {
        str += ",bitrate (kbps)";
    }
    if (nSelect & PERF_MONITOR_BITRATE_AVG) {
        str += ",bitrate avg (kbps)";
    }
    if (nSelect & PERF_MONITOR_IO_READ) {
        str += ",read (MB/s)";
    }
    if (nSelect & PERF_MONITOR_IO_WRITE) {
        str += ",write (MB/s)";
    }
    str += "\n";
    fwrite(str.c_str(), 1, str.length(), fp);
    fflush(fp);
}

int CPerfMonitor::init(tstring filename, const TCHAR *pPythonPath,
    int interval, int nSelectOutputLog, int nSelectOutputPlot,
    std::unique_ptr<void, handle_deleter> thMainThread,
    std::shared_ptr<RGYLog> pRGYLog, CPerfMonitorPrm *prm) {
    clear();
    m_pRGYLog = pRGYLog;

    m_nCreateTime100ns = (int64_t)(clock() * (1e7 / CLOCKS_PER_SEC) + 0.5);
    m_sMonitorFilename = filename;
    m_nInterval = interval;
    m_nSelectOutputPlot = nSelectOutputPlot;
    m_nSelectOutputLog = nSelectOutputLog;
    m_nSelectCheck = m_nSelectOutputLog | m_nSelectOutputPlot;
    m_thMainThread = std::move(thMainThread);

    if (!m_fpLog && m_sMonitorFilename.length() > 0) {
        m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(m_sMonitorFilename.c_str(), _T("a")));
        if (!m_fpLog) {
            m_pRGYLog->write(RGY_LOG_WARN, _T("Failed to open performance monitor log file: %s\n"), m_sMonitorFilename.c_str());
            m_pRGYLog->write(RGY_LOG_WARN, _T("performance monitoring disabled.\n"));
            return 1;
        }
    }
#if ENABLE_METRIC_FRAMEWORK
    //LoadAllを使用する場合、下記のように使わないモジュールを書くことで取得するモジュールを制限できる
    //putenv("GM_EXTENSION_LIB_SKIP_LIST=SEPPublisher,PVRPublisher,CPUInfoPublisher,RenderPerfPublisher");
    m_pLoader = ExtensionLoader::Create();
    //m_pLoader->AddSearchPath(loadPath.c_str());
    if (m_pLoader->Load("DefaultManager") == 0) {
        pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load DefaultManager\n"));
    } else if (m_pLoader->CommitExtensions() == 0) {
    //} else if (m_pLoader->Load("LogPublisher") == 0) {
        //pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load LogPublisher\n"));
    //下記のようにLoadAllでもよいが非常に重い
    //} else if (m_pLoader->LoadAll() == 0) {
        //pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load Metric dlls\n"));
    //mfxの使用率をとるには下記の2つが必要
    } else if (m_pLoader->Load("MediaPerfPublisher") == 0) {
        pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load MediaPerfPublisher\n"));
    } else if (m_pLoader->Load("RenderPerfPublisher") == 0) {
        pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load RenderPerfPublisher\n"));
    //以下でGPU平均使用率などがとれるはずだが・・・
    //} else if (m_pLoader->Load("GfxDrvSampledPublisher") == 0) {
        //pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load GfxDrvSampledPublisher\n"));
    } else if (m_pLoader->CommitExtensions() == 0) {
        //pRGYLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to CommitExtensions\n"));
    } else {
        //定義した情報の受け取り口を登録
        m_pLoader->AddExtension("CQSVConsumer", &m_Consumer);
        m_pManager.reset(GM_GET_DEFAULT_CLIENT_MANAGER(m_pLoader));
        if (m_pManager == nullptr) {
            pRGYLog->write(RGY_LOG_WARN, _T("No default Client Manager available\n"));
        } else {
            RegistrySearcher regsearcher(m_pManager.get(), RESOURCE_TYPE_METRIC, PAYLOAD_TYPE_ANY, 0);
            std::map<MetricHandle, std::string> validMetrics;
            for (int i = 0; i < _countof(METRIC_NAMES); i++) {
                PathHandle h = regsearcher[METRIC_NAMES[i]];
                if (h != 0) {
                    validMetrics[h] = METRIC_NAMES[i];
                }
            }
            std::map<MetricHandle, std::string> subscribedMetrics;
            for (auto metric = validMetrics.cbegin(); metric != validMetrics.cend(); metric++) {
                GM_STATUS status = m_pManager->SubscribeMetric(m_Consumer.GetHandle(), metric->first);
                if (GM_STATUS_SUCCESS != status) {
                    pRGYLog->write(RGY_LOG_WARN, _T("Failure to subscribe %s metric: %d.\n"), char_to_tstring(metric->second).c_str(), status);
                } else {
                    pRGYLog->write(RGY_LOG_DEBUG, _T("subscribed %s metric\n"), char_to_tstring(metric->second).c_str());
                    subscribedMetrics[metric->first] = metric->second;
                }
            }
            m_Consumer.AddMetrics(subscribedMetrics);
            if (subscribedMetrics.size() != _countof(METRIC_NAMES)) {
                pRGYLog->write(RGY_LOG_DEBUG, _T("metrics was not fully load, disable metric framework features.\n"));
                if (m_pManager) {
                    const auto metricsUsed = m_Consumer.getMetricUsed();
                    for (auto metric = metricsUsed.cbegin(); metric != metricsUsed.cend(); metric++) {
                        m_pManager->UnsubscribeMetric(m_Consumer, metric->first);
                    }
                }
                m_pManager.reset();
            }
        }
    }
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    auto nvml_ret = m_nvmlMonitor.Init(prm->pciBusId);
    if (nvml_ret != NVML_SUCCESS) {
        pRGYLog->write(RGY_LOG_INFO, _T("Failed to start NVML Monitoring for \"%s\": %s.\n"), char_to_tstring(prm->pciBusId).c_str(), nvmlErrStr(nvml_ret));
    } else {
        pRGYLog->write(RGY_LOG_DEBUG, _T("Eanble NVML Monitoring\n"));
    }
#else
    UNREFERENCED_PARAMETER(prm);
#endif //#if ENABLE_NVML

    if (m_nSelectOutputPlot) {
#if defined(_WIN32) || defined(_WIN64)
        m_pProcess = std::unique_ptr<RGYPipeProcess>(new RGYPipeProcessWin());
        m_pipes.stdIn.mode = PIPE_MODE_ENABLE;
        TCHAR tempDir[1024] = { 0 };
        TCHAR tempPath[1024] = { 0 };
        GetModuleFileName(NULL, tempDir, _countof(tempDir));
        PathRemoveFileSpec(tempDir);
        PathCombine(tempPath, tempDir, strsprintf(_T("qsvencc_perf_monitor_%d.pyw"), GetProcessId(GetCurrentProcess())).c_str());
        m_sPywPath = tempPath;
        uint32_t priority = NORMAL_PRIORITY_CLASS;
#else
        m_pProcess = std::unique_ptr<RGYPipeProcess>(new RGYPipeProcessLinux());
        m_pipes.stdIn.mode = PIPE_MODE_ENABLE;
        m_sPywPath = tstring(_T("/tmp/")) + strsprintf(_T("qsvencc_perf_monitor_%d.pyw"), (int)getpid());
        uint32_t priority = 0;
#endif
        tstring sPythonPath = (pPythonPath) ? pPythonPath : _T("python");
#if defined(_WIN32) || defined(_WIN64)
        sPythonPath = tstring(_T("\"")) + sPythonPath + tstring(_T("\""));
        m_sPywPath = tstring(_T("\"")) + m_sPywPath + tstring(_T("\""));
#else
        int ret = 0;
        if (0 > (ret = system((sPythonPath + " --version > /dev/null 2>&1").c_str()))) {
            m_pRGYLog->write(RGY_LOG_WARN, _T("Failed to run \"%s\". \n")
                _T("--perf-monitor-plot requires python3.x, please set python3 path by \"--python\".\n"), sPythonPath.c_str());
            m_nSelectOutputPlot = 0;
        } else if (0 > (ret = system((sPythonPath + " -c \"print 'test'\" > /dev/null 2>&1").c_str())) || WEXITSTATUS(ret) == 0) {
            m_pRGYLog->write(RGY_LOG_WARN, _T("\"%s\" is not python3.x.\n")
                    _T("--perf-monitor-plot requires python3.x, please set python3 path by \"--python\".\n"), sPythonPath.c_str());
            m_nSelectOutputPlot = 0;
        }
#endif
        if (createPerfMpnitorPyw(m_sPywPath.c_str())) {
            m_pRGYLog->write(RGY_LOG_WARN, _T("Failed to create file qsvencc_perf_monitor.pyw for performance monitor plot.\n"));
            m_pRGYLog->write(RGY_LOG_WARN, _T("performance monitor plot disabled.\n"));
            m_nSelectOutputPlot = 0;
        } else {
            tstring sInterval = strsprintf(_T("%d"), interval);
            std::vector<const TCHAR *> args;
            args.push_back(sPythonPath.c_str());
            args.push_back(m_sPywPath.c_str());
            args.push_back(_T("-i"));
            args.push_back(sInterval.c_str());
            args.push_back(nullptr);
            if (m_pProcess->run(args, nullptr, &m_pipes, priority, false, false)) {
                m_pRGYLog->write(RGY_LOG_WARN, _T("Failed to run performance monitor plot.\n"));
                m_pRGYLog->write(RGY_LOG_WARN, _T("performance monitor plot disabled.\n"));
                m_nSelectOutputPlot = 0;
#if defined(_WIN32) || defined(_WIN64)
            } else {
                WaitForInputIdle(dynamic_cast<RGYPipeProcessWin *>(m_pProcess.get())->getProcessInfo().hProcess, INFINITE);
#endif
            }
        }
    }

    //未実装
    m_nSelectCheck &= (~PERF_MONITOR_FRAME_IN);

    //未実装
#if !(defined(_WIN32) || defined(_WIN64))
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_MAIN);
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_ENC);
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_AUDP);
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_AUDE);
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_OUT);
    m_nSelectCheck &= (~PERF_MONITOR_THREAD_IN);
    m_nSelectCheck &= (~PERF_MONITOR_GPU_CLOCK);
    m_nSelectCheck &= (~PERF_MONITOR_GPU_LOAD);
    m_nSelectCheck &= (~PERF_MONITOR_MFX_LOAD);
#endif //#if defined(_WIN32) || defined(_WIN64)

#if ENCODER_QSV
    m_nSelectCheck &= (~PERF_MONITOR_VED_LOAD);
    m_nSelectCheck &= (~PERF_MONITOR_VEE_LOAD);
    m_nSelectCheck &= (~PERF_MONITOR_VE_CLOCK);
#endif
#if ENCODER_NVENC
    m_nSelectCheck &= (~PERF_MONITOR_MFX_LOAD);
    //うまくとれてなさそう
    m_nSelectCheck &= (~PERF_MONITOR_VED_LOAD);
#endif

    m_nSelectOutputLog &= m_nSelectCheck;
    m_nSelectOutputPlot &= m_nSelectCheck;

    pRGYLog->write(RGY_LOG_DEBUG, _T("Performace Monitor: %s\n"), CPerfMonitor::SelectedCounters(m_nSelectOutputLog).c_str());
    pRGYLog->write(RGY_LOG_DEBUG, _T("Performace Plot   : %s\n"), CPerfMonitor::SelectedCounters(m_nSelectOutputPlot).c_str());

    write_header(m_fpLog.get(),   m_nSelectOutputLog);
    write_header(m_pipes.f_stdin, m_nSelectOutputPlot);

    m_thCheck = std::thread(loader, this);
    return 0;
}

void CPerfMonitor::SetEncStatus(std::shared_ptr<EncodeStatus> encStatus) {
    m_pEncStatus = encStatus;
    EncodeStatusData data = m_pEncStatus->GetEncodeData();
    m_nOutputFPSScale = data.outputFPSScale;
    m_nOutputFPSRate = data.outputFPSRate;
}

void CPerfMonitor::SetThreadHandles(HANDLE thEncThread, HANDLE thInThread, HANDLE thOutThread, HANDLE thAudProcThread, HANDLE thAudEncThread) {
    m_thEncThread = thEncThread;
    m_thInThread = thInThread;
    m_thOutThread = thOutThread;
    m_thAudProcThread = thAudProcThread;
    m_thAudEncThread = thAudEncThread;
}

void CPerfMonitor::check() {
    PerfInfo *pInfoNew = &m_info[(m_nStep + 1) & 1];
    PerfInfo *pInfoOld = &m_info[ m_nStep      & 1];
    memcpy(pInfoNew, pInfoOld, sizeof(pInfoNew[0]));

#if defined(_WIN32) || defined(_WIN64)
    const auto hProcess = GetCurrentProcess();
    auto getThreadTime = [](HANDLE hThread, PROCESS_TIME *time) {
        GetThreadTimes(hThread, (FILETIME *)&time->creation, (FILETIME *)&time->exit, (FILETIME *)&time->kernel, (FILETIME *)&time->user);
    };

    //メモリ情報
    PROCESS_MEMORY_COUNTERS mem_counters = { 0 };
    mem_counters.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    GetProcessMemoryInfo(hProcess, &mem_counters, sizeof(mem_counters));
    pInfoNew->mem_private = mem_counters.WorkingSetSize;
    pInfoNew->mem_virtual = mem_counters.PagefileUsage;

    //IO情報
    IO_COUNTERS io_counters = { 0 };
    GetProcessIoCounters(hProcess, &io_counters);
    pInfoNew->io_total_read = io_counters.ReadTransferCount;
    pInfoNew->io_total_write = io_counters.WriteTransferCount;

    //現在時刻
    uint64_t current_time = 0;
    SYSTEMTIME systime = { 0 };
    GetSystemTime(&systime);
    SystemTimeToFileTime(&systime, (FILETIME *)&current_time);

    //CPU情報
    PROCESS_TIME pt = { 0 };
    GetProcessTimes(hProcess, (FILETIME *)&pt.creation, (FILETIME *)&pt.exit, (FILETIME *)&pt.kernel, (FILETIME *)&pt.user);
    pInfoNew->time_us = (current_time - pt.creation) / 10;
    const double time_diff_inv = 1.0 / (pInfoNew->time_us - pInfoOld->time_us);

    //GPU情報
    m_bGPUZInfoValid = false;
    pInfoNew->gpu_info_valid = FALSE;
#if ENABLE_METRIC_FRAMEWORK
    QSVGPUInfo qsvinfo = { 0 };
    if (m_Consumer.getMFXLoad(&qsvinfo)) {
        pInfoNew->gpu_info_valid = TRUE;
        pInfoNew->mfx_load_percent = qsvinfo.dMFXLoad;
        pInfoNew->gpu_load_percent = qsvinfo.dEULoad;
        pInfoNew->gpu_clock = qsvinfo.GPUFreq;
    } else {
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    pInfoNew->gpu_info_valid   = FALSE;
    pInfoNew->gpu_clock = 0.0;
    pInfoNew->gpu_load_percent = 0.0;
    pInfoNew->ve_clock = 0.0;
    pInfoNew->vee_load_percent = 0.0;
    pInfoNew->ved_load_percent = 0.0;
    pInfoNew->pcie_gen = 0;
    pInfoNew->pcie_link = 0;
    pInfoNew->pcie_throughput_tx_per_sec = 0;
    pInfoNew->pcie_throughput_rx_per_sec = 0;
    NVMLMonitorInfo nvmlInfo;
    if (m_nvmlMonitor.getData(&nvmlInfo) == NVML_SUCCESS) {
        m_nvmlInfo = nvmlInfo;
        pInfoNew->gpu_info_valid   = TRUE;
        pInfoNew->gpu_clock        = m_nvmlInfo.GPUFreq;
        pInfoNew->gpu_load_percent = m_nvmlInfo.GPULoad;
        pInfoNew->ve_clock         = m_nvmlInfo.VEFreq;
        pInfoNew->vee_load_percent = m_nvmlInfo.VEELoad;
        pInfoNew->ved_load_percent = m_nvmlInfo.VEDLoad;
        pInfoNew->pcie_gen         = m_nvmlInfo.pcieGen;
        pInfoNew->pcie_link        = m_nvmlInfo.pcieLink;
        pInfoNew->pcie_throughput_tx_per_sec = m_nvmlInfo.pcieLoadTX;
        pInfoNew->pcie_throughput_rx_per_sec = m_nvmlInfo.pcieLoadRX;
    } else {
#endif //#if ENABLE_NVML
        pInfoNew->gpu_clock = 0.0;
        pInfoNew->gpu_load_percent = 0.0;
        pInfoNew->ve_clock = 0.0;
        pInfoNew->vee_load_percent = 0.0;
        pInfoNew->ved_load_percent = 0.0;
        pInfoNew->pcie_gen = 0;
        pInfoNew->pcie_link = 0;
        pInfoNew->pcie_throughput_tx_per_sec = 0;
        pInfoNew->pcie_throughput_rx_per_sec = 0;
#if ENABLE_GPUZ_INFO
        memset(&m_GPUZInfo, 0, sizeof(m_GPUZInfo));
        if (0 == get_gpuz_info(&m_GPUZInfo)) {
            m_bGPUZInfoValid = true;
            pInfoNew->gpu_info_valid = TRUE;
            pInfoNew->gpu_load_percent = gpu_load(&m_GPUZInfo);
            pInfoNew->vee_load_percent = video_engine_load(&m_GPUZInfo, nullptr);
            pInfoNew->gpu_clock = gpu_core_clock(&m_GPUZInfo);
        }
#endif //#if ENABLE_GPUZ_INFO
#if ENABLE_METRIC_FRAMEWORK || ENABLE_NVML
    }
#endif //#if ENABLE_METRIC_FRAMEWORK || ENABLE_NVML
#else
    struct rusage usage = { 0 };
    getrusage(RUSAGE_SELF, &usage);

    //現在時間
    uint64_t current_time = clock() * (1e7 / CLOCKS_PER_SEC);

    std::string proc_dir = strsprintf("/proc/%d/", (int)getpid());
    //メモリ情報
    FILE *fp_mem = popen((std::string("cat ") + proc_dir + std::string("status")).c_str(), "r");
    if (fp_mem) {
        char buffer[2048] = { 0 };
        while (NULL != fgets(buffer, _countof(buffer), fp_mem)) {
            if (nullptr != strstr(buffer, "VmSize")) {
                long long i = 0;
                if (1 == sscanf(buffer, "VmSize: %lld kB", &i)) {
                    pInfoNew->mem_virtual = i << 10;
                }
            } else if (nullptr != strstr(buffer, "VmRSS")) {
                long long i = 0;
                if (1 == sscanf(buffer, "VmRSS: %lld kB", &i)) {
                    pInfoNew->mem_private = i << 10;
                }
            }
        }
        fclose(fp_mem);
    }
    //IO情報
    FILE *fp_io = popen((std::string("cat ") + proc_dir + std::string("io")).c_str(), "r");
    if (fp_io) {
        char buffer[2048] = { 0 };
        while (NULL != fgets(buffer, _countof(buffer), fp_io)) {
            if (nullptr != strstr(buffer, "rchar:")) {
                long long i = 0;
                if (1 == sscanf(buffer, "rchar: %lld", &i)) {
                    pInfoNew->io_total_read = i;
                }
            } else if (nullptr != strstr(buffer, "wchar")) {
                long long i = 0;
                if (1 == sscanf(buffer, "wchar: %lld", &i)) {
                    pInfoNew->io_total_write = i;
                }
            }
        }
        fclose(fp_io);
    }

    //CPU情報
    pInfoNew->time_us = (current_time - m_nCreateTime100ns) / 10;
    const double time_diff_inv = 1.0 / (pInfoNew->time_us - pInfoOld->time_us);
#endif

    if (pInfoNew->time_us > pInfoOld->time_us) {
#if defined(_WIN32) || defined(_WIN64)
        pInfoNew->cpu_total_us = (pt.user + pt.kernel) / 10;
        pInfoNew->cpu_total_kernel_us = pt.kernel / 10;
#else
        int64_t cpu_user_us = usage.ru_utime.tv_sec * 1000000 + usage.ru_utime.tv_usec;
        int64_t cpu_kernel_us = usage.ru_stime.tv_sec * 1000000 + usage.ru_stime.tv_usec;
        pInfoNew->cpu_total_us = cpu_user_us + cpu_kernel_us;
        pInfoNew->cpu_total_kernel_us = cpu_kernel_us;
#endif //#if defined(_WIN32) || defined(_WIN64)

        //CPU使用率
        const double logical_cpu_inv       = 1.0 / m_nLogicalCPU;
        pInfoNew->cpu_percent        = (pInfoNew->cpu_total_us        - pInfoOld->cpu_total_us) * 100.0 * logical_cpu_inv * time_diff_inv;
        pInfoNew->cpu_kernel_percent = (pInfoNew->cpu_total_kernel_us - pInfoOld->cpu_total_kernel_us) * 100.0 * logical_cpu_inv * time_diff_inv;

        //IO情報
        pInfoNew->io_read_per_sec = (pInfoNew->io_total_read - pInfoOld->io_total_read) * time_diff_inv * 1e6;
        pInfoNew->io_write_per_sec = (pInfoNew->io_total_write - pInfoOld->io_total_write) * time_diff_inv * 1e6;

#if defined(_WIN32) || defined(_WIN64)
        //スレッドCPU使用率
        if (m_thMainThread) {
            getThreadTime(m_thMainThread.get(), &pt);
            pInfoNew->main_thread_total_active_us = (pt.user + pt.kernel) / 10;
            pInfoNew->main_thread_percent = (pInfoNew->main_thread_total_active_us - pInfoOld->main_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
        }

        if (m_thEncThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thEncThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thEncThread, &pt);
                pInfoNew->enc_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->enc_thread_percent  = (pInfoNew->enc_thread_total_active_us  - pInfoOld->enc_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->enc_thread_percent = 0.0;
            }
        }

        if (m_thAudProcThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thAudProcThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thAudProcThread, &pt);
                pInfoNew->aud_proc_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->aud_proc_thread_percent  = (pInfoNew->aud_proc_thread_total_active_us  - pInfoOld->aud_proc_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->aud_proc_thread_percent = 0.0;
            }
        }

        if (m_thAudEncThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thAudEncThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thAudEncThread, &pt);
                pInfoNew->aud_enc_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->aud_enc_thread_percent  = (pInfoNew->aud_enc_thread_total_active_us  - pInfoOld->aud_enc_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->aud_enc_thread_percent = 0.0;
            }
        }

        if (m_thInThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thInThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thInThread, &pt);
                pInfoNew->in_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->in_thread_percent  = (pInfoNew->in_thread_total_active_us  - pInfoOld->in_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->in_thread_percent = 0.0;
            }
        }

        if (m_thOutThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thOutThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thOutThread, &pt);
                pInfoNew->out_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->out_thread_percent  = (pInfoNew->out_thread_total_active_us  - pInfoOld->out_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->out_thread_percent = 0.0;
            }
        }
#endif //defined(_WIN32) || defined(_WIN64)
    }

    if (!m_bEncStarted && m_pEncStatus) {
        m_bEncStarted = m_pEncStatus->getEncStarted();
        if (m_bEncStarted) {
            m_nEncStartTime = m_pEncStatus->getStartTimeMicroSec();
        }
    }

    pInfoNew->bitrate_kbps = 0;
    pInfoNew->frames_out_byte = 0;
    pInfoNew->fps = 0.0;
    if (m_bEncStarted && m_pEncStatus) {
        EncodeStatusData data = m_pEncStatus->GetEncodeData();

        //fps情報
        pInfoNew->frames_out = data.frameTotal;
        if (pInfoNew->frames_out > pInfoOld->frames_out) {
            pInfoNew->fps_avg = pInfoNew->frames_out / (double)(current_time / 10 - m_nEncStartTime) * 1e6;
            if (pInfoNew->time_us > pInfoOld->time_us) {
                pInfoNew->fps     = (pInfoNew->frames_out - pInfoOld->frames_out) * time_diff_inv * 1e6;
            }

            //ビットレート情報
            double videoSec     = pInfoNew->frames_out * m_nOutputFPSScale / (double)m_nOutputFPSRate;
            double videoSecDiff = (pInfoNew->frames_out - pInfoOld->frames_out) * m_nOutputFPSScale / (double)m_nOutputFPSRate;

            pInfoNew->frames_out_byte = data.outFileSize;
            pInfoNew->bitrate_kbps_avg =  pInfoNew->frames_out_byte * 8.0 / videoSec * 1e-3;
            if (pInfoNew->time_us > pInfoOld->time_us) {
                pInfoNew->bitrate_kbps     = (pInfoNew->frames_out_byte - pInfoOld->frames_out_byte) * 8.0 / videoSecDiff * 1e-3;
            }
        }
    }

    m_nStep++;
}

void CPerfMonitor::write(FILE *fp, int nSelect) {
    if (fp == NULL) {
        return;
    }
    const PerfInfo *pInfo = &m_info[m_nStep & 1];
    std::string str = strsprintf("%lf", pInfo->time_us * 1e-6);
    if (nSelect & PERF_MONITOR_CPU) {
        str += strsprintf(",%lf", pInfo->cpu_percent);
    }
    if (nSelect & PERF_MONITOR_CPU_KERNEL) {
        str += strsprintf(",%lf", pInfo->cpu_kernel_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_MAIN) {
        str += strsprintf(",%lf", pInfo->main_thread_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_ENC) {
        str += strsprintf(",%lf", pInfo->enc_thread_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_AUDP) {
        str += strsprintf(",%lf", pInfo->aud_proc_thread_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_AUDE) {
        str += strsprintf(",%lf", pInfo->aud_enc_thread_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_IN) {
        str += strsprintf(",%lf", pInfo->in_thread_percent);
    }
    if (nSelect & PERF_MONITOR_THREAD_OUT) {
        str += strsprintf(",%lf", pInfo->out_thread_percent);
    }
    if (nSelect & PERF_MONITOR_GPU_LOAD) {
        str += strsprintf(",%lf", pInfo->gpu_load_percent);
    }
    if (nSelect & PERF_MONITOR_GPU_CLOCK) {
        str += strsprintf(",%lf", pInfo->gpu_clock);
    }
    if (nSelect & PERF_MONITOR_MFX_LOAD) {
        str += strsprintf(",%lf", pInfo->mfx_load_percent);
    }
    if (nSelect & PERF_MONITOR_VEE_LOAD) {
        str += strsprintf(",%lf", pInfo->vee_load_percent);
    }
    if (nSelect & PERF_MONITOR_VED_LOAD) {
        str += strsprintf(",%lf", pInfo->ved_load_percent);
    }
    if (nSelect & PERF_MONITOR_VE_CLOCK) {
        str += strsprintf(",%lf", pInfo->ve_clock);
    }
    if (nSelect & PERF_MONITOR_PCIE_LOAD) {
        str += strsprintf(",PCIe %dx%d", pInfo->pcie_gen, pInfo->pcie_link);
        str += strsprintf(",%lf", pInfo->pcie_throughput_tx_per_sec);
        str += strsprintf(",%lf", pInfo->pcie_throughput_rx_per_sec);
    }
    if (nSelect & PERF_MONITOR_QUEUE_VID_IN) {
        str += strsprintf(",%d", (int)m_QueueInfo.usage_vid_in);
    }
    if (nSelect & PERF_MONITOR_QUEUE_AUD_IN) {
        str += strsprintf(",%d", (int)m_QueueInfo.usage_aud_in);
    }
    if (nSelect & PERF_MONITOR_QUEUE_VID_OUT) {
        str += strsprintf(",%d", (int)m_QueueInfo.usage_vid_out);
    }
    if (nSelect & PERF_MONITOR_QUEUE_AUD_OUT) {
        str += strsprintf(",%d", (int)m_QueueInfo.usage_aud_out);
    }
    if (nSelect & PERF_MONITOR_MEM_PRIVATE) {
        str += strsprintf(",%.2lf", pInfo->mem_private / (double)(1024 * 1024));
    }
    if (nSelect & PERF_MONITOR_MEM_VIRTUAL) {
        str += strsprintf(",%.2lf", pInfo->mem_virtual / (double)(1024 * 1024));
    }
    if (nSelect & PERF_MONITOR_FRAME_IN) {
        str += strsprintf(",%d", pInfo->frames_in);
    }
    if (nSelect & PERF_MONITOR_FRAME_OUT) {
        str += strsprintf(",%d", pInfo->frames_out);
    }
    if (nSelect & PERF_MONITOR_FPS) {
        str += strsprintf(",%lf", pInfo->fps);
    }
    if (nSelect & PERF_MONITOR_FPS_AVG) {
        str += strsprintf(",%lf", pInfo->fps_avg);
    }
    if (nSelect & PERF_MONITOR_BITRATE) {
        str += strsprintf(",%lf", pInfo->bitrate_kbps);
    }
    if (nSelect & PERF_MONITOR_BITRATE_AVG) {
        str += strsprintf(",%lf", pInfo->bitrate_kbps_avg);
    }
    if (nSelect & PERF_MONITOR_IO_READ) {
        str += strsprintf(",%lf", pInfo->io_read_per_sec / (double)(1024 * 1024));
    }
    if (nSelect & PERF_MONITOR_IO_WRITE) {
        str += strsprintf(",%lf", pInfo->io_write_per_sec / (double)(1024 * 1024));
    }
    str += "\n";
    fwrite(str.c_str(), 1, str.length(), fp);
    if (fp == m_pipes.f_stdin) {
        fflush(fp);
    }
}

void CPerfMonitor::loader(void *prm) {
    reinterpret_cast<CPerfMonitor*>(prm)->run();
}

void CPerfMonitor::run() {
    while (!m_bAbort) {
        check();
        if (m_pProcess && !m_pProcess->processAlive()) {
            if (m_pipes.f_stdin) {
                fclose(m_pipes.f_stdin);
            }
            m_pipes.f_stdin = NULL;
            if (m_nSelectOutputPlot) {
                m_pRGYLog->write(RGY_LOG_WARN, _T("Error occured running python for perf-monitor-plot.\n"));
                m_nSelectOutputPlot = 0;
            }
        }
        write(m_fpLog.get(),   m_nSelectOutputLog);
        write(m_pipes.f_stdin, m_nSelectOutputPlot);
        std::this_thread::sleep_for(std::chrono::milliseconds(m_nInterval));
    }
    check();
    write(m_fpLog.get(),   m_nSelectOutputLog);
    write(m_pipes.f_stdin, m_nSelectOutputPlot);
}
