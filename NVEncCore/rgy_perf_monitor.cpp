// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
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

CPerfMonitor::CPerfMonitor() {
    memset(m_info, 0, sizeof(m_info));
    memset(&m_pipes, 0, sizeof(m_pipes));
    memset(&m_QueueInfo, 0, sizeof(m_QueueInfo));
#if ENABLE_METRIC_FRAMEWORK
    m_pManager = nullptr;
#endif //#if ENABLE_METRIC_FRAMEWORK

    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);
    m_nLogicalCPU = cpu_info.logical_cores;
    m_thAudProcThread = NULL;
    m_thEncThread = NULL;
    m_thOutThread = NULL;
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
    m_pQSVLog.reset();
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
    std::shared_ptr<RGYLog> pQSVLog) {
    clear();
    m_pQSVLog = pQSVLog;

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
            m_pQSVLog->write(RGY_LOG_WARN, _T("Failed to open performance monitor log file: %s\n"), m_sMonitorFilename.c_str());
            m_pQSVLog->write(RGY_LOG_WARN, _T("performance monitoring disabled.\n"));
            return 1;
        }
    }
#if ENABLE_METRIC_FRAMEWORK
    //LoadAllを使用する場合、下記のように使わないモジュールを書くことで取得するモジュールを制限できる
    //putenv("GM_EXTENSION_LIB_SKIP_LIST=SEPPublisher,PVRPublisher,CPUInfoPublisher,RenderPerfPublisher");
    m_pLoader = ExtensionLoader::Create();
    //m_pLoader->AddSearchPath(loadPath.c_str());
    if (m_pLoader->Load("DefaultManager") == 0) {
        pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load DefaultManager\n"));
    } else if (m_pLoader->CommitExtensions() == 0) {
    //} else if (m_pLoader->Load("LogPublisher") == 0) {
        //pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load LogPublisher\n"));
    //下記のようにLoadAllでもよいが非常に重い
    //} else if (m_pLoader->LoadAll() == 0) {
        //pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load Metric dlls\n"));
    //mfxの使用率をとるには下記の2つが必要
    } else if (m_pLoader->Load("MediaPerfPublisher") == 0) {
        pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load MediaPerfPublisher\n"));
    } else if (m_pLoader->Load("RenderPerfPublisher") == 0) {
        pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load RenderPerfPublisher\n"));
    //以下でGPU平均使用率などがとれるはずだが・・・
    //} else if (m_pLoader->Load("GfxDrvSampledPublisher") == 0) {
        //pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to load GfxDrvSampledPublisher\n"));
    } else if (m_pLoader->CommitExtensions() == 0) {
        //pQSVLog->write(RGY_LOG_DEBUG, _T("PerfMonitor: Failed to CommitExtensions\n"));
    } else {
        //定義した情報の受け取り口を登録
        m_pLoader->AddExtension("CQSVConsumer", &m_Consumer);
        m_pManager.reset(GM_GET_DEFAULT_CLIENT_MANAGER(m_pLoader));
        if (m_pManager == nullptr) {
            pQSVLog->write(RGY_LOG_WARN, _T("No default Client Manager available\n"));
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
                    pQSVLog->write(RGY_LOG_WARN, _T("Failure to subscribe %s metric: %d.\n"), char_to_tstring(metric->second).c_str(), status);
                } else {
                    pQSVLog->write(RGY_LOG_DEBUG, _T("subscribed %s metric\n"), char_to_tstring(metric->second).c_str());
                    subscribedMetrics[metric->first] = metric->second;
                }
            }
            m_Consumer.AddMetrics(subscribedMetrics);
            if (subscribedMetrics.size() != _countof(METRIC_NAMES)) {
                pQSVLog->write(RGY_LOG_DEBUG, _T("metrics was not fully load, disable metric framework features.\n"));
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
        m_pProcess = std::unique_ptr<RGYPipeProcess>(new CPipeProcessLinux());
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
            m_pQSVLog->write(RGY_LOG_WARN, _T("Failed to create file qsvencc_perf_monitor.pyw for performance monitor plot.\n"));
            m_pQSVLog->write(RGY_LOG_WARN, _T("performance monitor plot disabled.\n"));
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
                m_pQSVLog->write(RGY_LOG_WARN, _T("Failed to run performance monitor plot.\n"));
                m_pQSVLog->write(RGY_LOG_WARN, _T("performance monitor plot disabled.\n"));
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

    m_nSelectOutputLog &= m_nSelectCheck;
    m_nSelectOutputPlot &= m_nSelectCheck;

    pQSVLog->write(RGY_LOG_DEBUG, _T("Performace Monitor: %s\n"), CPerfMonitor::SelectedCounters(m_nSelectOutputLog).c_str());
    pQSVLog->write(RGY_LOG_DEBUG, _T("Performace Plot   : %s\n"), CPerfMonitor::SelectedCounters(m_nSelectOutputPlot).c_str());

    write_header(m_fpLog.get(),   m_nSelectOutputLog);
    write_header(m_pipes.f_stdin, m_nSelectOutputPlot);

    if ((m_nSelectOutputLog || m_nSelectOutputPlot) && (m_fpLog || m_pipes.f_stdin)) {
        m_thCheck = std::thread(loader, this);
    }
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
    pInfoNew->gpu_info_valid = FALSE;
#if ENABLE_METRIC_FRAMEWORK
    QSVGPUInfo qsvinfo = { 0 };
    if (m_Consumer.getMFXLoad(&qsvinfo)) {
        pInfoNew->gpu_info_valid = TRUE;
        pInfoNew->mfx_load_percent = qsvinfo.dMFXLoad;
        pInfoNew->gpu_load_percent = qsvinfo.dEULoad;
        pInfoNew->gpu_clock = qsvinfo.dGPUFreq;
    } else {
#endif //#if ENABLE_METRIC_FRAMEWORK
        pInfoNew->gpu_clock = 0.0;
        pInfoNew->gpu_load_percent = 0.0;
        GPUZ_SH_MEM gpu_info = { 0 };
        if (0 == get_gpuz_info(&gpu_info)) {
            pInfoNew->gpu_info_valid = TRUE;
            pInfoNew->gpu_load_percent = gpu_load(&gpu_info);
            pInfoNew->gpu_clock = gpu_core_clock(&gpu_info);
        }
#if ENABLE_METRIC_FRAMEWORK
    }
#endif //#if ENABLE_METRIC_FRAMEWORK
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
                m_pQSVLog->write(RGY_LOG_WARN, _T("Error occured running python for perf-monitor-plot.\n"));
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
