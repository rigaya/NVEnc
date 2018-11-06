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

#pragma once
#ifndef __RGY_PERF_MONITOR_H__
#define __RGY_PERF_MONITOR_H__

#include <thread>
#include <cstdint>
#include <climits>
#include <memory>
#include <map>
#include "cpu_info.h"
#include "rgy_util.h"
#include "rgy_pipe.h"
#include "rgy_log.h"
#include "gpuz_info.h"

#if ENABLE_METRIC_FRAMEWORK
#pragma warning(push)
#pragma warning(disable: 4456)
#pragma warning(disable: 4819)
#include <observation/gmframework.h>
#include <observation/building_blocks.h>
#pragma warning(pop)
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
#include "nvml.h"
#define NVML_DLL_PATH _T(R"(C:\Program Files\NVIDIA Corporation\nvsmi\nvml.dll)")
#endif
#define NVSMI_PATH _T(R"(C:\Program Files\NVIDIA Corporation\nvsmi\nvidia-smi.exe)")

#ifndef HANDLE
typedef void * HANDLE;
#endif

class EncodeStatus;

enum : int {
    PERF_MONITOR_CPU           = 0x00000001,
    PERF_MONITOR_CPU_KERNEL    = 0x00000002,
    PERF_MONITOR_MEM_PRIVATE   = 0x00000004,
    PERF_MONITOR_MEM_VIRTUAL   = 0x00000008,
    PERF_MONITOR_FPS           = 0x00000010,
    PERF_MONITOR_FPS_AVG       = 0x00000020,
    PERF_MONITOR_BITRATE       = 0x00000040,
    PERF_MONITOR_BITRATE_AVG   = 0x00000080,
    PERF_MONITOR_IO_READ       = 0x00000100,
    PERF_MONITOR_IO_WRITE      = 0x00000200,
    PERF_MONITOR_THREAD_MAIN   = 0x00000400,
    PERF_MONITOR_THREAD_ENC    = 0x00000800,
    PERF_MONITOR_THREAD_AUDP   = 0x00001000,
    PERF_MONITOR_THREAD_AUDE   = 0x00002000,
    PERF_MONITOR_THREAD_OUT    = 0x00004000,
    PERF_MONITOR_THREAD_IN     = 0x00008000,
    PERF_MONITOR_FRAME_IN      = 0x00010000,
    PERF_MONITOR_FRAME_OUT     = 0x00020000,
    PERF_MONITOR_GPU_LOAD      = 0x00040000,
    PERF_MONITOR_GPU_CLOCK     = 0x00080000,
    PERF_MONITOR_QUEUE_VID_IN  = 0x00100000,
    PERF_MONITOR_QUEUE_VID_OUT = 0x00200000,
    PERF_MONITOR_QUEUE_AUD_IN  = 0x00400000,
    PERF_MONITOR_QUEUE_AUD_OUT = 0x00800000,
    PERF_MONITOR_MFX_LOAD      = 0x01000000,
    PERF_MONITOR_VE_CLOCK      = 0x02000000,
    PERF_MONITOR_VEE_LOAD      = 0x04000000,
    PERF_MONITOR_VED_LOAD      = 0x08000000,
    PERF_MONITOR_ALL         = (int)UINT_MAX,
};

static const CX_DESC list_pref_monitor[] = {
    { _T("all"),         PERF_MONITOR_ALL },
    { _T("cpu"),         PERF_MONITOR_CPU | PERF_MONITOR_CPU_KERNEL | PERF_MONITOR_THREAD_MAIN | PERF_MONITOR_THREAD_ENC | PERF_MONITOR_THREAD_OUT | PERF_MONITOR_THREAD_IN },
    { _T("cpu_total"),   PERF_MONITOR_CPU },
    { _T("cpu_kernel"),  PERF_MONITOR_CPU_KERNEL },
    { _T("cpu_main"),    PERF_MONITOR_THREAD_MAIN },
    { _T("cpu_enc"),     PERF_MONITOR_THREAD_ENC },
    { _T("cpu_in"),      PERF_MONITOR_THREAD_IN },
    { _T("cpu_aud"),     PERF_MONITOR_THREAD_AUDP | PERF_MONITOR_THREAD_AUDE },
    { _T("cpu_aud_proc"),PERF_MONITOR_THREAD_AUDP },
    { _T("cpu_aud_enc"), PERF_MONITOR_THREAD_AUDE },
    { _T("cpu_out"),     PERF_MONITOR_THREAD_OUT },
    { _T("mem"),         PERF_MONITOR_MEM_PRIVATE | PERF_MONITOR_MEM_VIRTUAL },
    { _T("mem_private"), PERF_MONITOR_MEM_PRIVATE },
    { _T("mem_virtual"), PERF_MONITOR_MEM_VIRTUAL },
    { _T("io"),          PERF_MONITOR_IO_READ | PERF_MONITOR_IO_WRITE },
    { _T("io_read"),     PERF_MONITOR_IO_READ },
    { _T("io_write"),    PERF_MONITOR_IO_WRITE },
    { _T("fps"),         PERF_MONITOR_FPS },
    { _T("fps_avg"),     PERF_MONITOR_FPS_AVG },
    { _T("bitrate"),     PERF_MONITOR_BITRATE },
    { _T("bitrate_avg"), PERF_MONITOR_BITRATE_AVG },
    { _T("frame_out"),   PERF_MONITOR_FRAME_OUT },
    { _T("gpu"),         PERF_MONITOR_GPU_LOAD | PERF_MONITOR_VEE_LOAD | PERF_MONITOR_VED_LOAD | PERF_MONITOR_GPU_CLOCK | PERF_MONITOR_VE_CLOCK },
    { _T("gpu_load"),    PERF_MONITOR_GPU_LOAD },
    { _T("gpu_clock"),   PERF_MONITOR_GPU_CLOCK },
#if ENABLE_METRIC_FRAMEWORK
    { _T("mfx"),         PERF_MONITOR_MFX_LOAD },
#endif
    { _T("vee_load"),    PERF_MONITOR_VEE_LOAD },
    { _T("ved_load"),    PERF_MONITOR_VEE_LOAD },
    { _T("ve_clock"),    PERF_MONITOR_VE_CLOCK },
    { _T("queue"),       PERF_MONITOR_QUEUE_VID_IN | PERF_MONITOR_QUEUE_VID_OUT | PERF_MONITOR_QUEUE_AUD_IN | PERF_MONITOR_QUEUE_AUD_OUT },
    { nullptr, 0 }
};

struct PerfInfo {
    int64_t time_us;
    int64_t cpu_total_us;
    int64_t cpu_total_kernel_us;

    int64_t main_thread_total_active_us;
    int64_t enc_thread_total_active_us;
    int64_t aud_proc_thread_total_active_us;
    int64_t aud_enc_thread_total_active_us;
    int64_t out_thread_total_active_us;
    int64_t in_thread_total_active_us;

    int64_t mem_private;
    int64_t mem_virtual;

    int64_t io_total_read;
    int64_t io_total_write;

    int64_t frames_in;
    int64_t frames_out;
    int64_t frames_out_byte;

    double  fps;
    double  fps_avg;

    double  bitrate_kbps;
    double  bitrate_kbps_avg;

    double  io_read_per_sec;
    double  io_write_per_sec;

    double  cpu_percent;
    double  cpu_kernel_percent;

    double  main_thread_percent;
    double  enc_thread_percent;
    double  aud_proc_thread_percent;
    double  aud_enc_thread_percent;
    double  out_thread_percent;
    double  in_thread_percent;

    BOOL    gpu_info_valid;
    double  gpu_load_percent;
    double  gpu_clock;

    double  mfx_load_percent;

    double  vee_load_percent;
    double  ved_load_percent;
    double  ve_clock;
};

struct PerfOutputInfo {
    int flag;
    const TCHAR *fmt;
    ptrdiff_t offset;
};

struct PerfQueueInfo {
    size_t usage_vid_in;
    size_t usage_aud_in;
    size_t usage_vid_out;
    size_t usage_aud_out;
    size_t usage_aud_enc;
    size_t usage_aud_proc;
};

#if ENABLE_METRIC_FRAMEWORK

struct QSVGPUInfo {
    double dMFXLoad;
    double dEULoad;
    double dGPUFreq;
};

class CQSVConsumer : public IConsumer {
public:
    CQSVConsumer() : m_bInfoValid(false), m_QSVInfo(), m_MetricsUsed() {
        m_QSVInfo.dMFXLoad = 0.0;
        m_QSVInfo.dEULoad  = 0.0;
        m_QSVInfo.dGPUFreq = 0.0;
    };
    virtual void OnMetricUpdated(uint32_t count, MetricHandle * metrics, const uint64_t * types, const void ** buffers, uint64_t * sizes) override;

    void AddMetrics(const std::map<MetricHandle, std::string>& metrics);

    bool getMFXLoad(QSVGPUInfo *info) {
        if (!m_bInfoValid)
            return false;
        memcpy(info, &m_QSVInfo, sizeof(m_QSVInfo));
        return true;
    }
    const std::map<MetricHandle, std::string>& getMetricUsed() {
        return m_MetricsUsed;
    }
private:
    void SetValue(const std::string& metricName, double value);

    bool m_bInfoValid;
    QSVGPUInfo m_QSVInfo;
    std::map<MetricHandle, std::string> m_MetricsUsed;
};
#endif //#if ENABLE_METRIC_FRAMEWORK

struct NVMLMonitorInfo {
    bool bDataValid;
    double dGPULoad;
    double dGPUFreq;
    double dVEELoad;
    double dVEDLoad;
    double dVEFreq;
    int64_t nMemFree;
    int64_t nMemUsage;
    int64_t nMemMax;
};

#if ENABLE_NVML
const TCHAR *nvmlErrStr(nvmlReturn_t ret);

#define NVML_FUNCPTR(x) typedef decltype(x)* pf ## x;

NVML_FUNCPTR(nvmlInit);
NVML_FUNCPTR(nvmlShutdown);
NVML_FUNCPTR(nvmlErrorString);
NVML_FUNCPTR(nvmlDeviceGetCount);
NVML_FUNCPTR(nvmlDeviceGetHandleByPciBusId);
NVML_FUNCPTR(nvmlDeviceGetUtilizationRates);
NVML_FUNCPTR(nvmlDeviceGetEncoderUtilization);
NVML_FUNCPTR(nvmlDeviceGetDecoderUtilization);
NVML_FUNCPTR(nvmlDeviceGetMemoryInfo);
NVML_FUNCPTR(nvmlDeviceGetClockInfo);
NVML_FUNCPTR(nvmlDeviceGetPcieThroughput);
NVML_FUNCPTR(nvmlDeviceGetCurrPcieLinkGeneration);
NVML_FUNCPTR(nvmlDeviceGetCurrPcieLinkWidth);
NVML_FUNCPTR(nvmlDeviceGetMaxPcieLinkGeneration);
NVML_FUNCPTR(nvmlDeviceGetMaxPcieLinkWidth);
NVML_FUNCPTR(nvmlSystemGetDriverVersion);
NVML_FUNCPTR(nvmlSystemGetNVMLVersion);

#undef NVML_FUNCPTR

#define NVML_FUNC(x) pf ## x f_ ## x;

struct NVMLFuncList {
    NVML_FUNC(nvmlInit)
    NVML_FUNC(nvmlShutdown)
    NVML_FUNC(nvmlErrorString)
    NVML_FUNC(nvmlDeviceGetCount)
    NVML_FUNC(nvmlDeviceGetHandleByPciBusId)
    NVML_FUNC(nvmlDeviceGetUtilizationRates)
    NVML_FUNC(nvmlDeviceGetEncoderUtilization)
    NVML_FUNC(nvmlDeviceGetDecoderUtilization)
    NVML_FUNC(nvmlDeviceGetMemoryInfo)
    NVML_FUNC(nvmlDeviceGetClockInfo)
    NVML_FUNC(nvmlDeviceGetPcieThroughput)
    NVML_FUNC(nvmlDeviceGetCurrPcieLinkGeneration)
    NVML_FUNC(nvmlDeviceGetCurrPcieLinkWidth)
    NVML_FUNC(nvmlDeviceGetMaxPcieLinkGeneration)
    NVML_FUNC(nvmlDeviceGetMaxPcieLinkWidth)
    NVML_FUNC(nvmlSystemGetDriverVersion)
    NVML_FUNC(nvmlSystemGetNVMLVersion)
};
#undef NVML_FUNC

class NVMLMonitor {
private:
    HMODULE m_hDll;
    NVMLFuncList m_func;
    nvmlDevice_t m_device;

    nvmlReturn_t LoadDll();
    void Close();
public:
    NVMLMonitor() : m_hDll(NULL), m_func({ 0 }) {};
    ~NVMLMonitor() {
        Close();
    }
    nvmlReturn_t Init(const std::string& pciBusId);
    nvmlReturn_t getData(NVMLMonitorInfo *info);
    nvmlReturn_t getDriverVersionx1000(int& ver);
    nvmlReturn_t getMaxPCIeLink(int& gen, int& width);
};

#endif //#if ENABLE_NVML

class NVSMIInfo {
private:
    std::string m_NVSMIOut;
public:
    NVSMIInfo() {};
    ~NVSMIInfo() {};
    int getData(NVMLMonitorInfo *info, const std::string& gpu_pcibusid);
};

struct CPerfMonitorPrm {
#if ENABLE_NVML
    const char *pciBusId;
#endif
    char reserved[256];
};

class CPerfMonitor {
public:
    CPerfMonitor();
    int init(tstring filename, const TCHAR *pPythonPath,
        int interval, int nSelectOutputLog, int nSelectOutputMatplot,
        std::unique_ptr<void, handle_deleter> thMainThread,
        std::shared_ptr<RGYLog> pRGYLog, CPerfMonitorPrm *prm);
    ~CPerfMonitor();

    void SetEncStatus(std::shared_ptr<EncodeStatus> encStatus);
    void SetThreadHandles(HANDLE thEncThread, HANDLE thInThread, HANDLE thOutThread, HANDLE thAudProcThread, HANDLE thAudEncThread);
    PerfQueueInfo *GetQueueInfoPtr() {
        return &m_QueueInfo;
    }
#if ENABLE_METRIC_FRAMEWORK
    bool GetQSVInfo(QSVGPUInfo *info) {
        return m_Consumer.getMFXLoad(info);
    }
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    bool GetNVMLInfo(NVMLMonitorInfo *info) {
        memcpy(info, &m_nvmlInfo, sizeof(m_nvmlInfo));
        return m_nvmlInfo.bDataValid;
    }
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_GPUZ_INFO
    bool GetGPUZInfo(GPUZ_SH_MEM *info) {
        memcpy(info, &m_GPUZInfo, sizeof(m_GPUZInfo));
        return m_bGPUZInfoValid;
    }
#endif //#if ENABLE_GPUZ_INFO

    void clear();
protected:
    int createPerfMpnitorPyw(const TCHAR *pywPath);
    void check();
    void run();
    void write_header(FILE *fp, int nSelect);
    void write(FILE *fp, int nSelect);

    static void loader(void *prm);

    tstring SelectedCounters(int select);

    int m_nStep;
    tstring m_sPywPath;
    PerfInfo m_info[2];
    std::thread m_thCheck;
    std::unique_ptr<void, handle_deleter> m_thMainThread;
    std::unique_ptr<RGYPipeProcess> m_pProcess;
    ProcessPipe m_pipes;
    HANDLE m_thEncThread;
    HANDLE m_thInThread;
    HANDLE m_thOutThread;
    HANDLE m_thAudProcThread;
    HANDLE m_thAudEncThread;
    int m_nLogicalCPU;
    std::shared_ptr<EncodeStatus> m_pEncStatus;
    int64_t m_nEncStartTime;
    int64_t m_nOutputFPSRate;
    int64_t m_nOutputFPSScale;
    int64_t m_nCreateTime100ns;
    bool m_bAbort;
    bool m_bEncStarted;
    int m_nInterval;
    tstring m_sMonitorFilename;
    std::unique_ptr<FILE, fp_deleter> m_fpLog;
    int m_nSelectCheck;
    int m_nSelectOutputLog;
    int m_nSelectOutputPlot;
    PerfQueueInfo m_QueueInfo;
    std::shared_ptr<RGYLog> m_pRGYLog;

#if ENABLE_METRIC_FRAMEWORK
    IExtensionLoader *m_pLoader;
    std::unique_ptr<IClientManager> m_pManager;
    CQSVConsumer m_Consumer;
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_NVML
    NVMLMonitor m_nvmlMonitor;
    NVMLMonitorInfo m_nvmlInfo;
#endif //#if ENABLE_NVML
#if ENABLE_GPUZ_INFO
    GPUZ_SH_MEM m_GPUZInfo;
#endif //#if ENABLE_GPUZ_INFO
    bool m_bGPUZInfoValid;
};


#endif //#ifndef __RGY_PERF_MONITOR_H__
