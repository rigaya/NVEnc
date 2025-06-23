// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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
// ------------------------------------------------------------------------------------------

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "rgy_log.h"
#include "cpu_info.h"
#include "rgy_err.h"
#include "rgy_perf_monitor.h"
#include "rgy_parallel_enc.h"
#include "gpuz_info.h"
#include "rgy_status.h"

EncodeStatus::EncodeStatus() :
    m_sData(),
    m_pause(false),
    m_pRGYLog(),
    m_pPerfMonitor(),
    m_sStartTime(std::make_unique<PROCESS_TIME>()),
    m_tmStart(),
    m_tmLastUpdate(std::chrono::system_clock::now()),
    m_peStatusShare(nullptr),
    m_childStatus(),
    m_bStdErrWriteToConsole(false),
    m_bEncStarted(false) {
}
EncodeStatus::~EncodeStatus() {
    if (m_pRGYLog) m_pRGYLog->write_log(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("Closing EncodeStatus...\n"));
    m_pPerfMonitor.reset();
    m_pRGYLog.reset();
    m_sStartTime.reset();
}

void EncodeStatus::Init(uint32_t outputFPSRate, uint32_t outputFPSScale,
    uint32_t totalInputFrames, double totalDuration, const sTrimParam& trim,
    shared_ptr<RGYLog> pRGYLog, shared_ptr<CPerfMonitor> pPerfMonitor, RGYParallelEncodeStatusData *peStatusShare) {
    m_pause = false;
    m_pRGYLog = pRGYLog;
    m_pPerfMonitor = pPerfMonitor;
    m_peStatusShare = peStatusShare;
    m_sData.outputFPSRate = outputFPSRate;
    m_sData.outputFPSScale = outputFPSScale;
    m_sData.frameTotal = totalInputFrames;
    m_sData.totalDuration = totalDuration;
    if (trim.list.size() > 0 && trim.list.back().fin != TRIM_MAX) {
        //途中終了することになる
        const auto estFrames = std::max((uint32_t)(m_sData.totalDuration * outputFPSRate / outputFPSScale + 0.5), m_sData.frameTotal);
        m_sData.frameTotal = std::min<uint32_t>(estFrames, trim.list.back().fin);
        m_sData.totalDuration = std::min(m_sData.totalDuration, trim.list.back().fin * outputFPSScale / (double)outputFPSRate);
    }
#if defined(_WIN32) || defined(_WIN64)
    DWORD mode = 0;
    m_bStdErrWriteToConsole = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか
#endif //#if defined(_WIN32) || defined(_WIN64)
}

void EncodeStatus::SetStart() {
    m_tmStart = std::chrono::system_clock::now();
    m_bEncStarted = true;
    GetProcessTime(m_sStartTime.get());
}
void EncodeStatus::SetOutputData(RGY_FRAMETYPE picType, uint64_t outputBytes, uint32_t frameAvgQP) {
    m_sData.outFileSize    += outputBytes;
    m_sData.frameOut       += 1;
    m_sData.frameOutIDR    += (picType & RGY_FRAMETYPE_IDR) >> 7;
    m_sData.frameOutI      += (picType & RGY_FRAMETYPE_IDR) >> 7;
    m_sData.frameOutI      += (picType & RGY_FRAMETYPE_I);
    m_sData.frameOutP      += (picType & RGY_FRAMETYPE_P) >> 1;
    m_sData.frameOutB      += (picType & RGY_FRAMETYPE_B) >> 2;
    m_sData.frameOutISize  += (0-((picType & RGY_FRAMETYPE_IDR) >> 7)) & outputBytes;
    m_sData.frameOutISize  += (0- (picType & RGY_FRAMETYPE_I))         & outputBytes;
    m_sData.frameOutPSize  += (0-((picType & RGY_FRAMETYPE_P)   >> 1)) & outputBytes;
    m_sData.frameOutBSize  += (0-((picType & RGY_FRAMETYPE_B)   >> 2)) & outputBytes;
    m_sData.frameOutIQPSum += (0-((picType & RGY_FRAMETYPE_IDR) >> 7)) & frameAvgQP;
    m_sData.frameOutIQPSum += (0- (picType & RGY_FRAMETYPE_I))         & frameAvgQP;
    m_sData.frameOutPQPSum += (0-((picType & RGY_FRAMETYPE_P)   >> 1)) & frameAvgQP;
    m_sData.frameOutBQPSum += (0-((picType & RGY_FRAMETYPE_B)   >> 2)) & frameAvgQP;
}
#pragma warning(push)
#pragma warning(disable: 4100)
void EncodeStatus::UpdateDisplay(const TCHAR *mes, double progressPercent) {
    if (m_pRGYLog != nullptr && m_pRGYLog->getLogLevel(RGY_LOGT_CORE_PROGRESS) > RGY_LOG_INFO) {
        return;
    }
#if UNICODE
    char *mes_char = NULL;
    if (!m_bStdErrWriteToConsole) {
        //コンソールへの出力でなければ、ANSIに変換する
        const int buf_length = (int)(wcslen(mes) + 1) * 2;
        if (NULL != (mes_char = (char *)calloc(buf_length, 1))) {
            WideCharToMultiByte(CP_THREAD_ACP, 0, mes, -1, mes_char, buf_length, NULL, NULL);
            fprintf(stderr, "%s\r", mes_char);
            free(mes_char);
        }
    } else
#endif
        _ftprintf(stderr, _T("%s\r"), mes);

    fflush(stderr); //リダイレクトした場合でもすぐ読み取れるようflush
}
#pragma warning(pop)

RGY_ERR EncodeStatus::UpdateDisplayByCurrentDuration(double currentDuration) {
    double progressPercent = 0.0;
    if (m_sData.totalDuration > 0.0) {
        progressPercent = 100.0 * currentDuration / m_sData.totalDuration;
    }
    return UpdateDisplay(progressPercent);
}
RGY_ERR EncodeStatus::UpdateDisplay(double progressPercent) {
    if (m_peStatusShare == nullptr && m_pRGYLog != nullptr && m_pRGYLog->getLogLevel(RGY_LOGT_CORE_PROGRESS) > RGY_LOG_INFO) {
        return RGY_ERR_NONE;
    }
    if (m_sData.frameOut + m_sData.frameDrop <= 0) {
        return RGY_ERR_NONE;
    }
    auto tm = std::chrono::system_clock::now();
    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastUpdate).count() < UPDATE_INTERVAL) {
        return RGY_ERR_NONE;
    }
    m_tmLastUpdate = tm;
    m_sData.progressPercent = progressPercent;

    bool qsv_metric = false;
    bool bVideoEngineUsage = false;
    bool bGPUUsage = false;
    double gpudecoder_usage = 0.0;
    double gpuencoder_usage = 0.0;
    double gpuusage = 0.0;
#if ENABLE_METRIC_FRAMEWORK
    QSVGPUInfo info = { 0 };
    bGPUUsage = bVideoEngineUsage = m_pPerfMonitor && m_pPerfMonitor->GetQSVInfo(&info);
    if (bVideoEngineUsage) {
        qsv_metric = true;
        gpuusage = info.dEULoad;
        gpuencoder_usage = info.dMFXLoad;
    }
#endif //#if ENABLE_METRIC_FRAMEWORK
#if ENABLE_PERF_COUNTER
    if (m_pPerfMonitor) {
        const auto counters = m_pPerfMonitor->GetPerfCounters();
        bGPUUsage = bVideoEngineUsage = counters.size() > 0;
        if (bVideoEngineUsage) {
            if (!qsv_metric) { //QSVではMETRIC_FRAMEWORKを優先
                gpuencoder_usage = std::max(
                    RGYGPUCounterWinEntries(counters).filter_type(L"encode").max(),
                    RGYGPUCounterWinEntries(counters).filter_type(L"codec").max()); //vce rx5xxx
                bVideoEngineUsage = !ENCODER_QSV || gpuencoder_usage > 0.0; //QSVのMFX使用率はこれでは取れない
            }
            gpuusage = std::max(std::max(std::max(
                RGYGPUCounterWinEntries(counters).filter_type(L"cuda").max(), //nvenc
                RGYGPUCounterWinEntries(counters).filter_type(L"compute").max()), //vce-opencl
                RGYGPUCounterWinEntries(counters).filter_type(L"3d").max()), //qsv
                RGYGPUCounterWinEntries(counters).filter_type(L"videoprocessing").max()); //qsv
            gpudecoder_usage = RGYGPUCounterWinEntries(counters).filter_type(L"decode").max();
        }
    }
#endif //#if ENABLE_PERF_COUNTER
#if ENABLE_NVML
    NVMLMonitorInfo info;
    const bool bNVML = m_pPerfMonitor && m_pPerfMonitor->GetNVMLInfo(&info);
    if (bNVML) {
        if (!bGPUUsage) {
            gpuusage = info.GPULoad;
            gpuencoder_usage = info.VEELoad;
            gpudecoder_usage = info.VEDLoad;
        }
        m_sData.GPUClockTotal += info.GPUFreq;
        m_sData.VEClockTotal += info.VEFreq;
        bGPUUsage = bVideoEngineUsage = bNVML;
    }
#endif //#if ENABLE_NVML
#if ENABLE_GPUZ_INFO
    if (!bGPUUsage) {
        GPUZ_SH_MEM gpu_info = { 0 };
        if ((m_pPerfMonitor && m_pPerfMonitor->GetGPUZInfo(&gpu_info))
            || 0 == get_gpuz_info(&gpu_info)) {
            bGPUUsage = true;
            gpuusage = gpu_load(&gpu_info);
            gpuencoder_usage = video_engine_load(&gpu_info, &bVideoEngineUsage);
            m_sData.GPUClockTotal += gpu_core_clock(&gpu_info);
        } else {
            m_sData.GPUInfoCountFail++;
        }
    }
#endif //#if ENABLE_GPUZ_INFO
    if (bGPUUsage) {
        m_sData.GPUInfoCountSuccess++;
        m_sData.GPULoadPercentTotal += gpuusage;
        m_sData.VEELoadPercentTotal += gpuencoder_usage;
        m_sData.VEDLoadPercentTotal += gpudecoder_usage;
    }

    double elapsedTime = (double)duration_cast<std::chrono::milliseconds>(tm - m_tmStart).count();
    if (m_sData.frameOut + m_sData.frameDrop >= 30) {
        TCHAR mes[256] = { 0 };

        int consoleWidth = 0;
#if defined(_WIN32) || defined(_WIN64)
        HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
        DWORD mode = 0;
        bool stderr_write_to_console = 0 != GetConsoleMode(hStdErr, &mode); //stderrの出力先がコンソールかどうか
        if (stderr_write_to_console) {
            CONSOLE_SCREEN_BUFFER_INFO consoleinfo;
            GetConsoleScreenBufferInfo(hStdErr, &consoleinfo);
            consoleWidth = consoleinfo.dwSize.X;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / elapsedTime;
        m_sData.bitrateKbps = (double)m_sData.outFileSize * (m_sData.outputFPSRate / (double)m_sData.outputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
        std::vector<EncodeStatusData> childStsList;
        for (size_t i = 1; i < m_childStatus.size(); i++) { // 最初のエンコーダは自分自身と同じなので飛ばして1から
            if (m_childStatus[i].second) {
                EncodeStatusData data;
                if (m_childStatus[i].second->get(data)) { // 進捗表示を取得できたら
                    data.progressPercent *= m_childStatus[i].first; // 個々の進捗率を全体の進捗率に換算する
                    childStsList.push_back(data);
                }
            }
        }
        enum {
            MES_PROGRESS_PERCENT,
            MES_CURRENT_FRAME,
            MES_FRAME_TOTAL,
            MES_FRAMES,
            MES_FPS,
            MES_KBPS,
            MES_REMAIN,
            MES_DROP,
            MES_GPU,
            MES_GPU_DEC,
            MES_EST_FILE_SIZE,
            MES_ID_MAX
        };
        struct mes_data {
            int len;
            TCHAR str[64];
        };
        std::array<mes_data, MES_ID_MAX> chunks;
        for (auto &c : chunks) {
            c.len = 0;
        }
        double totalProgressPercent = 0.0;
        if (m_sData.frameTotal > 0 || progressPercent > 0.0) { //progress percent
            totalProgressPercent = progressPercent; // std::accumulate(childStsList.begin(), childStsList.end(), progressPercent, [](double sum, const EncodeStatusData& child) { return sum + child.progressPercent; });
            if (progressPercent == 0.0) {
                progressPercent = m_sData.frameIn * 100 / (double)m_sData.frameTotal;
                m_sData.progressPercent = progressPercent;
            }
            if (totalProgressPercent == 0.0) {
                const auto totalFrameIn = std::accumulate(childStsList.begin(), childStsList.end(), m_sData.frameIn, [](uint32_t sum, const EncodeStatusData& child) { return sum + child.frameIn; });
                totalProgressPercent = totalFrameIn * 100 / (double)m_sData.frameTotal;
            }
            totalProgressPercent = (std::min)(totalProgressPercent, 100.0);
            uint32_t remaining_time = (uint32_t)(elapsedTime * (100.0 - totalProgressPercent) / totalProgressPercent + 0.5);
            const int hh = remaining_time / (60*60*1000);
            remaining_time -= hh * (60*60*1000);
            const int mm = remaining_time / (60*1000);
            remaining_time -= mm * (60*1000);
            const int ss = remaining_time / 1000;

            chunks[MES_PROGRESS_PERCENT].len = _stprintf_s(chunks[MES_PROGRESS_PERCENT].str, _T("[%.1lf%%] "), totalProgressPercent);
            chunks[MES_REMAIN].len           = _stprintf_s(chunks[MES_REMAIN].str, _T(", remain %d:%02d:%02d"), hh, mm, ss);

            const auto totalOutFileSize = m_sData.outFileSize; // std::accumulate(childStsList.begin(), childStsList.end(), m_sData.outFileSize, [](uint64_t sum, const EncodeStatusData& child) { return sum + child.outFileSize; });
            const double est_file_size = (double)totalOutFileSize / (totalProgressPercent * 0.01);
            chunks[MES_EST_FILE_SIZE].len = _stprintf_s(chunks[MES_EST_FILE_SIZE].str, _T(", est out size %.1fMB"), est_file_size * (1.0 / (1024.0 * 1024.0)));
        }
        const auto totalFrameOut = m_sData.frameOut + m_sData.frameDrop; // std::accumulate(childStsList.begin(), childStsList.end(), m_sData.frameOut + m_sData.frameDrop, [](uint32_t sum, const EncodeStatusData& child) { return sum + child.frameOut + child.frameDrop; });
        chunks[MES_CURRENT_FRAME].len = _stprintf_s(chunks[MES_CURRENT_FRAME].str, _T("%d"), totalFrameOut);
        if (m_sData.frameTotal > 0) {
            chunks[MES_FRAME_TOTAL].len = _stprintf_s(chunks[MES_FRAME_TOTAL].str, _T("/%d"), m_sData.frameTotal);
        }
        chunks[MES_FRAMES].len = _stprintf_s(chunks[MES_FRAMES].str, _T(" frames: "));
        const double totalEncodeFps = m_sData.encodeFps; // std::accumulate(childStsList.begin(), childStsList.end(), m_sData.encodeFps, [](double sum, const EncodeStatusData& child) { return sum + child.encodeFps; });
        chunks[MES_FPS].len = _stprintf_s(chunks[MES_FPS].str, _T("%.2lf"), totalEncodeFps);
        chunks[MES_FPS].len += _stprintf_s(chunks[MES_FPS].str + chunks[MES_FPS].len, _countof(chunks[MES_FPS].str) - chunks[MES_FPS].len, _T(" fps, "));
        auto totalBitrateKbps = m_sData.bitrateKbps;
        if (false && childStsList.size() > 0 && totalProgressPercent > 0.0) {
            totalBitrateKbps = std::accumulate(childStsList.begin(), childStsList.end(), m_sData.bitrateKbps * m_sData.progressPercent,
                [](double sum, const EncodeStatusData& child) { return sum + child.bitrateKbps * child.progressPercent; }) / totalProgressPercent;
        }
        chunks[MES_KBPS].len = _stprintf_s(chunks[MES_KBPS].str, _T("%d kbps"), (int)(totalBitrateKbps + 0.5));
        if (m_sData.frameDrop) {
            auto totalFrameDrop = m_sData.frameDrop; // std::accumulate(childStsList.begin(), childStsList.end(), m_sData.frameDrop, [](uint32_t sum, const EncodeStatusData& child) { return sum + child.frameDrop; });
            chunks[MES_DROP].len = _stprintf_s(chunks[MES_DROP].str, _T(", afs drop %d/%d"), totalFrameDrop, totalFrameOut);
        }
        if (bGPUUsage) {
            chunks[MES_GPU].len = _stprintf_s(chunks[MES_GPU].str, _T(", GPU %d%%"), std::min((int)(gpuusage + 0.5), 100));
            if (bVideoEngineUsage) {
                chunks[MES_GPU].len += _stprintf_s(chunks[MES_GPU].str + chunks[MES_GPU].len, _countof(chunks[MES_GPU].str) - chunks[MES_GPU].len, _T(", %s %d%%"), (ENCODER_QSV) ? _T("MFX") : _T("VE"), std::min((int)(gpuencoder_usage + 0.5), 100));
            }
            if (gpudecoder_usage > 0) {
                chunks[MES_GPU_DEC].len += _stprintf_s(chunks[MES_GPU_DEC].str, _countof(chunks[MES_GPU_DEC].str), _T(", VD %d%%"), std::min((int)(gpudecoder_usage + 0.5), 100));
            }
        }

        int mesLength = 0;
        auto check_add_length = [&mesLength, &chunks, consoleWidth](int mes_id) {
            if (consoleWidth <= 0 || mesLength + chunks[mes_id].len < consoleWidth) {
                mesLength += chunks[mes_id].len;
            } else {
                chunks[mes_id].len = 0;
            }
        };
        check_add_length(MES_PROGRESS_PERCENT);
        check_add_length(MES_CURRENT_FRAME);
        check_add_length(MES_FRAME_TOTAL);
        check_add_length(MES_FRAMES);
        check_add_length(MES_FPS);
        check_add_length(MES_KBPS);
        check_add_length(MES_REMAIN);
        check_add_length(MES_DROP);
        check_add_length(MES_GPU);
        check_add_length(MES_GPU_DEC);
        check_add_length(MES_EST_FILE_SIZE);
        check_add_length(MES_FRAME_TOTAL);

        int len = 0;
        for (const auto &c : chunks) {
            if (c.len > 0) {
                _tcscpy_s(mes + len, _countof(mes) - len, c.str);
                len += c.len;
            }
        }
        const int fillWidth = (consoleWidth > 0) ? std::min(consoleWidth, (int)_countof(mes))-1 : 79;
        for (; len < fillWidth; len++) {
            mes[len] = _T(' ');
        }
        mes[len] = _T('\0');
        if (m_peStatusShare) {
            m_peStatusShare->set(m_sData);
        }
        UpdateDisplay(mes, progressPercent);
    }
    return RGY_ERR_NONE;
}
void EncodeStatus::WriteResults() {
    auto tm_result = std::chrono::system_clock::now();
    const auto time_elapsed64 = std::chrono::duration_cast<std::chrono::milliseconds>(tm_result - m_tmStart).count();
    m_sData.encodeFps = m_sData.frameOut * 1000.0 / (double)time_elapsed64;
    m_sData.bitrateKbps = (double)(m_sData.outFileSize * 8) *  (m_sData.outputFPSRate / (double)m_sData.outputFPSScale) / (1000.0 * m_sData.frameOut);

    int consoleWidth = 0;
#if defined(_WIN32) || defined(_WIN64)
    HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode = 0;
    bool stderr_write_to_console = 0 != GetConsoleMode(hStdErr, &mode); //stderrの出力先がコンソールかどうか
    if (stderr_write_to_console) {
        CONSOLE_SCREEN_BUFFER_INFO consoleinfo;
        GetConsoleScreenBufferInfo(hStdErr, &consoleinfo);
        consoleWidth = consoleinfo.dwSize.X;
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    TCHAR mes[512] = { 0 };
    for (int i = 0; i < std::max(consoleWidth-1, 79); i++)
        mes[i] = ' ';
    WriteResultLine(mes);

    m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / (double)time_elapsed64;
    m_sData.bitrateKbps = (m_sData.frameOut + m_sData.frameDrop == 0) ? 0 : (double)m_sData.outFileSize * (m_sData.outputFPSRate / (double)m_sData.outputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
    m_tmLastUpdate = tm_result;

    _stprintf_s(mes, _countof(mes), _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
        m_sData.frameOut,
        m_sData.encodeFps,
        m_sData.bitrateKbps,
        (double)m_sData.outFileSize / (double)(1024 * 1024)
    );
    WriteResultLine(mes);

    int hh = (int)(time_elapsed64 / (60*60*1000));
    int time_elapsed = (int)(time_elapsed64 - (hh * (60*60*1000)));
    int mm = time_elapsed / (60*1000);
    time_elapsed -= mm * (60*1000);
    int ss = time_elapsed / 1000;
    m_sData.CPUUsagePercent = GetProcessAvgCPUUsage(m_sStartTime.get());
    if (m_sData.GPUInfoCountSuccess > m_sData.GPUInfoCountFail) {
        const double gpu_load = m_sData.GPULoadPercentTotal / m_sData.GPUInfoCountSuccess;
        const double vee_load = m_sData.VEELoadPercentTotal / m_sData.GPUInfoCountSuccess;
        const double ved_load = m_sData.VEDLoadPercentTotal / m_sData.GPUInfoCountSuccess;
        const int gpu_clock_avg = (int)(m_sData.GPUClockTotal / m_sData.GPUInfoCountSuccess + 0.5);
        tstring tmes = strsprintf(_T("encode time %d:%02d:%02d, CPU: %.1f%%"), hh, mm, ss, m_sData.CPUUsagePercent);
        if (gpu_load > 0.0) {
            tmes += strsprintf(_T(", GPU: %.1f%%"), std::min(gpu_load, 100.0));
        }
        if (vee_load > 0.0) {
            tmes += strsprintf(_T(", %s: %.1f%%"), (ENCODER_QSV) ? _T("MFX") : _T("VE"), std::min(vee_load, 100.0));
        }
        if (ved_load > 0.0) {
            tmes += strsprintf(_T(", VD: %.1f%%"), std::min(ved_load, 100.0));
        }
        if (gpu_clock_avg > 0) {
            tmes += strsprintf(_T(", GPUClock: %dMHz"), gpu_clock_avg);
        }
#if ENABLE_NVML
        const int ve_clock_avg = (int)(m_sData.VEClockTotal / m_sData.GPUInfoCountSuccess + 0.5);
        if (ve_clock_avg > 0) {
            tmes += strsprintf(_T(", VEClock: %dMHz"), ve_clock_avg);
        }
#endif //#if ENABLE_NVML
        tmes += _T("\n");
        WriteResultLineDirect(tmes.c_str());
    } else {
        _stprintf_s(mes, _T("encode time %d:%02d:%02d, CPULoad: %.1f%%\n"), hh, mm, ss, m_sData.CPUUsagePercent);
        WriteResultLineDirect(mes);
    }

    uint32_t maxCount = (std::max)(m_sData.frameOutI, (std::max)(m_sData.frameOutP, m_sData.frameOutB));
    uint64_t maxFrameSize = (std::max)(m_sData.frameOutISize, (std::max)(m_sData.frameOutPSize, m_sData.frameOutBSize));

    WriteFrameTypeResult(_T("frame type IDR "), m_sData.frameOutIDR, maxCount, 0, maxFrameSize, -1.0);
    WriteFrameTypeResult(_T("frame type I   "), m_sData.frameOutI, maxCount, m_sData.frameOutISize, maxFrameSize, (m_sData.frameOutI && m_sData.frameOutIQPSum) ? m_sData.frameOutIQPSum / (double)m_sData.frameOutI : -1);
    WriteFrameTypeResult(_T("frame type P   "), m_sData.frameOutP, maxCount, m_sData.frameOutPSize, maxFrameSize, (m_sData.frameOutP && m_sData.frameOutPQPSum) ? m_sData.frameOutPQPSum / (double)m_sData.frameOutP : -1);
    WriteFrameTypeResult(_T("frame type B   "), m_sData.frameOutB, maxCount, m_sData.frameOutBSize, maxFrameSize, (m_sData.frameOutB && m_sData.frameOutBQPSum) ? m_sData.frameOutBQPSum / (double)m_sData.frameOutB : -1);
}
int64_t EncodeStatus::getStartTimeMicroSec() {
#if defined(_WIN32) || defined(_WIN64)
    return m_sStartTime->creation / 10;
#else
    return (int)(m_sStartTime->creation * (double)(1e6 / CLOCKS_PER_SEC) + 0.5);
#endif
}
bool EncodeStatus::getEncStarted() {
    return m_bEncStarted;
}
#pragma warning(push)
#pragma warning(disable: 4100)
void EncodeStatus::SetPrivData(void *pPrivateData) {

}
#pragma warning(pop)
EncodeStatusData EncodeStatus::GetEncodeData() {
    return m_sData;
}

void EncodeStatus::addChildStatus(const std::pair<double, RGYParallelEncodeStatusData*>& encStatus) {
    m_childStatus.push_back(encStatus);
}

void EncodeStatus::WriteResultLine(const TCHAR *mes) {
    if (m_pRGYLog != nullptr && m_pRGYLog->getLogLevel(RGY_LOGT_CORE_RESULT) > RGY_LOG_INFO) {
        return;
    }
    m_pRGYLog->write(RGY_LOG_INFO, RGY_LOGT_CORE_RESULT, _T("%s\n"), mes);
}
void EncodeStatus::WriteResultLineDirect(const TCHAR *mes) {
    if (m_pRGYLog != nullptr && m_pRGYLog->getLogLevel(RGY_LOGT_CORE_RESULT) > RGY_LOG_INFO) {
        return;
    }
    m_pRGYLog->write_log(RGY_LOG_INFO, RGY_LOGT_CORE_RESULT, mes);
}
void EncodeStatus::WriteFrameTypeResult(const TCHAR *header, uint32_t count, uint32_t maxCount, uint64_t frameSize, uint64_t maxFrameSize, double avgQP) {
    if (count) {
        TCHAR mes[512] = { 0 };
        int mes_len = 0;
        const int header_len = (int)_tcslen(header);
        memcpy(mes, header, header_len * sizeof(mes[0]));
        mes_len += header_len;

        for (int i = (std::max)(0, (int)std::log10((double)count)); i < (int)std::log10((double)maxCount) && mes_len < _countof(mes); i++, mes_len++) {
            mes[mes_len] = _T(' ');
        }
        mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%u"), count);

        if (avgQP >= 0.0) {
            mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T(",  avgQP  %4.2f"), avgQP);
        }

        if (frameSize > 0) {
            const TCHAR *TOTAL_SIZE = _T(",  total size  ");
            memcpy(mes + mes_len, TOTAL_SIZE, _tcslen(TOTAL_SIZE) * sizeof(mes[0]));
            mes_len += (int)_tcslen(TOTAL_SIZE);

            for (int i = (std::max)(0, (int)std::log10((double)frameSize / (double)(1024 * 1024))); i < (int)std::log10((double)maxFrameSize / (double)(1024 * 1024)) && mes_len < _countof(mes); i++, mes_len++) {
                mes[mes_len] = _T(' ');
            }

            mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%.2f MB"), (double)frameSize / (double)(1024 * 1024));
        }

        WriteResultLine(mes);
    }
}

CProcSpeedControl::CProcSpeedControl(uint32_t maxProcessPerSec, uint32_t checkInterval) :
    m_nCountLast(0),
    m_nCheckInterval(checkInterval),
    m_bEnable(true),
    m_tmThreshold(std::chrono::microseconds(1)),
    m_tmLastCheck(std::chrono::high_resolution_clock::now()) {
    setSpeed(maxProcessPerSec);
};
CProcSpeedControl::~CProcSpeedControl() {
};
void CProcSpeedControl::setSpeed(uint32_t maxProcessPerSec) {
    m_bEnable = maxProcessPerSec != 0;
    m_tmThreshold = (maxProcessPerSec != 0) ? std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)) / maxProcessPerSec : std::chrono::microseconds(1);
}
void CProcSpeedControl::reset() {
    m_nCountLast = 0;
    m_tmLastCheck = std::chrono::high_resolution_clock::now();
}
bool CProcSpeedControl::wait() {
    return wait(m_nCountLast + 1);
};
bool CProcSpeedControl::wait(uint32_t nCount) {
    bool ret = false;
    if (m_bEnable && m_nCountLast != nCount && nCount % m_nCheckInterval == 0) {
        auto tmNow = std::chrono::high_resolution_clock::now();
        //前回のチェックからこのくらい経っているとよい
        auto tmInterval = m_tmThreshold * m_nCheckInterval;
        //実際に経過した時間との差
        auto tmSleep = tmInterval - (tmNow - m_tmLastCheck);
        if (tmSleep > std::chrono::milliseconds(1)) {
            std::this_thread::sleep_for(tmSleep);
            ret = true;
            //実際にどのくらい経っていようとここでは、基準時間分進んだのだ
            m_tmLastCheck += tmInterval;
        } else {
            m_tmLastCheck = tmNow;
        }
    }
    m_nCountLast = nCount;
    return ret;
}
