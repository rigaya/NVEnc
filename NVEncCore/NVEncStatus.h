// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#pragma once

#define WIN32_MEAN_AND_LEAN
#define NOMINMAX
#include <Windows.h>
#include <stdio.h>
#include <stdint.h>
#include <tchar.h>
#include <string>
#include <chrono>
#include <memory>
#include <algorithm>
#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "nvEncodeAPI.h"
#include "NVEncLog.h"
#include "cpu_info.h"

using std::chrono::duration_cast;

static const int UPDATE_INTERVAL = 800;
#define NV_ENC_ERR_ABORT ((NVENCSTATUS)-1)

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

enum {
    NVENC_THREAD_RUNNING = 0,

    NVENC_THREAD_FINISHED = -1,
    NVENC_THREAD_ABORT = -2,

    NVENC_THREAD_ERROR = 1,
};

typedef struct EncodeStatusData {
    uint64_t outFileSize;      //出力ファイルサイズ
    std::chrono::system_clock::time_point tmStart;          //エンコード開始時刻
    std::chrono::system_clock::time_point tmLastUpdate;     //最終更新時刻
    uint32_t frameTotal;       //入力予定の全フレーム数
    uint32_t frameOut;         //出力したフレーム数
    uint32_t frameOutIDR;      //出力したIDRフレーム
    uint32_t frameOutI;        //出力したIフレーム
    uint32_t frameOutP;        //出力したPフレーム
    uint32_t frameOutB;        //出力したBフレーム
    uint64_t frameOutISize;    //出力したIフレームのサイズ
    uint64_t frameOutPSize;    //出力したPフレームのサイズ
    uint64_t frameOutBSize;    //出力したBフレームのサイズ
    uint32_t frameOutIQPSum;   //出力したIフレームの平均QP
    uint32_t frameOutPQPSum;   //出力したPフレームの平均QP
    uint32_t frameOutBQPSum;   //出力したBフレームの平均QP
    uint32_t frameIn;          //エンコーダに入力したフレーム数 (drop含まず)
    uint32_t frameDrop;        //ドロップしたフレーム数
    double encodeFps;          //エンコード速度
    double bitrateKbps;        //ビットレート
} EncodeStatusData;

class EncodeStatus {
public:
    EncodeStatus() {
        ZeroMemory(&m_sData, sizeof(m_sData));

        m_sData.tmLastUpdate = std::chrono::system_clock::now();
        m_pause = FALSE;
        DWORD mode = 0;
        m_bStdErrWriteToConsole = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか
    }
    ~EncodeStatus() { m_pNVLog.reset(); };

    virtual void init(shared_ptr<CNVEncLog> pQSVLog) {
        m_pause = FALSE;
        m_sData.tmLastUpdate = std::chrono::system_clock::now();
        m_pNVLog = pQSVLog;
    }

    virtual void SetStart() {
        m_sData.tmStart = std::chrono::system_clock::now();
        GetProcessTime(GetCurrentProcess(), &m_sStartTime);
    }
    virtual void SetOutputData(NV_ENC_PIC_TYPE picType, uint32_t outputBytes, uint32_t frameAvgQP) {
        m_sData.outFileSize    += outputBytes;
        m_sData.frameOut       += 1;
        m_sData.frameOutIDR    += (NV_ENC_PIC_TYPE_IDR == picType);
        m_sData.frameOutI      += (NV_ENC_PIC_TYPE_IDR == picType);
        m_sData.frameOutI      += (NV_ENC_PIC_TYPE_I   == picType);
        m_sData.frameOutP      += (NV_ENC_PIC_TYPE_P   == picType);
        m_sData.frameOutB      += (NV_ENC_PIC_TYPE_B   == picType);
        m_sData.frameOutISize  += (0-(NV_ENC_PIC_TYPE_IDR == picType)) & outputBytes;
        m_sData.frameOutISize  += (0-(NV_ENC_PIC_TYPE_I   == picType)) & outputBytes;
        m_sData.frameOutPSize  += (0-(NV_ENC_PIC_TYPE_P   == picType)) & outputBytes;
        m_sData.frameOutBSize  += (0-(NV_ENC_PIC_TYPE_B   == picType)) & outputBytes;
        m_sData.frameOutIQPSum += (0-(NV_ENC_PIC_TYPE_IDR == picType)) & frameAvgQP;
        m_sData.frameOutIQPSum += (0-(NV_ENC_PIC_TYPE_I   == picType)) & frameAvgQP;
        m_sData.frameOutPQPSum += (0-(NV_ENC_PIC_TYPE_P   == picType)) & frameAvgQP;
        m_sData.frameOutBQPSum += (0-(NV_ENC_PIC_TYPE_B   == picType)) & frameAvgQP;
    }
#pragma warning(push)
#pragma warning(disable: 4100)
    virtual void UpdateDisplay(const TCHAR *mes, double progressPercent = 0.0) {
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

    virtual void WriteFrameTypeResult(const TCHAR *header, uint32_t count, uint32_t maxCount, uint64_t frameSize, uint64_t maxFrameSize, double avgQP) {
        if (count) {
            TCHAR mes[512] = { 0 };
            int mes_len = 0;
            const int header_len = (int)_tcslen(header);
            memcpy(mes, header, header_len * sizeof(mes[0]));
            mes_len += header_len;

            for (int i = std::max(0, (int)log10((double)count)); i < (int)log10((double)maxCount) && mes_len < _countof(mes); i++, mes_len++) {
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

                for (int i = std::max(0, (int)log10((double)frameSize / (double)(1024 * 1024))); i < (int)log10((double)maxFrameSize / (double)(1024 * 1024)) && mes_len < _countof(mes); i++, mes_len++) {
                    mes[mes_len] = _T(' ');
                }

                mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%.2f MB"), (double)frameSize / (double)(1024 * 1024));
            }

            WriteLine(mes);
        }
    }
    virtual void WriteLine(const TCHAR *mes) {
        if (m_pNVLog != nullptr && m_pNVLog->getLogLevel() > NV_LOG_INFO) {
            return;
        }
        m_pNVLog->write(NV_LOG_INFO, _T("%s\n"), mes);
    }
    virtual int UpdateDisplay(double progressPercent = 0.0) {
        if (m_pNVLog != nullptr && m_pNVLog->getLogLevel() > NV_LOG_INFO) {
            return NVENC_THREAD_RUNNING;
        }
        if (m_sData.frameOut + m_sData.frameDrop <= 0) {
            return NVENC_THREAD_RUNNING;
        }
        auto tm = std::chrono::system_clock::now();
        if (duration_cast<std::chrono::milliseconds>(tm - m_sData.tmLastUpdate).count() < UPDATE_INTERVAL) {
            return NVENC_THREAD_RUNNING;
        }
        m_sData.tmLastUpdate = tm;
        double elapsedTime = (double)duration_cast<std::chrono::milliseconds>(tm - m_sData.tmStart).count();
        if (m_sData.frameOut + m_sData.frameDrop) {
            TCHAR mes[256] = { 0 };
            m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / elapsedTime;
            m_sData.bitrateKbps = (double)m_sData.outFileSize * (m_nOutputFPSRate / (double)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
            if (0 < m_sData.frameTotal || progressPercent > 0.0) {
                if (progressPercent == 0.0) {
                    progressPercent = (m_sData.frameOut + m_sData.frameDrop) * 100 / (double)m_sData.frameTotal;
                }
                progressPercent = (std::min)(progressPercent, 100.0);
                uint32_t remaining_time = (uint32_t)(elapsedTime * (100.0 - progressPercent) / progressPercent + 0.5);
                int hh = remaining_time / (60*60*1000);
                remaining_time -= hh * (60*60*1000);
                int mm = remaining_time / (60*1000);
                remaining_time -= mm * (60*1000);
                int ss = (remaining_time + 500) / 1000;

                int len = _stprintf_s(mes, _countof(mes), _T("[%.1lf%%] %d frames: %.2lf fps, %0.2lf kb/s, remain %d:%02d:%02d  "),
                    progressPercent,
                    (m_sData.frameOut + m_sData.frameDrop),
                    m_sData.encodeFps,
                    m_sData.bitrateKbps,
                    hh, mm, ss );
                if (m_sData.frameDrop)
                    _stprintf_s(mes + len - 2, _countof(mes) - len + 2, _T(", afs drop %d/%d  "), m_sData.frameDrop, (m_sData.frameOut + m_sData.frameDrop));
            } else {
                _stprintf_s(mes, _countof(mes), _T("%d frames: %0.2lf fps, %0.2lf kbps  "), 
                    (m_sData.frameOut + m_sData.frameDrop),
                    m_sData.encodeFps,
                    m_sData.bitrateKbps
                    );
            }
            UpdateDisplay(mes);
        }
        return NVENC_THREAD_RUNNING;
    }
    virtual void writeResult() {
        auto tm_result = std::chrono::system_clock::now();
        const auto time_elapsed64 = duration_cast<std::chrono::milliseconds>(tm_result - m_sData.tmStart).count();

        TCHAR mes[512] = { 0 };
        for (int i = 0; i < 79; i++)
            mes[i] = ' ';
        WriteLine(mes);

        m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / (double)time_elapsed64;
        m_sData.bitrateKbps = (double)m_sData.outFileSize * (m_nOutputFPSRate / (double)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
        m_sData.tmLastUpdate = tm_result;

        _stprintf_s(mes, _countof(mes), _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
            m_sData.frameOut,
            m_sData.encodeFps,
            m_sData.bitrateKbps,
            (double)m_sData.outFileSize / (double)(1024 * 1024)
            );
        WriteLine(mes);

        int hh = (int)(time_elapsed64 / (60*60*1000));
        int time_elapsed = (int)(time_elapsed64 - (hh * (60*60*1000)));
        int mm = time_elapsed / (60*1000);
        time_elapsed -= mm * (60*1000);
        int ss = time_elapsed / 1000;
        _stprintf_s(mes, _countof(mes), _T("encode time %d:%02d:%02d / CPU Usage: %.2f%%\n"), hh, mm, ss, GetProcessAvgCPUUsage(GetCurrentProcess(), &m_sStartTime));
        WriteLine(mes);
        
        uint32_t maxCount = std::max(m_sData.frameOutI, std::max(m_sData.frameOutP, m_sData.frameOutB));
        uint64_t maxFrameSize = std::max(m_sData.frameOutISize, std::max(m_sData.frameOutPSize, m_sData.frameOutBSize));

        WriteFrameTypeResult(_T("frame type IDR "), m_sData.frameOutIDR, maxCount,                     0, maxFrameSize, -1.0);
        WriteFrameTypeResult(_T("frame type I   "), m_sData.frameOutI,   maxCount, m_sData.frameOutISize, maxFrameSize, (m_sData.frameOutI) ? m_sData.frameOutIQPSum / (double)m_sData.frameOutI : -1);
        WriteFrameTypeResult(_T("frame type P   "), m_sData.frameOutP,   maxCount, m_sData.frameOutPSize, maxFrameSize, (m_sData.frameOutP) ? m_sData.frameOutPQPSum / (double)m_sData.frameOutP : -1);
        WriteFrameTypeResult(_T("frame type B   "), m_sData.frameOutB,   maxCount, m_sData.frameOutBSize, maxFrameSize, (m_sData.frameOutB) ? m_sData.frameOutBQPSum / (double)m_sData.frameOutB : -1);
    }
#pragma warning(push)
#pragma warning(disable: 4100)
    virtual void SetPrivData(void *pPrivateData) {

    }
#pragma warning(pop)
public:
    BOOL m_pause;
    std::shared_ptr<CNVEncLog> m_pNVLog;
    PROCESS_TIME m_sStartTime;
    EncodeStatusData m_sData;
    uint32_t m_nOutputFPSRate = 0;
    uint32_t m_nOutputFPSScale = 0;
    bool m_bStdErrWriteToConsole;
};

class CProcSpeedControl {
public:
    CProcSpeedControl(uint32_t maxProcessPerSec, uint32_t checkInterval = 4) :
        m_nCount(0),
        m_nCheckInterval(checkInterval),
        m_bEnable(true),
        m_tmThreshold(std::chrono::microseconds(1)),
        m_tmLastCheck(std::chrono::high_resolution_clock::now()) {
        setSpeed(maxProcessPerSec);
    };
    virtual ~CProcSpeedControl() {
    };
    void setSpeed(uint32_t maxProcessPerSec) {
        m_bEnable = maxProcessPerSec != 0;
        m_tmThreshold = (maxProcessPerSec != 0) ? std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)) / maxProcessPerSec : std::chrono::microseconds(1);
    }
    void reset() {
        m_nCount = 0;
        m_tmLastCheck = std::chrono::high_resolution_clock::now();
    }
    bool wait() {
        bool ret = false;
        m_nCount++;
        if (m_bEnable && m_nCount % m_nCheckInterval == 0) {
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
        return ret;
    };
private:
    uint32_t m_nCount;
    uint32_t m_nCheckInterval;
    bool m_bEnable;
    std::chrono::microseconds m_tmThreshold;
    std::chrono::high_resolution_clock::time_point m_tmLastCheck;
};
