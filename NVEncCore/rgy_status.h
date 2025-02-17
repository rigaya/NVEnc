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

#pragma once
#ifndef __RGY_STATUS_H__
#define __RGY_STATUS_H__

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "rgy_err.h"
#include "rgy_def.h"

using std::chrono::duration_cast;

class CPerfMonitor;
class RGYParallelEncodeStatusData;
class RGYLog;
struct PROCESS_TIME;

static const int UPDATE_INTERVAL = 800;

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

typedef struct EncodeStatusData {
    uint32_t outputFPSRate;
    uint32_t outputFPSScale;
    uint64_t outFileSize;      //出力ファイルサイズ
    double   totalDuration;    //入力予定の動画の総時間(s)
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
    double CPUUsagePercent;
    int    GPUInfoCountSuccess;
    int    GPUInfoCountFail;
    double GPULoadPercentTotal;
    double VEELoadPercentTotal;
    double VEDLoadPercentTotal;
    double VEClockTotal;
    double GPUClockTotal;
    double progressPercent;
} EncodeStatusData;

class EncodeStatus {
public:
    EncodeStatus();
    virtual ~EncodeStatus();
    virtual void Init(uint32_t outputFPSRate, uint32_t outputFPSScale,
        uint32_t totalInputFrames, double totalDuration, const sTrimParam &trim,
        std::shared_ptr<RGYLog> pRGYLog, std::shared_ptr<CPerfMonitor> pPerfMonitor,
        RGYParallelEncodeStatusData *peStatusShare // 子エンコーダ側から親への進捗表示共有するためのクラスへのポインタ (実体はRGYParallelEncProcess::m_sendData::encStatus)
    );

    void SetStart();
    void SetOutputData(RGY_FRAMETYPE picType, uint64_t outputBytes, uint32_t frameAvgQP);
    virtual void UpdateDisplay(const TCHAR *mes, double progressPercent = 0.0);

    virtual RGY_ERR UpdateDisplayByCurrentDuration(double currentDuration);
    virtual RGY_ERR UpdateDisplay(double progressPercent = 0.0);
    void WriteResults();
    int64_t getStartTimeMicroSec();
    bool getEncStarted();
    virtual void SetPrivData(void *pPrivateData);
    void addChildStatus(const std::pair<double, RGYParallelEncodeStatusData*>& encStatus);  // 親側で子エンコーダの担当割合と進捗表示共有クラスへのポインタ (実体はRGYParallelEncProcess::m_sendData::encStatus)を追加
    EncodeStatusData GetEncodeData();
    EncodeStatusData m_sData;
protected:
    virtual void WriteResultLine(const TCHAR *mes);
    virtual void WriteResultLineDirect(const TCHAR *mes);
    void WriteFrameTypeResult(const TCHAR *header, uint32_t count, uint32_t maxCount, uint64_t frameSize, uint64_t maxFrameSize, double avgQP);

    bool m_pause;
    std::shared_ptr<RGYLog> m_pRGYLog;
    std::shared_ptr<CPerfMonitor> m_pPerfMonitor;
    std::unique_ptr<PROCESS_TIME> m_sStartTime;
    std::chrono::system_clock::time_point m_tmStart;          //エンコード開始時刻
    std::chrono::system_clock::time_point m_tmLastUpdate;     //最終更新時刻
    RGYParallelEncodeStatusData *m_peStatusShare; // 子エンコーダ側から親への進捗表示共有するためのクラスへのポインタ (実体はRGYParallelEncProcess::m_sendData::encStatus)
    std::vector<std::pair<double, RGYParallelEncodeStatusData*>> m_childStatus; // 親側で使用する、子エンコーダの担当割合と子エンコーダから進捗表示を取得するクラスへのポインタ (実体はRGYParallelEncProcess::m_sendData::encStatus)
    bool m_bStdErrWriteToConsole;
    bool m_bEncStarted;
};

class CProcSpeedControl {
public:
    CProcSpeedControl(uint32_t maxProcessPerSec, uint32_t checkInterval = 4);
    virtual ~CProcSpeedControl();
    void setSpeed(uint32_t maxProcessPerSec);
    void reset();
    bool wait();
    bool wait(uint32_t nCount);
private:
    uint32_t m_nCountLast;
    uint32_t m_nCheckInterval;
    bool m_bEnable;
    std::chrono::microseconds m_tmThreshold;
    std::chrono::high_resolution_clock::time_point m_tmLastCheck;
};

#endif //__RGY_STATUS_H__
