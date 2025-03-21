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

#pragma once
#ifndef __RGY_PARALLEL_ENC_H__
#define __RGY_PARALLEL_ENC_H__

#include <thread>
#include <optional>
#include <mutex>
#include "rgy_osdep.h"
#include "rgy_err.h"
#include "rgy_event.h"
#include "rgy_log.h"
#include "rgy_prm.h"
#include "rgy_queue.h"

struct RGYOutputRawPEExtHeader;
class EncodeStatus;
struct EncodeStatusData;
class RGYInput;
#if ENCODER_QSV
struct sInputParams;
using encParams = sInputParams;

class CQSVPipeline;
using encCore = CQSVPipeline;
#elif ENCODER_NVENC
struct InEncodeVideoParam;
using encParams = InEncodeVideoParam;

class NVEncCore;
using encCore = NVEncCore;
#elif ENCODER_VCEENC
struct VCEParam;
using encParams = VCEParam;

class VCECore;
using encCore = VCECore;
#elif ENCODER_MPP
struct MPPParam;
using encParams = MPPParam;

class MPPCore;
using encCore = MPPCore;
#endif

class RGYParallelEncodeStatusData {
protected:
    std::unique_ptr<EncodeStatusData> encStatusData;
    std::mutex mtx_;
public:
    RGYParallelEncodeStatusData();
    void set(const EncodeStatusData& data);
    bool get(EncodeStatusData& data);
    void reset();
};

enum class RGYParallelEncProcessStatus {
    Init,
    Running,
    Finished,
    Error,
};

struct RGYParallelEncSendData {
    RGYParallelEncProcessStatus processStatus;

    std::shared_ptr<std::mutex> logMutex;

    unique_event eventChildHasSentFirstKeyPts; // 子→親へ最初のキーフレームのptsを通知するためのイベント
    int64_t videoFirstKeyPts; // 子の最初のキーフレームのpts
    RGYParallelEncodeStatusData encStatus; // 子のエンコード状態を共有用 (親へはEncodeStatus::addChildStatusで渡す、子へはEncodeStatus::initで渡す)

    unique_event eventParentHasSentFinKeyPts; // 親→子へ最後のptsを通知するためのイベント
    int64_t videoFinKeyPts;  // 子がエンコードすべき最後のpts(このptsを含まない)
    
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessData; // 最初の子エンコードから親へエンコード結果を転送するキュー
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessDataFree; // 転送し終わった(不要になった)ポインタを回収するキュー
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessDataFreeLarge; // 転送し終わった(不要になった)ポインタを回収するキュー(大きいサイズ用)

    RGYParallelEncSendData() :
        processStatus(RGYParallelEncProcessStatus::Init),
        logMutex(),
        eventChildHasSentFirstKeyPts(unique_event(nullptr, nullptr)),
        videoFirstKeyPts(-1),
        encStatus(),
        eventParentHasSentFinKeyPts(unique_event(nullptr, nullptr)),
        videoFinKeyPts(-1),
        qFirstProcessData(nullptr),
        qFirstProcessDataFree(nullptr),
        qFirstProcessDataFreeLarge(nullptr) {};
};

class RGYParallelEncProcess {
public:
    RGYParallelEncProcess(const int id, const tstring& tmpfile, std::shared_ptr<RGYLog> log);
    ~RGYParallelEncProcess();
    RGY_ERR startThread(const encParams& peParams);
    RGY_ERR run(const encParams& peParams);
    int id() const { return m_id; }
    RGY_ERR sendEndPts(const int64_t endPts);
    RGY_ERR close(const bool deleteTempFile);
    tstring tmpPath() const { return m_tmpfile; }
    RGYParamParallelEncCache cacheMode() const { return m_cacheMode; }
    RGY_ERR getNextPacket(RGYOutputRawPEExtHeader **ptr);
    RGY_ERR putFreePacket(RGYOutputRawPEExtHeader *ptr);

    int waitProcessFinished(const uint32_t timeout);
    std::optional<RGY_ERR> getThreadRunResult();

    int waitProcessStarted(const uint32_t timeout);
    int64_t getVideoFirstKeyPts() const { return m_sendData.videoFirstKeyPts; } // waitProcessStarted してから呼ぶこと
    RGYParallelEncodeStatusData *getEncodeStatus() { return &m_sendData.encStatus; } // waitProcessStarted してから呼ぶこと
    RGYParallelEncProcessStatus processStatus() const { return m_sendData.processStatus; }
    HANDLE eventProcessFinished() const { return m_processFinished.get(); }
protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_APP, strsprintf(_T("PE%d: %s\n"), m_id, line.c_str()).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    int m_id;
    std::unique_ptr<encCore> m_process;
    std::unique_ptr<RGYQueueMPMP<RGYOutputRawPEExtHeader*>> m_qFirstProcessData;
    std::unique_ptr<RGYQueueMPMP<RGYOutputRawPEExtHeader*>> m_qFirstProcessDataFree;
    std::unique_ptr<RGYQueueMPMP<RGYOutputRawPEExtHeader*>> m_qFirstProcessDataFreeLarge;
    RGYParamParallelEncCache m_cacheMode;
    RGYParallelEncSendData m_sendData;
    tstring m_tmpfile;
    std::thread m_thRunProcess;
    std::optional<RGY_ERR> m_thRunProcessRet;
    unique_event m_processFinished;
    bool m_thAbort;
    std::shared_ptr<RGYLog> m_log;
};

class RGYParallelEnc {
public:
    RGYParallelEnc(std::shared_ptr<RGYLog> log);
    virtual ~RGYParallelEnc();
    static std::pair<RGY_ERR, const TCHAR *> isParallelEncPossible(const encParams *prm, const RGYInput *input);
    RGY_ERR parallelRun(encParams *prm, const RGYInput *input, rgy_rational<int> outputTimebase, EncodeStatus *encStatus);
    void close(const bool deleteTempFiles);
    int64_t getVideoEndKeyPts() const { return m_videoEndKeyPts; }
    void setVideoFinished() { m_videoFinished = true; }
    bool videoFinished() const { return m_videoFinished; }
    int id() const { return m_id; }
    int waitProcessFinished(const int id, const uint32_t timeout);
    std::optional<RGY_ERR> processReturnCode(const int id);
    RGY_ERR checkAllProcessErrors();
    void encStatusReset(const int id);
    
    int64_t getVideofirstKeyPts(const int ichunk) const;
    RGY_ERR getNextPacket(const int ichunk, RGYOutputRawPEExtHeader **ptr);
    RGY_ERR putFreePacket(const int ichunk, RGYOutputRawPEExtHeader *ptr);
    RGYParamParallelEncCache cacheMode(const int ichunk) const;
    tstring tmpPath(const int ichunk) const;
    int parallelCount() const { return m_parallelCount; }
    int chunks() const { return m_chunks; }
protected:
    encParams genPEParam(const int ip, const encParams *prm, rgy_rational<int> outputTimebase, const tstring& tmpfile);
    RGY_ERR startChunkProcess(int ichunk, const encParams *prm, int64_t parentFirstKeyPts, rgy_rational<int> outputTimebase, EncodeStatus *encStatus);
    RGY_ERR startParallelThreads(const encParams *prm, const RGYInput *input, rgy_rational<int> outputTimebase, EncodeStatus *encStatus);
    RGY_ERR parallelChild(const encParams *prm, const RGYInput *input);

    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_CORE_PARALLEL)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_CORE_PARALLEL, (_T("parallel: ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_CORE_PARALLEL)) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }

    int m_id;
    std::vector<std::unique_ptr<RGYParallelEncProcess>> m_encProcess;
    std::shared_ptr<RGYLog> m_log;
    std::thread m_thParallelRun;
    bool m_thParallelRunAbort;
    int64_t m_videoEndKeyPts;
    bool m_videoFinished;
    int m_parallelCount;
    int m_chunks;
};


#endif //__RGY_PARALLEL_ENC_H__
