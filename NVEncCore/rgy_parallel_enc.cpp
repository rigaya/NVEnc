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

#include "rgy_parallel_enc.h"
#include "rgy_filesystem.h"
#include "rgy_input.h"
#include "rgy_output.h"
#if ENCODER_QSV
#include "qsv_pipeline.h"
#elif ENCODER_NVENC
#include "NVEncCore.h"
#elif ENCODER_VCEENC
#include "vce_core.h"
#elif ENCODER_MPP
#include "mpp_core.h"
#endif
#include "rgy_perf_monitor.h"

static const int RGY_PARALLEL_ENC_TIMEOUT = 10000;

static RGY_CODEC enc_codec(const encParams *prm) {
#if ENCODER_NVENC
    return prm->codec_rgy;
#else
    return prm->codec;
#endif
}

RGYParallelEncodeStatusData::RGYParallelEncodeStatusData() : encStatusData(), mtx_() {};

void RGYParallelEncodeStatusData::set(const EncodeStatusData& data) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (encStatusData) {
        *encStatusData = data;
    } else {
        encStatusData = std::make_unique<EncodeStatusData>(data);
    }
}

bool RGYParallelEncodeStatusData::get(EncodeStatusData& data) {
    if (!encStatusData) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (!encStatusData) {
        return false;
    }
    data = *encStatusData;
    return true;
}

void RGYParallelEncodeStatusData::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    encStatusData.reset();
}

RGYParallelEncProcess::RGYParallelEncProcess(const int id, const tstring& tmpfile, std::shared_ptr<RGYLog> log) :
    m_id(id),
    m_process(),
    m_qFirstProcessData(),
    m_qFirstProcessDataFree(),
    m_qFirstProcessDataFreeLarge(),
    m_cacheMode(RGYParamParallelEncCache::Mem),
    m_sendData(),
    m_tmpfile(tmpfile),
    m_thRunProcess(),
    m_thRunProcessRet(),
    m_processFinished(unique_event(nullptr, nullptr)),
    m_thAbort(false),
    m_log(log) {
}

RGYParallelEncProcess::~RGYParallelEncProcess() {
    close(false);
}

RGY_ERR RGYParallelEncProcess::close(const bool deleteTempFiles) {
    auto err = RGY_ERR_NONE;
    if (m_thRunProcess.joinable()) {
        m_thAbort = true;
        m_thRunProcess.join();
        m_sendData.processStatus = RGYParallelEncProcessStatus::Finished;
        if (m_qFirstProcessData) {
            m_qFirstProcessData->close([](RGYOutputRawPEExtHeader **ptr) { if (*ptr) free(*ptr); });
            m_qFirstProcessData.reset();
        }
        if (m_qFirstProcessDataFree) {
            m_qFirstProcessDataFree->close([](RGYOutputRawPEExtHeader **ptr) { if (*ptr) free(*ptr); });
            m_qFirstProcessDataFree.reset();
        }
        if (m_qFirstProcessDataFreeLarge) {
            m_qFirstProcessDataFreeLarge->close([](RGYOutputRawPEExtHeader **ptr) { if (*ptr) free(*ptr); });
            m_qFirstProcessDataFreeLarge.reset();
        }
    }
    if (deleteTempFiles && m_tmpfile.length() > 0 && rgy_file_exists(m_tmpfile)) {
        rgy_file_remove(m_tmpfile.c_str());
    }
    m_processFinished.reset();
    return err;
}

RGY_ERR RGYParallelEncProcess::run(const encParams& peParams) {
#if ENABLE_PARALLEL_ENC
    m_process = std::make_unique<encCore>();
    m_cacheMode = peParams.ctrl.parallelEnc.cacheMode;

    encParams encParam = peParams;
    encParam.ctrl.parallelEnc.sendData = &m_sendData;
#if ENCODER_QSV || ENCODER_NVENC
    auto sts = m_process->Init(&encParam);
#elif ENCODER_VCEENC
    auto sts = m_process->init(&encParam);
#endif
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_process->SetAbortFlagPointer(&m_thAbort);

#if ENCODER_QSV
    if ((sts = m_process->CheckCurrentVideoParam()) == RGY_ERR_NONE
        && (sts = m_process->Run()) == RGY_ERR_NONE) {
        m_process->Close();
    }
#elif ENCODER_NVENC
    if ((sts = m_process->Encode()) == RGY_ERR_NONE) {
        m_process->Deinitialize();
    }
#elif ENCODER_VCEENC
    if ((sts = m_process->run2()) == RGY_ERR_NONE) {
        m_process->Terminate();
    }
#endif
    m_process.reset();
    m_sendData.logMutex.reset();
    return sts;
#else
    return RGY_ERR_UNSUPPORTED;
#endif
}

RGY_ERR RGYParallelEncProcess::startThread(const encParams& peParams, CPerfMonitor *perfMonitor) {
    m_sendData.logMutex = m_log->getLock();
    m_sendData.eventChildHasSentFirstKeyPts = CreateEventUnique(nullptr, FALSE, FALSE);
    m_sendData.eventParentHasSentFinKeyPts = CreateEventUnique(nullptr, FALSE, FALSE);
#if ENABLE_PERF_COUNTER
    if (perfMonitor) {
        m_sendData.perfCounter = perfMonitor->perfCounter();
    }
#endif
    m_processFinished = CreateEventUnique(nullptr, TRUE, FALSE); // 処理終了の通知用
    if (peParams.ctrl.parallelEnc.parallelId == 0 || peParams.ctrl.parallelEnc.cacheMode == RGYParamParallelEncCache::Mem) {
        // 最初のプロセスあるいはキャッシュメモリモードでは、キューを介してデータをやり取りする
        m_qFirstProcessData = std::make_unique<RGYQueueMPMP<RGYOutputRawPEExtHeader*>>();
        m_qFirstProcessDataFree = std::make_unique<RGYQueueMPMP<RGYOutputRawPEExtHeader*>>();
        m_qFirstProcessDataFreeLarge = std::make_unique<RGYQueueMPMP<RGYOutputRawPEExtHeader*>>();
        m_qFirstProcessData->init();
        m_qFirstProcessDataFree->init();
        m_qFirstProcessDataFreeLarge->init();
        m_sendData.qFirstProcessData = m_qFirstProcessData.get(); // キューのポインタを渡す
        m_sendData.qFirstProcessDataFree = m_qFirstProcessDataFree.get(); // キューのポインタを渡す
        m_sendData.qFirstProcessDataFreeLarge = m_qFirstProcessDataFreeLarge.get(); // キューのポインタを渡す
    }
    m_thRunProcess = std::thread([&]() {
        AddMessage(RGY_LOG_DEBUG, _T("\nPE%d[%d]: Start thread...\n"), m_id, GetCurrentThreadId());
        try {
            m_thRunProcessRet = run(peParams);
        } catch (...) {
            m_thRunProcessRet = RGY_ERR_UNKNOWN;
        }
        // 進捗表示のfpsを0にする
        EncodeStatusData encStatusData;
        if (m_sendData.encStatus.get(encStatusData)) {
            encStatusData.encodeFps = 0.0;
            m_sendData.encStatus.set(encStatusData);
        }
        SetEvent(m_processFinished.get()); // 処理終了を通知するのを忘れないように
        AddMessage(m_thRunProcessRet.value_or(RGY_ERR_UNKNOWN) == RGY_ERR_NONE ? RGY_LOG_DEBUG : RGY_LOG_ERROR,
            _T("\nPE%d[%d]: Processing finished: %s\n"), m_id, GetCurrentThreadId(), get_err_mes(m_thRunProcessRet.value()));
        m_sendData.processStatus = RGYParallelEncProcessStatus::Finished;
        // そのチャンクに関する進捗表示はこのまま残す (最後に最終状態が記録されている)
    });
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEncProcess::getNextPacket(RGYOutputRawPEExtHeader **ptr) {
    if (!m_qFirstProcessData) {
        return RGY_ERR_NULL_PTR;
    }
    size_t nSize = 0;
    *ptr = nullptr;
    while (!m_qFirstProcessData->front_copy_and_pop_no_lock(ptr, &nSize)) {
        if (nSize == 0 && m_thRunProcessRet.has_value()) { // キューにデータがなく、かつ処理が終了している
            return m_thRunProcessRet.value() == RGY_ERR_NONE ? RGY_ERR_MORE_BITSTREAM : m_thRunProcessRet.value();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if ((*ptr == nullptr)) {
        return RGY_ERR_MORE_BITSTREAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEncProcess::putFreePacket(RGYOutputRawPEExtHeader *ptr) {
    if (!m_qFirstProcessDataFree) {
        return RGY_ERR_NULL_PTR;
    }
    if (ptr->allocSize == 0) {
        return RGY_ERR_UNKNOWN;
    }
    // もう終了していた場合は再利用する必要はないのでメモリを解放する
    if (m_sendData.processStatus == RGYParallelEncProcessStatus::Finished) {
        free(ptr);
        return RGY_ERR_NONE;
    }
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *freeQueue = (ptr->allocSize <= RGY_PE_EXT_HEADER_DATA_NORMAL_BUF_SIZE) ? m_qFirstProcessDataFree.get() : m_qFirstProcessDataFreeLarge.get();
    freeQueue->push(ptr);
    return RGY_ERR_NONE;
}

int RGYParallelEncProcess::waitProcessStarted(const uint32_t timeout) {
    return WaitForSingleObject(m_sendData.eventChildHasSentFirstKeyPts.get(), timeout) == WAIT_OBJECT_0 ? 0 : 1;
}

RGY_ERR RGYParallelEncProcess::sendEndPts(const int64_t endPts) {
    if (!m_sendData.eventParentHasSentFinKeyPts) {
        return RGY_ERR_UNDEFINED_BEHAVIOR;
    }
    m_sendData.videoFinKeyPts = endPts;
    m_sendData.processStatus = RGYParallelEncProcessStatus::Running;
    SetEvent(m_sendData.eventParentHasSentFinKeyPts.get());
    return RGY_ERR_NONE;
}

std::optional<RGY_ERR> RGYParallelEncProcess::getThreadRunResult() {
    if (!m_thRunProcessRet.has_value() && m_thRunProcess.native_handle()) {
        const bool threadActive = RGYThreadStillActive(m_thRunProcess.native_handle());
        if (!threadActive) {
            m_thRunProcessRet = RGY_ERR_UNKNOWN;
            m_sendData.processStatus = RGYParallelEncProcessStatus::Finished;
            SetEvent(m_processFinished.get());
        }
    }
    return m_thRunProcessRet;
}

int RGYParallelEncProcess::waitProcessFinished(const uint32_t timeout) {
    if (!m_processFinished) {
        return -1;
    }
    return WaitForSingleObject(m_processFinished.get(), timeout);
}

RGYParallelEnc::RGYParallelEnc(std::shared_ptr<RGYLog> log) :
    m_id(-1),
    m_encProcess(),
    m_log(log),
    m_thParallelRun(),
    m_thParallelRunAbort(false),
    m_videoEndKeyPts(-1),
    m_videoFinished(false),
    m_parallelCount(0),
    m_chunks(0) {}

RGYParallelEnc::~RGYParallelEnc() {
    close(false);
}

void RGYParallelEnc::close(const bool deleteTempFiles) {
    if (m_thParallelRun.joinable()) {
        m_thParallelRunAbort = true;
        m_thParallelRun.join();
    }
    for (auto &proc : m_encProcess) {
        proc->close(deleteTempFiles);
    }
    m_encProcess.clear();
    m_videoEndKeyPts = -1;
}

int64_t RGYParallelEnc::getVideofirstKeyPts(const int ichunk) const {
    if (ichunk >= (int)m_encProcess.size()) {
        return -1;
    }
    return m_encProcess[ichunk]->getVideoFirstKeyPts();
}

tstring RGYParallelEnc::tmpPath(const int ichunk) const {
    if (ichunk >= (int)m_encProcess.size()) {
        return _T("");
    }
    return m_encProcess[ichunk]->tmpPath();
}

RGYParamParallelEncCache RGYParallelEnc::cacheMode(const int ichunk) const {
    if (ichunk >= (int)m_encProcess.size()) {
        return RGYParamParallelEncCache::Mem;
    }
    return m_encProcess[ichunk]->cacheMode();
}

RGY_ERR RGYParallelEnc::getNextPacket(const int ichunk, RGYOutputRawPEExtHeader **ptr) {
    if (m_encProcess.size() == 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid call for getNextPacketFromFirst.\n"));
        return RGY_ERR_UNKNOWN;
    }
    return m_encProcess[ichunk]->getNextPacket(ptr);
}

RGY_ERR RGYParallelEnc::putFreePacket(const int ichunk, RGYOutputRawPEExtHeader *ptr) {
    if (m_encProcess.size() == 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid call for pushNextPacket.\n"));
        return RGY_ERR_UNKNOWN;
    }
    return m_encProcess[ichunk]->putFreePacket(ptr);
}

int RGYParallelEnc::waitProcessFinished(const int id, const uint32_t timeout) {
    if (id >= (int)m_encProcess.size()) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parallel id #%d for waitProcess.\n"), id);
        return -1;
    }
    return m_encProcess[id]->waitProcessFinished(timeout);
}

std::optional<RGY_ERR> RGYParallelEnc::processReturnCode(const int id) {
    if (id >= (int)m_encProcess.size()) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parallel id #%d for processReturnCode.\n"), id);
        return std::nullopt;
    }
    return m_encProcess[id]->getThreadRunResult();
}

RGY_ERR RGYParallelEnc::checkAllProcessErrors() {
    for (const auto& proc : m_encProcess) {
        auto returnCode = proc->getThreadRunResult();
        if (returnCode.has_value() && returnCode.value() != RGY_ERR_NONE) {
            return returnCode.value();
        }
    }
    return RGY_ERR_NONE;
}

void RGYParallelEnc::encStatusReset(const int id) {
    if (id >= (int)m_encProcess.size()) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parallel id #%d for encStatusReset.\n"), id);
        return;
    }
    m_encProcess[id]->getEncodeStatus()->reset();
}

std::pair<RGY_ERR, const TCHAR *> RGYParallelEnc::isParallelEncPossible(const encParams *prm, const RGYInput *input) {
    if (input->isPipe() && prm->ctrl.parallelEnc.chunkPipeHandles.size() == 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: input is pipe.\n") };
    }
    if (!input->seekable() && prm->ctrl.parallelEnc.chunkPipeHandles.size() == 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: input does not support parallel encoding or input is not seekable.\n") };
    }
    if (!input->timestampStable()) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: input frame timestamp (sometimes) unclear.\n") };
    }
    if (input->GetVideoFirstKeyPts() < 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: invalid first key PTS.\n") };
    }
    if (enc_codec(prm) == RGY_CODEC_RAW || enc_codec(prm) == RGY_CODEC_AVCODEC) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: encoding is not activated.\n") };
    }
    if (prm->common.nTrimCount != 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --trim is eanbled.\n") };
    }
#if ENCODER_QSV || ENCODER_NVENC
    if (prm->dynamicRC.size() > 0 && prm->ctrl.parallelEnc.chunkPipeHandles.size() == 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --dynamic-rc is eanbled.\n") };
    }
#endif
    if (prm->common.timecodeFile.length() != 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --timecode is specified.\n") };
    }
    if (prm->common.tcfileIn.length() != 0 && prm->ctrl.parallelEnc.chunkPipeHandles.size() == 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --tcfile-in is specified.\n") };
    }
    if (prm->common.keyFile.length() != 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --keyfile is specified.\n") };
    }
    if (prm->common.metric.enabled()) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: ssim/psnr/vmaf is enabled.\n") };
    }
    if (prm->common.keyOnChapter) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --key-on-chapter is enabled.\n") };
    }
    if (prm->vpp.subburn.size() != 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --vpp-subburn is specified.\n") };
    }
    if (prm->vpp.fruc.enable != 0) {
        return { RGY_ERR_UNSUPPORTED, _T("Parallel encoding is not possible: --vpp-fruc is enabled.\n") };
    }
    return { RGY_ERR_NONE, _T("") };
}

RGY_ERR RGYParallelEnc::parallelChild(const encParams *prm, const RGYInput *input) {
    // 起動したプロセスから最初のキーフレームのptsを取得して、親プロセスに送る
    auto sendData = prm->ctrl.parallelEnc.sendData;
    sendData->videoFirstKeyPts = input->GetVideoFirstKeyPts();
    SetEvent(sendData->eventChildHasSentFirstKeyPts.get());

    // 親プロセスから終了時刻を受け取る
    WaitForSingleObject(sendData->eventParentHasSentFinKeyPts.get(), INFINITE);
    m_videoEndKeyPts = sendData->videoFinKeyPts;
    sendData->processStatus = RGYParallelEncProcessStatus::Running;
    AddMessage(RGY_LOG_DEBUG, _T("Start chunk process %d.\n"), prm->ctrl.parallelEnc.parallelId);
    return RGY_ERR_NONE;
}

RGYParamLogLevel RGYParallelEnc::setChildLogLevel(const RGYParamLogLevel& logLevel) {
    RGYParamLogLevel logLevelChild = logLevel;
    for (int i = 0; i < RGY_LOGT_FIN; i++) {
        const auto logType = (RGYLogType)(i);
        if (logType != RGY_LOGT_APP && logType != RGY_LOGT_ALL) { // 一括設定のものは除く
            const auto level = logLevel.get(logType);
            logLevelChild.set(level == RGY_LOG_INFO ? RGY_LOG_WARN : level, logType); // INFO->WARNに変更
        }
    }
    return logLevelChild;
}

encParams RGYParallelEnc::genPEParam(const int ip, const encParams *prm, rgy_rational<int> outputTimebase, const bool delayChildSync, const tstring& tmpfile) {
    encParams prmParallel = *prm;
    prmParallel.ctrl.parallelEnc.parallelId = ip;
    prmParallel.ctrl.parentProcessID = GetCurrentProcessId();
    prmParallel.ctrl.loglevel = setChildLogLevel(prm->ctrl.loglevel);
    prmParallel.ctrl.parallelEnc.cacheMode = (ip == 0) ? RGYParamParallelEncCache::Mem : prm->ctrl.parallelEnc.cacheMode; // parallelId = 0 は必ずMem キャッシュモード
    prmParallel.ctrl.parallelEnc.delayChildSync = delayChildSync;
    // 親が子の実行すべきchunkを選択して先頭に設定しておく
    if (prm->ctrl.parallelEnc.chunkPipeHandles.size() > 0) {
        prmParallel.ctrl.parallelEnc.chunkPipeHandles = { prm->ctrl.parallelEnc.chunkPipeHandles[ip] };
    }
#if __has_include("rgy_opencl.h")
    prmParallel.ctrl.openclBuildThreads = std::max(1, (prm->ctrl.openclBuildThreads > 0 ? prm->ctrl.openclBuildThreads : std::min(RGY_OPENCL_BUILD_THREAD_DEFAULT_MAX, (int)std::thread::hardware_concurrency())) / prm->ctrl.parallelEnc.parallelCount); // 並列数の制限
#endif
    prmParallel.common.muxOutputFormat = _T("raw");
    prmParallel.common.outputFilename = tmpfile; // ip==0の場合のみ、実際にはキューを介してデータをやり取りするがとりあえずファイル名はそのまま入れる
    prmParallel.common.AVMuxTarget = RGY_MUX_NONE;
    prmParallel.common.audioSource.clear();
    prmParallel.common.subSource.clear();
    prmParallel.common.attachmentSource.clear();
    prmParallel.common.nAudioSelectCount = 0;
    prmParallel.common.ppAudioSelectList = nullptr;
    prmParallel.common.nSubtitleSelectCount = 0;
    prmParallel.common.ppSubtitleSelectList = nullptr;
    prmParallel.common.nDataSelectCount = 0;
    prmParallel.common.ppDataSelectList = nullptr;
    prmParallel.common.nAttachmentSelectCount = 0;
    prmParallel.common.ppAttachmentSelectList = nullptr;
    prmParallel.common.outReplayCodec = RGY_CODEC_UNKNOWN;
    prmParallel.common.outReplayFile.clear();
    prmParallel.common.seekRatio = ip / (float)prmParallel.ctrl.parallelEnc.chunks;
    prmParallel.common.timebase = outputTimebase; // timebaseがずれると致命的なので、強制的に上書きする
    prmParallel.common.dynamicHdr10plusJson.clear(); // hdr10plusのファイルからの読み込みは親プロセスでmux時に行う
    prmParallel.common.doviRpuFile.clear(); // doviRpuのファイルからの読み込みは親プロセスでmux時に行う
    prmParallel.common.masterDisplay.clear(); // 親プロセスでmux時に行う
    prmParallel.common.maxCll.clear(); // 親プロセスでmux時に行う
    prmParallel.common.timecode = false; // 親プロセスでmux時に行う
    prmParallel.common.timecodeFile.clear(); // 親プロセスでmux時に行う
#if ENCODER_QSV || ENCODER_NVENC
    if (prmParallel.dynamicRC.size() > 0) {
        const auto chunkStartFrameId = prmParallel.ctrl.parallelEnc.chunkPipeHandles.front().startFrameId;
        if (chunkStartFrameId < 0) {
            prmParallel.dynamicRC.clear(); // ここには来ないはず
        } else {
            const auto origDynamicRC = prmParallel.dynamicRC;
            prmParallel.dynamicRC.clear();
            for (auto& origRC : origDynamicRC) {
                auto rc = origRC;
                rc.start = std::max(0, rc.start - chunkStartFrameId);
                rc.end = rc.end - chunkStartFrameId;
                if (rc.end >= 0) {
                    prmParallel.dynamicRC.push_back(rc);
                }
            }
        }
    }
#endif
    return prmParallel;
}

RGY_ERR RGYParallelEnc::startChunkProcess(const int ip, const encParams *prm, int64_t parentFirstKeyPts, rgy_rational<int> outputTimebase, const bool delayChildSync, EncodeStatus *encStatus, CPerfMonitor *perfMonitor) {
    const auto tmpfile = prm->common.outputFilename + _T(".pe") + std::to_tstring(ip);
    const auto peParam = genPEParam(ip, prm, outputTimebase, delayChildSync, tmpfile);
    auto process = std::make_unique<RGYParallelEncProcess>(ip, tmpfile, m_log);
    if (auto err = process->startThread(peParam, perfMonitor); err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to run PE%d: %s.\n"), ip, get_err_mes(err));
        return err;
    }
    // 起動したプロセスから最初のキーフレームのptsを取得
    while (process->waitProcessStarted(16)) {
        const auto ret = process->getThreadRunResult();
        if (ret.has_value() || process->processStatus() == RGYParallelEncProcessStatus::Finished) {
            return ret.value_or(RGY_ERR_UNKNOWN);
        }
    }
    const auto firstKeyPts = process->getVideoFirstKeyPts();
    if (firstKeyPts < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get first key pts from PE%d.\n"), ip);
        return RGY_ERR_UNKNOWN;
    }
    if (ip == 0) {
        if (firstKeyPts != parentFirstKeyPts) {
            AddMessage(RGY_LOG_ERROR, _T("First key pts mismatch: parent %lld, PE0 %lld.\n"), parentFirstKeyPts, firstKeyPts);
            return RGY_ERR_UNKNOWN;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("PE%d: Got first key pts: raw %lld, offset %lld.\n"), ip, firstKeyPts, firstKeyPts - parentFirstKeyPts);
    encStatus->addChildStatus({ 1.0f / prm->ctrl.parallelEnc.parallelCount, process->getEncodeStatus() });

    AddMessage(RGY_LOG_DEBUG, _T("Started encoder PE%d.\n"), ip);
    m_encProcess.push_back(std::move(process));
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEnc::startParallelThreads(const encParams *prm, const RGYInput *input, rgy_rational<int> outputTimebase, const bool delayChildSync, EncodeStatus *encStatus, CPerfMonitor *perfMonitor) {
    const auto parentFirstKeyPts = input->GetVideoFirstKeyPts();
    m_encProcess.clear();
    // とりあえず、並列数分起動する
    int startId = 0;
    for (; startId < prm->ctrl.parallelEnc.parallelCount; startId++) {
        if (auto err = startChunkProcess(startId, prm, parentFirstKeyPts, outputTimebase, delayChildSync, encStatus, perfMonitor); err != RGY_ERR_NONE) {
            return err;
        }

        // 起動したプロセスの最初のキーフレームはひとつ前のプロセスのエンコード終了時刻
        if (startId > 0) {
            // ひとつ前のプロセスの終了時刻として転送
            const auto firstKeyPts = m_encProcess[startId]->getVideoFirstKeyPts();
            AddMessage(RGY_LOG_DEBUG, _T("Send PE%d end key pts %lld.\n"), startId - 1, firstKeyPts);
            auto err = m_encProcess[startId-1]->sendEndPts(firstKeyPts);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to send end pts to PE%d: %s.\n"), startId, get_err_mes(err));
                return err;
            }
        }
    }
    // 並列数とチャック数が同じなら、これで起動は完了
    if (prm->ctrl.parallelEnc.parallelCount >= prm->ctrl.parallelEnc.chunks) {
        //最後のプロセスの終了時刻(=終わりまで)を転送
        AddMessage(RGY_LOG_DEBUG, _T("Send PE%d end key pts -1.\n"), (int)m_encProcess.size() - 1);
        auto err = m_encProcess.back()->sendEndPts(-1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to send end pts to encoder PE%d: %s.\n"), (int)m_encProcess.size() - 1, get_err_mes(err));
            return err;
        }
    } else {
        // プロセス起動用のスレッドを開始する
        m_thParallelRun = std::thread([this](int startId, const int parallelCount, const int chunkCount,
            encParams prm, rgy_rational<int> outputTimebase, const bool delayChildSync, EncodeStatus *encStatus, CPerfMonitor *perfMonitor) {

            int runCount = startId-1; // 実行したチャック数

            // とりあえず並列数分開始準備
            for (; startId < chunkCount; startId++) {
                if (auto err = startChunkProcess(startId, &prm, -1, outputTimebase, delayChildSync, encStatus, perfMonitor); err != RGY_ERR_NONE) {
                    return err;
                }
            }

            while (!m_thParallelRunAbort && runCount < chunkCount) {
                // 実行中のプロセス数を取得
                const auto runningCount = std::count_if(m_encProcess.begin(), m_encProcess.end(), [](const auto& proc) {
                    return proc->processStatus() == RGYParallelEncProcessStatus::Running;
                });
                if (runningCount < parallelCount) {
                    // 実行中のプロセス数が並列数より少ない場合、新しいプロセスを起動する
                    if (runCount < chunkCount-1) {
                        // ひとつ前のプロセスの終了時刻として転送
                        const auto firstKeyPts = m_encProcess[runCount+1]->getVideoFirstKeyPts();
                        AddMessage(RGY_LOG_DEBUG, _T("Send PE%d end key pts %lld.\n"), runCount, firstKeyPts);
                        auto err = m_encProcess[runCount]->sendEndPts(firstKeyPts);
                        if (err != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_ERROR, _T("Failed to send end pts to PE%d: %s.\n"), startId, get_err_mes(err));
                            return err;
                        }
                    } else {
                        auto err = m_encProcess.back()->sendEndPts(-1);
                        if (err != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_ERROR, _T("Failed to send end pts to encoder PE%d: %s.\n"), (int)m_encProcess.size() - 1, get_err_mes(err));
                            return err;
                        }
                    }
                    runCount++; // 実行したチャック数をインクリメント
                } else {
                    // 実行中のプロセス数が並列数と同じ場合、いずれかのプロセスが終了するまで待つ
                    std::vector<HANDLE> eventProcessFinished;
                    for (const auto& proc : m_encProcess) {
                        if (proc->processStatus() != RGYParallelEncProcessStatus::Finished) {
                            eventProcessFinished.push_back(proc->eventProcessFinished());
                        }
                    }
                    WaitForMultipleObjects((uint32_t)eventProcessFinished.size(), eventProcessFinished.data(), FALSE, 16);
                }
            }
            return RGY_ERR_NONE;
        }, startId, prm->ctrl.parallelEnc.parallelCount, prm->ctrl.parallelEnc.chunks, *prm, outputTimebase, delayChildSync, encStatus, perfMonitor);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEnc::parallelRun(encParams *prm, const RGYInput *input, rgy_rational<int> outputTimebase, const bool delayChildSync, EncodeStatus *encStatus, CPerfMonitor *perfMonitor) {
    if (!prm->ctrl.parallelEnc.isEnabled()) {
        return RGY_ERR_NONE;
    }
    m_id = prm->ctrl.parallelEnc.parallelId;
    m_parallelCount = prm->ctrl.parallelEnc.parallelCount;
    if (prm->ctrl.parallelEnc.isChild()) { // 子プロセスから呼ばれた
        return parallelChild(prm, input); // 子プロセスの処理
    }
    if (prm->ctrl.parallelEnc.chunkPipeHandles.size() > 0) {
        prm->ctrl.parallelEnc.chunks = (int)prm->ctrl.parallelEnc.chunkPipeHandles.size();
    } else if (prm->ctrl.parallelEnc.chunks <= 0) {
        prm->ctrl.parallelEnc.chunks = prm->ctrl.parallelEnc.parallelCount;
    }
    m_chunks = prm->ctrl.parallelEnc.chunks;
    AddMessage(RGY_LOG_DEBUG, _T("parallelRun: parallel count %d, chunks %d\n"), prm->ctrl.parallelEnc.parallelCount, prm->ctrl.parallelEnc.chunks);
    auto [sts, errmes ] = isParallelEncPossible(prm, input);
    if (sts != RGY_ERR_NONE
        || (sts = startParallelThreads(prm, input, outputTimebase, delayChildSync, encStatus, perfMonitor)) != RGY_ERR_NONE) {
        // chunkPipeHandlesの場合は、無効にして続行はできないので、エラー終了
        if (prm->ctrl.parallelEnc.chunkPipeHandles.size() > 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to start parallel threads: %s.\n"), get_err_mes(sts));
            return RGY_ERR_UNKNOWN;
        }
        // 並列処理を無効化して続行する
        // まず終了させるため、スレッドの処理を続行させる
        if (m_encProcess.size() > 0) {
            m_encProcess.back()->sendEndPts(-1);
        }
        m_encProcess.clear();
        prm->ctrl.parallelEnc.parallelCount = 0;
        prm->ctrl.parallelEnc.parallelId = -1;
        return sts;
    }
    return RGY_ERR_NONE;
}