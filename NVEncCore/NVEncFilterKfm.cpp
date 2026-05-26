// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "NVEncFilterKfm.h"
#include "NVEncFilterDegrain.h"
#include "NVEncFilterRtgmcBob.h"
#include "rgy_filesystem.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace {
static constexpr int KFM_SOURCE_VPAD = 4;
static constexpr int KFM_FMCOUNT_PAIRS = 9;
static constexpr int KFM_FMCOUNT_COUNT = KFM_FMCOUNT_PAIRS * 2;
static constexpr int KFM_FMCOUNT_SOURCE_FRAMES = KFM_FMCOUNT_PAIRS + 2;
static constexpr int KFM_FMCOUNT_ASYNC_DELAY_CYCLES = 1;
static constexpr int KFM_THRESH_MOVE_Y = 20;
static constexpr int KFM_THRESH_SHIMA_Y = 12;
static constexpr int KFM_THRESH_MOVE_C = 24;
static constexpr int KFM_THRESH_SHIMA_C = 16;
static constexpr int KFM_CLEAN_THRESH_Y = 10;
static constexpr int KFM_CLEAN_THRESH_C = 8;
static constexpr int KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN = 64;
static constexpr int KFM_REALTIMEPLUS_DEINT60_CACHE_MARGIN = KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN * 2;
static constexpr int KFM_VFR_SOURCE_TRIM_LOOKBEHIND = 8;
static constexpr int KFM_VFR_DEINT60_TRIM_LOOKBEHIND = 16;

static bool kfmDisableCCDuration() {
    const char *env = std::getenv("NVENC_KFM_DISABLE_CC_DURATION");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmCspHasInterleavedUV(const RGY_CSP csp) {
    return csp == RGY_CSP_NV12 || csp == RGY_CSP_P010
        || csp == RGY_CSP_NV16 || csp == RGY_CSP_P210;
}

static int kfmFrameParity(const RGYFrameInfo *frame) {
    return (frame && (frame->picstruct & RGY_PICSTRUCT_BFF)) ? 0 : 1;
}

static std::string kfmStageDumpName(const char *stage) {
    std::string name = (stage && stage[0]) ? stage : "stage";
    for (auto& c : name) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return name + ".y4m";
}

static RGY_ERR kfmWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto& waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            const auto sts = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR kfmRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto sts = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}
}

tstring NVEncFilterParamKfm::print() const {
    return kfm.print();
}

NVEncFilterKfm::NVEncFilterKfm() :
    NVEncFilter(),
    m_rtgmc(),
    m_deint60Rtgmc(),
    m_before60Rtgmc(),
    m_after60Rtgmc(),
    m_nrFilter(),
    m_analyzer(),
    m_kfmFramePool(),
    m_sourceCache(),
    m_deint60Cache(),
    m_before60Cache(),
    m_after60Cache(),
    m_ucfNoiseCache(),
    m_pendingUcfNoiseResults(),
    m_fmCountBufPool(),
    m_ucfNoiseResultBufPool(),
    m_ucfNoiseResultCache(),
    m_pendingUcfNoiseDump(),
    m_deint60SubmittedSourceFrames(0),
    m_before60SubmittedSourceFrames(0),
    m_after60SubmittedSourceFrames(0),
    m_deint60CacheCopyEvent(),
    m_before60CacheCopyEvent(),
    m_after60CacheCopyEvent(),
    m_staticFlag(),
    m_staticWorkFrames(),
    m_analyzeFlags(),
    m_pendingFMCounts(),
    m_pendingVfrOutputs(),
    m_telecineSuperRaw(),
    m_telecineSuperFrames(),
    m_telecineSuperNeighborFrames(),
    m_switchFlagFrames(),
    m_containsCombeFrames(),
    m_combeMaskFrames(),
    m_patchCombeFrames(),
    m_ucfNoiseFieldFrames(),
    m_ucfNoiseGaussTmpFrames(),
    m_ucfNoiseGaussFrames(),
    m_ucfNoiseGaussVert(),
    m_ucfNoiseGaussHori(),
    m_switchFlagWork(),
    m_switchFlagWorkEvent(),
    m_containsCombeCount(),
    m_fpResult(nullptr),
    m_fpFMCount(nullptr),
    m_fpTimecode(nullptr),
    m_fpFrameInfo(nullptr),
    m_fpContainsCombe(nullptr),
    m_fpUcfNoise(nullptr),
    m_switchDurationPath(),
    m_switchTimecodePath(),
    m_stageDumpDir(),
    m_lastAnalyzeResult(),
    m_analyzerOutputResults(),
    m_hasLastAnalyzeResult(false),
    m_analyzerFinalized(false),
    m_switchTimingDumped(false),
    m_analyzeSourceFrames(0),
    m_nextAnalyzeCycle(0),
    m_nextFMCountSubmitCycle(0),
    m_nextFMCountDumpFrame(0),
    m_cachedSourceFrames(0),
    m_nextSwitchN60(0),
    m_nextSwitchPts(0),
    m_hasLastSwitchTiming(false),
    m_lastSwitchStart60(0),
    m_lastSwitchDuration60(0),
    m_lastSwitchStart120(0),
    m_lastSwitchIsFrame24(false),
    m_switchSingleFrameN60(),
    m_stageDumpFrameCounts(),
    m_stageDumpFrameIndices(),
    m_stageDumpTargetFrames(),
    m_nextTelecine24Frame(0),
    m_nextTelecine24Pts(0),
    m_telecineSuperBufferIndex(0),
    m_maskBranchBufferIndex(0),
    m_patchCombeBufferIndex(0),
    m_stageDumpMaxFrames(0),
    m_timecodeFrameIndex(0),
    m_outputBufferIndex(0),
    m_workFrameBuf(),
    m_workBufferIndex(0) {
    m_name = _T("kfm");
}

NVEncFilterKfm::~NVEncFilterKfm() {
    close();
}

std::shared_ptr<CUFrameBuf> NVEncFilterKfm::SharedFramePool::acquire(const RGYFrameInfo& info) {
    for (auto it = m_pool.begin(); it != m_pool.end(); ++it) {
        auto frame = *it;
        if (frame && frame.use_count() == 1
            && !cmpFrameInfoCspResolution(&frame->frame, &info)
            && RGY_CSP_BIT_DEPTH[frame->frame.csp] == RGY_CSP_BIT_DEPTH[info.csp]) {
            m_pool.erase(it);
            return frame;
        }
    }
    auto frame = std::make_shared<CUFrameBuf>(info);
    frame->releasePtr();
    if (frame->alloc() != RGY_ERR_NONE) {
        return nullptr;
    }
    return frame;
}

void NVEncFilterKfm::SharedFramePool::clear() {
    m_pool.clear();
}

std::shared_ptr<CUFrameBuf> NVEncFilterKfm::acquireKfmFrame(const RGYFrameInfo& info, const TCHAR *label) {
    if (!m_kfmFramePool) {
        m_kfmFramePool = std::make_shared<SharedFramePool>();
    }
    auto frame = m_kfmFramePool->acquire(info);
    if (!frame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s frame.\n"), label ? label : _T("cache"));
    }
    return frame;
}

RGY_ERR NVEncFilterKfm::allocWorkFrameBuf(const RGYFrameInfo& frame, int frames) {
    if ((int)m_workFrameBuf.size() == frames
        && !m_workFrameBuf.empty()
        && !cmpFrameInfoCspResolution(&m_workFrameBuf[0]->frame, &frame)
        && RGY_CSP_BIT_DEPTH[m_workFrameBuf[0]->frame.csp] == RGY_CSP_BIT_DEPTH[frame.csp]) {
        bool valid = true;
        for (const auto& work : m_workFrameBuf) {
            if (!work) {
                valid = false;
                break;
            }
            for (int iplane = 0; iplane < RGY_CSP_PLANES[work->frame.csp]; iplane++) {
                if (work->frame.ptr[iplane] == nullptr) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }
        }
        if (valid) {
            m_workBufferIndex = 0;
            return RGY_ERR_NONE;
        }
    }
    m_workFrameBuf.clear();
    for (int i = 0; i < frames; i++) {
        auto work = std::make_unique<CUFrameBuf>(frame);
        work->releasePtr();
        const auto sts = work->alloc();
        if (sts != RGY_ERR_NONE) {
            m_workFrameBuf.clear();
            return sts;
        }
        m_workFrameBuf.push_back(std::move(work));
    }
    m_workBufferIndex = 0;
    return RGY_ERR_NONE;
}

RGYFrameInfo *NVEncFilterKfm::nextOutputFrame() {
    if (m_frameBuf.empty()) {
        return nullptr;
    }
    auto out = &m_frameBuf[m_outputBufferIndex]->frame;
    m_outputBufferIndex = (m_outputBufferIndex + 1) % (int)m_frameBuf.size();
    return out;
}

RGYFrameInfo *NVEncFilterKfm::nextWorkFrame() {
    if (m_workFrameBuf.empty()) {
        return nullptr;
    }
    auto out = &m_workFrameBuf[m_workBufferIndex]->frame;
    m_workBufferIndex = (m_workBufferIndex + 1) % (int)m_workFrameBuf.size();
    return out;
}

RGY_ERR NVEncFilterKfm::initAnalyzer(const NVEncFilterParamKfm& prm) {
    flushUcfNoiseResultDump();
    RGYKFM::KFMAnalyzeParam analyzeParam;
    analyzeParam.pastCycles = prm.kfm.pastCycles;
    analyzeParam.NGThresh = prm.kfm.thswitch;
    m_analyzer = std::make_unique<RGYKFM::KFMAnalyze>(analyzeParam);
    m_analyzeSourceFrames = 0;
    m_nextAnalyzeCycle = 0;
    m_nextFMCountSubmitCycle = 0;
    m_nextFMCountDumpFrame = 0;
    m_cachedSourceFrames = 0;
    m_timecodeFrameIndex = 0;
    m_hasLastAnalyzeResult = false;
    m_analyzerFinalized = false;
    m_switchTimingDumped = false;
    m_switchDurationPath.clear();
    m_switchTimecodePath.clear();
    m_stageDumpDir.clear();
    m_analyzerOutputResults.clear();
    m_switchSingleFrameN60.clear();
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    m_nextSwitchN60 = 0;
    m_nextSwitchPts = 0;
    m_hasLastSwitchTiming = false;
    m_lastSwitchStart60 = 0;
    m_lastSwitchDuration60 = 0;
    m_lastSwitchStart120 = 0;
    m_lastSwitchIsFrame24 = false;
    m_nextTelecine24Frame = 0;
    m_nextTelecine24Pts = 0;
    m_telecineSuperBufferIndex = 0;
    m_maskBranchBufferIndex = 0;
    m_patchCombeBufferIndex = 0;
    m_stageDumpMaxFrames = 0;
    m_deint60SubmittedSourceFrames = 0;
    m_before60SubmittedSourceFrames = 0;
    m_after60SubmittedSourceFrames = 0;
    m_ucfNoiseCache.clear();
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();

    if (m_fpResult) {
        fclose(m_fpResult);
        m_fpResult = nullptr;
    }
    if (m_fpFMCount) {
        fclose(m_fpFMCount);
        m_fpFMCount = nullptr;
    }
    if (m_fpTimecode) {
        fclose(m_fpTimecode);
        m_fpTimecode = nullptr;
    }
    if (m_fpFrameInfo) {
        fclose(m_fpFrameInfo);
        m_fpFrameInfo = nullptr;
    }
    if (m_fpContainsCombe) {
        fclose(m_fpContainsCombe);
        m_fpContainsCombe = nullptr;
    }
    if (m_fpUcfNoise) {
        fclose(m_fpUcfNoise);
        m_fpUcfNoise = nullptr;
    }
    if (prm.kfm.timecode.length() > 0) {
        if (prm.kfm.mode == VppKfmMode::P24 || prm.kfm.mode == VppKfmMode::VFR) {
            m_switchDurationPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".duration.txt");
        }
        if (_tfopen_s(&m_fpTimecode, prm.kfm.timecode.c_str(), _T("wb")) != 0 || m_fpTimecode == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM timecode dump file \"%s\".\n"), prm.kfm.timecode.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpTimecode, "# timecode format v2\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM timecode dump file \"%s\".\n"), prm.kfm.timecode.c_str());
    }
    if (prm.kfm.debug && prm.kfm.timecode.length() > 0) {
        const auto resultPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".result.dat");
        if (_tfopen_s(&m_fpResult, resultPath.c_str(), _T("wb")) != 0 || m_fpResult == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM result dump file \"%s\".\n"), resultPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM result dump file \"%s\".\n"), resultPath.c_str());

        const auto fmCountPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".fmcount.dat");
        if (_tfopen_s(&m_fpFMCount, fmCountPath.c_str(), _T("wb")) != 0 || m_fpFMCount == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM FMCount dump file \"%s\".\n"), fmCountPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM FMCount dump file \"%s\".\n"), fmCountPath.c_str());

        const auto frameInfoPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".frameinfo.tsv");
        if (_tfopen_s(&m_fpFrameInfo, frameInfoPath.c_str(), _T("w")) != 0 || m_fpFrameInfo == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM frame info dump file \"%s\".\n"), frameInfoPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpFrameInfo, "#stage\tidx\tinputFrameId\ttimestamp\tduration\ttime_ms\tduration_ms\twidth\theight\tcsp\tpicstruct\tflags\tpattern\tis60p\tscore\tcost\treliability\tkfm_n60\tkfm_n24\tkfm_baseType\tkfm_sourceStart\tkfm_numSourceFrames\tkfm_duration60\tkfm_duration120\tkfm_pattern\tkfm_cost\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM frame info dump file \"%s\".\n"), frameInfoPath.c_str());

        const auto containsCombePath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".contains_combe.tsv");
        if (_tfopen_s(&m_fpContainsCombe, containsCombePath.c_str(), _T("w")) != 0 || m_fpContainsCombe == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM contains-combe dump file \"%s\".\n"), containsCombePath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpContainsCombe, "#stage\tidx\tn60\tn24\tbaseType\tsourceStart\tnumSourceFrames\tduration60\tduration120\tcontainsCombeCount\tdurationApplied\tpattern\tcost\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM contains-combe dump file \"%s\".\n"), containsCombePath.c_str());

        if (prm.kfm.ucf) {
            const auto ucfNoisePath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".ucfnoise.tsv");
            if (_tfopen_s(&m_fpUcfNoise, ucfNoisePath.c_str(), _T("w")) != 0 || m_fpUcfNoise == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open KFM UCF noise dump file \"%s\".\n"), ucfNoisePath.c_str());
                return RGY_ERR_FILE_OPEN;
            }
            fprintf(m_fpUcfNoise, "frame\tplane\tnoise0\tnoise1\tnoiseR0\tnoiseR1\tdiff0\tdiff1\tclass0\tclass1\tfield_diff0\tfield_diff1\tdiff0_calc\tdiff1_calc\n");
            AddMessage(RGY_LOG_DEBUG, _T("opened KFM UCF noise dump file \"%s\".\n"), ucfNoisePath.c_str());
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::initStageDumpConfig(const NVEncFilterParamKfm& prm) {
    UNREFERENCED_PARAMETER(prm);
    m_stageDumpDir.clear();
    m_stageDumpMaxFrames = 0;
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    m_stageDumpTargetFrames.clear();

    const char *dumpDir = std::getenv("NVENC_KFM_DUMP_DIR");
    if (dumpDir == nullptr || dumpDir[0] == '\0') {
        return;
    }
    m_stageDumpDir = dumpDir;
    if (!CreateDirectoryRecursive(m_stageDumpDir.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create KFM stage dump directory \"%s\".\n"),
            char_to_tstring(m_stageDumpDir).c_str());
        m_stageDumpDir.clear();
        return;
    }

    const char *maxFrames = std::getenv("NVENC_KFM_DUMP_MAX_FRAMES");
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const auto value = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames) {
            m_stageDumpMaxFrames = (int)std::min<long>(std::max<long>(0, value), std::numeric_limits<int>::max());
        }
    }
    const char *frameList = std::getenv("NVENC_KFM_DUMP_FRAME_LIST");
    if (frameList != nullptr && frameList[0] != '\0') {
        const char *p = frameList;
        while (*p != '\0') {
            char *endptr = nullptr;
            const auto value = std::strtol(p, &endptr, 10);
            if (endptr != p && value >= 0 && value <= std::numeric_limits<int>::max()) {
                m_stageDumpTargetFrames.insert((int)value);
            }
            p = endptr;
            while (*p == ',' || *p == ';' || *p == ':' || *p == ' ' || *p == '\t') {
                p++;
            }
            if (endptr == p && *p != '\0') {
                p++;
            }
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("enabled KFM stage dump directory \"%s\", max frames %d, target frames %zu.\n"),
        char_to_tstring(m_stageDumpDir).c_str(), m_stageDumpMaxFrames, m_stageDumpTargetFrames.size());
}

bool NVEncFilterKfm::stageDumpRequested(int frame24Index) const {
    return !m_stageDumpDir.empty()
        && ((m_stageDumpMaxFrames > 0 && frame24Index < m_stageDumpMaxFrames)
            || m_stageDumpTargetFrames.find(frame24Index) != m_stageDumpTargetFrames.end());
}

RGY_ERR NVEncFilterKfm::padSourceFrame(RGYFrameInfo *pPaddedFrame, const RGYFrameInfo *pSourceFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pPaddedFrame || !pSourceFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    if (pPaddedFrame->csp != pSourceFrame->csp || pPaddedFrame->width != pSourceFrame->width) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto sts = kfmWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    const auto planes = RGY_CSP_PLANES[pPaddedFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pPaddedFrame, (RGY_PLANE)iplane);
        const auto src = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const int vpad = (dst.height - src.height) >> 1;
        if (dst.width != src.width || dst.height != src.height + vpad * 2 || vpad <= 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM padded source plane size (plane %d, src %dx%d, dst %dx%d).\n"),
                iplane, src.width, src.height, dst.width, dst.height);
            return RGY_ERR_INVALID_PARAM;
        }
        sts = run_kfm_pad_plane(&dst, &src, vpad, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_pad (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
    }
    sts = kfmRecordEvent(stream, event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    copyFramePropWithoutRes(pPaddedFrame, pSourceFrame);
    pPaddedFrame->picstruct = RGY_PICSTRUCT_FRAME;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::cacheSourceFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    KfmCachedSource entry;
    entry.sourceIndex = m_cachedSourceFrames++;
    entry.inputFrameId = frame->inputFrameId;
    entry.timestamp = frame->timestamp;
    entry.frame = acquireKfmFrame(*frame, _T("source cache"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto sts = kfmWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = copyFrameAsync(&entry.frame->frame, frame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to cache KFM source frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = kfmRecordEvent(stream, &entry.event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    copyFramePropWithoutRes(&entry.frame->frame, frame);

    auto paddedFrameInfo = *frame;
    paddedFrameInfo.height += KFM_SOURCE_VPAD * 2;
    entry.paddedFrame = acquireKfmFrame(paddedFrameInfo, _T("padded source cache"));
    if (!entry.paddedFrame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto padWaitEvents = wait_events;
    if (entry.event() != nullptr) {
        padWaitEvents.push_back(entry.event);
    }
    sts = padSourceFrame(&entry.paddedFrame->frame, &entry.frame->frame, stream, padWaitEvents, &entry.paddedEvent);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to pad KFM source frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    writeFrameInfoDump("source-pad", &entry.paddedFrame->frame);

    m_sourceCache.push_back(std::move(entry));
    const auto cacheLimit = sourceCacheLimit();
    const auto trimFloor = sourceCacheTrimFloor();
    while (m_sourceCache.size() > cacheLimit && m_sourceCache.front().sourceIndex < trimFloor) {
        m_sourceCache.pop_front();
    }
    sts = analyzeAvailableSource(false, stream);
    writeFrameInfoDump("source", frame);
    return sts;
}

RGY_ERR NVEncFilterKfm::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(pParam);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    m_param = pParam;
    m_pLog = pPrintMes;
    auto sts = initAnalyzer(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    initStageDumpConfig(*prm);
    AddMessage(RGY_LOG_ERROR, _T("KFM CUDA filter body is not wired yet.\n"));
    return RGY_ERR_UNSUPPORTED;
}

int NVEncFilterKfm::requiredOutputFrames() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm) {
        return 0;
    }
    switch (prm->kfm.mode) {
    case VppKfmMode::VFR:
    case VppKfmMode::P24:
        return 16;
    default:
        return 0;
    }
}

size_t NVEncFilterKfm::sourceCacheLimit() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm) {
        return 16;
    }
    if (prm->kfm.timing == VppKfmTiming::Strict) {
        return std::numeric_limits<size_t>::max();
    }
    if (prm->kfm.timing == VppKfmTiming::RealtimePlus) {
        return std::max<size_t>(16, static_cast<size_t>(std::max(0, prm->kfm.pastCycles)) * 5 + KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN);
    }
    return 16;
}

size_t NVEncFilterKfm::deint60CacheLimit() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || !m_deint60Rtgmc) {
        return 32;
    }
    if (prm->kfm.timing == VppKfmTiming::Strict) {
        return std::numeric_limits<size_t>::max();
    }
    if (prm->kfm.timing == VppKfmTiming::RealtimePlus) {
        return std::max<size_t>(32, static_cast<size_t>(std::max(0, prm->kfm.pastCycles)) * 10 + KFM_REALTIMEPLUS_DEINT60_CACHE_MARGIN);
    }
    return 32;
}

int NVEncFilterKfm::sourceCacheTrimFloor() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || prm->kfm.mode != VppKfmMode::VFR || m_nextSwitchN60 <= 0) {
        return 0;
    }
    auto trimFloor = std::max(0, (m_nextSwitchN60 >> 1) - KFM_VFR_SOURCE_TRIM_LOOKBEHIND);
    for (const auto& pending : m_pendingUcfNoiseResults) {
        if (pending.sourceIndex >= 0) {
            trimFloor = std::min(trimFloor, pending.sourceIndex);
        }
    }
    if (m_ucfNoiseCache.size() >= 3) {
        for (const auto& noise : m_ucfNoiseCache) {
            if (noise.fieldIndex >= 0) {
                trimFloor = std::min(trimFloor, noise.fieldIndex >> 1);
            }
        }
    }
    return trimFloor;
}

int NVEncFilterKfm::deint60CacheTrimFloor() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || prm->kfm.mode != VppKfmMode::VFR || m_nextSwitchN60 <= 0) {
        return 0;
    }
    return std::max(0, m_nextSwitchN60 - KFM_VFR_DEINT60_TRIM_LOOKBEHIND);
}

const RGYFrameInfo *NVEncFilterKfm::findDeint60Frame(int n60, std::vector<RGYCudaEvent> *wait_events) const {
    const auto *entry = findCachedDeint60Frame(m_deint60Cache, n60, wait_events);
    return entry && entry->frame ? &entry->frame->frame : nullptr;
}

const NVEncFilterKfm::KfmCachedDeint60 *NVEncFilterKfm::findCachedDeint60Frame(const std::deque<KfmCachedDeint60>& cache, int n60, std::vector<RGYCudaEvent> *wait_events) const {
    for (auto it = cache.rbegin(); it != cache.rend(); ++it) {
        if (it->n60 == n60) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &(*it);
        }
    }
    return nullptr;
}

const NVEncFilterKfm::KfmUcfNoiseDumpRecord *NVEncFilterKfm::findUcfNoiseResult(int sourceIndex) const {
    for (auto it = m_ucfNoiseResultCache.rbegin(); it != m_ucfNoiseResultCache.rend(); ++it) {
        if (it->sourceIndex == sourceIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

const RGYFrameInfo *NVEncFilterKfm::findSourceFrame(const RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events) {
    if (!frame) {
        return nullptr;
    }
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->inputFrameId == frame->inputFrameId && it->timestamp == frame->timestamp) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &it->frame->frame;
        }
    }
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->inputFrameId == frame->inputFrameId) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &it->frame->frame;
        }
    }
    return nullptr;
}

const NVEncFilterKfm::KfmCachedSource *NVEncFilterKfm::findSourceByIndex(int sourceIndex) const {
    if (m_sourceCache.empty()) {
        return nullptr;
    }
    const int clampedIndex = clamp(sourceIndex, m_sourceCache.front().sourceIndex, m_sourceCache.back().sourceIndex);
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->sourceIndex == clampedIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

const NVEncFilterKfm::KfmCachedSource *NVEncFilterKfm::findSourceByIndexExact(int sourceIndex) const {
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->sourceIndex == sourceIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

RGY_ERR NVEncFilterKfm::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, int frame24Index, cudaStream_t stream) {
    if (!stage || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    const std::string stageKey(stage);
    auto& frameCount = m_stageDumpFrameCounts[stageKey];
    const int dumpIndex = (frame24Index >= 0) ? frame24Index : frameCount;
    if (!stageDumpRequested(dumpIndex)) {
        return RGY_ERR_NONE;
    }
    auto *dumpedFrameIndices = (frame24Index >= 0) ? &m_stageDumpFrameIndices[stageKey] : nullptr;
    if (dumpedFrameIndices && dumpedFrameIndices->find(frame24Index) != dumpedFrameIndices->end()) {
        return RGY_ERR_NONE;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    const bool isYuv420 = RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_YUV420;
    const bool isMono = RGY_CSP_PLANES[frame->csp] == 1 || RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_MONOCHROME;
    if (bitdepth > 8 || (!isYuv420 && !isMono)) {
        if (frameCount == 0) {
            AddMessage(RGY_LOG_WARN, _T("KFM stage dump skipped for unsupported csp %s at stage %s.\n"),
                RGY_CSP_NAMES[frame->csp], char_to_tstring(stage).c_str());
        }
        return RGY_ERR_NONE;
    }

    const auto planeY = getPlane(frame, RGY_PLANE_Y);
    if (!planeY.ptr[0] || planeY.width <= 0 || planeY.height <= 0) {
        return RGY_ERR_NONE;
    }
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostPlaneY(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostPlaneY.ptr[0] = hostY.data();
    hostPlaneY.pitch[0] = planeY.width;
    auto sts = copyPlaneAsync(&hostPlaneY, &planeY, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump Y plane (%s): %s.\n"),
            char_to_tstring(stage).c_str(), get_err_mes(sts));
        return sts;
    }

    const int chromaWidth = (hostPlaneY.width + 1) >> 1;
    const int chromaHeight = (hostPlaneY.height + 1) >> 1;
    std::vector<uint8_t> hostU((size_t)chromaWidth * chromaHeight, 128);
    std::vector<uint8_t> hostV((size_t)chromaWidth * chromaHeight, 128);
    std::vector<uint8_t> hostUV;
    if (isYuv420 && !isMono) {
        if (kfmCspHasInterleavedUV(frame->csp)) {
            const auto planeUV = getPlane(frame, RGY_PLANE_U);
            if (planeUV.ptr[0]) {
                hostUV.resize((size_t)planeUV.width * planeUV.height);
                RGYFrameInfo hostPlaneUV(planeUV.width, planeUV.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                hostPlaneUV.ptr[0] = hostUV.data();
                hostPlaneUV.pitch[0] = planeUV.width;
                sts = copyPlaneAsync(&hostPlaneUV, &planeUV, stream);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump UV plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
            }
        } else {
            const auto planeU = getPlane(frame, RGY_PLANE_U);
            const auto planeV = getPlane(frame, RGY_PLANE_V);
            if (planeU.ptr[0] && planeV.ptr[0] && planeU.width == chromaWidth && planeU.height == chromaHeight && planeV.width == chromaWidth && planeV.height == chromaHeight) {
                RGYFrameInfo hostPlaneU(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                RGYFrameInfo hostPlaneV(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                hostPlaneU.ptr[0] = hostU.data();
                hostPlaneV.ptr[0] = hostV.data();
                hostPlaneU.pitch[0] = chromaWidth;
                hostPlaneV.pitch[0] = chromaWidth;
                sts = copyPlaneAsync(&hostPlaneU, &planeU, stream);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump U plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                sts = copyPlaneAsync(&hostPlaneV, &planeV, stream);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump V plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
            }
        }
    }

    sts = err_to_rgy(cudaStreamSynchronize(stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM stage dump readback (%s): %s.\n"),
            char_to_tstring(stage).c_str(), get_err_mes(sts));
        return sts;
    }
    if (!hostUV.empty()) {
        for (int y = 0; y < chromaHeight; y++) {
            const auto *src = hostUV.data() + (size_t)y * chromaWidth * 2;
            auto *dstU = hostU.data() + (size_t)y * chromaWidth;
            auto *dstV = hostV.data() + (size_t)y * chromaWidth;
            for (int x = 0; x < chromaWidth; x++) {
                dstU[x] = src[x * 2 + 0];
                dstV[x] = src[x * 2 + 1];
            }
        }
    }

    const auto path = PathCombineS(m_stageDumpDir, kfmStageDumpName(stage));
    std::ofstream dump(path, std::ios::out | std::ios::binary | (frameCount == 0 ? std::ios::trunc : std::ios::app));
    if (!dump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open KFM stage dump \"%s\".\n"), char_to_tstring(path).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    if (frameCount == 0) {
        dump << "YUV4MPEG2 W" << hostPlaneY.width << " H" << hostPlaneY.height << " F30000:1001 Ip A0:0 "
             << (isMono ? "Cmono" : "C420jpeg") << "\n";
    }
    dump << "FRAME\n";
    for (int y = 0; y < hostPlaneY.height; y++) {
        dump.write(reinterpret_cast<const char *>(hostPlaneY.ptr[0] + (size_t)y * hostPlaneY.pitch[0]), hostPlaneY.width);
    }
    if (!isMono) {
        dump.write(reinterpret_cast<const char *>(hostU.data()), hostU.size());
        dump.write(reinterpret_cast<const char *>(hostV.data()), hostV.size());
    }
    if (!dump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write KFM stage dump \"%s\".\n"), char_to_tstring(path).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    frameCount++;
    if (dumpedFrameIndices) {
        dumpedFrameIndices->insert(frame24Index);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::finalizeAnalyzerResults(VppKfmTiming timing) {
    if (!m_analyzer || m_analyzerFinalized) {
        return;
    }
    const auto resultCount = static_cast<size_t>(m_nextAnalyzeCycle);
    m_analyzer->analyzeTrailingCycles(m_analyzer->param().cycleRange);
    if (timing == VppKfmTiming::Strict) {
        writeAnalyzerResultsFinal(resultCount, true);
    } else if (timing == VppKfmTiming::RealtimePlus) {
        appendAnalyzerResults(resultCount, true, true);
    } else {
        writeAnalyzerResultsFinal(resultCount, false);
    }
    m_analyzerFinalized = true;
}

std::vector<RGYKFM::KFMResult> NVEncFilterKfm::analyzerResultsSnapshot(bool mark60p) const {
    std::vector<RGYKFM::KFMResult> results;
    if (!m_analyzer) {
        return results;
    }
    results = m_analyzer->results();
    if (!mark60p || results.empty()) {
        return results;
    }
    for (auto& result : results) {
        result.is60p = false;
    }
    const auto& param = m_analyzer->param();
    bool is60p = true;
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        auto& cur = results[i];
        if (is60p) {
            if (cur.cost < param.th24) {
                if (cur.reliability < param.rel24) {
                    is60p = false;
                }
            } else {
                cur.is60p = true;
            }
        } else {
            if (cur.cost >= param.th60) {
                is60p = true;
                for (int t = i; t >= 0; --t) {
                    auto& prev = results[t];
                    if (prev.cost < param.th24) {
                        if (prev.reliability < param.rel24) {
                            break;
                        }
                    } else {
                        prev.is60p = true;
                    }
                }
            }
        }
    }
    return results;
}

void NVEncFilterKfm::writeFMCountDump(const std::array<RGYKFM::FMCount, 18>& counts, int cycle) {
    if (!m_fpFMCount) {
        return;
    }
    const int firstCountFrame = cycle * 5 - 2;
    while (m_nextFMCountDumpFrame < m_cachedSourceFrames) {
        const int windowIndex = m_nextFMCountDumpFrame - firstCountFrame;
        if (windowIndex < 0) {
            m_nextFMCountDumpFrame++;
            continue;
        }
        if (windowIndex >= KFM_FMCOUNT_PAIRS) {
            break;
        }
        fwrite(&counts[windowIndex * 2], sizeof(RGYKFM::FMCount), 2, m_fpFMCount);
        m_nextFMCountDumpFrame++;
    }
    fflush(m_fpFMCount);
}

void NVEncFilterKfm::writeAnalyzerResult(const RGYKFM::KFMResult& result, bool dump) {
    m_lastAnalyzeResult = result;
    m_hasLastAnalyzeResult = true;
    m_analyzerOutputResults.push_back(result);
    if (dump && m_fpResult) {
        fwrite(&result, sizeof(result), 1, m_fpResult);
        fflush(m_fpResult);
    }
}

void NVEncFilterKfm::appendAnalyzerResults(size_t resultCount, bool dump, bool mark60p) {
    if (!m_analyzer) {
        return;
    }
    const auto results = analyzerResultsSnapshot(mark60p);
    resultCount = std::min(resultCount, results.size());
    if (resultCount <= m_analyzerOutputResults.size()) {
        return;
    }
    while (m_analyzerOutputResults.size() < resultCount) {
        const auto& result = results[m_analyzerOutputResults.size()];
        m_analyzerOutputResults.push_back(result);
        m_lastAnalyzeResult = result;
        m_hasLastAnalyzeResult = true;
        if (dump && m_fpResult) {
            fwrite(&result, sizeof(result), 1, m_fpResult);
        }
    }
    if (dump && m_fpResult) {
        fflush(m_fpResult);
    }
}

void NVEncFilterKfm::writeAnalyzerResultsFinal(size_t resultCount, bool mark60p) {
    if (!m_analyzer) {
        return;
    }
    const auto results = analyzerResultsSnapshot(mark60p);
    resultCount = std::min(resultCount, results.size());
    if (resultCount == 0) {
        return;
    }
    if (m_fpResult) {
        if (fseek(m_fpResult, 0, SEEK_SET) != 0) {
            AddMessage(RGY_LOG_WARN, _T("failed to seek KFM result dump file.\n"));
        } else {
            fwrite(results.data(), sizeof(results[0]), resultCount, m_fpResult);
            fflush(m_fpResult);
        }
    }
    m_analyzerOutputResults.assign(results.begin(), results.begin() + resultCount);
    m_lastAnalyzeResult = results[resultCount - 1];
    m_hasLastAnalyzeResult = true;
}

void NVEncFilterKfm::writeFrameTimecode(const RGYFrameInfo *frame) {
    if (!m_fpTimecode || !frame || frame->timestamp < 0) {
        return;
    }
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || !prm->timebase.is_valid()) {
        return;
    }
    const auto timeMs = (double)frame->timestamp * prm->timebase.qdouble() * 1000.0;
    if (prm->kfm.mode == VppKfmMode::VFR) {
        const auto timeMsInt = (int64_t)std::floor(timeMs + 0.5 - 1.0e-9);
        fprintf(m_fpTimecode, "%lld\n", (lls)timeMsInt);
    } else {
        fprintf(m_fpTimecode, "%.6lf\n", timeMs);
    }
    fflush(m_fpTimecode);
}

std::vector<NVEncFilterKfm::KfmSwitchTiming> NVEncFilterKfm::deriveSwitchTimings(int total60) const {
    std::vector<KfmSwitchTiming> timings;
    if (!m_analyzer || m_analyzerOutputResults.empty() || total60 <= 0) {
        return timings;
    }
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    const auto timingMode = prm ? prm->kfm.timing : VppKfmTiming::Realtime;
    const auto thswitch = prm ? prm->kfm.thswitch : 0.5f;
    const auto resultAt = [&](int cycle) -> const RGYKFM::KFMResult& {
        return m_analyzerOutputResults[clamp(cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
    };
    const auto frameInfoAt = [&](int n60, const RGYKFM::KFMResult& fm) {
        KfmSwitchTiming info;
        info.start60 = n60;
        info.start120 = n60 * 2;
        info.duration60 = 1;
        info.duration120 = 2;
        info.sourceStart = n60;
        info.sourceIndex = n60;
        info.numSourceFrames = 1;
        info.baseType = KFM_FRAME_60;
        info.frame24Index = -1;
        info.isFrame24 = false;
        info.isFrame60 = true;

        const bool force60 = (thswitch >= 0.0f)
            && ((timingMode == VppKfmTiming::Realtime && fm.cost > thswitch)
                || (timingMode != VppKfmTiming::Realtime && fm.is60p != 0));
        if (force60) {
            return info;
        }
        if (RGYKFM::PulldownPatterns::is30p(fm.pattern)) {
            info.baseType = KFM_FRAME_30;
            info.sourceStart = n60 >> 1;
            info.sourceIndex = info.sourceStart;
            info.isFrame60 = false;
            return info;
        }
        const auto f = m_analyzer->patterns().getFrame60(fm.pattern, n60);
        int frameIndex = f.frameIndex + f.fieldShift;
        int n24 = f.cycleIndex * 4 + frameIndex;
        if (frameIndex < 0) {
            n24 = f.cycleIndex * 4 - 1;
            if (n24 < 0) {
                info.baseType = KFM_FRAME_60;
                info.sourceStart = n60 >> 1;
                info.sourceIndex = info.sourceStart;
                info.isFrame60 = true;
                return info;
            }
        } else if (frameIndex >= 4) {
            const auto next = m_analyzer->patterns().getFrame24(resultAt(n60 / 10 + 1).pattern, 0);
            n24 = f.cycleIndex * 4 + (next.fieldStartIndex > 0 ? 3 : 4);
        }
        info.baseType = KFM_FRAME_24;
        info.frame24Index = n24;
        info.sourceStart = n24;
        info.sourceIndex = n24;
        info.isFrame24 = false;
        info.isFrame60 = false;
        return info;
    };

    int current = 0;
    while (current < total60) {
        auto info = frameInfoAt(current, resultAt(current / 10));
        const bool forceSingle = (info.baseType == KFM_FRAME_24 || info.baseType == KFM_FRAME_30) && isSwitchSingleFrameN60(current);
        const int maxDuration = forceSingle ? 1 : info.baseType == KFM_FRAME_24 ? 4 : info.baseType == KFM_FRAME_30 ? 2 : 1;
        int duration = maxDuration;
        for (int i = 1; i < maxDuration; i++) {
            if (current + i >= total60) {
                duration = i;
                info.isFrame24 = info.baseType == KFM_FRAME_24;
                break;
            }
            const auto next = frameInfoAt(current + i, resultAt((current + i) / 10));
            if (next.baseType != info.baseType || next.sourceIndex != info.sourceIndex) {
                duration = i;
                info.isFrame24 = info.baseType == KFM_FRAME_24;
                break;
            }
        }
        info.duration60 = duration;
        info.duration120 = duration * 2;
        info.numSourceFrames = std::max(1, divCeil(duration, 2));
        timings.push_back(info);
        current += duration;
    }

    if (prm && prm->kfm.is120) {
        for (size_t i = 1; i < timings.size(); ++i) {
            if (timings[i - 1].isFrame24 && timings[i].isFrame24
                && timings[i - 1].duration60 >= 2 && timings[i].duration60 >= 2
                && timings[i - 1].duration60 + timings[i].duration60 == 5) {
                timings[i].start120 = timings[i - 1].start120 + 5;
            }
        }
        for (size_t i = 0; i + 1 < timings.size(); ++i) {
            const int duration120 = timings[i + 1].start120 - timings[i].start120;
            if (duration120 > 0) {
                timings[i].duration120 = duration120;
            }
        }
    }
    return timings;
}

int64_t NVEncFilterKfm::sourceFrameDuration(const KfmCachedSource *source) const {
    if (!source || !source->frame) {
        return 1;
    }
    const auto duration = source->frame->frame.duration;
    if (duration > 0) {
        return duration;
    }
    const auto *next = findSourceByIndexExact(source->sourceIndex + 1);
    if (next && next->timestamp > source->timestamp) {
        return next->timestamp - source->timestamp;
    }
    const auto *prev = findSourceByIndexExact(source->sourceIndex - 1);
    if (prev && source->timestamp > prev->timestamp) {
        return source->timestamp - prev->timestamp;
    }
    return 1;
}

bool NVEncFilterKfm::isSwitchSingleFrameN60(int n60) const {
    return std::find(m_switchSingleFrameN60.begin(), m_switchSingleFrameN60.end(), n60) != m_switchSingleFrameN60.end();
}

void NVEncFilterKfm::markSwitchSingleFrameN60Range(int start60, int duration60) {
    for (int i = 0; i < duration60; i++) {
        const int n60 = start60 + i;
        if (!isSwitchSingleFrameN60(n60)) {
            m_switchSingleFrameN60.push_back(n60);
        }
    }
}

bool NVEncFilterKfm::switchSingleFrameDurationEnabled() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    return prm && prm->kfm.thswitch >= 0.0f && !kfmDisableCCDuration();
}

void NVEncFilterKfm::writeSwitchTimingDump() {
    if (m_switchTimingDumped || m_switchDurationPath.empty()) {
        return;
    }
    const auto timings = deriveSwitchTimings(m_cachedSourceFrames * 2);
    FILE *fp = nullptr;
    if (_tfopen_s(&fp, m_switchDurationPath.c_str(), _T("wb")) != 0 || fp == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("failed to open KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
        return;
    }
    for (const auto& timing : timings) {
        fprintf(fp, "%d\n", timing.duration60);
    }
    fclose(fp);
    m_switchTimingDumped = true;
    AddMessage(RGY_LOG_DEBUG, _T("wrote KFM switch duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
}

void NVEncFilterKfm::writeTelecine24DurationDump() {
    if (m_switchTimingDumped || m_switchDurationPath.empty() || !m_analyzer) {
        return;
    }
    FILE *fp = nullptr;
    if (_tfopen_s(&fp, m_switchDurationPath.c_str(), _T("wb")) != 0 || fp == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("failed to open KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
        return;
    }
    for (int frame24Index = 0; frame24Index < m_nextTelecine24Frame; frame24Index++) {
        if (frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
            break;
        }
        const auto& result = m_analyzerOutputResults[frame24Index / 4];
        try {
            const auto info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
            const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
            const int totalFields = m_cachedSourceFrames * 2;
            const int availableFields = (totalFields > firstField) ? std::min(info.numFields, totalFields - firstField) : info.numFields;
            fprintf(fp, "%d\n", std::max(1, availableFields));
        } catch (...) {
            fprintf(fp, "2\n");
        }
    }
    fclose(fp);
    m_switchTimingDumped = true;
    AddMessage(RGY_LOG_DEBUG, _T("wrote KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
}

void NVEncFilterKfm::writeFrameInfoDump(const char *stage, const RGYFrameInfo *frame, const RGYKFM::KFMResult *result) {
    if (!m_fpFrameInfo || !stage || !frame) {
        return;
    }
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    const auto timebase = (prm && prm->timebase.is_valid()) ? prm->timebase.qdouble() : 0.0;
    const auto timeMs = (frame->timestamp >= 0 && timebase > 0.0) ? (double)frame->timestamp * timebase * 1000.0 : -1.0;
    const auto durationMs = (frame->duration >= 0 && timebase > 0.0) ? (double)frame->duration * timebase * 1000.0 : -1.0;
    const auto *r = result ? result : (m_hasLastAnalyzeResult ? &m_lastAnalyzeResult : nullptr);
    fprintf(m_fpFrameInfo, "%s\t%d\t%d\t%lld\t%lld\t%.6lf\t%.6lf\t%d\t%d\t%d\t%d\t%d",
        stage,
        m_timecodeFrameIndex,
        frame->inputFrameId,
        (long long)frame->timestamp,
        (long long)frame->duration,
        timeMs,
        durationMs,
        frame->width,
        frame->height,
        (int)frame->csp,
        (int)frame->picstruct,
        (int)frame->flags);
    if (r) {
        fprintf(m_fpFrameInfo, "\t%d\t%d\t%.6f\t%.6f\t%.6f",
            r->pattern, r->is60p, r->score, r->cost, r->reliability);
    } else {
        fprintf(m_fpFrameInfo, "\t-1\t0\t0.000000\t0.000000\t0.000000");
    }
    auto switchData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data && data->dataType() == RGY_FRAME_DATA_KFM_SWITCH;
    });
    if (switchData != frame->dataList.end()) {
        const auto kfm = std::dynamic_pointer_cast<RGYFrameDataKfmSwitch>(*switchData);
        fprintf(m_fpFrameInfo, "\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
            kfm->n60(), kfm->n24(), kfm->baseType(), kfm->sourceStart(), kfm->numSourceFrames(),
            kfm->duration60(), kfm->duration120(), kfm->pattern(), kfm->cost());
    } else {
        fprintf(m_fpFrameInfo, "\t-1\t-1\t-1\t-1\t0\t0\t0\t-1\t0.000000\n");
    }
    fflush(m_fpFrameInfo);
}

void NVEncFilterKfm::writeContainsCombeDump(const char *stage, const KfmSwitchTiming& timing, uint32_t containsCombeCount, bool durationApplied, const RGYKFM::KFMResult *result) {
    if (!m_fpContainsCombe || !stage) {
        return;
    }
    fprintf(m_fpContainsCombe, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%.6f\n",
        stage,
        m_timecodeFrameIndex,
        timing.start60,
        timing.frame24Index,
        timing.baseType,
        timing.sourceStart,
        timing.numSourceFrames,
        timing.duration60,
        timing.duration120,
        containsCombeCount,
        durationApplied ? 1 : 0,
        result ? result->pattern : -1,
        result ? result->cost : 0.0f);
    fflush(m_fpContainsCombe);
}

void NVEncFilterKfm::attachSwitchFrameData(RGYFrameInfo *frame, const KfmSwitchTiming& timing, const RGYKFM::KFMResult *result) const {
    if (!frame) {
        return;
    }
    frame->dataList.erase(std::remove_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data && data->dataType() == RGY_FRAME_DATA_KFM_SWITCH;
    }), frame->dataList.end());
    frame->dataList.push_back(std::make_shared<RGYFrameDataKfmSwitch>(
        timing.start60, timing.frame24Index, timing.baseType, timing.sourceStart, timing.numSourceFrames,
        timing.duration60, timing.duration120, result ? result->pattern : -1, result ? result->cost : 0.0f));
}

int NVEncFilterKfm::telecine24FrameCount(bool drain) const {
    if (m_analyzerOutputResults.empty()) {
        return 0;
    }
    const int analyzedFrames = (int)m_analyzerOutputResults.size() * 4;
    const int totalFields = m_cachedSourceFrames * 2;
    int readyFrames = 0;
    for (int frame24Index = 0; frame24Index < analyzedFrames; frame24Index++) {
        const auto& result = m_analyzerOutputResults[frame24Index / 4];
        RGYKFM::Frame24Info info;
        try {
            info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
        } catch (...) {
            break;
        }
        const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
        if (totalFields > 0 && firstField >= totalFields) {
            break;
        }
        if (drain) {
            readyFrames++;
            continue;
        }
        const int lastField = firstField + info.numFields - 1;
        const int firstSource = (firstField & ~1) >> 1;
        const int lastSource = std::max(firstSource, lastField >> 1);
        const auto *first = firstSource < 0 ? findSourceByIndex(firstSource) : findSourceByIndexExact(firstSource);
        const auto *last = lastSource < 0 ? findSourceByIndex(lastSource) : findSourceByIndexExact(lastSource);
        if (!first || !last) {
            break;
        }
        readyFrames++;
    }
    return readyFrames;
}

RGY_ERR NVEncFilterKfm::submitFMCounts(int cycle, bool drain, cudaStream_t stream) {
    if (std::find_if(m_pendingFMCounts.begin(), m_pendingFMCounts.end(), [cycle](const KfmPendingFMCount& pending) {
        return pending.cycle == cycle;
    }) != m_pendingFMCounts.end()) {
        return RGY_ERR_NONE;
    }
    const int firstSourceIndex = cycle * 5 - 3;
    const int lastSourceIndex = firstSourceIndex + KFM_FMCOUNT_SOURCE_FRAMES - 1;
    if (!drain && m_cachedSourceFrames <= lastSourceIndex) {
        return RGY_ERR_MORE_DATA;
    }
    if (m_cachedSourceFrames <= 0 || m_sourceCache.empty()) {
        return RGY_ERR_MORE_DATA;
    }

    std::array<const KfmCachedSource *, KFM_FMCOUNT_SOURCE_FRAMES> src = {};
    for (int i = 0; i < KFM_FMCOUNT_SOURCE_FRAMES; i++) {
        src[i] = findSourceByIndex(firstSourceIndex + i);
        if (!src[i] || !src[i]->paddedFrame || !src[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    const size_t countBytes = sizeof(RGYKFM::FMCount) * 2;
    KfmPendingFMCount pending;
    pending.cycle = cycle;
    pending.pairCounts.resize(KFM_FMCOUNT_PAIRS);
    pending.pairEvents.resize(KFM_FMCOUNT_PAIRS);
    for (auto& pairCount : pending.pairCounts) {
        pairCount = std::make_unique<CUMemBufPair>(countBytes);
        auto sts = pairCount->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM FMCount buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
        auto& fmCountBuf = pending.pairCounts[pair];
        auto sts = run_kfm_init_fmcount(reinterpret_cast<RGYKFM::FMCount *>(fmCountBuf->ptrDevice), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM FMCount buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }

        const auto csp = src[pair + 1]->frame->frame.csp;
        const bool interleavedUV = kfmCspHasInterleavedUV(csp);
        const int targetPlanes = (RGY_CSP_PLANES[csp] >= 3) ? 3 : (interleavedUV ? 3 : 1);
        const int countParity = kfmFrameParity(&src[3]->frame->frame);
        for (int iplane = 0; iplane < targetPlanes; iplane++) {
            const bool interleavedChroma = interleavedUV && iplane > 0;
            const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
            const int pixelStep = interleavedChroma ? 2 : 1;
            const int pixelOffset = interleavedChroma ? iplane - 1 : 0;
            const auto prevSrc0 = getPlane(&src[pair + 0]->frame->frame, plane);
            const auto prevSrc1 = getPlane(&src[pair + 1]->frame->frame, plane);
            const auto curSrc0 = prevSrc1;
            const auto curSrc1 = getPlane(&src[pair + 2]->frame->frame, plane);
            if (!prevSrc0.ptr[0] || !prevSrc1.ptr[0] || !curSrc1.ptr[0]) {
                continue;
            }
            if (prevSrc0.width != prevSrc1.width || prevSrc0.height != prevSrc1.height
                || prevSrc0.width != curSrc1.width || prevSrc0.height != curSrc1.height) {
                return RGY_ERR_INVALID_CALL;
            }

            const int gridWidth = (prevSrc0.width / pixelStep) >> 2;
            const int gridHeight = prevSrc0.height >> 2;
            if (gridWidth < 2 || gridHeight < 2) {
                continue;
            }

            std::vector<RGYCudaEvent> countWaitEvents;
            if (src[pair + 0]->paddedEvent() != nullptr) {
                countWaitEvents.push_back(src[pair + 0]->paddedEvent);
            }
            if (src[pair + 1]->paddedEvent() != nullptr) {
                countWaitEvents.push_back(src[pair + 1]->paddedEvent);
            }
            if (src[pair + 2]->paddedEvent() != nullptr) {
                countWaitEvents.push_back(src[pair + 2]->paddedEvent);
            }
            sts = kfmWaitEvents(stream, countWaitEvents);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            const bool chroma = iplane > 0;
            const int threshMove = chroma ? KFM_THRESH_MOVE_C : KFM_THRESH_MOVE_Y;
            const int threshShima = chroma ? KFM_THRESH_SHIMA_C : KFM_THRESH_SHIMA_Y;
            const int cleanThresh = chroma ? KFM_CLEAN_THRESH_C : KFM_CLEAN_THRESH_Y;
            sts = run_kfm_analyze_count_cmflags_clean(
                reinterpret_cast<RGYKFM::FMCount *>(fmCountBuf->ptrDevice),
                &prevSrc0, &prevSrc1, &curSrc0, &curSrc1,
                gridWidth - 1, gridHeight - 1,
                kfmFrameParity(&src[pair + 0]->frame->frame),
                kfmFrameParity(&src[pair + 1]->frame->frame),
                countParity,
                pixelStep, pixelOffset,
                threshMove, threshShima, threshShima * 3, cleanThresh,
                stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze_count_cmflags_clean (pair %d, plane %d): %s.\n"), pair, iplane, get_err_mes(sts));
                return sts;
            }
        }

        sts = err_to_rgy(cudaMemcpyAsync(fmCountBuf->ptrHost, fmCountBuf->ptrDevice, countBytes, cudaMemcpyDeviceToHost, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM FMCount buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &pending.pairEvents[pair]);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    m_pendingFMCounts.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::readbackFMCounts(std::array<RGYKFM::FMCount, 18>& counts, int cycle, bool drain, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(stream);
    if (m_pendingFMCounts.empty() || m_pendingFMCounts.front().cycle != cycle) {
        return RGY_ERR_MORE_DATA;
    }
    if (!drain && m_nextFMCountSubmitCycle - cycle <= KFM_FMCOUNT_ASYNC_DELAY_CYCLES) {
        return RGY_ERR_MORE_DATA;
    }

    auto& pending = m_pendingFMCounts.front();
    counts = {};
    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
        auto& fmCountBuf = pending.pairCounts[pair];
        if (!fmCountBuf) {
            AddMessage(RGY_LOG_ERROR, _T("KFM FMCount pending buffer is missing.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (pending.pairEvents[pair]() != nullptr) {
            const auto waitSts = err_to_rgy(cudaEventSynchronize(pending.pairEvents[pair]()));
            if (waitSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM FMCount copy event: %s.\n"), get_err_mes(waitSts));
                return waitSts;
            }
        }
        const auto *gpuCounts = reinterpret_cast<const RGYKFM::FMCount *>(fmCountBuf->ptrHost);
        if (!gpuCounts) {
            AddMessage(RGY_LOG_ERROR, _T("failed to access KFM FMCount buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        const int countFrameIndex = pending.cycle * 5 - 3 + pair + 1;
        if (countFrameIndex >= 0) {
            counts[pair * 2 + 0] = gpuCounts[0];
            counts[pair * 2 + 1] = gpuCounts[1];
        }
    }
    const int firstSourceIndex = pending.cycle * 5 - 3;
    const int firstValidPair = std::max(0, -(firstSourceIndex + 1));
    if (firstValidPair > 0 && firstValidPair < KFM_FMCOUNT_PAIRS) {
        for (int pair = 0; pair < firstValidPair; pair++) {
            counts[pair * 2 + 0] = counts[firstValidPair * 2 + 0];
            counts[pair * 2 + 1] = counts[firstValidPair * 2 + 1];
        }
    }
    m_pendingFMCounts.pop_front();
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::analyzeAvailableSource(bool drain, cudaStream_t stream) {
    if (!m_analyzer || m_cachedSourceFrames <= 0) {
        return RGY_ERR_NONE;
    }

    const int readyCycles = drain
        ? divCeil(m_cachedSourceFrames, 5)
        : (m_cachedSourceFrames >= 8 ? ((m_cachedSourceFrames - 8) / 5 + 1) : 0);
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    const auto timing = prm ? prm->kfm.timing : VppKfmTiming::Realtime;
    while (m_nextFMCountSubmitCycle < readyCycles) {
        auto sts = submitFMCounts(m_nextFMCountSubmitCycle, drain, stream);
        if (sts == RGY_ERR_MORE_DATA) {
            break;
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_nextFMCountSubmitCycle++;
    }

    if (m_nextAnalyzeCycle >= readyCycles) {
        if (drain && timing != VppKfmTiming::Realtime) {
            finalizeAnalyzerResults(timing);
        }
        return RGY_ERR_NONE;
    }

    const auto *frame = m_sourceCache.empty() ? nullptr : &m_sourceCache.back().frame->frame;
    if (!frame) {
        return RGY_ERR_NONE;
    }
    while (m_nextAnalyzeCycle < readyCycles) {
        std::array<RGYKFM::FMCount, KFM_FMCOUNT_COUNT> counts = {};
        auto sts = readbackFMCounts(counts, m_nextAnalyzeCycle, drain, stream);
        if (sts == RGY_ERR_MORE_DATA) {
            return RGY_ERR_NONE;
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        writeFMCountDump(counts, m_nextAnalyzeCycle);
        try {
            if (timing == VppKfmTiming::Realtime) {
                const auto result = m_analyzer->realtimeFromCounts(counts.data(), frame->width, frame->height);
                writeAnalyzerResult(result, true);
            } else {
                m_analyzer->analyzeCycleFromCounts(counts.data(), frame->width, frame->height);
                if (timing == VppKfmTiming::RealtimePlus) {
                    const auto resultCount = m_analyzer->results().size();
                    const auto delay = static_cast<size_t>(std::max(0, m_analyzer->param().pastCycles));
                    appendAnalyzerResults(resultCount > delay ? resultCount - delay : 0, true, true);
                }
            }
        } catch (const std::exception& e) {
            AddMessage(RGY_LOG_ERROR, _T("failed to analyze KFM cycle %d: %S.\n"), m_nextAnalyzeCycle, e.what());
            return RGY_ERR_INVALID_CALL;
        }
        m_nextAnalyzeCycle++;
    }
    if (drain && timing != VppKfmTiming::Realtime) {
        finalizeAnalyzerResults(timing);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::flushUcfNoiseResultDump() {
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();
}

RGY_ERR NVEncFilterKfm::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(pInputFrame);
    UNREFERENCED_PARAMETER(ppOutputFrames);
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    UNREFERENCED_PARAMETER(stream);
    return RGY_ERR_UNSUPPORTED;
}

void NVEncFilterKfm::close() {
    flushUcfNoiseResultDump();
    m_rtgmc.reset();
    m_deint60Rtgmc.reset();
    m_before60Rtgmc.reset();
    m_after60Rtgmc.reset();
    m_nrFilter.reset();
    m_analyzer.reset();
    m_sourceCache.clear();
    m_deint60Cache.clear();
    m_before60Cache.clear();
    m_after60Cache.clear();
    m_ucfNoiseCache.clear();
    if (m_kfmFramePool) {
        m_kfmFramePool->clear();
    }
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingVfrOutputs.clear();
    m_deint60CacheCopyEvent = RGYCudaEvent();
    m_before60CacheCopyEvent = RGYCudaEvent();
    m_after60CacheCopyEvent = RGYCudaEvent();
    m_staticFlag.reset();
    for (auto& frame : m_staticWorkFrames) {
        frame.reset();
    }
    for (auto& flag : m_analyzeFlags) {
        flag.reset();
    }
    m_pendingFMCounts.clear();
    m_fmCountBufPool.clear();
    for (auto& raw : m_telecineSuperRaw) {
        raw.reset();
    }
    for (auto& frame : m_telecineSuperFrames) {
        frame.reset();
    }
    for (auto& frame : m_telecineSuperNeighborFrames) {
        frame.reset();
    }
    for (auto& frame : m_switchFlagFrames) {
        frame.reset();
    }
    for (auto& frame : m_containsCombeFrames) {
        frame.reset();
    }
    for (auto& frame : m_combeMaskFrames) {
        frame.reset();
    }
    for (auto& frame : m_patchCombeFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseFieldFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseGaussTmpFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseGaussFrames) {
        frame.reset();
    }
    for (auto& programs : m_ucfNoiseGaussVert) {
        for (auto& program : programs) {
            program = KfmUcfGaussProgram();
        }
    }
    for (auto& programs : m_ucfNoiseGaussHori) {
        for (auto& program : programs) {
            program = KfmUcfGaussProgram();
        }
    }
    for (auto& work : m_switchFlagWork) {
        work.reset();
    }
    m_switchFlagWorkEvent = RGYCudaEvent();
    m_containsCombeCount.reset();
    if (m_fpResult) {
        fclose(m_fpResult);
        m_fpResult = nullptr;
    }
    if (m_fpFMCount) {
        fclose(m_fpFMCount);
        m_fpFMCount = nullptr;
    }
    if (m_fpTimecode) {
        fclose(m_fpTimecode);
        m_fpTimecode = nullptr;
    }
    if (m_fpFrameInfo) {
        fclose(m_fpFrameInfo);
        m_fpFrameInfo = nullptr;
    }
    if (m_fpContainsCombe) {
        fclose(m_fpContainsCombe);
        m_fpContainsCombe = nullptr;
    }
    if (m_fpUcfNoise) {
        fclose(m_fpUcfNoise);
        m_fpUcfNoise = nullptr;
    }
    m_switchDurationPath.clear();
    m_switchTimecodePath.clear();
    m_stageDumpDir.clear();
    m_analyzerOutputResults.clear();
    m_switchSingleFrameN60.clear();
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    m_stageDumpTargetFrames.clear();
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();
    m_hasLastAnalyzeResult = false;
    m_analyzerFinalized = false;
    m_switchTimingDumped = false;
    m_analyzeSourceFrames = 0;
    m_nextAnalyzeCycle = 0;
    m_nextFMCountSubmitCycle = 0;
    m_nextFMCountDumpFrame = 0;
    m_cachedSourceFrames = 0;
    m_deint60SubmittedSourceFrames = 0;
    m_before60SubmittedSourceFrames = 0;
    m_after60SubmittedSourceFrames = 0;
    m_nextSwitchN60 = 0;
    m_nextSwitchPts = 0;
    m_hasLastSwitchTiming = false;
    m_lastSwitchStart60 = 0;
    m_lastSwitchDuration60 = 0;
    m_lastSwitchStart120 = 0;
    m_lastSwitchIsFrame24 = false;
    m_nextTelecine24Frame = 0;
    m_nextTelecine24Pts = 0;
    m_telecineSuperBufferIndex = 0;
    m_maskBranchBufferIndex = 0;
    m_patchCombeBufferIndex = 0;
    m_stageDumpMaxFrames = 0;
    m_timecodeFrameIndex = 0;
    m_outputBufferIndex = 0;
    m_workBufferIndex = 0;
    m_workFrameBuf.clear();
    m_frameBuf.clear();
}
