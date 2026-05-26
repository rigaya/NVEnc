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
#include "NVEncFilterRtgmc.h"
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
static constexpr int KFM_REMOVE_COMBE_THRESH_Y = 6;
static constexpr int KFM_REMOVE_COMBE_THRESH_C = 6;
static constexpr int KFM_SWITCH_FLAG_THRESH_Y = 60;
static constexpr int KFM_SWITCH_FLAG_THRESH_C = 80;
static constexpr int KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN = 64;
static constexpr int KFM_REALTIMEPLUS_DEINT60_CACHE_MARGIN = KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN * 2;
static constexpr int KFM_VFR_SOURCE_TRIM_LOOKBEHIND = 8;
static constexpr int KFM_VFR_DEINT60_TRIM_LOOKBEHIND = 16;

static int kfmFloorDiv2(const int value) {
    return value >= 0 ? value / 2 : -((1 - value) / 2);
}

static int kfmDepthScale(RGY_CSP csp) {
    return 1 << std::max(0, RGY_CSP_BIT_DEPTH[csp] - 8);
}

static int kfmPow2Shift(int scale) {
    if (scale <= 0 || (scale & (scale - 1)) != 0) {
        return -1;
    }
    int shift = 0;
    while ((1 << shift) < scale) {
        shift++;
    }
    return shift;
}

static bool kfmUseFusedSwitchFlagBinaryExtend() {
    const char *env = std::getenv("NVENC_KFM_SWITCH_FLAG_BINARY_EXTEND_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static bool kfmUseFusedCleanSuper() {
    const char *env = std::getenv("NVENC_KFM_CLEAN_SUPER_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static bool kfmDisableCCDuration() {
    const char *env = std::getenv("NVENC_KFM_DISABLE_CC_DURATION");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmDeint60BranchEnabled() {
    const char *env = std::getenv("NVENC_KFM_ENABLE_DEINT60_BRANCH");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmCspHasInterleavedUV(const RGY_CSP csp) {
    return csp == RGY_CSP_NV12 || csp == RGY_CSP_P010
        || csp == RGY_CSP_NV16 || csp == RGY_CSP_P210;
}

static void kfmEraseInternalFrameData(RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    frame->dataList.erase(std::remove_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data
            && !(data->dataType() == RGY_FRAME_DATA_QP
                || data->dataType() == RGY_FRAME_DATA_METADATA
                || data->dataType() == RGY_FRAME_DATA_HDR10PLUS
                || data->dataType() == RGY_FRAME_DATA_DOVIRPU
                || data->dataType() == RGY_FRAME_DATA_KFM_SWITCH);
    }), frame->dataList.end());
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

RGY_ERR NVEncFilterKfm::initRtgmc(const std::shared_ptr<NVEncFilterParamKfm>& prm, std::unique_ptr<NVEncFilterRtgmc>& rtgmc, bool updateOutputParam, int useFlag) {
    auto rtgmcParam = std::make_shared<NVEncFilterParamRtgmc>();
    rtgmcParam->rtgmc.enable = true;
    rtgmcParam->rtgmc.preset = prm->kfm.preset;
    apply_vpp_rtgmc_preset(rtgmcParam->rtgmc, prm->kfm.preset, rtgmcParam->rtgmc.tuning);
    if (useFlag > 0) {
        rtgmcParam->rtgmc.tr1.useFlag = useFlag;
        rtgmcParam->rtgmc.tr2.useFlag = useFlag;
        rtgmcParam->rtgmc.analyze.useFlag = useFlag;
    }
    rtgmcParam->frameIn = prm->frameIn;
    rtgmcParam->frameOut = prm->frameOut;
    rtgmcParam->baseFps = prm->baseFps;
    rtgmcParam->timebase = prm->timebase;
    rtgmcParam->bOutOverwrite = false;

    rtgmc = std::make_unique<NVEncFilterRtgmc>();
    auto sts = rtgmc->init(rtgmcParam, m_pLog);
    if (sts != RGY_ERR_NONE) {
        rtgmc.reset();
        return sts;
    }

    if (updateOutputParam) {
        prm->frameOut = rtgmcParam->frameOut;
        prm->baseFps = rtgmcParam->baseFps;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::initNrFilter(const std::shared_ptr<NVEncFilterParamKfm>& prm) {
    m_nrFilter.reset();
    if (!prm->kfm.nr) {
        return RGY_ERR_NONE;
    }

    auto nrParam = std::make_shared<NVEncFilterParamDegrain>();
    nrParam->frameIn = prm->frameOut;
    nrParam->frameOut = prm->frameOut;
    nrParam->baseFps = prm->baseFps;
    nrParam->bOutOverwrite = false;
    nrParam->degrain.enable = true;
    nrParam->degrain.preset = VppDegrainPreset::Auto;
    nrParam->degrain.mode = VppDegrainMode::Degrain;
    nrParam->degrain.stage = VppDegrainStage::TR1;
    nrParam->degrain.delta = 1;
    nrParam->degrain.search = 4;
    nrParam->degrain.thsad = 300;
    nrParam->degrain.thsadc = 150;
    nrParam->degrain.thscd1 = 1600;
    nrParam->degrain.thscd2 = 130;
    nrParam->degrain.pel = 1;
    nrParam->degrain.blksize = 16;
    nrParam->degrain.overlap = nrParam->degrain.blksize / 2;
    nrParam->degrain.levels = 2;
    nrParam->degrain.chroma = true;
    nrParam->degrain.binomial = 1;
    nrParam->degrain.tvRange = true;

    auto nrFilter = std::make_unique<NVEncFilterDegrain>();
    auto sts = nrFilter->init(nrParam, m_pLog);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize KFM NR Auto degrain(TR1) filter: %s.\n"), get_err_mes(sts));
        return sts;
    }
    AddMessage(RGY_LOG_INFO, _T("--vpp-kfm nr=true is wired to degrain preset=auto,tr=1,blksize=16,levels=2,binomial=true on the final KFM output stream.\n"));
    m_nrFilter = std::move(nrFilter);
    return RGY_ERR_NONE;
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
    m_pLog = pPrintMes;
    if (prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->kfm.ucf) {
        AddMessage(RGY_LOG_INFO, _T("--vpp-kfm ucf=true enables the UCF debug field/crop noise pre-stage.\n"));
    }

    m_pathThrough = FILTER_PATHTHROUGH_NONE;
    m_rtgmc.reset();
    m_deint60Rtgmc.reset();
    m_before60Rtgmc.reset();
    m_after60Rtgmc.reset();
    m_before60Cache.clear();
    m_after60Cache.clear();
    m_before60SubmittedSourceFrames = 0;
    m_after60SubmittedSourceFrames = 0;
    m_before60CacheCopyEvent = RGYCudaEvent();
    m_after60CacheCopyEvent = RGYCudaEvent();

    auto sts = RGY_ERR_NONE;
    if (prm->kfm.mode == VppKfmMode::P60) {
        sts = initRtgmc(prm, m_rtgmc, true);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            sts = initRtgmc(prm, m_before60Rtgmc, false, 1);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = initRtgmc(prm, m_after60Rtgmc, false, 2);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }

    const bool needTelecineWorkFrames = prm->kfm.mode == VppKfmMode::P24
        || prm->kfm.mode == VppKfmMode::VFR;
    const int frameBufCount = needTelecineWorkFrames ? (prm->kfm.ucf ? 16 : 8) : 8;
    sts = AllocFrameBuf(prm->frameOut, frameBufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (needTelecineWorkFrames || prm->kfm.mode == VppKfmMode::P60) {
        sts = allocWorkFrameBuf(prm->frameOut, frameBufCount);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM work buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_workFrameBuf.clear();
        m_workBufferIndex = 0;
    }
    if (prm->kfm.mode == VppKfmMode::VFR || prm->kfm.mode == VppKfmMode::P60 || prm->kfm.ucf) {
        m_staticFlag = std::make_unique<CUFrameBuf>(prm->frameOut);
        m_staticFlag->releasePtr();
        sts = m_staticFlag->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static flag frame: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_staticFlag.reset();
    }
    if (prm->kfm.mode == VppKfmMode::VFR
        || (prm->kfm.mode == VppKfmMode::P24 && kfmDeint60BranchEnabled())) {
        sts = initRtgmc(prm, m_deint60Rtgmc, false);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (prm->kfm.mode != VppKfmMode::P60 && prm->kfm.ucf) {
        sts = initRtgmc(prm, m_before60Rtgmc, false, 1);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = initRtgmc(prm, m_after60Rtgmc, false, 2);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    sts = initAnalyzer(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    initStageDumpConfig(*prm);
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initNrFilter(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->kfm.mode == VppKfmMode::VFR) {
        prm->baseFps *= 2;
    }

    m_sourceCache.clear();
    m_deint60Cache.clear();
    m_deint60SubmittedSourceFrames = 0;
    m_deint60CacheCopyEvent = RGYCudaEvent();
    m_outputBufferIndex = 0;
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
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

RGY_ERR NVEncFilterKfm::runDeint60Branch(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_deint60Rtgmc) {
        return RGY_ERR_NONE;
    }

    int deint60OutNum = 0;
    RGYFrameInfo *deint60OutFrames[8] = { 0 };
    RGYCudaEvent deint60Event;
    auto sts = m_deint60Rtgmc->filter(const_cast<RGYFrameInfo *>(frame), deint60OutFrames, &deint60OutNum, stream, wait_events, &deint60Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    std::vector<RGYCudaEvent> cacheWaitEvents;
    if (deint60Event() != nullptr) {
        cacheWaitEvents.push_back(deint60Event);
    }
    for (int i = 0; i < deint60OutNum; i++) {
        RGYCudaEvent cacheEvent;
        sts = cacheDeint60Frame(deint60OutFrames[i], stream, cacheWaitEvents, &cacheEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cacheEvent() != nullptr) {
            m_deint60CacheCopyEvent = cacheEvent;
        }
        if (cachedFrames) {
            (*cachedFrames)++;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::drainDeint60Branch(cudaStream_t stream, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_deint60Rtgmc) {
        return RGY_ERR_NONE;
    }
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    for (int iter = 0; !m_deint60Rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            AddMessage(RGY_LOG_ERROR, _T("KFM deint60 RTGMC drain did not complete after %d iterations.\n"), maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        int drainedFrames = 0;
        auto sts = runDeint60Branch(nullptr, stream, {}, &drainedFrames);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cachedFrames) {
            *cachedFrames += drainedFrames;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::cacheDeint60Frame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    KfmCachedDeint60 entry;
    entry.n60 = m_deint60SubmittedSourceFrames++;
    entry.inputFrameId = frame->inputFrameId;
    entry.timestamp = frame->timestamp;
    entry.duration = frame->duration;
    entry.frame = acquireKfmFrame(*frame, _T("deint60 cache"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto mergeWaitEvents = wait_events;
    const int sourceIndex = entry.n60 >> 1;
    const auto *source = findSourceByIndexExact(sourceIndex);
    if (!source) {
        AddMessage(RGY_LOG_ERROR, _T("KFM source frame is missing for deint60 output n60=%d, sourceIndex=%d, inputFrameId=%d.\n"),
            entry.n60, sourceIndex, frame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    if (source->event() != nullptr) {
        mergeWaitEvents.push_back(source->event);
    }
    auto sts = dumpStageFrame("rtgmc60-raw", frame, entry.n60, stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent staticEvent;
    sts = analyzeStaticFlag(source->sourceIndex, stream, mergeWaitEvents, &staticEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (staticEvent() != nullptr) {
        mergeWaitEvents.push_back(staticEvent);
    }
    sts = dumpStageFrame("static-flag", &m_staticFlag->frame, sourceIndex, stream,
        (staticEvent() != nullptr) ? std::vector<RGYCudaEvent>{ staticEvent } : mergeWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = mergeStatic(&entry.frame->frame, frame, &source->frame->frame, stream, mergeWaitEvents, &entry.event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to merge/cache KFM deint60 frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if (event && entry.event() != nullptr) {
        *event = entry.event;
    }
    writeFrameInfoDump("deint60", &entry.frame->frame);
    sts = dumpStageFrame("deint60", &entry.frame->frame, entry.n60, stream,
        (entry.event() != nullptr) ? std::vector<RGYCudaEvent>{ entry.event } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_deint60Cache.push_back(std::move(entry));
    const auto cacheLimit = deint60CacheLimit();
    const auto trimFloor = deint60CacheTrimFloor();
    while (m_deint60Cache.size() > cacheLimit && m_deint60Cache.front().n60 < trimFloor) {
        m_deint60Cache.pop_front();
    }
    return RGY_ERR_NONE;
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

RGY_ERR NVEncFilterKfm::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, int frame24Index,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
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
    auto sts = kfmWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostPlaneY(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostPlaneY.ptr[0] = hostY.data();
    hostPlaneY.pitch[0] = planeY.width;
    sts = copyPlaneAsync(&hostPlaneY, &planeY, stream);
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

RGY_ERR NVEncFilterKfm::clearStaticFlag(cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto planes = RGY_CSP_PLANES[m_staticFlag->frame.csp];
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto plane = getPlane(&m_staticFlag->frame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_zero_plane(&plane, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_zero (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::analyzeStaticFlag(cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }
    if (m_sourceCache.empty()) {
        return clearStaticFlag(stream, wait_events, event);
    }
    return analyzeStaticFlag(m_sourceCache.back().sourceIndex, stream, wait_events, event);
}

RGY_ERR NVEncFilterKfm::analyzeStaticFlag(int sourceIndex, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto *cur = findSourceByIndex(sourceIndex);
    if (!cur || !cur->frame || !cur->frame->frame.ptr[0] || !cur->paddedFrame || !cur->paddedFrame->frame.ptr[0]) {
        return clearStaticFlag(stream, wait_events, event);
    }
    for (auto& frame : m_staticWorkFrames) {
        if (!frame
            || frame->frame.width != m_staticFlag->frame.width
            || frame->frame.height != m_staticFlag->frame.height
            || frame->frame.csp != m_staticFlag->frame.csp) {
            frame = std::make_unique<CUFrameBuf>(m_staticFlag->frame);
            frame->releasePtr();
            const auto sts = frame->alloc();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static work frame: %s.\n"), get_err_mes(sts));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    std::vector<RGYCudaEvent> sourceWaitEvents = wait_events;
    for (int offset = -3; offset <= 3; offset++) {
        const auto *src = findSourceByIndex(sourceIndex + offset);
        if (!src || !src->frame || !src->frame->frame.ptr[0]) {
            return clearStaticFlag(stream, wait_events, event);
        }
        if (src->event() != nullptr) {
            sourceWaitEvents.push_back(src->event);
        }
    }
    if (cur->paddedEvent() != nullptr) {
        sourceWaitEvents.push_back(cur->paddedEvent);
    }

    const auto planes = RGY_CSP_PLANES[m_staticFlag->frame.csp];
    const auto runCalcCombe = [&](RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const std::vector<RGYCudaEvent>& waits, RGYCudaEvent *outEvent) -> RGY_ERR {
        RGYCudaEvent prevEvent;
        for (int iplane = 0; iplane < planes; iplane++) {
            auto dst = getPlane(dstFrame, (RGY_PLANE)iplane);
            auto src = getPlane(srcFrame, (RGY_PLANE)iplane);
            const int srcYOffset = (src.height - dst.height) >> 1;
            if (src.width != dst.width || src.height != dst.height + srcYOffset * 2 || srcYOffset < 2) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM static padded source plane size (plane %d, dst %dx%d, src %dx%d, offset %d).\n"),
                    iplane, dst.width, dst.height, src.width, src.height, srcYOffset);
                return RGY_ERR_INVALID_PARAM;
            }
            const int width4 = dst.width >> 2;
            if (width4 <= 0 || (dst.width & 3) != 0) {
                return RGY_ERR_INVALID_PARAM;
            }
            const auto waitHere = (iplane == 0)
                ? waits
                : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
            auto sts = kfmWaitEvents(stream, waitHere);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = run_kfm_static_calc_combe_plane(&dst, &src, srcYOffset, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_calc_combe (plane %d): %s.\n"), iplane, get_err_mes(sts));
                return sts;
            }
            sts = kfmRecordEvent(stream, &prevEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (outEvent && prevEvent() != nullptr) {
            *outEvent = prevEvent;
        }
        return RGY_ERR_NONE;
    };
    const auto runTemporalMinDiff3 = [&](RGYFrameInfo *dstFrame, int centerIndex, const std::vector<RGYCudaEvent>& waits, RGYCudaEvent *outEvent) -> RGY_ERR {
        std::array<const KfmCachedSource *, 7> src = {};
        for (int i = 0; i < (int)src.size(); i++) {
            src[i] = findSourceByIndex(centerIndex + i - 3);
            if (!src[i] || !src[i]->frame || !src[i]->frame->frame.ptr[0]) {
                return RGY_ERR_MORE_DATA;
            }
        }
        RGYCudaEvent prevEvent;
        for (int iplane = 0; iplane < planes; iplane++) {
            auto dst = getPlane(dstFrame, (RGY_PLANE)iplane);
            auto src0 = getPlane(&src[0]->frame->frame, (RGY_PLANE)iplane);
            auto src1 = getPlane(&src[1]->frame->frame, (RGY_PLANE)iplane);
            auto src2 = getPlane(&src[2]->frame->frame, (RGY_PLANE)iplane);
            auto src3 = getPlane(&src[3]->frame->frame, (RGY_PLANE)iplane);
            auto src4 = getPlane(&src[4]->frame->frame, (RGY_PLANE)iplane);
            auto src5 = getPlane(&src[5]->frame->frame, (RGY_PLANE)iplane);
            auto src6 = getPlane(&src[6]->frame->frame, (RGY_PLANE)iplane);
            const int width4 = dst.width >> 2;
            if (width4 <= 0 || (dst.width & 3) != 0
                || src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
                || src3.width != dst.width || src4.width != dst.width || src5.width != dst.width || src6.width != dst.width
                || src0.height != dst.height || src1.height != dst.height || src2.height != dst.height
                || src3.height != dst.height || src4.height != dst.height || src5.height != dst.height || src6.height != dst.height) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM temporal min-diff source plane size (plane %d).\n"), iplane);
                return RGY_ERR_INVALID_PARAM;
            }
            const auto waitHere = (iplane == 0)
                ? waits
                : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
            auto sts = kfmWaitEvents(stream, waitHere);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = run_kfm_temporal_min_diff5_3_plane(&dst, &src0, &src1, &src2, &src3, &src4, &src5, &src6, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_temporal_min_diff5_3 (plane %d): %s.\n"), iplane, get_err_mes(sts));
                return sts;
            }
            sts = kfmRecordEvent(stream, &prevEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (outEvent && prevEvent() != nullptr) {
            *outEvent = prevEvent;
        }
        return RGY_ERR_NONE;
    };
    const auto runMergeUV = [&](RGYFrameInfo *frame, const std::vector<RGYCudaEvent>& waits, RGYCudaEvent *outEvent) -> RGY_ERR {
        if (planes < 3) {
            if (outEvent && !waits.empty()) {
                *outEvent = waits.back();
            }
            return RGY_ERR_NONE;
        }
        auto y = getPlane(frame, RGY_PLANE_Y);
        auto u = getPlane(frame, RGY_PLANE_U);
        auto v = getPlane(frame, RGY_PLANE_V);
        auto sts = kfmWaitEvents(stream, waits);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_merge_uv_coefs_plane(&y, &u, &v, 1, 1, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_merge_uv_coefs: %s.\n"), get_err_mes(sts));
            return sts;
        }
        return kfmRecordEvent(stream, outEvent);
    };
    const auto runExtend = [&](RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const std::vector<RGYCudaEvent>& waits, RGYCudaEvent *outEvent) -> RGY_ERR {
        auto dst = getPlane(dstFrame, RGY_PLANE_Y);
        auto src = getPlane(srcFrame, RGY_PLANE_Y);
        const int width4 = dst.width >> 2;
        if (width4 <= 0 || (dst.width & 3) != 0) {
            return RGY_ERR_INVALID_PARAM;
        }
        auto sts = kfmWaitEvents(stream, waits);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_extend_coefs_plane(&dst, &src, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_extend_coefs: %s.\n"), get_err_mes(sts));
            return sts;
        }
        return kfmRecordEvent(stream, outEvent);
    };

    RGYCudaEvent combEvent;
    auto sts = runCalcCombe(&m_staticWorkFrames[0]->frame, &cur->paddedFrame->frame, sourceWaitEvents, &combEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto phaseWait = (combEvent() != nullptr) ? std::vector<RGYCudaEvent>{ combEvent } : std::vector<RGYCudaEvent>();
    RGYCudaEvent mergeUvEvent;
    sts = runMergeUV(&m_staticWorkFrames[0]->frame, phaseWait, &mergeUvEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    phaseWait = (mergeUvEvent() != nullptr) ? std::vector<RGYCudaEvent>{ mergeUvEvent } : phaseWait;
    RGYCudaEvent flagcEvent;
    sts = runExtend(&m_staticFlag->frame, &m_staticWorkFrames[0]->frame, phaseWait, &flagcEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYCudaEvent prevEvent;
    sts = runTemporalMinDiff3(&m_staticWorkFrames[4]->frame, sourceIndex, sourceWaitEvents, &prevEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    phaseWait = (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>();
    RGYCudaEvent temporalMergeUvEvent;
    sts = runMergeUV(&m_staticWorkFrames[4]->frame, phaseWait, &temporalMergeUvEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    phaseWait = (temporalMergeUvEvent() != nullptr) ? std::vector<RGYCudaEvent>{ temporalMergeUvEvent } : phaseWait;
    RGYCudaEvent flagdEvent;
    sts = runExtend(&m_staticWorkFrames[0]->frame, &m_staticWorkFrames[4]->frame, phaseWait, &flagdEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto flagcY = getPlane(&m_staticFlag->frame, RGY_PLANE_Y);
    auto flagdY = getPlane(&m_staticWorkFrames[0]->frame, RGY_PLANE_Y);
    std::vector<RGYCudaEvent> andWaitEvents;
    if (flagcEvent() != nullptr) {
        andWaitEvents.push_back(flagcEvent);
    }
    if (flagdEvent() != nullptr) {
        andWaitEvents.push_back(flagdEvent);
    }
    sts = kfmWaitEvents(stream, andWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent andEvent;
    sts = run_kfm_and_coefs_plane(&flagcY, &flagdY, 1.0f / 30.0f, 1.0f / 15.0f, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_and_coefs: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = kfmRecordEvent(stream, &andEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    if (planes >= 3) {
        auto flagU = getPlane(&m_staticFlag->frame, RGY_PLANE_U);
        auto flagV = getPlane(&m_staticFlag->frame, RGY_PLANE_V);
        auto applyWait = (andEvent() != nullptr) ? std::vector<RGYCudaEvent>{ andEvent } : std::vector<RGYCudaEvent>();
        sts = kfmWaitEvents(stream, applyWait);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_apply_uv_coefs_420_plane(&flagU, &flagV, &flagcY, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_apply_uv_coefs_420: %s.\n"), get_err_mes(sts));
            return sts;
        }
        return kfmRecordEvent(stream, event);
    }
    if (event && andEvent() != nullptr) {
        *event = andEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::mergeStatic(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !pDeint60Frame || !pSourceFrame || !m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }

    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src60 = getPlane(pDeint60Frame, (RGY_PLANE)iplane);
        auto src30 = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        auto flag = getPlane(&m_staticFlag->frame, (RGY_PLANE)iplane);
        const int width4 = dst.width >> 2;
        if (width4 <= 0 || (dst.width & 3) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("KFM mode=60 requires plane width to be multiple of 4 (plane %d, width %d).\n"), iplane, dst.width);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_merge_static_plane(&dst, &src60, &src30, &flag, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_merge_static (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pDeint60Frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderTelecine24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !m_analyzer) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    const int firstSource = (firstField & ~1) >> 1;
    const int paritySourceIndex = (frame24Index / 4) * 5;
    const auto *paritySource = (drain || paritySourceIndex < 0) ? findSourceByIndex(paritySourceIndex) : findSourceByIndexExact(paritySourceIndex);
    if (!paritySource || !paritySource->frame || !paritySource->frame->frame.ptr[0]) {
        return RGY_ERR_MORE_DATA;
    }
    std::array<const KfmCachedSource *, 3> src = {};
    for (int i = 0; i < (int)src.size(); i++) {
        const int sourceIndex = firstSource + i;
        src[i] = (drain || sourceIndex < 0) ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
        if (!src[i] || !src[i]->paddedFrame || !src[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    std::vector<RGYCudaEvent> sourceWaitEvents = wait_events;
    for (const auto *s : src) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYCudaEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src0 = getPlane(&src[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto src1 = getPlane(&src[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto src2 = getPlane(&src[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int srcYOffset = (src0.height - dst.height) >> 1;
        if (src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
            || src0.height != src1.height || src0.height != src2.height
            || src0.height != dst.height + srcYOffset * 2 || srcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM 24p padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, src0.width, src0.height, src1.width, src1.height, src2.width, src2.height, srcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_telecine_weave_plane(&dst, &src0, &src1, &src2, srcYOffset,
            firstField, info.numFields, kfmFrameParity(&paritySource->frame->frame), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_telecine_weave (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, &src[0]->frame->frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    if (pOutputFrame->duration > 0) {
        pOutputFrame->duration = std::max<int64_t>(1, (pOutputFrame->duration * 5 + 2) / 4);
    }
    if (m_nextTelecine24Frame == frame24Index) {
        pOutputFrame->timestamp = m_nextTelecine24Pts;
    }
    KfmSwitchTiming timing;
    timing.start60 = firstField;
    timing.start120 = firstField * 2;
    timing.sourceIndex = src[0]->sourceIndex;
    timing.frame24Index = frame24Index;
    timing.baseType = KFM_FRAME_24;
    timing.sourceStart = firstSource;
    timing.numSourceFrames = 3;
    const int totalFields = m_cachedSourceFrames * 2;
    const int availableFields = (totalFields > firstField) ? std::min(info.numFields, totalFields - firstField) : info.numFields;
    timing.duration60 = std::max(1, availableFields);
    timing.duration120 = timing.duration60 * 2;
    timing.isFrame24 = true;
    timing.isFrame60 = false;
    attachSwitchFrameData(pOutputFrame, timing, &result);
    writeFrameInfoDump("deint24", pOutputFrame, &result);
    const auto dumpSts = dumpStageFrame("deint24", pOutputFrame, frame24Index, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderDoubleWeaveFrame(RGYFrameInfo *pOutputFrame, int firstField, int fieldCount, bool drain,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || firstField < 0 || fieldCount <= 0 || fieldCount > 6) {
        return RGY_ERR_INVALID_CALL;
    }

    const int fieldBase = firstField & ~1;
    const int firstSource = fieldBase >> 1;
    const int lastField = firstField + fieldCount - 1;
    const int lastSourceOffset = std::max(0, ((lastField & ~1) - fieldBase) >> 1);

    auto sourceAt = [this, drain](const int sourceIndex) -> const KfmCachedSource * {
        return drain ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
    };

    std::array<const KfmCachedSource *, 3> src = {};
    for (int i = 0; i < (int)src.size(); i++) {
        const int sourceIndex = firstSource + std::min(i, lastSourceOffset);
        src[i] = sourceAt(sourceIndex);
        if (!src[i] || !src[i]->paddedFrame || !src[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    std::vector<RGYCudaEvent> sourceWaitEvents = wait_events;
    for (const auto *s : src) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYCudaEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src0 = getPlane(&src[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto src1 = getPlane(&src[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto src2 = getPlane(&src[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int srcYOffset = (src0.height - dst.height) >> 1;
        if (src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
            || src0.height != src1.height || src0.height != src2.height
            || src0.height != dst.height + srcYOffset * 2 || srcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF24 DoubleWeave padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, src0.width, src0.height, src1.width, src1.height, src2.width, src2.height, srcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_telecine_weave_plane(&dst, &src0, &src1, &src2, srcYOffset,
            firstField, fieldCount, kfmFrameParity(&src[0]->frame->frame), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_telecine_weave (UCF24 DoubleWeave, plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, &src[0]->frame->frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    KfmSwitchTiming timing;
    timing.start60 = firstField;
    timing.start120 = firstField * 2;
    timing.sourceIndex = firstSource;
    timing.frame24Index = firstField;
    timing.baseType = KFM_FRAME_UCF;
    timing.sourceStart = firstSource;
    timing.numSourceFrames = lastSourceOffset + 1;
    timing.duration60 = fieldCount;
    timing.duration120 = fieldCount * 2;
    timing.isFrame24 = false;
    timing.isFrame60 = false;
    attachSwitchFrameData(pOutputFrame, timing, nullptr);
    writeFrameInfoDump("ucf24-dweave", pOutputFrame);
    const auto dumpSts = dumpStageFrame("ucf24-dweave", pOutputFrame, firstField, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderCleanSuperFields(RGYFrameInfo *pOutputFrame, int firstSuperField, int lastSuperField, int propSourceIndex, int outputFrameId, bool drain,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    if (firstSuperField > lastSuperField) {
        return RGY_ERR_INVALID_PARAM;
    }

    const bool interleavedUV = kfmCspHasInterleavedUV(pOutputFrame->csp);
    const int targetPlanes = (RGY_CSP_PLANES[pOutputFrame->csp] >= 3) ? 3 : (interleavedUV ? 3 : 1);
    RGYCudaEvent prevEvent;

    auto sourceAt = [this, drain](const int sourceIndex) -> const KfmCachedSource * {
        return (drain || sourceIndex < 0) ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
    };

    auto renderRawSuper = [&](const int bufIndex, const int sourceIndex, const RGY_PLANE plane, const int pixelStep, const int pixelOffset,
                              const int rawPitch, const int gridWidth, const int gridHeight,
                              const std::vector<RGYCudaEvent>& baseWaitEvents, RGYCudaEvent *rawEvent) -> RGY_ERR {
        const auto *src0 = sourceAt(sourceIndex);
        const auto *src1 = sourceAt(sourceIndex + 1);
        if (!src0 || !src0->frame || !src0->frame->frame.ptr[0]
            || !src1 || !src1->frame || !src1->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        auto src0Plane = getPlane(&src0->frame->frame, plane);
        auto src1Plane = getPlane(&src1->frame->frame, plane);
        if (!src0Plane.ptr[0] || !src1Plane.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        if (src0Plane.width != src1Plane.width || src0Plane.height != src1Plane.height) {
            return RGY_ERR_INVALID_CALL;
        }
        const int srcLogicalWidth = src0Plane.width / pixelStep;
        const int srcGridWidth = srcLogicalWidth >> 2;
        const int srcGridHeight = src0Plane.height >> 2;
        if (srcLogicalWidth <= 0 || (srcLogicalWidth & 3) != 0 || srcGridWidth != gridWidth || srcGridHeight != gridHeight) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super raw source size (source %d, plane %d, logical %dx%d, grid %dx%d, expected %dx%d).\n"),
                sourceIndex, (int)plane, srcLogicalWidth, src0Plane.height, srcGridWidth, srcGridHeight, gridWidth, gridHeight);
            return RGY_ERR_INVALID_PARAM;
        }

        const size_t rawBytes = (size_t)rawPitch * gridHeight * 2;
        if (!m_telecineSuperRaw[bufIndex] || m_telecineSuperRaw[bufIndex]->nSize < rawBytes) {
            auto raw = std::make_unique<CUMemBuf>(rawBytes);
            const auto allocSts = raw->alloc();
            if (allocSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super raw buffer %d: %s.\n"), bufIndex, get_err_mes(allocSts));
                return allocSts;
            }
            m_telecineSuperRaw[bufIndex] = std::move(raw);
        }

        std::vector<RGYCudaEvent> clearWaitEvents = baseWaitEvents;
        if (src0->event() != nullptr) {
            clearWaitEvents.push_back(src0->event);
        }
        if (src1->event() != nullptr) {
            clearWaitEvents.push_back(src1->event);
        }

        auto sts = kfmWaitEvents(stream, clearWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = err_to_rgy(cudaMemsetAsync(m_telecineSuperRaw[bufIndex]->ptr, 0, rawBytes, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM telecine-super raw buffer %d: %s.\n"), bufIndex, get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent clearEvent;
        sts = kfmRecordEvent(stream, &clearEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        sts = kfmWaitEvents(stream, { clearEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_analyze_plane((uint8_t *)m_telecineSuperRaw[bufIndex]->ptr, rawPitch,
            &src0Plane, &src1Plane, gridWidth, gridHeight, kfmFrameParity(&src0->frame->frame),
            pixelStep, pixelOffset, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze (telecine-super raw, source %d): %s.\n"), sourceIndex, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, rawEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    };

    auto getSuperSourcePlanes = [&](const int sourceIndex, const RGY_PLANE plane, const int pixelStep,
                                    const int gridWidth, const int gridHeight,
                                    RGYFrameInfo *src0Plane, RGYFrameInfo *src1Plane,
                                    const KfmCachedSource **src0, const KfmCachedSource **src1) -> RGY_ERR {
        *src0 = sourceAt(sourceIndex);
        *src1 = sourceAt(sourceIndex + 1);
        if (!*src0 || !(*src0)->frame || !(*src0)->frame->frame.ptr[0]
            || !*src1 || !(*src1)->frame || !(*src1)->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        *src0Plane = getPlane(&(*src0)->frame->frame, plane);
        *src1Plane = getPlane(&(*src1)->frame->frame, plane);
        if (!src0Plane->ptr[0] || !src1Plane->ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        if (src0Plane->width != src1Plane->width || src0Plane->height != src1Plane->height) {
            return RGY_ERR_INVALID_CALL;
        }
        const int srcLogicalWidth = src0Plane->width / pixelStep;
        const int srcGridWidth = srcLogicalWidth >> 2;
        const int srcGridHeight = src0Plane->height >> 2;
        if (srcLogicalWidth <= 0 || (srcLogicalWidth & 3) != 0 || srcGridWidth != gridWidth || srcGridHeight != gridHeight) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super source size (source %d, plane %d, logical %dx%d, grid %dx%d, expected %dx%d).\n"),
                sourceIndex, (int)plane, srcLogicalWidth, src0Plane->height, srcGridWidth, srcGridHeight, gridWidth, gridHeight);
            return RGY_ERR_INVALID_PARAM;
        }
        return RGY_ERR_NONE;
    };

    auto renderCleanSuperFused = [&](const int prevSourceIndex, const int curSourceIndex, const RGY_PLANE plane, const int pixelStep, const int pixelOffset,
                                     RGYFrameInfo& dst, const int widthPairs, const int logicalHeight, const int field,
                                     const int cleanThresh, const int maxMode,
                                     const int dstStep, const int dstOffset,
                                     const std::vector<RGYCudaEvent>& baseWaitEvents, RGYCudaEvent *cleanEvent) -> RGY_ERR {
        RGYFrameInfo prevSrc0Plane, prevSrc1Plane, curSrc0Plane, curSrc1Plane;
        const KfmCachedSource *prevSrc0 = nullptr, *prevSrc1 = nullptr, *curSrc0 = nullptr, *curSrc1 = nullptr;
        auto sts = getSuperSourcePlanes(prevSourceIndex, plane, pixelStep, widthPairs, logicalHeight,
            &prevSrc0Plane, &prevSrc1Plane, &prevSrc0, &prevSrc1);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = getSuperSourcePlanes(curSourceIndex, plane, pixelStep, widthPairs, logicalHeight,
            &curSrc0Plane, &curSrc1Plane, &curSrc0, &curSrc1);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        std::vector<RGYCudaEvent> cleanWaitEvents = baseWaitEvents;
        for (const auto *src : { prevSrc0, prevSrc1, curSrc0, curSrc1 }) {
            if (src && src->event() != nullptr) {
                cleanWaitEvents.push_back(src->event);
            }
        }
        sts = kfmWaitEvents(stream, cleanWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_clean_super_direct_max_plane(&dst,
            &prevSrc0Plane, &prevSrc1Plane, kfmFrameParity(&prevSrc0->frame->frame),
            &curSrc0Plane, &curSrc1Plane, kfmFrameParity(&curSrc0->frame->frame),
            widthPairs, logicalHeight, field & 1, cleanThresh, maxMode,
            dstStep, dstOffset, pixelStep, pixelOffset, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_clean_super_direct_max (field %d, plane %d): %s.\n"), field, (int)plane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, cleanEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    };

    const bool useFusedCleanSuper = kfmUseFusedCleanSuper();

    for (int iplane = 0; iplane < targetPlanes; iplane++) {
        const bool interleavedChroma = interleavedUV && iplane > 0;
        const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
        auto dst = getPlane(pOutputFrame, plane);
        if (!dst.ptr[0]) {
            continue;
        }
        const int dstStep = interleavedChroma ? 2 : 1;
        const int dstOffset = interleavedChroma ? iplane - 1 : 0;
        const int logicalWidth = dst.width / dstStep;
        const int logicalHeight = dst.height;
        const int widthPairs = logicalWidth >> 1;
        if (logicalWidth <= 0 || logicalHeight <= 0 || (logicalWidth & 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super output plane size (plane %d, logical %dx%d).\n"),
                iplane, logicalWidth, logicalHeight);
            return RGY_ERR_INVALID_PARAM;
        }

        const int rawPitch = widthPairs * (int)sizeof(uchar2);
        const int gridHeight = logicalHeight;
        const int cleanThresh = (iplane > 0) ? KFM_CLEAN_THRESH_C : KFM_CLEAN_THRESH_Y;
        const int pixelStep = interleavedChroma ? 2 : 1;
        const int pixelOffset = interleavedChroma ? iplane - 1 : 0;
        bool firstWrite = true;
        for (int field = firstSuperField; field <= lastSuperField; field++) {
            const int curSourceIndex = kfmFloorDiv2(field);
            const int prevSourceIndex = kfmFloorDiv2(field - 1);
            std::vector<RGYCudaEvent> fieldWaitEvents = wait_events;
            if (prevEvent() != nullptr) {
                fieldWaitEvents.push_back(prevEvent);
            }

            RGYCudaEvent cleanEvent;
            auto sts = RGY_ERR_NONE;
            if (useFusedCleanSuper) {
                sts = renderCleanSuperFused(prevSourceIndex, curSourceIndex, plane, pixelStep, pixelOffset, dst,
                    widthPairs, logicalHeight, field, cleanThresh, firstWrite ? 0 : 1, dstStep, dstOffset, fieldWaitEvents, &cleanEvent);
            } else {
                RGYCudaEvent rawPrevEvent;
                sts = renderRawSuper(0, prevSourceIndex, plane, pixelStep, pixelOffset, rawPitch, widthPairs, gridHeight, fieldWaitEvents, &rawPrevEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                RGYCudaEvent rawCurEvent;
                sts = renderRawSuper(1, curSourceIndex, plane, pixelStep, pixelOffset, rawPitch, widthPairs, gridHeight, fieldWaitEvents, &rawCurEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                sts = kfmWaitEvents(stream, { rawPrevEvent, rawCurEvent });
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                sts = run_kfm_clean_separated_super_max_plane(&dst,
                    (const uint8_t *)m_telecineSuperRaw[0]->ptr,
                    (const uint8_t *)m_telecineSuperRaw[1]->ptr,
                    rawPitch,
                    widthPairs, logicalHeight,
                    field & 1,
                    cleanThresh,
                    firstWrite ? 0 : 1,
                    dstStep, dstOffset, stream);
                if (sts == RGY_ERR_NONE) {
                    sts = kfmRecordEvent(stream, &cleanEvent);
                }
            }
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at KFM clean super render (field %d, plane %d): %s.\n"), field, iplane, get_err_mes(sts));
                return sts;
            }
            prevEvent = cleanEvent;
            firstWrite = false;
        }
    }

    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    const auto *propSource = sourceAt(propSourceIndex);
    if (propSource && propSource->frame) {
        copyFramePropWithoutRes(pOutputFrame, &propSource->frame->frame);
    }
    pOutputFrame->inputFrameId = outputFrameId;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderTelecineSuper24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !m_analyzer) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p super frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    const int firstSource = (firstField & ~1) >> 1;
    const int lastField = firstField + info.numFields - 2;
    RGYCudaEvent superEvent;
    auto sts = renderCleanSuperFields(pOutputFrame, firstField, lastField, firstSource, frame24Index, drain, stream, wait_events, &superEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && superEvent() != nullptr) {
        *event = superEvent;
    }
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump("telecine-super", pOutputFrame, &result);
    const auto dumpSts = dumpStageFrame("telecine-super", pOutputFrame, frame24Index, stream,
        (superEvent() != nullptr) ? std::vector<RGYCudaEvent>{ superEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderSuper30(RGYFrameInfo *pOutputFrame, int frame30Index, bool drain,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    RGYCudaEvent superEvent;
    const int field = frame30Index * 2;
    auto sts = renderCleanSuperFields(pOutputFrame, field, field, frame30Index, frame30Index, drain, stream, wait_events, &superEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && superEvent() != nullptr) {
        *event = superEvent;
    }
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump("super30", pOutputFrame);
    const auto dumpSts = dumpStageFrame("super30", pOutputFrame, frame30Index, stream,
        (superEvent() != nullptr) ? std::vector<RGYCudaEvent>{ superEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::removeCombeFields(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeintFrame, const RGYFrameInfo *pTelecineSuperFrame,
    int firstField, int fieldCount, int stageFrameIndex, const char *stageName,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !pDeintFrame || !pTelecineSuperFrame
        || fieldCount <= 0 || fieldCount > 6 || !stageName) {
        return RGY_ERR_INVALID_CALL;
    }
    const int fieldBase = firstField & ~1;
    const int firstSource = kfmFloorDiv2(fieldBase);
    const int lastField = firstField + fieldCount - 1;
    const int lastSourceOffset = std::max(0, ((lastField & ~1) - fieldBase) >> 1);
    const auto *paritySource = findSourceByIndex(firstSource);
    if (!paritySource || !paritySource->frame || !paritySource->frame->frame.ptr[0]) {
        return RGY_ERR_MORE_DATA;
    }
    std::array<const KfmCachedSource *, 3> teleSrc = {};
    for (int i = 0; i < (int)teleSrc.size(); i++) {
        const int sourceIndex = firstSource + std::min(i, lastSourceOffset);
        teleSrc[i] = findSourceByIndex(sourceIndex);
        if (!teleSrc[i] || !teleSrc[i]->paddedFrame || !teleSrc[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }
    std::vector<RGYCudaEvent> sourceWaitEvents = wait_events;
    for (const auto *s : teleSrc) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYCudaEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src = getPlane(pDeintFrame, (RGY_PLANE)iplane);
        auto combe = getPlane(pTelecineSuperFrame, (RGY_PLANE)iplane);
        auto telePlane0 = getPlane(&teleSrc[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto telePlane1 = getPlane(&teleSrc[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        auto telePlane2 = getPlane(&teleSrc[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int teleSrcYOffset = (telePlane0.height - dst.height) >> 1;
        if (telePlane0.width != dst.width || telePlane1.width != dst.width || telePlane2.width != dst.width
            || telePlane0.height != telePlane1.height || telePlane0.height != telePlane2.height
            || telePlane0.height != dst.height + teleSrcYOffset * 2 || teleSrcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, telePlane0.width, telePlane0.height, telePlane1.width, telePlane1.height, telePlane2.width, telePlane2.height, teleSrcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        const bool chroma = iplane > 0;
        const int threshold = (chroma ? KFM_REMOVE_COMBE_THRESH_C : KFM_REMOVE_COMBE_THRESH_Y) * kfmDepthScale(dst.csp);
        sts = run_kfm_remove_combe_binomial_plane(&dst, &src, &combe,
            &telePlane0, &telePlane1, &telePlane2,
            threshold,
            1, 0, 1, 0,
            teleSrcYOffset, firstField, fieldCount, kfmFrameParity(&paritySource->frame->frame), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_remove_combe_binomial (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pDeintFrame);
    pOutputFrame->dataList = pDeintFrame->dataList;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump(stageName, pOutputFrame);
    const auto dumpSts = dumpStageFrame(stageName, pOutputFrame, stageFrameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::removeCombe24(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint24Frame, const RGYFrameInfo *pTelecineSuperFrame, int frame24Index,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !pDeint24Frame || !pTelecineSuperFrame || !m_analyzer) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    return removeCombeFields(pOutputFrame, pDeint24Frame, pTelecineSuperFrame,
        firstField, info.numFields, frame24Index, "remove-combe", stream, wait_events, event);
}

RGY_ERR NVEncFilterKfm::patchCombe(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame,
    int frameIndex, const char *stageName, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !pBaseFrame || !pPatchFrame || !pMaskFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    RGYCudaEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto base = getPlane(pBaseFrame, (RGY_PLANE)iplane);
        auto patch = getPlane(pPatchFrame, (RGY_PLANE)iplane);
        auto mask = getPlane(pMaskFrame, (RGY_PLANE)iplane);
        if (dst.width != base.width || dst.height != base.height
            || dst.width != patch.width || dst.height != patch.height
            || dst.width != mask.width || dst.height != mask.height) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM patch-combe plane size (plane %d).\n"), iplane);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_patch_combe_plane(&dst, &base, &patch, &mask, 0, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_patch_combe (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pBaseFrame);
    pOutputFrame->dataList = pBaseFrame->dataList;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    const char *stage = (stageName && stageName[0]) ? stageName : "patch-combe";
    writeFrameInfoDump(stage, pOutputFrame);
    const auto dumpSts = dumpStageFrame(stage, pOutputFrame, frameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::ensureMaskBranchFrames(RGYFrameInfo **ppSwitchFlagFrame, RGYFrameInfo **ppContainsCombeFrame, RGYFrameInfo **ppCombeMaskFrame,
    const RGYFrameInfo *pTelecineSuperFrame, const TCHAR *stageLabel) {
    if (!ppSwitchFlagFrame || !ppContainsCombeFrame || !ppCombeMaskFrame || !pTelecineSuperFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto superY = getPlane(pTelecineSuperFrame, RGY_PLANE_Y);
    const int innerWidth = superY.width >> 2;
    const int innerHeight = superY.height >> 1;
    if (innerWidth <= 0 || innerHeight <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM %s mask source size (%dx%d).\n"), stageLabel ? stageLabel : _T(""), superY.width, superY.height);
        return RGY_ERR_INVALID_PARAM;
    }

    const int index = m_maskBranchBufferIndex++ & 3;
    auto switchInfo = prm->frameOut;
    switchInfo.width = innerWidth + 8;
    switchInfo.height = innerHeight + 4;
    switchInfo.csp = RGY_CSP_Y8;
    switchInfo.bitdepth = RGY_CSP_BIT_DEPTH[RGY_CSP_Y8];
    if (!m_switchFlagFrames[index]
        || m_switchFlagFrames[index]->frame.width != switchInfo.width
        || m_switchFlagFrames[index]->frame.height != switchInfo.height
        || m_switchFlagFrames[index]->frame.csp != switchInfo.csp) {
        auto frame = std::make_unique<CUFrameBuf>(switchInfo);
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s switch-flag-min frame: %s.\n"), stageLabel ? stageLabel : _T(""), get_err_mes(sts));
            return sts;
        }
        m_switchFlagFrames[index] = std::move(frame);
    }

    auto containsInfo = switchInfo;
    containsInfo.width = 4;
    containsInfo.height = 1;
    if (!m_containsCombeFrames[index]
        || m_containsCombeFrames[index]->frame.width != containsInfo.width
        || m_containsCombeFrames[index]->frame.height != containsInfo.height
        || m_containsCombeFrames[index]->frame.csp != containsInfo.csp) {
        auto frame = std::make_unique<CUFrameBuf>(containsInfo);
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s contains-combe frame: %s.\n"), stageLabel ? stageLabel : _T(""), get_err_mes(sts));
            return sts;
        }
        m_containsCombeFrames[index] = std::move(frame);
    }

    auto combeInfo = prm->frameOut;
    if (!m_combeMaskFrames[index]
        || m_combeMaskFrames[index]->frame.width != combeInfo.width
        || m_combeMaskFrames[index]->frame.height != combeInfo.height
        || m_combeMaskFrames[index]->frame.csp != combeInfo.csp) {
        auto frame = std::make_unique<CUFrameBuf>(combeInfo);
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s combe-mask-min frame: %s.\n"), stageLabel ? stageLabel : _T(""), get_err_mes(sts));
            return sts;
        }
        m_combeMaskFrames[index] = std::move(frame);
    }

    *ppSwitchFlagFrame = &m_switchFlagFrames[index]->frame;
    *ppContainsCombeFrame = &m_containsCombeFrames[index]->frame;
    *ppCombeMaskFrame = &m_combeMaskFrames[index]->frame;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderMaskBranch(RGYFrameInfo *pSwitchFlagFrame, RGYFrameInfo *pContainsCombeFrame, RGYFrameInfo *pCombeMaskFrame,
    const RGYFrameInfo *pTelecineSuperPrevFrame, const RGYFrameInfo *pTelecineSuperFrame, const RGYFrameInfo *pTelecineSuperNextFrame,
    const char *switchFlagStage, const char *containsCombeStage, const char *combeMaskStage,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, uint32_t *containsCombeCount) {
    if (!pSwitchFlagFrame || !pContainsCombeFrame || !pCombeMaskFrame
        || !pTelecineSuperPrevFrame || !pTelecineSuperFrame || !pTelecineSuperNextFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto superPrevY = getPlane(pTelecineSuperPrevFrame, RGY_PLANE_Y);
    const auto superY = getPlane(pTelecineSuperFrame, RGY_PLANE_Y);
    const auto superNextY = getPlane(pTelecineSuperNextFrame, RGY_PLANE_Y);
    const bool superInterleavedUV = kfmCspHasInterleavedUV(pTelecineSuperFrame->csp);
    const auto superPrevUV = (RGY_CSP_PLANES[pTelecineSuperPrevFrame->csp] > 1) ? getPlane(pTelecineSuperPrevFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superUV = (RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 1) ? getPlane(pTelecineSuperFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superNextUV = (RGY_CSP_PLANES[pTelecineSuperNextFrame->csp] > 1) ? getPlane(pTelecineSuperNextFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superPrevV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperPrevFrame->csp] > 2) ? getPlane(pTelecineSuperPrevFrame, RGY_PLANE_V) : superPrevUV;
    const auto superV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 2) ? getPlane(pTelecineSuperFrame, RGY_PLANE_V) : superUV;
    const auto superNextV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperNextFrame->csp] > 2) ? getPlane(pTelecineSuperNextFrame, RGY_PLANE_V) : superNextUV;
    auto switchY = getPlane(pSwitchFlagFrame, RGY_PLANE_Y);
    const int superYPitch = superY.pitch[0];
    if (superPrevY.pitch[0] != superYPitch || superNextY.pitch[0] != superYPitch) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (Y plane: prev %d, cur %d, next %d).\n"),
            superPrevY.pitch[0], superYPitch, superNextY.pitch[0]);
        return RGY_ERR_INVALID_PARAM;
    }
    const int superUVPitch = superUV.ptr[0] ? superUV.pitch[0] : superYPitch;
    const int superVPitch = superV.ptr[0] ? superV.pitch[0] : superUVPitch;
    if (superUV.ptr[0]) {
        if (superPrevUV.pitch[0] != superUVPitch || superNextUV.pitch[0] != superUVPitch) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (UV plane: prev %d, cur %d, next %d).\n"),
                superPrevUV.pitch[0], superUVPitch, superNextUV.pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 2) {
            if (superPrevV.pitch[0] != superVPitch || superNextV.pitch[0] != superVPitch) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (U/V plane: prev %d/%d, cur %d/%d, next %d/%d).\n"),
                    superPrevUV.pitch[0], superPrevV.pitch[0], superUVPitch, superV.pitch[0], superNextUV.pitch[0], superNextV.pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }
    const int innerWidth = superY.width >> 2;
    const int innerHeight = superY.height >> 1;
    const int combeWidth = superY.width >> 1;
    const int combeHeight = superY.height;
    const int combeCWidth = superUV.ptr[0] ? (superInterleavedUV ? (superUV.width >> 2) : (superUV.width >> 1)) : 0;
    const int combeCHeight = superUV.ptr[0] ? superUV.height : 0;
    if (switchY.width != innerWidth + 8 || switchY.height != innerHeight + 4) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag-min size (super %dx%d, flag %dx%d).\n"),
            superY.width, superY.height, switchY.width, switchY.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (superPrevY.width != superY.width || superPrevY.height != superY.height
        || superNextY.width != superY.width || superNextY.height != superY.height
        || combeWidth <= 0 || combeHeight <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag super triplet size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int maskDumpFrameIndex = pTelecineSuperFrame->inputFrameId >= 0 ? pTelecineSuperFrame->inputFrameId : m_timecodeFrameIndex;

    const int combeYPitch = combeWidth;
    const int combeCPitch = std::max(1, combeCWidth);
    const int flagPitch = switchY.width;
    const size_t combeYBytes = (size_t)combeYPitch * combeHeight;
    const size_t combeCBytes = (size_t)combeCPitch * std::max(1, combeCHeight);
    const size_t flagBytes = (size_t)flagPitch * switchY.height;
    const std::array<size_t, 4> workBytes = { std::max(combeYBytes, flagBytes), std::max(combeCBytes, flagBytes), flagBytes, flagBytes };
    for (int i = 0; i < (int)workBytes.size(); i++) {
        if (!m_switchFlagWork[i] || m_switchFlagWork[i]->nSize < workBytes[i]) {
            auto work = std::make_unique<CUMemBuf>(workBytes[i]);
            const auto allocSts = work->alloc();
            if (allocSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM switch-flag work buffer %d: %s.\n"), i, get_err_mes(allocSts));
                return allocSts;
            }
            m_switchFlagWork[i] = std::move(work);
        }
    }

    auto workWaitEvents = wait_events;
    if (m_switchFlagWorkEvent() != nullptr) {
        workWaitEvents.push_back(m_switchFlagWorkEvent);
    }

    auto dumpSwitchWorkFrame = [&](const char *stage, const std::unique_ptr<CUMemBuf>& work, int width, int height, int pitch, const RGYCudaEvent& waitEvent) -> RGY_ERR {
        if (!stageDumpRequested(maskDumpFrameIndex) || !work) {
            return RGY_ERR_NONE;
        }
        auto sts = kfmWaitEvents(stream, { waitEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_copy_u8_buffer_to_plane(&switchY, (const uint8_t *)work->ptr, pitch, width, height, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_copy_u8_buffer_to_plane (%s): %s.\n"), char_to_tstring(stage).c_str(), get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent copyEvent;
        sts = kfmRecordEvent(stream, &copyEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return dumpStageFrame(stage, pSwitchFlagFrame, maskDumpFrameIndex, stream, { copyEvent });
    };

    auto sts = kfmWaitEvents(stream, workWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = run_kfm_switch_flag_combe_min((uint8_t *)m_switchFlagWork[0]->ptr, combeYPitch,
        (uint8_t *)m_switchFlagWork[1]->ptr, combeCPitch,
        &superPrevY, &superY, &superNextY,
        superPrevUV.ptr[0] ? &superPrevUV : nullptr,
        superUV.ptr[0] ? &superUV : nullptr,
        superNextUV.ptr[0] ? &superNextUV : nullptr,
        superPrevV.ptr[0] ? &superPrevV : nullptr,
        superV.ptr[0] ? &superV : nullptr,
        superNextV.ptr[0] ? &superNextV : nullptr,
        combeWidth, combeHeight,
        combeCWidth, combeCHeight,
        superUV.ptr[0] ? 1 : 0,
        superInterleavedUV ? 1 : 0, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_combe_min: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent combeEvent;
    sts = kfmRecordEvent(stream, &combeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = kfmWaitEvents(stream, { combeEvent });
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = run_kfm_switch_flag_from_combe_min((uint8_t *)m_switchFlagWork[2]->ptr, flagPitch,
        (uint8_t *)m_switchFlagWork[3]->ptr, flagPitch,
        (const uint8_t *)m_switchFlagWork[0]->ptr, combeYPitch,
        (const uint8_t *)m_switchFlagWork[1]->ptr, combeCPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight,
        combeWidth, combeHeight,
        combeCWidth, combeCHeight, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_from_combe_min: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent fromCombeEvent;
    sts = kfmRecordEvent(stream, &fromCombeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-from-combe-y", m_switchFlagWork[2], switchY.width, switchY.height, flagPitch, fromCombeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-from-combe-c", m_switchFlagWork[3], switchY.width, switchY.height, flagPitch, fromCombeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto boxPass = [&](int dstIndex, int srcIndex, const RGYCudaEvent& waitEvent, const char *errStage, RGYCudaEvent *outEvent) -> RGY_ERR {
        auto boxSts = kfmWaitEvents(stream, { waitEvent });
        if (boxSts != RGY_ERR_NONE) {
            return boxSts;
        }
        boxSts = run_kfm_switch_flag_box3x3_min((uint8_t *)m_switchFlagWork[dstIndex]->ptr, flagPitch,
            (const uint8_t *)m_switchFlagWork[srcIndex]->ptr, flagPitch,
            switchY.width, switchY.height,
            innerWidth, innerHeight, stream);
        if (boxSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_box3x3_min (%s): %s.\n"), char_to_tstring(errStage).c_str(), get_err_mes(boxSts));
            return boxSts;
        }
        return kfmRecordEvent(stream, outEvent);
    };

    RGYCudaEvent boxY0Event;
    sts = boxPass(0, 2, fromCombeEvent, "Y pass 0", &boxY0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box1-y", m_switchFlagWork[0], switchY.width, switchY.height, flagPitch, boxY0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent boxY1Event;
    sts = boxPass(2, 0, boxY0Event, "Y pass 1", &boxY1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box2-y", m_switchFlagWork[2], switchY.width, switchY.height, flagPitch, boxY1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent boxC0Event;
    sts = boxPass(1, 3, fromCombeEvent, "C pass 0", &boxC0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box1-c", m_switchFlagWork[1], switchY.width, switchY.height, flagPitch, boxC0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent boxC1Event;
    sts = boxPass(3, 1, boxC0Event, "C pass 1", &boxC1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box2-c", m_switchFlagWork[3], switchY.width, switchY.height, flagPitch, boxC1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = kfmWaitEvents(stream, { boxY1Event, boxC1Event });
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYCudaEvent switchEvent;
    if (kfmUseFusedSwitchFlagBinaryExtend()) {
        sts = run_kfm_switch_flag_binary_extend_hv_min(&switchY,
            (const uint8_t *)m_switchFlagWork[2]->ptr, flagPitch,
            (const uint8_t *)m_switchFlagWork[3]->ptr, flagPitch,
            innerWidth, innerHeight,
            KFM_SWITCH_FLAG_THRESH_Y, KFM_SWITCH_FLAG_THRESH_C, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_binary_extend_hv_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &switchEvent);
    } else {
        sts = run_kfm_switch_flag_binary_min(&switchY,
            (const uint8_t *)m_switchFlagWork[2]->ptr, flagPitch,
            (const uint8_t *)m_switchFlagWork[3]->ptr, flagPitch,
            innerWidth, innerHeight,
            KFM_SWITCH_FLAG_THRESH_Y, KFM_SWITCH_FLAG_THRESH_C, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_binary_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent binaryEvent;
        sts = kfmRecordEvent(stream, &binaryEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = kfmWaitEvents(stream, { binaryEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_switch_flag_extend_h_min((uint8_t *)m_switchFlagWork[0]->ptr, flagPitch, &switchY,
            innerWidth, innerHeight, 4, 2, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_extend_h_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent extendHEvent;
        sts = kfmRecordEvent(stream, &extendHEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = kfmWaitEvents(stream, { extendHEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_switch_flag_extend_v_min(&switchY, (const uint8_t *)m_switchFlagWork[0]->ptr, flagPitch,
            innerWidth, innerHeight, 4, 2, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_extend_v_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &switchEvent);
    }
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_switchFlagWorkEvent = switchEvent;

    if (!m_containsCombeCount || m_containsCombeCount->nSize < sizeof(uint32_t)) {
        auto count = std::make_unique<CUMemBuf>(sizeof(uint32_t));
        const auto allocSts = count->alloc();
        if (allocSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM contains-combe count buffer: %s.\n"), get_err_mes(allocSts));
            return allocSts;
        }
        m_containsCombeCount = std::move(count);
    }
    sts = run_kfm_contains_combe_init((uint32_t *)m_containsCombeCount->ptr, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_init: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent initEvent;
    sts = kfmRecordEvent(stream, &initEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = kfmWaitEvents(stream, { switchEvent, initEvent });
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = run_kfm_contains_combe_count(&switchY, (uint32_t *)m_containsCombeCount->ptr, 128, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_count: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent countEvent;
    sts = kfmRecordEvent(stream, &countEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (containsCombeCount) {
        *containsCombeCount = 0;
        sts = kfmWaitEvents(stream, { countEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = err_to_rgy(cudaMemcpyAsync(containsCombeCount, m_containsCombeCount->ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to submit KFM contains-combe count readback: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = err_to_rgy(cudaStreamSynchronize(stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM contains-combe count readback: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    auto containsY = getPlane(pContainsCombeFrame, RGY_PLANE_Y);
    sts = kfmWaitEvents(stream, { countEvent });
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = run_kfm_contains_combe_mark(&containsY, (const uint32_t *)m_containsCombeCount->ptr, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_mark: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent markEvent;
    sts = kfmRecordEvent(stream, &markEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYCudaEvent prevEvent = markEvent;
    const int planes = RGY_CSP_PLANES[pCombeMaskFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pCombeMaskFrame->csp);
    for (int iplane = 0; iplane < planes; iplane++) {
        const bool interleavedChroma = interleavedUV && iplane > 0;
        const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
        auto dst = getPlane(pCombeMaskFrame, plane);
        const int step = interleavedChroma ? 2 : 1;
        const int offset = interleavedChroma ? iplane - 1 : 0;
        const int logicalWidth = dst.width / step;
        const int logicalHeight = dst.height;
        const int scaleX = logicalWidth / innerWidth;
        const int scaleY = logicalHeight / innerHeight;
        const int shiftX = kfmPow2Shift(scaleX);
        const int shiftY = kfmPow2Shift(scaleY);
        if (logicalWidth <= 0 || logicalHeight <= 0 || logicalWidth != innerWidth * scaleX || logicalHeight != innerHeight * scaleY || shiftX < 0 || shiftY < 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported KFM combe-mask-min scale (plane %d, dst %dx%d, flag inner %dx%d).\n"),
                iplane, logicalWidth, logicalHeight, innerWidth, innerHeight);
            return RGY_ERR_INVALID_PARAM;
        }
        sts = kfmWaitEvents(stream, { prevEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_combe_mask_resize_bilinear_min_plane(&dst, &switchY,
            step, offset,
            scaleX, shiftX,
            scaleY, shiftY,
            innerWidth, innerHeight, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_combe_mask_resize_bilinear_min (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    writeFrameInfoDump(switchFlagStage, pSwitchFlagFrame);
    auto dumpSts = dumpStageFrame(switchFlagStage, pSwitchFlagFrame, maskDumpFrameIndex, stream, { switchEvent });
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    writeFrameInfoDump(containsCombeStage, pContainsCombeFrame);
    dumpSts = dumpStageFrame(containsCombeStage, pContainsCombeFrame, maskDumpFrameIndex, stream, { markEvent });
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    writeFrameInfoDump(combeMaskStage, pCombeMaskFrame);
    dumpSts = dumpStageFrame(combeMaskStage, pCombeMaskFrame, maskDumpFrameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::runNrFilter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!ppOutputFrame) {
        return RGY_ERR_INVALID_PARAM;
    }
    *ppOutputFrame = nullptr;
    if (!m_nrFilter) {
        *ppOutputFrame = pInputFrame;
        return RGY_ERR_NONE;
    }

    auto sts = kfmWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYFrameInfo *nrOutFrames[2] = {};
    int nrOutNum = 0;
    sts = m_nrFilter->filter(pInputFrame, nrOutFrames, &nrOutNum, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run KFM NR Auto degrain filter: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if (nrOutNum > 1) {
        AddMessage(RGY_LOG_ERROR, _T("KFM NR Auto degrain returned unexpected output count %d.\n"), nrOutNum);
        return RGY_ERR_INVALID_CALL;
    }
    *ppOutputFrame = (nrOutNum > 0) ? nrOutFrames[0] : nullptr;
    if (*ppOutputFrame) {
        sts = kfmRecordEvent(stream, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::emitOutputFrame(RGYFrameInfo *pFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const RGYCudaEvent &frameEvent, RGYCudaEvent *event) {
    if (!pFrame || !ppOutputFrames || !pOutputFrameNum) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo *emitFrame = pFrame;
    RGYCudaEvent nrEvent;
    if (m_nrFilter) {
        std::vector<RGYCudaEvent> nrWaitEvents;
        if (frameEvent() != nullptr) {
            nrWaitEvents.push_back(frameEvent);
        }
        auto sts = runNrFilter(pFrame, &emitFrame, stream, nrWaitEvents, &nrEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!emitFrame) {
            if (event && nrEvent() != nullptr) {
                *event = nrEvent;
            }
            return RGY_ERR_NONE;
        }
    }
    auto outputFrame = nextOutputFrame();
    if (!outputFrame || !outputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    std::vector<RGYCudaEvent> copyWaitEvents;
    if (nrEvent() != nullptr) {
        copyWaitEvents.push_back(nrEvent);
    } else if (frameEvent() != nullptr) {
        copyWaitEvents.push_back(frameEvent);
    }
    RGYCudaEvent outputEvent;
    if (outputFrame != emitFrame) {
        auto sts = kfmWaitEvents(stream, copyWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = copyFrameAsync(outputFrame, emitFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM output frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &outputEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        copyFramePropWithoutRes(outputFrame, emitFrame);
    } else if (!copyWaitEvents.empty()) {
        outputEvent = copyWaitEvents.back();
    }
    kfmEraseInternalFrameData(outputFrame);
    if (event) {
        if (outputEvent() != nullptr) {
            *event = outputEvent;
        }
    }
    ppOutputFrames[(*pOutputFrameNum)++] = outputFrame;
    writeFrameInfoDump("output", outputFrame);
    writeFrameTimecode(outputFrame);
    m_timecodeFrameIndex++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::queueVfrOutputFrame(const RGYFrameInfo *pFrame, cudaStream_t stream, const RGYCudaEvent &frameEvent) {
    if (!pFrame || !pFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    KfmPendingVfrOutput pending;
    pending.frame = acquireKfmFrame(*pFrame, _T("VFR delayed output"));
    if (!pending.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    std::vector<RGYCudaEvent> copyWaitEvents;
    if (frameEvent() != nullptr) {
        copyWaitEvents.push_back(frameEvent);
    }
    auto sts = kfmWaitEvents(stream, copyWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = copyFrameAsync(&pending.frame->frame, pFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR delayed output frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = kfmRecordEvent(stream, &pending.event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    copyFramePropWithoutRes(&pending.frame->frame, pFrame);
    m_pendingVfrOutputs.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::emitPendingVfrOutput(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    if (m_pendingVfrOutputs.empty()) {
        return RGY_ERR_NONE;
    }
    auto pending = std::move(m_pendingVfrOutputs.front());
    m_pendingVfrOutputs.pop_front();
    if (!pending.frame || !pending.frame->frame.ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return emitOutputFrame(&pending.frame->frame, ppOutputFrames, pOutputFrameNum, stream, pending.event, event);
}

RGY_ERR NVEncFilterKfm::emitPendingVfrOutputs(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event, int keepFrames) {
    keepFrames = std::max(0, keepFrames);
    const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
    while ((int)m_pendingVfrOutputs.size() > keepFrames && *pOutputFrameNum < maxOutputFrames) {
        const int outputFrameNumBefore = *pOutputFrameNum;
        auto sts = emitPendingVfrOutput(ppOutputFrames, pOutputFrameNum, stream, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (*pOutputFrameNum == outputFrameNumBefore) {
            break;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::drainNrFilter(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    if (!m_nrFilter || !ppOutputFrames || !pOutputFrameNum) {
        return RGY_ERR_NONE;
    }
    RGYFrameInfo *emitFrame = nullptr;
    RGYCudaEvent nrEvent;
    auto sts = runNrFilter(nullptr, &emitFrame, stream, {}, &nrEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!emitFrame) {
        return RGY_ERR_NONE;
    }
    if (event && nrEvent() != nullptr) {
        *event = nrEvent;
    }
    kfmEraseInternalFrameData(emitFrame);
    ppOutputFrames[(*pOutputFrameNum)++] = emitFrame;
    writeFrameInfoDump("output", emitFrame);
    writeFrameTimecode(emitFrame);
    m_timecodeFrameIndex++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::processMainRtgmcOutputs(const NVEncFilterParamKfm& prm, RGYFrameInfo **rtgmcOutFrames, int rtgmcOutNum,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (rtgmcOutNum <= 0) {
        return RGY_ERR_NONE;
    }
    std::vector<RGYCudaEvent> mergeWaitEvents = wait_events;
    RGYCudaEvent staticEvent;
    auto sts = analyzeStaticFlag(stream, wait_events, &staticEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (staticEvent() != nullptr) {
        mergeWaitEvents.push_back(staticEvent);
    }
    for (int i = 0; i < rtgmcOutNum; i++) {
        auto out = nextWorkFrame();
        if (!out) {
            return RGY_ERR_INVALID_CALL;
        }
        auto frameWaitEvents = mergeWaitEvents;
        const auto source = findSourceFrame(rtgmcOutFrames[i], &frameWaitEvents);
        if (!source) {
            AddMessage(RGY_LOG_ERROR, _T("KFM source frame is missing for output inputFrameId=%d.\n"), rtgmcOutFrames[i]->inputFrameId);
            return RGY_ERR_INVALID_CALL;
        }
        RGYCudaEvent mergeEvent;
        sts = mergeStatic(out, rtgmcOutFrames[i], source, stream, frameWaitEvents, &mergeEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm.kfm.ucf) {
            AddMessage(RGY_LOG_ERROR, _T("KFM ucf=true output path is not ported yet.\n"));
            return RGY_ERR_UNSUPPORTED;
        } else {
            sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, stream, mergeEvent, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::drainMainRtgmcBranch(const NVEncFilterParamKfm& prm, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    for (int iter = 0; *pOutputFrameNum == 0 && !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            AddMessage(RGY_LOG_ERROR, _T("KFM main RTGMC drain did not complete after %d iterations.\n"), maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        int rtgmcOutNum = 0;
        RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
        RGYCudaEvent rtgmcEvent;
        auto sts = m_rtgmc->filter(nullptr, rtgmcOutFrames, &rtgmcOutNum, stream, {}, &rtgmcEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        std::vector<RGYCudaEvent> processWaitEvents;
        if (rtgmcEvent() != nullptr) {
            processWaitEvents.push_back(rtgmcEvent);
        }
        sts = processMainRtgmcOutputs(prm, rtgmcOutFrames, rtgmcOutNum, ppOutputFrames, pOutputFrameNum, stream, processWaitEvents, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
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
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_CALL;
    }

    if (m_rtgmc) {
        auto sts = cacheSourceFrame(pInputFrame, stream, {});
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, stream);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (prm->kfm.ucf) {
            AddMessage(RGY_LOG_ERROR, _T("KFM ucf=true RTGMC branches are not ported yet.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        int rtgmcOutNum = 0;
        RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
        RGYCudaEvent rtgmcEvent;
        sts = m_rtgmc->filter(const_cast<RGYFrameInfo *>(pInputFrame), rtgmcOutFrames, &rtgmcOutNum, stream, {}, &rtgmcEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        *pOutputFrameNum = 0;
        std::vector<RGYCudaEvent> processWaitEvents;
        if (rtgmcEvent() != nullptr) {
            processWaitEvents.push_back(rtgmcEvent);
        }
        sts = processMainRtgmcOutputs(*prm, rtgmcOutFrames, rtgmcOutNum, ppOutputFrames, pOutputFrameNum, stream, processWaitEvents, nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) && *pOutputFrameNum == 0 && !m_rtgmc->drainComplete()) {
            sts = drainMainRtgmcBranch(*prm, ppOutputFrames, pOutputFrameNum, stream, nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) && *pOutputFrameNum == 0 && m_rtgmc->drainComplete()) {
            sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, stream, nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return RGY_ERR_NONE;
    }

    AddMessage(RGY_LOG_ERROR, _T("KFM mode %s is not ported yet.\n"), get_cx_desc(list_vpp_kfm_mode, (int)prm->kfm.mode));
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
