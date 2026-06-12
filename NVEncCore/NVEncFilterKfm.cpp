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
static constexpr int KFM_UCF_NOISE_LIMIT_NMIN = 1;
static constexpr int KFM_UCF_NOISE_LIMIT_RANGE = 128;
static constexpr int KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY = 2;
static constexpr int KFM_UCF_LAZY_SOURCE_CACHE_MARGIN = 512;
static constexpr double KFM_UCF_GAUSS_P = 2.5;
static constexpr double KFM_UCF_GAUSS_CROP_EPS = 0.0001;

static double kfmUcfGaussValue(const double value, const double p) {
    const auto param = std::min(std::max(p, 0.1), 100.0);
    return std::pow(2.0, -(param * 0.1) * value * value);
}

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

static bool kfmUcfNoGaussForTest() {
    const char *env = std::getenv("NVENC_KFM_UCF_NO_GAUSS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmUseFusedUcfPreprocess() {
    const char *env = std::getenv("NVENC_KFM_UCF_PREPROCESS_FUSED");
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

static bool kfmForceEagerRtgmc() {
    const char *env = std::getenv("NVENC_KFM_FORCE_EAGER_RTGMC");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmCspHasInterleavedUV(const RGY_CSP csp) {
    return csp == RGY_CSP_NV12 || csp == RGY_CSP_P010
        || csp == RGY_CSP_NV16 || csp == RGY_CSP_P210;
}

static const char *kfmUcfKernelName(VppKfmMode mode) {
    switch (mode) {
    case VppKfmMode::P60:
        return "kernel_kfm_ucf_60";
    case VppKfmMode::P24:
        return "kernel_kfm_ucf_24";
    case VppKfmMode::VFR:
    default:
        return "kernel_kfm_ucf_param";
    }
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

static RGYFrameInfo *kfmDebugStageFrame(VppKfmDebugStage stage, RGYFrameInfo *switchFlag, RGYFrameInfo *containsCombe, RGYFrameInfo *combeMask) {
    switch (stage) {
    case VppKfmDebugStage::SwitchFlag:
        return switchFlag;
    case VppKfmDebugStage::ContainsCombe:
        return containsCombe;
    case VppKfmDebugStage::CombeMask:
        return combeMask;
    case VppKfmDebugStage::None:
    default:
        return nullptr;
    }
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

struct KfmUcfCalcDumpInfo {
    const char *classification;
    double fieldDiff;
    double diff;
};

static double kfmUcfCalcNoiseDiff(const RGYKFM::UCFNoiseMeta& meta, const RGYKFM::DecombUCFParam& param,
    const RGYKFM::NoiseResult *result0, const RGYKFM::NoiseResult *result1, bool second) {
    const double noisepixels = static_cast<double>(meta.noisew) * meta.noiseh;
    const double noisepixelsUV = static_cast<double>(meta.noiseUVw) * meta.noiseUVh * 2.0;

    const double noise_t_y = (second ? result0[0].noise1 : result0[0].noise0) / noisepixels;
    const double noise_t_uv = (second ? result0[1].noise1 : result0[1].noise0) / noisepixelsUV;
    const double noise_b_y = (second ? result1[0].noise0 : result0[0].noise1) / noisepixels;
    const double noise_b_uv = (second ? result1[1].noise0 : result0[1].noise1) / noisepixelsUV;
    const double diff1_y = noise_t_y - noise_b_y;
    const double diff1_uv = noise_t_uv - noise_b_uv;

    double diff1 = 0.0;
    if (param.chroma == 0) {
        diff1 = diff1_y;
    } else if (param.chroma == 1) {
        diff1 = diff1_uv;
    } else {
        diff1 = (diff1_y + diff1_uv) / 2.0;
    }

    const double absdiff1 = std::abs(diff1);
    return absdiff1 < 1.8 ? diff1 * 10.0
        : absdiff1 < 5.0 ? diff1 * 5.0 + (diff1 / absdiff1) * 9.0
        : absdiff1 < 10.0 ? diff1 * 2.0 + (diff1 / absdiff1) * 24.0
        : diff1 + (diff1 / absdiff1) * 34.0;
}

static KfmUcfCalcDumpInfo kfmUcfCalcDumpInfo(const RGYKFM::UCFNoiseMeta& meta, const RGYKFM::NoiseResult *result0,
    const RGYKFM::NoiseResult *result1, bool second) {
    RGYKFM::DecombUCFParam param;
    const auto classification = RGYKFM::CalcDecombUCF(&meta, &param, result0, result1, second);
    const double pixels = static_cast<double>(meta.srcw) * meta.srch;
    const double fieldDiff = (second
        ? static_cast<double>(result0[0].diff1 + result0[1].diff1)
        : static_cast<double>(result0[0].diff0 + result0[1].diff0)) / (6.0 * pixels) * 100.0;
    return { RGYKFM::decombUCFResultToString(classification), fieldDiff, kfmUcfCalcNoiseDiff(meta, param, result0, result1, second) };
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
    m_kfmSourceSlotFree(),
    m_kfmSourceSlotRetired(),
    m_sourceCache(),
    m_deint60Lane(),
    m_before60Lane(),
    m_after60Lane(),
    m_ucfNoiseCache(),
    m_pendingUcfNoiseResults(),
    m_fmCountBufPool(),
    m_ucfNoiseResultBufPool(),
    m_ucfNoiseResultCache(),
    m_pendingUcfNoiseDump(),
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

NVEncFilterKfm::KfmRtgmcLane::KfmRtgmcLane() :
    m_owner(nullptr),
    m_rtgmc(nullptr),
    m_stage(nullptr),
    m_cacheLabel(nullptr),
    m_dumpStaticFlag(false),
    m_cache(),
    m_intermediateGenerations(),
    m_submittedFrames(0),
    m_nextFeedSourceIndex(-1),
    m_nextOutputN60(0),
    m_hotUntilSourceIndex(-1),
    m_cacheFloorN60(0),
    m_feedCount(0),
    m_cacheCopyEvent() {
}

void NVEncFilterKfm::KfmRtgmcLane::init(NVEncFilterKfm *owner, NVEncFilterRtgmc *rtgmc, const char *stage, const TCHAR *cacheLabel, bool dumpStaticFlag) {
    m_owner = owner;
    m_rtgmc = rtgmc;
    m_stage = stage;
    m_cacheLabel = cacheLabel;
    m_dumpStaticFlag = dumpStaticFlag;
    reset();
}

void NVEncFilterKfm::KfmRtgmcLane::clear() {
    m_cache.clear();
    m_intermediateGenerations.clear();
    m_submittedFrames = 0;
    m_cacheCopyEvent = RGYCudaEvent();
}

void NVEncFilterKfm::KfmRtgmcLane::reset() {
    clear();
    m_nextFeedSourceIndex = -1;
    m_nextOutputN60 = 0;
    m_hotUntilSourceIndex = -1;
    m_cacheFloorN60 = 0;
    m_feedCount = 0;
    m_resetCounts.fill(0);
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::feed(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }

    int rtgmcOutNum = 0;
    RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
    RGYCudaEvent rtgmcEvent;
    if (frame && frame->ptr[0]) {
        m_feedCount++;
    }
    auto sts = m_rtgmc->filter(const_cast<RGYFrameInfo *>(frame), rtgmcOutFrames, &rtgmcOutNum, stream, wait_events, &rtgmcEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    std::vector<RGYCudaEvent> cacheWaitEvents;
    if (rtgmcEvent() != nullptr) {
        cacheWaitEvents.push_back(rtgmcEvent);
    }
    for (int i = 0; i < rtgmcOutNum; i++) {
        RGYCudaEvent cacheEvent;
        sts = cacheFrame(rtgmcOutFrames[i], stream, cacheWaitEvents, &cacheEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cacheEvent() != nullptr) {
            m_cacheCopyEvent = cacheEvent;
        }
        if (cachedFrames) {
            (*cachedFrames)++;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::drain(cudaStream_t stream, int maxDrainIterations, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    for (int iter = 0; !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM %S RTGMC drain did not complete after %d iterations.\n"), m_stage, maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        int drainedFrames = 0;
        auto sts = feed(nullptr, stream, {}, &drainedFrames);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cachedFrames) {
            *cachedFrames += drainedFrames;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::drainTo(int n60end, cudaStream_t stream) {
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    const auto maxDrainIterations = std::max(256, m_owner ? m_owner->m_cachedSourceFrames * 4 + 256 : 256);
    for (int iter = 0; m_nextOutputN60 < n60end && !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM %S RTGMC demand drain did not reach n60=%d after %d iterations.\n"),
                m_stage, n60end, maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        auto sts = feed(nullptr, stream, {});
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_nextOutputN60 = m_submittedFrames;
    }
    return (m_nextOutputN60 >= n60end) ? RGY_ERR_NONE : RGY_ERR_MORE_DATA;
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::ensureRange(int n60begin, int n60end, cudaStream_t stream) {
    if (!m_rtgmc || n60begin >= n60end) {
        return RGY_ERR_NONE;
    }
    n60begin = std::max(0, n60begin);
    n60end = std::max(n60begin, n60end);

    int firstMissingN60 = -1;
    for (int n60 = n60begin; n60 < n60end; n60++) {
        if (!find(n60, nullptr)) {
            firstMissingN60 = n60;
            break;
        }
    }
    if (firstMissingN60 < 0) {
        return RGY_ERR_NONE;
    }

    const int sourceBegin = n60begin >> 1;
    const bool cold = m_nextFeedSourceIndex < 0;
    // 欠けている最初のフレームが現在の出力位置より先 (末尾欠けのみ) なら
    // 巻き戻し不要で前進 feed のキャッチアップで埋まる。MORE_DATA 後の
    // リトライをフル reset 扱いにすると、source 到着待ちのたびに
    // re-priming が走ってしまう。
    const bool rewind = m_nextOutputN60 > firstMissingN60;
    // 注意: 「feed 位置が要求開始 source を越えている」ことは reset 理由にしない。
    // RTGMC はパイプライン遅延を持つため feed は出力より常に先行する。出力は
    // feed 順に必ず生成されるので、出力位置が要求以前なら前進 feed で到達できる。
    const bool feedPastRequest = false;
    const bool nextFeedTrimmed = !cold && m_owner && m_nextFeedSourceIndex < m_owner->m_cachedSourceFrames && !m_owner->findSourceByIndexExact(m_nextFeedSourceIndex);
    // 前方の需要は「現在位置からのキャッチアップ feed」と「reset + priming」の
    // 安い方を選ぶ。hot 窓超過を一律 reset にすると、60p サイクルが頻出する
    // コンテンツで数フレームの前進のたびに re-priming が走り大幅に遅くなる。
    const int sourceEndFeed = (n60end - 1) >> 1;
    const int catchupCost = !cold ? std::max(0, sourceEndFeed - m_nextFeedSourceIndex + 1) : INT_MAX;
    const int resetCost = sourceEndFeed - std::max(0, sourceBegin - (m_rtgmc ? m_rtgmc->requiredPrimingSourceFrames() : 0)) + 1;
    const bool farJump = !cold && catchupCost > resetCost;
    if (cold || rewind || feedPastRequest || farJump || nextFeedTrimmed) {
        m_resetCounts[cold ? 0 : (rewind ? 1 : (feedPastRequest ? 2 : (farJump ? 3 : 4)))]++;
        m_rtgmc->resetTemporalState();
        m_cache.clear();
        m_cacheCopyEvent = RGYCudaEvent();
        const int primingFrames = m_rtgmc->requiredPrimingSourceFrames();
        const int primeStart = std::max(0, sourceBegin - primingFrames);
        m_nextFeedSourceIndex = primeStart;
        m_submittedFrames = primeStart * 2;
        m_nextOutputN60 = m_submittedFrames;
        m_hotUntilSourceIndex = -1;
    }

    m_cacheFloorN60 = n60begin;
    while (m_nextOutputN60 < n60end) {
        if (!m_owner) {
            return RGY_ERR_INVALID_CALL;
        }
        if (m_nextFeedSourceIndex >= m_owner->m_cachedSourceFrames) {
            if (m_owner->m_analyzerFinalized) {
                return drainTo(n60end, stream);
            }
            return RGY_ERR_MORE_DATA;
        }
        const auto *source = m_owner->findSourceByIndexExact(m_nextFeedSourceIndex);
        if (!source || !source->frame || !source->frame->frame.ptr[0]) {
            if (m_owner->m_analyzerFinalized) {
                return drainTo(n60end, stream);
            }
            return RGY_ERR_MORE_DATA;
        }
        std::vector<RGYCudaEvent> waitEvents;
        if (source->event() != nullptr) {
            waitEvents.push_back(source->event);
        }
        uint64_t intermediateGeneration = 0;
        if (sharedDeint60AnalysisLane()) {
            auto ensureSts = m_owner->ensureDeint60IntermediateForSource(source->sourceIndex, stream);
            if (ensureSts != RGY_ERR_NONE) {
                return ensureSts;
            }
            if (!m_owner->pushDeint60Intermediates(m_rtgmc, source->sourceIndex, &intermediateGeneration)) {
                m_owner->AddMessage(RGY_LOG_WARN, _T("KFM %S RTGMC skipped sourceIndex=%d because deint60 intermediates were not ready.\n"),
                    m_stage, source->sourceIndex);
                return RGY_ERR_MORE_DATA;
            }
        }
        auto sts = feed(&source->frame->frame, stream, waitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (sharedDeint60AnalysisLane()) {
            markIntermediateGeneration(source->sourceIndex, intermediateGeneration);
        }
        if (m_owner && m_rtgmc == m_owner->m_deint60Rtgmc.get()) {
            m_owner->captureDeint60Intermediates(source->sourceIndex);
        }
        m_nextFeedSourceIndex++;
        m_nextOutputN60 = m_submittedFrames;
    }
    m_hotUntilSourceIndex = std::max(m_hotUntilSourceIndex, m_nextFeedSourceIndex + HOT_KEEP_SOURCE_FRAMES - 1);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::feedHot(cudaStream_t stream) {
    if (!m_rtgmc || !m_owner || m_nextFeedSourceIndex < 0 || m_nextFeedSourceIndex > m_hotUntilSourceIndex) {
        return RGY_ERR_NONE;
    }
    const auto *source = m_owner->findSourceByIndexExact(m_nextFeedSourceIndex);
    if (!source || !source->frame || !source->frame->frame.ptr[0]) {
        return RGY_ERR_NONE;
    }
    std::vector<RGYCudaEvent> waitEvents;
    if (source->event() != nullptr) {
        waitEvents.push_back(source->event);
    }
    uint64_t intermediateGeneration = 0;
    if (sharedDeint60AnalysisLane()) {
        auto ensureSts = m_owner->ensureDeint60IntermediateForSource(source->sourceIndex, stream);
        if (ensureSts != RGY_ERR_NONE) {
            return RGY_ERR_NONE;
        }
        if (!m_owner->pushDeint60Intermediates(m_rtgmc, source->sourceIndex, &intermediateGeneration)) {
            m_owner->AddMessage(RGY_LOG_WARN, _T("KFM %S RTGMC hot feed stopped at sourceIndex=%d because deint60 intermediates were not ready.\n"),
                m_stage, source->sourceIndex);
            return RGY_ERR_NONE;
        }
    }
    auto sts = feed(&source->frame->frame, stream, waitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (sharedDeint60AnalysisLane()) {
        markIntermediateGeneration(source->sourceIndex, intermediateGeneration);
    }
    if (m_owner && m_rtgmc == m_owner->m_deint60Rtgmc.get()) {
        m_owner->captureDeint60Intermediates(source->sourceIndex);
    }
    m_nextFeedSourceIndex++;
    m_nextOutputN60 = m_submittedFrames;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::KfmRtgmcLane::cacheFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_owner || !m_stage || !m_stage[0] || !m_cacheLabel || !m_owner->m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }

    if (m_owner && (m_rtgmc == m_owner->m_before60Rtgmc.get() || m_rtgmc == m_owner->m_after60Rtgmc.get()) && frame->inputFrameId >= 0) {
        const int frameN60Base = frame->inputFrameId * 2;
        if (m_submittedFrames < frameN60Base || m_submittedFrames > frameN60Base + 1) {
            m_submittedFrames = frameN60Base;
        }
    }

    KfmCachedDeint60 entry;
    entry.n60 = m_submittedFrames++;
    if (entry.n60 < m_cacheFloorN60) {
        return RGY_ERR_NONE;
    }
    entry.inputFrameId = frame->inputFrameId;
    entry.timestamp = frame->timestamp;
    entry.duration = frame->duration;
    entry.frame = m_owner->acquireKfmFrame(*frame, m_cacheLabel);
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto mergeWaitEvents = wait_events;
    const int sourceIndex = entry.n60 >> 1;
    const auto *source = m_owner->findSourceByIndexExact(sourceIndex);
    if (!source) {
        m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM source frame is missing for %S output n60=%d, sourceIndex=%d, inputFrameId=%d.\n"),
            m_stage, entry.n60, sourceIndex, frame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    if (source->event() != nullptr) {
        mergeWaitEvents.push_back(source->event);
    }

    const auto rawStage = m_dumpStaticFlag ? std::string("rtgmc60-raw") : (std::string(m_stage) + "-raw");
    auto sts = m_owner->dumpStageFrame(rawStage.c_str(), frame, entry.n60, stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYCudaEvent staticEvent;
    sts = m_owner->analyzeStaticFlag(source->sourceIndex, stream, mergeWaitEvents, &staticEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (staticEvent() != nullptr) {
        mergeWaitEvents.push_back(staticEvent);
    }
    if (m_dumpStaticFlag) {
        sts = m_owner->dumpStageFrame("static-flag", &m_owner->m_staticFlag->frame, sourceIndex, stream,
            (staticEvent() != nullptr) ? std::vector<RGYCudaEvent>{ staticEvent } : mergeWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    sts = m_owner->mergeStatic(&entry.frame->frame, frame, &source->frame->frame, stream, mergeWaitEvents, &entry.event);
    if (sts != RGY_ERR_NONE) {
        m_owner->AddMessage(RGY_LOG_ERROR, _T("failed to merge/cache KFM %S frame: %s.\n"), m_stage, get_err_mes(sts));
        return sts;
    }
    if (event && entry.event() != nullptr) {
        *event = entry.event;
    }
    m_owner->writeFrameInfoDump(m_stage, &entry.frame->frame);
    sts = m_owner->dumpStageFrame(m_stage, &entry.frame->frame, entry.n60, stream,
        (entry.event() != nullptr) ? std::vector<RGYCudaEvent>{ entry.event } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_cache.push_back(std::move(entry));
    m_owner->trimDeint60Cache(m_cache);
    return RGY_ERR_NONE;
}

const NVEncFilterKfm::KfmCachedDeint60 *NVEncFilterKfm::KfmRtgmcLane::find(int n60, std::vector<RGYCudaEvent> *wait_events) const {
    for (auto it = m_cache.rbegin(); it != m_cache.rend(); ++it) {
        if (it->n60 == n60) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &(*it);
        }
    }
    return nullptr;
}

void NVEncFilterKfm::KfmRtgmcLane::trim(int n60floor, size_t cacheLimit) {
    while (!m_cache.empty() && m_cache.front().n60 < n60floor) {
        m_cache.pop_front();
    }
    while (m_cache.size() > cacheLimit && !m_cache.empty() && m_cache.front().n60 < n60floor) {
        m_cache.pop_front();
    }
}

int NVEncFilterKfm::KfmRtgmcLane::requiredPrimingSourceFrames() const {
    return m_rtgmc ? m_rtgmc->requiredPrimingSourceFrames() : 0;
}

bool NVEncFilterKfm::KfmRtgmcLane::sharedDeint60AnalysisLane() const {
    return m_owner && (m_rtgmc == m_owner->m_before60Rtgmc.get() || m_rtgmc == m_owner->m_after60Rtgmc.get());
}

void NVEncFilterKfm::KfmRtgmcLane::markIntermediateGeneration(int sourceIndex, uint64_t generation) {
    if (sourceIndex < 0) {
        return;
    }
    for (auto& entry : m_intermediateGenerations) {
        if (entry.first == sourceIndex) {
            entry.second = generation;
            return;
        }
    }
    m_intermediateGenerations.push_back(std::make_pair(sourceIndex, generation));
    while (!m_intermediateGenerations.empty() && m_intermediateGenerations.front().first + HOT_KEEP_SOURCE_FRAMES * 4 < m_nextFeedSourceIndex) {
        m_intermediateGenerations.pop_front();
    }
}

void NVEncFilterKfm::KfmRtgmcLane::rewindIfIntermediateChanged(int sourceIndex, uint64_t generation) {
    if (!m_rtgmc || sourceIndex < 0) {
        return;
    }
    const auto used = std::find_if(m_intermediateGenerations.begin(), m_intermediateGenerations.end(), [sourceIndex](const std::pair<int, uint64_t>& entry) {
        return entry.first == sourceIndex;
    });
    if (used == m_intermediateGenerations.end() || used->second == generation) {
        return;
    }
    const int rewindN60 = sourceIndex * 2;
    const bool hasGeneratedRange = std::any_of(m_cache.begin(), m_cache.end(), [rewindN60](const KfmCachedDeint60& entry) {
        return entry.n60 >= rewindN60;
    }) || m_nextFeedSourceIndex > sourceIndex;
    if (!hasGeneratedRange) {
        used->second = generation;
        return;
    }

    m_rtgmc->resetTemporalState();
    for (auto it = m_cache.begin(); it != m_cache.end();) {
        if (it->n60 >= rewindN60) {
            it = m_cache.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_intermediateGenerations.begin(); it != m_intermediateGenerations.end();) {
        if (it->first >= sourceIndex) {
            it = m_intermediateGenerations.erase(it);
        } else {
            ++it;
        }
    }
    const int primeStart = std::max(0, sourceIndex - requiredPrimingSourceFrames());
    m_nextFeedSourceIndex = primeStart;
    m_submittedFrames = primeStart * 2;
    m_nextOutputN60 = m_submittedFrames;
    m_hotUntilSourceIndex = -1;
}

std::shared_ptr<CUFrameBuf> NVEncFilterKfm::SharedFramePool::acquire(const RGYFrameInfo& info) {
    auto resetFrameState = [](RGYFrameInfo& frame) {
        frame.timestamp = 0;
        frame.duration = 0;
        frame.picstruct = RGY_PICSTRUCT_UNKNOWN;
        frame.flags = RGY_FRAME_FLAG_NONE;
        frame.inputFrameId = -1;
        frame.dataList.clear();
    };
    auto matchesInfo = [&info](const std::shared_ptr<CUFrameBuf>& frame) {
        return frame
            && !cmpFrameInfoCspResolution(&frame->frame, &info)
            && RGY_CSP_BIT_DEPTH[frame->frame.csp] == RGY_CSP_BIT_DEPTH[info.csp];
    };
    auto trimPool = [&]() {
        while (m_pool.size() > MAX_POOL_FRAMES) {
            auto it = m_pool.begin();
            for (; it != m_pool.end(); ++it) {
                if (*it && (*it).use_count() == 1) {
                    break;
                }
            }
            if (it == m_pool.end()) {
                break;
            }
            m_pool.erase(it);
        }
    };
    for (auto& frame : m_pool) {
        if (frame && frame.use_count() == 1
            && matchesInfo(frame)) {
            // KFM pool frames are never returned outside this filter; emitOutputFrame()
            // copies them to the normal output ring first. All KFM users run on the
            // filter stream, so stream ordering makes reuse safe without host sync.
            resetFrameState(frame->frame);
            auto selected = frame;
            trimPool();
            return selected;
        }
    }
    auto frame = std::make_shared<CUFrameBuf>(info);
    frame->releasePtr();
    if (frame->alloc() != RGY_ERR_NONE) {
        return nullptr;
    }
    resetFrameState(frame->frame);
    m_pool.push_back(frame);
    trimPool();
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

std::shared_ptr<NVEncFilterKfm::KfmSourceSlot> NVEncFilterKfm::acquireKfmSourceSlot(const RGYFrameInfo& sourceInfo) {
    collectRetiredKfmSourceSlots();
    auto matchSlot = [&sourceInfo](const std::shared_ptr<KfmSourceSlot>& slot) {
        return slot && slot->sourceFrame && slot->paddedFrame
            && !cmpFrameInfoCspResolution(&slot->sourceFrame->frame, &sourceInfo)
            && RGY_CSP_BIT_DEPTH[slot->sourceFrame->frame.csp] == RGY_CSP_BIT_DEPTH[sourceInfo.csp];
    };
    auto pooled = std::find_if(m_kfmSourceSlotFree.begin(), m_kfmSourceSlotFree.end(), matchSlot);
    if (pooled != m_kfmSourceSlotFree.end()) {
        auto slot = std::move(*pooled);
        m_kfmSourceSlotFree.erase(pooled);
        slot->readyEvent.reset();
        slot->sourceFrame->frame.dataList.clear();
        slot->paddedFrame->frame.dataList.clear();
        return slot;
    }

    auto paddedInfo = sourceInfo;
    paddedInfo.height += KFM_SOURCE_VPAD * 2;
    auto paddedFrame = acquireKfmFrame(paddedInfo, _T("padded source cache"));
    if (!paddedFrame) {
        return nullptr;
    }

    RGYFrameInfo viewInfo = sourceInfo;
    viewInfo.mem_type = RGY_MEM_TYPE_GPU;
    for (int i = 0; i < _countof(viewInfo.ptr); i++) {
        viewInfo.ptr[i] = nullptr;
        viewInfo.pitch[i] = 0;
    }
    const int planes = RGY_CSP_PLANES[sourceInfo.csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto parent = getPlane(&paddedFrame->frame, (RGY_PLANE)iplane);
        const auto view = getPlane(&sourceInfo, (RGY_PLANE)iplane);
        const int vpad = (parent.height - view.height) >> 1;
        if (!parent.ptr[0] || parent.width != view.width || parent.height != view.height + vpad * 2 || vpad <= 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM source slot plane size (plane %d, src %dx%d, padded %dx%d).\n"),
                iplane, view.width, view.height, parent.width, parent.height);
            return nullptr;
        }
        viewInfo.ptr[iplane] = parent.ptr[0] + parent.pitch[0] * vpad;
        viewInfo.pitch[iplane] = parent.pitch[0];
    }

    auto sourceFrame = std::shared_ptr<CUFrameBuf>(
        new CUFrameBuf(viewInfo),
        [paddedKeepAlive = paddedFrame](CUFrameBuf *frame) {
            frame->releasePtr();
            delete frame;
        });

    auto slot = std::make_shared<KfmSourceSlot>();
    slot->paddedFrame = paddedFrame;
    slot->sourceFrame = sourceFrame;
    return slot;
}

RGY_ERR NVEncFilterKfm::retireKfmSourceSlot(std::shared_ptr<KfmSourceSlot>&& slot, cudaStream_t stream) {
    if (!slot) {
        return RGY_ERR_NONE;
    }
    slot->sourceFrame->frame.dataList.clear();
    slot->paddedFrame->frame.dataList.clear();
    slot->readyEvent.reset();
    auto sts = kfmRecordEvent(stream, &slot->readyEvent);
    if (sts != RGY_ERR_NONE) {
        sts = err_to_rgy(cudaStreamSynchronize(stream));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_kfmSourceSlotFree.emplace_back(std::move(slot));
        trimFreeKfmSourceSlots();
        return RGY_ERR_NONE;
    }
    m_kfmSourceSlotRetired.emplace_back(std::move(slot));
    collectRetiredKfmSourceSlots();
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::collectRetiredKfmSourceSlots() {
    for (auto it = m_kfmSourceSlotRetired.begin(); it != m_kfmSourceSlotRetired.end();) {
        auto& slot = *it;
        if (!slot || slot->readyEvent() == nullptr || cudaEventQuery(slot->readyEvent()) == cudaSuccess) {
            if (slot) {
                slot->readyEvent.reset();
                m_kfmSourceSlotFree.emplace_back(std::move(slot));
            }
            it = m_kfmSourceSlotRetired.erase(it);
        } else {
            ++it;
        }
    }
    trimFreeKfmSourceSlots();
}

void NVEncFilterKfm::trimFreeKfmSourceSlots() {
    const auto keep = std::max<size_t>(16, std::min<size_t>(sourceCacheLimit(), 256) + 8);
    while (m_kfmSourceSlotFree.size() > keep) {
        m_kfmSourceSlotFree.pop_front();
    }
}

void NVEncFilterKfm::clearKfmSourceSlotPool(bool wait) {
    if (wait) {
        for (auto& slot : m_kfmSourceSlotRetired) {
            if (slot && slot->readyEvent() != nullptr) {
                cudaEventSynchronize(slot->readyEvent());
                slot->readyEvent.reset();
            }
        }
    }
    m_kfmSourceSlotRetired.clear();
    m_kfmSourceSlotFree.clear();
}

RGY_ERR NVEncFilterKfm::trimSourceCache(cudaStream_t stream) {
    const auto trimFloor = sourceCacheTrimFloor();
    while (!m_sourceCache.empty() && m_sourceCache.front().sourceIndex < trimFloor) {
        auto sts = retireKfmSourceSlot(std::move(m_sourceCache.front().slot), stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_sourceCache.pop_front();
    }
    const auto cacheLimit = sourceCacheLimit();
    while (m_sourceCache.size() > cacheLimit && !m_sourceCache.empty() && m_sourceCache.front().sourceIndex < trimFloor) {
        auto sts = retireKfmSourceSlot(std::move(m_sourceCache.front().slot), stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_sourceCache.pop_front();
    }
    collectRetiredKfmSourceSlots();
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::trimDeint60Cache(std::deque<KfmCachedDeint60>& cache) {
    const auto trimFloor = deint60CacheTrimFloor();
    while (!cache.empty() && cache.front().n60 < trimFloor) {
        cache.pop_front();
    }
    const auto cacheLimit = deint60CacheLimit();
    while (cache.size() > cacheLimit && !cache.empty() && cache.front().n60 < trimFloor) {
        cache.pop_front();
    }
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
    m_deint60Lane.reset();
    m_before60Lane.reset();
    m_after60Lane.reset();
    m_ucfNoiseCache.clear();
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();

    auto clearFMCountSts = clearPendingFMCounts();
    m_fmCountBufPool.clear();
    if (clearFMCountSts != RGY_ERR_NONE) {
        return clearFMCountSts;
    }

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

RGY_ERR NVEncFilterKfm::initRtgmc(const std::shared_ptr<NVEncFilterParamKfm>& prm, std::unique_ptr<NVEncFilterRtgmc>& rtgmc, bool updateOutputParam, int useFlag, bool sharedAnalysisMode) {
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
    rtgmcParam->sharedAnalysisMode = sharedAnalysisMode;

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
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, bool sourceInPaddedFrame) {
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
        sts = sourceInPaddedFrame
            ? run_kfm_padv_inplace_plane(&dst, src.height, vpad, stream)
            : run_kfm_pad_plane(&dst, &src, vpad, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %S (plane %d): %s.\n"), sourceInPaddedFrame ? "kernel_kfm_padv_inplace" : "kernel_kfm_pad", iplane, get_err_mes(sts));
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
    entry.slot = acquireKfmSourceSlot(*frame);
    if (!entry.slot || !entry.slot->sourceFrame || !entry.slot->paddedFrame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    entry.frame = entry.slot->sourceFrame;
    entry.paddedFrame = entry.slot->paddedFrame;
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
    m_sourceCache.push_back(std::move(entry));
    auto& cachedEntry = m_sourceCache.back();

    sts = analyzeAvailableSource(false, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto padWaitEvents = wait_events;
    if (cachedEntry.event() != nullptr) {
        padWaitEvents.push_back(cachedEntry.event);
    }
    sts = padSourceFrame(&cachedEntry.paddedFrame->frame, &cachedEntry.frame->frame, stream, padWaitEvents, &cachedEntry.paddedEvent, true);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to pad KFM source frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    writeFrameInfoDump("source-pad", &cachedEntry.paddedFrame->frame);

    sts = trimSourceCache(stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    writeFrameInfoDump("source", frame);
    return RGY_ERR_NONE;
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
    m_deint60Lane.init(this, nullptr, "deint60", _T("deint60 cache"), true);
    m_before60Lane.init(this, nullptr, "before60", _T("before60"), false);
    m_after60Lane.init(this, nullptr, "after60", _T("after60"), false);

    auto sts = RGY_ERR_NONE;
    if (prm->kfm.mode == VppKfmMode::P60) {
        sts = initRtgmc(prm, m_rtgmc, true);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            m_rtgmc->enableIntermediateCapture(true);
            sts = initRtgmc(prm, m_before60Rtgmc, false, 1, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = initRtgmc(prm, m_after60Rtgmc, false, 2, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            auto sharedData = m_rtgmc->getSharedAnalysisData();
            m_before60Rtgmc->setSharedAnalysisData(sharedData);
            m_after60Rtgmc->setSharedAnalysisData(sharedData);
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
    if (prm->kfm.mode == VppKfmMode::VFR
        || prm->kfm.mode == VppKfmMode::P60
        || (prm->kfm.mode == VppKfmMode::P24 && kfmDeint60BranchEnabled())
        || prm->kfm.ucf) {
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
        NVEncFilterRtgmc::RtgmcSharedAnalysisData sharedData;
        bool useSharedAnalysisMode = false;
        if (m_deint60Rtgmc) {
            sharedData = m_deint60Rtgmc->getSharedAnalysisData();
            useSharedAnalysisMode = sharedData.analyzeFilter != nullptr;
            if (useSharedAnalysisMode) {
                m_deint60Rtgmc->enableIntermediateCapture(true);
            }
        }
        sts = initRtgmc(prm, m_before60Rtgmc, false, 1, useSharedAnalysisMode);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = initRtgmc(prm, m_after60Rtgmc, false, 2, useSharedAnalysisMode);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (useSharedAnalysisMode) {
            m_before60Rtgmc->setSharedAnalysisData(sharedData);
            m_after60Rtgmc->setSharedAnalysisData(sharedData);
        }
    }
    m_deint60Lane.init(this, m_deint60Rtgmc.get(), "deint60", _T("deint60 cache"), true);
    m_before60Lane.init(this, m_before60Rtgmc.get(), "before60", _T("before60"), false);
    m_after60Lane.init(this, m_after60Rtgmc.get(), "after60", _T("after60"), false);
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
    clearKfmSourceSlotPool(true);
    m_deint60Lane.reset();
    m_before60Lane.reset();
    m_after60Lane.reset();
    m_deint60IntermediateQueue.clear();
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
    if (lazyDeint60Enabled(*prm)) {
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
    int lazyLookbehind = 0;
    if (lazyDeint60Enabled(*prm) && m_deint60Rtgmc) {
        lazyLookbehind = m_deint60Rtgmc->requiredPrimingSourceFrames();
        if (prm->kfm.ucf && (m_before60Rtgmc || m_after60Rtgmc)) {
            lazyLookbehind += std::max(m_before60Lane.requiredPrimingSourceFrames(), m_after60Lane.requiredPrimingSourceFrames())
                + KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY + 4 + KFM_UCF_LAZY_SOURCE_CACHE_MARGIN;
        }
    }
    auto trimFloor = std::max(0, (m_nextSwitchN60 >> 1) - KFM_VFR_SOURCE_TRIM_LOOKBEHIND - lazyLookbehind);
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

bool NVEncFilterKfm::lazyDeint60Enabled(const NVEncFilterParamKfm& prm) const {
    return prm.kfm.mode == VppKfmMode::VFR && !kfmForceEagerRtgmc();
}

const RGYFrameInfo *NVEncFilterKfm::findDeint60Frame(int n60, std::vector<RGYCudaEvent> *wait_events) const {
    const auto *entry = findCachedDeint60Frame(m_deint60Lane, n60, wait_events);
    return entry && entry->frame ? &entry->frame->frame : nullptr;
}

const NVEncFilterKfm::KfmCachedDeint60 *NVEncFilterKfm::findCachedDeint60Frame(const KfmRtgmcLane& lane, int n60, std::vector<RGYCudaEvent> *wait_events) const {
    return lane.find(n60, wait_events);
}

const NVEncFilterKfm::KfmUcfNoiseDumpRecord *NVEncFilterKfm::findUcfNoiseResult(int sourceIndex) const {
    for (auto it = m_ucfNoiseResultCache.rbegin(); it != m_ucfNoiseResultCache.rend(); ++it) {
        if (it->sourceIndex == sourceIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

RGY_ERR NVEncFilterKfm::runUcfRtgmcBranches(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_before60Rtgmc && !m_after60Rtgmc) {
        return runUcfNoiseAnalysisFromSource(frame, stream, wait_events);
    }
    if (!frame || !frame->ptr[0]) {
        auto sts = drainUcfRtgmcBranch(m_before60Lane, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = drainUcfRtgmcBranch(m_after60Lane, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return runUcfNoiseAnalysisFromSource(frame, stream, wait_events);
    }

    auto sts = runUcfRtgmcBranch(m_before60Lane, frame, stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = runUcfRtgmcBranch(m_after60Lane, frame, stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return runUcfNoiseAnalysisFromSource(frame, stream, wait_events);
}

RGY_ERR NVEncFilterKfm::runUcfNoiseAnalysisFromSource(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto sts = RGY_ERR_NONE;
    const int sourceIndex = m_sourceCache.empty() ? m_cachedSourceFrames - 1 : m_sourceCache.back().sourceIndex;
    for (int parity = 0; parity < 2; parity++) {
        const int fieldIndex = sourceIndex * 2 + parity;
        const bool useFusedUcfPreprocess = kfmUseFusedUcfPreprocess() && !stageDumpRequested(fieldIndex);
        RGYFrameInfo *gaussFrame = nullptr;
        RGYCudaEvent gaussEvent;
        RGYFrameInfo *fieldFrame = nullptr;
        if (useFusedUcfPreprocess && !kfmUcfNoGaussForTest()) {
            sts = prepareUcfNoiseGaussFrameFromSource(&gaussFrame, sourceIndex, parity, frame, stream, wait_events, &gaussEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            RGYCudaEvent fieldEvent;
            sts = prepareUcfNoiseFieldCropFrame(&fieldFrame, sourceIndex, parity, frame, stream, wait_events, &fieldEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            std::vector<RGYCudaEvent> gaussWaitEvents;
            if (fieldEvent() != nullptr) {
                gaussWaitEvents.push_back(fieldEvent);
            }
            if (kfmUcfNoGaussForTest()) {
                gaussFrame = fieldFrame;
                gaussEvent = fieldEvent;
                writeFrameInfoDump("ucf-noise-gauss", gaussFrame);
                sts = dumpStageFrame("ucf-noise-gauss", gaussFrame, fieldIndex, stream, gaussWaitEvents);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            } else {
                sts = prepareUcfNoiseGaussFrame(&gaussFrame, parity, fieldFrame, stream, gaussWaitEvents, &gaussEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        std::vector<RGYCudaEvent> deintWaitEvents;
        if (gaussEvent() != nullptr) {
            deintWaitEvents.push_back(gaussEvent);
        }
        sts = (useFusedUcfPreprocess && fieldFrame == nullptr)
            ? runUcfNoiseLimitStageFromSource(*prm, frame, gaussFrame, fieldIndex, parity, stream, deintWaitEvents)
            : runUcfNoiseLimitStage(*prm, fieldFrame, gaussFrame, fieldIndex, stream, deintWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::ensureUcfRtgmcRange(KfmUcfLaneType laneType, int n60begin, int n60end, cudaStream_t stream) {
    if (laneType == KFM_UCF_LANE_NONE || n60begin >= n60end) {
        return RGY_ERR_NONE;
    }
    // eager (FORCE_EAGER or non-lazy) では intake で全フレーム feed 済みのため ensure 不要。
    // ここで ensureRange を呼ぶと進行中の eager パイプラインを resetTemporalState で
    // 破壊して再 priming が発生する (長尺の eager 実行で性能劣化と出力差の原因になる)。
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(m_param);
    if (!prm || !lazyDeint60Enabled(*prm)) {
        return RGY_ERR_NONE;
    }
    auto *lane = (laneType == KFM_UCF_LANE_BEFORE) ? &m_before60Lane : &m_after60Lane;
    if (!m_deint60Rtgmc || (laneType == KFM_UCF_LANE_BEFORE && !m_before60Rtgmc) || (laneType == KFM_UCF_LANE_AFTER && !m_after60Rtgmc)) {
        return RGY_ERR_NONE;
    }
    const int sourceBegin = std::max(0, n60begin) >> 1;
    const int primeStart = std::max(0, sourceBegin - lane->requiredPrimingSourceFrames());
    const int sourceEndDelay = (laneType == KFM_UCF_LANE_AFTER) ? KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY : 0;
    const int sourceEnd = std::max(primeStart, divCeil(std::max(0, n60end), 2) + sourceEndDelay);
    if (!m_analyzerFinalized && m_cachedSourceFrames < sourceEnd) {
        return RGY_ERR_MORE_DATA;
    }
    for (int sourceIndex = primeStart; sourceIndex < sourceEnd; sourceIndex++) {
        if (hasDeint60Intermediate(sourceIndex)) {
            continue;
        }
        auto sts = ensureDeint60IntermediateForSource(sourceIndex, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!hasDeint60Intermediate(sourceIndex)) {
            AddMessage(RGY_LOG_WARN, _T("KFM UCF RTGMC could not prepare deint60 intermediates for sourceIndex=%d.\n"),
                sourceIndex);
            return RGY_ERR_MORE_DATA;
        }
    }
    const int sideN60End = (laneType == KFM_UCF_LANE_AFTER) ? std::max(n60end, sourceEnd * 2) : n60end;
    auto sts = lane->ensureRange(n60begin, sideN60End, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    for (int n60 = n60begin; n60 < n60end; n60++) {
        if (!lane->find(n60, nullptr)) {
            return RGY_ERR_MORE_DATA;
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::captureDeint60Intermediates(int sourceIndex) {
    if (!m_deint60Rtgmc || sourceIndex < 0) {
        return;
    }
    const auto& captured = m_deint60Rtgmc->getCapturedIntermediates();
    if (!captured.empty()) {
        for (auto& group : m_deint60IntermediateQueue) {
            if (group.sourceIndex == sourceIndex) {
                group.generation++;
                group.intermediates = captured;
                m_before60Lane.rewindIfIntermediateChanged(sourceIndex, group.generation);
                m_after60Lane.rewindIfIntermediateChanged(sourceIndex, group.generation);
                m_deint60Rtgmc->clearCapturedIntermediates();
                return;
            }
        }
        KfmMainIntermediateGroup group;
        group.sourceIndex = sourceIndex;
        group.generation = 1;
        group.intermediates = captured;
        m_deint60IntermediateQueue.push_back(std::move(group));
        trimDeint60Intermediates();
    }
    m_deint60Rtgmc->clearCapturedIntermediates();
}

bool NVEncFilterKfm::hasDeint60Intermediate(int sourceIndex, uint64_t *generation) const {
    if (generation) {
        *generation = 0;
    }
    if (sourceIndex < 0) {
        return false;
    }
    const auto it = std::find_if(m_deint60IntermediateQueue.begin(), m_deint60IntermediateQueue.end(), [sourceIndex](const KfmMainIntermediateGroup& group) {
        return group.sourceIndex == sourceIndex && !group.intermediates.empty();
    });
    if (it == m_deint60IntermediateQueue.end()) {
        return false;
    }
    if (generation) {
        *generation = it->generation;
    }
    return true;
}

bool NVEncFilterKfm::hasDeint60Intermediates(int sourceBegin, int sourceEnd) const {
    for (int sourceIndex = sourceBegin; sourceIndex < sourceEnd; sourceIndex++) {
        if (!hasDeint60Intermediate(sourceIndex)) {
            return false;
        }
    }
    return true;
}

RGY_ERR NVEncFilterKfm::ensureDeint60IntermediateForSource(int sourceIndex, cudaStream_t stream) {
    if (sourceIndex < 0) {
        return RGY_ERR_NONE;
    }
    if (hasDeint60Intermediate(sourceIndex)) {
        return RGY_ERR_NONE;
    }
    if (!m_deint60Rtgmc) {
        return RGY_ERR_MORE_DATA;
    }
    purgeDeint60Intermediates(sourceIndex, sourceIndex + 1);
    auto& deint60Cache = m_deint60Lane.cache();
    for (auto it = deint60Cache.begin(); it != deint60Cache.end();) {
        if (sourceIndex * 2 <= it->n60 && it->n60 < (sourceIndex + 1) * 2) {
            it = deint60Cache.erase(it);
        } else {
            ++it;
        }
    }
    auto sts = m_deint60Lane.ensureRange(sourceIndex * 2, (sourceIndex + 1) * 2, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!hasDeint60Intermediate(sourceIndex)) {
        AddMessage(RGY_LOG_WARN, _T("KFM UCF RTGMC could not prepare deint60 intermediates for sourceIndex=%d.\n"), sourceIndex);
        return RGY_ERR_MORE_DATA;
    }
    return RGY_ERR_NONE;
}

bool NVEncFilterKfm::pushDeint60Intermediates(NVEncFilterRtgmc *rtgmc, int sourceIndex, uint64_t *generation) {
    if (generation) {
        *generation = 0;
    }
    if (!rtgmc || sourceIndex < 0) {
        return false;
    }
    for (const auto& group : m_deint60IntermediateQueue) {
        if (group.sourceIndex == sourceIndex && !group.intermediates.empty()) {
            for (const auto& captured : group.intermediates) {
                rtgmc->pushIntermediateInput(captured);
            }
            if (generation) {
                *generation = group.generation;
            }
            return true;
        }
    }
    const char *stage = (rtgmc == m_before60Rtgmc.get()) ? "before60" : ((rtgmc == m_after60Rtgmc.get()) ? "after60" : "unknown");
    AddMessage(RGY_LOG_WARN, _T("KFM %S RTGMC shared-analysis intermediates missing for sourceIndex=%d; feed postponed.\n"),
        stage, sourceIndex);
    return false;
}

void NVEncFilterKfm::purgeDeint60Intermediates(int sourceBegin, int sourceEnd) {
    sourceBegin = std::max(0, sourceBegin);
    sourceEnd = std::max(sourceBegin, sourceEnd);
    for (auto it = m_deint60IntermediateQueue.begin(); it != m_deint60IntermediateQueue.end();) {
        if (sourceBegin <= it->sourceIndex && it->sourceIndex < sourceEnd) {
            m_before60Lane.rewindIfIntermediateChanged(it->sourceIndex, it->generation + 1);
            m_after60Lane.rewindIfIntermediateChanged(it->sourceIndex, it->generation + 1);
            it = m_deint60IntermediateQueue.erase(it);
        } else {
            ++it;
        }
    }
}

void NVEncFilterKfm::trimDeint60Intermediates() {
    const int keepSourceFrames = std::max<int>(32, (int)sourceCacheLimit());
    while (!m_deint60IntermediateQueue.empty() && m_deint60IntermediateQueue.front().sourceIndex + keepSourceFrames < m_cachedSourceFrames) {
        m_deint60IntermediateQueue.pop_front();
    }
}

RGY_ERR NVEncFilterKfm::runUcfRtgmcBranch(KfmRtgmcLane& lane, const RGYFrameInfo *frame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    return lane.feed(frame, stream, wait_events);
}

RGY_ERR NVEncFilterKfm::drainUcfRtgmcBranch(KfmRtgmcLane& lane, cudaStream_t stream) {
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    return lane.drain(stream, maxDrainIterations);
}

RGY_ERR NVEncFilterKfm::copyUcfFrame(const NVEncFilterParamKfm& prm, RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pOutputFrame || !pInputFrame) {
        return RGY_ERR_INVALID_CALL;
    }

    const auto kernelName = kfmUcfKernelName(prm.kfm.mode);
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src = getPlane(pInputFrame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_copy_plane(&dst, &src, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %S (plane %d): %s.\n"), kernelName, iplane, get_err_mes(sts));
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
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    pOutputFrame->dataList = pInputFrame->dataList;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::createUcfGaussProgram(KfmUcfGaussProgram& program, int sourceSize, double cropSize, int targetSize, double p) {
    const double cropStart = 0.0;
    const double filterScale = (double)targetSize / cropSize;
    const double filterStep = std::min(filterScale, 1.0);
    const double filterSupport = 4.0 / filterStep;
    const int filterSize = (int)std::ceil(filterSupport * 2.0);
    if (sourceSize <= filterSupport || targetSize <= 0 || filterSize <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF gaussresize size source=%d target=%d filter=%d.\n"),
            sourceSize, targetSize, filterSize);
        return RGY_ERR_INVALID_PARAM;
    }

    std::vector<int> offset(targetSize);
    std::vector<float> coeff((size_t)targetSize * filterSize);
    double pos = (filterSize == 1) ? cropStart : cropStart + ((cropSize - targetSize) / (targetSize * 2.0));
    const double posStep = cropSize / targetSize;
    for (int i = 0; i < targetSize; i++) {
        int endPos = (int)(pos + filterSupport);
        if (endPos > sourceSize - 1) {
            endPos = sourceSize - 1;
        }
        int startPos = endPos - filterSize + 1;
        if (startPos < 0) {
            startPos = 0;
        }
        offset[i] = startPos;
        const double okPos = std::min(std::max(pos, 0.0), (double)(sourceSize - 1));
        double total = 0.0;
        for (int j = 0; j < filterSize; j++) {
            total += kfmUcfGaussValue((startPos + j - okPos) * filterStep, p);
        }
        if (total == 0.0) {
            total = 1.0;
        }
        double value = 0.0;
        for (int k = 0; k < filterSize; k++) {
            const double newValue = value + kfmUcfGaussValue((startPos + k - okPos) * filterStep, p) / total;
            coeff[(size_t)i * filterSize + k] = (float)(newValue - value);
            value = newValue;
        }
        pos += posStep;
    }
    program.sourceSize = sourceSize;
    program.targetSize = targetSize;
    program.filterSize = filterSize;
    program.offset = std::make_unique<CUMemBuf>(offset.size() * sizeof(offset[0]));
    program.coeff = std::make_unique<CUMemBuf>(coeff.size() * sizeof(coeff[0]));
    auto sts = program.offset->alloc();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = program.coeff->alloc();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaMemcpy(program.offset->ptr, offset.data(), offset.size() * sizeof(offset[0]), cudaMemcpyHostToDevice));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaMemcpy(program.coeff->ptr, coeff.data(), coeff.size() * sizeof(coeff[0]), cudaMemcpyHostToDevice));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::prepareUcfNoiseFieldCropFrame(RGYFrameInfo **ppFieldFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!ppFieldFrame || !pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppFieldFrame = nullptr;
    const int cropX = 4;
    const int cropY = 4;
    RGYFrameInfo fieldInfo = *pInputFrame;
    fieldInfo.width = pInputFrame->width - cropX * 2;
    fieldInfo.height = (pInputFrame->height >> 1) - cropY * 2;
    if (fieldInfo.width <= 0 || fieldInfo.height <= 0 || (fieldInfo.width & 1) || (fieldInfo.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF field/crop size from %dx%d.\n"), pInputFrame->width, pInputFrame->height);
        return RGY_ERR_INVALID_PARAM;
    }
    fieldInfo.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        fieldInfo.ptr[i] = nullptr;
        fieldInfo.pitch[i] = 0;
    }
    const int frameIndex = parity & 1;
    auto& fieldBuffer = m_ucfNoiseFieldFrames[frameIndex];
    if (!fieldBuffer
        || fieldBuffer->frame.width != fieldInfo.width
        || fieldBuffer->frame.height != fieldInfo.height
        || fieldBuffer->frame.csp != fieldInfo.csp) {
        fieldBuffer = std::make_unique<CUFrameBuf>(fieldInfo);
        fieldBuffer->releasePtr();
        const auto allocSts = fieldBuffer->alloc();
        if (allocSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF field/crop frame: %s.\n"), get_err_mes(allocSts));
            return allocSts;
        }
    }

    auto *pFieldFrame = &fieldBuffer->frame;
    const int fieldParity = parity & 1;
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pInputFrame->csp];
    const auto planes = RGY_CSP_PLANES[pFieldFrame->csp];
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        auto dst = getPlane(pFieldFrame, planeType);
        auto src = getPlane(pInputFrame, planeType);
        if (!dst.ptr[0] || !src.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_field_crop_plane(&dst, &src, srcXOffset, srcYOffset, srcYStep, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_field_crop (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    copyFramePropWithoutRes(pFieldFrame, pInputFrame);
    pFieldFrame->timestamp = pInputFrame->timestamp;
    pFieldFrame->duration = pInputFrame->duration;
    pFieldFrame->inputFrameId = pInputFrame->inputFrameId;
    pFieldFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pFieldFrame->flags = RGY_FRAME_FLAG_NONE;
    pFieldFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-field", pFieldFrame);
    auto sts = dumpStageFrame("ucf-field", pFieldFrame, sourceIndex * 2 + fieldParity, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppFieldFrame = pFieldFrame;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::prepareUcfNoiseGaussFrame(RGYFrameInfo **ppGaussFrame, int parity, const RGYFrameInfo *pInputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!ppGaussFrame || !pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppGaussFrame = nullptr;
    const int frameIndex = parity & 1;
    RGYFrameInfo gaussInfo = *pInputFrame;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        gaussInfo.ptr[i] = nullptr;
        gaussInfo.pitch[i] = 0;
    }
    auto allocFrame = [&](std::unique_ptr<CUFrameBuf>& frame, const TCHAR *label) -> RGY_ERR {
        if (!frame
            || frame->frame.width != gaussInfo.width
            || frame->frame.height != gaussInfo.height
            || frame->frame.csp != gaussInfo.csp) {
            frame = std::make_unique<CUFrameBuf>(gaussInfo);
            frame->releasePtr();
            const auto allocSts = frame->alloc();
            if (allocSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF %s frame: %s.\n"), label, get_err_mes(allocSts));
                return allocSts;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocFrame(m_ucfNoiseGaussTmpFrames[frameIndex], _T("gauss temporary"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = allocFrame(m_ucfNoiseGaussFrames[frameIndex], _T("gauss output"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto *pTmpFrame = &m_ucfNoiseGaussTmpFrames[frameIndex]->frame;
    auto *pGaussFrame = &m_ucfNoiseGaussFrames[frameIndex]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto src = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto tmp = getPlane(pTmpFrame, (RGY_PLANE)iplane);
        auto dst = getPlane(pGaussFrame, (RGY_PLANE)iplane);
        if (!src.ptr[0] || !tmp.ptr[0] || !dst.ptr[0]) {
            continue;
        }
        const int chromaIndex = (iplane == 0) ? 0 : 1;
        const bool interleavedUVPlane = interleavedUV && iplane != 0;
        const int srcWidthForGauss = interleavedUVPlane ? (src.width >> 1) : src.width;
        const int dstWidthForGauss = interleavedUVPlane ? (dst.width >> 1) : dst.width;
        const double cropWidth = (pInputFrame->width > 0)
            ? (double)srcWidthForGauss + KFM_UCF_GAUSS_CROP_EPS * (double)srcWidthForGauss / pInputFrame->width
            : (double)srcWidthForGauss + KFM_UCF_GAUSS_CROP_EPS;
        const double cropHeight = (pInputFrame->height > 0)
            ? (double)src.height + KFM_UCF_GAUSS_CROP_EPS * (double)src.height / pInputFrame->height
            : (double)src.height + KFM_UCF_GAUSS_CROP_EPS;
        auto& progV = m_ucfNoiseGaussVert[frameIndex][chromaIndex];
        auto& progH = m_ucfNoiseGaussHori[frameIndex][chromaIndex];
        if (!progV.offset || progV.sourceSize != src.height || progV.targetSize != dst.height) {
            sts = createUcfGaussProgram(progV, src.height, cropHeight, dst.height, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (!progH.offset || progH.sourceSize != srcWidthForGauss || progH.targetSize != dstWidthForGauss) {
            sts = createUcfGaussProgram(progH, srcWidthForGauss, cropWidth, dstWidthForGauss, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_gaussresize_v_plane(&tmp, &src, (const int *)progV.offset->ptr, (const float *)progV.coeff->ptr, progV.filterSize, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_gaussresize_v (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent evV;
        sts = kfmRecordEvent(stream, &evV);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = kfmWaitEvents(stream, { evV });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (interleavedUVPlane) {
            sts = run_kfm_ucf_gaussresize_h_uv_interleaved_plane(&dst, &tmp, dstWidthForGauss,
                (const int *)progH.offset->ptr, (const float *)progH.coeff->ptr, progH.filterSize, stream);
        } else {
            sts = run_kfm_ucf_gaussresize_h_plane(&dst, &tmp,
                (const int *)progH.offset->ptr, (const float *)progH.coeff->ptr, progH.filterSize, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_gaussresize_h (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    copyFramePropWithoutRes(pGaussFrame, pInputFrame);
    pGaussFrame->timestamp = pInputFrame->timestamp;
    pGaussFrame->duration = pInputFrame->duration;
    pGaussFrame->inputFrameId = pInputFrame->inputFrameId;
    pGaussFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pGaussFrame->flags = RGY_FRAME_FLAG_NONE;
    pGaussFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-noise-gauss", pGaussFrame);
    sts = dumpStageFrame("ucf-noise-gauss", pGaussFrame, m_timecodeFrameIndex * 2 + frameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppGaussFrame = pGaussFrame;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::prepareUcfNoiseGaussFrameFromSource(RGYFrameInfo **ppGaussFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!ppGaussFrame || !pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppGaussFrame = nullptr;
    const int cropX = 4;
    const int cropY = 4;
    RGYFrameInfo fieldInfo = *pInputFrame;
    fieldInfo.width = pInputFrame->width - cropX * 2;
    fieldInfo.height = (pInputFrame->height >> 1) - cropY * 2;
    if (fieldInfo.width <= 0 || fieldInfo.height <= 0 || (fieldInfo.width & 1) || (fieldInfo.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF field/crop size from %dx%d.\n"), pInputFrame->width, pInputFrame->height);
        return RGY_ERR_INVALID_PARAM;
    }
    fieldInfo.picstruct = RGY_PICSTRUCT_FRAME;
    fieldInfo.flags = RGY_FRAME_FLAG_NONE;
    fieldInfo.inputFrameId = pInputFrame->inputFrameId;
    fieldInfo.timestamp = pInputFrame->timestamp;
    fieldInfo.duration = pInputFrame->duration;
    writeFrameInfoDump("ucf-field", &fieldInfo);

    const int frameIndex = parity & 1;
    RGYFrameInfo gaussInfo = fieldInfo;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        gaussInfo.ptr[i] = nullptr;
        gaussInfo.pitch[i] = 0;
    }
    auto allocFrame = [&](std::unique_ptr<CUFrameBuf>& frame, const TCHAR *label) -> RGY_ERR {
        if (!frame
            || frame->frame.width != gaussInfo.width
            || frame->frame.height != gaussInfo.height
            || frame->frame.csp != gaussInfo.csp) {
            frame = std::make_unique<CUFrameBuf>(gaussInfo);
            frame->releasePtr();
            const auto allocSts = frame->alloc();
            if (allocSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF %s frame: %s.\n"), label, get_err_mes(allocSts));
                return allocSts;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocFrame(m_ucfNoiseGaussTmpFrames[frameIndex], _T("gauss temporary"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = allocFrame(m_ucfNoiseGaussFrames[frameIndex], _T("gauss output"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto *pTmpFrame = &m_ucfNoiseGaussTmpFrames[frameIndex]->frame;
    auto *pGaussFrame = &m_ucfNoiseGaussFrames[frameIndex]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pInputFrame->csp];
    const int fieldParity = parity & 1;
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        auto src = getPlane(pInputFrame, planeType);
        auto tmp = getPlane(pTmpFrame, planeType);
        auto dst = getPlane(pGaussFrame, planeType);
        if (!src.ptr[0] || !tmp.ptr[0] || !dst.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const int chromaIndex = (iplane == 0) ? 0 : 1;
        const bool interleavedUVPlane = interleavedUV && iplane != 0;
        const int dstWidthForGauss = interleavedUVPlane ? (dst.width >> 1) : dst.width;
        const double cropWidth = (fieldInfo.width > 0)
            ? (double)dstWidthForGauss + KFM_UCF_GAUSS_CROP_EPS * (double)dstWidthForGauss / fieldInfo.width
            : (double)dstWidthForGauss + KFM_UCF_GAUSS_CROP_EPS;
        const double cropHeight = (fieldInfo.height > 0)
            ? (double)dst.height + KFM_UCF_GAUSS_CROP_EPS * (double)dst.height / fieldInfo.height
            : (double)dst.height + KFM_UCF_GAUSS_CROP_EPS;
        auto& progV = m_ucfNoiseGaussVert[frameIndex][chromaIndex];
        auto& progH = m_ucfNoiseGaussHori[frameIndex][chromaIndex];
        if (!progV.offset || progV.sourceSize != dst.height || progV.targetSize != dst.height) {
            sts = createUcfGaussProgram(progV, dst.height, cropHeight, dst.height, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (!progH.offset || progH.sourceSize != dstWidthForGauss || progH.targetSize != dstWidthForGauss) {
            sts = createUcfGaussProgram(progH, dstWidthForGauss, cropWidth, dstWidthForGauss, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_field_crop_gaussresize_v_plane(&tmp, &src, srcXOffset, srcYOffset, srcYStep,
            (const int *)progV.offset->ptr, (const float *)progV.coeff->ptr, progV.filterSize, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_field_crop_gaussresize_v (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent evV;
        sts = kfmRecordEvent(stream, &evV);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = kfmWaitEvents(stream, { evV });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (interleavedUVPlane) {
            sts = run_kfm_ucf_gaussresize_h_uv_interleaved_plane(&dst, &tmp, dstWidthForGauss,
                (const int *)progH.offset->ptr, (const float *)progH.coeff->ptr, progH.filterSize, stream);
        } else {
            sts = run_kfm_ucf_gaussresize_h_plane(&dst, &tmp,
                (const int *)progH.offset->ptr, (const float *)progH.coeff->ptr, progH.filterSize, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at KFM UCF fused gauss h (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    copyFramePropWithoutRes(pGaussFrame, &fieldInfo);
    pGaussFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-noise-gauss", pGaussFrame);
    sts = dumpStageFrame("ucf-noise-gauss", pGaussFrame, sourceIndex * 2 + frameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppGaussFrame = pGaussFrame;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::runUcfNoiseLimitStageFromSource(const NVEncFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int fieldIndex, int parity, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    UNREFERENCED_PARAMETER(prm);
    if (!pSrcFrame || !pNoiseFrame || !pSrcFrame->ptr[0] || !pNoiseFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    const int cropX = 4;
    const int cropY = 4;
    if (pSrcFrame->width - cropX * 2 != pNoiseFrame->width || (pSrcFrame->height >> 1) - cropY * 2 != pNoiseFrame->height
        || pSrcFrame->csp != pNoiseFrame->csp) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF fused noise limit input pair (src %dx%d %s, noise %dx%d %s).\n"),
            pSrcFrame->width, pSrcFrame->height, RGY_CSP_NAMES[pSrcFrame->csp],
            pNoiseFrame->width, pNoiseFrame->height, RGY_CSP_NAMES[pNoiseFrame->csp]);
        return RGY_ERR_INVALID_PARAM;
    }
    KfmCachedUcfNoise entry;
    entry.fieldIndex = fieldIndex;
    entry.inputFrameId = pSrcFrame->inputFrameId;
    entry.timestamp = pSrcFrame->timestamp;
    entry.frame = acquireKfmFrame(*pNoiseFrame, _T("UCF fused noise limit"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto *pOutputFrame = &entry.frame->frame;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pSrcFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pSrcFrame->csp];
    const int fieldParity = parity & 1;
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        auto dst = getPlane(pOutputFrame, planeType);
        auto src = getPlane(pSrcFrame, planeType);
        auto noise = getPlane(pNoiseFrame, planeType);
        if (!dst.ptr[0] || !src.ptr[0] || !noise.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_source_crop_noise_limit_plane(&dst, &src, &noise,
            srcXOffset, srcYOffset, srcYStep, KFM_UCF_NOISE_LIMIT_NMIN, KFM_UCF_NOISE_LIMIT_RANGE, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_source_crop_noise_limit (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    copyFramePropWithoutRes(pOutputFrame, pNoiseFrame);
    pOutputFrame->dataList = pSrcFrame->dataList;
    writeFrameInfoDump("ucf-noise-clip", pOutputFrame);
    auto sts = dumpStageFrame("ucf-noise-clip", pOutputFrame, fieldIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prevEvent() != nullptr) {
        entry.event = prevEvent;
    }
    m_ucfNoiseCache.push_back(std::move(entry));
    while (m_ucfNoiseCache.size() > sourceCacheLimit()) {
        m_ucfNoiseCache.pop_front();
    }
    return analyzeUcfNoiseDebug(stream);
}

RGY_ERR NVEncFilterKfm::runUcfNoiseLimitStage(const NVEncFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int fieldIndex, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    UNREFERENCED_PARAMETER(prm);
    if (!pSrcFrame || !pNoiseFrame || !pSrcFrame->ptr[0] || !pNoiseFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (pSrcFrame->width != pNoiseFrame->width || pSrcFrame->height != pNoiseFrame->height || pSrcFrame->csp != pNoiseFrame->csp) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF noise limit input pair (src %dx%d %s, noise %dx%d %s).\n"),
            pSrcFrame->width, pSrcFrame->height, RGY_CSP_NAMES[pSrcFrame->csp],
            pNoiseFrame->width, pNoiseFrame->height, RGY_CSP_NAMES[pNoiseFrame->csp]);
        return RGY_ERR_INVALID_PARAM;
    }
    KfmCachedUcfNoise entry;
    entry.fieldIndex = fieldIndex;
    entry.inputFrameId = pSrcFrame->inputFrameId;
    entry.timestamp = pSrcFrame->timestamp;
    entry.frame = acquireKfmFrame(*pSrcFrame, _T("UCF noise limit"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto *pOutputFrame = &entry.frame->frame;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYCudaEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto src = getPlane(pSrcFrame, (RGY_PLANE)iplane);
        auto noise = getPlane(pNoiseFrame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
        auto sts = kfmWaitEvents(stream, waitHere);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_noise_limit_plane(&dst, &src, &noise, KFM_UCF_NOISE_LIMIT_NMIN, KFM_UCF_NOISE_LIMIT_RANGE, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_noise_limit (plane %d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    copyFramePropWithoutRes(pOutputFrame, pSrcFrame);
    pOutputFrame->dataList = pSrcFrame->dataList;
    writeFrameInfoDump("ucf-noise-clip", pOutputFrame);
    auto sts = dumpStageFrame("ucf-noise-clip", pOutputFrame, fieldIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prevEvent() != nullptr) {
        entry.event = prevEvent;
    }
    m_ucfNoiseCache.push_back(std::move(entry));
    while (m_ucfNoiseCache.size() > sourceCacheLimit()) {
        m_ucfNoiseCache.pop_front();
    }
    return analyzeUcfNoiseDebug(stream);
}

RGY_ERR NVEncFilterKfm::runDeint60Branch(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, int *cachedFrames) {
    return m_deint60Lane.feed(frame, stream, wait_events, cachedFrames);
}

RGY_ERR NVEncFilterKfm::drainDeint60Branch(cudaStream_t stream, int *cachedFrames) {
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    return m_deint60Lane.drain(stream, maxDrainIterations, cachedFrames);
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
        frame->releasePtr();
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
        frame->releasePtr();
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
        frame->releasePtr();
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

RGY_ERR NVEncFilterKfm::resolveContainsCombeCount(KfmContainsCombeReadback& readback, uint32_t *containsCombeCount) {
    if (!readback.submitted) {
        if (containsCombeCount) {
            *containsCombeCount = 0;
        }
        return RGY_ERR_NONE;
    }
    if (readback.event() != nullptr) {
        const auto waitSts = err_to_rgy(cudaEventSynchronize(readback.event()));
        if (waitSts != RGY_ERR_NONE) {
            readback.submitted = false;
            AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM contains-combe count readback: %s.\n"), get_err_mes(waitSts));
            return waitSts;
        }
    }
    if (containsCombeCount) {
        *containsCombeCount = readback.count;
    }
    readback.submitted = false;
    readback.event.reset();
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::renderMaskBranch(RGYFrameInfo *pSwitchFlagFrame, RGYFrameInfo *pContainsCombeFrame, RGYFrameInfo *pCombeMaskFrame,
    const RGYFrameInfo *pTelecineSuperPrevFrame, const RGYFrameInfo *pTelecineSuperFrame, const RGYFrameInfo *pTelecineSuperNextFrame,
    const char *switchFlagStage, const char *containsCombeStage, const char *combeMaskStage,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, KfmContainsCombeReadback *containsCombeReadback) {
    if (!pSwitchFlagFrame || !pContainsCombeFrame || !pCombeMaskFrame
        || !pTelecineSuperPrevFrame || !pTelecineSuperFrame || !pTelecineSuperNextFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    if (containsCombeReadback) {
        containsCombeReadback->count = 0;
        containsCombeReadback->event.reset();
        containsCombeReadback->submitted = false;
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
    auto cleanupContainsCombeReadback = [&]() {
        if (!containsCombeReadback) {
            return RGY_ERR_NONE;
        }
        return resolveContainsCombeCount(*containsCombeReadback, nullptr);
    };
    if (containsCombeReadback) {
        containsCombeReadback->count = 0;
        sts = kfmWaitEvents(stream, { countEvent });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = err_to_rgy(cudaMemcpyAsync(&containsCombeReadback->count, m_containsCombeCount->ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to submit KFM contains-combe count readback: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = kfmRecordEvent(stream, &containsCombeReadback->event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record KFM contains-combe count readback event: %s.\n"), get_err_mes(sts));
            const auto syncSts = err_to_rgy(cudaStreamSynchronize(stream));
            if (syncSts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM contains-combe count readback after event error: %s.\n"), get_err_mes(syncSts));
                return syncSts;
            }
            return sts;
        }
        containsCombeReadback->submitted = true;
    }

    auto containsY = getPlane(pContainsCombeFrame, RGY_PLANE_Y);
    sts = kfmWaitEvents(stream, { countEvent });
    if (sts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return sts;
    }
    sts = run_kfm_contains_combe_mark(&containsY, (const uint32_t *)m_containsCombeCount->ptr, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_mark: %s.\n"), get_err_mes(sts));
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return sts;
    }
    RGYCudaEvent markEvent;
    sts = kfmRecordEvent(stream, &markEvent);
    if (sts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
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
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return RGY_ERR_INVALID_PARAM;
        }
        sts = kfmWaitEvents(stream, { prevEvent });
        if (sts != RGY_ERR_NONE) {
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return sts;
        }
        sts = run_kfm_combe_mask_resize_bilinear_min_plane(&dst, &switchY,
            step, offset,
            scaleX, shiftX,
            scaleY, shiftY,
            innerWidth, innerHeight, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_combe_mask_resize_bilinear_min (plane %d): %s.\n"), iplane, get_err_mes(sts));
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return sts;
        }
        sts = kfmRecordEvent(stream, &prevEvent);
        if (sts != RGY_ERR_NONE) {
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return sts;
        }
    }

    copyFramePropWithoutRes(pSwitchFlagFrame, pTelecineSuperFrame);
    copyFramePropWithoutRes(pContainsCombeFrame, pTelecineSuperFrame);
    copyFramePropWithoutRes(pCombeMaskFrame, pTelecineSuperFrame);
    pSwitchFlagFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pContainsCombeFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pCombeMaskFrame->picstruct = RGY_PICSTRUCT_FRAME;
    writeFrameInfoDump(switchFlagStage, pSwitchFlagFrame);
    auto dumpSts = dumpStageFrame(switchFlagStage, pSwitchFlagFrame, maskDumpFrameIndex, stream, { switchEvent });
    if (dumpSts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return dumpSts;
    }
    writeFrameInfoDump(containsCombeStage, pContainsCombeFrame);
    dumpSts = dumpStageFrame(containsCombeStage, pContainsCombeFrame, maskDumpFrameIndex, stream, { markEvent });
    if (dumpSts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return dumpSts;
    }
    writeFrameInfoDump(combeMaskStage, pCombeMaskFrame);
    dumpSts = dumpStageFrame(combeMaskStage, pCombeMaskFrame, maskDumpFrameIndex, stream,
        (prevEvent() != nullptr) ? std::vector<RGYCudaEvent>{ prevEvent } : std::vector<RGYCudaEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return dumpSts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
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
            auto ucfOut = nextWorkFrame();
            if (!ucfOut) {
                return RGY_ERR_INVALID_CALL;
            }
            std::vector<RGYCudaEvent> ucfWaitEvents = frameWaitEvents;
            if (mergeEvent() != nullptr) {
                ucfWaitEvents.push_back(mergeEvent);
            }
            sts = resolveUcfNoiseResults((m_timecodeFrameIndex >> 1) + 3, stream);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            const auto *ucfInput = selectUcfDecomb60Frame(m_timecodeFrameIndex, out, &ucfWaitEvents);
            RGYCudaEvent ucfEvent;
            sts = copyUcfFrame(prm, ucfOut, ucfInput, stream, ucfWaitEvents, &ucfEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            copyFramePropWithoutRes(ucfOut, out);
            sts = emitOutputFrame(ucfOut, ppOutputFrames, pOutputFrameNum, stream, ucfEvent, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
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

RGY_ERR NVEncFilterKfm::analyzeUcfNoiseDebug(cudaStream_t stream) {
    if (m_ucfNoiseCache.size() < 3) {
        return RGY_ERR_NONE;
    }
    while (m_ucfNoiseCache.size() >= 3) {
        const auto& noise0 = m_ucfNoiseCache[0];
        const auto& noise1 = m_ucfNoiseCache[1];
        const auto& noise2 = m_ucfNoiseCache[2];
        if (noise0.fieldIndex < 0 || (noise0.fieldIndex & 1) != 0
            || noise1.fieldIndex != noise0.fieldIndex + 1
            || noise2.fieldIndex != noise0.fieldIndex + 2) {
            m_ucfNoiseCache.pop_front();
            continue;
        }
        const int sourceIndex = noise0.fieldIndex >> 1;
        const auto *source0 = findSourceByIndexExact(sourceIndex);
        const auto *source1 = findSourceByIndexExact(sourceIndex + 1);
        if (!source0 || !source1 || !source0->paddedFrame || !source1->paddedFrame) {
            break;
        }
        auto sts = submitUcfNoiseResult(noise0, noise1, noise2, *source0, *source1, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_ucfNoiseCache.pop_front();
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::submitUcfNoiseResult(const KfmCachedUcfNoise& noise0, const KfmCachedUcfNoise& noise1, const KfmCachedUcfNoise& noise2,
    const KfmCachedSource& source0, const KfmCachedSource& source1, cudaStream_t stream) {
    if (!noise0.frame || !noise1.frame || !noise2.frame || !source0.paddedFrame || !source1.paddedFrame) {
        return RGY_ERR_INVALID_CALL;
    }

    KfmPendingUcfNoiseResult pending;
    pending.sourceIndex = noise0.fieldIndex >> 1;
    pending.meta.srcw = source0.frame->frame.width;
    pending.meta.srch = source0.frame->frame.height;
    pending.meta.srcUVw = pending.meta.srcw >> 1;
    pending.meta.srcUVh = pending.meta.srch >> 1;
    const auto noiseY = getPlane(&noise0.frame->frame, RGY_PLANE_Y);
    const auto noiseUV = (RGY_CSP_PLANES[noise0.frame->frame.csp] > 1) ? getPlane(&noise0.frame->frame, RGY_PLANE_U) : RGYFrameInfo();
    pending.meta.noisew = noiseY.width;
    pending.meta.noiseh = noiseY.height;
    pending.meta.noiseUVw = noiseUV.ptr[0] ? (kfmCspHasInterleavedUV(noise0.frame->frame.csp) ? (noiseUV.width >> 1) : noiseUV.width) : 0;
    pending.meta.noiseUVh = noiseUV.ptr[0] ? noiseUV.height : 0;

    std::vector<RGYCudaEvent> events;
    int partialCount = 0;
    const int localX = 32;
    const int localY = 8;
    const auto reservePartials = [&](int width4, int height) {
        return divCeil(width4, localX) * divCeil(height, localY);
    };
    const auto addPartialCount = [&](const RGYFrameInfo *frame, bool diff) {
        const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[frame->csp];
        const int planes = RGY_CSP_PLANES[frame->csp];
        for (int iplane = 0; iplane < planes; iplane++) {
            const auto plane = getPlane(frame, (RGY_PLANE)iplane);
            const int width4 = plane.width >> 2;
            const bool chromaPlane = iplane != 0 && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
            const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
            const int vpad = KFM_SOURCE_VPAD >> yShift;
            const int height = diff ? plane.height - vpad * 2 : plane.height;
            if (width4 > 0 && height > 0 && (plane.width & 3) == 0) {
                partialCount += reservePartials(width4, height);
            }
        }
    };
    addPartialCount(&noise0.frame->frame, false);
    addPartialCount(&source0.paddedFrame->frame, true);
    if (partialCount <= 0) {
        RGYKFM::NoiseResult results[2] = {};
        pushUcfNoiseResultDump(pending.sourceIndex, results, pending.meta);
        return RGY_ERR_NONE;
    }
    const size_t requiredBytes = sizeof(RGYKFM::NoiseResult) * (size_t)partialCount;
    pending.resultBuf = acquireUcfNoiseResultBuf(requiredBytes);
    if (!pending.resultBuf) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF noise result buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    int offset = 0;
    const auto launchNoise = [&](RGY_PLANE planeType, int resultPlane, std::vector<RGYCudaEvent> waitEvents) -> RGY_ERR {
        auto p0 = getPlane(&noise0.frame->frame, planeType);
        auto p1 = getPlane(&noise1.frame->frame, planeType);
        auto p2 = getPlane(&noise2.frame->frame, planeType);
        const int width4 = p0.width >> 2;
        if (!p0.ptr[0] || !p1.ptr[0] || !p2.ptr[0] || width4 <= 0 || p0.height <= 0 || (p0.width & 3) != 0) {
            return RGY_ERR_NONE;
        }
        if (p0.width != p1.width || p0.width != p2.width || p0.height != p1.height || p0.height != p2.height) {
            return RGY_ERR_INVALID_PARAM;
        }
        const int count = reservePartials(width4, p0.height);
        auto sts = kfmWaitEvents(stream, waitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_analyze_noise_partial((RGYKFM::NoiseResult *)pending.resultBuf->ptrDevice, offset, &p0, &p1, &p2, width4, p0.height, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_analyze_noise_partial (plane %d): %s.\n"), (int)planeType, get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent event;
        sts = kfmRecordEvent(stream, &event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        pending.segments.push_back({ offset, count, resultPlane });
        offset += count;
        if (event() != nullptr) {
            events.push_back(event);
        }
        return RGY_ERR_NONE;
    };
    const auto launchDiff = [&](RGY_PLANE planeType, int resultPlane, std::vector<RGYCudaEvent> waitEvents) -> RGY_ERR {
        auto p0 = getPlane(&source0.paddedFrame->frame, planeType);
        auto p1 = getPlane(&source1.paddedFrame->frame, planeType);
        const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[source0.paddedFrame->frame.csp];
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int vpad = KFM_SOURCE_VPAD >> yShift;
        const int height = p0.height - vpad * 2;
        const int width4 = p0.width >> 2;
        if (!p0.ptr[0] || !p1.ptr[0] || width4 <= 0 || height <= 0 || (p0.width & 3) != 0) {
            return RGY_ERR_NONE;
        }
        if (p0.width != p1.width || p0.height != p1.height) {
            return RGY_ERR_INVALID_PARAM;
        }
        const int count = reservePartials(width4, height);
        auto sts = kfmWaitEvents(stream, waitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = run_kfm_ucf_analyze_diff_partial((RGYKFM::NoiseResult *)pending.resultBuf->ptrDevice, offset, &p0, &p1, width4, height, vpad, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_analyze_diff_partial (plane %d): %s.\n"), (int)planeType, get_err_mes(sts));
            return sts;
        }
        RGYCudaEvent event;
        sts = kfmRecordEvent(stream, &event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        pending.segments.push_back({ offset, count, resultPlane });
        offset += count;
        if (event() != nullptr) {
            events.push_back(event);
        }
        return RGY_ERR_NONE;
    };

    std::vector<RGYCudaEvent> noiseWaits;
    if (noise0.event() != nullptr) noiseWaits.push_back(noise0.event);
    if (noise1.event() != nullptr) noiseWaits.push_back(noise1.event);
    if (noise2.event() != nullptr) noiseWaits.push_back(noise2.event);
    auto sts = launchNoise(RGY_PLANE_Y, 0, noiseWaits);
    if (sts != RGY_ERR_NONE) return sts;
    const bool interleavedUV = kfmCspHasInterleavedUV(noise0.frame->frame.csp);
    if (interleavedUV) {
        sts = launchNoise(RGY_PLANE_U, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (RGY_CSP_PLANES[noise0.frame->frame.csp] >= 3) {
        sts = launchNoise(RGY_PLANE_U, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
        sts = launchNoise(RGY_PLANE_V, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
    }

    std::vector<RGYCudaEvent> diffWaits;
    if (source0.paddedEvent() != nullptr) diffWaits.push_back(source0.paddedEvent);
    if (source1.paddedEvent() != nullptr) diffWaits.push_back(source1.paddedEvent);
    sts = launchDiff(RGY_PLANE_Y, 0, diffWaits);
    if (sts != RGY_ERR_NONE) return sts;
    if (interleavedUV) {
        sts = launchDiff(RGY_PLANE_U, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (RGY_CSP_PLANES[source0.paddedFrame->frame.csp] >= 3) {
        sts = launchDiff(RGY_PLANE_U, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
        sts = launchDiff(RGY_PLANE_V, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
    }

    sts = kfmWaitEvents(stream, events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaMemcpyAsync(pending.resultBuf->ptrHost, pending.resultBuf->ptrDevice, pending.resultBuf->nSize, cudaMemcpyDeviceToHost, stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM UCF noise result buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = kfmRecordEvent(stream, &pending.event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_pendingUcfNoiseResults.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::resolveUcfNoiseResult(KfmPendingUcfNoiseResult& pending, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(stream);
    if (!pending.resultBuf) {
        return RGY_ERR_NULL_PTR;
    }
    if (pending.event() != nullptr) {
        const auto waitSts = err_to_rgy(cudaEventSynchronize(pending.event()));
        if (waitSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM UCF noise result copy event: %s.\n"), get_err_mes(waitSts));
            return waitSts;
        }
    }
    const auto *partials = reinterpret_cast<const RGYKFM::NoiseResult *>(pending.resultBuf->ptrHost);
    if (!partials) {
        return RGY_ERR_NULL_PTR;
    }
    RGYKFM::NoiseResult results[2] = {};
    for (const auto& segment : pending.segments) {
        auto& result = results[segment.plane];
        for (int i = 0; i < segment.count; i++) {
            const auto& partial = partials[segment.offset + i];
            result.noise0 += partial.noise0;
            result.noise1 += partial.noise1;
            result.noiseR0 += partial.noiseR0;
            result.noiseR1 += partial.noiseR1;
            result.diff0 += partial.diff0;
            result.diff1 += partial.diff1;
        }
    }
    pushUcfNoiseResultDump(pending.sourceIndex, results, pending.meta);
    releaseUcfNoiseResultBuf(std::move(pending.resultBuf));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::resolveUcfNoiseResults(int sourceIndex, cudaStream_t stream) {
    while (!m_pendingUcfNoiseResults.empty() && m_pendingUcfNoiseResults.front().sourceIndex <= sourceIndex) {
        auto sts = resolveUcfNoiseResult(m_pendingUcfNoiseResults.front(), stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_pendingUcfNoiseResults.pop_front();
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterKfm::resolveAllUcfNoiseResults(cudaStream_t stream) {
    while (!m_pendingUcfNoiseResults.empty()) {
        auto sts = resolveUcfNoiseResult(m_pendingUcfNoiseResults.front(), stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_pendingUcfNoiseResults.pop_front();
    }
    return RGY_ERR_NONE;
}

std::unique_ptr<CUMemBufPair> NVEncFilterKfm::acquireUcfNoiseResultBuf(size_t requiredBytes) {
    auto it = std::find_if(m_ucfNoiseResultBufPool.begin(), m_ucfNoiseResultBufPool.end(),
        [requiredBytes](const std::unique_ptr<CUMemBufPair>& buf) {
            return buf && buf->nSize >= requiredBytes;
        });
    if (it != m_ucfNoiseResultBufPool.end()) {
        auto buf = std::move(*it);
        m_ucfNoiseResultBufPool.erase(it);
        return buf;
    }
    auto buf = std::make_unique<CUMemBufPair>(requiredBytes);
    const auto sts = buf->alloc();
    if (sts != RGY_ERR_NONE) {
        return nullptr;
    }
    return buf;
}

void NVEncFilterKfm::releaseUcfNoiseResultBuf(std::unique_ptr<CUMemBufPair>&& buf) {
    if (!buf) {
        return;
    }
    static constexpr size_t KFM_UCF_NOISE_RESULT_BUF_POOL_MAX = 8;
    m_ucfNoiseResultBufPool.push_back(std::move(buf));
    while (m_ucfNoiseResultBufPool.size() > KFM_UCF_NOISE_RESULT_BUF_POOL_MAX) {
        m_ucfNoiseResultBufPool.pop_front();
    }
}

RGY_ERR NVEncFilterKfm::clearPendingFMCounts() {
    if (m_pendingFMCounts.empty()) {
        return RGY_ERR_NONE;
    }

    RGY_ERR sts = RGY_ERR_NONE;
    for (auto& pending : m_pendingFMCounts) {
        if (pending.event() != nullptr) {
            const auto waitSts = err_to_rgy(cudaEventSynchronize(pending.event()));
            if (waitSts != RGY_ERR_NONE && sts == RGY_ERR_NONE) {
                sts = waitSts;
            }
        }
        pending.countBuf.reset();
    }
    m_pendingFMCounts.clear();
    return sts;
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
        if (!src[i] || !src[i]->frame || !src[i]->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    const size_t countBytes = sizeof(RGYKFM::FMCount) * KFM_FMCOUNT_PAIRS * 2;
    KfmPendingFMCount pending;
    pending.cycle = cycle;
    pending.countBuf = std::make_unique<CUMemBufPair>(countBytes);
    auto sts = pending.countBuf->alloc();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = err_to_rgy(cudaMemsetAsync(pending.countBuf->ptrDevice, 0, countBytes, stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }

    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
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
            if (src[pair + 0]->event() != nullptr) {
                countWaitEvents.push_back(src[pair + 0]->event);
            }
            if (src[pair + 1]->event() != nullptr) {
                countWaitEvents.push_back(src[pair + 1]->event);
            }
            if (src[pair + 2]->event() != nullptr) {
                countWaitEvents.push_back(src[pair + 2]->event);
            }
            sts = kfmWaitEvents(stream, countWaitEvents);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            const bool chroma = iplane > 0;
            const int threshMove = chroma ? KFM_THRESH_MOVE_C : KFM_THRESH_MOVE_Y;
            const int threshShima = chroma ? KFM_THRESH_SHIMA_C : KFM_THRESH_SHIMA_Y;
            const int cleanThresh = chroma ? KFM_CLEAN_THRESH_C : KFM_CLEAN_THRESH_Y;
            const int dstOffset = pair * 2;
            sts = run_kfm_analyze_count_cmflags_clean(
                reinterpret_cast<RGYKFM::FMCount *>(pending.countBuf->ptrDevice),
                dstOffset,
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

    }
    sts = err_to_rgy(cudaMemcpyAsync(pending.countBuf->ptrHost, pending.countBuf->ptrDevice, countBytes, cudaMemcpyDeviceToHost, stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = kfmRecordEvent(stream, &pending.event);
    if (sts != RGY_ERR_NONE) {
        return sts;
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
    auto& fmCountBuf = pending.countBuf;
    if (!fmCountBuf) {
        AddMessage(RGY_LOG_ERROR, _T("KFM FMCount pending buffer is missing.\n"));
        return RGY_ERR_NULL_PTR;
    }
    if (pending.event() != nullptr) {
        const auto waitSts = err_to_rgy(cudaEventSynchronize(pending.event()));
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
    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
        const int countFrameIndex = pending.cycle * 5 - 3 + pair + 1;
        if (countFrameIndex >= 0) {
            counts[pair * 2 + 0] = gpuCounts[pair * 2 + 0];
            counts[pair * 2 + 1] = gpuCounts[pair * 2 + 1];
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
    if (m_pendingUcfNoiseDump.valid) {
        writeUcfNoiseResultDump(m_pendingUcfNoiseDump, nullptr);
        m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();
    }
}

void NVEncFilterKfm::pushUcfNoiseResultDump(int sourceIndex, const RGYKFM::NoiseResult (&results)[2], const RGYKFM::UCFNoiseMeta& meta) {
    KfmUcfNoiseDumpRecord record;
    record.sourceIndex = sourceIndex;
    record.results[0] = results[0];
    record.results[1] = results[1];
    record.meta = meta;
    record.valid = true;

    m_ucfNoiseResultCache.push_back(record);
    auto neededByPending = [this](int sourceIndex) {
        for (const auto& pending : m_pendingUcfNoiseResults) {
            if (sourceIndex == pending.sourceIndex || sourceIndex == pending.sourceIndex + 1) {
                return true;
            }
        }
        return false;
    };
    while (m_ucfNoiseResultCache.size() > sourceCacheLimit() && !neededByPending(m_ucfNoiseResultCache.front().sourceIndex)) {
        m_ucfNoiseResultCache.pop_front();
    }

    if (m_fpUcfNoise) {
        if (m_pendingUcfNoiseDump.valid) {
            writeUcfNoiseResultDump(m_pendingUcfNoiseDump, &record);
        }
        m_pendingUcfNoiseDump = record;
    }
}

void NVEncFilterKfm::writeUcfNoiseResultDump(const KfmUcfNoiseDumpRecord& record, const KfmUcfNoiseDumpRecord *nextRecord) {
    if (!m_fpUcfNoise) {
        return;
    }
    KfmUcfCalcDumpInfo calc0 = {};
    KfmUcfCalcDumpInfo calc1 = {};
    bool hasCalc0 = false;
    bool hasCalc1 = false;
    try {
        calc0 = kfmUcfCalcDumpInfo(record.meta, record.results, nullptr, false);
        hasCalc0 = true;
        if (nextRecord) {
            calc1 = kfmUcfCalcDumpInfo(record.meta, record.results, nextRecord->results, true);
            hasCalc1 = true;
        }
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_WARN, _T("failed to calculate KFM UCF classification dump for frame %d: %S.\n"), record.sourceIndex, e.what());
    }
    static const char *planeNames[2] = { "Y", "UV" };
    for (int i = 0; i < 2; i++) {
        const auto& r = record.results[i];
        fprintf(m_fpUcfNoise, "%d\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\t%s\t",
            record.sourceIndex, planeNames[i],
            (unsigned long long)r.noise0,
            (unsigned long long)r.noise1,
            (unsigned long long)r.noiseR0,
            (unsigned long long)r.noiseR1,
            (unsigned long long)r.diff0,
            (unsigned long long)r.diff1,
            hasCalc0 ? calc0.classification : "",
            hasCalc1 ? calc1.classification : "");
        if (hasCalc0) {
            fprintf(m_fpUcfNoise, "%.12g", calc0.fieldDiff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc1) {
            fprintf(m_fpUcfNoise, "%.12g", calc1.fieldDiff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc0) {
            fprintf(m_fpUcfNoise, "%.12g", calc0.diff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc1) {
            fprintf(m_fpUcfNoise, "%.12g", calc1.diff);
        }
        fprintf(m_fpUcfNoise, "\n");
    }
    fflush(m_fpUcfNoise);
}

const RGYFrameInfo *NVEncFilterKfm::selectUcfDecomb30Frame(int sourceIndex, const RGYFrameInfo *deint30, std::vector<RGYCudaEvent> *wait_events) const {
    if (!deint30 || sourceIndex < 0) {
        return deint30;
    }
    const auto selection = planUcfDecomb30Frame(sourceIndex);
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (selection.type == KFM_UCF24_SELECT_FRAME && lane && selection.n60 >= 0) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        return (entry && entry->frame && entry->frame->frame.ptr[0]) ? &entry->frame->frame : deint30;
    }
    return deint30;
}

NVEncFilterKfm::KfmUcf24Selection NVEncFilterKfm::planUcfDecomb30Frame(int sourceIndex) const {
    KfmUcf24Selection selection;
    if (sourceIndex < 0) {
        return selection;
    }
    const auto *noise = findUcfNoiseResult(sourceIndex);
    if (!noise || !noise->valid) {
        return selection;
    }

    RGYKFM::DecombUCFParam param;
    RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
    try {
        result = RGYKFM::CalcDecombUCF(&noise->meta, &param, noise->results, nullptr, false);
    } catch (...) {
        return selection;
    }

    const int n60 = sourceIndex * 2;
    if (result == RGYKFM::DECOMB_UCF_USE_0) {
        selection.type = KFM_UCF24_SELECT_FRAME;
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60;
        return selection;
    }
    if (result == RGYKFM::DECOMB_UCF_USE_1) {
        selection.type = KFM_UCF24_SELECT_FRAME;
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60 + 1;
        return selection;
    }
    return selection;
}

bool NVEncFilterKfm::getUcf60FieldDiff(int nstart, double (&diff)[4]) const {
    for (int i = 0; i < 4; i++) {
        const int n = nstart + i;
        const auto *noise = findUcfNoiseResult(n / 2);
        if (!noise || !noise->valid || noise->meta.srcw <= 0 || noise->meta.srch <= 0) {
            return false;
        }
        const double pixels = (double)noise->meta.srcw * noise->meta.srch;
        const uint64_t diffY = (n & 1) ? noise->results[0].diff1 : noise->results[0].diff0;
        const uint64_t diffUV = (n & 1) ? noise->results[1].diff1 : noise->results[1].diff0;
        diff[i] = (double)(diffY + diffUV) / (6.0 * pixels) * 100.0;
    }
    return true;
}

NVEncFilterKfm::KfmUcf60Flag NVEncFilterKfm::calcUcf60Flag(int n60) const {
    static const RGYKFM::DECOMB_UCF_RESULT replaceResults[2] = {
        RGYKFM::DECOMB_UCF_USE_0,
        RGYKFM::DECOMB_UCF_USE_1,
    };
    static constexpr double UCF60_SC_THRESH = 256.0;
    static constexpr double UCF60_DUP_THRESH = 2.5;

    RGYKFM::DecombUCFParam param;
    int useFrame = n60;
    bool isDirty = false;
    for (int i = 0; i < 2; i++) {
        const int n = n60 + i - 1;
        const auto *noise0 = findUcfNoiseResult(n / 2);
        const auto *noise1 = findUcfNoiseResult(n / 2 + 1);
        if (!noise0 || !noise0->valid || !noise1 || !noise1->valid) {
            continue;
        }

        RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
        try {
            result = RGYKFM::CalcDecombUCF(&noise0->meta, &param, noise0->results, noise1->results, (n & 1) != 0);
        } catch (...) {
            continue;
        }

        if (result == replaceResults[i]) {
            double diff[4] = {};
            if (i == 0) {
                if (getUcf60FieldDiff(n60 - 3, diff)) {
                    const double sc = diff[3] / (std::max(diff[0], diff[1]) + 0.0001);
                    if (sc > UCF60_DUP_THRESH && diff[3] > UCF60_SC_THRESH) {
                        useFrame = n60 - 1;
                    }
                }
            } else {
                if (getUcf60FieldDiff(n60 - 1, diff)) {
                    const double sc = diff[0] / (std::max(diff[2], diff[3]) + 0.0001);
                    if (sc > UCF60_DUP_THRESH && diff[0] > UCF60_SC_THRESH) {
                        useFrame = n60 + 1;
                    }
                }
            }
        } else if (result == RGYKFM::DECOMB_UCF_NOISY) {
            isDirty = true;
        }
    }

    if (useFrame == n60 && isDirty) {
        return KFM_UCF60_NR;
    }
    if (useFrame < n60) {
        return KFM_UCF60_PREV;
    }
    if (useFrame > n60) {
        return KFM_UCF60_NEXT;
    }
    return KFM_UCF60_NONE;
}

const RGYFrameInfo *NVEncFilterKfm::selectUcfDecomb60Frame(int n60, const RGYFrameInfo *deint60, std::vector<RGYCudaEvent> *wait_events) const {
    if (!deint60 || n60 < 0) {
        return deint60;
    }
    const auto selection = planUcfDecomb60Frame(n60);
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (lane && selection.n60 >= 0) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        return (entry && entry->frame && entry->frame->frame.ptr[0]) ? &entry->frame->frame : deint60;
    }
    return deint60;
}

NVEncFilterKfm::KfmUcf60Selection NVEncFilterKfm::planUcfDecomb60Frame(int n60) const {
    KfmUcf60Selection selection;
    if (n60 < 0) {
        return selection;
    }
    const auto centerFlag = calcUcf60Flag(n60);
    KfmUcf60Flag sideFlag = KFM_UCF60_NONE;
    for (int i = -1; i <= 1; i += 2) {
        const auto flag = calcUcf60Flag(n60 + i);
        if (flag == KFM_UCF60_PREV || flag == KFM_UCF60_NEXT) {
            if (i == -1) {
                sideFlag = KFM_UCF60_NEXT;
            } else {
                sideFlag = KFM_UCF60_PREV;
            }
        }
    }

    if (centerFlag == KFM_UCF60_PREV) {
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60 - 1;
    } else if (centerFlag == KFM_UCF60_NEXT) {
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60 + 1;
    } else if (sideFlag == KFM_UCF60_PREV) {
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60;
    } else if (sideFlag == KFM_UCF60_NEXT) {
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60;
    }
    return selection;
}

NVEncFilterKfm::KfmUcf24Selection NVEncFilterKfm::selectUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo, const RGYFrameInfo *deint24, std::vector<RGYCudaEvent> *wait_events) const {
    auto selection = planUcfDecomb24Frame(frameInfo);
    selection.frame = deint24;
    if (selection.type != KFM_UCF24_SELECT_FRAME || selection.n60 < 0) {
        return selection;
    }
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (lane) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        if (entry && entry->frame && entry->frame->frame.ptr[0]) {
            selection.frame = &entry->frame->frame;
        }
    }
    return selection;
}

NVEncFilterKfm::KfmUcf24Selection NVEncFilterKfm::planUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo) const {
    KfmUcf24Selection selection;
    if (frameInfo.numFields <= 0 || frameInfo.numFields > 6) {
        return selection;
    }

    bool cleanField[6] = { true, true, true, true, true, true };
    RGYKFM::DecombUCFParam param;
    for (int i = 0; i < frameInfo.numFields - 1; i++) {
        const int n60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + i;
        const auto *noise0 = findUcfNoiseResult(n60 / 2);
        const auto *noise1 = findUcfNoiseResult(n60 / 2 + 1);
        if (!noise0 || !noise0->valid || !noise1 || !noise1->valid) {
            return selection;
        }

        RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
        try {
            result = RGYKFM::CalcDecombUCF(&noise0->meta, &param, noise0->results, noise1->results, (n60 & 1) != 0);
        } catch (...) {
            return selection;
        }

        if (result == RGYKFM::DECOMB_UCF_USE_0) {
            cleanField[i + 1] = false;
        } else if (result == RGYKFM::DECOMB_UCF_USE_1) {
            cleanField[i + 0] = false;
        } else if (result == RGYKFM::DECOMB_UCF_NOISY) {
            cleanField[i + 0] = false;
            cleanField[i + 1] = false;
        }
    }

    const bool hasDirty = std::find(cleanField, cleanField + frameInfo.numFields, false) != cleanField + frameInfo.numFields;
    if (!hasDirty) {
        return selection;
    }
    for (int i = 0; i < frameInfo.numFields - 1; i++) {
        if (cleanField[i] && cleanField[i + 1]) {
            selection.type = KFM_UCF24_SELECT_DWEAVE;
            selection.n60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + i;
            selection.frame = nullptr;
            return selection;
        }
    }
    if (frameInfo.numFields <= 2) {
        const int n60start = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
        if (cleanField[0]) {
            selection.type = KFM_UCF24_SELECT_FRAME;
            selection.lane = KFM_UCF_LANE_BEFORE;
            selection.n60 = n60start;
            return selection;
        }
        if (cleanField[1]) {
            selection.type = KFM_UCF24_SELECT_FRAME;
            selection.lane = KFM_UCF_LANE_AFTER;
            selection.n60 = n60start + 1;
            return selection;
        }
    }
    return selection;
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
        int rtgmcOutNum = 0;
        RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
        RGYCudaEvent rtgmcEvent;
        sts = m_rtgmc->filter(const_cast<RGYFrameInfo *>(pInputFrame), rtgmcOutFrames, &rtgmcOutNum, stream, {}, &rtgmcEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            for (auto &captured : m_rtgmc->getCapturedIntermediates()) {
                if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
            }
            m_rtgmc->clearCapturedIntermediates();
            sts = runUcfRtgmcBranches(pInputFrame, stream, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(stream);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
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

    if (prm->kfm.mode == VppKfmMode::VFR) {
        auto sts = RGY_ERR_NONE;
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, stream);
        } else {
            sts = cacheSourceFrame(pInputFrame, stream, {});
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        const bool lazyDeint60 = lazyDeint60Enabled(*prm);
        sts = lazyDeint60
            ? ((pInputFrame && pInputFrame->ptr[0]) ? m_deint60Lane.feedHot(stream) : RGY_ERR_NONE)
            : ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) ? drainDeint60Branch(stream) : runDeint60Branch(pInputFrame, stream, {}));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!lazyDeint60 && prm->kfm.ucf && m_deint60Rtgmc) {
            for (auto &captured : m_deint60Rtgmc->getCapturedIntermediates()) {
                if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
            }
            m_deint60Rtgmc->clearCapturedIntermediates();
        }
        if (prm->kfm.ucf) {
            sts = lazyDeint60
                ? runUcfNoiseAnalysisFromSource(pInputFrame, stream, {})
                : runUcfRtgmcBranches(pInputFrame, stream, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(stream);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }

        const auto ensureFrame = [&](std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& info, const TCHAR *label) -> RGY_ERR {
            if (!frame
                || frame->frame.width != info.width
                || frame->frame.height != info.height
                || frame->frame.csp != info.csp) {
                auto newFrame = std::make_unique<CUFrameBuf>(info);
                newFrame->releasePtr();
                const auto allocSts = newFrame->alloc();
                if (allocSts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s frame: %s.\n"), label, get_err_mes(allocSts));
                    return allocSts;
                }
                frame = std::move(newFrame);
            }
            return RGY_ERR_NONE;
        };
        const auto copyFrameWithEvent = [&](RGYFrameInfo *dst, const RGYFrameInfo *src, const std::vector<RGYCudaEvent>& waitEvents, RGYCudaEvent *copyEvent, const TCHAR *label) -> RGY_ERR {
            auto copySts = kfmWaitEvents(stream, waitEvents);
            if (copySts != RGY_ERR_NONE) {
                return copySts;
            }
            copySts = copyFrameAsync(dst, src, stream);
            if (copySts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR %s frame: %s.\n"), label, get_err_mes(copySts));
                return copySts;
            }
            return kfmRecordEvent(stream, copyEvent);
        };
        auto ensureDeint60Range = [&](int n60begin, int n60end) -> RGY_ERR {
            return lazyDeint60 ? m_deint60Lane.ensureRange(n60begin, n60end, stream) : RGY_ERR_NONE;
        };

        *pOutputFrameNum = 0;
        const bool drain = pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr;
        const int rawAvailableN60 = drain
            ? m_cachedSourceFrames * 2
            : std::min(m_cachedSourceFrames * 2, static_cast<int>(m_analyzerOutputResults.size()) * 10);
        const int vfrTailHold60 = switchSingleFrameDurationEnabled() ? 8 : 4;
        const int availableN60 = drain ? rawAvailableN60 : std::max(0, rawAvailableN60 - vfrTailHold60);
        const auto timings = deriveSwitchTimings(availableN60);
        const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
        const int vfrOutputDelay = switchSingleFrameDurationEnabled() ? 1 : 0;
        auto emitReadyPending = [&](int keepFrames) -> RGY_ERR {
            return emitPendingVfrOutputs(ppOutputFrames, pOutputFrameNum, stream, nullptr, keepFrames);
        };
        sts = emitReadyPending(drain ? 0 : vfrOutputDelay);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        while (*pOutputFrameNum < maxOutputFrames) {
            auto itTiming = std::find_if(timings.begin(), timings.end(), [this](const KfmSwitchTiming& timing) {
                return timing.start60 == m_nextSwitchN60;
            });
            if (itTiming == timings.end()) {
                itTiming = std::find_if(timings.begin(), timings.end(), [this](const KfmSwitchTiming& timing) {
                    return timing.start60 < m_nextSwitchN60 && m_nextSwitchN60 < timing.start60 + timing.duration60;
                });
                if (itTiming == timings.end()) {
                    break;
                }
            }
            auto outputTiming = *itTiming;
            if (outputTiming.start60 < m_nextSwitchN60) {
                const auto consumed60 = m_nextSwitchN60 - outputTiming.start60;
                outputTiming.start60 = m_nextSwitchN60;
                outputTiming.start120 += consumed60 * 2;
                outputTiming.duration60 = std::max(1, outputTiming.duration60 - consumed60);
                outputTiming.duration120 = outputTiming.duration60 * 2;
                outputTiming.numSourceFrames = std::max(1, divCeil(outputTiming.duration60, 2));
            }
            if (!drain && outputTiming.start60 + outputTiming.duration60 >= availableN60) {
                break;
            }
            const auto rawStart120 = [](const KfmSwitchTiming& timing) {
                return static_cast<int64_t>(timing.start60) * 2;
            };
            auto sourcePtsFrom120 = [&](const int64_t pos120) {
                if (m_sourceCache.empty()) {
                    return pos120;
                }
                const int64_t sourceIndex = pos120 >> 2;
                const int offset120 = static_cast<int>(pos120 & 3);
                const auto *source = findSourceByIndex(static_cast<int>(sourceIndex));
                const auto duration = sourceFrameDuration(source);
                if (!source || source->timestamp < 0) {
                    return (duration * pos120 + 2) / 4;
                }
                const int64_t sourceOffset120 = (sourceIndex - source->sourceIndex) * 4 + offset120;
                return source->timestamp + (duration * sourceOffset120 + 2) / 4;
            };
            const auto canUse120Cadence = [](bool prevIsFrame24, int prevDuration60, const KfmSwitchTiming& cur) {
                return prevIsFrame24 && cur.isFrame24
                    && prevDuration60 >= 2 && cur.duration60 >= 2
                    && prevDuration60 + cur.duration60 == 5;
            };
            int64_t outputStart120 = rawStart120(outputTiming);
            if (prm->kfm.is120
                && m_hasLastSwitchTiming
                && m_lastSwitchStart60 + m_lastSwitchDuration60 == outputTiming.start60
                && canUse120Cadence(m_lastSwitchIsFrame24, m_lastSwitchDuration60, outputTiming)) {
                outputStart120 = m_lastSwitchStart120 + 5;
            }
            int64_t nextStart120 = outputStart120 + outputTiming.duration60 * 2;
            const auto itNextTiming = std::find_if(timings.begin(), timings.end(), [&outputTiming](const KfmSwitchTiming& timing) {
                return timing.start60 == outputTiming.start60 + outputTiming.duration60;
            });
            if (itNextTiming != timings.end()) {
                nextStart120 = rawStart120(*itNextTiming);
                if (prm->kfm.is120 && canUse120Cadence(outputTiming.isFrame24, outputTiming.duration60, *itNextTiming)) {
                    nextStart120 = outputStart120 + 5;
                }
            }
            outputTiming.start120 = static_cast<int>(outputStart120);
            if (nextStart120 > outputStart120) {
                outputTiming.duration120 = static_cast<int>(nextStart120 - outputStart120);
            }
            const int copySourceIndex = outputTiming.baseType == KFM_FRAME_60 ? (outputTiming.start60 >> 1) : outputTiming.sourceIndex;
            const auto *source = findSourceByIndex(copySourceIndex);
            const auto *switchResult = &m_analyzerOutputResults[clamp(outputTiming.start60 / 10, 0, (int)m_analyzerOutputResults.size() - 1)];
            RGYCudaEvent outputEvent;
            RGYFrameInfo *out = nullptr;
            if (outputTiming.baseType == KFM_FRAME_24) {
                const auto savedWorkBufferIndex = m_workBufferIndex;
                const auto savedTelecineSuperBufferIndex = m_telecineSuperBufferIndex;
                auto deint24 = nextWorkFrame();
                out = nextWorkFrame();
                if (!deint24 || !out) {
                    return RGY_ERR_INVALID_CALL;
                }

                const int superIndex = m_telecineSuperBufferIndex++ & 1;
                if (!m_telecineSuperFrames[superIndex]) {
                    auto superInfo = prm->frameOut;
                    superInfo.width = std::max(1, superInfo.width >> 1);
                    superInfo.height = std::max(1, superInfo.height >> 2);
                    sts = ensureFrame(m_telecineSuperFrames[superIndex], superInfo, _T("telecine-super"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                }
                auto super24 = &m_telecineSuperFrames[superIndex]->frame;

                const auto savedTelecine24Frame = m_nextTelecine24Frame;
                const auto savedTelecine24Pts = m_nextTelecine24Pts;
                RGYCudaEvent deintEvent;
                sts = renderTelecine24(deint24, outputTiming.frame24Index, drain, stream, {}, &deintEvent);
                m_nextTelecine24Frame = savedTelecine24Frame;
                m_nextTelecine24Pts = savedTelecine24Pts;
                if (sts == RGY_ERR_MORE_DATA) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                std::vector<RGYCudaEvent> superWaitEvents;
                if (deintEvent() != nullptr) {
                    superWaitEvents.push_back(deintEvent);
                }
                RGYCudaEvent superEvent;
                sts = renderTelecineSuper24(super24, outputTiming.frame24Index, drain, stream, superWaitEvents, &superEvent);
                if (sts == RGY_ERR_MORE_DATA) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                std::vector<RGYCudaEvent> removeWaitEvents = superWaitEvents;
                if (superEvent() != nullptr) {
                    removeWaitEvents.push_back(superEvent);
                }
                RGYFrameInfo *superPrev24 = super24;
                RGYFrameInfo *superNext24 = super24;
                std::vector<RGYCudaEvent> maskWaitEvents = removeWaitEvents;
                auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                    sts = ensureFrame(m_telecineSuperNeighborFrames[index], *super24, _T("telecine-super neighbor"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    *frame = &m_telecineSuperNeighborFrames[index]->frame;
                    return RGY_ERR_NONE;
                };
                if (outputTiming.frame24Index > 0) {
                    RGYCudaEvent prevSuperEvent;
                    sts = ensureNeighborSuper(0, &superPrev24);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = renderTelecineSuper24(superPrev24, outputTiming.frame24Index - 1, true, stream, superWaitEvents, &prevSuperEvent);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    if (prevSuperEvent() != nullptr) {
                        maskWaitEvents.push_back(prevSuperEvent);
                    }
                }
                const int analyzed24Frames = (int)m_analyzerOutputResults.size() * 4;
                if (outputTiming.frame24Index + 1 < analyzed24Frames) {
                    RGYCudaEvent nextSuperEvent;
                    sts = ensureNeighborSuper(1, &superNext24);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = renderTelecineSuper24(superNext24, outputTiming.frame24Index + 1, drain, stream, superWaitEvents, &nextSuperEvent);
                    if (sts == RGY_ERR_MORE_DATA) {
                        m_workBufferIndex = savedWorkBufferIndex;
                        m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                        break;
                    } else if (sts != RGY_ERR_NONE) {
                        return sts;
                    } else if (nextSuperEvent() != nullptr) {
                        maskWaitEvents.push_back(nextSuperEvent);
                    }
                } else if (!drain) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                RGYFrameInfo *switchFlag = nullptr;
                RGYFrameInfo *containsCombe = nullptr;
                RGYFrameInfo *combeMask = nullptr;
                sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, super24, _T("24p"));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                RGYCudaEvent maskEvent;
                uint32_t containsCombeCount = 0;
                KfmContainsCombeReadback containsCombeReadback;
                const bool patchCombe24Enabled = kfmDeint60BranchEnabled() && outputTiming.frame24Index >= 0 && m_deint60Rtgmc && m_analyzer;
                const bool needsContainsCombeCount = switchSingleFrameDurationEnabled() || patchCombe24Enabled;
                sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev24, super24, superNext24,
                    "switch-flag-min", "contains-combe", "combe-mask-min", stream, maskWaitEvents, &maskEvent,
                    needsContainsCombeCount ? &containsCombeReadback : nullptr);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (maskEvent() != nullptr) {
                    removeWaitEvents.push_back(maskEvent);
                }
                auto resolveContainsCombeDuration = [&]() -> RGY_ERR {
                    auto readSts = resolveContainsCombeCount(containsCombeReadback, needsContainsCombeCount ? &containsCombeCount : nullptr);
                    if (readSts != RGY_ERR_NONE) {
                        return readSts;
                    }
                    writeContainsCombeDump("24p", outputTiming, containsCombeCount, containsCombeCount > 0, switchResult);
                    if (containsCombeCount > 0) {
                        markSwitchSingleFrameN60Range(outputTiming.start60, outputTiming.duration60);
                        outputTiming.duration60 = 1;
                        outputTiming.duration120 = 2;
                        outputTiming.numSourceFrames = 1;
                    }
                    return RGY_ERR_NONE;
                };
                if (auto debugOut = kfmDebugStageFrame(prm->kfm.debugStage, switchFlag, containsCombe, combeMask)) {
                    copyFramePropWithoutRes(debugOut, deint24);
                    debugOut->picstruct = RGY_PICSTRUCT_FRAME;
                    debugOut->flags = RGY_FRAME_FLAG_NONE;
                    out = debugOut;
                    outputEvent = maskEvent;
                    sts = resolveContainsCombeDuration();
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                } else {
                    sts = removeCombe24(out, deint24, super24, outputTiming.frame24Index, stream, removeWaitEvents, &outputEvent);
                    if (sts != RGY_ERR_NONE) {
                        resolveContainsCombeCount(containsCombeReadback, nullptr);
                        return sts;
                    }
                    sts = resolveContainsCombeDuration();
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    int patchN60 = -1;
                    if (patchCombe24Enabled) {
                        try {
                            static const int patchFieldIndex[4] = { 1, 3, 6, 8 };
                            const int frame24Cycle = outputTiming.frame24Index / 4;
                            const int frame24InCycle = outputTiming.frame24Index & 3;
                            const auto& patchResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                            const auto frameInfo = m_analyzer->patterns().getFrame24(patchResult.pattern, outputTiming.frame24Index);
                            patchN60 = clamp(patchFieldIndex[frame24InCycle], frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1) + frameInfo.cycleIndex * 10;
                        } catch (...) {
                            patchN60 = -1;
                        }
                    }
                    if (patchN60 >= 0 && containsCombeCount > 0) {
                        std::vector<RGYCudaEvent> patchWaitEvents = removeWaitEvents;
                        if (outputEvent() != nullptr) {
                            patchWaitEvents.push_back(outputEvent);
                        }
                        sts = ensureDeint60Range(patchN60, patchN60 + 1);
                        if (sts == RGY_ERR_MORE_DATA) {
                            m_workBufferIndex = savedWorkBufferIndex;
                            m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                            break;
                        }
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                        if (deint60 && deint60->ptr[0]) {
                            const int patchIndex = m_patchCombeBufferIndex++ & 3;
                            sts = ensureFrame(m_patchCombeFrames[patchIndex], prm->frameOut, _T("patch-combe"));
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                            RGYCudaEvent patchEvent;
                            sts = patchCombe(&m_patchCombeFrames[patchIndex]->frame, out, deint60, combeMask,
                                outputTiming.frame24Index, "patch-combe", stream, patchWaitEvents, &patchEvent);
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                            out = &m_patchCombeFrames[patchIndex]->frame;
                            outputEvent = patchEvent;
                        }
                    }
                }
                if (prm->kfm.ucf && m_analyzer && !m_analyzerOutputResults.empty() && outputTiming.frame24Index >= 0) {
                    try {
                        const int frame24Cycle = outputTiming.frame24Index / 4;
                        const auto& ucfResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                        const auto frameInfo = m_analyzer->patterns().getFrame24(ucfResult.pattern, outputTiming.frame24Index);
                        std::vector<RGYCudaEvent> ucfWaitEvents;
                        if (outputEvent() != nullptr) {
                            ucfWaitEvents.push_back(outputEvent);
                        }
                        const int lastUcfN60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + frameInfo.numFields - 2;
                        sts = resolveUcfNoiseResults((lastUcfN60 >> 1) + 1, stream);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto ucf24Plan = planUcfDecomb24Frame(frameInfo);
                        if (ucf24Plan.type == KFM_UCF24_SELECT_FRAME && ucf24Plan.n60 >= 0) {
                            sts = ensureUcfRtgmcRange(ucf24Plan.lane, ucf24Plan.n60, ucf24Plan.n60 + 1, stream);
                            if (sts == RGY_ERR_MORE_DATA) {
                                m_workBufferIndex = savedWorkBufferIndex;
                                m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                                break;
                            }
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                        }
                        const auto ucf24 = selectUcfDecomb24Frame(frameInfo, out, &ucfWaitEvents);
                        if (ucf24.type == KFM_UCF24_SELECT_FRAME && ucf24.frame && ucf24.frame != out) {
                            auto ucfOut = nextWorkFrame();
                            if (!ucfOut) {
                                return RGY_ERR_INVALID_CALL;
                            }
                            RGYCudaEvent ucfEvent;
                            sts = copyFrameWithEvent(ucfOut, ucf24.frame, ucfWaitEvents, &ucfEvent, _T("UCF24 frame"));
                            if (sts != RGY_ERR_NONE) {
                                AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR UCF24 frame: %s.\n"), get_err_mes(sts));
                                return sts;
                            }
                            copyFramePropWithoutRes(ucfOut, out);
                            out = ucfOut;
                            outputEvent = ucfEvent;
                        } else if (ucf24.type == KFM_UCF24_SELECT_DWEAVE && ucf24.n60 >= 0) {
                            auto dweave = nextWorkFrame();
                            if (!dweave) {
                                return RGY_ERR_INVALID_CALL;
                            }
                            RGYCudaEvent dweaveEvent;
                            sts = renderDoubleWeaveFrame(dweave, ucf24.n60, 2, drain, stream, ucfWaitEvents, &dweaveEvent);
                            if (sts == RGY_ERR_MORE_DATA) {
                                sts = RGY_ERR_NONE;
                            } else if (sts != RGY_ERR_NONE) {
                                return sts;
                            } else {
                                const int ucfSuperIndex = m_telecineSuperBufferIndex++ & 1;
                                auto superInfo = prm->frameOut;
                                superInfo.width = std::max(1, superInfo.width >> 1);
                                superInfo.height = std::max(1, superInfo.height >> 2);
                                sts = ensureFrame(m_telecineSuperFrames[ucfSuperIndex], superInfo, _T("UCF24 dweave-super"));
                                if (sts != RGY_ERR_NONE) {
                                    return sts;
                                }
                                auto dweaveSuper = &m_telecineSuperFrames[ucfSuperIndex]->frame;
                                std::vector<RGYCudaEvent> dweaveSuperWaitEvents = ucfWaitEvents;
                                if (dweaveEvent() != nullptr) {
                                    dweaveSuperWaitEvents.push_back(dweaveEvent);
                                }
                                RGYCudaEvent dweaveSuperEvent;
                                sts = renderCleanSuperFields(dweaveSuper, ucf24.n60, ucf24.n60, ucf24.n60 >> 1, ucf24.n60, drain, stream, dweaveSuperWaitEvents, &dweaveSuperEvent);
                                if (sts == RGY_ERR_MORE_DATA) {
                                    sts = RGY_ERR_NONE;
                                } else if (sts != RGY_ERR_NONE) {
                                    return sts;
                                } else {
                                    writeFrameInfoDump("ucf24-dweave-super", dweaveSuper);
                                    sts = dumpStageFrame("ucf24-dweave-super", dweaveSuper, ucf24.n60, stream,
                                        (dweaveSuperEvent() != nullptr) ? std::vector<RGYCudaEvent>{ dweaveSuperEvent } : std::vector<RGYCudaEvent>());
                                    if (sts != RGY_ERR_NONE) {
                                        return sts;
                                    }
                                    auto ucfOut = nextWorkFrame();
                                    if (!ucfOut) {
                                        return RGY_ERR_INVALID_CALL;
                                    }
                                    std::vector<RGYCudaEvent> dweaveRemoveWaitEvents = ucfWaitEvents;
                                    if (dweaveEvent() != nullptr) {
                                        dweaveRemoveWaitEvents.push_back(dweaveEvent);
                                    }
                                    if (dweaveSuperEvent() != nullptr) {
                                        dweaveRemoveWaitEvents.push_back(dweaveSuperEvent);
                                    }
                                    RGYCudaEvent ucfEvent;
                                    sts = removeCombeFields(ucfOut, dweave, dweaveSuper, ucf24.n60, 2, ucf24.n60, "ucf24-dweave-remove-combe", stream, dweaveRemoveWaitEvents, &ucfEvent);
                                    if (sts != RGY_ERR_NONE) {
                                        return sts;
                                    }
                                    copyFramePropWithoutRes(ucfOut, out);
                                    out = ucfOut;
                                    outputEvent = ucfEvent;
                                }
                            }
                        }
                    } catch (...) {
                    }
                }
            } else if (outputTiming.baseType == KFM_FRAME_60) {
                std::vector<RGYCudaEvent> copyWaitEvents;
                KfmUcf60Selection ucf60Plan;
                if (prm->kfm.ucf) {
                    sts = resolveUcfNoiseResults((outputTiming.start60 >> 1) + 3, stream);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    ucf60Plan = planUcfDecomb60Frame(outputTiming.start60);
                    if (ucf60Plan.lane != KFM_UCF_LANE_NONE && ucf60Plan.n60 >= 0) {
                        sts = ensureUcfRtgmcRange(ucf60Plan.lane, ucf60Plan.n60, ucf60Plan.n60 + 1, stream);
                        if (sts == RGY_ERR_MORE_DATA) {
                            break;
                        }
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                    }
                }
                sts = ensureDeint60Range(outputTiming.start60, outputTiming.start60 + outputTiming.duration60);
                if (sts == RGY_ERR_MORE_DATA) {
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                const auto *deint60 = findDeint60Frame(outputTiming.start60, &copyWaitEvents);
                if (!deint60 || !deint60->ptr[0]) {
                    break;
                }
                if (prm->kfm.ucf) {
                    deint60 = selectUcfDecomb60Frame(outputTiming.start60, deint60, &copyWaitEvents);
                }
                out = nextWorkFrame();
                if (!out) {
                    return RGY_ERR_INVALID_CALL;
                }
                sts = copyFrameWithEvent(out, deint60, copyWaitEvents, &outputEvent, _T("deint60 output"));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                copyFramePropWithoutRes(out, deint60);
            } else if (outputTiming.baseType == KFM_FRAME_30) {
                if (!source || !source->frame || !source->frame->frame.ptr[0]) {
                    break;
                }
                std::vector<RGYCudaEvent> deintWaitEvents;
                if (source->event() != nullptr) {
                    deintWaitEvents.push_back(source->event);
                }
                auto deint30 = nextWorkFrame();
                out = nextWorkFrame();
                if (!deint30 || !out) {
                    return RGY_ERR_INVALID_CALL;
                }

                RGYCudaEvent deintEvent;
                sts = copyFrameWithEvent(deint30, &source->frame->frame, deintWaitEvents, &deintEvent, _T("deint30 source"));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                copyFramePropWithoutRes(deint30, &source->frame->frame);
                deint30->picstruct = RGY_PICSTRUCT_FRAME;
                deint30->flags = RGY_FRAME_FLAG_NONE;
                attachSwitchFrameData(deint30, outputTiming, switchResult);
                writeFrameInfoDump("deint30", deint30, switchResult);

                const int superIndex = m_telecineSuperBufferIndex++ & 1;
                if (!m_telecineSuperFrames[superIndex]) {
                    auto superInfo = prm->frameOut;
                    superInfo.width = std::max(1, superInfo.width >> 1);
                    superInfo.height = std::max(1, superInfo.height >> 2);
                    sts = ensureFrame(m_telecineSuperFrames[superIndex], superInfo, _T("super30"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                }
                RGYCudaEvent superEvent;
                const auto superSts = renderSuper30(&m_telecineSuperFrames[superIndex]->frame, outputTiming.sourceIndex, drain, stream, deintWaitEvents, &superEvent);
                if (superSts != RGY_ERR_NONE && superSts != RGY_ERR_MORE_DATA) {
                    return superSts;
                }

                std::vector<RGYCudaEvent> copyWaitEvents;
                if (deintEvent() != nullptr) {
                    copyWaitEvents.push_back(deintEvent);
                }
                const bool patchCombe30Enabled = kfmDeint60BranchEnabled() && m_deint60Rtgmc;
                bool patched30 = false;
                if (superSts == RGY_ERR_NONE) {
                    std::vector<RGYCudaEvent> maskWaitEvents;
                    if (superEvent() != nullptr) {
                        maskWaitEvents.push_back(superEvent);
                    }
                    RGYFrameInfo *superPrev30 = &m_telecineSuperFrames[superIndex]->frame;
                    RGYFrameInfo *superNext30 = &m_telecineSuperFrames[superIndex]->frame;
                    auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                        sts = ensureFrame(m_telecineSuperNeighborFrames[index], m_telecineSuperFrames[superIndex]->frame, _T("super30 neighbor"));
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        *frame = &m_telecineSuperNeighborFrames[index]->frame;
                        return RGY_ERR_NONE;
                    };
                    if (outputTiming.sourceIndex > 0) {
                        RGYFrameInfo *candidatePrev30 = nullptr;
                        RGYCudaEvent prevSuperEvent;
                        sts = ensureNeighborSuper(0, &candidatePrev30);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        sts = renderSuper30(candidatePrev30, outputTiming.sourceIndex - 1, true, stream, deintWaitEvents, &prevSuperEvent);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        superPrev30 = candidatePrev30;
                        if (prevSuperEvent() != nullptr) {
                            maskWaitEvents.push_back(prevSuperEvent);
                        }
                    }
                    {
                        RGYFrameInfo *candidateNext30 = nullptr;
                        RGYCudaEvent nextSuperEvent;
                        sts = ensureNeighborSuper(1, &candidateNext30);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        sts = renderSuper30(candidateNext30, outputTiming.sourceIndex + 1, drain, stream, deintWaitEvents, &nextSuperEvent);
                        if (sts != RGY_ERR_NONE && sts != RGY_ERR_MORE_DATA) {
                            return sts;
                        }
                        if (sts == RGY_ERR_NONE) {
                            superNext30 = candidateNext30;
                            if (nextSuperEvent() != nullptr) {
                                maskWaitEvents.push_back(nextSuperEvent);
                            }
                        }
                    }
                    RGYFrameInfo *switchFlag = nullptr;
                    RGYFrameInfo *containsCombe = nullptr;
                    RGYFrameInfo *combeMask = nullptr;
                    sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, &m_telecineSuperFrames[superIndex]->frame, _T("30p"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    RGYCudaEvent maskEvent;
                    uint32_t containsCombeCount = 0;
                    KfmContainsCombeReadback containsCombeReadback;
                    const bool needsContainsCombeCount = switchSingleFrameDurationEnabled() || patchCombe30Enabled;
                    sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev30, &m_telecineSuperFrames[superIndex]->frame, superNext30,
                        "switch-flag30-min", "contains-combe30", "combe-mask30-min", stream, maskWaitEvents, &maskEvent,
                        needsContainsCombeCount ? &containsCombeReadback : nullptr);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = resolveContainsCombeCount(containsCombeReadback, needsContainsCombeCount ? &containsCombeCount : nullptr);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    writeContainsCombeDump("30p", outputTiming, containsCombeCount, containsCombeCount > 0, switchResult);
                    if (containsCombeCount > 0) {
                        markSwitchSingleFrameN60Range(outputTiming.start60, outputTiming.duration60);
                        outputTiming.duration60 = 1;
                        outputTiming.duration120 = 2;
                        outputTiming.numSourceFrames = 1;
                    }
                    if (maskEvent() != nullptr) {
                        copyWaitEvents.push_back(maskEvent);
                    }
                    if (patchCombe30Enabled && containsCombeCount > 0) {
                        std::vector<RGYCudaEvent> patchWaitEvents = copyWaitEvents;
                        const int patchN60 = outputTiming.sourceIndex * 2;
                        sts = ensureDeint60Range(patchN60, patchN60 + 1);
                        if (sts == RGY_ERR_MORE_DATA) {
                            break;
                        }
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                        if (!deint60 || !deint60->ptr[0]) {
                            break;
                        }
                        sts = patchCombe(out, deint30, deint60, combeMask, outputTiming.sourceIndex, "patch-combe30", stream, patchWaitEvents, &outputEvent);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        copyFramePropWithoutRes(out, deint30);
                        patched30 = true;
                    }
                }
                if (!patched30) {
                    const RGYFrameInfo *ucf30 = deint30;
                    if (prm->kfm.ucf) {
                        sts = resolveUcfNoiseResults(outputTiming.sourceIndex, stream);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto ucf30Plan = planUcfDecomb30Frame(outputTiming.sourceIndex);
                        if (ucf30Plan.type == KFM_UCF24_SELECT_FRAME && ucf30Plan.n60 >= 0) {
                            sts = ensureUcfRtgmcRange(ucf30Plan.lane, ucf30Plan.n60, ucf30Plan.n60 + 1, stream);
                            if (sts == RGY_ERR_MORE_DATA) {
                                break;
                            }
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                        }
                        ucf30 = selectUcfDecomb30Frame(outputTiming.sourceIndex, deint30, &copyWaitEvents);
                    }
                    sts = copyFrameWithEvent(out, ucf30, copyWaitEvents, &outputEvent, _T("deint30 output"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    copyFramePropWithoutRes(out, ucf30);
                }
            } else {
                if (!source || !source->frame || !source->frame->frame.ptr[0]) {
                    break;
                }
                std::vector<RGYCudaEvent> copyWaitEvents;
                if (source->event() != nullptr) {
                    copyWaitEvents.push_back(source->event);
                }
                out = nextWorkFrame();
                if (!out) {
                    return RGY_ERR_INVALID_CALL;
                }
                sts = copyFrameWithEvent(out, &source->frame->frame, copyWaitEvents, &outputEvent, _T("fallback output"));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                copyFramePropWithoutRes(out, &source->frame->frame);
            }
            if (outputTiming.duration60 == 1) {
                outputStart120 = rawStart120(outputTiming);
                outputTiming.start120 = static_cast<int>(outputStart120);
                outputTiming.duration120 = 2;
            }
            const auto outputEnd120 = static_cast<int64_t>(outputTiming.start120) + outputTiming.duration120;
            const auto outputStartPts = sourcePtsFrom120(outputTiming.start120);
            const auto outputEndPts = sourcePtsFrom120(outputEnd120);
            out->timestamp = std::max(outputStartPts, m_nextSwitchPts);
            out->duration = std::max<int64_t>(1, outputEndPts - out->timestamp);
            out->picstruct = RGY_PICSTRUCT_FRAME;
            out->flags = RGY_FRAME_FLAG_NONE;
            attachSwitchFrameData(out, outputTiming, switchResult);
            sts = queueVfrOutputFrame(out, stream, outputEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_nextSwitchPts = out->timestamp + out->duration;
            m_hasLastSwitchTiming = true;
            m_lastSwitchStart60 = outputTiming.start60;
            m_lastSwitchDuration60 = outputTiming.duration60;
            m_lastSwitchStart120 = outputStart120;
            m_lastSwitchIsFrame24 = outputTiming.isFrame24;
            m_nextSwitchN60 += outputTiming.duration60;
            sts = emitReadyPending(drain ? 0 : vfrOutputDelay);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (*pOutputFrameNum == 0 && !m_pendingVfrOutputs.empty()) {
            sts = emitReadyPending(drain ? 0 : (int)m_pendingVfrOutputs.size() - 1);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (drain && *pOutputFrameNum < maxOutputFrames && !m_pendingVfrOutputs.empty()) {
            sts = emitReadyPending(0);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (drain && m_pendingVfrOutputs.empty() && (timings.empty() || m_nextSwitchN60 >= m_cachedSourceFrames * 2)) {
            writeSwitchTimingDump();
            if (*pOutputFrameNum == 0) {
                sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, stream, nullptr);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        return RGY_ERR_NONE;
    }

    if (prm->kfm.mode == VppKfmMode::P24) {
        auto sts = RGY_ERR_NONE;
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, stream);
        } else {
            sts = cacheSourceFrame(pInputFrame, stream, {});
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr)
            ? drainDeint60Branch(stream)
            : runDeint60Branch(pInputFrame, stream, {});
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            if (m_deint60Rtgmc) {
                for (auto &captured : m_deint60Rtgmc->getCapturedIntermediates()) {
                    if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                    if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
                }
                m_deint60Rtgmc->clearCapturedIntermediates();
            }
            sts = runUcfRtgmcBranches(pInputFrame, stream, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(stream);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }

        *pOutputFrameNum = 0;
        const bool drain = pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr;
        const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
        while (*pOutputFrameNum < maxOutputFrames && m_nextTelecine24Frame < telecine24FrameCount(drain)) {
            auto deint24 = nextWorkFrame();
            auto out = nextWorkFrame();
            if (!deint24 || !out) {
                return RGY_ERR_INVALID_CALL;
            }

            const int superIndex = m_telecineSuperBufferIndex++ & 1;
            if (!m_telecineSuperFrames[superIndex]) {
                auto superInfo = prm->frameOut;
                superInfo.width = std::max(1, superInfo.width >> 1);
                superInfo.height = std::max(1, superInfo.height >> 2);
                auto frame = std::make_unique<CUFrameBuf>(superInfo);
                frame->releasePtr();
                const auto allocSts = frame->alloc();
                if (allocSts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super frame: %s.\n"), get_err_mes(allocSts));
                    return allocSts;
                }
                m_telecineSuperFrames[superIndex] = std::move(frame);
            }
            auto super24 = &m_telecineSuperFrames[superIndex]->frame;

            RGYCudaEvent deintEvent;
            sts = renderTelecine24(deint24, m_nextTelecine24Frame, drain, stream, {}, &deintEvent);
            if (sts == RGY_ERR_MORE_DATA) {
                break;
            }
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            std::vector<RGYCudaEvent> superWaitEvents;
            if (deintEvent() != nullptr) {
                superWaitEvents.push_back(deintEvent);
            }
            RGYCudaEvent superEvent;
            sts = renderTelecineSuper24(super24, m_nextTelecine24Frame, drain, stream, superWaitEvents, &superEvent);
            if (sts == RGY_ERR_MORE_DATA) {
                break;
            }
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            std::vector<RGYCudaEvent> removeWaitEvents = superWaitEvents;
            if (superEvent() != nullptr) {
                removeWaitEvents.push_back(superEvent);
            }
            RGYFrameInfo *superPrev24 = super24;
            RGYFrameInfo *superNext24 = super24;
            std::vector<RGYCudaEvent> maskWaitEvents = removeWaitEvents;
            auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                if (!m_telecineSuperNeighborFrames[index]
                    || m_telecineSuperNeighborFrames[index]->frame.width != super24->width
                    || m_telecineSuperNeighborFrames[index]->frame.height != super24->height
                    || m_telecineSuperNeighborFrames[index]->frame.csp != super24->csp) {
                    auto superInfo = *super24;
                    auto neighbor = std::make_unique<CUFrameBuf>(superInfo);
                    neighbor->releasePtr();
                    const auto allocSts = neighbor->alloc();
                    if (allocSts != RGY_ERR_NONE) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super neighbor frame: %s.\n"), get_err_mes(allocSts));
                        return allocSts;
                    }
                    m_telecineSuperNeighborFrames[index] = std::move(neighbor);
                }
                *frame = &m_telecineSuperNeighborFrames[index]->frame;
                return RGY_ERR_NONE;
            };
            if (m_nextTelecine24Frame > 0) {
                RGYCudaEvent prevSuperEvent;
                sts = ensureNeighborSuper(0, &superPrev24);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                sts = renderTelecineSuper24(superPrev24, m_nextTelecine24Frame - 1, true, stream, superWaitEvents, &prevSuperEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (prevSuperEvent() != nullptr) {
                    maskWaitEvents.push_back(prevSuperEvent);
                }
            }
            if (m_nextTelecine24Frame + 1 < telecine24FrameCount(drain)) {
                RGYCudaEvent nextSuperEvent;
                sts = ensureNeighborSuper(1, &superNext24);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                sts = renderTelecineSuper24(superNext24, m_nextTelecine24Frame + 1, drain, stream, superWaitEvents, &nextSuperEvent);
                if (sts == RGY_ERR_MORE_DATA) {
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (nextSuperEvent() != nullptr) {
                    maskWaitEvents.push_back(nextSuperEvent);
                }
            } else if (!drain) {
                break;
            }
            RGYFrameInfo *switchFlag = nullptr;
            RGYFrameInfo *containsCombe = nullptr;
            RGYFrameInfo *combeMask = nullptr;
            sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, super24, _T("24p"));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            RGYCudaEvent maskEvent;
            sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev24, super24, superNext24,
                "switch-flag-min", "contains-combe", "combe-mask-min", stream, maskWaitEvents, &maskEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (maskEvent() != nullptr) {
                removeWaitEvents.push_back(maskEvent);
            }
            RGYCudaEvent outputEvent;
            if (auto debugOut = kfmDebugStageFrame(prm->kfm.debugStage, switchFlag, containsCombe, combeMask)) {
                copyFramePropWithoutRes(debugOut, deint24);
                debugOut->picstruct = RGY_PICSTRUCT_FRAME;
                debugOut->flags = RGY_FRAME_FLAG_NONE;
                out = debugOut;
                outputEvent = maskEvent;
            } else {
                sts = removeCombe24(out, deint24, super24, m_nextTelecine24Frame, stream, removeWaitEvents, &outputEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                int patchN60 = -1;
                if (kfmDeint60BranchEnabled() && m_deint60Rtgmc && m_analyzer) {
                    try {
                        static const int patchFieldIndex[4] = { 1, 3, 6, 8 };
                        const int frame24Cycle = m_nextTelecine24Frame / 4;
                        const int frame24InCycle = m_nextTelecine24Frame & 3;
                        const auto& patchResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                        const auto frameInfo = m_analyzer->patterns().getFrame24(patchResult.pattern, m_nextTelecine24Frame);
                        patchN60 = clamp(patchFieldIndex[frame24InCycle], frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1) + frameInfo.cycleIndex * 10;
                    } catch (...) {
                        patchN60 = -1;
                    }
                }
                if (patchN60 >= 0) {
                    std::vector<RGYCudaEvent> patchWaitEvents = removeWaitEvents;
                    if (outputEvent() != nullptr) {
                        patchWaitEvents.push_back(outputEvent);
                    }
                    const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                    if (!deint60 || !deint60->ptr[0]) {
                        break;
                    }
                    const int patchIndex = m_patchCombeBufferIndex++ & 3;
                    if (!m_patchCombeFrames[patchIndex]
                        || m_patchCombeFrames[patchIndex]->frame.width != prm->frameOut.width
                        || m_patchCombeFrames[patchIndex]->frame.height != prm->frameOut.height
                        || m_patchCombeFrames[patchIndex]->frame.csp != prm->frameOut.csp) {
                        auto frame = std::make_unique<CUFrameBuf>(prm->frameOut);
                        frame->releasePtr();
                        const auto allocSts = frame->alloc();
                        if (allocSts != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM patch-combe frame: %s.\n"), get_err_mes(allocSts));
                            return allocSts;
                        }
                        m_patchCombeFrames[patchIndex] = std::move(frame);
                    }
                    RGYCudaEvent patchEvent;
                    sts = patchCombe(&m_patchCombeFrames[patchIndex]->frame, out, deint60, combeMask,
                        m_nextTelecine24Frame, "patch-combe", stream, patchWaitEvents, &patchEvent);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    out = &m_patchCombeFrames[patchIndex]->frame;
                    outputEvent = patchEvent;
                }
            }
            if (prm->kfm.ucf && m_analyzer && !m_analyzerOutputResults.empty()) {
                try {
                    const int frame24Cycle = m_nextTelecine24Frame / 4;
                    const auto& ucfResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                    const auto frameInfo = m_analyzer->patterns().getFrame24(ucfResult.pattern, m_nextTelecine24Frame);
                    std::vector<RGYCudaEvent> ucfWaitEvents;
                    if (outputEvent() != nullptr) {
                        ucfWaitEvents.push_back(outputEvent);
                    }
                    const int lastUcfN60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + frameInfo.numFields - 2;
                    sts = resolveUcfNoiseResults((lastUcfN60 >> 1) + 1, stream);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    const auto ucf24 = selectUcfDecomb24Frame(frameInfo, out, &ucfWaitEvents);
                    if (ucf24.type == KFM_UCF24_SELECT_FRAME && ucf24.frame && ucf24.frame != out) {
                        auto ucfOut = nextWorkFrame();
                        if (!ucfOut) {
                            return RGY_ERR_INVALID_CALL;
                        }
                        RGYCudaEvent ucfEvent;
                        auto copySts = kfmWaitEvents(stream, ucfWaitEvents);
                        if (copySts != RGY_ERR_NONE) {
                            return copySts;
                        }
                        copySts = copyFrameAsync(ucfOut, ucf24.frame, stream);
                        if (copySts != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM UCF24 frame: %s.\n"), get_err_mes(copySts));
                            return copySts;
                        }
                        sts = kfmRecordEvent(stream, &ucfEvent);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        copyFramePropWithoutRes(ucfOut, out);
                        out = ucfOut;
                        outputEvent = ucfEvent;
                    } else if (ucf24.type == KFM_UCF24_SELECT_DWEAVE && ucf24.n60 >= 0) {
                        auto dweave = nextWorkFrame();
                        if (!dweave) {
                            return RGY_ERR_INVALID_CALL;
                        }
                        RGYCudaEvent dweaveEvent;
                        sts = renderDoubleWeaveFrame(dweave, ucf24.n60, 2, drain, stream, ucfWaitEvents, &dweaveEvent);
                        if (sts == RGY_ERR_MORE_DATA) {
                            sts = RGY_ERR_NONE;
                        } else if (sts != RGY_ERR_NONE) {
                            return sts;
                        } else {
                            const int ucfSuperIndex = m_telecineSuperBufferIndex++ & 1;
                            if (!m_telecineSuperFrames[ucfSuperIndex]) {
                                auto superInfo = prm->frameOut;
                                superInfo.width = std::max(1, superInfo.width >> 1);
                                superInfo.height = std::max(1, superInfo.height >> 2);
                                auto frame = std::make_unique<CUFrameBuf>(superInfo);
                                frame->releasePtr();
                                const auto allocSts = frame->alloc();
                                if (allocSts != RGY_ERR_NONE) {
                                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF24 dweave-super frame: %s.\n"), get_err_mes(allocSts));
                                    return allocSts;
                                }
                                m_telecineSuperFrames[ucfSuperIndex] = std::move(frame);
                            }
                            auto dweaveSuper = &m_telecineSuperFrames[ucfSuperIndex]->frame;
                            std::vector<RGYCudaEvent> dweaveSuperWaitEvents = ucfWaitEvents;
                            if (dweaveEvent() != nullptr) {
                                dweaveSuperWaitEvents.push_back(dweaveEvent);
                            }
                            RGYCudaEvent dweaveSuperEvent;
                            sts = renderCleanSuperFields(dweaveSuper, ucf24.n60, ucf24.n60, ucf24.n60 >> 1, ucf24.n60, drain, stream, dweaveSuperWaitEvents, &dweaveSuperEvent);
                            if (sts == RGY_ERR_MORE_DATA) {
                                sts = RGY_ERR_NONE;
                            } else if (sts != RGY_ERR_NONE) {
                                return sts;
                            } else {
                                writeFrameInfoDump("ucf24-dweave-super", dweaveSuper);
                                sts = dumpStageFrame("ucf24-dweave-super", dweaveSuper, ucf24.n60, stream,
                                    (dweaveSuperEvent() != nullptr) ? std::vector<RGYCudaEvent>{ dweaveSuperEvent } : std::vector<RGYCudaEvent>());
                                if (sts != RGY_ERR_NONE) {
                                    return sts;
                                }
                                auto ucfOut = nextWorkFrame();
                                if (!ucfOut) {
                                    return RGY_ERR_INVALID_CALL;
                                }
                                std::vector<RGYCudaEvent> dweaveRemoveWaitEvents = ucfWaitEvents;
                                if (dweaveEvent() != nullptr) {
                                    dweaveRemoveWaitEvents.push_back(dweaveEvent);
                                }
                                if (dweaveSuperEvent() != nullptr) {
                                    dweaveRemoveWaitEvents.push_back(dweaveSuperEvent);
                                }
                                RGYCudaEvent ucfEvent;
                                sts = removeCombeFields(ucfOut, dweave, dweaveSuper, ucf24.n60, 2, ucf24.n60, "ucf24-dweave-remove-combe", stream, dweaveRemoveWaitEvents, &ucfEvent);
                                if (sts != RGY_ERR_NONE) {
                                    return sts;
                                }
                                copyFramePropWithoutRes(ucfOut, out);
                                out = ucfOut;
                                outputEvent = ucfEvent;
                            }
                        }
                    }
                } catch (...) {
                }
            }
            sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, stream, outputEvent, nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_nextTelecine24Pts += std::max<int64_t>(1, out->duration);
            m_nextTelecine24Frame++;
        }
        if (drain && m_nextTelecine24Frame >= telecine24FrameCount(true)) {
            writeTelecine24DurationDump();
            if (*pOutputFrameNum == 0) {
                sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, stream, nullptr);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        return RGY_ERR_NONE;
    }

    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        auto sts = analyzeAvailableSource(true, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        *pOutputFrameNum = 0;
        sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, stream, nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }

    auto sts = cacheSourceFrame(pInputFrame, stream, {});
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    *pOutputFrameNum = 1;
    sts = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy input frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    RGYCudaEvent outputEvent;
    sts = kfmRecordEvent(stream, &outputEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = pInputFrame->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0]->flags = RGY_FRAME_FLAG_NONE;
    ppOutputFrames[0]->dataList = pInputFrame->dataList;

    RGYFrameInfo *out = ppOutputFrames[0];
    *pOutputFrameNum = 0;
    sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, stream, outputEvent, nullptr);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterKfm::close() {
    flushUcfNoiseResultDump();
    AddMessage(RGY_LOG_INFO, _T("KFM RTGMC feed count: deint60=%lld, before60=%lld, after60=%lld.\n"),
        (long long)m_deint60Lane.feedCount(), (long long)m_before60Lane.feedCount(), (long long)m_after60Lane.feedCount());
    const auto& deint60Resets = m_deint60Lane.resetCounts();
    AddMessage(RGY_LOG_INFO, _T("KFM RTGMC deint60 resets: cold=%lld rewind=%lld feedPast=%lld farJump=%lld trimmed=%lld.\n"),
        (long long)deint60Resets[0], (long long)deint60Resets[1], (long long)deint60Resets[2], (long long)deint60Resets[3], (long long)deint60Resets[4]);
    m_rtgmc.reset();
    m_deint60Rtgmc.reset();
    m_before60Rtgmc.reset();
    m_after60Rtgmc.reset();
    m_nrFilter.reset();
    m_analyzer.reset();
    m_sourceCache.clear();
    clearKfmSourceSlotPool(true);
    m_deint60Lane.init(this, nullptr, "deint60", _T("deint60 cache"), true);
    m_before60Lane.init(this, nullptr, "before60", _T("before60"), false);
    m_after60Lane.init(this, nullptr, "after60", _T("after60"), false);
    m_ucfNoiseCache.clear();
    if (m_kfmFramePool) {
        m_kfmFramePool->clear();
    }
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingVfrOutputs.clear();
    m_staticFlag.reset();
    for (auto& frame : m_staticWorkFrames) {
        frame.reset();
    }
    for (auto& flag : m_analyzeFlags) {
        flag.reset();
    }
    if (auto clearFMCountSts = clearPendingFMCounts(); clearFMCountSts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM pending FMCount buffers: %s.\n"), get_err_mes(clearFMCountSts));
    }
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
    m_deint60Lane.reset();
    m_before60Lane.reset();
    m_after60Lane.reset();
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
