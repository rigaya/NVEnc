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

#include "NVEncFilterRtgmcSearchPrefilter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

#include "NVEncFilter.h"
#include "rgy_frame_info.h"
#include "rgy_util.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#define NVENC_RTGMC_SEARCH_PREFILTER_DECLARE_ONLY
#include "NVEncFilterRtgmcSearchPrefilter.cuh"
#undef NVENC_RTGMC_SEARCH_PREFILTER_DECLARE_ONLY

class NVEncFilterResizePlaneProxy : public NVEncFilterResize {
public:
    using NVEncFilterResize::NVEncFilterResize;
};

static bool rtgmcSearchPrefilterDumpYuvStage(const std::string &stage) {
    return stage == "finalyuv"
        || stage == "search_correction_delta" || stage == "positive_correction_gate" || stage == "negative_correction_gate"
        || stage == "corrected_search_base" || stage == "field_stable_search";
}

static bool rtgmcSearchPrefilterUseSearchRefine2Chain(const NVEncFilterParamRtgmcSearchPrefilter &prm) {
    if (prm.searchRefine < 2) {
        return false;
    }
    return true;
}

static bool rtgmcSearchPrefilterMergeSearchRefineEnabled() {
    const char *env = std::getenv("NVENC_RTGMC_KERNEL_MERGE_SEARCH_REFINE");
    return env == nullptr || env[0] != '0';
}

static bool rtgmcSearchPrefilterMergeSearchRefine2TileEnabled() {
    const char *env = std::getenv("NVENC_RTGMC_KERNEL_MERGE_SEARCH_REFINE2_TILE");
    return env == nullptr || env[0] != '0';
}

static const NVEncRtgmcSearchPrefilterLaunchFuncs *rtgmcSearchPrefilterLaunchFuncs(const RGYFrameInfo &frame, const int tr0) {
    if (RGY_CSP_BIT_DEPTH[frame.csp] > 8) {
        return (tr0 <= 0) ? getNVEncRtgmcSearchPrefilterU16TR0() : getNVEncRtgmcSearchPrefilterU16TRP();
    }
    return (tr0 <= 0) ? getNVEncRtgmcSearchPrefilterU8TR0() : getNVEncRtgmcSearchPrefilterU8TRP();
}

static std::array<float, 5> rtgmcSearchPrefilterGaussWeights(const float gaussP) {
    std::array<float, 5> weights = {};
    float sum = 0.0f;
    for (int i = 0; i < (int)weights.size(); i++) {
        weights[i] = (float)std::exp2(-(gaussP * 0.1f) * (float)(i * i));
        sum += (i == 0) ? weights[i] : weights[i] * 2.0f;
    }
    if (sum > 0.0f) {
        for (auto &weight : weights) {
            weight /= sum;
        }
    }
    return weights;
}

tstring NVEncFilterParamRtgmcSearchPrefilter::print() const {
    return strsprintf(_T("rtgmc-search-prefilter: tr0=%d, search_refine=%d, rep0-thin=%d, rep0-pad=%d, tv_range=%s, chroma_motion=%s%s%s"),
        tr0, searchRefine, rep0Thin, rep0Pad, tvRange ? _T("on") : _T("off"), chromaMotion ? _T("on") : _T("off"),
        dumpStage.empty() ? _T("") : strsprintf(_T(", dump_stage=%s"), dumpStage.c_str()).c_str(),
        attachSearchLuma ? _T(", attach-search-luma") : _T(""));
}

RGYFrameDataRtgmcSearchLuma::RGYFrameDataRtgmcSearchLuma(std::shared_ptr<CUFrameBuf> frame, int bitdepth) :
    m_frame(frame),
    m_bitdepth(bitdepth) {
    m_dataType = RGY_FRAME_DATA_RTGMC_SEARCH_LUMA;
}

const RGYFrameInfo *RGYFrameDataRtgmcSearchLuma::frame() const {
    return m_frame ? &m_frame->frame : nullptr;
}

static void eraseRtgmcSearchLumaFrameData(std::vector<std::shared_ptr<RGYFrameData>>& dataList) {
    // The attached search-luma data owns a pooled frame. Drop stale attachments
    // before reusing frame metadata or the search-luma pool cannot recycle.
    dataList.erase(std::remove_if(dataList.begin(), dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data && data->dataType() == RGY_FRAME_DATA_RTGMC_SEARCH_LUMA;
    }), dataList.end());
}

static RGY_CSP rtgmcSearchLumaCsp(const RGYFrameInfo &frameInfo) {
    return (RGY_CSP_BIT_DEPTH[frameInfo.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
}

static RGYFrameInfo rtgmcSearchPrefilterPlaneFrameInfo(const RGYFrameInfo &planeInfo) {
    return RGYFrameInfo(planeInfo.width, planeInfo.height, rtgmcSearchLumaCsp(planeInfo), planeInfo.bitdepth, planeInfo.picstruct, planeInfo.mem_type);
}

static RGYFrameInfo rtgmcSearchPrefilterSearchFrameInfo(const RGYFrameInfo &frameInfo, const bool includeChroma) {
    return includeChroma ? frameInfo : rtgmcSearchPrefilterPlaneFrameInfo(frameInfo);
}

NVEncFilterRtgmcSearchPrefilter::NVEncFilterRtgmcSearchPrefilter() :
    NVEncFilter(),
    m_cacheFrames(),
    m_sceneChangeBufferPool(),
    m_pendingSearchPrefilterFrames(),
    m_cacheFramePool(std::make_shared<SharedFramePool>()),
    m_searchLumaPool(std::make_shared<SharedFramePool>()),
    m_buildOptions(),
    m_searchLumaDump(),
    m_searchLumaDumpPath(),
    m_searchLumaDumpStage("final"),
    m_searchLumaDumpMaxFrames(0),
    m_searchLumaDumpFrameCount(0),
    m_searchLumaDumpEnabled(false),
    m_searchLumaDumpHeaderWritten(false),
    m_inputCount(0),
    m_drainCount(0) {
    m_name = _T("rtgmc-search-prefilter");
}

std::shared_ptr<CUFrameBuf> NVEncFilterRtgmcSearchPrefilter::SharedFramePool::get(const RGYFrameInfo &frameInfo) {
    if (frameInfo.width <= 0 || frameInfo.height <= 0) {
        return nullptr;
    }
    auto pooled = std::find_if(frames.begin(), frames.end(), [&frameInfo](const Entry &candidate) {
        return candidate.frame && !cmpFrameInfoCspResolution(&candidate.frame->frame, &frameInfo);
    });
    // 返却直後の画像へのホスト待ちでGPU投入が途切れないよう、FIFOに空き4枚を残す。
    constexpr size_t reuseLag = 4;
    if (frames.size() <= reuseLag) {
        pooled = frames.end();
    }
    std::unique_ptr<CUFrameBuf> frame;
    if (pooled != frames.end()) {
        if (pooled->readyEvent) {
            cudaEventSynchronize(*pooled->readyEvent);
            pooled->readyEvent.reset();
        }
        frame = std::move(pooled->frame);
        frames.erase(pooled);
    } else {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        RGYCudaAllocStatsTag allocStatsTagScope("RTGMC search prefilter pool");
        if (frame->alloc() != RGY_ERR_NONE) {
            frame.reset();
        }
    }
    if (!frame) {
        return nullptr;
    }
    return std::shared_ptr<CUFrameBuf>(frame.release(), [pool = shared_from_this()](CUFrameBuf *recycleFrame) {
        pool->recycle(recycleFrame);
    });
}

void NVEncFilterRtgmcSearchPrefilter::SharedFramePool::recycle(CUFrameBuf *frame) {
    if (frame) {
        frame->frame.dataList.clear();
        Entry entry;
        entry.frame.reset(frame);
        entry.readyEvent = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        if (cudaEventCreateWithFlags(entry.readyEvent.get(), cudaEventDisableTiming) == cudaSuccess) {
            cudaEventRecord(*entry.readyEvent, 0);
        } else {
            entry.readyEvent.reset();
        }
        frames.emplace_back(std::move(entry));
    }
}

void NVEncFilterRtgmcSearchPrefilter::SharedFramePool::clear() {
    for (auto &entry : frames) {
        if (entry.readyEvent) {
            cudaEventSynchronize(*entry.readyEvent);
            entry.readyEvent.reset();
        }
        if (entry.frame) {
            entry.frame->frame.dataList.clear();
        }
    }
    frames.clear();
}

NVEncFilterRtgmcSearchPrefilter::~NVEncFilterRtgmcSearchPrefilter() {
    close();
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcSearchPrefilter> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->tr0 < -1 || prm->tr0 > 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter tr0 must be -1, 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->searchRefine < 0 || prm->searchRefine > 3) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter search_refine must be 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!rgy_rtgmc_repair_thin_level_is_valid(prm->rep0Thin)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter rep0-thin must be 0-7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!rgy_rtgmc_repair_pad_level_is_valid(prm->rep0Pad)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter rep0-pad must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcSearchPrefilter> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int limitedYMin = (bitdepth >= 16) ? (16 << 8) : (16 << std::max(bitdepth - 8, 0));
    const int limitedYRange = (bitdepth >= 16) ? (219 << 8) : (219 << std::max(bitdepth - 8, 0));
    const int limitedCOffset = (bitdepth >= 16) ? (128 << 8) : (128 << std::max(bitdepth - 8, 0));
    const int limitedCRange = (bitdepth >= 16) ? (112 << 8) : (112 << std::max(bitdepth - 8, 0));
    const auto gaussWeights = rtgmcSearchPrefilterGaussWeights(2.0f);
    m_buildOptions = strsprintf(
        "-D TypePixel=%s"
        " -D RTGMC_SEARCH_PREFILTER_PIXEL_MAX=%d"
        " -D RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN=%d"
        " -D RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE=%d"
        " -D RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET=%d"
        " -D RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE=%d"
        " -D rtgmc_search_prefilter_block_x=%d"
        " -D rtgmc_search_prefilter_block_y=%d"
        " -D RTGMC_SEARCH_REFINE2_GAUSS_W0=%.9ff"
        " -D RTGMC_SEARCH_REFINE2_GAUSS_W1=%.9ff"
        " -D RTGMC_SEARCH_REFINE2_GAUSS_W2=%.9ff"
        " -D RTGMC_SEARCH_REFINE2_GAUSS_W3=%.9ff"
        " -D RTGMC_SEARCH_REFINE2_GAUSS_W4=%.9ff",
        bitdepth > 8 ? "ushort" : "uchar",
        pixelMax,
        limitedYMin,
        limitedYRange,
        limitedCOffset,
        limitedCRange,
        RTGMC_SEARCH_PREFILTER_BLOCK_X,
        RTGMC_SEARCH_PREFILTER_BLOCK_Y,
        gaussWeights[0],
        gaussWeights[1],
        gaussWeights[2],
        gaussWeights[3],
        gaussWeights[4]);
    AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernels for rtgmc-search-prefilter: %s\n"),
        char_to_tstring(m_buildOptions).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::recordRepairUse(const int planeIndex, const RGY_ERR status, cudaStream_t stream) {
    auto &resources = m_repairPlaneResources[planeIndex];
    const auto recorded = err_to_rgy(cudaEventRecord(*resources.lastEvent, stream));
    if (recorded != RGY_ERR_NONE) {
        // 記録できない場合も、投入済み処理が終わるまで資源を解放・再利用しない。
        cudaStreamSynchronize(stream);
    }
    return status != RGY_ERR_NONE ? status : recorded;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::setupRepairResources(const NVEncFilterParamRtgmcSearchPrefilter &prm, const bool processChroma) {
    for (auto &resources : m_repairPlaneResources) resources.clear();
    const auto setupPlane = [&](int planeIndex, RGY_PLANE plane) -> RGY_ERR {
        auto info = rtgmcSearchPrefilterPlaneFrameInfo(getPlane(&prm.frameIn, plane));
        auto &resources = m_repairPlaneResources[planeIndex];
        resources.temporal = createPlaneFrame(info);
        resources.delta = createPlaneFrame(info);
        if (!resources.temporal || !resources.delta) return RGY_ERR_MEMORY_ALLOC;
        if (resources.temporal->frame.pitch[0] != resources.delta->frame.pitch[0]) return RGY_ERR_INVALID_PARAM;
        // 半解像度の4タップは寸法が4未満だと画面外を評価する。
        // 通常寸法はハロー0、極小プレーンだけ必要な仮想座標までFCを持つ。
        resources.fcHx = std::max(4 - info.width, 0);
        resources.fcHy = std::max(4 - info.height, 0);
        auto correctedInfo = info;
        correctedInfo.width += 2 * resources.fcHx;
        correctedInfo.height += 2 * resources.fcHy;
        resources.corrected = createPlaneFrame(correctedInfo);
        if (!resources.corrected) return RGY_ERR_MEMORY_ALLOC;
        // 後段の参照半径を逆順に加算し、画面外でも評価座標をクランプしない。
        const bool level4 = (prm.repairProfile.restoreFlags & RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH) != 0;
        const bool repairEnabled = (prm.repairProfile.restoreFlags & RGY_RTGMC_REPAIR_RESTORE_ENABLED) != 0;
        resources.stageCount = repairEnabled ? (level4 ? 4 : ((prm.rep0Pad == 2) ? 6 : 5)) : 0;
        int hx = resources.fcHx, hy = resources.fcHy;
        for (int stage = resources.stageCount - 1; stage >= 0; stage--) {
            auto &target = resources.stages[stage];
            target.hx = hx;
            target.hy = hy;
            auto padded = info;
            padded.width += 2 * hx;
            padded.height += 2 * hy;
            for (auto &frame : target.polarity) {
                frame = createPlaneFrame(padded);
                if (!frame) return RGY_ERR_MEMORY_ALLOC;
            }
            if (target.polarity[0]->frame.pitch[0] != target.polarity[1]->frame.pitch[0]) return RGY_ERR_INVALID_PARAM;
            if (level4) {
                // 縦envelope→mid→mean→core。midはmeanと同座標のcoreを直接読む。
                // sampleYの端での置換はカーネルで行い、ここでは最大半径を確保する。
                if (stage == 3) hy += 2;
                if (stage == 1) { hx++; hy++; }
            } else {
                // pad=2はR2→R1→Bで半径を2回加算する。pad=3はBの3x3極値。
                if (stage == 5 || (stage == 4 && prm.rep0Pad != 0)) { hx++; hy++; }
                if (stage == 3) hy += 2 + ((prm.repairProfile.restoreFlags & RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE) ? 1 : 0);
                if (stage == 2 && (prm.repairProfile.thinRejectFlags & RGY_RTGMC_REPAIR_THIN_RANK_LIMIT)) { hx++; hy++; }
                if (stage == 1 && (prm.repairProfile.thinRejectFlags & RGY_RTGMC_REPAIR_THIN_CORE_BLEND)) { hx++; hy++; }
            }
            // coreはDの縦近傍を読むが、Dだけは元画素と同様にクランプしてよい。
        }
        resources.lastEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
        const auto eventStatus = err_to_rgy(cudaEventCreateWithFlags(resources.lastEvent.get(), cudaEventDisableTiming));
        if (eventStatus != RGY_ERR_NONE) {
            resources.lastEvent.reset();
            return eventStatus;
        }
        return RGY_ERR_NONE;
    };
    auto sts = setupPlane(0, RGY_PLANE_Y);
    if (sts == RGY_ERR_NONE && processChroma) sts = setupPlane(1, RGY_PLANE_U);
    return sts;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::emitRepairFC(const int planeIndex,
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const NVEncFilterParamRtgmcSearchPrefilter &prm, const uint32_t *sceneFlags, cudaStream_t stream) {
    auto &resources = m_repairPlaneResources[planeIndex];
    if (!resources.temporal || !resources.delta || !resources.lastEvent || !resources.corrected) return RGY_ERR_NULL_PTR;
    // 同じ領域をU/Vや次フレームが再利用する前に、最後のconsumerを待つ。
    auto sts = err_to_rgy(cudaStreamWaitEvent(stream, *resources.lastEvent, 0));
    if (sts != RGY_ERR_NONE) return sts;
    const auto tc = getPlane(&resources.temporal->frame, RGY_PLANE_Y);
    const auto delta = getPlane(&resources.delta->frame, RGY_PLANE_Y);
    const auto profile = rgy_rtgmc_repair_profile_pack(prm.repairProfile);
    sts = launchRtgmcSearchPrefilterRepairTemporal(prev2, prev, cur, next, next2,
        tc, delta, prm.tr0, sceneFlags, stream);
    sts = recordRepairUse(planeIndex, sts, stream);
    if (sts != RGY_ERR_NONE) return sts;
    for (int stage = 0; stage < resources.stageCount; stage++) {
        const auto &target = resources.stages[stage];
        const auto positive = getPlane(&target.polarity[1]->frame, RGY_PLANE_Y);
        const auto negative = getPlane(&target.polarity[0]->frame, RGY_PLANE_Y);
        auto sourcePos = delta;
        auto sourceNeg = delta;
        int sourceHx = 0, sourceHy = 0;
        if (stage > 0) {
            const auto &source = resources.stages[stage - 1];
            sourcePos = getPlane(&source.polarity[1]->frame, RGY_PLANE_Y);
            sourceNeg = getPlane(&source.polarity[0]->frame, RGY_PLANE_Y);
            sourceHx = source.hx;
            sourceHy = source.hy;
        }
        const auto &core = resources.stages[0];
        const auto corePos = getPlane(&core.polarity[1]->frame, RGY_PLANE_Y);
        const auto coreNeg = getPlane(&core.polarity[0]->frame, RGY_PLANE_Y);
        sts = launchRtgmcSearchPrefilterRepairStage(sourcePos, sourceNeg, sourceHx, sourceHy,
            positive, negative, target.hx, target.hy, cur.width, cur.height, profile, stage,
            corePos, coreNeg, core.hx, core.hy, stream);
        sts = recordRepairUse(planeIndex, sts, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    // 無修復時もTC/D/FCを生成し、gateには有効なTCを渡す。
    auto gatePos = tc;
    auto gateNeg = tc;
    int gateHx = 0, gateHy = 0;
    if (resources.stageCount > 0) {
        const auto &gate = resources.stages[resources.stageCount - 1];
        gatePos = getPlane(&gate.polarity[1]->frame, RGY_PLANE_Y);
        gateNeg = getPlane(&gate.polarity[0]->frame, RGY_PLANE_Y);
        gateHx = gate.hx;
        gateHy = gate.hy;
    }
    const auto dst = getPlane(&resources.corrected->frame, RGY_PLANE_Y);
    sts = launchRtgmcSearchPrefilterRepairApply(tc, delta, gatePos, gateNeg, gateHx, gateHy,
        dst, resources.fcHx, resources.fcHy, cur.width, cur.height, profile, stream);
    return recordRepairUse(planeIndex, sts, stream);
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::emitFieldStableSearch(const int planeIndex,
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const NVEncFilterParamRtgmcSearchPrefilter &prm, const uint32_t *sceneFlags, cudaStream_t stream) {
    auto sts = emitRepairFC(planeIndex, prev2, prev, cur, next, next2, prm, sceneFlags, stream);
    if (sts != RGY_ERR_NONE) return sts;
    auto &resources = m_repairPlaneResources[planeIndex];
    const auto fc = getPlane(&resources.corrected->frame, RGY_PLANE_Y);
    sts = launchRtgmcSearchPrefilterRepairCopy(fc, resources.fcHx, resources.fcHy, dst, 0, -1, stream);
    return recordRepairUse(planeIndex, sts, stream);
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::emitLumaSearch(const int planeIndex,
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const NVEncFilterParamRtgmcSearchPrefilter &prm, const int fullRangeMode, const uint32_t *sceneFlags, cudaStream_t stream) {
    auto sts = emitRepairFC(planeIndex, prev2, prev, cur, next, next2, prm, sceneFlags, stream);
    if (sts != RGY_ERR_NONE) return sts;
    auto &resources = m_repairPlaneResources[planeIndex];
    const auto fc = getPlane(&resources.corrected->frame, RGY_PLANE_Y);
    sts = launchRtgmcSearchPrefilterRepairLuma(fc, resources.fcHx, resources.fcHy, cur, dst, prm.searchRefine, fullRangeMode, stream);
    return recordRepairUse(planeIndex, sts, stream);
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::allocCacheFrames(const RGYFrameInfo &frameInfo) {
    bool reuse = true;
    for (const auto &frame : m_cacheFrames) {
        if (!frame || cmpFrameInfoCspResolution(&frame->frame, &frameInfo)) {
            reuse = false;
            break;
        }
        for (int i = 0; i < RGY_CSP_PLANES[frame->frame.csp]; i++) {
            if (frame->frame.ptr[i] == nullptr) {
                reuse = false;
                break;
            }
        }
        if (!reuse) {
            break;
        }
    }
    if (reuse) {
        return RGY_ERR_NONE;
    }

    for (auto &frame : m_cacheFrames) {
        frame.reset();
    }
    for (auto &frame : m_cacheFrames) {
        frame = m_cacheFramePool ? m_cacheFramePool->get(frameInfo) : nullptr;
        if (!frame) {
            for (auto &clearFrame : m_cacheFrames) {
                clearFrame.reset();
            }
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::setupSearchRefine1Resources(const RGYFrameInfo &frameInfo, const bool processChroma) {
    for (auto &resources : m_searchRefine1PlaneResources) {
        resources.clear();
    }
    for (auto &resources : m_searchRefine2PlaneResources) {
        resources.clear();
    }
    const auto setupPlane = [&](const int planeIndex, const RGYFrameInfo &planeBaseInfo) -> RGY_ERR {
        if (planeBaseInfo.width <= 0 || planeBaseInfo.height <= 0) {
            return RGY_ERR_INVALID_PARAM;
        }

        auto fullInfo = rtgmcSearchPrefilterPlaneFrameInfo(planeBaseInfo);
        auto halfInfo = fullInfo;
        halfInfo.width = std::max(fullInfo.width >> 1, 1);
        halfInfo.height = std::max(fullInfo.height >> 1, 1);

        auto motionGuide = createPlaneFrame(fullInfo);
        auto halfSearchBase = createPlaneFrame(halfInfo);
        auto halfSearchSmoothed = createPlaneFrame(halfInfo);
        if (!motionGuide || !halfSearchBase || !halfSearchSmoothed) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_searchRefine1PlaneResources[planeIndex].motionGuide = std::move(motionGuide);
        m_searchRefine1PlaneResources[planeIndex].halfSearchBase = std::move(halfSearchBase);
        m_searchRefine1PlaneResources[planeIndex].halfSearchSmoothed = std::move(halfSearchSmoothed);

        return RGY_ERR_NONE;
    };

    auto sts = setupPlane(0, getPlane(&frameInfo, RGY_PLANE_Y));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (processChroma && RGY_CSP_PLANES[frameInfo.csp] > 1) {
        sts = setupPlane(1, getPlane(&frameInfo, RGY_PLANE_U));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::setupSearchRefine2Resources(const RGYFrameInfo &frameInfo, const bool processChroma) {
    for (auto &resources : m_searchRefine2PlaneResources) {
        resources.clear();
    }
    for (auto &resize : m_searchRefine2ResizeEdgeSoftenedSearch) {
        resize.reset();
    }

    const auto setupPlane = [&](const int planeIndex, const RGYFrameInfo &planeBaseInfo) -> RGY_ERR {
        if (planeBaseInfo.width <= 0 || planeBaseInfo.height <= 0) {
            return RGY_ERR_INVALID_PARAM;
        }
        auto planeInfo = rtgmcSearchPrefilterPlaneFrameInfo(planeBaseInfo);
        auto motionGuide = createPlaneFrame(planeInfo);
        auto searchSmoothed3x3 = createPlaneFrame(planeInfo);
        auto edgeSoftenedSearch = createPlaneFrame(planeInfo);
        auto preStabilizedSearch = createPlaneFrame(planeInfo);
        if (!motionGuide || !searchSmoothed3x3 || !edgeSoftenedSearch || !preStabilizedSearch) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_searchRefine2PlaneResources[planeIndex].motionGuide = std::move(motionGuide);
        m_searchRefine2PlaneResources[planeIndex].searchSmoothed3x3 = std::move(searchSmoothed3x3);
        m_searchRefine2PlaneResources[planeIndex].edgeSoftenedSearch = std::move(edgeSoftenedSearch);
        m_searchRefine2PlaneResources[planeIndex].preStabilizedSearch = std::move(preStabilizedSearch);
        return RGY_ERR_NONE;
    };

    auto sts = setupPlane(0, getPlane(&frameInfo, RGY_PLANE_Y));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (processChroma && RGY_CSP_PLANES[frameInfo.csp] > 1) {
        sts = setupPlane(1, getPlane(&frameInfo, RGY_PLANE_U));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcSearchPrefilter>(pParam);
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    prm->repairProfile = rgy_rtgmc_repair_profile_from_levels(prm->rep0Thin, prm->rep0Pad);

    close();

    sts = buildKernel(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = allocCacheFrames(prm->frameIn);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-search-prefilter cache: %s.\n"), get_err_mes(sts));
        return sts;
    }
    const bool processChroma = prm->chromaMotion && RGY_CSP_PLANES[prm->frameIn.csp] > 1;
    sts = setupRepairResources(*prm, processChroma);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("修復の中間プレーンを確保できません: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if (RTGMC_SEARCH_PREFILTER_USE_SEARCH_REFINE1_CHAIN && prm->searchRefine == 1) {
        sts = setupSearchRefine1Resources(prm->frameIn, processChroma);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to prepare rtgmc-search-prefilter search_refine1 resources: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (rtgmcSearchPrefilterUseSearchRefine2Chain(*prm)) {
        sts = setupSearchRefine2Resources(prm->frameIn, processChroma);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to prepare rtgmc-search-prefilter search_refine2 resources: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    sts = initSearchLumaDump(prm->frameIn, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = AllocFrameBuf(prm->frameOut, RTGMC_SEARCH_PREFILTER_CACHE_SIZE);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_inputCount = 0;
    m_drainCount = 0;
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_DATA);

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::initSearchLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcSearchPrefilter &prm) {
    m_searchLumaDumpEnabled = false;
    m_searchLumaDumpHeaderWritten = false;
    m_searchLumaDumpFrameCount = 0;
    m_searchLumaDumpMaxFrames = 0;
    m_searchLumaDumpPath.clear();
    m_searchLumaDumpStage = "final";
    if (m_searchLumaDump.is_open()) {
        m_searchLumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("NVENC_RTGMC_SEARCH_LUMA_DUMP_Y4M");
    std::string dumpPath = tchar_to_string(prm.dumpY4m, CP_UTF8);
    if (dumpPath.empty() && dumpPathEnv != nullptr && dumpPathEnv[0] != '\0') {
        dumpPath = dumpPathEnv;
    }
    if (dumpPath.empty()) {
        return RGY_ERR_NONE;
    }

    std::string dumpStage = tchar_to_string(prm.dumpStage, CP_UTF8);
    const char *dumpStageEnv = std::getenv("NVENC_RTGMC_SEARCH_LUMA_DUMP_STAGE");
    if (dumpStage.empty() && dumpStageEnv != nullptr && dumpStageEnv[0] != '\0') {
        dumpStage = dumpStageEnv;
    }
    if (!dumpStage.empty()) {
        m_searchLumaDumpStage = dumpStage;
        std::transform(m_searchLumaDumpStage.begin(), m_searchLumaDumpStage.end(), m_searchLumaDumpStage.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("NVENC_RTGMC_SEARCH_LUMA_DUMP_Y4M supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    const char *maxFrames = std::getenv("NVENC_RTGMC_SEARCH_LUMA_DUMP_MAX_FRAMES");
    if (prm.dumpMaxFrames > 0) {
        m_searchLumaDumpMaxFrames = prm.dumpMaxFrames;
    } else if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_searchLumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }

    m_searchLumaDumpPath = dumpPath;
    m_searchLumaDump.open(m_searchLumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_searchLumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc-search-prefilter search luma dump: %s.\n"),
            char_to_tstring(m_searchLumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_searchLumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc-search-prefilter search luma dump enabled: %s (stage=%s).\n"),
        char_to_tstring(m_searchLumaDumpPath).c_str(), char_to_tstring(m_searchLumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::dumpSearchLumaFrame(CUFrameBuf *searchLuma, const RGYFrameInfo &sourceFrame,
    cudaStream_t stream) {
    if (!m_searchLumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_searchLumaDumpMaxFrames > 0 && m_searchLumaDumpFrameCount >= m_searchLumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    if (searchLuma == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    const auto planeY = getPlane(&searchLuma->frame, RGY_PLANE_Y);
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostFrame(planeY.width, planeY.height, RGY_CSP_Y8, 8, sourceFrame.picstruct, RGY_MEM_TYPE_CPU);
    hostFrame.ptr[0] = hostY.data();
    hostFrame.pitch[0] = planeY.width;
    auto err = copyPlaneAsync(&hostFrame, &planeY, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (!m_searchLumaDumpHeaderWritten) {
        m_searchLumaDump << "YUV4MPEG2 W" << hostFrame.width << " H" << hostFrame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_searchLumaDumpHeaderWritten = true;
    }
    m_searchLumaDump << "FRAME\n";
    for (int y = 0; y < hostFrame.height; y++) {
        m_searchLumaDump.write(reinterpret_cast<const char *>(hostFrame.ptr[0] + (size_t)y * hostFrame.pitch[0]), hostFrame.width);
    }
    const int chromaWidth = (hostFrame.width + 1) >> 1;
    const int chromaHeight = (hostFrame.height + 1) >> 1;
    std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
    m_searchLumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    m_searchLumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    m_searchLumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

std::unique_ptr<CUFrameBuf> NVEncFilterRtgmcSearchPrefilter::createPlaneFrame(const RGYFrameInfo &frameInfo) {
    auto frame = std::make_unique<CUFrameBuf>(frameInfo);
    RGYCudaAllocStatsTag allocStatsTagScope("RTGMC search prefilter plane");
    if (frame->alloc() != RGY_ERR_NONE) {
        frame.reset();
    }
    return frame;
}

std::shared_ptr<CUFrameBuf> NVEncFilterRtgmcSearchPrefilter::createSearchLumaFrame(const RGYFrameInfo &frameInfo, const bool includeChroma) {
    const auto searchFrameInfo = rtgmcSearchPrefilterSearchFrameInfo(frameInfo, includeChroma);
    return m_searchLumaPool ? m_searchLumaPool->get(searchFrameInfo) : nullptr;
}

int NVEncFilterRtgmcSearchPrefilter::cacheIndex(int frame) const {
    return frame % RTGMC_SEARCH_PREFILTER_CACHE_SIZE;
}

int NVEncFilterRtgmcSearchPrefilter::outputDelay() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcSearchPrefilter>(m_param);
    return prm ? std::max(prm->tr0, 0) : 0;
}

int NVEncFilterRtgmcSearchPrefilter::drainFrameCount() const {
    return std::min(outputDelay(), m_inputCount);
}

const RGYFrameInfo *NVEncFilterRtgmcSearchPrefilter::resolveCacheFrame(int frameIndex) const {
    auto frame = resolveCacheFrameShared(frameIndex);
    return frame ? &frame->frame : nullptr;
}

std::shared_ptr<CUFrameBuf> NVEncFilterRtgmcSearchPrefilter::resolveCacheFrameShared(int frameIndex) const {
    if (m_inputCount <= 0) {
        return nullptr;
    }
    const int clampedFrame = clamp(frameIndex, 0, m_inputCount - 1);
    return m_cacheFrames[cacheIndex(clampedFrame)];
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::pushCacheFrame(const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto &cacheFrame = m_cacheFrames[cacheIndex(m_inputCount)];
    if (!cacheFrame || cmpFrameInfoCspResolution(&cacheFrame->frame, pInputFrame) || cacheFrame.use_count() > 1) {
        cacheFrame = m_cacheFramePool ? m_cacheFramePool->get(*pInputFrame) : nullptr;
        if (!cacheFrame) {
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    auto pCacheFrame = &cacheFrame->frame;
    auto err = copyFrameAsync(pCacheFrame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy input to rtgmc-search-prefilter cache slot %d: %s.\n"),
            cacheIndex(m_inputCount), get_err_mes(err));
        return err;
    }
    copyFrameProp(pCacheFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::checkSameResolutionPlanePitches(const TCHAR *stageName, const std::vector<const RGYFrameInfo *> &planes) {
    if (planes.empty() || std::any_of(planes.begin(), planes.end(), [](const RGYFrameInfo *plane) { return plane == nullptr; })) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto base = planes.front();
    for (size_t i = 1; i < planes.size(); i++) {
        const auto plane = planes[i];
        if (plane->width != base->width || plane->height != base->height) {
            AddMessage(RGY_LOG_ERROR,
                _T("rtgmc-search-prefilter %s resolution mismatch: base=%dx%d, plane[%d]=%dx%d.\n"),
                stageName ? stageName : _T("plane"),
                base->width, base->height, (int)i, plane->width, plane->height);
            return RGY_ERR_INVALID_PARAM;
        }
        if (plane->pitch[0] != base->pitch[0]) {
            AddMessage(RGY_LOG_ERROR,
                _T("rtgmc-search-prefilter %s pitch mismatch: base=%d, plane[%d]=%d.\n"),
                stageName ? stageName : _T("plane"),
                base->pitch[0], (int)i, plane->pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::checkTemporalPlanePitches(const TCHAR *planeName,
    const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2) {
    return checkSameResolutionPlanePitches(
        strsprintf(_T("temporal %s"), planeName ? planeName : _T("plane")).c_str(),
        { prev2, prev, cur, next, next2 });
}

std::unique_ptr<CUMemBuf> NVEncFilterRtgmcSearchPrefilter::getSceneChangeBuffer(const size_t requiredSize,
    std::shared_ptr<cudaEvent_t> *readyEvent) {
    auto pooled = std::find_if(m_sceneChangeBufferPool.begin(), m_sceneChangeBufferPool.end(), [requiredSize](const SceneChangeBufferEntry &entry) {
        return entry.buffer && entry.buffer->nSize >= requiredSize;
    });
    if (pooled != m_sceneChangeBufferPool.end()) {
        auto buf = std::move(pooled->buffer);
        *readyEvent = pooled->readyEvent;
        m_sceneChangeBufferPool.erase(pooled);
        return buf;
    }
    readyEvent->reset();
    auto buf = std::make_unique<CUMemBuf>(requiredSize);
    if (buf->alloc() != RGY_ERR_NONE) buf.reset();
    return buf;
}

void NVEncFilterRtgmcSearchPrefilter::recycleSceneChangeBuffer(std::unique_ptr<CUMemBuf> &&buf,
    const std::shared_ptr<cudaEvent_t> &readyEvent) {
    if (buf) m_sceneChangeBufferPool.push_back({ std::move(buf), readyEvent });
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::submitSceneChangePlane(PendingSceneChangePlane *pending,
    const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2,
    const RGY_PLANE plane, const TCHAR *planeName, const int smoothRadius, cudaStream_t stream) {
    if (!pending) {
        return RGY_ERR_INVALID_PARAM;
    }
    *pending = PendingSceneChangePlane();
    pending->plane = plane;
    pending->planeName = planeName ? planeName : _T("plane");
    pending->smoothRadius = smoothRadius;
    if (smoothRadius <= 0) {
        return RGY_ERR_NONE;
    }
    if (!prev2 || !prev || !cur || !next || !next2 || !cur->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    auto sts = checkTemporalPlanePitches(planeName, prev2, prev, cur, next, next2);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const int groupCountX = divCeil(cur->width, RTGMC_SEARCH_PREFILTER_BLOCK_X);
    const int groupCountY = divCeil(cur->height, RTGMC_SEARCH_PREFILTER_BLOCK_Y);
    const int groupCount = std::max(groupCountX * groupCountY, 1);
    const size_t requiredSize = ((size_t)groupCount + 1) * 4 * sizeof(uint32_t);
    pending->partial = getSceneChangeBuffer(requiredSize, &pending->readyEvent);
    if (!pending->partial) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    pending->groupCount = groupCount;
    pending->sceneThreshold = (uint64_t)RTGMC_SEARCH_PREFILTER_SCENECHANGE * (uint64_t)cur->width * (uint64_t)cur->height;
    if (pending->readyEvent) {
        const auto waitErr = err_to_rgy(cudaStreamWaitEvent(stream, *pending->readyEvent, 0));
        if (waitErr != RGY_ERR_NONE) return waitErr;
    }
    auto ready = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto err = err_to_rgy(cudaEventCreateWithFlags(ready.get(), cudaEventDisableTiming));
    if (err != RGY_ERR_NONE) return err;
    err = rtgmcSearchPrefilterLaunchFuncs(*cur, smoothRadius)->scenechange(*prev2, *prev, *cur, *next, *next2,
        (uint32_t *)pending->partial->ptr, groupCount, stream);
    if (err == RGY_ERR_NONE) {
        err = launchRtgmcSearchPrefilterScenechangeReduce((uint32_t *)pending->partial->ptr, groupCount,
            pending->sceneThreshold, smoothRadius, stream);
    }
    // 途中失敗時も、投入済みの部分和カーネルまでを保護する。
    const auto recordErr = err_to_rgy(cudaEventRecord(*ready, stream));
    if (recordErr == RGY_ERR_NONE) pending->readyEvent = ready;
    if (err != RGY_ERR_NONE || recordErr != RGY_ERR_NONE) {
        if (recordErr == RGY_ERR_NONE) recycleSceneChangeBuffer(std::move(pending->partial), ready);
        return err != RGY_ERR_NONE ? err : recordErr;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::checkSceneChangePlane(PendingSceneChangePlane *pending,
    const int currentFrame, cudaStream_t stream) {
    if (!pending || !pending->partial) return RGY_ERR_NONE;
    auto err = err_to_rgy(cudaStreamWaitEvent(stream, *pending->readyEvent, 0));
    if (err != RGY_ERR_NONE) return err;
    std::vector<uint32_t> state(((size_t)pending->groupCount + 1) * 4);
    err = err_to_rgy(cudaMemcpyAsync(state.data(), pending->partial->ptr, state.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) return err;
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;
    bool matched = true;
    for (int dir = 0; dir < 4; dir++) {
        uint64_t sum = 0;
        for (int group = 0; group < pending->groupCount; group++) {
            sum += state[4 + (size_t)dir * pending->groupCount + group];
        }
        const uint32_t expected = pending->smoothRadius > 0 && (dir < 2 || pending->smoothRadius >= 2) && sum >= pending->sceneThreshold;
        matched &= state[dir] == expected;
    }
    AddMessage(RGY_LOG_DEBUG, _T("scene-check frame=%d plane=%s flags=%u,%u,%u,%u match=%d\n"),
        currentFrame, pending->planeName.c_str(), state[0], state[1], state[2], state[3], matched ? 1 : 0);
    if (!matched) {
        AddMessage(RGY_LOG_ERROR, _T("探索前処理のシーン判定がCPU参照と一致しません。\n"));
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

const NVEncFilterRtgmcSearchPrefilter::PendingSceneChangePlane *NVEncFilterRtgmcSearchPrefilter::sceneChangeForPlane(
    const PendingSearchPrefilterFrame &pending, const RGY_PLANE plane) const {
    for (const auto &entry : pending.scenePlanes) {
        if (entry.plane == plane) return &entry;
    }
    return nullptr;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::dumpSearchYuvFrame(const RGYFrameInfo &yFrame, const RGYFrameInfo *chromaFrame,
    cudaStream_t stream) {
    if (!m_searchLumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_searchLumaDumpMaxFrames > 0 && m_searchLumaDumpFrameCount >= m_searchLumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[yFrame.csp];
    if (bitdepth > 8 || RGY_CSP_PLANES[yFrame.csp] <= 0) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc-search-prefilter YUV dump disabled by unsupported Y frame csp: %s.\n"),
            RGY_CSP_NAMES[yFrame.csp]);
        m_searchLumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }
    const auto planeY = getPlane(&yFrame, RGY_PLANE_Y);
    if (planeY.ptr[0] == nullptr || planeY.width <= 0 || planeY.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter YUV dump has invalid Y plane.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostFrame(planeY.width, planeY.height, RGY_CSP_Y8, 8, yFrame.picstruct, RGY_MEM_TYPE_CPU);
    hostFrame.ptr[0] = hostY.data();
    hostFrame.pitch[0] = planeY.width;
    auto err = copyPlaneAsync(&hostFrame, &planeY, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    std::array<std::vector<uint8_t>, 3> chromaPlanes;
    const RGYFrameInfo *chromaSrc = chromaFrame;
    if (chromaSrc != nullptr) {
        const int chromaBitdepth = RGY_CSP_BIT_DEPTH[chromaSrc->csp];
        if (chromaBitdepth > 8 || RGY_CSP_CHROMA_FORMAT[chromaSrc->csp] != RGY_CHROMAFMT_YUV420 || RGY_CSP_PLANES[chromaSrc->csp] < 3) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc-search-prefilter YUV dump chroma source disabled by unsupported frame csp: %s.\n"),
                RGY_CSP_NAMES[chromaSrc->csp]);
            chromaSrc = nullptr;
        }
    }
    if (chromaSrc != nullptr) {
        for (int iplane = 1; iplane < 3; iplane++) {
            const auto plane = getPlane(chromaSrc, (RGY_PLANE)iplane);
            if (plane.ptr[0] == nullptr || plane.width <= 0 || plane.height <= 0) {
                AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter YUV dump has invalid chroma plane %d.\n"), iplane);
                return RGY_ERR_INVALID_CALL;
            }
            chromaPlanes[iplane].resize((size_t)plane.width * plane.height);
            RGYFrameInfo chromaHost(plane.width, plane.height, RGY_CSP_Y8, 8, yFrame.picstruct, RGY_MEM_TYPE_CPU);
            chromaHost.ptr[0] = chromaPlanes[iplane].data();
            chromaHost.pitch[0] = plane.width;

            auto copyErr = copyPlaneAsync(&chromaHost, &plane, stream);
            if (copyErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc-search-prefilter YUV dump chroma plane %d: %s.\n"), iplane, get_err_mes(copyErr));
                return copyErr;
            }
            copyErr = err_to_rgy(cudaStreamSynchronize(stream));
            if (copyErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc-search-prefilter YUV dump chroma plane %d: %s.\n"), iplane, get_err_mes(copyErr));
                return copyErr;
            }
        }
    }
    if (!m_searchLumaDumpHeaderWritten) {
        m_searchLumaDump << "YUV4MPEG2 W" << hostFrame.width << " H" << hostFrame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_searchLumaDumpHeaderWritten = true;
    }
    m_searchLumaDump << "FRAME\n";
    for (int y = 0; y < hostFrame.height; y++) {
        m_searchLumaDump.write(reinterpret_cast<const char *>(hostFrame.ptr[0] + (size_t)y * hostFrame.pitch[0]), hostFrame.width);
    }
    if (chromaSrc != nullptr) {
        for (int iplane = 1; iplane < 3; iplane++) {
            const auto plane = getPlane(chromaSrc, (RGY_PLANE)iplane);
            for (int y = 0; y < plane.height; y++) {
                m_searchLumaDump.write(reinterpret_cast<const char *>(chromaPlanes[iplane].data() + (size_t)y * plane.width), plane.width);
            }
        }
    } else {
        const int chromaWidth = (hostFrame.width + 1) >> 1;
        const int chromaHeight = (hostFrame.height + 1) >> 1;
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_searchLumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_searchLumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_searchLumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc-search-prefilter YUV dump: %s.\n"),
            char_to_tstring(m_searchLumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_searchLumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::emitPrefilteredFrame(PendingSearchPrefilterFrame &pending, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcSearchPrefilter>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto prev2 = pending.refs[0] ? &pending.refs[0]->frame : nullptr;
    const auto prev = pending.refs[1] ? &pending.refs[1]->frame : nullptr;
    const auto cur = pending.refs[2] ? &pending.refs[2]->frame : nullptr;
    const auto next = pending.refs[3] ? &pending.refs[3]->frame : nullptr;
    const auto next2 = pending.refs[4] ? &pending.refs[4]->frame : nullptr;
    if (!prev2 || !prev || !cur || !next || !next2 || !cur->ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter cache frames are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    std::shared_ptr<CUFrameBuf> searchLumaFrame;
    const bool useSeparateSearchLuma = prm->attachSearchLuma || m_searchLumaDumpEnabled;
    const bool attachSearchChroma = prm->attachSearchLuma && prm->chromaMotion && RGY_CSP_PLANES[cur->csp] > 1;
    auto outFrameBuf = m_frameBuf[m_nFrameIdx].get();
    auto pOut = &outFrameBuf->frame;
    m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    auto err = copyFrameAsync(pOut, cur, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-search-prefilter base frame: %s.\n"), get_err_mes(err));
        return err;
    }
    if (useSeparateSearchLuma) {
        searchLumaFrame = createSearchLumaFrame(*cur, attachSearchChroma);
        if (!searchLumaFrame) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-search-prefilter search luma frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        copyFramePropWithoutRes(&searchLumaFrame->frame, cur);
        searchLumaFrame->frame.dataList.clear();
    }

    const auto planePrev2 = getPlane(prev2, RGY_PLANE_Y);
    const auto planePrev = getPlane(prev, RGY_PLANE_Y);
    const auto planeCur = getPlane(cur, RGY_PLANE_Y);
    const auto planeNext = getPlane(next, RGY_PLANE_Y);
    const auto planeNext2 = getPlane(next2, RGY_PLANE_Y);
    auto *pSearchLuma = useSeparateSearchLuma ? &searchLumaFrame->frame : pOut;
    const auto planeDst = getPlane(pSearchLuma, RGY_PLANE_Y);
    if (planePrev2.ptr[0] == nullptr || planePrev.ptr[0] == nullptr || planeCur.ptr[0] == nullptr
        || planeNext.ptr[0] == nullptr || planeNext2.ptr[0] == nullptr || planeDst.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter requires valid luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    err = checkTemporalPlanePitches(_T("Y"), &planePrev2, &planePrev, &planeCur, &planeNext, &planeNext2);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = checkSameResolutionPlanePitches(_T("luma output"), { &planeCur, &planeDst });
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const auto *lumaScene = sceneChangeForPlane(pending, RGY_PLANE_Y);
    const auto *lumaSceneFlags = (lumaScene && lumaScene->partial) ? (const uint32_t *)lumaScene->partial->ptr : nullptr;
    const auto *launch = rtgmcSearchPrefilterLaunchFuncs(planeCur, prm->tr0);
    const bool useSearchRefine1Chain = RTGMC_SEARCH_PREFILTER_USE_SEARCH_REFINE1_CHAIN && prm->searchRefine == 1;
    const bool useSearchRefine2Chain = rtgmcSearchPrefilterUseSearchRefine2Chain(*prm);
    const bool mergeSearchRefine = rtgmcSearchPrefilterMergeSearchRefineEnabled();
    const bool mergeSearchRefine2Tile = rtgmcSearchPrefilterMergeSearchRefine2TileEnabled();
    auto emitSearchRefine1Plane = [&](const int planeIndex,
        const RGYFrameInfo &planePrev2Src, const RGYFrameInfo &planePrevSrc, const RGYFrameInfo &planeCurSrc,
        const RGYFrameInfo &planeNextSrc, const RGYFrameInfo &planeNext2Src, const RGYFrameInfo &planeDstSrc,
        const uint32_t *sceneFlags, const int fullRangeMode) -> RGY_ERR {
        auto &resources = m_searchRefine1PlaneResources[planeIndex];
        if (!resources.motionGuide || !resources.halfSearchBase || !resources.halfSearchSmoothed) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter search_refine1 resources are not ready for plane %d.\n"), planeIndex);
            return RGY_ERR_NULL_PTR;
        }

        auto planeMotionGuide = getPlane(&resources.motionGuide->frame, RGY_PLANE_Y);
        auto planeHalfSearchBase = getPlane(&resources.halfSearchBase->frame, RGY_PLANE_Y);
        auto planeHalfSearchSmoothed = getPlane(&resources.halfSearchSmoothed->frame, RGY_PLANE_Y);
        auto planeDstWork = planeDstSrc;

        auto sts = checkSameResolutionPlanePitches(_T("search_refine1 motion guide"),
            { &planePrev2Src, &planePrevSrc, &planeCurSrc, &planeNextSrc, &planeNext2Src, &planeMotionGuide });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = emitFieldStableSearch(planeIndex,
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, *prm, sceneFlags, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_field_stable_search"), planeIndex, get_err_mes(sts));
            return sts;
        }

        auto &repair = m_repairPlaneResources[planeIndex];
        const auto fc = getPlane(&repair.corrected->frame, RGY_PLANE_Y);
        sts = launchRtgmcSearchPrefilterRepairHalfSearch(false, fc, repair.fcHx, repair.fcHy, planeHalfSearchBase, stream);
        sts = recordRepairUse(planeIndex, sts, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_half_search_base_from_fc"),
                planeIndex, get_err_mes(sts));
            return sts;
        }

        sts = checkSameResolutionPlanePitches(_T("search_refine1 half smooth"),
            { &planeHalfSearchBase, &planeHalfSearchSmoothed });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = launch->searchSmoothed3x3(planeHalfSearchBase, planeHalfSearchSmoothed, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_search_smoothed3x3"), planeIndex, get_err_mes(sts));
            return sts;
        }

        sts = launch->halfResolutionUpsample(planeHalfSearchSmoothed, planeDstWork, fullRangeMode, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_half_resolution_upsample"),
                planeIndex, get_err_mes(sts));
        }
        return sts;
    };
    auto emitSearchRefine2Plane = [&](const int planeIndex,
        const RGYFrameInfo &planePrev2Src, const RGYFrameInfo &planePrevSrc, const RGYFrameInfo &planeCurSrc,
        const RGYFrameInfo &planeNextSrc, const RGYFrameInfo &planeNext2Src, const RGYFrameInfo &planeDstSrc,
        const uint32_t *sceneFlags, const int fullRangeMode) -> RGY_ERR {
        auto &resources = m_searchRefine2PlaneResources[planeIndex];
        if (!resources.motionGuide || !resources.searchSmoothed3x3 || !resources.edgeSoftenedSearch || !resources.preStabilizedSearch) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter search_refine2 resources are not ready for plane %d.\n"), planeIndex);
            return RGY_ERR_NULL_PTR;
        }

        auto planeMotionGuide = getPlane(&resources.motionGuide->frame, RGY_PLANE_Y);
        auto planeSearchSmoothed3x3 = getPlane(&resources.searchSmoothed3x3->frame, RGY_PLANE_Y);
        auto planeEdgeSoftenedSearch = getPlane(&resources.edgeSoftenedSearch->frame, RGY_PLANE_Y);
        auto planePreStabilizedSearch = getPlane(&resources.preStabilizedSearch->frame, RGY_PLANE_Y);

        auto sts = checkSameResolutionPlanePitches(_T("search_refine2 motion guide"),
            { &planePrev2Src, &planePrevSrc, &planeCurSrc, &planeNextSrc, &planeNext2Src, &planeMotionGuide });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = emitFieldStableSearch(planeIndex,
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, *prm, sceneFlags, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_field_stable_search"), planeIndex, get_err_mes(sts));
            return sts;
        }

        const bool needsSearchRefine2IntermediateDump = m_searchLumaDumpEnabled
            && (m_searchLumaDumpStage == "search_smoothed3x3"
                || m_searchLumaDumpStage == "edge_softened_search"
                || m_searchLumaDumpStage == "softened_search_blend");
        const bool useMergedSearchRefine2Tile = prm->searchRefine == 2
            && mergeSearchRefine2Tile
            && !needsSearchRefine2IntermediateDump;
        if (useMergedSearchRefine2Tile) {
            sts = checkSameResolutionPlanePitches(_T("search_refine2 tile"), { &planeMotionGuide, &planeDstSrc });
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = launch->refine2Tile(planeMotionGuide, planeDstSrc, fullRangeMode, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                    _T("kernel_rtgmc_search_prefilter_refine2_tile"), planeIndex, get_err_mes(sts));
            }
            return sts;
        }

        sts = checkSameResolutionPlanePitches(_T("search_refine2 3x3"), { &planeMotionGuide, &planeSearchSmoothed3x3 });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = launch->searchSmoothed3x3(planeMotionGuide, planeSearchSmoothed3x3, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_search_smoothed3x3"), planeIndex, get_err_mes(sts));
            return sts;
        }

        sts = launch->edgeSoftenedSearch(planeSearchSmoothed3x3, planeEdgeSoftenedSearch, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to edge-softened search rtgmc-search-prefilter search_refine2 plane %d: %s.\n"),
                planeIndex, get_err_mes(sts));
            return sts;
        }

        const bool useMergedSearchRefine = prm->searchRefine >= 3
            && mergeSearchRefine
            && !(m_searchLumaDumpEnabled && m_searchLumaDumpStage == "pre_stabilized_search");
        if (useMergedSearchRefine) {
            sts = checkSameResolutionPlanePitches(_T("search_refine3 merged"),
                { &planeEdgeSoftenedSearch, &planeMotionGuide, &planeCurSrc, &planeDstSrc });
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = launch->softenedSearchBlendStabilized(
                planeEdgeSoftenedSearch, planeMotionGuide, planeCurSrc, planeDstSrc, fullRangeMode, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                    _T("kernel_rtgmc_search_prefilter_softened_search_blend_stabilized"), planeIndex, get_err_mes(sts));
            }
            return sts;
        }

        const auto &softenedSearchBlendDst = (prm->searchRefine >= 3) ? planePreStabilizedSearch : planeDstSrc;
        sts = checkSameResolutionPlanePitches(_T("search_refine2 blend"), { &planeEdgeSoftenedSearch, &planeMotionGuide, &softenedSearchBlendDst });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = launch->softenedSearchBlend(
            planeEdgeSoftenedSearch, planeMotionGuide, softenedSearchBlendDst, (prm->searchRefine >= 3) ? 0 : fullRangeMode, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_softened_search_blend"), planeIndex, get_err_mes(sts));
            return sts;
        }
        if (prm->searchRefine < 3) {
            return RGY_ERR_NONE;
        }
        sts = checkSameResolutionPlanePitches(_T("search_refine3 stabilize"), { &planeMotionGuide, &planeCurSrc, &planePreStabilizedSearch, &planeDstSrc });
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = launch->stabilizedSearch(
            planeMotionGuide, planeCurSrc, planePreStabilizedSearch, planeDstSrc, fullRangeMode, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_stabilized_search"), planeIndex, get_err_mes(sts));
        }
        return sts;
    };

    if (useSearchRefine1Chain) {
        err = emitSearchRefine1Plane(0, planePrev2, planePrev, planeCur, planeNext, planeNext2, planeDst, lumaSceneFlags, prm->tvRange ? 1 : 0);
    } else if (useSearchRefine2Chain) {
        err = emitSearchRefine2Plane(0, planePrev2, planePrev, planeCur, planeNext, planeNext2, planeDst, lumaSceneFlags, prm->tvRange ? 1 : 0);
    } else {
        err = emitLumaSearch(0, planePrev2, planePrev, planeCur, planeNext, planeNext2,
            planeDst, *prm, prm->tvRange ? 1 : 0, lumaSceneFlags, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), _T("kernel_rtgmc_search_prefilter_luma"), get_err_mes(err));
            return err;
        }
    }

    if (err != RGY_ERR_NONE) return err;

    CUFrameBuf *dumpFrame = searchLumaFrame ? searchLumaFrame.get() : outFrameBuf;
    std::unique_ptr<CUFrameBuf> debugDumpFrame;
    if (m_searchLumaDumpEnabled && (m_searchLumaDumpStage == "half_search_base" || m_searchLumaDumpStage == "half_search_smoothed")) {
        if (prm->searchRefine != 1) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter dump stage %s requires search_refine=1.\n"),
                char_to_tstring(m_searchLumaDumpStage).c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        RGYFrameInfo debugFrameInfo(std::max(planeCur.width >> 1, 1), std::max(planeCur.height >> 1, 1),
            rtgmcSearchLumaCsp(*cur), cur->bitdepth, cur->picstruct, cur->mem_type);
        debugDumpFrame = createPlaneFrame(debugFrameInfo);
        if (!debugDumpFrame) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-search-prefilter debug dump frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        auto planeDebugDump = getPlane(&debugDumpFrame->frame, RGY_PLANE_Y);
        auto &repair = m_repairPlaneResources[0];
        const auto fc = getPlane(&repair.corrected->frame, RGY_PLANE_Y);
        err = launchRtgmcSearchPrefilterRepairHalfSearch(m_searchLumaDumpStage == "half_search_smoothed",
            fc, repair.fcHx, repair.fcHy, planeDebugDump, stream);
        err = recordRepairUse(0, err, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
                (m_searchLumaDumpStage == "half_search_base") ? _T("kernel_rtgmc_search_prefilter_half_search_base") : _T("kernel_rtgmc_search_prefilter_half_search_smoothed"),
                get_err_mes(err));
            return err;
        }
        dumpFrame = debugDumpFrame.get();
    }
    if (m_searchLumaDumpEnabled && (m_searchLumaDumpStage == "search_smoothed3x3" || m_searchLumaDumpStage == "edge_softened_search"
        || m_searchLumaDumpStage == "softened_search_blend" || m_searchLumaDumpStage == "pre_stabilized_search" || m_searchLumaDumpStage == "stabilized_search")) {
        if (m_searchLumaDumpStage == "search_smoothed3x3" || m_searchLumaDumpStage == "edge_softened_search" || m_searchLumaDumpStage == "softened_search_blend") {
            if (prm->searchRefine < 2) {
                AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter dump stage %s requires search_refine>=2.\n"),
                    char_to_tstring(m_searchLumaDumpStage).c_str());
                return RGY_ERR_INVALID_PARAM;
            }
        } else if (prm->searchRefine != 3) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter dump stage %s requires search_refine=3.\n"),
                char_to_tstring(m_searchLumaDumpStage).c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        auto &resources = m_searchRefine2PlaneResources[0];
        if (m_searchLumaDumpStage == "search_smoothed3x3") {
            dumpFrame = resources.searchSmoothed3x3.get();
        } else if (m_searchLumaDumpStage == "edge_softened_search") {
            dumpFrame = resources.edgeSoftenedSearch.get();
        } else if (m_searchLumaDumpStage == "pre_stabilized_search") {
            dumpFrame = resources.preStabilizedSearch.get();
        }
        if (dumpFrame == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter dump stage %s resource is not ready.\n"),
                char_to_tstring(m_searchLumaDumpStage).c_str());
            return RGY_ERR_NULL_PTR;
        }
    }
    if (m_searchLumaDumpEnabled && (m_searchLumaDumpStage == "temporal_candidate" || m_searchLumaDumpStage == "field_stable_search" || m_searchLumaDumpStage == "search_correction_delta"
        || m_searchLumaDumpStage == "positive_correction_gate" || m_searchLumaDumpStage == "negative_correction_gate"
        || m_searchLumaDumpStage == "corrected_search_base")) {
        const int debugStage = (m_searchLumaDumpStage == "temporal_candidate") ? 9
            : (m_searchLumaDumpStage == "field_stable_search") ? 10
            : (m_searchLumaDumpStage == "search_correction_delta") ? 11
            : (m_searchLumaDumpStage == "positive_correction_gate") ? 12
            : (m_searchLumaDumpStage == "negative_correction_gate") ? 13
            : 14;
        RGYFrameInfo debugFrameInfo(planeCur.width, planeCur.height, rtgmcSearchLumaCsp(*cur), cur->bitdepth, cur->picstruct, cur->mem_type);
        debugDumpFrame = createPlaneFrame(debugFrameInfo);
        if (!debugDumpFrame) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-search-prefilter debug dump frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        auto planeDebugDump = getPlane(&debugDumpFrame->frame, RGY_PLANE_Y);
        err = checkSameResolutionPlanePitches(_T("debug full-resolution dump"),
            { &planePrev2, &planePrev, &planeCur, &planeNext, &planeNext2, &planeDebugDump });
        if (err != RGY_ERR_NONE) {
            return err;
        }
        auto &repair = m_repairPlaneResources[0];
        auto source = getPlane(&repair.corrected->frame, RGY_PLANE_Y);
        int hx = repair.fcHx, hy = repair.fcHy, constantValue = -1;
        if (debugStage == 9 || debugStage == 11) {
            source = getPlane(debugStage == 9 ? &repair.temporal->frame : &repair.delta->frame, RGY_PLANE_Y);
            hx = hy = 0;
        } else if (debugStage == 12 || debugStage == 13) {
            if (repair.stageCount > 0) {
                const auto &gate = repair.stages[repair.stageCount - 1];
                source = getPlane(&gate.polarity[debugStage == 12 ? 1 : 0]->frame, RGY_PLANE_Y);
                hx = gate.hx;
                hy = gate.hy;
            } else {
                // 無修復時のgateは元ソースと同じ中立値を表示する。
                constantValue = RGY_CSP_BIT_DEPTH[planeCur.csp] > 8 ? 32768 : 128;
            }
        }
        err = launchRtgmcSearchPrefilterRepairCopy(source, hx, hy, planeDebugDump, 0, constantValue, stream);
        err = recordRepairUse(0, err, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at rtgmc-search-prefilter debug stage %d: %s.\n"), debugStage, get_err_mes(err));
            return err;
        }
        dumpFrame = debugDumpFrame.get();
    }

    const bool dumpYuvStage = rtgmcSearchPrefilterDumpYuvStage(m_searchLumaDumpStage);
    if (m_searchLumaDumpEnabled && !dumpYuvStage) {
        err = dumpSearchLumaFrame(dumpFrame, *cur, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    if (useSeparateSearchLuma && !prm->attachSearchLuma) {
        auto planeSearchLuma = getPlane(&searchLumaFrame->frame, RGY_PLANE_Y);
        auto planeOutY = getPlane(pOut, RGY_PLANE_Y);
        err = copyPlaneAsync(&planeOutY, &planeSearchLuma, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-search-prefilter dumped search luma to output: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    if (prm->chromaMotion && RGY_CSP_PLANES[pOut->csp] > 1) {
        for (int iplane = 1; iplane < RGY_CSP_PLANES[pOut->csp]; iplane++) {
            const auto planePrev2C = getPlane(prev2, (RGY_PLANE)iplane);
            const auto planePrevC = getPlane(prev, (RGY_PLANE)iplane);
            const auto planeCurC = getPlane(cur, (RGY_PLANE)iplane);
            const auto planeNextC = getPlane(next, (RGY_PLANE)iplane);
            const auto planeNext2C = getPlane(next2, (RGY_PLANE)iplane);
            const auto planeDstC = getPlane(attachSearchChroma ? pSearchLuma : pOut, (RGY_PLANE)iplane);
            if (planePrev2C.ptr[0] == nullptr || planePrevC.ptr[0] == nullptr || planeCurC.ptr[0] == nullptr
                || planeNextC.ptr[0] == nullptr || planeNext2C.ptr[0] == nullptr || planeDstC.ptr[0] == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter requires valid chroma planes.\n"));
                return RGY_ERR_INVALID_CALL;
            }
            const auto chromaPlaneName = strsprintf(_T("UV plane %d"), iplane);
            err = checkTemporalPlanePitches(chromaPlaneName.c_str(), &planePrev2C, &planePrevC, &planeCurC, &planeNextC, &planeNext2C);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            err = checkSameResolutionPlanePitches(strsprintf(_T("chroma output %d"), iplane).c_str(), { &planeCurC, &planeDstC });
            if (err != RGY_ERR_NONE) {
                return err;
            }
            const auto *chromaScene = sceneChangeForPlane(pending, (RGY_PLANE)iplane);
            const auto *chromaSceneFlags = (chromaScene && chromaScene->partial) ? (const uint32_t *)chromaScene->partial->ptr : nullptr;
            if (useSearchRefine1Chain) {
                err = emitSearchRefine1Plane(1,
                    planePrev2C, planePrevC, planeCurC, planeNextC, planeNext2C, planeDstC,
                    chromaSceneFlags, prm->tvRange ? 2 : 0);
            } else if (useSearchRefine2Chain) {
                err = emitSearchRefine2Plane(1,
                    planePrev2C, planePrevC, planeCurC, planeNextC, planeNext2C, planeDstC,
                    chromaSceneFlags, prm->tvRange ? 2 : 0);
            } else {
                err = emitLumaSearch(1, planePrev2C, planePrevC, planeCurC, planeNextC, planeNext2C,
                    planeDstC, *prm, prm->tvRange ? 2 : 0, chromaSceneFlags, stream);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at rtgmc-search-prefilter chroma plane %d: %s.\n"), iplane, get_err_mes(err));
                return err;
            }
        }
    }

    if (m_searchLumaDumpEnabled && dumpYuvStage) {
        err = dumpSearchYuvFrame(dumpFrame ? dumpFrame->frame : *pOut, pOut, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    copyFramePropWithoutRes(pOut, cur);
    eraseRtgmcSearchLumaFrameData(pOut->dataList);
    if (prm->attachSearchLuma) {
        pOut->dataList.push_back(std::make_shared<RGYFrameDataRtgmcSearchLuma>(searchLumaFrame, RGY_CSP_BIT_DEPTH[cur->csp]));
    }
    ppOutputFrames[0] = pOut;
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::submitPendingSearchPrefilterFrame(const int currentFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcSearchPrefilter>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    PendingSearchPrefilterFrame pending;
    pending.currentFrame = currentFrame;
    pending.refs[0] = resolveCacheFrameShared(currentFrame - 2);
    pending.refs[1] = resolveCacheFrameShared(currentFrame - 1);
    pending.refs[2] = resolveCacheFrameShared(currentFrame);
    pending.refs[3] = resolveCacheFrameShared(currentFrame + 1);
    pending.refs[4] = resolveCacheFrameShared(currentFrame + 2);
    if (!pending.refs[0] || !pending.refs[1] || !pending.refs[2] || !pending.refs[3] || !pending.refs[4] || !pending.refs[2]->frame.ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-search-prefilter cache frames are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    const auto clearLocalPending = [&]() {
        for (auto &plane : pending.scenePlanes) {
            recycleSceneChangeBuffer(std::move(plane.partial), plane.readyEvent);
        }
        pending.scenePlanes.clear();
    };

    const auto prev2 = &pending.refs[0]->frame;
    const auto prev = &pending.refs[1]->frame;
    const auto cur = &pending.refs[2]->frame;
    const auto next = &pending.refs[3]->frame;
    const auto next2 = &pending.refs[4]->frame;

    const auto planePrev2 = getPlane(prev2, RGY_PLANE_Y);
    const auto planePrev = getPlane(prev, RGY_PLANE_Y);
    const auto planeCur = getPlane(cur, RGY_PLANE_Y);
    const auto planeNext = getPlane(next, RGY_PLANE_Y);
    const auto planeNext2 = getPlane(next2, RGY_PLANE_Y);
    PendingSceneChangePlane lumaScene;
    auto err = submitSceneChangePlane(&lumaScene,
        &planePrev2, &planePrev, &planeCur, &planeNext, &planeNext2,
        RGY_PLANE_Y, _T("Y"), prm->tr0, stream);
    if (err != RGY_ERR_NONE) {
        clearLocalPending();
        AddMessage(RGY_LOG_ERROR, _T("failed to submit rtgmc-search-prefilter luma scene change: %s.\n"), get_err_mes(err));
        return err;
    }
    pending.scenePlanes.emplace_back(std::move(lumaScene));

    if (prm->chromaMotion && RGY_CSP_PLANES[cur->csp] > 1) {
        for (int iplane = 1; iplane < RGY_CSP_PLANES[cur->csp]; iplane++) {
            const auto planeName = strsprintf(_T("UV plane %d"), iplane);
            const auto planePrev2C = getPlane(prev2, (RGY_PLANE)iplane);
            const auto planePrevC = getPlane(prev, (RGY_PLANE)iplane);
            const auto planeCurC = getPlane(cur, (RGY_PLANE)iplane);
            const auto planeNextC = getPlane(next, (RGY_PLANE)iplane);
            const auto planeNext2C = getPlane(next2, (RGY_PLANE)iplane);
            PendingSceneChangePlane chromaScene;
            err = submitSceneChangePlane(&chromaScene,
                &planePrev2C, &planePrevC, &planeCurC, &planeNextC, &planeNext2C,
                (RGY_PLANE)iplane, planeName.c_str(), prm->tr0, stream);
            if (err != RGY_ERR_NONE) {
                clearLocalPending();
                AddMessage(RGY_LOG_ERROR, _T("failed to submit rtgmc-search-prefilter chroma scene change for plane %d: %s.\n"),
                    iplane, get_err_mes(err));
                return err;
            }
            pending.scenePlanes.emplace_back(std::move(chromaScene));
        }
    }

    m_pendingSearchPrefilterFrames.emplace_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::resolvePendingSearchPrefilterFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    if (m_pendingSearchPrefilterFrames.empty()) {
        return RGY_ERR_NONE;
    }

    auto &pending = m_pendingSearchPrefilterFrames.front();
    const char *checkScene = std::getenv("RGY_RTGMC_SEARCH_SCENE_CHECK");
    for (auto &plane : pending.scenePlanes) {
        if (checkScene && checkScene[0] != '0') {
            const auto err = checkSceneChangePlane(&plane, pending.currentFrame, stream);
            if (err != RGY_ERR_NONE) return err;
        }
        if (plane.readyEvent) {
            const auto err = err_to_rgy(cudaStreamWaitEvent(stream, *plane.readyEvent, 0));
            if (err != RGY_ERR_NONE) return err;
        }
    }
    auto completed = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto err = err_to_rgy(cudaEventCreateWithFlags(completed.get(), cudaEventDisableTiming));
    if (err != RGY_ERR_NONE) return err;
    err = emitPrefilteredFrame(pending, ppOutputFrames, pOutputFrameNum, stream);
    // 出力コピーを含む最後の消費まで保持し、途中失敗時も投入済み処理を保護する。
    const auto retireErr = err_to_rgy(cudaEventRecord(*completed, stream));
    for (auto &plane : pending.scenePlanes) {
        if (retireErr == RGY_ERR_NONE) recycleSceneChangeBuffer(std::move(plane.partial), completed);
    }
    if (err == RGY_ERR_NONE) err = retireErr;
    m_pendingSearchPrefilterFrames.pop_front();
    return err;
}

void NVEncFilterRtgmcSearchPrefilter::clearPendingSearchPrefilterFrames() {
    for (auto &pending : m_pendingSearchPrefilterFrames) {
        for (auto &plane : pending.scenePlanes) {
            recycleSceneChangeBuffer(std::move(plane.partial), plane.readyEvent);
        }
    }
    m_pendingSearchPrefilterFrames.clear();
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (pInputFrame && pInputFrame->ptr[0]) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_cacheFrames[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        m_drainCount = 0;
        auto err = pushCacheFrame(pInputFrame, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        m_inputCount++;
        if (m_inputCount < outputDelay() + 1) {
            return RGY_ERR_NONE;
        }

        const int currentFrame = m_inputCount - outputDelay() - 1;
        err = submitPendingSearchPrefilterFrame(currentFrame, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (m_pendingSearchPrefilterFrames.size() > 1) {
            return resolvePendingSearchPrefilterFrame(ppOutputFrames, pOutputFrameNum, stream);
        }
        return RGY_ERR_NONE;
    }

    if (m_drainCount < drainFrameCount()) {
        const int currentFrame = std::max(0, m_inputCount - drainFrameCount()) + m_drainCount;
        auto err = submitPendingSearchPrefilterFrame(currentFrame, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        m_drainCount++;
    }

    if (!m_pendingSearchPrefilterFrames.empty()) {
        return resolvePendingSearchPrefilterFrame(ppOutputFrames, pOutputFrameNum, stream);
    }

    return RGY_ERR_NONE;
}

void NVEncFilterRtgmcSearchPrefilter::resetTemporalState() {
    // 未消費の判定バッファを完了イベント付きで回収する。
    clearPendingSearchPrefilterFrames();
    // NOTE: Keep the m_cacheFrames ring-buffer slots and their pooled GPU allocations intact.
    // run_filter() unconditionally dereferences m_cacheFrames[0]->frame (e.g. for memcpyKind),
    // so nulling these out here caused a null shared_ptr dereference (ACCESS_VIOLATION) on the
    // first input frame after a reset. The ring contents are overwritten in cold-start order by
    // pushCacheFrame() once m_inputCount is rewound to 0, so stale frame data is harmless; any
    // slot still referenced by in-flight work is detected via use_count()>1 and re-acquired there.
    // Input/drain counters
    m_inputCount = 0;
    m_drainCount = 0;
    m_nFrameIdx = 0;
}

void NVEncFilterRtgmcSearchPrefilter::close() {
    for (auto &resources : m_repairPlaneResources) resources.clear();
    clearPendingSearchPrefilterFrames();
    if (m_searchLumaDump.is_open()) {
        m_searchLumaDump.close();
    }
    m_searchLumaDumpPath.clear();
    m_searchLumaDumpEnabled = false;
    m_searchLumaDumpHeaderWritten = false;
    m_searchLumaDumpFrameCount = 0;
    m_searchLumaDumpMaxFrames = 0;
    m_buildOptions.clear();
    m_sceneChangeBufferPool.clear();
    if (m_searchLumaPool) {
        m_searchLumaPool->clear();
    }
    for (auto &resources : m_searchRefine1PlaneResources) {
        resources.clear();
    }
    for (auto &resources : m_searchRefine2PlaneResources) {
        resources.clear();
    }
    for (auto &resize : m_searchRefine2ResizeEdgeSoftenedSearch) {
        resize.reset();
    }
    for (auto &frame : m_cacheFrames) {
        frame.reset();
    }
    if (m_cacheFramePool) {
        m_cacheFramePool->clear();
    }
    m_inputCount = 0;
    m_drainCount = 0;
    m_frameBuf.clear();
}
