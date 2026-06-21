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
    for (auto &resize : m_searchRefine1ResizeDown) {
        resize.reset();
    }
    for (auto &resize : m_searchRefine1ResizeUp) {
        resize.reset();
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

        auto downParam = std::make_shared<NVEncFilterParamResize>();
        downParam->frameIn = fullInfo;
        downParam->frameOut = halfInfo;
        downParam->interp = RGY_VPP_RESIZE_BILINEAR;
        downParam->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        downParam->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        downParam->bOutOverwrite = false;
        auto downResize = std::make_unique<NVEncFilterResizePlaneProxy>();
        auto sts = downResize->init(downParam, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        auto upParam = std::make_shared<NVEncFilterParamResize>();
        upParam->frameIn = halfInfo;
        upParam->frameOut = fullInfo;
        upParam->interp = RGY_VPP_RESIZE_BILINEAR;
        upParam->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        upParam->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        upParam->bOutOverwrite = false;
        auto upResize = std::make_unique<NVEncFilterResizePlaneProxy>();
        sts = upResize->init(upParam, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        m_searchRefine1ResizeDown[planeIndex] = std::move(downResize);
        m_searchRefine1ResizeUp[planeIndex] = std::move(upResize);
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

std::unique_ptr<CUMemBufPair> NVEncFilterRtgmcSearchPrefilter::getSceneChangeBuffer(const size_t requiredSize) {
    auto pooled = std::find_if(m_sceneChangeBufferPool.begin(), m_sceneChangeBufferPool.end(), [requiredSize](const std::unique_ptr<CUMemBufPair> &buf) {
        return buf && buf->nSize >= requiredSize;
    });
    if (pooled != m_sceneChangeBufferPool.end()) {
        auto buf = std::move(*pooled);
        m_sceneChangeBufferPool.erase(pooled);
        return buf;
    }
    auto buf = std::make_unique<CUMemBufPair>();
    if (buf->alloc(requiredSize) != RGY_ERR_NONE) {
        buf.reset();
    }
    return buf;
}

void NVEncFilterRtgmcSearchPrefilter::recycleSceneChangeBuffer(std::unique_ptr<CUMemBufPair> &&buf) {
    if (buf) {
        m_sceneChangeBufferPool.emplace_back(std::move(buf));
    }
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
    pending->flags.fill(0);
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
    const size_t requiredSize = (size_t)groupCount * 4 * sizeof(uint32_t);
    pending->partial = getSceneChangeBuffer(requiredSize);
    if (!pending->partial) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    pending->groupCount = groupCount;
    pending->sceneThreshold = (uint64_t)RTGMC_SEARCH_PREFILTER_SCENECHANGE * (uint64_t)cur->width * (uint64_t)cur->height;
    auto err = err_to_rgy(cudaMemsetAsync(pending->partial->ptrDevice, 0, requiredSize, stream));
    if (err != RGY_ERR_NONE) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        return err;
    }
    err = rtgmcSearchPrefilterLaunchFuncs(*cur, smoothRadius)->scenechange(*prev2, *prev, *cur, *next, *next2, (uint32_t *)pending->partial->ptrDevice, groupCount, stream);
    if (err != RGY_ERR_NONE) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        AddMessage(RGY_LOG_ERROR, _T("error at %s %s: %s.\n"),
            _T("kernel_rtgmc_search_prefilter_scenechange"), planeName ? planeName : _T("plane"), get_err_mes(err));
        return err;
    }
    err = pending->partial->copyDtoHAsync(stream);
    if (err != RGY_ERR_NONE) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        return err;
    }
    pending->mapEvent = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
    err = err_to_rgy(cudaEventCreateWithFlags(pending->mapEvent.get(), cudaEventDisableTiming));
    if (err != RGY_ERR_NONE) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        return err;
    }
    err = err_to_rgy(cudaEventRecord(*pending->mapEvent, stream));
    if (err != RGY_ERR_NONE) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        return err;
    }
    pending->mapSubmitted = true;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcSearchPrefilter::resolveSceneChangePlane(PendingSceneChangePlane *pending) {
    if (!pending) {
        return RGY_ERR_INVALID_PARAM;
    }
    pending->flags.fill(0);
    if (!pending->mapSubmitted) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        return RGY_ERR_NONE;
    }

    auto err = err_to_rgy(cudaEventSynchronize(*pending->mapEvent));
    if (err != RGY_ERR_NONE) {
        pending->mapSubmitted = false;
        recycleSceneChangeBuffer(std::move(pending->partial));
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc-search-prefilter scene-change readback: %s.\n"), get_err_mes(err));
        return err;
    }
    pending->mapSubmitted = false;
    if (!pending->partial || !pending->partial->ptrHost) {
        recycleSceneChangeBuffer(std::move(pending->partial));
        AddMessage(RGY_LOG_ERROR, _T("failed to access rtgmc-search-prefilter scene-change buffer.\n"));
        return RGY_ERR_NULL_PTR;
    }

    const auto partial = reinterpret_cast<const uint32_t *>(pending->partial->ptrHost);
    for (int i = 0; i < 4; i++) {
        if ((pending->smoothRadius < 2) && i >= 2) {
            continue;
        }
        uint64_t sum = 0;
        const auto *refPartial = partial + (size_t)i * pending->groupCount;
        for (int group = 0; group < pending->groupCount; group++) {
            sum += refPartial[group];
        }
        pending->flags[i] = (sum >= pending->sceneThreshold) ? 1 : 0;
    }
    recycleSceneChangeBuffer(std::move(pending->partial));
    return RGY_ERR_NONE;
}

std::array<int, 4> NVEncFilterRtgmcSearchPrefilter::sceneChangeFlagsForPlane(const PendingSearchPrefilterFrame &pending, const RGY_PLANE plane) const {
    for (const auto &planeFlags : pending.scenePlanes) {
        if (planeFlags.plane == plane) {
            return planeFlags.flags;
        }
    }
    std::array<int, 4> flags = {};
    flags.fill(0);
    return flags;
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

    const auto lumaSceneFlags = sceneChangeFlagsForPlane(pending, RGY_PLANE_Y);
    const auto &planePrev2Eff = lumaSceneFlags[2] ? planeCur : planePrev2;
    const auto &planePrevEff = lumaSceneFlags[0] ? planeCur : planePrev;
    const auto &planeNextEff = lumaSceneFlags[1] ? planeCur : planeNext;
    const auto &planeNext2Eff = lumaSceneFlags[3] ? planeCur : planeNext2;
    const auto repairProfile = rgy_rtgmc_repair_profile_pack(prm->repairProfile);
    const auto *launch = rtgmcSearchPrefilterLaunchFuncs(planeCur, prm->tr0);
    const int smoothRadius = (prm->tr0 <= 0) ? 0 : (prm->tr0 >= 2) ? 2 : 1;
    const bool useSearchRefine1Chain = RTGMC_SEARCH_PREFILTER_USE_SEARCH_REFINE1_CHAIN && prm->searchRefine == 1;
    const bool useSearchRefine2Chain = rtgmcSearchPrefilterUseSearchRefine2Chain(*prm);
    const bool mergeSearchRefine = rtgmcSearchPrefilterMergeSearchRefineEnabled();
    const bool mergeSearchRefine2Tile = rtgmcSearchPrefilterMergeSearchRefine2TileEnabled();
    auto emitSearchRefine1Plane = [&](const int planeIndex,
        const RGYFrameInfo &planePrev2Src, const RGYFrameInfo &planePrevSrc, const RGYFrameInfo &planeCurSrc,
        const RGYFrameInfo &planeNextSrc, const RGYFrameInfo &planeNext2Src, const RGYFrameInfo &planeDstSrc,
        const int fullRangeMode) -> RGY_ERR {
        auto *resizeDown = m_searchRefine1ResizeDown[planeIndex].get();
        auto *resizeUp = m_searchRefine1ResizeUp[planeIndex].get();
        auto &resources = m_searchRefine1PlaneResources[planeIndex];
        if (!resizeDown || !resizeUp || !resources.motionGuide || !resources.halfSearchBase || !resources.halfSearchSmoothed) {
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
        sts = launch->fieldStable(
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, repairProfile, smoothRadius, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_field_stable_search"), planeIndex, get_err_mes(sts));
            return sts;
        }

        RGYFrameInfo *halfSearchBaseFrame = &planeHalfSearchBase;
        int halfSearchBaseFrames = 0;
        auto planeMotionGuideInput = planeMotionGuide;
        sts = resizeDown->filter(&planeMotionGuideInput, &halfSearchBaseFrame, &halfSearchBaseFrames, stream);
        if (sts == RGY_ERR_NONE && (halfSearchBaseFrames != 1 || halfSearchBaseFrame == nullptr)) {
            sts = RGY_ERR_INVALID_CALL;
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to bilinear downsample rtgmc-search-prefilter search_refine1 plane %d: %s.\n"),
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

        RGYFrameInfo *dstFrame = &planeDstWork;
        int dstFrames = 0;
        auto planeHalfSearchSmoothedInput = planeHalfSearchSmoothed;
        sts = resizeUp->filter(&planeHalfSearchSmoothedInput, &dstFrame, &dstFrames, stream);
        if (sts == RGY_ERR_NONE && (dstFrames != 1 || dstFrame == nullptr)) {
            sts = RGY_ERR_INVALID_CALL;
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to bilinear upsample rtgmc-search-prefilter search_refine1 plane %d: %s.\n"),
                planeIndex, get_err_mes(sts));
            return sts;
        }
        if (fullRangeMode != 0) {
            sts = launch->rangeConvert(planeDstWork, fullRangeMode, stream);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                    _T("kernel_rtgmc_search_prefilter_range_convert"), planeIndex, get_err_mes(sts));
            }
        }
        return sts;
    };
    auto emitSearchRefine2Plane = [&](const int planeIndex,
        const RGYFrameInfo &planePrev2Src, const RGYFrameInfo &planePrevSrc, const RGYFrameInfo &planeCurSrc,
        const RGYFrameInfo &planeNextSrc, const RGYFrameInfo &planeNext2Src, const RGYFrameInfo &planeDstSrc,
        const int fullRangeMode) -> RGY_ERR {
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
        sts = launch->fieldStable(
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, repairProfile, smoothRadius, stream);
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
        err = emitSearchRefine1Plane(0, planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff, planeDst, prm->tvRange ? 1 : 0);
    } else if (useSearchRefine2Chain) {
        err = emitSearchRefine2Plane(0, planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff, planeDst, prm->tvRange ? 1 : 0);
    } else {
        err = launch->luma(
            planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDst, prm->searchRefine, repairProfile, prm->tvRange ? 1 : 0, smoothRadius, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), _T("kernel_rtgmc_search_prefilter_luma"), get_err_mes(err));
            return err;
        }
    }

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
        err = launch->halfSearch(
            m_searchLumaDumpStage == "half_search_smoothed",
            planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDebugDump, repairProfile, smoothRadius, stream);
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
            { &planePrev2Eff, &planePrevEff, &planeCur, &planeNextEff, &planeNext2Eff, &planeDebugDump });
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = launch->debug(
            debugStage, planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDebugDump, repairProfile, smoothRadius, stream);
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
            const auto chromaSceneFlags = sceneChangeFlagsForPlane(pending, (RGY_PLANE)iplane);
            const auto &planePrev2CEff = chromaSceneFlags[2] ? planeCurC : planePrev2C;
            const auto &planePrevCEff = chromaSceneFlags[0] ? planeCurC : planePrevC;
            const auto &planeNextCEff = chromaSceneFlags[1] ? planeCurC : planeNextC;
            const auto &planeNext2CEff = chromaSceneFlags[3] ? planeCurC : planeNext2C;
            if (useSearchRefine1Chain) {
                err = emitSearchRefine1Plane(1,
                    planePrev2CEff, planePrevCEff, planeCurC, planeNextCEff, planeNext2CEff, planeDstC,
                    prm->tvRange ? 2 : 0);
            } else if (useSearchRefine2Chain) {
                err = emitSearchRefine2Plane(1,
                    planePrev2CEff, planePrevCEff, planeCurC, planeNextCEff, planeNext2CEff, planeDstC,
                    prm->tvRange ? 2 : 0);
            } else {
                err = launch->luma(
                    planePrev2CEff, planePrevCEff, planeCurC, planeNextCEff, planeNext2CEff,
                    planeDstC, prm->searchRefine, repairProfile, prm->tvRange ? 2 : 0, smoothRadius, stream);
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
            if (plane.mapSubmitted && plane.mapEvent) {
                cudaEventSynchronize(*plane.mapEvent);
                plane.mapSubmitted = false;
            }
            recycleSceneChangeBuffer(std::move(plane.partial));
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
    for (auto &plane : pending.scenePlanes) {
        auto err = resolveSceneChangePlane(&plane);
        if (err != RGY_ERR_NONE) {
            m_pendingSearchPrefilterFrames.pop_front();
            return err;
        }
    }

    auto err = emitPrefilteredFrame(pending, ppOutputFrames, pOutputFrameNum, stream);
    m_pendingSearchPrefilterFrames.pop_front();
    return err;
}

void NVEncFilterRtgmcSearchPrefilter::clearPendingSearchPrefilterFrames() {
    for (auto &pending : m_pendingSearchPrefilterFrames) {
        for (auto &plane : pending.scenePlanes) {
            if (plane.mapSubmitted && plane.mapEvent) {
                cudaEventSynchronize(*plane.mapEvent);
                plane.mapSubmitted = false;
            }
            recycleSceneChangeBuffer(std::move(plane.partial));
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
    // Synchronize and discard any pending scene-change readbacks, then clear the pending queue.
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
    for (auto &resize : m_searchRefine1ResizeDown) {
        resize.reset();
    }
    for (auto &resize : m_searchRefine1ResizeUp) {
        resize.reset();
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
