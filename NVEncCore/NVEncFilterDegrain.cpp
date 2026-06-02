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

#include "NVEncFilterDegrain.h"
#include "NVEncFilterDegrainCommon.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <vector>

#include "rgy_frame_info.h"
#include "rgy_util.h"

RGY_ERR launchNVEncDegrainTemporalSmoothLuma(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const RGYFrameInfo &dst, int tr0, int searchRefine, int rep0, int tvRange, cudaStream_t stream);
RGY_ERR launchNVEncDegrainDebug(
    const RGYFrameInfo &dst, VppDegrainMode mode, const CUMemBuf &mv, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout, int pel, cudaStream_t stream);
RGY_ERR launchNVEncDegrainDownsampleLuma2x(
    const RGYFrameInfo &src, const CUMemBuf &dst, int dstPitch, int dstWidth, int dstHeight, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchSeedAnchorVectors(
    CUMemBuf &vectors, const CUMemBuf &frameAverageMV, int planeBase, int planeStride,
    int planeCount, int pel, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchSeedZeroVectors(
    CUMemBuf &vectors, CUMemBuf &vectorsPrev, CUMemBuf &sads, int planeBase,
    int sadBase, int blockCount, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchExpandCoarseVectors(
    const CUMemBuf &srcVectorsFinal, CUMemBuf &dstVectors, CUMemBuf &dstVectorsPrev, CUMemBuf &dstSads,
    int srcFinalBase, int dstPlaneBase, int dstSadBase, int srcBlockCount,
    int dstBlockCount, int srcBlocksX, int srcBlocksY, int dstBlocksX, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchExportSad(
    CUMemBuf &vectorsFinal, CUMemBuf &sadsInternal, CUMemBuf *outputMotion, CUMemBuf *outputSad,
    int finalBase, int sadBase, int blockCount, int outOffset,
    int referenceDirection, int refs, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchSearchParallel(
    const uint8_t *sourcePlane, const uint8_t *referencePlane, CUMemBuf &vectors,
    int pitch, int width, int height, int planeBase, int blockCount,
    const RGYDegrainBlockLayout &layout, int pixelBytes, int pel, int subpelInterp,
    int pad, int motionCostScale, int lowSadWeightScale,
    int zeroCandidateCostScale, int frameAverageCandidateCostScale,
    int newCandidateCostScale, int level, cudaStream_t stream);
RGY_ERR launchNVEncDegrainMotionSearchSpatialRefine(
    const uint8_t *sourcePlane, const uint8_t *referencePlane,
    CUMemBuf &vectors, const CUMemBuf &vectorsPrev, CUMemBuf &vectorsFinal,
    int pitch, int width, int height, int planeBase, int finalBase,
    int blockCount, const RGYDegrainBlockLayout &layout, int pixelBytes,
    int pel, int subpelInterp, int pad, int motionCostScale,
    int lowSadWeightScale, int newCandidateCostScale, cudaStream_t stream);
RGY_ERR launchNVEncDegrainBuildTemporalMixPlan(
    CUMemBuf &temporalMixPlan, const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    int blockCount, uint32_t thsad, uint32_t disableMask, int refs, cudaStream_t stream);
RGY_ERR launchNVEncDegrainSceneChangeMask(
    CUMemBuf &sceneChangeCounts, CUMemBuf &disableMaskBuf, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout, uint32_t thscd1, uint32_t baseDisableMask, uint64_t thscd2, cudaStream_t stream);
RGY_ERR launchNVEncDegrainOverlapPlane(
    uint8_t *dst, int dstPitch, int pixelBytes,
    const uint8_t *cur, int curPitch,
    const uint8_t *ref0,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    int width, int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    int coveredWidth, int coveredHeight,
    int planeScaleX, int planeScaleY,
    VppDegrainMode mode, int refDirection,
    uint32_t thsad, uint32_t disableMask,
    int refs, int pel, int subpelInterp, cudaStream_t stream);
RGY_ERR launchNVEncDegrainCompensateOverlapPlaneRamp(
    uint8_t *dst, int dstPitch, int pixelBytes,
    const uint8_t *cur, int curPitch,
    const uint8_t *ref0, const uint8_t *ref,
    int refDirection, int width, int height,
    const CUMemBuf &mv, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout,
    int coveredWidth, int coveredHeight,
    int planeScaleX, int planeScaleY,
    uint32_t thsad, uint32_t disableMask,
    const CUMemBuf &windowRamp,
    int refs, int pel, int subpelInterp, cudaStream_t stream);
RGY_ERR launchNVEncDegrainDegrainOverlapPlane(
    uint8_t *dst, int dstPitch, int pixelBytes,
    const uint8_t *cur, int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    int width, int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    int coveredWidth, int coveredHeight,
    int planeScaleX, int planeScaleY,
    uint32_t thsad, uint32_t disableMask,
    int refs, int pel, int subpelInterp, cudaStream_t stream);
RGY_ERR launchNVEncDegrainDegrainOverlapPlanePreweightedRamp(
    uint8_t *dst, int dstPitch, int pixelBytes,
    const uint8_t *cur, int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    int width, int height,
    const CUMemBuf &mv,
    const RGYDegrainBlockLayout &layout,
    int coveredWidth, int coveredHeight,
    int planeScaleX, int planeScaleY,
    const CUMemBuf &windowRamp, const CUMemBuf &temporalMixPlan,
    int refs, int pel, int subpelInterp, cudaStream_t stream);
RGY_ERR launchNVEncDegrainPixelTrace(
    const uint8_t *cur, int curPitch, int pixelBytes,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    int width, int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    int coveredWidth, int coveredHeight,
    int planeScaleX, int planeScaleY,
    uint32_t thsad, uint32_t disableMask,
    int targetX, int targetY,
    CUMemBuf &trace,
    int refs, int pel, int subpelInterp, cudaStream_t stream);

namespace {
uint32_t degrainDisableMask(const RGYDegrainRefDisableArray &disableRefs, const int temporalDirections) {
    uint32_t mask = 0;
    for (int refDirection = 0; refDirection < std::min(temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        if (disableRefs[refDirection]) {
            mask |= (1u << refDirection);
        }
    }
    return mask;
}

size_t degrainTemporalMixPlanBytes(const RGYDegrainBlockLayout &layout) {
    return layout.blockCount() * ((size_t)std::max(layout.temporalDirections, 0) + 1u) * sizeof(float);
}

size_t degrainOverlapBlendTableBytes(const int overlapX, const int overlapY) {
    return ((size_t)std::max(overlapX, 0) + (size_t)std::max(overlapY, 0)) * sizeof(float);
}

void degrainFillOverlapBlendAxis(float *dst, const int overlap) {
    if (dst == nullptr || overlap <= 0) {
        return;
    }
    constexpr float pi = 3.14159265358979323846f;
    for (int i = 0; i < overlap; i++) {
        const float t = ((float)i + 0.5f) / (float)overlap;
        dst[i] = 0.5f + 0.5f * std::cos(pi * t);
    }
}

void appendDegrainPixelTraceRefs(std::ostringstream &oss, const int *record) {
    oss << "\"refs\":[";
    for (int refDirection = 0; refDirection < 4; refDirection++) {
        if (refDirection > 0) {
            oss << ',';
        }
        const int traceOffset = 15 + refDirection * 6;
        oss << "{\"ref_slot\":" << refDirection
            << ",\"confidence\":" << record[traceOffset + 0]
            << ",\"sample\":" << record[traceOffset + 1]
            << ",\"dx\":" << record[traceOffset + 2]
            << ",\"dy\":" << record[traceOffset + 3]
            << ",\"sad\":" << record[traceOffset + 4]
            << ",\"valid\":" << record[traceOffset + 5]
            << '}';
    }
    oss << ']';
}

void logDegrainPixelTraceRecords(RGYLog *log, const int *trace, const RGYFrameInfo &planeCur,
    const RGY_PLANE plane, const int currentFrame, const VppDegrainStage stage, const int reqDelta) {
    if (!log || !trace || trace[0] != 0x4d435054) {
        return;
    }
    std::ostringstream head;
    head << "{\"type\":\"degrain_pixel_trace\",\"kind\":\"summary\""
        << ",\"frame\":" << planeCur.inputFrameId
        << ",\"current_frame\":" << currentFrame
        << ",\"pts\":" << (long long)planeCur.timestamp
        << ",\"dur\":" << (long long)planeCur.duration
        << ",\"stage\":" << (int)stage
        << ",\"request_delta\":" << reqDelta
        << ",\"plane\":" << (int)plane
        << ",\"x\":" << trace[1]
        << ",\"y\":" << trace[2]
        << ",\"width\":" << trace[3]
        << ",\"height\":" << trace[4]
        << ",\"fallback\":" << trace[5]
        << ",\"covered\":" << trace[6]
        << ",\"scale_x\":" << trace[7]
        << ",\"scale_y\":" << trace[8]
        << ",\"block_size_x\":" << trace[9]
        << ",\"block_size_y\":" << trace[10]
        << ",\"overlap_x\":" << trace[11]
        << ",\"overlap_y\":" << trace[12]
        << ",\"step_x\":" << trace[13]
        << ",\"primary_bx\":" << trace[14]
        << ",\"primary_by\":" << trace[15]
        << ",\"block_count_x\":" << trace[16]
        << ",\"block_count_y\":" << trace[17]
        << ",\"sample_sum\":" << trace[18]
        << ",\"sample_count\":" << trace[19]
        << ",\"result\":" << trace[20]
        << ",\"sad_limit\":" << trace[21]
        << ",\"disable_mask\":" << trace[22]
        << ",\"blocks_x\":" << trace[23]
        << ",\"blocks_y\":" << trace[24]
        << ",\"records\":" << trace[25]
        << '}';
    log->write(RGY_LOG_INFO, RGY_LOGT_VPP, _T("%s\n"), char_to_tstring(head.str()).c_str());

    const int recordCount = std::min(trace[25], 4);
    for (int i = 0; i < recordCount; i++) {
        const int *record = trace + 32 + i * 48;
        std::ostringstream oss;
        oss << "{\"type\":\"degrain_pixel_trace\",\"kind\":\"block\""
            << ",\"frame\":" << planeCur.inputFrameId
            << ",\"current_frame\":" << currentFrame
            << ",\"pts\":" << (long long)planeCur.timestamp
            << ",\"dur\":" << (long long)planeCur.duration
            << ",\"stage\":" << (int)stage
            << ",\"request_delta\":" << reqDelta
            << ",\"plane\":" << (int)plane
            << ",\"x\":" << trace[1]
            << ",\"y\":" << trace[2]
            << ",\"block_x\":" << record[0]
            << ",\"block_y\":" << record[1]
            << ",\"block\":" << record[2]
            << ",\"base_x\":" << record[3]
            << ",\"base_y\":" << record[4]
            << ",\"local_x\":" << record[5]
            << ",\"local_y\":" << record[6]
            << ",\"window\":" << record[7]
            << ",\"src_sample\":" << record[8]
            << ",\"sample\":" << record[9]
            << ",\"contribution\":" << record[10]
            << ",\"source_mix\":" << record[11]
            << ",\"reference_mix_sum\":" << record[12]
            << ",\"confidence_sum_raw\":" << record[13]
            << ",\"source_confidence_raw\":" << record[14]
            << ',';
        appendDegrainPixelTraceRefs(oss, record);
        oss << '}';
        log->write(RGY_LOG_INFO, RGY_LOGT_VPP, _T("%s\n"), char_to_tstring(oss.str()).c_str());
    }
}

float degrainTraceReferenceAffinityFromSad(const int thresholdSad, const int measuredSad) {
    if (thresholdSad <= measuredSad) {
        return 0.0f;
    }
    const float sadRatio = (float)measuredSad / (float)thresholdSad;
    const float sadRatio2 = sadRatio * sadRatio;
    return (1.0f - sadRatio2) / (1.0f + sadRatio2);
}

int degrainPlaneScaleX(const RGYFrameInfo *frame, const RGY_PLANE plane) {
    if (!frame || plane == RGY_PLANE_Y) {
        return 1;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
        return 2;
    default:
        return 1;
    }
}

int degrainPlaneScaleY(const RGYFrameInfo *frame, const RGY_PLANE plane) {
    if (!frame || plane == RGY_PLANE_Y) {
        return 1;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
        return 2;
    default:
        return 1;
    }
}

int degrainScaleCovered(const int covered, const int scale) {
    return (covered + std::max(scale, 1) - 1) / std::max(scale, 1);
}

bool degrainCanProcessChroma(const RGYFrameInfo *frame) {
    if (!frame || RGY_CSP_PLANES[frame->csp] < 3) {
        return false;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return true;
    default:
        return false;
    }
}

std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> degrainRenderRefPlanes(const RGYFilterDegrainFrameSet &frames, const RGY_PLANE plane) {
    auto planes = std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS>();
    const auto planeCur = getPlane(frames.cur, plane);
    planes.fill(planeCur);
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        const auto backward = frames.backwardRef(delta) ? frames.backwardRef(delta) : frames.cur;
        const auto forward = frames.forwardRef(delta) ? frames.forwardRef(delta) : frames.cur;
        planes[rgy_degrain_ref_index(delta, false)] = getPlane(backward, plane);
        planes[rgy_degrain_ref_index(delta, true)] = getPlane(forward, plane);
    }
    return planes;
}

bool degrainChromaBuildEnabled(const RGY_CSP csp, const VppDegrain &degrain) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return degrain.chroma;
    default:
        return false;
    }
}

int degrainChromaScaleX(const RGY_CSP csp) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
        return 2;
    case RGY_CHROMAFMT_YUV444:
        return 1;
    default:
        return 1;
    }
}

int degrainChromaScaleY(const RGY_CSP csp) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
        return 2;
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return 1;
    default:
        return 1;
    }
}

RGY_ERR degrainWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto &waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            auto err = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR degrainRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto err = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}

uint32_t degrainAnalyzeFlags(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const bool usesAnalysisLuma, const bool includesChromaSad) {
    uint32_t flags = RGY_DEGRAIN_FRAME_META_FLAG_NONE;
    if (prm && usesAnalysisLuma) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_ANALYSIS_LUMA;
    }
    if (prm && prm->degrain.levels >= 2) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_LEVEL1_REFINE;
    }
    if (prm && includesChromaSad) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_CHROMA_SAD;
    }
    return flags;
}

bool degrainTraceEnvEnabled(const char *name) {
    const auto value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int degrainTraceEnvInt(const char *name, const int fallback) {
    const auto value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

bool degrainEnvFlagEnabled(const char *name) {
    const auto value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

bool degrainMotionSearchProfileEnabled() {
    static const bool enabled = degrainEnvFlagEnabled("NVENC_DEGRAIN_MOTION_SEARCH_PROFILE");
    return enabled;
}

struct RGYDegrainAnalyzeChromaPlanes {
    RGYFrameInfo curU;
    RGYFrameInfo curV;
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refU;
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refV;
    int width;
    int height;
    int enable;
};

bool degrainChromaAnalysisFrameSupported(const RGYFrameInfo *frame) {
    if (!frame || RGY_CSP_PLANES[frame->csp] < 3) {
        return false;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return true;
    default:
        return false;
    }
}

bool degrainValidAnalysisPlane(const RGYFrameInfo &plane) {
    return plane.ptr[0] != nullptr && plane.pitch[0] > 0 && plane.width > 0 && plane.height > 0;
}

RGYDegrainAnalyzeChromaPlanes degrainMakeAnalyzeChromaPlanes(
    const RGYFilterDegrainFrameSet &analysisFrames,
    const RGYFrameInfo &planeCur,
    const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> &refPlanes,
    const int temporalDirections,
    const int requiredDelta,
    const bool requestChroma,
    const bool usedSearchLuma) {
    RGYDegrainAnalyzeChromaPlanes planes = {};
    planes.curU = getPlane(analysisFrames.cur, RGY_PLANE_U);
    planes.curV = getPlane(analysisFrames.cur, RGY_PLANE_V);
    planes.refU.fill(planeCur);
    planes.refV.fill(planeCur);

    bool enable = requestChroma
        && degrainChromaAnalysisFrameSupported(analysisFrames.cur)
        && degrainValidAnalysisPlane(planes.curU)
        && degrainValidAnalysisPlane(planes.curV);
    for (int delta = 1; delta <= requiredDelta; delta++) {
        const int backwardIndex = rgy_degrain_ref_index(delta, false);
        const int forwardIndex = rgy_degrain_ref_index(delta, true);
        const auto backward = analysisFrames.backwardRef(delta);
        const auto forward = analysisFrames.forwardRef(delta);
        planes.refU[backwardIndex] = getPlane(backward, RGY_PLANE_U);
        planes.refV[backwardIndex] = getPlane(backward, RGY_PLANE_V);
        planes.refU[forwardIndex] = getPlane(forward, RGY_PLANE_U);
        planes.refV[forwardIndex] = getPlane(forward, RGY_PLANE_V);
        enable = enable
            && degrainChromaAnalysisFrameSupported(backward)
            && degrainChromaAnalysisFrameSupported(forward);
    }
    for (int dir = 0; dir < temporalDirections; dir++) {
        enable = enable
            && degrainValidAnalysisPlane(planes.refU[dir])
            && degrainValidAnalysisPlane(planes.refV[dir]);
    }
    (void)usedSearchLuma;

    if (!degrainValidAnalysisPlane(planes.curU)) {
        planes.curU = planeCur;
    }
    if (!degrainValidAnalysisPlane(planes.curV)) {
        planes.curV = planeCur;
    }
    for (int dir = 0; dir < temporalDirections; dir++) {
        if (!degrainValidAnalysisPlane(planes.refU[dir])) {
            planes.refU[dir] = refPlanes[dir];
        }
        if (!degrainValidAnalysisPlane(planes.refV[dir])) {
            planes.refV[dir] = refPlanes[dir];
        }
    }
    planes.width = planes.curU.width;
    planes.height = planes.curU.height;
    planes.enable = enable ? 1 : 0;
    return planes;
}

}

NVEncFilterDegrain::NVEncFilterDegrain() :
    NVEncFilter(),
    m_cacheFrames(),
    m_degrain(),
    m_degrainChroma(),
    m_degrainPel1(),
    m_degrainMotionSearchPrograms(),
    m_analysis(),
    m_directAnalyzeResultSet(),
    m_boundAnalyzeResult(),
    m_frameAnalysisData(),
    m_frameAnalysisLayout(),
    m_pendingSceneChange(),
    m_sceneChangeReadbackSAD(),
    m_sceneChangeCounts(),
    m_sceneChangeDisableMask(),
    m_sceneChangeReadbackSADIndex(0),
    m_inputCount(0),
    m_drainCount(0),
    m_bInterlacedWarn(false),
    m_lastAnalysisUsedSearchLuma(false),
    m_lastAnalysisIncludedChroma(false),
    m_useDegrainChromaProgram(false),
    m_debugEnv() {
    m_name = _T("degrain");
}

NVEncFilterDegrain::~NVEncFilterDegrain() {
    close();
}

bool NVEncFilterDegrain::modeImplemented(VppDegrainMode mode) const {
    switch (mode) {
    case VppDegrainMode::Source:
    case VppDegrainMode::Analyze:
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
    case VppDegrainMode::Degrain:
        return true;
    default:
        return false;
    }
}

bool NVEncFilterDegrain::modeRequiresAnalysis(VppDegrainMode mode) const {
    switch (mode) {
    case VppDegrainMode::Analyze:
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
    case VppDegrainMode::Degrain:
        return true;
    case VppDegrainMode::Source:
    default:
        return false;
    }
}

RGY_ERR NVEncFilterDegrain::checkParam(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("degrain requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.blksize <= 0 || prm->degrain.search < 0
        || prm->degrain.thsad < 0 || prm->degrain.thsadc < 0 || prm->degrain.thscd1 < 0 || prm->degrain.thscd2 < 0) {
        AddMessage(RGY_LOG_ERROR, _T("degrain requires non-negative block/search/threshold parameters.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.thscd2 > 255) {
        AddMessage(RGY_LOG_ERROR, _T("degrain thscd2 must satisfy 0 <= thscd2 <= 255.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.pel != 1 && prm->degrain.pel != 2 && prm->degrain.pel != 4) {
        AddMessage(RGY_LOG_ERROR, _T("degrain supports only pel=1, pel=2, or pel=4.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.levels < 1 || prm->degrain.levels > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain Step2c currently supports only levels=1 or levels=2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode) && prm->degrain.levels != 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis requires levels=2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp];
    if (bitDepth == 16 && chromaFormat != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_ERROR, _T("degrain 16-bit input currently supports only YUV420.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode)
        && bitDepth != 8
        && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis supports only 8-bit or 16-bit YUV420 input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode)
        && prm->degrain.blksize != 8 && prm->degrain.blksize != 16 && prm->degrain.blksize != 32) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis supports only blksize=8, 16, or 32.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.overlap < 0 || prm->degrain.overlap >= prm->degrain.blksize) {
        AddMessage(RGY_LOG_ERROR, _T("degrain overlap must satisfy 0 <= overlap < blksize.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.overlap != 0 && prm->degrain.overlap != prm->degrain.blksize / 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain Step2a currently supports only overlap=0 or overlap=blksize/2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.delta < 1 || prm->degrain.delta > 5) {
        AddMessage(RGY_LOG_ERROR, _T("degrain delta must satisfy 1 <= delta <= 5.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const bool allowExtendedDelta =
        prm->degrain.mode == VppDegrainMode::Analyze
        || (prm->degrain.mode == VppDegrainMode::Degrain
            && prm->degrain.stage == VppDegrainStage::TR2);
    if (prm->degrain.delta > 2 && !allowExtendedDelta) {
        AddMessage(RGY_LOG_ERROR, _T("degrain delta>2 is currently supported only for analyze or degrain stage=tr2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.tr0 < 0 || prm->degrain.tr0 > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain tr0 must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.rep0 < 0 || prm->degrain.rep0 > 7) {
        AddMessage(RGY_LOG_ERROR, _T("degrain rep0 must be 0 - 7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.searchRefine < 0 || prm->degrain.searchRefine > 3) {
        AddMessage(RGY_LOG_ERROR, _T("degrain search_refine must be 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.mvSpatialRefine < -1) {
        AddMessage(RGY_LOG_ERROR, _T("degrain mv_spatial_refine must be -1 or greater.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.subpelInterp < 0 || prm->degrain.subpelInterp > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain subpelinterp must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.searchParam < 0 || prm->degrain.pelSearch < 0
        || prm->degrain.lambda < 0 || prm->degrain.lsad < 0
        || prm->degrain.pnew < 0 || prm->degrain.plevel < 0
        || prm->degrain.dct < 0) {
        AddMessage(RGY_LOG_ERROR, _T("degrain searchparam/pelsearch/lambda/lsad/pnew/plevel/dct must be non-negative.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.useFlag < 0 || prm->degrain.useFlag > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain useFlag must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.binomial < -1 || prm->degrain.binomial > 1) {
        AddMessage(RGY_LOG_ERROR, _T("degrain binomial must be auto, true, or false.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::allocCacheFrames(const RGYFrameInfo &frameInfo) {
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
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        if (!frame || frame->alloc() != RGY_ERR_NONE) {
            for (auto &clearFrame : m_cacheFrames) {
                clearFrame.reset();
            }
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::buildKernels(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
    const auto layout = rgy_degrain_make_block_layout(prm->frameOut, prm->degrain);
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int binomial = (prm->degrain.binomial < 0)
        ? ((prm->degrain.stage != VppDegrainStage::TR2) ? 1 : 0)
        : prm->degrain.binomial;
    const auto makeOptions = [&](const int pel, const int planeScaleX, const int planeScaleY) {
        return strsprintf(
        "-D TypePixel=%s"
        " -D DEGRAIN_BLOCK_SIZE=%d"
        " -D DEGRAIN_OVERLAP=%d"
        " -D DEGRAIN_STEP=%d"
        " -D DEGRAIN_BLOCKS_X=%d"
        " -D DEGRAIN_BLOCKS_Y=%d"
        " -D DEGRAIN_COVERED_WIDTH=%d"
        " -D DEGRAIN_COVERED_HEIGHT=%d"
        " -D DEGRAIN_SEARCH=%d"
        " -D DEGRAIN_REFS=%d"
        " -D DEGRAIN_PEL=%d"
        " -D DEGRAIN_SUBPEL_INTERP=%d"
        " -D DEGRAIN_TR0=%d"
        " -D DEGRAIN_SEARCH_REFINE=%d"
        " -D DEGRAIN_REP0=%d"
        " -D DEGRAIN_PIXEL_MAX=%d"
        " -D DEGRAIN_BINOMIAL=%d"
        " -D DEGRAIN_CHROMA=%d"
        " -D DEGRAIN_CHROMA_SCALE_X=%d"
        " -D DEGRAIN_CHROMA_SCALE_Y=%d"
        " -D DEGRAIN_PLANE_SCALE_X=%d"
        " -D DEGRAIN_PLANE_SCALE_Y=%d"
        " -D DEGRAIN_TV_RANGE=%d"
        " -D DEGRAIN_ANALYSIS_LUMA_PRECONVERTED=%d"
        " -D DEGRAIN_FAST_OVERLAP_TRIG=1",
        bitdepth > 8 ? "ushort" : "uchar",
        layout.blockSize,
        layout.overlap,
        layout.step,
        layout.blocksX,
        layout.blocksY,
        layout.coveredWidth,
        layout.coveredHeight,
        layout.search,
        layout.temporalDirections,
        pel,
        prm->degrain.subpelInterp,
        prm->degrain.tr0,
        prm->degrain.searchRefine,
        prm->degrain.rep0,
        pixelMax,
        binomial,
        degrainChromaBuildEnabled(prm->frameOut.csp, prm->degrain) ? 1 : 0,
        degrainChromaScaleX(prm->frameOut.csp),
        degrainChromaScaleY(prm->frameOut.csp),
        planeScaleX,
        planeScaleY,
        prm->degrain.tvRange ? 1 : 0,
        prm->degrain.tvRange ? 1 : 0);
    };
    const int chromaScaleX = degrainChromaScaleX(prm->frameOut.csp);
    const int chromaScaleY = degrainChromaScaleY(prm->frameOut.csp);
    const bool buildChromaProgram = chromaScaleX != 1 || chromaScaleY != 1;

    const auto options = makeOptions(prm->degrain.pel, 1, 1);
    m_degrain = options;
    if (buildChromaProgram) {
        const auto optionsChroma = makeOptions(prm->degrain.pel, chromaScaleX, chromaScaleY);
        m_degrainChroma = optionsChroma;
        m_useDegrainChromaProgram = true;
    } else {
        m_degrainChroma.clear();
        m_useDegrainChromaProgram = false;
    }
    const auto optionsPel1 = makeOptions(1, 1, 1);
    m_degrainPel1 = optionsPel1;
    return RGY_ERR_NONE;
}

NVEncDegrainKernelProgram *NVEncFilterDegrain::degrainRenderProgram(RGY_PLANE plane) {
    return (plane != RGY_PLANE_Y && m_useDegrainChromaProgram) ? &m_degrainChroma : &m_degrain;
}

NVEncDegrainKernelProgram *NVEncFilterDegrain::getDegrainMotionSearchProgram(const std::string &normalizedBuildOptions) {
    if (normalizedBuildOptions.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search build options are empty.\n"));
        return nullptr;
    }
    auto program = m_degrainMotionSearchPrograms.find(normalizedBuildOptions);
    if (program == m_degrainMotionSearchPrograms.end()) {
        auto inserted = m_degrainMotionSearchPrograms.emplace(normalizedBuildOptions, normalizedBuildOptions);
        return &inserted.first->second;
    }
    return &program->second;
}

int NVEncFilterDegrain::cacheIndex(int frame) const {
    return frame % DEGRAIN_CACHE_SIZE;
}

int NVEncFilterDegrain::analysisCacheIndex(int frame) const {
    return frame % RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE;
}

bool NVEncFilterDegrain::useAnalysisLumaCache() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    return prm && modeRequiresAnalysis(prm->degrain.mode) && degrainRequiresAnalysisLumaCache(prm->degrain);
}

bool NVEncFilterDegrain::degrainDebugLogEnabled() const {
    return m_pLog != nullptr && m_pLog->getLogLevel(RGY_LOGT_VPP) <= RGY_LOG_DEBUG;
}

void NVEncFilterDegrain::logAnalyzeBinding(const TCHAR *sourceName, const RGYFrameInfo *frame, const RGYDegrainAnalyzeResult &result) {
    if (!degrainDebugLogEnabled() || !sourceName || !frame || !result.valid()) {
        return;
    }
    const int slotDelta = std::max(1, result.layout.temporalDirections / 2);
    AddMessage(RGY_LOG_DEBUG,
        _T("degrain bind %s slot=d%d cur{id=%d,pts=%lld,dur=%lld} meta{frame=%d,id=%d,pts=%lld,dur=%lld,dirs=%d,flags=0x%08x}.\n"),
        sourceName,
        slotDelta,
        frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
        result.frameIndex, result.inputFrameId, (long long)result.timestamp, (long long)result.duration,
        result.layout.temporalDirections, result.flags);
}

void NVEncFilterDegrain::logLocalAnalysis(const TCHAR *sourceName, const RGYFilterDegrainFrameSet &frames) {
    if (!degrainDebugLogEnabled() || !sourceName || !frames.cur) {
        return;
    }
    const int slotDelta = std::max(1, analysisLayout().temporalDirections / 2);
    AddMessage(RGY_LOG_DEBUG,
        _T("degrain bind %s slot=d%d cur{id=%d,pts=%lld,dur=%lld} analysis{dirs=%d,search-luma=%s}.\n"),
        sourceName,
        slotDelta,
        frames.cur->inputFrameId, (long long)frames.cur->timestamp, (long long)frames.cur->duration,
        analysisLayout().temporalDirections,
        m_lastAnalysisUsedSearchLuma ? _T("attached") : _T("source"));
}

void NVEncFilterDegrain::logAnalysisSamples(const TCHAR *sourceName, const RGYFrameInfo *frame, cudaStream_t stream) {
    if (!degrainDebugLogEnabled() || !sourceName || !frame) {
        return;
    }

    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    const auto &layout = analysisLayout();
    if (!mv || !sad || layout.blockCount() == 0 || layout.temporalDirections <= 0) {
        return;
    }

    const auto entryCount = layout.blockCount() * (size_t)layout.temporalDirections;
    if (layout.mvCount() < entryCount || layout.sadCount() < entryCount) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain sample src=%s skipped: invalid entry count (blocks=%llu, stride=%d).\n"),
            sourceName, (unsigned long long)layout.blockCount(), layout.temporalDirections);
        return;
    }

    if (analysisEvent()() != nullptr) {
        cudaStreamWaitEvent(stream, analysisEvent()(), 0);
    }
    std::vector<RGYDegrainMV> mvValues(layout.mvCount());
    std::vector<RGYDegrainSAD> sadValues(layout.sadCount());
    auto err = err_to_rgy(cudaMemcpyAsync(mvValues.data(), mv->ptr, rgy_degrain_mv_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to copy MV buffer: %s.\n"), sourceName, get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaMemcpyAsync(sadValues.data(), sad->ptr, rgy_degrain_sad_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to copy SAD buffer: %s.\n"), sourceName, get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to synchronize MV/SAD copy: %s.\n"), sourceName, get_err_mes(err));
        return;
    }

    std::array<size_t, 3> sampleBlocks = {
        0,
        layout.blockCount() / 2,
        layout.blockCount() - 1
    };
    static const TCHAR *sampleNames[] = { _T("first"), _T("mid"), _T("last") };
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    const int reqDelta = requestedDelta();
    const auto stage = (prm) ? prm->degrain.stage : VppDegrainStage::Auto;
    for (int dir = 0; dir < std::min(layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); dir++) {
        const int delta = rgy_degrain_delta_from_ref_index(dir);
        const bool forward = rgy_degrain_ref_index_is_forward(dir);
        for (int sample = 0; sample < (int)sampleBlocks.size(); sample++) {
            const size_t block = sampleBlocks[sample];
            if ((sample > 0 && block == sampleBlocks[sample - 1])
                || (sample > 1 && block == sampleBlocks[sample - 2])) {
                continue;
            }
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)dir;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            AddMessage(RGY_LOG_DEBUG,
                _T("degrain sample src=%s cur{id=%d,pts=%lld,dur=%lld} request{delta=%d,stage=%s} layout{blocks=%llu,stride=%d,directions=%d} slot=d%d%s sample=%s block=%llu entry=%llu motion{dx=%d,dy=%d,sad=%u,ref_slot=%u} sad_stats{sad=%u,src=%u,ref=%u} diag{base_sad=%u,score=%u,sad_score=%u}.\n"),
                sourceName,
                frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
                reqDelta, get_cx_desc(list_vpp_degrain_stage, (int)stage),
                (unsigned long long)layout.blockCount(), layout.temporalDirections, layout.temporalDirections,
                delta, forward ? _T("f") : _T("b"),
                sampleNames[sample],
                (unsigned long long)block,
                (unsigned long long)entry,
                (int)mvValue.dx, (int)mvValue.dy, (unsigned int)mvValue.sad, (unsigned int)mvValue.refdir,
                (unsigned int)sadValue.sad, (unsigned int)sadValue.srcAvg, (unsigned int)sadValue.refAvg,
                (unsigned int)mvValue.flags, (unsigned int)mvValue.reserved, (unsigned int)sadValue.reserved);
        }
    }

}

void NVEncFilterDegrain::logReferenceGate(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames,
    const RGYDegrainRefDisableArray &availabilityDisableRefs, const RGYDegrainRefDisableArray &useFlagDisableRefs,
    const RGYDegrainRefDisableArray &disableRefs,
    const std::array<size_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> *sceneChangeBlockCounts,
    const uint32_t scaledThSad, const uint32_t scaledThSCD1, const uint64_t scaledThSCD2) {
    if (!degrainDebugLogEnabled() || !prm || !frames.cur) {
        return;
    }

    auto line = strsprintf(
        _T("degrain gate cur{id=%d,pts=%lld,dur=%lld} slot=d%d mode=%s stage=%s sad_limit=%u scene_limit1=%u scene_limit2=%llu reference_policy=%d"),
        frames.cur->inputFrameId, (long long)frames.cur->timestamp, (long long)frames.cur->duration,
        std::max(1, analysisLayout().temporalDirections / 2),
        get_cx_desc(list_vpp_degrain_mode, (int)prm->degrain.mode),
        get_cx_desc(list_vpp_degrain_stage, (int)prm->degrain.stage),
        scaledThSad,
        scaledThSCD1,
        (unsigned long long)scaledThSCD2,
        prm->degrain.useFlag);
    for (int dir = 0; dir < std::min(analysisLayout().temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); dir++) {
        const int delta = rgy_degrain_delta_from_ref_index(dir);
        const bool forward = rgy_degrain_ref_index_is_forward(dir);
        const bool inRange = forward ? frames.forwardRefInRange(delta) : frames.backwardRefInRange(delta);
        const auto *ref = forward ? frames.forwardRef(delta) : frames.backwardRef(delta);
        const bool availabilityDisabled = availabilityDisableRefs[dir];
        const bool useFlagDisabled = useFlagDisableRefs[dir] && !availabilityDisabled;
        const bool sceneChangeDisabled = disableRefs[dir] && !availabilityDisabled && !useFlagDisabled;
        const auto sceneChangeText = sceneChangeBlockCounts
            ? strsprintf(_T("%llu/%llu"), (unsigned long long)(*sceneChangeBlockCounts)[dir], (unsigned long long)scaledThSCD2)
            : tstring(_T("-"));
        line += strsprintf(
            _T(" d%d%s{id=%d,in=%d,policy=%d,scene=%s,disabled=%d}"),
            delta, forward ? _T("f") : _T("b"),
            ref ? ref->inputFrameId : -1,
            inRange ? 1 : 0,
            useFlagDisabled ? 1 : 0,
            sceneChangeText.c_str(),
            (availabilityDisabled || sceneChangeDisabled || useFlagDisabled) ? 1 : 0);
    }
    AddMessage(RGY_LOG_DEBUG, _T("%s.\n"), line.c_str());
}

int NVEncFilterDegrain::requestedDelta() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    return prm ? clamp(prm->degrain.delta, 1, RGY_DEGRAIN_MAX_DELTA) : 1;
}

int NVEncFilterDegrain::outputDelay() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        return FILTER_DEFAULT_DEGRAIN_DELTA;
    }
    return prm->degrain.delta + (modeRequiresAnalysis(prm->degrain.mode) ? prm->degrain.tr0 : 0);
}

int NVEncFilterDegrain::drainFrameCount() const {
    return std::min(outputDelay(), m_inputCount);
}

RGY_ERR NVEncFilterDegrain::pushCacheFrame(const RGYFrameInfo *pInputFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto pCacheFrame = &m_cacheFrames[cacheIndex(m_inputCount)]->frame;
    auto err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyFrameAsync(pCacheFrame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy input to degrain cache slot %d: %s.\n"),
            cacheIndex(m_inputCount), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain cache copy event: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFrameProp(pCacheFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGYFilterDegrainFrameSet NVEncFilterDegrain::resolveFrameSet(const int currentFrame) const {
    RGYFilterDegrainFrameSet frames;
    if (m_inputCount <= 0) {
        return frames;
    }
    if (currentFrame < 0) {
        return frames;
    }

    frames.cur = &m_cacheFrames[cacheIndex(currentFrame)]->frame;
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        frames.backwardInRange[delta - 1] = (currentFrame + delta) < m_inputCount;
        frames.forwardInRange[delta - 1] = (currentFrame - delta) >= 0;
        const int backwardFrame = frames.backwardInRange[delta - 1] ? (currentFrame + delta) : currentFrame;
        const int forwardFrame = frames.forwardInRange[delta - 1] ? (currentFrame - delta) : currentFrame;
        frames.backward[delta - 1] = &m_cacheFrames[cacheIndex(backwardFrame)]->frame;
        frames.forward[delta - 1] = &m_cacheFrames[cacheIndex(forwardFrame)]->frame;
    }
    return frames;
}

RGYFilterDegrainProcessFrameSet NVEncFilterDegrain::resolveFrames(bool hasInput) const {
    RGYFilterDegrainProcessFrameSet frames = {};
    frames.currentFrame = -1;
    if (m_inputCount <= 0) {
        return frames;
    }

    const int currentFrame = hasInput
        ? (m_inputCount - outputDelay() - 1)
        : (std::max(0, m_inputCount - drainFrameCount()) + m_drainCount);
    if (currentFrame < 0) {
        return frames;
    }

    frames.currentFrame = currentFrame;
    frames.render = resolveFrameSet(currentFrame);
    frames.analysis = (degrainGetAttachedSearchLuma(frames.render.cur) || !useAnalysisLumaCache())
        ? frames.render
        : resolveAnalysisFrameSet(currentFrame);
    return frames;
}

RGY_ERR NVEncFilterDegrain::emitSourceFrame(const RGYFrameInfo *pCurrentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pCurrentFrame || pCurrentFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getCudaMemcpyKind(pCurrentFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyFrameAsync(ppOutputFrames[0], pCurrentFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain source output: %s.\n"), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain source copy event: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pCurrentFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::runSourceMode(const RGYFilterDegrainFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    return emitSourceFrame(frames.cur, ppOutputFrames, pOutputFrameNum, stream, {}, event);
}

RGY_ERR NVEncFilterDegrain::unsupportedModeError(VppDegrainMode mode) {
    AddMessage(RGY_LOG_ERROR,
        _T("degrain mode %s is not implemented. Supported today: source, analyze, mv(debug scaffold), sad(debug scaffold), motionback, motionforw, motionback2, motionforw2, degrain.\n"),
        get_cx_desc(list_vpp_degrain_mode, (int)mode));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    close();
    loadDebugEnv();

    sts = buildKernels(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = allocCacheFrames(prm->frameIn);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain cache: %s.\n"), get_err_mes(sts));
        return sts;
    }

    sts = allocAnalysisBuffers(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_inputCount = 0;
    m_drainCount = 0;
    m_bInterlacedWarn = false;
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    clearDirectAnalyzeResult();
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_DATA);

    if (modeRequiresAnalysis(prm->degrain.mode)) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain %s mode uses MV analysis output.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)prm->degrain.mode));
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::runResolvedFrames(const RGYFilterDegrainProcessFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    switch (prm->degrain.mode) {
    case VppDegrainMode::Source:
        return runSourceMode(frames.render, ppOutputFrames, pOutputFrameNum, stream, event);
    case VppDegrainMode::Analyze:
        return runAnalyzeMode(frames, frames.currentFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
        return runDebugMode(frames, frames.currentFrame, prm->degrain.mode, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
        return runCompensateMode(frames, frames.currentFrame, prm->degrain.mode, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::Degrain:
        return runDegrainMode(frames, frames.currentFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    default:
        return unsupportedModeError(prm->degrain.mode);
    }
}

RGY_ERR NVEncFilterDegrain::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    const std::vector<RGYCudaEvent> wait_events;
    RGYCudaEvent filterEvent;
    auto *event = &filterEvent;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!modeImplemented(prm->degrain.mode)) {
        return unsupportedModeError(prm->degrain.mode);
    }

    const bool hasInput = (pInputFrame != nullptr && pInputFrame->ptr[0] != nullptr);
    if (hasInput) {
        if (!m_bInterlacedWarn && (pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED)) {
            AddMessage(RGY_LOG_WARN, _T("degrain Step1 skeleton is frame-domain only; current modes ignore interlaced-specific processing.\n"));
            m_bInterlacedWarn = true;
        }

        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_cacheFrames[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        m_drainCount = 0;
        RGYCudaEvent cacheCopyEvent;
        auto sts = pushCacheFrame(pInputFrame, stream, wait_events, &cacheCopyEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        m_inputCount++;
        if (useAnalysisLumaCache()) {
            sts = ensureAnalysisLumaGenerated(m_inputCount - 1 - prm->degrain.tr0, stream, { cacheCopyEvent });
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (m_inputCount < outputDelay() + 1) {
            return RGY_ERR_NONE;
        }

        return runResolvedFrames(resolveFrames(true), ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    }

    if (m_drainCount < drainFrameCount()) {
        if (useAnalysisLumaCache()) {
            const int currentFrame = std::max(0, m_inputCount - drainFrameCount()) + m_drainCount;
            const auto sts = ensureAnalysisLumaGenerated(currentFrame + prm->degrain.delta, stream, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto frames = resolveFrames(false);
        m_drainCount++;
        return runResolvedFrames(frames, ppOutputFrames, pOutputFrameNum, stream, {}, event);
    }

    if (prm->degrain.mode == VppDegrainMode::Degrain && !m_pendingSceneChange.empty()) {
        return resolvePendingSceneChangeFrame(ppOutputFrames, pOutputFrameNum, stream, event);
    }

    return RGY_ERR_NONE;
}

void NVEncFilterDegrain::close() {
    m_degrain.clear();
    m_degrainChroma.clear();
    m_degrainPel1.clear();
    m_degrainMotionSearchPrograms.clear();
    m_useDegrainChromaProgram = false;
    clearPendingSceneChange();
    for (auto &sad : m_sceneChangeReadbackSAD) {
        sad.clear();
        sad.shrink_to_fit();
    }
    m_sceneChangeCounts.reset();
    m_sceneChangeDisableMask.reset();
    m_sceneChangeReadbackSADIndex = 0;
    m_analysis.mv.reset();
    m_analysis.sad.reset();
    m_analysis.windowRampY.reset();
    m_analysis.windowRampC.reset();
    m_analysis.temporalMixPlanY.reset();
    m_analysis.temporalMixPlanC.reset();
    m_analysis.temporalMixPrior.reset();
    clearDirectAnalyzeResult();
    clearFrameAnalysisData();
    for (auto &luma : m_analysis.analysisLuma) {
        luma.reset();
    }
    for (auto &frame : m_analysis.analysisLumaFrame) {
        frame = RGYFrameInfo();
    }
    m_analysis.analysisLumaFrameNumbers.fill(-1);
    for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
        analysisLumaEvent.reset();
    }
    m_analysis.analysisLumaEvent.reset();
    for (auto &luma : m_analysis.lumaLevel1) {
        luma.reset();
    }
    m_analysis.event.reset();
    m_analysis.mvBytes = 0;
    m_analysis.sadBytes = 0;
    m_analysis.analysisLumaBytes = 0;
    m_analysis.lumaLevel1Bytes = 0;
    m_analysis.analysisLumaWidth = 0;
    m_analysis.analysisLumaHeight = 0;
    m_analysis.analysisLumaPitch = 0;
    m_analysis.analysisLumaGeneratedUntil = -1;
    m_analysis.lumaLevel1Width = 0;
    m_analysis.lumaLevel1Height = 0;
    m_analysis.lumaLevel1Pitch = 0;
    m_analysis.lastFrameIndex = -1;
    m_analysis.lastInputFrameId = -1;
    m_analysis.lastTimestamp = 0;
    m_analysis.lastDuration = 0;
    m_analysis.layout = {};
    m_analysis.layoutLevel1 = {};
    m_analysis.mode = VppDegrainMode::Source;
    for (auto &frame : m_cacheFrames) {
        frame.reset();
    }
    m_inputCount = 0;
    m_drainCount = 0;
    m_bInterlacedWarn = false;
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    m_frameBuf.clear();
}

RGY_ERR NVEncFilterDegrain::emitDebugFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, RGYCudaEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain debug output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (mode != VppDegrainMode::MV && mode != VppDegrainMode::SAD) {
        AddMessage(RGY_LOG_ERROR, _T("invalid debug mode for degrain: %s.\n"), get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getCudaMemcpyKind(frames.cur->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (analysisEvent()() != nullptr) {
        auto err = err_to_rgy(cudaStreamWaitEvent(stream, analysisEvent()(), 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain analysis event for debug output: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    auto err = copyFrameAsync(ppOutputFrames[0], frames.cur, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain debug output: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto planeDst = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    err = launchNVEncDegrainDebug(planeDst, mode, *mv, *sad, analysisLayout(), prm->degrain.pel, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to render degrain %s debug frame: %s.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain debug event: %s.\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::emitCompensateFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
    const RGYDegrainRefDisableArray &disableRefs, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, RGYCudaEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!m_analysis.temporalMixPrior) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate output requires temporal mix prior table.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    RGYDegrainRefDir refDirection = RGYDegrainRefDir::Backward;
    if (!rgy_degrain_refdir_from_mode(mode, &refDirection)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid compensate mode for degrain: %s.\n"), get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }
    const int refIndex = rgy_degrain_refdir_index(refDirection);
    if (refIndex >= analysisLayout().temporalDirections) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s is not available for delta=%d.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), rgy_degrain_delta_from_ref_index(analysisLayout().temporalDirections - 1));
        return RGY_ERR_UNSUPPORTED;
    }

    const int refDelta = rgy_degrain_delta_from_ref_index(refIndex);
    const bool refForward = rgy_degrain_ref_index_is_forward(refIndex);
    const RGYFrameInfo *reference = refForward ? frames.forwardRef(refDelta) : frames.backwardRef(refDelta);
    if (!reference || reference->ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires a valid reference frame.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    if (analysisEvent()() != nullptr) {
        auto err = err_to_rgy(cudaStreamWaitEvent(stream, analysisEvent()(), 0));
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    auto err = copyFrameAsync(ppOutputFrames[0], frames.cur, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain %s output: %s.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    const uint32_t compensateThSad = std::numeric_limits<uint32_t>::max();
    const uint32_t disableMask = degrainDisableMask(disableRefs, analysisLayout().temporalDirections);
    const bool useOverlapRamp = analysisLayout().overlap > 0;
    const int pixelBytes = (RGY_CSP_BIT_DEPTH[frames.cur->csp] > 8) ? 2 : 1;
    auto ensureWindowRamp = [&](RGYDegrainWindowRampState &state, const int planeScaleX, const int planeScaleY) {
        const int planeOverlapX = std::max(analysisLayout().overlap / std::max(planeScaleX, 1), 0);
        const int planeOverlapY = std::max(analysisLayout().overlap / std::max(planeScaleY, 1), 0);
        const auto rampBytes = degrainOverlapBlendTableBytes(planeOverlapX, planeOverlapY);
        if (rampBytes == 0) {
            state.reset();
            return RGY_ERR_NONE;
        }
        if (state.reusable(planeOverlapX, planeOverlapY, rampBytes)) {
            return RGY_ERR_NONE;
        }

        std::vector<float> ramp(planeOverlapX + planeOverlapY);
        degrainFillOverlapBlendAxis(ramp.data(), planeOverlapX);
        degrainFillOverlapBlendAxis(ramp.data() + planeOverlapX, planeOverlapY);
        auto rampBuf = std::make_unique<CUMemBuf>(rampBytes);
        auto err = rampBuf->alloc();
        if (err != RGY_ERR_NONE) {
            state.reset();
            return err;
        }
        err = err_to_rgy(cudaMemcpyAsync(rampBuf->ptr, ramp.data(), rampBytes, cudaMemcpyHostToDevice, stream));
        if (err != RGY_ERR_NONE) {
            state.reset();
            return err;
        }
        state.ramp = std::move(rampBuf);
        state.bytes = rampBytes;
        state.overlapX = planeOverlapX;
        state.overlapY = planeOverlapY;
        return RGY_ERR_NONE;
    };

    auto renderPlane = [&](const RGY_PLANE plane, const uint32_t planeThSad) {
        const auto planeDst = getPlane(ppOutputFrames[0], plane);
        const auto planeCur = getPlane(frames.cur, plane);
        const auto refPlanes = degrainRenderRefPlanes(frames, plane);
        if (planeDst.ptr[0] == nullptr || planeCur.ptr[0] == nullptr || refPlanes[0].ptr[0] == nullptr || refPlanes[1].ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires valid plane %d.\n"),
                get_cx_desc(list_vpp_degrain_mode, (int)mode), (int)plane);
            return RGY_ERR_INVALID_CALL;
        }
        const int planePitch = planeCur.pitch[0];
        for (int i = 0; i < (int)refPlanes.size(); i++) {
            if (refPlanes[i].pitch[0] != planePitch) {
                AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires matching plane %d pitch: cur=%d, ref[%d]=%d.\n"),
                    get_cx_desc(list_vpp_degrain_mode, (int)mode), (int)plane, planePitch, i, refPlanes[i].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        const int planeScaleX = degrainPlaneScaleX(frames.cur, plane);
        const int planeScaleY = degrainPlaneScaleY(frames.cur, plane);
        const auto refPlane = refPlanes[refIndex];
        CUMemBuf *windowRamp = nullptr;
        if (useOverlapRamp) {
            auto &rampState = (plane == RGY_PLANE_Y) ? m_analysis.windowRampY : m_analysis.windowRampC;
            const auto rampErr = ensureWindowRamp(rampState, planeScaleX, planeScaleY);
            if (rampErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain overlap ramp buffer: %s.\n"), get_err_mes(rampErr));
                return rampErr;
            } else if (rampState.ramp) {
                windowRamp = rampState.ramp.get();
            }
        }
        if (windowRamp) {
            return launchNVEncDegrainCompensateOverlapPlaneRamp(
                planeDst.ptr[0], planeDst.pitch[0], pixelBytes,
                planeCur.ptr[0], planePitch,
                planeCur.ptr[0],
                refPlane.ptr[0],
                refIndex,
                planeDst.width, planeDst.height,
                *mv, *sad,
                analysisLayout(),
                degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                planeScaleX,
                planeScaleY,
                planeThSad,
                disableMask,
                *windowRamp,
                analysisLayout().temporalDirections, prm->degrain.pel, prm->degrain.subpelInterp, stream);
        }
        return launchNVEncDegrainOverlapPlane(
            planeDst.ptr[0], planeDst.pitch[0], pixelBytes,
            planeCur.ptr[0], planePitch,
            planeCur.ptr[0],
            refPlanes[0].ptr[0],
            refPlanes[1].ptr[0],
            refPlanes[2].ptr[0],
            refPlanes[3].ptr[0],
            refPlanes[4].ptr[0],
            refPlanes[5].ptr[0],
            refPlanes[6].ptr[0],
            refPlanes[7].ptr[0],
            refPlanes[8].ptr[0],
            refPlanes[9].ptr[0],
            planeDst.width, planeDst.height,
            *mv, *sad, *m_analysis.temporalMixPrior,
            analysisLayout(),
            degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
            degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
            planeScaleX,
            planeScaleY,
            mode,
            refIndex,
            planeThSad,
            disableMask,
            analysisLayout().temporalDirections, prm->degrain.pel, prm->degrain.subpelInterp, stream);
    };

    const bool processChroma = degrainCanProcessChroma(frames.cur);
    const std::array<RGY_PLANE, 3> planes = { RGY_PLANE_Y, RGY_PLANE_U, RGY_PLANE_V };
    for (int iplane = 0; iplane < (processChroma ? (int)planes.size() : 1); iplane++) {
        const auto plane = planes[iplane];
        err = renderPlane(plane, compensateThSad);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to render degrain %s %s output: %s.\n"),
                plane == RGY_PLANE_Y ? _T("luma") : _T("chroma"),
                get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
            return err;
        }
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain compensate event: %s.\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::emitDegrainFrame(const RGYFilterDegrainFrameSet &frames,
    const RGYDegrainRefDisableArray &disableRefs, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, RGYCudaEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!m_analysis.temporalMixPrior) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain output requires temporal mix prior table.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto planeDstY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    const auto planeCurY = getPlane(frames.cur, RGY_PLANE_Y);
    if (planeDstY.ptr[0] == nullptr || planeCurY.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires valid luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    if (analysisEvent()() != nullptr) {
        auto err = err_to_rgy(cudaStreamWaitEvent(stream, analysisEvent()(), 0));
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    const bool processChroma = degrainCanProcessChroma(frames.cur);
    const bool copyDegrainOutput = m_debugEnv.forceDegrainCopy
        || rgy_csp_has_alpha(frames.cur->csp)
        || RGY_CSP_PLANES[frames.cur->csp] != (processChroma ? 3 : 1)
        || (!processChroma && RGY_CSP_CHROMA_FORMAT[frames.cur->csp] != RGY_CHROMAFMT_MONOCHROME);
    auto err = RGY_ERR_NONE;
    if (copyDegrainOutput) {
        err = copyFrameAsync(ppOutputFrames[0], frames.cur, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain degrain output: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    const uint32_t disableMask = degrainDisableMask(disableRefs, analysisLayout().temporalDirections);
    const bool useOverlapRamp = analysisLayout().overlap > 0;
    const bool pixelTrace = m_debugEnv.pixelTrace;
    const int pixelTraceX = m_debugEnv.pixelTraceX;
    const int pixelTraceY = m_debugEnv.pixelTraceY;
    const int pixelTraceFrame = m_debugEnv.pixelTraceFrame;
    const int pixelBytes = (RGY_CSP_BIT_DEPTH[frames.cur->csp] > 8) ? 2 : 1;
    auto ensureWindowRamp = [&](RGYDegrainWindowRampState &state, const int planeScaleX, const int planeScaleY) {
        const int planeOverlapX = std::max(analysisLayout().overlap / std::max(planeScaleX, 1), 0);
        const int planeOverlapY = std::max(analysisLayout().overlap / std::max(planeScaleY, 1), 0);
        const auto rampBytes = degrainOverlapBlendTableBytes(planeOverlapX, planeOverlapY);
        if (rampBytes == 0) {
            state.reset();
            return RGY_ERR_NONE;
        }
        if (state.reusable(planeOverlapX, planeOverlapY, rampBytes)) {
            return RGY_ERR_NONE;
        }

        std::vector<float> ramp(planeOverlapX + planeOverlapY);
        degrainFillOverlapBlendAxis(ramp.data(), planeOverlapX);
        degrainFillOverlapBlendAxis(ramp.data() + planeOverlapX, planeOverlapY);
        auto rampBuf = std::make_unique<CUMemBuf>(rampBytes);
        auto err = rampBuf->alloc();
        if (err != RGY_ERR_NONE) {
            state.reset();
            return err;
        }
        err = err_to_rgy(cudaMemcpyAsync(rampBuf->ptr, ramp.data(), rampBytes, cudaMemcpyHostToDevice, stream));
        if (err != RGY_ERR_NONE) {
            state.reset();
            return err;
        }
        state.ramp = std::move(rampBuf);
        state.bytes = rampBytes;
        state.overlapX = planeOverlapX;
        state.overlapY = planeOverlapY;
        return RGY_ERR_NONE;
    };
    bool temporalMixPlanYReady = false;
    bool temporalMixPlanCReady = false;
    auto ensureTemporalMixPlan = [&](RGYDegrainTemporalMixPlanState &state, bool &ready, const uint32_t scaledThSad) {
        const auto planBytes = degrainTemporalMixPlanBytes(analysisLayout());
        if (ready && state.reusable(planBytes, scaledThSad, disableMask)) {
            return RGY_ERR_NONE;
        }
        if (planBytes == 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid degrain temporal mix plan buffer geometry.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (!state.plan || state.plan->nSize != planBytes) {
            state.plan = std::make_unique<CUMemBuf>(planBytes);
            auto allocErr = state.plan->alloc();
            if (allocErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain temporal mix plan buffer.\n"));
                return allocErr;
            }
        }

        auto err = launchNVEncDegrainBuildTemporalMixPlan(
            *state.plan,
            *mv,
            *sad,
            *m_analysis.temporalMixPrior,
            (int)analysisLayout().blockCount(),
            scaledThSad,
            disableMask,
            analysisLayout().temporalDirections,
            stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build degrain temporal mix plan: %s.\n"), get_err_mes(err));
            state.event.reset();
            return err;
        }
        err = degrainRecordEvent(stream, &state.event);
        if (err != RGY_ERR_NONE) {
            state.event.reset();
            return err;
        }
        state.bytes = planBytes;
        state.thsad = scaledThSad;
        state.disableMask = disableMask;
        ready = true;
        return RGY_ERR_NONE;
    };
    auto renderPlane = [&](const RGY_PLANE plane, const uint32_t scaledThSad) {
        const auto planeDst = getPlane(ppOutputFrames[0], plane);
        const auto planeCur = getPlane(frames.cur, plane);
        const auto refPlanes = degrainRenderRefPlanes(frames, plane);
        if (planeDst.ptr[0] == nullptr || planeCur.ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires valid plane %d.\n"), (int)plane);
            return RGY_ERR_INVALID_CALL;
        }
        const int planePitch = planeCur.pitch[0];
        for (int i = 0; i < (int)refPlanes.size(); i++) {
            if (refPlanes[i].pitch[0] != planePitch) {
                AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires matching plane %d pitch: cur=%d, ref[%d]=%d.\n"),
                    (int)plane, planePitch, i, refPlanes[i].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        const int planeScaleX = degrainPlaneScaleX(frames.cur, plane);
        const int planeScaleY = degrainPlaneScaleY(frames.cur, plane);
        CUMemBuf *windowRamp = nullptr;
        if (useOverlapRamp) {
            auto &rampState = (plane == RGY_PLANE_Y) ? m_analysis.windowRampY : m_analysis.windowRampC;
            const auto rampErr = ensureWindowRamp(rampState, planeScaleX, planeScaleY);
            if (rampErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain overlap ramp buffer: %s.\n"), get_err_mes(rampErr));
                return rampErr;
            } else if (rampState.ramp) {
                windowRamp = rampState.ramp.get();
            }
        }
        CUMemBuf *temporalMixPlan = nullptr;
        if (windowRamp) {
            auto &planState = (plane == RGY_PLANE_Y) ? m_analysis.temporalMixPlanY : m_analysis.temporalMixPlanC;
            auto &planReady = (plane == RGY_PLANE_Y) ? temporalMixPlanYReady : temporalMixPlanCReady;
            auto err = ensureTemporalMixPlan(planState, planReady, scaledThSad);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (planState.plan) {
                temporalMixPlan = planState.plan.get();
            } else {
                AddMessage(RGY_LOG_ERROR, _T("degrain temporal mix plan buffer was not prepared.\n"));
                return RGY_ERR_INVALID_CALL;
            }
        }
        if (pixelTrace && plane == RGY_PLANE_Y
            && (pixelTraceFrame < 0 || planeCur.inputFrameId == pixelTraceFrame)) {
            CUMemBuf traceBuf(sizeof(int) * 256);
            auto traceErr = traceBuf.alloc();
            if (traceErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_WARN, _T("failed to allocate degrain pixel trace buffer.\n"));
            } else {
                traceErr = launchNVEncDegrainPixelTrace(
                    planeCur.ptr[0], planePitch, pixelBytes,
                    refPlanes[0].ptr[0],
                    refPlanes[1].ptr[0],
                    refPlanes[2].ptr[0],
                    refPlanes[3].ptr[0],
                    refPlanes[4].ptr[0],
                    refPlanes[5].ptr[0],
                    refPlanes[6].ptr[0],
                    refPlanes[7].ptr[0],
                    refPlanes[8].ptr[0],
                    refPlanes[9].ptr[0],
                    planeDst.width, planeDst.height,
                    *mv, *sad, *m_analysis.temporalMixPrior,
                    analysisLayout(),
                    degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                    degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                    planeScaleX,
                    planeScaleY,
                    scaledThSad,
                    disableMask,
                    pixelTraceX,
                    pixelTraceY,
                    traceBuf,
                    analysisLayout().temporalDirections, prm->degrain.pel, prm->degrain.subpelInterp, stream);
                if (traceErr != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_WARN, _T("failed to launch degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                } else {
                    std::array<int, 256> trace = {};
                    traceErr = err_to_rgy(cudaMemcpyAsync(trace.data(), traceBuf.ptr, sizeof(int) * trace.size(), cudaMemcpyDeviceToHost, stream));
                    if (traceErr != RGY_ERR_NONE) {
                        AddMessage(RGY_LOG_WARN, _T("failed to copy degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                    } else {
                        traceErr = err_to_rgy(cudaStreamSynchronize(stream));
                        if (traceErr != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_WARN, _T("failed to synchronize degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                        } else {
                            logDegrainPixelTraceRecords(m_pLog.get(), trace.data(), planeCur, plane, -1, prm->degrain.stage, requestedDelta());
                        }
                    }
                }
            }
        }
        if (windowRamp) {
            return launchNVEncDegrainDegrainOverlapPlanePreweightedRamp(
                planeDst.ptr[0], planeDst.pitch[0], pixelBytes,
                planeCur.ptr[0], planePitch,
                refPlanes[0].ptr[0],
                refPlanes[1].ptr[0],
                refPlanes[2].ptr[0],
                refPlanes[3].ptr[0],
                refPlanes[4].ptr[0],
                refPlanes[5].ptr[0],
                refPlanes[6].ptr[0],
                refPlanes[7].ptr[0],
                refPlanes[8].ptr[0],
                refPlanes[9].ptr[0],
                planeDst.width, planeDst.height,
                *mv,
                analysisLayout(),
                degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                planeScaleX,
                planeScaleY,
                *windowRamp,
                *temporalMixPlan,
                analysisLayout().temporalDirections, prm->degrain.pel, prm->degrain.subpelInterp, stream);
        }
        return launchNVEncDegrainDegrainOverlapPlane(
            planeDst.ptr[0], planeDst.pitch[0], pixelBytes,
            planeCur.ptr[0], planePitch,
            refPlanes[0].ptr[0],
            refPlanes[1].ptr[0],
            refPlanes[2].ptr[0],
            refPlanes[3].ptr[0],
            refPlanes[4].ptr[0],
            refPlanes[5].ptr[0],
            refPlanes[6].ptr[0],
            refPlanes[7].ptr[0],
            refPlanes[8].ptr[0],
            refPlanes[9].ptr[0],
            planeDst.width, planeDst.height,
            *mv, *sad, *m_analysis.temporalMixPrior,
            analysisLayout(),
            degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
            degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
            planeScaleX,
            planeScaleY,
            scaledThSad,
            disableMask,
            analysisLayout().temporalDirections, prm->degrain.pel, prm->degrain.subpelInterp, stream);
    };

    const bool includeChromaSad = analysisSADIncludesChroma(prm);
    const uint32_t scaledThSad = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, includeChromaSad);
    const uint32_t scaledThSadC = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsadc, includeChromaSad);
    const std::array<RGY_PLANE, 3> planes = { RGY_PLANE_Y, RGY_PLANE_U, RGY_PLANE_V };
    for (int iplane = 0; iplane < (processChroma ? (int)planes.size() : 1); iplane++) {
        const auto plane = planes[iplane];
        err = renderPlane(plane, plane == RGY_PLANE_Y ? scaledThSad : scaledThSadC);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to render degrain degrain %s output: %s.\n"),
                plane == RGY_PLANE_Y ? _T("luma") : _T("chroma"), get_err_mes(err));
            return err;
        }
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain apply event: %s.\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::attachAnalysisData(const RGYFrameInfo *sourceFrame, RGYFrameInfo *outputFrame,
    const int currentFrame, cudaStream_t stream, const RGYCudaEvent &frameCopyEvent, RGYCudaEvent *event) {
    if (!sourceFrame || !outputFrame || !m_analysis.mv || !m_analysis.sad) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto mv = std::make_unique<CUMemBuf>(m_analysis.mvBytes);
    auto err = mv->alloc();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain frame MV side data buffer.\n"));
        return err;
    }
    auto sad = std::make_unique<CUMemBuf>(m_analysis.sadBytes);
    err = sad->alloc();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain frame SAD side data buffer.\n"));
        return err;
    }

    if (frameCopyEvent() != nullptr) {
        err = err_to_rgy(cudaStreamWaitEvent(stream, frameCopyEvent(), 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain frame copy event for side data: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (m_analysis.event() != nullptr) {
        err = err_to_rgy(cudaStreamWaitEvent(stream, m_analysis.event(), 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain analysis event for side data: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    err = err_to_rgy(cudaMemcpyAsync(mv->ptr, m_analysis.mv->ptr, m_analysis.mvBytes, cudaMemcpyDeviceToDevice, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain MV side data buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    err = err_to_rgy(cudaMemcpyAsync(sad->ptr, m_analysis.sad->ptr, m_analysis.sadBytes, cudaMemcpyDeviceToDevice, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain SAD side data buffer: %s.\n"), get_err_mes(err));
        return err;
    }

    RGYCudaEvent sideDataEvent;
    err = degrainRecordEvent(stream, &sideDataEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain side data event: %s.\n"), get_err_mes(err));
        return err;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    const uint32_t flags = degrainAnalyzeFlags(prm, useAnalysisLumaCache() || m_lastAnalysisUsedSearchLuma, m_lastAnalysisIncludedChroma);
    auto frameData = std::make_shared<RGYFrameDataDegrain>(
        rgy_degrain_make_frame_meta_header(m_analysis.layout, flags),
        std::move(mv),
        std::move(sad),
        sideDataEvent,
        currentFrame,
        sourceFrame->inputFrameId,
        sourceFrame->timestamp,
        sourceFrame->duration,
        m_analysis.lastAvailabilityDisableRefs);
    rgy_degrain_erase_frame_data(outputFrame->dataList);
    outputFrame->dataList.push_back(frameData);
    if (event) {
        *event = sideDataEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::prepareAnalysisState(const RGYFilterDegrainFrameSet &frames, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    const int requiredDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, m_analysis.layout.temporalDirections / 2));
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    if (!frames.cur || frames.cur->ptr[0] == nullptr || !m_analysis.mv || !m_analysis.sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis buffers are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    for (int delta = 1; delta <= requiredDelta; delta++) {
        if (!frames.backwardRef(delta) || !frames.forwardRef(delta)
            || frames.backwardRef(delta)->ptr[0] == nullptr || frames.forwardRef(delta)->ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain analysis reference frames for delta=%d are not ready.\n"), delta);
            return RGY_ERR_INVALID_CALL;
        }
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid degrain parameter type in analysis.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto analysisFrames = frames;
    const auto attachedCur = degrainAttachedSearchLumaFrame(frames.cur);
    bool hasAttachedRefs = (attachedCur != nullptr);
    for (int delta = 1; delta <= requiredDelta && hasAttachedRefs; delta++) {
        const auto attachedBackward = degrainAttachedSearchLumaFrame(frames.backwardRef(delta));
        const auto attachedForward = degrainAttachedSearchLumaFrame(frames.forwardRef(delta));
        hasAttachedRefs &= (attachedBackward != nullptr && attachedForward != nullptr);
        if (hasAttachedRefs) {
            analysisFrames.backward[delta - 1] = attachedBackward;
            analysisFrames.forward[delta - 1] = attachedForward;
        }
    }
    if (hasAttachedRefs) {
        analysisFrames.cur = attachedCur;
        m_lastAnalysisUsedSearchLuma = true;
    }
    logLocalAnalysis(_T("local"), analysisFrames);

    const auto planeCur = getPlane(analysisFrames.cur, RGY_PLANE_Y);
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refPlanes;
    refPlanes.fill(planeCur);
    for (int delta = 1; delta <= requiredDelta; delta++) {
        refPlanes[rgy_degrain_ref_index(delta, false)] = getPlane(analysisFrames.backwardRef(delta), RGY_PLANE_Y);
        refPlanes[rgy_degrain_ref_index(delta, true)] = getPlane(analysisFrames.forwardRef(delta), RGY_PLANE_Y);
    }
    if (planeCur.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis requires valid luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
        if (refPlanes[dir].ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain analysis reference plane dir=%d is invalid.\n"), dir);
            return RGY_ERR_INVALID_CALL;
        }
    }
    const int pitchY = planeCur.pitch[0];
    for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
        if (refPlanes[dir].pitch[0] != pitchY) {
            AddMessage(RGY_LOG_ERROR,
                _T("degrain analysis luma pitch mismatch: cur=%d, refdir%d=%d.\n"),
                pitchY, dir, refPlanes[dir].pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    const auto chromaPlanes = degrainMakeAnalyzeChromaPlanes(
        analysisFrames,
        planeCur,
        refPlanes,
        m_analysis.layout.temporalDirections,
        requiredDelta,
        prm->degrain.chroma,
        m_lastAnalysisUsedSearchLuma);
    m_lastAnalysisIncludedChroma = chromaPlanes.enable != 0;
    const int pitchUV = chromaPlanes.curU.pitch[0];
    if (chromaPlanes.enable) {
        if (chromaPlanes.curV.pitch[0] != pitchUV) {
            AddMessage(RGY_LOG_ERROR,
                _T("degrain analysis chroma pitch mismatch: curU=%d, curV=%d.\n"),
                pitchUV, chromaPlanes.curV.pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
        for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
            if (chromaPlanes.refU[dir].pitch[0] != pitchUV || chromaPlanes.refV[dir].pitch[0] != pitchUV) {
                AddMessage(RGY_LOG_ERROR,
                    _T("degrain analysis chroma pitch mismatch: curUV=%d, refdir%d U=%d, V=%d.\n"),
                    pitchUV, dir, chromaPlanes.refU[dir].pitch[0], chromaPlanes.refV[dir].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }

    std::vector<RGYCudaEvent> analysisWaitEvents = wait_events;
    if (useAnalysisLumaCache()) {
        for (const auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            if (analysisLumaEvent() != nullptr) {
                analysisWaitEvents.push_back(analysisLumaEvent);
            }
        }
    }
    const auto motionSearchErr = prepareAnalysisStateMotionSearch(planeCur, refPlanes, stream, analysisWaitEvents);
    if (motionSearchErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search analysis failed: %s.\n"), get_err_mes(motionSearchErr));
        return motionSearchErr;
    }
    logAnalysisSamples(_T("local"), frames.cur, stream);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::prepareAnalysisStateMotionSearch(const RGYFrameInfo &planeCur, const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> &refPlanes,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        return RGY_ERR_UNSUPPORTED;
    }
    auto &ws = m_analysis.motionSearchWorkspace;
    if (!ws.level0.vectors || !ws.level0.vectorsPrev || !ws.level0.vectorsFinal || !ws.level0.sads
        || !ws.level1.vectors || !ws.level1.vectorsPrev || !ws.level1.vectorsFinal || !ws.level1.sads
        || !ws.frameAverageMV || !m_analysis.mv || !m_analysis.sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search workspace is not ready.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto spatialRefineCount = [&](const int level) {
        if (prm->degrain.mvSpatialRefine >= 0) {
            return prm->degrain.mvSpatialRefine;
        }
        const int innerLevel = (prm->degrain.levels > 1) ? 1 : 0;
        return (level == innerLevel) ? 1 : 0;
    };
    constexpr int vectorSentinelCount = 2;
    auto copyMotionSearchVectors = [&](CUMemBuf *src, const size_t srcVectorOffset, CUMemBuf *dst, const size_t dstVectorOffset,
        const int vectorCount, const std::vector<RGYCudaEvent> &waitEvents, RGYCudaEvent *copyEvent, const TCHAR *stage) {
        auto err = degrainWaitEvents(stream, waitEvents);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain motion search %s vector copy dependency: %s.\n"), stage, get_err_mes(err));
            return err;
        }
        const auto bytes = (size_t)vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES;
        err = err_to_rgy(cudaMemcpyAsync(
            reinterpret_cast<uint8_t *>(dst->ptr) + dstVectorOffset * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES,
            reinterpret_cast<const uint8_t *>(src->ptr) + srcVectorOffset * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain motion search %s vectors: %s.\n"), stage, get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, copyEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record degrain motion search %s vector copy event: %s.\n"), stage, get_err_mes(err));
        }
        return err;
    };

    using ProfileClock = std::chrono::steady_clock;
    const bool profileEnabled = degrainMotionSearchProfileEnabled();
    const auto profileTotalStart = profileEnabled ? ProfileClock::now() : ProfileClock::time_point();
    double profileDownsampleMs = 0.0;
    double profileInitConstVecMs = 0.0;
    double profileLevel1SeedMs = 0.0;
    double profileLevel1SearchMs = 0.0;
    double profileLevel1ExportSadMs = 0.0;
    double profileInterpolateMs = 0.0;
    double profileLevel0SearchMs = 0.0;
    double profileLevel0ExportSadMs = 0.0;
    const auto profileNow = [&]() {
        return profileEnabled ? ProfileClock::now() : ProfileClock::time_point();
    };
    const auto profileElapsedMs = [](const ProfileClock::time_point &start, const ProfileClock::time_point &end) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    };
    const auto profileFinishStep = [&](const TCHAR *stepName, double &totalMs, const ProfileClock::time_point &start, const int dir) {
        if (!profileEnabled) {
            return RGY_ERR_NONE;
        }
        auto err = err_to_rgy(cudaStreamSynchronize(stream));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to finish degrain motion search profile step %s: %s.\n"), stepName, get_err_mes(err));
            return err;
        }
        const auto elapsedMs = profileElapsedMs(start, ProfileClock::now());
        totalMs += elapsedMs;
        if (dir >= 0) {
            AddMessage(RGY_LOG_DEBUG, _T("degrain motion search profile refdir=%d %s: %.3f ms.\n"), dir, stepName, elapsedMs);
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("degrain motion search profile %s: %.3f ms.\n"), stepName, elapsedMs);
        }
        return RGY_ERR_NONE;
    };

    const int refs = m_analysis.layout.temporalDirections;
    const int level1FrameCount = degrainLevel1FrameCount(refs);
    for (int i = 0; i < level1FrameCount; i++) {
        if (!m_analysis.lumaLevel1[i]) {
            AddMessage(RGY_LOG_ERROR, _T("degrain motion search level1 luma workspace is not ready.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    const auto motionSearchConfig = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layout, 0, 8);
    auto motionSearchConfigLevel1 = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layoutLevel1, 1, 4);
    motionSearchConfigLevel1.width = m_analysis.lumaLevel1Width;
    motionSearchConfigLevel1.height = m_analysis.lumaLevel1Height;
    const auto pixelBytes = motionSearchConfig.pixelBytes;

    RGY_ERR err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain motion search input dependency: %s.\n"), get_err_mes(err));
        return err;
    }

    RGYCudaEvent frameAverageMVEvent;
    {
        err = err_to_rgy(cudaMemsetAsync(ws.frameAverageMV->ptr, 0, ws.frameAverageMVBytes, stream));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to clear degrain motion search frameAverageMV buffer: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &frameAverageMVEvent);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_LEVEL1_LUMA_FRAMES> level0Planes = {};
    level0Planes[0] = planeCur;
    for (int dir = 0; dir < refs; dir++) {
        level0Planes[dir + 1] = refPlanes[dir];
    }
    std::vector<RGYCudaEvent> downsampleEvents(level1FrameCount);
    for (int i = 0; i < level1FrameCount; i++) {
        const auto profileStepStart = profileNow();
        err = launchNVEncDegrainDownsampleLuma2x(
            level0Planes[i],
            *m_analysis.lumaLevel1[i],
            m_analysis.lumaLevel1Pitch,
            m_analysis.lumaLevel1Width,
            m_analysis.lumaLevel1Height,
            stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to downsample degrain motion search level1 luma: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &downsampleEvents[i]);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = profileFinishStep(_T("downsample"), profileDownsampleMs, profileStepStart, -1);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    const auto blockCount0 = (int)m_analysis.layout.blockCount();
    const auto blockCount1 = (int)m_analysis.layoutLevel1.blockCount();
    const int planeStride0 = 2 + blockCount0;
    const int planeStride1 = 2 + blockCount1;
    const auto levelPlaneBase = [](const int dir, const int planeStride) { return dir * planeStride; };
    const auto blockPlaneBase = [](const int dir, const int blockCount) { return dir * blockCount; };

    RGYCudaEvent initLevel1Event;
    auto profileStepStart = profileNow();
    err = degrainWaitEvents(stream, { frameAverageMVEvent });
    if (err == RGY_ERR_NONE) {
        err = launchNVEncDegrainMotionSearchSeedAnchorVectors(
            *ws.level1.vectors,
            *ws.frameAverageMV,
            0,
            planeStride1,
            refs,
            prm->degrain.pel,
            stream);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize degrain motion search level1 const vectors: %s.\n"), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, &initLevel1Event);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = profileFinishStep(_T("init const vec level1"), profileInitConstVecMs, profileStepStart, -1);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    RGYCudaEvent initLevel0Event;
    profileStepStart = profileNow();
    err = degrainWaitEvents(stream, { frameAverageMVEvent });
    if (err == RGY_ERR_NONE) {
        err = launchNVEncDegrainMotionSearchSeedAnchorVectors(
            *ws.level0.vectors,
            *ws.frameAverageMV,
            0,
            planeStride0,
            refs,
            prm->degrain.pel,
            stream);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize degrain motion search level0 const vectors: %s.\n"), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, &initLevel0Event);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = profileFinishStep(_T("init const vec level0"), profileInitConstVecMs, profileStepStart, -1);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    RGYCudaEvent previousEvent;
    for (int dir = 0; dir < refs; dir++) {
        const int planeBase1 = levelPlaneBase(dir, planeStride1);
        const int blockBase1 = blockPlaneBase(dir, blockCount1);

        std::vector<RGYCudaEvent> seedLevel1Wait = { initLevel1Event };
        if (previousEvent() != nullptr) {
            seedLevel1Wait.push_back(previousEvent);
        }
        RGYCudaEvent seedLevel1Event;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, seedLevel1Wait);
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchSeedZeroVectors(
                *ws.level1.vectors,
                *ws.level1.vectorsPrev,
                *ws.level1.sads,
                planeBase1,
                blockBase1,
                blockCount1,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to seed degrain motion search level1 vectors: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &seedLevel1Event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = profileFinishStep(_T("level1 seed"), profileLevel1SeedMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYCudaEvent searchLevel1Event;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, { seedLevel1Event, downsampleEvents[0], downsampleEvents[dir + 1] });
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchSearchParallel(
                reinterpret_cast<const uint8_t *>(m_analysis.lumaLevel1[0]->ptr),
                reinterpret_cast<const uint8_t *>(m_analysis.lumaLevel1[dir + 1]->ptr),
                *ws.level1.vectors,
                m_analysis.lumaLevel1Pitch,
                m_analysis.lumaLevel1Width,
                m_analysis.lumaLevel1Height,
                planeBase1,
                blockCount1,
                m_analysis.layoutLevel1,
                pixelBytes,
                motionSearchConfigLevel1.pel,
                motionSearchConfigLevel1.subpelInterp,
                motionSearchConfigLevel1.pad,
                motionSearchConfigLevel1.motionCostScale,
                motionSearchConfigLevel1.lowSadWeightScale,
                motionSearchConfigLevel1.zeroCandidateCostScale,
                motionSearchConfigLevel1.frameAverageCandidateCostScale,
                motionSearchConfigLevel1.newCandidateCostScale,
                motionSearchConfigLevel1.level,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run degrain motion search level1 search stub: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &searchLevel1Event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        RGYCudaEvent level1VectorReadyEvent = searchLevel1Event;
        const int level1SpatialRefineCount = spatialRefineCount(1);
        if (level1SpatialRefineCount <= 0) {
            RGYCudaEvent copyEvent;
            err = copyMotionSearchVectors(ws.level1.vectors.get(), (size_t)planeBase1 + vectorSentinelCount,
                ws.level1.vectorsFinal.get(), (size_t)blockBase1, blockCount1, { level1VectorReadyEvent },
                &copyEvent, _T("level1 current-to-final"));
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level1VectorReadyEvent = copyEvent;
        }
        for (int refine = 0; refine < level1SpatialRefineCount; refine++) {
            RGYCudaEvent refineEvent;
            err = degrainWaitEvents(stream, { level1VectorReadyEvent });
            if (err == RGY_ERR_NONE) {
                err = launchNVEncDegrainMotionSearchSpatialRefine(
                    reinterpret_cast<const uint8_t *>(m_analysis.lumaLevel1[0]->ptr),
                    reinterpret_cast<const uint8_t *>(m_analysis.lumaLevel1[dir + 1]->ptr),
                    *ws.level1.vectors,
                    *ws.level1.vectorsPrev,
                    *ws.level1.vectorsFinal,
                    m_analysis.lumaLevel1Pitch,
                    m_analysis.lumaLevel1Width,
                    m_analysis.lumaLevel1Height,
                    planeBase1,
                    blockBase1,
                    blockCount1,
                    m_analysis.layoutLevel1,
                    pixelBytes,
                    motionSearchConfigLevel1.pel,
                    motionSearchConfigLevel1.subpelInterp,
                    motionSearchConfigLevel1.pad,
                    motionSearchConfigLevel1.motionCostScale,
                    motionSearchConfigLevel1.lowSadWeightScale,
                    motionSearchConfigLevel1.newCandidateCostScale,
                    stream);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to refine degrain motion search level1 spatial predictors: %s.\n"), get_err_mes(err));
                return err;
            }
            err = degrainRecordEvent(stream, &refineEvent);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level1VectorReadyEvent = refineEvent;
            if (refine + 1 < level1SpatialRefineCount) {
                RGYCudaEvent copyEvent;
                err = copyMotionSearchVectors(ws.level1.vectorsFinal.get(), (size_t)blockBase1,
                    ws.level1.vectors.get(), (size_t)planeBase1 + vectorSentinelCount, blockCount1, { level1VectorReadyEvent },
                    &copyEvent, _T("level1 final-to-current"));
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                level1VectorReadyEvent = copyEvent;
            }
        }
        err = profileFinishStep(_T("level1 search"), profileLevel1SearchMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYCudaEvent exportLevel1Event;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, { level1VectorReadyEvent });
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchExportSad(
                *ws.level1.vectorsFinal,
                *ws.level1.sads,
                nullptr,
                nullptr,
                blockBase1,
                blockBase1,
                blockCount1,
                0,
                dir,
                refs,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to export degrain motion search level1 SAD: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &exportLevel1Event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = profileFinishStep(_T("level1 export_sad"), profileLevel1ExportSadMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        const int planeBase0 = levelPlaneBase(dir, planeStride0);
        const int blockBase0 = blockPlaneBase(dir, blockCount0);
        RGYCudaEvent interpolateEvent;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, { exportLevel1Event, initLevel0Event });
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchExpandCoarseVectors(
                *ws.level1.vectorsFinal,
                *ws.level0.vectors,
                *ws.level0.vectorsPrev,
                *ws.level0.sads,
                blockBase1,
                planeBase0,
                blockBase0,
                blockCount1,
                blockCount0,
                m_analysis.layoutLevel1.blocksX,
                m_analysis.layoutLevel1.blocksY,
                m_analysis.layout.blocksX,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to interpolate degrain motion search predictor: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &interpolateEvent);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = profileFinishStep(_T("interpolate"), profileInterpolateMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYCudaEvent searchLevel0Event;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, { interpolateEvent });
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchSearchParallel(
                planeCur.ptr[0],
                refPlanes[dir].ptr[0],
                *ws.level0.vectors,
                planeCur.pitch[0],
                planeCur.width,
                planeCur.height,
                planeBase0,
                blockCount0,
                m_analysis.layout,
                pixelBytes,
                motionSearchConfig.pel,
                motionSearchConfig.subpelInterp,
                motionSearchConfig.pad,
                motionSearchConfig.motionCostScale,
                motionSearchConfig.lowSadWeightScale,
                motionSearchConfig.zeroCandidateCostScale,
                motionSearchConfig.frameAverageCandidateCostScale,
                motionSearchConfig.newCandidateCostScale,
                motionSearchConfig.level,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run degrain motion search level0 search stub: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &searchLevel0Event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        RGYCudaEvent level0VectorReadyEvent = searchLevel0Event;
        const int level0SpatialRefineCount = spatialRefineCount(0);
        if (level0SpatialRefineCount <= 0) {
            RGYCudaEvent copyEvent;
            err = copyMotionSearchVectors(ws.level0.vectors.get(), (size_t)planeBase0 + vectorSentinelCount,
                ws.level0.vectorsFinal.get(), (size_t)blockBase0, blockCount0, { level0VectorReadyEvent },
                &copyEvent, _T("level0 current-to-final"));
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level0VectorReadyEvent = copyEvent;
        }
        for (int refine = 0; refine < level0SpatialRefineCount; refine++) {
            RGYCudaEvent refineEvent;
            err = degrainWaitEvents(stream, { level0VectorReadyEvent });
            if (err == RGY_ERR_NONE) {
                err = launchNVEncDegrainMotionSearchSpatialRefine(
                    planeCur.ptr[0],
                    refPlanes[dir].ptr[0],
                    *ws.level0.vectors,
                    *ws.level0.vectorsPrev,
                    *ws.level0.vectorsFinal,
                    planeCur.pitch[0],
                    planeCur.width,
                    planeCur.height,
                    planeBase0,
                    blockBase0,
                    blockCount0,
                    m_analysis.layout,
                    pixelBytes,
                    motionSearchConfig.pel,
                    motionSearchConfig.subpelInterp,
                    motionSearchConfig.pad,
                    motionSearchConfig.motionCostScale,
                    motionSearchConfig.lowSadWeightScale,
                    motionSearchConfig.newCandidateCostScale,
                    stream);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to refine degrain motion search level0 spatial predictors: %s.\n"), get_err_mes(err));
                return err;
            }
            err = degrainRecordEvent(stream, &refineEvent);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level0VectorReadyEvent = refineEvent;
            if (refine + 1 < level0SpatialRefineCount) {
                RGYCudaEvent copyEvent;
                err = copyMotionSearchVectors(ws.level0.vectorsFinal.get(), (size_t)blockBase0,
                    ws.level0.vectors.get(), (size_t)planeBase0 + vectorSentinelCount, blockCount0, { level0VectorReadyEvent },
                    &copyEvent, _T("level0 final-to-current"));
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                level0VectorReadyEvent = copyEvent;
            }
        }
        err = profileFinishStep(_T("level0 search"), profileLevel0SearchMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYCudaEvent exportLevel0Event;
        profileStepStart = profileNow();
        err = degrainWaitEvents(stream, { level0VectorReadyEvent });
        if (err == RGY_ERR_NONE) {
            err = launchNVEncDegrainMotionSearchExportSad(
                *ws.level0.vectorsFinal,
                *ws.level0.sads,
                m_analysis.mv.get(),
                m_analysis.sad.get(),
                blockBase0,
                blockBase0,
                blockCount0,
                0,
                dir,
                refs,
                stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to export degrain motion search level0 SAD: %s.\n"), get_err_mes(err));
            return err;
        }
        err = degrainRecordEvent(stream, &exportLevel0Event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = profileFinishStep(_T("level0 export_sad"), profileLevel0ExportSadMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        previousEvent = exportLevel0Event;
    }

    if (previousEvent() == nullptr) {
        return RGY_ERR_UNSUPPORTED;
    }
    m_analysis.event = previousEvent;
    m_lastAnalysisIncludedChroma = false;
    if (profileEnabled) {
        const auto profileTotalMs = profileElapsedMs(profileTotalStart, ProfileClock::now());
        AddMessage(RGY_LOG_INFO,
            _T("degrain motion search profile summary: downsample=%.3f ms, seed_anchor_vectors=%.3f ms, level1_seed=%.3f ms, level1_search=%.3f ms, level1_export_sad=%.3f ms, expand_coarse_vectors=%.3f ms, level0_search=%.3f ms, level0_export_sad=%.3f ms, total=%.3f ms.\n"),
            profileDownsampleMs,
            profileInitConstVecMs,
            profileLevel1SeedMs,
            profileLevel1SearchMs,
            profileLevel1ExportSadMs,
            profileInterpolateMs,
            profileLevel0SearchMs,
            profileLevel0ExportSadMs,
            profileTotalMs);
    }
    AddMessage(RGY_LOG_DEBUG, _T("degrain motion search analyze path was used.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::runAnalyzeMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    clearFrameAnalysisData();
    auto err = prepareAnalysisState(frames.analysis, stream, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_analysis.lastAvailabilityDisableRefs = degrainReferenceAvailability(frames.analysis);
    m_analysis.lastFrameIndex = currentFrame;
    m_analysis.lastInputFrameId = (frames.render.cur) ? frames.render.cur->inputFrameId : -1;
    m_analysis.lastTimestamp = (frames.render.cur) ? frames.render.cur->timestamp : 0;
    m_analysis.lastDuration = (frames.render.cur) ? frames.render.cur->duration : 0;
    RGYCudaEvent copyEvent;
    err = emitSourceFrame(frames.render.cur, ppOutputFrames, pOutputFrameNum, stream, {}, &copyEvent);
    if (err != RGY_ERR_NONE || *pOutputFrameNum <= 0 || ppOutputFrames[0] == nullptr) {
        if (event) {
            *event = copyEvent;
        }
        return err;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (prm && !prm->attachAnalysisData) {
        if (event) {
            *event = copyEvent;
        }
        return RGY_ERR_NONE;
    }
    return attachAnalysisData(frames.render.cur, ppOutputFrames[0], currentFrame, stream, copyEvent, event);
}

RGY_ERR NVEncFilterDegrain::runDebugMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, stream)) {
        auto err = prepareAnalysisState(frames.analysis, stream, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return emitDebugFrame(frames.render, mode, ppOutputFrames, pOutputFrameNum, stream, event);
}

RGY_ERR NVEncFilterDegrain::runCompensateMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, stream)) {
        auto err = prepareAnalysisState(frames.analysis, stream, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeRefs(std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param), frames.analysis, stream, &disableRefs);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return emitCompensateFrame(frames.render, mode, disableRefs, ppOutputFrames, pOutputFrameNum, stream, event);
}

RGY_ERR NVEncFilterDegrain::runDegrainMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pendingReadbackStable = [](const PendingSceneChange &pending) {
        return !pending.mapSubmitted || pending.mapEvent() != nullptr;
    };
    bool pendingOutputEmitted = false;
    if (!m_pendingSceneChange.empty() && !pendingReadbackStable(*m_pendingSceneChange.front())) {
        auto pendingOutput = std::move(m_pendingSceneChange.front());
        m_pendingSceneChange.pop_front();
        auto err = emitResolvedSceneChangeFrame(*pendingOutput, ppOutputFrames, pOutputFrameNum, stream, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        pendingOutputEmitted = true;
    }

    if (m_pendingSceneChange.size() > SCENE_CHANGE_PIPELINE_DEPTH) {
        if (pendingOutputEmitted) {
            AddMessage(RGY_LOG_ERROR, _T("degrain scene-change pipeline has more pending frames than expected.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto err = resolvePendingSceneChangeFrame(ppOutputFrames, pOutputFrameNum, stream, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (*pOutputFrameNum > 0) {
            pendingOutputEmitted = true;
        }
    }

    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, stream)) {
        auto err = prepareAnalysisState(frames.analysis, stream, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    const auto currentFrameAnalysisData = m_frameAnalysisData;
    const auto currentBoundAnalyzeResult = m_boundAnalyzeResult;
    const auto currentFrameAnalysisLayout = m_frameAnalysisLayout;
    const bool canDeferSceneChange = !m_boundAnalyzeResult.valid() || m_frameAnalysisData;
    if (canDeferSceneChange && !pendingOutputEmitted && m_pendingSceneChange.size() >= SCENE_CHANGE_PIPELINE_DEPTH) {
        auto err = resolvePendingSceneChangeFrame(ppOutputFrames, pOutputFrameNum, stream, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (*pOutputFrameNum > 0) {
            pendingOutputEmitted = true;
        }
        m_frameAnalysisData = currentFrameAnalysisData;
        m_boundAnalyzeResult = currentBoundAnalyzeResult;
        m_frameAnalysisLayout = currentFrameAnalysisLayout;
    }

    if (!canDeferSceneChange) {
        if (pendingOutputEmitted) {
            AddMessage(RGY_LOG_ERROR, _T("degrain scene-change pipeline cannot emit both pending and current frame in immediate mode.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        PendingSceneChange pending;
        auto err = submitSceneChangeReadback(prm, frames, stream, &pending);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        RGYDegrainRefDisableArray disableRefs;
        err = resolveSceneChangeReadback(pending, stream, &disableRefs);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        logApplyTrace(prm, frames, disableRefs, stream);
        return emitDegrainFrame(frames.render, disableRefs, ppOutputFrames, pOutputFrameNum, stream, event);
    }

    auto pending = std::make_unique<PendingSceneChange>();
    auto err = submitSceneChangeReadback(prm, frames, stream, pending.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_pendingSceneChange.push_back(std::move(pending));
    if (pendingOutputEmitted) {
        return RGY_ERR_NONE;
    }
    if (m_pendingSceneChange.size() > SCENE_CHANGE_PIPELINE_DEPTH) {
        return resolvePendingSceneChangeFrame(ppOutputFrames, pOutputFrameNum, stream, event);
    }

    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::submitSceneChangeReadback(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
    cudaStream_t stream, PendingSceneChange *pending) {
    if (!pending) {
        return RGY_ERR_INVALID_PARAM;
    }
    *pending = PendingSceneChange();
    pending->prm = prm;
    pending->frames = frames;
    pending->frameAnalysisData = m_frameAnalysisData;
    pending->boundAnalyzeResult = m_boundAnalyzeResult;
    pending->frameAnalysisLayout = m_frameAnalysisLayout;
    pending->layout = analysisLayout();

    const auto availabilityDisableRefs = analysisAvailabilityDisableRefs(frames.analysis);
    auto useFlagDisableRefs = RGYDegrainRefDisableArray();
    useFlagDisableRefs.fill(false);
    auto disableRefs = availabilityDisableRefs;
    if (prm) {
        if (prm->degrain.useFlag == 1) {
            for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
                const int refIndex = rgy_degrain_ref_index(delta, true);
                useFlagDisableRefs[refIndex] = true;
                disableRefs[refIndex] = true;
            }
        } else if (prm->degrain.useFlag == 2) {
            for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
                const int refIndex = rgy_degrain_ref_index(delta, false);
                useFlagDisableRefs[refIndex] = true;
                disableRefs[refIndex] = true;
            }
        }
    }

    auto *sad = analysisSAD();
    const bool includeChromaSad = analysisSADIncludesChroma(prm);
    const uint32_t scaledThSad = (prm) ? rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, includeChromaSad) : 0;
    const uint32_t scaledThSCD1 = (prm) ? rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thscd1, includeChromaSad) : 0;
    const uint64_t scaledThSCD2 = (prm) ? rgy_degrain_scale_scene_change_block_threshold(pending->layout.blockCount(), prm->degrain.thscd2) : 0;
    pending->availabilityDisableRefs = availabilityDisableRefs;
    pending->useFlagDisableRefs = useFlagDisableRefs;
    pending->disableRefs = disableRefs;
    pending->scaledThSad = scaledThSad;
    pending->scaledThSCD1 = scaledThSCD1;
    pending->scaledThSCD2 = scaledThSCD2;
    pending->sad = sad;
    pending->disableMask = degrainDisableMask(disableRefs, pending->layout.temporalDirections);
    if (!prm || !sad || pending->layout.blockCount() == 0) {
        return RGY_ERR_NONE;
    }
    bool allDirectionsDisabled = true;
    for (int refDirection = 0; refDirection < std::min(pending->layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        allDirectionsDisabled &= pending->disableRefs[refDirection];
    }
    if (allDirectionsDisabled) {
        return RGY_ERR_NONE;
    }

    const auto countBytes = sizeof(uint32_t) * RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS;
    if (!m_sceneChangeCounts || m_sceneChangeCounts->nSize != countBytes) {
        m_sceneChangeCounts = std::make_unique<CUMemBuf>(countBytes);
        auto err = m_sceneChangeCounts->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain scene change count buffer: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (!m_sceneChangeDisableMask || m_sceneChangeDisableMask->nSize != sizeof(uint32_t)) {
        m_sceneChangeDisableMask = std::make_unique<CUMemBuf>(sizeof(uint32_t));
        auto err = m_sceneChangeDisableMask->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain scene change mask buffer: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (analysisEvent()() != nullptr) {
        auto err = err_to_rgy(cudaStreamWaitEvent(stream, analysisEvent()(), 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain analysis event for scene change detection: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    auto err = launchNVEncDegrainSceneChangeMask(*m_sceneChangeCounts, *m_sceneChangeDisableMask, *sad,
        pending->layout, pending->scaledThSCD1, pending->disableMask, pending->scaledThSCD2, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build degrain scene change mask: %s.\n"), get_err_mes(err));
        return err;
    }
    err = err_to_rgy(cudaMemcpyAsync(pending->sceneChangeBlockCounts.data(), m_sceneChangeCounts->ptr,
        sizeof(uint32_t) * pending->sceneChangeBlockCounts.size(), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain scene change counts: %s.\n"), get_err_mes(err));
        return err;
    }
    err = err_to_rgy(cudaMemcpyAsync(&pending->disableMask, m_sceneChangeDisableMask->ptr,
        sizeof(pending->disableMask), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain scene change mask: %s.\n"), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, &pending->mapEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain scene change readback event: %s.\n"), get_err_mes(err));
        return err;
    }
    pending->mapSubmitted = true;
    return RGY_ERR_NONE;
}

std::vector<RGYDegrainSAD> *NVEncFilterDegrain::acquireSceneChangeReadbackSAD(const size_t count) {
    auto &buf = m_sceneChangeReadbackSAD[m_sceneChangeReadbackSADIndex];
    m_sceneChangeReadbackSADIndex = (m_sceneChangeReadbackSADIndex + 1) % (int)m_sceneChangeReadbackSAD.size();
    try {
        buf.resize(count);
    } catch (...) {
        return nullptr;
    }
    return &buf;
}

RGY_ERR NVEncFilterDegrain::resolveSceneChangeReadback(PendingSceneChange &pending, cudaStream_t, RGYDegrainRefDisableArray *disableRefs) {
    if (!disableRefs) {
        return RGY_ERR_INVALID_PARAM;
    }
    *disableRefs = pending.disableRefs;
    if (!pending.prm || !pending.sad || pending.layout.blockCount() == 0) {
        logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
            *disableRefs, nullptr, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
        return RGY_ERR_NONE;
    }
    if (!pending.mapSubmitted) {
        logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
            *disableRefs, nullptr, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
        return RGY_ERR_NONE;
    }
    if (pending.mapEvent() != nullptr) {
        auto err = err_to_rgy(cudaEventSynchronize(pending.mapEvent()));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain SAD readback for scene change detection: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    std::array<size_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> sceneChangeBlockCounts = {};
    for (int refDirection = 0; refDirection < RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS; refDirection++) {
        sceneChangeBlockCounts[refDirection] = pending.sceneChangeBlockCounts[refDirection];
    }
    pending.mapSubmitted = false;

    for (int refDirection = 0; refDirection < std::min(pending.layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        pending.disableRefs[refDirection] = ((pending.disableMask >> refDirection) & 1u) != 0u;
    }
    *disableRefs = pending.disableRefs;
    logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
        *disableRefs, &sceneChangeBlockCounts, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::resolvePendingSceneChangeFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    if (m_pendingSceneChange.empty()) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto pending = std::move(m_pendingSceneChange.front());
    m_pendingSceneChange.pop_front();
    return emitResolvedSceneChangeFrame(*pending, ppOutputFrames, pOutputFrameNum, stream, event);
}

RGY_ERR NVEncFilterDegrain::emitResolvedSceneChangeFrame(PendingSceneChange &pending, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    applyPendingSceneChangeAnalysisContext(pending);
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeReadback(pending, stream, &disableRefs);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    logApplyTrace(pending.prm, pending.frames, disableRefs, stream);
    return emitDegrainFrame(pending.frames.render, disableRefs, ppOutputFrames, pOutputFrameNum, stream, event);
}

void NVEncFilterDegrain::applyPendingSceneChangeAnalysisContext(const PendingSceneChange &pending) {
    m_frameAnalysisData = pending.frameAnalysisData;
    m_boundAnalyzeResult = pending.boundAnalyzeResult;
    m_frameAnalysisLayout = pending.frameAnalysisLayout;
}

void NVEncFilterDegrain::clearPendingSceneChange() {
    for (auto &pending : m_pendingSceneChange) {
        if (pending && pending->mapSubmitted && pending->mapEvent() != nullptr) {
            cudaEventSynchronize(pending->mapEvent());
            pending->mapSubmitted = false;
        }
    }
    m_pendingSceneChange.clear();
}

RGY_ERR NVEncFilterDegrain::resolveSceneChangeRefs(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, cudaStream_t stream,
    RGYDegrainRefDisableArray *disableRefs) {
    PendingSceneChange pending;
    RGYFilterDegrainProcessFrameSet processFrames = {};
    processFrames.render = frames;
    processFrames.analysis = frames;
    auto err = submitSceneChangeReadback(prm, processFrames, stream, &pending);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return resolveSceneChangeReadback(pending, stream, disableRefs);
}

RGY_ERR NVEncFilterDegrain::resolveSceneChange(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, cudaStream_t stream,
    bool *disableBackward, bool *disableForward) {
    if (!disableBackward || !disableForward) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeRefs(prm, frames, stream, &disableRefs);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    *disableBackward = disableRefs[rgy_degrain_refdir_index(RGYDegrainRefDir::Backward)];
    *disableForward = disableRefs[rgy_degrain_refdir_index(RGYDegrainRefDir::Forward)];
    return RGY_ERR_NONE;
}

void NVEncFilterDegrain::loadDebugEnv() {
    m_debugEnv.applyTrace = degrainTraceEnvEnabled("NVENC_DEGRAIN_APPLY_TRACE");
    m_debugEnv.applyTraceBlock = degrainTraceEnvInt("NVENC_DEGRAIN_APPLY_TRACE_BLOCK", -1);
    m_debugEnv.forceDegrainCopy = degrainTraceEnvEnabled("NVENC_DEGRAIN_DEGRAIN_COPY");
    m_debugEnv.pixelTrace = degrainTraceEnvEnabled("NVENC_DEGRAIN_PIXEL_TRACE");
    m_debugEnv.pixelTraceX = degrainTraceEnvInt("NVENC_DEGRAIN_PIXEL_TRACE_X", 0);
    m_debugEnv.pixelTraceY = degrainTraceEnvInt("NVENC_DEGRAIN_PIXEL_TRACE_Y", 0);
    m_debugEnv.pixelTraceFrame = degrainTraceEnvInt("NVENC_DEGRAIN_PIXEL_TRACE_FRAME", -1);
}

RGY_ERR NVEncFilterDegrain::allocAnalysisBuffers(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
    m_analysis.mode = prm->degrain.mode;
    m_analysis.layout = rgy_degrain_make_block_layout(prm->frameOut, prm->degrain);
    m_analysis.layoutLevel1 = rgy_degrain_make_pyramid_block_layout(prm->frameOut, prm->degrain);
    for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
        analysisLumaEvent.reset();
    }
    m_analysis.analysisLumaEvent.reset();
    m_analysis.event.reset();

    if (!modeRequiresAnalysis(prm->degrain.mode)) {
        m_analysis.mv.reset();
        m_analysis.sad.reset();
        m_analysis.windowRampY.reset();
        m_analysis.windowRampC.reset();
        m_analysis.temporalMixPlanY.reset();
        m_analysis.temporalMixPlanC.reset();
        m_analysis.temporalMixPrior.reset();
        m_analysis.motionSearchWorkspace.reset();
        for (auto &luma : m_analysis.analysisLuma) {
            luma.reset();
        }
        for (auto &frame : m_analysis.analysisLumaFrame) {
            frame = RGYFrameInfo();
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        for (auto &luma : m_analysis.lumaLevel1) {
            luma.reset();
        }
        m_analysis.mvBytes = 0;
        m_analysis.sadBytes = 0;
        m_analysis.analysisLumaBytes = 0;
        m_analysis.lumaLevel1Bytes = 0;
        m_analysis.analysisLumaWidth = 0;
        m_analysis.analysisLumaHeight = 0;
        m_analysis.analysisLumaPitch = 0;
        m_analysis.analysisLumaGeneratedUntil = -1;
        m_analysis.lumaLevel1Width = 0;
        m_analysis.lumaLevel1Height = 0;
        m_analysis.lumaLevel1Pitch = 0;
        m_analysis.lastFrameIndex = -1;
        m_analysis.lastInputFrameId = -1;
        return RGY_ERR_NONE;
    }

    const bool binomial = (prm->degrain.binomial < 0)
        ? (prm->degrain.stage != VppDegrainStage::TR2)
        : (prm->degrain.binomial != 0);
    const auto temporalMixPrior = degrainBuildTemporalMixPriorTable(m_analysis.layout.temporalDirections, binomial);
    const auto temporalMixPriorBytes = temporalMixPrior.size() * sizeof(temporalMixPrior[0]);
    if (!m_analysis.temporalMixPrior || m_analysis.temporalMixPrior->nSize != temporalMixPriorBytes) {
        m_analysis.temporalMixPrior = std::make_unique<CUMemBuf>(temporalMixPriorBytes);
        auto err = m_analysis.temporalMixPrior->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain temporal mix prior table buffer.\n"));
            return err;
        }
    }
    auto err = err_to_rgy(cudaMemcpy(m_analysis.temporalMixPrior->ptr, temporalMixPrior.data(), temporalMixPriorBytes, cudaMemcpyHostToDevice));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain temporal mix prior table: %s.\n"), get_err_mes(err));
        return err;
    }

    m_analysis.mvBytes = rgy_degrain_mv_bytes(m_analysis.layout);
    m_analysis.sadBytes = rgy_degrain_sad_bytes(m_analysis.layout);
    if (!m_analysis.mv || m_analysis.mv->nSize != m_analysis.mvBytes) {
        m_analysis.mv = std::make_unique<CUMemBuf>(m_analysis.mvBytes);
        err = m_analysis.mv->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain MV buffer.\n"));
            return err;
        }
    }
    if (!m_analysis.sad || m_analysis.sad->nSize != m_analysis.sadBytes) {
        m_analysis.sad = std::make_unique<CUMemBuf>(m_analysis.sadBytes);
        err = m_analysis.sad->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain SAD buffer.\n"));
            return err;
        }
    }

    const auto motionSearchConfig = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layout, 0, 8);
    auto motionSearchConfigLevel1 = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layoutLevel1, 1, 4);
    motionSearchConfigLevel1.width = std::max(1, (prm->frameOut.width + 1) / 2);
    motionSearchConfigLevel1.height = std::max(1, (prm->frameOut.height + 1) / 2);
    auto &motionSearchWorkspace = m_analysis.motionSearchWorkspace;
    motionSearchWorkspace.buildOptionsLevel0 = makeDegrainMotionSearchBuildOptions(motionSearchConfig);
    motionSearchWorkspace.buildOptionsLevel1 = makeDegrainMotionSearchBuildOptions(motionSearchConfigLevel1);

    auto allocBuffer = [&](std::unique_ptr<CUMemBuf> &buf, size_t &currentBytes, const size_t requiredBytes, const TCHAR *name) {
        currentBytes = requiredBytes;
        if (requiredBytes == 0) {
            buf.reset();
            return RGY_ERR_NONE;
        }
        if (!buf || buf->nSize != requiredBytes) {
            buf = std::make_unique<CUMemBuf>(requiredBytes);
            const auto allocErr = buf->alloc();
            if (allocErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain %s buffer.\n"), name);
                return allocErr;
            }
        }
        return RGY_ERR_NONE;
    };
    auto allocLevelWorkspace = [&](RGYDegrainMotionSearchLevelWorkspace &levelWorkspace, const RGYDegrainBlockLayout &layout, const TCHAR *levelName) {
        const size_t planeCount = (size_t)layout.temporalDirections;
        const size_t blockCount = layout.blockCount();
        const size_t vectorCount = (2 + blockCount) * planeCount;
        const size_t finalVectorCount = blockCount * planeCount;
        const size_t sadCount = blockCount * planeCount;
        auto levelErr = allocBuffer(levelWorkspace.vectors, levelWorkspace.vectorsBytes,
            vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES, strsprintf(_T("motion search %s vectors workspace"), levelName).c_str());
        if (levelErr != RGY_ERR_NONE) return levelErr;
        levelErr = allocBuffer(levelWorkspace.vectorsPrev, levelWorkspace.vectorsPrevBytes,
            vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES, strsprintf(_T("motion search %s prev vectors workspace"), levelName).c_str());
        if (levelErr != RGY_ERR_NONE) return levelErr;
        levelErr = allocBuffer(levelWorkspace.vectorsFinal, levelWorkspace.vectorsFinalBytes,
            finalVectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES, strsprintf(_T("motion search %s final vectors workspace"), levelName).c_str());
        if (levelErr != RGY_ERR_NONE) return levelErr;
        return allocBuffer(levelWorkspace.sads, levelWorkspace.sadsBytes,
            sadCount * RGYDegrainMotionSearchWorkspace::SAD_BYTES, strsprintf(_T("motion search %s sads workspace"), levelName).c_str());
    };
    err = allocLevelWorkspace(motionSearchWorkspace.level0, m_analysis.layout, _T("level0"));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (prm->degrain.levels > 1) {
        err = allocLevelWorkspace(motionSearchWorkspace.level1, m_analysis.layoutLevel1, _T("level1"));
        if (err != RGY_ERR_NONE) {
            return err;
        }
    } else {
        motionSearchWorkspace.level1.reset();
    }
    const size_t motionSearchFrameAverageVectorCount = (size_t)m_analysis.layout.temporalDirections;
    err = allocBuffer(motionSearchWorkspace.frameAverageMV, motionSearchWorkspace.frameAverageMVBytes,
        motionSearchFrameAverageVectorCount * RGYDegrainMotionSearchWorkspace::FRAME_AVERAGE_MV_BYTES, _T("motion search frameAverageMV workspace"));
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelBytes = (bitdepth > 8) ? 2 : 1;
    if (degrainRequiresAnalysisLumaCache(prm->degrain)) {
        m_analysis.analysisLumaWidth = prm->frameOut.width;
        m_analysis.analysisLumaHeight = prm->frameOut.height;
        m_analysis.analysisLumaPitch = m_analysis.analysisLumaWidth * pixelBytes;
        m_analysis.analysisLumaBytes = (size_t)m_analysis.analysisLumaPitch * (size_t)m_analysis.analysisLumaHeight;
        const auto analysisCsp = degrainAnalysisLumaCsp(prm->frameOut);
        for (int i = 0; i < (int)m_analysis.analysisLuma.size(); i++) {
            auto &luma = m_analysis.analysisLuma[i];
            if (!luma || luma->nSize != m_analysis.analysisLumaBytes) {
                luma = std::make_unique<CUMemBuf>(m_analysis.analysisLumaBytes);
                err = luma->alloc();
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain analysis luma buffer.\n"));
                    return err;
                }
            }
            auto &frame = m_analysis.analysisLumaFrame[i];
            frame = RGYFrameInfo(m_analysis.analysisLumaWidth, m_analysis.analysisLumaHeight, analysisCsp, bitdepth, RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_GPU);
            frame.ptr[0] = reinterpret_cast<uint8_t *>(m_analysis.analysisLuma[i]->ptr);
            frame.pitch[0] = m_analysis.analysisLumaPitch;
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        m_analysis.analysisLumaGeneratedUntil = -1;
    } else {
        for (auto &luma : m_analysis.analysisLuma) {
            luma.reset();
        }
        for (auto &frame : m_analysis.analysisLumaFrame) {
            frame = RGYFrameInfo();
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        m_analysis.analysisLumaBytes = 0;
        m_analysis.analysisLumaWidth = 0;
        m_analysis.analysisLumaHeight = 0;
        m_analysis.analysisLumaPitch = 0;
        m_analysis.analysisLumaGeneratedUntil = -1;
    }
    m_analysis.lumaLevel1Width = std::max(1, (prm->frameOut.width + 1) / 2);
    m_analysis.lumaLevel1Height = std::max(1, (prm->frameOut.height + 1) / 2);
    m_analysis.lumaLevel1Pitch = m_analysis.lumaLevel1Width * pixelBytes;
    m_analysis.lumaLevel1Bytes = (size_t)m_analysis.lumaLevel1Pitch * (size_t)m_analysis.lumaLevel1Height;
    const int requiredLevel1Frames = degrainLevel1FrameCount(m_analysis.layout.temporalDirections);
    for (int i = 0; i < (int)m_analysis.lumaLevel1.size(); i++) {
        auto &luma = m_analysis.lumaLevel1[i];
        if (i >= requiredLevel1Frames) {
            luma.reset();
            continue;
        }
        if (!luma || luma->nSize != m_analysis.lumaLevel1Bytes) {
            luma = std::make_unique<CUMemBuf>(m_analysis.lumaLevel1Bytes);
            err = luma->alloc();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain level1 luma buffer.\n"));
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

const RGYFrameInfo *NVEncFilterDegrain::resolveAnalysisLumaSourceFrame(const int frameIndex) const {
    if (m_inputCount <= 0) {
        return nullptr;
    }
    const int clampedFrame = clamp(frameIndex, 0, m_inputCount - 1);
    return &m_cacheFrames[cacheIndex(clampedFrame)]->frame;
}

RGYFilterDegrainFrameSet NVEncFilterDegrain::resolveAnalysisFrameSet(int currentFrame) const {
    auto frames = resolveFrameSet(currentFrame);
    if (!useAnalysisLumaCache()) {
        return frames;
    }

    const auto analysisFrame = [this](const int frameIndex) -> const RGYFrameInfo * {
        const int slot = analysisCacheIndex(frameIndex);
        return (m_analysis.analysisLumaFrameNumbers[slot] == frameIndex) ? &m_analysis.analysisLumaFrame[slot] : nullptr;
    };
    frames.cur = analysisFrame(currentFrame);
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        const int backwardFrame = frames.backwardRefInRange(delta) ? (currentFrame + delta) : currentFrame;
        const int forwardFrame = frames.forwardRefInRange(delta) ? (currentFrame - delta) : currentFrame;
        frames.backward[delta - 1] = analysisFrame(backwardFrame);
        frames.forward[delta - 1] = analysisFrame(forwardFrame);
    }
    return frames;
}

RGY_ERR NVEncFilterDegrain::generateAnalysisLumaFrame(const int centerFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm || !useAnalysisLumaCache()) {
        return RGY_ERR_NONE;
    }
    if (centerFrame < 0 || centerFrame >= m_inputCount) {
        return RGY_ERR_NONE;
    }

    RGYFilterDegrainFrameSet srcFrames = {};
    srcFrames.backward[1] = resolveAnalysisLumaSourceFrame(centerFrame + 2);
    srcFrames.backward[0] = resolveAnalysisLumaSourceFrame(centerFrame + 1);
    srcFrames.cur = resolveAnalysisLumaSourceFrame(centerFrame);
    srcFrames.forward[0] = resolveAnalysisLumaSourceFrame(centerFrame - 1);
    srcFrames.forward[1] = resolveAnalysisLumaSourceFrame(centerFrame - 2);
    if (!srcFrames.backwardRef(2) || !srcFrames.backwardRef(1) || !srcFrames.cur || !srcFrames.forwardRef(1) || !srcFrames.forwardRef(2)) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis luma source frames are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    const auto planePrev2 = getPlane(srcFrames.forwardRef(2), RGY_PLANE_Y);
    const auto planePrev = getPlane(srcFrames.forwardRef(1), RGY_PLANE_Y);
    const auto planeCur = getPlane(srcFrames.cur, RGY_PLANE_Y);
    const auto planeNext = getPlane(srcFrames.backwardRef(1), RGY_PLANE_Y);
    const auto planeNext2 = getPlane(srcFrames.backwardRef(2), RGY_PLANE_Y);
    if (planePrev2.ptr[0] == nullptr || planePrev.ptr[0] == nullptr || planeCur.ptr[0] == nullptr
        || planeNext.ptr[0] == nullptr || planeNext2.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis luma requires valid source luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const int srcPitch = planeCur.pitch[0];
    if (planePrev2.pitch[0] != srcPitch || planePrev.pitch[0] != srcPitch
        || planeNext.pitch[0] != srcPitch || planeNext2.pitch[0] != srcPitch) {
        AddMessage(RGY_LOG_ERROR,
            _T("degrain analysis luma pitch mismatch: prev2=%d, prev=%d, cur=%d, next=%d, next2=%d.\n"),
            planePrev2.pitch[0], planePrev.pitch[0], srcPitch, planeNext.pitch[0], planeNext2.pitch[0]);
        return RGY_ERR_INVALID_PARAM;
    }

    auto err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait degrain analysis luma input events: %s.\n"), get_err_mes(err));
        return err;
    }
    const int slot = analysisCacheIndex(centerFrame);
    auto &analysisFrame = m_analysis.analysisLumaFrame[slot];
    err = launchNVEncDegrainTemporalSmoothLuma(
        planePrev2, planePrev, planeCur, planeNext, planeNext2,
        analysisFrame, prm->degrain.tr0, prm->degrain.searchRefine, prm->degrain.rep0, prm->degrain.tvRange ? 1 : 0, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to generate degrain analysis luma: %s.\n"), get_err_mes(err));
        return err;
    }

    RGYCudaEvent lumaEvent;
    err = degrainRecordEvent(stream, &lumaEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain analysis luma event: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&analysisFrame, srcFrames.cur);
    analysisFrame.width = m_analysis.analysisLumaWidth;
    analysisFrame.height = m_analysis.analysisLumaHeight;
    m_analysis.analysisLumaFrameNumbers[slot] = centerFrame;
    m_analysis.analysisLumaGeneratedUntil = centerFrame;
    m_analysis.analysisLumaEvents[slot] = lumaEvent;
    m_analysis.analysisLumaEvent = lumaEvent;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::ensureAnalysisLumaGenerated(int targetFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!useAnalysisLumaCache()) {
        return RGY_ERR_NONE;
    }
    if (m_inputCount <= 0) {
        return RGY_ERR_NONE;
    }

    const int clampedTargetFrame = std::min(targetFrame, m_inputCount - 1);
    if (clampedTargetFrame < 0) {
        return RGY_ERR_NONE;
    }

    const int firstFrame = m_analysis.analysisLumaGeneratedUntil + 1;
    auto chainedWaitEvents = wait_events;
    for (int frame = firstFrame; frame <= clampedTargetFrame; frame++) {
        const auto err = generateAnalysisLumaFrame(frame, stream, chainedWaitEvents);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        chainedWaitEvents.clear();
        if (m_analysis.analysisLumaEvent() != nullptr) {
            chainedWaitEvents.push_back(m_analysis.analysisLumaEvent);
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterDegrain::clearFrameAnalysisData() {
    m_boundAnalyzeResult = RGYDegrainAnalyzeResult();
    m_frameAnalysisData.reset();
    m_frameAnalysisLayout = {};
}

bool NVEncFilterDegrain::degrainApplyTraceEnabled() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    return m_debugEnv.applyTrace
        && prm
        && prm->degrain.mode == VppDegrainMode::Degrain
        && (prm->degrain.stage == VppDegrainStage::TR1 || prm->degrain.stage == VppDegrainStage::TR2);
}

void NVEncFilterDegrain::logApplyTrace(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
    const RGYDegrainRefDisableArray &disableRefs, cudaStream_t stream) {
    if (!degrainApplyTraceEnabled() || !prm) {
        return;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    const auto &layout = analysisLayout();
    const auto *cur = frames.render.cur ? frames.render.cur : frames.analysis.cur;
    if (!mv || !sad || !cur || layout.blockCount() == 0 || layout.temporalDirections <= 0) {
        return;
    }
    const auto entryCount = layout.blockCount() * (size_t)layout.temporalDirections;
    if (layout.mvCount() < entryCount || layout.sadCount() < entryCount) {
        AddMessage(RGY_LOG_INFO, _T("degrain apply trace skipped: invalid entry count (blocks=%llu, stride=%d).\n"),
            (unsigned long long)layout.blockCount(), layout.temporalDirections);
        return;
    }

    if (analysisEvent()() != nullptr) {
        auto err = err_to_rgy(cudaStreamWaitEvent(stream, analysisEvent()(), 0));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to wait analysis event: %s.\n"), get_err_mes(err));
            return;
        }
    }
    std::vector<RGYDegrainMV> mvValues(layout.mvCount());
    std::vector<RGYDegrainSAD> sadValues(layout.sadCount());
    auto err = err_to_rgy(cudaMemcpyAsync(mvValues.data(), mv->ptr, rgy_degrain_mv_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to copy MV buffer: %s.\n"), get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaMemcpyAsync(sadValues.data(), sad->ptr, rgy_degrain_sad_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to copy SAD buffer: %s.\n"), get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to synchronize MV/SAD copy: %s.\n"), get_err_mes(err));
        return;
    }

    auto availabilityDisableRefs = analysisAvailabilityDisableRefs(frames.analysis);
    auto useFlagDisableRefs = RGYDegrainRefDisableArray();
    useFlagDisableRefs.fill(false);
    if (prm->degrain.useFlag == 1) {
        for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
            useFlagDisableRefs[rgy_degrain_ref_index(delta, true)] = true;
        }
    } else if (prm->degrain.useFlag == 2) {
        for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
            useFlagDisableRefs[rgy_degrain_ref_index(delta, false)] = true;
        }
    }

    const uint32_t scaledThSad = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, analysisSADIncludesChroma(prm));
    const uint32_t disableMask = degrainDisableMask(disableRefs, layout.temporalDirections);
    const bool binomial = prm->degrain.stage != VppDegrainStage::TR2;
    const auto temporalMixPrior = degrainBuildTemporalMixPriorTable(layout.temporalDirections, binomial);
    const float sourceConfidenceRaw = temporalMixPrior[0];
    const int refDirectionCount = std::min(layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS);
    const TCHAR *stageName = get_cx_desc(list_vpp_degrain_stage, (int)prm->degrain.stage);

    std::array<size_t, 4> sampleBlocks = {
        0,
        layout.blockCount() / 2,
        layout.blockCount() - 1,
        layout.blockCount()
    };
    static const TCHAR *sampleNames[] = { _T("first"), _T("mid"), _T("last"), _T("target") };
    const int targetBlock = m_debugEnv.applyTraceBlock;
    if (targetBlock >= 0 && (size_t)targetBlock < layout.blockCount()) {
        sampleBlocks[3] = (size_t)targetBlock;
    }
    for (int sample = 0; sample < (int)sampleBlocks.size(); sample++) {
        const size_t block = sampleBlocks[sample];
        if (block >= layout.blockCount()
            || (sample > 0 && block == sampleBlocks[sample - 1])
            || (sample > 1 && block == sampleBlocks[sample - 2])) {
            continue;
        }
        std::array<float, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> referenceConfidenceRaw = {};
        float confidenceSum = sourceConfidenceRaw;
        float referenceMixTotal = 0.0f;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)refDirection;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            const bool directionDisabled = disableRefs[refDirection];
            const bool validMotion = (int)mvValue.refdir == refDirection;
            const bool underThSad = sadValue.sad < scaledThSad;
            if (!directionDisabled && validMotion && underThSad) {
                referenceConfidenceRaw[refDirection] = degrainTraceReferenceAffinityFromSad((int)scaledThSad, (int)sadValue.sad)
                    * temporalMixPrior[1 + refDirection];
                confidenceSum += referenceConfidenceRaw[refDirection];
            }
        }
        const float invWeightSum = (confidenceSum > 0.0f) ? (1.0f / confidenceSum) : 0.0f;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            referenceMixTotal += (referenceConfidenceRaw[refDirection] > 0.0f) ? (referenceConfidenceRaw[refDirection] * invWeightSum) : 0.0f;
        }
        const float sourceMixNorm = sourceConfidenceRaw * invWeightSum;

        const int blockX = (layout.blocksX > 0) ? (int)(block % (size_t)layout.blocksX) : 0;
        const int blockY = (layout.blocksX > 0) ? (int)(block / (size_t)layout.blocksX) : 0;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)refDirection;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            const int delta = rgy_degrain_delta_from_ref_index(refDirection);
            const bool forward = rgy_degrain_ref_index_is_forward(refDirection);
            const bool availabilityDisabled = availabilityDisableRefs[refDirection];
            const bool useFlagDisabled = useFlagDisableRefs[refDirection] && !availabilityDisabled;
            const bool sceneChangeDisabled = disableRefs[refDirection] && !availabilityDisabled && !useFlagDisabled;
            const bool validMotion = (int)mvValue.refdir == refDirection;
            const bool underThSad = sadValue.sad < scaledThSad;
            const float referenceMixNorm = (referenceConfidenceRaw[refDirection] > 0.0f) ? (referenceConfidenceRaw[refDirection] * invWeightSum) : 0.0f;
            AddMessage(RGY_LOG_INFO,
                _T("{\"type\":\"degrain_mix_trace\",\"frame\":%d,\"pts\":%lld,\"dur\":%lld,\"stage\":\"%s\",\"request_delta\":%d,\"delta\":%d,\"ref_slot\":%d,\"temporal_side\":\"%s\",\"sample\":\"%s\",\"block\":%llu,\"block_x\":%d,\"block_y\":%d,\"entry\":%llu,\"motion\":{\"dx\":%d,\"dy\":%d,\"sad\":%u,\"ref_slot\":%u},\"sad_stats\":{\"sad\":%u,\"src_avg\":%u,\"ref_avg\":%u},\"sad_limit\":%u,\"reference_policy\":%d,\"disable\":{\"mask\":%u,\"availability\":%d,\"policy\":%d,\"scene\":%d,\"final\":%d},\"valid\":{\"motion\":%d,\"sad\":%d},\"selected\":%d,\"mix\":{\"confidence_raw\":%.9g,\"reference_norm\":%.9g,\"source_norm\":%.9g,\"reference_norm_sum\":%.9g,\"confidence_sum\":%.9g,\"source_raw\":%.9g},\"layout\":{\"blocks\":%llu,\"blocks_x\":%d,\"blocks_y\":%d,\"directions\":%d,\"block_size\":%d,\"overlap\":%d,\"step\":%d}}\n"),
                cur->inputFrameId, (long long)cur->timestamp, (long long)cur->duration,
                stageName, requestedDelta(), delta, refDirection, forward ? _T("prev") : _T("next"),
                sampleNames[sample],
                (unsigned long long)block, blockX, blockY, (unsigned long long)entry,
                (int)mvValue.dx, (int)mvValue.dy, (unsigned int)mvValue.sad, (unsigned int)mvValue.refdir,
                (unsigned int)sadValue.sad, (unsigned int)sadValue.srcAvg, (unsigned int)sadValue.refAvg,
                (unsigned int)scaledThSad, prm->degrain.useFlag,
                (unsigned int)disableMask, availabilityDisabled ? 1 : 0, useFlagDisabled ? 1 : 0, sceneChangeDisabled ? 1 : 0, disableRefs[refDirection] ? 1 : 0,
                validMotion ? 1 : 0, underThSad ? 1 : 0, referenceMixNorm > 0.0f ? 1 : 0,
                (double)referenceConfidenceRaw[refDirection], (double)referenceMixNorm, (double)sourceMixNorm, (double)referenceMixTotal, (double)confidenceSum, (double)sourceConfidenceRaw,
                (unsigned long long)layout.blockCount(), layout.blocksX, layout.blocksY, layout.temporalDirections, layout.blockSize, layout.overlap, layout.step);
        }
    }
}

bool NVEncFilterDegrain::validateAnalyzeResultFrame(const RGYDegrainAnalyzeResult &result, const RGYFrameInfo *frame, const int currentFrame, const TCHAR *sourceName, const bool requireFrameIndex) {
    if (!frame || !sourceName) {
        return false;
    }
    if (!result.hasFrameIdentity()) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain %s MV/SAD frame identity is missing; falling back to frame data/local analysis.\n"), sourceName);
        return false;
    }
    if ((requireFrameIndex && result.frameIndex != currentFrame)
        || result.inputFrameId != frame->inputFrameId
        || result.timestamp != frame->timestamp
        || result.duration != frame->duration) {
        AddMessage(RGY_LOG_ERROR,
            _T("degrain %s MV/SAD frame mismatch; expected frameIndex=%d, inputFrameId=%d, timestamp=%lld, duration=%lld, got frameIndex=%d, inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
            sourceName, currentFrame, frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
            result.frameIndex, result.inputFrameId, (long long)result.timestamp, (long long)result.duration);
        assert((!requireFrameIndex || result.frameIndex == currentFrame)
            && result.inputFrameId == frame->inputFrameId
            && result.timestamp == frame->timestamp
            && result.duration == frame->duration);
        return false;
    }
    return true;
}

bool NVEncFilterDegrain::bindDirectAnalyzeResult(const RGYFrameInfo *frame, const int currentFrame, cudaStream_t stream) {
    const auto result = m_directAnalyzeResultSet.get(requestedDelta());
    if (!result) {
        return false;
    }
    if (!validateAnalyzeResultFrame(*result, frame, currentFrame, _T("direct"), true)) {
        return false;
    }
    if (!rgy_degrain_layout_equal(result->layout, m_analysis.layout)) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain direct MV/SAD layout mismatch; falling back to frame data/local analysis.\n"));
        return false;
    }
    m_boundAnalyzeResult = *result;
    m_frameAnalysisLayout = result->layout;
    logAnalyzeBinding(_T("direct"), frame, *result);
    logAnalysisSamples(_T("direct"), frame, stream);
    return true;
}

bool NVEncFilterDegrain::bindFrameAnalysisData(const RGYFrameInfo *frame, const int currentFrame, cudaStream_t stream) {
    clearFrameAnalysisData();
    auto frameAnalysis = rgy_degrain_get_frame_data(frame);
    if (frameAnalysis) {
        const auto result = frameAnalysis->analyzeResult();
        const auto layout = result.layout;
        if (result.hasFrameIdentity() && !validateAnalyzeResultFrame(result, frame, currentFrame, _T("attached"), false)) {
            return false;
        }
        if (!rgy_degrain_layout_equal(layout, m_analysis.layout)) {
            AddMessage(RGY_LOG_DEBUG, _T("degrain attached MV/SAD layout mismatch; falling back to local analysis.\n"));
            return false;
        }
        m_boundAnalyzeResult = result;
        m_frameAnalysisLayout = layout;
        m_frameAnalysisData = frameAnalysis;
        logAnalyzeBinding(_T("attached"), frame, result);
        logAnalysisSamples(_T("attached"), frame, stream);
        return true;
    }
    return bindDirectAnalyzeResult(frame, currentFrame, stream);
}

CUMemBuf *NVEncFilterDegrain::analysisMV() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.mv : m_analysis.mv.get();
}

CUMemBuf *NVEncFilterDegrain::analysisSAD() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.sad : m_analysis.sad.get();
}

const RGYDegrainBlockLayout &NVEncFilterDegrain::analysisLayout() const {
    return m_boundAnalyzeResult.valid() ? m_frameAnalysisLayout : m_analysis.layout;
}

const RGYCudaEvent &NVEncFilterDegrain::analysisEvent() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.event : m_analysis.event;
}

bool NVEncFilterDegrain::analysisSADIncludesChroma(const std::shared_ptr<NVEncFilterParamDegrain> &prm) const {
    if (!prm || !prm->degrain.chroma) {
        return false;
    }
    if (m_boundAnalyzeResult.valid()) {
        return (m_boundAnalyzeResult.flags & RGY_DEGRAIN_FRAME_META_FLAG_CHROMA_SAD) != 0;
    }
    return m_lastAnalysisIncludedChroma;
}

RGYDegrainRefDisableArray NVEncFilterDegrain::analysisAvailabilityDisableRefs(const RGYFilterDegrainFrameSet &frames) const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.availabilityDisableRefs : degrainReferenceAvailability(frames);
}

RGYDegrainAnalyzeResult NVEncFilterDegrain::analyzeResult() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    RGYDegrainAnalyzeResult result;
    if (!prm || !modeRequiresAnalysis(prm->degrain.mode) || !m_analysis.mv || !m_analysis.sad || m_analysis.event() == nullptr) {
        return result;
    }
    result.flags = degrainAnalyzeFlags(prm, useAnalysisLumaCache() || m_lastAnalysisUsedSearchLuma, m_lastAnalysisIncludedChroma);
    result.layout = m_analysis.layout;
    result.mv = m_analysis.mv.get();
    result.sad = m_analysis.sad.get();
    result.event = m_analysis.event;
    result.frameIndex = m_analysis.lastFrameIndex;
    result.inputFrameId = m_analysis.lastInputFrameId;
    result.timestamp = m_analysis.lastTimestamp;
    result.duration = m_analysis.lastDuration;
    result.availabilityDisableRefs = m_analysis.lastAvailabilityDisableRefs;
    return result;
}

RGYDegrainAnalyzeResultSet NVEncFilterDegrain::analyzeResultSet() const {
    RGYDegrainAnalyzeResultSet resultSet;
    const auto baseResult = analyzeResult();
    if (!baseResult.valid()) {
        return resultSet;
    }
    const int maxDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, baseResult.layout.temporalDirections / 2));
    for (int delta = 1; delta <= maxDelta; delta++) {
        auto slot = baseResult;
        slot.layout.temporalDirections = rgy_degrain_temporal_direction_count(delta);
        resultSet.slots[delta] = slot;
    }
    return resultSet;
}

bool NVEncFilterDegrain::setDirectAnalyzeResult(const RGYDegrainAnalyzeResult &result) {
    RGYDegrainAnalyzeResultSet resultSet;
    if (result.valid() && result.hasFrameIdentity()) {
        const int maxDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, result.layout.temporalDirections / 2));
        for (int delta = 1; delta <= maxDelta; delta++) {
            auto slot = result;
            slot.layout.temporalDirections = rgy_degrain_temporal_direction_count(delta);
            resultSet.slots[delta] = slot;
        }
    }
    return setDirectAnalyzeResultSet(resultSet);
}

bool NVEncFilterDegrain::setDirectAnalyzeResultSet(const RGYDegrainAnalyzeResultSet &resultSet) {
    m_directAnalyzeResultSet = resultSet;
    return m_directAnalyzeResultSet.valid(requestedDelta());
}

void NVEncFilterDegrain::clearDirectAnalyzeResult() {
    m_directAnalyzeResultSet = RGYDegrainAnalyzeResultSet();
}
