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

#include "NVEncFilterDegrain.cuh"

#define NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(PIXEL, BLK) \
RGY_ERR launchNVEncDegrainMotionSearchSearchParallelU##PIXEL##Blk##BLK( \
    const uint8_t *sourcePlane, const uint8_t *referencePlane, CUMemBuf &vectors, \
    const int pitch, const int width, const int height, const int planeBase, const int blockCount, \
    const RGYDegrainBlockLayout &layout, const int pel, const int subpelInterp, \
    const int pad, const int motionCostScale, const int lowSadWeightScale, \
    const int zeroCandidateCostScale, const int frameAverageCandidateCostScale, \
    const int newCandidateCostScale, const int level, cudaStream_t stream); \
RGY_ERR launchNVEncDegrainMotionSearchSpatialRefineU##PIXEL##Blk##BLK( \
    const uint8_t *sourcePlane, const uint8_t *referencePlane, \
    CUMemBuf &vectors, const CUMemBuf &vectorsPrev, CUMemBuf &vectorsFinal, \
    const int pitch, const int width, const int height, const int planeBase, const int finalBase, \
    const int blockCount, const RGYDegrainBlockLayout &layout, \
    const int pel, const int subpelInterp, const int pad, const int motionCostScale, \
    const int lowSadWeightScale, const int newCandidateCostScale, cudaStream_t stream)

NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(8, 8);
NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(8, 16);
NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(8, 32);
NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(16, 8);
NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(16, 16);
NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS(16, 32);

#undef NVENC_DEGRAIN_DECLARE_MOTION_SEARCH_LAUNCHERS

RGY_ERR launchNVEncDegrainTemporalSmoothLuma(
    const RGYFrameInfo& prev2, const RGYFrameInfo& prev, const RGYFrameInfo& cur, const RGYFrameInfo& next, const RGYFrameInfo& next2,
    const RGYFrameInfo& dst, const int tr0, const int searchRefine, const int rep0, const int tvRange, cudaStream_t stream) {
    return launchNVEncDegrainTemporalSmoothLumaImpl(prev2, prev, cur, next, next2, dst, tr0, searchRefine, rep0, tvRange, stream);
}

RGY_ERR launchNVEncDegrainDownsampleLuma2x(
    const RGYFrameInfo& src, const CUMemBuf& dst, const int dstPitch, const int dstWidth, const int dstHeight, cudaStream_t stream) {
    return launchNVEncDegrainDownsampleLuma2xImpl(src, dst, dstPitch, dstWidth, dstHeight, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchSeedAnchorVectors(
    CUMemBuf& vectors, const CUMemBuf& frameAverageMV, const int frameAverageIndex, const int planeStride,
    const int refs, const int pel, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchSeedAnchorVectorsImpl(vectors, frameAverageMV, frameAverageIndex, planeStride, refs, pel, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchSeedZeroVectors(
    CUMemBuf& vectors, CUMemBuf& vectorsPrev, CUMemBuf& sads,
    const int planeBase, const int sadBase, const int blockCount, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchSeedZeroVectorsImpl(vectors, vectorsPrev, sads, planeBase, sadBase, blockCount, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchExpandCoarseVectors(
    const CUMemBuf& srcVectorsFinal, CUMemBuf& dstVectors, CUMemBuf& dstVectorsPrev, CUMemBuf& dstSads,
    const int srcFinalBase, const int dstPlaneBase, const int dstSadBase, const int srcBlockCount,
    const int dstBlockCount, const int srcBlocksX, const int srcBlocksY, const int dstBlocksX, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchExpandCoarseVectorsImpl(srcVectorsFinal, dstVectors, dstVectorsPrev, dstSads,
        srcFinalBase, dstPlaneBase, dstSadBase, srcBlockCount, dstBlockCount, srcBlocksX, srcBlocksY, dstBlocksX, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchExportSad(
    CUMemBuf& vectorsFinal, CUMemBuf& sadsInternal, CUMemBuf *outputMotion, CUMemBuf *outputSad,
    const int finalBase, const int sadBase, const int blockCount, const int outOffset,
    const int referenceDirection, const int refs, cudaStream_t stream) {
    return launchNVEncDegrainMotionSearchExportSadImpl(vectorsFinal, sadsInternal, outputMotion, outputSad,
        finalBase, sadBase, blockCount, outOffset, referenceDirection, refs, stream);
}

RGY_ERR launchNVEncDegrainMotionSearchSearchParallel(
    const uint8_t *sourcePlane, const uint8_t *referencePlane, CUMemBuf &vectors,
    const int pitch, const int width, const int height, const int planeBase, const int blockCount,
    const RGYDegrainBlockLayout &layout, const int pixelBytes, const int pel, const int subpelInterp,
    const int pad, const int motionCostScale, const int lowSadWeightScale,
    const int zeroCandidateCostScale, const int frameAverageCandidateCostScale,
    const int newCandidateCostScale, const int level, cudaStream_t stream) {
#define NVENC_DEGRAIN_DISPATCH_SEARCH(PIXEL, BLK) \
    return launchNVEncDegrainMotionSearchSearchParallelU##PIXEL##Blk##BLK(sourcePlane, referencePlane, vectors, \
        pitch, width, height, planeBase, blockCount, layout, pel, subpelInterp, pad, motionCostScale, lowSadWeightScale, \
        zeroCandidateCostScale, frameAverageCandidateCostScale, newCandidateCostScale, level, stream)
    if (pixelBytes > 1) {
        switch (layout.blockSize) {
        case 8:  NVENC_DEGRAIN_DISPATCH_SEARCH(16, 8);
        case 16: NVENC_DEGRAIN_DISPATCH_SEARCH(16, 16);
        case 32: NVENC_DEGRAIN_DISPATCH_SEARCH(16, 32);
        default: return RGY_ERR_INVALID_PARAM;
        }
    } else {
        switch (layout.blockSize) {
        case 8:  NVENC_DEGRAIN_DISPATCH_SEARCH(8, 8);
        case 16: NVENC_DEGRAIN_DISPATCH_SEARCH(8, 16);
        case 32: NVENC_DEGRAIN_DISPATCH_SEARCH(8, 32);
        default: return RGY_ERR_INVALID_PARAM;
        }
    }
#undef NVENC_DEGRAIN_DISPATCH_SEARCH
}

RGY_ERR launchNVEncDegrainMotionSearchSpatialRefine(
    const uint8_t *sourcePlane, const uint8_t *referencePlane,
    CUMemBuf &vectors, const CUMemBuf &vectorsPrev, CUMemBuf &vectorsFinal,
    const int pitch, const int width, const int height, const int planeBase, const int finalBase,
    const int blockCount, const RGYDegrainBlockLayout &layout, const int pixelBytes,
    const int pel, const int subpelInterp, const int pad, const int motionCostScale,
    const int lowSadWeightScale, const int newCandidateCostScale, cudaStream_t stream) {
#define NVENC_DEGRAIN_DISPATCH_REFINE(PIXEL, BLK) \
    return launchNVEncDegrainMotionSearchSpatialRefineU##PIXEL##Blk##BLK(sourcePlane, referencePlane, vectors, vectorsPrev, vectorsFinal, \
        pitch, width, height, planeBase, finalBase, blockCount, layout, pel, subpelInterp, pad, motionCostScale, \
        lowSadWeightScale, newCandidateCostScale, stream)
    if (pixelBytes > 1) {
        switch (layout.blockSize) {
        case 8:  NVENC_DEGRAIN_DISPATCH_REFINE(16, 8);
        case 16: NVENC_DEGRAIN_DISPATCH_REFINE(16, 16);
        case 32: NVENC_DEGRAIN_DISPATCH_REFINE(16, 32);
        default: return RGY_ERR_INVALID_PARAM;
        }
    } else {
        switch (layout.blockSize) {
        case 8:  NVENC_DEGRAIN_DISPATCH_REFINE(8, 8);
        case 16: NVENC_DEGRAIN_DISPATCH_REFINE(8, 16);
        case 32: NVENC_DEGRAIN_DISPATCH_REFINE(8, 32);
        default: return RGY_ERR_INVALID_PARAM;
        }
    }
#undef NVENC_DEGRAIN_DISPATCH_REFINE
}

RGY_ERR launchNVEncDegrainBuildTemporalMixPlan(
    CUMemBuf &temporalMixPlan, const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const int blockCount, const uint32_t thsad, const uint32_t disableMask, const int refs, cudaStream_t stream) {
    return launchNVEncDegrainBuildTemporalMixPlanImpl(temporalMixPlan, mv, sad, temporalMixPrior, blockCount, thsad, disableMask, refs, stream);
}

static __global__ void kernel_degrain_scene_change_count_cuda(
    uint32_t *sceneChangeCounts, const RGYDegrainSAD *sad,
    const int blockCount, const int temporalDirections, const uint32_t thscd1, const uint32_t baseDisableMask) {
    const int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int count = blockCount * temporalDirections;
    if (idx >= count) {
        return;
    }
    const int refDirection = idx % temporalDirections;
    if (((baseDisableMask >> refDirection) & 1u) != 0u) {
        return;
    }
    if (sad[idx].sad > thscd1) {
        atomicAdd(&sceneChangeCounts[refDirection], 1u);
    }
}

static __global__ void kernel_degrain_scene_change_mask_cuda(
    uint32_t *disableMaskPtr, const uint32_t *sceneChangeCounts,
    const int temporalDirections, const uint32_t baseDisableMask, const uint64_t thscd2) {
    uint32_t disableMask = baseDisableMask;
    for (int refDirection = 0; refDirection < temporalDirections; refDirection++) {
        if ((uint64_t)sceneChangeCounts[refDirection] > thscd2) {
            disableMask |= (1u << refDirection);
        }
    }
    disableMaskPtr[0] = disableMask;
}

RGY_ERR launchNVEncDegrainSceneChangeMask(
    CUMemBuf &sceneChangeCounts, CUMemBuf &disableMaskBuf, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout, const uint32_t thscd1, const uint32_t baseDisableMask, const uint64_t thscd2, cudaStream_t stream) {
    auto err = err_to_rgy(cudaMemsetAsync(sceneChangeCounts.ptr, 0, sceneChangeCounts.nSize, stream));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    const int block = 256;
    const int count = (int)(layout.blockCount() * (size_t)layout.temporalDirections);
    const int grid = divCeil(count, block);
    if (count > 0) {
        kernel_degrain_scene_change_count_cuda<<<grid, block, 0, stream>>>(
            reinterpret_cast<uint32_t *>(sceneChangeCounts.ptr),
            reinterpret_cast<const RGYDegrainSAD *>(sad.ptr),
            (int)layout.blockCount(), layout.temporalDirections, thscd1, baseDisableMask);
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    kernel_degrain_scene_change_mask_cuda<<<1, 1, 0, stream>>>(
        reinterpret_cast<uint32_t *>(disableMaskBuf.ptr),
        reinterpret_cast<const uint32_t *>(sceneChangeCounts.ptr),
        layout.temporalDirections, baseDisableMask, thscd2);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR launchNVEncDegrainOverlapPlane(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *ref0,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const VppDegrainMode mode, const int refDirection,
    const uint32_t thsad, const uint32_t disableMask,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    return launchNVEncDegrainOverlapPlaneImpl(dst, dstPitch, pixelBytes, cur, curPitch, ref0,
        refBackward1, refForward1, refBackward2, refForward2, refBackward3, refForward3,
        refBackward4, refForward4, refBackward5, refForward5, width, height, mv, sad, temporalMixPrior,
        layout, coveredWidth, coveredHeight, planeScaleX, planeScaleY, mode, refDirection,
        thsad, disableMask, refs, pel, subpelInterp, stream);
}

RGY_ERR launchNVEncDegrainCompensateOverlapPlaneRamp(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *ref0, const uint8_t *ref,
    const int refDirection, const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const CUMemBuf &windowRamp,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    return launchNVEncDegrainCompensateOverlapPlaneRampImpl(dst, dstPitch, pixelBytes, cur, curPitch,
        ref0, ref, refDirection, width, height, mv, sad, layout, coveredWidth, coveredHeight,
        planeScaleX, planeScaleY, thsad, disableMask, windowRamp, refs, pel, subpelInterp, stream);
}

RGY_ERR launchNVEncDegrainDegrainOverlapPlane(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    return launchNVEncDegrainDegrainOverlapPlaneImpl(dst, dstPitch, pixelBytes, cur, curPitch,
        refBackward1, refForward1, refBackward2, refForward2, refBackward3, refForward3,
        refBackward4, refForward4, refBackward5, refForward5, width, height, mv, sad, temporalMixPrior,
        layout, coveredWidth, coveredHeight, planeScaleX, planeScaleY, thsad, disableMask,
        refs, pel, subpelInterp, stream);
}

RGY_ERR launchNVEncDegrainDegrainOverlapPlanePreweightedRamp(
    uint8_t *dst, const int dstPitch, const int pixelBytes,
    const uint8_t *cur, const int curPitch,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const CUMemBuf &windowRamp, const CUMemBuf &temporalMixPlan,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    return launchNVEncDegrainDegrainOverlapPlanePreweightedRampImpl(dst, dstPitch, pixelBytes, cur, curPitch,
        refBackward1, refForward1, refBackward2, refForward2, refBackward3, refForward3,
        refBackward4, refForward4, refBackward5, refForward5, width, height, mv, layout,
        coveredWidth, coveredHeight, planeScaleX, planeScaleY, windowRamp, temporalMixPlan,
        refs, pel, subpelInterp, stream);
}

RGY_ERR launchNVEncDegrainPixelTrace(
    const uint8_t *cur, const int curPitch, const int pixelBytes,
    const uint8_t *refBackward1, const uint8_t *refForward1,
    const uint8_t *refBackward2, const uint8_t *refForward2,
    const uint8_t *refBackward3, const uint8_t *refForward3,
    const uint8_t *refBackward4, const uint8_t *refForward4,
    const uint8_t *refBackward5, const uint8_t *refForward5,
    const int width, const int height,
    const CUMemBuf &mv, const CUMemBuf &sad, const CUMemBuf &temporalMixPrior,
    const RGYDegrainBlockLayout &layout,
    const int coveredWidth, const int coveredHeight,
    const int planeScaleX, const int planeScaleY,
    const uint32_t thsad, const uint32_t disableMask,
    const int targetX, const int targetY,
    CUMemBuf &trace,
    const int refs, const int pel, const int subpelInterp, cudaStream_t stream) {
    return launchNVEncDegrainPixelTraceImpl(cur, curPitch, pixelBytes,
        refBackward1, refForward1, refBackward2, refForward2, refBackward3, refForward3,
        refBackward4, refForward4, refBackward5, refForward5, width, height,
        mv, sad, temporalMixPrior, layout, coveredWidth, coveredHeight, planeScaleX, planeScaleY,
        thsad, disableMask, targetX, targetY, trace, refs, pel, subpelInterp, stream);
}

RGY_ERR launchNVEncDegrainDebug(
    const RGYFrameInfo& dst, const VppDegrainMode mode, const CUMemBuf& mv, const CUMemBuf& sad,
    const RGYDegrainBlockLayout& layout, const int pel, cudaStream_t stream) {
    return launchNVEncDegrainDebugImpl(dst, mode, mv, sad, layout, pel, stream);
}
