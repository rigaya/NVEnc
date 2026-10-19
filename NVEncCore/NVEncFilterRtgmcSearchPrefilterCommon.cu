// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

RGY_ERR launchRtgmcSearchPrefilterScenechangeU8(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterScenechangeTyped<uint8_t>(prev2, prev, cur, next, next2, partial, groupCount, stream);
}

RGY_ERR launchRtgmcSearchPrefilterScenechangeU16(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterScenechangeTyped<uint16_t>(prev2, prev, cur, next, next2, partial, groupCount, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRefine2TileU8(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRefine2TileTyped<uint8_t>(motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRefine2TileU16(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterRefine2TileTyped<uint16_t>(motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU8(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterEdgeSoftenedSearchTyped<uint8_t>(searchSmoothed3x3, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU16(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterEdgeSoftenedSearchTyped<uint16_t>(searchSmoothed3x3, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U8(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSearchSmoothed3x3Typed<uint8_t>(src, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U16(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSearchSmoothed3x3Typed<uint16_t>(src, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU8(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendTyped<uint8_t>(spatialGuide, motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU16(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendTyped<uint16_t>(spatialGuide, motionGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU8(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedTyped<uint8_t>(spatialGuide, motionGuide, fieldGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU16(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedTyped<uint16_t>(spatialGuide, motionGuide, fieldGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU8(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterStabilizedSearchTyped<uint8_t>(motionGuide, fieldGuide, spatialGuide, dst, fullRangeMode, stream);
}

RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU16(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    return launchRtgmcSearchPrefilterStabilizedSearchTyped<uint16_t>(motionGuide, fieldGuide, spatialGuide, dst, fullRangeMode, stream);
}





// 4方向を各256スレッドで集約し、部分和をCPUへ読み戻さずに判定する。
__global__ void kernel_rtgmc_search_prefilter_scenechange_reduce_cuda(
    uint32_t *state, const int groupCount, const uint64_t sceneThreshold, const int smoothRadius) {
    const int lid = threadIdx.x;
    const int direction = blockIdx.x;
    __shared__ uint64_t sums[256];
    uint64_t sum = 0;
    if (smoothRadius >= 1 && (direction < 2 || smoothRadius >= 2)) {
        for (int i = lid; i < groupCount; i += 256) {
            sum += (uint64_t)state[4 + direction * groupCount + i];
        }
    }
    sums[lid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) sums[lid] += sums[lid + stride];
        __syncthreads();
    }
    if (lid == 0) {
        state[direction] = (smoothRadius >= 1 && (direction < 2 || smoothRadius >= 2) && sums[0] >= sceneThreshold) ? 1u : 0u;
    }
}

RGY_ERR launchRtgmcSearchPrefilterScenechangeReduce(uint32_t *state, const int groupCount,
    const uint64_t sceneThreshold, const int smoothRadius, cudaStream_t stream) {
    kernel_rtgmc_search_prefilter_scenechange_reduce_cuda<<<4, 256, 0, stream>>>(state, groupCount, sceneThreshold, smoothRadius);
    return err_to_rgy(cudaGetLastError());
}













RGY_ERR launchRtgmcSearchPrefilterRepairTemporal(const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &tc, const RGYFrameInfo &delta, int tr0, const uint32_t *sceneFlags, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[cur.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairTemporalTyped<uint16_t>(prev2, prev, cur, next, next2, tc, delta, tr0, sceneFlags, stream)
        : launchRtgmcSearchPrefilterRepairTemporalTyped<uint8_t>(prev2, prev, cur, next, next2, tc, delta, tr0, sceneFlags, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRepairStage(const RGYFrameInfo &srcPos, const RGYFrameInfo &srcNeg, int srcHx, int srcHy, const RGYFrameInfo &dstPos, const RGYFrameInfo &dstNeg, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, int stage, const RGYFrameInfo &corePos, const RGYFrameInfo &coreNeg, int coreHx, int coreHy, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[srcPos.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairStageTyped<uint16_t>(srcPos, srcNeg, srcHx, srcHy, dstPos, dstNeg, dstHx, dstHy, width, height, repairProfile, stage, corePos, coreNeg, coreHx, coreHy, stream)
        : launchRtgmcSearchPrefilterRepairStageTyped<uint8_t>(srcPos, srcNeg, srcHx, srcHy, dstPos, dstNeg, dstHx, dstHy, width, height, repairProfile, stage, corePos, coreNeg, coreHx, coreHy, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRepairApply(const RGYFrameInfo &tc, const RGYFrameInfo &delta, const RGYFrameInfo &gatePos, const RGYFrameInfo &gateNeg, int gateHx, int gateHy, const RGYFrameInfo &dst, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[tc.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairApplyTyped<uint16_t>(tc, delta, gatePos, gateNeg, gateHx, gateHy, dst, dstHx, dstHy, width, height, repairProfile, stream)
        : launchRtgmcSearchPrefilterRepairApplyTyped<uint8_t>(tc, delta, gatePos, gateNeg, gateHx, gateHy, dst, dstHx, dstHy, width, height, repairProfile, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRepairCopy(const RGYFrameInfo &src, int srcHx, int srcHy, const RGYFrameInfo &dst, int fullRangeMode, int constantValue, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[dst.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairCopyTyped<uint16_t>(src, srcHx, srcHy, dst, fullRangeMode, constantValue, stream)
        : launchRtgmcSearchPrefilterRepairCopyTyped<uint8_t>(src, srcHx, srcHy, dst, fullRangeMode, constantValue, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRepairHalfSearch(bool smoothed, const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &dst, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[fc.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairHalfSearchTyped<uint16_t>(smoothed, fc, fcHx, fcHy, dst, stream)
        : launchRtgmcSearchPrefilterRepairHalfSearchTyped<uint8_t>(smoothed, fc, fcHx, fcHy, dst, stream);
}

RGY_ERR launchRtgmcSearchPrefilterRepairLuma(const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &cur, const RGYFrameInfo &dst, int searchRefine, int fullRangeMode, cudaStream_t stream) {
    return (RGY_CSP_BIT_DEPTH[cur.csp] > 8)
        ? launchRtgmcSearchPrefilterRepairLumaTyped<uint16_t>(fc, fcHx, fcHy, cur, dst, searchRefine, fullRangeMode, stream)
        : launchRtgmcSearchPrefilterRepairLumaTyped<uint8_t>(fc, fcHx, fcHy, cur, dst, searchRefine, fullRangeMode, stream);
}
