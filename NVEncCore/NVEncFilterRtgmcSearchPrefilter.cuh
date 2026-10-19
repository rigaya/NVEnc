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

#pragma once

#include "NVEncFilterRtgmcSearchPrefilter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "NVEncFilter.h"
#include "rgy_frame_info.h"
#include "rgy_util.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static constexpr int RTGMC_SEARCH_PREFILTER_BLOCK_X = 16;
static constexpr int RTGMC_SEARCH_PREFILTER_BLOCK_Y = 16;
static constexpr bool RTGMC_SEARCH_PREFILTER_USE_SEARCH_REFINE1_CHAIN = true;
static constexpr int RTGMC_SEARCH_PREFILTER_PIXEL_MAX_8 = 255;
static constexpr int RTGMC_SEARCH_PREFILTER_PIXEL_MAX_16 = 65535;
static constexpr int RTGMC_SEARCH_PREFILTER_SCENECHANGE = 28;
static constexpr int RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN_8 = 16;
static constexpr int RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE_8 = 219;
static constexpr int RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET_8 = 128;
static constexpr int RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE_8 = 112;
static constexpr float RTGMC_SEARCH_REFINE2_GAUSS_W0 = 0.227027029f;
static constexpr float RTGMC_SEARCH_REFINE2_GAUSS_W1 = 0.197707996f;
static constexpr float RTGMC_SEARCH_REFINE2_GAUSS_W2 = 0.130435750f;
static constexpr float RTGMC_SEARCH_REFINE2_GAUSS_W3 = 0.065223776f;
static constexpr float RTGMC_SEARCH_REFINE2_GAUSS_W4 = 0.024685025f;

struct NVEncRtgmcSearchPrefilterLaunchFuncs {
    RGY_ERR (*scenechange)(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, uint32_t *, int, cudaStream_t);
    RGY_ERR (*refine2Tile)(const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
    RGY_ERR (*edgeSoftenedSearch)(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
    RGY_ERR (*searchSmoothed3x3)(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
    RGY_ERR (*softenedSearchBlend)(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
    RGY_ERR (*softenedSearchBlendStabilized)(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
    RGY_ERR (*stabilizedSearch)(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
    RGY_ERR (*halfResolutionUpsample)(const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
};

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU8TR0();
const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU8TRP();
const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU16TR0();
const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU16TRP();

RGY_ERR launchRtgmcSearchPrefilterScenechangeReduce(uint32_t *state, int groupCount, uint64_t sceneThreshold, int smoothRadius, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterScenechangeU8(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, uint32_t *, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterScenechangeU16(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, uint32_t *, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterRefine2TileU8(const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterRefine2TileU16(const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU8(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchU16(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U8(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3U16(const RGYFrameInfo&, const RGYFrameInfo&, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU8(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendU16(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU8(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU16(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU8(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);
RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchU16(const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, const RGYFrameInfo&, int, cudaStream_t);

RGY_ERR launchRtgmcSearchPrefilterRepairTemporal(const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &tc, const RGYFrameInfo &delta, int tr0, const uint32_t *sceneFlags, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterRepairStage(const RGYFrameInfo &srcPos, const RGYFrameInfo &srcNeg, int srcHx, int srcHy, const RGYFrameInfo &dstPos, const RGYFrameInfo &dstNeg, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, int stage, const RGYFrameInfo &corePos, const RGYFrameInfo &coreNeg, int coreHx, int coreHy, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterRepairApply(const RGYFrameInfo &tc, const RGYFrameInfo &delta, const RGYFrameInfo &gatePos, const RGYFrameInfo &gateNeg, int gateHx, int gateHy, const RGYFrameInfo &dst, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterRepairCopy(const RGYFrameInfo &src, int srcHx, int srcHy, const RGYFrameInfo &dst, int fullRangeMode, int constantValue, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterRepairHalfSearch(bool smoothed, const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &dst, cudaStream_t stream);
RGY_ERR launchRtgmcSearchPrefilterRepairLuma(const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &cur, const RGYFrameInfo &dst, int searchRefine, int fullRangeMode, cudaStream_t stream);

#ifndef NVENC_RTGMC_SEARCH_PREFILTER_DECLARE_ONLY

template<typename T>
__device__ __forceinline__ int rtgmc_search_prefilter_pixel_max();
template<>
__device__ __forceinline__ int rtgmc_search_prefilter_pixel_max<uint8_t>() { return RTGMC_SEARCH_PREFILTER_PIXEL_MAX_8; }
template<>
__device__ __forceinline__ int rtgmc_search_prefilter_pixel_max<uint16_t>() { return RTGMC_SEARCH_PREFILTER_PIXEL_MAX_16; }

__device__ __forceinline__ int rtgmc_search_prefilter_clamp_int(const int value, const int minValue, const int maxValue) {
    return min(max(value, minValue), maxValue);
}

__device__ __forceinline__ float rtgmc_search_prefilter_clamp_float(const float value, const float minValue, const float maxValue) {
    return fminf(fmaxf(value, minValue), maxValue);
}

template<typename TypePixel>
__device__ __forceinline__ TypePixel rtgmc_search_prefilter_clamp_pixel(const int value) {
    return (TypePixel)rtgmc_search_prefilter_clamp_int(value, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_pixel_load(
    const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    const int px = rtgmc_search_prefilter_clamp_int(x, 0, width - 1);
    const int py = rtgmc_search_prefilter_clamp_int(y, 0, height - 1);
    return (int)(*(const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

template<typename TypePixel>
__device__ __forceinline__ void rtgmc_search_prefilter_pixel_store(
    uint8_t *dst, const int pitch, const int x, const int y, const int value) {
    *(TypePixel *)(dst + y * pitch + x * (int)sizeof(TypePixel)) = rtgmc_search_prefilter_clamp_pixel<TypePixel>(value);
}

__device__ __forceinline__ int rtgmc_search_repair_profile_restore_padding_level(const uint32_t repairProfile) {
    return (int)((repairProfile >> 8) & 0xffu);
}

__device__ __forceinline__ uint32_t rtgmc_search_repair_profile_thin_reject_flags(const uint32_t repairProfile) {
    return (repairProfile >> 16) & 0xffu;
}

__device__ __forceinline__ uint32_t rtgmc_search_repair_profile_restore_flags(const uint32_t repairProfile) {
    return (repairProfile >> 24) & 0xffu;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_range_half() {
    return (rtgmc_search_prefilter_pixel_max<TypePixel>() + 1) >> 1;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_range_scale() {
    return max((rtgmc_search_prefilter_pixel_max<TypePixel>() + 1) >> 8, 1);
}

__device__ __forceinline__ int rtgmc_search_prefilter_extreme_seed(const int highSide) {
    return highSide ? 0 : INT_MAX;
}

__device__ __forceinline__ int rtgmc_search_prefilter_extreme_merge(const int value, const int sample, const int highSide) {
    return highSide ? max(value, sample) : min(value, sample);
}

__device__ __forceinline__ int rtgmc_search_prefilter_polarity_core_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(!positive);
}

__device__ __forceinline__ int rtgmc_search_prefilter_polarity_core_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, !positive);
}

__device__ __forceinline__ int rtgmc_search_prefilter_polarity_envelope_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(positive);
}

__device__ __forceinline__ int rtgmc_search_prefilter_polarity_envelope_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, positive);
}

__device__ __forceinline__ void rtgmc_search_prefilter_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

__device__ __forceinline__ void rtgmc_search_prefilter_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

__device__ __forceinline__ void rtgmc_search_prefilter_sort8(int *v) {
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2_desc(&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[6]); rtgmc_search_prefilter_sort2_desc(&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[4]); rtgmc_search_prefilter_sort2     (&v[1], &v[5]); rtgmc_search_prefilter_sort2     (&v[2], &v[6]); rtgmc_search_prefilter_sort2     (&v[3], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[6]); rtgmc_search_prefilter_sort2     (&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2     (&v[6], &v[7]);
}

__device__ __forceinline__ int rtgmc_search_prefilter_blur3x3_weighted(
    const int p00, const int p10, const int p20, const int p01, const int p11, const int p21, const int p02, const int p12, const int p22) {
    const int sum = p00 + 2 * p10 + p20 + 2 * p01 + 4 * p11 + 2 * p21 + p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_temporal_sample(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int slot) {
    switch (slot) {
    case 0: return rtgmc_search_prefilter_pixel_load<TypePixel>(srcPrev2, pitch, srcWidth, srcHeight, px, py);
    case 1: return rtgmc_search_prefilter_pixel_load<TypePixel>(srcPrev,  pitch, srcWidth, srcHeight, px, py);
    case 3: return rtgmc_search_prefilter_pixel_load<TypePixel>(srcNext,  pitch, srcWidth, srcHeight, px, py);
    case 4: return rtgmc_search_prefilter_pixel_load<TypePixel>(srcNext2, pitch, srcWidth, srcHeight, px, py);
    default: return rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur,   pitch, srcWidth, srcHeight, px, py);
    }
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_temporal_weighted_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int tapCount) {
    int sum = 0;
    if (tapCount >= 5) {
        const int taps[5] = { 1, 4, 6, 4, 1 };
#pragma unroll
        for (int i = 0; i < 5; i++) {
            sum += taps[i] * rtgmc_search_prefilter_temporal_sample<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, i);
        }
        return (sum + 4) >> 4;
    }
    const int taps[3] = { 1, 2, 1 };
#pragma unroll
    for (int i = 0; i < 3; i++) {
        sum += taps[i] * rtgmc_search_prefilter_temporal_sample<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, i + 1);
    }
    return (sum + 2) >> 2;
}

template<typename TypePixel, int SMOOTH_RADIUS>
__device__ __forceinline__ int rtgmc_search_prefilter_temporal_candidate_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius) {
    if constexpr (SMOOTH_RADIUS >= 2) {
        return rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 5);
    }
    if constexpr (SMOOTH_RADIUS >= 1) {
        return rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 3);
    }
    if constexpr (SMOOTH_RADIUS < 0) {
        return (smoothRadius >= 2) ? rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 5)
            : (smoothRadius >= 1) ? rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 3)
            : rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px, py);
    }
    return rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px, py);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_makediff_value(const int ref, const int src) {
    return rtgmc_search_prefilter_clamp_int(ref - src + rtgmc_search_prefilter_range_half<TypePixel>(), 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_round_float_to_pixel(const float value) {
    return rtgmc_search_prefilter_clamp_int((int)(value + 0.5f), 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_extreme_seed(const int highSide) {
    return highSide ? 0 : rtgmc_search_prefilter_pixel_max<TypePixel>();
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_polarity_core_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed<TypePixel>(!positive);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_polarity_envelope_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed<TypePixel>(positive);
}

__device__ __forceinline__ int rtgmc_search_prefilter_select_signed_correction(
    const int proposedSigned, const int positiveMaskSigned, const int negativeMaskSigned, const int threshold) {
    if (proposedSigned >= threshold) {
        return (positiveMaskSigned > 0) ? positiveMaskSigned : 0;
    }
    if (proposedSigned <= -threshold) {
        return (negativeMaskSigned < 0) ? negativeMaskSigned : 0;
    }
    return 0;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_apply_signed_correction(
    const int src, const int proposedSigned, const int positiveMaskSigned, const int negativeMaskSigned, const int threshold) {
    const int appliedSigned = rtgmc_search_prefilter_select_signed_correction(proposedSigned, positiveMaskSigned, negativeMaskSigned, threshold);
    return rtgmc_search_prefilter_clamp_int(src + appliedSigned, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_to_full_range(const int value, const int planeMode);

// search_refine=1 段階実行用。half_resolution_search_value と同じ係数・丸め手順で、
// 半解像度 smoothed plane をフル解像度へ戻し、fullRangeMode を同時に適用する。
template<typename TypePixel>
__device__ int rtgmc_search_prefilter_half_resolution_upsample_value(
    const uint8_t *smoothed, const int smoothedPitch, const int width, const int height, const int px, const int py, const int fullRangeMode) {
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
    const int filterSize = 2;
    const float filterSupport = 1.0f;
    const float posY = -0.25f + 0.5f * (float)py;
    int endY = (int)(posY + filterSupport);
    endY = min(endY, halfHeight - 1);
    const int startY = max(endY - filterSize + 1, 0);
    const float okPosY = rtgmc_search_prefilter_clamp_float(posY, 0.0f, (float)(halfHeight - 1));

    float totalY = 0.0f;
    float coeffY[2];
    for (int iy = 0; iy < filterSize; iy++) {
        const float d = fabsf((float)(startY + iy) - okPosY);
        coeffY[iy] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalY += coeffY[iy];
    }
    totalY = (totalY == 0.0f) ? 1.0f : totalY;

    const float posX = -0.25f + 0.5f * (float)px;
    int endX = (int)(posX + filterSupport);
    endX = min(endX, halfWidth - 1);
    const int startX = max(endX - filterSize + 1, 0);
    const float okPosX = rtgmc_search_prefilter_clamp_float(posX, 0.0f, (float)(halfWidth - 1));

    float totalX = 0.0f;
    float coeffX[2];
    for (int ix = 0; ix < filterSize; ix++) {
        const float d = fabsf((float)(startX + ix) - okPosX);
        coeffX[ix] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalX += coeffX[ix];
    }
    totalX = (totalX == 0.0f) ? 1.0f : totalX;

    float sumY = 0.5f;
    for (int iy = 0; iy < filterSize; iy++) {
        float sumX = 0.5f;
        for (int ix = 0; ix < filterSize; ix++) {
            const int sample = rtgmc_search_prefilter_pixel_load<TypePixel>(
                smoothed, smoothedPitch, halfWidth, halfHeight,
                rtgmc_search_prefilter_clamp_int(startX + ix, 0, halfWidth - 1),
                rtgmc_search_prefilter_clamp_int(startY + iy, 0, halfHeight - 1));
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = rtgmc_search_prefilter_clamp_int((int)sumX, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return rtgmc_search_prefilter_to_full_range<TypePixel>(
        rtgmc_search_prefilter_clamp_int((int)sumY, 0, rtgmc_search_prefilter_pixel_max<TypePixel>()), fullRangeMode);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_search_smoothed3x3_value(
    const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x, y);
    }
    const int p00 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x - 1, y - 1);
    const int p10 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x,     y - 1);
    const int p20 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x + 1, y - 1);
    const int p01 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x - 1, y);
    const int p11 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x,     y);
    const int p21 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x + 1, y);
    const int p02 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x - 1, y + 1);
    const int p12 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x,     y + 1);
    const int p22 = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x + 1, y + 1);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_motion_guide_blend_value(const int spatialGuide, const int motionGuide) {
    const float guideWeight = 0.10f;
    const float value = (float)spatialGuide * (1.0f - guideWeight) + (float)motionGuide * guideWeight;
    return rtgmc_search_prefilter_clamp_int(__float2int_rn(value), 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_motion_guide_stabilize_value(
    const int motionGuide, const int fieldGuide, const int spatialGuide) {
    const float guideEnvelope = 4.0f;
    const float residualGain = 0.50f;
    const float residualLimit = 3.0f;
    const float scale = (float)rtgmc_search_prefilter_range_scale<TypePixel>();
    const float invScale = 1.0f / scale;
    const float motionGuidef = motionGuide * invScale;
    const float fieldGuidef = fieldGuide * invScale;
    const float spatialGuidef = spatialGuide * invScale;
    const float candidate = rtgmc_search_prefilter_clamp_float(fieldGuidef, motionGuidef - guideEnvelope, motionGuidef + guideEnvelope);

    const float residual = candidate - spatialGuidef;
    const float normalized = residual * (residualGain / residualLimit);
    const float correction = residualGain * residual * rsqrtf(1.0f + normalized * normalized);
    const float ret = spatialGuidef + correction;
    return rtgmc_search_prefilter_round_float_to_pixel<TypePixel>(ret * scale);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_motion_guide_blend_stabilized_value(
    const int spatialGuide, const int motionGuide, const int fieldGuide) {
    const int blendedGuide = rtgmc_search_prefilter_motion_guide_blend_value<TypePixel>(spatialGuide, motionGuide);
    return rtgmc_search_prefilter_motion_guide_stabilize_value<TypePixel>(motionGuide, fieldGuide, blendedGuide);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_to_full_range(const int value, const int planeMode) {
    const int pixelMax = rtgmc_search_prefilter_pixel_max<TypePixel>();
    const int scale = rtgmc_search_prefilter_range_scale<TypePixel>();
    if (planeMode == 1) {
        const int minY = RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN_8 * scale;
        const int rangeY = RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE_8 * scale;
        const float lumaScale = (float)pixelMax / (float)rangeY;
        const float lumaOffset = -(float)minY * lumaScale;
        return rtgmc_search_prefilter_clamp_int(__float2int_rn(fmaf((float)value, lumaScale, lumaOffset)), 0, pixelMax);
    }
    if (planeMode == 2) {
        const float rangeHalfF = (float)((pixelMax + 1) >> 1);
        const float offsetC = (float)(RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET_8 * scale);
        const float rangeC = (float)(RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE_8 * scale);
        const float converted = ((float)value - offsetC) * (rangeHalfF / rangeC) + rangeHalfF;
        return rtgmc_search_prefilter_clamp_int((int)(converted + 0.5f), 0, pixelMax);
    }
    return value;
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_scenechange_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int src_pitch, uint32_t *partial, const int groupCount, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lid = threadIdx.y * RTGMC_SEARCH_PREFILTER_BLOCK_X + threadIdx.x;
    const int groupIndex = blockIdx.y * gridDim.x + blockIdx.x;
    __shared__ uint32_t sadPrev[RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y];
    __shared__ uint32_t sadNext[RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y];
    __shared__ uint32_t sadPrev2[RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y];
    __shared__ uint32_t sadNext2[RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y];
    uint32_t diffPrev = 0, diffNext = 0, diffPrev2 = 0, diffNext2 = 0;
    if (x < width && y < height) {
        const int value = rtgmc_search_prefilter_pixel_load<TypePixel>(cur, src_pitch, width, height, x, y);
        diffPrev = (uint32_t)abs(value - rtgmc_search_prefilter_pixel_load<TypePixel>(prev, src_pitch, width, height, x, y));
        diffNext = (uint32_t)abs(value - rtgmc_search_prefilter_pixel_load<TypePixel>(next, src_pitch, width, height, x, y));
        diffPrev2 = (uint32_t)abs(value - rtgmc_search_prefilter_pixel_load<TypePixel>(prev2, src_pitch, width, height, x, y));
        diffNext2 = (uint32_t)abs(value - rtgmc_search_prefilter_pixel_load<TypePixel>(next2, src_pitch, width, height, x, y));
    }
    sadPrev[lid] = diffPrev;
    sadNext[lid] = diffNext;
    sadPrev2[lid] = diffPrev2;
    sadNext2[lid] = diffNext2;
    __syncthreads();
    for (int stride = (RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sadPrev[lid] += sadPrev[lid + stride];
            sadNext[lid] += sadNext[lid + stride];
            sadPrev2[lid] += sadPrev2[lid + stride];
            sadNext2[lid] += sadNext2[lid + stride];
        }
        __syncthreads();
    }
    if (lid == 0 && groupIndex < groupCount) {
        partial[4 + groupIndex + groupCount * 0] = sadPrev[0];
        partial[4 + groupIndex + groupCount * 1] = sadNext[0];
        partial[4 + groupIndex + groupCount * 2] = sadPrev2[0];
        partial[4 + groupIndex + groupCount * 3] = sadNext2[0];
    }
}

// 参照なしの特殊化では判定バッファを読まない。整数演算へ渡す前に参照だけを置換する。
template<int SMOOTH_RADIUS>
__device__ __forceinline__ void rtgmc_search_prefilter_apply_scene_flags(
    const uint8_t *&prev2, const uint8_t *&prev, const uint8_t *cur, const uint8_t *&next, const uint8_t *&next2,
    const uint32_t *sceneFlags, const int smoothRadius) {
    if constexpr (SMOOTH_RADIUS != 0) {
        if (sceneFlags && (SMOOTH_RADIUS >= 1 || smoothRadius >= 1)) {
            prev = sceneFlags[0] ? cur : prev;
            next = sceneFlags[1] ? cur : next;
            if (SMOOTH_RADIUS >= 2 || (SMOOTH_RADIUS < 0 && smoothRadius >= 2)) {
                prev2 = sceneFlags[2] ? cur : prev2;
                next2 = sceneFlags[3] ? cur : next2;
            }
        }
    }
}

// 中間プレーンは仮想座標を保持し、画面内へのクランプはTC/Dの参照で行う。
template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_repair_plane_load(
    const uint8_t *src, const int pitch,
    const int hx, const int hy, const int x, const int y) {
    return (int)(*(const TypePixel *)(src + (y + hy) * pitch + (x + hx) * (int)sizeof(TypePixel)));
}

// 端分岐はハローの格納座標ではなく、元の仮想的な評価座標で判定する。
template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_repair_plane_mean3x3(
    const uint8_t *src, const int pitch, const int hx, const int hy,
    const int width, const int height, const int x, const int y) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x, y);
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x + dx, y + dy);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_repair_stage_value(
    const uint8_t *src, const int pitch, const int hx, const int hy,
    const int width, const int height, const int x, const int y,
    const uint32_t repairProfile, const int stage, const int positive,
    const uint8_t *core, const int corePitch, const int coreHx, const int coreHy) {
    const uint32_t thinFlags = rtgmc_search_repair_profile_thin_reject_flags(repairProfile);
    const uint32_t restoreFlags = rtgmc_search_repair_profile_restore_flags(repairProfile);
    if (restoreFlags & RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH) {
        if (stage == 0 || stage == 3) {
            int value = (stage == 0) ? rtgmc_search_prefilter_polarity_core_seed<TypePixel>(positive)
                : rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
            for (int dy = -2; dy <= 2; dy++) {
                // はみ出した行は評価行へ戻す。例えばy=1,dy=-2では0ではなく1を読む。
                // Dの画面クランプとは別の置換であり、仮想座標での評価でもこの順序を保つ。
                const int sampleY = (y + dy < 0 || y + dy >= height) ? y : y + dy;
                const int sample = (stage == 0)
                    ? rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x, sampleY)
                    : rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x, sampleY);
                value = (stage == 0) ? rtgmc_search_prefilter_polarity_core_merge(value, sample, positive)
                    : rtgmc_search_prefilter_polarity_envelope_merge(value, sample, positive);
            }
            return value;
        }
        if (stage == 1) {
            return rtgmc_search_prefilter_repair_plane_mean3x3<TypePixel>(src, pitch, hx, hy, width, height, x, y);
        }
        // level4のmidは前段の平均とstage0のcoreを合成する。平均だけをcoreと再合成してはいけない。
        const int mean = rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x, y);
        const int center = rtgmc_search_prefilter_repair_plane_load<TypePixel>(core, corePitch, coreHx, coreHy, x, y);
        return rtgmc_search_prefilter_polarity_core_merge(center, mean, positive);
    }
    if (stage == 0) {
        const int radius = 2 + ((thinFlags & RGY_RTGMC_REPAIR_THIN_WIDE_CORE) ? 1 : 0);
        int value = rtgmc_search_prefilter_polarity_core_seed<TypePixel>(positive);
        for (int dy = -radius; dy <= radius; dy++) {
            // Dだけは画面サイズ。旧実装の入力画素読み込みと同じクランプを保つ。
            const int sample = rtgmc_search_prefilter_pixel_load<TypePixel>(src, pitch, width, height, x, y + dy);
            value = rtgmc_search_prefilter_polarity_core_merge(value, sample, positive);
        }
        return value;
    }
    if (stage == 3) {
        const int radius = 2 + ((restoreFlags & RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE) ? 1 : 0);
        int value = rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
        for (int dy = -radius; dy <= radius; dy++) {
            const int sample = rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x, y + dy);
            value = rtgmc_search_prefilter_polarity_envelope_merge(value, sample, positive);
        }
        return value;
    }
    const int center = rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x, y);
    if (stage == 1 && (thinFlags & RGY_RTGMC_REPAIR_THIN_CORE_BLEND)) {
        const int mean = rtgmc_search_prefilter_repair_plane_mean3x3<TypePixel>(src, pitch, hx, hy, width, height, x, y);
        return rtgmc_search_prefilter_polarity_core_merge(center, mean, positive);
    }
    if (stage == 2 && (thinFlags & RGY_RTGMC_REPAIR_THIN_RANK_LIMIT)) {
        if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
            return center;
        }
        int v[8];
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    v[count++] = rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x + dx, y + dy);
                }
            }
        }
        rtgmc_search_prefilter_sort8(v);
        return rtgmc_search_prefilter_clamp_int(center, v[3], v[4]);
    }
    const int padding = rtgmc_search_repair_profile_restore_padding_level(repairProfile);
    if ((stage == 4 && (padding == 1 || padding == 2)) || stage == 5) {
        // pad=2ではR1を保持した後、同じ端分岐と整数丸めでR2を別段として評価する。
        const int mean = rtgmc_search_prefilter_repair_plane_mean3x3<TypePixel>(src, pitch, hx, hy, width, height, x, y);
        return rtgmc_search_prefilter_extreme_merge(center, mean, positive);
    }
    if (stage == 4 && padding >= 3) {
        // area envelopeには平均の端分岐がない。画面外のBも仮想座標のまま参照する。
        int value = rtgmc_search_prefilter_extreme_seed<TypePixel>(positive);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                const int sample = rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, pitch, hx, hy, x + dx, y + dy);
                value = rtgmc_search_prefilter_extreme_merge(value, sample, positive);
            }
        }
        return value;
    }
    return center;
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_repair_temporal(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur,
    const uint8_t *next, const uint8_t *next2, const int srcPitch,
    uint8_t *tc, uint8_t *delta, const int planePitch,
    const int width, const int height, const int tr0, const uint32_t *sceneFlags) {
    rtgmc_search_prefilter_apply_scene_flags<-1>(prev2, prev, cur, next, next2, sceneFlags, tr0);
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_temporal_candidate_value<TypePixel, -1>(
        prev2, prev, cur, next, next2, srcPitch, width, height, x, y, tr0);
    const int ref = rtgmc_search_prefilter_pixel_load<TypePixel>(cur, srcPitch, width, height, x, y);
    rtgmc_search_prefilter_pixel_store<TypePixel>(tc, planePitch, x, y, value);
    rtgmc_search_prefilter_pixel_store<TypePixel>(delta, planePitch, x, y, rtgmc_search_prefilter_makediff_value<TypePixel>(ref, value));
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_repair_stage(
    const uint8_t *srcPos, const uint8_t *srcNeg,
    const int srcPitch, const int srcHx, const int srcHy,
    uint8_t *dstPos, uint8_t *dstNeg,
    const int dstPitch, const int dstHx, const int dstHy,
    const int width, const int height, const uint32_t repairProfile, const int stage,
    const uint8_t *corePos, const uint8_t *coreNeg,
    const int corePitch, const int coreHx, const int coreHy) {
    const int sx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int sy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (sx >= width + 2 * dstHx || sy >= height + 2 * dstHy) {
        return;
    }
    const int x = sx - dstHx;
    const int y = sy - dstHy;
    const int positive = rtgmc_search_prefilter_repair_stage_value<TypePixel>(
        srcPos, srcPitch, srcHx, srcHy, width, height, x, y, repairProfile, stage, 1, corePos, corePitch, coreHx, coreHy);
    const int negative = rtgmc_search_prefilter_repair_stage_value<TypePixel>(
        srcNeg, srcPitch, srcHx, srcHy, width, height, x, y, repairProfile, stage, 0, coreNeg, corePitch, coreHx, coreHy);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dstPos, dstPitch, sx, sy, positive);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dstNeg, dstPitch, sx, sy, negative);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_repair_apply(
    const uint8_t *tc, const uint8_t *delta, const int temporalPitch,
    const uint8_t *gatePos, const uint8_t *gateNeg,
    const int gatePitch, const int gateHx, const int gateHy,
    uint8_t *dst, const int dstPitch, const int dstHx, const int dstHy,
    const int width, const int height, const uint32_t repairProfile) {
    const int sx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int sy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (sx >= width + 2 * dstHx || sy >= height + 2 * dstHy) {
        return;
    }
    const int x = sx - dstHx;
    const int y = sy - dstHy;
    const int base = rtgmc_search_prefilter_pixel_load<TypePixel>(tc, temporalPitch, width, height, x, y);
    int value = base;
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) & RGY_RTGMC_REPAIR_RESTORE_ENABLED) {
        const int rangeHalf = rtgmc_search_prefilter_range_half<TypePixel>();
        const int diff = rtgmc_search_prefilter_pixel_load<TypePixel>(delta, temporalPitch, width, height, x, y);
        const int positive = rtgmc_search_prefilter_repair_plane_load<TypePixel>(gatePos, gatePitch, gateHx, gateHy, x, y);
        const int negative = rtgmc_search_prefilter_repair_plane_load<TypePixel>(gateNeg, gatePitch, gateHx, gateHy, x, y);
        value = rtgmc_search_prefilter_apply_signed_correction<TypePixel>(
            base, diff - rangeHalf, positive - rangeHalf, negative - rangeHalf, rtgmc_search_prefilter_range_scale<TypePixel>());
    }
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, sx, sy, value);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_half_search_base_from_fc_value(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int hx, const int hy) {
    const int filterSize = 4;
    const float filterSupport = 2.0f;
    const float filterStep = 0.5f;
    const float posY = 0.5f + 2.0f * (float)hy;
    int endY = (int)(posY + filterSupport);
    endY = min(endY, srcHeight - 1);
    int startY = max(endY - filterSize + 1, 0);
    const float okPosY = rtgmc_search_prefilter_clamp_float(posY, 0.0f, (float)(srcHeight - 1));

    float totalY = 0.0f;
    float coeffY[4];
    for (int iy = 0; iy < filterSize; iy++) {
        const float d = fabsf(((float)(startY + iy) - okPosY) * filterStep);
        coeffY[iy] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalY += coeffY[iy];
    }
    totalY = (totalY == 0.0f) ? 1.0f : totalY;

    const float posX = 0.5f + 2.0f * (float)hx;
    int endX = (int)(posX + filterSupport);
    endX = min(endX, srcWidth - 1);
    int startX = max(endX - filterSize + 1, 0);
    const float okPosX = rtgmc_search_prefilter_clamp_float(posX, 0.0f, (float)(srcWidth - 1));

    float totalX = 0.0f;
    float coeffX[4];
    for (int ix = 0; ix < filterSize; ix++) {
        const float d = fabsf(((float)(startX + ix) - okPosX) * filterStep);
        coeffX[ix] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalX += coeffX[ix];
    }
    totalX = (totalX == 0.0f) ? 1.0f : totalX;

    float sumY = 0.5f;
    for (int iy = 0; iy < filterSize; iy++) {
        float sumX = 0.5f;
        for (int ix = 0; ix < filterSize; ix++) {
            // 小寸法では4タップが画面外へ出る。FCハローの仮想座標をクランプせず読む。
            const int sample = rtgmc_search_prefilter_repair_plane_load<TypePixel>(
                fc, fcPitch, fcHx, fcHy, startX + ix, startY + iy);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = rtgmc_search_prefilter_clamp_int((int)sumX, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return rtgmc_search_prefilter_clamp_int((int)sumY, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_half_search_smoothed_from_fc_value(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int hx, const int hy) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    if (hx <= 0 || hy <= 0 || hx >= halfWidth - 1 || hy >= halfHeight - 1) {
        return rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(
            fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight,
            rtgmc_search_prefilter_clamp_int(hx, 0, halfWidth - 1), rtgmc_search_prefilter_clamp_int(hy, 0, halfHeight - 1));
    }
    const int x0 = rtgmc_search_prefilter_clamp_int(hx - 1, 0, halfWidth - 1);
    const int x1 = rtgmc_search_prefilter_clamp_int(hx,     0, halfWidth - 1);
    const int x2 = rtgmc_search_prefilter_clamp_int(hx + 1, 0, halfWidth - 1);
    const int y0 = rtgmc_search_prefilter_clamp_int(hy - 1, 0, halfHeight - 1);
    const int y1 = rtgmc_search_prefilter_clamp_int(hy,     0, halfHeight - 1);
    const int y2 = rtgmc_search_prefilter_clamp_int(hy + 1, 0, halfHeight - 1);
    const int p00 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y0);
    const int p10 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y0);
    const int p20 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y0);
    const int p01 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y1);
    const int p11 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y1);
    const int p21 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y1);
    const int p02 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y2);
    const int p12 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y2);
    const int p22 = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y2);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_half_resolution_search_from_fc_value(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int px, const int py) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    const int filterSize = 2;
    const float filterSupport = 1.0f;
    const float posY = -0.25f + 0.5f * (float)py;
    int endY = (int)(posY + filterSupport);
    endY = min(endY, halfHeight - 1);
    int startY = max(endY - filterSize + 1, 0);
    const float okPosY = rtgmc_search_prefilter_clamp_float(posY, 0.0f, (float)(halfHeight - 1));

    float totalY = 0.0f;
    float coeffY[2];
    for (int iy = 0; iy < filterSize; iy++) {
        const float d = fabsf((float)(startY + iy) - okPosY);
        coeffY[iy] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalY += coeffY[iy];
    }
    totalY = (totalY == 0.0f) ? 1.0f : totalY;

    const float posX = -0.25f + 0.5f * (float)px;
    int endX = (int)(posX + filterSupport);
    endX = min(endX, halfWidth - 1);
    int startX = max(endX - filterSize + 1, 0);
    const float okPosX = rtgmc_search_prefilter_clamp_float(posX, 0.0f, (float)(halfWidth - 1));

    float totalX = 0.0f;
    float coeffX[2];
    for (int ix = 0; ix < filterSize; ix++) {
        const float d = fabsf((float)(startX + ix) - okPosX);
        coeffX[ix] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalX += coeffX[ix];
    }
    totalX = (totalX == 0.0f) ? 1.0f : totalX;

    float sumY = 0.5f;
    for (int iy = 0; iy < filterSize; iy++) {
        float sumX = 0.5f;
        for (int ix = 0; ix < filterSize; ix++) {
            const int sample = rtgmc_search_prefilter_half_search_smoothed_from_fc_value<TypePixel>(
                fc, fcPitch, fcHx, fcHy,
                srcWidth, srcHeight,
                rtgmc_search_prefilter_clamp_int(startX + ix, 0, halfWidth - 1),
                rtgmc_search_prefilter_clamp_int(startY + iy, 0, halfHeight - 1));
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = rtgmc_search_prefilter_clamp_int((int)sumX, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return rtgmc_search_prefilter_clamp_int((int)sumY, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_half_search_base_from_fc(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= max(width >> 1, 1) || y >= max(height >> 1, 1)) return;
    const int value = rtgmc_search_prefilter_half_search_base_from_fc_value<TypePixel>(
        fc, fcPitch, fcHx, fcHy, width, height, x, y);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_half_search_smoothed_from_fc(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= max(width >> 1, 1) || y >= max(height >> 1, 1)) return;
    const int value = rtgmc_search_prefilter_half_search_smoothed_from_fc_value<TypePixel>(
        fc, fcPitch, fcHx, fcHy, width, height, x, y);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_luma_from_fc(
    const uint8_t *fc, const int fcPitch, const int fcHx, const int fcHy,
    uint8_t *dst, const int dstPitch, const int width, const int height,
    const int search_refine, const int fullRangeMode) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    const int value = (search_refine >= 1)
        ? rtgmc_search_prefilter_half_resolution_search_from_fc_value<TypePixel>(fc, fcPitch, fcHx, fcHy, width, height, x, y)
        : rtgmc_search_prefilter_repair_plane_load<TypePixel>(fc, fcPitch, fcHx, fcHy, x, y);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, x, y,
        rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode));
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_copy_from_plane(
    const uint8_t *src, const int srcPitch, const int srcHx, const int srcHy,
    uint8_t *dst, const int dstPitch, const int width, const int height, const int fullRangeMode, const int constantValue) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    const int value = (constantValue >= 0) ? constantValue
        : rtgmc_search_prefilter_repair_plane_load<TypePixel>(src, srcPitch, srcHx, srcHy, x, y);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, x, y,
        rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode));
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_search_smoothed3x3_cuda(
    const uint8_t *src, const int pitch, uint8_t *dst, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y,
        rtgmc_search_prefilter_search_smoothed3x3_value<TypePixel>(src, pitch, width, height, x, y));
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_softened_search_blend_cuda(
    const uint8_t *spatialGuide, const uint8_t *motionGuide, uint8_t *dst,
    const int pitch, const int width, const int height, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(spatialGuide, pitch, width, height, x, y);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(motionGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_value<TypePixel>(spatialGuideValue, motionGuideValue);
    value = rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_softened_search_blend_stabilized_cuda(
    const uint8_t *spatialGuide, const uint8_t *motionGuide, const uint8_t *fieldGuide, uint8_t *dst,
    const int pitch, const int width, const int height, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(spatialGuide, pitch, width, height, x, y);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(motionGuide, pitch, width, height, x, y);
    const int fieldGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(fieldGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_stabilized_value<TypePixel>(spatialGuideValue, motionGuideValue, fieldGuideValue);
    value = rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_stabilized_search_cuda(
    const uint8_t *motionGuide, const uint8_t *fieldGuide, const uint8_t *spatialGuide, uint8_t *dst,
    const int pitch, const int width, const int height, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(motionGuide, pitch, width, height, x, y);
    const int fieldGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(fieldGuide, pitch, width, height, x, y);
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(spatialGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_stabilize_value<TypePixel>(motionGuideValue, fieldGuideValue, spatialGuideValue);
    value = rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_half_resolution_upsample_cuda(
    const uint8_t *smoothed, const int smoothedPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_half_resolution_upsample_value<TypePixel>(
        smoothed, smoothedPitch, width, height, x, y, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dstPitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_refine2_tile_cuda(
    const uint8_t *motionGuide, const int pitch, uint8_t *dst, const int width, const int height, const int fullRangeMode) {
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int localIndex = ly * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx;
    const int localCount = RTGMC_SEARCH_PREFILTER_BLOCK_X * RTGMC_SEARCH_PREFILTER_BLOCK_Y;
    const int tileW = RTGMC_SEARCH_PREFILTER_BLOCK_X + 8;
    const int tileH = RTGMC_SEARCH_PREFILTER_BLOCK_Y + 8;
    const int groupX = blockIdx.x * RTGMC_SEARCH_PREFILTER_BLOCK_X;
    const int groupY = blockIdx.y * RTGMC_SEARCH_PREFILTER_BLOCK_Y;
    __shared__ int smoothTile[(RTGMC_SEARCH_PREFILTER_BLOCK_X + 8) * (RTGMC_SEARCH_PREFILTER_BLOCK_Y + 8)];
    __shared__ float gaussHTile[(RTGMC_SEARCH_PREFILTER_BLOCK_Y + 8) * RTGMC_SEARCH_PREFILTER_BLOCK_X];
    for (int i = localIndex; i < tileW * tileH; i += localCount) {
        const int tx = i % tileW;
        const int ty = i / tileW;
        const int sx = rtgmc_search_prefilter_clamp_int(groupX + tx - 4, 0, width - 1);
        const int sy = rtgmc_search_prefilter_clamp_int(groupY + ty - 4, 0, height - 1);
        smoothTile[i] = rtgmc_search_prefilter_search_smoothed3x3_value<TypePixel>(motionGuide, pitch, width, height, sx, sy);
    }
    __syncthreads();
    for (int i = localIndex; i < tileH * RTGMC_SEARCH_PREFILTER_BLOCK_X; i += localCount) {
        const int hx = i % RTGMC_SEARCH_PREFILTER_BLOCK_X;
        const int hy = i / RTGMC_SEARCH_PREFILTER_BLOCK_X;
        const int base = hy * tileW + hx;
        const float value =
            (float)smoothTile[base + 0] * RTGMC_SEARCH_REFINE2_GAUSS_W4 +
            (float)smoothTile[base + 1] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
            (float)smoothTile[base + 2] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
            (float)smoothTile[base + 3] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
            (float)smoothTile[base + 4] * RTGMC_SEARCH_REFINE2_GAUSS_W0 +
            (float)smoothTile[base + 5] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
            (float)smoothTile[base + 6] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
            (float)smoothTile[base + 7] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
            (float)smoothTile[base + 8] * RTGMC_SEARCH_REFINE2_GAUSS_W4;
        gaussHTile[i] = value;
    }
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const float blur =
        gaussHTile[(ly + 0) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W4 +
        gaussHTile[(ly + 1) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
        gaussHTile[(ly + 2) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
        gaussHTile[(ly + 3) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
        gaussHTile[(ly + 4) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W0 +
        gaussHTile[(ly + 5) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
        gaussHTile[(ly + 6) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
        gaussHTile[(ly + 7) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
        gaussHTile[(ly + 8) * RTGMC_SEARCH_PREFILTER_BLOCK_X + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W4;
    const int spatialGuideValue = (int)(rtgmc_search_prefilter_clamp_float(blur, 0.0f, (float)rtgmc_search_prefilter_pixel_max<TypePixel>()) + 0.5f);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load<TypePixel>(motionGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_value<TypePixel>(spatialGuideValue, motionGuideValue);
    value = rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_edge_softened_search_cuda(
    const uint8_t *searchSmoothed3x3, const int pitch, uint8_t *dst, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    constexpr float gaussP = 2.0f;
    const int srcFirstX = max(0, x - 4);
    const int srcEndX = min(width - 1, x + 4);
    const int srcFirstY = max(0, y - 4);
    const int srcEndY = min(height - 1, y + 4);

    float sumWeightY = 0.0f;
    float sumY = 0.0f;
    for (int yy = srcFirstY; yy <= srcEndY; yy++) {
        const float wy = exp2f(-(gaussP * 0.1f) * (float)((yy - y) * (yy - y)));
        float sumWeightX = 0.0f;
        float sumX = 0.0f;
        for (int xx = srcFirstX; xx <= srcEndX; xx++) {
            const float wx = exp2f(-(gaussP * 0.1f) * (float)((xx - x) * (xx - x)));
            sumWeightX += wx;
            sumX += wx * (float)rtgmc_search_prefilter_pixel_load<TypePixel>(searchSmoothed3x3, pitch, width, height, xx, yy);
        }
        if (sumWeightX > 0.0f) {
            sumX /= sumWeightX;
        }
        sumWeightY += wy;
        sumY += wy * sumX;
    }
    if (sumWeightY > 0.0f) {
        sumY /= sumWeightY;
    }
    const int value = (int)(rtgmc_search_prefilter_clamp_float(sumY, 0.0f, (float)rtgmc_search_prefilter_pixel_max<TypePixel>()) + 0.5f);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

static dim3 rtgmcSearchPrefilterBlock() {
    return dim3(RTGMC_SEARCH_PREFILTER_BLOCK_X, RTGMC_SEARCH_PREFILTER_BLOCK_Y);
}

static dim3 rtgmcSearchPrefilterGrid(const RGYFrameInfo &frame) {
    return dim3(divCeil(frame.width, RTGMC_SEARCH_PREFILTER_BLOCK_X), divCeil(frame.height, RTGMC_SEARCH_PREFILTER_BLOCK_Y));
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterScenechangeTyped(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(cur);
    kernel_rtgmc_search_prefilter_scenechange_cuda<TypePixel><<<grid, block, 0, stream>>>(
        prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], partial, groupCount, cur.width, cur.height);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRefine2TileTyped(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_refine2_tile_cuda<TypePixel><<<grid, block, 0, stream>>>(
        motionGuide.ptr[0], motionGuide.pitch[0], dst.ptr[0], dst.width, dst.height, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearchTyped(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_edge_softened_search_cuda<TypePixel><<<grid, block, 0, stream>>>(
        searchSmoothed3x3.ptr[0], searchSmoothed3x3.pitch[0], dst.ptr[0], dst.width, dst.height);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3Typed(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_search_smoothed3x3_cuda<TypePixel><<<grid, block, 0, stream>>>(
        src.ptr[0], src.pitch[0], dst.ptr[0], dst.width, dst.height);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendTyped(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_softened_search_blend_cuda<TypePixel><<<grid, block, 0, stream>>>(
        spatialGuide.ptr[0], motionGuide.ptr[0], dst.ptr[0], motionGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedTyped(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_softened_search_blend_stabilized_cuda<TypePixel><<<grid, block, 0, stream>>>(
        spatialGuide.ptr[0], motionGuide.ptr[0], fieldGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterStabilizedSearchTyped(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_stabilized_search_cuda<TypePixel><<<grid, block, 0, stream>>>(
        motionGuide.ptr[0], fieldGuide.ptr[0], spatialGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterHalfResolutionUpsampleTyped(
    const RGYFrameInfo &smoothed, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_half_resolution_upsample_cuda<TypePixel><<<grid, block, 0, stream>>>(
        smoothed.ptr[0], smoothed.pitch[0], dst.ptr[0], dst.pitch[0], dst.width, dst.height, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairTemporalTyped(const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &tc, const RGYFrameInfo &delta, int tr0, const uint32_t *sceneFlags, cudaStream_t stream) {
    if (tc.pitch[0] != delta.pitch[0]) return RGY_ERR_INVALID_PARAM;
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(cur);
    kernel_rtgmc_search_prefilter_repair_temporal<TypePixel><<<grid, block, 0, stream>>>(
        prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], tc.ptr[0], delta.ptr[0], tc.pitch[0], cur.width, cur.height, tr0, sceneFlags);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairStageTyped(const RGYFrameInfo &srcPos, const RGYFrameInfo &srcNeg, int srcHx, int srcHy, const RGYFrameInfo &dstPos, const RGYFrameInfo &dstNeg, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, int stage, const RGYFrameInfo &corePos, const RGYFrameInfo &coreNeg, int coreHx, int coreHy, cudaStream_t stream) {
    if (srcPos.pitch[0] != srcNeg.pitch[0] || dstPos.pitch[0] != dstNeg.pitch[0] || corePos.pitch[0] != coreNeg.pitch[0]) return RGY_ERR_INVALID_PARAM;
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dstPos);
    kernel_rtgmc_search_prefilter_repair_stage<TypePixel><<<grid, block, 0, stream>>>(
        srcPos.ptr[0], srcNeg.ptr[0], srcPos.pitch[0], srcHx, srcHy, dstPos.ptr[0], dstNeg.ptr[0], dstPos.pitch[0], dstHx, dstHy, width, height, repairProfile, stage, corePos.ptr[0], coreNeg.ptr[0], corePos.pitch[0], coreHx, coreHy);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairApplyTyped(const RGYFrameInfo &tc, const RGYFrameInfo &delta, const RGYFrameInfo &gatePos, const RGYFrameInfo &gateNeg, int gateHx, int gateHy, const RGYFrameInfo &dst, int dstHx, int dstHy, int width, int height, uint32_t repairProfile, cudaStream_t stream) {
    if (tc.pitch[0] != delta.pitch[0] || gatePos.pitch[0] != gateNeg.pitch[0]) return RGY_ERR_INVALID_PARAM;
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_repair_apply<TypePixel><<<grid, block, 0, stream>>>(
        tc.ptr[0], delta.ptr[0], tc.pitch[0], gatePos.ptr[0], gateNeg.ptr[0], gatePos.pitch[0], gateHx, gateHy, dst.ptr[0], dst.pitch[0], dstHx, dstHy, width, height, repairProfile);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairCopyTyped(const RGYFrameInfo &src, int srcHx, int srcHy, const RGYFrameInfo &dst, int fullRangeMode, int constantValue, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_copy_from_plane<TypePixel><<<grid, block, 0, stream>>>(
        src.ptr[0], src.pitch[0], srcHx, srcHy, dst.ptr[0], dst.pitch[0], dst.width, dst.height, fullRangeMode, constantValue);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairHalfSearchTyped(bool smoothed, const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &dst, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (smoothed) {
        kernel_rtgmc_search_prefilter_half_search_smoothed_from_fc<TypePixel><<<grid, block, 0, stream>>>(
            fc.ptr[0], fc.pitch[0], fcHx, fcHy, dst.ptr[0], dst.pitch[0], fc.width - 2 * fcHx, fc.height - 2 * fcHy);
    } else {
        kernel_rtgmc_search_prefilter_half_search_base_from_fc<TypePixel><<<grid, block, 0, stream>>>(
            fc.ptr[0], fc.pitch[0], fcHx, fcHy, dst.ptr[0], dst.pitch[0], fc.width - 2 * fcHx, fc.height - 2 * fcHy);
    }
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static RGY_ERR launchRtgmcSearchPrefilterRepairLumaTyped(const RGYFrameInfo &fc, int fcHx, int fcHy, const RGYFrameInfo &cur, const RGYFrameInfo &dst, int searchRefine, int fullRangeMode, cudaStream_t stream) {
    if (dst.width != cur.width || dst.height != cur.height) return RGY_ERR_INVALID_PARAM;
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    kernel_rtgmc_search_prefilter_luma_from_fc<TypePixel><<<grid, block, 0, stream>>>(
        fc.ptr[0], fc.pitch[0], fcHx, fcHy, dst.ptr[0], dst.pitch[0], cur.width, cur.height, searchRefine, fullRangeMode);
    return err_to_rgy(cudaGetLastError());
}

template<typename TypePixel>
static const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterLaunchFuncs() {
    constexpr bool highbit = std::is_same<TypePixel, uint16_t>::value;
    static const NVEncRtgmcSearchPrefilterLaunchFuncs funcs = {
        highbit ? launchRtgmcSearchPrefilterScenechangeU16 : launchRtgmcSearchPrefilterScenechangeU8,
        highbit ? launchRtgmcSearchPrefilterRefine2TileU16 : launchRtgmcSearchPrefilterRefine2TileU8,
        highbit ? launchRtgmcSearchPrefilterEdgeSoftenedSearchU16 : launchRtgmcSearchPrefilterEdgeSoftenedSearchU8,
        highbit ? launchRtgmcSearchPrefilterSearchSmoothed3x3U16 : launchRtgmcSearchPrefilterSearchSmoothed3x3U8,
        highbit ? launchRtgmcSearchPrefilterSoftenedSearchBlendU16 : launchRtgmcSearchPrefilterSoftenedSearchBlendU8,
        highbit ? launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU16 : launchRtgmcSearchPrefilterSoftenedSearchBlendStabilizedU8,
        highbit ? launchRtgmcSearchPrefilterStabilizedSearchU16 : launchRtgmcSearchPrefilterStabilizedSearchU8,
        launchRtgmcSearchPrefilterHalfResolutionUpsampleTyped<TypePixel>,
    };
    return &funcs;
}

#endif // NVENC_RTGMC_SEARCH_PREFILTER_DECLARE_ONLY
