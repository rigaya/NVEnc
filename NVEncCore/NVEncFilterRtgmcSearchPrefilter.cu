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

static constexpr int RTGMC_SEARCH_PREFILTER_BLOCK_X = 16;
static constexpr int RTGMC_SEARCH_PREFILTER_BLOCK_Y = 16;
static constexpr bool RTGMC_SEARCH_PREFILTER_USE_SEARCH_REFINE1_CHAIN = false;
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

__device__ __forceinline__ int rtgmc_search_repair_profile_thin_reject_level(const uint32_t repairProfile) {
    return (int)(repairProfile & 0xffu);
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

__device__ __forceinline__ int rtgmc_search_prefilter_edge_soften_cross(
    const int left, const int up, const int center, const int down, const int right) {
    return (left + up + 4 * center + down + right + 4) >> 3;
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

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_temporal_candidate_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius) {
    if (smoothRadius >= 2) {
        return rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 5);
    }
    if (smoothRadius >= 1) {
        return rtgmc_search_prefilter_temporal_weighted_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, 3);
    }
    return rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px, py);
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_makediff_value(const int ref, const int src) {
    return rtgmc_search_prefilter_clamp_int(ref - src + rtgmc_search_prefilter_range_half<TypePixel>(), 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_adddiff_value(const int src, const int diff) {
    return rtgmc_search_prefilter_clamp_int(src + diff - rtgmc_search_prefilter_range_half<TypePixel>(), 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
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

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_mean3x3_diff_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        const int src = rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
        const int ref = rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px, py);
        return rtgmc_search_prefilter_makediff_value<TypePixel>(ref, src);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int src = rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius);
            const int ref = rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px + ix, py + iy);
            sum += rtgmc_search_prefilter_makediff_value<TypePixel>(ref, src);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ __forceinline__ int rtgmc_search_prefilter_search_correction_delta_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius) {
    const int src = rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
    const int ref = rtgmc_search_prefilter_pixel_load<TypePixel>(srcCur, pitch, srcWidth, srcHeight, px, py);
    return rtgmc_search_prefilter_makediff_value<TypePixel>(ref, src);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_removegrain4_diff_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius) {
    const int s = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int v[8];
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                v[count++] = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(
                    srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + dx, py + dy, smoothRadius);
            }
        }
    }
    rtgmc_search_prefilter_sort8(v);
    return rtgmc_search_prefilter_clamp_int(s, v[3], v[4]);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_vertical_thin_reject_diff_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int radius, const int positive) {
    int value = rtgmc_search_prefilter_polarity_core_seed<TypePixel>(positive);
    for (int iy = -radius; iy <= radius; iy++) {
        const int diff = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py + iy, smoothRadius);
        value = rtgmc_search_prefilter_polarity_core_merge(value, diff, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_vertical_restore_diff_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const int thinRejectRadius, const int restorePaddingRadius, const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
    for (int iy = -restorePaddingRadius; iy <= restorePaddingRadius; iy++) {
        const int diff = rtgmc_search_prefilter_vertical_thin_reject_diff_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py + iy, smoothRadius, thinRejectRadius, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, diff, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_area_envelope_diff_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int diff = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius);
            value = rtgmc_search_prefilter_polarity_envelope_merge(value, diff, positive);
        }
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_thin_core_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    const int thinRejectRadius = 2 + ((rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RGY_RTGMC_REPAIR_THIN_WIDE_CORE) ? 1 : 0);
    return rtgmc_search_prefilter_vertical_thin_reject_diff_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, thinRejectRadius, positive);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_mean3x3_correction_gate_thin_core_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_thin_core_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_thin_core_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    int value = rtgmc_search_prefilter_correction_gate_thin_core_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    if (rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RGY_RTGMC_REPAIR_THIN_CORE_BLEND) {
        const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_thin_core_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
        value = rtgmc_search_prefilter_polarity_core_merge(value, mean3x3, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_rank_limit4_correction_gate_mid_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int v[8] = {
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 0, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 0, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 0, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 0, py + 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius, repairProfile, positive)
    };
    rtgmc_search_prefilter_sort8(v);
    return rtgmc_search_prefilter_clamp_int(s, v[3], v[4]);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_mid_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    if (rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RGY_RTGMC_REPAIR_THIN_RANK_LIMIT) {
        return rtgmc_search_prefilter_rank_limit4_correction_gate_mid_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    return rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_base_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    const int restorePaddingRadius = 2 + ((rtgmc_search_repair_profile_restore_flags(repairProfile) & RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE) ? 1 : 0);
    int value = rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
    for (int iy = -restorePaddingRadius; iy <= restorePaddingRadius; iy++) {
        const int cur = rtgmc_search_prefilter_correction_gate_mid_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py + iy, smoothRadius, repairProfile, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, cur, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_mean3x3_correction_gate_base_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_base_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_base_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive, const int useMax) {
    const int s = rtgmc_search_prefilter_correction_gate_base_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_base_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    return rtgmc_search_prefilter_extreme_merge(s, mean3x3, useMax);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_mean3x3_correction_gate_rank_smooth1_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive, const int useMax) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_rank_smooth1_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_rank_smooth1_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive, useMax);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_rank_smooth2_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive, const int useMax) {
    const int s = rtgmc_search_prefilter_correction_gate_rank_smooth1_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_rank_smooth1_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    return rtgmc_search_prefilter_extreme_merge(s, mean3x3, useMax);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_area_envelope_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive, const int useMax) {
    int value = rtgmc_search_prefilter_extreme_seed<TypePixel>(useMax);
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int cur = rtgmc_search_prefilter_correction_gate_base_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
            value = rtgmc_search_prefilter_extreme_merge(value, cur, useMax);
        }
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_level4_core_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int positive) {
    int value = rtgmc_search_prefilter_polarity_core_seed<TypePixel>(positive);
    for (int iy = -2; iy <= 2; iy++) {
        const int sampleY = ((py + iy) < 0 || (py + iy) >= srcHeight) ? py : (py + iy);
        const int diff = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, sampleY, smoothRadius);
        value = rtgmc_search_prefilter_polarity_core_merge(value, diff, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_level4_mean3x3_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_level4_core_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, positive);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_level4_core_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + ix, py + iy, smoothRadius, positive);
        }
    }
    return (sum + 4) / 9;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_level4_mid_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_level4_core_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, positive);
    const int mean3x3 = rtgmc_search_prefilter_correction_gate_level4_mean3x3_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, positive);
    return rtgmc_search_prefilter_polarity_core_merge(s, mean3x3, positive);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_level4_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed<TypePixel>(positive);
    for (int iy = -2; iy <= 2; iy++) {
        const int sampleY = ((py + iy) < 0 || (py + iy) >= srcHeight) ? py : (py + iy);
        const int cur = rtgmc_search_prefilter_correction_gate_level4_mid_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, sampleY, smoothRadius, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, cur, positive);
    }
    return value;
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_correction_gate_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius,
    const uint32_t repairProfile, const int positive) {
    const int restorePaddingLevel = rtgmc_search_repair_profile_restore_padding_level(repairProfile);
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) & RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH) {
        return rtgmc_search_prefilter_correction_gate_level4_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, positive);
    }
    switch (restorePaddingLevel) {
    case 0:
        return rtgmc_search_prefilter_correction_gate_base_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    case 1:
        return rtgmc_search_prefilter_correction_gate_rank_smooth1_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    case 2:
        return rtgmc_search_prefilter_correction_gate_rank_smooth2_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    default:
        return rtgmc_search_prefilter_correction_gate_area_envelope_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    }
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
__device__ int rtgmc_search_prefilter_field_corrected_search_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py, const int smoothRadius, const uint32_t repairProfile) {
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) == 0) {
        return rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
    }
    const int base = rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
    const int diff = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius);
    const int positiveMask = rtgmc_search_prefilter_correction_gate_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, 1);
    const int negativeMask = rtgmc_search_prefilter_correction_gate_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px, py, smoothRadius, repairProfile, 0);
    const int rangeHalf = rtgmc_search_prefilter_range_half<TypePixel>();
    return rtgmc_search_prefilter_apply_signed_correction<TypePixel>(
        base, diff - rangeHalf, positiveMask - rangeHalf, negativeMask - rangeHalf, rtgmc_search_prefilter_range_scale<TypePixel>());
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_half_search_base_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int hx, const int hy,
    const int smoothRadius, const uint32_t repairProfile) {
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
            const int sample = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                startX + ix, startY + iy, smoothRadius, repairProfile);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = rtgmc_search_prefilter_clamp_int((int)sumX, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return rtgmc_search_prefilter_clamp_int((int)sumY, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_half_search_smoothed_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int hx, const int hy,
    const int smoothRadius, const uint32_t repairProfile) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    if (hx <= 0 || hy <= 0 || hx >= halfWidth - 1 || hy >= halfHeight - 1) {
        return rtgmc_search_prefilter_half_search_base_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight,
            rtgmc_search_prefilter_clamp_int(hx, 0, halfWidth - 1),
            rtgmc_search_prefilter_clamp_int(hy, 0, halfHeight - 1),
            smoothRadius, repairProfile);
    }
    const int x0 = rtgmc_search_prefilter_clamp_int(hx - 1, 0, halfWidth - 1);
    const int x1 = rtgmc_search_prefilter_clamp_int(hx,     0, halfWidth - 1);
    const int x2 = rtgmc_search_prefilter_clamp_int(hx + 1, 0, halfWidth - 1);
    const int y0 = rtgmc_search_prefilter_clamp_int(hy - 1, 0, halfHeight - 1);
    const int y1 = rtgmc_search_prefilter_clamp_int(hy,     0, halfHeight - 1);
    const int y2 = rtgmc_search_prefilter_clamp_int(hy + 1, 0, halfHeight - 1);
    const int p00 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y0, smoothRadius, repairProfile);
    const int p10 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y0, smoothRadius, repairProfile);
    const int p20 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y0, smoothRadius, repairProfile);
    const int p01 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y1, smoothRadius, repairProfile);
    const int p11 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y1, smoothRadius, repairProfile);
    const int p21 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y1, smoothRadius, repairProfile);
    const int p02 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y2, smoothRadius, repairProfile);
    const int p12 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y2, smoothRadius, repairProfile);
    const int p22 = rtgmc_search_prefilter_half_search_base_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y2, smoothRadius, repairProfile);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_half_resolution_search_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py,
    const int smoothRadius, const uint32_t repairProfile) {
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
            const int sample = rtgmc_search_prefilter_half_search_smoothed_value<TypePixel>(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                rtgmc_search_prefilter_clamp_int(startX + ix, 0, halfWidth - 1),
                rtgmc_search_prefilter_clamp_int(startY + iy, 0, halfHeight - 1),
                smoothRadius, repairProfile);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = rtgmc_search_prefilter_clamp_int((int)sumX, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return rtgmc_search_prefilter_clamp_int((int)sumY, 0, rtgmc_search_prefilter_pixel_max<TypePixel>());
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py,
    const int smoothRadius, const int searchRefine, const uint32_t repairProfile) {
    if (searchRefine >= 1) {
        return rtgmc_search_prefilter_half_resolution_search_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile);
    }
    return rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile);
}

template<typename TypePixel>
__device__ int rtgmc_search_prefilter_field_corrected_search_weighted3x3_value(
    const uint8_t *srcPrev2, const uint8_t *srcPrev, const uint8_t *srcCur, const uint8_t *srcNext, const uint8_t *srcNext2,
    const int pitch, const int srcWidth, const int srcHeight, const int px, const int py,
    const int smoothRadius, const uint32_t repairProfile) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile);
    }
    const int p00 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius, repairProfile);
    const int p10 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py - 1, smoothRadius, repairProfile);
    const int p20 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius, repairProfile);
    const int p01 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py,     smoothRadius, repairProfile);
    const int p11 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py,     smoothRadius, repairProfile);
    const int p21 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py,     smoothRadius, repairProfile);
    const int p02 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius, repairProfile);
    const int p12 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py + 1, smoothRadius, repairProfile);
    const int p22 = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius, repairProfile);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
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
        return ((value - minY) * pixelMax + (rangeY >> 1)) / rangeY;
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
        partial[groupIndex + groupCount * 0] = sadPrev[0];
        partial[groupIndex + groupCount * 1] = sadNext[0];
        partial[groupIndex + groupCount * 2] = sadPrev2[0];
        partial[groupIndex + groupCount * 3] = sadNext2[0];
    }
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_field_stable_search_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
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
__global__ void kernel_rtgmc_search_prefilter_half_search_base_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int src_pitch, uint8_t *dst, const int dst_pitch, const int width, const int height,
    const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
    if (x >= halfWidth || y >= halfHeight) {
        return;
    }
    const int value = rtgmc_search_prefilter_half_search_base_value<TypePixel>(
        prev2, prev, cur, next, next2, src_pitch, width, height, x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dst_pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_half_search_smoothed_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int src_pitch, uint8_t *dst, const int dst_pitch, const int width, const int height,
    const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
    if (x >= halfWidth || y >= halfHeight) {
        return;
    }
    const int value = rtgmc_search_prefilter_half_search_smoothed_value<TypePixel>(
        prev2, prev, cur, next, next2, src_pitch, width, height, x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, dst_pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_range_convert_cuda(
    uint8_t *dst, const int pitch, const int width, const int height, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_to_full_range<TypePixel>(
        rtgmc_search_prefilter_pixel_load<TypePixel>(dst, pitch, width, height, x, y),
        fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_debug_temporal_candidate_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    (void)repairProfile;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_temporal_candidate_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_debug_field_stable_search_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_field_corrected_search_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_debug_search_correction_delta_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    (void)repairProfile;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_search_correction_delta_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_debug_positive_correction_gate_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_correction_gate_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0, repairProfile, 1);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_debug_negative_correction_gate_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const uint32_t repairProfile) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_correction_gate_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0, repairProfile, 0);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_search_prefilter_luma_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int pitch, uint8_t *dst, const int width, const int height, const int tr0, const int searchRefine,
    const uint32_t repairProfile, const int fullRangeMode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int value = rtgmc_search_prefilter_value<TypePixel>(
        prev2, prev, cur, next, next2, pitch, width, height, x, y, tr0, searchRefine, repairProfile);
    value = rtgmc_search_prefilter_to_full_range<TypePixel>(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store<TypePixel>(dst, pitch, x, y, value);
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

static RGY_ERR launchRtgmcSearchPrefilterScenechange(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    uint32_t *partial, const int groupCount, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(cur);
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        kernel_rtgmc_search_prefilter_scenechange_cuda<uint16_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], partial, groupCount, cur.width, cur.height);
    } else {
        kernel_rtgmc_search_prefilter_scenechange_cuda<uint8_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], partial, groupCount, cur.width, cur.height);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterFieldStable(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const RGYFrameInfo &dst, const int tr0, const uint32_t repairProfile, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        kernel_rtgmc_search_prefilter_field_stable_search_cuda<uint16_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
    } else {
        kernel_rtgmc_search_prefilter_field_stable_search_cuda<uint8_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterLuma(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const RGYFrameInfo &dst, const int tr0, const int searchRefine, const uint32_t repairProfile, const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        kernel_rtgmc_search_prefilter_luma_cuda<uint16_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0],
            dst.width, dst.height, tr0, searchRefine, repairProfile, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_luma_cuda<uint8_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0],
            dst.width, dst.height, tr0, searchRefine, repairProfile, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterRefine2Tile(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[motionGuide.csp] > 8) {
        kernel_rtgmc_search_prefilter_refine2_tile_cuda<uint16_t><<<grid, block, 0, stream>>>(
            motionGuide.ptr[0], motionGuide.pitch[0], dst.ptr[0], dst.width, dst.height, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_refine2_tile_cuda<uint8_t><<<grid, block, 0, stream>>>(
            motionGuide.ptr[0], motionGuide.pitch[0], dst.ptr[0], dst.width, dst.height, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterEdgeSoftenedSearch(
    const RGYFrameInfo &searchSmoothed3x3, const RGYFrameInfo &dst, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[searchSmoothed3x3.csp] > 8) {
        kernel_rtgmc_search_prefilter_edge_softened_search_cuda<uint16_t><<<grid, block, 0, stream>>>(
            searchSmoothed3x3.ptr[0], searchSmoothed3x3.pitch[0], dst.ptr[0], dst.width, dst.height);
    } else {
        kernel_rtgmc_search_prefilter_edge_softened_search_cuda<uint8_t><<<grid, block, 0, stream>>>(
            searchSmoothed3x3.ptr[0], searchSmoothed3x3.pitch[0], dst.ptr[0], dst.width, dst.height);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterSearchSmoothed3x3(
    const RGYFrameInfo &src, const RGYFrameInfo &dst, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[src.csp] > 8) {
        kernel_rtgmc_search_prefilter_search_smoothed3x3_cuda<uint16_t><<<grid, block, 0, stream>>>(
            src.ptr[0], src.pitch[0], dst.ptr[0], dst.width, dst.height);
    } else {
        kernel_rtgmc_search_prefilter_search_smoothed3x3_cuda<uint8_t><<<grid, block, 0, stream>>>(
            src.ptr[0], src.pitch[0], dst.ptr[0], dst.width, dst.height);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlend(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[spatialGuide.csp] > 8) {
        kernel_rtgmc_search_prefilter_softened_search_blend_cuda<uint16_t><<<grid, block, 0, stream>>>(
            spatialGuide.ptr[0], motionGuide.ptr[0], dst.ptr[0], motionGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_softened_search_blend_cuda<uint8_t><<<grid, block, 0, stream>>>(
            spatialGuide.ptr[0], motionGuide.ptr[0], dst.ptr[0], motionGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterSoftenedSearchBlendStabilized(
    const RGYFrameInfo &spatialGuide, const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[spatialGuide.csp] > 8) {
        kernel_rtgmc_search_prefilter_softened_search_blend_stabilized_cuda<uint16_t><<<grid, block, 0, stream>>>(
            spatialGuide.ptr[0], motionGuide.ptr[0], fieldGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_softened_search_blend_stabilized_cuda<uint8_t><<<grid, block, 0, stream>>>(
            spatialGuide.ptr[0], motionGuide.ptr[0], fieldGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterStabilizedSearch(
    const RGYFrameInfo &motionGuide, const RGYFrameInfo &fieldGuide, const RGYFrameInfo &spatialGuide, const RGYFrameInfo &dst,
    const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[motionGuide.csp] > 8) {
        kernel_rtgmc_search_prefilter_stabilized_search_cuda<uint16_t><<<grid, block, 0, stream>>>(
            motionGuide.ptr[0], fieldGuide.ptr[0], spatialGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_stabilized_search_cuda<uint8_t><<<grid, block, 0, stream>>>(
            motionGuide.ptr[0], fieldGuide.ptr[0], spatialGuide.ptr[0], dst.ptr[0], fieldGuide.pitch[0], dst.width, dst.height, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterHalfSearch(
    const bool smoothed, const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const int tr0, const uint32_t repairProfile, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        if (smoothed) {
            kernel_rtgmc_search_prefilter_half_search_smoothed_cuda<uint16_t><<<grid, block, 0, stream>>>(
                prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
                cur.width, cur.height, tr0, repairProfile);
        } else {
            kernel_rtgmc_search_prefilter_half_search_base_cuda<uint16_t><<<grid, block, 0, stream>>>(
                prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
                cur.width, cur.height, tr0, repairProfile);
        }
    } else {
        if (smoothed) {
            kernel_rtgmc_search_prefilter_half_search_smoothed_cuda<uint8_t><<<grid, block, 0, stream>>>(
                prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
                cur.width, cur.height, tr0, repairProfile);
        } else {
            kernel_rtgmc_search_prefilter_half_search_base_cuda<uint8_t><<<grid, block, 0, stream>>>(
                prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
                cur.width, cur.height, tr0, repairProfile);
        }
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterRangeConvert(
    const RGYFrameInfo &dst, const int fullRangeMode, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[dst.csp] > 8) {
        kernel_rtgmc_search_prefilter_range_convert_cuda<uint16_t><<<grid, block, 0, stream>>>(
            dst.ptr[0], dst.pitch[0], dst.width, dst.height, fullRangeMode);
    } else {
        kernel_rtgmc_search_prefilter_range_convert_cuda<uint8_t><<<grid, block, 0, stream>>>(
            dst.ptr[0], dst.pitch[0], dst.width, dst.height, fullRangeMode);
    }
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR launchRtgmcSearchPrefilterDebug(
    const int debugStage, const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur,
    const RGYFrameInfo &next, const RGYFrameInfo &next2, const RGYFrameInfo &dst,
    const int tr0, const uint32_t repairProfile, cudaStream_t stream) {
    const auto block = rtgmcSearchPrefilterBlock();
    const auto grid = rtgmcSearchPrefilterGrid(dst);
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        if (debugStage == 9) {
            kernel_rtgmc_search_prefilter_debug_temporal_candidate_cuda<uint16_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 11) {
            kernel_rtgmc_search_prefilter_debug_search_correction_delta_cuda<uint16_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 12) {
            kernel_rtgmc_search_prefilter_debug_positive_correction_gate_cuda<uint16_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 13) {
            kernel_rtgmc_search_prefilter_debug_negative_correction_gate_cuda<uint16_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else {
            kernel_rtgmc_search_prefilter_debug_field_stable_search_cuda<uint16_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        }
    } else {
        if (debugStage == 9) {
            kernel_rtgmc_search_prefilter_debug_temporal_candidate_cuda<uint8_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 11) {
            kernel_rtgmc_search_prefilter_debug_search_correction_delta_cuda<uint8_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 12) {
            kernel_rtgmc_search_prefilter_debug_positive_correction_gate_cuda<uint8_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else if (debugStage == 13) {
            kernel_rtgmc_search_prefilter_debug_negative_correction_gate_cuda<uint8_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        } else {
            kernel_rtgmc_search_prefilter_debug_field_stable_search_cuda<uint8_t><<<grid, block, 0, stream>>>(prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.width, dst.height, tr0, repairProfile);
        }
    }
    return err_to_rgy(cudaGetLastError());
}

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
    err = launchRtgmcSearchPrefilterScenechange(*prev2, *prev, *cur, *next, *next2, (uint32_t *)pending->partial->ptrDevice, groupCount, stream);
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
    auto pOut = &m_frameBuf[0]->frame;
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
        sts = launchRtgmcSearchPrefilterFieldStable(
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, prm->tr0, repairProfile, stream);
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
        sts = launchRtgmcSearchPrefilterSearchSmoothed3x3(planeHalfSearchBase, planeHalfSearchSmoothed, stream);
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
            sts = launchRtgmcSearchPrefilterRangeConvert(planeDstWork, fullRangeMode, stream);
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
        sts = launchRtgmcSearchPrefilterFieldStable(
            planePrev2Src, planePrevSrc, planeCurSrc, planeNextSrc, planeNext2Src,
            planeMotionGuide, prm->tr0, repairProfile, stream);
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
            sts = launchRtgmcSearchPrefilterRefine2Tile(planeMotionGuide, planeDstSrc, fullRangeMode, stream);
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
        sts = launchRtgmcSearchPrefilterSearchSmoothed3x3(planeMotionGuide, planeSearchSmoothed3x3, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s plane %d: %s.\n"),
                _T("kernel_rtgmc_search_prefilter_search_smoothed3x3"), planeIndex, get_err_mes(sts));
            return sts;
        }

        sts = launchRtgmcSearchPrefilterEdgeSoftenedSearch(planeSearchSmoothed3x3, planeEdgeSoftenedSearch, stream);
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
            sts = launchRtgmcSearchPrefilterSoftenedSearchBlendStabilized(
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
        sts = launchRtgmcSearchPrefilterSoftenedSearchBlend(
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
        sts = launchRtgmcSearchPrefilterStabilizedSearch(
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
        err = launchRtgmcSearchPrefilterLuma(
            planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDst, prm->tr0, prm->searchRefine, repairProfile, prm->tvRange ? 1 : 0, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), _T("kernel_rtgmc_search_prefilter_luma"), get_err_mes(err));
            return err;
        }
    }

    CUFrameBuf *dumpFrame = searchLumaFrame ? searchLumaFrame.get() : m_frameBuf[0].get();
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
        err = launchRtgmcSearchPrefilterHalfSearch(
            m_searchLumaDumpStage == "half_search_smoothed",
            planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDebugDump, prm->tr0, repairProfile, stream);
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
        err = launchRtgmcSearchPrefilterDebug(
            debugStage, planePrev2Eff, planePrevEff, planeCur, planeNextEff, planeNext2Eff,
            planeDebugDump, prm->tr0, repairProfile, stream);
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
                err = launchRtgmcSearchPrefilterLuma(
                    planePrev2CEff, planePrevCEff, planeCurC, planeNextCEff, planeNext2CEff,
                    planeDstC, prm->tr0, prm->searchRefine, repairProfile, prm->tvRange ? 2 : 0, stream);
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
