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

#include <cstdint>

#include "rgy_cuda_util.h"
#include "rgy_err.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static constexpr int DEGRAIN_BLOCK_X = 16;
static constexpr int DEGRAIN_BLOCK_Y = 16;

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelMax();
template<>
__device__ __forceinline__ int degrainPixelMax<uint8_t>() { return 255; }
template<>
__device__ __forceinline__ int degrainPixelMax<uint16_t>() { return 65535; }

__device__ __forceinline__ int degrainClampInt(const int value, const int minValue, const int maxValue) {
    return min(max(value, minValue), maxValue);
}

template<typename TypePixel>
__device__ __forceinline__ TypePixel degrainClampPixel(const int value) {
    return (TypePixel)degrainClampInt(value, 0, degrainPixelMax<TypePixel>());
}

template<typename TypePixel>
__device__ __forceinline__ int degrainPixelLoad(
    const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    const int px = degrainClampInt(x, 0, width - 1);
    const int py = degrainClampInt(y, 0, height - 1);
    return (int)(*(const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

template<typename TypePixel>
__device__ __forceinline__ int degrainAnalysisLumaToFullRange(const int value, const int tvRange) {
    if (!tvRange) {
        return value;
    }
    const int pixelMax = degrainPixelMax<TypePixel>();
    if (pixelMax <= 255) {
        const int converted = ((value - 16) * pixelMax + (219 >> 1)) / 219;
        return degrainClampInt(converted, 0, pixelMax);
    }
    const int limitedScale = max((pixelMax + 1) >> 8, 1);
    const int limitedOffset = 16 * limitedScale;
    const int limitedRange = 219 * limitedScale;
    const int delta = value - limitedOffset;
    const int converted = delta + (delta * (pixelMax - limitedRange) + (limitedRange >> 1)) / limitedRange;
    return degrainClampInt(converted, 0, pixelMax);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainTemporalSmoothValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius) {
    int value = degrainPixelLoad<TypePixel>(srcCur, srcCurPitch, srcWidth, srcHeight, px, py);
    if (smoothRadius >= 2) {
        const int sum =
            degrainPixelLoad<TypePixel>(srcPrev2, srcPrev2Pitch, srcWidth, srcHeight, px, py)
          + 4 * degrainPixelLoad<TypePixel>(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 6 * value
          + 4 * degrainPixelLoad<TypePixel>(srcNext, srcNextPitch, srcWidth, srcHeight, px, py)
          + degrainPixelLoad<TypePixel>(srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py);
        value = (sum + 8) >> 4;
    } else if (smoothRadius >= 1) {
        const int sum =
            degrainPixelLoad<TypePixel>(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 2 * value
          + degrainPixelLoad<TypePixel>(srcNext, srcNextPitch, srcWidth, srcHeight, px, py);
        value = (sum + 2) >> 2;
    }
    return value;
}

__device__ __forceinline__ int degrainBlur3x3Weighted(
    const int p00, const int p10, const int p20,
    const int p01, const int p11, const int p21,
    const int p02, const int p12, const int p22) {
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

__device__ __forceinline__ int degrainEdgeSoftenCross(const int left, const int up, const int center, const int down, const int right) {
    return (left + up + 4 * center + down + right + 4) >> 3;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainSearchRefine1Blend(
    const int center, const int blur, const int edgeSoft, const int left, const int up, const int right, const int down) {
    const int edgeScale = max((degrainPixelMax<TypePixel>() + 31) / 32, 1);
    const int edgeStrength = abs(left - right) + abs(up - down) + abs(center - blur);
    const int edgeWeight = degrainClampInt((edgeStrength + (edgeScale >> 1)) / edgeScale, 0, 4);
    return (blur * (4 - edgeWeight) + edgeSoft * edgeWeight + 2) >> 2;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainSearchRefine1Value(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius) {
    const int p00 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius);
    const int p10 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py - 1, smoothRadius);
    const int p20 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius);
    const int p01 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py,     smoothRadius);
    const int p11 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py,     smoothRadius);
    const int p21 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py,     smoothRadius);
    const int p02 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius);
    const int p12 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py + 1, smoothRadius);
    const int p22 = degrainTemporalSmoothValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius);
    const int blur = degrainBlur3x3Weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
    const int edgeSoft = degrainEdgeSoftenCross(p01, p10, p11, p12, p21);
    return degrainSearchRefine1Blend<TypePixel>(p11, blur, edgeSoft, p01, p10, p21, p12);
}

template<typename TypePixel>
__device__ __forceinline__ int degrainAnalysisPrefilterValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius, const int searchRefine) {
    int value = degrainTemporalSmoothValue<TypePixel>(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    if (searchRefine >= 1) {
        value = degrainSearchRefine1Value<TypePixel>(
            srcPrev2, srcPrev2Pitch,
            srcPrev, srcPrevPitch,
            srcCur, srcCurPitch,
            srcNext, srcNextPitch,
            srcNext2, srcNext2Pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius);
    }
    return value;
}

template<typename TypePixel>
__device__ __forceinline__ int degrainRep0RepairValue(
    const uint8_t *srcPrev2, const int srcPrev2Pitch,
    const uint8_t *srcPrev, const int srcPrevPitch,
    const uint8_t *srcCur, const int srcCurPitch,
    const uint8_t *srcNext, const int srcNextPitch,
    const uint8_t *srcNext2, const int srcNext2Pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius, const int searchRefine) {
    const int p0 = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py, smoothRadius, searchRefine);
    const int p1u = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py - 1, smoothRadius, searchRefine);
    const int p1d = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py + 1, smoothRadius, searchRefine);
    const int p2u = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py - 2, smoothRadius, searchRefine);
    const int p2d = degrainAnalysisPrefilterValue<TypePixel>(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py + 2, smoothRadius, searchRefine);

    const int vertAvg = (p1u + p1d + 1) >> 1;
    const int nearDiff = abs(p1u - p1d);
    const int farDiff = abs(p2u - p2d);
    const int centerDiff = abs(p0 - vertAvg);
    const int flatness = nearDiff + (farDiff >> 1);
    const int threshold = max(flatness + max(degrainPixelMax<TypePixel>() / 64, 1), max(degrainPixelMax<TypePixel>() / 32, 1));
    if (centerDiff <= threshold) {
        return p0;
    }

    const int repair = (p0 + 3 * vertAvg + 2) >> 2;
    const int lo = min(min(p1u, p1d), min(p2u, p2d));
    const int hi = max(max(p1u, p1d), max(p2u, p2d));
    return degrainClampInt(repair, lo, hi);
}

template<typename TypePixel>
__global__ void kernel_degrain_temporal_smooth_luma_cuda(
    const uint8_t *prev2, const uint8_t *prev, const uint8_t *cur, const uint8_t *next, const uint8_t *next2,
    const int srcPitch, uint8_t *dst, const int dstPitch, const int width, const int height,
    const int tr0, const int searchRefine, const int rep0, const int tvRange) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    const int analysisValue = (rep0 >= 1)
        ? degrainRep0RepairValue<TypePixel>(prev2, srcPitch, prev, srcPitch, cur, srcPitch, next, srcPitch, next2, srcPitch, width, height, x, y, tr0, searchRefine)
        : degrainAnalysisPrefilterValue<TypePixel>(prev2, srcPitch, prev, srcPitch, cur, srcPitch, next, srcPitch, next2, srcPitch, width, height, x, y, tr0, searchRefine);
    *(TypePixel *)(dst + y * dstPitch + x * (int)sizeof(TypePixel)) =
        degrainClampPixel<TypePixel>(degrainAnalysisLumaToFullRange<TypePixel>(analysisValue, tvRange));
}

RGY_ERR launchNVEncDegrainTemporalSmoothLuma(
    const RGYFrameInfo &prev2, const RGYFrameInfo &prev, const RGYFrameInfo &cur, const RGYFrameInfo &next, const RGYFrameInfo &next2,
    const RGYFrameInfo &dst, const int tr0, const int searchRefine, const int rep0, const int tvRange, cudaStream_t stream) {
    const auto block = dim3(DEGRAIN_BLOCK_X, DEGRAIN_BLOCK_Y);
    const auto grid = dim3(divCeil(dst.width, DEGRAIN_BLOCK_X), divCeil(dst.height, DEGRAIN_BLOCK_Y));
    if (RGY_CSP_BIT_DEPTH[cur.csp] > 8) {
        kernel_degrain_temporal_smooth_luma_cuda<uint16_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
            dst.width, dst.height, tr0, searchRefine, rep0, tvRange);
    } else {
        kernel_degrain_temporal_smooth_luma_cuda<uint8_t><<<grid, block, 0, stream>>>(
            prev2.ptr[0], prev.ptr[0], cur.ptr[0], next.ptr[0], next2.ptr[0], cur.pitch[0], dst.ptr[0], dst.pitch[0],
            dst.width, dst.height, tr0, searchRefine, rep0, tvRange);
    }
    return err_to_rgy(cudaGetLastError());
}
