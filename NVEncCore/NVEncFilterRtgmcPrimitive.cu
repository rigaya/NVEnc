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

#include "convert_csp.h"
#include "NVEncFilterRtgmcPrimitive.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#include <algorithm>
#include <array>
#include <vector>

namespace {
static constexpr int RTGMC_PRIMITIVE_BLOCK_X = 32;
static constexpr int RTGMC_PRIMITIVE_BLOCK_Y = 8;
static constexpr int RTGMC_PRIMITIVE_GAUSS_RADIUS = 4;

static RGY_ERR rtgmcPrimitiveWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
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

static RGY_ERR rtgmcPrimitiveRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
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

template<typename Type>
__device__ int rtgmc_primitive_read_pix(
    const uint8_t *__restrict__ src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(src + y * pitch + x * sizeof(Type)));
}

template<typename Type>
__device__ void rtgmc_primitive_write_pix(
    uint8_t *__restrict__ dst, int x, int y, const int pitch, const int value, const int max_val
) {
    Type *dstPix = (Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

__device__ int rtgmc_primitive_make_diff(const int a, const int b, const int range_half, const int max_val) {
    return clamp(a - b + range_half, 0, max_val);
}

__device__ int rtgmc_primitive_add_diff(const int src, const int diff, const int range_half, const int max_val) {
    return clamp(src + diff - range_half, 0, max_val);
}

__device__ int rtgmc_primitive_add_weighted_diff(const int src, const int diff, const float weight, const int range_half, const int max_val) {
    const float value = fmaf((float)(diff - range_half), weight, (float)src);
    return clamp(__float2int_rn(value), 0, max_val);
}

__device__ int rtgmc_primitive_merge_weighted(const int src0, const int src1, const float weight, const int max_val) {
    const float value = (float)src0 + ((float)src1 - (float)src0) * weight;
    return clamp(__float2int_rn(value), 0, max_val);
}

__device__ float rtgmc_primitive_gauss_weight(const int targetPos, const float srcPos, const float ratioClamped, const float gaussP) {
    const float delta = ((float)targetPos - (srcPos - 0.5f)) * ratioClamped;
    const float x = fabsf(delta);
    if (x > (float)RTGMC_PRIMITIVE_GAUSS_RADIUS) {
        return 0.0f;
    }
    return exp2f(-(gaussP * 0.1f) * x * x);
}

__device__ void rtgmc_primitive_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

__device__ void rtgmc_primitive_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

__device__ void rtgmc_primitive_sort8(int *v) {
    rtgmc_primitive_sort2     (&v[0], &v[1]); rtgmc_primitive_sort2_desc(&v[2], &v[3]); rtgmc_primitive_sort2     (&v[4], &v[5]); rtgmc_primitive_sort2_desc(&v[6], &v[7]);
    rtgmc_primitive_sort2     (&v[0], &v[2]); rtgmc_primitive_sort2     (&v[1], &v[3]); rtgmc_primitive_sort2_desc(&v[4], &v[6]); rtgmc_primitive_sort2_desc(&v[5], &v[7]);
    rtgmc_primitive_sort2     (&v[0], &v[1]); rtgmc_primitive_sort2     (&v[2], &v[3]); rtgmc_primitive_sort2_desc(&v[4], &v[5]); rtgmc_primitive_sort2_desc(&v[6], &v[7]);
    rtgmc_primitive_sort2     (&v[0], &v[4]); rtgmc_primitive_sort2     (&v[1], &v[5]); rtgmc_primitive_sort2     (&v[2], &v[6]); rtgmc_primitive_sort2     (&v[3], &v[7]);
    rtgmc_primitive_sort2     (&v[0], &v[2]); rtgmc_primitive_sort2     (&v[1], &v[3]); rtgmc_primitive_sort2     (&v[4], &v[6]); rtgmc_primitive_sort2     (&v[5], &v[7]);
    rtgmc_primitive_sort2     (&v[0], &v[1]); rtgmc_primitive_sort2     (&v[2], &v[3]); rtgmc_primitive_sort2     (&v[4], &v[5]); rtgmc_primitive_sort2     (&v[6], &v[7]);
}

__device__ void rtgmc_primitive_sort9(int *v) {
    rtgmc_primitive_sort8(v);
    rtgmc_primitive_sort2(&v[7], &v[8]);
    rtgmc_primitive_sort2(&v[6], &v[7]);
    rtgmc_primitive_sort2(&v[5], &v[6]);
    rtgmc_primitive_sort2(&v[4], &v[5]);
    rtgmc_primitive_sort2(&v[3], &v[4]);
    rtgmc_primitive_sort2(&v[2], &v[3]);
    rtgmc_primitive_sort2(&v[1], &v[2]);
    rtgmc_primitive_sort2(&v[0], &v[1]);
}

__device__ int rtgmc_primitive_on_inner_pixel(const int x, const int y, const int width, const int height) {
    return x > 0 && x < width - 1 && y > 0 && y < height - 1;
}

template<typename Type>
__device__ void rtgmc_primitive_gather_ring3(
    int *dst,
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                dst[count++] = rtgmc_primitive_read_pix<Type>(src, x + dx, y + dy, pitch, width, height);
            }
        }
    }
}

template<typename Type>
__device__ void rtgmc_primitive_gather_ref_window_with_center(
    int *dst,
    const uint8_t *__restrict__ ref, const int centerValue,
    const int x, const int y,
    const int refPitch, const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            dst[count++] = (dx == 0 && dy == 0)
                ? centerValue
                : rtgmc_primitive_read_pix<Type>(ref, x + dx, y + dy, refPitch, width, height);
        }
    }
}

template<typename Type>
__device__ int rtgmc_primitive_box3_sum(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int weighted
) {
    int sum = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        const int wy = weighted ? (2 - abs(dy)) : 1;
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            const int wx = weighted ? (2 - abs(dx)) : 1;
            sum += wx * wy * rtgmc_primitive_read_pix<Type>(src, x + dx, y + dy, pitch, width, height);
        }
    }
    return sum;
}

template<typename Type>
__device__ int rtgmc_primitive_rank_clipped_center(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int mode
) {
    const int s = rtgmc_primitive_read_pix<Type>(src, x, y, pitch, width, height);
    if (!rtgmc_primitive_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[8];
    rtgmc_primitive_gather_ring3<Type>(v, src, x, y, pitch, width, height);
    rtgmc_primitive_sort8(v);
    const int n = clamp(mode, 1, 4);
    return clamp(s, v[n - 1], v[8 - n]);
}

template<typename Type>
__device__ int rtgmc_primitive_weighted_box3_center(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int s = rtgmc_primitive_read_pix<Type>(src, x, y, pitch, width, height);
    if (!rtgmc_primitive_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    const int sum = rtgmc_primitive_box3_sum<Type>(src, x, y, pitch, width, height, 1);
    return (sum + 8) >> 4;
}

template<typename Type>
__device__ int rtgmc_primitive_average_box3_center(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int s = rtgmc_primitive_read_pix<Type>(src, x, y, pitch, width, height);
    if (!rtgmc_primitive_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    const int sum = rtgmc_primitive_box3_sum<Type>(src, x, y, pitch, width, height, 0);
    return (sum + 4) / 9;
}

template<typename Type>
__device__ int rtgmc_primitive_removegrain(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int mode
) {
    if (mode == 11 || mode == 12) {
        return rtgmc_primitive_weighted_box3_center<Type>(src, x, y, pitch, width, height);
    }
    if (mode == 20) {
        return rtgmc_primitive_average_box3_center<Type>(src, x, y, pitch, width, height);
    }
    return rtgmc_primitive_rank_clipped_center<Type>(src, x, y, pitch, width, height, mode);
}

template<typename Type>
__device__ int rtgmc_primitive_ref_rank_clipped_center(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height,
    const int mode
) {
    const int s = rtgmc_primitive_read_pix<Type>(src, x, y, srcPitch, width, height);
    if (!rtgmc_primitive_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[9];
    rtgmc_primitive_gather_ref_window_with_center<Type>(v, ref, s, x, y, refPitch, width, height);
    rtgmc_primitive_sort9(v);
    const int n = clamp(mode, 1, 4);
    return clamp(s, v[n - 1], v[9 - n]);
}

template<typename Type>
__device__ int rtgmc_primitive_ref_inner_range_clipped_center(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height
) {
    const int s = rtgmc_primitive_read_pix<Type>(src, x, y, srcPitch, width, height);
    if (!rtgmc_primitive_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[8];
    rtgmc_primitive_gather_ring3<Type>(v, ref, x, y, refPitch, width, height);
    rtgmc_primitive_sort8(v);
    const int c = rtgmc_primitive_read_pix<Type>(ref, x, y, refPitch, width, height);
    const int lo = min(v[1], c);
    const int hi = max(v[6], c);
    return clamp(s, lo, hi);
}

template<typename Type>
__device__ int rtgmc_primitive_repair(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height,
    const int mode
) {
    if (mode == 12) {
        return rtgmc_primitive_ref_inner_range_clipped_center<Type>(src, ref, x, y, srcPitch, refPitch, width, height);
    }
    return rtgmc_primitive_ref_rank_clipped_center<Type>(src, ref, x, y, srcPitch, refPitch, width, height, mode);
}

template<typename Type>
__device__ int rtgmc_primitive_vertical_window5_extreme(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int takeMax
) {
    const int center = rtgmc_primitive_read_pix<Type>(src, x, y, pitch, width, height);
    int value = center;
    for (int offset = -2; offset <= 2; offset++) {
        const int yy = y + offset;
        const int sample = (yy >= 0 && yy < height) ? rtgmc_primitive_read_pix<Type>(src, x, yy, pitch, width, height) : center;
        value = takeMax ? max(value, sample) : min(value, sample);
    }
    return value;
}

template<typename Type>
__device__ int rtgmc_primitive_make_diff_from_frames(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height,
    const int range_half, const int max_val
) {
    const int srcValue = rtgmc_primitive_read_pix<Type>(src, x, y, srcPitch, width, height);
    const int refValue = rtgmc_primitive_read_pix<Type>(ref, x, y, refPitch, width, height);
    return rtgmc_primitive_make_diff(srcValue, refValue, range_half, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_copy(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width,
    const int height,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_gauss_h(
    uint8_t *__restrict__ pTmp, const int tmpPitch, const int tmpWidth, const int tmpHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX,
    const float gaussP
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= tmpWidth || iy >= tmpHeight) return;

    const float ratioInvX = 1.0f / ratioX;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = (float)RTGMC_PRIMITIVE_GAUSS_RADIUS / ratioClampedX;
    const float srcX = ((float)ix + 0.5f) * ratioInvX;
    const int srcFirstX = max(0, (int)floorf(srcX - srcWindowX));
    const int srcEndX = min(srcWidth - 1, (int)ceilf(srcX + srcWindowX));

    float clr = 0.0f;
    float sumWeight = 0.0f;
    const Type *__restrict__ srcPtr = (const Type *)(pSrc + iy * srcPitch + srcFirstX * sizeof(Type));
    for (int i = srcFirstX; i <= srcEndX; i++, srcPtr++) {
        const float wx = rtgmc_primitive_gauss_weight(i, srcX, ratioClampedX, gaussP);
        sumWeight += wx;
        clr += (float)srcPtr[0] * wx;
    }
    if (sumWeight > 0.0f) {
        clr /= sumWeight;
    }
    ((float *)(pTmp + iy * tmpPitch) + ix)[0] = clr;
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_gauss_v(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pTmp, const int tmpPitch, const int tmpWidth, const int tmpHeight,
    const float ratioY,
    const float gaussP,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) return;

    const float ratioInvY = 1.0f / ratioY;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = (float)RTGMC_PRIMITIVE_GAUSS_RADIUS / ratioClampedY;
    const float srcY = ((float)iy + 0.5f) * ratioInvY;
    const int srcFirstY = max(0, (int)floorf(srcY - srcWindowY));
    const int srcEndY = min(tmpHeight - 1, (int)ceilf(srcY + srcWindowY));

    float clr = 0.0f;
    float sumWeight = 0.0f;
    for (int j = srcFirstY; j <= srcEndY; j++) {
        const float wy = rtgmc_primitive_gauss_weight(j, srcY, ratioClampedY, gaussP);
        sumWeight += wy;
        clr += ((const float *)(pTmp + j * tmpPitch) + ix)[0] * wy;
    }
    if (sumWeight > 0.0f) {
        clr /= sumWeight;
    }
    Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)clamp(clr + 0.5f, 0.0f, (float)max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_makediff(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, rtgmc_primitive_make_diff(src, ref, range_half, max_val), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_makediff_removegrain20(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int value = rtgmc_primitive_make_diff_from_frames<Type>(pSrc, pRef, ix, iy, srcPitch, refPitch, width, height, range_half, max_val);
    if (ix > 0 && ix < width - 1 && iy > 0 && iy < height - 1) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += rtgmc_primitive_make_diff_from_frames<Type>(pSrc, pRef, ix + dx, iy + dy, srcPitch, refPitch, width, height, range_half, max_val);
            }
        }
        value = (sum + 4) / 9;
    }
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_makediff_removegrain20_adddiff(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int diff = rtgmc_primitive_make_diff_from_frames<Type>(pSrc, pRef, ix, iy, srcPitch, refPitch, width, height, range_half, max_val);
    if (ix > 0 && ix < width - 1 && iy > 0 && iy < height - 1) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += rtgmc_primitive_make_diff_from_frames<Type>(pSrc, pRef, ix + dx, iy + dy, srcPitch, refPitch, width, height, range_half, max_val);
            }
        }
        diff = (sum + 4) / 9;
    }
    const int base = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, rtgmc_primitive_add_diff(base, diff, range_half, max_val), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_adddiff(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int diff = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, rtgmc_primitive_add_diff(src, diff, range_half, max_val), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_addweighteddiff(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const float weight,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int diff = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, rtgmc_primitive_add_weighted_diff(src, diff, weight, range_half, max_val), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_removegrain(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_primitive_removegrain<Type>(pSrc, ix, iy, srcPitch, width, height, mode);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_repair(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_primitive_repair<Type>(pSrc, pRef, ix, iy, srcPitch, refPitch, width, height, mode);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_merge(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const float weight,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, rtgmc_primitive_merge_weighted(src, ref, weight, max_val), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_vertical_min5(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_primitive_vertical_window5_extreme<Type>(pSrc, ix, iy, srcPitch, width, height, 0);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_vertical_max5(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_primitive_vertical_window5_extreme<Type>(pSrc, ix, iy, srcPitch, width, height, 1);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_logicmin(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, min(src, ref), max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_primitive_logicmax(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pRef, const int refPitch,
    const int width,
    const int height,
    const int mode,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_primitive_read_pix<Type>(pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_primitive_read_pix<Type>(pRef, ix, iy, refPitch, width, height);
    rtgmc_primitive_write_pix<Type>(pDst, ix, iy, dstPitch, max(src, ref), max_val);
}

NVEncFilterParamRtgmcPrimitive::NVEncFilterParamRtgmcPrimitive() :
    op(RGYRtgmcPrimitiveOp::Copy),
    refMode(RGYRtgmcPrimitiveRefMode::Disabled),
    mode(0),
    weight(0.5f),
    planes(0x07),
    processChroma(true) {
}

const TCHAR *NVEncFilterRtgmcPrimitive::opToStr(RGYRtgmcPrimitiveOp op) {
    switch (op) {
    case RGYRtgmcPrimitiveOp::Copy:        return _T("copy");
    case RGYRtgmcPrimitiveOp::MakeDiff:    return _T("makediff");
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20: return _T("makediff_removegrain20");
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff: return _T("makediff_removegrain20_adddiff");
    case RGYRtgmcPrimitiveOp::AddDiff:     return _T("adddiff");
    case RGYRtgmcPrimitiveOp::AddWeightedDiff: return _T("addweighteddiff");
    case RGYRtgmcPrimitiveOp::RemoveGrain: return _T("removegrain");
    case RGYRtgmcPrimitiveOp::Repair:      return _T("repair");
    case RGYRtgmcPrimitiveOp::Merge:       return _T("merge");
    case RGYRtgmcPrimitiveOp::GaussResize: return _T("gaussresize");
    case RGYRtgmcPrimitiveOp::VerticalMin5: return _T("verticalmin5");
    case RGYRtgmcPrimitiveOp::VerticalMax5: return _T("verticalmax5");
    case RGYRtgmcPrimitiveOp::LogicMin:    return _T("logicmin");
    case RGYRtgmcPrimitiveOp::LogicMax:    return _T("logicmax");
    default:                               return _T("unknown");
    }
}

const TCHAR *NVEncFilterRtgmcPrimitive::refModeToStr(RGYRtgmcPrimitiveRefMode refMode) {
    switch (refMode) {
    case RGYRtgmcPrimitiveRefMode::Disabled:      return _T("none");
    case RGYRtgmcPrimitiveRefMode::RemoveGrain20: return _T("removegrain20");
    default:                                      return _T("unknown");
    }
}

bool NVEncFilterRtgmcPrimitive::needsRef(RGYRtgmcPrimitiveOp op) {
    switch (op) {
    case RGYRtgmcPrimitiveOp::MakeDiff:
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20:
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff:
    case RGYRtgmcPrimitiveOp::AddDiff:
    case RGYRtgmcPrimitiveOp::AddWeightedDiff:
    case RGYRtgmcPrimitiveOp::Repair:
    case RGYRtgmcPrimitiveOp::Merge:
    case RGYRtgmcPrimitiveOp::LogicMin:
    case RGYRtgmcPrimitiveOp::LogicMax:
        return true;
    default:
        return false;
    }
}

tstring NVEncFilterParamRtgmcPrimitive::print() const {
    return strsprintf(_T("rtgmc-primitive: op=%s ref=%s mode=%d weight=%.3f planes=0x%x chroma=%s"),
        NVEncFilterRtgmcPrimitive::opToStr(op), NVEncFilterRtgmcPrimitive::refModeToStr(refMode), mode, weight, planes,
        processChroma ? _T("true") : _T("false"));
}

NVEncFilterRtgmcPrimitive::NVEncFilterRtgmcPrimitive() :
    NVEncFilter(),
    m_buildOptions(),
    m_useKernel(false) {
    m_name = _T("rtgmc-primitive");
}

NVEncFilterRtgmcPrimitive::~NVEncFilterRtgmcPrimitive() {
    close();
}

RGY_ERR NVEncFilterRtgmcPrimitive::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcPrimitive> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->op < RGYRtgmcPrimitiveOp::Copy || prm->op > RGYRtgmcPrimitiveOp::LogicMax) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive: unsupported op.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->refMode < RGYRtgmcPrimitiveRefMode::Disabled || prm->refMode > RGYRtgmcPrimitiveRefMode::RemoveGrain20) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive: unsupported ref mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!needsRef(prm->op) && prm->refMode != RGYRtgmcPrimitiveRefMode::Disabled) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive ref=%s is ignored for op=%s.\n"), refModeToStr(prm->refMode), opToStr(prm->op));
    }
    if (prm->op == RGYRtgmcPrimitiveOp::RemoveGrain) {
        if (!((prm->mode >= 1 && prm->mode <= 4) || prm->mode == 11 || prm->mode == 12 || prm->mode == 20)) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive removegrain mode supports 1-4, 11, 12 and 20.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (prm->op == RGYRtgmcPrimitiveOp::Repair) {
        if (!((prm->mode >= 1 && prm->mode <= 4) || prm->mode == 12)) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive repair mode supports 1-4 and 12.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (prm->op == RGYRtgmcPrimitiveOp::Merge) {
        if (prm->weight < 0.0f || prm->weight > 1.0f) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive merge weight should be 0.0-1.0; clamped.\n"));
            prm->weight = clamp(prm->weight, 0.0f, 1.0f);
        }
    } else if (prm->op == RGYRtgmcPrimitiveOp::AddWeightedDiff) {
        if (prm->weight < -1.0f || prm->weight > 1.0f) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive addweighteddiff weight should be -1.0-1.0; clamped.\n"));
            prm->weight = clamp(prm->weight, -1.0f, 1.0f);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcPrimitive> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d -D rtgmc_primitive_block_x=%d -D rtgmc_primitive_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf,
        RTGMC_PRIMITIVE_BLOCK_X,
        RTGMC_PRIMITIVE_BLOCK_Y);
    AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernel for rtgmc-primitive: %s\n"),
        char_to_tstring(m_buildOptions).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::setupGaussResize(const NVEncFilterParamRtgmcPrimitive &prm) {
    if (prm.op != RGYRtgmcPrimitiveOp::GaussResize) {
        for (auto& tmp : m_gaussTmp) {
            tmp.reset();
        }
        return RGY_ERR_NONE;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcPrimitive>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcPrimitive>(m_param);
    if (m_useKernel
        && (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-primitive kernel.\n"));
            return sts;
        }
    }

    const int frameBufCount = (needsRef(prm->op) && prm->refMode != RGYRtgmcPrimitiveRefMode::Disabled) ? 2 : 1;
    sts = AllocFrameBuf(prm->frameOut, frameBufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = setupGaussResize(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

bool NVEncFilterRtgmcPrimitive::processPlane(int iplane, const NVEncFilterParamRtgmcPrimitive &prm) const {
    return ((prm.planes & (1 << iplane)) != 0) && (iplane == 0 || prm.processChroma);
}

RGYFrameInfo *NVEncFilterRtgmcPrimitive::generatedRefFrame() {
    return (m_frameBuf.size() >= 2) ? &m_frameBuf[1]->frame : nullptr;
}

RGY_ERR NVEncFilterRtgmcPrimitive::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcPrimitive &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitDepth >= 16) ? ((1 << 16) - 1) : ((1 << bitDepth) - 1);
    const int rangeHalf = 1 << (bitDepth - 1);

    auto launchKernel = [&](RGYRtgmcPrimitiveOp op, const char *kernelName, int iplane, const std::vector<RGYCudaEvent> &wait, RGYCudaEvent *ev) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        const auto refPlane = pRefFrame ? getPlane(pRefFrame, (RGY_PLANE)iplane) : RGYFrameInfo();
        const dim3 blockSize(RTGMC_PRIMITIVE_BLOCK_X, RTGMC_PRIMITIVE_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        auto err = rtgmcPrimitiveWaitEvents(stream, wait);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }

#define LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel, ...) \
        do { \
            if (bitDepth <= 8) { \
                kernel<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } else { \
                kernel<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } \
        } while (0)

        switch (op) {
        case RGYRtgmcPrimitiveOp::Copy:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_copy,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::AddWeightedDiff:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_addweighteddiff,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.weight,
                rangeHalf,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::Merge:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_merge,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.weight,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::MakeDiff:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_makediff,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                rangeHalf,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_makediff_removegrain20,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                rangeHalf,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_makediff_removegrain20_adddiff,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                rangeHalf,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::AddDiff:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_adddiff,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                rangeHalf,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::Repair:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_repair,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::LogicMin:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_logicmin,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::LogicMax:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_logicmax,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::RemoveGrain:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_removegrain,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::VerticalMin5:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_vertical_min5,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        case RGYRtgmcPrimitiveOp::VerticalMax5:
            LAUNCH_RTGMC_PRIMITIVE_KERNEL(kernel_rtgmc_primitive_vertical_max5,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode,
                maxVal);
            break;
        default:
            err = RGY_ERR_UNSUPPORTED;
            break;
        }
#undef LAUNCH_RTGMC_PRIMITIVE_KERNEL

        if (err == RGY_ERR_NONE) {
            const auto cudaerr = cudaGetLastError();
            if (cudaerr != cudaSuccess) {
                err = err_to_rgy(cudaerr);
            }
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
        err = rtgmcPrimitiveRecordEvent(stream, ev);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record %s event (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
        }
        return err;
    };

    std::vector<RGYCudaEvent> planeWaitEvents = wait_events;
    RGYCudaEvent planeEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const bool doProcess = processPlane(iplane, prm);
        auto op = doProcess ? prm.op : RGYRtgmcPrimitiveOp::Copy;
        const char *kernelName = "kernel_rtgmc_primitive_copy";
        if (doProcess) {
            switch (op) {
            case RGYRtgmcPrimitiveOp::MakeDiff:    kernelName = "kernel_rtgmc_primitive_makediff"; break;
            case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20: kernelName = "kernel_rtgmc_primitive_makediff_removegrain20"; break;
            case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff: kernelName = "kernel_rtgmc_primitive_makediff_removegrain20_adddiff"; break;
            case RGYRtgmcPrimitiveOp::AddDiff:     kernelName = "kernel_rtgmc_primitive_adddiff"; break;
            case RGYRtgmcPrimitiveOp::AddWeightedDiff: kernelName = "kernel_rtgmc_primitive_addweighteddiff"; break;
            case RGYRtgmcPrimitiveOp::RemoveGrain: kernelName = "kernel_rtgmc_primitive_removegrain"; break;
            case RGYRtgmcPrimitiveOp::Repair:      kernelName = "kernel_rtgmc_primitive_repair"; break;
            case RGYRtgmcPrimitiveOp::Merge:       kernelName = "kernel_rtgmc_primitive_merge"; break;
            case RGYRtgmcPrimitiveOp::VerticalMin5: kernelName = "kernel_rtgmc_primitive_vertical_min5"; break;
            case RGYRtgmcPrimitiveOp::VerticalMax5: kernelName = "kernel_rtgmc_primitive_vertical_max5"; break;
            case RGYRtgmcPrimitiveOp::LogicMin:    kernelName = "kernel_rtgmc_primitive_logicmin"; break;
            case RGYRtgmcPrimitiveOp::LogicMax:    kernelName = "kernel_rtgmc_primitive_logicmax"; break;
                default:                               kernelName = "kernel_rtgmc_primitive_copy"; break;
            }
        }
        auto err = launchKernel(op, kernelName, iplane, planeWaitEvents, (iplane == planes - 1) ? event : &planeEvent);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        planeWaitEvents.clear();
        if (planeEvent() != nullptr) {
            planeWaitEvents.push_back(planeEvent);
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::processGaussResize(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const NVEncFilterParamRtgmcPrimitive &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const float gaussP = clamp((float)prm.mode, 0.1f, 100.0f);
    int processPlanes = 0;
    for (int iplane = 0; iplane < planes; iplane++) {
        processPlanes += processPlane(iplane, prm) ? 1 : 0;
    }
    if (processPlanes == 0) {
        auto copyPrm = prm;
        copyPrm.op = RGYRtgmcPrimitiveOp::Copy;
        return processFrame(pOutputFrame, pInputFrame, nullptr, copyPrm, stream, wait_events, event);
    }

    std::vector<RGYCudaEvent> planeWaitEvents = wait_events;
    RGYCudaEvent planeEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!processPlane(iplane, prm)) {
            continue;
        }
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        RGYFrameInfo tmpInfo(dstPlane.width, srcPlane.height, RGY_CSP_Y_F32, 32);
        auto& tmpFrame = m_gaussTmp[iplane];
        if (!tmpFrame
            || tmpFrame->frame.width != tmpInfo.width
            || tmpFrame->frame.height != tmpInfo.height
            || tmpFrame->frame.csp != tmpInfo.csp) {
            tmpFrame = std::make_unique<CUFrameBuf>(tmpInfo);
            if (!tmpFrame || tmpFrame->alloc() != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-primitive gauss resize tmp frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        const auto tmpPlane = getPlane(&tmpFrame->frame, RGY_PLANE_Y);
        const dim3 blockSize(RTGMC_PRIMITIVE_BLOCK_X, RTGMC_PRIMITIVE_BLOCK_Y);
        const dim3 gridH(divCeil(tmpPlane.width, blockSize.x), divCeil(tmpPlane.height, blockSize.y));
        const dim3 gridV(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        auto err = rtgmcPrimitiveWaitEvents(stream, planeWaitEvents);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for kernel_rtgmc_primitive_gauss_h (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
#define LAUNCH_RTGMC_PRIMITIVE_GAUSS_H(kernel, ...) \
        do { \
            if (bitDepth <= 8) { \
                kernel<uint8_t><<<gridH, blockSize, 0, stream>>>(__VA_ARGS__); \
            } else { \
                kernel<uint16_t><<<gridH, blockSize, 0, stream>>>(__VA_ARGS__); \
            } \
        } while (0)
#define LAUNCH_RTGMC_PRIMITIVE_GAUSS_V(kernel, ...) \
        do { \
            if (bitDepth <= 8) { \
                kernel<uint8_t><<<gridV, blockSize, 0, stream>>>(__VA_ARGS__); \
            } else { \
                kernel<uint16_t><<<gridV, blockSize, 0, stream>>>(__VA_ARGS__); \
            } \
        } while (0)
        LAUNCH_RTGMC_PRIMITIVE_GAUSS_H(kernel_rtgmc_primitive_gauss_h,
            tmpPlane.ptr[0], tmpPlane.pitch[0], tmpPlane.width, tmpPlane.height,
            srcPlane.ptr[0], srcPlane.pitch[0], srcPlane.width, srcPlane.height,
            (float)dstPlane.width / (float)srcPlane.width,
            gaussP);
#undef LAUNCH_RTGMC_PRIMITIVE_GAUSS_H
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_rtgmc_primitive_gauss_h (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        RGYCudaEvent eventH;
        err = rtgmcPrimitiveRecordEvent(stream, &eventH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record kernel_rtgmc_primitive_gauss_h event (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        err = rtgmcPrimitiveWaitEvents(stream, { eventH });
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for kernel_rtgmc_primitive_gauss_v (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        LAUNCH_RTGMC_PRIMITIVE_GAUSS_V(kernel_rtgmc_primitive_gauss_v,
            dstPlane.ptr[0], dstPlane.pitch[0], dstPlane.width, dstPlane.height,
            tmpPlane.ptr[0], tmpPlane.pitch[0], tmpPlane.width, tmpPlane.height,
            (float)dstPlane.height / (float)srcPlane.height,
            gaussP,
            maxVal);
#undef LAUNCH_RTGMC_PRIMITIVE_GAUSS_V
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_rtgmc_primitive_gauss_v (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        planeWaitEvents.clear();
        err = rtgmcPrimitiveRecordEvent(stream, &planeEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record kernel_rtgmc_primitive_gauss_v event (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        if (planeEvent() != nullptr) {
            planeWaitEvents.push_back(planeEvent);
        }
    }

    if (processPlanes != planes) {
        std::vector<RGYCudaEvent> copyWaitEvents = planeWaitEvents;
        RGYCudaEvent copyEvent;
        const int copyPlanes = planes - processPlanes;
        int copiedPlanes = 0;
        for (int iplane = 0; iplane < planes; iplane++) {
            if (processPlane(iplane, prm)) {
                continue;
            }
            copiedPlanes++;
            const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
            const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
            const dim3 blockSize(RTGMC_PRIMITIVE_BLOCK_X, RTGMC_PRIMITIVE_BLOCK_Y);
            const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
            auto err = rtgmcPrimitiveWaitEvents(stream, copyWaitEvents);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error waiting for kernel_rtgmc_primitive_copy (plane %d): %s.\n"),
                    iplane, get_err_mes(err));
                return err;
            }
#define LAUNCH_RTGMC_PRIMITIVE_COPY(kernel, ...) \
            do { \
                if (bitDepth <= 8) { \
                    kernel<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
                } else { \
                    kernel<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
                } \
            } while (0)
            LAUNCH_RTGMC_PRIMITIVE_COPY(kernel_rtgmc_primitive_copy,
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                maxVal);
#undef LAUNCH_RTGMC_PRIMITIVE_COPY
            err = err_to_rgy(cudaGetLastError());
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_rtgmc_primitive_copy (plane %d): %s.\n"),
                    iplane, get_err_mes(err));
                return err;
            }
            copyWaitEvents.clear();
            auto copyOutputEvent = (copiedPlanes == copyPlanes) ? event : &copyEvent;
            err = rtgmcPrimitiveRecordEvent(stream, copyOutputEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to record kernel_rtgmc_primitive_copy event (plane %d): %s.\n"),
                    iplane, get_err_mes(err));
                return err;
            }
            if (copyEvent() != nullptr) {
                copyWaitEvents.push_back(copyEvent);
            }
        }
    } else {
        auto err = rtgmcPrimitiveRecordEvent(stream, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcPrimitive>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const RGYFrameInfo *actualRefFrame = pRefFrame;
    RGYCudaEvent refEvent;
    std::vector<RGYCudaEvent> processWaitEvents = wait_events;
    if (needsRef(prm->op) && (!actualRefFrame || !actualRefFrame->ptr[0])) {
        if (prm->refMode == RGYRtgmcPrimitiveRefMode::RemoveGrain20) {
            auto refFrame = generatedRefFrame();
            if (!refFrame || !refFrame->ptr[0]) {
                AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive ref=%s has no frame buffer.\n"), refModeToStr(prm->refMode));
                return RGY_ERR_UNSUPPORTED;
            }
            auto refPrm = *prm;
            refPrm.op = RGYRtgmcPrimitiveOp::RemoveGrain;
            refPrm.mode = 20;
            refPrm.refMode = RGYRtgmcPrimitiveRefMode::Disabled;
            auto refErr = processFrame(refFrame, pInputFrame, nullptr, refPrm, stream, wait_events, &refEvent);
            if (refErr != RGY_ERR_NONE) {
                return refErr;
            }
            actualRefFrame = refFrame;
            processWaitEvents = { refEvent };
        } else {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires a reference frame.\n"), opToStr(prm->op));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (needsRef(prm->op) && actualRefFrame && actualRefFrame->ptr[0]
        && (pInputFrame->csp != actualRefFrame->csp || pInputFrame->width != actualRefFrame->width || pInputFrame->height != actualRefFrame->height)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive reference frame must match input csp and resolution.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        const auto refMemcpyKind = actualRefFrame ? getCudaMemcpyKind(actualRefFrame->mem_type, m_frameBuf[0]->frame.mem_type) : cudaMemcpyDeviceToDevice;
        if (memcpyKind == cudaMemcpyDeviceToDevice && refMemcpyKind == cudaMemcpyDeviceToDevice) {
            if (prm->op == RGYRtgmcPrimitiveOp::GaussResize) {
                return processGaussResize(&pOutFrame->frame, pInputFrame, *prm, stream, processWaitEvents, event);
            }
            return processFrame(&pOutFrame->frame, pInputFrame, actualRefFrame, *prm, stream, processWaitEvents, event);
        }
        if (prm->op != RGYRtgmcPrimitiveOp::Copy) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires device-to-device CUDA frames.\n"), opToStr(prm->op));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    auto sts = rtgmcPrimitiveWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto copyErr = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    sts = rtgmcPrimitiveRecordEvent(stream, event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcPrimitive::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcPrimitive>(m_param);
    if (prm && needsRef(prm->op) && prm->refMode == RGYRtgmcPrimitiveRefMode::Disabled) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires a reference frame; call two-input run_filter().\n"), opToStr(prm->op));
        return RGY_ERR_UNSUPPORTED;
    }
    return run_filter(pInputFrame, nullptr, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

void NVEncFilterRtgmcPrimitive::close() {
    m_buildOptions.clear();
    for (auto& tmp : m_gaussTmp) {
        tmp.reset();
    }
    m_frameBuf.clear();
    m_useKernel = false;
}
