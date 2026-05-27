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

#include "NVEncFilterRtgmcLossless.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#include <cstdlib>
#include <vector>

namespace {
static constexpr int RTGMC_LOSSLESS_BLOCK_X = 32;
static constexpr int RTGMC_LOSSLESS_BLOCK_Y = 8;

static RGY_ERR rtgmcLosslessWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
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

static RGY_ERR rtgmcLosslessRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
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
__device__ int rtgmc_lossless_read_pix(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch
) {
    return (int)(*(const Type *)(src + y * pitch + x * sizeof(Type)));
}

template<typename Type>
__device__ void rtgmc_lossless_write_pix(
    uint8_t *__restrict__ dst, const int x, const int y, const int pitch, const int value, const int max_val
) {
    Type *dstPix = (Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

template<typename Type>
__device__ int rtgmc_lossless_read_clamped(
    const uint8_t *__restrict__ src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return rtgmc_lossless_read_pix<Type>(src, x, y, pitch);
}

__device__ void rtgmc_lossless_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

__device__ void rtgmc_lossless_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

__device__ void rtgmc_lossless_sort8(int *v) {
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2_desc(&v[2], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[5]); rtgmc_lossless_sort2_desc(&v[6], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[2]); rtgmc_lossless_sort2     (&v[1], &v[3]); rtgmc_lossless_sort2_desc(&v[4], &v[6]); rtgmc_lossless_sort2_desc(&v[5], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2     (&v[2], &v[3]); rtgmc_lossless_sort2_desc(&v[4], &v[5]); rtgmc_lossless_sort2_desc(&v[6], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[4]); rtgmc_lossless_sort2     (&v[1], &v[5]); rtgmc_lossless_sort2     (&v[2], &v[6]); rtgmc_lossless_sort2     (&v[3], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[2]); rtgmc_lossless_sort2     (&v[1], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[6]); rtgmc_lossless_sort2     (&v[5], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2     (&v[2], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[5]); rtgmc_lossless_sort2     (&v[6], &v[7]);
}

__device__ void rtgmc_lossless_sort9(int *v) {
    rtgmc_lossless_sort8(v);
    rtgmc_lossless_sort2(&v[7], &v[8]);
    rtgmc_lossless_sort2(&v[6], &v[7]);
    rtgmc_lossless_sort2(&v[5], &v[6]);
    rtgmc_lossless_sort2(&v[4], &v[5]);
    rtgmc_lossless_sort2(&v[3], &v[4]);
    rtgmc_lossless_sort2(&v[2], &v[3]);
    rtgmc_lossless_sort2(&v[1], &v[2]);
    rtgmc_lossless_sort2(&v[0], &v[1]);
}

__device__ int rtgmc_lossless_make_diff(const int a, const int b, const int range_half, const int max_val) {
    return clamp(a - b + range_half, 0, max_val);
}

__device__ int rtgmc_lossless_median3(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

__device__ int rtgmc_field_restore_select_offset(const int offsetA, const int offsetB) {
    if (offsetA == 0 || offsetB == 0 || ((offsetA < 0) != (offsetB < 0))) {
        return 0;
    }
    return (abs(offsetA) <= abs(offsetB)) ? offsetA : offsetB;
}

__device__ int rtgmc_field_restore_pick_consistent_delta(const int candA, const int candB, const int range_half) {
    const int neutral = range_half;
    const int selectedOffset = rtgmc_field_restore_select_offset(candA - neutral, candB - neutral);
    return neutral + selectedOffset;
}

template<typename Type>
__device__ int rtgmc_field_restore_reference_sample(
    const uint8_t *__restrict__ processed, const int processedPitch,
    const uint8_t *__restrict__ source, const int sourcePitch,
    int x, int y, const int width, const int height,
    const int sourceField
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    if ((y & 1) == sourceField) {
        return rtgmc_lossless_read_pix<Type>(source, x, y, sourcePitch);
    }
    return rtgmc_lossless_read_pix<Type>(processed, x, y, processedPitch);
}

template<typename Type>
__device__ int rtgmc_field_restore_direct_vertical_delta(
    const uint8_t *__restrict__ processed, const int processedPitch,
    const uint8_t *__restrict__ source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField, const int range_half, const int max_val
) {
    const int b = rtgmc_field_restore_reference_sample<Type>(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField);
    if (y <= 0 || y >= height - 1) {
        return range_half;
    }
    const int a = rtgmc_field_restore_reference_sample<Type>(processed, processedPitch, source, sourcePitch, x, y - 1, width, height, sourceField);
    const int c = rtgmc_field_restore_reference_sample<Type>(processed, processedPitch, source, sourcePitch, x, y + 1, width, height, sourceField);
    const int verticalMedian = rtgmc_lossless_median3(a, b, c);
    return rtgmc_lossless_make_diff(b, verticalMedian, range_half, max_val);
}

template<typename Type>
__device__ int rtgmc_field_restore_reference_vertical_median(
    const uint8_t *__restrict__ reference, const int referencePitch,
    const int x, const int y, const int width, const int height
) {
    const int b = rtgmc_lossless_read_clamped<Type>(reference, x, y, referencePitch, width, height);
    if (y <= 0 || y >= height - 1) {
        return b;
    }
    const int a = rtgmc_lossless_read_clamped<Type>(reference, x, y - 1, referencePitch, width, height);
    const int c = rtgmc_lossless_read_clamped<Type>(reference, x, y + 1, referencePitch, width, height);
    return rtgmc_lossless_median3(a, b, c);
}

template<typename Type>
__device__ int rtgmc_field_restore_vertical_delta(
    const uint8_t *__restrict__ reference, const int referencePitch,
    const int x, const int y, const int width, const int height,
    const int range_half, const int max_val
) {
    const int p = rtgmc_lossless_read_clamped<Type>(reference, x, y, referencePitch, width, height);
    const int vm = rtgmc_field_restore_reference_vertical_median<Type>(reference, referencePitch, x, y, width, height);
    return rtgmc_lossless_make_diff(p, vm, range_half, max_val);
}

template<typename Type>
__device__ int rtgmc_field_restore_stabilized_delta(
    const uint8_t *__restrict__ delta, const int deltaPitch,
    const int x, const int y, const int width, const int height,
    const int range_half
) {
    const int b = rtgmc_lossless_read_clamped<Type>(delta, x, y, deltaPitch, width, height);
    int cleaned = b;
    if (y - 2 >= 0 && y + 2 < height) {
        const int a = rtgmc_lossless_read_clamped<Type>(delta, x, y - 2, deltaPitch, width, height);
        const int c = rtgmc_lossless_read_clamped<Type>(delta, x, y + 2, deltaPitch, width, height);
        cleaned = rtgmc_lossless_median3(a, b, c);
    }
    return rtgmc_field_restore_pick_consistent_delta(cleaned, b, range_half);
}

template<typename Type>
__device__ int rtgmc_field_restore_rank_smooth_delta(
    const uint8_t *__restrict__ delta, const int deltaPitch,
    const int x, const int y, const int width, const int height,
    const int range_half
) {
    const int s = rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x, y, width, height, range_half);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[8] = {
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x - 1, y - 2, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x + 0, y - 2, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x + 1, y - 2, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x - 1, y + 0, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x + 1, y + 0, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x - 1, y + 2, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x + 0, y + 2, width, height, range_half),
        rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x + 1, y + 2, width, height, range_half)
    };
    rtgmc_lossless_sort8(v);
    return clamp(s, v[1], v[6]);
}

template<typename Type>
__device__ int rtgmc_field_restore_bounded_delta(
    const uint8_t *__restrict__ delta, const int deltaPitch,
    const int x, const int y, const int width, const int height,
    const int range_half
) {
    const int s = rtgmc_field_restore_stabilized_delta<Type>(delta, deltaPitch, x, y, width, height, range_half);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[9] = {
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x - 1, y - 2, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x + 0, y - 2, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x + 1, y - 2, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x - 1, y + 0, width, height, range_half),
        s,
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x + 1, y + 0, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x - 1, y + 2, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x + 0, y + 2, width, height, range_half),
        rtgmc_field_restore_rank_smooth_delta<Type>(delta, deltaPitch, x + 1, y + 2, width, height, range_half)
    };
    rtgmc_lossless_sort9(v);
    return clamp(s, v[0], v[8]);
}

template<typename Type>
__device__ int rtgmc_field_restore_direct_stabilized_delta(
    const uint8_t *__restrict__ processed, const int processedPitch,
    const uint8_t *__restrict__ source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField, const int range_half, const int max_val
) {
    const int b = rtgmc_field_restore_direct_vertical_delta<Type>(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField, range_half, max_val);
    int cleaned = b;
    if (y - 2 >= 0 && y + 2 < height) {
        const int a = rtgmc_field_restore_direct_vertical_delta<Type>(processed, processedPitch, source, sourcePitch, x, y - 2, width, height, sourceField, range_half, max_val);
        const int c = rtgmc_field_restore_direct_vertical_delta<Type>(processed, processedPitch, source, sourcePitch, x, y + 2, width, height, sourceField, range_half, max_val);
        cleaned = rtgmc_lossless_median3(a, b, c);
    }
    return rtgmc_field_restore_pick_consistent_delta(cleaned, b, range_half);
}

template<typename Type>
__device__ int rtgmc_field_restore_direct_rank_smooth_delta(
    const uint8_t *__restrict__ processed, const int processedPitch,
    const uint8_t *__restrict__ source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField, const int range_half, const int max_val
) {
    const int s = rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField, range_half, max_val);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[8] = {
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x + 0, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y + 0, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y + 0, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y + 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x + 0, y + 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y + 2, width, height, sourceField, range_half, max_val)
    };
    rtgmc_lossless_sort8(v);
    return clamp(s, v[1], v[6]);
}

template<typename Type>
__device__ int rtgmc_field_restore_direct_bounded_delta(
    const uint8_t *__restrict__ processed, const int processedPitch,
    const uint8_t *__restrict__ source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField, const int range_half, const int max_val
) {
    const int s = rtgmc_field_restore_direct_stabilized_delta<Type>(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField, range_half, max_val);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[9] = {
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x + 0, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y - 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y + 0, width, height, sourceField, range_half, max_val),
        s,
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y + 0, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x - 1, y + 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x + 0, y + 2, width, height, sourceField, range_half, max_val),
        rtgmc_field_restore_direct_rank_smooth_delta<Type>(processed, processedPitch, source, sourcePitch, x + 1, y + 2, width, height, sourceField, range_half, max_val)
    };
    rtgmc_lossless_sort9(v);
    return clamp(s, v[0], v[8]);
}

template<typename Type>
__global__ void kernel_rtgmc_lossless_build_reference_frame(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pProcessed, const int processedPitch,
    const uint8_t *__restrict__ pSource, const int sourcePitch,
    const int width,
    const int height,
    const int sourceField,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int value = ((iy & 1) == sourceField)
        ? rtgmc_lossless_read_pix<Type>(pSource, ix, iy, sourcePitch)
        : rtgmc_lossless_read_pix<Type>(pProcessed, ix, iy, processedPitch);
    rtgmc_lossless_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_lossless_build_delta_map(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pReference, const int referencePitch,
    const int width,
    const int height,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int value = rtgmc_field_restore_vertical_delta<Type>(
        pReference, referencePitch, ix, iy, width, height, range_half, max_val);
    rtgmc_lossless_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_lossless_stabilize_delta_map(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pDelta, const int deltaPitch,
    const int width,
    const int height,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int value = rtgmc_field_restore_bounded_delta<Type>(
        pDelta, deltaPitch, ix, iy, width, height, range_half);
    rtgmc_lossless_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_lossless_apply_delta(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pProcessed, const int processedPitch,
    const uint8_t *__restrict__ pSource, const int sourcePitch,
    const uint8_t *__restrict__ pDelta, const int deltaPitch,
    const int width,
    const int height,
    const int sourceField,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int value = 0;
    if ((iy & 1) == sourceField) {
        value = rtgmc_lossless_read_pix<Type>(pSource, ix, iy, sourcePitch);
    } else {
        const int newField = rtgmc_lossless_read_pix<Type>(pProcessed, ix, iy, processedPitch);
        const int delta = rtgmc_lossless_read_pix<Type>(pDelta, ix, iy, deltaPitch);
        value = rtgmc_lossless_make_diff(newField, delta, range_half, max_val);
    }
    rtgmc_lossless_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

template<typename Type>
__global__ void kernel_rtgmc_lossless_apply_direct_delta(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pProcessed, const int processedPitch,
    const uint8_t *__restrict__ pSource, const int sourcePitch,
    const int width,
    const int height,
    const int sourceField,
    const int range_half,
    const int max_val
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int value = 0;
    if ((iy & 1) == sourceField) {
        value = rtgmc_lossless_read_pix<Type>(pSource, ix, iy, sourcePitch);
    } else {
        const int newField = rtgmc_lossless_read_pix<Type>(pProcessed, ix, iy, processedPitch);
        const int delta = rtgmc_field_restore_direct_bounded_delta<Type>(pProcessed, processedPitch, pSource, sourcePitch, ix, iy, width, height, sourceField, range_half, max_val);
        value = rtgmc_lossless_make_diff(newField, delta, range_half, max_val);
    }
    rtgmc_lossless_write_pix<Type>(pDst, ix, iy, dstPitch, value, max_val);
}

tstring NVEncFilterParamRtgmcLossless::print() const {
    return strsprintf(_T("rtgmc-lossless: level=%d input_type=%d source_field=%d"),
        level, inputType, sourceField);
}

NVEncFilterRtgmcLossless::NVEncFilterRtgmcLossless() :
    NVEncFilter(),
    m_buildOptions(),
    m_useKernel(false) {
    m_name = _T("rtgmc-lossless");
}

NVEncFilterRtgmcLossless::~NVEncFilterRtgmcLossless() {
    close();
}

RGY_ERR NVEncFilterRtgmcLossless::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcLossless> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->level != 1 && prm->level != 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless level must be 1 or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->inputType == 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless is incompatible with inputType=1.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->inputType < 0 || prm->inputType > 3) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless inputType must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->sourceField != 0 && prm->sourceField != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless sourceField must be 0(top/even) or 1(bottom/odd).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcLossless> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D max_val=%d -D range_half=%d -D rtgmc_lossless_block_x=%d -D rtgmc_lossless_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        pixelMax,
        rangeHalf,
        RTGMC_LOSSLESS_BLOCK_X,
        RTGMC_LOSSLESS_BLOCK_Y);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcLossless>(pParam);
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

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcLossless>(m_param);
    if (m_useKernel
        && (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernel(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-lossless kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 5);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::processFrameFused(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const char *kernelName = "kernel_rtgmc_lossless_apply_direct_delta";
    const int planes = RGY_CSP_PLANES[pProcessedFrame->csp];
    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitDepth >= 16) ? ((1 << 16) - 1) : ((1 << bitDepth) - 1);
    const int rangeHalf = 1 << (bitDepth - 1);
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto processedPlane = getPlane(pProcessedFrame, (RGY_PLANE)iplane);
        const auto sourcePlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);

        const dim3 blockSize(RTGMC_LOSSLESS_BLOCK_X, RTGMC_LOSSLESS_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        const auto errWait = rtgmcLosslessWaitEvents(stream, (iplane == 0) ? wait_events : std::vector<RGYCudaEvent>());
        if (errWait != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(errWait));
            return errWait;
        }
        if (bitDepth <= 8) {
            kernel_rtgmc_lossless_apply_direct_delta<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                dstPlane.ptr[0], dstPlane.pitch[0],
                processedPlane.ptr[0], processedPlane.pitch[0],
                sourcePlane.ptr[0], sourcePlane.pitch[0],
                dstPlane.width, dstPlane.height,
                sourceField,
                rangeHalf,
                maxVal);
        } else {
            kernel_rtgmc_lossless_apply_direct_delta<uint16_t><<<gridSize, blockSize, 0, stream>>>(
                dstPlane.ptr[0], dstPlane.pitch[0],
                processedPlane.ptr[0], processedPlane.pitch[0],
                sourcePlane.ptr[0], sourcePlane.pitch[0],
                dstPlane.width, dstPlane.height,
                sourceField,
                rangeHalf,
                maxVal);
        }
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    auto err = rtgmcLosslessRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record %s event: %s.\n"),
            char_to_tstring(kernelName).c_str(), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(pOutputFrame, pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::processFramePassSplit(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto *pReferenceFrame = &m_frameBuf[3]->frame;
    auto *pDeltaFrame = &m_frameBuf[4]->frame;
    const int planes = RGY_CSP_PLANES[pProcessedFrame->csp];
    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitDepth >= 16) ? ((1 << 16) - 1) : ((1 << bitDepth) - 1);
    const int rangeHalf = 1 << (bitDepth - 1);
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto processedPlane = getPlane(pProcessedFrame, (RGY_PLANE)iplane);
        const auto sourcePlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const auto referencePlane = getPlane(pReferenceFrame, (RGY_PLANE)iplane);
        const auto deltaPlane = getPlane(pDeltaFrame, (RGY_PLANE)iplane);

        const dim3 blockSize(RTGMC_LOSSLESS_BLOCK_X, RTGMC_LOSSLESS_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));

        auto err = rtgmcLosslessWaitEvents(stream, (iplane == 0) ? wait_events : std::vector<RGYCudaEvent>());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_build_reference_frame"), iplane, get_err_mes(err));
            return err;
        }

#define LAUNCH_RTGMC_LOSSLESS_KERNEL(kernel, ...) \
        do { \
            if (bitDepth <= 8) { \
                kernel<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } else { \
                kernel<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } \
        } while (0)

        LAUNCH_RTGMC_LOSSLESS_KERNEL(kernel_rtgmc_lossless_build_reference_frame,
            referencePlane.ptr[0], referencePlane.pitch[0],
            processedPlane.ptr[0], processedPlane.pitch[0],
            sourcePlane.ptr[0], sourcePlane.pitch[0],
            referencePlane.width, referencePlane.height,
            sourceField,
            maxVal);
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_build_reference_frame"), iplane, get_err_mes(err));
            return err;
        }

        LAUNCH_RTGMC_LOSSLESS_KERNEL(kernel_rtgmc_lossless_build_delta_map,
            deltaPlane.ptr[0], deltaPlane.pitch[0],
            referencePlane.ptr[0], referencePlane.pitch[0],
            deltaPlane.width, deltaPlane.height,
            rangeHalf,
            maxVal);
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_build_delta_map"), iplane, get_err_mes(err));
            return err;
        }

        LAUNCH_RTGMC_LOSSLESS_KERNEL(kernel_rtgmc_lossless_stabilize_delta_map,
            referencePlane.ptr[0], referencePlane.pitch[0],
            deltaPlane.ptr[0], deltaPlane.pitch[0],
            referencePlane.width, referencePlane.height,
            rangeHalf,
            maxVal);
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_stabilize_delta_map"), iplane, get_err_mes(err));
            return err;
        }

        LAUNCH_RTGMC_LOSSLESS_KERNEL(kernel_rtgmc_lossless_apply_delta,
            dstPlane.ptr[0], dstPlane.pitch[0],
            processedPlane.ptr[0], processedPlane.pitch[0],
            sourcePlane.ptr[0], sourcePlane.pitch[0],
            referencePlane.ptr[0], referencePlane.pitch[0],
            dstPlane.width, dstPlane.height,
            sourceField,
            rangeHalf,
            maxVal);
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_apply_delta"), iplane, get_err_mes(err));
            return err;
        }
#undef LAUNCH_RTGMC_LOSSLESS_KERNEL
    }
    auto err = rtgmcLosslessRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record %s event: %s.\n"),
            _T("kernel_rtgmc_lossless_apply_delta"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(pOutputFrame, pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const char *forcePassSplitEnv = std::getenv("NVENC_RTGMC_LOSSLESS_FORCE_PASS_SPLIT");
    if (forcePassSplitEnv == nullptr || forcePassSplitEnv[0] == '\0') {
        forcePassSplitEnv = std::getenv("QSVENC_RTGMC_LOSSLESS_FORCE_PASS_SPLIT");
    }
    const bool forcePassSplit = forcePassSplitEnv != nullptr && forcePassSplitEnv[0] != '\0' && forcePassSplitEnv[0] != '0';
    if (forcePassSplit) {
        return processFramePassSplit(pOutputFrame, pProcessedFrame, pSourceFrame, sourceField, stream, wait_events, event);
    }
    return processFrameFused(pOutputFrame, pProcessedFrame, pSourceFrame, sourceField, stream, wait_events, event);
}

RGY_ERR NVEncFilterRtgmcLossless::run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pProcessedFrame || !pProcessedFrame->ptr[0] || !pSourceFrame || !pSourceFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (sourceField != 0 && sourceField != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless sourceField must be 0(top/even) or 1(bottom/odd).\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcLossless>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto processedMemcpyKind = getCudaMemcpyKind(pProcessedFrame->mem_type, pOutFrame->frame.mem_type);
        const auto sourceMemcpyKind = getCudaMemcpyKind(pSourceFrame->mem_type, pOutFrame->frame.mem_type);
        if (processedMemcpyKind == cudaMemcpyDeviceToDevice && sourceMemcpyKind == cudaMemcpyDeviceToDevice) {
            return processFrame(&pOutFrame->frame, pProcessedFrame, pSourceFrame, sourceField, stream, wait_events, event);
        }

        auto pProcessedTmp = &m_frameBuf[1]->frame;
        auto pSourceTmp = &m_frameBuf[2]->frame;
        auto sts = rtgmcLosslessWaitEvents(stream, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        auto copyErr = copyFrameAsync(pProcessedTmp, pProcessedFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-lossless processed frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        copyErr = copyFrameAsync(pSourceTmp, pSourceFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-lossless source frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        RGYCudaEvent copyEvent;
        sts = rtgmcLosslessRecordEvent(stream, &copyEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return processFrame(&pOutFrame->frame, pProcessedTmp, pSourceTmp, sourceField, stream, { copyEvent }, event);
    }

    auto sts = rtgmcLosslessWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto copyErr = copyFrameAsync(ppOutputFrames[0], pProcessedFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    sts = rtgmcLosslessRecordEvent(stream, event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcLossless::run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcLossless>(m_param);
    return run_filter(pProcessedFrame, pSourceFrame, prm ? prm->sourceField : 0, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
}

RGY_ERR NVEncFilterRtgmcLossless::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    return run_filter(pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

void NVEncFilterRtgmcLossless::close() {
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
}
