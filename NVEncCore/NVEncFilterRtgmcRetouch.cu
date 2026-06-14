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

#include "NVEncFilterRtgmcRetouch.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <fstream>
#include <vector>

#include "rgy_cuda_util_kernel.h"
#include "NVEncFilterDegrain.cuh"
#include "rgy_util.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

namespace {
static constexpr int RTGMC_RETOUCH_BLOCK_X = 32;
static constexpr int RTGMC_RETOUCH_BLOCK_Y = 8;
static constexpr int RTGMC_RETOUCH_FRAMEBUF_COUNT = 5;
static constexpr int RTGMC_RETOUCH_BACKBLEND_GAUSS_RADIUS = 4;
static constexpr float RTGMC_RETOUCH_BACKBLEND_GAUSS_P = 5.0f;
static constexpr float RTGMC_RETOUCH_DETAIL_BASE_GAIN = 0.20f;
static constexpr float RTGMC_RETOUCH_DETAIL_TR1_GAIN = 0.15f;
static constexpr float RTGMC_RETOUCH_DETAIL_TR2_GAIN = 0.25f;
static constexpr float RTGMC_RETOUCH_DETAIL_SMODE1_BIAS = 0.10f;
static constexpr float RTGMC_RETOUCH_EDGE_NARROW_GAIN_SCALE = 6.0f;

float calcRtgmcDetailGain(const VppRtgmcRetouch& retouch) {
    float limitModeBoost = 1.0f;
    if (retouch.slmode == 2 || retouch.slmode == 4) {
        limitModeBoost = 2.0f;
    } else if (retouch.slmode == 1 || retouch.slmode == 3) {
        limitModeBoost = 1.5f;
    }

    float radiusGain = RTGMC_RETOUCH_DETAIL_BASE_GAIN;
    radiusGain += retouch.tr1 * RTGMC_RETOUCH_DETAIL_TR1_GAIN;
    radiusGain += retouch.tr2 * RTGMC_RETOUCH_DETAIL_TR2_GAIN;

    float gain = limitModeBoost * radiusGain;
    if (retouch.smode == 1) {
        gain += RTGMC_RETOUCH_DETAIL_SMODE1_BIAS;
    }
    return retouch.sharpness * gain;
}

float calcRtgmcEdgeNarrowGain(const VppRtgmcRetouch& retouch) {
    return retouch.svthin * RTGMC_RETOUCH_EDGE_NARROW_GAIN_SCALE;
}

bool isFrameCompatible(const RGYFrameInfo *base, const RGYFrameInfo *frame) {
    return base && frame
        && base->csp == frame->csp
        && base->width == frame->width
        && base->height == frame->height
        && base->bitdepth == frame->bitdepth
        && base->mem_type == frame->mem_type;
}

bool isRetouchChromaPlane(const int iplane) {
    return iplane > 0;
}

bool isRtgmcRetouchDumpStageSupported(const std::string& stage) {
    return stage == "input"
        || stage == "detail_boost_edge_ref"
        || stage == "detail_boost_regularized_ref"
        || stage == "detail_boost_blur_ref"
        || stage == "detail_boost"
        || stage == "prelimit_rollback"
        || stage == "prelimit_rollback_delta"
        || stage == "prelimit_rollback_smooth_delta"
        || stage == "prelimit_rollback_soft_delta"
        || stage == "spatial_guard"
        || stage == "postlimit_rollback"
        || stage == "postlimit_rollback_delta"
        || stage == "postlimit_rollback_smooth_delta"
        || stage == "postlimit_rollback_soft_delta"
        || stage == "edge_narrow_delta"
        || stage == "edge_narrow_blur_delta"
        || stage == "edge_narrow_guard_delta"
        || stage == "edge_narrow"
        || stage == "postlimit_spatial_guard_src"
        || stage == "postlimit_spatial_guard_ref"
        || stage == "postlimit_spatial_guard"
        || stage == "temporal_guard_src"
        || stage == "temporal_guard_ref"
        || stage == "temporal_guard_motionback"
        || stage == "temporal_guard_motionforw"
        || stage == "temporal_guard"
        || stage == "postlimit_temporal_guard_src"
        || stage == "postlimit_temporal_guard_ref"
        || stage == "postlimit_temporal_guard_motionback"
        || stage == "postlimit_temporal_guard_motionforw"
        || stage == "postlimit_temporal_guard";
}

bool rtgmcRetouchDumpStageChromaReady(const std::string& stage) {
    return stage == "input"
        || (stage.size() >= 4 && stage.compare(stage.size() - 4, 4, "_ref") == 0)
        || (stage.size() >= 11 && stage.compare(stage.size() - 11, 11, "_motionback") == 0)
        || (stage.size() >= 11 && stage.compare(stage.size() - 11, 11, "_motionforw") == 0);
}

const char *rtgmcRetouchDumpTargetForStage(const std::string& stage) {
    if (stage.find("rollback") != std::string::npos) {
        return "rollback";
    }
    if (stage.rfind("postlimit_", 0) == 0) {
        return "postlimit";
    }
    if (stage.rfind("temporal_guard_", 0) == 0 || stage == "temporal_guard") {
        return "limitover";
    }
    if (stage.rfind("edge_narrow", 0) == 0) {
        return "edge_narrow";
    }
    return "retouch";
}

static RGY_ERR rtgmcRetouchWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
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

static RGY_ERR rtgmcRetouchRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
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

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

template<typename Type>
__device__ int rtgmc_retouch_read_pix(
    const uint8_t *__restrict__ src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(src + y * pitch + x * (int)sizeof(Type)));
}

template<typename Type>
__device__ void rtgmc_retouch_write_pix(
    uint8_t *__restrict__ dst, int x, int y, const int pitch, const int value, const int maxVal
) {
    *(Type *)(dst + y * pitch + x * (int)sizeof(Type)) = (Type)clamp(value, 0, maxVal);
}

__device__ int rtgmc_retouch_median3(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

__device__ void rtgmc_retouch_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

__device__ void rtgmc_retouch_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

__device__ void rtgmc_retouch_sort8(int *v) {
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2_desc(&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[6]); rtgmc_retouch_sort2_desc(&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[4]); rtgmc_retouch_sort2     (&v[1], &v[5]); rtgmc_retouch_sort2     (&v[2], &v[6]); rtgmc_retouch_sort2     (&v[3], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[6]); rtgmc_retouch_sort2     (&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2     (&v[6], &v[7]);
}

template<typename Type>
__device__ int rtgmc_retouch_detail_ref_vertical_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int pixCenter = rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    const int pixUpper = (y > 0) ? rtgmc_retouch_read_pix<Type>(src, x, y - 1, pitch, width, height) : pixCenter;
    const int pixLower = (y + 1 < height) ? rtgmc_retouch_read_pix<Type>(src, x, y + 1, pitch, width, height) : pixCenter;
    const int triadSum = pixUpper + pixCenter + pixLower;
    const int pairLowerMin = min(pixUpper, pixCenter);
    const int pairLowerMax = max(pixUpper, pixCenter);
    const int triadMedian = max(pairLowerMin, min(pairLowerMax, pixLower));
    return (triadSum - triadMedian + 1) >> 1;
}

template<typename Type>
__device__ int rtgmc_retouch_removegrain12_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    }
    const int p00 = rtgmc_retouch_read_pix<Type>(src, x - 1, y - 1, pitch, width, height);
    const int p10 = rtgmc_retouch_read_pix<Type>(src, x,     y - 1, pitch, width, height);
    const int p20 = rtgmc_retouch_read_pix<Type>(src, x + 1, y - 1, pitch, width, height);
    const int p01 = rtgmc_retouch_read_pix<Type>(src, x - 1, y,     pitch, width, height);
    const int p11 = rtgmc_retouch_read_pix<Type>(src, x,     y,     pitch, width, height);
    const int p21 = rtgmc_retouch_read_pix<Type>(src, x + 1, y,     pitch, width, height);
    const int p02 = rtgmc_retouch_read_pix<Type>(src, x - 1, y + 1, pitch, width, height);
    const int p12 = rtgmc_retouch_read_pix<Type>(src, x,     y + 1, pitch, width, height);
    const int p22 = rtgmc_retouch_read_pix<Type>(src, x + 1, y + 1, pitch, width, height);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

template<typename Type>
__device__ int rtgmc_retouch_removegrain_smooth_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    return rtgmc_retouch_removegrain12_value<Type>(src, x, y, pitch, width, height);
}

template<typename Type>
__device__ int rtgmc_retouch_verticalcleaner1_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (y <= 0 || y >= height - 1) {
        return rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    }
    const int top = rtgmc_retouch_read_pix<Type>(src, x, y - 1, pitch, width, height);
    const int center = rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    const int bottom = rtgmc_retouch_read_pix<Type>(src, x, y + 1, pitch, width, height);
    return rtgmc_retouch_median3(top, center, bottom);
}

template<typename Type>
__device__ int rtgmc_retouch_blur10h_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int center = rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    const int left = (x > 0) ? rtgmc_retouch_read_pix<Type>(src, x - 1, y, pitch, width, height) : center;
    const int right = (x + 1 < width) ? rtgmc_retouch_read_pix<Type>(src, x + 1, y, pitch, width, height) : center;
    return (left + 2 * center + right + 2) >> 2;
}

__device__ int rtgmc_retouch_precise_clamp_value(const int src, const int ref, const int maxVal) {
    if (src < ref) {
        return min(src + 1, maxVal);
    }
    if (src > ref) {
        return max(src - 1, 0);
    }
    return src;
}

__device__ int rtgmc_retouch_make_diff_value(const int a, const int b, const int rangeHalf, const int maxVal) {
    return clamp(a - b + rangeHalf, 0, maxVal);
}

__device__ int rtgmc_retouch_add_diff_value(const int src, const int diff, const int rangeHalf, const int maxVal) {
    return clamp(src + diff - rangeHalf, 0, maxVal);
}

__device__ int rtgmc_retouch_round_clamp(const float value, const int maxVal) {
    return (int)(clamp(value, 0.0f, (float)maxVal) + 0.5f);
}

template<typename Type>
__device__ void rtgmc_retouch_collect_ref_ring(
    int *dst, const uint8_t *__restrict__ ref,
    const int x, const int y, const int refPitch,
    const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                dst[count++] = rtgmc_retouch_read_pix<Type>(ref, x + dx, y + dy, refPitch, width, height);
            }
        }
    }
}

template<typename Type>
__device__ int rtgmc_retouch_repair_mode1_value(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_read_pix<Type>(src, x, y, srcPitch, width, height);
    }
    const int s = rtgmc_retouch_read_pix<Type>(src, x, y, srcPitch, width, height);
    int minv = s;
    int maxv = s;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                const int sample = rtgmc_retouch_read_pix<Type>(ref, x + dx, y + dy, refPitch, width, height);
                minv = min(minv, sample);
                maxv = max(maxv, sample);
            }
        }
    }
    return clamp(s, minv, maxv);
}

template<typename Type>
__device__ int rtgmc_retouch_repair_mode12_value(
    const uint8_t *__restrict__ src, const uint8_t *__restrict__ ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    const int s = rtgmc_retouch_read_pix<Type>(src, x, y, srcPitch, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return s;
    }
    int v[8];
    rtgmc_retouch_collect_ref_ring<Type>(v, ref, x, y, refPitch, width, height);
    rtgmc_retouch_sort8(v);
    const int c = rtgmc_retouch_read_pix<Type>(ref, x, y, refPitch, width, height);
    const int lo = min(v[1], c);
    const int hi = max(v[6], c);
    return clamp(s, lo, hi);
}

template<typename Type>
__device__ int rtgmc_retouch_removegrain12_diff_value(
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ base, const int basePitch,
    const int x, const int y, const int width, const int height,
    const int rangeHalf, const int maxVal
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_make_diff_value(
            rtgmc_retouch_read_pix<Type>(src, x, y, srcPitch, width, height),
            rtgmc_retouch_read_pix<Type>(base, x, y, basePitch, width, height),
            rangeHalf, maxVal);
    }
    const int p00 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x - 1, y - 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x - 1, y - 1, basePitch, width, height), rangeHalf, maxVal);
    const int p10 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x,     y - 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x,     y - 1, basePitch, width, height), rangeHalf, maxVal);
    const int p20 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x + 1, y - 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x + 1, y - 1, basePitch, width, height), rangeHalf, maxVal);
    const int p01 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x - 1, y,     srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x - 1, y,     basePitch, width, height), rangeHalf, maxVal);
    const int p11 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x,     y,     srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x,     y,     basePitch, width, height), rangeHalf, maxVal);
    const int p21 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x + 1, y,     srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x + 1, y,     basePitch, width, height), rangeHalf, maxVal);
    const int p02 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x - 1, y + 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x - 1, y + 1, basePitch, width, height), rangeHalf, maxVal);
    const int p12 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x,     y + 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x,     y + 1, basePitch, width, height), rangeHalf, maxVal);
    const int p22 = rtgmc_retouch_make_diff_value(rtgmc_retouch_read_pix<Type>(src, x + 1, y + 1, srcPitch, width, height), rtgmc_retouch_read_pix<Type>(base, x + 1, y + 1, basePitch, width, height), rangeHalf, maxVal);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

template<typename Type>
__device__ int rtgmc_retouch_detail_ref_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise, const int maxVal
) {
    const int detailRef = rtgmc_retouch_detail_ref_vertical_value<Type>(src, x, y, pitch, width, height);
    if (precise == 0) {
        return detailRef;
    }
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    return rtgmc_retouch_precise_clamp_value(detailRef, srcPix, maxVal);
}

template<typename Type>
__device__ int rtgmc_retouch_detail_ref_blur_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise, const int maxVal
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_detail_ref_value<Type>(src, x, y, pitch, width, height, precise, maxVal);
    }
    const int p00 = rtgmc_retouch_detail_ref_value<Type>(src, x - 1, y - 1, pitch, width, height, precise, maxVal);
    const int p10 = rtgmc_retouch_detail_ref_value<Type>(src, x,     y - 1, pitch, width, height, precise, maxVal);
    const int p20 = rtgmc_retouch_detail_ref_value<Type>(src, x + 1, y - 1, pitch, width, height, precise, maxVal);
    const int p01 = rtgmc_retouch_detail_ref_value<Type>(src, x - 1, y,     pitch, width, height, precise, maxVal);
    const int p11 = rtgmc_retouch_detail_ref_value<Type>(src, x,     y,     pitch, width, height, precise, maxVal);
    const int p21 = rtgmc_retouch_detail_ref_value<Type>(src, x + 1, y,     pitch, width, height, precise, maxVal);
    const int p02 = rtgmc_retouch_detail_ref_value<Type>(src, x - 1, y + 1, pitch, width, height, precise, maxVal);
    const int p12 = rtgmc_retouch_detail_ref_value<Type>(src, x,     y + 1, pitch, width, height, precise, maxVal);
    const int p22 = rtgmc_retouch_detail_ref_value<Type>(src, x + 1, y + 1, pitch, width, height, precise, maxVal);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

__device__ int rtgmc_retouch_stronger_non_neutral(const int candidate, const int baseline, const int rangeHalf) {
    const int candidateOffset = candidate - rangeHalf;
    const int baselineOffset = baseline - rangeHalf;
    return (abs(candidateOffset) > abs(baselineOffset)) ? candidate : rangeHalf;
}

template<typename Type>
__device__ int rtgmc_retouch_vertical_balance_delta_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain, const int rangeHalf, const int maxVal
) {
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, x, y, pitch, width, height);
    const int cleaned = rtgmc_retouch_verticalcleaner1_value<Type>(src, x, y, pitch, width, height);
    const float value = fmaf((float)(cleaned - srcPix), edgeNarrowingGain, (float)rangeHalf);
    return rtgmc_retouch_round_clamp(value, maxVal);
}

template<typename Type>
__device__ int rtgmc_retouch_horizontal_balance_delta_value(
    const uint8_t *__restrict__ src, int x, int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain, const int rangeHalf, const int maxVal
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const int center = rtgmc_retouch_vertical_balance_delta_value<Type>(src, x, y, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int left = (x > 0)
        ? rtgmc_retouch_vertical_balance_delta_value<Type>(src, x - 1, y, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal)
        : center;
    const int right = (x + 1 < width)
        ? rtgmc_retouch_vertical_balance_delta_value<Type>(src, x + 1, y, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal)
        : center;
    return (left + 2 * center + right + 2) >> 2;
}

template<typename Type>
__device__ int rtgmc_retouch_area_balance_delta_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain, const int rangeHalf, const int maxVal
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x, y, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    }
    const int p00 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x - 1, y - 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p10 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x,     y - 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p20 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x + 1, y - 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p01 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x - 1, y,     pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p11 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x,     y,     pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p21 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x + 1, y,     pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p02 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x - 1, y + 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p12 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x,     y + 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int p22 = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, x + 1, y + 1, pitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

template<typename Type>
__device__ int rtgmc_retouch_temporal_detail_guard_value(
    const int srcPix,
    const uint8_t *__restrict__ ref,
    const uint8_t *__restrict__ motionBack,
    const uint8_t *__restrict__ motionForw,
    const int x, const int y,
    const int refPitch, const int motionBackPitch, const int motionForwPitch,
    const int width, const int height,
    const int sovs, const int maxVal
) {
    const int refPix = rtgmc_retouch_read_pix<Type>(ref, x, y, refPitch, width, height);
    const int motionBackPix = rtgmc_retouch_read_pix<Type>(motionBack, x, y, motionBackPitch, width, height);
    const int motionForwPix = rtgmc_retouch_read_pix<Type>(motionForw, x, y, motionForwPitch, width, height);
    const int lower = min(refPix, min(motionBackPix, motionForwPix)) - sovs;
    const int upper = max(refPix, max(motionBackPix, motionForwPix)) + sovs;
    return clamp(srcPix, max(0, lower), min(maxVal, upper));
}

template<typename Type>
__device__ int rtgmc_retouch_temporal_detail_guard_value_inline_comp(
    const int srcPix,
    const uint8_t *__restrict__ ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height,
    const RGYDegrainCompensateInlineParams &compParams,
    const int sovs, const int maxVal
) {
    const int refPix = rtgmc_retouch_read_pix<Type>(ref, x, y, refPitch, width, height);
    const int motionBackPix = degrainCompensateOverlapPixelValue<Type>(
        compParams.cur, compParams.cur_pitch,
        compParams.cur, compParams.refBack,
        compParams.refDirBack,
        compParams.width, compParams.height,
        compParams.mv, compParams.sad,
        compParams.blocksX, compParams.blocksY,
        compParams.blockSize, compParams.overlap, compParams.step,
        compParams.coveredWidth, compParams.coveredHeight,
        compParams.planeScaleX, compParams.planeScaleY,
        compParams.thsad, compParams.disableMask,
        compParams.windowRamp,
        x, y,
        compParams.refs, compParams.pel, compParams.subpelInterp);
    const int motionForwPix = degrainCompensateOverlapPixelValue<Type>(
        compParams.cur, compParams.cur_pitch,
        compParams.cur, compParams.refForw,
        compParams.refDirForw,
        compParams.width, compParams.height,
        compParams.mv, compParams.sad,
        compParams.blocksX, compParams.blocksY,
        compParams.blockSize, compParams.overlap, compParams.step,
        compParams.coveredWidth, compParams.coveredHeight,
        compParams.planeScaleX, compParams.planeScaleY,
        compParams.thsad, compParams.disableMask,
        compParams.windowRamp,
        x, y,
        compParams.refs, compParams.pel, compParams.subpelInterp);
    const int lower = min(refPix, min(motionBackPix, motionForwPix)) - sovs;
    const int upper = max(refPix, max(motionBackPix, motionForwPix)) + sovs;
    return clamp(srcPix, max(0, lower), min(maxVal, upper));
}

template<typename Type>
__device__ int rtgmc_retouch_spatial_min(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius, const int maxVal
) {
    int value = maxVal;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = min(value, rtgmc_retouch_read_pix<Type>(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

template<typename Type>
__device__ int rtgmc_retouch_spatial_max(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius
) {
    int value = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = max(value, rtgmc_retouch_read_pix<Type>(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_copy(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_repair1(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ ref, const int refPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode1_value<Type>(src, ref, ix, iy, srcPitch, refPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_repair12(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ ref, const int refPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode12_value<Type>(src, ref, ix, iy, srcPitch, refPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_removegrain12(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value<Type>(src, ix, iy, srcPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_removegrain11(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value<Type>(src, ix, iy, srcPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_detail_ref_vertical(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_detail_ref_vertical_value<Type>(src, ix, iy, srcPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_precise_clamp(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ ref, const int refPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int refPix = rtgmc_retouch_read_pix<Type>(ref, ix, iy, refPitch, width, height);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_precise_clamp_value(srcPix, refPix, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_detail_boost(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ blur, const int blurPitch,
    const int width, const int height,
    const float detailGain, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int blurPix = rtgmc_retouch_read_pix<Type>(blur, ix, iy, blurPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_detail_boost_fused(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value<Type>(src, ix, iy, srcPitch, width, height, precise, maxVal)
        : rtgmc_retouch_removegrain12_value<Type>(src, ix, iy, srcPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_detail_boost_edge_narrow_fused(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain,
    const float edgeNarrowingGain,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value<Type>(src, ix, iy, srcPitch, width, height, precise, maxVal)
        : rtgmc_retouch_removegrain12_value<Type>(src, ix, iy, srcPitch, width, height);
    const float boosted = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    const int boostedPix = rtgmc_retouch_round_clamp(boosted, maxVal);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value<Type>(src, ix, iy, srcPitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value<Type>(src, ix, iy, srcPitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff, rangeHalf);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(boostedPix, correction, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_edge_narrow_delta(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const float edgeNarrowingGain,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_vertical_balance_delta_value<Type>(src, ix, iy, srcPitch, width, height, edgeNarrowingGain, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_blur_h(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height, const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_blur10h_value<Type>(src, ix, iy, srcPitch, width, height), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_edge_narrow_guard_delta(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value<Type>(src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix, rangeHalf);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, value, maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_edge_narrow_guard_delta11(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value<Type>(src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix, rangeHalf);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, value, maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_adddiff(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ diff, const int diffPitch,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int diffPix = rtgmc_retouch_read_pix<Type>(diff, ix, iy, diffPitch, width, height);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, diffPix, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_edge_narrow_fused(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ base, const int basePitch,
    const int width, const int height,
    const float edgeNarrowingGain,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value<Type>(base, ix, iy, basePitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value<Type>(base, ix, iy, basePitch, width, height, edgeNarrowingGain, rangeHalf, maxVal);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff, rangeHalf);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, correction, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_make_delta(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ base, const int basePitch,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int srcPix = rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int basePix = rtgmc_retouch_read_pix<Type>(base, ix, iy, basePitch, width, height);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_make_diff_value(srcPix, basePix, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_smooth_delta_fused(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ base, const int basePitch,
    const int width, const int height,
    const int rangeHalf,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_diff_value<Type>(src, srcPitch, base, basePitch, ix, iy, width, height, rangeHalf, maxVal), maxVal);
}

template<typename Type>
__device__ float rtgmc_retouch_gauss_weight(const int targetPos, const float srcPos) {
    const float delta = (float)targetPos - (srcPos - 0.5f);
    return exp2f(-(RTGMC_RETOUCH_BACKBLEND_GAUSS_P * 0.1f) * delta * delta);
}

template<typename Type>
__device__ float rtgmc_retouch_gauss_h_value(
    const uint8_t *__restrict__ src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    (void)height;
    const float srcX = (float)x + 0.5f;
    const int srcFirstX = max(0, (int)floorf(srcX - RTGMC_RETOUCH_BACKBLEND_GAUSS_RADIUS));
    const int srcEndX = min(width - 1, (int)ceilf(srcX + RTGMC_RETOUCH_BACKBLEND_GAUSS_RADIUS));
    float clr = 0.0f;
    float sumWeight = 0.0f;
    for (int ix = srcFirstX; ix <= srcEndX; ix++) {
        const float wx = rtgmc_retouch_gauss_weight<Type>(ix, srcX);
        sumWeight += wx;
        clr += (float)rtgmc_retouch_read_pix<Type>(src, ix, y, pitch, width, height) * wx;
    }
    return (sumWeight > 0.0f) ? (clr / sumWeight) : 0.0f;
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_gauss_soft_delta(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const int width, const int height,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const float srcY = (float)iy + 0.5f;
    const int srcFirstY = max(0, (int)floorf(srcY - RTGMC_RETOUCH_BACKBLEND_GAUSS_RADIUS));
    const int srcEndY = min(height - 1, (int)ceilf(srcY + RTGMC_RETOUCH_BACKBLEND_GAUSS_RADIUS));
    float clr = 0.0f;
    float sumWeight = 0.0f;
    for (int jy = srcFirstY; jy <= srcEndY; jy++) {
        const float wy = rtgmc_retouch_gauss_weight<Type>(jy, srcY);
        sumWeight += wy;
        clr += rtgmc_retouch_gauss_h_value<Type>(src, ix, jy, srcPitch, width, height) * wy;
    }
    if (sumWeight > 0.0f) {
        clr /= sumWeight;
    }
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, (int)(clamp(clr, 0.0f, (float)maxVal) + 0.5f), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_limit(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ base, const int basePitch,
    const uint8_t *__restrict__ ref, const int refPitch,
    const uint8_t *__restrict__ motionBack, const int motionBackPitch,
    const uint8_t *__restrict__ motionForw, const int motionForwPitch,
    const int width, const int height,
    const int slmode,
    const int slrad,
    const int sovs,
    const float limitStrength,
    const int useTemporalLimit,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float value = (float)rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    if ((slmode == 2 || slmode == 4) && useTemporalLimit != 0) {
        value = (float)rtgmc_retouch_temporal_detail_guard_value<Type>(
            rtgmc_retouch_round_clamp(value, maxVal),
            ref, motionBack, motionForw,
            ix, iy,
            refPitch, motionBackPitch, motionForwPitch,
            width, height,
            sovs, maxVal);
    } else if (slmode == 1 || slmode == 2 || slmode == 4 || limitStrength > 0.0f) {
        const int radius = clamp(slrad, 1, 3);
        const float localMin = (float)max(0, rtgmc_retouch_spatial_min<Type>(base, ix, iy, basePitch, width, height, radius, maxVal) - sovs);
        const float localMax = (float)min(maxVal, rtgmc_retouch_spatial_max<Type>(base, ix, iy, basePitch, width, height, radius) + sovs);
        const float limited = clamp(value, localMin, localMax);
        const float strength = (slmode == 1 || slmode == 2) ? 1.0f : clamp(limitStrength, 0.0f, 1.0f);
        value = value + (limited - value) * strength;
    }
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value, maxVal), maxVal);
}

template<typename Type>
__global__ void kernel_rtgmc_retouch_limit_inline_comp(
    uint8_t *__restrict__ dst, const int dstPitch,
    const uint8_t *__restrict__ src, const int srcPitch,
    const uint8_t *__restrict__ ref, const int refPitch,
    const RGYDegrainCompensateInlineParams compParams,
    const int width, const int height,
    const int sovs,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const float value = (float)rtgmc_retouch_read_pix<Type>(src, ix, iy, srcPitch, width, height);
    const int result = rtgmc_retouch_temporal_detail_guard_value_inline_comp<Type>(
        rtgmc_retouch_round_clamp(value, maxVal),
        ref, ix, iy, refPitch,
        width, height,
        compParams,
        sovs, maxVal);
    rtgmc_retouch_write_pix<Type>(dst, ix, iy, dstPitch, result, maxVal);
}

NVEncFilterRtgmcRetouch::NVEncFilterRtgmcRetouch() :
    NVEncFilter(),
    m_buildOptions(),
    m_lumaDump(),
    m_lumaDumpPath(),
    m_lumaDumpStage("edge_narrow_blur_delta"),
    m_lumaDumpTarget(),
    m_lumaDumpMaxFrames(0),
    m_lumaDumpFrameCount(0),
    m_lumaDumpEnabled(false),
    m_lumaDumpHeaderWritten(false),
    m_lumaDumpChroma(false),
    m_useKernel(false),
    m_temporalLimitFrames(),
    m_spatialLimitBaseFrame(nullptr),
    m_loggedTemporalFallback(false) {
    m_name = _T("rtgmc-retouch");
}

NVEncFilterRtgmcRetouch::~NVEncFilterRtgmcRetouch() {
    close();
}

RGY_ERR NVEncFilterRtgmcRetouch::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcRetouch> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc_retouch.sharpness < 0.0f || prm->rtgmc_retouch.sharpness > 1.0f) {
        prm->rtgmc_retouch.sharpness = clamp(prm->rtgmc_retouch.sharpness, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sharpness should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.limit < 0.0f || prm->rtgmc_retouch.limit > 1.0f) {
        prm->rtgmc_retouch.limit = clamp(prm->rtgmc_retouch.limit, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch limit should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.smode < 0 || prm->rtgmc_retouch.smode > 2) {
        prm->rtgmc_retouch.smode = clamp(prm->rtgmc_retouch.smode, 0, 2);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch smode should be in range of 0 - 2.\n"));
    }
    if (prm->rtgmc_retouch.slmode < 0 || prm->rtgmc_retouch.slmode > 4) {
        prm->rtgmc_retouch.slmode = clamp(prm->rtgmc_retouch.slmode, 0, 4);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch slmode should be in range of 0 - 4.\n"));
    }
    if (prm->rtgmc_retouch.slrad < 0 || prm->rtgmc_retouch.slrad > 3) {
        prm->rtgmc_retouch.slrad = clamp(prm->rtgmc_retouch.slrad, 0, 3);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch slrad should be in range of 0 - 3.\n"));
    }
    if (prm->rtgmc_retouch.sovs < 0) {
        prm->rtgmc_retouch.sovs = 0;
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sovs should be 0 or greater.\n"));
    }
    if (prm->rtgmc_retouch.svthin < 0.0f || prm->rtgmc_retouch.svthin > 1.0f) {
        prm->rtgmc_retouch.svthin = clamp(prm->rtgmc_retouch.svthin, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch svthin should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.sbb < 0 || prm->rtgmc_retouch.sbb > 3) {
        prm->rtgmc_retouch.sbb = clamp(prm->rtgmc_retouch.sbb, 0, 3);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sbb should be in range of 0 - 3.\n"));
    }
    if ((prm->rtgmc_retouch.smode == 0 || prm->rtgmc_retouch.sharpness <= 0.0f || prm->rtgmc_retouch.slrad <= 0)
        && prm->rtgmc_retouch.slmode < 3) {
        prm->rtgmc_retouch.slmode = 0;
    }
    if (prm->rtgmc_retouch.slmode >= 3 && prm->skipPostTR2LimitModes) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch slmode=%d is a post-TR2 mode and is not applied in this stage.\n"), prm->rtgmc_retouch.slmode);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterRtgmcRetouch::setSpatialLimitBaseFrame(const RGYFrameInfo *frame) {
    m_spatialLimitBaseFrame = frame;
}

void NVEncFilterRtgmcRetouch::clearSpatialLimitBaseFrame() {
    m_spatialLimitBaseFrame = nullptr;
}

void NVEncFilterRtgmcRetouch::setTemporalLimitFrames(const RGYRtgmcRetouchTemporalLimitFrames &frames) {
    m_temporalLimitFrames = frames;
    m_temporalLimitFrames.useInlineComp = false;
    m_loggedTemporalFallback = false;
}

void NVEncFilterRtgmcRetouch::clearTemporalLimitFrames() {
    m_temporalLimitFrames = RGYRtgmcRetouchTemporalLimitFrames();
    m_loggedTemporalFallback = false;
}

void NVEncFilterRtgmcRetouch::setTemporalLimitInlineComp(const RGYFrameInfo *ref, const std::array<RGYDegrainCompensateInlineParams, 3> &params, bool processChroma) {
    m_temporalLimitFrames.ref = ref;
    m_temporalLimitFrames.motionBack = nullptr;
    m_temporalLimitFrames.motionForw = nullptr;
    m_temporalLimitFrames.useInlineComp = true;
    m_temporalLimitFrames.inlineCompChroma = processChroma;
    m_temporalLimitFrames.inlineCompParams = params;
    m_loggedTemporalFallback = false;
}

bool NVEncFilterRtgmcRetouch::temporalLimitFramesCompatible(const RGYFrameInfo *srcFrame) const {
    const auto &frames = m_temporalLimitFrames;
    if (frames.useInlineComp) {
        return isFrameCompatible(srcFrame, frames.ref);
    }
    return isFrameCompatible(srcFrame, frames.ref)
        && isFrameCompatible(srcFrame, frames.motionBack)
        && isFrameCompatible(srcFrame, frames.motionForw);
}

bool NVEncFilterRtgmcRetouch::temporalLimitFramesReady(const RGYFrameInfo *srcFrame) const {
    return m_temporalLimitFrames.valid() && temporalLimitFramesCompatible(srcFrame);
}

RGY_ERR NVEncFilterRtgmcRetouch::buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcRetouch> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "Type=%s bit_depth=%d max_val=%d range_half=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf);
    m_useKernel = bitdepth <= 16;
    return m_useKernel ? RGY_ERR_NONE : RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterRtgmcRetouch::initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcRetouch &prm) {
    UNREFERENCED_PARAMETER(prm);
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "edge_narrow_blur_delta";
    m_lumaDumpTarget.clear();
    m_lumaDumpChroma = false;
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("NVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M");
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        dumpPathEnv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M");
    }
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        return RGY_ERR_NONE;
    }
    m_lumaDumpPath = dumpPathEnv;

    if (const char *stageEnv = std::getenv("NVENC_RTGMC_RETOUCH_LUMA_DUMP_STAGE"); stageEnv != nullptr && stageEnv[0] != '\0') {
        m_lumaDumpStage = stageEnv;
    } else if (const char *stageEnvQsv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_STAGE"); stageEnvQsv != nullptr && stageEnvQsv[0] != '\0') {
        m_lumaDumpStage = stageEnvQsv;
    }
    if (!isRtgmcRetouchDumpStageSupported(m_lumaDumpStage)) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported rtgmc retouch luma dump stage: %s.\n"),
            char_to_tstring(m_lumaDumpStage).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (const char *targetEnv = std::getenv("NVENC_RTGMC_RETOUCH_LUMA_DUMP_TARGET"); targetEnv != nullptr && targetEnv[0] != '\0') {
        m_lumaDumpTarget = targetEnv;
    } else if (const char *targetEnvQsv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_TARGET"); targetEnvQsv != nullptr && targetEnvQsv[0] != '\0') {
        m_lumaDumpTarget = targetEnvQsv;
    }
    const char *activeTarget = rtgmcRetouchDumpTargetForStage(m_lumaDumpStage);
    if (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != activeTarget) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc retouch luma dump target %s skipped for inactive %s instance.\n"),
            char_to_tstring(m_lumaDumpTarget).c_str(), char_to_tstring(activeTarget).c_str());
        return RGY_ERR_NONE;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("NVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frameInfo.csp] != 1) {
        AddMessage(RGY_LOG_WARN, _T("NVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M supports only 4:2:0/Y8 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    const char *maxFrames = std::getenv("NVENC_RTGMC_RETOUCH_LUMA_DUMP_MAX_FRAMES");
    if (maxFrames == nullptr || maxFrames[0] == '\0') {
        maxFrames = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_MAX_FRAMES");
    }
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_lumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }
    const char *dumpChroma = std::getenv("NVENC_RTGMC_RETOUCH_DUMP_CHROMA");
    if (dumpChroma == nullptr || dumpChroma[0] == '\0') {
        dumpChroma = std::getenv("QSVENC_RTGMC_RETOUCH_DUMP_CHROMA");
    }
    if (dumpChroma != nullptr && dumpChroma[0] != '\0' && dumpChroma[0] != '0') {
        m_lumaDumpChroma = true;
    }

    m_lumaDump.open(m_lumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc retouch luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc retouch luma dump enabled: %s (target=%s, stage=%s).\n"),
        char_to_tstring(m_lumaDumpPath).c_str(), char_to_tstring(activeTarget).c_str(), char_to_tstring(m_lumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcRetouch::dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, bool dumpChroma) {
    if (!m_lumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpMaxFrames > 0 && m_lumaDumpFrameCount >= m_lumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    if (frame == nullptr || frame->ptr[0] == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    if (bitdepth > 8 || (RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc retouch luma dump disabled by unsupported frame csp: %s.\n"),
            RGY_CSP_NAMES[frame->csp]);
        m_lumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }

    auto sts = rtgmcRetouchWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    CUFrameBuf hostFrame(frame->width, frame->height, frame->csp);
    hostFrame.frame.mem_type = RGY_MEM_TYPE_CPU;
    sts = hostFrame.allocHost();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc retouch luma dump host buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = copyFrameAsync(&hostFrame.frame, frame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc retouch luma dump frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = err_to_rgy(cudaStreamSynchronize(stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch luma dump read: %s.\n"), get_err_mes(sts));
        return sts;
    }

    const auto planeY = getPlane(&hostFrame.frame, RGY_PLANE_Y);
    const auto planeU = getPlane(&hostFrame.frame, RGY_PLANE_U);
    const auto planeV = getPlane(&hostFrame.frame, RGY_PLANE_V);
    const int chromaWidth = (planeY.width + 1) >> 1;
    const int chromaHeight = (planeY.height + 1) >> 1;
    std::vector<uint8_t> hostU;
    std::vector<uint8_t> hostV;
    if (dumpChroma && RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_YUV420 && planeU.ptr[0] != nullptr) {
        hostU.resize((size_t)chromaWidth * chromaHeight);
        hostV.resize((size_t)chromaWidth * chromaHeight);
        if (frame->csp == RGY_CSP_NV12) {
            for (int y = 0; y < chromaHeight; y++) {
                const auto *srcUV = planeU.ptr[0] + (size_t)y * planeU.pitch[0];
                auto *dstU = hostU.data() + (size_t)y * chromaWidth;
                auto *dstV = hostV.data() + (size_t)y * chromaWidth;
                for (int x = 0; x < chromaWidth; x++) {
                    dstU[x] = srcUV[x * 2 + 0];
                    dstV[x] = srcUV[x * 2 + 1];
                }
            }
        } else if (planeV.ptr[0] != nullptr) {
            for (int y = 0; y < chromaHeight; y++) {
                memcpy(hostU.data() + (size_t)y * chromaWidth, planeU.ptr[0] + (size_t)y * planeU.pitch[0], chromaWidth);
                memcpy(hostV.data() + (size_t)y * chromaWidth, planeV.ptr[0] + (size_t)y * planeV.pitch[0], chromaWidth);
            }
        } else {
            hostU.clear();
            hostV.clear();
        }
    }

    if (!m_lumaDumpHeaderWritten) {
        m_lumaDump << "YUV4MPEG2 W" << planeY.width << " H" << planeY.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_lumaDumpHeaderWritten = true;
    }
    m_lumaDump << "FRAME\n";
    for (int y = 0; y < planeY.height; y++) {
        m_lumaDump.write(reinterpret_cast<const char *>(planeY.ptr[0] + (size_t)y * planeY.pitch[0]), planeY.width);
    }
    if (!hostU.empty() && !hostV.empty()) {
        m_lumaDump.write(reinterpret_cast<const char *>(hostU.data()), hostU.size());
        m_lumaDump.write(reinterpret_cast<const char *>(hostV.data()), hostV.size());
    } else {
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc retouch luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcRetouch::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_lumaDumpEnabled || m_lumaDumpStage != stage || (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != target)) {
        return RGY_ERR_NONE;
    }
    const bool dumpChroma = m_lumaDumpChroma && rtgmcRetouchDumpStageChromaReady(m_lumaDumpStage);
    return dumpLumaFrame(frame, stream, wait_events, dumpChroma);
}

RGY_ERR NVEncFilterRtgmcRetouch::setupDetailRollbackGaussFilter(const NVEncFilterParamRtgmcRetouch &prm) {
    UNREFERENCED_PARAMETER(prm);
    return RGY_ERR_NONE;
}

NVEncFilterRtgmcRetouch::RtgmcRetouchPlan NVEncFilterRtgmcRetouch::buildRtgmcRetouchPlan(
    const VppRtgmcRetouch &retouch, bool chromaPlane, bool skipPostTR2LimitModes, float detailGain) const {
    RtgmcRetouchPlan plan;

    const bool allowDetailBoost = retouch.smode > 0 && detailGain > 0.0f;
    const bool allowSpatialGuard = retouch.slmode == 1 || (retouch.slmode == 3 && !skipPostTR2LimitModes);
    const bool allowTemporalGuard = retouch.slmode == 2 || (retouch.slmode == 4 && !skipPostTR2LimitModes) || retouch.limit > 0.0f;
    if (allowDetailBoost) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::DetailBoost, 0, 1, 3, 4, "detail_boost" });
    }

    if (!chromaPlane) {
        if (retouch.svthin > 0.0f) {
            plan.nodes.push_back({ RtgmcRetouchNodeKind::EdgeNarrowCorrection, 0, 2, 3, 4, "edge_narrow" });
        }
        if (retouch.sbb == 1 || retouch.sbb == 3) {
            plan.nodes.push_back({ RtgmcRetouchNodeKind::PreLimitRollback, 0, 2, 3, 4, "prelimit_rollback" });
        }
    }
    if (allowSpatialGuard) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::SpatialOvershootGuard, 0, 2, 3, 4,
            (retouch.slmode == 3) ? "postlimit_spatial_guard" : "spatial_guard" });
    }
    if (allowTemporalGuard) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::TemporalOvershootGuard, 0, 2, 3, 4,
            (retouch.slmode == 4) ? "postlimit_temporal_guard" : "temporal_guard" });
    }
    if (!chromaPlane && retouch.sbb >= 2) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::PostLimitRollback, 0, 2, 3, 4, "postlimit_rollback" });
    }
    return plan;
}

std::string NVEncFilterRtgmcRetouch::describeRtgmcRetouchPlan(const RtgmcRetouchPlan &plan) const {
    if (plan.nodes.empty()) {
        return "copy";
    }
    std::string desc;
    for (const auto &node : plan.nodes) {
        const char *kind = "unknown";
        switch (node.kind) {
        case RtgmcRetouchNodeKind::DetailBoost:
            kind = "detail_boost";
            break;
        case RtgmcRetouchNodeKind::EdgeNarrowCorrection:
            kind = "edge_narrow";
            break;
        case RtgmcRetouchNodeKind::PreLimitRollback:
            kind = "prelimit_rollback";
            break;
        case RtgmcRetouchNodeKind::SpatialOvershootGuard:
            kind = "spatial_guard";
            break;
        case RtgmcRetouchNodeKind::TemporalOvershootGuard:
            kind = "temporal_guard";
            break;
        case RtgmcRetouchNodeKind::PostLimitRollback:
            kind = "postlimit_rollback";
            break;
        }
        if (!desc.empty()) {
            desc += " -> ";
        }
        desc += kind;
        if (node.dumpStage != nullptr && node.dumpStage[0] != '\0') {
            desc += "(";
            desc += node.dumpStage;
            desc += ")";
        }
    }
    return desc;
}

RGY_ERR NVEncFilterRtgmcRetouch::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcRetouch>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcRetouch>(m_param);
    if (!m_param
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-retouch kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, RTGMC_RETOUCH_FRAMEBUF_COUNT);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initLumaDump(m_frameBuf[0]->frame, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = setupDetailRollbackGaussFilter(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    const float detailGain = calcRtgmcDetailGain(prm->rtgmc_retouch);
    for (int iplane = 0; iplane < RGY_CSP_PLANES[prm->frameOut.csp]; iplane++) {
        const bool chromaPlane = isRetouchChromaPlane(iplane);
        const auto plan = buildRtgmcRetouchPlan(prm->rtgmc_retouch, chromaPlane, prm->skipPostTR2LimitModes, detailGain);
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch plan plane %d (%s): %s.\n"),
            iplane,
            chromaPlane ? _T("chroma") : _T("luma"),
            char_to_tstring(describeRtgmcRetouchPlan(plan)).c_str());
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcRetouch::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const NVEncFilterParamRtgmcRetouch &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const float detailGain = calcRtgmcDetailGain(prm.rtgmc_retouch);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const int scaledSovs = prm.rtgmc_retouch.sovs << std::max(bitdepth - 8, 0);
    const bool hasTemporalLimitFrames = temporalLimitFramesReady(pInputFrame);
    const auto refFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.ref : pInputFrame;
    const auto motionBackFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.motionBack : pInputFrame;
    const auto motionForwFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.motionForw : pInputFrame;
    const auto baseFrame = isFrameCompatible(pInputFrame, m_spatialLimitBaseFrame) ? m_spatialLimitBaseFrame : pInputFrame;
    const char *disableFusionEnv = std::getenv("NVENC_RTGMC_RETOUCH_DISABLE_FUSION");
    if (disableFusionEnv == nullptr || disableFusionEnv[0] == '\0') {
        disableFusionEnv = std::getenv("QSVENC_RTGMC_RETOUCH_DISABLE_FUSION");
    }
    const bool disableFusionByEnv = disableFusionEnv != nullptr && disableFusionEnv[0] != '\0' && disableFusionEnv[0] != '0';
    const bool disableFusion = m_lumaDumpEnabled || disableFusionByEnv;
    const char *mergeDetailLineEnv = std::getenv("NVENC_RTGMC_KERNEL_MERGE_RETOUCH_DETAIL_LINE");
    if (mergeDetailLineEnv == nullptr) {
        mergeDetailLineEnv = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_RETOUCH_DETAIL_LINE");
    }
    const bool enableDetailLineMerge = mergeDetailLineEnv == nullptr || mergeDetailLineEnv[0] != '0';
    const char *mergeRollbackEnv = std::getenv("NVENC_RTGMC_KERNEL_MERGE_RETOUCH_ROLLBACK");
    if (mergeRollbackEnv == nullptr) {
        mergeRollbackEnv = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_RETOUCH_ROLLBACK");
    }
    const bool enableRollbackMerge = mergeRollbackEnv == nullptr || mergeRollbackEnv[0] != '0';

    auto launchCommon = [&](const char *kernelName, int iplane, const std::vector<RGYCudaEvent> &wait) {
        auto err = rtgmcRetouchWaitEvents(stream, wait);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    };
    auto checkLaunch = [&](const char *kernelName, int iplane) {
        const auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto err = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    };

#define LAUNCH_RETOUCH_TYPED(kernel, gridSize, blockSize, ...) \
    do { \
        if (bitdepth <= 8) { \
            kernel<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
        } else { \
            kernel<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
        } \
    } while (0)

    auto launchCopy = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait, RGYCudaEvent *ev) {
        const char *kernelName = "kernel_rtgmc_retouch_copy";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_copy, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        err = checkLaunch(kernelName, iplane);
        if (err != RGY_ERR_NONE) return err;
        return rtgmcRetouchRecordEvent(stream, ev);
    };
    auto launchRemoveGrain = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane, const int smoothingMode) {
        const char *kernelName = (smoothingMode == 11) ? "kernel_rtgmc_retouch_removegrain11" : "kernel_rtgmc_retouch_removegrain12";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        if (smoothingMode == 11) {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_removegrain11, gridSize, blockSize,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                maxVal);
        } else {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_removegrain12, gridSize, blockSize,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                maxVal);
        }
        return checkLaunch(kernelName, iplane);
    };
    auto launchRepairMode1 = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_repair1";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_repair1, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchRepairMode12 = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_repair12";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_repair12, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchDetailBoostEdgeRef = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_detail_ref_vertical";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_detail_ref_vertical, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchPreciseClamp = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_precise_clamp";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_precise_clamp, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchBlurH = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_blur_h";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_blur_h, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchEdgeNarrowDelta = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_edge_narrow_delta";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_edge_narrow_delta, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch),
            rangeHalf,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchEdgeNarrowGuardDelta = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane, const int smoothingMode) {
        const char *kernelName = (smoothingMode == 11) ? "kernel_rtgmc_retouch_edge_narrow_guard_delta11" : "kernel_rtgmc_retouch_edge_narrow_guard_delta";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        if (smoothingMode == 11) {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_edge_narrow_guard_delta11, gridSize, blockSize,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                rangeHalf,
                maxVal);
        } else {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_edge_narrow_guard_delta, gridSize, blockSize,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                rangeHalf,
                maxVal);
        }
        return checkLaunch(kernelName, iplane);
    };
    auto launchDetailRollback = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *tmpFrame, const RGYFrameInfo *auxFrame,
        const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame, const int iplane, const char *dumpPrefix) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto tmpPlane = getPlane(tmpFrame, (RGY_PLANE)iplane);
        const auto auxPlane = getPlane(auxFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        const bool needRollbackDeltaDump = iplane == 0 && dumpPrefix != nullptr && m_lumaDumpEnabled
            && m_lumaDumpStage == std::string(dumpPrefix) + "_delta"
            && (m_lumaDumpTarget.empty() || m_lumaDumpTarget == "rollback");
        const bool mergeRollbackSmoothDelta = enableRollbackMerge && !disableFusionByEnv && !needRollbackDeltaDump;
        auto err = RGY_ERR_NONE;
        if (mergeRollbackSmoothDelta) {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_smooth_delta_fused, gridSize, blockSize,
                (uint8_t *)tmpPlane.ptr[0], tmpPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                (const uint8_t *)basePlane.ptr[0], basePlane.pitch[0],
                tmpPlane.width, tmpPlane.height,
                rangeHalf,
                maxVal);
            err = checkLaunch("kernel_rtgmc_retouch_smooth_delta_fused", iplane);
            if (err != RGY_ERR_NONE) return err;
        } else {
            LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_make_delta, gridSize, blockSize,
                (uint8_t *)auxPlane.ptr[0], auxPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                (const uint8_t *)basePlane.ptr[0], basePlane.pitch[0],
                auxPlane.width, auxPlane.height,
                rangeHalf,
                maxVal);
            err = checkLaunch("kernel_rtgmc_retouch_make_delta", iplane);
            if (err != RGY_ERR_NONE) return err;
            if (iplane == 0 && dumpPrefix != nullptr) {
                const auto stage = std::string(dumpPrefix) + "_delta";
                err = dumpStageFrame(stage.c_str(), auxFrame, "rollback", stream, {});
                if (err != RGY_ERR_NONE) return err;
            }
            err = launchRemoveGrain(tmpFrame, auxFrame, iplane, 12);
            if (err != RGY_ERR_NONE) return err;
        }
        if (iplane == 0 && dumpPrefix != nullptr) {
            const auto stage = std::string(dumpPrefix) + "_smooth_delta";
            err = dumpStageFrame(stage.c_str(), tmpFrame, "rollback", stream, {});
            if (err != RGY_ERR_NONE) return err;
        }
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_gauss_soft_delta, gridSize, blockSize,
            (uint8_t *)auxPlane.ptr[0], auxPlane.pitch[0],
            (const uint8_t *)tmpPlane.ptr[0], tmpPlane.pitch[0],
            auxPlane.width, auxPlane.height,
            maxVal);
        err = checkLaunch("kernel_rtgmc_retouch_gauss_soft_delta", iplane);
        if (err != RGY_ERR_NONE) return err;
        if (iplane == 0 && dumpPrefix != nullptr) {
            const auto stage = std::string(dumpPrefix) + "_soft_delta";
            err = dumpStageFrame(stage.c_str(), auxFrame, "rollback", stream, {});
            if (err != RGY_ERR_NONE) return err;
        }
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_make_delta, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)auxPlane.ptr[0], auxPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            rangeHalf,
            maxVal);
        return checkLaunch("kernel_rtgmc_retouch_make_delta", iplane);
    };
    auto launchDetailBoost = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *blurFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_detail_boost";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto blurPlane = getPlane(blurFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_detail_boost, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)blurPlane.ptr[0], blurPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            detailGain,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchDetailBoostFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_detail_boost_fused";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_detail_boost_fused, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.smode,
            prm.rtgmc_retouch.precise ? 1 : 0,
            detailGain,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchDetailBoostEdgeNarrowFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_detail_boost_edge_narrow_fused";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_detail_boost_edge_narrow_fused, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.smode,
            prm.rtgmc_retouch.precise ? 1 : 0,
            detailGain,
            calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch),
            rangeHalf,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchAddDiff = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *diffFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_adddiff";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto diffPlane = getPlane(diffFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_adddiff, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)diffPlane.ptr[0], diffPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            rangeHalf,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchEdgeNarrowFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame,
        const int iplane, const std::vector<RGYCudaEvent> &wait) {
        const char *kernelName = "kernel_rtgmc_retouch_edge_narrow_fused";
        auto err = launchCommon(kernelName, iplane, wait);
        if (err != RGY_ERR_NONE) return err;
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_edge_narrow_fused, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)basePlane.ptr[0], basePlane.pitch[0],
            dstPlane.width, dstPlane.height,
            calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch),
            rangeHalf,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchLimitSink = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame,
        const RGYFrameInfo *refLimitFrame, const RGYFrameInfo *motionBackLimitFrame, const RGYFrameInfo *motionForwLimitFrame,
        const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_limit";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refLimitFrame, (RGY_PLANE)iplane);
        const auto motionBackPlane = getPlane(motionBackLimitFrame, (RGY_PLANE)iplane);
        const auto motionForwPlane = getPlane(motionForwLimitFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_limit, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)basePlane.ptr[0], basePlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            (const uint8_t *)motionBackPlane.ptr[0], motionBackPlane.pitch[0],
            (const uint8_t *)motionForwPlane.ptr[0], motionForwPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.slmode,
            prm.rtgmc_retouch.slrad,
            scaledSovs,
            prm.rtgmc_retouch.limit,
            hasTemporalLimitFrames ? 1 : 0,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };
    auto launchLimitSinkInlineComp = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame,
        const RGYFrameInfo *refLimitFrame, const int iplane) {
        const char *kernelName = "kernel_rtgmc_retouch_limit_inline_comp";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refLimitFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        const auto &compParams = m_temporalLimitFrames.inlineCompParams[iplane];
        LAUNCH_RETOUCH_TYPED(kernel_rtgmc_retouch_limit_inline_comp, gridSize, blockSize,
            (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
            (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            compParams,
            dstPlane.width, dstPlane.height,
            scaledSovs,
            maxVal);
        return checkLaunch(kernelName, iplane);
    };

    auto *curA = &m_frameBuf[1]->frame;
    auto *curB = &m_frameBuf[2]->frame;
    auto *work0 = &m_frameBuf[3]->frame;
    auto *work1 = &m_frameBuf[4]->frame;

    for (int iplane = 0; iplane < planes; iplane++) {
        const bool chromaPlane = isRetouchChromaPlane(iplane);
        const int smoothingMode = prm.rtgmc_retouch.precise ? 11 : 12;
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYCudaEvent>();
        if (chromaPlane && !prm.processChroma) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane, waitHere, (iplane == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            continue;
        }
        const auto plan = buildRtgmcRetouchPlan(prm.rtgmc_retouch, chromaPlane, prm.skipPostTR2LimitModes, detailGain);

        if (iplane == 0) {
            auto err = dumpStageFrame("input", pInputFrame, "retouch", stream, waitHere);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }

        if (plan.nodes.empty()) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane, waitHere, (iplane == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            continue;
        }

        const RGYFrameInfo *curFrame = pInputFrame;
        RGYFrameInfo *curDst = curA;
        RGYFrameInfo *altDst = curB;

        for (size_t inode = 0; inode < plan.nodes.size(); inode++) {
            const auto &node = plan.nodes[inode];
            switch (node.kind) {
            case RtgmcRetouchNodeKind::DetailBoost: {
            const bool mergeDetailLine = enableDetailLineMerge
                && !disableFusion
                && iplane == 0
                && detailGain > 0.0f
                && prm.rtgmc_retouch.svthin > 0.0f
                && prm.rtgmc_retouch.smode > 0
                && inode + 1 < plan.nodes.size()
                && plan.nodes[inode + 1].kind == RtgmcRetouchNodeKind::EdgeNarrowCorrection;
            if (mergeDetailLine) {
                auto err = launchDetailBoostEdgeNarrowFused(curDst, pInputFrame, iplane, waitHere);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = curDst;
                std::swap(curDst, altDst);
                inode++;
                break;
            }
            if (!disableFusion) {
                auto err = launchDetailBoostFused(curDst, pInputFrame, iplane, waitHere);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = curDst;
                std::swap(curDst, altDst);
                break;
            }
            auto err = (prm.rtgmc_retouch.smode == 2)
                ? launchDetailBoostEdgeRef(work0, pInputFrame, iplane, waitHere)
                : RGY_ERR_NONE;
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0 && prm.rtgmc_retouch.smode == 2) {
                err = dumpStageFrame("detail_boost_edge_ref", work0, "retouch", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            const RGYFrameInfo *blurInput = pInputFrame;
            if (prm.rtgmc_retouch.smode == 2) {
                blurInput = work0;
                if (prm.rtgmc_retouch.precise) {
                    err = launchPreciseClamp(work1, work0, pInputFrame, iplane);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                    if (iplane == 0) {
                        err = dumpStageFrame("detail_boost_regularized_ref", work1, "retouch", stream, {});
                        if (err != RGY_ERR_NONE) {
                            return err;
                        }
                    }
                    blurInput = work1;
                }
            }
            err = launchRemoveGrain(work1, blurInput, iplane, smoothingMode);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("detail_boost_blur_ref", work1, "retouch", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchDetailBoost(curDst, pInputFrame, work1, iplane, prm.rtgmc_retouch.smode == 2 ? std::vector<RGYCudaEvent>() : waitHere);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("detail_boost", curDst, "retouch", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = curDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::EdgeNarrowCorrection: {
            if (!disableFusion) {
                const auto &thinWait = (curFrame == pInputFrame) ? waitHere : std::vector<RGYCudaEvent>();
                auto err = launchEdgeNarrowFused(altDst, curFrame, pInputFrame, iplane, thinWait);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = altDst;
                std::swap(curDst, altDst);
                break;
            }
            auto err = launchEdgeNarrowDelta(work0, pInputFrame, iplane, (curFrame == pInputFrame) ? waitHere : std::vector<RGYCudaEvent>());
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_delta", work0, "edge_narrow", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchBlurH(work1, work0, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_blur_delta", work1, "edge_narrow", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchEdgeNarrowGuardDelta(work0, work1, iplane, smoothingMode);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_guard_delta", work0, "edge_narrow", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchAddDiff(altDst, curFrame, work0, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow", altDst, "edge_narrow", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::PreLimitRollback: {
            auto err = launchDetailRollback(altDst, work0, work1, curFrame, pInputFrame, iplane, node.dumpStage);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage, altDst, "rollback", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::SpatialOvershootGuard: {
            if (iplane == 0 && prm.rtgmc_retouch.slmode == 3) {
                auto err = dumpStageFrame("postlimit_spatial_guard_src", curFrame, "postlimit", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                err = dumpStageFrame("postlimit_spatial_guard_ref", baseFrame, "postlimit", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            auto err = RGY_ERR_NONE;
            if (prm.rtgmc_retouch.slrad <= 1) {
                err = launchRepairMode1(altDst, curFrame, baseFrame, iplane);
            } else {
                err = launchRepairMode12(work0, curFrame, baseFrame, iplane);
                if (err == RGY_ERR_NONE) {
                    err = launchRepairMode1(altDst, curFrame, work0, iplane);
                }
            }
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage,
                    altDst, (prm.rtgmc_retouch.slmode == 3) ? "postlimit" : "retouch", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::TemporalOvershootGuard: {
            if (iplane == 0 && (prm.rtgmc_retouch.slmode == 2 || prm.rtgmc_retouch.slmode == 4)) {
                const char *dumpTarget = (prm.rtgmc_retouch.slmode == 4) ? "postlimit" : "limitover";
                const char *srcStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_src" : "temporal_guard_src";
                const char *refStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_ref" : "temporal_guard_ref";
                const char *motionBackStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_motionback" : "temporal_guard_motionback";
                const char *motionForwStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_motionforw" : "temporal_guard_motionforw";
                auto err = dumpStageFrame(srcStage, curFrame, dumpTarget, stream, {});
                if (err != RGY_ERR_NONE) return err;
                err = dumpStageFrame(refStage, refFrame, dumpTarget, stream, {});
                if (err != RGY_ERR_NONE) return err;
                err = dumpStageFrame(motionBackStage, motionBackFrame, dumpTarget, stream, {});
                if (err != RGY_ERR_NONE) return err;
                err = dumpStageFrame(motionForwStage, motionForwFrame, dumpTarget, stream, {});
                if (err != RGY_ERR_NONE) return err;
            }
            RGY_ERR err;
            const bool useInlineCompForPlane = m_temporalLimitFrames.useInlineComp && (!chromaPlane || m_temporalLimitFrames.inlineCompChroma);
            if (useInlineCompForPlane) {
                err = launchLimitSinkInlineComp(altDst, curFrame, refFrame, iplane);
            } else {
                const auto motionBackForPlane = motionBackFrame ? motionBackFrame : refFrame;
                const auto motionForwForPlane = motionForwFrame ? motionForwFrame : refFrame;
                err = launchLimitSink(altDst, curFrame, baseFrame, refFrame, motionBackForPlane, motionForwForPlane, iplane);
            }
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0 && (prm.rtgmc_retouch.slmode == 2 || prm.rtgmc_retouch.slmode == 4)) {
                err = dumpStageFrame(node.dumpStage,
                    altDst, (prm.rtgmc_retouch.slmode == 4) ? "postlimit" : "limitover", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::PostLimitRollback: {
            auto err = launchDetailRollback(altDst, work0, work1, curFrame, pInputFrame, iplane, node.dumpStage);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage, altDst, "rollback", stream, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }
            default:
                AddMessage(RGY_LOG_ERROR, _T("unknown rtgmc retouch node kind.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
        }

        const auto &copyWait = (curFrame == pInputFrame) ? waitHere : std::vector<RGYCudaEvent>();
        auto err = launchCopy(pOutputFrame, curFrame, iplane, copyWait, (iplane == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

#undef LAUNCH_RETOUCH_TYPED

    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcRetouch::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (m_useKernel && !m_frameBuf.size()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load rtgmc-retouch kernel (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcRetouch>(m_param);
    if (!prm || (prm->rtgmc_retouch.smode == 0
        && prm->rtgmc_retouch.slmode == 0
        && prm->rtgmc_retouch.limit <= 0.0f
        && prm->rtgmc_retouch.svthin <= 0.0f
        && prm->rtgmc_retouch.sbb == 0)) {
        auto sts = rtgmcRetouchWaitEvents(stream, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        auto copyErr = pOutFrame->copyFrameAsync(pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
        return rtgmcRetouchRecordEvent(stream, event);
    }

    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && m_temporalLimitFrames.any() && !m_temporalLimitFrames.valid()) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch slmode=%d temporal detail guard requires ref/motionBack/motionForw frames together.\n"), prm->rtgmc_retouch.slmode);
        return RGY_ERR_INVALID_PARAM;
    }
    const bool hasTemporalLimitFrames = temporalLimitFramesReady(pInputFrame);
    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && m_temporalLimitFrames.valid() && !temporalLimitFramesCompatible(pInputFrame)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch slmode=%d temporal detail guard frames are not compatible with the current source frame.\n"), prm->rtgmc_retouch.slmode);
        return RGY_ERR_INVALID_PARAM;
    }
    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && !hasTemporalLimitFrames && !m_loggedTemporalFallback) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch slmode=%d temporal detail guard inputs are not wired; using spatial fallback.\n"), prm->rtgmc_retouch.slmode);
        m_loggedTemporalFallback = true;
    }

    if (m_useKernel) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        const auto baseMemcpyKind = m_spatialLimitBaseFrame ? getCudaMemcpyKind(m_spatialLimitBaseFrame->mem_type, m_frameBuf[0]->frame.mem_type) : cudaMemcpyDeviceToDevice;
        const auto refMemcpyKind = m_temporalLimitFrames.ref ? getCudaMemcpyKind(m_temporalLimitFrames.ref->mem_type, m_frameBuf[0]->frame.mem_type) : cudaMemcpyDeviceToDevice;
        const auto motionBackMemcpyKind = m_temporalLimitFrames.motionBack ? getCudaMemcpyKind(m_temporalLimitFrames.motionBack->mem_type, m_frameBuf[0]->frame.mem_type) : cudaMemcpyDeviceToDevice;
        const auto motionForwMemcpyKind = m_temporalLimitFrames.motionForw ? getCudaMemcpyKind(m_temporalLimitFrames.motionForw->mem_type, m_frameBuf[0]->frame.mem_type) : cudaMemcpyDeviceToDevice;
        if (memcpyKind == cudaMemcpyDeviceToDevice
            && baseMemcpyKind == cudaMemcpyDeviceToDevice
            && refMemcpyKind == cudaMemcpyDeviceToDevice
            && motionBackMemcpyKind == cudaMemcpyDeviceToDevice
            && motionForwMemcpyKind == cudaMemcpyDeviceToDevice) {
            return processFrame(ppOutputFrames[0], pInputFrame, *prm, stream, wait_events, event);
        }
    }

    AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch requires device-to-device CUDA frames.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterRtgmcRetouch::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    return run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum, stream, std::vector<RGYCudaEvent>(), nullptr);
}

void NVEncFilterRtgmcRetouch::resetTemporalState() {
    // Clear frame references that carry temporal context.
    // m_buildOptions and kernel objects are preserved.
    clearTemporalLimitFrames();
    clearSpatialLimitBaseFrame();
    // m_loggedTemporalFallback is already reset inside clearTemporalLimitFrames().
}

void NVEncFilterRtgmcRetouch::close() {
    m_buildOptions.clear();
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "edge_narrow_blur_delta";
    m_lumaDumpTarget.clear();
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpChroma = false;
    m_useKernel = false;
    clearTemporalLimitFrames();
    clearSpatialLimitBaseFrame();
    m_frameBuf.clear();
}
