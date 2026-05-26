// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#include "NVEncFilterNnedi.h"
#include "convert_csp.h"
#include "rgy_filesystem.h"
#include "rgy_resource.h"
#include "rgy_cuda_util_kernel.h"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace {

static constexpr std::array<RGYNnediNSizeDesc, 7> NNEDI_NSIZE_DESC = {
    RGYNnediNSizeDesc{  8, 6 },
    RGYNnediNSizeDesc{ 16, 6 },
    RGYNnediNSizeDesc{ 32, 6 },
    RGYNnediNSizeDesc{ 48, 6 },
    RGYNnediNSizeDesc{  8, 4 },
    RGYNnediNSizeDesc{ 16, 4 },
    RGYNnediNSizeDesc{ 32, 4 },
};

static constexpr std::array<int, 5> NNEDI_NNS_VALUE = { 16, 32, 64, 128, 256 };
#if defined(_WIN32) || defined(_WIN64)
static const TCHAR *NNEDI_DEFAULT_WEIGHT_FILE = _T("nnedi3_weights.bin");
#endif

struct RGYNnediPlaneValueRange {
    int planeRangeMode;
    int valMin;
    int valMax;
};

static const TCHAR *nnedi_nsize_name(const int nsize) {
    static const TCHAR *names[] = {
        _T("8x6"), _T("16x6"), _T("32x6"), _T("48x6"), _T("8x4"), _T("16x4"), _T("32x4")
    };
    return (0 <= nsize && nsize < (int)_countof(names)) ? names[nsize] : _T("unknown");
}

static RGYNnediPlaneValueRange nnediPlaneValueRange(const int clamp, const RGY_CSP csp, const RGY_PLANE plane, const int bitDepth) {
    const bool isAlpha = plane == RGY_PLANE_A;
    const bool isYuv = csp != RGY_CSP_Y8 && RGY_CSP_PLANES[csp] >= 3;
    int planeRangeMode = clamp;
    if (isAlpha) {
        planeRangeMode = 1;
    } else if (clamp != 1 && clamp != 4) {
        if (isYuv) {
            planeRangeMode = (plane == RGY_PLANE_Y) ? 2 : 3;
        } else {
            planeRangeMode = (clamp == 0) ? 2 : clamp;
        }
    }

    const int fullMax = (1 << bitDepth) - 1;
    const int scale = 1 << std::max(0, bitDepth - 8);
    RGYNnediPlaneValueRange valueRange;
    valueRange.planeRangeMode = planeRangeMode;
    switch (planeRangeMode) {
    case 2:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = 235 * scale;
        break;
    case 3:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = 240 * scale;
        break;
    case 4:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = fullMax;
        break;
    case 1:
    default:
        valueRange.valMin = 0;
        valueRange.valMax = fullMax;
        break;
    }
    return valueRange;
}

static bool nnediPlaneEnabled(const RGYNnediParam& prm, const int plane) {
    return 0 <= plane && plane < (int)prm.processPlane.size() && prm.processPlane[plane];
}

static bool nnediSupportedPlanarCsp(const RGY_CSP csp) {
    switch (csp) {
    case RGY_CSP_Y8:
    case RGY_CSP_Y16:
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_16:
    case RGY_CSP_YUV422:
    case RGY_CSP_YUV422_16:
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_16:
        return true;
    default:
        return false;
    }
}

static int nnediBytesPerSample(const RGY_CSP csp) {
    return (RGY_CSP_BIT_DEPTH[csp] > 8) ? 2 : 1;
}

static int nnediFindEnabledPlane(const RGYNnediParam& prm, const RGY_CSP csp, const bool reverse = false) {
    const int planes = RGY_CSP_PLANES[csp];
    for (int i = 0; i < planes; i++) {
        const int plane = reverse ? (planes - 1 - i) : i;
        if (nnediPlaneEnabled(prm, plane)) {
            return plane;
        }
    }
    return -1;
}

#if defined(_WIN32) || defined(_WIN64)
static tstring nnediDefaultWeightFilePath(HMODULE hModule) {
    const tstring filename = NNEDI_DEFAULT_WEIGHT_FILE;
    if (rgy_file_exists(filename.c_str())) {
        return filename;
    }

    const auto modulePath = getModulePath(hModule);
    if (modulePath.length() > 0) {
        const auto moduleDir = PathRemoveFileSpecFixed(modulePath).second;
        const auto path = PathCombineS(moduleDir, filename);
        if (rgy_file_exists(path.c_str())) {
            return path;
        }
    }

    const auto exeDir = getExeDir();
    if (exeDir.length() > 0) {
        const auto path = PathCombineS(exeDir, filename);
        if (rgy_file_exists(path.c_str())) {
            return path;
        }
    }

    return filename;
}
#endif

static __device__ __forceinline__ int nnedi_mirror_index_device(const int pos, const int length) {
    if (length <= 0) {
        return 0;
    }
    if (pos < 0) {
        return -pos - 1;
    }
    if (pos >= length) {
        return length - (pos - length) - 1;
    }
    return pos;
}

template<typename T> static __device__ __forceinline__ int4 nnedi_to_int4_device(const T v);
template<> __device__ __forceinline__ int4 nnedi_to_int4_device<uchar4>(const uchar4 v) { return make_int4(v.x, v.y, v.z, v.w); }
template<> __device__ __forceinline__ int4 nnedi_to_int4_device<ushort4>(const ushort4 v) { return make_int4(v.x, v.y, v.z, v.w); }

static __device__ __forceinline__ int nnedi_clamp_int(const int v, const int lo, const int hi) {
    return min(max(v, lo), hi);
}

static __device__ __forceinline__ float nnedi_expf_device(const float f) {
    return __expf(min(f, 88.0f));
}

static __device__ __forceinline__ int nnedi_round_to_pixel_match_device(const float value) {
    return (int)(value + 0.5f);
}

static __device__ __forceinline__ float4 nnedi_f4_fma_scalar(const int v, const float4 w, const float4 sum) {
    const float vf = (float)v;
    return make_float4(fmaf(vf, w.x, sum.x), fmaf(vf, w.y, sum.y), fmaf(vf, w.z, sum.z), fmaf(vf, w.w, sum.w));
}

static __device__ __forceinline__ float4 nnedi_f4_activate(const float4 v) {
    return make_float4(
        v.x / (fabsf(v.x) + 1.0f),
        v.y / (fabsf(v.y) + 1.0f),
        v.z / (fabsf(v.z) + 1.0f),
        v.w / (fabsf(v.w) + 1.0f));
}

static __device__ __forceinline__ float nnedi_dot4(const float4 a, const float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

template<typename Type, typename Type4>
__global__ void kernel_nnedi_pad_ref_and_copy_half_scalar_cuda(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstOffset,
    uint8_t *__restrict__ pRef, const int refPitch, const int refOffset,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcOffset,
    const int width, const int height,
    const int hpad, const int vpad
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int xBase = (gx << 2) - hpad;
    const int y = gy - vpad;
    const int paddedWidth = ((width + hpad * 2 + 3) >> 2) << 2;

    if (xBase >= paddedWidth - hpad || y >= height + vpad) {
        return;
    }

    const bool pady = (y < 0 || y >= height);
    const int srcy = nnedi_mirror_index_device(y, height);
    if (xBase >= 0 && xBase + 3 < width) {
        const Type *src = (const Type *)(pSrc + srcOffset + srcy * srcPitch + xBase * (int)sizeof(Type));
        const Type4 packed = *(const Type4 *)src;
        Type *ref = (Type *)(pRef + refOffset + (y + vpad) * refPitch + (xBase + hpad) * (int)sizeof(Type));
        *(Type4 *)ref = packed;
        if (!pady) {
            Type *dst = (Type *)(pDst + dstOffset + y * dstPitch + xBase * (int)sizeof(Type));
            *(Type4 *)dst = packed;
        }
        return;
    }

    for (int lane = 0; lane < 4; lane++) {
        const int x = xBase + lane;
        if (x >= paddedWidth - hpad) {
            continue;
        }
        const bool padx = (x < 0 || x >= width);
        const int srcx = nnedi_mirror_index_device(x, width);
        const Type v = *(const Type *)(pSrc + srcOffset + srcy * srcPitch + srcx * (int)sizeof(Type));

        *(Type *)(pRef + refOffset + (y + vpad) * refPitch + (x + hpad) * (int)sizeof(Type)) = v;

        if (!padx && !pady) {
            *(Type *)(pDst + dstOffset + y * dstPitch + x * (int)sizeof(Type)) = v;
        }
    }
}

static constexpr int NNEDI_TILE_GROUPS_X = 32;
static constexpr int NNEDI_TILE_ROWS = 16;
static constexpr int NNEDI_TILE_PIXELS_X = NNEDI_TILE_GROUPS_X * 4;
static constexpr int NNEDI_TILE_MASK_COUNT = NNEDI_TILE_GROUPS_X * NNEDI_TILE_ROWS;
static constexpr int NNEDI_TILE_MAX_CANDIDATES = NNEDI_TILE_MASK_COUNT * 4;
static constexpr int NNEDI_PRE_OUTPUT_LANES = 4;
static constexpr int NNEDI_PRE_SAMPLE_ROWS = 4;
static constexpr int NNEDI_PRE_ROW_TAPS = 16;
static constexpr int NNEDI_PRE_SAMPLE_COUNT = NNEDI_PRE_SAMPLE_ROWS * NNEDI_PRE_ROW_TAPS;
static constexpr int NNEDI_PRE_HIDDEN_WEIGHT4_OFFSET = 0;
static constexpr int NNEDI_PRE_HIDDEN_SCALE4_INDEX = NNEDI_PRE_HIDDEN_WEIGHT4_OFFSET + NNEDI_PRE_SAMPLE_COUNT;
static constexpr int NNEDI_PRE_HIDDEN_BIAS4_INDEX = NNEDI_PRE_HIDDEN_SCALE4_INDEX + 1;
static constexpr int NNEDI_PRE_OUTPUT_MIX4_OFFSET = NNEDI_PRE_HIDDEN_BIAS4_INDEX + 1;
static constexpr int NNEDI_PRE_OUTPUT_BIAS4_INDEX = NNEDI_PRE_OUTPUT_MIX4_OFFSET + NNEDI_PRE_OUTPUT_LANES;
static constexpr int NNEDI_PRE_WEIGHT_FLOAT4_COUNT = NNEDI_PRE_OUTPUT_BIAS4_INDEX + 1;

static __device__ __forceinline__ int nnedi_prescreen_valid_lane_mask_device(const int xpixel, const int width) {
    const int remain = width - xpixel;
    return (remain >= 4) ? 15 : ((remain <= 0) ? 0 : ((1 << remain) - 1));
}

static __device__ __forceinline__ int nnedi_prescreen_candidate_mask_device(const float4 result, const int validLaneMask) {
    const int bits =
        (result.x <= 0.0f ? 1 : 0)
        | (result.y <= 0.0f ? 2 : 0)
        | (result.z <= 0.0f ? 4 : 0)
        | (result.w <= 0.0f ? 8 : 0);
    return bits & validLaneMask;
}

static __device__ __forceinline__ int nnedi_prescreen_lane_count_device(const int mask) {
    return __popc((unsigned int)(mask & 15));
}

static __device__ __forceinline__ int4 nnedi_keys_cubic_fallback4_device(
    const int4 upperOuterPixels,
    const int4 upperInnerPixels,
    const int4 lowerInnerPixels,
    const int4 lowerOuterPixels,
    const int valMin,
    const int valMax
) {
    return make_int4(
        nnedi_clamp_int((upperOuterPixels.x * -3 + (upperInnerPixels.x + lowerInnerPixels.x) * 19 + lowerOuterPixels.x * -3 + 16) >> 5, valMin, valMax),
        nnedi_clamp_int((upperOuterPixels.y * -3 + (upperInnerPixels.y + lowerInnerPixels.y) * 19 + lowerOuterPixels.y * -3 + 16) >> 5, valMin, valMax),
        nnedi_clamp_int((upperOuterPixels.z * -3 + (upperInnerPixels.z + lowerInnerPixels.z) * 19 + lowerOuterPixels.z * -3 + 16) >> 5, valMin, valMax),
        nnedi_clamp_int((upperOuterPixels.w * -3 + (upperInnerPixels.w + lowerInnerPixels.w) * 19 + lowerOuterPixels.w * -3 + 16) >> 5, valMin, valMax));
}

template<typename Type, typename Type4>
static __device__ __forceinline__ void nnedi_prescreen_write_cubic4_device(
    uint8_t *__restrict__ pDst,
    const int dstPitch,
    const int dstOffset,
    const uint8_t *__restrict__ pRef,
    const int refPitch,
    const int refOffset,
    const int xbase,
    const int ybase,
    const int xpixel,
    const int valMin,
    const int valMax,
    const int laneMask
) {
    const Type4 *ref4 = (const Type4 *)(pRef + refOffset);
    const int refPitch4 = refPitch / (4 * (int)sizeof(Type));
    const int4 upperOuterPixels = nnedi_to_int4_device(ref4[(xbase + 2) + (ybase + 0) * refPitch4]);
    const int4 upperInnerPixels = nnedi_to_int4_device(ref4[(xbase + 2) + (ybase + 1) * refPitch4]);
    const int4 lowerInnerPixels = nnedi_to_int4_device(ref4[(xbase + 2) + (ybase + 2) * refPitch4]);
    const int4 lowerOuterPixels = nnedi_to_int4_device(ref4[(xbase + 2) + (ybase + 3) * refPitch4]);
    const int4 cubicInterpolatedPixels = nnedi_keys_cubic_fallback4_device(
        upperOuterPixels, upperInnerPixels, lowerInnerPixels, lowerOuterPixels, valMin, valMax);
    Type *dstPixels = (Type *)(pDst + dstOffset + ybase * dstPitch + xpixel * (int)sizeof(Type));
    if (laneMask & 1) dstPixels[0] = (Type)cubicInterpolatedPixels.x;
    if (laneMask & 2) dstPixels[1] = (Type)cubicInterpolatedPixels.y;
    if (laneMask & 4) dstPixels[2] = (Type)cubicInterpolatedPixels.z;
    if (laneMask & 8) dstPixels[3] = (Type)cubicInterpolatedPixels.w;
}

template<typename Type, typename Type4>
static __device__ __forceinline__ float4 nnedi_prescreen_classify4_device(
    const uint8_t *__restrict__ pRef,
    const int refPitch,
    const int refOffset,
    const int xbase,
    const int ybase,
    const float4 *__restrict__ prescreenWeights
) {
    const Type4 *ref4 = (const Type4 *)(pRef + refOffset);
    const int refPitch4 = refPitch / (4 * (int)sizeof(Type));
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int y = 0; y < NNEDI_PRE_SAMPLE_ROWS; y++) {
        const int refRow = (y + ybase) * refPitch4;
        const int4 v0 = nnedi_to_int4_device(ref4[xbase + 0 + refRow]);
        const int4 v1 = nnedi_to_int4_device(ref4[xbase + 1 + refRow]);
        const int4 v2 = nnedi_to_int4_device(ref4[xbase + 2 + refRow]);
        const int4 v3 = nnedi_to_int4_device(ref4[xbase + 3 + refRow]);
        const int4 v4 = nnedi_to_int4_device(ref4[xbase + 4 + refRow]);
        const float4 *rowWeights = prescreenWeights + NNEDI_PRE_HIDDEN_WEIGHT4_OFFSET + y * NNEDI_PRE_ROW_TAPS;
        sum = nnedi_f4_fma_scalar(v0.z, rowWeights[ 0], sum);
        sum = nnedi_f4_fma_scalar(v0.w, rowWeights[ 1], sum);
        sum = nnedi_f4_fma_scalar(v1.x, rowWeights[ 2], sum);
        sum = nnedi_f4_fma_scalar(v1.y, rowWeights[ 3], sum);
        sum = nnedi_f4_fma_scalar(v1.z, rowWeights[ 4], sum);
        sum = nnedi_f4_fma_scalar(v1.w, rowWeights[ 5], sum);
        sum = nnedi_f4_fma_scalar(v2.x, rowWeights[ 6], sum);
        sum = nnedi_f4_fma_scalar(v2.y, rowWeights[ 7], sum);
        sum = nnedi_f4_fma_scalar(v2.z, rowWeights[ 8], sum);
        sum = nnedi_f4_fma_scalar(v2.w, rowWeights[ 9], sum);
        sum = nnedi_f4_fma_scalar(v3.x, rowWeights[10], sum);
        sum = nnedi_f4_fma_scalar(v3.y, rowWeights[11], sum);
        sum = nnedi_f4_fma_scalar(v3.z, rowWeights[12], sum);
        sum = nnedi_f4_fma_scalar(v3.w, rowWeights[13], sum);
        sum = nnedi_f4_fma_scalar(v4.x, rowWeights[14], sum);
        sum = nnedi_f4_fma_scalar(v4.y, rowWeights[15], sum);
    }
    const float4 scale = prescreenWeights[NNEDI_PRE_HIDDEN_SCALE4_INDEX];
    const float4 bias = prescreenWeights[NNEDI_PRE_HIDDEN_BIAS4_INDEX];
    const float4 hiddenInput = make_float4(
        fmaf(sum.x, scale.x, bias.x),
        fmaf(sum.y, scale.y, bias.y),
        fmaf(sum.z, scale.z, bias.z),
        fmaf(sum.w, scale.w, bias.w));
    const float4 hiddenActivation = nnedi_f4_activate(hiddenInput);
    const float4 *outputMix = prescreenWeights + NNEDI_PRE_OUTPUT_MIX4_OFFSET;
    const float4 outputBias = prescreenWeights[NNEDI_PRE_OUTPUT_BIAS4_INDEX];
    return make_float4(
        nnedi_dot4(hiddenActivation, outputMix[0]) + outputBias.x,
        nnedi_dot4(hiddenActivation, outputMix[1]) + outputBias.y,
        nnedi_dot4(hiddenActivation, outputMix[2]) + outputBias.z,
        nnedi_dot4(hiddenActivation, outputMix[3]) + outputBias.w);
}

template<typename Type, typename Type4>
__global__ void kernel_nnedi_prescreen_cubic_cuda(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstOffset,
    const uint8_t *__restrict__ pRef, const int refPitch, const int refOffset,
    const float *__restrict__ weights,
    uint8_t *__restrict__ candidateMask, int *__restrict__ numblocks,
    const int width4, const int width, const int height, const int valMin, const int valMax
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = tx + ty * NNEDI_TILE_GROUPS_X;
    const int xbase = tx + blockIdx.x * NNEDI_TILE_GROUPS_X;
    const int ybase = ty + blockIdx.y * NNEDI_TILE_ROWS;

    __shared__ float4 prescreenWeights[NNEDI_PRE_WEIGHT_FLOAT4_COUNT];
    __shared__ int candidateCount[NNEDI_TILE_MASK_COUNT];

    const float4 *weights4 = (const float4 *)weights;
    for (int i = tid; i < NNEDI_PRE_WEIGHT_FLOAT4_COUNT; i += NNEDI_TILE_MASK_COUNT) {
        prescreenWeights[i] = weights4[i];
    }
    __syncthreads();

    float4 result = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const bool valid = xbase < width4 && ybase < height;
    const int xpixel = xbase << 2;
    const int validLaneMask = valid ? nnedi_prescreen_valid_lane_mask_device(xpixel, width) : 0;
    if (valid) {
        result = nnedi_prescreen_classify4_device<Type, Type4>(
            pRef, refPitch, refOffset, xbase, ybase,
            prescreenWeights);
    }

    const int mask = nnedi_prescreen_candidate_mask_device(result, validLaneMask);
    const int num = nnedi_prescreen_lane_count_device(mask);
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    candidateMask[bid * NNEDI_TILE_MASK_COUNT + tid] = (uint8_t)mask;

    candidateCount[tid] = num;
    __syncthreads();
    for (int stride = NNEDI_TILE_MASK_COUNT >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            candidateCount[tid] += candidateCount[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        numblocks[bid] = candidateCount[0];
    }

    if (num < 4 && valid) {
        nnedi_prescreen_write_cubic4_device<Type, Type4>(
            pDst, dstPitch, dstOffset,
            pRef, refPitch, refOffset,
            xbase, ybase, xpixel,
            valMin, valMax,
            validLaneMask);
    }
}

static __device__ __forceinline__ unsigned short nnedi_pack_tile_xy_device(const int x, const int y) {
    return (unsigned short)((y << 7) | x);
}

static __device__ __forceinline__ void nnedi_unpack_tile_xy_device(const unsigned short packed, int *x, int *y) {
    *x = (int)(packed & 127);
    *y = (int)(packed >> 7);
}

static __device__ __forceinline__ void nnedi_local_prefix512_device(int *prefix, const int tid, const int count) {
    prefix[tid] = count;
    __syncthreads();
    for (int offset = 1; offset < NNEDI_TILE_MASK_COUNT; offset <<= 1) {
        const int addend = (tid >= offset) ? prefix[tid - offset] : 0;
        __syncthreads();
        prefix[tid] += addend;
        __syncthreads();
    }
}

static __device__ __forceinline__ void nnedi_predictor_expand_tile_masks_device(
    unsigned short *candidateQueue,
    int *candidatePrefix,
    const uint8_t *candidateMask,
    const int maskOffset,
    const int tid
) {
    const int mask = (int)(candidateMask[maskOffset + tid] & 15);
    const int count = nnedi_prescreen_lane_count_device(mask);
    nnedi_local_prefix512_device(candidatePrefix, tid, count);

    int index = candidatePrefix[tid] - count;
    const int groupX = tid - (tid / NNEDI_TILE_GROUPS_X) * NNEDI_TILE_GROUPS_X;
    const int y = tid / NNEDI_TILE_GROUPS_X;
    const int x = groupX << 2;
    if (mask & 1) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 0, y);
    if (mask & 2) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 1, y);
    if (mask & 4) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 2, y);
    if (mask & 8) candidateQueue[index++] = nnedi_pack_tile_xy_device(x + 3, y);
    __syncthreads();
}

template<typename Type, int XDIA, int YDIA, int PRED_K>
static __device__ __forceinline__ void nnedi_load_patch_distributed_device(
    Type *patch,
    const uint8_t *pRef,
    const int refPitch,
    const int refOffset,
    const int x,
    const int y,
    const bool active,
    const int tx
) {
    for (int k = tx; k < PRED_K; k += 16) {
        const int row = k / XDIA;
        const int col = k - row * XDIA;
        Type v = (Type)0;
        if (active) {
            v = *(const Type *)(pRef + refOffset + (y + row) * refPitch + (x + col) * (int)sizeof(Type));
        }
        patch[k] = v;
    }
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_stats_u8_device(
    int *sumPart,
    int *sumsqPart,
    float *avg,
    float *stddev,
    float *invvar,
    const int row
) {
    int sumAll = 0;
    int sumsqAll = 0;
    int *sumPartRow = sumPart + row * LOCAL_X;
    int *sumsqPartRow = sumsqPart + row * LOCAL_X;
    for (int i = 0; i < LOCAL_X; i++) {
        sumAll += sumPartRow[i];
        sumsqAll += sumsqPartRow[i];
    }
    const float scale = 1.0f / (float)PRED_K;
    const float avg_ = (float)sumAll * scale;
    float stddev_ = (float)sumsqAll * scale - avg_ * avg_;
    float invvar_ = 0.0f;
    if (stddev_ <= 1.192092896e-07F) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrtf(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    avg[row] = avg_;
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_avg_device(
    int *sumPart,
    float *avg,
    const int row
) {
    int sumAll = 0;
    int *sumPartRow = sumPart + row * LOCAL_X;
    for (int i = 0; i < LOCAL_X; i++) {
        sumAll += sumPartRow[i];
    }
    avg[row] = (float)sumAll * (1.0f / (float)PRED_K);
}

template<int PRED_K, int LOCAL_X>
static __device__ __forceinline__ void nnedi_predictor_finalize_patch_stddev_device(
    float *varPart,
    float *stddev,
    float *invvar,
    const int row
) {
    const int partBase = row * LOCAL_X;
    float stddev_ = 0.0f;
    for (int i = 0; i < LOCAL_X; i++) {
        stddev_ += varPart[partBase + i];
    }
    stddev_ *= (1.0f / (float)PRED_K);
    float invvar_ = 0.0f;
    if (stddev_ <= 1.192092896e-07F) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrtf(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}

template<typename Type, int PRED_K, int NNS, int QUAL, int LOCAL_X>
static __device__ __forceinline__ float2 nnedi_predictor_lane_vote_device(
    const Type *patch,
    const float2 *weightsBody,
    const float2 *weightsBias,
    const int tx,
    const int q,
    const float invvar
) {
    static constexpr int PRED_GROUPS = NNS / LOCAL_X;
    static constexpr int PRED_BLOCK_FLOAT2_COUNT = PRED_K * LOCAL_X;
    static constexpr int PRED_QUAL_BODY_FLOAT2_COUNT = PRED_GROUPS * PRED_BLOCK_FLOAT2_COUNT;
    float weightedElliottVoteSum = 0.0f;
    float softmaxVoteWeightSum = 0.0f;
    const float2 *neuronBlockWeights = weightsBody + q * PRED_QUAL_BODY_FLOAT2_COUNT;
    const float2 *neuronBiasPtr = weightsBias + q * NNS + tx;
    for (int neuronGroup = 0; neuronGroup < PRED_GROUPS; neuronGroup++) {
        const float2 *sampleWeights = neuronBlockWeights + tx;
        const Type *patchPtr = patch;
        float2 weightedPatchSums = make_float2(0.0f, 0.0f);
        for (int sampleIndex = 0; sampleIndex < PRED_K; sampleIndex++) {
            const int patchPixelValue = (int)(*patchPtr);
            weightedPatchSums.x = fmaf((float)patchPixelValue, sampleWeights->x, weightedPatchSums.x);
            weightedPatchSums.y = fmaf((float)patchPixelValue, sampleWeights->y, weightedPatchSums.y);
            patchPtr++;
            sampleWeights += LOCAL_X;
        }

        const float2 neuronBias = *neuronBiasPtr;
        const float softmaxLogit = fmaf(weightedPatchSums.x, invvar, neuronBias.x);
        const float elliottInput = fmaf(weightedPatchSums.y, invvar, neuronBias.y);
        const float softmaxVoteWeight = nnedi_expf_device(softmaxLogit);
        weightedElliottVoteSum += softmaxVoteWeight * (elliottInput / (1.0f + fabsf(elliottInput)));
        softmaxVoteWeightSum += softmaxVoteWeight;
        neuronBlockWeights += PRED_BLOCK_FLOAT2_COUNT;
        neuronBiasPtr += LOCAL_X;
    }
    return make_float2(weightedElliottVoteSum, softmaxVoteWeightSum);
}

template<int LOCAL_X>
static __device__ __forceinline__ float nnedi_predictor_merge_votes_device(
    const float *weightedElliottVoteParts,
    const float *softmaxVoteWeightParts,
    const int row,
    const float avg,
    const float stddev,
    const float result
) {
    const int partBase = row * LOCAL_X;
    float weightedElliottVoteSum = 0.0f;
    float softmaxVoteWeightSum = 0.0f;
    for (int i = 0; i < LOCAL_X; i++) {
        weightedElliottVoteSum += weightedElliottVoteParts[partBase + i];
        softmaxVoteWeightSum += softmaxVoteWeightParts[partBase + i];
    }
    if (softmaxVoteWeightSum > 1.0e-10f) {
        return result + ((5.0f * weightedElliottVoteSum) / softmaxVoteWeightSum) * stddev + avg;
    }
    return result + avg;
}

template<typename Type, int QUAL>
static __device__ __forceinline__ void nnedi_predictor_write_result_device(
    uint8_t *pDst,
    const int dstPitch,
    const int dstOffset,
    const int x,
    const int y,
    const float result,
    const int valMin,
    const int valMax
) {
    const float scale = 1.0f / (float)QUAL;
    *(Type *)(pDst + dstOffset + y * dstPitch + x * (int)sizeof(Type)) =
        (Type)nnedi_clamp_int(nnedi_round_to_pixel_match_device(result * scale), valMin, valMax);
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int NNS, int QUAL>
__global__ void kernel_nnedi_predictor_network_cuda(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstOffset,
    const uint8_t *__restrict__ pRef, const int refPitch, const int refOffset,
    const uint8_t *__restrict__ candidateMask, const int *__restrict__ numblocks,
    const float *__restrict__ weights,
    const int width4, const int height, const int valMin, const int valMax
) {
    static constexpr int LOCAL_X = 16;
    static constexpr int LOCAL_Y = 32;
    static constexpr int PRED_K = XDIA * YDIA;
    static constexpr int PRED_GROUPS = NNS / LOCAL_X;
    static constexpr int PRED_BLOCK_FLOAT2_COUNT = PRED_K * LOCAL_X;
    static constexpr int PRED_QUAL_BODY_FLOAT2_COUNT = PRED_GROUPS * PRED_BLOCK_FLOAT2_COUNT;
    static constexpr int PRED_BODY_FLOAT2_COUNT = QUAL * PRED_QUAL_BODY_FLOAT2_COUNT;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int maskOffset = bid * NNEDI_TILE_MASK_COUNT;
    const int tid = ty * LOCAL_X + tx;
    const int xbase = blockIdx.x * NNEDI_TILE_PIXELS_X;
    const int ybase = blockIdx.y * NNEDI_TILE_ROWS;

    __shared__ Type patchPixels[LOCAL_Y * PRED_K];
    __shared__ unsigned short candidateQueue[NNEDI_TILE_MAX_CANDIDATES];
    __shared__ int candidatePrefix[NNEDI_TILE_MASK_COUNT];
    __shared__ int sumPart[LOCAL_Y * LOCAL_X];
#if BIT_DEPTH <= 8
    __shared__ int sumsqPart[LOCAL_Y * LOCAL_X];
#else
    __shared__ float varPart[LOCAL_Y * LOCAL_X];
#endif
    __shared__ float avg[LOCAL_Y];
    __shared__ float stddev[LOCAL_Y];
    __shared__ float invvar[LOCAL_Y];
    __shared__ float weightedElliottVoteParts[LOCAL_Y * LOCAL_X];
    __shared__ float softmaxVoteWeightParts[LOCAL_Y * LOCAL_X];

    const int nb = numblocks[bid];
    if (nb <= 0) {
        return;
    }
    const float2 *weightsBody = (const float2 *)weights;
    const float2 *weightsBias = weightsBody + PRED_BODY_FLOAT2_COUNT;

    nnedi_predictor_expand_tile_masks_device(candidateQueue, candidatePrefix, candidateMask, maskOffset, tid);

    for (int b = 0; b < nb; b += LOCAL_Y) {
        int localX = 0;
        int localY = 0;
        const bool active = (b + ty < nb);
        if (active) {
            nnedi_unpack_tile_xy_device(candidateQueue[b + ty], &localX, &localY);
        }
        const int x = xbase + localX;
        const int y = ybase + localY;

        Type *patch = &patchPixels[ty * PRED_K];
        nnedi_load_patch_distributed_device<Type, XDIA, YDIA, PRED_K>(patch, pRef, refPitch, refOffset, x, y, active, tx);
        __syncthreads();

        float avgValue = 0.0f;
        float stddevValue = 0.0f;
        float invvarValue = 0.0f;
#if BIT_DEPTH <= 8
        int sum = 0;
        int sumsq = 0;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            const int v = (int)patch[k];
            sum += v;
            sumsq += v * v;
        }
        sumPart[ty * LOCAL_X + tx] = sum;
        sumsqPart[ty * LOCAL_X + tx] = sumsq;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_stats_u8_device<PRED_K, LOCAL_X>(sumPart, sumsqPart, avg, stddev, invvar, ty);
        }
        __syncthreads();
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#else
        int sum = 0;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            sum += (int)patch[k];
        }
        sumPart[ty * LOCAL_X + tx] = sum;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_avg_device<PRED_K, LOCAL_X>(sumPart, avg, ty);
        }
        __syncthreads();

        float sumsq = 0.0f;
        for (int k = tx; k < PRED_K; k += LOCAL_X) {
            const float diff = (float)patch[k] - avg[ty];
            sumsq = fmaf(diff, diff, sumsq);
        }
        varPart[ty * LOCAL_X + tx] = sumsq;
        __syncthreads();

        if (tx == 0) {
            nnedi_predictor_finalize_patch_stddev_device<PRED_K, LOCAL_X>(varPart, stddev, invvar, ty);
        }
        __syncthreads();
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#endif

        float result = 0.0f;
        const int partBase = ty * LOCAL_X;
        for (int q = 0; q < QUAL; q++) {
            const float2 vote = nnedi_predictor_lane_vote_device<Type, PRED_K, NNS, QUAL, LOCAL_X>(
                patch, weightsBody, weightsBias,
                tx, q, invvarValue);
            weightedElliottVoteParts[partBase + tx] = vote.x;
            softmaxVoteWeightParts[partBase + tx] = vote.y;
            __syncthreads();

            if (tx == 0) {
                result = nnedi_predictor_merge_votes_device<LOCAL_X>(
                    weightedElliottVoteParts, softmaxVoteWeightParts,
                    ty, avgValue, stddevValue, result);
            }
            __syncthreads();
        }

        if (tx == 0 && active) {
            nnedi_predictor_write_result_device<Type, QUAL>(pDst, dstPitch, dstOffset, x, y, result, valMin, valMax);
        }
        __syncthreads();
    }
}

static RGY_ERR cudaCheckNnedi(const cudaError_t err) {
    return (err == cudaSuccess) ? RGY_ERR_NONE : err_to_rgy(err);
}

template<typename Type, typename Type4>
static RGY_ERR launchPadRefAndCopy(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane, const RGYFrameInfo& srcPlane,
    int dstOffset, int srcOffset, int srcPitch, int fieldHeight, int refOffset, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid(((srcPlane.width + RGY_NNEDI_HPAD * 2 + 3) >> 2) + block.x - 1, fieldHeight + RGY_NNEDI_VPAD * 2 + block.y - 1);
    grid.x /= block.x;
    grid.y /= block.y;
    kernel_nnedi_pad_ref_and_copy_half_scalar_cuda<Type, Type4><<<grid, block, 0, stream>>>(
        (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0] * 2, dstOffset,
        (uint8_t *)refPlane.ptr[0], refPlane.pitch[0], refOffset,
        (const uint8_t *)srcPlane.ptr[0], srcPitch, srcOffset,
        srcPlane.width, fieldHeight,
        RGY_NNEDI_HPAD, RGY_NNEDI_VPAD);
    return cudaCheckNnedi(cudaGetLastError());
}

template<typename Type, typename Type4>
static RGY_ERR launchPrescreen(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *prescreenerWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax, cudaStream_t stream) {
    dim3 block(NNEDI_TILE_GROUPS_X, NNEDI_TILE_ROWS);
    dim3 grid((width4 + NNEDI_TILE_GROUPS_X - 1) / NNEDI_TILE_GROUPS_X,
        (height + NNEDI_TILE_ROWS - 1) / NNEDI_TILE_ROWS);
    kernel_nnedi_prescreen_cubic_cuda<Type, Type4><<<grid, block, 0, stream>>>(
        (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0] * 2, dstOffset,
        (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0], refEvalOffset,
        (const float *)prescreenerWeightBuf->ptr,
        (uint8_t *)workNNBuf->ptr, (int *)numBlocksBuf->ptr,
        width4, dstPlane.width, height, valMin, valMax);
    return cudaCheckNnedi(cudaGetLastError());
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int NNS, int QUAL>
static RGY_ERR launchPredictorImpl(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax, cudaStream_t stream) {
    dim3 block(16, 32);
    dim3 grid((width4 + NNEDI_TILE_GROUPS_X - 1) / NNEDI_TILE_GROUPS_X,
        (height + NNEDI_TILE_ROWS - 1) / NNEDI_TILE_ROWS);
    kernel_nnedi_predictor_network_cuda<Type, BIT_DEPTH, XDIA, YDIA, NNS, QUAL><<<grid, block, 0, stream>>>(
        (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0] * 2, dstOffset,
        (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0], refEvalOffset,
        (const uint8_t *)workNNBuf->ptr, (const int *)numBlocksBuf->ptr,
        (const float *)predictorWeightBuf->ptr,
        width4, height, valMin, valMax);
    return cudaCheckNnedi(cudaGetLastError());
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA, int QUAL>
static RGY_ERR launchPredictorNns(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nns, cudaStream_t stream) {
    switch (nns) {
    case 16:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 16,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 32:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 32,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 64:  return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 64,  QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 128: return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 128, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    case 256: return launchPredictorImpl<Type, BIT_DEPTH, XDIA, YDIA, 256, QUAL>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, stream);
    default:  return RGY_ERR_INVALID_PARAM;
    }
}

template<typename Type, int BIT_DEPTH, int XDIA, int YDIA>
static RGY_ERR launchPredictorQual(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nns, int quality, cudaStream_t stream) {
    switch (quality) {
    case VPP_NNEDI_QUALITY_FAST: return launchPredictorNns<Type, BIT_DEPTH, XDIA, YDIA, VPP_NNEDI_QUALITY_FAST>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    case VPP_NNEDI_QUALITY_SLOW: return launchPredictorNns<Type, BIT_DEPTH, XDIA, YDIA, VPP_NNEDI_QUALITY_SLOW>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, stream);
    default: return RGY_ERR_INVALID_PARAM;
    }
}

template<typename Type, int BIT_DEPTH>
static RGY_ERR launchPredictorNsize(RGYFrameInfo& dstPlane, RGYFrameInfo& refPlane,
    const CUMemBuf *predictorWeightBuf, CUMemBuf *workNNBuf, CUMemBuf *numBlocksBuf,
    int dstOffset, int refEvalOffset, int width4, int height, int valMin, int valMax,
    int nsize, int nns, int quality, cudaStream_t stream) {
    switch (nsize) {
    case VPP_NNEDI_NSIZE_8x6:  return launchPredictorQual<Type, BIT_DEPTH,  8, 6>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_16x6: return launchPredictorQual<Type, BIT_DEPTH, 16, 6>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_32x6: return launchPredictorQual<Type, BIT_DEPTH, 32, 6>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_48x6: return launchPredictorQual<Type, BIT_DEPTH, 48, 6>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_8x4:  return launchPredictorQual<Type, BIT_DEPTH,  8, 4>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_16x4: return launchPredictorQual<Type, BIT_DEPTH, 16, 4>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    case VPP_NNEDI_NSIZE_32x4: return launchPredictorQual<Type, BIT_DEPTH, 32, 4>(dstPlane, refPlane, predictorWeightBuf, workNNBuf, numBlocksBuf, dstOffset, refEvalOffset, width4, height, valMin, valMax, nns, quality, stream);
    default: return RGY_ERR_INVALID_PARAM;
    }
}

} // namespace

RGYNnediParam::RGYNnediParam() :
    enable(false),
    processPlane{ true, true, true, false },
    field(VPP_NNEDI_FIELD_BOB),
    nsize(VPP_NNEDI_NSIZE_16x6),
    nns(32),
    quality(VPP_NNEDI_QUALITY_FAST),
    prescreen(2),
    errortype(VPP_NNEDI_ETYPE_ABS),
    doubleHeight(false),
    weightfile(_T("")) {
    clamp = 1;
}

bool RGYNnediParam::operator==(const RGYNnediParam& x) const {
    return enable == x.enable
        && processPlane == x.processPlane
        && field == x.field
        && nsize == x.nsize
        && nns == x.nns
        && quality == x.quality
        && prescreen == x.prescreen
        && errortype == x.errortype
        && clamp == x.clamp
        && doubleHeight == x.doubleHeight
        && weightfile == x.weightfile;
}

bool RGYNnediParam::operator!=(const RGYNnediParam& x) const {
    return !(*this == x);
}

tstring RGYNnediParam::print() const {
    const auto nsizeIndex = (int)nsize;
    return strsprintf(
        _T("nnedi: field %s, nsize %s, nns %d, quality %s\n")
        _T("                         prescreen %d, errortype %s, clamp %d, double_height %s, weight \"%s\""),
        get_cx_desc(list_vpp_nnedi_field, field),
        nnedi_nsize_name(nsizeIndex),
        nns,
        get_cx_desc(list_vpp_nnedi_quality, quality),
        prescreen,
        get_cx_desc(list_vpp_nnedi_error_type, errortype),
        clamp,
        doubleHeight ? _T("on") : _T("off"),
        ((weightfile.length()) ? weightfile.c_str() : _T("default")));
}

const RGYNnediNSizeDesc& rgy_nnedi_nsize_desc(const int nsize) {
    return NNEDI_NSIZE_DESC[nsize];
}

int rgy_nnedi_nns_value(const int nns) {
    return (0 <= nns && nns < (int)NNEDI_NNS_VALUE.size()) ? NNEDI_NNS_VALUE[nns] : 0;
}

int rgy_nnedi_nns_index(const int nns) {
    const auto it = std::find(NNEDI_NNS_VALUE.begin(), NNEDI_NNS_VALUE.end(), nns);
    return (it == NNEDI_NNS_VALUE.end()) ? -1 : (int)std::distance(NNEDI_NNS_VALUE.begin(), it);
}

NVEncFilterParamNnedi::NVEncFilterParamNnedi() :
    nnedi(),
    compute_capability(std::make_pair(0, 0)),
    hModule(NULL),
    timebase() {
}

NVEncFilterNnedi::NVEncFilterNnedi() :
    NVEncFilter(),
    m_weights(),
    m_refBuf(),
    m_prescreenerWeightBuf(),
    m_predictorWeightBuf(),
    m_workNNBuf(),
    m_numBlocksBuf(),
    m_tileGroupsX(NNEDI_TILE_GROUPS_X),
    m_tileRows(NNEDI_TILE_ROWS),
    m_predLocalX(16),
    m_predLocalY(32),
    m_defaultTff(true) {
    m_name = _T("nnedi");
}

NVEncFilterNnedi::~NVEncFilterNnedi() {
    close();
}

RGY_ERR NVEncFilterNnedi::validateParam(const RGYNnediParam& prm) {
    const auto field = (int)prm.field;
    if (field < -2 || 3 < field) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid field=%d.\n"), field);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.doubleHeight && (field < -1 || 1 < field)) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI double_height=true supports only field=auto,top,bottom.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto nsizeIndex = (int)prm.nsize;
    if (nsizeIndex < 0 || (int)NNEDI_NSIZE_DESC.size() <= nsizeIndex) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid nsize=%d.\n"), nsizeIndex);
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_nnedi_nns_index(prm.nns) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid nns=%d, expected 16, 32, 64, 128, or 256.\n"), prm.nns);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.quality != VPP_NNEDI_QUALITY_FAST && prm.quality != VPP_NNEDI_QUALITY_SLOW) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid quality=%d.\n"), (int)prm.quality);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.clamp < 0 || 4 < prm.clamp) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid clamp=%d, expected 0-4.\n"), prm.clamp);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.prescreen < 2 || 4 < prm.prescreen) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI unsupported prescreen=%d, supported prescreen values are 2, 3, and 4; prescreen=0/1 use an unsupported prescreener path.\n"), prm.prescreen);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= prm.errortype) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI invalid errortype=%d.\n"), (int)prm.errortype);
        return RGY_ERR_INVALID_PARAM;
    }
    if (std::none_of(prm.processPlane.begin(), prm.processPlane.end(), [](const bool enabled) { return enabled; })) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI requires at least one target plane to be enabled.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

std::shared_ptr<const std::vector<uint8_t>> NVEncFilterNnedi::readWeights(const tstring& weightFile, HMODULE hModule) {
    const auto expectedFileSize = RGYNnediParam::WEIGHTS_FILE_SIZE;
    uint64_t weightFileSize = 0;

#if !(defined(_WIN32) || defined(_WIN64))
    if (weightFile.length() == 0) {
        void *pDataPtr = nullptr;
        weightFileSize = getEmbeddedResource(&pDataPtr, _T("NNEDI_WEIGHTBIN"), _T("EXE_DATA"), hModule);
        if (pDataPtr == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get NNEDI weights data.\n"));
            return nullptr;
        }
        if (expectedFileSize != weightFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("NNEDI weights data has unexpected size %lld [expected: %u].\n"),
                (long long int)weightFileSize, expectedFileSize);
            return nullptr;
        }
        return std::make_shared<const std::vector<uint8_t>>(
            (const uint8_t *)pDataPtr, (const uint8_t *)pDataPtr + weightFileSize);
    }
    const auto weightFilePath = weightFile;
#else
    const auto weightFilePath = (weightFile.length() > 0) ? weightFile : nnediDefaultWeightFilePath(hModule);
#endif

    if (!rgy_file_exists(weightFilePath.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI weight file \"%s\" does not exist.\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (!rgy_get_filesize(weightFilePath.c_str(), &weightFileSize)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get filesize of NNEDI weight file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (weightFileSize != expectedFileSize) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI weights file \"%s\" has unexpected file size %lld [expected: %u].\n"),
            weightFilePath.c_str(), (long long int)weightFileSize, expectedFileSize);
        return nullptr;
    }

    auto weights = std::make_shared<std::vector<uint8_t>>(weightFileSize);
    std::ifstream fin(weightFilePath, std::ios::in | std::ios::binary);
    if (!fin.good()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to open NNEDI weights file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (fin.read((char *)weights->data(), weights->size()).gcount() != (int64_t)weights->size()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to read NNEDI weights file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    return weights;
}

RGY_ERR NVEncFilterNnedi::initParams(const std::shared_ptr<NVEncFilterParamNnedi> prm) {
    auto err = validateParam(prm->nnedi);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    if (bitDepth != 8 && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI supports only 8-bit or 16-bit planar input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    m_weights = readWeights(prm->nnedi.weightfile, prm->hModule);
    if (!m_weights) {
        return RGY_ERR_INVALID_PARAM;
    }
    if ((m_weights->size() % sizeof(float)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI weights data is not float-aligned.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto selfCheck = rgy_filter_nnedi_weights_self_check();
    if (!selfCheck.success) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI weights self-check failed: %s.\n"), char_to_tstring(selfCheck.message).c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFilterNnediWeightsParam weightsParam;
    weightsParam.nsize = prm->nnedi.nsize;
    weightsParam.nns = prm->nnedi.nns;
    weightsParam.quality = prm->nnedi.quality;
    weightsParam.prescreen = prm->nnedi.prescreen;
    weightsParam.errortype = prm->nnedi.errortype;
    weightsParam.bitsPerPixel = bitDepth;
    std::string weightsError;
    if (!rgy_filter_nnedi_transform_weights(m_transformedWeights,
        reinterpret_cast<const float *>(m_weights->data()), m_weights->size() / sizeof(float), weightsParam, &weightsError)) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI weights transform failed: %s.\n"), char_to_tstring(weightsError).c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_transformedWeights.prescreenerFp32.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI fp32 prescreener weights are empty.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_transformedWeights.predictorFp32.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI fp32 predictor weights are empty.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnedi::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto err = initParams(prm);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    if (bitDepth != 8 && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI supports only 8-bit or 16-bit planar input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!nnediSupportedPlanarCsp(prm->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI supports only planar Y/YV12/YUV422/YUV444 8-bit or 16-bit input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (nnediFindEnabledPlane(prm->nnedi, prm->frameIn.csp) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI requires at least one target plane present in the input format.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.doubleHeight) {
        const int planes = RGY_CSP_PLANES[prm->frameIn.csp];
        for (int iplane = 0; iplane < planes; iplane++) {
            if (!nnediPlaneEnabled(prm->nnedi, iplane)) {
                AddMessage(RGY_LOG_ERROR, _T("NNEDI double_height=true changes output height, so all existing input planes must be enabled; plane %d is disabled.\n"), iplane);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }
    if ((prm->frameIn.height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("NNEDI requires even input height.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;
    RGYNnediTopology topology;
    err = rgy_nnedi_resolve_topology(&topology, (int)prm->nnedi.field, m_defaultTff);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    if (prm->nnedi.doubleHeight) {
        prm->frameOut.height *= 2;
    }
    prm->baseFps *= topology.fpsMultiplier;
    m_pathThrough &= ~FILTER_PATHTHROUGH_PICSTRUCT;
    if (topology.doubleRate) {
        m_pathThrough &= ~FILTER_PATHTHROUGH_TIMESTAMP;
    }

    setFilterInfo(prm->nnedi.print());

    const int outputSlots = topology.frameMultiplier;
    err = AllocFrameBuf(prm->frameOut, outputSlots);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output frames: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    RGYFrameInfo refFrame = prm->frameOut;
    refFrame.width += RGY_NNEDI_HPAD * 4;
    refFrame.height = (refFrame.height >> 1) + RGY_NNEDI_VPAD * 4;
    m_refBuf.clear();
    for (int i = 0; i < outputSlots; i++) {
        auto ref = std::make_unique<CUFrameBuf>(refFrame);
        ref->releasePtr();
        err = ref->alloc();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate NNEDI ref buffer %d: %s.\n"), i, get_err_mes(err));
            return err;
        }
        m_refBuf.push_back(std::move(ref));
    }

    m_prescreenerWeightBuf = std::make_unique<CUMemBuf>(m_transformedWeights.prescreenerFp32.size() * sizeof(m_transformedWeights.prescreenerFp32[0]));
    err = m_prescreenerWeightBuf->alloc();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate NNEDI prescreener weights buffer.\n"));
        return err;
    }
    err = cudaCheckNnedi(cudaMemcpy(m_prescreenerWeightBuf->ptr, m_transformedWeights.prescreenerFp32.data(), m_prescreenerWeightBuf->nSize, cudaMemcpyHostToDevice));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy NNEDI prescreener weights buffer.\n"));
        return err;
    }
    m_predictorWeightBuf = std::make_unique<CUMemBuf>(m_transformedWeights.predictorFp32.size() * sizeof(m_transformedWeights.predictorFp32[0]));
    err = m_predictorWeightBuf->alloc();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate NNEDI predictor weights buffer.\n"));
        return err;
    }
    err = cudaCheckNnedi(cudaMemcpy(m_predictorWeightBuf->ptr, m_transformedWeights.predictorFp32.data(), m_predictorWeightBuf->nSize, cudaMemcpyHostToDevice));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy NNEDI predictor weights buffer.\n"));
        return err;
    }

    int maxWidth4 = 0;
    int maxHeight = 0;
    const auto planes = RGY_CSP_PLANES[prm->frameIn.csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto plane = getPlane(&prm->frameOut, (RGY_PLANE)iplane);
        maxWidth4 = std::max(maxWidth4, (plane.width + 3) >> 2);
        maxHeight = std::max(maxHeight, plane.height >> 1);
    }
    const int blocksX = (maxWidth4 + m_tileGroupsX - 1) / m_tileGroupsX;
    const int blocksY = (maxHeight + m_tileRows - 1) / m_tileRows;
    const size_t numBlocks = std::max(1, blocksX * blocksY);
    const size_t candidateMaskBytes = numBlocks * m_tileGroupsX * m_tileRows;
    const size_t numBlocksBytes = numBlocks * sizeof(int);
    m_workNNBuf.clear();
    m_numBlocksBuf.clear();
    for (int i = 0; i < outputSlots * planes; i++) {
        auto workNN = std::make_unique<CUMemBuf>(candidateMaskBytes);
        auto numBlock = std::make_unique<CUMemBuf>(numBlocksBytes);
        err = workNN->alloc();
        if (err == RGY_ERR_NONE) {
            err = numBlock->alloc();
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate NNEDI prescreen work buffers.\n"));
            return err;
        }
        m_workNNBuf.push_back(std::move(workNN));
        m_numBlocksBuf.push_back(std::move(numBlock));
    }

    m_param = pParam;
    return RGY_ERR_NONE;
}

bool NVEncFilterNnedi::getInputTff(const RGYFrameInfo *frame) const {
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

void NVEncFilterNnedi::setDoubleRateTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp    = pInputFrame->timestamp;
    ppOutputFrames[0]->duration     = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp    = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration     = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR NVEncFilterNnedi::prepareFieldReference(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream) {
    if (outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    const bool doubleHeight = prm->nnedi.doubleHeight;
    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!nnediPlaneEnabled(prm->nnedi, iplane)) {
            continue;
        }
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (srcPlane.ptr[0] == nullptr || dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const auto dstOffset = frameMap.sourceFieldOffset * dstPlane.pitch[0];
        const auto srcOffset = doubleHeight ? 0 : frameMap.sourceFieldOffset * srcPlane.pitch[0];
        const auto srcPitch = doubleHeight ? srcPlane.pitch[0] : srcPlane.pitch[0] * 2;
        const auto fieldHeight = doubleHeight ? srcPlane.height : (srcPlane.height >> 1);
        const auto refBaseHpad = (refPlane.width - srcPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - fieldHeight) >> 1;
        const auto refOffset = refBaseVpad * refPlane.pitch[0] + refBaseHpad * nnediBytesPerSample(pInputFrame->csp);
        auto err = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8)
            ? launchPadRefAndCopy<uint16_t, ushort4>(dstPlane, refPlane, srcPlane, (int)dstOffset, (int)srcOffset, (int)srcPitch, fieldHeight, (int)refOffset, stream)
            : launchPadRefAndCopy<uint8_t, uchar4>(dstPlane, refPlane, srcPlane, (int)dstOffset, (int)srcOffset, (int)srcPitch, fieldHeight, (int)refOffset, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_nnedi_pad_ref_and_copy_half_scalar (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnedi::classifyPixelsAndSeedOutput(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream) {
    if (!m_prescreenerWeightBuf || outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!nnediPlaneEnabled(prm->nnedi, iplane)) {
            continue;
        }
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const int workIndex = outputSlot * planes + iplane;
        if (workIndex < 0 || workIndex >= (int)m_workNNBuf.size() || workIndex >= (int)m_numBlocksBuf.size()) {
            return RGY_ERR_INVALID_PARAM;
        }

        const auto refBaseHpad = (refPlane.width - dstPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - (dstPlane.height >> 1)) >> 1;
        const int bytesPerSample = nnediBytesPerSample(pInputFrame->csp);
        const auto refOriginalOffset = (refBaseVpad + RGY_NNEDI_VPAD) * refPlane.pitch[0] + (refBaseHpad + RGY_NNEDI_HPAD) * bytesPerSample;
        const auto refEvalOffset = refOriginalOffset + frameMap.evalRefOffsetY * refPlane.pitch[0] - refPlane.pitch[0] - 8 * bytesPerSample;
        const auto dstOffset = (int)frameMap.generateField * dstPlane.pitch[0];
        const int width4 = (dstPlane.width + 3) >> 2;
        const int height = dstPlane.height >> 1;
        const auto valueRange = nnediPlaneValueRange(prm->nnedi.clamp, pInputFrame->csp, (RGY_PLANE)iplane, RGY_CSP_BIT_DEPTH[pInputFrame->csp]);

        auto err = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8)
            ? launchPrescreen<uint16_t, ushort4>(dstPlane, refPlane, m_prescreenerWeightBuf.get(), m_workNNBuf[workIndex].get(), m_numBlocksBuf[workIndex].get(), (int)dstOffset, (int)refEvalOffset, width4, height, valueRange.valMin, valueRange.valMax, stream)
            : launchPrescreen<uint8_t, uchar4>(dstPlane, refPlane, m_prescreenerWeightBuf.get(), m_workNNBuf[workIndex].get(), m_numBlocksBuf[workIndex].get(), (int)dstOffset, (int)refEvalOffset, width4, height, valueRange.valMin, valueRange.valMax, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_nnedi_prescreen_cubic (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnedi::resolveClassifiedPixels(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap, cudaStream_t stream) {
    if (!m_predictorWeightBuf || outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!nnediPlaneEnabled(prm->nnedi, iplane)) {
            continue;
        }
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const int workIndex = outputSlot * planes + iplane;
        if (workIndex < 0 || workIndex >= (int)m_workNNBuf.size() || workIndex >= (int)m_numBlocksBuf.size()) {
            return RGY_ERR_INVALID_PARAM;
        }

        const auto refBaseHpad = (refPlane.width - dstPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - (dstPlane.height >> 1)) >> 1;
        const int bytesPerSample = nnediBytesPerSample(pInputFrame->csp);
        const auto refOriginalOffset = (refBaseVpad + RGY_NNEDI_VPAD) * refPlane.pitch[0] + (refBaseHpad + RGY_NNEDI_HPAD) * bytesPerSample;
        const auto refEvalOffset = refOriginalOffset + frameMap.evalRefOffsetY * refPlane.pitch[0]
            - (((m_transformedWeights.layout.ydia >> 1) - 1) * refPlane.pitch[0] + ((m_transformedWeights.layout.xdia >> 1) - 1) * bytesPerSample);
        const auto dstOffset = (int)frameMap.generateField * dstPlane.pitch[0];
        const int width4 = (dstPlane.width + 3) >> 2;
        const int height = dstPlane.height >> 1;
        const auto valueRange = nnediPlaneValueRange(prm->nnedi.clamp, pInputFrame->csp, (RGY_PLANE)iplane, RGY_CSP_BIT_DEPTH[pInputFrame->csp]);

        auto err = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8)
            ? launchPredictorNsize<uint16_t, 16>(dstPlane, refPlane, m_predictorWeightBuf.get(), m_workNNBuf[workIndex].get(), m_numBlocksBuf[workIndex].get(), (int)dstOffset, (int)refEvalOffset, width4, height, valueRange.valMin, valueRange.valMax, (int)prm->nnedi.nsize, prm->nnedi.nns, (int)prm->nnedi.quality, stream)
            : launchPredictorNsize<uint8_t, 8>(dstPlane, refPlane, m_predictorWeightBuf.get(), m_workNNBuf[workIndex].get(), m_numBlocksBuf[workIndex].get(), (int)dstOffset, (int)refEvalOffset, width4, height, valueRange.valMin, valueRange.valMax, (int)prm->nnedi.nsize, prm->nnedi.nns, (int)prm->nnedi.quality, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_nnedi_predictor_network (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnedi::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYNnediTopology topology;
    auto err = rgy_nnedi_resolve_topology(&topology, (int)prm->nnedi.field, getInputTff(pInputFrame));
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const bool doubleHeight = prm->nnedi.doubleHeight;
    const bool autoField = prm->nnedi.field == VPP_NNEDI_FIELD_AUTO || prm->nnedi.field == VPP_NNEDI_FIELD_BOB;
    const int outputFrames = topology.frameMultiplier;
    if (!doubleHeight && autoField && (pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
        for (int i = 0; i < outputFrames; i++) {
            auto pOut = &m_frameBuf[i]->frame;
            err = copyFrameAsync(pOut, pInputFrame, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy progressive NNEDI passthrough frame %d: %s.\n"), i, get_err_mes(err));
                return err;
            }
            copyFramePropWithoutRes(pOut, pInputFrame);
            pOut->picstruct = RGY_PICSTRUCT_FRAME;
            ppOutputFrames[i] = pOut;
        }
        *pOutputFrameNum = outputFrames;
        if (topology.doubleRate) {
            setDoubleRateTimestamp(pInputFrame, ppOutputFrames);
        }
        return RGY_ERR_NONE;
    }
    for (int i = 0; i < outputFrames; i++) {
        if (!doubleHeight) {
            err = copyFrameAsync(&m_frameBuf[i]->frame, pInputFrame, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to initialize NNEDI output frame %d: %s.\n"), i, get_err_mes(err));
                return err;
            }
        }
        RGYNnediFrameMap frameMap;
        err = rgy_nnedi_map_output_frame(&frameMap, topology, i);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = prepareFieldReference(pInputFrame, i, frameMap, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = classifyPixelsAndSeedOutput(pInputFrame, i, frameMap, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = resolveClassifiedPixels(pInputFrame, i, frameMap, stream);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        auto pOut = &m_frameBuf[i]->frame;
        copyFramePropWithoutRes(pOut, pInputFrame);
        pOut->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[i] = pOut;
    }
    *pOutputFrameNum = outputFrames;
    if (topology.doubleRate) {
        setDoubleRateTimestamp(pInputFrame, ppOutputFrames);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterNnedi::close() {
    m_weights.reset();
    m_transformedWeights = RGYFilterNnediTransformedWeights();
    m_refBuf.clear();
    m_prescreenerWeightBuf.reset();
    m_predictorWeightBuf.reset();
    m_workNNBuf.clear();
    m_numBlocksBuf.clear();
    m_tileGroupsX = NNEDI_TILE_GROUPS_X;
    m_tileRows = NNEDI_TILE_ROWS;
    m_predLocalX = 16;
    m_predLocalY = 32;
    m_frameBuf.clear();
}
