// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without __restrict__ion, including without limitation the rights
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

#include <map>
#include <array>
#include "convert_csp.h"
#include "NVEncFilterDenoiseNLMeans.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "vector_types.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define ENABLE_CUDA_FP16_DEVICE (__CUDACC_VER_MAJOR__ >= 10 && __CUDA_ARCH__ >= 530)
#define ENABLE_CUDA_FP16_HOST   (__CUDACC_VER_MAJOR__ >= 10)

static const int NLEANS_BLOCK_X = 32;
static const int NLEANS_BLOCK_Y = 16;

// atomic_addを試したが、___syncthreadsしたほうが速い
#define ATOMIC_OPT 0

static const int maxPatchRadius = 10;
static const int maxSearchRadius = 10;

static bool shared_opt_avail(const int search_radius) {
    return search_radius * 2 <= NLEANS_BLOCK_X && search_radius <= NLEANS_BLOCK_Y;
}

static std::vector<std::pair<int, int>> nxnylist(const int search_radius) {
    std::vector<std::pair<int, int>> nxny;
    for (int ny = -search_radius; ny <= 0; ny++) {
        for (int nx = -search_radius; nx <= search_radius; nx++) {
            if (ny * (2 * search_radius - 1) + nx < 0) {
                nxny.push_back(std::make_pair(nx, ny));
            }
        }
    }
    return nxny;
}

static std::vector<std::pair<int, int>> nxnylist_full(const int search_radius) {
    std::vector<std::pair<int, int>> nxny;
    nxny.reserve((2 * search_radius + 1) * (2 * search_radius + 1));
    for (int ny = -search_radius; ny <= search_radius; ny++) {
        for (int nx = -search_radius; nx <= search_radius; nx++) {
            nxny.push_back(std::make_pair(nx, ny));
        }
    }
    return nxny;
}

enum RGYFilterDenoiseNLMeansTmpBufIdx {
    TMP_U,
    TMP_V,
    TMP_IW0,
    TMP_IW1,
    TMP_IW2,
    TMP_IW3,
    TMP_IW4,
    TMP_IW5,
    TMP_IW6,
    TMP_IW7,
    TMP_IW8,
    TMP_LAST = TMP_IW8,
    TMP_TOTAL,
};

#if ENABLE_VPP_NLMEANS

template<typename Type, int bit_depth>
__device__ __inline__ float8 calc_sqdiff(Type val0, float8 val1) {
    float8 val0_1 = (float8)val0 - val1;
    const float8 fdiff = val0_1 * (float)(1.0f / ((1 << bit_depth) - 1));
    return fdiff * fdiff;
}

template<typename TmpVType8>
__device__ __inline__ TmpVType8 toTmpVType8(float8 v) {
    return v;
}

template<>
__device__ __inline__ half8 toTmpVType8<half8>(float8 v) {
#if ENABLE_CUDA_FP16_DEVICE
    half8 a;
    a.h2.s0 = __floats2half2_rn(v.s0, v.s1);
    a.h2.s1 = __floats2half2_rn(v.s2, v.s3);
    a.h2.s2 = __floats2half2_rn(v.s4, v.s5);
    a.h2.s3 = __floats2half2_rn(v.s6, v.s7);
    return a;
#else
    return half8(0.0f);
#endif
}

template<typename Type>
__device__ __inline__ Type get_xyoffset_pix(
    const char *__restrict__ pSrc, const int srcPitch,
    const int ix, const int iy, const int xoffset, const int yoffset, const int width, const int height) {
    const int jx = clamp(ix + xoffset, 0, width - 1);
    const int jy = clamp(iy + yoffset, 0, height - 1);
    const char *ptr1 = pSrc + jy * srcPitch + jx * sizeof(Type);
    return *(const Type *)ptr1;
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, int offset_count>
__global__ void kernel_calc_diff_square(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height, int8 xoffset, int8 yoffset
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const char *ptr0 = pSrc + iy * srcPitch + ix * sizeof(Type);
        const Type val0 = *(const Type *)ptr0;

        float8 val1 = float8(
                                  (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s0, yoffset.s0, width, height),
            (offset_count >= 2) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s1, yoffset.s1, width, height) : 0.0f,
            (offset_count >= 3) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s2, yoffset.s2, width, height) : 0.0f,
            (offset_count >= 4) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s3, yoffset.s3, width, height) : 0.0f,
            (offset_count >= 5) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s4, yoffset.s4, width, height) : 0.0f,
            (offset_count >= 6) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s5, yoffset.s5, width, height) : 0.0f,
            (offset_count >= 7) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s6, yoffset.s6, width, height) : 0.0f,
            (offset_count >= 8) ? (float)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s7, yoffset.s7, width, height) : 0.0f);

        TmpVType8 *ptrDst = (TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        ptrDst[0] = toTmpVType8<TmpVType8>(calc_sqdiff<Type, bit_depth>(val0, val1));
    }
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, int offset_count>
RGY_ERR nlmeansCalcDiffSquareOffsetCount(
    RGYFrameInfo *pTmpUPlane,
    const RGYFrameInfo *pInputPlane,
    const int8 xoffset, const int8 yoffset,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pInputPlane->width, NLEANS_BLOCK_X), divCeil(pInputPlane->height, NLEANS_BLOCK_Y));
    kernel_calc_diff_square<Type, bit_depth, TmpVType, TmpVType8, offset_count><<<gridSize, blockSize, 0, stream >>>(
        (char *)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
        (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, xoffset, yoffset);
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8>
RGY_ERR nlmeansCalcDiffSquare(
    RGYFrameInfo *pTmpUPlane,
    const RGYFrameInfo *pInputPlane,
    const int8 xoffset, const int8 yoffset,
    const int offset_count,
    cudaStream_t stream
) {
    switch (offset_count) {
    case 1:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 1>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 2:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 2>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 3:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 3>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 4:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 4>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 5:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 5>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 6:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 6>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 7:  return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 7>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    case 8:
    default: return nlmeansCalcDiffSquareOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 8>(pTmpUPlane, pInputPlane, xoffset, yoffset, stream);
    }
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, int offset_count>
__global__ void kernel_calc_diff_square_temporal(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const char *__restrict__ pRef, const int refPitch,
    const int width, const int height, int8 xoffset, int8 yoffset
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const char *ptr0 = pSrc + iy * srcPitch + ix * sizeof(Type);
        const Type val0 = *(const Type *)ptr0;

        float8 val1 = float8(
                                  (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s0, yoffset.s0, width, height),
            (offset_count >= 2) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s1, yoffset.s1, width, height) : 0.0f,
            (offset_count >= 3) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s2, yoffset.s2, width, height) : 0.0f,
            (offset_count >= 4) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s3, yoffset.s3, width, height) : 0.0f,
            (offset_count >= 5) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s4, yoffset.s4, width, height) : 0.0f,
            (offset_count >= 6) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s5, yoffset.s5, width, height) : 0.0f,
            (offset_count >= 7) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s6, yoffset.s6, width, height) : 0.0f,
            (offset_count >= 8) ? (float)get_xyoffset_pix<Type>(pRef, refPitch, ix, iy, xoffset.s7, yoffset.s7, width, height) : 0.0f);

        TmpVType8 *ptrDst = (TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        ptrDst[0] = toTmpVType8<TmpVType8>(calc_sqdiff<Type, bit_depth>(val0, val1));
    }
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, int offset_count>
RGY_ERR nlmeansCalcDiffSquareTemporalOffsetCount(
    RGYFrameInfo *pTmpUPlane,
    const RGYFrameInfo *pInputPlane,
    const RGYFrameInfo *pRefPlane,
    const int8 xoffset, const int8 yoffset,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pInputPlane->width, NLEANS_BLOCK_X), divCeil(pInputPlane->height, NLEANS_BLOCK_Y));
    kernel_calc_diff_square_temporal<Type, bit_depth, TmpVType, TmpVType8, offset_count><<<gridSize, blockSize, 0, stream>>>(
        (char *)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
        (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        (const char *)pRefPlane->ptr[0], pRefPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, xoffset, yoffset);
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8>
RGY_ERR nlmeansCalcDiffSquareTemporal(
    RGYFrameInfo *pTmpUPlane,
    const RGYFrameInfo *pInputPlane,
    const RGYFrameInfo *pRefPlane,
    const int8 xoffset, const int8 yoffset,
    const int offset_count,
    cudaStream_t stream
) {
    switch (offset_count) {
    case 1:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 1>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 2:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 2>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 3:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 3>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 4:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 4>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 5:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 5>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 6:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 6>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 7:  return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 7>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    case 8:
    default: return nlmeansCalcDiffSquareTemporalOffsetCount<Type, bit_depth, TmpVType, TmpVType8, 8>(pTmpUPlane, pInputPlane, pRefPlane, xoffset, yoffset, stream);
    }
}

template<typename Type, int template_radius, typename TmpVType8>
__global__ void kernel_denoise_nlmeans_calc_v(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        TmpVType8 sum = TmpVType8(0.0f);
        for (int j = - template_radius; j <= template_radius; j++) {
            const int srcy = clamp(iy + j, 0, height - 1);
            for (int i = - template_radius; i <= template_radius; i++) {
                const int srcx = clamp(ix + i, 0, width - 1);
                const TmpVType8 vals = TmpVType8::load((TmpVType8 *)(pSrc + srcy * srcPitch + srcx * sizeof(TmpVType8)));
                sum += vals;
            }
        }
        TmpVType8 *ptr = (TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        ptr[0] = sum;
    }
}

template<typename Type, int template_radius, typename TmpVType8>
RGY_ERR nlmeansCalcV(
    RGYFrameInfo *pTmpVPlane,
    const RGYFrameInfo *pTmpUPlane,
    const int width, const int height,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(width, NLEANS_BLOCK_X), divCeil(height, NLEANS_BLOCK_Y));
    kernel_denoise_nlmeans_calc_v<Type, template_radius, TmpVType8> <<<gridSize, blockSize, 0, stream>>> (
        (char *)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
        (const char *)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
        width, height);
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

template<typename TmpWPType, typename TmpWPType2>
__device__ __inline__ void add_reverse_side_offset(char *__restrict__ pImgW, const int tmpPitch, const int width, const int height, const int ix, const int iy, const int xoffset, const int yoffset, const TmpWPType pixNormalized, const TmpWPType weight) {
    static_assert(sizeof(TmpWPType) * 2 == sizeof(TmpWPType2), "sizeof(TmpWPType) * 2 == sizeof(TmpWPType2)");
    if ((xoffset | yoffset) == 0) return;
    const int jx = ix + xoffset;
    const int jy = iy + yoffset;
    if (0 <= jx && jx < width && 0 <= jy && jy < height) {
        TmpWPType2 *ptrImgW = (TmpWPType2 *)(pImgW + jy * tmpPitch + jx * sizeof(TmpWPType2));
        TmpWPType2 weight_pix_2 = { weight * pixNormalized, weight };
        ptrImgW[0] += weight_pix_2;
    }
}

template<typename Type, int bit_depth, typename TmpWPType>
__device__ __inline__ TmpWPType getSrcPixXYOffset(const char *__restrict__ pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int xoffset, const int yoffset) {
    const Type pix = *(const Type *)(pSrc + clamp(iy+yoffset, 0, height-1) * srcPitch + clamp(ix+xoffset,0,width-1) * sizeof(Type));
    return (TmpWPType)(pix * (float)(1.0f / ((1<<bit_depth) - 1)));
}

template<typename Type, int bit_depth, typename TmpWPType, typename TmpWPType8, int offset_count>
__device__ TmpWPType8 getSrcPixXYOffset8(const char *__restrict__ pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int8 xoffset, const int8 yoffset) {
    static_assert(sizeof(TmpWPType) * 8 == sizeof(TmpWPType8), "sizeof(TmpWPType) * 8 == sizeof(TmpWPType8)");
    TmpWPType8 pix8 = TmpWPType8(
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s0, yoffset.s0),
        (offset_count >= 2) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s1, yoffset.s1) : (TmpWPType)0.0f,
        (offset_count >= 3) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s2, yoffset.s2) : (TmpWPType)0.0f,
        (offset_count >= 4) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s3, yoffset.s3) : (TmpWPType)0.0f,
        (offset_count >= 5) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s4, yoffset.s4) : (TmpWPType)0.0f,
        (offset_count >= 6) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s5, yoffset.s5) : (TmpWPType)0.0f,
        (offset_count >= 7) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s6, yoffset.s6) : (TmpWPType)0.0f,
        (offset_count >= 8) ? getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s7, yoffset.s7) : (TmpWPType)0.0f);
    return pix8;
}

template<typename TmpWPType, typename TmpVType8>
__device__ __inline__ TmpWPType toTmpWPType8(TmpVType8 v) {
    return v;
}

template<>
__device__ __inline__ float8 toTmpWPType8<float8, half8>(half8 v) {
#if ENABLE_CUDA_FP16_DEVICE
    float2 f20 = __half22float2(v.h2.s0);
    float2 f21 = __half22float2(v.h2.s1);
    float2 f22 = __half22float2(v.h2.s2);
    float2 f23 = __half22float2(v.h2.s3);
    return float8(
        f20.x,
        f20.y,
        f21.x,
        f21.y,
        f22.x,
        f22.y,
        f23.x,
        f23.y);
#else
    return float8(0.0f);
#endif
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int offset_count>
__global__ void kernel_denoise_nlmeans_calc_weight(
    char *__restrict__ pImgW0,
    char *__restrict__ pImgW1, char *__restrict__ pImgW2, char *__restrict__ pImgW3, char *__restrict__ pImgW4,
    char *__restrict__ pImgW5, char *__restrict__ pImgW6, char *__restrict__ pImgW7, char *__restrict__ pImgW8,
    const int tmpPitch,
    const char *__restrict__ pV, const int vPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset
) {
    static_assert(sizeof(TmpWPType) * 2 == sizeof(TmpWPType2), "sizeof(TmpWPType) * 2 == sizeof(TmpWPType2)");
    static_assert(sizeof(TmpWPType) * 8 == sizeof(TmpWPType8), "sizeof(TmpWPType) * 8 == sizeof(TmpWPType8)");
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const TmpVType8 v_vt8 = TmpVType8::load((const TmpVType8 *)(pV + iy * vPitch + ix * sizeof(TmpVType8)));
        const TmpWPType8 v_tmpv8 = toTmpWPType8<TmpWPType8, TmpVType8>(v_vt8); // expを使う前にfp32に変換
        const TmpWPType8 weight = __expf(max(v_tmpv8 - (TmpWPType)(2.0f * sigma), (TmpWPType8)0.0f) * (TmpWPType)-inv_param_h_h);

        // 自分のほうはここですべて同じバッファ(ptrImgW0)に足し込んでしまう
        {
            TmpWPType2 *ptrImgW0 = (TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
            TmpWPType8 pix8 = getSrcPixXYOffset8<Type, bit_depth, TmpWPType, TmpWPType8, offset_count>(pSrc, srcPitch, width, height, ix, iy, xoffset, yoffset);
            TmpWPType8 weight_pix8 = weight * pix8;
            TmpWPType2 weight_pix_2 = { vec_sum(weight_pix8), vec_sum(weight) };
            ptrImgW0[0] += weight_pix_2;
        }
        // 反対側は衝突を避けるため、別々に足し込む
        const Type pix = *(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        const TmpWPType pixNormalized = (TmpWPType)(pix * (float)(1.0f / ((1<<bit_depth) - 1)));
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW1, tmpPitch, width, height, ix, iy, xoffset.s0, yoffset.s0, pixNormalized, weight.f0());
        if (offset_count >= 2) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW2, tmpPitch, width, height, ix, iy, xoffset.s1, yoffset.s1, pixNormalized, weight.f1());
        if (offset_count >= 3) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW3, tmpPitch, width, height, ix, iy, xoffset.s2, yoffset.s2, pixNormalized, weight.f2());
        if (offset_count >= 4) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW4, tmpPitch, width, height, ix, iy, xoffset.s3, yoffset.s3, pixNormalized, weight.f3());
        if (offset_count >= 5) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW5, tmpPitch, width, height, ix, iy, xoffset.s4, yoffset.s4, pixNormalized, weight.f4());
        if (offset_count >= 6) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW6, tmpPitch, width, height, ix, iy, xoffset.s5, yoffset.s5, pixNormalized, weight.f5());
        if (offset_count >= 7) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW7, tmpPitch, width, height, ix, iy, xoffset.s6, yoffset.s6, pixNormalized, weight.f6());
        if (offset_count >= 8) add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW8, tmpPitch, width, height, ix, iy, xoffset.s7, yoffset.s7, pixNormalized, weight.f7());
    }
}

template<typename TmpWPType, typename TmpWPType2, int search_radius>
__device__ __inline__ void add_tmpwp_local(TmpWPType2 tmpWP[search_radius + NLEANS_BLOCK_Y][search_radius * 2 + NLEANS_BLOCK_X], const TmpWPType2 tmpWeightPix, const int thx, const int thy, const int xoffset, const int yoffset) {
#if ATOMIC_OPT
#if __CUDA_ARCH__ >= 900
    atomicAdd(&tmpWP[thy + yoffset + search_radius][thx + xoffset + search_radius], tmpWeightPix);
#else
    atomicAdd(&tmpWP[thy + yoffset + search_radius][thx + xoffset + search_radius].x, tmpWeightPix.x);
    atomicAdd(&tmpWP[thy + yoffset + search_radius][thx + xoffset + search_radius].y, tmpWeightPix.y);
#endif
#else
    tmpWP[thy + yoffset + search_radius][thx + xoffset + search_radius] += tmpWeightPix;
#endif
}

template<typename TmpWPType, typename TmpWPType2, int search_radius>
__device__ __inline__ void add_tmpwp_local(TmpWPType2 tmpWP[search_radius + NLEANS_BLOCK_Y][search_radius * 2 + NLEANS_BLOCK_X], const TmpWPType pixNormalized, const TmpWPType weight, const int thx, const int thy, const int xoffset, const int yoffset) {
    TmpWPType2 tmp = { weight * pixNormalized, weight };
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, tmp, thx, thy, xoffset, yoffset);
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int search_radius, int offset_count>
__global__ void kernel_denoise_nlmeans_calc_weight_shared_opt(
    char *__restrict__ pImgW0, char *__restrict__ pImgW1, char *__restrict__ pImgW2, char *__restrict__ pImgW3,
    const int tmpPitch,
    const char *__restrict__ pV, const int vPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset, const int yoffsetmin
) {
    static_assert(sizeof(TmpWPType) * 2 == sizeof(TmpWPType2), "sizeof(TmpWPType) * 2 == sizeof(TmpWPType2)");
    static_assert(sizeof(TmpWPType) * 8 == sizeof(TmpWPType8), "sizeof(TmpWPType) * 8 == sizeof(TmpWPType8)");
    const int bx = blockIdx.x * NLEANS_BLOCK_X;
    const int by = blockIdx.y * NLEANS_BLOCK_Y;
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int ix = bx + thx;
    const int iy = by + thy;

    char *__restrict__ pImgW;
    // 対象のポインタを決める
    // xoffset, yoffsetの分、最大x方向には+-search_radius, y方向には-search_radiusの分だけ広く書き込むため
    // メモリへの書き込みが衝突しないよう、ブロックごとに書き込み先のバッファを分ける
    if (blockIdx.y & 1) {
        pImgW = (blockIdx.x & 1) ? pImgW3 : pImgW2;
    } else {
        pImgW = (blockIdx.x & 1) ? pImgW1 : pImgW0;
    }

    // x方向には+-search_radius, y方向には-search_radiusの分だけ広く確保する
    __shared__ TmpWPType2 tmpWP[search_radius + NLEANS_BLOCK_Y][search_radius * 2 + NLEANS_BLOCK_X];
    /*
                          bx              bx+NLEANS_BLOCK_X
    global                 |                        |
    shared    |            |                        |                          |
              0        search_radius  search_radius+NLEANS_BLOCK_X   2*search_radius+NLEANS_BLOCK_X
    */
    // tmpWPにpImgWの一部コピー
    // y方向は、実際のyoffsetの最小値yoffsetminを考慮してロードして余分なロードをしないようにする
    for (int j = thy + search_radius + yoffsetmin; j < search_radius + NLEANS_BLOCK_Y; j += NLEANS_BLOCK_Y) {
        for (int i = thx; i < search_radius * 2 + NLEANS_BLOCK_X; i += NLEANS_BLOCK_X) {
            const int srcx = bx + i - search_radius;
            const int srcy = by + j - search_radius;
            if (0 <= srcx && srcx < width && 0 <= srcy && srcy < height) {
                const TmpWPType2 val = *(const TmpWPType2 *)(pImgW + srcy * tmpPitch + srcx * sizeof(TmpWPType2));
                tmpWP[j][i] = val;
            }
        }
    }

#if ATOMIC_OPT
#define SYNC_THREADS
#else
#define SYNC_THREADS __syncthreads()
#endif
    __syncthreads();

    TmpWPType8 weight = TmpWPType8(0.0f);
    if (ix < width && iy < height) {
        const TmpVType8 v_vt8 = TmpVType8::load((const TmpVType8 *)(pV + iy * vPitch + ix * sizeof(TmpVType8)));
        const TmpWPType8 v_tmpv8 = toTmpWPType8<TmpWPType8, TmpVType8>(v_vt8); // expを使う前にfp32に変換
        weight = __expf(max(v_tmpv8 - (TmpWPType)(2.0f * sigma), (TmpWPType8)0.0f) * (TmpWPType)-inv_param_h_h);

        // 自分のほうはここですべて同じバッファ(ptrImgW0)に足し込んでしまう
        {
            TmpWPType8 pix8 = getSrcPixXYOffset8<Type, bit_depth, TmpWPType, TmpWPType8, offset_count>(pSrc, srcPitch, width, height, ix, iy, xoffset, yoffset);
            TmpWPType8 weight_pix8 = weight * pix8;
            TmpWPType2 weight_pix_2 = { vec_sum(weight_pix8), vec_sum(weight) };
            add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, weight_pix_2, thx, thy, 0, 0);
        }
    }
    // 共有メモリ上ですべて足し込んでしまう
    // 計算が衝突しないよう、書き込みごとに同期する
    SYNC_THREADS;
    TmpWPType pixNormalized = 0.0f;
    if (ix < width && iy < height) {
        const Type pix = *(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        pixNormalized = (TmpWPType)(pix * (float)(1.0f / ((1 << bit_depth) - 1)));
    }
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f0(), thx, thy, xoffset.s0, yoffset.s0);
    if (offset_count >= 2) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f1(), thx, thy, xoffset.s1, yoffset.s1);
    }
    if (offset_count >= 3) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f2(), thx, thy, xoffset.s2, yoffset.s2);
    }
    if (offset_count >= 4) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f3(), thx, thy, xoffset.s3, yoffset.s3);
    }
    if (offset_count >= 5) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f4(), thx, thy, xoffset.s4, yoffset.s4);
    }
    if (offset_count >= 6) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f5(), thx, thy, xoffset.s5, yoffset.s5);
    }
    if (offset_count >= 7) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f6(), thx, thy, xoffset.s6, yoffset.s6);
    }
    if (offset_count >= 8) {
        SYNC_THREADS;
        add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.f7(), thx, thy, xoffset.s7, yoffset.s7);
    }
    __syncthreads();

    // tmpWPからpImgWにコピー
    // y方向は、実際のyoffsetの最小値yoffsetminを考慮してロードして余分な書き込みをしないようにする
    for (int j = thy + search_radius + yoffsetmin; j < search_radius + NLEANS_BLOCK_Y; j += NLEANS_BLOCK_Y) {
        for (int i = thx; i < search_radius * 2 + NLEANS_BLOCK_X; i += NLEANS_BLOCK_X) {
            const int srcx = bx + i - search_radius;
            const int srcy = by + j - search_radius;
            if (0 <= srcx && srcx < width && 0 <= srcy && srcy < height) {
                TmpWPType2 *ptr = (TmpWPType2 *)(pImgW + srcy * tmpPitch + srcx * sizeof(TmpWPType2));
                ptr[0] = tmpWP[j][i];
            }
        }
    }
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int search_radius, int offset_count>
RGY_ERR nlmeansCalcWeightOffsetCount(
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pTmpVPlane,
    const RGYFrameInfo *pInputPlane,
    const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset, const int yoffsetmin, const bool shared_opt,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pInputPlane->width, NLEANS_BLOCK_X), divCeil(pInputPlane->height, NLEANS_BLOCK_Y));
    if (shared_opt && shared_opt_avail(search_radius)) {
        kernel_denoise_nlmeans_calc_weight_shared_opt<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, offset_count> <<<gridSize, blockSize, 0, stream >>> (
            (char *)pTmpIWPlane[0].ptr[0], (char *)pTmpIWPlane[1].ptr[0], (char *)pTmpIWPlane[2].ptr[0], (char *)pTmpIWPlane[3].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (const char *)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
            (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pInputPlane->width, pInputPlane->height,
            sigma, inv_param_h_h,
            xoffset, yoffset, yoffsetmin);
    } else {
        kernel_denoise_nlmeans_calc_weight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, offset_count> << <gridSize, blockSize, 0, stream >> > (
            (char *)pTmpIWPlane[0].ptr[0],
            (char *)pTmpIWPlane[1].ptr[0], (char *)pTmpIWPlane[2].ptr[0], (char *)pTmpIWPlane[3].ptr[0], (char *)pTmpIWPlane[4].ptr[0],
            (char *)pTmpIWPlane[5].ptr[0], (char *)pTmpIWPlane[6].ptr[0], (char *)pTmpIWPlane[7].ptr[0], (char *)pTmpIWPlane[8].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (const char *)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
            (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pInputPlane->width, pInputPlane->height,
            sigma, inv_param_h_h,
            xoffset, yoffset);
    }
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int search_radius>
RGY_ERR nlmeansCalcWeight(
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pTmpVPlane,
    const RGYFrameInfo *pInputPlane,
    const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset, const int yoffsetmin, const int offset_count, const bool shared_opt,
    cudaStream_t stream
) {
    // 今のところ、offset_countは3か4か8しかない
    switch (offset_count) {
    //case 1:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 1>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    //case 2:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 2>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    case 3:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 3>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    case 4:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 4>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    //case 5:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 5>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    //case 6:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 6>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    //case 7:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 7>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    case 8:  return nlmeansCalcWeightOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius, 8>(pTmpIWPlane, pTmpVPlane, pInputPlane, sigma, inv_param_h_h, xoffset, yoffset, yoffsetmin, shared_opt, stream);
    default:
        return RGY_ERR_OUT_OF_RANGE;
    }
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int offset_count>
__global__ void kernel_denoise_nlmeans_calc_weight_temporal(
    char *__restrict__ pImgW0,
    const int tmpPitch,
    const char *__restrict__ pV, const int vPitch,
    const char *__restrict__ pRef, const int refPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset
) {
    static_assert(sizeof(TmpWPType) * 2 == sizeof(TmpWPType2), "sizeof(TmpWPType) * 2 == sizeof(TmpWPType2)");
    static_assert(sizeof(TmpWPType) * 8 == sizeof(TmpWPType8), "sizeof(TmpWPType) * 8 == sizeof(TmpWPType8)");
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const TmpVType8 v_vt8 = TmpVType8::load((const TmpVType8 *)(pV + iy * vPitch + ix * sizeof(TmpVType8)));
        const TmpWPType8 v_tmpv8 = toTmpWPType8<TmpWPType8, TmpVType8>(v_vt8);
        const TmpWPType8 weight = __expf(max(v_tmpv8 - (TmpWPType)(2.0f * sigma), (TmpWPType8)0.0f) * (TmpWPType)-inv_param_h_h);

        TmpWPType8 pix8 = getSrcPixXYOffset8<Type, bit_depth, TmpWPType, TmpWPType8, offset_count>(pRef, refPitch, width, height, ix, iy, xoffset, yoffset);
        TmpWPType8 weight_pix8 = weight * pix8;
        TmpWPType2 weight_pix_2 = { vec_sum(weight_pix8), vec_sum(weight) };
        TmpWPType2 *ptrImgW0 = (TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        ptrImgW0[0] += weight_pix_2;
    }
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int offset_count>
RGY_ERR nlmeansCalcWeightTemporalOffsetCount(
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pTmpVPlane,
    const RGYFrameInfo *pRefPlane,
    const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pRefPlane->width, NLEANS_BLOCK_X), divCeil(pRefPlane->height, NLEANS_BLOCK_Y));
    kernel_denoise_nlmeans_calc_weight_temporal<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, offset_count><<<gridSize, blockSize, 0, stream>>>(
        (char *)pTmpIWPlane[0].ptr[0],
        pTmpIWPlane[0].pitch[0],
        (const char *)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
        (const char *)pRefPlane->ptr[0], pRefPlane->pitch[0],
        pRefPlane->width, pRefPlane->height,
        sigma, inv_param_h_h,
        xoffset, yoffset);
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8>
RGY_ERR nlmeansCalcWeightTemporal(
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pTmpVPlane,
    const RGYFrameInfo *pRefPlane,
    const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset, const int offset_count,
    cudaStream_t stream
) {
    switch (offset_count) {
    case 1:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 1>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 2:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 2>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 3:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 3>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 4:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 4>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 5:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 5>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 6:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 6>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 7:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 7>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    case 8:  return nlmeansCalcWeightTemporalOffsetCount<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 8>(pTmpIWPlane, pTmpVPlane, pRefPlane, sigma, inv_param_h_h, xoffset, yoffset, stream);
    default:
        return RGY_ERR_OUT_OF_RANGE;
    }
}

template<typename Type, int bit_depth, typename TmpWPType2>
__global__ void kernel_denoise_nlmeans_normalize(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pImgW0,
    const char *__restrict__ pImgW1, const char *__restrict__ pImgW2, const char *__restrict__ pImgW3, const char *__restrict__ pImgW4,
    const char *__restrict__ pImgW5, const char *__restrict__ pImgW6, const char *__restrict__ pImgW7, const char *__restrict__ pImgW8,
    const int tmpPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const Type      srcPix = *(const Type *      )(pSrc   + iy * srcPitch + ix * sizeof(Type));
        const TmpWPType2 imgW0 = *(const TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW1 = *(const TmpWPType2 *)(pImgW1 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW2 = *(const TmpWPType2 *)(pImgW2 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW3 = *(const TmpWPType2 *)(pImgW3 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW4 = *(const TmpWPType2 *)(pImgW4 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW5 = *(const TmpWPType2 *)(pImgW5 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW6 = *(const TmpWPType2 *)(pImgW6 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW7 = *(const TmpWPType2 *)(pImgW7 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW8 = *(const TmpWPType2 *)(pImgW8 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = (float)imgW0.x + (float)imgW1.x + (float)imgW2.x + (float)imgW3.x + (float)imgW4.x + (float)imgW5.x + (float)imgW6.x + (float)imgW7.x + (float)imgW8.x;
        const float weight = (float)imgW0.y + (float)imgW1.y + (float)imgW2.y + (float)imgW3.y + (float)imgW4.y + (float)imgW5.y + (float)imgW6.y + (float)imgW7.y + (float)imgW8.y;
        const float srcPixF = srcPix * (float)(1.0f / ((1 << bit_depth) - 1));
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp((imgW + srcPixF) * __frcp_rn(weight + 1.0f) * ((1 << bit_depth) - 1), 0.0f, (1 << bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth, typename TmpWPType2>
__global__ void kernel_denoise_nlmeans_normalize_shared_opt(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pImgW0, const char *__restrict__ pImgW1, const char *__restrict__ pImgW2, const char *__restrict__ pImgW3,
    const int tmpPitch,
    const char *__restrict__ pSrc, const int srcPitch,
    const int width, const int height
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const Type      srcPix = *(const Type *      )(pSrc   + iy * srcPitch + ix * sizeof(Type));
        const TmpWPType2 imgW0 = *(const TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW1 = *(const TmpWPType2 *)(pImgW1 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW2 = *(const TmpWPType2 *)(pImgW2 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW3 = *(const TmpWPType2 *)(pImgW3 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = (float)imgW0.x + (float)imgW1.x + (float)imgW2.x + (float)imgW3.x;
        const float weight = (float)imgW0.y + (float)imgW1.y + (float)imgW2.y + (float)imgW3.y;
        const float srcPixF = srcPix * (float)(1.0f / ((1 << bit_depth) - 1));
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp((imgW + srcPixF) * __frcp_rn(weight + 1.0f) * ((1 << bit_depth) - 1), 0.0f, (1 << bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth, typename TmpWPType2>
RGY_ERR nlmeansNormalize(
    RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pInputPlane,
    const bool shared_opt,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, NLEANS_BLOCK_X), divCeil(pOutputPlane->height, NLEANS_BLOCK_Y));
    if (shared_opt) {
        kernel_denoise_nlmeans_normalize_shared_opt<Type, bit_depth, TmpWPType2> << <gridSize, blockSize, 0, stream >> > (
            (char *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (const char *)pTmpIWPlane[0].ptr[0], (const char *)pTmpIWPlane[1].ptr[0], (const char *)pTmpIWPlane[2].ptr[0], (const char *)pTmpIWPlane[3].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pOutputPlane->width, pOutputPlane->height);
    } else {
        kernel_denoise_nlmeans_normalize<Type, bit_depth, TmpWPType2> << <gridSize, blockSize, 0, stream >> > (
            (char *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (const char *)pTmpIWPlane[0].ptr[0],
            (const char *)pTmpIWPlane[1].ptr[0], (const char *)pTmpIWPlane[2].ptr[0], (const char *)pTmpIWPlane[3].ptr[0], (const char *)pTmpIWPlane[4].ptr[0],
            (const char *)pTmpIWPlane[5].ptr[0], (const char *)pTmpIWPlane[6].ptr[0], (const char *)pTmpIWPlane[7].ptr[0], (const char *)pTmpIWPlane[8].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pOutputPlane->width, pOutputPlane->height);
    }
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}


class NLMeansFuncsBase {
public:
    NLMeansFuncsBase() {};
    virtual ~NLMeansFuncsBase() {};

    virtual decltype(&nlmeansCalcDiffSquare<uint8_t, 8, float, float8>) calcDiffSquare() = 0;
    virtual decltype(&nlmeansCalcDiffSquareTemporal<uint8_t, 8, float, float8>) calcDiffSquareTemporal() = 0;
    virtual decltype(&nlmeansCalcV<uint8_t, 1, float8>) calcV(int template_radius) = 0;
    virtual decltype(&nlmeansCalcWeight<uint8_t, 8, float8, float, float2, float8, 1>) calcWeight(int search_radius) = 0;
    virtual decltype(&nlmeansCalcWeightTemporal<uint8_t, 8, float8, float, float2, float8>) calcWeightTemporal() = 0;
    virtual decltype(&nlmeansNormalize<uint8_t, 8, float2>) normalize() = 0;
};

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8>
class NLMeansFuncs : public NLMeansFuncsBase {
public:
    NLMeansFuncs() {};
    virtual ~NLMeansFuncs() {};

    virtual decltype(&nlmeansCalcDiffSquare<Type, bit_depth, TmpVType, TmpVType8>) calcDiffSquare() override { return nlmeansCalcDiffSquare<Type, bit_depth, TmpVType, TmpVType8>; }
    virtual decltype(&nlmeansCalcDiffSquareTemporal<Type, bit_depth, TmpVType, TmpVType8>) calcDiffSquareTemporal() override { return nlmeansCalcDiffSquareTemporal<Type, bit_depth, TmpVType, TmpVType8>; }
    virtual decltype(&nlmeansCalcV<Type, 1, TmpVType8>) calcV(int template_radius) override {
        switch (template_radius) {
        case 1:  return nlmeansCalcV<Type,  1, TmpVType8>;
        case 2:  return nlmeansCalcV<Type,  2, TmpVType8>;
        case 3:  return nlmeansCalcV<Type,  3, TmpVType8>;
        case 4:  return nlmeansCalcV<Type,  4, TmpVType8>;
        case 5:  return nlmeansCalcV<Type,  5, TmpVType8>;
        case 6:  return nlmeansCalcV<Type,  6, TmpVType8>;
        case 7:  return nlmeansCalcV<Type,  7, TmpVType8>;
        case 8:  return nlmeansCalcV<Type,  8, TmpVType8>;
        case 9:  return nlmeansCalcV<Type,  9, TmpVType8>;
        case 10: return nlmeansCalcV<Type, 10, TmpVType8>;
        default: return nullptr;
        }
        
    }
    virtual decltype(&nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 1>) calcWeight(int search_radius) override {
        switch (search_radius) {
        case 1:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  1>;
        case 2:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  2>;
        case 3:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  3>;
        case 4:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  4>;
        case 5:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  5>;
        case 6:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  6>;
        case 7:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  7>;
        case 8:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  8>;
        case 9:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  9>;
        case 10: return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 10>;
        default: return nullptr;
        }
    }
    virtual decltype(&nlmeansCalcWeightTemporal<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8>) calcWeightTemporal() override {
        return nlmeansCalcWeightTemporal<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8>;
    }
    virtual decltype(&nlmeansNormalize<Type, bit_depth, TmpWPType2>) normalize() override { return nlmeansNormalize<Type, bit_depth, TmpWPType2>; }
};

std::unique_ptr<NLMeansFuncsBase> getNLMeansFunc(const RGY_CSP csp, const VppNLMeansFP16Opt fp16) {
    switch (RGY_CSP_DATA_TYPE[csp]) {
    case RGY_DATA_TYPE_U8:
        if (fp16 == VppNLMeansFP16Opt::All)       return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint8_t, 8, __half, half8, __half, __half2, half8>());
        if (fp16 == VppNLMeansFP16Opt::BlockDiff) return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint8_t, 8, __half, half8, float, float2, float8>());
                                                  return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint8_t, 8, float, float8, float, float2, float8>());
    case RGY_DATA_TYPE_U16:
        if (fp16 == VppNLMeansFP16Opt::All)       return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint16_t, 16, __half, half8, __half, __half2, half8>());
        if (fp16 == VppNLMeansFP16Opt::BlockDiff) return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint16_t, 16, __half, half8, float, float2, float8>());
                                                  return std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint16_t, 16, float, float8, float, float2, float8>());
    default:
        return nullptr;
    }
}

// https://lcondat.github.io/publis/condat_resreport_NLmeansv3.pdf
RGY_ERR NVEncFilterDenoiseNLMeans::denoisePlane(
    RGYFrameInfo *pOutputPlane,
    RGYFrameInfo *pTmpUPlane, RGYFrameInfo *pTmpVPlane,
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pInputPlane,
    const std::vector<const RGYFrameInfo *> &refPlanes,
    cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto func = getNLMeansFunc(pOutputPlane->csp, prm->nlmeans.fp16);
    if (!func) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    // 一時バッファを初期化
    auto err = RGY_ERR_NONE;
    for (int i = 0; i < RGY_NLMEANS_DXDY_STEP+1; i++) {
        if (pTmpIWPlane[i].ptr[0]) {
            err = setPlaneAsync(&pTmpIWPlane[i], 0, stream);
            CUDA_DEBUG_SYNC_ERR;
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error setPlane[IW%d](%s): %s.\n"), i, RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }
    const int template_radius = prm->nlmeans.patchSize / 2;

    // 計算すべきnx-nyの組み合わせを列挙
    const int search_radius = prm->nlmeans.searchSize / 2;
    const auto nxny = nxnylist(search_radius);
    // nx-nyの組み合わせをRGY_NLMEANS_DXDY_STEP個ずつまとめて計算して高速化
    for (size_t inxny = 0; inxny < nxny.size(); inxny += RGY_NLMEANS_DXDY_STEP) {
        const int offset_count = std::min((int)(nxny.size() - inxny), RGY_NLMEANS_DXDY_STEP);
        int nx0arr[RGY_NLMEANS_DXDY_STEP], ny0arr[RGY_NLMEANS_DXDY_STEP];
        int nymin = 0;
        for (int i = 0; i < RGY_NLMEANS_DXDY_STEP; i++) {
            nx0arr[i] = (inxny + i < nxny.size()) ? nxny[inxny + i].first : 0;
            ny0arr[i] = (inxny + i < nxny.size()) ? nxny[inxny + i].second : 0;
            nymin = std::min(nymin, ny0arr[i]);
        }
        //kernel引数に渡すために、int8に押し込む
        int8 nx0, ny0;
        memcpy(&nx0, nx0arr, sizeof(nx0));
        memcpy(&ny0, ny0arr, sizeof(ny0));
        {
            err = func->calcDiffSquare()(pTmpUPlane, pInputPlane, nx0, ny0, offset_count, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcDiffSquare (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
        {
            auto funcV = func->calcV(template_radius);
            if (!funcV) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcDiffSquare (denoisePlane(%s)): unsupported patchSize %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], prm->nlmeans.patchSize);
                return RGY_ERR_UNSUPPORTED;
            }
            err = funcV(
                pTmpVPlane, pTmpUPlane,
                pOutputPlane->width, pOutputPlane->height, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcV (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
        {
            auto funcCalcWeight = func->calcWeight(search_radius);
            if (!funcCalcWeight) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcWeight (denoisePlane(%s)): unsupported search size %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], prm->nlmeans.searchSize);
                return RGY_ERR_UNSUPPORTED;
            }
            err = funcCalcWeight(
                pTmpIWPlane, pTmpVPlane, pInputPlane,
                prm->nlmeans.sigma, 1.0f / (prm->nlmeans.h * prm->nlmeans.h),
                nx0, ny0, nymin, offset_count, prm->nlmeans.sharedMem, stream);
            if (err == RGY_ERR_OUT_OF_RANGE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcWeight (denoisePlane(%s)): unexpected offset_count %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], offset_count);
                return err;
            } else if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcWeight (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }
    // Temporal pass: reference-frame patches are accumulated one-way into IW0.
    if (!refPlanes.empty()) {
        const int search_radius_t = prm->nlmeans.searchSizeT / 2;
        const auto nxny_full = nxnylist_full(search_radius_t);
        for (const auto pRefPlane : refPlanes) {
            if (!pRefPlane || !pRefPlane->ptr[0]) continue;
            for (size_t inxny = 0; inxny < nxny_full.size(); inxny += RGY_NLMEANS_DXDY_STEP) {
                const int offset_count = std::min((int)(nxny_full.size() - inxny), RGY_NLMEANS_DXDY_STEP);
                int nx0arr[RGY_NLMEANS_DXDY_STEP] = { 0 };
                int ny0arr[RGY_NLMEANS_DXDY_STEP] = { 0 };
                for (int i = 0; i < offset_count; i++) {
                    nx0arr[i] = nxny_full[inxny + i].first;
                    ny0arr[i] = nxny_full[inxny + i].second;
                }
                int8 nx0, ny0;
                memcpy(&nx0, nx0arr, sizeof(nx0));
                memcpy(&ny0, ny0arr, sizeof(ny0));
                err = func->calcDiffSquareTemporal()(pTmpUPlane, pInputPlane, pRefPlane, nx0, ny0, offset_count, stream);
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at calcDiffSquareTemporal (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                    return err;
                }
                auto funcV = func->calcV(template_radius);
                if (!funcV) {
                    AddMessage(RGY_LOG_ERROR, _T("error at calcVTemporal (denoisePlane(%s)): unsupported patchSize %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], prm->nlmeans.patchSize);
                    return RGY_ERR_UNSUPPORTED;
                }
                err = funcV(pTmpVPlane, pTmpUPlane, pOutputPlane->width, pOutputPlane->height, stream);
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at calcVTemporal (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                    return err;
                }
                err = func->calcWeightTemporal()(
                    pTmpIWPlane, pTmpVPlane, pRefPlane,
                    prm->nlmeans.sigma, 1.0f / (prm->nlmeans.h * prm->nlmeans.h),
                    nx0, ny0, offset_count, stream);
                if (err == RGY_ERR_OUT_OF_RANGE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at calcWeightTemporal (denoisePlane(%s)): unexpected offset_count %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], offset_count);
                    return err;
                } else if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at calcWeightTemporal (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                    return err;
                }
            }
        }
    }
    // 最後に規格化
    {
        err = func->normalize()(pOutputPlane, pTmpIWPlane, pInputPlane, prm->nlmeans.sharedMem, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at normalize (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseNLMeans::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const std::vector<const RGYFrameInfo *> &refFrames, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!prm->nlmeans.processChroma && RGY_CSP_PLANES[pOutputFrame->csp] > 1) {
        auto copyErr = copyFrameAsync(pOutputFrame, pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame before luma-only denoise(nlmeans) (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(copyErr));
            return copyErr;
        }
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeTmpU = getPlane(&m_tmpBuf[TMP_U]->frame, RGY_PLANE_Y);
        auto planeTmpV = getPlane(&m_tmpBuf[TMP_V]->frame, RGY_PLANE_Y);
        std::array<RGYFrameInfo, RGY_NLMEANS_DXDY_STEP + 1> pTmpIWPlane;
        for (size_t j = 0; j < pTmpIWPlane.size(); j++) {
            if (m_tmpBuf[TMP_IW0 + j]) {
                pTmpIWPlane[j] = getPlane(&m_tmpBuf[TMP_IW0 + j]->frame, RGY_PLANE_Y);
            } else {
                pTmpIWPlane[j] = RGYFrameInfo();
            }
        }
        std::vector<RGYFrameInfo> refPlaneInfo;
        std::vector<const RGYFrameInfo *> refPlanePtrs;
        refPlaneInfo.reserve(refFrames.size());
        refPlanePtrs.reserve(refFrames.size());
        for (const auto pRef : refFrames) {
            if (pRef && pRef->ptr[0]) {
                refPlaneInfo.push_back(getPlane(pRef, RGY_PLANE_Y));
                refPlanePtrs.push_back(&refPlaneInfo.back());
            }
        }
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, pTmpIWPlane.data(), &planeSrc, refPlanePtrs, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) luma plane: %s\n"), get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeTmpU = getPlane(&m_tmpBuf[TMP_U]->frame, (RGY_PLANE)i);
        auto planeTmpV = getPlane(&m_tmpBuf[TMP_V]->frame, (RGY_PLANE)i);
        std::array<RGYFrameInfo, RGY_NLMEANS_DXDY_STEP + 1> pTmpIWPlane;
        for (size_t j = 0; j < pTmpIWPlane.size(); j++) {
            if (m_tmpBuf[TMP_IW0 + j]) {
                pTmpIWPlane[j] = getPlane(&m_tmpBuf[TMP_IW0 + j]->frame, (RGY_PLANE)i);
            } else {
                pTmpIWPlane[j] = RGYFrameInfo();
            }
        }
        std::vector<RGYFrameInfo> refPlaneInfo;
        std::vector<const RGYFrameInfo *> refPlanePtrs;
        refPlaneInfo.reserve(refFrames.size());
        refPlanePtrs.reserve(refFrames.size());
        for (const auto pRef : refFrames) {
            if (pRef && pRef->ptr[0]) {
                refPlaneInfo.push_back(getPlane(pRef, (RGY_PLANE)i));
                refPlanePtrs.push_back(&refPlaneInfo.back());
            }
        }
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, pTmpIWPlane.data(), &planeSrc, refPlanePtrs, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) frame(%d) %s: %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseNLMeans::emitFrame(int idx_cur, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int d = prm->nlmeans.d;
    const int cacheSize = (int)m_cacheFrames.size();
    const RGYFrameInfo *pInputFrame = &m_cacheFrames[idx_cur]->frame;

    std::vector<const RGYFrameInfo *> refFrames;
    refFrames.reserve(2 * d);
    for (int k = 1; k <= d; k++) {
        const int abs_past = m_outputCount - k;
        if (abs_past >= 0) {
            refFrames.push_back(&m_cacheFrames[abs_past % cacheSize]->frame);
        }
        const int abs_future = m_outputCount + k;
        if (abs_future < m_inputCount) {
            refFrames.push_back(&m_cacheFrames[abs_future % cacheSize]->frame);
        }
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    copyFrameProp(ppOutputFrames[0], pInputFrame);

    auto sts = denoiseFrame(ppOutputFrames[0], pInputFrame, refFrames, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_outputCount++;
    return RGY_ERR_NONE;
}

#endif //ENABLE_VPP_NLMEANS

NVEncFilterDenoiseNLMeans::NVEncFilterDenoiseNLMeans() :
    m_tmpBuf(),
    m_cacheFrames(),
    m_inputCount(0),
    m_outputCount(0),
    m_drained(false) {
    m_name = _T("nlmeans");
}

NVEncFilterDenoiseNLMeans::~NVEncFilterDenoiseNLMeans() {
    close();
}

RGY_ERR NVEncFilterDenoiseNLMeans::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
#if ENABLE_VPP_NLMEANS
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.patchSize % 2 == 0) {
        prm->nlmeans.patchSize++; // 奇数にする
    }
    if (prm->nlmeans.patchSize <= 2) {
        AddMessage(RGY_LOG_ERROR, _T("patch must be 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.patchSize / 2 > maxPatchRadius) {
        AddMessage(RGY_LOG_ERROR, _T("patch size too big: %d.\n"), prm->nlmeans.patchSize);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->nlmeans.searchSize % 2 == 0) {
        prm->nlmeans.searchSize++; // 奇数にする
    }
    if (prm->nlmeans.searchSize <= 2) {
        AddMessage(RGY_LOG_ERROR, _T("support must be a 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.searchSize / 2 > maxSearchRadius) {
        AddMessage(RGY_LOG_ERROR, _T("search size too big: %d.\n"), prm->nlmeans.searchSize);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->nlmeans.sigma < 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("sigma should be 0 or larger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.h <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("h should be larger than 0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.d < 0 || prm->nlmeans.d > FILTER_NLMEANS_D_MAX) {
        prm->nlmeans.d = clamp(prm->nlmeans.d, 0, FILTER_NLMEANS_D_MAX);
        AddMessage(RGY_LOG_WARN, _T("d should be in range of 0 - %d.\n"), FILTER_NLMEANS_D_MAX);
    }
    if (prm->nlmeans.searchSizeT % 2 == 0) {
        prm->nlmeans.searchSizeT++;
    }
    if (prm->nlmeans.searchSizeT < 3) {
        AddMessage(RGY_LOG_ERROR, _T("search_t must be 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.searchSizeT / 2 > maxSearchRadius) {
        AddMessage(RGY_LOG_ERROR, _T("search_t size too big: %d.\n"), prm->nlmeans.searchSizeT);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->nlmeans.d > 0 && prm->nlmeans.sharedMem) {
        AddMessage(RGY_LOG_DEBUG, _T("disabling sharedMem optimisation because d > 0.\n"));
        prm->nlmeans.sharedMem = false;
    }
    if (prm->nlmeans.d > 0) {
        m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_DATA);
    } else {
        m_pathThrough = FILTER_PATHTHROUGH_ALL;
    }
    if (prm->nlmeans.fp16 != VppNLMeansFP16Opt::NoOpt && prm->compute_capability.first < 7) {
        prm->nlmeans.fp16 = VppNLMeansFP16Opt::NoOpt;
    }
    const int search_radius = prm->nlmeans.searchSize / 2;
    // メモリへの書き込みが衝突しないよう、ブロックごとに書き込み先のバッファを分けるが、それがブロックサイズを超えてはいけない
    // x方向は正負両方向にsearch_radius分はみ出し、y方向は負方向にのみsearch_radius分はみ出す
    if (prm->nlmeans.sharedMem && !shared_opt_avail(search_radius)) {
        prm->nlmeans.sharedMem = false;
    }

    const bool use_vtype_fp16 = prm->nlmeans.fp16 != VppNLMeansFP16Opt::NoOpt;
    for (size_t i = 0; i < m_tmpBuf.size(); i++) {
        int tmpBufWidth = 0;
        if (i == TMP_U || i == TMP_V) {
            tmpBufWidth = prm->frameOut.width * ((prm->nlmeans.fp16 != VppNLMeansFP16Opt::NoOpt) ? 16 /*half8*/ : 32/*float8*/);
        } else {
            tmpBufWidth = prm->frameOut.width * ((prm->nlmeans.fp16 == VppNLMeansFP16Opt::All) ? 4 /*half2*/ : 8 /*float2*/);
        }
        // sharedメモリを使う場合、TMP_U, TMP_VとTMP_IW0～TMP_IW3のみ使用する(TMP_IW4以降は不要)
        if (prm->nlmeans.sharedMem && i >= 6) {
            m_tmpBuf[i].reset();
            continue;
        }
        const int tmpBufHeight = prm->frameOut.height;
        if (m_tmpBuf[i]
            && (m_tmpBuf[i]->frame.width != tmpBufWidth || m_tmpBuf[i]->frame.height != tmpBufHeight)) {
            m_tmpBuf[i].reset();
        }
        if (!m_tmpBuf[i]) {
            auto bufCsp = RGY_CSP_NA;
            switch (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp]) {
            case RGY_CHROMAFMT_RGB:
                bufCsp = (rgy_csp_has_alpha(prm->frameOut.csp)) ? RGY_CSP_RGBA : RGY_CSP_RGB;
                break;
            case RGY_CHROMAFMT_YUV444:
                bufCsp = (rgy_csp_has_alpha(prm->frameOut.csp)) ? RGY_CSP_YUVA444 : RGY_CSP_YUV444;
                break;
            case RGY_CHROMAFMT_YUV420:
                bufCsp = (rgy_csp_has_alpha(prm->frameOut.csp)) ? RGY_CSP_YUVA420 : RGY_CSP_YV12;
                break;
            default:
                AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            m_tmpBuf[i] = std::make_unique<CUFrameBuf>(tmpBufWidth, tmpBufHeight, bufCsp);
            sts = m_tmpBuf[i]->alloc();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int requiredCacheSize = (prm->nlmeans.d > 0) ? (2 * prm->nlmeans.d + 1) : 0;
    const bool cacheResetRequired = (int)m_cacheFrames.size() != requiredCacheSize
        || (requiredCacheSize > 0 && !m_cacheFrames.empty()
            && cmpFrameInfoCspResolution(&m_cacheFrames[0]->frame, &prm->frameIn));
    if (cacheResetRequired) {
        m_cacheFrames.clear();
        for (int i = 0; i < requiredCacheSize; i++) {
            auto cacheFrame = std::make_unique<CUFrameBuf>(prm->frameIn);
            sts = cacheFrame->alloc();
            if (sts != RGY_ERR_NONE) {
                m_cacheFrames.clear();
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame %d: %s.\n"), i, get_err_mes(sts));
                return sts;
            }
            m_cacheFrames.push_back(std::move(cacheFrame));
        }
        m_inputCount = 0;
        m_outputCount = 0;
        m_drained = false;
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
#else
    AddMessage(RGY_LOG_ERROR, _T("nlmeans not compiled in this build.\n"));
    return RGY_ERR_UNSUPPORTED;
#endif //ENABLE_VPP_NLMEANS
}

tstring NVEncFilterParamDenoiseNLMeans::print() const {
    return nlmeans.print();
}

RGY_ERR NVEncFilterDenoiseNLMeans::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
#if ENABLE_VPP_NLMEANS
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    if (prm->nlmeans.d == 0) {
        if (!hasInput) {
            return sts;
        }
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
        }
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        if (interlaced(*pInputFrame)) {
            return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
        }
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        sts = denoiseFrame(ppOutputFrames[0], pInputFrame, std::vector<const RGYFrameInfo *>(), stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
        return sts;
    }

    const int d = prm->nlmeans.d;
    const int cacheSize = (int)m_cacheFrames.size();

    if (hasInput) {
        if (interlaced(*pInputFrame)) {
            AddMessage(RGY_LOG_ERROR, _T("temporal nlmeans does not support interlaced input.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_cacheFrames[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }

        const int slot = m_inputCount % cacheSize;
        RGYFrameInfo *pSlot = &m_cacheFrames[slot]->frame;
        auto copyErr = copyFrameAsync(pSlot, pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input to cache slot %d: %s.\n"), slot, get_err_mes(copyErr));
            return copyErr;
        }
        copyFrameProp(pSlot, pInputFrame);

        m_inputCount++;
        if (m_inputCount <= d) {
            return RGY_ERR_NONE;
        }
        const int idx_cur = m_outputCount % cacheSize;
        return emitFrame(idx_cur, ppOutputFrames, pOutputFrameNum, stream);
    }

    if (m_outputCount < m_inputCount) {
        const int idx_cur = m_outputCount % cacheSize;
        return emitFrame(idx_cur, ppOutputFrames, pOutputFrameNum, stream);
    }
    m_drained = true;
    return RGY_ERR_NONE;
#else
    AddMessage(RGY_LOG_ERROR, _T("nlmeans not compiled in this build.\n"));
    return RGY_ERR_UNSUPPORTED;
#endif
}

void NVEncFilterDenoiseNLMeans::close() {
    for (auto& cache : m_cacheFrames) {
        cache.reset();
    }
    m_cacheFrames.clear();
    m_inputCount = 0;
    m_outputCount = 0;
    m_drained = false;
    m_frameBuf.clear();
}
