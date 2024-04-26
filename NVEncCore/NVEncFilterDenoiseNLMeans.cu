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

// atomic_addを試したが、___syncthreadsしたほうが速い
#define ATOMIC_OPT 0

static const int maxPatchRadius = 10;

static bool shared_opt_avail(const int search_radius) {
    return search_radius * 2 <= NLEANS_BLOCK_X && search_radius <= NLEANS_BLOCK_Y;
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

template<typename Type, int bit_depth, typename TmpVType>
__device__ __inline__ float8 calc_sqdiff(Type val0, float8 val1) {
    float8 val0_1 = (float8)val0 - val1;
    const float8 fdiff = val0_1 * (float)(1.0f / ((1 << bit_depth) - 1));
    return fdiff * fdiff;
}

template<typename Type, int bit_depth, typename TmpVType>
__device__ __inline__ half8 calc_sqdiff(Type val0, half8 val1) {
#if ENABLE_CUDA_FP16_DEVICE
    half8 val0_1 = val1 - (TmpVType)val0;
    const half8 fdiff = val0_1 * (__half)(1.0f / ((1 << bit_depth) - 1));
    return fdiff * fdiff;
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

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8>
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

        TmpVType8 val1 = TmpVType8(
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s0, yoffset.s0, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s1, yoffset.s1, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s2, yoffset.s2, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s3, yoffset.s3, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s4, yoffset.s4, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s5, yoffset.s5, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s6, yoffset.s6, width, height),
            (TmpVType)get_xyoffset_pix<Type>(pSrc, srcPitch, ix, iy, xoffset.s7, yoffset.s7, width, height));

        TmpVType8 *ptrDst = (TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        ptrDst[0] = calc_sqdiff<Type, bit_depth, TmpVType>(val0, val1);
    }
}

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8>
RGY_ERR nlmeansCalcDiffSquare(
    RGYFrameInfo *pTmpUPlane,
    const RGYFrameInfo *pInputPlane,
    const int8 xoffset, const int8 yoffset,
    cudaStream_t stream
) {
    dim3 blockSize(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
    dim3 gridSize(divCeil(pInputPlane->width, NLEANS_BLOCK_X), divCeil(pInputPlane->height, NLEANS_BLOCK_Y));
    kernel_calc_diff_square<Type, bit_depth, TmpVType, TmpVType8><<<gridSize, blockSize, 0, stream >>>(
        (char *)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
        (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, xoffset, yoffset);
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}

__device__ __inline__ float8 calc_v_add(float8 val0, float8 val1) {
    val0 += val1;
    return val0;
}

__device__ __inline__ half8 calc_v_add(half8 val0, half8 val1) {
#if ENABLE_CUDA_FP16_DEVICE
    val0 += val1;
#endif
    return val0;
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
                sum = calc_v_add(sum, vals);
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
__device__ __inline__ void add_reverse_side_offset(char *__restrict__ pImgW, const int tmpPitch, const int width, const int height, const int jx, const int jy, const TmpWPType pixNormalized, const TmpWPType weight) {
    static_assert(sizeof(TmpWPType) * 2 == sizeof(TmpWPType2), "sizeof(TmpWPType) * 2 == sizeof(TmpWPType2)");
    if (0 <= jx && jx < width && 0 <= jy && jy < height) {
        TmpWPType2 *ptrImgW = (TmpWPType2 *)(pImgW + jy * tmpPitch + jx * sizeof(TmpWPType2));
        TmpWPType2 weight_pix_2 = { weight * pixNormalized, weight };
        ptrImgW[0] += weight_pix_2;
    }
}

template<typename Type, int bit_depth, typename TmpWPType>
__device__ __inline__ TmpWPType getSrcPixXYOffset(const char *__restrict__ pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int xoffset, const int yoffset) {
    const Type pix = *(const Type *)(pSrc + clamp(iy+yoffset, 0, height-1) * srcPitch + clamp(ix+xoffset,0,width-1) * sizeof(Type));
    return (TmpWPType)pix * (1.0f / ((1<<bit_depth) - 1));
}

template<typename Type, int bit_depth, typename TmpWPType, typename TmpWPType8>
__device__ TmpWPType8 getSrcPixXYOffset8(const char *__restrict__ pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int8 xoffset, const int8 yoffset) {
    static_assert(sizeof(TmpWPType) * 8 == sizeof(TmpWPType8), "sizeof(TmpWPType) * 8 == sizeof(TmpWPType8)");
    TmpWPType8 pix8 = TmpWPType8(
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s0, yoffset.s0),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s1, yoffset.s1),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s2, yoffset.s2),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s3, yoffset.s3),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s4, yoffset.s4),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s5, yoffset.s5),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s6, yoffset.s6),
        getSrcPixXYOffset<Type, bit_depth, TmpWPType>(pSrc, srcPitch, width, height, ix, iy, xoffset.s7, yoffset.s7));
    return pix8;
}

__device__ __inline__ float8 to_float8(half8 v) {
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

__device__ __inline__ float8 to_float8(float8 v) {
    return v;
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8>
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
        const TmpWPType8 v_tmpv8 = to_float8(v_vt8); // expを使う前にfp32に変換
        const TmpWPType8 weight = __expf(max(v_tmpv8 - (2.0f * sigma), (TmpWPType8)0.0f) * -inv_param_h_h);

        // 自分のほうはここですべて同じバッファ(ptrImgW0)に足し込んでしまう
        {
            TmpWPType2 *ptrImgW0 = (TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
            TmpWPType8 pix8 = getSrcPixXYOffset8<Type, bit_depth, TmpWPType, TmpWPType8>(pSrc, srcPitch, width, height, ix, iy, xoffset, yoffset);
            TmpWPType8 weight_pix8 = weight * pix8;
            TmpWPType2 weight_pix_2 = {
                weight_pix8.s0 + weight_pix8.s1 + weight_pix8.s2 + weight_pix8.s3 + weight_pix8.s4 + weight_pix8.s5 + weight_pix8.s6 + weight_pix8.s7,
                weight.s0 + weight.s1 + weight.s2 + weight.s3 + weight.s4 + weight.s5 + weight.s6 + weight.s7
            };
            ptrImgW0[0] += weight_pix_2;
        }
        // 反対側は衝突を避けるため、別々に足し込む
        const Type pix = *(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        const TmpWPType pixNormalized = (TmpWPType)pix * (1.0f / ((1<<bit_depth) - 1));
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW1, tmpPitch, width, height, ix + xoffset.s0, iy + yoffset.s0, pixNormalized, weight.s0);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW2, tmpPitch, width, height, ix + xoffset.s1, iy + yoffset.s1, pixNormalized, weight.s1);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW3, tmpPitch, width, height, ix + xoffset.s2, iy + yoffset.s2, pixNormalized, weight.s2);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW4, tmpPitch, width, height, ix + xoffset.s3, iy + yoffset.s3, pixNormalized, weight.s3);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW5, tmpPitch, width, height, ix + xoffset.s4, iy + yoffset.s4, pixNormalized, weight.s4);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW6, tmpPitch, width, height, ix + xoffset.s5, iy + yoffset.s5, pixNormalized, weight.s5);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW7, tmpPitch, width, height, ix + xoffset.s6, iy + yoffset.s6, pixNormalized, weight.s6);
        add_reverse_side_offset<TmpWPType, TmpWPType2>(pImgW8, tmpPitch, width, height, ix + xoffset.s7, iy + yoffset.s7, pixNormalized, weight.s7);
    }
}

template<typename TmpWPType, typename TmpWPType2, int search_radius>
__device__ __inline__ void add_tmpwp_local(TmpWPType2 tmpWP[search_radius + NLEANS_BLOCK_Y][search_radius * 2 + NLEANS_BLOCK_X], const TmpWPType pixNormalized, const TmpWPType weight, const int thx, const int thy) {
    TmpWPType2 tmp = { weight * pixNormalized, weight };
#if ATOMIC_OPT
#if __CUDA_ARCH__ >= 900
    atomicAdd(&tmpWP[thy + search_radius][thx + search_radius], tmp);
#else
    atomicAdd(&tmpWP[thy + search_radius][thx + search_radius].x, tmp.x);
    atomicAdd(&tmpWP[thy + search_radius][thx + search_radius].y, tmp.y);
#endif
#else
    tmpWP[thy + search_radius][thx + search_radius] += tmp;
#endif
}

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int search_radius>
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

    TmpWPType8 weight = TmpWPType8(0.0f);
    if (ix < width && iy < height) {
        const TmpVType8 v_vt8 = TmpVType8::load((const TmpVType8 *)(pV + iy * vPitch + ix * sizeof(TmpVType8)));
        const TmpWPType8 v_tmpv8 = to_float8(v_vt8); // expを使う前にfp32に変換
        weight = __expf(max(v_tmpv8 - (2.0f * sigma), (TmpWPType8)0.0f) * -inv_param_h_h);

        // 自分のほうはここですべて同じバッファ(ptrImgW0)に足し込んでしまう
        {
            TmpWPType8 pix8 = getSrcPixXYOffset8<Type, bit_depth, TmpWPType, TmpWPType8>(pSrc, srcPitch, width, height, ix, iy, xoffset, yoffset);
            TmpWPType8 weight_pix8 = weight * pix8;
            TmpWPType2 weight_pix_2 = {
                weight_pix8.s0 + weight_pix8.s1 + weight_pix8.s2 + weight_pix8.s3 + weight_pix8.s4 + weight_pix8.s5 + weight_pix8.s6 + weight_pix8.s7,
                weight.s0 + weight.s1 + weight.s2 + weight.s3 + weight.s4 + weight.s5 + weight.s6 + weight.s7
            };
            tmpWP[thy + search_radius][thx + search_radius] += weight_pix_2;
        }
    }
    // 共有メモリ上ですべて足し込んでしまう
    // 計算が衝突しないよう、書き込みごとに同期する
    __syncthreads();
    TmpWPType pixNormalized = 0.0f;
    if (ix < width && iy < height) {
        const Type pix = *(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        pixNormalized = pix * (1.0f / ((1 << bit_depth) - 1));
    }
#if ATOMIC_OPT
#define SYNC_THREADS
#else
#define SYNC_THREADS __syncthreads()
#endif

    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s0, thx + xoffset.s0, thy + yoffset.s0);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s1, thx + xoffset.s1, thy + yoffset.s1);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s2, thx + xoffset.s2, thy + yoffset.s2);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s3, thx + xoffset.s3, thy + yoffset.s3);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s4, thx + xoffset.s4, thy + yoffset.s4);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s5, thx + xoffset.s5, thy + yoffset.s5);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s6, thx + xoffset.s6, thy + yoffset.s6);
    SYNC_THREADS;
    add_tmpwp_local<TmpWPType, TmpWPType2, search_radius>(tmpWP, pixNormalized, weight.s7, thx + xoffset.s7, thy + yoffset.s7);
    SYNC_THREADS;

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

template<typename Type, int bit_depth, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8, int search_radius>
RGY_ERR nlmeansCalcWeight(
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
        kernel_denoise_nlmeans_calc_weight_shared_opt<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, search_radius> <<<gridSize, blockSize, 0, stream >>> (
            (char *)pTmpIWPlane[0].ptr[0], (char *)pTmpIWPlane[1].ptr[0], (char *)pTmpIWPlane[2].ptr[0], (char *)pTmpIWPlane[3].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (const char *)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
            (const char *)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pInputPlane->width, pInputPlane->height,
            sigma, inv_param_h_h,
            xoffset, yoffset, yoffsetmin);
    } else {
        kernel_denoise_nlmeans_calc_weight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8> << <gridSize, blockSize, 0, stream >> > (
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

template<typename Type, int bit_depth, typename TmpWPType2>
__global__ void kernel_denoise_nlmeans_normalize(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pImgW0,
    const char *__restrict__ pImgW1, const char *__restrict__ pImgW2, const char *__restrict__ pImgW3, const char *__restrict__ pImgW4,
    const char *__restrict__ pImgW5, const char *__restrict__ pImgW6, const char *__restrict__ pImgW7, const char *__restrict__ pImgW8,
    const int tmpPitch,
    const int width, const int height
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const TmpWPType2 imgW0 = *(const TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW1 = *(const TmpWPType2 *)(pImgW1 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW2 = *(const TmpWPType2 *)(pImgW2 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW3 = *(const TmpWPType2 *)(pImgW3 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW4 = *(const TmpWPType2 *)(pImgW4 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW5 = *(const TmpWPType2 *)(pImgW5 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW6 = *(const TmpWPType2 *)(pImgW6 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW7 = *(const TmpWPType2 *)(pImgW7 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW8 = *(const TmpWPType2 *)(pImgW8 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = imgW0.x + imgW1.x + imgW2.x + imgW3.x + imgW4.x + imgW5.x + imgW6.x + imgW7.x + imgW8.x;
        const float weight = imgW0.y + imgW1.y + imgW2.y + imgW3.y + imgW4.y + imgW5.y + imgW6.y + imgW7.y + imgW8.y;
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(imgW * __frcp_rn(weight) * ((1<<bit_depth) - 1), 0.0f, (1<<bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth, typename TmpWPType2>
__global__ void kernel_denoise_nlmeans_normalize_shared_opt(
    char *__restrict__ pDst, const int dstPitch,
    const char *__restrict__ pImgW0, const char *__restrict__ pImgW1, const char *__restrict__ pImgW2, const char *__restrict__ pImgW3,
    const int tmpPitch,
    const int width, const int height
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const TmpWPType2 imgW0 = *(const TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW1 = *(const TmpWPType2 *)(pImgW1 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW2 = *(const TmpWPType2 *)(pImgW2 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW3 = *(const TmpWPType2 *)(pImgW3 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = imgW0.x + imgW1.x + imgW2.x + imgW3.x;
        const float weight = imgW0.y + imgW1.y + imgW2.y + imgW3.y;
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(imgW * __frcp_rn(weight) * ((1 << bit_depth) - 1), 0.0f, (1 << bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth, typename TmpWPType2>
RGY_ERR nlmeansNormalize(
    RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pTmpIWPlane,
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
            pOutputPlane->width, pOutputPlane->height);
    } else {
        kernel_denoise_nlmeans_normalize<Type, bit_depth, TmpWPType2> << <gridSize, blockSize, 0, stream >> > (
            (char *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (const char *)pTmpIWPlane[0].ptr[0],
            (const char *)pTmpIWPlane[1].ptr[0], (const char *)pTmpIWPlane[2].ptr[0], (const char *)pTmpIWPlane[3].ptr[0], (const char *)pTmpIWPlane[4].ptr[0],
            (const char *)pTmpIWPlane[5].ptr[0], (const char *)pTmpIWPlane[6].ptr[0], (const char *)pTmpIWPlane[7].ptr[0], (const char *)pTmpIWPlane[8].ptr[0],
            pTmpIWPlane[0].pitch[0],
            pOutputPlane->width, pOutputPlane->height);
    }
    CUDA_DEBUG_SYNC_ERR;
    return err_to_rgy(cudaGetLastError());
}


class NLMeansFuncsBase {
public:
    NLMeansFuncsBase() {};
    virtual ~NLMeansFuncsBase() {};

    virtual decltype(nlmeansCalcDiffSquare<uint8_t, 8, float, float8>)* calcDiffSquare() = 0;
    virtual decltype(nlmeansCalcV<uint8_t, 1, float8>)* calcV(int template_radius) = 0;
    virtual decltype(nlmeansCalcWeight<uint8_t, 8, float8, float, float2, float8, 1>)* calcWeight(int search_radius) = 0;
    virtual decltype(nlmeansNormalize<uint8_t, 8, float2>)* normalize() = 0;
};

template<typename Type, int bit_depth, typename TmpVType, typename TmpVType8, typename TmpWPType, typename TmpWPType2, typename TmpWPType8>
class NLMeansFuncs : public NLMeansFuncsBase {
public:
    NLMeansFuncs() {};
    virtual ~NLMeansFuncs() {};

    virtual decltype(nlmeansCalcDiffSquare<Type, bit_depth, TmpVType, TmpVType8>)* calcDiffSquare() override { return nlmeansCalcDiffSquare<Type, bit_depth, TmpVType, TmpVType8>; }
    virtual decltype(nlmeansCalcV<Type, 1, TmpVType8>)* calcV(int template_radius) override {
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
    virtual decltype(nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8, 1>)* calcWeight(int search_radius) override {
        switch (search_radius) {
        case 1:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  1>;
        case 2:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  2>;
        case 3:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  3>;
        case 4:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  4>;
        case 5:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  5>;
        case 6:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  6>;
        case 7:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  7>;
        case 8:  return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  8>;
        default: return nlmeansCalcWeight<Type, bit_depth, TmpVType8, TmpWPType, TmpWPType2, TmpWPType8,  9>;
        }
    }
    virtual decltype(nlmeansNormalize<Type, bit_depth, TmpWPType2>)* normalize() override { return nlmeansNormalize<Type, bit_depth, TmpWPType2>; }
};

std::unique_ptr<NLMeansFuncsBase> getNLMeansFunc(const RGY_CSP csp, VppFpPrecision weightPrec) {
    switch (csp) {
    case RGY_CSP_YV12:
    case RGY_CSP_YUV444:
        return (weightPrec == VPP_FP_PRECISION_FP32)
            ? std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint8_t, 8, float, float8, float, float2, float8>())
            : std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint8_t, 8, __half, half8, float, float2, float8>());
    case RGY_CSP_P010:
    case RGY_CSP_YUV444_16:
        return (weightPrec == VPP_FP_PRECISION_FP32)
            ? std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint16_t, 16, float, float8, float, float2, float8>())
            : std::unique_ptr<NLMeansFuncsBase>(new NLMeansFuncs<uint16_t, 16, __half, half8, float, float2, float8>());
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
    cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto func = getNLMeansFunc(pOutputPlane->csp, prm->nlmeans.prec);
    if (!func) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    // 一時バッファを初期化
    auto err = setPlaneAsync(&pTmpIWPlane[0], 0, stream);
    CUDA_DEBUG_SYNC_ERR;
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error setPlane[IW0](%s): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    for (int i = 1; i < RGY_NLMEANS_DXDY_STEP+1; i++) {
        err = setPlaneAsync(&pTmpIWPlane[i], 0, stream);
        CUDA_DEBUG_SYNC_ERR;
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error setPlane[IW%d](%s): %s.\n"), i, RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    const int template_radius = prm->nlmeans.patchSize / 2;

    // 計算すべきnx-nyの組み合わせを列挙
    const int search_radius = prm->nlmeans.searchSize / 2;
    std::vector<std::pair<int, int>> nxny;
    for (int ny = -search_radius; ny <= 0; ny++) {
        for (int nx = -search_radius; nx <= search_radius; nx++) {
            if (ny * (2 * search_radius - 1) + nx < 0) { // nx-nyの対称性を使って半分のみ計算 (0,0)
                nxny.push_back(std::make_pair(nx, ny));
            }
        }
    }
    // nx-nyの組み合わせをRGY_NLMEANS_DXDY_STEP個ずつまとめて計算して高速化
    for (size_t inxny = 0; inxny < nxny.size(); inxny += RGY_NLMEANS_DXDY_STEP) {
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
            err = func->calcDiffSquare()(pTmpUPlane, pInputPlane, nx0, ny0, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcDiffSquare (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
        {
            auto funcV = func->calcV(template_radius);
            if (!funcV) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcDiffSquare (denoisePlane(%s)): unsupported patchSize %d.\n"), RGY_CSP_NAMES[pInputPlane->csp], prm->nlmeans.patchSize);
                return err;
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
            err = func->calcWeight(search_radius)(
                pTmpIWPlane, pTmpVPlane, pInputPlane,
                prm->nlmeans.sigma, 1.0f / (prm->nlmeans.h * prm->nlmeans.h),
                nx0, ny0, nymin, prm->nlmeans.sharedMem, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at calcWeight (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }
    // 最後に規格化
    {
        err = func->normalize()(pOutputPlane, pTmpIWPlane, prm->nlmeans.sharedMem, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at normalize (denoisePlane(%s)): %s.\n"), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseNLMeans::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
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
                memset(&pTmpIWPlane[j], 0, sizeof(pTmpIWPlane[j]));
            }
        }
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, pTmpIWPlane.data(), &planeSrc, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) frame(%d) %s: %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoiseNLMeans::NVEncFilterDenoiseNLMeans() : m_tmpBuf() {
    m_name = _T("nlmeans");
}

NVEncFilterDenoiseNLMeans::~NVEncFilterDenoiseNLMeans() {
    close();
}

RGY_ERR NVEncFilterDenoiseNLMeans::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
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
    //if (pNLMeansParam->nlmeans.radius > KNN_RADIUS_MAX) {
    //    AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
    //    return RGY_ERR_INVALID_PARAM;
    //}
    if (prm->nlmeans.sigma < 0.0 || 1.0 < prm->nlmeans.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("sigma should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.h < 0.0 || 1.0 < prm->nlmeans.h) {
        AddMessage(RGY_LOG_ERROR, _T("h should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.prec != VPP_FP_PRECISION_FP32) {
        prm->nlmeans.prec =
#if ENABLE_CUDA_FP16_HOST
            ((prm->compute_capability.first == 6 && prm->compute_capability.second == 0)
                || prm->compute_capability.first >= 7)
            ? VPP_FP_PRECISION_FP16 : VPP_FP_PRECISION_FP32;
#else
            VPP_FP_PRECISION_FP32;
#endif
    }
    const int search_radius = prm->nlmeans.searchSize / 2;
    // メモリへの書き込みが衝突しないよう、ブロックごとに書き込み先のバッファを分けるが、それがブロックサイズを超えてはいけない
    // x方向は正負両方向にsearch_radius分はみ出し、y方向は負方向にのみsearch_radius分はみ出す
    if (prm->nlmeans.sharedMem && !shared_opt_avail(search_radius)) {
        prm->nlmeans.sharedMem = false;
    }

    const bool use_vtype_fp16 = prm->nlmeans.prec != VPP_FP_PRECISION_FP32;
    for (size_t i = 0; i < m_tmpBuf.size(); i++) {
        int tmpBufWidth = 0;
        if (i == TMP_U || i == TMP_V) {
            tmpBufWidth = prm->frameOut.width * ((use_vtype_fp16) ? 16 /*half8*/ : 32/*float8*/);
        } else {
            tmpBufWidth = prm->frameOut.width * 8 /*float2*/;
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
            case RGY_CHROMAFMT_YUV444:
                bufCsp = RGY_CSP_YUV444;
                break;
            case RGY_CHROMAFMT_YUV420:
                bufCsp = RGY_CSP_YV12;
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

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseNLMeans::print() const {
    return nlmeans.print();
}

RGY_ERR NVEncFilterDenoiseNLMeans::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    if (pInputFrame->ptr[0] == nullptr) {
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
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
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
    sts = denoiseFrame(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterDenoiseNLMeans::close() {
    m_frameBuf.clear();
}
