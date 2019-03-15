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

#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterNnedi.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#if __CUDACC_VER_MAJOR__ >= 10
#include "cuda_fp16.h"
#include "cuda_fp16.hpp"
#endif
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util.h"

static const int NNEDI_BLOCK_X       = 32;
static const int NNEDI_BLOCK_Y       = 8;

static const int weight0size = 49 * 4 + 5 * 4 + 9 * 4;
static const int weight0sizenew = 4 * 65 + 4 * 5;

static const int TRASNPOSE_BLOCK_DIM = 16;
static const int TRASNPOSE_TILE_DIM  = 64;

template<typename TypePixel4>
__global__ void kernel_transpose_frame(
    uint8_t *__restrict__ pDst,
    const int dstPitch,
    const int dstWidth,  // = srcHeight
    const int dstHeight, // = srcWidth
    const uint8_t *__restrict__ pSrc,
    const int srcPitch
    ) {
    __shared__ decltype(TypePixel4::x) stemp[TRASNPOSE_TILE_DIM][TRASNPOSE_TILE_DIM + 4];
    {
        const int gSrcIdX = blockIdx.y * TRASNPOSE_TILE_DIM + threadIdx.x * 4;
        const int gSrcIdY = blockIdx.x * TRASNPOSE_TILE_DIM + threadIdx.y;
        const int srcWidth = dstHeight;
        const int srcHeight = dstWidth;
        if (gSrcIdX < srcWidth) {
            for (int j = 0; j < TRASNPOSE_TILE_DIM; j++) {
                TypePixel4 val ={ 0, 0, 0, 0 };
                if (gSrcIdY + j < srcWidth) {
                    TypePixel4 *ptr_src = (TypePixel4 *)(pSrc + (gSrcIdY+j) * srcPitch + gSrcIdX * sizeof(TypePixel4));
                    val = ptr_src[0];
                }
                *(TypePixel4 *)&stemp[threadIdx.y+j][threadIdx.x * 4] = val;
            }
        }
    }
    __syncthreads();

    const int gDstIdX = blockIdx.x * TRASNPOSE_TILE_DIM + threadIdx.x * 4;
    const int gDstIdY = blockIdx.y * TRASNPOSE_TILE_DIM + threadIdx.y;
    if (gDstIdX < dstWidth) {
        for (int j = 0; j < TRASNPOSE_TILE_DIM; j++) {
            if (gDstIdY + j < dstHeight) {
                TypePixel4 val;
                val.x = stemp[threadIdx.x * 4 + 0][threadIdx.y+j];
                val.y = stemp[threadIdx.x * 4 + 1][threadIdx.y+j];
                val.z = stemp[threadIdx.x * 4 + 2][threadIdx.y+j];
                val.w = stemp[threadIdx.x * 4 + 3][threadIdx.y+j];
                TypePixel4 *ptr_dst = (TypePixel4 *)(pDst + (gDstIdY+j) * dstPitch + gDstIdX * sizeof(TypePixel4));
                *ptr_dst = val;
            }
        }
    }
};

__device__ __inline__
static float elliott(float val) {
    return val * __frcp_rn(1.0f + fabs(val));
}

__device__ __inline__
static float exp_(float val) {
    return __expf(clamp(val, -80.0f, 80.0f));
}

#define ENABLE_CUDA_FP16_DEVICE (__CUDACC_VER_MAJOR__ >= 10 && __CUDA_ARCH__ >= 530)
#define ENABLE_CUDA_FP16_HOST   (__CUDACC_VER_MAJOR__ >= 10)

//dot_product1で重み(nns)方向のループアンロールを行う
//これにより、一度sharedメモリからレジスタにのせたpixel情報を使いまわすことができる
#define ENABLE_DP1_WEIGHT_LOOP_UNROLL 1

//重み(nns)方向のループアンロール数
//やりすぎると使用レジスタ数が増え、かえって遅くなる
#define WEIGHT_LOOP 4
static_assert(WEIGHT_LOOP <= WARP_SIZE, "WEIGHT_LOOP < WARP_SIZE");

//ENABLE_DP1_WEIGHT_LOOP_UNROLLに対応して通常の重みの並び [nns*2][nnxy]を変更する
//並びは[nns/WEIGHT_LOOP][nnxy][WEIGHT_LOOP][2]
#define ENABLE_DP1_WEIGHT_ARRAY_OPT (1 && ENABLE_DP1_WEIGHT_LOOP_UNROLL)

//shuffle命令を使ったweight係数の分配により高速化する
#define ENABLE_DP1_SHUFFLE_OPT 1

//スレッド内で複数の出力を同時に計算する
#define THREAD_Y_LOOP 4

#define SSRC(x,y) ((y)*ssrc_dim+(x))
#define SWHT_IDX(i,thIdWeight) ((thIdWeight)*sweight_dim+(i))

template<typename TypeSSrc>
__device__ __inline__
void load_texSrc(TypeSSrc *const ptr_src, const int ssrc_dim, cudaTextureObject_t texSrc, const int nnx, const int nny, const int nnx_2_m1, const int nny_2, const int thIdX, const int thIdY, const int gIdX, const int gIdY, const int pix_x_per_thread, const int thread_y_loop);

template<>
__device__ __inline__
void load_texSrc<float>(float *const ptr_src, const int ssrc_dim, cudaTextureObject_t texSrc, const int nnx, const int nny, const int nnx_2_m1, const int nny_2, const int thIdX, const int thIdY, const int gIdX, const int gIdY, const int pix_x_per_thread, const int thread_y_loop) {
    for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny; y += NNEDI_BLOCK_Y) {
        for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
            const float px = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + thIdX + x - nnx_2_m1 + 0.5f;
            const float py = blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2 + 0.5f;
            ptr_src[SSRC(x + thIdX, y + thIdY)] = (float)tex2D<float>(texSrc, px, py) * 256.0f;
        }
    }
}
#if ENABLE_CUDA_FP16_HOST
template<>
__device__ __inline__
void load_texSrc<__half2>(__half2 *const ptr_src, const int ssrc_dim, cudaTextureObject_t texSrc, const int nnx, const int nny, const int nnx_2_m1, const int nny_2, const int thIdX, const int thIdY, const int gIdX, const int gIdY, const int pix_x_per_thread, const int thread_y_loop) {
#if ENABLE_CUDA_FP16_DEVICE
    for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny; y += NNEDI_BLOCK_Y) {
        for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
            const float px = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + thIdX + x - nnx_2_m1 + 0.5f;
            const float py = blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2 + 0.5f;
            ptr_src[SSRC(x + thIdX, y + thIdY)] = __floats2half2_rn(
                tex2D<float>(texSrc, px, py),
                tex2D<float>(texSrc, px+1.0f, py));
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<typename TypePixel, int bit_depth>
__device__ __inline__
TypePixel prescreen_flag() {
    return (1<<bit_depth)-1;
}

template<typename TypePixel, typename TypeWeight, bool scale_dummy, bool src_is_frame, int thread_y_loop>
__device__ __inline__
void dot_product0(
    TypeWeight sum[thread_y_loop][WEIGHT_LOOP],
    const TypePixel *const ptr_src, const int ssrc_dim,
    const TypeWeight *const ptr_weight, const int sweight_dim,
    const TypeWeight *__restrict__ weight_offset,
    const int nnx, const int nny, const int thIdX, const int thIdY,
    const int pix_x_per_thread,
    const float mstd[thread_y_loop][4]
) {
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < WEIGHT_LOOP; i++) {
            sum[ithy][i] = 0.0f;
        }
    }
    const TypeWeight *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const int src_index = (src_is_frame) ? SSRC(thIdX * pix_x_per_thread, thIdY * thread_y_loop + y) : SSRC(0, thIdY * thread_y_loop * NNEDI_BLOCK_X + thIdX);
        const TypePixel *ptr_s = &ptr_src[src_index];

        for (int x = 0; x < nnx; x++, ptr_w++, ptr_s++) {
            TypePixel s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[(src_is_frame) ? (SSRC(0, ithy)) : (SSRC(0, ithy * NNEDI_BLOCK_X))];
            }
            #pragma unroll
            for (int i = 0; i < WEIGHT_LOOP; i++) {
                TypeWeight w0 = ptr_w[SWHT_IDX(0, i)];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum[ithy][i] += s0[ithy] * w0;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < WEIGHT_LOOP; i++, weight_offset++) {
        const TypeWeight wo = weight_offset[0];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            const TypeWeight scale = (scale_dummy) ? 1.0f : mstd[ithy][2];
            sum[ithy][i] = sum[ithy][i] * scale + wo;
        }
    }
}

template<typename TypePixel, int bit_depth, typename TypeSSrc, int thread_y_loop>
__device__ __inline__
static TypePixel interp_ret(const TypeSSrc *const ptr_src, const int ssrc_dim,
    const bool flag, const int nnx, const int nny, const int thIdX, const int thIdY, int ithy, const int nnx_2_m1, const int nny_2) {
    TypePixel val = prescreen_flag<TypePixel, bit_depth>();
    if (flag) {
        const float tmp =
            (19.0f/32.0f) * (ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 1)] + ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 2)])
            - (3.0f/32.0f) * (ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 0)] + ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 3)]);
        val = (TypePixel)clamp(tmp * ((1<<bit_depth) / 256.0f) + 0.5f, 0, (1<<bit_depth)-1);
    }
    return val;
}

template<typename TypePixel4, int bit_depth, typename TypeSSrc, typename TypeWeight, bool prescreen_orig, int thread_y_loop>
__global__ void kernel_comute_network0(
    uint8_t *__restrict__ pDst, //top field / bottom field は考慮済みとする
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    cudaTextureObject_t texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
    const TypeWeight *__restrict__ weight,
    const NnediTargetField targetField
    ) {
    const int pix_x_per_thread = prescreen_orig ? 1 : 4/*4ピクセル分一度に処理する*/;
    const int nnx = (prescreen_orig) ? 12 : 16;
    const int nny = 4;
    const int nnxy = nnx * nny;
    const int nns = 4;
    const int thIdX      = threadIdx.x; //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = threadIdx.y; //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       =(blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX) * pix_x_per_thread;
    const int gIdY       =(blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * thread_y_loop; //フィールド単位
    const int stmp_dim = (prescreen_orig) ? 8 : 4;

    //sharedメモリのサイズと使途
    //1.src:    (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0])
    //2.temp:   NNEDI_BLOCK_X * NNEDI_BLOCK_Y * stmp_dim * sizeof(ptr_temp[0])
    __shared__ char shared[
        (NNEDI_BLOCK_X * pix_x_per_thread + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(TypeSSrc) +
        NNEDI_BLOCK_X * NNEDI_BLOCK_Y * thread_y_loop * stmp_dim * sizeof(float)
    ];
    TypeSSrc *const ptr_src = (TypeSSrc *)shared;
    const int ssrc_dim = NNEDI_BLOCK_X * pix_x_per_thread + nnx;

    //input(texture) -> shared
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = (prescreen_orig) ? 5 : 6;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc<TypeSSrc>(ptr_src, ssrc_dim, texSrc, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY, gIdX, gIdY, pix_x_per_thread, thread_y_loop);
    __syncthreads();

    float *const ptr_temp = (float *)((char *)ptr_src
        + ((NNEDI_BLOCK_X * pix_x_per_thread) + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0]));
#define STMP_IDX(i,x,y) ( ((y)*(NNEDI_BLOCK_X)+(x)) * stmp_dim + (i))

    float dummy[thread_y_loop][4];
    const int sweight_dim = nnxy;
    if (prescreen_orig) {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
            float sum[thread_y_loop][WEIGHT_LOOP]; //レジスタにのることを期待する
            dot_product0<TypeSSrc, TypeWeight, true, true, thread_y_loop>(sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+48*4+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
            float sum[thread_y_loop][WEIGHT_LOOP]; //レジスタにのることを期待する
            dot_product0<TypeSSrc, TypeWeight, true, false, thread_y_loop>(sum, ptr_temp, stmp_dim, weight+49*4+iw*4, /*sweight_dim=nnxy=*/4, weight+49*4 + 4*4+iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                    ptr_temp[STMP_IDX(4+iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        float ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
            float sum[thread_y_loop][WEIGHT_LOOP]; //レジスタにのることを期待する
            dot_product0<TypeSSrc, TypeWeight, true, false, thread_y_loop>(sum, ptr_temp, stmp_dim, weight + 4*49 + 4*5+iw*8, /*sweight_dim=nnxy=*/8, weight + 4*49 + 4*5 + 4*8+iw, /*nnx=*/8, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                    ret[ithy][ithw+iw] = sum[ithy][ithw];
                }
            }
        }

        if (gIdX < dstWidth) {
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    const bool flag = (fmaxf(ret[ithy][2], ret[ithy][3]) <= fmaxf(ret[ithy][0], ret[ithy][1])) ? true : false;
                    decltype(TypePixel4::x) *const ptr_dst = (decltype(TypePixel4::x) *)((uint8_t *)pDst + (gIdY + ithy) * dstPitch + gIdX * sizeof(TypePixel4::x));
                    ptr_dst[0] = interp_ret<decltype(TypePixel4::x), bit_depth, TypeSSrc, thread_y_loop>(ptr_src, ssrc_dim, flag, nnx, nny, thIdX, thIdY, ithy, nnx_2_m1, nny_2);
                }
            }
        }
    } else {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
            float sum[thread_y_loop][WEIGHT_LOOP]; //レジスタにのることを期待する
            dot_product0<TypeSSrc, TypeWeight, true, true, thread_y_loop>(sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+64*4+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        float ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
            float sum[thread_y_loop][WEIGHT_LOOP]; //レジスタにのることを期待する
            dot_product0<TypeSSrc, TypeWeight, true, false, thread_y_loop>(sum, ptr_temp, stmp_dim, weight+65*4+iw*4, /*sweight_dim=nnxy=*/4, weight+65*4 + 4*4 + iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                    ret[ithy][ithw+iw] = sum[ithy][ithw];
                }
            }
        }

        if (gIdX < dstWidth) {
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    TypePixel4 *const ptr_dst = (TypePixel4 *)((uint8_t *)pDst + (gIdY + ithy) * dstPitch + gIdX * sizeof(decltype(TypePixel4::x)));
                    //1スレッドで4pixel分出力する
                    TypePixel4 out;
                    out.x = interp_ret<decltype(TypePixel4::x), bit_depth, TypeSSrc, thread_y_loop>(ptr_src+0, ssrc_dim, ret[ithy][0] > 0.0f, nnx, nny, thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    out.y = interp_ret<decltype(TypePixel4::x), bit_depth, TypeSSrc, thread_y_loop>(ptr_src+1, ssrc_dim, ret[ithy][1] > 0.0f, nnx, nny, thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    out.z = interp_ret<decltype(TypePixel4::x), bit_depth, TypeSSrc, thread_y_loop>(ptr_src+2, ssrc_dim, ret[ithy][2] > 0.0f, nnx, nny, thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    out.w = interp_ret<decltype(TypePixel4::x), bit_depth, TypeSSrc, thread_y_loop>(ptr_src+3, ssrc_dim, ret[ithy][3] > 0.0f, nnx, nny, thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    ptr_dst[0] = out;
                }
            }
        }
    }
}

template<typename T> __device__ __inline__ T setval(float val);
template<> __device__ __inline__ float setval(float val) { return val; };
template<typename TypeSSrc> __device__ __inline__ int kernel_comute_network1_calc_scale_step();
template<> __device__ __inline__ int kernel_comute_network1_calc_scale_step<float>() { return 1; };

template<typename TypeSSrc> __device__
    void kernel_comute_network1_calc_scale_get_sum_sumsq(float& sum, float& sumsq, TypeSSrc tsum, TypeSSrc tsumsq);
template<> __device__ __inline__
    void kernel_comute_network1_calc_scale_get_sum_sumsq<float>(float& sum, float& sumsq, float tsum, float tsumsq) {
    sum = tsum, sumsq = tsumsq;
}
#if ENABLE_CUDA_FP16_HOST
template<> __device__ __inline__ int kernel_comute_network1_calc_scale_step<__half2>() { return 2; };
template<> __device__ __inline__ __half2 setval(float val) { return __float2half2_rn(val); }
template<> __device__ __inline__
    void kernel_comute_network1_calc_scale_get_sum_sumsq<__half2>(float& sum, float& sumsq, __half2 tsum, __half2 tsumsq) {
    sum = ((float)tsum.x + (float)tsum.y) * 256.0f;
    sumsq = ((float)tsumsq.x + (float)tsumsq.y) * 256.0f * 256.0f;
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<typename TypeSSrc, typename TypeWeight>
__device__ __inline__
void kernel_comute_network1_calc_scale(
    float mstd[THREAD_Y_LOOP][4],
    TypeWeight *__restrict__ const ptr_temp,
    const TypeSSrc *__restrict__ const ptr_src, const int ssrc_dim,
    const int nnx, const int nny, const int nnxy,
    const int thIdX, const int thIdY) {
    const int step = kernel_comute_network1_calc_scale_step<TypeSSrc>();
#define TMP_IDX(x,y,i) ((((i)*(nny + NNEDI_BLOCK_Y * THREAD_Y_LOOP)+(y))*NNEDI_BLOCK_X)+(x))
    for (int y = 0; y + thIdY < nny + NNEDI_BLOCK_Y * THREAD_Y_LOOP; y += NNEDI_BLOCK_Y) {
        TypeSSrc sum = setval<TypeSSrc>(0.0f), sumsq = setval<TypeSSrc>(0.0f);
        //まず各ピクセルごとに、x方向の総和をとる
        #pragma unroll (4)
        for (int x = 0; x < nnx; x += step) {
            const auto value = ptr_src[SSRC(x + thIdX, y + thIdY)];
            sum += value;
            sumsq += value * value;
        }
        //一度sharedメモリに格納
        ptr_temp[TMP_IDX(thIdX, thIdY+y, 0)] = sum;
        ptr_temp[TMP_IDX(thIdX, thIdY+y, 1)] = sumsq;
    }
    __syncthreads();

    const float inv_nnxy = __frcp_rn(nnxy);

    //次にy方向の総和をとる
    #pragma unroll
    for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
        TypeSSrc tsum = setval<TypeSSrc>(0.0f), tsumsq = setval<TypeSSrc>(0.0f);
        #pragma unroll
        for (int y = 0; y < nny; y++) {
            tsum   += ptr_temp[TMP_IDX(thIdX, thIdY*THREAD_Y_LOOP+ithy+y, 0)];
            tsumsq += ptr_temp[TMP_IDX(thIdX, thIdY*THREAD_Y_LOOP+ithy+y, 1)];
        }

        float sum, sumsq;
        kernel_comute_network1_calc_scale_get_sum_sumsq<TypeSSrc>(sum, sumsq, tsum, tsumsq);

        mstd[ithy][3] = 0.0f;
        mstd[ithy][0] = sum * inv_nnxy;
        float tmp = sumsq * inv_nnxy - mstd[ithy][0] * mstd[ithy][0];
        //if (thIdX == 0 && thIdY == 0 && blockIdx.x == 2 && blockIdx.y == 2) {
        //    printf("%e, %e, %e, %e, %e\n", inv_nnxy, tmp, sum, sumsq, mstd[ithy][0]);
        //}
        if (tmp <= FLT_EPSILON) {
            mstd[ithy][1] = 0.0f;
            mstd[ithy][2] = 0.0f;
        } else {
            mstd[ithy][1] = __fsqrt_rn(tmp);
            mstd[ithy][2] = __frcp_rn(mstd[ithy][1]);
        }
    }
#undef TMP_IDX
}
#if ENABLE_CUDA_FP16_HOST && (!ENABLE_CUDA_FP16_DEVICE)
template<>
__device__ __inline__
void kernel_comute_network1_calc_scale(
    float mstd[THREAD_Y_LOOP][4],
    __half2 *__restrict__ const ptr_temp,
    const __half2 *__restrict__ const ptr_src, const int ssrc_dim,
    const int nnx, const int nny, const int nnxy,
    const int thIdX, const int thIdY) {
    //ダミー
}
#endif //#if ENABLE_CUDA_FP16_HOST && (!ENABLE_CUDA_FP16_DEVICE)

template<typename TypeSSrc, typename TypeWeight>
__device__ __inline__
void dot_product_frame1_fp32(
    float sum0[THREAD_Y_LOOP][WEIGHT_LOOP], //レジスタにのることを期待する
    float sum1[THREAD_Y_LOOP][WEIGHT_LOOP], //レジスタにのることを期待する
    TypeSSrc *__restrict__ const ptr_src, const int ssrc_dim,
    const TypeWeight *__restrict__ const ptr_weight, const int sweight_dim,
    const TypeWeight *__restrict__ weight_offset,
    const int nnx, const int nny, const int nns, const int thIdX, const int thIdY,
    const float mstd[THREAD_Y_LOOP][4]
) {
    #pragma unroll
    for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
        #pragma unroll
        for (int i = 0; i < WEIGHT_LOOP; i++) {
            sum0[ithy][i] = sum1[ithy][i] = 0.0f;
        }
    }
    const TypeWeight *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const TypeSSrc *ptr_s = &ptr_src[SSRC(thIdX, thIdY * THREAD_Y_LOOP + y)];
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
        //#pragma unroll (4)
        for (int x = 0; x < nnx; x++, ptr_s++) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            TypeSSrc s0[THREAD_Y_LOOP];
            #pragma unroll
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
#if ENABLE_DP1_SHUFFLE_OPT
            TypeWeight w;
            if (thIdX < WEIGHT_LOOP*2) w = ptr_w[thIdX];
            ptr_w += WEIGHT_LOOP*2;
            #pragma unroll
            for (int i = 0; i < WEIGHT_LOOP; i++) {
                const auto w0 = __shfl(w, i*2+0);
                const auto w1 = __shfl(w, i*2+1);
                #pragma unroll
                for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                    sum0[ithy][i] += s0[ithy] * w0;
                    sum1[ithy][i] += s0[ithy] * w1;
                }
            }
#else
            #pragma unroll
            for (int i = 0; i < WEIGHT_LOOP; i++, ptr_w += 2) {
                const auto w0 = ptr_w[0];
                const auto w1 = ptr_w[1];
                #pragma unroll
                for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                    sum0[i][ithy] += s0[ithy] * w0;
                    sum1[i][ithy] += s0[ithy] * w1;
                }
            }
#endif
        }
    }
#else
    #pragma unroll (4)
    for (int x = 0; x < nnx; x++, ptr_w++, ptr_s++) {
        //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
        TypePixel s0[THREAD_Y_LOOP];
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            s0[ithy] = ptr_s[SSRC(0, ithy*NNEDI_BLOCK_Y)];
        }
        #pragma unroll
        for (int i = 0; i < WEIGHT_LOOP; i++) {
            TypeWeight w0 = ptr_w[SWHT_IDX(0, i)];
            TypeWeight w1 = ptr_w[SWHT_IDX(0, i+nns)];
            #pragma unroll
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                sum0[i][ithy] += s0[ithy] * w0;
                sum1[i][ithy] += s0[ithy] * w1;
            }
        }
    }
#endif
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    #pragma unroll
    for (int i = 0; i < WEIGHT_LOOP; i++, weight_offset += 2) {
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0];
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[1];
        }
    }
#else
    #pragma unroll
    for (int i = 0; i < WEIGHT_LOOP; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0];
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[nns];
        }
    }
#endif
}

#if ENABLE_CUDA_FP16_HOST
__device__ __inline__
void dot_product_frame1_fp16(
    __half2 sum[THREAD_Y_LOOP][WEIGHT_LOOP],
    __half2 *__restrict__ const ptr_src, const int ssrc_dim,
    const __half2 *__restrict__ const ptr_weight, const int sweight_dim,
    const __half2 *__restrict__ weight_offset,
    const int nnx, const int nny, const int nns, const int thIdX, const int thIdY,
    const __half2 weight_scale[THREAD_Y_LOOP]
) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
        #pragma unroll
        for (int i = 0; i < WEIGHT_LOOP; i++) {
            sum[ithy][i] = setval<__half2>(0.0f);
        }
    }
    const __half2 *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const __half2 *ptr_s = &ptr_src[SSRC(thIdX, thIdY * THREAD_Y_LOOP + y)];
        //#pragma unroll (4)
        for (int x = 0; x < nnx; x += 2, ptr_s += 2) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            __half2 s0[THREAD_Y_LOOP];
            #pragma unroll
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
            //[nns/WEIGHT_LOOP][nnxy][WEIGHT_LOOP][2]
            __half2 w;
            if (thIdX < WEIGHT_LOOP*2) w = ptr_w[thIdX];
            ptr_w += WEIGHT_LOOP*2;
            #pragma unroll
            for (int i = 0; i < WEIGHT_LOOP; i++) {
                __half2 w0 = __shfl(w,            +i);
                __half2 w1 = __shfl(w, WEIGHT_LOOP+i);
                #pragma unroll
                for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                    sum[ithy][i] += __low2half2(s0[ithy]) * w0;
                    sum[ithy][i] += __high2half2(s0[ithy]) * w1;
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < WEIGHT_LOOP; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            sum[ithy][i] = sum[ithy][i] * weight_scale[ithy] + weight_offset[0];
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

__device__ __inline__
void kernel_comute_network1_dot_product(
    float wsum[THREAD_Y_LOOP],
    float vsum[THREAD_Y_LOOP],
    float *const ptr_src, const int ssrc_dim,
    const float *const weight,
    float mstd[THREAD_Y_LOOP][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
        float sum0[THREAD_Y_LOOP][WEIGHT_LOOP]; //レジスタにのることを期待する
        dot_product0<float, float, false, true, THREAD_Y_LOOP>(sum0, ptr_src, ssrc_dim, weight+ (iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        float sum1[THREAD_Y_LOOP][WEIGHT_LOOP]; //レジスタにのることを期待する
        dot_product0<float, float, false, true, THREAD_Y_LOOP>(sum1, ptr_src, ssrc_dim, weight+ (nns+iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy+nns + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]);
                float ret1 = sum1[ithy][ithw];
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
__device__ __inline__
void kernel_comute_network1_dot_product(
    float wsum[THREAD_Y_LOOP],
    float vsum[THREAD_Y_LOOP],
    __half2 *const ptr_src, const int ssrc_dim,
    const __half2 *const weight,
    float mstd[THREAD_Y_LOOP][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    //未実装
    assert(0);
}
#endif //#if ENABLE_CUDA_FP16_HOST

__device__ __inline__
void kernel_comute_network1_dot_product_opt(
    float wsum[THREAD_Y_LOOP],
    float vsum[THREAD_Y_LOOP],
    float *const ptr_src, const int ssrc_dim,
    const float *const weight,
    float mstd[THREAD_Y_LOOP][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
        float sum0[THREAD_Y_LOOP][WEIGHT_LOOP]; //レジスタにのることを期待する
        float sum1[THREAD_Y_LOOP][WEIGHT_LOOP]; //レジスタにのることを期待する
        // 重み(nns)方向に、WEIGHT_LOOP分のdotproduct
        // sum0[i] <- iw     - iw+WEIGHT_LOOP
        // sum1[i] <- iw+nns - iw+WEIGHT_LOOP+nns
        dot_product_frame1_fp32(sum0, sum1, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + (nns*2)*nnxy + iw*2, nnx, nny, nns, thIdX, thIdY, mstd);
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]);
                float ret1 = sum1[ithy][ithw];
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
__device__ __inline__
void kernel_comute_network1_dot_product_opt(
    float wsum[THREAD_Y_LOOP],
    float vsum[THREAD_Y_LOOP],
    __half2 *const ptr_src, const int ssrc_dim,
    const __half2 *const weight,
    float mstd[THREAD_Y_LOOP][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
#if ENABLE_CUDA_FP16_DEVICE
    const int sweight_dim = nnxy;
    for (int iw = 0; iw < nns; iw += WEIGHT_LOOP) {
        __half2 sum[THREAD_Y_LOOP][WEIGHT_LOOP]; //レジスタにのることを期待する
        // 重み(nns)方向に、WEIGHT_LOOP分のdotproduct
        // sum0[i] <- iw     - iw+WEIGHT_LOOP
        // sum1[i] <- iw+nns - iw+WEIGHT_LOOP+nns
        __half2 weight_scale[THREAD_Y_LOOP];
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            weight_scale[ithy] = __float2half2_rn(mstd[ithy][2] * 256.0f); // *weight[nns*(nnxy+1)] <<<<<<<< scalingはとりあえずなしで
        }
        dot_product_frame1_fp16(sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + nns*nnxy + iw, nnx, nny, nns, thIdX, thIdY, weight_scale);
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < WEIGHT_LOOP; ithw++) {
                float ret0 = exp_(__low2float(sum[ithy][ithw]));
                float ret1 = __high2float(sum[ithy][ithw]);
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST


template<typename TypePixel, int bit_depth, typename TypeSSrc, typename TypeWeight, int nnx, int nny>
__global__ void kernel_comute_network1(
    uint8_t *__restrict__ pDst, //top field / bottom field は考慮済みとする
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    cudaTextureObject_t texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
    const TypeWeight *__restrict__ weight10,
    const TypeWeight *__restrict__ weight11,
    const int nns,  // len = nns*2
    const int quals,
    const NnediTargetField targetField,
    bool prescreen
) {
    const int thIdX      = threadIdx.x; //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = threadIdx.y; //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX;
    const int gIdY       =(blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * THREAD_Y_LOOP; //フィールド単位
    const int nnxy       = nnx * nny;

    //sharedメモリのサイズと使途
    //1.src: (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * sizeof(ptr_src[0])
    //2.tmp: (nny + NNEDI_BLOCK_Y * THREAD_Y_LOOP) * NNEDI_BLOCK_X * 2 * sizeof(ptr_temp[0])
    alignas(128) extern __shared__ char shared[];
    TypeSSrc *const ptr_src = (TypeSSrc *)shared;
    const int ssrc_dim = NNEDI_BLOCK_X + nnx;

    //input(texture) -> shared
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = nnx / 2 - 1;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc<TypeSSrc>(ptr_src, ssrc_dim, texSrc, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY, gIdX, gIdY, 1, THREAD_Y_LOOP);
    __syncthreads();

    TypeWeight *const ptr_temp = (TypeWeight *)((char *)shared
        + (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * sizeof(ptr_src[0]));

    float mstd[THREAD_Y_LOOP][4];
    kernel_comute_network1_calc_scale(mstd, ptr_temp, ptr_src, ssrc_dim, nnx, nny, nnxy, thIdX, thIdY);

    uint8_t *const ptr_dst_base = (uint8_t *)pDst + gIdY * dstPitch + gIdX * sizeof(TypePixel);
    uint32_t flag_sum = 0xffffffff; //処理するかどうかのフラグ
    if (prescreen) {
        flag_sum = 0x00;
        uint8_t *ptr_dst = ptr_dst_base;
        #pragma unroll
        for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++, ptr_dst += dstPitch) {
            uint32_t flag = 0x00;
            if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                flag = (((TypePixel *)ptr_dst)[0] == prescreen_flag<TypePixel, bit_depth>()) ? 0x01 << ithy : 0x00;
            }
            flag_sum |= flag;
            static_assert(THREAD_Y_LOOP <= sizeof(flag_sum) * 8, "THREAD_Y_LOOP <= sizeof(flag_sum) * 8");
        }
    }

#if 0
                                      |<-------- nns*2 --------->|
                                    WEIGHT_LOOP
                                      |<-->| ---> 繰り返し処理
                                 ---  |--------------------------|
                                      |                          |
                                      |                          |
                                      |                          |
                             nnxy     |                          |
                                      |                          |
                                      |                          |
                                      |                          |
                                 ---  |--------------------------|

                |<----   nnxy  --->|
            --- |------------------|  |----|
NNEDI_BLOCK_X   |                  |  |    | <-- 各スレッドはこの出力の1pixel分(縦方向)をそれぞれ担当
*NNEDI_BLOCK_Y  |                  |  |    |      横: WEIGHT_LOOP
            --- |                  |  |----|      縦: NNEDI_BLOCK_X * NNEDI_BLOCK_Y
                |                  |
                |                  |
        pixels  |                  |
           |    |                  |
           |    |                  |
        　↓    |                  |

#endif
    //weightの先頭のポインタ
    if (__any(flag_sum)) { //どのpixelも処理する必要がなければ、スキップする
        for (int iquality = 0; iquality < quals; iquality++) {
            const TypeWeight *const weight = (iquality) ? weight11 : weight10;
            float wsum[THREAD_Y_LOOP], vsum[THREAD_Y_LOOP];
            #pragma unroll
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                wsum[ithy] = vsum[ithy] = 0.0f;
            }
            if (ENABLE_DP1_WEIGHT_LOOP_UNROLL) {
                kernel_comute_network1_dot_product_opt(
                    wsum, vsum, ptr_src, ssrc_dim, weight, mstd, nnx, nny, nnxy, nns, thIdX, thIdY);
            } else {
                kernel_comute_network1_dot_product(
                    wsum, vsum, ptr_src, ssrc_dim, weight, mstd, nnx, nny, nnxy, nns, thIdX, thIdY);
            }

            const float min_weight_sum = 1e-10f;
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++) {
                if (wsum[ithy] > min_weight_sum) {
                    mstd[ithy][3] += ((5.0f * vsum[ithy]) * __frcp_rn(wsum[ithy])) * mstd[ithy][1];
                }
                mstd[ithy][3] += mstd[ithy][0];
            }
        }

        if (gIdX < dstWidth) {
            const float scale = (1<<bit_depth) / 256.0f * ((quals > 1) ? 0.5f : 1.0f);
            uint8_t *ptr_dst = (uint8_t *)ptr_dst_base;
            for (int ithy = 0; ithy < THREAD_Y_LOOP; ithy++, ptr_dst += dstPitch) {
                if ((flag_sum & (1<<ithy)) && (gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    ((TypePixel *)ptr_dst)[0] = (TypePixel)clamp(mstd[ithy][3] * scale + 0.5f, 0.0f, (1<<bit_depth)-1.0f);
                }
            }
        }
    }
}

template<typename TypePixel>
cudaError_t setTexField(cudaTextureObject_t& texSrc, const FrameInfo *pFrame, const NnediTargetField targetField) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch * 2; //1行おきなので通常の2倍
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height / 2; //フィールドなので半分
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pFrame->ptr
        + (pFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 1 : 0)); //有効なほうのフィールドを選択

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeClamp;
    texDescSrc.addressMode[1]   = cudaAddressModeClamp;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename TypePixel4, int bit_depth, typename TypeSSrc, typename TypeWeight>
cudaError_t nnedi_compute_network_0(FrameInfo *pOutputPlane,
    cudaTextureObject_t texSrc,
    const TypeWeight *weight0,
    const VppNnediPreScreen pre_screen,
    const NnediTargetField targetField,
    cudaStream_t stream
) {
    dim3 blockSize(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);

    auto cudaerr = cudaSuccess;
    if (pre_screen == VPP_NNEDI_PRE_SCREEN_ORIGINAL) {
        const int thread_y_loop_org = 2;
        dim3 gridSize(
            divCeil(pOutputPlane->width, blockSize.x),
            divCeil(pOutputPlane->height / 2, blockSize.y * thread_y_loop_org));
        kernel_comute_network0<TypePixel4, bit_depth, TypeSSrc, TypeWeight, true, thread_y_loop_org><<<gridSize, blockSize, 0, stream>>>(
            (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch * 2, //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
            weight0, targetField);
        cudaerr = cudaGetLastError();
    } else if (pre_screen >= VPP_NNEDI_PRE_SCREEN_NEW) {
        const int thread_y_loop_new = 2;
        dim3 gridSize(
            divCeil(pOutputPlane->width, blockSize.x * 4 /*4ピクセル分一度に処理する*/),
            divCeil(pOutputPlane->height / 2, blockSize.y * thread_y_loop_new));
        kernel_comute_network0<TypePixel4, bit_depth, TypeSSrc, TypeWeight, false, thread_y_loop_new><<<gridSize, blockSize, 0, stream>>>(
            (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch * 2, //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
            weight0, targetField);
        cudaerr = cudaGetLastError();
    } else {
        const auto outputFrameInfoEx = getFrameInfoExtra(pOutputPlane);
        cudaerr = cudaMemset2DAsync(
            (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1),
            pOutputPlane->pitch * 2, //1行おきなので通常の2倍
            -1, //value
            outputFrameInfoEx.width_byte,
            pOutputPlane->height / 2,
            stream);
    }
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypePixel, int bit_depth, typename TypeSSrc, typename TypeWeight>
cudaError_t nnedi_compute_network_1(
    FrameInfo *pOutputFrame,
    cudaTextureObject_t texSrc,
    const TypeWeight *weight10,
    const TypeWeight *weight11,
    const NnediTargetField targetField,
    const VppNnediNSize nsize,
    const int nns,
    const VppNnediQuality quality,
    const VppNnediPreScreen pre_screen,
    cudaStream_t stream
) {
    dim3 blockSize(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);
    dim3 gridSize(
        divCeil(pOutputFrame->width, blockSize.x),
        divCeil(pOutputFrame->height / 2, blockSize.y * THREAD_Y_LOOP));

    const int nnx = NVEncFilterNnedi::sizeNX[nsize];
    const int nny = NVEncFilterNnedi::sizeNY[nsize];
    const int shared_mem_size =
        (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * sizeof(TypeSSrc) + //src
        (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * NNEDI_BLOCK_X * 2 * sizeof(TypeWeight); //temp

    switch (nsize) {
    case VPP_NNEDI_NSIZE_8x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 8, 6><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_16x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 16, 6><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_32x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 32, 6><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_48x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 48, 6><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_8x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 8, 4><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_16x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 16, 4><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    case VPP_NNEDI_NSIZE_32x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeSSrc, TypeWeight, 32, 4><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen != VPP_NNEDI_PRE_SCREEN_NONE);
        break;
    default:
        return cudaErrorAssert;
    }
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypePixel, typename TypePixel4, int bit_depth, typename TypeSSrc, typename TypeWeight>
cudaError_t proc_plane(
    FrameInfo *pOutputPlane,
    const FrameInfo *pInputPlane,
    const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam,
    const NnediTargetField targetField,
    const float *weight0,
    const TypeWeight *weight10,
    const TypeWeight *weight11,
    cudaStream_t stream
) {
    const auto inputFrameInfoEx = getFrameInfoExtra(pInputPlane);
    // 有効なほうのフィールドをコピー
    auto cudaerr = cudaMemcpy2DAsync(
        (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 1 : 0),
        pOutputPlane->pitch * 2, //1行おきなので通常の2倍
        (uint8_t *)pInputPlane->ptr + pInputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 1 : 0),
        pInputPlane->pitch * 2,  //1行おきなので通常の2倍
        inputFrameInfoEx.width_byte,
        pInputPlane->height / 2,
        cudaMemcpyDeviceToDevice,
        stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    cudaTextureObject_t texSrc = 0;
    cudaerr = setTexField<TypePixel>(texSrc, pInputPlane, targetField);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = nnedi_compute_network_0<TypePixel4, bit_depth, float, float>(pOutputPlane,
        texSrc,
        weight0,
        pNnediParam->nnedi.pre_screen,
        targetField,
        stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = nnedi_compute_network_1<TypePixel, bit_depth, TypeSSrc, TypeWeight>(
        pOutputPlane,
        texSrc,
        weight10,
        weight11,
        targetField,
        pNnediParam->nnedi.nsize,
        pNnediParam->nnedi.nns,
        pNnediParam->nnedi.quality,
        pNnediParam->nnedi.pre_screen,
        stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypePixel, typename TypePixel4, int bit_depth, typename TypeSSrc, typename TypeWeight>
cudaError_t proc_frame(FrameInfo *pOutputFrame,
    const FrameInfo *pInputFrame,
    const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam,
    const NnediTargetField targetField,
    const void *weight0,
    const void *weight10,
    const void *weight11,
    cudaStream_t stream
) {
    static_assert(sizeof(TypePixel4) == sizeof(TypePixel) * 4, "sizeof(TypePixel4) == sizeof(TypePixel) * 4");
    cudaError_t cudaerr = cudaSuccess;
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeSSrc, TypeWeight>(&planeOutputY, &planeInputY, pNnediParam, targetField, (const float *)weight0, (const TypeWeight *)weight10, (const TypeWeight *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeSSrc, TypeWeight>(&planeOutputU, &planeInputU, pNnediParam, targetField, (const float *)weight0, (const TypeWeight *)weight10, (const TypeWeight *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeSSrc, TypeWeight>(&planeOutputV, &planeInputV, pNnediParam, targetField, (const float *)weight0, (const TypeWeight *)weight10, (const TypeWeight *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

const int NVEncFilterNnedi::sizeNX[] = { 8, 16, 32, 48, 8, 16, 32 };
const int NVEncFilterNnedi::sizeNY[] = { 6, 6, 6, 6, 4, 4, 4 };
const int NVEncFilterNnedi::sizeNN[] = { 16, 32, 64, 128, 256 };

NVEncFilterNnedi::NVEncFilterNnedi() : m_weight0(), m_weight1() {
    m_sFilterName = _T("nnedi");
}

NVEncFilterNnedi::~NVEncFilterNnedi() {
    close();
}

NVENCSTATUS NVEncFilterNnedi::checkParam(const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    if (pNnediParam->frameOut.height <= 0 || pNnediParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.field <= VPP_NNEDI_FIELD_UNKNOWN || VPP_NNEDI_FIELD_MAX <= pNnediParam->nnedi.field) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"field\": %d\n"), pNnediParam->nnedi.field);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.nns < 16 || 256 < pNnediParam->nnedi.nns) {
        pNnediParam->nnedi.nns = clamp(pNnediParam->nnedi.nns, 16, 256);
        AddMessage(RGY_LOG_WARN, _T("nns should be in range of %d - %d.\n"), 16, 256);
    }
    if (pNnediParam->nnedi.nsize <= VPP_NNEDI_NSIZE_UNKNOWN || VPP_NNEDI_NSIZE_MAX <= pNnediParam->nnedi.nsize) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"nsize\": %d\n"), pNnediParam->nnedi.nsize);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.quality <= VPP_NNEDI_QUALITY_UNKNOWN || VPP_NNEDI_QUALITY_MAX <= pNnediParam->nnedi.quality) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"quality\": %d\n"), pNnediParam->nnedi.quality);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.pre_screen < VPP_NNEDI_PRE_SCREEN_NONE || VPP_NNEDI_PRE_SCREEN_MAX <= pNnediParam->nnedi.pre_screen) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"pre_screen\": %d\n"), pNnediParam->nnedi.pre_screen);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.precision < VPP_NNEDI_PRECISION_UNKNOWN || VPP_NNEDI_PRECISION_MAX <= pNnediParam->nnedi.precision) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"prec\": %d\n"), pNnediParam->nnedi.precision);
        return NV_ENC_ERR_INVALID_PARAM;
    }
#if !ENABLE_CUDA_FP16_HOST
    if (pNnediParam->nnedi.precision == VPP_NNEDI_PRECISION_FP16) {
        AddMessage(RGY_LOG_WARN, _T("prec=fp16 not compiled in this build, switching to fp32.\n"));
        pNnediParam->nnedi.precision = VPP_NNEDI_PRECISION_FP32;
    }
#endif
    return NV_ENC_SUCCESS;
}

std::vector<float> NVEncFilterNnedi::readWeights(const tstring& weightFile) {
    std::vector<float> weights;
    const uint32_t expectedFileSize = 13574928u;
    uint64_t weightFileSize = 0;
    if (weightFile.length() == 0) {
        HMODULE hModule = GetModuleHandle(NULL);
        HRSRC hResource = NULL;
        HGLOBAL hResourceData = NULL;
        const char *pDataPtr = NULL;
        if (NULL == hModule) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get module handle.\n"));
        } else if (NULL == (hResource = FindResource(hModule, _T("NNEDI_WEIGHTBIN"), _T("EXE_DATA")))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get resource handle for \"NNEDI_WEIGHTBIN\".\n"));
        } else if (NULL == (hResourceData = LoadResource(hModule, hResource))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to load resource \"NNEDI_WEIGHTBIN\".\n"));
        } else if (NULL == (pDataPtr = (const char *)LockResource(hResourceData))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to lock resource \"NNEDI_WEIGHTBIN\".\n"));
        } else if (expectedFileSize != (weightFileSize = SizeofResource(hModule, hResource))) {
            AddMessage(RGY_LOG_ERROR, _T("Weights data has unexpected size %u [expected: %u].\n"),
                weightFile.c_str(), (uint32_t)weightFileSize, expectedFileSize);
        } else {
            weights.resize(weightFileSize);
            memcpy(weights.data(), pDataPtr, weightFileSize);
        }
    } else {
        if (!PathFileExists(weightFile.c_str())) {
            AddMessage(RGY_LOG_ERROR, _T("weight file \"%s\" does not exist.\n"), weightFile.c_str());
        } else if (!rgy_get_filesize(weightFile.c_str(), &weightFileSize)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get filesize of weight file \"%s\".\n"), weightFile.c_str());
        } else if (weightFileSize != expectedFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("Weights file \"%s\" has unexpected file size %u [expected: %u].\n"),
                weightFile.c_str(), (uint32_t)weightFileSize, expectedFileSize);
        } else {
            weights.resize(weightFileSize);
            std::ifstream fin(weightFile, std::ios::in | std::ios::binary);
            if (!fin.good()) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open weights file \"%s\".\n"), weightFile.c_str());
            } else if (fin.read((char *)weights.data(), weightFileSize).gcount() != (int64_t)weightFileSize) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to read weights file \"%s\".\n"), weightFile.c_str());
            }
            fin.close();
        }
    }
    return std::move(weights);
}

NVENCSTATUS NVEncFilterNnedi::initParams(const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    std::vector<float> weights = readWeights(pNnediParam->nnedi.weightfile);
    if (weights.size() == 0) {
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.precision == VPP_NNEDI_PRECISION_AUTO) {
        pNnediParam->nnedi.precision =
#if ENABLE_CUDA_FP16_HOST
            ((pNnediParam->compute_cpability.first == 6 && pNnediParam->compute_cpability.second == 0)
                || pNnediParam->compute_cpability.first >= 7)
            ? VPP_NNEDI_PRECISION_FP16 : VPP_NNEDI_PRECISION_FP32;
#else
            VPP_NNEDI_PRECISION_FP32;
#endif
    }

    const int weight1size = pNnediParam->nnedi.nns * 2 * (sizeNX[pNnediParam->nnedi.nsize] * sizeNY[pNnediParam->nnedi.nsize] + 1);
    const int sizeofweight1 = (pNnediParam->nnedi.precision == VPP_NNEDI_PRECISION_FP32) ? 4 : 2;
    int weight1size_tsize = 0;
    int weight1size_offset = 0;
    for (int j = 0; j < (int)_countof(sizeNN); j++) {
        for (int i = 0; i < (int)_countof(sizeNX); i++) {
            if (i == pNnediParam->nnedi.nsize
                && j == get_cx_index(list_vpp_nnedi_nns, pNnediParam->nnedi.nns)) {
                weight1size_offset = weight1size_tsize;
            }
            weight1size_tsize += sizeNN[j] * (sizeNX[i] * sizeNY[i] + 1) * 4;
        }
    }

    std::vector<float> weight0f;
    std::array<std::vector<char>, 2> weight1;

    for (size_t i = 0; i < weight1.size(); i++) {
        weight1[i].resize(weight1size * sizeofweight1, 0);
    }

    if (pNnediParam->nnedi.pre_screen >= VPP_NNEDI_PRE_SCREEN_NEW) {
        auto index = [](int j, int k) {
            return ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
        };

        const auto ptr_w = weights.data() + weight0size + weight0sizenew * (pNnediParam->nnedi.pre_screen - VPP_NNEDI_PRE_SCREEN_NEW);
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 64; k++) {
                sum += ptr_w[index(j, k)];
            }
            avg[j] = sum * (1.0 / 64.0);
        }

        weight0f.resize(weight0sizenew);
#if 0
        for (int j = 0; j < 4; j++) {
            double mval = 0.0;
            for (int k = 0; k < 64; k++) {
                mval = std::max(mval, std::abs((ptr_w[index(j, k)] - avg[j]) * (1.0 / 127.5)));
            }
            const double scale = 32767.0 / mval;
            for (int k = 0; k < 64; k++) {
                m_weight0s[index(j, k)] = (int16_t)(((ptr_w[index(j, k)] - avg[j]) * (1.0 / 127.5)) * scale + 0.5);
            }
            weight0f.push_back(mval * (1.0 / 32767.0));
        }
        for (int i = 0; i < weight0sizenew - 4 * 64; i++) {
            weight0f[i+4] = ptr_w[i + 4 * 64];
        }
#else
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 64; k++) {
                //weight0f[index(j, k)] = (float)((ptr_w[index(j, k)] - avg[j]) * halfinv);
                weight0f[j*64+k] = (float)((ptr_w[index(j, k)] - avg[j]) * halfinv);
            }
        }
        for (int i = 0; i < 4; i++) {
            weight0f[4*64+i] = ptr_w[4*64+i];
        }
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                weight0f[4*65+j*4+k] = ptr_w[4*65+ j + k*4]; //転置
            }
        }
        for (int i = 0; i < 4; i++) {
            weight0f[4*65+4*4+i] = ptr_w[4*65+4*4+i];
        }
#endif
    } else {
        const auto ptr_w = weights.data();
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 48; k++) {
                sum += ptr_w[j * 48 + k];
            }
            avg[j] = sum * (1.0 / 48.0);
        }
        weight0f.resize(weight0size);
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 48; k++) {
                weight0f[j * 48 + k] = (float)((ptr_w[j * 48 + k] - avg[j]) * halfinv);
            }
        }
        for (int i = 4 * 48; i < weight0size; i++) {
            weight0f[i] = ptr_w[i];
        }
    }

    for (int i = 0; i < 2; i++) {
        const float *ptrW = weights.data() + weight0size + weight0sizenew * 3 + weight1size_tsize * pNnediParam->nnedi.errortype + weight1size_offset + i * weight1size;
        if (pNnediParam->nnedi.precision == VPP_NNEDI_PRECISION_FP32) {
            setWeight1<float>((float *)weight1[i].data(), ptrW, pNnediParam);
        } else {
#if ENABLE_CUDA_FP16_HOST
            setWeight1<__half>((__half *)weight1[i].data(), ptrW, pNnediParam);
#endif //#if ENABLE_CUDA_FP16_HOST
        }
    }
    m_weight0 = CUMemBuf(weight0f.size() * sizeof(weight0f[0]));
    m_weight0.alloc();
    cudaMemcpy(m_weight0.ptr, weight0f.data(), m_weight0.nSize, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < weight1.size(); i++) {
        m_weight1[i] = CUMemBuf(weight1[i].size());
        m_weight1[i].alloc();
        cudaMemcpy(m_weight1[i].ptr, weight1[i].data(), m_weight1[i].nSize, cudaMemcpyHostToDevice);
    }
    return NV_ENC_SUCCESS;
}

template<typename TypeWeight> TypeWeight toWeight(float f, float scale);
template<> float toWeight<float>(float f, float scale) { return f; }
#if ENABLE_CUDA_FP16_HOST
template<> __half toWeight<__half>(float f, float scale) { return __float2half_rn(f * scale); }
#endif

template<typename TypeWeight>
void NVEncFilterNnedi::setWeight1(TypeWeight *ptrDst, const float *ptrW, const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    const int sizeNXY = sizeNX[pNnediParam->nnedi.nsize] * sizeNY[pNnediParam->nnedi.nsize];

    std::vector<double> mean0(pNnediParam->nnedi.nns * 2, 0.0);
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        const float *ptr = ptrW + j * sizeNXY;
        mean0[j] = std::accumulate(ptr, ptr + sizeNXY, 0.0) / (double)sizeNXY;
    }

    const double inv_nns = 1.0 / (double)pNnediParam->nnedi.nns;
    std::vector<double> mean1(sizeNXY, 0.0);
    for (int j = 0; j < pNnediParam->nnedi.nns; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            mean1[k] += (ptrW[j * sizeNXY + k] - mean0[j]) * inv_nns;
        }
    }

    const float *ptr = ptrW + pNnediParam->nnedi.nns * 2 * sizeNXY;
    const double mean2 = std::accumulate(ptr, ptr + pNnediParam->nnedi.nns, 0.0) * inv_nns;

    vector<float> buf(pNnediParam->nnedi.nns * 2 * sizeNXY);
    float max0 = 0.0f, max1 = 0.0f;
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            buf[j * sizeNXY + k] = (float)(ptrW[j * sizeNXY + k] - mean0[j] - (j < pNnediParam->nnedi.nns ? mean1[k] : 0.0));
            if (j < pNnediParam->nnedi.nns) {
                max0 = std::max(max0, buf[j * sizeNXY + k]);
            } else {
                max1 = std::max(max1, buf[j * sizeNXY + k]);
            }
        }
        ptrDst[pNnediParam->nnedi.nns * 2 * sizeNXY + j] = toWeight<TypeWeight>(ptrW[pNnediParam->nnedi.nns * 2 * sizeNXY + j] - (float)(j < pNnediParam->nnedi.nns ? mean2 : 0.0), 1.0f);
    }
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        float scale = 1.0f; // std::min(1.0f, (j < pNnediParam->nnedi.nns) ? 4096.0f / max0 : 4096.0f / max1);
        for (int k = 0; k < sizeNXY; k++) {
            ptrDst[j * sizeNXY + k] = toWeight<TypeWeight>(buf[j * sizeNXY + k], scale);
        }
    }
    ptrDst[pNnediParam->nnedi.nns * 2 * (sizeNXY + 1) + 0] = toWeight<TypeWeight>(1.0f /*std::max(1.0f, max0 / 4096.0f)*/, 1.0f);
    ptrDst[pNnediParam->nnedi.nns * 2 * (sizeNXY + 1) + 1] = toWeight<TypeWeight>(1.0f /*std::max(1.0f, max1 / 4096.0f)*/, 1.0f);

#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    //最適化のため、本来の並びを変更する
    //[2][nns][nnxy] -> [nns/WEIGHT_LOOP][nnxy][WEIGHT_LOOP][2]
    vector<TypeWeight> tmp(pNnediParam->nnedi.nns * 2 * (sizeNXY + 1));
    memcpy(tmp.data(), ptrDst, sizeof(tmp[0]) * tmp.size());
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            const int j1 = j  / pNnediParam->nnedi.nns;
            const int j2 = j  % pNnediParam->nnedi.nns;
            const int j3 = j2 / WEIGHT_LOOP;
            const int w  = j2 % WEIGHT_LOOP;
            ptrDst[((j3 * sizeNXY + k) * WEIGHT_LOOP + w) * 2 + j1] = tmp[j * sizeNXY + k];
        }
    }
    ptrDst += pNnediParam->nnedi.nns * 2 * sizeNXY;
    auto tmp2 = tmp.data() + pNnediParam->nnedi.nns * 2 * sizeNXY;
    for (int j = 0; j < pNnediParam->nnedi.nns; j++) {
        ptrDst[j * 2 + 0] = tmp2[j];
        ptrDst[j * 2 + 1] = tmp2[pNnediParam->nnedi.nns + j];
    }
#endif
}

NVENCSTATUS NVEncFilterNnedi::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pNnediParam = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(pParam);
    if (!pNnediParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(pNnediParam)) != NV_ENC_SUCCESS) {
        return sts;
    }

    auto cudaerr = AllocFrameBuf(pNnediParam->frameOut, pNnediParam->nnedi.isbob() ? 2 : 1);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pNnediParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    auto pNnediParamPrev = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_pParam);
    if (!pNnediParamPrev
        || pNnediParamPrev->nnedi != pNnediParam->nnedi) {
        if ((sts = initParams(pNnediParam)) != NV_ENC_SUCCESS) {
            return sts;
        }
    }
    if (pNnediParam->nnedi.isbob()) {
        pParam->baseFps *= 2;
        m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
    }

    m_sFilterInfo = strsprintf(
        _T("nnedi: field %s, nns %d, nsize %s, quality %s, prec %s\n")
        _T("                       pre_screen %s, errortype %s, weight \"%s\""),
        get_cx_desc(list_vpp_nnedi_field, pNnediParam->nnedi.field),
        pNnediParam->nnedi.nns,
        get_cx_desc(list_vpp_nnedi_nsize, pNnediParam->nnedi.nsize),
        get_cx_desc(list_vpp_nnedi_quality, pNnediParam->nnedi.quality),
        get_cx_desc(list_vpp_nnedi_prec, pNnediParam->nnedi.precision),
        get_cx_desc(list_vpp_nnedi_pre_screen, pNnediParam->nnedi.pre_screen),
        get_cx_desc(list_vpp_nnedi_error_type, pNnediParam->nnedi.errortype),
        ((pNnediParam->nnedi.weightfile.length()) ? pNnediParam->nnedi.weightfile.c_str() : _T("internal")));

    //コピーを保存
    m_pParam = pNnediParam;
    return sts;
}

NVENCSTATUS NVEncFilterNnedi::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }
    auto pNnediParam = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_pParam);
    if (!pNnediParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
        if (pNnediParam->nnedi.isbob()) {
            pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            ppOutputFrames[1] = &pOutFrame->frame;
            ppOutputFrames[1]->picstruct = pInputFrame->picstruct;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
            *pOutputFrameNum = 2;
        }
    }

    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    NnediTargetField targetField = NNEDI_GEN_FIELD_UNKNOWN;
    if (   pNnediParam->nnedi.field == VPP_NNEDI_FIELD_USE_AUTO
        || pNnediParam->nnedi.field == VPP_NNEDI_FIELD_BOB_AUTO) {
        if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
            const auto inputFrameInfoEx = getFrameInfoExtra(pInputFrame);
            cudaMemcpy2DAsync(
                ppOutputFrames[0]->ptr,
                ppOutputFrames[0]->pitch,
                pInputFrame->ptr,
                pInputFrame->pitch,
                inputFrameInfoEx.width_byte,
                inputFrameInfoEx.height_total,
                memcpyKind
            );
            return NV_ENC_SUCCESS;
        } else if (pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_TFF) {
            targetField = NNEDI_GEN_FIELD_BOTTOM;
        } else if (pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_BFF) {
            targetField = NNEDI_GEN_FIELD_TOP;
        }
    } else if (pNnediParam->nnedi.field == VPP_NNEDI_FIELD_USE_TOP
        || pNnediParam->nnedi.field == VPP_NNEDI_FIELD_BOB_TOP_BOTTOM) {
        targetField = NNEDI_GEN_FIELD_BOTTOM;
    } else if (pNnediParam->nnedi.field == VPP_NNEDI_FIELD_USE_BOTTOM
        || pNnediParam->nnedi.field == VPP_NNEDI_FIELD_BOB_BOTTOM_TOP) {
        targetField = NNEDI_GEN_FIELD_TOP;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("Not implemented yet.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, uchar4, 8, float, float>)*> func_list_fp32 ={
        { RGY_CSP_YV12,      proc_frame<uint8_t,  uchar4,   8, float, float> },
        { RGY_CSP_YV12_16,   proc_frame<uint16_t, ushort4, 16, float, float> },
        { RGY_CSP_YUV444,    proc_frame<uint8_t,  uchar4,   8, float, float> },
        { RGY_CSP_YUV444_16, proc_frame<uint16_t, ushort4, 16, float, float> }
    };
#if ENABLE_CUDA_FP16_HOST
    static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, uchar4, 8, __half2, __half2>)*> func_list_fp16 ={
        { RGY_CSP_YV12,      proc_frame<uint8_t,  uchar4,   8, __half2, __half2> },
        { RGY_CSP_YV12_16,   proc_frame<uint16_t, ushort4, 16, __half2, __half2> },
        { RGY_CSP_YUV444,    proc_frame<uint8_t,  uchar4,   8, __half2, __half2> },
        { RGY_CSP_YUV444_16, proc_frame<uint16_t, ushort4, 16, __half2, __half2> }
    };
    const auto& func_list = (pNnediParam->nnedi.precision == VPP_NNEDI_PRECISION_FP32) ? func_list_fp32 : func_list_fp16;
#else
    const auto& func_list = func_list_fp32;
#endif
    if (func_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    func_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        pNnediParam, targetField,
        m_weight0.ptr,
        m_weight1[0].ptr,
        m_weight1[1].ptr,
        (cudaStream_t)0
        );
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at nnedi(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;

    if (pNnediParam->nnedi.isbob()) {
        targetField = (targetField == NNEDI_GEN_FIELD_BOTTOM) ? NNEDI_GEN_FIELD_TOP : NNEDI_GEN_FIELD_BOTTOM;
        func_list.at(pInputFrame->csp)(ppOutputFrames[1], pInputFrame,
            pNnediParam, targetField,
            m_weight0.ptr,
            m_weight1[0].ptr,
            m_weight1[1].ptr,
            (cudaStream_t)0
            );
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at nnedi(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
        ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration = (pInputFrame->duration + 1) / 2;
        ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
        ppOutputFrames[1]->duration = pInputFrame->duration - ppOutputFrames[0]->duration;
    }
    return sts;
}

void NVEncFilterNnedi::close() {
    m_pFrameBuf.clear();
}
