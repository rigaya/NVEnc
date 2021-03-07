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
#include "rgy_cuda_util_kernel.h"

extern "C" {
extern char _binary_resource_nnedi3_weights_bin_start[];
extern char _binary_resource_nnedi3_weights_bin_end[];
extern char _binary_resource_nnedi3_weights_bin_size[];
}

static const int NNEDI_BLOCK_X       = 32;
static const int NNEDI_BLOCK_Y       = 8;

static const int weight0size = 49 * 4 + 5 * 4 + 9 * 4;
static const int weight0sizenew = 4 * 65 + 4 * 5;

__device__ __inline__
static float exp_(float val) {
    return __expf(clamp(val, -80.0f, 80.0f));
}

#define ENABLE_CUDA_FP16_DEVICE (__CUDACC_VER_MAJOR__ >= 10 && __CUDA_ARCH__ >= 530)
#define ENABLE_CUDA_FP16_HOST   (__CUDACC_VER_MAJOR__ >= 10)

//dot_product1で重み(nns)方向のループアンロールを行う
//これにより、一度sharedメモリからレジスタにのせたpixel情報を使いまわすことができる
#define ENABLE_DP1_WEIGHT_LOOP_UNROLL 1

//ENABLE_DP1_WEIGHT_LOOP_UNROLLに対応して通常の重みの並び [nns*2][nnxy]を変更する
//並びは[nns/WEIGHT_LOOP][nnxy][WEIGHT_LOOP][2]
#define ENABLE_DP1_WEIGHT_ARRAY_OPT (1 && ENABLE_DP1_WEIGHT_LOOP_UNROLL)

//shuffle命令を使ったweight係数の分配により高速化する
#define ENABLE_DP1_SHUFFLE_OPT 1

#define SSRC(x,y) ((y)*(ssrc_dim)+(x))
#define SPIX(x,y) ((y)*(spix_dim)+(x))
#define SWHT_IDX(i,thIdWeight) ((thIdWeight)*sweight_dim+(i))

__device__ __inline__
float elliott(float val) {
    return val * __frcp_rn(1.0f + fabs(val));
}
#if ENABLE_CUDA_FP16_HOST
__device__ __inline__
__half2 __half2_abs(__half2 val) {
    __half2 h;
    __HALF2_TO_UI(h) = __HALF2_TO_UI(val) & 0x7fff7fffu;
    return h;
}

__device__ __inline__
__half2 elliott(__half2 val) {
#if ENABLE_CUDA_FP16_DEVICE
    return val * h2rcp(__float2half2_rn(1.0f) + __half2_abs(val));
#else
    return val; //dummy
#endif
}
#endif

template<int pix_x_per_thread, int thread_y_loop, bool load_for_interp, typename TypePixel, int bit_depth>
__device__ __inline__
void load_texSrc(float *const ptr_src, const int ssrc_dim, TypePixel *const ptr_pix, const int spix_dim, cudaTextureObject_t texSrc, const int nnx, const int nny, const int nnx_2_m1, const int nny_2, const int thIdX, const int thIdY) {
    for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny; y += NNEDI_BLOCK_Y) {
        for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
            const float px = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + thIdX + x - nnx_2_m1 + 0.5f;
            const float py = blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2 + 0.5f;
            const float value = (float)tex2D<float>(texSrc, px, py);
            ptr_src[SSRC(x + thIdX, y + thIdY)] = value * 256.0f; //floatのときはここで256倍して8bit相当に戻す
            if (load_for_interp && 0 <= thIdX + x - nnx_2_m1 && thIdX + x - nnx_2_m1 < spix_dim) {
                ptr_pix[SPIX(x + thIdX - nnx_2_m1, y + thIdY)] = (TypePixel)(value * (float)(1<<bit_depth) + 0.5f);
            }
        }
    }
}
#if ENABLE_CUDA_FP16_HOST
template<int pix_x_per_thread, int thread_y_loop, bool load_for_interp, typename TypePixel, int bit_depth>
__device__ __inline__
void load_texSrc(__half2 *const ptr_src, const int ssrc_dim, TypePixel *const ptr_pix, const int spix_dim, cudaTextureObject_t texSrc, const int nnx, const int nny, const int nnx_2_m1, const int nny_2, const int thIdX, const int thIdY) {
#if ENABLE_CUDA_FP16_DEVICE
    static_assert(pix_x_per_thread == 1 || pix_x_per_thread == 4, "pix_x_per_thread == 1 or 4");
    if (pix_x_per_thread == 1) {
        //sharedメモリ上に、以下のように重複配置する
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny; y += NNEDI_BLOCK_Y) {
            for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
                const float px = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX + x - nnx_2_m1 + 0.5f;
                const float py = blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2 + 0.5f;
                const float v0 = tex2D<float>(texSrc, px, py);
                const float v1 = tex2D<float>(texSrc, px+1.0f, py);
                ptr_src[SSRC(x + thIdX, y + thIdY)] = __floats2half2_rn(v0, v1); //half2のときはここでは256倍せず、0～1の範囲を使用する
                if (load_for_interp && 0 <= thIdX + x - nnx_2_m1 && thIdX + x - nnx_2_m1 < spix_dim) {
                    ptr_pix[SPIX(x + thIdX - nnx_2_m1, y + thIdY)] = (TypePixel)(v0 * (float)(1<<bit_depth) + 0.5f);
                }
            }
        }
    } else { //pix_x_per_thread == 4
        //sharedメモリ上に、以下のように配置する
        // | 0, 1 | 2, 3 | 4, 5 | ...
        for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny; y += NNEDI_BLOCK_Y) {
            for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
                const int load_x = (thIdX + x) * 2 - nnx_2_m1;
                const float px = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + load_x + 0.5f;
                const float py = blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2 + 0.5f;
                const float v0 = tex2D<float>(texSrc, px, py);
                const float v1 = tex2D<float>(texSrc, px+1.0f, py);
                ptr_src[SSRC(x + thIdX, y + thIdY)] = __floats2half2_rn(v0, v1); //half2のときはここでは256倍せず、0～1の範囲を使用する
                if (load_for_interp && 0 <= load_x && load_x < spix_dim) {
                    struct __align__(sizeof(TypePixel) * 2) TypePixel2 {
                        TypePixel x, y;
                    } p;
                    p.x = (TypePixel)(v0 * (float)(1<<bit_depth) + 0.5f);
                    p.y = (TypePixel)(v1 * (float)(1<<bit_depth) + 0.5f);
                    *(TypePixel2 *)&ptr_pix[SPIX(load_x, y + thIdY)] = p;
                }
            }
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

template<typename T> __device__ __inline__ T setval(float val);
template<> __device__ __inline__ float setval(float val) { return val; };
template<typename TypeCalc> __device__ __inline__ constexpr int kernel_comute_network1_calc_scale_step();
template<> __device__ __inline__ constexpr int kernel_comute_network1_calc_scale_step<float>() { return 1; };
template<int thread_y_loop, int nns>
__device__ __inline__ bool compute_kernel0_get_flag_original(const float ret[thread_y_loop][nns], int ithy) {
    return (fmaxf(ret[ithy][2], ret[ithy][3]) <= fmaxf(ret[ithy][0], ret[ithy][1]));
}
template<int thread_y_loop, int nns>
__device__ __inline__ void compute_kernel0_get_flags_new(bool flags[4], const float ret[thread_y_loop][nns], int ithy) {
    flags[0] = ret[ithy][0] > 0.0f;
    flags[1] = ret[ithy][1] > 0.0f;
    flags[2] = ret[ithy][2] > 0.0f;
    flags[3] = ret[ithy][3] > 0.0f;
}
#if ENABLE_CUDA_FP16_HOST
template<> __device__ __inline__ constexpr int kernel_comute_network1_calc_scale_step<__half2>() { return 2; };
template<> __device__ __inline__ __half setval(float val) { return __half(val); };
template<> __device__ __inline__ __half2 setval(float val) { return __float2half2_rn(val); }
__device__ __inline__ __half half_max(const __half& a, const __half& b) {
#if ENABLE_CUDA_FP16_DEVICE
    return a < b ? b : a;
#else
    return a; //dummy
#endif
}
template<int thread_y_loop, int nns>
__device__ __inline__ bool compute_kernel0_get_flag_original(const __half2 ret[thread_y_loop][nns], int ithy) {
#if ENABLE_CUDA_FP16_DEVICE
    //__hlaf2には重み方向に2つの値が入っている
    //やっていることはfloat版と同じ
    return (half_max(ret[ithy][1].x, ret[ithy][1].y) <= half_max(ret[ithy][0].x, ret[ithy][0].y));
#else
    return true; //dummy
#endif
}
template<int thread_y_loop, int nns>
__device__ __inline__ void compute_kernel0_get_flags_new(bool flags[4], const __half2 ret[thread_y_loop][nns], int ithy) {
#if ENABLE_CUDA_FP16_DEVICE
    flags[0] = ret[ithy][0].x > __half(0.0f);
    flags[1] = ret[ithy][0].y > __half(0.0f);
    flags[2] = ret[ithy][1].x > __half(0.0f);
    flags[3] = ret[ithy][1].y > __half(0.0f);
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<bool scale_dummy, bool src_is_frame, int thread_y_loop, int weight_loop, bool prescreen_new>
__device__ __inline__
void dot_product0(
    float sum[thread_y_loop][weight_loop],
    const float *const ptr_src, const int ssrc_dim,
    const float *const ptr_weight, const int sweight_dim,
    const float *__restrict__ weight_offset,
    const int nnx, const int nny, const int thIdX, const int thIdY,
    const int pix_x_per_thread,
    const float mstd[thread_y_loop][4]
) {
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = 0.0f;
        }
    }
    const auto *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const int src_index = (src_is_frame)
            //srcがフレームのキャッシュを指しているとき
            //通常、pix_x_per_thread=1なので、thIdXによって各スレッドが担当するpixelをロードする
            //pre_screen=newの時には各スレッドが4pixel担当するので、pix_x_per_threadが4になり、とびとびの値をロードする
            ? SSRC(thIdX * pix_x_per_thread, thIdY * thread_y_loop + y)
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合
            //担当pixelはstmp_dim(ssrc_dim)ごとに並んでいるので、x=0、y=担当行でロードする
            : SSRC(0, thIdY * thread_y_loop * NNEDI_BLOCK_X + thIdX);
        const auto *ptr_s = &ptr_src[src_index];

        for (int x = 0; x < nnx; x++, ptr_s++, ptr_w++) {
            float s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[(src_is_frame) ? (SSRC(0, ithy)) : (SSRC(0, ithy * NNEDI_BLOCK_X))];
            }
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                auto w0 = ptr_w[SWHT_IDX(0, i)];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum[ithy][i] += s0[ithy] * w0;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        const auto wo = weight_offset[0];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            const auto scale = setval<float>((scale_dummy) ? 1.0f : mstd[ithy][2]);
            sum[ithy][i] = sum[ithy][i] * scale + wo;
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
template<bool scale_dummy, bool src_is_frame, int thread_y_loop, int weight_loop, bool prescreen_new>
__device__ __inline__
void dot_product0(
    __half2 sum[thread_y_loop][weight_loop],
    const __half2 *const ptr_src, const int ssrc_dim,
    const __half2 *const ptr_weight, const int sweight_dim,
    const __half2 *__restrict__ weight_offset,
    const int nnx, const int nny, const int thIdX, const int thIdY,
    const int pix_x_per_thread,
    const float mstd[thread_y_loop][4]
) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = setval<__half2>(0.0f);
        }
    }
    const int pix_x_per_thread_for_half2 = (prescreen_new) ? 2 : 1;
    const int wstep = kernel_comute_network1_calc_scale_step<__half2>();
    const auto *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const int src_index = (src_is_frame)
            //srcがフレームのキャッシュを指しているとき
            //通常、pix_x_per_thread=1なので、thIdXによって各スレッドが担当するpixelをロードする
            //pre_screen=originalでは、重複配置をしているので、各スレッドは、__hlaf2ごとにロードすればよい
            //   th=0   th=1   th=2   th=3   th=4
            // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
            //pre_screen=newの時には各スレッドが4pixel担当するので、とびとびの値をロードする。
            //このとき、half2に2pixel分収まり、pre_screen=originalのときのように重複配置はしていないので、
            //pix_x_per_thread_for_half2=2をthIdXに積算する
            //   th=0          th=1          th=2
            // | 0, 1 | 2, 3 | 3, 4 | 5, 6 | 7, 8 |
            ? SSRC(thIdX * pix_x_per_thread_for_half2, thIdY * thread_y_loop + y)
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合
            //担当pixelはstmp_dim(ssrc_dim)ごとに並んでいるので、x=0、y=担当行でロードする
            : SSRC(0, thIdY * thread_y_loop * NNEDI_BLOCK_X + thIdX);
        const auto *ptr_s = &ptr_src[src_index];

        //src_is_frame = trueのとき
        //pre_screen=originalでは、重複配置をしているので、各スレッドは、2つおきに読む必要がある
        //  最初           次           その次
        //   ↓            ↓            ↓
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        //
        //pre_screen=newの時には重複配置ではないので、各スレッドはすぐ隣を読めばよい
        //  最初    次    その次
        //   ↓     ↓     ↓
        // | 0, 1 | 2, 3 | 3, 4 | 5, 6 | 7, 8 |
        const int sstep = ((src_is_frame && !prescreen_new) ? wstep : 1);

        for (int x = 0; x < nnx; x += wstep, ptr_s += sstep) {
            __half2 s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[(src_is_frame) ? (SSRC(0, ithy)) : (SSRC(0, ithy * NNEDI_BLOCK_X))];
            }

            //kernel_comute_network0ではhalf2の場合 nns= 4 / 2
            //なので、weight_loopが2より大きいとおかしなことになる
            static_assert(weight_loop <= 2, "weight_loop <= 2");

            //wに連続するweight_loop*2の値を読み込み、shuffleによりbroadcastする
            //重みは、nns方向にまず並んでいる
            //基本的には下記のような感じでロード
            //   <------   nns  ------>
            //   <-- half2-->
            //     w0    w1    w2   w3
            //  |0----|---->|1----|---->|   <<< x0にかかる重み
            //  |2----|---->|3----|---->|   <<< x1にかかる重み
            //     w4    w5    w6   w7
            __half2 w;
            if (thIdX < weight_loop*2) w = ptr_w[thIdX];
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                auto w0 = __shfl(w, i+0);           //x0にかかる重み
                auto w1 = __shfl(w, i+weight_loop); //x1にかかる重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    //nns方向の計算をhalf2内で同時に行っていくイメージ
                    sum[ithy][i] += __low2half2(s0[ithy]) * w0;  //x0 * (w0, w1)
                    sum[ithy][i] += __high2half2(s0[ithy]) * w1; //x1 * (w4, w5)
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        const auto wo = weight_offset[0];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //srcがフレームのキャッシュを指しているときは、
            //half2の場合、ロード時に256倍していないので、ここで256倍する
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合は必要ない
            //なお、ここで256倍しないと、後段のelliottが本来の値を返さない
            const auto scale = setval<__half2>((src_is_frame) ? 256.0f : 1.0f);
            sum[ithy][i] = sum[ithy][i] * scale + wo;
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<typename TypePixel, int bit_depth, typename TypeCalc, int thread_y_loop>
__device__ __inline__
static TypePixel interp_ret(const TypeCalc *const ptr_src, const int ssrc_dim,
    const bool flag, const int thIdX, const int thIdY, int ithy, const int nnx_2_m1, const int nny_2) {
    TypePixel val = prescreen_flag<TypePixel, bit_depth>();
    if (flag) {
        float tmp =
            (19.0f / 32.0f) * ((float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 1)] + (float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 2)])
            - (3.0f / 32.0f) * ((float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 0)] + (float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 3)]);
        val = (TypePixel)clamp(tmp + 0.5f, 0.0f, (1<<bit_depth)-1.0f);
    }
    return val;
}

template<typename TypePixel4, int bit_depth, typename TypeCalc, bool prescreen_new, int thread_y_loop, int weight_loop>
__global__ void kernel_compute_network0(
    uint8_t *__restrict__ pDst, //top field / bottom field は考慮済みとする
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    cudaTextureObject_t texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
    const TypeCalc *__restrict__ weight,
    const NnediTargetField targetField
    ) {
    const int wstep = kernel_comute_network1_calc_scale_step<TypeCalc>(); //half2なら2, floatなら1
    const int pix_x_per_thread = prescreen_new ? 4/*4ピクセル分一度に処理する*/ : 1;
    const int nnx = (prescreen_new) ? 16 : 12;
    const int nny = 4;
    const int nnxy = nnx * nny;
    const int nns = 4 / wstep; //half2の場合、nns方向を2つ格納できる
    const int thIdX      = threadIdx.x; //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = threadIdx.y; //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       =(blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX) * pix_x_per_thread;
    const int gIdY       =(blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * thread_y_loop; //フィールド単位
    const int stmp_dim = ((prescreen_new) ? 4 : 8) / wstep; //half2の場合、値を2つ格納できる
    const int ssrc_dim = (prescreen_new && wstep == 2 /*__half2使用*/)
        ? (NNEDI_BLOCK_X * pix_x_per_thread + nnx) / 2 //prescreen=new かつ __half2使用の時は、重複した配置を行わない
        : NNEDI_BLOCK_X * pix_x_per_thread + nnx; //floatの時のサイズ　また、__half2でもprescreen=originalの時は重複配置するので、floatと同じサイズ

    //sharedメモリのサイズと使途
    __shared__ char shared[
        ssrc_dim * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(TypeCalc) + //src 計算用
        NNEDI_BLOCK_X * NNEDI_BLOCK_Y * thread_y_loop * stmp_dim * sizeof(TypeCalc) + //tmp (計算結果の一時保管用)
        (NNEDI_BLOCK_X * pix_x_per_thread) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(decltype(TypePixel4::x)) //interp_retで補間に使うため
    ];
    TypeCalc *const ptr_src = (TypeCalc *)shared;

    TypeCalc *const ptr_temp = (TypeCalc *)((char *)ptr_src
        + (ssrc_dim * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0])));
#define STMP_IDX(i,x,y) ( ((y)*(NNEDI_BLOCK_X)+(x)) * stmp_dim + (i))

    //interp_ret()で補間を行う時に使用するデータ
    //16bit精度(int)の場合、fp16では精度が落ちる可能性があるため、ptr_srcとは別に保持することにした
    //interp_ret()では縦方向にしか補間しないので、ptr_srcのようにnnx分余分に読む必要はない
    //ここではsharedメモリ節約のため、floatではなく整数で保持する
    decltype(TypePixel4::x) *const ptr_pix = (decltype(TypePixel4::x) *)((char *)ptr_temp
        + NNEDI_BLOCK_X * NNEDI_BLOCK_Y * thread_y_loop * stmp_dim * sizeof(TypeCalc));
    const int spix_dim = NNEDI_BLOCK_X * pix_x_per_thread;

    //input(texture) -> shared, spix
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = (prescreen_new) ? 6 : 5;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc<pix_x_per_thread, thread_y_loop, true, decltype(TypePixel4::x), bit_depth>(ptr_src, ssrc_dim, ptr_pix, spix_dim, texSrc, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY);
    __syncthreads();

    float dummy[thread_y_loop][4];
    const int sweight_dim = (wstep == 1) ? nnxy : nnxy * weight_loop;
    if (!prescreen_new) {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0<true, true, thread_y_loop, weight_loop, prescreen_new>(sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+48*nns+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0<true, false, thread_y_loop, weight_loop, prescreen_new>(sum, ptr_temp, stmp_dim, weight+49*nns+iw*nns, /*sweight_dim=nnxy=*/4, weight+49*nns + 4*nns+iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    //half2なら、値を2つ格納できることに注意して、4/wstepとする
                    ptr_temp[STMP_IDX(4/wstep+iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        TypeCalc ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0<true, false, thread_y_loop, weight_loop, prescreen_new>(sum, ptr_temp, stmp_dim, weight + nns*49 + nns*5+stmp_dim*iw, /*sweight_dim=nnxy=*/8, weight + nns*49 + nns*5 + nns*8+iw, /*nnx=*/8, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ret[ithy][ithw+iw] = sum[ithy][ithw];
                }
            }
        }

        if (gIdX < dstWidth) {
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    const bool flag = compute_kernel0_get_flag_original<thread_y_loop, nns>(ret, ithy);
                    decltype(TypePixel4::x) *const ptr_dst = (decltype(TypePixel4::x) *)((uint8_t *)pDst + (gIdY + ithy) * dstPitch + gIdX * sizeof(TypePixel4::x));
                    //ptr_dst[0] = interp_ret<decltype(TypePixel4::x), bit_depth, TypeCalc, thread_y_loop>(ptr_src, ssrc_dim, flag, thIdX, thIdY, ithy, nnx_2_m1, nny_2);
                    ptr_dst[0] = interp_ret<decltype(TypePixel4::x), bit_depth, decltype(TypePixel4::x), thread_y_loop>(ptr_pix, spix_dim, flag, thIdX, thIdY, ithy, 0, nny_2);
                }
            }
        }
    } else {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0<true, true, thread_y_loop, weight_loop, prescreen_new>(sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+64*nns+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        __syncthreads();

        TypeCalc ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0<true, false, thread_y_loop, weight_loop, prescreen_new>(sum, ptr_temp, stmp_dim, weight+65*nns+iw*nns, /*sweight_dim=nnxy=*/4, weight+65*nns + 4*nns + iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
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
                    bool flags[4];
                    compute_kernel0_get_flags_new<thread_y_loop>(flags, ret, ithy);
                    TypePixel4 out;
                    //out.x = interp_ret<decltype(TypePixel4::x), bit_depth, TypeCalc, thread_y_loop>(ptr_src+0, ssrc_dim, flags[0], thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    //out.y = interp_ret<decltype(TypePixel4::x), bit_depth, TypeCalc, thread_y_loop>(ptr_src+1, ssrc_dim, flags[1], thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    //out.z = interp_ret<decltype(TypePixel4::x), bit_depth, TypeCalc, thread_y_loop>(ptr_src+2, ssrc_dim, flags[2], thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    //out.w = interp_ret<decltype(TypePixel4::x), bit_depth, TypeCalc, thread_y_loop>(ptr_src+3, ssrc_dim, flags[3], thIdX * pix_x_per_thread, thIdY, ithy, nnx_2_m1, nny_2);
                    out.x = interp_ret<decltype(TypePixel4::x), bit_depth, decltype(TypePixel4::x), thread_y_loop>(ptr_pix+0, spix_dim, flags[0], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.y = interp_ret<decltype(TypePixel4::x), bit_depth, decltype(TypePixel4::x), thread_y_loop>(ptr_pix+1, spix_dim, flags[1], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.z = interp_ret<decltype(TypePixel4::x), bit_depth, decltype(TypePixel4::x), thread_y_loop>(ptr_pix+2, spix_dim, flags[2], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.w = interp_ret<decltype(TypePixel4::x), bit_depth, decltype(TypePixel4::x), thread_y_loop>(ptr_pix+3, spix_dim, flags[3], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    ptr_dst[0] = out;
                }
            }
        }
    }
}

template<typename TypeCalc> __device__
    void kernel_comute_network1_calc_scale_get_sum_sumsq(float& sum, float& sumsq, TypeCalc tsum, TypeCalc tsumsq);
template<> __device__ __inline__
    void kernel_comute_network1_calc_scale_get_sum_sumsq<float>(float& sum, float& sumsq, float tsum, float tsumsq) {
    sum = tsum, sumsq = tsumsq;
}
#if ENABLE_CUDA_FP16_HOST
template<> __device__ __inline__
    void kernel_comute_network1_calc_scale_get_sum_sumsq<__half2>(float& sum, float& sumsq, __half2 tsum, __half2 tsumsq) {
    //half2では、textureからのロード時に256倍していない
    //ここで、256倍して、本来の値に戻す(ここで256倍しないと、後段のelliottが本来の値を返さない)
    //なお、textureからのロード時に256倍してしまうとtsumsqの計算がオーバーフローしてしまう
    sum = ((float)tsum.x + (float)tsum.y) * 256.0f;
    sumsq = ((float)tsumsq.x + (float)tsumsq.y) * 256.0f * 256.0f;
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<typename TypeCalc>
__device__ __inline__
void kernel_comute_network1_calc_scale(
    float mstd[][4],
    TypeCalc *__restrict__ const ptr_temp,
    const TypeCalc *__restrict__ const ptr_src, const int ssrc_dim,
    const int nnx, const int nny, const int nnxy,
    const int thIdX, const int thIdY,
    const int thread_y_loop) {
    const int step = kernel_comute_network1_calc_scale_step<TypeCalc>();
#define TMP_IDX(x,y,i) ((((i)*(nny + NNEDI_BLOCK_Y * thread_y_loop)+(y))*NNEDI_BLOCK_X)+(x))
    for (int y = 0; y + thIdY < nny + NNEDI_BLOCK_Y * thread_y_loop; y += NNEDI_BLOCK_Y) {
        TypeCalc sum = setval<TypeCalc>(0.0f), sumsq = setval<TypeCalc>(0.0f);
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
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        TypeCalc tsum = setval<TypeCalc>(0.0f), tsumsq = setval<TypeCalc>(0.0f);
        #pragma unroll
        for (int y = 0; y < nny; y++) {
            tsum   += ptr_temp[TMP_IDX(thIdX, thIdY*thread_y_loop+ithy+y, 0)];
            tsumsq += ptr_temp[TMP_IDX(thIdX, thIdY*thread_y_loop+ithy+y, 1)];
        }

        //half2使用時に並列で計算したものを集約するとともに、256倍の補正を適用する
        float sum, sumsq;
        kernel_comute_network1_calc_scale_get_sum_sumsq<TypeCalc>(sum, sumsq, tsum, tsumsq);

        mstd[ithy][3] = 0.0f;
        mstd[ithy][0] = sum * inv_nnxy;
        float tmp = sumsq * inv_nnxy - mstd[ithy][0] * mstd[ithy][0];
        if (tmp <= RGY_FLT_EPS) {
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
    float mstd[][4],
    __half2 *__restrict__ const ptr_temp,
    const __half2 *__restrict__ const ptr_src, const int ssrc_dim,
    const int nnx, const int nny, const int nnxy,
    const int thIdX, const int thIdY,
    const int thread_y_loop) {
    //ダミー
    assert(false);
}
#endif //#if ENABLE_CUDA_FP16_HOST && (!ENABLE_CUDA_FP16_DEVICE)

template<typename TypeCalc, int thread_y_loop, int weight_loop>
__device__ __inline__
void dot_product_frame1_fp32(
    float sum0[thread_y_loop][weight_loop], //レジスタにのることを期待する
    float sum1[thread_y_loop][weight_loop], //レジスタにのることを期待する
    TypeCalc *__restrict__ const ptr_src, const int ssrc_dim,
    const TypeCalc *__restrict__ const ptr_weight, const int sweight_dim,
    const TypeCalc *__restrict__ weight_offset,
    const int nnx, const int nny, const int nns, const int thIdX, const int thIdY,
    const float mstd[thread_y_loop][4]
) {
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum0[ithy][i] = sum1[ithy][i] = 0.0f;
        }
    }
    const TypeCalc *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const TypeCalc *ptr_s = &ptr_src[SSRC(thIdX, thIdY * thread_y_loop + y)];
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
        //#pragma unroll (4)
        for (int x = 0; x < nnx; x++, ptr_s++) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            TypeCalc s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
#if ENABLE_DP1_SHUFFLE_OPT
            //[nns/weight_loop][nnxy][weight_loop][2]
            //最後の2つには、nns方向の[i]と[i+nns]のものを配置している
            //   <---------------   nns  -------------------->
            //   <---  weight_loop  --->  (weight_loop = 2の場合)
            //    [0]  [nns]  [1] [1+nns]
            //  |0----|1--->|2----|3--->|
            //まず、各スレッドでweight_loop*2分だけ重みをwにロードし、
            //これをshuffleで全スレッドにbroadcastして使用するようにする
            TypeCalc w;
            if (thIdX < weight_loop*2) w = ptr_w[thIdX];
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                const auto w0 = __shfl(w, i*2+0); //[i]の重み
                const auto w1 = __shfl(w, i*2+1); //[i+nns]の重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum0[ithy][i] += s0[ithy] * w0;
                    sum1[ithy][i] += s0[ithy] * w1;
                }
            }
#else
            #pragma unroll
            for (int i = 0; i < weight_loop; i++, ptr_w += 2) {
                const auto w0 = ptr_w[0];
                const auto w1 = ptr_w[1];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
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
        TypePixel s0[thread_y_loop];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            s0[ithy] = ptr_s[SSRC(0, ithy*NNEDI_BLOCK_Y)];
        }
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            TypeCalc w0 = ptr_w[SWHT_IDX(0, i)];
            TypeCalc w1 = ptr_w[SWHT_IDX(0, i+nns)];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                sum0[i][ithy] += s0[ithy] * w0;
                sum1[i][ithy] += s0[ithy] * w1;
            }
        }
    }
#endif
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset += 2) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //weight offsetもw([i], [i+nns])の並びになっている
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0]; //w[i]用のweight_offset
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[1]; //w[i+nns]用のweight_offset
        }
    }
#else
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0];
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[nns];
        }
    }
#endif
}

#if ENABLE_CUDA_FP16_HOST
template<int thread_y_loop, int weight_loop>
__device__ __inline__
void dot_product_frame1_fp16(
    __half2 sum[thread_y_loop][weight_loop],
    __half2 *__restrict__ const ptr_src, const int ssrc_dim,
    const __half2 *__restrict__ const ptr_weight, const int sweight_dim,
    const __half2 *__restrict__ weight_offset,
    const int nnx, const int nny, const int nns, const int thIdX, const int thIdY,
    const __half2 weight_scale[thread_y_loop]
) {
#if ENABLE_CUDA_FP16_DEVICE
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = setval<__half2>(0.0f);
        }
    }
    const __half2 *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const __half2 *ptr_s = &ptr_src[SSRC(thIdX, thIdY * thread_y_loop + y)];

        //ptr_srcでは、重複配置をしているので、各スレッドは、2つおきに読む必要がある
        //  最初           次           その次
        //   ↓            ↓            ↓
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        for (int x = 0; x < nnx; x += 2, ptr_s += 2) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            __half2 s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
            //[nns/weight_loop][nnxy][weight_loop][2]
            //最後の2つには、nns方向の[i]と[i+nns]のものを配置しているので、これがセットでhalf2に乗る
            //   <---------------   nns  -------------------->
            //   <---  weight_loop  --->  (weight_loop = 2の場合)
            //   <-- half2-->
            //    [0]  [nns]  [1] [1+nns]
            //  |0----|---->|1----|---->|   <<< x0にかかる重み
            //  |2----|---->|3----|---->|   <<< x1にかかる重み
            //まず、各スレッドでweight_loop*2分だけ重みをwにロードし、
            //これをshuffleで全スレッドにbroadcastして使用するようにする
            __half2 w;
            if (thIdX < weight_loop*2) w = ptr_w[thIdX];
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                __half2 w0 = __shfl(w,            +i); //x0にかかる重み
                __half2 w1 = __shfl(w, weight_loop+i); //x1にかかる重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum[ithy][i] += __low2half2(s0[ithy]) * w0;   //x0 * w([i], [i+nns])
                    sum[ithy][i] += __high2half2(s0[ithy]) * w1;  //x1 * w([i], [i+nns])
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //weight offsetもw([i], [i+nns])の並びになっている
            sum[ithy][i] = sum[ithy][i] * weight_scale[ithy] + weight_offset[0];
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<int thread_y_loop, int weight_loop>
__device__ __inline__
void kernel_comute_network1_dot_product(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    float *const ptr_src, const int ssrc_dim,
    const float *const weight,
    float mstd[thread_y_loop][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        float sum0[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        dot_product0<false, true, thread_y_loop, weight_loop, false>(sum0, ptr_src, ssrc_dim, weight+ (iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        float sum1[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        dot_product0<false, true, thread_y_loop, weight_loop, false>(sum1, ptr_src, ssrc_dim, weight+ (nns+iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy+nns + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]);
                float ret1 = sum1[ithy][ithw];
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
template<int thread_y_loop, int weight_loop>
__device__ __inline__
void kernel_comute_network1_dot_product(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    __half2 *const ptr_src, const int ssrc_dim,
    const __half2 *const weight,
    float mstd[thread_y_loop][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    //未実装
    assert(false);
}
#endif //#if ENABLE_CUDA_FP16_HOST

template<int thread_y_loop, int weight_loop>
__device__ __inline__
void kernel_comute_network1_dot_product_opt(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    float *const ptr_src, const int ssrc_dim,
    const float *const weight,
    float mstd[thread_y_loop][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
    //ENABLE_DP1_WEIGHT_ARRAY_OPTが有効の場合、
    //[iw]と[iw+nns]の重みが隣り合って並んでいるので、sweight_dimは2倍
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        float sum0[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        float sum1[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        // 重み(nns)方向に、weight_loop分のdotproduct
        // sum0[i] <- iw     - iw+weight_loop
        // sum1[i] <- iw+nns - iw+weight_loop+nns
        dot_product_frame1_fp32<float, thread_y_loop, weight_loop>(
            sum0, sum1, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + (nns*2)*nnxy + iw*2, nnx, nny, nns, thIdX, thIdY, mstd);
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]); // iw     - iw+weight_loop     の計算結果
                float ret1 = sum1[ithy][ithw];       // iw+nns - iw+weight_loop+nns の計算結果
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
}

#if ENABLE_CUDA_FP16_HOST
template<int thread_y_loop, int weight_loop>
__device__ __inline__
void kernel_comute_network1_dot_product_opt(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    __half2 *const ptr_src, const int ssrc_dim,
    const __half2 *const weight,
    float mstd[thread_y_loop][4],
    const int nnx, const int nny, const int nnxy, const int nns,
    const int thIdX, const int thIdY) {
#if ENABLE_CUDA_FP16_DEVICE
    //[iw]と[iw+nns]の重みが隣り合って_half2に入るので、half2としてはnnxyのまま
    const int sweight_dim = nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        __half2 sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        // 重み(nns)方向に、weight_loop分のdotproduct
        //ひとつの__half2に[iw, iw+nns]の両方の内積の結果が入っている
        // sum0[i](iw, iw+nns)
        __half2 weight_scale[thread_y_loop];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            weight_scale[ithy] = __float2half2_rn(mstd[ithy][2]);
        }
        dot_product_frame1_fp16<thread_y_loop, weight_loop>(
            sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + nns*nnxy + iw, nnx, nny, nns, thIdX, thIdY, weight_scale);
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                //half2使用時には、オーバーフローを避けるため、textureからのロード時に256倍していないので、ここでfloatにしてから補正する
                float ret0 = exp_(__low2float(sum[ithy][ithw]) * 256.0f);
                float ret1 = __high2float(sum[ithy][ithw]) * 256.0f;
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * __frcp_rn(1.0f + fabs(ret1)));
            }
        }
    }
#endif //#if ENABLE_CUDA_FP16_DEVICE
}
#endif //#if ENABLE_CUDA_FP16_HOST


template<typename TypePixel, int bit_depth, typename TypeCalc, int nnx, int nny, int thread_y_loop, int weight_loop>
__global__ void kernel_comute_network1(
    uint8_t *__restrict__ pDst, //top field / bottom field は考慮済みとする
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    cudaTextureObject_t texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
    const TypeCalc *__restrict__ weight10,
    const TypeCalc *__restrict__ weight11,
    const int nns,  // len = nns*2
    const int quals,
    const NnediTargetField targetField,
    const VppNnediPreScreen prescreen
) {
    const int thIdX      = threadIdx.x; //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = threadIdx.y; //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       = blockIdx.x * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX;
    const int gIdY       =(blockIdx.y * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * thread_y_loop; //フィールド単位
    const int nnxy       = nnx * nny;

    //sharedメモリのサイズと使途
    //1.src: (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0])
    //2.tmp: (nny + NNEDI_BLOCK_Y * thread_y_loop) * NNEDI_BLOCK_X * 2 * sizeof(ptr_temp[0])
    alignas(128) extern __shared__ char shared[];
    TypeCalc *const ptr_src = (TypeCalc *)shared;
    const int ssrc_dim = NNEDI_BLOCK_X + nnx;

    //input(texture) -> shared
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = nnx / 2 - 1;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc<1, thread_y_loop, false, float/*実際には使わないのでなんでもいい*/, bit_depth>(
        ptr_src, ssrc_dim, nullptr, 0, texSrc, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY);
    __syncthreads();

    TypeCalc *const ptr_temp = (TypeCalc *)((char *)shared
        + (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0]));

    float mstd[thread_y_loop][4];
    kernel_comute_network1_calc_scale(mstd, ptr_temp, ptr_src, ssrc_dim, nnx, nny, nnxy, thIdX, thIdY, thread_y_loop);

    uint8_t *const ptr_dst_base = (uint8_t *)pDst + gIdY * dstPitch + gIdX * sizeof(TypePixel);
    uint32_t flag_sum = 0xffffffff; //処理するかどうかのフラグ
    if (((uint32_t)prescreen & (uint32_t)VPP_NNEDI_PRE_SCREEN_MODE) != 0) { //prescreenをやっていれば確認する
        flag_sum = 0x00;
        uint8_t *ptr_dst = ptr_dst_base;
        //自分のスレッドの担当するpixelについて調査する
        //処理対象となっていたらビットを立てる
        //thread_y_loopについて、下のビットから使っていく
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++, ptr_dst += dstPitch) {
            uint32_t flag = 0x00;
            if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                flag = (((TypePixel *)ptr_dst)[0] == prescreen_flag<TypePixel, bit_depth>()) ? 0x01 << ithy : 0x00;
            }
            flag_sum |= flag;
            //ビットを使い切らないようにチェック
            static_assert(thread_y_loop <= sizeof(flag_sum) * 8, "thread_y_loop <= sizeof(flag_sum) * 8");
        }
    }

/*
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

*/
    //weightの先頭のポインタ
    if (__any(flag_sum)) { //どのpixelも処理する必要がなければ、スキップする
        for (int iquality = 0; iquality < quals; iquality++) {
            const TypeCalc *const weight = (iquality) ? weight11 : weight10;
            float wsum[thread_y_loop], vsum[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                wsum[ithy] = vsum[ithy] = 0.0f;
            }
            if (ENABLE_DP1_WEIGHT_LOOP_UNROLL) {
                kernel_comute_network1_dot_product_opt<thread_y_loop, weight_loop>(
                    wsum, vsum, ptr_src, ssrc_dim, weight, mstd, nnx, nny, nnxy, nns, thIdX, thIdY);
            } else {
                kernel_comute_network1_dot_product<thread_y_loop, weight_loop>(
                    wsum, vsum, ptr_src, ssrc_dim, weight, mstd, nnx, nny, nnxy, nns, thIdX, thIdY);
            }

            const float min_weight_sum = 1e-10f;
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if (wsum[ithy] > min_weight_sum) {
                    mstd[ithy][3] += ((5.0f * vsum[ithy]) * __frcp_rn(wsum[ithy])) * mstd[ithy][1];
                }
                mstd[ithy][3] += mstd[ithy][0];
            }
        }

        if (gIdX < dstWidth) {
            const float scale = (1<<bit_depth) / 256.0f * ((quals > 1) ? 0.5f : 1.0f);
            uint8_t *ptr_dst = (uint8_t *)ptr_dst_base;
            for (int ithy = 0; ithy < thread_y_loop; ithy++, ptr_dst += dstPitch) {
                if ((((uint32_t)prescreen & (uint32_t)VPP_NNEDI_PRE_SCREEN_BLOCK) || (flag_sum & (1<<ithy))) //処理対象かチェック、ブロックモードなら常に処理する
                    && (gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    ((TypePixel *)ptr_dst)[0] = (TypePixel)clamp(mstd[ithy][3] * scale + 0.5f, 0.0f, (1<<bit_depth)-1.0f);
                }
            }
        }
    }
}

template<typename TypePixel>
cudaError_t setTexFieldNnedi(cudaTextureObject_t& texSrc, const FrameInfo *pFrame, const NnediTargetField targetField) {
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
    texDescSrc.addressMode[0]   = cudaAddressModeWrap;
    texDescSrc.addressMode[1]   = cudaAddressModeWrap;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename TypePixel4, int bit_depth, typename TypeCalc>
cudaError_t nnedi_compute_network_0(FrameInfo *pOutputPlane,
    cudaTextureObject_t texSrc,
    const TypeCalc *weight0,
    const VppNnediPreScreen pre_screen,
    const NnediTargetField targetField,
    cudaStream_t stream
) {
    dim3 blockSize(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);

    auto cudaerr = cudaSuccess;
    if ((pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) == VPP_NNEDI_PRE_SCREEN_ORIGINAL) {
        const int thread_y_loop_org = 2;
        dim3 gridSize(
            divCeil(pOutputPlane->width, blockSize.x),
            divCeil(pOutputPlane->height / 2, blockSize.y * thread_y_loop_org));
        kernel_compute_network0<TypePixel4, bit_depth, TypeCalc, false, thread_y_loop_org, 2><<<gridSize, blockSize, 0, stream>>>(
            (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch * 2, //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
            weight0, targetField);
        cudaerr = cudaGetLastError();
    } else if ((pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) {
        const int thread_y_loop_new = 2;
        dim3 gridSize(
            divCeil(pOutputPlane->width, blockSize.x * 4 /*4ピクセル分一度に処理する*/),
            divCeil(pOutputPlane->height / 2, blockSize.y * thread_y_loop_new));
        kernel_compute_network0<TypePixel4, bit_depth, TypeCalc, true, thread_y_loop_new, 2><<<gridSize, blockSize, 0, stream>>>(
            (uint8_t *)pOutputPlane->ptr + pOutputPlane->pitch * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch * 2, //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            texSrc, //有効フィールドのみのテクスチャ(縦解像度は半分)
            weight0, targetField);
        cudaerr = cudaGetLastError();
    } else {
        cudaerr = setPlaneFieldAsync(pOutputPlane, -1, targetField == NNEDI_GEN_FIELD_TOP /* 生成するほうのフィールドを選択 */, stream);
    }
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypePixel, int bit_depth, typename TypeCalc, int WEIGHT_LOOP_1>
cudaError_t nnedi_compute_network_1(
    FrameInfo *pOutputFrame,
    cudaTextureObject_t texSrc,
    const TypeCalc *weight10,
    const TypeCalc *weight11,
    const NnediTargetField targetField,
    const VppNnediNSize nsize,
    const int nns,
    const VppNnediQuality quality,
    const VppNnediPreScreen pre_screen,
    cudaStream_t stream
) {
    //スレッド内で複数の出力を同時に計算する
    static const int THREAD_Y_LOOP = 4;
    //重み(nns)方向のループアンロール数
    //やりすぎると使用レジスタ数が増え、かえって遅くなる
    static_assert(WEIGHT_LOOP_1 <= WARP_SIZE, "WEIGHT_LOOP < WARP_SIZE");

    dim3 blockSize(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);
    dim3 gridSize(
        divCeil(pOutputFrame->width, blockSize.x),
        divCeil(pOutputFrame->height / 2, blockSize.y * THREAD_Y_LOOP));

    const int nnx = NVEncFilterNnedi::sizeNX[nsize];
    const int nny = NVEncFilterNnedi::sizeNY[nsize];
    const int shared_mem_size =
        (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * sizeof(TypeCalc) + //src
        (NNEDI_BLOCK_Y * THREAD_Y_LOOP + nny) * NNEDI_BLOCK_X * 2 * sizeof(TypeCalc); //temp

    switch (nsize) {
    case VPP_NNEDI_NSIZE_8x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 8, 6, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_16x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 16, 6, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_32x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 32, 6, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_48x6:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 48, 6, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_8x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 8, 4, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_16x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 16, 4, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
        break;
    case VPP_NNEDI_NSIZE_32x4:
        kernel_comute_network1<TypePixel, bit_depth, TypeCalc, 32, 4, THREAD_Y_LOOP, WEIGHT_LOOP_1><<<gridSize, blockSize, shared_mem_size, stream>>>(
            (uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
            pOutputFrame->pitch * 2, //1行おきなので通常の2倍
            pOutputFrame->width,
            pOutputFrame->height,
            texSrc,
            weight10, weight11,
            nns, (int)quality, targetField, pre_screen);
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

template<typename TypePixel, typename TypePixel4, int bit_depth, typename TypeCalc, int WEIGHT_LOOP_1>
cudaError_t proc_plane(
    FrameInfo *pOutputPlane,
    const FrameInfo *pInputPlane,
    const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam,
    const NnediTargetField targetField,
    const TypeCalc *weight0,
    const TypeCalc *weight10,
    const TypeCalc *weight11,
    cudaStream_t stream
) {
    // 有効なほうのフィールドをコピー
    auto cudaerr = copyPlaneFieldAsync(pOutputPlane, pInputPlane, targetField != NNEDI_GEN_FIELD_TOP, targetField != NNEDI_GEN_FIELD_TOP, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    cudaTextureObject_t texSrc = 0;
    cudaerr = setTexFieldNnedi<TypePixel>(texSrc, pInputPlane, targetField);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = nnedi_compute_network_0<TypePixel4, bit_depth, TypeCalc>(pOutputPlane,
        texSrc,
        weight0,
        (pNnediParam->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE),
        targetField,
        stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    if (!(pNnediParam->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_ONLY)) {
        cudaerr = nnedi_compute_network_1<TypePixel, bit_depth, TypeCalc, WEIGHT_LOOP_1>(
            pOutputPlane,
            texSrc,
            weight10,
            weight11,
            targetField,
            pNnediParam->nnedi.nsize,
            pNnediParam->nnedi.nns,
            pNnediParam->nnedi.quality,
            (pNnediParam->nnedi.pre_screen & (VPP_NNEDI_PRE_SCREEN_MODE | VPP_NNEDI_PRE_SCREEN_BLOCK)),
            stream);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename TypePixel, typename TypePixel4, int bit_depth, typename TypeCalc, int WEIGHT_LOOP_1>
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

    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeCalc, WEIGHT_LOOP_1>(&planeOutputY, &planeInputY, pNnediParam, targetField, (const TypeCalc *)weight0, (const TypeCalc *)weight10, (const TypeCalc *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeCalc, WEIGHT_LOOP_1>(&planeOutputU, &planeInputU, pNnediParam, targetField, (const TypeCalc *)weight0, (const TypeCalc *)weight10, (const TypeCalc *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = proc_plane<TypePixel, TypePixel4, bit_depth, TypeCalc, WEIGHT_LOOP_1>(&planeOutputV, &planeInputV, pNnediParam, targetField, (const TypeCalc *)weight0, (const TypeCalc *)weight10, (const TypeCalc *)weight11, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

const int NVEncFilterNnedi::weight_loop_1 = 4;
const int NVEncFilterNnedi::sizeNX[] = { 8, 16, 32, 48, 8, 16, 32 };
const int NVEncFilterNnedi::sizeNY[] = { 6, 6, 6, 6, 4, 4, 4 };
const int NVEncFilterNnedi::sizeNN[] = { 16, 32, 64, 128, 256 };

NVEncFilterNnedi::NVEncFilterNnedi() : m_weight0(), m_weight1() {
    m_sFilterName = _T("nnedi");
}

NVEncFilterNnedi::~NVEncFilterNnedi() {
    close();
}

RGY_ERR NVEncFilterNnedi::checkParam(const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    if (pNnediParam->frameOut.height <= 0 || pNnediParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.field <= VPP_NNEDI_FIELD_UNKNOWN || VPP_NNEDI_FIELD_MAX <= pNnediParam->nnedi.field) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"field\": %d\n"), pNnediParam->nnedi.field);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.nns < 16 || 256 < pNnediParam->nnedi.nns) {
        pNnediParam->nnedi.nns = clamp(pNnediParam->nnedi.nns, 16, 256);
        AddMessage(RGY_LOG_WARN, _T("nns should be in range of %d - %d.\n"), 16, 256);
    }
    if (pNnediParam->nnedi.nsize <= VPP_NNEDI_NSIZE_UNKNOWN || VPP_NNEDI_NSIZE_MAX <= pNnediParam->nnedi.nsize) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"nsize\": %d\n"), pNnediParam->nnedi.nsize);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.quality <= VPP_NNEDI_QUALITY_UNKNOWN || VPP_NNEDI_QUALITY_MAX <= pNnediParam->nnedi.quality) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"quality\": %d\n"), pNnediParam->nnedi.quality);
        return RGY_ERR_INVALID_PARAM;
    }
    if (VPP_NNEDI_PRE_SCREEN_MAX <= pNnediParam->nnedi.pre_screen) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"pre_screen\": %d\n"), pNnediParam->nnedi.pre_screen);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.precision < VPP_FP_PRECISION_UNKNOWN || VPP_FP_PRECISION_MAX <= pNnediParam->nnedi.precision) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"prec\": %d\n"), pNnediParam->nnedi.precision);
        return RGY_ERR_INVALID_PARAM;
    }
#if !ENABLE_CUDA_FP16_HOST
    if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP16) {
        AddMessage(RGY_LOG_WARN, _T("prec=fp16 not compiled in this build, switching to fp32.\n"));
        pNnediParam->nnedi.precision = VPP_FP_PRECISION_FP32;
    }
#endif
    return RGY_ERR_NONE;
}

shared_ptr<const float> NVEncFilterNnedi::readWeights(const tstring& weightFile, HMODULE hModule) {
    shared_ptr<const float> weights;
    const uint32_t expectedFileSize = 13574928u;
    uint64_t weightFileSize = 0;
    if (weightFile.length() == 0) {
        //埋め込みデータを使用する
#if defined(_WIN32) || defined(_WIN64)
        if (hModule == NULL) {
            hModule = GetModuleHandle(NULL);
        }
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
            AddMessage(RGY_LOG_ERROR, _T("Weights data has unexpected size %lld [expected: %u].\n"),
                (long long int)weightFileSize, expectedFileSize);
        } else {
            weights = shared_ptr<const float>((const float *)pDataPtr, [](const float *x) { UNREFERENCED_PARAMETER(x); return; /*何もしない*/ });
        }
#else
        const char *pDataPtr = _binary_resource_nnedi3_weights_bin_start;
        weightFileSize = (size_t)(_binary_resource_nnedi3_weights_bin_end - _binary_resource_nnedi3_weights_bin_start);
        if (pDataPtr == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get Weights data.\n"));
        } else if (expectedFileSize != weightFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("Weights data has unexpected size %lld [expected: %u].\n"),
                (long long int)weightFileSize, expectedFileSize);
        } else {
            weights = shared_ptr<const float>((const float *)pDataPtr, [](const float *x) { UNREFERENCED_PARAMETER(x); return; /*何もしない*/ });
        }
#endif
    } else {
        if (!rgy_file_exists(weightFile)) {
            AddMessage(RGY_LOG_ERROR, _T("weight file \"%s\" does not exist.\n"), weightFile.c_str());
        } else if (!rgy_get_filesize(weightFile.c_str(), &weightFileSize)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get filesize of weight file \"%s\".\n"), weightFile.c_str());
        } else if (weightFileSize != expectedFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("Weights file \"%s\" has unexpected file size %lld [expected: %u].\n"),
                weightFile.c_str(), (long long int)weightFileSize, expectedFileSize);
        } else {
            std::ifstream fin(weightFile, std::ios::in | std::ios::binary);
            if (!fin.good()) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open weights file \"%s\".\n"), weightFile.c_str());
            } else {
                float *buffer = new float[weightFileSize / sizeof(float)];
                if (!buffer) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to allocate buffer memory for \"%s\".\n"), weightFile.c_str());
                } else {
                    weights = shared_ptr<float>(buffer, std::default_delete<float[]>());
                    if (fin.read((char *)weights.get(), weightFileSize).gcount() != (int64_t)weightFileSize) {
                        AddMessage(RGY_LOG_ERROR, _T("Failed to read weights file \"%s\".\n"), weightFile.c_str());
                        weights.reset();
                    }
                }
                fin.close();
            }
        }
    }
    return weights;
}

RGY_ERR NVEncFilterNnedi::initParams(const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    auto weights = readWeights(pNnediParam->nnedi.weightfile, pNnediParam->hModule);
    if (!weights) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_AUTO) {
        pNnediParam->nnedi.precision =
#if ENABLE_CUDA_FP16_HOST
            ((pNnediParam->compute_capability.first == 6 && pNnediParam->compute_capability.second == 0)
                || pNnediParam->compute_capability.first >= 7)
            ? VPP_FP_PRECISION_FP16 : VPP_FP_PRECISION_FP32;
#else
            VPP_FP_PRECISION_FP32;
#endif
    }

    const int weight1size = pNnediParam->nnedi.nns * 2 * (sizeNX[pNnediParam->nnedi.nsize] * sizeNY[pNnediParam->nnedi.nsize] + 1);
    const int sizeofweight = (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP32) ? 4 : 2;
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

    std::vector<char> weight0f;
    weight0f.resize((((pNnediParam->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) ? weight0sizenew : weight0size) * sizeofweight);
    if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP32) {
        setWeight0<float>((float *)weight0f.data(), weights.get(), pNnediParam);
    } else {
#if ENABLE_CUDA_FP16_HOST
        setWeight0<__half>((__half *)weight0f.data(), weights.get(), pNnediParam);
#endif //#if ENABLE_CUDA_FP16_HOST
    }

    std::array<std::vector<char>, 2> weight1;
    for (int i = 0; i < 2; i++) {
        weight1[i].resize(weight1size * sizeofweight, 0);
        const float *ptrW = weights.get() + weight0size + weight0sizenew * 3 + weight1size_tsize * pNnediParam->nnedi.errortype + weight1size_offset + i * weight1size;
        if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP32) {
            setWeight1<float>((float *)weight1[i].data(), ptrW, pNnediParam);
        } else {
#if ENABLE_CUDA_FP16_HOST
            setWeight1<__half>((__half *)weight1[i].data(), ptrW, pNnediParam);
#endif //#if ENABLE_CUDA_FP16_HOST
        }
    }
    m_weight0 = CUMemBuf(weight0f.size());
    m_weight0.alloc();
    cudaMemcpy(m_weight0.ptr, weight0f.data(), m_weight0.nSize, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < weight1.size(); i++) {
        m_weight1[i] = CUMemBuf(weight1[i].size());
        m_weight1[i].alloc();
        cudaMemcpy(m_weight1[i].ptr, weight1[i].data(), m_weight1[i].nSize, cudaMemcpyHostToDevice);
    }
    return RGY_ERR_NONE;
}

template<typename TypeCalc> TypeCalc toWeight(float f);
template<> float toWeight<float>(float f) { return f; }
#if ENABLE_CUDA_FP16_HOST
template<> __half toWeight<__half>(float f) { return __float2half_rn(f); }
#endif

template<typename TypeCalc>
void NVEncFilterNnedi::setWeight0(TypeCalc *ptrDst, const float *ptrW, const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
    if ((pNnediParam->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) {
        auto index = [](int j, int k) {
            return ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
        };

        const auto ptr_w = ptrW + weight0size + weight0sizenew * ((pNnediParam->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) - VPP_NNEDI_PRE_SCREEN_NEW);
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 64; k++) {
                sum += ptr_w[index(j, k)];
            }
            avg[j] = sum * (1.0 / 64.0);
        }
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 64; k++) {
                //ptrDst[index(j, k)] = (float)((ptr_w[index(j, k)] - avg[j]) * halfinv);
                ptrDst[j*64+k] = toWeight<TypeCalc>((float)((ptr_w[index(j, k)] - avg[j]) * halfinv));
            }
        }
        for (int i = 0; i < 4; i++) {
            ptrDst[4*64+i] = toWeight<TypeCalc>(ptr_w[4*64+i]);
        }
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                ptrDst[4*65+j*4+k] = toWeight<TypeCalc>(ptr_w[4*65+ j + k*4]); //転置
            }
        }
        for (int i = 0; i < 4; i++) {
            ptrDst[4*65+4*4+i] = toWeight<TypeCalc>(ptr_w[4*65+4*4+i]);
        }
        //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

        if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP16) {
            //並べ替え
            std::vector<TypeCalc> tmp(ptrDst, ptrDst + weight0sizenew);
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 64; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[(j2 * 64 + k) * 4 + j3] = tmp[j * 64 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[64*4 + j] = tmp[64*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[65*4 + (j2 * 4 + k) * 4 + j3] = tmp[65*4 + j * 4 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[65*4+4*4 + j] = tmp[65*4+4*4 + j];
            }
        }
    } else {
        const auto ptr_w = ptrW;
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 48; k++) {
                sum += ptr_w[j * 48 + k];
            }
            avg[j] = sum * (1.0 / 48.0);
        }
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 48; k++) {
                ptrDst[j * 48 + k] = toWeight<TypeCalc>((float)((ptr_w[j * 48 + k] - avg[j]) * halfinv));
            }
        }
        for (int i = 4 * 48; i < 4*49; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4 * 49; i < 4*49 + 4*4; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4 * 49 + 4*4; i < 4*49 + 4*5; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4*49 + 4*5; i < 4*49 + 4*5+ 4*8; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4*49 + 4*5+ 4*8; i < 4*49 + 4*5+ 4*9; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

        if (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP16) {
            //並べ替え
            std::vector<TypeCalc> tmp(ptrDst, ptrDst + weight0size);
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 48; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[(j2 * 48 + k) * 4 + j3] = tmp[j * 48 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[48*4 + j] = tmp[48*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[49*4+(j2 * 4 + k) * 4 + j3] = tmp[49*4+j * 4 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[49*4+4*4 + j] = tmp[49*4+4*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 8; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[49*4+5*4 + (j2 * 8 + k) * 4 + j3] = tmp[49*4+5*4 + j * 8 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[49*4+5*4+8*4 + j] = tmp[49*4+5*4+8*4 + j];
            }
        }
    }
}

template<typename TypeCalc>
void NVEncFilterNnedi::setWeight1(TypeCalc *ptrDst, const float *ptrW, const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam) {
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
        //fp16の場合、オーバーフローを避けるため途中まで0～1の範囲で計算するので、offsetの部分には1/256が必要
        float scale = (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP16) ? 1.0f / 256.0f : 1.0f;
        ptrDst[pNnediParam->nnedi.nns * 2 * sizeNXY + j] = toWeight<TypeCalc>((ptrW[pNnediParam->nnedi.nns * 2 * sizeNXY + j] - (float)(j < pNnediParam->nnedi.nns ? mean2 : 0.0)) * scale);
    }
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            ptrDst[j * sizeNXY + k] = toWeight<TypeCalc>(buf[j * sizeNXY + k]);
        }
    }
    //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    //最適化のため、本来の並びを変更する
    //[2][nns][nnxy] -> [nns/weight_loop_1][nnxy][weight_loop_1][2]
    vector<TypeCalc> tmp(pNnediParam->nnedi.nns * 2 * (sizeNXY + 1));
    memcpy(tmp.data(), ptrDst, sizeof(tmp[0]) * tmp.size());
    for (int j = 0; j < pNnediParam->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            const int j1 = j  / pNnediParam->nnedi.nns;
            const int j2 = j  % pNnediParam->nnedi.nns;
            const int j3 = j2 / weight_loop_1;
            const int w  = j2 % weight_loop_1;
            ptrDst[((j3 * sizeNXY + k) * weight_loop_1 + w) * 2 + j1] = tmp[j * sizeNXY + k];
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

RGY_ERR NVEncFilterNnedi::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pNnediParam = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(pParam);
    if (!pNnediParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(pNnediParam)) != RGY_ERR_NONE) {
        return sts;
    }

    auto cudaerr = AllocFrameBuf(pNnediParam->frameOut, pNnediParam->nnedi.isbob() ? 2 : 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pNnediParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    auto pNnediParamPrev = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_pParam);
    if (!pNnediParamPrev
        || pNnediParamPrev->nnedi != pNnediParam->nnedi) {
        if ((sts = initParams(pNnediParam)) != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (pNnediParam->nnedi.isbob()) {
        pParam->baseFps *= 2;
        m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
    }

    setFilterInfo(pParam->print());
    m_pParam = pNnediParam;
    return sts;
}

tstring NVEncFilterParamNnedi::print() const {
    return nnedi.print();
}

RGY_ERR NVEncFilterNnedi::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }
    auto pNnediParam = std::dynamic_pointer_cast<NVEncFilterParamNnedi>(m_pParam);
    if (!pNnediParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
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
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    NnediTargetField targetField = NNEDI_GEN_FIELD_UNKNOWN;
    if (   pNnediParam->nnedi.field == VPP_NNEDI_FIELD_USE_AUTO
        || pNnediParam->nnedi.field == VPP_NNEDI_FIELD_BOB_AUTO) {
        if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
            copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
            return RGY_ERR_NONE;
        } else if ((pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF) {
            targetField = NNEDI_GEN_FIELD_BOTTOM;
        } else if ((pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF) {
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
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, uchar4, 8, float, weight_loop_1>)*> func_list_fp32 ={
        { RGY_CSP_YV12,      proc_frame<uint8_t,  uchar4,   8, float, weight_loop_1> },
        { RGY_CSP_YV12_16,   proc_frame<uint16_t, ushort4, 16, float, weight_loop_1> },
        { RGY_CSP_YUV444,    proc_frame<uint8_t,  uchar4,   8, float, weight_loop_1> },
        { RGY_CSP_YUV444_16, proc_frame<uint16_t, ushort4, 16, float, weight_loop_1> }
    };
#if ENABLE_CUDA_FP16_HOST
    static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, uchar4, 8, __half2, weight_loop_1>)*> func_list_fp16 ={
        { RGY_CSP_YV12,      proc_frame<uint8_t,  uchar4,   8, __half2, weight_loop_1> },
        { RGY_CSP_YV12_16,   proc_frame<uint16_t, ushort4, 16, __half2, weight_loop_1> },
        { RGY_CSP_YUV444,    proc_frame<uint8_t,  uchar4,   8, __half2, weight_loop_1> },
        { RGY_CSP_YUV444_16, proc_frame<uint16_t, ushort4, 16, __half2, weight_loop_1> }
    };
    const auto& func_list = (pNnediParam->nnedi.precision == VPP_FP_PRECISION_FP32) ? func_list_fp32 : func_list_fp16;
#else
    const auto& func_list = func_list_fp32;
#endif
    if (func_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    func_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        pNnediParam, targetField,
        m_weight0.ptr,
        m_weight1[0].ptr,
        m_weight1[1].ptr,
        stream
        );
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at nnedi(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return RGY_ERR_CUDA;
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;

    if (pNnediParam->nnedi.isbob()) {
        targetField = (targetField == NNEDI_GEN_FIELD_BOTTOM) ? NNEDI_GEN_FIELD_TOP : NNEDI_GEN_FIELD_BOTTOM;
        func_list.at(pInputFrame->csp)(ppOutputFrames[1], pInputFrame,
            pNnediParam, targetField,
            m_weight0.ptr,
            m_weight1[0].ptr,
            m_weight1[1].ptr,
            stream
            );
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at nnedi(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration = (pInputFrame->duration + 1) / 2;
        ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
        ppOutputFrames[1]->duration = pInputFrame->duration - ppOutputFrames[0]->duration;
        ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
    }
    return sts;
}

void NVEncFilterNnedi::close() {
    m_pFrameBuf.clear();
}
