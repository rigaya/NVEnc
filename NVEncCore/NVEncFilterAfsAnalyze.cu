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
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterAfs.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

#define BLOCK_INT_X  (32) //blockDim(x) = スレッド数/ブロック
#define BLOCK_Y       (8) //blockDim(y) = スレッド数/ブロック
#define BLOCK_LOOP_Y (16) //ブロックのy方向反復数

#define SHARED_INT_X (BLOCK_INT_X) //sharedメモリの幅
#define SHARED_Y     (16) //sharedメモリの縦

#define PREFER_IMAGE  1


//      7       6         5        4        3        2        1       0
// | motion  |         non-shift        | motion  |          shift          |
// |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
const uint8_t motion_flag = 0x08u;
const uint8_t motion_shift = 0x80u;

const uint8_t non_shift_sign  = 0x40u;
const uint8_t non_shift_shift = 0x20u;
const uint8_t non_shift_deint = 0x10u;

const uint8_t shift_sign  = 0x04u;
const uint8_t shift_shift = 0x02u;
const uint8_t shift_deint = 0x01u;

typedef uint32_t Flags;

#define _PERM(d,c,b,a) (((d)<<12) | ((c)<<8) | ((b)<<4) | (a))
#define u8x4(x)  (uint32_t)(((uint32_t)(x)) | (((uint32_t)(x)) <<  8) | (((uint32_t)(x)) << 16) | (((uint32_t)(x)) << 24))
#define u16x2(x) (uint32_t)(((uint32_t)(x)) | (((uint32_t)(x)) << 16))

__inline__ __device__
Flags analyze_motion(uint32_t p0, uint32_t p1, uint32_t thre_motion4, uint32_t thre_shift4) {
    uint32_t abs = __vabsdiffu4(p0, p1);
    Flags mask_motion =  __vcmpgtu4(thre_motion4, abs) & u8x4(motion_flag);
    Flags mask_shift  =  __vcmpgtu4(thre_shift4,  abs) & u8x4(motion_shift);
    return mask_motion | mask_shift;
}

__inline__ __device__
Flags analyze_motion(uint2 p0, uint2 p1, uint32_t thre_motion2, uint32_t thre_shift2) {
    uint32_t abs0 = __vabsdiffu2(p0.x, p1.x);
    uint32_t abs1 = __vabsdiffu2(p0.y, p1.y);
    Flags mask_motion =  __byte_perm(__vcmpgtu2(thre_motion2, abs0), __vcmpgtu2(thre_motion2, abs1), _PERM(6,4,2,0)) & u8x4(motion_flag);
    Flags mask_shift  =  __byte_perm(__vcmpgtu2(thre_shift2,  abs0), __vcmpgtu2(thre_shift2,  abs1), _PERM(6,4,2,0)) & u8x4(motion_shift);

    return mask_motion | mask_shift;
}

__inline__ __device__
Flags analyze_motion(float p0, float p1, const float thre_motionf, const float thre_shiftf, int flag_offset) {
    const float abs = std::abs(p0 - p1);

    Flags mask_motion = (thre_motionf > abs) ? u8x4(motion_flag)  & (0x000000ffu << flag_offset) : 0u;
    Flags mask_shift  = (thre_shiftf  > abs) ? u8x4(motion_shift) & (0x000000ffu << flag_offset) : 0u;

    return mask_motion | mask_shift;
}

__inline__ __device__
Flags analyze_stripe(uint32_t p0, uint32_t p1, uint8_t flag_sign, uint8_t flag_deint, uint8_t flag_shift, const uint32_t thre_deint4, const uint32_t thre_shift4) {
    uint32_t abs     = __vabsdiffu4(p0, p1);
    Flags new_sign   = __vcmpgeu4(p0, p1) & u8x4(flag_sign);
    Flags mask_deint = __vcmpgtu4(abs, thre_deint4) & u8x4(flag_deint);
    Flags mask_shift = __vcmpgtu4(abs, thre_shift4) & u8x4(flag_shift);
    return new_sign | mask_deint | mask_shift;
}

__inline__ __device__
Flags analyze_stripe(uint2 p0, uint2 p1, uint8_t flag_sign, uint8_t flag_deint, const uint8_t flag_shift, const uint32_t thre_deint2, const uint32_t thre_shift2) {
    uint32_t abs0 = __vabsdiffu2(p0.x, p1.x);
    uint32_t abs1 = __vabsdiffu2(p0.y, p1.y);
    Flags new_sign   = __byte_perm(__vcmpgeu2(p0.x, p1.x), __vcmpgeu2(p0.y, p1.y), _PERM(6,4,2,0)) & u8x4(flag_sign);
    Flags mask_deint = __byte_perm(__vcmpgtu2(abs0, thre_deint2), __vcmpgtu2(abs1, thre_deint2), _PERM(6,4,2,0)) & u8x4(flag_deint);
    Flags mask_shift = __byte_perm(__vcmpgtu2(abs0, thre_shift2), __vcmpgtu2(abs1, thre_shift2), _PERM(6,4,2,0)) & u8x4(flag_shift);
    return new_sign | mask_deint | mask_shift;
}

__inline__ __device__
Flags analyze_stripe(float p0, float p1, uint8_t flag_sign, uint8_t flag_deint, uint8_t flag_shift, const float thre_deintf, const float thre_shiftf, int flag_offset) {
    const float abs0 = std::abs(p1 - p0);

    Flags new_sign   = (p0 >= p1) ? (uint32_t)flag_sign  << flag_offset : 0u;
    Flags mask_deint = (abs0 > thre_deintf) ? (uint32_t)flag_deint << flag_offset : 0u;
    Flags mask_shift = (abs0 > thre_shiftf) ? (uint32_t)flag_shift << flag_offset : 0u;

    return new_sign | mask_deint | mask_shift;
}

template<typename Type4, bool tb_order>
__inline__ __device__ Flags analyze_y(
    cudaTextureObject_t src_p0,
    cudaTextureObject_t src_p1,
    int ix, int iy,
    uint32_t thre_motion, uint32_t thre_deint, uint32_t thre_shift) {
    const float ifx = ix + 0.5f;
    const float ify = iy + 0.5f;

    //motion
    Type4 p0 = tex2D<Type4>(src_p0, ifx, ify);
    Type4 p1 = tex2D<Type4>(src_p1, ifx, ify);
    Type4 p2 = p1;
    Flags flag = analyze_motion(p0, p1, thre_motion, thre_shift);

    if (ify >= 1.0f) {
        //non-shift
        p1 = tex2D<Type4>(src_p0, ifx, ify-1.0f);
        flag |= analyze_stripe(p0, p1, non_shift_sign, non_shift_deint, non_shift_shift, thre_deint, thre_shift);

        //shift
        if (tb_order) {
            if (iy & 1) {
                p0 = p2;
            } else {
                p1 = tex2D<Type4>(src_p1, ifx, ify-1.0f);
            }
        } else {
            if (iy & 1) {
                p1 = tex2D<Type4>(src_p1, ifx, ify-1.0f);
            } else {
                p0 = p2;
            }
        }
        flag |= analyze_stripe(p1, p0, shift_sign, shift_deint, shift_shift, thre_deint, thre_shift);
    }
    return flag;
}

__inline__ __device__
float get_uv(cudaTextureObject_t src_p0_0, cudaTextureObject_t src_p0_1, float ifx, int iy) {
    //static const float WEIGHT[4] = {
    //    7.0f / 8.0f,
    //    5.0f / 8.0f,
    //    3.0f / 8.0f,
    //    1.0f / 8.0f
    //};
    //const float ifytex = ify + WEIGHT[iy & 3];
    const float ify = ((iy - 2) >> 2) + 0.5f;
    const float ifytex = ify + (3.5f - (float)(iy & 3)) * 0.25f;
    return tex2D<float>((iy & 1) ? src_p0_1 : src_p0_0, ifx, ifytex);
}

template<typename Type4, bool tb_order>
__inline__ __device__ Flags analyze_c(
    cudaTextureObject_t src_p0_0,
    cudaTextureObject_t src_p0_1,
    cudaTextureObject_t src_p1_0,
    cudaTextureObject_t src_p1_1,
    int ix, int iy,
    const float thre_motionf, const float thre_deintf, const float thre_shiftf) {
    Flags flag4 = 0;
    float ifx = (ix << 1) + 0.5f;

    #pragma unroll
    for (int i = 0; i < 4; i++, ifx += 0.5f) {
        //motion
        float p0 = get_uv(src_p0_0, src_p0_1, ifx, iy);
        float p1 = get_uv(src_p1_0, src_p1_1, ifx, iy);
        float p2 = p1;
        Flags flag = analyze_motion(p0, p1, thre_motionf, thre_shiftf, i * 8);

        if (iy > 0) {
            //non-shift
            p1 = get_uv(src_p0_0, src_p0_1, ifx, iy-1);
            flag |= analyze_stripe(p0, p1, non_shift_sign, non_shift_deint, non_shift_shift, thre_deintf, thre_shiftf, i * 8);

            //shift
            if (tb_order) {
                if (iy & 1) {
                    p0 = p2;
                } else {
                    p1 = get_uv(src_p1_0, src_p1_1, ifx, iy-1);
                }
            } else {
                if (iy & 1) {
                    p1 = get_uv(src_p1_0, src_p1_1, ifx, iy-1);
                } else {
                    p0 = p2;
                }
            }
            flag |= analyze_stripe(p1, p0, shift_sign, shift_deint, shift_shift, thre_deintf, thre_shiftf, i * 8);
        }
        flag4 |= flag;
    }
    return flag4;
}

__inline__ __device__
int shared_int_idx(int x, int y, int dep) {
    return dep * SHARED_INT_X * SHARED_Y + (y&15) * SHARED_INT_X + x;
}

__inline__ __device__
void count_flags_skip(Flags dat0, Flags dat1, Flags& count_deint, Flags& count_shift) {
    Flags deint, shift, mask;
    mask = (dat0 ^ dat1) & u8x4(non_shift_sign  | shift_sign);
    deint = dat0         & u8x4(non_shift_deint | shift_deint);
    shift = dat0         & u8x4(non_shift_shift | shift_shift);
    mask >>= 1;
    //最初はshiftの位置にしかビットはたっていない
    //deintにマスクはいらない
    //count_deint &= mask;
    count_shift &= mask;
    count_deint  = deint; //deintに値は入っていないので代入でよい
    count_shift += shift;
}

__inline__ __device__
void count_flags(Flags dat0, Flags dat1, Flags& count_deint, Flags& count_shift) {
    Flags deint, shift, mask;
    mask = (dat0 ^ dat1) & u8x4(non_shift_sign  | shift_sign);
    deint = dat0         & u8x4(non_shift_deint | shift_deint);
    shift = dat0         & u8x4(non_shift_shift | shift_shift);
    mask |= (mask << 1);
    mask |= (mask >> 2);
    count_deint &= mask;
    count_shift &= mask;
    count_deint += deint;
    count_shift += shift;
}

__inline__ __device__
Flags generate_flags(int ly, int idepth, uint32_t *__restrict__ ptr_shared) {
    Flags count_deint = 0;
    Flags dat0, dat1;

    //sharedメモリはあらかじめ-4もデータを作ってあるので、問題なく使用可能
    dat1 = ptr_shared[shared_int_idx(0, ly-3, idepth)];
    Flags count_shift = dat1 & u8x4(non_shift_shift | shift_shift);

    dat0 = ptr_shared[shared_int_idx(0, ly-2, idepth)];
    count_flags_skip(dat0, dat1, count_deint, count_shift);

    dat1 = ptr_shared[shared_int_idx(0, ly-1, idepth)];
    count_flags(dat1, dat0, count_deint, count_shift);

    dat0 = ptr_shared[shared_int_idx(0, ly+0, idepth)];
    count_flags(dat0, dat1, count_deint, count_shift);

    //      7       6         5        4        3        2        1       0
    // | motion  |         non-shift        | motion  |          shift          |
    // |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
    //motion 0x8888 -> 0x4444 とするため右シフト
    Flags flag0 = (dat0 & u8x4(motion_flag | motion_shift)) >> 1; //motion flag / motion shift

    //nonshift deint - countbit:654 / setbit 0x01
    //if ((count_deint & (0x70u << 0)) > (2u<<(4+ 0))) flag0 |= 0x01u<< 0; //nonshift deint(0)
    //if ((count_deint & (0x70u << 8)) > (2u<<(4+ 8))) flag1 |= 0x01u<< 8; //nonshift deint(1)
    //if ((count_deint & (0x70u <<16)) > (2u<<(4+16))) flag0 |= 0x01u<<16; //nonshift deint(2)
    //if ((count_deint & (0x70u <<24)) > (2u<<(4+24))) flag1 |= 0x01u<<24; //nonshift deint(3)
    Flags flag1 = __vcmpgtu4(count_deint & u8x4(0x70u), u8x4(2u << 4)) & u8x4(0x01u); //nonshift deint
    //nonshift shift - countbit:765 / setbit 0x10
    //if ((count_shift & (0xE0u << 0)) > (3u<<(5+ 0))) flag0 |= 0x01u<< 4; //nonshift shift(0)
    //if ((count_shift & (0xE0u << 8)) > (3u<<(5+ 8))) flag1 |= 0x01u<<12; //nonshift shift(1)
    //if ((count_shift & (0xE0u <<16)) > (3u<<(5+16))) flag0 |= 0x01u<<20; //nonshift shift(2)
    //if ((count_shift & (0xE0u <<24)) > (3u<<(5+24))) flag1 |= 0x01u<<28; //nonshift shift(3)
    Flags flag2 = __vcmpgtu4(count_shift & u8x4(0xE0u), u8x4(3u << 5)) & u8x4(0x10u); //nonshift shift
    //shift deint - countbit:210 / setbit 0x02
    //if ((count_deint & (0x07u << 0)) > (2u<<(0+ 0))) flag0 |= 0x01u<< 1; //shift deint(0)
    //if ((count_deint & (0x07u << 8)) > (2u<<(0+ 8))) flag1 |= 0x01u<< 9; //shift deint(1)
    //if ((count_deint & (0x07u <<16)) > (2u<<(0+16))) flag0 |= 0x01u<<17; //shift deint(2)
    //if ((count_deint & (0x07u <<24)) > (2u<<(0+24))) flag1 |= 0x01u<<25; //shift deint(3)
    Flags flag3 = __vcmpgtu4(count_deint & u8x4(0x07u), u8x4(2u << 0)) & u8x4(0x02u); //shift deint
    //shift shift - countbit:321 / setbit 0x20
    //if ((count_shift & (0x0Eu << 0)) > (3u<<(1+ 0))) flag0 |= 0x01u<< 5; //shift shift(0)
    //if ((count_shift & (0x0Eu << 8)) > (3u<<(1+ 8))) flag1 |= 0x01u<<13; //shift shift(1)
    //if ((count_shift & (0x0Eu <<16)) > (3u<<(1+16))) flag0 |= 0x01u<<21; //shift shift(2)
    //if ((count_shift & (0x0Eu <<24)) > (3u<<(1+24))) flag1 |= 0x01u<<29; //shift shift(3)
    Flags flag4 = __vcmpgtu4(count_shift & u8x4(0x0Eu), u8x4(3u << 1)) & u8x4(0x20u); //shift shift

    return flag0 | flag1 | flag2 | flag3 | flag4;
}

__inline__ __device__
void merge_mask(Flags masky, Flags masku, Flags maskv, Flags& mask0, Flags& mask1) {
    mask0 = masky & masku & maskv;
    mask1 = masky | masku | maskv;

    mask0 &= u8x4(0xcc); //motion
    mask1 &= u8x4(0x33); //shift/deint

    mask0 |= mask1;
}

template<typename Type, typename Type4, bool tb_order, bool yuv420>
__global__ void kernel_afs_analyze_12(
    uint32_t *__restrict__ ptr_dst,
    int *__restrict__ ptr_count,
    cudaTextureObject_t src_p0y,
    cudaTextureObject_t src_p0u0,
    cudaTextureObject_t src_p0u1, //yuv444では使用されない
    cudaTextureObject_t src_p0v0,
    cudaTextureObject_t src_p0v1, //yuv444では使用されない
    cudaTextureObject_t src_p1y,
    cudaTextureObject_t src_p1u0,
    cudaTextureObject_t src_p1u1, //yuv444では使用されない
    cudaTextureObject_t src_p1v0,
    cudaTextureObject_t src_p1v1, //yuv444では使用されない
    const int width_int, const int si_pitch_int, const int h,
    const uint32_t thre_Ymotion, const uint32_t thre_deint, const uint32_t thre_shift,
    const uint32_t thre_Cmotion, const float thre_Cmotionf, const float thre_deintf, const float thre_shiftf,
    const uint32_t scan_left, const uint32_t scan_top, const uint32_t scan_width, const uint32_t scan_height) {

    __shared__ uint32_t shared[SHARED_INT_X * SHARED_Y * 5]; //int単位でアクセスする
    const int lx = threadIdx.x; //スレッド数=BLOCK_INT_X
    int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * BLOCK_INT_X /*blockDim.x*/ + threadIdx.x;
    int imgy = (gidy * BLOCK_LOOP_Y * BLOCK_Y + ly);
    const int imgy_block_fin = min(h, ((gidy + 1) * BLOCK_LOOP_Y) * BLOCK_Y);
    uint32_t motion_count = 0;

#define CALL_ANALYZE_Y(p0, p1, y_offset) analyze_y<Type4, tb_order>((p0), (p1), (imgx), (imgy+(y_offset)), thre_Ymotion,  thre_deint,  thre_shift)
#define CALL_ANALYZE_C(p0_0, p0_1, p1_0, p1_1, y_offset) \
    (yuv420) ? analyze_c<Type4, tb_order>((p0_0), (p0_1), (p1_0), (p1_1), (imgx), (imgy+(y_offset)), thre_Cmotionf, thre_deintf, thre_shiftf) \
             : analyze_y<Type4, tb_order>((p0_0), (p1_0), (imgx), (imgy+(y_offset)), thre_Cmotion, thre_deint, thre_shift)

    uint32_t *ptr_shared = shared + shared_int_idx(lx,0,0);
    ptr_dst += (imgy-4) * si_pitch_int + imgx;

    //前の4ライン分、計算しておく
    //sharedの SHARED_Y-4 ～ SHARED_Y-1 を埋める
    if (ly < 4) {
        //正方向に4行先読みする
        ptr_shared[shared_int_idx(0, ly, 0)] = CALL_ANALYZE_Y(src_p0y, src_p1y, 0);
        ptr_shared[shared_int_idx(0, ly, 1)] = CALL_ANALYZE_C(src_p0u0, src_p0u1, src_p1u0, src_p1u1, 0);
        ptr_shared[shared_int_idx(0, ly, 2)] = CALL_ANALYZE_C(src_p0v0, src_p0v1, src_p1v0, src_p1v1, 0);
    }

    for (int iloop = 0; iloop <= BLOCK_LOOP_Y; iloop++,
        ptr_dst += BLOCK_Y * si_pitch_int, imgy += BLOCK_Y,
        ly += BLOCK_Y
    ) {
        { //差分情報を計算
            ptr_shared[shared_int_idx(0, ly+4, 0)] = CALL_ANALYZE_Y(src_p0y, src_p1y, 4);
            ptr_shared[shared_int_idx(0, ly+4, 1)] = CALL_ANALYZE_C(src_p0u0, src_p0u1, src_p1u0, src_p1u1, 4);
            ptr_shared[shared_int_idx(0, ly+4, 2)] = CALL_ANALYZE_C(src_p0v0, src_p0v1, src_p1v0, src_p1v1, 4);
            __syncthreads();
        }
        Flags mask1;
        { //マスク生成
            Flags masky = generate_flags(ly, 0, ptr_shared);
            Flags masku = generate_flags(ly, 1, ptr_shared);
            Flags maskv = generate_flags(ly, 2, ptr_shared);
            Flags mask0;
            merge_mask(masky, masku, maskv, mask0, mask1);
            ptr_shared[shared_int_idx(0, ly, 3)] = mask0;
            __syncthreads();
        }
        { //最終出力
            //ly+4とか使っているので準備ができてないうちから、次の列のデータを使うことになってまずい
            Flags mask4, mask5, mask6, mask7;
            mask4 = ptr_shared[shared_int_idx(0, ly-1, 3)];
            mask5 = ptr_shared[shared_int_idx(0, ly-2, 3)];
            mask6 = ptr_shared[shared_int_idx(0, ly-3, 3)];
            mask7 = ptr_shared[shared_int_idx(0, ly-4, 3)];
            mask1 &= u8x4(0x30);
            mask4 |= mask5 | mask6;
            mask4 &= u8x4(0x33);
            mask1 |= mask4 | mask7;
            if (imgx < width_int && (imgy - 4) < imgy_block_fin && ly - 4 >= 0) {
                //motion_countの実行
                if ((((uint32_t)imgx - scan_left) < scan_width) && (((uint32_t)(imgy - 4) - scan_top) < scan_height)) {
                    motion_count += __popc((~mask1) & u8x4(0x40)); //opencl版を変更、xorしてからマスク
                }
                //判定結果の出力
                ptr_dst[0] = mask1;
            }
            //次に書き換えるのは(x,y,0)(x,y,1)(x,y,2)なので、(x,y,3)の読み込みと同時に行うことができる
            //ここでの同期は不要
            //__syncthreads()
        }
    }


    //motion countの総和演算
    // 32               16              0
    //  |  count_latter ||  count_first |
    int motion_count_01;
    static_assert(BLOCK_INT_X * sizeof(int) * BLOCK_Y * BLOCK_LOOP_Y < (1<<(sizeof(short)*8-1)), "reduce block size for proper reduction in 16bit.");
    if (tb_order) {
        motion_count_01 = (int)(( ly      & 1) ? (uint32_t)motion_count << 16 : (uint32_t)motion_count);
    } else {
        motion_count_01 = (int)(((ly + 1) & 1) ? (uint32_t)motion_count << 16 : (uint32_t)motion_count);
    }

    motion_count_01 = block_sum<decltype(motion_count_01), BLOCK_INT_X, BLOCK_Y>(motion_count_01, (int *)shared);

    const int lid = threadIdx.y * BLOCK_INT_X + threadIdx.x;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        ptr_count[gid] = motion_count_01;
    }
}

template<typename Type>
cudaError_t textureCreateAnalyze(cudaTextureObject_t& tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = ptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = filterMode;
    texDesc.readMode         = readMode;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

template<typename Type, typename Type4, int bit_depth, bool tb_order, bool yuv420>
cudaError_t run_analyze_stripe(uint8_t *dst, const int dstPitch,
    const FrameInfo *pFrame0, const FrameInfo *pFrame1,
    CUMemBufPair *count_motion,
    const VppAfs *pAfsPrm, cudaStream_t stream) {
    auto cudaerr = cudaSuccess;
    const auto p0Y = getPlane(pFrame0, RGY_PLANE_Y);
    const auto p0U = getPlane(pFrame0, RGY_PLANE_U);
    const auto p0V = getPlane(pFrame0, RGY_PLANE_V);
    const auto p1Y = getPlane(pFrame1, RGY_PLANE_Y);
    const auto p1U = getPlane(pFrame1, RGY_PLANE_U);
    const auto p1V = getPlane(pFrame1, RGY_PLANE_V);

    if (   p0Y.width != p1Y.width || p0Y.height != p1Y.height
        || p0U.width != p1U.width || p0U.height != p1U.height
        || p0V.width != p1V.width || p0V.height != p1V.height) {
        return cudaErrorUnknown;
    }

    cudaTextureObject_t texP0Y = 0;
    cudaTextureObject_t texP1Y = 0;
    if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP0Y, cudaFilterModePoint, cudaReadModeElementType, p0Y.ptr, p0Y.pitch, (p0Y.width + 3) / 4, p0Y.height))) return cudaerr;
    if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP1Y, cudaFilterModePoint, cudaReadModeElementType, p1Y.ptr, p1Y.pitch, (p1Y.width + 3) / 4, p1Y.height))) return cudaerr;

    cudaTextureObject_t texP0U0 = 0;
    cudaTextureObject_t texP0U1 = 0; //yuv444では使用されない
    cudaTextureObject_t texP0V0 = 0;
    cudaTextureObject_t texP0V1 = 0; //yuv444では使用されない
    cudaTextureObject_t texP1U0 = 0;
    cudaTextureObject_t texP1U1 = 0; //yuv444では使用されない
    cudaTextureObject_t texP1V0 = 0;
    cudaTextureObject_t texP1V1 = 0; //yuv444では使用されない
    if (yuv420) {
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP0U0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0U.ptr + p0U.pitch * 0, p0U.pitch * 2, p0U.width, p0U.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP0U1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0U.ptr + p0U.pitch * 1, p0U.pitch * 2, p0U.width, p0U.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP0V0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0V.ptr + p0V.pitch * 0, p0V.pitch * 2, p0V.width, p0V.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP0V1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0V.ptr + p0V.pitch * 1, p0V.pitch * 2, p0V.width, p0V.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP1U0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1U.ptr + p1U.pitch * 0, p1U.pitch * 2, p1U.width, p1U.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP1U1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1U.ptr + p1U.pitch * 1, p1U.pitch * 2, p1U.width, p1U.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP1V0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1V.ptr + p1V.pitch * 0, p1V.pitch * 2, p1V.width, p1V.height >> 1))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type>(texP1V1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1V.ptr + p1V.pitch * 1, p1V.pitch * 2, p1V.width, p1V.height >> 1))) return cudaerr;
    } else {
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP0U0, cudaFilterModePoint, cudaReadModeElementType, p0U.ptr, p0U.pitch, (p0U.width + 3) / 4, p0U.height))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP0V0, cudaFilterModePoint, cudaReadModeElementType, p0V.ptr, p0V.pitch, (p0V.width + 3) / 4, p0V.height))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP1U0, cudaFilterModePoint, cudaReadModeElementType, p1U.ptr, p1U.pitch, (p1U.width + 3) / 4, p1U.height))) return cudaerr;
        if (cudaSuccess != (cudaerr = textureCreateAnalyze<Type4>(texP1V0, cudaFilterModePoint, cudaReadModeElementType, p1V.ptr, p1V.pitch, (p1V.width + 3) / 4, p1V.height))) return cudaerr;
    }

    dim3 blockSize(BLOCK_INT_X, BLOCK_Y);
    //横方向は1スレッドで4pixel処理する
    dim3 gridSize(divCeil(p0Y.width, blockSize.x * 4), divCeil(p0Y.height, blockSize.y * BLOCK_LOOP_Y));

    const uint32_t grid_count = gridSize.x * gridSize.y;
    if (count_motion->nSize < grid_count) {
        count_motion->clear();
        cudaerr = count_motion->alloc(grid_count * sizeof(int));
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    //opencl版を変更、横方向は1スレッドで4pixel処理するため、1/4にする必要がある
    const uint32_t scan_left   = pAfsPrm->clip.left >> 2;
    const uint32_t scan_width  = (p0Y.width - pAfsPrm->clip.left - pAfsPrm->clip.right) >> 2;
    const uint32_t scan_top    = pAfsPrm->clip.top;
    const uint32_t scan_height = (p0Y.height - pAfsPrm->clip.top - pAfsPrm->clip.bottom) & ~1;

    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリングのシフト値
    const int thre_rsft = 12 - (bit_depth - 8);
    //8bitなら最大127まで、16bitなら最大32627まで (bitshift等を使って比較する都合)
    const uint32_t thre_max = (1 << (sizeof(Type) * 8 - 1)) - 1;
    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリング
    uint32_t thre_shift_yuv   = (uint32_t)clamp((pAfsPrm->thre_shift   * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_deint_yuv   = (uint32_t)clamp((pAfsPrm->thre_deint   * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_Ymotion_yuv = (uint32_t)clamp((pAfsPrm->thre_Ymotion * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_Cmotion_yuv = (uint32_t)clamp((pAfsPrm->thre_Cmotion * 224 + 2112)>>thre_rsft, 0, thre_max);
    if (sizeof(Type) == 1) {
        thre_shift_yuv   = u8x4(thre_shift_yuv);
        thre_deint_yuv   = u8x4(thre_deint_yuv);
        thre_Ymotion_yuv = u8x4(thre_Ymotion_yuv);
        thre_Cmotion_yuv = u8x4(thre_Cmotion_yuv);
    } else if (sizeof(Type) == 2) {
        thre_shift_yuv   = u16x2(thre_shift_yuv);
        thre_deint_yuv   = u16x2(thre_deint_yuv);
        thre_Ymotion_yuv = u16x2(thre_Ymotion_yuv);
        thre_Cmotion_yuv = u16x2(thre_Cmotion_yuv);
    } else {
        return cudaErrorUnknown;
    }

    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリング
    //色差はcudaReadModeNormalizedFloatをつ開くので、そのぶんのスケーリングも必要
    const float thre_mul = (224.0f / (float)(4096 >> (bit_depth - 8))) * (1.0f / (1 << (sizeof(Type) * 8)));
    const float thre_shift_yuvf   = std::max(0.0f, pAfsPrm->thre_shift * thre_mul);
    const float thre_deint_yuvf   = std::max(0.0f, pAfsPrm->thre_deint * thre_mul);
    const float thre_Cmotion_yuvf = std::max(0.0f, pAfsPrm->thre_Cmotion * thre_mul);

    kernel_afs_analyze_12<Type, Type4, tb_order, yuv420><<<gridSize, blockSize, 0, stream>>>(
        (uint32_t *)dst, (int *)count_motion->ptrDevice,
        texP0Y, texP0U0, texP0U1, texP0V0, texP0V1,
        texP1Y, texP1U0, texP1U1, texP1V0, texP1V1,
        divCeil(p0Y.width, 4), dstPitch / sizeof(uint32_t), p0Y.height,
        thre_Ymotion_yuv, thre_deint_yuv, thre_shift_yuv,
        thre_Cmotion_yuv, thre_Cmotion_yuvf, thre_deint_yuvf, thre_shift_yuvf,
        scan_left, scan_top, scan_width, scan_height);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    cudaDestroyTextureObject(texP0Y);
    cudaDestroyTextureObject(texP1Y);
    cudaDestroyTextureObject(texP0U0);
    cudaDestroyTextureObject(texP0V0);
    cudaDestroyTextureObject(texP1U0);
    cudaDestroyTextureObject(texP1V0);
    if (yuv420) {
        cudaDestroyTextureObject(texP0U1);
        cudaDestroyTextureObject(texP0V1);
        cudaDestroyTextureObject(texP1U1);
        cudaDestroyTextureObject(texP1V1);
    }
    return cudaGetLastError();
}

cudaError_t NVEncFilterAfs::analyze_stripe(CUFrameBuf *p0, CUFrameBuf *p1, AFS_SCAN_DATA *sp, CUMemBufPair *count_motion, const NVEncFilterParamAfs *pAfsParam, cudaStream_t stream) {
    struct analyze_func {
        decltype(run_analyze_stripe<uint8_t, uint32_t, 8, true, true>)* func[2];
        analyze_func(decltype(run_analyze_stripe<uint8_t, uint32_t, 8, true, true>)* tb_order_0, decltype(run_analyze_stripe<uint8_t, uint32_t, 8, true, true>)* tb_order_1) {
            func[0] = tb_order_0;
            func[1] = tb_order_1;
        };
    };

    static const std::map<RGY_CSP, analyze_func> analyze_stripe_func_list = {
        { RGY_CSP_YV12,      analyze_func(run_analyze_stripe<uint8_t,  uint32_t, 8, false, true>,   run_analyze_stripe<uint8_t,  uint32_t, 8, true, true>)  },
        { RGY_CSP_YV12_16,   analyze_func(run_analyze_stripe<uint16_t, uint2,   16, false, true>,   run_analyze_stripe<uint16_t, uint2,   16, true, true>)  },
        { RGY_CSP_YUV444,    analyze_func(run_analyze_stripe<uint8_t,  uint32_t, 8, false, false>,  run_analyze_stripe<uint8_t,  uint32_t, 8, true, false>) },
        { RGY_CSP_YUV444_16, analyze_func(run_analyze_stripe<uint16_t, uint2,   16, false, false>,  run_analyze_stripe<uint16_t, uint2,   16, true, false>) },
    };
    if (p1->frame.pitch % sizeof(int) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("frame pitch must be mod4\n"));
        return cudaErrorNotSupported;
    }
    if (analyze_stripe_func_list.count(pAfsParam->frameIn.csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for afs_analyze_stripe: %s\n"), RGY_CSP_NAMES[pAfsParam->frameIn.csp]);
        return cudaErrorNotSupported;
    }
    auto cudaerr = analyze_stripe_func_list.at(pAfsParam->frameIn.csp).func[!!pAfsParam->afs.tb_order](
        sp->map.frame.ptr, sp->map.frame.pitch, &p0->frame, &p1->frame,
        count_motion, &pAfsParam->afs, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaSuccess;
}
