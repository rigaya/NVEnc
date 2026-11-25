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
#define BLOCK_LOOP_Y (4) //ブロックのy方向反復数

#define SHARED_INT_X (BLOCK_INT_X) //sharedメモリの幅
#define SHARED_Y     (16) //sharedメモリの縦

#define PREFER_IMAGE  1

const uint8_t SELECT_PLANE_Y = 0x01u;
const uint8_t SELECT_PLANE_U = 0x02u;
const uint8_t SELECT_PLANE_V = 0x04u;

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

// 4:2:0の色差を、インターレースを考慮して輝度解像度(4:4:4相当)へ整数演算で拡大する。
// 以前はanalyzeカーネル内で色差をtex2D<float>の線形補間で毎回求めていたが、
// その結果をfloatのスカラーで判定するため、輝度のようなSWAR(4画素同時処理)ができず、
// フェッチ数と命令数がanalyzeの大半を占めていた。
// ここで1フレームにつき1回だけ拡大しておき、analyzeでは輝度と同じType4のSWAR処理に乗せる。
// 重みは従来のテクスチャ補間と同じ:
//   縦: 輝度の行iy=4m+r に対し、同じフィールド(iy&1)の色差の行Fを使う (重みは1/8単位)
//       r=0: F0[m-1]*1 + F0[m]*7, r=1: F1[m-1]*3 + F1[m]*5,
//       r=2: F0[m]*5 + F0[m+1]*3, r=3: F1[m]*7 + F1[m+1]*1   (F0=偶数行, F1=奇数行, 端はclamp)
//   横: 輝度の偶数画素は色差の画素そのまま、奇数画素は左右の色差の平均
// 丸めは1回だけ行う (floatの値との差は許容する)。
// 出力: 上半分(0～height-1行)にU、下半分(height～2*height-1行)にV
// 1スレッドで色差4画素(輝度8画素)を処理する (1画素ずつだと命令発行が律速になったため)
template<typename Type, typename Type4, typename Type8>
__global__ void kernel_afs_upsample_uv420(
    uint8_t *__restrict__ dst, const int dst_pitch,
    const uint8_t *__restrict__ src_u, const uint8_t *__restrict__ src_v, const int src_pitch,
    const int uv_width, const int uv_height, const int height) {
    const int cx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // 色差のx
    const int iy = blockIdx.y * blockDim.y + threadIdx.y; // 輝度のy
    const int plane = blockIdx.z; // 0:U, 1:V
    if (cx >= uv_width || iy >= height) return;

    const uint8_t *src = (plane) ? src_v : src_u;
    const int field = iy & 1;
    const int m = iy >> 2;
    const int r = iy & 3;
    const int field_h = uv_height >> 1; // 従来のテクスチャ(高さ uv_height>>1)に合わせる
    const int fa = (r < 2) ? m - 1 : m;
    const int wb = 7 - 2 * r; // r=0:7, 1:5, 2:3, 3:1
    const int wa = 8 - wb;
    const int ya = clamp(fa,     0, field_h - 1) * 2 + field;
    const int yb = clamp(fa + 1, 0, field_h - 1) * 2 + field;
    const Type *line_a = (const Type *)(src + ya * src_pitch);
    const Type *line_b = (const Type *)(src + yb * src_pitch);

    //縦補間した値 (8倍) を5画素分求める。右端より先はpitch内のゴミなので、最後の有効画素で置き換える
    int v[5];
    {
        const Type4 a4 = *(const Type4 *)(line_a + cx);
        const Type4 b4 = *(const Type4 *)(line_b + cx);
        const Type *a = (const Type *)&a4;
        const Type *b = (const Type *)&b4;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v[k] = wa * (int)a[k] + wb * (int)b[k];
        }
        const int cx4 = min(cx + 4, uv_width - 1);
        v[4] = wa * (int)line_a[cx4] + wb * (int)line_b[cx4];
    }
    #pragma unroll
    for (int k = 1; k < 5; k++) {
        if (cx + k >= uv_width) v[k] = v[k-1];
    }
    Type8 out;
    Type *o = (Type *)&out;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        o[2*k+0] = (Type)((v[k] + 4) >> 3);
        o[2*k+1] = (Type)((v[k] + v[k+1] + 8) >> 4);
    }
    *(Type8 *)(dst + (plane * height + iy) * dst_pitch + cx * 2 * sizeof(Type)) = out;
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

//      7       6         5        4        3        2        1       0
// | motion  |         non-shift        | motion  |          shift          |
// |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
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
void merge_mask(Flags masky, Flags masku, Flags maskv, Flags& mask0, Flags& mask1, uint8_t select_plane) {
    // select_planeの値に基づいてビット演算でマスクを選択
    // 各ビット位置で選択するマスクを決定
    const Flags select_y = (select_plane & SELECT_PLANE_Y) ? 0xffffffff : 0x00;
    const Flags select_u = (select_plane & SELECT_PLANE_U) ? 0xffffffff : 0x00;
    const Flags select_v = (select_plane & SELECT_PLANE_V) ? 0xffffffff : 0x00;

    mask0 = (masky | (~select_y)) & (masku | (~select_u)) & (maskv | (~select_v));
    mask1 = (masky & select_y)    | (masku & select_u)    | (maskv & select_v);

    mask0 &= u8x4(0xcc); //motion
    mask1 &= u8x4(0x33); //shift/deint

    mask0 |= mask1;
}

//色差はyuv420の場合もkernel_afs_upsample_uv420で輝度解像度に拡大したものを受け取り、輝度と同じSWAR処理で判定する
template<typename Type, typename Type4, bool tb_order>
__global__ void kernel_afs_analyze_12(
    uint32_t *__restrict__ ptr_dst,
    int *__restrict__ ptr_count,
    cudaTextureObject_t src_p0y,
    cudaTextureObject_t src_p0u,
    cudaTextureObject_t src_p0v,
    cudaTextureObject_t src_p1y,
    cudaTextureObject_t src_p1u,
    cudaTextureObject_t src_p1v,
    const int width_int, const int si_pitch_int, const int h,
    const uint32_t thre_Ymotion, const uint32_t thre_deint, const uint32_t thre_shift,
    const uint32_t thre_Cmotion, const uint32_t thre_Cdeint, const uint32_t thre_Cshift,
    const uint32_t scan_left, const uint32_t scan_top, const uint32_t scan_width, const uint32_t scan_height, const uint8_t select_plane) {

    __shared__ uint32_t shared[SHARED_INT_X * SHARED_Y * 5]; //int単位でアクセスする
    const int lx = threadIdx.x; //スレッド数=BLOCK_INT_X
    int ly = threadIdx.y; //スレッド数=BLOCK_Y
    const int gidy = blockIdx.y; //グループID
    const int imgx = blockIdx.x * BLOCK_INT_X /*blockDim.x*/ + threadIdx.x;
    int imgy = (gidy * BLOCK_LOOP_Y * BLOCK_Y + ly);
    const int imgy_block_fin = min(h, ((gidy + 1) * BLOCK_LOOP_Y) * BLOCK_Y);
    uint32_t motion_count = 0;

#define CALL_ANALYZE_Y(p0, p1, y_offset) analyze_y<Type4, tb_order>((p0), (p1), (imgx), (imgy+(y_offset)), thre_Ymotion,  thre_deint,  thre_shift)
#define CALL_ANALYZE_C(p0, p1, y_offset) analyze_y<Type4, tb_order>((p0), (p1), (imgx), (imgy+(y_offset)), thre_Cmotion, thre_Cdeint, thre_Cshift)

    uint32_t *ptr_shared = shared + shared_int_idx(lx,0,0);
    ptr_dst += (imgy-4) * si_pitch_int + imgx;

    //前の4ライン分、計算しておく
    //sharedの SHARED_Y-4 ～ SHARED_Y-1 (ブロック直前の4行) と、0 ～ 3 (ブロック先頭の4行) を埋める
    //generate_flags(ly) は ly-3 ～ ly を参照するため、ブロック直前の4行が未設定だと、
    //未初期化のsharedメモリを読んでブロック境界 (128行ごと) の判定が不定になる
    //出力の行rは、解析結果の r-3 ～ r+4 行を使う (generate_flagsが -3 ～ 0、mask1が +4 行目)。
    //そのためブロック直前の -4 行目は不要で、計算しない (1warp = 1行なので分岐はwarp単位)
    //iy-4 がブロック先頭からの相対位置 (-4 ～ 3)、y_offset はこのスレッドの行(imgy)からの相対位置
    for (int iy = ly; iy < 8; iy += BLOCK_Y) {
        if (iy - 4 == -4) continue;
        const int y_offset = iy - ly - 4;
        const bool in_frame = imgy + y_offset >= 0; //フレーム上端より上は判定なし(0)とする
        ptr_shared[shared_int_idx(0, iy-4, 0)] = (in_frame) ? CALL_ANALYZE_Y(src_p0y, src_p1y, y_offset) : 0;
        ptr_shared[shared_int_idx(0, iy-4, 1)] = (in_frame) ? CALL_ANALYZE_C(src_p0u, src_p1u, y_offset) : 0;
        ptr_shared[shared_int_idx(0, iy-4, 2)] = (in_frame) ? CALL_ANALYZE_C(src_p0v, src_p1v, y_offset) : 0;
    }

    for (int iloop = 0; iloop <= BLOCK_LOOP_Y; iloop++,
        ptr_dst += BLOCK_Y * si_pitch_int, imgy += BLOCK_Y,
        ly += BLOCK_Y
    ) {
        //最後のループ(iloop == BLOCK_LOOP_Y)では、出力するのはブロック末尾の4行のみ。
        //その4行が使う解析結果は前のループまでで揃っているので、解析は不要。
        //マスクもその4行に必要な上半分(threadIdx.y < 4)だけ作ればよい。
        //(分岐はwarp単位で揃う。__syncthreadsは全スレッドが通るよう条件の外に置く)
        const bool last_loop = iloop == BLOCK_LOOP_Y;
        { //差分情報を計算
            if (!last_loop) {
                ptr_shared[shared_int_idx(0, ly+4, 0)] = CALL_ANALYZE_Y(src_p0y, src_p1y, 4);
                ptr_shared[shared_int_idx(0, ly+4, 1)] = CALL_ANALYZE_C(src_p0u, src_p1u, 4);
                ptr_shared[shared_int_idx(0, ly+4, 2)] = CALL_ANALYZE_C(src_p0v, src_p1v, 4);
            }
            __syncthreads();
        }
        Flags mask1 = 0;
        { //マスク生成
            if (!last_loop || threadIdx.y < 4) {
                Flags masky = generate_flags(ly, 0, ptr_shared);
                Flags masku = generate_flags(ly, 1, ptr_shared);
                Flags maskv = generate_flags(ly, 2, ptr_shared);
                Flags mask0;
                merge_mask(masky, masku, maskv, mask0, mask1, select_plane);
                ptr_shared[shared_int_idx(0, ly, 3)] = mask0; //motion/shift/deint
            }
            __syncthreads();
        }
        { //最終出力
            //ly+4とか使っているので準備ができてないうちから、次の列のデータを使うことになってまずい
            Flags mask4, mask5, mask6, mask7;
            mask4 = ptr_shared[shared_int_idx(0, ly-1, 3)];
            mask5 = ptr_shared[shared_int_idx(0, ly-2, 3)];
            mask6 = ptr_shared[shared_int_idx(0, ly-3, 3)];
            mask7 = ptr_shared[shared_int_idx(0, ly-4, 3)]; // これが自分自身
            mask1 &= u8x4(0x30); // non-shift shift/deint
            mask4 |= mask5 | mask6;
            mask4 &= u8x4(0x33); // non-shift+shift shift/deint
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
cudaError_t textureCreateAnalyze(afsTextureCache *texCache, cudaTextureObject_t& tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
    return texCache->get(tex, cudaCreateChannelDesc<Type>(), filterMode, readMode, ptr, pitch, width, height);
}

template<typename Type, typename Type4, int bit_depth, bool tb_order, bool yuv420>
RGY_ERR run_analyze_stripe(uint8_t *dst, const int dstPitch,
    const RGYFrameInfo *pFrame0, const RGYFrameInfo *pFrame1,
    const RGYFrameInfo *pUV444_0, const RGYFrameInfo *pUV444_1, //yuv420の場合の輝度解像度に拡大した色差 (上半分U/下半分V)
    CUMemBufPair *count_motion, afsTextureCache *texCache,
    const VppAfs *pAfsPrm, cudaStream_t stream) {
    const auto p0Y = getPlane(pFrame0, RGY_PLANE_Y);
    const auto p1Y = getPlane(pFrame1, RGY_PLANE_Y);
    auto p0U = getPlane(pFrame0, RGY_PLANE_U);
    auto p0V = getPlane(pFrame0, RGY_PLANE_V);
    auto p1U = getPlane(pFrame1, RGY_PLANE_U);
    auto p1V = getPlane(pFrame1, RGY_PLANE_V);

    if (   p0Y.width != p1Y.width || p0Y.height != p1Y.height
        || p0U.width != p1U.width || p0U.height != p1U.height
        || p0V.width != p1V.width || p0V.height != p1V.height) {
        return RGY_ERR_UNKNOWN;
    }
    if (yuv420) {
        //拡大済みの色差を輝度と同じサイズの平面として扱う
        auto setUV444Plane = [&p0Y](RGYFrameInfo& planeU, RGYFrameInfo& planeV, const RGYFrameInfo *pUV444) {
            planeU = *pUV444;
            planeU.width = p0Y.width;
            planeU.height = p0Y.height;
            planeV = planeU;
            planeV.ptr[0] = planeU.ptr[0] + (size_t)planeU.pitch[0] * p0Y.height;
        };
        setUV444Plane(p0U, p0V, pUV444_0);
        setUV444Plane(p1U, p1V, pUV444_1);
    }

    auto sts = RGY_ERR_NONE;
    cudaTextureObject_t texP0Y = 0, texP0U = 0, texP0V = 0;
    cudaTextureObject_t texP1Y = 0, texP1U = 0, texP1V = 0;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP0Y, cudaFilterModePoint, cudaReadModeElementType, p0Y.ptr[0], p0Y.pitch[0], (p0Y.width + 3) / 4, p0Y.height)))) return sts;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP1Y, cudaFilterModePoint, cudaReadModeElementType, p1Y.ptr[0], p1Y.pitch[0], (p1Y.width + 3) / 4, p1Y.height)))) return sts;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP0U, cudaFilterModePoint, cudaReadModeElementType, p0U.ptr[0], p0U.pitch[0], (p0U.width + 3) / 4, p0U.height)))) return sts;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP0V, cudaFilterModePoint, cudaReadModeElementType, p0V.ptr[0], p0V.pitch[0], (p0V.width + 3) / 4, p0V.height)))) return sts;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP1U, cudaFilterModePoint, cudaReadModeElementType, p1U.ptr[0], p1U.pitch[0], (p1U.width + 3) / 4, p1U.height)))) return sts;
    if (RGY_ERR_NONE != (sts = err_to_rgy(textureCreateAnalyze<Type4>(texCache, texP1V, cudaFilterModePoint, cudaReadModeElementType, p1V.ptr[0], p1V.pitch[0], (p1V.width + 3) / 4, p1V.height)))) return sts;

    dim3 blockSize(BLOCK_INT_X, BLOCK_Y);
    //横方向は1スレッドで4pixel処理する
    dim3 gridSize(divCeil(p0Y.width, blockSize.x * 4), divCeil(p0Y.height, blockSize.y * BLOCK_LOOP_Y));

    const uint32_t grid_count = gridSize.x * gridSize.y;
    if (count_motion->nSize < grid_count) {
        count_motion->clear();
        sts = count_motion->alloc(grid_count * sizeof(int));
        if (sts != RGY_ERR_NONE) {
            return sts;
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
    const int thre_max = (1 << (sizeof(Type) * 8 - 1)) - 1;
    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリング
    uint32_t thre_shift_yuv   = (uint32_t)clamp((pAfsPrm->thre_shift   * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_deint_yuv   = (uint32_t)clamp((pAfsPrm->thre_deint   * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_Ymotion_yuv = (uint32_t)clamp((pAfsPrm->thre_Ymotion * 219 +  383)>>thre_rsft, 0, thre_max);
    uint32_t thre_Cmotion_yuv = (uint32_t)clamp((pAfsPrm->thre_Cmotion * 224 + 2112)>>thre_rsft, 0, thre_max);
    uint32_t thre_Cdeint_yuv  = thre_deint_yuv;
    uint32_t thre_Cshift_yuv  = thre_shift_yuv;
    if (yuv420) {
        //以前のfloat版(色差のスケール 224/4096)の判定にできるだけ合わせた整数の閾値
        //motion : (thre > abs) の判定なので切り上げ
        //deint/shift : (abs > thre) の判定なので切り捨て
        //(shiftはmotion側でも thre > abs で使うが、既定値では割り切れるので差は出ない)
        thre_Cmotion_yuv = (uint32_t)clamp((pAfsPrm->thre_Cmotion * 224 + (1 << thre_rsft) - 1) >> thre_rsft, 0, thre_max);
        thre_Cdeint_yuv  = (uint32_t)clamp((pAfsPrm->thre_deint   * 224) >> thre_rsft, 0, thre_max);
        thre_Cshift_yuv  = (uint32_t)clamp((pAfsPrm->thre_shift   * 224) >> thre_rsft, 0, thre_max);
    }
    auto pack_thre = [](uint32_t thre) {
        return (sizeof(Type) == 1) ? u8x4(thre) : u16x2(thre);
    };
    static_assert(sizeof(Type) == 1 || sizeof(Type) == 2, "unsupported Type");
    thre_shift_yuv   = pack_thre(thre_shift_yuv);
    thre_deint_yuv   = pack_thre(thre_deint_yuv);
    thre_Ymotion_yuv = pack_thre(thre_Ymotion_yuv);
    thre_Cmotion_yuv = pack_thre(thre_Cmotion_yuv);
    thre_Cdeint_yuv  = pack_thre(thre_Cdeint_yuv);
    thre_Cshift_yuv  = pack_thre(thre_Cshift_yuv);

    uint8_t select_plane = SELECT_PLANE_Y | SELECT_PLANE_U | SELECT_PLANE_V;
    if (pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_SHIFT_Y || pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_NONSHIFT_Y) {
        select_plane = SELECT_PLANE_Y;
    } else if (pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_SHIFT_U || pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_NONSHIFT_U) {
        select_plane = SELECT_PLANE_U;
    } else if (pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_SHIFT_V || pAfsPrm->tune == AFS_TUNE_MODE_ANALYZE_NONSHIFT_V) {
        select_plane = SELECT_PLANE_V;
    }

    kernel_afs_analyze_12<Type, Type4, tb_order><<<gridSize, blockSize, 0, stream>>>(
        (uint32_t *)dst, (int *)count_motion->ptrDevice,
        texP0Y, texP0U, texP0V,
        texP1Y, texP1U, texP1V,
        divCeil(p0Y.width, 4), dstPitch / sizeof(uint32_t), p0Y.height,
        thre_Ymotion_yuv, thre_deint_yuv, thre_shift_yuv,
        thre_Cmotion_yuv, thre_Cdeint_yuv, thre_Cshift_yuv,
        scan_left, scan_top, scan_width, scan_height, select_plane);
    //テクスチャはtexCacheが保持するので、ここでは破棄しない
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, typename Type4, typename Type8>
RGY_ERR run_upsample_uv420(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    const auto planeY = getPlane(pSrc, RGY_PLANE_Y);
    const auto planeU = getPlane(pSrc, RGY_PLANE_U);
    const auto planeV = getPlane(pSrc, RGY_PLANE_V);
    if (planeU.pitch[0] != planeV.pitch[0]
        || pDst->width < planeY.width || pDst->height < planeY.height * 2) {
        return RGY_ERR_UNSUPPORTED;
    }
    //ベクトルでの読み書きのため、各行の先頭がType4/Type8単位で揃っている必要がある
    if (   (planeU.pitch[0] % sizeof(Type4)) != 0 || ((size_t)planeU.ptr[0] % sizeof(Type4)) != 0 || ((size_t)planeV.ptr[0] % sizeof(Type4)) != 0
        || (pDst->pitch[0] % sizeof(Type8)) != 0 || ((size_t)pDst->ptr[0] % sizeof(Type8)) != 0) {
        return RGY_ERR_UNSUPPORTED;
    }
    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divCeil(planeU.width, blockSize.x * 4), divCeil(planeY.height, blockSize.y), 2);
    kernel_afs_upsample_uv420<Type, Type4, Type8><<<gridSize, blockSize, 0, stream>>>(
        pDst->ptr[0], pDst->pitch[0],
        planeU.ptr[0], planeV.ptr[0], planeU.pitch[0],
        planeU.width, planeU.height, planeY.height);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterAfs::upsample_uv420(CUFrameBuf *pDst, const CUFrameBuf *pSrc, cudaStream_t stream) {
    switch (pSrc->frame.csp) {
    case RGY_CSP_YV12:    return run_upsample_uv420<uint8_t,  uint32_t, uint2>(&pDst->frame, &pSrc->frame, stream);
    case RGY_CSP_YV12_16: return run_upsample_uv420<uint16_t, uint2,    uint4>(&pDst->frame, &pSrc->frame, stream);
    default:
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for afs_upsample_uv420: %s\n"), RGY_CSP_NAMES[pSrc->frame.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterAfs::analyze_stripe(CUFrameBuf *p0, CUFrameBuf *p1, const CUFrameBuf *p0uv444, const CUFrameBuf *p1uv444, AFS_SCAN_DATA *sp, CUMemBufPair *count_motion, const NVEncFilterParamAfs *pAfsParam, cudaStream_t stream) {
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
    for (int i = 0; i < RGY_CSP_PLANES[p1->frame.csp]; i++) {
        if (p1->frame.pitch[i] % sizeof(int) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("frame pitch must be mod4\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (analyze_stripe_func_list.count(pAfsParam->frameIn.csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for afs_analyze_stripe: %s\n"), RGY_CSP_NAMES[pAfsParam->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    const bool yuv420 = RGY_CSP_CHROMA_FORMAT[pAfsParam->frameIn.csp] == RGY_CHROMAFMT_YUV420;
    if (yuv420 && (p0uv444 == nullptr || p1uv444 == nullptr)) {
        AddMessage(RGY_LOG_ERROR, _T("upsampled chroma buffer not set for afs_analyze_stripe.\n"));
        return RGY_ERR_NULL_PTR;
    }
    auto sts = analyze_stripe_func_list.at(pAfsParam->frameIn.csp).func[!!pAfsParam->afs.tb_order](
        sp->map.frame.ptr[0], sp->map.frame.pitch[0], &p0->frame, &p1->frame,
        (yuv420) ? &p0uv444->frame : nullptr, (yuv420) ? &p1uv444->frame : nullptr,
        count_motion, &m_texCache, &pAfsParam->afs, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}
