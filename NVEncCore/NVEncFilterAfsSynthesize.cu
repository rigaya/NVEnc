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
#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#pragma warning (pop)
#include "convert_csp.h"
#include "NVEncFilterAfs.h"
#include "NVEncParam.h"

#define SYN_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
#define SYN_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group
#define SYN_BLOCK_LOOP_Y  (1) //work groupのy方向反復数

#define u8x4(x)  (((uint32_t)x) | (((uint32_t)x) <<  8) | (((uint32_t)x) << 16) | (((uint32_t)x) << 24))

template <typename T>
__device__ __inline__
T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

__device__ __inline__
uint32_t blend(uint32_t a_if_0, uint32_t b_if_1, uint32_t mask) {
    return (a_if_0 & (~mask)) | (b_if_1 & (mask));
}

// 後方フィールド判定
__device__ __inline__
int is_latter_field(int pos_y, int tb_order) {
    return (((pos_y + tb_order + 1) & 1));
}

__device__ __inline__
uint32_t deint(int src1, int src3, int src4, int src5, int src7, uint32_t flag, uint32_t mask, int max) {
    const int tmp2 = src1 + src7;
    const int tmp3 = src3 + src5;
    //const int tmp = (tmp3 - ((tmp2 - tmp3) >> 3) + 1) >> 1;
    const int tmp = clamp(__rhadd(tmp3, (tmp3 - tmp2) >> 3), 0, max);
    return (uint32_t)(((flag & mask) == 0) ? tmp : src4);
}

__device__ __inline__
float deint(float src1, float src3, float src4, float src5, float src7, uint32_t flag, uint32_t mask) {
    //const float tmp2 = src1 + src7;
    //const float tmp3 = src3 + src5;
    //const float tmp = (tmp3 - ((tmp2 - tmp3) * 0.125f)) * 0.5f;
    const float tmp = (src3 + src5) * 0.5625f - (src1 + src7) * 0.0625f;
    return (((flag & mask) == 0) ? tmp : src4);
}

__device__ __inline__
uint32_t blend(int src1, int src2, int src3, uint32_t flag, uint32_t mask) {
    int tmp = (src1 + src3 + src2 + src2 + 2) >> 2;
    return (uint32_t)(((flag & mask) == 0) ? tmp : src2);
}

__device__ __inline__
float blend(float src1, float src2, float src3, uint32_t flag, uint32_t mask) {
    float tmp = (src1 + src3 + 2.0f * src2) * 0.25f;
    return ((flag & mask) == 0) ? tmp : src2;
}

__device__ __inline__
uint32_t mie_inter(int src1, int src2, int src3, int src4) {
    return (uint32_t)((src1 + src2 + src3 + src4 + 2) >> 2);
}

__device__ __inline__
float mie_inter(float src1, float src2, float src3, float src4) {
    return (src1 + src2 + src3 + src4) * 0.25f;
}

__device__ __inline__
uint32_t mie_spot(int src1, int src2, int src3, int src4, int src_spot) {
    return __urhadd(mie_inter(src1, src2, src3, src4), src_spot);
}

__device__ __inline__
float mie_spot(float src1, float src2, float src3, float src4, float src_spot) {
    return (mie_inter(src1, src2, src3, src4) + src_spot) * 0.5f;
}

__device__ __inline__
uint32_t deint4(uint32_t src1, uint32_t src3, uint32_t src4, uint32_t src5, uint32_t src7, uint32_t flag, uint32_t mask) {
    uint32_t p0 = deint((int)(src1 & 0x000000ff),       (int)(src3 & 0x000000ff),       (int)(src4 & 0x000000ff),       (int)(src5 & 0x000000ff),       (int)(src7 & 0x000000ff),       flag & 0x000000ff, mask, 0xff);
    uint32_t p1 = deint((int)(src1 & 0x0000ff00) >>  8, (int)(src3 & 0x0000ff00) >>  8, (int)(src4 & 0x0000ff00) >>  8, (int)(src5 & 0x0000ff00) >>  8, (int)(src7 & 0x0000ff00) >>  8, flag & 0x0000ff00, mask, 0xff) <<  8;
    uint32_t p2 = deint((int)(src1 & 0x00ff0000) >> 16, (int)(src3 & 0x00ff0000) >> 16, (int)(src4 & 0x00ff0000) >> 16, (int)(src5 & 0x00ff0000) >> 16, (int)(src7 & 0x00ff0000) >> 16, flag & 0x00ff0000, mask, 0xff) << 16;
    uint32_t p3 = deint((int)(src1 >> 24),              (int)(src3 >> 24),              (int)(src4 >> 24),              (int)(src5 >> 24),              (int)(src7 >> 24),              flag >> 24,        mask, 0xff) << 24;
    return p0 | p1 | p2 | p3;
}

__device__ __inline__
uint32_t deint2(uint32_t src1, uint32_t src3, uint32_t src4, uint32_t src5, uint32_t src7, uint32_t flag, uint32_t mask) {
    uint32_t p0 = deint((int)(src1 & 0x0000ffff), (int)(src3 & 0x0000ffff), (int)(src4 & 0x0000ffff), (int)(src5 & 0x0000ffff), (int)(src7 & 0x0000ffff), flag & 0x000000ff, mask, 0x0000ffff);
    uint32_t p1 = deint((int)(src1 >> 16), (int)(src3 >> 16), (int)(src4 >> 16), (int)(src5 >> 16), (int)(src7 >> 16), flag & 0x0000ff00, mask, 0x0000ffff) << 16;
    return p0 | p1;
}

__device__ __inline__
uint32_t blend4(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t flag, uint32_t mask) {
    uint32_t p0 = blend((int)(src1 & 0x000000ff), (int)(src2 & 0x000000ff), (int)(src3 & 0x000000ff), flag & 0x000000ff, mask);
    uint32_t p1 = blend((int)(src1 & 0x0000ff00), (int)(src2 & 0x0000ff00), (int)(src3 & 0x0000ff00), flag & 0x0000ff00, mask) & 0x0000ff00;
    uint32_t p2 = blend((int)(src1 & 0x00ff0000), (int)(src2 & 0x00ff0000), (int)(src3 & 0x00ff0000), flag & 0x00ff0000, mask) & 0x00ff0000;
    uint32_t p3 = blend((int)(src1 >> 24),        (int)(src2 >> 24),        (int)(src3 >> 24),        flag >> 24,        mask) << 24;
    return p0 | p1 | p2 | p3;
}

__device__ __inline__
uint32_t blend2(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t flag, uint32_t mask) {
    uint32_t p0 = blend((int)(src1 & 0x0000ffff), (int)(src2 & 0x0000ffff), (int)(src3 & 0x0000ffff), flag & 0x0000ffff, mask);
    uint32_t p1 = blend((int)(src1 >> 16), (int)(src2 >> 16), (int)(src3 >> 16), flag & 0x0000ff00, mask) << 16;
    return p0 | p1;
}

__device__ __inline__
uint32_t mie_inter4(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t src4) {
    uint32_t p0 = mie_inter((int)(src1 & 0x000000ff), (int)(src2 & 0x000000ff), (int)(src3 & 0x000000ff), (int)(src4 & 0x000000ff));
    uint32_t p1 = mie_inter((int)(src1 & 0x0000ff00), (int)(src2 & 0x0000ff00), (int)(src3 & 0x0000ff00), (int)(src4 & 0x0000ff00)) & 0x0000ff00;
    uint32_t p2 = mie_inter((int)(src1 & 0x00ff0000), (int)(src2 & 0x00ff0000), (int)(src3 & 0x00ff0000), (int)(src4 & 0x00ff0000)) & 0x00ff0000;
    uint32_t p3 = mie_inter((int)(src1 >> 24),        (int)(src2 >> 24),        (int)(src3 >> 24),        (int)(src4 >> 24)       ) << 24;
    return p0 | p1 | p2 | p3;
}

__device__ __inline__
uint32_t mie_inter2(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t src4) {
    uint32_t p0 = mie_inter((int)(src1 & 0x0000ffff), (int)(src2 & 0x0000ffff), (int)(src3 & 0x0000ffff), (int)(src4 & 0x0000ffff));
    uint32_t p1 = mie_inter((int)(src1 >> 16), (int)(src2 >> 16), (int)(src3 >> 16), (int)(src4 >> 16)) << 16;
    return p0 | p1;
}

__device__ __inline__
uint32_t mie_spot4(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t src4, uint32_t src_spot) {
    uint32_t p0 = mie_spot((int)(src1 & 0x000000ff), (int)(src2 & 0x000000ff), (int)(src3 & 0x000000ff), (int)(src4 & 0x000000ff), (int)(src_spot & 0x000000ff));
    uint32_t p1 = mie_spot((int)(src1 & 0x0000ff00), (int)(src2 & 0x0000ff00), (int)(src3 & 0x0000ff00), (int)(src4 & 0x0000ff00), (int)(src_spot & 0x0000ff00)) & 0x0000ff00;
    uint32_t p2 = mie_spot((int)(src1 & 0x00ff0000), (int)(src2 & 0x00ff0000), (int)(src3 & 0x00ff0000), (int)(src4 & 0x00ff0000), (int)(src_spot & 0x00ff0000)) & 0x00ff0000;
    uint32_t p3 = mie_spot((int)(src1 >> 24),        (int)(src2 >> 24),        (int)(src3 >> 24),        (int)(src4 >> 24),        (int)(src_spot >> 24)       ) << 24;
    return p0 | p1 | p2 | p3;
}

__device__ __inline__
uint32_t mie_spot2(uint32_t src1, uint32_t src2, uint32_t src3, uint32_t src4, uint32_t src_spot) {
    uint32_t p0 = mie_spot((int)(src1 & 0x0000ffff), (int)(src2 & 0x0000ffff), (int)(src3 & 0x0000ffff), (int)(src4 & 0x0000ffff), (int)(src_spot & 0x0000ffff));
    uint32_t p1 = mie_spot((int)(src1 >> 16), (int)(src2 >> 16), (int)(src3 >> 16), (int)(src4 >> 16), (int)(src_spot >> 16)) << 16;
    return p0 | p1;
}

__device__ __inline__
uint2 deint8(uint2 src1, uint2 src3, uint2 src4, uint2 src5, uint2 src7, uint2 flag, uint32_t mask) {
    uint2 pout;
    pout.x = deint4(src1.x, src3.x, src4.x, src5.x, src7.x, flag.x, mask);
    pout.y = deint4(src1.y, src3.y, src4.y, src5.y, src7.y, flag.y, mask);
    return pout;
}

__device__ __inline__
uint4 deint8(uint4 src1, uint4 src3, uint4 src4, uint4 src5, uint4 src7, uint2 flag, uint32_t mask) {
    uint4 pout;
    pout.x = deint2(src1.x, src3.x, src4.x, src5.x, src7.x, flag.x >>  0, mask);
    pout.y = deint2(src1.y, src3.y, src4.y, src5.y, src7.y, flag.x >> 16, mask);
    pout.z = deint2(src1.z, src3.z, src4.z, src5.z, src7.z, flag.y >>  0, mask);
    pout.w = deint2(src1.w, src3.w, src4.w, src5.w, src7.w, flag.y >> 16, mask);
    return pout;
}

__device__ __inline__
uint2 blend8(uint2 src1, uint2 src2, uint2 src3, uint2 flag, uint32_t mask) {
    uint2 pout;
    pout.x = blend4(src1.x, src2.x, src3.x, flag.x, mask);
    pout.y = blend4(src1.y, src2.y, src3.y, flag.y, mask);
    return pout;
}

__device__ __inline__
uint4 blend8(uint4 src1, uint4 src2, uint4 src3, uint2 flag, uint32_t mask) {
    uint4 pout;
    pout.x = blend2(src1.x, src2.x, src3.x, flag.x >>  0, mask);
    pout.y = blend2(src1.y, src2.y, src3.y, flag.x >> 16, mask);
    pout.z = blend2(src1.z, src2.z, src3.z, flag.y >>  0, mask);
    pout.w = blend2(src1.w, src2.w, src3.w, flag.y >> 16, mask);
    return pout;
}

__device__ __inline__
uint2 mie_inter8(uint2 src1, uint2 src2, uint2 src3, uint2 src4) {
    uint2 pout;
    pout.x = mie_inter4(src1.x, src2.x, src3.x, src4.x);
    pout.y = mie_inter4(src1.y, src2.y, src3.y, src4.y);
    return pout;
}

__device__ __inline__
uint4 mie_inter8(uint4 src1, uint4 src2, uint4 src3, uint4 src4) {
    uint4 pout;
    pout.x = mie_inter2(src1.x, src2.x, src3.x, src4.x);
    pout.y = mie_inter2(src1.y, src2.y, src3.y, src4.y);
    pout.z = mie_inter2(src1.z, src2.z, src3.z, src4.z);
    pout.w = mie_inter2(src1.w, src2.w, src3.w, src4.w);
    return pout;
}

__device__ __inline__
uint2 mie_spot8(uint2 src1, uint2 src2, uint2 src3, uint2 src4, uint2 src_spot) {
    uint2 pout;
    pout.x = mie_spot4(src1.x, src2.x, src3.x, src4.x, src_spot.x);
    pout.y = mie_spot4(src1.y, src2.y, src3.y, src4.y, src_spot.y);
    return pout;
}

__device__ __inline__
uint4 mie_spot8(uint4 src1, uint4 src2, uint4 src3, uint4 src4, uint4 src_spot) {
    uint4 pout;
    pout.x = mie_spot2(src1.x, src2.x, src3.x, src4.x, src_spot.x);
    pout.y = mie_spot2(src1.y, src2.y, src3.y, src4.y, src_spot.y);
    pout.z = mie_spot2(src1.z, src2.z, src3.z, src4.z, src_spot.z);
    pout.w = mie_spot2(src1.z, src2.w, src3.w, src4.w, src_spot.w);
    return pout;
}

template<typename Type8, int plane, int line>
__device__ __inline__
Type8 piny(
    const uint8_t *__restrict__ p0,
    const uint8_t *__restrict__ p1,
    int y_h1_pos, int y_h2_pos, int y_h3_pos,
    int y_h4_pos, int y_h5_pos, int y_h6_pos, int y_h7_pos) {
    const uint8_t *ptr = (plane) ? p1 : p0;
    switch (line) {
    case 1: ptr += y_h1_pos; break;
    case 2: ptr += y_h2_pos; break;
    case 3: ptr += y_h3_pos; break;
    case 4: ptr += y_h4_pos; break;
    case 5: ptr += y_h5_pos; break;
    case 6: ptr += y_h6_pos; break;
    case 7: ptr += y_h7_pos; break;
    default: break;
    }
    return *(Type8 *)ptr;
}

template<typename Type8, int mode>
__device__ __inline__
void proc_y(
    uint8_t *__restrict__ dst,
    const uint8_t *__restrict__ p0,
    const uint8_t *__restrict__ p1,
    const uint8_t *__restrict__ sip,
    const int tb_order, const uint8_t status,
    int y_h1_pos, int y_h2_pos, int y_h3_pos,
    int y_h4_pos, int y_h5_pos, int y_h6_pos, int y_h7_pos
    ) {
    static_assert(-1 <= mode && mode <= 4, "mode should be -1 - 4");
#define pin(plane, line) piny<Type8, plane, line>(p0, p1, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos)

    Type8 pout;
    if (mode == 1) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = mie_inter8(pin(0, 2), pin(1, 1), pin(1, 2), pin(1, 3));
            } else {
                pout = mie_spot8(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(1, 2));
            }
        } else {
            if (is_latter_field(0, tb_order)) {
                pout = mie_inter8(pin(0, 1), pin(0, 2), pin(0, 3), pin(1, 2));
            } else {
                pout = mie_spot8(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(0, 2));
            }
        }
    } else if (mode == 2) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = blend8(pin(1,1), pin(0,2), pin(1,3), *(uint2 *)sip, 0x02020202);
            } else {
                pout = blend8(pin(0,1), pin(1,2), pin(0,3), *(uint2 *)sip, 0x02020202);
            }
        } else {
            pout = blend8(pin(0,1), pin(0,2), pin(0,3), *(uint2 *)sip, 0x01010101);
        }
    } else if (mode == 3) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = blend8(pin(1, 1), pin(0, 2), pin(1, 3), *(uint2 *)sip, 0x06060606);
            } else {
                pout = blend8(pin(0, 1), pin(1, 2), pin(0, 3), *(uint2 *)sip, 0x06060606);
            }
        } else {
            pout = blend8(pin(0, 1), pin(0, 2), pin(0, 3), *(uint2 *)sip, 0x05050505);
        }
    } else if (mode == 4) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = deint8(pin(1,1), pin(1,3), pin(0,4), pin(1,5), pin(1,7), *(uint2 *)sip, 0x06060606);
            } else {
                pout = pin(1, 4);
            }
        } else {
            if (is_latter_field(0, tb_order)) {
                pout = deint8(pin(0,1), pin(0,3), pin(0,4), pin(0,5), pin(0,7), *(uint2 *)sip, 0x05050505);
            } else {
                pout = pin(0, 4);
            }
        }
    }
    *(Type8 *)dst = pout;
#undef pin
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

template<typename Type, typename Type4, int mode>
__device__ __inline__
void proc_uv(
    Type4 *__restrict__ dst,
    cudaTextureObject_t src_p0_0,
    cudaTextureObject_t src_p0_1,
    cudaTextureObject_t src_p1_0,
    cudaTextureObject_t src_p1_1,
    const uint8_t *__restrict__ sip,
    const int sip_pitch,
    const int src_width, const int src_height,
    const int imgx,  //グローバルスレッドID x
    const int imgy,  //グローバルスレッドID y
    const int lx, const int ly, //ローカルスレッドID
    const int tb_order, const uint8_t status
) {
    static_assert(-1 <= mode && mode <= 4, "mode should be -1 - 4");
#define PREREAD_Y ((mode == 4) ? 6 : 2) //必要になる前後の行のぶん
#define SHARED_C_Y (SYN_BLOCK_Y * 2 + PREREAD_Y)
#define SHARED_C_XY (SHARED_C_Y * SYN_BLOCK_INT_X)
#define SOFFSET(x,y,depth) ((depth) * SHARED_C_XY + (y) * SYN_BLOCK_INT_X + (x))
    __shared__ float s_tmp[3][SHARED_C_Y][SYN_BLOCK_INT_X];
    __shared__ Type s_out[SYN_BLOCK_Y][SYN_BLOCK_INT_X * 4];
    float *pSharedX = (float *)&s_tmp[0][0][0] + SOFFSET(lx, 0, 0);
    Type *psOut = &s_out[ly][lx];
    int ix = blockIdx.x * SYN_BLOCK_INT_X * 4 + threadIdx.x; //YUV422ベースの色差インデックス 1スレッドは横方向に4つの色差pixelを担当
    int iy = blockIdx.y * SYN_BLOCK_Y * 2 + threadIdx.y;     //YUV422ベースの色差インデックス 1スレッドは縦方向に2つの色差pixelを担当 (出力時はYUV420なので1つの色差pixelを担当)
    float ifx = (float)ix + 0.5f;

    //この関数内でsipだけはYUV444のデータであることに注意
    sip += iy * sip_pitch + ix * 4/*4画素/スレッド*/ * 2/*YUV444->YUV420*/ * sizeof(uint8_t);

    //sharedメモリ上に、YUV422相当のデータ(32x(16+PREREAD))を縦方向のテクスチャ補間で作ってから、
    //blendを実行して、YUV422相当の合成データ(32x16)を作り、
    //その後YUV420相当のデータ(32x8)をs_outに出力する
    //横方向に4回ループを回して、32pixel x4の出力結果をs_out(横:128pixel)に格納する
    for (int i = 0; i < 4; i++, ifx += SYN_BLOCK_INT_X, psOut += SYN_BLOCK_INT_X, sip += 2/*YUV444->YUV420*/) {
        //shredメモリに値をロード
        //縦方向のテクスチャ補間を使って、YUV422相当のデータとしてロード
        //横方向には補間しない
        if (ly < PREREAD_Y) {
            float *pShared = pSharedX + SOFFSET(0, ly, 0);
            pShared[SOFFSET(0,0,0)] = get_uv(src_p0_0, src_p0_1, ifx, iy - (PREREAD_Y >> 1));
            pShared[SOFFSET(0,0,1)] = get_uv(src_p1_0, src_p1_1, ifx, iy - (PREREAD_Y >> 1));
        }
        float *pShared = pSharedX + SOFFSET(0, ly + PREREAD_Y, 0);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            pShared[SOFFSET(0,j*SYN_BLOCK_Y,0)] = get_uv(src_p0_0, src_p0_1, ifx, iy + (PREREAD_Y >> 1) + j * SYN_BLOCK_Y);
            pShared[SOFFSET(0,j*SYN_BLOCK_Y,1)] = get_uv(src_p1_0, src_p1_1, ifx, iy + (PREREAD_Y >> 1) + j * SYN_BLOCK_Y);
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            //sipのy (境界チェックに必要)
            const int iy_sip = iy + j * SYN_BLOCK_Y;
            const uint8_t *psip = sip + SYN_BLOCK_Y * sip_pitch;

            // -1するのは、pinのlineは最小値が1だから
            float *pShared = pSharedX + SOFFSET(0, ly-1+j*SYN_BLOCK_Y, 0);
#define pin(plane, line) (pShared[SOFFSET(0,line,plane)])

            float pout;
            if (mode == 1) {
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = mie_inter(pin(0, 2), pin(1, 1), pin(1, 2), pin(1, 3));
                    } else {
                        pout = mie_spot(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(1, 2));
                    }
                } else {
                    if (is_latter_field(ly, tb_order)) {
                        pout = mie_inter(pin(0, 1), pin(0, 2), pin(0, 3), pin(1, 2));
                    } else {
                        pout = mie_spot(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(0, 2));
                    }
                }
            } else if (mode == 2) {
                const uint8_t sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = blend(pin(1, 1), pin(0, 2), pin(1, 3), sip0, 0x02020202);
                    } else {
                        pout = blend(pin(0, 1), pin(1, 2), pin(0, 3), sip0, 0x02020202);
                    }
                } else {
                    pout = blend(pin(0, 1), pin(0, 2), pin(0, 3), sip0, 0x01010101);
                }
            } else if (mode == 3) {
                const uint8_t sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = blend(pin(1, 1), pin(0, 2), pin(1, 3), sip0, 0x06060606);
                    } else {
                        pout = blend(pin(0, 1), pin(1, 2), pin(0, 3), sip0, 0x06060606);
                    }
                } else {
                    pout = blend(pin(0, 1), pin(0, 2), pin(0, 3), sip0, 0x05050505);
                }
            } else if (mode == 4) {
                const uint8_t sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = deint(pin(1, 1), pin(1, 3), pin(0, 4), pin(1, 5), pin(1, 7), sip0, 0x06060606);
                    } else {
                        pout = pin(1, 4);
                    }
                } else {
                    if (is_latter_field(ly, tb_order)) {
                        pout = deint(pin(0, 1), pin(0, 3), pin(0, 4), pin(0, 5), pin(0, 7), sip0, 0x05050505);
                    } else {
                        pout = pin(0, 4);
                    }
                }
            }
            pSharedX[SOFFSET(0, ly+j*SYN_BLOCK_Y, 2)] = pout;
        }
        __syncthreads();
        //sharedメモリ内でYUV422->YUV420
        const int sy = (ly << 1) - (ly & 1);
        pShared = pSharedX + SOFFSET(0,sy,2);
        psOut[0] = (Type)(lerp(pShared[SOFFSET(0,0,0)], pShared[SOFFSET(0,2,0)], (ly & 1) ? 0.75f : 0.25f) * (float)(1<<(8*sizeof(Type))) + 0.5f);
    }
    __syncthreads();
    //s_outに出力したものをメモリに書き出す
    if (imgx < (src_width >> 1) && imgy < (src_height >> 1)) {
        *dst = *(Type4 *)(&s_out[ly][lx << 2]);
    }
#undef SHARED_C_Y
#undef SHARED_C_XY
#undef SOFFSET
#undef PREREAD_Y
}

template<typename Type8, int mode>
__inline__ __device__
void set_y_h_pos(const int imgx, const int y_h_center,  int height, const int src_pitch, int& y_h1_pos, int& y_h2_pos, int& y_h3_pos, int& y_h4_pos, int& y_h5_pos, int& y_h6_pos, int& y_h7_pos, int& y_h8_pos) {
    if (mode == 4) {
        y_h4_pos = y_h_center * src_pitch + imgx * sizeof(Type8);
        y_h3_pos = y_h4_pos + ((y_h_center - 1 >= 0) ? -src_pitch : src_pitch);
        y_h2_pos = y_h3_pos + ((y_h_center - 2 >= 0) ? -src_pitch : src_pitch);
        y_h1_pos = y_h2_pos + ((y_h_center - 3 >= 0) ? -src_pitch : src_pitch);
        y_h5_pos = y_h4_pos + ((y_h_center < height - 1) ? src_pitch : -src_pitch);
        y_h6_pos = y_h5_pos + ((y_h_center < height - 2) ? src_pitch : -src_pitch);
        y_h7_pos = y_h6_pos + ((y_h_center < height - 3) ? src_pitch : -src_pitch);
        y_h8_pos = y_h7_pos + ((y_h_center < height - 4) ? src_pitch : -src_pitch);
    } else {
        y_h2_pos = y_h_center * src_pitch + imgx * sizeof(Type8);
        y_h1_pos = y_h2_pos + ((y_h_center - 1 >= 0) ? -src_pitch : src_pitch);
        y_h3_pos = y_h2_pos + ((y_h_center < height - 1) ? src_pitch : -src_pitch);
        y_h4_pos = y_h3_pos + ((y_h_center < height - 2) ? src_pitch : -src_pitch);
    }
}

template<typename Type, typename Type4, typename Type8, int mode>
__global__ void kernel_synthesize_mode_1234_yuv420(
    uint8_t *__restrict__ dstY,
    uint8_t *__restrict__ dstU,
    uint8_t *__restrict__ dstV,
    const uint8_t *__restrict__ p0,
    const uint8_t *__restrict__ p1,
    const uint8_t *__restrict__ sip,
    cudaTextureObject_t src_u0_0,
    cudaTextureObject_t src_u0_1,
    cudaTextureObject_t src_u1_0,
    cudaTextureObject_t src_u1_1,
    cudaTextureObject_t src_v0_0,
    cudaTextureObject_t src_v0_1,
    cudaTextureObject_t src_v1_0,
    cudaTextureObject_t src_v1_1,
    const int width, const int height,
    const int src_pitch_y, const int dst_pitch_y, const int dst_pitch_uv, const int sip_pitch,
    const int tb_order, const uint8_t status) {
    const int lx = threadIdx.x; //スレッド数=SYN_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=SYN_BLOCK_Y
    const int imgx = blockIdx.x * SYN_BLOCK_INT_X /*blockDim.x*/ + lx; //グローバルスレッドID x
    const int imgy = blockIdx.y * SYN_BLOCK_Y     /*blockDim.y*/ + ly; //グローバルスレッドID y

    if (imgx * 8 < width && imgy < (height >> 1)) {
        //y
        const int y_h_center = imgy << 1;

        int y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos;
        set_y_h_pos<Type8, mode>(imgx, y_h_center, height, src_pitch_y, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        uint8_t *dst_y       = dstY +  y_h_center * dst_pitch_y + imgx * sizeof(Type8);
        const uint8_t *sip_y = sip  + (y_h_center * sip_pitch   + imgx * sizeof(uint8_t) * 8);
        proc_y<Type8, mode>(dst_y +           0, p0, p1, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y<Type8, mode>(dst_y + dst_pitch_y, p0, p1, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);
    }
    {
        //u
        const int uv_pos_dst = imgy * dst_pitch_uv + imgx * sizeof(Type4);
        Type4 *dst_u = (Type4 *)(dstU + uv_pos_dst);
        proc_uv<Type, Type4, mode>(dst_u, src_u0_0, src_u0_1, src_u1_0, src_u1_1, sip, sip_pitch, width, height, imgx, imgy,
            lx, ly, tb_order, status);

        //v
        Type4 *dst_v = (Type4 *)((uint8_t *)dstV + uv_pos_dst);
        proc_uv<Type, Type4, mode>(dst_v, src_v0_0, src_v0_1, src_v1_0, src_v1_1, sip, sip_pitch, width, height, imgx, imgy,
            lx, ly, tb_order, status);
    }
}

template<typename Type, typename Type4, typename Type8, int mode>
__global__ void kernel_synthesize_mode_1234_yuv444(
    uint8_t *__restrict__ dstY,
    uint8_t *__restrict__ dstU,
    uint8_t *__restrict__ dstV,
    const uint8_t *__restrict__ p0Y,
    const uint8_t *__restrict__ p0U,
    const uint8_t *__restrict__ p0V,
    const uint8_t *__restrict__ p1Y,
    const uint8_t *__restrict__ p1U,
    const uint8_t *__restrict__ p1V,
    const uint8_t *__restrict__ sip,
    const int width, const int height,
    const int src_pitch, const int dst_pitch, const int sip_pitch,
    const int tb_order, const uint8_t status) {
    const int lx = threadIdx.x; //スレッド数=SYN_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=SYN_BLOCK_Y
    const int imgx = blockIdx.x * SYN_BLOCK_INT_X /*blockDim.x*/ + lx; //グローバルスレッドID x
    const int imgy = blockIdx.y * SYN_BLOCK_Y     /*blockDim.y*/ + ly; //グローバルスレッドID y

    if (imgx * 8 < width && imgy < (height >> 1)) {
        //y
        const int y_h_center = imgy << 1;

        int y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos;
        set_y_h_pos<Type8, mode>(imgx, y_h_center, height, dst_pitch, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        const int pix_offset = y_h_center * dst_pitch + imgx * sizeof(Type8);
        uint8_t *dst_y = dstY + pix_offset;
        uint8_t *dst_u = dstU + pix_offset;
        uint8_t *dst_v = dstV + pix_offset;
        const uint8_t *sip_y = sip  + (y_h_center * sip_pitch + imgx * sizeof(uint8_t) * 8);

        proc_y<Type8, mode>(dst_y +         0, p0Y, p1Y, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y<Type8, mode>(dst_y + dst_pitch, p0Y, p1Y, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        proc_y<Type8, mode>(dst_u +         0, p0U, p1U, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y<Type8, mode>(dst_u + dst_pitch, p0U, p1U, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        proc_y<Type8, mode>(dst_v +         0, p0V, p1V, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y<Type8, mode>(dst_v + dst_pitch, p0V, p1V, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);
    }
}

template<typename Type4, typename Type8, bool yuv420>
__global__ void kernel_synthesize_mode_0(
    uint8_t *__restrict__ dstY,
    uint8_t *__restrict__ dstU,
    uint8_t *__restrict__ dstV,
    const uint8_t *__restrict__ p0Y,
    const uint8_t *__restrict__ p0U,
    const uint8_t *__restrict__ p0V,
    const uint8_t *__restrict__ p1Y,
    const uint8_t *__restrict__ p1U,
    const uint8_t *__restrict__ p1V,
    const int width, const int height,
    const int dst_pitch_y, const int dst_pitch_uv,
    const int src_pitch_y, const int src_pitch_uv,
    const int tb_order, const uint8_t status) {
    const int lx = threadIdx.x; //スレッド数=SYN_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=SYN_BLOCK_Y
    const int imgx = blockIdx.x * SYN_BLOCK_INT_X /*blockDim.x*/ + lx;
    const int imgy = blockIdx.y * SYN_BLOCK_Y     /*blockDim.y*/ + ly;

    if (imgx * 8 < width) {
        const uint8_t *srcY = (is_latter_field(ly, tb_order) & (status & AFS_FLAG_SHIFT0)) ? p1Y : p0Y;
        const uint8_t *srcU = (is_latter_field(ly, tb_order) & (status & AFS_FLAG_SHIFT0)) ? p1U : p0U;
        const uint8_t *srcV = (is_latter_field(ly, tb_order) & (status & AFS_FLAG_SHIFT0)) ? p1V : p0V;
        {
            //y
            const int y_line = (blockIdx.y * SYN_BLOCK_Y * 2) + ly;
            Type8       *dst_y = (      Type8 *)(dstY + y_line * dst_pitch_y + imgx * sizeof(Type8));
            const Type8 *src_y = (const Type8 *)(srcY + y_line * src_pitch_y + imgx * sizeof(Type8));
            Type8 *dst_u, *dst_v;
            const Type8 *src_u, *src_v;
            if (y_line < height) {
                dst_y[0] = src_y[0];
                if (!yuv420) {
                    dst_u = (Type8 *)(dstU + y_line * dst_pitch_uv + imgx * sizeof(Type8));
                    dst_v = (Type8 *)(dstV + y_line * dst_pitch_uv + imgx * sizeof(Type8));
                    src_u = (const Type8 *)(srcU + y_line * src_pitch_uv + imgx * sizeof(Type8));
                    src_v = (const Type8 *)(srcV + y_line * src_pitch_uv + imgx * sizeof(Type8));
                    dst_u[0] = src_u[0];
                    dst_v[0] = src_v[0];
                }
            }
            if (y_line + SYN_BLOCK_Y < height) {
                dst_y = (      Type8 *)((      uint8_t *)dst_y + dst_pitch_y * SYN_BLOCK_Y);
                src_y = (const Type8 *)((const uint8_t *)src_y + src_pitch_y * SYN_BLOCK_Y);
                dst_y[0] = src_y[0];
                if (!yuv420) {
                    dst_u = (      Type8 *)((      uint8_t *)dst_u + dst_pitch_uv * SYN_BLOCK_Y);
                    dst_v = (      Type8 *)((      uint8_t *)dst_v + dst_pitch_uv * SYN_BLOCK_Y);
                    src_u = (const Type8 *)((const uint8_t *)src_u + src_pitch_uv * SYN_BLOCK_Y);
                    src_v = (const Type8 *)((const uint8_t *)src_v + src_pitch_uv * SYN_BLOCK_Y);
                    dst_u[0] = src_u[0];
                    dst_v[0] = src_v[0];
                }
            }
        }

        if (yuv420 && ((imgy << 1) < height)) {
            //u
            const int uv_pos_dst = imgy * dst_pitch_uv + imgx * sizeof(Type4);
            const int uv_pos_src = imgy * src_pitch_uv + imgx * sizeof(Type4);
            Type4 *dst_u = (Type4 *)(dstU + uv_pos_dst);
            Type4 *dst_v = (Type4 *)(dstV + uv_pos_dst);
            const Type4 *src_u = (const Type4 *)(srcU + uv_pos_src);
            const Type4 *src_v = (const Type4 *)(srcV + uv_pos_dst);
            dst_u[0] = src_u[0];
            dst_v[0] = src_v[0];
        }
    }
}

enum {
    TUNE_COLOR_BLACK = 0,
    TUNE_COLOR_GREY,
    TUNE_COLOR_BLUE,
    TUNE_COLOR_LIGHT_BLUE,
};

__device__ __inline__
int synthesize_mode_tune_select_color(const uint8_t sip, const uint8_t status) {
    int ret = 0;
    if (status & AFS_FLAG_SHIFT0) {
        if (!(sip & 0x06))
            ret = TUNE_COLOR_LIGHT_BLUE;
        else if (~sip & 0x02)
            ret = TUNE_COLOR_GREY;
        else if (~sip & 0x04)
            ret = TUNE_COLOR_BLUE;
        else
            ret = TUNE_COLOR_BLACK;
    } else {
        if (!(sip & 0x05))
            ret = TUNE_COLOR_LIGHT_BLUE;
        else if (~sip & 0x01)
            ret = TUNE_COLOR_GREY;
        else if (~sip & 0x04)
            ret = TUNE_COLOR_BLUE;
        else
            ret = TUNE_COLOR_BLACK;
    }
    return ret;
}

template<typename Type, typename Type2, bool yuv420>
__global__ void kernel_synthesize_mode_tune(
    uint8_t *__restrict__ dstY,
    uint8_t *__restrict__ dstU,
    uint8_t *__restrict__ dstV,
    const uint8_t *__restrict__ sip,
    const int width, const int height,
    const int dst_pitch_y, const int dst_pitch_uv,
    const int sip_pitch, const int bit_depth,
    const int tb_order, const uint8_t status) {
    const int lx = threadIdx.x; //スレッド数=SYN_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=SYN_BLOCK_Y
    const int imgc_x = blockIdx.x * blockDim.x + lx;
    const int imgc_y = blockIdx.y * blockDim.y + ly;
    const int imgy_x = imgc_x << 1;
    const int imgy_y = imgc_y << 1;
    static const int YUY2_COLOR[4][3] = {
        {  16,  128, 128 },
        {  98,  128, 128 },
        {  41,  240, 110 },
        { 169,  166,  16 }
    };

    if (imgy_x < width && imgy_y < height) {
        sip                  += imgy_y * sip_pitch   + imgy_x * sizeof(uint8_t);
        uint8_t *dst_y = dstY + imgy_y * dst_pitch_y + imgy_x * sizeof(Type);

        uchar2 sip2 = *(uchar2 *)sip;
        const int c00 = synthesize_mode_tune_select_color(sip2.x, status);
        const int c01 = synthesize_mode_tune_select_color(sip2.y, status);
        sip2 = *(uchar2 *)(sip + sip_pitch);
        const int c10 = synthesize_mode_tune_select_color(sip2.x, status);
        const int c11 = synthesize_mode_tune_select_color(sip2.y, status);

        Type2 dst_y2;
        dst_y2.x = (Type)(YUY2_COLOR[c00][0] << (bit_depth - 8));
        dst_y2.y = (Type)(YUY2_COLOR[c01][0] << (bit_depth - 8));
        *(Type2 *)dst_y = dst_y2;
        dst_y2.x = (Type)(YUY2_COLOR[c10][0] << (bit_depth - 8));
        dst_y2.y = (Type)(YUY2_COLOR[c11][0] << (bit_depth - 8));
        *(Type2 *)(dst_y + dst_pitch_y) = dst_y2;

        if (yuv420) {
            uint8_t *dst_u = dstU + imgc_y * dst_pitch_uv + imgc_x * sizeof(Type);
            uint8_t *dst_v = dstV + imgc_y * dst_pitch_uv + imgc_x * sizeof(Type);
            *(Type *)dst_u = (Type)(((YUY2_COLOR[c00][1] + YUY2_COLOR[c01][1] + YUY2_COLOR[c10][1] + YUY2_COLOR[c11][1] + 2) << (bit_depth - 8)) >> 2);
            *(Type *)dst_v = (Type)(((YUY2_COLOR[c00][2] + YUY2_COLOR[c01][2] + YUY2_COLOR[c10][2] + YUY2_COLOR[c11][2] + 2) << (bit_depth - 8)) >> 2);
        } else {
            uint8_t *dst_u = dstU + imgy_y * dst_pitch_uv + imgy_x * sizeof(Type);
            uint8_t *dst_v = dstV + imgy_y * dst_pitch_uv + imgy_x * sizeof(Type);

            Type2 dst_u2;
            dst_u2.x = (Type)(YUY2_COLOR[c00][1] << (bit_depth - 8));
            dst_u2.y = (Type)(YUY2_COLOR[c01][1] << (bit_depth - 8));
            *(Type2 *)dst_u = dst_u2;
            dst_u2.x = (Type)(YUY2_COLOR[c10][1] << (bit_depth - 8));
            dst_u2.y = (Type)(YUY2_COLOR[c11][1] << (bit_depth - 8));
            *(Type2 *)(dst_u + dst_pitch_uv) = dst_u2;

            Type2 dst_v2;
            dst_v2.x = (Type)(YUY2_COLOR[c00][2] << (bit_depth - 8));
            dst_v2.y = (Type)(YUY2_COLOR[c01][2] << (bit_depth - 8));
            *(Type2 *)dst_v = dst_v2;
            dst_v2.x = (Type)(YUY2_COLOR[c10][2] << (bit_depth - 8));
            dst_v2.y = (Type)(YUY2_COLOR[c11][2] << (bit_depth - 8));
            *(Type2 *)(dst_v + dst_pitch_uv) = dst_v2;
        }
    }
}

template<typename Type>
cudaError_t textureCreateSynthesize(cudaTextureObject_t& tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
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

template<typename Type, typename Type2, typename Type4, typename Type8, int mode, bool yuv420>
cudaError_t run_synthesize(FrameInfo *pFrameOut,
    const FrameInfo *pFrame0, const FrameInfo *pFrame1, uint8_t *sip, const int sipPitch,
    const int tb_order, const uint8_t status, const RGY_CSP csp,
    cudaStream_t stream) {
    auto cudaerr = cudaSuccess;
    auto pDstY = getPlane(pFrameOut, RGY_PLANE_Y);
    auto pDstU = getPlane(pFrameOut, RGY_PLANE_U);
    auto pDstV = getPlane(pFrameOut, RGY_PLANE_V);
    const auto p0Y = getPlane(pFrame0, RGY_PLANE_Y);
    const auto p0U = getPlane(pFrame0, RGY_PLANE_U);
    const auto p0V = getPlane(pFrame0, RGY_PLANE_V);
    const auto p1Y = getPlane(pFrame1, RGY_PLANE_Y);
    const auto p1U = getPlane(pFrame1, RGY_PLANE_U);
    const auto p1V = getPlane(pFrame1, RGY_PLANE_V);

    if (   p0Y.width != p1Y.width || p0Y.height != p1Y.height
        || p0U.width != p1U.width || p0U.height != p1U.height
        || p0V.width != p1V.width || p0V.height != p1V.height
        || p0Y.pitch != p1Y.pitch || p0U.pitch  != p1U.pitch) {
        return cudaErrorUnknown;
    }
    if (   pDstU.pitch != pDstV.pitch
        || p0U.pitch != p0V.pitch
        || p1U.pitch != p1V.pitch) {
        return cudaErrorUnknown;
    }
    if (!yuv420) {
        if (   pDstY.pitch != pDstU.pitch
            || p0Y.pitch != p0U.pitch
            || p1Y.pitch != p1U.pitch) {
            return cudaErrorUnknown;
        }
    }

    if (mode < 0) {
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 2), divCeil(pDstY.height, blockSize.y * 2));

        kernel_synthesize_mode_tune<Type, Type2, yuv420><<<gridSize, blockSize, 0, stream>>>(
            pDstY.ptr, pDstU.ptr, pDstV.ptr, sip,
            pDstY.width, pDstY.height, pDstY.pitch, pDstU.pitch, sipPitch, RGY_CSP_BIT_DEPTH[csp],
            tb_order, status);
    } else if (mode == 0) {
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 8), divCeil(pDstY.height, blockSize.y * 2));

        kernel_synthesize_mode_0<Type4, Type8, yuv420><<<gridSize, blockSize, 0, stream>>>(
            pDstY.ptr, pDstU.ptr, pDstV.ptr,
            p0Y.ptr, p0U.ptr, p0V.ptr,
            p1Y.ptr, p1U.ptr, p1V.ptr,
            pDstY.width, pDstY.height,
            pDstY.pitch, pDstU.pitch,
            p0Y.pitch, p0U.pitch,
            tb_order, status);
    } else {
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 8), divCeil(pDstY.height, blockSize.y * 2));

        if (yuv420) {
            cudaTextureObject_t texP0U0, texP0U1, texP0V0, texP0V1, texP1U0, texP1U1, texP1V0,texP1V1;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP0U0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0U.ptr + p0U.pitch * 0, p0U.pitch * 2, p0U.width, p0U.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP0U1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0U.ptr + p0U.pitch * 1, p0U.pitch * 2, p0U.width, p0U.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP0V0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0V.ptr + p0V.pitch * 0, p0V.pitch * 2, p0V.width, p0V.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP0V1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p0V.ptr + p0V.pitch * 1, p0V.pitch * 2, p0V.width, p0V.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP1U0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1U.ptr + p1U.pitch * 0, p1U.pitch * 2, p1U.width, p1U.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP1U1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1U.ptr + p1U.pitch * 1, p1U.pitch * 2, p1U.width, p1U.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP1V0, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1V.ptr + p1V.pitch * 0, p1V.pitch * 2, p1V.width, p1V.height >> 1))) return cudaerr;
            if (cudaSuccess != (cudaerr = textureCreateSynthesize<Type>(texP1V1, cudaFilterModeLinear, cudaReadModeNormalizedFloat, p1V.ptr + p1V.pitch * 1, p1V.pitch * 2, p1V.width, p1V.height >> 1))) return cudaerr;

            kernel_synthesize_mode_1234_yuv420<Type, Type4, Type8, mode><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr, pDstU.ptr, pDstV.ptr,
                p0Y.ptr, p1Y.ptr, sip,
                texP0U0, texP0U1, texP1U0, texP1U1,
                texP0V0, texP0V1, texP1V0, texP1V1,
                p0Y.width, p0Y.height, p0Y.pitch, pDstY.pitch, pDstU.pitch, sipPitch,
                tb_order, status);
            cudaerr = cudaGetLastError();
            if (cudaerr != cudaSuccess) {
                return cudaerr;
            }

            cudaDestroyTextureObject(texP0U0);
            cudaDestroyTextureObject(texP0U1);
            cudaDestroyTextureObject(texP0V0);
            cudaDestroyTextureObject(texP0V1);
            cudaDestroyTextureObject(texP1U0);
            cudaDestroyTextureObject(texP1U1);
            cudaDestroyTextureObject(texP1V0);
            cudaDestroyTextureObject(texP1V1);
        } else {
            kernel_synthesize_mode_1234_yuv444<Type, Type4, Type8, mode><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr, pDstU.ptr, pDstV.ptr,
                p0Y.ptr, p0U.ptr, p0V.ptr,
                p1Y.ptr, p1U.ptr, p1V.ptr,
                sip,
                p0Y.width, p0Y.height, p0Y.pitch, pDstY.pitch, sipPitch,
                tb_order, status);
        }
    }
    return cudaGetLastError();
}

cudaError_t NVEncFilterAfs::synthesize(int iframe, CUFrameBuf *pOut, CUFrameBuf *p0, CUFrameBuf *p1, AFS_STRIPE_DATA *sip, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream) {
    struct synthesize_func {
        decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2, 3, true>)* func[6];
        synthesize_func(
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2, -1, true>)* mode_tune,
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2,  0, true>)* mode_0,
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2,  1, true>)* mode_1,
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2,  2, true>)* mode_2,
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2,  3, true>)* mode_3,
            decltype(run_synthesize<uint8_t, uchar2, uint32_t, uint2,  4, true>)* mode_4) {
            func[0] = mode_tune;
            func[1] = mode_0;
            func[2] = mode_1;
            func[3] = mode_2;
            func[4] = mode_3;
            func[5] = mode_4;
        };
    };

    static const std::map<RGY_CSP, synthesize_func> synthesize_func_list = {
        { RGY_CSP_YV12, synthesize_func(
            run_synthesize<uint8_t, uchar2, uint32_t, uint2, -1, true>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  0, true>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  1, true>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  2, true>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  3, true>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  4, true>) },
        { RGY_CSP_YV12_16, synthesize_func(
            run_synthesize<uint16_t, ushort2, uint64_t, uint4, -1, true>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  0, true>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  1, true>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  2, true>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  3, true>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  4, true>) },
        { RGY_CSP_YUV444, synthesize_func(
            run_synthesize<uint8_t, uchar2, uint32_t, uint2, -1, false>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  0, false>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  1, false>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  2, false>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  3, false>,
            run_synthesize<uint8_t, uchar2, uint32_t, uint2,  4, false>) },
        { RGY_CSP_YUV444_16, synthesize_func(
            run_synthesize<uint16_t, ushort2, uint64_t, uint4, -1, false>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  0, false>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  1, false>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  2, false>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  3, false>,
            run_synthesize<uint16_t, ushort2, uint64_t, uint4,  4, false>) }
    };
    if (synthesize_func_list.count(pAfsPrm->frameIn.csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for afs_synthesize: %s\n"), RGY_CSP_NAMES[pAfsPrm->frameIn.csp]);
        return cudaErrorNotSupported;
    }
    int mode = pAfsPrm->afs.analyze;
    if (pAfsPrm->afs.tune) {
        mode = -1;
    }
    auto cudaerr = synthesize_func_list.at(pAfsPrm->frameIn.csp).func[mode+1](
        &pOut->frame, &p0->frame, &p1->frame, sip->map.frame.ptr, sip->map.frame.pitch,
        pAfsPrm->afs.tb_order, m_status[iframe], pOut->frame.csp, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaSuccess;
}
