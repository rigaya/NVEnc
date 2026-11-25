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
uint32_t blend(int src1, int src2, int src3, uint32_t flag, uint32_t mask) {
    int tmp = (src1 + src3 + src2 + src2 + 2) >> 2;
    return (uint32_t)(((flag & mask) == 0) ? tmp : src2);
}

__device__ __inline__
uint32_t mie_inter(int src1, int src2, int src3, int src4) {
    return (uint32_t)((src1 + src2 + src3 + src4 + 2) >> 2);
}

__device__ __inline__
uint32_t mie_spot(int src1, int src2, int src3, int src4, int src_spot) {
    return __urhadd(mie_inter(src1, src2, src3, src4), src_spot);
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
    pout.w = mie_spot2(src1.w, src2.w, src3.w, src4.w, src_spot.w);
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

template<typename Type, typename Type4>
__device__ __forceinline__
void load_uv_422(int (&out)[4], const uint8_t *src, int pitch, int lastFieldRow, int x, int row) {
    const int field = row & 1;
    const int base = (row - 2) >> 2;
    const int weight = 7 - ((row & 3) << 1);
    const int row0 = 2 * clamp(base, 0, lastFieldRow) + field;
    const int row1 = 2 * clamp(base + 1, 0, lastFieldRow) + field;
    const Type4 a = reinterpret_cast<const Type4 *>(src + row0 * pitch)[x];
    const Type4 b = reinterpret_cast<const Type4 *>(src + row1 * pitch)[x];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int av = static_cast<Type>(a >> (i * sizeof(Type) * 8));
        const int bv = static_cast<Type>(b >> (i * sizeof(Type) * 8));
        out[i] = (8 - weight) * av + weight * bv;
    }
}

template<typename Type, typename Type4, int mode>
__device__ __forceinline__
void synth_uv_422(int (&out)[4], const uint8_t *p0, const uint8_t *p1, int pitch,
    int lastFieldRow, int x, int row, uint32_t flags, bool latter, bool shift) {
    const uint8_t *center = (shift && latter) ? p1 : p0;
    const uint8_t *neighbor = (shift && !latter) ? p1 : p0;
    int middle[4];
    load_uv_422<Type, Type4>(middle, center, pitch, lastFieldRow, x, row);
    if (mode == 1) {
        int top[4], bottom[4];
        load_uv_422<Type, Type4>(top, neighbor, pitch, lastFieldRow, x, row - 1);
        load_uv_422<Type, Type4>(bottom, neighbor, pitch, lastFieldRow, x, row + 1);
        if (shift != latter) {
            int other[4];
            load_uv_422<Type, Type4>(other, p1, pitch, lastFieldRow, x, row);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[i] = mie_inter(top[i], middle[i], bottom[i], other[i]);
            }
        } else {
            int otherTop[4], otherBottom[4];
            load_uv_422<Type, Type4>(otherTop, p1, pitch, lastFieldRow, x, row - 1);
            load_uv_422<Type, Type4>(otherBottom, p1, pitch, lastFieldRow, x, row + 1);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[i] = mie_spot(top[i], bottom[i], otherTop[i], otherBottom[i], middle[i]);
            }
        }
    } else if (mode == 4) {
        if (shift != latter) {
            int top[4], bottom[4], farTop[4], farBottom[4];
            load_uv_422<Type, Type4>(top, neighbor, pitch, lastFieldRow, x, row - 1);
            load_uv_422<Type, Type4>(bottom, neighbor, pitch, lastFieldRow, x, row + 1);
            load_uv_422<Type, Type4>(farTop, neighbor, pitch, lastFieldRow, x, row - 3);
            load_uv_422<Type, Type4>(farBottom, neighbor, pitch, lastFieldRow, x, row + 3);
            const uint32_t mask = shift ? 0x06 : 0x05;
            const int maxScaled = ((1 << (8 * sizeof(Type))) - 1) * 8;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int filtered = (9 * (top[i] + bottom[i]) - (farTop[i] + farBottom[i]) + 8) >> 4;
                out[i] = ((flags >> (8 * i)) & mask) ? middle[i] : clamp(filtered, 0, maxScaled);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[i] = middle[i];
            }
        }
    } else {
        int top[4], bottom[4];
        load_uv_422<Type, Type4>(top, neighbor, pitch, lastFieldRow, x, row - 1);
        load_uv_422<Type, Type4>(bottom, neighbor, pitch, lastFieldRow, x, row + 1);
        const uint32_t mask = (mode == 2) ? (shift ? 0x02 : 0x01) : (shift ? 0x06 : 0x05);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            out[i] = blend(top[i], middle[i], bottom[i], flags >> (8 * i), mask);
        }
    }
}

__device__ __forceinline__
uint32_t load_uv_flags(const uint8_t *sip, int pitch, int x, int row, int height) {
    if (row >= height) {
        return 0;
    }
    const uint2 flags = reinterpret_cast<const uint2 *>(sip + row * pitch)[x];
    return __byte_perm(flags.x, flags.y, 0x6420);
}

template<typename Type, typename Type4, int mode>
__device__ __forceinline__
Type4 proc_uv(const uint8_t *p0, const uint8_t *p1, int pitch, int lastFieldRow, int x, int row,
    bool odd, uint32_t flags0, uint32_t flags1, bool latter, bool shift) {
    // 420→422の補間値は8倍の整数で持ち回り、共有メモリやブロック同期を省く。
    int s0[4], s1[4];
    synth_uv_422<Type, Type4, mode>(s0, p0, p1, pitch, lastFieldRow, x, row, flags0, latter, shift);
    synth_uv_422<Type, Type4, mode>(s1, p0, p1, pitch, lastFieldRow, x, row + 2, flags1, latter, shift);
    Type4 packed = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int v = odd ? (s0[i] + 3 * s1[i]) : (3 * s0[i] + s1[i]);
        // 整数の画素値に戻すため、旧経路の正規化分母と復元倍率の差による偏りがない。
        packed |= static_cast<Type4>((v + 16) >> 5) << (i * sizeof(Type) * 8);
    }
    return packed;
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
    const uint8_t *__restrict__ p0U,
    const uint8_t *__restrict__ p1U,
    const uint8_t *__restrict__ p0V,
    const uint8_t *__restrict__ p1V,
    const int width, const int height, const int uv_height,
    const int src_pitch_y, const int src_pitch_uv, const int dst_pitch_y, const int dst_pitch_uv, const int sip_pitch,
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
    if (imgx * 8 < width && imgy < uv_height) {
        const bool odd = imgy & 1;
        const int row = 2 * imgy - odd;
        const int lastFieldRow = (uv_height >> 1) - 1;
        const uint32_t flags0 = mode >= 2 ? load_uv_flags(sip, sip_pitch, imgx, row, height) : 0;
        const uint32_t flags1 = mode >= 2 ? load_uv_flags(sip, sip_pitch, imgx, row + 2, height) : 0;
        const bool latter = is_latter_field(row, tb_order);
        const bool shift = (status & AFS_FLAG_SHIFT0) != 0;
        const int uv_pos_dst = imgy * dst_pitch_uv + imgx * sizeof(Type4);
        *reinterpret_cast<Type4 *>(dstU + uv_pos_dst) = proc_uv<Type, Type4, mode>(
            p0U, p1U, src_pitch_uv, lastFieldRow, imgx, row, odd, flags0, flags1, latter, shift);
        *reinterpret_cast<Type4 *>(dstV + uv_pos_dst) = proc_uv<Type, Type4, mode>(
            p0V, p1V, src_pitch_uv, lastFieldRow, imgx, row, odd, flags0, flags1, latter, shift);
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
    TUNE_COLOR_RED,
    TUNE_COLOR_LIGHT_RED,
    TUNE_COLOR_PURPLE,
    TUNE_COLOR_LIGHT_PURPLE,
    TUNE_COLOR_MAX,
};

template<bool showScanData>
__device__ __inline__
int synthesize_mode_tune_select_color(const uint8_t sip, const uint8_t status, const uint8_t tune_mode) {
    int ret = 0;
    if (showScanData) {
    //      7       6         5        4        3        2        1       0
    // | motion  |         non-shift        | motion  |          shift          |
    // |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
        if (tune_mode == AFS_TUNE_MODE_ANALYZE_SHIFT_ALL
            || tune_mode == AFS_TUNE_MODE_ANALYZE_SHIFT_Y
            || tune_mode == AFS_TUNE_MODE_ANALYZE_SHIFT_U
            || tune_mode == AFS_TUNE_MODE_ANALYZE_SHIFT_V) {
            auto tmp = (sip & 0x03) | ((~sip) & 0x40);
            if (tmp == 0x01) {
                ret = TUNE_COLOR_GREY;
            } else if (tmp == 0x02) {
                ret = TUNE_COLOR_RED;
            } else if (tmp == 0x40) {
                ret = TUNE_COLOR_BLUE;
            } else if (tmp == (0x01 + 0x02)) {
                ret = TUNE_COLOR_LIGHT_RED;
            } else if (tmp == (0x01 + 0x40)) {
                ret = TUNE_COLOR_LIGHT_BLUE;
            } else if (tmp == (0x02 + 0x40)) {
                ret = TUNE_COLOR_PURPLE;
            } else if (tmp == (0x01 + 0x02 + 0x40)) {
                ret = TUNE_COLOR_LIGHT_PURPLE;
            }
        } else if (tune_mode == AFS_TUNE_MODE_ANALYZE_NONSHIFT_ALL
            || tune_mode == AFS_TUNE_MODE_ANALYZE_NONSHIFT_Y
            || tune_mode == AFS_TUNE_MODE_ANALYZE_NONSHIFT_U
            || tune_mode == AFS_TUNE_MODE_ANALYZE_NONSHIFT_V) {
            auto tmp = (sip & 0x30) | ((~sip) & 0x04);
            if (tmp == 0x10) {
                ret = TUNE_COLOR_GREY;
            } else if (tmp == 0x20) {
                ret = TUNE_COLOR_RED;
            } else if (tmp == 0x04) {
                ret = TUNE_COLOR_BLUE;
            } else if (tmp == (0x10+0x20)) {
                ret = TUNE_COLOR_LIGHT_RED;
            } else if (tmp == (0x10+0x04)) {
                ret = TUNE_COLOR_LIGHT_BLUE;
            } else if (tmp == (0x20+0x04)) {
                ret = TUNE_COLOR_PURPLE;
            } else if (tmp == (0x10+0x20+0x04)) {
                ret = TUNE_COLOR_LIGHT_PURPLE;
            }
        }
    } else if (status & AFS_FLAG_SHIFT0) {
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

template<typename Type, typename Type2, bool yuv420, bool showScanData>
__global__ void kernel_synthesize_mode_tune(
    uint8_t *__restrict__ dstY,
    uint8_t *__restrict__ dstU,
    uint8_t *__restrict__ dstV,
    const uint8_t *__restrict__ sip,
    const int width, const int height,
    const int dst_pitch_y, const int dst_pitch_uv,
    const int sip_pitch, const int bit_depth,
    const uint8_t tune_mode, const uint8_t status) {
    const int lx = threadIdx.x; //スレッド数=SYN_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=SYN_BLOCK_Y
    const int imgc_x = blockIdx.x * blockDim.x + lx;
    const int imgc_y = blockIdx.y * blockDim.y + ly;
    const int imgy_x = imgc_x << 1;
    const int imgy_y = imgc_y << 1;
    static const int YUY2_COLOR[TUNE_COLOR_MAX][3] = {
        {  16,  128, 128 }, // TUNE_COLOR_BLACK
        {  98,  128, 128 }, // TUNE_COLOR_GREY
        {  41,  240, 110 }, // TUNE_COLOR_BLUE
        { 169,  166,  16 }, // TUNE_COLOR_LIGHT_BLUE
        {  95,   98, 216 }, // TUNE_COLOR_RED
        { 173,  115, 167 }, // TUNE_COLOR_LIGHT_RED
        {  83,  167, 142 }, // TUNE_COLOR_PURPLE
        { 167,  152, 136 } // TUNE_COLOR_LIGHT_PURPLE
    };

    if (imgy_x < width && imgy_y < height) {
        sip                  += imgy_y * sip_pitch   + imgy_x * sizeof(uint8_t);
        uint8_t *dst_y = dstY + imgy_y * dst_pitch_y + imgy_x * sizeof(Type);

        uchar2 sip2 = *(uchar2 *)sip;
        const int c00 = synthesize_mode_tune_select_color<showScanData>(sip2.x, status, tune_mode);
        const int c01 = synthesize_mode_tune_select_color<showScanData>(sip2.y, status, tune_mode);
        sip2 = *(uchar2 *)(sip + sip_pitch);
        const int c10 = synthesize_mode_tune_select_color<showScanData>(sip2.x, status, tune_mode);
        const int c11 = synthesize_mode_tune_select_color<showScanData>(sip2.y, status, tune_mode);

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

template<typename Type, typename Type2, typename Type4, typename Type8, int mode, bool yuv420>
RGY_ERR run_synthesize(RGYFrameInfo *pFrameOut,
    const RGYFrameInfo *pFrame0, const RGYFrameInfo *pFrame1, uint8_t *sip, const int sipPitch,
    const int tb_order, const uint8_t status, const RGY_CSP csp,
    cudaStream_t stream) {
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
        || p0Y.pitch[0] != p1Y.pitch[0] || p0U.pitch[0] != p1U.pitch[0]) {
        return RGY_ERR_UNSUPPORTED;
    }
    if (   pDstU.pitch[0] != pDstV.pitch[0]
        || p0U.pitch[0] != p0V.pitch[0]
        || p1U.pitch[0] != p1V.pitch[0]) {
        return RGY_ERR_UNSUPPORTED;
    }
    if (!yuv420) {
        if (   pDstY.pitch[0] != pDstU.pitch[0]
            || p0Y.pitch[0] != p0U.pitch[0]
            || p1Y.pitch[0] != p1U.pitch[0]) {
            return RGY_ERR_UNSUPPORTED;
        }
    }

    if (mode < 0) {
        const uint8_t tune_mode = (uint8_t)tb_order; // tune_modeが入っている
        const bool showScanData = tune_mode > AFS_TUNE_MODE_FINAL;
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 2), divCeil(pDstY.height, blockSize.y * 2));
        if (showScanData) {
            kernel_synthesize_mode_tune<Type, Type2, yuv420, true><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr[0], pDstU.ptr[0], pDstV.ptr[0], sip,
                pDstY.width, pDstY.height, pDstY.pitch[0], pDstU.pitch[0], sipPitch, RGY_CSP_BIT_DEPTH[csp],
                tune_mode, status);
        } else {
            kernel_synthesize_mode_tune<Type, Type2, yuv420, false><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr[0], pDstU.ptr[0], pDstV.ptr[0], sip,
                pDstY.width, pDstY.height, pDstY.pitch[0], pDstU.pitch[0], sipPitch, RGY_CSP_BIT_DEPTH[csp],
                tune_mode, status);
        }
    } else if (mode == 0) {
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 8), divCeil(pDstY.height, blockSize.y * 2));

        kernel_synthesize_mode_0<Type4, Type8, yuv420><<<gridSize, blockSize, 0, stream>>>(
            pDstY.ptr[0], pDstU.ptr[0], pDstV.ptr[0],
            p0Y.ptr[0], p0U.ptr[0], p0V.ptr[0],
            p1Y.ptr[0], p1U.ptr[0], p1V.ptr[0],
            pDstY.width, pDstY.height,
            pDstY.pitch[0], pDstU.pitch[0],
            p0Y.pitch[0], p0U.pitch[0],
            tb_order, status);
    } else {
        const dim3 blockSize(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const dim3 gridSize(divCeil(pDstY.width, blockSize.x * 8), divCeil(pDstY.height, blockSize.y * 2));

        if (yuv420) {
            kernel_synthesize_mode_1234_yuv420<Type, Type4, Type8, mode><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr[0], pDstU.ptr[0], pDstV.ptr[0],
                p0Y.ptr[0], p1Y.ptr[0], sip,
                p0U.ptr[0], p1U.ptr[0], p0V.ptr[0], p1V.ptr[0],
                p0Y.width, p0Y.height, p0U.height, p0Y.pitch[0], p0U.pitch[0], pDstY.pitch[0], pDstU.pitch[0], sipPitch,
                tb_order, status);
        } else {
            kernel_synthesize_mode_1234_yuv444<Type, Type4, Type8, mode><<<gridSize, blockSize, 0, stream>>>(
                pDstY.ptr[0], pDstU.ptr[0], pDstV.ptr[0],
                p0Y.ptr[0], p0U.ptr[0], p0V.ptr[0],
                p1Y.ptr[0], p1U.ptr[0], p1V.ptr[0],
                sip,
                p0Y.width, p0Y.height, p0Y.pitch[0], pDstY.pitch[0], sipPitch,
                tb_order, status);
        }
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterAfs::synthesize(int iframe, CUFrameBuf *pOut, CUFrameBuf *p0, CUFrameBuf *p1, AFS_STRIPE_DATA *sip, AFS_SCAN_DATA *sp, const NVEncFilterParamAfs *pAfsPrm, cudaStream_t stream) {
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
        return RGY_ERR_UNSUPPORTED;
    }
    int mode = pAfsPrm->afs.analyze;
    if (pAfsPrm->afs.tune) {
        mode = -1;
    }
    const bool showScanData = pAfsPrm->afs.tune > AFS_TUNE_MODE_FINAL;
    auto sts = synthesize_func_list.at(pAfsPrm->frameIn.csp).func[mode+1](
        &pOut->frame, &p0->frame, &p1->frame,
        (showScanData) ? sp->map.frame.ptr[0] : sip->map.frame.ptr[0],
        (showScanData) ? sp->map.frame.pitch[0] : sip->map.frame.pitch[0],
        (mode < 0) ? pAfsPrm->afs.tune : pAfsPrm->afs.tb_order, m_status[iframe], pOut->frame.csp, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}
