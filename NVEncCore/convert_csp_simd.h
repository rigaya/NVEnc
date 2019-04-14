// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
#ifndef _CONVERT_CSP_SIMD_H_
#define _CONVERT_CSP_SIMD_H_

#include "rgy_simd.h"
#include <cstdint>
#include <cstring>
#include <emmintrin.h> //イントリンシック命令 SSE2
#if USE_SSSE3
#include <tmmintrin.h> //イントリンシック命令 SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //イントリンシック命令 SSE4.1
#endif
#include "convert_csp.h"
#include "convert_const.h"
#include <utility>

static void __forceinline memcpy_sse(uint8_t *dst, const uint8_t *src, int size) {
    if (size < 64) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128 x0, x1, x2, x3;
    const int start_align_diff = (int)((size_t)dst & 15);
    if (start_align_diff) {
        x0 = _mm_loadu_ps((float*)src);
        _mm_storeu_ps((float*)dst, x0);
        dst += 16 - start_align_diff;
        src += 16 - start_align_diff;
    }
    for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_ps((float*)(src +  0));
        x1 = _mm_loadu_ps((float*)(src + 16));
        x2 = _mm_loadu_ps((float*)(src + 32));
        x3 = _mm_loadu_ps((float*)(src + 48));
        _mm_store_ps((float*)(dst +  0), x0);
        _mm_store_ps((float*)(dst + 16), x1);
        _mm_store_ps((float*)(dst + 32), x2);
        _mm_store_ps((float*)(dst + 48), x3);
    }
    uint8_t *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_ps((float*)(src +  0));
    x1 = _mm_loadu_ps((float*)(src + 16));
    x2 = _mm_loadu_ps((float*)(src + 32));
    x3 = _mm_loadu_ps((float*)(src + 48));
    _mm_storeu_ps((float*)(dst_tmp +  0), x0);
    _mm_storeu_ps((float*)(dst_tmp + 16), x1);
    _mm_storeu_ps((float*)(dst_tmp + 32), x2);
    _mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

#define _mm_store_switch_si128(ptr, xmm) ((aligned_store) ? _mm_store_si128(ptr, xmm) : _mm_storeu_si128(ptr, xmm))

#if USE_SSSE3
#define _mm_alignr_epi8_simd(a,b,i) _mm_alignr_epi8(a,b,i)
#else
#define _mm_alignr_epi8_simd(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )
#endif

static __forceinline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
    return _mm_blendv_epi8(a, b, mask);
#else
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}

static __forceinline __m128i _mm_packus_epi32_simd(__m128i a, __m128i b) {
#if USE_SSE41
    return _mm_packus_epi32(a, b);
#else
    alignas(64) static const uint32_t VAL[2][4] = {
        { 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
        { 0x80008000, 0x80008000, 0x80008000, 0x80008000 }
    };
#define LOAD_32BIT_0x8000 _mm_load_si128((__m128i *)VAL[0])
#define LOAD_16BIT_0x8000 _mm_load_si128((__m128i *)VAL[1])
    a = _mm_sub_epi32(a, LOAD_32BIT_0x8000);
    b = _mm_sub_epi32(b, LOAD_32BIT_0x8000);
    a = _mm_packs_epi32(a, b);
    return _mm_add_epi16(a, LOAD_16BIT_0x8000);
#undef LOAD_32BIT_0x8000
#undef LOAD_16BIT_0x8000
#endif
}

#pragma warning (push)
#pragma warning (disable: 4100)
template<bool highbit_depth>
static void __forceinline copy_nv12_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int pixel_size = highbit_depth ? 2 : 1;
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        uint8_t *srcYLine = (uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + src_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy_sse(dstLine, srcYLine, y_width * pixel_size);
        }
    }
}
#pragma warning (pop)

#if USE_SSSE3
alignas(32) static const uint8_t  Array_INTERLACE_WEIGHT[2][32] = {
    {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
    {3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define xC_INTERLACE_WEIGHT(i) _mm_load_si128((__m128i*)Array_INTERLACE_WEIGHT[i])
#endif

static __forceinline void separate_low_up(__m128i& x0_return_lower, __m128i& x1_return_upper) {
    __m128i x4, x5;
    const __m128i xMaskLowByte = _mm_srli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8);
    x4 = _mm_srli_epi16(x0_return_lower, 8);
    x5 = _mm_srli_epi16(x1_return_upper, 8);

    x0_return_lower = _mm_and_si128(x0_return_lower, xMaskLowByte);
    x1_return_upper = _mm_and_si128(x1_return_upper, xMaskLowByte);

    x0_return_lower = _mm_packus_epi16(x0_return_lower, x1_return_upper);
    x1_return_upper = _mm_packus_epi16(x4, x5);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
static void __forceinline convert_yuy2_to_nv12_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstCLine = (uint8_t *)dst[1] + dst_y_pitch_byte * (y_range.start_dst >> 1);
    for (int y = 0; y < y_range.len; y += 2) {
        uint8_t *p = srcLine;
        uint8_t *pw = p + src_y_pitch_byte;
        const int x_fin = width - crop_right - crop_left;
        __m128i x0, x1, x3;
        for (int x = 0; x < x_fin; x += 16, p += 32, pw += 32) {
            //-----------1行目---------------
            x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(p+16));

            separate_low_up(x0, x1);
            x3 = x1;

            _mm_store_si128((__m128i *)(dstYLine + x), x0);
            //-----------1行目終了---------------

            //-----------2行目---------------
            x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(pw+16));

            separate_low_up(x0, x1);

            _mm_store_si128((__m128i *)(dstYLine + dst_y_pitch_byte + x), x0);
            //-----------2行目終了---------------

            x1 = _mm_avg_epu8(x1, x3);
            _mm_store_si128((__m128i *)(dstCLine + x), x1);
        }
        srcLine  += src_y_pitch_byte << 1;
        dstYLine += dst_y_pitch_byte << 1;
        dstCLine += dst_y_pitch_byte;
    }
}

static __forceinline __m128i yuv422_to_420_i_interpolate(__m128i y_up, __m128i y_down, int i) {
    __m128i x0, x1;
#if USE_SSSE3
    x0 = _mm_unpacklo_epi8(y_down, y_up);
    x1 = _mm_unpackhi_epi8(y_down, y_up);
    x0 = _mm_maddubs_epi16(x0, xC_INTERLACE_WEIGHT(i));
    x1 = _mm_maddubs_epi16(x1, xC_INTERLACE_WEIGHT(i));
#else
    __m128i x2, x3, xC[2];
    xC[0] = y_up;
    xC[1] = y_down;
    x0 = _mm_unpacklo_epi8(xC[i], _mm_setzero_si128());
    x1 = _mm_unpackhi_epi8(xC[i], _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, _mm_set1_epi16(3));
    x1 = _mm_mullo_epi16(x1, _mm_set1_epi16(3));
    x2 = _mm_unpacklo_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
    x3 = _mm_unpackhi_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
    x0 = _mm_add_epi16(x0, x2);
    x1 = _mm_add_epi16(x1, x3);
#endif
    x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
    x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
    x0 = _mm_srai_epi16(x0, 2);
    x1 = _mm_srai_epi16(x1, 2);
    x0 = _mm_packus_epi16(x0, x1);
    return x0;
}

static void __forceinline convert_yuy2_to_nv12_i_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstCLine = (uint8_t *)dst[1] + dst_y_pitch_byte * (y_range.start_dst >> 1);
    for (int y = 0; y < y_range.len; y += 4) {
        for (int i = 0; i < 2; i++) {
            uint8_t *p = srcLine;
            uint8_t *pw = p + (src_y_pitch_byte<<1);
            __m128i x0, x1, x3;
            const int x_fin = width - crop_right - crop_left;
            for (int x = 0; x < x_fin; x += 16, p += 32, pw += 32) {
                //-----------    1+i行目   ---------------
                x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(p+16));

                separate_low_up(x0, x1);
                x3 = x1;

                _mm_store_si128((__m128i *)(dstYLine + x), x0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(pw+16));

                separate_low_up(x0, x1);

                _mm_store_si128((__m128i *)(dstYLine + (dst_y_pitch_byte<<1) + x), x0);
                //-----------3+i行目終了---------------
                x0 = yuv422_to_420_i_interpolate(x3, x1, i);

                _mm_store_si128((__m128i *)(dstCLine + x), x0);
            }
            srcLine  += src_y_pitch_byte;
            dstYLine += dst_y_pitch_byte;
            dstCLine += dst_y_pitch_byte;
        }
        srcLine  += src_y_pitch_byte << 1;
        dstYLine += dst_y_pitch_byte << 1;
    }
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<bool uv_only>
static void __forceinline convert_yv12_to_nv12_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy_sse(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        __m128i x0, x1, x2;
        for (int x = crop_left; x < x_fin; x += 32, src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

            x2 = _mm_unpackhi_epi8(x0, x1);
            x0 = _mm_unpacklo_epi8(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 16), x2);
        }
    }
}



static void __forceinline convert_yv12_to_yv12_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        memcpy_sse(dstLine, srcYLine, y_width);
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    for (int i = 1; i < 3; i++) {
        uint8_t *srcLine = (uint8_t *)src[i] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
        dstLine = (uint8_t *)dst[i] + (dst_y_pitch_byte >> 1) * uv_range.start_dst;
        const int copy_length = (width - crop_right - crop_left) >> 1;
        for (int y = 0; y < uv_range.len; y++) {
            memcpy_sse(dstLine, srcLine, copy_length);
            srcLine += src_uv_pitch_byte;
            dstLine += dst_y_pitch_byte >> 1;
        }
    }
}

static void __forceinline convert_yuv422_to_nv16_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        memcpy_sse(dstLine, srcYLine, y_width);
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * y_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    const int uv_fin = height - crop_bottom;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        __m128i x0, x1, x2;
        for (int x = crop_left; x < x_fin; x += 32, src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

            x2 = _mm_unpackhi_epi8(x0, x1);
            x0 = _mm_unpacklo_epi8(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 16), x2);
        }
    }
}

#if USE_SSSE3
static void __forceinline convert_rgb24_to_rgb32_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + (src_y_pitch_byte * y_range.start_src) + crop_left * 3;;
    uint8_t *dstLine = (uint8_t *)dst[0] + (dst_y_pitch_byte * y_range.start_dst);
    alignas(16) const char MASK_RGB3_TO_RGB4[] = { 0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1 };
    __m128i xMask = _mm_load_si128((__m128i*)MASK_RGB3_TO_RGB4);
    for (int y = 0; y < y_range.len; y++, srcLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        uint8_t *ptr_src = srcLine;
        uint8_t *ptr_dst = dstLine;
        int x = 0, x_fin = width - crop_left - crop_right - 16;
        for ( ; x < x_fin; x += 16, ptr_dst += 64, ptr_src += 48) {
            __m128i x0 = _mm_loadu_si128((__m128i*)(ptr_src +  0));
            __m128i x1 = _mm_loadu_si128((__m128i*)(ptr_src + 16));
            __m128i x2 = _mm_loadu_si128((__m128i*)(ptr_src + 32));
            __m128i x3 = _mm_srli_si128(x2, 4);
            x3 = _mm_shuffle_epi8(x3, xMask);
            x2 = _mm_alignr_epi8(x2, x1, 8);
            x2 = _mm_shuffle_epi8(x2, xMask);
            x1 = _mm_alignr_epi8(x1, x0, 12);
            x1 = _mm_shuffle_epi8(x1, xMask);
            x0 = _mm_shuffle_epi8(x0, xMask);
            _mm_storeu_si128((__m128i*)(ptr_dst + 48), x3);
            _mm_storeu_si128((__m128i*)(ptr_dst + 32), x2);
            _mm_storeu_si128((__m128i*)(ptr_dst + 16), x1);
            _mm_storeu_si128((__m128i*)(ptr_dst +  0), x0);
        }
        x_fin = width - crop_left - crop_right;
        for ( ; x < x_fin; x++, ptr_dst += 4, ptr_src += 3) {
            *(int *)ptr_dst = *(int *)ptr_src;
            ptr_dst[3] = 0;
        }
    }
}

static void __forceinline convert_rgb24r_to_rgb32_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + (src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1)) + crop_left * 3;;
    uint8_t *dstLine = (uint8_t *)dst[0] + (dst_y_pitch_byte * y_range.start_dst);
    alignas(16) const char MASK_RGB3_TO_RGB4[] = { 0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1 };
    __m128i xMask = _mm_load_si128((__m128i*)MASK_RGB3_TO_RGB4);
    for (int y = 0; y  < y_range.len; y++, srcLine -= src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        uint8_t *ptr_src = srcLine;
        uint8_t *ptr_dst = dstLine;
        int x = 0, x_fin = width - crop_left - crop_right - 16;
        for (; x < x_fin; x += 16, ptr_dst += 64, ptr_src += 48) {
            __m128i x0 = _mm_loadu_si128((__m128i*)(ptr_src +  0));
            __m128i x1 = _mm_loadu_si128((__m128i*)(ptr_src + 16));
            __m128i x2 = _mm_loadu_si128((__m128i*)(ptr_src + 32));
            __m128i x3 = _mm_srli_si128(x2, 4);
            x3 = _mm_shuffle_epi8(x3, xMask);
            x2 = _mm_alignr_epi8(x2, x1, 8);
            x2 = _mm_shuffle_epi8(x2, xMask);
            x1 = _mm_alignr_epi8(x1, x0, 12);
            x1 = _mm_shuffle_epi8(x1, xMask);
            x0 = _mm_shuffle_epi8(x0, xMask);
            _mm_storeu_si128((__m128i*)(ptr_dst + 48), x3);
            _mm_storeu_si128((__m128i*)(ptr_dst + 32), x2);
            _mm_storeu_si128((__m128i*)(ptr_dst + 16), x1);
            _mm_storeu_si128((__m128i*)(ptr_dst +  0), x0);
        }
        x_fin = width - crop_left - crop_right;
        for (; x < x_fin; x++, ptr_dst += 4, ptr_src += 3) {
            *(int *)ptr_dst = *(int *)ptr_src;
            ptr_dst[3] = 0;
        }
    }
}
#endif

static void __forceinline convert_rgb24_to_rgb24_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        memcpy_sse(dstLine, srcLine, y_width * 3);
    }
}

static void __forceinline convert_rgb24r_to_rgb24_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * (y_range.start_src + y_range.len - 1) + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        memcpy_sse(dstLine, srcLine, y_width * 3);
    }
}

static void __forceinline convert_rgb32_to_rgb32_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        memcpy_sse(dstLine, srcLine, y_width * 4);
    }
}

static void __forceinline convert_rgb32r_to_rgb32_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * (y_range.start_src + y_range.len - 1) + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        memcpy_sse(dstLine, srcLine, y_width * 4);
    }
}

template<bool uv_only>
static void convert_yv12_to_p010_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            uint16_t *dst_ptr = (uint16_t *)dstLine;
            uint8_t *src_ptr = srcYLine;
            uint8_t *src_ptr_fin = src_ptr + y_width;
            __m128i x0, x1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 16, src_ptr += 16) {
                x1 = _mm_loadu_si128((const __m128i *)src_ptr);
                x0 = _mm_unpacklo_epi8(_mm_setzero_si128(), x1);
                x1 = _mm_unpackhi_epi8(_mm_setzero_si128(), x1);
                x0 = _mm_add_epi16(x0, _mm_set1_epi16(2 << 6));
                x1 = _mm_add_epi16(x1, _mm_set1_epi16(2 << 6));
                _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
                _mm_storeu_si128((__m128i *)(dst_ptr + 8), x1);
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine  = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = (uint16_t *)dstLine;
        uint16_t *dst_ptr_fin = dst_ptr + x_fin;
        __m128i x0, x1, x2, x3;
        for (; dst_ptr < dst_ptr_fin; src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x2 = _mm_loadu_si128((const __m128i *)src_v_ptr);
            x1 = _mm_unpacklo_epi8(x0, x2);
            x3 = _mm_unpackhi_epi8(x0, x2);

            x0 = _mm_unpacklo_epi8(_mm_setzero_si128(), x1);
            x1 = _mm_unpackhi_epi8(_mm_setzero_si128(), x1);
            x0 = _mm_add_epi16(x0, _mm_set1_epi16(2 << 6));
            x1 = _mm_add_epi16(x1, _mm_set1_epi16(2 << 6));

            x2 = _mm_unpacklo_epi8(_mm_setzero_si128(), x3);
            x3 = _mm_unpackhi_epi8(_mm_setzero_si128(), x3);
            x2 = _mm_add_epi16(x2, _mm_set1_epi16(2 << 6));
            x3 = _mm_add_epi16(x3, _mm_set1_epi16(2 << 6));

            _mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr +  8), x1);
            _mm_storeu_si128((__m128i *)(dst_ptr + 16), x2);
            _mm_storeu_si128((__m128i *)(dst_ptr + 24), x3);
        }
    }
}

template<int in_bit_depth, bool uv_only>
static void convert_yv12_high_to_nv12_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint8_t *dst_ptr = dstLine;
            uint16_t *src_ptr = srcYLine;
            uint16_t *src_ptr_fin = src_ptr + y_width;
            __m128i x0, x1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 16, src_ptr += 16) {
                x0 = _mm_loadu_si128((const __m128i *)(src_ptr + 0));
                x1 = _mm_loadu_si128((const __m128i *)(src_ptr + 8));

                x0 = _mm_srli_epi16(x0, in_bit_depth - 8);
                x1 = _mm_srli_epi16(x1, in_bit_depth - 8);

                x0 = _mm_packus_epi16(x0, x1);

                _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine  = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        uint8_t *dst_ptr_fin = dst_ptr + x_fin;
        __m128i x0, x1;
        for (; dst_ptr < dst_ptr_fin; src_u_ptr += 8, src_v_ptr += 8, dst_ptr += 16) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

            x0 = _mm_srli_epi16(x0, in_bit_depth - 8);
            x1 = _mm_slli_epi16(x1, 16 - in_bit_depth);
            const __m128i xMaskHighByte = _mm_slli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8);
            x1 = _mm_and_si128(x1, xMaskHighByte);

            x0 = _mm_or_si128(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
        }
    }
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<int in_bit_depth, bool uv_only>
static void __forceinline convert_yv12_high_to_p010_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                memcpy_sse((uint8_t *)dstLine, (uint8_t *)srcYLine, y_width * sizeof(uint16_t));
            } else {
                uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 8, dst_ptr += 8, src_ptr += 8) {
                    __m128i x0 = _mm_loadu_si128((const __m128i *)src_ptr);
                    x0 = _mm_slli_epi16(x0, 16 - in_bit_depth);
                    _mm_storeu_si128((__m128i *)dst_ptr, x0);
                }
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *dstLine = (uint16_t *)dst[1] + dst_y_pitch * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        __m128i x0, x1, x2;
        for (int x = crop_left; x < x_fin; x += 16, src_u_ptr += 8, src_v_ptr += 8, dst_ptr += 16) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

            if (in_bit_depth < 16) {
                x0 = _mm_slli_epi16(x0, 16 - in_bit_depth);
                x1 = _mm_slli_epi16(x1, 16 - in_bit_depth);
            }

            x2 = _mm_unpackhi_epi16(x0, x1);
            x0 = _mm_unpacklo_epi16(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 8), x2);
        }
    }
}

static void __forceinline convert_yuv422_to_p210_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        uint8_t *src_ptr = srcYLine;
        uint16_t *dst_ptr = dstLine;
        for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
            __m128i x0, x1;
            x0 = _mm_loadu_si128((const __m128i *)src_ptr);
            x1 = _mm_unpackhi_epi8(_mm_setzero_si128(), x0);
            x0 = _mm_unpacklo_epi8(_mm_setzero_si128(), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 8), x1);
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte;
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint16_t *)dst[1] + dst_y_pitch * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = width - crop_right;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        __m128i x0, x1, x2, x3, x4;
        for (int x = crop_left; x < x_fin; x += 16, src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);
            x2 = _mm_unpackhi_epi8(_mm_setzero_si128(), x0);
            x0 = _mm_unpacklo_epi8(_mm_setzero_si128(), x0);
            x3 = _mm_unpackhi_epi8(_mm_setzero_si128(), x1);
            x1 = _mm_unpacklo_epi8(_mm_setzero_si128(), x1);

            x4 = _mm_unpackhi_epi16(x0, x1);
            x0 = _mm_unpacklo_epi16(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr +  8), x4);

            x4 = _mm_unpackhi_epi16(x2, x3);
            x0 = _mm_unpacklo_epi16(x2, x3);

            _mm_storeu_si128((__m128i *)(dst_ptr + 16), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 24), x4);
        }
    }
}

template<int in_bit_depth>
static void __forceinline convert_yuv422_high_to_p210_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        if (in_bit_depth == 16) {
            memcpy_sse((uint8_t *)dstLine, (uint8_t *)srcYLine, y_width * sizeof(uint16_t));
        } else {
            uint16_t *src_ptr = srcYLine;
            uint16_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 8, dst_ptr += 8, src_ptr += 8) {
                __m128i x0 = _mm_loadu_si128((const __m128i *)src_ptr);
                x0 = _mm_slli_epi16(x0, 16 - in_bit_depth);
                _mm_storeu_si128((__m128i *)dst_ptr, x0);
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint16_t *)dst[1] + dst_y_pitch * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        __m128i x0, x1, x2;
        for (int x = crop_left; x < x_fin; x += 16, src_u_ptr += 8, src_v_ptr += 8, dst_ptr += 16) {
            x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
            x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

            if (in_bit_depth < 16) {
                x0 = _mm_slli_epi16(x0, 16 - in_bit_depth);
                x1 = _mm_slli_epi16(x1, 16 - in_bit_depth);
            }

            x2 = _mm_unpackhi_epi16(x0, x1);
            x0 = _mm_unpacklo_epi16(x0, x1);

            _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
            _mm_storeu_si128((__m128i *)(dst_ptr + 8), x2);
        }
    }
}

static void __forceinline copy_yuv444_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint8_t *srcYLine = (uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy_sse(dstLine, srcYLine, y_width);
        }
    }
}

static void __forceinline convert_yuv444_to_yuv444_16_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint8_t *srcYLine = (uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch) {
            uint8_t *src_ptr = srcYLine;
            uint16_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                __m128i x0, x1;
                x0 = _mm_loadu_si128((const __m128i *)src_ptr);
                x1 = _mm_unpackhi_epi8(_mm_setzero_si128(), x0);
                x0 = _mm_unpacklo_epi8(_mm_setzero_si128(), x0);
                _mm_storeu_si128((__m128i *)(dst_ptr + 0), x0);
                _mm_storeu_si128((__m128i *)(dst_ptr + 8), x1);
            }
        }
    }
}

template<int in_bit_depth>
static void __forceinline convert_yuv444_high_to_yuv444_16_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint16_t *srcYLine = (uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                memcpy_sse((uint8_t *)dstLine, (uint8_t *)srcYLine, y_width * sizeof(uint16_t));
            } else {
                uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 8, dst_ptr += 8, src_ptr += 8) {
                    __m128i x0 = _mm_loadu_si128((const __m128i *)src_ptr);
                    x0 = _mm_slli_epi16(x0, 16 - in_bit_depth);
                    _mm_storeu_si128((__m128i *)dst_ptr, x0);
                }
            }
        }
    }
}

template<int in_bit_depth>
static void __forceinline convert_yuv444_high_to_yuv444_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint16_t *srcYLine = (uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint16_t *src_ptr = srcYLine;
            uint8_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                __m128i x0 = _mm_loadu_si128((const __m128i *)(src_ptr + 0));
                __m128i x1 = _mm_loadu_si128((const __m128i *)(src_ptr + 8));
                x0 = _mm_srli_epi16(x0, in_bit_depth - 8);
                x1 = _mm_srli_epi16(x1, in_bit_depth - 8);
                x0 = _mm_packus_epi16(x0, x1);
                _mm_storeu_si128((__m128i *)dst_ptr, x0);
            }
        }
    }
}

typedef    struct {
    short    y;                    //    画素(輝度    )データ (     0 ～ 4096 )
    short    cb;                    //    画素(色差(青))データ ( -2048 ～ 2048 )
    short    cr;                    //    画素(色差(赤))データ ( -2048 ～ 2048 )
                                    //    画素データは範囲外に出ていることがあります
                                    //    また範囲内に収めなくてもかまいません
} PIXEL_YC;

typedef struct {
    uint16_t y, cb, cr;
} PIXEL_LW48;

typedef struct {
    int   count;       //planarの枚数。packedなら1
    uint8_t *data[3];  //planarの先頭へのポインタ
    int   size[3];     //planarのサイズ
    int   total_size;  //全planarのサイズの総和
} CONVERT_CF_DATA;

#include "convert_const.h"



static __forceinline void gather_y_uv_from_yc48(__m128i& x0, __m128i& x1, __m128i x2) {
#if USE_SSE41
    __m128i x3;
    const int MASK_INT_Y  = 0x80 + 0x10 + 0x02;
    const int MASK_INT_UV = 0x40 + 0x20 + 0x01;
    x3 = _mm_blend_epi16(x0, x1, MASK_INT_Y);
    x3 = _mm_blend_epi16(x3, x2, MASK_INT_Y>>2);

    x1 = _mm_blend_epi16(x0, x1, MASK_INT_UV);
    x1 = _mm_blend_epi16(x1, x2, MASK_INT_UV>>2);
    x1 = _mm_alignr_epi8_simd(x1, x1, 2);
    x1 = _mm_shuffle_epi32(x1, _MM_SHUFFLE(1,2,3,0));//UV1行目

    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);
#else
    __m128i x3;
    x3 = select_by_mask(x0, x1, xC_MASK_YCP2Y(0));
    x3 = select_by_mask(x3, x2, xC_MASK_YCP2Y(1));

    x1 = select_by_mask(x0, x1, xC_MASK_YCP2UV(0));
    x1 = select_by_mask(x1, x2, xC_MASK_YCP2UV(1));
    x1 = _mm_alignr_epi8_simd(x1, x1, 2);
    x1 = _mm_shuffle_epi32(x1, _MM_SHUFFLE(1,2,3,0));
#if USE_SSSE3
    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);
#else
    x0 = _mm_shuffle_epi32(  x3, _MM_SHUFFLE(3,1,2,0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(1,2,3,0));
    x0 = _mm_shuffle_epi32(  x0, _MM_SHUFFLE(1,2,3,0));
    x0 = _mm_shufflelo_epi16(x0, _MM_SHUFFLE(1,2,3,0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(3,0,1,2));
#endif //USE_SSSE3
#endif //USE_SSE41
}

static __forceinline __m128i convert_y_range_from_yc48(__m128i x0, const __m128i& xC_Y_MA_16, int Y_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    __m128i x7;
    x7 = _mm_unpackhi_epi16(x0, xC_pw_one);
    x0 = _mm_unpacklo_epi16(x0, xC_pw_one);

    x0 = _mm_madd_epi16(x0, xC_Y_MA_16);
    x7 = _mm_madd_epi16(x7, xC_Y_MA_16);
    x0 = _mm_srai_epi32(x0, Y_RSH_16);
    x7 = _mm_srai_epi32(x7, Y_RSH_16);
    x0 = _mm_add_epi32(x0, xC_YCC);
    x7 = _mm_add_epi32(x7, xC_YCC);

    x0 = _mm_packus_epi32_simd(x0, x7);

    return x0;
}

static __forceinline __m128i convert_uv_range_after_adding_offset(__m128i x0, const __m128i& xC_UV_MA_16, int UV_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    __m128i x1;
    x1 = _mm_unpackhi_epi16(x0, xC_pw_one);
    x0 = _mm_unpacklo_epi16(x0, xC_pw_one);

    x0 = _mm_madd_epi16(x0, xC_UV_MA_16);
    x1 = _mm_madd_epi16(x1, xC_UV_MA_16);
    x0 = _mm_srai_epi32(x0, UV_RSH_16);
    x1 = _mm_srai_epi32(x1, UV_RSH_16);
    x0 = _mm_add_epi32(x0, xC_YCC);
    x1 = _mm_add_epi32(x1, xC_YCC);

    x0 = _mm_packus_epi32_simd(x0, x1);

    return x0;
}

static __forceinline __m128i convert_uv_range_from_yc48(__m128i x0, const __m128i& xC_UV_OFFSET_x1, const __m128i& xC_UV_MA_16, int UV_RSH_16, __m128i xC_YCC, const __m128i& xC_pw_one) {
    x0 = _mm_add_epi16(x0, xC_UV_OFFSET_x1);

    return convert_uv_range_after_adding_offset(x0, xC_UV_MA_16, UV_RSH_16, xC_YCC, xC_pw_one);
}
static __forceinline __m128i convert_uv_range_from_yc48_yuv420p(__m128i x0, __m128i x1, const __m128i& xC_UV_OFFSET_x2, const __m128i& xC_UV_MA_16, int UV_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    x0 = _mm_add_epi16(x0, x1);
    x0 = _mm_add_epi16(x0, xC_UV_OFFSET_x2);

    return convert_uv_range_after_adding_offset(x0, xC_UV_MA_16, UV_RSH_16, xC_YCC, xC_pw_one);
}
static __forceinline __m128i convert_uv_range_from_yc48_420i(__m128i x0, __m128i x1, const __m128i& xC_UV_OFFSET_x1, const __m128i& xC_UV_MA_16_0, const __m128i& xC_UV_MA_16_1, int UV_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    __m128i x2, x3, x6, x7;
    x0 = _mm_add_epi16(x0, xC_UV_OFFSET_x1);
    x1 = _mm_add_epi16(x1, xC_UV_OFFSET_x1);

    x7 = _mm_unpackhi_epi16(x0, xC_pw_one);
    x6 = _mm_unpacklo_epi16(x0, xC_pw_one);
    x3 = _mm_unpackhi_epi16(x1, xC_pw_one);
    x2 = _mm_unpacklo_epi16(x1, xC_pw_one);

    x6 = _mm_madd_epi16(x6, xC_UV_MA_16_0);
    x7 = _mm_madd_epi16(x7, xC_UV_MA_16_0);
    x2 = _mm_madd_epi16(x2, xC_UV_MA_16_1);
    x3 = _mm_madd_epi16(x3, xC_UV_MA_16_1);
    x0 = _mm_add_epi32(x6, x2);
    x7 = _mm_add_epi32(x7, x3);
    x0 = _mm_srai_epi32(x0, UV_RSH_16);
    x7 = _mm_srai_epi32(x7, UV_RSH_16);
    x0 = _mm_add_epi32(x0, xC_YCC);
    x7 = _mm_add_epi32(x7, xC_YCC);

    x0 = _mm_packus_epi32_simd(x0, x7);

    return x0;
}

static __forceinline __m128i convert_y_range_to_yc48(__m128i x0) {
    //coef = 4788
    //((( y - 32768 ) * coef) >> 16 ) + (coef/2 - 299)
    const __m128i xC_0x8000 = _mm_slli_epi16(_mm_cmpeq_epi32(x0, x0), 15);
    x0 = _mm_add_epi16(x0, xC_0x8000); // -32768
    x0 = _mm_mulhi_epi16(x0, _mm_set1_epi16(4788));
    x0 = _mm_adds_epi16(x0, _mm_set1_epi16(4788/2 - 299));
    return x0;
}

static __forceinline __m128i convert_uv_range_to_yc48(__m128i x0) {
    //coeff = 4682
    //UV = (( uv - 32768 ) * coef + (1<<15) ) >> 16
    const __m128i xC_coeff = _mm_unpacklo_epi16(_mm_set1_epi16(4682), _mm_set1_epi16(-1));
    const __m128i xC_0x8000 = _mm_slli_epi16(_mm_cmpeq_epi32(x0, x0), 15);
    __m128i x1;
    x0 = _mm_add_epi16(x0, xC_0x8000); // -32768
    x1 = _mm_unpackhi_epi16(x0, xC_0x8000);
    x0 = _mm_unpacklo_epi16(x0, xC_0x8000);
    x0 = _mm_madd_epi16(x0, xC_coeff);
    x1 = _mm_madd_epi16(x1, xC_coeff);
    x0 = _mm_srai_epi32(x0, 16);
    x1 = _mm_srai_epi32(x1, 16);
    x0 = _mm_packs_epi32(x0, x1);
    return x0;
}

static __forceinline void gather_y_u_v_from_yc48(__m128i& x0, __m128i& x1, __m128i& x2) {
#if USE_SSE41
    __m128i x3, x4, x5;
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    x3 = _mm_blend_epi16(x2, x0, MASK_INT);
    x4 = _mm_blend_epi16(x1, x2, MASK_INT);
    x5 = _mm_blend_epi16(x0, x1, MASK_INT);

    x3 = _mm_blend_epi16(x3, x1, MASK_INT<<1);
    x4 = _mm_blend_epi16(x4, x0, MASK_INT<<1);
    x5 = _mm_blend_epi16(x5, x2, MASK_INT<<1);

    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);
    x1 = _mm_shuffle_epi8(x4, _mm_alignr_epi8_simd(xC_SUFFLE_YCP_Y, xC_SUFFLE_YCP_Y, 6));
    x2 = _mm_shuffle_epi8(x5, _mm_alignr_epi8_simd(xC_SUFFLE_YCP_Y, xC_SUFFLE_YCP_Y, 12));
#else
    //code from afs v7.5a+10
    __m128i x5, x6, x7, xMask;
    //select y
    alignas(16) static const uint16_t maskY_select[8] = { 0xffff, 0x0000, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_select);

    x5 = select_by_mask(x2, x0, xMask);
    xMask = _mm_slli_si128(xMask, 2);
    x5 = select_by_mask(x5, x1, xMask); //52741630

    x6 = _mm_unpacklo_epi16(x5, x5);    //11663300
    x7 = _mm_unpackhi_epi16(x5, x5);    //55227744

    alignas(16) static const uint16_t maskY_shuffle[8] = { 0xffff, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_shuffle);
    x5 = select_by_mask(x7, x6, xMask);                 //51627340
    x5 = _mm_shuffle_epi32(x5, _MM_SHUFFLE(1, 2, 3, 0));   //73625140

    x5 = _mm_unpacklo_epi16(x5, _mm_srli_si128(x5, 8)); //75316420
    x5 = _mm_unpacklo_epi16(x5, _mm_srli_si128(x5, 8)); //76543210

                                                        //select uv
    xMask = _mm_srli_si128(_mm_cmpeq_epi8(xMask, xMask), 8); //0x00000000, 0x00000000, 0xffffffff, 0xffffffff
    x6 = select_by_mask(_mm_srli_si128(x1, 2), _mm_srli_si128(x2, 2), xMask); //x  x v4 u4 v6 u6 x  x
    x7 = select_by_mask(x0, x1, xMask);               //x  x  v1 u1 v3 u3 x  x
    xMask = _mm_slli_si128(xMask, 4);                 //0x00000000, 0xffffffff, 0xffffffff, 0x00000000
    x0 = _mm_alignr_epi8_simd(x1, x0, 2);             //v2 u2  x  x  x  x v0 u0
    x6 = select_by_mask(x0, x6, xMask);               //v2 u2 v4 u4 v6 u6 v0 u0
    x7 = select_by_mask(x2, x7, xMask);               //v7 u7 v1 u1 v3 u3 v5 u5
    x0 = _mm_shuffle_epi32(x6, _MM_SHUFFLE(1, 2, 3, 0)); //v6 u6 v4 u4 v2 u2 v0 u0
    x1 = _mm_shuffle_epi32(x7, _MM_SHUFFLE(3, 0, 1, 2)); //v7 u7 v5 u5 v3 u3 v1 u1

    x6 = _mm_unpacklo_epi16(x0, x1); //v3 v2 u3 u2 v1 v0 u1 u0
    x7 = _mm_unpackhi_epi16(x0, x1); //v7 v6 u7 u6 v5 v4 u5 u4

    x0 = _mm_unpacklo_epi32(x6, x7); //v5 v4 v1 v0 u5 u4 u1 u0
    x1 = _mm_unpackhi_epi32(x6, x7); //v7 v6 v3 v2 u7 u6 u3 u2

    x6 = _mm_unpacklo_epi32(x0, x1); //u7 u6 u5 u4 u3 u2 u1 u0
    x7 = _mm_unpackhi_epi32(x0, x1); //v7 v6 v5 v4 v3 v2 v1 v0

    x0 = x5;
    x1 = x6;
    x2 = x7;
#endif //USE_SSE41
}


static __forceinline void gather_y_u_v_to_yc48(__m128i& x0, __m128i& x1, __m128i& x2) {
    __m128i x3, x4;
#if USE_SSE41
    alignas(16) static const uint8_t shuffle_yc48[16] = {
        0x00, 0x01, 0x06, 0x07, 0x0C, 0x0D, 0x02, 0x03, 0x08, 0x09, 0x0E, 0x0F, 0x04, 0x05, 0x0A, 0x0B
    };
    x4 = _mm_load_si128((__m128i *)shuffle_yc48);
    x0 = _mm_shuffle_epi8(x0, x4);                          //5,2,7,4,1,6,3,0
    x1 = _mm_shuffle_epi8(x1, _mm_alignr_epi8(x4, x4, 14)); //2,7,4,1,6,3,0,5
    x2 = _mm_shuffle_epi8(x2, _mm_alignr_epi8(x4, x4, 12)); //7,4,1,6,3,0,5,2

    x3 = _mm_blend_epi16(x0, x1, 0x80 + 0x10 + 0x02);
    x3 = _mm_blend_epi16(x3, x2, 0x20 + 0x04);

    x4 = _mm_blend_epi16(x2, x1, 0x20 + 0x04);
    x4 = _mm_blend_epi16(x4, x0, 0x80 + 0x10 + 0x02);

    x2 = _mm_blend_epi16(x2, x0, 0x20 + 0x04);
    x2 = _mm_blend_epi16(x2, x1, 0x40 + 0x08 + 0x01);

    x0 = x3;
    x1 = x4;
#else
    x0 = _mm_shufflelo_epi16(x0, _MM_SHUFFLE(3,1,2,0)); // 7,6,5,4,3,1,2,0
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(3,1,2,0)); // 7,5,6,4,3,1,2,0
    x0 = _mm_shuffle_epi32(  x0, _MM_SHUFFLE(3,1,2,0)); // 7,5,3,1,6,4,2,0
    x0 = _mm_shufflelo_epi16(x0, _MM_SHUFFLE(3,1,2,0)); // 7,5,3,1,6,2,4,0
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(3,1,2,0)); // 7,3,5,1,6,2,4,0

    x1 = _mm_shufflelo_epi16(x1, _MM_SHUFFLE(3,1,2,0));
    x1 = _mm_shufflehi_epi16(x1, _MM_SHUFFLE(3,1,2,0));
    x1 = _mm_shuffle_epi32(  x1, _MM_SHUFFLE(3,1,2,0));
    x1 = _mm_shufflelo_epi16(x1, _MM_SHUFFLE(3,1,2,0));
    x1 = _mm_shufflehi_epi16(x1, _MM_SHUFFLE(3,1,2,0));

    x2 = _mm_shufflelo_epi16(x2, _MM_SHUFFLE(3,1,2,0));
    x2 = _mm_shufflehi_epi16(x2, _MM_SHUFFLE(3,1,2,0));
    x2 = _mm_shuffle_epi32(  x2, _MM_SHUFFLE(3,1,2,0));
    x2 = _mm_shufflelo_epi16(x2, _MM_SHUFFLE(3,1,2,0));
    x2 = _mm_shufflehi_epi16(x2, _MM_SHUFFLE(3,1,2,0));

    x3 = _mm_shuffle_epi32(x0, _MM_SHUFFLE(3,2,3,2));
    x0 = _mm_unpacklo_epi16(x0, x1);
    x1 = _mm_unpackhi_epi16(x1, x2);
    x2 = _mm_unpacklo_epi16(x2, x3);

    x3 = _mm_shuffle_epi32(x0, _MM_SHUFFLE(3,2,3,2));
    x0 = _mm_unpacklo_epi32(x0, x2);
    x2 = _mm_unpackhi_epi32(x2, x1);
    x1 = _mm_unpacklo_epi32(x1, x3);

    x3 = _mm_shuffle_epi32(x0, _MM_SHUFFLE(3,2,3,2));
    x0 = _mm_unpacklo_epi64(x0, x1);
    x1 = _mm_unpackhi_epi64(x1, x2);
    x2 = _mm_unpacklo_epi64(x2, x3);

    x4 = x2;
    x2 = x1;
    x1 = x4;
#endif
}

template <bool aligned_store>
static __forceinline void convert_yc48_to_p010_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int x, y;
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    __m128i x0, x1, x2, x3;
    for (y = y_range.start_src; y < (y_range.start_src + y_range.len); y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = dst_Y + dst_y_pitch * y;
        C   = dst_C + dst_y_pitch * y / 2;
        for (x = 0; x < width; x += 8, ycp += 24, ycpw += 24) {
            x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
            x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
            x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
            _mm_prefetch((const char *)ycpw, _MM_HINT_T1);
            gather_y_uv_from_yc48(x1, x2, x3);
            x0 = x2;

            _mm_store_switch_si128((__m128i *)(Y + x), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

            x1 = _mm_loadu_si128((__m128i *)(ycpw +  0));
            x2 = _mm_loadu_si128((__m128i *)(ycpw +  8));
            x3 = _mm_loadu_si128((__m128i *)(ycpw + 16));
            gather_y_uv_from_yc48(x1, x2, x3);

            _mm_store_switch_si128((__m128i *)(Y + x + dst_y_pitch), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

            x0 = convert_uv_range_from_yc48_yuv420p(x0, x2, _mm_set1_epi16(UV_OFFSET_x2), xC_UV_L_MA_16_420P, UV_L_RSH_16_420P, xC_YCC, xC_pw_one);

            _mm_store_switch_si128((__m128i *)(C + x), x0);
        }
    }
}

template <bool aligned_store>
static __forceinline void convert_yc48_to_p010_i_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int x, y, i;
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = nullptr, *C = nullptr;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    __m128i x0, x1, x2, x3;
    for (y = y_range.start_src; y < (y_range.start_src + y_range.len); y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = dst_Y + dst_y_pitch * (y + i);
            C   = dst_C + dst_y_pitch * (y + i*2) / 2;
            for (x = 0; x < width; x += 8, ycp += 24, ycpw += 24) {
                x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
                x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
                x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
                _mm_prefetch((const char *)ycpw, _MM_HINT_T1);
                gather_y_uv_from_yc48(x1, x2, x3);
                x0 = x2;
                _mm_store_switch_si128((__m128i *)(Y + x), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                x1 = _mm_loadu_si128((__m128i *)(ycpw +  0));
                x2 = _mm_loadu_si128((__m128i *)(ycpw +  8));
                x3 = _mm_loadu_si128((__m128i *)(ycpw + 16));
                gather_y_uv_from_yc48(x1, x2, x3);
                _mm_store_switch_si128((__m128i *)(Y + x + dst_y_pitch*2), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                _mm_store_switch_si128((__m128i *)(C + x), convert_uv_range_from_yc48_420i(x0, x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_420I(i), xC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, xC_YCC, xC_pw_one));
            }
        }
    }
}

template <bool aligned_store>
static void __forceinline convert_yc48_to_yuv444_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    uint8_t *YLine   = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *ULine   = (uint8_t *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *VLine   = (uint8_t *)dst[2] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *ycpLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x1, x2, x3, xY, xU, xV;
    for (int y = 0; y < y_range.len; y++, ycpLine += src_y_pitch_byte, YLine += dst_y_pitch_byte, ULine += dst_y_pitch_byte, VLine += dst_y_pitch_byte) {
        uint8_t *Y = YLine;
        uint8_t *U = ULine;
        uint8_t *V = VLine;
        int16_t *const ycp_fin = (int16_t *)ycpLine + width * 3;
        for (int16_t *ycp = (int16_t *)ycpLine; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
            x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
            x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
            x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
            gather_y_u_v_from_yc48(x1, x2, x3);

            x1 = convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one);
            x2 = convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
            x3 = convert_uv_range_from_yc48(x3, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
            xY = _mm_srli_epi16(x1, 8);
            xU = _mm_srli_epi16(x2, 8);
            xV = _mm_srli_epi16(x3, 8);

            x1 = _mm_loadu_si128((__m128i *)(ycp + 24));
            x2 = _mm_loadu_si128((__m128i *)(ycp + 32));
            x3 = _mm_loadu_si128((__m128i *)(ycp + 40));
            gather_y_u_v_from_yc48(x1, x2, x3);

            x1 = convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one);
            x2 = convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
            x3 = convert_uv_range_from_yc48(x3, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
            x1 = _mm_srli_epi16(x1, 8);
            x2 = _mm_srli_epi16(x2, 8);
            x3 = _mm_srli_epi16(x3, 8);

            xY = _mm_packus_epi16(xY, x1);
            xU = _mm_packus_epi16(xU, x2);
            xV = _mm_packus_epi16(xV, x3);

            _mm_store_switch_si128((__m128i*)Y, xY);
            _mm_store_switch_si128((__m128i*)U, xU);
            _mm_store_switch_si128((__m128i*)V, xV);
        }
    }
}

template <bool aligned_store>
static __forceinline void convert_yc48_to_yuv444_16bit_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    char *Y_line = (char *)dst[0] + dst_y_pitch_byte + y_range.start_dst;
    char *U_line = (char *)dst[1] + dst_y_pitch_byte + y_range.start_dst;
    char *V_line = (char *)dst[2] + dst_y_pitch_byte + y_range.start_dst;
    char *pixel = (char *)src[0] + src_y_pitch_byte * y_range.start_src;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x1, x2, x3;
    for (int y = 0; y < y_range.len; y++, pixel += src_y_pitch_byte, Y_line += dst_y_pitch_byte, U_line += dst_y_pitch_byte, V_line += dst_y_pitch_byte) {
        short *Y = (short *)Y_line;
        short *U = (short *)U_line;
        short *V = (short *)V_line;
        short *const ycp_fin = (short *)pixel + width * 3;
        for (short *ycp = (short *)pixel; ycp < ycp_fin; ycp += 24, Y += 8, U += 8, V += 8) {
            x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
            x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
            x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
            gather_y_u_v_from_yc48(x1, x2, x3);
            _mm_store_switch_si128((__m128i *)Y, convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));
            _mm_store_switch_si128((__m128i *)U, convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one));
            _mm_store_switch_si128((__m128i *)V, convert_uv_range_from_yc48(x3, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one));
        }
    }
}

template <bool aligned_store>
static __forceinline void convert_yuv444_16bit_to_yc48_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    char *Y_line = (char *)src[0] + src_y_pitch_byte * y_range.start_src;
    char *U_line = (char *)src[1] + src_y_pitch_byte * y_range.start_src;
    char *V_line = (char *)src[2] + src_y_pitch_byte * y_range.start_src;
    char *pixel = (char *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    __m128i x1, x2, x3;
    for (int y = 0; y < y_range.len; y++, pixel += dst_y_pitch_byte, Y_line += src_y_pitch_byte, U_line += src_y_pitch_byte, V_line += src_y_pitch_byte) {
        short *Y = (short *)Y_line;
        short *U = (short *)U_line;
        short *V = (short *)V_line;
        short *const ycp_fin = (short *)pixel + width * 3;
        for (short *ycp = (short *)pixel; ycp < ycp_fin; ycp += 24, Y += 8, U += 8, V += 8) {
            x1 = _mm_loadu_si128((__m128i *)(Y));
            x2 = _mm_loadu_si128((__m128i *)(U));
            x3 = _mm_loadu_si128((__m128i *)(V));
            x1 = convert_y_range_to_yc48(x1);
            x2 = convert_uv_range_to_yc48(x2);
            x3 = convert_uv_range_to_yc48(x3);
            gather_y_u_v_to_yc48(x1, x2, x3);
            _mm_store_switch_si128((__m128i *)(ycp +  0), x1);
            _mm_store_switch_si128((__m128i *)(ycp +  8), x2);
            _mm_store_switch_si128((__m128i *)(ycp + 16), x3);
        }
    }
}

#pragma warning (pop)

#endif //_CONVERT_CSP_SIMD_H_
