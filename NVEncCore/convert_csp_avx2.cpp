﻿// -----------------------------------------------------------------------------------------
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
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1

#include <immintrin.h>
#include "rgy_simd.h"
#include <stdint.h>
#include <string.h>
#include "convert_csp.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

#if defined(_MSC_VER) || defined(__AVX2__)

template<bool use_stream>
static void RGY_FORCEINLINE avx2_memcpy(uint8_t *dst, const uint8_t *src, int size) {
    if (size < 128) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 31) & ~31) - 128);
    __m256i y0, y1, y2, y3;
    const int start_align_diff = (int)((size_t)dst & 31);
    if (start_align_diff) {
        y0 = _mm256_loadu_si256((const __m256i*)src);
        _mm256_storeu_si256((__m256i*)dst, y0);
        dst += 32 - start_align_diff;
        src += 32 - start_align_diff;
    }
#define _mm256_stream_switch_si256(x, ymm) ((use_stream) ? _mm256_stream_si256((x), (ymm)) : _mm256_store_si256((x), (ymm)))
    for ( ; dst < dst_aligned_fin; dst += 128, src += 128) {
        y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
        y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
        y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
        y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
        _mm256_stream_switch_si256((__m256i*)(dst +  0), y0);
        _mm256_stream_switch_si256((__m256i*)(dst + 32), y1);
        _mm256_stream_switch_si256((__m256i*)(dst + 64), y2);
        _mm256_stream_switch_si256((__m256i*)(dst + 96), y3);
    }
#undef _mm256_stream_switch_si256
    uint8_t *dst_tmp = dst_fin - 128;
    src -= (dst - dst_tmp);
    y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
    y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
    y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
    y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
    _mm256_storeu_si256((__m256i*)(dst_tmp +  0), y0);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 32), y1);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 64), y2);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 96), y3);
    _mm256_zeroupper();
}

//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#ifndef _mm256_bsrli_epi128
#define _mm256_bsrli_epi128 _mm256_srli_si256
#endif
#ifndef _mm256_bslli_epi128
#define _mm256_bslli_epi128 _mm256_slli_si256
#endif
//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))

#ifndef _MSC_VER

#ifndef __forceinline
#define __forceinline __attribute__((always_inline))
#endif

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
#endif //#ifndef _mm256_set_m128i

#ifndef _mm256_loadu2_m128i
#define _mm256_loadu2_m128i(hiptr, loptr) \
    _mm256_inserti128_si256(_mm256_castsi128_si256( \
        _mm_loadu_si128((__m128i*)(loptr))), \
        _mm_loadu_si128((__m128i*)(hiptr)),1)
#endif //#ifndef _mm256_loadu2_m128i

#endif //#ifndef _MSC_VER

alignas(32) static const uint8_t  Array_INTERLACE_WEIGHT[2][32] = {
    {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
    {3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define yC_INTERLACE_WEIGHT(i) _mm256_load_si256((__m256i*)Array_INTERLACE_WEIGHT[i])

static RGY_FORCEINLINE void separate_low_up(__m256i& x0_return_lower, __m256i& x1_return_upper) {
    __m256i x4, x5;
    const __m256i xMaskLowByte = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
    x4 = _mm256_srli_epi16(x0_return_lower, 8);
    x5 = _mm256_srli_epi16(x1_return_upper, 8);

    x0_return_lower = _mm256_and_si256(x0_return_lower, xMaskLowByte);
    x1_return_upper = _mm256_and_si256(x1_return_upper, xMaskLowByte);

    x0_return_lower = _mm256_packus_epi16(x0_return_lower, x1_return_upper);
    x1_return_upper = _mm256_packus_epi16(x4, x5);
}

#pragma warning (push)
#pragma warning (disable: 4100)
template<bool highbit_depth>
void copy_nv12_to_nv12_avx2_internal(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int pixel_size = highbit_depth ? 2 : 1;
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            avx2_memcpy<true>(dstLine, srcYLine, y_width * pixel_size);
        }
    }
}
void copy_nv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12_to_nv12_avx2_internal<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_p010_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12_to_nv12_avx2_internal<true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_nv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            const uint8_t *src_ptr = (const uint8_t *)srcYLine;
            uint16_t *dst_ptr = (uint16_t *)dstLine;
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0, y1;
                y0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3, 1, 2, 0));
                y1 = _mm256_unpackhi_epi8(_mm256_setzero_si256(), y0);
                y0 = _mm256_unpacklo_epi8(_mm256_setzero_si256(), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr + 0), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y1);
            }
        }
    }
}
void copy_p010_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int in_bit_depth = 16;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            const uint16_t *src_ptr = (const uint16_t *)srcYLine;
            uint8_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 16), (const __m128i *)(src_ptr + 0));
                __m256i y1 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 24), (const __m128i *)(src_ptr + 8));
                y0 = _mm256_adds_epi16(y0, yrsftAdd);
                y1 = _mm256_adds_epi16(y1, yrsftAdd);
                y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
                y1 = _mm256_srli_epi16(y1, in_bit_depth - 8);
                y0 = _mm256_packus_epi16(y0, y1);
                _mm256_storeu_si256((__m256i *)dst_ptr, y0);
            }

        }
    }
}

void convert_yuy2_to_nv12_avx2(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const void *src = src_array[0];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst_array[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstCLine = (uint8_t *)dst_array[1] + dst_y_pitch_byte * (y_range.start_dst >> 1);
    for (int y = 0; y < y_range.len; y += 2) {
        uint8_t *p = srcLine;
        uint8_t *pw = p + src_y_pitch_byte;
        const int x_fin = width - crop_right - crop_left;
        __m256i y0, y1, y3;
        for (int x = 0; x < x_fin; x += 32, p += 64, pw += 64) {
            //-----------1行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

            separate_low_up(y0, y1);
            y3 = y1;

            _mm256_storeu_si256((__m256i *)(dstYLine + x), y0);
            //-----------1行目終了---------------

            //-----------2行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

            separate_low_up(y0, y1);

            _mm256_storeu_si256((__m256i *)(dstYLine + dst_y_pitch_byte + x), y0);
            //-----------2行目終了---------------

            y1 = _mm256_avg_epu8(y1, y3);  //VUVUVUVUVUVUVUVU
            _mm256_storeu_si256((__m256i *)(dstCLine + x), y1);
        }
        srcLine  += src_y_pitch_byte << 1;
        dstYLine += dst_y_pitch_byte << 1;
        dstCLine += dst_y_pitch_byte;
    }
    _mm256_zeroupper();
}
#pragma warning (pop)

static RGY_FORCEINLINE __m256i yuv422_to_420_i_interpolate(__m256i y_up, __m256i y_down, int i) {
    __m256i y0, y1;
    y0 = _mm256_unpacklo_epi8(y_down, y_up);
    y1 = _mm256_unpackhi_epi8(y_down, y_up);
    y0 = _mm256_maddubs_epi16(y0, yC_INTERLACE_WEIGHT(i));
    y1 = _mm256_maddubs_epi16(y1, yC_INTERLACE_WEIGHT(i));
    y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2));
    y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2));
    y0 = _mm256_srai_epi16(y0, 2);
    y1 = _mm256_srai_epi16(y1, 2);
    y0 = _mm256_packus_epi16(y0, y1);
    return y0;
}

#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4100)
void convert_yuy2_to_nv12_i_avx2(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const void *src = src_array[0];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst_array[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstCLine = (uint8_t *)dst_array[1] + dst_y_pitch_byte * (y_range.start_dst >> 1);
    for (int y = 0; y < y_range.len; y += 4) {
        for (int i = 0; i < 2; i++) {
            uint8_t *p = srcLine;
            uint8_t *pw = p + (src_y_pitch_byte<<1);
            __m256i y0, y1, y3;
            const int x_fin = width - crop_right - crop_left;
            for (int x = 0; x < x_fin; x += 32, p += 64, pw += 64) {
                //-----------    1+i行目   ---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

                separate_low_up(y0, y1);
                y3 = y1;

                _mm256_storeu_si256((__m256i *)(dstYLine + x), y0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

                separate_low_up(y0, y1);

                _mm256_storeu_si256((__m256i *)(dstYLine + (dst_y_pitch_byte<<1) + x), y0);
                //-----------3+i行目終了---------------
                y0 = yuv422_to_420_i_interpolate(y3, y1, i);

                _mm256_storeu_si256((__m256i *)(dstCLine + x), y0);
            }
            srcLine  += src_y_pitch_byte;
            dstYLine += dst_y_pitch_byte;
            dstCLine += dst_y_pitch_byte;
        }
        srcLine  += src_y_pitch_byte << 1;
        dstYLine += dst_y_pitch_byte << 1;
    }
    _mm256_zeroupper();
}

#pragma warning (push)
#pragma warning (disable: 4127)
template<bool uv_only>
static void RGY_FORCEINLINE convert_yv12_to_nv12_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
            avx2_memcpy<false>(dstLine, srcYLine, y_width);
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
        __m256i y0, y1, y2;
        for (int x = crop_left; x < x_fin; x += 64, src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 64) {
            y0 = _mm256_loadu_si256((const __m256i *)src_u_ptr);
            y1 = _mm256_loadu_si256((const __m256i *)src_v_ptr);

            y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3,1,2,0));
            y1 = _mm256_permute4x64_epi64(y1, _MM_SHUFFLE(3,1,2,0));

            y2 = _mm256_unpackhi_epi8(y0, y1);
            y0 = _mm256_unpacklo_epi8(y0, y1);

            _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 32), y2);
        }
    }
    _mm256_zeroupper();
}
#pragma warning (pop)

void convert_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_to_nv12_avx2_base<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_uv_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_to_nv12_avx2_base<true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}



#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4100)
void convert_rgb24_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    alignas(32) const char MASK_RGB3_TO_RGB4[] ={
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1
    };
    __m256i yMask = _mm256_load_si256((__m256i*)MASK_RGB3_TO_RGB4);
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        uint8_t *ptr_src = srcLine;
        uint8_t *ptr_dst = dstLine;
        int x = 0, x_fin = width - crop_left - crop_right - 32;
        for (; x < x_fin; x += 32, ptr_dst += 128, ptr_src += 96) {
            __m256i y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+48)), _mm_loadu_si128((__m128i*)(ptr_src+ 0))); //384,   0
            __m256i y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+64)), _mm_loadu_si128((__m128i*)(ptr_src+16))); //512, 128
            __m256i y2 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+80)), _mm_loadu_si128((__m128i*)(ptr_src+32))); //640, 256
            __m256i y3 = _mm256_srli_si256(y2, 4);
            y3 = _mm256_shuffle_epi8(y3, yMask); // 896, 384
            y2 = _mm256_alignr_epi8(y2, y1, 8);
            y2 = _mm256_shuffle_epi8(y2, yMask); // 768, 256
            y1 = _mm256_alignr_epi8(y1, y0, 12);
            y1 = _mm256_shuffle_epi8(y1, yMask); // 640, 128
            y0 = _mm256_shuffle_epi8(y0, yMask); // 512,   0
            _mm256_storeu_si256((__m256i*)(ptr_dst +  0), _mm256_permute2x128_si256(y0, y1, (2<<4) | 0)); // 128,   0
            _mm256_storeu_si256((__m256i*)(ptr_dst + 32), _mm256_permute2x128_si256(y2, y3, (2<<4) | 0)); // 384, 256
            _mm256_storeu_si256((__m256i*)(ptr_dst + 64), _mm256_permute2x128_si256(y0, y1, (3<<4) | 1)); // 640, 512
            _mm256_storeu_si256((__m256i*)(ptr_dst + 96), _mm256_permute2x128_si256(y2, y3, (3<<4) | 1)); // 896, 768
        }
        x_fin = width - crop_left - crop_right;
        for (; x < x_fin; x++, ptr_dst += 4, ptr_src += 3) {
            *(int *)ptr_dst = *(int *)ptr_src;
            ptr_dst[3] = 0;
        }
    }
    _mm256_zeroupper();
}

void convert_bgr24r_to_bgr32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1) + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    alignas(32) const char MASK_RGB3_TO_RGB4[] = {
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1
    };
    __m256i yMask = _mm256_load_si256((__m256i*)MASK_RGB3_TO_RGB4);
    for (int y = 0; y < y_range.len; y++, srcLine -= src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        uint8_t *ptr_src = srcLine;
        uint8_t *ptr_dst = dstLine;
        int x = 0, x_fin = width - crop_left - crop_right - 32;
        for ( ; x < x_fin; x += 32, ptr_dst += 128, ptr_src += 96) {
            __m256i y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+48)), _mm_loadu_si128((__m128i*)(ptr_src+ 0))); //384,   0
            __m256i y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+64)), _mm_loadu_si128((__m128i*)(ptr_src+16))); //512, 128
            __m256i y2 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(ptr_src+80)), _mm_loadu_si128((__m128i*)(ptr_src+32))); //640, 256
            __m256i y3 = _mm256_srli_si256(y2, 4);
            y3 = _mm256_shuffle_epi8(y3, yMask); // 896, 384
            y2 = _mm256_alignr_epi8(y2, y1, 8);
            y2 = _mm256_shuffle_epi8(y2, yMask); // 768, 256
            y1 = _mm256_alignr_epi8(y1, y0, 12);
            y1 = _mm256_shuffle_epi8(y1, yMask); // 640, 128
            y0 = _mm256_shuffle_epi8(y0, yMask); // 512,   0
            _mm256_storeu_si256((__m256i*)(ptr_dst +  0), _mm256_permute2x128_si256(y0, y1, (2<<4) | 0)); // 128,   0
            _mm256_storeu_si256((__m256i*)(ptr_dst + 32), _mm256_permute2x128_si256(y2, y3, (2<<4) | 0)); // 384, 256
            _mm256_storeu_si256((__m256i*)(ptr_dst + 64), _mm256_permute2x128_si256(y0, y1, (3<<4) | 1)); // 640, 512
            _mm256_storeu_si256((__m256i*)(ptr_dst + 96), _mm256_permute2x128_si256(y2, y3, (3<<4) | 1)); // 896, 768
        }
        x_fin = width - crop_left - crop_right;
        for ( ; x < x_fin; x++, ptr_dst += 4, ptr_src += 3) {
            *(int *)ptr_dst = *(int *)ptr_src;
            ptr_dst[3] = 0;
        }
    }
    _mm256_zeroupper();
}

void convert_rgb32_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 4);
    }
    _mm256_zeroupper();
}

void convert_bgr32r_to_bgr32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1) + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 4);
    }
    _mm256_zeroupper();
}

void convert_rgb24_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 3);
    }
    _mm256_zeroupper();
}

void convert_bgr24r_to_bgr24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1) + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 3);
    }
    _mm256_zeroupper();
}

template<bool uv_only>
static void convert_yv12_to_p010_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
            __m256i y0, y1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                y0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src_ptr +  0)));
                y1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src_ptr + 16)));
                y0 = _mm256_slli_epi16(y0, 8);
                y1 = _mm256_slli_epi16(y1, 8);
                y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2 << 6));
                y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2 << 6));
                _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y1);
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
        __m256i y0, y1, y2, y3;
        for (; dst_ptr < dst_ptr_fin; src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 64) {
            y0 = _mm256_loadu_si256((const __m256i *)src_u_ptr); // 31-28  27-24 23-20  19-16 | 15-12  11-8  7-4  3-0
            y2 = _mm256_loadu_si256((const __m256i *)src_v_ptr);

            alignas(32) static const int SHUFFLE_MASK[] = { 0, 2, 4, 6, 1, 3, 5, 7 };
            y0 = _mm256_permutevar8x32_epi32(y0, _mm256_load_si256((const __m256i *)SHUFFLE_MASK)); // 31-28  23-20  15-12  7-4 | 27-24  19-16  11-8  3-0
            y2 = _mm256_permutevar8x32_epi32(y2, _mm256_load_si256((const __m256i *)SHUFFLE_MASK));

            y1 = _mm256_unpacklo_epi8(y0, y2); // 15-12    7-4 | 11- 8   3- 0
            y3 = _mm256_unpackhi_epi8(y0, y2); // 31-28  23-20 | 27-24  19-16

            y0 = _mm256_unpacklo_epi8(_mm256_setzero_si256(), y1);  //   7-4 |  3- 0
            y1 = _mm256_unpackhi_epi8(_mm256_setzero_si256(), y1);  // 15-12 | 11- 8
            y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2 << 6));
            y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2 << 6));

            y2 = _mm256_unpacklo_epi8(_mm256_setzero_si256(), y3);  // 23-20 | 19-16
            y3 = _mm256_unpackhi_epi8(_mm256_setzero_si256(), y3);  // 31-28 | 27-24
            y2 = _mm256_add_epi16(y2, _mm256_set1_epi16(2 << 6));
            y3 = _mm256_add_epi16(y3, _mm256_set1_epi16(2 << 6));

            _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y1);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 32), y2);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 48), y3);
        }
    }
}
#pragma warning (pop)

void convert_yv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_to_p010_avx2_base<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4100)
template<int in_bit_depth, bool uv_only>
static void convert_yv12_high_to_nv12_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint8_t *dst_ptr = dstLine;
            uint16_t *src_ptr = srcYLine;
            uint16_t *src_ptr_fin = src_ptr + y_width;
            __m256i y0, y1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 16)), _mm_loadu_si128((__m128i*)(src_ptr +  0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 24)), _mm_loadu_si128((__m128i*)(src_ptr +  8)));

                y0 = _mm256_adds_epi16(y0, yrsftAdd);
                y1 = _mm256_adds_epi16(y1, yrsftAdd);

                y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
                y1 = _mm256_srli_epi16(y1, in_bit_depth - 8);

                y0 = _mm256_packus_epi16(y0, y1);

                _mm256_storeu_si256((__m256i *)(dst_ptr + 0), y0);
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
        __m256i y0, y1;
        for (; dst_ptr < dst_ptr_fin; src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            y0 = _mm256_loadu_si256((const __m256i *)src_u_ptr);
            y1 = _mm256_loadu_si256((const __m256i *)src_v_ptr);

            y0 = _mm256_adds_epi16(y0, yrsftAdd);
            y1 = _mm256_adds_epi16(y1, yrsftAdd);

            y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
            y1 = _mm256_slli_epi16(y1, 16 - in_bit_depth);
            const __m256i xMaskHighByte = _mm256_slli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
            y1 = _mm256_and_si256(y1, xMaskHighByte);

            y0 = _mm256_or_si256(y0, y1);

            _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
        }
    }
}
#pragma warning (pop)

void convert_yv12_16_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_14_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<14, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<12, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_10_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<10, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_09_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<9, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth>
void RGY_FORCEINLINE copy_y_plane(void *dst, int dst_y_pitch_byte, const void*src, int src_y_pitch_byte, const int width, const int* crop, const THREAD_Y_RANGE& y_range) {
    const int crop_left   = crop[0];
    const int crop_right  = crop[2];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());

    const Tin* srcYLine = (const Tin*)src + src_y_pitch * y_range.start_src + crop_left;
    Tout* dstLine = (Tout*)dst + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
            memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
        } else if (sizeof(Tin) == 2 && sizeof(Tout) == 1) {
            Tout* dst_ptr = dstLine;
            const Tin* src_ptr = srcYLine;
            const Tin* src_ptr_fin = src_ptr + y_width;
            __m256i y0, y1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 16)), _mm_loadu_si128((__m128i*)(src_ptr + 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 24)), _mm_loadu_si128((__m128i*)(src_ptr + 8)));

                y0 = _mm256_adds_epi16(y0, yrsftAdd);
                y1 = _mm256_adds_epi16(y1, yrsftAdd);

                y0 = _mm256_srli_epi16(y0, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, 0>());
                y1 = _mm256_srli_epi16(y1, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, 0>());

                y0 = _mm256_packus_epi16(y0, y1);

                _mm256_storeu_si256((__m256i*)(dst_ptr + 0), y0);
            }
        } else if (sizeof(Tin) == 1 && sizeof(Tout) == 2) {
            Tout* dst_ptr = (Tout*)dstLine;
            const Tin* src_ptr = srcYLine;
            const Tin* src_ptr_fin = src_ptr + y_width;
            __m256i y0, y1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                y0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src_ptr +  0)));
                y1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src_ptr + 16)));
                y0 = _mm256_slli_epi16(y0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
                y1 = _mm256_slli_epi16(y1, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
                _mm256_storeu_si256((__m256i*)(dst_ptr +  0), y0);
                _mm256_storeu_si256((__m256i*)(dst_ptr + 16), y1);
            }
        } else if (sizeof(Tin) == 2 && sizeof(Tout) == 2) {
            Tout* dst_ptr = dstLine;
            const Tin* src_ptr = srcYLine;
            const Tin* src_ptr_fin = src_ptr + y_width;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                __m256i y0 = _mm256_loadu_si256((const __m256i*)(src_ptr +  0));
                __m256i y1 = _mm256_loadu_si256((const __m256i*)(src_ptr + 16));
                if (out_bit_depth > in_bit_depth) {
                    y0 = _mm256_slli_epi16(y0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
                    y1 = _mm256_slli_epi16(y1, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
                } else if (out_bit_depth < in_bit_depth) {
                    const __m256i rsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<out_bit_depth, in_bit_depth, 0>());
                    y0 = _mm256_add_epi16(y0, rsftAdd);
                    y1 = _mm256_add_epi16(y1, rsftAdd);
                    y0 = _mm256_srli_epi16(y0, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, 0>());
                    y1 = _mm256_srli_epi16(y1, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, 0>());
                }
                _mm256_storeu_si256((__m256i*)(dst_ptr +  0), y0);
                _mm256_storeu_si256((__m256i*)(dst_ptr + 16), y1);
            }
        }
    }
}

void convert_nv12_to_yv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const int uv_width = y_width >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    copy_y_plane<uint8_t, 8, uint8_t, 8>(dst[0], dst_y_pitch_byte, src[0], src_y_pitch_byte, width, crop, y_range);

    // UV planes
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const uint8_t *srcUVline = (const uint8_t *)src[1] + src_uv_pitch_byte * uv_range.start_src + crop_left;
    uint8_t *dstUline = (uint8_t *)dst[1] + dst_uv_pitch_byte * uv_range.start_dst;
    uint8_t *dstVline = (uint8_t *)dst[2] + dst_uv_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcUVline += src_uv_pitch_byte, dstUline += dst_uv_pitch_byte, dstVline += dst_uv_pitch_byte) {
        const uint8_t *srcUV = srcUVline;
        uint8_t *dstU = dstUline;
        uint8_t *dstV = dstVline;
        for (int x = 0; x < uv_width; x += 32, dstU += 32, dstV += 32, srcUV += 64) {
            // 64バイト（32バイト*2）をロード
            __m256i uv0 = _mm256_loadu_si256((const __m256i *)(srcUV +  0));  // | 128 |   0 |
            __m256i uv1 = _mm256_loadu_si256((const __m256i *)(srcUV + 32));  // | 384 | 256 |

            // shuffle命令での入れ替え
            __m256i uv2 = _mm256_permute2x128_si256(uv0, uv1, (2 << 4) | 0); // | 256 |   0 |
            __m256i uv3 = _mm256_permute2x128_si256(uv0, uv1, (3 << 4) | 1); // | 384 | 128 |

            // UとVを分離
            __m256i u0 = _mm256_packus_epi16( // | 384 | 256 | 128 |  0 |
                _mm256_and_si256(uv2, _mm256_set1_epi16(0x00FF)),  // | 256 |   0 |
                _mm256_and_si256(uv3, _mm256_set1_epi16(0x00FF))); // | 384 | 128 |
            __m256i v0 = _mm256_packus_epi16(_mm256_srli_epi16(uv2, 8), _mm256_srli_epi16(uv3, 8));

            // ストア
            _mm256_storeu_si256((__m256i *)dstU, u0);
            _mm256_storeu_si256((__m256i *)dstV, v0);
        }
    }
    _mm256_zeroupper();
}

void convert_p010_to_yv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const int uv_width = y_width >> 1;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, 16, 0>());
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    copy_y_plane<uint16_t, 16, uint8_t, 8>(dst[0], dst_y_pitch_byte, src[0], src_y_pitch_byte, width, crop, y_range);

    // UV planes
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const uint16_t *srcUVline = (const uint16_t *)src[1] + (src_uv_pitch_byte / 2) * uv_range.start_src + crop_left;
    uint8_t *dstUline = (uint8_t *)dst[1] + dst_uv_pitch_byte * uv_range.start_dst;
    uint8_t *dstVline = (uint8_t *)dst[2] + dst_uv_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcUVline += (src_uv_pitch_byte / 2), dstUline += dst_uv_pitch_byte, dstVline += dst_uv_pitch_byte) {
        const uint16_t *srcUV = srcUVline;
        uint8_t *dstU = dstUline;
        uint8_t *dstV = dstVline;
        for (int x = 0; x < uv_width; x += 32, dstU += 32, dstV += 32, srcUV += 64) {
            __m256i uv0 = _mm256_loadu_si256((const __m256i *)(srcUV +  0));  // | 128 |   0 |
            __m256i uv1 = _mm256_loadu_si256((const __m256i *)(srcUV + 16));  // | 384 | 256 |
            __m256i uv2 = _mm256_loadu_si256((const __m256i *)(srcUV + 32));  // | 640 | 512 |
            __m256i uv3 = _mm256_loadu_si256((const __m256i *)(srcUV + 48));  // | 896 | 768 |
            uv0 = _mm256_srli_epi16(_mm256_adds_epi16(uv0, yrsftAdd), 8);
            uv1 = _mm256_srli_epi16(_mm256_adds_epi16(uv1, yrsftAdd), 8);
            uv2 = _mm256_srli_epi16(_mm256_adds_epi16(uv2, yrsftAdd), 8);
            uv3 = _mm256_srli_epi16(_mm256_adds_epi16(uv3, yrsftAdd), 8);
            __m256i uv01 = _mm256_packus_epi16(uv0, uv1);  // | 384 | 128 | 256 |   0 |
            __m256i uv23 = _mm256_packus_epi16(uv2, uv3);  // | 896 | 640 | 768 | 512 |

            __m256i u0 = _mm256_packus_epi16( // | 896 | 384 | 640 | 128 | 768 | 256 | 512 |   0 |
                _mm256_and_si256(uv0, _mm256_set1_epi16(0x00FF)),  // | 384 | 128 | 256 |   0 |
                _mm256_and_si256(uv1, _mm256_set1_epi16(0x00FF))); // | 896 | 640 | 768 | 512 |
            __m256i v0 = _mm256_packus_epi16(_mm256_srli_epi16(uv0, 8), _mm256_srli_epi16(uv1, 8));

            alignas(32) int perm[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            u0 = _mm256_permutevar8x32_epi32(u0, _mm256_load_si256((const __m256i *)perm));
            v0 = _mm256_permutevar8x32_epi32(v0, _mm256_load_si256((const __m256i *)perm));

            _mm256_storeu_si256((__m256i *)dstU, u0);
            _mm256_storeu_si256((__m256i *)dstV, v0);
        }
    }
}

template<int out_bit_depth>
void convert_nv12_to_yuv420_high_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const int uv_width = y_width >> 1;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<out_bit_depth, 8, 0>());
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    copy_y_plane<uint8_t, 8, uint16_t, out_bit_depth>(dst[0], dst_y_pitch_byte, src[0], src_y_pitch_byte, width, crop, y_range);

    // UV planes
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const uint8_t *srcUVline = (const uint8_t *)src[1] + src_uv_pitch_byte * uv_range.start_src + crop_left;
    uint16_t *dstUline = (uint16_t *)dst[1] + (dst_uv_pitch_byte / 2) * uv_range.start_dst;
    uint16_t *dstVline = (uint16_t *)dst[2] + (dst_uv_pitch_byte / 2) * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcUVline += src_uv_pitch_byte, dstUline += (dst_uv_pitch_byte / 2), dstVline += (dst_uv_pitch_byte / 2)) {
        const uint8_t *srcUV = srcUVline;
        uint16_t *dstU = dstUline;
        uint16_t *dstV = dstVline;
        for (int x = 0; x < uv_width; x += 16, dstU += 16, dstV += 16, srcUV += 32) {
            __m256i uv = _mm256_loadu_si256((const __m256i *)(srcUV));
            __m256i u16 = _mm256_slli_epi16(_mm256_and_si256(uv, _mm256_set1_epi16(0x00FF)), conv_bit_depth_lsft_<out_bit_depth, 8, 0>());
            __m256i v16 = _mm256_slli_epi16(_mm256_srli_epi16(uv, 8),                        conv_bit_depth_lsft_<out_bit_depth, 8, 0>());
            _mm256_storeu_si256((__m256i *)dstU, u16);
            _mm256_storeu_si256((__m256i *)dstV, v16);
        }
    }
}



void convert_nv12_to_yuv420_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv12_to_yuv420_high_avx2<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_nv12_to_yuv420_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv12_to_yuv420_high_avx2<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_nv12_to_yuv420_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv12_to_yuv420_high_avx2<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_nv12_to_yuv420_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv12_to_yuv420_high_avx2<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int out_bit_depth>
void convert_p010_to_yuv420_high_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const int uv_width = y_width >> 1;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<out_bit_depth, 16, 0>());
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    copy_y_plane<uint16_t, 16, uint16_t, out_bit_depth>(dst[0], dst_y_pitch_byte, src[0], src_y_pitch_byte, width, crop, y_range);

    // UV planes
    const auto mask0000ffff = _mm256_set1_epi32(0x0000ffff);
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const uint16_t *srcUVline = (const uint16_t *)src[1] + (src_uv_pitch_byte / 2) * uv_range.start_src + crop_left;
    uint16_t *dstUline = (uint16_t *)dst[1] + (dst_uv_pitch_byte / 2) * uv_range.start_dst;
    uint16_t *dstVline = (uint16_t *)dst[2] + (dst_uv_pitch_byte / 2) * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcUVline += (src_uv_pitch_byte / 2), dstUline += (dst_uv_pitch_byte / 2), dstVline += (dst_uv_pitch_byte / 2)) {
        const uint16_t *srcUV = srcUVline;
        uint16_t *dstU = dstUline;
        uint16_t *dstV = dstVline;
        for (int x = 0; x < uv_width; x += 16, dstU += 16, dstV += 16, srcUV += 32) {
            __m256i uv0 = _mm256_loadu_si256((const __m256i *)(srcUV +  0));
            __m256i uv1 = _mm256_loadu_si256((const __m256i *)(srcUV + 16));
            if (out_bit_depth < 16) {
                uv0 = _mm256_adds_epi16(uv0, yrsftAdd);
                uv1 = _mm256_adds_epi16(uv1, yrsftAdd);
                uv0 = _mm256_srli_epi16(uv0, 16 - out_bit_depth);
                uv1 = _mm256_srli_epi16(uv1, 16 - out_bit_depth);
            }
            // shuffle命令での入れ替え
            __m256i uv2 = _mm256_permute2x128_si256(uv0, uv1, (2 << 4) | 0); // | 256 |   0 |
            __m256i uv3 = _mm256_permute2x128_si256(uv0, uv1, (3 << 4) | 1); // | 384 | 128 |

            __m256i u0 = _mm256_packus_epi32( // | 384 | 256 | 128 |   0 |
                _mm256_and_si256(uv2, mask0000ffff),   // | 256 |   0 |
                _mm256_and_si256(uv3, mask0000ffff));  // | 384 | 128 |
            __m256i v0 = _mm256_packus_epi32(_mm256_srli_epi32(uv2, 16), _mm256_srli_epi32(uv3, 16));

            _mm256_storeu_si256((__m256i *)dstU, u0);
            _mm256_storeu_si256((__m256i *)dstV, v0);
        }
    }
}

void convert_p010_to_yuv420_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_p010_to_yuv420_high_avx2<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_p010_to_yuv420_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_p010_to_yuv420_high_avx2<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_p010_to_yuv420_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_p010_to_yuv420_high_avx2<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_p010_to_yuv420_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_p010_to_yuv420_high_avx2<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
void RGY_FORCEINLINE convert_yuv444_to_nv12_p_avx2_base(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    static_assert((sizeof(Tin) == 1 && in_bit_depth == 8) || (sizeof(Tin) == 2 && 8 < in_bit_depth && in_bit_depth <= 16), "invalid input bit depth.");
    static_assert((sizeof(Tout) == 1 && out_bit_depth == 8) || (sizeof(Tout) == 2 && 8 < out_bit_depth && out_bit_depth <= 16), "invalid output bit depth.");
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());
    //Y成分のコピー
    if (!uv_only) {
        copy_y_plane<Tin, in_bit_depth, Tout, out_bit_depth>(dst[0], dst_y_pitch_byte, src[0], src_y_pitch_byte, width, crop, y_range);
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin* srcULine = (Tin*)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tin* srcVLine = (Tin*)src[2] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tout* dstLine = (Tout*)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    for (int y = 0; y < y_range.len; y += 2, srcULine += src_uv_pitch * 2, srcVLine += src_uv_pitch * 2, dstLine += dst_y_pitch) {
        Tout* dstC = dstLine;
        Tin* srcU = srcULine;
        Tin* srcV = srcVLine;
        const int x_fin = width - crop_right - crop_left;
        //まずは8bit→8bit
        for (int x = 0; x < x_fin; x += 32, dstC += 32, srcU += 32, srcV += 32) {
            if (sizeof(Tout) == 1) {
                __m256i u0, v0;
                if (sizeof(Tin) == 1) {
                    __m256i u1, v1;
                    const auto mask00ff = _mm256_set1_epi16(0x00ff);
                    u0 = _mm256_and_si256( _mm256_loadu_si256((const __m256i*)(srcU + 0)), mask00ff);
                    v0 = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(srcV + 0)), 8);
                    u1 = _mm256_and_si256( _mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch + 0)), mask00ff);
                    v1 = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch + 0)), 8);
                    u0 = _mm256_avg_epu8(u0, u1); // 30 - 0
                    v0 = _mm256_avg_epu8(v0, v1); // 30 - 0
                } else {
                    const auto mask0000ffff = _mm256_set1_epi32(0x0000ffff);
                    __m256i u00 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU +  0)), mask0000ffff); // 14 -  0
                    __m256i u01 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + 16)), mask0000ffff); // 30 - 16
                    __m256i v00 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV +  0)), mask0000ffff);
                    __m256i v01 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + 16)), mask0000ffff);
                    __m256i u10 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch +  0)), mask0000ffff);
                    __m256i u11 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch + 16)), mask0000ffff);
                    __m256i v10 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch +  0)), mask0000ffff);
                    __m256i v11 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch + 16)), mask0000ffff);
                    u00 = _mm256_add_epi32(u00, u10);
                    u01 = _mm256_add_epi32(u01, u11);
                    v00 = _mm256_add_epi32(v00, v10);
                    v01 = _mm256_add_epi32(v01, v11);
                    const int shift_offset = 1;
                    const __m256i rsftAdd = _mm256_set1_epi32((short)conv_bit_depth_rsft_add_<out_bit_depth, in_bit_depth, shift_offset>());
                    u00 = _mm256_add_epi32(u00, rsftAdd); // 14 -  0
                    u01 = _mm256_add_epi32(u01, rsftAdd); // 30 - 16
                    v00 = _mm256_add_epi32(v00, rsftAdd);
                    v01 = _mm256_add_epi32(v01, rsftAdd);
                    u00 = _mm256_srli_epi32(u00, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    u01 = _mm256_srli_epi32(u01, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    v00 = _mm256_srli_epi32(v00, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    v01 = _mm256_srli_epi32(v01, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    u00 = _mm256_packus_epi32(u00, u01); // 30 - 24 | 14 -  8 | 22 - 16 | 6 - 0
                    v00 = _mm256_packus_epi32(v00, v01); // 30 - 24 | 14 -  8 | 22 - 16 | 6 - 0
                    u0 = _mm256_permute4x64_epi64(u00, _MM_SHUFFLE(3, 1, 2, 0));
                    v0 = _mm256_permute4x64_epi64(v00, _MM_SHUFFLE(3, 1, 2, 0));
                    v0 = _mm256_slli_epi16(v0, 8);
                }
                __m256i c0 = _mm256_or_si256(u0, v0);
                _mm256_storeu_si256((__m256i*)(dstC +  0), c0);
            } else if (sizeof(Tout) == 2) {
                __m256i u0, v0;
                if (sizeof(Tin) == 1) {
                    __m256i u1, v1;
                    const auto mask00ff = _mm256_set1_epi16(0x00ff);
                    u0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + 0)), mask00ff);
                    v0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + 0)), mask00ff);
                    u1 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch + 0)), mask00ff);
                    v1 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch + 0)), mask00ff);
#if 0
                    const int shift_offset = 1;
                    if (out_bit_depth > in_bit_depth + shift_offset) {
                        u0 = _mm256_slli_epi16(u0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v0 = _mm256_slli_epi16(v0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        u1 = _mm256_slli_epi16(u1, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v1 = _mm256_slli_epi16(v1, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    }
                    u0 = _mm256_adds_epu16(u0, u1); // 30 - 0
                    v0 = _mm256_adds_epu16(v0, v1); // 30 - 0
#else
                    u0 = _mm256_avg_epu8(u0, u1); // 30 - 0
                    v0 = _mm256_avg_epu8(v0, v1); // 30 - 0
                    u0 = _mm256_slli_epi16(u0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
                    v0 = _mm256_slli_epi16(v0, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, 0>());
#endif
                } else {
                    const auto mask0000ffff = _mm256_set1_epi32(0x0000ffff);
                    __m256i u00 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU +  0)), mask0000ffff); // 14 -  0
                    __m256i u01 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + 16)), mask0000ffff); // 30 - 16
                    __m256i v00 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV +  0)), mask0000ffff);
                    __m256i v01 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + 16)), mask0000ffff);
                    __m256i u10 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch +  0)), mask0000ffff);
                    __m256i u11 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcU + src_uv_pitch + 16)), mask0000ffff);
                    __m256i v10 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch +  0)), mask0000ffff);
                    __m256i v11 = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(srcV + src_uv_pitch + 16)), mask0000ffff);
                    u00 = _mm256_add_epi32(u00, u10);
                    u01 = _mm256_add_epi32(u01, u11);
                    v00 = _mm256_add_epi32(v00, v10);
                    v01 = _mm256_add_epi32(v01, v11);
                    const int shift_offset = 1;
                    if (out_bit_depth > in_bit_depth + shift_offset) {
                        u00 = _mm256_slli_epi32(u00, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        u01 = _mm256_slli_epi32(u01, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v00 = _mm256_slli_epi32(v00, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v01 = _mm256_slli_epi32(v01, conv_bit_depth_lsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    } else if (out_bit_depth < in_bit_depth + shift_offset) {
                        const __m256i rsftAdd = _mm256_set1_epi32((short)conv_bit_depth_rsft_add_<out_bit_depth, in_bit_depth, shift_offset>());
                        u00 = _mm256_add_epi32(u00, rsftAdd); // 14 -  0
                        u01 = _mm256_add_epi32(u01, rsftAdd); // 30 - 16
                        v00 = _mm256_add_epi32(v00, rsftAdd);
                        v01 = _mm256_add_epi32(v01, rsftAdd);
                        u00 = _mm256_srli_epi32(u00, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        u01 = _mm256_srli_epi32(u01, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v00 = _mm256_srli_epi32(v00, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                        v01 = _mm256_srli_epi32(v01, conv_bit_depth_rsft_<out_bit_depth, in_bit_depth, shift_offset>());
                    }
                    u00 = _mm256_packus_epi32(u00, u01); // 30 - 24 | 14 -  8 | 22 - 16 | 6 - 0
                    v00 = _mm256_packus_epi32(v00, v01); // 30 - 24 | 14 -  8 | 22 - 16 | 6 - 0
                    u0 = _mm256_permute4x64_epi64(u00, _MM_SHUFFLE(3, 1, 2, 0));
                    v0 = _mm256_permute4x64_epi64(v00, _MM_SHUFFLE(3, 1, 2, 0));
                }
                __m256i c0 = _mm256_unpacklo_epi16(u0, v0); // 22 - 16 |  6 -  0
                __m256i c1 = _mm256_unpackhi_epi16(u0, v0); // 30 - 24 | 14 -  8
                __m256i c2 = _mm256_permute2x128_si256(c0, c1, (2 << 4) + 0);
                __m256i c3 = _mm256_permute2x128_si256(c0, c1, (3 << 4) + 1);
                _mm256_storeu_si256((__m256i*)(dstC +  0), c2);
                _mm256_storeu_si256((__m256i*)(dstC + 16), c3);
            }
        }
    }
}

void convert_yuv444_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_09_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 9, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_10_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_12_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_14_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_16_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_09_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 9, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_10_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_12_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_14_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void convert_yuv444_16_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_to_nv12_p_avx2_base<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<int in_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_yv12_high_to_p010_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                avx2_memcpy<true>((uint8_t *)dstLine, (uint8_t *)srcYLine, y_width * (int)sizeof(uint16_t));
            } else {
                uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                    __m256i y0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                    y0 = _mm256_slli_epi16(y0, 16 - in_bit_depth);
                    _mm256_storeu_si256((__m256i *)dst_ptr, y0);
                }
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *dstLine = (uint16_t *)dst[1] + dst_y_pitch * uv_range.start_dst;;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        __m256i y0, y1, y2;
        for (int x = crop_left; x < x_fin; x += 32, src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            y0 = _mm256_loadu_si256((const __m256i *)src_u_ptr);
            y1 = _mm256_loadu_si256((const __m256i *)src_v_ptr);

            if (in_bit_depth < 16) {
                y0 = _mm256_slli_epi16(y0, 16 - in_bit_depth);
                y1 = _mm256_slli_epi16(y1, 16 - in_bit_depth);
            }

            y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3,1,2,0));
            y1 = _mm256_permute4x64_epi64(y1, _MM_SHUFFLE(3,1,2,0));

            y2 = _mm256_unpackhi_epi16(y0, y1);
            y0 = _mm256_unpacklo_epi16(y0, y1);

            _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y2);
        }
    }
}
#pragma warning (pop)

void convert_yv12_16_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_avx2_base<16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_14_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_avx2_base<14, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_avx2_base<12, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_10_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_avx2_base<10, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_09_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_avx2_base<9, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_to_ayuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte  * y_range.start_src + crop_left;
    uint8_t *srcULine = (uint8_t *)src[1] + src_uv_pitch_byte * y_range.start_src + crop_left;
    uint8_t *srcVLine = (uint8_t *)src[2] + src_uv_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        uint8_t *src_y_ptr = srcYLine;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        for (int x = 0; x < y_width; x += 32, src_y_ptr += 32, src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 128) {
            __m256i pixY = _mm256_loadu_si256((const __m256i *)src_y_ptr); // 31 - 0
            __m256i pixU = _mm256_loadu_si256((const __m256i *)src_u_ptr); // 31 - 0
            __m256i pixV = _mm256_loadu_si256((const __m256i *)src_v_ptr); // 31 - 0

            __m256i pixAY0 = _mm256_unpacklo_epi8(pixY, _mm256_setzero_si256()); // 23 - 16 |  7 - 0
            __m256i pixAY1 = _mm256_unpackhi_epi8(pixY, _mm256_setzero_si256()); // 31 - 24 | 15 - 8
            __m256i pixUV0 = _mm256_unpacklo_epi8(pixV, pixU); // 23 - 16 |  7 - 0
            __m256i pixUV1 = _mm256_unpackhi_epi8(pixV, pixU); // 31 - 24 | 15 - 8
            __m256i pixVUYA0 = _mm256_unpacklo_epi16(pixAY0, pixUV0); // 19 - 16 |  3 -  0
            __m256i pixVUYA1 = _mm256_unpackhi_epi16(pixAY0, pixUV0); // 23 - 20 |  7 -  4
            __m256i pixVUYA2 = _mm256_unpacklo_epi16(pixAY1, pixUV1); // 27 - 24 | 11 -  8
            __m256i pixVUYA3 = _mm256_unpackhi_epi16(pixAY1, pixUV1); // 31 - 28 | 15 - 12

            _mm256_storeu_si256((__m256i *)(dst_ptr +  0), _mm256_permute2x128_si256(pixVUYA0, pixVUYA1, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i *)(dst_ptr + 32), _mm256_permute2x128_si256(pixVUYA2, pixVUYA3, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i *)(dst_ptr + 64), _mm256_permute2x128_si256(pixVUYA0, pixVUYA1, (3 << 4) + 1));
            _mm256_storeu_si256((__m256i *)(dst_ptr + 96), _mm256_permute2x128_si256(pixVUYA2, pixVUYA3, (3 << 4) + 1));
        }
    }
}

template<int in_bit_depth>
static void RGY_FORCEINLINE copy_yuv444_high_to_ayuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const __m256i xrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());
    uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch  * y_range.start_src + crop_left;
    uint16_t *srcULine = (uint16_t *)src[1] + src_uv_pitch * y_range.start_src + crop_left;
    uint16_t *srcVLine = (uint16_t *)src[2] + src_uv_pitch * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0]  + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch_byte) {
        uint16_t *src_y_ptr = srcYLine;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        for (int x = 0; x < y_width; x += 32, src_y_ptr += 32, src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 128) {
            __m256i pixY0 = _mm256_loadu_si256((const __m256i *)(src_y_ptr + 0)); // 15 -  0
            __m256i pixU0 = _mm256_loadu_si256((const __m256i *)(src_u_ptr + 0)); // 15 -  0
            __m256i pixV0 = _mm256_loadu_si256((const __m256i *)(src_v_ptr + 0)); // 15 -  0
            __m256i pixY1 = _mm256_loadu_si256((const __m256i *)(src_y_ptr + 8)); // 31 - 16
            __m256i pixU1 = _mm256_loadu_si256((const __m256i *)(src_u_ptr + 8)); // 31 - 16
            __m256i pixV1 = _mm256_loadu_si256((const __m256i *)(src_v_ptr + 8)); // 31 - 16
            pixY0 = _mm256_adds_epi16(pixY0, xrsftAdd);
            pixU0 = _mm256_adds_epi16(pixU0, xrsftAdd);
            pixV0 = _mm256_adds_epi16(pixV0, xrsftAdd);
            pixY1 = _mm256_adds_epi16(pixY1, xrsftAdd);
            pixU1 = _mm256_adds_epi16(pixU1, xrsftAdd);
            pixV1 = _mm256_adds_epi16(pixV1, xrsftAdd);
            pixY0 = _mm256_srli_epi16(pixY0, in_bit_depth - 8);
            pixU0 = _mm256_srli_epi16(pixU0, in_bit_depth - 8);
            pixV0 = _mm256_srli_epi16(pixV0, in_bit_depth - 8);
            pixY1 = _mm256_srli_epi16(pixY1, in_bit_depth - 8);
            pixU1 = _mm256_srli_epi16(pixU1, in_bit_depth - 8);
            pixV1 = _mm256_srli_epi16(pixV1, in_bit_depth - 8);
            __m256i pixY = _mm256_packus_epi16(pixY0, pixY1); // 31 - 24 | 15 - 8 | 23 - 16 |  7 - 0
            __m256i pixU = _mm256_packus_epi16(pixU0, pixU1); // 31 - 24 | 15 - 8 | 23 - 16 |  7 - 0
            __m256i pixV = _mm256_packus_epi16(pixV0, pixV1); // 31 - 24 | 15 - 8 | 23 - 16 |  7 - 0
            pixY = _mm256_permute4x64_epi64(pixY, _MM_SHUFFLE(3,1,2,0)); // 31 - 0
            pixU = _mm256_permute4x64_epi64(pixU, _MM_SHUFFLE(3,1,2,0)); // 31 - 0
            pixV = _mm256_permute4x64_epi64(pixV, _MM_SHUFFLE(3,1,2,0)); // 31 - 0

            __m256i pixAY0 = _mm256_unpacklo_epi8(pixY, _mm256_setzero_si256()); // 23 - 16 |  7 - 0
            __m256i pixAY1 = _mm256_unpackhi_epi8(pixY, _mm256_setzero_si256()); // 31 - 24 | 15 - 8
            __m256i pixUV0 = _mm256_unpacklo_epi8(pixV, pixU); // 23 - 16 |  7 - 0
            __m256i pixUV1 = _mm256_unpackhi_epi8(pixV, pixU); // 31 - 24 | 15 - 8
            __m256i pixVUYA0 = _mm256_unpacklo_epi16(pixAY0, pixUV0); // 19 - 16 |  3 -  0
            __m256i pixVUYA1 = _mm256_unpackhi_epi16(pixAY0, pixUV0); // 23 - 20 |  7 -  4
            __m256i pixVUYA2 = _mm256_unpacklo_epi16(pixAY1, pixUV1); // 27 - 24 | 11 -  8
            __m256i pixVUYA3 = _mm256_unpackhi_epi16(pixAY1, pixUV1); // 31 - 28 | 15 - 12

            _mm256_storeu_si256((__m256i*)(dst_ptr +  0), _mm256_permute2x128_si256(pixVUYA0, pixVUYA1, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 32), _mm256_permute2x128_si256(pixVUYA2, pixVUYA3, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 64), _mm256_permute2x128_si256(pixVUYA0, pixVUYA1, (3 << 4) + 1));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 96), _mm256_permute2x128_si256(pixVUYA2, pixVUYA3, (3 << 4) + 1));
        }
    }
}

void copy_yuv444_16_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    copy_yuv444_high_to_ayuv444_avx2<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_14_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    copy_yuv444_high_to_ayuv444_avx2<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_12_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    copy_yuv444_high_to_ayuv444_avx2<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_10_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    copy_yuv444_high_to_ayuv444_avx2<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_09_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    copy_yuv444_high_to_ayuv444_avx2<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    const int in_bit_depth = 8;
    const int out_bit_depth = 10;
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte;
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(uint32_t);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t* srcYLine = (uint8_t*)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint8_t* srcULine = (uint8_t*)src[1] + src_y_pitch * y_range.start_src + crop_left;
    uint8_t* srcVLine = (uint8_t*)src[2] + src_y_pitch * y_range.start_src + crop_left;
    uint32_t* dstLine = (uint32_t*)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, srcULine += src_y_pitch, srcVLine += src_y_pitch, dstLine += dst_y_pitch) {
        uint8_t* src_y_ptr = srcYLine;
        uint8_t* src_u_ptr = srcULine;
        uint8_t* src_v_ptr = srcVLine;
        uint32_t* dst_ptr = dstLine;
        for (int x = 0; x < y_width; x++, src_y_ptr += 32, src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 32) {
            __m256i pixY = _mm256_loadu_si256((const __m256i*)(src_y_ptr + 0));
            __m256i pixU = _mm256_loadu_si256((const __m256i*)(src_u_ptr + 0));
            __m256i pixV = _mm256_loadu_si256((const __m256i*)(src_v_ptr + 0));

            __m256i pixY0 = _mm256_unpacklo_epi8(pixY, _mm256_setzero_si256()); // 23 - 16 |  7 - 0
            __m256i pixY1 = _mm256_unpackhi_epi8(pixY, _mm256_setzero_si256()); // 31 - 24 | 15 - 8
            __m256i pixY410_0 = _mm256_slli_epi32(_mm256_unpacklo_epi16(pixY0, _mm256_setzero_si256()), 10+(out_bit_depth-in_bit_depth)); // 19 - 16 |  3 -  0
            __m256i pixY410_1 = _mm256_slli_epi32(_mm256_unpackhi_epi16(pixY0, _mm256_setzero_si256()), 10+(out_bit_depth-in_bit_depth)); // 23 - 20 |  7 -  4
            __m256i pixY410_2 = _mm256_slli_epi32(_mm256_unpacklo_epi16(pixY1, _mm256_setzero_si256()), 10+(out_bit_depth-in_bit_depth)); // 27 - 24 | 11 -  8
            __m256i pixY410_3 = _mm256_slli_epi32(_mm256_unpackhi_epi16(pixY1, _mm256_setzero_si256()), 10+(out_bit_depth-in_bit_depth)); // 31 - 28 | 15 - 12

            __m256i pixU0 = _mm256_unpacklo_epi8(pixU, _mm256_setzero_si256()); // 23 - 16 |  7 - 0
            __m256i pixU1 = _mm256_unpackhi_epi8(pixU, _mm256_setzero_si256()); // 31 - 24 | 15 - 8
            pixY410_0 = _mm256_or_si256(pixY410_0, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixU0, _mm256_setzero_si256()), (out_bit_depth - in_bit_depth))); // 19 - 16 |  3 -  0
            pixY410_1 = _mm256_or_si256(pixY410_1, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixU0, _mm256_setzero_si256()), (out_bit_depth - in_bit_depth))); // 23 - 20 |  7 -  4
            pixY410_2 = _mm256_or_si256(pixY410_2, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixU1, _mm256_setzero_si256()), (out_bit_depth - in_bit_depth))); // 27 - 24 | 11 -  8
            pixY410_3 = _mm256_or_si256(pixY410_3, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixU1, _mm256_setzero_si256()), (out_bit_depth - in_bit_depth))); // 31 - 28 | 15 - 12

            __m256i pixV0 = _mm256_unpacklo_epi8(pixV, _mm256_setzero_si256()); // 23 - 16 |  7 - 0
            __m256i pixV1 = _mm256_unpackhi_epi8(pixV, _mm256_setzero_si256()); // 31 - 24 | 15 - 8
            pixY410_0 = _mm256_or_si256(pixY410_0, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixV0, _mm256_setzero_si256()), 20+(out_bit_depth-in_bit_depth))); // 19 - 16 |  3 -  0
            pixY410_1 = _mm256_or_si256(pixY410_1, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixV0, _mm256_setzero_si256()), 20+(out_bit_depth-in_bit_depth))); // 23 - 20 |  7 -  4
            pixY410_2 = _mm256_or_si256(pixY410_2, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixV1, _mm256_setzero_si256()), 20+(out_bit_depth-in_bit_depth))); // 27 - 24 | 11 -  8
            pixY410_3 = _mm256_or_si256(pixY410_3, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixV1, _mm256_setzero_si256()), 20+(out_bit_depth-in_bit_depth))); // 31 - 28 | 15 - 12

            _mm256_storeu_si256((__m256i*)(dst_ptr +  0), _mm256_permute2x128_si256(pixY410_0, pixY410_1, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr +  8), _mm256_permute2x128_si256(pixY410_2, pixY410_3, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 16), _mm256_permute2x128_si256(pixY410_0, pixY410_1, (3 << 4) + 1));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 24), _mm256_permute2x128_si256(pixY410_2, pixY410_3, (3 << 4) + 1));
        }
    }
}

template<int in_bit_depth>
void convert_yuv444_high_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    static_assert(10 <= in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 10-16.");
    const int out_bit_depth = 10;
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(uint16_t);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(uint32_t);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const __m256i xrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<10, in_bit_depth, 0>());
    uint16_t* srcYLine = (uint16_t*)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t* srcULine = (uint16_t*)src[1] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t* srcVLine = (uint16_t*)src[2] + src_y_pitch * y_range.start_src + crop_left;
    uint32_t* dstLine = (uint32_t*)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, srcULine += src_y_pitch, srcVLine += src_y_pitch, dstLine += dst_y_pitch) {
        uint16_t* src_y_ptr = srcYLine;
        uint16_t* src_u_ptr = srcULine;
        uint16_t* src_v_ptr = srcVLine;
        uint32_t* dst_ptr = dstLine;
        for (int x = 0; x < y_width; x++, src_y_ptr += 32, src_u_ptr += 32, src_v_ptr += 32, dst_ptr += 32) {
            __m256i pixY0 = _mm256_loadu_si256((const __m256i*)(src_y_ptr + 0)); // 15 -  0
            __m256i pixY1 = _mm256_loadu_si256((const __m256i*)(src_y_ptr + 8)); // 31 - 16
            __m256i pixU0 = _mm256_loadu_si256((const __m256i*)(src_u_ptr + 0)); // 15 -  0
            __m256i pixU1 = _mm256_loadu_si256((const __m256i*)(src_u_ptr + 8)); // 31 - 16
            __m256i pixV0 = _mm256_loadu_si256((const __m256i*)(src_v_ptr + 0)); // 15 -  0
            __m256i pixV1 = _mm256_loadu_si256((const __m256i*)(src_v_ptr + 8)); // 31 - 16

            if (in_bit_depth > out_bit_depth) {
                pixY0 = _mm256_srli_epi16(_mm256_add_epi16(pixY0, xrsftAdd), in_bit_depth - out_bit_depth);
                pixY1 = _mm256_srli_epi16(_mm256_add_epi16(pixY1, xrsftAdd), in_bit_depth - out_bit_depth);
                pixU0 = _mm256_srli_epi16(_mm256_add_epi16(pixU0, xrsftAdd), in_bit_depth - out_bit_depth);
                pixU1 = _mm256_srli_epi16(_mm256_add_epi16(pixU1, xrsftAdd), in_bit_depth - out_bit_depth);
                pixV0 = _mm256_srli_epi16(_mm256_add_epi16(pixV0, xrsftAdd), in_bit_depth - out_bit_depth);
                pixV1 = _mm256_srli_epi16(_mm256_add_epi16(pixV1, xrsftAdd), in_bit_depth - out_bit_depth);
            }
            pixY0 = _mm256_min_epu16(pixY0, _mm256_set1_epi16((1<<out_bit_depth)-1));
            pixY1 = _mm256_min_epu16(pixY1, _mm256_set1_epi16((1<<out_bit_depth)-1));
            pixU0 = _mm256_min_epu16(pixU0, _mm256_set1_epi16((1<<out_bit_depth)-1));
            pixU1 = _mm256_min_epu16(pixU1, _mm256_set1_epi16((1<<out_bit_depth)-1));
            pixV0 = _mm256_min_epu16(pixV0, _mm256_set1_epi16((1<<out_bit_depth)-1));
            pixV1 = _mm256_min_epu16(pixV1, _mm256_set1_epi16((1<<out_bit_depth)-1));

            __m256i pixY410_0 = _mm256_slli_epi32(_mm256_unpacklo_epi16(pixY0, _mm256_setzero_si256()), 10); // 11 -  8 |  3 -  0
            __m256i pixY410_1 = _mm256_slli_epi32(_mm256_unpackhi_epi16(pixY0, _mm256_setzero_si256()), 10); // 15 - 12 |  7 -  4
            __m256i pixY410_2 = _mm256_slli_epi32(_mm256_unpacklo_epi16(pixY1, _mm256_setzero_si256()), 10); // 27 - 24 | 19 - 16
            __m256i pixY410_3 = _mm256_slli_epi32(_mm256_unpackhi_epi16(pixY1, _mm256_setzero_si256()), 10); // 31 - 28 | 23 - 20

            pixY410_0 = _mm256_or_si256(pixY410_0, _mm256_unpacklo_epi16(pixU0, _mm256_setzero_si256()));
            pixY410_1 = _mm256_or_si256(pixY410_1, _mm256_unpackhi_epi16(pixU0, _mm256_setzero_si256()));
            pixY410_2 = _mm256_or_si256(pixY410_2, _mm256_unpacklo_epi16(pixU1, _mm256_setzero_si256()));
            pixY410_3 = _mm256_or_si256(pixY410_3, _mm256_unpackhi_epi16(pixU1, _mm256_setzero_si256()));

            pixY410_0 = _mm256_or_si256(pixY410_0, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixV0, _mm256_setzero_si256()), 20));
            pixY410_1 = _mm256_or_si256(pixY410_1, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixV0, _mm256_setzero_si256()), 20));
            pixY410_2 = _mm256_or_si256(pixY410_2, _mm256_slli_epi32(_mm256_unpacklo_epi16(pixV1, _mm256_setzero_si256()), 20));
            pixY410_3 = _mm256_or_si256(pixY410_3, _mm256_slli_epi32(_mm256_unpackhi_epi16(pixV1, _mm256_setzero_si256()), 20));

            _mm256_storeu_si256((__m256i*)(dst_ptr +  0), _mm256_permute2x128_si256(pixY410_0, pixY410_1, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr +  8), _mm256_permute2x128_si256(pixY410_0, pixY410_1, (3 << 4) + 1));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 16), _mm256_permute2x128_si256(pixY410_2, pixY410_3, (2 << 4) + 0));
            _mm256_storeu_si256((__m256i*)(dst_ptr + 24), _mm256_permute2x128_si256(pixY410_2, pixY410_3, (3 << 4) + 1));
        }
    }
}

void convert_yuv444_16_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_avx2<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_avx2<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_avx2<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_avx2<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void copy_yuv444_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            avx2_memcpy<true>(dstLine, srcYLine, y_width);
        }
    }
}

template<int in_bit_depth, int out_bit_depth>
static void RGY_FORCEINLINE convert_yuv444_high_to_yuv444_high_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    static_assert(8 < out_bit_depth && out_bit_depth <= 16, "out_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        const uint16_t *srcYLine = (const uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == out_bit_depth) {
                avx2_memcpy<true>((uint8_t *)dstLine, (const uint8_t *)srcYLine, y_width * (int)sizeof(uint16_t));
            } else if (out_bit_depth > in_bit_depth) {
                const uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                    __m256i x0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                    if (out_bit_depth > in_bit_depth) {
                        x0 = _mm256_slli_epi16(x0, std::max(out_bit_depth - in_bit_depth, 0));
                    } else {
                        x0 = _mm256_srli_epi16(x0, std::max(in_bit_depth - out_bit_depth, 0));
                    }
                    _mm256_storeu_si256((__m256i *)dst_ptr, x0);
                }
            }
        }
    }
}
#pragma warning(pop)

void convert_yuv444_16_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<16, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<14, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<12, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<10, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<9, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_16_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<16, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<14, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<12, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<10, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<9, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_16_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<16, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<14, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<12, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<10, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<9, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_16_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<16, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<14, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<12, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<10, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_high_avx2_base<9, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<int out_bit_depth>
void convert_yuv444_to_yuv444_high_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < out_bit_depth && out_bit_depth <= 16, "out_bit_depth must be 9-16.");
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
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0, y1;
                y0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3,1,2,0));
                y1 = _mm256_unpackhi_epi8(_mm256_setzero_si256(), y0);
                y0 = _mm256_unpacklo_epi8(_mm256_setzero_si256(), y0);
                if (out_bit_depth < 16) {
                    y0 = _mm256_srli_epi16(y0, 16 - out_bit_depth);
                    y1 = _mm256_srli_epi16(y1, 16 - out_bit_depth);
                }
                _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y1);
            }
        }
    }
}

void convert_yuv444_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_yuv444_high_avx2_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_yuv444_14_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_yuv444_high_avx2_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_yuv444_12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_yuv444_high_avx2_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_yuv444_10_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_yuv444_high_avx2_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int in_bit_depth>
static void RGY_FORCEINLINE convert_yuv444_high_to_yuv444_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const __m256i yrsftAdd = _mm256_set1_epi16((short)conv_bit_depth_rsft_add_<8, in_bit_depth, 0>());
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint16_t *srcYLine = (uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint16_t *src_ptr = srcYLine;
            uint8_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 16), (const __m128i *)(src_ptr +  0));
                __m256i y1 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 24), (const __m128i *)(src_ptr +  8));
                y0 = _mm256_adds_epi16(y0, yrsftAdd);
                y1 = _mm256_adds_epi16(y1, yrsftAdd);
                y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
                y1 = _mm256_srli_epi16(y1, in_bit_depth - 8);
                y0 = _mm256_packus_epi16(y0, y1);
                _mm256_storeu_si256((__m256i *)dst_ptr, y0);
            }
        }
    }
}
#pragma warning(pop)

void convert_yuv444_16_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#include "convert_const.h"

static RGY_FORCEINLINE void gather_y_uv_from_yc48(__m256i& y0, __m256i& y1, __m256i y2) {
    const int MASK_INT_Y  = 0x80 + 0x10 + 0x02;
    const int MASK_INT_UV = 0x40 + 0x20 + 0x01;
    __m256i y3 = y0;
    __m256i y4 = y1;
    __m256i y5 = y2;

    y0 = _mm256_blend_epi32(y3, y4, 0xf0);                    // 384, 0
    y1 = _mm256_permute2x128_si256(y3, y5, (0x02<<4) + 0x01); // 512, 128
    y2 = _mm256_blend_epi32(y4, y5, 0xf0);                    // 640, 256

    y3 = _mm256_blend_epi16(y0, y1, MASK_INT_Y);
    y3 = _mm256_blend_epi16(y3, y2, MASK_INT_Y>>2);

    y1 = _mm256_blend_epi16(y0, y1, MASK_INT_UV);
    y1 = _mm256_blend_epi16(y1, y2, MASK_INT_UV>>2);
    y1 = _mm256_alignr_epi8(y1, y1, 2);
    y1 = _mm256_shuffle_epi32(y1, _MM_SHUFFLE(1, 2, 3, 0));//UV1行目

    y0 = _mm256_shuffle_epi8(y3, yC_SUFFLE_YCP_Y);
}

static RGY_FORCEINLINE __m256i convert_y_range_from_yc48(__m256i y0, __m256i yC_Y_MA_16, int Y_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    __m256i y7;

    y7 = _mm256_unpackhi_epi16(y0, yC_pw_one);
    y0 = _mm256_unpacklo_epi16(y0, yC_pw_one);

    y0 = _mm256_madd_epi16(y0, yC_Y_MA_16);
    y7 = _mm256_madd_epi16(y7, yC_Y_MA_16);
    y0 = _mm256_srai_epi32(y0, Y_RSH_16);
    y7 = _mm256_srai_epi32(y7, Y_RSH_16);
    y0 = _mm256_add_epi32(y0, yC_YCC);
    y7 = _mm256_add_epi32(y7, yC_YCC);

    y0 = _mm256_packus_epi32(y0, y7);

    return y0;
}
static RGY_FORCEINLINE __m256i convert_uv_range_after_adding_offset(__m256i y0, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    __m256i y7;
    y7 = _mm256_unpackhi_epi16(y0, yC_pw_one);
    y0 = _mm256_unpacklo_epi16(y0, yC_pw_one);

    y0 = _mm256_madd_epi16(y0, yC_UV_MA_16);
    y7 = _mm256_madd_epi16(y7, yC_UV_MA_16);
    y0 = _mm256_srai_epi32(y0, UV_RSH_16);
    y7 = _mm256_srai_epi32(y7, UV_RSH_16);
    y0 = _mm256_add_epi32(y0, yC_YCC);
    y7 = _mm256_add_epi32(y7, yC_YCC);

    y0 = _mm256_packus_epi32(y0, y7);

    return y0;
}
static RGY_FORCEINLINE __m256i convert_uv_range_from_yc48(__m256i y0, const __m256i& yC_UV_OFFSET_x1, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x1);

    return convert_uv_range_after_adding_offset(y0, yC_UV_MA_16, UV_RSH_16, yC_YCC, yC_pw_one);
}
static RGY_FORCEINLINE __m256i convert_uv_range_from_yc48_yuv420p(__m256i y0, __m256i y1, const __m256i& yC_UV_OFFSET_x2, __m256i yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    y0 = _mm256_add_epi16(y0, y1);
    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x2);

    return convert_uv_range_after_adding_offset(y0, yC_UV_MA_16, UV_RSH_16, yC_YCC, yC_pw_one);
}
static RGY_FORCEINLINE __m256i convert_uv_range_from_yc48_420i(__m256i y0, __m256i y1, const __m256i& yC_UV_OFFSET_x1, const __m256i& yC_UV_MA_16_0, const __m256i& yC_UV_MA_16_1, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    __m256i y2, y3, y6, y7;

    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x1);
    y1 = _mm256_add_epi16(y1, yC_UV_OFFSET_x1);

    y7 = _mm256_unpackhi_epi16(y0, yC_pw_one);
    y6 = _mm256_unpacklo_epi16(y0, yC_pw_one);
    y3 = _mm256_unpackhi_epi16(y1, yC_pw_one);
    y2 = _mm256_unpacklo_epi16(y1, yC_pw_one);

    y6 = _mm256_madd_epi16(y6, yC_UV_MA_16_0);
    y7 = _mm256_madd_epi16(y7, yC_UV_MA_16_0);
    y2 = _mm256_madd_epi16(y2, yC_UV_MA_16_1);
    y3 = _mm256_madd_epi16(y3, yC_UV_MA_16_1);
    y0 = _mm256_add_epi32(y6, y2);
    y7 = _mm256_add_epi32(y7, y3);
    y0 = _mm256_srai_epi32(y0, UV_RSH_16);
    y7 = _mm256_srai_epi32(y7, UV_RSH_16);
    y0 = _mm256_add_epi32(y0, yC_YCC);
    y7 = _mm256_add_epi32(y7, yC_YCC);

    y0 = _mm256_packus_epi32(y0, y7);

    return y0;
}

static RGY_FORCEINLINE __m256i convert_y_range_to_yc48(__m256i y0) {
    //coef = 4788
    //((( y - 32768 ) * coef) >> 16 ) + (coef/2 - 299)
    const __m256i yC_0x8000 = _mm256_slli_epi16(_mm256_cmpeq_epi32(y0, y0), 15);
    y0 = _mm256_add_epi16(y0, yC_0x8000); // -32768
    y0 = _mm256_mulhi_epi16(y0, _mm256_set1_epi16(4788));
    y0 = _mm256_adds_epi16(y0, _mm256_set1_epi16(4788/2 - 299));
    return y0;
}

static RGY_FORCEINLINE __m256i convert_uv_range_to_yc48(__m256i y0) {
    //coeff = 4682
    //UV = (( uv - 32768 ) * coef + (1<<15) ) >> 16
    const __m256i yC_coeff = _mm256_unpacklo_epi16(_mm256_set1_epi16(4682), _mm256_set1_epi16(-1));
    const __m256i yC_0x8000 = _mm256_slli_epi16(_mm256_cmpeq_epi32(y0, y0), 15);
    __m256i y1;
    y0 = _mm256_add_epi16(y0, yC_0x8000); // -32768
    y1 = _mm256_unpackhi_epi16(y0, yC_0x8000);
    y0 = _mm256_unpacklo_epi16(y0, yC_0x8000);
    y0 = _mm256_madd_epi16(y0, yC_coeff);
    y1 = _mm256_madd_epi16(y1, yC_coeff);
    y0 = _mm256_srai_epi32(y0, 16);
    y1 = _mm256_srai_epi32(y1, 16);
    y0 = _mm256_packs_epi32(y0, y1);
    return y0;
}

static RGY_FORCEINLINE void gather_y_u_v_from_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
    __m256i y3, y4, y5;
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    y3 = _mm256_blend_epi32(y0, y1, 0xf0);                    // 384, 0
    y4 = _mm256_permute2x128_si256(y0, y2, (0x02<<4) + 0x01); // 512, 128
    y5 = _mm256_blend_epi32(y1, y2, 0xf0);                    // 640, 256

    y0 = _mm256_blend_epi16(y5, y3, MASK_INT);
    y1 = _mm256_blend_epi16(y4, y5, MASK_INT);
    y2 = _mm256_blend_epi16(y3, y4, MASK_INT);

    y0 = _mm256_blend_epi16(y0, y4, MASK_INT<<1);
    y1 = _mm256_blend_epi16(y1, y3, MASK_INT<<1);
    y2 = _mm256_blend_epi16(y2, y5, MASK_INT<<1);

    y0 = _mm256_shuffle_epi8(y0, yC_SUFFLE_YCP_Y);
    y1 = _mm256_shuffle_epi8(y1, _mm256_alignr_epi8(yC_SUFFLE_YCP_Y, yC_SUFFLE_YCP_Y, 6));
    y2 = _mm256_shuffle_epi8(y2, _mm256_alignr_epi8(yC_SUFFLE_YCP_Y, yC_SUFFLE_YCP_Y, 12));
}


static RGY_FORCEINLINE void gather_y_u_v_to_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
    __m256i y3, y4, y5;

    alignas(16) static const uint8_t shuffle_yc48[32] = {
        0x00, 0x01, 0x06, 0x07, 0x0C, 0x0D, 0x02, 0x03, 0x08, 0x09, 0x0E, 0x0F, 0x04, 0x05, 0x0A, 0x0B,
        0x00, 0x01, 0x06, 0x07, 0x0C, 0x0D, 0x02, 0x03, 0x08, 0x09, 0x0E, 0x0F, 0x04, 0x05, 0x0A, 0x0B
    };
    y5 = _mm256_load_si256((__m256i *)shuffle_yc48);
    y0 = _mm256_shuffle_epi8(y0, y5);                             //5,2,7,4,1,6,3,0
    y1 = _mm256_shuffle_epi8(y1, _mm256_alignr_epi8(y5, y5, 14)); //2,7,4,1,6,3,0,5
    y2 = _mm256_shuffle_epi8(y2, _mm256_alignr_epi8(y5, y5, 12)); //7,4,1,6,3,0,5,2

    y3 = _mm256_blend_epi16(y0, y1, 0x80 + 0x10 + 0x02);
    y3 = _mm256_blend_epi16(y3, y2, 0x20 + 0x04);        //384, 0

    y4 = _mm256_blend_epi16(y2, y1, 0x20 + 0x04);
    y4 = _mm256_blend_epi16(y4, y0, 0x80 + 0x10 + 0x02); //512, 128

    y2 = _mm256_blend_epi16(y2, y0, 0x20 + 0x04);
    y2 = _mm256_blend_epi16(y2, y1, 0x40 + 0x08 + 0x01); //640, 256

    y0 = _mm256_permute2x128_si256(y3, y4, (0x02<<4) + 0x00); // 128, 0
    y1 = _mm256_blend_epi32(y2, y3, 0xf0);                    // 384, 256
    y2 = _mm256_permute2x128_si256(y4, y2, (0x03<<4) + 0x01); // 640, 512
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void convert_yc48_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int x, y;
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    __m256i y0, y1, y2, y3;
    for (y = y_range.start_src; y < (y_range.start_src + y_range.len); y += 2) {
        ycp = (short*)pixel + src_y_pitch * y;
        ycpw= ycp + src_y_pitch;
        Y   = (short*)dst_Y + dst_y_pitch * y;
        C   = (short*)dst_C + dst_y_pitch * y / 2;
        for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
            y1 = _mm256_loadu_si256((__m256i *)(ycp +  0)); // 128, 0
            y2 = _mm256_loadu_si256((__m256i *)(ycp + 16)); // 384, 256
            y3 = _mm256_loadu_si256((__m256i *)(ycp + 32)); // 640, 512

            gather_y_uv_from_yc48(y1, y2, y3);
            y0 = y2;

            _mm256_storeu_si256((__m256i *)(Y + x), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            y1 = _mm256_loadu_si256((__m256i *)(ycpw +  0));
            y2 = _mm256_loadu_si256((__m256i *)(ycpw + 16));
            y3 = _mm256_loadu_si256((__m256i *)(ycpw + 32));

            gather_y_uv_from_yc48(y1, y2, y3);

            _mm256_storeu_si256((__m256i *)(Y + x + dst_y_pitch), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            _mm256_storeu_si256((__m256i *)(C + x), convert_uv_range_from_yc48_yuv420p(y0, y2,  _mm256_set1_epi16(UV_OFFSET_x2), yC_UV_L_MA_16_420P, UV_L_RSH_16_420P, yC_YCC, yC_pw_one));
        }
    }
    _mm256_zeroupper();
}

void convert_yc48_to_p010_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int x, y, i;
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    __m256i y0, y1, y2, y3;
    for (y = y_range.start_src; y < (y_range.start_src + y_range.len); y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + src_y_pitch * (y + i);
            ycpw= ycp + src_y_pitch*2;
            Y   = (short*)dst_Y + dst_y_pitch * (y + i);
            C   = (short*)dst_C + dst_y_pitch * (y + i*2) / 2;
            for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
                y1 = _mm256_loadu_si256((__m256i *)(ycp +  0)); // 128, 0
                y2 = _mm256_loadu_si256((__m256i *)(ycp + 16)); // 384, 256
                y3 = _mm256_loadu_si256((__m256i *)(ycp + 32)); // 640, 512

                gather_y_uv_from_yc48(y1, y2, y3);
                y0 = y2;

                _mm256_storeu_si256((__m256i *)(Y + x), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                y1 = _mm256_loadu_si256((__m256i *)(ycpw +  0));
                y2 = _mm256_loadu_si256((__m256i *)(ycpw + 16));
                y3 = _mm256_loadu_si256((__m256i *)(ycpw + 32));

                gather_y_uv_from_yc48(y1, y2, y3);

                _mm256_storeu_si256((__m256i *)(Y + x + dst_y_pitch*2), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                _mm256_storeu_si256((__m256i *)(C + x), convert_uv_range_from_yc48_420i(y0, y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_420I(i), yC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, yC_YCC, yC_pw_one));
            }
        }
    }
    _mm256_zeroupper();
}

void convert_yc48_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    char *Y_line = (char *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    char *U_line = (char *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    char *V_line = (char *)dst[2] + dst_y_pitch_byte * y_range.start_dst;
    char *pixel = (char *)src[0] + src_y_pitch_byte * y_range.start_src;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y1, y2, y3;
    for (int y = 0; y < y_range.len; y++, pixel += src_y_pitch_byte, Y_line += dst_y_pitch_byte, U_line += dst_y_pitch_byte, V_line += dst_y_pitch_byte) {
        short *Y = (short *)Y_line;
        short *U = (short *)U_line;
        short *V = (short *)V_line;
        short *const ycp_fin = (short *)pixel + width * 3;
        for (short *ycp = (short *)pixel; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
            y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
            y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
            y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

            gather_y_u_v_from_yc48(y1, y2, y3);

            _mm256_storeu_si256((__m256i *)Y, convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));
            _mm256_storeu_si256((__m256i *)U, convert_uv_range_from_yc48(y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one));
            _mm256_storeu_si256((__m256i *)V, convert_uv_range_from_yc48(y3, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one));
        }
    }
    _mm256_zeroupper();
}

void convert_yuv444_16bit_to_yc48_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const auto y_range = thread_y_range(0, height, thread_id, thread_n);
    char *Y_line = (char *)src[0] + src_y_pitch_byte * y_range.start_src;
    char *U_line = (char *)src[1] + src_y_pitch_byte * y_range.start_src;
    char *V_line = (char *)src[2] + src_y_pitch_byte * y_range.start_src;
    char *pixel = (char *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    __m256i y1, y2, y3;
    for (int y = 0; y < y_range.len; y++, pixel += dst_y_pitch_byte, Y_line += src_y_pitch_byte, U_line += src_y_pitch_byte, V_line += src_y_pitch_byte) {
        short *Y = (short *)Y_line;
        short *U = (short *)U_line;
        short *V = (short *)V_line;
        short *const ycp_fin = (short *)pixel + width * 3;
        for (short *ycp = (short *)pixel; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
            y1 = _mm256_loadu_si256((__m256i *)(Y));
            y2 = _mm256_loadu_si256((__m256i *)(U));
            y3 = _mm256_loadu_si256((__m256i *)(V));
            y1 = convert_y_range_to_yc48(y1);
            y2 = convert_uv_range_to_yc48(y2);
            y3 = convert_uv_range_to_yc48(y3);
            gather_y_u_v_to_yc48(y1, y2, y3);
            _mm256_storeu_si256((__m256i *)(ycp +  0), y1);
            _mm256_storeu_si256((__m256i *)(ycp + 16), y2);
            _mm256_storeu_si256((__m256i *)(ycp + 32), y3);
        }
    }
}

static __forceinline void separate_8bit_packed(__m256i& yA, __m256i& yB, __m256i& yC, const __m256i& y0, const __m256i& y1, const __m256i& y2) {
    alignas(32) static const unsigned char mask_select[] = {
        0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu,
        0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu, 0x00u, 0x00u, 0xffu
    };
    alignas(32) static const char mask_shuffle0[] = {
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13
    };
    alignas(32) static const char mask_shuffle1[] = {
        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14
    };
    alignas(32) static const char mask_shuffle2[] = {
        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
    };
    static_assert(sizeof(mask_select) == 32);
    static_assert(sizeof(mask_shuffle0) == 32);
    static_assert(sizeof(mask_shuffle1) == 32);
    static_assert(sizeof(mask_shuffle2) == 32);

    yA = _mm256_blend_epi32(y0, y1, 0xF0);
    yB = _mm256_permute2x128_si256(y0, y2, (2 << 4) | 1);
    yC = _mm256_blend_epi32(y1, y2, 0xF0);

    __m256i mask0_256 = _mm256_load_si256((__m256i*)mask_select);
    __m256i mask1_256 = _mm256_slli_si256(mask0_256, 1);
    __m256i mask2_256 = _mm256_slli_si256(mask0_256, 2);

    __m256i a32_0 = _mm256_and_si256(yA, mask0_256);
    __m256i a32_1 = _mm256_and_si256(yB, mask2_256);
    __m256i a32_2 = _mm256_and_si256(yC, mask1_256);

    __m256i b32_0 = _mm256_and_si256(yA, mask1_256);
    __m256i b32_1 = _mm256_and_si256(yB, mask0_256);
    __m256i b32_2 = _mm256_and_si256(yC, mask2_256);

    __m256i c32_0 = _mm256_and_si256(yA, mask2_256);
    __m256i c32_1 = _mm256_and_si256(yB, mask1_256);
    __m256i c32_2 = _mm256_and_si256(yC, mask0_256);

    __m256i a32_012 = _mm256_or_si256(a32_0, _mm256_or_si256(a32_1, a32_2));
    __m256i b32_012 = _mm256_or_si256(b32_0, _mm256_or_si256(b32_1, b32_2));
    __m256i c32_012 = _mm256_or_si256(c32_0, _mm256_or_si256(c32_1, c32_2));

    yA = _mm256_shuffle_epi8(a32_012, _mm256_load_si256((__m256i*)mask_shuffle0));
    yB = _mm256_shuffle_epi8(b32_012, _mm256_load_si256((__m256i*)mask_shuffle1));
    yC = _mm256_shuffle_epi8(c32_012, _mm256_load_si256((__m256i*)mask_shuffle2));
}

template<int bit_depth>
static __forceinline void convert_rgb2yuv(__m256& y_f1, __m256& u_f1, __m256& v_f1, 
    const __m256& r_f1, const __m256& g_f1, const __m256& b_f1,
    const __m256& coeff_ry, const __m256& coeff_gy, const __m256& coeff_by,
    const __m256& coeff_ru, const __m256& coeff_gu, const __m256& coeff_bu,
    const __m256& coeff_rv, const __m256& coeff_gv, const __m256& coeff_bv) {
    const __m256 offset_y = _mm256_set1_ps(16.0f * (1 << (bit_depth - 8)));
    const __m256 offset_uv = _mm256_set1_ps(128.0f * (1 << (bit_depth - 8)));
    y_f1 = _mm256_fmadd_ps(coeff_ry, r_f1, 
           _mm256_fmadd_ps(coeff_gy, g_f1, 
           _mm256_fmadd_ps(coeff_by, b_f1, offset_y)));
    u_f1 = _mm256_fmadd_ps(coeff_ru, r_f1,
           _mm256_fmadd_ps(coeff_gu, g_f1,
           _mm256_fmadd_ps(coeff_bu, b_f1, offset_uv)));
    v_f1 = _mm256_fmadd_ps(coeff_rv, r_f1,
           _mm256_fmadd_ps(coeff_gv, g_f1,
           _mm256_fmadd_ps(coeff_bv, b_f1, offset_uv)));
}

void convert_bgr24r_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 8;
    const float *coeff_table = COEFF_RGB2YUV[1];

    // AVX2用の係数をロード
    const __m256 coeff_ry = _mm256_set1_ps(coeff_table[0]);
    const __m256 coeff_gy = _mm256_set1_ps(coeff_table[1]);
    const __m256 coeff_by = _mm256_set1_ps(coeff_table[2]);
    const __m256 coeff_ru = _mm256_set1_ps(coeff_table[3]);
    const __m256 coeff_gu = _mm256_set1_ps(coeff_table[4]);
    const __m256 coeff_bu = _mm256_set1_ps(coeff_table[5]);
    const __m256 coeff_rv = _mm256_set1_ps(coeff_table[6]);
    const __m256 coeff_gv = _mm256_set1_ps(coeff_table[7]);
    const __m256 coeff_bv = _mm256_set1_ps(coeff_table[8]);
    
    const __m256 round_offset = _mm256_set1_ps(0.5f);

    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1) + crop_left * 3;
    uint8_t *dstYLine = (uint8_t *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    uint8_t *dstULine = (uint8_t *)dst[1] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    uint8_t *dstVLine = (uint8_t *)dst[2] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    for (int y = 0; y < y_range.len; y++, dstYLine += dst_y_pitch_byte, dstULine += dst_y_pitch_byte,dstVLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        uint8_t *ptr_src = srcLine;
        uint8_t *dstY = dstYLine;
        uint8_t *dstU = dstULine;
        uint8_t *dstV = dstVLine;
        int x = 0, x_fin = width - crop_left - crop_right;
        // AVX2で32ピクセルずつ処理
        for (; x <= x_fin - 32; x += 32) {
            // 32ピクセル分のBGRデータをロード (96バイト)
            __m256i bgr0 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3));      // 32バイト
            __m256i bgr1 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3 + 32)); // 32バイト
            __m256i bgr2 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3 + 64)); // 32バイト

            __m256i b_8, g_8, r_8;
            separate_8bit_packed(b_8, g_8, r_8, bgr0, bgr1, bgr2);

            __m256i b_16_0 = _mm256_unpacklo_epi8(b_8, _mm256_setzero_si256());
            __m256i b_16_1 = _mm256_unpackhi_epi8(b_8, _mm256_setzero_si256());
            __m256i g_16_0 = _mm256_unpacklo_epi8(g_8, _mm256_setzero_si256());
            __m256i g_16_1 = _mm256_unpackhi_epi8(g_8, _mm256_setzero_si256());
            __m256i r_16_0 = _mm256_unpacklo_epi8(r_8, _mm256_setzero_si256());
            __m256i r_16_1 = _mm256_unpackhi_epi8(r_8, _mm256_setzero_si256());

            __m256i b_32_0 = _mm256_unpacklo_epi16(b_16_0, _mm256_setzero_si256());
            __m256i b_32_1 = _mm256_unpackhi_epi16(b_16_0, _mm256_setzero_si256());
            __m256i b_32_2 = _mm256_unpacklo_epi16(b_16_1, _mm256_setzero_si256());
            __m256i b_32_3 = _mm256_unpackhi_epi16(b_16_1, _mm256_setzero_si256());

            __m256i g_32_0 = _mm256_unpacklo_epi16(g_16_0, _mm256_setzero_si256());
            __m256i g_32_1 = _mm256_unpackhi_epi16(g_16_0, _mm256_setzero_si256());
            __m256i g_32_2 = _mm256_unpacklo_epi16(g_16_1, _mm256_setzero_si256());
            __m256i g_32_3 = _mm256_unpackhi_epi16(g_16_1, _mm256_setzero_si256());
            
            __m256i r_32_0 = _mm256_unpacklo_epi16(r_16_0, _mm256_setzero_si256());
            __m256i r_32_1 = _mm256_unpackhi_epi16(r_16_0, _mm256_setzero_si256());
            __m256i r_32_2 = _mm256_unpacklo_epi16(r_16_1, _mm256_setzero_si256());
            __m256i r_32_3 = _mm256_unpackhi_epi16(r_16_1, _mm256_setzero_si256());
            
            // グループ1: ピクセル0-7 (下位128ビット)
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);

            __m256 y_f0, u_f0, v_f0;
            convert_rgb2yuv<8>(y_f0, u_f0, v_f0, r_f0, g_f0, b_f0, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ2: ピクセル8-15 (上位128ビット)
            __m256 b_f1 = _mm256_cvtepi32_ps(b_32_1);
            __m256 g_f1 = _mm256_cvtepi32_ps(g_32_1);
            __m256 r_f1 = _mm256_cvtepi32_ps(r_32_1);

            __m256 y_f1, u_f1, v_f1;
            convert_rgb2yuv<8>(y_f1, u_f1, v_f1, r_f1, g_f1, b_f1, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ3: ピクセル16-23（上位128ビットから
            __m256 b_f2 = _mm256_cvtepi32_ps(b_32_2);
            __m256 g_f2 = _mm256_cvtepi32_ps(g_32_2);
            __m256 r_f2 = _mm256_cvtepi32_ps(r_32_2);

            __m256 y_f2, u_f2, v_f2;
            convert_rgb2yuv<8>(y_f2, u_f2, v_f2, r_f2, g_f2, b_f2, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ4: ピクセル24-31
            __m256 b_f3 = _mm256_cvtepi32_ps(b_32_3);
            __m256 g_f3 = _mm256_cvtepi32_ps(g_32_3);
            __m256 r_f3 = _mm256_cvtepi32_ps(r_32_3);

            __m256 y_f3, u_f3, v_f3;
            convert_rgb2yuv<8>(y_f3, u_f3, v_f3, r_f3, g_f3, b_f3, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // 四捨五入して整数に変換
            __m256i y_i0 = _mm256_cvttps_epi32(_mm256_add_ps(y_f0, round_offset));
            __m256i u_i0 = _mm256_cvttps_epi32(_mm256_add_ps(u_f0, round_offset));
            __m256i v_i0 = _mm256_cvttps_epi32(_mm256_add_ps(v_f0, round_offset));
            __m256i y_i1 = _mm256_cvttps_epi32(_mm256_add_ps(y_f1, round_offset));
            __m256i u_i1 = _mm256_cvttps_epi32(_mm256_add_ps(u_f1, round_offset));
            __m256i v_i1 = _mm256_cvttps_epi32(_mm256_add_ps(v_f1, round_offset));
            __m256i y_i2 = _mm256_cvttps_epi32(_mm256_add_ps(y_f2, round_offset));
            __m256i u_i2 = _mm256_cvttps_epi32(_mm256_add_ps(u_f2, round_offset));
            __m256i v_i2 = _mm256_cvttps_epi32(_mm256_add_ps(v_f2, round_offset));
            __m256i y_i3 = _mm256_cvttps_epi32(_mm256_add_ps(y_f3, round_offset));
            __m256i u_i3 = _mm256_cvttps_epi32(_mm256_add_ps(u_f3, round_offset));
            __m256i v_i3 = _mm256_cvttps_epi32(_mm256_add_ps(v_f3, round_offset));

            // 32bit -> 16bit変換
            __m256i y_16_0 = _mm256_packus_epi32(y_i0, y_i1);  // 0-15
            __m256i y_16_1 = _mm256_packus_epi32(y_i2, y_i3);  // 16-31
            __m256i u_16_0 = _mm256_packus_epi32(u_i0, u_i1);
            __m256i u_16_1 = _mm256_packus_epi32(u_i2, u_i3);
            __m256i v_16_0 = _mm256_packus_epi32(v_i0, v_i1);
            __m256i v_16_1 = _mm256_packus_epi32(v_i2, v_i3);
            
            // 16bit -> 8bit変換
            __m256i y_8 = _mm256_packus_epi16(y_16_0, y_16_1);  // 32ピクセル
            __m256i u_8 = _mm256_packus_epi16(u_16_0, u_16_1);
            __m256i v_8 = _mm256_packus_epi16(v_16_0, v_16_1);
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(dstY + x), y_8);
            _mm256_storeu_si256((__m256i*)(dstU + x), u_8);
            _mm256_storeu_si256((__m256i*)(dstV + x), v_8);
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            const float b = (float)ptr_src[x*3 + 0];
            const float g = (float)ptr_src[x*3 + 1];
            const float r = (float)ptr_src[x*3 + 2];
            const float py = (coeff_table[0] * r + coeff_table[1] * g + coeff_table[2] * b +  16.0f) * (1 << (out_bit_depth - 8));
            const float pu = (coeff_table[3] * r + coeff_table[4] * g + coeff_table[5] * b + 128.0f) * (1 << (out_bit_depth - 8));
            const float pv = (coeff_table[6] * r + coeff_table[7] * g + coeff_table[8] * b + 128.0f) * (1 << (out_bit_depth - 8));
            dstY[x] = (uint8_t)std::min(std::max((int)(py + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstU[x] = (uint8_t)std::min(std::max((int)(pu + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstV[x] = (uint8_t)std::min(std::max((int)(pv + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
    
    _mm256_zeroupper();
}

void convert_bgr24r_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 16;
    const float *coeff_table = COEFF_RGB2YUV[1];
    
    // AVX2用の係数をロード
    const __m256 coeff_ry = _mm256_set1_ps(coeff_table[0]);
    const __m256 coeff_gy = _mm256_set1_ps(coeff_table[1]);
    const __m256 coeff_by = _mm256_set1_ps(coeff_table[2]);
    const __m256 coeff_ru = _mm256_set1_ps(coeff_table[3]);
    const __m256 coeff_gu = _mm256_set1_ps(coeff_table[4]);
    const __m256 coeff_bu = _mm256_set1_ps(coeff_table[5]);
    const __m256 coeff_rv = _mm256_set1_ps(coeff_table[6]);
    const __m256 coeff_gv = _mm256_set1_ps(coeff_table[7]);
    const __m256 coeff_bv = _mm256_set1_ps(coeff_table[8]);
    
    const __m256 offset_y = _mm256_set1_ps(16.0f);
    const __m256 offset_uv = _mm256_set1_ps(128.0f);
    const __m256 round_offset = _mm256_set1_ps(0.5f);
    const __m256 limit_offset = _mm256_set1_ps((float)(1 << (out_bit_depth - 8)));

    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    char *srcLine = (char *)src[0] + src_y_pitch_byte * ((y_range.start_src + y_range.len) - 1) + crop_left * 3;
    char *dstYLine = (char *)dst[0] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    char *dstULine = (char *)dst[1] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    char *dstVLine = (char *)dst[2] + dst_y_pitch_byte * (height - (y_range.start_dst + y_range.len));
    for (int y = 0; y < y_range.len; y++, dstYLine += dst_y_pitch_byte, dstULine += dst_y_pitch_byte, dstVLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        uint8_t *ptr_src = (uint8_t *)srcLine;
        uint16_t *dstY = (uint16_t *)dstYLine;
        uint16_t *dstU = (uint16_t *)dstULine;
        uint16_t *dstV = (uint16_t *)dstVLine;
        int x = 0, x_fin = width - crop_left - crop_right;
        // AVX2で32ピクセルずつ処理
        for (; x <= x_fin - 32; x += 32) {
            // 32ピクセル分のBGRデータをロード (96バイト)
            __m256i bgr0 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3));      // 32バイト
            __m256i bgr1 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3 + 32)); // 32バイト
            __m256i bgr2 = _mm256_loadu_si256((__m256i*)(ptr_src + x*3 + 64)); // 32バイト

            __m256i b_8, g_8, r_8;
            separate_8bit_packed(b_8, g_8, r_8, bgr0, bgr1, bgr2);

            __m256i b_16_0 = _mm256_unpacklo_epi8(b_8, _mm256_setzero_si256());
            __m256i b_16_1 = _mm256_unpackhi_epi8(b_8, _mm256_setzero_si256());
            __m256i g_16_0 = _mm256_unpacklo_epi8(g_8, _mm256_setzero_si256());
            __m256i g_16_1 = _mm256_unpackhi_epi8(g_8, _mm256_setzero_si256());
            __m256i r_16_0 = _mm256_unpacklo_epi8(r_8, _mm256_setzero_si256());
            __m256i r_16_1 = _mm256_unpackhi_epi8(r_8, _mm256_setzero_si256());

            __m256i b_32_0 = _mm256_unpacklo_epi16(b_16_0, _mm256_setzero_si256());
            __m256i b_32_1 = _mm256_unpackhi_epi16(b_16_0, _mm256_setzero_si256());
            __m256i b_32_2 = _mm256_unpacklo_epi16(b_16_1, _mm256_setzero_si256());
            __m256i b_32_3 = _mm256_unpackhi_epi16(b_16_1, _mm256_setzero_si256());

            __m256i g_32_0 = _mm256_unpacklo_epi16(g_16_0, _mm256_setzero_si256());
            __m256i g_32_1 = _mm256_unpackhi_epi16(g_16_0, _mm256_setzero_si256());
            __m256i g_32_2 = _mm256_unpacklo_epi16(g_16_1, _mm256_setzero_si256());
            __m256i g_32_3 = _mm256_unpackhi_epi16(g_16_1, _mm256_setzero_si256());
            
            __m256i r_32_0 = _mm256_unpacklo_epi16(r_16_0, _mm256_setzero_si256());
            __m256i r_32_1 = _mm256_unpackhi_epi16(r_16_0, _mm256_setzero_si256());
            __m256i r_32_2 = _mm256_unpacklo_epi16(r_16_1, _mm256_setzero_si256());
            __m256i r_32_3 = _mm256_unpackhi_epi16(r_16_1, _mm256_setzero_si256());
            
            // グループ1: ピクセル0-7 (下位128ビット)
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);

            __m256 y_f0, u_f0, v_f0;
            convert_rgb2yuv<8>(y_f0, u_f0, v_f0, r_f0, g_f0, b_f0, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ2: ピクセル8-15 (上位128ビット)
            __m256 b_f1 = _mm256_cvtepi32_ps(b_32_1);
            __m256 g_f1 = _mm256_cvtepi32_ps(g_32_1);
            __m256 r_f1 = _mm256_cvtepi32_ps(r_32_1);

            __m256 y_f1, u_f1, v_f1;
            convert_rgb2yuv<8>(y_f1, u_f1, v_f1, r_f1, g_f1, b_f1, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ3: ピクセル16-23（上位128ビットから
            __m256 b_f2 = _mm256_cvtepi32_ps(b_32_2);
            __m256 g_f2 = _mm256_cvtepi32_ps(g_32_2);
            __m256 r_f2 = _mm256_cvtepi32_ps(r_32_2);

            __m256 y_f2, u_f2, v_f2;
            convert_rgb2yuv<8>(y_f2, u_f2, v_f2, r_f2, g_f2, b_f2, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ4: ピクセル24-31
            __m256 b_f3 = _mm256_cvtepi32_ps(b_32_3);
            __m256 g_f3 = _mm256_cvtepi32_ps(g_32_3);
            __m256 r_f3 = _mm256_cvtepi32_ps(r_32_3);

            __m256 y_f3, u_f3, v_f3;
            convert_rgb2yuv<8>(y_f3, u_f3, v_f3, r_f3, g_f3, b_f3, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // 四捨五入して整数に変換
            __m256i y_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f0, limit_offset, round_offset));
            __m256i u_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f0, limit_offset, round_offset));
            __m256i v_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f0, limit_offset, round_offset));
            __m256i y_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f1, limit_offset, round_offset));
            __m256i u_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f1, limit_offset, round_offset));
            __m256i v_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f1, limit_offset, round_offset));
            __m256i y_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f2, limit_offset, round_offset));
            __m256i u_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f2, limit_offset, round_offset));
            __m256i v_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f2, limit_offset, round_offset));
            __m256i y_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f3, limit_offset, round_offset));
            __m256i u_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f3, limit_offset, round_offset));
            __m256i v_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f3, limit_offset, round_offset));

            // 32bit -> 16bit変換
            __m256i y_16_0 = _mm256_packus_epi32(y_i0, y_i1);  // 0-15
            __m256i y_16_1 = _mm256_packus_epi32(y_i2, y_i3);  // 16-31
            __m256i u_16_0 = _mm256_packus_epi32(u_i0, u_i1);
            __m256i u_16_1 = _mm256_packus_epi32(u_i2, u_i3);
            __m256i v_16_0 = _mm256_packus_epi32(v_i0, v_i1);
            __m256i v_16_1 = _mm256_packus_epi32(v_i2, v_i3);
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(dstY + x +  0), _mm256_permute2x128_si256(y_16_0, y_16_1, (2 << 4) | 0));
            _mm256_storeu_si256((__m256i*)(dstU + x +  0), _mm256_permute2x128_si256(u_16_0, u_16_1, (2 << 4) | 0));
            _mm256_storeu_si256((__m256i*)(dstV + x +  0), _mm256_permute2x128_si256(v_16_0, v_16_1, (2 << 4) | 0));
            _mm256_storeu_si256((__m256i*)(dstY + x + 16), _mm256_permute2x128_si256(y_16_0, y_16_1, (3 << 4) | 1));
            _mm256_storeu_si256((__m256i*)(dstU + x + 16), _mm256_permute2x128_si256(u_16_0, u_16_1, (3 << 4) | 1));
            _mm256_storeu_si256((__m256i*)(dstV + x + 16), _mm256_permute2x128_si256(v_16_0, v_16_1, (3 << 4) | 1));
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            const float b = (float)ptr_src[x*3 + 0];
            const float g = (float)ptr_src[x*3 + 1];
            const float r = (float)ptr_src[x*3 + 2];
            const float py = (coeff_table[0] * r + coeff_table[1] * g + coeff_table[2] * b +  16.0f) * (1 << (out_bit_depth - 8));
            const float pu = (coeff_table[3] * r + coeff_table[4] * g + coeff_table[5] * b + 128.0f) * (1 << (out_bit_depth - 8));
            const float pv = (coeff_table[6] * r + coeff_table[7] * g + coeff_table[8] * b + 128.0f) * (1 << (out_bit_depth - 8));
            dstY[x] = (uint16_t)std::min(std::max((int)(py + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstU[x] = (uint16_t)std::min(std::max((int)(pu + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstV[x] = (uint16_t)std::min(std::max((int)(pv + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
    
    _mm256_zeroupper();
}

// 22bit精度 rcp
static RGY_FORCEINLINE __m256 _mm256_rcpnr_fma_ps(__m256 x) {
	__m256 rcp = _mm256_rcp_ps(x); // 11bit精度
	//rcp*(2-rcp*x)
	return _mm256_mul_ps(rcp, _mm256_fnmadd_ps(x, rcp, _mm256_set1_ps(2.0f)));
}

static RGY_FORCEINLINE void unpremultiply_pa64_avx2(__m256& r, __m256& g, __m256& b, const __m256& a) {
    // 乗算付きalphaなrgbをunpremultiplyに変換
    __m256 a_inv = _mm256_mul_ps(_mm256_set1_ps(65535.0f), _mm256_rcpnr_fma_ps(a));
    r = _mm256_mul_ps(r, a_inv);
    g = _mm256_mul_ps(g, a_inv);
    b = _mm256_mul_ps(b, a_inv);
}

void convert_rgba64_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 8;
    const float *coeff_table = COEFF_RGB2YUV[1];
    
    // AVX2用の係数をロード
    const __m256 coeff_ry = _mm256_set1_ps(coeff_table[0]);
    const __m256 coeff_gy = _mm256_set1_ps(coeff_table[1]);
    const __m256 coeff_by = _mm256_set1_ps(coeff_table[2]);
    const __m256 coeff_ru = _mm256_set1_ps(coeff_table[3]);
    const __m256 coeff_gu = _mm256_set1_ps(coeff_table[4]);
    const __m256 coeff_bu = _mm256_set1_ps(coeff_table[5]);
    const __m256 coeff_rv = _mm256_set1_ps(coeff_table[6]);
    const __m256 coeff_gv = _mm256_set1_ps(coeff_table[7]);
    const __m256 coeff_bv = _mm256_set1_ps(coeff_table[8]);
    
    const __m256 offset_y = _mm256_set1_ps(16.0f);
    const __m256 offset_uv = _mm256_set1_ps(128.0f);
    const __m256 round_offset = _mm256_set1_ps(0.5f);
    const __m256 limit_offset = _mm256_set1_ps(1.0f / (float)(1 << (16 - out_bit_depth)));
    
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    char *Y_line = (char *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    char *U_line = (char *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    char *V_line = (char *)dst[2] + dst_y_pitch_byte * y_range.start_dst;
    char *pixel = (char *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * sizeof(uint16_t);
    for (int y = 0; y < y_range.len; y++, pixel += src_y_pitch_byte, Y_line += dst_y_pitch_byte, U_line += dst_y_pitch_byte, V_line += dst_y_pitch_byte) {
        uint8_t *dstY = (uint8_t *)Y_line;
        uint8_t *dstU = (uint8_t *)U_line;
        uint8_t *dstV = (uint8_t *)V_line;
        uint16_t *srcp  = (uint16_t*)pixel;
        
        int x = 0;
        // AVX2で32ピクセルずつ処理
        for (; x <= y_width - 32; x += 32) {
            __m256i y0 = _mm256_loadu_si256((__m256i *)(srcp + x*4 +  0));
            __m256i y1 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 16));
            __m256i y2 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 32));
            __m256i y3 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 48));
            __m256i y4 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 64));
            __m256i y5 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 80));
            __m256i y6 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 96));
            __m256i y7 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 112));

            __m256i y01_0 = _mm256_permute2x128_si256(y0, y1, (2 << 4) | 0);
            __m256i y01_1 = _mm256_permute2x128_si256(y0, y1, (3 << 4) | 1);
            __m256i y23_0 = _mm256_permute2x128_si256(y2, y3, (2 << 4) | 0);
            __m256i y23_1 = _mm256_permute2x128_si256(y2, y3, (3 << 4) | 1);
            __m256i y45_0 = _mm256_permute2x128_si256(y4, y5, (2 << 4) | 0);
            __m256i y45_1 = _mm256_permute2x128_si256(y4, y5, (3 << 4) | 1);
            __m256i y67_0 = _mm256_permute2x128_si256(y6, y7, (2 << 4) | 0);
            __m256i y67_1 = _mm256_permute2x128_si256(y6, y7, (3 << 4) | 1);

            y0 = _mm256_packus_epi32(_mm256_and_si256(y01_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y01_1, _mm256_set1_epi32(0xFFFF))); // 3b	3r	2b	2r	1b	1r	0b	0r
            y1 = _mm256_packus_epi32(_mm256_srli_epi32(y01_0, 16), _mm256_srli_epi32(y01_1, 16)); // 3a	3g	2a	2g	1a	1g	0a	0g
            y2 = _mm256_packus_epi32(_mm256_and_si256(y23_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y23_1, _mm256_set1_epi32(0xFFFF)));
            y3 = _mm256_packus_epi32(_mm256_srli_epi32(y23_0, 16), _mm256_srli_epi32(y23_1, 16));
            y4 = _mm256_packus_epi32(_mm256_and_si256(y45_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y45_1, _mm256_set1_epi32(0xFFFF))); // 3b	3r	2b	2r	1b	1r	0b	0r
            y5 = _mm256_packus_epi32(_mm256_srli_epi32(y45_0, 16), _mm256_srli_epi32(y45_1, 16)); // 3a	3g	2a	2g	1a	1g	0a	0g
            y6 = _mm256_packus_epi32(_mm256_and_si256(y67_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y67_1, _mm256_set1_epi32(0xFFFF)));
            y7 = _mm256_packus_epi32(_mm256_srli_epi32(y67_0, 16), _mm256_srli_epi32(y67_1, 16));

            __m256i r_32_0 = _mm256_and_si256(y0, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_0 = _mm256_srli_epi32(y0, 16);
            __m256i r_32_1 = _mm256_and_si256(y2, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_1 = _mm256_srli_epi32(y2, 16);
            __m256i g_32_0 = _mm256_and_si256(y1, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_0 = _mm256_srli_epi32(y1, 16);
            __m256i g_32_1 = _mm256_and_si256(y3, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_1 = _mm256_srli_epi32(y3, 16);
            __m256i r_32_2 = _mm256_and_si256(y4, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_2 = _mm256_srli_epi32(y4, 16);
            __m256i r_32_3 = _mm256_and_si256(y6, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_3 = _mm256_srli_epi32(y6, 16);
            __m256i g_32_2 = _mm256_and_si256(y5, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_2 = _mm256_srli_epi32(y5, 16);
            __m256i g_32_3 = _mm256_and_si256(y7, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_3 = _mm256_srli_epi32(y7, 16);

            // グループ1: ピクセル0-7
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);
            __m256 a_f0 = _mm256_cvtepi32_ps(a_32_0);
            unpremultiply_pa64_avx2(r_f0, g_f0, b_f0, a_f0);

            __m256 y_f0, u_f0, v_f0;
            convert_rgb2yuv<16>(y_f0, u_f0, v_f0, r_f0, g_f0, b_f0, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ2: ピクセル8-15
            __m256 b_f1 = _mm256_cvtepi32_ps(b_32_1);
            __m256 g_f1 = _mm256_cvtepi32_ps(g_32_1);
            __m256 r_f1 = _mm256_cvtepi32_ps(r_32_1);
            __m256 a_f1 = _mm256_cvtepi32_ps(a_32_1);
            unpremultiply_pa64_avx2(r_f1, g_f1, b_f1, a_f1);

            __m256 y_f1, u_f1, v_f1;
            convert_rgb2yuv<16>(y_f1, u_f1, v_f1, r_f1, g_f1, b_f1, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);

            // グループ3: ピクセル16-23
            __m256 b_f2 = _mm256_cvtepi32_ps(b_32_2);
            __m256 g_f2 = _mm256_cvtepi32_ps(g_32_2);
            __m256 r_f2 = _mm256_cvtepi32_ps(r_32_2);
            __m256 a_f2 = _mm256_cvtepi32_ps(a_32_2);
            unpremultiply_pa64_avx2(r_f2, g_f2, b_f2, a_f2);

            
            __m256 y_f2, u_f2, v_f2;
            convert_rgb2yuv<16>(y_f2, u_f2, v_f2, r_f2, g_f2, b_f2, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);

            // グループ4: ピクセル24-31
            __m256 b_f3 = _mm256_cvtepi32_ps(b_32_3);
            __m256 g_f3 = _mm256_cvtepi32_ps(g_32_3);
            __m256 r_f3 = _mm256_cvtepi32_ps(r_32_3);
            __m256 a_f3 = _mm256_cvtepi32_ps(a_32_3);
            unpremultiply_pa64_avx2(r_f3, g_f3, b_f3, a_f3);

            __m256 y_f3, u_f3, v_f3;    
            convert_rgb2yuv<16>(y_f3, u_f3, v_f3, r_f3, g_f3, b_f3, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);

            // 四捨五入して整数に変換
            __m256i y_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f0, limit_offset, round_offset));
            __m256i u_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f0, limit_offset, round_offset));
            __m256i v_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f0, limit_offset, round_offset));
            __m256i y_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f1, limit_offset, round_offset));
            __m256i u_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f1, limit_offset, round_offset));
            __m256i v_i1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f1, limit_offset, round_offset));
            __m256i y_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f2, limit_offset, round_offset));
            __m256i u_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f2, limit_offset, round_offset));
            __m256i v_i2 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f2, limit_offset, round_offset));
            __m256i y_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(y_f3, limit_offset, round_offset));
            __m256i u_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(u_f3, limit_offset, round_offset));
            __m256i v_i3 = _mm256_cvttps_epi32(_mm256_fmadd_ps(v_f3, limit_offset, round_offset));

            // 32bit -> 16bit変換
            __m256i y_16_0 = _mm256_packus_epi32(y_i0, y_i1);  // 0-15
            __m256i u_16_0 = _mm256_packus_epi32(u_i0, u_i1);
            __m256i v_16_0 = _mm256_packus_epi32(v_i0, v_i1);
            __m256i y_16_1 = _mm256_packus_epi32(y_i2, y_i3);  // 16-31
            __m256i u_16_1 = _mm256_packus_epi32(u_i2, u_i3);
            __m256i v_16_1 = _mm256_packus_epi32(v_i2, v_i3);
            
            // 16bit -> 8bit変換
            __m256i y_8 = _mm256_packus_epi16(y_16_0, y_16_1);  // 32ピクセル
            __m256i u_8 = _mm256_packus_epi16(u_16_0, u_16_1);
            __m256i v_8 = _mm256_packus_epi16(v_16_0, v_16_1);

            y_8 = _mm256_permute4x64_epi64(y_8, _MM_SHUFFLE(3,1,2,0));
            u_8 = _mm256_permute4x64_epi64(u_8, _MM_SHUFFLE(3,1,2,0));
            v_8 = _mm256_permute4x64_epi64(v_8, _MM_SHUFFLE(3,1,2,0));

            y_8 = _mm256_shuffle_epi32(y_8, _MM_SHUFFLE(3,1,2,0));
            u_8 = _mm256_shuffle_epi32(u_8, _MM_SHUFFLE(3,1,2,0));
            v_8 = _mm256_shuffle_epi32(v_8, _MM_SHUFFLE(3,1,2,0));
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(dstY + x), y_8);
            _mm256_storeu_si256((__m256i*)(dstU + x), u_8);
            _mm256_storeu_si256((__m256i*)(dstV + x), v_8);
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            float b = (float)srcp[x*4 + 0];
            float g = (float)srcp[x*4 + 1];
            float r = (float)srcp[x*4 + 2];
            float a = (float)srcp[x*4 + 3];
            float a_inv = 65535.0f / a;
            b *= a_inv, g *= a_inv, r *= a_inv;
            const float py = (coeff_table[0] * r + coeff_table[1] * g + coeff_table[2] * b +  16.0f) * (1 << (out_bit_depth - 8));
            const float pu = (coeff_table[3] * r + coeff_table[4] * g + coeff_table[5] * b + 128.0f) * (1 << (out_bit_depth - 8));
            const float pv = (coeff_table[6] * r + coeff_table[7] * g + coeff_table[8] * b + 128.0f) * (1 << (out_bit_depth - 8));
            dstY[x] = (uint8_t)std::min(std::max((int)(py + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstU[x] = (uint8_t)std::min(std::max((int)(pu + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstV[x] = (uint8_t)std::min(std::max((int)(pv + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
}

void convert_rgba64_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 16;
    const float *coeff_table = COEFF_RGB2YUV[1];
    
    // AVX2用の係数をロード
    const __m256 coeff_ry = _mm256_set1_ps(coeff_table[0]);
    const __m256 coeff_gy = _mm256_set1_ps(coeff_table[1]);
    const __m256 coeff_by = _mm256_set1_ps(coeff_table[2]);
    const __m256 coeff_ru = _mm256_set1_ps(coeff_table[3]);
    const __m256 coeff_gu = _mm256_set1_ps(coeff_table[4]);
    const __m256 coeff_bu = _mm256_set1_ps(coeff_table[5]);
    const __m256 coeff_rv = _mm256_set1_ps(coeff_table[6]);
    const __m256 coeff_gv = _mm256_set1_ps(coeff_table[7]);
    const __m256 coeff_bv = _mm256_set1_ps(coeff_table[8]);
    
    const __m256 offset_y = _mm256_set1_ps(16.0f);
    const __m256 offset_uv = _mm256_set1_ps(128.0f);
    const __m256 round_offset = _mm256_set1_ps(0.5f);
    
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int y_width = width - crop_right - crop_left;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    char *Y_line = (char *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    char *U_line = (char *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    char *V_line = (char *)dst[2] + dst_y_pitch_byte * y_range.start_dst;
    char *pixel = (char *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * sizeof(uint16_t);
    for (int y = 0; y < y_range.len; y++, pixel += src_y_pitch_byte, Y_line += dst_y_pitch_byte, U_line += dst_y_pitch_byte, V_line += dst_y_pitch_byte) {
        uint16_t *dstY = (uint16_t *)Y_line;
        uint16_t *dstU = (uint16_t *)U_line;
        uint16_t *dstV = (uint16_t *)V_line;
        uint16_t *srcp = (uint16_t*)pixel;
        
        int x = 0;
        // AVX2で32ピクセルずつ処理
        for (; x <= y_width - 16; x += 16) {
            __m256i y0 = _mm256_loadu_si256((__m256i *)(srcp + x*4 +  0));
            __m256i y1 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 16));
            __m256i y2 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 32));
            __m256i y3 = _mm256_loadu_si256((__m256i *)(srcp + x*4 + 48));
            
            __m256i y01_0 = _mm256_permute2x128_si256(y0, y1, (2 << 4) | 0);
            __m256i y01_1 = _mm256_permute2x128_si256(y0, y1, (3 << 4) | 1);
            __m256i y23_0 = _mm256_permute2x128_si256(y2, y3, (2 << 4) | 0);
            __m256i y23_1 = _mm256_permute2x128_si256(y2, y3, (3 << 4) | 1);

            y0 = _mm256_packus_epi32(_mm256_and_si256(y01_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y01_1, _mm256_set1_epi32(0xFFFF))); // 3b	3r	2b	2r	1b	1r	0b	0r
            y1 = _mm256_packus_epi32(_mm256_srli_epi32(y01_0, 16), _mm256_srli_epi32(y01_1, 16)); // 3a	3g	2a	2g	1a	1g	0a	0g
            y2 = _mm256_packus_epi32(_mm256_and_si256(y23_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y23_1, _mm256_set1_epi32(0xFFFF)));
            y3 = _mm256_packus_epi32(_mm256_srli_epi32(y23_0, 16), _mm256_srli_epi32(y23_1, 16));
            
            __m256i r_32_0 = _mm256_and_si256(y0, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_0 = _mm256_srli_epi32(y0, 16);
            __m256i g_32_0 = _mm256_and_si256(y1, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_0 = _mm256_srli_epi32(y1, 16);
            __m256i r_32_1 = _mm256_and_si256(y2, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_1 = _mm256_srli_epi32(y2, 16);
            __m256i g_32_1 = _mm256_and_si256(y3, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_1 = _mm256_srli_epi32(y3, 16);
            
            // グループ1: ピクセル0-7
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);
            __m256 a_f0 = _mm256_cvtepi32_ps(a_32_0);
            unpremultiply_pa64_avx2(r_f0, g_f0, b_f0, a_f0);

            __m256 y_f0, u_f0, v_f0;
            convert_rgb2yuv<16>(y_f0, u_f0, v_f0, r_f0, g_f0, b_f0, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // グループ2: ピクセル8-15
            __m256 b_f1 = _mm256_cvtepi32_ps(b_32_1);
            __m256 g_f1 = _mm256_cvtepi32_ps(g_32_1);
            __m256 r_f1 = _mm256_cvtepi32_ps(r_32_1);
            __m256 a_f1 = _mm256_cvtepi32_ps(a_32_1);
            unpremultiply_pa64_avx2(r_f1, g_f1, b_f1, a_f1);

            __m256 y_f1, u_f1, v_f1;
            convert_rgb2yuv<16>(y_f1, u_f1, v_f1, r_f1, g_f1, b_f1, coeff_ry, coeff_gy, coeff_by, coeff_ru, coeff_gu, coeff_bu, coeff_rv, coeff_gv, coeff_bv);
            
            // 四捨五入して整数に変換
            __m256i y_i0 = _mm256_cvttps_epi32(_mm256_add_ps(y_f0, round_offset));
            __m256i u_i0 = _mm256_cvttps_epi32(_mm256_add_ps(u_f0, round_offset));
            __m256i v_i0 = _mm256_cvttps_epi32(_mm256_add_ps(v_f0, round_offset));
            __m256i y_i1 = _mm256_cvttps_epi32(_mm256_add_ps(y_f1, round_offset));
            __m256i u_i1 = _mm256_cvttps_epi32(_mm256_add_ps(u_f1, round_offset));
            __m256i v_i1 = _mm256_cvttps_epi32(_mm256_add_ps(v_f1, round_offset));

            // 32bit -> 16bit変換
            __m256i y_16_0 = _mm256_packus_epi32(y_i0, y_i1);  // 0-15
            __m256i u_16_0 = _mm256_packus_epi32(u_i0, u_i1);
            __m256i v_16_0 = _mm256_packus_epi32(v_i0, v_i1);
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(dstY + x +  0), _mm256_permute4x64_epi64(y_16_0, _MM_SHUFFLE(3,1,2,0)));
            _mm256_storeu_si256((__m256i*)(dstU + x +  0), _mm256_permute4x64_epi64(u_16_0, _MM_SHUFFLE(3,1,2,0)));
            _mm256_storeu_si256((__m256i*)(dstV + x +  0), _mm256_permute4x64_epi64(v_16_0, _MM_SHUFFLE(3,1,2,0)));
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            float b = (float)srcp[x*4 + 0];
            float g = (float)srcp[x*4 + 1];
            float r = (float)srcp[x*4 + 2];
            float a = (float)srcp[x*4 + 3];
            float a_inv = 65535.0f / a;
            b *= a_inv, g *= a_inv, r *= a_inv;
            const float py = (coeff_table[0] * r + coeff_table[1] * g + coeff_table[2] * b +  16.0f) * (1 << (out_bit_depth - 8));
            const float pu = (coeff_table[3] * r + coeff_table[4] * g + coeff_table[5] * b + 128.0f) * (1 << (out_bit_depth - 8));
            const float pv = (coeff_table[6] * r + coeff_table[7] * g + coeff_table[8] * b + 128.0f) * (1 << (out_bit_depth - 8));
            dstY[x] = (uint16_t)std::min(std::max((int)(py + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstU[x] = (uint16_t)std::min(std::max((int)(pu + 0.5f), 0), (1 << out_bit_depth) - 1);
            dstV[x] = (uint16_t)std::min(std::max((int)(pv + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
}

void convert_rgba64_to_rgba_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 8;
    uint8_t *ptrDst = (uint8_t *)dst[0];
    uint16_t *ptrSrc = (uint16_t *)src[0];
    const float mul = 1.0f /(float)(1 << (16 - out_bit_depth));
    const __m256 round_offset = _mm256_set1_ps(0.5f);
    const __m256 mul256 = _mm256_set1_ps(mul);
    
    for (int y = 0; y < height; y++) {
        uint8_t  *ptr_dst = (uint8_t *)((char*)ptrDst + y*dst_y_pitch_byte);
        uint16_t *ptr_src = (uint16_t *)((char*)ptrSrc + y*src_y_pitch_byte);
        
        int x = 0;
        // AVX2で32ピクセルずつ処理
        for (; x <= width - 8; x += 8) {
            __m256i y0 = _mm256_loadu_si256((__m256i *)(ptr_src + x*4 +  0));
            __m256i y1 = _mm256_loadu_si256((__m256i *)(ptr_src + x*4 + 16));
            
            __m256i y01_0 = _mm256_permute2x128_si256(y0, y1, (2 << 4) | 0);
            __m256i y01_1 = _mm256_permute2x128_si256(y0, y1, (3 << 4) | 1);

            y0 = _mm256_packus_epi32(_mm256_and_si256(y01_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y01_1, _mm256_set1_epi32(0xFFFF))); // 3b	3r	2b	2r	1b	1r	0b	0r
            y1 = _mm256_packus_epi32(_mm256_srli_epi32(y01_0, 16), _mm256_srli_epi32(y01_1, 16)); // 3a	3g	2a	2g	1a	1g	0a	0g
            
            __m256i r_32_0 = _mm256_and_si256(y0, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_0 = _mm256_srli_epi32(y0, 16);
            __m256i g_32_0 = _mm256_and_si256(y1, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_0 = _mm256_srli_epi32(y1, 16);
            
            // グループ1: ピクセル0-7
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);
            __m256 a_f0 = _mm256_cvtepi32_ps(a_32_0);
            unpremultiply_pa64_avx2(r_f0, g_f0, b_f0, a_f0);

            // 四捨五入して整数に変換
            __m256i r_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(r_f0, mul256, round_offset));
            __m256i g_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(g_f0, mul256, round_offset));
            __m256i b_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(b_f0, mul256, round_offset));
            __m256i a_i0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(a_f0, mul256, round_offset));

            r_i0 = _mm256_min_epi32(_mm256_max_epi32(r_i0, _mm256_setzero_si256()), _mm256_set1_epi32(255));
            g_i0 = _mm256_min_epi32(_mm256_max_epi32(g_i0, _mm256_setzero_si256()), _mm256_set1_epi32(255));
            b_i0 = _mm256_min_epi32(_mm256_max_epi32(b_i0, _mm256_setzero_si256()), _mm256_set1_epi32(255));
            a_i0 = _mm256_min_epi32(_mm256_max_epi32(a_i0, _mm256_setzero_si256()), _mm256_set1_epi32(255));

            y0 = _mm256_or_si256(b_i0, _mm256_slli_epi32(g_i0, 8));
            y1 = _mm256_or_si256(r_i0, _mm256_slli_epi32(a_i0, 8));
            y0 = _mm256_or_si256(y0, _mm256_slli_epi32(y1, 16));
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(ptr_dst + x*4), y0);
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            float b = (float)ptr_src[x*4 + 0];
            float g = (float)ptr_src[x*4 + 1];
            float r = (float)ptr_src[x*4 + 2];
            float a = (float)ptr_src[x*4 + 3];
            float a_inv = 65535.0f * mul / a;
            b *= a_inv, g *= a_inv, r *= a_inv, a *= mul;
            ptr_dst[x*4 + 0] = (uint8_t)std::min(std::max((int)(b + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 1] = (uint8_t)std::min(std::max((int)(g + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 2] = (uint8_t)std::min(std::max((int)(r + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 3] = (uint8_t)std::min(std::max((int)(a + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
}

void convert_rgba64_to_rgba_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int out_bit_depth = 16;
    uint16_t *ptrDst = (uint16_t *)dst[0];
    uint16_t *ptrSrc = (uint16_t *)src[0];
    const __m256 round_offset = _mm256_set1_ps(0.5f);
    
    for (int y = 0; y < height; y++) {
        uint16_t *ptr_dst = (uint16_t *)((char*)ptrDst + y*dst_y_pitch_byte);
        uint16_t *ptr_src = (uint16_t *)((char*)ptrSrc + y*src_y_pitch_byte);
        
        int x = 0;
        // AVX2で32ピクセルずつ処理
        for (; x <= width - 8; x += 8) {
            __m256i y0 = _mm256_loadu_si256((__m256i *)(ptr_src + x*4 +  0));
            __m256i y1 = _mm256_loadu_si256((__m256i *)(ptr_src + x*4 + 16));
            
            __m256i y01_0 = _mm256_permute2x128_si256(y0, y1, (2 << 4) | 0);
            __m256i y01_1 = _mm256_permute2x128_si256(y0, y1, (3 << 4) | 1);

            y0 = _mm256_packus_epi32(_mm256_and_si256(y01_0, _mm256_set1_epi32(0xFFFF)), _mm256_and_si256(y01_1, _mm256_set1_epi32(0xFFFF))); // 3b	3r	2b	2r	1b	1r	0b	0r
            y1 = _mm256_packus_epi32(_mm256_srli_epi32(y01_0, 16), _mm256_srli_epi32(y01_1, 16)); // 3a	3g	2a	2g	1a	1g	0a	0g
            
            __m256i r_32_0 = _mm256_and_si256(y0, _mm256_set1_epi32(0xFFFF));
            __m256i b_32_0 = _mm256_srli_epi32(y0, 16);
            __m256i g_32_0 = _mm256_and_si256(y1, _mm256_set1_epi32(0xFFFF));
            __m256i a_32_0 = _mm256_srli_epi32(y1, 16);
            
            // グループ1: ピクセル0-7
            __m256 b_f0 = _mm256_cvtepi32_ps(b_32_0);
            __m256 g_f0 = _mm256_cvtepi32_ps(g_32_0);
            __m256 r_f0 = _mm256_cvtepi32_ps(r_32_0);
            __m256 a_f0 = _mm256_cvtepi32_ps(a_32_0);
            unpremultiply_pa64_avx2(r_f0, g_f0, b_f0, a_f0);

            // 四捨五入して整数に変換
            __m256i r_i0 = _mm256_cvttps_epi32(_mm256_add_ps(r_f0, round_offset));
            __m256i g_i0 = _mm256_cvttps_epi32(_mm256_add_ps(g_f0, round_offset));
            __m256i b_i0 = _mm256_cvttps_epi32(_mm256_add_ps(b_f0, round_offset));
            __m256i a_i0 = _mm256_cvttps_epi32(_mm256_add_ps(a_f0, round_offset));

            // 32bit -> 16bit変換
            y0 = _mm256_packus_epi32(b_i0, r_i0);  // 0-15
            y1 = _mm256_packus_epi32(g_i0, a_i0);

            __m256i y01 = _mm256_blend_epi16(y0, _mm256_slli_epi64(y1, 32), 0xCC);
            __m256i y23 = _mm256_blend_epi16(_mm256_srli_epi64(y0, 32), y1, 0xCC);
            
            alignas(32) static const char Array_Y_16_TO_8[32] = {
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
            };

            y01 = _mm256_shuffle_epi8(y01, _mm256_load_si256((__m256i *)Array_Y_16_TO_8));
            y23 = _mm256_shuffle_epi8(y23, _mm256_load_si256((__m256i *)Array_Y_16_TO_8));
            
            y0 = _mm256_permute2x128_si256(y01, y23, (2 << 4) | 0);
            y1 = _mm256_permute2x128_si256(y01, y23, (3 << 4) | 1);
            
            // 結果を256ビットレジスタで格納（32ピクセル = 32バイト）
            _mm256_storeu_si256((__m256i*)(ptr_dst + x*4 +  0), y0);
            _mm256_storeu_si256((__m256i*)(ptr_dst + x*4 + 16), y1);
        }
        
        // 残りのピクセルを従来の方法で処理
        for (; x < width; x++) {
            float b = (float)ptr_src[x*4 + 0];
            float g = (float)ptr_src[x*4 + 1];
            float r = (float)ptr_src[x*4 + 2];
            float a = (float)ptr_src[x*4 + 3];
            float a_inv = 65535.0f / a;
            b *= a_inv, g *= a_inv, r *= a_inv;
            ptr_dst[x*4 + 0] = (uint16_t)std::min(std::max((int)(b + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 1] = (uint16_t)std::min(std::max((int)(g + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 2] = (uint16_t)std::min(std::max((int)(r + 0.5f), 0), (1 << out_bit_depth) - 1);
            ptr_dst[x*4 + 3] = (uint16_t)std::min(std::max((int)(a + 0.5f), 0), (1 << out_bit_depth) - 1);
        }
    }
}


#pragma warning(pop)
#endif //#if defined(_MSC_VER) || defined(__AVX2__)
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
