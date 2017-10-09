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

#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1

#include "rgy_simd.h"
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "convert_csp.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

#if defined(_MSC_VER) || defined(__AVX2__)

template<bool use_stream>
static void __forceinline avx2_memcpy(uint8_t *dst, const uint8_t *src, int size) {
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
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256
//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))

alignas(32) static const uint8_t  Array_INTERLACE_WEIGHT[2][32] = {
    {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
    {3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define yC_INTERLACE_WEIGHT(i) _mm256_load_si256((__m256i*)Array_INTERLACE_WEIGHT[i])

static __forceinline void separate_low_up(__m256i& x0_return_lower, __m256i& x1_return_upper) {
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
void convert_yuy2_to_nv12_avx2(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const void *src = src_array[0];
    uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * crop_up + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst_array[0];
    uint8_t *dstCLine = (uint8_t *)dst_array[1];
    const int y_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < y_fin; y += 2) {
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

static __forceinline __m256i yuv422_to_420_i_interpolate(__m256i y_up, __m256i y_down, int i) {
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
void convert_yuy2_to_nv12_i_avx2(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const void *src = src_array[0];
    uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * crop_up + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst_array[0];
    uint8_t *dstCLine = (uint8_t *)dst_array[1];
    const int y_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < y_fin; y += 4) {
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
static void __forceinline convert_yv12_to_nv12_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            avx2_memcpy<false>(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *srcVLine = (uint8_t *)src[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *dstLine = (uint8_t *)dst[1];
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
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

void convert_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_to_nv12_avx2_base<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_uv_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_to_nv12_avx2_base<true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}



#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4100)
void convert_rgb24_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *dstLine = (uint8_t *)dst[0];
    alignas(32) const char MASK_RGB3_TO_RGB4[] ={
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1
    };
    __m256i yMask = _mm256_load_si256((__m256i*)MASK_RGB3_TO_RGB4);
    for (int y = crop_up; y < height - crop_bottom; y++, dstLine += dst_y_pitch_byte) {
        uint8_t *ptr_src = (uint8_t *)src[0] + (src_y_pitch_byte * y) + crop_left * 3;
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

void convert_rgb24r_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *dstLine = (uint8_t *)dst[0];
    alignas(32) const char MASK_RGB3_TO_RGB4[] = {
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1
    };
    __m256i yMask = _mm256_load_si256((__m256i*)MASK_RGB3_TO_RGB4);
    for (int y = height - crop_up - 1; y >= crop_bottom; y--, dstLine += dst_y_pitch_byte) {
        uint8_t *ptr_src = (uint8_t *)src[0] + (src_y_pitch_byte * y) + crop_left * 3;
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

void convert_rgb32_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0];
    const int y_fin = height - crop_bottom - crop_up;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_fin; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 4);
    }
    _mm256_zeroupper();
}

void convert_rgb32r_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * (height - crop_up - 1) + crop_left * 4;
    uint8_t *dstLine = (uint8_t *)dst[0];
    const int y_fin = height - crop_bottom - crop_up;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_fin; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 4);
    }
    _mm256_zeroupper();
}

void convert_rgb24_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0];
    const int y_fin = height - crop_bottom - crop_up;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_fin; y++, dstLine += dst_y_pitch_byte, srcLine += src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 3);
    }
    _mm256_zeroupper();
}

void convert_rgb24r_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * (height - crop_up - 1) + crop_left * 3;
    uint8_t *dstLine = (uint8_t *)dst[0];
    const int y_fin = height - crop_bottom - crop_up;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_fin; y++, dstLine += dst_y_pitch_byte, srcLine -= src_y_pitch_byte) {
        avx2_memcpy<false>(dstLine, srcLine, y_width * 3);
    }
    _mm256_zeroupper();
}

template<bool uv_only>
static void convert_yv12_to_p010_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
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
    uint8_t *srcULine = (uint8_t *)src[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *srcVLine = (uint8_t *)src[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *dstLine  = (uint8_t *)dst[1];
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
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

void convert_yv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_to_p010_avx2_base<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (push)
#pragma warning (disable: 4127)
#pragma warning (disable: 4100)
template<int in_bit_depth, bool uv_only>
static void convert_yv12_high_to_nv12_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    //Y成分のコピー
    if (!uv_only) {
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * crop_up + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint8_t *dst_ptr = dstLine;
            uint16_t *src_ptr = srcYLine;
            uint16_t *src_ptr_fin = src_ptr + y_width;
            __m256i y0, y1;
            for (; src_ptr < src_ptr_fin; dst_ptr += 32, src_ptr += 32) {
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 16)), _mm_loadu_si128((__m128i*)(src_ptr +  0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(src_ptr + 24)), _mm_loadu_si128((__m128i*)(src_ptr +  8)));

                y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
                y1 = _mm256_srli_epi16(y1, in_bit_depth - 8);

                y0 = _mm256_packus_epi16(y0, y1);

                _mm256_storeu_si256((__m256i *)(dst_ptr + 0), y0);
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + (((src_uv_pitch * crop_up) + crop_left) >> 1);
    uint16_t *srcVLine = (uint16_t *)src[2] + (((src_uv_pitch * crop_up) + crop_left) >> 1);
    uint8_t *dstLine  = (uint8_t *)dst[1];
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        uint8_t *dst_ptr_fin = dst_ptr + x_fin;
        __m256i y0, y1;
        for (; dst_ptr < dst_ptr_fin; src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
            y0 = _mm256_loadu_si256((const __m256i *)src_u_ptr);
            y1 = _mm256_loadu_si256((const __m256i *)src_v_ptr);

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

void convert_yv12_16_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_14_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<14, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<12, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_10_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<10, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_09_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_nv12_avx2_base<9, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<int in_bit_depth, bool uv_only>
static void __forceinline convert_yv12_high_to_p010_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    if (!uv_only) {
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * crop_up + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                avx2_memcpy<true>((uint8_t *)dstLine, (uint8_t *)srcYLine, y_width * (int)sizeof(uint16_t));
            } else {
                uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 8, dst_ptr += 8, src_ptr += 8) {
                    __m256i y0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                    y0 = _mm256_slli_epi16(y0, 16 - in_bit_depth);
                    _mm256_storeu_si256((__m256i *)dst_ptr, y0);
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + (((src_uv_pitch * crop_up) + crop_left) >> 1);
    uint16_t *srcVLine = (uint16_t *)src[2] + (((src_uv_pitch * crop_up) + crop_left) >> 1);
    uint16_t *dstLine = (uint16_t *)dst[1];
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = width - crop_right;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        __m256i y0, y1, y2;
        for (int x = crop_left; x < x_fin; x += 16, src_u_ptr += 8, src_v_ptr += 8, dst_ptr += 16) {
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

            _mm256_storeu_si256((__m256i *)(dst_ptr + 0), y0);
            _mm256_storeu_si256((__m256i *)(dst_ptr + 8), y2);
        }
    }
}
#pragma warning (pop)

void convert_yv12_16_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_p010_avx2_base<16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_14_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_p010_avx2_base<14, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_p010_avx2_base<12, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_10_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_p010_avx2_base<10, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_09_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_high_to_p010_avx2_base<9, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void copy_yuv444_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    for (int i = 0; i < 3; i++) {
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            avx2_memcpy<true>(dstLine, srcYLine, y_width);
        }
    }
}

template<int in_bit_depth>
static void __forceinline convert_yuv444_high_to_yuv444_16_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    for (int i = 0; i < 3; i++) {
        const uint16_t *srcYLine = (const uint16_t *)src[i] + src_y_pitch * crop_up + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                avx2_memcpy<true>((uint8_t *)dstLine, (const uint8_t *)srcYLine, y_width * (int)sizeof(uint16_t));
            } else {
                const uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                    __m256i x0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                    x0 = _mm256_slli_epi16(x0, 16 - in_bit_depth);
                    _mm256_storeu_si256((__m256i *)dst_ptr, x0);
                }
            }
        }
    }
}
#pragma warning(pop)

void convert_yuv444_16_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_16_avx2_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_14_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_16_avx2_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_12_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_16_avx2_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_10_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_16_avx2_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_09_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_16_avx2_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void convert_yuv444_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    for (int i = 0; i < 3; i++) {
        uint8_t *srcYLine = (uint8_t *)src[i] + src_y_pitch_byte * crop_up + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch) {
            uint8_t *src_ptr = srcYLine;
            uint16_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0, y1;
                y0 = _mm256_loadu_si256((const __m256i *)src_ptr);
                y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3,1,2,0));
                y1 = _mm256_unpackhi_epi8(_mm256_setzero_si256(), y0);
                y0 = _mm256_unpacklo_epi8(_mm256_setzero_si256(), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr +  0), y0);
                _mm256_storeu_si256((__m256i *)(dst_ptr + 16), y1);
            }
        }
    }
}

template<int in_bit_depth>
static void __forceinline convert_yuv444_high_to_yuv444_avx2_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    for (int i = 0; i < 3; i++) {
        uint16_t *srcYLine = (uint16_t *)src[i] + src_y_pitch * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint16_t *src_ptr = srcYLine;
            uint8_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x += 32, dst_ptr += 32, src_ptr += 32) {
                __m256i y0 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 16), (const __m128i *)(src_ptr +  0));
                __m256i y1 = _mm256_loadu2_m128i((const __m128i *)(src_ptr + 24), (const __m128i *)(src_ptr +  8));
                y0 = _mm256_srli_epi16(y0, in_bit_depth - 8);
                y1 = _mm256_srli_epi16(y1, in_bit_depth - 8);
                y0 = _mm256_packus_epi16(y0, y1);
                _mm256_storeu_si256((__m256i *)dst_ptr, y0);
            }
        }
    }
}
#pragma warning(pop)

void convert_yuv444_16_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_14_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_12_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_10_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuv444_09_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yuv444_high_to_yuv444_avx2_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#include "convert_const.h"

static __forceinline void gather_y_uv_from_yc48(__m256i& y0, __m256i& y1, __m256i y2) {
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

static __forceinline __m256i convert_y_range_from_yc48(__m256i y0, __m256i yC_Y_MA_16, int Y_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
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
static __forceinline __m256i convert_uv_range_after_adding_offset(__m256i y0, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
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
static __forceinline __m256i convert_uv_range_from_yc48(__m256i y0, const __m256i& yC_UV_OFFSET_x1, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x1);

    return convert_uv_range_after_adding_offset(y0, yC_UV_MA_16, UV_RSH_16, yC_YCC, yC_pw_one);
}
static __forceinline __m256i convert_uv_range_from_yc48_yuv420p(__m256i y0, __m256i y1, const __m256i& yC_UV_OFFSET_x2, __m256i yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    y0 = _mm256_add_epi16(y0, y1);
    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x2);

    return convert_uv_range_after_adding_offset(y0, yC_UV_MA_16, UV_RSH_16, yC_YCC, yC_pw_one);
}
static __forceinline __m256i convert_uv_range_from_yc48_420i(__m256i y0, __m256i y1, const __m256i& yC_UV_OFFSET_x1, const __m256i& yC_UV_MA_16_0, const __m256i& yC_UV_MA_16_1, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
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

static __forceinline __m256i convert_y_range_to_yc48(__m256i y0) {
    //coef = 4788
    //((( y - 32768 ) * coef) >> 16 ) + (coef/2 - 299)
    __m256i xC_0x8000 = _mm256_slli_epi16(_mm256_cmpeq_epi32(y0, y0), 15);
    y0 = _mm256_add_epi16(y0, xC_0x8000); // -32768
    y0 = _mm256_mulhi_epi16(y0, _mm256_set1_epi16(4788));
    y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(4788/2 - 299));
    return y0;
}

static __forceinline __m256i convert_uv_range_to_yc48(__m256i y0) {
    //coeff = 4682
    //UV = (( uv - 32768 ) * coef + (1<<15) ) >> 16
    //   = (( uv - 32768 + (1<<15)/coef ) * coef ) >> 16
    y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(-32768 + ((1<<15)/4682)));
    y0 = _mm256_mulhi_epi16(y0, _mm256_set1_epi16(4682));
    return y0;
}

static __forceinline void gather_y_u_v_from_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
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


static __forceinline void gather_y_u_v_to_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
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
void convert_yc48_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int x, y;
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    __m256i y0, y1, y2, y3;
    for (y = 0; y < height; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
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

void convert_yc48_to_p010_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int x, y, i;
    short *dst_Y = (short *)dst[0];
    short *dst_C = (short *)dst[1];
    const void  *pixel = src[0];
    const short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    __m256i y0, y1, y2, y3;
    for (y = 0; y < height; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
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

void convert_yc48_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    char *Y_line = (char *)dst[0];
    char *U_line = (char *)dst[1];
    char *V_line = (char *)dst[2];
    char *pixel = (char *)src[0];
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y1, y2, y3;
    for (int y = 0; y < height; y++, pixel += src_y_pitch_byte, Y_line += dst_y_pitch_byte, U_line += dst_y_pitch_byte, V_line += dst_y_pitch_byte) {
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

void convert_yuv444_16bit_to_yc48_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    char *Y_line = (char *)src[0];
    char *U_line = (char *)src[1];
    char *V_line = (char *)src[2];
    char *pixel = (char *)dst[0];
    __m256i y1, y2, y3;
    for (int y = 0; y < height; y++, pixel += dst_y_pitch_byte, Y_line += src_y_pitch_byte, U_line += src_y_pitch_byte, V_line += src_y_pitch_byte) {
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

#pragma warning(pop)
#endif //#if defined(_MSC_VER) || defined(__AVX2__)
