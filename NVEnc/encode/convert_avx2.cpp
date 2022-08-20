// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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
// --------------------------------------------------------------------------------------------

//AVX2用コード
#include <immintrin.h> //イントリンシック命令 AVX / AVX2

#include "convert.h"
#include "convert_const.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif


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

void convert_audio_16to8_avx2(BYTE *dst, short *src, int n) {
    BYTE *byte = dst;
    short *sh = src;
    BYTE * const loop_start = (BYTE *)(((size_t)dst + 31) & ~31);
    BYTE * const loop_fin   = (BYTE *)(((size_t)dst + n) & ~31);
    BYTE * const fin = dst + n;
    __m256i ySA, ySB;
    static const __m256i yConst = _mm256_set1_epi16(128);
    //アライメント調整
    while (byte < loop_start) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
    //メインループ
    while (byte < loop_fin) {
        ySA = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 16)), _mm_loadu_si128((__m128i*)(sh +  0)));
        ySB = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 24)), _mm_loadu_si128((__m128i*)(sh +  8)));
        ySA = _mm256_srai_epi16(ySA, 8);
        ySB = _mm256_srai_epi16(ySB, 8);
        ySA = _mm256_add_epi16(ySA, yConst);
        ySB = _mm256_add_epi16(ySB, yConst);
        ySA = _mm256_packus_epi16(ySA, ySB);
        _mm256_stream_si256((__m256i *)byte, ySA);
        sh += 32;
        byte += 32;
    }
    //残り
    while (byte < fin) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
}

void split_audio_16to8x2_avx2(BYTE *dst, short *src, int n) {
    BYTE *byte0 = dst;
    BYTE *byte1 = dst + n;
    short *sh = src;
    short *sh_fin = src + (n & ~15);
    __m256i y0, y1, y2, y3;
    __m256i yMask = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
    __m256i yConst = _mm256_set1_epi8(-128);
    for ( ; sh < sh_fin; sh += 32, byte0 += 32, byte1 += 32) {
        y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 16)), _mm_loadu_si128((__m128i*)(sh + 0)));
        y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 24)), _mm_loadu_si128((__m128i*)(sh + 8)));
        y2 = _mm256_and_si256(y0, yMask); //Lower8bit
        y3 = _mm256_and_si256(y1, yMask); //Lower8bit
        y0 = _mm256_srli_epi16(y0, 8);    //Upper8bit
        y1 = _mm256_srli_epi16(y1, 8);    //Upper8bit
        y2 = _mm256_packus_epi16(y2, y3);
        y0 = _mm256_packus_epi16(y0, y1);
        y2 = _mm256_add_epi8(y2, yConst);
        y0 = _mm256_add_epi8(y0, yConst);
        _mm256_storeu_si256((__m256i*)byte0, y0);
        _mm256_storeu_si256((__m256i*)byte1, y2);
    }
    sh_fin = sh + (n & 15);
    for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
        *byte0 = (*sh >> 8)   + 128;
        *byte1 = (*sh & 0xff) + 128;
    }
}


static __forceinline void separate_low_up(__m256i& y0_return_lower, __m256i& y1_return_upper) {
    __m256i y4, y5;
    const __m256i xMaskLowByte = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
    y4 = _mm256_srli_epi16(y0_return_lower, 8);
    y5 = _mm256_srli_epi16(y1_return_upper, 8);

    y0_return_lower = _mm256_and_si256(y0_return_lower, xMaskLowByte);
    y1_return_upper = _mm256_and_si256(y1_return_upper, xMaskLowByte);

    y0_return_lower = _mm256_packus_epi16(y0_return_lower, y1_return_upper);
    y1_return_upper = _mm256_packus_epi16(y4, y5);
}
static __forceinline void separate_low_up_16bit(__m256i& y0_return_lower, __m256i& y1_return_upper) {
    __m256i y4, y5;
    const __m256i xMaskLowByte = _mm256_srli_epi32(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 16);

    y4 = y0_return_lower; //128,   0
    y5 = y1_return_upper; //384, 256
    y0_return_lower = _mm256_permute2x128_si256(y4, y5, (2<<4)+0); //256,   0
    y1_return_upper = _mm256_permute2x128_si256(y4, y5, (3<<4)+1); //384, 128

    y4 = _mm256_srli_epi32(y0_return_lower, 16);
    y5 = _mm256_srli_epi32(y1_return_upper, 16);

    y0_return_lower = _mm256_and_si256(y0_return_lower, xMaskLowByte);
    y1_return_upper = _mm256_and_si256(y1_return_upper, xMaskLowByte);

    y0_return_lower = _mm256_packus_epi32(y0_return_lower, y1_return_upper);
    y1_return_upper = _mm256_packus_epi32(y4, y5);
}

void convert_yuy2_to_nv12_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y;
    BYTE *p, *pw, *Y, *C;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_C = pixel_data->data[1];
    __m256i y0, y1, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 2) {
        x  = y * width;
        p  = (BYTE *)frame + (x<<1);
        pw = p + (width<<1);
        Y  = (BYTE *)dst_Y +  x;
        C  = (BYTE *)dst_C + (x>>1);
        for (x = 0; x < width; x += 32, p += 64, pw += 64) {
            //-----------1行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

            separate_low_up(y0, y1);
            y3 = y1;

            _mm256_storeu_si256((__m256i *)(Y + x), y0);
            //-----------1行目終了---------------

            //-----------2行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

            separate_low_up(y0, y1);

            _mm256_storeu_si256((__m256i *)(Y + width + x), y0);
            //-----------2行目終了---------------

            y1 = _mm256_avg_epu8(y1, y3);  //VUVUVUVUVUVUVUVU
            _mm256_storeu_si256((__m256i *)(C + x), y1);
        }
    }
    _mm256_zeroupper();
}

void convert_yuy2_to_yv12_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y;
    BYTE *p, *pw, *Y, *U, *V;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_U = pixel_data->data[1];
    BYTE *dst_V = pixel_data->data[2];
    __m256i y0, y1, y3, y6;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 2) {
        x  = y * width;
        p  = (BYTE *)frame + (x<<1);
        pw = p + (width<<1);
        Y  = (BYTE *)dst_Y +  x;
        U  = (BYTE *)dst_U + (x>>2);
        V  = (BYTE *)dst_V + (x>>2);
        for (x = 0; x < width; x += 64, p += 128, pw += 128) {
            //-----------1行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

            separate_low_up(y0, y1);
            y3 = y1;

            _mm256_storeu_si256((__m256i *)(Y + x), y0);

            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+ 96)), _mm_loadu_si128((__m128i*)(p+64)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+112)), _mm_loadu_si128((__m128i*)(p+80)));

            separate_low_up(y0, y1);
            y6 = y1;

            _mm256_storeu_si256((__m256i *)(Y + x + 32), y0);
            //-----------1行目終了---------------

            //-----------2行目---------------
            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

            separate_low_up(y0, y1);

            _mm256_storeu_si256((__m256i *)(Y + width + x), y0);

            y3 = _mm256_avg_epu8(y1, y3);

            y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+ 96)), _mm_loadu_si128((__m128i*)(pw+64)));
            y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+112)), _mm_loadu_si128((__m128i*)(pw+80)));

            separate_low_up(y0, y1);

            _mm256_storeu_si256((__m256i *)(Y + width + x + 32), y0);
            //-----------2行目終了---------------

            y6 = _mm256_avg_epu8(y1, y6);  //VUVUVUVUVUVUVUVU

            y0 = _mm256_permute2x128_si256(y3, y6, (0x02<< 4) + 0x00);
            y1 = _mm256_permute2x128_si256(y3, y6, (0x03<< 4) + 0x01);
            separate_low_up(y0, y1);

            _mm256_storeu_si256((__m256i *)(U + (x>>1)), y0);
            _mm256_storeu_si256((__m256i *)(V + (x>>1)), y1);
        }
    }
    _mm256_zeroupper();
}
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

void convert_yuy2_to_nv12_i_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y, i;
    BYTE *p, *pw, *Y, *C;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_C = pixel_data->data[1];
    __m256i y0, y1, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 4) {
        for (i = 0; i < 2; i++) {
            x  = (y + i) * width;
            p  = (BYTE *)frame + (x<<1);
            pw = p + (width<<2);
            Y  = (BYTE *)dst_Y +  x;
            C  = (BYTE *)dst_C + ((x+width*i)>>1);
            for (x = 0; x < width; x += 32, p += 64, pw += 64) {
                //-----------    1+i行目   ---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

                separate_low_up(y0, y1);
                y3 = y1;

                _mm256_storeu_si256((__m256i *)(Y + x), y0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

                separate_low_up(y0, y1);

                _mm256_storeu_si256((__m256i *)(Y + (width<<1) + x), y0);
                //-----------3+i行目終了---------------

                y0 = yuv422_to_420_i_interpolate(y3, y1, i);

                _mm256_storeu_si256((__m256i *)(C + x), y0);
            }
        }
    }
    _mm256_zeroupper();
}

void convert_yuy2_to_yv12_i_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y, i;
    BYTE *p, *pw, *Y, *U, *V;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_U = pixel_data->data[1];
    BYTE *dst_V = pixel_data->data[2];
    __m256i y0, y1, y3, y6;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 4) {
        for (i = 0; i < 2; i++) {
            x  = (y + i) * width;
            p  = (BYTE *)frame + (x<<1);
            pw = p + (width<<2);
            Y  = (BYTE *)dst_Y +  x;
            U  = (BYTE *)dst_U + ((x+width*i)>>2);
            V  = (BYTE *)dst_V + ((x+width*i)>>2);
            for (x = 0; x < width; x += 64, p += 128, pw += 128) {
                //-----------    1+i行目   ---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

                separate_low_up(y0, y1);
                y3 = y1;

                _mm256_storeu_si256((__m256i *)(Y + x), y0);

                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+ 96)), _mm_loadu_si128((__m128i*)(p+64)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+112)), _mm_loadu_si128((__m128i*)(p+80)));

                separate_low_up(y0, y1);
                y6 = y1;

                _mm256_storeu_si256((__m256i *)(Y + x + 32), y0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

                separate_low_up(y0, y1);

                _mm256_storeu_si256((__m256i *)(Y + (width<<1) + x), y0);

                y3 = yuv422_to_420_i_interpolate(y3, y1, i);

                y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+ 96)), _mm_loadu_si128((__m128i*)(pw+64)));
                y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+112)), _mm_loadu_si128((__m128i*)(pw+80)));

                separate_low_up(y0, y1);

                _mm256_storeu_si256((__m256i *)(Y + (width<<1) + x + 32), y0);
                //-----------3+i行目終了---------------

                y0 = y3;
                y1 = yuv422_to_420_i_interpolate(y6, y1, i);

                _mm256_storeu_si256((__m256i *)(U + (x>>1)), y0);
                _mm256_storeu_si256((__m256i *)(V + (x>>1)), y1);
            }
        }
    }
    _mm256_zeroupper();
}

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

void convert_yc48_to_nv12_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y0, y1, y2, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = (short*)dst_Y + width * y;
        C   = (short*)dst_C + width * y / 2;
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

            _mm256_storeu_si256((__m256i *)(Y + x + width), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            _mm256_storeu_si256((__m256i *)(C + x), convert_uv_range_from_yc48_yuv420p(y0, y2,  _mm256_set1_epi16(UV_OFFSET_x2), yC_UV_L_MA_16_420P, UV_L_RSH_16_420P, yC_YCC, yC_pw_one));
        }
    }
    _mm256_zeroupper();
}

void convert_yc48_to_nv12_i_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y, i;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y0, y1, y2, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = (short*)dst_Y + width * (y + i);
            C   = (short*)dst_C + width * (y + i*2) / 2;
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

                _mm256_storeu_si256((__m256i *)(Y + x + width*2), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                _mm256_storeu_si256((__m256i *)(C + x), convert_uv_range_from_yc48_420i(y0, y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_420I(i), yC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, yC_YCC, yC_pw_one));
            }
        }
    }
    _mm256_zeroupper();
}

void convert_yc48_to_yv12_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_U = (short *)pixel_data->data[1];
    short *dst_V = (short *)pixel_data->data[2];
    short *ycp, *ycpw;
    short *Y = NULL, *U = NULL, *V = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y0, y1, y2, y3, y4;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = (short*)dst_Y + width * y;
        U   = (short*)dst_U + width * y / 4;
        V   = (short*)dst_V + width * y / 4;
        for (x = 0; x < width; x += 32, ycp += 96, ycpw += 96) {
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

            _mm256_storeu_si256((__m256i *)(Y + x + width), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            y4 = convert_uv_range_from_yc48_yuv420p(y0, y2,  _mm256_set1_epi16(UV_OFFSET_x2), yC_UV_L_MA_16_420P, UV_L_RSH_16_420P, yC_YCC, yC_pw_one);

            y1 = _mm256_loadu_si256((__m256i *)(ycp + 48)); // 128, 0
            y2 = _mm256_loadu_si256((__m256i *)(ycp + 64)); // 384, 256
            y3 = _mm256_loadu_si256((__m256i *)(ycp + 80)); // 640, 512

            gather_y_uv_from_yc48(y1, y2, y3);
            y0 = y2;

            _mm256_storeu_si256((__m256i *)(Y + x + 16), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            y1 = _mm256_loadu_si256((__m256i *)(ycpw + 48));
            y2 = _mm256_loadu_si256((__m256i *)(ycpw + 64));
            y3 = _mm256_loadu_si256((__m256i *)(ycpw + 80));

            gather_y_uv_from_yc48(y1, y2, y3);

            _mm256_storeu_si256((__m256i *)(Y + x + 16 + width), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

            y0 = convert_uv_range_from_yc48_yuv420p(y0, y2,  _mm256_set1_epi16(UV_OFFSET_x2), yC_UV_L_MA_16_420P, UV_L_RSH_16_420P, yC_YCC, yC_pw_one);

            separate_low_up_16bit(y4, y0);

            _mm256_storeu_si256((__m256i *)(U + (x>>1)), y4);
            _mm256_storeu_si256((__m256i *)(V + (x>>1)), y0);
        }
    }
    _mm256_zeroupper();
}

void convert_yc48_to_yv12_i_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y, i;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_U = (short *)pixel_data->data[1];
    short *dst_V = (short *)pixel_data->data[2];
    short *ycp, *ycpw;
    short *Y = NULL, *U = NULL, *V = NULL;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y0, y1, y2, y3, y4;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = (short*)dst_Y + width * (y + i);
            U   = (short*)dst_U + width * (y + i*2) / 4;
            V   = (short*)dst_V + width * (y + i*2) / 4;
            for (x = 0; x < width; x += 32, ycp += 96, ycpw += 96) {
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

                _mm256_storeu_si256((__m256i *)(Y + x + width*2), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                y4 = convert_uv_range_from_yc48_420i(y0, y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_420I(i), yC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, yC_YCC, yC_pw_one);

                y1 = _mm256_loadu_si256((__m256i *)(ycp + 48)); // 128, 0
                y2 = _mm256_loadu_si256((__m256i *)(ycp + 64)); // 384, 256
                y3 = _mm256_loadu_si256((__m256i *)(ycp + 80)); // 640, 512

                gather_y_uv_from_yc48(y1, y2, y3);
                y0 = y2;

                _mm256_storeu_si256((__m256i *)(Y + x + 16), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                y1 = _mm256_loadu_si256((__m256i *)(ycpw + 48));
                y2 = _mm256_loadu_si256((__m256i *)(ycpw + 64));
                y3 = _mm256_loadu_si256((__m256i *)(ycpw + 80));

                gather_y_uv_from_yc48(y1, y2, y3);

                _mm256_storeu_si256((__m256i *)(Y + x + 16 + width*2), convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));

                y0 = convert_uv_range_from_yc48_420i(y0, y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_420I(i), yC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, yC_YCC, yC_pw_one);

                separate_low_up_16bit(y4, y0);

                _mm256_storeu_si256((__m256i *)(U + (x>>1)), y4);
                _mm256_storeu_si256((__m256i *)(V + (x>>1)), y0);
            }
        }
    }
    _mm256_zeroupper();
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

void convert_yc48_to_yuv444_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    BYTE *Y = (BYTE *)pixel_data->data[0] + width * y_range.s;
    BYTE *U = (BYTE *)pixel_data->data[1] + width * y_range.s;
    BYTE *V = (BYTE *)pixel_data->data[2] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short *const ycp_fin = (short *)pixel + width * y_range.e * 3;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y1, y2, y3, yY, yU, yV;
    for (; ycp < ycp_fin; ycp += 96, Y += 32, U += 32, V += 32) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_u_v_from_yc48(y1, y2, y3);

        y1 = convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one);
        y2 = convert_uv_range_from_yc48(y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one);
        y3 = convert_uv_range_from_yc48(y3, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one);
        yY = _mm256_srli_epi16(y1, 8);
        yU = _mm256_srli_epi16(y2, 8);
        yV = _mm256_srli_epi16(y3, 8);

        y1 = _mm256_loadu_si256((__m256i *)(ycp + 48));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 64));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 80));

        gather_y_u_v_from_yc48(y1, y2, y3);

        y1 = convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one);
        y2 = convert_uv_range_from_yc48(y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one);
        y3 = convert_uv_range_from_yc48(y3, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one);
        y1 = _mm256_srli_epi16(y1, 8);
        y2 = _mm256_srli_epi16(y2, 8);
        y3 = _mm256_srli_epi16(y3, 8);

        yY = _mm256_packus_epi16(yY, y1);
        yU = _mm256_packus_epi16(yU, y2);
        yV = _mm256_packus_epi16(yV, y3);

        yY = _mm256_permute4x64_epi64(yY, _MM_SHUFFLE(3,1,2,0));
        yU = _mm256_permute4x64_epi64(yU, _MM_SHUFFLE(3,1,2,0));
        yV = _mm256_permute4x64_epi64(yV, _MM_SHUFFLE(3,1,2,0));

        _mm256_storeu_si256((__m256i *)Y, yY);
        _mm256_storeu_si256((__m256i *)U, yU);
        _mm256_storeu_si256((__m256i *)V, yV);
    }
    _mm256_zeroupper();
}
void convert_yc48_to_yuv444_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    short *Y = (short *)pixel_data->data[0] + width * y_range.s;
    short *U = (short *)pixel_data->data[1] + width * y_range.s;
    short *V = (short *)pixel_data->data[2] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short *const ycp_fin = (short *)pixel + width * y_range.e * 3;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y1, y2, y3;
    for (; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_u_v_from_yc48(y1, y2, y3);

        _mm256_storeu_si256((__m256i *)Y, convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));
        _mm256_storeu_si256((__m256i *)U, convert_uv_range_from_yc48(y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one));
        _mm256_storeu_si256((__m256i *)V, convert_uv_range_from_yc48(y3, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one));
    }
    _mm256_zeroupper();
}

void convert_yuy2_to_nv16_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    BYTE *p = (BYTE *)pixel + width * y_range.s * 2;
    BYTE * const p_fin = (BYTE *)pixel + width * y_range.e * 2;
    BYTE *dst_Y = pixel_data->data[0] + width * y_range.s;
    BYTE *dst_C = pixel_data->data[1] + width * y_range.s;
    __m256i y0, y1;
    for (; p < p_fin; p += 64, dst_Y += 32, dst_C += 32) {
        y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
        y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

        separate_low_up(y0, y1);

        _mm256_storeu_si256((__m256i *)dst_Y, y0);
        _mm256_storeu_si256((__m256i *)dst_C, y1);
    }
    _mm256_zeroupper();
}

void convert_yc48_to_nv16_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    short *dst_Y = (short *)pixel_data->data[0] + width * y_range.s;
    short *dst_C = (short *)pixel_data->data[1] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short * const ycp_fin = (short *)pixel + width * y_range.e * 3;
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    const __m256i yC_YCC = _mm256_set1_epi32(1<<LSFT_YCC_16);
    __m256i y1, y2, y3;
    for (; ycp < ycp_fin; ycp += 48, dst_Y += 16, dst_C += 16) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_uv_from_yc48(y1, y2, y3);

        _mm256_storeu_si256((__m256i *)dst_Y, convert_y_range_from_yc48(y1, yC_Y_L_MA_16, Y_L_RSH_16, yC_YCC, yC_pw_one));
        _mm256_storeu_si256((__m256i *)dst_C, convert_uv_range_from_yc48(y2, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_16_444, UV_L_RSH_16_444, yC_YCC, yC_pw_one));
    }
    _mm256_zeroupper();
}
void convert_lw48_to_nv12_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    __m256i y0, y1, y2, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = (short*)dst_Y + width * y;
        C   = (short*)dst_C + width * y / 2;
        for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
            y1 = _mm256_loadu_si256((__m256i *)(ycp +  0)); // 128, 0
            y2 = _mm256_loadu_si256((__m256i *)(ycp + 16)); // 384, 256
            y3 = _mm256_loadu_si256((__m256i *)(ycp + 32)); // 640, 512

            gather_y_uv_from_yc48(y1, y2, y3);
            y0 = y2;

            _mm256_storeu_si256((__m256i *)(Y + x), y1);

            y1 = _mm256_loadu_si256((__m256i *)(ycpw +  0));
            y2 = _mm256_loadu_si256((__m256i *)(ycpw + 16));
            y3 = _mm256_loadu_si256((__m256i *)(ycpw + 32));

            gather_y_uv_from_yc48(y1, y2, y3);

            _mm256_storeu_si256((__m256i *)(Y + x + width), y1);

            y0 = _mm256_avg_epu16(y0, y2);

            _mm256_storeu_si256((__m256i *)(C + x), y0);
        }
    }
    _mm256_zeroupper();
}

void convert_lw48_to_nv12_i_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    int x, y, i;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    __m256i y0, y1, y2, y3;
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    for (y = y_range.s; y < y_range.e; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = (short*)dst_Y + width * (y + i);
            C   = (short*)dst_C + width * (y + i*2) / 2;
            for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
                y1 = _mm256_loadu_si256((__m256i *)(ycp +  0)); // 128, 0
                y2 = _mm256_loadu_si256((__m256i *)(ycp + 16)); // 384, 256
                y3 = _mm256_loadu_si256((__m256i *)(ycp + 32)); // 640, 512

                gather_y_uv_from_yc48(y1, y2, y3);
                y0 = y2;

                _mm256_storeu_si256((__m256i *)(Y + x), y1);

                y1 = _mm256_loadu_si256((__m256i *)(ycpw +  0));
                y2 = _mm256_loadu_si256((__m256i *)(ycpw + 16));
                y3 = _mm256_loadu_si256((__m256i *)(ycpw + 32));

                gather_y_uv_from_yc48(y1, y2, y3);

                _mm256_storeu_si256((__m256i *)(Y + x + width*2), y1);

                y1 = _mm256_unpacklo_epi16(y0, y2);
                y0 = _mm256_unpackhi_epi16(y0, y2);
                y1 = _mm256_madd_epi16(y1, yC_INTERLACE_WEIGHT(i));
                y0 = _mm256_madd_epi16(y0, yC_INTERLACE_WEIGHT(i));
                y1 = _mm256_srli_epi32(y1, 2);
                y0 = _mm256_srli_epi32(y0, 2);
                y1 = _mm256_packus_epi32(y1, y0);

                _mm256_storeu_si256((__m256i *)(C + x), y1);
            }
        }
    }
    _mm256_zeroupper();
}
void convert_lw48_to_nv16_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    short *dst_Y = (short *)pixel_data->data[0] + width * y_range.s;
    short *dst_C = (short *)pixel_data->data[1] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short * const ycp_fin = (short *)pixel + width * y_range.e * 3;
    __m256i y1, y2, y3;
    for (; ycp < ycp_fin; ycp += 48, dst_Y += 16, dst_C += 16) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_uv_from_yc48(y1, y2, y3);

        _mm256_storeu_si256((__m256i *)dst_Y, y1);
        _mm256_storeu_si256((__m256i *)dst_C, y2);
    }
    _mm256_zeroupper();
}
void convert_lw48_to_yuv444_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    BYTE *Y = (BYTE *)pixel_data->data[0] + width * y_range.s;
    BYTE *U = (BYTE *)pixel_data->data[1] + width * y_range.s;
    BYTE *V = (BYTE *)pixel_data->data[2] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short *const ycp_fin = (short *)pixel + width * y_range.e * 3;
    __m256i y1, y2, y3, yY, yU, yV;
    for (; ycp < ycp_fin; ycp += 96, Y += 16, U += 16, V += 16) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_u_v_from_yc48(y1, y2, y3);

        yY = _mm256_srli_epi16(y1, 8);
        yU = _mm256_srli_epi16(y2, 8);
        yV = _mm256_srli_epi16(y3, 8);

        y1 = _mm256_loadu_si256((__m256i *)(ycp + 48));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 64));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 80));

        gather_y_u_v_from_yc48(y1, y2, y3);

        y1 = _mm256_srli_epi16(y1, 8);
        y2 = _mm256_srli_epi16(y2, 8);
        y3 = _mm256_srli_epi16(y3, 8);

        yY = _mm256_packus_epi16(yY, y1);
        yU = _mm256_packus_epi16(yU, y2);
        yV = _mm256_packus_epi16(yV, y3);

        yY = _mm256_permute4x64_epi64(yY, _MM_SHUFFLE(3,1,2,0));
        yU = _mm256_permute4x64_epi64(yU, _MM_SHUFFLE(3,1,2,0));
        yV = _mm256_permute4x64_epi64(yV, _MM_SHUFFLE(3,1,2,0));

        _mm256_storeu_si256((__m256i *)Y, yY);
        _mm256_storeu_si256((__m256i *)U, yU);
        _mm256_storeu_si256((__m256i *)V, yV);
    }
    _mm256_zeroupper();
}
void convert_lw48_to_yuv444_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    const auto y_range = thread_y_range(height, thread_id, thread_n);
    short *Y = (short *)pixel_data->data[0] + width * y_range.s;
    short *U = (short *)pixel_data->data[1] + width * y_range.s;
    short *V = (short *)pixel_data->data[2] + width * y_range.s;
    short *ycp = (short *)pixel + width * y_range.s * 3;
    short *const ycp_fin = (short *)pixel + width * y_range.e * 3;
    __m256i y1, y2, y3;
    for (; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
        y1 = _mm256_loadu_si256((__m256i *)(ycp +  0));
        y2 = _mm256_loadu_si256((__m256i *)(ycp + 16));
        y3 = _mm256_loadu_si256((__m256i *)(ycp + 32));

        gather_y_u_v_from_yc48(y1, y2, y3);

        _mm256_storeu_si256((__m256i *)Y, y1);
        _mm256_storeu_si256((__m256i *)U, y2);
        _mm256_storeu_si256((__m256i *)V, y3);
    }
    _mm256_zeroupper();
}
