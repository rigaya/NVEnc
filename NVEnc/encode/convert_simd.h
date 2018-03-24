//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#ifndef _CONVERT_SIMD_H_
#define _CONVERT_SIMD_H_
#include <emmintrin.h> //イントリンシック命令 SSE2
#if USE_SSSE3
#include <tmmintrin.h> //イントリンシック命令 SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //イントリンシック命令 SSE4.1
#endif
#include "convert.h"
#include "convert_const.h"

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
    static const _declspec(align(64)) DWORD VAL[2][4] = {
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

static __forceinline void separate_low_up_16bit(__m128i& x0_return_lower, __m128i& x1_return_upper) {
    __m128i x4, x5;
    const __m128i xMaskLowByte = _mm_srli_epi32(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 16);
    x4 = _mm_srli_epi32(x0_return_lower, 16);
    x5 = _mm_srli_epi32(x1_return_upper, 16);

    x0_return_lower = _mm_and_si128(x0_return_lower, xMaskLowByte);
    x1_return_upper = _mm_and_si128(x1_return_upper, xMaskLowByte);

    x0_return_lower = _mm_packus_epi32_simd(x0_return_lower, x1_return_upper);
    x1_return_upper = _mm_packus_epi32_simd(x4, x5);
}

//YUY2->NV12 SSE2版
template <BOOL aligned_store>
static __forceinline void convert_yuy2_to_nv12_simd(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y;
    BYTE *p, *pw, *Y, *C;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_C = pixel_data->data[1];
    __m128i x0, x1, x3;
    for (y = 0; y < height; y += 2) {
        x  = y * width;
        p  = (BYTE *)frame + (x<<1);
        pw = p + (width<<1);
        Y  = (BYTE *)dst_Y +  x;
        C  = (BYTE *)dst_C + (x>>1);
        for (x = 0; x < width; x += 16, p += 32, pw += 32) {
            //-----------1行目---------------
            x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(p+16));
            
            separate_low_up(x0, x1);
            x3 = x1;

            _mm_store_switch_si128((__m128i *)(Y + x), x0);
            //-----------1行目終了---------------

            //-----------2行目---------------
            x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(pw+16));
            
            separate_low_up(x0, x1);

            _mm_store_switch_si128((__m128i *)(Y + width + x), x0);
            //-----------2行目終了---------------

            x1 = _mm_avg_epu8(x1, x3);
            _mm_store_switch_si128((__m128i *)(C + x), x1);
        }
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

template <BOOL aligned_store>
static __forceinline void convert_yuy2_to_nv12_i_simd(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y, i;
    BYTE *p, *pw, *Y, *C;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_C = pixel_data->data[1];
    __m128i x0, x1, x3;
    for (y = 0; y < height; y += 4) {
        for (i = 0; i < 2; i++) {
            x  = (y + i) * width;
            p  = (BYTE *)frame + (x<<1);
            pw = p + (width<<2);
            Y  = (BYTE *)dst_Y +  x;
            C  = (BYTE *)dst_C + ((x+width*i)>>1);
            for (x = 0; x < width; x += 16, p += 32, pw += 32) {
                //-----------    1+i行目   ---------------
                x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(p+16));
            
                separate_low_up(x0, x1);
                x3 = x1;

                _mm_store_switch_si128((__m128i *)(Y + x), x0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(pw+16));
            
                separate_low_up(x0, x1);

                _mm_store_switch_si128((__m128i *)(Y + (width<<1) + x), x0);
                //-----------3+i行目終了---------------
                x0 = yuv422_to_420_i_interpolate(x3, x1, i);

                _mm_store_switch_si128((__m128i *)(C + x), x0);
            }
        }
    }
}

//YUY2->YV12 SSE2版
template <BOOL aligned_store>
static void convert_yuy2_to_yv12_simd(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y;
    BYTE *p, *pw, *Y, *U, *V;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_U = pixel_data->data[1];
    BYTE *dst_V = pixel_data->data[2];
    __m128i x0, x1, x3, x6;
    for (y = 0; y < height; y += 2) {
        x  = y * width;
        p  = (BYTE *)frame + (x<<1);
        pw = p + (width<<1);
        Y  = (BYTE *)dst_Y +  x;
        U  = (BYTE *)dst_U + (x>>2);
        V  = (BYTE *)dst_V + (x>>2);
        for (x = 0; x < width; x += 32, p += 64, pw += 64) {

            x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(p+16));
            
            separate_low_up(x0, x1);
            x3 = x1;

            _mm_store_switch_si128((__m128i *)(Y + x), x0);

            x0 = _mm_loadu_si128((const __m128i *)(p+32));
            x1 = _mm_loadu_si128((const __m128i *)(p+48));
            
            separate_low_up(x0, x1);
            x6 = x1;

            _mm_store_switch_si128((__m128i *)(Y + x + 16), x0);

            x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
            x1 = _mm_loadu_si128((const __m128i *)(pw+16));
            
            separate_low_up(x0, x1);

            _mm_store_switch_si128((__m128i *)(Y + width + x), x0);

            x3 = _mm_avg_epu8(x1, x3);

            x0 = _mm_loadu_si128((const __m128i *)(pw+32));
            x1 = _mm_loadu_si128((const __m128i *)(pw+48));
            
            separate_low_up(x0, x1);

            _mm_store_switch_si128((__m128i *)(Y + width + x + 16), x0);

            x0 = x3;
            x1 = _mm_avg_epu8(x1, x6);
            
            separate_low_up(x0, x1);
            
            _mm_store_switch_si128((__m128i *)(U + (x>>1)), x0);
            _mm_store_switch_si128((__m128i *)(V + (x>>1)), x1);
        }
    }
}

template <BOOL aligned_store>
static __forceinline void convert_yuy2_to_yv12_i_simd(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y, i;
    BYTE *p, *pw, *Y, *U, *V;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_U = pixel_data->data[1];
    BYTE *dst_V = pixel_data->data[2];
    __m128i x0, x1, x3, x6;
    for (y = 0; y < height; y += 4) {
        for (i = 0; i < 2; i++) {
            x  = (y + i) * width;
            p  = (BYTE *)frame + (x<<1);
            pw = p + (width<<2);
            Y  = (BYTE *)dst_Y +  x;
            U  = (BYTE *)dst_U + ((x+width*i)>>2);
            V  = (BYTE *)dst_V + ((x+width*i)>>2);
            for (x = 0; x < width; x += 32, p += 64, pw += 64) {
                //-----------    1+i行目   ---------------
                x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(p+16));

                separate_low_up(x0, x1);
                x3 = x1;

                _mm_store_switch_si128((__m128i *)(Y + x), x0);

                x0 = _mm_loadu_si128((const __m128i *)(p+32));
                x1 = _mm_loadu_si128((const __m128i *)(p+48));
            
                separate_low_up(x0, x1);
                x6 = x1;

                _mm_store_switch_si128((__m128i *)(Y + x + 16), x0);
                //-----------1+i行目終了---------------

                //-----------3+i行目---------------
                x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(pw+16));
            
                separate_low_up(x0, x1);

                _mm_store_switch_si128((__m128i *)(Y + (width<<1) + x), x0);
                
                x3 = yuv422_to_420_i_interpolate(x3, x1, i);

                x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
                x1 = _mm_loadu_si128((const __m128i *)(pw+16));
            
                separate_low_up(x0, x1);

                _mm_store_switch_si128((__m128i *)(Y + (width<<1) + x + 16), x0);
                
                x0 = x3;
                x1 = yuv422_to_420_i_interpolate(x6, x1, i);
            
                separate_low_up(x0, x1);
            
                _mm_store_switch_si128((__m128i *)(U + (x>>1)), x0);
                _mm_store_switch_si128((__m128i *)(V + (x>>1)), x1);
            }
        }
    }
}

template <BOOL aligned_store>
static void convert_yuy2_to_nv16_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    BYTE *p = (BYTE *)pixel;
    BYTE * const p_fin = p + width * height * 2;
    BYTE *dst_Y = pixel_data->data[0];
    BYTE *dst_C = pixel_data->data[1];
    __m128i x0, x1;
    for (; p < p_fin; p += 32, dst_Y += 16, dst_C += 16) {
        x0 = _mm_loadu_si128((__m128i *)(p+ 0));
        x1 = _mm_loadu_si128((__m128i *)(p+16));
        
        separate_low_up(x0, x1);

        _mm_store_switch_si128((__m128i *)dst_Y, x0);
        _mm_store_switch_si128((__m128i *)dst_C, x1);
    }
}

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

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_nv12_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x0, x1, x2, x3;
    for (y = 0; y < height; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = (short*)dst_Y + width * y;
        C   = (short*)dst_C + width * y / 2;
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

            _mm_store_switch_si128((__m128i *)(Y + x + width), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

            x0 = convert_uv_range_from_yc48_yuv420p(x0, x2, _mm_set1_epi16(UV_OFFSET_x2), xC_UV_L_MA_16_420P, UV_L_RSH_16_420P, xC_YCC, xC_pw_one);

            _mm_store_switch_si128((__m128i *)(C + x), x0);
        }
    }
}

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_nv12_i_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y, i;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp, *ycpw;
    short *Y = NULL, *C = NULL;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x0, x1, x2, x3;
    for (y = 0; y < height; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = (short*)dst_Y + width * (y + i);
            C   = (short*)dst_C + width * (y + i*2) / 2;
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
                _mm_store_switch_si128((__m128i *)(Y + x + width*2), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                _mm_store_switch_si128((__m128i *)(C + x), convert_uv_range_from_yc48_420i(x0, x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_420I(i), xC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, xC_YCC, xC_pw_one));
            }
        }
    }
}

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_yv12_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_U = (short *)pixel_data->data[1];
    short *dst_V = (short *)pixel_data->data[2];
    short *ycp, *ycpw;
    short *Y = NULL, *U = NULL, *V = NULL;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x0, x1, x2, x3, x4;
    for (y = 0; y < height; y += 2) {
        ycp = (short*)pixel + width * y * 3;
        ycpw= ycp + width*3;
        Y   = (short*)dst_Y + width * y;
        U   = (short*)dst_U + width * y / 4;
        V   = (short*)dst_V + width * y / 4;
        for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
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

            _mm_store_switch_si128((__m128i *)(Y + x + width), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));
            
            x4 = convert_uv_range_from_yc48_yuv420p(x0, x2, _mm_set1_epi16(UV_OFFSET_x2), xC_UV_L_MA_16_420P, UV_L_RSH_16_420P, xC_YCC, xC_pw_one);
            
            x1 = _mm_loadu_si128((__m128i *)(ycp + 24));
            x2 = _mm_loadu_si128((__m128i *)(ycp + 32));
            x3 = _mm_loadu_si128((__m128i *)(ycp + 40));
            _mm_prefetch((const char *)ycpw, _MM_HINT_T1);
            gather_y_uv_from_yc48(x1, x2, x3);
            x0 = x2;

            _mm_store_switch_si128((__m128i *)(Y + x + 8), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

            x1 = _mm_loadu_si128((__m128i *)(ycpw + 24));
            x2 = _mm_loadu_si128((__m128i *)(ycpw + 32));
            x3 = _mm_loadu_si128((__m128i *)(ycpw + 40));
            gather_y_uv_from_yc48(x1, x2, x3);

            _mm_store_switch_si128((__m128i *)(Y + x + 8 + width), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

            x0 = convert_uv_range_from_yc48_yuv420p(x0, x2, _mm_set1_epi16(UV_OFFSET_x2), xC_UV_L_MA_16_420P, UV_L_RSH_16_420P, xC_YCC, xC_pw_one);

            separate_low_up_16bit(x4, x0);
            
            _mm_store_switch_si128((__m128i *)(U + (x>>1)), x4);
            _mm_store_switch_si128((__m128i *)(V + (x>>1)), x0);
        }
    }
}

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_yv12_i_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    int x, y, i;
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_U = (short *)pixel_data->data[1];
    short *dst_V = (short *)pixel_data->data[2];
    short *ycp, *ycpw;
    short *Y = NULL, *U = NULL, *V = NULL;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x0, x1, x2, x3, x4;
    for (y = 0; y < height; y += 4) {
        for (i = 0; i < 2; i++) {
            ycp = (short*)pixel + width * (y + i) * 3;
            ycpw= ycp + width*2*3;
            Y   = (short*)dst_Y + width * (y + i);
            U   = (short*)dst_U + width * (y + i*2) / 4;
            V   = (short*)dst_V + width * (y + i*2) / 4;
            for (x = 0; x < width; x += 16, ycp += 48, ycpw += 48) {
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
                _mm_store_switch_si128((__m128i *)(Y + x + width*2), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                x4 = convert_uv_range_from_yc48_420i(x0, x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_420I(i), xC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, xC_YCC, xC_pw_one);
                
                x1 = _mm_loadu_si128((__m128i *)(ycp + 24));
                x2 = _mm_loadu_si128((__m128i *)(ycp + 32));
                x3 = _mm_loadu_si128((__m128i *)(ycp + 40));
                _mm_prefetch((const char *)ycpw, _MM_HINT_T1);
                gather_y_uv_from_yc48(x1, x2, x3);
                x0 = x2;
                _mm_store_switch_si128((__m128i *)(Y + x + 8), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                x1 = _mm_loadu_si128((__m128i *)(ycpw + 24));
                x2 = _mm_loadu_si128((__m128i *)(ycpw + 32));
                x3 = _mm_loadu_si128((__m128i *)(ycpw + 40));
                gather_y_uv_from_yc48(x1, x2, x3);
                _mm_store_switch_si128((__m128i *)(Y + x + 8 + width*2), convert_y_range_from_yc48(x1, xC_Y_L_MA_16, Y_L_RSH_16, xC_YCC, xC_pw_one));

                x0 = convert_uv_range_from_yc48_420i(x0, x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_420I(i), xC_UV_L_MA_16_420I((i+1)&0x01), UV_L_RSH_16_420I, xC_YCC, xC_pw_one);
                 
                separate_low_up_16bit(x4, x0);
                
                _mm_store_switch_si128((__m128i *)(U + x), x4);
                _mm_store_switch_si128((__m128i *)(V + x), x0);
            }
        }
    }
}

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_nv16_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    short *dst_Y = (short *)pixel_data->data[0];
    short *dst_C = (short *)pixel_data->data[1];
    short *ycp = (short *)pixel;
    short * const ycp_fin = ycp + width * height * 3;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x1, x2, x3;
    for (; ycp < ycp_fin; ycp += 24, dst_Y += 8, dst_C += 8) {
        x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
        x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
        x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
        gather_y_uv_from_yc48(x1, x2, x3);
        _mm_store_switch_si128((__m128i *)dst_Y, convert_y_range_from_yc48( x1,                                xC_Y_L_MA_16,      Y_L_RSH_16,     xC_YCC, xC_pw_one));
        _mm_store_switch_si128((__m128i *)dst_C, convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one));
    }
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
    static const _declspec(align(16)) USHORT maskY_select[8] = { 0xffff, 0x0000, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_select);

    x5 = select_by_mask(x2, x0, xMask);
    xMask = _mm_slli_si128(xMask, 2);
    x5 = select_by_mask(x5, x1, xMask); //52741630

    x6 = _mm_unpacklo_epi16(x5, x5);    //11663300
    x7 = _mm_unpackhi_epi16(x5, x5);    //55227744
    
    static const _declspec(align(16)) USHORT maskY_shuffle[8] = { 0xffff, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_shuffle);
    x5 = select_by_mask(x7, x6, xMask);                 //51627340
    x5 = _mm_shuffle_epi32(x5, _MM_SHUFFLE(1,2,3,0));   //73625140

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
    x0 = _mm_shuffle_epi32(x6, _MM_SHUFFLE(1,2,3,0)); //v6 u6 v4 u4 v2 u2 v0 u0
    x1 = _mm_shuffle_epi32(x7, _MM_SHUFFLE(3,0,1,2)); //v7 u7 v5 u5 v3 u3 v1 u1

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

template <BOOL aligned_store>
static __forceinline void convert_yc48_to_yuv444_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    BYTE *Y = (BYTE *)pixel_data->data[0];
    BYTE *U = (BYTE *)pixel_data->data[1];
    BYTE *V = (BYTE *)pixel_data->data[2];
    short *ycp;
    short *const ycp_fin = (short *)pixel + width * height * 3;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x1, x2, x3, xY, xU, xV;
    for (ycp = (short *)pixel; ycp < ycp_fin; ycp += 48, Y += 16, U += 16, V += 16) {
        x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
        x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
        x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
        gather_y_u_v_from_yc48(x1, x2, x3);

        x1 = convert_y_range_from_yc48( x1,                                xC_Y_L_MA_16,      Y_L_RSH_16,     xC_YCC, xC_pw_one);
        x2 = convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
        x3 = convert_uv_range_from_yc48(x3, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one);
        xY = _mm_srli_epi16(x1, 8);
        xU = _mm_srli_epi16(x2, 8);
        xV = _mm_srli_epi16(x3, 8);

        x1 = _mm_loadu_si128((__m128i *)(ycp + 24));
        x2 = _mm_loadu_si128((__m128i *)(ycp + 32));
        x3 = _mm_loadu_si128((__m128i *)(ycp + 40));
        gather_y_u_v_from_yc48(x1, x2, x3);

        x1 = convert_y_range_from_yc48( x1,                                xC_Y_L_MA_16,      Y_L_RSH_16,     xC_YCC, xC_pw_one);
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
template <BOOL aligned_store>
static __forceinline void convert_yc48_to_yuv444_16bit_simd(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    short *Y = (short *)pixel_data->data[0];
    short *U = (short *)pixel_data->data[1];
    short *V = (short *)pixel_data->data[2];
    short *ycp;
    short *const ycp_fin = (short *)pixel + width * height * 3;
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    const __m128i xC_YCC = _mm_set1_epi32(1<<LSFT_YCC_16);
    __m128i x1, x2, x3;
    for (ycp = (short *)pixel; ycp < ycp_fin; ycp += 24, Y += 8, U += 8, V += 8) {
        x1 = _mm_loadu_si128((__m128i *)(ycp +  0));
        x2 = _mm_loadu_si128((__m128i *)(ycp +  8));
        x3 = _mm_loadu_si128((__m128i *)(ycp + 16));
        gather_y_u_v_from_yc48(x1, x2, x3);
        _mm_store_switch_si128((__m128i *)Y, convert_y_range_from_yc48( x1,                                xC_Y_L_MA_16,      Y_L_RSH_16,     xC_YCC, xC_pw_one));
        _mm_store_switch_si128((__m128i *)U, convert_uv_range_from_yc48(x2, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one));
        _mm_store_switch_si128((__m128i *)V, convert_uv_range_from_yc48(x3, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_16_444, UV_L_RSH_16_444, xC_YCC, xC_pw_one));
    }
}


#if USE_SSSE3
static __forceinline void sort_to_rgb_simd(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    static const __m128i xC_SHUF = _mm_set_epi8(16, 12, 13, 14, 9, 10,  11,  6,  7,  8,  3,  4,  5,  0,  1,  2);
    BYTE *ptr = pixel_data->data[0];
    BYTE *dst, *src, *src_fin;
    int y0 = 0, y1 = height - 1;
    const int step = (width*3 + 3) & ~3;
    for (; y0 < height; y0++, y1--) {
        dst = ptr          + y0*width*3;
        src = (BYTE*)frame + y1*step;
        src_fin = src + width*3;
        for (; src < src_fin; src += 15, dst += 15) {
            __m128i x0 = _mm_loadu_si128((const __m128i *)src);
            x0 = _mm_shuffle_epi8(x0, xC_SHUF);
            _mm_storeu_si128((__m128i *)dst, x0);
        }
    }
}
#endif
#endif //_CONVERT_SIMD_H_
