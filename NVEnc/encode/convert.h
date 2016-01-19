// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 1999-2016 rigaya
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

#ifndef _CONVERT_H_
#define _CONVERT_H_

typedef    struct {
    short    y;                    //    画素(輝度    )データ (     0 ～ 4096 )
    short    cb;                    //    画素(色差(青))データ ( -2048 ～ 2048 )
    short    cr;                    //    画素(色差(赤))データ ( -2048 ～ 2048 )
                                //    画素データは範囲外に出ていることがあります
                                //    また範囲内に収めなくてもかまいません
} PIXEL_YC;

typedef struct {
    int   count;       //planarの枚数。packedなら1
    BYTE *data[3];     //planarの先頭へのポインタ
    int   size[3];     //planarのサイズ
    int   total_size;  //全planarのサイズの総和
} CONVERT_CF_DATA;

//音声16bit->8bit変換
typedef void (*func_audio_16to8) (BYTE *dst, short *src, int n);

func_audio_16to8 get_audio_16to8_func(BOOL split); //使用する音声16bit->8bit関数の選択

void convert_audio_16to8(BYTE *dst, short *src, int n);
void convert_audio_16to8_sse2(BYTE *dst, short *src, int n);

void split_audio_16to8x2(BYTE *dst, short *src, int n);
void split_audio_16to8x2_sse2(BYTE *dst, short *src, int n);

#if (_MSC_VER >= 1700)
void convert_audio_16to8_avx2(BYTE *dst, short *src, int n);
void split_audio_16to8x2_avx2(BYTE *dst, short *src, int n);
#endif

typedef void (*func_convert_frame) (void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);

void copy_yuy2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);

void convert_yuy2_to_nv12_sse2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_sse2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_sse2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_sse2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_ssse3(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_ssse3_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);

#if (_MSC_VER >= 1600)
void convert_yuy2_to_nv12_avx(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_avx_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_avx(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_avx_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
#endif
#if (_MSC_VER >= 1700)
void convert_yuy2_to_nv12_avx2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_avx2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_avx2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
void convert_yuy2_to_nv12_i_avx2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch);
#endif

#define ALIGN32_CONST_ARRAY static const _declspec(align(32))

ALIGN32_CONST_ARRAY BYTE  Array_INTERLACE_WEIGHT[2][32] = { 
    {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
    {3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define xC_INTERLACE_WEIGHT(i) _mm_load_si128((__m128i*)Array_INTERLACE_WEIGHT[i])

#endif //_CONVERT_H_
