// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#ifndef _CONVERT_H_
#define _CONVERT_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

enum {
    OUT_CSP_NV12,
    OUT_CSP_NV16,
    OUT_CSP_YUV444,
    OUT_CSP_RGB,
    OUT_CSP_YUY2,
};

//x264のinput-cspとして使用するもの
//OUT_CSP_NV12, OUT_CSP_YUV444, OUT_CSP_RGB に合わせる
static const char * const specify_csp[] = {
    "nv12", //OUT_CSP_NV12
    "nv16", //OUT_CSP_NV16
    "i444", //OUT_CSP_YUV444
    "rgb"   //OUT_CSP_RGB
};

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

struct AUO_Y_RANGE {
    int s, e;
};

static inline AUO_Y_RANGE thread_y_range(int h, int thread_id, int thread_n) {
    AUO_Y_RANGE y_range;
    y_range.s = ((((h *  thread_id)    / thread_n) + 3) & ~3);
    y_range.e = ((((h * (thread_id+1)) / thread_n) + 3) & ~3);
    if (y_range.e > h) {
        y_range.e = h;
    }
    return y_range;
}

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

typedef void(*func_convert_frame) (void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void copy_yuy2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void copy_yuy2_sse2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void sort_to_rgb(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void sort_to_rgb_ssse3(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);



//YUY2 -> nv12 (8bit)
void convert_yuy2_to_nv12(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_nv12_sse2_mod16(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_sse2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i_sse2_mod16(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i_sse2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_nv12_i_ssse3_mod16(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i_ssse3(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_nv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv12_i_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);


//YUY2 -> yv12 (8bit)
void convert_yuy2_to_yv12(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_yv12_sse2_mod32(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_sse2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i_sse2_mod32(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i_sse2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_yv12_i_ssse3_mod32(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i_ssse3(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_yv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_yv12_i_avx2(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);






//YC48 -> nv12 (16bit)
void convert_yc48_to_nv12_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_nv12_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_nv12_16bit_ssse3(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_ssse3(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_16bit_ssse3_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_ssse3_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_nv12_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_nv12_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv12_i_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);


//YC48 -> yv12 (16bit)
void convert_yc48_to_yv12_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yv12_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yv12_16bit_ssse3(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_ssse3(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_16bit_ssse3_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_ssse3_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yv12_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yv12_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

//YC48 -> yv12 (10bit)
void convert_yc48_to_yv12_10bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yv12_i_10bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);


//YUY2 -> nv16
void convert_yuy2_to_nv16(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yuy2_to_nv16_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv16_sse2_mod16(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv16_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yuy2_to_nv16_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

//YUY2 -> yuv422
void convert_yuy2_to_yuv422(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

//YC48 -> nv16 (16bit)
void convert_yc48_to_nv16_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_nv16_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_ssse3(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_ssse3_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_nv16_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);



//YC48 -> yuv444
void convert_yc48_to_yuv444(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yuv444_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_sse2_mod16(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_sse41_mod16(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yuv444_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);


//YC48 -> yuv444 (10bit)
void convert_yc48_to_yuv444_10bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
//YC48 -> yuv444 (16bit)
void convert_yc48_to_yuv444_16bit(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

void convert_yc48_to_yuv444_16bit_sse2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_16bit_sse2_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_16bit_sse41(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_16bit_sse41_mod8(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_16bit_avx(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);
void convert_yc48_to_yuv444_16bit_avx2(void *pixel, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n);

#endif //_CONVERT_H_
