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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "auo.h"
#include "auo_util.h"
#include "auo_video.h"
#include "auo_frm.h"
#include "convert.h"
#include "rgy_simd.h"

//音声の16bit->8bit変換の選択
func_audio_16to8 get_audio_16to8_func(BOOL split) {
    static const func_audio_16to8 FUNC_CONVERT_AUDIO[][2] = {
        { convert_audio_16to8,      split_audio_16to8x2      },
        { convert_audio_16to8_sse2, split_audio_16to8x2_sse2 },
        { convert_audio_16to8_avx2, split_audio_16to8x2_avx2 },
    };
    const auto simd = get_availableSIMD();
    int simdidx = 0;
    if ((simd & RGY_SIMD::AVX2) != RGY_SIMD::NONE)
        simdidx = 2;
    else if ((simd & RGY_SIMD::SSE2) != RGY_SIMD::NONE)
        simdidx = 1;
    return FUNC_CONVERT_AUDIO[simdidx][!!split];
}

#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1 || ENCODER_FFMPEG || ENCODER_VVENC
enum eInterlace {
    A = -1, //区別の必要なし
    P = 0,  //プログレッシブ用
    I = 1   //インターレース用
};

typedef struct {
    int        input_from_aviutl; //Aviutlからの入力に使用する
    int        output_csp;        //出力色空間
    int        bit_depth;         //bit深度
    eInterlace for_interlaced;    //インタレース用関数であるかどうか
    DWORD      mod;               //幅(横解像)に制限(割り切れるかどうか)
    RGY_SIMD   SIMD;              //対応するSIMD
    func_convert_frame func;      //関数へのポインタ
} COVERT_FUNC_INFO;

//表でうっとおしいので省略する
#define NONE  RGY_SIMD::NONE
#define SSE2  RGY_SIMD::SSE2
#define SSE3  RGY_SIMD::SSE3
#define SSSE3 RGY_SIMD::SSSE3
#define SSE41 RGY_SIMD::SSE41
#define SSE42 RGY_SIMD::SSE42
#define AVX   RGY_SIMD::AVX
#define AVX2  RGY_SIMD::AVX2
#define AVX512BW    RGY_SIMD::AVX512BW
#define AVX512VBMI  RGY_SIMD::AVX512VBMI

//なんの数字かわかりやすいようにこう定義する
static const int BIT_8 =  8;
static const int BIT10 = 10;
static const int BIT12 = 12;
static const int BIT16 = 16;

#define ENABLE_NV12 (ENCODER_X264 != 0 || ENCODER_FFMPEG != 0)
#define ENABLE_16BIT (ENCODER_SVTAV1 == 0 && ENCODER_VVENC == 0)

//変換関数のテーブル
//上からチェックするので、より厳しい条件で速い関数を上に書くこと
static const COVERT_FUNC_INFO FUNC_TABLE[] = {
    //YUY2をそのまま渡す
    { CF_YUY2, OUT_CSP_YUY2,   BIT_8, A,  1,  SSE2,                  copy_yuy2_sse2 },
    { CF_YUY2, OUT_CSP_YUY2,   BIT_8, A,  1,  NONE,                  copy_yuy2 },
#if ENABLE_NV12
    //YUY2 -> nv12(8bit)
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  AVX512BW,             convert_yuy2_to_nv12_avx512 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  AVX512BW,             convert_yuy2_to_nv12_i_avx512 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  AVX2|AVX,             convert_yuy2_to_nv12_avx2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  AVX2|AVX,             convert_yuy2_to_nv12_i_avx2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  AVX|SSE2,             convert_yuy2_to_nv12_avx },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  AVX|SSE2,             convert_yuy2_to_nv12_i_avx },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P, 16,  SSE2,                 convert_yuy2_to_nv12_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  SSE2,                 convert_yuy2_to_nv12_sse2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  NONE,                 convert_yuy2_to_nv12 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I, 16,  SSSE3|SSE2,           convert_yuy2_to_nv12_i_ssse3_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  SSSE3|SSE2,           convert_yuy2_to_nv12_i_ssse3 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I, 16,  SSE2,                 convert_yuy2_to_nv12_i_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  SSE2,                 convert_yuy2_to_nv12_i_sse2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  NONE,                 convert_yuy2_to_nv12_i },
    
    //YUY2 -> nv12(16bit)
    { CF_YUY2, OUT_CSP_NV12,   BIT16, P,  1,  AVX2|AVX,             convert_yuy2_to_nv12_16bit_avx2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT16, I,  1,  AVX2|AVX,             convert_yuy2_to_nv12_i_16bit_avx2 },
#else
    //YUY2 -> yv12 (8bit)
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  AVX512BW,             convert_yuy2_to_yv12_avx512 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  AVX512BW,             convert_yuy2_to_yv12_i_avx512 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  AVX2|AVX,             convert_yuy2_to_yv12_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  AVX2|AVX,             convert_yuy2_to_yv12_i_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  AVX|SSE2,             convert_yuy2_to_yv12_avx },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  AVX|SSE2,             convert_yuy2_to_yv12_i_avx },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P, 32,  SSE2,                 convert_yuy2_to_yv12_sse2_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  SSE2,                 convert_yuy2_to_yv12_sse2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  NONE,                 convert_yuy2_to_yv12 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I, 32,  SSSE3|SSE2,           convert_yuy2_to_yv12_i_ssse3_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  SSSE3|SSE2,           convert_yuy2_to_yv12_i_ssse3 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I, 32,  SSE2,                 convert_yuy2_to_yv12_i_sse2_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  SSE2,                 convert_yuy2_to_yv12_i_sse2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  NONE,                 convert_yuy2_to_yv12_i },
    
    //YUY2 -> nv12(16bit)
    { CF_YUY2, OUT_CSP_YV12,   BIT16, P,  1,  AVX2|AVX,             convert_yuy2_to_yv12_16bit_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT16, I,  1,  AVX2|AVX,             convert_yuy2_to_yv12_i_16bit_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT10, P,  1,  AVX2|AVX,             convert_yuy2_to_yv12_10bit_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT10, I,  1,  AVX2|AVX,             convert_yuy2_to_yv12_i_10bit_avx2 },
#endif
#if ENABLE_16BIT
#if ENABLE_NV12
    //YC48 -> nv12 (16bit)
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX512VBMI,           convert_yc48_to_nv12_16bit_avx512vbmi },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX512BW,             convert_yc48_to_nv12_16bit_avx512bw },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX2|AVX,             convert_yc48_to_nv12_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_16bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSE2,                 convert_yc48_to_nv12_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSE2,                 convert_yc48_to_nv12_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  NONE,                 convert_yc48_to_nv12_16bit },
    
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX512VBMI,           convert_yc48_to_nv12_i_16bit_avx512vbmi },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX512BW,             convert_yc48_to_nv12_i_16bit_avx512bw },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX2|AVX,             convert_yc48_to_nv12_i_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_i_16bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_i_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_i_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSE2,                 convert_yc48_to_nv12_i_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSE2,                 convert_yc48_to_nv12_i_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  NONE,                 convert_yc48_to_nv12_i_16bit },
#else
    //YC48 -> yv12 (16bit)
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX512VBMI,           convert_yc48_to_yv12_16bit_avx512vbmi },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX512BW,             convert_yc48_to_yv12_16bit_avx512bw },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX2|AVX,             convert_yc48_to_yv12_16bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_16bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_16bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_16bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSE2,                 convert_yc48_to_yv12_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSE2,                 convert_yc48_to_yv12_16bit_sse2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  NONE,                 convert_yc48_to_yv12_16bit },
    
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX512VBMI,           convert_yc48_to_yv12_i_16bit_avx512vbmi },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX512BW,             convert_yc48_to_yv12_i_16bit_avx512bw },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX2|AVX,             convert_yc48_to_yv12_i_16bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_i_16bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_16bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_i_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_i_16bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSE2,                 convert_yc48_to_yv12_i_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSE2,                 convert_yc48_to_yv12_i_16bit_sse2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  NONE,                 convert_yc48_to_yv12_i_16bit },
#endif
#else
#if ENABLE_NV12
    //YC48 -> nv12 (10bit)
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  AVX512VBMI,           convert_yc48_to_nv12_10bit_avx512vbmi },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  AVX512BW,             convert_yc48_to_nv12_10bit_avx512bw },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  AVX2|AVX,             convert_yc48_to_nv12_10bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_10bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_10bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_10bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_10bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_10bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  8,  SSE2,                 convert_yc48_to_nv12_10bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, P,  1,  SSE2,                 convert_yc48_to_nv12_10bit_sse2 },
        
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  AVX512VBMI,      convert_yc48_to_nv12_i_10bit_avx512vbmi },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  AVX512BW,             convert_yc48_to_nv12_i_10bit_avx512bw },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  AVX2|AVX,             convert_yc48_to_nv12_i_10bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_i_10bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_10bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_10bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_i_10bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_i_10bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  8,  SSE2,                 convert_yc48_to_nv12_i_10bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT10, I,  1,  SSE2,                 convert_yc48_to_nv12_i_10bit_sse2 },
#else
    //YC48 -> yv12 (10bit)
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  AVX512VBMI,           convert_yc48_to_yv12_10bit_avx512vbmi },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  AVX512BW,             convert_yc48_to_yv12_10bit_avx512bw },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  AVX2|AVX,             convert_yc48_to_yv12_10bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_10bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_10bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_10bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_10bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_10bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  8,  SSE2,                 convert_yc48_to_yv12_10bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  SSE2,                 convert_yc48_to_yv12_10bit_sse2 },
        
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  AVX512VBMI,           convert_yc48_to_yv12_i_10bit_avx512vbmi },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  AVX512BW,             convert_yc48_to_yv12_i_10bit_avx512bw },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  AVX2|AVX,             convert_yc48_to_yv12_i_10bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_i_10bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_10bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_10bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_i_10bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_i_10bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  8,  SSE2,                 convert_yc48_to_yv12_i_10bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  SSE2,                 convert_yc48_to_yv12_i_10bit_sse2 },
#endif
#endif
#if ENABLE_NV12
    //YUY2 -> nv16(8bit)
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  AVX2|AVX,             convert_yuy2_to_nv16_avx2 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  AVX|SSE2,             convert_yuy2_to_nv16_avx },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A, 16,  SSE2,                 convert_yuy2_to_nv16_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  SSE2,                 convert_yuy2_to_nv16_sse2 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  NONE,                 convert_yuy2_to_nv16 },
    //YUY2 -> nv16(16bit)
    { CF_YUY2, OUT_CSP_NV16,   BIT16, A,  1,  AVX2|AVX,             convert_yuy2_to_nv16_16bit_avx2 },
    //YC48 -> nv16(16bit)
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  AVX2|AVX,             convert_yc48_to_nv16_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv16_16bit_avx },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv16_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv16_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSSE3|SSE2,           convert_yc48_to_nv16_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSSE3|SSE2,           convert_yc48_to_nv16_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSE2,                 convert_yc48_to_nv16_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSE2,                 convert_yc48_to_nv16_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  NONE,                 convert_yc48_to_nv16_16bit },
#else
    //YUY2 -> yuv422(8bit)
    { CF_YUY2, OUT_CSP_YUV422, BIT_8, A,  1,  AVX2|AVX,             convert_yuy2_to_yuv422_avx2 },
    { CF_YUY2, OUT_CSP_YUV422, BIT_8, A,  1,  NONE,                 convert_yuy2_to_yuv422 },
    
    //YUY2 -> yuv422(16bit)
    { CF_YUY2, OUT_CSP_YUV422, BIT16, A,  1,  AVX2|AVX,             convert_yuy2_to_yuv422_16bit_avx2 },

    //YC48 -> yuv422(16bit)
    { CF_YC48, OUT_CSP_YUV422, BIT16, A,  1,  NONE,                 convert_yc48_to_yuv422_16bit },
#endif
    //YC48 -> yuv444(8bit)
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX512VBMI,           convert_yc48_to_yuv444_avx512vbmi },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX512BW,             convert_yc48_to_yuv444_avx512bw },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX2|AVX,             convert_yc48_to_yuv444_avx2 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yuv444_avx },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_sse41_mod16 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_sse41 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE2,                 convert_yc48_to_yuv444_sse2_mod16 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE2,                 convert_yc48_to_yuv444_sse2 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  NONE,                 convert_yc48_to_yuv444 },

    //YC48 -> yuv444(10bit)
    { CF_YC48, OUT_CSP_YUV444, BIT10, A,  1,  NONE,                 convert_yc48_to_yuv444_10bit },

    //YC48 -> yuv444(16bit)
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX512VBMI,           convert_yc48_to_yuv444_16bit_avx512vbmi },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX512BW,             convert_yc48_to_yuv444_16bit_avx512bw },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX2|AVX,             convert_yc48_to_yuv444_16bit_avx2 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yuv444_16bit_avx },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_16bit_sse41 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  8,  SSE2,                 convert_yc48_to_yuv444_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  SSE2,                 convert_yc48_to_yuv444_16bit_sse2 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  NONE,                 convert_yc48_to_yuv444_16bit },
#if ENABLE_NV12
    //LW48 -> nv12 (8bit)
    { CF_LW48, OUT_CSP_NV12,   BIT_8, P,  1,  NONE,                 convert_lw48_to_nv12 },
    { CF_LW48, OUT_CSP_NV12,   BIT_8, I,  1,  NONE,                 convert_lw48_to_nv12_i },
    //LW48 -> nv12 (16bit)
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  AVX512VBMI,           convert_lw48_to_nv12_i_16bit_avx512vbmi },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  AVX512VBMI,           convert_lw48_to_nv12_16bit_avx512vbmi },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  AVX512BW,             convert_lw48_to_nv12_i_16bit_avx512bw },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  AVX512BW,             convert_lw48_to_nv12_16bit_avx512bw },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_nv12_i_16bit_avx2 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  AVX2|AVX,             convert_lw48_to_nv12_16bit_avx2 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_nv12_i_16bit_avx },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_nv12_16bit_avx },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  8,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv12_16bit_sse41_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv12_16bit_sse41 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  8,  SSSE3|SSE2,           convert_lw48_to_nv12_16bit_ssse3_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  SSSE3|SSE2,           convert_lw48_to_nv12_16bit_ssse3 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  8,  SSE2,                 convert_lw48_to_nv12_16bit_sse2_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  SSE2,                 convert_lw48_to_nv12_16bit_sse2 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, P,  1,  NONE,                 convert_lw48_to_nv12_16bit },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  8,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv12_i_16bit_sse41_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv12_i_16bit_sse41 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  8,  SSSE3|SSE2,           convert_lw48_to_nv12_i_16bit_ssse3_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  SSSE3|SSE2,           convert_lw48_to_nv12_i_16bit_ssse3 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  8,  SSE2,                 convert_lw48_to_nv12_i_16bit_sse2_mod8 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  SSE2,                 convert_lw48_to_nv12_i_16bit_sse2 },
    { CF_LW48, OUT_CSP_NV12,   BIT16, I,  1,  NONE,                 convert_lw48_to_nv12_i_16bit },

    //LW48 -> nv16 (8bit)
    { CF_LW48, OUT_CSP_NV16,   BIT_8, A,  1,  NONE,                 convert_lw48_to_nv16 },

    //LW48 -> nv16 (16bit)
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  AVX2|AVX,             convert_lw48_to_nv16_16bit_avx2 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_nv16_16bit_avx },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv16_16bit_sse41_mod8 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_lw48_to_nv16_16bit_sse41 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  8,  SSSE3|SSE2,           convert_lw48_to_nv16_16bit_ssse3_mod8 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  SSSE3|SSE2,           convert_lw48_to_nv16_16bit_ssse3 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  8,  SSE2,                 convert_lw48_to_nv16_16bit_sse2_mod8 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  SSE2,                 convert_lw48_to_nv16_16bit_sse2 },
    { CF_LW48, OUT_CSP_NV16,   BIT16, A,  1,  NONE,                 convert_lw48_to_nv16_16bit },
#endif
    //LW48 -> yuv444 (8bit)
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX512VBMI,           convert_lw48_to_yuv444_avx512vbmi },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX512BW,             convert_lw48_to_yuv444_avx512bw },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX2|AVX,             convert_lw48_to_yuv444_avx2 },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_yuv444_avx },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE41|SSSE3|SSE2,     convert_lw48_to_yuv444_sse41_mod16 },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE41|SSSE3|SSE2,     convert_lw48_to_yuv444_sse41 },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE2,                 convert_lw48_to_yuv444_sse2_mod16 },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE2,                 convert_lw48_to_yuv444_sse2 },
    { CF_LW48, OUT_CSP_YUV444, BIT_8, A,  1,  NONE,                 convert_lw48_to_yuv444 },

    //LW48 -> yuv444 (16bit)
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  AVX512VBMI,           convert_lw48_to_yuv444_16bit_avx512vbmi },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  AVX512BW,             convert_lw48_to_yuv444_16bit_avx512bw },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  AVX2|AVX,             convert_lw48_to_yuv444_16bit_avx2 },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_lw48_to_yuv444_16bit_avx },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_lw48_to_yuv444_16bit_sse41_mod8 },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_lw48_to_yuv444_16bit_sse41 },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  8,  SSE2,                 convert_lw48_to_yuv444_16bit_sse2_mod8 },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  SSE2,                 convert_lw48_to_yuv444_16bit_sse2 },
    { CF_LW48, OUT_CSP_YUV444, BIT16, A,  1,  NONE,                 convert_lw48_to_yuv444_16bit },
#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1
    //Copy RGB
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  SSSE3|SSE2,           sort_to_rgb_ssse3 },
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  NONE,                 sort_to_rgb },
#elif ENCODER_FFMPEG
    //Copy RGB
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  SSE2,                 copy_rgb_sse2 },
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  NONE,                 copy_rgb },
    //Copy RGBA
    { CF_RGBA,  OUT_CSP_RGBA,  BIT_8, A,  1,  SSE2,                 copy_rgba_sse2 },
    { CF_RGBA,  OUT_CSP_RGBA,  BIT_8, A,  1,  NONE,                 copy_rgba },
#endif
    //Convert RGB to YUV444
    { CF_RGB,  OUT_CSP_YUV444, BIT_8, A,  1,  AVX2|AVX,             convert_rgb_to_yuv444_avx2 },
    { CF_RGB,  OUT_CSP_YUV444, BIT_8, A,  1,  NONE,                 convert_rgb_to_yuv444 },
    { CF_RGB,  OUT_CSP_YUV444, BIT16, A,  1,  AVX2|AVX,             convert_rgb_to_yuv444_16bit_avx2 },
    { CF_RGB,  OUT_CSP_YUV444, BIT16, A,  1,  NONE,                 convert_rgb_to_yuv444_16bit },
    //Convert PA64 to YUV444
    { CF_PA64,  OUT_CSP_YUV444, BIT_8, A,  1,  AVX2|AVX,            convert_pa64_to_yuv444_avx2 },
    { CF_PA64,  OUT_CSP_YUV444, BIT16, A,  1,  AVX2|AVX,            convert_pa64_to_yuv444_16bit_avx2 },
    //Convert PA64 to RGBA
    { CF_PA64,  OUT_CSP_RGBA,  BIT_8, A,  1,  AVX2|AVX,             convert_pa64_to_rgba_avx2 },
    { CF_PA64,  OUT_CSP_RGBA,  BIT16, A,  1,  AVX2|AVX,             convert_pa64_to_rgba_16bit_avx2 },
    { 0, 0, 0, A, 0, NONE, NULL }
};

static void build_simd_info(RGY_SIMD simd, wchar_t *buf, DWORD nSize) {
    ZeroMemory(buf, nSize);
    if (simd != NONE) {
        wcscpy_s(buf, nSize, L", using");
        if ((simd & SSE2)   == SSE2) wcscat_s(buf, nSize, L" SSE2");
        if ((simd & SSE3)   == SSE3) wcscat_s(buf, nSize, L" SSE3");
        if ((simd & SSSE3)  == SSSE3) wcscat_s(buf, nSize, L" SSSE3");
        if ((simd & SSE41)  == SSE41) wcscat_s(buf, nSize, L" SSE4.1");
        if ((simd & SSE42)  == SSE42) wcscat_s(buf, nSize, L" SSE4.2");
        if ((simd & AVX)    == AVX) wcscat_s(buf, nSize, L" AVX");
        if ((simd & AVX2)   == AVX2) wcscat_s(buf, nSize, L" AVX2");
        if ((simd & AVX512BW  ) == AVX512BW  ) wcscat_s(buf, nSize, L" AVX512BW");
        if ((simd & AVX512VBMI) == AVX512VBMI) wcscat_s(buf, nSize, L" AVX512VBMI");
    }
}
#undef NONE
#undef SSE2
#undef SSE3
#undef SSSE3
#undef SSE41
#undef SSE42
#undef AVX
#undef AVX2
#undef AVX512BW
#undef AVX512VBMI

static void auo_write_func_info(const COVERT_FUNC_INFO *func_info) {
    wchar_t simd_buf[128];
    build_simd_info(func_info->SIMD, simd_buf, _countof(simd_buf));

    if (func_info->output_csp == OUT_CSP_YUY2) {
        write_log_auo_line_fmt(LOG_INFO, L"Passing YUY2%s", simd_buf);
        return;
    }

    if (func_info->output_csp == OUT_CSP_RGB) {
        write_log_auo_line_fmt(LOG_INFO, L"Copying RGB%s", simd_buf);
        return;
    }

    const wchar_t *interlaced = NULL;
    switch (func_info->for_interlaced) {
        case P: interlaced = L"p"; break;
        case I: interlaced = L"i"; break;
        case A:
        default:interlaced = L""; break;
    }
    const wchar_t *bit_depth = L"";
    switch (func_info->bit_depth) {
        case BIT10: bit_depth = L"(10bit)"; break;
        case BIT12: bit_depth = L"(12bit)"; break;
        case BIT16: bit_depth = L"(16bit)"; break;
        default: break;
    }

    write_log_auo_line_fmt(LOG_INFO, L"converting %s -> %s%s%s%s",
        CF_NAME[func_info->input_from_aviutl],
        specify_csp[func_info->output_csp],
        interlaced,
        bit_depth,
        simd_buf);
};

//C4189 : ローカル変数が初期化されましたが、参照されていません。
#pragma warning( push )
#pragma warning( disable: 4189 )
//使用する関数を選択する
func_convert_frame get_convert_func(int width, int input_csp, int bit_depth, BOOL interlaced, int output_csp) {
    const auto availableSIMD = get_availableSIMD();

    const COVERT_FUNC_INFO *func_info = NULL;
    for (int i = 0; FUNC_TABLE[i].func; i++) {
        if (FUNC_TABLE[i].input_from_aviutl != input_csp)
            continue;
        if (FUNC_TABLE[i].output_csp != output_csp)
            continue;
        if (FUNC_TABLE[i].bit_depth != bit_depth)
            continue;
        if (FUNC_TABLE[i].for_interlaced != A &&
            FUNC_TABLE[i].for_interlaced != (eInterlace)interlaced)
            continue;
        if ((width % FUNC_TABLE[i].mod) != 0)
            continue;
        if ((FUNC_TABLE[i].SIMD & availableSIMD) != FUNC_TABLE[i].SIMD)
            continue;

        func_info = &FUNC_TABLE[i];
        break;
    }

    if (func_info == NULL)
        return NULL;

    auo_write_func_info(func_info);
    return func_info->func;
}
#pragma warning( pop )

static uint32_t get_align_size(const RGY_SIMD simd_check, const int to_yv12) {
    if ((simd_check & (RGY_SIMD::AVX512F|RGY_SIMD::AVX512DQ|RGY_SIMD::AVX512BW|RGY_SIMD::AVX512VBMI|RGY_SIMD::AVX512VNNI)) != RGY_SIMD::NONE) {
        return (to_yv12) ? 128 : 64;
    } else if ((simd_check & RGY_SIMD::AVX2) != RGY_SIMD::NONE) {
        return (to_yv12) ? 64 : 32;
    } else if ((simd_check & RGY_SIMD::SSE2) != RGY_SIMD::NONE) {
        return (to_yv12) ? 32 : 16;
    } else {
        return 1;
    }
}

BOOL malloc_pixel_data(CONVERT_CF_DATA * const pixel_data, int width, int height, int output_csp, int bit_depth) {
    BOOL ret = TRUE;
#if ENABLE_NV12
    const int to_yv12 = FALSE;
#else
    const int to_yv12 = (output_csp == OUT_CSP_YV12);
#endif
    const DWORD pixel_size = (bit_depth > 8) ? sizeof(short) : sizeof(BYTE);
    const auto simd_check = get_availableSIMD();
    const DWORD align_size = get_align_size(simd_check,to_yv12);
#define ALIGN_NEXT(i, align) (((i) + (align-1)) & (~(align-1))) //alignは2の累乗(1,2,4,8,16,32...)
    const DWORD extra = align_size * 2;
    const DWORD frame_size = ALIGN_NEXT(width * height * pixel_size + extra, align_size);
#undef ALIGN_NEXT

    ZeroMemory(pixel_data->data, sizeof(pixel_data->data));
    switch (output_csp) {
        case OUT_CSP_YUY2: //YUY2であってもコピーフレーム機能をサポートするためにはコピーが必要となる
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 2, std::max(align_size, 16ul))) == NULL)
                ret = FALSE;
            break;
#if ENABLE_NV12
        case OUT_CSP_NV16:
            if (   (pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size, std::max(align_size, 16ul))) == NULL
                || (pixel_data->data[1] = (BYTE *)_mm_malloc(frame_size, std::max(align_size, 16ul))) == NULL)
                ret = FALSE;
            break;
        case OUT_CSP_NV12:
        case OUT_CSP_P010:
        default:
            if (   ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size,             std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[1] = (BYTE *)_mm_malloc(frame_size / 2 + extra, std::max(align_size, 16ul))) == NULL))
                ret = FALSE;
            break;
#else
        case OUT_CSP_YUV422:
            if (   ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size,             std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[1] = (BYTE *)_mm_malloc(frame_size / 2 + extra, std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[2] = (BYTE *)_mm_malloc(frame_size / 2 + extra, std::max(align_size, 16ul))) == NULL))
                ret = FALSE;
            break;
        case OUT_CSP_YV12:
            if (   ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size,             std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[1] = (BYTE *)_mm_malloc(frame_size / 4 + extra, std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[2] = (BYTE *)_mm_malloc(frame_size / 4 + extra, std::max(align_size, 16ul))) == NULL))
                ret = FALSE;
            break;
#endif
        case OUT_CSP_YUV444:
        case OUT_CSP_YUV444_16:
            if (   ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size, std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[1] = (BYTE *)_mm_malloc(frame_size, std::max(align_size, 16ul))) == NULL)
                || ((pixel_data->data[2] = (BYTE *)_mm_malloc(frame_size, std::max(align_size, 16ul))) == NULL))
                ret = FALSE;
            break;
        case OUT_CSP_RGB:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 3, std::max(align_size, 16ul))) == NULL)
                ret = FALSE;
            break;
        case OUT_CSP_RGBA:
        case OUT_CSP_RGBA_16:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 4, std::max(align_size, 16ul))) == NULL)
                ret = FALSE;
            break;
    }
    pixel_data->colormatrix = height >= 720 ? 1 : 0;
    return ret;
}

void free_pixel_data(CONVERT_CF_DATA *pixel_data) {
    for (size_t i = 0; i < _countof(pixel_data->data); i++)
        if (pixel_data->data[i])
            _mm_free(pixel_data->data[i]);
    ZeroMemory(pixel_data, sizeof(CONVERT_CF_DATA));
}
#endif
