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
#include <stdio.h>
#include <mmintrin.h>  //イントリンシック命令 SSE
#include <emmintrin.h> //イントリンシック命令 SSE2

#include "convert.h"
#include "convert_const.h"

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

//直前の16byteアライメント
static inline void * get_aligned_next(void *p) {
    return (void *)(((size_t)p + 15) & ~15);
}
//直後の16byteアライメント
static inline void * get_aligned_prev(void *p) {
    return (void *)(((size_t)p) & ~15);
}
//16bit音声 -> 8bit音声
void convert_audio_16to8(BYTE *dst, short *src, int n) {
    BYTE *byte = dst;
    const BYTE *fin = byte + n;
    short *sh = src;
    while (byte < fin) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
}

void split_audio_16to8x2(BYTE *dst, short *src, int n) {
    BYTE *byte0 = dst;
    BYTE *byte1 = dst + n;
    short *sh = src;
    short *sh_fin = src + n;
    for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
        *byte0 = (*sh >> 8)   + 128;
        *byte1 = (*sh & 0xff) + 128;
    }
}

void split_audio_16to8x2_sse2(BYTE *dst, short *src, int n) {
    BYTE *byte0 = dst;
    BYTE *byte1 = dst + n;
    short *sh = src;
    short *sh_fin = src + (n & ~15);
    __m128i x0, x1, x2, x3;
    __m128i xMask = _mm_srli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8);
    __m128i xConst = _mm_set1_epi8(-128);
    for ( ; sh < sh_fin; sh += 16, byte0 += 16, byte1 += 16) {
        x0 = _mm_loadu_si128((__m128i*)(sh + 0));
        x1 = _mm_loadu_si128((__m128i*)(sh + 8));
        x2 = _mm_and_si128(x0, xMask); //Lower8bit
        x3 = _mm_and_si128(x1, xMask); //Lower8bit
        x0 = _mm_srli_epi16(x0, 8);    //Upper8bit
        x1 = _mm_srli_epi16(x1, 8);    //Upper8bit
        x2 = _mm_packus_epi16(x2, x3);
        x0 = _mm_packus_epi16(x0, x1);
        x2 = _mm_add_epi8(x2, xConst);
        x0 = _mm_add_epi8(x0, xConst);
        _mm_storeu_si128((__m128i*)byte0, x0);
        _mm_storeu_si128((__m128i*)byte1, x2);
    }
    sh_fin = sh + (n & 15);
    for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
        *byte0 = (*sh >> 8)   + 128;
        *byte1 = (*sh & 0xff) + 128;
    }
}

//上のSSE2版
void convert_audio_16to8_sse2(BYTE *dst, short *src, int n) {
    BYTE *byte = dst;
    short *sh = src;
    BYTE * const loop_start = (BYTE *)get_aligned_next(dst);
    BYTE * const loop_fin   = (BYTE *)get_aligned_prev(dst + n);
    BYTE * const fin = dst + n;
    __m128i xSA, xSB;
    static const __m128i xConst = _mm_set1_epi16(128);
    //アライメント調整
    while (byte < loop_start) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
    //メインループ
    while (byte < loop_fin) {
        xSA = _mm_loadu_si128((const __m128i *)sh);
        sh += 8;
        xSA = _mm_srai_epi16(xSA, 8);
        xSA = _mm_add_epi16(xSA, xConst);
        xSB = _mm_loadu_si128((const __m128i *)sh);
        sh += 8;
        xSB = _mm_srai_epi16(xSB, 8);
        xSB = _mm_add_epi16(xSB, xConst);
        xSA = _mm_packus_epi16(xSA, xSB);
        _mm_stream_si128((__m128i *)byte, xSA);
        byte += 16;
    }
    //残り
    while (byte < fin) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
}
