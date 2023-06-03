// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2023 rigaya
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

#define RGY_MEMMEM_AVX2
#include "rgy_faw.h"

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)

size_t rgy_memmem_fawstart1_avx2(const void *data_, const size_t data_size) {
    return rgy_memmem_avx2_imp(data_, data_size, fawstart1.data(), fawstart1.size());
}

void rgy_convert_audio_16to8_avx2(uint8_t *dst, const short *src, const size_t n) {
    uint8_t *byte = dst;
    const short *sh = src;
    uint8_t * const loop_start = (uint8_t *)(((size_t)dst + 31) & ~31);
    uint8_t * const loop_fin = (uint8_t *)(((size_t)dst + n) & ~31);
    uint8_t * const fin = dst + n;
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
        ySA = _mm256_set_m128i(_mm_loadu_si128((const __m128i*)(sh + 16)), _mm_loadu_si128((const __m128i*)(sh + 0)));
        ySB = _mm256_set_m128i(_mm_loadu_si128((const __m128i*)(sh + 24)), _mm_loadu_si128((const __m128i*)(sh + 8)));
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

void rgy_split_audio_16to8x2_avx2(uint8_t *dst0, uint8_t *dst1, const short *src, const size_t n) {
    const short *sh = src;
    const short *sh_fin = src + (n & ~15);
    __m256i y0, y1, y2, y3;
    __m256i yMask = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
    __m256i yConst = _mm256_set1_epi8(-128);
    for (; sh < sh_fin; sh += 32, dst0 += 32, dst1 += 32) {
        y0 = _mm256_set_m128i(_mm_loadu_si128((const __m128i*)(sh + 16)), _mm_loadu_si128((const __m128i*)(sh + 0)));
        y1 = _mm256_set_m128i(_mm_loadu_si128((const __m128i*)(sh + 24)), _mm_loadu_si128((const __m128i*)(sh + 8)));
        y2 = _mm256_and_si256(y0, yMask); //Lower8bit
        y3 = _mm256_and_si256(y1, yMask); //Lower8bit
        y0 = _mm256_srli_epi16(y0, 8);    //Upper8bit
        y1 = _mm256_srli_epi16(y1, 8);    //Upper8bit
        y2 = _mm256_packus_epi16(y2, y3);
        y0 = _mm256_packus_epi16(y0, y1);
        y2 = _mm256_add_epi8(y2, yConst);
        y0 = _mm256_add_epi8(y0, yConst);
        _mm256_storeu_si256((__m256i*)dst0, y0);
        _mm256_storeu_si256((__m256i*)dst1, y2);
    }
    sh_fin = sh + (n & 15);
    for (; sh < sh_fin; sh++, dst0++, dst1++) {
        *dst0 = (*sh >> 8) + 128;
        *dst1 = (*sh & 0xff) + 128;
    }
}
#endif
