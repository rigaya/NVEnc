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

#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1

#include "convert_simd.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

//AVXが使用可能なSandyBridge以降のCPUでは、
//メモリがalignされていれば、_mm_store_si128 / _mm_storeu_si128 に速度差はないため
//関数も_mm_storeu_si128のもののみ用意する

void convert_yuy2_to_nv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yuy2_to_nv12_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yuy2_to_nv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yuy2_to_nv12_i_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yuy2_to_yv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yuy2_to_yv12_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yuy2_to_yv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yuy2_to_yv12_i_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yuy2_to_nv16_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yuy2_to_nv16_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}

void convert_yc48_to_nv12_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_nv12_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_nv12_i_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_nv12_i_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_yv12_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_yv12_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_yv12_i_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_yv12_i_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_nv16_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_nv16_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_yuv444_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_yuv444_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
void convert_yc48_to_yuv444_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height, const int thread_id, const int thread_n) {
    return convert_yc48_to_yuv444_16bit_simd<FALSE>(frame, pixel_data, width, height, thread_id, thread_n);
}
