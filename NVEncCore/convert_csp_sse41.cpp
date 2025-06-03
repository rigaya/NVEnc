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
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   0
#define USE_AVX2  0

#include "convert_csp_simd.h"

#pragma warning (push)
#pragma warning (disable: 4100)


void convert_yuv444_16_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_simd<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_simd<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_simd<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop) {
    convert_yuv444_high_to_y410_simd<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yc48_to_p010_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yc48_to_p010_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yc48_to_p010_i_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yc48_to_p010_i_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yc48_to_yuv444_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yc48_to_yuv444_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yc48_to_yuv444_16bit_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yc48_to_yuv444_16bit_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_16bit_to_yc48_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_16bit_to_yc48_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, dst_uv_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
#pragma warning (pop)
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
