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

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include "rgy_tchar.h"
#include "rgy_simd.h"
#include "rgy_version.h"
#include "convert_csp.h"
#include "rgy_osdep.h"

void copy_nv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_p010_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_nv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_p010_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuy2_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuy2_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_i_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_i_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuy2_to_nv12_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_uv_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_uv_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_uv_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb24_to_rgb32_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24r_to_rgb32_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24r_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24r_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb32_to_rgb32_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32r_to_rgb32_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32r_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32r_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb24_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24r_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb24_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24r_to_rgb24_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_to_p010_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yv12_16_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_16_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_14_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_14_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_10_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_10_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_09_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_09_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yv12_16_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_16_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_14_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_14_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_12_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_10_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_10_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_09_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yv12_09_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

#if defined(_MSC_VER) || defined(__AVX2__)
#define FUNC_AVX2(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },
#else
#define FUNC_AVX2(from, to, uv_only, funcp, funci, simd)
#endif

void convert_yuv422_to_nv16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_09_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_10_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_12_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_14_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_16_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void copy_yuv444_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuv444_16_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_16_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_14_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_14_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_12_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_12_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_10_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_10_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_09_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_09_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuv444_to_yuv444_16_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuv444_16_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_16_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_14_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_14_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_12_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_12_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_10_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_10_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_09_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_09_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yc48_to_yuv444_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yc48_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_i_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_i_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_i_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_p010_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yc48_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_16bit_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_16bit_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_16bit_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yc48_to_yuv444_16bit_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_yuv444_16bit_to_yc48_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_16bit_to_yc48_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_16bit_to_yc48_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv444_16bit_to_yc48_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

//適当。
#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void convert_yuy2_to_nv12(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    void *dst = dst_array[0];
    const void *src = src_array[0];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcFrame = (uint8_t *)src;
    uint8_t *dstYFrame = (uint8_t *)dst;
    uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
    for (int y = y_range.start_dst; y < (y_range.start_dst + y_range.len); y += 2) {
        uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
        uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
        uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up) + crop_left;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
            dstY[0*dst_y_pitch_byte  + 0] = srcP[0*src_y_pitch_byte + 0];
            dstY[0*dst_y_pitch_byte  + 1] = srcP[0*src_y_pitch_byte + 2];
            dstY[1*dst_y_pitch_byte  + 0] = srcP[1*src_y_pitch_byte + 0];
            dstY[1*dst_y_pitch_byte  + 1] = srcP[1*src_y_pitch_byte + 2];
            dstC[0*dst_y_pitch_byte/2+ 0] =(srcP[0*src_y_pitch_byte + 1] + srcP[1*src_y_pitch_byte + 1] + 1)/2;
            dstC[0*dst_y_pitch_byte/2+ 1] =(srcP[0*src_y_pitch_byte + 3] + srcP[1*src_y_pitch_byte + 3] + 1)/2;
        }
    }
}

//これも適当。
void convert_yuy2_to_nv12_i(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    void *dst = dst_array[0];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const void *src = src_array[0];
    uint8_t *srcFrame = (uint8_t *)src;
    uint8_t *dstYFrame = (uint8_t *)dst;
    uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
    for (int y = y_range.start_dst; y < (y_range.start_dst + y_range.len); y += 4) {
        uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
        uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
        uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up) + crop_left;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
            dstY[0*dst_y_pitch_byte   + 0] = srcP[0*src_y_pitch_byte + 0];
            dstY[0*dst_y_pitch_byte   + 1] = srcP[0*src_y_pitch_byte + 2];
            dstY[1*dst_y_pitch_byte   + 0] = srcP[1*src_y_pitch_byte + 0];
            dstY[1*dst_y_pitch_byte   + 1] = srcP[1*src_y_pitch_byte + 2];
            dstY[2*dst_y_pitch_byte   + 0] = srcP[2*src_y_pitch_byte + 0];
            dstY[2*dst_y_pitch_byte   + 1] = srcP[2*src_y_pitch_byte + 2];
            dstY[3*dst_y_pitch_byte   + 0] = srcP[3*src_y_pitch_byte + 0];
            dstY[3*dst_y_pitch_byte   + 1] = srcP[3*src_y_pitch_byte + 2];
            dstC[0*dst_y_pitch_byte/2 + 0] =(srcP[0*src_y_pitch_byte + 1] * 3 + srcP[2*src_y_pitch_byte + 1] * 1 + 2)>>2;
            dstC[0*dst_y_pitch_byte/2 + 1] =(srcP[0*src_y_pitch_byte + 3] * 3 + srcP[2*src_y_pitch_byte + 3] * 1 + 2)>>2;
            dstC[1*dst_y_pitch_byte/2 + 0] =(srcP[1*src_y_pitch_byte + 1] * 1 + srcP[3*src_y_pitch_byte + 1] * 3 + 2)>>2;
            dstC[1*dst_y_pitch_byte/2 + 1] =(srcP[1*src_y_pitch_byte + 3] * 1 + srcP[3*src_y_pitch_byte + 3] * 3 + 2)>>2;
        }
    }
}

#if defined(__GNUC__)
//template展開部分で、実際には通らない箇所であっても反応してしまう
#pragma GCC diagnostic ignored "-Wshift-count-negative"
#endif

#define CHANGE_BIT_DEPTH_2(c0, c1, offset) \
    if (out_bit_depth > in_bit_depth + offset) { \
        c0 <<= (out_bit_depth - in_bit_depth - offset); \
        c1 <<= (out_bit_depth - in_bit_depth - offset); \
    } else if (out_bit_depth < in_bit_depth + offset) { \
        c0 >>= (in_bit_depth + offset - out_bit_depth); \
        c1 >>= (in_bit_depth + offset - out_bit_depth); \
    }

#define CHANGE_BIT_DEPTH_4(c0, c1, c2, c3, offset) \
    if (out_bit_depth > in_bit_depth + offset) { \
        c0 <<= (out_bit_depth - in_bit_depth - offset); \
        c1 <<= (out_bit_depth - in_bit_depth - offset); \
        c2 <<= (out_bit_depth - in_bit_depth - offset); \
        c3 <<= (out_bit_depth - in_bit_depth - offset); \
    } else if (out_bit_depth < in_bit_depth + offset) { \
        c0 >>= (in_bit_depth + offset - out_bit_depth); \
        c1 >>= (in_bit_depth + offset - out_bit_depth); \
        c2 >>= (in_bit_depth + offset - out_bit_depth); \
        c3 >>= (in_bit_depth + offset - out_bit_depth); \
    }

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void __forceinline convert_yuv444_to_nv12_p_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert((sizeof(Tin)  == 1 && in_bit_depth  == 8) || (sizeof(Tin)  == 2 && 8 < in_bit_depth  && in_bit_depth  <= 16), "invalid input bit depth.");
    static_assert((sizeof(Tout) == 1 && out_bit_depth == 8) || (sizeof(Tout) == 2 && 8 < out_bit_depth && out_bit_depth <= 16), "invalid output bit depth.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    //Y成分のコピー
    if (!uv_only) {
        Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        Tout *dstLine = (Tout *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                for (int x = 0; x < y_width; x++) {
                    if (out_bit_depth > in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) << std::max(out_bit_depth - in_bit_depth, 0));
                    } else if (out_bit_depth < in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) >> std::max(in_bit_depth - out_bit_depth, 0));
                    } else {
                        dstLine[x] = (Tout)srcYLine[x];
                    }
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcULine = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tin *srcVLine = (Tin *)src[2] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tout *dstLine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    for (int y = 0; y < y_range.len; y += 2, srcULine += src_uv_pitch * 2, srcVLine += src_uv_pitch * 2, dstLine += dst_y_pitch) {
        Tout *dstC = dstLine;
        Tin *srcU = srcULine;
        Tin *srcV = srcVLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcU += 2, srcV += 2) {
            int cy0u = srcU[0*src_uv_pitch + 0];
            int cy0v = srcV[0*src_uv_pitch + 0];
            int cy1u = srcU[1*src_uv_pitch + 0];
            int cy1v = srcV[1*src_uv_pitch + 0];

            int cu = cy0u + cy1u + 1;
            int cv = cy0v + cy1v + 1;
            CHANGE_BIT_DEPTH_2(cu, cv, 1);

            dstC[0] = (Tout)cu;
            dstC[1] = (Tout)cv;
        }
    }
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void __forceinline convert_yuv444_to_nv12_i_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert((sizeof(Tin)  == 1 && in_bit_depth  == 8) || (sizeof(Tin)  == 2 && 8 < in_bit_depth  && in_bit_depth  <= 16), "invalid input bit depth.");
    static_assert((sizeof(Tout) == 1 && out_bit_depth == 8) || (sizeof(Tout) == 2 && 8 < out_bit_depth && out_bit_depth <= 16), "invalid output bit depth.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    //Y成分のコピー
    if (!uv_only) {
        Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        Tout *dstLine = (Tout *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                for (int x = 0; x < y_width; x++) {
                    if (out_bit_depth > in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) << std::max(out_bit_depth - in_bit_depth, 0));
                    } else if (out_bit_depth < in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) >> std::max(in_bit_depth - out_bit_depth, 0));
                    } else {
                        dstLine[x] = (Tout)srcYLine[x];
                    }
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcULine = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tin *srcVLine = (Tin *)src[2] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tout *dstLine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    const int uv_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < y_range.len; y += 4, srcULine += src_uv_pitch * 4, srcVLine += src_uv_pitch * 4, dstLine += dst_y_pitch * 2) {
        Tout *dstC = dstLine;
        Tin *srcU = srcULine;
        Tin *srcV = srcVLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcU += 2, srcV += 2) {
            int cy0u = srcU[0*src_uv_pitch + 0];
            int cy0v = srcV[0*src_uv_pitch + 0];
            int cy1u = srcU[1*src_uv_pitch + 0];
            int cy1v = srcV[1*src_uv_pitch + 0];
            int cy2u = srcU[2*src_uv_pitch + 0];
            int cy2v = srcV[2*src_uv_pitch + 0];
            int cy3u = srcU[3*src_uv_pitch + 0];
            int cy3v = srcV[3*src_uv_pitch + 0];

            int cu_y0 = cy0u * 3 + cy2u * 1 + 2;
            int cu_y1 = cy1u * 1 + cy3u * 3 + 2;
            int cv_y0 = cy0v * 3 + cy2v * 1 + 2;
            int cv_y1 = cy1v * 1 + cy3v * 3 + 2;
            CHANGE_BIT_DEPTH_4(cu_y0, cu_y1, cv_y0, cv_y1, 2);

            dstC[0*dst_y_pitch + 0] = (Tout)cu_y0;
            dstC[0*dst_y_pitch + 1] = (Tout)cv_y0;
            dstC[1*dst_y_pitch + 0] = (Tout)cu_y1;
            dstC[1*dst_y_pitch + 1] = (Tout)cv_y1;
        }
    }
}

static void convert_yuv444_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_16_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_16_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_14_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_14_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_12_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_12_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_10_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_10_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_09_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 9, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_09_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 9, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_16_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_16_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_14_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_14_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_12_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_12_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_10_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_10_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_09_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_p_c<uint16_t, 9, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_09_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_nv12_i_c<uint16_t, 9, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv422_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    //Y成分のコピー
    if (true) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    for (int ic = 1; ic < 3; ic++) {
        uint8_t *srcCLine = (uint8_t *)src[ic] + src_uv_pitch_byte * y_range.start_src + (crop_left >> 1);
        uint8_t *dstLine = (uint8_t *)dst[ic] + dst_y_pitch_byte * y_range.start_dst;
        for (int y = 0; y < y_range.len; y++, srcCLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
            uint8_t *dstC = dstLine;
            uint8_t *srcP = srcCLine;
            const int x_fin = width - crop_right - crop_left;
            for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                int cxplus = (x + 2 < x_fin);
                int cy1x0 = srcP[0*src_uv_pitch_byte + 0];
                int cy1x1 = srcP[0*src_uv_pitch_byte + cxplus];
                dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
            }
        }
    }
}

static void convert_yuy2_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * 2;
    uint8_t *dstYLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstULine = (uint8_t *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstVLine = (uint8_t *)dst[2] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcLine += src_y_pitch_byte, dstYLine += dst_y_pitch_byte, dstULine += dst_y_pitch_byte, dstVLine += dst_y_pitch_byte) {
        uint8_t *srcP = srcLine;
        uint8_t *dstY = dstYLine;
        uint8_t *dstU = dstULine;
        uint8_t *dstV = dstVLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, srcP += 4, dstY += 2, dstU += 2, dstV += 2) {
            int cxplus = (x + 2 < x_fin) ? 4 : 0;
            dstY[0] = srcP[0];
            dstY[1] = srcP[2];

            int ux0 = srcP[1];
            int vx0 = srcP[3];
            int ux2 = srcP[1+cxplus];
            int vx2 = srcP[3+cxplus];
            dstU[0] = (uint8_t)ux0;
            dstU[1] = (uint8_t)((ux0 + ux2 + 1) >> 1);
            dstV[0] = (uint8_t)vx0;
            dstV[1] = (uint8_t)((vx2 + vx2 + 1) >> 1);
        }
    }
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void __forceinline convert_yv12_p_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert((sizeof(Tin)  == 1 && in_bit_depth  == 8) || (sizeof(Tin)  == 2 && 8 < in_bit_depth  && in_bit_depth  <= 16), "invalid input bit depth.");
    static_assert((sizeof(Tout) == 1 && out_bit_depth == 8) || (sizeof(Tout) == 2 && 8 < out_bit_depth && out_bit_depth <= 16), "invalid output bit depth.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    //Y成分のコピー
    if (!uv_only) {
        Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        Tout *dstLine = (Tout *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                for (int x = 0; x < y_width; x++) {
                    if (out_bit_depth > in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) << std::max(out_bit_depth - in_bit_depth, 0));
                    } else if (out_bit_depth < in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) >> std::max(in_bit_depth - out_bit_depth, 0));
                    } else {
                        dstLine[x] = (Tout)srcYLine[x];
                    }
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    for (int ic = 1; ic < 3; ic++) {
        Tin *srcCLine = (Tin *)src[ic] + (((src_uv_pitch * y_range.start_src) + crop_left) >> 1);
        Tout *dstLine = (Tout *)dst[ic] + dst_y_pitch * y_range.start_dst;
        for (int y = 0; y < y_range.len; y += 2, srcCLine += src_uv_pitch, dstLine += dst_y_pitch * 2) {
            Tout *dstC = dstLine;
            Tin *srcP = srcCLine;
            const int x_fin = width - crop_right - crop_left;
            if (y == 0) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy4x0 = srcP[ 1*src_uv_pitch + 0];
                    int cy0x1 = srcP[ 0*src_uv_pitch + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch + cxplus];
                    int cy4x1 = srcP[ 1*src_uv_pitch + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2);
                    CHANGE_BIT_DEPTH_4(cy1x0, cy3x0, cy1x1, cy3x1, 2);

                    dstC[0*dst_y_pitch   + 0] = (Tout)cy1x0;
                    dstC[0*dst_y_pitch   + 1] = (Tout)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch   + 0] = (Tout)cy3x0;
                    dstC[1*dst_y_pitch   + 1] = (Tout)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else if (y >= height-2) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[-1*src_uv_pitch + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy4x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy0x1 = srcP[-1*src_uv_pitch + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch + cxplus];
                    int cy4x1 = srcP[ 0*src_uv_pitch + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2);
                    CHANGE_BIT_DEPTH_4(cy1x0, cy3x0, cy1x1, cy3x1, 2);

                    dstC[0*dst_y_pitch   + 0] = (Tout)cy1x0;
                    dstC[0*dst_y_pitch   + 1] = (Tout)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch   + 0] = (Tout)cy3x0;
                    dstC[1*dst_y_pitch   + 1] = (Tout)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[-1*src_uv_pitch + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy4x0 = srcP[ 1*src_uv_pitch + 0];
                    int cy0x1 = srcP[-1*src_uv_pitch + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch + cxplus];
                    int cy4x1 = srcP[ 1*src_uv_pitch + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2);
                    CHANGE_BIT_DEPTH_4(cy1x0, cy3x0, cy1x1, cy3x1, 2);

                    dstC[0*dst_y_pitch   + 0] = (Tout)cy1x0;
                    dstC[0*dst_y_pitch   + 1] = (Tout)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch   + 0] = (Tout)cy3x0;
                    dstC[1*dst_y_pitch   + 1] = (Tout)((cy3x0 + cy3x1 + 1) >> 1);
                }
            }
        }
    }
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void __forceinline convert_yv12_i_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert((sizeof(Tin)  == 1 && in_bit_depth  == 8) || (sizeof(Tin)  == 2 && 8 < in_bit_depth  && in_bit_depth  <= 16), "invalid input bit depth.");
    static_assert((sizeof(Tout) == 1 && out_bit_depth == 8) || (sizeof(Tout) == 2 && 8 < out_bit_depth && out_bit_depth <= 16), "invalid output bit depth.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    //Y成分のコピー
    if (!uv_only) {
        Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        Tout *dstLine = (Tout *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == out_bit_depth) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                for (int x = 0; x < y_width; x++) {
                    if (out_bit_depth > in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) << std::max(out_bit_depth - in_bit_depth, 0));
                    } else if (out_bit_depth < in_bit_depth) {
                        dstLine[x] = (Tout)((int)(srcYLine[x]) >> std::max(in_bit_depth - out_bit_depth, 0));
                    } else {
                        dstLine[x] = (Tout)srcYLine[x];
                    }
                }
            }
        }
    }
    //UV成分のコピー
    const int uv_fin = height - crop_bottom;
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    for (int ic = 1; ic < 3; ic++) {
        Tin *srcCLine = (Tin *)src[ic] + (((src_uv_pitch * y_range.start_src) + crop_left) >> 1);
        Tout *dstLine = (Tout *)dst[ic] + dst_y_pitch * y_range.start_dst;
        for (int y = 0; y < y_range.len; y += 4, srcCLine += src_uv_pitch * 2, dstLine += dst_y_pitch * 4) {
            Tout *dstC = dstLine;
            Tin *srcP = srcCLine;
            const int x_fin = width - crop_right - crop_left;

            int y_m2 = (y >= 4) ? -2 : 0;
            int y_m1 = (y >= 2) ? -1 : 1;
            int y_p1 = (y < uv_fin - 2) ? 1 : -1;
            int y_p2 = (y < uv_fin - 4) ? 2 :  0;
            int y_p3 = (y < uv_fin - 6) ? 3 : ((y < uv_fin - 2) ? 1 : -1);

            int sy0x0 = srcP[y_m2*src_uv_pitch + 0];
            int sy1x0 = srcP[y_m1*src_uv_pitch + 0];
            int sy2x0 = srcP[   0*src_uv_pitch + 0];
            int sy3x0 = srcP[y_p1*src_uv_pitch + 0];
            int sy4x0 = srcP[y_p2*src_uv_pitch + 0];
            int sy5x0 = srcP[y_p3*src_uv_pitch + 0];

            int cy0x0 = (sy0x0 * 1 + sy2x0 * 7 + 4);
            int cy1x0 = (sy1x0 * 3 + sy3x0 * 5 + 4);
            int cy2x0 = (sy2x0 * 5 + sy4x0 * 3 + 4);
            int cy3x0 = (sy3x0 * 7 + sy5x0 * 1 + 4);
            CHANGE_BIT_DEPTH_4(cy0x0, cy1x0, cy2x0, cy3x0, 3);

            for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                int cxplus = (x + 2 < x_fin);
                int sy0x1 = srcP[y_m2*src_uv_pitch + cxplus];
                int sy1x1 = srcP[y_m1*src_uv_pitch + cxplus];
                int sy2x1 = srcP[   0*src_uv_pitch + cxplus];
                int sy3x1 = srcP[y_p1*src_uv_pitch + cxplus];
                int sy4x1 = srcP[y_p2*src_uv_pitch + cxplus];
                int sy5x1 = srcP[y_p3*src_uv_pitch + cxplus];

                int cy0x1 = (sy0x1 * 1 + sy2x1 * 7 + 4);
                int cy1x1 = (sy1x1 * 3 + sy3x1 * 5 + 4);
                int cy2x1 = (sy2x1 * 5 + sy4x1 * 3 + 4);
                int cy3x1 = (sy3x1 * 7 + sy5x1 * 1 + 4);
                CHANGE_BIT_DEPTH_4(cy0x1, cy1x1, cy2x1, cy3x1, 3);

                dstC[0*dst_y_pitch   + 0] = (Tout)cy0x0;
                dstC[0*dst_y_pitch   + 1] = (Tout)((cy0x0 + cy0x1 + 1) >> 1);
                dstC[1*dst_y_pitch   + 0] = (Tout)cy1x0;
                dstC[1*dst_y_pitch   + 1] = (Tout)((cy1x0 + cy1x1 + 1) >> 1);
                dstC[2*dst_y_pitch   + 0] = (Tout)cy2x0;
                dstC[2*dst_y_pitch   + 1] = (Tout)((cy2x0 + cy2x1 + 1) >> 1);
                dstC[3*dst_y_pitch   + 0] = (Tout)cy3x0;
                dstC[3*dst_y_pitch   + 1] = (Tout)((cy3x0 + cy3x1 + 1) >> 1);

                cy0x0 = cy0x1;
                cy1x0 = cy1x1;
                cy2x0 = cy2x1;
                cy3x0 = cy3x1;
            }
        }
    }
}

static void convert_yv12_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_16_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_16_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_14_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_14_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_12_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_12_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_10_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_10_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_09_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 9, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_09_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 9, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_16_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_16_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_14_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_14_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_12_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_12_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_10_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_10_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_09_p_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_p_to_yuv444_c<uint16_t, 9, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yv12_09_i_to_yuv444_16bit(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_i_to_yuv444_c<uint16_t, 9, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<bool uv_only>
static void convert_yv12_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            uint16_t *dst_ptr = (uint16_t *)dstLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint16_t)((((uint32_t)srcYLine[x]) << 8) + (2 << 6));
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine  = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = width - crop_right;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = (uint16_t *)dstLine;
        for (int x = crop_left; x < x_fin; x += 2, src_u_ptr++, src_v_ptr++, dst_ptr += 2) {
            dst_ptr[0] = (uint16_t)((((uint32_t)src_u_ptr[0]) << 8) + (2<<6));
            dst_ptr[1] = (uint16_t)((((uint32_t)src_v_ptr[0]) << 8) + (2<<6));
        }
    }
}

static void convert_yv12_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_to_p010_c<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#pragma warning (pop)

#if defined(_MSC_VER) || defined(__AVX__)
#define FUNC_AVX(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },
#else
#define FUNC_AVX(from, to, uv_only, funcp, funci, simd)
#endif
#define FUNC_SSE(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },


static const ConvertCSP funcList[] = {
    FUNC_AVX2( RGY_CSP_NV12,      RGY_CSP_NV12,      false,  copy_nv12_to_nv12_avx2,              copy_nv12_to_nv12_avx2,              AVX2|AVX)
    FUNC_SSE(  RGY_CSP_NV12,      RGY_CSP_NV12,      false,  copy_nv12_to_nv12_sse2,              copy_nv12_to_nv12_sse2,              SSE2 )
    FUNC_AVX2( RGY_CSP_P010,      RGY_CSP_P010,      false,  copy_p010_to_p010_avx2,              copy_p010_to_p010_avx2,              AVX2|AVX)
    FUNC_SSE(  RGY_CSP_P010,      RGY_CSP_P010,      false,  copy_p010_to_p010_sse2,              copy_p010_to_p010_sse2,              SSE2 )
    FUNC_AVX2( RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_avx2,           convert_yuy2_to_nv12_i_avx2,         AVX2|AVX)
    FUNC_AVX(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_avx,            convert_yuy2_to_nv12_i_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_ssse3,        SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_sse2,         SSE2 )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12,                convert_yuy2_to_nv12,                NONE )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_YUV444,    false,  convert_yuy2_to_yuv444,              convert_yuy2_to_yuv444,              NONE )
#if FOR_AUO
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_avx,          convert_yc48_to_yuv444_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_sse41,        convert_yc48_to_yuv444_sse41,        SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_sse2,         convert_yc48_to_yuv444_sse2,         SSE2 )
    FUNC_AVX2( RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_avx2,         AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_avx,            convert_yc48_to_p010_i_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse41,          convert_yc48_to_p010_i_sse41,        SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_ssse3,          convert_yc48_to_p010_i_ssse3,        SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_sse2,         SSE2 )
    FUNC_AVX2( RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_avx2,   convert_yc48_to_yuv444_16bit_avx2,   AVX2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_avx,    convert_yc48_to_yuv444_16bit_avx,    AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_sse41,  convert_yc48_to_yuv444_16bit_sse41,  SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_ssse3,  convert_yc48_to_yuv444_16bit_ssse3,  SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_sse2,   convert_yc48_to_yuv444_16bit_sse2,   SSE2 )
    FUNC_AVX2(RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_avx2,   convert_yuv444_16bit_to_yc48_avx2,   AVX2 )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_avx,    convert_yuv444_16bit_to_yc48_avx,    AVX )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_sse41,  convert_yuv444_16bit_to_yc48_sse41,  SSE41|SSSE3|SSE2 )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_sse2,   convert_yuv444_16bit_to_yc48_sse2,   SSE2 )
#endif
#if ENABLE_AVSW_READER || ENABLE_AVI_READER || ENABLE_AVISYNTH_READER || ENABLE_VAPOURSYNTH_READER || ENABLE_AVI_READER || ENABLE_RAW_READER
    FUNC_AVX2( RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_avx2,     convert_yv12_to_nv12_avx2,     AVX2|AVX)
    FUNC_AVX(  RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_avx,      convert_yv12_to_nv12_avx,      AVX )
    FUNC_SSE(  RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_sse2,     convert_yv12_to_nv12_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_YV12, RGY_CSP_YUV444, false, convert_yv12_p_to_yuv444,    convert_yv12_i_to_yuv444,      NONE )
    FUNC_AVX2( RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_avx2,  convert_uv_yv12_to_nv12_avx2,  AVX2|AVX )
    FUNC_AVX(  RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_avx,   convert_uv_yv12_to_nv12_avx,   AVX )
    FUNC_SSE(  RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_sse2,  convert_uv_yv12_to_nv12_sse2,  SSE2 )

    FUNC_AVX2( RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_avx2,      convert_rgb24_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX2( RGY_CSP_RGB24R, RGY_CSP_RGB32, false, convert_rgb24r_to_rgb32_avx2,     convert_rgb24r_to_rgb32_avx2,     AVX2|AVX )
    FUNC_AVX(  RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_avx,       convert_rgb24_to_rgb32_avx,       AVX )
    FUNC_AVX(  RGY_CSP_RGB24R, RGY_CSP_RGB32, false, convert_rgb24r_to_rgb32_avx,      convert_rgb24r_to_rgb32_avx,      AVX )
    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_ssse3,     convert_rgb24_to_rgb32_ssse3,     SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_RGB24R, RGY_CSP_RGB32, false, convert_rgb24r_to_rgb32_ssse3,    convert_rgb24r_to_rgb32_ssse3,    SSSE3|SSE2 )
    FUNC_AVX2( RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_avx2,      convert_rgb32_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX2( RGY_CSP_RGB32R, RGY_CSP_RGB32, false, convert_rgb32r_to_rgb32_avx2,     convert_rgb32r_to_rgb32_avx2,     AVX2|AVX )
    FUNC_AVX(  RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_avx,       convert_rgb32_to_rgb32_avx,       AVX )
    FUNC_AVX(  RGY_CSP_RGB32R, RGY_CSP_RGB32, false, convert_rgb32r_to_rgb32_avx,      convert_rgb32r_to_rgb32_avx,      AVX )
    FUNC_SSE(  RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_sse2,      convert_rgb32_to_rgb32_sse2,      SSE2 )
    FUNC_SSE(  RGY_CSP_RGB32R, RGY_CSP_RGB32, false, convert_rgb32r_to_rgb32_sse2,     convert_rgb32r_to_rgb32_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB24, false, convert_rgb24_to_rgb24_avx2,      convert_rgb24_to_rgb24_avx2,      AVX2|AVX)
    FUNC_SSE(  RGY_CSP_RGB24R, RGY_CSP_RGB24, false, convert_rgb24r_to_rgb24_avx2,     convert_rgb24r_to_rgb24_avx2,     AVX2|AVX)
    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB24, false, convert_rgb24_to_rgb24_sse2,      convert_rgb24_to_rgb24_sse2,      SSE2 )
    FUNC_SSE(  RGY_CSP_RGB24R, RGY_CSP_RGB24, false, convert_rgb24r_to_rgb24_sse2,     convert_rgb24r_to_rgb24_sse2,     SSE2 )

    FUNC_AVX2( RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_avx2,           convert_yv12_to_p010_avx2,    AVX2|AVX )
    FUNC_AVX(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_avx,            convert_yv12_to_p010_avx,     AVX )
    FUNC_SSE(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_sse2,           convert_yv12_to_p010_sse2,    SSE2 )
    FUNC_SSE(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010,                convert_yv12_to_p010,         NONE )
    FUNC_SSE(  RGY_CSP_YV12,      RGY_CSP_YUV444_16, false, convert_yv12_p_to_yuv444_16bit,      convert_yv12_i_to_yuv444_16bit, NONE )
    FUNC_AVX2( RGY_CSP_YV12_16,   RGY_CSP_NV12,      false, convert_yv12_16_to_nv12_avx2,        convert_yv12_16_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_16,   RGY_CSP_NV12,      false, convert_yv12_16_to_nv12_sse2,        convert_yv12_16_to_nv12_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_14,   RGY_CSP_NV12,      false, convert_yv12_14_to_nv12_avx2,        convert_yv12_14_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_NV12,      false, convert_yv12_14_to_nv12_sse2,        convert_yv12_14_to_nv12_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_12,   RGY_CSP_NV12,      false, convert_yv12_12_to_nv12_avx2,        convert_yv12_12_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_NV12,      false, convert_yv12_12_to_nv12_sse2,        convert_yv12_12_to_nv12_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_10_to_nv12_avx2,        convert_yv12_10_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_10_to_nv12_sse2,        convert_yv12_10_to_nv12_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_09,   RGY_CSP_NV12,      false, convert_yv12_09_to_nv12_avx2,        convert_yv12_09_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_NV12,      false, convert_yv12_09_to_nv12_sse2,        convert_yv12_09_to_nv12_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_16,   RGY_CSP_P010,      false, convert_yv12_16_to_p010_avx2,        convert_yv12_16_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_16,   RGY_CSP_P010,      false, convert_yv12_16_to_p010_sse2,        convert_yv12_16_to_p010_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_14,   RGY_CSP_P010,      false, convert_yv12_14_to_p010_avx2,        convert_yv12_14_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_P010,      false, convert_yv12_14_to_p010_sse2,        convert_yv12_14_to_p010_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_12,   RGY_CSP_P010,      false, convert_yv12_12_to_p010_avx2,        convert_yv12_12_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_P010,      false, convert_yv12_12_to_p010_sse2,        convert_yv12_12_to_p010_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_10,   RGY_CSP_P010,      false, convert_yv12_10_to_p010_avx2,        convert_yv12_10_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_P010,      false, convert_yv12_10_to_p010_sse2,        convert_yv12_10_to_p010_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_09,   RGY_CSP_P010,      false, convert_yv12_09_to_p010_avx2,        convert_yv12_09_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_P010,      false, convert_yv12_09_to_p010_sse2,        convert_yv12_09_to_p010_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YV12_16,   RGY_CSP_YUV444,    false, convert_yv12_16_p_to_yuv444,         convert_yv12_16_i_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_YUV444,    false, convert_yv12_14_p_to_yuv444,         convert_yv12_14_i_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_YUV444,    false, convert_yv12_12_p_to_yuv444,         convert_yv12_12_i_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_YUV444,    false, convert_yv12_10_p_to_yuv444,         convert_yv12_10_i_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_YUV444,    false, convert_yv12_09_p_to_yuv444,         convert_yv12_09_i_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YV12_16,   RGY_CSP_YUV444_16, false, convert_yv12_16_p_to_yuv444_16bit,   convert_yv12_16_i_to_yuv444_16bit, NONE )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_YUV444_16, false, convert_yv12_14_p_to_yuv444_16bit,   convert_yv12_14_i_to_yuv444_16bit, NONE )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_YUV444_16, false, convert_yv12_12_p_to_yuv444_16bit,   convert_yv12_12_i_to_yuv444_16bit, NONE )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_YUV444_16, false, convert_yv12_10_p_to_yuv444_16bit,   convert_yv12_10_i_to_yuv444_16bit, NONE )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_YUV444_16, false, convert_yv12_09_p_to_yuv444_16bit,   convert_yv12_09_i_to_yuv444_16bit, NONE )
    FUNC_SSE(  RGY_CSP_YUV422,    RGY_CSP_YUV444,    false, convert_yuv422_to_yuv444,            convert_yuv422_to_yuv444,  NONE )
    FUNC_SSE(  RGY_CSP_YUV422,    RGY_CSP_NV16,      false, convert_yuv422_to_nv16_sse2,         convert_yuv422_to_nv16_sse2,    SSE2)
    FUNC_SSE(  RGY_CSP_YUV422,    RGY_CSP_P210,      false, convert_yuv422_to_p210_sse2,         convert_yuv422_to_p210_sse2,    SSE2)
    FUNC_SSE(  RGY_CSP_YUV422_16, RGY_CSP_P210,      false, convert_yuv422_16_to_p210_sse2,      convert_yuv422_16_to_p210_sse2, SSE2)
    FUNC_SSE(  RGY_CSP_YUV422_14, RGY_CSP_P210,      false, convert_yuv422_14_to_p210_sse2,      convert_yuv422_14_to_p210_sse2, SSE2)
    FUNC_SSE(  RGY_CSP_YUV422_12, RGY_CSP_P210,      false, convert_yuv422_12_to_p210_sse2,      convert_yuv422_12_to_p210_sse2, SSE2)
    FUNC_SSE(  RGY_CSP_YUV422_10, RGY_CSP_P210,      false, convert_yuv422_10_to_p210_sse2,      convert_yuv422_10_to_p210_sse2, SSE2)
    FUNC_SSE(  RGY_CSP_YUV422_09, RGY_CSP_P210,      false, convert_yuv422_09_to_p210_sse2,      convert_yuv422_09_to_p210_sse2, SSE2)
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_NV12,      false, convert_yuv444_to_nv12_p,            convert_yuv444_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_P010,      false, convert_yuv444_to_p010_p,            convert_yuv444_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_YUV444,    false, copy_yuv444_to_yuv444_avx2,          copy_yuv444_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_YUV444,    false, copy_yuv444_to_yuv444_sse2,          copy_yuv444_to_yuv444_sse2, SSE2 )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_NV12,      false, convert_yuv444_16_to_nv12_p,         convert_yuv444_16_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_NV12,      false, convert_yuv444_14_to_nv12_p,         convert_yuv444_14_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_NV12,      false, convert_yuv444_12_to_nv12_p,         convert_yuv444_12_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_NV12,      false, convert_yuv444_10_to_nv12_p,         convert_yuv444_10_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_NV12,      false, convert_yuv444_09_to_nv12_p,         convert_yuv444_09_to_nv12_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_P010,      false, convert_yuv444_16_to_p010_p,         convert_yuv444_16_to_p010_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_P010,      false, convert_yuv444_14_to_p010_p,         convert_yuv444_14_to_p010_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_P010,      false, convert_yuv444_12_to_p010_p,         convert_yuv444_12_to_p010_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_P010,      false, convert_yuv444_10_to_p010_p,         convert_yuv444_10_to_p010_i, NONE )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_P010,      false, convert_yuv444_09_to_p010_p,         convert_yuv444_09_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_YUV444_16, false, convert_yuv444_16_to_yuv444_16_avx2, convert_yuv444_16_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_YUV444_16, false, convert_yuv444_16_to_yuv444_16_sse2, convert_yuv444_16_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_YUV444_16, false, convert_yuv444_14_to_yuv444_16_avx2, convert_yuv444_14_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_YUV444_16, false, convert_yuv444_14_to_yuv444_16_sse2, convert_yuv444_14_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_YUV444_16, false, convert_yuv444_12_to_yuv444_16_avx2, convert_yuv444_12_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_YUV444_16, false, convert_yuv444_12_to_yuv444_16_sse2, convert_yuv444_12_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_YUV444_16, false, convert_yuv444_10_to_yuv444_16_avx2, convert_yuv444_10_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_YUV444_16, false, convert_yuv444_10_to_yuv444_16_sse2, convert_yuv444_10_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_YUV444_16, false, convert_yuv444_09_to_yuv444_16_avx2, convert_yuv444_09_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_YUV444_16, false, convert_yuv444_09_to_yuv444_16_sse2, convert_yuv444_09_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_YUV444_16, false, convert_yuv444_to_yuv444_16_avx2,    convert_yuv444_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_YUV444_16, false, convert_yuv444_to_yuv444_16_sse2,    convert_yuv444_to_yuv444_16_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_YUV444,    false, convert_yuv444_16_to_yuv444_avx2,    convert_yuv444_16_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_YUV444,    false, convert_yuv444_16_to_yuv444_sse2,    convert_yuv444_16_to_yuv444_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_YUV444,    false, convert_yuv444_14_to_yuv444_avx2,    convert_yuv444_14_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_YUV444,    false, convert_yuv444_14_to_yuv444_sse2,    convert_yuv444_14_to_yuv444_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_YUV444,    false, convert_yuv444_12_to_yuv444_avx2,    convert_yuv444_12_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_YUV444,    false, convert_yuv444_12_to_yuv444_sse2,    convert_yuv444_12_to_yuv444_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_YUV444,    false, convert_yuv444_10_to_yuv444_avx2,    convert_yuv444_10_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_YUV444,    false, convert_yuv444_10_to_yuv444_sse2,    convert_yuv444_10_to_yuv444_sse2, SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_YUV444,    false, convert_yuv444_09_to_yuv444_avx2,    convert_yuv444_09_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_YUV444,    false, convert_yuv444_09_to_yuv444_sse2,    convert_yuv444_09_to_yuv444_sse2, SSE2 )
#endif
};

const ConvertCSP *get_convert_csp_func(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, uint32_t simd) {
    uint32_t availableSIMD = get_availableSIMD() & simd;
    const ConvertCSP *convert = nullptr;
    for (int i = 0; i < _countof(funcList); i++) {
        if (csp_from != funcList[i].csp_from)
            continue;

        if (csp_to != funcList[i].csp_to)
            continue;

        if (uv_only != funcList[i].uv_only)
            continue;

        if (funcList[i].simd != (availableSIMD & funcList[i].simd))
            continue;

        convert = &funcList[i];
        break;
    }
    return convert;
}

const TCHAR *get_simd_str(unsigned int simd) {
    static std::vector<std::pair<uint32_t, const TCHAR*>> simd_str_list = {
        { AVX2,  _T("AVX2")   },
        { AVX,   _T("AVX")    },
        { SSE42, _T("SSE4.2") },
        { SSE41, _T("SSE4.1") },
        { SSSE3, _T("SSSE3")  },
        { SSE2,  _T("SSE2")   },
    };
    for (auto simd_str : simd_str_list) {
        if (simd_str.first & simd)
            return simd_str.second;
    }
    return _T("-");
}
