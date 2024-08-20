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
#include "rgy_frame_info.h"
#include "rgy_osdep.h"

void copy_nv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_p010_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_nv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_p010_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void copy_p010_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_nv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

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

void convert_rgb24_to_rgb_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24_to_rgb_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_rgb_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32r_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgba_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32_to_rgba_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32r_to_rgba_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_argb32_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_abgr32_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_argb32_to_rgba_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_abgr32_to_rgba_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_gbr_to_rgb24_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_gbr_to_rgb32_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_bgr24_to_rgb24_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32_to_rgb32_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb24_to_rgb32_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_bgr32_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_bgr32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_bgr32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb32_to_rgb32_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32r_to_bgr32_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgb32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32r_to_bgr32_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb32_to_rgb32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr32r_to_bgr32_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void convert_rgb24_to_rgb24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_bgr24_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_rgb24_to_rgb24_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_bgr24r_to_bgr24_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void copy_rgb_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_gbr_to_rgb_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

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

void convert_yuv422_to_nv16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_09_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_10_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_12_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_14_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void convert_yuv422_16_to_p210_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

void copy_yuv444_to_ayuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_to_ayuv444_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_09_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_09_to_ayuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_10_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_10_to_ayuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_12_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_12_to_ayuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_14_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_14_to_ayuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);
void copy_yuv444_16_to_ayuv444_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void copy_yuv444_16_to_ayuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop);

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

void convert_yuv444_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_to_y410_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_10_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_10_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_10_to_y410_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_12_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_12_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_12_to_y410_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_14_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_14_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_14_to_y410_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_16_to_y410_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_16_to_y410_sse41(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_16_to_y410_sse2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);

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
template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth>
void copy_plane(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(Tout);
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    Tout *dstLine = (Tout *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
            memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
        } else {
            for (int x = 0; x < y_width; x++) {
                dstLine[x] = (Tout)conv_bit_depth_<out_bit_depth, in_bit_depth, 0>(srcYLine[x]);
            }
        }
    }
}

void copy_plane_u8_to_u8(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint8_t, 8, uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u8_to_u10(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint8_t, 8, uint16_t, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u8_to_u12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint8_t, 8, uint16_t, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u8_to_u16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint8_t, 8, uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u10_to_u8(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 10, uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u10_to_u10(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 10, uint16_t, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u10_to_u12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 10, uint16_t, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u10_to_u16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 10, uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u12_to_u8(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 12, uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u12_to_u10(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 12, uint16_t, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u12_to_u12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 12, uint16_t, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u12_to_u16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 12, uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u16_to_u8(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 16, uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u16_to_u10(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 16, uint16_t, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u16_to_u12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 16, uint16_t, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
void copy_plane_u16_to_u16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_plane<uint16_t, 16, uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<bool highbit_depth>
void copy_nv12_to_nv12_c_internal(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int pixel_size = highbit_depth ? 2 : 1;
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width * pixel_size);
        }
    }
}
void copy_nv12_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12_to_nv12_c_internal<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_p010_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12_to_nv12_c_internal<true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth>
void copy_nv12p010_to_nv12p010_c_internal(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    for (int i = 0; i < 2; i++) {
        const auto y_range = thread_y_range(crop_up >> i, (height - crop_bottom) >> i, thread_id, thread_n);
        const uint8_t *srcYLine = ((const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left * sizeof(uint16_t));
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            const int x_fin = width - crop_right - crop_left;
            const Tin *ptrSrc = (const Tin *)srcYLine;
            Tout *ptrDst = (Tout *)dstLine;
            for (int x = 0; x < x_fin; x++) {
                ptrDst[x] = (Tout)conv_bit_depth_<out_bit_depth, in_bit_depth, 0>(ptrSrc[x]);
            }
        }
    }
}

void copy_p010_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12p010_to_nv12p010_c_internal<uint8_t, 8, uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_nv12_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_nv12p010_to_nv12p010_c_internal<uint16_t, 16, uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

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

void convert_yv12_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2*x+0] = src_u_ptr[x];
            dst_ptr[2*x+1] = src_v_ptr[x];
        }
    }
}


template<int in_bit_depth>
static void convert_yv12_high_to_nv12_c_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    //Y成分のコピー
    {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint8_t *dst_ptr = dstLine;
            uint16_t *src_ptr = srcYLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint8_t)conv_bit_depth_<8, in_bit_depth, 0>(src_ptr[x]);
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint8_t *dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * uv_range.start_dst;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch_byte) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2*x+0] = (uint8_t)conv_bit_depth_<8, in_bit_depth, 0>(src_u_ptr[x]);
            dst_ptr[2*x+1] = (uint8_t)conv_bit_depth_<8, in_bit_depth, 0>(src_v_ptr[x]);
        }
    }
}

void convert_yv12_16_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_c_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_14_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_c_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_12_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_c_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_10_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_c_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_09_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_nv12_c_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int in_bit_depth>
static void RGY_FORCEINLINE convert_yv12_high_to_p010_c_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    {
        const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
        uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                memcpy(dstLine, srcYLine, y_width * (int)sizeof(uint16_t));
            } else {
                uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x++) {
                    dst_ptr[x] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_ptr[x]);
                }
            }
        }
    }
    //UV成分のコピー
    const auto uv_range = thread_y_range(crop_up >> 1, (height - crop_bottom) >> 1, thread_id, thread_n);
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * uv_range.start_src) + (crop_left >> 1));
    uint16_t *dstLine = (uint16_t *)dst[1] + dst_y_pitch * uv_range.start_dst;;
    for (int y = 0; y < uv_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2 * x + 0] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_u_ptr[x]);
            dst_ptr[2 * x + 1] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_v_ptr[x]);
        }
    }
}

void convert_yv12_16_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_c_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_14_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_c_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_12_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_c_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_10_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_c_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yv12_09_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yv12_high_to_p010_c_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

#if defined(__GNUC__)
//template展開部分で、実際には通らない箇所であっても反応してしまう
#pragma GCC diagnostic ignored "-Wshift-count-negative"
#endif

#define CHANGE_BIT_DEPTH_1(c0, offset) \
    conv_bit_depth_<out_bit_depth, in_bit_depth, offset>(c0)

#define CHANGE_BIT_DEPTH_2(c0, c1, offset) \
    c0 = CHANGE_BIT_DEPTH_1(c0, offset); \
    c1 = CHANGE_BIT_DEPTH_1(c1, offset);

#define CHANGE_BIT_DEPTH_4(c0, c1, c2, c3, offset) \
    c0 = CHANGE_BIT_DEPTH_1(c0, offset); \
    c1 = CHANGE_BIT_DEPTH_1(c1, offset); \
    c2 = CHANGE_BIT_DEPTH_1(c2, offset); \
    c3 = CHANGE_BIT_DEPTH_1(c3, offset);

template<typename Tin, int in_bit_depth>
void convert_yuv444_to_y410_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert((sizeof(Tin)  == 1 && in_bit_depth  == 8) || (sizeof(Tin)  == 2 && 8 < in_bit_depth  && in_bit_depth  <= 16), "invalid input bit depth.");
    const int out_bit_depth = 10;
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte / sizeof(Tin);
    const int dst_y_pitch = dst_y_pitch_byte / sizeof(uint32_t);
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    Tin *srcYLine = (Tin *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    Tin *srcULine = (Tin *)src[1] + src_y_pitch * y_range.start_src + crop_left;
    Tin *srcVLine = (Tin *)src[2] + src_y_pitch * y_range.start_src + crop_left;
    uint32_t *dstLine = (uint32_t *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, srcULine += src_y_pitch, srcVLine += src_y_pitch, dstLine += dst_y_pitch) {
        for (int x = 0; x < y_width; x++) {
            uint16_t pixY = std::min((uint16_t)CHANGE_BIT_DEPTH_1(srcYLine[x], 0), (uint16_t)1023);
            uint16_t pixU = std::min((uint16_t)CHANGE_BIT_DEPTH_1(srcULine[x], 0), (uint16_t)1023);
            uint16_t pixV = std::min((uint16_t)CHANGE_BIT_DEPTH_1(srcVLine[x], 0), (uint16_t)1023);
            uint32_t pix = (pixV << 20) | (pixY << 10) | pixU;
            dstLine[x] = pix;
        }
    }
}

static void convert_yuv444_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint8_t, 8>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_09_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint16_t, 9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_10_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint16_t, 10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_12_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint16_t, 12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_14_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint16_t, 14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv444_16_to_y410(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_to_y410_c<uint16_t, 16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_nv24_to_nv12_p_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcLine  = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left * 2);
    Tout *dstLine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    for (int y = 0; y < y_range.len; y += 2, srcLine += src_uv_pitch * 2, dstLine += dst_y_pitch) {
        Tout *dstC = dstLine;
        Tin *srcC = srcLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcC += 4) {
            int cy0u = srcC[0*src_uv_pitch + 0];
            int cy0v = srcC[0*src_uv_pitch + 1];
            int cy1u = srcC[1*src_uv_pitch + 0];
            int cy1v = srcC[1*src_uv_pitch + 1];

            int cu = cy0u + cy1u;
            int cv = cy0v + cy1v;
            CHANGE_BIT_DEPTH_2(cu, cv, 1);

            dstC[0] = (Tout)cu;
            dstC[1] = (Tout)cv;
        }
    }
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_nv24_to_nv12_i_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcLine = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tout *dstLine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    for (int y = 0; y < y_range.len; y += 4, srcLine += src_uv_pitch * 4, dstLine += dst_y_pitch * 2) {
        Tout *dstC = dstLine;
        Tin *srcC = srcLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcC += 4) {
            int cy0u = srcC[0*src_uv_pitch + 0];
            int cy0v = srcC[0*src_uv_pitch + 1];
            int cy1u = srcC[1*src_uv_pitch + 0];
            int cy1v = srcC[1*src_uv_pitch + 1];
            int cy2u = srcC[2*src_uv_pitch + 0];
            int cy2v = srcC[2*src_uv_pitch + 1];
            int cy3u = srcC[3*src_uv_pitch + 0];
            int cy3v = srcC[3*src_uv_pitch + 1];

            int cu_y0 = cy0u * 3 + cy2u * 1;
            int cu_y1 = cy1u * 1 + cy3u * 3;
            int cv_y0 = cy0v * 3 + cy2v * 1;
            int cv_y1 = cy1v * 1 + cy3v * 3;
            CHANGE_BIT_DEPTH_4(cu_y0, cu_y1, cv_y0, cv_y1, 2);

            dstC[0*dst_y_pitch + 0] = (Tout)cu_y0;
            dstC[0*dst_y_pitch + 1] = (Tout)cv_y0;
            dstC[1*dst_y_pitch + 0] = (Tout)cu_y1;
            dstC[1*dst_y_pitch + 1] = (Tout)cv_y1;
        }
    }
}

static void convert_nv24_to_nv12_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_nv12_p_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_nv12_i_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_p010_p(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_nv12_p_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_p010_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_nv12_i_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_nv24_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcLine  = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left * 2);
    Tout *dstULine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
    Tout *dstVLine = (Tout *)dst[2] + (dst_y_pitch >> 1) * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcLine += src_uv_pitch * 2, dstULine += dst_y_pitch, dstVLine += dst_y_pitch) {
        Tout *dstU = dstULine;
        Tout *dstV = dstVLine;
        Tin *srcC = srcLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x++, srcC += 2, dstU++, dstV++) {
            dstU[0] = (Tout)CHANGE_BIT_DEPTH_1(srcC[0], 0);
            dstV[1] = (Tout)CHANGE_BIT_DEPTH_1(srcC[1], 0);
        }
    }
}

static void convert_nv24_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_yuv444_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_yuv444_10(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_yuv444_c<uint8_t, 8, uint16_t, 10, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_yuv444_12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_yuv444_c<uint8_t, 8, uint16_t, 12, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_yuv444_14(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_yuv444_c<uint8_t, 8, uint16_t, 14, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_nv24_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_nv24_to_yuv444_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_yuv444_to_nv12_p_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
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

            int cu = cy0u + cy1u;
            int cv = cy0v + cy1v;
            CHANGE_BIT_DEPTH_2(cu, cv, 1);

            dstC[0] = (Tout)cu;
            dstC[1] = (Tout)cv;
        }
    }
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void RGY_FORCEINLINE convert_yuv444_to_nv12_i_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte / sizeof(Tin);
    Tin *srcULine = (Tin *)src[1] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tin *srcVLine = (Tin *)src[2] + ((src_uv_pitch * y_range.start_src) + crop_left);
    Tout *dstLine = (Tout *)dst[1] + (dst_y_pitch >> 1) * y_range.start_dst;
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

            int cu_y0 = cy0u * 3 + cy2u * 1;
            int cu_y1 = cy1u * 1 + cy3u * 3;
            int cv_y0 = cy0v * 3 + cy2v * 1;
            int cv_y1 = cy1v * 1 + cy3v * 3;
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

void convert_yuv444_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_09_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_10_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_12_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_14_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_16_to_nv12_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);

void convert_yuv444_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_09_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_10_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_12_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_14_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);
void convert_yuv444_16_to_p010_p_avx2(void** dst, const void** src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int* crop);

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only, bool interlaced>
static void convert_yuv422_to_nv12_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * sizeof(Tin);
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                const Tin *ptrsrc = (const Tin *)srcYLine;
                Tout *ptrdst = (Tout *)dstLine;
                for (int x = 0; x < y_width; x++) {
                    ptrdst[x] = (Tout)CHANGE_BIT_DEPTH_1(ptrsrc[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src[1] + src_uv_pitch_byte * y_range.start_src + (crop_left >> 1) * sizeof(Tin);
    uint8_t *srcVLine = (uint8_t *)src[2] + src_uv_pitch_byte * y_range.start_src + (crop_left >> 1) * sizeof(Tin);
    uint8_t *dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * (y_range.start_dst >> 1);
    if (interlaced) {
        for (int y = 0; y < y_range.len; y += 4, srcULine += src_uv_pitch_byte * 4, srcVLine += src_uv_pitch_byte * 4, dstLine += dst_y_pitch_byte * 2) {
            Tout *dstCY0 = (Tout *)(dstLine + dst_y_pitch_byte * 0);
            Tout *dstCY1 = (Tout *)(dstLine + dst_y_pitch_byte * 1);
            const Tin *srcUY0 = (const Tin *)(srcULine + src_uv_pitch_byte * 0);
            const Tin *srcUY1 = (const Tin *)(srcULine + src_uv_pitch_byte * 1);
            const Tin *srcUY2 = (const Tin *)(srcULine + src_uv_pitch_byte * 2);
            const Tin *srcUY3 = (const Tin *)(srcULine + src_uv_pitch_byte * 3);
            const Tin *srcVY0 = (const Tin *)(srcVLine + src_uv_pitch_byte * 0);
            const Tin *srcVY1 = (const Tin *)(srcVLine + src_uv_pitch_byte * 1);
            const Tin *srcVY2 = (const Tin *)(srcVLine + src_uv_pitch_byte * 2);
            const Tin *srcVY3 = (const Tin *)(srcVLine + src_uv_pitch_byte * 3);
            const int x_fin = (width - crop_right - crop_left) >> 1;
            for (int x = 0; x < x_fin; x++) {
                const int uy0x0 = srcUY0[x];
                const int uy1x0 = srcUY1[x];
                const int uy2x0 = srcUY2[x];
                const int uy3x0 = srcUY3[x];
                const int vy0x0 = srcVY0[x];
                const int vy1x0 = srcVY1[x];
                const int vy2x0 = srcVY2[x];
                const int vy3x0 = srcVY3[x];

                int u0 = uy0x0 * 3 + uy2x0 * 1;
                int u1 = uy1x0 * 1 + uy3x0 * 3;
                int v0 = vy0x0 * 3 + vy2x0 * 1;
                int v1 = vy1x0 * 1 + vy3x0 * 3;
                CHANGE_BIT_DEPTH_4(u0, u1, v0, v1, 2);
                dstCY0[2*x+0] = (Tout)u0;
                dstCY0[2*x+1] = (Tout)v0;
                dstCY1[2*x+0] = (Tout)u1;
                dstCY1[2*x+1] = (Tout)v1;
            }
        }
    } else {
        for (int y = 0; y < y_range.len; y += 2, srcULine += src_uv_pitch_byte * 2, srcVLine += src_uv_pitch_byte * 2, dstLine += dst_y_pitch_byte) {
            Tout *dstC = (Tout *)dstLine;
            const Tin *srcUY0 = (const Tin *)(srcULine + src_uv_pitch_byte * 0);
            const Tin *srcUY1 = (const Tin *)(srcULine + src_uv_pitch_byte * 1);
            const Tin *srcVY0 = (const Tin *)(srcVLine + src_uv_pitch_byte * 0);
            const Tin *srcVY1 = (const Tin *)(srcVLine + src_uv_pitch_byte * 1);
            const int x_fin = (width - crop_right - crop_left) >> 1;
            for (int x = 0; x < x_fin; x++) {
                const int uy0x0 = srcUY0[x];
                const int uy1x0 = srcUY1[x];
                const int vy0x0 = srcVY0[x];
                const int vy1x0 = srcVY1[x];
                int u = uy0x0 + uy1x0;
                int v = vy0x0 + vy1x0;
                CHANGE_BIT_DEPTH_2(u, v, 1);
                dstC[2*x+0] = (Tout)u;
                dstC[2*x+1] = (Tout)v;
            }
        }
    }
}

static void convert_yuv422_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint8_t, 8, uint8_t, 8, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_i_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint8_t, 8, uint8_t, 8, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint8_t, 8, uint16_t, 16, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_i_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint8_t, 8, uint16_t, 16, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 16, uint8_t, 8, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_i_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 16, uint8_t, 8, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 16, uint16_t, 16, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_i_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 16, uint16_t, 16, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 14, uint8_t, 8, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_i_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 14, uint8_t, 8, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 14, uint16_t, 16, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_i_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 14, uint16_t, 16, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 12, uint8_t, 8, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_i_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 12, uint8_t, 8, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 12, uint16_t, 16, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_i_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 12, uint16_t, 16, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 10, uint8_t, 8, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_i_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 10, uint8_t, 8, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 10, uint16_t, 16, false, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_i_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_nv12_c<uint16_t, 10, uint16_t, 16, false, true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

static void convert_yuv422_to_nv16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left = crop[0];
    const int crop_up = crop[1];
    const int crop_right = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        memcpy(dstLine, srcYLine, y_width);
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch_byte * y_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch_byte * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint8_t *)dst[1] + dst_y_pitch_byte * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint8_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2*x+0] = src_u_ptr[x];
            dst_ptr[2*x+1] = src_v_ptr[x];
        }
    }
}

static void convert_yuv422_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        uint8_t *src_ptr = srcYLine;
        uint16_t *dst_ptr = dstLine;
        for (int x = 0; x < y_width; x++) {
            dst_ptr[x] = (uint16_t)conv_bit_depth_<16, 8, 0>(src_ptr[x]);
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte;
    uint8_t *srcULine = (uint8_t *)src[1] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    uint8_t *srcVLine = (uint8_t *)src[2] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint16_t *)dst[1] + dst_y_pitch * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint8_t *src_u_ptr = srcULine;
        uint8_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2*x+0] = (uint16_t)conv_bit_depth_<16, 8, 0>(src_u_ptr[x]);
            dst_ptr[2*x+1] = (uint16_t)conv_bit_depth_<16, 8, 0>(src_v_ptr[x]);
        }
    }
}

template<int in_bit_depth>
static void RGY_FORCEINLINE convert_yuv422_high_to_p210_c_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    //Y成分のコピー
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint16_t *srcYLine = (uint16_t *)src[0] + src_y_pitch * y_range.start_src + crop_left;
    uint16_t *dstLine = (uint16_t *)dst[0] + dst_y_pitch * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
        if (in_bit_depth == 16) {
            memcpy(dstLine, srcYLine, y_width * sizeof(uint16_t));
        } else {
            uint16_t *src_ptr = srcYLine;
            uint16_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_ptr[x]);
            }
        }
    }
    //UV成分のコピー
    const int src_uv_pitch = src_uv_pitch_byte >> 1;
    uint16_t *srcULine = (uint16_t *)src[1] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    uint16_t *srcVLine = (uint16_t *)src[2] + ((src_uv_pitch * y_range.start_src) + (crop_left >> 1));
    dstLine = (uint16_t *)dst[1] + dst_y_pitch * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, srcULine += src_uv_pitch, srcVLine += src_uv_pitch, dstLine += dst_y_pitch) {
        const int x_fin = (width - crop_right - crop_left) >> 1;
        uint16_t *src_u_ptr = srcULine;
        uint16_t *src_v_ptr = srcVLine;
        uint16_t *dst_ptr = dstLine;
        for (int x = 0; x < x_fin; x++) {
            dst_ptr[2*x + 0] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_u_ptr[x]);
            dst_ptr[2*x + 1] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_v_ptr[x]);
        }
    }
}

void convert_yuv422_16_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_high_to_p210_c_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv422_14_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_high_to_p210_c_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv422_12_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_high_to_p210_c_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv422_10_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_high_to_p210_c_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv422_09_to_p210_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_high_to_p210_c_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void copy_yuv444_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        const uint8_t *srcYLine = (const uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
}

template<int in_bit_depth>
static void RGY_FORCEINLINE convert_yuv444_high_to_yuv444_16_c_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        const uint16_t *srcYLine = (const uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch) {
            if (in_bit_depth == 16) {
                memcpy(dstLine, srcYLine, y_width * (int)sizeof(uint16_t));
            } else {
                const uint16_t *src_ptr = srcYLine;
                uint16_t *dst_ptr = dstLine;
                for (int x = 0; x < y_width; x += 16, dst_ptr += 16, src_ptr += 16) {
                    dst_ptr[x] = (uint16_t)conv_bit_depth_<16, in_bit_depth, 0>(src_ptr[x]);
                }
            }
        }
    }
}

void convert_yuv444_16_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_16_c_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_16_c_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_16_c_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_16_c_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_16_c_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_to_yuv444_16_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int dst_y_pitch = dst_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint8_t *srcYLine = (uint8_t *)src[i] + src_y_pitch_byte * y_range.start_src + crop_left;
        uint16_t *dstLine = (uint16_t *)dst[i] + dst_y_pitch * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch) {
            uint8_t *src_ptr = srcYLine;
            uint16_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint16_t)conv_bit_depth_<16, 8, 0>(src_ptr[x]);
            }
        }
    }
}

template<int in_bit_depth>
static void RGY_FORCEINLINE convert_yuv444_high_to_yuv444_c_base(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(8 < in_bit_depth && in_bit_depth <= 16, "in_bit_depth must be 9-16.");
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const int src_y_pitch = src_y_pitch_byte >> 1;
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    for (int i = 0; i < 3; i++) {
        uint16_t *srcYLine = (uint16_t *)src[i] + src_y_pitch * y_range.start_src + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[i] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch, dstLine += dst_y_pitch_byte) {
            uint16_t *src_ptr = srcYLine;
            uint8_t *dst_ptr = dstLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint8_t)conv_bit_depth_<8, in_bit_depth, 0>(src_ptr[x]);
            }
        }
    }
}

void convert_yuv444_16_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_c_base<16>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_14_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_c_base<14>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_12_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_c_base<12>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_10_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_c_base<10>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_yuv444_09_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv444_high_to_yuv444_c_base<9>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<typename Tin, int in_bit_depth, typename Tout, int out_bit_depth, bool uv_only>
static void convert_yuv422_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);

    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * sizeof(Tin);
        uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            if (in_bit_depth == out_bit_depth && sizeof(Tin) == sizeof(Tout)) {
                memcpy(dstLine, srcYLine, y_width * sizeof(Tin));
            } else {
                const Tin *ptrsrc = (const Tin *)srcYLine;
                Tout *ptrdst = (Tout *)dstLine;
                for (int x = 0; x < y_width; x++) {
                    ptrdst[x] = (Tout)CHANGE_BIT_DEPTH_1(ptrsrc[x], 0);
                }
            }
        }
    }
    //UV成分のコピー
    for (int ic = 1; ic < 3; ic++) {
        uint8_t *srcCLine = (uint8_t *)src[ic] + src_uv_pitch_byte * y_range.start_src + (crop_left >> 1) * sizeof(Tin);
        uint8_t *dstLine = (uint8_t *)dst[ic] + dst_y_pitch_byte * y_range.start_dst;
        for (int y = 0; y < y_range.len; y++, srcCLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
            Tout *dstC = (Tout *)dstLine;
            const Tin *srcP = (const Tin *)srcCLine;
            const int x_fin = width - crop_right - crop_left;
            for (int x = 0; x < x_fin; x += 2) {
                int cxplus = (x + 2 < x_fin);
                int cy1x0 = srcP[(x>>1)+0];
                int cy1x1 = srcP[(x>>1)+cxplus];
                int cy1x01 = cy1x0 + cy1x1;
                dstC[x+0] = (Tout)CHANGE_BIT_DEPTH_1(cy1x0,  0);
                dstC[x+1] = (Tout)CHANGE_BIT_DEPTH_1(cy1x01, 1);
            }
        }
    }
}

static void convert_yuv422_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint8_t, 8, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint8_t, 8, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 10, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_10_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 10, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 12, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_12_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 12, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 14, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_14_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 14, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 16, uint8_t, 8, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}
static void convert_yuv422_16_to_yuv444_16(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    convert_yuv422_to_yuv444_c<uint16_t, 16, uint16_t, 16, false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
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
static void RGY_FORCEINLINE convert_yv12_p_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)CHANGE_BIT_DEPTH_1(srcYLine[x], 0);
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

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1);
                    CHANGE_BIT_DEPTH_4(cy1x0, cy3x0, cy1x1, cy3x1, 2);

                    dstC[0*dst_y_pitch   + 0] = (Tout)cy1x0;
                    dstC[0*dst_y_pitch   + 1] = (Tout)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch   + 0] = (Tout)cy3x0;
                    dstC[1*dst_y_pitch   + 1] = (Tout)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else if (y_range.start_dst + y >= height-2) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[-1*src_uv_pitch + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy4x0 = srcP[ 0*src_uv_pitch + 0];
                    int cy0x1 = srcP[-1*src_uv_pitch + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch + cxplus];
                    int cy4x1 = srcP[ 0*src_uv_pitch + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1);
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

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3);
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1);
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3);
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1);
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
static void RGY_FORCEINLINE convert_yv12_i_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
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
                    dstLine[x] = (Tout)conv_bit_depth_<out_bit_depth, in_bit_depth, 0>(srcYLine[x]);
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

            int cy0x0 = (sy0x0 * 1 + sy2x0 * 7);
            int cy1x0 = (sy1x0 * 3 + sy3x0 * 5);
            int cy2x0 = (sy2x0 * 5 + sy4x0 * 3);
            int cy3x0 = (sy3x0 * 7 + sy5x0 * 1);
            CHANGE_BIT_DEPTH_4(cy0x0, cy1x0, cy2x0, cy3x0, 3);

            for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                int cxplus = (x + 2 < x_fin);
                int sy0x1 = srcP[y_m2*src_uv_pitch + cxplus];
                int sy1x1 = srcP[y_m1*src_uv_pitch + cxplus];
                int sy2x1 = srcP[   0*src_uv_pitch + cxplus];
                int sy3x1 = srcP[y_p1*src_uv_pitch + cxplus];
                int sy4x1 = srcP[y_p2*src_uv_pitch + cxplus];
                int sy5x1 = srcP[y_p3*src_uv_pitch + cxplus];

                int cy0x1 = (sy0x1 * 1 + sy2x1 * 7);
                int cy1x1 = (sy1x1 * 3 + sy3x1 * 5);
                int cy2x1 = (sy2x1 * 5 + sy4x1 * 3);
                int cy3x1 = (sy3x1 * 7 + sy5x1 * 1);
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

template<int byte_per_pix>
void copy_rgb_packed_copy_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(byte_per_pix == 3 || byte_per_pix == 4);
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const uint8_t *srcYLine = (const uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * byte_per_pix;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        memcpy(dstLine, srcYLine, y_width * byte_per_pix);
    }
}

void convert_rgb24_packed_copy_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_rgb_packed_copy_c<3>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb32_packed_copy_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return copy_rgb_packed_copy_c<4>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int dstR, int dstG, int dstB, int src_byte_per_pix, int srcR, int srcG, int srcB>
void convert_rgb_packed_to_rgb_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(src_byte_per_pix == 3 || src_byte_per_pix == 4);
    static_assert(srcR < src_byte_per_pix && srcG < src_byte_per_pix && srcB < src_byte_per_pix);
    static_assert(dstR < src_byte_per_pix && dstG < src_byte_per_pix && dstB < src_byte_per_pix);
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const uint8_t *srcLine = (const uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * src_byte_per_pix;
    uint8_t *dstRLine = (uint8_t *)dst[dstR] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstGLine = (uint8_t *)dst[dstG] + dst_y_pitch_byte * y_range.start_dst;
    uint8_t *dstBLine = (uint8_t *)dst[dstB] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcLine += src_y_pitch_byte, dstRLine += dst_y_pitch_byte, dstGLine += dst_y_pitch_byte, dstBLine += dst_y_pitch_byte) {
        const uint8_t *srcptr  = srcLine;
        uint8_t *dstRptr = dstRLine;
        uint8_t *dstGptr = dstGLine;
        uint8_t *dstBptr = dstBLine;
        for (int x = 0; x < y_width; x++, srcptr += src_byte_per_pix, dstRptr++, dstGptr++, dstBptr++) {
            dstRptr[0] = srcptr[srcR];
            dstGptr[0] = srcptr[srcG];
            dstBptr[0] = srcptr[srcB];
        }
    }
}

void convert_rgb24_to_rgb_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_c<0, 1, 2, 3, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr24_to_rgb_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_c<0, 1, 2, 3, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb32_to_rgb_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_c<0, 1, 2, 4, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr32_to_rgb_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_c<0, 1, 2, 4, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int dst_byte_per_pix, int dstR, int dstG, int dstB, int src_byte_per_pix, int srcR, int srcG, int srcB>
void convert_rgb_packed_to_rgb_packed_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    static_assert(src_byte_per_pix == 3 || src_byte_per_pix == 4);
    static_assert(dst_byte_per_pix == 3 || dst_byte_per_pix == 4);
    static_assert(srcR < src_byte_per_pix && srcG < src_byte_per_pix && srcB < src_byte_per_pix);
    static_assert(dstR < dst_byte_per_pix && dstG < dst_byte_per_pix && dstB < dst_byte_per_pix);
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    const uint8_t *srcLine = (const uint8_t *)src[0] + src_y_pitch_byte * y_range.start_src + crop_left * src_byte_per_pix;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_range.len; y++, srcLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
        const uint8_t *srcptr  = srcLine;
        uint8_t *dstptr = dstLine;
        for (int x = 0; x < y_width; x++, srcptr += src_byte_per_pix, dstptr += dst_byte_per_pix) {
            dstptr[dstR] = srcptr[srcR];
            dstptr[dstG] = srcptr[srcG];
            dstptr[dstB] = srcptr[srcB];
        }
    }
}

void convert_rgb24_to_rgb32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<4, 0, 1, 2, 3, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr24_to_rgb32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<4, 0, 1, 2, 3, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb24_to_bgr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<4, 2, 1, 0, 3, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr24_to_bgr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<4, 2, 1, 0, 3, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb32_to_rgb24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<3, 0, 1, 2, 4, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr32_to_rgb24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<3, 0, 1, 2, 4, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb32_to_bgr24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<3, 2, 1, 0, 4, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_bgr32_to_bgr24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_packed_to_rgb_packed_c<3, 2, 1, 0, 4, 2, 1, 0>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

template<int dst_byte_per_pix, int dstR, int dstG, int dstB, int srcR, int srcG, int srcB>
static void RGY_FORCEINLINE convert_rgb_to_rgb_packed_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    const auto y_range = thread_y_range(crop_up, height - crop_bottom, thread_id, thread_n);
    uint8_t *srcRLine = (uint8_t *)src[srcR] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *srcGLine = (uint8_t *)src[srcG] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *srcBLine = (uint8_t *)src[srcB] + src_y_pitch_byte * y_range.start_src + crop_left;
    uint8_t *dstLine = (uint8_t *)dst[0] + dst_y_pitch_byte * y_range.start_dst;
    for (int y = 0; y < y_range.len; y++, dstLine += dst_y_pitch_byte, srcRLine += src_y_pitch_byte, srcGLine += src_y_pitch_byte, srcBLine += src_y_pitch_byte) {
        const uint8_t *ptr_srcR = srcRLine;
        const uint8_t *ptr_srcG = srcGLine;
        const uint8_t *ptr_srcB = srcBLine;
        uint8_t *ptr_dst = dstLine;
        const int x_fin = width - crop_left - crop_right;
        for (int x = 0; x < x_fin; x++, ptr_dst += dst_byte_per_pix, ptr_srcR++, ptr_srcG++, ptr_srcB++) {
            ptr_dst[dstR] = ptr_srcR[0];
            ptr_dst[dstG] = ptr_srcG[0];
            ptr_dst[dstB] = ptr_srcB[0];
        }
    }
}

void convert_gbr_to_bgr24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<3, 2, 1, 0, 2, 0, 1>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_gbr_to_bgr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 2, 1, 0, 2, 0, 1>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_gbr_to_rgb24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<3, 0, 1, 2, 2, 0, 1>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_gbr_to_rgb32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 0, 1, 2, 2, 0, 1>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_gbr_to_gbr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 2, 0, 1, 2, 0, 1>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb_to_bgr24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<3, 2, 1, 0, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb_to_bgr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 2, 1, 0, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb_to_rgb24_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<3, 0, 1, 2, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb_to_rgb32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 0, 1, 2, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}

void convert_rgb_to_gbr32_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int thread_id, int thread_n, int *crop) {
    return convert_rgb_to_rgb_packed_c<4, 2, 0, 1, 0, 1, 2>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, thread_id, thread_n, crop);
}


#pragma warning (pop)

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#define FUNC_AVX2(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },
#define FUNC_AVX(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },
#define FUNC_SSE(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },
#else
#define FUNC_AVX2(from, to, uv_only, funcp, funci, simd)
#define FUNC_AVX(from, to, uv_only, funcp, funci, simd)
#define FUNC_SSE(from, to, uv_only, funcp, funci, simd)
#endif
#define FUNC__C_(from, to, uv_only, funcp, funci, simd) { from, to, uv_only, { funcp, funci }, simd },

// テーブル作成の簡略化のため
#define AVX2  (RGY_SIMD::AVX2)
#define AVX   (RGY_SIMD::AVX)
#define SSE42 (RGY_SIMD::SSE42)
#define SSE41 (RGY_SIMD::SSE41)
#define SSSE3 (RGY_SIMD::SSSE3)
#define SSE2  (RGY_SIMD::SSE2)
#define NONE  (RGY_SIMD::NONE)

static const ConvertCSP funcList[] = {
#if !FOR_AUO
    FUNC_AVX2( RGY_CSP_NV12,      RGY_CSP_NV12,      false,  copy_nv12_to_nv12_avx2,              copy_nv12_to_nv12_avx2,              AVX2|AVX)
    FUNC_SSE(  RGY_CSP_NV12,      RGY_CSP_NV12,      false,  copy_nv12_to_nv12_sse2,              copy_nv12_to_nv12_sse2,              SSE2 )
    FUNC__C_(  RGY_CSP_NV12,      RGY_CSP_NV12,      false,  copy_nv12_to_nv12_c,                 copy_nv12_to_nv12_c,                 NONE )
    FUNC_AVX2( RGY_CSP_P010,      RGY_CSP_P010,      false,  copy_p010_to_p010_avx2,              copy_p010_to_p010_avx2,              AVX2|AVX)
    FUNC_SSE(  RGY_CSP_P010,      RGY_CSP_P010,      false,  copy_p010_to_p010_sse2,              copy_p010_to_p010_sse2,              SSE2 )
    FUNC__C_(  RGY_CSP_P010,      RGY_CSP_P010,      false,  copy_p010_to_p010_c,                 copy_p010_to_p010_c,                 NONE)
    FUNC_AVX2( RGY_CSP_NV12,      RGY_CSP_P010,      false,  copy_nv12_to_p010_avx2,              copy_nv12_to_p010_avx2,              AVX2|AVX)
    FUNC__C_(  RGY_CSP_NV12,      RGY_CSP_P010,      false,  copy_nv12_to_p010_c,                 copy_nv12_to_p010_c,                 NONE)
    FUNC_AVX2( RGY_CSP_P010,      RGY_CSP_NV12,      false,  copy_p010_to_nv12_avx2,              copy_p010_to_nv12_avx2,              AVX2|AVX)
    FUNC__C_(  RGY_CSP_P010,      RGY_CSP_NV12,      false,  copy_p010_to_nv12_c,                 copy_p010_to_nv12_c,                 NONE)
#endif
#if !CLFILTERS_AUF
    FUNC_AVX2( RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_avx2,           convert_yuy2_to_nv12_i_avx2,         AVX2|AVX)
    FUNC_AVX(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_avx,            convert_yuy2_to_nv12_i_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_ssse3,        SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_sse2,         SSE2 )
    FUNC__C_(  RGY_CSP_YUY2,      RGY_CSP_NV12,      false,  convert_yuy2_to_nv12,                convert_yuy2_to_nv12,                NONE )
    FUNC__C_(  RGY_CSP_YUY2,      RGY_CSP_YUV444,    false,  convert_yuy2_to_yuv444,              convert_yuy2_to_yuv444,              NONE )
#endif
#if FOR_AUO && !CLFILTERS_AUF
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_avx,          convert_yc48_to_yuv444_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_sse41,        convert_yc48_to_yuv444_sse41,        SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444,    false,  convert_yc48_to_yuv444_sse2,         convert_yc48_to_yuv444_sse2,         SSE2 )
    FUNC_AVX2( RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_avx2,         AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_avx,            convert_yc48_to_p010_i_avx,          AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse41,          convert_yc48_to_p010_i_sse41,        SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_ssse3,          convert_yc48_to_p010_i_ssse3,        SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_P010,      false,  convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_sse2,         SSE2 )
#endif
#if FOR_AUO
    FUNC_AVX2( RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_avx2,   convert_yc48_to_yuv444_16bit_avx2,   AVX2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_avx,    convert_yc48_to_yuv444_16bit_avx,    AVX )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_sse41,  convert_yc48_to_yuv444_16bit_sse41,  SSE41|SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_ssse3,  convert_yc48_to_yuv444_16bit_ssse3,  SSSE3|SSE2 )
    FUNC_SSE(  RGY_CSP_YC48,      RGY_CSP_YUV444_16, false,  convert_yc48_to_yuv444_16bit_sse2,   convert_yc48_to_yuv444_16bit_sse2,   SSE2 )
#endif
#if CLFILTERS_AUF
    FUNC_AVX2(RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_avx2,   convert_yuv444_16bit_to_yc48_avx2,   AVX2 )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_avx,    convert_yuv444_16bit_to_yc48_avx,    AVX )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_sse41,  convert_yuv444_16bit_to_yc48_sse41,  SSE41|SSSE3|SSE2 )
    FUNC_SSE( RGY_CSP_YUV444_16,  RGY_CSP_YC48,      false,  convert_yuv444_16bit_to_yc48_sse2,   convert_yuv444_16bit_to_yc48_sse2,   SSE2 )
#endif
#if ENABLE_AVSW_READER || ENABLE_AVI_READER || ENABLE_AVISYNTH_READER || ENABLE_VAPOURSYNTH_READER || ENABLE_AVI_READER || ENABLE_RAW_READER
    FUNC_AVX2( RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_avx2,     convert_yv12_to_nv12_avx2,     AVX2|AVX)
    FUNC_AVX(  RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_avx,      convert_yv12_to_nv12_avx,      AVX )
    FUNC_SSE(  RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_sse2,     convert_yv12_to_nv12_sse2,     SSE2 )
    FUNC__C_(  RGY_CSP_YV12, RGY_CSP_NV12, false, convert_yv12_to_nv12_c,        convert_yv12_to_nv12_c,        NONE )
    FUNC__C_(  RGY_CSP_YV12, RGY_CSP_YUV444, false, convert_yv12_p_to_yuv444,    convert_yv12_i_to_yuv444,      NONE )
    FUNC_AVX2( RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_avx2,  convert_uv_yv12_to_nv12_avx2,  AVX2|AVX )
    FUNC_AVX(  RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_avx,   convert_uv_yv12_to_nv12_avx,   AVX )
    FUNC_SSE(  RGY_CSP_YV12, RGY_CSP_NV12, true,  convert_uv_yv12_to_nv12_sse2,  convert_uv_yv12_to_nv12_sse2,  SSE2 )

    FUNC__C_(  RGY_CSP_BGR24,  RGY_CSP_BGR24, false, convert_rgb24_packed_copy_c,      convert_rgb24_packed_copy_c,    NONE )
    FUNC__C_(  RGY_CSP_BGR32,  RGY_CSP_BGR32, false, convert_rgb32_packed_copy_c,      convert_rgb32_packed_copy_c,    NONE )
    FUNC__C_(  RGY_CSP_RGB24,  RGY_CSP_RGB24, false, convert_rgb24_packed_copy_c,      convert_rgb24_packed_copy_c,    NONE )
    FUNC__C_(  RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_packed_copy_c,      convert_rgb32_packed_copy_c,    NONE )

    FUNC_SSE(  RGY_CSP_BGR24,  RGY_CSP_RGB24, false, convert_bgr24_to_rgb24_ssse3,     convert_bgr24_to_rgb24_ssse3,   SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_BGR32,  RGY_CSP_RGB32, false, convert_bgr32_to_rgb32_ssse3,     convert_bgr32_to_rgb32_ssse3,   SSSE3|SSE2)

    FUNC__C_(  RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_c,         convert_rgb24_to_rgb32_c,       NONE )
    FUNC__C_(  RGY_CSP_BGR24,  RGY_CSP_RGB32, false, convert_bgr24_to_rgb32_c,         convert_bgr24_to_rgb32_c,       NONE )
    FUNC__C_(  RGY_CSP_RGB24,  RGY_CSP_BGR32, false, convert_rgb24_to_bgr32_c,         convert_rgb24_to_bgr32_c,       NONE )
    FUNC__C_(  RGY_CSP_BGR24,  RGY_CSP_BGR32, false, convert_bgr24_to_bgr32_c,         convert_bgr24_to_bgr32_c,       NONE )

    FUNC__C_(  RGY_CSP_RGB32,  RGY_CSP_RGB24, false, convert_rgb32_to_rgb24_c,         convert_rgb32_to_rgb24_c,       NONE )
    FUNC__C_(  RGY_CSP_BGR32,  RGY_CSP_RGB24, false, convert_bgr32_to_rgb24_c,         convert_bgr32_to_rgb24_c,       NONE )
    FUNC__C_(  RGY_CSP_RGB32,  RGY_CSP_BGR24, false, convert_rgb32_to_bgr24_c,         convert_rgb32_to_bgr24_c,       NONE )
    FUNC__C_(  RGY_CSP_BGR32,  RGY_CSP_BGR24, false, convert_bgr32_to_bgr24_c,         convert_bgr32_to_bgr24_c,       NONE )
#endif
    FUNC__C_(  RGY_CSP_RGB24,  RGY_CSP_RGB,   false, convert_rgb24_to_rgb_c,           convert_rgb24_to_rgb_c,         NONE )
    FUNC__C_(  RGY_CSP_BGR24,  RGY_CSP_RGB,   false, convert_bgr24_to_rgb_c,           convert_bgr24_to_rgb_c,         NONE )
    FUNC__C_(  RGY_CSP_RGB32,  RGY_CSP_RGB,   false, convert_rgb32_to_rgb_c,           convert_rgb32_to_rgb_c,         NONE )
    FUNC__C_(  RGY_CSP_BGR32,  RGY_CSP_RGB,   false, convert_bgr32_to_rgb_c,           convert_bgr32_to_rgb_c,         NONE )

    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB,   false, convert_rgb24_to_rgb_ssse3,       convert_rgb24_to_rgb_ssse3,     SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_BGR24,  RGY_CSP_RGB,   false, convert_bgr24_to_rgb_ssse3,       convert_bgr24_to_rgb_ssse3,     SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_BGR24R, RGY_CSP_RGB,   false, convert_bgr24r_to_rgb_ssse3,      convert_bgr24r_to_rgb_ssse3,    SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_RGB32,  RGY_CSP_RGB,   false, convert_rgb32_to_rgb_sse2,        convert_rgb32_to_rgb_sse2,      SSE2 )
    FUNC_SSE(  RGY_CSP_BGR32,  RGY_CSP_RGB,   false, convert_bgr32_to_rgb_sse2,        convert_bgr32_to_rgb_sse2,      SSE2 )
    FUNC_SSE(  RGY_CSP_BGR32R, RGY_CSP_RGB,   false, convert_bgr32r_to_rgb_sse2,       convert_bgr32r_to_rgb_sse2,     SSE2 )

    FUNC_SSE(  RGY_CSP_RGB32,  RGY_CSP_RGBA,  false, convert_rgb32_to_rgba_sse2,       convert_rgb32_to_rgba_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_BGR32,  RGY_CSP_RGBA,  false, convert_bgr32_to_rgba_sse2,       convert_bgr32_to_rgba_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_BGR32R, RGY_CSP_RGBA,  false, convert_bgr32r_to_rgba_sse2,      convert_bgr32r_to_rgba_sse2,    SSE2 )
    
    FUNC_SSE(  RGY_CSP_ARGB32, RGY_CSP_RGB,   false, convert_argb32_to_rgb_sse2,       convert_argb32_to_rgb_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_ABGR32, RGY_CSP_RGB,   false, convert_abgr32_to_rgb_sse2,       convert_abgr32_to_rgb_sse2,     SSE2 )
    FUNC_SSE(  RGY_CSP_ARGB32, RGY_CSP_RGBA,  false, convert_argb32_to_rgba_sse2,      convert_argb32_to_rgba_sse2,    SSE2 )
    FUNC_SSE(  RGY_CSP_ABGR32, RGY_CSP_RGBA,  false, convert_abgr32_to_rgba_sse2,      convert_abgr32_to_rgba_sse2,    SSE2 )

#if ENABLE_AVSW_READER || ENABLE_AVI_READER || ENABLE_AVISYNTH_READER || ENABLE_VAPOURSYNTH_READER || ENABLE_AVI_READER || ENABLE_RAW_READER
    FUNC_SSE(  RGY_CSP_GBR,    RGY_CSP_RGB24, false, convert_gbr_to_rgb24_ssse3,       convert_gbr_to_rgb24_ssse3,     SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_GBRA,   RGY_CSP_RGB24, false, convert_gbr_to_rgb24_ssse3,       convert_gbr_to_rgb24_ssse3,     SSSE3|SSE2)
    FUNC_SSE(  RGY_CSP_GBR,    RGY_CSP_RGB32, false, convert_gbr_to_rgb32_sse2,        convert_gbr_to_rgb32_sse2,      SSE2 )
    FUNC_SSE(  RGY_CSP_GBRA,   RGY_CSP_RGB32, false, convert_gbr_to_rgb32_sse2,        convert_gbr_to_rgb32_sse2,      SSE2 )

    FUNC__C_(  RGY_CSP_GBR,    RGY_CSP_RGB24, false, convert_gbr_to_rgb24_c,           convert_gbr_to_rgb24_c,         NONE )
    FUNC__C_(  RGY_CSP_GBRA,   RGY_CSP_RGB24, false, convert_gbr_to_rgb24_c,           convert_gbr_to_rgb24_c,         NONE )
    FUNC__C_(  RGY_CSP_GBR,    RGY_CSP_RGB32, false, convert_gbr_to_rgb32_c,           convert_gbr_to_rgb32_c,         NONE )
    FUNC__C_(  RGY_CSP_GBRA,   RGY_CSP_RGB32, false, convert_gbr_to_rgb32_c,           convert_gbr_to_rgb32_c,         NONE )
    FUNC__C_(  RGY_CSP_GBR,    RGY_CSP_BGR32, false, convert_gbr_to_bgr32_c,           convert_gbr_to_bgr32_c,         NONE )
    FUNC__C_(  RGY_CSP_GBRA,   RGY_CSP_BGR32, false, convert_gbr_to_bgr32_c,           convert_gbr_to_bgr32_c,         NONE )

    FUNC__C_(  RGY_CSP_RGB,    RGY_CSP_RGB24, false, convert_rgb_to_rgb24_c,           convert_rgb_to_rgb24_c,         NONE )
    FUNC__C_(  RGY_CSP_RGBA,   RGY_CSP_RGB24, false, convert_rgb_to_rgb24_c,           convert_rgb_to_rgb24_c,         NONE )
    FUNC__C_(  RGY_CSP_RGB,    RGY_CSP_RGB32, false, convert_rgb_to_rgb32_c,           convert_rgb_to_rgb32_c,         NONE )
    FUNC__C_(  RGY_CSP_RGBA,   RGY_CSP_RGB32, false, convert_rgb_to_rgb32_c,           convert_rgb_to_rgb32_c,         NONE )
    FUNC__C_(  RGY_CSP_RGB,    RGY_CSP_RGB32, false, convert_rgb_to_bgr32_c,           convert_rgb_to_bgr32_c,         NONE )
    FUNC__C_(  RGY_CSP_RGBA,   RGY_CSP_RGB32, false, convert_rgb_to_bgr32_c,           convert_rgb_to_bgr32_c,         NONE )

    FUNC_SSE(  RGY_CSP_RGB,    RGY_CSP_RGB,   false, copy_rgb_to_rgb_sse2,             copy_rgb_to_rgb_sse2,           SSE2 )
    FUNC_SSE(  RGY_CSP_GBR,    RGY_CSP_RGB,   false, copy_gbr_to_rgb_sse2,             copy_gbr_to_rgb_sse2,           SSE2 )

    FUNC_AVX2( RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_avx2,      convert_rgb24_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX(  RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_avx,       convert_rgb24_to_rgb32_avx,       AVX )
    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB32, false, convert_rgb24_to_rgb32_ssse3,     convert_rgb24_to_rgb32_ssse3,     SSSE3|SSE2 )
    FUNC_AVX2( RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_avx2,      convert_rgb32_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX(  RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_avx,       convert_rgb32_to_rgb32_avx,       AVX )
    FUNC_SSE(  RGY_CSP_RGB32,  RGY_CSP_RGB32, false, convert_rgb32_to_rgb32_sse2,      convert_rgb32_to_rgb32_sse2,      SSE2 )
    FUNC_AVX2( RGY_CSP_RGB24,  RGY_CSP_RGB24, false, convert_rgb24_to_rgb24_avx2,      convert_rgb24_to_rgb24_avx2,      AVX2|AVX)
    FUNC_SSE(  RGY_CSP_RGB24,  RGY_CSP_RGB24, false, convert_rgb24_to_rgb24_sse2,      convert_rgb24_to_rgb24_sse2,      SSE2 )
    FUNC_AVX2( RGY_CSP_BGR24,  RGY_CSP_BGR32, false, convert_rgb24_to_rgb32_avx2,      convert_rgb24_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX(  RGY_CSP_BGR24,  RGY_CSP_BGR32, false, convert_rgb24_to_rgb32_avx,       convert_rgb24_to_rgb32_avx,       AVX )
    FUNC_SSE(  RGY_CSP_BGR24,  RGY_CSP_BGR32, false, convert_rgb24_to_rgb32_ssse3,     convert_rgb24_to_rgb32_ssse3,     SSSE3|SSE2 )
    FUNC_AVX2( RGY_CSP_BGR32,  RGY_CSP_BGR32, false, convert_rgb32_to_rgb32_avx2,      convert_rgb32_to_rgb32_avx2,      AVX2|AVX )
    FUNC_AVX(  RGY_CSP_BGR32,  RGY_CSP_BGR32, false, convert_rgb32_to_rgb32_avx,       convert_rgb32_to_rgb32_avx,       AVX )
    FUNC_SSE(  RGY_CSP_BGR32,  RGY_CSP_BGR32, false, convert_rgb32_to_rgb32_sse2,      convert_rgb32_to_rgb32_sse2,      SSE2 )
    FUNC_AVX2( RGY_CSP_BGR24,  RGY_CSP_BGR24, false, convert_rgb24_to_rgb24_avx2,      convert_rgb24_to_rgb24_avx2,      AVX2|AVX)
    FUNC_SSE(  RGY_CSP_BGR24,  RGY_CSP_BGR24, false, convert_rgb24_to_rgb24_sse2,      convert_rgb24_to_rgb24_sse2,      SSE2 )

    FUNC_AVX2( RGY_CSP_BGR24R, RGY_CSP_BGR32, false, convert_bgr24r_to_bgr32_avx2,     convert_bgr24r_to_bgr32_avx2,     AVX2|AVX )
    FUNC_AVX(  RGY_CSP_BGR24R, RGY_CSP_BGR32, false, convert_bgr24r_to_bgr32_avx,      convert_bgr24r_to_bgr32_avx,      AVX )
    FUNC_SSE(  RGY_CSP_BGR24R, RGY_CSP_BGR32, false, convert_bgr24r_to_bgr32_ssse3,    convert_bgr24r_to_bgr32_ssse3,    SSSE3|SSE2 )
    FUNC_AVX2( RGY_CSP_BGR32R, RGY_CSP_BGR32, false, convert_bgr32r_to_bgr32_avx2,     convert_bgr32r_to_bgr32_avx2,     AVX2|AVX )
    FUNC_AVX(  RGY_CSP_BGR32R, RGY_CSP_BGR32, false, convert_bgr32r_to_bgr32_avx,      convert_bgr32r_to_bgr32_avx,      AVX )
    FUNC_SSE(  RGY_CSP_BGR32R, RGY_CSP_BGR32, false, convert_bgr32r_to_bgr32_sse2,     convert_bgr32r_to_bgr32_sse2,     SSE2 )
    FUNC_AVX2( RGY_CSP_BGR24R, RGY_CSP_BGR24, false, convert_bgr24r_to_bgr24_avx2,     convert_bgr24r_to_bgr24_avx2,     AVX2|AVX)
    FUNC_SSE(  RGY_CSP_BGR24R, RGY_CSP_BGR24, false, convert_bgr24r_to_bgr24_sse2,     convert_bgr24r_to_bgr24_sse2,     SSE2 )

    FUNC_AVX2( RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_avx2,           convert_yv12_to_p010_avx2,    AVX2|AVX )
    FUNC_AVX(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_avx,            convert_yv12_to_p010_avx,     AVX )
    FUNC_SSE(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010_sse2,           convert_yv12_to_p010_sse2,    SSE2 )
    FUNC__C_(  RGY_CSP_YV12,      RGY_CSP_P010,      false, convert_yv12_to_p010,                convert_yv12_to_p010,         NONE )
    FUNC__C_(  RGY_CSP_YV12,      RGY_CSP_YUV444_16, false, convert_yv12_p_to_yuv444_16bit,      convert_yv12_i_to_yuv444_16bit, NONE )
    FUNC_AVX2( RGY_CSP_YV12_16,   RGY_CSP_NV12,      false, convert_yv12_16_to_nv12_avx2,        convert_yv12_16_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_16,   RGY_CSP_NV12,      false, convert_yv12_16_to_nv12_sse2,        convert_yv12_16_to_nv12_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_16,   RGY_CSP_NV12,      false, convert_yv12_16_to_nv12_c,           convert_yv12_16_to_nv12_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_14,   RGY_CSP_NV12,      false, convert_yv12_14_to_nv12_avx2,        convert_yv12_14_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_NV12,      false, convert_yv12_14_to_nv12_sse2,        convert_yv12_14_to_nv12_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_14,   RGY_CSP_NV12,      false, convert_yv12_14_to_nv12_c,           convert_yv12_14_to_nv12_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_12,   RGY_CSP_NV12,      false, convert_yv12_12_to_nv12_avx2,        convert_yv12_12_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_NV12,      false, convert_yv12_12_to_nv12_sse2,        convert_yv12_12_to_nv12_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_12,   RGY_CSP_NV12,      false, convert_yv12_12_to_nv12_c,           convert_yv12_12_to_nv12_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_10_to_nv12_avx2,        convert_yv12_10_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_10_to_nv12_sse2,        convert_yv12_10_to_nv12_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_10_to_nv12_c,           convert_yv12_10_to_nv12_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_09,   RGY_CSP_NV12,      false, convert_yv12_09_to_nv12_avx2,        convert_yv12_09_to_nv12_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_NV12,      false, convert_yv12_09_to_nv12_sse2,        convert_yv12_09_to_nv12_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_10,   RGY_CSP_NV12,      false, convert_yv12_09_to_nv12_c,           convert_yv12_09_to_nv12_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_16,   RGY_CSP_P010,      false, convert_yv12_16_to_p010_avx2,        convert_yv12_16_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_16,   RGY_CSP_P010,      false, convert_yv12_16_to_p010_sse2,        convert_yv12_16_to_p010_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_16,   RGY_CSP_P010,      false, convert_yv12_16_to_p010_c,           convert_yv12_16_to_p010_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_14,   RGY_CSP_P010,      false, convert_yv12_14_to_p010_avx2,        convert_yv12_14_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_14,   RGY_CSP_P010,      false, convert_yv12_14_to_p010_sse2,        convert_yv12_14_to_p010_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_14,   RGY_CSP_P010,      false, convert_yv12_14_to_p010_c,           convert_yv12_14_to_p010_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_12,   RGY_CSP_P010,      false, convert_yv12_12_to_p010_avx2,        convert_yv12_12_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_12,   RGY_CSP_P010,      false, convert_yv12_12_to_p010_sse2,        convert_yv12_12_to_p010_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_12,   RGY_CSP_P010,      false, convert_yv12_12_to_p010_c,           convert_yv12_12_to_p010_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_10,   RGY_CSP_P010,      false, convert_yv12_10_to_p010_avx2,        convert_yv12_10_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_10,   RGY_CSP_P010,      false, convert_yv12_10_to_p010_sse2,        convert_yv12_10_to_p010_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_10,   RGY_CSP_P010,      false, convert_yv12_10_to_p010_c,           convert_yv12_10_to_p010_c,    NONE )
    FUNC_AVX2( RGY_CSP_YV12_09,   RGY_CSP_P010,      false, convert_yv12_09_to_p010_avx2,        convert_yv12_09_to_p010_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YV12_09,   RGY_CSP_P010,      false, convert_yv12_09_to_p010_sse2,        convert_yv12_09_to_p010_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YV12_09,   RGY_CSP_P010,      false, convert_yv12_09_to_p010_c,           convert_yv12_09_to_p010_c,    NONE )
    FUNC__C_(  RGY_CSP_YV12_16,   RGY_CSP_YUV444,    false, convert_yv12_16_p_to_yuv444,         convert_yv12_16_i_to_yuv444,  NONE )
    FUNC__C_(  RGY_CSP_YV12_14,   RGY_CSP_YUV444,    false, convert_yv12_14_p_to_yuv444,         convert_yv12_14_i_to_yuv444,  NONE )
    FUNC__C_(  RGY_CSP_YV12_12,   RGY_CSP_YUV444,    false, convert_yv12_12_p_to_yuv444,         convert_yv12_12_i_to_yuv444,  NONE )
    FUNC__C_(  RGY_CSP_YV12_10,   RGY_CSP_YUV444,    false, convert_yv12_10_p_to_yuv444,         convert_yv12_10_i_to_yuv444,  NONE )
    FUNC__C_(  RGY_CSP_YV12_09,   RGY_CSP_YUV444,    false, convert_yv12_09_p_to_yuv444,         convert_yv12_09_i_to_yuv444,  NONE )
    FUNC__C_(  RGY_CSP_YV12_16,   RGY_CSP_YUV444_16, false, convert_yv12_16_p_to_yuv444_16bit,   convert_yv12_16_i_to_yuv444_16bit, NONE )
    FUNC__C_(  RGY_CSP_YV12_14,   RGY_CSP_YUV444_16, false, convert_yv12_14_p_to_yuv444_16bit,   convert_yv12_14_i_to_yuv444_16bit, NONE )
    FUNC__C_(  RGY_CSP_YV12_12,   RGY_CSP_YUV444_16, false, convert_yv12_12_p_to_yuv444_16bit,   convert_yv12_12_i_to_yuv444_16bit, NONE )
    FUNC__C_(  RGY_CSP_YV12_10,   RGY_CSP_YUV444_16, false, convert_yv12_10_p_to_yuv444_16bit,   convert_yv12_10_i_to_yuv444_16bit, NONE )
    FUNC__C_(  RGY_CSP_YV12_09,   RGY_CSP_YUV444_16, false, convert_yv12_09_p_to_yuv444_16bit,   convert_yv12_09_i_to_yuv444_16bit, NONE )
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_NV12,      false, convert_yuv422_to_nv12,              convert_yuv422_i_to_nv12,       NONE )
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_P010,      false, convert_yuv422_to_p010,              convert_yuv422_i_to_p010,       NONE )
    FUNC__C_(  RGY_CSP_YUV422_16, RGY_CSP_NV12,      false, convert_yuv422_16_to_nv12,           convert_yuv422_16_i_to_nv12,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_16, RGY_CSP_P010,      false, convert_yuv422_16_to_p010,           convert_yuv422_16_i_to_p010,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_14, RGY_CSP_NV12,      false, convert_yuv422_14_to_nv12,           convert_yuv422_14_i_to_nv12,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_14, RGY_CSP_P010,      false, convert_yuv422_14_to_p010,           convert_yuv422_14_i_to_p010,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_12, RGY_CSP_NV12,      false, convert_yuv422_12_to_nv12,           convert_yuv422_12_i_to_nv12,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_12, RGY_CSP_P010,      false, convert_yuv422_12_to_p010,           convert_yuv422_12_i_to_p010,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_10, RGY_CSP_NV12,      false, convert_yuv422_10_to_nv12,           convert_yuv422_10_i_to_nv12,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_10, RGY_CSP_P010,      false, convert_yuv422_10_to_p010,           convert_yuv422_10_i_to_p010,    NONE )
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_YUV444,    false, convert_yuv422_to_yuv444,            convert_yuv422_to_yuv444,       NONE )
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_YUV444_16, false, convert_yuv422_to_yuv444_16,         convert_yuv422_to_yuv444_16,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_16, RGY_CSP_YUV444,    false, convert_yuv422_16_to_yuv444,         convert_yuv422_16_to_yuv444,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_16, RGY_CSP_YUV444_16, false, convert_yuv422_16_to_yuv444_16,      convert_yuv422_16_to_yuv444_16, NONE )
    FUNC__C_(  RGY_CSP_YUV422_14, RGY_CSP_YUV444,    false, convert_yuv422_14_to_yuv444,         convert_yuv422_14_to_yuv444,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_14, RGY_CSP_YUV444_16, false, convert_yuv422_14_to_yuv444_16,      convert_yuv422_14_to_yuv444_16, NONE )
    FUNC__C_(  RGY_CSP_YUV422_12, RGY_CSP_YUV444,    false, convert_yuv422_12_to_yuv444,         convert_yuv422_12_to_yuv444,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_12, RGY_CSP_YUV444_16, false, convert_yuv422_12_to_yuv444_16,      convert_yuv422_12_to_yuv444_16, NONE )
    FUNC__C_(  RGY_CSP_YUV422_10, RGY_CSP_YUV444,    false, convert_yuv422_10_to_yuv444,         convert_yuv422_10_to_yuv444,    NONE )
    FUNC__C_(  RGY_CSP_YUV422_10, RGY_CSP_YUV444_16, false, convert_yuv422_10_to_yuv444_16,      convert_yuv422_10_to_yuv444_16, NONE )
    FUNC_SSE(  RGY_CSP_YUV422,    RGY_CSP_NV16,      false, convert_yuv422_to_nv16_sse2,         convert_yuv422_to_nv16_sse2,    SSE2)
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_NV16,      false, convert_yuv422_to_nv16_c,            convert_yuv422_to_nv16_c,       NONE)
    FUNC_SSE(  RGY_CSP_YUV422,    RGY_CSP_P210,      false, convert_yuv422_to_p210_sse2,         convert_yuv422_to_p210_sse2,    SSE2)
    FUNC__C_(  RGY_CSP_YUV422,    RGY_CSP_P210,      false, convert_yuv422_to_p210_c,            convert_yuv422_to_p210_c,       NONE)
    FUNC_SSE(  RGY_CSP_YUV422_16, RGY_CSP_P210,      false, convert_yuv422_16_to_p210_sse2,      convert_yuv422_16_to_p210_sse2, SSE2)
    FUNC__C_(  RGY_CSP_YUV422_16, RGY_CSP_P210,      false, convert_yuv422_16_to_p210_c,         convert_yuv422_16_to_p210_c,    NONE)
    FUNC_SSE(  RGY_CSP_YUV422_14, RGY_CSP_P210,      false, convert_yuv422_14_to_p210_sse2,      convert_yuv422_14_to_p210_sse2, SSE2)
    FUNC__C_(  RGY_CSP_YUV422_14, RGY_CSP_P210,      false, convert_yuv422_14_to_p210_c,         convert_yuv422_14_to_p210_c,    NONE)
    FUNC_SSE(  RGY_CSP_YUV422_12, RGY_CSP_P210,      false, convert_yuv422_12_to_p210_sse2,      convert_yuv422_12_to_p210_sse2, SSE2)
    FUNC__C_(  RGY_CSP_YUV422_12, RGY_CSP_P210,      false, convert_yuv422_12_to_p210_c,         convert_yuv422_12_to_p210_c,    NONE)
    FUNC_SSE(  RGY_CSP_YUV422_10, RGY_CSP_P210,      false, convert_yuv422_10_to_p210_sse2,      convert_yuv422_10_to_p210_sse2, SSE2)
    FUNC__C_(  RGY_CSP_YUV422_10, RGY_CSP_P210,      false, convert_yuv422_10_to_p210_c,         convert_yuv422_10_to_p210_c,    NONE)
    FUNC_SSE(  RGY_CSP_YUV422_09, RGY_CSP_P210,      false, convert_yuv422_09_to_p210_sse2,      convert_yuv422_09_to_p210_sse2, SSE2)
    FUNC__C_(  RGY_CSP_YUV422_09, RGY_CSP_P210,      false, convert_yuv422_09_to_p210_c,         convert_yuv422_09_to_p210_c,    NONE)
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_YUV444,    false, copy_yuv444_to_yuv444_avx2,          copy_yuv444_to_yuv444_avx2,      AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_YUV444,    false, copy_yuv444_to_yuv444_sse2,          copy_yuv444_to_yuv444_sse2,      SSE2 )
    FUNC__C_(  RGY_CSP_YUV444,    RGY_CSP_YUV444,    false, copy_yuv444_to_yuv444_c,             copy_yuv444_to_yuv444_c,         NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_VUYA,      false, copy_yuv444_16_to_ayuv444_avx2,      copy_yuv444_16_to_ayuv444_avx2,  AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_VUYA,      false, copy_yuv444_16_to_ayuv444_sse2,      copy_yuv444_16_to_ayuv444_sse2,  SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_VUYA,      false, copy_yuv444_14_to_ayuv444_avx2,      copy_yuv444_14_to_ayuv444_avx2,  AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_VUYA,      false, copy_yuv444_14_to_ayuv444_sse2,      copy_yuv444_14_to_ayuv444_sse2,  SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_VUYA,      false, copy_yuv444_12_to_ayuv444_avx2,      copy_yuv444_12_to_ayuv444_avx2,  AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_VUYA,      false, copy_yuv444_12_to_ayuv444_sse2,      copy_yuv444_12_to_ayuv444_sse2,  SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_VUYA,      false, copy_yuv444_10_to_ayuv444_avx2,      copy_yuv444_10_to_ayuv444_avx2,  AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_VUYA,      false, copy_yuv444_10_to_ayuv444_sse2,      copy_yuv444_10_to_ayuv444_sse2,  SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_VUYA,      false, copy_yuv444_09_to_ayuv444_avx2,      copy_yuv444_09_to_ayuv444_avx2,  AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_VUYA,      false, copy_yuv444_09_to_ayuv444_sse2,      copy_yuv444_09_to_ayuv444_sse2,  SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_VUYA,      false, copy_yuv444_to_ayuv444_avx2,         copy_yuv444_to_ayuv444_avx2,     AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_VUYA,      false, copy_yuv444_to_ayuv444_sse2,         copy_yuv444_to_ayuv444_sse2,     SSE2 )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_Y410,      false, convert_yuv444_16_to_y410_avx2,      convert_yuv444_16_to_y410_avx2,  AVX2|AVX)
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_Y410,      false, convert_yuv444_16_to_y410_sse41,     convert_yuv444_16_to_y410_sse41, SSE41)
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_Y410,      false, convert_yuv444_16_to_y410_sse2,      convert_yuv444_16_to_y410_sse2,  SSE2)
    FUNC__C_(  RGY_CSP_YUV444_16, RGY_CSP_Y410,      false, convert_yuv444_16_to_y410,           convert_yuv444_16_to_y410,       NONE)
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_Y410,      false, convert_yuv444_14_to_y410_avx2,      convert_yuv444_14_to_y410_avx2,  AVX2|AVX)
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_Y410,      false, convert_yuv444_14_to_y410_sse41,     convert_yuv444_14_to_y410_sse41, SSE41)
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_Y410,      false, convert_yuv444_14_to_y410_sse2,      convert_yuv444_14_to_y410_sse2,  SSE2)
    FUNC__C_(  RGY_CSP_YUV444_14, RGY_CSP_Y410,      false, convert_yuv444_14_to_y410,           convert_yuv444_14_to_y410,       NONE)
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_Y410,      false, convert_yuv444_12_to_y410_avx2,      convert_yuv444_12_to_y410_avx2,  AVX2|AVX)
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_Y410,      false, convert_yuv444_12_to_y410_sse41,     convert_yuv444_12_to_y410_sse41, SSE41)
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_Y410,      false, convert_yuv444_12_to_y410_sse2,      convert_yuv444_12_to_y410_sse2,  SSE2)
    FUNC__C_(  RGY_CSP_YUV444_12, RGY_CSP_Y410,      false, convert_yuv444_12_to_y410,           convert_yuv444_12_to_y410,       NONE)
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_Y410,      false, convert_yuv444_10_to_y410_avx2,      convert_yuv444_10_to_y410_avx2,  AVX2|AVX)
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_Y410,      false, convert_yuv444_10_to_y410_sse41,     convert_yuv444_10_to_y410_sse41, SSE41)
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_Y410,      false, convert_yuv444_10_to_y410_sse2,      convert_yuv444_10_to_y410_sse2,  SSE2)
    FUNC__C_(  RGY_CSP_YUV444_10, RGY_CSP_Y410,      false, convert_yuv444_10_to_y410,           convert_yuv444_10_to_y410,       NONE)
    FUNC__C_(  RGY_CSP_YUV444_09, RGY_CSP_Y410,      false, convert_yuv444_09_to_y410,           convert_yuv444_09_to_y410,       NONE)
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_Y410,      false, convert_yuv444_to_y410_avx2,         convert_yuv444_to_y410_avx2,     AVX2|AVX)
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_Y410,      false, convert_yuv444_to_y410_sse2,         convert_yuv444_to_y410_sse2,     SSE2)
    FUNC__C_(  RGY_CSP_YUV444,    RGY_CSP_Y410,      false, convert_yuv444_to_y410,              convert_yuv444_to_y410,          NONE)
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_NV12,      false, convert_nv24_to_nv12_p,              convert_nv24_to_nv12_i,      NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_P010,      false, convert_nv24_to_p010_p,              convert_nv24_to_p010_i,      NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_YUV444,    false, convert_nv24_to_yuv444,              convert_nv24_to_yuv444,      NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_YUV444_16, false, convert_nv24_to_yuv444_16,           convert_nv24_to_yuv444_16,   NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_YUV444_14, false, convert_nv24_to_yuv444_14,           convert_nv24_to_yuv444_14,   NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_YUV444_12, false, convert_nv24_to_yuv444_12,           convert_nv24_to_yuv444_12,   NONE )
    FUNC__C_(  RGY_CSP_NV24,      RGY_CSP_YUV444_10, false, convert_nv24_to_yuv444_10,           convert_nv24_to_yuv444_10,   NONE )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_NV12,      false, convert_yuv444_to_nv12_p_avx2,       convert_yuv444_to_nv12_i,    AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444,    RGY_CSP_NV12,      false, convert_yuv444_to_nv12_p,            convert_yuv444_to_nv12_i,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_P010,      false, convert_yuv444_to_p010_p_avx2,       convert_yuv444_to_p010_i,    AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444,    RGY_CSP_P010,      false, convert_yuv444_to_p010_p,            convert_yuv444_to_p010_i,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_NV12,      false, convert_yuv444_16_to_nv12_p_avx2,    convert_yuv444_16_to_nv12_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_16, RGY_CSP_NV12,      false, convert_yuv444_16_to_nv12_p,         convert_yuv444_16_to_nv12_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_NV12,      false, convert_yuv444_14_to_nv12_p_avx2,    convert_yuv444_14_to_nv12_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_14, RGY_CSP_NV12,      false, convert_yuv444_14_to_nv12_p,         convert_yuv444_14_to_nv12_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_NV12,      false, convert_yuv444_12_to_nv12_p_avx2,    convert_yuv444_12_to_nv12_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_12, RGY_CSP_NV12,      false, convert_yuv444_12_to_nv12_p,         convert_yuv444_12_to_nv12_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_NV12,      false, convert_yuv444_10_to_nv12_p_avx2,    convert_yuv444_10_to_nv12_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_10, RGY_CSP_NV12,      false, convert_yuv444_10_to_nv12_p,         convert_yuv444_10_to_nv12_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_NV12,      false, convert_yuv444_09_to_nv12_p_avx2,    convert_yuv444_09_to_nv12_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_09, RGY_CSP_NV12,      false, convert_yuv444_09_to_nv12_p,         convert_yuv444_09_to_nv12_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_P010,      false, convert_yuv444_16_to_p010_p_avx2,    convert_yuv444_16_to_p010_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_16, RGY_CSP_P010,      false, convert_yuv444_16_to_p010_p,         convert_yuv444_16_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_P010,      false, convert_yuv444_14_to_p010_p_avx2,    convert_yuv444_14_to_p010_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_14, RGY_CSP_P010,      false, convert_yuv444_14_to_p010_p,         convert_yuv444_14_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_P010,      false, convert_yuv444_12_to_p010_p_avx2,    convert_yuv444_12_to_p010_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_12, RGY_CSP_P010,      false, convert_yuv444_12_to_p010_p,         convert_yuv444_12_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_P010,      false, convert_yuv444_10_to_p010_p_avx2,    convert_yuv444_10_to_p010_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_10, RGY_CSP_P010,      false, convert_yuv444_10_to_p010_p,         convert_yuv444_10_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_P010,      false, convert_yuv444_09_to_p010_p_avx2,    convert_yuv444_09_to_p010_i, AVX2|AVX )
    FUNC__C_(  RGY_CSP_YUV444_09, RGY_CSP_P010,      false, convert_yuv444_09_to_p010_p,         convert_yuv444_09_to_p010_i, NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_YUV444_16, false, convert_yuv444_16_to_yuv444_16_avx2, convert_yuv444_16_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_YUV444_16, false, convert_yuv444_16_to_yuv444_16_sse2, convert_yuv444_16_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_16, RGY_CSP_YUV444_16, false, convert_yuv444_16_to_yuv444_16_c,    convert_yuv444_16_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_YUV444_16, false, convert_yuv444_14_to_yuv444_16_avx2, convert_yuv444_14_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_YUV444_16, false, convert_yuv444_14_to_yuv444_16_sse2, convert_yuv444_14_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_14, RGY_CSP_YUV444_16, false, convert_yuv444_14_to_yuv444_16_c,    convert_yuv444_14_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_YUV444_16, false, convert_yuv444_12_to_yuv444_16_avx2, convert_yuv444_12_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_YUV444_16, false, convert_yuv444_12_to_yuv444_16_sse2, convert_yuv444_12_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_12, RGY_CSP_YUV444_16, false, convert_yuv444_12_to_yuv444_16_c,    convert_yuv444_12_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_YUV444_16, false, convert_yuv444_10_to_yuv444_16_avx2, convert_yuv444_10_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_YUV444_16, false, convert_yuv444_10_to_yuv444_16_sse2, convert_yuv444_10_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_10, RGY_CSP_YUV444_16, false, convert_yuv444_10_to_yuv444_16_c,    convert_yuv444_10_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_YUV444_16, false, convert_yuv444_09_to_yuv444_16_avx2, convert_yuv444_09_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_YUV444_16, false, convert_yuv444_09_to_yuv444_16_sse2, convert_yuv444_09_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_09, RGY_CSP_YUV444_16, false, convert_yuv444_09_to_yuv444_16_c,    convert_yuv444_09_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444,    RGY_CSP_YUV444_16, false, convert_yuv444_to_yuv444_16_avx2,    convert_yuv444_to_yuv444_16_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444,    RGY_CSP_YUV444_16, false, convert_yuv444_to_yuv444_16_sse2,    convert_yuv444_to_yuv444_16_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444,    RGY_CSP_YUV444_16, false, convert_yuv444_to_yuv444_16_c,       convert_yuv444_to_yuv444_16_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_16, RGY_CSP_YUV444,    false, convert_yuv444_16_to_yuv444_avx2,    convert_yuv444_16_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_16, RGY_CSP_YUV444,    false, convert_yuv444_16_to_yuv444_sse2,    convert_yuv444_16_to_yuv444_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_16, RGY_CSP_YUV444,    false, convert_yuv444_16_to_yuv444_c,       convert_yuv444_16_to_yuv444_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_14, RGY_CSP_YUV444,    false, convert_yuv444_14_to_yuv444_avx2,    convert_yuv444_14_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_14, RGY_CSP_YUV444,    false, convert_yuv444_14_to_yuv444_sse2,    convert_yuv444_14_to_yuv444_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_14, RGY_CSP_YUV444,    false, convert_yuv444_14_to_yuv444_c,       convert_yuv444_14_to_yuv444_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_12, RGY_CSP_YUV444,    false, convert_yuv444_12_to_yuv444_avx2,    convert_yuv444_12_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_12, RGY_CSP_YUV444,    false, convert_yuv444_12_to_yuv444_sse2,    convert_yuv444_12_to_yuv444_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_12, RGY_CSP_YUV444,    false, convert_yuv444_12_to_yuv444_c,       convert_yuv444_12_to_yuv444_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_10, RGY_CSP_YUV444,    false, convert_yuv444_10_to_yuv444_avx2,    convert_yuv444_10_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_10, RGY_CSP_YUV444,    false, convert_yuv444_10_to_yuv444_sse2,    convert_yuv444_10_to_yuv444_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_10, RGY_CSP_YUV444,    false, convert_yuv444_10_to_yuv444_c,       convert_yuv444_10_to_yuv444_c,    NONE )
    FUNC_AVX2( RGY_CSP_YUV444_09, RGY_CSP_YUV444,    false, convert_yuv444_09_to_yuv444_avx2,    convert_yuv444_09_to_yuv444_avx2, AVX2|AVX )
    FUNC_SSE(  RGY_CSP_YUV444_09, RGY_CSP_YUV444,    false, convert_yuv444_09_to_yuv444_sse2,    convert_yuv444_09_to_yuv444_sse2, SSE2 )
    FUNC__C_(  RGY_CSP_YUV444_09, RGY_CSP_YUV444,    false, convert_yuv444_09_to_yuv444_c,       convert_yuv444_09_to_yuv444_c,    NONE )
#endif
};

const TCHAR *picstrcut_to_str(RGY_PICSTRUCT picstruct) {
    switch (picstruct) {
    case RGY_PICSTRUCT_FRAME:        return _T("progressive");
    case RGY_PICSTRUCT_FIELD_TOP:    return _T("tff_field");
    case RGY_PICSTRUCT_FIELD_BOTTOM: return _T("bff_field");
    case RGY_PICSTRUCT_FIELD:        return _T("field");
    case RGY_PICSTRUCT_TFF:          return _T("tff");
    case RGY_PICSTRUCT_BFF:          return _T("bff");
    case RGY_PICSTRUCT_FRAME_TFF:    return _T("tff_frame");
    case RGY_PICSTRUCT_FRAME_BFF:    return _T("bff_frame");
    case RGY_PICSTRUCT_AUTO:         return _T("auto");
    case RGY_PICSTRUCT_UNKNOWN:
    default: return _T("unknown");
    }
}

const TCHAR *get_memtype_str(RGY_MEM_TYPE type) {
    switch (type) {
    case RGY_MEM_TYPE_CPU: return _T("cpu");
    case RGY_MEM_TYPE_GPU: return _T("gpu");
    case RGY_MEM_TYPE_GPU_IMAGE: return _T("gpu_image");
    case RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED: return _T("gpu_image_norm");
    default: return _T("unknwon");
    }
}

const ConvertCSP *get_convert_csp_func(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, RGY_SIMD simd) {
    if (rgy_csp_has_alpha(csp_from) && rgy_csp_has_alpha(csp_to)) {
        return get_convert_csp_func(rgy_csp_alpha_base(csp_from), rgy_csp_alpha_base(csp_to), uv_only, simd);
    }
    if (rgy_csp_has_alpha(csp_from)) {
        return get_convert_csp_func(rgy_csp_alpha_base(csp_from), csp_to, uv_only, simd);
    }
    RGY_SIMD availableSIMD = get_availableSIMD() & simd;
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

funcConvertCSP get_copy_alpha_func(RGY_CSP csp_from, RGY_CSP csp_to) {
    const auto csp_base_from = rgy_csp_alpha_base(csp_from);
    const auto csp_base_to = rgy_csp_alpha_base(csp_to);
    if (csp_base_from != RGY_CSP_NA && csp_base_to != RGY_CSP_NA) {
        const int bit_depth_from = RGY_CSP_BIT_DEPTH[csp_base_from];
        const int bit_depth_to = RGY_CSP_BIT_DEPTH[csp_base_to];
        switch (bit_depth_from) {
        case 8:
            switch (bit_depth_to) {
            case 8:  return copy_plane_u8_to_u8;
            case 10: return copy_plane_u8_to_u10;
            case 12: return copy_plane_u8_to_u12;
            case 16: return copy_plane_u8_to_u16;
            default: break;
            }
            break;
        case 10:
            switch (bit_depth_to) {
            case 8:  return copy_plane_u10_to_u8;
            case 10: return copy_plane_u10_to_u10;
            case 12: return copy_plane_u10_to_u12;
            case 16: return copy_plane_u10_to_u16;
            default: break;
            }
            break;
        case 12:
            switch (bit_depth_to) {
            case 8:  return copy_plane_u12_to_u8;
            case 10: return copy_plane_u12_to_u10;
            case 12: return copy_plane_u12_to_u12;
            case 16: return copy_plane_u12_to_u16;
            default: break;
            }
            break;
        case 16:
            switch (bit_depth_to) {
            case 8:  return copy_plane_u16_to_u8;
            case 10: return copy_plane_u16_to_u10;
            case 12: return copy_plane_u16_to_u12;
            case 16: return copy_plane_u16_to_u16;
            default: break;
            }
            break;
        default:
            break;
        }
    }
    return nullptr;
}

const TCHAR *get_simd_str(RGY_SIMD simd) {
    static std::vector<std::pair<RGY_SIMD, const TCHAR*>> simd_str_list = {
        { AVX2,  _T("AVX2")   },
        { AVX,   _T("AVX")    },
        { SSE42, _T("SSE4.2") },
        { SSE41, _T("SSE4.1") },
        { SSSE3, _T("SSSE3")  },
        { SSE2,  _T("SSE2")   }
    };
    for (const auto& simd_str : simd_str_list) {
        if ((simd & simd_str.first) == simd_str.first) {
            return simd_str.second;
        }
    }
    return _T("-");
}



const TCHAR *rgb_order_str(RGY_RGB_ORDER order, bool include_alpha) {
    if (!include_alpha) {
        switch (order) {
        case RGY_RGB_ORDER_RGBA: order = RGY_RGB_ORDER_RGB; break;
        case RGY_RGB_ORDER_BGRA: order = RGY_RGB_ORDER_BGR; break;
        case RGY_RGB_ORDER_GBRA: order = RGY_RGB_ORDER_GBR; break;
        case RGY_RGB_ORDER_RBGA: order = RGY_RGB_ORDER_RBG; break;
        case RGY_RGB_ORDER_ARGB: return nullptr;
        case RGY_RGB_ORDER_ABGR: return nullptr;
        case RGY_RGB_ORDER_AGBR: return nullptr;
        default: break;
        }
    }
    switch (order) {
    case RGY_RGB_ORDER_RGB: return _T("rgb");
    case RGY_RGB_ORDER_BGR: return _T("bgr");
    case RGY_RGB_ORDER_GBR: return _T("gbr");
    case RGY_RGB_ORDER_RBG: return _T("rbg");
    case RGY_RGB_ORDER_RGBA: return _T("rgba");
    case RGY_RGB_ORDER_BGRA: return _T("bgra");
    case RGY_RGB_ORDER_GBRA: return _T("gbra");
    case RGY_RGB_ORDER_RBGA: return _T("rbga");
    case RGY_RGB_ORDER_ARGB: return _T("argb");
    case RGY_RGB_ORDER_ABGR: return _T("abgr");
    case RGY_RGB_ORDER_AGBR: return _T("agbr");
    default: return nullptr;
    }
}
