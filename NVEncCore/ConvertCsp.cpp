// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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

#include <cstdint>
#include <vector>
#include <tchar.h>
#include "ConvertCSP.h"

void convert_yuy2_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yuy2_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_uv_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_uv_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_uv_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_p010_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_16_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_16_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_14_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_14_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_10_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_10_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_09_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_09_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_16_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_16_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_14_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_14_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_12_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_12_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_10_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_10_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_09_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_09_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void copy_yuv444_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void copy_yuv444_to_yuv444_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yuv444_16_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuv444_14_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuv444_12_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuv444_10_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuv444_09_to_yuv444_16_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yc48_to_yuv444_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yc48_to_p010_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_i_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_i_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_i_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_p010_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yc48_to_yuv444_16bit_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_16bit_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_16bit_sse41(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_16bit_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yc48_to_yuv444_16bit_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

//適当。
#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
void convert_yuy2_to_nv12(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    void *dst = dst_array[0];
    const void *src = src_array[0];
    uint8_t *srcFrame = (uint8_t *)src;
    uint8_t *dstYFrame = (uint8_t *)dst;
    uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
    const int y_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < y_fin; y += 2) {
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
void convert_yuy2_to_nv12_i(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    void *dst = dst_array[0];
    const void *src = src_array[0];
    uint8_t *srcFrame = (uint8_t *)src;
    uint8_t *dstYFrame = (uint8_t *)dst;
    uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
    const int y_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < y_fin; y += 4) {
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

void convert_yuv444_to_nv12(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    //Y成分のコピー
    if (true) {
        uint8_t *srcYLine = (uint8_t *)src_array[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst_array[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src_array[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *srcVLine = (uint8_t *)src_array[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *dstLine = (uint8_t *)dst_array[1];
    int uv_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < uv_fin; y += 2, srcULine += src_uv_pitch_byte * 2, srcVLine += src_uv_pitch_byte * 2, dstLine += dst_y_pitch_byte) {
        uint8_t *srcU = srcULine;
        uint8_t *srcV = srcVLine;
        uint8_t *dstC = dstLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcU += 2, srcV += 2) {
            uint32_t uy0 = srcU[0 * src_uv_pitch_byte];
            uint32_t uy1 = srcU[1 * src_uv_pitch_byte];
            uint32_t vy0 = srcV[0 * src_uv_pitch_byte];
            uint32_t vy1 = srcV[1 * src_uv_pitch_byte];
            dstC[0] = (uint8_t)((uy0 + uy1 + 1) >> 1);
            dstC[1] = (uint8_t)((vy0 + vy1 + 1) >> 1);
        }
    }
}

//これも適当。
void convert_yuv444_to_nv12_i(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    int crop_left   = crop[0];
    int crop_up     = crop[1];
    int crop_right  = crop[2];
    int crop_bottom = crop[3];
    //Y成分のコピー
    if (true) {
        uint8_t *srcYLine = (uint8_t *)src_array[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst_array[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src_array[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *srcVLine = (uint8_t *)src_array[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *dstLine = (uint8_t *)dst_array[1];
    int uv_fin = height - crop_bottom - crop_up;
    for (int y = 0; y < uv_fin; y += 4, srcULine += src_uv_pitch_byte * 4, srcVLine += src_uv_pitch_byte * 4, dstLine += dst_y_pitch_byte * 2) {
        uint8_t *srcU = srcULine;
        uint8_t *srcV = srcVLine;
        uint8_t *dstC = dstLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, dstC += 2, srcU += 2, srcV += 2) {
            uint32_t uy0 = srcU[0 * src_uv_pitch_byte];
            uint32_t uy1 = srcU[1 * src_uv_pitch_byte];
            uint32_t uy2 = srcU[2 * src_uv_pitch_byte];
            uint32_t uy3 = srcU[3 * src_uv_pitch_byte];
            uint32_t vy0 = srcV[0 * src_uv_pitch_byte];
            uint32_t vy1 = srcV[1 * src_uv_pitch_byte];
            uint32_t vy2 = srcV[2 * src_uv_pitch_byte];
            uint32_t vy3 = srcV[3 * src_uv_pitch_byte];

            dstC[0 * dst_y_pitch_byte + 0] = (uint8_t)((uy0 * 3 + uy2 * 1 + 2) >> 2);
            dstC[0 * dst_y_pitch_byte + 1] = (uint8_t)((vy0 * 3 + vy2 * 1 + 2) >> 2);

            dstC[1 * dst_y_pitch_byte + 0] = (uint8_t)((uy1 * 1 + uy3 * 3 + 2) >> 2);
            dstC[1 * dst_y_pitch_byte + 1] = (uint8_t)((vy1 * 1 + vy3 * 3 + 2) >> 2);
        }
    }
}

static void convert_yuv422_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (true) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    for (int ic = 1; ic < 3; ic++) {
        uint8_t *srcCLine = (uint8_t *)src[ic] + src_uv_pitch_byte * crop_up + (crop_left >> 1);
        uint8_t *dstLine = (uint8_t *)dst[ic];
        const int uv_fin = height - crop_bottom - crop_up;
        for (int y = 0; y < uv_fin; y++, srcCLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
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

static void convert_yuy2_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    uint8_t *srcLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
    uint8_t *dstYLine = (uint8_t *)dst[0];
    uint8_t *dstULine = (uint8_t *)dst[1];
    uint8_t *dstVLine = (uint8_t *)dst[2];
    const int y_fin = height - crop_bottom;
    const int y_width = width - crop_right - crop_left;
    for (int y = 0; y < y_fin; y++, srcLine += src_y_pitch_byte, dstYLine += dst_y_pitch_byte, dstULine += dst_y_pitch_byte, dstVLine += dst_y_pitch_byte) {
        uint8_t *srcP = srcLine;
        uint8_t *dstY = dstYLine;
        uint8_t *dstU = dstULine;
        uint8_t *dstV = dstVLine;
        const int x_fin = width - crop_right - crop_left;
        for (int x = 0; x < x_fin; x += 2, srcP += 4, dstY += 2, dstU += 2, dstV += 2) {
            int cxplus = (x + 2 < x_fin) << 1;
            dstY[0] = srcP[0];
            dstY[1] = srcP[2];

            int vx0 = srcP[1];
            int ux0 = srcP[3];
            int vx2 = srcP[1+cxplus];
            int ux2 = srcP[3+cxplus];
            dstU[0] = (uint8_t)ux0;
            dstU[1] = (uint8_t)((ux0 + ux2 + 1) >> 1);
            dstV[0] = (uint8_t)vx0;
            dstV[1] = (uint8_t)((vx2 + vx2 + 1) >> 1);
        }
    }
}

template<bool uv_only>
static void __forceinline convert_yv12_p_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    for (int ic = 1; ic < 3; ic++) {
        uint8_t *srcCLine = (uint8_t *)src[ic] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
        uint8_t *dstLine = (uint8_t *)dst[ic];
        const int uv_fin = height - crop_bottom - crop_up;
        for (int y = 0; y < uv_fin; y += 2, srcCLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte * 2) {
            uint8_t *dstC = dstLine;
            uint8_t *srcP = srcCLine;
            const int x_fin = width - crop_right - crop_left;
            if (y == 0) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[ 1*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[ 0*src_uv_pitch_byte + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch_byte + cxplus];
                    int cy4x1 = srcP[ 1*src_uv_pitch_byte + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2) >> 2;
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2) >> 2;
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2) >> 2;
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2) >> 2;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else if (y == height-1) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[-1*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[-1*src_uv_pitch_byte + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch_byte + cxplus];
                    int cy4x1 = srcP[ 0*src_uv_pitch_byte + cxplus];

                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2) >> 2;
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2) >> 2;
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2) >> 2;
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2) >> 2;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cxplus = (x + 2 < x_fin);
                    int cy0x0 = srcP[-1*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[ 1*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[-1*src_uv_pitch_byte + cxplus];
                    int cy2x1 = srcP[ 0*src_uv_pitch_byte + cxplus];
                    int cy4x1 = srcP[ 1*src_uv_pitch_byte + cxplus];
                    
                    int cy1x0 = (cy0x0 * 1 + cy2x0 * 3 + 2) >> 2;
                    int cy3x0 = (cy2x0 * 3 + cy4x0 * 1 + 2) >> 2;
                    int cy1x1 = (cy0x1 * 1 + cy2x1 * 3 + 2) >> 2;
                    int cy3x1 = (cy2x1 * 3 + cy4x1 * 1 + 2) >> 2;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            }
        }
    }
}

template<bool uv_only>
static void __forceinline convert_yv12_i_to_yuv444_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            memcpy(dstLine, srcYLine, y_width);
        }
    }
    //UV成分のコピー
    for (int ic = 1; ic < 3; ic++) {
        uint8_t *srcCLine = (uint8_t *)src[ic] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
        uint8_t *dstLine = (uint8_t *)dst[ic];
        int uv_fin = height - crop_bottom - crop_up;
        for (int y = 0; y < uv_fin; y += 4, srcCLine += src_uv_pitch_byte * 2, dstLine += dst_y_pitch_byte * 4) {
            uint8_t *dstC = dstLine;
            uint8_t *srcP = srcCLine;
            const int x_fin = width - crop_right - crop_left;
            if (y <= 1) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cy0x0 = srcP[0*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[1*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[0*src_uv_pitch_byte + 0];
                    int cy6x0 = srcP[1*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[0*src_uv_pitch_byte + 1];
                    int cy2x1 = srcP[1*src_uv_pitch_byte + 1];
                    int cy4x1 = srcP[0*src_uv_pitch_byte + 1];
                    int cy6x1 = srcP[1*src_uv_pitch_byte + 1];

                    int cy1x0 = (cy0x0 * 1 + cy4x0 * 7 + 4) >> 3;
                    int cy3x0 = (cy2x0 * 3 + cy6x0 * 5 + 4) >> 3;
                    int cy1x1 = (cy0x1 * 1 + cy4x1 * 7 + 4) >> 3;
                    int cy3x1 = (cy2x1 * 3 + cy6x1 * 5 + 4) >> 3;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else if (y == height-1) {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cy0x0 = srcP[-2*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[-1*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy6x0 = srcP[-1*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[-2*src_uv_pitch_byte + 1];
                    int cy2x1 = srcP[-1*src_uv_pitch_byte + 1];
                    int cy4x1 = srcP[ 0*src_uv_pitch_byte + 1];
                    int cy6x1 = srcP[-1*src_uv_pitch_byte + 1];

                    int cy1x0 = (cy0x0 * 1 + cy4x0 * 7 + 4) >> 3;
                    int cy3x0 = (cy2x0 * 3 + cy6x0 * 5 + 4) >> 3;
                    int cy1x1 = (cy0x1 * 1 + cy4x1 * 7 + 4) >> 3;
                    int cy3x1 = (cy2x1 * 3 + cy6x1 * 5 + 4) >> 3;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            } else {
                for (int x = 0; x < x_fin; x += 2, dstC += 2, srcP++) {
                    int cy0x0 = srcP[-2*src_uv_pitch_byte + 0];
                    int cy2x0 = srcP[-1*src_uv_pitch_byte + 0];
                    int cy4x0 = srcP[ 0*src_uv_pitch_byte + 0];
                    int cy6x0 = srcP[ 1*src_uv_pitch_byte + 0];
                    int cy0x1 = srcP[-2*src_uv_pitch_byte + 1];
                    int cy2x1 = srcP[-1*src_uv_pitch_byte + 1];
                    int cy4x1 = srcP[ 0*src_uv_pitch_byte + 1];
                    int cy6x1 = srcP[ 1*src_uv_pitch_byte + 1];

                    int cy1x0 = (cy0x0 * 1 + cy4x0 * 7 + 4) >> 3;
                    int cy3x0 = (cy2x0 * 3 + cy6x0 * 5 + 4) >> 3;
                    int cy1x1 = (cy0x1 * 1 + cy4x1 * 7 + 4) >> 3;
                    int cy3x1 = (cy2x1 * 3 + cy6x1 * 5 + 4) >> 3;

                    dstC[0*dst_y_pitch_byte   + 0] = (uint8_t)cy1x0;
                    dstC[0*dst_y_pitch_byte   + 1] = (uint8_t)((cy1x0 + cy1x1 + 1) >> 1);
                    dstC[1*dst_y_pitch_byte   + 0] = (uint8_t)cy3x0;
                    dstC[1*dst_y_pitch_byte   + 1] = (uint8_t)((cy3x0 + cy3x1 + 1) >> 1);
                }
            }
        }
    }
}
#pragma warning (pop)

static void convert_yv12_p_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_p_to_yuv444_c<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

static void convert_yv12_i_to_yuv444(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_i_to_yuv444_c<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<bool uv_only>
static void convert_yv12_to_p010_c(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    const int crop_left   = crop[0];
    const int crop_up     = crop[1];
    const int crop_right  = crop[2];
    const int crop_bottom = crop[3];
    //Y成分のコピー
    if (!uv_only) {
        uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
        uint8_t *dstLine  = (uint8_t *)dst[0];
        const int y_fin = height - crop_bottom;
        const int y_width = width - crop_right - crop_left;
        for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
            uint16_t *dst_ptr = (uint16_t *)dstLine;
            for (int x = 0; x < y_width; x++) {
                dst_ptr[x] = (uint16_t)((((uint32_t)srcYLine[x]) << 8) + (2 << 6));
            }
        }
    }
    //UV成分のコピー
    uint8_t *srcULine = (uint8_t *)src[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *srcVLine = (uint8_t *)src[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
    uint8_t *dstLine  = (uint8_t *)dst[1];
    const int uv_fin = (height - crop_bottom) >> 1;
    for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
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

static void convert_yv12_to_p010(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    convert_yv12_to_p010_c<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

#pragma warning (pop)

enum {
    NONE  = 0x0000,
    SSE2  = 0x0001,
    SSE3  = 0x0002, //使用していない
    SSSE3 = 0x0004,
    SSE41 = 0x0008,
    SSE42 = 0x0010, //使用していない
    AVX   = 0x0020,
    AVX2  = 0x0040,
};

static const ConvertCSP funcList[] = {
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_NV12,      false, { convert_yuy2_to_nv12_avx2,           convert_yuy2_to_nv12_i_avx2         }, AVX2|AVX },
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_NV12,      false, { convert_yuy2_to_nv12_avx,            convert_yuy2_to_nv12_i_avx          }, AVX },
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_NV12,      false, { convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_ssse3        }, SSSE3|SSE2 },
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_NV12,      false, { convert_yuy2_to_nv12_sse2,           convert_yuy2_to_nv12_i_sse2         }, SSE2 },
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_NV12,      false, { convert_yuy2_to_nv12,                convert_yuy2_to_nv12                }, NONE },
    { NV_ENC_CSP_YUY2,      NV_ENC_CSP_YUV444,    false, { convert_yuy2_to_yuv444,              convert_yuy2_to_yuv444              }, NONE },
#if NVENC_AUO
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444,    false, { convert_yc48_to_yuv444_avx,          convert_yc48_to_yuv444_avx          }, AVX },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444,    false, { convert_yc48_to_yuv444_sse41,        convert_yc48_to_yuv444_sse41        }, SSE41|SSSE3|SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444,    false, { convert_yc48_to_yuv444_sse2,         convert_yc48_to_yuv444_sse2         }, SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_P010,      false, { convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_avx2         }, AVX2|AVX },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_P010,      false, { convert_yc48_to_p010_avx,            convert_yc48_to_p010_i_avx          }, AVX },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_P010,      false, { convert_yc48_to_p010_sse41,          convert_yc48_to_p010_i_sse41        }, SSE41|SSSE3|SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_P010,      false, { convert_yc48_to_p010_ssse3,          convert_yc48_to_p010_i_ssse3        }, SSSE3|SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_P010,      false, { convert_yc48_to_p010_sse2,           convert_yc48_to_p010_i_sse2         }, SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444_16, false, { convert_yc48_to_yuv444_16bit_avx2,   convert_yc48_to_yuv444_16bit_avx2   }, AVX2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444_16, false, { convert_yc48_to_yuv444_16bit_avx,    convert_yc48_to_yuv444_16bit_avx    }, AVX },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444_16, false, { convert_yc48_to_yuv444_16bit_sse41,  convert_yc48_to_yuv444_16bit_sse41  }, SSE41|SSSE3|SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444_16, false, { convert_yc48_to_yuv444_16bit_ssse3,  convert_yc48_to_yuv444_16bit_ssse3  }, SSSE3|SSE2 },
    { NV_ENC_CSP_YC48,      NV_ENC_CSP_YUV444_16, false, { convert_yc48_to_yuv444_16bit_sse2,   convert_yc48_to_yuv444_16bit_sse2   }, SSE2 },
#else
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_NV12,      false, { convert_yv12_to_nv12_avx2,           convert_yv12_to_nv12_avx2           }, AVX2|AVX },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_NV12,      false, { convert_yv12_to_nv12_avx,            convert_yv12_to_nv12_avx            }, AVX },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_NV12,      false, { convert_yv12_to_nv12_sse2,           convert_yv12_to_nv12_sse2           }, SSE2 },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_YUV444,    false, { convert_yv12_p_to_yuv444,            convert_yv12_i_to_yuv444            }, NONE },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_P010,      false, { convert_yv12_to_p010_avx2,           convert_yv12_to_p010_avx2           }, AVX2|AVX },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_P010,      false, { convert_yv12_to_p010_avx,            convert_yv12_to_p010_avx            }, AVX },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_P010,      false, { convert_yv12_to_p010_sse2,           convert_yv12_to_p010_sse2           }, SSE2 },
    { NV_ENC_CSP_YV12,      NV_ENC_CSP_P010,      false, { convert_yv12_to_p010,                convert_yv12_to_p010                }, NONE },
    { NV_ENC_CSP_YV12_16,   NV_ENC_CSP_NV12,      false, { convert_yv12_16_to_nv12_avx2,        convert_yv12_16_to_nv12_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_16,   NV_ENC_CSP_NV12,      false, { convert_yv12_16_to_nv12_sse2,        convert_yv12_16_to_nv12_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_14,   NV_ENC_CSP_NV12,      false, { convert_yv12_14_to_nv12_avx2,        convert_yv12_14_to_nv12_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_14,   NV_ENC_CSP_NV12,      false, { convert_yv12_14_to_nv12_sse2,        convert_yv12_14_to_nv12_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_12,   NV_ENC_CSP_NV12,      false, { convert_yv12_12_to_nv12_avx2,        convert_yv12_12_to_nv12_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_12,   NV_ENC_CSP_NV12,      false, { convert_yv12_12_to_nv12_sse2,        convert_yv12_12_to_nv12_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_10,   NV_ENC_CSP_NV12,      false, { convert_yv12_10_to_nv12_avx2,        convert_yv12_10_to_nv12_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_10,   NV_ENC_CSP_NV12,      false, { convert_yv12_10_to_nv12_sse2,        convert_yv12_10_to_nv12_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_09,   NV_ENC_CSP_NV12,      false, { convert_yv12_09_to_nv12_avx2,        convert_yv12_09_to_nv12_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_09,   NV_ENC_CSP_NV12,      false, { convert_yv12_09_to_nv12_sse2,        convert_yv12_09_to_nv12_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_16,   NV_ENC_CSP_P010,      false, { convert_yv12_16_to_p010_avx2,        convert_yv12_16_to_p010_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_16,   NV_ENC_CSP_P010,      false, { convert_yv12_16_to_p010_sse2,        convert_yv12_16_to_p010_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_14,   NV_ENC_CSP_P010,      false, { convert_yv12_14_to_p010_avx2,        convert_yv12_14_to_p010_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_14,   NV_ENC_CSP_P010,      false, { convert_yv12_14_to_p010_sse2,        convert_yv12_14_to_p010_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_12,   NV_ENC_CSP_P010,      false, { convert_yv12_12_to_p010_avx2,        convert_yv12_12_to_p010_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_12,   NV_ENC_CSP_P010,      false, { convert_yv12_12_to_p010_sse2,        convert_yv12_12_to_p010_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_10,   NV_ENC_CSP_P010,      false, { convert_yv12_10_to_p010_sse2,        convert_yv12_10_to_p010_sse2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_10,   NV_ENC_CSP_P010,      false, { convert_yv12_10_to_p010_sse2,        convert_yv12_10_to_p010_sse2        }, SSE2 },
    { NV_ENC_CSP_YV12_09,   NV_ENC_CSP_P010,      false, { convert_yv12_09_to_p010_avx2,        convert_yv12_09_to_p010_avx2        }, AVX2|AVX },
    { NV_ENC_CSP_YV12_09,   NV_ENC_CSP_P010,      false, { convert_yv12_09_to_p010_sse2,        convert_yv12_09_to_p010_sse2        }, SSE2 },
    { NV_ENC_CSP_YUV422,    NV_ENC_CSP_YUV444,    false, { convert_yuv422_to_yuv444,            convert_yuv422_to_yuv444            }, NONE },
    { NV_ENC_CSP_YUV444,    NV_ENC_CSP_NV12,      false, { convert_yuv444_to_nv12,              convert_yuv444_to_nv12_i            }, NONE },
    { NV_ENC_CSP_YUV444,    NV_ENC_CSP_YUV444,    false, { copy_yuv444_to_yuv444_avx2,          copy_yuv444_to_yuv444_avx2          }, AVX2|AVX },
    { NV_ENC_CSP_YUV444,    NV_ENC_CSP_YUV444,    false, { copy_yuv444_to_yuv444_sse2,          copy_yuv444_to_yuv444_sse2          }, SSE2 },
    { NV_ENC_CSP_YUV444_16, NV_ENC_CSP_YUV444_16, false, { convert_yuv444_16_to_yuv444_16_sse2, convert_yuv444_16_to_yuv444_16_sse2 }, SSE2 },
    { NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16, false, { convert_yuv444_14_to_yuv444_16_sse2, convert_yuv444_14_to_yuv444_16_sse2 }, SSE2 },
    { NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_16, false, { convert_yuv444_12_to_yuv444_16_sse2, convert_yuv444_12_to_yuv444_16_sse2 }, SSE2 },
    { NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_16, false, { convert_yuv444_10_to_yuv444_16_sse2, convert_yuv444_10_to_yuv444_16_sse2 }, SSE2 },
    { NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_16, false, { convert_yuv444_09_to_yuv444_16_sse2, convert_yuv444_09_to_yuv444_16_sse2 }, SSE2 },
#endif
};

static uint32_t nvenc_get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    uint32_t simd = NONE;
    if  (CPUInfo[3] & 0x04000000)
        simd |= SSE2;
    if  (CPUInfo[2] & 0x00000001)
        simd |= SSE3;
    if  (CPUInfo[2] & 0x00000200)
        simd |= SSSE3;
    if  (CPUInfo[2] & 0x00080000)
        simd |= SSE41;
    if  (CPUInfo[2] & 0x00100000)
        simd |= SSE42;
    uint64_t XGETBV = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        XGETBV = _xgetbv(0);
        if ((XGETBV & 0x06) == 0x06)
            simd |= AVX;
    }
    __cpuid(CPUInfo, 7);
    if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
        simd |= AVX2;
    return simd;
}

const ConvertCSP* get_convert_csp_func(NV_ENC_CSP csp_from, NV_ENC_CSP csp_to, bool uv_only) {
    uint32_t availableSIMD = nvenc_get_availableSIMD();
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
    static std::vector<std::pair<uint32_t, TCHAR*>> simd_str_list = {
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
