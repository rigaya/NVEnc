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
#include <array>
#include "ConvertCsp.h"
#include "NVEncFilterDenoisePmd.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#define SRC_TEXTURE g_texImageU16Src
#define GRF_TEXTURE g_texImageU16Grf

texture<uint16_t, cudaTextureType2D, cudaReadModeElementType> SRC_TEXTURE;
texture<uint16_t, cudaTextureType2D, cudaReadModeElementType> GRF_TEXTURE;

#include "NVEncFilterDenoisePmd.cuh"

cudaError_t denoise_yv12_u10(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 10, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u10_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 10, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u10(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 10, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u10_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 10, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 12, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u12_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 12, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 12, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u12_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 12, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u14(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 14, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u14_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 14, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u14(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 14, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u14_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 14, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u16(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 16, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yv12_u16_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yv12<uint16_t, 16, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u16(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 16, false>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

cudaError_t denoise_yuv444_u16_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    return denoise_yuv444<uint16_t, 16, true>(pOutputFrame, pGauss, pInputFrame, loop_count, strength, threshold);
}

