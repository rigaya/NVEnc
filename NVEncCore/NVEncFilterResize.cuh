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

#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#pragma warning (pop)
#include <memory>
#include <vector>
#include <cuda.h>
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

template<typename Type, int bit_depth>
__global__ void kernel_resize_texture_bilinear(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const float ratioX, const float ratioY) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;
    
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(tex2D(SRC_TEXTURE_LINEAR, x * ratioX, y * ratioY) * (float)(1<<bit_depth));
    }
}

template<typename Type, int bit_depth>
void resize_texture_bilinear(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight, const float ratioX, const float ratioY) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_resize_texture_bilinear<Type, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, ratioX, ratioY);
}

template<typename Type, int bit_depth>
static cudaError_t resize_texture_bilinear_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    SRC_TEXTURE_LINEAR.filterMode = cudaFilterModeLinear;
    SRC_TEXTURE_LINEAR.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE_LINEAR.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE_LINEAR.normalized     = true;
    SRC_TEXTURE_LINEAR.channelDesc    = cudaCreateChannelDesc<Type>();
    float ratioX = 1.0f / (float)(pOutputFrame->width);
    float ratioY = 1.0f / (float)(pOutputFrame->height);
    //Y
    auto cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        pInputFrame->ptr,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width >> 1, pInputFrame->height >> 1, pInputFrame->pitch);
    ratioX = 1.0f / (float)(pOutputFrame->width >> 1);
    ratioY = 1.0f / (float)(pOutputFrame->height >> 1);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width >> 1, pInputFrame->height >> 1, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth>
static cudaError_t resize_texture_bilinear_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    SRC_TEXTURE_LINEAR.filterMode = cudaFilterModeLinear;
    SRC_TEXTURE_LINEAR.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE_LINEAR.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE_LINEAR.normalized     = true;
    SRC_TEXTURE_LINEAR.channelDesc    = cudaCreateChannelDesc<Type>();
    const float ratioX = 1.0f / (float)(pOutputFrame->width);
    const float ratioY = 1.0f / (float)(pOutputFrame->height);
    //Y
    auto cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        pInputFrame->ptr,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE_LINEAR,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE_LINEAR);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}
