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

static __device__ float lerpf(float a, float b, float c) {
    return a + (b - a) * c;
}

template<typename Type, int knn_radius, int bit_depth>
__global__ void kernel_denoise_knn(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    const float knn_window_area = (float)((2 * knn_radius + 1) * (2 * knn_radius + 1));
    const float inv_knn_window_area = 1.0f / knn_window_area;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        //Add half of a texel to always address exact texel centers
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        float fCount = 0.0f;
        float sumWeights = 0.0f;
        float sum = 0.0f;
        float center = (float)tex2D(SRC_TEXTURE, x, y) * (1.0f / (1<<bit_depth));

        //Cycle through KNN window, surrounding (x, y) texel
        for (float i = -knn_radius; i <= knn_radius; i++) {
            for (float j = -knn_radius; j <= knn_radius; j++) {
                float clrIJ = (float)tex2D(SRC_TEXTURE, x + j, y + i) * (1.0f / (1<<bit_depth));
                float distanceIJ = (center - clrIJ) * (center - clrIJ);

                //Derive final weight from color distance
                float weightIJ = __expf(-(distanceIJ * strength + (i * i + j * j) * inv_knn_window_area));

                //Accumulate (x + j, y + i) texel color with computed weight
                sum += clrIJ * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > weight_threshold) ? inv_knn_window_area : 0;
            }
        }
        float lerpQ = (fCount > lerp_threshold) ? lerpC : 1.0f - lerpC;
    
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(lerpf(sum * __frcp_rn(sumWeights), center, lerpQ) * (1<<bit_depth));
    }
}

template<typename Type, int bit_depth>
void denoise_knn(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    dim3 blockSize(64, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    switch (radius) {
    case 1:
        kernel_denoise_knn<Type, 1, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 2:
        kernel_denoise_knn<Type, 2, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 3:
        kernel_denoise_knn<Type, 3, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 4:
        kernel_denoise_knn<Type, 4, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 5:
        //よりレジスタを使うので、ブロック当たりのスレッド数を低減
        blockSize = dim3(32, 16);
        gridSize = dim3(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
        kernel_denoise_knn<Type, 5, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    default:
        break;
    }
}

template<typename Type, int bit_depth>
static cudaError_t denoise_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    SRC_TEXTURE.filterMode = cudaFilterModePoint;
    SRC_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();
    //Y
    auto cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        pInputFrame->ptr,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width >> 1, pInputFrame->height >> 1, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width >> 1, pInputFrame->height >> 1, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth>
static cudaError_t denoise_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    SRC_TEXTURE.filterMode = cudaFilterModePoint;
    SRC_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();
    //Y
    auto cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        pInputFrame->ptr,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2,
        cudaCreateChannelDesc<Type>(),
        pInputFrame->width, pInputFrame->height, pInputFrame->pitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        radius, strength, lerpC, weight_threshold, lerp_threshold);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaUnbindTexture(SRC_TEXTURE);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}
