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

static __device__ float pmd_exp(float x, float strength2, float inv_threshold2) {
    return strength2 * __expf(-x*x * inv_threshold2);
}

static __device__ float pmd(float x, float strength2, float inv_threshold2) {
    const float range = 4.0f;
    return strength2 * __frcp_rn(1.0f + (x*x * inv_threshold2));
}

template<typename Type, int bit_depth>
__global__ void kernel_create_gauss(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight) {
    static const float weight[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f - 2.0f;
        const float y = (float)iy + 0.5f - 2.0f;
        float sum = 0.0f;
        for (int j = 0; j < 5; j++) {
            float sum_line = 0.0f;
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                sum_line += (float)tex2D(SRC_TEXTURE, x + (float)i, y + (float)j) * weight[i];
            }
            sum += sum_line * weight[j];
        }
        pDst += dstPitch * iy + ix * sizeof(Type);
        pDst[0] = (Type)(sum + 0.5f);
    }
}

template<typename Type, int bit_depth, bool useExp>
__global__ void kernel_denoise_pmd(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const float strength2, const float inv_threshold2) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;
        float clr   = tex2D(SRC_TEXTURE, x+0, y+0);
        float clrym = tex2D(SRC_TEXTURE, x+0, y-1);
        float clryp = tex2D(SRC_TEXTURE, x+0, y+1);
        float clrxm = tex2D(SRC_TEXTURE, x-1, y+0);
        float clrxp = tex2D(SRC_TEXTURE, x+1, y+0);
        float grf   = tex2D(GRF_TEXTURE, x+0, y+0);
        float grfym = tex2D(GRF_TEXTURE, x+0, y-1);
        float grfyp = tex2D(GRF_TEXTURE, x+0, y+1);
        float grfxm = tex2D(GRF_TEXTURE, x-1, y+0);
        float grfxp = tex2D(GRF_TEXTURE, x+1, y+0);
        clr += (useExp)
            ? (clrym - clr) * pmd_exp(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd_exp(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd_exp(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd_exp(grfxp - grf, strength2, inv_threshold2)
            : (clrym - clr) * pmd(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd(grfxp - grf, strength2, inv_threshold2);

        pDst += dstPitch * iy + ix * sizeof(Type);
        pDst[0] = (Type)(clamp(clr + 0.5f, 0.0f, (float)(1<<bit_depth)-0.1f));
#undef BIT_DEPTH_FROM_12
    }
}

template<typename Type, int bit_depth, bool useExp>
cudaError_t denoise_knn(uint8_t *pDst[2], uint8_t *pGauss, const int dstPitch, const int dstWidth, const int dstHeight,
    uint8_t *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    int loop_count, const float strength, const float threshold) {
    const float range = 4.0f;
    const float strength2 = strength / (range * 100.0f);
    const float threshold2 = std::pow(2.0f, threshold / 10.0f - (12 - bit_depth) * 2.0f);
    const float inv_threshold2 = 1.0f / threshold2;

    auto cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
        pSrc,
        cudaCreateChannelDesc<Type>(),
        srcWidth, srcHeight, srcPitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    dim3 blockSize(32, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_create_gauss<Type, bit_depth><<<gridSize, blockSize>>>(
        pGauss,
        dstPitch, dstWidth, dstHeight);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaBindTexture2D(0, GRF_TEXTURE,
        pGauss,
        cudaCreateChannelDesc<Type>(),
        dstWidth, dstHeight, dstPitch);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    int dst_index = 0;
    for (int i = 0; i < loop_count; i++) {
        dst_index = i & 1;
        kernel_denoise_pmd<Type, bit_depth, useExp><<<gridSize, blockSize>>>(pDst[dst_index],
            dstPitch, dstWidth, dstHeight, strength2, inv_threshold2);
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
        cudaerr = cudaUnbindTexture(SRC_TEXTURE);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
        if (i < loop_count-1) {
            cudaerr = cudaBindTexture2D(0, SRC_TEXTURE,
                pDst[dst_index],
                cudaCreateChannelDesc<Type>(),
                dstWidth, dstHeight, dstPitch);
            if (cudaerr != cudaSuccess) {
                return cudaerr;
            }
        }
    }
    return cudaSuccess;
}

template<typename Type, int bit_depth, bool useExp>
static cudaError_t denoise_yv12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    SRC_TEXTURE.filterMode = cudaFilterModePoint;
    SRC_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();

    GRF_TEXTURE.filterMode = cudaFilterModePoint;
    GRF_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    GRF_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    GRF_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();

    uint8_t *pDst[2] = { 0 };
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr;
    //Y
    auto cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width, pOutputFrame[0]->height,
        (uint8_t *)pInputFrame->ptr,
        pInputFrame->pitch, pInputFrame->width, pInputFrame->height,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr + pOutputFrame[0]->pitch * pOutputFrame[0]->height;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr + pOutputFrame[1]->pitch * pOutputFrame[1]->height;
    cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr + pGauss->pitch * pGauss->height,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width >> 1, pOutputFrame[0]->height >> 1,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        pInputFrame->pitch, pInputFrame->width >> 1, pInputFrame->height >> 1,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr + pOutputFrame[0]->pitch * pOutputFrame[0]->height * 3 / 2;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr + pOutputFrame[1]->pitch * pOutputFrame[1]->height * 3 / 2;
    cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr + pGauss->pitch * pGauss->height * 3 / 2,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width >> 1, pOutputFrame[0]->height >> 1,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2,
        pInputFrame->pitch, pInputFrame->width >> 1, pInputFrame->height >> 1,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth, bool useExp>
static cudaError_t denoise_yuv444(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    SRC_TEXTURE.filterMode = cudaFilterModePoint;
    SRC_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    SRC_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    SRC_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();

    GRF_TEXTURE.filterMode = cudaFilterModePoint;
    GRF_TEXTURE.addressMode[0] = cudaAddressModeClamp;
    GRF_TEXTURE.addressMode[1] = cudaAddressModeClamp;
    GRF_TEXTURE.channelDesc = cudaCreateChannelDesc<Type>();

    uint8_t *pDst[2] = { 0 };
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr;
    //Y
    auto cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width, pOutputFrame[0]->height,
        (uint8_t *)pInputFrame->ptr,
        pInputFrame->pitch, pInputFrame->width, pInputFrame->height,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr + pOutputFrame[0]->pitch * pOutputFrame[0]->height;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr + pOutputFrame[1]->pitch * pOutputFrame[1]->height;
    cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr + pGauss->pitch * pGauss->height,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width, pOutputFrame[0]->height,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height,
        pInputFrame->pitch, pInputFrame->width, pInputFrame->height,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr + pOutputFrame[0]->pitch * pOutputFrame[0]->height * 2;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr + pOutputFrame[1]->pitch * pOutputFrame[1]->height * 2;
    cudaerr = denoise_knn<Type, bit_depth, useExp>(
        pDst,
        (uint8_t *)pGauss->ptr + pGauss->pitch * pGauss->height * 2,
        pOutputFrame[0]->pitch, pOutputFrame[0]->width, pOutputFrame[0]->height,
        (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2,
        pInputFrame->pitch, pInputFrame->width, pInputFrame->height,
        loop_count, strength, threshold);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}
