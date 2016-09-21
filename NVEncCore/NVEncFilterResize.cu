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

#include <map>
#include <array>
#include "ConvertCsp.h"
#include "NVEncFilter.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

template<typename Type, int bit_depth>
__global__ void kernel_resize_texture_bilinear(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texObj,
    const float ratioX, const float ratioY) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(tex2D<float>(texObj, x * ratioX, y * ratioY) * (float)(1<<bit_depth));
    }
}

template<typename Type, int bit_depth>
void resize_texture_bilinear(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight, cudaTextureObject_t texObj, const float ratioX, const float ratioY) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_resize_texture_bilinear<Type, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texObj, ratioX, ratioY);
}

#pragma warning(push)
#pragma warning(disable:4100)
template<typename Type, int bit_depth>
cudaError_t resize_texture_bilinear_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const float *pgDummy) {
    //Y
    float ratioX = 1.0f / (float)(pOutputFrame->width);
    float ratioY = 1.0f / (float)(pOutputFrame->height);
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDesc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDesc.res.pitch2D.width = pInputFrame->width;
    resDesc.res.pitch2D.height = pInputFrame->height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    auto cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    ratioX = 1.0f / (float)(pOutputFrame->width >> 1);
    ratioY = 1.0f / (float)(pOutputFrame->height >> 1);
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    resDesc.res.pitch2D.width >>= 1;
    resDesc.res.pitch2D.height >>= 1;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth>
cudaError_t resize_texture_bilinear_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const float *pgDummy) {
    //Y
    float ratioX = 1.0f / (float)(pOutputFrame->width);
    float ratioY = 1.0f / (float)(pOutputFrame->height);
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDesc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDesc.res.pitch2D.width = pInputFrame->width;
    resDesc.res.pitch2D.height = pInputFrame->height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj;
    auto cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_texture_bilinear<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}
#pragma warning(pop)

template<typename Type, int bit_depth>
__global__ void kernel_resize_spline(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texObj,
    const float ratioX, const float ratioY, const float ratioDistX, const float ratioDistY, const float *__restrict__ pgFactor3x4) {
    static  const int radius = 3;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //重みをsharedメモリにコピー
    __shared__ float psCopyFactor[radius][4];
    if (threadIdx.y == 0 && threadIdx.x < radius * 4) {
        ((float *)psCopyFactor[0])[threadIdx.x] = pgFactor3x4[threadIdx.x];
    }
    __syncthreads();

    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        float pWeightX[radius * 2];
        float pWeightY[radius * 2];

        for (int i = 0; i < radius * 2; i++) {
            //+0.5fはピクセル中心とするため
            const float sx = floor(x) + i - radius + 1.0f + 0.5f;
            const float sy = floor(y) + i - radius + 1.0f + 0.5f;
            //拡大ならratioDistXは1.0f、縮小ならratioの逆数(縮小側の距離に変換)
            const float dx = std::abs(sx - x) * ratioDistX;
            const float dy = std::abs(sy - y) * ratioDistY;
            float *psWeightX = psCopyFactor[min((int)dx, radius-1)];
            float *psWeightY = psCopyFactor[min((int)dy, radius-1)];
            //重みを計算
            float wx = psWeightX[3];
            float wy = psWeightY[3];
            wx += dx * psWeightX[2];
            wy += dy * psWeightY[2];
            const float dx2 = dx * dx;
            const float dy2 = dy * dy;
            wx += dx2 * psWeightX[1];
            wy += dy2 * psWeightY[1];
            wx += dx2 * dx * psWeightX[0];
            wy += dy2 * dy * psWeightY[0];
            pWeightX[i] = wx;
            pWeightY[i] = wy;
        }

        float weightSum = 0.0f;
        float clr = 0.0f;
        for (int j = 0; j < radius * 2; j++) {
            const float sy = floor(y) + j - radius + 1.0f + 0.5f;
            const float weightY = pWeightY[j];
            for (int i = 0; i < radius * 2; i++) {
                const float sx = floor(x) + i - radius + 1.0f + 0.5f;
                const float weightXY = pWeightX[i] * weightY;
                clr += tex2D<Type>(texObj, sx, sy) * weightXY;
                weightSum += weightXY;
            }
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(clr * __frcp_rn(weightSum), 0.0f, (1<<bit_depth) - 0.1f);
    }
}

template<typename Type, int bit_depth>
void resize_spline36(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texObj, const float ratioX, const float ratioY, const float ratioDistX, const float ratioDistY, const float *pgFactor3x4) {
    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_resize_spline<Type, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
}

template<typename Type, int bit_depth>
static cudaError_t resize_spline36_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const float *pgFactor3x4) {
    const float ratioX = pInputFrame->width / (float)(pOutputFrame->width);
    const float ratioY = pInputFrame->height / (float)(pOutputFrame->height);
    const float ratioDistX = (pInputFrame->width <= pOutputFrame->width) ? 1.0f : pOutputFrame->width / (float)(pInputFrame->width);
    const float ratioDistY = (pInputFrame->height <= pOutputFrame->height) ? 1.0f : pOutputFrame->height / (float)(pInputFrame->height);
    //Y
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDesc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDesc.res.pitch2D.width = pInputFrame->width;
    resDesc.res.pitch2D.height = pInputFrame->height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    auto cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    resDesc.res.pitch2D.width >>= 1;
    resDesc.res.pitch2D.height >>= 1;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth>
static cudaError_t resize_spline36_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, const float *pgFactor3x4) {
    const float ratioX = pInputFrame->width / (float)(pOutputFrame->width);
    const float ratioY = pInputFrame->height / (float)(pOutputFrame->height);
    const float ratioDistX = (pInputFrame->width <= pOutputFrame->width) ? 1.0f : pOutputFrame->width / (float)(pInputFrame->width);
    const float ratioDistY = (pInputFrame->height <= pOutputFrame->height) ? 1.0f : pOutputFrame->height / (float)(pInputFrame->height);
    //Y
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDesc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDesc.res.pitch2D.width = pInputFrame->width;
    resDesc.res.pitch2D.height = pInputFrame->height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj;
    auto cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDesc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2;
    cudaerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    resize_spline36<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texObj, ratioX, ratioY, ratioDistX, ratioDistY, pgFactor3x4);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texObj);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename T, typename Tfunc>
static NppStatus resize_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, Tfunc funcResize, NppiInterpolationMode interpMode) {
    const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    auto srcRect = nppiroi(pInputFrame);
    auto dstRect = nppiroi(pOutputFrame);
    //Y
    NppStatus sts = funcResize(
        (const T *)pInputFrame->ptr,
        srcSize, pInputFrame->pitch, srcRect,
        (T *)pOutputFrame->ptr,
        pOutputFrame->pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //U
    srcSize.width  >>= 1;
    srcSize.height >>= 1;
    srcRect.width  >>= 1;
    srcRect.height >>= 1;
    dstRect.width  >>= 1;
    dstRect.height >>= 1;
    sts = funcResize(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height),
        srcSize, pInputFrame->pitch, srcRect,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height),
        pOutputFrame->pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //V
    sts = funcResize(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2),
        srcSize, pInputFrame->pitch, srcRect,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2),
        pOutputFrame->pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    return sts;
}

NVENCSTATUS NVEncFilterResize::resizeYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
#if _M_IX86
    AddMessage(NV_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return NV_ENC_ERR_UNSUPPORTED_PARAM;
#else
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto interp = (NppiInterpolationMode)pResizeParam->interp;
    static const auto supportedCspYV12High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == NV_ENC_CSP_YV12) {
        nppsts = resize_yv12<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_C1R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYV12High.begin(), supportedCspYV12High.end(), m_pParam->frameIn.csp) != supportedCspYV12High.end()) {
        nppsts = resize_yv12<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_C1R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
#endif
}

template<typename T, typename Tfunc>
static NppStatus resize_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, Tfunc funcResize, NppiInterpolationMode interpMode) {
    const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    auto srcRect = nppiroi(pInputFrame);
    auto dstRect = nppiroi(pOutputFrame);
    const T *pSrc[3] = {
        (const T *)pInputFrame->ptr,
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height),
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2)
    };
    T *pDst[3] = {
        (T *)pOutputFrame->ptr,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height),
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2)
    };
    NppStatus sts = funcResize(
        pSrc,
        srcSize, pInputFrame->pitch, srcRect,
        pDst,
        pOutputFrame->pitch, dstRect,
        factorX, factorY, 0.0, 0.0, interpMode);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    return sts;
}

NVENCSTATUS NVEncFilterResize::resizeYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
#if _M_IX86
    AddMessage(NV_LOG_ERROR, _T("npp filter not supported on x86.\n"));
    return NV_ENC_ERR_UNSUPPORTED_PARAM;
#else
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    const auto interp = (NppiInterpolationMode)pResizeParam->interp;
    static const auto supportedCspYUV444High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == NV_ENC_CSP_YUV444) {
        nppsts = resize_yuv444<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_P3R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYUV444High.begin(), supportedCspYUV444High.end(), m_pParam->frameIn.csp) != supportedCspYUV444High.end()) {
        nppsts = resize_yuv444<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_P3R, interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
#endif
}

NVEncFilterResize::NVEncFilterResize() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("resize");
}

NVEncFilterResize::~NVEncFilterResize() {
    close();
}

NVENCSTATUS NVEncFilterResize::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(pParam);
    if (!pResizeParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pResizeParam->interp <= NPPI_INTER_MAX && !check_if_nppi_dll_available()) {
        AddMessage(NV_LOG_WARN, _T("--vpp-resize %s requires \"%s\", not available on your system.\n"), get_chr_from_value(list_nppi_resize, pResizeParam->interp), NPPI_DLL_NAME);
        pResizeParam->interp = RESIZE_CUDA_SPLINE36;
        AddMessage(NV_LOG_WARN, _T("switching to %s."), get_chr_from_value(list_nppi_resize, pResizeParam->interp));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pResizeParam->frameOut.height <= 0 || pResizeParam->frameOut.width <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pResizeParam->frameOut, 2);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(NV_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pResizeParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    if (pResizeParam->interp == RESIZE_CUDA_SPLINE36) {
        static const float SPLINE36_WEIGHT[3][4] = {
            { 13.0f/11.0f, -453.0f/209.0f,    -3.0f/209.0f,  1.0f          },
            { -6.0f/11.0f,  612.0f/209.0f, -1038.0f/209.0f,  540.0f/209.0f },
            {  1.0f/11.0f, -159.0f/209.0f,   434.0f/209.0f, -384.0f/209.0f },
        };
        m_weightSpline36 = CUMemBuf(sizeof(SPLINE36_WEIGHT));
        if (CUDA_SUCCESS != (cudaerr = m_weightSpline36.alloc())) {
            AddMessage(NV_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
        cudaerr = cudaMemcpy(m_weightSpline36.ptr, SPLINE36_WEIGHT[0], sizeof(SPLINE36_WEIGHT), cudaMemcpyHostToDevice);
        if (cudaerr != CUDA_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to send weight to gpu memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }

    m_sFilterInfo = strsprintf(_T("resize(%s): %dx%d -> %dx%d"),
        get_chr_from_value(list_nppi_resize, pResizeParam->interp),
        pResizeParam->frameIn.width, pResizeParam->frameIn.height,
        pResizeParam->frameOut.width, pResizeParam->frameOut.height);

    //コピーを保存
    m_pParam = pResizeParam;
    return sts;
}

NVENCSTATUS NVEncFilterResize::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (pInputFrame->interlaced && !m_bInterlacedWarn) {
        AddMessage(NV_LOG_WARN, _T("Interlaced resize is not supported, resizing as progressive.\n"));
        AddMessage(NV_LOG_WARN, _T("This should result in poor quality.\n"));
        m_bInterlacedWarn = true;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(NV_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    static const auto supportedCspYV12   = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12, NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444, NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);

    auto pResizeParam = std::dynamic_pointer_cast<NVEncFilterParamResize>(m_pParam);
    if (!pResizeParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pResizeParam->interp <= NPPI_INTER_MAX) {
        if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), m_pParam->frameIn.csp) != supportedCspYV12.end()) {
            sts = resizeYV12(ppOutputFrames[0], pInputFrame);
        } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_pParam->frameIn.csp) != supportedCspYUV444.end()) {
            sts = resizeYUV444(ppOutputFrames[0], pInputFrame);
        } else {
            AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
            sts = NV_ENC_ERR_UNIMPLEMENTED;
        }
    } else {
        struct resizeFunc {
            decltype(resize_texture_bilinear_yv12<uint8_t, 8>)* bilinear;
            decltype(resize_spline36_yv12<uint8_t, 8>)* spline36;

            resizeFunc(decltype(resize_texture_bilinear_yv12<uint8_t, 8>)* _bilinear, decltype(resize_spline36_yv12<uint8_t, 8>)* _spline36) :
                bilinear(_bilinear), spline36(_spline36) {
            }
            decltype(resize_spline36_yv12<uint8_t, 8>)* func(int interp) const {
                return interp == RESIZE_CUDA_TEXTURE_BILINEAR ? bilinear : spline36;
            }
        };
        static const std::map<NV_ENC_CSP, resizeFunc> resize_list = {
            { NV_ENC_CSP_YV12,      resizeFunc(resize_texture_bilinear_yv12<uint8_t,  8>,    resize_spline36_yv12<uint8_t,   8>)   },
            { NV_ENC_CSP_YV12_10,   resizeFunc(resize_texture_bilinear_yv12<uint16_t, 10>,   resize_spline36_yv12<uint16_t, 10>)   },
            { NV_ENC_CSP_YV12_12,   resizeFunc(resize_texture_bilinear_yv12<uint16_t, 12>,   resize_spline36_yv12<uint16_t, 12>)   },
            { NV_ENC_CSP_YV12_14,   resizeFunc(resize_texture_bilinear_yv12<uint16_t, 14>,   resize_spline36_yv12<uint16_t, 14>)   },
            { NV_ENC_CSP_YV12_16,   resizeFunc(resize_texture_bilinear_yv12<uint16_t, 16>,   resize_spline36_yv12<uint16_t, 16>)   },
            { NV_ENC_CSP_YUV444,    resizeFunc(resize_texture_bilinear_yuv444<uint8_t,   8>, resize_spline36_yuv444<uint8_t,   8>) },
            { NV_ENC_CSP_YUV444_10, resizeFunc(resize_texture_bilinear_yuv444<uint16_t, 10>, resize_spline36_yuv444<uint16_t, 10>) },
            { NV_ENC_CSP_YUV444_12, resizeFunc(resize_texture_bilinear_yuv444<uint16_t, 12>, resize_spline36_yuv444<uint16_t, 12>) },
            { NV_ENC_CSP_YUV444_14, resizeFunc(resize_texture_bilinear_yuv444<uint16_t, 14>, resize_spline36_yuv444<uint16_t, 14>) },
            { NV_ENC_CSP_YUV444_16, resizeFunc(resize_texture_bilinear_yuv444<uint16_t, 16>, resize_spline36_yuv444<uint16_t, 16>) },
        };
        if (resize_list.count(pInputFrame->csp) == 0) {
            AddMessage(NV_LOG_ERROR, _T("unsupported csp %s.\n"), NV_ENC_CSP_NAMES[pInputFrame->csp]);
            return NV_ENC_ERR_UNIMPLEMENTED;
        }
        resize_list.at(pInputFrame->csp).func(pResizeParam->interp)(ppOutputFrames[0], pInputFrame, (float *)m_weightSpline36.ptr);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(NV_LOG_ERROR, _T("error at resize(%s): %s.\n"),
                NV_ENC_CSP_NAMES[pInputFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    }
    return sts;
}

void NVEncFilterResize::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
