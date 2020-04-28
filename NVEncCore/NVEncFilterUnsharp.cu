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
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterUnsharp.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int UNSHARP_RADIUS_MAX = 9;
static const int UNSHARP_BLOCK_X = 32;
static const int UNSHARP_BLOCK_Y = 16;

template<typename Type, int radius, int bit_depth>
__global__ void kernel_unsharp(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texSrc, const float *__restrict__ pGaussWeight, const float weight, const float threshold) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int shared_size = (2 * radius + 1) * (2 * radius + 1);
    __shared__ float shared[shared_size];
    if (local_id < shared_size) {
        shared[local_id] = pGaussWeight[local_id];
    }
    static_assert(UNSHARP_BLOCK_X * UNSHARP_BLOCK_Y >= shared_size, "radius too big.");
    __syncthreads();

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        float center = (float)tex2D<float>(texSrc, ix + 0.5f, iy + 0.5f);
        float *ptr_weight = shared;

        for (int j = -radius; j <= radius; j++) {
            #pragma unroll
            for (int i = -radius; i <= radius; i++) {
                sum += tex2D<float>(texSrc, ix + i + 0.5f, iy + j + 0.5f) * ptr_weight[0];
                ptr_weight++;
            }
        }

        const float diff = center - sum;
        if (std::abs(diff) >= threshold) {
            center += weight * diff;
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f-RGY_FLT_EPS) * (1 << (bit_depth)));
    }
}

template<typename Type, int bit_depth>
void unsharp(uint8_t *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texSrc, const float *__restrict__ pGaussWeight, int radius, const float weight, const float threshold) {
    dim3 blockSize(UNSHARP_BLOCK_X, UNSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    switch (radius) {
    case 1: kernel_unsharp<Type, 1, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 2: kernel_unsharp<Type, 2, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 3: kernel_unsharp<Type, 3, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 4: kernel_unsharp<Type, 4, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 5: kernel_unsharp<Type, 5, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 6: kernel_unsharp<Type, 6, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 7: kernel_unsharp<Type, 7, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 8: kernel_unsharp<Type, 8, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 9: kernel_unsharp<Type, 9, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    default: break;
    }
}

template<typename Type, int bit_depth>
static cudaError_t unsharp_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pGaussWeightY, CUMemBuf *pGaussWeightUV,
    int radius, const float weight, const float threshold) {
    //Y
    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDescSrc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDescSrc.res.pitch2D.width = pInputFrame->width;
    resDescSrc.res.pitch2D.height = pInputFrame->height;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeClamp;
    texDescSrc.addressMode[1]   = cudaAddressModeClamp;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, (const float *)pGaussWeightY->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    resDescSrc.res.pitch2D.width >>= 1;
    resDescSrc.res.pitch2D.height >>= 1;
    cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texSrc, (const float *)pGaussWeightUV->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2;
    cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texSrc, (const float *)pGaussWeightUV->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

template<typename Type, int bit_depth>
static cudaError_t unsharp_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pGaussWeightY, CUMemBuf *pGaussWeightUV,
    int radius, const float weight, const float threshold) {
    //Y
    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.devPtr = pInputFrame->ptr;
    resDescSrc.res.pitch2D.pitchInBytes = pInputFrame->pitch;
    resDescSrc.res.pitch2D.width = pInputFrame->width;
    resDescSrc.res.pitch2D.height = pInputFrame->height;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0]   = cudaAddressModeClamp;
    texDescSrc.addressMode[1]   = cudaAddressModeClamp;
    texDescSrc.filterMode       = cudaFilterModePoint;
    texDescSrc.readMode         = cudaReadModeNormalizedFloat;
    texDescSrc.normalizedCoords = 0;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, (const float *)pGaussWeightY->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //U
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height;
    cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, (const float *)pGaussWeightUV->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    //V
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2;
    cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, (const float *)pGaussWeightUV->ptr, radius, weight, threshold / (1 << (sizeof(Type)*8)));
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaerr = cudaDestroyTextureObject(texSrc);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

NVEncFilterUnsharp::NVEncFilterUnsharp() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("unsharp");
}

NVEncFilterUnsharp::~NVEncFilterUnsharp() {
    close();
}

RGY_ERR NVEncFilterUnsharp::setWeight(unique_ptr<CUMemBuf>& pGaussWeightBuf, int radius, float sigma) {
    const int nWeightCount = (2 * radius + 1) * (2 * radius + 1);
    const int nBufferSize = sizeof(float) * nWeightCount;
    pGaussWeightBuf = unique_ptr<CUMemBuf>(new CUMemBuf(nBufferSize));

    auto cudaerr = pGaussWeightBuf->alloc();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate weight buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    vector<float> weight(nWeightCount);
    float *ptr_weight = weight.data();
    double sum = 0.0;
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            const double w = 1.0f / (2.0f * (float)M_PI * sigma * sigma) * std::exp(-1.0f * (i * i + j * j) / (2.0f * sigma * sigma));
            *ptr_weight = (float)w;
            sum += (double)w;
            ptr_weight++;
        }
    }
    ptr_weight = weight.data();
    const float inv_sum = (float)(1.0 / sum);
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            *ptr_weight *= inv_sum;
            ptr_weight++;
        }
    }
    cudaerr = cudaMemcpy(pGaussWeightBuf->ptr, weight.data(), nBufferSize, cudaMemcpyHostToDevice);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy weight to device: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_CUDA;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterUnsharp::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(pParam);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pUnsharpParam->frameOut.height <= 0 || pUnsharpParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pUnsharpParam->unsharp.radius < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (radius).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pUnsharpParam->unsharp.radius < 1 && pUnsharpParam->unsharp.radius > UNSHARP_RADIUS_MAX) {
        AddMessage(RGY_LOG_WARN, _T("radius must be in range of 1-%d.\n"), UNSHARP_RADIUS_MAX);
        pUnsharpParam->unsharp.radius = clamp(pUnsharpParam->unsharp.radius, 1, UNSHARP_RADIUS_MAX);
    }
    if (pUnsharpParam->unsharp.weight < 0.0f || 10.0f < pUnsharpParam->unsharp.weight) {
        pUnsharpParam->unsharp.weight = clamp(pUnsharpParam->unsharp.weight, 0.0f, 10.0f);
        AddMessage(RGY_LOG_WARN, _T("weight should be in range of %.1f - %.1f.\n"), 0.0f, 10.0f);
    }
    if (pUnsharpParam->unsharp.threshold < 0.0f || 255.0f < pUnsharpParam->unsharp.threshold) {
        pUnsharpParam->unsharp.threshold = clamp(pUnsharpParam->unsharp.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }

    auto cudaerr = AllocFrameBuf(pUnsharpParam->frameOut, 1);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pUnsharpParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    if (!m_pParam
        || std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_pParam)->unsharp.radius != pUnsharpParam->unsharp.radius) {
        float sigmaY = 0.8f + 0.3f * pUnsharpParam->unsharp.radius;
        float sigmaUV = (RGY_CSP_CHROMA_FORMAT[pUnsharpParam->frameIn.csp] == RGY_CHROMAFMT_YUV420) ? 0.8f + 0.3f * (pUnsharpParam->unsharp.radius * 0.5f + 0.25f) : sigmaY;

        if (   RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufY,  pUnsharpParam->unsharp.radius, sigmaY))
            || RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufUV, pUnsharpParam->unsharp.radius, sigmaUV))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set weight: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return sts;
        }
    }

    setFilterInfo(pParam->print());
    m_pParam = pUnsharpParam;
    return sts;
}

tstring NVEncFilterParamUnsharp::print() const {
    return unsharp.print();
}

RGY_ERR NVEncFilterUnsharp::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_pParam);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(unsharp_yv12<uint8_t, 8>)*> denoise_list = {
        { RGY_CSP_YV12,      unsharp_yv12<uint8_t,     8> },
        { RGY_CSP_YV12_16,   unsharp_yv12<uint16_t,   16> },
        { RGY_CSP_YUV444,    unsharp_yuv444<uint8_t,   8> },
        { RGY_CSP_YUV444_16, unsharp_yuv444<uint16_t, 16> }
    };
    if (denoise_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    denoise_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame, m_pGaussWeightBufY.get(), m_pGaussWeightBufUV.get(),
        pUnsharpParam->unsharp.radius, pUnsharpParam->unsharp.weight, pUnsharpParam->unsharp.threshold);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at resize(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return RGY_ERR_CUDA;
    }
    return sts;
}

void NVEncFilterUnsharp::close() {
    m_pFrameBuf.clear();
    m_pGaussWeightBufY.reset();
    m_pGaussWeightBufUV.reset();
    m_bInterlacedWarn = false;
}
