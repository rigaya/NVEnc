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
#include "rgy_prm.h"
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
    cudaTextureObject_t texSrc, const float *__restrict__ pGaussWeight, const int radius, const float weight, const float threshold,
    cudaStream_t stream) {
    dim3 blockSize(UNSHARP_BLOCK_X, UNSHARP_BLOCK_Y);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    switch (radius) {
    case 1: kernel_unsharp<Type, 1, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 2: kernel_unsharp<Type, 2, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 3: kernel_unsharp<Type, 3, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 4: kernel_unsharp<Type, 4, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 5: kernel_unsharp<Type, 5, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 6: kernel_unsharp<Type, 6, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 7: kernel_unsharp<Type, 7, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 8: kernel_unsharp<Type, 8, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    case 9: kernel_unsharp<Type, 9, bit_depth><<<gridSize, blockSize, 0, stream>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc, pGaussWeight, weight, threshold); break;
    default: break;
    }
}

template<typename Type>
cudaError_t textureCreateDenoiseUnsharp(cudaTextureObject_t &tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, const int pitch, const int width, const int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = ptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

template<typename Type, int bit_depth>
static cudaError_t unsharp_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, CUMemBuf *pGaussWeight,
    const int radius, const float weight, const float threshold, cudaStream_t stream) {
    cudaTextureObject_t texSrc = 0;
    auto cudaerr = textureCreateDenoiseUnsharp<Type>(texSrc, cudaFilterModePoint, cudaReadModeNormalizedFloat, pInputFrame->ptr[0], pInputFrame->pitch[0], pInputFrame->width, pInputFrame->height);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    unsharp<Type, bit_depth>((uint8_t *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        texSrc, (const float *)pGaussWeight->ptr, radius, weight, threshold / (1 << (sizeof(Type) * 8)),
        stream);
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
static RGY_ERR unsharp_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, CUMemBuf *pGaussWeightY, CUMemBuf *pGaussWeightUV,
    const int radius, const float weight, const float threshold, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeInput = getPlane(pInputFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
        const bool isUV = (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420) && (plane == RGY_PLANE_U || plane == RGY_PLANE_V);
        auto err = unsharp_plane<Type, bit_depth>(&planeOutput, &planeInput, isUV ? pGaussWeightUV : pGaussWeightY, radius, weight, threshold, stream);
        if (err != cudaSuccess) {
            return err_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterUnsharp::NVEncFilterUnsharp() : m_bInterlacedWarn(false) {
    m_name = _T("unsharp");
}

NVEncFilterUnsharp::~NVEncFilterUnsharp() {
    close();
}

RGY_ERR NVEncFilterUnsharp::setWeight(unique_ptr<CUMemBuf>& pGaussWeightBuf, int radius, float sigma) {
    const int nWeightCount = (2 * radius + 1) * (2 * radius + 1);
    const int nBufferSize = sizeof(float) * nWeightCount;
    pGaussWeightBuf = unique_ptr<CUMemBuf>(new CUMemBuf(nBufferSize));

    auto sts = pGaussWeightBuf->alloc();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate weight buffer: %s.\n"), get_err_mes(sts));
        return sts;
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
    sts = err_to_rgy(cudaMemcpy(pGaussWeightBuf->ptr, weight.data(), nBufferSize, cudaMemcpyHostToDevice));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy weight to device: %s.\n"), get_err_mes(sts));
        return RGY_ERR_CUDA;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterUnsharp::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
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
    sts = AllocFrameBuf(pUnsharpParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pUnsharpParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if (!m_param
        || std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_param)->unsharp.radius != pUnsharpParam->unsharp.radius) {
        float sigmaY = 0.8f + 0.3f * pUnsharpParam->unsharp.radius;
        float sigmaUV = (RGY_CSP_CHROMA_FORMAT[pUnsharpParam->frameIn.csp] == RGY_CHROMAFMT_YUV420) ? 0.8f + 0.3f * (pUnsharpParam->unsharp.radius * 0.5f + 0.25f) : sigmaY;

        if (   RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufY,  pUnsharpParam->unsharp.radius, sigmaY))
            || RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufUV, pUnsharpParam->unsharp.radius, sigmaUV))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set weight: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    setFilterInfo(pParam->print());
    m_param = pUnsharpParam;
    return sts;
}

tstring NVEncFilterParamUnsharp::print() const {
    return unsharp.print();
}

RGY_ERR NVEncFilterUnsharp::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_param);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_DATA_TYPE, decltype(unsharp_frame<uint8_t, 8>)*> denoise_list = {
        { RGY_DATA_TYPE_U8,  unsharp_frame<uint8_t,   8> },
        { RGY_DATA_TYPE_U16, unsharp_frame<uint16_t, 16> }
    };
    if (denoise_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = denoise_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(ppOutputFrames[0], pInputFrame, m_pGaussWeightBufY.get(), m_pGaussWeightBufUV.get(),
        pUnsharpParam->unsharp.radius, pUnsharpParam->unsharp.weight, pUnsharpParam->unsharp.threshold,
        stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at unsharp(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterUnsharp::close() {
    m_frameBuf.clear();
    m_pGaussWeightBufY.reset();
    m_pGaussWeightBufUV.reset();
    m_bInterlacedWarn = false;
}
