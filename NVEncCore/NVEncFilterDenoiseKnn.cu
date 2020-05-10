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
#include "convert_csp.h"
#include "NVEncFilterDenoiseKnn.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int KNN_RADIUS_MAX = 5;

template<typename Type, int knn_radius, int bit_depth>
__global__ void kernel_denoise_knn(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t texSrc, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    const float knn_window_area = (float)((2 * knn_radius + 1) * (2 * knn_radius + 1));
    const float inv_knn_window_area = 1.0f / knn_window_area;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        float fCount = 0.0f;
        float sumWeights = 0.0f;
        float sum = 0.0f;
        float center = (float)tex2D<Type>(texSrc, x, y) * (1.0f / (1<<bit_depth));

        for (float i = -knn_radius; i <= knn_radius; i++) {
            for (float j = -knn_radius; j <= knn_radius; j++) {
                float clrIJ = (float)tex2D<Type>(texSrc, x + j, y + i) * (1.0f / (1<<bit_depth));
                float distanceIJ = (center - clrIJ) * (center - clrIJ);

                float weightIJ = __expf(-(distanceIJ * strength + (i * i + j * j) * inv_knn_window_area));

                sum += clrIJ * weightIJ;

                sumWeights += weightIJ;

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
    cudaTextureObject_t texSrc, int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    dim3 blockSize(64, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    switch (radius) {
    case 1:
        kernel_denoise_knn<Type, 1, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 2:
        kernel_denoise_knn<Type, 2, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 3:
        kernel_denoise_knn<Type, 3, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 4:
        kernel_denoise_knn<Type, 4, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    case 5:
        //よりレジスタを使うので、ブロック当たりのスレッド数を低減
        blockSize = dim3(32, 16);
        gridSize = dim3(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
        kernel_denoise_knn<Type, 5, bit_depth><<<gridSize, blockSize>>>(pDst, dstPitch, dstWidth, dstHeight, texSrc,
            1.0f / (strength * strength), lerpC, weight_threshold, lerp_threshold);
        break;
    default:
        break;
    }
}

template<typename Type, int bit_depth>
static cudaError_t denoise_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
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
    texDescSrc.readMode         = cudaReadModeElementType;
    texDescSrc.normalizedCoords = 0;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2,
        pOutputFrame->pitch, pOutputFrame->width >> 1, pOutputFrame->height >> 1,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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
static cudaError_t denoise_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
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
    texDescSrc.readMode         = cudaReadModeElementType;
    texDescSrc.normalizedCoords = 0;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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
    denoise_knn<Type, bit_depth>((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2,
        pOutputFrame->pitch, pOutputFrame->width, pOutputFrame->height,
        texSrc, radius, strength, lerpC, weight_threshold, lerp_threshold);
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

NVEncFilterDenoiseKnn::NVEncFilterDenoiseKnn() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("knn");
}

NVEncFilterDenoiseKnn::~NVEncFilterDenoiseKnn() {
    close();
}

RGY_ERR NVEncFilterDenoiseKnn::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(pParam);
    if (!pKnnParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pKnnParam->frameOut.height <= 0 || pKnnParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius > KNN_RADIUS_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.strength < 0.0 || 1.0 < pKnnParam->knn.strength) {
        AddMessage(RGY_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerpC < 0.0 || 1.0 < pKnnParam->knn.lerpC) {
        AddMessage(RGY_LOG_ERROR, _T("lerpC should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerp_threshold < 0.0 || 1.0 < pKnnParam->knn.lerp_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_lerp should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.weight_threshold < 0.0 || 1.0 < pKnnParam->knn.weight_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_weight should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pKnnParam->frameOut, 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pKnnParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseKnn::print() const {
    return knn.print();
}

RGY_ERR NVEncFilterDenoiseKnn::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(m_pParam);
    if (!pKnnParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(denoise_yv12<uint8_t, 8>)*> denoise_list = {
        { RGY_CSP_YV12,      denoise_yv12<uint8_t,   8> },
        { RGY_CSP_YV12_16,   denoise_yv12<uint16_t, 16> },
        { RGY_CSP_YUV444,    denoise_yuv444<uint8_t,   8> },
        { RGY_CSP_YUV444_16, denoise_yuv444<uint16_t, 16> },
    };
    if (denoise_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    denoise_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame, pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at resize(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return RGY_ERR_CUDA;
    }
    return sts;
}

void NVEncFilterDenoiseKnn::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
