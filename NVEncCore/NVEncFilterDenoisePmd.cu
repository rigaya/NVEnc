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
#include <map>
#include "ConvertCsp.h"
#include "NVEncFilterDenoisePmd.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static int final_dst_index(int loop_count) {
    return (loop_count - 1) & 1;
}

static __device__ float pmd_exp(float x, float strength2, float inv_threshold2) {
    return strength2 * __expf(-x*x * inv_threshold2);
}

static __device__ float pmd(float x, float strength2, float inv_threshold2) {
    return strength2 * __frcp_rn(1.0f + (x*x * inv_threshold2));
}

template<typename Type, int bit_depth>
__global__ void kernel_create_gauss(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight, cudaTextureObject_t texSrc) {
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
                sum_line += (float)tex2D<Type>(texSrc, x + (float)i, y + (float)j) * weight[i];
            }
            sum += sum_line * weight[j];
        }
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(sum + 0.5f);
    }
}

template<typename Type, int bit_depth, bool useExp>
__global__ void kernel_denoise_pmd(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    cudaTextureObject_t tSrc,
    cudaTextureObject_t tGrf,
    const float strength2, const float inv_threshold2) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;
        float clr   = tex2D<Type>(tSrc, x+0, y+0);
        float clrym = tex2D<Type>(tSrc, x+0, y-1);
        float clryp = tex2D<Type>(tSrc, x+0, y+1);
        float clrxm = tex2D<Type>(tSrc, x-1, y+0);
        float clrxp = tex2D<Type>(tSrc, x+1, y+0);
        float grf   = tex2D<Type>(tGrf, x+0, y+0);
        float grfym = tex2D<Type>(tGrf, x+0, y-1);
        float grfyp = tex2D<Type>(tGrf, x+0, y+1);
        float grfxm = tex2D<Type>(tGrf, x-1, y+0);
        float grfxp = tex2D<Type>(tGrf, x+1, y+0);
        clr += (useExp)
            ? (clrym - clr) * pmd_exp(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd_exp(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd_exp(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd_exp(grfxp - grf, strength2, inv_threshold2)
            : (clrym - clr) * pmd(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd(grfxp - grf, strength2, inv_threshold2);

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(clr + 0.5f, 0.0f, (float)(1<<bit_depth)-0.1f));
    }
}

template<typename Type, int bit_depth, bool useExp>
cudaError_t denoise_pmd(uint8_t *pDst[2], uint8_t *pGauss, const int dstPitch, const int dstWidth, const int dstHeight,
    uint8_t *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    int loop_count, const float strength, const float threshold) {
    const float range = 4.0f;
    const float strength2 = strength / (range * 100.0f);
    const float threshold2 = std::pow(2.0f, threshold / 10.0f - (12 - bit_depth) * 2.0f);
    const float inv_threshold2 = 1.0f / threshold2;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.devPtr = pSrc;
    resDescSrc.res.pitch2D.pitchInBytes = srcPitch;
    resDescSrc.res.pitch2D.width = srcWidth;
    resDescSrc.res.pitch2D.height = srcHeight;
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
    dim3 blockSize(32, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_create_gauss<Type, bit_depth><<<gridSize, blockSize>>>(
        pGauss,
        dstPitch, dstWidth, dstHeight, texSrc);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    cudaResourceDesc resDescGrf;
    memset(&resDescGrf, 0, sizeof(resDescGrf));
    resDescGrf.resType = cudaResourceTypePitch2D;
    resDescGrf.res.pitch2D.devPtr = pGauss;
    resDescGrf.res.pitch2D.pitchInBytes = dstPitch;
    resDescGrf.res.pitch2D.width = dstWidth;
    resDescGrf.res.pitch2D.height = dstHeight;
    resDescGrf.res.pitch2D.desc = cudaCreateChannelDesc<Type>();

    cudaTextureDesc texDescGrf;
    memset(&texDescGrf, 0, sizeof(texDescGrf));
    texDescGrf.addressMode[0]   = cudaAddressModeClamp;
    texDescGrf.addressMode[1]   = cudaAddressModeClamp;
    texDescGrf.filterMode       = cudaFilterModePoint;
    texDescGrf.readMode         = cudaReadModeElementType;
    texDescGrf.normalizedCoords = 0;

    cudaTextureObject_t texGrf = 0;
    cudaerr = cudaCreateTextureObject(&texGrf, &resDescGrf, &texDescGrf, nullptr);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    int dst_index = 0;
    for (int i = 0; i < loop_count; i++) {
        dst_index = i & 1;
        kernel_denoise_pmd<Type, bit_depth, useExp><<<gridSize, blockSize>>>(pDst[dst_index],
            dstPitch, dstWidth, dstHeight, texSrc, texGrf, strength2, inv_threshold2);
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
        cudaerr = cudaDestroyTextureObject(texSrc);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
        if (i < loop_count-1) {
            resDescSrc.res.pitch2D.devPtr = pDst[dst_index];
            resDescSrc.res.pitch2D.pitchInBytes = dstPitch;
            resDescSrc.res.pitch2D.width = dstWidth;
            resDescSrc.res.pitch2D.height = dstHeight;
            cudaerr = cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
            if (cudaerr != cudaSuccess) {
                return cudaerr;
            }
        }
    }
    cudaerr = cudaDestroyTextureObject(texGrf);
    return cudaSuccess;
}

template<typename Type, int bit_depth, bool useExp>
static cudaError_t denoise_yv12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold) {
    uint8_t *pDst[2] = { 0 };
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr;
    //Y
    auto cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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
    cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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
    cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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
    uint8_t *pDst[2] = { 0 };
    pDst[0] = (uint8_t *)pOutputFrame[0]->ptr;
    pDst[1] = (uint8_t *)pOutputFrame[1]->ptr;
    //Y
    auto cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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
    cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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
    cudaerr = denoise_pmd<Type, bit_depth, useExp>(
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


NVENCSTATUS NVEncFilterDenoisePmd::denoise(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame) {
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(m_pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    struct pmd_func {
        decltype(denoise_yv12<uint8_t, 8, true>)* func[2];
        pmd_func(decltype(denoise_yv12<uint8_t, 8, true>)* useexp, decltype(denoise_yv12<uint8_t, 8, false>)* noexp) {
            func[0] = noexp;
            func[1] = useexp;
        };
    };

    static const std::map<RGY_CSP, pmd_func> denoise_func_list = {
        { RGY_CSP_YV12,      pmd_func(denoise_yv12<uint8_t,   8, true>,   denoise_yv12<uint8_t,   8, false>) },
        { RGY_CSP_YV12_10,   pmd_func(denoise_yv12<uint16_t, 10, true>,   denoise_yv12<uint16_t, 10, false>) },
        { RGY_CSP_YV12_12,   pmd_func(denoise_yv12<uint16_t, 12, true>,   denoise_yv12<uint16_t, 12, false>) },
        { RGY_CSP_YV12_14,   pmd_func(denoise_yv12<uint16_t, 14, true>,   denoise_yv12<uint16_t, 14, false>) },
        { RGY_CSP_YV12_16,   pmd_func(denoise_yv12<uint16_t, 16, true>,   denoise_yv12<uint16_t, 16, false>) },
        { RGY_CSP_YUV444,    pmd_func(denoise_yuv444<uint8_t,   8, true>, denoise_yuv444<uint8_t,   8, false>) },
        { RGY_CSP_YUV444_10, pmd_func(denoise_yuv444<uint16_t, 10, true>, denoise_yuv444<uint16_t, 10, false>) },
        { RGY_CSP_YUV444_12, pmd_func(denoise_yuv444<uint16_t, 12, true>, denoise_yuv444<uint16_t, 12, false>) },
        { RGY_CSP_YUV444_14, pmd_func(denoise_yuv444<uint16_t, 14, true>, denoise_yuv444<uint16_t, 14, false>) },
        { RGY_CSP_YUV444_16, pmd_func(denoise_yuv444<uint16_t, 16, true>, denoise_yuv444<uint16_t, 16, false>) },
    };
    if (denoise_func_list.count(pPmdParam->frameIn.csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for denoise(pmd): %s\n"), RGY_CSP_NAMES[pPmdParam->frameIn.csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    auto cudaerr = denoise_func_list.at(pPmdParam->frameIn.csp).func[!!pPmdParam->pmd.useExp](pOutputFrame, pGauss, pInputFrame, pPmdParam->pmd.applyCount, pPmdParam->pmd.strength, pPmdParam->pmd.threshold);
    if (cudaerr != cudaSuccess) {
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}

NVEncFilterDenoisePmd::NVEncFilterDenoisePmd() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("pmd");
}

NVEncFilterDenoisePmd::~NVEncFilterDenoisePmd() {
    close();
}

NVENCSTATUS NVEncFilterDenoisePmd::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pPmdParam->frameOut.height <= 0 || pPmdParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.applyCount <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, apply_count must be a positive value.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.strength < 0.0f || 100.0f < pPmdParam->pmd.strength) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 100.0.\n"));
        pPmdParam->pmd.strength = clamp(pPmdParam->pmd.strength, 0.0f, 100.0f);
    }
    if (pPmdParam->pmd.threshold < 0.0f || 255.0f < pPmdParam->pmd.threshold) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 255.0.\n"));
        pPmdParam->pmd.threshold = clamp(pPmdParam->pmd.threshold, 0.0f, 255.0f);
    }

    auto cudaerr = AllocFrameBuf(pPmdParam->frameOut, 4);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pPmdParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    m_Gauss.frame = pPmdParam->frameOut;
    cudaerr = m_Gauss.alloc();
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_sFilterInfo = strsprintf(_T("denoise(pmd): strength %d, threshold %d, apply %d, exp %d"),
        (int)pPmdParam->pmd.strength, (int)pPmdParam->pmd.threshold, pPmdParam->pmd.applyCount, pPmdParam->pmd.useExp);

    m_pParam = pParam;
    return sts;
}

NVENCSTATUS NVEncFilterDenoisePmd::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(m_pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    FrameInfo *pOutputFrame[2] = {
        &m_pFrameBuf[(m_nFrameIdx++) % m_pFrameBuf.size()].get()->frame,
        &m_pFrameBuf[(m_nFrameIdx++) % m_pFrameBuf.size()].get()->frame,
    };

    ppOutputFrames[0] = pOutputFrame[final_dst_index(pPmdParam->pmd.applyCount)];
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (pInputFrame->interlaced && !m_bInterlacedWarn) {
        AddMessage(RGY_LOG_WARN, _T("Interlaced denoise is not supported, denoise as progressive.\n"));
        AddMessage(RGY_LOG_WARN, _T("This should result in poor quality.\n"));
        m_bInterlacedWarn = true;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    return denoise(pOutputFrame, &m_Gauss.frame, pInputFrame);
}

void NVEncFilterDenoisePmd::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
