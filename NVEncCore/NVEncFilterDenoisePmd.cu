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
#include "convert_csp.h"
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

template<typename Type>
cudaError_t textureCreateDenoisePmd(cudaTextureObject_t &tex, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, uint8_t *ptr, int pitch, int width, int height) {
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

template<typename Type, int bit_depth, bool useExp>
cudaError_t denoise_pmd_plane(uint8_t *pDst[2], uint8_t *pGauss, const int dstPitch, const int dstWidth, const int dstHeight,
    uint8_t *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    int loop_count, const float strength, const float threshold, cudaStream_t stream) {
    const float range = 4.0f;
    const float strength2 = strength / (range * 100.0f);
    const float threshold2 = std::pow(2.0f, threshold / 10.0f - (12 - bit_depth) * 2.0f);
    const float inv_threshold2 = 1.0f / threshold2;

    cudaTextureObject_t texSrc = 0;
    auto cudaerr = textureCreateDenoisePmd<Type>(texSrc, cudaFilterModePoint, cudaReadModeElementType, pSrc, srcPitch, srcWidth, srcHeight);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    dim3 blockSize(32, 16);
    dim3 gridSize(divCeil(dstWidth, blockSize.x), divCeil(dstHeight, blockSize.y));
    kernel_create_gauss<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        pGauss,
        dstPitch, dstWidth, dstHeight, texSrc);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }

    cudaTextureObject_t texGrf = 0;
    cudaerr = textureCreateDenoisePmd<Type>(texGrf, cudaFilterModePoint, cudaReadModeElementType, pGauss, dstPitch, dstWidth, dstHeight);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    int dst_index = 0;
    for (int i = 0; i < loop_count; i++) {
        dst_index = i & 1;
        kernel_denoise_pmd<Type, bit_depth, useExp><<<gridSize, blockSize, 0, stream>>>(pDst[dst_index],
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
            cudaerr = textureCreateDenoisePmd<Type>(texSrc, cudaFilterModePoint, cudaReadModeElementType, pDst[dst_index], dstPitch, dstWidth, dstHeight);
            if (cudaerr != cudaSuccess) {
                return cudaerr;
            }
        }
    }
    cudaerr = cudaDestroyTextureObject(texGrf);
    return cudaSuccess;
}

template<typename Type, int bit_depth, bool useExp>
static cudaError_t denoise_pmd_frame(RGYFrameInfo *pOutputFrame[2], RGYFrameInfo *pGauss, const RGYFrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeInput = getPlane(pInputFrame, plane);
        const auto planeGauss = getPlane(pGauss, plane);
        RGYFrameInfo planeOutput[2] = { getPlane(pOutputFrame[0], plane), getPlane(pOutputFrame[1], plane) };
        uint8_t *pDst[2];
        pDst[0] = planeOutput[0].ptr;
        pDst[1] = planeOutput[1].ptr;
        auto cudaerr = denoise_pmd_plane<Type, bit_depth, useExp>(
            pDst, planeGauss.ptr,
            planeOutput[0].pitch, planeOutput[0].width, planeOutput[0].height,
            planeInput.ptr,
            planeInput.pitch, planeInput.width, planeInput.height,
            loop_count, strength, threshold, stream);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    return cudaSuccess;
}

RGY_ERR NVEncFilterDenoisePmd::denoise(RGYFrameInfo *pOutputFrame[2], RGYFrameInfo *pGauss, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(m_pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    struct pmd_func {
        decltype(denoise_pmd_frame<uint8_t, 8, true>)* func[2];
        pmd_func(decltype(denoise_pmd_frame<uint8_t, 8, true>)* useexp, decltype(denoise_pmd_frame<uint8_t, 8, false>)* noexp) {
            func[0] = noexp;
            func[1] = useexp;
        };
    };

    static const std::map<RGY_CSP, pmd_func> denoise_func_list = {
        { RGY_CSP_YV12,      pmd_func(denoise_pmd_frame<uint8_t,   8, true>, denoise_pmd_frame<uint8_t,   8, false>) },
        { RGY_CSP_YV12_16,   pmd_func(denoise_pmd_frame<uint16_t, 16, true>, denoise_pmd_frame<uint16_t, 16, false>) },
        { RGY_CSP_YUV444,    pmd_func(denoise_pmd_frame<uint8_t,   8, true>, denoise_pmd_frame<uint8_t,   8, false>) },
        { RGY_CSP_YUV444_16, pmd_func(denoise_pmd_frame<uint16_t, 16, true>, denoise_pmd_frame<uint16_t, 16, false>) },
    };
    if (denoise_func_list.count(pPmdParam->frameIn.csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for denoise(pmd): %s\n"), RGY_CSP_NAMES[pPmdParam->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    auto cudaerr = denoise_func_list.at(pPmdParam->frameIn.csp).func[!!pPmdParam->pmd.useExp](pOutputFrame, pGauss, pInputFrame, pPmdParam->pmd.applyCount, pPmdParam->pmd.strength, pPmdParam->pmd.threshold, stream);
    if (cudaerr != cudaSuccess) {
        return RGY_ERR_CUDA;
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoisePmd::NVEncFilterDenoisePmd() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("pmd");
}

NVEncFilterDenoisePmd::~NVEncFilterDenoisePmd() {
    close();
}

RGY_ERR NVEncFilterDenoisePmd::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pPmdParam->frameOut.height <= 0 || pPmdParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.applyCount <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, apply_count must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.strength < 0.0f || 100.0f < pPmdParam->pmd.strength) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 100.0.\n"));
        pPmdParam->pmd.strength = clamp(pPmdParam->pmd.strength, 0.0f, 100.0f);
    }
    if (pPmdParam->pmd.threshold < 0.0f || 255.0f < pPmdParam->pmd.threshold) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 255.0.\n"));
        pPmdParam->pmd.threshold = clamp(pPmdParam->pmd.threshold, 0.0f, 255.0f);
    }

    auto cudaerr = AllocFrameBuf(pPmdParam->frameOut, 2);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pPmdParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    if (cmpFrameInfoCspResolution(&m_Gauss.frame, &pPmdParam->frameOut)) {
        m_Gauss.frame.width = pPmdParam->frameOut.width;
        m_Gauss.frame.height = pPmdParam->frameOut.height;
        m_Gauss.frame.pitch = pPmdParam->frameOut.pitch;
        m_Gauss.frame.picstruct = pPmdParam->frameOut.picstruct;
        m_Gauss.frame.deivce_mem = pPmdParam->frameOut.deivce_mem;
        m_Gauss.frame.csp = pPmdParam->frameOut.csp;
        cudaerr = m_Gauss.alloc();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamDenoisePmd::print() const {
    return pmd.print();
}

RGY_ERR NVEncFilterDenoisePmd::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {

    if (pInputFrame->ptr == nullptr) {
        return RGY_ERR_NONE;
    }
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(m_pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int out_idx = final_dst_index(pPmdParam->pmd.applyCount);

    *pOutputFrameNum = 1;
    RGYFrameInfo *pOutputFrame[2] = {
        &m_pFrameBuf[(m_nFrameIdx++) % m_pFrameBuf.size()].get()->frame,
        &m_pFrameBuf[(m_nFrameIdx++) % m_pFrameBuf.size()].get()->frame,
    };
    bool frame_swapped = false;
    if (ppOutputFrames[0] != nullptr) {
        //filter_as_interlaced_pair()の時の処理
        frame_swapped = true;
        pOutputFrame[out_idx] = ppOutputFrames[0];
        pOutputFrame[(out_idx + 1) & 1]->width     = pOutputFrame[out_idx]->width;
        pOutputFrame[(out_idx + 1) & 1]->height    = pOutputFrame[out_idx]->height;
        pOutputFrame[(out_idx + 1) & 1]->csp       = pOutputFrame[out_idx]->csp;
        pOutputFrame[(out_idx + 1) & 1]->picstruct = pOutputFrame[out_idx]->picstruct;
        pOutputFrame[(out_idx + 1) & 1]->flags     = pOutputFrame[out_idx]->flags;
    } else {
        ppOutputFrames[0] = pOutputFrame[out_idx];
    }
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

    auto ret = denoise(pOutputFrame, &m_Gauss.frame, pInputFrame, stream);
    if (frame_swapped) {
        //filter_as_interlaced_pair()の時の処理
        pOutputFrame[out_idx]->width     = pOutputFrame[(out_idx + 1) & 1]->width;
        pOutputFrame[out_idx]->height    = pOutputFrame[(out_idx + 1) & 1]->height;
        pOutputFrame[out_idx]->csp       = pOutputFrame[(out_idx + 1) & 1]->csp;
        pOutputFrame[out_idx]->picstruct = pOutputFrame[(out_idx + 1) & 1]->picstruct;
        pOutputFrame[out_idx]->flags     = pOutputFrame[(out_idx + 1) & 1]->flags;
    }
    return ret;
}

void NVEncFilterDenoisePmd::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
