﻿// -----------------------------------------------------------------------------------------
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
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

template<typename T, typename Tfunc>
static RGY_ERR denoise_nnpi_gauss_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, Tfunc funcGauss, NppiMaskSize masksize, cudaStream_t stream) {
    //const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    //const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    auto dstSize = nppisize(pOutputFrame);
    NppiPoint srcOffset = { 0 };
    
#if CUDA_VERSION >= 13000
    NppStreamContext nppStreamCtx = {};
    nppStreamCtx.hStream = stream;
    int dev = 0;
    cudaGetDevice(&dev);
    nppStreamCtx.nCudaDeviceId = dev;
    cudaDeviceProp prop = {};
    cudaGetDeviceProperties(&prop, dev);
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
#endif
    NppStatus sts = funcGauss(
        (const T *)pInputFrame->ptr[0],
        pInputFrame->pitch[0], srcSize, srcOffset,
        (T *)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0], dstSize, masksize, NPP_BORDER_REPLICATE
#if CUDA_VERSION >= 13000
        , nppStreamCtx
#endif
        );
    if (sts != NPP_SUCCESS) {
        return err_to_rgy(sts);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseGauss::denoisePlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pGaussParam = std::dynamic_pointer_cast<NVEncFilterParamGaussDenoise>(m_param);
    if (!pGaussParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_BIT_DEPTH[pGaussParam->frameIn.csp] <= 8) {
        
#if CUDA_VERSION >= 13000
        sts = denoise_nnpi_gauss_plane<Npp8u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_8u_C1R_Ctx, pGaussParam->masksize, stream);
#else
        sts = denoise_nnpi_gauss_plane<Npp8u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_8u_C1R, pGaussParam->masksize, stream);
#endif
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), sts, get_err_mes(sts));
            sts = RGY_ERR_UNKNOWN;
        }
    } else if (RGY_CSP_BIT_DEPTH[pGaussParam->frameIn.csp] <= 16) {
#if CUDA_VERSION >= 13000
        sts = denoise_nnpi_gauss_plane<Npp16u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_16u_C1R_Ctx, pGaussParam->masksize, stream);
#else
        sts = denoise_nnpi_gauss_plane<Npp16u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_16u_C1R, pGaussParam->masksize, stream);
#endif
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), sts, get_err_mes(sts));
            sts = RGY_ERR_UNKNOWN;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseGauss::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto sts = denoisePlane(&planeOutput, &planeSrc, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoiseGauss::NVEncFilterDenoiseGauss() : m_bInterlacedWarn(false) {
    m_name = _T("gauss");
}

NVEncFilterDenoiseGauss::~NVEncFilterDenoiseGauss() {
    close();
}

RGY_ERR NVEncFilterDenoiseGauss::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pGaussParam = std::dynamic_pointer_cast<NVEncFilterParamGaussDenoise>(pParam);
    if (!pGaussParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!check_if_nppi_dll_available()) {
        AddMessage(RGY_LOG_ERROR, _T("vpp-gauss requires \"%s\", not available on your system.\n"), NPPI_DLL_NAME_TSTR);
        return RGY_ERR_NOT_FOUND;
    }
    //パラメータチェック
    if (pGaussParam->frameOut.height <= 0 || pGaussParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = AllocFrameBuf(pGaussParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pGaussParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamGaussDenoise::print() const {
    return strsprintf(_T("denoise(gauss): mask size: %s"),
        get_chr_from_value(list_nppi_gauss, masksize));
}

RGY_ERR NVEncFilterDenoiseGauss::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    return denoiseFrame(ppOutputFrames[0], pInputFrame, stream);
}

void NVEncFilterDenoiseGauss::close() {
    m_frameBuf.clear();
    m_bInterlacedWarn = false;
}
