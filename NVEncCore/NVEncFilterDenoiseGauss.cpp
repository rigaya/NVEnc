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
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

template<typename T, typename Tfunc>
static RGY_ERR denoise_nnpi_gauss_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, Tfunc funcGauss, NppiMaskSize masksize) {
    //const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    //const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    auto dstSize = nppisize(pOutputFrame);
    NppiPoint srcOffset = { 0 };
    NppStatus sts = funcGauss(
        (const T *)pInputFrame->ptrArray[0],
        pInputFrame->pitchArray[0], srcSize, srcOffset,
        (T *)pOutputFrame->ptrArray[0],
        pOutputFrame->pitchArray[0], dstSize, masksize, NPP_BORDER_REPLICATE);
    if (sts != NPP_SUCCESS) {
        return err_to_rgy(sts);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseGauss::denoisePlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pGaussParam = std::dynamic_pointer_cast<NVEncFilterParamGaussDenoise>(m_pParam);
    if (!pGaussParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_BIT_DEPTH[pGaussParam->frameIn.csp] <= 8) {
        sts = denoise_nnpi_gauss_plane<Npp8u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_8u_C1R, pGaussParam->masksize);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), sts, get_err_mes(sts));
            sts = RGY_ERR_UNKNOWN;
        }
    } else if (RGY_CSP_BIT_DEPTH[pGaussParam->frameIn.csp] <= 16) {
        sts = denoise_nnpi_gauss_plane<Npp16u>(pOutputFrame, pInputFrame, nppiFilterGaussBorder_16u_C1R, pGaussParam->masksize);
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

RGY_ERR NVEncFilterDenoiseGauss::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame) {
    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    auto err = denoisePlane(&planeOutputY, &planeInputY);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = denoisePlane(&planeOutputU, &planeInputU);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = denoisePlane(&planeOutputV, &planeInputV);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return err;
}

NVEncFilterDenoiseGauss::NVEncFilterDenoiseGauss() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("gauss");
}

NVEncFilterDenoiseGauss::~NVEncFilterDenoiseGauss() {
    close();
}

RGY_ERR NVEncFilterDenoiseGauss::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
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
        pGaussParam->frameOut.pitchArray[i] = m_pFrameBuf[0]->frame.pitchArray[i];
    }

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamGaussDenoise::print() const {
    return strsprintf(_T("denoise(gauss): mask size: %s"),
        get_chr_from_value(list_nppi_gauss, masksize));
}

RGY_ERR NVEncFilterDenoiseGauss::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    if (pInputFrame->ptrArray[0] == nullptr) {
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
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto st = nppGetStream();
    if (st != stream) {
        nppSetStream(stream);
    }
    return denoiseFrame(ppOutputFrames[0], pInputFrame);
}

void NVEncFilterDenoiseGauss::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
