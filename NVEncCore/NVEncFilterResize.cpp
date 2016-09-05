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
#include "ConvertCsp.h"
#include "NVEncFilter.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

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
    return NPP_SUCCESS;
}

NVENCSTATUS NVEncFilterResize::resizeYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
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
    const auto supportedCspYV12High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == NV_ENC_CSP_YV12) {
        nppsts = resize_yv12<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_C1R, pResizeParam->interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYV12High.begin(), supportedCspYV12High.end(), m_pParam->frameIn.csp) != supportedCspYV12High.end()) {
        nppsts = resize_yv12<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_C1R, pResizeParam->interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
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
    const auto supportedCspYUV444High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == NV_ENC_CSP_YUV444) {
        nppsts = resize_yuv444<Npp8u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_8u_P3R, pResizeParam->interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYUV444High.begin(), supportedCspYUV444High.end(), m_pParam->frameIn.csp) != supportedCspYUV444High.end()) {
        nppsts = resize_yuv444<Npp16u>(pOutputFrame, pInputFrame, nppiResizeSqrPixel_16u_P3R, pResizeParam->interp);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to resize: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
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
    const auto supportedCspYV12   = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12, NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
    const auto supportedCspYUV444 = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444, NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);

    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), m_pParam->frameIn.csp) != supportedCspYV12.end()) {
        sts = resizeYV12(ppOutputFrames[0], pInputFrame);
    } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_pParam->frameIn.csp) != supportedCspYUV444.end()) {
        sts = resizeYUV444(ppOutputFrames[0], pInputFrame);
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
}

void NVEncFilterResize::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
