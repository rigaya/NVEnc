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
static NppStatus unsharp_yv12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pScratch, Tfunc func,
    float radius, float sigma, float weight, float threshold) {
    const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    NppiPoint srcOffset = { 0 };
    auto dstSize = nppisize(pOutputFrame);
    //Y
    NppStatus sts = func(
        (const T *)pInputFrame->ptr,
        pInputFrame->pitch, srcOffset,
        (T *)pOutputFrame->ptr,
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //U
    srcSize.width  >>= 1;
    srcSize.height >>= 1;
    dstSize.width  >>= 1;
    dstSize.height >>= 1;
    sts = func(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height),
        pInputFrame->pitch, srcOffset,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height),
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //V
    sts = func(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 3 / 2),
        pInputFrame->pitch, srcOffset,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 3 / 2),
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    return NPP_SUCCESS;
}

NVENCSTATUS NVEncFilterUnsharp::unsharpYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pScratch) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_pParam);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    static const auto supportedCspYV12High = make_array<RGY_CSP>(RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == RGY_CSP_YV12) {
        nppsts = unsharp_yv12<Npp8u>(pOutputFrame, pInputFrame, pScratch, nppiFilterUnsharpBorder_8u_C1R, pUnsharpParam->radius, pUnsharpParam->sigma, pUnsharpParam->weight, pUnsharpParam->threshold);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to unsharp: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYV12High.begin(), supportedCspYV12High.end(), m_pParam->frameIn.csp) != supportedCspYV12High.end()) {
        nppsts = unsharp_yv12<Npp16u>(pOutputFrame, pInputFrame, pScratch, nppiFilterUnsharpBorder_16u_C1R, pUnsharpParam->radius, pUnsharpParam->sigma, pUnsharpParam->weight, pUnsharpParam->threshold);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to unsharp: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
}

template<typename T, typename Tfunc>
static NppStatus unsharp_yuv444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pScratch, Tfunc func,
    float radius, float sigma, float weight, float threshold) {
    const double factorX = pOutputFrame->width / (double)pInputFrame->width;
    const double factorY = pOutputFrame->height / (double)pInputFrame->height;
    auto srcSize = nppisize(pInputFrame);
    NppiPoint srcOffset = { 0 };
    auto dstSize = nppisize(pOutputFrame);
    //Y
    NppStatus sts = func(
        (const T *)pInputFrame->ptr,
        pInputFrame->pitch, srcOffset,
        (T *)pOutputFrame->ptr,
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    sts = func(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height),
        pInputFrame->pitch, srcOffset,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height),
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    if (sts != NPP_SUCCESS) {
        return sts;
    }
    //V
    sts = func(
        (const T *)((const uint8_t *)pInputFrame->ptr + pInputFrame->pitch * pInputFrame->height * 2),
        pInputFrame->pitch, srcOffset,
        (T *)((uint8_t *)pOutputFrame->ptr + pOutputFrame->pitch * pOutputFrame->height * 2),
        pOutputFrame->pitch, dstSize,
        radius, sigma, weight, threshold, NPP_BORDER_REPLICATE, (Npp8u *)pScratch->ptr);
    return NPP_SUCCESS;
}

NVENCSTATUS NVEncFilterUnsharp::unsharpYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, CUMemBuf *pScratch) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(m_pParam);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    static const auto supportedCspYUV444High = make_array<RGY_CSP>(RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    NppStatus nppsts = NPP_SUCCESS;
    if (m_pParam->frameIn.csp == RGY_CSP_YUV444) {
        nppsts = unsharp_yuv444<Npp8u>(pOutputFrame, pInputFrame, pScratch, nppiFilterUnsharpBorder_8u_C1R, pUnsharpParam->radius, pUnsharpParam->sigma, pUnsharpParam->weight, pUnsharpParam->threshold);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to unsharp: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYUV444High.begin(), supportedCspYUV444High.end(), m_pParam->frameIn.csp) != supportedCspYUV444High.end()) {
        nppsts = unsharp_yuv444<Npp16u>(pOutputFrame, pInputFrame, pScratch, nppiFilterUnsharpBorder_16u_C1R, pUnsharpParam->radius, pUnsharpParam->sigma, pUnsharpParam->weight, pUnsharpParam->threshold);
        if (nppsts != NPP_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to unsharp: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
}

NVEncFilterUnsharp::NVEncFilterUnsharp() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("unsharp");
}

NVEncFilterUnsharp::~NVEncFilterUnsharp() {
    close();
}

cudaError_t NVEncFilterUnsharp::AllocScratch(int nScratchSize) {
    for (uint32_t i = 0; i < m_pFrameBuf.size(); i++) {
        unique_ptr<CUMemBuf> uptr(new CUMemBuf(nScratchSize));
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_pScratchBuf.clear();
            return ret;
        }
        m_pScratchBuf.push_back(std::move(uptr));
    }
    m_nFrameIdx = 0;
    return cudaSuccess;
}

NVENCSTATUS NVEncFilterUnsharp::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pUnsharpParam = std::dynamic_pointer_cast<NVEncFilterParamUnsharp>(pParam);
    if (!pUnsharpParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (!check_if_nppi_dll_available()) {
        AddMessage(RGY_LOG_ERROR, _T("vpp-unsharp requires \"%s\", not available on your system.\n"), NPPI_DLL_NAME);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pUnsharpParam->frameOut.height <= 0 || pUnsharpParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pUnsharpParam->frameOut, 2);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pUnsharpParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    int nBufferSize = 0;
    auto nppsts = nppiFilterUnsharpGetBufferSize_8u_C1R(pUnsharpParam->radius, pUnsharpParam->sigma, &nBufferSize);
    if (nppsts != NPP_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to unsharp: %d, %s.\n"), nppsts, char_to_tstring(_cudaGetErrorEnum(nppsts)).c_str());
        sts = NV_ENC_ERR_GENERIC;
    }
    cudaerr = AllocScratch(nBufferSize);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate scratch memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_sFilterInfo = strsprintf(_T("unsharp: radius %.1f, sigma %.1f, weight %.1f, threshold %.1f"),
        pUnsharpParam->radius, pUnsharpParam->sigma, pUnsharpParam->weight, pUnsharpParam->threshold);

    //コピーを保存
    m_pParam = pUnsharpParam;
    return sts;
}

NVENCSTATUS NVEncFilterUnsharp::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    *pOutputFrameNum = 1;
    CUMemBuf *pScratch = nullptr;
    if (ppOutputFrames[0] == nullptr) {
        pScratch = m_pScratchBuf[m_nFrameIdx].get();
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (pInputFrame->interlaced && !m_bInterlacedWarn) {
        AddMessage(RGY_LOG_WARN, _T("Interlaced unsharp is not supported, unsharp as progressive.\n"));
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
    static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);

    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), m_pParam->frameIn.csp) != supportedCspYV12.end()) {
        sts = unsharpYV12(ppOutputFrames[0], pInputFrame, pScratch);
    } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_pParam->frameIn.csp) != supportedCspYUV444.end()) {
        sts = unsharpYUV444(ppOutputFrames[0], pInputFrame, pScratch);
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
}

void NVEncFilterUnsharp::close() {
    m_pFrameBuf.clear();
    m_pScratchBuf.clear();
    m_bInterlacedWarn = false;
}
