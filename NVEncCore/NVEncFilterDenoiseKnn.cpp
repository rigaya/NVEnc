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
#include "NVEncFilterDenoiseKnn.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int KNN_RADIUS_MAX = 5;

cudaError_t denoise_yv12_u8(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float noise, const float lerpC, const float weight_threshold, const float lerp_threshold);
cudaError_t denoise_yv12_u16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float noise, const float lerpC, const float weight_threshold, const float lerp_threshold);
cudaError_t denoise_yuv444_u8(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float noise, const float lerpC, const float weight_threshold, const float lerp_threshold);
cudaError_t denoise_yuv444_u16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame,
    int radius, const float noise, const float lerpC, const float weight_threshold, const float lerp_threshold);

NVENCSTATUS NVEncFilterDenoiseKnn::denoiseYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(m_pParam);
    if (!pKnnParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    cudaError_t cudaerr = cudaSuccess;
    const auto supportedCspYV12High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YV12_09, NV_ENC_CSP_YV12_10, NV_ENC_CSP_YV12_12, NV_ENC_CSP_YV12_14, NV_ENC_CSP_YV12_16);
    if (pKnnParam->frameIn.csp == NV_ENC_CSP_YV12) {
        cudaerr = denoise_yv12_u8(pOutputFrame, pInputFrame, pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);
        if (cudaerr != NV_ENC_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYV12High.begin(), supportedCspYV12High.end(), pKnnParam->frameIn.csp) != supportedCspYV12High.end()) {
        cudaerr = denoise_yv12_u16(pOutputFrame, pInputFrame, pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);
        if (cudaerr != NV_ENC_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncFilterDenoiseKnn::denoiseYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(m_pParam);
    if (!pKnnParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    cudaError_t cudaerr = cudaSuccess;
    const auto supportedCspYUV444High = make_array<NV_ENC_CSP>(NV_ENC_CSP_YUV444_09, NV_ENC_CSP_YUV444_10, NV_ENC_CSP_YUV444_12, NV_ENC_CSP_YUV444_14, NV_ENC_CSP_YUV444_16);
    if (pKnnParam->frameIn.csp == NV_ENC_CSP_YUV444) {
        cudaerr = denoise_yuv444_u8(pOutputFrame, pInputFrame, pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);
        if (cudaerr != NV_ENC_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else if (std::find(supportedCspYUV444High.begin(), supportedCspYUV444High.end(), pKnnParam->frameIn.csp) != supportedCspYUV444High.end()) {
        cudaerr = denoise_yuv444_u16(pOutputFrame, pInputFrame, pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);
        if (cudaerr != NV_ENC_SUCCESS) {
            AddMessage(NV_LOG_ERROR, _T("failed to denoise: %d, %s.\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            sts = NV_ENC_ERR_GENERIC;
        }
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return NV_ENC_SUCCESS;
}

NVEncFilterDenoiseKnn::NVEncFilterDenoiseKnn() : m_bInterlacedWarn(false) {
    m_sFilterName = _T("knn");
}

NVEncFilterDenoiseKnn::~NVEncFilterDenoiseKnn() {
    close();
}

NVENCSTATUS NVEncFilterDenoiseKnn::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pKnnParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseKnn>(pParam);
    if (!pKnnParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pKnnParam->frameOut.height <= 0 || pKnnParam->frameOut.width <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius < 0) {
        AddMessage(NV_LOG_ERROR, _T("radius must be a positive value.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius > KNN_RADIUS_MAX) {
        AddMessage(NV_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.strength < 0.0 || 1.0 < pKnnParam->knn.strength) {
        AddMessage(NV_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerpC < 0.0 || 1.0 < pKnnParam->knn.lerpC) {
        AddMessage(NV_LOG_ERROR, _T("lerpC should be 0.0 - 1.0.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerp_threshold < 0.0 || 1.0 < pKnnParam->knn.lerp_threshold) {
        AddMessage(NV_LOG_ERROR, _T("th_lerp should be 0.0 - 1.0.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.weight_threshold < 0.0 || 1.0 < pKnnParam->knn.weight_threshold) {
        AddMessage(NV_LOG_ERROR, _T("th_weight should be 0.0 - 1.0.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pKnnParam->frameOut, 2);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(NV_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pKnnParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    m_sFilterInfo = strsprintf(_T("denoise(knn): radius %d, strength %.2f, lerp %.2f\n                              th_weight %.2f, th_lerp %.2f"),
        pKnnParam->knn.radius, pKnnParam->knn.strength, pKnnParam->knn.lerpC, pKnnParam->knn.weight_threshold, pKnnParam->knn.lerp_threshold);

    m_pParam = pParam;
    return sts;
}

NVENCSTATUS NVEncFilterDenoiseKnn::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
    }
    ppOutputFrames[0]->interlaced = pInputFrame->interlaced;
    if (pInputFrame->interlaced && !m_bInterlacedWarn) {
        AddMessage(NV_LOG_WARN, _T("Interlaced denoise is not supported, denoise as progressive.\n"));
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
        sts = denoiseYV12(ppOutputFrames[0], pInputFrame);
    } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), m_pParam->frameIn.csp) != supportedCspYUV444.end()) {
        sts = denoiseYUV444(ppOutputFrames[0], pInputFrame);
    } else {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp.\n"));
        sts = NV_ENC_ERR_UNIMPLEMENTED;
    }
    return sts;
}

void NVEncFilterDenoiseKnn::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
