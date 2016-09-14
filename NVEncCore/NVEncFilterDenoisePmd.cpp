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

cudaError_t denoise_yv12_u8(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yv12_u8_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u8(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u8_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);

cudaError_t denoise_yv12_u10(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yv12_u10_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u10(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u10_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);

cudaError_t denoise_yv12_u12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yv12_u12_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u12(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u12_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);

cudaError_t denoise_yv12_u14(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yv12_u14_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u14(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u14_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);

cudaError_t denoise_yv12_u16(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yv12_u16_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u16(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);
cudaError_t denoise_yuv444_u16_exp(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame,
    int loop_count, const float strength, const float threshold);

static int final_dst_index(int loop_count) {
    return (loop_count - 1) & 1;
}

NVENCSTATUS NVEncFilterDenoisePmd::denoise(FrameInfo *pOutputFrame[2], FrameInfo *pGauss, const FrameInfo *pInputFrame) {
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(NV_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(m_pParam);
    if (!pPmdParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    struct pmd_func {
        decltype(denoise_yv12_u8)* func[2];
        pmd_func(decltype(denoise_yv12_u8)* useexp, decltype(denoise_yv12_u8)* noexp) {
            func[0] = noexp;
            func[1] = useexp;
        };
    };

    static const std::map<NV_ENC_CSP, pmd_func> denoise_func_list = {
        { NV_ENC_CSP_YV12,      pmd_func(denoise_yv12_u8_exp,    denoise_yv12_u8)  },
        { NV_ENC_CSP_YV12_10,   pmd_func(denoise_yv12_u10_exp,   denoise_yv12_u10) },
        { NV_ENC_CSP_YV12_12,   pmd_func(denoise_yv12_u12_exp,   denoise_yv12_u12) },
        { NV_ENC_CSP_YV12_14,   pmd_func(denoise_yv12_u14_exp,   denoise_yv12_u14) },
        { NV_ENC_CSP_YV12_16,   pmd_func(denoise_yv12_u16_exp,   denoise_yv12_u16) },
        { NV_ENC_CSP_YUV444,    pmd_func(denoise_yuv444_u8_exp,  denoise_yuv444_u8)  },
        { NV_ENC_CSP_YUV444_10, pmd_func(denoise_yuv444_u10_exp, denoise_yuv444_u10) },
        { NV_ENC_CSP_YUV444_12, pmd_func(denoise_yuv444_u12_exp, denoise_yuv444_u12) },
        { NV_ENC_CSP_YUV444_14, pmd_func(denoise_yuv444_u14_exp, denoise_yuv444_u14) },
        { NV_ENC_CSP_YUV444_16, pmd_func(denoise_yuv444_u16_exp, denoise_yuv444_u16) },
    };
    if (denoise_func_list.count(pPmdParam->frameIn.csp) == 0) {
        AddMessage(NV_LOG_ERROR, _T("unsupported csp for denoise(pmd): %s\n"), NV_ENC_CSP_NAMES[pPmdParam->frameIn.csp]);
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

NVENCSTATUS NVEncFilterDenoisePmd::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pPmdParam = std::dynamic_pointer_cast<NVEncFilterParamDenoisePmd>(pParam);
    if (!pPmdParam) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pPmdParam->frameOut.height <= 0 || pPmdParam->frameOut.width <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.applyCount <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter, apply_count must be a positive value.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.strength < 0.0f || 100.0f < pPmdParam->pmd.strength) {
        AddMessage(NV_LOG_WARN, _T("strength must be in range of 0.0 - 100.0.\n"));
        pPmdParam->pmd.strength = clamp(pPmdParam->pmd.strength, 0.0f, 100.0f);
    }
    if (pPmdParam->pmd.threshold < 0.0f || 255.0f < pPmdParam->pmd.threshold) {
        AddMessage(NV_LOG_WARN, _T("strength must be in range of 0.0 - 255.0.\n"));
        pPmdParam->pmd.threshold = clamp(pPmdParam->pmd.threshold, 0.0f, 255.0f);
    }

    auto cudaerr = AllocFrameBuf(pPmdParam->frameOut, 4);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(NV_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }
    pPmdParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    m_Gauss.frame = pPmdParam->frameOut;
    cudaerr = m_Gauss.alloc();
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(NV_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
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
        AddMessage(NV_LOG_ERROR, _T("Invalid parameter type.\n"));
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

    return denoise(pOutputFrame, &m_Gauss.frame, pInputFrame);
}

void NVEncFilterDenoisePmd::close() {
    m_pFrameBuf.clear();
    m_bInterlacedWarn = false;
}
