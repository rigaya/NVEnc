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
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)


NVEncFilterPad::NVEncFilterPad() {
    m_name = _T("pad");
}

NVEncFilterPad::~NVEncFilterPad() {
    close();
}

RGY_ERR NVEncFilterPad::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pPadParam = std::dynamic_pointer_cast<NVEncFilterParamPad>(pParam);
    if (!pPadParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (RGY_CSP_CHROMA_FORMAT[pPadParam->frameIn.csp] == RGY_CHROMAFMT_YUV420
        && (pPadParam->pad.left   % 2 != 0
         || pPadParam->pad.top    % 2 != 0
         || pPadParam->pad.right  % 2 != 0
         || pPadParam->pad.bottom % 2 != 0)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, --vpp-pad only supports values which is multiple of 2 in YUV420.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->frameOut.width != pParam->frameIn.width + pPadParam->pad.right + pPadParam->pad.left
        || pParam->frameOut.height != pParam->frameIn.height + pPadParam->pad.top + pPadParam->pad.bottom) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[pPadParam->encoderCsp] == RGY_CHROMAFMT_YUV420
        && (pParam->frameOut.width  % 2 != 0
         || pParam->frameOut.height % 2 != 0)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, output resolution must be multiple of 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = AllocFrameBuf(pParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamPad::print() const {
    return strsprintf(_T("pad: [%dx%d]->[%dx%d] "),
        frameIn.width, frameIn.height,
        frameOut.width, frameOut.height)
        + pad.print();
}

RGY_ERR NVEncFilterPad::padPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int pad_color, const VppPad *pad, cudaStream_t stream) {
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, pOutputFrame->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pad->right == 0 && pad->left == 0) {
        if (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) {
            auto cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (uint16_t)pad_color, pOutputFrame->width, pad->top, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"), get_err_mes(sts));
                return sts;
            }
            cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr[0] + (pad->top + pInputFrame->height) * pOutputFrame->pitch[0], pOutputFrame->pitch[0],
                (uint16_t)pad_color, pOutputFrame->width, pad->bottom, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"), get_err_mes(sts));
                return sts;
            }
        } else { //RGY_CSP_BIT_DEPTH[pOutputFrame->csp] == 8
            auto cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (uint8_t)pad_color, pOutputFrame->width, pad->top, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"), get_err_mes(sts));
                return sts;
            }

            cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr[0] + (pad->top + pInputFrame->height) * pOutputFrame->pitch[0], pOutputFrame->pitch[0],
                (uint8_t)pad_color, pOutputFrame->width, pad->bottom, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    } else {
        if (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) {
            auto cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (uint16_t)pad_color, pOutputFrame->width, pOutputFrame->height, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"), get_err_mes(sts));
                return sts;
            }
        } else {// RGY_CSP_BIT_DEPTH[pOutputFrame->csp] == 8
            auto cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (uint8_t)pad_color, pOutputFrame->width, pOutputFrame->height, stream);
            if (cudaerr != CUDA_SUCCESS) {
                const auto sts = err_to_rgy(cudaerr);
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }
    auto cudaerr = cudaMemcpy2DAsync(pOutputFrame->ptr[0] + pad->top * pOutputFrame->pitch[0] + pad->left * bytesPerPix(pOutputFrame->csp), pOutputFrame->pitch[0],
            pInputFrame->ptr[0], pInputFrame->pitch[0],
            pInputFrame->width * bytesPerPix(pOutputFrame->csp), pInputFrame->height,
            memcpyKind);
    if (cudaerr != cudaSuccess) {
        const auto sts = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync: %s.\n"), get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterPad::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pPadParam = std::dynamic_pointer_cast<NVEncFilterParamPad>(m_param);
    if (!pPadParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    auto planeOutputU = getPlane(ppOutputFrames[0], RGY_PLANE_U);
    auto planeOutputV = getPlane(ppOutputFrames[0], RGY_PLANE_V);

    const int padColorY = (RGY_CSP_CHROMA_FORMAT[m_param->frameIn.csp] == RGY_CHROMAFMT_RGB) ? 0 : (uint16_t)(16 << (RGY_CSP_BIT_DEPTH[m_param->frameIn.csp] - 8));
    const int padColorC = (RGY_CSP_CHROMA_FORMAT[m_param->frameIn.csp] == RGY_CHROMAFMT_RGB) ? 0 : (uint16_t)(128 << (RGY_CSP_BIT_DEPTH[m_param->frameIn.csp] - 8));

    sts = padPlane(&planeOutputY, &planeInputY, padColorY, &pPadParam->pad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    auto uvPad = pPadParam->pad;
    if (RGY_CSP_CHROMA_FORMAT[m_param->frameIn.csp] == RGY_CHROMAFMT_YUV420) {
        uvPad.right >>= 1;
        uvPad.left >>= 1;
        uvPad.top >>= 1;
        uvPad.bottom >>= 1;
    } else if (RGY_CSP_CHROMA_FORMAT[m_param->frameIn.csp] == RGY_CHROMAFMT_YUV444) {
        //特に何もしない
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }

    sts = padPlane(&planeOutputU, &planeInputU, padColorC, &uvPad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    sts = padPlane(&planeOutputV, &planeInputV, padColorC, &uvPad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    if (rgy_csp_has_alpha(m_param->frameIn.csp)) {
        const auto planeInputA = getPlane(pInputFrame, RGY_PLANE_A);
        auto planeOutputA = getPlane(ppOutputFrames[0], RGY_PLANE_A);
        sts = padPlane(&planeOutputA, &planeInputA, (uint16_t)((1 << RGY_CSP_BIT_DEPTH[m_param->frameIn.csp]) - 1), &pPadParam->pad, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    return sts;
}

void NVEncFilterPad::close() {
    m_frameBuf.clear();
}
