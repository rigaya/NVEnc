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
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)


NVEncFilterPad::NVEncFilterPad() {
    m_sFilterName = _T("pad");
}

NVEncFilterPad::~NVEncFilterPad() {
    close();
}

RGY_ERR NVEncFilterPad::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pPadParam = std::dynamic_pointer_cast<NVEncFilterParamPad>(pParam);
    if (!pPadParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (   pPadParam->pad.left   % 2 != 0
        || pPadParam->pad.top    % 2 != 0
        || pPadParam->pad.right  % 2 != 0
        || pPadParam->pad.bottom % 2 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->frameOut.width != pParam->frameIn.width + pPadParam->pad.right + pPadParam->pad.left
        || pParam->frameOut.height != pParam->frameIn.height + pPadParam->pad.top + pPadParam->pad.bottom) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto cudaerr = AllocFrameBuf(pParam->frameOut, 1);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return RGY_ERR_MEMORY_ALLOC;
    }
    pParam->frameOut.pitch = m_pFrameBuf[0]->frame.pitch;

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamPad::print() const {
    return strsprintf(_T("pad: [%dx%d]->[%dx%d] "),
        frameIn.width, frameIn.height,
        frameOut.width, frameOut.height)
        + pad.print();
}

RGY_ERR NVEncFilterPad::padPlane(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, int pad_color, const VppPad *pad, cudaStream_t stream) {
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, pOutputFrame->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pad->right == 0 && pad->left == 0) {
        if (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) {
            auto cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr, pOutputFrame->pitch,
                (uint16_t)pad_color, pOutputFrame->width, pad->top, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }
            cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr + (pad->top + pInputFrame->height) * pOutputFrame->pitch, pOutputFrame->pitch,
                (uint16_t)pad_color, pOutputFrame->width, pad->bottom, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }
        } else { //RGY_CSP_BIT_DEPTH[pOutputFrame->csp] == 8
            auto cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr, pOutputFrame->pitch,
                (uint8_t)pad_color, pOutputFrame->width, pad->top, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }

            cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr + (pad->top + pInputFrame->height) * pOutputFrame->pitch, pOutputFrame->pitch,
                (uint8_t)pad_color, pOutputFrame->width, pad->bottom, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }
        }
    } else {
        if (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) {
            auto cudaerr = cuMemsetD2D16Async((CUdeviceptr)pOutputFrame->ptr, pOutputFrame->pitch,
                (uint16_t)pad_color, pOutputFrame->width, pOutputFrame->height, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D16Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }
        } else {// RGY_CSP_BIT_DEPTH[pOutputFrame->csp] == 8
            auto cudaerr = cuMemsetD2D8Async((CUdeviceptr)pOutputFrame->ptr, pOutputFrame->pitch,
                (uint8_t)pad_color, pOutputFrame->width, pOutputFrame->height, stream);
            if (cudaerr != CUDA_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("error at cuMemsetD2D8Async: %s.\n"),
                    char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            }
        }
    }
    const int pixel_byte = RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8 ? 2 : 1;
    auto cudaerr = cudaMemcpy2DAsync(pOutputFrame->ptr + pad->top * pOutputFrame->pitch + pad->right * pixel_byte, pOutputFrame->pitch,
            pInputFrame->ptr, pInputFrame->pitch,
            pInputFrame->width * pixel_byte, pInputFrame->height,
            memcpyKind);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at cudaMemcpy2DAsync: %s.\n"),
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterPad::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pPadParam = std::dynamic_pointer_cast<NVEncFilterParamPad>(m_pParam);
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

    sts = padPlane(&planeOutputY, &planeInputY, (uint16_t)(16 << (RGY_CSP_BIT_DEPTH[m_pParam->frameIn.csp] - 8)), &pPadParam->pad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    auto uvPad = pPadParam->pad;
    if (RGY_CSP_CHROMA_FORMAT[m_pParam->frameIn.csp] == RGY_CHROMAFMT_YUV420) {
        uvPad.right >>= 1;
        uvPad.left >>= 1;
        uvPad.top >>= 1;
        uvPad.bottom >>= 1;
    } else if (RGY_CSP_CHROMA_FORMAT[m_pParam->frameIn.csp] == RGY_CHROMAFMT_YUV444) {
        //特に何もしない
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }

    sts = padPlane(&planeOutputU, &planeInputU, (uint16_t)(128 << (RGY_CSP_BIT_DEPTH[m_pParam->frameIn.csp] - 8)), &uvPad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    sts = padPlane(&planeOutputV, &planeInputV, (uint16_t)(128 << (RGY_CSP_BIT_DEPTH[m_pParam->frameIn.csp] - 8)), &uvPad, stream);
    if (sts != RGY_ERR_NONE) return sts;

    return sts;
}

void NVEncFilterPad::close() {
    m_pFrameBuf.clear();
}
