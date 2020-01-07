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

#include <map>
#include <array>
#include "convert_csp.h"
#include "NVEncFilterRff.h"
#include "NVEncParam.h"
#pragma warning (push)


NVEncFilterRff::NVEncFilterRff() :
    m_nStatus(),
    m_fieldBuf(),
    m_nFieldBufUsed(-1),
    m_nFieldBufPicStruct(RGY_FRAME_FLAG_NONE) {
    m_sFilterName = _T("rff");
}

NVEncFilterRff::~NVEncFilterRff() {
    close();
}

RGY_ERR NVEncFilterRff::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pRffParam = std::dynamic_pointer_cast<NVEncFilterParamRff>(pParam);
    if (!pRffParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    pRffParam->frameOut.pitch = pRffParam->frameIn.pitch;

    if (cmpFrameInfoCspResolution(&m_fieldBuf.frame, &pRffParam->frameOut)) {
        m_fieldBuf.frame = pRffParam->frameOut;
        auto cudaerr = m_fieldBuf.alloc();
        if (cudaerr != CUDA_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    m_nFieldBufUsed = -1;
    m_nPathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT));

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamRff::print() const {
    return _T("rff");
}

RGY_ERR NVEncFilterRff::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    auto pRffParam = std::dynamic_pointer_cast<NVEncFilterParamRff>(m_pParam);
    if (!pRffParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //出力先のフレーム
    auto *pOutFrame = ppOutputFrames[0];

    if ((pInputFrame->flags & (RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY)) == RGY_FRAME_FLAG_RFF) {
        //RGY_FRAME_FLAG_RFFフラグのみ立っているとき、状態を反転する
        m_nStatus ^= 1;
    }

    const auto frameInfoEx = getFrameInfoExtra(pInputFrame);

    //コピー先
    RGY_FRAME_FLAGS bufPicStruct = RGY_FRAME_FLAG_NONE;
    int bufDst = -1; //コピーしない
    if (m_nStatus == 1) {
        //コピー先の決定
        bufDst = (m_nFieldBufUsed < 0) ? 0 : m_nFieldBufUsed ^ 1;

        //フィールドをバッファにコピー
        bufPicStruct = pInputFrame->flags & (RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        auto cudaerr = cudaMemcpy2DAsync(m_fieldBuf.frame.ptr + m_fieldBuf.frame.pitch * bufDst, m_fieldBuf.frame.pitch * 2,
            pInputFrame->ptr + pInputFrame->pitch * ((bufPicStruct & RGY_FRAME_FLAG_RFF_BFF) ? 1 : 0), pInputFrame->pitch * 2,
            frameInfoEx.width_byte, frameInfoEx.height_total >> 1, cudaMemcpyDeviceToDevice, cudaStreamDefault);
        if (cudaerr != CUDA_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame to field buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }
    if (m_nFieldBufUsed >= 0) {
        //バッファからフィールドをコピー
        auto cudaerr = cudaMemcpy2DAsync(pOutFrame->ptr + pOutFrame->pitch * ((m_nFieldBufPicStruct & RGY_FRAME_FLAG_RFF_BFF) ? 1 : 0), pOutFrame->pitch * 2,
            m_fieldBuf.frame.ptr + m_fieldBuf.frame.pitch * m_nFieldBufUsed, m_fieldBuf.frame.pitch * 2,
            frameInfoEx.width_byte, frameInfoEx.height_total >> 1, cudaMemcpyDeviceToDevice, cudaStreamDefault);
        if (cudaerr != CUDA_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame to field buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }
    m_nFieldBufUsed = bufDst;
    m_nFieldBufPicStruct = bufPicStruct;

    const int input_tff = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_TFF) ? 1 : 0;
    const int input_rff = (pInputFrame->flags & RGY_FRAME_FLAG_RFF) ? 1 : 0;
    const int output_tff = (input_tff + (input_rff ^ m_nStatus)) & 1;

    pOutFrame->picstruct = (output_tff) ? RGY_PICSTRUCT_FRAME_TFF : RGY_PICSTRUCT_FRAME_BFF;
    return sts;
}

void NVEncFilterRff::close() {
    m_pFrameBuf.clear();
    m_fieldBuf.clear();
}
