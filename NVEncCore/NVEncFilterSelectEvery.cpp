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
#include "NVEncFilterSelectEvery.h"
#include "NVEncParam.h"
#pragma warning (push)


NVEncFilterSelectEvery::NVEncFilterSelectEvery() :
    m_frames(0),
    m_totalDuration(0),
    m_outPts(-1) {
    m_sFilterName = _T("selectevery");
}

NVEncFilterSelectEvery::~NVEncFilterSelectEvery() {
    close();
}

RGY_ERR NVEncFilterSelectEvery::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto pSelectParam = std::dynamic_pointer_cast<NVEncFilterParamSelectEvery>(pParam);
    if (!pSelectParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pSelectParam->frameOut.pitch[i] = pSelectParam->frameIn.pitch[i];
    }
    pParam->baseFps /= pSelectParam->selectevery.step;

    sts = AllocFrameBuf(pSelectParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return sts;
    }

    m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));

    setFilterInfo(pParam->print());
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamSelectEvery::print() const {
    return selectevery.print();
}

RGY_ERR NVEncFilterSelectEvery::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto pSelectParam = std::dynamic_pointer_cast<NVEncFilterParamSelectEvery>(m_pParam);
    if (!pSelectParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    ppOutputFrames[0] = nullptr;
    *pOutputFrameNum = 0;
    if (pInputFrame->ptr[0] == nullptr) {
        if (m_totalDuration > 0
            && m_frames % pSelectParam->selectevery.step != (pSelectParam->selectevery.step-1)
            && m_frames % pSelectParam->selectevery.step >= (pSelectParam->selectevery.step / 2)) {
            auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            ppOutputFrames[0]->duration = m_totalDuration * pSelectParam->selectevery.step / (m_frames % pSelectParam->selectevery.step);
            ppOutputFrames[0]->timestamp = m_outPts;
            *pOutputFrameNum = 1;
            if (m_totalDuration < 0 || m_outPts < 0) {
                AddMessage(RGY_LOG_ERROR, _T("Invalid timestamp calculated for output: timestamp %lld, duration %lld.\n"),
                           m_outPts, m_totalDuration);
                return RGY_ERR_INVALID_PARAM;
            }
            m_outPts = -1;
            m_totalDuration = 0;
        }
        return sts;
    }

    if (m_outPts < 0) {
        if (pInputFrame->timestamp < 0 || pInputFrame->duration < 0) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid timestamp from input, frame #%d: timestamp %lld, duration %lld.\n"),
                m_frames, pInputFrame->timestamp, pInputFrame->duration);
            return RGY_ERR_INVALID_PARAM;
        }
        m_outPts = pInputFrame->timestamp;
    }
    m_totalDuration += pInputFrame->duration;

    if (m_frames % pSelectParam->selectevery.step == pSelectParam->selectevery.offset) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, pOutFrame->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        sts = copyFrameAsync(&pOutFrame->frame, pInputFrame, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame to buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        pOutFrame->frame.inputFrameId = pInputFrame->inputFrameId;
    }
    if (m_frames % pSelectParam->selectevery.step == (pSelectParam->selectevery.step-1)) {
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
            ppOutputFrames[0]->duration = m_totalDuration;
            ppOutputFrames[0]->timestamp = m_outPts;
            *pOutputFrameNum = 1;
            if (m_totalDuration < 0 || m_outPts < 0) {
                AddMessage(RGY_LOG_ERROR, _T("Invalid timestamp calculated for output: timestamp %lld, duration %lld.\n"),
                           m_outPts, m_totalDuration);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        m_outPts = -1;
        m_totalDuration = 0;
    }

    m_frames++;
    return sts;
}

void NVEncFilterSelectEvery::close() {
    m_pFrameBuf.clear();
}
