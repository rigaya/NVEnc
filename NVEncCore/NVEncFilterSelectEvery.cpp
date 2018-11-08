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

NVENCSTATUS NVEncFilterSelectEvery::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pSelectParam = std::dynamic_pointer_cast<NVEncFilterParamSelectEvery>(pParam);
    if (!pSelectParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    pSelectParam->frameOut.pitch = pSelectParam->frameIn.pitch;
    pParam->baseFps /= pSelectParam->selectevery.step;

    auto cudaerr = AllocFrameBuf(pSelectParam->frameOut, 1);
    if (cudaerr != CUDA_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
        return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));

    m_sFilterInfo = strsprintf(_T("selectevery %d (offset %d)"), pSelectParam->selectevery.step, pSelectParam->selectevery.offset);
    m_pParam = pParam;
    return sts;
}

NVENCSTATUS NVEncFilterSelectEvery::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    auto pSelectParam = std::dynamic_pointer_cast<NVEncFilterParamSelectEvery>(m_pParam);
    if (!pSelectParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }

    ppOutputFrames[0] = nullptr;
    *pOutputFrameNum = 0;
    if (pInputFrame->ptr == nullptr) {
        if (m_frames % pSelectParam->selectevery.step != (pSelectParam->selectevery.step-1)
            && m_frames % pSelectParam->selectevery.step >= (pSelectParam->selectevery.step / 2)) {
            auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            if (m_frames % pSelectParam->selectevery.step < pSelectParam->selectevery.offset) {
                const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, pOutFrame->frame.deivce_mem);
                if (memcpyKind != cudaMemcpyDeviceToDevice) {
                    AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
                    return NV_ENC_ERR_UNSUPPORTED_PARAM;
                }
                const auto frameInfoEx = getFrameInfoExtra(pInputFrame);
                auto cudaerr = cudaMemcpy2DAsync(pOutFrame->frame.ptr, pOutFrame->frame.pitch,
                    pInputFrame->ptr, pInputFrame->pitch, frameInfoEx.width_byte, frameInfoEx.height_total,
                    cudaMemcpyDeviceToDevice);
                if (cudaerr != CUDA_SUCCESS) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy frame to buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                    return NV_ENC_ERR_OUT_OF_MEMORY;
                }
            }
            ppOutputFrames[0] = &pOutFrame->frame;
            ppOutputFrames[0]->duration = m_totalDuration * pSelectParam->selectevery.step / ((m_frames % pSelectParam->selectevery.step) + 1);
            ppOutputFrames[0]->timestamp = m_outPts;
            *pOutputFrameNum = 1;
        }
        return sts;
    }

    if (m_outPts < 0) {
        m_outPts = pInputFrame->timestamp;
    }
    m_totalDuration += pInputFrame->duration;

    if (m_frames % pSelectParam->selectevery.step == pSelectParam->selectevery.offset) {
        auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, pOutFrame->frame.deivce_mem);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        const auto frameInfoEx = getFrameInfoExtra(pInputFrame);
        auto cudaerr = cudaMemcpy2DAsync(pOutFrame->frame.ptr, pOutFrame->frame.pitch,
            pInputFrame->ptr, pInputFrame->pitch, frameInfoEx.width_byte, frameInfoEx.height_total,
            cudaMemcpyDeviceToDevice);
        if (cudaerr != CUDA_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame to buffer: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
    }
    if (m_frames % pSelectParam->selectevery.step == (pSelectParam->selectevery.step-1)) {
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_pFrameBuf[m_nFrameIdx].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            m_nFrameIdx = (m_nFrameIdx + 1) % m_pFrameBuf.size();
            ppOutputFrames[0]->duration = m_totalDuration;
            ppOutputFrames[0]->timestamp = m_outPts;
            *pOutputFrameNum = 1;
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
