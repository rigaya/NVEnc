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

#include "NVEncFilter.h"

NVEncFilter::NVEncFilter() :
    m_sFilterName(), m_sFilterInfo(), m_pPrintMes(), m_pFrameBuf(), m_nFrameIdx(0),
    m_pFieldPairIn(), m_pFieldPairOut(),
    m_pParam(),
    m_nPathThrough(FILTER_PATHTHROUGH_ALL), m_bCheckPerformance(false),
    m_peFilterStart(), m_peFilterFin(), m_dFilterTimeMs(0.0), m_nFilterRunCount(0) {

}

NVEncFilter::~NVEncFilter() {
    m_pFrameBuf.clear();
    m_pFieldPairIn.reset();
    m_pFieldPairOut.reset();
    m_peFilterStart.reset();
    m_peFilterFin.reset();
    m_pParam.reset();
}

cudaError_t NVEncFilter::AllocFrameBuf(const FrameInfo& frame, int frames) {
    if (m_pFrameBuf.size() == frames
        && !cmpFrameInfoCspResolution(&m_pFrameBuf[0]->frame, &frame)) {
        //すべて確保されているか確認
        bool allocated = true;
        for (int i = 0; i < m_pFrameBuf.size(); i++) {
            if (m_pFrameBuf[i]->frame.ptr == nullptr) {
                allocated = false;
                break;
            }
        }
        if (allocated) {
            return cudaSuccess;
        }
    }
    m_pFrameBuf.clear();

    for (int i = 0; i < frames; i++) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(frame));
        uptr->frame.ptr = nullptr;
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_pFrameBuf.clear();
            return ret;
        }
        m_pFrameBuf.push_back(std::move(uptr));
    }
    m_nFrameIdx = 0;
    return cudaSuccess;
}

NVENCSTATUS NVEncFilter::filter(FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    cudaError_t cudaerr = cudaSuccess;
    if (m_bCheckPerformance) {
        cudaerr = cudaEventRecord(*m_peFilterStart.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventRecord(m_peFilterStart): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
    }

    if (pInputFrame == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    if (m_pParam
        && m_pParam->bOutOverwrite //上書きか?
        && pInputFrame != nullptr && pInputFrame->ptr != nullptr //入力が存在するか?
        && ppOutputFrames != nullptr && ppOutputFrames[0] == nullptr) { //出力先がセット可能か?
        ppOutputFrames[0] = pInputFrame;
        *pOutputFrameNum = 1;
    }
    const auto ret = run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum);
    const int nOutFrame = *pOutputFrameNum;
    if (!m_pParam->bOutOverwrite && nOutFrame > 0) {
        if (m_nPathThrough & FILTER_PATHTHROUGH_TIMESTAMP) {
            if (nOutFrame != 1) {
                AddMessage(RGY_LOG_ERROR, _T("timestamp path through can only be applied to 1-in/1-out filter.\n"));
                return NV_ENC_ERR_INVALID_CALL;
            } else {
                ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
                ppOutputFrames[0]->duration  = pInputFrame->duration;
            }
        }
        for (int i = 0; i < nOutFrame; i++) {
            if (m_nPathThrough & FILTER_PATHTHROUGH_FLAGS)     ppOutputFrames[i]->flags     = pInputFrame->flags;
            if (m_nPathThrough & FILTER_PATHTHROUGH_PICSTRUCT) ppOutputFrames[i]->picstruct = pInputFrame->picstruct;
        }
    }
    if (m_bCheckPerformance) {
        cudaerr = cudaEventRecord(*m_peFilterFin.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventRecord(m_peFilterFin): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
        cudaerr = cudaEventSynchronize(*m_peFilterFin.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventSynchronize(m_peFilterFin): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
        float time_ms = 0.0f;
        cudaerr = cudaEventElapsedTime(&time_ms, *m_peFilterStart.get(), *m_peFilterFin.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventElapsedTime(m_peFilterStart - m_peFilterFin): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
        m_dFilterTimeMs += time_ms;
        m_nFilterRunCount++;
    }
    return ret;
}

NVENCSTATUS NVEncFilter::filter_as_interlaced_pair(const FrameInfo *pInputFrame, FrameInfo *pOutputFrame, cudaStream_t stream) {
    if (!m_pFieldPairIn) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(*pInputFrame));
        uptr->frame.ptr = nullptr;
        uptr->frame.pitch = 0;
        uptr->frame.height >>= 1;
        uptr->frame.picstruct = RGY_PICSTRUCT_FRAME;
        uptr->frame.flags &= ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_pFrameBuf.clear();
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
        m_pFieldPairIn = std::move(uptr);
    }
    if (!m_pFieldPairOut) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(*pOutputFrame));
        uptr->frame.ptr = nullptr;
        uptr->frame.pitch = 0;
        uptr->frame.height >>= 1;
        uptr->frame.picstruct = RGY_PICSTRUCT_FRAME;
        uptr->frame.flags &= ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_pFrameBuf.clear();
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
        m_pFieldPairOut = std::move(uptr);
    }
    const auto inputFrameInfoEx = getFrameInfoExtra(pInputFrame);
    const auto outputFrameInfoEx = getFrameInfoExtra(pOutputFrame);

    for (int i = 0; i < 2; i++) {
        auto cudaerr = cudaMemcpy2DAsync(m_pFieldPairIn->frame.ptr, m_pFieldPairIn->frame.pitch,
            pInputFrame->ptr + pInputFrame->pitch * i, pInputFrame->pitch * 2,
            inputFrameInfoEx.width_byte, inputFrameInfoEx.height_total >> 1,
            cudaMemcpyDeviceToDevice, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to seprate field(0): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
        int nFieldOut = 0;
        auto pFieldOut = &m_pFieldPairOut->frame;
        auto err = run_filter(&m_pFieldPairIn->frame, &pFieldOut, &nFieldOut);
        if (err != NV_ENC_SUCCESS) {
            return err;
        }
        cudaerr = cudaMemcpy2DAsync(pOutputFrame->ptr + pOutputFrame->pitch * i, pOutputFrame->pitch * 2,
            pFieldOut->ptr, pFieldOut->pitch,
            outputFrameInfoEx.width_byte, outputFrameInfoEx.height_total >> 1,
            cudaMemcpyDeviceToDevice, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to merge field(1): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    }
    return NV_ENC_SUCCESS;
}

void NVEncFilter::CheckPerformance(bool flag) {
    if (flag == m_bCheckPerformance) {
        return;
    }
    m_bCheckPerformance = flag;
    if (!m_bCheckPerformance) {
        m_peFilterStart.reset();
        m_peFilterFin.reset();
    } else {
        auto deleter = [](cudaEvent_t *pEvent) {
            cudaEventDestroy(*pEvent);
            delete pEvent;
        };
        m_peFilterStart = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        m_peFilterFin = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        auto cudaerr = cudaEventCreate(m_peFilterStart.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventCreate(m_peFilterStart): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreate(m_peFilterStart)\n"));

        cudaerr = cudaEventCreate(m_peFilterFin.get());
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed cudaEventCreate(m_peFilterFin): %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreate(m_peFilterFin)\n"));
        m_dFilterTimeMs = 0.0;
        m_nFilterRunCount = 0;
    }
}

double NVEncFilter::GetAvgTimeElapsed() {
    if (!m_bCheckPerformance) {
        return 0.0;
    }
    return m_dFilterTimeMs / (double)m_nFilterRunCount;
}

bool check_if_nppi_dll_available() {
    HMODULE hModule = LoadLibrary(NPPI_DLL_NAME_TSTR);
    if (hModule == NULL)
        return false;
    FreeLibrary(hModule);
    return true;
}
