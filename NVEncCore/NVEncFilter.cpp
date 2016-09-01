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
    m_bCheckPerformance(false), m_peFilterStart(), m_peFilterFin(), m_dFilterTimeMs(0.0), m_nFilterRunCount(0) {

}

NVEncFilter::~NVEncFilter() {
    m_pFrameBuf.clear();
    m_peFilterStart.reset();
    m_peFilterFin.reset();
}

cudaError_t NVEncFilter::AllocFrameBuf(const FrameInfo& frame, int frames) {
    for (int i = 0; i < frames; i++) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(frame));
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

NVENCSTATUS NVEncFilter::filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    if (m_bCheckPerformance) {
        cudaEventRecord(*m_peFilterStart.get());
    }
    const auto ret = run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum);
    if (m_bCheckPerformance) {
        cudaEventRecord(*m_peFilterFin.get());
        cudaEventSynchronize(*m_peFilterFin.get());
        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, *m_peFilterStart.get(), *m_peFilterFin.get());
        m_dFilterTimeMs += time_ms;
        m_nFilterRunCount++;
    }
    return ret;
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
        cudaEventCreate(m_peFilterStart.get());
        cudaEventCreate(m_peFilterFin.get());
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
