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

NVEncFilter::NVEncFilter() : m_sFilterName(), m_sFilterInfo(), m_pPrintMes(), m_pFrameBuf(), m_nFrameIdx(0) {

}

NVEncFilter::~NVEncFilter() {
    m_pFrameBuf.clear();
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
