// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "NVEncFilterKfm.h"

tstring NVEncFilterParamKfm::print() const {
    return kfm.print();
}

NVEncFilterKfm::NVEncFilterKfm() :
    NVEncFilter() {
    m_name = _T("kfm");
}

NVEncFilterKfm::~NVEncFilterKfm() {
    close();
}

RGY_ERR NVEncFilterKfm::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamKfm>(pParam);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    m_param = pParam;
    m_pLog = pPrintMes;
    AddMessage(RGY_LOG_ERROR, _T("KFM CUDA filter body is not wired yet.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterKfm::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(pInputFrame);
    UNREFERENCED_PARAMETER(ppOutputFrames);
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    UNREFERENCED_PARAMETER(stream);
    return RGY_ERR_UNSUPPORTED;
}

void NVEncFilterKfm::close() {
    m_frameBuf.clear();
}
