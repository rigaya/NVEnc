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

#pragma once

#include <array>
#include "NVEncFilter.h"
#include "NVEncParam.h"

class NVEncFilterParamBwdif : public NVEncFilterParam {
public:
    VppBwdif bwdif;
    rgy_rational<int> timebase;

    NVEncFilterParamBwdif() : bwdif(), timebase() {};
    virtual ~NVEncFilterParamBwdif() {};
    virtual tstring print() const override { return bwdif.print(); };
};

RGY_ERR run_bwdif_frame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pPrev2,
    const RGYFrameInfo *pPrev,
    const RGYFrameInfo *pCur,
    const RGYFrameInfo *pNext,
    const RGYFrameInfo *pNext2,
    const int preserveTopField,
    const int thr,
    cudaStream_t stream);

class NVEncFilterBwdif : public NVEncFilter {
public:
    NVEncFilterBwdif();
    virtual ~NVEncFilterBwdif();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR check_param(shared_ptr<NVEncFilterParamBwdif> prm);

    RGY_ERR reconstructFrame(int idx_prev, int idx_cur, int idx_next,
                             bool inputTff, int preserveTopField, int outputSlot,
                             cudaStream_t stream);
    RGY_ERR generateOutput(int idx_prev, int idx_cur, int idx_next,
                           RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                           cudaStream_t stream);
    void setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames);
    bool getInputTff(const RGYFrameInfo *frame) const;
    bool shouldPassthrough(const RGYFrameInfo *frame) const;

    std::array<CUFrameBuf, 3> m_cacheFrames;
    int  m_inputCount;
    bool m_drained;
    bool m_defaultTff;
};
