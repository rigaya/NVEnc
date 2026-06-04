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

#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamChromaShift : public NVEncFilterParam {
public:
    VppChromaShift chromashift;

    NVEncFilterParamChromaShift() : chromashift() {};
    virtual ~NVEncFilterParamChromaShift() {};
    virtual tstring print() const override;
};

class NVEncFilterChromaShift : public NVEncFilter {
public:
    NVEncFilterChromaShift();
    virtual ~NVEncFilterChromaShift();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamChromaShift> pParam);

    std::unique_ptr<CUMemBuf> m_signY;
    std::unique_ptr<CUMemBuf> m_signUV;
    std::unique_ptr<CUMemBuf> m_statsBuf;
    std::vector<int>          m_statsHost;
    std::vector<double>       m_acceptedDx;
    std::vector<double>       m_acceptedDy;
    int                       m_seenAnalysisFrames;
    int                       m_skippedAutoFrames;
    int                       m_warmupSkippedFrames;
    bool                      m_analysisComplete;
    float                     m_resolvedShiftX;
    float                     m_resolvedShiftY;
};
