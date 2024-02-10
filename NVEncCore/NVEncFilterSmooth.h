// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2020 rigaya
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
#include "rgy_prm.h"

static const int VPP_SMOOTH_MAX_QUALITY_LEVEL = 6;

class NVEncFilterParamSmooth : public NVEncFilterParam {
public:
    VppSmooth smooth;
    std::pair<int, int> compute_capability;
    RGYListRef<RGYFrameDataQP> *qpTableRef;

    NVEncFilterParamSmooth() : smooth(), compute_capability(), qpTableRef(nullptr) {

    };
    virtual ~NVEncFilterParamSmooth() {};
    virtual tstring print() const override;
};

class NVEncFilterSmooth : public NVEncFilter {
public:
    NVEncFilterSmooth();
    virtual ~NVEncFilterSmooth();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    int qp_size(int res) { return divCeil(res + 15, 16); }
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR check_param(shared_ptr<NVEncFilterParamSmooth> prmYadif);
    float getQPMul(int qp_scale);

    CUFrameBuf m_qp;
    std::shared_ptr<RGYFrameDataQP> m_qpSrc;
    std::shared_ptr<RGYFrameDataQP> m_qpSrcB;
    RGYListRef<RGYFrameDataQP> *m_qpTableRef;
    int m_qpTableErrCount;
};
