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
#include <array>

// dxdyのペアを何並列で同時計算するか
static const int RGY_NLMEANS_DXDY_STEP = 8;

static const int NLEANS_BLOCK_X = 32;
static const int NLEANS_BLOCK_Y = 8;

class NVEncFilterParamDenoiseNLMeans : public NVEncFilterParam {
public:
    std::pair<int, int> compute_capability;
    VppNLMeans nlmeans;

    NVEncFilterParamDenoiseNLMeans() : compute_capability(), nlmeans() {

    };
    virtual ~NVEncFilterParamDenoiseNLMeans() {};
    virtual tstring print() const override;
};

class NVEncFilterDenoiseNLMeans : public NVEncFilter {
public:
    NVEncFilterDenoiseNLMeans();
    virtual ~NVEncFilterDenoiseNLMeans();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    RGY_ERR denoisePlane(
        RGYFrameInfo *pOutputPlane,
        RGYFrameInfo *pTmpUPlane, RGYFrameInfo *pTmpVPlane,
        RGYFrameInfo *pTmpIWPlane,
        const RGYFrameInfo *pInputPlane,
        cudaStream_t stream);
    RGY_ERR denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    std::array<std::unique_ptr<CUFrameBuf>, 2 + 1 + RGY_NLMEANS_DXDY_STEP> m_tmpBuf;
};
