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

class NVEncFilterParamMsmooth : public NVEncFilterParam {
public:
    VppMsmooth msmooth;

    NVEncFilterParamMsmooth() : msmooth() {};
    virtual ~NVEncFilterParamMsmooth() {};
    virtual tstring print() const override;
};

class NVEncFilterMsmooth : public NVEncFilter {
public:
    NVEncFilterMsmooth();
    virtual ~NVEncFilterMsmooth();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
private:
    RGY_ERR procPlaneBlur(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR procPlaneEdgeMask(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBlurFrame,
        float threshold, bool highq, cudaStream_t stream);
    RGY_ERR procPlaneSmooth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pMaskFrame, cudaStream_t stream);
    RGY_ERR procPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int ip, int strength, float threshold, bool highq, cudaStream_t stream);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    std::vector<std::unique_ptr<CUFrameBuf>> m_blur;
    std::vector<std::unique_ptr<CUFrameBuf>> m_mask;
    std::vector<std::unique_ptr<CUFrameBuf>> m_tmp[2];
};
