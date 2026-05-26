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

#pragma once

#include "NVEncFilter.h"
#include "NVEncFilterDegrainMV.h"

class NVEncFilterParamRtgmcLossless : public NVEncFilterParam {
public:
    int level;
    int inputType;
    int sourceField;

    NVEncFilterParamRtgmcLossless() : level(0), inputType(0), sourceField(0) {}
    virtual ~NVEncFilterParamRtgmcLossless() {}
    virtual tstring print() const override;
};

class NVEncFilterRtgmcLossless : public NVEncFilter {
public:
    NVEncFilterRtgmcLossless();
    virtual ~NVEncFilterRtgmcLossless();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGY_ERR run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcLossless> &prm);
    RGY_ERR buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcLossless> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR processFrameFused(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR processFramePassSplit(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

    std::string m_buildOptions;
    bool m_useKernel;
};
