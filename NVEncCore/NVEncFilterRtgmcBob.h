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

enum class RGYRtgmcBobFieldOrder {
    Auto,
    TFF,
    BFF
};

class NVEncFilterParamRtgmcBob : public NVEncFilterParam {
public:
    RGYRtgmcBobFieldOrder order;
    rgy_rational<int> timebase;

    NVEncFilterParamRtgmcBob() : order(RGYRtgmcBobFieldOrder::Auto), timebase() {};
    virtual ~NVEncFilterParamRtgmcBob() {};
    virtual tstring print() const override;
};

class NVEncFilterRtgmcBob : public NVEncFilter {
public:
    NVEncFilterRtgmcBob();
    virtual ~NVEncFilterRtgmcBob();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcBob> pParam);

    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int preservedParity,
                         const int phaseQuarter,
                         cudaStream_t stream);
    bool getInputTff(const RGYFrameInfo *frame) const;
    void setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames);

    std::string m_buildOptions;
    bool m_defaultTff;
};
