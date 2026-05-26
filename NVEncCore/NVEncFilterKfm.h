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
#include "rgy_filter_kfm_analyze.h"

class NVEncFilterParamKfm : public NVEncFilterParam {
public:
    VppKfm kfm;
    rgy_rational<int> timebase;

    NVEncFilterParamKfm() : kfm(), timebase() {};
    virtual ~NVEncFilterParamKfm() {};
    virtual tstring print() const override;
};

class RGYFrameDataKfmSwitch : public RGYFrameData {
public:
    RGYFrameDataKfmSwitch(int n60, int n24, int baseType, int sourceStart, int numSourceFrames, int duration60, int duration120, int pattern, float cost) :
        m_n60(n60),
        m_n24(n24),
        m_baseType(baseType),
        m_sourceStart(sourceStart),
        m_numSourceFrames(numSourceFrames),
        m_duration60(duration60),
        m_duration120(duration120),
        m_pattern(pattern),
        m_cost(cost) {
        m_dataType = RGY_FRAME_DATA_KFM_SWITCH;
    }
    virtual ~RGYFrameDataKfmSwitch() {};

    int n60() const { return m_n60; }
    int n24() const { return m_n24; }
    int baseType() const { return m_baseType; }
    int sourceStart() const { return m_sourceStart; }
    int numSourceFrames() const { return m_numSourceFrames; }
    int duration60() const { return m_duration60; }
    int duration120() const { return m_duration120; }
    int pattern() const { return m_pattern; }
    float cost() const { return m_cost; }

protected:
    int m_n60;
    int m_n24;
    int m_baseType;
    int m_sourceStart;
    int m_numSourceFrames;
    int m_duration60;
    int m_duration120;
    int m_pattern;
    float m_cost;
};

class NVEncFilterKfm : public NVEncFilter {
public:
    NVEncFilterKfm();
    virtual ~NVEncFilterKfm();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
};
