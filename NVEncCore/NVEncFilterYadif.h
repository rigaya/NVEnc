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

enum YadifTargetField {
    YADIF_GEN_FIELD_UNKNOWN = -1,
    YADIF_GEN_FIELD_TOP = 0,
    YADIF_GEN_FIELD_BOTTOM
};

class NVEncFilterParamYadif : public NVEncFilterParam {
public:
    VppYadif yadif;
    rgy_rational<int> timebase;

    NVEncFilterParamYadif() : yadif(), timebase() {

    };
    virtual ~NVEncFilterParamYadif() {};
    virtual tstring print() const override;
};

class NVEncFilterYadifSource {
public:
    NVEncFilterYadifSource();
    ~NVEncFilterYadifSource();
    RGY_ERR add(const RGYFrameInfo *pInputFrame, cudaStream_t stream = 0);
    RGY_ERR alloc(const RGYFrameInfo& frameInfo);
    void clear();
    CUFrameBuf *get(int iframe) {
        iframe = clamp(iframe, 0, m_nFramesInput-1);
        return &m_buf[iframe % m_buf.size()];
    }
    int inframe() const { return m_nFramesInput; }
private:
    int m_nFramesInput;
    int m_nFramesOutput;
    std::array<CUFrameBuf, 4> m_buf;
};

class NVEncFilterYadif : public NVEncFilter {
public:
    NVEncFilterYadif();
    virtual ~NVEncFilterYadif();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR check_param(shared_ptr<NVEncFilterParamYadif> prmYadif);
    void setBobTimestamp(const int iframe, RGYFrameInfo **ppOutputFrames);

    int m_nFrame;
    int64_t m_pts;
    NVEncFilterYadifSource m_source;
};
