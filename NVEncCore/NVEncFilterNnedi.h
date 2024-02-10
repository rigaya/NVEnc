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

enum NnediTargetField {
    NNEDI_GEN_FIELD_UNKNOWN = -1,
    NNEDI_GEN_FIELD_TOP = 0,
    NNEDI_GEN_FIELD_BOTTOM
};

class NVEncFilterParamNnedi : public NVEncFilterParam {
public:
    VppNnedi nnedi;
    std::pair<int, int> compute_capability;
    HMODULE hModule;
    rgy_rational<int> timebase;

    NVEncFilterParamNnedi() : nnedi(), compute_capability(std::make_pair(0, 0)), hModule(NULL), timebase() {};
    virtual ~NVEncFilterParamNnedi() {};
    virtual tstring print() const override;
};

class NVEncFilterNnedi : public NVEncFilter {
public:
    static const int weight_loop_1;
    static const int sizeNX[];
    static const int sizeNY[];
    static const int sizeNN[];
    static const int maxVal = 65535 >> 8;
public:
    NVEncFilterNnedi();
    virtual ~NVEncFilterNnedi();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamNnedi> pParam);
    virtual RGY_ERR initParams(const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam);
    void setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames);

    template<typename TypeWeight>
    void setWeight0(TypeWeight *ptrDst, const float *ptrW, const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam);

    template<typename TypeWeight>
    void setWeight1(TypeWeight *ptrDst, const float *ptrW, const std::shared_ptr<NVEncFilterParamNnedi> pNnediParam);
    virtual shared_ptr<const float> readWeights(const tstring& weightFile, HMODULE hModule);

    CUMemBuf m_weight0;
    std::array<CUMemBuf, 2> m_weight1;
};
