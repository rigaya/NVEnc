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

#include <deque>
#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamDeflicker : public NVEncFilterParam {
public:
    VppDeflicker deflicker;

    NVEncFilterParamDeflicker() : deflicker() {};
    virtual ~NVEncFilterParamDeflicker() {};
    virtual tstring print() const override;
};

class NVEncFilterDeflicker : public NVEncFilter {
public:
    NVEncFilterDeflicker();
    virtual ~NVEncFilterDeflicker();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamDeflicker> pParam);

    RGY_ERR computePlaneStats(const RGYFrameInfo *pPlane, double& meanOut, double& stddevOut, cudaStream_t stream);
    RGY_ERR runApply(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcPlane,
        float mult, float add, float blend, int is_chroma, cudaStream_t stream);

    std::unique_ptr<CUMemBuf> m_sumBuf;
    std::unique_ptr<CUMemBuf> m_sumSqBuf;
    std::vector<int64_t>      m_sumHost;
    std::vector<int64_t>      m_sumSqHost;
    size_t                    m_statsBufWGCount;
    std::deque<double>        m_rollingMeans;
    std::deque<double>        m_rollingSigmas;
    double                    m_prevMult;
    double                    m_prevAdd;
    bool                      m_haveDamping;
    int                       m_skippedSceneFrames;
};
