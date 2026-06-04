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

class NVEncFilterParamColorFix : public NVEncFilterParam {
public:
    VppColorFix colorfix;
    VideoVUIInfo vui;

    NVEncFilterParamColorFix() : colorfix(), vui() {};
    virtual ~NVEncFilterParamColorFix() {};
    virtual tstring print() const override;
};

class NVEncFilterColorFix : public NVEncFilter {
public:
    NVEncFilterColorFix();
    virtual ~NVEncFilterColorFix();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamColorFix> pParam);
    int resolveMatrix(const VppColorFix& cf, const VideoVUIInfo& vui, int height) const;
    int resolveSpace(const VppColorFix& cf) const;
    void getMatrixCoeffs(int resolvedMatrix, float& Kr, float& Kg, float& Kb) const;
    RGY_ERR setupCspConverters(const RGYFrameInfo& frameIn, RGY_CSP cspRgb, rgy_rational<int> baseFps);
    RGY_ERR allocReduceBuffer(const RGYFrameInfo& frameIn);
    RGY_ERR runApplyRGB(RGYFrameInfo *pTarget, float scaleR, float scaleG, float scaleB, float offsetR, float offsetG, float offsetB, cudaStream_t stream);
    RGY_ERR runReduceUV(RGYFrameInfo *pSrc, cudaStream_t stream);
    RGY_ERR runReduceRGB(RGYFrameInfo *pSrc, cudaStream_t stream);
    RGY_ERR runApplyUV(RGYFrameInfo *pTarget, int offsetU, int offsetV, cudaStream_t stream);
    RGY_ERR runApplyLuma(RGYFrameInfo *pTarget, float scaleY, float offsetY, cudaStream_t stream);
    RGY_ERR finaliseReduction(cudaStream_t stream, int numLongsPerGroup, std::vector<long long>& outTotals);

    int m_resolvedMatrix;
    int m_effectiveSpace;
    std::unique_ptr<NVEncFilterCspCrop> m_convToRgb;
    std::unique_ptr<NVEncFilterCspCrop> m_convToYuv;
    RGY_CSP m_cspRgb;
    std::unique_ptr<CUMemBuf> m_reducePartials;
    int m_numGroupsLastDispatch;
    bool m_analysisComplete;
    int m_analysedFrames;
    int m_skippedFrames;
    int m_totalSeenFrames;
    long long m_sumA;
    long long m_sumB;
    long long m_sumC;
    long long m_sumY;
    long long m_sumYsq;
    double m_rollingVarianceSum;
    int m_rollingVarianceCount;
    int m_offsetU;
    int m_offsetV;
    float m_scaleR;
    float m_scaleG;
    float m_scaleB;
};
