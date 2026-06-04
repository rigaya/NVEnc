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

#include <memory>
#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamMaa : public NVEncFilterParam {
public:
    VppMaa maa;

    NVEncFilterParamMaa() : maa() {};
    virtual ~NVEncFilterParamMaa() {};
    virtual tstring print() const override { return maa.print(); };
};

class NVEncFilterMaa : public NVEncFilter {
public:
    enum MaaEdgeMode {
        MAA_EDGE_SOBEL = 0,
        MAA_EDGE_PREWITT,
        MAA_EDGE_SOBEL_FULL,
        MAA_EDGE_SCHARR,
        MAA_EDGE_KIRSCH,
        MAA_EDGE_LAPLACIAN
    };

    NVEncFilterMaa();
    virtual ~NVEncFilterMaa();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamMaa> prm);
    RGY_ERR allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label);
    RGY_ERR allocWorkBuf(std::unique_ptr<CUMemBuf>& buf, size_t bytes, const TCHAR *label);

    RGY_ERR fturnLeftFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int planeCount, cudaStream_t stream);
    RGY_ERR fturnRightFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int planeCount, cudaStream_t stream);
    RGY_ERR sangnomPassPlane(const RGYFrameInfo *pSrc, RGYFrameInfo *pDst, RGY_PLANE plane, float aaf, cudaStream_t stream);
    RGY_ERR runEdge(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int mthreshScaled, cudaStream_t stream);
    RGY_ERR runInflate(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream);
    RGY_ERR runMergePlane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA, const RGYFrameInfo *pSrcB,
        const RGYFrameInfo *pMask, RGY_PLANE plane, cudaStream_t stream);
    RGY_ERR runMaskSubsample(RGYFrameInfo *pChromaMaskDst, const RGYFrameInfo *pLumaMaskSrc, cudaStream_t stream);
    RGY_ERR runShowOverlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pMask,
        RGY_PLANE plane, cudaStream_t stream);
    RGY_ERR runShowDarken(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane, cudaStream_t stream);

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
        int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    std::unique_ptr<NVEncFilterResize> m_resizeUp;
    std::unique_ptr<NVEncFilterResize> m_resizeDown;
    std::unique_ptr<NVEncFilterResize> m_resizeUpLuma;
    std::unique_ptr<NVEncFilterResize> m_resizeDownLuma;

    std::unique_ptr<CUFrameBuf> m_supersampled;
    std::unique_ptr<CUFrameBuf> m_rotated;
    std::unique_ptr<CUFrameBuf> m_rotatedAA;
    std::unique_ptr<CUFrameBuf> m_unrotatedAA;
    std::unique_ptr<CUFrameBuf> m_aaResult;
    std::unique_ptr<CUFrameBuf> m_edgeMask;
    std::unique_ptr<CUFrameBuf> m_inflatedMask;
    std::unique_ptr<CUFrameBuf> m_chromaMask;

    static constexpr int MAA_NUM_COST_BUFFERS = 9;
    std::unique_ptr<CUMemBuf> m_costRawPacked;
    std::unique_ptr<CUMemBuf> m_costSmoothPacked;
    int m_costPitch;
    int m_costSliceBytes;
    int m_costElemBytes;

    int m_ssW;
    int m_ssH;
    float m_aaf;
    float m_aacf;
    int m_mthreshScaled;
    MaaEdgeMode m_edgeMode;
};
