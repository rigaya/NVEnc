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

class NVEncFilterParamVinverse : public NVEncFilterParam {
public:
    VppVinverse vinverse;

    NVEncFilterParamVinverse() : vinverse() {};
    virtual ~NVEncFilterParamVinverse() {};
    virtual tstring print() const override;
};

class NVEncFilterVinverse : public NVEncFilter {
public:
    NVEncFilterVinverse();
    virtual ~NVEncFilterVinverse();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
private:
    RGY_ERR procPlaneVblur3(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, cudaStream_t stream);
    RGY_ERR procPlaneVblur35(RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane, const RGYFrameInfo *pInputPlane, cudaStream_t stream);
    RGY_ERR procPlaneMakediff(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pC1Plane, const RGYFrameInfo *pC2Plane, int h_offset, cudaStream_t stream);
    RGY_ERR procPlaneSbrCombine(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pSrcPlane, const RGYFrameInfo *pDiffPlane, const RGYFrameInfo *pBlurPlane, int h_offset, cudaStream_t stream);
    RGY_ERR procPlaneFinalize(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pPb3Plane, const RGYFrameInfo *pPb6Plane,
        float sstr, float scl, int thr_hbd, int amnt_hbd, cudaStream_t stream);
    RGY_ERR procPlane(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane,
        VppVinverseMode mode, float sstr, float scl, int thr_hbd, int amnt_hbd, int h_offset, cudaStream_t stream);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);

    std::vector<std::unique_ptr<CUFrameBuf>> m_pb3;
    std::vector<std::unique_ptr<CUFrameBuf>> m_pb6;
};
