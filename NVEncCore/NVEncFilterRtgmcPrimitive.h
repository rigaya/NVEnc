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

#include <array>

enum class RGYRtgmcPrimitiveOp {
    Copy = 0,
    MakeDiff,
    MakeDiffRemoveGrain20,
    MakeDiffRemoveGrain20AddDiff,
    AddDiff,
    AddWeightedDiff,
    RemoveGrain,
    Repair,
    Merge,
    GaussResize,
    VerticalMin5,
    VerticalMax5,
    LogicMin,
    LogicMax,
};

enum class RGYRtgmcPrimitiveRefMode {
    Disabled = 0,
    RemoveGrain20,
};

class NVEncFilterParamRtgmcPrimitive : public NVEncFilterParam {
public:
    RGYRtgmcPrimitiveOp op;
    RGYRtgmcPrimitiveRefMode refMode;
    int mode;
    float weight;
    int planes;
    bool processChroma;

    NVEncFilterParamRtgmcPrimitive();
    virtual ~NVEncFilterParamRtgmcPrimitive() {}
    virtual tstring print() const override;
};

class NVEncFilterRtgmcPrimitive : public NVEncFilter {
public:
    NVEncFilterRtgmcPrimitive();
    virtual ~NVEncFilterRtgmcPrimitive();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

    static bool needsRef(RGYRtgmcPrimitiveOp op);
    static const TCHAR *opToStr(RGYRtgmcPrimitiveOp op);
    static const TCHAR *refModeToStr(RGYRtgmcPrimitiveRefMode refMode);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcPrimitive> &prm);
    RGY_ERR buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcPrimitive> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        const NVEncFilterParamRtgmcPrimitive &prm,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

    bool processPlane(int iplane, const NVEncFilterParamRtgmcPrimitive &prm) const;
    RGYFrameInfo *generatedRefFrame();

    RGY_ERR setupGaussResize(const NVEncFilterParamRtgmcPrimitive &prm);
    RGY_ERR processGaussResize(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        const NVEncFilterParamRtgmcPrimitive &prm,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);

    std::string m_buildOptions;
    std::array<std::unique_ptr<CUFrameBuf>, 4> m_gaussTmp;
    bool m_useKernel;
};
