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
#include <vector>
#include "NVEncFilter.h"
#include "rgy_prm.h"

struct AVFormatContext;
struct AVStream;

class NVEncFilterParamDescale : public NVEncFilterParam {
public:
    VppDescale descale;
    tstring inputFilePath;
    int probeStartFrame;
    int probeEndFrame;

    NVEncFilterParamDescale() : descale(), inputFilePath(), probeStartFrame(0), probeEndFrame(0) {};
    virtual ~NVEncFilterParamDescale() {};
    virtual tstring print() const override { return descale.print(); };
};

class NVEncFilterDescaleCore {
public:
    NVEncFilterDescaleCore() : src_dim(0), dst_dim(0), bandwidth(0), c(0), weights_columns(0),
        weights(), left_idx(), right_idx(), lower(), upper(), diagonal() {};
    ~NVEncFilterDescaleCore() {};

    int src_dim;
    int dst_dim;
    int bandwidth;
    int c;
    int weights_columns;

    std::unique_ptr<CUMemBuf> weights;
    std::unique_ptr<CUMemBuf> left_idx;
    std::unique_ptr<CUMemBuf> right_idx;
    std::unique_ptr<CUMemBuf> lower;
    std::unique_ptr<CUMemBuf> upper;
    std::unique_ptr<CUMemBuf> diagonal;
};

class NVEncFilterDescale : public NVEncFilter {
public:
    NVEncFilterDescale();
    virtual ~NVEncFilterDescale();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
private:
    RGY_ERR prepareCore(NVEncFilterDescaleCore &core, int src_dim, int dst_dim,
        VppDescaleKernel kernel, double b, double c_param,
        double shift, VppDescaleBorder border);
    RGY_ERR runProbe(NVEncFilterParamDescale *prm);

    struct ProbeForwardWeights {
        int weights_columns;
        std::unique_ptr<CUMemBuf> weights;
        std::unique_ptr<CUMemBuf> left_idx;
        std::unique_ptr<CUMemBuf> right_idx;
    };
    struct ProbeCandidate {
        VppDescaleKernel kernel;
        float b;
        float c;
        int width;
        int height;
        double mse;
        tstring label;
    };
    RGY_ERR buildForwardWeights(ProbeForwardWeights &fw, int src_dim_low, int dst_dim_high,
        VppDescaleKernel kernel, double b, double c_param,
        double shift, VppDescaleBorder border);
    RGY_ERR scoreCandidates(std::vector<ProbeCandidate> &candidates,
        const std::vector<std::unique_ptr<CUMemBuf>> &lumaBufs,
        const std::vector<std::unique_ptr<CUMemBuf>> &edgeWeightsBufs,
        int src_w, int src_h, int src_pixel_bytes,
        bool symmetricForward = false);
    RGY_ERR runResolutionSearch(NVEncFilterParamDescale *prm,
        std::vector<ProbeCandidate> &candidates,
        const std::vector<std::unique_ptr<CUMemBuf>> &lumaBufs,
        const std::vector<std::unique_ptr<CUMemBuf>> &edgeWeightsBufs,
        int src_w, int src_h, int src_pixel_bytes,
        AVFormatContext *fmtCtx, AVStream *videoStream);
    RGY_ERR runHPlane(RGYFrameInfo *pIntermediateFloat, const RGYFrameInfo *pInputPlane,
        const NVEncFilterDescaleCore &core, cudaStream_t stream);
    RGY_ERR runVPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pIntermediateFloat,
        CUMemBuf *pVScratch, const NVEncFilterDescaleCore &core, cudaStream_t stream);

    std::array<std::array<NVEncFilterDescaleCore, 2>, 2> m_cores;
    std::array<std::unique_ptr<CUMemBuf>, 4> m_intermediateH;
    std::array<std::unique_ptr<CUMemBuf>, 4> m_intermediateV;
    std::array<int, 4> m_intermediatePitchFloats;
    int m_frameIdx;
};
