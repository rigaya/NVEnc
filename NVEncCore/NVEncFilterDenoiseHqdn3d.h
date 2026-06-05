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

static const int HQDN3D_LUT_RADIUS = 256;

class NVEncFilterParamDenoiseHqdn3d : public NVEncFilterParam {
public:
    VppHqdn3d hqdn3d;

    NVEncFilterParamDenoiseHqdn3d() : hqdn3d() {};
    virtual ~NVEncFilterParamDenoiseHqdn3d() {};
    virtual tstring print() const override;
};

class NVEncFilterDenoiseHqdn3d : public NVEncFilter {
public:
    NVEncFilterDenoiseHqdn3d();
    virtual ~NVEncFilterDenoiseHqdn3d();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
private:
    RGY_ERR denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
        CUMemBuf *pCoefSpatial, CUMemBuf *pCoefTemporal,
        CUMemBuf *pPrev, int prevPitchFloats, cudaStream_t stream);
    RGY_ERR denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    static void precalcCoefs(std::vector<float> &table, double dist25);

    std::array<std::unique_ptr<CUMemBuf>, 4> m_coefs;
    std::vector<std::unique_ptr<CUMemBuf>> m_framePrev;
    std::vector<int> m_framePrevPitchFloats;
    std::unique_ptr<CUMemBuf> m_tmpH;
    std::unique_ptr<CUMemBuf> m_tmpHV;
    int m_tmpPitchFloats;
    bool m_firstFrame;
};
