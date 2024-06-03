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

class NVEncFilterParamDenoiseFFT3D : public NVEncFilterParam {
public:
    VppDenoiseFFT3D fft3d;
    std::pair<int, int> compute_capability;

    NVEncFilterParamDenoiseFFT3D() : fft3d(), compute_capability() {

    };
    virtual ~NVEncFilterParamDenoiseFFT3D() {};
    virtual tstring print() const override;
};

class NVEncFilterDenoiseFFT3DBuffer {
public:
    NVEncFilterDenoiseFFT3DBuffer() : m_bufFFT() {};
    ~NVEncFilterDenoiseFFT3DBuffer() {};
    RGY_ERR alloc(int width, int height, RGY_CSP csp, int frames);
    CUFrameBuf *get(const int index) { return m_bufFFT[index % m_bufFFT.size()].get(); }
    void clear() { m_bufFFT.clear(); }
protected:
    std::vector<std::unique_ptr<CUFrameBuf>> m_bufFFT;
};

class NVEncFilterDenoiseFFT3D : public NVEncFilter {
public:
    NVEncFilterDenoiseFFT3D();
    virtual ~NVEncFilterDenoiseFFT3D();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    RGY_ERR checkParam(const NVEncFilterParamDenoiseFFT3D *prm);

    int m_bufIdx;
    int m_ov1;
    int m_ov2;

    NVEncFilterDenoiseFFT3DBuffer m_bufFFT;
    std::unique_ptr<CUFrameBuf> m_filteredBlocks;
    std::unique_ptr<CUMemBuf> m_windowBuf;
    std::unique_ptr<CUMemBuf> m_windowBufInverse;
};
