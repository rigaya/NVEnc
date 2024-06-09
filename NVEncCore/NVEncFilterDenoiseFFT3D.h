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

static std::pair<int, int> getBlockCount(const int width, const int height, const int block_size, const int ov1, const int ov2) {
    const int block_eff = block_size - ov1 * 2 - ov2;
    const int block_count_x = (width + block_eff - 1) / block_eff;
    const int block_count_y = (height + block_eff - 1) / block_eff;
    return std::make_pair(block_count_x, block_count_y);
}

typedef RGY_ERR (*func_fft3d_fft)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int ov1, const int ov2,
    const float *ptrBlockWindow, cudaStream_t stream);
typedef RGY_ERR (*func_fft3d_tfft_filter_ifft)(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrameA, const RGYFrameInfo *pInputFrameB, const RGYFrameInfo *pInputFrameC, const RGYFrameInfo *pInputFrameD,
    const float *ptrBlockWindowInverse,
    const int widthY, const int heightY, const int widthUV, const int heightUV, const int ov1, const int ov2,
    const float sigma, const float limit, const int filterMethod, cudaStream_t stream);
typedef RGY_ERR (*func_fft3d_merge)(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int ov1, const int ov2, cudaStream_t stream);

class DenoiseFFT3DBase {
public:
    DenoiseFFT3DBase() {};
    virtual ~DenoiseFFT3DBase() {};

    virtual func_fft3d_fft fft() = 0;
    virtual func_fft3d_tfft_filter_ifft tfft_filter_ifft(int temporalCurrentIdx, int temporalCount) = 0;
    virtual func_fft3d_merge merge() = 0;
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
