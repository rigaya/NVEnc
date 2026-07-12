// -----------------------------------------------------------------------------------------
//     NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
#ifndef __NVENC_FILTER_STDEINT_H__
#define __NVENC_FILTER_STDEINT_H__

#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "rgy_prm.h"
#include "rgy_onnxrt_cuda.h"
#include <memory>
#include <vector>

class NVEncFilterParamStDeint : public NVEncFilterParam {
public:
    tstring modelFile;
    tstring modelDir;
    tstring provider;
    tstring precision;
    tstring cacheDir;
    VppStDeintMode mode;
    CspMatrix colormatrix;
    CspColorRange colorrange;
    int deviceID;

    NVEncFilterParamStDeint() :
        modelFile(), modelDir(), provider(_T("auto")), precision(_T("fp32")), cacheDir(), mode(VppStDeintMode::Bob),
        colormatrix(RGY_MATRIX_AUTO), colorrange(RGY_COLORRANGE_AUTO), deviceID(-1) {};
    virtual tstring print() const override;
};

struct NVEncStDeintColorCoeffs {
    float yOff, yScale, yRange, cOff, cScale, cRange;
    float matVR, matUG, matVG, matUB;
    float matRY, matGY, matBY, matRU, matGU, matBU, matRV, matGV, matBV;
};

RGY_ERR run_stdeint_pack_rgb(const RGYFrameInfo *input, float *output,
    const NVEncStDeintColorCoeffs& coeffs, cudaStream_t stream);
RGY_ERR run_stdeint_weave_yuv(RGYFrameInfo *output, const float *input,
    const float *restoration, bool frameA, const NVEncStDeintColorCoeffs& coeffs,
    cudaStream_t stream);

class NVEncFilterStDeint : public NVEncFilter {
public:
    NVEncFilterStDeint();
    virtual ~NVEncFilterStDeint();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
        int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    void yuvToRGB(const RGYFrameInfo& input, float *dst);
    void rgbToYUV(const RGYFrameInfo& output, const float *src);
    void setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax);
    void setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const;
    void setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs);
    void weaveRestoration(float *dst, const float *restoration, bool frameA) const;
    RGY_ERR initCudaPath(cudaStream_t stream);
    RGY_ERR runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        const int sourceIndices[2], cudaStream_t stream);
    RGY_ERR runHost(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        const int sourceIndices[2], cudaStream_t stream);
    NVEncStDeintColorCoeffs colorCoeffs() const;

    std::unique_ptr<RGYOnnxRTCUDA> m_ov;
    int m_width;
    int m_height;
    VppStDeintMode m_mode;
    bool m_defaultTff;
    bool m_havePrevTimestamp;
    int64_t m_prevTimestamp;
    int64_t m_prevDuration;

    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV;

    std::vector<float> m_inputBuf;
    std::vector<float> m_outputBuf;
    std::vector<float> m_weaveBuf;
    std::unique_ptr<CUFrameBuf> m_inputStaging;
    std::vector<std::unique_ptr<CUFrameBuf>> m_outputStaging;
    std::unique_ptr<CUMemBuf> m_inputDevice;
    std::unique_ptr<CUMemBuf> m_outputDevice;
    tstring m_modelPath;
    RGYOnnxRTProvider m_provider;
    tstring m_precision;
    tstring m_cacheDir;
    int m_deviceID;
    bool m_cudaPathTried;
    bool m_cudaPath;
};

#endif //__NVENC_FILTER_STDEINT_H__
