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
#ifndef __NVENC_FILTER_ONNX_DEINT_H__
#define __NVENC_FILTER_ONNX_DEINT_H__

#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "rgy_prm.h"
#include "rgy_onnxrt_cuda.h"
#include <array>
#include <memory>
#include <vector>

// 選択したモデル方式のテンソル契約を一か所で保持する。
// modelHeight/modelWidth は入力フレーム解像度から初期化時に決定する。
struct OnnxDeintModelSpec {
    VppOnnxDeintArchitecture architecture;
    int inputChannels;
    int outputChannels;
    int modelHeight;
    int modelWidth;
    int outputHeight;
    int outputWidth;
    int lookaheadFrames;
    bool supportsSharedCuda;
};

class NVEncFilterParamOnnxDeint : public NVEncFilterParam {
public:
    tstring modelFile;
    tstring modelDir;
    tstring provider;
    tstring precision;
    tstring cacheDir;
    VppOnnxDeintMode mode;
    CspMatrix colormatrix;
    CspColorRange colorrange;
    int deviceID;

    NVEncFilterParamOnnxDeint() :
        modelFile(), modelDir(), provider(_T("auto")), precision(_T("fp32")), cacheDir(), mode(VppOnnxDeintMode::Bob),
        colormatrix(RGY_MATRIX_AUTO), colorrange(RGY_COLORRANGE_AUTO), deviceID(-1) {};
    virtual tstring print() const override;
};

RGY_ERR run_onnx_deint_weave_rgb(float *output, const float *input,
    const float *restoration, bool frameA, int width, int height, cudaStream_t stream);

class NVEncFilterOnnxDeint : public NVEncFilter {
public:
    NVEncFilterOnnxDeint();
    virtual ~NVEncFilterOnnxDeint();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
        int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    void setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const;
    void setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs);
    RGY_ERR copyProgressiveOutputs(const RGYFrameInfo *input, RGYFrameInfo **outputs, int *outputCount, cudaStream_t stream);
    RGYFrameInfo rgbFrame(float *ptr) const;
    RGY_ERR convertToRgb(const RGYFrameInfo *input, cudaStream_t stream);
    RGY_ERR convertFromRgb(RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR initCudaPath(cudaStream_t stream);
    RGY_ERR runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        const int sourceIndices[2], cudaStream_t stream);
    RGY_ERR runHost(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        const int sourceIndices[2], cudaStream_t stream);

    // 9ch入力/3ch出力で前後のフィールドを参照するtemporalモデル用の経路。
    struct TemporalFrame {
        std::unique_ptr<CUFrameBuf> frame;
        std::vector<float> rgb;
        bool tff;
        bool interlaced;
        TemporalFrame() : frame(), rgb(), tff(true), interlaced(false) {};
    };
    RGY_ERR allocTemporalRing(const RGYFrameInfo& frameInfo);
    RGY_ERR addTemporalFrame(const RGYFrameInfo *input, cudaStream_t stream);
    int temporalFieldParity(const int fieldIndex) const;
    void buildTemporalInput(const int frameIndex, const int fieldPos);
    void combineTemporalOutput(const int frameIndex, const int fieldPos, float *dst) const;
    RGY_ERR procTemporalField(const int frameIndex, const int fieldPos, RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR emitTemporalFrame(const int frameIndex, RGYFrameInfo **outputs, int *outputFrameNum, cudaStream_t stream);
    RGY_ERR runTemporal(const RGYFrameInfo *input, RGYFrameInfo **outputs, int *outputFrameNum, cudaStream_t stream);

    std::unique_ptr<RGYOnnxRTCUDA> m_ov;
    std::unique_ptr<NVEncFilterCspCrop> m_cropToRgb;
    std::unique_ptr<NVEncFilterCspCrop> m_cropFromRgb;
    int m_width;
    int m_height;
    VppOnnxDeintMode m_mode;
    bool m_defaultTff;
    bool m_havePrevTimestamp;
    int64_t m_prevTimestamp;
    int64_t m_prevDuration;

    std::vector<float> m_inputBuf;
    std::vector<float> m_outputBuf;
    std::unique_ptr<CUMemBuf> m_inputDevice;
    std::unique_ptr<CUMemBuf> m_outputDevice;
    std::unique_ptr<CUMemBuf> m_weaveDevice;
    tstring m_modelName;
    tstring m_modelPath;
    OnnxDeintModelSpec m_spec;
    RGYOnnxRTProvider m_provider;
    tstring m_precision;
    tstring m_cacheDir;
    int m_deviceID;
    bool m_cudaPathTried;
    bool m_cudaPath;
    int m_framesIn;
    int m_frameOut;
    std::vector<float> m_weaveBuf;
    std::array<TemporalFrame, 3> m_temporalRing;
};

#endif //__NVENC_FILTER_ONNX_DEINT_H__
