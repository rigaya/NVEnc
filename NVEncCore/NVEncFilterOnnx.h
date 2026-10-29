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
#ifndef __NVENC_FILTER_ONNX_H__
#define __NVENC_FILTER_ONNX_H__

#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "rgy_prm.h"
#include "rgy_onnxrt_cuda.h"
#include "NVEncFilterOnnxKernels.h"
#include <vector>
#include <memory>
#include <deque>

class NVEncFilterResize; // opt-in end-of-chain resize sub-filter (out_res=/resize=)

RGY_ERR run_onnx_pack_luma(float *dst, const RGYFrameInfo *src, cudaStream_t stream);
RGY_ERR run_onnx_unpack_luma(RGYFrameInfo *dst, const float *src, cudaStream_t stream);

class NVEncFilterParamOnnx : public NVEncFilterParam {
public:
    VppOnnx onnx;
    tstring modelDir;
    int sar[2] = { 0, 0 };  // input SAR (set by pipeline) -- resolves a negative out_res= (auto-aspect) DAR-correctly
    int deviceID = -1;      // CUDA device ordinal NVEnc selected (binds inference to the encoder's GPU; -1 = current device)
    NVEncFilterParamOnnx() : onnx(), modelDir() {};
    virtual ~NVEncFilterParamOnnx() {};
    virtual tstring print() const override;
};

// The pre/post a model needs, inferred from its input/output channel count.
// ONNX Runtime runs the network for any architecture with no per-model code; this
// enum only selects how pixels are packed into / unpacked from the tensor.
enum class OnnxIO {
    LumaSR,     // in1  -> out1 : Y plane through the net, integer scale, chroma bilinear-resampled (ArtCNN, vgg7-Y)
    GrayNoise,  // in2  -> out1 : [Y, sigma] through the net, scale=1, chroma copied (DRUNet gray)
    Chroma,     // in3  -> out2 : [Y, Cb, Cr] -> refined [Cb, Cr], scale=1, luma copied (ArtCNN Chroma)
    RGB,        // in3  -> out3 : YUV<->RGB bookend, integer scale (Real-ESRGAN / waifu2x / Real-CUGAN / BSRGAN / ArtCNN RGB)
    RGBNoise,   // in4  -> out3 : [R, G, B, sigma] -> RGB, integer scale (DPSR, DRUNet color)
};

// Standalone ONNX Runtime CNN VPP filter (NVEnc / CUDA port of the VCEEnc DirectML
// filter). One generic ONNX Runtime load-and-run replaces the per-network graphs;
// the network's I/O convention is inferred from its channel count (OnnxIO), so the
// same load-and-run covers every model family with no per-model code.
//
// 全経路の前後処理をCUDAに統一し、共有推論またはテンソル転送で実行する。
class NVEncFilterOnnx : public NVEncFilter {
public:
    NVEncFilterOnnx();
    virtual ~NVEncFilterOnnx();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    // 推論の共有経路とコピー経路でCUDA前後処理を共有する。
    RGY_ERR inferCuda(cudaStream_t stream);
    OnnxColorCoeffs colorCoeffs() const;
    // compute the YUV<->RGB matrix + range coefficients.
    void setupColorCoeffs(int matrixSelIn, int matrixSelOut, bool rangeTV, int pixMax);

    // 時系列窓用のRGBパックと遅延出力。
    RGY_ERR runTemporal(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream);
    RGY_ERR emitTemporalOutput(int64_t outIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream);
    RGY_ERR initMask(const std::shared_ptr<NVEncFilterParamOnnx> &prm, int inW, int inH, RGY_CSP inCsp);
    RGY_ERR runMask(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream);
    RGY_ERR reinitHostPath(tstring &errorMessage);
    RGY_ERR initCudaPath(cudaStream_t stream);
    RGY_ERR prepareLumaOutput(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR runCudaRGB(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR runCudaLuma(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR runCudaGrayNoise(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream);
    RGY_ERR runCudaYuv444(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream);
    RGYFrameInfo tensorFrame(float *ptr, int channels, int width, int height, RGY_CSP csp) const;

    std::unique_ptr<RGYOnnxRTCUDA> m_ov;
    OnnxIO m_io;                          // I/O convention inferred from channel counts
    int   m_inC, m_outC;                        // model input / output channel counts
    int m_modelInW, m_modelInH, m_padL, m_padT;
    int   m_scale;                              // integer upscale factor from the model (out/in)
    float m_maxval;                             // (1<<bitdepth)-1
    bool  m_ycbcr;                              // 3ch model fed as planar YCbCr instead of RGB
    float m_sigmaNorm;                          // noise sigma / 255 for the conditioning channel

    // colour coefficients (computed once at init)
    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;                                   // YUV -> RGB
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV; // RGB -> YUV

    // host-readback path scratch
    std::vector<float>           m_inBuf;       // network input tensor  (inC*inW*inH, CHW)
    std::vector<float>           m_outBuf;      // network output tensor (outC*outW*outH, CHW)

    // CUDA IoBinding経路
    std::unique_ptr<CUMemBuf> m_inputDevice;
    std::unique_ptr<CUMemBuf> m_outputDevice;
    std::unique_ptr<CUMemBuf> m_yuvInputDevice;
    std::unique_ptr<NVEncFilterCspCrop> m_cropToYuv444;
    std::unique_ptr<NVEncFilterCspCrop> m_cropFromYuv444;
    std::unique_ptr<NVEncFilterResize> m_lumaResize;
    tstring m_modelPath;
    RGYOnnxRTProvider m_provider;
    tstring m_precision;
    tstring m_cacheDir;
    int m_deviceID;
    bool m_cudaPathTried;
    bool m_cudaPath;

    int m_temporalT;
    struct RingFrame {
        std::unique_ptr<CUMemBuf> rgb;
        int64_t timestamp;
        int64_t duration;
        RGY_PICSTRUCT picstruct;
        RGY_FRAME_FLAGS flags;
        int inputFrameId;
        std::vector<std::shared_ptr<RGYFrameData>> dataList;
    };
    std::deque<RingFrame> m_ring;
    int64_t m_ringBaseIdx;
    int64_t m_recvCount;
    int64_t m_emitCount;

    int m_maskModelW, m_maskModelH;
    int m_imgPortIdx, m_mskPortIdx;
    float m_maskOutScale;
    std::vector<float> m_maskFrame, m_maskModel;
    std::unique_ptr<CUMemBuf> m_maskFrameDevice, m_maskModelDevice, m_frameRGBDevice, m_compositeDevice, m_maskRangeDevice;
    bool m_maskMode;

    // opt-in end-of-chain resize (out_res=): runs after the network core, fitting
    // the integer-scaled output to the requested final resolution. null when out_res=
    // is not used.
    std::unique_ptr<NVEncFilterResize>  m_postResize;
};

#endif //__NVENC_FILTER_ONNX_H__
