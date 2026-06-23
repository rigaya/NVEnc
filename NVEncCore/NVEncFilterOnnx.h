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
#include <vector>
#include <memory>

class NVEncFilterResize; // opt-in end-of-chain resize sub-filter (out_res=/resize=)

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
// The host-readback path (copy the input device frame to a host-mapped staging
// frame, build the input tensor, run the network on CUDA/TensorRT, write the result
// back to a host staging frame, copy that to the device output) handles all I/O
// modes. A zero-copy CUDA IoBinding fast path can be added later as an optimisation.
class NVEncFilterOnnx : public NVEncFilter {
public:
    NVEncFilterOnnx();
    virtual ~NVEncFilterOnnx();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    // host-readback path: copy the input to a host staging frame, build the input
    // tensor, run the network, write the result to a host staging frame, copy it to
    // the device output. Works for every OnnxIO mode.
    RGY_ERR runHost(const RGYFrameInfo *in, RGYFrameInfo *out, cudaStream_t stream);

    // host pre/post, dispatched on m_io. fillInputHost packs the host input frame
    // into m_inBuf (inC*inW*inH, CHW); writeOutputHost unpacks m_outBuf
    // (outC*outW*outH, CHW) into the host output frame.
    void fillInputHost(const RGYFrameInfo &hin);
    void writeOutputHost(const RGYFrameInfo &hout, const RGYFrameInfo &hin);
    // compute the YUV<->RGB matrix + range coefficients.
    void setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax);

    std::unique_ptr<RGYOnnxRTCUDA> m_ov;
    OnnxIO m_io;                          // I/O convention inferred from channel counts
    int   m_inC, m_outC;                        // model input / output channel counts
    int   m_scale;                              // integer upscale factor from the model (out/in)
    float m_maxval;                             // (1<<bitdepth)-1
    bool  m_ycbcr;                              // 3ch model fed as planar YCbCr instead of RGB
    float m_sigmaNorm;                          // noise sigma / 255 for the conditioning channel

    // colour coefficients (computed once at init)
    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;                                   // YUV -> RGB
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV; // RGB -> YUV

    // host-readback path scratch
    std::unique_ptr<CUFrameBuf>  m_inStaging;   // host (CPU) copy of the input frame
    std::unique_ptr<CUFrameBuf>  m_outStaging;  // host (CPU) scratch for the output frame
    std::vector<float>           m_inBuf;       // network input tensor  (inC*inW*inH, CHW)
    std::vector<float>           m_outBuf;      // network output tensor (outC*outW*outH, CHW)
    std::vector<float>           m_u444, m_v444;// normalised chroma at output luma res (for 4:2:0 downsample)

    // opt-in end-of-chain resize (out_res=): runs after the network core, fitting
    // the integer-scaled output to the requested final resolution. null when out_res=
    // is not used.
    std::unique_ptr<NVEncFilterResize>  m_postResize;
};

#endif //__NVENC_FILTER_ONNX_H__
