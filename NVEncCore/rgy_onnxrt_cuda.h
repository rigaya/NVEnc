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
#ifndef __RGY_ONNXRT_CUDA_H__
#define __RGY_ONNXRT_CUDA_H__

#include <memory>
#include <cstdint>
#include "rgy_err.h"

// Set to 1 by the build (preprocessor define) when NVEnc is built with the ONNX
// Runtime CUDA / TensorRT backend wired in. When 0 the wrapper still compiles but
// every call returns RGY_ERR_UNSUPPORTED and --vpp-onnx reports that this build
// has no ONNX Runtime support. onnxruntime.dll / libonnxruntime.so (a
// CUDA/TensorRT-enabled build) is loaded at runtime; no import library is linked,
// so the exact runtime library can be dropped next to the executable.
#ifndef ENABLE_ONNXRUNTIME
#define ENABLE_ONNXRUNTIME 0
#endif

// Execution provider selection for --vpp-onnx on NVEnc.
//   Auto     -> CUDA (the default; covers every NVEnc GPU the build's CUDA targets)
//   Cuda     -> ONNX Runtime CUDA execution provider
//   TensorRT -> ONNX Runtime TensorRT execution provider (Turing / sm_75 or newer;
//               the first run per model builds an engine, cached afterwards)
enum class RGYOnnxRTProvider {
    Auto,
    Cuda,
    TensorRT,
};

// Thin wrapper over an ONNX Runtime session that runs on the CUDA or TensorRT
// execution provider. This is the NVEnc counterpart of the VCEEnc RGYOnnxRTDML
// backend: one generic load-and-run covers every model family with no per-network
// code. Unlike DirectML, the NVIDIA execution providers take the CUDA device
// ordinal directly (NVEnc already selected one GPU at startup), so no DXGI/LUID
// match is needed -- init() binds inference to that ordinal so the network runs on
// the same GPU as the encoder.
//
// The whole ONNX Runtime include surface is confined to the .cpp via the pimpl
// below, so no other translation unit pulls in those headers.
class RGYOnnxRTCUDA {
public:
    RGYOnnxRTCUDA();
    ~RGYOnnxRTCUDA();

    // Load an ONNX model and create a session on the requested execution provider,
    // bound to CUDA device ordinal deviceID (pass the ordinal NVEnc selected). The
    // input is treated as [1, channels, height, width] (channels read from the
    // model); a probe inference discovers the output shape (and, for TensorRT,
    // warms / builds the engine). On failure errMessage carries the ONNX Runtime
    // error text. If TensorRT is requested but unavailable, falls back to CUDA.
    RGY_ERR init(const tstring &modelPath, const int deviceID, const RGYOnnxRTProvider provider,
                 const int height, const int width, tstring &errMessage);

    // Synchronous inference. in points to inChannels()*inHeight()*inWidth() floats
    // (CHW); out receives outChannels()*outHeight()*outWidth() floats (CHW).
    // Blocking; in only needs to stay valid for the call.
    RGY_ERR infer(const float *in, float *out);

    int inChannels()  const;
    int inHeight()    const;
    int inWidth()     const;
    int outChannels() const;
    int outHeight()   const;
    int outWidth()    const;
    size_t outElemCount() const; // outChannels()*outHeight()*outWidth()

    tstring deviceFullName() const;     // CUDA device name bound to
    tstring inferencePrecision() const; // "f32"
    tstring providerName() const;       // "cuda" or "tensorrt" (the EP actually used)

    static bool available() { return ENABLE_ONNXRUNTIME != 0; }

private:
    RGYOnnxRTCUDA(const RGYOnnxRTCUDA &) = delete;
    void operator=(const RGYOnnxRTCUDA &) = delete;

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif //__RGY_ONNXRT_CUDA_H__
