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

#include "rgy_onnxrt_cuda.h"

#if ENABLE_ONNXRUNTIME

#include <cstring>
#include <vector>
#include <string>
#include <mutex>

#include <cuda_runtime.h>
#include "rgy_onnxruntime.h"
#include "rgy_util.h"

// ------- one-time dynamic load of ONNX Runtime + Ort C++ API init -------------

namespace {
    std::once_flag       s_ortInitOnce;
    bool                 s_ortReady = false;
    std::string          s_ortError;

    RGYOnnxRuntimeLoader& onnxRuntime() {
        static RGYOnnxRuntimeLoader loader;
        return loader;
    }

    void loadOrtOnce() {
        std::call_once(s_ortInitOnce, []() {
            if (!onnxRuntime().load()) {
                s_ortError = onnxRuntime().errMessage();
                return;
            }
            s_ortReady = true;
        });
    }

    std::basic_string<ORTCHAR_T> stringToOrtPath(const std::string &s) {
#if defined(_WIN32) || defined(_WIN64)
        if (s.empty()) return std::wstring();
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
        std::wstring w(n, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
        return w;
#else
        return s;
#endif
    }

    std::string cudaDeviceName(int deviceID) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, deviceID) == cudaSuccess) {
            return std::string(prop.name);
        }
        return std::string();
    }
}

// ------------------------------- pimpl ---------------------------------------

class RGYOnnxRTCUDA::Impl {
public:
    Impl() {}
    // env / alloc construct the ONNX Runtime C++ objects, which require the API to
    // be initialised first (loadOrtOnce in init), so they are created lazily in
    // init() rather than in this constructor.
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> alloc;
    std::unique_ptr<Ort::Session> session{ nullptr };
    std::string inName, outName;     // owned copies of the model's first I/O names
    int inC = 0, inH = 0, inW = 0;
    int outC = 0, outH = 0, outW = 0;
    std::string deviceName;
    std::string provider = "cuda";   // the EP actually used
    std::string precision = "f32";
};

RGYOnnxRTCUDA::RGYOnnxRTCUDA() : m_impl(std::make_unique<Impl>()) {}
RGYOnnxRTCUDA::~RGYOnnxRTCUDA() {}

RGY_ERR RGYOnnxRTCUDA::init(const std::string &modelPath, const int deviceID, const RGYOnnxRTProvider provider,
                            const int height, const int width, std::string &errMessage) {
    loadOrtOnce();
    if (!s_ortReady) {
        errMessage = s_ortError;
        return RGY_ERR_UNSUPPORTED;
    }
    try {
        auto &I = *m_impl;
        // create the ORT env / allocator now that the API is initialised
        if (!I.env)   I.env   = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nvenc-onnx");
        if (!I.alloc) I.alloc = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Bind inference to the CUDA device ordinal NVEnc selected. TensorRT, when
        // requested, layers on top of CUDA (ORT requires the CUDA EP as the fallback
        // for any op TensorRT cannot run), so append TensorRT first, then CUDA.
        const bool wantTensorRT = (provider == RGYOnnxRTProvider::TensorRT);
        auto& ort = onnxRuntime();
        if (wantTensorRT && ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()) {
            OrtStatus *stTrt = ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()(static_cast<OrtSessionOptions*>(opts), deviceID);
            if (stTrt != nullptr) {
                errMessage = std::string("AppendExecutionProvider_Tensorrt failed: ")
                           + Ort::GetApi().GetErrorMessage(stTrt);
                Ort::GetApi().ReleaseStatus(stTrt);
                return RGY_ERR_UNSUPPORTED;
            }
            I.provider = "tensorrt";
        } else if (wantTensorRT && !ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()) {
            // requested TensorRT but the runtime library has no TensorRT provider: fall back to CUDA.
            I.provider = "cuda";
        }
        OrtStatus *stCuda = ort.p_OrtSessionOptionsAppendExecutionProviderCUDA()(static_cast<OrtSessionOptions*>(opts), deviceID);
        if (stCuda != nullptr) {
            errMessage = std::string("AppendExecutionProvider_CUDA failed: ")
                       + Ort::GetApi().GetErrorMessage(stCuda);
            Ort::GetApi().ReleaseStatus(stCuda);
            return RGY_ERR_UNSUPPORTED;
        }

        const auto ortPath = stringToOrtPath(modelPath);
        I.session = std::make_unique<Ort::Session>(*I.env, ortPath.c_str(), opts);

        if (I.session->GetInputCount() < 1 || I.session->GetOutputCount() < 1) {
            errMessage = "model has no input/output tensor.";
            return RGY_ERR_UNSUPPORTED;
        }
        // names (own the strings; the AllocatedStringPtr frees on scope exit)
        {
            auto inN  = I.session->GetInputNameAllocated(0, *I.alloc);
            auto outN = I.session->GetOutputNameAllocated(0, *I.alloc);
            I.inName  = inN.get();
            I.outName = outN.get();
        }
        // input channel count from the model (dim 1); N/H/W are pinned by us
        auto inInfo  = I.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto inShape = inInfo.GetShape(); // may contain -1 for dynamic dims
        I.inC = (inShape.size() >= 2 && inShape[1] > 0) ? (int)inShape[1] : 1;
        I.inH = height;
        I.inW = width;
        I.deviceName = cudaDeviceName(deviceID);

        // Probe inference with a zero input to discover the output shape and warm
        // the provider (for TensorRT the first run builds the engine).
        std::vector<int64_t> inDims = { 1, I.inC, I.inH, I.inW };
        std::vector<float> zero((size_t)I.inC * I.inH * I.inW, 0.0f);
        Ort::MemoryInfo memCpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inT = Ort::Value::CreateTensor<float>(memCpu, zero.data(), zero.size(),
                                                         inDims.data(), inDims.size());
        const char *inNames[]  = { I.inName.c_str() };
        const char *outNames[] = { I.outName.c_str() };
        auto outs = I.session->Run(Ort::RunOptions{ nullptr }, inNames, &inT, 1, outNames, 1);
        auto oShape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (oShape.size() != 4) {
            errMessage = "model output is not a 4D NCHW tensor.";
            return RGY_ERR_UNSUPPORTED;
        }
        I.outC = (int)oShape[1];
        I.outH = (int)oShape[2];
        I.outW = (int)oShape[3];
    } catch (const Ort::Exception &e) {
        errMessage = e.what();
        return RGY_ERR_UNKNOWN;
    } catch (const std::exception &e) {
        errMessage = e.what();
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOnnxRTCUDA::infer(const float *in, float *out) {
    if (!m_impl->session) return RGY_ERR_NULL_PTR;
    try {
        auto &I = *m_impl;
        std::vector<int64_t> inDims  = { 1, I.inC,  I.inH,  I.inW };
        std::vector<int64_t> outDims = { 1, I.outC, I.outH, I.outW };
        const size_t inCount  = (size_t)I.inC  * I.inH  * I.inW;
        const size_t outCount = (size_t)I.outC * I.outH * I.outW;
        Ort::MemoryInfo memCpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inT  = Ort::Value::CreateTensor<float>(memCpu, const_cast<float*>(in), inCount,
                                                          inDims.data(), inDims.size());
        // Pre-bind the output to our host buffer so ONNX Runtime copies the result
        // into CPU memory. With the CUDA/TensorRT EP an auto-allocated output lives in
        // device memory, so reading it on the host would be invalid; binding a CPU
        // output tensor makes Run() do the device->host copy for us.
        Ort::Value outT = Ort::Value::CreateTensor<float>(memCpu, out, outCount,
                                                          outDims.data(), outDims.size());
        const char *inNames[]  = { I.inName.c_str() };
        const char *outNames[] = { I.outName.c_str() };
        I.session->Run(Ort::RunOptions{ nullptr }, inNames, &inT, 1, outNames, &outT, 1);
    } catch (const Ort::Exception &e) {
        (void)e;
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

int RGYOnnxRTCUDA::inChannels()  const { return m_impl->inC; }
int RGYOnnxRTCUDA::inHeight()    const { return m_impl->inH; }
int RGYOnnxRTCUDA::inWidth()     const { return m_impl->inW; }
int RGYOnnxRTCUDA::outChannels() const { return m_impl->outC; }
int RGYOnnxRTCUDA::outHeight()   const { return m_impl->outH; }
int RGYOnnxRTCUDA::outWidth()    const { return m_impl->outW; }
size_t RGYOnnxRTCUDA::outElemCount() const {
    return (size_t)m_impl->outC * m_impl->outH * m_impl->outW;
}
std::string RGYOnnxRTCUDA::deviceFullName() const { return m_impl->deviceName; }
std::string RGYOnnxRTCUDA::inferencePrecision() const { return m_impl->precision; }
std::string RGYOnnxRTCUDA::providerName() const { return m_impl->provider; }

#else // !ENABLE_ONNXRUNTIME

class RGYOnnxRTCUDA::Impl {};
RGYOnnxRTCUDA::RGYOnnxRTCUDA() : m_impl(nullptr) {}
RGYOnnxRTCUDA::~RGYOnnxRTCUDA() {}
RGY_ERR RGYOnnxRTCUDA::init(const std::string &, const int, const RGYOnnxRTProvider, const int, const int, std::string &errMessage) {
    errMessage = "this build of NVEnc has no ONNX Runtime CUDA support.";
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOnnxRTCUDA::infer(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
int RGYOnnxRTCUDA::inChannels()  const { return 0; }
int RGYOnnxRTCUDA::inHeight()    const { return 0; }
int RGYOnnxRTCUDA::inWidth()     const { return 0; }
int RGYOnnxRTCUDA::outChannels() const { return 0; }
int RGYOnnxRTCUDA::outHeight()   const { return 0; }
int RGYOnnxRTCUDA::outWidth()    const { return 0; }
size_t RGYOnnxRTCUDA::outElemCount() const { return 0; }
std::string RGYOnnxRTCUDA::deviceFullName() const { return std::string(); }
std::string RGYOnnxRTCUDA::inferencePrecision() const { return std::string(); }
std::string RGYOnnxRTCUDA::providerName() const { return std::string(); }

#endif // ENABLE_ONNXRUNTIME
