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
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include "rgy_onnxruntime.h"
#include "rgy_util.h"

// ------- one-time dynamic load of ONNX Runtime + Ort C++ API init -------------

namespace {
    std::once_flag       s_ortInitOnce;
    bool                 s_ortReady = false;
    tstring              s_ortError;

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

    tstring cudaDeviceName(int deviceID) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, deviceID) == cudaSuccess) {
            return char_to_tstring(prop.name);
        }
        return tstring();
    }

    tstring cudaErrorMessage(const TCHAR *func, const int deviceID, const cudaError_t err) {
        return strsprintf(_T("%s(device=%d) failed: %s"),
            func, deviceID, char_to_tstring(cudaGetErrorString(err)).c_str());
    }

    cudaError_t selectCudaDevice(const int deviceID) {
        int currentDevice = -1;
        const auto err = cudaGetDevice(&currentDevice);
        if (err == cudaSuccess && currentDevice == deviceID) {
            return cudaSuccess;
        }
        return cudaSetDevice(deviceID);
    }

    class CudaContextRestorer {
    public:
        CudaContextRestorer() : m_context(nullptr), m_valid(cuCtxGetCurrent(&m_context) == CUDA_SUCCESS) {}
        ~CudaContextRestorer() {
            if (m_valid) {
                cuCtxSetCurrent(m_context);
            }
        }
    private:
        CUcontext m_context;
        bool m_valid;
    };
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
    int deviceID = 0;
    tstring deviceName;
    tstring provider = _T("cuda");   // the EP actually used
    tstring precision = _T("f32");
    tstring lastError;
    bool deviceIO = false;
    std::unique_ptr<Ort::MemoryInfo> deviceMemInfo;
    std::unique_ptr<Ort::Value> deviceInput;
    std::unique_ptr<Ort::Value> deviceOutput;
    std::unique_ptr<Ort::IoBinding> ioBinding;
    std::unique_ptr<Ort::RunOptions> deviceRunOptions;
    const float *boundInput = nullptr;
    float *boundOutput = nullptr;
};

RGYOnnxRTCUDA::RGYOnnxRTCUDA() : m_impl(std::make_unique<Impl>()) {}
RGYOnnxRTCUDA::~RGYOnnxRTCUDA() {}

RGY_ERR RGYOnnxRTCUDA::init(const tstring &modelPath, const int deviceID, const RGYOnnxRTProvider provider,
                            const int height, const int width, tstring &errMessage,
                            cudaStream_t userComputeStream, const tstring &precision, const tstring &cacheDir) {
    CudaContextRestorer contextRestorer;
    loadOrtOnce();
    if (!s_ortReady) {
        errMessage = s_ortError;
        return RGY_ERR_UNSUPPORTED;
    }
    try {
        auto &I = *m_impl;
        I.deviceID = deviceID;
        I.provider = _T("cuda");
        I.precision = _T("f32");
        I.deviceIO = false;
        I.deviceMemInfo.reset();
        I.deviceInput.reset();
        I.deviceOutput.reset();
        I.ioBinding.reset();
        I.deviceRunOptions.reset();
        I.boundInput = nullptr;
        I.boundOutput = nullptr;
        I.lastError.clear();
        auto cudaerr = selectCudaDevice(I.deviceID);
        if (cudaerr != cudaSuccess) {
            errMessage = cudaErrorMessage(_T("cudaSetDevice"), I.deviceID, cudaerr);
            return RGY_ERR_CUDA;
        }
        cudaGetLastError();
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
        bool tensorRTAttached = false;
        if (wantTensorRT) {
            tstring tensorRTV2Error;
            const auto& api = Ort::GetApi();
            if (api.CreateTensorRTProviderOptions != nullptr
                && api.UpdateTensorRTProviderOptions != nullptr
                && api.UpdateTensorRTProviderOptionsWithValue != nullptr
                && api.SessionOptionsAppendExecutionProvider_TensorRT_V2 != nullptr) {
                try {
                    const auto precLower = tolowercase(precision);
                    const bool useFP16 = (precLower == _T("auto") || precLower == _T("fp16") || precLower == _T("f16"));
                    Ort::TensorRTProviderOptions trtOptions;
                    std::unordered_map<std::string, std::string> optionValues = {
                        { "device_id", std::to_string(deviceID) },
                        { "trt_fp16_enable", useFP16 ? "1" : "0" }
                    };
                    if (userComputeStream != nullptr) {
                        optionValues["has_user_compute_stream"] = "1";
                    }
                    std::string cacheDirUtf8;
                    if (!cacheDir.empty()) {
                        cacheDirUtf8 = tchar_to_string(cacheDir, CP_UTF8);
                        optionValues["trt_engine_cache_enable"] = "1";
                        optionValues["trt_engine_cache_path"] = cacheDirUtf8;
                    }
                    trtOptions.Update(optionValues);
                    if (userComputeStream != nullptr) {
                        trtOptions.UpdateWithValue("user_compute_stream", userComputeStream);
                    }
                    opts.AppendExecutionProvider_TensorRT_V2(*trtOptions);
                    tensorRTAttached = true;
                    I.provider = _T("tensorrt");
                    I.precision = useFP16 ? _T("f16") : _T("f32");
                } catch (const Ort::Exception &e) {
                    tensorRTV2Error = tstring(_T("TensorRT V2 provider options failed: ")) + char_to_tstring(e.what());
                }
            } else {
                tensorRTV2Error = _T("TensorRT V2 provider options are unavailable.");
            }
            if (!tensorRTAttached && ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()) {
                OrtStatus *stTrt = ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()(static_cast<OrtSessionOptions*>(opts), deviceID);
                if (stTrt == nullptr) {
                    tensorRTAttached = true;
                    I.provider = _T("tensorrt");
                    I.precision = _T("f32");
                    I.lastError = tensorRTV2Error;
                } else {
                    const auto legacyError = tstring(_T("AppendExecutionProvider_Tensorrt failed: "))
                        + char_to_tstring(Ort::GetApi().GetErrorMessage(stTrt));
                    Ort::GetApi().ReleaseStatus(stTrt);
                    I.lastError = tensorRTV2Error.empty() ? legacyError : tensorRTV2Error + _T(" ") + legacyError;
                }
            }
            if (!tensorRTAttached && !ort.p_OrtSessionOptionsAppendExecutionProviderTensorRT()) {
                I.lastError = tensorRTV2Error.empty()
                    ? _T("TensorRT execution provider is unavailable.")
                    : tensorRTV2Error + _T(" TensorRT legacy provider is unavailable.");
            }
        }
        if (userComputeStream != nullptr) {
            try {
                Ort::CUDAProviderOptions cudaOptions;
                cudaOptions.Update({
                    { "device_id", std::to_string(deviceID) },
                    { "has_user_compute_stream", "1" }
                });
                cudaOptions.UpdateWithValue("user_compute_stream", userComputeStream);
                opts.AppendExecutionProvider_CUDA_V2(*cudaOptions);
                I.deviceIO = true;
            } catch (const Ort::Exception &e) {
                I.lastError = tstring(_T("CUDA V2 provider options failed: ")) + char_to_tstring(e.what());
                I.deviceIO = false;
            }
        }
        if (!I.deviceIO) {
            OrtStatus *stCuda = ort.p_OrtSessionOptionsAppendExecutionProviderCUDA()(static_cast<OrtSessionOptions*>(opts), deviceID);
            if (stCuda != nullptr) {
                errMessage = tstring(_T("AppendExecutionProvider_CUDA failed: "))
                           + char_to_tstring(Ort::GetApi().GetErrorMessage(stCuda));
                Ort::GetApi().ReleaseStatus(stCuda);
                return RGY_ERR_UNSUPPORTED;
            }
        }

        I.session = std::make_unique<Ort::Session>(*I.env, modelPath.c_str(), opts);

        if (I.session->GetInputCount() < 1 || I.session->GetOutputCount() < 1) {
            errMessage = _T("model has no input/output tensor.");
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
        auto inTypeInfo = I.session->GetInputTypeInfo(0);
        auto inInfo  = inTypeInfo.GetTensorTypeAndShapeInfo();
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
        cudaerr = selectCudaDevice(I.deviceID);
        if (cudaerr != cudaSuccess) {
            errMessage = cudaErrorMessage(_T("cudaSetDevice"), I.deviceID, cudaerr);
            return RGY_ERR_CUDA;
        }
        cudaGetLastError();
        auto outs = I.session->Run(Ort::RunOptions{ nullptr }, inNames, &inT, 1, outNames, 1);
        auto oShape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (oShape.size() != 4) {
            errMessage = _T("model output is not a 4D NCHW tensor.");
            return RGY_ERR_UNSUPPORTED;
        }
        I.outC = (int)oShape[1];
        I.outH = (int)oShape[2];
        I.outW = (int)oShape[3];
    } catch (const Ort::Exception &e) {
        errMessage = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    } catch (const std::exception &e) {
        errMessage = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOnnxRTCUDA::infer(const float *in, float *out) {
    if (!m_impl->session) return RGY_ERR_NULL_PTR;
    CudaContextRestorer contextRestorer;
    try {
        auto &I = *m_impl;
        I.lastError.clear();
        auto cudaerr = selectCudaDevice(I.deviceID);
        if (cudaerr != cudaSuccess) {
            I.lastError = cudaErrorMessage(_T("cudaSetDevice"), I.deviceID, cudaerr);
            return RGY_ERR_CUDA;
        }
        cudaGetLastError();
        std::vector<int64_t> inDims  = { 1, I.inC,  I.inH,  I.inW };
        std::vector<int64_t> outDims = { 1, I.outC, I.outH, I.outW };
        const size_t inCount  = (size_t)I.inC  * I.inH  * I.inW;
        const size_t outCount = (size_t)I.outC * I.outH * I.outW;
        Ort::MemoryInfo memCpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inT  = Ort::Value::CreateTensor<float>(memCpu, const_cast<float*>(in), inCount,
                                                          inDims.data(), inDims.size());
        Ort::Value outT = Ort::Value::CreateTensor<float>(memCpu, out, outCount,
                                                          outDims.data(), outDims.size());
        const char *inNames[]  = { I.inName.c_str() };
        const char *outNames[] = { I.outName.c_str() };
        I.session->Run(Ort::RunOptions{ nullptr }, inNames, &inT, 1, outNames, &outT, 1);
    } catch (const Ort::Exception &e) {
        m_impl->lastError = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    } catch (const std::exception &e) {
        m_impl->lastError = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOnnxRTCUDA::inferDevice(const float *inDevice, float *outDevice) {
    if (!m_impl->session || !m_impl->deviceIO) return RGY_ERR_UNSUPPORTED;
    CudaContextRestorer contextRestorer;
    try {
        auto &I = *m_impl;
        I.lastError.clear();
        auto cudaerr = selectCudaDevice(I.deviceID);
        if (cudaerr != cudaSuccess) {
            I.lastError = cudaErrorMessage(_T("cudaSetDevice"), I.deviceID, cudaerr);
            return RGY_ERR_CUDA;
        }
        cudaGetLastError();
        std::vector<int64_t> inDims  = { 1, I.inC,  I.inH,  I.inW };
        std::vector<int64_t> outDims = { 1, I.outC, I.outH, I.outW };
        const size_t inCount  = (size_t)I.inC  * I.inH  * I.inW;
        const size_t outCount = (size_t)I.outC * I.outH * I.outW;
        if (!I.ioBinding || I.boundInput != inDevice || I.boundOutput != outDevice) {
            I.deviceMemInfo = std::make_unique<Ort::MemoryInfo>("Cuda", OrtDeviceAllocator, I.deviceID, OrtMemTypeDefault);
            I.deviceInput = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(*I.deviceMemInfo,
                const_cast<float *>(inDevice), inCount, inDims.data(), inDims.size()));
            I.deviceOutput = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(*I.deviceMemInfo,
                outDevice, outCount, outDims.data(), outDims.size()));
            I.ioBinding = std::make_unique<Ort::IoBinding>(*I.session);
            I.ioBinding->BindInput(I.inName.c_str(), *I.deviceInput);
            I.ioBinding->BindOutput(I.outName.c_str(), *I.deviceOutput);
            I.deviceRunOptions = std::make_unique<Ort::RunOptions>();
            I.boundInput = inDevice;
            I.boundOutput = outDevice;
        }
        I.session->Run(*I.deviceRunOptions, *I.ioBinding);
    } catch (const Ort::Exception &e) {
        m_impl->lastError = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    } catch (const std::exception &e) {
        m_impl->lastError = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

bool RGYOnnxRTCUDA::deviceIOAvailable() const { return m_impl->deviceIO; }

int RGYOnnxRTCUDA::inChannels()  const { return m_impl->inC; }
int RGYOnnxRTCUDA::inHeight()    const { return m_impl->inH; }
int RGYOnnxRTCUDA::inWidth()     const { return m_impl->inW; }
int RGYOnnxRTCUDA::outChannels() const { return m_impl->outC; }
int RGYOnnxRTCUDA::outHeight()   const { return m_impl->outH; }
int RGYOnnxRTCUDA::outWidth()    const { return m_impl->outW; }
size_t RGYOnnxRTCUDA::outElemCount() const {
    return (size_t)m_impl->outC * m_impl->outH * m_impl->outW;
}
tstring RGYOnnxRTCUDA::deviceFullName() const { return m_impl->deviceName; }
tstring RGYOnnxRTCUDA::inferencePrecision() const { return m_impl->precision; }
tstring RGYOnnxRTCUDA::providerName() const { return m_impl->provider; }
tstring RGYOnnxRTCUDA::lastError() const { return m_impl->lastError; }

#else // !ENABLE_ONNXRUNTIME

class RGYOnnxRTCUDA::Impl {};
RGYOnnxRTCUDA::RGYOnnxRTCUDA() : m_impl(nullptr) {}
RGYOnnxRTCUDA::~RGYOnnxRTCUDA() {}
RGY_ERR RGYOnnxRTCUDA::init(const tstring &, const int, const RGYOnnxRTProvider, const int, const int, tstring &errMessage,
                            cudaStream_t, const tstring &, const tstring &) {
    errMessage = _T("this build of NVEnc has no ONNX Runtime CUDA support.");
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOnnxRTCUDA::infer(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOnnxRTCUDA::inferDevice(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
bool RGYOnnxRTCUDA::deviceIOAvailable() const { return false; }
int RGYOnnxRTCUDA::inChannels()  const { return 0; }
int RGYOnnxRTCUDA::inHeight()    const { return 0; }
int RGYOnnxRTCUDA::inWidth()     const { return 0; }
int RGYOnnxRTCUDA::outChannels() const { return 0; }
int RGYOnnxRTCUDA::outHeight()   const { return 0; }
int RGYOnnxRTCUDA::outWidth()    const { return 0; }
size_t RGYOnnxRTCUDA::outElemCount() const { return 0; }
tstring RGYOnnxRTCUDA::deviceFullName() const { return tstring(); }
tstring RGYOnnxRTCUDA::inferencePrecision() const { return tstring(); }
tstring RGYOnnxRTCUDA::providerName() const { return tstring(); }
tstring RGYOnnxRTCUDA::lastError() const { return tstring(); }

#endif // ENABLE_ONNXRUNTIME
