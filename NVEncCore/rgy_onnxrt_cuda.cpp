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
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <unordered_map>
#include <fstream>
#include <filesystem>

#include <cuda.h>
#include <cuda_runtime.h>
#include "rgy_onnxruntime.h"
#include "rgy_util.h"
#include "rgy_filesystem.h"
#include "rgy_rev.h"

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

    // FNV-1a 64bit over the model file bytes. The engine cache must key on the model
    // CONTENT: ORT's own cache filename hash covers only the file NAME and the graph
    // node names (not the weights, not the tensor shapes), so a re-exported model
    // under the same name would silently reuse a stale engine built from old weights.
    uint64_t fnv1a64File(const tstring &path) {
        uint64_t h = 14695981039346656037ULL;
        std::ifstream f(std::filesystem::path(path), std::ios::binary);
        if (!f.good()) return 0;
        std::vector<char> buf(1 << 20);
        for (;;) {
            f.read(buf.data(), buf.size());
            const auto n = f.gcount();
            if (n <= 0) break;
            for (std::streamsize i = 0; i < n; i++) {
                h ^= (uint64_t)(uint8_t)buf[i];
                h *= 1099511628211ULL;
            }
        }
        return h;
    }

    tstring sanitizeForDirName(const tstring &name) {
        tstring out;
        for (const auto c : name) {
            if ((c >= _T('0') && c <= _T('9')) || (c >= _T('a') && c <= _T('z')) || (c >= _T('A') && c <= _T('Z'))) {
                out.push_back(c);
            }
        }
        if (out.length() > 24) out.resize(24);
        return out;
    }

    tstring tensorRTCacheEnvironment(const cudaDeviceProp& prop) {
        tstring fingerprint = char_to_tstring(ENCODER_NAME) + _T("_") + VER_STR_FILEVERSION_TCHAR
            + _T("_rev") + char_to_tstring(ENCODER_REV);
        if (const auto getApiBase = onnxRuntime().p_OrtGetApiBase(); getApiBase != nullptr) {
            if (const auto version = getApiBase()->GetVersionString(); version != nullptr && version[0] != '\0') {
                fingerprint += _T("_ort") + char_to_tstring(version);
            }
        }
        int cudaDriverVersion = 0;
        if (cudaDriverGetVersion(&cudaDriverVersion) == cudaSuccess && cudaDriverVersion > 0) {
            fingerprint += strsprintf(_T("_cuda%d"), cudaDriverVersion);
        }
        fingerprint += strsprintf(_T("_sm%d%d_%s"), prop.major, prop.minor,
            sanitizeForDirName(char_to_tstring(prop.name)).c_str());
        return fingerprint;
    }

    bool clearTensorRTEngineCache(const tstring& dir, tstring& errorMessage) {
        std::error_code ec;
        const auto cacheDir = std::filesystem::path(dir);
        for (const auto& entry : std::filesystem::directory_iterator(cacheDir, ec)) {
            if (ec) break;
            const auto extension = entry.path().extension();
            if (entry.is_regular_file(ec)
                && (extension == std::filesystem::path(_T(".engine")) || extension == std::filesystem::path(_T(".profile")))) {
                if (!std::filesystem::remove(entry.path(), ec) || ec) {
                    break;
                }
            }
        }
        if (ec) {
            errorMessage = strsprintf(_T("TensorRT engine cacheの削除に失敗しました: %s: %s"),
                dir.c_str(), char_to_tstring(ec.message()).c_str());
            return false;
        }
        return true;
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
    std::string inName, outName;     // 互換用の先頭I/O名
    std::vector<std::string> inNames, outNames;
    std::vector<int> inCs, inHs, inWs, outCs, outHs, outWs;
    int inC = 0, inH = 0, inW = 0;
    int outC = 0, outH = 0, outW = 0;
    int deviceID = 0;
    tstring deviceName;
    tstring provider = _T("cuda");   // the EP actually used
    tstring precision = _T("f32");
    tstring lastError;
    tstring engineCacheDir;          // per-key TensorRT engine cache dir ("" = caching off)
    bool engineCacheHadFiles = false; // a cached engine existed there before session create
    bool engineCacheLoadFailure = false;
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
    auto err = initImpl(modelPath, deviceID, provider, height, width, errMessage, userComputeStream, precision, cacheDir);
    // A cached TensorRT engine is deserialized without content validation; a stale or
    // corrupt cache file fails the whole session instead of triggering a rebuild. If a
    // pre-existing cache was in play, clear that key directory and retry once cold.
    if (err != RGY_ERR_NONE && m_impl->engineCacheLoadFailure && !m_impl->engineCacheDir.empty()) {
        tstring clearError;
        if (!clearTensorRTEngineCache(m_impl->engineCacheDir, clearError)) {
            errMessage += _T(" (") + clearError + _T(")");
            return err;
        }
        tstring retryMessage;
        err = initImpl(modelPath, deviceID, provider, height, width, retryMessage, userComputeStream, precision, cacheDir);
        if (err == RGY_ERR_NONE) {
            m_impl->lastError = tstring(_T("cached TensorRT engine was unusable and has been rebuilt: ")) + errMessage;
            errMessage.clear();
        } else {
            errMessage += tstring(_T(" (retry after clearing engine cache: ")) + retryMessage + _T(")");
        }
    }
    return err;
}

RGY_ERR RGYOnnxRTCUDA::initImpl(const tstring &modelPath, const int deviceID, const RGYOnnxRTProvider provider,
                            const int height, const int width, tstring &errMessage,
                            cudaStream_t userComputeStream, const tstring &precision, const tstring &cacheDir) {
    enum class InitStage {
        Other,
        TensorRTSessionCreate,
        TensorRTProbe
    };
    auto initStage = InitStage::Other;
    auto &I = *m_impl;
    CudaContextRestorer contextRestorer;
    loadOrtOnce();
    if (!s_ortReady) {
        errMessage = s_ortError;
        return RGY_ERR_UNSUPPORTED;
    }
    try {
        I.deviceID = deviceID;
        I.provider = _T("cuda");
        I.precision = _T("f32");
        I.engineCacheDir.clear();
        I.engineCacheHadFiles = false;
        I.engineCacheLoadFailure = false;
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
        // 推論時の実形状とメモリパターンの計画が食い違うモデルでは、CUDAの
        // reduction処理が不正な一時バッファを参照するため、メモリパターンを無効化する。
        opts.DisableMemPattern();

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
                    if (!cacheDir.empty()) {
                        // ORT reuses a cached engine whenever its filename hash matches, but that
                        // hash covers neither the model weights nor the input shape: the same model
                        // at another resolution, or a re-exported model under the same name, maps to
                        // the same cache file and the stale engine fails the session (or worse, runs
                        // with the old weights). Key a subdirectory on everything a TensorRT engine
                        // is actually specific to: model content, WxH, precision, GPU (SM + name).
                        cudaDeviceProp prop;
                        memset(&prop, 0, sizeof(prop));
                        cudaGetDeviceProperties(&prop, deviceID);
                        const auto keyDirName = strsprintf(_T("trt_m%016llx_%dx%d_%s"),
                            (unsigned long long)fnv1a64File(modelPath), width, height,
                            useFP16 ? _T("fp16") : _T("fp32"));
                        // NVEnc・ORT・CUDAドライバAPI・GPUの環境が変わった場合は、
                        // TensorRTに古いキャッシュを渡さず別の親ディレクトリを使う。
                        const auto environmentDir = cacheDir + _T("/") + tensorRTCacheEnvironment(prop);
                        const auto engineCacheDir = environmentDir + _T("/") + keyDirName;
                        const auto timingCacheDir = environmentDir + _T("/trt_timing");
                        const bool engineCacheCreated = CreateDirectoryRecursive(engineCacheDir.c_str());
                        const bool timingCacheCreated = CreateDirectoryRecursive(timingCacheDir.c_str());
                        if (rgy_directory_exists(engineCacheDir)) {
                            optionValues["trt_engine_cache_enable"] = "1";
                            optionValues["trt_engine_cache_path"] = tchar_to_string(engineCacheDir, CP_UTF8);
                            I.engineCacheDir = engineCacheDir;
                            std::error_code ec;
                            for (const auto &entry : std::filesystem::directory_iterator(std::filesystem::path(engineCacheDir), ec)) {
                                if (entry.is_regular_file(ec) && entry.path().extension() == std::filesystem::path(_T(".engine"))) {
                                    I.engineCacheHadFiles = true;
                                    break;
                                }
                            }
                        } else if (!engineCacheCreated) {
                            I.lastError = _T("TensorRT engine cacheディレクトリを作成できないため、キャッシュを無効にしました: ") + engineCacheDir;
                        }
                        if (rgy_directory_exists(timingCacheDir)) {
                            // kernel tactic timings; keyed per compute capability inside the cache and
                            // shared across models, so cold engine builds on this GPU get much shorter
                            optionValues["trt_timing_cache_enable"] = "1";
                            optionValues["trt_timing_cache_path"] = tchar_to_string(timingCacheDir, CP_UTF8);
                        } else if (!timingCacheCreated) {
                            const auto timingError = _T("TensorRT timing cacheディレクトリを作成できないため、timing cacheを無効にしました: ") + timingCacheDir;
                            I.lastError += (I.lastError.empty() ? tstring() : _T(" ")) + timingError;
                        }
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
                    I.engineCacheDir.clear();
                    I.engineCacheHadFiles = false;
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

        initStage = (tensorRTAttached && I.engineCacheHadFiles) ? InitStage::TensorRTSessionCreate : InitStage::Other;
        I.session = std::make_unique<Ort::Session>(*I.env, modelPath.c_str(), opts);
        initStage = InitStage::Other;

        if (I.session->GetInputCount() < 1 || I.session->GetOutputCount() < 1) {
            errMessage = _T("model has no input/output tensor.");
            return RGY_ERR_UNSUPPORTED;
        }
        // names (own the strings; the AllocatedStringPtr frees on scope exit)
        I.inNames.clear(); I.inCs.clear(); I.inHs.clear(); I.inWs.clear();
        for (size_t i = 0; i < I.session->GetInputCount(); i++) {
            auto name = I.session->GetInputNameAllocated(i, *I.alloc);
            I.inNames.emplace_back(name.get());
            auto shape = I.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            const int c = (shape.size() >= 2 && shape[1] > 0) ? (int)shape[1] : 1;
            const int h = (shape.size() >= 4 && shape[2] > 0) ? (int)shape[2] : height;
            const int w = (shape.size() >= 4 && shape[3] > 0) ? (int)shape[3] : width;
            I.inCs.push_back(c); I.inHs.push_back(h); I.inWs.push_back(w);
        }
        I.inName = I.inNames.front();
        I.inC = I.inCs.front(); I.inH = I.inHs.front(); I.inW = I.inWs.front();
        I.outNames.clear(); I.outCs.clear(); I.outHs.clear(); I.outWs.clear();
        for (size_t i = 0; i < I.session->GetOutputCount(); i++) {
            auto name = I.session->GetOutputNameAllocated(i, *I.alloc);
            I.outNames.emplace_back(name.get());
        }
        I.outName = I.outNames.front();
        I.deviceName = cudaDeviceName(deviceID);

        // Probe inference with a zero input to discover the output shape and warm
        // the provider (for TensorRT the first run builds the engine).
        Ort::MemoryInfo memCpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inTensors;
        std::vector<std::vector<float>> zero;
        inTensors.reserve(I.inNames.size());
        zero.reserve(I.inNames.size());
        std::vector<std::array<int64_t, 4>> inDims(I.inNames.size());
        for (size_t i = 0; i < I.inNames.size(); i++) {
            inDims[i] = { 1, I.inCs[i], I.inHs[i], I.inWs[i] };
            zero.emplace_back((size_t)I.inCs[i] * I.inHs[i] * I.inWs[i], 0.0f);
            inTensors.emplace_back(Ort::Value::CreateTensor<float>(memCpu, zero.back().data(), zero.back().size(), inDims[i].data(), 4));
        }
        std::vector<const char *> inNames;
        for (const auto &n : I.inNames) inNames.push_back(n.c_str());
        std::vector<const char *> outNames;
        for (const auto &n : I.outNames) outNames.push_back(n.c_str());
        cudaerr = selectCudaDevice(I.deviceID);
        if (cudaerr != cudaSuccess) {
            errMessage = cudaErrorMessage(_T("cudaSetDevice"), I.deviceID, cudaerr);
            return RGY_ERR_CUDA;
        }
        cudaGetLastError();
        initStage = (tensorRTAttached && I.engineCacheHadFiles) ? InitStage::TensorRTProbe : InitStage::Other;
        auto outs = I.session->Run(Ort::RunOptions{ nullptr }, inNames.data(), inTensors.data(), inTensors.size(), outNames.data(), outNames.size());
        initStage = InitStage::Other;
        for (const auto &out : outs) {
            auto oShape = out.GetTensorTypeAndShapeInfo().GetShape();
            if (oShape.size() != 4) {
                errMessage = _T("model output is not a 4D NCHW tensor.");
                return RGY_ERR_UNSUPPORTED;
            }
            I.outCs.push_back((int)oShape[1]); I.outHs.push_back((int)oShape[2]); I.outWs.push_back((int)oShape[3]);
        }
        I.outC = I.outCs.front(); I.outH = I.outHs.front(); I.outW = I.outWs.front();
    } catch (const Ort::Exception &e) {
        I.engineCacheLoadFailure = (initStage != InitStage::Other);
        errMessage = char_to_tstring(e.what());
        return RGY_ERR_UNKNOWN;
    } catch (const std::exception &e) {
        I.engineCacheLoadFailure = (initStage != InitStage::Other);
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

RGY_ERR RGYOnnxRTCUDA::inferMulti(const std::vector<const float *> &inputs, const std::vector<float *> &outputs) {
    if (!m_impl->session || inputs.size() != m_impl->inNames.size() || outputs.size() != m_impl->outNames.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
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
        Ort::MemoryInfo memCpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inTensors, outTensors;
        std::vector<std::array<int64_t, 4>> inDims(I.inNames.size()), outDims(I.outNames.size());
        for (size_t i = 0; i < I.inNames.size(); i++) {
            inDims[i] = { 1, I.inCs[i], I.inHs[i], I.inWs[i] };
            const size_t count = (size_t)I.inCs[i] * I.inHs[i] * I.inWs[i];
            inTensors.emplace_back(Ort::Value::CreateTensor<float>(memCpu, const_cast<float *>(inputs[i]), count, inDims[i].data(), 4));
        }
        for (size_t i = 0; i < I.outNames.size(); i++) {
            outDims[i] = { 1, I.outCs[i], I.outHs[i], I.outWs[i] };
            const size_t count = (size_t)I.outCs[i] * I.outHs[i] * I.outWs[i];
            outTensors.emplace_back(Ort::Value::CreateTensor<float>(memCpu, outputs[i], count, outDims[i].data(), 4));
        }
        std::vector<const char *> inNames, outNames;
        for (const auto &n : I.inNames) inNames.push_back(n.c_str());
        for (const auto &n : I.outNames) outNames.push_back(n.c_str());
        I.session->Run(Ort::RunOptions{ nullptr }, inNames.data(), inTensors.data(), inTensors.size(), outNames.data(), outTensors.data(), outTensors.size());
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
int RGYOnnxRTCUDA::inputCount() const { return (int)m_impl->inNames.size(); }
int RGYOnnxRTCUDA::inputChannels(int index) const { return (index >= 0 && index < (int)m_impl->inCs.size()) ? m_impl->inCs[index] : 0; }
int RGYOnnxRTCUDA::inputHeight(int index) const { return (index >= 0 && index < (int)m_impl->inHs.size()) ? m_impl->inHs[index] : 0; }
int RGYOnnxRTCUDA::inputWidth(int index) const { return (index >= 0 && index < (int)m_impl->inWs.size()) ? m_impl->inWs[index] : 0; }
int RGYOnnxRTCUDA::outputCount() const { return (int)m_impl->outNames.size(); }
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
tstring RGYOnnxRTCUDA::cacheInfo() const {
    if (m_impl->engineCacheDir.empty()) return tstring();
    return strsprintf(_T("TensorRT engine cache %s: %s"),
        m_impl->engineCacheHadFiles ? _T("hit") : _T("cold, engine built"),
        m_impl->engineCacheDir.c_str());
}

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
RGY_ERR RGYOnnxRTCUDA::inferMulti(const std::vector<const float *> &, const std::vector<float *> &) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOnnxRTCUDA::inferDevice(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
bool RGYOnnxRTCUDA::deviceIOAvailable() const { return false; }
int RGYOnnxRTCUDA::inChannels()  const { return 0; }
int RGYOnnxRTCUDA::inputCount() const { return 0; }
int RGYOnnxRTCUDA::inputChannels(int) const { return 0; }
int RGYOnnxRTCUDA::inputHeight(int) const { return 0; }
int RGYOnnxRTCUDA::inputWidth(int) const { return 0; }
int RGYOnnxRTCUDA::outputCount() const { return 0; }
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
tstring RGYOnnxRTCUDA::cacheInfo() const { return tstring(); }

#endif // ENABLE_ONNXRUNTIME
