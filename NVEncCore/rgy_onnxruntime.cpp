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

#include "rgy_onnxruntime.h"

#if ENABLE_ONNXRUNTIME

#include "rgy_filesystem.h"
#include "rgy_util.h"

#include <filesystem>

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_ONNXRUNTIME_DLL_NAME = _T("onnxruntime.dll");
#else
const TCHAR *RGY_ONNXRUNTIME_DLL_NAME = _T("libonnxruntime.so");
#endif

RGYOnnxRuntimeLoader::RGYOnnxRuntimeLoader() :
    m_hModule(nullptr),
    m_loaded(false),
    m_errMessage(),
    m_OrtGetApiBase(nullptr),
    m_OrtSessionOptionsAppendExecutionProviderCUDA(nullptr),
    m_OrtSessionOptionsAppendExecutionProviderTensorRT(nullptr) {
}

RGYOnnxRuntimeLoader::~RGYOnnxRuntimeLoader() {
    close();
}

bool RGYOnnxRuntimeLoader::load() {
    if (m_loaded) {
        return true;
    }
    m_errMessage.clear();

#if defined(_WIN32) || defined(_WIN64)
    const auto runtimePath = PathCombineS(std::filesystem::path(getExePath()).remove_filename().wstring(), RGY_ONNXRUNTIME_DLL_NAME);
    m_hModule = RGY_LOAD_LIBRARY(runtimePath.c_str());
#else
    m_hModule = RGY_LOAD_LIBRARY(RGY_ONNXRUNTIME_DLL_NAME);
#endif
    if (m_hModule == nullptr) {
#if defined(_WIN32) || defined(_WIN64)
        const auto errorCode = GetLastError();
        m_errMessage = strsprintf(_T("could not load %s (Win32 error %u). ")
                                  _T("Place a CUDA/TensorRT-enabled ONNX Runtime and its provider DLLs next to the executable or in PATH."),
                                  RGY_ONNXRUNTIME_DLL_NAME, errorCode);
#else
        const auto errorMessage = dlerror();
        m_errMessage = strsprintf(_T("could not load %s: %s. ")
                                  _T("Place a CUDA/TensorRT-enabled ONNX Runtime and its provider libraries in the library search path."),
                                  RGY_ONNXRUNTIME_DLL_NAME, char_to_tstring(errorMessage ? errorMessage : "unknown error").c_str());
#endif
        return false;
    }

    const auto modulePath = getModulePath(m_hModule);
    auto loadFunc = [this, &modulePath](const char *funcName, void **func) {
#if !defined(_WIN32) && !defined(_WIN64)
        dlerror();
#endif
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
#if defined(_WIN32) || defined(_WIN64)
            const auto errorCode = GetLastError();
            m_errMessage = strsprintf(_T("loaded ONNX Runtime \"%s\", but required symbol \"%s\" was not found (Win32 error %u). ")
                                      _T("Use a CUDA/TensorRT-enabled ONNX Runtime build compatible with NVEnc."),
                                      modulePath.c_str(), char_to_tstring(funcName).c_str(), errorCode);
#else
            const auto errorMessage = dlerror();
            m_errMessage = strsprintf(_T("loaded ONNX Runtime \"%s\", but required symbol \"%s\" was not found: %s. ")
                                      _T("Use a CUDA/TensorRT-enabled ONNX Runtime build compatible with NVEnc."),
                                      modulePath.c_str(), char_to_tstring(funcName).c_str(), char_to_tstring(errorMessage ? errorMessage : "unknown error").c_str());
#endif
            close();
            return false;
        }
        return true;
    };

    if (!loadFunc("OrtGetApiBase", (void **)&m_OrtGetApiBase)) {
        return false;
    }
    if (!loadFunc("OrtSessionOptionsAppendExecutionProvider_CUDA", (void **)&m_OrtSessionOptionsAppendExecutionProviderCUDA)) {
        return false;
    }

    m_OrtSessionOptionsAppendExecutionProviderTensorRT =
        reinterpret_cast<PFN_OrtSessionOptionsAppendExecutionProviderTensorRT>(
            RGY_GET_PROC_ADDRESS(m_hModule, "OrtSessionOptionsAppendExecutionProvider_Tensorrt"));

    const OrtApi *api = nullptr;
    for (int v = ORT_API_VERSION; v >= 11; --v) {
        api = m_OrtGetApiBase()->GetApi((uint32_t)v);
        if (api) {
            break;
        }
    }
    if (!api) {
        m_errMessage = strsprintf(_T("loaded ONNX Runtime \"%s\", but it does not provide a compatible C API version ")
                                  _T("(requested %d down to 11)."), modulePath.c_str(), ORT_API_VERSION);
        close();
        return false;
    }
    Ort::InitApi(api);

    m_loaded = true;
    return true;
}

void RGYOnnxRuntimeLoader::close() {
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
    m_loaded = false;
    m_OrtGetApiBase = nullptr;
    m_OrtSessionOptionsAppendExecutionProviderCUDA = nullptr;
    m_OrtSessionOptionsAppendExecutionProviderTensorRT = nullptr;
}

#endif // ENABLE_ONNXRUNTIME
