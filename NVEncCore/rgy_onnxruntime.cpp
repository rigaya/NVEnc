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

#include "rgy_util.h"

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

    if ((m_hModule = RGY_LOAD_LIBRARY(RGY_ONNXRUNTIME_DLL_NAME)) == nullptr) {
        m_errMessage = strsprintf("could not load %s (a CUDA/TensorRT-enabled ONNX Runtime). "
                                  "place it and its provider libraries next to the executable or in the library search path.",
                                  tchar_to_string(RGY_ONNXRUNTIME_DLL_NAME).c_str());
        return false;
    }

    auto loadFunc = [this](const char *funcName, void **func) {
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
            m_errMessage = strsprintf("%s is missing %s (not a compatible ONNX Runtime library?).",
                                      tchar_to_string(RGY_ONNXRUNTIME_DLL_NAME).c_str(), funcName);
            close();
            return false;
        }
        return true;
    };

    if (!loadFunc("OrtGetApiBase", (void **)&m_OrtGetApiBase)) {
        return false;
    }

    const OrtApi *api = nullptr;
    for (int v = ORT_API_VERSION; v >= 11; --v) {
        api = m_OrtGetApiBase()->GetApi((uint32_t)v);
        if (api) {
            break;
        }
    }
    if (!api) {
        m_errMessage = strsprintf("%s is too old (no compatible ONNX Runtime API version).",
                                  tchar_to_string(RGY_ONNXRUNTIME_DLL_NAME).c_str());
        close();
        return false;
    }
    Ort::InitApi(api);

    if (!loadFunc("OrtSessionOptionsAppendExecutionProvider_CUDA", (void **)&m_OrtSessionOptionsAppendExecutionProviderCUDA)) {
        return false;
    }

    m_OrtSessionOptionsAppendExecutionProviderTensorRT =
        reinterpret_cast<PFN_OrtSessionOptionsAppendExecutionProviderTensorRT>(
            RGY_GET_PROC_ADDRESS(m_hModule, "OrtSessionOptionsAppendExecutionProvider_Tensorrt"));

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
