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
#ifndef __RGY_ONNXRUNTIME_H__
#define __RGY_ONNXRUNTIME_H__

#include "rgy_version.h"

#if ENABLE_ONNXRUNTIME

#include "rgy_osdep.h"
#include "rgy_tchar.h"

#define ORT_API_MANUAL_INIT
#ifdef Status
#undef Status
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4244 4267 4127 4100)
#endif
#include "onnxruntime_cxx_api.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

extern const TCHAR *RGY_ONNXRUNTIME_DLL_NAME;

using PFN_OrtGetApiBase = const OrtApiBase*(ORT_API_CALL *)();
using PFN_OrtSessionOptionsAppendExecutionProviderCUDA = OrtStatus*(ORT_API_CALL *)(OrtSessionOptions *options, int device_id);
using PFN_OrtSessionOptionsAppendExecutionProviderTensorRT = OrtStatus*(ORT_API_CALL *)(OrtSessionOptions *options, int device_id);

class RGYOnnxRuntimeLoader {
public:
    RGYOnnxRuntimeLoader();
    ~RGYOnnxRuntimeLoader();

    bool load();
    void close();
    bool loaded() const { return m_loaded; }
    const tstring& errMessage() const { return m_errMessage; }

    auto p_OrtGetApiBase() const { return m_OrtGetApiBase; }
    auto p_OrtSessionOptionsAppendExecutionProviderCUDA() const { return m_OrtSessionOptionsAppendExecutionProviderCUDA; }
    auto p_OrtSessionOptionsAppendExecutionProviderTensorRT() const { return m_OrtSessionOptionsAppendExecutionProviderTensorRT; }

private:
    HMODULE m_hModule;
    bool m_loaded;
    tstring m_errMessage;

    PFN_OrtGetApiBase m_OrtGetApiBase;
    PFN_OrtSessionOptionsAppendExecutionProviderCUDA m_OrtSessionOptionsAppendExecutionProviderCUDA;
    PFN_OrtSessionOptionsAppendExecutionProviderTensorRT m_OrtSessionOptionsAppendExecutionProviderTensorRT;
};

#endif // ENABLE_ONNXRUNTIME

#endif // __RGY_ONNXRUNTIME_H__
