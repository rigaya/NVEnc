// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <array>
#include "NVEncFilter.h"
#include "NVEncParam.h"
#if ENABLE_NVRTC
#pragma warning (push)
#pragma warning (disable: 4819)
#pragma warning (disable: 4117)
#pragma warning (disable: 4100)
#pragma warning (disable: 4456)
#pragma warning (disable: 4267)
#pragma warning (disable: 4706)
#pragma warning (disable: 4091)
#define JITIFY_PRINT_INSTANTIATION 0
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 0
#define DISABLE_DLFCN 1
#include "jitify.hpp"
#pragma warning (pop)
#endif //#if ENABLE_NVRTC


class NVEncFilterParamCustom : public NVEncFilterParam {
public:
    VppCustom custom;

    NVEncFilterParamCustom() : custom() {

    };
    virtual ~NVEncFilterParamCustom() {};
    virtual tstring print() const override;
};

class NVEncFilterCustom : public NVEncFilter {
public:
    static const std::string KERNEL_NAME;
    NVEncFilterCustom();
    virtual ~NVEncFilterCustom();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR check_param(shared_ptr<NVEncFilterParamCustom> prm);
    virtual RGY_ERR run_per_plane(FrameInfo *ppOutputFrames, const FrameInfo *pInputFrame, RGY_PLANE plane, cudaStream_t stream);
    virtual RGY_ERR run_per_plane(FrameInfo *ppOutputFrames, const FrameInfo *pInputFrame, cudaStream_t stream);
    virtual RGY_ERR run_planes(FrameInfo *ppOutputFrames, const FrameInfo *pInputFrame, cudaStream_t stream);

#if ENABLE_NVRTC
    jitify::JitCache m_kernel_cache;
    unique_ptr<jitify::Program> m_program;
#endif //#if ENABLE_NVRTC
};
