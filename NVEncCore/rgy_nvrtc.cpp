// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#include "rgy_tchar.h"
#include "rgy_osdep.h"
#define NVRTC_EXTERN
#include "rgy_nvrtc.h"

extern const TCHAR *NVRTC_DLL_NAME_TSTR;
extern const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR;

static HMODULE nvrtcHandle = nullptr;

int initNVRTCGlobal() {
    if (nvrtcHandle) {
        return 0;
    }
    if ((nvrtcHandle = RGY_LOAD_LIBRARY(NVRTC_DLL_NAME_TSTR)) == nullptr) {
        return 1;
    }

#define LOAD(name) \
    f_##name = (decltype(f_##name)) RGY_GET_PROC_ADDRESS(nvrtcHandle, #name); \
    if (f_##name == nullptr) { \
        RGY_FREE_LIBRARY(nvrtcHandle); \
        nvrtcHandle = nullptr; \
        return 1; \
    }

    LOAD(nvrtcGetErrorString);
    LOAD(nvrtcVersion);
    LOAD(nvrtcCreateProgram);
    LOAD(nvrtcDestroyProgram);
    LOAD(nvrtcCompileProgram);
    LOAD(nvrtcGetPTXSize);
    LOAD(nvrtcGetPTX);
    LOAD(nvrtcGetProgramLogSize);
    LOAD(nvrtcGetProgramLog);
    LOAD(nvrtcAddNameExpression);
    LOAD(nvrtcGetLoweredName);

    return 0;
}
