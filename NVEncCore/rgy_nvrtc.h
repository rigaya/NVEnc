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

#pragma once
#ifndef __RGY_NVRTC_H__
#define __RGY_NVRTC_H__

#include "rgy_osdep.h"
#include <nvrtc.h>

#ifndef CL_EXTERN
#define CL_EXTERN extern
#endif

#if defined(_WIN32)
#define CL_API_ENTRY
#define CL_API_CALL     __stdcall
#define CL_CALLBACK     __stdcall
#else
#define CL_API_ENTRY
#define CL_API_CALL
#define CL_CALLBACK
#endif

CL_EXTERN const char *(CL_API_CALL* f_nvrtcGetErrorString)(nvrtcResult result);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcVersion)(int *major, int *minor);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcCreateProgram)(nvrtcProgram *prog, const char *src, const char *name, int numHeaders, const char * const *headers, const char * const *includeNames);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcDestroyProgram)(nvrtcProgram *prog);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcCompileProgram)(nvrtcProgram prog, int numOptions, const char * const *options);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcGetPTXSize)(nvrtcProgram prog, size_t *ptxSizeRet);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcGetPTX)(nvrtcProgram prog, char *ptx);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t *logSizeRet);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcGetProgramLog)(nvrtcProgram prog, char *log);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcAddNameExpression)(nvrtcProgram prog, const char * const name_expression);
CL_EXTERN nvrtcResult (CL_API_CALL* f_nvrtcGetLoweredName)(nvrtcProgram prog, const char *const name_expression, const char** lowered_name);

#define nvrtcGetErrorString f_nvrtcGetErrorString
#define nvrtcVersion f_nvrtcVersion
#define nvrtcCreateProgram f_nvrtcCreateProgram
#define nvrtcDestroyProgram f_nvrtcDestroyProgram
#define nvrtcCompileProgram f_nvrtcCompileProgram
#define nvrtcGetPTXSize f_nvrtcGetPTXSize
#define nvrtcGetPTX f_nvrtcGetPTX
#define nvrtcGetProgramLogSize f_nvrtcGetProgramLogSize
#define nvrtcGetProgramLog f_nvrtcGetProgramLog
#define nvrtcAddNameExpression f_nvrtcAddNameExpression
#define nvrtcGetLoweredName f_nvrtcGetLoweredName

int initNVRTCGlobal();

#endif //__RGY_NVRTC_H__
