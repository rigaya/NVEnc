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
#ifndef __NVENC_NVOFFURC_H__
#define __NVENC_NVOFFURC_H__

#ifdef NVENC_NVOFFRUC_EXPORTS
#define NVENC_NVOFFRUC_API __declspec(dllexport) 
#else
#define NVENC_NVOFFRUC_API __declspec(dllimport)
#endif

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#if defined(_WIN32) || defined(_WIN64)
    static const TCHAR * NVOFFRUC_MODULENAME = _T("NvOFFRUC.dll");
    static const TCHAR * NVENC_NVOFFRUC_MODULENAME = _T("NVEncNVOFFRUC.dll");
#else
    static const TCHAR * NVOFFRUC_MODULENAME = _T("libNvOFFRUC.so");
    static const TCHAR * NVENC_NVOFFRUC_MODULENAME = _T("libNVEncNVOFFRUC.so");
#endif

typedef void * NVEncNVOFFRUCHandle;

typedef struct NVEncNVOFFRUCParams_ {
    void *frameIn;
    int64_t timestampIn;
    void *frameOut;
    int64_t timestampOut;
} NVEncNVOFFRUCParams;

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCCreate(NVEncNVOFFRUCHandle *ppNVOptFlow);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCLoad(NVEncNVOFFRUCHandle pNVOptFlow);
NVENC_NVOFFRUC_API void    __stdcall NVEncNVOFFRUCDelete(NVEncNVOFFRUCHandle pNVOptFlow);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCCreateFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow, int width, int height, bool nv12);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCRegisterResource(NVEncNVOFFRUCHandle pNVOptFlow, void *ptr0, void *ptr1, void *ptr2);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCCloseFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOFFRUCProc(NVEncNVOFFRUCHandle pNVOptFlow, NVEncNVOFFRUCParams *params);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif //__NVENC_NVOFFURC_H__
