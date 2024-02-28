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

typedef void * NVEncNVOFFRUCHandle;

typedef struct NVEncNVOFFRUCParams_ {
    void *frameIn;
    int64_t timestampIn;
    void *frameOut;
    int64_t timestampOut;
} NVEncNVOFFRUCParams;

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCreate(NVEncNVOFFRUCHandle *ppNVOptFlow);
NVENC_NVOFFRUC_API void    __stdcall NVEncNVOptFlowDelete(NVEncNVOFFRUCHandle pNVOptFlow);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCreateFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow, int width, int height);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCloseFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow);
NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowProc(NVEncNVOFFRUCHandle pNVOptFlow, NVEncNVOFFRUCParams *params);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif //__NVENC_NVOFFURC_H__
