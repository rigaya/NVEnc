// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2024 rigaya
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
// -------------------------------------------------------------------------------------------
#pragma once
#ifndef _NVENC_NVSDKNGX_H__
#define _NVENC_NVSDKNGX_H__

#include "rgy_err.h"

#if defined(_WIN32) || defined(_WIN64)
#ifdef NVENC_NVSDKNGX_EXPORTS
#define NVENC_NVSDKNGX_API __declspec(dllexport) 
#else
#define NVENC_NVSDKNGX_API __declspec(dllimport)
#endif
#else
#define NVENC_NVSDKNGX_API
#endif

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR * NVENC_NVSDKNGX_MODULENAME = _T("NVEncNVSDKNGX.dll");
#else
static const TCHAR * NVENC_NVOFFRUC_MODULENAME = _T("NVEncNVSDKNGX.so");
#endif

typedef void * NVEncNVSDKNGXHandle;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} NVEncNVSDKNGXRect;

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXCreate(NVEncNVSDKNGXHandle *ppNVSDKNGX);
NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXInit(NVEncNVSDKNGXHandle pNVSDKNGX, ID3D11Device* pD3DDevice, ID3D11DeviceContext* pD3D11DeviceContext);
NVENC_NVSDKNGX_API void    __stdcall NVEncNVSDKNGXDelete(NVEncNVSDKNGXHandle pNVSDKNGX);
NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXProcFrame(NVEncNVSDKNGXHandle pNVSDKNGX, ID3D11Texture2D* frameDst, const NVEncNVSDKNGXRect *rectDst, ID3D11Texture2D* frameSrc, const NVEncNVSDKNGXRect *rectSrc, const int quality);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif //#ifndef _NVENC_NVSDKNGX_H__