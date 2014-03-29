/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


//--------------------------------------------------------------------------------------
// File: dynlink_d3d11.h
//
// Shortcut macros and functions for using DX objects
//
// Copyright (c) Microsoft Corporation. All rights reserved
//--------------------------------------------------------------------------------------

#ifndef _DYNLINK_D3D11_H_
#define _DYNLINK_D3D11_H_

// Standard Windows includes
#include <windows.h>
#include <initguid.h>
#include <assert.h>
#include <wchar.h>
#include <mmsystem.h>
#include <commctrl.h> // for InitCommonControls() 
#include <shellapi.h> // for ExtractIcon()
#include <new.h>      // for placement new
#include <shlobj.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>

// CRT's memory leak detection
#if defined(DEBUG) || defined(_DEBUG)
#include <crtdbg.h>
#endif

// Direct3D9 includes
//#include <d3d9.h>
//#include <d3dx9.h>

// Direct3D10 includes
#include <dxgi.h>
#include <d3d11.h>
#include <d3dx11.h>
// #include <..\Samples\C++\Effects11\Inc\d3dx11effect.h>

// XInput includes
#include <xinput.h>

// HRESULT translation for Direct3D10 and other APIs
#include <dxerr.h>

// strsafe.h deprecates old unsecure string functions.  If you
// really do not want to it to (not recommended), then uncomment the next line
//#define STRSAFE_NO_DEPRECATE

#ifndef STRSAFE_NO_DEPRECATE
#pragma deprecated("strncpy")
#pragma deprecated("wcsncpy")
#pragma deprecated("_tcsncpy")
#pragma deprecated("wcsncat")
#pragma deprecated("strncat")
#pragma deprecated("_tcsncat")
#endif

#pragma warning( disable : 4996 ) // disable deprecated warning 
#include <strsafe.h>
#pragma warning( default : 4996 )

typedef HRESULT(WINAPI *LPCREATEDXGIFACTORY)(REFIID, void **);
typedef HRESULT(WINAPI *LPD3D11CREATEDEVICEANDSWAPCHAIN)(__in_opt IDXGIAdapter *pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software, UINT Flags, __in_ecount_opt(FeatureLevels) CONST D3D_FEATURE_LEVEL *pFeatureLevels, UINT FeatureLevels, UINT SDKVersion, __in_opt CONST DXGI_SWAP_CHAIN_DESC *pSwapChainDesc, __out_opt IDXGISwapChain **ppSwapChain, __out_opt ID3D11Device **ppDevice, __out_opt D3D_FEATURE_LEVEL *pFeatureLevel, __out_opt ID3D11DeviceContext **ppImmediateContext);
typedef HRESULT(WINAPI *LPD3D11CREATEDEVICE)(IDXGIAdapter *, D3D_DRIVER_TYPE, HMODULE, UINT32, D3D_FEATURE_LEVEL *, UINT, UINT32, ID3D11Device **, D3D_FEATURE_LEVEL *, ID3D11DeviceContext **);
typedef void (WINAPI *LPD3DX11COMPILEFROMMEMORY)(LPCSTR pSrcData, SIZE_T SrcDataLen, LPCSTR pFileName, CONST D3D10_SHADER_MACRO *pDefines, LPD3D10INCLUDE pInclude,
                                                 LPCSTR pFunctionName, LPCSTR pProfile, UINT Flags1, UINT Flags2, ID3DX11ThreadPump *pPump, ID3D10Blob **ppShader, ID3D10Blob **ppErrorMsgs, HRESULT *pHResult);

static HMODULE                              s_hModDXGI = NULL;
static LPCREATEDXGIFACTORY                  sFnPtr_CreateDXGIFactory11 = NULL;
static HMODULE                              s_hModD3D11 = NULL;
static HMODULE                              s_hModD3DX11 = NULL;
static LPD3D11CREATEDEVICE                  sFnPtr_D3D11CreateDevice = NULL;
static LPD3D11CREATEDEVICEANDSWAPCHAIN      sFnPtr_D3D11CreateDeviceAndSwapChain = NULL;
static LPD3DX11COMPILEFROMMEMORY            sFnPtr_D3DX11CompileFromMemory = NULL;

// unload the D3D10 DLLs
static bool dynlinkUnloadD3D11API(void)
{
    if (s_hModDXGI)
    {
        FreeLibrary(s_hModDXGI);
        s_hModDXGI = NULL;
    }

    if (s_hModD3D11)
    {
        FreeLibrary(s_hModD3D11);
        s_hModD3D11 = NULL;
    }

    if (s_hModD3DX11)
    {
        FreeLibrary(s_hModD3DX11);
        s_hModD3DX11 = NULL;
    }

    return true;
}

#pragma warning (push)
#pragma warning (disable:4702)
// Dynamically load the D3D11 DLLs loaded and map the function pointers
static bool dynlinkLoadD3D11API(void)
{
    // If both modules are non-NULL, this function has already been called.  Note
    // that this doesn't guarantee that all ProcAddresses were found.
    if (s_hModD3D11 != NULL && s_hModD3DX11 != NULL && s_hModDXGI != NULL)
    {
        return true;
    }

#if 1
    // This may fail if Direct3D 11 isn't installed
    s_hModD3D11 = LoadLibrary("d3d11.dll");

    if (s_hModD3D11 != NULL)
    {
        sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)GetProcAddress(s_hModD3D11, "D3D11CreateDevice");
        sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)GetProcAddress(s_hModD3D11, "D3D11CreateDeviceAndSwapChain");
    }

    // first try to load D3DX11CompileFromMemory from DirectX 2010 June
    s_hModD3DX11 = LoadLibrary("D3DX11d_43.dll");

    if (s_hModD3DX11 != NULL)
    {
        sFnPtr_D3DX11CompileFromMemory = (LPD3DX11COMPILEFROMMEMORY)     GetProcAddress(s_hModD3DX11, "D3DX11CompileFromMemory");
    }
    else    // if absent try to take it from DirectX 2010 Feb
    {
        s_hModD3DX11 = LoadLibrary("D3DX11d_42.dll");

        if (s_hModD3DX11 != NULL)
        {
            sFnPtr_D3DX11CompileFromMemory = (LPD3DX11COMPILEFROMMEMORY)     GetProcAddress(s_hModD3DX11, "D3DX11CompileFromMemory");
        }
    }

    if (!sFnPtr_CreateDXGIFactory11)
    {
        s_hModDXGI = LoadLibrary("dxgi.dll");

        if (s_hModDXGI)
        {
            sFnPtr_CreateDXGIFactory11 = (LPCREATEDXGIFACTORY)GetProcAddress(s_hModDXGI, "CreateDXGIFactory1");
        }

        return (s_hModDXGI != NULL) && (s_hModD3D11 != NULL);
    }

    return (s_hModD3D11 != NULL);
#else
    sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)D3D11CreateDeviceAndSwapChain;
    sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)D3D11CreateDeviceAndSwapChain;
    //sFnPtr_D3DX11CreateEffectFromMemory  = ( LPD3DX11CREATEEFFECTFROMMEMORY )D3DX11CreateEffectFromMemory;
    sFnPtr_D3DX11CompileFromMemory = (LPD3DX11COMPILEFROMMEMORY)D3DX11CompileFromMemory;
    sFnPtr_CreateDXGIFactory11 = (LPCREATEDXGIFACTORY)CreateDXGIFactory;
    return true;
#endif
    return true;
}
#pragma warning (pop)

#endif
