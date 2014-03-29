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
 * “MATERIALS”) ARE BEING PROVIDED “AS IS.” WITHOUT EXPRESS OR IMPLIED
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

/**
* \file nvEncoder.cpp
* \brief nvEncoder is the Class interface for the Hardware Encoder (NV Encode API)
* \date 2011-2013
*  This file contains the implementation of the CNvEncoder class
*/


#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#define NV_WINDOWS
#endif
#include <include/videoFormats.h>
#include "CNVEncoderH264.h"
#include "CNVEncoder.h"
#include "xcodeutil.h"
#include "nvEncodeAPI.h"

#if defined (NV_WINDOWS)
#if !defined (_NO_D3D)
#include <d3dx9.h>
#include <include/dynlink_d3d10.h>
#include <include/dynlink_d3d11.h>
#endif

static char __NVEncodeLibName32[] = "nvEncodeAPI.dll";
static char __NVEncodeLibName64[] = "nvEncodeAPI64.dll";
#elif defined __linux
#include <dlfcn.h>
#include <stddef.h>
static char __NVEncodeLibName[] = "libnvidia-encode.so";
#endif

#include <cuda.h>
#include <include/helper_cuda_drvapi.h>
#include <include/helper_nvenc.h>


#pragma warning (disable:4189)
#pragma warning (disable:4311)
#pragma warning (disable:4312)
#pragma warning (disable:4701)
#pragma warning (disable:4702)
#pragma warning (disable:4703)

/******************************FAKE KEY****************************************/
static const GUID NV_CLIENT_KEY_TEST = { 0x0, 0x0, 0x0, { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 } };

#define MAKE_FOURCC( ch0, ch1, ch2, ch3 )                               \
    ( (unsigned int)(unsigned char)(ch0) | ( (unsigned int)(unsigned char)(ch1) << 8 ) |    \
      ( (unsigned int)(unsigned char)(ch2) << 16 ) | ( (unsigned int)(unsigned char)(ch3) << 24 ) )

inline BOOL Is64Bit()
{
    return (sizeof(void *)!=sizeof(DWORD));
}

void queryAllEncoderCaps(CNvEncoder *pEncoder)
{
    int result=0;

    if (pEncoder)
    {
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_NUM_MAX_BFRAMES,             result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_FIELD_ENCODING, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_MONOCHROME, result)
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_FMO, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_QPELMV, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_BDIRECT_MODE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_CABAC, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_STEREO_MVC, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_LEVEL_MAX, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_LEVEL_MIN, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SEPARATE_COLOUR_PLANE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_WIDTH_MAX, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_HEIGHT_MAX, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_INTRA_REFRESH, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_PREPROC_SUPPORT, result);
        QUERY_PRINT_CAPS(pEncoder, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT, result);
    }
}


CNvEncoder::CNvEncoder() :
    m_hEncoder(NULL), m_dwEncodeGUIDCount(0), m_dwInputFmtCount(0), m_pAvailableSurfaceFmts(NULL),
    m_bEncoderInitialized(0), m_dwMaxSurfCount(0), m_dwCurrentSurfIdx(0), m_dwFrameWidth(0), m_dwFrameHeight(0),
    m_bEncodeAPIFound(false), m_pEncodeAPI(NULL), m_cuContext(NULL), m_pEncoderThread(NULL), m_bAsyncModeEncoding(true),
    m_fOutput(NULL), m_fInput(NULL), m_dwCodecProfileGUIDCount(0), m_uRefCount(0)
#if defined (NV_WINDOWS) && !defined (_NO_D3D)
    , m_pD3D(NULL), m_pD3D9Device(NULL), m_pD3D10Device(NULL),  m_pD3D11Device(NULL)
#endif
{
    m_dwInputFormat          = NV_ENC_BUFFER_FORMAT_NV12_PL;
    memset(&m_stInitEncParams,   0, sizeof(m_stInitEncParams));
    memset(&m_stEncoderInput,    0, sizeof(m_stEncoderInput));
    memset(&m_stInputSurface,    0, sizeof(m_stInputSurface));
    memset(&m_stBitstreamBuffer, 0, sizeof(m_stBitstreamBuffer));
    memset(&m_spspps, 0, sizeof(m_spspps));
    SET_VER(m_stInitEncParams, NV_ENC_INITIALIZE_PARAMS);
    memset(&m_stEncodeConfig, 0, sizeof(m_stEncodeConfig));
    SET_VER(m_stEncodeConfig, NV_ENC_CONFIG);

    memset(&m_stPresetConfig, 0, sizeof(NV_ENC_PRESET_CONFIG));
    SET_VER(m_stPresetConfig, NV_ENC_PRESET_CONFIG);
    SET_VER(m_stPresetConfig.presetCfg, NV_ENC_CONFIG);

	m_log_level = NV_LOG_INFO;

    m_dwCodecProfileGUIDCount = 0;
    memset(&m_stCodecProfileGUID, 0, sizeof(GUID)) ;
    memset(&m_stPresetGUID, 0, sizeof(GUID));
    m_dwReConfigIdx = 0;
    m_dwNumReConfig = 0;
    m_pYUV[0] = m_pYUV[1] = m_pYUV[2] = NULL;
    struct configs_s config[]=    {
    { "profile",                    0, offsetof(EncodeConfig, profile) },
    { "level",                      0, offsetof(EncodeConfig, level) },
    { "gopLength",                  0, offsetof(EncodeConfig, gopLength) },
    { "numBFrames",                 0, offsetof(EncodeConfig, numBFrames) },
    { "width",                      0, offsetof(EncodeConfig, width) },
    { "height",                     0, offsetof(EncodeConfig, height) },
    { "maxwidth",                   0, offsetof(EncodeConfig, maxWidth) },
    { "maxheight",                  0, offsetof(EncodeConfig, maxHeight) },
    { "fieldMode",                  0, offsetof(EncodeConfig, fieldEncoding) },
    { "bottomfieldFirst",           0, offsetof(EncodeConfig, bottomFieldFrist) },
    { "numSlices",                  0, offsetof(EncodeConfig, numSlices) },
    { "frameRateNum",               0, offsetof(EncodeConfig, frameRateNum) },
    { "frameRateDen",               0, offsetof(EncodeConfig, frameRateDen) },
    { "maxbitrate",                 0, offsetof(EncodeConfig, peakBitRate) },
    { "vbvBufferSize",              0, offsetof(EncodeConfig, vbvBufferSize) },
    { "vbvInitialDelay",            0, offsetof(EncodeConfig, vbvInitialDelay) },
    { "enableInitialRCQP",          0, offsetof(EncodeConfig, enableInitialRCQP) },
    { "rcMode",                     0, offsetof(EncodeConfig, rateControl) },
    { "bitrate",                    0, offsetof(EncodeConfig, avgBitRate) },
    { "initialQPI",                 0, offsetof(EncodeConfig, initialRCQP)+2*sizeof(int) },
    { "initialQPP",                 0, offsetof(EncodeConfig, initialRCQP)+0*sizeof(int) },
    { "initialQPB",                 0, offsetof(EncodeConfig, initialRCQP)+1*sizeof(int) },
    { "preset",                     0, offsetof(EncodeConfig, preset) },
    { "interfaceType",              0, offsetof(EncodeConfig, interfaceType) },
    { "syncMode",                   0, offsetof(EncodeConfig, syncMode) },
    { "enablePtd",                  0, offsetof(EncodeConfig, enablePTD) },
    { "inFile",                     1, offsetof(EncodeConfig, InputClip) },

    { 0, 0, 0 }
};
    memcpy(m_configs, config, sizeof( config));
    NVENCSTATUS nvStatus;
    MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI

#if defined (NV_WINDOWS)

    if (Is64Bit())
    {
        m_hinstLib = LoadLibrary(TEXT(__NVEncodeLibName64));
    }
    else
    {
        m_hinstLib = LoadLibrary(TEXT(__NVEncodeLibName32));
    }

#else
    m_hinstLib = dlopen(__NVEncodeLibName, RTLD_LAZY);
#endif

    if (m_hinstLib != NULL)
    {
		// Initialize NVENC API
#if defined (NV_WINDOWS)
        nvEncodeAPICreateInstance = (MYPROC) GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance");
#else
        nvEncodeAPICreateInstance = (MYPROC) dlsym(m_hinstLib, "NvEncodeAPICreateInstance");
#endif

        if (NULL != nvEncodeAPICreateInstance)
        {
            m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST;

            if (m_pEncodeAPI)
            {
                memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
                m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;
                nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI);
                m_bEncodeAPIFound = true;
            }
            else
            {
                m_bEncodeAPIFound = false;
            }
		}
        else
        {
            PRINTERR(("CNvEncoder::CNvEncoder() failed to find NvEncodeAPICreateInstance"));
        }
	}
    else
    {
        m_bEncodeAPIFound = false;

#if defined (NV_WINDOWS)
        if (Is64Bit())
        {
            PRINTERR(("CNvEncoder::CNvEncoder() failed to load %s!", __NVEncodeLibName64));
        }
        else
        {
            PRINTERR(("CNvEncoder::CNvEncoder() failed to load %s!", __NVEncodeLibName32));
        }
        throw((const char *)("CNvEncoder::CNvEncoder() was unable to load nvEncoder Library"));

#else
        PRINTERR(("CNvEncoder::CNvEncoder() failed to load %s!", __NVEncodeLibName));
#endif
    }

    memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
}


CNvEncoder::~CNvEncoder()
{
    // clean up encode API resources here
    if (m_pEncodeAPI)
        delete m_pEncodeAPI;

    if (m_hinstLib)
    {
#if defined (NV_WINDOWS)
        FreeLibrary(m_hinstLib);
#else
        dlclose(m_hinstLib);
#endif
    }
}

unsigned int CNvEncoder::GetCodecType(GUID encodeGUID)
{
    NvEncodeCompressionStd eEncodeCompressionStd = NV_ENC_Unknown;

    if (compareGUIDs(encodeGUID, NV_ENC_CODEC_H264_GUID))
    {
        eEncodeCompressionStd = NV_ENC_H264;
    }
    else
    {
        nvPrintf(stderr, NV_LOG_ERROR, " unsupported codec \n");
    }

    return eEncodeCompressionStd;
}

unsigned int CNvEncoder::GetCodecProfile(GUID encodeProfileGUID)
{
    if (compareGUIDs(encodeProfileGUID, NV_ENC_H264_PROFILE_BASELINE_GUID))
    {
        return 66;
    }
    else if (compareGUIDs(encodeProfileGUID, NV_ENC_H264_PROFILE_MAIN_GUID))
    {
        return 77;
    }
    else if (compareGUIDs(encodeProfileGUID, NV_ENC_H264_PROFILE_HIGH_GUID))
    {
        return 100;
    }
    else if (compareGUIDs(encodeProfileGUID, NV_ENC_H264_PROFILE_STEREO_GUID))
    {
        return 128;
    }
    else
    {
        // unknown profile
        nvPrintf(stderr, NV_LOG_ERROR, "CNvEncoder::GetCodecProfile is an unspecified GUID\n");
        return 0;
    }

    return 0;
}

GUID CNvEncoder::GetCodecProfileGuid(unsigned int profile)
{
    if (profile == 66)
    {
        return NV_ENC_H264_PROFILE_BASELINE_GUID;
    }
    else if (profile == 77)
    {
        return NV_ENC_H264_PROFILE_MAIN_GUID;
    }
    else if (profile == 100)
    {
        return NV_ENC_H264_PROFILE_HIGH_GUID;
    }
    else if (profile == 128)
    {
        return NV_ENC_H264_PROFILE_STEREO_GUID;
    }
    else
    {
        // unknown profile
        nvPrintf(stderr, NV_LOG_ERROR, "CNvEncoder::GetCodecProfile is an unspecified\n");
        return NV_ENC_H264_PROFILE_INVALID_GUID;
    }
    return NV_ENC_H264_PROFILE_INVALID_GUID;

}

#if defined (NV_WINDOWS) && !defined (_NO_D3D)
HRESULT CNvEncoder::InitD3D9(unsigned int deviceID)
{
    HRESULT hr = S_OK;
    D3DPRESENT_PARAMETERS d3dpp;

    unsigned int iAdapter    = NULL; // Our adapter

    // Create a Context for interfacing to DirectX9
    m_pD3D = Direct3DCreate9(D3D_SDK_VERSION);

    if (m_pD3D == NULL)
    {
        assert(m_pD3D);
        return E_FAIL;
    }

    D3DADAPTER_IDENTIFIER9 adapterId;

    NvPrintf("\n* Detected %d available D3D9 Devices *\n", m_pD3D->GetAdapterCount());

    for (iAdapter = deviceID; iAdapter < m_pD3D->GetAdapterCount(); iAdapter++)
    {
        HRESULT hr = m_pD3D->GetAdapterIdentifier(iAdapter, 0, &adapterId);

        if (FAILED(hr)) continue;

        NvPrintf("> Direct3D9 Display Device #%d: \"%s\"",
               iAdapter, adapterId.Description);

        if (iAdapter == deviceID)
        {
            NvPrintf(" (selected)\n");
        }
        else
        {
            NvPrintf("\n");
        }
    }

    if (deviceID >= m_pD3D->GetAdapterCount())
    {
        nvPrintf(stderr, NV_LOG_ERROR, "CNvEncoder::InitD3D() - deviceID=%d is outside range [%d,%d]\n", deviceID, 0, m_pD3D->GetAdapterCount());
        return E_FAIL;
    }

    // Create the Direct3D9 device and the swap chain. In this example, the swap
    // chain is the same size as the current display mode. The format is RGB-32.
    ZeroMemory(&d3dpp, sizeof(d3dpp));
    d3dpp.Windowed = true;
    d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
    d3dpp.BackBufferWidth  = 640;
    d3dpp.BackBufferHeight = 480;
    d3dpp.BackBufferCount  = 1;
    d3dpp.SwapEffect = D3DSWAPEFFECT_COPY;
    d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    d3dpp.Flags = D3DPRESENTFLAG_VIDEO;//D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
    DWORD dwBehaviorFlags = D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING;

    hr = m_pD3D->CreateDevice(deviceID,
                              D3DDEVTYPE_HAL,
                              NULL,
                              dwBehaviorFlags,
                              &d3dpp,
                              &m_pD3D9Device);

    return hr;
}

HRESULT CNvEncoder::InitD3D10(unsigned int deviceID)
{
    // TODO
    HRESULT hr = S_OK;

    bool bCheckD3D10 = dynlinkLoadD3D10API();

    // If D3D10 is not present, print an error message and then quit    if (!bCheckD3D10) {
    if (!bCheckD3D10)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "> nvEncoder did not detect a D3D10 device, exiting...\n");
        dynlinkUnloadD3D10API();
        return E_FAIL;
    }

    IDXGIFactory * pFactory = NULL;
    IDXGIAdapter * pAdapter; 
    if(sFnPtr_CreateDXGIFactory10(__uuidof(IDXGIFactory),(void**)&pFactory) !=S_OK)
    {
        return E_FAIL;
    }

    if(pFactory->EnumAdapters(deviceID, &pAdapter) != DXGI_ERROR_NOT_FOUND)
    {
           hr = sFnPtr_D3D10CreateDevice(pAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0,D3D10_SDK_VERSION, &m_pD3D10Device);
           if(hr !=S_OK)
               nvPrintf(stderr, NV_LOG_ERROR, "> Problem while creating %d D3d10 device \n",deviceID);
    }
    else
    {
        nvPrintf(stderr, NV_LOG_ERROR, "> nvEncoder did not find %d D3D10 device \n",deviceID);
        return E_FAIL;
    }

    return hr;
}


HRESULT CNvEncoder::InitD3D11(unsigned int deviceID)
{
    // TODO
    HRESULT hr = S_OK;

    bool bCheckD3D11 = dynlinkLoadD3D11API();

    // If D3D10 is not present, print an error message and then quit    if (!bCheckD3D10) {
    if (!bCheckD3D11)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "> nvEncoder did not detect a D3D11 device, exiting...\n");
        dynlinkUnloadD3D11API();
        return E_FAIL;
    }
    IDXGIFactory1 * pFactory = NULL;
    IDXGIAdapter * pAdapter=NULL; 
    
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    if(sFnPtr_CreateDXGIFactory11(__uuidof(IDXGIFactory1),(void**)&pFactory) !=S_OK)
    {
        return E_FAIL;
    }

    if(pFactory->EnumAdapters(deviceID, &pAdapter) != DXGI_ERROR_NOT_FOUND)
    {
           hr = sFnPtr_D3D11CreateDevice(pAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, createDeviceFlags, NULL,
                                  0, D3D11_SDK_VERSION, &m_pD3D11Device, NULL, NULL);
           if(hr !=S_OK)
               nvPrintf(stderr, NV_LOG_ERROR, "> Problem while creating %d D3d11 device \n",deviceID);
    }
    else
    {
        nvPrintf(stderr, NV_LOG_ERROR, "> nvEncoder did not find %d D3D11 device \n",deviceID);
        return E_FAIL;
    }
    return hr;
}
#endif

HRESULT CNvEncoder::InitCuda(unsigned int deviceID)
{
    CUresult        cuResult            = CUDA_SUCCESS;
    CUdevice        cuDevice            = 0;
    CUcontext       cuContextCurr;
    char gpu_name[100];
    int  deviceCount = 0;
    int  SMminor = 0, SMmajor = 0;

    NvPrintf("\n");

    // CUDA interfaces
    cuResult = cuInit(0);

    if (cuResult != CUDA_SUCCESS)
    {
        nvPrintf(stderr, NV_LOG_ERROR, ">> InitCUDA() - cuInit() failed error:0x%x\n", cuResult);
        return E_FAIL;
    }

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        nvPrintf(stderr, NV_LOG_ERROR, ">> InitCuda() - reports no devices available that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        NvPrintf(">> InitCUDA() has detected %d CUDA capable GPU device(s)<<\n", deviceCount);

        for (int currentDevice=0; currentDevice < deviceCount; currentDevice++)
        {
            checkCudaErrors(cuDeviceGet(&cuDevice, currentDevice));
            checkCudaErrors(cuDeviceGetName(gpu_name, 100, cuDevice));
            checkCudaErrors(cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice));
            NvPrintf("  [ GPU #%d - < %s > has Compute SM %d.%d, %s NVENC ]\n",
                   currentDevice, gpu_name, SMmajor, SMminor,
                   (((SMmajor << 4) + SMminor) >= 0x30) ? "Available" : "Not Available");
        }
    }

    // If dev is negative value, we clamp to 0
    if (deviceID < 0)
        deviceID = 0;

    if (deviceID > (unsigned int)deviceCount-1)
    {
        nvPrintf(stderr, NV_LOG_ERROR, ">> InitCUDA() - nvEncoder (-device=%d) is not a valid GPU device. <<\n\n", deviceID);
        exit(EXIT_FAILURE);
    }

    // Now we get the actual device
    checkCudaErrors(cuDeviceGet(&cuDevice, deviceID));
    checkCudaErrors(cuDeviceGetName(gpu_name, 100, cuDevice));
    checkCudaErrors(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
    NvPrintf("\n>> Select GPU #%d - < %s > supports SM %d.%d and NVENC\n", deviceID, gpu_name, SMmajor, SMminor);

    if (((SMmajor << 4) + SMminor) < 0x30)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "  [ GPU %d does not have NVENC capabilities] exiting\n", deviceID);
        exit(EXIT_FAILURE);
    }

    // Create the CUDA Context and Pop the current one
    checkCudaErrors(cuCtxCreate(&m_cuContext, 0, cuDevice));
    checkCudaErrors(cuCtxPopCurrent(&cuContextCurr));

    return S_OK;
}


HRESULT CNvEncoder::AllocateIOBuffers(unsigned int dwInputWidth, unsigned int dwInputHeight, unsigned int maxFrmCnt)
{
    m_dwMaxSurfCount = maxFrmCnt;
    NVENCSTATUS status = NV_ENC_SUCCESS;

    NvPrintf(" > CNvEncoder::AllocateIOBuffers() = Size (%dx%d @ %d frames)\n", dwInputWidth, dwInputHeight, maxFrmCnt);

    for (unsigned int i = 0; i < m_dwMaxSurfCount; i++)
    {
        m_stInputSurface[i].dwWidth  = dwInputWidth;
        m_stInputSurface[i].dwHeight = dwInputHeight;

        if (m_stEncoderInput[0].useMappedResources)
        {
            if (m_stEncoderInput[0].interfaceType == NV_ENC_CUDA)
            {
                if (i==0)
                {
                    NvPrintf(" > CUDA+NVENC InterOp using %d buffers.\n", m_dwMaxSurfCount);
                }

                // Illustrate how to use a Cuda buffer not allocated using NvEncCreateInputBuffer as input to the encoder.
                cuCtxPushCurrent(m_cuContext);
                CUcontext   cuContextCurr;
                CUdeviceptr devPtrDevice;
                CUresult    result = CUDA_SUCCESS;

                // Allocate Cuda buffer. We will use this to hold the input YUV data.
                result = cuMemAllocPitch(&devPtrDevice, (size_t *)&m_stInputSurface[i].dwCuPitch, dwInputWidth, dwInputHeight*3/2, 16);
                m_stInputSurface[i].pExtAlloc      = (void *)devPtrDevice;

                // Allocate Cuda buffer in host memory. We will use this to load data onto the Cuda buffer we want to use as input.
                result = cuMemAllocHost((void **)&m_stInputSurface[i].pExtAllocHost, m_stInputSurface[i].dwCuPitch*dwInputHeight*3/2);

                m_stInputSurface[i].type           = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
                m_stInputSurface[i].bufferFmt      = NV_ENC_BUFFER_FORMAT_NV12_PL;

                cuCtxPopCurrent(&cuContextCurr);
            }

#if defined(NV_WINDOWS) && !defined (_NO_D3D)

            if (m_stEncoderInput[0].interfaceType == NV_ENC_DX9)
            {
                if (i==0)
                {
                    NvPrintf(" > DirectX+NVENC InterOp using %d buffers.\n", m_dwMaxSurfCount);
                }

                // Illustrate how to use an externally allocated IDirect3DSurface9* as input to the encoder.
                IDirect3DSurface9 *pSurf = NULL;
                unsigned int dwFormat = MAKE_FOURCC('N', 'V', '1', '2');;
                HRESULT hr = S_OK;

                if (IsNV12Format(m_dwInputFormat))
                {
                    dwFormat = MAKE_FOURCC('N', 'V', '1', '2');
                }
                else if (IsYV12Format(m_dwInputFormat))
                {
                    dwFormat = MAKE_FOURCC('Y', 'V', '1', '2');
                }

                hr = m_pD3D9Device->CreateOffscreenPlainSurface(dwInputWidth, dwInputHeight, (D3DFORMAT)dwFormat, D3DPOOL_DEFAULT, (IDirect3DSurface9 **)&m_stInputSurface[i].pExtAlloc, NULL);
                m_stInputSurface[i].bufferFmt      = NV_ENC_BUFFER_FORMAT_NV12_PL;
                m_stInputSurface[i].type           = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
            }

#endif
            // Register the allocated buffer with NvEncodeAPI
            NV_ENC_REGISTER_RESOURCE stRegisterRes;
            memset(&stRegisterRes, 0, sizeof(NV_ENC_REGISTER_RESOURCE));
            SET_VER(stRegisterRes, NV_ENC_REGISTER_RESOURCE);
            stRegisterRes.resourceType = m_stInputSurface[i].type;
            // Pass the resource handle to be registered and mapped during registration.
            // Do not pass this handle while mapping
            stRegisterRes.resourceToRegister       = m_stInputSurface[i].pExtAlloc;
            stRegisterRes.width                    = m_stInputSurface[i].dwWidth;
            stRegisterRes.height                   = m_stInputSurface[i].dwHeight;
            stRegisterRes.pitch                    = m_stInputSurface[i].dwCuPitch;

            status = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &stRegisterRes);
            checkNVENCErrors(status);

            // Use this registered handle to retrieve an encoder-understandable mapped resource handle, through NvEncMapInputResource.
            // The mapped handle can be directly used with NvEncEncodePicture.
            m_stInputSurface[i].hRegisteredHandle = stRegisterRes.registeredResource;
        }
        else
        {
            if (i==0)
            {
                NvPrintf(" > System Memory with %d buffers.\n", m_dwMaxSurfCount);
            }

            // Allocate input surface
            NV_ENC_CREATE_INPUT_BUFFER stAllocInputSurface;
            memset(&stAllocInputSurface, 0, sizeof(stAllocInputSurface));
            SET_VER(stAllocInputSurface, NV_ENC_CREATE_INPUT_BUFFER);
            stAllocInputSurface.width              = (m_uMaxWidth  + 31)&~31;//dwFrameWidth;
            stAllocInputSurface.height             = (m_uMaxHeight + 31)&~31; //dwFrameHeight;
#if defined (NV_WINDOWS)
            stAllocInputSurface.memoryHeap         = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
            stAllocInputSurface.bufferFmt          = m_dwInputFormat; // NV_ENC_BUFFER_FORMAT_NV12_PL;
#else
            stAllocInputSurface.memoryHeap         = NV_ENC_MEMORY_HEAP_SYSMEM_UNCACHED;
            stAllocInputSurface.bufferFmt          = m_dwInputFormat;
#endif

            stAllocInputSurface.bufferFmt      = IsNV12Tiled64x16Format(m_dwInputFormat) ? NV_ENC_BUFFER_FORMAT_NV12_PL : m_dwInputFormat;

            status = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &stAllocInputSurface);
            checkNVENCErrors(status);

            m_stInputSurface[i].hInputSurface      = stAllocInputSurface.inputBuffer;
            m_stInputSurface[i].bufferFmt          = stAllocInputSurface.bufferFmt;
            m_stInputSurface[i].dwWidth            = (m_uMaxWidth  + 31)&~31;
            m_stInputSurface[i].dwHeight           = (m_uMaxHeight + 31)&~31;
        }

        m_stInputSurfQueue.Add(&m_stInputSurface[i]);

        //Allocate output surface
        m_stBitstreamBuffer[i].dwSize = 1024*1024;
        NV_ENC_CREATE_BITSTREAM_BUFFER stAllocBitstream;
        memset(&stAllocBitstream, 0, sizeof(stAllocBitstream));
        SET_VER(stAllocBitstream, NV_ENC_CREATE_BITSTREAM_BUFFER);
        stAllocBitstream.size                      =  m_stBitstreamBuffer[i].dwSize;
        stAllocBitstream.memoryHeap                = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

        status = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &stAllocBitstream);
        checkNVENCErrors(status);

        m_stBitstreamBuffer[i].hBitstreamBuffer    = stAllocBitstream.bitstreamBuffer;
        m_stBitstreamBuffer[i].pBitstreamBufferPtr = stAllocBitstream.bitstreamBufferPtr;

        NV_ENC_EVENT_PARAMS nvEventParams = {0};
        SET_VER(nvEventParams, NV_ENC_EVENT_PARAMS);

#if defined (NV_WINDOWS)
        m_stBitstreamBuffer[i].hOutputEvent        = CreateEvent(NULL, FALSE, FALSE, NULL);
        nvEventParams.completionEvent              = m_stBitstreamBuffer[i].hOutputEvent;
#else
        m_stBitstreamBuffer[i].hOutputEvent = NULL;
#endif
        // Register the resource for interop with NVENC
        nvEventParams.completionEvent              = m_stBitstreamBuffer[i].hOutputEvent;
        m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &nvEventParams);

        m_stOutputSurfQueue.Add(&m_stBitstreamBuffer[i]);
    }

    m_stEOSOutputBfr.bEOSFlag = true;
    NV_ENC_EVENT_PARAMS nvEventParams = {0};
    SET_VER(nvEventParams, NV_ENC_EVENT_PARAMS);

#if defined (NV_WINDOWS)
    m_stEOSOutputBfr.hOutputEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    nvEventParams.completionEvent = m_stEOSOutputBfr.hOutputEvent;
#else
    m_stEOSOutputBfr.hOutputEvent = NULL;
#endif
    nvEventParams.completionEvent = m_stEOSOutputBfr.hOutputEvent;
    m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &nvEventParams);

    return S_OK;
}


HRESULT CNvEncoder::ReleaseIOBuffers()
{
    for (unsigned int i = 0; i < m_dwMaxSurfCount; i++)
    {
        m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, m_stInputSurface[i].hInputSurface);
        m_stInputSurface[i].hInputSurface = NULL;

        m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, m_stInputSurface[i].hInputSurface);
        m_stInputSurface[i].hInputSurface = NULL;

        if (m_stEncoderInput[0].useMappedResources)
        {
            // Unregister the registered resource handle before destroying the allocated Cuda buffer[s]
            m_pEncodeAPI->nvEncUnregisterResource(m_hEncoder, m_stInputSurface[i].hRegisteredHandle);

            if (m_stEncoderInput[0].interfaceType == NV_ENC_CUDA)
            {
                cuCtxPushCurrent(m_cuContext);
                CUcontext cuContextCurrent;
                cuMemFree((CUdeviceptr) m_stInputSurface[i].pExtAlloc);
                cuMemFreeHost(m_stInputSurface[i].pExtAllocHost);
                cuCtxPopCurrent(&cuContextCurrent);
            }

#if defined(NV_WINDOWS) && !defined (_NO_D3D)

            if (m_stEncoderInput[0].interfaceType == NV_ENC_DX9)
            {
                IDirect3DSurface9 *pSurf = (IDirect3DSurface9 *)m_stInputSurface[i].pExtAlloc;

                if (pSurf)
                {
                    pSurf->Release();
                }
            }

#endif
        }
        else
        {
            m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, m_stInputSurface[i].hInputSurface);
            m_stInputSurface[i].hInputSurface = NULL;
        }

        m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, m_stBitstreamBuffer[i].hBitstreamBuffer);
        m_stBitstreamBuffer[i].hBitstreamBuffer = NULL;

        NV_ENC_EVENT_PARAMS nvEventParams = {0};
        SET_VER(nvEventParams, NV_ENC_EVENT_PARAMS);
        nvEventParams.completionEvent =  m_stBitstreamBuffer[i].hOutputEvent;
        m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &nvEventParams);

#if defined (NV_WINDOWS)
        CloseHandle(m_stBitstreamBuffer[i].hOutputEvent);
#endif
    }

    if (m_stEOSOutputBfr.hOutputEvent)
    {
        NV_ENC_EVENT_PARAMS nvEventParams = {0};
        SET_VER(nvEventParams, NV_ENC_EVENT_PARAMS);
        nvEventParams.completionEvent =  m_stEOSOutputBfr.hOutputEvent ;
        m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &nvEventParams);
#if defined (NV_WINDOWS)
        CloseHandle(m_stEOSOutputBfr.hOutputEvent);
#endif
        m_stEOSOutputBfr.hOutputEvent  = NULL;
    }

    return S_OK;
}


unsigned char *CNvEncoder::LockInputBuffer(void *hInputSurface, unsigned int *pLockedPitch)
{
    HRESULT hr = S_OK;
    NV_ENC_LOCK_INPUT_BUFFER stLockInputBuffer;
    memset(&stLockInputBuffer, 0, sizeof(stLockInputBuffer));
    SET_VER(stLockInputBuffer, NV_ENC_LOCK_INPUT_BUFFER);
    stLockInputBuffer.inputBuffer = hInputSurface;
    hr = m_pEncodeAPI->nvEncLockInputBuffer(m_hEncoder,&stLockInputBuffer);

    if (hr != S_OK)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "\n unable to lock buffer");
    }

    *pLockedPitch = stLockInputBuffer.pitch;
    return (unsigned char *)stLockInputBuffer.bufferDataPtr;
}


HRESULT CNvEncoder::UnlockInputBuffer(void *hInputSurface)
{
    HRESULT hr = S_OK;
    m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, hInputSurface);

    if (hr != S_OK)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "\n unable to unlock buffer");
    }

    return hr;
}


HRESULT CNvEncoder::CopyBitstreamData(EncoderThreadData stThreadData)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    HRESULT hr = S_OK;

    if (stThreadData.pOutputBfr->hBitstreamBuffer == NULL && stThreadData.pOutputBfr->bEOSFlag == false)
    {
        return E_FAIL;
    }
    if (stThreadData.pOutputBfr->bWaitOnEvent == true)
    {
        if (!stThreadData.pOutputBfr->hOutputEvent)
        {
            return E_FAIL;
        }

#if defined (NV_WINDOWS)
        WaitForSingleObject(stThreadData.pOutputBfr->hOutputEvent, INFINITE);
#endif
    }

    if (stThreadData.pOutputBfr->bEOSFlag)
        return S_OK;

    if (m_stEncoderInput[m_dwReConfigIdx].useMappedResources)
    {
        // unmap the mapped resource ptr
        nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, stThreadData.pInputBfr->hInputSurface);
        stThreadData.pInputBfr->hInputSurface = NULL;
    }

    NV_ENC_LOCK_BITSTREAM lockBitstreamData;
    nvStatus = NV_ENC_SUCCESS;
    memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
    SET_VER(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);

    if (m_stInitEncParams.reportSliceOffsets)
        lockBitstreamData.sliceOffsets = new unsigned int[m_stEncoderInput[m_dwReConfigIdx].numSlices];

    lockBitstreamData.outputBitstream = stThreadData.pOutputBfr->hBitstreamBuffer;
    lockBitstreamData.doNotWait = false;

    nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);

    if (nvStatus == NV_ENC_SUCCESS)
    {
        fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, m_fOutput);
        nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, stThreadData.pOutputBfr->hBitstreamBuffer);
        checkNVENCErrors(nvStatus);
    }

    if (!m_stOutputSurfQueue.Add(stThreadData.pOutputBfr))
    {
        assert(0);
    }

    if (!m_stInputSurfQueue.Add(stThreadData.pInputBfr))
    {
        assert(0);
    }

    if (lockBitstreamData.sliceOffsets)
        delete(lockBitstreamData.sliceOffsets);

    if (nvStatus != NV_ENC_SUCCESS)
        hr = E_FAIL;

    return hr;
}


HRESULT CNvEncoder::CopyFrameData(FrameThreadData stFrameData)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    HRESULT hr = S_OK;

    FILE_64BIT_HANDLE  fileOffset;
    DWORD numBytesRead = 0;

#if defined(WIN32) || defined(WIN64)
    fileOffset.QuadPart = (LONGLONG)stFrameData.dwFileWidth;
#else
    fileOffset = (LONGLONG)stFrameData.dwFileWidth;
#endif

    DWORD result;
    unsigned int dwInFrameSize     = stFrameData.dwFileWidth*stFrameData.dwFileHeight + (stFrameData.dwFileWidth*stFrameData.dwFileHeight)/2;
    unsigned char *yuv[3];

    bool bFieldPic = false, bTopField = false;

    yuv[0] = new unsigned char[stFrameData.dwFileWidth*stFrameData.dwFileHeight];
    yuv[1] = new unsigned char[stFrameData.dwFileWidth*stFrameData.dwFileHeight/4];
    yuv[2] = new unsigned char[stFrameData.dwFileWidth*stFrameData.dwFileHeight/4];

#if defined(WIN32) || defined(WIN64)
    fileOffset.QuadPart = (LONGLONG)(dwInFrameSize * stFrameData.dwFrmIndex);
    result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_BEGIN);
#else
    fileOffset = (LONGLONG)(dwInFrameSize * stFrameData.dwFrmIndex);
    result = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_SET);
#endif

    if (result == FILE_ERROR_SET_FP)
    {
        return E_FAIL;
    }

    if (bFieldPic)
    {
        if (!bTopField)
        {
            // skip the first line
#if defined(WIN32) || defined(WIN64)
            fileOffset.QuadPart  = (LONGLONG)stFrameData.dwFileWidth;
            result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_CURRENT);
#else
            fileOffset = (LONGLONG)stFrameData.dwFileWidth;
            result = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

            if (result == FILE_ERROR_SET_FP)
            {
                return E_FAIL;
            }
        }

        // read Y
        for (unsigned int i = 0 ; i < stFrameData.dwFileHeight/2; i++)
        {
#if defined(WIN32) || defined(WIN64)
            ReadFile(stFrameData.hInputYUVFile, yuv[0] + i*stFrameData.dwSurfWidth, stFrameData.dwFileWidth, &numBytesRead, NULL);
            // skip the next line
            fileOffset.QuadPart  = (LONGLONG)stFrameData.dwFileWidth;
            result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_CURRENT);
#else
            numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, yuv[0] + i*stFrameData.dwSurfWidth, stFrameData.dwFileWidth);
            fileOffset   = (LONGLONG)stFrameData.dwFileWidth;
            result       = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

            if (result == FILE_ERROR_SET_FP)
            {
                return E_FAIL;
            }
        }

        // read U,V
        for (int cbcr = 0; cbcr < 2; cbcr++)
        {
            //put file pointer at the beginning of chroma
#if defined(WIN32) || defined(WIN64)
            fileOffset.QuadPart  = (LONGLONG)(dwInFrameSize*stFrameData.dwFrmIndex + stFrameData.dwFileWidth*stFrameData.dwFileHeight + cbcr*((stFrameData.dwFileWidth*stFrameData.dwFileHeight)/4));
            result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_BEGIN);
#else
            fileOffset   = (LONGLONG)(dwInFrameSize*stFrameData.dwFrmIndex + stFrameData.dwFileWidth*stFrameData.dwFileHeight + cbcr*((stFrameData.dwFileWidth*stFrameData.dwFileHeight)/4));
            result       = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

            if (result == FILE_ERROR_SET_FP)
            {
                return E_FAIL;
            }

            if (!bTopField)
            {
#if defined(WIN32) || defined(WIN64)
                fileOffset.QuadPart  = (LONGLONG)(stFrameData.dwFileWidth/2);
                result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_CURRENT);
#else
                fileOffset = (LONGLONG)(stFrameData.dwFileWidth/2);
                result     = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

                if (result == FILE_ERROR_SET_FP)
                {
                    return E_FAIL;
                }
            }

            for (unsigned int i = 0 ; i < stFrameData.dwFileHeight/4; i++)
            {
#if defined(WIN32) || defined(WIN64)
                ReadFile(stFrameData.hInputYUVFile, yuv[cbcr + 1] + i*(stFrameData.dwSurfWidth/2), stFrameData.dwFileWidth/2, &numBytesRead, NULL);
                fileOffset.QuadPart  = (LONGLONG)stFrameData.dwFileWidth/2;
                result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_CURRENT);
#else
                numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, yuv[0] + i*stFrameData.dwSurfWidth, stFrameData.dwFileWidth);
                fileOffset   = (LONGLONG)stFrameData.dwFileWidth;
                result       = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

                if (result == FILE_ERROR_SET_FP)
                {
                    return E_FAIL;
                }
            }
        }
    }
    else if (stFrameData.dwFileWidth != stFrameData.dwSurfWidth)
    {
        // load the whole frame
        // read Y
        for (unsigned int i = 0 ; i < stFrameData.dwFileHeight; i++)
        {
#if defined(WIN32) || defined(WIN64)
            ReadFile(stFrameData.hInputYUVFile, yuv[0] + i*stFrameData.dwSurfWidth, stFrameData.dwFileWidth, &numBytesRead, NULL);
#else
            numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, yuv[0] + i*stFrameData.dwSurfWidth, stFrameData.dwFileWidth);
#endif
        }

        // read U,V
        for (int cbcr = 0; cbcr < 2; cbcr++)
        {
            // move in front of chroma
#if defined(WIN32) || defined(WIN64)
            fileOffset.QuadPart  = (LONGLONG)(dwInFrameSize*stFrameData.dwFrmIndex + stFrameData.dwFileWidth*stFrameData.dwFileHeight + cbcr*((stFrameData.dwFileWidth* stFrameData.dwFileHeight)/4));
            result = SetFilePointer(stFrameData.hInputYUVFile, fileOffset.LowPart, &fileOffset.HighPart, FILE_BEGIN);
#else
            fileOffset   = (LONGLONG)(dwInFrameSize*stFrameData.dwFrmIndex + stFrameData.dwFileWidth*stFrameData.dwFileHeight + cbcr*((stFrameData.dwFileWidth* stFrameData.dwFileHeight)/4));
            result       = lseek64((LONGLONG)stFrameData.hInputYUVFile, fileOffset, SEEK_CUR);
#endif

            if (result == FILE_ERROR_SET_FP)
            {
                return E_FAIL;
            }

            for (unsigned int i = 0 ; i < stFrameData.dwFileHeight/2; i++)
            {
#if defined(WIN32) || defined(WIN64)
                ReadFile(stFrameData.hInputYUVFile, yuv[cbcr + 1] + i*stFrameData.dwSurfWidth/2, stFrameData.dwFileWidth/2, &numBytesRead, NULL);
#else
                numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, yuv[cbcr + 1] + i*stFrameData.dwSurfWidth/2, stFrameData.dwFileWidth/2);
#endif
            }
        }
    }
    else
    {
#if defined(WIN32) || defined(WIN64)
        // direct file read
        ReadFile(stFrameData.hInputYUVFile, &yuv[0], stFrameData.dwFileWidth * stFrameData.dwFileHeight  , &numBytesRead, NULL);
        ReadFile(stFrameData.hInputYUVFile, &yuv[1], stFrameData.dwFileWidth * stFrameData.dwFileHeight/4, &numBytesRead, NULL);
        ReadFile(stFrameData.hInputYUVFile, &yuv[2], stFrameData.dwFileWidth * stFrameData.dwFileHeight/4, &numBytesRead, NULL);
#else
        numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, &yuv[0], stFrameData.dwFileWidth * stFrameData.dwFileHeight);
        numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, &yuv[1], stFrameData.dwFileWidth * stFrameData.dwFileHeight / 4);
        numBytesRead = read((LONGLONG)stFrameData.hInputYUVFile, &yuv[2], stFrameData.dwFileWidth * stFrameData.dwFileHeight / 4);
#endif
    }

    // We assume input is YUV420, and we want to convert to NV12 Tiled
    //    convertYUVpitchtoNV12tiled16x16(yuv[0], yuv[1], yuv[2], pInputSurface, pInputSurfaceCh, stFrameData.dwWidth, stFrameData.dwHeight, stFrameData.dwWidth, lockedPitch);
    delete yuv[0];
    delete yuv[1];
    delete yuv[2];
    return hr;
}


HRESULT CNvEncoder::FlushEncoder()
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    HRESULT hr = S_OK;
    memset(&m_stEncodePicParams, 0, sizeof(m_stEncodePicParams));
    SET_VER(m_stEncodePicParams, NV_ENC_PIC_PARAMS);
    // This EOS even signals indicates that all frames in the NVENC input queue have been flushed to the
    // HW Encoder and it has been processed.  In some cases, the client might not know when the last frame
    // sent to the encoder is, so it is easier to sometimes insert the EOS and wait for the EOSE event.
    // If used in async mode (windows only), the EOS event must be sent along with the packet to the driver.
    m_stEncodePicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    m_stEncodePicParams.completionEvent = (m_bAsyncModeEncoding == true) ? m_stEOSOutputBfr.hOutputEvent : NULL;
    nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &m_stEncodePicParams);

    if (nvStatus != NV_ENC_SUCCESS)
    {
        assert(0);
        hr = E_FAIL;
    }

    // incase of sync modes we might have frames queued for locking
    if ((m_bAsyncModeEncoding == false) && (m_stInitEncParams.enablePTD == 1))
    {
        EncoderThreadData stThreadData;

        while (m_pEncodeFrameQueue.Remove(stThreadData, 0))
        {
            m_pEncoderThread->QueueSample(stThreadData);
        }
    }

    EncoderThreadData  stThreadData;
    m_stEOSOutputBfr.bWaitOnEvent = m_bAsyncModeEncoding == true ? true : false;
    stThreadData.pOutputBfr = &m_stEOSOutputBfr;
    stThreadData.pInputBfr = NULL;

    if (nvStatus == NV_ENC_SUCCESS)
    {
        // Queue o/p Sample
        if (!m_pEncoderThread->QueueSample(stThreadData))
        {
            assert(0);
        }
    }

    return hr;
}


HRESULT CNvEncoder::ReleaseEncoderResources()
{
    if (m_bEncoderInitialized)
    {
        if (m_pEncoderThread)
        {
            WaitForCompletion();
            m_pEncoderThread->ThreadQuit();
            delete m_pEncoderThread;
            m_pEncoderThread = NULL;
        }

        m_uRefCount--;
    }

    if (m_uRefCount==0)
    {
        NvPrintf("CNvEncoder::ReleaseEncoderResources() m_RefCount == 0, releasing resources\n");
        ReleaseIOBuffers();
        m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);

        if (m_spspps.spsppsBuffer)
            delete[](unsigned char *)m_spspps.spsppsBuffer;

        if (m_spspps.outSPSPPSPayloadSize)
            delete[] m_spspps.outSPSPPSPayloadSize;

        if (m_pAvailableSurfaceFmts)
        {
            delete [] m_pAvailableSurfaceFmts;
            m_pAvailableSurfaceFmts = NULL;
        }

        if (m_cuContext)
        {
            CUresult cuResult = CUDA_SUCCESS;
            cuResult = cuCtxDestroy(m_cuContext);

            if (cuResult != CUDA_SUCCESS)
                nvPrintf(stderr, NV_LOG_ERROR, "cuCtxDestroy error:0x%x\n", cuResult);
        }

#if defined (NV_WINDOWS) && !defined (_NO_D3D)

        if (m_pD3D9Device)
        {
            m_pD3D9Device->Release();
            m_pD3D9Device = NULL;
        }

        if (m_pD3D10Device)
        {
            m_pD3D10Device->Release();
            m_pD3D10Device = NULL;
        }

        if (m_pD3D11Device)
        {
            m_pD3D11Device->Release();
            m_pD3D11Device = NULL;
        }

        if (m_pD3D)
        {
            m_pD3D->Release();
            m_pD3D = NULL;
        }

#endif
    }

    return S_OK;
}

HRESULT CNvEncoder::WaitForCompletion()
{
    if (m_pEncoderThread)
    {
        bool bIdle = false;

        do
        {
            m_pEncoderThread->ThreadLock();
            bIdle = m_pEncoderThread->IsIdle();
            m_pEncoderThread->ThreadUnlock();

            if (!bIdle)
            {
                m_pEncoderThread->ThreadTrigger();
                NvSleep(1);
            }
        }
        while (!bIdle);
    }

    return S_OK;
}


// Encoder thread
bool CNvEncoderThread::ThreadFunc()
{
    EncoderThreadData stThreadData;

    while (m_pEncoderQueue.Remove(stThreadData, 0))
    {
        m_pOwner->CopyBitstreamData(stThreadData);
    }

    return false;
}


bool CNvEncoderThread::QueueSample(EncoderThreadData &sThreadData)
{
    bool bIsEnqueued = m_pEncoderQueue.Add(sThreadData);
    ThreadTrigger();
    return bIsEnqueued;
}


HRESULT CNvEncoder::GetPresetConfig(int iPresetIdx)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    unsigned int uPresetCount = 0;
    unsigned int uPresetCount2 = 0;
    GUID *pPresetGUIDs = NULL;

    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, m_stEncodeGUID, &uPresetCount);

    if (nvStatus == NV_ENC_SUCCESS)
    {
        pPresetGUIDs = new GUID[uPresetCount];
        nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, m_stEncodeGUID, pPresetGUIDs, uPresetCount, &uPresetCount2);

        if (nvStatus == NV_ENC_SUCCESS)
        {
            nvStatus = NV_ENC_ERR_GENERIC;

            for (unsigned int i = 0; i < uPresetCount; i++)
            {
                if ((int)i == iPresetIdx)
                {
                    nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, m_stEncodeGUID, pPresetGUIDs[iPresetIdx], &m_stPresetConfig);

                    if (nvStatus == NV_ENC_SUCCESS)
                    {
                        m_stPresetGUID = pPresetGUIDs[iPresetIdx];
                    }

                    break;
                }
            }
        }

        delete[] pPresetGUIDs;
    }

    if (nvStatus != NV_ENC_SUCCESS)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "\n Wrong preset Value. Exiting \n");
        checkNVENCErrors(nvStatus);
        return E_FAIL;
    }
        return S_OK;

}


HRESULT CNvEncoder::QueryEncodeCaps(NV_ENC_CAPS caps_type, int *p_nCapsVal)
{
    HRESULT hr = S_OK;
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_CAPS_PARAM stCapsParam = {0};
    SET_VER(stCapsParam, NV_ENC_CAPS_PARAM);
    stCapsParam.capsToQuery = caps_type;

    nvStatus = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, m_stEncodeGUID, &stCapsParam, p_nCapsVal);
    checkNVENCErrors(nvStatus);

    if (nvStatus == NV_ENC_SUCCESS)
    {
        return S_OK;
    }
    else
    {
        return E_FAIL;
    }
}

void CNvEncoder::DisplayEncodingParams(EncoderAppParams pEncodeAppParams, int numConfigured)
{
	NvPrintf("\n\n\n> NVENC Encoder[%d] configuration parameters for configuration #%d\n", 0, numConfigured);
    NvPrintf("> GPU Device ID             = %d\n",            pEncodeAppParams.nDeviceID);
    NvPrintf("> Frames                    = %d frames \n",    pEncodeAppParams.numFramesToEncode);
    NvPrintf("> ConfigFile                = %s \n",           pEncodeAppParams.configFile);
    NvPrintf("> Frame at which %dth configuration will happen = %d \n", numConfigured, m_stEncoderInput[numConfigured].endFrame);

    NvPrintf("> maxWidth,maxHeight        = [%04d,%04d]\n",   m_stEncoderInput[numConfigured].maxWidth,  m_stEncoderInput[numConfigured].maxHeight);
    NvPrintf("> Width,Height              = [%04d,%04d]\n",   m_stEncoderInput[numConfigured].width,  m_stEncoderInput[numConfigured].height);
    NvPrintf("> Video Output Codec        = %d - %s\n",       m_stEncoderInput[numConfigured].codec, codec_names[ m_stEncoderInput[numConfigured].codec].name);
    NvPrintf("> Average Bitrate           = %d (bps/sec)\n",  m_stEncoderInput[numConfigured].avgBitRate);
    NvPrintf("> Peak Bitrate              = %d (bps/sec)\n",  m_stEncoderInput[numConfigured].peakBitRate);
    NvPrintf("> Rate Control Mode         = %d - %s\n",       m_stEncoderInput[numConfigured].rateControl, ratecontrol_names[ m_stEncoderInput[numConfigured].rateControl].name);
    NvPrintf("> Frame Rate (Num/Denom)    = (%d/%d) %4.4f fps\n", m_stEncoderInput[numConfigured].frameRateNum, m_stEncoderInput[numConfigured].frameRateDen, (float) m_stEncoderInput[numConfigured].frameRateNum/(float) m_stEncoderInput[numConfigured].frameRateDen);
    NvPrintf("> GOP Length                = %d\n",            m_stEncoderInput[numConfigured].gopLength);
    NvPrintf("> Set Initial RC      QP    = %d\n",            m_stEncoderInput[numConfigured].enableInitialRCQP);
    NvPrintf("> Initial RC QP (I,P,B)     = I(%d), P(%d), B(%d)\n", m_stEncoderInput[numConfigured].initialRCQP.qpIntra,  m_stEncoderInput[numConfigured].initialRCQP.qpInterP,  m_stEncoderInput[numConfigured].initialRCQP.qpInterB);
    NvPrintf("> Number of B Frames        = %d\n",            m_stEncoderInput[numConfigured].numBFrames);
    NvPrintf("> Display Aspect Ratio X    = %d\n",            m_stEncoderInput[numConfigured].width);
    NvPrintf("> Display Aspect Ratio Y    = %d\n",            m_stEncoderInput[numConfigured].height);
    NvPrintf("> Video codec profile       = %d\n",            m_stEncoderInput[numConfigured].profile);
    NvPrintf("> Video codec Level         = %d\n",            m_stEncoderInput[numConfigured].level);
    NvPrintf("> FieldEncoding             = %d\n",            m_stEncoderInput[numConfigured].fieldEncoding);

    NvPrintf("> Number slices per Frame   = %d\n",            m_stEncoderInput[numConfigured].numSlices);
    NvPrintf("> Encoder Preset            = %d - %s\n",       m_stEncoderInput[numConfigured].preset, preset_names[ m_stEncoderInput[numConfigured].preset].name);
#if defined (NV_WINDOWS)
    NvPrintf("> Asynchronous Mode         = %s\n",            m_stEncoderInput[numConfigured].syncMode ? "No" : "Yes");
#endif
    NvPrintf("> NVENC API Interface       = %d - %s\n",       m_stEncoderInput[numConfigured].interfaceType, nvenc_interface_names[ m_stEncoderInput[numConfigured].interfaceType].name);
}
bool CNvEncoder::ParseConfigString(const char *str)
{
    char *line, *name, *p;
    char in_string;
    configs_s *cfg;
    bool ret = false;
    EncodeConfig *par = &m_stEncoderInput[0];
    line = _strdup(str);

    // remove comments
    p = line;
    in_string = 0;
    while (*p)
    {
        if (in_string)
        {
            if (*p == in_string)
                in_string = 0;
        }
        else if (*p == '"' || *p == '\'')
        {
            in_string = *p;
        }
        else if (*p == '#' || (*p == '/' && *(p+1) == '/'))
        {
            *p = '\0';
            break;
        }
        p++;
    }

    name = line;
    while (*name && isspace(*name))
        name++;
    if (!*name)
        goto done;
    p = name;
    while (*p && !(isspace(*p) || *p == '='))
        p++;
    if (!*p)
        goto exit;
    *p++ = 0;
    while (*p && (isspace(*p) || *p == '='))
        p++;

    for (cfg=m_configs; cfg->str; cfg++)
    {
        if (strcmp(cfg->str, name) == 0)
            break;
    }
    if (!cfg->str)
        goto exit;
    switch (cfg->type)
    {
    case 0: // int
        sscanf(p, "%d", (int *)((unsigned char *)par + cfg->offset));
        break;
    case 1: //String
        sscanf(p, "%s", (char *)((unsigned char *)par + cfg->offset));
        break;
    default:
        goto exit;
    }
done:
    ret = true;
exit:
    if (line)
        free(line);
    if (!ret)
        nvPrintf(stderr, NV_LOG_ERROR, "Ignoring %s\n", str);
    return ret;
}


bool CNvEncoder::ParseConfigFile(const char *file)
{
    FILE *fd;
    char str[512];

    fd = fopen(file, "r");
    if (fd == NULL)
        return false;
    while (fgets(str, sizeof(str), fd))
        ParseConfigString(str);
    fclose(fd);
    return true;
}

bool CNvEncoder::ParseReConfigString(const char *str)
{
    char *line, *name, *p;
    char in_string;
    configs_s *cfg;
    configs_s cfg1 = {"frameNum", 0, offsetof(EncodeConfig, endFrame)};
    bool ret = false;
    EncodeConfig *par = &m_stEncoderInput[m_dwNumReConfig];
    line = _strdup(str);

    // remove comments
    p = line;
    in_string = 0;
    while (*p)
    {
        if (in_string)
        {
            if (*p == in_string)
                in_string = 0;
        }
        else if (*p == '"' || *p == '\'')
        {
            in_string = *p;
        }
        else if (*p == '#' || (*p == '/' && *(p+1) == '/'))
        {
            *p = '\0';
            break;
        }
        p++;
    }

    name = line;
    while (*name && isspace(*name))
        name++;
    if (!*name)
        goto done;
    p = name;
    while (*p && !(isspace(*p) || *p == '='))
        p++;
    if (!*p)
        goto exit;
    *p++ = 0;
    while (*p && (isspace(*p) || *p == '='))
        p++;
    if (strcmp(cfg1.str, name) == 0)
    {
        par = &m_stEncoderInput[m_dwNumReConfig];
        sscanf(p, "%d", (int *)((unsigned char *)par + cfg1.offset));
        m_dwNumReConfig = m_dwNumReConfig + 1;
        goto done;
    }
    if(m_dwNumReConfig >=1)
    {
        for (cfg=m_configs; cfg->str; cfg++)
        {
            if (strcmp(cfg->str, name) == 0)
                break;
        }
    }
    if (!cfg->str)
        goto exit;
    switch (cfg->type)
    {
    case 0: // int
        sscanf(p, "%d", (int *)((unsigned char *)par + cfg->offset));
        break;
    case 1: //String
        sscanf(p, "%s", (char *)((unsigned char *)par + cfg->offset));
        break;
    default:
        goto exit;
    }
done:
    ret = true;
exit:
    if (line)
        free(line);
    if (!ret)
        nvPrintf(stderr, NV_LOG_ERROR, "Ignoring %s\n", str);
    return ret;
}

bool CNvEncoder::ParseReConfigFile(char *reConfigFile)
{
    m_dwNumReConfig =0;
    FILE *fd;
    char str[512];

    fd = fopen(reConfigFile, "r");
    if (fd == NULL)
        return false;
    while (fgets(str, sizeof(str), fd))
    {
        ParseReConfigString(str);
        if( m_dwNumReConfig >9)
        {
            NvPrintf("\n Maximum 9 re-configuration supported Ignoring extra entries \n");
            break;
        }
    }
    fclose(fd);
    return true;
}
void CNvEncoder::ParseCommandlineInput(int argc, const char *argv[])
{
    m_stEncoderInput[0].fieldEncoding  = m_stEncodeConfig.frameFieldMode == NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME ? 0 : 1;
    m_stEncoderInput[0].numBFrames     = m_stEncodeConfig.frameIntervalP -1;
    m_stEncoderInput[0].gopLength      = m_stEncodeConfig.gopLength;
    m_stEncoderInput[0].mvPrecision    = m_stEncodeConfig.mvPrecision;
    m_stEncoderInput[0].profile        = GetCodecProfile( m_stEncodeConfig.profileGUID);
    m_stEncoderInput[0].rateControl    = m_stEncodeConfig.rcParams.rateControlMode;
    m_stEncoderInput[0].level          = m_stEncodeConfig.encodeCodecConfig.h264Config.level;

    //Command line overwriting the preset & configFile value
    getCmdLineArgumentValue(argc, (const char **)argv, "fieldMode",             &m_stEncoderInput[0].fieldEncoding);
    getCmdLineArgumentValue(argc, (const char **)argv, "bottomfieldFirst",      &m_stEncoderInput[0].bottomFieldFrist);
    getCmdLineArgumentValue(argc, (const char **)argv, "bitrate",               &m_stEncoderInput[0].avgBitRate);
    getCmdLineArgumentValue(argc, (const char **)argv, "maxbitrate",            &m_stEncoderInput[0].peakBitRate);
    getCmdLineArgumentValue(argc, (const char **)argv, "rcMode",                &m_stEncoderInput[0].rateControl);
    getCmdLineArgumentValue(argc, (const char **)argv, "numBFrames",            &m_stEncoderInput[0].numBFrames);
    getCmdLineArgumentValue(argc, (const char **)argv, "frameRateNum",          &m_stEncoderInput[0].frameRateNum);
    getCmdLineArgumentValue(argc, (const char **)argv, "frameRateDen",          &m_stEncoderInput[0].frameRateDen);
    getCmdLineArgumentValue(argc, (const char **)argv, "gopLength",             &m_stEncoderInput[0].gopLength);
    getCmdLineArgumentValue(argc, (const char **)argv, "enableInitialRCQP",     &m_stEncoderInput[0].enableInitialRCQP);
    getCmdLineArgumentValue(argc, (const char **)argv, "initialQPI",            &m_stEncoderInput[0].initialRCQP.qpIntra);
    getCmdLineArgumentValue(argc, (const char **)argv, "initialQPP",            &m_stEncoderInput[0].initialRCQP.qpInterP);
    getCmdLineArgumentValue(argc, (const char **)argv, "initialQPB",            &m_stEncoderInput[0].initialRCQP.qpInterB);
    getCmdLineArgumentValue(argc, (const char **)argv, "profile",               &m_stEncoderInput[0].profile);
    getCmdLineArgumentValue(argc, (const char **)argv, "level",                 &m_stEncoderInput[0].level);
    getCmdLineArgumentValue(argc, (const char **)argv, "numslices",             &m_stEncoderInput[0].numSlices);
    getCmdLineArgumentValue(argc, (const char **)argv, "vbvBufferSize",         &m_stEncoderInput[0].vbvBufferSize);
    getCmdLineArgumentValue(argc, (const char **)argv, "vbvInitialDelay",       &m_stEncoderInput[0].vbvInitialDelay);
    getCmdLineArgumentValue(argc, (const char **)argv, "enablePtd",             &m_stEncoderInput[0].enablePTD);

    char *InputFile =NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "inFile"            ,&InputFile);
    if( InputFile != NULL)
        memcpy(&m_stEncoderInput[0].InputClip, InputFile, 256);

    getCmdLineArgumentValue(argc, (const char **)argv, "syncMode",              &m_stEncoderInput[0].syncMode);
    getCmdLineArgumentValue(argc, (const char **)argv, "width",                 &m_stEncoderInput[0].width);
    getCmdLineArgumentValue(argc, (const char **)argv, "height",                &m_stEncoderInput[0].height);
    getCmdLineArgumentValue(argc, (const char **)argv, "maxwidth",              &m_stEncoderInput[0].maxWidth);
    getCmdLineArgumentValue(argc, (const char **)argv, "maxheight",             &m_stEncoderInput[0].maxHeight);

    m_stEncodeConfig.rcParams.averageBitRate    = m_stEncoderInput[0].avgBitRate;
    m_stEncodeConfig.frameFieldMode             = m_stEncoderInput[0].fieldEncoding ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME : NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
    m_stEncodeConfig.frameIntervalP             = m_stEncoderInput[0].numBFrames + 1;
    m_stEncodeConfig.gopLength                  = m_stEncoderInput[0].gopLength;
    m_stEncodeConfig.mvPrecision                = m_stEncoderInput[0].mvPrecision;
    m_stEncodeConfig.profileGUID                = GetCodecProfileGuid(m_stEncoderInput[0].profile);
    m_stEncodeConfig.rcParams.averageBitRate    = m_stEncoderInput[0].avgBitRate;
    m_stEncodeConfig.rcParams.enableInitialRCQP = m_stEncoderInput[0].enableInitialRCQP;
    m_stEncodeConfig.rcParams.initialRCQP.qpInterB  = m_stEncoderInput[0].initialRCQP.qpInterB;
    m_stEncodeConfig.rcParams.initialRCQP.qpInterP  = m_stEncoderInput[0].initialRCQP.qpInterP;
    m_stEncodeConfig.rcParams.initialRCQP.qpIntra   = m_stEncoderInput[0].initialRCQP.qpIntra;
    m_stEncodeConfig.rcParams.constQP.qpInterB      = m_stEncoderInput[0].initialRCQP.qpInterB;
    m_stEncodeConfig.rcParams.constQP.qpInterP      = m_stEncoderInput[0].initialRCQP.qpInterP;
    m_stEncodeConfig.rcParams.constQP.qpIntra       = m_stEncoderInput[0].initialRCQP.qpIntra;

    m_stEncodeConfig.rcParams.rateControlMode       = (NV_ENC_PARAMS_RC_MODE)m_stEncoderInput[0].rateControl;
    m_stEncodeConfig.rcParams.vbvBufferSize         = m_stEncoderInput[0].vbvBufferSize;
    m_stEncodeConfig.rcParams.vbvInitialDelay       = m_stEncoderInput[0].vbvInitialDelay;
    m_stEncodeConfig.encodeCodecConfig.h264Config.level = m_stEncoderInput[0].level;
}

HRESULT CNvEncoder::LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile, unsigned int dwFrmIndex)
{
    U64 fileOffset;
    U32 numBytesRead = 0;
    U32 result;
    unsigned int dwFileWidth    = m_stEncoderInput[m_dwReConfigIdx].width;
    unsigned int dwFileHeight   = m_stEncoderInput[m_dwReConfigIdx].height;
    U64 dwInFrameSize           = dwFileWidth*(U64 )dwFileHeight + (dwFileWidth*dwFileHeight)/2;

    dwInFrameSize = dwFileWidth * dwFileHeight + (dwFileWidth * dwFileHeight) / 2;

    fileOffset = ((U64)dwInFrameSize * (U64)dwFrmIndex);
    result = nvSetFilePointer64(hInputYUVFile, fileOffset, NULL, FILE_BEGIN);

    if (result == INVALID_SET_FILE_POINTER)
    {
        return E_FAIL;
    }

    // direct file read
    nvReadFile(hInputYUVFile, yuvInput[0], dwFileWidth * dwFileHeight, &numBytesRead, NULL);
    nvReadFile(hInputYUVFile, yuvInput[1], dwFileWidth * dwFileHeight/4, &numBytesRead, NULL);
    nvReadFile(hInputYUVFile, yuvInput[2], dwFileWidth * dwFileHeight/4, &numBytesRead, NULL);

    return S_OK;
}

HRESULT CNvEncoder::OpenEncodeSession(int argc, const char *argv[],unsigned int deviceID)
{
    HRESULT hr = S_OK;
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    m_fOutput = m_stEncoderInput[0].fOutput;
    bool bCodecFound = false;
    NV_ENC_CAPS_PARAM stCapsParam = {0};
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS stEncodeSessionParams = {0};
    unsigned int uArraysize = 0;
    SET_VER(stCapsParam, NV_ENC_CAPS_PARAM);
    SET_VER(stEncodeSessionParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);
    GUID clientKey = NV_CLIENT_KEY_TEST;

    stEncodeSessionParams.apiVersion = NVENCAPI_VERSION;
    stEncodeSessionParams.clientKeyPtr = &clientKey;

    switch (m_stEncoderInput[0].interfaceType)
    {
#if defined (NV_WINDOWS) && !defined (_NO_D3D)

        case NV_ENC_DX9:
            {
                InitD3D9(deviceID);
                stEncodeSessionParams.device = reinterpret_cast<void *>(m_pD3D9Device);
                stEncodeSessionParams.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
            }
            break;

        case NV_ENC_DX10:
            {
                InitD3D10(deviceID);
                stEncodeSessionParams.device = reinterpret_cast<void *>(m_pD3D10Device);
                stEncodeSessionParams.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
            }
            break;

        case NV_ENC_DX11:
            {
                InitD3D11(deviceID);
                stEncodeSessionParams.device = reinterpret_cast<void *>(m_pD3D11Device);
                stEncodeSessionParams.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
            }
            break;
#endif

        case NV_ENC_CUDA:
            {
                InitCuda(deviceID);
                stEncodeSessionParams.device = reinterpret_cast<void *>(m_cuContext);
                stEncodeSessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            }
            break;

        default:
            {
                assert("Encoder interface not supported");
                exit(EXIT_FAILURE);
            }
    }

    nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&stEncodeSessionParams, &m_hEncoder);
	m_uRefCount++;

    if (nvStatus != NV_ENC_SUCCESS)
    {
        nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncOpenEncodeSessionEx() returned with error %d\n", __FILE__, __LINE__, nvStatus);
        nvPrintf(stderr, NV_LOG_ERROR, "Note: GUID key may be invalid or incorrect.  Recommend to upgrade your drivers and obtain a new key\n");
        checkNVENCErrors(nvStatus);
    }
    else
    {
        // Enumerate the codec support by the HW Encoder
        GUID *stEncodeGUIDArray;
        nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &m_dwEncodeGUIDCount);

        if (nvStatus != NV_ENC_SUCCESS)
        {
            nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetEncodeGUIDCount() returned with error %d\n", __FILE__, __LINE__, nvStatus);
            checkNVENCErrors(nvStatus);
        }
        else
        {
            stEncodeGUIDArray       = new GUID[m_dwEncodeGUIDCount];
            memset(stEncodeGUIDArray, 0, sizeof(GUID) * m_dwEncodeGUIDCount);
            uArraysize = 0;
            nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, stEncodeGUIDArray, m_dwEncodeGUIDCount, &uArraysize);
            assert(uArraysize <= m_dwEncodeGUIDCount);

            if (nvStatus != NV_ENC_SUCCESS)
            {
                nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetEncodeGUIDs() returned with error %d\n", __FILE__, __LINE__, nvStatus);
                checkNVENCErrors(nvStatus);
            }
            else
            {
                for (unsigned int i = 0; i < uArraysize; i++)
                {
                    // check if HW encoder supports the particular codec
                    if (GetCodecType(stEncodeGUIDArray[i]) == (unsigned int)m_stEncoderInput[0].codec)
                    {
                        bCodecFound = true;
                        memcpy(&m_stEncodeGUID, &stEncodeGUIDArray[i], sizeof(GUID));
                        break;
                    }
                }
            }

            delete [] stEncodeGUIDArray;
        }

        if (bCodecFound == false)
        {
            assert(0);
            return E_FAIL;
        }

        if (m_stEncoderInput[0].preset > -1)
        {
            hr = GetPresetConfig(m_stEncoderInput[0].preset);

            if (hr == S_OK)
            {
                memcpy(&m_stEncodeConfig, &m_stPresetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
            }
            //Overwriting Default or Config file parameter using Preset parameter followed by command line input.
            ParseCommandlineInput(argc, (const char **)argv);
        }
        // Enumerate the codec profile GUIDS
        nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(m_hEncoder, m_stEncodeGUID, &m_dwCodecProfileGUIDCount);

        if (nvStatus != NV_ENC_SUCCESS)
        {
            nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetEncodeProfileGUIDCount() returned with error %d\n", __FILE__, __LINE__, nvStatus);
            checkNVENCErrors(nvStatus);
        }
        else
        {
            GUID *pProfileGUIDArray       = new GUID[m_dwCodecProfileGUIDCount];
            memset(pProfileGUIDArray, 0, sizeof(GUID) * m_dwCodecProfileGUIDCount);
            uArraysize = 0;
            nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder,  m_stEncodeGUID, pProfileGUIDArray, m_dwCodecProfileGUIDCount, &uArraysize);
            assert(uArraysize <= m_dwCodecProfileGUIDCount);

            if (nvStatus != NV_ENC_SUCCESS)
            {
                nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetEncodeProfileGUIDs() returned with error %d\n", __FILE__, __LINE__, nvStatus);
                checkNVENCErrors(nvStatus);
            }
            else
            {
                bCodecFound = false;

                for (unsigned int i = 0; i < uArraysize; i++)
                {
                    // check if HW encoder supports the particular codec
                    if (GetCodecProfile(pProfileGUIDArray[i]) == m_stEncoderInput[0].profile)
                    {
                        bCodecFound = true;
                        memcpy(&m_stCodecProfileGUID, &pProfileGUIDArray[i], sizeof(GUID));
                        break;
                    }
                }
            }

            delete [] pProfileGUIDArray;
        }

        if (bCodecFound == false)
        {
            assert(0);
            return E_FAIL;
        }

        nvStatus =  m_pEncodeAPI->nvEncGetInputFormatCount(m_hEncoder, m_stEncodeGUID, &m_dwInputFmtCount);

        if (nvStatus != NV_ENC_SUCCESS)
        {
            nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetInputFormatCount() returned with error %d\n", __FILE__, __LINE__, nvStatus);
            checkNVENCErrors(nvStatus);
        }
        else
        {
            m_pAvailableSurfaceFmts = new NV_ENC_BUFFER_FORMAT[m_dwInputFmtCount];
            memset(m_pAvailableSurfaceFmts, 0, sizeof(NV_ENC_BUFFER_FORMAT) * m_dwInputFmtCount);
            nvStatus = m_pEncodeAPI->nvEncGetInputFormats(m_hEncoder, m_stEncodeGUID, m_pAvailableSurfaceFmts, m_dwInputFmtCount, &uArraysize);

            if (nvStatus != NV_ENC_SUCCESS)
            {
                nvPrintf(stderr, NV_LOG_ERROR, "File: %s, Line: %d, nvEncGetInputFormats() returned with error %d\n", __FILE__, __LINE__, nvStatus);
                checkNVENCErrors(nvStatus);
            }
            else
            {
                bool bFmtFound = false;
                unsigned int idx;

                for (idx = 0; idx < m_dwInputFmtCount; idx++)
                {
                    if (m_pAvailableSurfaceFmts[idx] == NV_ENC_BUFFER_FORMAT_NV12_TILED64x16)
                    {
                        bFmtFound = true;
                        break;
                    }
                }

                if (bFmtFound == true)
                    m_dwInputFormat = m_pAvailableSurfaceFmts[idx];
                else
                    assert(0);

                assert(uArraysize <= m_dwInputFmtCount);
            }
        }
    }
    return hr;
}
#pragma warning(push)
#pragma warning(disable:4100)
int CNvEncoder::nvPrintf(FILE *fp, int log_level, const TCHAR *format, ...) {
	if (log_level < m_log_level)
		return 0;

	va_list args;
	va_start(args, format);

	int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
	TCHAR *buffer = (TCHAR *)malloc(len * sizeof(buffer[0]));
	if (NULL != buffer) {

		_vstprintf_s(buffer, len, format, args); // C4996

#ifdef UNICODE
		DWORD mode = 0;
		bool stderr_write_to_console = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか

		char *buffer_char = NULL;
		if (m_pStrLog || !stderr_write_to_console) {
			if (NULL != (buffer_char = (char *)calloc(len * 2, sizeof(buffer_char[0]))))
				WideCharToMultiByte(CP_THREAD_ACP, WC_NO_BEST_FIT_CHARS, buffer, -1, buffer_char, len * 2, NULL, NULL);
		}
		if (buffer_char) {
#else
			char *buffer_char = buffer;
#endif
			if (m_pLogFileName) {
				FILE *fp_log = NULL;
				//logはANSI(まあようはShift-JIS)で保存する
				if (0 == _tfopen_s(&fp_log, m_pLogFileName, _T("a")) && fp_log) {
					fprintf(fp_log, buffer_char);
					fclose(fp_log);
				}
			}
#ifdef UNICODE
			if (!stderr_write_to_console) //出力先がリダイレクトされるならANSIで
				nvPrintf(stderr, NV_LOG_ERROR, buffer_char);
			free(buffer_char);
		}
		if (stderr_write_to_console) //出力先がコンソールならWCHARで
#endif
			_ftprintf(fp, buffer);  
		free(buffer);
	}
	return len;
}
#pragma warning(pop)
