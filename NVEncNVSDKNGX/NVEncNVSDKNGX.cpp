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

#include "rgy_version.h"
#include "rgy_osdep.h"
#define NVENC_NVSDKNGX_EXPORTS
#include "NVEncNVSDKNGX.h"

#if ENABLE_NVSDKNGX
#include "rgy_log.h"
#include "rgy_device.h"
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_defs_truehdr.h>
#include <nvsdk_ngx_helpers_truehdr.h>
#include <nvsdk_ngx_defs_vsr.h>
#include <nvsdk_ngx_helpers_vsr.h>

#define APP_ID      0
#define APP_PATH    L"."

class NVEncNVSDKNGX {
public:
    NVEncNVSDKNGX();
    virtual ~NVEncNVSDKNGX();

    RGY_ERR init(ID3D11Device* pD3DDevice, ID3D11DeviceContext* pD3D11DeviceContext);
    void close();
    RGY_ERR procFrame(ID3D11Texture2D* frameDst, const NVEncNVSDKNGXRect *rectDst, ID3D11Texture2D* frameSrc, const NVEncNVSDKNGXRect *rectSrc, const int quality);
protected:
    ID3D11Device*               m_pD3D11Device;
    ID3D11DeviceContext*        m_pD3D11DeviceContext;
    ID3D10Multithread*          m_pMultiThread;

    NVSDK_NGX_Parameter*        m_ngxParameters;
    NVSDK_NGX_Handle*           m_VSRFeature;

    ID3D11Texture2D*            m_pDstTmpNGX;
    UINT                        m_dstTmpWidth;
    UINT                        m_dstTmpHeight;
};

// /MTでリンクするので、nvsdk_ngx_s.libをリンク
// /MDならnvsdk_ngx_d.libをリンクする必要があるがここでは/MTを前提とする
#if defined(NDEBUG)
#pragma comment( lib, "nvsdk_ngx_s.lib" )
#else
#pragma comment( lib, "nvsdk_ngx_s_dbg.lib" )
#endif

struct RGYErrMapNVSDKNGX {
    RGY_ERR rgy;
    NVSDK_NGX_Result nv;
};

#define NVSDKNGXERR_MAP(x) { RGY_ERR_NVSDK_NGX_ ##x, NVSDK_NGX_Result_FAIL_ ##x }
static const RGYErrMapNVSDKNGX ERR_MAP_NVOFFRUC[] = {
    { RGY_ERR_NONE, NVSDK_NGX_Result_Success },
    { RGY_ERR_UNKNOWN, NVSDK_NGX_Result_Fail },
    NVSDKNGXERR_MAP(FeatureNotSupported),
    NVSDKNGXERR_MAP(PlatformError),
    NVSDKNGXERR_MAP(FeatureAlreadyExists),
    NVSDKNGXERR_MAP(FeatureNotFound),
    NVSDKNGXERR_MAP(InvalidParameter),
    NVSDKNGXERR_MAP(ScratchBufferTooSmall),
    NVSDKNGXERR_MAP(NotInitialized),
    NVSDKNGXERR_MAP(UnsupportedInputFormat),
    NVSDKNGXERR_MAP(RWFlagMissing),
    NVSDKNGXERR_MAP(MissingInput),
    NVSDKNGXERR_MAP(UnableToInitializeFeature),
    NVSDKNGXERR_MAP(OutOfDate),
    NVSDKNGXERR_MAP(OutOfGPUMemory),
    NVSDKNGXERR_MAP(UnsupportedFormat),
    NVSDKNGXERR_MAP(UnableToWriteToAppDataPath),
    NVSDKNGXERR_MAP(UnsupportedParameter),
    NVSDKNGXERR_MAP(Denied),
    NVSDKNGXERR_MAP(NotImplemented),
};

static NVSDK_NGX_Result err_to_nvsdk_ngx(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return NVSDK_NGX_Result_Success;
    const RGYErrMapNVSDKNGX *ERR_MAP_FIN = (const RGYErrMapNVSDKNGX *)ERR_MAP_NVOFFRUC + _countof(ERR_MAP_NVOFFRUC);
    auto ret = std::find_if((const RGYErrMapNVSDKNGX *)ERR_MAP_NVOFFRUC, ERR_MAP_FIN, [err](const RGYErrMapNVSDKNGX map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? NVSDK_NGX_Result_Fail : ret->nv;
}

static RGY_ERR err_to_rgy(NVSDK_NGX_Result err) {
    if (err == NVSDK_NGX_Result_Success) return RGY_ERR_NONE;
    const RGYErrMapNVSDKNGX *ERR_MAP_FIN = (const RGYErrMapNVSDKNGX *)ERR_MAP_NVOFFRUC + _countof(ERR_MAP_NVOFFRUC);
    auto ret = std::find_if((const RGYErrMapNVSDKNGX *)ERR_MAP_NVOFFRUC, ERR_MAP_FIN, [err](const RGYErrMapNVSDKNGX map) {
        return map.nv == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

NVEncNVSDKNGX::NVEncNVSDKNGX() :
    m_pD3D11Device(nullptr),
    m_pD3D11DeviceContext(nullptr),
    m_pMultiThread(nullptr),
    m_ngxParameters(nullptr),
    m_VSRFeature(nullptr),
    m_pDstTmpNGX(nullptr),
    m_dstTmpWidth(0),
    m_dstTmpHeight(0) {

}

NVEncNVSDKNGX::~NVEncNVSDKNGX() {
    close();
}

void NVEncNVSDKNGX::close() {
    if (m_pD3D11Device) {
        NVSDK_NGX_D3D11_ReleaseFeature(m_VSRFeature);
        m_VSRFeature = nullptr;
        NVSDK_NGX_D3D11_Shutdown1(m_pD3D11Device);
        NVSDK_NGX_D3D11_DestroyParameters(m_ngxParameters);
        m_ngxParameters = nullptr;
    }
    if (m_pDstTmpNGX) {
        m_pDstTmpNGX->Release();
        m_pDstTmpNGX = nullptr;
    }
    if (m_pMultiThread) {
        m_pMultiThread->Release();
        m_pMultiThread = nullptr;
    }
    if (m_pD3D11DeviceContext) {
        m_pD3D11DeviceContext->Release();
        m_pD3D11DeviceContext = nullptr;
    }
    if (m_pD3D11Device) {
        m_pD3D11Device->Release();
        m_pD3D11Device = nullptr;
    }
}

RGY_ERR NVEncNVSDKNGX::init(ID3D11Device *pD3DDevice, ID3D11DeviceContext *pD3D11DeviceContext) {
    HRESULT hr = S_OK;

    // init NGX SDK
    auto err = err_to_rgy(NVSDK_NGX_D3D11_Init(APP_ID, APP_PATH, pD3DDevice));
    if (err != RGY_ERR_NONE) return err;

    // Get NGX parameters interface (managed and released by NGX)
    err = err_to_rgy(NVSDK_NGX_D3D11_GetCapabilityParameters(&m_ngxParameters));
    if (err != RGY_ERR_NONE) return err;

    // Now check if VSR is available on the system
    int VSRAvailable = 0;
    err = err_to_rgy(m_ngxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &VSRAvailable));
    if (err != RGY_ERR_NONE) return err;

    pD3DDevice->AddRef();
    pD3D11DeviceContext->AddRef();
    m_pD3D11Device = pD3DDevice;
    m_pD3D11DeviceContext = pD3D11DeviceContext;

    hr = pD3D11DeviceContext->QueryInterface(__uuidof(ID3D10Multithread), (void**)&m_pMultiThread);
    if (SUCCEEDED(hr)) {
        m_pMultiThread->SetMultithreadProtected(TRUE);
        m_pMultiThread->Enter();
    }

    NVSDK_NGX_Feature_Create_Params VSRCreateParams = {};
    err = err_to_rgy(NGX_D3D11_CREATE_VSR_EXT(pD3D11DeviceContext, &m_VSRFeature, m_ngxParameters, &VSRCreateParams));
    if (err != RGY_ERR_NONE) return err;

    if (m_pMultiThread) {
        m_pMultiThread->Leave();
    }

    return RGY_ERR_NONE;
}

RGY_ERR NVEncNVSDKNGX::procFrame(ID3D11Texture2D* frameDst, const NVEncNVSDKNGXRect *rectDst, ID3D11Texture2D* frameSrc, const NVEncNVSDKNGXRect *rectSrc, const int quality) {
    if (!m_pD3D11Device) {
        return RGY_ERR_NOT_INITIALIZED;
    }
    HRESULT hr = S_OK;
    bool useDstTmp = false;
    // check formats
    {
        // check input is DXGI_FORMAT_R8G8B8A8_UNORM or DXGI_FORMAT_B8G8R8A8_UNORM
        D3D11_TEXTURE2D_DESC inDesc = {};
        frameSrc->GetDesc(&inDesc);
        if (inDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && inDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
            return RGY_ERR_INVALID_FORMAT;
        }
        // verify input rect is within range
        if (rectSrc->left < 0 || rectSrc->left >= rectSrc->right || rectSrc->right  >(LONG)inDesc.Width
            || rectSrc->top  < 0 || rectSrc->top >= rectSrc->bottom || rectSrc->bottom >(LONG)inDesc.Height) {
            return RGY_ERR_INVALID_FORMAT;
        }
        // check output is DXGI_FORMAT_R8G8B8A8_UNORM or DXGI_FORMAT_B8G8R8A8_UNORM
        D3D11_TEXTURE2D_DESC outDesc = {};
        frameDst->GetDesc(&outDesc);
        if (outDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && outDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
            return RGY_ERR_INVALID_FORMAT;
        }
        // verify output rect is within range
        if (rectDst->left < 0 || rectDst->left >= rectDst->right || rectDst->right  >(LONG)outDesc.Width
            || rectDst->top  < 0 || rectDst->top >= rectDst->bottom || rectDst->bottom >(LONG)outDesc.Height) {
            return RGY_ERR_INVALID_FORMAT;
        }

        // The NGX dst surface must be created with BIND_UNORDERED_ACCESS, which swap buffers are not.
        // check for UNORDERED_ACCESS
        useDstTmp = !(outDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS);

        // verify DstTmp matches dest surface so copyRegion works
        if (useDstTmp && (!m_pDstTmpNGX || outDesc.Width != m_dstTmpWidth || outDesc.Height != m_dstTmpHeight)) {
            if (m_pDstTmpNGX) {
                m_pDstTmpNGX->Release();
                m_pDstTmpNGX = nullptr;
            }
            m_dstTmpWidth = outDesc.Width;
            m_dstTmpHeight = outDesc.Height;
            D3D11_TEXTURE2D_DESC texture2d_desc = { 0 };
            texture2d_desc.Width = m_dstTmpWidth;
            texture2d_desc.Height = m_dstTmpHeight;
            texture2d_desc.MipLevels = 1;
            texture2d_desc.ArraySize = 1;
            texture2d_desc.SampleDesc.Count = 1;
            texture2d_desc.MiscFlags = 0;
            texture2d_desc.Format = outDesc.Format;
            texture2d_desc.Usage = D3D11_USAGE_DEFAULT;
            texture2d_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

            hr = m_pD3D11Device->CreateTexture2D(&texture2d_desc, nullptr, &m_pDstTmpNGX);
            if (FAILED(hr)) return RGY_ERR_NULL_PTR;
        }
    }

    // setup VSR params
    NVSDK_NGX_D3D11_VSR_Eval_Params vsrEvalParams;
    vsrEvalParams.pInput = frameSrc;
    vsrEvalParams.pOutput = useDstTmp ? m_pDstTmpNGX : frameDst;
    vsrEvalParams.InputSubrectBase.X = rectSrc->left;
    vsrEvalParams.InputSubrectBase.Y = rectSrc->top;
    vsrEvalParams.InputSubrectSize.Width = rectSrc->right - rectSrc->left;
    vsrEvalParams.InputSubrectSize.Height = rectSrc->bottom - rectSrc->top;
    vsrEvalParams.OutputSubrectBase.X = rectDst->left;
    vsrEvalParams.OutputSubrectBase.Y = rectDst->top;
    vsrEvalParams.OutputSubrectSize.Width = rectDst->right - rectDst->left;
    vsrEvalParams.OutputSubrectSize.Height = rectDst->bottom - rectDst->top;
    vsrEvalParams.QualityLevel = (NVSDK_NGX_VSR_QualityLevel)quality;

    if (m_pMultiThread) {
        m_pMultiThread->Enter();
    }

    auto err = err_to_rgy(NGX_D3D11_EVALUATE_VSR_EXT(m_pD3D11DeviceContext, m_VSRFeature, m_ngxParameters, &vsrEvalParams));
    if (err != RGY_ERR_NONE) return err;

    if (m_pDstTmpNGX) {
        m_pD3D11DeviceContext->CopySubresourceRegion(frameDst, 0, 0, 0, 0, m_pDstTmpNGX, 0, nullptr);
    }

    if (m_pMultiThread) {
        m_pMultiThread->Leave();
    }

    return RGY_ERR_NONE;
}


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXCreate(NVEncNVSDKNGXHandle *ppNVSDKNGX) {
    if (ppNVSDKNGX == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    *ppNVSDKNGX = new NVEncNVSDKNGX();
    return RGY_ERR_NONE;
}

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXInit(NVEncNVSDKNGXHandle ppNVSDKNGX, ID3D11Device* pD3DDevice, ID3D11DeviceContext* pD3D11DeviceContext) {
    return ((NVEncNVSDKNGX *)ppNVSDKNGX)->init(pD3DDevice, pD3D11DeviceContext);
}

NVENC_NVSDKNGX_API void __stdcall NVEncNVSDKNGXDelete(NVEncNVSDKNGXHandle ppNVSDKNGX) {
    auto ptr = (NVEncNVSDKNGX *)ppNVSDKNGX;
    if (ptr) {
        delete ptr;
    }
}

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXProcFrame(NVEncNVSDKNGXHandle ppNVSDKNGX, ID3D11Texture2D* frameDst, const NVEncNVSDKNGXRect *rectDst, ID3D11Texture2D* frameSrc, const NVEncNVSDKNGXRect *rectSrc, const int quality) {
    return ((NVEncNVSDKNGX *)ppNVSDKNGX)->procFrame(frameDst, rectDst, frameSrc, rectSrc, quality);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // ENABLE_NVSDKNGX