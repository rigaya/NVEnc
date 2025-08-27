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

#include <cuda.h>
#include <cuda_runtime.h>

#define APP_ID      0
#define APP_PATH    L"."

class NVEncNVSDKNGX {
public:
    NVEncNVSDKNGX();
    virtual ~NVEncNVSDKNGX();

    virtual RGY_ERR init(int cudaDeviceOrdinal, CUcontext cuContextExt, CUstream cuStreamExt) = 0;
    virtual void close();
    virtual RGY_ERR procFrame(const NVEncNVSDKNGXRect *rectDst, const NVEncNVSDKNGXRect *rectSrc, const NVEncNVSDKNGXParam *param,
        const void *srcDevPtr, int srcPitch, void *dstDevPtr, int dstPitch, int srcBytesPerPix, int dstBytesPerPix) = 0;
protected:
    NVSDK_NGX_Parameter*        m_ngxParameters;
    NVSDK_NGX_Handle*           m_ngxFeature;

    CUcontext                   m_cuContext;
    CUdevice                    m_cuDevice;
    bool                        m_ownPrimaryCtx;
    CUstream                    m_cuStream;
    // reusable CUDA arrays/objects
    CUarray                     m_cuArraySrc;
    CUtexObject                 m_cuTexObjectSrc;
    size_t                      m_srcArrayWidth;
    size_t                      m_srcArrayHeight;
    CUarray                     m_cuArrayDst;
    CUsurfObject                m_cuSurfObjectDst;
    size_t                      m_dstArrayWidth;
    size_t                      m_dstArrayHeight;
};

class NVEncNVSDKNGXVSR : public NVEncNVSDKNGX {
public:
    NVEncNVSDKNGXVSR();
    virtual ~NVEncNVSDKNGXVSR();

    virtual RGY_ERR init(int cudaDeviceOrdinal, CUcontext cuContextExt, CUstream cuStreamExt) override;
    virtual RGY_ERR procFrame(const NVEncNVSDKNGXRect *rectDst, const NVEncNVSDKNGXRect *rectSrc, const NVEncNVSDKNGXParam *param,
        const void *srcDevPtr, int srcPitch, void *dstDevPtr, int dstPitch, int srcBytesPerPix, int dstBytesPerPix) override;
protected:
};

class NVEncNVSDKNGXTrueHDR : public NVEncNVSDKNGX {
public:
    NVEncNVSDKNGXTrueHDR();
    virtual ~NVEncNVSDKNGXTrueHDR();

    virtual RGY_ERR init(int cudaDeviceOrdinal, CUcontext cuContextExt, CUstream cuStreamExt) override;
    virtual RGY_ERR procFrame(const NVEncNVSDKNGXRect *rectDst, const NVEncNVSDKNGXRect *rectSrc, const NVEncNVSDKNGXParam *param,
        const void *srcDevPtr, int srcPitch, void *dstDevPtr, int dstPitch, int srcBytesPerPix, int dstBytesPerPix) override;
protected:
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
    m_ngxParameters(nullptr),
    m_ngxFeature(nullptr),
    m_cuContext(nullptr),
    m_cuDevice(0),
    m_ownPrimaryCtx(false),
    m_cuStream(nullptr),
    m_cuArraySrc(nullptr),
    m_cuTexObjectSrc(0),
    m_srcArrayWidth(0),
    m_srcArrayHeight(0),
    m_cuArrayDst(nullptr),
    m_cuSurfObjectDst(0),
    m_dstArrayWidth(0),
    m_dstArrayHeight(0) {
}

NVEncNVSDKNGX::~NVEncNVSDKNGX() {
    close();
}

void NVEncNVSDKNGX::close() {
    if (m_ngxFeature) {
        NVSDK_NGX_CUDA_ReleaseFeature(m_ngxFeature);
        m_ngxFeature = nullptr;
    }
    if (m_cuTexObjectSrc) {
        cuTexObjectDestroy(m_cuTexObjectSrc);
        m_cuTexObjectSrc = 0;
    }
    if (m_cuArraySrc) {
        cuArrayDestroy(m_cuArraySrc);
        m_cuArraySrc = nullptr;
    }
    if (m_cuSurfObjectDst) {
        cuSurfObjectDestroy(m_cuSurfObjectDst);
        m_cuSurfObjectDst = 0;
    }
    if (m_cuArrayDst) {
        cuArrayDestroy(m_cuArrayDst);
        m_cuArrayDst = nullptr;
    }
    if (m_ngxParameters) {
        NVSDK_NGX_CUDA_DestroyParameters(m_ngxParameters);
        m_ngxParameters = nullptr;
    }
    NVSDK_NGX_CUDA_Shutdown();
    if (m_ownPrimaryCtx && m_cuContext) {
        cuDevicePrimaryCtxRelease(m_cuDevice);
        m_cuContext = nullptr;
        m_ownPrimaryCtx = false;
    }
}

NVEncNVSDKNGXVSR::NVEncNVSDKNGXVSR() : NVEncNVSDKNGX() { }
NVEncNVSDKNGXVSR::~NVEncNVSDKNGXVSR() { }

RGY_ERR NVEncNVSDKNGXVSR::init(int cudaDeviceOrdinal, CUcontext cuContextExt, CUstream cuStreamExt) {
    m_cuContext = cuContextExt;
    m_cuStream = cuStreamExt;
    m_ownPrimaryCtx = false;
    if (m_cuContext == nullptr) {
        // 現在アクティブなコンテキストを取得
        CUresult result = cuCtxGetCurrent(&m_cuContext);
        if (result == CUDA_SUCCESS && m_cuContext != nullptr) {
            // 現在のコンテキストを使用
            CUdevice dev;
            cuCtxGetDevice(&dev);
            m_cuDevice = dev;
        } else {
            // 最後の手段としてプライマリコンテキストを使用
            if (cudaDeviceOrdinal >= 0) {
                cuDeviceGet(&m_cuDevice, cudaDeviceOrdinal);
            } else {
                int devOrdinal = 0;
                cudaGetDevice(&devOrdinal);
                cuDeviceGet(&m_cuDevice, devOrdinal);
            }
            cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice);
            m_ownPrimaryCtx = true;
        }
    } else {
        CUdevice dev;
        cuCtxGetDevice(&dev);
        m_cuDevice = dev;
    }

    auto err = err_to_rgy(NVSDK_NGX_CUDA_Init(APP_ID, APP_PATH));
    if (err != RGY_ERR_NONE) return err;

    err = err_to_rgy(NVSDK_NGX_CUDA_GetCapabilityParameters(&m_ngxParameters));
    if (err != RGY_ERR_NONE) return err;

    int VSRAvailable = 0;
    err = err_to_rgy(m_ngxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &VSRAvailable));
    if (err != RGY_ERR_NONE) return err;

    NVSDK_NGX_CUDA_VSR_Create_Params VSRCreateParams = {};
    VSRCreateParams.InCUContext = m_cuContext;
    VSRCreateParams.InCUStream = m_cuStream;
    err = err_to_rgy(NGX_CUDA_CREATE_VSR(&m_ngxFeature, m_ngxParameters, &VSRCreateParams));
    if (err != RGY_ERR_NONE) return err;

    return RGY_ERR_NONE;
}

RGY_ERR NVEncNVSDKNGXVSR::procFrame(const NVEncNVSDKNGXRect *rectDst, const NVEncNVSDKNGXRect *rectSrc, const NVEncNVSDKNGXParam *param,
    const void *srcDevPtr, int srcPitch, void *dstDevPtr, int dstPitch, int srcBytesPerPix, int dstBytesPerPix) {
    if (!m_ngxFeature) {
        return RGY_ERR_NOT_INITIALIZED;
    }
    // prepare reusable arrays/objects
    CUresult drvres;
    size_t srcW = rectSrc->right - rectSrc->left;
    size_t srcH = rectSrc->bottom - rectSrc->top;
    size_t dstW = rectDst->right - rectDst->left;
    size_t dstH = rectDst->bottom - rectDst->top;
    if (!m_cuArraySrc || m_srcArrayWidth != srcW || m_srcArrayHeight != srcH) {
        if (m_cuTexObjectSrc) { cuTexObjectDestroy(m_cuTexObjectSrc); m_cuTexObjectSrc = 0; }
        if (m_cuArraySrc) { cuArrayDestroy(m_cuArraySrc); m_cuArraySrc = nullptr; }
        CUDA_ARRAY_DESCRIPTOR ad{}; ad.Width = srcW; ad.Height = srcH; ad.NumChannels = 4; ad.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        drvres = cuArrayCreate(&m_cuArraySrc, &ad); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = m_cuArraySrc;
        CUDA_TEXTURE_DESC td{}; td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP; td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP; td.filterMode = CU_TR_FILTER_MODE_LINEAR; td.flags = CU_TRSF_NORMALIZED_COORDINATES;
        drvres = cuTexObjectCreate(&m_cuTexObjectSrc, &rd, &td, nullptr); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        m_srcArrayWidth = srcW; m_srcArrayHeight = srcH;
    }
    if (!m_cuArrayDst || m_dstArrayWidth != dstW || m_dstArrayHeight != dstH) {
        if (m_cuSurfObjectDst) { cuSurfObjectDestroy(m_cuSurfObjectDst); m_cuSurfObjectDst = 0; }
        if (m_cuArrayDst) { cuArrayDestroy(m_cuArrayDst); m_cuArrayDst = nullptr; }
        CUDA_ARRAY_DESCRIPTOR ad{}; ad.Width = dstW; ad.Height = dstH; ad.NumChannels = 4; ad.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        drvres = cuArrayCreate(&m_cuArrayDst, &ad); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = m_cuArrayDst;
        drvres = cuSurfObjectCreate(&m_cuSurfObjectDst, &rd); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        m_dstArrayWidth = dstW; m_dstArrayHeight = dstH;
    }

    // copy input from device pointer to array
    cudaMemcpy2DToArray((cudaArray_t)m_cuArraySrc, 0, 0, srcDevPtr, srcPitch, srcW * srcBytesPerPix, srcH, cudaMemcpyDeviceToDevice);

    const NVEncNVSDKNGXParamVSR *vsrParam = (const NVEncNVSDKNGXParamVSR *)param;

    NVSDK_NGX_CUDA_VSR_Eval_Params evalParams = {};
    evalParams.pInput = (CUtexObject *)&m_cuTexObjectSrc;
    evalParams.pOutput = (CUsurfObject *)&m_cuSurfObjectDst;
    evalParams.InputSubrectBase.X = rectSrc->left;
    evalParams.InputSubrectBase.Y = rectSrc->top;
    evalParams.InputSubrectSize.Width = rectSrc->right - rectSrc->left;
    evalParams.InputSubrectSize.Height = rectSrc->bottom - rectSrc->top;
    evalParams.OutputSubrectBase.X = rectDst->left;
    evalParams.OutputSubrectBase.Y = rectDst->top;
    evalParams.OutputSubrectSize.Width = rectDst->right - rectDst->left;
    evalParams.OutputSubrectSize.Height = rectDst->bottom - rectDst->top;
    evalParams.QualityLevel = (NVSDK_NGX_VSR_QualityLevel)vsrParam->quality;

    auto err = err_to_rgy(NGX_CUDA_EVALUATE_VSR(m_ngxFeature, m_ngxParameters, &evalParams));

    // copy output from array to device pointer
    if (err == RGY_ERR_NONE) {
        size_t widthBytes = (rectDst->right - rectDst->left) * dstBytesPerPix;
        size_t height = (rectDst->bottom - rectDst->top);
        cudaMemcpy2DFromArray(dstDevPtr, dstPitch, (cudaArray_t)m_cuArrayDst, 0, 0, widthBytes, height, cudaMemcpyDeviceToDevice);
    }

    if (err != RGY_ERR_NONE) return err;
    return RGY_ERR_NONE;
}


NVEncNVSDKNGXTrueHDR::NVEncNVSDKNGXTrueHDR() : NVEncNVSDKNGX() { }
NVEncNVSDKNGXTrueHDR::~NVEncNVSDKNGXTrueHDR() { }

RGY_ERR NVEncNVSDKNGXTrueHDR::init(int cudaDeviceOrdinal, CUcontext cuContextExt, CUstream cuStreamExt) {
    m_cuContext = cuContextExt;
    m_cuStream = cuStreamExt;

    m_ownPrimaryCtx = false;
    if (m_cuContext == nullptr) {
        // 現在アクティブなコンテキストを取得
        CUresult result = cuCtxGetCurrent(&m_cuContext);
        if (result == CUDA_SUCCESS && m_cuContext != nullptr) {
            // 現在のコンテキストを使用
            CUdevice dev;
            cuCtxGetDevice(&dev);
            m_cuDevice = dev;
        } else {
            // 最後の手段としてプライマリコンテキストを使用
            if (cudaDeviceOrdinal >= 0) {
                cuDeviceGet(&m_cuDevice, cudaDeviceOrdinal);
            } else {
                int devOrdinal = 0;
                cudaGetDevice(&devOrdinal);
                cuDeviceGet(&m_cuDevice, devOrdinal);
            }
            cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice);
            m_ownPrimaryCtx = true;
        }
    } else {
        CUdevice dev;
        cuCtxGetDevice(&dev);
        m_cuDevice = dev;
    }

    auto err = err_to_rgy(NVSDK_NGX_CUDA_Init(APP_ID, APP_PATH));
    if (err != RGY_ERR_NONE) return err;

    err = err_to_rgy(NVSDK_NGX_CUDA_GetCapabilityParameters(&m_ngxParameters));
    if (err != RGY_ERR_NONE) return err;

    int TrueHDRAvailable = 0;
    err = err_to_rgy(m_ngxParameters->Get(NVSDK_NGX_Parameter_TrueHDR_Available, &TrueHDRAvailable));
    if (err != RGY_ERR_NONE) return err;

    NVSDK_NGX_CUDA_TRUEHDR_Create_Params TrueHDRCreateParams = {};
    TrueHDRCreateParams.InCUContext = m_cuContext;
    TrueHDRCreateParams.InCUStream = m_cuStream;
    err = err_to_rgy(NGX_CUDA_CREATE_TRUEHDR(&m_ngxFeature, m_ngxParameters, &TrueHDRCreateParams));
    if (err != RGY_ERR_NONE) return err;

    return RGY_ERR_NONE;
}

RGY_ERR NVEncNVSDKNGXTrueHDR::procFrame(const NVEncNVSDKNGXRect *rectDst, const NVEncNVSDKNGXRect *rectSrc, const NVEncNVSDKNGXParam *param,
    const void *srcDevPtr, int srcPitch, void *dstDevPtr, int dstPitch, int srcBytesPerPix, int dstBytesPerPix) {
    if (!m_ngxFeature) {
        return RGY_ERR_NOT_INITIALIZED;
    }
    // prepare reusable arrays/objects
    CUresult drvres;
    size_t srcW = rectSrc->right - rectSrc->left;
    size_t srcH = rectSrc->bottom - rectSrc->top;
    size_t dstW = rectDst->right - rectDst->left;
    size_t dstH = rectDst->bottom - rectDst->top;
    if (!m_cuArraySrc || m_srcArrayWidth != srcW || m_srcArrayHeight != srcH) {
        if (m_cuTexObjectSrc) { cuTexObjectDestroy(m_cuTexObjectSrc); m_cuTexObjectSrc = 0; }
        if (m_cuArraySrc) { cuArrayDestroy(m_cuArraySrc); m_cuArraySrc = nullptr; }
        CUDA_ARRAY_DESCRIPTOR ad{}; ad.Width = srcW; ad.Height = srcH; ad.NumChannels = 4; ad.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        drvres = cuArrayCreate(&m_cuArraySrc, &ad); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = m_cuArraySrc;
        CUDA_TEXTURE_DESC td{}; td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP; td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP; td.filterMode = CU_TR_FILTER_MODE_LINEAR; td.flags = CU_TRSF_NORMALIZED_COORDINATES;
        drvres = cuTexObjectCreate(&m_cuTexObjectSrc, &rd, &td, nullptr); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        m_srcArrayWidth = srcW; m_srcArrayHeight = srcH;
    }
    if (!m_cuArrayDst || m_dstArrayWidth != dstW || m_dstArrayHeight != dstH) {
        if (m_cuSurfObjectDst) { cuSurfObjectDestroy(m_cuSurfObjectDst); m_cuSurfObjectDst = 0; }
        if (m_cuArrayDst) { cuArrayDestroy(m_cuArrayDst); m_cuArrayDst = nullptr; }
        // 出力配列のフォーマット: 10bit(A2R10G10B10)は4Bpp, 16Fは8Bpp。
        CUarray_format outFmt = (dstBytesPerPix == 8) ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR ad{}; ad.Width = dstW; ad.Height = dstH; ad.NumChannels = 4; ad.Format = outFmt;
        drvres = cuArrayCreate(&m_cuArrayDst, &ad); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        CUDA_RESOURCE_DESC rd{}; rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = m_cuArrayDst;
        drvres = cuSurfObjectCreate(&m_cuSurfObjectDst, &rd); if (drvres != CUDA_SUCCESS) return RGY_ERR_NULL_PTR;
        m_dstArrayWidth = dstW; m_dstArrayHeight = dstH;
    }

    // copy input from device pointer to array (input is RGBA8)
    cudaMemcpy2DToArray((cudaArray_t)m_cuArraySrc, 0, 0, srcDevPtr, srcPitch, srcW * srcBytesPerPix, srcH, cudaMemcpyDeviceToDevice);

    const auto truehdrParam = (const NVEncNVSDKNGXParamTrueHDR *)param;

    NVSDK_NGX_CUDA_TRUEHDR_Eval_Params evalParams = {};
    evalParams.pInput = (CUtexObject *)&m_cuTexObjectSrc;
    evalParams.pOutput = (CUsurfObject *)&m_cuSurfObjectDst;
    evalParams.InputSubrectTL.X = rectSrc->left;
    evalParams.InputSubrectTL.Y = rectSrc->top;
    evalParams.InputSubrectBR.Width = rectSrc->right;
    evalParams.InputSubrectBR.Height = rectSrc->bottom;
    evalParams.OutputSubrectTL.X = rectDst->left;
    evalParams.OutputSubrectTL.Y = rectDst->top;
    evalParams.OutputSubrectBR.Width = rectDst->right;
    evalParams.OutputSubrectBR.Height = rectDst->bottom;
    evalParams.Contrast = truehdrParam->contrast;
    evalParams.Saturation = truehdrParam->saturation;
    evalParams.MiddleGray = truehdrParam->middleGray;
    evalParams.MaxLuminance = truehdrParam->maxLuminance;

    auto err = err_to_rgy(NGX_CUDA_EVALUATE_TRUEHDR(m_ngxFeature, m_ngxParameters, &evalParams));

    // copy output from array to device pointer
    if (err == RGY_ERR_NONE) {
        size_t widthBytes = (rectDst->right - rectDst->left) * dstBytesPerPix;
        size_t height = (rectDst->bottom - rectDst->top);
        cudaMemcpy2DFromArray(dstDevPtr, dstPitch, (cudaArray_t)m_cuArrayDst, 0, 0, widthBytes, height, cudaMemcpyDeviceToDevice);
    }

    if (err != RGY_ERR_NONE) return err;
    return RGY_ERR_NONE;
}

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXCreate(NVEncNVSDKNGXHandle *ppNVSDKNGX, const NVEncNVSDKNGXFeature feature) {
    if (ppNVSDKNGX == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    switch (feature)
    {
    case NVSDK_NVX_VSR:
        *ppNVSDKNGX = new NVEncNVSDKNGXVSR();
        break;
    case NVSDK_NVX_TRUEHDR:
        *ppNVSDKNGX = new NVEncNVSDKNGXTrueHDR();
        break;
    default:
        break;
    }
    return RGY_ERR_NONE;
}

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXInit(NVEncNVSDKNGXHandle ppNVSDKNGX, int cudaDeviceOrdinal, void *cuContext, void *cuStream) {
    return ((NVEncNVSDKNGX *)ppNVSDKNGX)->init(cudaDeviceOrdinal, (CUcontext)cuContext, (CUstream)cuStream);
}

NVENC_NVSDKNGX_API void __stdcall NVEncNVSDKNGXDelete(NVEncNVSDKNGXHandle ppNVSDKNGX) {
    auto ptr = (NVEncNVSDKNGX *)ppNVSDKNGX;
    if (ptr) {
        delete ptr;
    }
}

NVENC_NVSDKNGX_API RGY_ERR __stdcall NVEncNVSDKNGXProcFrame(NVEncNVSDKNGXHandle ppNVSDKNGX,
    const NVEncNVSDKNGXRect *rectDst,
    const NVEncNVSDKNGXRect *rectSrc,
    const NVEncNVSDKNGXParam *param,
    const void *srcDevPtr, int srcPitch,
    void *dstDevPtr, int dstPitch,
    int srcBytesPerPix, int dstBytesPerPix) {
    return ((NVEncNVSDKNGX *)ppNVSDKNGX)->procFrame(rectDst, rectSrc, param, srcDevPtr, srcPitch, dstDevPtr, dstPitch, srcBytesPerPix, dstBytesPerPix);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // ENABLE_NVSDKNGX