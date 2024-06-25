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

#ifndef __NVENC_FILTER_NVSDK_NGX_H__
#define __NVENC_FILTER_NVSDK_NGX_H__

#include <array>
#include "rgy_version.h"
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "NVEncNVSDKNGX.h"

struct NVEncNVSDKNGXFuncs {
    HMODULE hModule;
    decltype(NVEncNVSDKNGXCreate) *fcreate;
    decltype(NVEncNVSDKNGXInit)   *finit;
    decltype(NVEncNVSDKNGXDelete) *fdelete;
    decltype(NVEncNVSDKNGXProcFrame) *fprocFrame;
    NVEncNVSDKNGXFuncs();
    ~NVEncNVSDKNGXFuncs();
    RGY_ERR load();
    void close();
};

class DeviceDX11;
struct ID3D11Texture2D;
struct ID3D11ShaderResourceView;
enum DXGI_FORMAT;

struct CUDADX11Texture {
public:
    ID3D11Texture2D *pTexture;
    ID3D11ShaderResourceView *pSRView;
    cudaGraphicsResource *cudaResource;
    cudaArray *cuArray;
    int width;
    int height;
    int offsetInShader;
    CUDADX11Texture();
    ~CUDADX11Texture();
    RGY_ERR create(ID3D11Device* pD3DDevice, ID3D11DeviceContext* pD3DDeviceCtx, const int width, const int height, const DXGI_FORMAT dxgiformat);
    RGY_ERR registerTexture();
    RGY_ERR map();
    RGY_ERR unmap();
    cudaArray *getMappedArray();
    RGY_ERR unregisterTexture();

    RGY_ERR release();
};

using unique_nvsdkngx_handle = std::unique_ptr<std::remove_pointer<NVEncNVSDKNGXHandle>::type, decltype(&NVEncNVSDKNGXDelete)>;

class NVEncFilterParamNGX : public NVEncFilterParam {
public:
    std::pair<int, int> compute_capability;
    DeviceDX11 *dx11;
    VideoVUIInfo vui;
    NVEncFilterParamNGX() : compute_capability(), dx11(nullptr), vui() {};
    virtual ~NVEncFilterParamNGX() {};
};

class NVEncFilterParamNGXVSR : public NVEncFilterParamNGX {
public:
    VppNGXVSR ngxvsr;
    NVEncFilterParamNGXVSR() : NVEncFilterParamNGX(), ngxvsr() {};
    virtual ~NVEncFilterParamNGXVSR() {};
    virtual tstring print() const;
};

class NVEncFilterParamNGXTrueHDR : public NVEncFilterParamNGX {
public:
    VppNGXTrueHDR trueHDR;
    NVEncFilterParamNGXTrueHDR() : NVEncFilterParamNGX(), trueHDR() {};
    virtual ~NVEncFilterParamNGXTrueHDR() {};
    virtual tstring print() const;
};

class NVEncFilterNGX : public NVEncFilter {
public:
    NVEncFilterNGX();
    virtual ~NVEncFilterNGX();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR initNGX(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes);
    virtual RGY_ERR initCommon(shared_ptr<NVEncFilterParam> pParam);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) = 0;
    virtual void setNGXParam(const NVEncFilterParam *param) = 0;
    virtual NVEncNVSDKNGXParam *getNGXParam() = 0;
    virtual NVEncNVSDKNGXFeature getNGXFeature() = 0;
    int getTextureBytePerPix(const DXGI_FORMAT format) const;

    std::unique_ptr<NVEncNVSDKNGXFuncs> m_func;
    unique_nvsdkngx_handle m_nvsdkNGX;
    RGY_CSP m_ngxCspIn;
    RGY_CSP m_ngxCspOut;
    DXGI_FORMAT m_dxgiformatIn;
    DXGI_FORMAT m_dxgiformatOut;
    std::unique_ptr<CUFrameBuf> m_ngxFrameBufOut;
    std::unique_ptr<CUDADX11Texture> m_ngxTextIn;
    std::unique_ptr<CUDADX11Texture> m_ngxTextOut;
    std::unique_ptr<NVEncFilter> m_srcColorspace;
    std::unique_ptr<NVEncFilter> m_dstColorspace;
    std::unique_ptr<NVEncFilter> m_srcCrop;
    std::unique_ptr<NVEncFilter> m_dstCrop;

    DeviceDX11 *m_dx11;
};

class NVEncFilterNGXVSR : public NVEncFilterNGX {
public:
    NVEncFilterNGXVSR();
    virtual ~NVEncFilterNGXVSR();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual void setNGXParam(const NVEncFilterParam *param) override;
    virtual NVEncNVSDKNGXParam *getNGXParam() override { return (NVEncNVSDKNGXParam *)&m_paramVSR; }
    virtual NVEncNVSDKNGXFeature getNGXFeature() override { return NVSDK_NVX_VSR; }

    NVEncNVSDKNGXParamVSR m_paramVSR;
};

class NVEncFilterNGXTrueHDR : public NVEncFilterNGX {
public:
    NVEncFilterNGXTrueHDR();
    virtual ~NVEncFilterNGXTrueHDR();
    VideoVUIInfo VuiOut() const { return m_vuiOut; }
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual void setNGXParam(const NVEncFilterParam *param) override;
    virtual NVEncNVSDKNGXParam *getNGXParam() override { return (NVEncNVSDKNGXParam *)&m_paramTrueHDR; }
    virtual NVEncNVSDKNGXFeature getNGXFeature() override { return NVSDK_NVX_TRUEHDR; }

    VideoVUIInfo m_vuiOut;
    NVEncNVSDKNGXParamTrueHDR m_paramTrueHDR;
};

#endif //#ifndef __NVENC_FILTER_NV_OPT_FLOW_H__