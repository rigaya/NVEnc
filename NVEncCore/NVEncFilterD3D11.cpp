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

#include <array>
#include <numeric>
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterD3D11.h"
#include "rgy_device.h"
#if ENABLE_D3D11
#include <cuda_d3d11_interop.h>

CUDADX11Texture::CUDADX11Texture() :
    m_pTexture(nullptr),
    m_pSRView(nullptr),
    m_cudaResource(nullptr),
    m_cuArray(nullptr),
    m_width(0),
    m_height(0),
    m_dxgiFormat(DXGI_FORMAT_UNKNOWN),
    m_offsetInShader(0) {
}

CUDADX11Texture::~CUDADX11Texture() {
    release();
}

RGY_ERR CUDADX11Texture::create(DeviceDX11 *dx11, const int w, const int h, const DXGI_FORMAT format) {
    m_dxgiFormat = format;
    m_width = w;
    m_height = h;
    m_pitch = w * getTextureBytePerPix();

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.MiscFlags = 0;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;

    auto pD3DDevice = dx11->GetDevice();
    if (FAILED(pD3DDevice->CreateTexture2D(&desc, NULL, &m_pTexture))) {
        return RGY_ERR_NULL_PTR;
    }

    if (FAILED(pD3DDevice->CreateShaderResourceView(m_pTexture, NULL, &m_pSRView))) {
        return RGY_ERR_NULL_PTR;
    }

    auto pD3DDeviceCtx = dx11->GetDeviceContext();
    m_offsetInShader = 0;  // to be clean we should look for the offset from the shader code
    pD3DDeviceCtx->PSSetShaderResources(m_offsetInShader, 1, &m_pSRView);
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::registerTexture() {
    auto sts = err_to_rgy(cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_pTexture, cudaGraphicsRegisterFlagsNone));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::map() {
    auto sts = err_to_rgy(cudaGraphicsMapResources(1, &m_cudaResource, 0));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::unmap() {
    auto sts = RGY_ERR_NONE;
    if (m_cuArray) {
        sts = err_to_rgy(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
        m_cuArray = nullptr;
    }
    return sts;
}

cudaArray *CUDADX11Texture::getMappedArray() {
    if (m_cuArray == nullptr) {
        cudaGraphicsSubResourceGetMappedArray(&m_cuArray, m_cudaResource, 0, 0);
    }
    return m_cuArray;
}

RGY_ERR CUDADX11Texture::unregisterTexture() {
    auto sts = unmap();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (m_cudaResource) {
        sts = err_to_rgy(cudaGraphicsUnregisterResource(m_cudaResource));
        m_cudaResource = nullptr;
    }
    return sts;
}

RGY_ERR CUDADX11Texture::release() {
    unregisterTexture();
    if (m_pSRView) {
        m_pSRView->Release();
        m_pSRView = nullptr;
    }
    if (m_pTexture) {
        m_pTexture->Release();
        m_pTexture = nullptr;
    }
    return RGY_ERR_NONE;
}

DXGI_FORMAT CUDADX11Texture::getTextureDXGIFormat() const { return m_dxgiFormat; }

int CUDADX11Texture::getTextureBytePerPix() const {
    switch (m_dxgiFormat) {
    case DXGI_FORMAT_R8_UNORM:
        return 1;
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return 8;
    default:
        return 0;
    }
}

#endif //#if ENABLE_D3D11
