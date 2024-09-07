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
    pTexture(nullptr),
    pSRView(nullptr),
    cudaResource(nullptr),
    cuArray(nullptr),
    width(0),
    height(0),
    offsetInShader(0) {
}

CUDADX11Texture::~CUDADX11Texture() {
    release();
}

RGY_ERR CUDADX11Texture::create(ID3D11Device* pD3DDevice, ID3D11DeviceContext* pD3DDeviceCtx, const int w, const int h, const DXGI_FORMAT dxgiformat) {
    this->width = w;
    this->height = h;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = dxgiformat;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.MiscFlags = 0;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;

    if (FAILED(pD3DDevice->CreateTexture2D(&desc, NULL, &pTexture))) {
        return RGY_ERR_NULL_PTR;
    }

    if (FAILED(pD3DDevice->CreateShaderResourceView(pTexture, NULL, &pSRView))) {
        return RGY_ERR_NULL_PTR;
    }

    offsetInShader = 0;  // to be clean we should look for the offset from the shader code
    pD3DDeviceCtx->PSSetShaderResources(offsetInShader, 1, &pSRView);
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::registerTexture() {
    auto sts = err_to_rgy(cudaGraphicsD3D11RegisterResource(&cudaResource, pTexture, cudaGraphicsRegisterFlagsNone));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::map() {
    auto sts = err_to_rgy(cudaGraphicsMapResources(1, &cudaResource, 0));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDADX11Texture::unmap() {
    auto sts = RGY_ERR_NONE;
    if (cuArray) {
        sts = err_to_rgy(cudaGraphicsUnmapResources(1, &cudaResource, 0));
        cuArray = nullptr;
    }
    return sts;
}

cudaArray *CUDADX11Texture::getMappedArray() {
    if (cuArray == nullptr) {
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
    }
    return cuArray;
}

RGY_ERR CUDADX11Texture::unregisterTexture() {
    auto sts = unmap();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (cudaResource) {
        sts = err_to_rgy(cudaGraphicsUnregisterResource(cudaResource));
        cudaResource = nullptr;
    }
    return sts;
}

RGY_ERR CUDADX11Texture::release() {
    unregisterTexture();
    if (pSRView) {
        pSRView->Release();
        pSRView = nullptr;
    }
    if (pTexture) {
        pTexture->Release();
        pTexture = nullptr;
    }
    return RGY_ERR_NONE;
}

#endif //#if ENABLE_D3D11
