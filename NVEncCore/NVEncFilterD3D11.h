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

#ifndef __NVENC_FILTER_D3D11_H__
#define __NVENC_FILTER_D3D11_H__

#include <array>
#include "rgy_version.h"
#include "convert_csp.h"

class DeviceDX11;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct ID3D11ShaderResourceView;

#if ENABLE_D3D11
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

#endif //#if ENABLE_D3D11

#endif //#ifndef __NVENC_FILTER_D3D11_H__