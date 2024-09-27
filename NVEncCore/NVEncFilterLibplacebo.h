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

#ifndef __NVENC_FILTER_LIBPLACEBO_H__
#define __NVENC_FILTER_LIBPLACEBO_H__

#include <array>
#include "rgy_version.h"
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "NVEncFilterD3D11.h"

class NVEncFilterParamLibplacebo : public NVEncFilterParam {
public:
    DeviceDX11 *dx11;
    VideoVUIInfo vui;
    NVEncFilterParamLibplacebo() : dx11(nullptr), vui() {};
    virtual ~NVEncFilterParamLibplacebo() {};
};

class NVEncFilterParamLibplaceboResample : public NVEncFilterParamLibplacebo {
public:
    RGY_VPP_RESIZE_ALGO resize_algo;
    VppLibplaceboResample resample;
    NVEncFilterParamLibplaceboResample() : NVEncFilterParamLibplacebo(), resize_algo(RGY_VPP_RESIZE_AUTO), resample() {};
    virtual ~NVEncFilterParamLibplaceboResample() {};
    virtual tstring print() const override;
};

class NVEncFilterParamLibplaceboDeband : public NVEncFilterParamLibplacebo {
public:
    VppLibplaceboDeband deband;
    NVEncFilterParamLibplaceboDeband() : NVEncFilterParamLibplacebo(), deband() {};
    virtual ~NVEncFilterParamLibplaceboDeband() {};
    virtual tstring print() const override;
};

#if ENABLE_LIBPLACEBO

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4819)
#include <libplacebo/dispatch.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders.h>
#include <libplacebo/utils/upload.h>
#include <libplacebo/d3d11.h>
#pragma warning (pop)

template<typename T>
struct RGYLibplaceboDeleter {
    RGYLibplaceboDeleter() : deleter(nullptr) {};
    RGYLibplaceboDeleter(std::function<void(T*)> deleter) : deleter(deleter) {};
    void operator()(T p) { deleter(&p); }
    std::function<void(T*)> deleter;
};

struct RGYLibplaceboTexDeleter {
    RGYLibplaceboTexDeleter() : gpu(nullptr) {};
    RGYLibplaceboTexDeleter(pl_gpu gpu_) : gpu(gpu_) {};
    void operator()(pl_tex p) { if (p) pl_tex_destroy(gpu, &p); }
    pl_gpu gpu;
};

std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> rgy_pl_tex_recreate(pl_gpu gpu, const pl_tex_params& tex_params);

class NVEncFilterLibplacebo : public NVEncFilter {
public:
    NVEncFilterLibplacebo();
    virtual ~NVEncFilterLibplacebo();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR initCommon(shared_ptr<NVEncFilterParam> pParam);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) = 0;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) = 0;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) = 0;
    int getTextureBytePerPix(const DXGI_FORMAT format) const;
    virtual RGY_ERR initLibplacebo(const NVEncFilterParam *param);
    RGY_CSP getTextureCsp(const RGY_CSP csp);
    DXGI_FORMAT getTextureDXGIFormat(const RGY_CSP csp);

    RGY_CSP m_textCspIn;
    RGY_CSP m_textCspOut;
    DXGI_FORMAT m_dxgiformatIn;
    DXGI_FORMAT m_dxgiformatOut;

    std::unique_ptr<std::remove_pointer<pl_log>::type, RGYLibplaceboDeleter<pl_log>> m_log;
    std::unique_ptr<std::remove_pointer<pl_d3d11>::type, RGYLibplaceboDeleter<pl_d3d11>> m_d3d11;
    std::unique_ptr<std::remove_pointer<pl_dispatch>::type, RGYLibplaceboDeleter<pl_dispatch>> m_dispatch;
    std::unique_ptr<std::remove_pointer<pl_renderer>::type, RGYLibplaceboDeleter<pl_renderer>> m_renderer;
    std::unique_ptr<pl_shader_obj, decltype(&pl_shader_obj_destroy)> m_dither_state;

    std::unique_ptr<CUFrameBuf> m_textFrameBufOut;
    std::vector<std::unique_ptr<CUDADX11Texture>> m_textIn;
    std::vector<std::unique_ptr<CUDADX11Texture>> m_textOut;
    std::unique_ptr<NVEncFilter> m_srcCrop;
    std::unique_ptr<NVEncFilter> m_dstCrop;

    DeviceDX11 *m_dx11;
};

class NVEncFilterLibplaceboResample : public NVEncFilterLibplacebo {
public:
    NVEncFilterLibplaceboResample();
    virtual ~NVEncFilterLibplaceboResample();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override;

    std::unique_ptr<pl_sample_filter_params> m_filter_params;
};

class NVEncFilterLibplaceboDeband : public NVEncFilterLibplacebo {
public:
    NVEncFilterLibplaceboDeband();
    virtual ~NVEncFilterLibplaceboDeband();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override;

    std::unique_ptr<pl_deband_params> m_filter_params;
    std::unique_ptr<pl_deband_params> m_filter_params_c;
    std::unique_ptr<pl_dither_params> m_dither_params;
    int m_frame_index;
};

#else

class NVEncFilterLibplaceboResample : public NVEncFilterDisabled {
public:
    NVEncFilterLibplaceboResample();
    virtual ~NVEncFilterLibplaceboResample();
};

class NVEncFilterLibplaceboDeband : public NVEncFilterDisabled {
public:
    NVEncFilterLibplaceboDeband();
    virtual ~NVEncFilterLibplaceboDeband();
};

#endif

#endif //#ifndef __NVENC_FILTER_LIBPLACEBO_H__