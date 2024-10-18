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
#include "rgy_bitstream.h"
#include "convert_csp.h"
#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "NVEncFilterD3D11.h"
#include "NVEncFilterVulkan.h"

class NVEncFilterParamLibplacebo : public NVEncFilterParam {
public:
    DeviceDX11 *dx11;
    DeviceVulkan *vk;
    VideoVUIInfo vui;
    NVEncFilterParamLibplacebo() : dx11(nullptr), vk(nullptr), vui() {};
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

class NVEncFilterParamLibplaceboToneMapping : public NVEncFilterParamLibplacebo {
public:
    VppLibplaceboToneMapping toneMapping;
    VideoVUIInfo vui;
    const RGYHDRMetadata *hdrMetadataIn;
    const RGYHDRMetadata *hdrMetadataOut;
    NVEncFilterParamLibplaceboToneMapping() : NVEncFilterParamLibplacebo(), toneMapping(), vui(), hdrMetadataIn(nullptr), hdrMetadataOut(nullptr) {};
    virtual ~NVEncFilterParamLibplaceboToneMapping() {};
    virtual tstring print() const override;
};

#if ENABLE_LIBPLACEBO

#include "rgy_libplacebo.h"

#if ENABLE_D3D11
using pl_device = pl_d3d11;
using PLDevice = DeviceDX11;
#define p_tex_wrap p_d3d11_wrap
using pl_tex_wrap_params = pl_d3d11_wrap_params;
using CUDAInteropTexture = CUDADX11Texture;
using CUDAInteropDataFormat = DXGI_FORMAT;
static const TCHAR *RGY_LIBPLACEBO_DEV_API = _T("d3d11");
#elif ENABLE_VULKAN
using pl_device = pl_vulkan;
using PLDevice = DeviceVulkan;
#define p_tex_wrap p_vulkan_wrap
using pl_tex_wrap_params = pl_vulkan_wrap_params;
using CUDAInteropTexture = CUDAVulkanFrame;
using CUDAInteropDataFormat = VkFormat;
static const TCHAR *RGY_LIBPLACEBO_DEV_API = _T("vulkan");
#endif

class NVEncFilterLibplacebo : public NVEncFilter {
public:
    NVEncFilterLibplacebo();
    virtual ~NVEncFilterLibplacebo();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR initCommon(shared_ptr<NVEncFilterParam> pParam);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual RGY_ERR setFrameParam(const RGYFrameInfo *pInputFrame) { UNREFERENCED_PARAMETER(pInputFrame); return RGY_ERR_NONE; }
    virtual void close() override;
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) = 0;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) = 0;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) = 0;
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) = 0;
    virtual RGY_ERR initLibplacebo(const NVEncFilterParam *param);
    virtual RGY_CSP getTextureCsp(const RGY_CSP csp);
    virtual CUDAInteropDataFormat getTextureDataFormat([[maybe_unused]] const RGY_CSP csp);
    virtual tstring printParams(const NVEncFilterParamLibplacebo *prm) const;
    virtual void setFrameProp(RGYFrameInfo *pFrame, const RGYFrameInfo *pSrcFrame) const { copyFramePropWithoutRes(pFrame, pSrcFrame); }

    bool m_procByFrame;

    RGY_CSP m_textCspIn;
    RGY_CSP m_textCspOut;
    CUDAInteropDataFormat m_dataformatIn;
    CUDAInteropDataFormat m_dataformatOut;

    std::unique_ptr<std::remove_pointer<pl_log>::type, RGYLibplaceboDeleter<pl_log>> m_log;
    std::unique_ptr<std::remove_pointer<pl_device>::type, RGYLibplaceboDeleter<pl_device>> m_pldevice;
    std::unique_ptr<std::remove_pointer<pl_dispatch>::type, RGYLibplaceboDeleter<pl_dispatch>> m_dispatch;
    std::unique_ptr<std::remove_pointer<pl_renderer>::type, RGYLibplaceboDeleter<pl_renderer>> m_renderer;
    std::unique_ptr<pl_shader_obj, decltype(&pl_shader_obj_destroy)> m_dither_state;

    std::unique_ptr<CUFrameBuf> m_textFrameBufOut;
    std::vector<std::unique_ptr<CUDAInteropTexture>> m_textIn;
    std::vector<std::unique_ptr<CUDAInteropTexture>> m_textOut;
#if ENABLE_VULKAN
    std::vector<std::unique_ptr<CUDAVulkanSemaphore>> m_semInVKWait;
    std::vector<std::unique_ptr<CUDAVulkanSemaphore>> m_semInVKStart;
    std::vector<std::unique_ptr<CUDAVulkanSemaphore>> m_semOutVKWait;
    std::vector<std::unique_ptr<CUDAVulkanSemaphore>> m_semOutVKStart;
#endif
    std::unique_ptr<NVEncFilter> m_srcCrop;
    std::unique_ptr<NVEncFilter> m_dstCrop;
    std::unique_ptr<RGYLibplaceboLoader> m_pl;

    PLDevice *m_device;
};

class NVEncFilterLibplaceboResample : public NVEncFilterLibplacebo {
public:
    NVEncFilterLibplaceboResample();
    virtual ~NVEncFilterLibplaceboResample();
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override;
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstFrame); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcFrame);
        return RGY_ERR_UNSUPPORTED;
    };

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
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstFrame); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcFrame);
        return RGY_ERR_UNSUPPORTED;
    };

    std::unique_ptr<pl_deband_params> m_filter_params;
    std::unique_ptr<pl_deband_params> m_filter_params_c;
    std::unique_ptr<pl_dither_params> m_dither_params;
    int m_frame_index;
};

struct RGYLibplaceboToneMappingParams {
    std::unique_ptr<pl_render_params> renderParams;
    VppLibplaceboToneMappingCSP cspSrc;
    VppLibplaceboToneMappingCSP cspDst;
    pl_color_space plCspSrc;
    pl_color_space plCspDst;
    float src_max_org;
    float src_min_org;
    float dst_max_org;
    float dst_min_org;
    bool is_subsampled;
    pl_chroma_location chromaLocation;
    bool use_dovi;
    std::unique_ptr<pl_color_map_params> colorMapParams;
    std::unique_ptr<pl_peak_detect_params> peakDetectParams;
    std::unique_ptr<pl_dovi_metadata> plDoviMeta;
    std::unique_ptr<pl_color_repr> reprSrc;
    std::unique_ptr<pl_color_repr> reprDst;
    VideoVUIInfo outVui;
};

class NVEncFilterLibplaceboToneMapping : public NVEncFilterLibplacebo {
public:
    NVEncFilterLibplaceboToneMapping();
    virtual ~NVEncFilterLibplaceboToneMapping();
    VideoVUIInfo VuiOut() const;
protected:
    virtual RGY_ERR checkParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const NVEncFilterParam *param) override;
    virtual RGY_ERR setFrameParam(const RGYFrameInfo *pInputFrame) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstPlane); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcPlane); UNREFERENCED_PARAMETER(planeIdx);
        return RGY_ERR_UNSUPPORTED;
    };
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override;

    virtual RGY_CSP getTextureCsp(const RGY_CSP csp) override;
    virtual CUDAInteropDataFormat getTextureDataFormat([[maybe_unused]] const RGY_CSP csp) override;
    virtual tstring printParams(const NVEncFilterParamLibplacebo *prm) const override;
    virtual void setFrameProp(RGYFrameInfo *pFrame, const RGYFrameInfo *pSrcFrame) const override;

    RGYLibplaceboToneMappingParams m_tonemap;
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

class NVEncFilterLibplaceboToneMapping : public NVEncFilterDisabled {
public:
    NVEncFilterLibplaceboToneMapping();
    virtual ~NVEncFilterLibplaceboToneMapping();
};

#endif

#endif //#ifndef __NVENC_FILTER_LIBPLACEBO_H__