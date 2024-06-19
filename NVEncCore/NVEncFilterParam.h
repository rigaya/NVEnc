// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#pragma once
#ifndef _NVENC_FILTER_PARAM_H_
#define _NVENC_FILTER_PARAM_H_

#include <limits.h>
#include <vector>
#include "rgy_osdep.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#pragma warning (disable: 4201)
#include <npp.h>
#pragma warning (pop)
#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_simd.h"
#include "rgy_prm.h"
#include "convert_csp.h"
#include "cuda.h"
#if ENCODER_NVENC
#include "dynlink_cuviddec.h"
#endif //#if ENCODER_NVENC

static const TCHAR *FILTER_DEFAULT_CUSTOM_KERNEL_NAME = _T("kernel_filter");
static const int FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_X = 32;
static const int FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_Y = 8;
static const int FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_X = 1;
static const int FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_Y = 1;

static const float FILTER_DEFAULT_NVVFX_DENOISE_STRENGTH = 0.0f;
static const int FILTER_DEFAULT_NVVFX_ARTIFACT_REDUCTION_MODE = 0;
static const float FILTER_DEFAULT_NVVFX_SUPER_RES_STRENGTH = 0.4f;
static const int FILTER_DEFAULT_NVVFX_SUPER_RES_MODE = 1;
static const float FILTER_DEFAULT_NVVFX_UPSCALER_STRENGTH = 0.4f;

static const int DEFAULT_CUDA_SCHEDULE = CU_CTX_SCHED_AUTO;

const CX_DESC list_nppi_gauss[] = {
    { _T("disabled"), 0 },
    { _T("3"), NPP_MASK_SIZE_3_X_3 },
    { _T("5"), NPP_MASK_SIZE_5_X_5 },
    { _T("7"), NPP_MASK_SIZE_7_X_7 },
    { NULL, 0 }
};

const CX_DESC list_cuda_schedule[] = {
    { _T("auto"),  CU_CTX_SCHED_AUTO },
    { _T("spin"),  CU_CTX_SCHED_SPIN },
    { _T("yield"), CU_CTX_SCHED_YIELD },
    { _T("sync"),  CU_CTX_SCHED_BLOCKING_SYNC },
    { NULL, 0 }
};

enum VppCustomInterface {
    VPP_CUSTOM_INTERFACE_PER_PLANE,
    VPP_CUSTOM_INTERFACE_PLANES,

    VPP_CUSTOM_INTERFACE_MAX,
};

const CX_DESC list_vpp_custom_interface[] = {
    { _T("per_plane"),    VPP_CUSTOM_INTERFACE_PER_PLANE },
    { _T("planes"),       VPP_CUSTOM_INTERFACE_PLANES },
    { NULL, 0 }
};

enum VppCustomInterlaceMode {
    VPP_CUSTOM_INTERLACE_UNSUPPORTED,
    VPP_CUSTOM_INTERLACE_PER_FIELD,
    VPP_CUSTOM_INTERLACE_FRAME,

    VPP_CUSTOM_INTERLACE_MAX,
};

const CX_DESC list_vpp_custom_interlace[] = {
    { _T("unsupported"), VPP_CUSTOM_INTERLACE_UNSUPPORTED },
    { _T("per_field"),   VPP_CUSTOM_INTERLACE_PER_FIELD },
    { _T("frame"),       VPP_CUSTOM_INTERLACE_FRAME },
    { NULL, 0 }
};

struct VppCustom {
    bool enable;
    tstring filter_name;
    tstring kernel_name;
    tstring kernel_path;
    std::string kernel;
    void *dev_params;
    std::string compile_options;
    VppCustomInterface kernel_interface;
    VppCustomInterlaceMode interlace;
    int threadPerBlockX;
    int threadPerBlockY;
    int pixelPerThreadX;
    int pixelPerThreadY;
    int dstWidth;
    int dstHeight;
    std::map<std::string, std::string> params;

    VppCustom();
    bool operator==(const VppCustom &x) const;
    bool operator!=(const VppCustom &x) const;
    tstring print() const;
};

const CX_DESC list_vpp_nvvfx_mode[] = {
    { _T("conservative"), 0 },
    { _T("aggressive"),   1 },
    { NULL, 0 }
};

struct VppNvvfxDenoise {
    bool enable;
    float strength;

    VppNvvfxDenoise();
    bool operator==(const VppNvvfxDenoise &x) const;
    bool operator!=(const VppNvvfxDenoise &x) const;
    tstring print() const;
};

struct VppNvvfxArtifactReduction {
    bool enable;
    int mode; // 0: conservative, 1: aggressive

    VppNvvfxArtifactReduction();
    bool operator==(const VppNvvfxArtifactReduction &x) const;
    bool operator!=(const VppNvvfxArtifactReduction &x) const;
    tstring print() const;
};

struct VppNvvfxSuperRes {
    bool enable;
    int mode; // 0: conservative, 1: aggressive
    float strength;

    VppNvvfxSuperRes();
    bool operator==(const VppNvvfxSuperRes &x) const;
    bool operator!=(const VppNvvfxSuperRes &x) const;
    tstring print() const;
};

struct VppNvvfxUpScaler {
    bool enable;
    float strength;

    VppNvvfxUpScaler();
    bool operator==(const VppNvvfxUpScaler &x) const;
    bool operator!=(const VppNvvfxUpScaler &x) const;
    tstring print() const;
};

struct VppNGXVSR {
    bool enable;
    int quality;

    VppNGXVSR();
    bool operator==(const VppNGXVSR &x) const;
    bool operator!=(const VppNGXVSR &x) const;
    tstring print() const;
};

struct VppNVSDKNGXTrueHDR {
    bool enable;
    uint32_t contrast;
    uint32_t saturation;
    uint32_t middleGray;
    uint32_t maxLuminance;

    VppNVSDKNGXTrueHDR();
    bool operator==(const VppNVSDKNGXTrueHDR &x) const;
    bool operator!=(const VppNVSDKNGXTrueHDR &x) const;
    tstring print() const;
};

struct VppParam {
#if ENCODER_NVENC
    cudaVideoDeinterlaceMode  deinterlace;
#endif //#if ENCODER_NVENC
    NppiMaskSize              gaussMaskSize;
    VppNvvfxDenoise           nvvfxDenoise;
    VppNvvfxArtifactReduction nvvfxArtifactReduction;
    VppNvvfxSuperRes          nvvfxSuperRes;
    VppNvvfxUpScaler          nvvfxUpScaler;
    tstring                   nvvfxModelDir;
    VppNGXVSR            nvsdkngxVSR;
    VppNVSDKNGXTrueHDR        nvsdkngxTrueHDR;

    VppParam();
};

#endif //_NVENC_FILTER_PARAM_H_
