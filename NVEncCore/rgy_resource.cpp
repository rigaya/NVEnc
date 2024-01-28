// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#include "rgy_version.h"
#include "rgy_tchar.h"
#include "rgy_resource.h"

#if !(defined(_WIN32) || defined(_WIN64))
extern "C" {
extern char _binary_PerfMonitor_perf_monitor_pyw_start[];
extern char _binary_PerfMonitor_perf_monitor_pyw_end[];
extern char _binary_resource_nnedi3_weights_bin_start[];
extern char _binary_resource_nnedi3_weights_bin_end[];
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
extern char _binary_clRNG_src_include_clRNG_clRNG_clh_start[];
extern char _binary_clRNG_src_include_clRNG_clRNG_clh_end[];
extern char _binary_clRNG_src_include_clRNG_mrg31k3p_clh_start[];
extern char _binary_clRNG_src_include_clRNG_mrg31k3p_clh_end[];
extern char _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_start[];
extern char _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_end[];

#if ENCODER_QSV
#define _BINARY_VPP_FILTER_FILE(FILENAME) \
    extern char _binary_QSVPipeline_##FILENAME##_start[]; \
    extern char _binary_QSVPipeline_##FILENAME##_end[];
#elif ENCODER_VCEENC
#define _BINARY_VPP_FILTER_FILE(FILENAME) \
    extern char _binary_VCECore_##FILENAME##_start[]; \
    extern char _binary_VCECore_##FILENAME##_end[];
#elif ENCODER_MPP
#define _BINARY_VPP_FILTER_FILE(FILENAME) \
    extern char _binary_mppcore_##FILENAME##_start[]; \
    extern char _binary_mppcore_##FILENAME##_end[];
#endif
_BINARY_VPP_FILTER_FILE(rgy_filter_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_afs_analyze_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_afs_filter_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_afs_merge_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_afs_synthesize_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_colorspace_func_h);
_BINARY_VPP_FILTER_FILE(rgy_filter_convolution3d_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_curves_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_deband_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_deband_gen_rand_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_decimate_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_delogo_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_denoise_dct_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_denoise_knn_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_denoise_pmd_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_edgelevel_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_mpdecimate_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_common_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_k0_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_k1_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_overlay_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_pad_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_resize_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_ssim_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_smooth_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_subburn_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_transform_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_tweak_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_unsharp_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_warpsharp_cl);
_BINARY_VPP_FILTER_FILE(rgy_filter_yadif_cl);

#undef _BINARY_VPP_FILTER_FILE

#endif //#if ENCODER_QSV || ENCODER_VCEENC
#if ENCODER_NVENC
extern char _binary_NVEncCore_NVEncFilterColorspaceFunc_h_start[];
extern char _binary_NVEncCore_NVEncFilterColorspaceFunc_h_end[];
#endif //#if ENCODER_NVENC
}

struct RGYResourceData {
    const TCHAR *type;
    const TCHAR *name;
    const char *start;
    const char *end;
};

static const RGYResourceData RGY_RESOURCE_DATA[] = {
    { _T("PERF_MONITOR_SRC"), _T("PERF_MONITOR_PYW"), _binary_PerfMonitor_perf_monitor_pyw_start, _binary_PerfMonitor_perf_monitor_pyw_end },
    
    { _T("EXE_DATA"), _T("NNEDI_WEIGHTBIN"), _binary_resource_nnedi3_weights_bin_start, _binary_resource_nnedi3_weights_bin_end },

#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_CLH"), _binary_clRNG_src_include_clRNG_clRNG_clh_start, _binary_clRNG_src_include_clRNG_clRNG_clh_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_MRG31K3P_CLH"), _binary_clRNG_src_include_clRNG_mrg31k3p_clh_start, _binary_clRNG_src_include_clRNG_mrg31k3p_clh_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_MRG31K3P_PRIVATE_CH"), _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_start, _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_end },

#if ENCODER_QSV
#define _BINARY_VPP_FILTER_FILE(FILENAME) _binary_QSVPipeline_##FILENAME##_start, _binary_QSVPipeline_##FILENAME##_end
#elif ENCODER_VCEENC
#define _BINARY_VPP_FILTER_FILE(FILENAME) _binary_VCECore_##FILENAME##_start, _binary_VCECore_##FILENAME##_end
#elif ENCODER_MPP
#define _BINARY_VPP_FILTER_FILE(FILENAME) _binary_mppcore_##FILENAME##_start, _binary_mppcore_##FILENAME##_end
#endif

    { _T("EXE_DATA"), _T("RGY_FILTER_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_ANALYZE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_afs_analyze_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_FILTER_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_afs_filter_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_MERGE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_afs_merge_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_SYNTHESIZE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_afs_synthesize_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_COLORSPACE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_colorspace_func_h) },
    { _T("EXE_DATA"), _T("RGY_FILTER_CONVOLUTION3D_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_convolution3d_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_CURVES_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_curves_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DEBAND_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_deband_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DEBAND_GEN_RAND_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_deband_gen_rand_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DECIMATE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_decimate_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DELOGO_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_delogo_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DENOISE_DCT_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_denoise_dct_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DENOISE_KNN_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_denoise_knn_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_DENOISE_PMD_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_denoise_pmd_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_EDGELEVEL_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_edgelevel_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_MPDECIMATE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_mpdecimate_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_COMMON_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_common_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_K0_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_k0_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_K1_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_nnedi_k1_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_OVERLAY_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_overlay_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_PAD_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_pad_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_RESIZE_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_resize_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_SSIM_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_ssim_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_SMOOTH_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_smooth_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_SUBBURN_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_subburn_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_TRANSFORM_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_transform_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_TWEAK_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_tweak_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_UNSHARP_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_unsharp_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_WARPSHARP_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_warpsharp_cl) },
    { _T("EXE_DATA"), _T("RGY_FILTER_YADIF_CL"), _BINARY_VPP_FILTER_FILE(rgy_filter_yadif_cl) }

#undef _BINARY_VPP_FILTER_FILE

#endif //#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
#if ENCODER_NVENC
    { _T("EXE_DATA"), _T("NVENC_FILTER_COLRSPACE_FUNC_HEADER"), _binary_NVEncCore_NVEncFilterColorspaceFunc_h_start, _binary_NVEncCore_NVEncFilterColorspaceFunc_h_end }
#endif //#if ENCODER_NVENC
};
#endif //#if !(defined(_WIN32) || defined(_WIN64))

int getEmbeddedResource(void **data, const TCHAR *name, const TCHAR *type, HMODULE hModule) {
    *data = nullptr;
#if defined(_WIN32) || defined(_WIN64)
    //埋め込みデータを使用する
    if (hModule == NULL) {
        hModule = GetModuleHandle(NULL);
    }
    if (hModule == NULL) {
        return 0;
    }
    HRSRC hResource = FindResource(hModule, name, type);
    if (hResource == NULL) {
        return 0;
    }
    HGLOBAL hResourceData = LoadResource(hModule, hResource);
    if (hResourceData == NULL) {
        return 0;
    }
    *data = LockResource(hResourceData);
    return (int)SizeofResource(hModule, hResource);
#else
    for (size_t i = 0; i < _countof(RGY_RESOURCE_DATA); i++) {
        if (_tcscmp(RGY_RESOURCE_DATA[i].type, type) == 0
            && _tcscmp(RGY_RESOURCE_DATA[i].name, name) == 0) {
            *data = (void *)RGY_RESOURCE_DATA[i].start;
            return (int)(size_t)(RGY_RESOURCE_DATA[i].end - RGY_RESOURCE_DATA[i].start);
        }
    }
    return 0;
#endif
}

std::string getEmbeddedResourceStr(const tstring& name, const tstring& type, HMODULE hModule) {
    std::string data_str;
    {
        char* data = nullptr;
        int size = getEmbeddedResource((void**)&data, name.c_str(), type.c_str(), hModule);
        if (size == 0) {
            return "";
        } else {

            auto datalen = size;
            {
                const uint8_t* ptr = (const uint8_t*)data;
                if (ptr[0] == 0xEF && ptr[1] == 0xBB && ptr[2] == 0xBF) { //skip UTF-8 BOM
                    data += 3;
                    datalen -= 3;
                }
            }
            data_str = std::string(data, datalen);
        }
    }
    return data_str;
}
