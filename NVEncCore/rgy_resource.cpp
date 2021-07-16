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
#if ENCODER_QSV
extern char _binary_QSVPipeline_rgy_filter_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_afs_analyze_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_afs_analyze_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_afs_filter_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_afs_filter_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_afs_merge_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_afs_merge_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_afs_synthesize_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_afs_synthesize_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_deband_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_deband_cl_end[];
extern char _binary_clRNG_src_include_clRNG_clRNG_clh_start[];
extern char _binary_clRNG_src_include_clRNG_clRNG_clh_end[];
extern char _binary_clRNG_src_include_clRNG_mrg31k3p_clh_start[];
extern char _binary_clRNG_src_include_clRNG_mrg31k3p_clh_end[];
extern char _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_start[];
extern char _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_end[];
extern char _binary_QSVPipeline_rgy_filter_deband_gen_rand_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_deband_gen_rand_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_decimate_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_decimate_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_delogo_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_delogo_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_denoise_knn_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_denoise_knn_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_denoise_pmd_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_denoise_pmd_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_edgelevel_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_edgelevel_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_mpdecimate_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_mpdecimate_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_common_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_common_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_k0_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_k0_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_k1_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_nnedi_k1_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_pad_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_pad_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_resize_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_resize_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_smooth_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_smooth_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_subburn_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_subburn_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_transform_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_transform_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_tweak_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_tweak_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_unsharp_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_unsharp_cl_end[];
extern char _binary_QSVPipeline_rgy_filter_warpsharp_cl_start[];
extern char _binary_QSVPipeline_rgy_filter_warpsharp_cl_end[];
#endif
#if ENCODER_NVENC
extern char _binary_NVEncCore_NVEncFilterColorspaceFunc_h_start[];
extern char _binary_NVEncCore_NVEncFilterColorspaceFunc_h_end[];
#endif
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

#if ENCODER_QSV
    { _T("EXE_DATA"), _T("RGY_FILTER_CL"), _binary_QSVPipeline_rgy_filter_cl_start, _binary_QSVPipeline_rgy_filter_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_ANALYZE_CL"), _binary_QSVPipeline_rgy_filter_afs_analyze_cl_start, _binary_QSVPipeline_rgy_filter_afs_analyze_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_FILTER_CL"), _binary_QSVPipeline_rgy_filter_afs_filter_cl_start, _binary_QSVPipeline_rgy_filter_afs_filter_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_MERGE_CL"), _binary_QSVPipeline_rgy_filter_afs_merge_cl_start, _binary_QSVPipeline_rgy_filter_afs_merge_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_AFS_SYNTHESIZE_CL"), _binary_QSVPipeline_rgy_filter_afs_synthesize_cl_start, _binary_QSVPipeline_rgy_filter_afs_synthesize_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_DEBAND_CL"), _binary_QSVPipeline_rgy_filter_deband_cl_start, _binary_QSVPipeline_rgy_filter_deband_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_DEBAND_GEN_RAND_CL"), _binary_QSVPipeline_rgy_filter_deband_gen_rand_cl_start, _binary_QSVPipeline_rgy_filter_deband_gen_rand_cl_end },


    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_CLH"), _binary_clRNG_src_include_clRNG_clRNG_clh_start, _binary_clRNG_src_include_clRNG_clRNG_clh_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_MRG31K3P_CLH"), _binary_clRNG_src_include_clRNG_mrg31k3p_clh_start, _binary_clRNG_src_include_clRNG_mrg31k3p_clh_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_CLRNG_MRG31K3P_PRIVATE_CH"), _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_start, _binary_clRNG_src_include_clRNG_private_mrg31k3p_c_h_end },

    { _T("EXE_DATA"), _T("RGY_FILTER_DECIMATE_CL"), _binary_QSVPipeline_rgy_filter_decimate_cl_start, _binary_QSVPipeline_rgy_filter_decimate_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_DELOGO_CL"), _binary_QSVPipeline_rgy_filter_delogo_cl_start, _binary_QSVPipeline_rgy_filter_delogo_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_DENOISE_KNN_CL"), _binary_QSVPipeline_rgy_filter_denoise_knn_cl_start, _binary_QSVPipeline_rgy_filter_denoise_knn_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_DENOISE_PMD_CL"), _binary_QSVPipeline_rgy_filter_denoise_pmd_cl_start, _binary_QSVPipeline_rgy_filter_denoise_pmd_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_EDGELEVEL_CL"), _binary_QSVPipeline_rgy_filter_edgelevel_cl_start, _binary_QSVPipeline_rgy_filter_edgelevel_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_MPDECIMATE_CL"), _binary_QSVPipeline_rgy_filter_mpdecimate_cl_start, _binary_QSVPipeline_rgy_filter_mpdecimate_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_COMMON_CL"), _binary_QSVPipeline_rgy_filter_nnedi_common_cl_start, _binary_QSVPipeline_rgy_filter_nnedi_common_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_K0_CL"), _binary_QSVPipeline_rgy_filter_nnedi_k0_cl_start, _binary_QSVPipeline_rgy_filter_nnedi_k0_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_NNEDI_K1_CL"), _binary_QSVPipeline_rgy_filter_nnedi_k1_cl_start, _binary_QSVPipeline_rgy_filter_nnedi_k1_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_RESIZE_CL"), _binary_QSVPipeline_rgy_filter_resize_cl_start, _binary_QSVPipeline_rgy_filter_resize_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_SMOOTH_CL"), _binary_QSVPipeline_rgy_filter_smooth_cl_start, _binary_QSVPipeline_rgy_filter_smooth_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_SUBBURN_CL"), _binary_QSVPipeline_rgy_filter_subburn_cl_start, _binary_QSVPipeline_rgy_filter_subburn_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_TRANSFORM_CL"), _binary_QSVPipeline_rgy_filter_transform_cl_start, _binary_QSVPipeline_rgy_filter_transform_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_TWEAK_CL"), _binary_QSVPipeline_rgy_filter_tweak_cl_start, _binary_QSVPipeline_rgy_filter_tweak_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_UNSHARP_CL"), _binary_QSVPipeline_rgy_filter_unsharp_cl_start, _binary_QSVPipeline_rgy_filter_unsharp_cl_end },
    { _T("EXE_DATA"), _T("RGY_FILTER_WARPSHARP_CL"), _binary_QSVPipeline_rgy_filter_warpsharp_cl_start, _binary_QSVPipeline_rgy_filter_warpsharp_cl_end }
#endif
#if ENCODER_NVENC
    { _T("EXE_DATA"), _T("NVENC_FILTER_COLRSPACE_FUNC_HEADER"), _binary_NVEncCore_NVEncFilterColorspaceFunc_h_start, _binary_NVEncCore_NVEncFilterColorspaceFunc_h_end }
#endif
};
#endif

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

