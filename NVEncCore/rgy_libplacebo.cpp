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

#include "rgy_libplacebo.h"
#include "rgy_def.h"
#include "rgy_prm.h"

#if ENABLE_LIBPLACEBO

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo-349.dll");
#else
const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo.so");
#endif // #if defined(_WIN32) || defined(_WIN64)

RGYLibplaceboLoader::RGYLibplaceboLoader() :
    m_hModule(nullptr),
    m_pl_color_space_bt2020_hlg(nullptr),
    m_pl_color_space_bt709(nullptr),
    m_pl_color_space_srgb(nullptr),
    m_pl_color_space_hdr10(nullptr),
    m_pl_hdr_metadata_empty(nullptr),
    m_pl_peak_detect_default_params(nullptr),
    m_pl_color_map_default_params(nullptr),
#if ENABLE_D3D11
    m_pl_d3d11_create(nullptr),
    m_pl_d3d11_destroy(nullptr),
    m_pl_d3d11_wrap(nullptr),
#elif ENABLE_VULKAN
    m_pl_vulkan_create(nullptr),
    m_pl_vulkan_destroy(nullptr),
    m_pl_vulkan_hold_ex(nullptr),
    m_pl_vulkan_release_ex(nullptr),
    m_pl_vulkan_wrap(nullptr),
#endif
    m_pl_gpu_finish(nullptr),
    m_pl_tex_destroy(nullptr),
    m_pl_tex_recreate(nullptr),
    m_pl_log_create(nullptr),
    m_pl_log_destroy(nullptr),
    m_pl_dispatch_create(nullptr),
    m_pl_dispatch_destroy(nullptr),
    m_pl_dispatch_begin(nullptr),
    m_pl_dispatch_finish(nullptr),
    m_pl_dispatch_abort(nullptr),
    m_pl_renderer_create(nullptr),
    m_pl_renderer_destroy(nullptr),
    m_pl_render_image(nullptr),
    m_pl_shader_sample_direct(nullptr),
    m_pl_shader_linearize(nullptr),
    m_pl_shader_sigmoidize(nullptr),
    m_pl_shader_sample_polar(nullptr),
    m_pl_shader_sample_ortho2(nullptr),
    m_pl_shader_obj_destroy(nullptr),
    m_pl_shader_reset(nullptr),
    m_pl_shader_deband(nullptr),
    m_pl_shader_dither(nullptr),
    m_pl_find_filter_config(nullptr),
    m_pl_hdr_rescale(nullptr),
    m_pl_lut_parse_cube(nullptr),
    m_pl_find_gamut_map_function(nullptr),
    m_pl_raw_primaries_get(nullptr),
    m_pl_raw_primaries_merge(nullptr),
    m_pl_color_space_infer_map(nullptr),
    m_pl_frame_set_chroma_location(nullptr),
    m_pl_shader_custom(nullptr),
    m_pl_mpv_user_shader_parse(nullptr),
    m_pl_mpv_user_shader_destroy(nullptr)
{
}

RGYLibplaceboLoader::~RGYLibplaceboLoader() {
    close();
}

bool RGYLibplaceboLoader::load() {
    if (m_hModule) {
        return true;
    }

    if ((m_hModule = RGY_LOAD_LIBRARY(RGY_LIBPLACEBO_DLL_NAME)) == nullptr) {
        return false;
    }

    auto loadFunc = [this](const char *funcName, void **func) {
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
            return false;
        }
        return true;
    };

    if (!loadFunc("pl_color_space_bt2020_hlg", (void**)&m_pl_color_space_bt2020_hlg)) return false;
    if (!loadFunc("pl_color_space_bt709", (void**)&m_pl_color_space_bt709)) return false;
    if (!loadFunc("pl_color_space_srgb", (void**)&m_pl_color_space_srgb)) return false;
    if (!loadFunc("pl_color_space_hdr10", (void**)&m_pl_color_space_hdr10)) return false;
    if (!loadFunc("pl_hdr_metadata_empty", (void**)&m_pl_hdr_metadata_empty)) return false;
    if (!loadFunc("pl_peak_detect_default_params", (void**)&m_pl_peak_detect_default_params)) return false;
    if (!loadFunc("pl_color_map_default_params", (void**)&m_pl_color_map_default_params)) return false;

    // 新しいメンバ変数の関数ポインタを取得して格納するコードを追加
#if ENABLE_D3D11
    if (!loadFunc("pl_d3d11_create", (void**)&m_pl_d3d11_create)) return false;
    if (!loadFunc("pl_d3d11_destroy", (void**)&m_pl_d3d11_destroy)) return false;
    if (!loadFunc("pl_d3d11_wrap", (void**)&m_pl_d3d11_wrap)) return false;
#elif ENABLE_VULKAN
    if (!loadFunc("pl_vulkan_create", (void**)&m_pl_vulkan_create)) return false;
    if (!loadFunc("pl_vulkan_destroy", (void**)&m_pl_vulkan_destroy)) return false;
    if (!loadFunc("pl_vulkan_hold_ex", (void**)&m_pl_vulkan_hold_ex)) return false;
    if (!loadFunc("pl_vulkan_release_ex", (void**)&m_pl_vulkan_release_ex)) return false;
    if (!loadFunc("pl_vulkan_wrap", (void**)&m_pl_vulkan_wrap)) return false;
#endif
    if (!loadFunc("pl_tex_destroy", (void**)&m_pl_tex_destroy)) return false;
    if (!loadFunc("pl_tex_recreate", (void**)&m_pl_tex_recreate)) return false;

    if (!loadFunc("pl_gpu_finish", (void**)&m_pl_gpu_finish)) return false;

    char pl_log_create_str[256] = { 0 };
    sprintf_s(pl_log_create_str, "pl_log_create_%d", PL_API_VER);
    loadFunc(pl_log_create_str, (void**)&m_pl_log_create);
    if (!loadFunc("pl_log_destroy", (void**)&m_pl_log_destroy)) return false;

    if (!loadFunc("pl_dispatch_create", (void**)&m_pl_dispatch_create)) return false;
    if (!loadFunc("pl_dispatch_destroy", (void**)&m_pl_dispatch_destroy)) return false;
    if (!loadFunc("pl_dispatch_begin", (void**)&m_pl_dispatch_begin)) return false;
    if (!loadFunc("pl_dispatch_finish", (void**)&m_pl_dispatch_finish)) return false;
    if (!loadFunc("pl_dispatch_abort", (void**)&m_pl_dispatch_abort)) return false;
    if (!loadFunc("pl_renderer_create", (void**)&m_pl_renderer_create)) return false;
    if (!loadFunc("pl_renderer_destroy", (void**)&m_pl_renderer_destroy)) return false;
    if (!loadFunc("pl_render_image", (void**)&m_pl_render_image)) return false;
    if (!loadFunc("pl_shader_sample_direct", (void**)&m_pl_shader_sample_direct)) return false;
    if (!loadFunc("pl_shader_linearize", (void**)&m_pl_shader_linearize)) return false;
    if (!loadFunc("pl_shader_sigmoidize", (void**)&m_pl_shader_sigmoidize)) return false;
    if (!loadFunc("pl_shader_sample_polar", (void**)&m_pl_shader_sample_polar)) return false;
    if (!loadFunc("pl_shader_sample_ortho2", (void**)&m_pl_shader_sample_ortho2)) return false;
    if (!loadFunc("pl_shader_obj_destroy", (void**)&m_pl_shader_obj_destroy)) return false;
    if (!loadFunc("pl_shader_reset", (void**)&m_pl_shader_reset)) return false;
    if (!loadFunc("pl_shader_deband", (void**)&m_pl_shader_deband)) return false;
    if (!loadFunc("pl_shader_dither", (void**)&m_pl_shader_dither)) return false;
    if (!loadFunc("pl_find_filter_config", (void**)&m_pl_find_filter_config)) return false;

    if (!loadFunc("pl_hdr_rescale", (void**)&m_pl_hdr_rescale)) return false;
    if (!loadFunc("pl_lut_parse_cube", (void**)&m_pl_lut_parse_cube)) return false;
    if (!loadFunc("pl_find_gamut_map_function", (void**)&m_pl_find_gamut_map_function)) return false;
    if (!loadFunc("pl_raw_primaries_get", (void**)&m_pl_raw_primaries_get)) return false;
    if (!loadFunc("pl_raw_primaries_merge", (void**)&m_pl_raw_primaries_merge)) return false;
    if (!loadFunc("pl_color_space_infer_map", (void**)&m_pl_color_space_infer_map)) return false;

    if (!loadFunc("pl_frame_set_chroma_location", (void**)&m_pl_frame_set_chroma_location)) return false;

    if (!loadFunc("pl_shader_custom", (void**)&m_pl_shader_custom)) return false;
    if (!loadFunc("pl_mpv_user_shader_parse", (void**)&m_pl_mpv_user_shader_parse)) return false;
    if (!loadFunc("pl_mpv_user_shader_destroy", (void**)&m_pl_mpv_user_shader_destroy)) return false;

    return true;
}

void RGYLibplaceboLoader::close() {
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }

    m_pl_color_space_bt2020_hlg = nullptr;
    m_pl_color_space_bt709 = nullptr;
    m_pl_color_space_srgb = nullptr;
    m_pl_color_space_hdr10 = nullptr;
}

static const auto RGY_LOG_LEVEL_TO_LIBPLACEBO = make_array<std::pair<RGYLogLevel, pl_log_level>>(
    std::make_pair(RGYLogLevel::RGY_LOG_QUIET, PL_LOG_NONE),
    std::make_pair(RGYLogLevel::RGY_LOG_ERROR, PL_LOG_ERR),
    std::make_pair(RGYLogLevel::RGY_LOG_WARN,  PL_LOG_WARN),
    std::make_pair(RGYLogLevel::RGY_LOG_INFO,  PL_LOG_INFO),
    std::make_pair(RGYLogLevel::RGY_LOG_DEBUG, PL_LOG_DEBUG),
    std::make_pair(RGYLogLevel::RGY_LOG_TRACE, PL_LOG_TRACE)
);

MAP_PAIR_0_1(loglevel, rgy, RGYLogLevel, libplacebo, pl_log_level, RGY_LOG_LEVEL_TO_LIBPLACEBO, RGYLogLevel::RGY_LOG_INFO, PL_LOG_INFO);

static const auto RGY_RESIZE_ALGO_TO_LIBPLACEBO = make_array<std::pair<RGY_VPP_RESIZE_ALGO, const char*>>(
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE16, "spline16"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE36, "spline36"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE64, "spline64"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_NEAREST, "nearest"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_BILINEAR, "bilinear"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_GAUSSIAN, "gaussian"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SINC, "sinc"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_LANCZOS, "lanczos"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_GINSENG, "ginseng"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_JINC, "ewa_jinc"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS, "ewa_lanczos"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOSSHARP, "ewa_lanczossharp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS4SHARPEST, "ewa_lanczos4sharpest"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_GINSENG, "ewa_ginseng"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANN, "ewa_hann"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANNING, "ewa_hanning"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_BICUBIC, "bicubic"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_TRIANGLE, "triangle"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_HERMITE, "hermite"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_CATMULL_ROM, "catmull_rom"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL, "mitchell"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL_CLAMP, "mitchell_clamp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUX, "robidoux"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUXSHARP, "robidouxsharp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUX, "ewa_robidoux"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUXSHARP, "ewa_robidouxsharp")
);

MAP_PAIR_0_1(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, libplacebo, const char*, RGY_RESIZE_ALGO_TO_LIBPLACEBO, RGY_VPP_RESIZE_UNKNOWN, nullptr);

 static const auto RGY_TONEMAP_METADATA_TO_LIBPLACEBO = make_array<std::pair<VppLibplaceboToneMappingMetadata, pl_hdr_metadata_type>>(
    std::make_pair(VppLibplaceboToneMappingMetadata::ANY, PL_HDR_METADATA_ANY),
    std::make_pair(VppLibplaceboToneMappingMetadata::NONE, PL_HDR_METADATA_NONE),
    std::make_pair(VppLibplaceboToneMappingMetadata::HDR10, PL_HDR_METADATA_HDR10),
    std::make_pair(VppLibplaceboToneMappingMetadata::HDR10PLUS, PL_HDR_METADATA_HDR10PLUS),
    std::make_pair(VppLibplaceboToneMappingMetadata::CIE_Y, PL_HDR_METADATA_CIE_Y)
);

MAP_PAIR_0_1(tone_map_metadata, rgy, VppLibplaceboToneMappingMetadata, libplacebo, pl_hdr_metadata_type, RGY_TONEMAP_METADATA_TO_LIBPLACEBO, VppLibplaceboToneMappingMetadata::ANY, PL_HDR_METADATA_ANY);

static const auto RGY_TRANSFER_TO_LIBPLACEBO = make_array<std::pair<CspTransfer, pl_color_transfer>>(
    std::make_pair(RGY_TRANSFER_UNKNOWN,      PL_COLOR_TRC_UNKNOWN),
    std::make_pair(RGY_TRANSFER_BT709,        PL_COLOR_TRC_BT_1886),
    std::make_pair(RGY_TRANSFER_BT601,        PL_COLOR_TRC_BT_1886),
    std::make_pair(RGY_TRANSFER_BT2020_10,    PL_COLOR_TRC_BT_1886),
    std::make_pair(RGY_TRANSFER_BT2020_12,    PL_COLOR_TRC_BT_1886),
    std::make_pair(RGY_TRANSFER_IEC61966_2_1, PL_COLOR_TRC_SRGB),
    std::make_pair(RGY_TRANSFER_LINEAR,       PL_COLOR_TRC_LINEAR),
    std::make_pair(RGY_TRANSFER_ST2084,       PL_COLOR_TRC_PQ),
    std::make_pair(RGY_TRANSFER_ARIB_B67,     PL_COLOR_TRC_HLG)
);

MAP_PAIR_0_1(transfer, rgy, CspTransfer, libplacebo, pl_color_transfer, RGY_TRANSFER_TO_LIBPLACEBO, RGY_TRANSFER_UNKNOWN, PL_COLOR_TRC_UNKNOWN);

static const auto RGY_COLORPRIM_TO_LIBPLACEBO = make_array<std::pair<CspColorprim, pl_color_primaries>>(
    std::make_pair(RGY_PRIM_UNKNOWN,     PL_COLOR_PRIM_UNKNOWN),
    std::make_pair(RGY_PRIM_BT709,       PL_COLOR_PRIM_BT_709),
    std::make_pair(RGY_PRIM_UNSPECIFIED, PL_COLOR_PRIM_UNKNOWN),
    std::make_pair(RGY_PRIM_BT470_M,     PL_COLOR_PRIM_BT_470M),
    std::make_pair(RGY_PRIM_BT470_BG,    PL_COLOR_PRIM_BT_601_625),
    std::make_pair(RGY_PRIM_ST170_M,     PL_COLOR_PRIM_BT_601_525),
    std::make_pair(RGY_PRIM_ST240_M,     PL_COLOR_PRIM_BT_601_525), // 近似値
    std::make_pair(RGY_PRIM_FILM,        PL_COLOR_PRIM_FILM_C),
    std::make_pair(RGY_PRIM_BT2020,      PL_COLOR_PRIM_BT_2020),
    std::make_pair(RGY_PRIM_ST428,       PL_COLOR_PRIM_CIE_1931),
    std::make_pair(RGY_PRIM_ST431_2,     PL_COLOR_PRIM_DCI_P3),
    std::make_pair(RGY_PRIM_ST432_1,     PL_COLOR_PRIM_DISPLAY_P3),
    std::make_pair(RGY_PRIM_EBU3213_E,   PL_COLOR_PRIM_EBU_3213)
);

MAP_PAIR_0_1(colorprim, rgy, CspColorprim, libplacebo, pl_color_primaries, RGY_COLORPRIM_TO_LIBPLACEBO, RGY_PRIM_UNKNOWN, PL_COLOR_PRIM_UNKNOWN);

static const auto RGY_CHROMALOC_TO_LIBPLACEBO = make_array<std::pair<CspChromaloc, pl_chroma_location>>(
    std::make_pair(RGY_CHROMALOC_UNSPECIFIED, PL_CHROMA_UNKNOWN),
    std::make_pair(RGY_CHROMALOC_LEFT, PL_CHROMA_LEFT),
    std::make_pair(RGY_CHROMALOC_CENTER, PL_CHROMA_CENTER),
    std::make_pair(RGY_CHROMALOC_TOPLEFT, PL_CHROMA_TOP_LEFT),
    std::make_pair(RGY_CHROMALOC_TOP, PL_CHROMA_TOP_CENTER),
    std::make_pair(RGY_CHROMALOC_BOTTOMLEFT, PL_CHROMA_BOTTOM_LEFT),
    std::make_pair(RGY_CHROMALOC_BOTTOM, PL_CHROMA_BOTTOM_CENTER)
);

MAP_PAIR_0_1(chromaloc, rgy, CspChromaloc, libplacebo, pl_chroma_location, RGY_CHROMALOC_TO_LIBPLACEBO, RGY_CHROMALOC_UNSPECIFIED, PL_CHROMA_UNKNOWN);

std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> rgy_pl_tex_recreate(const RGYLibplaceboLoader *pl, pl_gpu gpu, const pl_tex_params& tex_params) {
    pl_tex tex_tmp = { 0 };
    if (!pl->p_tex_recreate()(gpu, &tex_tmp, &tex_params)) {
        return std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>();
    }
    return std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>(
        tex_tmp, RGYLibplaceboTexDeleter(pl, gpu));
}

#endif //ENABLE_LIBPLACEBO
