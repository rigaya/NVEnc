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
#include <array>
#include <cstring>

#if ENABLE_LIBPLACEBO

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo-360.dll");
#elif LIBPLACEBO_STATIC_LINK
const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo (static)");
#else
const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo.so");
#endif // #if defined(_WIN32) || defined(_WIN64)

RGYLibplaceboLoader::RGYLibplaceboLoader() :
    m_hModule(nullptr),
    m_loaded(false),
    m_apiVersion(0),
    m_pl_color_space_bt2020_hlg(nullptr),
    m_pl_color_space_bt709(nullptr),
    m_pl_color_space_srgb(nullptr),
    m_pl_color_space_hdr10(nullptr),
    m_pl_hdr_metadata_empty(nullptr),
    m_pl_render_default_params(nullptr),
    m_pl_peak_detect_default_params(nullptr),
    m_pl_color_map_default_params(nullptr),
    m_pl_sigmoid_default_params(nullptr),
    m_pl_dither_default_params(nullptr),
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
    m_pl_tex_destroy(nullptr),
    m_pl_tex_recreate(nullptr),
    m_pl_gpu_finish(nullptr),
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
    m_pl_find_tone_map_function(nullptr),
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
    if (m_loaded) {
        return true;
    }

#if LIBPLACEBO_STATIC_LINK
    m_apiVersion = PL_API_VER;
    m_pl_color_space_bt2020_hlg = const_cast<pl_color_space*>(&pl_color_space_bt2020_hlg);
    m_pl_color_space_bt709 = const_cast<pl_color_space*>(&pl_color_space_bt709);
    m_pl_color_space_srgb = const_cast<pl_color_space*>(&pl_color_space_srgb);
    m_pl_color_space_hdr10 = const_cast<pl_color_space*>(&pl_color_space_hdr10);
    m_pl_hdr_metadata_empty = const_cast<pl_hdr_metadata*>(&pl_hdr_metadata_empty);
    m_pl_render_default_params = const_cast<pl_render_params*>(&pl_render_default_params);
    m_pl_peak_detect_default_params = const_cast<pl_peak_detect_params*>(&pl_peak_detect_default_params);
    m_pl_color_map_default_params = const_cast<pl_color_map_params*>(&pl_color_map_default_params);
    m_pl_sigmoid_default_params = const_cast<pl_sigmoid_params*>(&pl_sigmoid_default_params);
    m_pl_dither_default_params = const_cast<pl_dither_params*>(&pl_dither_default_params);
#if ENABLE_D3D11
    m_pl_d3d11_create = &pl_d3d11_create;
    m_pl_d3d11_destroy = &pl_d3d11_destroy;
    m_pl_d3d11_wrap = &pl_d3d11_wrap;
#elif ENABLE_VULKAN
    m_pl_vulkan_create = &pl_vulkan_create;
    m_pl_vulkan_destroy = &pl_vulkan_destroy;
    m_pl_vulkan_hold_ex = &pl_vulkan_hold_ex;
    m_pl_vulkan_release_ex = &pl_vulkan_release_ex;
    m_pl_vulkan_wrap = &pl_vulkan_wrap;
#endif
    m_pl_tex_destroy = &pl_tex_destroy;
    m_pl_tex_recreate = &pl_tex_recreate;
    m_pl_gpu_finish = &pl_gpu_finish;
    m_pl_log_create = &pl_log_create;
    m_pl_log_destroy = &pl_log_destroy;
    m_pl_dispatch_create = &pl_dispatch_create;
    m_pl_dispatch_destroy = &pl_dispatch_destroy;
    m_pl_dispatch_begin = &pl_dispatch_begin;
    m_pl_dispatch_finish = &pl_dispatch_finish;
    m_pl_dispatch_abort = &pl_dispatch_abort;
    m_pl_renderer_create = &pl_renderer_create;
    m_pl_renderer_destroy = &pl_renderer_destroy;
    m_pl_render_image = &pl_render_image;
    m_pl_shader_sample_direct = &pl_shader_sample_direct;
    m_pl_shader_linearize = &pl_shader_linearize;
    m_pl_shader_sigmoidize = &pl_shader_sigmoidize;
    m_pl_shader_sample_polar = &pl_shader_sample_polar;
    m_pl_shader_sample_ortho2 = &pl_shader_sample_ortho2;
    m_pl_shader_obj_destroy = &pl_shader_obj_destroy;
    m_pl_shader_reset = &pl_shader_reset;
    m_pl_shader_deband = &pl_shader_deband;
    m_pl_shader_dither = &pl_shader_dither;
    m_pl_find_filter_config = &pl_find_filter_config;
    m_pl_hdr_rescale = &pl_hdr_rescale;
    m_pl_lut_parse_cube = &pl_lut_parse_cube;
    m_pl_find_tone_map_function = &pl_find_tone_map_function;
    m_pl_find_gamut_map_function = &pl_find_gamut_map_function;
    m_pl_raw_primaries_get = &pl_raw_primaries_get;
    m_pl_raw_primaries_merge = &pl_raw_primaries_merge;
    m_pl_color_space_infer_map = &pl_color_space_infer_map;
    m_pl_frame_set_chroma_location = &pl_frame_set_chroma_location;
    m_pl_shader_custom = &pl_shader_custom;
    m_pl_mpv_user_shader_parse = &pl_mpv_user_shader_parse;
    m_pl_mpv_user_shader_destroy = &pl_mpv_user_shader_destroy;
    m_loaded = true;
    return true;
#else
    auto loadFunc = [this](const char *funcName, void **func) {
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
            return false;
        }
        return true;
    };

#if defined(_WIN32) || defined(_WIN64)
    const TCHAR *libraryNames[] = {
#if PL_API_VER >= 360
        _T("libplacebo-360.dll"),
#endif
#if PL_API_VER >= 351
        _T("libplacebo-351.dll"),
#endif
#if PL_API_VER >= 349
        _T("libplacebo-349.dll"),
#endif
        _T("libplacebo-338.dll")
    };
#else
    const TCHAR *libraryNames[] = {
#if PL_API_VER >= 360
        _T("libplacebo.so.360"),
#endif
#if PL_API_VER >= 351
        _T("libplacebo.so.351"),
#endif
#if PL_API_VER >= 349
        _T("libplacebo.so.349"),
#endif
        _T("libplacebo.so.338"),
        _T("libplacebo.so")
    };
#endif
    const int apiVersions[] = {
#if PL_API_VER >= 360
        360,
#endif
#if PL_API_VER >= 351
        351,
#endif
#if PL_API_VER >= 349
        349,
#endif
        338
    };
    for (const auto libraryName : libraryNames) {
        m_hModule = RGY_LOAD_LIBRARY(libraryName);
        if (m_hModule == nullptr) {
            continue;
        }
        for (const auto apiVersion : apiVersions) {
            char funcName[64] = { 0 };
            sprintf_s(funcName, "pl_log_create_%d", apiVersion);
            if (loadFunc(funcName, (void **)&m_pl_log_create)) {
                m_apiVersion = apiVersion;
                break;
            }
        }
        if (m_apiVersion != 0) {
            break;
        }
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
    if (m_hModule == nullptr) {
        return false;
    }

    if (!loadFunc("pl_color_space_bt2020_hlg", (void**)&m_pl_color_space_bt2020_hlg)) { close(); return false; }
    if (!loadFunc("pl_color_space_bt709", (void**)&m_pl_color_space_bt709)) { close(); return false; }
    if (!loadFunc("pl_color_space_srgb", (void**)&m_pl_color_space_srgb)) { close(); return false; }
    if (!loadFunc("pl_color_space_hdr10", (void**)&m_pl_color_space_hdr10)) { close(); return false; }
    if (!loadFunc("pl_hdr_metadata_empty", (void**)&m_pl_hdr_metadata_empty)) { close(); return false; }
    if (!loadFunc("pl_render_default_params", (void**)&m_pl_render_default_params)) { close(); return false; }
    if (!loadFunc("pl_peak_detect_default_params", (void**)&m_pl_peak_detect_default_params)) { close(); return false; }
    if (!loadFunc("pl_color_map_default_params", (void**)&m_pl_color_map_default_params)) { close(); return false; }
    if (!loadFunc("pl_sigmoid_default_params", (void**)&m_pl_sigmoid_default_params)) { close(); return false; }
    if (!loadFunc("pl_dither_default_params", (void**)&m_pl_dither_default_params)) { close(); return false; }

    // 新しいメンバ変数の関数ポインタを取得して格納するコードを追加
#if ENABLE_D3D11
    if (!loadFunc("pl_d3d11_create", (void**)&m_pl_d3d11_create)) { close(); return false; }
    if (!loadFunc("pl_d3d11_destroy", (void**)&m_pl_d3d11_destroy)) { close(); return false; }
    if (!loadFunc("pl_d3d11_wrap", (void**)&m_pl_d3d11_wrap)) { close(); return false; }
#elif ENABLE_VULKAN
    if (!loadFunc("pl_vulkan_create", (void**)&m_pl_vulkan_create)) { close(); return false; }
    if (!loadFunc("pl_vulkan_destroy", (void**)&m_pl_vulkan_destroy)) { close(); return false; }
    if (!loadFunc("pl_vulkan_hold_ex", (void**)&m_pl_vulkan_hold_ex)) { close(); return false; }
    if (!loadFunc("pl_vulkan_release_ex", (void**)&m_pl_vulkan_release_ex)) { close(); return false; }
    if (!loadFunc("pl_vulkan_wrap", (void**)&m_pl_vulkan_wrap)) { close(); return false; }
#endif
    if (!loadFunc("pl_tex_destroy", (void**)&m_pl_tex_destroy)) { close(); return false; }
    if (!loadFunc("pl_tex_recreate", (void**)&m_pl_tex_recreate)) { close(); return false; }

    if (!loadFunc("pl_gpu_finish", (void**)&m_pl_gpu_finish)) { close(); return false; }

    if (!loadFunc("pl_log_destroy", (void**)&m_pl_log_destroy)) { close(); return false; }

    if (!loadFunc("pl_dispatch_create", (void**)&m_pl_dispatch_create)) { close(); return false; }
    if (!loadFunc("pl_dispatch_destroy", (void**)&m_pl_dispatch_destroy)) { close(); return false; }
    if (!loadFunc("pl_dispatch_begin", (void**)&m_pl_dispatch_begin)) { close(); return false; }
    if (!loadFunc("pl_dispatch_finish", (void**)&m_pl_dispatch_finish)) { close(); return false; }
    if (!loadFunc("pl_dispatch_abort", (void**)&m_pl_dispatch_abort)) { close(); return false; }
    if (!loadFunc("pl_renderer_create", (void**)&m_pl_renderer_create)) { close(); return false; }
    if (!loadFunc("pl_renderer_destroy", (void**)&m_pl_renderer_destroy)) { close(); return false; }
    if (!loadFunc("pl_render_image", (void**)&m_pl_render_image)) { close(); return false; }
    if (!loadFunc("pl_shader_sample_direct", (void**)&m_pl_shader_sample_direct)) { close(); return false; }
    if (!loadFunc("pl_shader_linearize", (void**)&m_pl_shader_linearize)) { close(); return false; }
    if (!loadFunc("pl_shader_sigmoidize", (void**)&m_pl_shader_sigmoidize)) { close(); return false; }
    if (!loadFunc("pl_shader_sample_polar", (void**)&m_pl_shader_sample_polar)) { close(); return false; }
    if (!loadFunc("pl_shader_sample_ortho2", (void**)&m_pl_shader_sample_ortho2)) { close(); return false; }
    if (!loadFunc("pl_shader_obj_destroy", (void**)&m_pl_shader_obj_destroy)) { close(); return false; }
    if (!loadFunc("pl_shader_reset", (void**)&m_pl_shader_reset)) { close(); return false; }
    if (!loadFunc("pl_shader_deband", (void**)&m_pl_shader_deband)) { close(); return false; }
    if (!loadFunc("pl_shader_dither", (void**)&m_pl_shader_dither)) { close(); return false; }
    if (!loadFunc("pl_find_filter_config", (void**)&m_pl_find_filter_config)) { close(); return false; }

    if (!loadFunc("pl_hdr_rescale", (void**)&m_pl_hdr_rescale)) { close(); return false; }
    if (!loadFunc("pl_lut_parse_cube", (void**)&m_pl_lut_parse_cube)) { close(); return false; }
    if (!loadFunc("pl_find_tone_map_function", (void**)&m_pl_find_tone_map_function)) { close(); return false; }
    if (!loadFunc("pl_find_gamut_map_function", (void**)&m_pl_find_gamut_map_function)) { close(); return false; }
    if (!loadFunc("pl_raw_primaries_get", (void**)&m_pl_raw_primaries_get)) { close(); return false; }
    if (!loadFunc("pl_raw_primaries_merge", (void**)&m_pl_raw_primaries_merge)) { close(); return false; }
    if (!loadFunc("pl_color_space_infer_map", (void**)&m_pl_color_space_infer_map)) { close(); return false; }

    if (!loadFunc("pl_frame_set_chroma_location", (void**)&m_pl_frame_set_chroma_location)) { close(); return false; }

    if (!loadFunc("pl_shader_custom", (void**)&m_pl_shader_custom)) { close(); return false; }
    if (!loadFunc("pl_mpv_user_shader_parse", (void**)&m_pl_mpv_user_shader_parse)) { close(); return false; }
    if (!loadFunc("pl_mpv_user_shader_destroy", (void**)&m_pl_mpv_user_shader_destroy)) { close(); return false; }

    m_loaded = true;
    return true;
#endif
}

void RGYLibplaceboLoader::close() {
#if !LIBPLACEBO_STATIC_LINK
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
#endif
    m_loaded = false;
    m_apiVersion = 0;

    m_pl_color_space_bt2020_hlg = nullptr;
    m_pl_color_space_bt709 = nullptr;
    m_pl_color_space_srgb = nullptr;
    m_pl_color_space_hdr10 = nullptr;
    m_pl_hdr_metadata_empty = nullptr;
    m_pl_render_default_params = nullptr;
    m_pl_peak_detect_default_params = nullptr;
    m_pl_color_map_default_params = nullptr;
    m_pl_sigmoid_default_params = nullptr;
    m_pl_dither_default_params = nullptr;
}

#if PL_API_VER >= 360
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
namespace {

// API 360の公開構造体を、動的ロードした旧APIが期待するABI配置へ詰め替える。
struct PLPeakDetectParams338 {
    float smoothing_period;
    float scene_threshold_low;
    float scene_threshold_high;
    float percentile;
    bool allow_delayed;
    float overshoot_margin;
    float minimum_peak;
};

struct PLRenderBackground338 {
    float background_color[3];
    float background_transparency;
    bool skip_target_clearing;
    float corner_rounding;
    bool blend_against_tiles;
    float tile_colors[2][3];
    int tile_size;
};

struct alignas(pl_render_params) PLRenderParamsLegacy {
    std::array<uint8_t, sizeof(pl_render_params) - sizeof(float)> data;
};

static_assert(sizeof(PLPeakDetectParams338) == sizeof(pl_peak_detect_params));
static_assert(sizeof(PLRenderBackground338) == offsetof(pl_render_params, skip_anti_aliasing) - offsetof(pl_render_params, background) - sizeof(float));

static PLPeakDetectParams338 peak_detect_params_to_338(const pl_peak_detect_params& params) {
    PLPeakDetectParams338 legacy = {};
    legacy.smoothing_period = params.smoothing_period;
    legacy.scene_threshold_low = params.scene_threshold_low;
    legacy.scene_threshold_high = params.scene_threshold_high;
    legacy.percentile = params.percentile;
    legacy.allow_delayed = params.allow_delayed;
    legacy.minimum_peak = params.minimum_peak;
    return legacy;
}

static pl_peak_detect_params peak_detect_params_from_338(const PLPeakDetectParams338& legacy) {
    pl_peak_detect_params params = {};
    params.smoothing_period = legacy.smoothing_period;
    params.scene_threshold_low = legacy.scene_threshold_low;
    params.scene_threshold_high = legacy.scene_threshold_high;
    params.percentile = legacy.percentile;
    params.black_cutoff = 1.0f;
    params.allow_delayed = legacy.allow_delayed;
    params.minimum_peak = legacy.minimum_peak;
    return params;
}

static PLRenderParamsLegacy render_params_to_legacy(const pl_render_params& params, const int apiVersion) {
    PLRenderParamsLegacy legacy = {};
    if (apiVersion >= 349) {
        constexpr auto prefixSize = offsetof(pl_render_params, blur_radius);
        constexpr auto suffixOffset = offsetof(pl_render_params, corner_rounding);
        std::memcpy(legacy.data.data(), &params, prefixSize);
        std::memcpy(legacy.data.data() + prefixSize,
            reinterpret_cast<const uint8_t *>(&params) + suffixOffset,
            sizeof(pl_render_params) - suffixOffset);
        return legacy;
    }

    constexpr auto prefixSize = offsetof(pl_render_params, background);
    constexpr auto suffixOffset = offsetof(pl_render_params, skip_anti_aliasing);
    constexpr auto suffixSize = offsetof(pl_render_params, skip_target_clearing) - suffixOffset;
    std::memcpy(legacy.data.data(), &params, prefixSize);

    PLRenderBackground338 background = {};
    std::memcpy(background.background_color, params.background_color, sizeof(background.background_color));
    background.background_transparency = params.background_transparency;
    background.skip_target_clearing = params.skip_target_clearing || params.border == PL_CLEAR_SKIP;
    background.corner_rounding = params.corner_rounding;
    background.blend_against_tiles = params.blend_against_tiles || params.background == PL_CLEAR_TILES;
    std::memcpy(background.tile_colors, params.tile_colors, sizeof(background.tile_colors));
    background.tile_size = params.tile_size;
    std::memcpy(legacy.data.data() + prefixSize, &background, sizeof(background));
    std::memcpy(legacy.data.data() + prefixSize + sizeof(background),
        reinterpret_cast<const uint8_t *>(&params) + suffixOffset, suffixSize);
    return legacy;
}

static pl_render_params render_params_from_legacy(const void *legacyPtr, const int apiVersion) {
    pl_render_params params = {};
    const auto legacy = reinterpret_cast<const uint8_t *>(legacyPtr);
    if (apiVersion >= 349) {
        constexpr auto prefixSize = offsetof(pl_render_params, blur_radius);
        constexpr auto suffixOffset = offsetof(pl_render_params, corner_rounding);
        std::memcpy(&params, legacy, prefixSize);
        params.blur_radius = 16.0f;
        std::memcpy(reinterpret_cast<uint8_t *>(&params) + suffixOffset,
            legacy + prefixSize, sizeof(pl_render_params) - suffixOffset);
        return params;
    }

    constexpr auto prefixSize = offsetof(pl_render_params, background);
    constexpr auto suffixOffset = offsetof(pl_render_params, skip_anti_aliasing);
    constexpr auto suffixSize = offsetof(pl_render_params, skip_target_clearing) - suffixOffset;
    std::memcpy(&params, legacy, prefixSize);

    PLRenderBackground338 background = {};
    std::memcpy(&background, legacy + prefixSize, sizeof(background));
    params.background = background.blend_against_tiles ? PL_CLEAR_TILES : PL_CLEAR_COLOR;
    params.border = background.skip_target_clearing ? PL_CLEAR_SKIP : PL_CLEAR_COLOR;
    std::memcpy(params.background_color, background.background_color, sizeof(params.background_color));
    params.background_transparency = background.background_transparency;
    std::memcpy(params.tile_colors, background.tile_colors, sizeof(params.tile_colors));
    params.tile_size = background.tile_size;
    params.blur_radius = 16.0f;
    params.corner_rounding = background.corner_rounding;
    std::memcpy(reinterpret_cast<uint8_t *>(&params) + suffixOffset,
        legacy + prefixSize + sizeof(background), suffixSize);
    params.skip_target_clearing = background.skip_target_clearing;
    params.blend_against_tiles = background.blend_against_tiles;
    return params;
}

static pl_color_system color_system_to_legacy(const pl_color_system colorSystem, const int apiVersion) {
    if (apiVersion >= 360 || colorSystem < PL_COLOR_SYSTEM_YCGCO_RE) {
        return colorSystem;
    }
    if (colorSystem <= PL_COLOR_SYSTEM_YCGCO_RO) {
        return PL_COLOR_SYSTEM_UNKNOWN;
    }
    return static_cast<pl_color_system>(static_cast<int>(colorSystem) - 2);
}

static pl_color_system color_system_from_legacy(const pl_color_system colorSystem, const int apiVersion) {
    constexpr auto legacyRgb = static_cast<int>(PL_COLOR_SYSTEM_RGB) - 2;
    if (apiVersion >= 360 || static_cast<int>(colorSystem) < legacyRgb) {
        return colorSystem;
    }
    return static_cast<pl_color_system>(static_cast<int>(colorSystem) + 2);
}

} // 無名名前空間
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

pl_render_params RGYLibplaceboLoader::p_render_default_params() const {
#if PL_API_VER >= 360
    if (m_apiVersion < 360) {
        return render_params_from_legacy(m_pl_render_default_params, m_apiVersion);
    }
#endif
    return *m_pl_render_default_params;
}

pl_peak_detect_params RGYLibplaceboLoader::p_peak_detect_default_params() const {
#if PL_API_VER >= 360
    if (m_apiVersion < 349) {
        return peak_detect_params_from_338(*reinterpret_cast<const PLPeakDetectParams338 *>(m_pl_peak_detect_default_params));
    }
#endif
    return *m_pl_peak_detect_default_params;
}

#if ENABLE_D3D11
pl_d3d11 RGYLibplaceboLoader::create_d3d11(pl_log log, const pl_d3d11_params *params) const {
    return m_pl_d3d11_create(log, params);
}
#elif ENABLE_VULKAN
pl_vulkan RGYLibplaceboLoader::create_vulkan(pl_log log, const pl_vulkan_params *params) const {
#if PL_API_VER >= 351
    if (m_apiVersion < 351) {
        struct alignas(pl_vulkan_params) PLVulkanParamsLegacy {
            std::array<uint8_t, sizeof(pl_vulkan_params) - sizeof(uint32_t)> data;
        } legacy = {};
        constexpr auto prefixSize = offsetof(pl_vulkan_params, no_compute);
        constexpr auto suffixOffset = offsetof(pl_vulkan_params, extra_queues);
        static_assert(suffixOffset - prefixSize == sizeof(uint32_t));
        std::memcpy(legacy.data.data(), params, prefixSize);
        std::memcpy(legacy.data.data() + prefixSize,
            reinterpret_cast<const uint8_t *>(params) + suffixOffset,
            sizeof(pl_vulkan_params) - suffixOffset);
        return m_pl_vulkan_create(log, reinterpret_cast<const pl_vulkan_params *>(legacy.data.data()));
    }
#endif
    return m_pl_vulkan_create(log, params);
}
#endif

bool RGYLibplaceboLoader::render_image(pl_renderer renderer, const pl_frame *image, const pl_frame *target, const pl_render_params *params) const {
#if PL_API_VER >= 360
    if (m_apiVersion < 360) {
        auto imageLegacy = *image;
        auto targetLegacy = *target;
        imageLegacy.repr.sys = color_system_to_legacy(imageLegacy.repr.sys, m_apiVersion);
        targetLegacy.repr.sys = color_system_to_legacy(targetLegacy.repr.sys, m_apiVersion);

        auto paramsCopy = *params;
        PLPeakDetectParams338 peakDetectLegacy = {};
        if (m_apiVersion < 349 && paramsCopy.peak_detect_params != nullptr) {
            peakDetectLegacy = peak_detect_params_to_338(*paramsCopy.peak_detect_params);
            paramsCopy.peak_detect_params = reinterpret_cast<const pl_peak_detect_params *>(&peakDetectLegacy);
        }
        const auto paramsLegacy = render_params_to_legacy(paramsCopy, m_apiVersion);
        return m_pl_render_image(renderer, &imageLegacy, &targetLegacy,
            reinterpret_cast<const pl_render_params *>(paramsLegacy.data.data()));
    }
#endif
    return m_pl_render_image(renderer, image, target, params);
}

void RGYLibplaceboLoader::frame_set_chroma_location(pl_frame *frame, pl_chroma_location chroma_location) const {
#if PL_API_VER >= 360
    if (m_apiVersion < 360) {
        auto legacy = *frame;
        legacy.repr.sys = color_system_to_legacy(legacy.repr.sys, m_apiVersion);
        m_pl_frame_set_chroma_location(&legacy, chroma_location);
        legacy.repr.sys = color_system_from_legacy(legacy.repr.sys, m_apiVersion);
        *frame = legacy;
        return;
    }
#endif
    m_pl_frame_set_chroma_location(frame, chroma_location);
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
