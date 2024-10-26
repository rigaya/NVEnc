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

#ifndef __RGY_LIBPLACEBO_H__
#define __RGY_LIBPLACEBO_H__

#include "rgy_version.h"

#if ENABLE_LIBPLACEBO

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_prm.h"

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4819)
#include <libplacebo/dispatch.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders.h>
#include <libplacebo/shaders/custom.h>
#if ENABLE_D3D11
#include <libplacebo/d3d11.h>
#elif ENABLE_VULKAN
#include <libplacebo/vulkan.h>
#endif
#pragma warning (pop)

extern const TCHAR *RGY_LIBPLACEBO_DLL_NAME;

static const RGYLogType RGY_LOGT_LIBPLACEBO = RGY_LOGT_VPP;

class RGYLibplaceboLoader {
private:
    HMODULE m_hModule;

    pl_color_space *m_pl_color_space_bt2020_hlg;
    pl_color_space *m_pl_color_space_bt709;
    pl_color_space *m_pl_color_space_srgb;
    pl_color_space *m_pl_color_space_hdr10;
    pl_hdr_metadata *m_pl_hdr_metadata_empty;
    pl_peak_detect_params *m_pl_peak_detect_default_params;
    pl_color_map_params *m_pl_color_map_default_params;

#if ENABLE_D3D11
    decltype(&pl_d3d11_create) m_pl_d3d11_create;
    decltype(&pl_d3d11_destroy) m_pl_d3d11_destroy;
    decltype(&pl_d3d11_wrap) m_pl_d3d11_wrap;
#elif ENABLE_VULKAN
    decltype(&pl_vulkan_create) m_pl_vulkan_create;
    decltype(&pl_vulkan_destroy) m_pl_vulkan_destroy;
    decltype(&pl_vulkan_hold_ex) m_pl_vulkan_hold_ex;
    decltype(&pl_vulkan_release_ex) m_pl_vulkan_release_ex;
    decltype(&pl_vulkan_wrap) m_pl_vulkan_wrap;
#endif

    decltype(&pl_tex_destroy) m_pl_tex_destroy;
    decltype(&pl_tex_recreate) m_pl_tex_recreate;

    decltype(&pl_log_create) m_pl_log_create;
    decltype(&pl_log_destroy) m_pl_log_destroy;

    decltype(&pl_dispatch_create) m_pl_dispatch_create;
    decltype(&pl_dispatch_destroy) m_pl_dispatch_destroy;
    decltype(&pl_dispatch_begin) m_pl_dispatch_begin;
    decltype(&pl_dispatch_finish) m_pl_dispatch_finish;
    decltype(&pl_dispatch_abort) m_pl_dispatch_abort;

    decltype(&pl_renderer_create) m_pl_renderer_create;
    decltype(&pl_renderer_destroy) m_pl_renderer_destroy;
    decltype(&pl_render_image) m_pl_render_image;

    decltype(&pl_shader_sample_direct) m_pl_shader_sample_direct;
    decltype(&pl_shader_linearize) m_pl_shader_linearize;
    decltype(&pl_shader_sigmoidize) m_pl_shader_sigmoidize;
    decltype(&pl_shader_sample_polar) m_pl_shader_sample_polar;
    decltype(&pl_shader_sample_ortho2) m_pl_shader_sample_ortho2;
    decltype(&pl_shader_obj_destroy) m_pl_shader_obj_destroy;
    decltype(&pl_shader_reset) m_pl_shader_reset;
    decltype(&pl_shader_deband) m_pl_shader_deband;
    decltype(&pl_shader_dither) m_pl_shader_dither;

    decltype(&pl_find_filter_config) m_pl_find_filter_config;

    decltype(&pl_hdr_rescale) m_pl_hdr_rescale;

    decltype(&pl_lut_parse_cube) m_pl_lut_parse_cube;
    decltype(&pl_find_gamut_map_function) m_pl_find_gamut_map_function;
    decltype(&pl_raw_primaries_get) m_pl_raw_primaries_get;
    decltype(&pl_raw_primaries_merge) m_pl_raw_primaries_merge;
    decltype(&pl_color_space_infer_map) m_pl_color_space_infer_map;

    decltype(&pl_frame_set_chroma_location) m_pl_frame_set_chroma_location;

    decltype(&pl_shader_custom) m_pl_shader_custom;
    decltype(&pl_mpv_user_shader_parse) m_pl_mpv_user_shader_parse;
    decltype(&pl_mpv_user_shader_destroy) m_pl_mpv_user_shader_destroy;

public:
    RGYLibplaceboLoader();
    ~RGYLibplaceboLoader();

    bool load();
    void close();

    pl_color_space p_color_space_bt2020_hlg() const { return *m_pl_color_space_bt2020_hlg; }
    pl_color_space p_color_space_bt709() const { return *m_pl_color_space_bt709; }
    pl_color_space p_color_space_srgb() const { return *m_pl_color_space_srgb; }
    pl_color_space p_color_space_hdr10() const { return *m_pl_color_space_hdr10; }
    pl_hdr_metadata p_hdr_metadata_empty() const { return *m_pl_hdr_metadata_empty; }
    pl_peak_detect_params p_peak_detect_default_params() const { return *m_pl_peak_detect_default_params; }
    pl_color_map_params p_color_map_default_params() const { return *m_pl_color_map_default_params; }

#if ENABLE_D3D11
    auto p_d3d11_create() const { return m_pl_d3d11_create; }
    auto p_d3d11_destroy() const { return m_pl_d3d11_destroy; }
    auto p_d3d11_wrap() const { return m_pl_d3d11_wrap; }
#elif ENABLE_VULKAN
    auto p_vulkan_create() const { return m_pl_vulkan_create; }
    auto p_vulkan_destroy() const { return m_pl_vulkan_destroy; }
    auto p_vulkan_hold_ex() const { return m_pl_vulkan_hold_ex; }
    auto p_vulkan_release_ex() const { return m_pl_vulkan_release_ex; }
    auto p_vulkan_wrap() const { return m_pl_vulkan_wrap; }
#endif
    auto p_tex_destroy() const { return m_pl_tex_destroy; }
    auto p_tex_recreate() const { return m_pl_tex_recreate; }
    auto p_log_create() const { return m_pl_log_create; }
    auto p_log_destroy() const { return m_pl_log_destroy; }
    auto p_dispatch_create() const { return m_pl_dispatch_create; }
    auto p_dispatch_destroy() const { return m_pl_dispatch_destroy; }
    auto p_dispatch_begin() const { return m_pl_dispatch_begin; }
    auto p_dispatch_finish() const { return m_pl_dispatch_finish; }
    auto p_dispatch_abort() const { return m_pl_dispatch_abort; }
    auto p_renderer_create() const { return m_pl_renderer_create; }
    auto p_renderer_destroy() const { return m_pl_renderer_destroy; }
    auto p_render_image() const { return m_pl_render_image; }
    auto p_shader_sample_direct() const { return m_pl_shader_sample_direct; }
    auto p_shader_linearize() const { return m_pl_shader_linearize; }
    auto p_shader_sigmoidize() const { return m_pl_shader_sigmoidize; }
    auto p_shader_sample_polar() const { return m_pl_shader_sample_polar; }
    auto p_shader_sample_ortho2() const { return m_pl_shader_sample_ortho2; }
    auto p_shader_obj_destroy() const { return m_pl_shader_obj_destroy; }
    auto p_shader_reset() const { return m_pl_shader_reset; }
    auto p_shader_deband() const { return m_pl_shader_deband; }
    auto p_shader_dither() const { return m_pl_shader_dither; }
    auto p_find_filter_config() const { return m_pl_find_filter_config; }
    auto p_hdr_rescale() const { return m_pl_hdr_rescale; }
    auto p_lut_parse_cube() const { return m_pl_lut_parse_cube; }
    auto p_find_gamut_map_function() const { return m_pl_find_gamut_map_function; }
    auto p_raw_primaries_get() const { return m_pl_raw_primaries_get; }
    auto p_raw_primaries_merge() const { return m_pl_raw_primaries_merge; }
    auto p_color_space_infer_map() const { return m_pl_color_space_infer_map; }

    auto p_frame_set_chroma_location() const { return m_pl_frame_set_chroma_location; }

    auto p_shader_custom() const { return m_pl_shader_custom; }
    auto p_mpv_user_shader_parse() const { return m_pl_mpv_user_shader_parse; }
    auto p_mpv_user_shader_destroy() const { return m_pl_mpv_user_shader_destroy; }
};


template<typename T>
struct RGYLibplaceboDeleter {
    RGYLibplaceboDeleter() : deleter(nullptr) {};
    RGYLibplaceboDeleter(std::function<void(T*)> deleter) : deleter(deleter) {};
    void operator()(T p) { deleter(&p); }
    std::function<void(T*)> deleter;
};

struct RGYLibplaceboTexDeleter {
    RGYLibplaceboTexDeleter() : gpu(nullptr), deleter(nullptr) {};
    RGYLibplaceboTexDeleter(const RGYLibplaceboLoader *pl, pl_gpu gpu_) : gpu(gpu_), deleter(pl->p_tex_destroy()) {};
    void operator()(pl_tex p) { if (p && deleter) deleter(gpu, &p); }
    pl_gpu gpu;
    decltype(&pl_tex_destroy) deleter;
};

std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> rgy_pl_tex_recreate(const RGYLibplaceboLoader *pl, pl_gpu gpu, const pl_tex_params& tex_params);

MAP_PAIR_0_1_PROTO(loglevel, rgy, RGYLogLevel, libplacebo, pl_log_level);
MAP_PAIR_0_1_PROTO(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, libplacebo, const char*);
MAP_PAIR_0_1_PROTO(tone_map_metadata, rgy, VppLibplaceboToneMappingMetadata, libplacebo, pl_hdr_metadata_type);
MAP_PAIR_0_1_PROTO(transfer, rgy, CspTransfer, libplacebo, pl_color_transfer);
MAP_PAIR_0_1_PROTO(colorprim, rgy, CspColorprim, libplacebo, pl_color_primaries);
MAP_PAIR_0_1_PROTO(chromaloc, rgy, CspChromaloc, libplacebo, pl_chroma_location);

static void libplacebo_log_func(void *private_data, pl_log_level level, const char* msg) {
    auto log = static_cast<RGYLog*>(private_data);
    auto log_level = loglevel_libplacebo_to_rgy(level);
    if (log == nullptr || log_level < log->getLogLevel(RGY_LOGT_LIBPLACEBO)) {
        return;
    }
    log->write_log(log_level, RGY_LOGT_LIBPLACEBO, (tstring(_T("libplacebo: ")) + char_to_tstring(msg) + _T("\n")).c_str());
}

#endif // ENABLE_LIBPLACEBO

#endif // __RGY_LIBPLACEBO_H__
