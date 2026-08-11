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

#include "rgy_libass.h"

#if ENABLE_LIBASS_SUBBURN

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBASS_FILENAME = _T("libass-9.dll");
#elif LIBASS_STATIC_LINK
const TCHAR *RGY_LIBASS_FILENAME = _T("libass (static)");
#else
const TCHAR *RGY_LIBASS_FILENAME = _T("libass.so");
#endif

RGYLibassLoader::RGYLibassLoader() :
    m_hModule(nullptr),
    m_loaded(false),
    m_ass_library_init(nullptr),
    m_ass_library_done(nullptr),
    m_ass_set_message_cb(nullptr),
    m_ass_set_fonts_dir(nullptr),
    m_ass_add_font(nullptr),
    m_ass_set_extract_fonts(nullptr),
    m_ass_set_style_overrides(nullptr),
    m_ass_renderer_init(nullptr),
    m_ass_renderer_done(nullptr),
    m_ass_set_fonts(nullptr),
    m_ass_set_use_margins(nullptr),
    m_ass_set_font_scale(nullptr),
    m_ass_set_line_spacing(nullptr),
    m_ass_set_shaper(nullptr),
    m_ass_set_frame_size(nullptr),
    m_ass_set_storage_size(nullptr),
    m_ass_set_pixel_aspect(nullptr),
    m_ass_new_track(nullptr),
    m_ass_free_track(nullptr),
    m_ass_process_codec_private(nullptr),
    m_ass_process_chunk(nullptr),
    m_ass_render_frame(nullptr) {
}

RGYLibassLoader::~RGYLibassLoader() {
    close();
}

bool RGYLibassLoader::load() {
    if (m_loaded) {
        return true;
    }

#if LIBASS_STATIC_LINK
#define RGY_LIBASS_LOAD_FUNC(func) m_##func = &func
#else
    if ((m_hModule = RGY_LOAD_LIBRARY(RGY_LIBASS_FILENAME)) == nullptr) {
        return false;
    }
#define RGY_LIBASS_LOAD_FUNC(func) \
    if ((m_##func = reinterpret_cast<decltype(m_##func)>(RGY_GET_PROC_ADDRESS(m_hModule, #func))) == nullptr) { close(); return false; }
#endif

    RGY_LIBASS_LOAD_FUNC(ass_library_init);
    RGY_LIBASS_LOAD_FUNC(ass_library_done);
    RGY_LIBASS_LOAD_FUNC(ass_set_message_cb);
    RGY_LIBASS_LOAD_FUNC(ass_set_fonts_dir);
    RGY_LIBASS_LOAD_FUNC(ass_add_font);
    RGY_LIBASS_LOAD_FUNC(ass_set_extract_fonts);
    RGY_LIBASS_LOAD_FUNC(ass_set_style_overrides);
    RGY_LIBASS_LOAD_FUNC(ass_renderer_init);
    RGY_LIBASS_LOAD_FUNC(ass_renderer_done);
    RGY_LIBASS_LOAD_FUNC(ass_set_fonts);
    RGY_LIBASS_LOAD_FUNC(ass_set_use_margins);
    RGY_LIBASS_LOAD_FUNC(ass_set_font_scale);
    RGY_LIBASS_LOAD_FUNC(ass_set_line_spacing);
    RGY_LIBASS_LOAD_FUNC(ass_set_shaper);
    RGY_LIBASS_LOAD_FUNC(ass_set_frame_size);
    RGY_LIBASS_LOAD_FUNC(ass_set_storage_size);
    RGY_LIBASS_LOAD_FUNC(ass_set_pixel_aspect);
    RGY_LIBASS_LOAD_FUNC(ass_new_track);
    RGY_LIBASS_LOAD_FUNC(ass_free_track);
    RGY_LIBASS_LOAD_FUNC(ass_process_codec_private);
    RGY_LIBASS_LOAD_FUNC(ass_process_chunk);
    RGY_LIBASS_LOAD_FUNC(ass_render_frame);
#undef RGY_LIBASS_LOAD_FUNC

    m_loaded = true;
    return true;
}

void RGYLibassLoader::close() {
#if !LIBASS_STATIC_LINK
    if (m_hModule != nullptr) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
#endif
    m_loaded = false;
    m_ass_library_init = nullptr;
    m_ass_library_done = nullptr;
    m_ass_set_message_cb = nullptr;
    m_ass_set_fonts_dir = nullptr;
    m_ass_add_font = nullptr;
    m_ass_set_extract_fonts = nullptr;
    m_ass_set_style_overrides = nullptr;
    m_ass_renderer_init = nullptr;
    m_ass_renderer_done = nullptr;
    m_ass_set_fonts = nullptr;
    m_ass_set_use_margins = nullptr;
    m_ass_set_font_scale = nullptr;
    m_ass_set_line_spacing = nullptr;
    m_ass_set_shaper = nullptr;
    m_ass_set_frame_size = nullptr;
    m_ass_set_storage_size = nullptr;
    m_ass_set_pixel_aspect = nullptr;
    m_ass_new_track = nullptr;
    m_ass_free_track = nullptr;
    m_ass_process_codec_private = nullptr;
    m_ass_process_chunk = nullptr;
    m_ass_render_frame = nullptr;
}

#endif // ENABLE_LIBASS_SUBBURN
