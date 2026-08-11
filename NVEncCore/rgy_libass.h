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

#ifndef __RGY_LIBASS_H__
#define __RGY_LIBASS_H__

#include "rgy_version.h"

#if ENABLE_LIBASS_SUBBURN

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "ass/ass.h"

extern const TCHAR *RGY_LIBASS_FILENAME;

class RGYLibassLoader {
private:
    HMODULE m_hModule;
    bool m_loaded;

    decltype(&ass_library_init) m_ass_library_init;
    decltype(&ass_library_done) m_ass_library_done;
    decltype(&ass_set_message_cb) m_ass_set_message_cb;
    decltype(&ass_set_fonts_dir) m_ass_set_fonts_dir;
    decltype(&ass_add_font) m_ass_add_font;
    decltype(&ass_set_extract_fonts) m_ass_set_extract_fonts;
    decltype(&ass_set_style_overrides) m_ass_set_style_overrides;
    decltype(&ass_renderer_init) m_ass_renderer_init;
    decltype(&ass_renderer_done) m_ass_renderer_done;
    decltype(&ass_set_fonts) m_ass_set_fonts;
    decltype(&ass_set_use_margins) m_ass_set_use_margins;
    decltype(&ass_set_font_scale) m_ass_set_font_scale;
    decltype(&ass_set_line_spacing) m_ass_set_line_spacing;
    decltype(&ass_set_shaper) m_ass_set_shaper;
    decltype(&ass_set_frame_size) m_ass_set_frame_size;
    decltype(&ass_set_storage_size) m_ass_set_storage_size;
    decltype(&ass_set_pixel_aspect) m_ass_set_pixel_aspect;
    decltype(&ass_new_track) m_ass_new_track;
    decltype(&ass_free_track) m_ass_free_track;
    decltype(&ass_process_codec_private) m_ass_process_codec_private;
    decltype(&ass_process_chunk) m_ass_process_chunk;
    decltype(&ass_render_frame) m_ass_render_frame;

public:
    RGYLibassLoader();
    ~RGYLibassLoader();

    bool load();
    void close();
    bool loaded() const { return m_loaded; }

    auto p_ass_library_init() const { return m_ass_library_init; }
    auto p_ass_library_done() const { return m_ass_library_done; }
    auto p_ass_set_message_cb() const { return m_ass_set_message_cb; }
    auto p_ass_set_fonts_dir() const { return m_ass_set_fonts_dir; }
    auto p_ass_add_font() const { return m_ass_add_font; }
    auto p_ass_set_extract_fonts() const { return m_ass_set_extract_fonts; }
    auto p_ass_set_style_overrides() const { return m_ass_set_style_overrides; }
    auto p_ass_renderer_init() const { return m_ass_renderer_init; }
    auto p_ass_renderer_done() const { return m_ass_renderer_done; }
    auto p_ass_set_fonts() const { return m_ass_set_fonts; }
    auto p_ass_set_use_margins() const { return m_ass_set_use_margins; }
    auto p_ass_set_font_scale() const { return m_ass_set_font_scale; }
    auto p_ass_set_line_spacing() const { return m_ass_set_line_spacing; }
    auto p_ass_set_shaper() const { return m_ass_set_shaper; }
    auto p_ass_set_frame_size() const { return m_ass_set_frame_size; }
    auto p_ass_set_storage_size() const { return m_ass_set_storage_size; }
    auto p_ass_set_pixel_aspect() const { return m_ass_set_pixel_aspect; }
    auto p_ass_new_track() const { return m_ass_new_track; }
    auto p_ass_free_track() const { return m_ass_free_track; }
    auto p_ass_process_codec_private() const { return m_ass_process_codec_private; }
    auto p_ass_process_chunk() const { return m_ass_process_chunk; }
    auto p_ass_render_frame() const { return m_ass_render_frame; }
};

template<typename T, typename Func>
struct RGYLibassDeleter {
    RGYLibassDeleter(Func deleter = nullptr) : deleter(deleter) {}
    void operator()(T *ptr) const { if (ptr != nullptr && deleter != nullptr) deleter(ptr); }
    Func deleter;
};

using RGYLibassLibraryDeleter = RGYLibassDeleter<ASS_Library, decltype(&ass_library_done)>;
using RGYLibassRendererDeleter = RGYLibassDeleter<ASS_Renderer, decltype(&ass_renderer_done)>;
using RGYLibassTrackDeleter = RGYLibassDeleter<ASS_Track, decltype(&ass_free_track)>;

#endif // ENABLE_LIBASS_SUBBURN

#endif // __RGY_LIBASS_H__
