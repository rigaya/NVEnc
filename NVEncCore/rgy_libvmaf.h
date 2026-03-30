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

#ifndef __RGY_LIBVMAF_H__
#define __RGY_LIBVMAF_H__

#include "rgy_version.h"

#if ENABLE_VMAF

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_util.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4819)
#endif
extern "C" {
#include <libvmaf/libvmaf.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

extern const TCHAR *RGY_LIBVMAF_FILENAME;

enum class RGYLibVMAFVersion {
    UNKNOWN = 0,
    V2,
    V3_OR_LATER
};

class RGYLibVMAFLoader {
private:
    HMODULE m_hModule;
    bool m_loaded;

    decltype(&vmaf_init) m_vmaf_init;
    decltype(&vmaf_close) m_vmaf_close;
    decltype(&vmaf_use_features_from_model) m_vmaf_use_features_from_model;
    decltype(&vmaf_use_features_from_model_collection) m_vmaf_use_features_from_model_collection;
    decltype(&vmaf_use_feature) m_vmaf_use_feature;
    decltype(&vmaf_read_pictures) m_vmaf_read_pictures;
    decltype(&vmaf_score_pooled) m_vmaf_score_pooled;
    decltype(&vmaf_score_pooled_model_collection) m_vmaf_score_pooled_model_collection;
    decltype(&vmaf_model_load) m_vmaf_model_load;
    decltype(&vmaf_model_load_from_path) m_vmaf_model_load_from_path;
    decltype(&vmaf_model_destroy) m_vmaf_model_destroy;
    decltype(&vmaf_model_collection_load) m_vmaf_model_collection_load;
    decltype(&vmaf_model_collection_load_from_path) m_vmaf_model_collection_load_from_path;
    decltype(&vmaf_model_collection_destroy) m_vmaf_model_collection_destroy;
    decltype(&vmaf_picture_alloc) m_vmaf_picture_alloc;
    decltype(&vmaf_picture_unref) m_vmaf_picture_unref;
    decltype(&vmaf_version) m_vmaf_version;
    void *m_vmaf_use_vmafossexec_aliases;

    std::string m_version;
    RGYLibVMAFVersion m_versionClass;

public:
    RGYLibVMAFLoader();
    ~RGYLibVMAFLoader();

    bool load();
    void close();
    bool loaded() const { return m_loaded; }

    auto p_vmaf_init() const { return m_vmaf_init; }
    auto p_vmaf_close() const { return m_vmaf_close; }
    auto p_vmaf_use_features_from_model() const { return m_vmaf_use_features_from_model; }
    auto p_vmaf_use_features_from_model_collection() const { return m_vmaf_use_features_from_model_collection; }
    auto p_vmaf_use_feature() const { return m_vmaf_use_feature; }
    auto p_vmaf_read_pictures() const { return m_vmaf_read_pictures; }
    auto p_vmaf_score_pooled() const { return m_vmaf_score_pooled; }
    auto p_vmaf_score_pooled_model_collection() const { return m_vmaf_score_pooled_model_collection; }
    auto p_vmaf_model_load() const { return m_vmaf_model_load; }
    auto p_vmaf_model_load_from_path() const { return m_vmaf_model_load_from_path; }
    auto p_vmaf_model_destroy() const { return m_vmaf_model_destroy; }
    auto p_vmaf_model_collection_load() const { return m_vmaf_model_collection_load; }
    auto p_vmaf_model_collection_load_from_path() const { return m_vmaf_model_collection_load_from_path; }
    auto p_vmaf_model_collection_destroy() const { return m_vmaf_model_collection_destroy; }
    auto p_vmaf_picture_alloc() const { return m_vmaf_picture_alloc; }
    auto p_vmaf_picture_unref() const { return m_vmaf_picture_unref; }
    auto p_vmaf_version() const { return m_vmaf_version; }
    const std::string& version() const { return m_version; }
    RGYLibVMAFVersion version_class() const { return m_versionClass; }
};

#endif // ENABLE_VMAF

#endif // __RGY_LIBVMAF_H__
