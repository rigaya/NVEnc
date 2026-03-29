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

#include "rgy_libvmaf.h"

#if ENABLE_VMAF

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBVMAF_FILENAME = _T("libvmaf.dll");
#else
const TCHAR *RGY_LIBVMAF_FILENAME = _T("libvmaf.so");
#endif

RGYLibVMAFLoader::RGYLibVMAFLoader() :
    m_hModule(nullptr),
    m_vmaf_init(nullptr),
    m_vmaf_close(nullptr),
    m_vmaf_use_features_from_model(nullptr),
    m_vmaf_use_features_from_model_collection(nullptr),
    m_vmaf_use_feature(nullptr),
    m_vmaf_read_pictures(nullptr),
    m_vmaf_score_pooled(nullptr),
    m_vmaf_score_pooled_model_collection(nullptr),
    m_vmaf_model_load(nullptr),
    m_vmaf_model_load_from_path(nullptr),
    m_vmaf_model_destroy(nullptr),
    m_vmaf_model_collection_load(nullptr),
    m_vmaf_model_collection_load_from_path(nullptr),
    m_vmaf_model_collection_destroy(nullptr),
    m_vmaf_picture_alloc(nullptr),
    m_vmaf_picture_unref(nullptr) {
}

RGYLibVMAFLoader::~RGYLibVMAFLoader() {
    close();
}

bool RGYLibVMAFLoader::load() {
    if (m_hModule) {
        return true;
    }

    if ((m_hModule = RGY_LOAD_LIBRARY(RGY_LIBVMAF_FILENAME)) == nullptr) {
        return false;
    }

    auto loadFunc = [this](const char *funcName, void **func) {
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
            return false;
        }
        return true;
    };

    if (!loadFunc("vmaf_init", (void **)&m_vmaf_init)) { close(); return false; }
    if (!loadFunc("vmaf_close", (void **)&m_vmaf_close)) { close(); return false; }
    if (!loadFunc("vmaf_use_features_from_model", (void **)&m_vmaf_use_features_from_model)) { close(); return false; }
    if (!loadFunc("vmaf_use_features_from_model_collection", (void **)&m_vmaf_use_features_from_model_collection)) { close(); return false; }
    if (!loadFunc("vmaf_use_feature", (void **)&m_vmaf_use_feature)) { close(); return false; }
    if (!loadFunc("vmaf_read_pictures", (void **)&m_vmaf_read_pictures)) { close(); return false; }
    if (!loadFunc("vmaf_score_pooled", (void **)&m_vmaf_score_pooled)) { close(); return false; }
    if (!loadFunc("vmaf_score_pooled_model_collection", (void **)&m_vmaf_score_pooled_model_collection)) { close(); return false; }
    if (!loadFunc("vmaf_model_load", (void **)&m_vmaf_model_load)) { close(); return false; }
    if (!loadFunc("vmaf_model_load_from_path", (void **)&m_vmaf_model_load_from_path)) { close(); return false; }
    if (!loadFunc("vmaf_model_destroy", (void **)&m_vmaf_model_destroy)) { close(); return false; }
    if (!loadFunc("vmaf_model_collection_load", (void **)&m_vmaf_model_collection_load)) { close(); return false; }
    if (!loadFunc("vmaf_model_collection_load_from_path", (void **)&m_vmaf_model_collection_load_from_path)) { close(); return false; }
    if (!loadFunc("vmaf_model_collection_destroy", (void **)&m_vmaf_model_collection_destroy)) { close(); return false; }
    if (!loadFunc("vmaf_picture_alloc", (void **)&m_vmaf_picture_alloc)) { close(); return false; }
    if (!loadFunc("vmaf_picture_unref", (void **)&m_vmaf_picture_unref)) { close(); return false; }

    return true;
}

void RGYLibVMAFLoader::close() {
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }

    m_vmaf_init = nullptr;
    m_vmaf_close = nullptr;
    m_vmaf_use_features_from_model = nullptr;
    m_vmaf_use_features_from_model_collection = nullptr;
    m_vmaf_use_feature = nullptr;
    m_vmaf_read_pictures = nullptr;
    m_vmaf_score_pooled = nullptr;
    m_vmaf_score_pooled_model_collection = nullptr;
    m_vmaf_model_load = nullptr;
    m_vmaf_model_load_from_path = nullptr;
    m_vmaf_model_destroy = nullptr;
    m_vmaf_model_collection_load = nullptr;
    m_vmaf_model_collection_load_from_path = nullptr;
    m_vmaf_model_collection_destroy = nullptr;
    m_vmaf_picture_alloc = nullptr;
    m_vmaf_picture_unref = nullptr;
}

#endif // ENABLE_VMAF
