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

#include <algorithm>
#include <cctype>

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#include <limits.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBVMAF_FILENAME = _T("libvmaf.dll");
#elif LIBVMAF_STATIC_LINK
const TCHAR *RGY_LIBVMAF_FILENAME = _T("libvmaf (static)");
#else
const TCHAR *RGY_LIBVMAF_FILENAME = _T("libvmaf.so");
#endif

static RGYLibVMAFVersion rgy_libvmaf_version_class(const std::string& version, const std::string& modulePath, const bool hasVMAFOssExecAliases) {
    const auto soPos = modulePath.find(".so.");
    if (soPos != std::string::npos) {
        const auto majorPos = soPos + 4;
        size_t majorEnd = majorPos;
        while (majorEnd < modulePath.size() && std::isdigit((unsigned char)modulePath[majorEnd]) != 0) {
            majorEnd++;
        }
        if (majorEnd > majorPos) {
            const auto major = std::stoi(modulePath.substr(majorPos, majorEnd - majorPos));
            if (major >= 3) {
                return RGYLibVMAFVersion::V3_OR_LATER;
            }
            if (major == 2) {
                return RGYLibVMAFVersion::V2;
            }
        }
    }
    const auto firstDot = version.find('.');
    if (firstDot != std::string::npos) {
        const auto majorStr = version.substr(0, firstDot);
        if (!majorStr.empty() && std::all_of(majorStr.begin(), majorStr.end(), [](const char ch) { return std::isdigit((unsigned char)ch) != 0; })) {
            const auto major = std::stoi(majorStr);
            if (major >= 3) {
                return RGYLibVMAFVersion::V3_OR_LATER;
            }
            if (major == 2) {
                return RGYLibVMAFVersion::V2;
            }
        }
    }
    if (hasVMAFOssExecAliases) {
        return RGYLibVMAFVersion::V2;
    }
    if (!version.empty()) {
        return RGYLibVMAFVersion::V3_OR_LATER;
    }
    return RGYLibVMAFVersion::UNKNOWN;
}

RGYLibVMAFLoader::RGYLibVMAFLoader() :
    m_hModule(nullptr),
    m_loaded(false),
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
    m_vmaf_picture_unref(nullptr),
    m_vmaf_version(nullptr),
    m_vmaf_use_vmafossexec_aliases(nullptr),
    m_version(),
    m_versionClass(RGYLibVMAFVersion::UNKNOWN) {
}

RGYLibVMAFLoader::~RGYLibVMAFLoader() {
    close();
}

bool RGYLibVMAFLoader::load() {
    if (m_loaded) {
        return true;
    }

#if LIBVMAF_STATIC_LINK
    m_vmaf_init = &vmaf_init;
    m_vmaf_close = &vmaf_close;
    m_vmaf_use_features_from_model = &vmaf_use_features_from_model;
    m_vmaf_use_features_from_model_collection = &vmaf_use_features_from_model_collection;
    m_vmaf_use_feature = &vmaf_use_feature;
    m_vmaf_read_pictures = &vmaf_read_pictures;
    m_vmaf_score_pooled = &vmaf_score_pooled;
    m_vmaf_score_pooled_model_collection = &vmaf_score_pooled_model_collection;
    m_vmaf_model_load = &vmaf_model_load;
    m_vmaf_model_load_from_path = &vmaf_model_load_from_path;
    m_vmaf_model_destroy = &vmaf_model_destroy;
    m_vmaf_model_collection_load = &vmaf_model_collection_load;
    m_vmaf_model_collection_load_from_path = &vmaf_model_collection_load_from_path;
    m_vmaf_model_collection_destroy = &vmaf_model_collection_destroy;
    m_vmaf_picture_alloc = &vmaf_picture_alloc;
    m_vmaf_picture_unref = &vmaf_picture_unref;
    m_vmaf_version = &vmaf_version;
    m_vmaf_use_vmafossexec_aliases = nullptr;
    m_version = (m_vmaf_version != nullptr && m_vmaf_version() != nullptr) ? m_vmaf_version() : "";
    m_versionClass = rgy_libvmaf_version_class(m_version, std::string(), false);
    m_loaded = true;
    return true;
#else
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
    if (!loadFunc("vmaf_version", (void **)&m_vmaf_version)) { close(); return false; }

    m_vmaf_use_vmafossexec_aliases = RGY_GET_PROC_ADDRESS(m_hModule, "vmaf_use_vmafossexec_aliases");
    m_version = (m_vmaf_version != nullptr && m_vmaf_version() != nullptr) ? m_vmaf_version() : "";
    std::string modulePath;
#if defined(_WIN32) || defined(_WIN64)
    TCHAR modulePathBuf[4096] = { 0 };
    if (GetModuleFileName(m_hModule, modulePathBuf, _countof(modulePathBuf)) > 0) {
        modulePath = tchar_to_string(modulePathBuf);
    }
#else
    Dl_info info = { 0 };
    if (m_vmaf_init != nullptr && dladdr((void *)m_vmaf_init, &info) != 0 && info.dli_fname != nullptr) {
        modulePath = info.dli_fname;
        char resolvedPath[PATH_MAX] = { 0 };
        if (realpath(modulePath.c_str(), resolvedPath) != nullptr) {
            modulePath = resolvedPath;
        }
    }
#endif
    m_versionClass = rgy_libvmaf_version_class(m_version, modulePath, m_vmaf_use_vmafossexec_aliases != nullptr);
    m_loaded = true;
    return true;
#endif
}

void RGYLibVMAFLoader::close() {
#if !LIBVMAF_STATIC_LINK
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
#endif
    m_loaded = false;

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
    m_vmaf_version = nullptr;
    m_vmaf_use_vmafossexec_aliases = nullptr;
    m_version.clear();
    m_versionClass = RGYLibVMAFVersion::UNKNOWN;
}

#endif // ENABLE_VMAF
