// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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

#include "rgy_model_registry.h"
#include "rgy_filesystem.h"
#include "rgy_util.h"
#include "../json/json.hpp"
#include <fstream>

using json = nlohmann::json;

RGY_ERR RGYModelRegistry::load(const tstring& jsonPath, std::shared_ptr<RGYLog> log) {
    if (!rgy_file_exists(jsonPath)) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("models.json not found: %s\n"), jsonPath.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    const auto [ret, dir] = PathRemoveFileSpecFixed(jsonPath);
    m_baseDir = dir;

    std::ifstream ifs(tchar_to_string(jsonPath));
    if (!ifs.is_open()) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to open models.json: %s\n"), jsonPath.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    json j;
    try {
        ifs >> j;
    } catch (const json::exception& e) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to parse models.json: %s\n"), char_to_tstring(e.what()).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (!j.contains("version") || j["version"].get<int>() != 1) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("models.json: unsupported version (expected 1)\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (!j.contains("models") || !j["models"].is_object()) {
        log->write(RGY_LOG_WARN, RGY_LOGT_VPP, _T("models.json: no models defined\n"));
        return RGY_ERR_NONE;
    }

    for (auto& [name, val] : j["models"].items()) {
        if (!val.contains("path") || !val["path"].is_string() || val["path"].get<std::string>().empty()) {
            log->write(RGY_LOG_WARN, RGY_LOGT_VPP, _T("models.json: model \"%s\" has no path, skipping\n"),
                char_to_tstring(name.c_str()).c_str());
            continue;
        }
        OnnxModelEntry entry;
        entry.path       = char_to_tstring(val["path"].get<std::string>().c_str());
        entry.colorspace = val.contains("colorspace") && val["colorspace"].is_string()
                            ? char_to_tstring(val["colorspace"].get<std::string>().c_str())
                            : _T("rgb");
        entry.noise      = val.contains("noise") && val["noise"].is_number_integer()
                            ? val["noise"].get<int>()
                            : 15;
        m_models[char_to_tstring(name.c_str())] = std::move(entry);
    }

    log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("models.json loaded: %zu model(s) from %s\n"),
        m_models.size(), jsonPath.c_str());
    return RGY_ERR_NONE;
}

std::optional<OnnxModelEntry> RGYModelRegistry::find(const tstring& name) const {
    auto it = m_models.find(name);
    if (it == m_models.end()) return std::nullopt;
    return it->second;
}

tstring RGYModelRegistry::resolveModelPath(const tstring& name) const {
    auto it = m_models.find(name);
    if (it == m_models.end()) return tstring();
    return PathCombineS(m_baseDir, it->second.path);
}
