// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// ------------------------------------------------------------------------------------------

#include "rgy_vapoursynth_wrapper.h"
#include "rgy_log.h"
#include "rgy_util.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>
#endif

#if !defined(_WIN32) && !defined(_WIN64)
namespace {

bool vapoursynthModuleExistsInDir(const std::string& dir) {
    try {
        const auto path = std::filesystem::path(dir);
        if (!std::filesystem::is_directory(path)) {
            return false;
        }
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            const auto name = entry.path().filename().string();
            if (name.size() >= strlen("vapoursynth.so")
                && name.rfind("vapoursynth", 0) == 0
                && entry.path().extension() == ".so") {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool pathListContainsDir(const std::string& pathList, const std::string& dir) {
    size_t start = 0;
    while (start <= pathList.length()) {
        const auto end = pathList.find(':', start);
        const auto token = pathList.substr(start, (end == std::string::npos) ? std::string::npos : end - start);
        if (token == dir) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

std::vector<std::string> vapoursynthPythonPathCandidates() {
    std::vector<std::string> candidates;
    candidates.emplace_back("/usr/local/lib/python3/dist-packages");
    try {
        const auto base = std::filesystem::path("/usr/local/lib");
        if (std::filesystem::is_directory(base)) {
            for (const auto& entry : std::filesystem::directory_iterator(base)) {
                if (!entry.is_directory()) {
                    continue;
                }
                const auto name = entry.path().filename().string();
                if (name.rfind("python3.", 0) != 0) {
                    continue;
                }
                candidates.push_back((entry.path() / "dist-packages").string());
                candidates.push_back((entry.path() / "site-packages").string());
            }
        }
    } catch (...) {
        // ignore path discovery failure
    }
    return candidates;
}

} // namespace
#endif

void RGYPrepareVapourSynthPythonPath(RGYLog *log) {
#if !defined(_WIN32) && !defined(_WIN64)
    const auto currentPythonPath = std::getenv("PYTHONPATH");
    const std::string pythonPath = (currentPythonPath != nullptr) ? currentPythonPath : "";
    for (const auto& candidate : vapoursynthPythonPathCandidates()) {
        if (!vapoursynthModuleExistsInDir(candidate)) {
            continue;
        }
        if (pathListContainsDir(pythonPath, candidate)) {
            return;
        }
        const auto updated = pythonPath.empty() ? candidate : candidate + ":" + pythonPath;
        setenv("PYTHONPATH", updated.c_str(), 1);
        if (log) {
            log->write(RGY_LOG_DEBUG, RGY_LOGT_IN,
                _T("vpy: prepended VapourSynth Python path: %s.\n"),
                char_to_tstring(candidate).c_str());
        }
        return;
    }
    if (log && currentPythonPath == nullptr) {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_IN,
            _T("vpy: no additional VapourSynth Python module path detected.\n"));
    }
#else
    (void)log;
#endif
}

#if ENABLE_VAPOURSYNTH_READER

// Implemented in separate translation units to avoid header name collisions.
std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV4(const tstring& vsdir, RGYLog *log);
std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV3(const tstring& vsdir, RGYLog *log);

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapper(const tstring& vsdir, RGYLog *log) {
    if (auto v4 = CreateVapourSynthWrapperV4(vsdir, log)) {
        return v4;
    }
    return CreateVapourSynthWrapperV3(vsdir, log);
}

#else

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapper(const tstring&, RGYLog*) {
    return nullptr;
}

#endif


