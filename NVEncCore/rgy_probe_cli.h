#pragma once

#include <cstdio>
#include "rgy_tchar.h"
#include "rgy_version.h"

#if ENABLE_PROBE_BUILD
#include "../json/json.hpp"
#include "rgy_err.h"
#include "rgy_gpu_inventory.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace rgy_probe {

enum class Mode { report, devices, inventory };
using Json = nlohmann::json;

namespace detail {

struct State {
    std::mutex mutex;
    std::unique_ptr<FILE, int (*)(FILE *)> output{nullptr, std::fclose};
    Mode mode = Mode::report;
    Json build;
    Json backend_requested;
    Json backend_used;
    Json codec;
    Json device;
    Json stage;
    Json warnings = Json::array();
    Json failure;
    std::string last_error_log;
    uint64_t frames = 0;
    uint64_t bytes = 0;
    bool has_payload = false;
    bool finished = false;
};

inline State& state() {
    static State value;
    return value;
}

inline void write_line(FILE *output, Json line, const char *type) {
    line["protocol"] = 1;
    line["type"] = type;
    const auto text = line.dump(-1, ' ', false, Json::error_handler_t::replace);
    std::fwrite(text.data(), 1, text.size(), output);
    std::fputc('\n', output);
    std::fflush(output);
}

inline Json make_failure(Json stage, const char *reason, Json api = nullptr, Json code = nullptr,
                         Json message = nullptr, Json module = nullptr, Json symbol = nullptr) {
    return {
        {"stage", std::move(stage)}, {"reason", reason}, {"api", std::move(api)}, {"code", std::move(code)},
        {"message", std::move(message)}, {"module", std::move(module)}, {"symbol", std::move(symbol)}
    };
}

inline const char *reason_for(RGY_ERR error, const Json& stage) {
    switch (error) {
    case RGY_ERR_ACCESS_DENIED:
        return "permission_denied";
    case RGY_ERR_NO_DEVICE:
    case RGY_ERR_INVALID_DEVICE:
    case RGY_ERR_DEVICE_NOT_FOUND:
    case RGY_ERR_CUDA_ERROR_NO_DEVICE:
    case RGY_ERR_CUDA_ERROR_INVALID_DEVICE:
        return "device_not_visible";
    case RGY_ERR_INVALID_VERSION:
        return "driver_api_mismatch";
    case RGY_ERR_CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        return "symbol_missing";
    case RGY_ERR_INVALID_CODEC:
        return "unsupported_codec";
    default:
        break;
    }
    if (stage == "load_library") {
        return "library_load_failed";
    }
    if (stage == "session_open") {
        return "session_open_failed";
    }
    if (stage == "encode" || stage == "drain") {
        return "encode_failed";
    }
    return "encoder_init_failed";
}

inline void attach_error_log(Json& failure, const std::string& message) {
    if (failure.is_null() || message.empty()) {
        return;
    }
    auto& original = failure["message"];
    if (original.is_null() || original == "") {
        original = message;
    } else if (original != message) {
        failure["details"]["last_error_log"] = message;
    }
}

inline void finish_report_locked(State& s, Json failure) {
    attach_error_log(failure, s.last_error_log);
    const auto status = failure.is_null() ? Json("encoded") : failure.at("reason");
    s.finished = true;
    write_line(s.output.get(), {
        {"build", s.build}, {"codec", s.codec}, {"backend_requested", s.backend_requested}, {"backend_used", s.backend_used},
        {"device", s.device}, {"status", status}, {"frames", s.frames}, {"bytes", s.bytes},
        {"warnings", s.warnings}, {"failure", std::move(failure)}
    }, "result");
}

inline void finish_list_locked(State& s, Json devices, Json failure) {
    attach_error_log(failure, s.last_error_log);
    Json row = {{"build", s.build}, {"devices", std::move(devices)}, {"failure", std::move(failure)}};
    if (s.mode == Mode::devices) {
        row["backend"] = s.backend_requested;
    }
    s.finished = true;
    write_line(s.output.get(), std::move(row), s.mode == Mode::devices ? "devices" : "inventory");
}

#ifdef _WIN32
inline std::string wide_to_utf8(const wchar_t *text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return {};
    }
    std::string output(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, text, -1, output.data(), size, nullptr, nullptr) != size) {
        return {};
    }
    output.pop_back();
    return output;
}
#endif

} // namespace detail

inline std::string utf8(const TCHAR *text) {
    if (!text) {
        return {};
    }
#ifdef _WIN32
#ifdef _UNICODE
    return detail::wide_to_utf8(text);
#else
    const int size = MultiByteToWideChar(CP_ACP, 0, text, -1, nullptr, 0);
    if (size <= 0) {
        return {};
    }
    std::wstring wide(static_cast<size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_ACP, 0, text, -1, wide.data(), size) != size) {
        return {};
    }
    return detail::wide_to_utf8(wide.c_str());
#endif
#else
    return text;
#endif
}

// stderrの装飾を加える前に、最後のエラーログの文言を保持する。
inline void record_error_log(const TCHAR *text) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.output || s.finished) {
        return;
    }
    auto message = utf8(text);
    const auto end = message.find_last_not_of("\r\n");
    if (end != std::string::npos) {
        // JSONの1行上限を超えないよう、ログの長さを制限する。
        constexpr size_t max_error_log_bytes = 4096;
        message.resize(end >= max_error_log_bytes ? max_error_log_bytes : end + 1);
        s.last_error_log = std::move(message);
    }
}

inline Json identity_string(const std::optional<std::string>& value) {
    return value ? Json(*value) : Json(nullptr);
}

#if defined(_WIN32) && ENABLE_D3D11
using rgy_gpu_inventory::luid_string;

inline Json pci_bdf_from_luid(const LUID& luid) {
    return identity_string(rgy_gpu_inventory::pci_bdf_from_luid(luid));
}
#endif

#ifdef __linux__
inline Json pci_bdf_from_drm_fd(int fd) {
    return identity_string(rgy_gpu_inventory::pci_bdf_from_drm_fd(fd));
}

inline Json drm_render_node_from_fd(int fd) {
    return identity_string(rgy_gpu_inventory::drm_render_node_from_fd(fd));
}
#endif

inline bool start(Mode mode, const char *encoder, const char *version, const char *rev, const char *backend) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.output) {
        return false;
    }

    // 診断の応答を親だけが読めるように、複製した出力先は子プロセスへ引き継がない。
    std::fflush(stdout);
#ifdef _WIN32
    const int stdout_fd = _fileno(stdout);
    const int fd = _dup(stdout_fd);
    if (fd < 0) {
        return false;
    }
    if (!SetHandleInformation(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), HANDLE_FLAG_INHERIT, 0)
        || _setmode(fd, _O_BINARY) < 0) {
        _close(fd);
        return false;
    }
    auto *output = _fdopen(fd, "wb");
    if (!output) {
        _close(fd);
        return false;
    }
    if (_dup2(_fileno(stderr), stdout_fd) < 0) {
        std::fclose(output);
        return false;
    }
    // 標準入出力ライブラリを経由しない出力も、同じ診断ログの行き先に向ける。
    if (!SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(stdout_fd)))) {
        _dup2(fd, stdout_fd);
        std::fclose(output);
        return false;
    }
#else
    const int stdout_fd = fileno(stdout);
    const int fd = fcntl(stdout_fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    if (fd < 0) {
        return false;
    }
    auto *output = fdopen(fd, "w");
    if (!output) {
        close(fd);
        return false;
    }
    if (dup2(fileno(stderr), stdout_fd) < 0) {
        std::fclose(output);
        return false;
    }
#endif

    s.output.reset(output);
    s.mode = mode;
    const std::string revision = rev ? rev : "";
    s.build = {{"encoder", encoder}, {"version", version},
               {"rev", revision.empty() || revision.front() == 'r' ? revision : "r" + revision}};
    s.backend_requested = backend ? Json(backend) : Json(nullptr);
    s.backend_used = backend && std::strcmp(backend, "auto") != 0 ? Json(backend) : Json(nullptr);
    return true;
}

inline bool active() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.output != nullptr;
}

inline void event(const char *stage) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.output || s.finished || s.stage == stage) {
        return;
    }
    s.stage = stage;
    if (s.mode == Mode::report) {
        detail::write_line(s.output.get(), {{"stage", stage}}, "event");
    }
}

inline Json failure(const char *reason, Json api = nullptr, Json code = nullptr, Json message = nullptr,
                    Json module = nullptr, Json symbol = nullptr) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return detail::make_failure(s.stage, reason, std::move(api), std::move(code), std::move(message),
                                std::move(module), std::move(symbol));
}

inline void remember_failure(Json failure) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.output && !s.finished && s.failure.is_null()) {
        s.failure = std::move(failure);
    }
}

inline void clear_failure() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.finished) {
        s.failure = nullptr;
    }
}

inline Json get_failure() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.failure;
}

inline void set_backends(const char *requested, const char *used) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.backend_requested = requested ? Json(requested) : Json(nullptr);
    s.backend_used = used ? Json(used) : Json(nullptr);
}

inline void set_backend_used(const char *used) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.backend_used = used ? Json(used) : Json(nullptr);
}

inline void set_codec(const char *codec) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.codec = codec;
}

inline void set_device(Json device) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.device = std::move(device);
}

inline void set_encode_caps(Json& device, bool h264, bool hevc, bool av1) {
    device["encode"] = {{"h264", h264}, {"hevc", hevc}, {"av1", av1}};
    device["encode_failure"] = nullptr;
}

inline void set_encode_failure(Json& device, Json failure) {
    device["encode"] = nullptr;
    device["encode_failure"] = std::move(failure);
}

inline void add_output(uint64_t frames, uint64_t bytes, bool has_payload = true) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.frames += frames;
    s.bytes += bytes;
    s.has_payload |= has_payload && frames > 0 && bytes > 0;
}

inline void add_warning(Json warning) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.warnings.push_back(std::move(warning));
}

inline void add_warning(const char *api, int code, const TCHAR *message) {
    add_warning({{"api", api ? Json(api) : Json(nullptr)}, {"code", code}, {"message", message ? Json(utf8(message)) : Json(nullptr)}});
}

inline void finish_report(RGY_ERR error) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.output || s.finished || s.mode != Mode::report) {
        return;
    }
    if (s.failure.is_null() && error != RGY_ERR_NONE) {
        s.failure = detail::make_failure(s.stage, detail::reason_for(error, s.stage), nullptr,
                                         static_cast<int>(error), s.last_error_log.empty() ? utf8(get_err_mes(error)) : s.last_error_log);
    }
    if (s.failure.is_null() && !s.has_payload) {
        s.failure = detail::make_failure(s.stage, "no_output");
    }
    detail::finish_report_locked(s, std::move(s.failure));
}

inline void finish_list(Json devices, Json failure = nullptr) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.output && !s.finished && s.mode != Mode::report) {
        detail::finish_list_locked(s, std::move(devices), std::move(failure));
    }
}

#ifdef _WIN32
inline void finish_delay_load(bool load_library, unsigned long code, const char *module, Json symbol) {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.output || s.finished) {
        return;
    }
    auto failure = detail::make_failure(s.stage, load_library ? "library_missing" : "symbol_missing",
                                        load_library ? "LoadLibrary" : "GetProcAddress", code, nullptr, module, std::move(symbol));
    if (s.mode == Mode::report) {
        detail::finish_report_locked(s, std::move(failure));
    } else {
        detail::finish_list_locked(s, Json::array(), std::move(failure));
    }
}
#endif

} // namespace rgy_probe
#endif // ENABLE_PROBE_BUILD

// 診断用の縮小構成では、通常起動から省略された機能に到達しないようにする。
inline bool rgy_probe_cli_args(int& argc, TCHAR **argv) {
    if (argc == 2 && (_tcscmp(argv[1], _T("--version")) == 0 || _tcscmp(argv[1], _T("--help")) == 0)) {
        return true;
    }

    bool probe = false;
    int next = 1;
    for (int i = 1; i < argc; ++i) {
        if (_tcscmp(argv[i], _T("--probe-report")) == 0) {
            // 診断モードの印を除いて、エンコード引数を既存の解析処理に渡す。
            probe = true;
            continue;
        }
        if (_tcscmp(argv[i], _T("--probe-devices")) == 0 || _tcscmp(argv[i], _T("--probe-inventory")) == 0) {
            probe = true;
        }
        argv[next++] = argv[i];
    }
    argc = next;
    argv[argc] = nullptr;
    if (probe) {
        return true;
    }

    _ftprintf(stderr, _T("This binary is for GPU diagnostics only.\n"));
    return false;
}

#if ENABLE_PROBE_BUILD
inline int rgy_probe_main(int argc, TCHAR **argv, const char *encoder, const char *version,
                          const char *rev, const char *defaultBackend,
                          std::initializer_list<const char *> allowedBackends,
                          int (*inventory)(), int (*devices)(const char *backend), int (*report)(int, TCHAR **),
                          const char *defaultReportBackend = nullptr) {
    using rgy_probe::Mode;
    Mode mode = Mode::report;
    int modeCount = 0;
    for (int i = 1; i < argc; ++i) {
        if (_tcscmp(argv[i], _T("--probe-report")) == 0) {
            if (modeCount++ == 0) mode = Mode::report;
        } else if (_tcscmp(argv[i], _T("--probe-devices")) == 0) {
            if (modeCount++ == 0) mode = Mode::devices;
        } else if (_tcscmp(argv[i], _T("--probe-inventory")) == 0) {
            if (modeCount++ == 0) mode = Mode::inventory;
        }
    }
    if (modeCount == 0) {
        return rgy_probe_cli_args(argc, argv) ? report(argc, argv) : 1;
    }

    const char *devicesBackend = nullptr;
    if (mode == Mode::devices && modeCount == 1) {
        if (argc == 2) {
            devicesBackend = defaultBackend;
        } else if (argc == 4 && _tcscmp(argv[1], _T("--probe-devices")) == 0
            && _tcscmp(argv[2], _T("--backend")) == 0) {
            const auto requested = rgy_probe::utf8(argv[3]);
            for (const auto *backend : allowedBackends) {
                if (requested == backend) {
                    devicesBackend = backend;
                    break;
                }
            }
        }
    }
    const auto backend = mode == Mode::report && defaultReportBackend
        ? defaultReportBackend : devicesBackend ? devicesBackend : defaultBackend;
    if (!rgy_probe::start(mode, encoder, version, rev, backend)) {
        return 1;
    }
    const auto fail = [mode](const char *reason, const char *message) {
        auto failure = rgy_probe::failure(reason, nullptr, nullptr, message);
        if (mode == Mode::report) {
            rgy_probe::remember_failure(std::move(failure));
            rgy_probe::finish_report(RGY_ERR_UNKNOWN);
        } else {
            rgy_probe::finish_list(rgy_probe::Json::array(), std::move(failure));
        }
        return 1;
    };
    try {
        if (modeCount != 1) {
            return fail("protocol_error", "Exactly one probe mode is required.");
        }
        if (mode == Mode::inventory && inventory && argc == 2) {
            return inventory();
        }
        if (mode == Mode::devices && devices && devicesBackend) {
            return devices(devicesBackend);
        }
        if (mode != Mode::report || !rgy_probe_cli_args(argc, argv)) {
            return fail("protocol_error", "Invalid arguments for this probe mode.");
        }
        const auto ret = report(argc, argv);
        rgy_probe::finish_report(ret == 0 ? RGY_ERR_NONE : RGY_ERR_INVALID_PARAM);
        return ret == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        return fail("crash", e.what());
    } catch (...) {
        return fail("crash", "Unknown exception.");
    }
}
#endif
