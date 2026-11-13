// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <deque>
#include <mutex>
#include <thread>
#include <iostream>
#include <utility>
#include <stdexcept>
#include "../json/json.hpp"
#include "rgy_input.h"
#include "rgy_util.h"
#include "rgy_err.h"
#include <cerrno>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

using RGYPreviewRunFunc = std::function<RGY_ERR(const std::vector<tstring>&, std::atomic<bool>*, bool, int64_t)>;
using RGYPreviewProbeFunc = std::function<RGY_ERR(const std::vector<tstring>&, bool&)>;

// 出力先の上書き、標準入力の横取り、制御用コマンドの混入を防ぐ。
static inline bool rgy_preview_valid_args(const std::vector<tstring>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        const auto name = a.substr(0, a.find(_T('=')));
        if (name == _T("-o") || name == _T("--output") || name == _T("-c")
            || name == _T("--codec") || name == _T("--output-format")
            || name == _T("--preview-server") || name == _T("--option-file")) return false;
        if ((name == _T("-i") || name == _T("--input"))
            && ((i + 1 < args.size() && args[i+1] == _T("-")) || a == name + _T("=-"))) return false;
        // 通常の表示専用コマンドはstdoutにプロトコル外の文字列を出力し得る。
        if (name == _T("--help") || name == _T("-h") || name == _T("--version")
            || name == _T("-v") || name.find(_T("--check-")) == 0) return false;
    }
    return true;
}

// 要求を読むためのfdをstdinから切り離し、fd 0はNUL(/dev/null)に差し替える。戻り値は要求を読むfd。
// Windowsでは、読み取りスレッドがパイプのfd 0で次の要求を待つあいだ、CRTがfd 0のロックを持ち続ける。
// vpy入力ではVSScriptがPythonを初期化してsys.stdinを作るときにfd 0へ触れるため、
// 次の要求行が届くまでrunが止まっていた(BEGINの後に何も出ない)。
// 要求用のfdをエンコーダのほかの処理から見えなくし、入力の読み込み側が要求のバイト列を読む事故も防ぐ。
// 差し替えに失敗したときは、従来どおりfd 0から読む。
static inline int rgy_preview_detach_stdin() {
#if defined(_WIN32)
    const int stdinFd = _fileno(stdin);
    const int requestFd = _dup(stdinFd);
    if (requestFd < 0) return stdinFd;
    const int nullFd = _open("NUL", _O_RDONLY | _O_BINARY);
    if (nullFd < 0) { _close(requestFd); return stdinFd; }
    const int ret = _dup2(nullFd, stdinFd);
    _close(nullFd);
    if (ret != 0) { _close(requestFd); return stdinFd; }
    // _dup2はfd 0のハンドルを差し替えるが、GetStdHandleで取る処理のために明示的にも合わせる。
    SetStdHandle(STD_INPUT_HANDLE, (HANDLE)_get_osfhandle(stdinFd));
    _setmode(requestFd, _O_BINARY);
    return requestFd;
#else
    const int requestFd = dup(STDIN_FILENO);
    if (requestFd < 0) return STDIN_FILENO;
    const int nullFd = open("/dev/null", O_RDONLY);
    if (nullFd < 0) { close(requestFd); return STDIN_FILENO; }
    const int ret = dup2(nullFd, STDIN_FILENO);
    close(nullFd);
    if (ret < 0) { close(requestFd); return STDIN_FILENO; }
    return requestFd;
#endif
}

// fdから1行読む。改行を含めずに返し、改行のない最後の行もstd::getlineと同じく1行として返す。
static inline bool rgy_preview_read_line(const int fd, std::string& buffer, std::string& line) {
    for (;;) {
        const auto pos = buffer.find('\n');
        if (pos != std::string::npos) {
            line.assign(buffer, 0, pos);
            buffer.erase(0, pos + 1);
            return true;
        }
        char chunk[4096];
#if defined(_WIN32)
        const int got = _read(fd, chunk, (unsigned int)sizeof(chunk));
#else
        const ssize_t got = ::read(fd, chunk, sizeof(chunk));
        if (got < 0 && errno == EINTR) continue;
#endif
        if (got <= 0) {
            if (buffer.empty()) return false;
            line.swap(buffer);
            buffer.clear();
            return true;
        }
        buffer.append(chunk, (size_t)got);
    }
}

static inline int rgy_preview_server_main(RGYPreviewRunFunc run, const char *encoder, const char *encver, RGYPreviewProbeFunc probe = {}, bool windowSeek = false, bool outputStart = false) {
#if defined(_WIN32)
    if (_setmode(_fileno(stdin), _O_BINARY) < 0 || _setmode(_fileno(stdout), _O_BINARY) < 0) return 1;
#endif
    const int requestFd = rgy_preview_detach_stdin();
    struct Request { int64_t id = 0; std::vector<tstring> args; bool probe = false; bool windowSeek = false; int64_t outputStartUs = INT64_MIN; };
    std::mutex mutex;
    std::condition_variable cv;
    // probeは到着順を保ち、未実行runだけを最新要求へ置換する。新runはそれ以前のprobeを追い越さない。
    std::deque<Request> pending;
    bool quitting = false, active = false;
    int64_t activeId = 0;
    std::atomic<bool> abort(false);
    fprintf(stdout, "RGYPREVIEW READY version=1 encoder=%s encver=%s%s%s%s\n", encoder, encver, probe ? " probe=1" : "", windowSeek ? " window_seek=1" : "", outputStart ? " output_start=1" : "");
    fflush(stdout);
    std::thread reader([&]() {
        std::string line, buffer;
        while (rgy_preview_read_line(requestFd, buffer, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            try {
                const auto json = nlohmann::json::parse(line);
                const auto cmd = json.at("cmd").get<std::string>();
                if (cmd == "quit") break;
                if (cmd != "run" && cmd != "cancel" && cmd != "probe") throw std::runtime_error("cmd");
                if (!json.at("id").is_number_integer()) throw std::runtime_error("id");
                Request req;
                req.id = json.at("id").get<int64_t>();
                req.probe = cmd == "probe";
                if (json.contains("window_seek") && !json.at("window_seek").is_boolean()) throw std::runtime_error("window_seek");
                req.windowSeek = windowSeek && json.value("window_seek", false);
                if (json.contains("output_start_us")) {
                    // Viewerが指定する出力開始時刻(µs、入力の絶対時刻)。これより前の出力はpipeへ書かない。
                    // 窓の読み込み(window_seek)を使うrunだけで受け付ける。窓を使わない読み込みでは、
                    // 出力の時刻が入力の絶対時刻にならないため比べられない。
                    // 型が違う値や、int64に収まらない値は、丸めずにエラーにする。
                    const auto& value = json.at("output_start_us");
                    if (cmd != "run" || !outputStart || !req.windowSeek || !value.is_number_integer()
                        || (value.is_number_unsigned() && value.get<uint64_t>() > INT64_MAX)) throw std::runtime_error("output_start_us");
                    req.outputStartUs = value.get<int64_t>();
                    // 出力側で許容幅を引いても桁あふれせず、「指定なし」を表すINT64_MINとも重ならない値だけ受け付ける。
                    // 0や負の値は、入力の時刻として有効なので受け付ける。
                    if (req.outputStartUs < INT64_MIN + RGY_PREVIEW_OUTPUT_START_TOLERANCE_US) throw std::runtime_error("output_start_us");
                }
                if (cmd == "run" || req.probe) {
                    if (!json.at("args").is_array()) throw std::runtime_error("args");
                    for (const auto& arg : json.at("args")) {
                        const auto str = arg.get<std::string>();
                        if (str.find('\0') != std::string::npos) throw std::runtime_error("NUL");
                        req.args.push_back(char_to_tstring(str, CP_UTF8));
                    }
                }
                std::lock_guard<std::mutex> lock(mutex);
                if (cmd == "run") {
                    pending.erase(std::remove_if(pending.begin(), pending.end(), [](const Request& item) { return !item.probe; }), pending.end());
                    pending.push_back(std::move(req));
                    if (active) abort.store(true);
                } else if (req.probe) {
                    // 進行中のrunは取り消さず、ENDの後に同じワーカースレッドで解析する。
                    pending.push_back(std::move(req));
                } else {
                    if (active && activeId == req.id) abort.store(true);
                    pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const Request& item) {
                        return !item.probe && item.id == req.id;
                    }), pending.end());
                }
                cv.notify_one();
            } catch (const std::exception&) {
                fprintf(stderr, "RGYPREVIEW: Failed to parse request\n");
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        quitting = true;
        pending.clear();
        abort.store(true);
        cv.notify_one();
    });
    for (;;) {
        Request req;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() { return quitting || !pending.empty(); });
            if (quitting) break;
            req = std::move(pending.front());
            pending.pop_front();
            active = !req.probe;
            activeId = req.id;
            abort.store(false);
        }
        if (req.probe) {
            bool history = true;
            RGY_ERR rc = RGY_ERR_INVALID_PARAM;
            if (probe && rgy_preview_valid_args(req.args)) {
                req.args.insert(req.args.end(), { _T("-c"), _T("raw"), _T("--y4m-timestamp"), _T("-o"), _T("-") });
                try { rc = probe(req.args, history); }
                catch (...) { rc = RGY_ERR_UNKNOWN; }
            }
            // 失敗時は履歴なしと誤認させず、応答自体は常に1行で完了する。
            fprintf(stdout, "RGYPREVIEW PROBE id=%lld history=%d rc=%d\n", (long long)req.id,
                rc == RGY_ERR_NONE && !history ? 0 : 1, (int)rc);
            fflush(stdout);
            continue;
        }
        fprintf(stdout, "RGYPREVIEW BEGIN id=%lld\n", (long long)req.id);
        fflush(stdout);
        fprintf(stderr, "RGYPREVIEW BEGIN id=%lld\n", (long long)req.id);
        RGY_ERR rc = RGY_ERR_INVALID_PARAM;
        if (rgy_preview_valid_args(req.args)) {
            try {
                req.args.insert(req.args.end(), { _T("-c"), _T("raw"), _T("--y4m-timestamp"), _T("-o"), _T("-") });
                rc = run(req.args, &abort, req.windowSeek, req.outputStartUs);
            } catch (...) { rc = RGY_ERR_UNKNOWN; }
        }
        fflush(stdout);
        fprintf(stdout, "RGYPREVIEW END id=%lld rc=%d\n", (long long)req.id, (int)rc);
        fflush(stdout);
        fprintf(stderr, "RGYPREVIEW END id=%lld rc=%d\n", (long long)req.id, (int)rc);
        {
            std::lock_guard<std::mutex> lock(mutex);
            active = false;
        }
    }
    reader.join();
    return 0;
}
