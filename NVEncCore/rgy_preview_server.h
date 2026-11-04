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
#include "../json/json.hpp"
#include "rgy_util.h"
#include "rgy_err.h"
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

using RGYPreviewRunFunc = std::function<RGY_ERR(const std::vector<tstring>&, std::atomic<bool>*)>;
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

static inline int rgy_preview_server_main(RGYPreviewRunFunc run, const char *encoder, const char *encver, RGYPreviewProbeFunc probe = {}) {
#if defined(_WIN32)
    if (_setmode(_fileno(stdin), _O_BINARY) < 0 || _setmode(_fileno(stdout), _O_BINARY) < 0) return 1;
#endif
    struct Request { int64_t id = 0; std::vector<tstring> args; bool probe = false; };
    std::mutex mutex;
    std::condition_variable cv;
    // probeは到着順を保ち、未実行runだけを最新要求へ置換する。新runはそれ以前のprobeを追い越さない。
    std::deque<Request> pending;
    bool quitting = false, active = false;
    int64_t activeId = 0;
    std::atomic<bool> abort(false);
    fprintf(stdout, "RGYPREVIEW READY version=1 encoder=%s encver=%s%s\n", encoder, encver, probe ? " probe=1" : "");
    fflush(stdout);
    std::thread reader([&]() {
        std::string line;
        while (std::getline(std::cin, line)) {
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
            req.args.insert(req.args.end(), { _T("-c"), _T("raw"), _T("--y4m-timestamp"), _T("-o"), _T("-") });
            try { rc = run(req.args, &abort); }
            catch (...) { rc = RGY_ERR_UNKNOWN; }
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
