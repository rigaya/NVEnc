// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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

#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "auo_frm.h"
#include "auo_util.h"
#include <mmsystem.h>
#ifdef _MSC_VER
#pragma comment(lib, "winmm.lib")
#endif

#if AVIUTL_TARGET_VER == 2
#include "logger2.h"
#endif

const int NEW_LINE_THRESHOLD = 125;
const int MAKE_NEW_LINE_THRESHOLD = 140;

#if AVIUTL_TARGET_VER == 2
static LOG_HANDLE *g_aviutl2_logger = nullptr;
static std::wstring g_log_cache;
static WindowTitleOverride *g_window_title_override = nullptr;

static bool get_window_text_timeout(HWND hwnd, std::wstring& out) {
    out.clear();
    if (!hwnd || !IsWindow(hwnd)) return false;

    DWORD_PTR text_len = 0;
    if (!SendMessageTimeoutW(hwnd, WM_GETTEXTLENGTH, 0, 0, SMTO_ABORTIFHUNG | SMTO_BLOCK, 200, &text_len)) {
        return false;
    }
    // WM_GETTEXTLENGTH は終端NULを含まない
    const size_t len = (size_t)text_len;
    std::wstring buf;
    buf.resize(len + 1, L'\0');

    DWORD_PTR copied = 0;
    if (!SendMessageTimeoutW(hwnd, WM_GETTEXT, (WPARAM)buf.size(), (LPARAM)buf.data(), SMTO_ABORTIFHUNG | SMTO_BLOCK, 200, &copied)) {
        return false;
    }
    // copied は終端NULを除いた文字数
    if (copied > 0) {
        buf.resize((size_t)copied);
        out.swap(buf);
    } else {
        out.clear();
    }
    return true;
}

static bool set_window_text_timeout(HWND hwnd, const wchar_t* text) {
    if (!hwnd || !IsWindow(hwnd)) return false;
    if (!text) text = L"";
    DWORD_PTR result = 0;
    return SendMessageTimeoutW(hwnd, WM_SETTEXT, 0, (LPARAM)text, SMTO_ABORTIFHUNG | SMTO_BLOCK, 200, &result) != 0;
}

struct FindMainWindowParam {
    DWORD pid;
    HWND best_hwnd;
    long long best_area;
};

static BOOL CALLBACK enum_windows_find_main(HWND hwnd, LPARAM lParam) {
    auto *param = reinterpret_cast<FindMainWindowParam*>(lParam);
    if (!param) return FALSE;

    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid != param->pid) return TRUE;

    if (!IsWindowVisible(hwnd)) return TRUE;
    if (GetWindow(hwnd, GW_OWNER) != nullptr) return TRUE;
    if (GetParent(hwnd) != nullptr) return TRUE;

    const LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    const LONG_PTR exstyle = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
    if ((style & WS_CAPTION) == 0) return TRUE;
    if (exstyle & WS_EX_TOOLWINDOW) return TRUE;

    RECT rc = { 0 };
    if (!GetWindowRect(hwnd, &rc)) return TRUE;
    const long long w = (long long)rc.right - (long long)rc.left;
    const long long h = (long long)rc.bottom - (long long)rc.top;
    if (w <= 0 || h <= 0) return TRUE;
    const long long area = w * h;

    // タイトルが取れない/ハングするような窓は除外
    std::wstring title;
    if (!get_window_text_timeout(hwnd, title)) return TRUE;

    if (area > param->best_area) {
        param->best_area = area;
        param->best_hwnd = hwnd;
    }
    return TRUE;
}

static HWND find_current_process_main_window() {
    // まずはフォアグラウンドウィンドウを優先
    {
        HWND fg = GetForegroundWindow();
        if (fg) {
            DWORD pid = 0;
            GetWindowThreadProcessId(fg, &pid);
            if (pid == GetCurrentProcessId()) {
                // enum_windows_find_main と同等の条件でチェック
                if (IsWindowVisible(fg)
                    && GetWindow(fg, GW_OWNER) == nullptr
                    && GetParent(fg) == nullptr) {
                    const LONG_PTR style = GetWindowLongPtrW(fg, GWL_STYLE);
                    const LONG_PTR exstyle = GetWindowLongPtrW(fg, GWL_EXSTYLE);
                    if ((style & WS_CAPTION) != 0 && (exstyle & WS_EX_TOOLWINDOW) == 0) {
                        std::wstring title;
                        if (get_window_text_timeout(fg, title)) {
                            return fg;
                        }
                    }
                }
            }
        }
    }

    FindMainWindowParam param = { 0 };
    param.pid = GetCurrentProcessId();
    param.best_hwnd = nullptr;
    param.best_area = -1;
    EnumWindows(enum_windows_find_main, (LPARAM)&param);
    return param.best_hwnd;
}

WindowTitleOverride::WindowTitleOverride() {
    hwnd = find_current_process_main_window();
    has_original_title = false;
    enc_start_time = 0;
    add_progress = false;
    using_afs = false;
    total_frame = 0;
    if (hwnd) {
        has_original_title = get_window_text_timeout(hwnd, original_title);
    }
}

WindowTitleOverride::~WindowTitleOverride() {
    if (hwnd && has_original_title) {
        set_window_text_timeout(hwnd, original_title.c_str());
    }
}

void WindowTitleOverride::override_window_title(const wchar_t *chr) {
    if (!hwnd || !IsWindow(hwnd)) {
        hwnd = find_current_process_main_window();
    }
    if (!hwnd) return;

    // 取得できていなければ、初回呼び出し時に元タイトルを保存しておく
    if (!has_original_title) {
        has_original_title = get_window_text_timeout(hwnd, original_title);
    }
    set_window_text_timeout(hwnd, chr);
}

void WindowTitleOverride::set_enc_info(BOOL afs, BOOL _add_progress, DWORD start_time, int _total_frame) {
    add_progress = _add_progress != 0;
    using_afs = afs != 0;
    enc_start_time = start_time;
    total_frame = _total_frame;
}

void WindowTitleOverride::clear_enc_info() {
    add_progress = false;
    using_afs = false;
    enc_start_time = 0;
    total_frame = 0;
}

static std::wstring format_eta_hhmmss(DWORD seconds) {
    const int hh = (int)(seconds / 3600);
    seconds -= (DWORD)hh * 3600;
    const int mm = (int)(seconds / 60);
    seconds -= (DWORD)mm * 60;
    const int ss = (int)seconds;
    wchar_t buf[64] = { 0 };
    swprintf_s(buf, _countof(buf), L"%02d:%02d:%02d", hh, mm, ss);
    return std::wstring(buf);
}

std::wstring WindowTitleOverride::format_window_title_enc_mes(const wchar_t *chr, int total_drop, int frame_n) const {
    std::wstring title = (chr) ? chr : L"";

    if (total_frame <= 0 || frame_n <= 0) {
        // 進捗計算に必要な情報が揃っていない場合は、従来同様にベース文字列のみ
        if (using_afs) {
            title += L", current afs ";
            title += std::to_wstring(total_drop);
            title += L"/";
            title += std::to_wstring(frame_n);
        }
        return title;
    }

    const double progress = frame_n / (double)total_frame;

    // time_elapsed は frmLog と同様に timeGetTime 基準の差分（ms）として扱う
    const DWORD time_elapsed = timeGetTime() - enc_start_time;

    if (using_afs) {
        std::wstring sb = title;
        sb += L", current afs ";
        sb += std::to_wstring(total_drop);
        sb += L"/";
        sb += std::to_wstring(frame_n);

        if (add_progress) {
            const int remain_frames = total_frame - frame_n;
            const DWORD time_remain = (remain_frames <= 0) ? 0 : (DWORD)(time_elapsed * ((double)remain_frames / (double)frame_n)) / 1000;
            if (!ENCODER_SVTAV1) {
                wchar_t percent_buf[64] = { 0 };
                swprintf_s(percent_buf, _countof(percent_buf), L"%.1f%%", progress * 100.0);
                sb.insert(0, std::wstring(L"[") + percent_buf + L"] ");
            }
            sb += L", eta ";
            sb += format_eta_hhmmss(time_remain);
        }
        return sb;
    }

    return title;
}

void set_aviutl2_logger(LOG_HANDLE *logger) {
    g_aviutl2_logger = logger;
    g_log_cache.clear();
    g_log_cache.resize(256 * 1024);
}

static bool call_logger(LOG_HANDLE *logger, void (*fn)(LOG_HANDLE *, LPCWSTR), const wchar_t *message) {
    if (logger && fn && message) {
        fn(logger, message);
        return true;
    }
    return false;
}

static void aviutl2_logger_output(int log_type_index, const wchar_t *message) {
    LOG_HANDLE *logger = g_aviutl2_logger;
    if (logger && message) {
        g_log_cache += std::wstring(message) + L"\n";
        switch (log_type_index) {
        case LOG_ERROR:
            if (call_logger(logger, logger->error, message)) return;
            break;
        case LOG_WARNING:
            if (call_logger(logger, logger->warn, message)) return;
            break;
        case LOG_INFO:
            if (call_logger(logger, logger->info, message)) return;
            break;
        default:
            if (call_logger(logger, logger->verbose, message)) return;
            break;
        }
        if (call_logger(logger, logger->log, message)) return;
        if (call_logger(logger, logger->info, message)) return;
    }
    if (message && message[0] != L'\0') {
        OutputDebugStringW(message);
        OutputDebugStringW(L"\r\n");
    }
}

void show_log_window(const TCHAR * /*aviutl_dir*/, BOOL /*disable_visual_styles*/) {
    // AviUtl2では標準のログウィンドウを利用するため何もしない
    g_log_cache.clear();
}

void set_window_title_override(WindowTitleOverride *window_title_override) {
    g_window_title_override = window_title_override;
}

void set_window_title(const wchar_t *chr) {
    if (g_window_title_override) {
        if (ENCODER_SVTAV1) {
            auto escaped_chr = removeAnsiEscapeSequences(std::wstring(chr));
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + escaped_chr).c_str());
        } else {
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + chr).c_str());
        }
    }
}

void set_window_title(const wchar_t *chr, int /*progress_mode*/) {
    if (g_window_title_override) {
        if (ENCODER_SVTAV1) {
            auto escaped_chr = removeAnsiEscapeSequences(std::wstring(chr));
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + escaped_chr).c_str());
        } else {
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + chr).c_str());
        }
    }
}

void set_window_title_enc_mes(const wchar_t *chr, int total_drop, int frame_n) {
    if (g_window_title_override) {  
        if (ENCODER_SVTAV1) {
            auto escaped_chr = removeAnsiEscapeSequences(std::wstring(chr));
            const auto title = g_window_title_override->format_window_title_enc_mes(escaped_chr.data(), total_drop, frame_n);
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + title).c_str());
        } else {
            const auto title = g_window_title_override->format_window_title_enc_mes(chr, total_drop, frame_n);
            g_window_title_override->override_window_title((std::wstring(AUO_NAME_WITHOUT_EXT_W) + title).c_str());
        }
    }
}

void set_task_name(const wchar_t * /*chr*/) {
}

void set_log_progress(double /*progress*/) {
}

void write_log_auo_line(int log_type_index, const wchar_t *chr) {
    aviutl2_logger_output(log_type_index, chr);
}

void write_log_line(int log_type_index, const wchar_t *chr) {
    aviutl2_logger_output(log_type_index, chr);
}

void flush_audio_log() {
}

void enable_enc_control(DWORD *priority, bool *enc_pause, BOOL afs, BOOL add_progress, DWORD start_time, int _total_frame) {
    (void)priority;
    (void)enc_pause;
    if (g_window_title_override) {
        g_window_title_override->set_enc_info(afs, add_progress, start_time, _total_frame);
    }
}

void disable_enc_control() {
    if (g_window_title_override) {
        g_window_title_override->clear_enc_info();
    }
}

void set_prevent_log_close(BOOL /*prevent*/) {
}

void auto_save_log_file(const TCHAR *log_filepath) {
    if (log_filepath) {
        std::unique_ptr<FILE, decltype(&fclose)> fp(nullptr, fclose);
        FILE *fpptr = nullptr;
        if (_wfopen_s(&fpptr, log_filepath, L"a") == 0) {
            fp = std::unique_ptr<FILE, decltype(&fclose)>(fpptr, fclose);
            auto log_utf8 = wstring_to_string(g_log_cache.c_str(), CP_UTF8);
            fwrite(log_utf8.c_str(), 1, log_utf8.size(), fp.get());
        }
    }
}

void log_process_events() {
}

int  get_current_log_len(bool /*first_pass*/) {
    return 0;
}

void log_reload_settings() {
}

void close_log_window() {
}

bool is_log_window_closed() {
    return true;
}
#endif // AVIUTL_TARGET_VER == 2

static inline int check_log_type(char *mes) {
    if (strstr(mes, "warning")) return LOG_WARNING;
    if (strstr(mes, "error")) return LOG_ERROR;
    return LOG_INFO;
}

static inline void add_line_to_cache(LOG_CACHE *cache_line, const char *mes) {
    if (cache_line->idx >= cache_line->max_line) {
        //メモリ不足なら再確保
        if (NULL != (cache_line->lines = (wchar_t **)realloc(cache_line->lines, sizeof(cache_line->lines[0]) * cache_line->max_line * 2))) {
            memset(&cache_line->lines[cache_line->max_line], 0, sizeof(cache_line->lines[0]) * cache_line->max_line);
            cache_line->max_line *= 2;
        }
    }
    if (cache_line->lines) {
        //一行のデータを格納
        const auto wmes = char_to_wstring(mes, CP_UTF8);
        wchar_t *line_ptr = (wchar_t *)malloc((wmes.length() + 1) * sizeof(line_ptr[0]));
        wcscpy_s(line_ptr, wmes.length() + 1, wmes.c_str());
        cache_line->lines[cache_line->idx] = line_ptr;
        cache_line->idx++;
    }
}

void release_log_cache(LOG_CACHE *log_cache) {
    if (log_cache && log_cache->lines) {
        for (int i = 0; i < log_cache->idx; i++)
            if (log_cache->lines[i]) free(log_cache->lines[i]);
        free(log_cache->lines);
        log_cache->lines = NULL;
        log_cache->idx = 0;
    }
}

//LOG_CACHEの初期化、成功->0, 失敗->1
int init_log_cache(LOG_CACHE *log_cache) {
    release_log_cache(log_cache);
    log_cache->idx = 0;
    log_cache->max_line = 64;
    return NULL == (log_cache->lines = (wchar_t **)calloc(log_cache->max_line, sizeof(log_cache->lines[0])));
}

//長すぎたら適当に折り返す
static int write_log_enc_mes_line(char *const mes, LOG_CACHE *cache_line) {
    const int mes_len = (int)strlen(mes);
    const int mes_type = check_log_type(mes);
    char *const fin = mes + mes_len;
    char *const prefix_ptr = strstr(mes, "]: ");
    const int prefix_len = (prefix_ptr) ? (int)(prefix_ptr - mes) + (int)strlen("]: ") : 0;
    char *p = mes, *q = NULL;
    BOOL flag_continue = FALSE;
    do {
        const int threshold = NEW_LINE_THRESHOLD - (p != mes) * prefix_len;
        flag_continue = mes_len >= MAKE_NEW_LINE_THRESHOLD
            && (p + threshold) < fin
            && (q = strrchr(p, ' ', threshold)) != NULL;
        if (flag_continue) *q = '\0';
        if (p != mes)
            for (char *const prefix_adjust = p - prefix_len; p > prefix_adjust; p--)
                *(p-1) = ' ';
        (cache_line) ? add_line_to_cache(cache_line, p) : write_log_line(mes_type, char_to_wstring(p, CP_UTF8).c_str());
        p=q+1;
    } while (flag_continue);
    return mes_len;
}

#if ENCODER_SVTAV1
void set_reconstructed_title_mes(const char *mes, int total_drop, int current_frames, int total_frames) {
    static std::chrono::system_clock::time_point last_update = std::chrono::system_clock::now();
    auto current = std::chrono::system_clock::now();
    if ((current - last_update) < std::chrono::milliseconds(300)) {
        return;
    }
    double fps = 0.0, bitrate = 0.0;
    int i_frame = 0;
    char buffer[1024] = { 0 };
    const char *ptr = buffer;
    last_update = current;
    if (sscanf_s(mes, "Encoding frame %d %lf kbps %lf fps", &i_frame, &bitrate, &fps) == 3
        || sscanf_s(mes, "Encoding frame %d %lf kbps %lf fpm", &i_frame, &bitrate, &fps) == 3) {
        const bool isfpm = strstr(mes, " fpm");
        sprintf_s(buffer, _countof(buffer),
            (isfpm) ? "[%3.1lf%%] %d/%d frames, %.3lf fps, %.2lf kb/s"
                    : "[%3.1lf%%] %d/%d frames, %.2lf fps, %.2lf kb/s",
            current_frames * 100.0 / (double)total_frames,
            current_frames,
            total_frames,
            isfpm ? fps * (1.0 / 60.0) : fps,
            bitrate);
    } else {
        ptr = mes;
    }
    set_window_title_enc_mes(char_to_wstring(ptr).c_str(), total_drop, current_frames);
}
#else
void set_reconstructed_title_mes(const char *mes, int total_drop, int current_frames, int total_frames) {
    double progress = 0, fps = 0, bitrate = 0;
    int i_frame = 0, total_frame = 0;
    int remain_time[3] = { 0 }, elapsed_time[3] = { 0 };
    char buffer[1024] = { 0 };
    int length = 0;
    const char *ptr = buffer;
    if ('[' == mes[0]
        && 11 >= sscanf_s(mes, "[%lf%%] %d/%d %lf %lf %d:%d:%d %d:%d:%d %n",
            &progress, &i_frame, &total_frame, &fps, &bitrate,
            &remain_time[0], &remain_time[1], &remain_time[2],
            &elapsed_time[0], &elapsed_time[1], &elapsed_time[2],
            &length)) {
        const char *qtr = mes + length;
        while (' ' == *qtr) qtr++;
        while (' ' != *qtr && '\0' != *qtr) qtr++;
        while (' ' == *qtr) qtr++;
        while (' ' != *qtr && '\0' != *qtr) qtr++;
        while (' ' == *qtr) qtr++;
        sprintf_s(buffer, _countof(buffer), "[%3.1lf%%] %d/%d frames, %.2lf fps, %.2lf kb/s, eta %d:%02d:%02d, %s %s",
            progress, i_frame, total_frame, fps, bitrate, elapsed_time[0], elapsed_time[1], elapsed_time[2], ('\0' != *qtr) ? "est.size" : "", qtr);
    } else if (3 == sscanf_s(mes, "%d %lf %lf", &i_frame, &fps, &bitrate)) {
        sprintf_s(buffer, _countof(buffer), "%d frames, %.2lf fps, %.2lf kb/s", i_frame, fps, bitrate);
    } else {
        ptr = mes;
    }
    set_window_title_enc_mes(char_to_wstring(ptr).c_str(), total_drop, current_frames);
}
#endif

void write_log_enc_mes(char *const msg, DWORD *log_len, int total_drop, int current_frames, int total_frames, LOG_CACHE *cache_line) {
    char *a, *b, *mes = msg;
    char *const fin = mes + *log_len; //null文字の位置
    *fin = '\0';
    while ((a = strchr(mes, '\n')) != NULL) {
        if ((b = strrchr(mes, '\r', (int)(a - mes) - 2)) != NULL)
            mes = b + 1;
        *a = '\0';
        write_log_enc_mes_line(mes, cache_line);
        mes = a + 1;
    }
    if ((a = strrchr(mes, '\r', (int)(fin - mes) - 1)) != NULL) {
        b = a - 1;
        while (*b == ' ' || *b == '\r')
            b--;
        *(b+1) = '\0';
        if ((b = strrchr(mes, '\r', (int)(b - mes) - 2)) != NULL)
            mes = b + 1;
#if ENCODER_SVTAV1
        if (strstr(mes, "Encoding frame")) {
#else
        if ((ENCODER_X264 || ENCODER_X265 || ENCODER_FFMPEG) && NULL == strstr(mes, "frames")) {
#endif
            set_reconstructed_title_mes(mes, total_drop, current_frames, total_frames);
        } else {
            set_window_title_enc_mes(char_to_wstring(mes).c_str(), total_drop, current_frames);
        }
        mes = a + 1;
    }
    if (!ENCODER_SVTAV1) {
        if (mes == msg && *log_len)
            mes += write_log_enc_mes_line(mes, NULL);
    }
    memmove(msg, mes, ((*log_len = (int)(fin - mes)) + 1) * sizeof(msg[0]));
}

void write_args(const TCHAR *args) {
    size_t len = _tcslen(args);
    TCHAR *const c = (TCHAR *)malloc((len+1)*sizeof(c[0]));
    TCHAR *const fin = c + len;
    memcpy(c, args, (len+1)*sizeof(c[0]));
    TCHAR *p = c;
    for (TCHAR *q = NULL; p + NEW_LINE_THRESHOLD < fin && (q = strrchr(p, _T(' '), NEW_LINE_THRESHOLD)) != NULL; p = q+1) {
        *q = '\0';
        write_log_line(LOG_INFO, p);
    }
    write_log_line(LOG_INFO, p);
    free(c);
}

void write_log_exe_mes(char *const msg, DWORD *log_len, const wchar_t *exename, LOG_CACHE *cache_line) {
    char *a, *b, *mes = msg;
    char * const fin = mes + *log_len; //null文字の位置
    wchar_t * buffer = NULL;
    DWORD buffer_len = 0;
    *fin = '\0';
    while ((a = strchr(mes, '\n')) != NULL) {
        if ((b = strrchr(mes, '\r', (int)(a - mes) - 2)) != NULL)
            mes = b + 1;
        *a = '\0';
        write_log_enc_mes_line(mes, cache_line);
        mes = a + 1;
    }
    if ((a = strrchr(mes, '\r', (int)(fin - mes) - 1)) != NULL) {
        b = a - 1;
        while (*b == ' ' || *b == '\r')
            b--;
        *(b+1) = '\0';
        if ((b = strrchr(mes, '\r', (int)(b - mes) - 2)) != NULL)
            mes = b + 1;
        if (exename) {
            if (buffer_len == 0) buffer_len = (*log_len * 3) + (int)wcslen(exename) + 3;
            if (buffer != NULL || NULL != (buffer = (wchar_t*)malloc(buffer_len * sizeof(buffer[0])))) {
                swprintf_s(buffer, buffer_len, L"%s: %s", exename, char_to_wstring(mes).c_str());
                set_window_title(buffer);
            }
        }
        mes = a + 1;
    }
    if (mes == msg && *log_len)
        mes += write_log_enc_mes_line(mes, cache_line);
    memmove(msg, mes, ((*log_len = (int)(fin - mes)) + 1) * sizeof(msg[0]));
    if (buffer) free(buffer);
}
