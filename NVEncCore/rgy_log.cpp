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
// ------------------------------------------------------------------------------------------

#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include "rgy_log.h"
#include "rgy_version.h"
#include "rgy_util.h"
#include "rgy_def.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_filesystem.h"
#include "rgy_env.h"

int rgy_print_stderr(int log_level, const TCHAR *mes, void *handle_) {
    HANDLE handle = handle_;
#if defined(_WIN32) || defined(_WIN64)
    CONSOLE_SCREEN_BUFFER_INFO csbi = { 0 };
    static const WORD LOG_COLOR[] = {
        FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE, //水色
        FOREGROUND_INTENSITY | FOREGROUND_GREEN, //緑
        FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
        FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
        FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED, //黄色
        FOREGROUND_INTENSITY | FOREGROUND_RED //赤
    };
    if (handle == NULL) {
        handle = GetStdHandle(STD_ERROR_HANDLE);
    }
    if (handle && log_level != RGY_LOG_INFO) {
        GetConsoleScreenBufferInfo(handle, &csbi);
        SetConsoleTextAttribute(handle, LOG_COLOR[clamp(log_level, RGY_LOG_TRACE, RGY_LOG_ERROR) - RGY_LOG_TRACE] | (csbi.wAttributes & 0x00f0));
    }
    //このfprintfで"%"が消えてしまわないよう置換する
    int ret = _ftprintf(stderr, (nullptr == _tcschr(mes, _T('%'))) ? mes : str_replace(tstring(mes), _T("%"), _T("%%")).c_str());
    if (handle && log_level != RGY_LOG_INFO) {
        SetConsoleTextAttribute(handle, csbi.wAttributes); //元に戻す
    }
#else
    static const char *const LOG_COLOR[] = {
        "\x1b[36m", //水色
        "\x1b[32m", //緑
        "\x1b[39m", //デフォルト
        "\x1b[39m", //デフォルト
        "\x1b[33m", //黄色
        "\x1b[31m", //赤
    };
    int ret = _ftprintf(stderr, "%s%s%s", LOG_COLOR[clamp(log_level, RGY_LOG_TRACE, RGY_LOG_ERROR) - RGY_LOG_TRACE], mes, LOG_COLOR[RGY_LOG_INFO - RGY_LOG_TRACE]);
#endif //#if defined(_WIN32) || defined(_WIN64)
    fflush(stderr);
    return ret;
}

const char *RGYLog::HTML_FOOTER = "</body>\n</html>\n";

const TCHAR *rgy_log_level_to_str(RGYLogLevel level) {
    for (const auto& p : RGY_LOG_LEVEL_STR) {
        if (p.first == level) return p.second;
    }
    return nullptr;
}
const RGYLogLevel rgy_log_level_by_name(const TCHAR *type) {
    for (const auto& p : RGY_LOG_LEVEL_STR) {
        if (_tcscmp(p.second, type) == 0) return p.first;
    }
    return RGY_LOG_INFO;
}

const TCHAR *rgy_log_type_to_str(RGYLogType type) {
    for (const auto& p : RGY_LOG_TYPE_STR) {
        if (p.first == type) return p.second;
    }
    return nullptr;
}
const RGYLogType rgy_log_type_by_name(const TCHAR *type) {
    for (const auto& p : RGY_LOG_TYPE_STR) {
        if (_tcscmp(p.second, type) == 0) return p.first;
    }
    return RGY_LOGT_ALL;
}

RGYParamLogLevel::RGYParamLogLevel() :
    appcore_(RGY_LOG_INFO),
    appcoreprogress_(RGY_LOG_INFO),
    appcoreresult_(RGY_LOG_INFO),
    appcoreparallel_(RGY_LOG_INFO),
    appdevice_(RGY_LOG_INFO),
    appdecode_(RGY_LOG_INFO),
    appinput_(RGY_LOG_INFO),
    appoutput_(RGY_LOG_INFO),
    appvpp_(RGY_LOG_INFO),
    amf_(RGY_LOG_INFO),
    opencl_(RGY_LOG_INFO),
    libav_(RGY_LOG_INFO),
    libass_(RGY_LOG_INFO),
    perfmonitor_(RGY_LOG_INFO),
    caption2ass_(RGY_LOG_INFO)
{ };

RGYParamLogLevel::RGYParamLogLevel(const RGYLogLevel level) : RGYParamLogLevel() {
    set(level, RGY_LOGT_ALL);
};

bool RGYParamLogLevel::operator==(const RGYParamLogLevel &x) const {
    return appcore_ == x.appcore_
        && appcoreprogress_ == x.appcoreprogress_
        && appcoreresult_ == x.appcoreresult_
        && appcoreparallel_ == x.appcoreparallel_
        && appdevice_ == x.appdevice_
        && appdecode_ == x.appdecode_
        && appinput_ == x.appinput_
        && appoutput_ == x.appoutput_
        && appvpp_ == x.appvpp_
        && amf_ == x.amf_
        && opencl_ == x.opencl_
        && libav_ == x.libav_
        && libass_ == x.libass_
        && perfmonitor_ == x.perfmonitor_
        && caption2ass_ == x.caption2ass_;
}
bool RGYParamLogLevel::operator!=(const RGYParamLogLevel &x) const {
    return !(*this == x);
}

RGYLogLevel RGYParamLogLevel::set(const RGYLogLevel newLogLevel, const RGYLogType type) {
    RGYLogLevel prevLevel = RGY_LOG_INFO;
    switch (type) {
#define LOG_LEVEL_ADD_TYPE(TYPE, VAR) case (TYPE): { prevLevel = (VAR); (VAR) = newLogLevel; } break;
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_PROGRESS, appcoreprogress_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_RESULT, appcoreresult_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_PARALLEL, appcoreparallel_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_DEV,   appdevice_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_DEC,  appdecode_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_IN,    appinput_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_OUT,   appoutput_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_VPP,      appvpp_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_AMF,    amf_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_OPENCL,    opencl_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_LIBAV,        libav_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_LIBASS,       libass_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_PERF_MONITOR, perfmonitor_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CAPION2ASS,  caption2ass_);
#undef LOG_LEVEL_ADD_TYPE
    case RGY_LOGT_CORE: {
        prevLevel        = appcore_;
        appcore_         = newLogLevel;
        appcoreprogress_ = newLogLevel;
        appcoreresult_   = newLogLevel;
        appcoreparallel_ = newLogLevel;
        } break;
    case RGY_LOGT_APP: {
        prevLevel        = appcore_;
        appcore_         = newLogLevel;
        appcoreprogress_ = newLogLevel;
        appcoreresult_   = newLogLevel;
        appcoreparallel_ = newLogLevel;
        appdevice_       = newLogLevel;
        appdecode_       = newLogLevel;
        appinput_        = newLogLevel;
        appoutput_       = newLogLevel;
        appvpp_          = newLogLevel;
        opencl_          = newLogLevel;
        } break;
    case RGY_LOGT_ALL:
    default: {
        prevLevel        = appcore_;
        appcore_         = newLogLevel;
        appcoreprogress_ = newLogLevel;
        appcoreresult_   = newLogLevel;
        appcoreparallel_ = newLogLevel;
        appdevice_       = newLogLevel;
        appdecode_       = newLogLevel;
        appinput_        = newLogLevel;
        appoutput_       = newLogLevel;
        appvpp_          = newLogLevel;
        amf_             = newLogLevel;
        opencl_          = newLogLevel;
        libav_           = newLogLevel;
        libass_          = newLogLevel;
        perfmonitor_     = newLogLevel;
        caption2ass_     = newLogLevel;
        } break;
    }
    return prevLevel;
}

tstring RGYParamLogLevel::to_string() const {
    std::basic_stringstream<TCHAR> tmp;
    tmp << rgy_log_level_to_str(appcore_);
#define LOG_LEVEL_ADD_TYPE(TYPE, VAR) { if ((VAR) != appcore_) tmp << _T(",") << rgy_log_type_to_str(TYPE) << _T("=") << rgy_log_level_to_str(VAR); }
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_PROGRESS, appcoreprogress_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_RESULT, appcoreresult_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CORE_PARALLEL, appcoreparallel_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_DEV,   appdevice_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_DEC,  appdecode_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_IN,    appinput_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_OUT,   appoutput_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_VPP,      appvpp_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_AMF,    amf_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_OPENCL,    opencl_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_LIBAV,        libav_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_LIBASS,       libass_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_PERF_MONITOR, perfmonitor_);
    LOG_LEVEL_ADD_TYPE(RGY_LOGT_CAPION2ASS,  caption2ass_);
#undef LOG_LEVEL_ADD_TYPE
    return tmp.str();
}

RGYLog::RGYLog(const TCHAR *pLogFile, const RGYLogLevel log_level, bool showTime, bool addLogLevel) :
    m_nLogLevel(),
    m_pStrLog(),
    m_bHtml(false),
    m_showTime(showTime),
    m_addLogLevel(addLogLevel),
    m_mtx() {
    init(pLogFile, RGYParamLogLevel(log_level));
};

RGYLog::RGYLog(const TCHAR *pLogFile, const RGYParamLogLevel& log_level, bool showTime, bool addLogLevel) :
    m_nLogLevel(),
    m_pStrLog(),
    m_bHtml(false),
    m_showTime(showTime),
    m_addLogLevel(addLogLevel),
    m_mtx() {
    init(pLogFile, log_level);
}

RGYLog::~RGYLog() {
}

void RGYLog::init(const TCHAR *pLogFile, const RGYParamLogLevel& log_level) {
    m_nLogLevel = log_level;
    if (!m_mtx) {
        m_mtx = std::make_shared<std::mutex>();
    }
    if (pLogFile != nullptr && _tcslen(pLogFile) > 0) {
        m_pStrLog = pLogFile;
        CreateDirectoryRecursive(PathRemoveFileSpecFixed(pLogFile).second.c_str());
        FILE *fp = NULL;
        if (_tfopen_s(&fp, pLogFile, _T("a+")) || fp == NULL) {
            fprintf(stderr, "failed to open log file, log writing disabled.\n");
            pLogFile = nullptr;
        } else {
            if (check_ext(pLogFile, { ".html", ".htm" })) {
                _fseeki64(fp, 0, SEEK_SET);
                char buffer[1024] = { 0 };
                size_t file_read = fread(buffer, 1, sizeof(buffer)-1, fp);
                if (file_read == 0) {
                    m_bHtml = true;
                    writeHtmlHeader();
                } else {
                    std::transform(buffer, buffer + file_read, buffer, [](char in) -> char {return (char)tolower(in); });
                    if (strstr(buffer, "doctype") && strstr(buffer, "html")) {
                        m_bHtml = true;
                    }
                }
            }
            fclose(fp);
        }
    }
};

void RGYLog::writeHtmlHeader() {
    FILE *fp = NULL;
    if (_tfopen_s(&fp, m_pStrLog.c_str(), _T("wb"))) {
        std::wstring header =
            L"<!DOCTYPE html>\n"
            L"<html lang = \"ja\">\n"
            L"<head>\n"
            L"<meta charset = \"UTF-8\">\n"
            L"<title>" ENCODER_NAME " Log</title>\n"
            L"<style type=text/css>\n"
            L"   body   { \n"
            L"       background-color: #303030;\n"
            L"       line-height:1.0; font-family: \"MeiryoKe_Gothic\",\"遊ゴシック\",\"ＭＳ ゴシック\",sans-serif;\n"
            L"       margin: 10px;\n"
            L"       padding: 0px;\n"
            L"   }\n"
            L"   div {\n"
            L"       white-space: pre;\n"
            L"   }\n"
            L"   .error { color: #FA5858 }\n"
            L"   .warn  { color: #F7D358 }\n"
            L"   .more  { color: #CEF6F5 }\n"
            L"   .info  { color: #CEF6F5 }\n"
            L"   .debug { color: #ACFA58 }\n"
            L"   .trace { color: #ACFA58 }\n"
            L"</style>\n"
            L"</head>\n"
            L"<body>\n";
        RGY_DISABLE_WARNING_PUSH
        RGY_DISABLE_WARNING_STR("-Wnonnull")
        fprintf(fp, "%s", wstring_to_string(header, CP_UTF8).c_str());
        fprintf(fp, "%s", HTML_FOOTER);
        fclose(fp);
        RGY_DISABLE_WARNING_POP
    }
}

void RGYLog::writeFileHeader(const TCHAR *pDstFilename) {
    tstring fileHeader;
    int dstFilenameLen = (int)_tcslen(pDstFilename);
    static const TCHAR *const SEP5 = _T("-----");
    int sep_count = (std::max)(16, dstFilenameLen / 5 + 1);
    if (m_bHtml) {
        fileHeader += _T("<hr>");
    } else {
        for (int i = 0; i < sep_count; i++)
            fileHeader += SEP5;
    }
    fileHeader += _T("\n") + str_replace(tstring(pDstFilename), _T("%"), _T("%%")) + _T("\n");
    if (m_bHtml) {
        fileHeader += _T("<hr>");
    } else {
        for (int i = 0; i < sep_count; i++)
            fileHeader += SEP5;
    }
    fileHeader += _T("\n");
    write(RGY_LOG_INFO, RGY_LOGT_CORE, fileHeader.c_str());

    if (m_nLogLevel.get(RGY_LOGT_APP) <= RGY_LOG_DEBUG) {
        TCHAR cpuInfo[256] = { 0 };
        getCPUInfo(cpuInfo, _countof(cpuInfo));
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("%s    %s (%s)\n"), _T(ENCODER_NAME), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR);
#if defined(_WIN32) || defined(_WIN64)
        OSVERSIONINFOEXW osversioninfo = { 0 };
        tstring osversionstr = getOSVersion(&osversioninfo);
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("OS        %s %s (%d) [%s]\n"), osversionstr.c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber, getACPCodepageStr().c_str());
#else
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("OS        %s %s\n"), getOSVersion().c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"));
#endif
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("CPU Info  %s\n"), cpuInfo);
#if 0
        TCHAR gpu_info[1024] = { 0 };
        getGPUInfo(GPU_VENDOR, gpu_info, _countof(gpu_info));
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("GPU Info  %s\n"), gpu_info);
#endif //#if ENCODER_QSV
#if defined(_WIN32) || defined(_WIN64)
        write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("Locale    %s\n"), _tsetlocale(LC_ALL, nullptr));
#endif
    }
}
void RGYLog::writeFileFooter() {
    write(RGY_LOG_INFO, RGY_LOGT_CORE, _T("\n\n"));
}

void RGYLog::write_log(RGYLogLevel log_level, const RGYLogType logtype, const TCHAR *buffer, bool file_only) {
    if (log_level < m_nLogLevel.get(logtype)) {
        return;
    }

    auto convert_to_html = [log_level](std::string str) {
        //str = str_replace(str, "<", "&lt;");
        //str = str_replace(str, ">", "&gt;");
        //str = str_replace(str, "&", "&amp;");
        //str = str_replace(str, "\"", "&quot;");

        auto strLines = split(str, "\n");

        std::string strHtml;
        for (uint32_t i = 0; i < strLines.size() - 1; i++) {
            strHtml += strsprintf("<div class=\"%s\">", rgy_log_level_to_str(log_level));
            strHtml += strLines[i];
            strHtml += "</div>\n";
        }
        return strHtml;
    };
    auto add_time = [file_only](tstring str) {
        const auto tp = std::chrono::system_clock::now();
        const auto duration = tp.time_since_epoch();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        const auto sec1 = ms / 1000;
        const auto timeinfo = localtime(&sec1);
        TCHAR buf[64] = { 0 };
        _tcsftime(buf, _countof(buf), _T("[%Y-%m-%d %H:%M:%S"), timeinfo);
        tstring strWithTime = buf + strsprintf(_T(".%03d] "), ms - (sec1 * 1000));
        if (file_only) {
            // file_only の場合は分解するとおかしな出力になることがあるので途中の改行については無視して出力する
            return strWithTime + str;
        } else {
            const auto timeLength = strWithTime.length();

            auto strLines = split(str, _T("\n"));
            strWithTime.reserve(str.length() + strLines.size() * timeLength);

            strWithTime += strLines[0] + _T("\n");
            const auto blank = tstring(timeLength, _T(' '));
            for (uint32_t i = 1; i < strLines.size() - 1; i++) {
                strWithTime += blank + strLines[i] + _T("\n");
            }
        }
        return strWithTime;
    };

#if defined(_WIN32) || defined(_WIN64)
    HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
#else
    HANDLE hStdErr = NULL;
#endif //defined(_WIN32) || defined(_WIN64)

    tstring bufWithTime;
    if (m_showTime) {
        bufWithTime = add_time(buffer);
        buffer = bufWithTime.c_str();
    }
    std::string buffer_char;
#ifdef UNICODE
    char *buffer_ptr = NULL;
    DWORD mode = 0;
    bool stderr_write_to_console = 0 != GetConsoleMode(hStdErr, &mode); //stderrの出力先がコンソールかどうか
    if (m_pStrLog.length() > 0 || !stderr_write_to_console) {
        buffer_char = tchar_to_string(buffer, (m_bHtml) ? CP_UTF8 : CP_THREAD_ACP);
        if (m_bHtml) {
            buffer_char = convert_to_html(buffer_char);
        }
        buffer_ptr = &buffer_char[0];
    }
#else
    const char *buffer_ptr = &buffer[0];
    if (m_bHtml) {
        buffer_char = wstring_to_string(char_to_wstring(buffer_ptr), CP_UTF8);
        if (m_bHtml) {
            buffer_char = convert_to_html(buffer_char);
        }
        buffer_ptr = &buffer_char[0];
    }
#endif
    std::lock_guard<std::mutex> lock(*m_mtx.get());
    if (m_pStrLog.length() > 0) {
        FILE *fp_log = NULL;
        //logはANSI(まあようはShift-JIS)で保存する
        if (0 == _tfopen_s(&fp_log, m_pStrLog.c_str(), (m_bHtml) ? _T("rb+") : _T("a")) && fp_log) {
            if (m_bHtml) {
                _fseeki64(fp_log, 0, SEEK_END);
                int64_t pos = _ftelli64(fp_log);
                _fseeki64(fp_log, 0, SEEK_SET);
                _fseeki64(fp_log, pos -1 * strlen(HTML_FOOTER), SEEK_CUR);
            }
            fwrite(buffer_ptr, 1, strlen(buffer_ptr), fp_log);
            if (m_bHtml) {
                fwrite(HTML_FOOTER, 1, strlen(HTML_FOOTER), fp_log);
            }
            fclose(fp_log);
        }
    }
    if (!file_only) {
        if (m_addLogLevel) {
            auto strLines = split(buffer, _T("\n"));
            for (size_t i = 0; i < strLines.size(); i++) {
                const bool lastLine = i == strLines.size() - 1;
                if (lastLine && strLines[i].length() == 0) break;
                auto line = tstring(rgy_log_level_to_str(log_level)) + _T(":") + strLines[i] + _T("\n");
#ifdef UNICODE
                if (!stderr_write_to_console) { //出力先がリダイレクトされるならANSIで
                    buffer_char = wstring_to_string(tchar_to_wstring(line), CP_UTF8);
                    fprintf(stderr, buffer_ptr);
                }
                if (stderr_write_to_console) //出力先がコンソールならWCHARで
#endif
                    rgy_print_stderr(log_level, line.c_str(), hStdErr);
            }
        } else {
#ifdef UNICODE
            if (!stderr_write_to_console) //出力先がリダイレクトされるならANSIで
                fprintf(stderr, buffer_ptr);
            if (stderr_write_to_console) //出力先がコンソールならWCHARで
#endif
                rgy_print_stderr(log_level, buffer, hStdErr);
        }
    }
}

void RGYLog::write(RGYLogLevel log_level, const RGYLogType logtype, const wchar_t *format, va_list args) {
    if (log_level < m_nLogLevel.get(logtype)) {
        return;
    }

    int len = _vscwprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    std::vector<wchar_t> buffer(len, 0);
    if (buffer.data() != nullptr) {
        vswprintf_s(buffer.data(), len, format, args); // C4996
        write_log(log_level, logtype, wstring_to_tstring(buffer.data()).c_str());
    }
    va_end(args);
}

void RGYLog::write(RGYLogLevel log_level, const RGYLogType logtype, const char *format, va_list args, uint32_t codepage = CP_THREAD_ACP) {
    if (log_level < m_nLogLevel.get(logtype)) {
        return;
    }

    int len = _vscprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    std::vector<char> buffer(len, 0);
    if (buffer.data() != nullptr) {
        vsprintf_s(buffer.data(), len, format, args); // C4996
        write_log(log_level, logtype, char_to_tstring(buffer.data(), codepage).c_str());
    }
    va_end(args);
}

void RGYLog::write_line(RGYLogLevel log_level, const RGYLogType logtype, const char *format, va_list args, uint32_t codepage = CP_THREAD_ACP) {
    if (log_level < m_nLogLevel.get(logtype)) {
        return;
    }

    int len = _vscprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    std::vector<char> buffer(len, 0);
    if (buffer.data() != nullptr) {
        vsprintf_s(buffer.data(), len, format, args); // C4996
        tstring str = char_to_tstring(buffer.data(), codepage) + tstring(_T("\n"));
        write_log(log_level, logtype, str.c_str());
    }
    va_end(args);
}

void RGYLog::write(RGYLogLevel log_level, const RGYLogType logtype, const TCHAR *format, ...) {
    if (log_level < m_nLogLevel.get(logtype)) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    std::vector<TCHAR> buffer(len, 0);
    if (buffer.data() != nullptr) {
        _vstprintf_s(buffer.data(), len, format, args); // C4996
        write_log(log_level, logtype, buffer.data());
    }
    va_end(args);
}

