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

#include <stdio.h>
#include <vector>
#include <numeric>
#include <limits>
#include <memory>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <filesystem>
#ifndef _MSC_VER
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <iconv.h>
#endif
#include "rgy_util.h"
#include "rgy_log.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_tchar.h"
#include "rgy_osdep.h"
#include "rgy_version.h"
#include "rgy_codepage.h"

#pragma warning (push)
#pragma warning (disable: 4100)
#if defined(_WIN32) || defined(_WIN64)
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    if (wstr == nullptr) {
        str = "";
        return 0;
    }
    uint32_t flags = (codepage == CP_UTF8) ? 0 : WC_NO_BEST_FIT_CHARS;
    int multibyte_length = WideCharToMultiByte(codepage, flags, wstr, -1, nullptr, 0, nullptr, nullptr);
    std::vector<char> tmp(multibyte_length, 0);
    if (0 == WideCharToMultiByte(codepage, flags, wstr, -1, tmp.data(), (int)tmp.size(), nullptr, nullptr)) {
        str.clear();
        return 0;
    }
    str = tmp.data();
    return multibyte_length;
}
#else
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    if (wstr == nullptr) {
        str = "";
        return 0;
    }
    auto codepage_str_to = codepage_str(codepage);
    if (codepage_str_to == nullptr) codepage_str_to = "UTF-8";
    auto ic = iconv_open(codepage_str_to, "wchar_t"); //to, from
    auto input_len = (wcslen(wstr)+1) * 4;
    std::vector<char> buf(input_len, 0);
    memcpy(buf.data(), wstr, input_len);
    auto output_len = input_len * 8;
    std::vector<char> bufout(output_len, 0);
    char *outbuf = bufout.data();
    char *input = buf.data();
    iconv(ic, &input, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    str = bufout.data();
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage) {
    if (wstr == nullptr) {
        return "";
    }
    std::string str;
    wstring_to_string(wstr, str, codepage);
    return str;
}

std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr.c_str(), str, codepage);
    return str;
}

unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage) {
#if UNICODE
    return wstring_to_string(tstr, str, codepage);
#else
    str = (tstr) ? std::string(tstr) : "";
    return (unsigned int)str.length();
#endif
}

std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage) {
    if (tstr == nullptr) {
        return "";
    }
    std::string str;
    tchar_to_string(tstr, str, codepage);
    return str;
}

std::wstring tchar_to_wstring(const tstring& tstr, uint32_t codepage) {
#if UNICODE
    return std::wstring(tstr);
#else
    return char_to_wstring(tstr, codepage);
#endif
}

std::wstring tchar_to_wstring(const TCHAR *tstr, uint32_t codepage) {
    if (tstr == nullptr) {
        return L"";
    }
    return tchar_to_wstring(tstring(tstr), codepage);
}

std::string tchar_to_string(const tstring& tstr, uint32_t codepage) {
    std::string str;
    tchar_to_string(tstr.c_str(), str, codepage);
    return str;
}

unsigned int wstring_to_tstring(const WCHAR *wstr, tstring& tstr, uint32_t codepage) {
    if (wstr == nullptr) {
        tstr = _T("");
        return 0;
    }
#if UNICODE
    tstr = std::wstring(wstr);
#else
    return wstring_to_string(wstr, tstr, codepage);
#endif
    return (unsigned int)tstr.length();
}

tstring wstring_to_tstring(const WCHAR *wstr, uint32_t codepage) {
    if (wstr == nullptr) {
        return _T("");
    }
    tstring tstr;
    wstring_to_tstring(wstr, tstr, codepage);
    return tstr;
}

tstring wstring_to_tstring(const std::wstring& wstr, uint32_t codepage) {
    tstring tstr;
    wstring_to_tstring(wstr.c_str(), tstr, codepage);
    return tstr;
}

#if defined(_WIN32) || defined(_WIN64)
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    int widechar_length = MultiByteToWideChar(codepage, 0, str, -1, nullptr, 0);
    std::vector<wchar_t> tmp(widechar_length, 0);
    if (0 == MultiByteToWideChar(codepage, 0, str, -1, tmp.data(), (int)tmp.size())) {
        wstr.clear();
        return 0;
    }
    wstr = tmp.data();
    return widechar_length;
}
unsigned int char_to_string(std::string& dst, uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    if (src == nullptr) {
        dst = "";
        return 0;
    }
    if (codepage_to == codepage_from) {
        dst = src;
        return (unsigned int)dst.length();
    }
    std::wstring wstrtemp;
    char_to_wstring(wstrtemp, src, codepage_from);
    wstring_to_string(wstrtemp.c_str(), dst, codepage_to);
    return (unsigned int)dst.length();
}
#else
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    auto codepage_str_from = codepage_str(codepage);
    if (codepage_str_from == nullptr) codepage_str_from = "UTF-8";
    auto ic = iconv_open("wchar_t", codepage_str_from); //to, from
    if ((int64_t)ic == -1) {
        fprintf(stderr, "iconv_error\n");
    }
    auto input_len = strlen(str)+1;
    std::vector<char> buf(input_len);
    strcpy(buf.data(), str);
    auto output_len = (input_len + 1) * 8;
    std::vector<char> bufout(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = bufout.data();
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    wstr = std::wstring((WCHAR *)bufout.data());
    return wstr.length();
}

unsigned int char_to_string(std::string& dst, uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    if (src == nullptr) {
        dst = "";
        return 0;
    }
    auto codepage_str_from = codepage_str(codepage_from);
    if (codepage_str_from == nullptr) codepage_str_from = "UTF-8";
    auto codepage_str_to = codepage_str(codepage_to);
    if (codepage_str_to == nullptr) codepage_str_to = "UTF-8";
    if (codepage_to == codepage_from
        || strcmp(codepage_str_to, codepage_str_from) == 0) {
        dst = src;
        return dst.length();
    }
    auto ic = iconv_open(codepage_str_to, codepage_str_from); //to, from
    if ((int64_t)ic == -1) {
        fprintf(stderr, "iconv_error\n");
    }
    auto input_len = strlen(src)+1;
    std::vector<char> buf(input_len);
    strcpy(buf.data(), src);
    auto output_len = (input_len + 1) * 12;
    std::vector<char> bufout(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = bufout.data();
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    iconv_close(ic);
    dst = std::string(bufout.data());
    return dst.length();
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::wstring char_to_wstring(const char *str, uint32_t codepage) {
    if (str == nullptr) {
        return L"";
    }
    std::wstring wstr;
    char_to_wstring(wstr, str, codepage);
    return wstr;
}
std::wstring char_to_wstring(const std::string& str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str.c_str(), codepage);
    return wstr;
}
std::string char_to_string(uint32_t codepage_to, const char *src, uint32_t codepage_from) {
    std::string dst;
    char_to_string(dst, codepage_to, src, codepage_from);
    return dst;
}

unsigned int char_to_tstring(tstring& tstr, const char *str, uint32_t codepage) {
#if UNICODE
    return char_to_wstring(tstr, str, codepage);
#else
    tstr = (str) ? std::string(str) : _T("");
    return (unsigned int)tstr.length();
#endif
}

tstring char_to_tstring(const char *str, uint32_t codepage) {
    if (str == nullptr) {
        return _T("");
    }
    tstring tstr;
    char_to_tstring(tstr, str, codepage);
    return tstr;
}
tstring char_to_tstring(const std::string& str, uint32_t codepage) {
    tstring tstr;
    char_to_tstring(tstr, str.c_str(), codepage);
    return tstr;
}
std::string strsprintf(const char* format, ...) {
    if (format == nullptr) {
        return "";
    }
    va_list args;
    va_start(args, format);
    const size_t len = _vscprintf(format, args) + 1;

    std::vector<char> buffer(len, 0);
    vsprintf(buffer.data(), format, args);
    va_end(args);
    std::string retStr = std::string(buffer.data());
    return retStr;
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring strsprintf(const WCHAR* format, ...) {
    if (format == nullptr) {
        return L"";
    }
    va_list args;
    va_start(args, format);
    const size_t len = _vscwprintf(format, args) + 1;

    std::vector<WCHAR> buffer(len, 0);
    vswprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);
    std::wstring retStr = std::wstring(buffer.data());
    return retStr;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string str_replace(std::string str, const std::string& from, const std::string& to) {
    std::string::size_type pos = 0;
    while(pos = str.find(from, pos), pos != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}

#if defined(_WIN32) || defined(_WIN64)
std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to) {
    std::wstring::size_type pos = 0;
    while (pos = str.find(from, pos), pos != std::wstring::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#pragma warning (pop)
#if defined(_WIN32) || defined(_WIN64)
std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim, bool bTrim) {
    std::vector<std::wstring> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::wstring::npos != (found = str.find(delim, current))) {
        auto segment = std::wstring(str, current, found - current);
        if (bTrim) {
            segment = trim(segment);
        }
        if (!bTrim || segment.length()) {
            res.push_back(segment);
        }
        current = found + delimlen;
    }
    auto segment = std::wstring(str, current, str.size() - current);
    if (bTrim) {
        segment = trim(segment);
    }
    if (!bTrim || segment.length()) {
        res.push_back(std::wstring(segment.c_str()));
    }
    return res;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::vector<std::string> split(const std::string &str, const std::string &delim, bool bTrim) {
    std::vector<std::string> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::string::npos != (found = str.find(delim, current))) {
        auto segment = std::string(str, current, found - current);
        if (bTrim) {
            segment = trim(segment);
        }
        if (!bTrim || segment.length()) {
            res.push_back(segment);
        }
        current = found + delimlen;
    }
    auto segment = std::string(str, current, str.size() - current);
    if (bTrim) {
        segment = trim(segment);
    }
    if (!bTrim || segment.length()) {
        res.push_back(std::string(segment.c_str()));
    }
    return res;
}

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::wstring> sep_cmd(const std::wstring& cmd) {
    std::vector<std::wstring> args;
    int argc = 0;
    auto ptr = CommandLineToArgvW(cmd.c_str(), &argc);
    for (int i = 0; i < argc; i++) {
        args.push_back(ptr[i]);
    }
    args.push_back(L"");
    LocalFree(ptr);
    return std::move(args);
}

std::vector<std::string> sep_cmd(const std::string& cmd) {
    std::vector<std::string> args;
    std::wstring wcmd = char_to_wstring(cmd);
    for (const auto &warg : sep_cmd(wcmd)) {
        args.push_back(wstring_to_string(warg));
    }
    return std::move(args);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string lstrip(const std::string& string, const char* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        result = string.substr(left, 0);
    }
    return result;
}

std::string rstrip(const std::string& string, const char* trim) {
    auto result = string;
    auto right = string.find_last_not_of(trim);
    if (right != std::string::npos) {
        result = string.substr(0, right);
    }
    return result;
}

std::string trim(const std::string& string, const char* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        auto right = string.find_last_not_of(trim);
        result = string.substr(left, right - left + 1);
    }
    return result;
}

std::wstring lstrip(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        result = string.substr(left, 0);
    }
    return result;
}

std::wstring rstrip(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto right = string.find_last_not_of(trim);
    if (right != std::string::npos) {
        result = string.substr(0, right);
    }
    return result;
}

std::wstring trim(const std::wstring& string, const WCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        auto right = string.find_last_not_of(trim);
        result = string.substr(left, right - left + 1);
    }
    return result;
}

std::string GetFullPath(const char *path) {
    return std::filesystem::absolute(std::filesystem::path(strlen(path) ? path : ".")).lexically_normal().string();
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring GetFullPath(const WCHAR *path) {
    return std::filesystem::absolute(std::filesystem::path(wcslen(path) ? path : L".")).lexically_normal().wstring();
}
//ルートディレクトリを取得
std::string PathGetRoot(const char *path) {
    return std::filesystem::path(GetFullPath(path)).root_name().string();
}
std::wstring PathGetRoot(const WCHAR *path) {
    return std::filesystem::path(GetFullPath(path)).root_name().wstring();
}

//パスのルートが存在するかどうか
static bool PathRootExists(const char *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}
static bool PathRootExists(const WCHAR *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path) {
    const auto newPath = std::filesystem::path(path).remove_filename().string();
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path) {
    const auto newPath = std::filesystem::path(path).remove_filename().wstring();
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::string PathRemoveExtensionS(const std::string& path) {
    const auto lastdot = path.find_last_of(".");
    if (lastdot == std::string::npos) return path;
    return path.substr(0, lastdot);
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring PathRemoveExtensionS(const std::wstring& path) {
    const auto lastdot = path.find_last_of(L".");
    if (lastdot == std::string::npos) return path;
    return path.substr(0, lastdot);
}
std::string PathCombineS(const std::string& dir, const std::string& filename) {
    return std::filesystem::path(dir).append(filename).string();
}
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename) {
    return std::filesystem::path(dir).append(filename).wstring();
}
#endif //#if defined(_WIN32) || defined(_WIN64)
//フォルダがあればOK、なければ作成する
bool CreateDirectoryRecursive(const char *dir) {
    auto targetDir = std::filesystem::path(strlen(dir) ? dir : ".");
    if (std::filesystem::exists(targetDir)) {
        return true;
    }
    return std::filesystem::create_directories(targetDir);
}
#if defined(_WIN32) || defined(_WIN64)
bool CreateDirectoryRecursive(const WCHAR *dir) {
    auto targetDir = std::filesystem::path(wcslen(dir) ? dir : L".");
    if (std::filesystem::exists(targetDir)) {
        return true;
    }
    return std::filesystem::create_directories(targetDir);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list) {
    const auto target = tolowercase(std::filesystem::path(filename).extension().string());
    if (target.length() > 0) {
        for (auto ext : ext_list) {
            if (target == tolowercase(ext)) {
                return true;
            }
        }
    }
    return false;
}

struct RGYSIPrefix {
    char prefix;
    bool inverse;
    int64_t pow2;
    int64_t pow10;

    RGYSIPrefix(char prefix_, bool inverse_, int64_t pow2_, int64_t pow10_) :
        prefix(prefix_), inverse(inverse_), pow2(pow2_), pow10(pow10_) {};
};

const auto RGY_SI_PREFIX_LIST = make_array<RGYSIPrefix>(
    RGYSIPrefix{ 'a', true,  rgy_pow_int<60,int64_t>(2), rgy_pow_int<18,int64_t>(10) },
    RGYSIPrefix{ 'f', true,  rgy_pow_int<50,int64_t>(2), rgy_pow_int<15,int64_t>(10) },
    RGYSIPrefix{ 'p', true,  rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'n', true,  rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 'u', true,  rgy_pow_int<20,int64_t>(2), rgy_pow_int< 6,int64_t>(10) },
    RGYSIPrefix{ 'm', true,  rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'k', false, rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'K', false, rgy_pow_int<10,int64_t>(2), rgy_pow_int< 3,int64_t>(10) },
    RGYSIPrefix{ 'M', false, rgy_pow_int<20,int64_t>(2), rgy_pow_int< 6,int64_t>(10) },
    RGYSIPrefix{ 'g', false, rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 'G', false, rgy_pow_int<30,int64_t>(2), rgy_pow_int< 9,int64_t>(10) },
    RGYSIPrefix{ 't', false, rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'T', false, rgy_pow_int<40,int64_t>(2), rgy_pow_int<12,int64_t>(10) },
    RGYSIPrefix{ 'P', false, rgy_pow_int<50,int64_t>(2), rgy_pow_int<15,int64_t>(10) },
    RGYSIPrefix{ 'E', false, rgy_pow_int<60,int64_t>(2), rgy_pow_int<18,int64_t>(10) }
    );

template<typename T>
static void rgy_apply_si_prefix(T& val, const TCHAR *endptr) {
    const auto prefix = tchar_to_string(endptr, CODE_PAGE_UTF8);
    if (prefix[0] != '\0') {
        auto siprefix = std::find_if(RGY_SI_PREFIX_LIST.begin(), RGY_SI_PREFIX_LIST.end(), [p = prefix[0]](const RGYSIPrefix& si) { return si.prefix == p; });
        if (siprefix != RGY_SI_PREFIX_LIST.end()) {
            const bool usepow2 = prefix[1] != 'i';
            if (siprefix->inverse) {
                val /= (usepow2) ? siprefix->pow2 : siprefix->pow10;
            } else {
                val *= (usepow2) ? siprefix->pow2 : siprefix->pow10;
            }
        }
    }
}

int rgy_parse_num(int& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        int64_t val64 = std::stoll(str, &idx, 0);
        auto endptr = str.c_str() + idx;
        rgy_apply_si_prefix(val64, endptr);
        if (val64 < std::numeric_limits<int>::min() || std::numeric_limits<int>::max() < val64) {
            val = 0;
            return 1;
        }
        val = (int)val64;
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(int64_t& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        val = std::stoll(str, &idx, 0);
        auto endptr = str.c_str() + idx;
        const auto prefix = tchar_to_string(endptr, CODE_PAGE_UTF8);
        rgy_apply_si_prefix(val, endptr);
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(float& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        double vald = std::stod(str, &idx);
        rgy_apply_si_prefix(vald, str.c_str() + idx);
        val = (float)vald;
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_parse_num(double& val, const tstring& str) {
    val = 0;
    try {
        size_t idx = 0;
        val = std::stod(str, &idx);
        rgy_apply_si_prefix(val, str.c_str() + idx);
    } catch (...) {
        return 1;
    }
    return 0;
}

tstring rgy_print_num_with_siprefix(const int64_t value) {
    const RGYSIPrefix *usePrefix = nullptr;
    for (const auto& prefix : RGY_SI_PREFIX_LIST) {
        if (!prefix.inverse && value > prefix.pow10) {
            usePrefix = &prefix;
        }
    }
    if (usePrefix) {
        return strsprintf(_T("%.3f%c"), value / (double)usePrefix->pow10, usePrefix->prefix);
    } else {
        return strsprintf(_T("%lld"), value);
    }
}

bool check_ext(const tstring& filename, const std::vector<const char*>& ext_list) {
    return check_ext(filename.c_str(), ext_list);
}

BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return tolowercase(std::filesystem::path(filename).extension().string()) == tolowercase(tchar_to_string(ext));
}

bool rgy_file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_file_exists(const std::wstring& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_get_filesize(const char *filepath, uint64_t *filesize) {
#if defined(_WIN32) || defined(_WIN64)
    const auto filepathw = char_to_wstring(filepath);
    return rgy_get_filesize(filepathw.c_str(), filesize);
#else //#if defined(_WIN32) || defined(_WIN64)
    struct stat stat;
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL || fstat(fileno(fp), &stat)) {
        *filesize = 0;
        return 1;
    }
    if (fp) {
        fclose(fp);
    }
    *filesize = stat.st_size;
    return 0;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
bool rgy_get_filesize(const WCHAR *filepath, uint64_t *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? true : false; // No MAX_PATH Limitation
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}

std::vector<tstring> get_file_list(const tstring& pattern, const tstring& dir) {
    std::vector<tstring> list;

    auto buf = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(pattern).wstring());

    WIN32_FIND_DATA win32fd;
    HANDLE hFind = FindFirstFile(buf.c_str(), &win32fd); // FindFirstFileW No MAX_PATH Limitation

    if (hFind == INVALID_HANDLE_VALUE) {
        return list;
    }

    do {
        if ((win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            && _tcscmp(win32fd.cFileName, _T("..")) !=0
            && _tcscmp(win32fd.cFileName, _T(".")) != 0) {
            const auto buf2 = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(win32fd.cFileName).wstring());
            vector_cat(list, get_file_list(pattern, buf2));
        } else {
            buf = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(win32fd.cFileName).wstring());
            list.push_back(buf);
        }
    } while (FindNextFile(hFind, &win32fd));
    FindClose(hFind);
    return list;
}

bool PathFileExistsA(const char *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool PathFileExistsW(const WCHAR *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}


tstring getExePath() {
    TCHAR exePath[1024];
    memset(exePath, 0, sizeof(exePath));
    GetModuleFileName(NULL, exePath, _countof(exePath));
    return exePath;
}

#else
tstring getExePath() {
    char prg_path[4096];
    auto ret = readlink("/proc/self/exe", prg_path, sizeof(prg_path));
    if (ret <= 0) {
        prg_path[0] = '\0';
    }
    return prg_path;
}

#endif //#if defined(_WIN32) || defined(_WIN64)
tstring getExeDir() {
    return PathRemoveFileSpecFixed(getExePath()).second;
}

tstring print_time(double time) {
    int sec = (int)time;
    time -= sec;
    int miniute = (int)(sec / 60);
    sec -= miniute * 60;
    int hour = miniute / 60;
    miniute -= hour * 60;
    tstring frac = strsprintf(_T("%.3f"), time);
    return strsprintf(_T("%d:%02d:%02d%s"), hour, miniute, sec, frac.substr(frac.find_first_of(_T("."))).c_str());
}

int rgy_print_stderr(int log_level, const TCHAR *mes, HANDLE handle) {
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

size_t malloc_degeneracy(void **ptr, size_t nSize, size_t nMinSize) {
    *ptr = nullptr;
    nMinSize = (std::max<size_t>)(nMinSize, 1);
    nSize = (std::max<size_t>)(nSize, nMinSize);
    //確保できなかったら、サイズを小さくして再度確保を試みる (最終的に1MBも確保できなかったら諦める)
    while (nSize >= nMinSize) {
        void *qtr = malloc(nSize);
        if (qtr != nullptr) {
            *ptr = qtr;
            return nSize;
        }
        size_t nNextSize = 0;
        for (size_t i = nMinSize; i < nSize; i<<=1) {
            nNextSize = i;
        }
        nSize = nNextSize;
    }
    return 0;
}

#if defined(_WIN32) || defined(_WIN64)

#include "rgy_osdep.h"
#include <process.h>
#include <VersionHelpers.h>

typedef void (WINAPI *RtlGetVersion_FUNC)(OSVERSIONINFOEXW*);

static int getRealWindowsVersion(DWORD *major, DWORD *minor, DWORD *build) {
    *major = 0;
    *minor = 0;
    OSVERSIONINFOEXW osver;
    HMODULE hModule = NULL;
    RtlGetVersion_FUNC func = NULL;
    int ret = 1;
    if (   NULL != (hModule = LoadLibrary(_T("ntdll.dll")))
        && NULL != (func = (RtlGetVersion_FUNC)GetProcAddress(hModule, "RtlGetVersion"))) {
        func(&osver);
        *major = osver.dwMajorVersion;
        *minor = osver.dwMinorVersion;
        *build = osver.dwBuildNumber;
        ret = 0;
    }
    if (hModule) {
        FreeLibrary(hModule);
    }
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

BOOL check_OS_Win8orLater() {
#if defined(_WIN32) || defined(_WIN64)
#if (_MSC_VER >= 1800)
    return IsWindows8OrGreater();
#else
    OSVERSIONINFO osvi = { 0 };
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osvi);
    return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
#endif //(_MSC_VER >= 1800)
#else //#if defined(_WIN32) || defined(_WIN64)
    return FALSE;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
#pragma warning(push)
#pragma warning(disable:4996) // warning C4996: 'GetVersionExW': が古い形式として宣言されました。
tstring getOSVersion(OSVERSIONINFOEXW *osinfo) {
    const TCHAR *ptr = _T("Unknown");
    OSVERSIONINFOW info = { 0 };
    OSVERSIONINFOEXW infoex = { 0 };
    info.dwOSVersionInfoSize = sizeof(info);
    infoex.dwOSVersionInfoSize = sizeof(infoex);
    GetVersionExW(&info);
    switch (info.dwPlatformId) {
    case VER_PLATFORM_WIN32_WINDOWS:
        if (4 <= info.dwMajorVersion) {
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows 95"); break;
            case 10: ptr = _T("Windows 98"); break;
            case 90: ptr = _T("Windows Me"); break;
            default: break;
            }
        }
        break;
    case VER_PLATFORM_WIN32_NT:
        if (info.dwMajorVersion >= 6 || (info.dwMajorVersion == 5 && info.dwMinorVersion >= 2)) {
            GetVersionExW((OSVERSIONINFOW *)&infoex);
        } else {
            memcpy(&infoex, &info, sizeof(info));
        }
        if (info.dwMajorVersion == 6) {
            getRealWindowsVersion(&infoex.dwMajorVersion, &infoex.dwMinorVersion, &infoex.dwBuildNumber);
        }
        if (osinfo) {
            memcpy(osinfo, &infoex, sizeof(infoex));
        }
        switch (infoex.dwMajorVersion) {
        case 3:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = _T("Windows NT 3"); break;
            case 1:  ptr = _T("Windows NT 3.1"); break;
            case 5:  ptr = _T("Windows NT 3.5"); break;
            case 51: ptr = _T("Windows NT 3.51"); break;
            default: break;
            }
            break;
        case 4:
            if (0 == infoex.dwMinorVersion)
                ptr = _T("Windows NT 4.0");
            break;
        case 5:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = _T("Windows 2000"); break;
            case 1:  ptr = _T("Windows XP"); break;
            case 2:  ptr = _T("Windows Server 2003"); break;
            default: break;
            }
            break;
        case 6:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows Vista") : _T("Windows Server 2008");    break;
            case 1:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 7")     : _T("Windows Server 2008 R2"); break;
            case 2:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 8")     : _T("Windows Server 2012");    break;
            case 3:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 8.1")   : _T("Windows Server 2012 R2"); break;
            case 4:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 10")    : _T("Windows Server 2016");    break;
            default:
                if (5 <= infoex.dwMinorVersion) {
                    ptr = _T("Later than Windows 10");
                }
                break;
            }
            break;
        case 10:
            ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 10") : _T("Windows Server 2016"); break;
        default:
            if (10 <= infoex.dwMajorVersion) {
                ptr = _T("Later than Windows 10");
            }
            break;
        }
        break;
    default:
        break;
    }
    return tstring(ptr);
}
#pragma warning(pop)

tstring getOSVersion() {
    OSVERSIONINFOEXW osversioninfo = { 0 };
    tstring osversionstr = getOSVersion(&osversioninfo);
    osversionstr += strsprintf(_T(" %s (%d)"), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber);
    return osversionstr;
}
#else //#if defined(_WIN32) || defined(_WIN64)
tstring getOSVersion() {
    std::string str = "";
    FILE *fp = fopen("/etc/os-release", "r");
    if (fp != NULL) {
        char buffer[2048];
        while (fgets(buffer, _countof(buffer), fp) != NULL) {
            if (strncmp(buffer, "PRETTY_NAME=", strlen("PRETTY_NAME=")) == 0) {
                str = trim(std::string(buffer + strlen("PRETTY_NAME=")), " \"\t\n");
                break;
            }
        }
        fclose(fp);
    }
    if (str.length() == 0) {
        struct stat buffer;
        if (stat ("/usr/bin/lsb_release", &buffer) == 0) {
            FILE *fp = popen("/usr/bin/lsb_release -a", "r");
            if (fp != NULL) {
                char buffer[2048];
                while (NULL != fgets(buffer, _countof(buffer), fp)) {
                    str += buffer;
                }
                pclose(fp);
                if (str.length() > 0) {
                    auto sep = split(str, "\n");
                    for (auto line : sep) {
                        if (line.find("Description") != std::string::npos) {
                            std::string::size_type pos = line.find(":");
                            if (pos == std::string::npos) {
                                pos = std::string("Description").length();
                            }
                            pos++;
                            str = line.substr(pos);
                            break;
                        }
                    }
                }
            }
        }
    }
    if (str.length() == 0) {
        struct utsname buf;
        uname(&buf);
        str += buf.sysname;
        str += " ";
        str += buf.release;
    }
    return char_to_tstring(trim(str));
}
#endif //#if defined(_WIN32) || defined(_WIN64)

BOOL rgy_is_64bit_os() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sinfo = { 0 };
    GetNativeSystemInfo(&sinfo);
    return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct utsname buf;
    uname(&buf);
    return NULL != strstr(buf.machine, "x64")
        || NULL != strstr(buf.machine, "x86_64")
        || NULL != strstr(buf.machine, "amd64");
#endif //#if defined(_WIN32) || defined(_WIN64)
}

uint64_t getPhysicalRamSize(uint64_t *ramUsed) {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX msex ={ 0 };
    msex.dwLength = sizeof(msex);
    GlobalMemoryStatusEx(&msex);
    if (NULL != ramUsed) {
        *ramUsed = msex.ullTotalPhys - msex.ullAvailPhys;
    }
    return msex.ullTotalPhys;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct sysinfo info;
    sysinfo(&info);
    if (NULL != ramUsed) {
        *ramUsed = info.totalram - info.freeram;
    }
    return info.totalram;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
tstring getACPCodepageStr() {
    const auto codepage = GetACP();
    auto codepage_ptr = codepage_str((uint32_t)codepage);
    if (codepage_ptr != nullptr) {
        return char_to_tstring(codepage_ptr);
    }
    return _T("CP") + char_to_tstring(std::to_string(codepage));
}
#endif

tstring getEnviromentInfo(int device_id) {
    tstring buf;

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));
    uint64_t UsedRamSize = 0;
    uint64_t totalRamsize = getPhysicalRamSize(&UsedRamSize);

    buf += _T("Environment Info\n");
#if defined(_WIN32) || defined(_WIN64)
    OSVERSIONINFOEXW osversioninfo = { 0 };
    tstring osversionstr = getOSVersion(&osversioninfo);
    buf += strsprintf(_T("OS : %s %s (%d) [%s]\n"), osversionstr.c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber, getACPCodepageStr().c_str());
#else
    buf += strsprintf(_T("OS : %s %s\n"), getOSVersion().c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"));
#endif
    buf += strsprintf(_T("CPU: %s\n"), cpu_info);
    buf += strsprintf(_T("RAM: Used %d MB, Total %d MB\n"), (uint32_t)(UsedRamSize >> 20), (uint32_t)(totalRamsize >> 20));

#if ENCODER_QSV
    TCHAR gpu_info[1024] = { 0 };
    getGPUInfo(GPU_VENDOR, gpu_info, _countof(gpu_info));
    buf += strsprintf(_T("GPU: %s\n"), gpu_info);
#endif //#if ENCODER_QSV
    return buf;
}

struct sar_option_t {
    int key;
    int sar[2];
};

static const sar_option_t SAR_LIST[] = {
    {  0, {  0,  0 } },
    {  1, {  1,  1 } },
    {  2, { 12, 11 } },
    {  3, { 10, 11 } },
    {  4, { 16, 11 } },
    {  5, { 40, 33 } },
    {  6, { 24, 11 } },
    {  7, { 20, 11 } },
    {  8, { 32, 11 } },
    {  9, { 80, 33 } },
    { 10, { 18, 11 } },
    { 11, { 15, 11 } },
    { 12, { 64, 33 } },
    { 13, {160, 99 } },
    { 14, {  4,  3 } },
    { 15, {  3,  2 } },
    { 16, {  2,  1 } }
};

std::pair<int, int> get_h264_sar(int idx) {
    for (int i = 0; i < _countof(SAR_LIST); i++) {
        if (SAR_LIST[i].key == idx)
            return std::make_pair(SAR_LIST[i].sar[0], SAR_LIST[i].sar[1]);
    }
    return std::make_pair(0, 0);
}

int get_h264_sar_idx(std::pair<int, int> sar) {

    if (0 != sar.first && 0 != sar.second) {
        rgy_reduce(sar);
    }

    for (int i = 0; i < _countof(SAR_LIST); i++) {
        if (SAR_LIST[i].sar[0] == sar.first && SAR_LIST[i].sar[1] == sar.second) {
            return SAR_LIST[i].key;
        }
    }
    return -1;
}

void adjust_sar(int *sar_w, int *sar_h, int width, int height) {
    int aspect_w = *sar_w;
    int aspect_h = *sar_h;
    //正負チェック
    if (aspect_w * aspect_h <= 0)
        aspect_w = aspect_h = 0;
    else if (aspect_w < 0) {
        //負で与えられている場合はDARでの指定
        //SAR比に変換する
        int dar_x = -1 * aspect_w;
        int dar_y = -1 * aspect_h;
        int x = dar_x * height;
        int y = dar_y * width;
        //多少のづれは容認する
        if (abs(y - x) > 16 * dar_y) {
            //gcd
            int a = x, b = y, c;
            while ((c = a % b) != 0)
                a = b, b = c;
            *sar_w = x / b;
            *sar_h = y / b;
        } else {
            *sar_w = *sar_h = 1;
        }
    } else {
        //sarも一応gcdをとっておく
        int a = aspect_w, b = aspect_h, c;
        while ((c = a % b) != 0)
            a = b, b = c;
        *sar_w = aspect_w / b;
        *sar_h = aspect_h / b;
    }
}

void get_dar_pixels(unsigned int* width, unsigned int* height, int sar_w, int sar_h) {
    int w = *width;
    int h = *height;
    if (0 != (w * h * sar_w * sar_h)) {
        int x = w * sar_w;
        int y = h * sar_h;
        int a = x, b = y, c;
        while ((c = a % b) != 0)
            a = b, b = c;
        x /= b;
        y /= b;
        const double ratio = (sar_w >= sar_h) ? h / (double)y : w / (double)x;
        *width  = (int)(x * ratio + 0.5);
        *height = (int)(y * ratio + 0.5);
    }
}

std::pair<int, int> get_sar(unsigned int width, unsigned int height, unsigned int darWidth, unsigned int darHeight) {
    int x = darWidth  * height;
    int y = darHeight *  width;
    int a = x, b = y, c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return std::make_pair<int, int>(x / b, y / b);
}

void set_auto_resolution(int& dst_w, int& dst_h, int dst_sar_w, int dst_sar_h, int src_w, int src_h, int src_sar_w, int src_sar_h, const sInputCrop& crop) {
    if (dst_w * dst_h < 0) {
        src_w -= (crop.e.left + crop.e.right);
        src_h -= (crop.e.bottom + crop.e.up);
        double dar = src_w / (double)src_h;
        if (src_sar_w * src_sar_h > 0) {
            if (src_sar_w < 0) {
                dar = src_sar_w / (double)src_sar_h;
            } else {
                dar = (src_w * (double)src_sar_w) / (src_h * (double)src_sar_h);
            }
        }
        if (dst_sar_w * dst_sar_h <= 0) {
            dst_sar_w = dst_sar_h = 1;
        }
        dar /= (dst_sar_w / (double)dst_sar_h);
        if (dst_w < 0) {
            const int div = std::abs(dst_w);
            dst_w = (((int)(dst_h * dar) + (div >> 1)) / div) * div;
        } else { //dst_h < 0
            const int div = std::abs(dst_h);
            dst_h = (((int)(dst_w / dar) + (div >> 1)) / div) * div;
        }
    }
}

// convert float to half precision floating point
unsigned short float2half(float value) {
    // 1 : 8 : 23
    union {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

        // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0) {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    } else if (exponent == 0xFF) {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    } else {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31) {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        } else if (newexp <= 0) {
            // underflow
            if (newexp >= -10) {
                // denormal half-precision
                unsigned short sig = (unsigned short)((significand | 0x800000) >> (14 - newexp));
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            } else {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        } else {
            fp16 = (unsigned short)((sign << 15) | (newexp << 10) | (significand >> 13));
        }
    }
    return fp16;
}
