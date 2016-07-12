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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <VersionHelpers.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <tchar.h>
#include "cpu_info.h"
#include "gpu_info.h"
#include "NVEncUtil.h"
#include "NVEncLog.h"

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
    str.resize(multibyte_length, 0);
    if (0 == WideCharToMultiByte(codepage, flags, wstr, -1, &str[0], multibyte_length, nullptr, nullptr)) {
        str.clear();
        return 0;
    }
    return multibyte_length;
}
#else
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    if (wstr == nullptr) {
        str = "";
        return 0;
    }
    auto ic = iconv_open("UTF-8", "wchar_t"); //to, from
    auto input_len = wcslen(wstr);
    auto output_len = input_len * 4;
    str.resize(output_len, 0);
    char *outbuf = &str[0];
    iconv(ic, (char **)&wstr, &input_len, &outbuf, &output_len);
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

std::string tchar_to_string(const tstring& tstr, uint32_t codepage) {
    std::string str;
    tchar_to_string(tstr.c_str(), str, codepage);
    return str;
}

#if defined(_WIN32) || defined(_WIN64)
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    int widechar_length = MultiByteToWideChar(codepage, 0, str, -1, nullptr, 0);
    wstr.resize(widechar_length, 0);
    if (0 == MultiByteToWideChar(codepage, 0, str, -1, &wstr[0], (int)wstr.size())) {
        wstr.clear();
        return 0;
    }
    return widechar_length;
}
#else
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    if (str == nullptr) {
        wstr = L"";
        return 0;
    }
    auto ic = iconv_open("wchar_t", "UTF-8"); //to, from
    auto input_len = strlen(str);
    std::vector<char> buf(input_len + 1);
    strcpy(buf.data(), str);
    auto output_len = input_len;
    wstr.resize(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = (char *)&wstr[0];
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    return output_len;
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

unsigned int wstring_to_tstring(const WCHAR *wstr, tstring& tstr, uint32_t codepage) {
#if UNICODE
    tstr = std::wstring(wstr);
#else
    return wstring_to_string(wstr, tstr, codepage);
#endif
    return (unsigned int)tstr.length();
}

tstring wstring_to_tstring(const WCHAR *wstr, uint32_t codepage) {
    tstring tstr;
    wstring_to_tstring(wstr, tstr, codepage);
    return tstr;
}

tstring wstring_to_tstring(const std::wstring& wstr, uint32_t codepage) {
    tstring tstr;
    wstring_to_tstring(wstr.c_str(), tstr, codepage);
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
    vswprintf(buffer.data(), format, args);
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
    return std::move(str);
}

#if defined(_WIN32) || defined(_WIN64)
std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to) {
    std::wstring::size_type pos = 0;
    while (pos = str.find(from, pos), pos != std::wstring::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return std::move(str);
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
#if defined(_WIN32) || defined(_WIN64)
    if (PathIsRelativeA(path) == FALSE)
        return std::string(path);
#endif //#if defined(_WIN32) || defined(_WIN64)
    std::vector<char> buffer(strlen(path) + 1024, 0);
    _fullpath(buffer.data(), path, buffer.size());
    return std::string(buffer.data());
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring GetFullPath(const WCHAR *path) {
    if (PathIsRelativeW(path) == FALSE)
        return std::wstring(path);

    std::vector<WCHAR> buffer(wcslen(path) + 1024, 0);
    _wfullpath(buffer.data(), path, buffer.size());
    return std::wstring(buffer.data());
}
//ルートディレクトリを取得
std::string PathGetRoot(const char *path) {
    auto fullpath = GetFullPath(path);
    std::vector<char> buffer(fullpath.length() + 1, 0);
    memcpy(buffer.data(), fullpath.c_str(), fullpath.length() * sizeof(fullpath[0]));
    PathStripToRootA(buffer.data());
    return buffer.data();
}
std::wstring PathGetRoot(const WCHAR *path) {
    auto fullpath = GetFullPath(path);
    std::vector<WCHAR> buffer(fullpath.length() + 1, 0);
    memcpy(buffer.data(), fullpath.c_str(), fullpath.length() * sizeof(fullpath[0]));
    PathStripToRootW(buffer.data());
    return buffer.data();
}

//パスのルートが存在するかどうか
static bool PathRootExists(const char *path) {
    if (path == nullptr)
        return false;
    return PathIsDirectoryA(PathGetRoot(path).c_str()) != 0;
}
static bool PathRootExists(const WCHAR *path) {
    if (path == nullptr)
        return false;
    return PathIsDirectoryW(PathGetRoot(path).c_str()) != 0;
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path) {
    const char *ptr = path.c_str();
    const char *qtr = PathFindFileNameA(ptr);
    if (qtr == ptr) {
        return std::make_pair(0, path);
    }
    std::string newPath = path.substr(0, qtr - ptr - 1);
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path) {
    const WCHAR *ptr = path.c_str();
    WCHAR *qtr = PathFindFileNameW(ptr);
    if (qtr == ptr) {
        return std::make_pair(0, path);
    }
    std::wstring newPath = path.substr(0, qtr - ptr - 1);
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#endif //#if defined(_WIN32) || defined(_WIN64)
//フォルダがあればOK、なければ作成する
bool CreateDirectoryRecursive(const char *dir) {
    if (PathIsDirectoryA(dir)) {
        return true;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (!PathRootExists(dir)) {
        return false;
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
    auto ret = PathRemoveFileSpecFixed(dir);
    if (ret.first == 0) {
        return false;
    }
    if (!CreateDirectoryRecursive(ret.second.c_str())) {
        return false;
    }
    return CreateDirectoryA(dir, NULL) != 0;
}
#if defined(_WIN32) || defined(_WIN64)
bool CreateDirectoryRecursive(const WCHAR *dir) {
    if (PathIsDirectoryW(dir)) {
        return true;
    }
    if (!PathRootExists(dir)) {
        return false;
    }
    auto ret = PathRemoveFileSpecFixed(dir);
    if (ret.first == 0) {
        return false;
    }
    if (!CreateDirectoryRecursive(ret.second.c_str())) {
        return false;
    }
    return CreateDirectoryW(dir, NULL) != 0;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list) {
    const TCHAR *target = PathFindExtension(filename);
    if (target) {
        for (auto ext : ext_list) {
            if (0 == _tcsicmp(target, char_to_tstring(ext).c_str())) {
                return true;
            }
        }
    }
    return false;
}

bool get_filesize(const char *filepath, uint64_t *filesize) {
#if defined(_WIN32) || defined(_WIN64)
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd)) ? true : false;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
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
bool get_filesize(const WCHAR *filepath, uint64_t *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? true : false;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

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

int nv_print_stderr(int log_level, const TCHAR *mes, HANDLE handle) {
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
    if (handle && log_level != NV_LOG_INFO) {
        GetConsoleScreenBufferInfo(handle, &csbi);
        SetConsoleTextAttribute(handle, LOG_COLOR[clamp(log_level, NV_LOG_TRACE, NV_LOG_ERROR) - NV_LOG_TRACE] | (csbi.wAttributes & 0x00f0));
    }
    //このfprintfで"%"が消えてしまわないよう置換する
    int ret = _ftprintf(stderr, (nullptr == _tcschr(mes, _T('%'))) ? mes : str_replace(tstring(mes), _T("%"), _T("%%")).c_str());
    if (handle && log_level != NV_LOG_INFO) {
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
    int ret = _ftprintf(stderr, "%s%s%s", LOG_COLOR[clamp(log_level, NV_LOG_TRACE, NV_LOG_ERROR) - NV_LOG_TRACE], mes, LOG_COLOR[NV_LOG_INFO - NV_LOG_TRACE]);
#endif //#if defined(_WIN32) || defined(_WIN64)
    fflush(stderr);
    return ret;
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
        c = ((y + h - 1) / h) * h;
        *width  = x * c;
        *height = y * c;
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

size_t malloc_degeneracy(void **ptr, size_t nSize, size_t nMinSize) {
    *ptr = nullptr;
    nMinSize = (std::max<size_t>)(nMinSize, 1);
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

static const std::map<int, std::pair<int, int>> sar_list = {
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
    for (auto i_sar : sar_list) {
        if (i_sar.first == idx)
            return i_sar.second;
    }
    return std::make_pair(0, 0);
}

int get_h264_sar_idx(std::pair<int, int> sar) {

    if (0 != sar.first && 0 != sar.second) {
        const int gcd = nv_get_gcd(sar);
        sar.first  /= gcd;
        sar.second /= gcd;
    }

    for (auto i_sar : sar_list) {
        if (i_sar.second == sar)
            return i_sar.first;
    }
    return -1;
}

/*
int ParseY4MHeader(char *buf, mfxFrameInfo *info) {
    char *p, *q = NULL;
    memset(info, 0, sizeof(mfxFrameInfo));
    for (p = buf; (p = strtok_s(p, " ", &q)) != NULL; ) {
        switch (*p) {
            case 'W':
                {
                    char *eptr = NULL;
                    int w = strtol(p+1, &eptr, 10);
                    if (*eptr == '\0' && w)
                        info->Width = (mfxU16)w;
                }
                break;
            case 'H':
                {
                    char *eptr = NULL;
                    int h = strtol(p+1, &eptr, 10);
                    if (*eptr == '\0' && h)
                        info->Height = (mfxU16)h;
                }
                break;
            case 'F':
                {
                    int rate = 0, scale = 0;
                    if (   (info->FrameRateExtN == 0 || info->FrameRateExtD == 0)
                        && sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
                            if (rate && scale) {
                                info->FrameRateExtN = rate;
                                info->FrameRateExtD = scale;
                            }
                    }
                }
                break;
            case 'A':
                {
                    int sar_x = 0, sar_y = 0;
                    if (   (info->AspectRatioW == 0 || info->AspectRatioH == 0)
                        && sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
                            if (sar_x && sar_y) {
                                info->AspectRatioW = (mfxU16)sar_x;
                                info->AspectRatioH = (mfxU16)sar_y;
                            }
                    }
                }
                break;
            case 'I':
                switch (*(p+1)) {
            case 'b':
                info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
                break;
            case 't':
            case 'm':
                info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
                break;
            default:
                break;
                }
                break;
            case 'C':
                if (   0 != _strnicmp(p+1, "420",      strlen("420"))
                    && 0 != _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
                    && 0 != _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
                    && 0 != _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
                    return MFX_PRINT_OPTION_ERR;
                }
                break;
            default:
                break;
        }
        p = NULL;
    }
    return MFX_ERR_NONE;
}
*/
#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>
#include <process.h>

typedef void (WINAPI *RtlGetVersion_FUNC)(OSVERSIONINFOEXW*);

static int getRealWindowsVersion(DWORD *major, DWORD *minor) {
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

tstring getOSVersion() {
#if defined(_WIN32) || defined(_WIN64)
    const TCHAR *ptr = _T("Unknown");
    OSVERSIONINFO info = { 0 };
    info.dwOSVersionInfoSize = sizeof(info);
    GetVersionEx(&info);
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
        if (info.dwMajorVersion == 6) {
            getRealWindowsVersion(&info.dwMajorVersion, &info.dwMinorVersion);
        }
        switch (info.dwMajorVersion) {
        case 3:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows NT 3"); break;
            case 1:  ptr = _T("Windows NT 3.1"); break;
            case 5:  ptr = _T("Windows NT 3.5"); break;
            case 51: ptr = _T("Windows NT 3.51"); break;
            default: break;
            }
            break;
        case 4:
            if (0 == info.dwMinorVersion)
                ptr = _T("Windows NT 4.0");
            break;
        case 5:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows 2000"); break;
            case 1:  ptr = _T("Windows XP"); break;
            case 2:  ptr = _T("Windows Server 2003"); break;
            default: break;
            }
            break;
        case 6:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows Vista"); break;
            case 1:  ptr = _T("Windows 7"); break;
            case 2:  ptr = _T("Windows 8"); break;
            case 3:  ptr = _T("Windows 8.1"); break;
            case 4:  ptr = _T("Windows 10"); break;
            default:
                if (5 <= info.dwMinorVersion) {
                    ptr = _T("Later than Windows 10");
                }
                break;
            }
            break;
        case 10:
            ptr = _T("Windows 10");
            break;
        default:
            if (10 <= info.dwMajorVersion) {
                ptr = _T("Later than Windows 10");
            }
            break;
        }
        break;
    default:
        break;
    }
    return tstring(ptr);
#else //#if defined(_WIN32) || defined(_WIN64)
    std::string str = "";
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
    if (str.length() == 0) {
        struct utsname buf;
        uname(&buf);
        str += buf.sysname;
        str += " ";
        str += buf.release;
    }
    return char_to_tstring(trim(str));
#endif //#if defined(_WIN32) || defined(_WIN64)
}

BOOL nv_is_64bit_os() {
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

tstring getEnviromentInfo(bool add_ram_info) {
    tstring buf;

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));

    TCHAR gpu_info[1024] = { 0 };
    getGPUInfo("Intel", gpu_info, _countof(gpu_info));

    uint64_t UsedRamSize = 0;
    uint64_t totalRamsize = getPhysicalRamSize(&UsedRamSize);

    buf += _T("Environment Info\n");
    buf += strsprintf(_T("OS : %s (%s)\n"), getOSVersion().c_str(), nv_is_64bit_os() ? _T("x64") : _T("x86"));
    buf += strsprintf(_T("CPU: %s\n"), cpu_info);
    add_ram_info = false;
    buf += strsprintf(_T("%s Used %d MB, Total %d MB\n"), (add_ram_info) ? _T("    ") : _T("RAM:"), (uint32_t)(UsedRamSize >> 20), (uint32_t)(totalRamsize >> 20));
    buf += strsprintf(_T("GPU: %s\n"), gpu_info);
    return buf;
}
