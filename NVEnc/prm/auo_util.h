// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#ifndef _AUO_UTIL_H_
#define _AUO_UTIL_H_

#include <Windows.h>
#if (_MSC_VER >= 1800)
#include <VersionHelpers.h>
#endif
#include <string.h>
#include <vector>
#include <string>
#include <cstdarg>
#include <stddef.h>
#include <stdio.h>
#include <algorithm>
#include <intrin.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "auo.h"
#include "auo_version.h"

//日本語環境の一般的なコードページ一覧
#define CODE_PAGE_SJIS        932 //Shift-JIS
#define CODE_PAGE_JIS         50220
#define CODE_PAGE_EUC_JP      51932
#define CODE_PAGE_UTF8        CP_UTF8
#define CODE_PAGE_UTF16_LE    CP_WINUNICODE //WindowsのUnicode WCHAR のコードページ
#define CODE_PAGE_UTF16_BE    1201
#define CODE_PAGE_US_ASCII    20127
#define CODE_PAGE_WEST_EUROPE 1252  //厄介な西ヨーロッパ言語
#define CODE_PAGE_UNSET       0xffffffff

//BOM文字リスト
static const int MAX_UTF8_CHAR_LENGTH = 6;
static const BYTE UTF8_BOM[]     = { 0xEF, 0xBB, 0xBF };
static const BYTE UTF16_LE_BOM[] = { 0xFF, 0xFE };
static const BYTE UTF16_BE_BOM[] = { 0xFE, 0xFF };

//関数マクロ
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#define foreach(it,a) \
    for (auto (it)=(a).begin();(it)!=(a).end();(it)++)

static std::string strprintf(const char* format, ...) {
    std::va_list arg;
    va_start(arg, format);

    std::string ret;
    ret.resize(_vscprintf(format, arg) + 1);
    int n = vsprintf_s(&ret[0], ret.size(), format, arg);
    ret.resize(n);
    va_end(arg);
    return ret;
}

static std::wstring strprintf(const wchar_t* format, ...) {
    std::va_list arg;
    va_start(arg, format);

    std::wstring ret;
    ret.resize(_vscwprintf(format, arg) + 1);
    int n = vswprintf_s(&ret[0], ret.size(), format, arg);
    ret.resize(n);
    va_end(arg);
    return ret;
}

template<typename T>
static std::basic_string<T> replace(std::basic_string<T> targetString, std::basic_string<T> oldStr, std::basic_string<T> newStr) {
    for (std::basic_string<T>::size_type pos(targetString.find(oldStr)); std::basic_string<T>::npos != pos;
        pos = targetString.find(oldStr, pos + newStr.length()) ) {
        targetString.replace(pos, oldStr.length(), newStr);
    }
    return targetString;
}

template<typename T>
static std::vector<std::basic_string<T>> split(const std::basic_string<T> &str, T delim) {
    std::vector<std::basic_string<T>> res;
    string::size_type current = 0, found;
    for (; std::basic_string<T>::npos != (found = str.find_first_of(delim, current)); current = found + 1) {
        res.push_back(std::basic_string<T>(str, current, found - current));
        current = found + 1;
    }
    std::basic_string<T> last_line = std::basic_string<T>(str, current, str.length() - current);
    if (std::wcslen(last_line.c_str()))
        res.push_back(last_line);
    return res;
}

//基本的な関数
static inline double pow2(double a) {
    return a * a;
}
static inline int pow2(int a) {
    return a * a;
}
static inline BOOL check_range(int value, int min, int max) {
    return (min <= value && value <= max);
}
static inline BOOL check_range(double value, double min, double max) {
    return (min <= value && value <= max);
}
static inline BOOL check_range(void* value, void* min, void* max) {
    return (min <= value && value <= max);
}
static inline int ceil_div_int(int i, int div) {
    return (i + (div-1)) / div;
}
static inline DWORD ceil_div_int(DWORD i, int div) {
    return (i + (div-1)) / div;
}
static inline __int64 ceil_div_int64(__int64 i, int div) {
    return (i + (div-1)) / div;
}
static inline UINT64 ceil_div_int64(UINT64 i, int div) {
    return (i + (div-1)) / div;
}
static inline int get_gcd(int a, int b) {
    int c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return b;
}

//大文字小文字を無視して、1文字検索
static inline const char *strichr(const char *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}
static inline char *strichr(char *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}

//大文字小文字を無視して、文字列を検索
static inline const char *stristr(const char *str, const char *substr) {
    size_t len = 0;
    if (substr && (len = strlen(substr)) != NULL)
        for (; (str = strichr(str, substr[0])) != NULL; str++)
            if (_strnicmp(str, substr, len) == NULL)
                return str;
    return NULL;
}
static inline char *stristr(char *str, const char *substr) {
    size_t len = 0;
    if (substr && (len = strlen(substr)) != NULL)
        for (; (str = strichr(str, substr[0])) != NULL; str++)
            if (_strnicmp(str, substr, len) == NULL)
                return str;
    return NULL;
}

//指定した場所から後ろ向きに1文字検索
static inline const char *strrchr(const char *str, int c, int start_index) {
    if (start_index < 0) return NULL;
    const char *result = str + start_index;
    str--;
    for (; result - str; result--)
        if (*result == c)
            return result;
    return NULL;
}
static inline char *strrchr(char *str, int c, int start_index) {
    if (start_index < 0) return NULL;
    char *result = str + start_index;
    str--;
    for (; result - str; result--)
        if (*result == c)
            return result;
    return NULL;
}

//strのcount byteを検索し、substrとの一致を返す
static inline const char * strnstr(const char *str, const char *substr, int count) {
    const char *ptr = strstr(str, substr);
    if (ptr && ptr - str >= count)
        ptr = NULL;
    return ptr;
}
static inline char * strnstr(char *str, const char *substr, int count) {
    char *ptr = strstr(str, substr);
    if (ptr && ptr - str >= count)
        ptr = NULL;
    return ptr;
}

//strのsubstrとの最後の一致を返す
static inline const char * strrstr(const char *str, const char *substr) {
    const char *last_ptr = NULL;
    for (const char *ptr = str; *ptr && (ptr = strstr(ptr, substr)) != NULL; ptr++ )
        last_ptr = ptr;
    return last_ptr;
}
static inline char * strrstr(char *str, const char *substr) {
    char *last_ptr = NULL;
    for (char *ptr = str; *ptr && (ptr = strstr(ptr, substr)) != NULL; ptr++ )
        last_ptr = ptr;
    return last_ptr;
}

//strのcount byteを検索し、substrとの最後の一致を返す
static inline const char * strnrstr(const char *str, const char *substr, int count) {
    const char *last_ptr = NULL;
    if (count > 0)
        for (const char *ptr = str; *ptr && (ptr = strnstr(ptr, substr, count - (ptr - str))) != NULL; ptr++)
            last_ptr = ptr;
    return last_ptr;
}
static inline char * strnrstr(char *str, const char *substr, int count) {
    char *last_ptr = NULL;
    if (count > 0)
        for (char *ptr = str; *ptr && (ptr = strnstr(ptr, substr, count - (ptr - str))) != NULL; ptr++)
            last_ptr = ptr;
    return last_ptr;
}

//文字列中の文字「ch」の数を数える
static inline int countchr(const char *str, int ch) {
    int i = 0;
    for (; *str; str++)
        if (*str == ch)
            i++;
    return i;
}
static inline int countchr(const WCHAR *str, int ch) {
    int i = 0;
    for (; *str; str++)
        if (*str == ch)
            i++;
    return i;
}

//文字列の末尾についている '\r' '\n' ' ' を削除する
static inline size_t deleteCRLFSpace_at_End(WCHAR *str) {
    WCHAR *pw = str + wcslen(str) - 1;
    WCHAR * const qw = pw;
    while ((*pw == L'\n' || *pw == L'\r' || *pw == L' ') && pw >= str) {
        *pw = L'\0';
        pw--;
    }
    return qw - pw;
}

static inline size_t deleteCRLFSpace_at_End(char *str) {
    char *pw = str + strlen(str) - 1;
    char *qw = pw;
    while ((*pw == '\n' || *pw == '\r' || *pw == ' ') && pw >= str) {
        *pw = '\0';
        pw--;
    }
    return qw - pw;
}

static inline BOOL str_has_char(const char *str) {
    BOOL ret = FALSE;
    for (; !ret && *str != '\0'; str++)
        ret = (*str != ' ');
    return ret;
}

static inline BOOL str_has_char(const WCHAR *str) {
    BOOL ret = FALSE;
    for (; !ret && *str != L'\0'; str++)
        ret = (*str != ' ');
    return ret;
}

static DWORD cpu_core_count() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

static BOOL is_64bit_os() {
    SYSTEM_INFO sinfo = { 0 };
    GetNativeSystemInfo(&sinfo);
    return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
}

static size_t get_intlen(int i) {
    char str[256];
    sprintf_s(str, _countof(str), "%d", i);
    return strlen(str);
}

//mmx    cpuid_1 CPUInfo[3] & 0x00800000
//sse    cpuid_1 CPUInfo[3] & 0x02000000
//sse2   cpuid_1 CPUInfo[3] & 0x04000000
//sse3   cpuid_1 CPUInfo[2] & 0x00000001
//ssse3  cpuid_1 CPUInfo[2] & 0x00000200
//sse4.1 cpuid_1 CPUInfo[2] & 0x00080000
//sse4.2 cpuid_1 CPUInfo[2] & 0x00100000
//avx    cpuid_1 CPUInfo[2] & 0x18000000 == 0x18000000 + OSチェック
//avx2   cpuid_7 CPUInfo[1] & 0x00000020 + OSチェック
static BOOL check_sse2() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[3] & 0x04000000) != 0;
}

static BOOL check_ssse3() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[2] & 0x00000200) != 0;
}

static BOOL check_sse3() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[2] & 0x00000001) != 0;
}

static BOOL check_sse4_1() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[2] & 0x00080000) != 0;
}
static BOOL check_avx() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        UINT64 XGETBV = _xgetbv(0);
        if ((XGETBV & 0x06) == 0x06)
            return TRUE;
    }
    return FALSE;
}
static BOOL check_avx2() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        UINT64 XGETBV = _xgetbv(0);
        if ((XGETBV & 0x06) == 0x06) {
            __cpuid(CPUInfo, 7);
            if ((CPUInfo[1] & 0x00000020))
                return TRUE;
        }
    }
    return FALSE;
}
#if 0
static DWORD get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    DWORD simd = AUO_SIMD_NONE;
    if  (CPUInfo[3] & 0x04000000)
        simd |= AUO_SIMD_SSE2;
    if  (CPUInfo[2] & 0x00000001)
        simd |= AUO_SIMD_SSE3;
    if  (CPUInfo[2] & 0x00000200)
        simd |= AUO_SIMD_SSSE3;
    if  (CPUInfo[2] & 0x00080000)
        simd |= AUO_SIMD_SSE41;
    if  (CPUInfo[2] & 0x00100000)
        simd |= AUO_SIMD_SSE42;
    UINT64 XGETBV = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        XGETBV = _xgetbv(0);
        if ((XGETBV & 0x06) == 0x06)
            simd |= AUO_SIMD_AVX;
    }
    __cpuid(CPUInfo, 7);
    if ((simd & AUO_SIMD_AVX) && (CPUInfo[1] & 0x00000020))
        simd |= AUO_SIMD_AVX2;
    return simd;
}
#endif

static BOOL check_OS_Win7orLater() {
#if (_MSC_VER >= 1800)
    return IsWindowsVersionOrGreater(6, 1, 0);
#else
    OSVERSIONINFO osvi = { 0 };
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osvi);
    return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 1) || osvi.dwMajorVersion > 6));
#endif
}

static inline const char *GetFullPath(const char *path, char *buffer, size_t nSize) {
    if (PathIsRelativeA(path) == FALSE)
        return path;

    _fullpath(buffer, path, nSize);
    return buffer;
}
static inline const WCHAR *GetFullPath(const WCHAR *path, WCHAR *buffer, size_t nSize) {
    if (PathIsRelativeW(path) == FALSE)
        return path;

    _wfullpath(buffer, path, nSize);
    return buffer;
}
//文字列の置換に必要な領域を計算する
static size_t calc_replace_mem_required(char *str, const char *old_str, const char *new_str) {
    size_t size = strlen(str) + 1;
    const int move_len = strlen(new_str) - strlen(old_str);
    if (move_len <= 0)
        return size;
    char *p = str;
    while ((p == strstr(p, old_str)) != NULL)
        size += move_len;
    return size;
}
static size_t calc_replace_mem_required(WCHAR *str, const WCHAR *old_str, const WCHAR *new_str) {
    size_t size = wcslen(str) + 1;
    const int move_len = wcslen(new_str) - wcslen(old_str);
    if (move_len <= 0)
        return size;
    WCHAR *p = str;
    while ((p == wcsstr(p, old_str)) != NULL)
        size += move_len;
    return size;
}
//文字列の置換 str内で置き換える 置換を実行した回数を返す
static inline int replace(char *str, size_t nSize, const char *old_str, const char *new_str) {
    char *c = str;
    char *p = NULL;
    char *fin = str + strlen(str) + 1;//null文字まで
    char * const limit = str + nSize;
    int count = 0;
    const size_t old_len = strlen(old_str);
    const size_t new_len = strlen(new_str);
    const int move_len = (int)(new_len - old_len);
    if (old_len) {
        while ((p = strstr(c, old_str)) != NULL) {
            if (move_len) {
                if (fin + move_len > limit)
                    break;
                memmove((c = p + new_len), p + old_len, (fin - (p + old_len)) * sizeof(str[0]));
                fin += move_len;
            }
            memcpy(p, new_str, new_len * sizeof(str[0]));
            count++;
        }
    }
    return count;
}
static inline int replace(WCHAR *str, size_t nSize, const WCHAR *old_str, const WCHAR *new_str) {
    WCHAR *c = str;
    WCHAR *p = NULL;
    WCHAR *fin = str + wcslen(str) + 1;//null文字まで
    WCHAR * const limit = str + nSize;
    int count = 0;
    const size_t old_len = wcslen(old_str);
    const size_t new_len = wcslen(new_str);
    const int move_len = (int)(new_len - old_len);
    if (old_len) {
        while ((p = wcsstr(c, old_str)) != NULL) {
            if (move_len) {
                if (fin + move_len > limit)
                    break;
                memmove((c = p + new_len), p + old_len, (fin - (p + old_len)) * sizeof(str[0]));
                fin += move_len;
            }
            memcpy(p, new_str, new_len * sizeof(str[0]));
            count++;
        }
    }
    return count;
}

//ファイル名(拡張子除く)の後ろに文字列を追加する
static inline void apply_appendix(char *new_filename, size_t new_filename_size, const char *orig_filename, const char *appendix) {
    if (new_filename != orig_filename)
        strcpy_s(new_filename, new_filename_size, orig_filename);
    strcpy_s(PathFindExtensionA(new_filename), new_filename_size - (PathFindExtensionA(new_filename) - new_filename), appendix);
}
static inline void apply_appendix(WCHAR *new_filename, size_t new_filename_size, const WCHAR *orig_filename, const WCHAR *appendix) {
    if (new_filename != orig_filename)
        wcscpy_s(new_filename, new_filename_size, orig_filename);
    wcscpy_s(PathFindExtensionW(new_filename), new_filename_size - (PathFindExtensionW(new_filename) - new_filename), appendix);
}

static inline void insert_before_ext(char *filename, size_t nSize, const char *insert_str) {
    char *ext = PathFindExtensionA(filename);
    if (ext == NULL)
        strcat_s(filename, nSize, insert_str);
    else {
        const size_t insert_len = strlen(insert_str);
        const size_t filename_len = strlen(filename);
        if (nSize > filename_len + insert_len) {
            memmove(ext + insert_len, ext, sizeof(insert_str[0]) * (strlen(ext)+1));
            memcpy(ext, insert_str, sizeof(insert_str[0]) * insert_len);
        }
    }
}
static inline void insert_before_ext(WCHAR *filename, size_t nSize, const WCHAR *insert_str) {
    WCHAR *ext = PathFindExtensionW(filename);
    if (ext == NULL)
        wcscat_s(filename, nSize, insert_str);
    else {
        const size_t insert_len = wcslen(insert_str);
        const size_t filename_len = wcslen(filename);
        if (nSize > filename_len + insert_len) {
            memmove(ext + insert_len, ext, sizeof(insert_str[0]) * (wcslen(ext)+1));
            memcpy(ext, insert_str, sizeof(insert_str[0]) * insert_len);
        }
    }
}
static inline void insert_before_ext(char *filename, size_t nSize, int insert_num) {
    char tmp[22];
    sprintf_s(tmp, _countof(tmp), "%d", insert_num);
    insert_before_ext(filename, nSize, tmp);
}
static inline void insert_before_ext(WCHAR *filename, size_t nSize, int insert_num) {
    WCHAR tmp[22];
    swprintf_s(tmp, _countof(tmp), L"%d", insert_num);
    insert_before_ext(filename, nSize, tmp);
}

//拡張子が一致するか確認する
static inline BOOL check_ext(const char *filename, const char *ext) {
    return (_stricmp(PathFindExtensionA(filename), ext) == NULL) ? TRUE : FALSE;
}
static inline BOOL check_ext(const WCHAR *filename, const WCHAR *ext) {
    return (_wcsicmp(PathFindExtensionW(filename), ext) == NULL) ? TRUE : FALSE;
}

//パスの拡張子を変更する
static inline void change_ext(char *filename, size_t nSize, const char *ext) {
    size_t len_to_ext;
    char *ext_ptr = PathFindExtensionA(filename);
    len_to_ext = (ext_ptr) ? ext_ptr - filename : strlen(filename);
    strcpy_s(filename + len_to_ext, nSize - len_to_ext, ext);
}
static inline void change_ext(WCHAR *filename, size_t nSize, const WCHAR *ext) {
    size_t len_to_ext;
    WCHAR *ext_ptr = PathFindExtensionW(filename);
    len_to_ext = (ext_ptr) ? ext_ptr - filename : wcslen(filename);
    wcscpy_s(filename + len_to_ext, nSize - len_to_ext, ext);
}

//ルートディレクトリを取得
static inline BOOL PathGetRoot(const char *path, char *root, size_t nSize) {
    if (PathIsRelativeA(path) == FALSE)
        strcpy_s(root, nSize, path);
    else
        _fullpath(root, path, nSize);
    return PathStripToRootA(root);
}
static inline BOOL PathGetRoot(const WCHAR *path, WCHAR *root, size_t nSize) {
    if (PathIsRelativeW(path) == FALSE)
        wcscpy_s(root, nSize, path);
    else
        _wfullpath(root, path, nSize);
    return PathStripToRootW(root);
}

//パスのルートが存在するかどうか
static BOOL PathRootExists(const char *path) {
    if (path == NULL)
        return FALSE;
    char root[MAX_PATH_LEN];
    return (PathGetRoot(path, root, _countof(root)) && PathIsDirectoryA(root));
}
static BOOL PathRootExists(const WCHAR *path) {
    if (path == NULL)
        return FALSE;
    WCHAR root[MAX_PATH_LEN];
    return (PathGetRoot(path, root, _countof(root)) && PathIsDirectoryW(root));
}

//PathRemoveFileSpecFixedがVistaでは5C問題を発生させるため、その回避策
static BOOL PathRemoveFileSpecFixed(char *path) {
    char *ptr = PathFindFileNameA(path);
    if (path == ptr)
        return FALSE;
    *(ptr - 1) = '\0';
    return TRUE;
}
static BOOL PathRemoveFileSpecFixed(WCHAR *path) {
    WCHAR *ptr = PathFindFileNameW(path);
    if (path == ptr)
        return FALSE;
    *(ptr - 1) = L'\0';
    return TRUE;
}

//フォルダがあればOK、なければ作成する
static BOOL DirectoryExistsOrCreate(const char *dir) {
    if (PathIsDirectoryA(dir))
        return TRUE;
    return (PathRootExists(dir) && CreateDirectoryA(dir, NULL) != NULL) ? TRUE : FALSE;
}
static BOOL DirectoryExistsOrCreate(const WCHAR *dir) {
    if (PathIsDirectoryW(dir))
        return TRUE;
    return (PathRootExists(dir) && CreateDirectoryW(dir, NULL) != NULL) ? TRUE : FALSE;
}

//ファイルの存在と0byteより大きいかを確認
static BOOL FileExistsAndHasSize(const char *path) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    return GetFileAttributesExA(path, GetFileExInfoStandard, &fd) && ((((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow) > 0;
}
static BOOL FileExistsAndHasSize(const WCHAR *path) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    return GetFileAttributesExW(path, GetFileExInfoStandard, &fd) && ((((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow) > 0;
}

static void PathGetDirectory(char *dir, size_t nSize, const char *path) {
    strcpy_s(dir, nSize, path);
    PathRemoveFileSpecFixed(dir);
}
static void PathGetDirectory(WCHAR *dir, size_t nSize, const WCHAR *path) {
    wcscpy_s(dir, nSize, path);
    PathRemoveFileSpecFixed(dir);
}

static BOOL GetFileSizeDWORD(const char *filepath, DWORD *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    BOOL ret = (GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd)) ? TRUE : FALSE;
    *filesize = (ret) ? fd.nFileSizeLow : 0;
    return ret;
}
static BOOL GetFileSizeDWORD(const WCHAR *filepath, DWORD *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    BOOL ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? TRUE : FALSE;
    *filesize = (ret) ? fd.nFileSizeLow : 0;
    return ret;
}

//64bitでファイルサイズを取得,TRUEで成功
static BOOL GetFileSizeUInt64(const char *filepath, UINT64 *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    BOOL ret = (GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd)) ? TRUE : FALSE;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}
static BOOL GetFileSizeUInt64(const WCHAR *filepath, UINT64 *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    BOOL ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? TRUE : FALSE;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}

static UINT64 GetFileLastUpdate(const char *filepath) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd);
    return ((UINT64)fd.ftLastWriteTime.dwHighDateTime << 32) + (UINT64)fd.ftLastWriteTime.dwLowDateTime;
}
static UINT64 GetFileLastUpdate(const WCHAR *filepath) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd);
    return ((UINT64)fd.ftLastWriteTime.dwHighDateTime << 32) + (UINT64)fd.ftLastWriteTime.dwLowDateTime;
}

static inline size_t append_str(char **dst, size_t *nSize, const char *append) {
    size_t len = strlen(append);
    if (*nSize - 1 <= len)
        return 0;
    memcpy(*dst, append, (len + 1) * sizeof(dst[0][0]));
    *dst += len;
    *nSize -= len;
    return len;
}
static inline size_t append_str(WCHAR **dst, size_t *nSize, const WCHAR *append) {
    size_t len = wcslen(append);
    if (*nSize - 1 <= len)
        return 0;
    memcpy(*dst, append, (len + 1) * sizeof(dst[0][0]));
    *dst += len;
    *nSize -= len;
    return len;
}

//多くのPath～関数はMAX_LEN(260)以上でもOKだが、一部は不可
//これもそのひとつ
static inline BOOL PathAddBackSlashLong(char *dir) {
    size_t len = strlen(dir);
    if (dir[len-1] != '\\') {
        dir[len] = '\\';
        dir[len+1] = '\0';
        return TRUE;
    }
    return FALSE;
}
static inline BOOL PathAddBackSlashLong(WCHAR *dir) {
    size_t len = wcslen(dir);
    if (dir[len-1] != L'\\') {
        dir[len] = L'\\';
        dir[len+1] = L'\0';
        return TRUE;
    }
    return FALSE;
}
//PathCombineもMAX_LEN(260)以上不可
static BOOL PathCombineLong(char *path, size_t nSize, const char *dir, const char *filename) {
    size_t dir_len;
    if (path == dir) {
        dir_len = strlen(path);
    } else {
        dir_len = strlen(dir);
        if (nSize <= dir_len)
            return FALSE;

        memcpy(path, dir, (dir_len+1) * sizeof(path[0]));
    }
    dir_len += PathAddBackSlashLong(path);

    size_t filename_len = strlen(filename);
    if (nSize - dir_len <= filename_len)
        return FALSE;
    memcpy(path + dir_len, filename, (filename_len+1) * sizeof(path[0]));
    return TRUE;
}
static BOOL PathCombineLong(WCHAR *path, size_t nSize, const WCHAR *dir, const WCHAR *filename) {
    size_t dir_len;
    if (path == dir) {
        dir_len = wcslen(path);
    } else {
        dir_len = wcslen(dir);
        if (nSize <= dir_len)
            return FALSE;

        memcpy(path, dir, (dir_len+1) * sizeof(path[0]));
    }
    dir_len += PathAddBackSlashLong(path);

    size_t filename_len = wcslen(filename);
    if (nSize - dir_len <= filename_len)
        return FALSE;
    memcpy(path + dir_len, filename, (filename_len+1) * sizeof(path[0]));
    return TRUE;
}

static BOOL GetPathRootFreeSpace(const char *path, UINT64 *freespace) {
    //指定されたドライブが存在するかどうか
    char temp_root[MAX_PATH_LEN];
    strcpy_s(temp_root, _countof(temp_root), path);
    PathStripToRootA(temp_root);
    //ドライブの空き容量取得
    ULARGE_INTEGER drive_avail_space = { 0 };
    if (GetDiskFreeSpaceExA(temp_root, &drive_avail_space, NULL, NULL)) {
        *freespace = drive_avail_space.QuadPart;
        return TRUE;
    }
    return FALSE;
}
static BOOL GetPathRootFreeSpace(const WCHAR *path, UINT64 *freespace) {
    //指定されたドライブが存在するかどうか
    WCHAR temp_root[MAX_PATH_LEN];
    wcscpy_s(temp_root, _countof(temp_root), path);
    PathStripToRootW(temp_root);
    //ドライブの空き容量取得
    ULARGE_INTEGER drive_avail_space = { 0 };
    if (GetDiskFreeSpaceExW(temp_root, &drive_avail_space, NULL, NULL)) {
        *freespace = drive_avail_space.QuadPart;
        return TRUE;
    }
    return FALSE;
}

static inline BOOL PathForceRemoveBackSlash(char *path) {
    size_t len = strlen(path);
    int ret = FALSE;
    if (path != NULL && len) {
        char *ptr = path + len - 1;
        if (*ptr == '\\') {
            *ptr = '\0';
            ret = TRUE;
        }
    }
    return ret;
}
static inline BOOL PathForceRemoveBackSlash(WCHAR *path) {
    size_t len = wcslen(path);
    int ret = FALSE;
    if (path != NULL && len) {
        WCHAR *ptr = path + len - 1;
        if (*ptr == L'\\') {
            *ptr = L'\0';
            ret = TRUE;
        }
    }
    return ret;
}

//pathのbaseDirに対する相対パスを取得する
//baseDirがnull場合は、CurrentDirctoryに対する相対パスを取得する
//attr ... pathのFILE_ATTRIBUTE
static void GetRelativePathTo(char *buf, size_t nSize, const char *path, DWORD attr, const char *baseDir) {
    if (!str_has_char(path) || PathIsRelativeA(path) == TRUE) {
        strcpy_s(buf, nSize, path);
    } else {
        char baseDirPath[MAX_PATH_LEN];
        if (baseDir && str_has_char(baseDir))
            strcpy_s(baseDirPath, _countof(baseDirPath), baseDir);
        else
            GetCurrentDirectoryA(_countof(baseDirPath), baseDirPath);
        PathAddBackSlashLong(baseDirPath);

        if (FALSE == PathRelativePathToA(buf, baseDirPath, FILE_ATTRIBUTE_DIRECTORY, path, attr))
            strcpy_s(buf, nSize, path); //失敗

        //WinXPのPathRelativePathToのバグ対策
        // ".\dir" となるべきものが、"\dir"と返ってしまう
        if (buf[0] == '\\') {
            //先頭に'.'を追加して、フルパスに変換後、もとのパスと比較してみる
            char check_fullpath[MAX_PATH_LEN];
            PathRemoveBackslashA(baseDirPath);
            PathCombineA(check_fullpath, baseDirPath, buf + 1);
            if (0 == strcmp(check_fullpath, path)) {
                memmove(buf + 1, buf, sizeof(buf[0]) * (strlen(buf) + 1));
                buf[0] = '.';
            }
        }
    }
}
static inline void GetRelativePathTo(char *buf, size_t nSize, const char *path, DWORD attr) {
    GetRelativePathTo(buf, nSize, path, attr, NULL);
}
static inline void GetRelativePathTo(char *buf, size_t nSize, const char *path) {
    GetRelativePathTo(buf, nSize, path, FILE_ATTRIBUTE_NORMAL, NULL);
}
static inline void GetRelativePathTo(char *path, size_t nSize, DWORD attr) {
    size_t len = strlen(path);
    char *target_path = (char *)malloc((len + 1) * sizeof(target_path[0]));
    memcpy(target_path, path, (len + 1) * sizeof(target_path[0]));
    GetRelativePathTo(path, nSize, target_path, attr, NULL);
    free(target_path);
}
static inline void GetRelativePathTo(char *path, size_t nSize) {
    GetRelativePathTo(path, nSize, FILE_ATTRIBUTE_NORMAL);
}
static void GetRelativePathTo(WCHAR *buf, size_t nSize, const WCHAR *path, DWORD attr, const WCHAR *baseDir) {
    if (!str_has_char(path) || PathIsRelativeW(path) == TRUE) {
        wcscpy_s(buf, nSize, path);
    } else {
        WCHAR baseDirPath[MAX_PATH_LEN];
        if (baseDir && str_has_char(baseDir))
            wcscpy_s(baseDirPath, _countof(baseDirPath), baseDir);
        else
            GetCurrentDirectoryW(_countof(baseDirPath), baseDirPath);
        PathAddBackSlashLong(baseDirPath);

        if (FALSE == PathRelativePathToW(buf, baseDirPath, FILE_ATTRIBUTE_DIRECTORY, path, attr))
            wcscpy_s(buf, nSize, path); //失敗

        //WinXPのPathRelativePathToのバグ対策
        // ".\dir" となるべきものが、"\dir"と返ってしまう
        if (buf[0] == L'\\') {
            //先頭に'.'を追加して、フルパスに変換後、もとのパスと比較してみる
            WCHAR check_fullpath[MAX_PATH_LEN];
            PathRemoveBackslashW(baseDirPath);
            PathCombineW(check_fullpath, baseDirPath, buf + 1);
            if (0 == wcscmp(check_fullpath, path)) {
                memmove(buf + 1, buf, sizeof(buf[0]) * (wcslen(buf) + 1));
                buf[0] = L'.';
            }
        }
    }
}
static inline void GetRelativePathTo(WCHAR *buf, size_t nSize, const WCHAR *path, DWORD attr) {
    GetRelativePathTo(buf, nSize, path, attr, NULL);
}
static inline void GetRelativePathTo(WCHAR *buf, size_t nSize, const WCHAR *path) {
    GetRelativePathTo(buf, nSize, path, FILE_ATTRIBUTE_NORMAL, NULL);
}
static inline void GetRelativePathTo(WCHAR *path, size_t nSize, DWORD attr) {
    size_t len = wcslen(path);
    WCHAR *target_path = (WCHAR *)malloc((len + 1) * sizeof(target_path[0]));
    memcpy(target_path, path, (len + 1) * sizeof(target_path[0]));
    GetRelativePathTo(path, nSize, target_path, attr, NULL);
    free(target_path);
}
static inline void GetRelativePathTo(WCHAR *path, size_t nSize) {
    GetRelativePathTo(path, nSize, FILE_ATTRIBUTE_NORMAL);
}

static inline BOOL check_process_exitcode(PROCESS_INFORMATION *pi) {
    DWORD exit_code;
    if (!GetExitCodeProcess(pi->hProcess, &exit_code))
        return TRUE;
    return exit_code != 0;
}

static BOOL swap_file(const char *fileA, const char *fileB) {
    if (!PathFileExistsA(fileA) || !PathFileExistsA(fileB))
        return FALSE;

    char filetemp[MAX_PATH_LEN];
    char appendix[MAX_APPENDIX_LEN];
    for (int i = 0; !i || PathFileExistsA(filetemp); i++) {
        sprintf_s(appendix, _countof(appendix), ".swap%d.tmp", i);
        apply_appendix(filetemp, _countof(filetemp), fileA, appendix);
    }
    if (rename(fileA, filetemp))
        return FALSE;
    if (rename(fileB, fileA))
        return FALSE;
    if (rename(filetemp, fileB))
        return FALSE;
    return TRUE;
}
static BOOL swap_file(const WCHAR *fileA, const WCHAR *fileB) {
    if (!PathFileExistsW(fileA) || !PathFileExistsW(fileB))
        return FALSE;

    WCHAR filetemp[MAX_PATH_LEN];
    WCHAR appendix[MAX_APPENDIX_LEN];
    for (int i = 0; !i || PathFileExistsW(filetemp); i++) {
        swprintf_s(appendix, _countof(appendix), L".swap%d.tmp", i);
        apply_appendix(filetemp, _countof(filetemp), fileA, appendix);
    }
    if (_wrename(fileA, filetemp))
        return FALSE;
    if (_wrename(fileB, fileA))
        return FALSE;
    if (_wrename(filetemp, fileB))
        return FALSE;
    return TRUE;
}
//最後に"\"なしで戻る
static inline void get_aviutl_dir(char *aviutl_dir, size_t nSize) {
    GetModuleFileNameA(NULL, aviutl_dir, (DWORD)nSize);
    PathRemoveFileSpecFixed(aviutl_dir);
}
static inline void get_aviutl_dir(WCHAR *aviutl_dir, size_t nSize) {
    GetModuleFileNameW(NULL, aviutl_dir, (DWORD)nSize);
    PathRemoveFileSpecFixed(aviutl_dir);
}
static inline void get_auo_path(char *auo_path, size_t nSize) {
    GetModuleFileNameA(GetModuleHandleA(AUO_NAME), auo_path, (DWORD)nSize);
}
static inline void get_auo_path(WCHAR *auo_path, size_t nSize) {
    GetModuleFileNameW(GetModuleHandleW(AUO_NAME_W), auo_path, (DWORD)nSize);
}

static inline int replace_cmd_CRLF_to_Space(char *cmd, size_t nSize) {
    int ret = 0;
    ret += replace(cmd, nSize, "\r\n", " ");
    ret += replace(cmd, nSize, "\r",   " ");
    ret += replace(cmd, nSize, "\n",   " ");
    return ret;
}
static inline int replace_cmd_CRLF_to_Space(WCHAR *cmd, size_t nSize) {
    int ret = 0;
    ret += replace(cmd, nSize, L"\r\n", L" ");
    ret += replace(cmd, nSize, L"\r",   L" ");
    ret += replace(cmd, nSize, L"\n",   L" ");
    return ret;
}

//文字列先頭がBOM文字でないか確認する
DWORD check_bom(const void* chr);

//与えられた文字列から主に日本語について文字コード判定を行う
DWORD get_code_page(const void *str, DWORD size_in_byte);

//CODE_PAGE_SJIS / CODE_PAGE_UTF8 / CODE_PAGE_EUC_JP についてのみ判定を行う
DWORD jpn_check(const void *str, DWORD size_in_byte);

//IMultipleLanguge2 の DetectInoutCodePageがたまに的外れな「西ヨーロッパ言語」を返すので
//西ヨーロッパ言語 なら Shift-JIS にしてしまう
BOOL fix_ImulL_WesternEurope(UINT *code_page);

//cmd中のtarget_argを抜き出し削除する
//del_valueが+1ならその後の値を削除する、-1ならその前の値を削除する
//値を削除できたらTRUEを返す
BOOL del_arg(char *cmd, char *target_arg, int del_arg_delta);

//TargetProcessIdに指定したプロセスのスレッドのうち、
//スレッドのModuleがTargetModuleに指定した文字列に一致した場合(_strnicmpによる比較)
//スレッド優先度をThreadPriorityに設定する
//TargetModuleがNULLならTargetProcessIdの全スレッドに適用
BOOL SetThreadPriorityForModule(DWORD TargetProcessId, const char *TargetModule, int ThreadPriority);
BOOL SetThreadAffinityForModule(DWORD TargetProcessId, const char *TargetModule, DWORD_PTR ThreadAffinityMask);

BOOL getProcessorCount(DWORD *physical_processor_core, DWORD *logical_processor_core);

const TCHAR *getOSVersion();

#endif //_AUO_UTIL_H_
