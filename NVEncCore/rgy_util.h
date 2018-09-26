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
#ifndef __RGY_UTIL_H__
#define __RGY_UTIL_H__

#include "rgy_tchar.h"
#include <emmintrin.h>
#if defined(_WIN32) || defined(_WIN64)
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <memory>
#include <algorithm>
#include <climits>
#include <map>
#include <list>
#include <sstream>
#include <functional>
#include <type_traits>
#include "rgy_err.h"
#include "rgy_osdep.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "convert_csp.h"

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(x)
#endif

typedef std::basic_string<TCHAR> tstring;
using std::vector;
using std::unique_ptr;
using std::shared_ptr;

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define ALIGN(x,align) (((x)+((align)-1))&(~((align)-1)))
#define ALIGN16(x) (((x)+15)&(~15))
#define ALIGN32(x) (((x)+31)&(~31))

#define MAP_PAIR_0_1_PROTO(prefix, name0, type0, name1, type1) \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0); \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1);

#define MAP_PAIR_0_1(prefix, name0, type0, name1, type1, map_pair, default0, default1) \
    RGY_NOINLINE \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var0](std::pair<type0, type1> a) { \
            return a.first == var0; \
        }); \
        return (ret == map_pair.end()) ? default1 : ret->second; \
    } \
    RGY_NOINLINE  \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var1](std::pair<type0, type1> a) { \
            return a.second == var1; \
        }); \
        return (ret == map_pair.end()) ? default0 : ret->first; \
    }

typedef long long lls;
typedef unsigned long long llu;

#define RGY_MEMSET_ZERO(x) { memset(&(x), 0, sizeof(x)); }

template<typename T, size_t size>
std::vector<T> make_vector(T(&ptr)[size]) {
    return std::vector<T>(ptr, ptr + size);
}
template<typename T, size_t size>
std::vector<T> make_vector(const T(&ptr)[size]) {
    return std::vector<T>(ptr, ptr + size);
}
template<typename T0, typename T1>
std::vector<T0> make_vector(const T0 *ptr, T1 size) {
    static_assert(std::is_integral<T1>::value == true, "T1 should be integral");
    return (ptr && size) ? std::vector<T0>(ptr, ptr + size) : std::vector<T0>();
}
template<typename T0, typename T1>
std::vector<T0> make_vector(T0 *ptr, T1 size) {
    static_assert(std::is_integral<T1>::value == true, "T1 should be integral");
    return (ptr && size) ? std::vector<T0>(ptr, ptr + size) : std::vector<T0>();
}
template<typename T, typename ...Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args&&... args) {
    return std::array<T, sizeof...(Args)>{ static_cast<Args&&>(args)... };
}
template<typename T, std::size_t N>
constexpr std::size_t array_size(const std::array<T, N>&) {
    return N;
}
template<typename T, std::size_t N>
constexpr std::size_t array_size(T(&)[N]) {
    return N;
}
template<typename T>
void vector_cat(vector<T>& v1, const vector<T>& v2) {
    if (v2.size()) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    }
}
template<typename T>
void vector_cat(std::vector<T>& v1, const T *ptr, size_t nCount) {
    if (nCount) {
        size_t currentSize = v1.size();
        v1.resize(currentSize + nCount);
        memcpy(v1.data() + currentSize, ptr, sizeof(T) * nCount);
    }
}
template<typename T>
static void rgy_free(T& ptr) {
    static_assert(std::is_pointer<T>::value == true, "T should be pointer");
    if (ptr) {
        free(ptr);
        ptr = nullptr;
    }
}

template<typename T>
using unique_ptr_custom = std::unique_ptr<T, std::function<void(T*)>>;

struct aligned_malloc_deleter {
    void operator()(void* ptr) const {
        _aligned_free(ptr);
    }
};

struct malloc_deleter {
    void operator()(void* ptr) const {
        free(ptr);
    }
};

struct fp_deleter {
    void operator()(FILE* fp) const {
        if (fp) {
            fflush(fp);
            fclose(fp);
        }
    }
};

struct handle_deleter {
    void operator()(HANDLE handle) const {
        if (handle) {
#if defined(_WIN32) || defined(_WIN64)
            CloseHandle(handle);
#endif //#if defined(_WIN32) || defined(_WIN64)
        }
    }
};

struct module_deleter {
    void operator()(void *hmodule) const {
        if (hmodule) {
#if defined(_WIN32) || defined(_WIN64)
            FreeLibrary((HMODULE)hmodule);
#endif //#if defined(_WIN32) || defined(_WIN64)
        }
    }
};

template<typename T>
static inline T rgy_gcd(T a, T b) {
    static_assert(std::is_integral<T>::value, "rgy_gcd is defined only for integer.");
    if (a == 0) return b;
    if (b == 0) return a;
    T c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return b;
}

template<typename T>
static inline T rgy_gcd(std::pair<T, T> int2) {
    return rgy_gcd(int2.first, int2.second);
}

template<typename T>
static inline T rgy_lcm(T a, T b) {
    static_assert(std::is_integral<T>::value, "rgy_lcm is defined only for integer.");
    if (a == 0) return 0;
    if (b == 0) return 0;
    T gcd = rgy_gcd(a, b);
    a /= gcd;
    b /= gcd;
    return a * b * gcd;
}

template<typename T>
static inline T rgy_lcm(std::pair<T, T> int2) {
    return rgy_lcm(int2.first, int2.second);
}

template<typename T>
static inline void rgy_reduce(T& a, T& b) {
    static_assert(std::is_integral<T>::value, "rgy_reduce is defined only for integer.");
    if (a == 0 || b == 0) return;
    T gcd = rgy_gcd(a, b);
    a /= gcd;
    b /= gcd;
}

template<typename T>
static inline void rgy_reduce(std::pair<T, T>& int2) {
    rgy_reduce(int2.first, int2.second);
}

template<typename T>
class rgy_rational {
    static_assert(std::is_integral<T>::value, "rgy_rational is defined only for integer.");
private:
    T num, den;
public:
    rgy_rational() : num(0), den(1) {}
    rgy_rational(T _num, T _den) : num(_num), den(_den) { reduce(); }
    rgy_rational(const rgy_rational<T>& r) : num(r.num), den(r.den) { reduce(); }
    rgy_rational<T>& operator=(const rgy_rational<T> &r) { num = r.num; den = r.den; reduce(); return *this; }
    bool is_valid() const { return den != 0; };
    T n() const {
        return this->num;
    }
    T d() const {
        return this->den;
    }
    float qfloat() const {
        return (float)qdouble();
    }
    double qdouble() const {
        return (double)num / (double)den;
    }
    void reduce() {
        if (den == 0) {
            return;
        }
        rgy_reduce(num, den);
        if (den < 0) {
            num = -num;
            den = -den;
        }
    }
    rgy_rational<T> inv() const {
        rgy_rational<T> tmp(den, num);
        if (tmp.den == 0) {
            tmp.den = 0;
            tmp.num = 0;
        } else if (tmp.den < 0) {
            tmp.num = -tmp.num;
            tmp.den = -tmp.den;
        }
        return tmp;
    }

    rgy_rational<T> operator+ () {
        return *this;
    }
    rgy_rational<T> operator- () {
        return rgy_rational<T>(-1 * this->num, this->den);
    }

    rgy_rational<T>& operator+= (const rgy_rational<T>& r) {
        if (r.den == 0 || den == 0) {
            den = 0;
            num = 0;
            return *this;
        }

        T gcd0 = rgy_gcd(den, r.den);
        den /= gcd0;
        T tmp = r.den / gcd0;
        num = num * tmp + r.num * den;
        T gcd1 = rgy_gcd(num, gcd0);
        num /= gcd1;
        tmp = r.den / gcd1;
        den *= tmp;

        return *this;
    }
    rgy_rational<T>& operator-= (const rgy_rational<T>& r) {
        rgy_rational<T> tmp(r);
        tmp.num *= -1;
        *this += tmp;
        return *this;
    }
    rgy_rational<T>& operator*= (const rgy_rational<T>& r) {
        if (r.den == 0 || den == 0) {
            den = 0;
            num = 0;
            return *this;
        }
        T gcd0 = rgy_gcd(num, r.den);
        T gcd1 = rgy_gcd(den, r.num);
        T a0 = num / gcd0;
        T a1 = r.num / gcd1;
        T b0 = den / gcd1;
        T b1 = r.den / gcd0;
        num = a0 * a1;
        den = b0 * b1;

        if (den < 0) {
            num = -num;
            den = -den;
        }
        return *this;
    }
    rgy_rational<T>& operator/= (const rgy_rational<T>& r) {
        *this *= r.inv();
        return *this;
    }

    rgy_rational<T>& operator+= (const T& i) {
        num += i * den;
        return *this;
    }
    rgy_rational<T>& operator-= (const T& i) {
        num -= i * den;
        return *this;
    }
    rgy_rational<T>& operator*= (const T& i) {
        T gcd = rgy_gcd(i, den);
        num *= i / gcd;
        den /= gcd;
        return *this;
    }
    rgy_rational<T>& operator/= (const T& i) {
        if (i == 0) {
            num = 0;
            den = 0;
        } else if (num != 0) {
            T gcd = rgy_gcd(num, i);
            num /= gcd;
            den *= i / gcd;
            if (den < 0) {
                num = -num;
                den = -den;
            }
        }
        return *this;
    }

    template<typename Arg>
    rgy_rational<T> operator + (const Arg& a) {
        rgy_rational<T> t(*this);
        t += a;
        return t;
    }
    template<typename Arg>
    rgy_rational<T> operator - (const Arg& a) {
        rgy_rational<T> t(*this);
        t -= a;
        return t;
    }
    template<typename Arg>
    rgy_rational<T> operator * (const Arg& a) {
        rgy_rational<T> t(*this);
        t *= a;
        return t;
    }
    template<typename Arg>
    rgy_rational<T> operator / (const Arg& a) {
        rgy_rational<T> t(*this);
        t /= a;
        return t;
    }
    const rgy_rational<T>& operator++() { num += den; return *this; }
    const rgy_rational<T>& operator--() { num -= den; return *this; }

    bool operator== (const rgy_rational<T>& r) const {
        return ((num == r.num) && (den == r.den));
    }
    bool operator!= (const rgy_rational<T>& r) const {
        return ((num != r.num) || (den != r.den));
    }

    std::string print() const {
        std::stringstream ss;
        ss << num << "/" << den;
        return ss.str();
    }

    std::wstring printw() const {
        std::wstringstream ss;
        ss << num << "/" << den;
        return ss.str();
    }

    tstring printt() const {
#if _UNICODE
        return printw();
#else
        return print();
#endif
    }
};

#if UNICODE
#define to_tstring to_wstring
#else
#define to_tstring to_string
#endif

typedef std::basic_stringstream<TCHAR> TStringStream;

unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage = CP_THREAD_ACP);
std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage = CP_THREAD_ACP);
std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage = CP_THREAD_ACP);
std::wstring char_to_wstring(const char *str, uint32_t = CP_THREAD_ACP);
std::wstring char_to_wstring(const std::string& str, uint32_t codepage = CP_THREAD_ACP);
#if defined(_WIN32) || defined(_WIN64)
std::wstring strsprintf(const WCHAR* format, ...);

std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to);
std::wstring GetFullPath(const WCHAR *path);
bool rgy_get_filesize(const WCHAR *filepath, uint64_t *filesize);
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path);
std::wstring PathRemoveExtensionS(const std::wstring& path);
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename);
std::string PathCombineS(const std::string& dir, const std::string& filename);
bool CreateDirectoryRecursive(const WCHAR *dir);
#endif

std::wstring tchar_to_wstring(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
std::wstring tchar_to_wstring(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int char_to_tstring(tstring& tstr, const char *str, uint32_t codepage = CP_THREAD_ACP);
tstring char_to_tstring(const char *str, uint32_t codepage = CP_THREAD_ACP);
tstring char_to_tstring(const std::string& str, uint32_t codepage = CP_THREAD_ACP);
unsigned int wstring_to_tstring(const WCHAR *wstr, tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
tstring wstring_to_tstring(const WCHAR *wstr, uint32_t codepage = CP_THREAD_ACP);
tstring wstring_to_tstring(const std::wstring& wstr, uint32_t codepage = CP_THREAD_ACP);
std::string strsprintf(const char* format, ...);
std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim, bool bTrim = false);
std::vector<std::string> split(const std::string &str, const std::string &delim, bool bTrim = false);
std::string lstrip(const std::string& string, const char* trim = " \t\v\r\n");
std::string rstrip(const std::string& string, const char* trim = " \t\v\r\n");
std::string trim(const std::string& string, const char* trim = " \t\v\r\n");
std::wstring lstrip(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");
std::wstring rstrip(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");
std::wstring trim(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");

std::string str_replace(std::string str, const std::string& from, const std::string& to);
std::string GetFullPath(const char *path);
bool rgy_get_filesize(const char *filepath, uint64_t *filesize);
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path);
std::string PathRemoveExtensionS(const std::string& path);
bool CreateDirectoryRecursive(const char *dir);

tstring print_time(double time);

static inline uint16_t readUB16(const void *ptr) {
    uint16_t i = *(uint16_t *)ptr;
    return (i >> 8) | (i << 8);
}

static inline uint32_t readUB32(const void *ptr) {
    uint32_t i = *(uint32_t *)ptr;
    return (i >> 24) | ((i & 0xff0000) >> 8) | ((i & 0xff00) << 8) | ((i & 0xff) << 24);
}

static inline uint32_t check_range_unsigned(uint32_t value, uint32_t min, uint32_t max) {
    return (value - min) <= (max - min);
}

static inline uint32_t popcnt32(uint32_t bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
}

static inline uint32_t popcnt64(uint64_t bits) {
    bits = (bits & 0x5555555555555555) + (bits >> 1 & 0x5555555555555555);
    bits = (bits & 0x3333333333333333) + (bits >> 2 & 0x3333333333333333);
    bits = (bits & 0x0f0f0f0f0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f0f0f0f0f);
    bits = (bits & 0x00ff00ff00ff00ff) + (bits >> 8 & 0x00ff00ff00ff00ff);
    bits = (bits & 0x0000ffff0000ffff) + (bits >>16 & 0x0000ffff0000ffff);
    bits = (bits & 0x00000000ffffffff) + (bits >>32 & 0x00000000ffffffff);
    return (uint32_t)bits;
}

template<typename type>
static std::basic_string<type> repeatStr(std::basic_string<type> str, int count) {
    std::basic_string<type> ret;
    for (int i = 0; i < count; i++) {
        ret += str;
    }
    return ret;
}

static tstring fourccToStr(uint32_t nFourCC) {
    tstring fcc;
    for (int i = 0; i < 4; i++) {
        fcc.push_back((TCHAR)*(i + (char*)&nFourCC));
    }
    return fcc;
}

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list);
bool check_ext(const tstring& filename, const std::vector<const char*>& ext_list);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return (_tcsicmp(PathFindExtension(filename), ext) == 0) ? TRUE : FALSE;
}

int rgy_print_stderr(int log_level, const TCHAR *mes, HANDLE handle = NULL);

#if defined(_WIN32) || defined(_WIN64)
tstring getOSVersion(OSVERSIONINFOEXW *osinfo = nullptr);
#else
tstring getOSVersion();
#endif
BOOL rgy_is_64bit_os();
uint64_t getPhysicalRamSize(uint64_t *ramUsed);
tstring getEnviromentInfo(bool add_ram_info = true);

BOOL check_OS_Win8orLater();

static void RGY_FORCEINLINE sse_memcpy(uint8_t *dst, const uint8_t *src, int size) {
    if (size < 64) {
        memcpy(dst, src, size);
        return;
    }
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128 x0, x1, x2, x3;
    const int start_align_diff = (int)((size_t)dst & 15);
    if (start_align_diff) {
        x0 = _mm_loadu_ps((const float*)src);
        _mm_storeu_ps((float*)dst, x0);
        dst += 16 - start_align_diff;
        src += 16 - start_align_diff;
    }
    for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_ps((const float*)(src +  0));
        x1 = _mm_loadu_ps((const float*)(src + 16));
        x2 = _mm_loadu_ps((const float*)(src + 32));
        x3 = _mm_loadu_ps((const float*)(src + 48));
        _mm_store_ps((float*)(dst +  0), x0);
        _mm_store_ps((float*)(dst + 16), x1);
        _mm_store_ps((float*)(dst + 32), x2);
        _mm_store_ps((float*)(dst + 48), x3);
    }
    uint8_t *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_ps((const float*)(src +  0));
    x1 = _mm_loadu_ps((const float*)(src + 16));
    x2 = _mm_loadu_ps((const float*)(src + 32));
    x3 = _mm_loadu_ps((const float*)(src + 48));
    _mm_storeu_ps((float*)(dst_tmp +  0), x0);
    _mm_storeu_ps((float*)(dst_tmp + 16), x1);
    _mm_storeu_ps((float*)(dst_tmp + 32), x2);
    _mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

//確保できなかったら、サイズを小さくして再度確保を試みる (最終的にnMinSizeも確保できなかったら諦める)
size_t malloc_degeneracy(void **ptr, size_t nSize, size_t nMinSize);

const int MAX_FILENAME_LEN = 1024;

enum {
    RGY_LOG_TRACE = -3,
    RGY_LOG_DEBUG = -2,
    RGY_LOG_MORE  = -1,
    RGY_LOG_INFO  = 0,
    RGY_LOG_WARN  = 1,
    RGY_LOG_ERROR = 2,
    RGY_LOG_QUIET = 3,
};

enum RGY_FRAMETYPE : uint32_t {
    RGY_FRAMETYPE_UNKNOWN = 0,

    RGY_FRAMETYPE_I       = 1<<0,
    RGY_FRAMETYPE_P       = 1<<1,
    RGY_FRAMETYPE_B       = 1<<2,

    RGY_FRAMETYPE_REF     = 1<<6,
    RGY_FRAMETYPE_IDR     = 1<<7,

    RGY_FRAMETYPE_xI      = 1<<8,
    RGY_FRAMETYPE_xP      = 1<<9,
    RGY_FRAMETYPE_xB      = 1<<10,

    RGY_FRAMETYPE_xREF    = 1<<14,
    RGY_FRAMETYPE_xIDR    = 1<<15
};

static RGY_FRAMETYPE operator|(RGY_FRAMETYPE a, RGY_FRAMETYPE b) {
    return (RGY_FRAMETYPE)((uint32_t)a | (uint32_t)b);
}

static RGY_FRAMETYPE operator|=(RGY_FRAMETYPE& a, RGY_FRAMETYPE b) {
    a = a | b;
    return a;
}

static RGY_FRAMETYPE operator&(RGY_FRAMETYPE a, RGY_FRAMETYPE b) {
    return (RGY_FRAMETYPE)((uint32_t)a & (uint32_t)b);
}

static RGY_FRAMETYPE operator&=(RGY_FRAMETYPE& a, RGY_FRAMETYPE b) {
    a = (RGY_FRAMETYPE)((uint32_t)a & (uint32_t)b);
    return a;
}

enum RGY_CODEC {
    RGY_CODEC_UNKNOWN = 0,
    RGY_CODEC_H264,
    RGY_CODEC_HEVC,
    RGY_CODEC_MPEG1,
    RGY_CODEC_MPEG2,
    RGY_CODEC_MPEG4,
    RGY_CODEC_VP8,
    RGY_CODEC_VP9,
    RGY_CODEC_VC1,

    RGY_CODEC_NUM,
};

static tstring CodecToStr(RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264:  return _T("H.264/AVC");
    case RGY_CODEC_HEVC:  return _T("H.265/HEVC");
    case RGY_CODEC_MPEG2: return _T("MPEG2");
    case RGY_CODEC_MPEG1: return _T("MPEG1");
    case RGY_CODEC_VC1:   return _T("VC-1");
    case RGY_CODEC_MPEG4: return _T("MPEG4");
    case RGY_CODEC_VP8:   return _T("VP8");
    case RGY_CODEC_VP9:   return _T("VP9");
    default: return _T("unknown");
    }
}

struct RGY_CODEC_DATA {
    RGY_CODEC codec;
    int codecProfile;

    RGY_CODEC_DATA() : codec(RGY_CODEC_UNKNOWN), codecProfile(0) {}
    RGY_CODEC_DATA(RGY_CODEC _codec, int profile) : codec(_codec), codecProfile(profile) {}

    bool operator<(const RGY_CODEC_DATA& right) const {
        return codec == right.codec ? codec < right.codec : codecProfile < right.codecProfile;
    }
    bool operator==(const RGY_CODEC_DATA& right) const {
        return codec == right.codec && codecProfile == right.codecProfile;
    }
};

enum RGY_INPUT_FMT {
    RGY_INPUT_FMT_AUTO = 0,
    RGY_INPUT_FMT_AUO = 0,
    RGY_INPUT_FMT_RAW,
    RGY_INPUT_FMT_Y4M,
    RGY_INPUT_FMT_AVI,
    RGY_INPUT_FMT_AVS,
    RGY_INPUT_FMT_VPY,
    RGY_INPUT_FMT_VPY_MT,
    RGY_INPUT_FMT_AVHW,
    RGY_INPUT_FMT_AVSW,
    RGY_INPUT_FMT_AVANY,
};

#pragma warning(push)
#pragma warning(disable: 4201)
typedef union sInputCrop {
    struct {
        int left, up, right, bottom;
    } e;
    int c[4];
} sInputCrop;
#pragma warning(pop)

static inline bool cropEnabled(const sInputCrop& crop) {
    return 0 != (crop.c[0] | crop.c[1] | crop.c[2] | crop.c[3]);
}

typedef struct CX_DESC {
    const TCHAR *desc;
    int value;
} CX_DESC;

typedef struct FEATURE_DESC {
    const TCHAR *desc;
    uint64_t value;
} FEATURE_DESC;

static const TCHAR *get_chr_from_value(const CX_DESC * list, int v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return list[i].desc;
    return _T("unknown");
}

static int get_cx_index(const CX_DESC * list, int v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return i;
    return 0;
}

static int get_cx_index(const CX_DESC * list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (0 == _tcscmp(list[i].desc, chr))
            return i;
    return 0;
}

static int get_cx_value(const CX_DESC * list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (0 == _tcscmp(list[i].desc, chr))
            return list[i].value;
    return 0;
}

static int PARSE_ERROR_FLAG = INT_MIN;
static int get_value_from_chr(const CX_DESC *list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (_tcsicmp(list[i].desc, chr) == 0)
            return list[i].value;
    return PARSE_ERROR_FLAG;
}

struct VideoVUIInfo {
    int descriptpresent;
    int colorprim;
    int matrix;
    int transfer;
    int format;
    int fullrange;
    int chromaloc;
};

struct VideoInfo {
    //[ i    ] 入力モジュールに渡す際にセットする
    //[    i ] 入力モジュールによってセットされる
    //[ o    ] 出力モジュールに渡す際にセットする

    //[ i (i)] 種類 (RGY_INPUT_FMT_xxx)
    //  i      使用する入力モジュールの種類
    //     i   変更があれば
    RGY_INPUT_FMT type;

    //[(i) i ] 入力横解像度
    uint32_t srcWidth;

    //[(i) i ] 入力縦解像度
    uint32_t srcHeight;

    //[(i)(i)] 入力ピッチ 0なら入力横解像度に同じ
    uint32_t srcPitch;

    uint32_t codedWidth;     //[   (i)]
    uint32_t codedHeight;    //[   (i)]

                             //[      ] 出力解像度
    uint32_t dstWidth;

    //[      ] 出力解像度
    uint32_t dstHeight;

    //[      ] 出力解像度
    uint32_t dstPitch;

    //[    i ] 入力の取得した総フレーム数 (不明なら0)
    int frames;

    //[   (i)] 右shiftすべきビット数
    int shift;

    //[   (i)] 入力の取得したフレームレート (分子)
    int fpsN;

    //[   (i)] 入力の取得したフレームレート (分母)
    int fpsD;

    //[ i    ] 入力時切り落とし
    sInputCrop crop;

    //[   (i)] 入力の取得したアスペクト比
    int sar[2];

    //[(i) i ] 入力色空間 (RGY_CSP_xxx)
    //  i      取得したい色空間をセット
    //     i   入力の取得する色空間
    RGY_CSP csp;

    //[(i)(i)] RGY_PICSTRUCT_xxx
    //  i      ユーザー指定の設定をセット
    //     i   入力の取得した値、あるいはそのまま
    RGY_PICSTRUCT picstruct;

    //[    i ] 入力コーデック (デコード時使用)
    //     i   HWデコード時セット
    RGY_CODEC codec;

    //[      ] 入力コーデックのヘッダー
    void *codecExtra;

    //[      ] 入力コーデックのヘッダーの大きさ
    uint32_t codecExtraSize;

    //[      ] 入力コーデックのレベル
    int codecLevel;

    //[      ] 入力コーデックのプロファイル
    int codecProfile;

    //[      ] 入力コーデックの遅延
    int videoDelay;

    //[      ] 入力コーデックのVUI情報
    VideoVUIInfo vui;
};

void get_dar_pixels(unsigned int* width, unsigned int* height, int sar_w, int sar_h);
std::pair<int, int> get_sar(unsigned int width, unsigned int height, unsigned int darWidth, unsigned int darHeight);
void adjust_sar(int *sar_w, int *sar_h, int width, int height);
int get_h264_sar_idx(std::pair<int, int>sar);
std::pair<int, int> get_h264_sar(int idx);

enum {
    RGY_RESAMPLER_SWR,
    RGY_RESAMPLER_SOXR,
};

enum {
    DELOGO_MODE_REMOVE = 0,
    DELOGO_MODE_ADD,
};

static const int RGY_OUTPUT_THREAD_AUTO = -1;
static const int RGY_AUDIO_THREAD_AUTO = -1;
static const int RGY_INPUT_THREAD_AUTO = -1;

typedef struct {
    int start, fin;
} sTrim;

typedef struct {
    std::vector<sTrim> list;
    int offset;
} sTrimParam;

static const int TRIM_MAX = INT_MAX;
static const int TRIM_OVERREAD_FRAMES = 128;

typedef std::map<RGY_CODEC, vector<RGY_CSP>> CodecCsp;
typedef std::vector<std::pair<int, CodecCsp>> DeviceCodecCsp;

typedef std::vector<std::pair<tstring, tstring>> muxOptList;

static bool inline trim_active(const sTrimParam *pTrim) {
    if (pTrim == nullptr) {
        return false;
    }
    if (pTrim->list.size() == 0) {
        return false;
    }
    if (pTrim->list[0].start == 0 && pTrim->list[0].fin == TRIM_MAX) {
        return false;
    }
    return true;
}

static bool inline frame_inside_range(int frame, const std::vector<sTrim>& trimList) {
    if (trimList.size() == 0)
        return true;
    if (frame < 0)
        return false;
    for (auto trim : trimList) {
        if (trim.start <= frame && frame <= trim.fin) {
            return true;
        }
    }
    return false;
}

static bool inline rearrange_trim_list(int frame, int offset, std::vector<sTrim>& trimList) {
    if (trimList.size() == 0)
        return true;
    if (frame < 0)
        return false;
    for (uint32_t i = 0; i < trimList.size(); i++) {
        if (trimList[i].start >= frame) {
            trimList[i].start = clamp(trimList[i].start + offset, 0, TRIM_MAX);
        }
        if (trimList[i].fin && trimList[i].fin >= frame) {
            trimList[i].fin = (int)clamp((int64_t)trimList[i].fin + offset, 0, (int64_t)TRIM_MAX);
        }
    }
    return false;
}

enum RGYAVSync : uint32_t {
    RGY_AVSYNC_ASSUME_CFR = 0x00,
    RGY_AVSYNC_FORCE_CFR  = 0x01,
    RGY_AVSYNC_VFR        = 0x02,
};

static RGYAVSync operator|(RGYAVSync a, RGYAVSync b) {
    return (RGYAVSync)((uint32_t)a | (uint32_t)b);
}

static RGYAVSync operator|=(RGYAVSync& a, RGYAVSync b) {
    a = a | b;
    return a;
}

static RGYAVSync operator&(RGYAVSync a, RGYAVSync b) {
    return (RGYAVSync)((uint32_t)a & (uint32_t)b);
}

static RGYAVSync operator&=(RGYAVSync& a, RGYAVSync b) {
    a = a & b;
    return a;
}

static RGYAVSync operator~(RGYAVSync a) {
    return (RGYAVSync)(~(uint32_t)a);
}

static const int CHECK_PTS_MAX_INSERT_FRAMES = 8;

enum {
    RGY_MUX_NONE     = 0x00,
    RGY_MUX_VIDEO    = 0x01,
    RGY_MUX_AUDIO    = 0x02,
    RGY_MUX_SUBTITLE = 0x04,
};

static const uint32_t MAX_SPLIT_CHANNELS = 32;
static const uint64_t RGY_CHANNEL_AUTO = UINT64_MAX;
static const int RGY_OUTPUT_BUF_MB_MAX = 128;

template <uint32_t size>
static bool bSplitChannelsEnabled(uint64_t(&pnStreamChannels)[size]) {
    bool bEnabled = false;
    for (uint32_t i = 0; i < size; i++) {
        bEnabled |= pnStreamChannels[i] != 0;
    }
    return bEnabled;
}

template <uint32_t size>
static void setSplitChannelAuto(uint64_t(&pnStreamChannels)[size]) {
    for (uint32_t i = 0; i < size; i++) {
        pnStreamChannels[i] = ((uint64_t)1) << i;
    }
}

template <uint32_t size>
static bool isSplitChannelAuto(uint64_t(&pnStreamChannels)[size]) {
    bool isAuto = true;
    for (uint32_t i = 0; isAuto && i < size; i++) {
        isAuto &= (pnStreamChannels[i] == (((uint64_t)1) << i));
    }
    return isAuto;
}

typedef struct sAudioSelect {
    int    nAudioSelect;               //選択した音声トラックのリスト 1,2,...(1から連番で指定)
    TCHAR *pAVAudioEncodeCodec;        //音声エンコードのコーデック
    TCHAR *pAVAudioEncodeCodecPrm;     //音声エンコードのコーデックのパラメータ
    TCHAR *pAVAudioEncodeCodecProfile; //音声エンコードのコーデックのプロファイル
    int    nAVAudioEncodeBitrate;      //音声エンコードに選択した音声トラックのビットレート
    int    nAudioSamplingRate;         //サンプリング周波数
    TCHAR *pAudioExtractFilename;      //抽出する音声のファイル名のリスト
    TCHAR *pAudioExtractFormat;        //抽出する音声ファイルのフォーマット
    TCHAR *pAudioFilter;               //音声フィルタ
    uint64_t pnStreamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t pnStreamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル
} sAudioSelect;

const CX_DESC list_empty[] = {
    { NULL, 0 }
};

const CX_DESC list_log_level[] = {
    { _T("trace"), RGY_LOG_TRACE },
    { _T("debug"), RGY_LOG_DEBUG },
    { _T("more"),  RGY_LOG_MORE  },
    { _T("info"),  RGY_LOG_INFO  },
    { _T("warn"),  RGY_LOG_WARN  },
    { _T("error"), RGY_LOG_ERROR },
    { NULL, 0 }
};

const CX_DESC list_avsync[] = {
    { _T("cfr"),      RGY_AVSYNC_ASSUME_CFR   },
    { _T("vfr"),      RGY_AVSYNC_VFR       },
    { _T("forcecfr"), RGY_AVSYNC_FORCE_CFR },
    { NULL, 0 }
};

const CX_DESC list_resampler[] = {
    { _T("swr"),  RGY_RESAMPLER_SWR  },
    { _T("soxr"), RGY_RESAMPLER_SOXR },
    { NULL, 0 }
};

#if ENCODER_QSV == 0
const CX_DESC list_interlaced[] = {
    { _T("progressive"), RGY_PICSTRUCT_FRAME     },
    { _T("tff"),         RGY_PICSTRUCT_FRAME_TFF },
    { _T("bff"),         RGY_PICSTRUCT_FRAME_BFF },
    { NULL, 0 }
};
#endif

int rgy_avx_dummy_if_avail(int bAVXAvail);

struct rgy_time {
    int h, m, s, ms, us, ns;

    rgy_time() : h(0), m(0), s(0), ms(0), us(0), ns(0) {};
    rgy_time(double time_sec) : h(0), m(0), s(0), ms(0), us(0), ns(0) {
        s = (int)time_sec;
        time_sec -= s;
        ns = (int)(time_sec * 1e9 + 0.5);
        us = ns / 1000;
        ns -= us * 1000;
        ms = us / 1000;
        us -= ms * 1000;

        m = (int)(s / 60);
        s -= m * 60;
        h = m / 60;
        m -= h * 60;
    };
    rgy_time(uint32_t millisec) : h(0), m(0), s(0), ms(0), us(0), ns(0) {
        s = (int)(millisec / 1000);
        ms = (int)(millisec - s * 1000);
        m = s / 60;
        s -= m * 60;
        h = m / 60;
        m -= h * 60;
    };
    rgy_time(int64_t millisec) : h(0), m(0), s(0), ms(0), us(0), ns(0) {
        int64_t sec = millisec / 1000;
        ms = (int)(millisec - sec * 1000);

        int64_t min = sec / 60;
        s = (int)(sec - min * 60);

        h = (int)(min / 60);
        m = (int)(min - h * 60);
    }
    int64_t in_sec() {
        return (((int64_t)h * 60 + m) * 60 + s + ((ms + ((us >= 500) ? 1 : 0)) >= 500) ? 1 : 0);
    };
    int64_t in_ms() {
        return (((int64_t)h * 60 + m) * 60 + s) * 1000 + ms + ((us >= 500) ? 1 : 0);
    };
    tstring print() {
        auto str = strsprintf(_T("%d:%02d:%02d.%3d"), h, m, s, ms);
        if (us) {
            str += std::to_wstring(us);
        }
        if (ns) {
            str += std::to_wstring(ns);
        }
        return str;
    };
};

class rgy_stream {
    uint8_t *bufptr_;
    int64_t buf_size_;
    int64_t data_length_;
    int64_t offset_;

    uint32_t data_flag_;
    int duration_;
    int64_t pts_;
    int64_t dts_;
public:
    uint8_t *bufptr() const {
        return bufptr_;
    }
    uint8_t *data() const {
        return bufptr_ + offset_;
    }
    int64_t size() const {
        return data_length_;
    }
    int64_t buf_size() const {
        return buf_size_;
    }
    void add_offset(int64_t add) {
        if (data_length_ < add) {
            add = data_length_;
        }
        offset_ += add;
        data_length_ -= add;
    }

    void clear() {
        if (bufptr_) {
            _aligned_free(bufptr_);
        }
        bufptr_ = nullptr;
        buf_size_ = 0;
        data_length_ = 0;
        offset_ = 0;
    }
    RGY_ERR alloc(int64_t size) {
        clear();

        if (size > 0) {
            if (nullptr == (bufptr_ = (uint8_t *)_aligned_malloc(size, 32))) {
                return RGY_ERR_NULL_PTR;
            }
            buf_size_ = size;
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR realloc(int64_t size) {
        if (bufptr_ == nullptr) {
            return alloc(size);
        }
        if (size > 0) {
            auto newptr = (uint8_t *)_aligned_malloc(size, 32);
            if (newptr == nullptr) {
                return RGY_ERR_NULL_PTR;
            }
            auto newdatalen = std::min(size, data_length_);
            memcpy(newptr, bufptr_ + offset_, newdatalen);
            _aligned_free(bufptr_);
            bufptr_ = newptr;
            offset_ = 0;
            data_length_ = newdatalen;
            buf_size_ = size;
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR init() {
        bufptr_ = nullptr;
        buf_size_ = 0;
        data_length_ = 0;
        offset_ = 0;

        data_flag_ = 0;
        duration_ = 0;
        pts_ = 0;
        dts_ = 0;
    }

    void trim() {
        if (offset_ > 0 && data_length_ > 0) {
            memmove(bufptr_, bufptr_ + offset_, data_length_);
            offset_ = 0;
        }
    }

    RGY_ERR copy(const uint8_t *data, int64_t size) {
        if (data == nullptr || size == 0) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (buf_size_ < size) {
            clear();
            auto sts = alloc(size);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        data_length_ = size;
        offset_ = 0;
        memcpy(bufptr_, data, size);
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const uint8_t *data, int64_t size, int64_t pts) {
        pts_ = pts;
        return copy(data, size);
    }

    RGY_ERR copy(const uint8_t *data, int64_t size, int64_t pts, int64_t dts) {
        dts_ = dts;
        return copy(data, size, pts);
    }

    RGY_ERR copy(const uint8_t *data, int64_t size, int64_t pts, int64_t dts, int duration) {
        duration_ = duration;
        return copy(data, size, pts, dts);
    }

    RGY_ERR copy(const uint8_t *data, int64_t size, int64_t pts, int64_t dts, int duration, uint32_t flag) {
        data_flag_ = flag;
        return copy(data, size, pts, dts, duration);
    }

    RGY_ERR copy(const rgy_stream *pBitstream) {
        auto sts = copy(pBitstream->data(), pBitstream->size());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return copy(pBitstream->data(), pBitstream->size(), pBitstream->pts(), pBitstream->dts(), pBitstream->duration(), pBitstream->data_flag());
    }

    RGY_ERR append(const uint8_t *append_data, int64_t append_size) {
        if (append_data && append_size > 0) {
            const auto new_data_length = append_size + data_length_;
            if (buf_size_ < new_data_length) {
                auto sts = realloc(new_data_length * 2);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }

            if (buf_size_ < new_data_length + offset_) {
                trim();
            }
            memcpy(bufptr_ + data_length_ + offset_, append_data, append_size);
            data_length_ = new_data_length;
        }
        return RGY_ERR_NONE;
    }

    uint32_t data_flag() const {
        return data_flag_;
    }
    void set_data_flag(uint32_t flag) {
        data_flag_  = flag;
    }
    int duration() const {
        return duration_;
    }
    void set_duration(int duration) {
        duration_ = duration;
    }
    int64_t pts() const {
        return pts_;
    }
    void set_pts(int64_t pts) {
        pts_ = pts;
    }
    int64_t dts() const {
        return dts_;
    }
    void set_dts(int64_t dts) {
        dts_ = dts;
    }
};

#endif //__RGY_UTIL_H__
