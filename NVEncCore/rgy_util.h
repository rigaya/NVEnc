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
#include <memory>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "rgy_osdep.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "ConvertCsp.h"

typedef std::basic_string<TCHAR> tstring;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;

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
    return std::vector<T0>(ptr, ptr + size);
}
template<typename T0, typename T1>
std::vector<T0> make_vector(T0 *ptr, T1 size) {
    static_assert(std::is_integral<T1>::value == true, "T1 should be integral");
    return std::vector<T0>(ptr, ptr + size);
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
void vector_cat(std::vector<T>& v1, const std::vector<T>& v2) {
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
static inline int rgy_gcd(std::pair<T, T> int2) {
    return rgy_gcd(int2.first, int2.second);
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

#if UNICODE
#define to_tstring to_wstring
#else
#define to_tstring to_string
#endif

typedef std::basic_string<TCHAR> tstring;
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
bool CreateDirectoryRecursive(const WCHAR *dir);
#endif

unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
std::wstring tchar_to_wstring(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
std::wstring tchar_to_wstring(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
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
    return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}

int rgy_print_stderr(int log_level, const TCHAR *mes, HANDLE handle = NULL);

#if defined(_WIN32) || defined(_WIN64)
tstring getOSVersion(OSVERSIONINFOEXW *osinfo = nullptr);
#else
tstring getOSVersion();
#endif
uint64_t getPhysicalRamSize(uint64_t *ramUsed);
tstring getEnviromentInfo(bool add_ram_info);

BOOL rgy_is_64bit_os();

//mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);

static void __forceinline sse_memcpy(BYTE *dst, const BYTE *src, int size) {
    if (size < 64) {
        memcpy(dst, src, size);
        return;
    }
    BYTE *dst_fin = dst + size;
    BYTE *dst_aligned_fin = (BYTE *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128 x0, x1, x2, x3;
    const int start_align_diff = (int)((size_t)dst & 15);
    if (start_align_diff) {
        x0 = _mm_loadu_ps((float*)src);
        _mm_storeu_ps((float*)dst, x0);
        dst += 16 - start_align_diff;
        src += 16 - start_align_diff;
    }
    for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_ps((float*)(src +  0));
        x1 = _mm_loadu_ps((float*)(src + 16));
        x2 = _mm_loadu_ps((float*)(src + 32));
        x3 = _mm_loadu_ps((float*)(src + 48));
        _mm_store_ps((float*)(dst +  0), x0);
        _mm_store_ps((float*)(dst + 16), x1);
        _mm_store_ps((float*)(dst + 32), x2);
        _mm_store_ps((float*)(dst + 48), x3);
    }
    BYTE *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_ps((float*)(src +  0));
    x1 = _mm_loadu_ps((float*)(src + 16));
    x2 = _mm_loadu_ps((float*)(src + 32));
    x3 = _mm_loadu_ps((float*)(src + 48));
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

typedef struct {
    TCHAR *desc;
    int value;
} CX_DESC;

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

struct VideoVUIInfo {
    int descriptpresent;
    int colorprim;
    int matrix;
    int transfer;
    int format;
    int fullrange;
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

struct nal_info {
    const uint8_t *ptr;
    uint8_t type;
    uint32_t size;
};

enum : uint8_t {
    NALU_H264_UNDEF    = 0,
    NALU_H264_NONIDR   = 1,
    NALU_H264_SLICEA   = 2,
    NALU_H264_SLICEB   = 3,
    NALU_H264_SLICEC   = 4,
    NALU_H264_IDR      = 5,
    NALU_H264_SEI      = 6,
    NALU_H264_SPS      = 7,
    NALU_H264_PPS      = 8,
    NALU_H264_AUD      = 9,
    NALU_H264_EOSEQ    = 10,
    NALU_H264_EOSTREAM = 11,
    NALU_H264_FILLER   = 12,
    NALU_H264_SPSEXT   = 13,
    NALU_H264_PREFIX   = 14,
    NALU_H264_SUBSPS   = 15,

    NALU_HEVC_UNDEF    = 0,
    NALU_HEVC_VPS      = 32,
    NALU_HEVC_SPS      = 33,
    NALU_HEVC_PPS      = 34,
    NALU_HEVC_AUD      = 35,
    NALU_HEVC_EOS      = 36,
    NALU_HEVC_EOB      = 37,
    NALU_HEVC_FILLER     = 38,
    NALU_HEVC_PREFIX_SEI = 39,
    NALU_HEVC_SUFFIX_SEI = 40,
};

static std::vector<nal_info> parse_nal_unit_h264(const uint8_t *data, uint32_t size) {
    std::vector<nal_info> nal_list;
    nal_info nal_start ={ nullptr, 0, 0 };
    const int i_fin = size - 3;
    for (int i = 0; i < i_fin; i++) {
        if (data[i+0] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i-1] == 0);
            nal_start.type = data[i+3] & 0x1f;
            nal_start.size = (int)(data + size - nal_start.ptr);
            if (nal_list.size()) {
                auto prev = nal_list.end()-1;
                prev->size = (int)(nal_start.ptr - prev->ptr);
            }
            i += 3;
        }
    }
    if (nal_start.ptr) {
        nal_list.push_back(nal_start);
    }
    return nal_list;
}

static std::vector<nal_info> parse_nal_unit_hevc(const uint8_t *data, uint32_t size) {
    std::vector<nal_info> nal_list;
    nal_info nal_start ={ nullptr, 0, 0 };
    const int i_fin = size - 3;

    for (int i = 0; i < i_fin; i++) {
        if (data[i+0] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i-1] == 0);
            nal_start.type = (data[i+3] & 0x7f) >> 1;
            nal_start.size = (int)(data + size - nal_start.ptr);
            if (nal_list.size()) {
                auto prev = nal_list.end()-1;
                prev->size = (int)(nal_start.ptr - prev->ptr);
            }
            i += 3;
        }
    }
    if (nal_start.ptr) {
        nal_list.push_back(nal_start);
    }
    return nal_list;
}

int rgy_avx_dummy_if_avail(int bAVXAvail);

#endif //__RGY_UTIL_H__
