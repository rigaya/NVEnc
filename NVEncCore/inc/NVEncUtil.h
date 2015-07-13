//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <Windows.h>
#include <tchar.h>
#include <shlwapi.h>
#include <emmintrin.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>
#include <string>

#include "cpu_info.h"
#include "gpu_info.h"

typedef std::basic_string<TCHAR> tstring;

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

static tstring to_tchar(const char *string) {
#if UNICODE
    int required_length = MultiByteToWideChar(CP_ACP, 0, string, -1, NULL, 0);
    tstring tstr(1+required_length, _T('\0'));
    MultiByteToWideChar(CP_ACP, 0, string, -1, &tstr[0], (int)tstr.size());
#else
    tstring tstr = string;
#endif
    return tstr;
}

template<typename T>
static inline T nv_get_gcd(T a, T b) {
    T c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return b;
}
static inline int nv_get_gcd(std::pair<int, int> int2) {
    return nv_get_gcd(int2.first, int2.second);
}

void adjust_sar(int *sar_w, int *sar_h, int width, int height);
int get_h264_sar_idx(std::pair<int, int>sar);
std::pair<int, int> get_h264_sar(int idx);

double getCPUMaxTurboClock(DWORD num_thread = 1); //やや時間がかかるので注意 (～1/4秒)
int getCPUInfo(TCHAR *buffer, size_t nSize); //やや時間がかかるので注意 (～1/4秒)
double getCPUDefaultClock();
const TCHAR *getOSVersion();
UINT64 getPhysicalRamSize(UINT64 *ramUsed);
void getEnviromentInfo(TCHAR *buf, unsigned int buffer_size);

void adjust_sar(int *sar_w, int *sar_h, int width, int height);
void get_dar_pixels(unsigned int* width, unsigned int* height, int sar_w, int sar_h);
std::pair<int, int> get_sar(unsigned int width, unsigned int height, unsigned int darWidth, unsigned int darHeight);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}

BOOL nv_check_os_win8_or_later();

BOOL nv_is_64bit_os();

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

const int MAX_FILENAME_LEN = 1024;
