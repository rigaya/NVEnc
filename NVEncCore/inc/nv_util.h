#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include <Windows.h>
#include <tchar.h>
#include <shlwapi.h>
#include <emmintrin.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>
#include <string>

#include "cpu_info.h"
#include "gpu_info.h"

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

static inline int nv_get_gcd(int a, int b) {
	int c;
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

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
	return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}

BOOL check_OS_Win8orLater();

BOOL is_64bit_os();

//mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);

static void __forceinline sse_memcpy(BYTE *dst, const BYTE *src, int size) {
	BYTE *dst_fin = dst + size;
	BYTE *dst_aligned_fin = (BYTE *)(((size_t)dst_fin & ~15) - 64);
	__m128 x0, x1, x2, x3;
	const int start_align_diff = (int)((size_t)dst & 15);
	if (start_align_diff) {
		x0 = _mm_loadu_ps((float*)src);
		_mm_storeu_ps((float*)dst, x0);
		dst += start_align_diff;
		src += start_align_diff;
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

#endif //_QSV_UTIL_H_
