//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <cstdint>
#include <vector>
#include <tchar.h>
#include "ConvertCSP.h"

void convert_yuy2_to_nv12(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yuy2_to_nv12_i(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_ssse3(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_uv_yv12_to_nv12_sse2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_uv_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_uv_yv12_to_nv12_avx2(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

//適当。
#pragma warning (push)
#pragma warning (disable: 4100)
void convert_yuy2_to_nv12(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	int crop_left   = crop[0];
	int crop_up     = crop[1];
	int crop_right  = crop[2];
	int crop_bottom = crop[3];
	void *dst = dst_array[0];
	const void *src = src_array[0];
	uint8_t *srcFrame = (uint8_t *)src;
	uint8_t *dstYFrame = (uint8_t *)dst;
	uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 2) {
		uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
		uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
		uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up) + crop_left;
		const int x_fin = width - crop_right - crop_left;
		for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
			dstY[0*dst_y_pitch_byte  + 0] = srcP[0*src_y_pitch_byte + 0];
			dstY[0*dst_y_pitch_byte  + 1] = srcP[0*src_y_pitch_byte + 2];
			dstY[1*dst_y_pitch_byte  + 0] = srcP[1*src_y_pitch_byte + 0];
			dstY[1*dst_y_pitch_byte  + 1] = srcP[1*src_y_pitch_byte + 2];
			dstC[0*dst_y_pitch_byte/2+ 0] =(srcP[0*src_y_pitch_byte + 1] + srcP[1*src_y_pitch_byte + 1] + 1)/2;
			dstC[0*dst_y_pitch_byte/2+ 1] =(srcP[0*src_y_pitch_byte + 3] + srcP[1*src_y_pitch_byte + 3] + 1)/2;
		}
    }
}

//これも適当。
void convert_yuy2_to_nv12_i(void **dst_array, const void **src_array, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	int crop_left   = crop[0];
	int crop_up     = crop[1];
	int crop_right  = crop[2];
	int crop_bottom = crop[3];
	void *dst = dst_array[0];
	const void *src = src_array[0];
	uint8_t *srcFrame = (uint8_t *)src;
	uint8_t *dstYFrame = (uint8_t *)dst;
	uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 4) {
		uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
		uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
		uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up);
		const int x_fin = width - crop_right - crop_left;
		for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
			dstY[0*dst_y_pitch_byte   + 0] = srcP[0*src_y_pitch_byte + 0];
			dstY[0*dst_y_pitch_byte   + 1] = srcP[0*src_y_pitch_byte + 2];
			dstY[1*dst_y_pitch_byte   + 0] = srcP[1*src_y_pitch_byte + 0];
			dstY[1*dst_y_pitch_byte   + 1] = srcP[1*src_y_pitch_byte + 2];
			dstY[2*dst_y_pitch_byte   + 0] = srcP[2*src_y_pitch_byte + 0];
			dstY[2*dst_y_pitch_byte   + 1] = srcP[2*src_y_pitch_byte + 2];
			dstY[3*dst_y_pitch_byte   + 0] = srcP[3*src_y_pitch_byte + 0];
			dstY[3*dst_y_pitch_byte   + 1] = srcP[3*src_y_pitch_byte + 2];
			dstC[0*dst_y_pitch_byte/2 + 0] =(srcP[0*src_y_pitch_byte + 1] * 3 + srcP[2*src_y_pitch_byte + 1] * 1 + 2)>>2;
			dstC[0*dst_y_pitch_byte/2 + 1] =(srcP[0*src_y_pitch_byte + 3] * 3 + srcP[2*src_y_pitch_byte + 3] * 1 + 2)>>2;
			dstC[1*dst_y_pitch_byte/2 + 0] =(srcP[1*src_y_pitch_byte + 1] * 1 + srcP[3*src_y_pitch_byte + 1] * 3 + 2)>>2;
			dstC[1*dst_y_pitch_byte/2 + 1] =(srcP[1*src_y_pitch_byte + 3] * 1 + srcP[3*src_y_pitch_byte + 3] * 3 + 2)>>2;
		}
	}
}
#pragma warning (pop)


enum {
	NONE  = 0x0000,
	SSE2  = 0x0001,
	SSE3  = 0x0002, //使用していない
	SSSE3 = 0x0004,
	SSE41 = 0x0008,
	SSE42 = 0x0010, //使用していない
	AVX   = 0x0020,
	AVX2  = 0x0040,
};

static const ConvertCSP funcList[] = {
	{ NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12, false, { convert_yuy2_to_nv12_avx2,     convert_yuy2_to_nv12_i_avx2   }, AVX2|AVX },
	{ NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12, false, { convert_yuy2_to_nv12_avx,      convert_yuy2_to_nv12_i_avx    }, AVX },
	{ NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12, false, { convert_yuy2_to_nv12_sse2,     convert_yuy2_to_nv12_i_ssse3  }, SSSE3|SSE2 },
	{ NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12, false, { convert_yuy2_to_nv12_sse2,     convert_yuy2_to_nv12_i_sse2   }, SSE2 },
	{ NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12, false, { convert_yuy2_to_nv12,          convert_yuy2_to_nv12          }, NONE },
#if !NVENC_AUO
	{ NV_ENC_CSP_YV12, NV_ENC_CSP_NV12, false, { convert_yv12_to_nv12_avx2,     convert_yv12_to_nv12_avx2     }, AVX2|AVX },
	{ NV_ENC_CSP_YV12, NV_ENC_CSP_NV12, false, { convert_yv12_to_nv12_avx,      convert_yv12_to_nv12_avx      }, AVX },
	{ NV_ENC_CSP_YV12, NV_ENC_CSP_NV12, false, { convert_yv12_to_nv12_sse2,     convert_yv12_to_nv12_sse2     }, SSE2 },
#endif
	{ NV_ENC_CSP_NA, NV_ENC_CSP_NA, 0, false, 0x0, 0 },
};

static uint32_t nvenc_get_availableSIMD() {
	int CPUInfo[4];
	__cpuid(CPUInfo, 1);
	uint32_t simd = NONE;
	if  (CPUInfo[3] & 0x04000000)
		simd |= SSE2;
	if  (CPUInfo[2] & 0x00000001)
		simd |= SSE3;
	if  (CPUInfo[2] & 0x00000200)
		simd |= SSSE3;
	if  (CPUInfo[2] & 0x00080000)
		simd |= SSE41;
	if  (CPUInfo[2] & 0x00100000)
		simd |= SSE42;
	uint64_t XGETBV = 0;
	if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
		XGETBV = _xgetbv(0);
		if ((XGETBV & 0x06) == 0x06)
			simd |= AVX;
	}
	__cpuid(CPUInfo, 7);
	if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
		simd |= AVX2;
	return simd;
}

const ConvertCSP* get_convert_csp_func(NV_ENC_CSP csp_from, NV_ENC_CSP csp_to, bool uv_only) {
	uint32_t availableSIMD = nvenc_get_availableSIMD();
	const ConvertCSP *convert = nullptr;
	for (int i = 0; funcList[i].func; i++) {
		if (csp_from != funcList[i].csp_from)
			continue;
		
		if (csp_to != funcList[i].csp_to)
			continue;
		
		if (uv_only != funcList[i].uv_only)
			continue;
		
		if (funcList[i].simd != (availableSIMD & funcList[i].simd))
			continue;

		convert = &funcList[i];
		break;
	}
	return convert;
}

const TCHAR *get_simd_str(unsigned int simd) {
	static std::vector<std::pair<uint32_t, TCHAR*>> simd_str_list = {
		{ AVX2,  _T("AVX2")   },
		{ AVX,   _T("AVX")    },
		{ SSE42, _T("SSE4.2") },
		{ SSE41, _T("SSE4.1") },
		{ SSSE3, _T("SSSE3")  },
		{ SSE2,  _T("SSE2")   },
	};
	for (auto simd_str : simd_str_list) {
		if (simd_str.first & simd)
			return simd_str.second;
	}
	return _T("-");
}
