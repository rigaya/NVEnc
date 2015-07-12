//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1

#include "ConvertCSPSIMD.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

#pragma warning (push)
#pragma warning (disable: 4100)
void convert_yuy2_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	return convert_yuy2_to_nv12_simd(dst[0], src[0], width, src_y_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yuy2_to_nv12_i_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	return convert_yuy2_to_nv12_i_simd(dst[0], src[0], width, src_y_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	convert_yv12_to_nv12_simd<false>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}

void convert_uv_yv12_to_nv12_avx(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	convert_yv12_to_nv12_simd<true>(dst, src, width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte, height, dst_height, crop);
}
#pragma warning (pop)

