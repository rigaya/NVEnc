//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1

#include "convert_simd.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

//AVXが使用可能なSandyBridge以降のCPUでは、
//メモリがalignされていれば、_mm_store_si128 / _mm_storeu_si128 に速度差はないため
//関数も_mm_storeu_si128のもののみ用意する

void convert_yuy2_to_nv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yuy2_to_nv12_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yuy2_to_nv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yuy2_to_nv12_i_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yuy2_to_yv12_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yuy2_to_yv12_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yuy2_to_yv12_i_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yuy2_to_yv12_i_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yuy2_to_nv16_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yuy2_to_nv16_simd<FALSE>(frame, pixel_data, width, height);
}

void convert_yc48_to_nv12_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_nv12_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_nv12_i_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_nv12_i_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_yv12_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_yv12_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_yv12_i_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_yv12_i_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_nv16_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_nv16_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_yuv444_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_yuv444_simd<FALSE>(frame, pixel_data, width, height);
}
void convert_yc48_to_yuv444_16bit_avx(void *frame, CONVERT_CF_DATA *pixel_data, const int width, const int height) {
    return convert_yc48_to_yuv444_16bit_simd<FALSE>(frame, pixel_data, width, height);
}
