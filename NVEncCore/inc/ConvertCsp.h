//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _CONVERT_CSP_H_
#define _CONVERT_CSP_H_

typedef void (*funcConvertCSP) (void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yv12_to_nv12(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yuy2_to_nv12(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_sse2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_avx2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

void convert_yuy2_to_nv12_i(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_sse2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_ssse3(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
void convert_yuy2_to_nv12_i_avx2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

#endif //_CONVERT_CSP_H_
