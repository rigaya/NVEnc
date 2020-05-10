//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_CONVERT_H_
#define _AUO_CONVERT_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "convert.h"

func_audio_16to8 get_audio_16to8_func(BOOL split); //使用する音声16bit->8bit関数の選択
func_convert_frame get_convert_func(int width, int input_ccsp, int bit_depth, BOOL interlaced, int output_csp); //使用する関数の選択

BOOL malloc_pixel_data(CONVERT_CF_DATA * const pixel_data, int width, int height, int output_csp, int bit_depth); //映像バッファ用メモリ確保
void free_pixel_data(CONVERT_CF_DATA *pixel_data); //映像バッファ用メモリ開放

#endif //_AUO_CONVERT_H_
