//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_AUDIO_H_
#define _AUO_AUDIO_H_

#include <Windows.h>
#include "output.h"
#include "auo_conf.h"
#include "auo_system.h"

void *get_audio_data(const OUTPUT_INFO *oip, PRM_ENC *pe, int start, int length, int *readed);

void auo_faw_check(CONF_AUDIO *aud, const OUTPUT_INFO *oip, PRM_ENC *pe, const guiEx_settings *ex_stg);

AUO_RESULT audio_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat); //音声処理を実行
AUO_RESULT audio_output_parallel(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat);

#endif //_AUO_AUDIO_H_