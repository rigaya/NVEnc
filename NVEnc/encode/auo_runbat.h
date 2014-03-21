//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_RUNBAT_H_
#define _AUO_RUNBAT_H_

#include <Windows.h>
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"

AUO_RESULT run_bat_file(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, DWORD run_bat_mode);

#endif //_AUO_RUNBAT_H_