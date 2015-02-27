//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_VIDEO_H_
#define _AUO_VIDEO_H_

#include "output.h"
#include "convert.h"
#include "auo_conf.h"
#include "auo_system.h"

static const int CF_YUY2 = 0;
static const int CF_YC48 = 1;
static const int CF_RGB  = 2;

static const int DROP_FRAME_FLAG = INT_MAX;

typedef struct {
	DWORD FOURCC;   //FOURCC
	DWORD size;  //1ピクセルあたりバイト数
} COLORFORMAT_DATA;

static const COLORFORMAT_DATA COLORFORMATS[] = {
	{ MAKEFOURCC('Y', 'U', 'Y', '2'), 2 }, //YUY2
	{ MAKEFOURCC('Y', 'C', '4', '8'), 6 }, //YC48
	{ NULL,                           3 }  //RGB
};

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe);
void close_afsvideo(PRM_ENC *pe);

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat);

#endif //_AUO_VIDEO_H_