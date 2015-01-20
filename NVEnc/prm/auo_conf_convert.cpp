//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "auo_util.h"
#include "auo_conf.h"

typedef struct CONF_NVENC_OLD {
	NV_ENC_CONFIG enc_config;
	NV_ENC_PIC_STRUCT pic_struct;
	int preset;
	int deviceID;
	int inputBuffer;
	int par[2];
} CONF_NVENC_OLD;

void guiEx_config::convert_nvencstg_to_nvencstgv2(CONF_GUIEX *conf, const void *dat) {
	const CONF_GUIEX *old_data = (const CONF_GUIEX *)dat;
	init_CONF_GUIEX(conf, FALSE);

	//まずそのままコピーするブロックはそうする
#define COPY_BLOCK(block, block_idx) { memcpy(&conf->block, ((BYTE *)old_data) + old_data->block_head_p[block_idx], min(sizeof(conf->block), old_data->block_size[block_idx])); }
	COPY_BLOCK(nvenc, 2);
	COPY_BLOCK(aud, 3);
	COPY_BLOCK(mux, 4);
	COPY_BLOCK(oth, 5);
#undef COPY_BLOCK

	conf->nvenc.codecConfig[NV_ENC_H264] = old_data->nvenc.enc_config.encodeCodecConfig;
}
