//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <string>
#include "NVEncUtil.h"
#include "NVEncStatus.h"
#include "NVEncVersion.h"
#include "ConvertCsp.h"

enum {
	NV_ENC_INPUT_UNKNWON = 0,
	NV_ENC_INPUT_AUO = 0,
	NV_ENC_INPUT_RAW,
	NV_ENC_INPUT_Y4M,
#if AVI_READER
	NV_ENC_INPUT_AVI,
#endif
#if AVS_READER
	NV_ENC_INPUT_AVS
#endif
};

typedef struct InputVideoInfo {
	int type;    //種類 (NV_ENC_INPUT_xxx)
	int width;   //横解像度
	int height;  //縦解像度
	int scale;   //フレームレート (分子)
	int rate;    //フレームレート (分母)
	int crop[4]; //入力時切り落とし
	tstring filename; //入力ファイル名
	void *otherPrm; //その他入力情報
} InputVideoInfo;

typedef struct EncodeInputSurfaceInfo {
	int width;
	int src_pitch;
	int height;
	int crop[4];
} EncodeInputSurfaceInfo;

class NVEncBasicInput {
public:
	NVEncBasicInput();
	~NVEncBasicInput();

	virtual int Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus);
	virtual int LoadNextFrame(void *dst, int dst_pitch);
	virtual void Close();

	const TCHAR *getInputMes() {
		return m_inputMes.c_str();
	}

protected:
	virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *convSIMD, const TCHAR *outputCSpName, const InputVideoInfo *inputPrm);

	virtual void setSurfaceInfo(InputVideoInfo *inputPrm);

	EncodeInputSurfaceInfo m_stSurface;
	EncodeStatus *m_pStatus = NULL;
	FILE *m_fp = NULL;
	tstring m_inputMes;
	uint32_t m_tmLastUpdate = 0;
	const ConvertCSP *m_pConvCSPInfo = nullptr;
};
