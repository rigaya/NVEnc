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


typedef struct InputVideoInfo {
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

typedef struct InputInfo {
	bool isY4m;
} BasicInputInfo;

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
	virtual int ParseY4MHeader(char *buf, InputVideoInfo *inputPrm);

	virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *convSIMD, const TCHAR *outputCSpName, const InputVideoInfo *inputPrm);

	virtual void setSurfaceInfo(InputVideoInfo *inputPrm);

	EncodeInputSurfaceInfo m_stSurface;
	EncodeStatus *m_pStatus = NULL;
	FILE *m_fp = NULL;
	bool m_bIsY4m = false;
	tstring m_inputMes;
	uint8_t *m_inputBuffer = NULL;
};
