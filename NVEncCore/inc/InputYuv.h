//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _INPUT_YUV_H_
#define _INPUT_YUV_H_

#include <stdio.h>
#include <tchar.h>
#include <string>
#include "NVEncStatus.h"


#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

typedef struct InputVideoInfo {
	int width;   //横解像度
	int height;  //縦解像度
	int scale;   //フレームレート (分子)
	int rate;    //フレームレート (分母)
	int crop[4]; //入力時切り落とし
	void *otherPrm; //その他入力情報
} InputVideoInfo;

typedef struct InputInfo {
	const TCHAR *filename;
	bool isY4m;
} BasicInputInfo;

class BasicInput {
public:
	BasicInput();
	~BasicInput();

	virtual int Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus);
	virtual int LoadNextFrame(EncodeInputSurfaceInfo *surface);
	virtual void Close();

	const TCHAR *getInputMes() {
		return m_inputMes.c_str();
	}

protected:
	virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *convSIMD, const TCHAR *outputCSpName, const InputVideoInfo *inputPrm);

	EncodeStatus *m_pStatus = NULL;
	FILE *m_fp = NULL;
	bool m_bIsY4m = false;
	std::basic_string<TCHAR> m_inputMes;
	uint8_t *m_inputBuffer = NULL;
	int m_crop[4];
};

#endif //_INPUT_YUV_H_
