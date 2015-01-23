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
#include "NVEncInput.h"

#if RAW_READER

class NVEncRawInput : public NVEncBasicInput {
public:
	NVEncRawInput();
	~NVEncRawInput();

	virtual int Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) override;
	virtual int LoadNextFrame(void *dst, int dst_pitch) override;
	virtual void Close() override;

protected:
	virtual int ParseY4MHeader(char *buf, InputVideoInfo *inputPrm);
};

#endif RAW_READER
