//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_VCE_H_
#define _AUO_VCE_H_

#include <Windows.h>
#include <stdio.h>
#include "output.h"
#include "auo.h"
#include "auo_conf.h"
#include "auo_system.h"

#include "CNVEncoderH264.h"

class CAuoNvEncoderH264 : public CNvEncoderH264
{
private:
	const OUTPUT_INFO *oip;
	CONF_GUIEX *conf;
	PRM_ENC *pe;
	int frames;
	int *jitter;
public:
	CAuoNvEncoderH264() {
		oip = NULL;
		conf = NULL;
		pe = NULL;
		frames = 0;
		jitter = NULL;
	};
	~CAuoNvEncoderH264() {
		oip = NULL;
		conf = NULL;
		pe = NULL;
		frames = 0;
		jitter = NULL;
	};
private:
	virtual HRESULT LoadCurrentFrame(unsigned char *yuvInput[3], HANDLE hInputYUVFile, unsigned int dwFrmIndex,
		unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
		bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc);
	virtual int CalculateFramesFromInput(HANDLE hInputFile, const char *filename, int width, int height) {
		return 0;
	}
	virtual void passAuoInfo(const OUTPUT_INFO *_oip, CONF_GUIEX *_conf, PRM_ENC *_pe, int _frames, int *_jitter);
	virtual void setAuoInfo(NV_ENC_CONFIG *nvenc, EncoderInputParams *nvenc2, const OUTPUT_INFO *_oip);
	virtual int NVEncPrintf(int log_level, const char *format, ...);
	virtual int NvPrintProgress(double progress, const char *format, ...);
public:
	virtual HRESULT NVEncAvailable() {
		return OpenEncodeSession();
	}
};

unsigned int checkNumberEncoders(EncoderGPUInfo *encoderInfo);
bool nvencAvailable();
void getDefaultParam(NV_ENC_CONFIG *nvenc, EncoderInputParams *nvenc2);

#endif //_AUO_VCE_H_