//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _NVENC_CORE_H_
#define _NVENC_CORE_H_

#include <Windows.h>
#include <stdint.h>
#include <CNVEncoder.h>
#include <CNVEncoderH264.h>
#include <tchar.h>
#include <vector>
#include <string>
#include "InputYuv.h"

typedef std::basic_string<TCHAR> tstring;

enum {
	NVENC_THREAD_RUNNING = 0,

	NVENC_THREAD_FINISHED = -1,
	NVENC_THREAD_ABORT = -2,

	NVENC_THREAD_ERROR = 1,
};

static const char *get_name_from_guid(GUID guid, const guid_desc *desc, int desc_length) {
	for (int i = 0; i < desc_length; i++) {
		if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
			return desc[i].name;
		}
	}
	return _T("Unknown");
};

static int get_value_from_guid(GUID guid, const guid_desc *desc, int desc_length) {
	for (int i = 0; i < desc_length; i++) {
		if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
			return desc[i].value;
		}
	}
	return 0;
};

static GUID get_guid_from_value(int value, const guid_desc *desc, int desc_length) {
	for (int i = 0; i < desc_length; i++) {
		if (desc[i].value == (uint32_t)value) {
			return desc[i].id;
		}
	}
	return GUID{ 0 };
};

static tstring to_tchar(const char *string) {
#if UNICODE
	int required_length = MultiByteToWideChar(CP_ACP, 0, string, -1, NULL, 0);
	tstring tstr(1+required_length, _T('\0'));
	MultiByteToWideChar(CP_ACP, 0, string, -1, &tstr[0], (int)tstr.size());
#else
	tstring tstr = string;
#endif
	return tstr;
}

bool check_if_nvcuda_dll_available();

typedef struct NVEncCap {
	int id;
	const TCHAR *name;
	int value;
} NVEncCap;

class NVEncoderGPUInfo
{
public:
	NVEncoderGPUInfo();
	~NVEncoderGPUInfo();
	const std::vector<std::pair<uint32_t, tstring>> getGPUList() {
		return GPUList;
	}
private:
	std::vector<std::pair<uint32_t, tstring>> GPUList;
};

typedef struct InEncodeVideoParam {
	InputVideoInfo input;
	tstring outputFilename;
	int preset;
	int deviceID;
	int inputBuffer;
	int par[2];
	NV_ENC_PIC_STRUCT pic_struct;
	NV_ENC_CONFIG enc_config;
} InEncodeVideoParam;

static inline bool is_interlaced(NV_ENC_PIC_STRUCT pic_struct) {
	return pic_struct != NV_ENC_PIC_STRUCT_FRAME;
}

#define INPUT_THREAD_BUF \
	EncodeInputSurfaceInfo *pInput; \
	EncodeOutputBuffer *pOutputBitstream; \
	HANDLE heInputStart; \
	HANDLE heInputDone; \
	
typedef struct {
	INPUT_THREAD_BUF;
} InputThreadBufRaw;

typedef struct {
	INPUT_THREAD_BUF;
	uint8_t reserved[64-sizeof(InputThreadBufRaw)];
} InputThreadBuf;

class NVEncEncodeThread 
{
public:
	NVEncEncodeThread();
	~NVEncEncodeThread();

	int Init(uint32_t bufferSize);
	void Close();
	//終了を待機する
	void WaitToFinish(int sts);
	uint32_t GetBufferSize() {
		return m_nFrameBuffer;
	}
	HANDLE GetHandleEncThread() {
		return m_thEncode;
	}

	int GetNextFrame(EncodeInputSurfaceInfo **pInput, EncodeOutputBuffer **pOutputBitstream);
	int SetNextSurface(EncodeInputSurfaceInfo *pInput, EncodeOutputBuffer *pOutputBitstream);

	int RunEncFuncbyThread(unsigned (__stdcall * func) (void *), void *pClass);

	uint32_t m_bthForceAbort = FALSE;
	InputThreadBuf *m_InputBuf = NULL;
	uint32_t m_nFrameGet = 0;
	uint32_t m_nFrameSet = 0;
	uint32_t m_nFrameBuffer = 1;
	int m_stsThread = 0;
	tstring m_errMes;
protected:
	HANDLE m_thEncode = NULL;
};

class NVEncCore : public CNvEncoderH264
{
public:
	NVEncCore();
	~NVEncCore();
protected:
	virtual void PreInitializeEncoder() override;
	virtual int SetInputParam(const InEncodeVideoParam *param);
	virtual int NVEncCoreOpenEncodeSession(int deviceID, GUID profileGUID = NV_ENC_PRESET_DEFAULT_GUID, int preset_idx = -1);

	static unsigned int __stdcall RunEncThreadLauncher(void *pParam);
	virtual int RunEncodeThread();

	virtual int InitInput();
public:
	static NV_ENC_CONFIG initializedParam();
	static void setInitializedVUIParam(NV_ENC_CONFIG *conf);

	virtual bool checkNVEncAvialable();

	virtual int EncoderStartup(const InEncodeVideoParam *param);
	virtual int Run();

	int get_limit(NV_ENC_CAPS flag);

	virtual void Close();
protected:
	BasicInput *m_pInput;
	NVEncEncodeThread m_EncThread;
	InEncodeVideoParam m_InputParam;
	EncodeStatus *m_pStatus;
	std::vector<GUID> m_EncodeCodecGUIDs;
	std::vector<GUID> m_CodecProfileGUIDs;
	std::vector<NVEncCap> m_nvEncCaps;

	tstring GetEncodingParamsInfo(int output_level);
	std::vector<NVEncCap> GetCurrentDeviceNVEncCapability();

	virtual NVENCSTATUS setEncodeCodecList(void *encode);
	virtual NVENCSTATUS setCodecProfileList(void *encode, GUID encodecCodec);
	virtual NVENCSTATUS setInputFormatList(void *encode, GUID encodeCodec);

	virtual HRESULT CopyBitstreamData(EncoderThreadData stThreadData) override;
};

#endif //_NVENC_CORE_H_
