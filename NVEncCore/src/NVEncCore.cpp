//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <tchar.h>
#include <cuda.h>
#include <CNVEncoder.h>
#include "process.h"
#pragma comment(lib, "winmm.lib")
#include "helper_nvenc.h"
#include "NVEncCore.h"
#include "NVEncVersion.h"
#include "NVEncParam.h"
#include "nv_util.h"

bool check_if_nvcuda_dll_available() {
	//check for nvcuda.dll
	HMODULE hModule = LoadLibrary("nvcuda.dll");
	if (hModule == NULL)
		return false;
	FreeLibrary(hModule);
	return true;
}

NVEncoderGPUInfo::NVEncoderGPUInfo() {
	CUresult cuResult = CUDA_SUCCESS;

	if (!check_if_nvcuda_dll_available())
		return;

	if (CUDA_SUCCESS != (cuResult = cuInit(0)))
		return;

	int deviceCount = 0;
	if (CUDA_SUCCESS != (cuDeviceGetCount(&deviceCount)) || 0 == deviceCount)
		return;

	for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
		char gpu_name[1024] = { 0 };
		int SMminor = 0, SMmajor = 0;
		CUdevice cuDevice = 0;
		if (   CUDA_SUCCESS == cuDeviceGet(&cuDevice, currentDevice)
			&& CUDA_SUCCESS == cuDeviceGetName(gpu_name, _countof(gpu_name), cuDevice)
			&& CUDA_SUCCESS == cuDeviceComputeCapability(&SMmajor, &SMminor, currentDevice)
			&& (((SMmajor << 4) + SMminor) >= 0x30)) {
			GPUList.push_back(std::make_pair(currentDevice, to_tchar(gpu_name)));
		}
	}
};

NVEncoderGPUInfo::~NVEncoderGPUInfo() {

};

NVEncEncodeThread::NVEncEncodeThread() {

}
NVEncEncodeThread::~NVEncEncodeThread() {
	Close();
}

int NVEncEncodeThread::GetNextFrame(EncodeInputSurfaceInfo **pInput, EncodeOutputBuffer **pOutputBitstream) {
	const int inputBufIdx = m_nFrameGet % m_nFrameBuffer;
	InputThreadBuf *pInputBuf = &m_InputBuf[inputBufIdx];

	WaitForSingleObject(pInputBuf->heInputDone, INFINITE);
	//エラー・中断要求などでの終了
	if (m_bthForceAbort || NVENC_THREAD_ABORT == m_stsThread)
		return NVENC_THREAD_ABORT;
	*pInput = pInputBuf->pInput;
	*pOutputBitstream = pInputBuf->pOutputBitstream;
	m_nFrameGet++;
	return m_stsThread;
}

int NVEncEncodeThread::SetNextSurface(EncodeInputSurfaceInfo *pInput, EncodeOutputBuffer *pOutputBitstream) {
	const int inputBufIdx = m_nFrameSet % m_nFrameBuffer;
	InputThreadBuf *pInputBuf = &m_InputBuf[inputBufIdx];
	pInputBuf->pInput = pInput;
	pInputBuf->pOutputBitstream = pOutputBitstream;
	SetEvent(pInputBuf->heInputStart);
	m_nFrameSet++;
	return 0;
}

int NVEncEncodeThread::Init(uint32_t bufferSize) {
	Close();

	m_nFrameBuffer = bufferSize;

	if (NULL == (m_InputBuf = (InputThreadBuf *)_aligned_malloc(sizeof(m_InputBuf[0]) * m_nFrameBuffer, 64))) {
		return 1;
	}
	ZeroMemory(m_InputBuf, sizeof(m_InputBuf[0]) * m_nFrameBuffer);
	for (uint32_t i = 0; i < m_nFrameBuffer; i++) {
		if (   NULL == (m_InputBuf[i].heInputDone  = CreateEvent(NULL, FALSE, FALSE, NULL))
			|| NULL == (m_InputBuf[i].heInputStart = CreateEvent(NULL, FALSE, FALSE, NULL))) {
			return 1;
		}
	}
	return 0;
}

int NVEncEncodeThread::RunEncFuncbyThread(unsigned(__stdcall * func) (void *), void *pClass) {
	if (NULL == (m_thEncode = (HANDLE)_beginthreadex(NULL, 0, func, pClass, 0, 0))) {
		return 1;
	}
	return 0;
}

void NVEncEncodeThread::Close() {
	if (m_thEncode) {
		WaitToFinish(NVENC_THREAD_FINISHED);
	}
	if (m_InputBuf) {
		for (uint32_t i = 0; i < m_nFrameBuffer; i++) {
			CloseHandle(m_InputBuf[i].heInputDone);
			CloseHandle(m_InputBuf[i].heInputStart);
		}
		_aligned_free(m_InputBuf);
		m_InputBuf = NULL;
	}
	m_errMes.clear();
	m_bthForceAbort = FALSE;
	m_nFrameGet = 0;
	m_nFrameSet = 0;
	m_nFrameBuffer = 1;
	m_stsThread = 0;
}

void NVEncEncodeThread::WaitToFinish(int sts) {
	//読み込み終了(MFX_ERR_MORE_DATA)ではなく、エラーや中断だった場合、
	//直ちに終了する
	if (NVENC_THREAD_FINISHED != sts) {
		InterlockedIncrement((DWORD*)&m_bthForceAbort);
		for (uint32_t i = 0; i < m_nFrameBuffer; i++) {
			SetEvent(m_InputBuf[i].heInputDone);
		}
	}
	//RunEncodeの終了を待つ
	WaitForSingleObject(m_thEncode, INFINITE);
	CloseHandle(m_thEncode);
	m_thEncode = NULL;
}

NVEncCore::NVEncCore() {
	m_pStatus = NULL;
	m_pInput = NULL;
};

NVEncCore::~NVEncCore() {
	Close();
};

void NVEncCore::Close() {
	if (NULL != m_pStatus) {
		delete m_pStatus;
		m_pStatus = NULL;
	}
	if (NULL != m_pInput) {
		delete m_pInput;
		m_pInput = NULL;
	}
	if (m_fOutput) {
		fclose(m_fOutput);
		m_fOutput = NULL;
	}
}

NVENCSTATUS NVEncCore::setEncodeCodecList(void *encode) {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(encode, &m_dwEncodeGUIDCount))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetEncodeGUIDCount() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}
	uint32_t uArraysize = 0;
	GUID guid_init = { 0 };
	m_EncodeCodecGUIDs.resize(m_dwEncodeGUIDCount, guid_init);
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(encode, &m_EncodeCodecGUIDs[0], m_dwEncodeGUIDCount, &uArraysize))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetEncodeGUIDs() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}
	return nvStatus;
}

NVENCSTATUS NVEncCore::setCodecProfileList(void *encode, GUID encodeCodec) {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(encode, encodeCodec, &m_dwCodecProfileGUIDCount))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetEncodeProfileGUIDCount() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}
	uint32_t uArraysize = 0;
	GUID guid_init = { 0 };
	m_CodecProfileGUIDs.resize(m_dwCodecProfileGUIDCount, guid_init);
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(encode, encodeCodec, &m_CodecProfileGUIDs[0], m_dwCodecProfileGUIDCount, &uArraysize))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetEncodeProfileGUIDs() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}
	return nvStatus;
}

NVENCSTATUS NVEncCore::setInputFormatList(void *encode, GUID encodeCodec) {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormatCount(encode, encodeCodec, &m_dwInputFmtCount))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetInputFormatCount() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}
	uint32_t uArraysize = 0;
	m_pAvailableSurfaceFmts = new NV_ENC_BUFFER_FORMAT[m_dwInputFmtCount];
	memset(m_pAvailableSurfaceFmts, 0, sizeof(m_pAvailableSurfaceFmts[0]) * m_dwInputFmtCount);
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetInputFormats(encode, encodeCodec, m_pAvailableSurfaceFmts, m_dwInputFmtCount, &uArraysize))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncGetInputFormats() がエラーを返しました。: %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		return nvStatus;
	}

	return nvStatus;
}

int NVEncCore::NVEncCoreOpenEncodeSession(int deviceID, GUID profileGUID, int preset_idx) {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

	uint32_t target_codec = NV_ENC_H264;

	NV_ENC_CAPS_PARAM stCapsParam = { 0 };
	SET_VER(stCapsParam, NV_ENC_CAPS_PARAM);

	GUID clientKey = { 0 };
	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS stEncodeSessionParams = { 0 };
	SET_VER(stEncodeSessionParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);
	stEncodeSessionParams.apiVersion = NVENCAPI_VERSION;
	stEncodeSessionParams.clientKeyPtr = &clientKey;

	InitCuda(deviceID);
	stEncodeSessionParams.device = reinterpret_cast<void *>(m_cuContext);
	stEncodeSessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;

	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&stEncodeSessionParams, &m_hEncoder))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("nvEncOpenEncodeSessionEx() がエラーを返しました。 %d (%s)\n"), nvStatus, to_tchar(_nvencGetErrorEnum(nvStatus)).c_str());
		TCHAR buffer[1024] = { 0 };
		getEnviromentInfo(buffer, _countof(buffer));
		nvPrintf(stderr, NV_LOG_ERROR, _T("%s\n"), buffer);
		nvPrintf(stderr, NV_LOG_ERROR, _T("NVEncの使用にはドライバ334.89以降が必要です。\n"));
		nvPrintf(stderr, NV_LOG_ERROR, _T("NVIDIAのドライバを最新のものに更新して再度試してください。\n"));
		return nvStatus;
	}
	m_uRefCount++;

	//コーデックのサポートの確認
	if (NV_ENC_SUCCESS != (nvStatus = setEncodeCodecList(m_hEncoder)))
		return nvStatus;

	bool codecAvailable = false;
	for (auto codecGUID : m_EncodeCodecGUIDs) {
		if (GetCodecType(codecGUID) == target_codec) {
			m_stEncodeGUID = codecGUID;
			codecAvailable = true;
			break;
		}
	}
	if (!codecAvailable) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("コーデックがサポートされていません。\n"));
		return 1;
	}

	if (0 <= preset_idx)
		GetPresetConfig(preset_idx);


	//プロファイルのサポートの確認
	if (NV_ENC_SUCCESS != (nvStatus = setCodecProfileList(m_hEncoder, m_stEncodeGUID))) {
		return nvStatus;
	}
	bool profileAvailable = false;
	for (auto codecProfileGUID : m_CodecProfileGUIDs) {
		if (0 == memcmp(&codecProfileGUID, &profileGUID, sizeof(GUID))) {
			profileAvailable = true;
			break;
		}
	}
	if (!profileAvailable) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("選択されたH.264のプロファイルがサポートされていません。\n"));
		nvPrintf(stderr, NV_LOG_ERROR, _T("もう一度、設定を見なおしてみてください。\n"));
		return 1;
	}


	//InputFormatのサポートの確認
	if (NV_ENC_SUCCESS != (nvStatus = setInputFormatList(m_hEncoder, m_stEncodeGUID))) {
		return nvStatus;
	}
	bool bFmtFound = false;
	for (uint32_t idx = 0; idx < m_dwInputFmtCount; idx++) {
		if (NV_ENC_BUFFER_FORMAT_NV12_TILED64x16 == m_pAvailableSurfaceFmts[idx]) {
			m_dwInputFormat = m_pAvailableSurfaceFmts[idx];
			bFmtFound = true;
		}
	}
	if (!bFmtFound) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("入力フォーマットが決定できません。\n"));
		return 1;
	}

	return nvStatus;
}

std::vector<NVEncCap> NVEncCore::GetCurrentDeviceNVEncCapability() {
	std::vector<NVEncCap> caps;

	auto add_cap_info = [&](NV_ENC_CAPS cap_id, const TCHAR *cap_name) {
		NV_ENC_CAPS_PARAM param = { 0 };
		SET_VER(param, NV_ENC_CAPS_PARAM);
		param.capsToQuery = cap_id;
		int value = 0;
		if (NV_ENC_SUCCESS == m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, m_stEncodeGUID, &param, &value)) {
			NVEncCap cap = { 0 };
			cap.id = cap_id;
			cap.name = cap_name;
			cap.value = value;
			caps.push_back(cap);
		}
	};

	add_cap_info(NV_ENC_CAPS_NUM_MAX_BFRAMES,              "Max Bframes");
	add_cap_info(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,  "RC Modes");
	add_cap_info(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,       "Field Encoding");
	add_cap_info(NV_ENC_CAPS_SUPPORT_MONOCHROME,           "MonoChrome");
	add_cap_info(NV_ENC_CAPS_SUPPORT_FMO,                  "FMO");
	add_cap_info(NV_ENC_CAPS_SUPPORT_QPELMV,               "Quater-Pel MV");
	add_cap_info(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,         "B Direct Mode");
	add_cap_info(NV_ENC_CAPS_SUPPORT_CABAC,                "CABAC");
	add_cap_info(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,   "Adaptive Transform");
	add_cap_info(NV_ENC_CAPS_SUPPORT_STEREO_MVC,           "StereoMVC");
	add_cap_info(NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,      "Max Temporal Layers");
	add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES, "Hierarchial P Frames");
	add_cap_info(NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES, "Hierarchial B Frames");
	add_cap_info(NV_ENC_CAPS_LEVEL_MAX,                    "Max H.264 Level");
	add_cap_info(NV_ENC_CAPS_LEVEL_MIN,                    "Min H.264 Level");
	add_cap_info(NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,        "4:4:4");
	add_cap_info(NV_ENC_CAPS_WIDTH_MAX,                    "Max Width");
	add_cap_info(NV_ENC_CAPS_HEIGHT_MAX,                   "Max Height");
	add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,       "Dynamic Resolution Change");
	add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,   "Dynamic Bitrate Change");
	add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,    "Forced constant QP");
	add_cap_info(NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,    "Dynamic RC Mode Change");
	add_cap_info(NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,    "Subframe Readback");
	add_cap_info(NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING, "Constrained Encoding");
	add_cap_info(NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,        "Intra Refresh");
	add_cap_info(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,  "Custom VBV Bufsize");
	add_cap_info(NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,   "Dynamic Slice Mode");
	add_cap_info(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION, "Ref Pic Invalidiation");
	add_cap_info(NV_ENC_CAPS_PREPROC_SUPPORT,              "PreProcess");
	add_cap_info(NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,         "Async Encoding");
	add_cap_info(NV_ENC_CAPS_MB_NUM_MAX,                   "Max MBs");

	return caps;
}

int NVEncCore::get_limit(NV_ENC_CAPS flag) {
	return get_value(flag, m_nvEncCaps);
}

void NVEncCore::PreInitializeEncoder() {

}

int NVEncCore::SetInputParam(const InEncodeVideoParam *param) {
	m_bAsyncModeEncoding = true;
	
	m_uMaxWidth          = param->input.width  - param->input.crop[0] - param->input.crop[2];
	m_uMaxHeight         = param->input.height - param->input.crop[1] - param->input.crop[3];
	m_dwFrameWidth       = m_uMaxWidth;
	m_dwFrameHeight      = m_uMaxHeight;

	m_stEncodeConfig = param->enc_config;

	//制限事項チェック
	if ((m_uMaxWidth & 1) || (m_uMaxHeight & (1 + 2 * !!is_interlaced(param->pic_struct)))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("解像度が無効です。: %dx%d\n"), m_uMaxWidth, m_uMaxHeight);
		nvPrintf(stderr, NV_LOG_ERROR, _T("縦横の解像度は2の倍数である必要があります。\n"));
		if (is_interlaced(param->pic_struct)) {
			nvPrintf(stderr, NV_LOG_ERROR, _T("さらに、インタレ保持エンコードでは縦解像度は4の倍数である必要があります。\n"));
		}
		return 1;
	}
	//環境による制限
	if (m_uMaxWidth > (uint32_t)get_limit(NV_ENC_CAPS_WIDTH_MAX) || m_uMaxHeight > (uint32_t)get_limit(NV_ENC_CAPS_HEIGHT_MAX)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("解像度が上限を超えています。: %dx%d [上限: %dx%d]\n"), m_uMaxWidth, m_uMaxHeight, get_limit(NV_ENC_CAPS_WIDTH_MAX), get_limit(NV_ENC_CAPS_HEIGHT_MAX));
		return 1;
	}

	if (is_interlaced(param->pic_struct) && !get_limit(NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("インターレース保持エンコードはサポートされていません。\n"));
		return 1;
	}
	if (m_stEncodeConfig.rcParams.rateControlMode != (m_stEncodeConfig.rcParams.rateControlMode & get_limit(NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES))) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("選択されたレート制御モードはサポートされていません。\n"));
		return 1;
	}
	if (m_stEncodeConfig.frameIntervalP - 1 > get_limit(NV_ENC_CAPS_NUM_MAX_BFRAMES)) {
		m_stEncodeConfig.frameIntervalP = get_limit(NV_ENC_CAPS_NUM_MAX_BFRAMES) + 1;
		nvPrintf(stderr, NV_LOG_WARNING, _T("Bフレームの最大数は%dです。\n"), get_limit(NV_ENC_CAPS_NUM_MAX_BFRAMES));
	}
	if (NV_ENC_H264_ENTROPY_CODING_MODE_CABAC == m_stEncodeConfig.encodeCodecConfig.h264Config.entropyCodingMode && !get_limit(NV_ENC_CAPS_SUPPORT_CABAC)) {
		m_stEncodeConfig.encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
		nvPrintf(stderr, NV_LOG_WARNING, _T("CABACはサポートされていません。\n"));
	}
	if (NV_ENC_H264_FMO_ENABLE == m_stEncodeConfig.encodeCodecConfig.h264Config.fmoMode && !get_limit(NV_ENC_CAPS_SUPPORT_FMO)) {
		m_stEncodeConfig.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;
		nvPrintf(stderr, NV_LOG_WARNING, _T("FMOはサポートされていません。\n"));
	}
	if (NV_ENC_H264_BDIRECT_MODE_TEMPORAL & m_stEncodeConfig.encodeCodecConfig.h264Config.bdirectMode && !get_limit(NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)) {
		m_stEncodeConfig.encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
		nvPrintf(stderr, NV_LOG_WARNING, _T("B Direct モードはサポートされていません。\n"));
	}
	if ((NV_ENC_MV_PRECISION_QUARTER_PEL == m_stEncodeConfig.mvPrecision) && !get_limit(NV_ENC_CAPS_SUPPORT_QPELMV)) {
		m_stEncodeConfig.mvPrecision = NV_ENC_MV_PRECISION_HALF_PEL;
		nvPrintf(stderr, NV_LOG_WARNING, _T("1/4画素精度MV探索はサポートされていません。\n"));
	}
	if (NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE != m_stEncodeConfig.encodeCodecConfig.h264Config.adaptiveTransformMode && !get_limit(NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)) {
		m_stEncodeConfig.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
		nvPrintf(stderr, NV_LOG_WARNING, _T("Adaptive Tranform はサポートされていません。\n"));
	}
	if (0 != m_stEncodeConfig.rcParams.vbvBufferSize && !get_limit(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
		m_stEncodeConfig.rcParams.vbvBufferSize = 0;
		nvPrintf(stderr, NV_LOG_WARNING, _T("VBVバッファサイズの指定はサポートされていません。\n"));
	}
	//自動決定パラメータ
	if (0 == m_stEncodeConfig.gopLength) {
		m_stEncodeConfig.gopLength = (int)(param->input.rate / (double)param->input.scale + 0.5) * 10;
	}
	//SAR自動設定
	std::pair<int, int> par = std::make_pair(param->par[0], param->par[1]);
	adjust_sar(&par.first, &par.second, m_uMaxWidth, m_uMaxHeight);
	int sar_idx = get_h264_sar_idx(par);
	if (sar_idx < 0) {
		nvPrintf(stderr, NV_LOG_WARNING, _T("適切なSAR値を決定できませんでした。\n"));
		sar_idx = 0;
	}
	if (sar_idx) {
		;//と思ったが、aspect_ratioは設定できないのかな?
	}
	//色空間設定自動
	int frame_height = m_uMaxHeight;
	auto apply_auto_colormatrix = [frame_height](uint32_t& value, const CX_DESC *list) {
		if (COLOR_VALUE_AUTO == value)
			value = (frame_height >= HD_HEIGHT_THRESHOLD) ? list[HD_INDEX].value : list[SD_INDEX].value;
	};
	apply_auto_colormatrix(m_stEncodeConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries, list_colorprim);
	apply_auto_colormatrix(m_stEncodeConfig.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics, list_transfer);
	apply_auto_colormatrix(m_stEncodeConfig.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix, list_colormatrix);

	memset(&m_stInitEncParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
	SET_VER(m_stInitEncParams, NV_ENC_INITIALIZE_PARAMS);
	m_stInitEncParams.encodeConfig        = &m_stEncodeConfig;
	m_stInitEncParams.darHeight           = m_dwFrameHeight;
	m_stInitEncParams.darWidth            = m_dwFrameWidth;
	m_stInitEncParams.encodeHeight        = m_dwFrameHeight;
	m_stInitEncParams.encodeWidth         = m_dwFrameWidth;

	m_stInitEncParams.maxEncodeHeight     = m_uMaxHeight;
	m_stInitEncParams.maxEncodeWidth      = m_uMaxWidth;

	m_stInitEncParams.frameRateNum        = m_InputParam.input.rate;
	m_stInitEncParams.frameRateDen        = m_InputParam.input.scale;
	//Fix me add theading model
	m_stInitEncParams.enableEncodeAsync   = m_bAsyncModeEncoding;
	m_stInitEncParams.enablePTD           = true;
	m_stInitEncParams.encodeGUID          = m_stEncodeGUID;
	m_stInitEncParams.presetGUID          = m_stPresetGUID;

	//整合性チェック (一般)
	m_stInitEncParams.encodeConfig->frameFieldMode = (param->pic_struct == NV_ENC_PIC_STRUCT_FRAME) ? NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME : NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
	m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = (m_stEncoderInput[0].profile > 66) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
	m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod   = m_stInitEncParams.encodeConfig->gopLength;
	if (m_stInitEncParams.encodeConfig->frameIntervalP - 1 <= 0) {
		nvPrintf(stderr, NV_LOG_WARNING, _T("Bフレーム無しの場合、B Direct モードはサポートされていません。\n"));
		m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_DISABLE;
	}
	
	//整合性チェック (H.264 VUI)
	m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfoPresentFlag =
		(m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo) ? 1 : 0;

	m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag = 
		(get_cx_value(list_videoformat, "undef") != (int)m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat
		|| m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag) ? 1 : 0;

	m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag = 
		(  get_cx_value(list_colorprim,   "undef") != (int)m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries
		|| get_cx_value(list_transfer,    "undef") != (int)m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics
		|| get_cx_value(list_colormatrix, "undef") != (int)m_stInitEncParams.encodeConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix) ? 1 : 0;

	return 0;
}

int NVEncCore::InitInput() {
	m_pStatus = new EncodeStatus();
	m_pInput = new BasicInput();
	int ret = m_pInput->Init(&m_InputParam.input, m_pStatus);
	if (0 == m_InputParam.input.rate || 0 == m_InputParam.input.scale) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("フレーム読み込みの開始に失敗しました。\n"));
		nvPrintf(stderr, NV_LOG_ERROR, _T("%s\n"), m_pInput->getInputMes());
		return 1;
	}
	if (0 == m_InputParam.input.rate || 0 == m_InputParam.input.scale) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("フレームレートが正しく取得できていません。: %d/%d fps\n"), m_InputParam.input.rate, m_InputParam.input.scale);
		return 1;
	}
	//自動GCD
	int gcd_fps = nv_get_gcd(m_InputParam.input.rate, m_InputParam.input.scale);
	m_InputParam.input.rate /= gcd_fps;
	m_InputParam.input.scale /= gcd_fps;
	m_pStatus->m_nOutputFPSRate = m_InputParam.input.rate;
	m_pStatus->m_nOutputFPSScale = m_InputParam.input.scale;
	return ret;
}

int NVEncCore::EncoderStartup(const InEncodeVideoParam *param) {
	Close();

	m_InputParam = *param;

	m_log_level = NV_LOG_INFO;

	if (!check_if_nvcuda_dll_available()) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("CUDAが使用できないため、NVEncによるエンコードが行えません。(check_if_nvcuda_dll_available)\n"));
		return 1;
	}

	NVEncoderGPUInfo gpuInfo;
	if (0 >= gpuInfo.getGPUList().size()) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("NVEncが使用可能なGPUが見つかりませんでした。\n"));
		m_log_level = NV_LOG_DEBUG;
		InitCuda(m_InputParam.deviceID);
		return 1;
	}

	//Open OutputFile
	if (_tfopen_s(&m_fOutput, m_InputParam.outputFilename.c_str(), _T("wb")) || NULL == m_fOutput) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("出力ファイルのオープンに失敗しました。\n"));
		return 1;
	}

	//Open Input, Get Input Info
	if (InitInput()) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("入力ファイルを開けませんでした。\n"));
		return 1;
	}


	//m_stEncoderInput[0]は使わないが初期化しておく
	InitDefault();

	if (NVEncCoreOpenEncodeSession(m_InputParam.deviceID, m_InputParam.enc_config.profileGUID, m_InputParam.preset)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("エンコーダを開けませんでした。(NVEncCoreOpenEncodeSession)\n"));
		return 1;
	}

	m_nvEncCaps = GetCurrentDeviceNVEncCapability();
	if (0 == m_nvEncCaps.size()) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("エンコーダ機能のチェックに失敗しました。\n"));
		return 1;
	}

	PreInitializeEncoder();
	if (SetInputParam(&m_InputParam)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("パラメータが不適切なため、エンコードを続行できません。\n"));
		return 1;
	}
	if (S_OK != InitializeEncoder()) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("エンコーダを開けませんでした。(InitializeEncoder)\n"));
		return 1;
	}

	if (m_EncThread.Init(param->inputBuffer)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("エンコードスレッド用バッファの確保に失敗しました。\n"));
		return 1;
	}

	nvPrintf(stderr, NV_LOG_INFO, "%s\n", GetEncodingParamsInfo(m_log_level).c_str());

	return 0;
}

unsigned int __stdcall NVEncCore::RunEncThreadLauncher(void *pParam) {
	NVEncCore *ptr = reinterpret_cast<NVEncCore *>(pParam);
	return ptr->RunEncodeThread();
}

int NVEncCore::RunEncodeThread() {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	const uint32_t width = m_dwFrameWidth;
	const uint32_t height = m_dwFrameHeight;
	EncodeFrameConfig stEncodeFrame = { 0 };
	stEncodeFrame.width = width;
	stEncodeFrame.height = height;
	stEncodeFrame.stride[0] = width;
	stEncodeFrame.stride[1] = width / 2;
	stEncodeFrame.stride[2] = width / 2;
	stEncodeFrame.picStruct = m_InputParam.pic_struct;

	const uint32_t preDecodeFrame = m_EncThread.GetBufferSize() - 1;
	for (uint32_t i_frame = 0; i_frame < preDecodeFrame; i_frame++) {
		//入力フレームのポインタ取得
		EncodeInputSurfaceInfo *pInput = NULL;
		EncodeOutputBuffer *pOutputBitstream = NULL;
		m_stInputSurfQueue.Remove(pInput, INFINITE);
		m_stOutputSurfQueue.Remove(pOutputBitstream, INFINITE);

		//これをスレッドに渡す
		m_EncThread.SetNextSurface(pInput, pOutputBitstream);
	}

	for (uint32_t i_frame = 0; ; i_frame++) {
		//入力フレームのポインタ取得
		EncodeInputSurfaceInfo *pInput = NULL;
		EncodeOutputBuffer *pOutputBitstream = NULL;
		m_stInputSurfQueue.Remove(pInput, INFINITE);
		m_stOutputSurfQueue.Remove(pOutputBitstream, INFINITE);

		//これをスレッドに渡す
		m_EncThread.SetNextSurface(pInput, pOutputBitstream);

		//デコード・入力結果をもらう
		m_EncThread.GetNextFrame(&pInput, &pOutputBitstream);
		if (m_EncThread.m_bthForceAbort || (NVENC_THREAD_FINISHED == m_EncThread.m_stsThread && i_frame == m_pStatus->m_sData.frameIn))
			break;

		EncodeFrameConfig frame = stEncodeFrame;
		//EncodeFrame(&stEncodeFrame, false);
		NV_ENC_MAP_INPUT_RESOURCE mapRes = { 0 };

		//CUDAを使って転送
		cuCtxPushCurrent(m_cuContext);
		CUcontext cuContextCurr;
		CUresult result = cuMemcpyHtoD((CUdeviceptr)pInput->pExtAlloc, pInput->pExtAllocHost, pInput->dwCuPitch*pInput->dwHeight*3/2);
		if (CUDA_SUCCESS != result) {
			m_EncThread.m_errMes = _T("GPUへのフレーム転送に失敗しました。\n");
			nvStatus = NV_ENC_ERR_GENERIC;
			break;
		}
		cuCtxPopCurrent(&cuContextCurr);

		SET_VER(mapRes, NV_ENC_MAP_INPUT_RESOURCE);
		mapRes.registeredResource  = pInput->hRegisteredHandle;
		if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapRes))) {
			m_EncThread.m_errMes = _T("フレームのエンコードに失敗しました。\n  nvEncMapInputResource - ");
			m_EncThread.m_errMes += to_tchar(_nvencGetErrorEnum(nvStatus)).c_str();
			break;
		}
		pInput->hInputSurface = mapRes.mappedResource;

		memset(&m_stEncodePicParams, 0, sizeof(m_stEncodePicParams));
		SET_VER(m_stEncodePicParams, NV_ENC_PIC_PARAMS);
		m_stEncodePicParams.inputBuffer     = pInput->hInputSurface;
		m_stEncodePicParams.bufferFmt       = pInput->bufferFmt;
		m_stEncodePicParams.inputWidth      = pInput->dwWidth;
		m_stEncodePicParams.inputHeight     = pInput->dwHeight;
		m_stEncodePicParams.outputBitstream = pOutputBitstream->hBitstreamBuffer;
		m_stEncodePicParams.completionEvent = m_bAsyncModeEncoding == true ? pOutputBitstream->hOutputEvent : NULL;
		m_stEncodePicParams.pictureStruct   = frame.picStruct;
		m_stEncodePicParams.encodePicFlags  = 0;
		m_stEncodePicParams.inputTimeStamp  = 0;
		m_stEncodePicParams.inputDuration   = 0;
		m_stEncodePicParams.codecPicParams.h264PicParams.sliceMode = m_stEncodeConfig.encodeCodecConfig.h264Config.sliceMode;
		m_stEncodePicParams.codecPicParams.h264PicParams.sliceModeData = m_stEncodeConfig.encodeCodecConfig.h264Config.sliceModeData;
		memcpy(&m_stEncodePicParams.rcParams, &m_stEncodeConfig.rcParams, sizeof(m_stEncodePicParams.rcParams));

		if (NV_ENC_SUCCESS == (nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &m_stEncodePicParams))) {
			EncoderThreadData stThreadData;
			stThreadData.pOutputBfr = pOutputBitstream;
			stThreadData.pInputBfr = pInput;
			pOutputBitstream->bWaitOnEvent = true;
			stThreadData.pOutputBfr->bReconfiguredflag = frame.bReconfigured;

			// Queue o/p Sample
			if (!m_pEncoderThread->QueueSample(stThreadData)) {
				assert(0);
			}
		} else {
			m_EncThread.m_errMes = _T("フレームのエンコードに失敗しました。\n  nvEncEncodePicture - ");
			m_EncThread.m_errMes += to_tchar(_nvencGetErrorEnum(nvStatus)).c_str();
			break;
		}
	}

	FlushEncoder();
	
	_InterlockedExchange((uint32_t *)&m_EncThread.m_stsThread, (NV_ENC_SUCCESS != nvStatus) ? NVENC_THREAD_ERROR : NVENC_THREAD_FINISHED);

	return 0;
}

int NVEncCore::Run() {
	m_pStatus->m_sData.tmStart = timeGetTime();

	if (m_EncThread.RunEncFuncbyThread(RunEncThreadLauncher, this)) {
		nvPrintf(stderr, NV_LOG_ERROR, _T("エンコードスレッドの起動に失敗しました。\n"));
		return 1;
	}
	int ret = NVENC_THREAD_RUNNING;

	uint32_t encThreadBufSize = m_EncThread.GetBufferSize();
	for (uint32_t i_frame = 0; NVENC_THREAD_RUNNING == m_EncThread.m_stsThread; i_frame++) {
		//開いているフレームバッファへのポインタをもらう
		InputThreadBuf *pInputBuf = &m_EncThread.m_InputBuf[i_frame % encThreadBufSize];
		WaitForSingleObject(pInputBuf->heInputStart, INFINITE);

		//入力フレーム取得 (デコード), 色空間しながらコピー
		if (0 != (ret = m_pInput->LoadNextFrame(pInputBuf->pInput))) {
			_InterlockedExchange((uint32_t *)&m_EncThread.m_stsThread, ret);
		}

		//フレームバッファへの格納を通知
		SetEvent(pInputBuf->heInputDone);

		if (0 != ret)
			break;

		m_pStatus->UpdateDisplay();

		if (NVENC_THREAD_ERROR == m_EncThread.m_stsThread) {
			ret = NVENC_THREAD_ERROR;
			break;
		}
	}

	m_EncThread.WaitToFinish(ret);
	DestroyEncoder();

	if (ret == NVENC_THREAD_ERROR && _tcslen(m_EncThread.m_errMes.c_str()))
		nvPrintf(stderr, NV_LOG_ERROR, "%s\n", m_EncThread.m_errMes.c_str());

	//結果表示
	m_pStatus->writeResult();

	if (m_fOutput) {
		fclose(m_fOutput);
		m_fOutput = NULL;
	}

	return (ret == NVENC_THREAD_FINISHED) ? 0 : 1;
}

void NVEncCore::setInitializedVUIParam(NV_ENC_CONFIG *conf) {
	conf->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo = 0;
	conf->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix = get_cx_value(list_colormatrix, "undef");
	conf->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries = get_cx_value(list_colorprim, "undef");
	conf->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics = get_cx_value(list_transfer, "undef");
	conf->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat = get_cx_value(list_videoformat, "undef");
}

NV_ENC_CONFIG NVEncCore::initializedParam() {
	const int DEFAULT_GOP_LENGTH  = 300;
	const int DEFAULT_B_FRAMES    = 3;
	const int DEFAULT_REF_FRAMES  = 3;
	const int DEFAULT_NUM_SLICES  = 1;
	const int DEFAUTL_QP_I        = 21;
	const int DEFAULT_QP_P        = 24;
	const int DEFAULT_QP_B        = 26;
	const int DEFAULT_AVG_BITRATE = 6000000;
	const int DEFAULT_MAX_BITRATE = 15000000;

	NV_ENC_CONFIG config = { 0 };
	SET_VER(config, NV_ENC_CONFIG);
	config.frameFieldMode                 = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
	config.profileGUID                    = NV_ENC_H264_PROFILE_HIGH_GUID;
	config.gopLength                      = DEFAULT_GOP_LENGTH;
	config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_CONSTQP;
	config.encodeCodecConfig.h264Config.level;
	config.frameIntervalP                 = DEFAULT_B_FRAMES + 1;
	config.mvPrecision                    = NV_ENC_MV_PRECISION_QUARTER_PEL;
	config.monoChromeEncoding             = 0;
	config.rcParams.averageBitRate        = DEFAULT_AVG_BITRATE;
	config.rcParams.maxBitRate            = DEFAULT_MAX_BITRATE;
	config.rcParams.enableInitialRCQP     = 1;
	config.rcParams.initialRCQP.qpInterB  = DEFAULT_QP_B;
	config.rcParams.initialRCQP.qpInterP  = DEFAULT_QP_P;
	config.rcParams.initialRCQP.qpIntra   = DEFAUTL_QP_I;
	config.rcParams.constQP.qpInterB      = DEFAULT_QP_B;
	config.rcParams.constQP.qpInterP      = DEFAULT_QP_P;
	config.rcParams.constQP.qpIntra       = DEFAUTL_QP_I;

	config.rcParams.vbvBufferSize         = 0;
	config.rcParams.vbvInitialDelay       = 0;
	config.encodeCodecConfig.h264Config.level = NV_ENC_LEVEL_AUTOSELECT;

	config.encodeCodecConfig.h264Config.idrPeriod      = config.gopLength;
	config.encodeCodecConfig.h264Config.bdirectMode    = (config.frameIntervalP - 1 > 0) ? NV_ENC_H264_BDIRECT_MODE_TEMPORAL : NV_ENC_H264_BDIRECT_MODE_DISABLE;

	config.encodeCodecConfig.h264Config.disableDeblockingFilterIDC = 0;
	config.encodeCodecConfig.h264Config.disableSPSPPS  = 0;
	config.encodeCodecConfig.h264Config.sliceMode      = 3;
	config.encodeCodecConfig.h264Config.sliceModeData  = DEFAULT_NUM_SLICES;
	config.encodeCodecConfig.h264Config.maxNumRefFrames = DEFAULT_REF_FRAMES;
	config.encodeCodecConfig.h264Config.bdirectMode    = NV_ENC_H264_BDIRECT_MODE_AUTOSELECT;

	setInitializedVUIParam(&config);

	return config;
}

bool NVEncCore::checkNVEncAvialable() {
	NVEncoderGPUInfo gpuInfo;
	if (0 >= gpuInfo.getGPUList().size()) {
		return false;
	}
	return 0 == NVEncCoreOpenEncodeSession(0);
}

tstring NVEncCore::GetEncodingParamsInfo(int output_level) {
	tstring str;
	auto add_str =[output_level, &str](int info_level, const TCHAR *fmt, ...) {
		if (info_level >= output_level) {
			va_list args;
			va_start(args, fmt);
			const size_t append_len = _vsctprintf(fmt, args) + 1;
			size_t current_len = _tcslen(str.c_str());
			str.resize(current_len + append_len, 0);
			_vstprintf_s(&str[current_len], append_len, fmt, args);
		}
	};

	auto value_or_auto =[](int value, int value_auto) {
		tstring str;
		if (value == value_auto) {
			str = _T("auto");
		} else {
			TCHAR buf[256];
			sprintf_s(buf, _countof(buf), _T("%d"), value);
			str = buf;
		}
		return str;
	};

	auto on_off =[](int value) {
		return (value) ? _T("on") : _T("off");
	};

	TCHAR cpu_info[1024] = { 0 };
	getCPUInfo(cpu_info, _countof(cpu_info));

	TCHAR gpu_info[1024] = { 0 };
	getGPUInfo("NVIDIA", gpu_info, _countof(gpu_info));

	add_str(NV_LOG_ERROR, _T("NVEnc %s (%s), using NVENC API v%d.%d\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
	add_str(NV_LOG_INFO,  _T("OS バージョン           %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
	add_str(NV_LOG_INFO,  _T("CPU情報                 %s\n"), cpu_info);
	add_str(NV_LOG_INFO,  _T("GPU情報                 %s\n"), gpu_info);
	add_str(NV_LOG_ERROR, _T("NVENC API Interface     %s\n"), to_tchar(nvenc_interface_names[m_stEncoderInput[0].interfaceType].name).c_str());
	add_str(NV_LOG_ERROR, _T("入力フレームバッファ    %d frames\n"), m_EncThread.GetBufferSize());
	add_str(NV_LOG_ERROR, _T("入力フレーム情報        %s\n"), m_pInput->getInputMes());
	add_str(NV_LOG_ERROR, _T("出力動画情報            %s\n"), to_tchar(get_name_from_guid(m_stEncodeConfig.profileGUID, codecprofile_names, _countof(codecprofile_names))).c_str());
	add_str(NV_LOG_ERROR, _T("                        %dx%d%s %d:%d %.3ffps (%d/%dfps)\n"), m_dwFrameWidth, m_dwFrameHeight, (m_stEncodeConfig.frameFieldMode != NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME) ? _T("i") : _T("p"), 0,0, m_stInitEncParams.frameRateNum / (double)m_stInitEncParams.frameRateDen, m_stInitEncParams.frameRateNum, m_stInitEncParams.frameRateDen);
	add_str(NV_LOG_DEBUG, _T("Encoder Preset          %s\n"), to_tchar(preset_names[m_InputParam.preset].name).c_str());
	add_str(NV_LOG_ERROR, _T("レート制御モード        %s\n"), to_tchar(ratecontrol_names[m_stEncodeConfig.rcParams.rateControlMode].name).c_str());
	if (NV_ENC_PARAMS_RC_CONSTQP == m_stEncodeConfig.rcParams.rateControlMode) {
		add_str(NV_LOG_ERROR, _T("CQP値                   I:%d  P:%d  B:%d\n"), m_stEncodeConfig.rcParams.constQP.qpIntra, m_stEncodeConfig.rcParams.constQP.qpInterP, m_stEncodeConfig.rcParams.constQP.qpInterB);
	} else {
		add_str(NV_LOG_ERROR, _T("ビットレート            %d kbps (Max: %d kbps)\n"), m_stEncodeConfig.rcParams.averageBitRate / 1000, m_stEncodeConfig.rcParams.maxBitRate / 1000);
		if (m_stEncodeConfig.rcParams.enableInitialRCQP)
			add_str(NV_LOG_INFO,  _T("初期QP値                I:%d  P:%d  B:%d\n"), m_stEncodeConfig.rcParams.constQP.qpIntra, m_stEncodeConfig.rcParams.constQP.qpInterP, m_stEncodeConfig.rcParams.constQP.qpInterB);
		if (m_stEncodeConfig.rcParams.enableMaxQP || m_stEncodeConfig.rcParams.enableMinQP) {
			int minQPI = (m_stEncodeConfig.rcParams.enableMinQP) ? m_stEncodeConfig.rcParams.minQP.qpIntra  :  0;
			int maxQPI = (m_stEncodeConfig.rcParams.enableMaxQP) ? m_stEncodeConfig.rcParams.maxQP.qpIntra  : 51;
			int minQPP = (m_stEncodeConfig.rcParams.enableMinQP) ? m_stEncodeConfig.rcParams.minQP.qpInterP :  0;
			int maxQPP = (m_stEncodeConfig.rcParams.enableMaxQP) ? m_stEncodeConfig.rcParams.maxQP.qpInterP : 51;
			int minQPB = (m_stEncodeConfig.rcParams.enableMinQP) ? m_stEncodeConfig.rcParams.minQP.qpInterB :  0;
			int maxQPB = (m_stEncodeConfig.rcParams.enableMaxQP) ? m_stEncodeConfig.rcParams.maxQP.qpInterB : 51;
			add_str(NV_LOG_INFO,  _T("QP制御範囲              I:%d-%d  P:%d-%d  B:%d-%d\n"), minQPI, maxQPI, minQPP, maxQPP, minQPB, maxQPB);
		}
		add_str(NV_LOG_DEBUG, _T("VBV設定                 BufSize: %s  InitialDelay:%s\n"), value_or_auto(m_stEncodeConfig.rcParams.vbvBufferSize, 0).c_str(), value_or_auto(m_stEncodeConfig.rcParams.vbvInitialDelay, 0).c_str());
	}
	add_str(NV_LOG_INFO,  _T("GOP長                   %d frames\n"), m_stEncodeConfig.gopLength);
	add_str(NV_LOG_INFO,  _T("連続Bフレーム数         %d frames\n"), m_stEncodeConfig.frameIntervalP - 1);
	add_str(NV_LOG_DEBUG, _T("hierarchical Frames     P:%s  B:%s\n"), on_off(m_stEncodeConfig.encodeCodecConfig.h264Config.hierarchicalPFrames), on_off(m_stEncodeConfig.encodeCodecConfig.h264Config.hierarchicalBFrames));
	add_str(NV_LOG_DEBUG, _T("出力                    "));
	TCHAR bitstream_info[256] = { 0 };
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.outputBufferingPeriodSEI) _tcscat_s(bitstream_info, _countof(bitstream_info), _T("BufferingPeriodSEI,"));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.outputPictureTimingSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("PicTimingSEI,"));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.outputAUD)                _tcscat_s(bitstream_info, _countof(bitstream_info), _T("AUD,"));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.outputFramePackingSEI)    _tcscat_s(bitstream_info, _countof(bitstream_info), _T("FramePackingSEI,"));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.outputRecoveryPointSEI)   _tcscat_s(bitstream_info, _countof(bitstream_info), _T("RecoveryPointSEI,"));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.repeatSPSPPS)             _tcscat_s(bitstream_info, _countof(bitstream_info), _T("repeatSPSPPS,"));
	if (_tcslen(bitstream_info)) {
		bitstream_info[_tcslen(bitstream_info)-1] = _T('\0');
	} else {
		_tcscpy_s(bitstream_info, _countof(bitstream_info), _T("-"));
	}
	add_str(NV_LOG_DEBUG, "%s\n", bitstream_info);

	add_str(NV_LOG_DEBUG, _T("可変フレームレート      %s\n"), on_off(m_stEncodeConfig.encodeCodecConfig.h264Config.enableVFR));
	add_str(NV_LOG_DEBUG, _T("LTR                     %s"),   on_off(m_stEncodeConfig.encodeCodecConfig.h264Config.enableLTR));
	if (m_stEncodeConfig.encodeCodecConfig.h264Config.enableLTR) {
		add_str(NV_LOG_DEBUG, _T(", Mode:%d, NumFrames:%d"), m_stEncodeConfig.encodeCodecConfig.h264Config.ltrTrustMode, m_stEncodeConfig.encodeCodecConfig.h264Config.ltrNumFrames);
	}
	add_str(NV_LOG_DEBUG, _T("\n"));
	add_str(NV_LOG_DEBUG, _T("YUV 4:4:4               %s\n"), on_off(m_stEncodeConfig.encodeCodecConfig.h264Config.separateColourPlaneFlag));
	add_str(NV_LOG_INFO,  _T("参照距離                %d\n"), m_stEncodeConfig.encodeCodecConfig.h264Config.maxNumRefFrames);
	if (3 == m_stEncodeConfig.encodeCodecConfig.h264Config.sliceMode) {
		add_str(NV_LOG_INFO,  _T("スライス数              %d\n"), m_stEncodeConfig.encodeCodecConfig.h264Config.sliceModeData);
	} else {
		add_str(NV_LOG_INFO,  _T("スライス                Mode:%d, ModeData:%d\n"), m_stEncodeConfig.encodeCodecConfig.h264Config.sliceMode, m_stEncodeConfig.encodeCodecConfig.h264Config.sliceModeData);
	}
	add_str(NV_LOG_INFO,  _T("Adaptive Transform      %s\n"), get_desc(list_adapt_transform, m_stEncodeConfig.encodeCodecConfig.h264Config.adaptiveTransformMode));
	add_str(NV_LOG_DEBUG, _T("FMO                     %s\n"), get_desc(list_fmo, m_stEncodeConfig.encodeCodecConfig.h264Config.fmoMode));
	add_str(NV_LOG_DEBUG, _T("Coding Mode             %s\n"), get_desc(list_entropy_coding, m_stEncodeConfig.encodeCodecConfig.h264Config.entropyCodingMode));
	add_str(NV_LOG_INFO,  _T("動き予測方式            %s\n"), get_desc(list_bdirect, m_stEncodeConfig.encodeCodecConfig.h264Config.bdirectMode));
	return str;
}

HRESULT NVEncCore::CopyBitstreamData(EncoderThreadData stThreadData) {
	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	HRESULT hr = S_OK;

	if (stThreadData.pOutputBfr->hBitstreamBuffer == NULL && stThreadData.pOutputBfr->bEOSFlag == false)
	{
		return E_FAIL;
	}
	if (stThreadData.pOutputBfr->bWaitOnEvent == true)
	{
		if (!stThreadData.pOutputBfr->hOutputEvent)
		{
			return E_FAIL;
		}

#if defined (NV_WINDOWS)
		WaitForSingleObject(stThreadData.pOutputBfr->hOutputEvent, INFINITE);
#endif
	}

	if (stThreadData.pOutputBfr->bEOSFlag)
		return S_OK;

	if (m_stEncoderInput[m_dwReConfigIdx].useMappedResources)
	{
		// unmap the mapped resource ptr
		nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, stThreadData.pInputBfr->hInputSurface);
		stThreadData.pInputBfr->hInputSurface = NULL;
	}

	NV_ENC_LOCK_BITSTREAM lockBitstreamData;
	nvStatus = NV_ENC_SUCCESS;
	memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
	SET_VER(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);

	if (m_stInitEncParams.reportSliceOffsets)
		lockBitstreamData.sliceOffsets = new unsigned int[m_stEncoderInput[m_dwReConfigIdx].numSlices];

	lockBitstreamData.outputBitstream = stThreadData.pOutputBfr->hBitstreamBuffer;
	lockBitstreamData.doNotWait = false;

	nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);

	if (nvStatus == NV_ENC_SUCCESS)
	{
		m_pStatus->AddOutputInfo(&lockBitstreamData);
		fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, m_fOutput);
		nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, stThreadData.pOutputBfr->hBitstreamBuffer);
		checkNVENCErrors(nvStatus);
	}

	if (!m_stOutputSurfQueue.Add(stThreadData.pOutputBfr))
	{
		assert(0);
	}

	if (!m_stInputSurfQueue.Add(stThreadData.pInputBfr))
	{
		assert(0);
	}

	if (lockBitstreamData.sliceOffsets)
		delete(lockBitstreamData.sliceOffsets);

	if (nvStatus != NV_ENC_SUCCESS)
		hr = E_FAIL;

	return hr;
}
