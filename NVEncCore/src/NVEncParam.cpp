//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "NVEncCore.h"
#include "NVEncParam.h"

NVEncParam::NVEncParam() {
	mCurrentDeviceID = -1;
	thCreateCache = NULL;
}

NVEncParam::~NVEncParam() {
	mCurrentDeviceID = -1;
	DestroyEncoder();
	if (thCreateCache) {
		CloseHandle(thCreateCache);
		thCreateCache = NULL;
	}
}

int NVEncParam::OpenEncoder(int deviceID) {
	if (NULL == m_pEncodeAPI)
		return 0;
	if (mCurrentDeviceID == deviceID)
		return 0;
	if (mCurrentDeviceID >= 0) {
		DestroyEncoder();
	}
	if (!check_if_nvcuda_dll_available())
		return 1;

	mCurrentDeviceID = -1;

	NVEncoderGPUInfo gpuInfo;
	if (deviceID >= (int)gpuInfo.getGPUList().size())
		return 1;

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
		return 1;
	}
	m_uRefCount++;

	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &m_dwEncodeGUIDCount))) {
		return 1;
	}
	uint32_t uArraysize = 0;
	std::vector<GUID> encodeCodecGUIDs(m_dwEncodeGUIDCount, GUID{ 0 });
	if (NV_ENC_SUCCESS != (nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, &encodeCodecGUIDs[0], m_dwEncodeGUIDCount, &uArraysize))) {
		return 1;
	}
	bool codecAvailable = false;
	for (auto codecGUID : encodeCodecGUIDs) {
		if (GetCodecType(codecGUID) == target_codec) {
			m_stEncodeGUID = codecGUID;
			codecAvailable = true;
			break;
		}
	}
	if (!codecAvailable) {
		return 1;
	}

	mCurrentDeviceID = deviceID;

	return 0;
}

std::vector<NV_ENC_CONFIG> NVEncParam::GetNVEncH264Preset(int deviceID) {
	std::vector<NV_ENC_CONFIG> config;

	if (OpenEncoder(deviceID))
		return config;

	NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
	uint32_t uPresetCount = 0;
	std::vector<GUID> presetGUIDs;
	if (NV_ENC_SUCCESS == (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, m_stEncodeGUID, &uPresetCount))) {
		presetGUIDs.resize(uPresetCount, GUID{ 0 });
		config.resize(uPresetCount, NV_ENC_CONFIG{ 0 });
		uint32_t uPresetCount2 = 0;
		if (NV_ENC_SUCCESS == (nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, m_stEncodeGUID, &presetGUIDs[0], uPresetCount, &uPresetCount2))) {

            for (unsigned int i = 0; i < uPresetCount; i++) {
				NV_ENC_PRESET_CONFIG preset_config;
				memset(&preset_config, 0, sizeof(NV_ENC_PRESET_CONFIG));
				SET_VER(preset_config, NV_ENC_PRESET_CONFIG);
				SET_VER(preset_config.presetCfg, NV_ENC_CONFIG);
                nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, m_stEncodeGUID, presetGUIDs[i], &preset_config);
				setInitializedVUIParam(&preset_config.presetCfg);
				config[i] = preset_config.presetCfg;
            }
        }
    }
	return config;
}

std::vector<NVEncCap> NVEncParam::GetNVEncCapability(int deviceID) {
	std::vector<NVEncCap> caps;

	if (OpenEncoder(deviceID))
		return caps;

	return GetCurrentDeviceNVEncCapability();
}

unsigned int __stdcall NVEncParam::createCacheLoader(void *prm) {
	NVEncParam *nvencParam = reinterpret_cast<NVEncParam *>(prm);
	int deviceID = nvencParam->mCurrentDeviceID;
	nvencParam->mCurrentDeviceID = -1;
	nvencParam->createCache(deviceID);
	return 0;
}

void NVEncParam::createCache(int deviceID) {
	if (OpenEncoder(deviceID))
		return;
	m_presetCache = GetNVEncH264Preset(deviceID);
	m_capsCache = GetNVEncCapability(deviceID);
	return;
}

int NVEncParam::createCacheAsync(int deviceID) {
	mCurrentDeviceID = deviceID;
	thCreateCache = (HANDLE)_beginthreadex(NULL, 0, createCacheLoader, this, 0, NULL);
	if (NULL == thCreateCache) {
		return 1;
	}
	return 0;
}

std::vector<NV_ENC_CONFIG> NVEncParam::GetCachedNVEncH264Preset() {
	if (thCreateCache) {
		WaitForSingleObject(thCreateCache, INFINITE);
		CloseHandle(thCreateCache);
		thCreateCache = NULL;
	}
	return m_presetCache;
}

std::vector<NVEncCap> NVEncParam::GetCachedNVEncCapability() {
	if (thCreateCache) {
		WaitForSingleObject(thCreateCache, INFINITE);
		CloseHandle(thCreateCache);
		thCreateCache = NULL;
	}
	return m_capsCache;
}
