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
	m_pNVEncCore = nullptr;
	m_hThCreateCache = NULL;
	m_hEvCreateCache = NULL;
	m_hEvCreateCodecCache = NULL;
	m_bH264 = false;
	m_bHEVC = false;
}

NVEncParam::~NVEncParam() {
	if (nullptr != m_pNVEncCore) {
		delete m_pNVEncCore;
		m_pNVEncCore = nullptr;
	}
	if (m_hThCreateCache) {
		WaitForSingleObject(m_hThCreateCache, INFINITE);
		CloseHandle(m_hThCreateCache);
		m_hThCreateCache = NULL;
	}
	if (m_hEvCreateCache) {
		CloseHandle(m_hEvCreateCache);
		m_hEvCreateCache = NULL;
	}
	if (m_hEvCreateCodecCache) {
		CloseHandle(m_hEvCreateCodecCache);
		m_hEvCreateCodecCache = NULL;
	}
}

int NVEncParam::createCache(int deviceID) {
	if (!check_if_nvcuda_dll_available())
		return 1;

	m_pNVEncCore = new NVEncCore();

	InEncodeVideoParam inputParam = { 0 };
	inputParam.encConfig = NVEncCore::DefaultParam();
	inputParam.deviceID = deviceID;
	if (NV_ENC_SUCCESS != m_pNVEncCore->Initialize(&inputParam))
		return 1;

	m_pNVEncCore->createDeviceCodecList();
	m_EncodeFeatures = m_pNVEncCore->GetNVEncCapability();
	m_bH264 = nullptr != GetH264Features(m_EncodeFeatures);
	m_bHEVC = nullptr != GetHEVCFeatures(m_EncodeFeatures);
	SetEvent(m_hEvCreateCodecCache);

	m_pNVEncCore->createDeviceFeatureList();
	m_EncodeFeatures = m_pNVEncCore->GetNVEncCapability();
	if (nullptr != m_pNVEncCore) {
		delete m_pNVEncCore;
		m_pNVEncCore = nullptr;
	}
	SetEvent(m_hEvCreateCache);
	return 0;
}

unsigned int __stdcall NVEncParam::createCacheLoader(void *prm) {
	NVEncParam *nvencParam = reinterpret_cast<NVEncParam *>(prm);
	int deviceID = nvencParam->m_nTargetDeviceID;
	nvencParam->m_nTargetDeviceID = -1;
	nvencParam->createCache(deviceID);
	return 0;
}

int NVEncParam::createCacheAsync(int deviceID) {
	m_nTargetDeviceID = deviceID;
	GetCachedNVEncCapability(); //スレッドが生きていたら終了を待機
	//一度リソース開放
	if (m_hEvCreateCodecCache) CloseHandle(m_hEvCreateCodecCache);
	if (m_hEvCreateCache) CloseHandle(m_hEvCreateCache);
	if (m_hThCreateCache) CloseHandle(m_hThCreateCache);
	//イベントの作成とスレッドの起動
	m_hEvCreateCodecCache = (HANDLE)CreateEvent(NULL, TRUE, FALSE, NULL);
	m_hEvCreateCache      = (HANDLE)CreateEvent(NULL, TRUE, FALSE, NULL);
	m_hThCreateCache      = (HANDLE)_beginthreadex(NULL, 0, createCacheLoader, this, 0, NULL);
	if (NULL == m_hThCreateCache) {
		return 1;
	}
	return 0;
}

bool NVEncParam::H264Available() {
	WaitForSingleObject(m_hEvCreateCodecCache, INFINITE);
	return m_bH264;
}

bool NVEncParam::HEVCAvailable() {
	WaitForSingleObject(m_hEvCreateCodecCache, INFINITE);
	return m_bHEVC;
}

const std::vector<NVEncCodecFeature>& NVEncParam::GetCachedNVEncCapability() {
	WaitForSingleObject(m_hEvCreateCache, INFINITE);
	return m_EncodeFeatures;
}

const NVEncCodecFeature *NVEncParam::GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures) {
	for (uint32_t i = 0; i < codecFeatures.size(); i++) {
		if (0 == memcmp(&codecFeatures[i].codec, &NV_ENC_CODEC_HEVC_GUID, sizeof(GUID))) {
			return &codecFeatures[i];
		}
	}
	return nullptr;
}
const NVEncCodecFeature *NVEncParam::GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures) {
	for (uint32_t i = 0; i < codecFeatures.size(); i++) {
		if (0 == memcmp(&codecFeatures[i].codec, &NV_ENC_CODEC_H264_GUID, sizeof(GUID))) {
			return &codecFeatures[i];
		}
	}
	return nullptr;
}