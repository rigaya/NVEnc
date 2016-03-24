// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "NVEncCore.h"
#include "NVEncFeature.h"
#include "NVEncFeature.h"

typedef void* nvfeature_t;

nvfeature_t nvfeature_create() {
    return new NVEncFeature();
}
int nvfeature_createCacheAsync(nvfeature_t obj, int deviceID) {
    return reinterpret_cast<NVEncFeature *>(obj)->createCacheAsync(deviceID);
}
const std::vector<NVEncCodecFeature>& nvfeature_GetCachedNVEncCapability(nvfeature_t obj) {
    return reinterpret_cast<NVEncFeature *>(obj)->GetCachedNVEncCapability();
}

//featureリストからHEVCのリストを取得 (HEVC非対応ならnullptr)
const NVEncCodecFeature *nvfeature_GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures) {
    return NVEncFeature::GetHEVCFeatures(codecFeatures);
}
//featureリストからHEVCのリストを取得 (H.264対応ならnullptr)
const NVEncCodecFeature *nvfeature_GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures) {
    return NVEncFeature::GetH264Features(codecFeatures);
}

//H.264が使用可能かどうかを取得 (取得できるまで待機)
bool nvfeature_H264Available(nvfeature_t obj) {
    return reinterpret_cast<NVEncFeature *>(obj)->H264Available();
}

//HEVCが使用可能かどうかを取得 (取得できるまで待機)
bool nvfeature_HEVCAvailable(nvfeature_t obj) {
    return reinterpret_cast<NVEncFeature *>(obj)->HEVCAvailable();
}

void nvfeature_close(nvfeature_t obj) {
    NVEncFeature *ptr = reinterpret_cast<NVEncFeature *>(obj);
    delete ptr;
}

NVEncFeature::NVEncFeature() {
    m_pNVEncCore = nullptr;
    m_hThCreateCache = NULL;
    m_hEvCreateCache = NULL;
    m_hEvCreateCodecCache = NULL;
    m_bH264 = false;
    m_bHEVC = false;
}

NVEncFeature::~NVEncFeature() {
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

int NVEncFeature::createCache(int deviceID) {
    if (!check_if_nvcuda_dll_available()) {
        SetEvent(m_hEvCreateCodecCache);
    } else {

        m_pNVEncCore = new NVEncCore();

        InEncodeVideoParam inputParam;
        inputParam.encConfig = NVEncCore::DefaultParam();
        inputParam.deviceID = deviceID;
        if (NV_ENC_SUCCESS != m_pNVEncCore->Initialize(&inputParam)) {
            SetEvent(m_hEvCreateCodecCache);
        } else {
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
        }
    }
    SetEvent(m_hEvCreateCache);
    return 0;
}

unsigned int __stdcall NVEncFeature::createCacheLoader(void *prm) {
    NVEncFeature *nvencParam = reinterpret_cast<NVEncFeature *>(prm);
    int deviceID = nvencParam->m_nTargetDeviceID;
    nvencParam->m_nTargetDeviceID = -1;
    nvencParam->createCache(deviceID);
    return 0;
}

int NVEncFeature::createCacheAsync(int deviceID) {
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

bool NVEncFeature::H264Available() {
    WaitForSingleObject(m_hEvCreateCodecCache, INFINITE);
    return m_bH264;
}

bool NVEncFeature::HEVCAvailable() {
    WaitForSingleObject(m_hEvCreateCodecCache, INFINITE);
    return m_bHEVC;
}

const std::vector<NVEncCodecFeature>& NVEncFeature::GetCachedNVEncCapability() {
    WaitForSingleObject(m_hEvCreateCache, INFINITE);
    return m_EncodeFeatures;
}

const NVEncCodecFeature *NVEncFeature::GetHEVCFeatures(const std::vector<NVEncCodecFeature>& codecFeatures) {
    for (uint32_t i = 0; i < codecFeatures.size(); i++) {
        if (0 == memcmp(&codecFeatures[i].codec, &NV_ENC_CODEC_HEVC_GUID, sizeof(GUID))) {
            return &codecFeatures[i];
        }
    }
    return nullptr;
}
const NVEncCodecFeature *NVEncFeature::GetH264Features(const std::vector<NVEncCodecFeature>& codecFeatures) {
    for (uint32_t i = 0; i < codecFeatures.size(); i++) {
        if (0 == memcmp(&codecFeatures[i].codec, &NV_ENC_CODEC_H264_GUID, sizeof(GUID))) {
            return &codecFeatures[i];
        }
    }
    return nullptr;
}