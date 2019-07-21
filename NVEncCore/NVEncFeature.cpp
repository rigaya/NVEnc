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

NVEncFeature::NVEncFeature() :
    m_nTargetDeviceID(0),
    m_pNVEncCore(),
    m_hThCreateCache(),
    m_hEvCreateCache(),
    m_hEvCreateCodecCache(),
    m_EncodeFeatures() {
}

NVEncFeature::~NVEncFeature() {
    m_pNVEncCore.reset();
    if (m_hThCreateCache.joinable()) {
        m_hThCreateCache.join();
    }
    m_hEvCreateCache.reset();
    m_hEvCreateCodecCache.reset();
}

int NVEncFeature::createCache(int deviceID, int loglevel) {
    if (!check_if_nvcuda_dll_available()) {
        SetEvent(m_hEvCreateCodecCache.get());
    } else {

        m_pNVEncCore.reset(new NVEncCore());

        InEncodeVideoParam inputParam;
        inputParam.encConfig = DefaultParam();
        inputParam.deviceID = deviceID;
        inputParam.ctrl.loglevel = loglevel;
        if (   NV_ENC_SUCCESS != m_pNVEncCore->Initialize(&inputParam)
            || NV_ENC_SUCCESS != m_pNVEncCore->InitDevice(&inputParam)) {
            SetEvent(m_hEvCreateCodecCache.get());
        } else {
            m_pNVEncCore->createDeviceCodecList();
            m_EncodeFeatures = m_pNVEncCore->GetNVEncCapability();
            SetEvent(m_hEvCreateCodecCache.get());

            m_pNVEncCore->createDeviceFeatureList();
            m_EncodeFeatures = m_pNVEncCore->GetNVEncCapability();
            m_pNVEncCore.reset();
        }
    }
    SetEvent(m_hEvCreateCache.get());
    return 0;
}

int NVEncFeature::createCacheAsync(int deviceID, int loglevel) {
    m_nTargetDeviceID = deviceID;
    GetCachedNVEncCapability(); //スレッドが生きていたら終了を待機
    //一度リソース開放
    m_hEvCreateCodecCache = std::unique_ptr<void, handle_deleter>(CreateEvent(NULL, TRUE, FALSE, NULL), handle_deleter());
    m_hEvCreateCache = std::unique_ptr<void, handle_deleter>(CreateEvent(NULL, TRUE, FALSE, NULL), handle_deleter());
    m_hThCreateCache = std::thread([this, deviceID, loglevel]() {
        createCache(deviceID, loglevel);
    });
    return 0;
}

bool NVEncFeature::H264Available() {
    WaitForSingleObject(m_hEvCreateCodecCache.get(), INFINITE);
    return GetH264Features(m_EncodeFeatures) != nullptr;
}

bool NVEncFeature::HEVCAvailable() {
    WaitForSingleObject(m_hEvCreateCodecCache.get(), INFINITE);
    return GetHEVCFeatures(m_EncodeFeatures) != nullptr;
}

const std::vector<NVEncCodecFeature>& NVEncFeature::GetCachedNVEncCapability() {
    WaitForSingleObject(m_hEvCreateCache.get(), INFINITE);
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