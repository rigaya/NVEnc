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

#include <sstream>
#include <cstdio>
#include "rgy_device_info_cache_nvenc.h"

namespace {

const TCHAR *UNKNOWN_NVENC_CAP_NAME = _T("Unknown");

std::vector<std::string> split_str(const std::string& str, const char delim) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::string guid_to_string(const GUID& guid) {
    return strsprintf("%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        guid.Data1, guid.Data2, guid.Data3,
        guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
        guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);
}

bool guid_from_string(const std::string& str, GUID& guid) {
    unsigned int data1 = 0, data2 = 0, data3 = 0;
    unsigned int data4[8] = { 0 };
    if (std::sscanf(str.c_str(), "%8x-%4x-%4x-%2x%2x-%2x%2x%2x%2x%2x%2x",
        &data1, &data2, &data3,
        &data4[0], &data4[1], &data4[2], &data4[3],
        &data4[4], &data4[5], &data4[6], &data4[7]) != 11) {
        return false;
    }
    guid.Data1 = data1;
    guid.Data2 = (decltype(guid.Data2))data2;
    guid.Data3 = (decltype(guid.Data3))data3;
    for (int i = 0; i < 8; i++) {
        guid.Data4[i] = (decltype(guid.Data4[i]))data4[i];
    }
    return true;
}

bool equal_guid(const GUID& a, const GUID& b) {
    return memcmp(&a, &b, sizeof(GUID)) == 0;
}

bool equal_codec_feature(const NVEncCodecFeature& a, const NVEncCodecFeature& b) {
    if (!equal_guid(a.codec, b.codec)) {
        return false;
    }
    if (a.profiles.size() != b.profiles.size()
        || a.presets.size() != b.presets.size()
        || a.surfaceFmt.size() != b.surfaceFmt.size()
        || a.caps.size() != b.caps.size()) {
        return false;
    }
    for (size_t i = 0; i < a.profiles.size(); i++) {
        if (!equal_guid(a.profiles[i], b.profiles[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < a.presets.size(); i++) {
        if (!equal_guid(a.presets[i], b.presets[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < a.surfaceFmt.size(); i++) {
        if (a.surfaceFmt[i] != b.surfaceFmt[i]) {
            return false;
        }
    }
    for (size_t i = 0; i < a.caps.size(); i++) {
        if (a.caps[i].id != b.caps[i].id || a.caps[i].value != b.caps[i].value) {
            return false;
        }
    }
    return true;
}

bool equal_codec_feature_list(const std::vector<NVEncCodecFeature>& a, const std::vector<NVEncCodecFeature>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!equal_codec_feature(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

void fill_nvenc_cap_metadata(const RGY_CODEC codec, NVEncCap& cap) {
    const bool check_h264 = codec == RGY_CODEC_H264;
    cap.name = UNKNOWN_NVENC_CAP_NAME;
    cap.isBool = false;
    cap.desc = nullptr;
    cap.desc_bit_flag = nullptr;
    switch ((NV_ENC_CAPS)cap.id) {
    case NV_ENC_CAPS_NUM_ENCODER_ENGINES:
        cap.name = _T("Encoder Engines");
        break;
    case NV_ENC_CAPS_NUM_MAX_BFRAMES:
        cap.name = _T("Max Bframes");
        break;
    case NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE:
        cap.name = _T("B Ref Mode");
        cap.desc = list_nvenc_caps_bref_mode;
        break;
    case NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES:
        cap.name = _T("RC Modes");
        cap.desc_bit_flag = list_nvenc_rc_method_en;
        break;
    case NV_ENC_CAPS_SUPPORT_FIELD_ENCODING:
        cap.name = _T("Field Encoding");
        cap.desc = list_nvenc_caps_field_encoding;
        break;
    case NV_ENC_CAPS_SUPPORT_MONOCHROME:
        cap.name = _T("MonoChrome");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING:
        cap.name = _T("Alpha Channel");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_FMO:
        if (check_h264) {
            cap.name = _T("FMO");
            cap.isBool = true;
        }
        break;
    case NV_ENC_CAPS_SUPPORT_QPELMV:
        cap.name = _T("Quater-Pel MV");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_BDIRECT_MODE:
        cap.name = _T("B Direct Mode");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_CABAC:
        if (check_h264) {
            cap.name = _T("CABAC");
            cap.isBool = true;
        }
        break;
    case NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM:
        if (check_h264) {
            cap.name = _T("Adaptive Transform");
            cap.isBool = true;
        }
        break;
    case NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS:
        cap.name = _T("Max Temporal Layers");
        break;
    case NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES:
        cap.name = _T("Hierarchial P Frames");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES:
        cap.name = _T("Hierarchial B Frames");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_LEVEL_MAX:
        cap.name = _T("Max Level");
        cap.desc = get_level_list(codec);
        break;
    case NV_ENC_CAPS_LEVEL_MIN:
        cap.name = _T("Min Level");
        cap.desc = get_level_list(codec);
        break;
    case NV_ENC_CAPS_SUPPORT_YUV444_ENCODE:
        cap.name = _T("4:4:4");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_YUV422_ENCODE:
        cap.name = _T("4:2:2");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_WIDTH_MIN:
        cap.name = _T("Min Width");
        break;
    case NV_ENC_CAPS_WIDTH_MAX:
        cap.name = _T("Max Width");
        break;
    case NV_ENC_CAPS_HEIGHT_MIN:
        cap.name = _T("Min Height");
        break;
    case NV_ENC_CAPS_HEIGHT_MAX:
        cap.name = _T("Max Height");
        break;
    case NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES:
        cap.name = _T("Multiple Refs");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_NUM_MAX_LTR_FRAMES:
        cap.name = _T("Max LTR Frames");
        break;
    case NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE:
        cap.name = _T("Dynamic Resolution Change");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE:
        cap.name = _T("Dynamic Bitrate Change");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP:
        cap.name = _T("Forced constant QP");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE:
        cap.name = _T("Dynamic RC Mode Change");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK:
        cap.name = _T("Subframe Readback");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING:
        cap.name = _T("Constrained Encoding");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_INTRA_REFRESH:
        cap.name = _T("Intra Refresh");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE:
        cap.name = _T("Custom VBV Bufsize");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE:
        cap.name = _T("Dynamic Slice Mode");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION:
        cap.name = _T("Ref Pic Invalidiation");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_PREPROC_SUPPORT:
        cap.name = _T("PreProcess");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT:
        cap.name = _T("Async Encoding");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_MB_NUM_MAX:
        cap.name = _T("Max MBs");
        break;
    case NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE:
        cap.name = _T("Lossless");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_SAO:
        cap.name = _T("SAO");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_MEONLY_MODE:
        cap.name = _T("Me Only Mode");
        cap.desc = list_nvenc_caps_me_only;
        break;
    case NV_ENC_CAPS_SUPPORT_LOOKAHEAD:
        cap.name = _T("Lookahead");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ:
        cap.name = _T("AQ (temporal)");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION:
        cap.name = _T("Weighted Prediction");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_TEMPORAL_FILTER:
        cap.name = _T("Temporal Filter");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_LOOKAHEAD_LEVEL:
        cap.name = _T("Lookahead Level");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B:
        cap.name = _T("Undirectional B");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC:
        cap.name = _T("Temporal SVC");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_MVHEVC_ENCODE:
        cap.name = _T("MV-HEVC");
        cap.isBool = true;
        break;
    case NV_ENC_CAPS_SUPPORT_10BIT_ENCODE:
        cap.name = _T("10bit depth");
        cap.isBool = true;
        break;
    default:
        break;
    }
}

void fill_nvenc_feature_metadata(NVEncCodecFeature& feature) {
    const auto codec = codec_guid_enc_to_rgy(feature.codec);
    for (auto& cap : feature.caps) {
        fill_nvenc_cap_metadata(codec, cap);
    }
}

} // namespace

NVEncDeviceInfoCache::NVEncDeviceInfoCache() : RGYDeviceInfoCache(), m_deviceEncFeatures() {}

NVEncDeviceInfoCache::~NVEncDeviceInfoCache() {
}

RGY_ERR NVEncDeviceInfoCache::parseEncFeatures(std::ifstream& cacheFile) {
    m_deviceEncFeatures.clear();
    std::map<int, std::vector<NVEncCodecFeature>> deviceFeatures;

    std::string line;
    int currentDeviceId = -1;
    bool inCodec = false;
    NVEncCodecFeature currentFeature;
    while (std::getline(cacheFile, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.rfind("device\t", 0) == 0) {
            if (inCodec) {
                return RGY_ERR_INVALID_FORMAT;
            }
            try {
                currentDeviceId = std::stoi(line.substr(strlen("device\t")));
            } catch (...) {
                return RGY_ERR_INVALID_FORMAT;
            }
            continue;
        }
        if (line == "enddevice") {
            if (inCodec) {
                return RGY_ERR_INVALID_FORMAT;
            }
            currentDeviceId = -1;
            continue;
        }
        if (currentDeviceId < 0) {
            return RGY_ERR_INVALID_FORMAT;
        }
        if (line.rfind("codec\t", 0) == 0) {
            if (inCodec) {
                return RGY_ERR_INVALID_FORMAT;
            }
            GUID codecGuid = {};
            if (!guid_from_string(line.substr(strlen("codec\t")), codecGuid)) {
                return RGY_ERR_INVALID_FORMAT;
            }
            currentFeature = NVEncCodecFeature(codecGuid);
            inCodec = true;
            continue;
        }
        if (line == "endcodec") {
            if (!inCodec) {
                return RGY_ERR_INVALID_FORMAT;
            }
            fill_nvenc_feature_metadata(currentFeature);
            deviceFeatures[currentDeviceId].push_back(currentFeature);
            currentFeature = NVEncCodecFeature();
            inCodec = false;
            continue;
        }
        if (!inCodec) {
            return RGY_ERR_INVALID_FORMAT;
        }

        const auto tabPos = line.find('\t');
        if (tabPos == std::string::npos) {
            return RGY_ERR_INVALID_FORMAT;
        }
        const auto key = line.substr(0, tabPos);
        const auto value = line.substr(tabPos + 1);
        if (key == "profiles") {
            currentFeature.profiles.clear();
            if (!value.empty()) {
                for (const auto& item : split_str(value, ',')) {
                    GUID guid = {};
                    if (!guid_from_string(item, guid)) {
                        return RGY_ERR_INVALID_FORMAT;
                    }
                    currentFeature.profiles.push_back(guid);
                }
            }
        } else if (key == "presets") {
            currentFeature.presets.clear();
            if (!value.empty()) {
                for (const auto& item : split_str(value, ',')) {
                    GUID guid = {};
                    if (!guid_from_string(item, guid)) {
                        return RGY_ERR_INVALID_FORMAT;
                    }
                    currentFeature.presets.push_back(guid);
                }
            }
        } else if (key == "surface_fmt") {
            currentFeature.surfaceFmt.clear();
            if (!value.empty()) {
                for (const auto& item : split_str(value, ',')) {
                    try {
                        currentFeature.surfaceFmt.push_back((NV_ENC_BUFFER_FORMAT)std::stoi(item));
                    } catch (...) {
                        return RGY_ERR_INVALID_FORMAT;
                    }
                }
            }
        } else if (key == "caps") {
            currentFeature.caps.clear();
            if (!value.empty()) {
                for (const auto& item : split_str(value, ',')) {
                    const auto eqPos = item.find('=');
                    if (eqPos == std::string::npos) {
                        return RGY_ERR_INVALID_FORMAT;
                    }
                    NVEncCap cap = { 0 };
                    try {
                        cap.id = std::stoi(item.substr(0, eqPos));
                        cap.value = std::stoi(item.substr(eqPos + 1));
                    } catch (...) {
                        return RGY_ERR_INVALID_FORMAT;
                    }
                    currentFeature.caps.push_back(cap);
                }
            }
        } else {
            return RGY_ERR_INVALID_FORMAT;
        }
    }
    if (inCodec || currentDeviceId >= 0) {
        return RGY_ERR_INVALID_FORMAT;
    }

    for (auto& [deviceId, features] : deviceFeatures) {
        m_deviceEncFeatures.push_back(std::make_pair(deviceId, features));
    }
    return RGY_ERR_NONE;
}

void NVEncDeviceInfoCache::clearFeatureCache() {
    RGYDeviceInfoCache::clearFeatureCache();
    m_deviceEncFeatures.clear();
}

void NVEncDeviceInfoCache::setEncFeatures(const std::map<int, RGYDeviceInfoCacheKey>& deviceInfos, const DeviceEncodeFeatures& deviceEncodeFeatures) {
    setDeviceInfos(deviceInfos);
    for (const auto& [deviceId, features] : deviceEncodeFeatures) {
        auto it = std::find_if(m_deviceEncFeatures.begin(), m_deviceEncFeatures.end(), [deviceId](const auto& data) {
            return data.first == deviceId;
        });
        if (it == m_deviceEncFeatures.end()) {
            m_deviceEncFeatures.push_back(std::make_pair(deviceId, features));
            m_dataUpdated = true;
            continue;
        }
        if (!equal_codec_feature_list(it->second, features)) {
            it->second = features;
            m_dataUpdated = true;
        }
    }
}

const std::vector<NVEncCodecFeature> *NVEncDeviceInfoCache::getEncFeatures(int deviceId, const RGYDeviceInfoCacheKey& deviceInfo) const {
    const auto itDevice = m_deviceInfos.find(deviceId);
    if (itDevice == m_deviceInfos.end() || itDevice->second != deviceInfo) {
        return nullptr;
    }
    const auto it = std::find_if(m_deviceEncFeatures.begin(), m_deviceEncFeatures.end(), [deviceId](const auto& data) {
        return data.first == deviceId;
    });
    return (it == m_deviceEncFeatures.end()) ? nullptr : &it->second;
}

void NVEncDeviceInfoCache::writeEncFeatures(std::ofstream& cacheFile) {
    cacheFile << ENC_FEATURES_START_LINE << std::endl;
    for (const auto& [deviceId, features] : m_deviceEncFeatures) {
        cacheFile << "device\t" << deviceId << std::endl;
        for (const auto& feature : features) {
            cacheFile << "codec\t" << guid_to_string(feature.codec) << std::endl;

            cacheFile << "profiles\t";
            for (size_t i = 0; i < feature.profiles.size(); i++) {
                if (i > 0) cacheFile << ",";
                cacheFile << guid_to_string(feature.profiles[i]);
            }
            cacheFile << std::endl;

            cacheFile << "presets\t";
            for (size_t i = 0; i < feature.presets.size(); i++) {
                if (i > 0) cacheFile << ",";
                cacheFile << guid_to_string(feature.presets[i]);
            }
            cacheFile << std::endl;

            cacheFile << "surface_fmt\t";
            for (size_t i = 0; i < feature.surfaceFmt.size(); i++) {
                if (i > 0) cacheFile << ",";
                cacheFile << (int)feature.surfaceFmt[i];
            }
            cacheFile << std::endl;

            cacheFile << "caps\t";
            for (size_t i = 0; i < feature.caps.size(); i++) {
                if (i > 0) cacheFile << ",";
                cacheFile << feature.caps[i].id << "=" << feature.caps[i].value;
            }
            cacheFile << std::endl;
            cacheFile << "endcodec" << std::endl;
        }
        cacheFile << "enddevice" << std::endl;
    }
}
