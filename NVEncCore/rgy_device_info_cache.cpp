// -----------------------------------------------------------------------------------------
//     QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2024 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <thread>
#include <chrono>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include "rgy_osdep.h"
#include "rgy_filesystem.h"
#include "rgy_device_info_cache.h"
#include "rgy_rev.h"

#define DEVICE_INFO_CACHE_FILE_NAME _T("device_info_") _T(ENCODER_NAME) _T("_cache.txt")
#define DEVICE_INFO_CACHE_HEADER "device_info_" ENCODER_NAME "_cache_v2"

const char *RGYDeviceInfoCache::DEVICE_INFO_START_LINE = "Device Info";
const char *RGYDeviceInfoCache::DEC_CSP_START_LINE = "Decoder CSP";
const char *RGYDeviceInfoCache::ENC_FEATURES_START_LINE = "Encoder Features";

RGYDeviceInfoCache::RGYDeviceInfoCache() : m_deviceIds(), m_deviceInfos(), m_deviceDecCodecCsp(), m_dataUpdated(false) {}

RGYDeviceInfoCache::~RGYDeviceInfoCache() {
    updateCacheFile();
}

std::string RGYDeviceInfoCache::getExpectedVersionInfo() const {
    return std::string(ENCODER_NAME) + " " + tchar_to_string(BUILD_ARCH_STR) + " " + VER_STR_FILEVERSION + " rev" + ENCODER_REV;
}

tstring RGYDeviceInfoCache::getCacheFilePath() const {
#if defined(_WIN32) || defined(_WIN64)
    TCHAR tempPath[4096];
    GetTempPath(_countof(tempPath), tempPath);
    return tstring(tempPath) + DEVICE_INFO_CACHE_FILE_NAME;
#else
    return tstring(_T("/tmp/")) + DEVICE_INFO_CACHE_FILE_NAME;
#endif
}

RGY_ERR RGYDeviceInfoCache::loadCacheFile() {
    m_deviceIds.clear();
    m_deviceInfos.clear();
    m_deviceDecCodecCsp.clear();

    const auto cachFilePath = getCacheFilePath();

    if (!rgy_file_exists(cachFilePath)) {
        return RGY_ERR_NOT_FOUND;
    }

    std::ifstream cacheFile;
    for (int i = 0; i < 10; i++) {
        cacheFile.open(cachFilePath, std::ios::in);
        if (cacheFile.is_open()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!cacheFile.is_open()) {
        return RGY_ERR_FILE_OPEN;
    }

    std::string line;
    if (!std::getline(cacheFile, line) || line != DEVICE_INFO_CACHE_HEADER) {
        return RGY_ERR_INVALID_VERSION;
    }

    if (!std::getline(cacheFile, line)) {
        return RGY_ERR_INVALID_FORMAT;
    }
    std::istringstream iss(line);
    time_t cacheTime = 0;
    iss >> cacheTime;

#if defined(_WIN32) || defined(_WIN64)
    ULONGLONG uptime = GetTickCount64() / 1000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    time_t uptime = ts.tv_sec;
#endif
    time_t bootTime = time(nullptr) - uptime;
    if (cacheTime < bootTime) {
        return RGY_ERR_INVALID_VERSION;
    }

    if (!std::getline(cacheFile, line)) {
        return RGY_ERR_INVALID_VERSION;
    }
    if (line != getExpectedVersionInfo()) {
        return RGY_ERR_INVALID_VERSION;
    }

    if (!std::getline(cacheFile, line) || line != DEVICE_INFO_START_LINE) {
        return RGY_ERR_INVALID_FORMAT;
    }

    auto sts = parseDeviceInfo(cacheFile);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = parseDecCsp(cacheFile);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = parseEncFeatures(cacheFile);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYDeviceInfoCache::parseDeviceInfo(std::ifstream& cacheFile) {
    m_deviceIds.clear();
    m_deviceInfos.clear();

    std::string line;
    while (std::getline(cacheFile, line)) {
        if (line == DEC_CSP_START_LINE) {
            return RGY_ERR_NONE;
        }
        if (line.empty()) {
            continue;
        }
        const auto cols = split(line, std::string("\t"));
        if (cols.size() != 5) {
            return RGY_ERR_INVALID_FORMAT;
        }
        try {
            const int deviceId = std::stoi(cols[0]);
            RGYDeviceInfoCacheKey info;
            info.deviceName = cols[1];
            info.deviceId = cols[2];
            info.deviceSubId = cols[3];
            info.driverVersion = cols[4];
            m_deviceIds[deviceId] = info.deviceName;
            m_deviceInfos[deviceId] = info;
        } catch (...) {
            return RGY_ERR_INVALID_FORMAT;
        }
    }
    return RGY_ERR_INVALID_FORMAT;
}

RGY_ERR RGYDeviceInfoCache::parseDecCsp(std::ifstream& cacheFile) {
    m_deviceDecCodecCsp.clear();

    std::string line;
    while (std::getline(cacheFile, line)) {
        if (line == ENC_FEATURES_START_LINE) {
            return RGY_ERR_NONE;
        }
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        int deviceId = 0;
        if (!(iss >> deviceId)) {
            return RGY_ERR_INVALID_FORMAT;
        }
        auto it_dev = std::find_if(m_deviceDecCodecCsp.begin(), m_deviceDecCodecCsp.end(), [deviceId](const auto& a) { return a.first == deviceId; });
        if (it_dev == m_deviceDecCodecCsp.end()) {
            m_deviceDecCodecCsp.push_back(std::make_pair(deviceId, CodecCsp{}));
            it_dev = m_deviceDecCodecCsp.end() - 1;
        }

        std::string codecNameStr;
        if (!(iss >> codecNameStr)) {
            return RGY_ERR_INVALID_FORMAT;
        }
        const RGY_CODEC codec = (RGY_CODEC)get_cx_value(list_rgy_codec, char_to_tstring(codecNameStr).c_str());
        if (codec == RGY_CODEC_UNKNOWN) {
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        std::vector<RGY_CSP> cspList;
        std::string colorSpaceStr;
        while (std::getline(iss, colorSpaceStr, ',')) {
            const tstring colorSpaceTchar = trim(char_to_tstring(colorSpaceStr));
            RGY_CSP colorSpace = RGY_CSP_NA;
            for (int i = 0; i < RGY_CSP_COUNT; i++) {
                if (colorSpaceTchar == RGY_CSP_NAMES[i]) {
                    colorSpace = (RGY_CSP)i;
                    break;
                }
            }
            if (colorSpace == RGY_CSP_NA) {
                return RGY_ERR_INVALID_FORMAT;
            }
            cspList.push_back(colorSpace);
        }
        it_dev->second[codec] = cspList;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYDeviceInfoCache::parseEncFeatures([[maybe_unused]] std::ifstream& cacheFile) {
    return RGY_ERR_NONE;
}

void RGYDeviceInfoCache::clearFeatureCache() {
    m_deviceDecCodecCsp.clear();
    m_dataUpdated = true;
}

void RGYDeviceInfoCache::setDeviceIds(const std::map<int, std::string>& deviceIds) {
    std::map<int, RGYDeviceInfoCacheKey> deviceInfos;
    for (const auto& [devId, devName] : deviceIds) {
        const auto it = m_deviceInfos.find(devId);
        deviceInfos[devId] = (it != m_deviceInfos.end()) ? it->second : RGYDeviceInfoCacheKey{};
        deviceInfos[devId].deviceName = devName;
    }
    setDeviceInfos(deviceInfos);
}

void RGYDeviceInfoCache::setDeviceInfos(const std::map<int, RGYDeviceInfoCacheKey>& deviceInfos) {
    bool deviceInfoMismatch = m_deviceInfos.size() != deviceInfos.size();
    if (!deviceInfoMismatch) {
        for (const auto& [devId, deviceInfo] : deviceInfos) {
            const auto it = m_deviceInfos.find(devId);
            if (it == m_deviceInfos.end() || it->second != deviceInfo) {
                deviceInfoMismatch = true;
                break;
            }
        }
    }
    if (deviceInfoMismatch) {
        clearFeatureCache();
    }
    if (m_deviceInfos != deviceInfos) {
        m_deviceInfos = deviceInfos;
        m_deviceIds.clear();
        for (const auto& [devId, deviceInfo] : m_deviceInfos) {
            m_deviceIds[devId] = deviceInfo.deviceName;
        }
        m_dataUpdated = true;
    }
}

void RGYDeviceInfoCache::setDecCodecCsp(const std::map<int, std::string>& deviceIds, const DeviceCodecCsp& deviceCodecCspList) {
    setDeviceIds(deviceIds);
    for (const auto& deviceCodecCsp : deviceCodecCspList) {
        if (deviceIds.count(deviceCodecCsp.first) > 0) {
            setDecCodecCsp(deviceIds.at(deviceCodecCsp.first), deviceCodecCsp);
        }
    }
}

void RGYDeviceInfoCache::setDecCodecCsp(const std::map<int, RGYDeviceInfoCacheKey>& deviceInfos, const DeviceCodecCsp& deviceCodecCspList) {
    setDeviceInfos(deviceInfos);
    for (const auto& deviceCodecCsp : deviceCodecCspList) {
        if (deviceInfos.count(deviceCodecCsp.first) > 0) {
            setDecCodecCsp(deviceInfos.at(deviceCodecCsp.first).deviceName, deviceCodecCsp);
        }
    }
}

void RGYDeviceInfoCache::setDecCodecCsp(const std::string& devName, const std::pair<int, CodecCsp>& deviceCodecCsp) {
    if (m_deviceIds.count(deviceCodecCsp.first) == 0 || m_deviceIds[deviceCodecCsp.first] != devName) {
        clearFeatureCache();
        return;
    }
    for (auto& dcc : m_deviceDecCodecCsp) {
        if (dcc.first == deviceCodecCsp.first) {
            bool updateRequired = false;
            for (const auto& [codec, support_csp] : deviceCodecCsp.second) {
                const auto support_csp_set = std::set(support_csp.begin(), support_csp.end());
                if (dcc.second.count(codec) == 0) {
                    updateRequired = true;
                    break;
                } else {
                    auto& check_csp = dcc.second[codec];
                    if (support_csp_set != std::set(check_csp.begin(), check_csp.end())) {
                        updateRequired = true;
                        break;
                    }
                }
            }
            if (updateRequired || dcc.second.size() != deviceCodecCsp.second.size()) {
                dcc = deviceCodecCsp;
                m_dataUpdated = true;
            }
            return;
        }
    }
    m_deviceDecCodecCsp.push_back(deviceCodecCsp);
    m_dataUpdated = true;
}

const CodecCsp *RGYDeviceInfoCache::getDecCodecCsp(int deviceId, const RGYDeviceInfoCacheKey& deviceInfo) const {
    const auto itDevice = m_deviceInfos.find(deviceId);
    if (itDevice == m_deviceInfos.end() || itDevice->second != deviceInfo) {
        return nullptr;
    }
    const auto it = std::find_if(m_deviceDecCodecCsp.begin(), m_deviceDecCodecCsp.end(), [deviceId](const auto& data) {
        return data.first == deviceId;
    });
    return (it == m_deviceDecCodecCsp.end()) ? nullptr : &it->second;
}

void RGYDeviceInfoCache::updateCacheFile() {
    if (m_dataUpdated) {
        saveCacheFile();
        m_dataUpdated = false;
    }
}

void RGYDeviceInfoCache::writeHeader(std::ofstream& cacheFile) {
    cacheFile << DEVICE_INFO_CACHE_HEADER << std::endl;
    cacheFile << std::time(nullptr) << std::endl;
    cacheFile << getExpectedVersionInfo() << std::endl;
}

void RGYDeviceInfoCache::writeDeviceInfo(std::ofstream& cacheFile) {
    cacheFile << DEVICE_INFO_START_LINE << std::endl;
    for (const auto& [devId, deviceInfo] : m_deviceInfos) {
        cacheFile
            << devId << '\t'
            << deviceInfo.deviceName << '\t'
            << deviceInfo.deviceId << '\t'
            << deviceInfo.deviceSubId << '\t'
            << deviceInfo.driverVersion
            << std::endl;
    }
}

void RGYDeviceInfoCache::writeDecCsp(std::ofstream& cacheFile) {
    cacheFile << DEC_CSP_START_LINE << std::endl;

    for (const auto& deviceCodecCsp : m_deviceDecCodecCsp) {
        for (const auto& colorSpace : deviceCodecCsp.second) {
            cacheFile << deviceCodecCsp.first << " " << tchar_to_string(get_cx_desc(list_rgy_codec, colorSpace.first)) << " ";
            for (size_t icsp = 0; icsp < colorSpace.second.size(); icsp++) {
                if (icsp > 0) cacheFile << ",";
                cacheFile << tchar_to_string(RGY_CSP_NAMES[colorSpace.second[icsp]]);
            }
            cacheFile << std::endl;
        }
    }
}

void RGYDeviceInfoCache::writeEncFeatures([[maybe_unused]] std::ofstream& cacheFile) {
    return;
}

RGY_ERR RGYDeviceInfoCache::saveCacheFile() {
    if (!m_dataUpdated) {
        return RGY_ERR_NONE;
    }
    const auto cachFilePath = getCacheFilePath();
    const bool cacheFileExists = rgy_file_exists(cachFilePath);
    {
        std::ofstream cacheFile(cachFilePath);
        if (!cacheFile.is_open()) {
            return RGY_ERR_FILE_OPEN;
        }
        writeHeader(cacheFile);
        writeDeviceInfo(cacheFile);
        writeDecCsp(cacheFile);
        writeEncFeatures(cacheFile);
        cacheFile.close();
    }
    if (!cacheFileExists) {
#if !(defined(_WIN32) || defined(_WIN64))
        chmod(cachFilePath.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
#endif
    }
    return RGY_ERR_NONE;
}
