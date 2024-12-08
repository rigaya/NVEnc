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

#pragma once
#ifndef __RGY_DEVICE_INFO_CACHE_H__
#define __RGY_DEVICE_INFO_CACHE_H__

#include <memory>
#include <vector>
#include <map>
#include <chrono>
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_def.h"
#include "rgy_err.h"

class RGYDeviceInfoCache {
public:
    RGYDeviceInfoCache();
    virtual ~RGYDeviceInfoCache();

    const std::map<int, std::string>& getDeviceIds() const { return m_deviceIds; }
    const DeviceCodecCsp& getDeviceDecCodecCsp() const { return m_deviceDecCodecCsp; }
    std::string getExpectedVersionInfo() const;
    RGY_ERR loadCacheFile();
    RGY_ERR saveCacheFile();
    void setDeviceIds(const std::map<int, std::string>& deviceIds);
    void setDecCodecCsp(const std::map<int, std::string>& deviceIds, const DeviceCodecCsp& deviceCodecCspList);
    void setDecCodecCsp(const std::string& devName, const std::pair<int, CodecCsp>& deviceCodecCsp);
    void updateCacheFile();
protected:
    tstring getCacheFilePath() const;

    RGY_ERR parseDecCsp(std::ifstream& cacheFile);
    virtual RGY_ERR parseEncFeatures(std::ifstream& cacheFile);

    void writeHeader(std::ofstream& cacheFile);
    void writeDecCsp(std::ofstream& cacheFile);
    virtual void writeEncFeatures(std::ofstream& cacheFile);

    virtual void clearFeatureCache();

    static const char *DEC_CSP_START_LINE;
    static const char *ENC_FEATURES_START_LINE;

    std::map<int, std::string> m_deviceIds;
    DeviceCodecCsp m_deviceDecCodecCsp;
    bool m_dataUpdated;
};

#endif //#if __RGY_DEVICE_INFO_CACHE_H__
