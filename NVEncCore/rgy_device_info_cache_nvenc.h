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

#pragma once
#ifndef __RGY_DEVICE_INFO_CACHE_NVENC_H__
#define __RGY_DEVICE_INFO_CACHE_NVENC_H__

#include "rgy_device_info_cache.h"
#include "NVEncDevice.h"

typedef std::vector<std::pair<int, std::vector<NVEncCodecFeature>>> DeviceEncodeFeatures;

class NVEncDeviceInfoCache : public RGYDeviceInfoCache {
public:
    NVEncDeviceInfoCache();
    virtual ~NVEncDeviceInfoCache();

    const DeviceEncodeFeatures& getDeviceEncFeatures() const { return m_deviceEncFeatures; }
    void setEncFeatures(const std::map<int, RGYDeviceInfoCacheKey>& deviceInfos, const DeviceEncodeFeatures& deviceEncodeFeatures);
    const std::vector<NVEncCodecFeature> *getEncFeatures(int deviceId, const RGYDeviceInfoCacheKey& deviceInfo) const;

protected:
    virtual RGY_ERR parseEncFeatures(std::ifstream& cacheFile) override;
    virtual void writeEncFeatures(std::ofstream& cacheFile) override;
    virtual void clearFeatureCache() override;

    DeviceEncodeFeatures m_deviceEncFeatures;
};

#endif //#if __RGY_DEVICE_INFO_CACHE_NVENC_H__
