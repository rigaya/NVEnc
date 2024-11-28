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
#ifndef __RGY_DEVICE_USAGE_H__
#define __RGY_DEVICE_USAGE_H__

#include <memory>
#include <vector>
#include <chrono>
#include "rgy_osdep.h"
#include "rgy_version.h"
#include "rgy_shared_mem.h"
#include "rgy_pipe.h"
#include "rgy_err.h"

#define RGY_DEVICE_USAGE_SHARED_MEM_NAME ("RGY_DEVICE_USAGE_SHARED_MEM_" ENCODER_NAME)
static const int RGY_DEVICE_USAGE_SHARED_MEM_KEY_ID = 34589;
static const int RGY_DEVICE_USAGE_MAX_ENTRY = 1024;
static const int RGY_DEVICE_USAGE_HEADER_STR_SIZE = 64;

#pragma pack(push,1)
struct RGYDeviceUsageHeader {
    char header[RGY_DEVICE_USAGE_HEADER_STR_SIZE];
    int32_t lock;
    int32_t reserved[127];
};

struct RGYDeviceUsageEntry {
    uint32_t process_id;
    int32_t device_id;
    time_t start_time;
};
#pragma pack(pop)

class RGYDeviceUsageLockManager {
    RGYDeviceUsageHeader *m_header;
public:
    RGYDeviceUsageLockManager(RGYDeviceUsageHeader *header);
    ~RGYDeviceUsageLockManager();
};

class RGYDeviceUsage {
public:
    RGYDeviceUsage();
    ~RGYDeviceUsage();

    RGY_ERR open();
    RGY_ERR add(int32_t device_id);
    void check(const time_t now_time_from_epoch);
    void release();
    void close();
    void resetEntry();
    RGY_ERR startProcessMonitor(int32_t device_id);
    std::vector<std::pair<int, int64_t>> getUsage();
protected:
    std::unique_ptr<RGYSharedMem> m_sharedMem;
    RGYDeviceUsageHeader *m_header;
    RGYDeviceUsageEntry *m_entries;
    std::unique_ptr<RGYPipeProcess> m_monitorProcess;
    bool m_addedEntry;
};

int processMonitorRGYDeviceUsage(const uint32_t ppid, const int32_t deviceID);
int processMonitorRGYDeviceResetEntry();

#endif //#if __RGY_DEVICE_USAGE_H__
