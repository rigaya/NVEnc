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
#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#else
#include <sys/wait.h>
#endif
#include "rgy_device_usage.h"
#include "rgy_util.h"
#include "rgy_filesystem.h"

RGYDeviceUsageLockManager::RGYDeviceUsageLockManager(RGYDeviceUsageHeader *header) : m_header(header) {
    int32_t expected = 0;
    int32_t desired = 1;

#if 0
    std::atomic_ref<int32_t> lock(m_header->lock);
    while (!lock.atomic_compare_exchange_weak(&expected, desired)) {
        std::this_thread::yield();
        expected = 1;
    }
#elif defined(_WIN32) || defined(_WIN64)
    //InterlockedCompareExchange(&m_header->lock, desired, expected);
    for (;;) {
        static_assert(sizeof(m_header->lock) == sizeof(long));
        if (_InterlockedCompareExchange((long *)&m_header->lock, (long)desired, (long)expected) == expected) {
            break;
        }
        std::this_thread::yield();
    }
#else
    while (__sync_val_compare_and_swap(&m_header->lock, expected, desired) != expected) {
        std::this_thread::yield();
    }
#endif
}


RGYDeviceUsageLockManager::~RGYDeviceUsageLockManager() {
    m_header->lock = 0;
}

RGYDeviceUsage::RGYDeviceUsage() : m_sharedMem(), m_header(nullptr), m_entries(nullptr), m_monitorProcess(), m_addedEntry(false) {
}


RGYDeviceUsage::~RGYDeviceUsage() {
    close();
}

void RGYDeviceUsage::close() {
    release();
    m_header = nullptr;
    m_entries = nullptr;
    if (m_sharedMem) {
        m_sharedMem->detach();
    }
    m_sharedMem.reset();
    m_monitorProcess.reset();
}

RGY_ERR RGYDeviceUsage::open() {
    if (m_sharedMem) {
        return RGY_ERR_NONE;
    }
#if defined(_WIN32) || defined(_WIN64)
    m_sharedMem = std::make_unique<RGYSharedMemWin>();
#else
    m_sharedMem = std::make_unique<RGYSharedMemLinux>();
#endif
    const char *sm_name = RGY_DEVICE_USAGE_SHARED_MEM_NAME;
    m_sharedMem->open(sm_name, sizeof(RGYDeviceUsageHeader) + sizeof(RGYDeviceUsageEntry) * RGY_DEVICE_USAGE_MAX_ENTRY);
    if (!m_sharedMem->is_open()) {
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    m_header = (RGYDeviceUsageHeader *)m_sharedMem->ptr();
    m_entries = (RGYDeviceUsageEntry *)(m_header + 1);
    char header[RGY_DEVICE_USAGE_HEADER_STR_SIZE] = { 0 };
    memcpy(header, RGY_DEVICE_USAGE_SHARED_MEM_NAME, strlen(RGY_DEVICE_USAGE_SHARED_MEM_NAME) + 1);
    if (memcmp(m_header->header, header, sizeof(header)) != 0) {
        memset(m_header, 0, sizeof(RGYDeviceUsageHeader));
        RGYDeviceUsageLockManager lock(m_header);
        memcpy(m_header->header, header, sizeof(header));
        memset(m_entries, 0, sizeof(RGYDeviceUsageEntry) * RGY_DEVICE_USAGE_MAX_ENTRY);
    }
    return RGY_ERR_NONE;
}

void RGYDeviceUsage::check(const time_t now_time_from_epoch) {
    bool removed = false;
    for (int i = 0; i < RGY_DEVICE_USAGE_MAX_ENTRY; i++) {
        if (m_entries[i].process_id == 0) {
            break;
        }
        if (now_time_from_epoch - m_entries[i].start_time > 10 * 60) {
            // m_entries[i].process_id のプロセスが存在しない場合、エントリを削除する
            if (!RGYProcessExists(m_entries[i].process_id)) {
                m_entries[i].process_id = 0;
                removed = true;
            }
        }
    }
    if (removed) {
        std::vector<RGYDeviceUsageEntry> tmp;
        for (int i = 0; i < RGY_DEVICE_USAGE_MAX_ENTRY; i++) {
            if (m_entries[i].process_id != 0) {
                tmp.push_back(m_entries[i]);
            }
        }
        memcpy(m_entries, tmp.data(), sizeof(m_entries[0]) * tmp.size());
    }
}

RGY_ERR RGYDeviceUsage::add(int32_t device_id) {
    if (!m_sharedMem) {
        open();
    }
    if (m_header == nullptr || m_entries == nullptr) {
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    const auto time_from_epoch = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto process_id = GetCurrentProcessId();

    RGYDeviceUsageLockManager lock(m_header);

    check(time_from_epoch);

    for (int i = 0; i < RGY_DEVICE_USAGE_MAX_ENTRY; i++) {
        if (m_entries[i].process_id == 0) {
            m_entries[i].process_id = process_id;
            m_entries[i].device_id = device_id;
            m_entries[i].start_time = time_from_epoch;
            m_addedEntry = true;
            return RGY_ERR_NONE;
        }
    }
    return RGY_ERR_DEVICE_NOT_FOUND;
}

void RGYDeviceUsage::resetEntry() {
    if (!m_sharedMem) {
        open();
    }
    RGYDeviceUsageLockManager lock(m_header);
    memset(m_entries, 0, sizeof(RGYDeviceUsageEntry) * RGY_DEVICE_USAGE_MAX_ENTRY);
}

std::vector<std::pair<int, int64_t>> RGYDeviceUsage::getUsage() {
    std::vector<std::pair<int, int64_t>> usage;
    if (!m_sharedMem) {
        open();
    }
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const auto time_from_epoch = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    RGYDeviceUsageLockManager lock(m_header);
    check(time_from_epoch);
    for (int i = 0; i < RGY_DEVICE_USAGE_MAX_ENTRY; i++) {
        if (m_entries[i].process_id == 0) {
            break;
        }
        if (m_entries[i].device_id >= (int)usage.size()) {
            usage.resize(m_entries[i].device_id + 1, std::make_pair(0, std::numeric_limits<int>::max()));
        }
        usage[m_entries[i].device_id].first++;
        usage[m_entries[i].device_id].second = std::min(usage[m_entries[i].device_id].second, (int64_t)time_from_epoch - m_entries[i].start_time);
    }
    return usage;
}

void RGYDeviceUsage::release() {
    if (!m_addedEntry || !m_entries) {
        return;
    }
    const auto process_id = GetCurrentProcessId();
    RGYDeviceUsageLockManager lock(m_header);
    for (int i = 0; i < RGY_DEVICE_USAGE_MAX_ENTRY; i++) {
        if (m_entries[i].process_id == 0) {
            break;
        }
        
        if (m_entries[i].process_id == process_id) {
            m_entries[i].process_id = 0;
            // ひとつ前にずらす
            // i.. 4, j .. 5,6,7...
            for (int j = i + 1; j < RGY_DEVICE_USAGE_MAX_ENTRY; j++) {
                if (m_entries[j].process_id == 0) {
                    if (j - i - 1 > 0) {
                        memmove(&m_entries[i], &m_entries[i+1], sizeof(m_entries[0]) * (j - i - 1));
                    }
                    m_entries[j-1].process_id = 0;
                    break;
                }
            }
            break;
        }
    }
    m_addedEntry = false;
}

RGY_ERR RGYDeviceUsage::startProcessMonitor(int32_t device_id) {
    m_monitorProcess = createRGYPipeProcess();
    m_monitorProcess->init(PIPE_MODE_DISABLE, PIPE_MODE_DISABLE, PIPE_MODE_DISABLE);

    std::vector<tstring> args = {
        getExePath(),
        _T("--process-monitor-dev-usage"),
        _T("--parent-pid"),
        strsprintf(_T("%x"), GetCurrentProcessId()),
        _T("-d"),
        strsprintf(_T("%d"), device_id),
        _T("-i"), // ダミー
        _T("-"), // ダミー
        _T("-o"), // ダミー
        _T("-") // ダミー
    };

    if (auto err = m_monitorProcess->run(args, nullptr, 0, true, true); err != 0) {
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

int processMonitorRGYDeviceUsage(const uint32_t ppid, const int32_t deviceID) {
#if defined(_WIN32) || defined(_WIN64)
    auto parentHandle = std::unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter>(OpenProcess(SYNCHRONIZE, FALSE, ppid), handle_deleter());
    if (!parentHandle) {
        fprintf(stderr, "Failed to open parent process (pid: %d)\n", ppid);
        return 1;
    }
    int ret = 0;
    RGYDeviceUsage deviceUsage;
    if (deviceUsage.open() != RGY_ERR_NONE) {
        fprintf(stderr, "Failed to open shared memory\n"); ret = 1;
    } else if (deviceUsage.add(deviceID) != RGY_ERR_NONE) {
        fprintf(stderr, "Failed to add entry\n"); ret = 1;
    } else {
        WaitForSingleObject(parentHandle.get(), INFINITE);
    }
    parentHandle.reset();
    return ret;
#else
    // Linuxで指定のpidのプロセスの終了を待機
    int status = 0;
    auto parent_pid = (pid_t)ppid;
    if (waitpid(parent_pid, &status, 0) == -1) { perror("waitpid"); return 1; }
    return 0;
#endif
}

int processMonitorRGYDeviceResetEntry() {
    int ret = 0;
    RGYDeviceUsage deviceUsage;
    if (deviceUsage.open() != RGY_ERR_NONE) {
        fprintf(stderr, "Failed to open shared memory\n"); ret = 1;
    } else {
        deviceUsage.resetEntry();
    }
    return ret;
}