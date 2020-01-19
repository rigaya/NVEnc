// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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
// --------------------------------------------------------------------------------------------

#include <cstdio>
#include <regex>
#include <numeric>
#include <chrono>

#include "rgy_perf_counter.h"
#define _WIN32_DCOM
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")
#include <atlstr.h>

CounterEntry::CounterEntry() :
    luid({ 0,0 }),
    pid(0),
    phys(0),
    eng(0),
    type(),
    val(0) {

}

int CounterEntry::set(const wchar_t *name, DWORD value) {
    //pid_23656_luid_0x00000000_0x00017403_phys_0_eng_3_engtype_VideoDecode
    const wchar_t *pattern = LR"(pid_(\d+)_luid_0x([0-9A-Fa-f]{8})_0x([0-9A-Fa-f]{8})_phys_(\d+)_eng_(\d+)_engtype_(.+))";
    std::wregex re(pattern);
    std::wcmatch match;
    if (!std::regex_match(name, match, re)) {
        return 1;
    }
    try {
        pid = std::stoi(match[1]);
        luid.HighPart = std::stoi(match[2], nullptr, 16);
        luid.LowPart = std::stoi(match[3], nullptr, 16);
        phys = std::stoi(match[4]);
        eng = std::stoi(match[5]);
    } catch (...) {
        return 1;
    }
    type = match[6];
    val = (int)value;
    return 0;
}

void CounterEntry::print() const {
    wprintf(L"pid: %8d, luid: 0x%08x-0x%08x, engtype: %s, util: %d\n", pid, luid.HighPart, luid.LowPart, type.c_str(), (int)(val + 0.5f));
}

RGYGPUCounterWinEntries::RGYGPUCounterWinEntries(const std::vector<CounterEntry> &counterEntries) : entries() {
    entries.reserve(counterEntries.size());
    for (size_t i = 0; i < counterEntries.size(); i++) {
        entries.push_back(&counterEntries[i]);
    }
}

RGYGPUCounterWinEntries::~RGYGPUCounterWinEntries() {
    entries.clear();
}

RGYGPUCounterWinEntries RGYGPUCounterWinEntries::filter_luid(LUID luid) {
    std::vector<const CounterEntry *> newEntries;
    newEntries.reserve(entries.size());
    for (const auto &entry : entries) {
        if (memcmp(&entry->luid, &luid, sizeof(LUID)) == 0) {
            newEntries.push_back(entry);
        }
    }
    entries = newEntries;
    return *this;
}

RGYGPUCounterWinEntries RGYGPUCounterWinEntries::filter_pid(int pid) {
    std::vector<const CounterEntry *> newEntries;
    newEntries.reserve(entries.size());
    for (const auto &entry : entries) {
        if (entry->pid == pid) {
            newEntries.push_back(entry);
        }
    }
    entries = newEntries;
    return *this;
}

RGYGPUCounterWinEntries RGYGPUCounterWinEntries::filter_phys(int phys) {
    std::vector<const CounterEntry *> newEntries;
    newEntries.reserve(entries.size());
    for (const auto &entry : entries) {
        if (entry->phys == phys) {
            newEntries.push_back(entry);
        }
    }
    entries = newEntries;
    return *this;
}

RGYGPUCounterWinEntries RGYGPUCounterWinEntries::filter_eng(int eng) {
    std::vector<const CounterEntry *> newEntries;
    newEntries.reserve(entries.size());
    for (const auto &entry : entries) {
        if (entry->eng == eng) {
            newEntries.push_back(entry);
        }
    }
    entries = newEntries;
    return *this;
}

RGYGPUCounterWinEntries RGYGPUCounterWinEntries::filter_type(const std::wstring &type) {
    const auto type_lower = tolowercase(type);
    std::vector<const CounterEntry *> newEntries;
    newEntries.reserve(entries.size());
    for (const auto &entry : entries) {
        if (wcsstr(tolowercase(entry->type).c_str(), type_lower.c_str()) != nullptr) {
            newEntries.push_back(entry);
        }
    }
    entries = newEntries;
    return *this;
}

std::vector<CounterEntry> RGYGPUCounterWinEntries::get() const {
    std::vector<CounterEntry> counters;
    counters.reserve(entries.size());
    for (const auto &entry : entries) {
        counters.push_back(*entry);
    }
    return counters;
}

int RGYGPUCounterWinEntries::sum() const {
    return std::accumulate(entries.begin(), entries.end(), 0,
        [](int sum, const CounterEntry *e) { return sum + e->val; });
}

std::wstring RGYGPUCounterWinEntries::tolowercase(const std::wstring &str) {
    auto temp = _wcsdup(str.data());
    _wcslwr(temp);
    std::wstring str_lo = temp;
    free(temp);
    return str_lo;
}

RGYGPUCounterWin::RGYGPUCounterWin() :
    thRefresh(),
    mtxRefresh(),
    m_refreshed(false),
    m_refreshedTime(),
    counters(),
    m_nameHandle(0),
    m_utilizationHandle(0),
    pConfig(nullptr),
    pRefresher(nullptr),
    pEnum(nullptr),
    pNameSpace(nullptr),
    m_abort(false) {
}

RGYGPUCounterWin::~RGYGPUCounterWin() {
    thread_fin();
}

void RGYGPUCounterWin::close() {
    if (pNameSpace != nullptr) {
        pNameSpace->Release();
        pNameSpace = nullptr;
    }
    if (pEnum != nullptr) {
        pEnum->Release();
        pEnum = nullptr;
    }
    if (pConfig != nullptr) {
        pConfig->Release();
        pConfig = nullptr;
    }
    if (pRefresher != nullptr) {
        pRefresher->Release();
        pRefresher = nullptr;
    }
}

int RGYGPUCounterWin::init() {
    auto hr = S_OK;

    if (FAILED(hr = CoInitializeSecurity(
        NULL,
        -1,
        NULL,
        NULL,
        RPC_C_AUTHN_LEVEL_NONE,
        RPC_C_IMP_LEVEL_IMPERSONATE,
        NULL, EOAC_NONE, 0))) {
        return 1;
    }

    IWbemLocator *wbemLocator = nullptr;
    if (FAILED(hr = CoCreateInstance(
        CLSID_WbemLocator,
        NULL,
        CLSCTX_INPROC_SERVER,
        IID_IWbemLocator,
        (void **)&wbemLocator))) {
        return 1;
    }
    CComBSTR bstrNameSpace(L"\\\\.\\root\\cimv2");
    if (FAILED(hr = wbemLocator->ConnectServer(
        bstrNameSpace,
        NULL, // User name
        NULL, // Password
        NULL, // Locale
        0L,   // Security flags
        NULL, // Authority
        NULL, // Wbem context
        &pNameSpace))) {
        return 1;
    }
    wbemLocator->Release();
    wbemLocator = nullptr;

    if (FAILED(hr = CoCreateInstance(
        CLSID_WbemRefresher,
        NULL,
        CLSCTX_INPROC_SERVER,
        IID_IWbemRefresher,
        (void **)&pRefresher))) {
        return 1;
    }

    if (FAILED(hr = pRefresher->QueryInterface(
        IID_IWbemConfigureRefresher,
        (void **)&pConfig))) {
        return 1;
    }

    // Add an enumerator to the refresher.
    long lID = 0;
    if (FAILED(hr = pConfig->AddEnum(
        pNameSpace,
        L"Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
        0,
        NULL,
        &pEnum,
        &lID))) {
        return 1;
    }
    return 0;
}

int RGYGPUCounterWin::refreshCounters() {
    auto hr = S_OK;
    if (FAILED(hr = pRefresher->Refresh(0L))) {
        return 1;
    }

    ULONG numReturned = 0;
    hr = pEnum->GetObjects(0L, 0, nullptr, &numReturned);
    if (numReturned <= 0) {
        return 0;
    }

    std::vector<IWbemObjectAccess *> apEnumAccess(numReturned, nullptr);
    if (FAILED(hr = pEnum->GetObjects(0L, (ULONG)apEnumAccess.size(), apEnumAccess.data(), &numReturned))) {
        return 1;
    }

    if (m_utilizationHandle == 0) {
        CIMTYPE utilizationHandleType;
        if (FAILED(hr = apEnumAccess[0]->GetPropertyHandle(L"UtilizationPercentage", &utilizationHandleType, &m_utilizationHandle))) {
            return 1;
        }

        CIMTYPE nameHandleType;
        if (FAILED(hr = apEnumAccess[0]->GetPropertyHandle(L"Name", &nameHandleType, &m_nameHandle))) {
            return 1;
        }
    }

    std::vector<CounterEntry> new_counters;
    for (auto& acc : apEnumAccess) {
        DWORD value = 0;
        if (FAILED(hr = acc->ReadDWORD(m_utilizationHandle, &value))) {
            return 1;
        }
        if (value > 0) {
            long size = 0;
            byte str[2048];
            if (FAILED(hr = acc->ReadPropertyValue(m_nameHandle, sizeof(str), &size, str))) {
                return 1;
            }

            CounterEntry entry;
            if (entry.set((const wchar_t *)str, value) == 0) {
                //entry.print();
                new_counters.push_back(entry);
            }
        }
        acc->Release();
    }

    std::lock_guard<std::mutex> lock(mtxRefresh);
    counters = new_counters;
    return 0;
}

int RGYGPUCounterWin::thread_func() {
    auto hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (init()) {
        return 1;
    }
    while (!m_abort) {
        auto timenow = std::chrono::system_clock::now();
        if (timenow - m_refreshedTime > std::chrono::milliseconds(500)) {
            auto ret = refreshCounters();
            if (ret) break;
            m_refreshed = true;
            m_refreshedTime = timenow;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    close();
    CoUninitialize();
    return 0;
}

void RGYGPUCounterWin::thread_run() {
    m_refreshedTime = std::chrono::system_clock::now() - std::chrono::milliseconds(500);
    thRefresh = std::thread(&RGYGPUCounterWin::thread_func, this);
}

void RGYGPUCounterWin::send_thread_fin() {
    m_abort = true;
}

int RGYGPUCounterWin::thread_fin() {
    try {
        if (thRefresh.joinable()) {
            m_abort = true;
            thRefresh.join();
        }
    } catch (...) {
        return 1;
    }
    return 0;
}
