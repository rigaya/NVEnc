// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2020 rigaya
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

#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <climits>
#include <condition_variable>
#include "rgy_tchar.h"
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <emmintrin.h>
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#include "rgy_osdep.h"
#include "rgy_arch.h"
#include "rgy_util.h"
#include "rgy_version.h"
#include "cpu_info.h"
#if ENCODER_QSV
#include "qsv_query.h"
#endif

int getCPUName(char *buffer, size_t nSize) {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    int CPUInfo[4] = {-1};
    __cpuid(CPUInfo, 0x80000000);
    unsigned int nExIds = CPUInfo[0];
    if (nSize < 0x40)
        return 1;

    memset(buffer, 0, 0x40);
    for (unsigned int i = 0x80000000; i <= nExIds; i++) {
        __cpuid(CPUInfo, i);
        int offset = 0;
        switch (i) {
            case 0x80000002: offset =  0; break;
            case 0x80000003: offset = 16; break;
            case 0x80000004: offset = 32; break;
            default:
                continue;
        }
        memcpy(buffer + offset, CPUInfo, sizeof(CPUInfo));
    }
    auto remove_string =[](char *target_str, const char *remove_str) {
        char *ptr = strstr(target_str, remove_str);
        if (nullptr != ptr) {
            memmove(ptr, ptr + strlen(remove_str), (strlen(ptr) - strlen(remove_str) + 1) *  sizeof(target_str[0]));
        }
    };
    remove_string(buffer, "(R)");
    remove_string(buffer, "(TM)");
    remove_string(buffer, "CPU");
    //crop space beforce string
    for (int i = 0; buffer[i]; i++) {
        if (buffer[i] != ' ') {
            if (i)
                memmove(buffer, buffer + i, strlen(buffer + i) + 1);
            break;
        }
    }
    //remove space which continues.
    for (int i = 0; buffer[i]; i++) {
        if (buffer[i] == ' ') {
            int space_idx = i;
            while (buffer[i+1] == ' ')
                i++;
            if (i != space_idx)
                memmove(buffer + space_idx + 1, buffer + i + 1, strlen(buffer + i + 1) + 1);
        }
    }
    //delete last blank
    if (0 < strlen(buffer)) {
        char *last_ptr = buffer + strlen(buffer) - 1;
        if (' ' == *last_ptr)
            *last_ptr = '\0';
    }
    return 0;
#else
    std::string arch;
    std::string name;
    memset(buffer, 0, 0x40);
    FILE *fp = NULL;
    const char *cmdline = "lscpu";
    if ((fp = popen(cmdline, "r")) == NULL) {
        return 1;
    }
    char buf[1024];
    while (!feof(fp)) {
        if (fgets(buf, sizeof(buf), fp) == nullptr) {
            break;
        }
        if (strstr(buf, "Architecture:") != nullptr) {
            //改行の削除
            char *ptr = buf + strlen(buf) - 1;
            if (*ptr == '\n') *ptr = '\0';
            //Architectureの部分の取得
            ptr = buf + strlen("Architecture:");
            while (*ptr == ' ')
                ptr++;
            arch = ptr;
        }
        if (strstr(buf, "Model name:") != nullptr) {
            //改行の削除
            char *ptr = buf + strlen(buf) - 1;
            if (*ptr == '\n') *ptr = '\0';
            //Model nameの部分の取得
            ptr = buf + strlen("Model name:");
            while (*ptr == ' ')
                ptr++;
            name = ptr;
        }
    }
    sprintf(buffer, "%s %s", name.c_str(), arch.c_str());
    return 0;
#endif
}

#if _MSC_VER
static int getCPUName(wchar_t *buffer, size_t nSize) {
    int ret = 0;
    char *buf = (char *)calloc(nSize, sizeof(char));
    if (NULL == buf) {
        buffer[0] = L'\0';
        ret = 1;
    } else {
        if (0 == (ret = getCPUName(buf, nSize))) {
            if (MultiByteToWideChar(CP_ACP, 0, buf, -1, buffer, (DWORD)nSize) == 0) {
                buffer[0] = L'\0';
                ret = 1;
            }
        }
        free(buf);
    }
    return ret;
}
#endif //#if _MSC_VER

double getCPUDefaultClockFromCPUName() {
    double defaultClock = 0.0;
    TCHAR buffer[1024] = { 0 };
    getCPUName(buffer, _countof(buffer));
    TCHAR *ptr_mhz = _tcsstr(buffer, _T("MHz"));
    TCHAR *ptr_ghz = _tcsstr(buffer, _T("GHz"));
    TCHAR *ptr = _tcschr(buffer, _T('@'));
    bool clockInfoAvailable = (NULL != ptr_mhz || ptr_ghz != NULL) && NULL != ptr;
    if (clockInfoAvailable && 1 == _stscanf_s(ptr+1, _T("%lf"), &defaultClock)) {
        return defaultClock * ((NULL == ptr_ghz) ? 1000.0 : 1.0);
    }
    return 0.0;
}

#if defined(_WIN32) || defined(_WIN64)

typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

static DWORD CountSetBits(ULONG_PTR bitMask) {
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    for (ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT; bitTest; bitTest >>= 1)
        bitSetCount += ((bitMask & bitTest) != 0);

    return bitSetCount;
}

bool get_cpu_info(cpu_info_t *cpu_info) {
    if (nullptr == cpu_info)
        return false;

    memset(cpu_info, 0, sizeof(cpu_info[0]));

    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(_T("kernel32")), "GetLogicalProcessorInformation");
    if (nullptr == glpi)
        return false;

    DWORD returnLength = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = nullptr;
    while (FALSE == glpi(buffer, &returnLength)) {
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            if (buffer)
                free(buffer);
            if (NULL == (buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength)))
                return FALSE;
        }
    }

    DWORD processorPackageCount = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    for (DWORD byteOffset = 0; byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength;
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
        switch (ptr->Relationship) {
        case RelationNumaNode:
            // Non-NUMA systems report a single record of this type.
            cpu_info->nodes++;
            break;
        case RelationProcessorCore:
            cpu_info->physical_cores++;
            // A hyperthreaded core supplies more than one logical processor.
            cpu_info->logical_cores += CountSetBits(ptr->ProcessorMask);
            break;

        case RelationCache:
        {
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
            PCACHE_DESCRIPTOR Cache = &ptr->Cache;
            if (1 <= Cache->Level && Cache->Level <= _countof(cpu_info->caches)) {
                cache_info_t *cache = &cpu_info->caches[Cache->Level-1];
                cache->count++;
                cache->level = Cache->Level;
                cache->linesize = Cache->LineSize;
                cache->size += Cache->Size;
                cache->associativity = Cache->Associativity;
                cpu_info->max_cache_level = (std::max)(cpu_info->max_cache_level, cache->level);
            }
            break;
        }
        case RelationProcessorPackage:
            // Logical processors share a physical package.
            processorPackageCount++;
            break;

        default:
            //Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.
            break;
        }
        ptr++;
    }
    if (buffer)
        free(buffer);

    return true;
}

#else //#if defined(_WIN32) || defined(_WIN64)
#include <iostream>
#include <fstream>

bool get_cpu_info(cpu_info_t *cpu_info) {
    memset(cpu_info, 0, sizeof(cpu_info[0]));
    std::ifstream inputFile("/proc/cpuinfo");
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string script_data = std::string(data_begin, data_end);
    inputFile.close();

    std::vector<processor_info_t> processor_list;
    processor_info_t info = { 0 };
    info.processor_id = info.core_id = info.socket_id = -1;

    for (auto line : split(script_data, "\n")) {
        auto pos = line.find("processor");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), "%d", &i)) {
                if (info.processor_id >= 0) {
                    if (info.socket_id < 0) info.socket_id = 0; // physical id がない場合
                    if (info.core_id < 0) info.core_id = info.processor_id; // core id がない場合
                    processor_list.push_back(info);
                    info.processor_id = info.core_id = info.socket_id = -1; // 次に備えて初期化
                }
                info.processor_id = i;
            }
            continue;
        }
        pos = line.find("core id");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), "%d", &i)) {
                info.core_id = i;
            }
            continue;
        }
        pos = line.find("physical id");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), "%d", &i)) {
                info.socket_id = i;
            }
            continue;
        }
    }
    if (info.processor_id >= 0) {
        if (info.socket_id < 0) info.socket_id = 0; // physical id がない場合
        if (info.core_id < 0) info.core_id = info.processor_id; // core id がない場合
        processor_list.push_back(info);
    }

    std::sort(processor_list.begin(), processor_list.end(), [](const processor_info_t& a, const processor_info_t& b) {
        if (a.socket_id != b.socket_id) return a.socket_id < b.socket_id;
        if (a.core_id != b.core_id) return a.core_id < b.core_id;
        return a.processor_id < b.processor_id;
    });
    int physical_core_count = 0;
    uint64_t last_key = UINT64_MAX;
    for (uint32_t i = 0; i < processor_list.size(); i++) {
        uint64_t key = ((uint64_t)processor_list[i].socket_id << 32) | processor_list[i].core_id;
        physical_core_count += key != last_key;
        last_key = key;
    }
    memcpy(cpu_info->proc_list, processor_list.data(), sizeof(processor_list[0]) * processor_list.size());
    cpu_info->nodes = processor_list.back().socket_id + 1;
    cpu_info->physical_cores = physical_core_count;
    cpu_info->logical_cores = processor_list.size();
    return true;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

cpu_info_t get_cpu_info() {
    cpu_info_t cpu;
    get_cpu_info(&cpu);
    return cpu;
}

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
const int TEST_COUNT = 5000;
RGY_NOINLINE
int64_t runl_por(int loop_count, int& dummy_dep) {
    unsigned int dummy;
    const auto ts = __rdtscp(&dummy);
    int i = loop_count;
#define ADD_XOR { i += loop_count; i ^= loop_count; }
#define ADD_XOR4 {ADD_XOR;ADD_XOR;ADD_XOR;ADD_XOR;}
#define ADD_XOR16 {ADD_XOR4;ADD_XOR4;ADD_XOR4;ADD_XOR4;}
    do {
        ADD_XOR16;
        ADD_XOR16;
        ADD_XOR16;
        ADD_XOR16;
        loop_count--;
    } while (loop_count > 0);
    const auto te = __rdtscp(&dummy);
    dummy_dep = i;
    return te - ts;
}

//rdtscpを使うと0xc0000096例外 (一般ソフトウェア例外)を発する場合があるらしい
//そこでそれを検出する
bool check_rdtscp_available() {
#if defined(_WIN32) || defined(_WIN64)
    __try {
        UINT dummy;
        __rdtscp(&dummy);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#endif //defined(_WIN32) || defined(_WIN64)
    return true;
}

static double get_tick_per_clock() {
    const int outer_loop_count = 100;
    const int inner_loop_count = TEST_COUNT;
    int dummy = 0;
    auto tick_min = runl_por(inner_loop_count, dummy);
    for (int i = 0; i < outer_loop_count; i++) {
        auto ret = runl_por(inner_loop_count, dummy);
        tick_min = std::min(tick_min, ret);
    }
    return tick_min / (128.0 * inner_loop_count);
}

static double get_tick_per_sec() {
    const int outer_loop_count = TEST_COUNT;
    int dummy = 0;
    runl_por(outer_loop_count, dummy);
    auto start = std::chrono::high_resolution_clock::now();
    auto tick = runl_por(outer_loop_count, dummy);
    auto fin = std::chrono::high_resolution_clock::now();
    double second = std::chrono::duration_cast<std::chrono::microseconds>(fin - start).count() * 1e-6;
    return tick / second;
}
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)

//__rdtscが定格クロックに基づいた値を返すのを利用して、実際の動作周波数を得る
//やや時間がかかるので注意
double getCPUMaxTurboClock() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    static double turboClock = 0.0;
    if (turboClock > 0.0) {
        return turboClock;
    }
    //http://instlatx64.atw.hu/
    //によれば、Sandy/Ivy/Haswell/Silvermont
    //いずれでもサポートされているのでノーチェックでも良い気がするが...
    //固定クロックのタイマーを持つかチェック (Fn:8000_0007:EDX8)
    int CPUInfo[4] = { -1 };
    __cpuid(CPUInfo, 0x80000007);
    if (0 == (CPUInfo[3] & (1 << 8))) {
        return 0.0;
    }
    //rdtscp命令のチェック (Fn:8000_0001:EDX27)
    __cpuid(CPUInfo, 0x80000001);
    if (0 == (CPUInfo[3] & (1 << 27))) {
        return 0.0;
    }
#if defined(_WIN32) || defined(_WIN64)
    //例外が発生するなら処理を中断する
    if (!check_rdtscp_available()) {
        return 0.0;
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    const double tick_per_clock = get_tick_per_clock();
    const double tick_per_sec = get_tick_per_sec();
    turboClock = (tick_per_sec / tick_per_clock) * 1e-9;
    return turboClock;
#else
    return 0.0;
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
}

double getCPUDefaultClock() {
    return getCPUDefaultClockFromCPUName();
}

int getCPUInfo(TCHAR *buffer, size_t nSize
#if ENCODER_QSV
    , MFXVideoSession *pSession
#endif
) {
    int ret = 0;
    buffer[0] = _T('\0');
    cpu_info_t cpu_info;
    if (getCPUName(buffer, nSize) || !get_cpu_info(&cpu_info)) {
        buffer[0] = _T('\0');
        ret = 1;
    } else {
#if defined(_WIN32) || defined(_WIN64) //Linuxでは環境によっては、正常に動作しない場合がある
        const double defaultClock = getCPUDefaultClockFromCPUName();
        bool noDefaultClockInCPUName = (0.0 >= defaultClock);
        const double maxFrequency = getCPUMaxTurboClock();
        if (defaultClock > 0.0) {
            if (noDefaultClockInCPUName) {
                _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" @ %.2fGHz"), defaultClock);
            }
            //大きな違いがなければ、TurboBoostはないものとして表示しない
            if (maxFrequency / defaultClock > 1.01) {
                _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" [TB: %.2fGHz]"), maxFrequency);
            }
        } else if (maxFrequency > 0.0) {
            _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" [%.2fGHz]"), maxFrequency);
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" (%dC/%dT)"), cpu_info.physical_cores, cpu_info.logical_cores);
#if ENCODER_QSV && !FOR_AUO
        if (pSession != nullptr) {
            int cpuGen = getCPUGen(pSession);
            if (cpuGen != CPU_GEN_UNKNOWN) {
                _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" <%s>"), CPU_GEN_STR[cpuGen]);
            }
        }
#endif
    }
    return ret;
}

BOOL GetProcessTime(HANDLE hProcess, PROCESS_TIME *time) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEMTIME systime;
    GetSystemTime(&systime);
    return (NULL != hProcess
        && GetProcessTimes(hProcess, (FILETIME *)&time->creation, (FILETIME *)&time->exit, (FILETIME *)&time->kernel, (FILETIME *)&time->user)
        && (WAIT_OBJECT_0 == WaitForSingleObject(hProcess, 0) || SystemTimeToFileTime(&systime, (FILETIME *)&time->exit)));
#else //#if defined(_WIN32) || defined(_WIN64)
    struct tms tm;
    times(&tm);
    time->exit = time->creation;
    time->creation = clock();
    time->kernel = tm.tms_stime;
    time->user = tm.tms_utime;
    return 0;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

BOOL GetProcessTime(PROCESS_TIME *time) {
#if defined(_WIN32) || defined(_WIN64)
    return GetProcessTime(GetCurrentProcess(), time);
#else
    return GetProcessTime(NULL, time);
#endif
}

double GetProcessAvgCPUUsage(HANDLE hProcess, PROCESS_TIME *start) {
    PROCESS_TIME current = { 0 };
    cpu_info_t cpu_info;
    double result = 0;
    if (NULL != hProcess
        && get_cpu_info(&cpu_info)
        && GetProcessTime(hProcess, &current)) {
        uint64_t current_total_time = current.kernel + current.user;
        uint64_t start_total_time = (nullptr == start) ? 0 : start->kernel + start->user;
        result = (current_total_time - start_total_time) * 100.0 / (double)(cpu_info.logical_cores * (current.exit - ((nullptr == start) ? current.creation : start->exit)));
    }
    return result;
}

double GetProcessAvgCPUUsage(PROCESS_TIME *start) {
#if defined(_WIN32) || defined(_WIN64)
    return GetProcessAvgCPUUsage(GetCurrentProcess(), start);
#else
    return GetProcessAvgCPUUsage(NULL, start);
#endif
}

