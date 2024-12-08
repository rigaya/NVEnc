// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc/rkmppenc by rigaya
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
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <map>
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


#pragma warning(push)
#pragma warning(disable: 4127) //warning C4127: 条件式が定数です。
static inline int CountSetBits(size_t bits_) {
    if (sizeof(size_t) > 4) {
        uint64_t bits = (uint64_t)bits_;
        bits = (bits & 0x5555555555555555) + (bits >>  1 & 0x5555555555555555);
        bits = (bits & 0x3333333333333333) + (bits >>  2 & 0x3333333333333333);
        bits = (bits & 0x0f0f0f0f0f0f0f0f) + (bits >>  4 & 0x0f0f0f0f0f0f0f0f);
        bits = (bits & 0x00ff00ff00ff00ff) + (bits >>  8 & 0x00ff00ff00ff00ff);
        bits = (bits & 0x0000ffff0000ffff) + (bits >> 16 & 0x0000ffff0000ffff);
        bits = (bits & 0x00000000ffffffff) + (bits >> 32 & 0x00000000ffffffff);
        return (int)bits;
    } else {
        uint32_t bits = (uint32_t)bits_;
        bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
        bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
        bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
        bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
        bits = (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
        return (int)bits;
    }
}
#pragma warning(pop)

#if (defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__) || defined(__ARM_ARCH))

std::map<uint32_t, uint64_t> getCPUPartARM() {
    std::map<int, uint64_t> cpu_architecture;
    std::map<uint32_t, uint64_t> cpu_variant;
    std::map<uint32_t, uint64_t> cpu_part;
    
    std::ifstream inputFile("/proc/cpuinfo");
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string script_data = std::string(data_begin, data_end);
    inputFile.close();

    int processorID = -1;

    for (auto line : split(script_data, "\n")) {
        auto pos = line.find("processor");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " %d", &i)) {
                processorID = i;
            }
            continue;
        }
        pos = line.find("CPU architecture");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " %d", &i)) {
                if (cpu_architecture.count(i) == 0) {
                    cpu_architecture[i] = 0;
                }
                cpu_architecture[i] |= (1llu << processorID);
            }
            continue;
        }
        pos = line.find("CPU variant");
        if (pos != std::string::npos) {
            uint32_t i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " 0x%x", &i)) {
                if (cpu_variant.count(i) == 0) {
                    cpu_variant[i] = 0;
                }
                cpu_variant[i] |= (1llu << processorID);
            }
            continue;
        }
        pos = line.find("CPU part");
        if (pos != std::string::npos) {
            uint32_t i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " 0x%x", &i)) {
                if (cpu_part.count(i) == 0) {
                    cpu_part[i] = 0;
                }
                cpu_part[i] |= (1llu << processorID);
            }
            continue;
        }
    }
    return cpu_part;
}

std::string getCPUNameARM() {
    std::unordered_map<int, int> cpu_architecture;
    std::unordered_map<uint32_t, int> cpu_variant;
    std::unordered_map<uint32_t, int> cpu_part;
    
    std::ifstream inputFile("/proc/cpuinfo");
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string script_data = std::string(data_begin, data_end);
    inputFile.close();

    for (auto line : split(script_data, "\n")) {
        auto pos = line.find("CPU architecture");
        if (pos != std::string::npos) {
            int i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " %d", &i)) {
                if (cpu_architecture.count(i) == 0) {
                    cpu_architecture[i] = 1;
                } else {
                    cpu_architecture[i]++;
                }
            }
            continue;
        }
        pos = line.find("CPU variant");
        if (pos != std::string::npos) {
            uint32_t i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " 0x%x", &i)) {
                if (cpu_variant.count(i) == 0) {
                    cpu_variant[i] = 1;
                } else {
                    cpu_variant[i]++;
                }
            }
            continue;
        }
        pos = line.find("CPU part");
        if (pos != std::string::npos) {
            uint32_t i = 0;
            if (1 == sscanf(line.substr(line.find(":") + 1).c_str(), " 0x%x", &i)) {
                if (cpu_part.count(i) == 0) {
                    cpu_part[i] = 1;
                } else {
                    cpu_part[i]++;
                }
            }
            continue;
        }
    }

    std::string name;
    if (cpu_part.size() > 0) {
        for (auto& [part, count] : cpu_part) {
            //https://en.wikipedia.org/wiki/Comparison_of_ARM_processors#ARMv8-A
            const char *part_name = nullptr;
            switch (part) {
                case 0xD01: part_name = "Cortex-A32"; break;
                case 0xD02: part_name = "Cortex-A34"; break;
                case 0xD03: part_name = "Cortex-A53"; break;
                case 0xD04: part_name = "Cortex-A35"; break;
                case 0xD05: part_name = "Cortex-A55"; break;
                case 0xD06: part_name = "Cortex-A65"; break;
                case 0xD07: part_name = "Cortex-A57"; break;
                case 0xD08: part_name = "Cortex-A72"; break;
                case 0xD09: part_name = "Cortex-A73"; break;
                case 0xD0A: part_name = "Cortex-A75"; break;
                case 0xD0B: part_name = "Cortex-A76"; break;
                case 0xD0D: part_name = "Cortex-A77"; break;
                case 0xD0E: part_name = "Cortex-A76AE"; break;
                case 0xD41: part_name = "Cortex-A78"; break;
                case 0xD43: part_name = "Cortex-A65AE"; break;
                case 0xD44: part_name = "Cortex-X1"; break;
            }
            if (part_name) {
                if (name.length() > 0) {
                    name += " + ";
                }
                name += strsprintf("%sx%d", part_name, count);
            }
        }
    }
    return name;
}
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
#if (defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__) || defined(__ARM_ARCH))
    if (name.length() == 0) {
        name = getCPUNameARM();
    }
#endif
    sprintf(buffer, "%s %s", name.c_str(), arch.c_str());
    return 0;
#endif
}

bool getCPUHybridMasks(cpu_info_t *info) {
    info->maskSystem = 0;
    info->maskCoreP = 0;
    info->maskCoreE = 0;
#if _MSC_VER
    DWORD_PTR maskProcess = 0;
    DWORD_PTR maskSysAff = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &maskProcess, &maskSysAff) == 0) {
        info->maskSystem = 0;
        info->maskCoreP = 0;
        info->maskCoreE = 0;
        for (uint64_t i = 0; i < info->logical_cores; i++) {
            info->maskSystem |= (1llu << i);
        }
        return false;
    }
    info->maskSystem = maskSysAff;
    const auto threadCount = CountSetBits(info->maskSystem);
#else
    const auto threadCount = info->physical_cores;
#endif
#if defined(__x86__) || defined(__x86_64__) || defined(_M_X86) || defined(_M_IX86) || defined(_M_X64)
    const auto hThread = GetCurrentThread();
    size_t maskOriginal = 0;
    for (int ith = 0; ith < threadCount; ith++) {
        const auto maskTarget = (size_t)1u << ith;
        auto maskPrev = SetThreadAffinityMask(hThread, maskTarget);
        if (maskOriginal == 0) {
            maskOriginal = maskPrev;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(0));
        int CPUInfo[4] = { 0 };
        __cpuid(CPUInfo, 0x1A);
        const auto hybridInfo = CPUInfo[0 /*EAX*/] >> 24;
        if (hybridInfo == 0x20) {
            info->maskCoreE |= maskTarget;
        } else if (hybridInfo == 0x40) {
            info->maskCoreP |= maskTarget;
        }
    }
    SetThreadAffinityMask(hThread, maskOriginal); // 元に戻す

    info->physical_cores_e = 0;
    info->physical_cores_p = 0;
    for (int i = 0; i < info->physical_cores; i++) {
        const auto maskTarget = info->proc_list[i].mask;
        if (info->maskCoreP & maskTarget) {
            info->physical_cores_p++;
        } else if (info->maskCoreE & maskTarget) {
            info->physical_cores_e++;
        }
    }
#endif //#if defined(__x86__) || defined(__x86_64__) || defined(_M_X86) || defined(_M_X64)
#if (defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__) || defined(__ARM_ARCH))
    const auto cpu_part = getCPUPartARM();
    if (cpu_part.size() > 1) {
        for (auto it = cpu_part.begin(); it != cpu_part.end(); it++) {
            if (it == cpu_part.begin()) { //partが一番小さいのがEcore
                info->maskCoreE |= it->second;
            } else {
                info->maskCoreP |= it->second;
            }
        }
    }

    info->physical_cores_e = 0;
    info->physical_cores_p = 0;
    for (int i = 0; i < info->physical_cores; i++) {
        const auto maskTarget = info->proc_list[i].mask;
        if (info->maskCoreP & maskTarget) {
            info->physical_cores_p++;
        } else if (info->maskCoreE & maskTarget) {
            info->physical_cores_e++;
        }
    }
#endif
    return true;
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

bool get_cpu_info(cpu_info_t *cpu_info) {
    if (cpu_info == nullptr)
        return false;

    static cpu_info_t s_cpu_info = { 0 };
    if (s_cpu_info.physical_cores > 0) {
        *cpu_info = s_cpu_info;
        return true;
    }

    s_cpu_info = cpu_info_t({ 0 });

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
            s_cpu_info.nodes[s_cpu_info.node_count++].mask = ptr->ProcessorMask;
            break;
        case RelationProcessorCore: {
            auto& proc = s_cpu_info.proc_list[s_cpu_info.physical_cores];
            proc.core_id = s_cpu_info.physical_cores;
            proc.processor_id = s_cpu_info.physical_cores;
            proc.logical_cores = CountSetBits(ptr->ProcessorMask);
            proc.mask = ptr->ProcessorMask;
            // A hyperthreaded core supplies more than one logical processor.
            s_cpu_info.logical_cores += proc.logical_cores;
            s_cpu_info.physical_cores++;
        } break;
        case RelationCache:
        {
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
            PCACHE_DESCRIPTOR Cache = &ptr->Cache;
            if (1 <= Cache->Level && Cache->Level <= _countof(s_cpu_info.cache_count)) {
                const int cacheIdx = s_cpu_info.cache_count[Cache->Level - 1]++;
                cache_info_t *cache = &s_cpu_info.caches[Cache->Level-1][cacheIdx];
                cache->type = (RGYCacheType)Cache->Type;
                cache->level = (RGYCacheLevel)Cache->Level;
                cache->linesize = Cache->LineSize;
                cache->size = Cache->Size;
                cache->associativity = Cache->Associativity;
                cache->mask = ptr->ProcessorMask;
                s_cpu_info.max_cache_level = (std::max)(s_cpu_info.max_cache_level, (int)cache->level);
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

    getCPUHybridMasks(&s_cpu_info);
    *cpu_info = s_cpu_info;
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

    //ここまでで論理コアの情報を作った
    //cpu_infoに登録するのは物理コアの情報なので、整理しなおす
    //いったんsocket→core→processorの順でソート
    std::sort(processor_list.begin(), processor_list.end(), [](const processor_info_t& a, const processor_info_t& b) {
        if (a.socket_id != b.socket_id) return a.socket_id < b.socket_id;
        if (a.core_id != b.core_id) return a.core_id < b.core_id;
        return a.processor_id < b.processor_id;
    });

    cpu_info->max_cache_level = 0;
    cpu_info->physical_cores = 0;
    cpu_info->logical_cores = processor_list.size();

    processor_info_t *prevCore = nullptr;
    for (size_t ip = 0; ip < processor_list.size(); ip++) {
        if (prevCore != nullptr
            && prevCore->socket_id == processor_list[ip].socket_id
            && prevCore->core_id   == processor_list[ip].core_id) {
            // 同じソケットの同じコアならそれは論理コア
            prevCore->logical_cores++;
            prevCore->mask |= 1llu << processor_list[ip].processor_id;
        } else {
            auto targetCore = &cpu_info->proc_list[cpu_info->physical_cores];
            *targetCore = processor_list[ip];
            targetCore->logical_cores = 1;
            targetCore->mask = 1llu << processor_list[ip].processor_id;
            cpu_info->physical_cores++;
            prevCore = targetCore;
        }
    }

    //キャッシュの情報を作る
    std::vector<cache_info_t> caches;
    for (int ip = 0; ip < cpu_info->physical_cores; ip++) {
        const auto& targetCore = &cpu_info->proc_list[ip];
        uint64_t mask = 0;
        for (int index = 0; ; index++) {
            cache_info_t cacheinfo;

            char buffer[256];
            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d", targetCore->processor_id, index);
            struct stat st;
            if (stat(buffer, &st) != 0) break;

            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", targetCore->processor_id, index);
            FILE *fp = fopen(buffer, "r");
            if (fp) {
                while (fgets(buffer, _countof(buffer), fp) != NULL) {
                    for (auto numstr : split(buffer, ",")) {
                        int value0 = 0, value1 = 0;
                        if (sscanf_s(numstr.c_str(), "%d-%d", &value0, &value1) == 2) {
                            for (int iv = value0; iv <= value1; iv++) {
                                mask |= 1llu << iv;
                            }
                        } else if (sscanf_s(numstr.c_str(), "%d", &value0) == 1) {
                            mask |= 1llu << value0;
                        }
                    }
                }
                fclose(fp);
            }
            cacheinfo.mask = mask;

            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d/level", targetCore->processor_id, index);
            fp = fopen(buffer, "r");
            if (fp) {
                while (fgets(buffer, _countof(buffer), fp) != NULL) {
                    int value = 0;
                    if (sscanf_s(buffer, "%d", &value) == 1) {
                        cacheinfo.level = (RGYCacheLevel)value;
                    }
                }
                fclose(fp);
            }

            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d/size", targetCore->processor_id, index);
            fp = fopen(buffer, "r");
            if (fp) {
                while (fgets(buffer, _countof(buffer), fp) != NULL) {
                    int value = 0;
                    if (sscanf_s(buffer, "%dK", &value) == 1) {
                        cacheinfo.size = value * 1024;
                    } else if (sscanf_s(buffer, "%dM", &value) == 1) {
                        cacheinfo.size = value * 1024 * 1024;
                    } else if (sscanf_s(buffer, "%dG", &value) == 1) {
                        cacheinfo.size = value * 1024 * 1024 * 1024;
                    } else if (sscanf_s(buffer, "%d", &value) == 1) {
                        cacheinfo.size = value;
                    }
                }
                fclose(fp);
            }

            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d/ways_of_associativity", targetCore->processor_id, index);
            fp = fopen(buffer, "r");
            if (fp) {
                while (fgets(buffer, _countof(buffer), fp) != NULL) {
                    int value = 0;
                    if (sscanf_s(buffer, "%d", &value) == 1) {
                        cacheinfo.associativity = value;
                    }
                }
                fclose(fp);
            }

            sprintf_s(buffer, "/sys/devices/system/cpu/cpu%d/cache/index%d/type", targetCore->processor_id, index);
            fp = fopen(buffer, "r");
            if (fp) {
                while (fgets(buffer, _countof(buffer), fp) != NULL) {
                    if (strncasecmp(buffer, "Instruction", strlen("Instruction")) == 0) {
                        cacheinfo.type = RGYCacheType::Instruction;
                        break;
                    } else if (strncasecmp(buffer, "Data", strlen("Data")) == 0) {
                        cacheinfo.type = RGYCacheType::Data;
                        break;
                    } else if (strncasecmp(buffer, "Unified", strlen("Unified")) == 0) {
                        cacheinfo.type = RGYCacheType::Unified;
                        break;
                    }
                }
                fclose(fp);
            }

            auto sameCache = std::find_if(caches.begin(), caches.end(), [&cacheinfo](const cache_info_t& c){
                return cacheinfo.type == c.type
                    && cacheinfo.level == c.level
                    && ((cacheinfo.mask & c.mask) != 0);
            });
            if (sameCache != caches.end()) {
                sameCache->mask |= cacheinfo.mask;
            } else {
                caches.push_back(cacheinfo);
            }
        }
    }

    for (int ilevel = 0; ilevel < MAX_CACHE_LEVEL; ilevel++) {
        cpu_info->cache_count[ilevel] = 0;
    }
    for (const auto& c : caches) {
        const int ilevel = (int)c.level - 1;
        const int icacheidx = cpu_info->cache_count[ilevel]++;
        cpu_info->caches[ilevel][icacheidx] = c;
    }
    for (int ilevel = 0; ilevel < MAX_CACHE_LEVEL; ilevel++) {
        if (cpu_info->cache_count[ilevel] > 0) {
            cpu_info->max_cache_level = ilevel+1;
        }
    }

    //ノードの情報を作る
    cpu_info->node_count = processor_list.back().socket_id + 1;
    //初期化
    for (int in = 0; in < cpu_info->node_count; in++) {
        cpu_info->nodes[in].mask = 0;
    }
    for (int ip = 0; ip < cpu_info->physical_cores; ip++) {
        auto& targetCore = cpu_info->proc_list[ip];
        cpu_info->nodes[targetCore.socket_id].mask |= targetCore.mask;
    }

    getCPUHybridMasks(cpu_info);
    return true;
}
#endif //#if defined(_WIN32) || defined(_WIN64)


const processor_info_t *get_core_info(const cpu_info_t *cpu_info, RGYCoreType type, int id) {
    switch (type) {
    case RGYCoreType::Physical: return (id < cpu_info->physical_cores) ? &cpu_info->proc_list[id] : nullptr;
    case RGYCoreType::Logical: {
        const uint64_t mask = 1llu << id;
        for (int i = 0; i < cpu_info->physical_cores; i++) {
            if (mask & cpu_info->proc_list[i].mask) {
                return &cpu_info->proc_list[i];
            }
        }
        return nullptr;
    }
    default: return nullptr;
    }
}
uint64_t get_core_mask(const cpu_info_t *cpu_info, RGYCoreType type, int id) {
    auto ptr = get_core_info(cpu_info, type, id);
    return (ptr) ? ptr->mask : 0;
}
const cache_info_t *get_cache_info(const cpu_info_t *cpu_info, RGYCacheLevel level, int id) {
    if (RGYCacheLevel::L1 <= level && level <= RGYCacheLevel::L4) {
        return (id < cpu_info->cache_count[(int)level - 1]) ? &cpu_info->caches[(int)level - 1][id] : nullptr;
    }
    return nullptr;
}
uint64_t get_cache_mask(const cpu_info_t *cpu_info, RGYCacheLevel level, int id) {
    auto ptr = get_cache_info(cpu_info, level, id);
    return (ptr) ? ptr->mask : 0;
}
uint64_t get_mask(const cpu_info_t *cpu_info, RGYUnitType unit_type, int level, int id) {
    switch (unit_type) {
    case RGYUnitType::Core:  return get_core_mask(cpu_info, (RGYCoreType)level, id);
    case RGYUnitType::Cache: return get_cache_mask(cpu_info, (RGYCacheLevel)level, id);
    case RGYUnitType::Node:  return cpu_info->nodes[id].mask;
    default: return 0;
    }
}

cpu_info_t get_cpu_info() {
    cpu_info_t cpu;
    get_cpu_info(&cpu);
    return cpu;
}

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
const int TEST_COUNT = 4000;
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
    const int outer_loop_count = 20;
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
        _tcscpy_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" ("));
        if (cpu_info.maskCoreP != 0 && cpu_info.maskCoreE != 0 && cpu_info.physical_cores <= 64) {
            _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T("%dP+%dE,"), cpu_info.physical_cores_p, cpu_info.physical_cores_e);
        }
        _stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T("%dC/%dT)"), cpu_info.physical_cores, cpu_info.logical_cores);
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

const TCHAR *RGYCacheTypeToStr(RGYCacheType type) {
    switch (type) {
    case RGYCacheType::Unified:     return _T(" ");
    case RGYCacheType::Instruction: return _T("I");
    case RGYCacheType::Data:        return _T("D");
    default:                        return _T("-");
    }
}

tstring print_cpu_info(const cpu_info_t *cpu_info) {
    TCHAR buffer[256];
    getCPUInfo(buffer, _countof(buffer));

    tstring str = buffer;
    str += _T("\n");
    str += _T("CPU cores\n");
    for (int ip = 0; ip < cpu_info->physical_cores; ip++) {
        auto& targetCore = cpu_info->proc_list[ip];
        str += strsprintf(_T("  core %2d "), ip);
        if ((cpu_info->maskCoreP & targetCore.mask) == targetCore.mask) {
            str += _T("P");
        } else if ((cpu_info->maskCoreE & targetCore.mask) == targetCore.mask) {
            str += _T("E");
        } else {
            str += _T(" ");
        }
        str += _T(" : ");
        for (int il = 0; il < cpu_info->logical_cores; il++) {
            const auto mask = 1llu << il;
            str += (mask & targetCore.mask) ? _T("*") : _T("-");
        }
        str += _T("\n");
    }

    if (cpu_info->cache_count[0] > 0) {
        str += _T("CPU caches\n");
        for (int icache_level = 0; icache_level < MAX_CACHE_LEVEL; icache_level++) {
            for (int ic = 0; ic < cpu_info->cache_count[icache_level]; ic++) {
                auto& targetCache = cpu_info->caches[icache_level][ic];
                str += strsprintf(_T("  cache L%d%s : "), icache_level + 1, RGYCacheTypeToStr(targetCache.type));
                for (int il = 0; il < cpu_info->logical_cores; il++) {
                    const auto mask = 1llu << il;
                    str += (mask & targetCache.mask) ? _T("*") : _T("-");
                }
                str += strsprintf(_T(" : %2dway %6dKB\n"), targetCache.associativity, targetCache.size / 1024);
            }
        }
    }
    return str;
}
