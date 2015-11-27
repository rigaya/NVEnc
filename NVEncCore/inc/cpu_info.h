//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _CPU_INFO_H_
#define _CPU_INFO_H_

#include <stdint.h>
#include <tchar.h>
#include <Windows.h>

typedef struct cache_info_t {
    uint32_t count;
    uint32_t level;
    uint32_t associativity;
    uint32_t linesize;
    uint32_t type;
    uint32_t size;
} cache_info_t;

typedef struct {
    uint32_t nodes;
    uint32_t physical_cores;
    uint32_t logical_cores;
    uint32_t max_cache_level;
    cache_info_t caches[4];
} cpu_info_t;


bool get_cpu_info(cpu_info_t *cpu_info);

int getCPUInfo(TCHAR *buffer, size_t nSize);

template <size_t size>
int inline getCPUInfo(TCHAR(&buffer)[size]) {
    return getCPUInfo(buffer, size);
}

double getCPUDefaultClock();
double getCPUMaxTurboClock(unsigned int num_thread);

typedef struct PROCESS_TIME {
    uint64_t creation, exit, kernel, user;
} PROCESS_TIME;

BOOL GetProcessTime(PROCESS_TIME *time);
BOOL GetProcessTime(HANDLE hProcess, PROCESS_TIME *time);
double GetProcessAvgCPUUsage(HANDLE hProcess, PROCESS_TIME *start = nullptr);
double GetProcessAvgCPUUsage(PROCESS_TIME *start = nullptr);

#endif //_CPU_INFO_H_
