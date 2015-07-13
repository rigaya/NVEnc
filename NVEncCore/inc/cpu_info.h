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

#include <Windows.h>
#include <tchar.h>

BOOL getProcessorCount(DWORD *physical_processor_core, DWORD *logical_processor_core);

int getCPUInfo(TCHAR *buffer, size_t nSize);

template <size_t size>
int inline getCPUInfo(TCHAR(&buffer)[size]) {
    return getCPUInfo(buffer, size);
}

double getCPUDefaultClock();
double getCPUMaxTurboClock(unsigned int num_thread);

typedef struct PROCESS_TIME {
    UINT64 creation, exit, kernel, user;
} PROCESS_TIME;

BOOL GetProcessTime(HANDLE hProcess, PROCESS_TIME *time);
double GetProcessAvgCPUUsage(HANDLE hProcess, PROCESS_TIME *start = nullptr);

#endif //_CPU_INFO_H_
