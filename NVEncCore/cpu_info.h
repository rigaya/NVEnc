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

#ifndef _CPU_INFO_H_
#define _CPU_INFO_H_

#include <stdint.h>
#include "rgy_tchar.h"
#include "rgy_osdep.h"
#include "rgy_version.h"

static const int MAX_CACHE_LEVEL = 4;
static const int MAX_CORE_COUNT = 512;
static const int MAX_NODE_COUNT = 8;

enum class RGYCacheLevel {
    L0,
    L1,
    L2,
    L3,
    L4
};

enum class RGYCacheType {
    Unified,
    Instruction,
    Data,
    Trace
};

enum class RGYUnitType {
   Core,
   Cache,
   Node
};

enum class RGYCoreType {
    Physical,
    Logical
};

typedef struct node_info_t {
    size_t mask;
} node_info_t;

typedef struct cache_info_t {
    RGYCacheType type;
    RGYCacheLevel level;
    int associativity;
    int linesize;
    int size;
    size_t mask;
} cache_info_t;

typedef struct {
    int processor_id;   // プロセッサID
    int core_id;        // コアID
    int socket_id;      // ソケットID
    int logical_cores;  // 論理コア数
    size_t mask;        // 対応する物理コアのマスク
} processor_info_t;     // 物理コアの情報

typedef struct {
    int node_count;           // ノード数
    node_info_t nodes[MAX_NODE_COUNT];
    int physical_cores;  // 物理コア数
    int physical_cores_p; // 物理コア数
    int physical_cores_e; // 物理コア数
    int logical_cores;   // 論理コア数
    int max_cache_level; // キャッシュの最大レベル
    int cache_count[MAX_CACHE_LEVEL];       // 各階層のキャッシュの数
    cache_info_t caches[MAX_CACHE_LEVEL][MAX_CORE_COUNT]; // 各階層のキャッシュの情報
    processor_info_t proc_list[MAX_CORE_COUNT]; // 物理コアの情報
    size_t maskCoreP;  // Performanceコアのマスク
    size_t maskCoreE;  // Efficiencyコアのマスク
    size_t maskSystem; // システム全体のマスク
} cpu_info_t;


int getCPUName(char *buffer, size_t nSize);
bool get_cpu_info(cpu_info_t *cpu_info);
cpu_info_t get_cpu_info();
uint64_t get_mask(const cpu_info_t *cpu_info, RGYUnitType unit_type, int level, int id);

tstring print_cpu_info(const cpu_info_t *cpu_info);

#if ENCODER_QSV
class MFXVideoSession;
int getCPUInfo(TCHAR *buffer, size_t nSize, MFXVideoSession *pSession = nullptr);
#else
int getCPUInfo(TCHAR *buffer, size_t nSize);
#endif

template <size_t size>
int inline getCPUInfo(TCHAR(&buffer)[size]) {
    return getCPUInfo(buffer, size);
}

double getCPUDefaultClock();
double getCPUMaxTurboClock();

typedef struct PROCESS_TIME {
    uint64_t creation, exit, kernel, user;
} PROCESS_TIME;

BOOL GetProcessTime(PROCESS_TIME *time);
BOOL GetProcessTime(HANDLE hProcess, PROCESS_TIME *time);
double GetProcessAvgCPUUsage(HANDLE hProcess, PROCESS_TIME *start = nullptr);
double GetProcessAvgCPUUsage(PROCESS_TIME *start = nullptr);

#endif //_CPU_INFO_H_
