// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#include <stdio.h>
#include <cstdint>
#include <vector>
#include <numeric>
#include <chrono>
#include <climits>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <condition_variable>

#include "cpu_info.h"
#include "rgy_simd.h"
#include "ram_speed.h"

typedef struct {
    int mode;
    uint32_t check_size_bytes;
    uint32_t thread_id;
    uint32_t physical_cores;
    double megabytes_per_sec;
} RAM_SPEED_THREAD;


typedef struct {
    std::atomic<uint32_t> check_bit;
    uint32_t check_bit_all;
} RAM_SPEED_THREAD_WAKE;

#ifdef __cplusplus
extern "C" {
#endif
extern void read_sse(uint8_t *src, uint32_t size, uint32_t count_n);
extern void read_avx(uint8_t *src, uint32_t size, uint32_t count_n);
extern void write_sse(uint8_t *dst, uint32_t size, uint32_t count_n);
extern void write_avx(uint8_t *dst, uint32_t size, uint32_t count_n);
#ifdef __cplusplus
}
#endif


typedef void(*func_ram_test)(uint8_t *dst, uint32_t size, uint32_t count_n);

void ram_speed_func(RAM_SPEED_THREAD *thread_prm, RAM_SPEED_THREAD_WAKE *thread_wk) {
    const int TEST_COUNT = 4;
    uint32_t check_size_bytes = (thread_prm->check_size_bytes + 255) & ~255;
    const uint32_t test_kilo_bytes   = (uint32_t)(((thread_prm->mode == RAM_SPEED_MODE_READ) ? 1 : 0.5) * thread_prm->physical_cores * 1024 * 1024 / (std::max)(1.0, std::log2(check_size_bytes / 1024.0)) + 0.5);
    const uint32_t warmup_kilo_bytes = test_kilo_bytes * 2;
    uint8_t *ptr = (uint8_t *)_aligned_malloc(check_size_bytes, 64);
    for (uint32_t i = 0; i < check_size_bytes; i++)
        ptr[i] = 0;
    uint32_t count_n = std::max(1, (int)(test_kilo_bytes * 1024.0 / check_size_bytes + 0.5));
    int avx = 0 != (get_availableSIMD() & AVX);
    int64_t result[TEST_COUNT];
    static const func_ram_test RAM_TEST_LIST[][2] = {
        {read_sse, write_sse},
        {read_avx, write_avx},
    };

    const func_ram_test ram_test = RAM_TEST_LIST[avx][thread_prm->mode];

    thread_wk->check_bit |= 1 << thread_prm->thread_id;
    while (thread_wk->check_bit.load() != thread_wk->check_bit_all) {
        ram_test(ptr, check_size_bytes, std::max(1, (int)(warmup_kilo_bytes * 1024.0 / check_size_bytes + 0.5)));
    }

    for (int i = 0; i < TEST_COUNT; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        ram_test(ptr, check_size_bytes, count_n);
        auto fin = std::chrono::high_resolution_clock::now();
        result[i] = std::chrono::duration_cast<std::chrono::microseconds>(fin - start).count();
    }
    ram_test(ptr, check_size_bytes, std::max(1, (int)(warmup_kilo_bytes * 1024.0 / check_size_bytes + 0.5)));
    _aligned_free(ptr);

    int64_t time_min = LLONG_MAX;
    for (int i = 0; i < TEST_COUNT; i++)
        time_min = (std::min)(time_min, result[i]);

    thread_prm->megabytes_per_sec = (check_size_bytes * (double)count_n / (1024.0 * 1024.0)) / (time_min * 0.000001);
}

int ram_speed_thread_id(int thread_index, const cpu_info_t& cpu_info) {
    return (thread_index % cpu_info.physical_cores) * (cpu_info.logical_cores / cpu_info.physical_cores) + (int)(thread_index / cpu_info.physical_cores);
}

double ram_speed_mt(int check_size_kilobytes, int mode, int thread_n) {
    std::vector<std::thread> threads(thread_n);
    std::vector<RAM_SPEED_THREAD> thread_prm(thread_n);
    RAM_SPEED_THREAD_WAKE thread_wake;
    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);

    thread_wake.check_bit = 0;
    thread_wake.check_bit_all = 0;
    for (uint32_t i = 0; i < threads.size(); i++) {
        thread_wake.check_bit_all |= 1 << ram_speed_thread_id(i, cpu_info);
    }
    for (uint32_t i = 0; i < threads.size(); i++) {
        thread_prm[i].physical_cores = cpu_info.physical_cores;
        thread_prm[i].mode = (mode == RAM_SPEED_MODE_RW) ? (i & 1) : mode;
        thread_prm[i].check_size_bytes = (check_size_kilobytes * 1024 / thread_n + 255) & ~255;
        thread_prm[i].thread_id = ram_speed_thread_id(i, cpu_info);
        threads[i] = std::thread(ram_speed_func, &thread_prm[i], &thread_wake);
        //渡されたスレッドIDからスレッドAffinityを決定
        //特定のコアにスレッドを縛り付ける
        SetThreadAffinityMask(threads[i].native_handle(), (uint64_t)1 << (int)thread_prm[i].thread_id);
        //高優先度で実行
        SetThreadPriority(threads[i].native_handle(), THREAD_PRIORITY_HIGHEST);
    }
    for (uint32_t i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    double sum = 0.0;
    for (const auto& prm : thread_prm) {
        sum += prm.megabytes_per_sec;
    }
    return sum;
}

std::vector<double> ram_speed_mt_list(int check_size_kilobytes, int mode, bool logical_core) {
    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);

    std::vector<double> results;
    for (uint32_t ith = 1; ith <= cpu_info.physical_cores; ith++) {
        results.push_back(ram_speed_mt(check_size_kilobytes, mode, ith));
    }
    if (logical_core && cpu_info.logical_cores != cpu_info.physical_cores) {
        int smt = cpu_info.logical_cores / cpu_info.physical_cores;
        for (int i_smt = 2; i_smt <= smt; i_smt++) {
            results.push_back(ram_speed_mt(check_size_kilobytes, mode, cpu_info.physical_cores * i_smt));
        }
    }
    return results;
}
