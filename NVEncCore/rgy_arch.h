// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#pragma once
#ifndef __RGY_ARCH_H__
#define __RGY_ARCH_H__

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#include <xmmintrin.h>
static inline void rgy_yield() {
    _mm_pause();
}
#if !(defined(_WIN32) || defined(_WIN64))
static inline void __cpuid(int cpuInfo[4], int param) {
    int eax = 0, ebx = 0, ecx = 0, edx = 0;
     __asm("xor %%ecx, %%ecx\n\t"
           "cpuid" : "=a"(eax), "=b" (ebx), "=c"(ecx), "=d"(edx)
                   : "0"(param));
    cpuInfo[0] = eax;
    cpuInfo[1] = ebx;
    cpuInfo[2] = ecx;
    cpuInfo[3] = edx;
}

#ifndef _MSC_VER
static inline unsigned long long rgy_xgetbv(unsigned int index) {
  unsigned int eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((unsigned long long)edx << 32) | eax;
}
#endif

#if NO_RDTSCP_INTRIN
static inline uint64_t __rdtscp(uint32_t *Aux) {
    uint32_t aux;
    uint64_t rax,rdx;
    asm volatile ( "rdtscp\n" : "=a" (rax), "=d" (rdx), "=c" (aux) : : );
    *Aux = aux;
    return (rdx << 32) + rax;
}
#endif //#if NO_RDTSCP_INTRIN

//uint64_t __rdtsc() {
//    unsigned int eax, edx;
//    __asm__ volatile("rdtsc" : "=a"(eax), "=d"(edx));
//    return ((uint64_t)edx << 32) | eax;
//}
#endif //#if !(defined(_WIN32) || defined(_WIN64))

#elif (defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__) || defined(__ARM_ARCH))

static inline void rgy_yield() {
    __asm__ __volatile__("isb\n");
}

#endif

#endif //__RGY_ARCH_H__
