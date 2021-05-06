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

#include <cstdint>
#include "rgy_osdep.h"
#include "rgy_simd.h"
#if defined(_M_IX86) || defined(_M_X64)
#if _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif //_MSC_VER

uint32_t get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    uint32_t simd = NONE;
    if (CPUInfo[3] & 0x04000000) simd |= SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
    uint64_t xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        xgetbv = _xgetbv(0);
        if ((xgetbv & 0x06) == 0x06)
            simd |= AVX;
    }
    __cpuid(CPUInfo, 7);
    if ((simd & AVX) && (CPUInfo[1] & 0x00000020)) {
        simd |= AVX2;
    }
    if ((simd & AVX) && ((xgetbv >> 5) & 7) == 7) {
        if (CPUInfo[1] & (1u << 16)) simd |= AVX512F;
        if (simd & AVX512F) {
            if (CPUInfo[1] & (1u << 17)) simd |= AVX512DQ;
            if (CPUInfo[1] & (1u << 21)) simd |= AVX512IFMA;
            if (CPUInfo[1] & (1u << 26)) simd |= AVX512PF;
            if (CPUInfo[1] & (1u << 27)) simd |= AVX512ER;
            if (CPUInfo[1] & (1u << 28)) simd |= AVX512CD;
            if (CPUInfo[1] & (1u << 30)) simd |= AVX512BW;
            if (CPUInfo[1] & (1u << 31)) simd |= AVX512VL;
            if (CPUInfo[2] & (1u <<  1)) simd |= AVX512VBMI;
        }
    }
    return simd;
}
#else
uint32_t get_availableSIMD() {
    return NONE;
}
#endif
