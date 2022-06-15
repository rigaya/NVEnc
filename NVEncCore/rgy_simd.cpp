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
#include "rgy_arch.h"
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#if _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif //_MSC_VER

RGY_SIMD get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    RGY_SIMD simd = RGY_SIMD::NONE;
    if (CPUInfo[3] & 0x04000000) simd |= RGY_SIMD::SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= RGY_SIMD::SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= RGY_SIMD::SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= RGY_SIMD::SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= RGY_SIMD::SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= RGY_SIMD::POPCNT;
    uint64_t xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
#if _MSC_VER
        xgetbv = _xgetbv(0);
#else
        xgetbv = rgy_xgetbv(0);
#endif
        if ((xgetbv & 0x06) == 0x06)
            simd |= RGY_SIMD::AVX;
    }
    __cpuid(CPUInfo, 7);
    if (!!(simd & RGY_SIMD::AVX) && (CPUInfo[1] & 0x00000020)) {
        simd |= RGY_SIMD::AVX2;
    }
    if (!!(simd & RGY_SIMD::AVX) && ((xgetbv >> 5) & 7) == 7) {
        if (CPUInfo[1] & (1u <<  3)) simd |= RGY_SIMD::BMI1;
        if (CPUInfo[1] & (1u <<  8)) simd |= RGY_SIMD::BMI2;
        if (CPUInfo[1] & (1u << 16)) simd |= RGY_SIMD::AVX512F;
        if (!!(simd & RGY_SIMD::AVX512F)) {
            if (CPUInfo[1] & (1u << 17)) simd |= RGY_SIMD::AVX512DQ;
            if (CPUInfo[1] & (1u << 21)) simd |= RGY_SIMD::AVX512IFMA;
            if (CPUInfo[1] & (1u << 26)) simd |= RGY_SIMD::AVX512PF;
            if (CPUInfo[1] & (1u << 27)) simd |= RGY_SIMD::AVX512ER;
            if (CPUInfo[1] & (1u << 28)) simd |= RGY_SIMD::AVX512CD;
            if (CPUInfo[1] & (1u << 30)) simd |= RGY_SIMD::AVX512BW;
            if (CPUInfo[1] & (1u << 31)) simd |= RGY_SIMD::AVX512VL;
            if (CPUInfo[2] & (1u <<  1)) simd |= RGY_SIMD::AVX512VBMI;
            if (CPUInfo[2] & (1u <<  6)) simd |= RGY_SIMD::AVX512VBMI2;
            if (CPUInfo[2] & (1u << 11)) simd |= RGY_SIMD::AVX512VNNI;
            if (CPUInfo[2] & (1u << 12)) simd |= RGY_SIMD::AVX512BITALG;
            if (CPUInfo[2] & (1u << 14)) simd |= RGY_SIMD::AVX512VPOPCNTDQ;
        }
    }
    return simd;
}
#else
RGY_SIMD get_availableSIMD() {
    return RGY_SIMD::NONE;
}
#endif
