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
// -------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_SIMD_H__
#define __RGY_SIMD_H__

#include <cstdint>
#include <limits>

#ifndef _MSC_VER

#ifndef __forceinline
#define __forceinline __attribute__((always_inline))
#endif

#endif //#ifndef _MSC_VER

enum class RGY_SIMD : uint64_t {
    NONE            = 0x000000,
    SSE2            = 0x000001,
    SSE3            = 0x000002,
    SSSE3           = 0x000004,
    SSE41           = 0x000008,
    SSE42           = 0x000010,
    POPCNT          = 0x000020,
    AVX             = 0x000040,
    AVX2            = 0x000080,
    BMI1            = 0x000100,
    BMI2            = 0x000200,
    AVX512F         = 0x000400,
    AVX512DQ        = 0x000800,
    AVX512IFMA      = 0x001000,
    AVX512PF        = 0x002000,
    AVX512ER        = 0x004000,
    AVX512CD        = 0x008000,
    AVX512BW        = 0x010000,
    AVX512VL        = 0x020000,
    AVX512VBMI      = 0x040000,
    AVX512VBMI2     = 0x080000,
    AVX512VNNI      = 0x100000,
    AVX512BITALG    = 0x200000,
    AVX512VPOPCNTDQ = 0x400000,

    SIMD_ALL        = std::numeric_limits<uint64_t>::max(),
};

static bool operator!(RGY_SIMD e) {
    return e == static_cast<RGY_SIMD>(0);
}

static RGY_SIMD operator|(RGY_SIMD a, RGY_SIMD b) {
    return (RGY_SIMD)((uint64_t)a | (uint64_t)b);
}

static RGY_SIMD operator|=(RGY_SIMD &a, RGY_SIMD b) {
    a = a | b;
    return a;
}

static RGY_SIMD operator&(RGY_SIMD a, RGY_SIMD b) {
    return (RGY_SIMD)((uint64_t)a & (uint64_t)b);
}

static RGY_SIMD operator&=(RGY_SIMD &a, RGY_SIMD b) {
    a = a & b;
    return a;
}

RGY_SIMD get_availableSIMD();

#endif //__RGY_SIMD_H__
