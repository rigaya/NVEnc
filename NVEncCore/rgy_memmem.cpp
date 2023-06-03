// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2023 rigaya
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
#include <cstring>
#include "rgy_simd.h"
#include "rgy_memmem.h"

size_t rgy_memmem_c(const void *data_, const size_t data_size, const void *target_, const size_t target_size) {
    const uint8_t *data = (const uint8_t *)data_;
    if (data_size < target_size) {
        return RGY_MEMMEM_NOT_FOUND;
    }
    for (size_t i = 0; i <= data_size - target_size; i++) {
        if (memcmp(data + i, target_, target_size) == 0) {
            return i;
        }
    }
    return RGY_MEMMEM_NOT_FOUND;
}

decltype(rgy_memmem_c)* get_memmem_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
#if defined(_M_X64) || defined(__x86_64)
    if ((simd & RGY_SIMD::AVX512BW) == RGY_SIMD::AVX512BW) return rgy_memmem_avx512bw;
#endif
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return rgy_memmem_avx2;
#endif
    return rgy_memmem_c;
}
