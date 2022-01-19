// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#include "rgy_bitstream.h"

#if defined(_M_X64) || defined(__x86_64)

#include <immintrin.h>

#if _MSC_VER >= 1800 && !defined(__AVX512BW__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX512 for this file.");
#endif

#define CLEAR_LEFT_BIT(x) ((x) & ((x) - 1))

#if defined(_WIN32) || defined(_WIN64)
#define CTZ32(x) _tzcnt_u32(x)
#define CTZ64(x) _tzcnt_u64(x)
#else
#define CTZ32(x) __builtin_ctz(x)
#define CTZ64(x) __builtin_ctzll(x)
#endif

static RGY_FORCEINLINE __m512i _mm512_loadu_si512_exact(const uint8_t *const data, const uint8_t *const data_fin) {
    alignas(64) static const uint8_t inctable[] = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };
    const __m512i inc = _mm512_load_si512((const __m512i*)inctable);
    const uint8_t remain_size = (uint8_t)std::min<decltype(data_fin - data)>(data_fin - data, 64);
    const auto mask = _mm512_cmplt_epi8_mask(inc, _mm512_set1_epi8(remain_size));
    return _mm512_maskz_loadu_epi8(mask, (const __m512i*)data);
}

static RGY_FORCEINLINE int64_t memmem_avx512(const void *data_, const int64_t data_size, const void *target_, const int64_t target_size) {
    uint8_t *data = (uint8_t *)data_;
    const uint8_t *target = (const uint8_t *)target_;
    const __m512i target_first = _mm512_set1_epi8(target[0]);
    const __m512i target_last = _mm512_set1_epi8(target[target_size - 1]);
    const int64_t fin = data_size - target_size + 1 - 64; // r1の64byteロードが安全に行える限界

    //まずは単純なロードで行えるところまでループ
    int64_t i = 0;
    for (; i < fin; i += 64) {
        const __m512i r0 = _mm512_loadu_si512((const __m512i*)(data + i));
        const __m512i r1 = _mm512_loadu_si512((const __m512i*)(data + i + target_size - 1));
        uint64_t mask = _mm512_mask_cmpeq_epi8_mask(_mm512_cmpeq_epi8_mask(r0, target_first), r1, target_last);
        while (mask != 0) {
            const int64_t j = (int64_t)CTZ64(mask);
            if (memcmp(data + i + j + 1, target + 1, target_size - 2) == 0) {
                const auto ret = i + j;
                return ret;
            }
            mask = CLEAR_LEFT_BIT(mask);
        }
    }
    //ロード範囲をmaskで考慮しながらロード
    uint8_t *data_fin = data + data_size;
    for (; i < data_size; i += 64) {
        const __m512i r0 = _mm512_loadu_si512_exact(data + i, data_fin);
        const __m512i r1 = _mm512_loadu_si512_exact(data + i + target_size - 1, data_fin);
        uint64_t mask = _mm512_mask_cmpeq_epi8_mask(_mm512_cmpeq_epi8_mask(r0, target_first), r1, target_last);
        while (mask != 0) {
            const int64_t j = (int64_t)CTZ64(mask);
            if (memcmp(data + i + j + 1, target + 1, target_size - 2) == 0) {
                const auto ret = i + j;
                return ret;
            }
            mask = CLEAR_LEFT_BIT(mask);
        }
    }
    return -1;
}

std::vector<nal_info> parse_nal_unit_h264_avx512bw(const uint8_t * data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const int64_t next = memmem_avx512((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next < 0) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = data[i + 3] & 0x1f;
            nal_start.size = data + size - nal_start.ptr;
            if (nal_list.size()) {
                auto prev = nal_list.end() - 1;
                prev->size = nal_start.ptr - prev->ptr;
            }
            i += 3;
        }
        if (nal_start.ptr) {
            nal_list.push_back(nal_start);
        }
    }
    return nal_list;
}

std::vector<nal_info> parse_nal_unit_hevc_avx512bw(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const int64_t next = memmem_avx512((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next < 0) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = (data[i + 3] & 0x7f) >> 1;
            nal_start.size = data + size - nal_start.ptr;
            if (nal_list.size()) {
                auto prev = nal_list.end() - 1;
                prev->size = nal_start.ptr - prev->ptr;
            }
            i += 3;
        }
        if (nal_start.ptr) {
            nal_list.push_back(nal_start);
        }
    }
    return nal_list;
}

int64_t find_header_avx512bw(const uint8_t *data, size_t size) {
    return memmem_avx512(data, size, DOVIRpu::rpu_header, sizeof(DOVIRpu::rpu_header));
}

#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
