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
#define RGY_MEMMEM_AVX2
#include "rgy_memmem.h"

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)

std::vector<nal_info> parse_nal_unit_h264_avx2(const uint8_t * data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const auto next = rgy_memmem_avx2_imp((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next == RGY_MEMMEM_NOT_FOUND) break;

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
    _mm256_zeroupper();
    return nal_list;
}

std::vector<nal_info> parse_nal_unit_hevc_avx2(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const auto next = rgy_memmem_avx2_imp((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next == RGY_MEMMEM_NOT_FOUND) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = (data[i + 3] & 0x7f) >> 1;
            nal_start.nuh_layer_id = ((data[i+3] & 1) << 5) | ((data[i+4] & 0xf8) >> 3);
            nal_start.temporal_id = (data[i+4] & 0x07) - 1;
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
    _mm256_zeroupper();
    return nal_list;
}

size_t find_header_avx2(const uint8_t *data, size_t size) {
    return rgy_memmem_avx2_imp(data, size, DOVIRpu::rpu_header, sizeof(DOVIRpu::rpu_header));
}

#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
