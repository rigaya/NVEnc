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

#include <string>
#include "rgy_osdep.h"
#include "rgy_wav_parser.h"

static inline uint32_t read_u32(const uint8_t* data) {
    return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
}

static inline uint16_t read_u16(const uint8_t* data) {
    return data[0] | (data[1] << 8);
}

uint32_t RGYWAVHeader::parseHeader(const uint8_t *data) {
    const uint8_t *data_ptr = data;

    strncpy_s(file_id, (const char *)data_ptr, 4);
    data_ptr += 4;

    file_size = read_u32(data_ptr);
    data_ptr += 4;

    strncpy_s(format, (const char *)data_ptr, 4);
    data_ptr += 4;

    strncpy_s(subchunk_id, (const char *)data_ptr, 4);
    data_ptr += 4;

    subchunk_size = read_u32(data_ptr);
    data_ptr += 4;

    audio_format = read_u16(data_ptr);
    data_ptr += 2;

    number_of_channels = read_u16(data_ptr);
    data_ptr += 2;

    sample_rate = read_u32(data_ptr);
    data_ptr += 4;

    byte_rate = read_u32(data_ptr);
    data_ptr += 4;

    block_align = read_u16(data_ptr);
    data_ptr += 2;

    bits_per_sample = read_u16(data_ptr);
    data_ptr += 2;

    strncpy_s(data_id, (const char *)data_ptr, 4);
    data_ptr += 4;

    data_size = read_u32(data_ptr);
    data_ptr += 4;

    return (uint32_t)(data_ptr - data);
}
