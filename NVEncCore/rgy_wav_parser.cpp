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

std::vector<uint8_t> RGYWAVHeader::createHeader() {
    std::vector<uint8_t> buffer(WAVE_HEADER_SIZE);
    auto head = buffer.data();

    static const char * const RIFF_HEADER = "RIFF";
    static const char * const WAVE_HEADER = "WAVE";
    static const char * const FMT_CHUNK = "fmt ";
    static const char * const DATA_CHUNK = "data";
    const int32_t FMT_SIZE = 16;
    const int16_t FMT_ID = 1;
    const int   size = bits_per_sample / 8;

    memcpy(head + 0, RIFF_HEADER, strlen(RIFF_HEADER));
    *(int32_t*)(head + 4) = data_size + WAVE_HEADER_SIZE - 8;
    memcpy(head +  8, WAVE_HEADER, strlen(WAVE_HEADER));
    memcpy(head + 12, FMT_CHUNK, strlen(FMT_CHUNK));
    *(int32_t*)(head + 16) = FMT_SIZE;
    *(int16_t*)(head + 20) = FMT_ID;
    *(int16_t*)(head + 22) = (int16_t)number_of_channels;
    *(int32_t*)(head + 24) = sample_rate;
    *(int32_t*)(head + 28) = sample_rate * number_of_channels * size;
    *(int16_t*)(head + 32) = (int16_t)(size * number_of_channels);
    *(int16_t*)(head + 34) = (int16_t)(size * 8);
    memcpy(head + 36, DATA_CHUNK, strlen(DATA_CHUNK));
    *(int32_t*)(head + 40) = data_size;
    //計44byte(WAVE_HEADER_SIZE)
    return buffer;
}
