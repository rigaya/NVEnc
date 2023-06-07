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

#ifndef __RGY_WAV_PARSER_H__
#define __RGY_WAV_PARSER_H__

#include <cstdint>
#include <vector>

static const uint32_t WAVE_HEADER_SIZE = 44;

struct RGYWAVHeader {
    char file_id[5]; // "RIFF"
    uint32_t file_size;
    char format[5]; // "WAVE"
    char subchunk_id[5]; // "fmt "
    uint32_t subchunk_size; // 16 for PCM
    uint16_t audio_format; // PCM = 1
    uint16_t number_of_channels;
    uint32_t sample_rate;
    uint32_t byte_rate; // sample_rate * number of channels * bits per sample / 8
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_id[5]; //"data"
    uint32_t data_size; // samples * number of channels * bits per sample / 8 (Actual number of bytes)

    uint32_t parseHeader(const uint8_t *data);
    std::vector<uint8_t> createHeader();
};

#endif //__RGY_WAV_PARSER_H__
