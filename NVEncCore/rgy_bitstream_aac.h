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

#ifndef __RGY_BITSTREAM_AAC_H__
#define __RGY_BITSTREAM_AAC_H__

#include <cstdint>

static const int AAC_HEADER_MIN_SIZE = 7;
static const uint32_t AAC_BLOCK_SAMPLES = 1024;

struct RGYAACHeader {
    static const int HEADER_BYTE_SIZE = 7;
    bool id;
    bool protection;
    int profile;     // 00 ... main, 01 ... lc, 10 ... ssr
    int samplerate;
    bool private_bit;
    uint32_t channel;
    bool original;
    bool home;
    bool copyright;
    bool copyright_start;
    uint32_t aac_frame_length; // AACヘッダを含む
    int adts_buffer_fullness;
    int no_raw_data_blocks_in_frame;

    static bool is_adts_sync(const uint16_t *ptr);
    static bool is_valid(const uint8_t *buf, const size_t size);
    int parse(const uint8_t *buf, const size_t size = RGYAACHeader::HEADER_BYTE_SIZE);
    int sampleRateIdxToRate(const uint32_t idx);
};

#endif //__RGY_BITSTREAM_AAC_H__