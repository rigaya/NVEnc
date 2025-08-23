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

#include <vector>
#include <array>
#include "rgy_bitstream_aac.h"

int RGYAACHeader::sampleRateIdxToRate(const uint32_t idx) {
    static const int samplerateList[] = {
        96000,
        88200,
        64000,
        48000,
        44100,
        32000,
        24000,
        22050,
        16000,
        12000,
        11025,
        8000,
        7350,
        0
    };
    return samplerateList[std::min<uint32_t>(idx, (sizeof(samplerateList) / sizeof(samplerateList[0])) - 1)];
}

bool RGYAACHeader::is_adts_sync(const uint16_t *ptr) {
    return ptr[0] == 0xfff0;
}

bool RGYAACHeader::is_valid(const uint8_t *buf, const size_t size) {
    RGYAACHeader aacHeader;
    return aacHeader.parse(buf, size) == 0 && aacHeader.aac_frame_length == size;
}

int RGYAACHeader::parse(const uint8_t *buf, const size_t size) {
    if (size < RGYAACHeader::HEADER_BYTE_SIZE) {
        return 1;
    }
    const uint8_t buf0 = buf[0];
    const uint8_t buf1 = buf[1];
    if (buf0 != 0xff || (buf1 & 0xf0) != 0xf0) {
        return 1;
    }
    const uint8_t buf2 = buf[2];
    const uint8_t buf3 = buf[3];
    const uint8_t buf4 = buf[4];
    const uint8_t buf5 = buf[5];
    const uint8_t buf6 = buf[6];
    id = (buf1 & 0x08) != 0;
    protection = (buf1 & 0x01) != 0;
    profile = (buf2 & 0xC0) >> 6;
    samplerate = sampleRateIdxToRate((buf2 & 0x3C) >> 2);
    private_bit = (buf2 & 0x02) >> 1;
    channel = ((buf2 & 0x01) << 2) | ((buf3 & 0xC0) >> 6);
    original = (buf3 & 0x20) != 0;
    home = (buf3 & 0x10) != 0;
    copyright = (buf3 & 0x08) != 0;
    copyright_start = (buf3 & 0x04) != 0;
    aac_frame_length = ((buf3 & 0x03) << 11) | (buf4 << 3) | (buf5 >> 5);
    adts_buffer_fullness = ((buf5 & 0x1f) << 6) | (buf6 >> 2);
    no_raw_data_blocks_in_frame = buf6 & 0x03;
    return (samplerate != 0) && (channel != 0) ? 0 : 1;
}
