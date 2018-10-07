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

#pragma once
#ifndef __RGY_BITSTREAM_H__
#define __RGY_BITSTREAM_H__

#include <vector>
#include <cstdint>
#include <string>

struct nal_info {
    const uint8_t *ptr;
    uint8_t type;
    uint32_t size;
};

enum : uint8_t {
    NALU_H264_UNDEF    = 0,
    NALU_H264_NONIDR   = 1,
    NALU_H264_SLICEA   = 2,
    NALU_H264_SLICEB   = 3,
    NALU_H264_SLICEC   = 4,
    NALU_H264_IDR      = 5,
    NALU_H264_SEI      = 6,
    NALU_H264_SPS      = 7,
    NALU_H264_PPS      = 8,
    NALU_H264_AUD      = 9,
    NALU_H264_EOSEQ    = 10,
    NALU_H264_EOSTREAM = 11,
    NALU_H264_FILLER   = 12,
    NALU_H264_SPSEXT   = 13,
    NALU_H264_PREFIX   = 14,
    NALU_H264_SUBSPS   = 15,

    NALU_HEVC_UNDEF    = 0,
    NALU_HEVC_VPS      = 32,
    NALU_HEVC_SPS      = 33,
    NALU_HEVC_PPS      = 34,
    NALU_HEVC_AUD      = 35,
    NALU_HEVC_EOS      = 36,
    NALU_HEVC_EOB      = 37,
    NALU_HEVC_FILLER     = 38,
    NALU_HEVC_PREFIX_SEI = 39,
    NALU_HEVC_SUFFIX_SEI = 40,
};

static std::vector<nal_info> parse_nal_unit_h264(const uint8_t *data, uint32_t size) {
    std::vector<nal_info> nal_list;
    nal_info nal_start = { nullptr, 0, 0 };
    const int i_fin = size - 3;
    for (int i = 0; i < i_fin; i++) {
        if (data[i+0] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i-1] == 0);
            nal_start.type = data[i+3] & 0x1f;
            nal_start.size = (int)(data + size - nal_start.ptr);
            if (nal_list.size()) {
                auto prev = nal_list.end()-1;
                prev->size = (int)(nal_start.ptr - prev->ptr);
            }
            i += 3;
        }
    }
    if (nal_start.ptr) {
        nal_list.push_back(nal_start);
    }
    return nal_list;
}

static std::vector<nal_info> parse_nal_unit_hevc(const uint8_t *data, uint32_t size) {
    std::vector<nal_info> nal_list;
    nal_info nal_start = { nullptr, 0, 0 };
    const int i_fin = size - 3;

    for (int i = 0; i < i_fin; i++) {
        if (data[i+0] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i-1] == 0);
            nal_start.type = (data[i+3] & 0x7f) >> 1;
            nal_start.size = (int)(data + size - nal_start.ptr);
            if (nal_list.size()) {
                auto prev = nal_list.end()-1;
                prev->size = (int)(nal_start.ptr - prev->ptr);
            }
            i += 3;
        }
    }
    if (nal_start.ptr) {
        nal_list.push_back(nal_start);
    }
    return nal_list;
}

struct HEVCHDRSeiPrm {
    int maxcll;
    int maxfall;
    int masterdisplay[10];
    bool masterdisplay_set;
public:
    HEVCHDRSeiPrm();
};

class HEVCHDRSei {
private:
    HEVCHDRSeiPrm prm;

public:
    HEVCHDRSei();

    int parse(std::string maxcll, std::string masterdisplay);
    HEVCHDRSeiPrm getprm() const;
    std::vector<uint8_t> gen_nal() const;
    std::vector<uint8_t> gen_nal(HEVCHDRSeiPrm prm);
private:
    std::vector<uint8_t> sei_maxcll() const;
    std::vector<uint8_t> sei_masterdisplay() const;
    void to_nal(std::vector<uint8_t>& data) const;
    void add_u16(std::vector<uint8_t>& data, uint16_t u16) const;
    void add_u32(std::vector<uint8_t>& data, uint32_t u32) const;
};

#endif //__RGY_BITSTREAM_H__
