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
#include "rgy_def.h"

struct nal_info {
    const uint8_t *ptr;
    uint8_t type;
    size_t size;
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

enum PayloadType {
    BUFFERING_PERIOD                     = 0,
    PICTURE_TIMING                       = 1,
    PAN_SCAN_RECT                        = 2,
    FILLER_PAYLOAD                       = 3,
    USER_DATA_REGISTERED_ITU_T_T35       = 4,
    USER_DATA_UNREGISTERED               = 5,
    RECOVERY_POINT                       = 6,
    SCENE_INFO                           = 9,
    PICTURE_SNAPSHOT                     = 15,
    PROGRESSIVE_REFINEMENT_SEGMENT_START = 16,
    PROGRESSIVE_REFINEMENT_SEGMENT_END   = 17,
    FILM_GRAIN_CHARACTERISTICS           = 19,
    POST_FILTER_HINT                     = 22,
    TONE_MAPPING_INFO                    = 23,
    FRAME_PACKING                        = 45,
    DISPLAY_ORIENTATION                  = 47,
    GREEN_METADATA                       = 56,
    SOP_DESCRIPTION                      = 128,
    ACTIVE_PARAMETER_SETS                = 129,
    DECODING_UNIT_INFO                   = 130,
    TEMPORAL_LEVEL0_INDEX                = 131,
    DECODED_PICTURE_HASH                 = 132,
    SCALABLE_NESTING                     = 133,
    REGION_REFRESH_INFO                  = 134,
    NO_DISPLAY                           = 135,
    TIME_CODE                            = 136,
    MASTERING_DISPLAY_COLOUR_VOLUME      = 137,
    SEGM_RECT_FRAME_PACKING              = 138,
    TEMP_MOTION_CONSTRAINED_TILE_SETS    = 139,
    CHROMA_RESAMPLING_FILTER_HINT        = 140,
    KNEE_FUNCTION_INFO                   = 141,
    COLOUR_REMAPPING_INFO                = 142,
    DEINTERLACE_FIELD_IDENTIFICATION     = 143,
    CONTENT_LIGHT_LEVEL_INFO             = 144,
    DEPENDENT_RAP_INDICATION             = 145,
    CODED_REGION_COMPLETION              = 146,
    ALTERNATIVE_TRANSFER_CHARACTERISTICS = 147,
    AMBIENT_VIEWING_ENVIRONMENT          = 148,
    CONTENT_COLOUR_VOLUME                = 149,
    EQUIRECTANGULAR_PROJECTION           = 150,
    SPHERE_ROTATION                      = 154,
    OMNI_VIEWPORT                        = 156,
    CUBEMAP_PROJECTION                   = 151,
    REGION_WISE_PACKING                  = 155,
    REGIONAL_NESTING                     = 157,
};

std::vector<uint8_t> unnal(const uint8_t *ptr, size_t len);

std::vector<nal_info> parse_nal_unit_h264_c(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_c(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_h264_avx2(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_avx2(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_h264_avx512bw(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_avx512bw(const uint8_t *data, size_t size);

decltype(parse_nal_unit_h264_c)* get_parse_nal_unit_h264_func();
decltype(parse_nal_unit_hevc_c)* get_parse_nal_unit_hevc_func();

struct HEVCHDRSeiPrm {
    int maxcll;
    int maxfall;
    bool contentlight_set;
    int masterdisplay[10];
    bool masterdisplay_set;
    CspTransfer atcSei;
public:
    HEVCHDRSeiPrm();
};

class HEVCHDRSei {
private:
    HEVCHDRSeiPrm prm;

public:
    HEVCHDRSei();

    void set_maxcll(int maxcll, int maxfall);
    int parse_maxcll(std::string maxcll);
    void set_masterdisplay(const int masterdisplay[10]);
    int parse_masterdisplay(std::string masterdisplay);
    void set_atcsei(CspTransfer atcSei);
    HEVCHDRSeiPrm getprm() const;
    std::string print_masterdisplay() const;
    std::string print_maxcll() const;
    std::string print_atcsei() const;
    std::string print() const;
    std::vector<uint8_t> gen_nal() const;
    std::vector<uint8_t> gen_nal(HEVCHDRSeiPrm prm);
private:
    std::vector<uint8_t> sei_maxcll() const;
    std::vector<uint8_t> sei_masterdisplay() const;
    std::vector<uint8_t> sei_atcsei() const;
    void to_nal(std::vector<uint8_t>& data) const;
    void add_u16(std::vector<uint8_t>& data, uint16_t u16) const;
    void add_u32(std::vector<uint8_t>& data, uint32_t u32) const;
};

#endif //__RGY_BITSTREAM_H__
