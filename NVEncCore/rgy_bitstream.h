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
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <string>
#include "rgy_def.h"
#include "rgy_util.h"

struct nal_info {
    const uint8_t *ptr;
    uint8_t type;
    size_t size;
};

struct unit_info {
    uint8_t type;
    std::vector<uint8_t> unit_data;
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
    NALU_HEVC_FILLER      = 38,
    NALU_HEVC_PREFIX_SEI  = 39,
    NALU_HEVC_SUFFIX_SEI  = 40,
    NALU_HEVC_UNSPECIFIED = 62,
    NALU_HEVC_INVALID     = 64,

    OBU_SEQUENCE_HEADER        = 1,
    OBU_TEMPORAL_DELIMITER     = 2,
    OBU_FRAME_HEADER           = 3,
    OBU_TILE_GROUP             = 4,
    OBU_METADATA               = 5,
    OBU_FRAME                  = 6,
    OBU_REDUNDANT_FRAME_HEADER = 7,
    OBU_TILE_LIST              = 8,
    OBU_PADDING                = 15,

    AV1_METADATA_TYPE_AOM_RESERVED_0 = 0,
    AV1_METADATA_TYPE_HDR_CLL        = 1,
    AV1_METADATA_TYPE_HDR_MDCV       = 2,
    AV1_METADATA_TYPE_SCALABILITY    = 3,
    AV1_METADATA_TYPE_ITUT_T35       = 4,
    AV1_METADATA_TYPE_TIMECODE       = 5,
    AV1_METADATA_TYPE_FRAME_SIZE     = 6,
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
void to_nal(std::vector<uint8_t>& data);
void add_u16(std::vector<uint8_t>& data, uint16_t u16);
void add_u32(std::vector<uint8_t>& data, uint32_t u32);

std::vector<nal_info> parse_nal_unit_h264_c(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_c(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_h264_avx2(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_avx2(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_h264_avx512bw(const uint8_t *data, size_t size);
std::vector<nal_info> parse_nal_unit_hevc_avx512bw(const uint8_t *data, size_t size);

decltype(parse_nal_unit_h264_c)* get_parse_nal_unit_h264_func();
decltype(parse_nal_unit_hevc_c)* get_parse_nal_unit_hevc_func();

size_t find_header_c(const uint8_t *data, size_t size);
size_t find_header_avx2(const uint8_t *data, size_t size);
size_t find_header_avx512bw(const uint8_t *data, size_t size);

decltype(find_header_c)* get_find_header_func();

std::deque<std::unique_ptr<unit_info>> parse_unit_av1(const uint8_t *data, const size_t size);

uint8_t gen_obu_header(const uint8_t obu_type);
size_t get_av1_uleb_size_bytes(uint64_t value);
std::vector<uint8_t> get_av1_uleb_size_data(uint64_t value);
std::vector<uint8_t> gen_av1_obu_metadata(const uint8_t metadata_type, const std::vector<uint8_t>& metadata);

struct RGYHDRMetadataPrm {
    int maxcll;
    int maxfall;
    bool contentlight_set;
    rgy_rational<int> masterdisplay[10]; //G,B,R,WP,L(max,min)
    bool masterdisplay_set;
    CspTransfer atcSei;
public:
    RGYHDRMetadataPrm();
    bool hasPrmSet() const;
};

class RGYHDRMetadata {
private:
    RGYHDRMetadataPrm prm;

public:
    RGYHDRMetadata();

    void set_maxcll(int maxcll, int maxfall);
    int parse_maxcll(std::string maxcll);
    void set_masterdisplay(const rgy_rational<int> *masterdisplay);
    int parse_masterdisplay(std::string masterdisplay);
    void set_atcsei(CspTransfer atcSei);
    RGYHDRMetadataPrm getprm() const;
    std::string print_masterdisplay() const;
    std::string print_maxcll() const;
    std::string print_atcsei() const;
    std::string print() const;
    std::vector<uint8_t> gen_nal() const;
    std::vector<uint8_t> gen_nal(RGYHDRMetadataPrm prm);

    std::vector<uint8_t> gen_masterdisplay_obu() const;
    std::vector<uint8_t> gen_maxcll_obu() const;
    std::vector<uint8_t> gen_obu() const;
private:
    std::vector<uint8_t> raw_maxcll() const;
    std::vector<uint8_t> raw_masterdisplay(const bool forAV1) const;
    std::vector<uint8_t> raw_atcsei() const;

    std::vector<uint8_t> sei_maxcll() const;
    std::vector<uint8_t> sei_masterdisplay() const;
    std::vector<uint8_t> sei_atcsei() const;
};

struct DOVIProfile {
    int profile;

    bool HRDSEI;
    bool videoSignalTypeDescript;
    bool aud;

    VideoVUIInfo vui;
};

const DOVIProfile *getDOVIProfile(const int id);

class DOVIRpu {
public:
    static const uint8_t rpu_header[4];

    DOVIRpu();
    ~DOVIRpu();
    int init(const TCHAR *rpu_file);
    int get_next_rpu_nal(std::vector<uint8_t>& bytes, const int64_t id);
    const tstring& get_filepath() const;

protected:
    int fillBuffer();
    int get_next_rpu(std::vector<uint8_t>& bytes);
    int get_next_rpu(std::vector<uint8_t>& bytes, const int64_t id);

    decltype(find_header_c)* m_find_header;
    tstring m_filepath;
    std::unique_ptr<FILE, decltype(&fclose)> m_fp;
    std::vector<uint8_t> m_buffer;
    int64_t m_datasize;
    int64_t m_dataoffset;
    int64_t m_count;

    std::unordered_map<int64_t, std::vector<uint8_t>> m_rpus;
};

#endif //__RGY_BITSTREAM_H__
