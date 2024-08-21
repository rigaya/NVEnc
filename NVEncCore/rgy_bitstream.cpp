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

#include <regex>
#include "rgy_util.h"
#include "rgy_bitstream.h"
#include "rgy_memmem.h"

std::vector<uint8_t> unnal(const uint8_t *ptr, size_t len) {
    std::vector<uint8_t> data;
    data.reserve(len);
    data.push_back(ptr[0]);
    data.push_back(ptr[1]);
    for (size_t i = 2; i < len; i++) {
        if (ptr[i-2] == 0x00 && ptr[i-1] == 0x00 && ptr[i] == 0x03) {
            //skip
        } else {
            data.push_back(ptr[i]);
        }
    }
    return data;
}

void to_nal(std::vector<uint8_t>& data) {
    for (auto it = data.begin(); it < data.end() - 2; it++) {
        if (*it == 0
            && *(it + 1) == 0
            && (*(it + 2) & (~(0x03))) == 0) {
            it = data.insert(it + 2, 0x03);
        }
    }
}

void add_u16(std::vector<uint8_t>& data, uint16_t u16) {
    data.push_back((uint8_t)((u16 & 0xff00) >> 8));
    data.push_back((uint8_t)(u16 & 0x00ff));
}

void add_u32(std::vector<uint8_t>& data, uint32_t u32) {
    data.push_back((uint8_t)((u32 & 0xff000000) >> 24));
    data.push_back((uint8_t)((u32 & 0x00ff0000) >> 16));
    data.push_back((uint8_t)((u32 & 0x0000ff00) >>  8));
    data.push_back((uint8_t)((u32 & 0x000000ff) >>  0));
}

int get_hevc_sei_size(size_t& size, const uint8_t *ptr) {
    const auto orig_ptr = ptr;
    size = 0;
    while (ptr[0] == 0xff) {
        size += 0xff;
        ptr++;
    }
    size += ptr[0];
    return (int)(ptr - orig_ptr);
}

std::vector<uint8_t> gen_hevc_alpha_channel_info_sei(const int mode) {
    // 下記資料に基づいて生成する
    // https://developer.apple.com/av-foundation/HEVC-Video-with-Alpha-Interoperability-Profile.pdf
    std::vector<uint8_t> header = { 0x00, 0x00, 0x00, 0x01 };
    std::vector<uint8_t> buf;
    uint16_t u16 = 0x00;
    u16 |= (NALU_HEVC_PREFIX_SEI << 9) | 1;
    add_u16(buf, u16);
    buf.push_back(ALPHA_CHANNEL_INFO);
    buf.push_back(4); // size
    buf.push_back((mode & 0x07) << 4);
    buf.push_back(0);
    buf.push_back(0x7f);
    buf.push_back(0x90);
    to_nal(buf);

    std::vector<uint8_t> nalbuf;
    vector_cat(nalbuf, header);
    vector_cat(nalbuf, buf);
    return nalbuf;
}

uint8_t gen_obu_header(const uint8_t obu_type) {
    const uint8_t extension_flag = 0;
    const uint8_t has_size_flag = 1;
    return (obu_type << 3)
        | (extension_flag << 2)
        | (has_size_flag << 1);
}

size_t get_av1_uleb_size_bytes(uint64_t value) {
    size_t size = 0;
    do {
        size++;
    } while ((value >>= 7) != 0);
    return size;
}

std::vector<uint8_t> get_av1_uleb_size_data(uint64_t value) {
    std::vector<uint8_t> buffer(get_av1_uleb_size_bytes(value));
    for (size_t i = 0; i < buffer.size(); i++) {
        uint8_t byte = value & 0x7f;
        value >>= 7;
        if (value != 0) {
            byte |= 0x80; // 続きがある
        }
        buffer[i] = byte;
    }
    return buffer;
}

std::vector<uint8_t> gen_av1_obu_metadata(const uint8_t metadata_type, const std::vector<uint8_t>& metadata) {
    if (metadata.size() == 0) {
        return metadata;
    }
    std::vector<uint8_t> metadata_buf;
    metadata_buf.reserve(128);
    const uint8_t obu_header = gen_obu_header(OBU_METADATA);
    const size_t payload_size = sizeof(metadata_type) + metadata.size() + 1 /*last 0x80*/;
    metadata_buf.push_back(obu_header);
    vector_cat(metadata_buf, get_av1_uleb_size_data(payload_size));
    metadata_buf.push_back(metadata_type);
    vector_cat(metadata_buf, metadata);
    metadata_buf.push_back(0x80);
    return metadata_buf;
}

RGYHDRMetadataPrm::RGYHDRMetadataPrm() : maxcll(-1), maxfall(-1), contentlight_set(false), masterdisplay(), masterdisplay_set(false), atcSei(RGY_TRANSFER_UNKNOWN) {
    memset(&masterdisplay, 0, sizeof(masterdisplay));
}

bool RGYHDRMetadataPrm::hasPrmSet() const {
    return contentlight_set
        || masterdisplay_set
        || atcSei != RGY_TRANSFER_UNKNOWN;
}

RGYHDRMetadata::RGYHDRMetadata() : prm() {
}

void RGYHDRMetadata::set_maxcll(int maxcll, int maxfall) {
    prm.maxcll = maxcll;
    prm.maxfall = maxfall;
    prm.contentlight_set = true;
}

int RGYHDRMetadata::parse_maxcll(std::string str_maxcll) {
    if (str_maxcll.length()) {
        std::regex re_maxcll(R"((\d+),(\d+))");
        std::smatch match_maxcll;
        if (!regex_search(str_maxcll, match_maxcll, re_maxcll) || match_maxcll.size() != 3) {
            return 1;
        }

        try {
            prm.maxcll = std::stoi(match_maxcll[1]);
            prm.maxfall = std::stoi(match_maxcll[2]);
            prm.contentlight_set = true;
        } catch (...) {
            return 1;
        }
    }
    return 0;
}

void RGYHDRMetadata::set_masterdisplay(const rgy_rational<int> *masterdisplay) {
    for (int i = 0; i < 10; i++) {
        prm.masterdisplay[i] = masterdisplay[i];
    }
    prm.masterdisplay_set = true;
}

int RGYHDRMetadata::parse_masterdisplay(std::string str_masterdisplay) {
    if (str_masterdisplay.length()) {
        std::regex re_masterdisplay(R"(G\((\d+),(\d+)\)B\((\d+),(\d+)\)R\((\d+),(\d+)\)WP\((\d+),(\d+)\)L\((\d+),(\d+)\))");
        std::smatch match_masterdisplay;
        if (!regex_search(str_masterdisplay, match_masterdisplay, re_masterdisplay) || match_masterdisplay.size() != 11) {
            return 1;
        }

        try {
            for (int i = 0; i < 8; i++) {
                prm.masterdisplay[i] = rgy_rational<int>(std::stoi(match_masterdisplay[i + 1]), 50000);
            }
            for (int i = 8; i < 10; i++) {
                prm.masterdisplay[i] = rgy_rational<int>(std::stoi(match_masterdisplay[i + 1]), 10000);
            }
            prm.masterdisplay_set = true;
        } catch (...) {
            return 1;
        }
    }
    return 0;
}

void RGYHDRMetadata::set_atcsei(CspTransfer atcSei) {
    prm.atcSei = atcSei;
}

RGYHDRMetadataPrm RGYHDRMetadata::getprm() const {
    return prm;
}
std::string RGYHDRMetadata::print_masterdisplay() const {
    std::string str;
    if (prm.masterdisplay_set) {
        str += strsprintf("G(%f %f) B(%f %f) R(%f %f) WP(%f %f) L(%f %f)",
            prm.masterdisplay[0].qfloat(),
            prm.masterdisplay[1].qfloat(),
            prm.masterdisplay[2].qfloat(),
            prm.masterdisplay[3].qfloat(),
            prm.masterdisplay[4].qfloat(),
            prm.masterdisplay[5].qfloat(),
            prm.masterdisplay[6].qfloat(),
            prm.masterdisplay[7].qfloat(),
            prm.masterdisplay[8].qfloat(),
            prm.masterdisplay[9].qfloat());
    }
    return str;
}

std::string RGYHDRMetadata::print_maxcll() const {
    std::string str;
    if (prm.contentlight_set && prm.maxcll >= 0 && prm.maxfall >= 0) {
        str += strsprintf("%d/%d", prm.maxcll, prm.maxfall);
    }
    return str;
}

std::string RGYHDRMetadata::print_atcsei() const {
    std::string str;
    if (prm.atcSei != RGY_TRANSFER_UNKNOWN) {
        str += tchar_to_string(get_cx_desc(list_transfer, prm.atcSei));
    }
    return str;
}

std::string RGYHDRMetadata::print() const {
    std::string str = print_masterdisplay();
    std::string str1 = print_maxcll();
    std::string str2 = print_atcsei();
    if (str.length() > 0) {
        str = "Mastering Display: " + str + "\n";
    }
    if (str1.length() > 0) {
        str += "MaxCLL/MaxFALL: " + str1 + "\n";
    }
    if (str2.length() > 0) {
        str += "AtcSei: " + str2 + "\n";
    }
    return str;
}

std::vector<uint8_t> RGYHDRMetadata::gen_nal(RGYHDRMetadataPrm prm_set) {
    prm = prm_set;
    return gen_nal();
}

std::vector<uint8_t> RGYHDRMetadata::gen_nal() const {
    std::vector<uint8_t> data;
    data.reserve(128);

    auto data_maxcll = sei_maxcll();
    auto data_masterdisplay = sei_masterdisplay();
    auto data_atcsei = sei_atcsei();
    if (data_maxcll.size() == 0
        && data_masterdisplay.size() == 0
        && data_atcsei.size() == 0) {
        return data;
    }
    std::vector<uint8_t> header = { 0x00, 0x00, 0x00, 0x01 };
#if 0
    header.reserve(128);

    uint16_t u16 = 0x00;
    u16 |= (39 << 9) | 1;
    add_u16(data, u16);

    vector_cat(data, data_maxcll);
    vector_cat(data, data_masterdisplay);
    to_nal(data);

    vector_cat(header, data);
    header.push_back(0x80);
    return header;
#else
    if (data_maxcll.size() > 0) {
        std::vector<uint8_t> buf;
        uint16_t u16 = 0x00;
        u16 |= (NALU_HEVC_PREFIX_SEI << 9) | 1;
        add_u16(buf, u16);
        vector_cat(buf, data_maxcll);
        to_nal(buf);

        std::vector<uint8_t> nal_maxcll;
        nal_maxcll.reserve(128);
        vector_cat(nal_maxcll, header);
        vector_cat(nal_maxcll, buf);
        nal_maxcll.push_back(0x80);

        vector_cat(data, nal_maxcll);
    }

    if (data_masterdisplay.size() > 0) {
        std::vector<uint8_t> buf;
        uint16_t u16 = 0x00;
        u16 |= (NALU_HEVC_PREFIX_SEI << 9) | 1;
        add_u16(buf, u16);
        vector_cat(buf, data_masterdisplay);
        to_nal(buf);

        std::vector<uint8_t> nal_masterdisplay;
        nal_masterdisplay.reserve(128);
        vector_cat(nal_masterdisplay, header);
        vector_cat(nal_masterdisplay, buf);
        nal_masterdisplay.push_back(0x80);

        vector_cat(data, nal_masterdisplay);
    }

    if (data_atcsei.size() > 0) {
        std::vector<uint8_t> buf;
        uint16_t u16 = 0x00;
        u16 |= (NALU_HEVC_PREFIX_SEI << 9) | 1;
        add_u16(buf, u16);
        vector_cat(buf, data_atcsei);
        to_nal(buf);

        std::vector<uint8_t> nal_atcsei;
        nal_atcsei.reserve(128);
        vector_cat(nal_atcsei, header);
        vector_cat(nal_atcsei, buf);
        nal_atcsei.push_back(0x80);

        vector_cat(data, nal_atcsei);
    }
#endif
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::raw_maxcll() const {
    std::vector<uint8_t> data;
    add_u16(data, (uint16_t)prm.maxcll);
    add_u16(data, (uint16_t)prm.maxfall);
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::raw_masterdisplay(const bool forAV1) const {
    std::vector<uint8_t> data;
    if (forAV1) {
        const double ratio = (double)(1 << 16);
        add_u16(data, (uint16_t)(prm.masterdisplay[4].qdouble() * ratio + 0.5)); //R
        add_u16(data, (uint16_t)(prm.masterdisplay[5].qdouble() * ratio + 0.5)); //R
        add_u16(data, (uint16_t)(prm.masterdisplay[0].qdouble() * ratio + 0.5)); //G
        add_u16(data, (uint16_t)(prm.masterdisplay[1].qdouble() * ratio + 0.5)); //G
        add_u16(data, (uint16_t)(prm.masterdisplay[2].qdouble() * ratio + 0.5)); //B
        add_u16(data, (uint16_t)(prm.masterdisplay[3].qdouble() * ratio + 0.5)); //B
        add_u16(data, (uint16_t)(prm.masterdisplay[6].qdouble() * ratio + 0.5));
        add_u16(data, (uint16_t)(prm.masterdisplay[7].qdouble() * ratio + 0.5));
    } else {
        const double ratio = 50000;
        for (int i = 0; i < 8; i++) {
            add_u16(data, (uint16_t)(prm.masterdisplay[i].qdouble() * ratio + 0.5));
        }
    }
    const double lumaMinRatio = forAV1 ? 16384 : 10000;
    const double lumaMaxRatio = forAV1 ?   256 : 10000;
    add_u32(data, (uint32_t)(prm.masterdisplay[8].qdouble() * lumaMaxRatio + 0.5));
    add_u32(data, (uint32_t)(prm.masterdisplay[9].qdouble() * lumaMinRatio + 0.5));
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::raw_atcsei() const {
    std::vector<uint8_t> data;
    data.push_back((uint8_t)prm.atcSei);
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::sei_maxcll() const {
    std::vector<uint8_t> data;
    data.reserve(256);
    if (prm.contentlight_set && prm.maxcll >= 0 && prm.maxfall >= 0) {
        const auto maxcll = raw_maxcll();
        assert(maxcll.size() == 4);

        data.push_back(CONTENT_LIGHT_LEVEL_INFO);
        data.push_back((uint8_t)maxcll.size());
        vector_cat(data, maxcll);
    }
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::sei_masterdisplay() const {
    std::vector<uint8_t> data;
    data.reserve(256);
    if (prm.masterdisplay_set) {
        const auto masterdisplay = raw_masterdisplay(false);
        assert(masterdisplay.size() == 24);

        data.push_back(MASTERING_DISPLAY_COLOUR_VOLUME);
        data.push_back((uint8_t)masterdisplay.size());
        vector_cat(data, raw_masterdisplay(false));
    }
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::sei_atcsei() const {
    std::vector<uint8_t> data;
    data.reserve(8);
    if (prm.atcSei != RGY_TRANSFER_UNKNOWN) {
        const auto atcsei = raw_atcsei();
        assert(atcsei.size() == 1);

        data.push_back(ALTERNATIVE_TRANSFER_CHARACTERISTICS);
        data.push_back((uint8_t)atcsei.size());
        vector_cat(data, atcsei);
    }
    return data;
}

std::vector<uint8_t> RGYHDRMetadata::gen_maxcll_obu() const {
    if (prm.contentlight_set && prm.maxcll >= 0 && prm.maxfall >= 0) {
        return gen_av1_obu_metadata(AV1_METADATA_TYPE_HDR_CLL, raw_maxcll());
    }
    return {};
}

std::vector<uint8_t> RGYHDRMetadata::gen_masterdisplay_obu() const {
    if (prm.masterdisplay_set) {
        return gen_av1_obu_metadata(AV1_METADATA_TYPE_HDR_MDCV, raw_masterdisplay(true));
    }
    return {};
}

std::vector<uint8_t> RGYHDRMetadata::gen_obu() const {
    std::vector<uint8_t> data;
    data.reserve(128);
    vector_cat(data, gen_masterdisplay_obu());
    vector_cat(data, gen_maxcll_obu());
    return data;
}

DOVIRpu::DOVIRpu() : m_find_header(get_find_header_func()), m_filepath(), m_fp(nullptr, fp_deleter()), m_buffer(), m_datasize(0), m_dataoffset(0), m_count(0), m_rpus() {};
DOVIRpu::~DOVIRpu() { m_fp.reset(); };

const uint8_t DOVIRpu::rpu_header[4] = { 0, 0, 0, 1 };

const tstring& DOVIRpu::get_filepath() const {
    return m_filepath;
}

int DOVIRpu::init(const TCHAR *rpu_file) {
    m_filepath.clear();
    FILE *fp = NULL;
    if (_tfopen_s(&fp, rpu_file, _T("rb")) != 0) {
        return 1;
    }
    m_fp.reset(fp);
    m_filepath = rpu_file;

    m_buffer.resize(256 * 1024);
    return 0;
}

int DOVIRpu::fillBuffer() {
    int64_t bufRemain = m_buffer.size() - (m_dataoffset + m_datasize);
    if (bufRemain < 4) {
        if (m_dataoffset > 4) {
            memmove(m_buffer.data(), m_buffer.data() + m_dataoffset, m_datasize);
            m_dataoffset = 0;
        } else {
            m_buffer.resize(m_buffer.size() * 2);
        }
        bufRemain = m_buffer.size() - (m_dataoffset + m_datasize);
    }
    const auto bytes_read = (int)fread(m_buffer.data() + m_dataoffset + m_datasize, sizeof(uint8_t), bufRemain, m_fp.get());
    m_datasize += bytes_read;
    return bytes_read;
}

int DOVIRpu::get_next_rpu(std::vector<uint8_t>& bytes) {
    if (m_datasize <= 4) {
        if (fillBuffer() == 0) {
            return 1; //EOF
        }
    }
    if (memcmp(m_buffer.data() + m_dataoffset, &DOVIRpu::rpu_header, sizeof(DOVIRpu::rpu_header)) != 0) {
        return 1;
    }
    m_dataoffset += sizeof(DOVIRpu::rpu_header);
    m_datasize -= sizeof(DOVIRpu::rpu_header);

    int64_t next_size = 0;
    for (;;) {
        auto dataptr = m_buffer.data() + m_dataoffset;
        const auto pos = m_find_header(dataptr, m_datasize);
        if (pos != RGY_MEMMEM_NOT_FOUND) {
            const auto next_header = dataptr + pos;
            next_size = next_header - dataptr;
            break;
        }
        if (fillBuffer() == 0) { // EOF
            next_size = m_datasize;
            break;
        }
    }
    if (next_size <= 0) {
        return 1;
    }

    bytes.resize(next_size);
    const auto dataptr = m_buffer.data() + m_dataoffset;
    memcpy(bytes.data(), dataptr, next_size);
    m_dataoffset += next_size;
    m_datasize -= next_size;
    return 0;
}

int DOVIRpu::get_next_rpu(std::vector<uint8_t>& bytes, const int64_t id) {
    bytes.clear();
    for (; m_count <= id; m_count++) {
        std::vector<uint8_t> rpu;
        if (int ret = get_next_rpu(rpu); ret != 0) {
            return ret;
        }
        m_rpus[m_count] = rpu;
    }
    if (auto it = m_rpus.find(id); it != m_rpus.end()) {
        bytes = std::move(it->second);
        m_rpus.erase(it);
    } else {
        return 1;
    }
    return 0;
}

int DOVIRpu::get_next_rpu_nal(std::vector<uint8_t>& bytes, const int64_t id) {
    std::vector<uint8_t> rpu;
    if (int ret = get_next_rpu(rpu, id); ret != 0) {
        return ret;
    }
    //to_nal(rpu); // get_next_rpuはすでにこの処理を実施済みのものを返す
    if (rpu.back() == 0x00) { // 最後が0x00の場合
        rpu.push_back(0x03);
    }

    bytes.resize(sizeof(DOVIRpu::rpu_header));
    memcpy(bytes.data(), &DOVIRpu::rpu_header, sizeof(DOVIRpu::rpu_header));

    uint16_t u16 = 0x00;
    u16 |= (NALU_HEVC_UNSPECIFIED << 9) | 1;
    add_u16(bytes, u16);
    vector_cat(bytes, rpu);
    return 0;
}

int DOVIRpu::get_next_rpu_obu(std::vector<uint8_t>& bytes, const int64_t id) {
    std::vector<uint8_t> tmp;
    if (int ret = get_next_rpu(tmp, id); ret != 0) {
        return ret;
    }

    auto rpu = unnal(tmp.data(), tmp.size());

    std::vector<uint8_t> buf;
    if (rpu.size() > sizeof(av1_itut_t35_header_dovirpu) && memcmp(rpu.data(), av1_itut_t35_header_dovirpu, sizeof(av1_itut_t35_header_dovirpu)) == 0) {
        buf = rpu;
    } else {
        buf = make_vector<uint8_t>(av1_itut_t35_header_dovirpu);
        vector_cat(buf, rpu);
    }
    bytes = gen_av1_obu_metadata(AV1_METADATA_TYPE_ITUT_T35, buf);
    return 0;
}

int DOVIRpu::get_next_rpu(std::vector<uint8_t>& bytes, const int64_t id, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_HEVC: return get_next_rpu_nal(bytes, id);
    case RGY_CODEC_AV1: return get_next_rpu_obu(bytes, id);
    default: return 1;
    }
}

const DOVIProfile *getDOVIProfile(const int id) {
    static const std::array<DOVIProfile, 4> DOVI_PROFILES = {
        DOVIProfile{ 50, true, true, true, VideoVUIInfo(1, RGY_PRIM_UNSPECIFIED, RGY_MATRIX_UNSPECIFIED, RGY_TRANSFER_UNSPECIFIED, 5, RGY_COLORRANGE_FULL,    RGY_CHROMALOC_UNSPECIFIED) },
        DOVIProfile{ 81, true, true, true, VideoVUIInfo(1, RGY_PRIM_BT2020,      RGY_MATRIX_BT2020_NCL,  RGY_TRANSFER_ST2084,      5, RGY_COLORRANGE_LIMITED, RGY_CHROMALOC_UNSPECIFIED) },
        DOVIProfile{ 82, true, true, true, VideoVUIInfo(1, RGY_PRIM_BT709,       RGY_MATRIX_BT709,       RGY_TRANSFER_BT709,       5, RGY_COLORRANGE_LIMITED, RGY_CHROMALOC_UNSPECIFIED) },
        DOVIProfile{ 84, true, true, true, VideoVUIInfo(1, RGY_PRIM_BT2020,      RGY_MATRIX_BT2020_NCL,  RGY_TRANSFER_ARIB_B67,    5, RGY_COLORRANGE_LIMITED, RGY_CHROMALOC_UNSPECIFIED) }
    };
    for (const auto& profile : DOVI_PROFILES) {
        if (profile.profile == id) {
            return &profile;
        }
    }
    return nullptr;
}

std::vector<nal_info> parse_nal_unit_h264_c(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const auto next = rgy_memmem_c((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next == RGY_MEMMEM_NOT_FOUND) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i-1] == 0);
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

std::vector<nal_info> parse_nal_unit_hevc_c(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const auto next = rgy_memmem_c((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next == RGY_MEMMEM_NOT_FOUND) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = (data[i + 3] & 0x7f) >> 1;
            nal_start.nuh_layer_id = ((data[i + 3] & 1) << 5) | ((data[i + 4] & 0xf8) >> 3);
            nal_start.temporal_id = (data[i + 4] & 0x07) - 1;
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

std::vector<nal_info> parse_nal_unit_vvc_c(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size >= 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0, 0, 0 };
        int64_t i = 0;
        for (;;) {
            const auto next = rgy_memmem_c((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next == RGY_MEMMEM_NOT_FOUND) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.nuh_layer_id =  data[i + 3] & 0x3f;
            nal_start.type = (data[i + 4] & 0xf8) >> 3;
            nal_start.temporal_id = (data[i + 4] & 0x07) - 1;
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

size_t find_header_c(const uint8_t *data, size_t size) {
    return rgy_memmem_c(data, size, DOVIRpu::rpu_header, sizeof(DOVIRpu::rpu_header));
}

#include "rgy_simd.h"

decltype(parse_nal_unit_h264_c)* get_parse_nal_unit_h264_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
#if defined(_M_X64) || defined(__x86_64)
    if ((simd & RGY_SIMD::AVX512BW) == RGY_SIMD::AVX512BW) return parse_nal_unit_h264_avx512bw;
#endif
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return parse_nal_unit_h264_avx2;
#endif
    return parse_nal_unit_h264_c;
}
decltype(parse_nal_unit_hevc_c)* get_parse_nal_unit_hevc_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
#if defined(_M_X64) || defined(__x86_64)
    if ((simd & RGY_SIMD::AVX512BW) == RGY_SIMD::AVX512BW) return parse_nal_unit_hevc_avx512bw;
#endif
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return parse_nal_unit_hevc_avx2;
#endif
    return parse_nal_unit_hevc_c;
}

decltype(find_header_c)* get_find_header_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
#if defined(_M_X64) || defined(__x86_64)
    if ((simd & RGY_SIMD::AVX512BW) == RGY_SIMD::AVX512BW) return find_header_avx512bw;
#endif
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return find_header_avx2;
#endif
    return find_header_c;
}

static std::unique_ptr<unit_info> get_unit(const uint8_t *data, const size_t size) {
    std::unique_ptr<unit_info> unit;
    if (size <= 1) {
        return unit;
    }
    const uint8_t *const start_pos = data;
    const uint8_t firstbyte = *data++;
    const uint8_t type = (firstbyte & (0x78)) >> 3;
    const uint8_t extension_flag = (firstbyte & 0x04) >> 2;
    const uint8_t has_size_flag = (firstbyte & 0x02) >> 1;

    unit = std::make_unique<unit_info>();
    unit->type = type;
    unit->extension_flag = extension_flag;
    unit->has_size_flag = has_size_flag;

    if (extension_flag) {
        data++;
    }
    if (!has_size_flag) {
        size_t ret = size - 1 - extension_flag;
        unit->unit_data.resize(ret);
    } else {
        size_t obu_size = 0;
        for (int i = 0; i < 8; i++) {
            uint8_t byte = *data++;
            obu_size |= (int64_t)(byte & 0x7f) << (i * 7);
            if (!(byte & 0x80))
                break;
        }

        const size_t ret = obu_size + (data - start_pos);
        unit->unit_data.resize(ret);
    }
    unit->obu_offset = (int)(data - start_pos);
    if (unit->unit_data.size() > 0) {
        memcpy(unit->unit_data.data(), start_pos, unit->unit_data.size());
    }
    return unit;
}

std::deque<std::unique_ptr<unit_info>> parse_unit_av1(const uint8_t *data, const size_t size) {
    std::deque<std::unique_ptr<unit_info>> list;
    int64_t size_remain = (int64_t)size;
    while (size_remain > 0) {
        auto unit = get_unit(data, size);
        const auto unit_size = unit->unit_data.size();
        if (unit_size == 0) {
            break;
        }
        list.push_back(std::move(unit));
        data += unit_size;
        size_remain -= unit_size;
    }
    return list;
}

#if 0


class RGYBitStreamReader {
public:
    RGYBitStreamReader() : data_(nullptr), size_(0), pos_(0) {}

    void set(const uint8_t *data, size_t size) {
        data_ = data;
        size_ = size;
        pos_ = 0;
    }

    uint8_t getBit() {
        if (pos_ >= size_ * 8) {
            throw std::runtime_error("ビットストリームの終端を超えています。");
        }
        uint8_t bit = (data_[pos_ / 8] >> (7 - (pos_ % 8))) & 1;
        pos_++;
        return bit;
    }

    uint32_t getBits(size_t numBits) {
        uint32_t result = 0;
        for (size_t i = 0; i < numBits; i++) {
            result |= getBit() << (numBits - 1 - i);
        }
        return result;
    }

    uint32_t getUEGolomb() {
        uint32_t zero_count = 0;
        while (getBit() == 0) {
            zero_count++;
        }
        uint32_t result = (1 << zero_count) - 1;
        for (uint32_t i = 0; i < zero_count; i++) {
            result |= getBit() << (zero_count - 1 - i);
        }
        return result;
    }

    void getBytes(uint8_t *data, size_t size) {
        memcpy(data, data_ + pos_/8, size);
        pos_ += size * 8;
    }

private:
    const uint8_t *data_;
    size_t size_;
    size_t pos_;
};

static const int HEVC_MAX_SUB_LAYERS = 7;
struct RGYHEVCSublayerHdrParams {
    uint32_t bit_rate_value_minus1[32];
    uint32_t cpb_size_value_minus1[32];
    uint32_t cpb_size_du_value_minus1[32];
    uint32_t bit_rate_du_value_minus1[32];
    uint32_t cbr_flag;
};

struct RGYHEVCHdrFlags {
    uint8_t fixed_pic_rate_general_flag;
    uint8_t fixed_pic_rate_within_cvs_flag;
    uint8_t low_delay_hrd_flag;
};

struct RGYHEVCHdrParams {
    RGYHEVCHdrFlags flags;
    uint8_t nal_hrd_parameters_present_flag;
    uint8_t vcl_hrd_parameters_present_flag;
    uint8_t sub_pic_hrd_params_present_flag;
    uint8_t sub_pic_cpb_params_in_pic_timing_sei_flag;

    uint8_t tick_divisor_minus2;
    uint8_t du_cpb_removal_delay_increment_length_minus1;
    uint8_t dpb_output_delay_du_length_minus1;
    uint8_t bit_rate_scale;
    uint8_t cpb_size_scale;
    uint8_t cpb_size_du_scale;
    uint8_t initial_cpb_removal_delay_length_minus1;
    uint8_t au_cpb_removal_delay_length_minus1;
    uint8_t dpb_output_delay_length_minus1;
    uint8_t cpb_cnt_minus1[HEVC_MAX_SUB_LAYERS];
    uint16_t elemental_duration_in_tc_minus1[HEVC_MAX_SUB_LAYERS];

    RGYHEVCSublayerHdrParams nal_params[HEVC_MAX_SUB_LAYERS];
    RGYHEVCSublayerHdrParams vcl_params[HEVC_MAX_SUB_LAYERS];
};

struct RGYHEVCVPSProfileLevelTierData {
    uint8_t profile_space;
    uint8_t tier_flag;
    uint8_t profile_idc;
    uint8_t data[10];
    uint8_t level_idc;
};


struct RGYHEVCVPSProfileLevelTier {
    RGYHEVCVPSProfileLevelTierData ptl;
    RGYHEVCVPSProfileLevelTierData sub_layer_ptl[HEVC_MAX_SUB_LAYERS];

    uint8_t sub_layer_profile_present_flag[HEVC_MAX_SUB_LAYERS];
    uint8_t sub_layer_level_present_flag[HEVC_MAX_SUB_LAYERS];
};

class RGYHEVCVPSReader {
    RGYBitStreamReader m_reader;
public:
    RGYHEVCVPSReader() : m_reader() {}

    void parseProfileLevelTier(RGYHEVCVPSProfileLevelTierData &profileLevelTier) {
        profileLevelTier.profile_space = m_reader.getBits(2);
        profileLevelTier.tier_flag = m_reader.getBit();
        profileLevelTier.profile_idc = m_reader.getBits(5);
        m_reader.getBytes(profileLevelTier.data, 10);
    }

    void parseProfileLevelTier(RGYHEVCVPSProfileLevelTier &profileLevelTier, int max_num_sub_layers) {
        parseProfileLevelTier(profileLevelTier.ptl);
        profileLevelTier.ptl.level_idc = m_reader.getBits(8);
        for (int i = 0; i < max_num_sub_layers - 1; i++) {
            profileLevelTier.sub_layer_profile_present_flag[i] = m_reader.getBit();
            profileLevelTier.sub_layer_level_present_flag[i] = m_reader.getBit();
        }
        if (max_num_sub_layers - 1 > 0) {
            for (int i = max_num_sub_layers - 1; i < 8; i++) {
                m_reader.getBits(2); //skip
            }
        }
        for (int i = 0; i < max_num_sub_layers - 1; i++) {
            if (profileLevelTier.sub_layer_profile_present_flag[i]) {
                parseProfileLevelTier(profileLevelTier.sub_layer_ptl[i]);
            }
            if (profileLevelTier.sub_layer_level_present_flag[i]) {
                profileLevelTier.sub_layer_ptl[i].level_idc = m_reader.getBits(8);
            }
        }
    }

    void parseSubLayerHRD(RGYHEVCSublayerHdrParams &hdr, const int cpb_cnt_minus1, const int subpic_params_present) {
        for (int i = 0; i <= cpb_cnt_minus1; i++) {
            hdr.bit_rate_value_minus1[i] = m_reader.getUEGolomb();
            hdr.cpb_size_value_minus1[i] = m_reader.getUEGolomb();
            if (subpic_params_present) {
                hdr.cpb_size_du_value_minus1[i] = m_reader.getUEGolomb();
                hdr.bit_rate_du_value_minus1[i] = m_reader.getUEGolomb();
            }
            hdr.cbr_flag |= m_reader.getBit() << i;
        }
    }

    void parseHEVCHRD(RGYHEVCHdrParams &hdr, uint8_t cprms_present_flag, int max_num_sub_layers) {
        if (cprms_present_flag) {
            hdr.nal_hrd_parameters_present_flag = m_reader.getBit();
            hdr.vcl_hrd_parameters_present_flag = m_reader.getBit();
            if (hdr.nal_hrd_parameters_present_flag || hdr.vcl_hrd_parameters_present_flag) {
                hdr.sub_pic_hrd_params_present_flag = m_reader.getBit();
                if (hdr.sub_pic_hrd_params_present_flag) {
                    hdr.tick_divisor_minus2 = m_reader.getBits(8);
                    hdr.du_cpb_removal_delay_increment_length_minus1 = m_reader.getBits(5);
                    hdr.sub_pic_cpb_params_in_pic_timing_sei_flag = m_reader.getBit();
                    hdr.dpb_output_delay_du_length_minus1 = m_reader.getBits(5);
                }
                hdr.bit_rate_scale = m_reader.getBits(4);
                hdr.cpb_size_scale = m_reader.getBits(4);
                if (hdr.sub_pic_hrd_params_present_flag) {
                    hdr.cpb_size_du_scale = m_reader.getBits(4);
                }
                hdr.initial_cpb_removal_delay_length_minus1 = m_reader.getBits(5);
                hdr.au_cpb_removal_delay_length_minus1 = m_reader.getBits(5);
                hdr.dpb_output_delay_length_minus1 = m_reader.getBits(5);
            }
        }

        for (int i = 0; i < max_num_sub_layers; i++) {
            uint32_t fixed_pic_rate_general_flag = m_reader.getBit();
            uint32_t fixed_pic_rate_within_cvs_flag = 0;
            if (!fixed_pic_rate_general_flag) {
                fixed_pic_rate_within_cvs_flag = m_reader.getBit();
            }

            uint32_t low_delay_hrd_flag = 0;
            if (fixed_pic_rate_within_cvs_flag || fixed_pic_rate_general_flag) {
                hdr.elemental_duration_in_tc_minus1[i] = m_reader.getUEGolomb();
            } else {
                low_delay_hrd_flag = m_reader.getBit();
            }
            hdr.flags.fixed_pic_rate_general_flag |= fixed_pic_rate_general_flag << i;
            hdr.flags.fixed_pic_rate_within_cvs_flag |= fixed_pic_rate_within_cvs_flag << i;
            hdr.flags.low_delay_hrd_flag |= low_delay_hrd_flag << i;

            if (!low_delay_hrd_flag) {
                hdr.cpb_cnt_minus1[i] = m_reader.getUEGolomb();
            }

            if (hdr.nal_hrd_parameters_present_flag) {
                parseSubLayerHRD(hdr.nal_params[i], hdr.cpb_cnt_minus1[i], hdr.sub_pic_hrd_params_present_flag);
            }

            if (hdr.vcl_hrd_parameters_present_flag) {
                parseSubLayerHRD(hdr.vcl_params[i], hdr.cpb_cnt_minus1[i], hdr.sub_pic_hrd_params_present_flag);
            }
        }
    }

    void parseHEVCVPS(const uint8_t *data, size_t size) {
        const auto vps = unnal(data + 6, size - 6);

        m_reader.set(vps.data(), vps.size());

        uint8_t vps_video_parameter_set_id = m_reader.getBits(4);
        uint8_t vps_base_layer_internal_flag = m_reader.getBit();
        uint8_t vps_base_layer_available_flag = m_reader.getBit();
        uint8_t vps_max_layers_minus1 = m_reader.getBits(6);
        uint8_t vps_max_sub_layers_minus1 = m_reader.getBits(3);
        uint8_t vps_temporal_id_nesting_flag = m_reader.getBit();
        uint16_t vps_reserved_0xffff_16bits = m_reader.getBits(16);

        RGYHEVCVPSProfileLevelTier vps_profile_level_tier = { 0 };
        parseProfileLevelTier(vps_profile_level_tier, vps_max_sub_layers_minus1 + 1);

        uint8_t vps_sub_layer_ordering_info_present_flag = m_reader.getBit();
        for (int i = (vps_sub_layer_ordering_info_present_flag ? 0 : vps_max_sub_layers_minus1); i <= vps_max_sub_layers_minus1; i++) {
            uint8_t vps_max_dec_pic_buffering_minus1 = m_reader.getUEGolomb();
            uint8_t vps_max_num_reorder_pics = m_reader.getUEGolomb();
            uint8_t vps_max_latency_increase_plus1 = m_reader.getUEGolomb();
            vps_max_latency_increase_plus1 = vps_max_latency_increase_plus1;
        }
        uint8_t vps_max_layer_id = m_reader.getBits(6);
        uint16_t vps_num_layer_sets_minus1 = m_reader.getUEGolomb();

        std::vector<uint8_t> layer_id_included_flag(vps_num_layer_sets_minus1 + 1);
        for (int i = 1; i < vps_num_layer_sets_minus1+1; i++) {
            for (int j = 0; j <= vps_max_layer_id; j++) {
                layer_id_included_flag[i] |= m_reader.getBit() << j;
            }
        }

        RGYHEVCHdrParams hdr = { 0 };
        uint8_t vps_timing_info_present_flag = m_reader.getBit();
        if (vps_timing_info_present_flag) {
            uint32_t vps_num_units_in_tick = m_reader.getBits(32);
            uint32_t vps_time_scale = m_reader.getBits(32);
            uint8_t vps_poc_proportional_to_timing_flag = m_reader.getBit();
            if (vps_poc_proportional_to_timing_flag) {
                uint32_t vps_num_ticks_poc_diff_one_minus1 = m_reader.getUEGolomb();
            }
            uint32_t vps_num_hrd_parameters = m_reader.getUEGolomb();
            for (int i = 0; i < vps_num_hrd_parameters; i++) {
                uint32_t vps_hrd_layer_set_id = m_reader.getUEGolomb();
                uint8_t cprms_present_flag = 1;
                if (i > 0) {
                    cprms_present_flag = m_reader.getBit();
                }
                parseHEVCHRD(hdr, cprms_present_flag, vps_max_sub_layers_minus1 + 1);
            }
        }
        uint8_t vps_extension_flag = m_reader.getBit();
        vps_extension_flag = vps_extension_flag;
    }
};
#endif