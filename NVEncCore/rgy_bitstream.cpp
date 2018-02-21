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

HEVCHDRSeiPrm::HEVCHDRSeiPrm() : maxcll(-1), maxfall(-1), masterdisplay_set(false), masterdisplay() {
    memset(&masterdisplay, 0, sizeof(masterdisplay));
}

HEVCHDRSei::HEVCHDRSei() : prm() {
}

int HEVCHDRSei::parse(std::string str_maxcll, std::string str_masterdisplay) {
    if (str_maxcll.length()) {
        std::regex re_maxcll(R"((\d+),(\d+))");
        std::smatch match_maxcll;
        if (!regex_search(str_maxcll, match_maxcll, re_maxcll) || match_maxcll.size() != 3) {
            return 1;
        }

        try {
            prm.maxcll = std::stoi(match_maxcll[1]);
            prm.maxfall = std::stoi(match_maxcll[2]);
        } catch (...) {
            return 1;
        }
    }

    if (str_masterdisplay.length()) {
        std::regex re_masterdisplay(R"(G\((\d+),(\d+)\)B\((\d+),(\d+)\)R\((\d+),(\d+)\)WP\((\d+),(\d+)\)L\((\d+),(\d+)\))");
        std::smatch match_masterdisplay;
        if (!regex_search(str_masterdisplay, match_masterdisplay, re_masterdisplay) || match_masterdisplay.size() != 11) {
            return 1;
        }

        try {
            for (int i = 0; i < 10; i++) {
                prm.masterdisplay[i] = std::stoi(match_masterdisplay[i+1]);
            }
        } catch (...) {
            return 1;
        }
        prm.masterdisplay_set = true;
    }
    return 0;
}

HEVCHDRSeiPrm HEVCHDRSei::getprm() {
    return prm;
}

void HEVCHDRSei::add_u16(std::vector<uint8_t>& data, uint16_t u16) {
    data.push_back((uint8_t)((u16 & 0xff00) >> 8));
    data.push_back((uint8_t)(u16 & 0x00ff));
}

void HEVCHDRSei::add_u32(std::vector<uint8_t>& data, uint32_t u32) {
    data.push_back((uint8_t)((u32 & 0xff000000) >> 24));
    data.push_back((uint8_t)((u32 & 0x00ff0000) >> 16));
    data.push_back((uint8_t)((u32 & 0x0000ff00) >>  8));
    data.push_back((uint8_t)((u32 & 0x000000ff) >>  0));
}

std::vector<uint8_t> HEVCHDRSei::gen_nal(HEVCHDRSeiPrm prm_set) {
    prm = prm_set;
    return gen_nal();
}

std::vector<uint8_t> HEVCHDRSei::gen_nal() {
    std::vector<uint8_t> data;
    data.reserve(128);

    auto data_maxcll = sei_maxcll();
    auto data_masterdisplay = sei_masterdisplay();
    if (data_maxcll.size() == 0 && data_masterdisplay.size() == 0) {
        return data;
    }
    std::vector<uint8_t> header = { 0x00, 0x00, 0x00, 0x01 };
    header.reserve(128);

    uint16_t u16 = 0x00;
    u16 |= (39 << 9) | 1;
    add_u16(data, u16);

    vector_cat(data, data_maxcll);
    vector_cat(data, data_masterdisplay);
    to_nal(data);

    vector_cat(header, data);
    header.push_back(0x00);
    return header;
}


std::vector<uint8_t> HEVCHDRSei::sei_maxcll() {
    std::vector<uint8_t> data;
    data.reserve(256);
    if (prm.maxcll >= 0 && prm.maxfall >= 0) {
        data.push_back(144);
        data.push_back(4);
        add_u16(data, (uint16_t)prm.maxcll);
        add_u16(data, (uint16_t)prm.maxfall);
    }
    return data;
}

std::vector<uint8_t> HEVCHDRSei::sei_masterdisplay() {
    std::vector<uint8_t> data;
    data.reserve(256);
    if (prm.masterdisplay_set) {
        data.push_back(137);
        data.push_back(24);
        for (int i = 0; i < 8; i++) {
            add_u16(data, (uint16_t)prm.masterdisplay[i]);
        }
        add_u32(data, (uint32_t)prm.masterdisplay[8]);
        add_u32(data, (uint32_t)prm.masterdisplay[9]);
    }
    return data;
}

void HEVCHDRSei::to_nal(std::vector<uint8_t>& data) {
    for (auto it = data.begin(); it < data.end() - 2; it++) {
        if (    *it == 0
            && *(it+1) == 0) {
            it = data.insert(it+2, 0x03);
            it++;
        }
    }
}