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
// ------------------------------------------------------------------------------------------

#include <sstream>
#include <iostream>
#include <fstream>
#include <set>
#include "rgy_input.h"

std::vector<int> read_keyfile(tstring keyfile) {
    std::set<int> s; //重複回避のため
    std::ifstream ifs(keyfile);
    if (ifs.is_open()) {
        std::string buff;
        while (std::getline(ifs, buff)) {
            if (buff.length() > 0) {
                try {
                    s.insert(std::stoi(buff));
                } catch (...) {
                    return vector<int>();
                }
            }
        }
    }
    return vector<int>(s.begin(), s.end());
}

RGYConvertCSPPrm::RGYConvertCSPPrm() :
    abort(false),
    dst(nullptr),
    src(nullptr),
    interlaced(false),
    width(0),
    src_y_pitch_byte(0),
    src_uv_pitch_byte(0),
    dst_y_pitch_byte(0),
    height(0),
    dst_height(0),
    crop(nullptr) {

}


RGYConvertCSP::RGYConvertCSP() : RGYConvertCSP(0) {
}

RGYConvertCSP::RGYConvertCSP(int threads) :
    m_csp(nullptr),
    m_csp_from(RGY_CSP_NA),
    m_csp_to(RGY_CSP_NA),
    m_uv_only(false),
    m_threads(threads),
    m_th(), m_heStart(), m_heFin(), m_heFinCopy(),
    m_prm() {
};

RGYConvertCSP::~RGYConvertCSP() {
    m_prm.abort = true;
    for (size_t i = 0; i < m_heStart.size(); i++) {
        SetEvent(m_heStart[i].get());
    }
    for (size_t i = 0; i < m_th.size(); i++) {
        m_th[i].join();
    }
    m_heFinCopy.clear();
    m_heStart.clear();
    m_heFin.clear();
    m_th.clear();
};
const ConvertCSP *RGYConvertCSP::getFunc(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, uint32_t simd) {
    if (m_csp == nullptr
        || (m_csp_from != csp_from || m_csp_to != csp_to || m_uv_only != uv_only)) {
        m_csp_from = csp_from;
        m_csp_to = csp_to;
        m_uv_only = uv_only;
        m_csp = get_convert_csp_func(csp_from, csp_to, uv_only, simd);
    }
    return m_csp;
}

int RGYConvertCSP::run(int interlaced, void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
    if (m_threads == 0) {
        const int div = (m_csp->simd == 0) ? 2 : 4;
        const int max = (m_csp->simd == 0) ? 8 : 4;
        m_threads = std::min(max, ((int)get_cpu_info().physical_cores + div) / div);
    }
    if (m_threads > 1 && m_th.size() == 0) {
        m_heFinCopy.clear();
        m_heStart.clear();
        m_heFin.clear();
        for (int ith = 1; ith < m_threads; ith++) {
            auto heStart = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            auto heFin = std::unique_ptr<void, handle_deleter>(CreateEvent(nullptr, false, false, nullptr), handle_deleter());
            m_th.push_back(std::thread([heStart = heStart.get(), heFin = heFin.get(), ithId = ith, threadN = m_threads, prm = &m_prm, cspfunc = &m_csp]() {
                WaitForSingleObject((HANDLE)heStart, INFINITE);
                while (!prm->abort) {
                    (*cspfunc)->func[prm->interlaced](prm->dst, prm->src,
                        prm->width, prm->src_y_pitch_byte, prm->src_uv_pitch_byte, prm->dst_y_pitch_byte,
                        prm->height, prm->dst_height, ithId, threadN, prm->crop);
                    SetEvent((HANDLE)heFin);
                    WaitForSingleObject((HANDLE)heStart, INFINITE);
                }
            }));
            m_heFinCopy.push_back(heFin.get());
            m_heStart.push_back(std::move(heStart));
            m_heFin.push_back(std::move(heFin));
        }
    }
    m_prm.abort = false;
    m_prm.interlaced = interlaced;
    m_prm.dst = dst;
    m_prm.src = src;
    m_prm.width = width;
    m_prm.src_y_pitch_byte = src_y_pitch_byte;
    m_prm.src_uv_pitch_byte = src_uv_pitch_byte;
    m_prm.dst_y_pitch_byte = dst_y_pitch_byte;
    m_prm.height = height;
    m_prm.dst_height = dst_height;
    m_prm.crop = crop;
    for (size_t i = 0; i < m_heStart.size(); i++) {
        SetEvent(m_heStart[i].get());
    }
    m_csp->func[interlaced](dst, src,
        width, src_y_pitch_byte, src_uv_pitch_byte, dst_y_pitch_byte,
        height, dst_height, 0, (int)m_th.size()+1, crop);
    if (m_th.size() > 0) {
        WaitForMultipleObjects((DWORD)m_heFinCopy.size(), m_heFinCopy.data(), TRUE, INFINITE);
    }
    return 0;
}

RGYInput::RGYInput() :
    m_pEncSatusInfo(),
    m_inputVideoInfo(),
    m_InputCsp(RGY_CSP_NA),
    m_sConvert(nullptr),
    m_pPrintMes(),
    m_strInputInfo(),
    m_strReaderName(_T("unknown")),
    m_sTrimParam() {
    m_sTrimParam.list.clear();
    m_sTrimParam.offset = 0;
    memset(&m_inputVideoInfo, 0, sizeof(m_inputVideoInfo));
}

RGYInput::~RGYInput() {
    Close();
}

void RGYInput::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));

    m_pEncSatusInfo.reset();
    m_sConvert = nullptr;

    m_strInputInfo.empty();

    m_sTrimParam.list.clear();
    m_sTrimParam.offset = 0;
    AddMessage(RGY_LOG_DEBUG, _T("Close...\n"));
    m_pPrintMes.reset();
}

void RGYInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const VideoInfo *inputPrm) {
    std::basic_stringstream<TCHAR> ss;

    ss << inputTypeName;
    ss << _T("(") << inputCSpName << _T(")");
    ss << _T("->") << outputCSpName;
    if (convSIMD && _tcslen(convSIMD)) {
        ss << _T(" [") << convSIMD << _T("]");
    }
    ss << _T(", ");
    ss << inputPrm->srcWidth << _T("x") << inputPrm->srcHeight << _T(", ");
    ss << inputPrm->fpsN << _T("/") << inputPrm->fpsD << _T(" fps");

    if (cropEnabled(inputPrm->crop)) {
        ss << "crop(" << inputPrm->crop.e.left << "," << inputPrm->crop.e.up << "," << inputPrm->crop.e.right << "," << inputPrm->crop.e.bottom << ")";
    }

    m_strInputInfo = ss.str();
}
