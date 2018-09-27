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

    m_strInputInfo = ss.str();
}
