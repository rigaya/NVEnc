// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <io.h>
#include <fcntl.h>
#include <string>
#include <sstream>
#include "NVEncStatus.h"
#include "nvEncodeAPI.h"
#include "NVEncInput.h"
#include "ConvertCSP.h"


NVEncBasicInput::NVEncBasicInput() {
    memset(&m_sTrimParam, 0, sizeof(m_sTrimParam));
    memset(&m_sInputCrop, 0, sizeof(m_sInputCrop));
    memset(&m_sDecParam, 0, sizeof(m_sDecParam));
    m_nInputCodec = cudaVideoCodec_NumCodecs;
}

NVEncBasicInput::~NVEncBasicInput() {
    Close();
}

#pragma warning(push)
#pragma warning(disable:4100)
int NVEncBasicInput::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {

    return 0;
}

int NVEncBasicInput::LoadNextFrame(void *dst, int dst_pitch) {
    return 0;
}
#pragma warning(pop)

void NVEncBasicInput::CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const InputVideoInfo *inputPrm) {
    std::basic_stringstream<TCHAR> ss;

    ss << inputTypeName;
    ss << _T("(") << inputCSpName << _T(")");
    ss << _T("->") << outputCSpName;
    if (convSIMD && _tcslen(convSIMD)) {
        ss << _T(" [") << convSIMD << _T("]");
    }
    ss << _T(", ");
    ss << inputPrm->width << _T("x") << inputPrm->height << _T(", ");
    ss << inputPrm->rate << _T("/") << inputPrm->scale << _T(" fps");

    m_strInputInfo = ss.str();
}

void NVEncBasicInput::Close() {
    m_pEncSatusInfo.reset();
}
