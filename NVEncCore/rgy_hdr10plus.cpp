// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#include "rgy_osdep.h"
#include "rgy_hdr10plus.h"
#include "rgy_filesystem.h"

#if defined(_WIN32) || defined(_WIN64)
#include <fcntl.h>
const TCHAR *RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME =  _T("hdr10plus_gen.exe");
#else
const TCHAR *RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME =  _T("hdr10plus_gen");
#endif

RGYHDR10Plus::RGYHDR10Plus() :
    m_proc(),
    m_buffer(std::make_pair(-1, vector<uint8_t>())) {
}

RGYHDR10Plus::~RGYHDR10Plus() {
    m_proc.reset();
}

RGY_ERR RGYHDR10Plus::init(const tstring &inputJson) {
    if (!(rgy_file_exists(inputJson))) {
        return RGY_ERR_NOT_FOUND;
    }
    m_inputJson = inputJson;
#if defined(_WIN32) || defined(_WIN64)
    tstring HDR10PlusGenExePath = getExeDir() + _T("\\") + HDR10PLUS_GEN_EXE_NAME;
#else
    tstring HDR10PlusGenExePath = getExeDir() + _T("/") + HDR10PLUS_GEN_EXE_NAME;
#endif
    if (!rgy_file_exists(HDR10PLUS_GEN_EXE_NAME)) {
        HDR10PlusGenExePath = HDR10PLUS_GEN_EXE_NAME;
    }
    const tstring HDR10PlusGenExePathWithQuotes = tstring(_T("\"")) + HDR10PlusGenExePath + _T("\"");
    const tstring inputJsonWithQuotes = tstring(_T("\"")) + inputJson + _T("\"");
    const std::vector<tstring> args = {
        HDR10PlusGenExePathWithQuotes,
        _T("-i"), inputJsonWithQuotes,
        _T("-o"), _T("-")
    };

    m_proc = createRGYPipeProcess();
    m_proc->init(PIPE_MODE_DISABLE, PIPE_MODE_ENABLE | PIPE_MODE_ENABLE_FP, PIPE_MODE_DISABLE);
    if (m_proc->run(args, nullptr, 0, true, true)) {
        return RGY_ERR_RUN_PROCESS;
    }
    return RGY_ERR_NONE;
}

const vector<uint8_t> *RGYHDR10Plus::getData(int iframe) {
    while (m_buffer.first != iframe) {
        m_buffer.second.clear();
        int header[2];
        if (m_proc->stdOutFpRead(header, sizeof(header)) != sizeof(header)) {
            return nullptr;
        }
        const int frameNum = header[0];
        const int dataSize = header[1];
        m_buffer.second.resize(dataSize, 0);
        if (m_proc->stdOutFpRead(m_buffer.second.data(), m_buffer.second.size()) != m_buffer.second.size()) {
            return nullptr;
        }
        m_buffer.first = frameNum;
    }
    return &m_buffer.second;
}
