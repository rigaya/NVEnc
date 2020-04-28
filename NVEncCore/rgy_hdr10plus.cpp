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

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME =  _T("hdr10plus_gen.exe");
#else
const TCHAR *RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME =  _T("hdr10plus_gen");
#endif

RGYHDR10Plus::RGYHDR10Plus() :
    m_proc(), m_pipes(),
    m_fpStdOut(std::unique_ptr<FILE, decltype(&fclose)>(nullptr, fclose)),
    m_buffer(std::make_pair(-1, vector<uint8_t>())){
}

RGYHDR10Plus::~RGYHDR10Plus() {
    m_fpStdOut.reset();
    m_proc.reset();
}

RGY_ERR RGYHDR10Plus::init(const tstring &inputJson) {
    if (!(PathFileExists(inputJson.c_str()))) {
        return RGY_ERR_NOT_FOUND;
    }
    m_inputJson = inputJson;
    const tstring HDR10PlusGenExePath = (PathFileExists(HDR10PLUS_GEN_EXE_NAME)) ? HDR10PLUS_GEN_EXE_NAME : getExeDir() + _T("\\") + HDR10PLUS_GEN_EXE_NAME;
    if (!(PathFileExists(HDR10PlusGenExePath.c_str()))) {
        return RGY_ERR_NOT_FOUND;
    }
    const tstring HDR10PlusGenExePathWithQuotes = tstring(_T("\"")) + HDR10PlusGenExePath + _T("\"");
    const tstring inputJsonWithQuotes = tstring(_T("\"")) + inputJson + _T("\"");
    std::vector<const TCHAR *> args;
    args.push_back(HDR10PlusGenExePathWithQuotes.c_str());
    args.push_back(_T("-i"));
    args.push_back(inputJsonWithQuotes.c_str());
    args.push_back(_T("-o"));
    args.push_back(_T("-"));

    m_pipes.stdOut.mode = PIPE_MODE_ENABLE;
    m_pipes.stdOut.bufferSize = 1024;

    m_proc = createRGYPipeProcess();
    m_proc->init();
    if (m_proc->run(args, nullptr, &m_pipes, 0, true, true)) {
        return RGY_ERR_RUN_PROCESS;
    }
#if defined(_WIN32) || defined(_WIN64)
    m_fpStdOut = std::unique_ptr<FILE, decltype(&fclose)>(_fdopen(_open_osfhandle((intptr_t)m_pipes.stdOut.h_read, _O_BINARY), "rb"), fclose);
#else
    m_fpStdOut = std::unique_ptr<FILE, decltype(&fclose)>(m_pipes.f_stdout, fclose);
#endif
    if (!m_fpStdOut) {
        return RGY_ERR_INVALID_HANDLE;
    }
    return RGY_ERR_NONE;
}

const vector<uint8_t> *RGYHDR10Plus::getData(int iframe) {
    if (!m_fpStdOut) {
        return nullptr;
    }
    while (m_buffer.first != iframe) {
        m_buffer.second.clear();
        int header[2];
        if (fread(header, sizeof(header[0]), _countof(header), m_fpStdOut.get()) != _countof(header)) {
            return nullptr;
        }
        const int frameNum = header[0];
        const int dataSize = header[1];
        m_buffer.second.resize(dataSize, 0);
        if (fread(m_buffer.second.data(), 1, m_buffer.second.size(), m_fpStdOut.get()) != m_buffer.second.size()) {
            return nullptr;
        }
        m_buffer.first = frameNum;
    }
    return &m_buffer.second;
}
