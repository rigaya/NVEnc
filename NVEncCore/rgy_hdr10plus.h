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

#pragma once
#ifndef __RGY_HDR10PLUS_H__
#define __RGY_HDR10PLUS_H__

#include <string>
#include <memory>
#include "rgy_err.h"
#include "rgy_pipe.h"
#include "rgy_util.h"

class RGYHDR10Plus {
public:
    static const TCHAR *HDR10PLUS_GEN_EXE_NAME;
    RGYHDR10Plus();
    virtual ~RGYHDR10Plus();

    RGY_ERR init(const tstring& inputJson);
    const vector<uint8_t> *getData(int iframe);
    const tstring &inputJson() const { return m_inputJson; };
protected:
    tstring m_inputJson;
    std::unique_ptr<RGYPipeProcess> m_proc;
    std::pair<int, std::vector<uint8_t>> m_buffer;
};

#endif //__RGY_HDR10PLUS_H__
