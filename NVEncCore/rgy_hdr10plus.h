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
#include <vector>
#include "rgy_err.h"
#include "rgy_tchar.h"

struct Hdr10PlusRsJsonOpaque;

typedef void (*funcHdr10PlusRsJsonOpaqueDelete)(Hdr10PlusRsJsonOpaque *ptr);

class RGYHDR10Plus {
public:
    RGYHDR10Plus();
    virtual ~RGYHDR10Plus();

    RGY_ERR init(const tstring& inputJson);
    const std::vector<uint8_t> getData(int64_t iframe);
    const tstring &inputJson() const { return m_inputJson; };
    tstring getError();
protected:
    std::unique_ptr<Hdr10PlusRsJsonOpaque, funcHdr10PlusRsJsonOpaqueDelete> m_hdr10plusJson;
    tstring m_inputJson;
};

#endif //__RGY_HDR10PLUS_H__
