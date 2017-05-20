// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <string>
#include "NVEncUtil.h"
#include "rgy_status.h"
#include "rgy_version.h"
#include "NVEncInput.h"

#if ENABLE_RAW_READER

class NVEncInputRaw : public NVEncBasicInput {
public:
    NVEncInputRaw();
    ~NVEncInputRaw();

    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) override;
    RGY_ERR ParseY4MHeader(char *buf, VideoInfo *inputPrm);

    bool m_by4m;
    FILE *m_fSource;

    uint32_t m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;
};

#endif //ENABLE_RAW_READER
