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

#pragma once
#ifndef __RGY_INPUT_SM_H__
#define __RGY_INPUT_SM_H__

#include "rgy_input.h"
#include "rgy_shared_mem.h"

static const char *RGYInputSMPrmSM       = "RGYInputSMPrmSM";
static const char *RGYInputSMBuffer      = "RGYInputSMBuffer";
static const char *RGYInputSMEventEmpty  = "RGYInputSMEventEmpty";
static const char *RGYInputSMEventFilled = "RGYInputSMEventFilled";

#pragma pack(push)
#pragma pack(1)
struct RGYInputSMPrm {
    int w, h;
    int fpsN, fpsD;
    int pitch;
    RGY_CSP csp;
    RGY_PICSTRUCT picstruct;
    uint32_t bufSize;
    bool abort;
};
#pragma pack(pop)

#if ENABLE_SM_READER

class RGYInputSM : public RGYInput {
public:
    RGYInputSM();
    virtual ~RGYInputSM();

    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) override;

    std::unique_ptr<RGYSharedMemWin> m_prm;
    std::unique_ptr<RGYSharedMem> m_sm;
    std::unique_ptr<void, handle_deleter> m_buf_empty;
    std::unique_ptr<void, handle_deleter> m_buf_filled;
};

#endif //#if ENABLE_SM_READER

#endif //__RGY_INPUT_RAW_H__
