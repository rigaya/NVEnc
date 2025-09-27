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
#ifndef __RGY_INPUT_RAW_H__
#define __RGY_INPUT_RAW_H__

#include "rgy_input.h"

#if ENABLE_RAW_READER



class RGYInputPrmRaw : public RGYInputPrm {
public:
    RGY_CSP inputCsp;
    RGYParamParallelEncPipeHandle chunkPipeHandle;

    RGYInputPrmRaw(RGYInputPrm base) : RGYInputPrm(base), inputCsp(RGY_CSP_YV12), chunkPipeHandle() {};
    virtual ~RGYInputPrmRaw() {};
};

class RGYInputRaw : public RGYInput {
public:
    RGYInputRaw();
    virtual ~RGYInputRaw();

    virtual void Close() override;

    virtual bool isPipe() const override {
        return m_isPipe;
    }
    virtual bool seekable() const override {
        return !isPipe();
    }
    virtual bool timestampStable() const override {
        return true;
    }
    virtual int64_t GetVideoFirstKeyPts() const override;

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) override;
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *pSurface) override;
    RGY_ERR ParseY4MHeader(char *buf, VideoInfo *pInfo);

    FILE *m_fSource;

    uint32_t m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;
    bool m_isPipe;
    RGYParamParallelEncPipeHandle m_chunkPipeHandle;
    int64_t m_firstKeyPts;
};

#endif //ENABLE_RAW_READER

#endif //__RGY_INPUT_RAW_H__
