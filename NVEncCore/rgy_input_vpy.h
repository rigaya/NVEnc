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
#ifndef __RGY_INPUT_VPY_H__
#define __RGY_INPUT_VPY_H__

#include "rgy_version.h"
#if ENABLE_VAPOURSYNTH_READER
#include "rgy_osdep.h"
#include "rgy_input.h"

// VapourSynth headers/types are intentionally NOT included here.
// VapourSynth(v3/v4) specific code is isolated behind a wrapper implemented in .cpp files.
class RGYVapourSynthWrapper;

const int ASYNC_BUFFER_2N = 7;
const int ASYNC_BUFFER_SIZE = 1<<ASYNC_BUFFER_2N;

class RGYInputVpyPrm : public RGYInputPrm {
public:
    tstring vsdir;
    float seekRatio; //開始位置を指定する場合の割合 (0.0～1.0)、並列エンコード時に使用
    RGYInputVpyPrm(RGYInputPrm base);

    virtual ~RGYInputVpyPrm() {};
};

class RGYInputVpy : public RGYInput {
public:
    RGYInputVpy();
    virtual ~RGYInputVpy();

    virtual void Close() override;

    void setFrameToAsyncBuffer(int n, const void* f); // f is VSFrameRef* (v3) or VSFrame* (v4)

    virtual int64_t GetVideoFirstKeyPts() const override;
    virtual bool seekable() const override {
        return true;
    }
    virtual bool timestampStable() const override {
        return true;
    }

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) override;
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *pSurface) override;

    int initAsyncEvents();
    void closeAsyncEvents();
    const void* getFrameFromAsyncBuffer(int n) {
        WaitForSingleObject(m_hAsyncEventFrameSetFin[n & (ASYNC_BUFFER_SIZE-1)], INFINITE);
        const void *frame = m_pAsyncBuffer[n & (ASYNC_BUFFER_SIZE-1)];
        SetEvent(m_hAsyncEventFrameSetStart[n & (ASYNC_BUFFER_SIZE-1)]);
        return frame;
    }
    const void* m_pAsyncBuffer[ASYNC_BUFFER_SIZE];
    HANDLE m_hAsyncEventFrameSetFin[ASYNC_BUFFER_SIZE];
    HANDLE m_hAsyncEventFrameSetStart[ASYNC_BUFFER_SIZE];

    bool m_bAbortAsync;
    uint32_t m_nCopyOfInputFrames;

    std::unique_ptr<RGYVapourSynthWrapper> m_vs; // v3/v4 wrapper
    int m_asyncThreads;
    int m_asyncFrames;
    int m_startFrame;
};

#endif //ENABLE_VAPOURSYNTH_READER

#endif //__RGY_INPUT_VPY_H__
