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
#ifndef __RGY_INPUT_AVI_H__
#define __RGY_INPUT_AVI_H__

#include "rgy_version.h"
#if ENABLE_AVI_READER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <vfw.h>
#pragma comment(lib, "vfw32.lib")
#include "rgy_input.h"

class RGYInputAvi : public RGYInput
{
public:
    RGYInputAvi();
    virtual ~RGYInputAvi();
    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) override;

    PAVIFILE m_pAviFile;
    PAVISTREAM m_pAviStream;
    PGETFRAME m_pGetFrame;
    LPBITMAPINFOHEADER m_pBitmapInfoHeader;
    int m_nYPitchMultiplizer;

    uint32_t m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;
};

#endif //ENABLE_AVI_READER

#endif //__RGY_INPUT_AVI_H__
