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

#pragma once
#ifndef __CUVID_DECODE_H__
#define __CUVID_DECODE_H__

#include <cuda.h>
#include <nvcuvid.h>
#include "NVEncLog.h"
#include "FrameQueue.h"
#include "avcodec_qsv.h"
#include "NVEncFrameInfo.h"

#if ENABLE_AVCUVID_READER
#define NVEncCtxAutoLock(x) CCtxAutoLock x
#else
#define NVEncCtxAutoLock(x)
#endif

#if ENABLE_AVCUVID_READER

bool check_if_nvcuvid_dll_available();

struct InputVideoInfo;

class CuvidDecode {
public:
    CuvidDecode();
    ~CuvidDecode();

    CUresult InitDecode(CUvideoctxlock ctxLock, const InputVideoInfo *input, const VppParam *vpp, shared_ptr<CNVEncLog> pLog, bool ignoreDynamicFormatChange = false);
    void CloseDecoder();
    CUresult DecodePacket(uint8_t *data, size_t nSize, int64_t timestamp, AVRational streamtimebase);
    CUresult FlushParser();

    void* GetDecoder() { return m_videoDecoder; };

    CUVIDDECODECREATEINFO GetDecodeInfo() { return m_videoDecodeCreateInfo; };
    FrameInfo GetDecFrameInfo();

    bool GetError() { return m_bError; };

    int DecVideoData(CUVIDSOURCEDATAPACKET* pPacket);
    int DecPictureDecode(CUVIDPICPARAMS* pPicParams);
    int DecVideoSequence(CUVIDEOFORMAT* pFormat);
    int DecPictureDisplay(CUVIDPARSERDISPINFO* pPicParams);
    cudaVideoDeinterlaceMode getDeinterlaceMode() {
        return m_deinterlaceMode;
    }
    FrameQueue *frameQueue() {
        return m_pFrameQueue;
    }
protected:
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (_T("cuvid: ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }

    CUresult CreateDecoder();

    FrameQueue                  *m_pFrameQueue;
    int                          m_decodedFrames;
    CUvideoparser                m_videoParser;
    CUvideodecoder               m_videoDecoder;
    CUvideoctxlock               m_ctxLock;
    CUVIDDECODECREATEINFO        m_videoDecodeCreateInfo;
    CUVIDEOFORMATEX              m_videoFormatEx;
    shared_ptr<CNVEncLog>        m_pPrintMes;  //ログ出力
    bool                         m_bIgnoreDynamicFormatChange;
    bool                         m_bError;
    cudaVideoDeinterlaceMode     m_deinterlaceMode;
};

#endif //#if ENABLE_AVCUVID_READER

#endif //__CUVID_DECODE_H__