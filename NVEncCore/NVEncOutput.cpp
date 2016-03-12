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
#include "NVEncOutput.h"

#define WRITE_CHECK(writtenBytes, expected) { \
    if (writtenBytes != expected) { \
        AddMessage(NV_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\n")); \
        return 1; \
    } }

NVEncOut::NVEncOut() :
    m_pEncSatusInfo(),
    m_fDest(),
    m_bOutputIsStdout(false),
    m_bInited(false),
    m_bNoOutput(false),
    m_OutType(OUT_TYPE_BITSTREAM),
    m_bSourceHWMem(false),
    m_bY4mHeaderWritten(false),
    m_strWriterName(),
    m_strOutputInfo(),
    m_pPrintMes(),
    m_pOutputBuffer(),
    m_pReadBuffer(),
    m_pUVBuffer() {
}

NVEncOut::~NVEncOut() {
    Close();
}

void NVEncOut::Close() {
    AddMessage(NV_LOG_DEBUG, _T("Closing...\n"));
    if (m_fDest) {
        m_fDest.reset();
        AddMessage(NV_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    m_pOutputBuffer.reset();
    m_pReadBuffer.reset();
    m_pUVBuffer.reset();

    m_pEncSatusInfo.reset();
    m_bNoOutput = false;
    m_bInited = false;
    m_bSourceHWMem = false;
    m_bY4mHeaderWritten = false;
    AddMessage(NV_LOG_DEBUG, _T("Closed.\n"));
    m_pPrintMes.reset();
}

NVEncOutBitstream::NVEncOutBitstream() {
    m_strWriterName = _T("bitstream");
    m_OutType = OUT_TYPE_BITSTREAM;
}

NVEncOutBitstream::~NVEncOutBitstream() {
}

#pragma warning(push)
#pragma warning(disable:4100)
int NVEncOutBitstream::Init(const TCHAR *strFileName, const void *prm, shared_ptr<EncodeStatus> pEncSatusInfo) {
    CQSVOutRawPrm *rawPrm = (CQSVOutRawPrm *)prm;
    if (!rawPrm->bBenchmark && _tcslen(strFileName) == 0) {
        AddMessage(NV_LOG_ERROR, _T("output filename not set.\n"));
        return 1;
    }

    Close();

    m_pEncSatusInfo = pEncSatusInfo;

    if (rawPrm->bBenchmark) {
        m_bNoOutput = true;
        AddMessage(NV_LOG_DEBUG, _T("no output for benchmark mode.\n"));
    } else {
        if (_tcscmp(strFileName, _T("-")) == 0) {
            m_fDest.reset(stdout);
            m_bOutputIsStdout = true;
            AddMessage(NV_LOG_DEBUG, _T("using stdout\n"));
        } else {
            CreateDirectoryRecursive(PathRemoveFileSpecFixed(strFileName).second.c_str());
            FILE *fp = NULL;
            int error = _tfopen_s(&fp, strFileName, _T("wb+"));
            if (error != 0 || fp == NULL) {
                AddMessage(NV_LOG_ERROR, _T("failed to open output file \"%s\": %s\n"), strFileName, _tcserror(error));
                return 1;
            }
            m_fDest.reset(fp);
            AddMessage(NV_LOG_DEBUG, _T("Opened file \"%s\"\n"), strFileName);

            int bufferSizeByte = clamp(rawPrm->nBufSizeMB, 0, NV_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
            if (bufferSizeByte) {
                void *ptr = nullptr;
                bufferSizeByte = (int)malloc_degeneracy(&ptr, bufferSizeByte, 1024 * 1024);
                if (bufferSizeByte) {
                    m_pOutputBuffer.reset((char*)ptr);
                    setvbuf(m_fDest.get(), m_pOutputBuffer.get(), _IOFBF, bufferSizeByte);
                    AddMessage(NV_LOG_DEBUG, _T("Added %d MB output buffer.\n"), bufferSizeByte / (1024 * 1024));
                }
            }
        }
    }
    m_bInited = true;
    return 0;
}
int NVEncOutBitstream::SetVideoParam(const InputVideoInfo *pVideoPrm) { return 0; };
#pragma warning(pop)
int NVEncOutBitstream::WriteNextFrame(const NV_ENC_LOCK_BITSTREAM *pBitstream) {
    if (pBitstream == nullptr) {
        AddMessage(NV_LOG_ERROR, _T("Invalid call: WriteNextFrame\n"));
        return 1;
    }

    uint32_t nBytesWritten = 0;
    if (!m_bNoOutput) {
        nBytesWritten = (uint32_t)fwrite(pBitstream->bitstreamBufferPtr, 1, pBitstream->bitstreamSizeInBytes, m_fDest.get());
        WRITE_CHECK(nBytesWritten, pBitstream->bitstreamSizeInBytes);
    }

    m_pEncSatusInfo->SetOutputData(pBitstream);
    return 0;
}

#pragma warning(push)
#pragma warning(disable: 4100)
int NVEncOutBitstream::WriteNextFrame(uint8_t *ptr, uint32_t nSize) {
    return 1;
}
#pragma warning(pop)
