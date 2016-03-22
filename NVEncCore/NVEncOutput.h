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

#include <tchar.h>
#include <memory>
#include <vector>
#include "NVEncUtil.h"
#include "NVEncInput.h"
#include "NVEncLog.h"

using std::unique_ptr;
using std::shared_ptr;

struct nvBitstream {
    uint8_t *Data;
    uint32_t DataLength;
    uint32_t MaxLength;
    uint32_t frameIdx;
    uint64_t outputTimeStamp;
    uint64_t outputDuration;
    NV_ENC_PIC_TYPE pictureType;
    NV_ENC_PIC_STRUCT pictureStruct;
    uint32_t frameAvgQP;
};

static inline void nvBitstreamClear(nvBitstream *pBitstream) {
    if (pBitstream->Data) {
        _aligned_free(pBitstream->Data);
    }
    memset(pBitstream, 0, sizeof(pBitstream[0]));
}

static inline void nvBitstreamFreeData(nvBitstream *pBitstream) {
    if (pBitstream->Data) {
        _aligned_free(pBitstream->Data);
    }
    pBitstream->DataLength = 0;
    pBitstream->MaxLength = 0;
}

static inline int nvBitstreamAlloc(nvBitstream *pBitstream, uint32_t nSize) {
    nvBitstreamFreeData(pBitstream);
    if (nullptr == (pBitstream->Data = (uint8_t *)_aligned_malloc(nSize, 32))) {
        return 1;
    }

    pBitstream->MaxLength = nSize;
    return 0;
}

static inline int nvBitstreamInit(nvBitstream *pBitstream, uint32_t nSize) {
    nvBitstreamClear(pBitstream);
    return nvBitstreamAlloc(pBitstream, nSize);
}

static inline int nvBitstreamCopy(nvBitstream *pBitstreamCopy, const nvBitstream *pBitstream) {
    uint8_t *ptr    = pBitstreamCopy->Data;
    auto nMaxLength = pBitstreamCopy->MaxLength;
    if (nMaxLength < pBitstream->DataLength) {
        nMaxLength = pBitstream->DataLength;
        auto sts = nvBitstreamAlloc(pBitstreamCopy, nMaxLength);
        if (sts) return sts;
        ptr = pBitstreamCopy->Data;
    }
    memcpy(pBitstreamCopy, pBitstream, sizeof(pBitstreamCopy[0]));
    pBitstreamCopy->Data = ptr;
    pBitstreamCopy->MaxLength = nMaxLength;
    memcpy(pBitstreamCopy->Data, pBitstream->Data, pBitstream->DataLength);
    pBitstreamCopy->DataLength = pBitstream->DataLength;
    return 0;
}

static inline int nvBitstreamCopy(nvBitstream *pBitstreamCopy, const NV_ENC_LOCK_BITSTREAM *pBitstream) {
    nvBitstream bitstreamSrc;
    bitstreamSrc.Data = (uint8_t *)pBitstream->bitstreamBufferPtr;
    bitstreamSrc.DataLength = pBitstream->bitstreamSizeInBytes;
    bitstreamSrc.MaxLength = pBitstream->bitstreamSizeInBytes;
    bitstreamSrc.frameIdx = pBitstream->frameIdx;
    bitstreamSrc.outputTimeStamp = pBitstream->outputTimeStamp;
    bitstreamSrc.outputDuration = pBitstream->outputDuration;
    bitstreamSrc.pictureStruct = pBitstream->pictureStruct;
    bitstreamSrc.pictureType = pBitstream->pictureType;
    bitstreamSrc.frameAvgQP = pBitstream->frameAvgQP;
    return nvBitstreamCopy(pBitstreamCopy, &bitstreamSrc);
}

static inline int nvBitstreamExtend(nvBitstream *pBitstream, uint32_t nSize) {
    uint8_t *pData = (uint8_t *)_aligned_malloc(nSize, 32);
    if (nullptr == pData) {
        return 1;
    }

    auto nDataLen = pBitstream->DataLength;
    if (nDataLen) {
        memmove(pData, pBitstream->Data, nDataLen);
    }

    nvBitstreamFreeData(pBitstream);

    pBitstream->Data       = pData;
    pBitstream->DataLength = nDataLen;
    pBitstream->MaxLength  = nSize;

    return 0;
}

static inline int nvBitstreamAppend(nvBitstream *pBitstream, const uint8_t *data, uint32_t size) {
    int sts = 0;
    if (data && size) {
        const uint32_t new_data_length = pBitstream->DataLength + size;
        if (pBitstream->MaxLength < new_data_length) {
            if (0 != (sts = nvBitstreamExtend(pBitstream, new_data_length))) {
                return sts;
            }
        }
        memcpy(pBitstream->Data + pBitstream->DataLength, data, size);
        pBitstream->DataLength = new_data_length;
    }
    return sts;
}

enum OutputType {
    OUT_TYPE_NONE = 0,
    OUT_TYPE_BITSTREAM,
    OUT_TYPE_SURFACE
};

class NVEncOut {
public:
    NVEncOut();
    virtual ~NVEncOut();

    virtual void SetNVEncLogPtr(shared_ptr<CNVEncLog> pLog) {
        m_pPrintMes = pLog;
    }
    virtual int Init(const TCHAR *strFileName, const void *prm, shared_ptr<EncodeStatus> pEncSatusInfo) = 0;

    virtual int SetVideoParam(const NV_ENC_CONFIG *pEncConfig, NV_ENC_PIC_STRUCT pic_struct, const NV_ENC_SEQUENCE_PARAM_PAYLOAD *pSequenceParam) = 0;

    virtual int WriteNextFrame(const NV_ENC_LOCK_BITSTREAM *pBitstream) = 0;
    virtual int WriteNextFrame(uint8_t *ptr, uint32_t nSize) = 0;
    virtual void Close();

    virtual bool outputStdout() {
        return m_bOutputIsStdout;
    }

    virtual OutputType getOutType() {
        return m_OutType;
    }

    const TCHAR *GetOutputMessage() {
        const TCHAR *mes = m_strOutputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_strWriterName + _T(": ") + line + _T("\n")).c_str());
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
protected:
    shared_ptr<EncodeStatus> m_pEncSatusInfo;
    unique_ptr<FILE, fp_deleter>  m_fDest;
    bool        m_bOutputIsStdout;
    bool        m_bInited;
    bool        m_bNoOutput;
    OutputType  m_OutType;
    bool        m_bSourceHWMem;
    bool        m_bY4mHeaderWritten;
    tstring     m_strWriterName;
    tstring     m_strOutputInfo;
    shared_ptr<CNVEncLog> m_pPrintMes;  //ログ出力
    unique_ptr<char, malloc_deleter>            m_pOutputBuffer;
    unique_ptr<uint8_t, aligned_malloc_deleter> m_pReadBuffer;
    unique_ptr<uint8_t, aligned_malloc_deleter>   m_pUVBuffer;
};

struct CQSVOutRawPrm {
    bool bBenchmark;
    int nBufSizeMB;
};

class NVEncOutBitstream : public NVEncOut {
public:

    NVEncOutBitstream();
    virtual ~NVEncOutBitstream();

    virtual int Init(const TCHAR *strFileName, const void *prm, shared_ptr<EncodeStatus> pEncSatusInfo) override;

    virtual int SetVideoParam(const NV_ENC_CONFIG *pEncConfig, NV_ENC_PIC_STRUCT pic_struct, const NV_ENC_SEQUENCE_PARAM_PAYLOAD *pSequenceParam) override;

    virtual int WriteNextFrame(const NV_ENC_LOCK_BITSTREAM *pBitstream) override;
    virtual int WriteNextFrame(uint8_t *ptr, uint32_t nSize) override;
};
