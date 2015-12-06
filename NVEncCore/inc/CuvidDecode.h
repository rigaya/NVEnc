//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#ifndef __CUVID_DECODE_H__
#define __CUVID_DECODE_H__

#include <cuda.h>
#include <nvcuvid.h>
#include "NVEncLog.h"
#include "FrameQueue.h"
#include "avcodec_qsv.h"

#if ENABLE_AVCUVID_READER

bool check_if_nvcuvid_dll_available();

struct InputVideoInfo;
struct VppParam;

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
                (*m_pPrintMes)(log_level, (_T("cuvid: ") + line + _T("\n")).c_str());
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