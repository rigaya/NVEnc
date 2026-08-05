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
#pragma warning(push)
#pragma warning(disable: 4201)
#include "dynlink_nvcuvid.h"
#pragma warning(pop)
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "FrameQueue.h"
#include "NVEncParam.h"
#include "rgy_log.h"
#include "rgy_util.h"
#include "rgy_frame_info.h"
#include "rgy_avutil.h"

#if ENABLE_AVSW_READER
#define NVEncCtxAutoLock(x) CCtxAutoLock x
#else
#define NVEncCtxAutoLock(x)
#endif

#if ENABLE_AVSW_READER

bool check_if_nvcuvid_dll_available();
CodecCsp getHWDecCodecCsp(bool skipHWDecodeCheck);

struct VideoInfo;

class CuvidDecode {
public:
    CuvidDecode();
    ~CuvidDecode();

    //adaptResolutionは表示解像度で指定された初回作成上限。実際のcoded sizeへのアラインとcaps検証はInitDecode内で行う。
    //既存の内部デコーダ呼び出しを変えないため末尾の省略可能引数とし、未指定時は従来のコンテナ宣言解像度を使う。
    CUresult InitDecode(CUvideoctxlock ctxLock, const VideoInfo *input, const VppParam *vpp, AVRational streamtimebase, shared_ptr<RGYLog> pLog, int nDecType, bool bCuvidResize, bool lowLatency = false, const std::pair<int, int>& adaptResolution = { 0, 0 });
    RGY_ERR CloseDecoder();
    CUresult DecodePacket(uint8_t *data, size_t nSize, int64_t timestamp, AVRational streamtimebase);
    CUresult FlushParser();

    void* GetDecoder() { return m_videoDecoder; };

    CUVIDDECODECREATEINFO GetDecodeInfo() { return m_videoDecodeCreateInfo; };
    RGYFrameInfo GetDecFrameInfo();

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
    //入力途中の解像度変更に伴うデコーダリセットの待ち合わせ用。デコードスレッドがformatChangeReq()を立てて待機するので、
    //パイプライン側は下流のフレームがすべて解放されたことを確認してallowFormatChange()で解除する。
    bool formatChangeReq() const {
        return m_formatChangeReq.load();
    }
    void allowFormatChange();
protected:
    void AddMessage(RGYLogLevel log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel(RGY_LOGT_DEC)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, RGY_LOGT_DEC, (_T("cuvid: ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel(RGY_LOGT_DEC)) {
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
    void SetDecodeCreateInfo(CUVIDEOFORMAT *pFormat);
    CUresult CreateDecoder(CUVIDEOFORMAT *pFormat);
    CUresult ReconfigureDecoder(CUVIDEOFORMAT *pFormat);

    FrameQueue                  *m_pFrameQueue;
    int64_t                      m_decodedFrames;
    int64_t                      m_parsedPackets;
    CUvideoparser                m_videoParser;
    CUvideodecoder               m_videoDecoder;
    CUvideoctxlock               m_ctxLock;
    CUVIDDECODECREATEINFO        m_videoDecodeCreateInfo;
    CUVIDDECODECAPS              m_videoDecodeCaps;    //デコーダの対応解像度範囲。解像度変更時の上限clampに使用(bIsSupported=falseなら未取得)
    CUVIDEOFORMATEX              m_videoFormatEx;
    shared_ptr<RGYLog>           m_pPrintMes;  //ログ出力
    bool                         m_bError;
    cudaVideoDeinterlaceMode     m_deinterlaceMode;
    VideoInfo                    m_videoInfo;
    int                          m_nDecType;
    //以下4つはデコーダリセットバリア用。デコードスレッド(parserコールバック)とパイプラインスレッドの間の同期に使う
    std::atomic<bool>            m_formatChangeReq;     //リセット要求中か。パイプライン側からロックなしで見るためatomic
    std::mutex                   m_formatChangeMtx;
    std::condition_variable      m_formatChangeCv;
    bool                         m_formatChangeAllowed; //リセット許可が下りたか(mutex保護下。cvのspurious wakeup対策も兼ねる)
};

#endif //#if ENABLE_AVSW_READER

#endif //__CUVID_DECODE_H__
