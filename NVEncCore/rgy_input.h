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
#ifndef __RGY_INPUT_H__
#define __RGY_INPUT_H__

#include <memory>
#include <thread>
#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_log.h"
#include "rgy_event.h"
#include "rgy_status.h"
#include "convert_csp.h"
#include "rgy_err.h"
#include "rgy_util.h"
#include "rgy_prm.h"
#include "rgy_avutil.h"
#include "rgy_frame.h"
#if ENCODER_NVENC
#include "NVEncUtil.h"
#endif //#if ENCODER_NVENC
#if ENCODER_QSV
#include "qsv_util.h"
#endif //#if ENCODER_QSV
#if ENCODER_VCEENC
#include "vce_util.h"
#endif //#if ENCODER_VCEENC

std::vector<int> read_keyfile(tstring keyfile);

#if ENABLE_AVSW_READER
typedef struct AVDemuxStream {
    int                       index;                  //音声・字幕のストリームID (libavのストリームID)
    int                       trackId;                //音声のトラックID (QSVEncC独自, 1,2,3,...)、字幕は0
    int                       subStreamId;            //通常は0、音声のチャンネルを分離する際に複製として作成
    int                       sourceFileIndex;        //audio/sub/data-source経由の場合、そのファイルインデックス
    AVStream                 *stream;                 //音声・字幕のストリーム (caption2assから字幕生成の場合、nullptrとなる)
    int                       addDelayMs;             //設定すべき遅延の量(millisecond)
    int                       lastVidIndex;           //音声の直前の相当する動画の位置
    int64_t                   extractErrExcess;       //音声抽出のあまり (音声が多くなっていれば正、足りなくなっていれば負)
    int64_t                   trimOffset;             //trimによる補正量 (stream timebase基準)
    int64_t                   aud0_fin;               //直前に有効だったパケットのpts(stream timebase基準)
    int                       appliedTrimBlock;       //trim blockをどこまで適用したか
    AVPacket                  pktSample;              //サンプル用の音声・字幕データ
    uint64_t                  streamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t                  streamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル
    AVRational                timebase;               //streamのtimebase [stream = nullptrの場合でも使えるように]
    void                     *subtitleHeader;         //stream = nullptrの場合 caption2assのヘッダー情報 (srt形式でもass用のヘッダーが入っている)
    int                       subtitleHeaderSize;     //stream = nullptrの場合 caption2assのヘッダー情報のサイズ
    C2AFormat                 caption2ass;            //stream = nullptrの場合 caption2assのformat
    char                      lang[4];                //trackの言語情報(3文字)
} AVDemuxStream;

static int trackFullID(AVMediaType media_type, int trackID) {
    return (((uint32_t)media_type) << 12) | trackID;
}
static AVMediaType trackMediaType(int trackID) {
    return (AVMediaType)((((uint32_t)trackID) & 0xf000) >> 12);
}
static int trackID(int trackID) {
    return (int)(((uint32_t)trackID) & 0x0fff);
}
static const char *trackMediaTypeStr(int trackID) {
    return av_get_media_type_string(trackMediaType(trackID));
}
#endif //#if ENABLE_AVSW_READER

struct RGYConvertCSPPrm {
    bool abort;
    void **dst;
    const void **src;
    int interlaced;
    int width;
    int src_y_pitch_byte;
    int src_uv_pitch_byte;
    int dst_y_pitch_byte;
    int height;
    int dst_height;
    int *crop;

    RGYConvertCSPPrm();
};

class RGYConvertCSP {
private:
    const ConvertCSP *m_csp;
    RGY_CSP m_csp_from;
    RGY_CSP m_csp_to;
    bool m_uv_only;
    int m_threads;
    std::vector<std::thread> m_th;
    std::vector<std::unique_ptr<void, handle_deleter>> m_heStart;
    std::vector<std::unique_ptr<void, handle_deleter>> m_heFin;
    std::vector<HANDLE> m_heFinCopy;
    RGYConvertCSPPrm m_prm;
public:
    RGYConvertCSP();
    RGYConvertCSP(int threads);
    ~RGYConvertCSP();
    const ConvertCSP *getFunc(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, uint32_t simd);
    const ConvertCSP *getFunc() const { return m_csp; };

    int run(int interlaced, void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);
};

class RGYInputPrm {
public:
    int threadCsp;
    uint32_t simdCsp;

    RGYInputPrm() : threadCsp(-1), simdCsp(0) {};
    virtual ~RGYInputPrm() {};
};

class RGYInput {
public:
    RGYInput();
    virtual ~RGYInput();

    RGY_ERR Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm, shared_ptr<RGYLog> log, shared_ptr<EncodeStatus> encSatusInfo) {
        Close();
        m_printMes = log;
        m_encSatusInfo = encSatusInfo;
        return Init(strFileName, inputInfo, prm);
    };

    virtual RGY_ERR LoadNextFrame(RGYFrame *surface) = 0;

#pragma warning(push)
#pragma warning(disable: 4100)
    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは消す)
    virtual RGY_ERR GetNextBitstream(RGYBitstream *bitstream) {
        return RGY_ERR_NONE;
    }

    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは残す)
    virtual RGY_ERR GetNextBitstreamNoDelete(RGYBitstream *bitstream) {
        return RGY_ERR_NONE;
    }

    //ストリームのヘッダ部分を取得する
    virtual RGY_ERR GetHeader(RGYBitstream *bitstream) {
        return RGY_ERR_NONE;
    }
#pragma warning(pop)

    virtual void Close();

    void SetTrimParam(const sTrimParam& trim) {
        m_trimParam = trim;
    }

    sTrimParam GetTrimParam() {
        return m_trimParam;
    }

    sInputCrop GetInputCropInfo() {
        return m_inputVideoInfo.crop;
    }
    VideoInfo GetInputFrameInfo() {
        return m_inputVideoInfo;
    }
    void SetInputFrames(int frames) {
        m_inputVideoInfo.frames = frames;
    }
    virtual rgy_rational<int> getInputTimebase() {
        return rgy_rational<int>();
    }

#if ENABLE_AVSW_READER
#pragma warning(push)
#pragma warning(disable: 4100)
    //音声・字幕パケットの配列を取得する
    virtual vector<AVPacket> GetStreamDataPackets(int inputFrame) {
        return vector<AVPacket>();
    }

    //音声・字幕のコーデックコンテキストを取得する
    virtual vector<AVDemuxStream> GetInputStreamInfo() {
        return vector<AVDemuxStream>();
    }
#pragma warning(pop)
#endif //#if ENABLE_AVSW_READER

    //入力ファイルに存在する音声のトラック数を返す
    virtual int GetAudioTrackCount() {
        return 0;
    }
    //入力ファイルに存在する字幕のトラック数を返す
    virtual int GetSubtitleTrackCount() {
        return 0;
    }
    //入力ファイルに存在するデータのトラック数を返す
    virtual int GetDataTrackCount() {
        return 0;
    }
    const TCHAR *GetInputMessage() {
        const TCHAR *mes = m_inputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(RGYLogLevel log_level, const tstring& str) {
        if (m_printMes == nullptr || log_level < m_printMes->getLogLevel(RGY_LOGT_IN)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_printMes->write(log_level, RGY_LOGT_IN, (m_readerName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ... ) {
        if (m_printMes == nullptr || log_level < m_printMes->getLogLevel(RGY_LOGT_IN)) {
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

    //HWデコードを行う場合のコーデックを返す
    //行わない場合はRGY_CODEC_UNKNOWNを返す
    RGY_CODEC getInputCodec() {
        return m_inputVideoInfo.codec;
    }
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) = 0;
    virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const VideoInfo *inputPrm);

    //trim listを参照し、動画の最大フレームインデックスを取得する
    int getVideoTrimMaxFramIdx() {
        if (m_trimParam.list.size() == 0) {
            return INT_MAX;
        }
        return m_trimParam.list[m_trimParam.list.size()-1].fin;
    }

    shared_ptr<EncodeStatus> m_encSatusInfo;

    VideoInfo m_inputVideoInfo;

    RGY_CSP m_inputCsp;
    unique_ptr<RGYConvertCSP> m_convert;
    shared_ptr<RGYLog> m_printMes;  //ログ出力

    tstring m_inputInfo;
    tstring m_readerName;    //読み込みの名前

    sTrimParam m_trimParam;
};

RGY_ERR initReaders(
    shared_ptr<RGYInput> &pFileReader,
    vector<shared_ptr<RGYInput>> &otherReaders,
    VideoInfo *input,
    const RGY_CSP inputCspOfRawReader,
    const shared_ptr<EncodeStatus> pStatus,
    const RGYParamCommon *common,
    const RGYParamControl *ctrl,
    DeviceCodecCsp &HWDecCodecCsp,
    const int subburnTrackId,
    const bool vpp_afs,
    const bool vpp_rff,
    RGYListRef<RGYFrameDataQP> *qpTableListRef,
    CPerfMonitor *perfMonitor,
    shared_ptr<RGYLog> log
);

#endif //__RGY_INPUT_H__

