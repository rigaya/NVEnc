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
#include "rgy_timecode.h"
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
#if ENCODER_MPP
#include "mpp_util.h"
#endif //#if ENCODER_MPP

std::vector<int> read_keyfile(tstring keyfile);

RGY_CSP csp_y4mheader_to_rgy(const char *str);
const char *csp_rgy_to_y4mheader(const RGY_CSP csp);

#if ENABLE_AVSW_READER
struct AVDemuxStream {
    int                       index;                  //音声・字幕のストリームID (libavのストリームID)
    int                       trackId;                //音声のトラックID (QSVEncC独自, 1,2,3,...)、字幕は0
    int                       subStreamId;            //通常は0、音声のチャンネルを分離する際に複製として作成
    int                       sourceFileIndex;        //audio/sub/data-source経由の場合、そのファイルインデックス
    AVStream                 *stream;                 //音声・字幕のストリーム (caption2assから字幕生成の場合、nullptrとなる)
    double                    addDelayMs;             //設定すべき遅延の量(millisecond)
    int                       lastVidIndex;           //音声の直前の相当する動画の位置
    int64_t                   extractErrExcess;       //音声抽出のあまり (音声が多くなっていれば正、足りなくなっていれば負)
    int64_t                   trimOffset;             //trimによる補正量 (stream timebase基準)
    int64_t                   aud0_fin;               //直前に有効だったパケットのpts(stream timebase基準)
    int                       appliedTrimBlock;       //trim blockをどこまで適用したか
    AVPacket                 *pktSample;              //サンプル用の音声・字幕データ
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelSelect; //入力音声の使用するチャンネル
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelOut;    //出力音声のチャンネル
    AVRational                timebase;               //streamのtimebase [stream = nullptrの場合でも使えるように]
    void                     *subtitleHeader;         //stream = nullptrの場合 caption2assのヘッダー情報 (srt形式でもass用のヘッダーが入っている)
    int                       subtitleHeaderSize;     //stream = nullptrの場合 caption2assのヘッダー情報のサイズ
    char                      lang[4];                //trackの言語情報(3文字)
    std::vector<AVPacket*>    subPacketTemporalBuffer; //字幕のタイムスタンプが入れ違いになっているのを解決する一時的なキュー

    AVDemuxStream() :
        index(0),
        trackId(0),
        subStreamId(0),
        sourceFileIndex(0),
        stream(nullptr),
        addDelayMs(0.0),
        lastVidIndex(0),
        extractErrExcess(0),
        trimOffset(0),
        aud0_fin(0),
        appliedTrimBlock(0),
        pktSample(nullptr),
        streamChannelSelect(),
        streamChannelOut(),
        timebase({ 0, 0 }),
        subtitleHeader(nullptr),
        subtitleHeaderSize(0),
        lang(),
        subPacketTemporalBuffer() {
    };
};

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
    int dst_uv_pitch_byte;
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
    funcConvertCSP m_alpha;
    int m_threads;
    std::vector<std::thread> m_th;
    std::vector<std::unique_ptr<void, handle_deleter>> m_heStart;
    std::vector<std::unique_ptr<void, handle_deleter>> m_heFin;
    std::vector<HANDLE> m_heFinCopy;
    RGYParamThread m_threadParam;
    RGYConvertCSPPrm m_prm;
public:
    RGYConvertCSP();
    RGYConvertCSP(int threads, RGYParamThread threadParam);
    ~RGYConvertCSP();
    const ConvertCSP *getFunc(RGY_CSP csp_from, RGY_CSP csp_to, RGY_SIMD simd);
    const ConvertCSP *getFunc(RGY_CSP csp_from, RGY_CSP csp_to, bool uv_only, RGY_SIMD simd);
    const ConvertCSP *getFunc() const { return m_csp; };

    int run(int interlaced, void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int dst_uv_pitch_byte, int height, int dst_height, int *crop);
};

class RGYInputPrm {
public:
    int threadCsp;
    RGY_SIMD simdCsp;
    RGYParamThread threadParamCsp;
    RGYPoolAVPacket *poolPkt;
    RGYPoolAVFrame *poolFrame;
    tstring tcfileIn;
    rgy_rational<int> timebase;

    RGYInputPrm() : threadCsp(-1), simdCsp(RGY_SIMD::NONE), threadParamCsp(), poolPkt(nullptr), poolFrame(nullptr),
        tcfileIn(), timebase() {};
    virtual ~RGYInputPrm() {};
};

class RGYInput {
public:
    RGYInput();
    virtual ~RGYInput();

    RGY_ERR Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm, shared_ptr<RGYLog> log, shared_ptr<EncodeStatus> encSatusInfo);

    RGY_ERR LoadNextFrame(RGYFrame *surface);

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

    RGY_ERR readTimecode(int64_t& pts, int64_t& duration);

    sTrimParam GetTrimParam() {
        return m_trimParam;
    }

#pragma warning(push)
#pragma warning(disable: 4100)
    virtual bool checkTimeSeekTo(int64_t pts, rgy_rational<int> timebase, float marginSec = 0.0f) {
        return true;
    }
#pragma warning(pop)

    sInputCrop GetInputCropInfo() {
        return m_inputVideoInfo.crop;
    }
    VideoInfo GetInputFrameInfo() {
        return m_inputVideoInfo;
    }
    void SetInputFrames(int frames) {
        m_inputVideoInfo.frames = frames;
    }
    virtual rgy_rational<int> getInputTimebase() const {
    	if (m_timebase.is_valid()) return m_timebase;
        auto inputFps = rgy_rational<int>(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
        return inputFps.inv() * rgy_rational<int>(1, 4);
    }
    virtual int64_t GetVideoFirstKeyPts() const {
        return -1;
    }
    virtual bool rffAware() const {
        return false;
    }
    virtual bool seekable() const {
        return false;
    }
    virtual bool timestampStable() const {
        return false;
    }
    virtual bool isPipe() const {
        return false;
    }

#if ENABLE_AVSW_READER
#pragma warning(push)
#pragma warning(disable: 4100)
    //音声・字幕パケットの配列を取得する
    virtual std::vector<AVPacket*> GetStreamDataPackets(int inputFrame) {
        return std::vector<AVPacket*>();
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
    virtual RGYDOVIProfile getInputDOVIProfile() {
        return RGY_DOVI_PROFILE_UNSET;
    }
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) = 0;
    virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const VideoInfo *inputPrm);
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *surface) = 0;

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

    std::pair<float, float> m_seek;
    sTrimParam m_trimParam;
    RGYPoolAVPacket *m_poolPkt; //AVPacketのpool
    RGYPoolAVFrame *m_poolFrame; //AVFrameのpool
    std::unique_ptr<RGYTimecodeReader> m_timecode;
    rgy_rational<int> m_timebase;
};

RGY_ERR initReaders(
    shared_ptr<RGYInput> &pFileReader,
    vector<shared_ptr<RGYInput>> &otherReaders,
    VideoInfo *input,
    const RGYParamInput *inprm,
    const RGY_CSP inputCspOfRawReader,
    const shared_ptr<EncodeStatus> pStatus,
    const RGYParamCommon *common,
    const RGYParamControl *ctrl,
    DeviceCodecCsp &HWDecCodecCsp,
    const int subburnTrackId,
    const bool vpp_afs,
    const bool vpp_rff,
    const bool vpp_require_hdr_metadata,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    RGYListRef<RGYFrameDataQP> *qpTableListRef,
    CPerfMonitor *perfMonitor,
    shared_ptr<RGYLog> log
);

#endif //__RGY_INPUT_H__

