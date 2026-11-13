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
#include <atomic>
#include <thread>
#include <utility>
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
    int                       indexCurrent;           //現在パケットが流れてくるstream index (PMT変更追従用、初期値=index)
    int                       pmtTrackPos;            //追従対象program内で同じcodec_typeの何番目か (-1: 追従対象外)
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
        indexCurrent(0),
        pmtTrackPos(-1),
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

// --preview-serverで、Viewerが指定する出力開始時刻(output_start_us)と、実際の出力フレームの時刻を
// 比べるときの許容幅(µs)。Viewerは開始時刻をµs単位に切り捨てて送ってくるうえ、Viewer側では
// 範囲の判定に2µsの許容を持ち、y4mの時刻(Xts)も小数6桁へ丸めているため最大0.5µsずれる。
// これらを合わせた2.5µsを整数に切り上げ、開始時刻よりこの幅だけ手前のフレームまでは書き出す。
// 余分に数枚書いても、Viewerが不要な分を捨てるだけなので問題ない。
static const int64_t RGY_PREVIEW_OUTPUT_START_TOLERANCE_US = 3;

// --preview-serverの1回の要求(run)の間だけ、入力の読み込み方の設定を有効にする。
//
// 入力の初期化は、パイプラインの初期化の中から(場合によっては別スレッドで)呼ばれるため、
// サーバの要求の情報を引数で直接渡す経路がない。そこで、このcppファイル内の変数を経由して渡し、
// RGYInput::configurePreviewWindow()が入力の初期化時に自分の状態(m_preview)へ写し取る。
// サーバは要求を1つずつ順に処理するので、複数のscopeが同時に有効になることはない。
// 通常のエンコードや、probe要求ではscopeを作らないので、従来の動作は変わらない。
class RGYPreviewInputScope {
    bool m_previousEnabled;
    std::atomic<bool> *m_previousAbort;
    int64_t m_previousOutputStartUs;
public:
    // 構築時に設定を有効にし、破棄時に構築前の値へ戻す。
    // enabled       : Viewerが窓の読み込み(RGYPreviewWindowを参照)を要求したrunならtrue。
    // abort         : Viewerがrunを取り消すとtrueになるフラグ。キーフレームの探索中も確認して中断する。
    // outputStartUs : Viewerが指定した出力開始時刻(µs、入力の絶対時刻)。これより前の出力はpipeへ書かない。
    //                 INT64_MINは指定なし。enabledがfalseのときは無視する。
    RGYPreviewInputScope(bool enabled, std::atomic<bool> *abort, int64_t outputStartUs = INT64_MIN);
    ~RGYPreviewInputScope();
};

// --preview-serverで、Viewerが要求した「窓」を読むための状態。
// 窓は、入力の絶対時刻で表した開始時刻(start)から、指定した枚数(limit)の入力フレームの範囲。
//
// 背景: TSなどでは、av_seek_frame()の着地位置が目標より後ろのキーフレームになることがあり、
// 従来の--seekでは窓の先頭が目標より後ろへずれて、Viewerが目標のフレームを得られないことがあった。
// そこで、開始時刻以前で最後のキーフレームを自分で探してそこから復号し(RGYInputAvcodec::seekPreviewWindow)、
// 開始時刻より前に復号されたフレームを判定して扱う(RGYInput::checkPreviewWindow)。
// 有限の枚数は、開始時刻以降の入力フレームだけで数える。
struct RGYPreviewWindow {
    bool enabled;                   // 窓の読み込みを使うか。使えない入力や指定では、従来の--seekの動作に戻す。
    std::atomic<bool> *abort;       // Viewerからの取消要求。
    int64_t outputStartUs;          // Viewerが指定した出力開始時刻(µs、入力の絶対時刻)。INT64_MINは指定なし。
    int64_t start;                  // 窓の開始時刻(timebase単位、入力の絶対時刻)。INT64_MINは未決定。
    rgy_rational<int> timebase;     // startの単位。
    int64_t keyPts;                 // 復号を始めたキーフレームの時刻(timebase単位)。INT64_MINは未決定。
    bool keepBeforeStart;           // 開始時刻より前に復号したフレームも、フィルタへ渡すか(checkPreviewWindowを参照)。
    int64_t limit;                  // 開始時刻以降に受け取る入力フレームの上限。0は上限なし。
    std::atomic<int64_t> accepted;  // 開始時刻以降に受け取った入力フレームの数。
    // 以下はDEBUGログ用の集計。
    bool report;                    // 入力を閉じるときに集計をログへ出すか。
    int64_t beforeStart;            // 開始時刻より前に復号したフレームの数。
    int64_t droppedBeforeStart;     // そのうち、フィルタへ渡さずに捨てた数。
    int64_t afterLimit;             // 上限に達した後に届いたフレームの数。

    RGYPreviewWindow() :
        enabled(false), abort(nullptr), outputStartUs(INT64_MIN), start(INT64_MIN), timebase(),
        keyPts(INT64_MIN), keepBeforeStart(false), limit(0), accepted(0),
        report(false), beforeStart(0), droppedBeforeStart(0), afterLimit(0) {}
};

// --preview-serverで、出力(RGYOutFrame)がpipeへ書く前に捨てるフレームの条件。
// 窓の開始時刻より前のフレームも、時間方向のフィルタの状態を作るためにフィルタへは通すことがあるが、
// その出力はViewerに不要なので、書き出す直前に捨てる。
struct RGYPreviewOutputCutoff {
    bool dropBeforeStart;           // 窓の開始時刻(start)より前の出力を書かない。
    int64_t start;                  // 窓の開始時刻(timebase単位、入力の絶対時刻)。
    rgy_rational<int> timebase;     // startの単位。
    int64_t outputStartUs;          // Viewerが指定した出力開始時刻(µs)。これより前の出力を書かない。INT64_MINは指定なし。

    RGYPreviewOutputCutoff() : dropBeforeStart(false), start(0), timebase(), outputStartUs(INT64_MIN) {}
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
    double seekSec; // --seek で先頭を飛ばす秒数。フレーム入力(avi/avs/vpy)で使用

    RGYInputPrm() : threadCsp(-1), simdCsp(RGY_SIMD::NONE), threadParamCsp(), poolPkt(nullptr), poolFrame(nullptr),
        tcfileIn(), timebase(), seekSec(0.0) {};
    virtual ~RGYInputPrm() {};
};

// avi/avs/vpy のように任意フレームを読める入力の開始・終了位置。
// userFrames は --frames (0は無指定)、srcFrames はソースの総フレーム数。
// 戻り値は [開始フレーム, 終了フレーム)。終了フレームは含まない。
// --seek なしのときは、従来どおり --frames で切った範囲を seekRatio で分割する。
// --seek ありのときは、ソース先頭からの時刻をフレームに直し、その位置から --frames 枚まで読む。
// seekRatio は並列エンコード用で、--seek 後に残った範囲をさらに分割する。
inline std::pair<int, int> rgy_apply_frame_seek(const double seekSec, const float seekRatio, const int userFrames, const int srcFrames, const int fpsN, const int fpsD) {
    const int src = (srcFrames > 0) ? srcFrames : 0;
    if (seekSec > 0.0 && fpsN > 0 && fpsD > 0 && src > 0) {
        // frame n の時刻は n * fpsD / fpsN 秒。時刻 >= seekSec になる最初のフレーム。
        // 秒をマイクロ秒の整数にしてから切り上げ、fps が割り切れないときの端数で1フレームずれないようにする。
        // ちょうどフレームの時刻 (例: 30fpsで17/30秒) はマイクロ秒への丸めで最大0.5us後ろへずれ、
        // そのまま切り上げると次のフレームになる。1us以内の差は同じ時刻とみなすため、1us引いてから切り上げる。
        const int64_t seekUs = (int64_t)(seekSec * 1000000.0 + 0.5);
        const int64_t den = (int64_t)fpsD * 1000000;
        const int64_t num = (seekUs - 1) * (int64_t)fpsN;
        int64_t start64 = (den > 0) ? (num + den - 1) / den : 0;
        if (start64 < 0) start64 = 0;
        if (start64 > src) start64 = src;
        int start = (int)start64;
        if (seekRatio > 0.0f && src > start) {
            start += (int)(seekRatio * (float)(src - start));
            if (start > src) start = src;
        }
        int end = src;
        if (userFrames > 0) {
            const int64_t limited = (int64_t)start + (int64_t)userFrames;
            if (limited < (int64_t)end) end = (int)limited;
        }
        return std::make_pair(start, end);
    }
    int end = src;
    if (userFrames > 0 && userFrames < end) end = userFrames;
    int start = 0;
    if (seekRatio > 0.0f && end > 0) {
        start = (int)(seekRatio * (float)end);
        if (start < 0) start = 0;
        if (start > end) start = end;
    }
    return std::make_pair(start, end);
}

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
    virtual RGY_ERR GetNextBitstreamNoDelete(RGYBitstream *bitstream, int idx) {
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

    // 入力の初期化時に呼び、RGYPreviewInputScopeで有効にされた設定を取り込む。
    // frames: --framesで指定された入力フレームの上限(0は上限なし)。窓では開始時刻以降の枚数として扱う。
    void configurePreviewWindow(int64_t frames);
    // 出力(RGYOutFrame)がpipeへ書く前に捨てるフレームの条件を返す。
    RGYPreviewOutputCutoff previewOutputCutoff() const;
    bool previewWindowEnabled() const { return m_preview.enabled; }
    // 開始時刻以降の入力フレームを上限まで受け取ったか。以降は読み込みを打ち切る。
    bool previewWindowFinished() const { return m_preview.limit > 0 && m_preview.accepted.load() >= m_preview.limit; }
    bool previewAbortRequested() const { return m_preview.abort && m_preview.abort->load(); }
    // 復号したフレームを、フィルタへ渡すか(keep)を判定し、窓の枚数を数える。
    // pts/timebase: 復号したフレームの時刻(入力の絶対時刻)。窓を使わない場合は常にkeep=trueを返す。
    RGY_ERR checkPreviewWindow(int64_t pts, rgy_rational<int> timebase, bool& keep);

    sInputCrop GetInputCropInfo() {
        return m_inputVideoInfo.crop;
    }
    VideoInfo GetInputFrameInfo() {
        return m_inputVideoInfo;
    }
    //実際に処理するフレーム情報と、ワークサーフェスの確保容量を分離するための取得口。
    //通常の入力は両者が同じだが、可変解像度入力では最大解像度で確保しつつ、
    //フィルタには各時点の実解像度を渡す必要がある。GetInputFrameInfo()を置き換えてはいけない。
    virtual VideoInfo GetInputFrameInfoForAlloc() {
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
    // CreateInputInfo の後に呼ぶ。開始フレームの時刻を入力情報へ足す。
    void addSeekToInputInfo(int startFrame);
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *surface) = 0;

    //trim listを参照し、動画の最大フレームインデックスを取得する
    int getVideoTrimMaxFramIdx() {
        if (m_trimParam.list.size() == 0) {
            return INT_MAX;
        }
        return m_trimParam.list[m_trimParam.list.size()-1].fin;
    }

    RGYPreviewWindow m_preview; // --preview-serverの窓の状態。

    shared_ptr<EncodeStatus> m_encSatusInfo;

    VideoInfo m_inputVideoInfo;

    RGY_CSP m_inputCsp;
    unique_ptr<RGYConvertCSP> m_convert;
    shared_ptr<RGYLog> m_printMes;  //ログ出力

    tstring m_inputInfo;
    tstring m_readerName;    //読み込みの名前

    std::pair<double, double> m_seek;
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
    const bool vpp_ivtc_expand_active,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    RGYListRef<RGYFrameDataQP> *qpTableListRef,
    CPerfMonitor *perfMonitor,
    shared_ptr<RGYLog> log
);

#endif //__RGY_INPUT_H__

