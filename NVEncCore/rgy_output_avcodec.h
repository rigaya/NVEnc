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
#ifndef __RGY_OUTPUT_AVCODEC_H__
#define __RGY_OUTPUT_AVCODEC_H__

#include "rgy_queue.h"
#include "rgy_version.h"

#if ENABLE_AVSW_READER
#include <thread>
#include <atomic>
#include <cstdint>
#include "rgy_avutil.h"
#include "rgy_bitstream.h"
#include "rgy_input_avcodec.h"
#include "rgy_output.h"
#include "rgy_perf_monitor.h"
#include "rgy_util.h"
#include "NVEncUtil.h"

using std::vector;

#define USE_CUSTOM_IO 1

static const int SUB_ENC_BUF_MAX_SIZE = 1024 * 1024;

static const int VID_BITSTREAM_QUEUE_SIZE_I  = 4;
static const int VID_BITSTREAM_QUEUE_SIZE_PB = 64;

struct AVMuxTimestamp {
    int64_t timestamp_list[8];

    void add(int64_t timestamp) {
        for (int idx = 0; idx < _countof(timestamp_list); idx++) {
            if (timestamp_list[idx] == AV_NOPTS_VALUE) {
                timestamp_list[idx] = timestamp; //空きエントリに格納
                break;
            }
        }
    }
    int64_t get_min_pts() {
        int64_t min_pts = std::numeric_limits<int64_t>::max();
        int idx = -1;
        for (int i = 0; i < _countof(timestamp_list); i++) {
            if (timestamp_list[i] != AV_NOPTS_VALUE) {
                //空きエントリ以外の最小のptsを探す
                if (timestamp_list[i] < min_pts) {
                    min_pts = timestamp_list[i];
                    idx = i;
                }
            }
        }
        //使ったtimestampは空きエントリにする
        timestamp_list[idx] = AV_NOPTS_VALUE;
        return min_pts;
    }
    void clear() {
        for (int i = 0; i < _countof(timestamp_list); i++) {
            timestamp_list[i] = AV_NOPTS_VALUE; //空きエントリ
        }
    }
};

typedef struct AVMuxFormat {
    const TCHAR          *pFilename;            //出力ファイル名
    AVFormatContext      *pFormatCtx;           //出力ファイルのformatContext
    char                  metadataStr[256];     //出力ファイルのエンコーダ名
    AVOutputFormat       *pOutputFmt;           //出力ファイルのoutputFormat

#if USE_CUSTOM_IO
    uint8_t              *pAVOutBuffer;         //avio_alloc_context用のバッファ
    uint32_t              nAVOutBufferSize;     //avio_alloc_context用のバッファサイズ
    FILE                 *fpOutput;             //出力ファイルポインタ
    char                 *pOutputBuffer;        //出力ファイルポインタ用のバッファ
    uint32_t              nOutputBufferSize;    //出力ファイルポインタ用のバッファサイズ
#endif //USE_CUSTOM_IO
    bool                  bStreamError;         //エラーが発生
    bool                  bIsMatroska;          //mkvかどうか
    bool                  bIsPipe;              //パイプ出力かどうか
    bool                  bFileHeaderWritten;   //ファイルヘッダを出力したかどうか
    AVDictionary         *pHeaderOptions;       //ヘッダオプション
} AVMuxFormat;

typedef struct AVMuxVideo {
    AVCodec              *pCodec;               //出力映像のCodec
    AVRational            nFPS;                 //出力映像のフレームレート
    AVStream             *pStreamOut;           //出力ファイルの映像ストリーム
    bool                  bDtsUnavailable;      //出力映像のdtsが無効 (API v1.6以下)
    AVRational            inputStreamTimebase;  //入力streamのtimebase
    int64_t               nInputFirstKeyPts;    //入力映像の最初のpts
    AVRational            rBitstreamTimebase;   //エンコーダのtimebase
    AVMuxTimestamp        timestampList;        //エンコーダから渡されたtimestampリスト
    int                   nFpsBaseNextDts;      //出力映像のfpsベースでのdts (API v1.6以下でdtsが計算されない場合に使用する)
    FILE                 *fpTsLogFile;          //mux timestampログファイル
    RGYBitstream          seiNal;               //追加のsei nal
    AVBSFContext         *pBsfc;                //必要なら使用するbitstreamfilter
    RGYTimestamp         *pTimestamp;           //timestampの情報
} AVMuxVideo;

typedef struct AVMuxAudio {
    int                   nInTrackId;           //ソースファイルの入力トラック番号
    int                   nInSubStream;         //ソースファイルの入力サブストリーム番号
    const AVStream       *pStreamIn;            //入力音声のストリーム
    int                   nStreamIndexIn;       //入力音声のStreamのindex
    AVStream             *pStreamOut;           //出力ファイルの音声ストリーム
    int                   nPacketWritten;       //出力したパケットの数
    int64_t               dec_rescale_delta;    //decode時のtimebase変換用

    //変換用
    AVCodec              *pOutCodecDecode;      //変換する元のコーデック
    AVCodecContext       *pOutCodecDecodeCtx;   //変換する元のCodecContext
    AVCodec              *pOutCodecEncode;      //変換先の音声のコーデック
    AVCodecContext       *pOutCodecEncodeCtx;   //変換先の音声のCodecContext
    uint32_t              nIgnoreDecodeError;   //デコード時に連続して発生したエラー回数がこの閾値を以下なら無視し、無音に置き換える
    uint32_t              nDecodeError;         //デコード処理中に連続してエラーが発生した回数
    bool                  bEncodeError;         //エンコード処理中にエラーが発生
    int64_t               nDecodeNextPts;       //デコードの次のpts (samplerateベース)

    //filter
    int                   nFilterInChannels;      //現在のchannel数      (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    uint64_t              nFilterInChannelLayout; //現在のchannel_layout (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    int                   nFilterInSampleRate;    //現在のsampling rate  (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    AVSampleFormat        FilterInSampleFmt;      //現在のSampleformat   (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    const TCHAR          *pFilter;
    AVFilterContext      *pFilterBufferSrcCtx;
    AVFilterContext      *pFilterBufferSinkCtx;
    AVFilterContext      *pFilterAudioFormat;
    AVFilterGraph        *pFilterGraph;

    //resampler
    int                   nAudioResampler;      //resamplerの選択 (QSV_RESAMPLER_xxx)
    AVFrame              *pDecodedFrameCache;   //デコードされたデータのキャッシュされたもの
    int                   channelMapping[MAX_SPLIT_CHANNELS];        //resamplerで使用するチャンネル割り当て(入力チャンネルの選択)
    uint64_t              pnStreamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t              pnStreamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル

    //AACの変換用
    AVBSFContext         *pAACBsfc;             //必要なら使用するbitstreamfilter
    int                   nAACBsfErrorFromStart; //開始直後からのbitstream filter errorの数

    int                   nOutputSamples;       //出力音声の出力済みsample数
    int64_t               nLastPtsIn;           //入力音声の前パケットのpts (input stream timebase)
    int64_t               nLastPtsOut;          //出力音声の前パケットのpts
} AVMuxAudio;

typedef struct AVMuxSub {
    int                   nInTrackId;           //ソースファイルの入力トラック番号
    const AVStream       *pStreamIn;            //入力字幕のストリーム
    int                   nStreamIndexIn;       //入力字幕のStreamのindex
    AVRational            streamInTimebase;     //入力字幕のストリームのtimebase
    AVStream             *pStreamOut;           //出力ファイルの字幕ストリーム

    //変換用
    AVCodec              *pOutCodecDecode;      //変換する元のコーデック
    AVCodecContext       *pOutCodecDecodeCtx;   //変換する元のCodecContext
    AVCodec              *pOutCodecEncode;      //変換先の音声のコーデック
    AVCodecContext       *pOutCodecEncodeCtx;   //変換先の音声のCodecContext

    uint8_t              *pBuf;                 //変換用のバッファ
} AVMuxSub;

enum {
    MUX_DATA_TYPE_NONE   = 0,
    MUX_DATA_TYPE_PACKET = 1, //AVPktMuxDataに入っているデータがAVPacket
    MUX_DATA_TYPE_FRAME  = 2, //AVPktMuxDataに入っているデータがAVFrame
};

typedef struct AVPktMuxData {
    int         type;        //MUX_DATA_TYPE_xxx
    AVPacket    pkt;         //type == MUX_DATA_TYPE_PACKET 時有効
    AVMuxAudio *pMuxAudio;   //type == MUX_DATA_TYPE_PACKET 時有効
    int64_t     dts;         //type == MUX_DATA_TYPE_PACKET 時有効
    int         samples;     //type == MUX_DATA_TYPE_PACKET 時有効
    AVFrame    *pFrame;      //type == MUX_DATA_TYPE_FRAME 時有効
    int         got_result;  //type == MUX_DATA_TYPE_FRAME 時有効
} AVPktMuxData;

enum {
    AUD_QUEUE_PROCESS = 0,
    AUD_QUEUE_ENCODE  = 1,
    AUD_QUEUE_OUT     = 2,
};

#if ENABLE_AVCODEC_OUT_THREAD
typedef struct AVMuxThread {
    bool                           bEnableOutputThread;       //出力スレッドを使用する
    bool                           bEnableAudProcessThread;   //音声処理スレッドを使用する
    bool                           bEnableAudEncodeThread;    //音声エンコードスレッドを使用する
    std::atomic<bool>              bAbortOutput;              //出力スレッドに停止を通知する
    std::thread                    thOutput;                  //出力スレッド(mux部分を担当)
    std::atomic<bool>              bThAudProcessAbort;        //音声処理スレッドに停止を通知する
    std::thread                    thAudProcess;              //音声処理スレッド(デコード/thAudEncodeがなければエンコードも担当)
    std::atomic<bool>              bThAudEncodeAbort;         //音声エンコードスレッドに停止を通知する
    std::thread                    thAudEncode;               //音声エンコードスレッド(エンコードを担当)
    HANDLE                         heEventPktAddedOutput;     //キューのいずれかにデータが追加されたことを通知する
    HANDLE                         heEventClosingOutput;      //出力スレッドが停止処理を開始したことを通知する
    HANDLE                         heEventPktAddedAudProcess; //キューのいずれかにデータが追加されたことを通知する
    HANDLE                         heEventClosingAudProcess;  //音声処理スレッドが停止処理を開始したことを通知する
    HANDLE                         heEventPktAddedAudEncode;  //キューのいずれかにデータが追加されたことを通知する
    HANDLE                         heEventClosingAudEncode;   //音声処理スレッドが停止処理を開始したことを通知する
    RGYQueueSPSP<RGYBitstream, 64> qVideobitstreamFreeI;      //映像 Iフレーム用に空いているデータ領域を格納する
    RGYQueueSPSP<RGYBitstream, 64> qVideobitstreamFreePB;     //映像 P/Bフレーム用に空いているデータ領域を格納する
    RGYQueueSPSP<RGYBitstream, 64> qVideobitstream;           //映像パケットを出力スレッドに渡すためのキュー
    RGYQueueSPSP<AVPktMuxData, 64> qAudioPacketProcess;       //処理前音声パケットをデコード/エンコードスレッドに渡すためのキュー
    RGYQueueSPSP<AVPktMuxData, 64> qAudioFrameEncode;         //デコード済み音声フレームをエンコードスレッドに渡すためのキュー
    RGYQueueSPSP<AVPktMuxData, 64> qAudioPacketOut;           //音声パケットを出力スレッドに渡すためのキュー
    PerfQueueInfo                 *pQueueInfo;                //キューの情報を格納する構造体
} AVMuxThread;
#endif

typedef struct AVMux {
    AVMuxFormat         format;
    AVMuxVideo          video;
    vector<AVMuxAudio>  audio;
    vector<AVMuxSub>    sub;
    vector<sTrim>       trim;
#if ENABLE_AVCODEC_OUT_THREAD
    AVMuxThread         thread;
#endif
} AVMux;

typedef struct AVOutputStreamPrm {
    AVDemuxStream src;                 //入力音声・字幕の情報
    const TCHAR  *pEncodeCodec;        //音声をエンコードするコーデック
    const TCHAR  *pEncodeCodecPrm;     //音声をエンコードするコーデックのパラメータ
    const TCHAR  *pEncodeCodecProfile; //音声をエンコードするコーデックのパラメータ
    int           nBitrate;            //ビットレートの指定
    int           nSamplingRate;       //サンプリング周波数の指定
    const TCHAR  *pFilter;             //音声フィルタ
} AVOutputStreamPrm;

struct AvcodecWriterPrm {
    const AVDictionary          *pInputFormatMetadata;    //入力ファイルのグローバルメタデータ
    const TCHAR                 *pOutputFormat;           //出力のフォーマット
    bool                         bVideoDtsUnavailable;    //出力映像のdtsが無効 (API v1.6以下)
    const AVStream              *pVideoInputStream;       //入力映像のストリーム
    AVRational                   rBitstreamTimebase;      //エンコーダのtimebase
    int64_t                      nVideoInputFirstKeyPts;  //入力映像の最初のpts
    vector<sTrim>                trimList;                //Trimする動画フレームの領域のリスト
    vector<AVOutputStreamPrm>    inputStreamList;         //入力ファイルの音声・字幕の情報
    vector<const AVChapter *>    chapterList;             //チャプターリスト
    bool                         bChapterNoTrim;          //チャプターにtrimを反映しない
    int                          nAudioResampler;         //音声のresamplerの選択
    uint32_t                     nAudioIgnoreDecodeError; //音声デコード時に発生したエラーを無視して、無音に置き換える
    int                          nBufSizeMB;              //出力バッファサイズ
    int                          nOutputThread;           //出力スレッド数
    int                          nAudioThread;            //音声処理スレッド数
    muxOptList                   vMuxOpt;                 //mux時に使用するオプション
    PerfQueueInfo               *pQueueInfo;              //キューの情報を格納する構造体
    const TCHAR                 *pMuxVidTsLogFile;        //mux timestampログファイル
    HEVCHDRSei                  *pHEVCHdrSei;             //HDR関連のmetadata
    RGYTimestamp                *pVidTimestamp;           //動画のtimestampの情報

    AvcodecWriterPrm() :
        pInputFormatMetadata(nullptr),
        pOutputFormat(nullptr),
        bVideoDtsUnavailable(),
        pVideoInputStream(nullptr),
        rBitstreamTimebase(av_make_q(0, 1)),
        nVideoInputFirstKeyPts(0),
        trimList(),
        inputStreamList(),
        chapterList(),
        bChapterNoTrim(false),
        nAudioResampler(0),
        nAudioIgnoreDecodeError(0),
        nBufSizeMB(0),
        nOutputThread(0),
        nAudioThread(0),
        vMuxOpt(),
        pQueueInfo(nullptr),
        pMuxVidTsLogFile(nullptr),
        pHEVCHdrSei(nullptr),
        pVidTimestamp(nullptr) {
    }
};

class RGYOutputAvcodec : public RGYOutput
{
public:
    RGYOutputAvcodec();
    virtual ~RGYOutputAvcodec();

    virtual RGY_ERR WriteNextFrame(RGYBitstream *pBitstream) override;

    virtual RGY_ERR WriteNextFrame(RGYFrame *pSurface) override;

    virtual RGY_ERR WriteNextPacket(AVPacket *pkt);

    virtual vector<int> GetStreamTrackIdList();

    virtual void WaitFin() override;

    virtual void Close() override;

#if USE_CUSTOM_IO
    int readPacket(uint8_t *buf, int buf_size);
    int writePacket(uint8_t *buf, int buf_size);
    int64_t seek(int64_t offset, int whence);
#endif //USE_CUSTOM_IO
    //出力スレッドのハンドルを取得する
    HANDLE getThreadHandleOutput();
    HANDLE getThreadHandleAudProcess();
    HANDLE getThreadHandleAudEncode();
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *option) override;

    //別のスレッドで実行する場合のスレッド関数 (出力)
    RGY_ERR WriteThreadFunc();

    //別のスレッドで実行する場合のスレッド関数 (音声処理)
    RGY_ERR ThreadFuncAudThread();

    //別のスレッドで実行する場合のスレッド関数 (音声エンコード処理)
    RGY_ERR ThreadFuncAudEncodeThread();

    //音声出力キューに追加 (音声処理スレッドが有効な場合のみ有効)
    RGY_ERR AddAudQueue(AVPktMuxData *pktData, int type);

    //AVPktMuxDataを初期化する
    AVPktMuxData pktMuxData(const AVPacket *pkt);

    //AVPktMuxDataを初期化する
    AVPktMuxData pktMuxData(AVFrame *pFrame);

    //WriteNextFrameの本体
    RGY_ERR WriteNextFrameInternal(RGYBitstream *pBitstream, int64_t *pWrittenDts);

    //WriteNextPacketの本体
    RGY_ERR WriteNextPacketInternal(AVPktMuxData *pktData, int64_t maxDtsToWrite);

    //WriteNextPacketの音声処理部分(デコード/thAudEncodeがなければエンコードも担当)
    RGY_ERR WriteNextPacketAudio(AVPktMuxData *pktData);

    //WriteNextPacketの音声処理部分(エンコード)
    RGY_ERR WriteNextPacketAudioFrame(vector<AVPktMuxData> audioFrames);

    //フィルタリング後のパケットをサブトラックに分配する
    RGY_ERR WriteNextPacketToAudioSubtracks(vector<AVPktMuxData> audioFrames);

    //音声フレームをエンコード
    RGY_ERR WriteNextAudioFrame(AVPktMuxData *pktData);

    //音声のフィルタリングを実行
    vector<AVPktMuxData> AudioFilterFrame(vector<AVPktMuxData> audioFrames);
    vector<AVPktMuxData> AudioFilterFrameFlush(AVMuxAudio *pMuxAudio);

    //CodecIDがPCM系かどうか判定
    bool codecIDIsPCM(AVCodecID targetCodec);

    //PCMのコーデックがwav出力時に変換を必要とするかを判定する
    AVCodecID PCMRequiresConversion(const AVCodecParameters *pCodecParm);

    //RGY_CODECのcodecからAVCodecのCodecIDを返す
    AVCodecID getAVCodecId(RGY_CODEC codec);

    //AAC音声にBitstreamフィルターを適用する
    RGY_ERR applyBitstreamFilterAAC(AVPacket *pkt, AVMuxAudio *pMuxAudio);

    //音声のプロファイルを取得する
    int AudioGetCodecProfile(tstring profile, AVCodecID codecId);

    //音声のプロファイル(文字列)を取得する
    tstring AudioGetCodecProfileStr(int profile, AVCodecID codecId);

    //H.264ストリームからPAFFのフィールドの長さを返す
    uint32_t getH264PAFFFieldLength(const uint8_t *ptr, uint32_t size, int *isIDR);

    //extradataをコピーする
    void SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size);
    void SetExtraData(AVCodecParameters *pCodecParam, const uint8_t *data, uint32_t size);

    //映像の初期化
    RGY_ERR InitVideo(const VideoInfo *pVideoOutputInfo, const AvcodecWriterPrm *prm);

    //音声フィルタの初期化
    RGY_ERR InitAudioFilter(AVMuxAudio *pMuxAudio, int channels, uint64_t channel_layout, int sample_rate, AVSampleFormat sample_fmt);

    //音声リサンプラの初期化
    RGY_ERR InitAudioResampler(AVMuxAudio *pMuxAudio, int channels, uint64_t channel_layout, int sample_rate, AVSampleFormat sample_fmt);

    //音声の初期化
    RGY_ERR InitAudio(AVMuxAudio *pMuxAudio, AVOutputStreamPrm *pInputAudio, uint32_t nAudioIgnoreDecodeError);

    //字幕の初期化
    RGY_ERR InitSubtitle(AVMuxSub *pMuxSub, AVOutputStreamPrm *pInputSubtitle);

    //チャプターをコピー
    RGY_ERR SetChapters(const vector<const AVChapter *>& chapterList, bool bChapterNoTrim);

    //メッセージを作成
    tstring GetWriterMes();

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxAudio *getAudioPacketStreamData(const AVPacket *pkt);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxAudio *getAudioStreamData(int nTrackId, int nSubStreamId = 0);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxSub *getSubPacketStreamData(const AVPacket *pkt);

    //音声のchannel_layoutを自動選択する
    uint64_t AutoSelectChannelLayout(const uint64_t *pChannelLayout, const AVCodecContext *pSrcAudioCtx);

    //音声のsample formatを自動選択する
    AVSampleFormat AutoSelectSampleFmt(const AVSampleFormat *pSamplefmtList, const AVCodecContext *pSrcAudioCtx);

    //音声のサンプリングレートを自動選択する
    int AutoSelectSamplingRate(const int *pSamplingRateList, int nSrcSamplingRate);

    //音声ストリームをすべて吐き出す
    void AudioFlushStream(AVMuxAudio *pMuxAudio, int64_t *pWrittenDts);

    //音声をデコード
    vector<unique_ptr<AVFrame, decltype(&av_frame_unref)>> AudioDecodePacket(AVMuxAudio *pMuxAudio, AVPacket *pkt);

    //音声をエンコード
    vector<AVPktMuxData> AudioEncodeFrame(AVMuxAudio *pMuxAudio, AVFrame *frame);

    //字幕パケットを書き出す
    RGY_ERR SubtitleTranscode(const AVMuxSub *pMuxSub, AVPacket *pkt);

    //字幕パケットを書き出す
    RGY_ERR SubtitleWritePacket(AVPacket *pkt);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVPktMuxData *pktData);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVPktMuxData *pktData, int64_t *pWrittenDts);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVMuxAudio *pMuxAudio, AVPacket *pkt, int samples, int64_t *pWrittenDts);

    //extradataにH264のヘッダーを追加する
    RGY_ERR AddH264HeaderToExtraData(const RGYBitstream *pBitstream);

    //extradataにHEVCのヘッダーを追加する
    RGY_ERR AddHEVCHeaderToExtraData(const RGYBitstream *pBitstream);

    //ファイルヘッダーを書き出す
    RGY_ERR WriteFileHeader(const RGYBitstream *pBitstream);

    //タイムスタンプをTrimなどを考慮しつつ計算しなおす
    //nTimeInがTrimで切り取られる領域の場合
    //lastValidFrame ... true 最後の有効なフレーム+1のtimestampを返す / false .. AV_NOPTS_VALUEを返す
    int64_t AdjustTimestampTrimmed(int64_t nTimeIn, AVRational timescaleIn, AVRational timescaleOut, bool lastValidFrame);

    void CloseSubtitle(AVMuxSub *pMuxSub);
    void CloseAudio(AVMuxAudio *pMuxAudio);
    void CloseVideo(AVMuxVideo *pMuxVideo);
    void CloseFormat(AVMuxFormat *pMuxFormat);
    void CloseThread();
    void CloseQueues();

    static const AVRational QUEUE_DTS_TIMEBASE;
    AVMux m_Mux;
    vector<AVPktMuxData> m_AudPktBufFileHead; //ファイルヘッダを書く前にやってきた音声パケットのバッファ
};

#endif //ENABLE_AVSW_READER

#endif //__RGY_OUTPUT_AVCODEC_H__
