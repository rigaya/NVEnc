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
#include <deque>
#include <atomic>
#include <unordered_map>
#include <cstdint>
#include "rgy_avutil.h"
#include "rgy_bitstream.h"
#include "rgy_input_avcodec.h"
#include "rgy_output.h"
#include "rgy_perf_monitor.h"
#include "rgy_util.h"
#if ENCODER_NVENC
#include "NVEncUtil.h"
#endif //#if ENCODER_NVENC
#if ENCODER_QSV
#include "qsv_util.h"
#endif //#if ENCODER_QSV
#if ENCODER_VCEENC
#include "vce_util.h"
#endif //#if ENCODER_VCEENC

using std::vector;

#define USE_CUSTOM_IO 1

static const int SUB_ENC_BUF_MAX_SIZE = 1024 * 1024;

static const int VID_BITSTREAM_QUEUE_SIZE_I  = 4;
static const int VID_BITSTREAM_QUEUE_SIZE_PB = 64;

enum RGYMetadataCopyDefault {
    RGY_METADATA_DEFAULT_CLEAR,
    RGY_METADATA_DEFAULT_COPY_LANG_ONLY,
    RGY_METADATA_DEFAULT_COPY
};

struct AVMuxTimestamp {
    int64_t timestamp_list[8];

    AVMuxTimestamp() : timestamp_list() { }

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

struct AVMuxFormat {
    const TCHAR          *filename;             //出力ファイル名
    AVFormatContext      *formatCtx;            //出力ファイルのformatContext
    char                  metadataStr[256];     //出力ファイルのエンコーダ名
    const AVOutputFormat *outputFmt;            //出力ファイルのoutputFormat

#if USE_CUSTOM_IO
    uint8_t              *AVOutBuffer;          //avio_alloc_context用のバッファ
    uint32_t              AVOutBufferSize;      //avio_alloc_context用のバッファサイズ
    FILE                 *fpOutput;             //出力ファイルポインタ
    char                 *outputBuffer;         //出力ファイルポインタ用のバッファ
    uint32_t              outputBufferSize;     //出力ファイルポインタ用のバッファサイズ
#endif //USE_CUSTOM_IO
    bool                  streamError;          //エラーが発生
    bool                  isMatroska;           //mkvかどうか
    bool                  isPipe;               //パイプ出力かどうか
    bool                  fileHeaderWritten;    //ファイルヘッダを出力したかどうか
    AVDictionary         *headerOptions;        //ヘッダオプション
    bool                  disableMp4Opt;        //mp4出力時のmuxの最適化(faststart)を無効にする
    bool                  lowlatency;           //低遅延モード
    bool                  allowOtherNegativePts; //音声・字幕の負のptsを許可するかどうか
    bool                  timestampPassThrough;  //タイムスタンプをそのまま出力するかどうか

    AVMuxFormat();
};

struct AVMuxVideo {
    const AVCodec        *codec;                //出力映像のCodec
    AVCodecContext       *codecCtx;             //出力映像のCodecCtx
    AVRational            outputFps;            //出力映像のフレームレート
    AVStream             *streamOut;            //出力ファイルの映像ストリーム
    bool                  dtsUnavailable;       //出力映像のdtsが無効 (API v1.6以下)
    AVRational            inputStreamTimebase;  //入力streamのtimebase
    int64_t               inputFirstKeyPts;     //入力映像の最初のpts
    AVRational            bitstreamTimebase;    //エンコーダのtimebase
    AVMuxTimestamp        timestampList;        //エンコーダから渡されたtimestampリスト
    int                   fpsBaseNextDts;       //出力映像のfpsベースでのdts (API v1.6以下でdtsが計算されない場合に使用する)
    std::unique_ptr<FILE, fp_deleter> fpTsLogFile; //mux timestampログファイル
    RGYBitstream          hdrBitstream;         //追加のsei nal
    DOVIRpu              *doviRpu;              //dovi rpu 追加用
    AVBSFContext         *bsfc;                 //必要なら使用するbitstreamfilter
    uint8_t              *bsfcBuffer;           //bitstreamfilter用のバッファ
    size_t                bsfcBufferLength;     //bitstreamfilter用のバッファの長さ
    RGYTimestamp         *timestamp;            //timestampの情報
    AVPacket             *pktOut;               //出力用のAVPacket
    AVPacket             *pktParse;             //parser用のAVPacket
    int64_t               prevEncodeFrameId;    //前回のエンコードフレームID
    int64_t               prevInputFrameId;     //前回の入力フレームID
    AVCodecParserContext *parserCtx;            //動画ストリームのParser (VCEのみ)
    int64_t               parserStreamPos;      //動画ストリームのバイト数
    bool                  afs;                  //入力が自動フィールドシフト
    bool                  debugDirectAV1Out;    //AV1出力のデバッグ用
    decltype(parse_nal_unit_h264_c) *parse_nal_h264; // H.264用のnal unit分解関数へのポインタ
    decltype(parse_nal_unit_hevc_c) *parse_nal_hevc; // HEVC用のnal unit分解関数へのポインタ

    AVMuxVideo();
};

struct AVMuxAudio {
    int                   inTrackId;            //ソースファイルの入力トラック番号
    int                   inSubStream;          //ソースファイルの入力サブストリーム番号
    const AVStream       *streamIn;             //入力音声のストリーム
    int                   streamIndexIn;        //入力音声のStreamのindex
    AVStream             *streamOut;            //出力ファイルの音声ストリーム
    int                   packetWritten;        //出力したパケットの数
    int64_t               dec_rescale_delta;    //decode時のtimebase変換用

    //変換用
    const AVCodec        *outCodecDecode;       //変換する元のコーデック
    AVCodecContext       *outCodecDecodeCtx;    //変換する元のCodecContext
    const AVCodec        *outCodecEncode;       //変換先の音声のコーデック
    AVCodecContext       *outCodecEncodeCtx;    //変換先の音声のCodecContext
    int64_t               decodeNextPts;        //デコードの次のpts (samplerateベース)
    uint32_t              ignoreDecodeError;    //デコード時に連続して発生したエラー回数がこの閾値を以下なら無視し、無音に置き換える
    uint32_t              decodeError;          //デコード処理中に連続してエラーが発生した回数
    bool                  encodeError;          //エンコード処理中にエラーが発生
    bool                  flushed;              //AudioFlushStream を完了したフラグ

    //filter
    int                   filterInChannels;      //現在のchannel数      (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    uniuqeRGYChannelLayout filterInChannelLayout; //現在のchannel_layout (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    int                   filterInSampleRate;    //現在のsampling rate  (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    AVSampleFormat        filterInSampleFmt;     //現在のSampleformat   (pSwrContext == nullptrなら、encoderの入力、そうでないならresamplerの入力)
    const TCHAR          *filter;
    AVFilterContext      *filterBufferSrcCtx;
    AVFilterContext      *filterBufferSinkCtx;
    AVFilterContext      *filterAudioFormat;
    AVFilterGraph        *filterGraph;

    //resampler
    int                   audioResampler;      //resamplerの選択 (QSV_RESAMPLER_xxx)
    std::string           audioResamplerPrm;
    AVFrame              *decodedFrameCache;   //デコードされたデータのキャッシュされたもの
    int                   channelMapping[MAX_SPLIT_CHANNELS];        //resamplerで使用するチャンネル割り当て(入力チャンネルの選択)
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelSelect; //入力音声の使用するチャンネル
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelOut;    //出力音声のチャンネル

    //AACの変換用
    AVBSFContext         *bsfc;              //必要なら使用するbitstreamfilter
    int                   bsfErrorFromStart; //開始直後からのbitstream filter errorの数

    int64_t               outputSampleOffset;   //出力音声のptsがAV_NOPTS_VALUEの補正用
    int64_t               outputSamples;        //出力音声の出力済みsample数
    int64_t               lastPtsIn;            //入力音声の前パケットのpts (input stream timebase)
    int64_t               lastPtsOut;           //出力音声の前パケットのpts

    std::unique_ptr<FILE, fp_deleter> fpTsLogFile; //mux timestampログファイル

    AVMuxAudio();
};

struct AVSubtitleData {
    AVSubtitle decodecSub; //デコードした字幕データ
    int64_t origPts;
    int64_t origDuration;

    AVSubtitleData();
    ~AVSubtitleData();
};

struct AVMuxOther {
    int                   inTrackId;           //ソースファイルの入力トラック番号
    const AVStream       *streamIn;            //入力字幕のストリーム
    int                   streamIndexIn;       //入力字幕のStreamのindex
    AVRational            streamInTimebase;     //入力字幕のストリームのtimebase
    AVStream             *streamOut;           //出力ファイルの字幕ストリーム

    //変換用
    const AVCodec        *outCodecDecode;      //変換する元のコーデック
    AVCodecContext       *outCodecDecodeCtx;   //変換する元のCodecContext
    const AVCodec        *outCodecEncode;      //変換先の音声のコーデック
    AVCodecContext       *outCodecEncodeCtx;   //変換先の音声のCodecContext

    uint8_t              *bufConvert;          //変換用のバッファ

    AVBSFContext         *bsfc;              //必要なら使用するbitstreamfilter

    std::vector<AVSubtitleData> decodedSub; //字幕データ

    AVMuxOther();
};

enum {
    MUX_DATA_TYPE_NONE   = 0,
    MUX_DATA_TYPE_PACKET = 1, //AVPktMuxDataに入っているデータがAVPacket
    MUX_DATA_TYPE_FRAME  = 2, //AVPktMuxDataに入っているデータがAVFrame
};

typedef struct AVPktMuxData {
    int         type;        //MUX_DATA_TYPE_xxx
    AVPacket   *pkt;         //type == MUX_DATA_TYPE_PACKET 時有効
    AVMuxAudio *muxAudio;    //type == MUX_DATA_TYPE_PACKET 時有効
    int64_t     dts;         //type == MUX_DATA_TYPE_PACKET 時有効
    int         samples;     //type == MUX_DATA_TYPE_PACKET 時有効
    AVFrame    *frame;      //type == MUX_DATA_TYPE_FRAME 時有効
    int         got_result;  //type == MUX_DATA_TYPE_FRAME 時有効
} AVPktMuxData;

enum {
    AUD_QUEUE_PROCESS = 0,
    AUD_QUEUE_ENCODE  = 1,
    AUD_QUEUE_OUT     = 2,
};

struct AVMuxThreadWorker {
    std::thread                    thread;          //音声処理スレッド(デコード/thAudEncodeがなければエンコードも担当)
    std::atomic<bool>              thAbort;         //音声処理スレッドに停止を通知する
    bool                           sentEOS;         //EOSパケットを送信側からこのworkerに送ったことを示す
    HANDLE                         heEventPktAdded; //キューのいずれかにデータが追加されたことを通知する
    HANDLE                         heEventClosing;  //音声処理スレッドが停止処理を開始したことを通知する
    RGYQueueMPMP<AVPktMuxData, 64> qPackets;        //音声パケットをスレッドに渡すためのキュー

    AVMuxThreadWorker();
    ~AVMuxThreadWorker();
    void close();
};


struct AVMuxThreadAudio {
    AVMuxThreadWorker encode;   //音声エンコードスレッド
    AVMuxThreadWorker process;  //音声処理スレッド

    AVMuxThreadAudio();
    ~AVMuxThreadAudio();
    bool threadActiveEncode();
    bool threadActiveProcess();
    void closeEncode();
    void closeProcess();
};

#if ENABLE_AVCODEC_OUT_THREAD
struct AVMuxThread {
    bool                           enableOutputThread;        //出力スレッドを使用する
    bool                           enableAudProcessThread;    //音声処理スレッドを使用する
    bool                           enableAudEncodeThread;     //音声エンコードスレッドを使用する
    std::unique_ptr<AVMuxThreadWorker> thOutput;              //出力スレッド
    RGYQueueMPMP<RGYBitstream, 64> qVideobitstreamFreeI;      //映像 Iフレーム用に空いているデータ領域を格納する
    RGYQueueMPMP<RGYBitstream, 64> qVideobitstreamFreePB;     //映像 P/Bフレーム用に空いているデータ領域を格納する
    RGYQueueMPMP<RGYBitstream, 64> qVideobitstream;           //映像パケットを出力スレッドに渡すためのキュー
    std::unordered_map<const AVMuxAudio *, std::unique_ptr<AVMuxThreadAudio>> thAud; //音声スレッド
    std::atomic<int64_t>           streamOutMaxDts;           //音声・字幕キューの最後のdts (timebase = QUEUE_DTS_TIMEBASE) (キューの同期に使用)
    PerfQueueInfo                 *queueInfo;                 //キューの情報を格納する構造体

    AVMuxThread();
    bool threadActiveAudio() const { return enableAudProcessThread; };
    bool threadActiveAudioEncode() const { return enableAudEncodeThread; };
    bool threadActiveAudioProcess() const { return enableAudProcessThread; };
};
#endif

struct AVMux {
    AVMuxFormat         format;
    AVMuxVideo          video;
    std::deque<std::unique_ptr<unit_info>> videoAV1Merge;
    vector<AVMuxAudio>  audio;
    vector<AVMuxOther>  other;
    vector<sTrim>       trim;
#if ENABLE_AVCODEC_OUT_THREAD
    AVMuxThread         thread;
#endif
    RGYPoolAVPacket    *poolPkt;
    RGYPoolAVFrame     *poolFrame;

    AVMux();
};

struct AVOutputStreamPrm {
    AVDemuxStream src;          //入力音声・字幕の情報
    tstring decodeCodecPrm;     //音声をデコードするコーデックのパラメータ
    tstring encodeCodec;        //音声をエンコードするコーデック
    tstring encodeCodecPrm;     //音声をエンコードするコーデックのパラメータ
    tstring encodeCodecProfile; //音声をエンコードするコーデックのパラメータ
    int     bitrate;            //ビットレートの指定
    std::pair<bool, int> quality; //品質の指定 (値が設定されているかと値)
    int     samplingRate;       //サンプリング周波数の指定
    tstring filter;             //音声フィルタ
    bool    asdata;             //バイナリデータとして転送する
    tstring bsf;                //適用すべきbsfの名前
    tstring disposition;        //disposition
    std::vector<tstring> metadata; //metadata
    std::string resamplerPrm;

    AVOutputStreamPrm() :
        src(),
        decodeCodecPrm(),
        encodeCodec(RGY_AVCODEC_COPY),
        encodeCodecPrm(),
        encodeCodecProfile(),
        bitrate(0),
        quality({ false, RGY_AUDIO_QUALITY_DEFAULT }),
        samplingRate(0),
        filter(),
        asdata(false),
        bsf(),
        disposition(),
        metadata(),
        resamplerPrm() {

    }
};

struct AvcodecWriterPrm {
    const AVDictionary          *inputFormatMetadata;     //入力ファイルのグローバルメタデータ
    tstring                      outputFormat;            //出力のフォーマット
    bool                         allowOtherNegativePts;   //音声・字幕の負のptsを許可するかどうか
    bool                         timestampPassThrough;    //タイムスタンプをそのまま出力するかどうか
    bool                         bVideoDtsUnavailable;    //出力映像のdtsが無効 (API v1.6以下)
    bool                         lowlatency;              //低遅延モード 
    const AVStream              *videoInputStream;        //入力映像のストリーム
    AVRational                   bitstreamTimebase;       //エンコーダのtimebase
    int64_t                      videoInputFirstKeyPts;   //入力映像の最初のpts
    vector<sTrim>                trimList;                //Trimする動画フレームの領域のリスト
    vector<AVOutputStreamPrm>    inputStreamList;         //入力ファイルの音声・字幕の情報
    vector<const AVChapter *>    chapterList;             //チャプターリスト
    bool                         chapterNoTrim;           //チャプターにtrimを反映しない
    vector<AttachmentSource>     attachments;             //attachment
    int                          audioResampler;          //音声のresamplerの選択
    uint32_t                     audioIgnoreDecodeError;  //音声デコード時に発生したエラーを無視して、無音に置き換える
    int                          bufSizeMB;               //出力バッファサイズ
    int                          threadOutput;            //出力スレッド数
    int                          threadAudio;             //音声処理スレッド数
    RGYParamThread               threadParamOutput;       //出力スレッドのパラメータ
    RGYParamThread               threadParamAudio;        //音声処理スレッドのパラメータ
    RGYOptList                   muxOpt;                  //mux時に使用するオプション
    PerfQueueInfo               *queueInfo;               //キューの情報を格納する構造体
    tstring                      muxVidTsLogFile;         //mux timestampログファイル
    const RGYHDRMetadata        *hdrMetadata;             //HDR関連のmetadata
    DOVIRpu                     *doviRpu;                 //DOVIRpu
    bool                         doviRpuMetadataCopy;     //doviのmetadataのコピー
    RGYDOVIProfile               doviProfile;             //doviのprofile
    RGYTimestamp                *vidTimestamp;            //動画のtimestampの情報
    std::string                  videoCodecTag;           //動画タグ
    std::vector<tstring>         videoMetadata;           //動画のmetadata
    std::vector<tstring>         formatMetadata;          //formatのmetadata
    bool                         afs;                     //入力が自動フィールドシフト
    bool                         disableMp4Opt;           //mp4出力時のmuxの最適化を無効にする
    bool                         debugDirectAV1Out;       //AV1出力のデバッグ用
    RGYPoolAVPacket             *poolPkt;                 //読み込み側からわたってきたパケットの返却先
    RGYPoolAVFrame              *poolFrame;               //読み込み側からわたってきたパケットの返却先

    AvcodecWriterPrm() :
        inputFormatMetadata(nullptr),
        outputFormat(),
        allowOtherNegativePts(false),
        timestampPassThrough(false),
        bVideoDtsUnavailable(),
        lowlatency(false),
        videoInputStream(nullptr),
        bitstreamTimebase(av_make_q(0, 1)),
        videoInputFirstKeyPts(0),
        trimList(),
        inputStreamList(),
        chapterList(),
        chapterNoTrim(false),
        attachments(),
        audioResampler(0),
        audioIgnoreDecodeError(0),
        bufSizeMB(0),
        threadOutput(0),
        threadAudio(0),
        threadParamOutput(),
        threadParamAudio(),
        muxOpt(),
        queueInfo(nullptr),
        muxVidTsLogFile(),
        hdrMetadata(nullptr),
        doviRpu(nullptr),
        doviRpuMetadataCopy(false),
        doviProfile(RGY_DOVI_PROFILE_UNSET),
        vidTimestamp(nullptr),
        videoCodecTag(),
        videoMetadata(),
        formatMetadata(),
        afs(false),
        disableMp4Opt(false),
        debugDirectAV1Out(false),
        poolPkt(nullptr),
        poolFrame(nullptr) {
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
    int writePacket(const uint8_t *buf, int buf_size);
    int64_t seek(int64_t offset, int whence);
#endif //USE_CUSTOM_IO
    //出力スレッドのハンドルを取得する
    HANDLE getThreadHandleOutput();
    HANDLE getThreadHandleAudProcess();
    HANDLE getThreadHandleAudEncode();
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, const VideoInfo *videoOutputInfo, const void *option) override;

    //別のスレッドで実行する場合のスレッド関数 (出力)
    RGY_ERR WriteThreadFunc(RGYParamThread threadParam);

    //別のスレッドで実行する場合のスレッド関数 (音声処理)
    RGY_ERR ThreadFuncAudThread(const AVMuxAudio *const muxAudio, RGYParamThread threadParam);

    //別のスレッドで実行する場合のスレッド関数 (音声エンコード処理)
    RGY_ERR ThreadFuncAudEncodeThread(const AVMuxAudio *const muxAudio, RGYParamThread threadParam);

    //対象パケットの担当スレッドを探す
    AVMuxThreadWorker *getPacketWorker(const AVMuxAudio *muxAudio, const int type);

    //音声出力キューに追加 (音声処理スレッドが有効な場合のみ有効)
    RGY_ERR AddAudQueue(AVPktMuxData *pktData, int type);

    //AVPktMuxDataを初期化する
    AVPktMuxData pktMuxData(AVPacket *pkt);

    //AVPktMuxDataを初期化する
    AVPktMuxData pktMuxData(AVFrame *frame);

    //WriteNextFrameの本体
    RGY_ERR WriteNextFrameInternal(RGYBitstream *bitstream, int64_t *writtenDts);
    RGY_ERR WriteNextFrameInternalOneFrame(RGYBitstream *bitstream, int64_t *writtenDts, const RGYTimestampMapVal& bs_framedata);
    RGY_ERR WriteNextFrameFinish(RGYBitstream *bitstream, const RGY_FRAMETYPE frameType);

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
    vector<AVPktMuxData> AudioFilterFrameFlush(AVMuxAudio *muxAudio);

    //CodecIDがPCM系かどうか判定
    bool codecIDIsPCM(AVCodecID targetCodec);

    //PCMのコーデックがwav出力時に変換を必要とするかを判定する
    AVCodecID PCMRequiresConversion(const AVCodecParameters *codecParm);

    //RGY_CODECのcodecからAVCodecのCodecIDを返す
    AVCodecID getAVCodecId(RGY_CODEC codec);

    //Bitstreamフィルターを適用する
    RGY_ERR applyBitstreamFilterAudio(AVPacket *pkt, AVMuxAudio *muxAudio);
    RGY_ERR applyBitstreamFilterOther(AVPacket* pkt, const AVMuxOther *muxOther);

    //音声のプロファイルを取得する
    int AudioGetCodecProfile(tstring profile, AVCodecID codecId);

    //音声のプロファイル(文字列)を取得する
    tstring AudioGetCodecProfileStr(int profile, AVCodecID codecId);

    //H.264ストリームからPAFFのフィールドの長さを返す
    uint32_t getH264PAFFFieldLength(const uint8_t *ptr, uint32_t size, int *isIDR);

    //extradataをコピーする
    void SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size);
    void SetExtraData(AVCodecParameters *codecParam, const uint8_t *data, uint32_t size);

    //映像の初期化
    RGY_ERR InitVideo(const VideoInfo *videoOutputInfo, const AvcodecWriterPrm *prm);

    //音声フィルタの初期化
    RGY_ERR InitAudioFilter(AVMuxAudio *muxAudio, int channels, const RGYChannelLayout *channel_layout, int sample_rate, AVSampleFormat sample_fmt, const std::string resamplerPrm);

    //音声リサンプラの初期化
    RGY_ERR InitAudioResampler(AVMuxAudio *muxAudio, int channels, const RGYChannelLayout *channel_layout, int sample_rate, AVSampleFormat sample_fmt);

    //音声の初期化
    RGY_ERR InitAudio(AVMuxAudio *muxAudio, AVOutputStreamPrm *inputAudio, uint32_t audioIgnoreDecodeError, bool audioDispositionSet, const tstring& muxTsLogFileBase);

    //Bitstream Filterの初期化
    AVBSFContext* InitStreamBsf(const tstring& bsfName, const AVStream* streamIn);

    //字幕の初期化
    RGY_ERR InitOther(AVMuxOther *pMuxSub, AVOutputStreamPrm *inputSubtitle, bool streamDispositionSet);

    //Attachmentの初期化
    RGY_ERR InitAttachment(AVMuxOther *pMuxAttach, const AttachmentSource& attachment);

    //チャプターをコピー
    RGY_ERR SetChapters(const vector<const AVChapter *>& chapterList, bool chapterNoTrim);

    //metadataの設定
    RGY_ERR SetMetadata(AVDictionary **metadata, const AVDictionary *srcMetadata, const std::vector<tstring>& metadataOpt, const RGYMetadataCopyDefault defaultCopy, const tstring &trackName);

    //メッセージを作成
    tstring GetWriterMes();

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxAudio *getAudioPacketStreamData(const AVPacket *pkt);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxAudio *getAudioStreamData(int trackId, int subStreamId = 0);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVMuxOther *getOtherPacketStreamData(const AVPacket *pkt);

    //音声のchannel_layoutを自動選択する
    uniuqeRGYChannelLayout AutoSelectChannelLayout(const AVCodec *codec, const AVCodecContext *srcAudioCtx);

    //音声のsample formatを自動選択する
    AVSampleFormat AutoSelectSampleFmt(const AVSampleFormat *samplefmtList, const AVCodecContext *srcAudioCtx);

    //音声のサンプリングレートを自動選択する
    int AutoSelectSamplingRate(const int *pSamplingRateList, int nSrcSamplingRate);

    //音声ストリームをすべて吐き出す
    void AudioFlushStream(AVMuxAudio *muxAudio, int64_t *writtenDts);

    //音声をデコード
    vector<unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>> AudioDecodePacket(AVMuxAudio *muxAudio, AVPacket *pkt);

    //音声をエンコード
    vector<AVPktMuxData> AudioEncodeFrame(AVMuxAudio *muxAudio, AVFrame *frame);

    //字幕パケットを書き出す
    RGY_ERR SubtitleTranscode(AVMuxOther *pMuxSub, AVPacket *pkt);

    //字幕パケットのエンコードと出力
    RGY_ERR SubtitleEncode(const AVMuxOther *muxSub, AVSubtitleData *sub);

    //その他のパケットを書き出す
    RGY_ERR WriteOtherPacket(AVPacket *pkt);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVPktMuxData *pktData);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVPktMuxData *pktData, int64_t *writtenDts);

    //パケットを実際に書き出す
    void WriteNextPacketProcessed(AVMuxAudio *muxAudio, AVPacket *pkt, int samples, int64_t *writtenDts);

    //ヘッダにbsfを適用する
    RGY_ERR applyBsfToHeader(std::vector<uint8_t>& result, const uint8_t *target, const size_t target_size);

    //extradataにH264のヘッダーを追加する
    RGY_ERR AddHeaderToExtraDataH264(const RGYBitstream *pBitstream);

    //extradataにHEVCのヘッダーを追加する
    RGY_ERR AddHeaderToExtraDataHEVC(const RGYBitstream *pBitstream);

    //extradataにAV1のヘッダーを追加する
    RGY_ERR AddHeaderToExtraDataAV1(const RGYBitstream *pBitstream);

    //ファイルヘッダーを書き出す
    RGY_ERR WriteFileHeader(const RGYBitstream *pBitstream);

    //タイムスタンプをTrimなどを考慮しつつ計算しなおす
    //nTimeInがTrimで切り取られる領域の場合
    //lastValidFrame ... true 最後の有効なフレーム+1のtimestampを返す / false .. AV_NOPTS_VALUEを返す
    int64_t AdjustTimestampTrimmed(int64_t nTimeIn, AVRational timescaleIn, AVRational timescaleOut, bool lastValidFrame);

    RGY_ERR VidCheckStreamAVParser(RGYBitstream *pBitstream);

    void CloseOther(AVMuxOther *pMuxOther);
    void CloseAudio(AVMuxAudio *muxAudio);
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
