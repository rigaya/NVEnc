//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _AVCODEC_READER_H_
#define _AVCODEC_READER_H_

#include "avcodec_qsv.h"

#if ENABLE_AVCUVID_READER
#include <mutex>
#include <chrono>
#include "CuvidDecode.h"
#include "NVEncParam.h"
#include "NVEncInput.h"
#include "NVEncStatus.h"

using std::vector;
using std::pair;

static const uint32_t AVCODEC_READER_INPUT_BUF_SIZE = 16 * 1024 * 1024;

enum {
    AVQSV_AUDIO_NONE         = 0x00,
    AVQSV_AUDIO_MUX          = 0x01,
    AVQSV_AUDIO_COPY_TO_FILE = 0x02,
};

enum : uint32_t {
    AVQSV_PTS_SOMETIMES_INVALID = 0x01, //時折、無効なptsやdtsを得る
    AVQSV_PTS_HALF_INVALID      = 0x02, //PAFFなため、半分のフレームのptsやdtsが無効
    AVQSV_PTS_ALL_INVALID       = 0x04, //すべてのフレームのptsやdtsが無効
    AVQSV_PTS_NONKEY_INVALID    = 0x08, //キーフレーム以外のフレームのptsやdtsが無効
};

//フレームの位置情報と長さを格納する
typedef struct FramePos {
    int64_t pts;  //pts
    int64_t dts;  //dts
    int duration; //該当フレームの表示時間
    int flags;    //flags
} FramePos;

//動画フレームのデータ
typedef struct VideoFrameData {
    FramePos *frame;      //各フレームの情報への配列 (デコードが開始されるフレームから取得する)
    int fixed_num;        //frame配列でフレーム順序が確定したフレーム数
    int num;              //frame配列で現在格納されているデータ数
    int capacity;         //frame配列を確保した数
    int64_t duration;     //合計の動画の長さ
} VideoFrameData;

typedef struct AVDemuxFormat {
    AVFormatContext          *pFormatCtx;            //動画ファイルのformatContext
    int                       nAnalyzeSec;           //動画ファイルを先頭から分析する時間
    bool                      bIsPipe;               //入力がパイプ
    uint32_t                  nPreReadBufferIdx;     //先読みバッファの読み込み履歴
    int                       nAudioTracks;          //存在する音声のトラック数
    int                       nSubtitleTracks;       //存在する字幕のトラック数
} AVDemuxFormat;

typedef struct AVDemuxVideo {
    AVCodecContext           *pCodecCtx;             //動画のcodecContext
    int                       nIndex;                //動画のストリームID
    int64_t                   nStreamFirstPts;       //動画ファイルの最初のpts
    uint32_t                  nStreamPtsInvalid;     //動画ファイルのptsが無効 (H.264/ES, 等)
    bool                      bGotFirstKeyframe;     //動画の最初のキーフレームを取得済み
    VideoFrameData            frameData;             //動画フレームのptsのリスト
    AVBitStreamFilterContext *pH264Bsfc;             //必要なら使用するbitstreamfilter
    uint8_t                  *pExtradata;            //動画のヘッダ情報
    int                       nExtradataSize;        //動画のヘッダサイズ
    AVPacket                  packet[2];             //取得した動画ストリームの1フレーム分のデータ
    AVRational                nAvgFramerate;         //動画のフレームレート
    bool                      bUseHEVCmp42AnnexB;    //HEVCのmp4->AnnexB変換

    uint32_t                  nSampleLoadCount;      //sampleをLoadNextFrameでロードした数
    uint32_t                  nSampleGetCount;       //sampleをGetNextBitstreamで取得した数
} AVDemuxVideo;

typedef struct AVDemuxStream {
    int                       nIndex;                 //音声・字幕のストリームID (libavのストリームID)
    int                       nTrackId;               //音声のトラックID (QSVEncC独自, 1,2,3,...)、字幕は0
    AVCodecContext           *pCodecCtx;              //音声・字幕のcodecContext
    AVStream                 *pStream;                //音声・字幕のストリーム
    int                       nLastVidIndex;          //音声の直前の相当する動画の位置
    int64_t                   nExtractErrExcess;      //音声抽出のあまり (音声が多くなっていれば正、足りなくなっていれば負)
    AVPacket                  pktSample;              //サンプル用の音声・字幕データ
    int                       nDelayOfStream;         //音声側の遅延 (pkt_timebase基準)
} AVDemuxStream;

typedef struct AVDemuxer {
    AVDemuxFormat            format;
    AVDemuxVideo             video;
    vector<AVDemuxStream>    stream;
    vector<const AVChapter*> chapter;
    std::mutex               mtx;     //frame配列アクセスへの排他制御用
} AVDemuxer;

typedef struct AvcodecReaderPrm {
    uint8_t        memType;                 //使用するメモリの種類
    bool           bReadVideo;              //映像の読み込みを行うかどうか
    uint32_t       nReadAudio;              //音声の読み込みを行うかどうか (AVQSV_AUDIO_xxx)
    bool           bReadSubtitle;           //字幕の読み込みを行うかどうか
    bool           bReadChapter;            //チャプターの読み込みを行うかどうか
    pair<int,int>  nVideoAvgFramerate;      //動画のフレームレート (映像のみ読み込ませるときに使用する)
    uint16_t       nAnalyzeSec;             //入力ファイルを分析する秒数
    uint16_t       nTrimCount;              //Trimする動画フレームの領域の数
    sTrim         *pTrimList;               //Trimする動画フレームの領域のリスト
    int            nAudioTrackStart;        //音声のトラック番号の開始点
    int            nSubtitleTrackStart;     //字幕のトラック番号の開始点
    int            nAudioSelectCount;       //muxする音声のトラック数
    sAudioSelect **ppAudioSelect;           //muxする音声のトラック番号のリスト 1,2,...(1から連番で指定)
    int            nSubtitleSelectCount;    //muxする字幕のトラック数
    const int     *pSubtitleSelect;         //muxする字幕のトラック番号のリスト 1,2,...(1から連番で指定)
    CUvideoctxlock ctxLock;
} AvcodecReaderPrm;


class CAvcodecReader : public NVEncBasicInput
{
public:
    CAvcodecReader();
    virtual ~CAvcodecReader();

    virtual int Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) override;

    virtual void Close();

    //動画ストリームの1フレーム分のデータをm_sPacketに格納する
    //m_sPacketからの取得はGetNextBitstreamで行う
    virtual int LoadNextFrame(void *dst, int dst_pitch) override;

    //動画ストリームの1フレーム分のデータをbitstreamに追加する
    virtual int GetNextBitstream(vector<uint8_t>& bitstream, int64_t *pts) override;

    //ストリームのヘッダ部分を取得する
    virtual int GetHeader(vector<uint8_t>& bitstream) override;

    //入力ファイルのグローバルメタデータを取得する
    const AVDictionary *GetInputFormatMetadata();

    //動画の入力情報を取得する
    const AVCodecContext *GetInputVideoCodecCtx();
    
    //音声・字幕パケットの配列を取得する
    vector<AVPacket> GetStreamDataPackets();

    //音声・字幕のコーデックコンテキストを取得する
    vector<AVDemuxStream> GetInputStreamInfo();

    //チャプターリストを取得する
    vector<const AVChapter *> GetChapterList();

    //入力ファイルに存在する音声のトラック数を返す
    //int GetAudioTrackCount() override;

    //入力ファイルに存在する字幕のトラック数を返す
    //int GetSubtitleTrackCount() override;

    //デコードするストリームの情報を取得する
    void GetDecParam(InputVideoInfo *decParam) {
        memcpy(decParam, &m_sDecParam, sizeof(m_sDecParam));
    }

    //動画の最初のフレームのptsを取得する
    int64_t GetVideoFirstPts();
private:
    //avcodecのコーデックIDからcuvidのコーデックのcodec idを取得
    cudaVideoCodec getCuvidcc(uint32_t id);

    //avcodecのストリームIDを取得 (typeはAVMEDIA_TYPE_xxxxx)
    vector<int> getStreamIndex(AVMediaType type);

    //VC-1のスタートコードの確認
    bool vc1StartCodeExists(uint8_t *ptr);

    //動画のptsをソートする
    void sortVideoPtsList();

    //動画のptsをリストに加える
    void addVideoPtsToList(FramePos pos);

    //対象ストリームのパケットを取得
    int getSample(AVPacket *pkt);

    //対象・字幕の音声パケットを追加するかどうか
    bool checkStreamPacketToAdd(const AVPacket *pkt, AVDemuxStream *pStream);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVDemuxStream *getPacketStreamData(const AVPacket *pkt);

    //bitstreamにpktの内容を追加する
    int setToMfxBitstream(vector<uint8_t>& bitstream, AVPacket *pkt);

    //音声パケットリストを開放
    void clearStreamPacketList(std::vector<AVPacket>& pktList);

    //音声パケットの配列を取得する (映像を読み込んでいないときに使用)
    vector<AVPacket> GetAudioDataPacketsWhenNoVideoRead();

    //QSVでデコードした際の最初のフレームのptsを取得する
    //さらに、平均フレームレートを推定する
    //fpsDecoderはdecoderの推定したfps
    int getFirstFramePosAndFrameRate(CuvidDecode *dec, const sTrim *pTrimList, int nTrimCount);

    //指定したptsとtimebaseから、該当する動画フレームを取得する
    int getVideoFrameIdx(int64_t pts, AVRational timebase, const FramePos *framePos, int framePosCount, int iStart);

    //ptsを動画のtimebaseから音声のtimebaseに変換する
    int64_t convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *pStream);

    //HEVCのmp4->AnnexB簡易変換
    void hevcMp42Annexb(AVPacket *pkt);

    //VC-1のヘッダの修正を行う
    void vc1FixHeader(int nLengthFix = -1);

    //VC-1のフレームヘッダを追加
    void vc1AddFrameHeader(AVPacket *pkt);

    void CloseStream(AVDemuxStream *pAudio);
    void CloseVideo(AVDemuxVideo *pVideo);
    void CloseFormat(AVDemuxFormat *pFormat);

    AVDemuxer        m_Demux;                      //デコード用情報
    vector<uint8_t>  m_hevcMp42AnnexbBuffer;       //HEVCのmp4->AnnexB簡易変換用バッファ
    vector<AVPacket> m_PreReadBuffer;              //解析用に先行取得した映像パケット
    vector<AVPacket> m_StreamPacketsBufferL1[2];    //音声のAVPacketのバッファ (マルチスレッドで追加されてくることに注意する)
    vector<AVPacket> m_StreamPacketsBufferL2;       //音声のAVPacketのバッファ
    uint32_t         m_StreamPacketsBufferL2Used;   //m_StreamPacketsBufferL2のパケットのうち、すでに使用したもの
};

#endif //ENABLE_AVCUVID_READER

#endif //_AVCODEC_READER_H_
