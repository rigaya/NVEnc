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
#ifndef __RGY_AVUTIL_H__
#define __RGY_AVUTIL_H__

#include "rgy_version.h"
#include "rgy_tchar.h"

#if ENABLE_AVSW_READER
#include <algorithm>
#include <vector>

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4819)
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/display.h>
#include <libavutil/mastering_display_metadata.h>
#if __has_include(<libavutil/dovi_meta.h>)
#define LIBAVUTIL_DOVI_META_AVAIL 1
#include <libavutil/dovi_meta.h>
#else
#define LIBAVUTIL_DOVI_META_AVAIL 0
#endif
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#if __has_include(<libavcodec/bsf.h>)
#include <libavcodec/bsf.h>
#endif
#include <libswresample/swresample.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#if ENABLE_LIBAVDEVICE
#include <libavdevice/avdevice.h>
#endif
}
#pragma comment (lib, "avcodec.lib")
#pragma comment (lib, "avformat.lib")
#pragma comment (lib, "avutil.lib")
#pragma comment (lib, "swresample.lib")
#pragma comment (lib, "avfilter.lib")
#if ENABLE_LIBAVDEVICE
#pragma comment (lib, "avdevice.lib")
#endif
#pragma warning (pop)

#include "rgy_log.h"
#include "rgy_def.h"
#include "rgy_util.h"
#include "rgy_queue.h"

#if _DEBUG
#define RGY_AV_LOG_LEVEL AV_LOG_WARNING
#else
#define RGY_AV_LOG_LEVEL AV_LOG_ERROR
#endif

#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
using RGYChannel = AVChannel;
using RGYChannelLayout = AVChannelLayout;
using uniuqeRGYChannelLayout = std::unique_ptr<RGYChannelLayout, decltype(&av_channel_layout_uninit)>;
#else
using RGYChannel = uint64_t;
using RGYChannelLayout = uint64_t;
using uniuqeRGYChannelLayout = std::unique_ptr<RGYChannelLayout>;
#endif

#ifndef FF_PROFILE_UNKNOWN
#define FF_PROFILE_UNKNOWN (AV_PROFILE_UNKNOWN)
#endif

template<typename T>
struct RGYAVDeleter {
    RGYAVDeleter() : deleter(nullptr) {};
    RGYAVDeleter(std::function<void(T**)> deleter) : deleter(deleter) {};
    void operator()(T *p) { deleter(&p); }
    std::function<void(T**)> deleter;
};

struct StreamInfoOptDeleter {
    int streamCount;
    StreamInfoOptDeleter(int streamCount_) : streamCount(streamCount_) {};
    void operator()(AVDictionary **dictArray) const {
        if (dictArray) {
            for (int i = 0; i < streamCount; i++) {
                if (dictArray[i]) {
                    av_dict_free(&dictArray[i]);
                }
            }
            av_freep(&dictArray);
        }
    }
};

#define RGYPOOLAV_DEBUG 0
#define RGYPOOLAV_COUNT 0

template<typename T, T *Talloc(), void Tunref(T* ptr), void Tfree(T** ptr)>
class RGYPoolAV {
private:
    RGYQueueMPMP<T*> queue;
#if RGYPOOLAV_COUNT
    std::atomic<uint64_t> allocated, reused;
#endif
public:
    RGYPoolAV() : queue()
#if RGYPOOLAV_COUNT
        , allocated(0), reused(0)
#endif
    { queue.init(); }
    ~RGYPoolAV() {
#if RGYPOOLAV_COUNT
        fprintf(stderr, "RGYPoolAV: allocated %lld, reused %lld\n", allocated, reused);
#endif
        queue.close([](T **ptr) { Tfree(ptr); });
    }
    std::unique_ptr<T, RGYAVDeleter<T>> getUnique(T *ptr) {
        return std::unique_ptr<T, RGYAVDeleter<T>>(ptr, RGYAVDeleter<T>([this](T **ptr) { returnFree(ptr); }));
    }
    std::unique_ptr<T, RGYAVDeleter<T>> getFree() {
#if RGYPOOLAV_DEBUG
        T *ptr = Talloc();
#else
        T *ptr = nullptr;
        if (!queue.front_copy_and_pop_no_lock(&ptr) || ptr == nullptr) {
            ptr = Talloc();
#if RGYPOOLAV_COUNT
            allocated++;
#endif
        }
#if RGYPOOLAV_COUNT
        else {
            reused++;
        }
#endif
#endif
        return getUnique(ptr);
    }
    void returnFree(T **ptr) {
        if (ptr == nullptr || *ptr == nullptr) return;
#if RGYPOOLAV_DEBUG
        Tfree(ptr);
#else
        Tunref(*ptr);
        queue.push(*ptr);
        *ptr = nullptr;
#endif
    }
};

using RGYPoolAVPacket = RGYPoolAV<AVPacket, av_packet_alloc, av_packet_unref, av_packet_free>;
using RGYPoolAVFrame = RGYPoolAV<AVFrame, av_frame_alloc, av_frame_unref, av_frame_free>;

typedef struct CodecMap {
    AVCodecID avcodec_id;   //avcodecのコーデックID
    RGY_CODEC rgy_codec; //QSVのfourcc
} CodecMap;

//QSVでデコード可能なコーデックのリスト
static const CodecMap HW_DECODE_LIST[] = {
    { AV_CODEC_ID_H264,       RGY_CODEC_H264 },
    { AV_CODEC_ID_HEVC,       RGY_CODEC_HEVC },
    { AV_CODEC_ID_MPEG2VIDEO, RGY_CODEC_MPEG2 },
#if !ENCODER_VCEENC
    { AV_CODEC_ID_VP8,        RGY_CODEC_VP8 },
#endif
    { AV_CODEC_ID_VP9,        RGY_CODEC_VP9 },
#if !ENCODER_QSV
    { AV_CODEC_ID_VC1,        RGY_CODEC_VC1   },
#endif
#if ENCODER_NVENC
    { AV_CODEC_ID_MPEG1VIDEO, RGY_CODEC_MPEG1 },
    { AV_CODEC_ID_MPEG4,      RGY_CODEC_MPEG4 },
#endif
    { AV_CODEC_ID_AV1,        RGY_CODEC_AV1 },
#if ENCODER_QSV && defined(AV_CODEC_ID_H266)
    { AV_CODEC_ID_VVC,        RGY_CODEC_VVC },
#endif
    //{ AV_CODEC_ID_WMV3,       RGY_CODEC_VC1   },
};

static inline AVCodecID getAVCodecId(RGY_CODEC codec) {
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++)
        if (HW_DECODE_LIST[i].rgy_codec == codec)
            return HW_DECODE_LIST[i].avcodec_id;
    return AV_CODEC_ID_NONE;
}

static const AVPixelFormat HW_DECODE_PIXFMT_LIST[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P10LE,
#if ENCODER_NVENC
    AV_PIX_FMT_YUV420P12LE,
#endif
};

static const int AVQSV_DEFAULT_AUDIO_BITRATE = 192;

static inline bool av_isvalid_q(AVRational q) {
    return q.den * q.num != 0;
}
template<typename T>
static inline AVRational av_make_q(rgy_rational<T> r) {
    return av_make_q(r.n(), r.d());
}
static rgy_rational<int> to_rgy(AVRational r) {
    return rgy_rational<int>(r.num, r.den);
}

static inline bool avcodecIsCopy(const TCHAR *codec) {
    return codec == nullptr || 0 == _tcsicmp(codec, RGY_AVCODEC_COPY);
}
static inline bool avcodecIsAuto(const TCHAR *codec) {
    return codec != nullptr && 0 == _tcsicmp(codec, RGY_AVCODEC_AUTO);
}
static inline bool avcodecIsCopy(const tstring& codec) {
    return codec.length() == 0 || avcodecIsCopy(codec.c_str());
}
static inline bool avcodecIsAuto(const tstring &codec) {
    return codec.length() > 0 && avcodecIsAuto(codec.c_str());
}

//AV_LOG_TRACE    56 - RGY_LOG_TRACE -3
//AV_LOG_DEBUG    48 - RGY_LOG_DEBUG -2
//AV_LOG_VERBOSE  40 - RGY_LOG_MORE  -1
//AV_LOG_INFO     32 - RGY_LOG_INFO   0
//AV_LOG_WARNING  24 - RGY_LOG_WARN   1
//AV_LOG_ERROR    16 - RGY_LOG_ERROR  2
static inline RGYLogLevel log_level_av2rgy(int level) {
    return (RGYLogLevel)clamp((AV_LOG_INFO / 8) - (level / 8), RGY_LOG_TRACE, RGY_LOG_ERROR);
}

static inline int log_level_rgy2av(RGYLogLevel level) {
    return clamp(AV_LOG_INFO - level * 8, AV_LOG_QUIET, AV_LOG_TRACE);
}

//"<mes> for codec"型のエラーメッセージを作成する
static tstring errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec) {
    return mes + tstring(_T(" for ")) + char_to_tstring(avcodec_get_name(targetCodec)) + tstring(_T(".\n"));
};

static const AVRational HW_NATIVE_TIMEBASE = { 1, (int)HW_TIMEBASE };
static const TCHAR *AVCODEC_DLL_NAME[] = {
    _T("avcodec-61.dll"), _T("avformat-61.dll"), _T("avutil-59.dll"), _T("avfilter-10.dll"), _T("swresample-5.dll")
#if ENABLE_LIBAVDEVICE
    , _T("avdevice-61.dll")
#endif
};

enum RGYAVCodecType : uint32_t {
    RGY_AVCODEC_DEC = 0x01,
    RGY_AVCODEC_ENC = 0x02,
};

enum RGYAVFormatType : uint32_t {
    RGY_AVFORMAT_DEMUX = 0x01,
    RGY_AVFORMAT_MUX   = 0x02,
};

static inline void pktFlagSetTrackID(AVPacket *pkt, const int trackID) {
    pkt->flags = (int)(((uint32_t)pkt->flags & 0xffff) | ((uint32_t)trackID << 16)); //flagsの上位16bitには、trackIdへのポインタを格納しておく
}

static inline int pktFlagGetTrackID(const AVPacket *pkt) {
    return (int)((uint32_t)pkt->flags >> 16);
}

// av_rescale_qのラッパー (v * from / to)
int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to);

// AVCodecContext::ticks_per_frameの代わり
// For some codecs, the time base is closer to the field rate than the frame rate.
// Most notably, H.264 and MPEG-2 specify time_base as half of frame duration
// if no telecine is used ...
// Set to time_base ticks per frame. Default 1, e.g., H.264/MPEG-2 set it to 2.
int getCodecTickPerFrames(const AVCodecID codecID);

//NV_ENC_PIC_STRUCTから、AVFieldOrderを返す
AVFieldOrder picstrcut_rgy_to_avfieldorder(RGY_PICSTRUCT picstruct);

//AVFrameのdurationを取得
int64_t rgy_avframe_get_duration(const AVFrame *frame);
int64_t& rgy_avframe_get_duration_ref(AVFrame *frame);

//AVFrameのインタレ関連フラグの確認
bool rgy_avframe_interlaced(const AVFrame *frame);
bool rgy_avframe_tff_flag(const AVFrame *frame);

//AVFrameの情報からRGY_PICSTRUCTを返す
RGY_PICSTRUCT picstruct_avframe_to_rgy(const AVFrame *frame);

//avcodecのエラーを表示
tstring qsv_av_err2str(int ret);

//コーデックが一致するか確認
bool avcodec_equal(const std::string& codec, const AVCodecID id);

//コーデックが存在するか確認
bool avcodec_exists(const std::string& codec, const AVMediaType type = AVMEDIA_TYPE_NB);
bool avcodec_exists_video(const std::string& codec);
bool avcodec_exists_audio(const std::string& codec);
bool avcodec_exists_subtitle(const std::string& codec);
bool avcodec_exists_data(const std::string& codec);

//コーデックの種類を表示
tstring get_media_type_string(AVCodecID codecId);

// trackの言語を取得
std::string getTrackLang(const AVStream *stream);

//必要なavcodecのdllがそろっているかを確認
bool check_avcodec_dll();

//avcodecのdllが存在しない場合のエラーメッセージ
tstring error_mes_avcodec_dll_not_found();

//avcodecのライセンスがLGPLであるかどうかを確認
bool checkAvcodecLicense();

//avqsvでサポートされている動画コーデックを表示
tstring getHWDecSupportedCodecList();

//利用可能な音声エンコーダ/デコーダを表示
tstring getAVCodecs(RGYAVCodecType flag, const std::vector<AVMediaType> mediatype);

//音声エンコーダで利用可能なプロファイルのリストを作成
std::vector<tstring> getAudioPofileList(const tstring& codec_name);

//利用可能なフォーマットを表示
tstring getAVFormats(RGYAVFormatType flag);

//利用可能なフィルターを表示
tstring getAVFilters();

// AVFrameのdeep copyを行う
void av_frame_deep_copy(AVFrame *copyFrame, const AVFrame *frame);

static bool channelLayoutSet(const RGYChannelLayout *a) {
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
    return a->nb_channels > 0;
#else
    return *a != 0;
#endif
}

static bool channelLayoutOrderUnspec([[maybe_unused]] const RGYChannelLayout *a) {
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
    return a->order == AV_CHANNEL_ORDER_UNSPEC;
#else
    return false;
#endif
}

static bool channelLayoutCompare(const RGYChannelLayout *a, const RGYChannelLayout *b) {
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
    return av_channel_layout_compare(a, b) != 0;
#else
    return *a != *b;
#endif
}

uniuqeRGYChannelLayout createChannelLayoutEmpty();
uniuqeRGYChannelLayout createChannelLayoutCopy(const RGYChannelLayout *ch_layout);

//コーデックのサポートするチャンネルレイアウトを取得
std::vector<RGYChannelLayout> getChannelLayoutSupportedCodec(const AVCodec *codec);

//チェンネル数をマスクから取得
int getChannelCount(const uint64_t channel_layout);
int getChannelCount(const RGYChannelLayout *channel_layout);
int getChannelCount(const AVCodecParameters *codecpar);
int getChannelCount(const AVCodecContext *ctx);

//チャンネルレイアウトを取得
uniuqeRGYChannelLayout getChannelLayout(const AVCodecContext *ctx);

//チャンネルレイアウトを表示
std::string getChannelLayoutChar(int channels, uint64_t channel_layout);
tstring getChannelLayoutString(int channels, uint64_t channel_layout);
std::string getChannelLayoutChar(const RGYChannelLayout *ch_layout);
tstring getChannelLayoutString(const RGYChannelLayout *ch_layout);
std::string getChannelLayoutChar(const AVCodecContext *ctx);
tstring getChannelLayoutString(const AVCodecContext *ctx);

//チャンネルレイアウトのマスクを文字列から取得
uint64_t getChannelLayoutMask(const std::string& channel_layout_str);
uniuqeRGYChannelLayout getChannelLayoutFromString(const std::string& channel_layout_str);

//デフォルトのチャンネルレイアウトを取得
uniuqeRGYChannelLayout getDefaultChannelLayout(const int nb_channels);

int getChannelLayoutIndexFromChannel(const RGYChannelLayout *ch_layout, const RGYChannel channel);
RGYChannel getChannelLayoutChannelFromIndex(const RGYChannelLayout *ch_layout, const int index);

bool ChannelLayoutExists(const RGYChannelLayout *target, const AVCodec *codec);

//時刻を表示
std::string getTimestampChar(int64_t ts, const AVRational& timebase);
tstring getTimestampString(int64_t ts, const AVRational& timebase);

//tag関連
uint32_t tagFromStr(std::string tagstr);
std::string tagToStr(uint32_t tag);

//AVStreamのside data関連
template<typename T>
std::unique_ptr<T, RGYAVDeleter<T>> AVStreamGetSideData(const AVStream *stream, const AVPacketSideDataType type, size_t& side_data_size) {
    std::unique_ptr<T, RGYAVDeleter<T>> side_data_copy(nullptr, RGYAVDeleter<T>(av_freep));
#if AVCODEC_PAR_CODED_SIDE_DATA_AVAIL
    auto side_data = av_packet_side_data_get(stream->codecpar->coded_side_data, stream->codecpar->nb_coded_side_data, type);
    if (side_data && side_data->type == type) {
        side_data_size = side_data->size;
        side_data_copy = unique_ptr<T, RGYAVDeleter<T>>((T *)av_malloc(side_data->size + AV_INPUT_BUFFER_PADDING_SIZE), RGYAVDeleter<T>(av_freep));
        memcpy(side_data_copy.get(), side_data->data, side_data->size);
    }
#else
    std::remove_pointer<RGYArgN<2U, decltype(av_stream_get_side_data)>::type>::type size = 0;
    auto side_data = av_stream_get_side_data(stream, type, &size);
    side_data_size = size;
    if (side_data) {
        side_data_copy = unique_ptr<T, RGYAVDeleter<T>>((T *)av_malloc(side_data_size + AV_INPUT_BUFFER_PADDING_SIZE), RGYAVDeleter<T>(av_freep));
        memcpy(side_data_copy.get(), side_data, side_data_size);
    }
#endif
    return side_data_copy;
}

template<typename T>
int AVStreamAddSideData(AVStream *stream, const AVPacketSideDataType type, std::unique_ptr<T, RGYAVDeleter<T>>& side_data, const size_t side_data_size) {
#if AVCODEC_PAR_CODED_SIDE_DATA_AVAIL
    auto ptr = av_packet_side_data_add(&stream->codecpar->coded_side_data, &stream->codecpar->nb_coded_side_data, type, (void *)side_data.get(), side_data_size, 0);
    int ret = ptr ? 0 : AVERROR(ENOMEM);
#else
    int ret = av_stream_add_side_data(stream, type, (uint8_t *)side_data.get(), side_data_size);
#endif
    if (ret == 0) {
        side_data.release();
    }
    return ret;
}

static void AVStreamCopySideData(AVStream *streamDst, const AVStream *streamSrc) {
#if AVCODEC_PAR_CODED_SIDE_DATA_AVAIL
    for (int i = 0; i < streamSrc->codecpar->nb_coded_side_data; i++) {
        const auto& side_data = streamSrc->codecpar->coded_side_data[i];
        av_packet_side_data_add(&streamDst->codecpar->coded_side_data, &streamDst->codecpar->nb_coded_side_data, side_data.type, side_data.data, side_data.size, 0);
    }
#else
    for (int i = 0; i < streamSrc->nb_side_data; i++) {
        const AVPacketSideData *const sidedataSrc = &streamSrc->side_data[i];
        uint8_t *const dst_data = av_stream_new_side_data(streamDst, sidedataSrc->type, sidedataSrc->size);
        memcpy(dst_data, sidedataSrc->data, sidedataSrc->size);
    }
#endif
}

//利用可能なプロトコル情報のリストを取得
vector<std::string> getAVProtocolList(int bOutput);

//利用可能なプロトコル情報を取得
tstring getAVProtocols();

//protocolを使用
bool usingAVProtocols(const std::string& filename, int bOutput);

//バージョン情報の取得
tstring getAVVersions();

//avdeviceの初期化
bool initAVDevices();

//avdeviceのリストを取得
tstring getAVDevices();

MAP_PAIR_0_1_PROTO(csp, avpixfmt, AVPixelFormat, rgy, RGY_CSP);

#define AV_DISPOSITION_UNSET (0x00000000)
#define AV_DISPOSITION_COPY  (0xffffffff)
MAP_PAIR_0_1_PROTO(disposition, str, tstring, av, uint32_t);

uint32_t parseDisposition(const tstring &disposition_str);
tstring getDispositionStr(uint32_t disposition);
RGYDOVIProfile getStreamDOVIProfile(const AVStream *stream);

#else
#define AV_NOPTS_VALUE (-1)
class RGYPoolAVPacket;
class RGYPoolAVFrame;
static bool avcodec_exists_video([[maybe_unused]] const std::string& codec) { return false; }
static bool avcodec_exists_audio([[maybe_unused]] const std::string& codec) { return false; }
static bool avcodec_exists_subtitle([[maybe_unused]] const std::string& codec) { return false; }
static bool avcodec_exists_data([[maybe_unused]] const std::string& codec) { return false; }
#endif //ENABLE_AVSW_READER

#endif //__RGY_AVUTIL_H__
