﻿// -----------------------------------------------------------------------------------------
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

#include <fcntl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <fstream>
#include <iostream>
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_filesystem.h"
#include "rgy_output_avcodec.h"
#include "rgy_avlog.h"
#include "rgy_bitstream.h"
#include "rgy_codepage.h"

#define WRITE_PTS_DEBUG (0)

static bool format_is_mp4(const AVFormatContext *formatCtx) {
    static const char *FORMAT_NAME_MP4[] = {
        "mov",
        "3gp",
        "mp4",
        "psp",
        "3g2",
        "ipod",
        "ismv",
        "f4v",
        "avif"
    };
    for (int i = 0; i < _countof(FORMAT_NAME_MP4); i++) {
        if (0 == _stricmp(formatCtx->oformat->name, FORMAT_NAME_MP4[i])) {
            return true;
        }
    }
    return false;
}
static bool format_is_mkv(const AVFormatContext *formatCtx) {
    static const char *FORMAT_NAME_MKV[] = {
        "matroska",
        "webm"
    };
    for (int i = 0; i < _countof(FORMAT_NAME_MKV); i++) {
        if (0 == _stricmp(formatCtx->oformat->name, FORMAT_NAME_MKV[i])) {
            return true;
        }
    }
    return false;
}
static bool format_is_flv(const AVFormatContext *formatCtx) {
    return _stricmp(formatCtx->oformat->name, "flv") == 0;
}
static bool format_is_latm(const AVFormatContext *formatCtx) {
    return _stricmp(formatCtx->oformat->name, "latm") == 0;
}
static bool format_is_ivf(const AVFormatContext *formatCtx) {
    return _stricmp(formatCtx->oformat->name, "ivf") == 0;
}
static bool format_is_mpegts(const AVFormatContext *formatCtx) {
    return _stricmp(formatCtx->oformat->name, "mpegts") == 0;
}
static bool format_is_y4m(const AVFormatContext *formatCtx) {
    return _stricmp(formatCtx->oformat->name, "yuv4mpegpipe") == 0;
}

#if ENABLE_AVSW_READER
#if USE_CUSTOM_IO
static int funcReadPacket(void *opaque, uint8_t *buf, int buf_size) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->readPacket(buf, buf_size);
}
static int funcWritePacket(void *opaque, const uint8_t *buf, int buf_size) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->writePacket(buf, buf_size);
}
static int64_t funcSeek(void *opaque, int64_t offset, int whence) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->seek(offset, whence);
}
#endif //USE_CUSTOM_IO

AVMuxFormat::AVMuxFormat() :
    filename(nullptr),
    formatCtx(nullptr),
    metadataStr(),
    outputFmt(nullptr),
#if USE_CUSTOM_IO
    AVOutBuffer(nullptr),
    AVOutBufferSize(0),
    fpOutput(nullptr),
    outputBuffer(nullptr),
    outputBufferSize(0),
#endif
    streamError(false),
    isMatroska(false),
    isPipe(false),
    fileHeaderWritten(false),
    headerOptions(nullptr),
    disableMp4Opt(false),
    lowlatency(false),
    offsetVideoDtsAdvance(false),
    allowOtherNegativePts(false),
    timestampPassThrough(false),
    fpTsLogMtx(),
    fpTsLogFile() {
}

AVMuxVideo::AVMuxVideo() :
    codec(nullptr),
    codecCtx(nullptr),
    outputFps(av_make_q(0, 0)),
    streamOut(nullptr),
    dtsUnavailable(false),
    inputStreamTimebase(av_make_q(0,0)),
    inputFirstKeyPts(0),
    bitstreamTimebase(av_make_q(0, 0)),
    timestampList(),
    fpsBaseNextDts(0),
    fpTsLogFile(),
    hdrBitstream(),
    hdr10plus(nullptr),
    hdr10plusMetadataCopy(false),
    doviProfileSrc(RGY_DOVI_PROFILE_UNSET),
    doviProfileDst(RGY_DOVI_PROFILE_UNSET),
    doviRpu(nullptr),
    doviRpuMetadataCopy(false),
    doviRpuConvertParam(),
    timestamp(nullptr),
    pktOut(nullptr),
    pktParse(nullptr),
    prevEncodeFrameId(-1),
    prevInputFrameId(-1),
    parserCtx(nullptr),
    parserStreamPos(0),
    afs(false),
    debugDirectAV1Out(false),
    rawVideoCodec(nullptr),
    rawVideoCodecCtx(),
    rawVideoFrame(),
    rawVideoConvert(),
    simdCsp(RGY_SIMD::SIMD_ALL),
    parse_nal_h264(get_parse_nal_unit_h264_func()),
    parse_nal_hevc(get_parse_nal_unit_hevc_func()) {
}

AVMuxAudio::AVMuxAudio() :
    inTrackId(0),
    inSubStream(0),
    streamIn(nullptr),
    streamIndexIn(0),
    streamOut(nullptr),
    packetWritten(0),
    dec_rescale_delta(0),
    outCodecDecode(nullptr),
    outCodecDecodeCtx(nullptr),
    outCodecEncode(nullptr),
    outCodecEncodeCtx(nullptr),
    decodeNextPts(0),
    ignoreDecodeError(0),
    decodeError(0),
    encodeError(false),
    flushed(false),
    filterInChannels(0),
    filterInChannelLayout(createChannelLayoutEmpty()),
    filterInSampleRate(0),
    filterInSampleFmt(AV_SAMPLE_FMT_NONE),
    filter(nullptr),
    filterBufferSrcCtx(nullptr),
    filterBufferSinkCtx(nullptr),
    filterAudioFormat(nullptr),
    filterGraph(nullptr),
    audioResampler(RGY_RESAMPLER_SWR),
    audioResamplerPrm(),
    decodedFrameCache(nullptr),
    channelMapping(),
    streamChannelSelect(),
    streamChannelOut(),
    bsfc(nullptr),
    bsfErrorFromStart(0),
    outputSampleOffset(0),
    outputSamples(0),
    lastPtsIn(0),
    lastPtsOut(0),
    fpTsLogFile() {

}

AVSubtitleData::AVSubtitleData() :
    decodecSub(),
    origPts(-1),
    origDuration(-1) {

}

AVSubtitleData::~AVSubtitleData() {
}

AVMuxOther::AVMuxOther() :
    inTrackId(0),
    streamIn(nullptr),
    streamIndexIn(0),
    streamInTimebase(av_make_q(0, 0)),
    streamOut(nullptr),
    outCodecDecode(nullptr),
    outCodecDecodeCtx(nullptr),
    outCodecEncode(nullptr),
    outCodecEncodeCtx(nullptr),
    bufConvert(nullptr),
    bsfc(nullptr),
    decodedSub(),
    lastPtsOut(AV_NOPTS_VALUE) {

}

AVMuxThreadWorker::AVMuxThreadWorker() :
    thread(),
    thAbort(false),
    sentEOS(false),
    heEventPktAdded(nullptr),
    heEventClosing(nullptr),
    qPackets() {}

AVMuxThreadWorker::~AVMuxThreadWorker() {
    if (heEventPktAdded) {
        CloseEvent(heEventPktAdded);
        heEventPktAdded = nullptr;
    }
    if (heEventClosing) {
        CloseEvent(heEventClosing);
        heEventClosing = nullptr;
    }
    qPackets.close();
}

void AVMuxThreadWorker::close() {
    thAbort = true;
    if (thread.joinable()) {
        //ここに来た時に、まだメインスレッドがループ中の可能性がある
        //その場合、SetEvent(thread.heEventPktAddedOutput)を一度やるだけだと、
        //そのあとにResetEvent(thread.heEventPktAddedOutput)が発生してしまい、
        //ここでスレッドが停止してしまう。
        //これを回避するため、thread.heEventClosingOutputがセットされるまで、
        //SetEvent(thread.heEventPktAddedOutput)を実行し続ける必要がある。
        while (WAIT_TIMEOUT == WaitForSingleObject(heEventClosing, 100)) {
            SetEvent(heEventPktAdded);
        }
        thread.join();
        if (heEventPktAdded) {
            CloseEvent(heEventPktAdded);
            heEventPktAdded = nullptr;
        }
        if (heEventClosing) {
            CloseEvent(heEventClosing);
            heEventClosing = nullptr;
        }
        qPackets.close();
    }
}

AVMuxThreadAudio::AVMuxThreadAudio() : encode(), process() {};

AVMuxThreadAudio::~AVMuxThreadAudio() {}

void AVMuxThreadAudio::closeEncode() {
    encode.close();
}

void AVMuxThreadAudio::closeProcess() {
    process.close();
}

bool AVMuxThreadAudio::threadActiveEncode() {
    return encode.thread.joinable();
}

bool AVMuxThreadAudio::threadActiveProcess() {
    return process.thread.joinable();
}

#if ENABLE_AVCODEC_OUT_THREAD
AVMuxThread::AVMuxThread() :
    enableOutputThread(false),
    enableAudProcessThread(false),
    enableAudEncodeThread(false),
    thOutput(),
    thRawVideo(),
    qVideobitstreamFreeI(),
    qVideobitstreamFreePB(),
    qVideobitstream(),
    thAud(),
    streamOutMaxDts(0),
    queueInfo(nullptr) {
}
#endif

AVMux::AVMux() :
    format(),
    video(),
    videoAV1Merge(),
    audio(),
    other(),
    trim(),
#if ENABLE_AVCODEC_OUT_THREAD
    thread(),
#endif
    poolPkt(nullptr),
    poolFrame(nullptr) {
}

const AVRational RGYOutputAvcodec::QUEUE_DTS_TIMEBASE = av_make_q(1, 90000);

RGYOutputAvcodec::RGYOutputAvcodec() : m_Mux(), m_AudPktBufFileHead() {
    m_strWriterName = _T("avout");
}

RGYOutputAvcodec::~RGYOutputAvcodec() {
    Close();
}

void RGYOutputAvcodec::CloseOther(AVMuxOther *muxOther) {
    //close decoder
    if (muxOther->outCodecDecodeCtx) {
        avcodec_free_context(&muxOther->outCodecDecodeCtx);
        muxOther->outCodecDecodeCtx = nullptr;
        muxOther->outCodecDecode = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecDecodeCtx.\n"));
    }

    //close encoder
    if (muxOther->outCodecEncodeCtx) {
        avcodec_free_context(&muxOther->outCodecEncodeCtx);
        muxOther->outCodecEncodeCtx = nullptr;
        muxOther->outCodecEncode = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecEncodeCtx.\n"));
    }
    if (muxOther->bufConvert) {
        av_free(muxOther->bufConvert);
        muxOther->bufConvert = nullptr;
    }

    if (muxOther->bsfc) {
        av_bsf_free(&muxOther->bsfc);
    }
    muxOther->streamIn = nullptr;
    muxOther->streamOut = nullptr;
    AddMessage(RGY_LOG_DEBUG, _T("Closed other.\n"));
}

void RGYOutputAvcodec::CloseAudio(AVMuxAudio *muxAudio) {
    //close decoder
    if (muxAudio->outCodecDecodeCtx
        && muxAudio->inSubStream == 0) { //サブストリームのものは単なるコピーなので開放不要
        avcodec_free_context(&muxAudio->outCodecDecodeCtx);
        muxAudio->outCodecDecodeCtx = nullptr;
        muxAudio->outCodecDecode = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecDecodeCtx.\n"));
    }

    //close encoder
    if (muxAudio->outCodecEncodeCtx) {
        avcodec_free_context(&muxAudio->outCodecEncodeCtx);
        muxAudio->outCodecEncodeCtx = nullptr;
        muxAudio->outCodecEncode = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecEncodeCtx.\n"));
    }

    //close filter
    if (muxAudio->filterGraph) {
        avfilter_graph_free(&muxAudio->filterGraph);
    }
    muxAudio->filter = nullptr;
    muxAudio->filterBufferSrcCtx = nullptr;
    muxAudio->filterBufferSinkCtx = nullptr;
    muxAudio->filterAudioFormat = nullptr;

    if (muxAudio->bsfc) {
        av_bsf_free(&muxAudio->bsfc);
    }
    muxAudio->streamIn = nullptr;
    muxAudio->streamOut = nullptr;
    muxAudio->fpTsLogFile.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed audio.\n"));
}

void RGYOutputAvcodec::CloseVideo(AVMuxVideo *muxVideo) {
    if (muxVideo->parserCtx) {
        av_parser_close(muxVideo->parserCtx);
        muxVideo->parserCtx = nullptr;
    }
    if (muxVideo->codecCtx) {
        avcodec_free_context(&muxVideo->codecCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed video context.\n"));
    }
    muxVideo->codec = nullptr;
    muxVideo->streamOut = nullptr;
    muxVideo->fpTsLogFile.reset();
    m_Mux.video.timestampList.clear();
    if (m_Mux.video.pktOut) {
        av_packet_unref(m_Mux.video.pktOut);
        av_packet_free(&m_Mux.video.pktOut);
    }
    if (m_Mux.video.pktParse) {
        av_packet_unref(m_Mux.video.pktParse);
        av_packet_free(&m_Mux.video.pktParse);
    }
    m_Mux.video.hdr10plus = nullptr;
    m_Mux.video.doviRpu = nullptr;
    m_Mux.video.timestamp = nullptr;

    if (muxVideo->rawVideoCodecCtx) {
        muxVideo->rawVideoCodecCtx.reset();
        muxVideo->rawVideoCodec = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed raw video encoder context.\n"));
    }

    AddMessage(RGY_LOG_DEBUG, _T("Closed video.\n"));
}

void RGYOutputAvcodec::CloseFormat(AVMuxFormat *muxFormat) {
    if (muxFormat->formatCtx) {
        if (!muxFormat->streamError && m_Mux.format.fileHeaderWritten) {
            av_write_trailer(muxFormat->formatCtx);
        }
#if USE_CUSTOM_IO
        if (!muxFormat->fpOutput) {
#endif
            avio_close(muxFormat->formatCtx->pb);
            AddMessage(RGY_LOG_DEBUG, _T("Closed AVIO Context.\n"));
#if USE_CUSTOM_IO
        }
#endif
        avformat_free_context(muxFormat->formatCtx);
        muxFormat->formatCtx = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
    muxFormat->fpTsLogFile.reset();
#if USE_CUSTOM_IO
    if (muxFormat->fpOutput) {
        fflush(muxFormat->fpOutput);
        fclose(muxFormat->fpOutput);
        muxFormat->fpOutput = nullptr;
        AddMessage(RGY_LOG_DEBUG, _T("Closed File Pointer.\n"));
    }

    if (muxFormat->AVOutBuffer) {
        av_free(muxFormat->AVOutBuffer);
        muxFormat->AVOutBuffer = nullptr;
    }

    if (muxFormat->outputBuffer) {
        free(muxFormat->outputBuffer);
        muxFormat->outputBuffer = nullptr;
    }
#endif //USE_CUSTOM_IO
    AddMessage(RGY_LOG_DEBUG, _T("Closed format.\n"));
}

void RGYOutputAvcodec::CloseQueues() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.qVideobitstream.close();
    m_Mux.thread.qVideobitstreamFreeI.close([](RGYBitstream *pBitstream) { pBitstream->clear(); });
    m_Mux.thread.qVideobitstreamFreePB.close([](RGYBitstream *pBitstream) { pBitstream->clear(); });
    AddMessage(RGY_LOG_DEBUG, _T("closed queues...\n"));
#endif
}

void RGYOutputAvcodec::CloseThread() {
#if ENABLE_AVCODEC_OUT_THREAD
    // process -> encode -> output の順に終了させる
    for (auto& [mux, thread] : m_Mux.thread.thAud) {
        if (thread->process.thread.joinable()) {
            thread->closeProcess();
            const auto target = (mux) ? strsprintf(_T("%d.%d"), trackID(mux->inTrackId), mux->inSubStream) : tstring(_T("default"));
            AddMessage(RGY_LOG_DEBUG, _T("closed audio process thread %s.\n"), target.c_str());
        }
    }
    for (auto& [mux, thread] : m_Mux.thread.thAud) {
        if (thread->encode.thread.joinable()) {
            thread->closeEncode();
            const auto target = (mux) ? strsprintf(_T("%d.%d"), trackID(mux->inTrackId), mux->inSubStream) : tstring(_T("default"));
            AddMessage(RGY_LOG_DEBUG, _T("closed audio encode thread %s.\n"), target.c_str());
        }
    }
    if (m_Mux.thread.thRawVideo) {
        m_Mux.thread.thRawVideo->close();
        AddMessage(RGY_LOG_DEBUG, _T("closed raw video thread...\n"));
    }
    if (m_Mux.thread.thOutput) {
        m_Mux.thread.thOutput->close();
        AddMessage(RGY_LOG_DEBUG, _T("closed output thread...\n"));
    }
    CloseQueues();
#endif
}

void RGYOutputAvcodec::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    CloseThread();
    CloseFormat(&m_Mux.format);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        CloseAudio(&m_Mux.audio[i]);
    }
    m_Mux.audio.clear();
    for (int i = 0; i < (int)m_Mux.other.size(); i++) {
        CloseOther(&m_Mux.other[i]);
    }
    m_Mux.other.clear();
    CloseVideo(&m_Mux.video);
    m_strOutputInfo.clear();
    m_encSatusInfo.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

AVCodecID RGYOutputAvcodec::getAVCodecId(RGY_CODEC codec) {
    if (codec == RGY_CODEC_RAW) {
        return AV_CODEC_ID_RAWVIDEO;
    }
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++)
        if (HW_DECODE_LIST[i].rgy_codec == codec)
            return HW_DECODE_LIST[i].avcodec_id;
    return AV_CODEC_ID_NONE;
}
bool RGYOutputAvcodec::codecIDIsPCM(AVCodecID targetCodec) {
    static const auto pcmCodecs = make_array<AVCodecID>(
        AV_CODEC_ID_FIRST_AUDIO,
        AV_CODEC_ID_PCM_S16LE,
        AV_CODEC_ID_PCM_S16BE,
        AV_CODEC_ID_PCM_U16LE,
        AV_CODEC_ID_PCM_U16BE,
        AV_CODEC_ID_PCM_S8,
        AV_CODEC_ID_PCM_U8,
        AV_CODEC_ID_PCM_MULAW,
        AV_CODEC_ID_PCM_ALAW,
        AV_CODEC_ID_PCM_S32LE,
        AV_CODEC_ID_PCM_S32BE,
        AV_CODEC_ID_PCM_U32LE,
        AV_CODEC_ID_PCM_U32BE,
        AV_CODEC_ID_PCM_S24LE,
        AV_CODEC_ID_PCM_S24BE,
        AV_CODEC_ID_PCM_U24LE,
        AV_CODEC_ID_PCM_U24BE,
        AV_CODEC_ID_PCM_S24DAUD,
        AV_CODEC_ID_PCM_ZORK,
        AV_CODEC_ID_PCM_S16LE_PLANAR,
        AV_CODEC_ID_PCM_DVD,
        AV_CODEC_ID_PCM_F32BE,
        AV_CODEC_ID_PCM_F32LE,
        AV_CODEC_ID_PCM_F64BE,
        AV_CODEC_ID_PCM_F64LE,
        AV_CODEC_ID_PCM_BLURAY,
        AV_CODEC_ID_PCM_LXF,
        AV_CODEC_ID_S302M,
        AV_CODEC_ID_PCM_S8_PLANAR,
        AV_CODEC_ID_PCM_S24LE_PLANAR,
        AV_CODEC_ID_PCM_S32LE_PLANAR,
        AV_CODEC_ID_PCM_S16BE_PLANAR
    );
    return (pcmCodecs.end() != std::find(pcmCodecs.begin(), pcmCodecs.end(), targetCodec));
}

AVCodecID RGYOutputAvcodec::PCMRequiresConversion(const AVCodecParameters *codecParm) {
    static const std::pair<AVCodecID, AVCodecID> pcmConvertCodecs[] = {
        { AV_CODEC_ID_FIRST_AUDIO,      AV_CODEC_ID_FIRST_AUDIO },
        { AV_CODEC_ID_PCM_DVD,          AV_CODEC_ID_FIRST_AUDIO },
        { AV_CODEC_ID_PCM_BLURAY,       AV_CODEC_ID_FIRST_AUDIO },
        { AV_CODEC_ID_PCM_S8_PLANAR,    AV_CODEC_ID_PCM_S8      },
        { AV_CODEC_ID_PCM_S16LE_PLANAR, AV_CODEC_ID_PCM_S16LE   },
        { AV_CODEC_ID_PCM_S16BE_PLANAR, AV_CODEC_ID_PCM_S16LE   },
        { AV_CODEC_ID_PCM_S16BE,        AV_CODEC_ID_PCM_S16LE   },
        { AV_CODEC_ID_PCM_S24LE_PLANAR, AV_CODEC_ID_PCM_S24LE   },
        { AV_CODEC_ID_PCM_S24BE,        AV_CODEC_ID_PCM_S24LE   },
        { AV_CODEC_ID_PCM_S32LE_PLANAR, AV_CODEC_ID_PCM_S32LE   },
        { AV_CODEC_ID_PCM_S32BE,        AV_CODEC_ID_PCM_S32LE   },
        { AV_CODEC_ID_PCM_F32BE,        AV_CODEC_ID_PCM_S32LE   },
        { AV_CODEC_ID_PCM_F64BE,        AV_CODEC_ID_PCM_S32LE   },
    };
    AVCodecID prmCodec = AV_CODEC_ID_NONE;
    for (int i = 0; i < _countof(pcmConvertCodecs); i++) {
        if (pcmConvertCodecs[i].first == codecParm->codec_id) {
            if (pcmConvertCodecs[i].second != AV_CODEC_ID_FIRST_AUDIO) {
                return pcmConvertCodecs[i].second;
            }
            switch (codecParm->bits_per_raw_sample) {
            case 32: prmCodec = AV_CODEC_ID_PCM_S32LE; break;
            case 24: prmCodec = AV_CODEC_ID_PCM_S24LE; break;
            case 8:  prmCodec = AV_CODEC_ID_PCM_S16LE; break;
            case 16:
            default: prmCodec = AV_CODEC_ID_PCM_S16LE; break;
            }
        }
    }
    if (prmCodec != AV_CODEC_ID_NONE) {
        AddMessage(RGY_LOG_DEBUG, _T("PCM requires conversion...\n"));
    }
    return prmCodec;
}

void RGYOutputAvcodec::SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecCtx->extradata)
        av_free(codecCtx->extradata);
    codecCtx->extradata_size = size;
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecCtx->extradata, data, size);
};

void RGYOutputAvcodec::SetExtraData(AVCodecParameters *codecParam, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecParam->extradata)
        av_free(codecParam->extradata);
    codecParam->extradata_size = size;
    codecParam->extradata      = (uint8_t *)av_malloc(codecParam->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecParam->extradata, data, size);
};

//音声のchannel_layoutを自動選択する
uniuqeRGYChannelLayout RGYOutputAvcodec::AutoSelectChannelLayout(const AVCodec *codec, const AVCodecContext *srcAudioCtx) {
    const int srcChannels = getChannelCount(srcAudioCtx);
    auto channelLayout = getChannelLayoutSupportedCodec(codec);
    if (codec == nullptr || channelLayout == nullptr) {
        return getDefaultChannelLayout(srcChannels);
    }
    for (int i = 0; channelLayoutSet(&channelLayout[i]); i++) {
        if (srcChannels == getChannelCount(&channelLayout[i])) {
            return createChannelLayoutCopy(&channelLayout[i]);
        }
    }
    //一致するチャンネルが見つからない場合、最も近いチャンネル数のものを取得するようにする
    int selectIdx = -1;
    int selectIdxDiff = std::numeric_limits<int>::max();
    for (int i = 0; channelLayoutSet(&channelLayout[i]); i++) {
        const int absDiff = std::abs(srcChannels - getChannelCount(&channelLayout[i]));
        if (absDiff < selectIdxDiff) {
            selectIdx = i;
            selectIdxDiff = absDiff;
        }
    }
    if (selectIdx >= 0) {
        return createChannelLayoutCopy(&channelLayout[selectIdx]);
    }
    return createChannelLayoutCopy(&channelLayout[0]);
}

int RGYOutputAvcodec::AutoSelectSamplingRate(const int *samplingRateList, int srcSamplingRate) {
    if (samplingRateList == nullptr) {
        return srcSamplingRate;
    }
    //一致するものがあれば、それを返す
    int i = 0;
    for (; samplingRateList[i]; i++) {
        if (srcSamplingRate == samplingRateList[i]) {
            return srcSamplingRate;
        }
    }
    //相対誤差が最も小さいものを選択する
    vector<double> diffrate(i);
    for (i = 0; samplingRateList[i]; i++) {
        diffrate[i] = std::abs(1 - samplingRateList[i] / (double)srcSamplingRate);
    }
    return samplingRateList[std::distance(diffrate.begin(), std::min_element(diffrate.begin(), diffrate.end()))];
}

AVSampleFormat RGYOutputAvcodec::AutoSelectSampleFmt(const AVSampleFormat *samplefmtList, const AVCodecContext *srcAudioCtx) {
    const AVSampleFormat srcFormat = srcAudioCtx->sample_fmt;
    if (samplefmtList == nullptr) {
        return srcFormat;
    }

    static const auto sampleFmtLevel = make_array<AVSampleFormat>(
        AV_SAMPLE_FMT_DBLP,
        AV_SAMPLE_FMT_DBL,
        AV_SAMPLE_FMT_FLTP,
        AV_SAMPLE_FMT_FLT,
        AV_SAMPLE_FMT_S32P,
        AV_SAMPLE_FMT_S32,
        AV_SAMPLE_FMT_S16P,
        AV_SAMPLE_FMT_S16,
        AV_SAMPLE_FMT_U8P,
        AV_SAMPLE_FMT_U8
    );
    for (const auto fmt : sampleFmtLevel) {
        for (int i = 0; samplefmtList[i] >= 0; i++) {
            if (fmt == samplefmtList[i]) {
                return fmt;
            }
        }
    }
    if (srcFormat == AV_SAMPLE_FMT_NONE) {
        return samplefmtList[0];
    }
    for (int i = 0; samplefmtList[i] >= 0; i++) {
        if (srcFormat == samplefmtList[i]) {
            return samplefmtList[i];
        }
    }
    return samplefmtList[0];
}

//音声のプロファイルを取得する
int RGYOutputAvcodec::AudioGetCodecProfile(tstring profile, AVCodecID codecId) {
    int selectedProfile = FF_PROFILE_UNKNOWN;
    auto codecDesc = avcodec_descriptor_get(codecId);
    if (codecDesc != nullptr) {
        const std::string codec_profile = tchar_to_string(profile);
        for (auto avprofile = codecDesc->profiles;
            avprofile != nullptr && avprofile->profile != FF_PROFILE_UNKNOWN;
            avprofile++) {
            if (_stricmp(avprofile->name, codec_profile.c_str()) == 0) {
                selectedProfile = avprofile->profile;
                break;
            }
        }
    }
    return selectedProfile;
}

//音声のプロファイル(文字列)を取得する
tstring RGYOutputAvcodec::AudioGetCodecProfileStr(int profile, AVCodecID codecId) {
    auto codecDesc = avcodec_descriptor_get(codecId);
    if (codecDesc != nullptr) {
        for (auto avprofile = codecDesc->profiles;
            avprofile != nullptr && avprofile->profile != FF_PROFILE_UNKNOWN;
            avprofile++) {
            if (avprofile->profile == profile && avprofile->name != nullptr) {
                return char_to_tstring(avprofile->name);
            }
        }
    }
    return _T("default");
}

RGY_ERR RGYOutputAvcodec::SetMetadata(AVDictionary **metadata, const AVDictionary *srcMetadata, const std::vector<tstring> &metadataOpt, const RGYMetadataCopyDefault defaultCopy, const tstring& trackName) {
    bool metadataCopyAll = false;
    bool metadataCopyLang = false;
    //デフォルトの操作
    switch (defaultCopy) {
    case RGY_METADATA_DEFAULT_CLEAR:
        metadataCopyAll = false;
        metadataCopyLang = false;
        break;
    case RGY_METADATA_DEFAULT_COPY_LANG_ONLY:
        metadataCopyAll = false;
        metadataCopyLang = true;
        break;
    case RGY_METADATA_DEFAULT_COPY:
        metadataCopyAll = true;
        metadataCopyLang = true;
        break;
    default:
        AddMessage(RGY_LOG_DEBUG, _T("Unknown default setting %d for %s metadata setting!\n"), defaultCopy, trackName.c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    if (metadata_clear(metadataOpt)) {
        metadataCopyAll = false;
        metadataCopyLang = false;
    } else if (metadata_copy(metadataOpt)) {
        metadataCopyAll = true;
        metadataCopyLang = true;
    }
    if (srcMetadata == nullptr) {
        metadataCopyAll = false;
        metadataCopyLang = false;
    }
    if (metadataCopyLang) {
        const auto language_data = av_dict_get(srcMetadata, "language", NULL, AV_DICT_MATCH_CASE);
        if (language_data) {
            av_dict_set(metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Set %s language: key %s, value %s\n"), trackName.c_str(), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
        }
    }
    if (metadataCopyAll) {
        for (AVDictionaryEntry *entry = nullptr;
            nullptr != (entry = av_dict_get(srcMetadata, "", entry, AV_DICT_IGNORE_SUFFIX));) {
            av_dict_set(metadata, entry->key, entry->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Copy %s Metadata: key %s, value %s\n"), trackName.c_str(), char_to_tstring(entry->key).c_str(), char_to_tstring(entry->value).c_str());
        }
    }
    //このあたりは矛盾することがあるので消去
    av_dict_set(&m_Mux.format.formatCtx->metadata, "duration", NULL, 0);
    av_dict_set(&m_Mux.format.formatCtx->metadata, "creation_time", NULL, 0);
    //ユーザー指定のパラメータの指定
    for (const auto& m : metadataOpt) {
        if (m == RGY_METADATA_CLEAR || m == RGY_METADATA_COPY) {
            continue;
        }
        const std::string m_utf8 = tchar_to_string(m, CODE_PAGE_UTF8);
        if (av_dict_parse_string(metadata, m_utf8.c_str(), "=", "", 0)) {
            AddMessage(RGY_LOG_WARN, _T("Failed to parse metadata \"%s\" for %s, ignored!\n"), m.c_str(), trackName.c_str());
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Set/Overwrite metadata \"%s\" for %s\n"), m.c_str(), trackName.c_str());
        }
    }
    return RGY_ERR_NONE;
}

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputAvcodec::InitVideo(const VideoInfo *videoOutputInfo, const AvcodecWriterPrm *prm) {
    m_Mux.format.formatCtx->video_codec_id = getAVCodecId(videoOutputInfo->codec);
    if (m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find codec id for video.\n"));
        return RGY_ERR_INVALID_CODEC;
    }
    if (NULL == (m_Mux.video.codec = avcodec_find_decoder(m_Mux.format.formatCtx->video_codec_id))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to codec for video codec %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_INVALID_CODEC;
    }
    if (NULL == (m_Mux.video.codecCtx = avcodec_alloc_context3(m_Mux.video.codec))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate context for video codec %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_INVALID_CODEC;
    }
    m_Mux.video.outputFps = av_make_q(videoOutputInfo->fpsN, videoOutputInfo->fpsD);
    AddMessage(RGY_LOG_DEBUG, _T("output video stream fps: %d/%d\n"), m_Mux.video.outputFps.num, m_Mux.video.outputFps.den);

    m_HEVCAlphaChannelMode = prm->HEVCAlphaChannelMode;
    m_enableHEVCAlphaChannelInfoSEIOverwrite = videoOutputInfo->codec == RGY_CODEC_HEVC && prm->HEVCAlphaChannel;
    if (m_enableHEVCAlphaChannelInfoSEIOverwrite) {
        AddMessage(RGY_LOG_DEBUG, _T("enableHEVCAlphaChannelInfoSEIFix : on\n"));
    }

    // raw videoなら、raw video encoderを初期化する
    if (m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_RAWVIDEO) {
        AddMessage(RGY_LOG_DEBUG, _T("Initializing raw video encoder for frame output...\n"));
        m_OutType = OUT_TYPE_SURFACE;
        
        //raw video encoderを見つける
        m_Mux.video.rawVideoCodec = avcodec_find_encoder(AV_CODEC_ID_RAWVIDEO);
        if (!m_Mux.video.rawVideoCodec) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to find raw video encoder.\n"));
            return RGY_ERR_INVALID_CODEC;
        }
        
        //raw video encoder contextを確保する
        m_Mux.video.rawVideoCodecCtx = std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>>(avcodec_alloc_context3(m_Mux.video.rawVideoCodec), RGYAVDeleter<AVCodecContext>(avcodec_free_context));
        if (!m_Mux.video.rawVideoCodecCtx) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate raw video encoder context.\n"));
            return RGY_ERR_NULL_PTR;
        }
        
        // Configure raw video encoder context
        m_Mux.video.rawVideoCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        m_Mux.video.rawVideoCodecCtx->codec_id = AV_CODEC_ID_RAWVIDEO;
        m_Mux.video.rawVideoCodecCtx->width = videoOutputInfo->dstWidth;
        m_Mux.video.rawVideoCodecCtx->height = videoOutputInfo->dstHeight;
        switch (RGY_CSP_CHROMA_FORMAT[videoOutputInfo->csp]) {
            case RGY_CHROMAFMT_YUV420:
                switch (RGY_CSP_BIT_DEPTH[videoOutputInfo->csp]) {
                case 8:  m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P; break;
                case 10: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P10LE; break;
                case 12: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P12LE; break;
                case 14: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P14LE; break;
                case 16: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P16LE; break;
                default:
                    AddMessage(RGY_LOG_ERROR, _T("Unsupported output bitdepth %d (%s) for raw output.\n"), RGY_CSP_BIT_DEPTH[videoOutputInfo->csp], RGY_CSP_NAMES[videoOutputInfo->csp]);
                    return RGY_ERR_UNSUPPORTED;
                }
                break;
#if ENCODER_NVENC
            case RGY_CHROMAFMT_YUV444:
                switch (RGY_CSP_BIT_DEPTH[videoOutputInfo->csp]) {
                case 8:  m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV444P; break;
                case 10: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV444P10LE; break;
                case 12: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV444P12LE; break;
                case 14: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV444P14LE; break;
                case 16: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_YUV444P16LE; break;
                default:
                    AddMessage(RGY_LOG_ERROR, _T("Unsupported output bitdepth %d (%s) for raw output.\n"), RGY_CSP_BIT_DEPTH[videoOutputInfo->csp], RGY_CSP_NAMES[videoOutputInfo->csp]);
                    return RGY_ERR_UNSUPPORTED;
                }
                break;
            case RGY_CHROMAFMT_RGB:
                switch (RGY_CSP_BIT_DEPTH[videoOutputInfo->csp]) {
                case 8:  m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_GBRP; break;
                case 10: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_GBRP10LE; break;
                case 16: m_Mux.video.rawVideoCodecCtx->pix_fmt = AV_PIX_FMT_GBRP16LE; break;
                default:
                    AddMessage(RGY_LOG_ERROR, _T("Unsupported output bitdepth %d (%s) for raw output.\n"), RGY_CSP_BIT_DEPTH[videoOutputInfo->csp], RGY_CSP_NAMES[videoOutputInfo->csp]);
                    return RGY_ERR_UNSUPPORTED;
                }
                break;
#endif
            default:
#if ENCODER_NVENC
                AddMessage(RGY_LOG_ERROR, _T("raw output supported for yuv420/yuv444 only.\n"));
#else
                AddMessage(RGY_LOG_ERROR, _T("raw output supported for yuv420 only.\n"));
#endif
                return RGY_ERR_UNSUPPORTED;
        }
        m_Mux.video.rawVideoCodecCtx->time_base = av_inv_q(m_Mux.video.outputFps);
        m_Mux.video.rawVideoCodecCtx->framerate = m_Mux.video.outputFps;
        auto codec_tag = avcodec_pix_fmt_to_codec_tag(m_Mux.video.rawVideoCodecCtx->pix_fmt);
        m_Mux.video.rawVideoCodecCtx->codec_tag = codec_tag;
        AddMessage(RGY_LOG_DEBUG, _T("Set raw video codec tag: %c%c%c%c for pixel format %s\n"),
            (codec_tag >> 0) & 0xFF, (codec_tag >> 8) & 0xFF, (codec_tag >> 16) & 0xFF, (codec_tag >> 24) & 0xFF,
            char_to_tstring(av_get_pix_fmt_name(m_Mux.video.rawVideoCodecCtx->pix_fmt)).c_str());

        m_Mux.video.simdCsp = prm->simdCsp;
        m_Mux.video.rawVideoConvert = std::make_unique<RGYConvertCSP>(prm->threadCsp, prm->threadParamCsp);
        if (m_Mux.video.rawVideoConvert->getFunc(videoOutputInfo->csp, csp_avpixfmt_to_rgy(m_Mux.video.rawVideoCodecCtx->pix_fmt), m_Mux.video.simdCsp) == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create raw video convert %s -> %s.\n"),
            RGY_CSP_NAMES[videoOutputInfo->csp], RGY_CSP_NAMES[csp_avpixfmt_to_rgy(m_Mux.video.rawVideoCodecCtx->pix_fmt)]);
            return RGY_ERR_NULL_PTR;
        }
        
        // raw video encoderを開く
        if (avcodec_open2(m_Mux.video.rawVideoCodecCtx.get(), m_Mux.video.rawVideoCodec, nullptr) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to open raw video encoder.\n"));
            return RGY_ERR_NULL_PTR;
        }
        
        AddMessage(RGY_LOG_DEBUG, _T("Raw video encoder initialized successfully.\n"));
    }

    m_Mux.video.codecCtx->codec_type              = AVMEDIA_TYPE_VIDEO;
    m_Mux.video.codecCtx->codec_id                = m_Mux.format.formatCtx->video_codec_id;
    m_Mux.video.codecCtx->width                   = videoOutputInfo->dstWidth;
    m_Mux.video.codecCtx->height                  = videoOutputInfo->dstHeight;
    m_Mux.video.codecCtx->pix_fmt                 = (m_Mux.video.rawVideoCodecCtx) ? m_Mux.video.rawVideoCodecCtx->pix_fmt : csp_rgy_to_avpixfmt(videoOutputInfo->csp);
    m_Mux.video.codecCtx->level                   = videoOutputInfo->codecLevel;
    m_Mux.video.codecCtx->profile                 = videoOutputInfo->codecProfile;
    m_Mux.video.codecCtx->sample_aspect_ratio.num = videoOutputInfo->sar[0];
    m_Mux.video.codecCtx->sample_aspect_ratio.den = videoOutputInfo->sar[1];
    m_Mux.video.codecCtx->chroma_sample_location  = (AVChromaLocation)clamp(videoOutputInfo->vui.chromaloc, 0, 6);
    m_Mux.video.codecCtx->field_order             = picstrcut_rgy_to_avfieldorder(videoOutputInfo->picstruct);
    m_Mux.video.codecCtx->delay                   = (m_VideoOutputInfo.codec == RGY_CODEC_AV1) ? 0 : videoOutputInfo->videoDelay;
    if (prm->videoCodecTag.length() > 0) {
        m_Mux.video.codecCtx->codec_tag           = tagFromStr(prm->videoCodecTag);
        AddMessage(RGY_LOG_DEBUG, _T("Set Video Codec Tag: %s\n"), char_to_tstring(tagToStr(m_Mux.video.codecCtx->codec_tag)).c_str());
    } else if (videoOutputInfo->codec == RGY_CODEC_HEVC) {
        // 特に指定の場合、HEVCでは再生互換性改善のため、 "hvc1"をデフォルトとする (libavformatのデフォルトは"hev1")
        // ただし、parallelEncodeが有効な場合は、"hve1"を使用する
        m_Mux.video.codecCtx->codec_tag = (prm->parallelEncode) ? tagFromStr("hev1") : tagFromStr("hvc1");
    } else if (videoOutputInfo->codec == RGY_CODEC_H264) {
        m_Mux.video.codecCtx->codec_tag = (prm->parallelEncode) ? tagFromStr("avc3") : tagFromStr("avc1");
    }
    if (videoOutputInfo->vui.descriptpresent
        //atcSeiを設定する場合は、コンテナ側にはVUI情報をもたせないようにする
        //コンテナ側にはatcの情報をもたせられないので、勝ちあってしまう
        && prm->hdrMetadataIn->getprm().atcSei == RGY_TRANSFER_UNKNOWN) {
        m_Mux.video.codecCtx->colorspace          = (AVColorSpace)videoOutputInfo->vui.matrix;
        m_Mux.video.codecCtx->color_primaries     = (AVColorPrimaries)videoOutputInfo->vui.colorprim;
        m_Mux.video.codecCtx->color_range         = (AVColorRange)videoOutputInfo->vui.colorrange;
        m_Mux.video.codecCtx->color_trc           = (AVColorTransferCharacteristic)videoOutputInfo->vui.transfer;
        AddMessage(RGY_LOG_DEBUG, _T("Set VUI Params: %s\n"), videoOutputInfo->vui.print_all().c_str());
    }
    if (m_Mux.format.formatCtx->flags & AVFMT_GLOBALHEADER) {
        m_Mux.video.codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (0 > avcodec_open2(m_Mux.video.codecCtx, m_Mux.video.codec, NULL)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s for video.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("opened video avcodec\n"));

    if (NULL == (m_Mux.video.streamOut = avformat_new_stream(m_Mux.format.formatCtx, m_Mux.video.codec))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for video.\n"));
        return RGY_ERR_NULL_PTR;
    }
    if (avcodec_parameters_from_context(m_Mux.video.streamOut->codecpar, m_Mux.video.codecCtx) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s for video.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_UNKNOWN;
    }

    m_Mux.video.bitstreamTimebase    = (av_isvalid_q(prm->bitstreamTimebase)) ? prm->bitstreamTimebase : HW_NATIVE_TIMEBASE;
    m_Mux.video.streamOut->time_base = (av_isvalid_q(prm->bitstreamTimebase)) ? prm->bitstreamTimebase : av_inv_q(m_Mux.video.outputFps);
    if (m_Mux.format.isMatroska) {
        m_Mux.video.streamOut->time_base = av_make_q(1, 1000);
    } else if (format_is_ivf(m_Mux.format.formatCtx) || format_is_y4m(m_Mux.format.formatCtx)) { // ivf, y4m形式の時は、time_baseをfpsの逆数にしないといけない
        m_Mux.video.streamOut->time_base = av_make_q(videoOutputInfo->fpsD, videoOutputInfo->fpsN);
    }
    m_Mux.video.streamOut->sample_aspect_ratio.num = videoOutputInfo->sar[0]; //mkvではこちらの指定も必要
    m_Mux.video.streamOut->sample_aspect_ratio.den = videoOutputInfo->sar[1];
    if (format_is_mp4(m_Mux.format.formatCtx) && videoOutputInfo->sar[0] * videoOutputInfo->sar[1] <= 0) {
        // mp4 muxerではsample_aspect_ratioが設定されていない(例えば0:0)だと、L-SMASHの"tkhd: Track Header Box" (mp4boxでは"Visual Track layout")のwidthは0になってしまう
        m_Mux.video.streamOut->sample_aspect_ratio.num = 1;
        m_Mux.video.streamOut->sample_aspect_ratio.den = 1;
    }
    m_Mux.video.streamOut->codecpar->sample_aspect_ratio = m_Mux.video.streamOut->sample_aspect_ratio;
    m_Mux.video.streamOut->avg_frame_rate.num = videoOutputInfo->fpsN; //mkvのTRACKDEFAULTDURATIONの出力に必要
    m_Mux.video.streamOut->avg_frame_rate.den = videoOutputInfo->fpsD;
    m_Mux.video.streamOut->start_time          = 0;
    if (m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_RAWVIDEO) {
        m_Mux.video.streamOut->codecpar->codec_tag = m_Mux.video.rawVideoCodecCtx->codec_tag;
    }
    m_Mux.video.dtsUnavailable    = prm->bVideoDtsUnavailable;
    m_Mux.video.inputFirstKeyPts  = prm->videoInputFirstKeyPts;
    m_Mux.video.timestamp         = prm->vidTimestamp;
    m_Mux.video.prevEncodeFrameId = -1;
    m_Mux.video.prevInputFrameId  = -1;
    m_Mux.video.pktOut            = av_packet_alloc();
    m_Mux.video.pktParse          = av_packet_alloc();
    m_Mux.video.afs               = prm->afs;
    m_Mux.video.debugDirectAV1Out = prm->debugDirectAV1Out;
    m_Mux.video.hdr10plus         = prm->hdr10plus;
    m_Mux.video.hdr10plusMetadataCopy = prm->hdr10plusMetadataCopy;
    m_Mux.video.doviRpu           = prm->doviRpu;
    m_Mux.video.doviRpuMetadataCopy = prm->doviRpuMetadataCopy;
    m_Mux.video.doviRpuConvertParam = prm->doviRpuConvertParam;

    auto retm = SetMetadata(&m_Mux.video.streamOut->metadata, (prm->videoInputStream) ? prm->videoInputStream->metadata : nullptr, prm->videoMetadata, RGY_METADATA_DEFAULT_COPY_LANG_ONLY, _T("Video"));
    if (retm != RGY_ERR_NONE) {
        return retm;
    }
    if (prm->videoInputStream) {
        m_Mux.video.inputStreamTimebase = prm->videoInputStream->time_base;
        m_Mux.video.streamOut->disposition = prm->videoInputStream->disposition;
        size_t side_data_size = 0;
        auto side_data_rotation = AVStreamGetSideData<int>(prm->videoInputStream, AV_PKT_DATA_DISPLAYMATRIX, side_data_size);
        if (side_data_rotation) {
            auto rotation = av_display_rotation_get(side_data_rotation.get());
            int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_DISPLAYMATRIX, side_data_rotation, side_data_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rotation %d from input\n"), rotation);
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("copied rotation %d from input\n"), rotation);
            side_data_rotation.reset();
        }
        if (videoOutputInfo->codec == RGY_CODEC_AV1 && prm->hdrMetadataIn != nullptr) {
            side_data_size = 0;
            auto side_data_coll = AVStreamGetSideData<AVContentLightMetadata>(prm->videoInputStream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, side_data_size);
            if (side_data_coll) {
                AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"), side_data_coll->MaxCLL, side_data_coll->MaxFALL);
                int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, side_data_coll, side_data_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_CONTENT_LIGHT_LEVEL from input\n"));
                side_data_coll.reset();
            }
            side_data_size = 0;
            auto side_data_mastering = AVStreamGetSideData<AVMasteringDisplayMetadata>(prm->videoInputStream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, side_data_size);
            if (side_data_mastering && side_data_size == sizeof(AVMasteringDisplayMetadata)) {
                AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
                    av_q2d(side_data_mastering->display_primaries[0][0]),
                    av_q2d(side_data_mastering->display_primaries[0][1]),
                    av_q2d(side_data_mastering->display_primaries[1][0]),
                    av_q2d(side_data_mastering->display_primaries[1][1]),
                    av_q2d(side_data_mastering->display_primaries[2][0]),
                    av_q2d(side_data_mastering->display_primaries[2][1]),
                    av_q2d(side_data_mastering->white_point[0]), av_q2d(side_data_mastering->white_point[1]),
                    av_q2d(side_data_mastering->min_luminance), av_q2d(side_data_mastering->max_luminance));

                int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, side_data_mastering, side_data_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_MASTERING_DISPLAY_METADATA from input\n"));
                side_data_mastering.reset();
            }
        }
    }
    m_Mux.video.doviProfileDst = prm->doviProfile;
    m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_UNSET;
    if (prm->doviProfile != RGY_DOVI_PROFILE_UNSET) {
#if LIBAVUTIL_DOVI_META_AVAIL
        std::unique_ptr<AVDOVIDecoderConfigurationRecord, RGYAVDeleter<AVDOVIDecoderConfigurationRecord>> doviconf;
        if (prm->videoInputStream) {
            // まず入力ファイルのdovi profileを読み取る
            size_t side_data_size = 0;
            doviconf = AVStreamGetSideData<AVDOVIDecoderConfigurationRecord>(prm->videoInputStream, AV_PKT_DATA_DOVI_CONF, side_data_size);
            if (doviconf) {
                m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_OTHER;
                if (videoOutputInfo->codec == RGY_CODEC_AV1 && doviconf->dv_profile != 10) {
                    // 出力がAV1の場合は、profile=10に読み替える
                    doviconf->dv_profile = 10;
                    if (   doviconf->dv_bl_signal_compatibility_id != 1
                        && doviconf->dv_bl_signal_compatibility_id != 2
                        && doviconf->dv_bl_signal_compatibility_id != 4) {
                        doviconf->dv_bl_signal_compatibility_id = 0;
                    }
                } else if (videoOutputInfo->codec == RGY_CODEC_HEVC && doviconf->dv_profile == 10) {
                    // 出力がHEVCの場合は、profile=10の場合はprofile=8に読み替える
                    if (doviconf->dv_bl_signal_compatibility_id == 1
                     || doviconf->dv_bl_signal_compatibility_id == 2
                     || doviconf->dv_bl_signal_compatibility_id == 4) {
                        doviconf->dv_profile = 8;
                    } else {
                        doviconf->dv_profile = 5;
                        doviconf->dv_bl_signal_compatibility_id = 0;
                    }
                }
                if (doviconf->dv_profile == 5) {
                    m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_50;
                } else if (doviconf->dv_profile == 7) {
                    m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_70;
                    doviconf->dv_profile = 8;
                    doviconf->dv_bl_signal_compatibility_id = 1;
                    m_Mux.video.doviProfileDst = RGY_DOVI_PROFILE_81;
                } else if (doviconf->dv_profile == 8) {
                    if (doviconf->dv_bl_signal_compatibility_id == 1) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_81;
                    } else if (doviconf->dv_bl_signal_compatibility_id == 2) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_82;
                    } else if (doviconf->dv_bl_signal_compatibility_id == 4) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_84;
                    }
                } else if (doviconf->dv_profile == 10) {
                    if (doviconf->dv_bl_signal_compatibility_id == 1) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_101;
                    } else if (doviconf->dv_bl_signal_compatibility_id == 2) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_102;
                    } else if (doviconf->dv_bl_signal_compatibility_id == 4) {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_104;
                    } else {
                        m_Mux.video.doviProfileSrc = RGY_DOVI_PROFILE_100;
                    }
                }
            }
        }
        if (prm->doviProfile != RGY_DOVI_PROFILE_COPY) {
            // コピーでない場合はコンテナに設定するdovi profileを新規作成あるいは上書きする
            size_t conf_size = 0;
            doviconf = std::unique_ptr<AVDOVIDecoderConfigurationRecord, RGYAVDeleter<AVDOVIDecoderConfigurationRecord>>(av_dovi_alloc(&conf_size), RGYAVDeleter<AVDOVIDecoderConfigurationRecord>(av_freep));
            doviconf->dv_version_major = 1;
            doviconf->dv_version_minor = 0;
            doviconf->dv_level = 10;
            doviconf->bl_present_flag = 1;
            switch (prm->doviProfile) {
            case RGY_DOVI_PROFILE_50:  doviconf->dv_profile = 5;  doviconf->dv_bl_signal_compatibility_id = 0; break;
            case RGY_DOVI_PROFILE_81:  doviconf->dv_profile = 8;  doviconf->dv_bl_signal_compatibility_id = 1; break;
            case RGY_DOVI_PROFILE_82:  doviconf->dv_profile = 8;  doviconf->dv_bl_signal_compatibility_id = 2; break;
            case RGY_DOVI_PROFILE_84:  doviconf->dv_profile = 8;  doviconf->dv_bl_signal_compatibility_id = 4; break;
            case RGY_DOVI_PROFILE_100: doviconf->dv_profile = 10; doviconf->dv_bl_signal_compatibility_id = 0; break;
            case RGY_DOVI_PROFILE_101: doviconf->dv_profile = 10; doviconf->dv_bl_signal_compatibility_id = 1; break;
            case RGY_DOVI_PROFILE_102: doviconf->dv_profile = 10; doviconf->dv_bl_signal_compatibility_id = 2; break;
            case RGY_DOVI_PROFILE_104: doviconf->dv_profile = 10; doviconf->dv_bl_signal_compatibility_id = 4; break;
            case RGY_DOVI_PROFILE_COPY: break; //ここには来ない
            default:
                AddMessage(RGY_LOG_ERROR, _T("Unsupported dolby vision profile: %d\n"), prm->doviProfile);
                return RGY_ERR_UNSUPPORTED;
            }
        }
        if (doviconf) {
            doviconf->el_present_flag = 0;
            doviconf->rpu_present_flag = prm->doviRpu || prm->doviRpuMetadataCopy ? 1 : 0;
            tstring bl_el_rpu;
            if (doviconf->bl_present_flag) bl_el_rpu += _T("+BL");
            if (doviconf->rpu_present_flag) bl_el_rpu += _T("+RPU");
            AddMessage(RGY_LOG_DEBUG, _T("dovi config ver %d.%d, %d.%d:%d %s\n"),
                doviconf->dv_version_major, doviconf->dv_version_minor,
                doviconf->dv_profile, doviconf->dv_level, doviconf->dv_bl_signal_compatibility_id,
                (bl_el_rpu.size() > 0) ? bl_el_rpu.substr(1).c_str() : _T(""));
            size_t side_data_size = 0;
            int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_DOVI_CONF, doviconf, side_data_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_DOVI_CONF\n"));
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_DOVI_CONF from input\n"));
            doviconf.reset();
        }
        if (   m_Mux.video.doviProfileSrc == m_Mux.video.doviProfileDst // 入出力が同じなら変換しない
            || (    m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_UNSET // 入力ファイルから情報が取得できなかった場合
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_81 //profile 8.x(HEVC) や 10.x(AV1) は変換しない
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_82
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_84
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_100
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_101
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_102
                 || m_Mux.video.doviProfileSrc == RGY_DOVI_PROFILE_104)) {
            m_Mux.video.doviRpuConvertParam.convertProfile = false;
        }
#else
        AddMessage(RGY_LOG_WARN, _T("dovi-profile copy noy supported in this build!\n"));
#endif //#if LIBAVUTIL_DOVI_META_AVAIL
    }

    m_Mux.video.timestampList.clear();

    if (prm->hdrMetadataIn != nullptr && prm->hdrMetadataIn->getprm().hasPrmSet()) {
        if (videoOutputInfo->codec == RGY_CODEC_HEVC) {
            auto hdrBitstream = prm->hdrMetadataIn->gen_nal();
            if (hdrBitstream.size() > 0) {
                m_Mux.video.hdrBitstream.copy(hdrBitstream.data(), (uint32_t)hdrBitstream.size());
                AddMessage(RGY_LOG_DEBUG, char_to_tstring(prm->hdrMetadataIn->print()));
            }
        } else if (videoOutputInfo->codec == RGY_CODEC_AV1) {
            auto hdrBitstream = prm->hdrMetadataIn->gen_obu();
            if (hdrBitstream.size() > 0) {
                m_Mux.video.hdrBitstream.copy(hdrBitstream.data(), (uint32_t)hdrBitstream.size());
                AddMessage(RGY_LOG_DEBUG, char_to_tstring(prm->hdrMetadataIn->print()));
            }
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Setting masterdisplay/contentlight not supported in %s encoding.\n"), CodecToStr(videoOutputInfo->codec).c_str());
            return RGY_ERR_UNSUPPORTED;
        }

        const auto HEVCHdrSeiPrm = prm->hdrMetadataIn->getprm();
        if (false && HEVCHdrSeiPrm.masterdisplay_set) {
            std::unique_ptr<AVMasteringDisplayMetadata, RGYAVDeleter<AVMasteringDisplayMetadata>> mastering(av_mastering_display_metadata_alloc(), RGYAVDeleter<AVMasteringDisplayMetadata>(av_freep));

            //streamのside dataとしてmasteringdisplay等を設定する
            mastering->display_primaries[1][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[0]); //G
            mastering->display_primaries[1][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[1]); //G
            mastering->display_primaries[2][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[2]); //B
            mastering->display_primaries[2][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[3]); //B
            mastering->display_primaries[0][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[4]); //R
            mastering->display_primaries[0][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[5]); //R
            mastering->white_point[0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[6]);
            mastering->white_point[1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[7]);
            mastering->max_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[8]);
            mastering->min_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[9]);
            mastering->has_primaries = 1;
            mastering->has_luminance = 1;

            AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f,%f) WP(%f,%f) L(%f,%f)\n"),
                av_q2d(mastering->display_primaries[0][0]),
                av_q2d(mastering->display_primaries[0][1]),
                av_q2d(mastering->display_primaries[1][0]),
                av_q2d(mastering->display_primaries[1][1]),
                av_q2d(mastering->display_primaries[2][0]),
                av_q2d(mastering->display_primaries[2][1]),
                av_q2d(mastering->white_point[0]), av_q2d(mastering->white_point[1]),
                av_q2d(mastering->max_luminance), av_q2d(mastering->min_luminance));

            int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, mastering, sizeof(mastering.get()[0]));
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
        }

        if (false && HEVCHdrSeiPrm.contentlight_set) {
            size_t coll_size = 0;
            std::unique_ptr<AVContentLightMetadata, RGYAVDeleter<AVContentLightMetadata>> coll(av_content_light_metadata_alloc(&coll_size), RGYAVDeleter<AVContentLightMetadata>(av_freep));
            coll->MaxCLL = HEVCHdrSeiPrm.maxcll;
            coll->MaxFALL = HEVCHdrSeiPrm.maxfall;
            AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"),
                coll->MaxCLL, coll->MaxFALL);
            int err = AVStreamAddSideData(m_Mux.video.streamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, coll, coll_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
        }
    }

    if (auto sts = InitVideoBsf(videoOutputInfo); sts != RGY_ERR_NONE) {
        return sts;
    }

    if (((ENCODER_VCEENC || ENCODER_MPP) && videoOutputInfo->codec != RGY_CODEC_RAW) || videoOutputInfo->codec == RGY_CODEC_AV1) {
        //parserを初期化 (frameType取得に使用、H.264/HEVCではVCEのみで必要)
        if (nullptr == (m_Mux.video.parserCtx = av_parser_init(m_Mux.format.formatCtx->video_codec_id))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init parser for %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
            return RGY_ERR_NULL_PTR;
        }
        m_Mux.video.parserCtx->flags |= PARSER_FLAG_COMPLETE_FRAMES;
        m_Mux.video.parserStreamPos = 0;
    }

    if (prm->muxVidTsLogFile.length() > 0) {
        const auto logfileName = PathRemoveExtensionS(prm->muxVidTsLogFile) + _T("_vid") + rgy_get_extension(prm->muxVidTsLogFile);
        FILE *fp = nullptr;
        if (_tfopen_s(&fp, logfileName.c_str(), _T("a"))) {
            AddMessage(RGY_LOG_WARN, _T("failed to open mux timestamp log file: \"%s\""), logfileName.c_str());
        } else {
            m_Mux.video.fpTsLogFile = std::unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened mux timestamp log file: \"%s\""), logfileName.c_str());
            tstring strFileHeadSep;
            for (int i = 0; i < 78; i++) {
                strFileHeadSep += _T("-");
            }
            _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("%s\n"), m_Mux.format.filename);
            _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("FrameType,      out pts,              out dts,                  pts,                dts, length,    size\n"));
            _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("output video stream timebase: %d/%d\n"), m_Mux.video.streamOut->time_base.num, m_Mux.video.streamOut->time_base.den);
    AddMessage(RGY_LOG_DEBUG, _T("bDtsUnavailable: %s\n"), (m_Mux.video.dtsUnavailable) ? _T("on") : _T("off"));
    return RGY_ERR_NONE;
}
#pragma warning (pop)

//音声フィルタの初期化
RGY_ERR RGYOutputAvcodec::InitAudioFilter(AVMuxAudio *muxAudio, int channels, const RGYChannelLayout *channel_layout, int sample_rate, AVSampleFormat sample_fmt, const std::string resamplerPrm) {
    //時折channel_layoutが設定されていない場合や、OrderがUnspecの場合がある
    auto channel_layout_next = (channelLayoutSet(channel_layout) && !channelLayoutOrderUnspec(channel_layout)) ? createChannelLayoutCopy(channel_layout) : getDefaultChannelLayout(channels);
    //filterを初期化
    //channelやsamplerate等の条件でfilterが必要なくとも、
    //frame_size等のずれで必要になる場合があるため、素通りするのだとしても常に有効化する
    if (!muxAudio->filterGraph  //フィルタが初期化されていない場合
        ||
        //フィルタがすでに初期化されている場合、再初期化
        (  muxAudio->filterInChannels      != channels
        || muxAudio->filterInSampleRate    != sample_rate
        || muxAudio->filterInSampleFmt     != sample_fmt
        || channelLayoutCompare(muxAudio->filterInChannelLayout.get(), channel_layout_next.get())
        )) {
        if (muxAudio->filterGraph) {
            //filterをflush
            auto filteredFrames = AudioFilterFrameFlush(muxAudio);
            WriteNextPacketAudioFrame(filteredFrames);

            //filterをclose
            avfilter_graph_free(&muxAudio->filterGraph);
        }
        muxAudio->filterInChannels      = channels;
        muxAudio->filterInChannelLayout = std::move(channel_layout_next);
        muxAudio->filterInSampleRate    = sample_rate;
        muxAudio->filterInSampleFmt     = sample_fmt;

        int ret = 0;
        muxAudio->filterGraph = avfilter_graph_alloc();
        av_opt_set_int(muxAudio->filterGraph, "threads", 1, 0);

        auto filterchain = tchar_to_string(muxAudio->filter);

        const auto select_channel_layout = (muxAudio->streamChannelSelect[muxAudio->inSubStream] == RGY_CHANNEL_AUTO) ? createChannelLayoutCopy(muxAudio->filterInChannelLayout.get()) : getChannelLayoutFromString(muxAudio->streamChannelSelect[muxAudio->inSubStream]);

        //チャンネルレイアウトの変更
        if (bSplitChannelsEnabled<MAX_SPLIT_CHANNELS>(muxAudio->streamChannelSelect)
            && channelLayoutCompare(select_channel_layout.get(), muxAudio->filterInChannelLayout.get())
            && getChannelCount(select_channel_layout.get()) < channels) {
            //初期化
            for (int inChannel = 0; inChannel < _countof(muxAudio->channelMapping); inChannel++) {
                muxAudio->channelMapping[inChannel] = -1;
            }
            const auto channelLayoutDec = getChannelLayout(muxAudio->outCodecDecodeCtx);
            //オプションによって指定されている、入力音声から抽出するべき音声レイアウト
            const int select_channel_count = getChannelCount(select_channel_layout.get());
            std::string channel_map = "pan=" + getChannelLayoutChar(getDefaultChannelLayout(select_channel_count).get());
            for (int inChannel = 0; inChannel < select_channel_count; inChannel++) {
                //オプションによって指定されているチャンネルレイアウトから、抽出する音声のチャンネルを順に取得する
                //実際には、「オプションによって指定されているチャンネルレイアウト」が入力音声に存在しない場合がある
                auto select_channel = getChannelLayoutChannelFromIndex(select_channel_layout.get(), std::min(inChannel, select_channel_count - 1));
                //対象のチャンネルのインデックスを取得する
                auto select_channel_index = getChannelLayoutIndexFromChannel(channelLayoutDec.get(), select_channel);
                if (select_channel_index < 0) {
                    //対応するチャンネルがもともとの入力音声ストリームにはない場合
                    const auto nChannels = (std::min)(inChannel, getChannelCount(channelLayoutDec.get())-1);
                    //入力音声のストリームから、抽出する音声のチャンネルを順に取得する
                    select_channel = getChannelLayoutChannelFromIndex(channelLayoutDec.get(), nChannels);
                    //対象のチャンネルのインデックスを取得する
                    select_channel_index = getChannelLayoutIndexFromChannel(channelLayoutDec.get(), select_channel);
                }
                muxAudio->channelMapping[inChannel] = select_channel_index;
                channel_map += "|c" + std::to_string(inChannel) + "=c" + std::to_string(select_channel_index);
            }
            if (filterchain.length() > 0) filterchain += ",";
            filterchain += channel_map;
            if (RGY_LOG_DEBUG >= m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                tstring channel_layout_str = strsprintf(_T("channel layout for track %d.%d:\n["), trackID(muxAudio->inTrackId), muxAudio->inSubStream);
                for (int inChannel = 0; inChannel < channels; inChannel++) {
                    channel_layout_str += strsprintf(_T("%4d"), muxAudio->channelMapping[inChannel]);
                }
                channel_layout_str += _T("]\n");
                AddMessage(RGY_LOG_DEBUG, channel_layout_str.c_str());
            }
        }

        if (filterchain.length() > 0) filterchain += ",";
        filterchain += strsprintf("aformat=sample_fmts=%s:sample_rates=%d:channel_layouts=%s",
            av_get_sample_fmt_name(muxAudio->outCodecEncodeCtx->sample_fmt),
            muxAudio->outCodecEncodeCtx->sample_rate,
            getChannelLayoutChar(muxAudio->outCodecEncodeCtx).c_str());

        AddMessage(RGY_LOG_DEBUG, _T("Parse filter description: %s\n"), char_to_tstring(filterchain).c_str());
        AVFilterInOut *filter_inputs = nullptr;
        AVFilterInOut *filter_outputs = nullptr;
        if (0 > (ret = avfilter_graph_parse2(muxAudio->filterGraph, filterchain.c_str(), &filter_inputs, &filter_outputs))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to parse filter description: %s: \"%s\": %s\n"), qsv_av_err2str(ret).c_str(), muxAudio->filter, char_to_tstring(filterchain).c_str());
            return RGY_ERR_INVALID_AUDIO_PARAM;
        }
        unique_ptr<AVFilterInOut, RGYAVDeleter<AVFilterInOut>> inputs(filter_inputs, RGYAVDeleter<AVFilterInOut>(avfilter_inout_free));
        unique_ptr<AVFilterInOut, RGYAVDeleter<AVFilterInOut>> outputs(filter_outputs, RGYAVDeleter<AVFilterInOut>(avfilter_inout_free));
        AddMessage(RGY_LOG_DEBUG, _T("Parsed filter: %s\n"), muxAudio->filter);
        const int nOutputCount = !!inputs  + (inputs  && inputs->next);
        const int nInputCount  = !!outputs + (outputs && outputs->next);
        if (nOutputCount != 1 || nInputCount != 1) {
            const TCHAR *pFilterCountStr[] = { _T("0"), _T("1"), _T(">1") };
            AddMessage(RGY_LOG_ERROR, _T("filtergraph has %s input(s) and %s output(s).\n"), pFilterCountStr[nInputCount], pFilterCountStr[nOutputCount]);
            AddMessage(RGY_LOG_ERROR, _T("only 1 in -> 1 out filtering is supported.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        //入力の設定
        const auto inargs = strsprintf("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
            1, sample_rate,
            sample_rate, av_get_sample_fmt_name(sample_fmt), getChannelLayoutChar(muxAudio->filterInChannelLayout.get()).c_str());
        const AVFilter *abuffersrc  = avfilter_get_by_name("abuffer");
        const auto inName = strsprintf("in_track_%d.%d", trackID(muxAudio->inTrackId), muxAudio->inSubStream);
        AddMessage(RGY_LOG_DEBUG, _T("create abuffer \"%s\": %s\n"), char_to_tstring(inName).c_str(), char_to_tstring(inargs).c_str());
        if (0 > (ret = avfilter_graph_create_filter(&muxAudio->filterBufferSrcCtx, abuffersrc, inName.c_str(), inargs.c_str(), nullptr, muxAudio->filterGraph))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create abuffer: %s: %s.\n"), qsv_av_err2str(ret).c_str(), char_to_tstring(inargs).c_str());
            return RGY_ERR_UNSUPPORTED;
        }
        if (0 > (ret = avfilter_link(muxAudio->filterBufferSrcCtx, 0, inputs->filter_ctx, inputs->pad_idx))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to link abuffer: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        inputs.reset();
        AddMessage(RGY_LOG_DEBUG, _T("filter linked with src buffer.\n"));

        //出力の設定
        const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
        const auto outName = strsprintf("out_track_%d.%d", trackID(muxAudio->inTrackId), muxAudio->inSubStream);
        AddMessage(RGY_LOG_DEBUG, _T("create abuffersink \"%s\"\n"), char_to_tstring(outName).c_str());
        if (0 > (ret = avfilter_graph_create_filter(&muxAudio->filterBufferSinkCtx, abuffersink, outName.c_str(), nullptr, nullptr, muxAudio->filterGraph))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNSUPPORTED;
        }
        if (0 > (ret = av_opt_set_int(muxAudio->filterBufferSinkCtx, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set channel counts to abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNSUPPORTED;
        }

        if (0 > (ret = avfilter_link(outputs->filter_ctx, outputs->pad_idx,
            muxAudio->filterBufferSinkCtx, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to link abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("filter linked with sink buffer.\n"));
        outputs.reset();

        //パラメータ設定
        std::string swr_opts = "";
        if (muxAudio->audioResampler == RGY_RESAMPLER_SOXR) {
            swr_opts = "resampler=soxr";
        }
        if (resamplerPrm.length() > 0) {
            if (swr_opts.length() > 0) swr_opts += ";";
            swr_opts += resamplerPrm;
        }
        muxAudio->filterGraph->scale_sws_opts = av_strdup(swr_opts.c_str());
        av_opt_set(muxAudio->filterGraph, "aresample_swr_opts", swr_opts.c_str(), 0);

        if (0 > (ret = avfilter_graph_config(muxAudio->filterGraph, nullptr))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to configure filter graph: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("filter config done, filter ready.\n"));

        av_buffersink_set_frame_size(muxAudio->filterBufferSinkCtx, muxAudio->outCodecEncodeCtx->frame_size);
    }
    return RGY_ERR_NONE;
}

AVBSFContext *RGYOutputAvcodec::InitStreamBsf(const tstring& bsfName, const AVStream * streamIn) {
    AddMessage(RGY_LOG_TRACE, _T("start initialize %s filter...\n"), bsfName.c_str());
    auto filter = av_bsf_get_by_name(tchar_to_string(bsfName).c_str());
    if (filter == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find %s.\n"), bsfName.c_str());
        return nullptr;
    }
    AVBSFContext *bsfc = nullptr;
    int ret = 0;
    if (0 > (ret = av_bsf_alloc(filter, &bsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsfName.c_str(), qsv_av_err2str(ret).c_str());
        return nullptr;
    }
    if (0 > (ret = avcodec_parameters_copy(bsfc->par_in, streamIn->codecpar))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsfName.c_str(), qsv_av_err2str(ret).c_str());
        return nullptr;
    }
    bsfc->time_base_in = streamIn->time_base;
    if (0 > (ret = av_bsf_init(bsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsfName.c_str(), qsv_av_err2str(ret).c_str());
        return nullptr;
    }
    return bsfc;
}

RGY_ERR RGYOutputAvcodec::InitAudio(AVMuxAudio *muxAudio, AVOutputStreamPrm *inputAudio, uint32_t audioIgnoreDecodeError, bool audioDispositionSet, const tstring& muxTsLogFileBase) {
    muxAudio->streamIn = inputAudio->src.stream;
    AddMessage(RGY_LOG_DEBUG, _T("start initializing audio ouput...\n"));
    AddMessage(RGY_LOG_DEBUG, _T("output stream index %d, trackId %d.%d\n"), inputAudio->src.index, trackID(inputAudio->src.trackId), inputAudio->src.subStreamId);
    AddMessage(RGY_LOG_DEBUG, _T("samplerate %d, stream pkt_timebase %d/%d\n"), muxAudio->streamIn->codecpar->sample_rate, muxAudio->streamIn->time_base.num, muxAudio->streamIn->time_base.den);

    if (NULL == (muxAudio->streamOut = avformat_new_stream(m_Mux.format.formatCtx, NULL))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for audio.\n"));
        return RGY_ERR_NULL_PTR;
    }
    muxAudio->decodedFrameCache = nullptr;
    muxAudio->ignoreDecodeError = audioIgnoreDecodeError;
    muxAudio->inTrackId = inputAudio->src.trackId;
    muxAudio->inSubStream = inputAudio->src.subStreamId;
    muxAudio->streamIndexIn = inputAudio->src.index;
    muxAudio->lastPtsIn = AV_NOPTS_VALUE;
    muxAudio->lastPtsOut = AV_NOPTS_VALUE;
    muxAudio->filter = inputAudio->filter.length() > 0 ? _tcsdup(inputAudio->filter.c_str()) : nullptr;
    muxAudio->streamChannelSelect = inputAudio->src.streamChannelSelect;
    muxAudio->streamChannelOut = inputAudio->src.streamChannelOut;

    if (inputAudio->bsf.length() > 0) {
        muxAudio->bsfc = InitStreamBsf(inputAudio->bsf, muxAudio->streamIn);
        if (muxAudio->bsfc == nullptr) {
            return RGY_ERR_UNKNOWN;
        }
    }

    //音声がwavの場合、フォーマット変換が必要な場合がある
    AVCodecID codecId = AV_CODEC_ID_NONE;
    if (!avcodecIsCopy(inputAudio->encodeCodec) || AV_CODEC_ID_NONE != (codecId = PCMRequiresConversion(muxAudio->streamIn->codecpar))) {
        //デコーダの作成は親ストリームのみ
        if (muxAudio->inSubStream == 0) {
            //setup decoder
            if (NULL == (muxAudio->outCodecDecode = avcodec_find_decoder(muxAudio->streamIn->codecpar->codec_id))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), inputAudio->src.stream->codecpar->codec_id));
                AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
                return RGY_ERR_INVALID_CODEC;
            }
            if (NULL == (muxAudio->outCodecDecodeCtx = avcodec_alloc_context3(muxAudio->outCodecDecode))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), inputAudio->src.stream->codecpar->codec_id));
                return RGY_ERR_NULL_PTR;
            }
            int ret;
            if (0 > (ret = avcodec_parameters_to_context(muxAudio->outCodecDecodeCtx, inputAudio->src.stream->codecpar))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s\n"),
                    char_to_tstring(avcodec_get_name(inputAudio->src.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            }
            muxAudio->outCodecDecodeCtx->pkt_timebase = inputAudio->src.stream->time_base;
            SetExtraData(muxAudio->outCodecDecodeCtx, inputAudio->src.stream->codecpar->extradata, inputAudio->src.stream->codecpar->extradata_size);
            if (nullptr != strstr(muxAudio->outCodecDecode->name, "wma")) {
                muxAudio->outCodecDecodeCtx->block_align = inputAudio->src.stream->codecpar->block_align;
            }
            //デコーダのオプションの作成
            AVDictionary *codecPrmDict = nullptr;
            unique_ptr<AVDictionary *, decltype(&av_dict_free)> codecPrmDictDeleter(&codecPrmDict, av_dict_free);
            unique_ptr<char, RGYAVDeleter<void>> prm_buf;
            if (inputAudio->decodeCodecPrm.length() > 0) {
                ret = av_dict_parse_string(&codecPrmDict, tchar_to_string(inputAudio->decodeCodecPrm).c_str(), "=", ",", 0);
                if (ret < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to parse param(s) for decoder %s for audio track %d: %s\n"),
                        char_to_tstring(muxAudio->outCodecDecode->name).c_str(), trackID(inputAudio->src.trackId), qsv_av_err2str(ret).c_str());
                    AddMessage(RGY_LOG_ERROR, _T("  prm: %s\n"), inputAudio->decodeCodecPrm.c_str());
                    return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
                }
                char *buf = nullptr;
                av_dict_get_string(codecPrmDict, &buf, '=', ',');
                prm_buf = unique_ptr<char, RGYAVDeleter<void>>(buf, RGYAVDeleter<void>(av_freep));
            }
            if (0 > (ret = avcodec_open2(muxAudio->outCodecDecodeCtx, muxAudio->outCodecDecode, &codecPrmDict))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                    char_to_tstring(avcodec_get_name(inputAudio->src.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            if (codecPrmDict) {
                for (const AVDictionaryEntry *t = nullptr; (t = av_dict_get(codecPrmDict, "", t, AV_DICT_IGNORE_SUFFIX)) != nullptr;) {
                    AddMessage(RGY_LOG_WARN, _T("Unknown option to audio decoder[%s]: %s=%s, this will be ignored.\n"),
                        char_to_tstring(muxAudio->outCodecDecode->name).c_str(),
                        char_to_tstring(t->key).c_str(),
                        char_to_tstring(t->value).c_str());
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decoder opened\n"));
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decode Info: %s, %dch[%s], %.1fkHz, %s, %d/%d %s\n"), char_to_tstring(avcodec_get_name(muxAudio->streamIn->codecpar->codec_id)).c_str(),
                getChannelCount(muxAudio->outCodecDecodeCtx), getChannelLayoutString(muxAudio->outCodecDecodeCtx).c_str(), muxAudio->outCodecDecodeCtx->sample_rate / 1000.0,
                char_to_tstring(av_get_sample_fmt_name(muxAudio->outCodecDecodeCtx->sample_fmt)).c_str(),
                muxAudio->outCodecDecodeCtx->pkt_timebase.num, muxAudio->outCodecDecodeCtx->pkt_timebase.den,
                char_to_tstring(prm_buf.get() ? prm_buf.get() : "default").c_str());
            muxAudio->decodeNextPts = AV_NOPTS_VALUE;
        }

        if (codecId != AV_CODEC_ID_NONE) {
            //PCM encoder
            if (NULL == (muxAudio->outCodecEncode = avcodec_find_encoder(codecId))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
                return RGY_ERR_INVALID_CODEC;
            }
            inputAudio->encodeCodec = RGY_AVCODEC_COPY;
        } else {
            if (avcodecIsAuto(inputAudio->encodeCodec)) {
                //エンコーダを探す (自動)
                if (NULL == (muxAudio->outCodecEncode = avcodec_find_encoder(m_Mux.format.outputFmt->audio_codec))) {
                    AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), m_Mux.format.outputFmt->audio_codec));
                    AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return RGY_ERR_INVALID_CODEC;
                }
                AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s for audio track %d\n"), char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
            } else {
                //エンコーダを探す (指定のもの)
                if (NULL == (muxAudio->outCodecEncode = avcodec_find_encoder_by_name(tchar_to_string(inputAudio->encodeCodec).c_str()))) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to find encoder for codec %s\n"), inputAudio->encodeCodec.c_str());
                    AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return RGY_ERR_INVALID_CODEC;
                }
                AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s selected for audio track %d\n"), char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
            }
            inputAudio->encodeCodec = _T("codec_something");
        }
        if (NULL == (muxAudio->outCodecEncodeCtx = avcodec_alloc_context3(muxAudio->outCodecEncode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return RGY_ERR_NULL_PTR;
        }

        auto enc_channel_layout = AutoSelectChannelLayout(muxAudio->outCodecEncode, muxAudio->outCodecDecodeCtx);
        //もしチャンネルの分離・変更があれば、それを反映してエンコーダの入力とする
        if (bSplitChannelsEnabled<MAX_SPLIT_CHANNELS>(muxAudio->streamChannelOut)) {
            enc_channel_layout = getChannelLayoutFromString(muxAudio->streamChannelOut[muxAudio->inSubStream]);
            if (muxAudio->streamChannelOut[muxAudio->inSubStream] == RGY_CHANNEL_AUTO) {
                //チャンネル選択の自動設定を反映
                uniuqeRGYChannelLayout channelSelect = (muxAudio->streamChannelSelect[muxAudio->inSubStream] == RGY_CHANNEL_AUTO) ? getChannelLayout(muxAudio->outCodecDecodeCtx) : getChannelLayoutFromString(muxAudio->streamChannelSelect[muxAudio->inSubStream]);
                if (ChannelLayoutExists(channelSelect.get(), muxAudio->outCodecEncode)) {
                    enc_channel_layout = std::move(channelSelect);
                } else {
                    enc_channel_layout = getDefaultChannelLayout(getChannelCount(channelSelect.get()));
                }
            }
        }
        int enc_sample_rate = (inputAudio->samplingRate) ? inputAudio->samplingRate : muxAudio->outCodecDecodeCtx->sample_rate;
        //select samplefmt
        muxAudio->outCodecEncodeCtx->sample_fmt          = AutoSelectSampleFmt(muxAudio->outCodecEncode->sample_fmts, muxAudio->outCodecDecodeCtx);
        muxAudio->outCodecEncodeCtx->sample_rate         = AutoSelectSamplingRate(muxAudio->outCodecEncode->supported_samplerates, enc_sample_rate);
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
        muxAudio->outCodecEncodeCtx->ch_layout           = (*enc_channel_layout.get());
#else
        muxAudio->outCodecEncodeCtx->channels            = getChannelCount(enc_channel_layout.get());
        muxAudio->outCodecEncodeCtx->channel_layout      = *enc_channel_layout;
#endif
        muxAudio->outCodecEncodeCtx->bits_per_raw_sample = muxAudio->outCodecDecodeCtx->bits_per_raw_sample;
        muxAudio->outCodecEncodeCtx->pkt_timebase        = av_make_q(1, muxAudio->outCodecDecodeCtx->sample_rate);
        if (!avcodecIsCopy(inputAudio->encodeCodec)) {
            if (inputAudio->quality.first) {
                muxAudio->outCodecEncodeCtx->flags |= AV_CODEC_FLAG_QSCALE;
                muxAudio->outCodecEncodeCtx->global_quality = inputAudio->quality.second * FF_QP2LAMBDA;
                if (inputAudio->bitrate) {
                    muxAudio->outCodecEncodeCtx->bit_rate = ((inputAudio->bitrate >= 0) ? inputAudio->bitrate : AVQSV_DEFAULT_AUDIO_BITRATE) * 1000;
                }
            } else if (inputAudio->bitrate) {
                muxAudio->outCodecEncodeCtx->bit_rate = ((inputAudio->bitrate >= 0) ? inputAudio->bitrate : AVQSV_DEFAULT_AUDIO_BITRATE) * 1000;
            }
        }
        if (m_Mux.format.outputFmt->flags & AVFMT_GLOBALHEADER) {
            muxAudio->outCodecEncodeCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        //音声プロファイルの設定
        if (inputAudio->encodeCodecProfile.length() > 0) {
            const int selected_profile = AudioGetCodecProfile(inputAudio->encodeCodecProfile, muxAudio->outCodecEncodeCtx->codec_id);
            if (selected_profile == FF_PROFILE_UNKNOWN) {
                AddMessage(RGY_LOG_ERROR, _T("unknown profile \"%s\" for codec %s (audio track %d).\n"),
                    inputAudio->encodeCodecProfile.c_str(),
                    char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
            if (muxAudio->outCodecEncode->profiles) {
                bool profileSupported = false;
                for (auto encoderProfile = muxAudio->outCodecEncode->profiles; encoderProfile->profile != FF_PROFILE_UNKNOWN; encoderProfile++) {
                    if (selected_profile == encoderProfile->profile) {
                        muxAudio->outCodecEncodeCtx->profile = selected_profile;
                        AddMessage(RGY_LOG_DEBUG, _T("profile %d (%s) selected for codec %s (audio track %d).\n"),
                            selected_profile, inputAudio->encodeCodecProfile.c_str(),
                            char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
                        profileSupported = true;
                    }
                }
                if (!profileSupported) {
                    AddMessage(RGY_LOG_WARN, _T("profile %d (%s) is not supported for codec %s (audio track %d), will be ignored.\n"),
                        char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
                }
            } else {
                AddMessage(RGY_LOG_WARN, _T("codec %s (audio track %d) does not have profile choice, profile settings to %d (%s) will be ignored and default profile will be used.\n"),
                    char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId),
                    selected_profile, inputAudio->encodeCodecProfile.c_str());
            }
        }
        //音声エンコーダのオプションの設定
        AVDictionary *codecPrmDict = nullptr;
        unique_ptr<AVDictionary*, decltype(&av_dict_free)> codecPrmDictDeleter(&codecPrmDict, av_dict_free);
        unique_ptr<char, RGYAVDeleter<void>> prm_buf;
        if (inputAudio->encodeCodecPrm.length() > 0) {
            int ret = av_dict_parse_string(&codecPrmDict, tchar_to_string(inputAudio->encodeCodecPrm).c_str(), "=", ",", 0);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to parse param(s) for codec %s for audio track %d: %s\n"),
                    char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId), qsv_av_err2str(ret).c_str());
                AddMessage(RGY_LOG_ERROR, _T("  prm: %s\n"), inputAudio->encodeCodecPrm.c_str());
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
            char *buf = nullptr;
            av_dict_get_string(codecPrmDict, &buf, '=', ',');
            prm_buf = unique_ptr<char, RGYAVDeleter<void>>(buf, RGYAVDeleter<void>(av_freep));
        }
        AddMessage(RGY_LOG_DEBUG, _T("Audio Encoder Param: %s, %dch[%s], %.1fkHz, %s, %d (%s), %d/%d, %s\n"),
            char_to_tstring(muxAudio->outCodecEncode->name).c_str(),
            getChannelCount(muxAudio->outCodecEncodeCtx), getChannelLayoutString(muxAudio->outCodecEncodeCtx).c_str(),
            muxAudio->outCodecEncodeCtx->sample_rate / 1000.0,
            char_to_tstring(av_get_sample_fmt_name(muxAudio->outCodecEncodeCtx->sample_fmt)).c_str(),
            muxAudio->outCodecEncodeCtx->profile,
            AudioGetCodecProfileStr(muxAudio->outCodecEncodeCtx->profile, muxAudio->outCodecEncodeCtx->codec_id).c_str(),
            muxAudio->outCodecEncodeCtx->pkt_timebase.num, muxAudio->outCodecEncodeCtx->pkt_timebase.den,
            char_to_tstring(prm_buf.get() ? prm_buf.get() : "default").c_str());
        if (muxAudio->outCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
            av_opt_set(muxAudio->outCodecEncodeCtx, "strict", "experimental", 0);
        }
        int ret = avcodec_open2(muxAudio->outCodecEncodeCtx, muxAudio->outCodecEncode, &codecPrmDict);
        if (ret < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open encoder(%s) for audio track %d: %s\n"),
                char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        if (codecPrmDict) {
            for (const AVDictionaryEntry *t = nullptr; (t = av_dict_get(codecPrmDict, "", t, AV_DICT_IGNORE_SUFFIX)) != nullptr;) {
                AddMessage(RGY_LOG_WARN, _T("Unknown option to audio encoder[%s]: %s=%s, this will be ignored.\n"),
                    char_to_tstring(muxAudio->outCodecEncode->name).c_str(),
                    char_to_tstring(t->key).c_str(),
                    char_to_tstring(t->value).c_str());
            }
        }
        
        muxAudio->audioResamplerPrm     = inputAudio->resamplerPrm;
        muxAudio->filterInChannels      = getChannelCount(muxAudio->outCodecEncodeCtx);
        muxAudio->filterInChannelLayout = getChannelLayout(muxAudio->outCodecEncodeCtx);
        muxAudio->filterInSampleRate    = muxAudio->outCodecEncodeCtx->sample_rate;
        muxAudio->filterInSampleFmt     = muxAudio->outCodecEncodeCtx->sample_fmt;
        const auto channelLayoutDec     = getChannelLayout(muxAudio->outCodecDecodeCtx);
        auto sts = InitAudioFilter(muxAudio,
            getChannelCount(channelLayoutDec.get()),
            channelLayoutDec.get(),
            muxAudio->outCodecDecodeCtx->sample_rate,
            // sample_fmtはデコーダのavcodec_open2時には設定されていないばあがある
            // そのときにはとりあえずAV_SAMPLE_FMT_S16で適当にフィルタを初期化しておく
            // 実際のsample_fmtはAVFrameに設定されており、フィルタ実行前の再度のInitAudioFilterで
            // 必要に応じて再初期化されるので、ここでは仮の値でも問題はない
            av_get_sample_fmt_name(muxAudio->outCodecDecodeCtx->sample_fmt) ? muxAudio->outCodecDecodeCtx->sample_fmt : AV_SAMPLE_FMT_FLTP,
            muxAudio->audioResamplerPrm);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (muxAudio->bsfc == nullptr && muxAudio->streamIn->codecpar->codec_id == AV_CODEC_ID_AAC && muxAudio->streamIn->codecpar->extradata == NULL && inputAudio->src.pktSample
        && (format_is_mp4(m_Mux.format.formatCtx) || format_is_mkv(m_Mux.format.formatCtx) || format_is_flv(m_Mux.format.formatCtx) || format_is_latm(m_Mux.format.formatCtx))) {
        muxAudio->bsfc = InitStreamBsf(_T("aac_adtstoasc"), muxAudio->streamIn);
        if (muxAudio->bsfc == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to find bsf \"%s\"\n."), _T("aac_adtstoasc"));
            return RGY_ERR_UNKNOWN;
        }
        if (inputAudio->src.pktSample) {
            //mkvではavformat_write_headerまでにAVCodecContextにextradataをセットしておく必要がある
            std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> inpkt(nullptr, RGYAVDeleter<AVPacket>(av_packet_free));
            for (inpkt.reset(av_packet_clone(inputAudio->src.pktSample)); 0 == av_bsf_send_packet(muxAudio->bsfc, inpkt.get()); inpkt.reset()) {
                std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> outpkt(av_packet_alloc(), RGYAVDeleter<AVPacket>(av_packet_free));
                int ret = av_bsf_receive_packet(muxAudio->bsfc, outpkt.get());
                if (ret == 0) {
                    if (muxAudio->bsfc->par_out->extradata) {
                        SetExtraData(muxAudio->streamIn->codecpar, muxAudio->bsfc->par_out->extradata, muxAudio->bsfc->par_out->extradata_size);
                    } else {
                        for (int i = 0; i < outpkt->side_data_elems; i++) {
                            if (outpkt->side_data[i].type == AV_PKT_DATA_NEW_EXTRADATA) {
                                SetExtraData(muxAudio->streamIn->codecpar, outpkt->side_data[i].data, (uint32_t)outpkt->side_data[i].size);
                                break;
                            }
                        }
                    }
                    break;
                }
                if (ret != AVERROR(EAGAIN) && !(inpkt && ret == AVERROR_EOF)) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to run aac_adtstoasc.\n"));
                    return RGY_ERR_UNKNOWN;
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("successfully attached packet sample from AAC\n."));
        }
    }

    //パラメータのコピー
    if (muxAudio->outCodecEncodeCtx) {
        avcodec_parameters_from_context(muxAudio->streamOut->codecpar, muxAudio->outCodecEncodeCtx);
    } else {
        auto codectag = muxAudio->streamOut->codecpar->codec_tag;
        avcodec_parameters_copy(muxAudio->streamOut->codecpar, inputAudio->src.stream->codecpar);

        //codec_tagの適合性のチェック
        auto src_codectag = inputAudio->src.stream->codecpar->codec_tag;
        if (src_codectag != 0
            && m_Mux.format.formatCtx->oformat->codec_tag) {
            uint32_t codec_tag_tmp;
            if (av_codec_get_id(m_Mux.format.formatCtx->oformat->codec_tag, src_codectag) == muxAudio->streamOut->codecpar->codec_id
                || !av_codec_get_tag2(m_Mux.format.formatCtx->oformat->codec_tag, muxAudio->streamOut->codecpar->codec_id, &codec_tag_tmp)) {
                codectag = src_codectag;
            }
        }
        muxAudio->streamOut->codecpar->codec_tag = codectag;

        avformat_transfer_internal_stream_timing_info(m_Mux.format.formatCtx->oformat, muxAudio->streamOut, inputAudio->src.stream, AVFMT_TBCF_AUTO);

        if (muxAudio->streamOut->codecpar->codec_id == AV_CODEC_ID_MP3) {
            if (   muxAudio->streamOut->codecpar->block_align == 1
                || muxAudio->streamOut->codecpar->block_align == 576
                || muxAudio->streamOut->codecpar->block_align == 1152) {
                muxAudio->streamOut->codecpar->block_align = 0;
            }
        } else if (muxAudio->streamOut->codecpar->codec_id == AV_CODEC_ID_AC3) {
            muxAudio->streamOut->codecpar->block_align = 0;
        }
        AVStreamCopySideData(muxAudio->streamOut, muxAudio->streamIn);
        if (muxAudio->streamIn->codecpar->extradata_size) {
            //aac_adtstoascから得たヘッダをコピーする
            //これをしておかないと、avformat_write_headerで"Error parsing AAC extradata, unable to determine samplerate."という
            //意味不明なエラーメッセージが表示される
            AddMessage(RGY_LOG_DEBUG, _T("set extradata from original packet...\n"));
            SetExtraData(muxAudio->streamOut->codecpar, muxAudio->streamIn->codecpar->extradata, muxAudio->streamIn->codecpar->extradata_size);
        }
    }
    muxAudio->streamOut->time_base = av_make_q(1, muxAudio->streamOut->codecpar->sample_rate);
    if (audioDispositionSet) {
        if (inputAudio->disposition.length() > 0) {
            auto disposition = parseDisposition(inputAudio->disposition);
            if (disposition == AV_DISPOSITION_COPY) {
                AddMessage(RGY_LOG_DEBUG, _T("Copy Disposition: %s\n"), getDispositionStr(inputAudio->src.stream->disposition).c_str());
                muxAudio->streamOut->disposition = inputAudio->src.stream->disposition;
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("Set Disposition: %s\n"), getDispositionStr(disposition).c_str());
                muxAudio->streamOut->disposition = disposition;
            }
        }
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("Copy Disposition: %s\n"), getDispositionStr(inputAudio->src.stream->disposition).c_str());
        muxAudio->streamOut->disposition = inputAudio->src.stream->disposition;
    }

    if (inputAudio->src.subStreamId != 0) {
        //substream(--audio-filterなどによる複製stream)の場合はデフォルトstreamではない
        muxAudio->streamOut->disposition &= (~AV_DISPOSITION_DEFAULT);
    }
    auto ret = SetMetadata(&muxAudio->streamOut->metadata, inputAudio->src.stream->metadata, inputAudio->metadata, RGY_METADATA_DEFAULT_COPY,
        strsprintf(_T("Audio #%d.%d"), trackID(muxAudio->inTrackId), muxAudio->inSubStream));
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    if (muxTsLogFileBase.length() > 0) {
        const auto logfileName = PathRemoveExtensionS(muxTsLogFileBase) + strsprintf(_T("_aud%d.%d"), trackID(inputAudio->src.trackId), inputAudio->src.subStreamId) + rgy_get_extension(muxTsLogFileBase);
        FILE *fp = nullptr;
        if (_tfopen_s(&fp, logfileName.c_str(), _T("a"))) {
            AddMessage(RGY_LOG_WARN, _T("failed to open mux timestamp log file: \"%s\""), logfileName.c_str());
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Opened mux timestamp log file: \"%s\""), logfileName.c_str());
            muxAudio->fpTsLogFile = std::unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
            tstring strFileHeadSep;
            for (int i = 0; i < 78; i++) {
                strFileHeadSep += _T("-");
            }
            _ftprintf(muxAudio->fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(muxAudio->fpTsLogFile.get(), _T("%s  - audio %d.%d\n"), m_Mux.format.filename, trackID(inputAudio->src.trackId), inputAudio->src.subStreamId);
            _ftprintf(muxAudio->fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::InitAttachment(AVMuxOther *pMuxAttach, const AttachmentSource& attachment) {
    std::ifstream ifs(attachment.filename, std::ios_base::in | std::ios_base::binary);
    if (ifs.fail()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to open file: \"%s\".\n"), attachment.filename.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened \"%s\" for attachment stream.\n"), attachment.filename.c_str());

    std::istreambuf_iterator<char> it_ifs_begin(ifs);
    std::istreambuf_iterator<char> it_ifs_end{};
    std::vector<char> input_data(it_ifs_begin, it_ifs_end);
    if (ifs.fail()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to read file: \"%s\".\n"), attachment.filename.c_str());
        return RGY_ERR_MORE_DATA;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Read %lld bytes for attachment stream.\n"), (int64_t)input_data.size());

    if (nullptr == (pMuxAttach->streamOut = avformat_new_stream(m_Mux.format.formatCtx, nullptr))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
        return RGY_ERR_NULL_PTR;
    }
    pMuxAttach->streamOut->codecpar->codec_type = AVMEDIA_TYPE_ATTACHMENT;
    pMuxAttach->streamOut->codecpar->extradata = (uint8_t *)av_malloc(input_data.size() + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(pMuxAttach->streamOut->codecpar->extradata, input_data.data(), input_data.size());
    pMuxAttach->streamOut->codecpar->extradata_size = (int)input_data.size();

    const auto attach_name_utf8 = tchar_to_string(PathGetFilename(attachment.filename), CP_UTF8);
    av_dict_set(&pMuxAttach->streamOut->metadata, "filename", attach_name_utf8.c_str(), AV_DICT_DONT_OVERWRITE);
    
    if (attachment.select.size() > 1) {
        AddMessage(RGY_LOG_ERROR, _T("Multiple setting for attachment file is unsupported: \"%s\".\n"), attachment.filename.c_str());
        return RGY_ERR_UNSUPPORTED;
    } else if (attachment.select.size() == 1) {
        //ユーザー指定のパラメータの指定
        SetMetadata(&pMuxAttach->streamOut->metadata, nullptr, attachment.select.begin()->second.metadata, RGY_METADATA_DEFAULT_CLEAR, _T("Attachment"));
    }
    AddMessage(RGY_LOG_DEBUG, _T("Add attachment stream: \"%s\".\n"), attachment.filename.c_str());
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::InitOther(AVMuxOther *muxSub, AVOutputStreamPrm *inputStream, bool streamDispositionSet) {
    const auto mediaType = (inputStream->asdata) ? AVMEDIA_TYPE_UNKNOWN : trackMediaType(inputStream->src.trackId);
    const auto mediaTypeStr = char_to_tstring(av_get_media_type_string(mediaType));
    AddMessage(RGY_LOG_DEBUG, _T("start initializing %s ouput...\n"), mediaTypeStr.c_str());

    AVCodecID codecId = inputStream->src.stream->codecpar->codec_id;

    if (avcodecIsCopy(inputStream->encodeCodec)
        && inputStream->bsf.length() == 0
        && codecId == AV_CODEC_ID_HDMV_PGS_SUBTITLE) {
        inputStream->bsf = _T("pgs_frame_merge"); //これがないと正しくmuxできない
        if (auto filter = av_bsf_get_by_name(tchar_to_string(inputStream->bsf).c_str()); filter != nullptr) {
            AddMessage(RGY_LOG_DEBUG, _T("Auto insert %s bsf filter for %s\n"), inputStream->bsf.c_str(), char_to_tstring(avcodec_get_name(codecId)).c_str());
        } else {
            inputStream->bsf.clear();
            AddMessage(RGY_LOG_DEBUG, _T("Failed to find %s bsf filter for %s, skipping...\n"), inputStream->bsf.c_str(), char_to_tstring(avcodec_get_name(codecId)).c_str());
        }
    }
    if (inputStream->bsf.length() > 0) {
        muxSub->bsfc = InitStreamBsf(inputStream->bsf, inputStream->src.stream);
        if (muxSub->bsfc == nullptr) {
            return RGY_ERR_UNKNOWN;
        }
    }

    if (mediaType == AVMEDIA_TYPE_UNKNOWN) {
        codecId = AV_CODEC_ID_NONE;
    } else if (!avcodecIsCopy(inputStream->encodeCodec)) {
        auto codec = avcodec_find_decoder_by_name(tchar_to_string(inputStream->encodeCodec).c_str());
        if (codec == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find decoder for %s.\n"), inputStream->encodeCodec.c_str());
            return RGY_ERR_INVALID_CODEC;
        }
        codecId = codec->id;
    } else if (format_is_mp4(m_Mux.format.formatCtx)) {
        if (avcodec_descriptor_get(codecId)->props & AV_CODEC_PROP_TEXT_SUB) {
            //mp4はmov_text形式しか使用できない
            codecId = AV_CODEC_ID_MOV_TEXT;
            if (inputStream->src.stream == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("--caption2ass is not supported when output format is mp4.\n"));
                return RGY_ERR_INVALID_FORMAT;
            }
            //if (inputStream->src.stream == nullptr && inputStream->src.caption2ass != FORMAT_SRT) {
            //    AddMessage(RGY_LOG_ERROR, _T("When output format is mp4, please select \"srt\" for caption2ass format.\n"));
            //    return RGY_ERR_INVALID_FORMAT;
            //}
        }
    } else if (codecId == AV_CODEC_ID_MOV_TEXT) {
        codecId = AV_CODEC_ID_ASS;
    }

    auto copy_subtitle_header = [](AVCodecContext *dstCtx, const uint8_t *header, const size_t header_size) {
        if (header) {
            dstCtx->subtitle_header_size = (decltype(dstCtx->subtitle_header_size))header_size;
            dstCtx->subtitle_header = (uint8_t *)av_mallocz(header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(dstCtx->subtitle_header, header, header_size);
        }
    };

    auto srcCodecParam = unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>>(
        avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

    avcodec_parameters_copy(srcCodecParam.get(), inputStream->src.stream->codecpar);

    if (nullptr == (muxSub->streamOut = avformat_new_stream(m_Mux.format.formatCtx, avcodec_find_decoder(codecId)))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("output stream index %d, pkt_timebase %d/%d, trackId %d\n"),
        inputStream->src.index, inputStream->src.stream->time_base.num, inputStream->src.stream->time_base.den, trackID(inputStream->src.trackId));
    if (inputStream->asdata) {
        srcCodecParam->codec_type = AVMEDIA_TYPE_UNKNOWN;
    } else if (mediaType == AVMEDIA_TYPE_DATA) {
        //なにもしない
    } else if (mediaType == AVMEDIA_TYPE_ATTACHMENT) {
        //srcCodecParam->codec_type = AVMEDIA_TYPE_ATTACHMENT;
        if (inputStream->src.stream) {
            av_packet_ref(&muxSub->streamOut->attached_pic, &inputStream->src.stream->attached_pic);
        }
    } else if (srcCodecParam->codec_id != codecId || codecId == AV_CODEC_ID_MOV_TEXT) {
        //setup decoder
        if (nullptr == (muxSub->outCodecDecode = avcodec_find_decoder(srcCodecParam->codec_id))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), srcCodecParam->codec_id));
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
            return RGY_ERR_INVALID_CODEC;
        }
        if (nullptr == (muxSub->outCodecDecodeCtx = avcodec_alloc_context3(muxSub->outCodecDecode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), srcCodecParam->codec_id));
            return RGY_ERR_NULL_PTR;
        }
        //設定されていない必須情報があれば設定する
        muxSub->outCodecDecodeCtx->pkt_timebase = inputStream->src.timebase;
        SetExtraData(muxSub->outCodecDecodeCtx, srcCodecParam->extradata, srcCodecParam->extradata_size);

        //デコーダのオプションの作成
        {
            AVDictionary *codecPrmDict = nullptr;
            unique_ptr<AVDictionary *, decltype(&av_dict_free)> codecPrmDictDeleter(&codecPrmDict, av_dict_free);
            unique_ptr<char, RGYAVDeleter<void>> prm_buf;
            if (inputStream->decodeCodecPrm.length() > 0) {
                int ret = av_dict_parse_string(&codecPrmDict, tchar_to_string(inputStream->decodeCodecPrm).c_str(), "=", ",", 0);
                if (ret < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to parse param(s) for decoder %s for subtitle track %d: %s\n"),
                        char_to_tstring(muxSub->outCodecDecode->name).c_str(), trackID(inputStream->src.trackId), qsv_av_err2str(ret).c_str());
                    AddMessage(RGY_LOG_ERROR, _T("  prm: %s\n"), inputStream->decodeCodecPrm.c_str());
                    return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
                }
                char *buf = nullptr;
                av_dict_get_string(codecPrmDict, &buf, '=', ',');
                prm_buf = unique_ptr<char, RGYAVDeleter<void>>(buf, RGYAVDeleter<void>(av_freep));
            }
            int ret = 0;
            if (0 > (ret = avcodec_open2(muxSub->outCodecDecodeCtx, muxSub->outCodecDecode, &codecPrmDict))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                    char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            if (codecPrmDict) {
                for (const AVDictionaryEntry *t = nullptr; (t = av_dict_get(codecPrmDict, "", t, AV_DICT_IGNORE_SUFFIX)) != nullptr;) {
                    AddMessage(RGY_LOG_WARN, _T("Unknown option to subtitle decoder[%s]: %s=%s, this will be ignored.\n"),
                        char_to_tstring(muxSub->outCodecDecode->name).c_str(),
                        char_to_tstring(t->key).c_str(),
                        char_to_tstring(t->value).c_str());
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
            AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d, %s\n"), char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(),
                muxSub->outCodecDecodeCtx->width, muxSub->outCodecDecodeCtx->height,
                char_to_tstring(prm_buf.get() ? prm_buf.get() : "default").c_str());
        }

        //エンコーダを探す
        if (nullptr == (muxSub->outCodecEncode = avcodec_find_encoder(codecId))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
            return RGY_ERR_INVALID_CODEC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s for subtitle track %d\n"), char_to_tstring(muxSub->outCodecEncode->name).c_str(), trackID(inputStream->src.trackId));

        if (NULL == (muxSub->outCodecEncodeCtx = avcodec_alloc_context3(muxSub->outCodecEncode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return RGY_ERR_NULL_PTR;
        }
        muxSub->outCodecEncodeCtx->time_base = av_make_q(1, 1000);

        //subtitle_headerをここで設定しないとavcodec_open2に失敗する
        //基本的にはass形式のヘッダーを設定する
        if (inputStream->src.subtitleHeader) {
            copy_subtitle_header(muxSub->outCodecEncodeCtx, (uint8_t *)inputStream->src.subtitleHeader, inputStream->src.subtitleHeaderSize);
        } else if (muxSub->outCodecDecodeCtx->subtitle_header) {
            copy_subtitle_header(muxSub->outCodecEncodeCtx, (uint8_t *)muxSub->outCodecDecodeCtx->subtitle_header, muxSub->outCodecDecodeCtx->subtitle_header_size);
        } else if (inputStream->src.stream && inputStream->src.stream->codecpar->extradata) {
            copy_subtitle_header(muxSub->outCodecEncodeCtx, (uint8_t *)inputStream->src.stream->codecpar->extradata, inputStream->src.stream->codecpar->extradata_size);
        }

        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Encoder Param: %s, %dx%d\n"), char_to_tstring(muxSub->outCodecEncode->name).c_str(),
            muxSub->outCodecEncodeCtx->width, muxSub->outCodecEncodeCtx->height);
        if (muxSub->outCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
            //問答無用で使うのだ
            av_opt_set(muxSub->outCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (m_Mux.format.formatCtx->flags & AVFMT_GLOBALHEADER) {
            muxSub->outCodecEncodeCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        //エンコーダのオプションの設定
        {
            AVDictionary *codecPrmDict = nullptr;
            unique_ptr<AVDictionary*, decltype(&av_dict_free)> codecPrmDictDeleter(&codecPrmDict, av_dict_free);
            unique_ptr<char, RGYAVDeleter<void>> prm_buf;
            if (inputStream->encodeCodecPrm.length() > 0) {
                int ret = av_dict_parse_string(&codecPrmDict, tchar_to_string(inputStream->encodeCodecPrm).c_str(), "=", ",", 0);
                if (ret < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to parse param(s) for codec %s for subtitle track %d: %s\n"),
                        char_to_tstring(muxSub->outCodecEncode->name).c_str(), trackID(inputStream->src.trackId), qsv_av_err2str(ret).c_str());
                    AddMessage(RGY_LOG_ERROR, _T("  prm: %s\n"), inputStream->encodeCodecPrm.c_str());
                    return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
                }
                char *buf = nullptr;
                av_dict_get_string(codecPrmDict, &buf, '=', ',');
                prm_buf = unique_ptr<char, RGYAVDeleter<void>>(buf, RGYAVDeleter<void>(av_freep));
            }
            if (muxSub->outCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
                av_opt_set(muxSub->outCodecEncodeCtx, "strict", "experimental", 0);
            }
            int ret = 0;
            if (0 > (ret = avcodec_open2(muxSub->outCodecEncodeCtx, muxSub->outCodecEncode, &codecPrmDict))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
                AddMessage(RGY_LOG_ERROR, _T(" %s\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            AddMessage(RGY_LOG_DEBUG, _T("Opened Subtitle Encoder Param: %s, %s\n"), char_to_tstring(muxSub->outCodecEncode->name).c_str(),
                char_to_tstring(prm_buf.get() ? prm_buf.get() : "default").c_str());
            if (codecPrmDict) {
                for (const AVDictionaryEntry *t = nullptr; (t = av_dict_get(codecPrmDict, "", t, AV_DICT_IGNORE_SUFFIX)) != nullptr;) {
                    AddMessage(RGY_LOG_WARN, _T("Unknown option to subtitle encoder[%s]: %s=%s, this will be ignored.\n"),
                        char_to_tstring(muxSub->outCodecEncode->name).c_str(),
                        char_to_tstring(t->key).c_str(),
                        char_to_tstring(t->value).c_str());
                }
            }
            if (nullptr == (muxSub->bufConvert = (uint8_t *)av_malloc(SUB_ENC_BUF_MAX_SIZE))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate buffer memory for subtitle encoding.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    muxSub->inTrackId     = inputStream->src.trackId;
    muxSub->streamIndexIn = inputStream->src.index;
    muxSub->streamIn      = inputStream->src.stream;
    muxSub->streamInTimebase = inputStream->src.timebase;
    muxSub->lastPtsOut = AV_NOPTS_VALUE;

    if (muxSub->outCodecEncodeCtx) {
        avcodec_parameters_from_context(srcCodecParam.get(), muxSub->outCodecEncodeCtx);
    }
    if (avcodec_parameters_copy(muxSub->streamOut->codecpar, srcCodecParam.get()) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Could not copy the stream parameters.\n"));
        return RGY_ERR_UNKNOWN;
    }

    if (!muxSub->streamOut->codecpar->codec_tag) {
        uint32_t codec_tag = 0;
        if (!m_Mux.format.formatCtx->oformat->codec_tag
            || av_codec_get_id(m_Mux.format.formatCtx->oformat->codec_tag, srcCodecParam->codec_tag) == srcCodecParam->codec_id
            || !av_codec_get_tag2(m_Mux.format.formatCtx->oformat->codec_tag, srcCodecParam->codec_id, &codec_tag)) {
            muxSub->streamOut->codecpar->codec_tag = srcCodecParam->codec_tag;
        }
    }
    if (muxSub->outCodecEncodeCtx) {
        SetExtraData(muxSub->streamOut->codecpar, (uint8_t *)muxSub->outCodecEncodeCtx->subtitle_header, muxSub->outCodecEncodeCtx->subtitle_header_size);
    } else if (inputStream->src.stream && inputStream->src.stream->codecpar->extradata) {
        SetExtraData(muxSub->streamOut->codecpar, (uint8_t *)inputStream->src.stream->codecpar->extradata, inputStream->src.stream->codecpar->extradata_size);
    } else if (inputStream->src.subtitleHeader) {
        SetExtraData(muxSub->streamOut->codecpar, (uint8_t *)inputStream->src.subtitleHeader, inputStream->src.subtitleHeaderSize);
    }

    muxSub->streamOut->time_base  = (mediaType == AVMEDIA_TYPE_SUBTITLE) ? av_make_q(1, 1000) : muxSub->streamInTimebase;
    muxSub->streamOut->start_time = 0;
    if (inputStream->src.stream) {
        if (streamDispositionSet) {
            if (inputStream->disposition.length() > 0) {
                auto disposition = parseDisposition(inputStream->disposition);
                if (disposition == AV_DISPOSITION_COPY) {
                    AddMessage(RGY_LOG_DEBUG, _T("Copy Disposition: %s\n"), getDispositionStr(inputStream->src.stream->disposition).c_str());
                    muxSub->streamOut->disposition = inputStream->src.stream->disposition;
                } else {
                    AddMessage(RGY_LOG_DEBUG, _T("Set Disposition: %s\n"), getDispositionStr(disposition).c_str());
                    muxSub->streamOut->disposition = disposition;
                }
            }
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Copy Disposition: %s\n"), getDispositionStr(inputStream->src.stream->disposition).c_str());
            muxSub->streamOut->disposition = inputStream->src.stream->disposition;
        }
    }
    auto ret = SetMetadata(&muxSub->streamOut->metadata, (inputStream->src.stream) ? inputStream->src.stream->metadata : nullptr, inputStream->metadata, RGY_METADATA_DEFAULT_COPY, strsprintf(_T("Other #%d"), inputStream->src.trackId));
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::SetChapters(const vector<const AVChapter *>& chapterList, bool chapterNoTrim) {
    vector<AVChapter *> outChapters;
    for (int i = 0; i < (int)chapterList.size(); i++) {
        int64_t start = (chapterNoTrim) ? chapterList[i]->start : AdjustTimestampTrimmed(chapterList[i]->start, chapterList[i]->time_base, chapterList[i]->time_base, true);
        int64_t end   = (chapterNoTrim) ? chapterList[i]->end   : AdjustTimestampTrimmed(chapterList[i]->end,   chapterList[i]->time_base, chapterList[i]->time_base, true);
        if (start < end) {
            AVChapter *pChap = (AVChapter *)av_mallocz(sizeof(pChap[0]));
            pChap->start     = start;
            pChap->end       = end;
            pChap->id        = chapterList[i]->id;
            pChap->time_base = chapterList[i]->time_base;
            av_dict_copy(&pChap->metadata, chapterList[i]->metadata, 0);
            outChapters.push_back(pChap);
        }
    }
    if (outChapters.size() > 0) {
        m_Mux.format.formatCtx->nb_chapters = (uint32_t)outChapters.size();
        m_Mux.format.formatCtx->chapters = (AVChapter **)av_realloc_f(m_Mux.format.formatCtx->chapters, outChapters.size(), sizeof(m_Mux.format.formatCtx->chapters[0]) * outChapters.size());
        for (int i = 0; i < (int)outChapters.size(); i++) {
            m_Mux.format.formatCtx->chapters[i] = outChapters[i];

            AddMessage(RGY_LOG_DEBUG, _T("chapter #%d: id %d, %lld -> %lld (timebase %d/%d), [%s -> %s]\n"),
                i, outChapters[i]->id, (long long int)outChapters[i]->start, (long long int)outChapters[i]->end,
                outChapters[i]->time_base.num, outChapters[i]->time_base.den,
                getTimestampString(outChapters[i]->start, outChapters[i]->time_base).c_str(),
                getTimestampString(outChapters[i]->end, outChapters[i]->time_base).c_str());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::Init(const TCHAR *strFileName, const VideoInfo *videoOutputInfo, const void *option) {
    m_Mux.format.streamError = true;
    AvcodecWriterPrm *prm = (AvcodecWriterPrm *)option;

    if (!check_avcodec_dll()) {
        AddMessage(RGY_LOG_ERROR, error_mes_avcodec_dll_not_found());
        return RGY_ERR_NULL_PTR;
    }

    std::string filename;
    if (0 == tchar_to_string(strFileName, filename, CP_UTF8)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to convert output filename to utf-8 characters.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    av_log_set_level((m_printMes->getLogLevel(RGY_LOGT_LIBAV) == RGY_LOG_DEBUG) ?  AV_LOG_DEBUG : RGY_AV_LOG_LEVEL);
    av_qsv_log_set(m_printMes);

    if (prm->outputFormat.length() > 0) {
        AddMessage(RGY_LOG_DEBUG, _T("output format specified: %s\n"), prm->outputFormat.c_str());
    }
    AddMessage(RGY_LOG_DEBUG, _T("output filename: \"%s\"\n"), strFileName);
    m_Mux.format.filename = strFileName;
    if (NULL == (m_Mux.format.outputFmt = av_guess_format((prm->outputFormat.length() > 0) ? tchar_to_string(prm->outputFormat).c_str() : NULL, filename.c_str(), NULL))) {
        AddMessage(RGY_LOG_ERROR,
            _T("failed to assume format from output filename.\n")
            _T("please set proper extension for output file, or specify format using option %s.\n"), (videoOutputInfo) ? _T("--format") : _T("--audio-file <format>:<filename>"));
        if (prm->outputFormat.length() > 0) {
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-formats to check available formats.\n"));
        }
        return RGY_ERR_INVALID_FORMAT;
    }
    if (0 == strcmp(filename.c_str(), "-")) {
        m_Mux.format.isPipe = true;
        m_outputIsStdout = true;
        filename = "pipe:1";
        AddMessage(RGY_LOG_DEBUG, _T("output is set to stdout\n"));
    } else if (filename.c_str() == strstr(filename.c_str(), R"(\\.\pipe\)")) {
        m_Mux.format.isPipe = true;
    }
    int err = avformat_alloc_output_context2(&m_Mux.format.formatCtx, (RGYArgN<1U, decltype(avformat_alloc_output_context2)>::type)m_Mux.format.outputFmt, nullptr, filename.c_str());
    if (m_Mux.format.formatCtx == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate format context: %s.\n"), qsv_av_err2str(err).c_str());
        return RGY_ERR_NULL_PTR;
    }

    m_Mux.format.isMatroska = format_is_mkv(m_Mux.format.formatCtx);
    m_Mux.format.disableMp4Opt = prm->disableMp4Opt;
    m_Mux.format.lowlatency = prm->lowlatency;
    m_Mux.format.offsetVideoDtsAdvance = prm->offsetVideoDtsAdvance;
    m_Mux.format.allowOtherNegativePts = prm->allowOtherNegativePts;
    m_Mux.format.timestampPassThrough = prm->timestampPassThrough;

#if USE_CUSTOM_IO
    if (m_Mux.format.isPipe || usingAVProtocols(filename, 1) || (m_Mux.format.formatCtx->oformat->flags & (AVFMT_NEEDNUMBER | AVFMT_NOFILE))) {
#endif //#if USE_CUSTOM_IO
        if (m_Mux.format.isPipe) {
            AddMessage(RGY_LOG_DEBUG, _T("output is pipe\n"));
#if defined(_WIN32) || defined(_WIN64)
            if (_setmode(_fileno(stdout), _O_BINARY) < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to switch stdout to binary mode.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
#endif //#if defined(_WIN32) || defined(_WIN64)
            if (0 == strcmp(filename.c_str(), "-")) {
                m_outputIsStdout = true;
                filename = "pipe:1";
                AddMessage(RGY_LOG_DEBUG, _T("output is set to stdout\n"));
            } else if (m_printMes->getLogLevel(RGY_LOGT_OUT) == RGY_LOG_DEBUG) {
                //AddMessage(RGY_LOG_DEBUG, _T("file name is %sunc path.\n"), (PathIsUNC(strFileName)) ? _T("") : _T("not "));
                if (rgy_file_exists(strFileName)) {
                    AddMessage(RGY_LOG_DEBUG, _T("file already exists and will overwrite.\n"));
                }
            }
        }
        if (!(m_Mux.format.formatCtx->oformat->flags & AVFMT_NOFILE)) {
            if (0 > (err = avio_open2(&m_Mux.format.formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to avio_open2 file \"%s\": %s\n"), char_to_tstring(filename, CP_UTF8).c_str(), qsv_av_err2str(err).c_str());
                return RGY_ERR_FILE_OPEN; // Couldn't open file
            }
        }
        AddMessage(RGY_LOG_DEBUG, _T("Opened file \"%s\".\n"), char_to_tstring(filename, CP_UTF8).c_str());
#if USE_CUSTOM_IO
    } else {
        m_Mux.format.outputBufferSize = clamp(prm->bufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
        if (m_Mux.format.outputBufferSize == 0) {
            //出力バッファが0とされている場合、libavformat用の内部バッファも量を減らす
            m_Mux.format.AVOutBufferSize = 128 * 1024;
            if (videoOutputInfo) {
                m_Mux.format.AVOutBufferSize *= 4;
            }
        } else {
            m_Mux.format.AVOutBufferSize = 1024 * 1024;
            if (videoOutputInfo) {
                m_Mux.format.AVOutBufferSize *= 8;
            } else {
                //動画を出力しない(音声のみの場合)場合、バッファを減らす
                m_Mux.format.outputBufferSize /= 4;
            }
        }

        if (NULL == (m_Mux.format.AVOutBuffer = (uint8_t *)av_malloc(m_Mux.format.AVOutBufferSize))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate muxer buffer of %d MB.\n"), m_Mux.format.AVOutBufferSize / (1024 * 1024));
            return RGY_ERR_MEMORY_ALLOC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("allocated internal buffer %d MB.\n"), m_Mux.format.AVOutBufferSize / (1024 * 1024));
        CreateDirectoryRecursive(PathRemoveFileSpecFixed(strFileName).second.c_str());

        //"movflags:faststart"にするには、共有モードで開けるようにする必要がある
        m_Mux.format.fpOutput = _tfsopen(strFileName, _T("wb"), _SH_DENYWR);
        if (m_Mux.format.fpOutput == NULL) {
            errno_t error = errno;
            AddMessage(RGY_LOG_ERROR, _T("failed to open %soutput file \"%s\": %s.\n"), (videoOutputInfo) ? _T("") : _T("audio "), strFileName, _tcserror(error));
            return RGY_ERR_FILE_OPEN; // Couldn't open file
        }
        if (0 < (m_Mux.format.outputBufferSize = (uint32_t)malloc_degeneracy((void **)&m_Mux.format.outputBuffer, m_Mux.format.outputBufferSize, 1024 * 1024))) {
            setvbuf(m_Mux.format.fpOutput, m_Mux.format.outputBuffer, _IOFBF, m_Mux.format.outputBufferSize);
            AddMessage(RGY_LOG_DEBUG, _T("set external output buffer %d MB.\n"), m_Mux.format.outputBufferSize / (1024 * 1024));
        }
        if (NULL == (m_Mux.format.formatCtx->pb = avio_alloc_context(m_Mux.format.AVOutBuffer, m_Mux.format.AVOutBufferSize, 1, this, funcReadPacket, (RGYArgN<5U, decltype(avio_alloc_context)>::type)funcWritePacket, funcSeek))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return RGY_ERR_NULL_PTR;
        }
    }
#endif //#if USE_CUSTOM_IO

    m_Mux.trim = prm->trimList;
    m_Mux.poolPkt = prm->poolPkt;
    m_Mux.poolFrame = prm->poolFrame;

    if (videoOutputInfo) {
        RGY_ERR sts = InitVideo(videoOutputInfo, prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Initialized video output.\n"));
    }

    const int audioStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return trackMediaType(prm.src.trackId) == AVMEDIA_TYPE_AUDIO; });
    if (audioStreamCount) {
        const bool audioDispositionSet = 0 != count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) {
            return trackMediaType(prm.src.trackId) == AVMEDIA_TYPE_AUDIO && prm.disposition.length() > 0;
        });
        m_Mux.audio.resize(audioStreamCount);
        int iAudioIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            if (trackMediaType(prm->inputStreamList[iStream].src.trackId) == AVMEDIA_TYPE_AUDIO) {
                m_Mux.audio[iAudioIdx].audioResampler = prm->audioResampler;
                //サブストリームの場合は、デコーダ情報は親ストリームのものをコピーする
                if (prm->inputStreamList[iStream].src.subStreamId > 0) {
                    auto pAudioMuxStream = getAudioStreamData(prm->inputStreamList[iStream].src.trackId, 0);
                    if (pAudioMuxStream) {
                        //デコード情報をコピー
                        m_Mux.audio[iAudioIdx].outCodecDecode    = pAudioMuxStream->outCodecDecode;
                        m_Mux.audio[iAudioIdx].outCodecDecodeCtx = pAudioMuxStream->outCodecDecodeCtx;
                    } else {
                        AddMessage(RGY_LOG_ERROR, _T("Substream #%d found for track %d, but root stream not found.\n"),
                            prm->inputStreamList[iStream].src.subStreamId, prm->inputStreamList[iStream].src.trackId);
                        return RGY_ERR_UNDEFINED_BEHAVIOR;
                    }
                }
                RGY_ERR sts = InitAudio(&m_Mux.audio[iAudioIdx], &prm->inputStreamList[iStream], prm->audioIgnoreDecodeError, audioDispositionSet, prm->muxVidTsLogFile);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized audio output - #%d: track %d, substream %d.\n"),
                    iAudioIdx, trackID(prm->inputStreamList[iStream].src.trackId), prm->inputStreamList[iStream].src.subStreamId);
                iAudioIdx++;
            }
        }
    }
    const int otherStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) {
        const auto type = trackMediaType(prm.src.trackId);
        return type == AVMEDIA_TYPE_SUBTITLE
            || type == AVMEDIA_TYPE_DATA
            || type == AVMEDIA_TYPE_ATTACHMENT;
    });
    if (otherStreamCount) {
        m_Mux.other.resize(otherStreamCount);
        int iSubIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            const auto mediaType = trackMediaType(prm->inputStreamList[iStream].src.trackId);
            if (mediaType == AVMEDIA_TYPE_SUBTITLE || mediaType == AVMEDIA_TYPE_DATA || mediaType == AVMEDIA_TYPE_ATTACHMENT) {
                const bool streamDispositionSet = 0 != count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [mediaType](AVOutputStreamPrm prm) {
                    return trackMediaType(prm.src.trackId) == mediaType && prm.disposition.length() > 0;
                });
                RGY_ERR sts = InitOther(&m_Mux.other[iSubIdx], &prm->inputStreamList[iStream], streamDispositionSet);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized %s output - %d.\n"), char_to_tstring(av_get_media_type_string(mediaType)).c_str(), iSubIdx);
                iSubIdx++;
            }
        }
    }

    for (const auto& attach : prm->attachments) {
        AVMuxOther attachStream;
        RGY_ERR sts = InitAttachment(&attachStream, attach);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_Mux.other.push_back(attachStream);
    }

    SetChapters(prm->chapterList, prm->chapterNoTrim);

    auto ret = SetMetadata(&m_Mux.format.formatCtx->metadata, prm->inputFormatMetadata, prm->formatMetadata, RGY_METADATA_DEFAULT_COPY, _T("Container"));
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    for (const auto& muxOpt : prm->muxOpt) {
        std::string optName = tchar_to_string(muxOpt.first);
        std::string optValue = tchar_to_string(muxOpt.second);
        if (0 > (err = av_dict_set(&m_Mux.format.headerOptions, optName.c_str(), optValue.c_str(), 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set mux opt: %s = %s.\n"), muxOpt.first.c_str(), muxOpt.second.c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        AddMessage(RGY_LOG_DEBUG, _T("set mux opt: %s = %s.\n"), muxOpt.first.c_str(), muxOpt.second.c_str());
    }

    tstring mes = GetWriterMes();
    AddMessage(RGY_LOG_DEBUG, mes);
    m_strOutputInfo += mes;
    m_Mux.format.streamError = false;

    if (prm->muxVidTsLogFile.length() > 0) {
        FILE *fp = nullptr;
        if (_tfopen_s(&fp, prm->muxVidTsLogFile.c_str(), _T("a"))) {
            AddMessage(RGY_LOG_WARN, _T("failed to open mux timestamp log file: \"%s\""), prm->muxVidTsLogFile.c_str());
        } else {
            m_Mux.format.fpTsLogFile = std::unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened mux timestamp log file: \"%s\""), prm->muxVidTsLogFile.c_str());
            tstring strFileHeadSep;
            for (int i = 0; i < 78; i++) {
                strFileHeadSep += _T("-");
            }
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("%s\n"), m_Mux.format.filename);
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("Type,StreamIdx,FrameType,out pts,              out dts,                  pts,                dts, length,    size\n"));
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("%s\n"), strFileHeadSep.c_str());
        }
    }

#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.streamOutMaxDts = 0;
    m_Mux.thread.queueInfo = prm->queueInfo;
    //スレッドの使用数を設定
    if (prm->threadOutput == RGY_OUTPUT_THREAD_AUTO) {
        prm->threadOutput = 1;
    }
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (prm->threadAudio == RGY_AUDIO_THREAD_AUTO) {
        prm->threadAudio = 3;
    }
    m_Mux.thread.enableAudProcessThread = prm->threadOutput > 0 && prm->threadAudio > 0;
    m_Mux.thread.enableAudEncodeThread  = prm->threadOutput > 0 && prm->threadAudio > 1;
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    m_Mux.thread.enableOutputThread     = prm->threadOutput > 0;
    if (m_Mux.thread.enableOutputThread) {
        AddMessage(RGY_LOG_DEBUG, _T("starting output thread...\n"));
        const int audioQueueCapacity = 4096;
        m_Mux.thread.qVideobitstream.init(4096, (std::max)(256, (m_Mux.video.outputFps.den) ? m_Mux.video.outputFps.num * 4 / m_Mux.video.outputFps.den : 0));
        m_Mux.thread.qVideobitstreamFreeI.init(256);
        m_Mux.thread.qVideobitstreamFreePB.init(3840);
        m_Mux.thread.thOutput = std::make_unique<AVMuxThreadWorker>();
        m_Mux.thread.thOutput->thAbort = false;
        m_Mux.thread.thOutput->qPackets.init(16384, audioQueueCapacity * std::max(1, (int)m_Mux.audio.size())); //字幕のみコピーするときのため、最低でもある程度は確保する
        m_Mux.thread.thOutput->heEventPktAdded = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.thOutput->heEventClosing = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.thOutput->thread = std::thread(&RGYOutputAvcodec::WriteThreadFunc, this, prm->threadParamOutput);
        AddMessage(RGY_LOG_DEBUG, _T("Set output thread param: %s.\n"), prm->threadParamOutput.desc().c_str());
        if (m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_RAWVIDEO) {
            m_Mux.thread.qVideoRawFrames.init(256, 64);
            m_Mux.thread.thRawVideo = std::make_unique<AVMuxThreadWorker>();
            m_Mux.thread.thRawVideo->thAbort = false;
            m_Mux.thread.thRawVideo->qPackets.init(256, 32);
            m_Mux.thread.thRawVideo->heEventPktAdded = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.thRawVideo->heEventClosing = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.thRawVideo->thread = std::thread(&RGYOutputAvcodec::WriteThreadFuncRawVideo, this, prm->threadParamOutput);
            AddMessage(RGY_LOG_DEBUG, _T("Set output thread param: %s.\n"), prm->threadParamOutput.desc().c_str());
        }
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.enableAudProcessThread) {
            auto muxAudioPtr = std::vector<const AVMuxAudio*>{ nullptr };
            if (prm->threadAudio > 2) {
                for (auto& aud : m_Mux.audio) {
                    muxAudioPtr.push_back(&aud);
                }
            }
            const auto audioQueueMultiplizer = (prm->threadAudio > 2) ? 2 : std::max(2, (int)m_Mux.audio.size());
            for (auto mux : muxAudioPtr) {
                const auto target = (mux) ? strsprintf(_T("%d.%d"), trackID(mux->inTrackId), mux->inSubStream) : tstring(_T("default"));
                AddMessage(RGY_LOG_DEBUG, _T("starting audio process thread %s...\n"), target.c_str());
                m_Mux.thread.thAud[mux] = std::make_unique<AVMuxThreadAudio>();
                m_Mux.thread.thAud[mux]->process.thAbort = false;
                m_Mux.thread.thAud[mux]->process.qPackets.init(16384, audioQueueCapacity * audioQueueMultiplizer, 4);
                m_Mux.thread.thAud[mux]->process.heEventPktAdded = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.thAud[mux]->process.heEventClosing = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.thAud[mux]->process.thread = std::thread(&RGYOutputAvcodec::ThreadFuncAudThread, this, mux, prm->threadParamAudio);
                AddMessage(RGY_LOG_DEBUG, _T("Set audio process thread param %s: %s.\n"), target.c_str(), prm->threadParamAudio.desc().c_str());
                if (m_Mux.thread.enableAudEncodeThread) {
                    AddMessage(RGY_LOG_DEBUG, _T("starting audio encode thread %s...\n"), target.c_str());
                    m_Mux.thread.thAud[mux]->encode.thAbort = false;
                    m_Mux.thread.thAud[mux]->encode.qPackets.init(16384, audioQueueCapacity * audioQueueMultiplizer, 4);
                    m_Mux.thread.thAud[mux]->encode.heEventPktAdded = CreateEvent(NULL, TRUE, FALSE, NULL);
                    m_Mux.thread.thAud[mux]->encode.heEventClosing = CreateEvent(NULL, TRUE, FALSE, NULL);
                    m_Mux.thread.thAud[mux]->encode.thread = std::thread(&RGYOutputAvcodec::ThreadFuncAudEncodeThread, this, mux, prm->threadParamAudio);
                    AddMessage(RGY_LOG_DEBUG, _T("Set audio encode thread param %s: %s.\n"), target.c_str(), prm->threadParamAudio.desc().c_str());
                }
            }
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    }
#endif //#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.video.streamOut == nullptr) {
        RGY_ERR sts = WriteFileHeader(nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_Mux.format.fileHeaderWritten = true;
    }
    m_inited = true;
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::AddHeaderToExtraDataH264(const RGYBitstream *bitstream) {
    const RGYBitstream *bs_target = bitstream;
    RGYBitstream bitstream_copy = RGYBitstreamInit();
    if (m_bsf) {
        bitstream_copy.copy(bitstream);
        auto err = m_bsf->applyBitstreamFilter(&bitstream_copy);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to apply bitstream filter to AV1 header.\n"));
            return err;
        }
        bs_target = &bitstream_copy;
    }

    std::vector<nal_info> nal_list = m_Mux.video.parse_nal_h264(bs_target->data(), bs_target->size());
    const auto h264_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_SPS; });
    const auto h264_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_PPS; });
    const bool header_check = (nal_list.end() != h264_sps_nal) && (nal_list.end() != h264_pps_nal);
    if (header_check) {
        m_Mux.video.streamOut->codecpar->extradata_size = (int)(h264_sps_nal->size + h264_pps_nal->size);
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.streamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, h264_sps_nal->ptr, h264_sps_nal->size);
        memcpy(new_ptr + h264_sps_nal->size, h264_pps_nal->ptr, h264_pps_nal->size);
        if (m_Mux.video.streamOut->codecpar->extradata) {
            av_free(m_Mux.video.streamOut->codecpar->extradata);
        }
        m_Mux.video.streamOut->codecpar->extradata = new_ptr;
    }
    bitstream_copy.clear();
    return RGY_ERR_NONE;
}

//extradataにHEVCのヘッダーを追加する
RGY_ERR RGYOutputAvcodec::AddHeaderToExtraDataHEVC(const RGYBitstream *bitstream) {
    const RGYBitstream *bs_target = bitstream;
    RGYBitstream bitstream_copy = RGYBitstreamInit();
    if (m_bsf) {
        bitstream_copy.copy(bitstream);
        auto err = m_bsf->applyBitstreamFilter(&bitstream_copy);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to apply bitstream filter to AV1 header.\n"));
            return err;
        }
        bs_target = &bitstream_copy;
    }

    const auto nal_list = m_Mux.video.parse_nal_hevc(bs_target->data(), bs_target->size());
    const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
    const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
    const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
    const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
    if (header_check) {
        std::vector<uint8_t> hevc_header;
        for (const auto& nal : nal_list) {
            if (nal.type == NALU_HEVC_SPS || nal.type == NALU_HEVC_VPS || nal.type == NALU_HEVC_PPS) {
                vector_cat(hevc_header, nal.ptr, nal.size);
            } else if (nal.nuh_layer_id == 0 && nal.type == NALU_HEVC_PREFIX_SEI) {
                auto ptr = nal.ptr;
                int nal_header_size = 0;
                static const uint8_t nal_header[4] = { 0x00, 0x00, 0x00, 0x01 };
                if (memcmp(ptr, nal_header, 4) == 0) {
                    nal_header_size += 4;
                } else if (memcmp(ptr, nal_header + 1, 3) == 0) {
                    nal_header_size += 3;
                }
                nal_header_size += 2;
                ptr += nal_header_size;
                const auto sei_data = unnal(ptr, nal.size - nal_header_size);
                const auto sei_type = sei_data[0];
                // alpha_channel_infoもextradataに追加しておく必要がある
                if (sei_type == ALPHA_CHANNEL_INFO) {
                    if (m_enableHEVCAlphaChannelInfoSEIOverwrite) {
                        vector_cat(hevc_header, gen_hevc_alpha_channel_info_sei(m_HEVCAlphaChannelMode));
                    } else {
                        vector_cat(hevc_header, nal.ptr, nal.size);
                    }
                }
            }
        }
        m_Mux.video.streamOut->codecpar->extradata_size = (int)hevc_header.size();
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.streamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, hevc_header.data(), hevc_header.size());
        if (m_Mux.video.streamOut->codecpar->extradata) {
            av_free(m_Mux.video.streamOut->codecpar->extradata);
        }
        m_Mux.video.streamOut->codecpar->extradata = new_ptr;
    }
    bitstream_copy.clear();
    return RGY_ERR_NONE;
}

//extradataにAV1のヘッダーを追加する
RGY_ERR RGYOutputAvcodec::AddHeaderToExtraDataAV1(const RGYBitstream *bitstream) {
    const RGYBitstream *bs_target = bitstream;
    RGYBitstream bitstream_copy = RGYBitstreamInit();
    if (m_bsf) {
        bitstream_copy.copy(bitstream);
        auto err = m_bsf->applyBitstreamFilter(&bitstream_copy);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to apply bitstream filter to AV1 header.\n"));
            return err;
        }
        bs_target = &bitstream_copy;
    }

    const auto unit_list = parse_unit_av1(bs_target->data(), bs_target->size());
    auto it_seq_header = std::find_if(unit_list.begin(), unit_list.end(), [](const std::unique_ptr<unit_info>& unit) {
        return unit->type == OBU_SEQUENCE_HEADER;
        });
    if (it_seq_header != unit_list.end()) {
        const auto& seq_header = (it_seq_header->get())->unit_data;
        m_Mux.video.streamOut->codecpar->extradata_size = (int)seq_header.size();
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.streamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, seq_header.data(), m_Mux.video.streamOut->codecpar->extradata_size);
        if (m_Mux.video.streamOut->codecpar->extradata) {
            av_free(m_Mux.video.streamOut->codecpar->extradata);
        }
        m_Mux.video.streamOut->codecpar->extradata = new_ptr;
    }
    bitstream_copy.clear();
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteFileHeader(const RGYBitstream *bitstream) {
    if (m_Mux.video.streamOut && bitstream) {
        RGY_ERR sts = RGY_ERR_NONE;
        switch (m_Mux.video.streamOut->codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            sts = AddHeaderToExtraDataH264(bitstream);
            break;
        case AV_CODEC_ID_HEVC:
            sts = AddHeaderToExtraDataHEVC(bitstream);
            break;
        case AV_CODEC_ID_AV1:
            sts = AddHeaderToExtraDataAV1(bitstream);
            break;
        default:
            break;
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to parse %s header.\n"), char_to_tstring(avcodec_get_name(m_Mux.video.streamOut->codecpar->codec_id)).c_str());
            return sts;
        }
    }

    //QSVEncCでエンコーダしたことを記録してみる
    //これは直接metadetaにセットする
    sprintf_s(m_Mux.format.metadataStr, ENCODER_NAME " (%s) %s", tchar_to_string(BUILD_ARCH_STR).c_str(), VER_STR_FILEVERSION);
    av_dict_set(&m_Mux.format.formatCtx->metadata, "encoding_tool", m_Mux.format.metadataStr, 0); //mp4
    //encoderではなく、encoding_toolを使用する。mp4はcomment, titleなどは設定可能, mkvではencode_byも可能

    //mp4のmajor_brandをisonからmp42に変更
    //これはmetadataではなく、avformat_write_headerのoptionsに渡す
    //この差ははっきり言って謎
    unique_ptr<AVDictionary*, decltype(&av_dict_free)> headerOptDeleter(&m_Mux.format.headerOptions, av_dict_free);
    if (m_Mux.video.streamOut) {
        if (   0 == strcmp(m_Mux.format.formatCtx->oformat->name, "mp4")
            || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "mov")) {
            av_dict_set(&m_Mux.format.headerOptions, "brand", "mp42", 0);
            AddMessage(RGY_LOG_DEBUG, _T("set format brand \"mp42\".\n"));

            if (!m_Mux.format.disableMp4Opt) {
                //moovを先頭に
                av_dict_set(&m_Mux.format.headerOptions, "movflags", "faststart", 0);
                AddMessage(RGY_LOG_DEBUG, _T("set faststart.\n"));
            }
        }
    }
    av_dict_set(&m_Mux.format.headerOptions, "strict", "experimental", 0);
    if (m_Mux.format.offsetVideoDtsAdvance && m_VideoOutputInfo.videoDelay > 0) {
        // output_ts_offset で補正する
        // まず、output_ts_offsetの指定があるかを検索
        double orig_offset = 0.0;
        auto entry = av_dict_get(m_Mux.format.headerOptions, "output_ts_offset", nullptr, AV_DICT_MATCH_CASE | AV_DICT_IGNORE_SUFFIX);
        if (!entry || sscanf_s(entry->value, "%lf", &orig_offset) != 1) {
            orig_offset = 0.0;
        }
        const AVRational fpsTimebase = (m_Mux.video.afs) ? av_inv_q(av_mul_q(m_Mux.video.outputFps, av_make_q(4, 5))) : av_inv_q(m_Mux.video.outputFps);
        const auto new_offset = orig_offset + m_VideoOutputInfo.videoDelay * av_q2d(fpsTimebase);
        const auto new_offset_str = strsprintf("%.17lf", new_offset);
        AddMessage(RGY_LOG_DEBUG, _T("Change output_ts_offset for %lf to avoid negative dts: %.17lf -> %.17lf.\n"), -1.0 * bitstream->dts() * av_q2d(m_Mux.video.streamOut->time_base), orig_offset, new_offset);
        av_dict_set(&m_Mux.format.headerOptions, "output_ts_offset", new_offset_str.c_str(), AV_DICT_MATCH_CASE);
    }

    //なんらかの問題があると、ここでよく死ぬ
    int ret = 0;
    if (0 > (ret = avformat_write_header(m_Mux.format.formatCtx, &m_Mux.format.headerOptions))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write header for output file: %s\n"), qsv_av_err2str(ret).c_str());
        m_Mux.format.streamError = true;
        return RGY_ERR_UNKNOWN;
    }
    //不正なオプションを渡していないかチェック
    for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(m_Mux.format.headerOptions, "", t, AV_DICT_IGNORE_SUFFIX));) {
        if (strcmp(t->key, "strict") != 0) {
            AddMessage(RGY_LOG_WARN, _T("Unknown option to muxer: %s=%s, this will be ignored\n"),
                char_to_tstring(t->key).c_str(),
                char_to_tstring(t->value).c_str());
        }
    }

    av_dump_format(m_Mux.format.formatCtx, 0, tchar_to_string(m_Mux.format.filename, CP_UTF8).c_str(), 1);

    //frame_sizeを表示
    for (const auto& audio : m_Mux.audio) {
        if (audio.outCodecDecodeCtx || audio.outCodecEncodeCtx) {
            tstring audioFrameSize = strsprintf(_T("audio track #%d:"), trackID(audio.inTrackId));
            if (audio.outCodecDecodeCtx) {
                audioFrameSize += strsprintf(_T(" %s frame_size %d sample/byte"), char_to_tstring(audio.outCodecDecode->name).c_str(), audio.outCodecDecodeCtx->frame_size);
            }
            if (audio.outCodecEncodeCtx) {
                audioFrameSize += strsprintf(_T(" -> %s frame_size %d sample/byte"), char_to_tstring(audio.outCodecEncode->name).c_str(), audio.outCodecEncodeCtx->frame_size);
            }
            AddMessage(RGY_LOG_DEBUG, audioFrameSize);
        }
    }
    return RGY_ERR_NONE;
}

int64_t RGYOutputAvcodec::AdjustTimestampTrimmed(int64_t nTimeIn, AVRational timescaleIn, AVRational timescaleOut, bool lastValidFrame) {
    AVRational timescaleFps = av_inv_q(m_Mux.video.outputFps);
    const int vidFrameIdx = (int)av_rescale_q(nTimeIn, timescaleIn, timescaleFps);
    int cutFrames = 0;
    if (m_Mux.trim.size() > 0) {
        int nLastFinFrame = 0;
        for (const auto& trim : m_Mux.trim) {
            if (vidFrameIdx < trim.start) {
                if (lastValidFrame) {
                    cutFrames += (vidFrameIdx - nLastFinFrame);
                    nLastFinFrame = vidFrameIdx;
                    break;
                }
                return AV_NOPTS_VALUE;
            }
            cutFrames += trim.start - nLastFinFrame;
            if (vidFrameIdx <= trim.fin) {
                nLastFinFrame = vidFrameIdx;
                break;
            }
            nLastFinFrame = trim.fin;
        }
        cutFrames += vidFrameIdx - nLastFinFrame;
    }
    int64_t tsTimeOut = av_rescale_q(nTimeIn,   timescaleIn,  timescaleOut);
    int64_t tsTrim    = av_rescale_q(cutFrames, timescaleFps, timescaleOut);
    return tsTimeOut - tsTrim;
}

tstring RGYOutputAvcodec::GetWriterMes() {
    std::string mes = "avwriter: ";
    int i_stream = 0;
    auto add_mes = [&mes](std::string str) {
        if (mes.length() - mes.find_last_of("\n") + str.length() >= 65) {
            if (0 == str.find_first_of(',')) {
                str = str.substr(1);
                mes += ",\n";
            } else {
                mes += "\n";
            }
        }
        mes += str;
    };

    if (m_Mux.video.streamOut) {
        add_mes(avcodec_get_name(m_Mux.video.streamOut->codecpar->codec_id));
        i_stream++;
    }
    for (const auto& audioStream : m_Mux.audio) {
        if (audioStream.streamOut) {
            std::string audiostr = (i_stream) ? ", " : "";
            if (audioStream.outCodecEncodeCtx) {
                //入力情報
                audiostr += strsprintf("#%d:%s/%s",
                    trackID(audioStream.inTrackId),
                    audioStream.outCodecDecode->name,
                    getChannelLayoutChar(audioStream.outCodecDecodeCtx).c_str());
                if (!audioStream.streamChannelSelect[audioStream.inSubStream].empty()) {
                    audiostr += ":" + audioStream.streamChannelSelect[audioStream.inSubStream];
                }
                //フィルタ情報
                if (audioStream.filter && _tcslen(audioStream.filter) > 0) {
                    audiostr += ":";
                    std::string filter_str;
                    auto filters = split(tchar_to_string(audioStream.filter, CP_UTF8), ",");
                    for (auto filter : filters) {
                        size_t pos = 0;
                        if ((pos = filter.find_first_of('=')) != std::string::npos) {
                            filter = filter.substr(0, pos);
                        }
                        if (filter_str.length()) filter_str += "+";
                        filter_str += filter;
                    }
                    audiostr += filter_str;
                }
                //エンコード情報
                audiostr += strsprintf(" -> %s/%s",
                    audioStream.outCodecEncode->name,
                    getChannelLayoutChar(audioStream.outCodecEncodeCtx).c_str());
                if (audioStream.outCodecEncodeCtx->flags & AV_CODEC_FLAG_QSCALE) {
                    audiostr += strsprintf("/q=%d", audioStream.outCodecEncodeCtx->global_quality / FF_QP2LAMBDA);
                }
                if (audioStream.outCodecEncodeCtx->bit_rate) {
                    audiostr += strsprintf("/%dkbps", audioStream.outCodecEncodeCtx->bit_rate / 1000);
                }
            } else {
                audiostr += strsprintf("%s", avcodec_get_name(audioStream.streamIn->codecpar->codec_id));
            }
            add_mes(audiostr);
            i_stream++;
        }
    }
    for (const auto& otherStream : m_Mux.other) {
        if (otherStream.streamOut) {
            add_mes(std::string((i_stream) ? ", " : "") + av_get_media_type_string(trackMediaType(otherStream.inTrackId)) + strsprintf("#%d", trackID(otherStream.inTrackId)));
            i_stream++;
        }
    }
    if (m_Mux.format.formatCtx->nb_chapters > 0) {
        add_mes(std::string((i_stream) ? ", " : "") + "chap");
        i_stream++;
    }
    std::string output = " => ";
    output += m_Mux.format.formatCtx->oformat->name;
    //if (m_Mux.format.outputBufferSize) {
    //    output += strsprintf(" (%dMB buf)", m_Mux.format.outputBufferSize / (1024 * 1024));
    //}
    add_mes(output);
    return char_to_tstring(mes.c_str());
}

uint32_t RGYOutputAvcodec::getH264PAFFFieldLength(const uint8_t *ptr, uint32_t size, int *isIDR) {
    int sliceNalu = 0;
    *isIDR = 0;
    uint8_t a = ptr[0], b = ptr[1], c = ptr[2], d = 0;
    for (uint32_t i = 3; i < size; i++) {
        d = ptr[i];
        if (((a | b) == 0) & (c == 1)) {
            if (sliceNalu) {
                return i-3-(ptr[i-4]==0)+1;
            }
            int nalType = d & 0x1F;
            sliceNalu += ((nalType == 1) | (nalType == 5));
            *isIDR = nalType == 5;
        }
        a = b, b = c, c = d;
    }
    return size;
}

RGY_ERR RGYOutputAvcodec::WriteNextFrame(RGYBitstream *bitstream) {
#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.thread.thOutput) {
        RGYBitstream copyStream = RGYBitstreamInit();
        bool bFrameI = (bitstream->frametype() & RGY_FRAMETYPE_I) != 0;
        bool bFrameP = (bitstream->frametype() & RGY_FRAMETYPE_P) != 0;
        //IフレームかPBフレームかでサイズが大きく違うため、空きのmfxBistreamは異なるキューで管理する
        auto& qVideoQueueFree = (bFrameI) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
        //空いているmfxBistreamを取り出す
        if (!qVideoQueueFree.front_copy_and_pop_no_lock(&copyStream) || copyStream.bufsize() < bitstream->size()) {
            //空いているmfxBistreamがない、あるいはそのバッファサイズが小さい場合は、領域を取り直す
            const auto allocate_bytes = bitstream->size() * ((bFrameI | bFrameP) ? 2 : 8);
            if (RGY_ERR_NONE != copyStream.init(allocate_bytes)) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for video bitstream output buffer, %sB.\n"), allocate_bytes);
                m_Mux.format.streamError = true;
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        //必要な情報をコピー
        copyStream.setDataflag(bitstream->dataflag());
        copyStream.setPts(bitstream->pts());
        copyStream.setDts(bitstream->dts());
        copyStream.setDuration(bitstream->duration());
        copyStream.setFrametype(bitstream->frametype());
        copyStream.setSize(bitstream->size());
        copyStream.setAvgQP(bitstream->avgQP());
        copyStream.setOffset(0);
        memcpy(copyStream.bufptr(), bitstream->data(), copyStream.size());
        //キューに押し込む
        if (!m_Mux.thread.qVideobitstream.push(copyStream)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for video bitstream queue.\n"));
            m_Mux.format.streamError = true;
        }
        bitstream->setSize(0);
        bitstream->setOffset(0);
        SetEvent(m_Mux.thread.thOutput->heEventPktAdded);
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }
#endif
    int64_t dts = 0;
    return WriteNextFrameInternal(bitstream, &dts);
}

RGY_ERR RGYOutputAvcodec::VidCheckStreamAVParser(RGYBitstream *pBitstream) {
    if (m_Mux.video.parserCtx == nullptr) return RGY_ERR_NONE;

    RGY_ERR err = RGY_ERR_NONE;
    m_Mux.video.parserStreamPos += pBitstream->size();
    AVPacket *pkt = m_Mux.video.pktParse;
    av_new_packet(pkt, (int)pBitstream->size());
    memcpy(pkt->data, pBitstream->data(), pBitstream->size());
    pkt->size = (int)pBitstream->size();
    pkt->pts = pBitstream->pts();
    pkt->dts = pBitstream->dts();
    pkt->pos = m_Mux.video.parserStreamPos;
    uint8_t *dummy = nullptr;
    int dummy_size = 0;
    if (0 < av_parser_parse2(m_Mux.video.parserCtx, m_Mux.video.codecCtx, &dummy, &dummy_size, pkt->data, pkt->size, pkt->pts, pkt->dts, pkt->pos)) {
        //pBitstream->PictStruct = m_Mux.video.parserCtx->picture_structure;
        //pBitstream->RepeatPict = m_Mux.video.parserCtx->repeat_pict;

        const auto pict_type = m_Mux.video.parserCtx->pict_type;

        RGY_FRAMETYPE frameType = RGY_FRAMETYPE_UNKNOWN;
        frameType |= (m_Mux.video.parserCtx->key_frame) ? (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I) : RGY_FRAMETYPE_UNKNOWN;
        frameType |= (pict_type == AV_PICTURE_TYPE_I) ? (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I) : RGY_FRAMETYPE_UNKNOWN;
        frameType |= (pict_type == AV_PICTURE_TYPE_P) ? RGY_FRAMETYPE_P : RGY_FRAMETYPE_UNKNOWN;
        frameType |= (pict_type == AV_PICTURE_TYPE_B) ? RGY_FRAMETYPE_B : RGY_FRAMETYPE_UNKNOWN;
        pBitstream->setFrametype(frameType);
    } else {
        AddMessage(RGY_LOG_ERROR, _T("AVParser error parsing VCE output."));
        err = RGY_ERR_UNKNOWN;
    }
    av_packet_unref(pkt);
    return err;
}

RGY_ERR RGYOutputAvcodec::WriteNextFrameFinish(RGYBitstream *bitstream, const RGY_FRAMETYPE frameType) {
#if ENABLE_AVCODEC_OUT_THREAD
    //最初のヘッダーを書いたパケットはコピーではないので、キューに入れない
    if (m_Mux.thread.thOutput) {
        //確保したメモリ領域を使いまわすためにキューに格納
        const auto frameI = (frameType & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) != 0;
        auto& qVideoQueueFree = (frameI) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
        auto queueFavoredSize = (frameI) ? VID_BITSTREAM_QUEUE_SIZE_I : VID_BITSTREAM_QUEUE_SIZE_PB;
        if ((int64_t)qVideoQueueFree.size() > queueFavoredSize) {
            //あまり多すぎると無駄にメモリを使用するので減らす
            bitstream->clear();
        } else {
            qVideoQueueFree.push(*bitstream);
        }
    } else {
#endif
        bitstream->setSize(0);
        bitstream->setOffset(0);
#if ENABLE_AVCODEC_OUT_THREAD
    }
#endif
    m_Mux.format.fileHeaderWritten = true;
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputAvcodec::WriteNextFrameInternalOneFrame(RGYBitstream *bitstream, int64_t *writtenDts, const RGYTimestampMapVal& bs_framedata) {
    //AVParserを使用して必要に応じてframeTypeを取得する
    VidCheckStreamAVParser(bitstream);

    if (m_bsf) {
        auto sts = m_bsf->applyBitstreamFilter(bitstream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    // NVENCのalpha_channel_info SEIの出力は変なので、適切なものに置き換える
    auto err = OverwriteHEVCAlphaChannelInfoSEI(bitstream);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    bool isIDR = (bitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_xIDR)) != 0; //IDRかどうかのフラグ
    bool isKey = (bitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_xIDR | RGY_FRAMETYPE_I | RGY_FRAMETYPE_xI)) != 0; //Keyフレームかどうかのフラグ
    if (m_Mux.video.streamOut->codecpar->field_order != AV_FIELD_PROGRESSIVE) {
        if (m_VideoOutputInfo.codec == RGY_CODEC_H264) {
            const auto nal_list = m_Mux.video.parse_nal_h264(bitstream->data(), bitstream->size());
            //インタレ保持の際、IDRかどうかのフラグが正しく設定されていないことがある
            //どちらかのフィールドがIDRならIDRのフラグを立てる
            isIDR = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_IDR; }) != nal_list.end();
            isKey |= isIDR;
        } else if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
            AddMessage(RGY_LOG_ERROR, _T("Interlaced HEVC encoding not supported!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    std::vector<std::unique_ptr<RGYOutputInsertMetadata>> metadataList;
    if (m_Mux.video.hdrBitstream.size() > 0) {
        std::vector<uint8_t> data(m_Mux.video.hdrBitstream.data(), m_Mux.video.hdrBitstream.data() + m_Mux.video.hdrBitstream.size());
        metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(data, true, RGYOutputInsertMetadataPosition::Prefix));
    }
    if (m_Mux.video.hdr10plus) {
        if (auto data = m_Mux.video.hdr10plus->getData(bs_framedata.inputFrameId, m_VideoOutputInfo.codec); data.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(data, false, RGYOutputInsertMetadata::dhdr10plus_pos(m_VideoOutputInfo.codec)));
        }
    } else if (m_Mux.video.hdr10plusMetadataCopy) {
        auto [err_hdr10plus, metadata_hdr10plus] = getMetadata<RGYFrameDataHDR10plus>(RGY_FRAME_DATA_HDR10PLUS, bs_framedata, nullptr);
        if (err_hdr10plus != RGY_ERR_NONE) {
            return err_hdr10plus;
        }
        if (metadata_hdr10plus.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(metadata_hdr10plus, false, RGYOutputInsertMetadata::dhdr10plus_pos(m_VideoOutputInfo.codec)));
        }
    }
    if (m_Mux.video.doviRpu) {
        std::vector<uint8_t> dovi_nal;
        if (m_Mux.video.doviRpu->get_next_rpu(dovi_nal, m_Mux.video.doviProfileDst, &m_Mux.video.doviRpuConvertParam, bs_framedata.inputFrameId, m_VideoOutputInfo.codec) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get dovi rpu for %lld.\n"), bs_framedata.inputFrameId);
        }
        if (dovi_nal.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(dovi_nal, false, RGYOutputInsertMetadata::dovirpu_pos(m_VideoOutputInfo.codec)));
        }
    } else if (m_Mux.video.doviRpuMetadataCopy) {
        auto doviRpuConvPrm = std::make_unique<RGYFrameDataDOVIRpuConvertParam>(m_Mux.video.doviProfileDst, m_Mux.video.doviRpuConvertParam);
        auto [err_dovirpu, metadata_dovi_rpu] = getMetadata<RGYFrameDataDOVIRpu>(RGY_FRAME_DATA_DOVIRPU, bs_framedata, doviRpuConvPrm.get());
        if (err_dovirpu != RGY_ERR_NONE) {
            return err_dovirpu;
        }
        if (metadata_dovi_rpu.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(metadata_dovi_rpu, false, RGYOutputInsertMetadata::dovirpu_pos(m_VideoOutputInfo.codec)));
        }
    }

    err = InsertMetadata(bitstream, metadataList);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    AVPacket *pkt = m_Mux.video.pktOut;
    av_new_packet(pkt, (int)bitstream->size());
    memcpy(pkt->data, bitstream->data(), bitstream->size());
    pkt->size = (int)bitstream->size();

    const AVRational streamTimebase = m_Mux.video.streamOut->time_base;
    pkt->stream_index = m_Mux.video.streamOut->index;
    pkt->flags = (isKey) ? AV_PKT_FLAG_KEY : 0;
#if ENCODER_QSV
    //QSVエンコーダでは、bitstreamからdurationの情報が取得できないので、別途取得する
    pkt->duration = bs_framedata.duration;
#else
    pkt->duration = bitstream->duration();
#endif
    pkt->pts = bitstream->pts();
#if ENCODER_QSV
    //QSVエンコーダだけは、HW_NATIVE_TIMEBASEで送られてくる
    pkt->duration = av_rescale_q(pkt->duration, HW_NATIVE_TIMEBASE, m_Mux.video.bitstreamTimebase);
    pkt->pts = av_rescale_q(pkt->pts, HW_NATIVE_TIMEBASE, m_Mux.video.bitstreamTimebase);
#endif
    if (av_cmp_q(m_Mux.video.bitstreamTimebase, streamTimebase) != 0) {
        pkt->duration = av_rescale_q(pkt->duration, m_Mux.video.bitstreamTimebase, streamTimebase);
        pkt->pts = av_rescale_q(pkt->pts, m_Mux.video.bitstreamTimebase, streamTimebase);
    }
    if (false && !m_Mux.video.dtsUnavailable) {
        pkt->dts = av_rescale_q(bitstream->dts(), m_Mux.video.bitstreamTimebase, streamTimebase);
    } else {
        m_Mux.video.timestampList.add(pkt->pts);
        pkt->dts = m_Mux.video.timestampList.get_min_pts();
    }
    if (WRITE_PTS_DEBUG) {
        AddMessage(RGY_LOG_WARN, _T("video pts %3d, %12s, dts %lld, pts, %lld (%d/%d) [%s]\n"),
            pkt->stream_index, char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->streams[pkt->stream_index]->codecpar->codec_id)).c_str(),
            pkt->dts, pkt->pts, streamTimebase.num, streamTimebase.den, getTimestampString(pkt->pts, streamTimebase).c_str());
    }
    const auto pts = pkt->pts, dts = pkt->dts, duration = pkt->duration;
    *writtenDts = av_rescale_q(pkt->dts, streamTimebase, QUEUE_DTS_TIMEBASE);
    const auto ret_write = av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
    if (ret_write != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Error: Failed to write video frame: %s.\n"), qsv_av_err2str(ret_write).c_str());
        m_Mux.format.streamError = true;
    }

    //インタレ保持の際、IDRかどうかのフラグが正しく設定されていないことがある
    //どちらかのフィールドがIDRならIDRのフラグを立ててているので、それを参照する
    const auto frameType = (isIDR) ? RGY_FRAMETYPE_IDR : bitstream->frametype();
    if (m_Mux.video.fpTsLogFile) {
        const TCHAR *pFrameTypeStr =
            (frameType & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) ? _T("I") : (((frameType & RGY_FRAMETYPE_B) == 0) ? _T("P") : _T("B"));
        _ftprintf(m_Mux.video.fpTsLogFile.get(), _T("%s, %20lld, %20lld, %20lld, %20lld, %d, %7zd\n"), pFrameTypeStr, (lls)bitstream->pts(), (lls)bitstream->dts(), (lls)pts, (lls)dts, (int)duration, bitstream->size());
        {
            std::lock_guard<std::mutex> lock(m_Mux.format.fpTsLogMtx);
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("v, %d, %s, %20lld, %20lld, %20lld, %20lld, %d, %7zd\n"), pkt->stream_index, pFrameTypeStr, (lls)bitstream->pts(), (lls)bitstream->dts(), (lls)pts, (lls)dts, (int)duration, bitstream->size());
        }
    }
    m_encSatusInfo->SetOutputData(frameType, bitstream->size(), bitstream->avgQP());
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteNextFrameInternal(RGYBitstream *bitstream, int64_t *writtenDts) {
    if (!m_Mux.format.fileHeaderWritten) {
#if 0 && ENCODER_QSV //HEVCエンコードやFixed Funcでは、DecodeTimeStampが正しく設定されないので無効化
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC && bitstream->dts() == MFX_TIMESTAMP_UNKNOWN) {
            m_Mux.video.dtsUnavailable = true;
        }
#else
        m_Mux.video.dtsUnavailable = true;
#endif

        //dts生成を初期化
        //何フレーム前からにすればよいかは、b-pyramid次第で異なるので、可能な限りエンコーダの情報を使用する
        const auto srcTimebase = (ENCODER_QSV) ? HW_NATIVE_TIMEBASE : m_Mux.video.bitstreamTimebase;
#if ENCODER_QSV
        if (bitstream->dts() != MFX_TIMESTAMP_UNKNOWN) {
            m_VideoOutputInfo.videoDelay = (m_VideoOutputInfo.codec == RGY_CODEC_AV1 && AV1_TIMESTAMP_OVERRIDE) ? 0 : -1 * (int)av_rescale_q(bitstream->dts() - bitstream->pts(), srcTimebase, av_inv_q(m_Mux.video.outputFps));
        }
#endif
        RGY_ERR sts = WriteFileHeader(bitstream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        // WriteFileHeaderでstreamOut->time_baseが変わることがあるので、まずWriteFileHeaderを先に行ってから、この処理を行う
        m_Mux.video.fpsBaseNextDts = 0 - m_VideoOutputInfo.videoDelay;
        AddMessage(RGY_LOG_DEBUG, _T("calc dts, first dts %d x (timebase).\n"), m_Mux.video.fpsBaseNextDts);

        const AVRational fpsTimebase = (m_Mux.video.afs) ? av_inv_q(av_mul_q(m_Mux.video.outputFps, av_make_q(4, 5))) : av_inv_q(m_Mux.video.outputFps);
        const AVRational streamTimebase = m_Mux.video.streamOut->time_base;
        const auto firstPacketPts = av_rescale_q(bitstream->pts(), srcTimebase, streamTimebase);
        bitstream->setDts(firstPacketPts + av_rescale_q(m_Mux.video.fpsBaseNextDts, fpsTimebase, streamTimebase));
        for (int i = m_Mux.video.fpsBaseNextDts; i < 0; i++) {
            m_Mux.video.timestampList.add(firstPacketPts + av_rescale_q(i, fpsTimebase, streamTimebase));
        }
    }

    const bool flush = bitstream->size() == 0;

    if (m_VideoOutputInfo.codec != RGY_CODEC_AV1 || m_Mux.video.debugDirectAV1Out) { // AV1以外
        if (flush) {
            return RGY_ERR_NONE; // 特にflushするものはない
        }
        RGYTimestampMapVal bs_framedata;
        if (m_Mux.video.timestamp) {
            bs_framedata = m_Mux.video.timestamp->get(bitstream->pts());
            if (bs_framedata.inputFrameId < 0) {
                bs_framedata.inputFrameId = m_Mux.video.prevInputFrameId;
                AddMessage(RGY_LOG_WARN, _T("Failed to get frame ID for pts %lld, using input frame ID %lld.\n"), bitstream->pts(), bs_framedata.inputFrameId);
            }
            m_Mux.video.prevInputFrameId = bs_framedata.inputFrameId;
            m_Mux.video.prevEncodeFrameId = bs_framedata.encodeFrameId;
        }
        auto err = WriteNextFrameInternalOneFrame(bitstream, writtenDts, bs_framedata);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        return WriteNextFrameFinish(bitstream, bitstream->frametype());
    }

    // AV1の場合、SDKの返すtimestampは滅茶苦茶
    // また、TEMPORAL_DELIMITERベースの区切りになっていないため、区切りをやり直す必要がある
    if (!m_Mux.video.timestamp && AV1_TIMESTAMP_OVERRIDE) {
        AddMessage(RGY_LOG_ERROR, _T("m_Mux.video.timestamp == nullptr!\n"));
        return RGY_ERR_NULL_PTR;
    }

    // まず、AV1をユニット単位に分割し、その種類を取得する
    auto new_units = parse_unit_av1(bitstream->data(), bitstream->size());
    // バッファに連結
    m_Mux.videoAV1Merge.insert(m_Mux.videoAV1Merge.end(), std::make_move_iterator(new_units.begin()), std::make_move_iterator(new_units.end()));
    bitstream->setSize(0);
    bitstream->setOffset(0);

    for (;;) {
        // 先頭ユニットは、OBU_AV1_TEMPORAL_DELIMITERになるようになっている
        // その次のOBU_AV1_TEMPORAL_DELIMITERが見つかったら、そこまでを一単位として送出する
        size_t next_delim = 0;
        for (size_t iunit = 1; iunit < m_Mux.videoAV1Merge.size(); iunit++) {
            if (m_Mux.videoAV1Merge[iunit]->type == OBU_TEMPORAL_DELIMITER) {
                next_delim = iunit;
                break;
            }
        }
        if (next_delim == 0) { // 見つからなかった
            if (flush) { // flushする場合は最後まで
                next_delim = m_Mux.videoAV1Merge.size();
            }
            if (next_delim == 0) {
                break; // 抜けて、次のデータが来るまで待つ
            }
        }
        //次のフレームの時刻情報を取得
        RGYTimestampMapVal bs_framedata = m_Mux.video.timestamp->getByEncodeFrameID(m_Mux.video.prevEncodeFrameId + 1);
        if (bs_framedata.inputFrameId < 0) {
            bs_framedata.inputFrameId = m_Mux.video.prevInputFrameId;
            AddMessage(RGY_LOG_WARN, _T("Failed to get timestamp for id %lld, using %lld.\n"), bitstream->pts(), bs_framedata.inputFrameId);
        } else {
            m_Mux.video.prevInputFrameId = bs_framedata.inputFrameId;
            m_Mux.video.prevEncodeFrameId++;
        }

        //送出すべきデータサイズを取得
        size_t data_size = 0;
        for (size_t iunit = 0; iunit < next_delim; iunit++) {
            data_size += m_Mux.videoAV1Merge[iunit]->unit_data.size();
        }
        //bitstreamを設定
        bitstream->init(data_size);
        bitstream->setSize(data_size);
        bitstream->setPts(bs_framedata.timestamp);
        bitstream->setDts(bs_framedata.timestamp);
        bitstream->setDuration(bs_framedata.duration);

        size_t copy_size = 0; // コピーしたデータサイズ
        for (size_t iunit = 0; iunit < next_delim; iunit++) {
            const auto& unit_data = m_Mux.videoAV1Merge[iunit]->unit_data;
            memcpy(bitstream->data() + copy_size, unit_data.data(), unit_data.size());
            copy_size += unit_data.size();
        }
        // コピーし終わったユニットを破棄
        for (size_t iunit = 0; iunit < next_delim; iunit++) {
            m_Mux.videoAV1Merge.pop_front();
        }

        auto err = WriteNextFrameInternalOneFrame(bitstream, writtenDts, bs_framedata);
        if (err != RGY_ERR_NONE) {
            break;
        }
    }
    return WriteNextFrameFinish(bitstream, bitstream->frametype());
}
#pragma warning (pop)

RGY_ERR RGYOutputAvcodec::WriteNextFrame(RGYFrame *surface) {
    if (m_Mux.format.formatCtx->video_codec_id != AV_CODEC_ID_RAWVIDEO) {
        AddMessage(RGY_LOG_ERROR, _T("Unsupported codec for WriteNextFrame(RGYFrame): %s\n"), avcodec_get_name(m_Mux.format.formatCtx->video_codec_id));
        m_Mux.format.streamError = true;
        return RGY_ERR_UNSUPPORTED;
    }

    if (!m_Mux.format.fileHeaderWritten) {
        RGY_ERR sts = WriteFileHeader(nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_Mux.video.fpsBaseNextDts = 0;
        m_Mux.video.timestampList.clear();
        m_Mux.format.fileHeaderWritten = true;
    }

    const AVRational streamTimebase = m_Mux.video.streamOut->time_base;
    
    // RGYFrame -> AVFrame
    auto avframe = m_Mux.video.rawVideoFrame.getFree();
    if (!avframe) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate AVFrame.\n"));
        m_Mux.format.streamError = true;
        return RGY_ERR_NULL_PTR;
    }
    
    avframe->format = m_Mux.video.rawVideoCodecCtx->pix_fmt;
    avframe->width = surface->width();
    avframe->height = surface->height();
    avframe->pts = surface->timestamp();
    if (av_cmp_q(m_Mux.video.bitstreamTimebase, streamTimebase) != 0) {
        rgy_avframe_get_duration_ref(avframe.get()) = av_rescale_q(rgy_avframe_get_duration(avframe.get()), m_Mux.video.bitstreamTimebase, streamTimebase);;
        avframe->pts = av_rescale_q(avframe->pts, m_Mux.video.bitstreamTimebase, streamTimebase);
    }
    
    int ret = av_frame_get_buffer(avframe.get(), 0);
    if (ret < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate buffer for AVFrame: %s\n"), qsv_av_err2str(ret).c_str());
        avframe.reset();
        return RGY_ERR_NULL_PTR;
    }

    //色空間変換
    if (m_Mux.video.rawVideoConvert->getFunc(surface->csp(), csp_avpixfmt_to_rgy(m_Mux.video.rawVideoCodecCtx->pix_fmt), m_Mux.video.simdCsp) == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[surface->csp()], RGY_CSP_NAMES[csp_avpixfmt_to_rgy(m_Mux.video.rawVideoCodecCtx->pix_fmt)]);
        m_Mux.format.streamError = true;
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    void *dst_array[RGY_MAX_PLANES];
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        dst_array[i] = avframe->data[i];
    }
    const void *src_array[RGY_MAX_PLANES];
    surface->ptrArray((void **)src_array);

    sInputCrop crop = initCrop();
    m_Mux.video.rawVideoConvert->run(
        rgy_avframe_interlaced(avframe.get()), dst_array, src_array,
        surface->width(), surface->pitch(), surface->pitch(RGY_PLANE_C),
        avframe->linesize[0], avframe->linesize[1], surface->height(), avframe->height, crop.c);

    if (m_Mux.thread.thRawVideo) {
        auto pktFrame = pktMuxData(avframe.release());
        m_Mux.thread.thRawVideo->qPackets.push(pktFrame);
    } else {
        return VideoEncodeRawFrame(avframe.get());
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::VideoEncodeRawFrame(AVFrame *avframe) {
    if (m_Mux.format.formatCtx->video_codec_id != AV_CODEC_ID_RAWVIDEO) {
        AddMessage(RGY_LOG_ERROR, _T("Unsupported codec for VideoEncodeRawFrame: %s\n"), avcodec_get_name(m_Mux.format.formatCtx->video_codec_id));
        return RGY_ERR_UNSUPPORTED;
    }
    auto pkt = m_Mux.poolPkt->getFree();
    if (!pkt) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate AVPacket.\n"));
        m_Mux.video.rawVideoFrame.returnFree(&avframe);
        m_Mux.format.streamError = true;
        return RGY_ERR_NULL_PTR;
    }
    
    auto ret = avcodec_send_frame(m_Mux.video.rawVideoCodecCtx.get(), avframe);
    if (ret < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to send frame to raw video encoder: %s\n"), qsv_av_err2str(ret).c_str());
        m_Mux.video.rawVideoFrame.returnFree(&avframe);
        m_Mux.format.streamError = true;
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.video.rawVideoFrame.returnFree(&avframe);
    
    ret = avcodec_receive_packet(m_Mux.video.rawVideoCodecCtx.get(), pkt.get());
    if (ret < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to receive packet from raw video encoder: %s\n"), qsv_av_err2str(ret).c_str());
        m_Mux.format.streamError = true;
        return RGY_ERR_NULL_PTR;
    }

    if (m_Mux.thread.thOutput) {
        auto pktFrame = pktMuxData(pkt.release());
        m_Mux.thread.qVideoRawFrames.push(pktFrame);
    } else {
        return WriteNextPacketRawVideo(pkt.release(), nullptr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteNextPacketRawVideo(AVPacket *pkt, int64_t *writtenDts) {
    const bool flush = pkt == nullptr;
    if (flush) {
        return RGY_ERR_NONE; // 特にflushするものはない
    }

    const int pktSize = pkt->size;
    const AVRational streamTimebase = m_Mux.video.streamOut->time_base;
    pkt->flags &= 0x0000ffff; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
    pkt->flags |= AV_PKT_FLAG_KEY;
    pkt->stream_index = m_Mux.video.streamOut->index;
    pkt->pos = -1;
    pkt->dts = pkt->pts;
    if (writtenDts) {
        *writtenDts = av_rescale_q(pkt->dts, streamTimebase, QUEUE_DTS_TIMEBASE);
    }
    if (WRITE_PTS_DEBUG) {
        AddMessage((pkt->pts == AV_NOPTS_VALUE) ? RGY_LOG_ERROR : RGY_LOG_WARN, _T("%3d, %12s, pts, %lld (%d/%d) [%s]\n"),
            m_Mux.video.streamOut->index, char_to_tstring(avcodec_get_name(m_Mux.video.streamOut->codecpar->codec_id)).c_str(),
            pkt->pts, streamTimebase.num, streamTimebase.den, getTimestampString(pkt->pts, streamTimebase).c_str());
    }
    const auto ret_write = av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
    if (ret_write != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Error: Failed to write %s stream %d frame: %s.\n"),
            get_media_type_string(m_Mux.video.streamOut->codecpar->codec_id).c_str(),
            m_Mux.video.streamOut->index, qsv_av_err2str(ret_write).c_str());
        m_Mux.format.streamError = true;
    }
    m_Mux.poolPkt->returnFree(&pkt);
    m_encSatusInfo->SetOutputData(RGY_FRAMETYPE_IDR, pktSize, 0);
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

vector<int> RGYOutputAvcodec::GetStreamTrackIdList() {
    vector<int> streamTrackId;
    streamTrackId.reserve(m_Mux.audio.size());
    for (const auto& audio : m_Mux.audio) {
        streamTrackId.push_back(audio.inTrackId);
    }
    for (const auto& sub : m_Mux.other) {
        streamTrackId.push_back(sub.inTrackId);
    }
    return streamTrackId;
}

AVMuxAudio *RGYOutputAvcodec::getAudioPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = pktFlagGetTrackID(pkt);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].streamIndexIn == streamIndex
            && m_Mux.audio[i].inTrackId == inTrackId) {
            return &m_Mux.audio[i];
        }
    }
    return nullptr;
}

AVMuxAudio *RGYOutputAvcodec::getAudioStreamData(int trackId, int subStreamId) {
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].inTrackId == trackId
            && m_Mux.audio[i].inSubStream == subStreamId) {
            return &m_Mux.audio[i];
        }
    }
    return nullptr;
}

AVMuxOther *RGYOutputAvcodec::getOtherPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = pktFlagGetTrackID(pkt);
    for (int i = 0; i < (int)m_Mux.other.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.other[i].streamIndexIn == streamIndex
            && m_Mux.other[i].inTrackId == inTrackId) {
            return &m_Mux.other[i];
        }
    }
    return NULL;
}

RGY_ERR RGYOutputAvcodec::applyBitstreamFilterOther(AVPacket *pkt, const AVMuxOther *muxOther) {
    int ret = 0;
    if (0 > (ret = av_bsf_send_packet(muxOther->bsfc, pkt))) {
        m_Mux.poolPkt->returnFree(&pkt);
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"), char_to_tstring(muxOther->bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(muxOther->bsfc, pkt);
    if (ret == AVERROR(EAGAIN)) {
        pkt->size = 0;
        pkt->duration = 0;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
        //最初のフレームとかでなければ、エラーを許容し、単に処理しないようにする
        //多くの場合、ここでのエラーはtsなどの最終音声フレームが不完全なことで発生する
        //先頭から連続30回エラーとなった場合はおかしいのでエラー終了するようにする
        AddMessage(RGY_LOG_WARN, _T("failed to run %s bitstream filter: %s.\n"), char_to_tstring(muxOther->bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        pkt->duration = 0; //書き込み処理が行われないように
        return RGY_WRN_FILTER_SKIPPED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::applyBitstreamFilterAudio(AVPacket *pkt, AVMuxAudio *muxAudio) {
    int ret = 0;
    const tstring filterName = char_to_tstring(muxAudio->bsfc->filter->name);
    if (0 > (ret = av_bsf_send_packet(muxAudio->bsfc, pkt))) {
        m_Mux.poolPkt->returnFree(&pkt);
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"), filterName.c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(muxAudio->bsfc, pkt);
    if (ret == AVERROR(EAGAIN)) {
        pkt->size = 0;
        pkt->duration = 0;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
        //最初のフレームとかでなければ、エラーを許容し、単に処理しないようにする
        //多くの場合、ここでのエラーはtsなどの最終音声フレームが不完全なことで発生する
        //先頭から連続30回エラーとなった場合はおかしいのでエラー終了するようにする
        if (muxAudio->packetWritten == 0) {
            muxAudio->bsfErrorFromStart++;
            static int AUDIO_BSFFILTER_ERROR_THRESHOLD = 30;
            if (muxAudio->bsfErrorFromStart > AUDIO_BSFFILTER_ERROR_THRESHOLD) {
                m_Mux.format.streamError = true;
                AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter for %d times: %s.\n"), filterName.c_str(), AUDIO_BSFFILTER_ERROR_THRESHOLD, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
        }
        AddMessage(RGY_LOG_WARN, _T("failed to run %s bitstream filter: %s.\n"), filterName.c_str(), qsv_av_err2str(ret).c_str());
        pkt->duration = 0; //書き込み処理が行われないように
        return RGY_WRN_FILTER_SKIPPED;
    }
    muxAudio->bsfErrorFromStart = 0;
    return RGY_ERR_NONE;
}

//音声/字幕パケットを実際に書き出す
// muxAudio ... [i]  pktに対応するストリーム情報
// pkt       ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// dts       ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVMuxAudio *muxAudio, AVPacket *pkt, int samples, int64_t *writtenDts) {
    if (pkt == nullptr || pkt->buf == nullptr) {
        //muxAudioのnullpacketが到着した
        muxAudio->flushed = true;
        //送出したEOSがすべて来たか確認しないといけない
        const auto loglevel_flush = RGY_LOG_DEBUG;
        if (std::find_if(m_Mux.audio.begin(), m_Mux.audio.end(), [](const auto& audio) {
            return !audio.flushed;
        }) == m_Mux.audio.end()) {
            *writtenDts = INT64_MAX;
            AddMessage(RGY_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        } else if (m_printMes && loglevel_flush >= m_printMes->getLogLevel(RGY_LOGT_OUT)) { // ログ出力用
            tstring tracks;
            for (auto& aud : m_Mux.audio) {
                if (!aud.flushed) {
                    if (tracks.length() > 0) tracks += _T(", ");
                    tracks += strsprintf(_T("%d"), trackID(aud.inTrackId));
                }
            }
            AddMessage(loglevel_flush, _T("null packet from %d, waiting for EOS to come from other tracks: [%s].\n"), muxAudio ? trackID(muxAudio->inTrackId) : 0, tracks.c_str());
        }
        return;
    }
    const auto orig_pts = pkt->pts;
    const auto orig_dts = pkt->dts;
    //durationについて、sample数から出力ストリームのtimebaseに変更する
    pkt->stream_index = muxAudio->streamOut->index;
    pkt->flags = AV_PKT_FLAG_KEY; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
    const AVRational samplerate = { 1, (muxAudio->outCodecEncodeCtx) ? muxAudio->outCodecEncodeCtx->sample_rate : muxAudio->streamIn->codecpar->sample_rate };
    const bool ptsInvalid = pkt->pts == AV_NOPTS_VALUE;
    if (!muxAudio->outCodecEncodeCtx) {
        if (samples > 0) {
            //av_rescale_deltaの入力ptsはAV_NOPTS_VALUEではない必要があるのでチェックする
            if (pkt->pts == AV_NOPTS_VALUE) {
                muxAudio->outputSampleOffset += samples;
                pkt->pts = muxAudio->lastPtsOut + (int)av_rescale_q(muxAudio->outputSampleOffset, samplerate, muxAudio->streamOut->time_base);
                muxAudio->dec_rescale_delta = AV_NOPTS_VALUE;
            } else {
                pkt->pts = av_rescale_delta(muxAudio->streamIn->time_base, pkt->pts, samplerate, samples, &muxAudio->dec_rescale_delta, muxAudio->streamOut->time_base);
            }
        } else {
            pkt->pts = av_rescale_q(pkt->pts, muxAudio->streamIn->time_base, muxAudio->streamOut->time_base);
        }
    } else {
        pkt->pts = av_rescale_q(pkt->pts, muxAudio->outCodecEncodeCtx->time_base, muxAudio->streamOut->time_base);
    }
    if (m_Mux.video.streamOut && m_Mux.video.inputFirstKeyPts != 0 && !m_Mux.format.timestampPassThrough) {
        pkt->pts -= av_rescale_q(m_Mux.video.inputFirstKeyPts, m_Mux.video.inputStreamTimebase, muxAudio->streamOut->time_base);
    }
    if (muxAudio->lastPtsOut != AV_NOPTS_VALUE) {
        //以前のptsより前になりそうになったら修正する
        const auto maxPts = muxAudio->lastPtsOut + !(m_Mux.format.formatCtx->oformat->flags & AVFMT_TS_NONSTRICT);
        if (pkt->pts < maxPts) {
            auto loglevel = (maxPts - pkt->pts > 2) ? RGY_LOG_WARN : RGY_LOG_DEBUG;
            if (loglevel < m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                AddMessage(loglevel, _T("Timestamp error in stream %d, previous: %lld, current: %lld [timebase: %d/%d].\n"),
                    muxAudio->streamOut->index,
                    (long long int)muxAudio->lastPtsOut,
                    (long long int)pkt->pts,
                    muxAudio->streamOut->time_base.num, muxAudio->streamOut->time_base.den);
                AddMessage(loglevel, _T("                              previous: %s current: %s\n"),
                    getTimestampString(muxAudio->lastPtsOut, muxAudio->streamOut->time_base).c_str(),
                    getTimestampString(pkt->pts, muxAudio->streamOut->time_base).c_str());
                AddMessage(loglevel, _T("Changing timestamp to %lld(%s), this may corrupt av-synchronization.\n"),
                    (long long int)maxPts, getTimestampString(maxPts, muxAudio->streamOut->time_base).c_str());
            }
            pkt->pts = maxPts;
        }
    }
    pkt->dts          = pkt->pts;
    pkt->duration     = (int)av_rescale_q(samples, samplerate, muxAudio->streamOut->time_base);
    if (pkt->duration == 0) {
        pkt->duration = (int)(pkt->pts - muxAudio->lastPtsOut);
    }
    if (!ptsInvalid) {
        muxAudio->lastPtsOut = pkt->pts;
        muxAudio->outputSampleOffset = 0;
    }
    *writtenDts = av_rescale_q(pkt->dts, muxAudio->streamOut->time_base, QUEUE_DTS_TIMEBASE);
    if (*writtenDts != AV_NOPTS_VALUE) {
        atomic_max(m_Mux.thread.streamOutMaxDts, *writtenDts);
    }
    if (WRITE_PTS_DEBUG) {
        AddMessage(RGY_LOG_WARN, _T("audio %3d [%3d.%3d], %12s, pts, %lld (%d/%d) [%s]\n"),
            pkt->stream_index,trackID(muxAudio->inTrackId), muxAudio->inSubStream, char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->streams[pkt->stream_index]->codecpar->codec_id)).c_str(),
            pkt->pts, muxAudio->streamOut->time_base.num, muxAudio->streamOut->time_base.den, getTimestampString(pkt->pts, muxAudio->streamOut->time_base).c_str());
    }
    if (muxAudio->fpTsLogFile) {
        _ftprintf(muxAudio->fpTsLogFile.get(), _T(" , %20lld, %8d, %d\n"), (lls)pkt->pts, (int)pkt->duration, pkt->size);
        {
            std::lock_guard<std::mutex> lock(m_Mux.format.fpTsLogMtx);
            _ftprintf(m_Mux.format.fpTsLogFile.get(), _T("a, %d,  , %20lld, %20lld, %20lld, %20lld, %d, %7zd\n"), pkt->stream_index, (lls)orig_pts, (lls)orig_dts, (lls)pkt->pts, (lls)pkt->dts, (int)pkt->duration, (lls)pkt->size);
        }
    }
    if (pkt->pts >= 0 || m_Mux.format.allowOtherNegativePts) {
        //av_interleaved_write_frameに渡ったパケットは開放する必要がない
        const auto ret_write = av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
        if (ret_write != 0) {
            AddMessage(RGY_LOG_ERROR, _T("Error: Failed to write %s stream %d frame: %s.\n"),
                get_media_type_string(muxAudio->streamOut->codecpar->codec_id).c_str(),
                muxAudio->streamOut->index, qsv_av_err2str(ret_write).c_str());
            m_Mux.format.streamError = true;
        }
        muxAudio->outputSamples += samples;
    }
    m_Mux.poolPkt->returnFree(&pkt);
}

//音声/字幕パケットを実際に書き出す (構造体版)
// pktData->muxAudio ... [i]  pktに対応するストリーム情報
// &pktData->pkt      ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// pktData->samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// &pktData->dts      ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData) {
    WriteNextPacketProcessed(pktData->muxAudio, pktData->pkt, pktData->samples, &pktData->dts);
}

void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData, int64_t *writtenDts) {
    WriteNextPacketProcessed(pktData->muxAudio, pktData->pkt, pktData->samples, &pktData->dts);
    *writtenDts = pktData->dts;
}

vector<unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>> RGYOutputAvcodec::AudioDecodePacket(AVMuxAudio *muxAudio, AVPacket *pkt) {
    vector<unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>> decodedFrames;
    if (muxAudio->decodeError > muxAudio->ignoreDecodeError) {
        return decodedFrames;
    }
    const auto in_pts = (pkt) ? pkt->pts : AV_NOPTS_VALUE;

    bool sent_packet = false;

    //最終的な出力フレーム
    int64_t recieved_samples = 0;
    int recv_ret = 0;
    for (;;) {
        unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> receivedData(nullptr, RGYAVDeleter<AVFrame>(av_frame_free));
        int send_ret = 0;
        //必ず一度はパケットを送る
        if (!sent_packet || (pkt && pkt->size > 0)) {
            sent_packet = true;
            //ひとつのパケットをデコーダに送る
            send_ret = avcodec_send_packet(muxAudio->outCodecDecodeCtx, pkt);
            //AVERROR(EAGAIN) -> パケットを送る前に受け取る必要がある (パケットが受け取られていないので解放しない)
            if (send_ret != AVERROR(EAGAIN)) {
                m_Mux.poolPkt->returnFree(&pkt);
            }
            if (send_ret == AVERROR_EOF) {
                AddMessage(RGY_LOG_DEBUG, _T("avcodec writer: failed to send packet to audio decoder, already flushed.\n"));
                break;
            } else if (send_ret == AVERROR(EINVAL)) {
                AddMessage(RGY_LOG_DEBUG, _T("avcodec writer: failed to send packet to audio decoder, requires flush.\n"));
                break;
            } else if (send_ret == AVERROR(ENOMEM)) {
                break;
            }
        }
        if (send_ret < 0 && send_ret != AVERROR(EAGAIN)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to send packet to audio decoder: %s.\n"), qsv_av_err2str(send_ret).c_str());
            muxAudio->decodeError++;
        } else {
            receivedData = m_Mux.poolFrame->getFree();
            recv_ret = avcodec_receive_frame(muxAudio->outCodecDecodeCtx, receivedData.get());
            if (recv_ret == AVERROR(EAGAIN)   //もっとパケットを送る必要がある
                || recv_ret == AVERROR_EOF) { //最後まで読み込んだ
                break;
            }
            if (recv_ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to receive frame from audio decoder: %s.\n"), qsv_av_err2str(recv_ret).c_str());
                muxAudio->decodeError++;
            } else {
                muxAudio->decodeError = 0;
            }

            if (receivedData) {
                if (receivedData->pts == AV_NOPTS_VALUE) {
                    receivedData->pts = in_pts;
                }
                if (receivedData->pts == AV_NOPTS_VALUE) {
                    const auto nextPts = muxAudio->decodeNextPts;
                    receivedData->pts = (nextPts == AV_NOPTS_VALUE) ? 0 : nextPts;
                } else {
                    //デコード後、samplerateを基準とする時間に変換する
                    const auto timebase_samplerate = av_make_q(1, muxAudio->outCodecDecodeCtx->sample_rate);
                    const auto orig_pts = receivedData->pts;
                    receivedData->pts = av_rescale_delta(muxAudio->streamIn->time_base, orig_pts,
                        timebase_samplerate, receivedData->nb_samples, &muxAudio->dec_rescale_delta, timebase_samplerate);
                    receivedData->pts += recieved_samples;
                    //fprintf(stderr, "pkt pts: %d, orig pts: %12d, out pts: %12d, samples: %6d\n", (int)pktInInfo.pts, (int)orig_pts, (int)receivedData->pts, receivedData->nb_samples);
                }
                muxAudio->decodeNextPts = receivedData->pts + receivedData->nb_samples;
                recieved_samples += receivedData->nb_samples;
            }
        }

        if (muxAudio->decodeError) {
            if (muxAudio->decodeError <= muxAudio->ignoreDecodeError) {
#if 0
                //デコードエラーを無視する場合、入力パケットのサイズ分、無音を挿入する
                unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> silentFrame(av_frame_alloc(), RGYAVDeleter<AVFrame>(av_frame_free));
                AVRational samplerate = { 1, muxAudio->outCodecDecodeCtx->sample_rate };
                silentFrame->nb_samples     = (int)av_rescale_q(pktInInfo.duration, muxAudio->streamIn->time_base, samplerate);
                silentFrame->channels       = muxAudio->outCodecDecodeCtx->channels;
                silentFrame->channel_layout = muxAudio->outCodecDecodeCtx->channel_layout;
                silentFrame->sample_rate    = muxAudio->outCodecDecodeCtx->sample_rate;
                silentFrame->format         = muxAudio->outCodecDecodeCtx->sample_fmt;
                silentFrame->pts            = receivedData->pts;
                av_frame_get_buffer(silentFrame.get(), 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
                av_samples_set_silence((uint8_t **)silentFrame->data, 0, silentFrame->nb_samples, silentFrame->channels, (AVSampleFormat)silentFrame->format);
                decodedFrames.push_back(std::move(silentFrame));
#else
                AddMessage(RGY_LOG_WARN, _T("avcodec writer: ignore error(%d) on audio #%d decode at %lld(%s)\n"),
                    muxAudio->decodeError, trackID(muxAudio->inTrackId), in_pts, getTimestampString(in_pts, muxAudio->streamIn->time_base).c_str());
#endif
            } else {
                AddMessage(RGY_LOG_ERROR, _T("avcodec writer: failed to decode audio #%d for %d times.\n"), trackID(muxAudio->inTrackId), muxAudio->decodeError);
                m_Mux.format.streamError = true;
                break;
            }
        } else if (receivedData) {
            decodedFrames.push_back(std::move(receivedData));
        }
    }
    return decodedFrames;
}

//音声をフィルタ
vector<AVPktMuxData> RGYOutputAvcodec::AudioFilterFrame(vector<AVPktMuxData> inputFrames) {
    vector<AVPktMuxData> outputFrames;
    for (auto& pktData : inputFrames) {
        AVMuxAudio *muxAudio = pktData.muxAudio;
        if (pktData.muxAudio->filterGraph == nullptr) {
            //フィルタリングなし
            outputFrames.push_back(pktData);
        } else {
            const bool flush = pktData.frame == nullptr;
            if (pktData.frame != nullptr) {
                //音声入力フォーマットに変更がないか確認し、もしあればresamplerを再初期化する
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
                auto sts = InitAudioFilter(muxAudio, getChannelCount(&pktData.frame->ch_layout), &pktData.frame->ch_layout, pktData.frame->sample_rate, (AVSampleFormat)pktData.frame->format, muxAudio->audioResamplerPrm);
#else
                auto sts = InitAudioFilter(muxAudio, pktData.frame->channels, &pktData.frame->channel_layout, pktData.frame->sample_rate, (AVSampleFormat)pktData.frame->format, muxAudio->audioResamplerPrm);
#endif
                if (sts != RGY_ERR_NONE) {
                    m_Mux.format.streamError = true;
                    break;
                }
            }
            { //フィルターチェーンにフレームを追加
                auto ret = av_buffersrc_add_frame_flags(muxAudio->filterBufferSrcCtx, pktData.frame, AV_BUFFERSRC_FLAG_PUSH);
                // AVFrame構造体の破棄
                m_Mux.poolFrame->returnFree(&pktData.frame);
                if (ret < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to feed the audio filtergraph\n"));
                    m_Mux.format.streamError = true;
                    break;
                }
            }
            for (;;) {
                auto filteredFrame = m_Mux.poolFrame->getFree();
                auto ret = av_buffersink_get_frame_flags(muxAudio->filterBufferSinkCtx, filteredFrame.get(), (flush) ? AV_BUFFERSINK_FLAG_NO_REQUEST : 0);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    m_Mux.format.streamError = true;
                    break;
                }
                AVPktMuxData pktFiltered = pktData;
                pktFiltered.samples = filteredFrame->nb_samples;
                pktFiltered.frame = filteredFrame.release();
                outputFrames.push_back(pktFiltered);
            }
            if (m_Mux.format.streamError) {
                break;
            }
        }
    }
    return outputFrames;
}

vector<AVPktMuxData> RGYOutputAvcodec::AudioFilterFrameFlush(AVMuxAudio *muxAudio) {
    vector<AVPktMuxData> flushFrame;
    AVPktMuxData pktData = { 0 };
    pktData.type = MUX_DATA_TYPE_FRAME;
    pktData.frame = nullptr;
    pktData.got_result = TRUE;
    pktData.muxAudio = muxAudio;
    flushFrame.push_back(pktData);
    return AudioFilterFrame(flushFrame);
}

//音声をエンコード
vector<AVPktMuxData> RGYOutputAvcodec::AudioEncodeFrame(AVMuxAudio *muxAudio, AVFrame *frame) {
    vector<AVPktMuxData> encPktDatas;

    if (frame) {
        //エンコーダのtimebaseに変換
        const auto timebase_filter = (muxAudio->filterGraph)
            ? av_buffersink_get_time_base(muxAudio->filterBufferSinkCtx)
            : av_make_q(1, muxAudio->outCodecDecodeCtx->sample_rate);
        frame->pts = av_rescale_q(frame->pts, timebase_filter, muxAudio->outCodecEncodeCtx->time_base);
    }
    int ret = avcodec_send_frame(muxAudio->outCodecEncodeCtx, frame);
    if (ret == AVERROR_EOF) {
        return encPktDatas;
    }
    if (ret < 0) {
        AddMessage(RGY_LOG_WARN, _T("avcodec writer: failed to send frame to audio encoder #%d: %s\n"), trackID(muxAudio->inTrackId), qsv_av_err2str(ret).c_str());
        muxAudio->encodeError = true;
        return encPktDatas;
    }

    while (ret >= 0) {
        auto pkt = m_Mux.poolPkt->getFree();
        ret = avcodec_receive_packet(muxAudio->outCodecEncodeCtx, pkt.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            AddMessage(RGY_LOG_WARN, _T("avcodec writer: failed to encode audio #%d: %s\n"), trackID(muxAudio->inTrackId), qsv_av_err2str(ret).c_str());
            muxAudio->encodeError = true;
        }
        AVPktMuxData pktData = { 0 };
        pktData.type = MUX_DATA_TYPE_PACKET;
        pktData.muxAudio = muxAudio;
        pktData.pkt = pkt.release();
        pktFlagSetTrackID(pktData.pkt, pktData.muxAudio->inTrackId);
        pktData.samples = (int)av_rescale_q(pktData.pkt->duration, muxAudio->outCodecEncodeCtx->pkt_timebase, { 1, muxAudio->streamIn->codecpar->sample_rate });
        encPktDatas.push_back(pktData);
    }
    return encPktDatas;
}

void RGYOutputAvcodec::AudioFlushStream(AVMuxAudio *muxAudio, int64_t *writtenDts) {
    if (muxAudio->flushed) { // AudioFlushStream は一度のみでOK
        return;
    }
    while (muxAudio->outCodecDecodeCtx && !muxAudio->encodeError) {
        auto decodedFrames = AudioDecodePacket(muxAudio, nullptr);
        if (decodedFrames.size() == 0) {
            break;
        }
        vector<AVPktMuxData> audioFrames;
        for (size_t i = 0; i < decodedFrames.size(); i++) {
            AVPktMuxData audPkt;
            audPkt.pkt = m_Mux.poolPkt->getFree().release();
            pktFlagSetTrackID(audPkt.pkt, muxAudio->inTrackId);
            audPkt.dts = AV_NOPTS_VALUE;
            audPkt.samples = 0;
            audPkt.type = MUX_DATA_TYPE_FRAME;
            audPkt.frame = decodedFrames[i].release();
            audPkt.got_result = audPkt.frame && audPkt.frame->nb_samples > 0;
            audPkt.muxAudio = muxAudio;
            audioFrames.push_back(audPkt);
        }
        //フィルタリングを行う
        WriteNextPacketToAudioSubtracks(std::move(audioFrames));
    }
    if (muxAudio->filterGraph) {
        WriteNextPacketAudioFrame(AudioFilterFrameFlush(muxAudio));
    }
    while (muxAudio->outCodecEncodeCtx) {
        auto encPktDatas = AudioEncodeFrame(muxAudio, nullptr);
        if (encPktDatas.size() == 0) {
            break;
        }
        if (muxAudio->decodeError > muxAudio->ignoreDecodeError)
            break;
        for (auto& pktMux : encPktDatas) {
            WriteNextPacketProcessed(&pktMux, writtenDts);
        }
    }
    muxAudio->flushed = true; // AudioFlushStream を完了したフラグ
}

RGY_ERR RGYOutputAvcodec::SubtitleEncode(const AVMuxOther *muxSub, AVSubtitleData *subData) {
    const int64_t ptsOffset = (m_Mux.video.streamOut && m_Mux.format.timestampPassThrough)
        ? 0ll : -1 * av_rescale_q(m_Mux.video.inputFirstKeyPts, m_Mux.video.inputStreamTimebase, muxSub->streamOut->time_base);

    //AV_CODEC_ID_DVB_SUBTITLEははじめりと終わりで2パケット
    const int nOutPackets = 1 + (muxSub->outCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE);
    for (int i = 0; i < nOutPackets; i++) {
        auto& sub = subData->decodecSub;
        sub.pts += av_rescale_q(sub.start_display_time, av_make_q(1, 1000), av_make_q(1, AV_TIME_BASE));
        sub.end_display_time -= sub.start_display_time;
        sub.start_display_time = 0;
        if (i > 0) {
            sub.num_rects = 0;
        }

        int sub_out_size = avcodec_encode_subtitle(muxSub->outCodecEncodeCtx, muxSub->bufConvert, SUB_ENC_BUF_MAX_SIZE, &subData->decodecSub);
        if (sub_out_size < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to encode subtitle.\n"));
            m_Mux.format.streamError = true;
            return RGY_ERR_UNKNOWN;
        }

        auto pktOut = m_Mux.poolPkt->getFree();
        pktOut->data = muxSub->bufConvert;
        pktOut->stream_index = muxSub->streamOut->index;
        pktOut->size = sub_out_size;
        pktOut->pts = av_rescale_q(sub.pts, av_make_q(1, AV_TIME_BASE), muxSub->streamOut->time_base);
        pktOut->duration = (int)av_rescale_q_rnd(sub.end_display_time - sub.start_display_time, av_make_q(1, 1000), muxSub->streamOut->time_base, AV_ROUND_ZERO);
        if (subData->origDuration > 0) {
            // pts + duration <= 次のptsとなるよう、オリジナルのptsでもチェックする
            auto ts_start = av_rescale_q(subData->origPts, muxSub->outCodecDecodeCtx->pkt_timebase, muxSub->streamOut->time_base);
            auto ts_end   = av_rescale_q(subData->origPts + subData->origDuration, muxSub->outCodecDecodeCtx->pkt_timebase, muxSub->streamOut->time_base);
            pktOut->duration = std::min(pktOut->duration, av_rescale_q_rnd(ts_end - ts_start, muxSub->streamOut->time_base, muxSub->streamOut->time_base, AV_ROUND_ZERO));
        }
        if (muxSub->outCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {
            pktOut->pts += 90 * ((i == 0) ? sub.start_display_time : sub.end_display_time);
        }
        pktOut->pts += ptsOffset;
        pktOut->dts = pktOut->pts;
        const auto ret_write = av_interleaved_write_frame(m_Mux.format.formatCtx, pktOut.get());
        if (ret_write != 0) {
            AddMessage(RGY_LOG_ERROR, _T("Error: Failed to write %s stream %d frame: %s.\n"),
                get_media_type_string(muxSub->streamOut->codecpar->codec_id).c_str(),
                muxSub->streamOut->index, qsv_av_err2str(ret_write).c_str());
            m_Mux.format.streamError = true;
        }
    }
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::SubtitleTranscode(AVMuxOther *muxSub, AVPacket *pkt) {
    //timescaleの変換が入ると、pts + duration > 次のpts となることがある
    //オリジナルのptsを使って再計算する
    AVSubtitleData subData;
    subData.origPts = pkt->pts;
    subData.origDuration = pkt->duration;

    int got_sub = 0;
    if (0 > avcodec_decode_subtitle2(muxSub->outCodecDecodeCtx, &subData.decodecSub, &got_sub, pkt)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
        m_Mux.format.streamError = true;
    }
    if (!muxSub->bufConvert) {
        AddMessage(RGY_LOG_ERROR, _T("No buffer for encoding subtitle.\n"));
        m_Mux.format.streamError = true;
    }
    m_Mux.poolPkt->returnFree(&pkt);
    if (m_Mux.format.streamError)
        return RGY_ERR_UNKNOWN;
    if (!got_sub || subData.decodecSub.num_rects == 0)
        return RGY_ERR_NONE;

    if (muxSub->decodedSub.size() > 0) {
        // durationが不明な字幕をためていたものをまず処理する
        // 現在のフレームの開始時刻までとする
        for (auto& cachedSub : muxSub->decodedSub) {
            cachedSub.origDuration = subData.origPts - cachedSub.origPts;
            cachedSub.decodecSub.end_display_time = cachedSub.decodecSub.start_display_time + (uint32_t)av_rescale_q_rnd(cachedSub.origDuration, muxSub->outCodecDecodeCtx->pkt_timebase, av_make_q(1, 1000), AV_ROUND_ZERO);
            auto ret = SubtitleEncode(muxSub, &cachedSub);
            if (ret != RGY_ERR_NONE) {
                return ret;
            }
        }
        muxSub->decodedSub.clear();
    }
    if (subData.origDuration == 0 && subData.decodecSub.end_display_time == std::numeric_limits<uint32_t>::max()) {
        // durationが不明な字幕の場合、次の字幕が来るまで保留する
        muxSub->decodedSub.push_back(subData);
        return RGY_ERR_NONE;
    }


    return SubtitleEncode(muxSub, &subData);
}

RGY_ERR RGYOutputAvcodec::WriteOtherPacket(AVPacket *pkt) {
    AVMuxOther* pMuxOther = getOtherPacketStreamData(pkt);
    if (pMuxOther->bsfc) {
        auto sts = applyBitstreamFilterOther(pkt, pMuxOther);
        //bitstream filterを正常に起動できなかった
        if (sts < RGY_ERR_NONE) {
            m_Mux.format.streamError = true;
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
    }
    if (pMuxOther->outCodecEncodeCtx) {
        return SubtitleTranscode(pMuxOther, pkt);
    }
    //字幕を処理する
    const bool ptsInvalid = pkt->pts == AV_NOPTS_VALUE;
    const AVRational vid_pkt_timebase = av_isvalid_q(m_Mux.video.inputStreamTimebase) ? m_Mux.video.inputStreamTimebase : av_inv_q(m_Mux.video.outputFps);
    const int64_t pts_offset = (m_Mux.format.timestampPassThrough) ? 0ll : av_rescale_q(m_Mux.video.inputFirstKeyPts, vid_pkt_timebase, pMuxOther->streamInTimebase);
    const AVRational timebase_conv = (pMuxOther->outCodecDecodeCtx) ? pMuxOther->outCodecDecodeCtx->pkt_timebase : pMuxOther->streamOut->time_base;
    if (pkt->pts != AV_NOPTS_VALUE) pkt->pts = av_rescale_q(std::max<int64_t>(0, pkt->pts - pts_offset), pMuxOther->streamInTimebase, timebase_conv);
    if (pkt->dts != AV_NOPTS_VALUE) pkt->dts = av_rescale_q(std::max<int64_t>(0, pkt->dts - pts_offset), pMuxOther->streamInTimebase, timebase_conv);
    if (pMuxOther->lastPtsOut != AV_NOPTS_VALUE) {
        //以前のptsより前になりそうになったら修正する
        const auto maxPts = pMuxOther->lastPtsOut + ((m_Mux.format.formatCtx->oformat->flags & AVFMT_TS_NONSTRICT) ? 0 : 1);
        if (pkt->pts < maxPts) {
            auto loglevel = (maxPts - pkt->pts > 2 && pMuxOther->streamOut->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE /*字幕の場合は頻繁に発生することがある*/) ? RGY_LOG_WARN : RGY_LOG_DEBUG;
            if (loglevel >= m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                AddMessage(loglevel, _T("Timestamp error in stream %d, previous: %lld, current: %lld [timebase: %d/%d].\n"),
                    pMuxOther->streamOut->index,
                    (long long int)pMuxOther->lastPtsOut,
                    (long long int)pkt->pts,
                    pMuxOther->streamOut->time_base.num, pMuxOther->streamOut->time_base.den);
                AddMessage(loglevel, _T("                              previous: %s current: %s\n"),
                    getTimestampString(pMuxOther->lastPtsOut, pMuxOther->streamOut->time_base).c_str(),
                    getTimestampString(pkt->pts, pMuxOther->streamOut->time_base).c_str());
                AddMessage(loglevel, _T("Changing timestamp to %lld(%s), this may corrupt synchronization of the stream.\n"),
                    (long long int)maxPts, getTimestampString(maxPts, pMuxOther->streamOut->time_base).c_str());
            }
            pkt->pts = maxPts;
            pkt->dts = maxPts;
        }
    }
    if (WRITE_PTS_DEBUG) {
        AddMessage((pkt->pts == AV_NOPTS_VALUE) ? RGY_LOG_ERROR : RGY_LOG_WARN, _T("%3d, %12s, pts, %lld (%d/%d) [%s]\n"),
            pMuxOther->streamOut->index, char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->streams[pMuxOther->streamOut->index]->codecpar->codec_id)).c_str(),
            pkt->pts, timebase_conv.num, timebase_conv.den, getTimestampString(pkt->pts, timebase_conv).c_str());
    }
    if (!ptsInvalid) {
        pMuxOther->lastPtsOut = pkt->pts;
    }
    pkt->flags &= 0x0000ffff; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
    pkt->duration = (int)av_rescale_q(pkt->duration, pMuxOther->streamInTimebase, pMuxOther->streamOut->time_base);
    pkt->stream_index = pMuxOther->streamOut->index;
    pkt->pos = -1;
    // 字幕やデータストリームに関しては、連続で来るとは限らないので、考慮しないことにする
    //if (pkt->dts != AV_NOPTS_VALUE) {
    //    atomic_max(m_Mux.thread.streamOutMaxDts, av_rescale_q(pkt->dts, timebase_conv, QUEUE_DTS_TIMEBASE));
    //}
    const auto ret_write = av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
    if (ret_write != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Error: Failed to write %s stream %d frame: %s.\n"),
            get_media_type_string(pMuxOther->streamOut->codecpar->codec_id).c_str(),
            pMuxOther->streamOut->index, qsv_av_err2str(ret_write).c_str());
        m_Mux.format.streamError = true;
    }
    m_Mux.poolPkt->returnFree(&pkt);
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

AVPktMuxData RGYOutputAvcodec::pktMuxData(AVPacket *pkt) {
    AVPktMuxData data = { 0 };
    data.type = MUX_DATA_TYPE_PACKET;
    if (pkt) {
        data.pkt = pkt;
        data.muxAudio = getAudioPacketStreamData(pkt);
    }
    return data;
}

AVPktMuxData RGYOutputAvcodec::pktMuxData(AVFrame *pFrame) {
    AVPktMuxData data = { 0 };
    data.type = MUX_DATA_TYPE_FRAME;
    data.frame = pFrame;
    return data;
}

//対象パケットの担当スレッドを探す
AVMuxThreadWorker *RGYOutputAvcodec::getPacketWorker(const AVMuxAudio *muxAudio, const int type) {
    if (type == AUD_QUEUE_OUT) {
        return m_Mux.thread.thOutput.get();
    }
    auto worker = m_Mux.thread.thAud.find(muxAudio);
    if (worker == m_Mux.thread.thAud.end()) {
        worker = m_Mux.thread.thAud.find(nullptr);
    }
    return (type == AUD_QUEUE_PROCESS) ? &worker->second->process : &worker->second->encode;
}

RGY_ERR RGYOutputAvcodec::WriteNextPacket(AVPacket *pkt) {
    AVPktMuxData pktData = pktMuxData(pkt);
#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.thread.thOutput) {
        if (pkt == nullptr) {
            //音声の全トラックにnullパケット送信
            for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
                auto mux = &m_Mux.audio[i];
                if (m_Mux.thread.threadActiveAudioEncode()) {
                    getPacketWorker(mux, AUD_QUEUE_ENCODE)->qPackets.set_keep_length(0);
                }
                if (m_Mux.thread.threadActiveAudioProcess()) {
                    getPacketWorker(mux, AUD_QUEUE_PROCESS)->qPackets.set_keep_length(0);
                }
                if (mux->inSubStream == 0) { // サブトラックには送信しない
                    const auto worker = getPacketWorker(mux, (m_Mux.thread.threadActiveAudioProcess()) ? AUD_QUEUE_PROCESS : AUD_QUEUE_OUT);
                    AddMessage(RGY_LOG_DEBUG, _T("Send null packet to worker %d.\n"), trackID(mux->inTrackId));
                    AVPktMuxData zeroFilled = { 0 };
                    zeroFilled.muxAudio = mux;
                    if (!worker->qPackets.push(zeroFilled)) {
                        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for audio packet queue.\n"));
                        m_Mux.format.streamError = true;
                    }
                }
            }
        } else {
            AVMuxThreadWorker *worker = (m_Mux.thread.threadActiveAudioProcess()) ? getPacketWorker(pktData.muxAudio, AUD_QUEUE_PROCESS) : m_Mux.thread.thOutput.get();
            auto& audioQueue = worker->qPackets;
            auto heEventPktAdd = worker->heEventPktAdded;
            if (!audioQueue.push(pktData)) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for audio packet queue.\n"));
                m_Mux.format.streamError = true;
            }
            SetEvent(heEventPktAdd);
        }
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }
#endif
    if (pkt == nullptr) {
        //音声の全トラックにnullパケット送信
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            AVPktMuxData pktDataCopy = pktData;
            pktDataCopy.muxAudio = &m_Mux.audio[i];
            auto err = WriteNextPacketInternal(&pktDataCopy, INT64_MAX);
            if (err != RGY_ERR_NONE) return err;
        }
        return RGY_ERR_NONE;
    }
    return WriteNextPacketInternal(&pktData, INT64_MAX);
}

//指定された音声キューに追加する
RGY_ERR RGYOutputAvcodec::AddAudQueue(AVPktMuxData *pktData, int type) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (m_Mux.thread.threadActiveAudioProcess()) {
        // もともとはこうだった
        // (type == AUD_QUEUE_OUT) ? m_Mux.thread.thOutput.get() : ((type == AUD_QUEUE_PROCESS) ? m_Mux.thread.qAudioPacketProcess : m_Mux.thread.qAudioFrameEncode;
        AVMuxThreadWorker *worker = getPacketWorker(pktData->muxAudio, type);

        //出力キューに追加する
        auto& qAudio       = worker->qPackets;
        auto& heEventAdded = worker->heEventPktAdded;
        if (!qAudio.push(*pktData)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for audio queue.\n"));
            m_Mux.format.streamError = true;
        }
        SetEvent(heEventAdded);
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    } else
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    {
        return RGY_ERR_NOT_INITIALIZED;
    }
}

//音声処理スレッドが存在する場合、この関数は音声処理スレッドによって処理される
//音声処理スレッドがなく、出力スレッドがあれば、出力スレッドにより処理される
//出力スレッドがなければメインエンコードスレッドが処理する
//maxDtsToWriteはm_AudPktBufFileHeadにキャッシュしてあるパケットを処理する際に、
//処理するdtsの上限を決める
RGY_ERR RGYOutputAvcodec::WriteNextPacketInternal(AVPktMuxData *pktData, int64_t maxDtsToWrite) {
    if (m_Mux.thread.threadActiveAudio()) {
        // 音声スレッドがある場合はフレームヘッダが書かれてからここに来るはず
        if (!m_Mux.format.fileHeaderWritten) {
            AddMessage(RGY_LOG_ERROR, _T("File header not written, unexpected error!\n"));
            return RGY_ERR_UNKNOWN;
        }
        // 音声スレッドがある場合はm_AudPktBufFileHeadはたまっていないはず
        if (!m_AudPktBufFileHead.empty()) {
            AddMessage(RGY_LOG_ERROR, _T("m_AudPktBufFileHead not empty, unexpected error!\n"));
            return RGY_ERR_UNKNOWN;
        }
    } else {
        if (!m_Mux.format.fileHeaderWritten) {
            //まだフレームヘッダーが書かれていなければ、パケットをキャッシュして終了
            m_AudPktBufFileHead.push_back(*pktData);
            return RGY_ERR_NONE;
        }
        //m_AudPktBufFileHeadにキャッシュしてあるパケットかどうかを調べる
        if (m_AudPktBufFileHead.end() == std::find_if(m_AudPktBufFileHead.begin(), m_AudPktBufFileHead.end(),
            [pktData](const AVPktMuxData& data) { return pktData->pkt == data.pkt; })) {
            //キャッシュしてあるパケットでないなら、キャッシュしてあるパケットをまず処理する
            for (auto bufPkt : m_AudPktBufFileHead) {
                RGY_ERR sts = WriteNextPacketInternal(&bufPkt, maxDtsToWrite);
                //処理するdtsの上限を超えたかチェック
                if (bufPkt.dts > maxDtsToWrite) {
                    pktData->dts = bufPkt.dts;
                    return RGY_ERR_NONE;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
            //キャッシュをすべて書き出したらクリア
            m_AudPktBufFileHead.clear();
        }
    }

    if (pktData->pkt == nullptr) {
        return WriteNextPacketAudio(pktData);
    } else if (trackMediaType(pktFlagGetTrackID(pktData->pkt)) != AVMEDIA_TYPE_AUDIO) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.threadActiveAudioProcess()) {
            //音声処理を別スレッドでやっている場合は、字幕パケットもその流れに乗せてやる必要がある
            //ひとまず、ここでは処理せず、次のキューに回す
            return AddAudQueue(pktData, (m_Mux.thread.threadActiveAudioEncode()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        return WriteOtherPacket(pktData->pkt);
    }
    return WriteNextPacketAudio(pktData);
}

//音声処理スレッドが存在する場合、この関数は音声処理スレッドによって処理される
//音声処理スレッドがなく、出力スレッドがあれば、出力スレッドにより処理される
//出力スレッドがなければメインエンコードスレッドが処理する
RGY_ERR RGYOutputAvcodec::WriteNextPacketAudio(AVPktMuxData *pktData) {
    pktData->samples = 0;
    AVMuxAudio *muxAudio = pktData->muxAudio;
    if (muxAudio == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get stream for input stream.\n"));
        m_Mux.format.streamError = true;
        m_Mux.poolPkt->returnFree(&pktData->pkt);
        return RGY_ERR_NULL_PTR;
    }

    if (pktData->pkt == nullptr || pktData->pkt->data == nullptr) {
        //デコーダをflush
        while (muxAudio->outCodecDecodeCtx && !muxAudio->encodeError) {
            auto decodedFrames = AudioDecodePacket(muxAudio, nullptr);
            if (decodedFrames.size() == 0) {
                break;
            }
            vector<AVPktMuxData> audioFrames;
            for (size_t i = 0; i < decodedFrames.size(); i++) {
                AVPktMuxData audPkt;
                audPkt.pkt = m_Mux.poolPkt->getFree().release();
                pktFlagSetTrackID(audPkt.pkt, muxAudio->inTrackId);
                audPkt.dts = AV_NOPTS_VALUE;
                audPkt.samples = 0;
                audPkt.type = MUX_DATA_TYPE_FRAME;
                audPkt.frame = decodedFrames[i].release();
                audPkt.got_result = audPkt.frame && audPkt.frame->nb_samples > 0;
                audPkt.muxAudio = muxAudio;
                audioFrames.push_back(audPkt);
            }
            //フィルタリングを行う
            WriteNextPacketToAudioSubtracks(std::move(audioFrames));
        }

        //終わったら後段にnull packetを渡してflushの指示を伝える
        //サブストリームが存在すれば、null packetをそれぞれに渡す
        AVMuxAudio *pMuxAudioSubStream = nullptr;
        for (int iSubStream = 1; nullptr != (pMuxAudioSubStream = getAudioStreamData(muxAudio->inTrackId, iSubStream)); iSubStream++) {
            auto pktDataCopy = *pktData;
            pktDataCopy.muxAudio = pMuxAudioSubStream;
            if (m_Mux.thread.threadActiveAudioProcess()) {
                AddAudQueue(&pktDataCopy, (m_Mux.thread.threadActiveAudioEncode()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
            } else {
                WriteNextAudioFrame(&pktDataCopy);
            }
        }
        //メインストリームにnull packetを渡してflushの指示を伝える
        if (m_Mux.thread.threadActiveAudioProcess()) {
            AddAudQueue(pktData, (m_Mux.thread.threadActiveAudioEncode()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        } else {
            WriteNextAudioFrame(pktData);
        }
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }

    //AACBsfでのエラーを無音挿入で回避する(音声エンコード時のみ)
    bool bSetSilenceDueToAACBsfError = false;
    AVRational samplerate = { 1, muxAudio->streamIn->codecpar->sample_rate };
    //このパケットのサンプル数
    const int nSamples = (int)av_rescale_q(pktData->pkt->duration, muxAudio->streamIn->time_base, samplerate);
    if (muxAudio->bsfc) {
        auto sts = applyBitstreamFilterAudio(pktData->pkt, muxAudio);
        //bitstream filterを正常に起動できなかった
        if (sts < RGY_ERR_NONE) {
            m_Mux.format.streamError = true;
            m_Mux.poolPkt->returnFree(&pktData->pkt);
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        //pktData->pkt.duration == 0 の場合はなにもせず終了する
        if (pktData->pkt->duration == 0) {
            m_Mux.poolPkt->returnFree(&pktData->pkt);
            //特にエラーでなければそのまま終了
            if (sts == RGY_ERR_NONE) {
                return RGY_ERR_NONE;
            }
            //音声エンコードしない場合はどうしようもないので終了
            if (!muxAudio->outCodecDecodeCtx || m_Mux.format.streamError) {
                return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
            }
            //音声エンコードの場合は無音挿入で時間をかせぐ
            bSetSilenceDueToAACBsfError = true;
        }
    }
    muxAudio->packetWritten++;
    auto writeOrSetNextPacketAudioProcessed = [this](AVPktMuxData *pktData) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.threadActiveAudioProcess()) {
            //ひとまず、ここでは処理せず、次のキューに回す
            AddAudQueue(pktData, (m_Mux.thread.threadActiveAudioEncode()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        } else {
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
            WriteNextPacketProcessed(pktData);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    };
    if (!muxAudio->outCodecDecodeCtx) {
        pktData->samples = av_get_audio_frame_duration2(muxAudio->streamIn->codecpar, pktData->pkt->size);
        if (!pktData->samples) {
            pktData->samples = (int)av_rescale_q(pktData->pkt->duration, muxAudio->streamIn->time_base, samplerate);
            // 1/1000 timebaseは信じるに値しないので、frame_sizeがあればその値を使用する
            if (0 == av_cmp_q(muxAudio->streamIn->time_base, { 1, 1000 })
                && muxAudio->streamIn->codecpar->frame_size) {
                pktData->samples = muxAudio->streamIn->codecpar->frame_size;
            } else {
                //このdurationから計算したsampleが信頼できるか計算する
                //mkvではたまにptsの差分とdurationが一致しないことがある
                //ptsDiffが動画の1フレーム分より小さいときのみ対象とする (カット編集によるものを混同する可能性がある)
                int64_t ptsDiff = pktData->pkt->pts - muxAudio->lastPtsIn;
                if (0 < ptsDiff
                    && ptsDiff < av_rescale_q(1, av_inv_q(m_Mux.video.outputFps), samplerate)
                    && muxAudio->lastPtsIn != AV_NOPTS_VALUE
                    && 1 < std::abs(ptsDiff - pktData->pkt->duration)) {
                    //ptsの差分から計算しなおす
                    pktData->samples = (int)av_rescale_q(ptsDiff, muxAudio->streamIn->time_base, samplerate);
                }
            }
        }
        muxAudio->lastPtsIn = pktData->pkt->pts;
        writeOrSetNextPacketAudioProcessed(pktData);
    } else if (!(muxAudio->decodeError > muxAudio->ignoreDecodeError) && !muxAudio->encodeError) {
        vector<AVPktMuxData> audioFrames;
        if (bSetSilenceDueToAACBsfError) {
            //無音挿入
            auto silentFrame = m_Mux.poolFrame->getFree();
            silentFrame->nb_samples     = nSamples;
#if AV_CHANNEL_LAYOUT_STRUCT_AVAIL
            av_channel_layout_copy(&silentFrame->ch_layout, &muxAudio->outCodecDecodeCtx->ch_layout);
            const int channel_count = getChannelCount(&silentFrame->ch_layout);
#else
            silentFrame->channels       = muxAudio->outCodecDecodeCtx->channels;
            silentFrame->channel_layout = muxAudio->outCodecDecodeCtx->channel_layout;
            const int channel_count     = muxAudio->outCodecDecodeCtx->channels;
#endif
            silentFrame->sample_rate    = muxAudio->outCodecDecodeCtx->sample_rate;
            silentFrame->format         = muxAudio->outCodecDecodeCtx->sample_fmt;
            silentFrame->pts            = pktData->pkt->pts;
            av_frame_get_buffer(silentFrame.get(), 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
            av_samples_set_silence((uint8_t **)silentFrame->data, 0, silentFrame->nb_samples, channel_count, (AVSampleFormat)silentFrame->format);

            AVPktMuxData silentPkt = *pktData;
            silentPkt.type = MUX_DATA_TYPE_FRAME;
            silentPkt.frame = silentFrame.release();
            silentPkt.got_result = silentFrame && silentFrame->nb_samples > 0;
            audioFrames.push_back(silentPkt);
        } else {
            auto decodedFrames = AudioDecodePacket(muxAudio, pktData->pkt);
            for (size_t i = 0; i < decodedFrames.size(); i++) {
                AVPktMuxData audPkt = *pktData;
                audPkt.type = MUX_DATA_TYPE_FRAME;
                audPkt.frame = decodedFrames[i].release();
                audPkt.samples = (audPkt.frame) ? audPkt.frame->nb_samples : 0;
                audPkt.got_result = audPkt.frame && audPkt.frame->nb_samples > 0;
                audioFrames.push_back(audPkt);
            }
        }
        WriteNextPacketToAudioSubtracks(std::move(audioFrames));
    }
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

//フィルタリング後のパケットをサブトラックに分配する
RGY_ERR RGYOutputAvcodec::WriteNextPacketToAudioSubtracks(vector<AVPktMuxData> audioFrames) {
    const auto origPkts = audioFrames.size();
    for (size_t i = 0; i < origPkts; i++) {
        //サブストリームが存在すれば、frameをコピーしてそれぞれに渡す
        AVMuxAudio *pMuxAudioSubStream = nullptr;
        for (int iSubStream = 1; nullptr != (pMuxAudioSubStream = getAudioStreamData(audioFrames[i].muxAudio->inTrackId, iSubStream)); iSubStream++) {
            auto pktDataCopy = audioFrames[i];
            pktDataCopy.muxAudio = pMuxAudioSubStream;
            pktDataCopy.frame = (audioFrames[i].frame) ? av_frame_clone(audioFrames[i].frame) : nullptr;
            audioFrames.push_back(pktDataCopy);
        }
    }
    return WriteNextPacketAudioFrame(AudioFilterFrame(audioFrames));
}

//フレームをresampleして後段に渡す
RGY_ERR RGYOutputAvcodec::WriteNextPacketAudioFrame(vector<AVPktMuxData> audioFrames) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    const bool bAudEncThread = m_Mux.thread.threadActiveAudioEncode();
#else
    const bool bAudEncThread = false;
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD

    for (auto& pktData : audioFrames) {
        if (pktData.got_result) {
            if (bAudEncThread) {
                AddAudQueue(&pktData, AUD_QUEUE_ENCODE);
            } else {
                WriteNextAudioFrame(&pktData);
            }
        }
    }
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

//音声フレームをエンコード
//音声エンコードスレッドが存在する場合、この関数は音声エンコードスレッドによって処理される
//音声エンコードスレッドが存在せず、音声処理スレッドが存在する場合、この関数は音声処理スレッドによって処理される
//音声処理スレッドが存在しない場合、この関数は出力スレッドによって処理される
//出力スレッドがなければメインエンコードスレッドが処理する
RGY_ERR RGYOutputAvcodec::WriteNextAudioFrame(AVPktMuxData *pktData) {
    if (pktData->type != MUX_DATA_TYPE_FRAME) {
        if (pktData->muxAudio) {
            //音声エンコードスレッドがこの関数を処理
            if (pktData->pkt == nullptr) {
                //null終端パケットなどが流れてきたもの
                while (pktData->muxAudio->outCodecEncodeCtx) {
                    auto encPktDatas = AudioEncodeFrame(pktData->muxAudio, nullptr);
                    if (encPktDatas.size() == 0) {
                        break;
                    }
                    if (pktData->muxAudio->decodeError > pktData->muxAudio->ignoreDecodeError)
                        break;
                    if (m_Mux.thread.threadActiveAudioProcess()) {
                        for (auto& pktMux : encPktDatas) {
                            AddAudQueue(&pktMux, AUD_QUEUE_OUT);
                        }
                    } else {
                        for (auto& pktMux : encPktDatas) {
                            WriteNextPacketProcessed(&pktMux, &pktData->dts);
                        }
                    }
                }
                if (m_Mux.thread.threadActiveAudioProcess()) {
                    //終わったら後段にflushの指示を伝える
                    AddAudQueue(pktData, AUD_QUEUE_OUT);
                } else {
                    WriteNextPacketProcessed(pktData);
                }
            } else {
                //音声コピーなどの際にパケットが流れてきたもの
                if (m_Mux.thread.threadActiveAudioProcess()) {
                    //終わったら後段にflushの指示を伝える
                    AddAudQueue(pktData, AUD_QUEUE_OUT);
                } else {
                    WriteNextPacketProcessed(pktData);
                }
            }
        } else if (m_Mux.thread.thOutput) {
            //AVPacketは字幕が流れてきたもの
            //これはそのまま出力キューに追加する
            AddAudQueue(pktData, AUD_QUEUE_OUT);
        } else {
            // ここには来ないはず
            AddMessage(RGY_LOG_ERROR, _T("Unexpected non-audio packet!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        return RGY_ERR_NONE;
    }
    auto encPktDatas = AudioEncodeFrame(pktData->muxAudio, pktData->frame);
    m_Mux.poolFrame->returnFree(&pktData->frame);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (m_Mux.thread.threadActiveAudioProcess()) {
        for (auto& pktMux : encPktDatas) {
            AddAudQueue(&pktMux, AUD_QUEUE_OUT);
        }
    } else {
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        for (auto& pktMux : encPktDatas) {
            WriteNextPacketProcessed(&pktMux);
        }
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::ThreadFuncAudEncodeThread(const AVMuxAudio *const muxAudio, RGYParamThread threadParam) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    threadParam.apply(GetCurrentThread());
    auto worker = getPacketWorker(muxAudio, AUD_QUEUE_ENCODE);
    WaitForSingleObject(worker->heEventPktAdded, INFINITE);
    while (!worker->thAbort) {
        if (!m_Mux.format.fileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (worker->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_enc : nullptr)) {
                //音声エンコードを実行、出力キューに追加する
                WriteNextAudioFrame(&pktData);
            }
        }
        if (m_Mux.format.lowlatency) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            ResetEvent(worker->heEventPktAdded);
            WaitForSingleObject(worker->heEventPktAdded, 16);
        }
    }
    {   //音声をすべてエンコード
        AVPktMuxData pktData = { 0 };
        while (worker->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_enc : nullptr)) {
            WriteNextAudioFrame(&pktData);
        }
    }
    SetEvent(worker->heEventClosing);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::ThreadFuncAudThread(const AVMuxAudio *const muxAudio, RGYParamThread threadParam) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    threadParam.apply(GetCurrentThread());
    auto worker = getPacketWorker(muxAudio, AUD_QUEUE_PROCESS);
    WaitForSingleObject(worker->heEventPktAdded, INFINITE);
    while (!worker->thAbort) {
        if (!m_Mux.format.fileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (worker->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_proc : nullptr)) {
                //音声処理を実行、出力キューに追加する
                WriteNextPacketInternal(&pktData, INT64_MAX);
            }
        }
        if (m_Mux.format.lowlatency) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            ResetEvent(worker->heEventPktAdded);
            WaitForSingleObject(worker->heEventPktAdded, 16);
        }
    }
    {   //音声をすべて書き出す
        AVPktMuxData pktData = { 0 };
        while (worker->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_proc : nullptr)) {
            //音声処理を実行、出力キューに追加する
            WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    SetEvent(worker->heEventClosing);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteThreadFuncRawVideo(RGYParamThread threadParam) {
    threadParam.apply(GetCurrentThread());
    while (!m_Mux.thread.thRawVideo->thAbort) {
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.thRawVideo->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr)) {
            VideoEncodeRawFrame(pktData.frame);
        }
    }
    SetEvent(m_Mux.thread.thRawVideo->heEventClosing);
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteThreadFunc(RGYParamThread threadParam) {
#if ENABLE_AVCODEC_OUT_THREAD
    threadParam.apply(GetCurrentThread());
    //映像と音声の同期をとる際に、それをあきらめるまでの閾値
    const int nWaitThreshold = 32;
    //キューにデータが存在するか
    bool bAudioExists = false;
    bool bVideoExists = false;
    const bool videoIsRaw = m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_RAWVIDEO;
    const auto fpsTimebase = av_inv_q(m_Mux.video.outputFps);
    const int VideoAudioPickSwitchThresholdFrames = m_Mux.format.lowlatency ? 1 : 2; // 映像-音声の切り替え間隔(フレーム数)
    const auto dtsThreshold = std::max<int64_t>(av_rescale_q(VideoAudioPickSwitchThresholdFrames, fpsTimebase, QUEUE_DTS_TIMEBASE), 4);
    //syncIgnoreDtsは映像と音声の同期を行う必要がないことを意味する
    //dtsThresholdを加算したときにオーバーフローしないよう、dtsThresholdを引いておく
    const int64_t syncIgnoreDts = INT64_MAX - dtsThreshold;
    int64_t audioDts = (m_Mux.audio.size() + m_Mux.other.size()) ? 0 : syncIgnoreDts;
    int64_t videoDts = (m_Mux.video.streamOut) ? 0 : syncIgnoreDts;
    auto writeProcessedPacket = [this](AVPktMuxData *pktData) {
        //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
        auto sts = RGY_ERR_NONE;
        if (pktData->pkt == nullptr) { // フラッシュ用
            WriteNextPacketProcessed(pktData);
        } else {
            if (trackMediaType(pktFlagGetTrackID(pktData->pkt)) == AVMEDIA_TYPE_AUDIO) {
                WriteNextPacketProcessed(pktData);
            } else {
                sts = WriteOtherPacket(pktData->pkt);
            }
        }
        return sts;
    };
    const bool MUX_DEBUG = false;
    std::unique_ptr<FILE, fp_deleter> fpMuxDebug;
    if (MUX_DEBUG) {
        fpMuxDebug = std::unique_ptr<FILE, fp_deleter>(_tfopen(_T("mux_debug.txt"), _T("w")));
        fprintf(fpMuxDebug.get(), "mux_debug\n");
        fprintf(fpMuxDebug.get(), "dtsThreshold : %20lld\n", (lls)dtsThreshold);
        fprintf(fpMuxDebug.get(), "syncIgnoreDts: %20lld\n", (lls)syncIgnoreDts);
        fprintf(fpMuxDebug.get(), "audioDts     : %20lld\n", (lls)audioDts);
        fprintf(fpMuxDebug.get(), "videoDts     : %20lld\n", (lls)videoDts);
        fprintf(fpMuxDebug.get(), "\n");
    }
    bool bThAudProcess = false;
    int audPacketsPerSec = 64;
    int nWaitAudio = 0;
    int nWaitVideo = 0;
    while (!m_Mux.thread.thOutput->thAbort) {
        // 起動遅れの場合がありえるのでここでチェック
        if (!bThAudProcess && m_Mux.thread.threadActiveAudioProcess()) {
            bThAudProcess = true;
        }
        do {
            if (!m_Mux.format.fileHeaderWritten) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //ヘッダー取得前に音声キューのサイズが足りず、エンコードが進まなくなってしまうことがある
                //キューのcapcityを増やすことでこれを回避する
                if (!m_Mux.thread.threadActiveAudioProcess()) {
                    auto worker = m_Mux.thread.thOutput.get();
                    auto& qAudio = worker->qPackets;
                    const auto nQueueCapacity = qAudio.capacity();
                    if (qAudio.size() >= nQueueCapacity) {
                        qAudio.set_capacity(nQueueCapacity * 3 / 2);
                    }
                } else {
                    for (auto& [mux, thAud] : m_Mux.thread.thAud) {
                        auto worker = getPacketWorker(mux, AUD_QUEUE_PROCESS);
                        auto& qAudio = worker->qPackets;
                        const auto nQueueCapacity = qAudio.capacity();
                        if (qAudio.size() >= nQueueCapacity) {
                            qAudio.set_capacity(nQueueCapacity * 3 / 2);
                        }
                    }
                }
                //動画キューになにもなかったら再度待機する
                if ((!videoIsRaw && m_Mux.thread.qVideobitstream.size() == 0) ||
                    (videoIsRaw && m_Mux.thread.qVideoRawFrames.size() == 0)) {
                    break;
                }
            }

            //映像・音声の同期待ちが必要な場合、falseとなってループから抜けるよう、ここでfalseに設定する
            bAudioExists = false;
            bVideoExists = false;
            if (videoIsRaw) {
                AVPktMuxData pktData = { 0 };
                while ((audioDts < 0 || videoDts <= audioDts + dtsThreshold)
                    && false != (bVideoExists = m_Mux.thread.qVideoRawFrames.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
                    WriteNextPacketRawVideo(pktData.pkt, &videoDts);
                    nWaitVideo = 0;
                    const auto log_level = RGY_LOG_TRACE;
                    if (m_printMes && log_level >= m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                        AddMessage(log_level, _T("videoDts=%8lld: %s.\n"), (lls)videoDts, getTimestampString(videoDts, QUEUE_DTS_TIMEBASE).c_str());
                    }
                    if (fpMuxDebug) fprintf(fpMuxDebug.get(), "video: v %3d, a %3d, [videoDts=%16lld],  audioDts=%16lld .\n", (int)m_Mux.thread.qVideoRawFrames.size(), (int)m_Mux.thread.thOutput->qPackets.size(), (lls)videoDts, (lls)audioDts);
                }
            } else {
                RGYBitstream bitstream = RGYBitstreamInit();
                while ((audioDts < 0 || videoDts <= audioDts + dtsThreshold)
                    && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
                    WriteNextFrameInternal(&bitstream, &videoDts);
                    nWaitVideo = 0;
                    const auto log_level = RGY_LOG_TRACE;
                    if (m_printMes && log_level >= m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                        AddMessage(log_level, _T("videoDts=%8lld: %s.\n"), (lls)videoDts, getTimestampString(videoDts, QUEUE_DTS_TIMEBASE).c_str());
                    }
                    if (fpMuxDebug) fprintf(fpMuxDebug.get(), "video: v %3d, a %3d, [videoDts=%16lld],  audioDts=%16lld .\n", (int)m_Mux.thread.qVideobitstream.size(), (int)m_Mux.thread.thOutput->qPackets.size(), (lls)videoDts, (lls)audioDts);
                }
            }
            AVPktMuxData pktData = { 0 };
            while ((videoDts < 0 || audioDts <= videoDts + dtsThreshold)
                && false != (bAudioExists = m_Mux.thread.thOutput->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr))) {
                bool isAudio = false;
                if (pktData.muxAudio && pktData.muxAudio->streamIn && pktData.pkt) {
                    audPacketsPerSec = (pktData.pkt->duration <= 0) ? pktData.muxAudio->streamIn->codecpar->sample_rate * 8 : std::max(audPacketsPerSec, (int)(1.0 / (av_q2d(pktData.muxAudio->streamIn->time_base) * pktData.pkt->duration) + 0.5));
                    const auto videoDelay = (audioDts - videoDts) * av_q2d(QUEUE_DTS_TIMEBASE);
                    const auto streamQueueCapacity = (int)(audPacketsPerSec * std::max(5.0, videoDelay * 1.5) * std::max((int)m_Mux.audio.size(), 1) + 0.5);
                    if ((int)m_Mux.thread.thOutput->qPackets.capacity() < streamQueueCapacity) {
                        m_Mux.thread.thOutput->qPackets.set_capacity(streamQueueCapacity);
                    }
                    isAudio = pktData.muxAudio->streamOut->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
                }
                const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : syncIgnoreDts;
                //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
                (m_Mux.thread.threadActiveAudioProcess()) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
                // 字幕やデータストリームに関しては、連続で来るとは限らないので、考慮しないことにする
                if (isAudio
                    && pktData.dts != AV_NOPTS_VALUE && pktData.dts != (int64_t)((uint64_t)AV_NOPTS_VALUE - 1)) {
                    // 複数のトラックがあり得るので、最大値をとる
                    audioDts = (std::max)(audioDts, (std::max)(pktData.dts, m_Mux.thread.streamOutMaxDts.load()));
                }
                nWaitAudio = 0;
                const auto log_level = RGY_LOG_TRACE;
                if (m_printMes && log_level >= m_printMes->getLogLevel(RGY_LOGT_OUT)) {
                    AddMessage(log_level, _T("audioDts=%8lld: %s, maxDst=%8lld.\n"), (lls)audioDts, getTimestampString(audioDts, QUEUE_DTS_TIMEBASE).c_str(), (lls)maxDts);
                }
                if (fpMuxDebug) fprintf(fpMuxDebug.get(), "audio: v %3d, a %3d,  videoDts=%16lld , [audioDts=%16lld].\n",
                    (videoIsRaw) ? (int)m_Mux.thread.qVideoRawFrames.size() : (int)m_Mux.thread.qVideobitstream.size(),
                    (int)m_Mux.thread.thOutput->qPackets.size(), (lls)videoDts, (lls)audioDts);
            }
            //一定以上の動画フレームがキューにたまっており、音声キューになにもなければ、
            //音声を無視して動画フレームの処理を開始させる
            //音声が途中までしかなかったり、途中からしかなかったりする場合にこうした処理が必要
            const size_t videoPacketThreshold = std::max<size_t>(std::min<size_t>(3072, (videoIsRaw) ? (int)m_Mux.thread.qVideoRawFrames.capacity() : (int)m_Mux.thread.qVideobitstream.capacity()), nWaitThreshold) - nWaitThreshold;
            auto vidQueueSize = (videoIsRaw) ? (int)m_Mux.thread.qVideoRawFrames.size() : (int)m_Mux.thread.qVideobitstream.size();
            if (m_Mux.thread.thOutput->qPackets.size() == 0 && vidQueueSize > videoPacketThreshold) {
                nWaitAudio++;
                if (nWaitAudio <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    //また、映像キューのサイズが足りないことが考えられるので、拡大する
                    if (videoIsRaw) {
                        m_Mux.thread.qVideoRawFrames.set_capacity(m_Mux.thread.qVideoRawFrames.capacity() + 50);
                    } else {
                        m_Mux.thread.qVideobitstream.set_capacity(m_Mux.thread.qVideobitstream.capacity() + 50);
                    }
                    break;
                }
                audioDts = videoDts;
                AddMessage(RGY_LOG_TRACE, _T("audio not coming: %d.\n"), nWaitAudio);
            }
            //一定以上の音声フレームがキューにたまっており、動画キューになにもなければ、
            //動画を無視して音声フレームの処理を開始させる
            const size_t audioPacketThreshold = std::max<size_t>(std::min<size_t>(10 * 1024 * 1024, m_Mux.thread.thOutput->qPackets.capacity()), nWaitThreshold) - nWaitThreshold;
            vidQueueSize = (videoIsRaw) ? (int)m_Mux.thread.qVideoRawFrames.size() : (int)m_Mux.thread.qVideobitstream.size();
            if (vidQueueSize == 0 && m_Mux.thread.thOutput->qPackets.size() > audioPacketThreshold) {
                nWaitVideo++;
                if (nWaitVideo <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    //また、音声キューのサイズが足りないことが考えられるので、拡大する
                    m_Mux.thread.thOutput->qPackets.set_capacity(m_Mux.thread.thOutput->qPackets.capacity() * 3 / 2);
                    break;
                }
                videoDts = audioDts;
                AddMessage(RGY_LOG_TRACE, _T("video not coming: %d\n"), nWaitVideo);
            }
        } while (bAudioExists || bVideoExists); //両方のキューがひとまず空になるか、映像・音声の同期待ちが必要になるまで回す
                                                //次のフレーム・パケットが送られてくるまで待機する
        //キューの容量が両方とも半分以下なら、すこし待ってみる
        //一方、どちらかのキューが半分以上使われていれば、なるべく早く処理する必要がある
        if (   m_Mux.thread.qVideobitstream.size() / (double)m_Mux.thread.qVideobitstream.capacity() < 0.5
            && m_Mux.thread.thOutput->qPackets.size() / (double)m_Mux.thread.thOutput->qPackets.capacity() < 0.5) {
            if (m_Mux.format.lowlatency) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                ResetEvent(m_Mux.thread.thOutput->heEventPktAdded);
                WaitForSingleObject(m_Mux.thread.thOutput->heEventPktAdded, 16);
            }
        } else {
            std::this_thread::yield();
        }
    }
    //メインループを抜けたことを通知する
    SetEvent(m_Mux.thread.thOutput->heEventClosing);
    m_Mux.thread.thOutput->qPackets.set_keep_length(0);
    m_Mux.thread.qVideobitstream.set_keep_length(0);
    bAudioExists = !m_Mux.thread.thOutput->qPackets.empty();
    bVideoExists = !m_Mux.thread.qVideobitstream.empty();
    //まずは映像と音声の同期をとって出力するためのループ
    while (bAudioExists && bVideoExists) {
        AVPktMuxData pktData = { 0 };
        while (audioDts <= videoDts + dtsThreshold
            && false != (bAudioExists = m_Mux.thread.thOutput->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr))) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : INT64_MAX;
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
            //複数のstreamがあり得るので最大値をとる
            if (pktData.dts != AV_NOPTS_VALUE && pktData.dts != (int64_t)((uint64_t)AV_NOPTS_VALUE - 1)) {
                audioDts = (std::max)(audioDts, pktData.dts);
            }
        }
        if (videoIsRaw) {
            pktData = AVPktMuxData();
            while (videoDts <= audioDts + dtsThreshold
                && false != (bVideoExists = m_Mux.thread.qVideoRawFrames.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
                WriteNextPacketRawVideo(pktData.pkt, &videoDts);
            }
        } else {
            RGYBitstream bitstream = RGYBitstreamInit();
            while (videoDts <= audioDts + dtsThreshold
                && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
                WriteNextFrameInternal(&bitstream, &videoDts);
            }
            bVideoExists = !m_Mux.thread.qVideobitstream.empty();
        }
        bAudioExists = !m_Mux.thread.thOutput->qPackets.empty();
    }
    { //音声を書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.thOutput->qPackets.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr)) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    if (videoIsRaw) { //動画を書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qVideoRawFrames.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr)) {
            WriteNextPacketRawVideo(pktData.pkt, &videoDts);
        }
        //nullptrを送って終了を通知する
        WriteNextPacketRawVideo(nullptr, &videoDts);
    } else {
        RGYBitstream bitstream = RGYBitstreamInit();
        while (m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr)) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
        //空のbitstreamを送って終了を通知する
        bitstream = RGYBitstreamInit();
        WriteNextFrameInternal(&bitstream, &videoDts);
    }
#endif
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

void RGYOutputAvcodec::WaitFin() {
    CloseThread();
}

HANDLE RGYOutputAvcodec::getThreadHandleOutput() {
#if ENABLE_AVCODEC_OUT_THREAD
    return (m_Mux.thread.thOutput) ? (HANDLE)m_Mux.thread.thOutput->thread.native_handle() : nullptr;
#else
    return NULL;
#endif
}

HANDLE RGYOutputAvcodec::getThreadHandleAudProcess() {
#if ENABLE_AVCODEC_OUT_THREAD && ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.thread.threadActiveAudioProcess()) ? (HANDLE)m_Mux.thread.thAud[nullptr]->process.thread.native_handle() : nullptr;
#else
    return NULL;
#endif
}

HANDLE RGYOutputAvcodec::getThreadHandleAudEncode() {
#if ENABLE_AVCODEC_OUT_THREAD && ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.thread.threadActiveAudioEncode()) ? (HANDLE)m_Mux.thread.thAud[nullptr]->encode.thread.native_handle() : nullptr;
#else
    return NULL;
#endif
}

#if USE_CUSTOM_IO
int RGYOutputAvcodec::readPacket(uint8_t *buf, int buf_size) {
    return (int)_fread_nolock(buf, 1, buf_size, m_Mux.format.fpOutput);
}
int RGYOutputAvcodec::writePacket(const uint8_t *buf, int buf_size) {
    int res = (int)_fwrite_nolock(buf, 1, buf_size, m_Mux.format.fpOutput);
    if (res < buf_size) {
        AddMessage(RGY_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\""));
        m_Mux.format.streamError = true;
    }
    return res;
}
int64_t RGYOutputAvcodec::seek(int64_t offset, int whence) {
    return _fseeki64(m_Mux.format.fpOutput, offset, whence);
}
#endif //USE_CUSTOM_IO

#endif //ENABLE_AVSW_READER
