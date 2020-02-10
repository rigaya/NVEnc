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

#include <fcntl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_output_avcodec.h"
#include "rgy_avlog.h"
#include "rgy_bitstream.h"

#if ENABLE_AVSW_READER
#if USE_CUSTOM_IO
static int funcReadPacket(void *opaque, uint8_t *buf, int buf_size) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->readPacket(buf, buf_size);
}
static int funcWritePacket(void *opaque, uint8_t *buf, int buf_size) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->writePacket(buf, buf_size);
}
static int64_t funcSeek(void *opaque, int64_t offset, int whence) {
    RGYOutputAvcodec *writer = reinterpret_cast<RGYOutputAvcodec *>(opaque);
    return writer->seek(offset, whence);
}
#endif //USE_CUSTOM_IO

const AVRational RGYOutputAvcodec::QUEUE_DTS_TIMEBASE = av_make_q(1, 90000);

RGYOutputAvcodec::RGYOutputAvcodec() {
    memset(&m_Mux.format, 0, sizeof(m_Mux.format));
    memset(&m_Mux.video,  0, sizeof(m_Mux.video));
    m_strWriterName = _T("avout");
}

RGYOutputAvcodec::~RGYOutputAvcodec() {
    Close();
}

void RGYOutputAvcodec::CloseOther(AVMuxOther *muxOther) {
    //close decoder
    if (muxOther->outCodecDecodeCtx) {
        avcodec_close(muxOther->outCodecDecodeCtx);
        av_free(muxOther->outCodecDecodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecDecodeCtx.\n"));
    }

    //close encoder
    if (muxOther->outCodecEncodeCtx) {
        avcodec_close(muxOther->outCodecEncodeCtx);
        av_free(muxOther->outCodecEncodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecEncodeCtx.\n"));
    }
    if (muxOther->bufConvert) {
        av_free(muxOther->bufConvert);
    }

    memset(muxOther, 0, sizeof(muxOther[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed other.\n"));
}

void RGYOutputAvcodec::CloseAudio(AVMuxAudio *muxAudio) {
    //close decoder
    if (muxAudio->outCodecDecodeCtx
        && muxAudio->inSubStream == 0) { //サブストリームのものは単なるコピーなので開放不要
        avcodec_close(muxAudio->outCodecDecodeCtx);
        av_free(muxAudio->outCodecDecodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecDecodeCtx.\n"));
    }

    //close encoder
    if (muxAudio->outCodecEncodeCtx) {
        avcodec_close(muxAudio->outCodecEncodeCtx);
        av_free(muxAudio->outCodecEncodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed outCodecEncodeCtx.\n"));
    }

    //close filter
    if (muxAudio->filterGraph) {
        avfilter_graph_free(&muxAudio->filterGraph);
    }

    if (muxAudio->AACBsfc) {
        av_bsf_free(&muxAudio->AACBsfc);
    }
    memset(muxAudio, 0, sizeof(muxAudio[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed audio.\n"));
}

void RGYOutputAvcodec::CloseVideo(AVMuxVideo *muxVideo) {
#if ENCODER_VCEENC
    if (muxVideo->parserCtx) {
        av_parser_close(muxVideo->parserCtx);
    }
#endif //#if ENCODER_VCEENC
    if (m_Mux.video.fpTsLogFile) {
        fclose(m_Mux.video.fpTsLogFile);
    }
    m_Mux.video.timestampList.clear();
    if (m_Mux.video.bsfc) {
        av_bsf_free(&m_Mux.video.bsfc);
    }
    memset(muxVideo, 0, sizeof(muxVideo[0]));
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
        AddMessage(RGY_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
#if USE_CUSTOM_IO
    if (muxFormat->fpOutput) {
        fflush(muxFormat->fpOutput);
        fclose(muxFormat->fpOutput);
        AddMessage(RGY_LOG_DEBUG, _T("Closed File Pointer.\n"));
    }

    if (muxFormat->AVOutBuffer) {
        av_free(muxFormat->AVOutBuffer);
    }

    if (muxFormat->outputBuffer) {
        free(muxFormat->outputBuffer);
    }
#endif //USE_CUSTOM_IO
    memset(muxFormat, 0, sizeof(muxFormat[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed format.\n"));
}

void RGYOutputAvcodec::CloseQueues() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.thAudEncodeAbort = true;
    m_Mux.thread.thAudProcessAbort = true;
    m_Mux.thread.abortOutput = true;
    m_Mux.thread.qVideobitstream.close();
    m_Mux.thread.qVideobitstreamFreeI.close([](RGYBitstream *pBitstream) { pBitstream->clear(); });
    m_Mux.thread.qVideobitstreamFreePB.close([](RGYBitstream *pBitstream) { pBitstream->clear(); });
    m_Mux.thread.qAudioPacketOut.close();
    m_Mux.thread.qAudioFrameEncode.close();
    m_Mux.thread.qAudioPacketProcess.close();
    AddMessage(RGY_LOG_DEBUG, _T("closed queues...\n"));
#endif
}

void RGYOutputAvcodec::CloseThread() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.thAudEncodeAbort = true;
    if (m_Mux.thread.thAudEncode.joinable()) {
        //下記同様に、m_Mux.thread.heEventThOutputClosingがセットされるまで、
        //SetEvent(m_Mux.thread.heEventThOutputPktAdded)を実行し続ける必要がある。
        while (WAIT_TIMEOUT == WaitForSingleObject(m_Mux.thread.heEventClosingAudEncode, 100)) {
            SetEvent(m_Mux.thread.heEventPktAddedAudEncode);
        }
        m_Mux.thread.thAudEncode.join();
        CloseEvent(m_Mux.thread.heEventPktAddedAudEncode);
        CloseEvent(m_Mux.thread.heEventClosingAudEncode);
        AddMessage(RGY_LOG_DEBUG, _T("closed audio encode thread...\n"));
    }
    m_Mux.thread.thAudProcessAbort = true;
    if (m_Mux.thread.thAudProcess.joinable()) {
        //下記同様に、m_Mux.thread.heEventThOutputClosingがセットされるまで、
        //SetEvent(m_Mux.thread.heEventThOutputPktAdded)を実行し続ける必要がある。
        while (WAIT_TIMEOUT == WaitForSingleObject(m_Mux.thread.heEventClosingAudProcess, 100)) {
            SetEvent(m_Mux.thread.heEventPktAddedAudProcess);
        }
        m_Mux.thread.thAudProcess.join();
        CloseEvent(m_Mux.thread.heEventPktAddedAudProcess);
        CloseEvent(m_Mux.thread.heEventClosingAudProcess);
        AddMessage(RGY_LOG_DEBUG, _T("closed audio process thread...\n"));
    }
    m_Mux.thread.abortOutput = true;
    if (m_Mux.thread.thOutput.joinable()) {
        //ここに来た時に、まだメインスレッドがループ中の可能性がある
        //その場合、SetEvent(m_Mux.thread.heEventPktAddedOutput)を一度やるだけだと、
        //そのあとにResetEvent(m_Mux.thread.heEventPktAddedOutput)が発生してしまい、
        //ここでスレッドが停止してしまう。
        //これを回避するため、m_Mux.thread.heEventClosingOutputがセットされるまで、
        //SetEvent(m_Mux.thread.heEventPktAddedOutput)を実行し続ける必要がある。
        while (WAIT_TIMEOUT == WaitForSingleObject(m_Mux.thread.heEventClosingOutput, 100)) {
            SetEvent(m_Mux.thread.heEventPktAddedOutput);
        }
        m_Mux.thread.thOutput.join();
        CloseEvent(m_Mux.thread.heEventPktAddedOutput);
        CloseEvent(m_Mux.thread.heEventClosingOutput);
        AddMessage(RGY_LOG_DEBUG, _T("closed output thread...\n"));
    }
    CloseQueues();
    m_Mux.thread.abortOutput = false;
    m_Mux.thread.thAudProcessAbort = false;
    m_Mux.thread.thAudEncodeAbort = false;
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
uint64_t RGYOutputAvcodec::AutoSelectChannelLayout(const uint64_t *channelLayout, const AVCodecContext *srcAudioCtx) {
    int srcChannels = av_get_channel_layout_nb_channels(srcAudioCtx->channel_layout);
    if (srcChannels == 0) {
        srcChannels = srcAudioCtx->channels;
    }
    if (channelLayout == nullptr) {
        switch (srcChannels) {
        case 1:  return AV_CH_LAYOUT_MONO;
        case 2:  return AV_CH_LAYOUT_STEREO;
        case 3:  return AV_CH_LAYOUT_2_1;
        case 4:  return AV_CH_LAYOUT_QUAD;
        case 5:  return AV_CH_LAYOUT_5POINT0;
        case 6:  return AV_CH_LAYOUT_5POINT1;
        case 7:  return AV_CH_LAYOUT_6POINT1;
        case 8:  return AV_CH_LAYOUT_7POINT1;
        default: return AV_CH_LAYOUT_NATIVE;
        }
    }

    for (int i = 0; channelLayout[i]; i++) {
        if (srcChannels == av_get_channel_layout_nb_channels(channelLayout[i])) {
            return channelLayout[i];
        }
    }
    return channelLayout[0];
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
    AVSampleFormat srcFormat = srcAudioCtx->sample_fmt;
    if (samplefmtList == nullptr) {
        return srcFormat;
    }
    if (srcFormat == AV_SAMPLE_FMT_NONE) {
        return samplefmtList[0];
    }
    for (int i = 0; samplefmtList[i] >= 0; i++) {
        if (srcFormat == samplefmtList[i]) {
            return samplefmtList[i];
        }
    }
    static const auto sampleFmtLevel = make_array<std::pair<AVSampleFormat, int>>(
        std::make_pair(AV_SAMPLE_FMT_DBLP, 8),
        std::make_pair(AV_SAMPLE_FMT_DBL,  8),
        std::make_pair(AV_SAMPLE_FMT_FLTP, 6),
        std::make_pair(AV_SAMPLE_FMT_FLT,  6),
        std::make_pair(AV_SAMPLE_FMT_S32P, 4),
        std::make_pair(AV_SAMPLE_FMT_S32,  4),
        std::make_pair(AV_SAMPLE_FMT_S16P, 2),
        std::make_pair(AV_SAMPLE_FMT_S16,  2),
        std::make_pair(AV_SAMPLE_FMT_U8P,  1),
        std::make_pair(AV_SAMPLE_FMT_U8,   1)
    );
    int srcFormatLevel = std::find_if(sampleFmtLevel.begin(), sampleFmtLevel.end(),
        [srcFormat](const std::pair<AVSampleFormat, int>& targetFormat) { return targetFormat.first == srcFormat;})->second;
    auto foundFormat = std::find_if(sampleFmtLevel.begin(), sampleFmtLevel.end(),
        [srcFormatLevel](const std::pair<AVSampleFormat, int>& targetFormat) { return targetFormat.second == srcFormatLevel; });
    for (; foundFormat != sampleFmtLevel.end(); foundFormat++) {
        for (int i = 0; samplefmtList[i] >= 0; i++) {
            if (foundFormat->first == samplefmtList[i]) {
                return samplefmtList[i];
            }
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
            if (stricmp(avprofile->name, codec_profile.c_str()) == 0) {
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

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputAvcodec::InitVideo(const VideoInfo *videoOutputInfo, const AvcodecWriterPrm *prm) {
    m_Mux.format.formatCtx->video_codec_id = getAVCodecId(videoOutputInfo->codec);
    if (m_Mux.format.formatCtx->video_codec_id == AV_CODEC_ID_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find codec id for video.\n"));
        return RGY_ERR_INVALID_CODEC;
    }
    m_Mux.format.formatCtx->oformat->video_codec = m_Mux.format.formatCtx->video_codec_id;
    if (NULL == (m_Mux.video.codec = avcodec_find_decoder(m_Mux.format.formatCtx->video_codec_id))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to codec for video codec %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_INVALID_CODEC;
    }
    if (NULL == (m_Mux.video.streamOut = avformat_new_stream(m_Mux.format.formatCtx, m_Mux.video.codec))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for video.\n"));
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.video.outputFps = av_make_q(videoOutputInfo->fpsN, videoOutputInfo->fpsD);
    AddMessage(RGY_LOG_DEBUG, _T("output video stream fps: %d/%d\n"), m_Mux.video.outputFps.num, m_Mux.video.outputFps.den);

    m_Mux.video.streamOut->codecpar->codec_type              = AVMEDIA_TYPE_VIDEO;
    m_Mux.video.streamOut->codecpar->codec_id                = m_Mux.format.formatCtx->video_codec_id;
    m_Mux.video.streamOut->codecpar->width                   = videoOutputInfo->dstWidth;
    m_Mux.video.streamOut->codecpar->height                  = videoOutputInfo->dstHeight;
    m_Mux.video.streamOut->codecpar->format                  = csp_rgy_to_avpixfmt(videoOutputInfo->csp);
    m_Mux.video.streamOut->codecpar->level                   = videoOutputInfo->codecLevel;
    m_Mux.video.streamOut->codecpar->profile                 = videoOutputInfo->codecProfile;
    m_Mux.video.streamOut->codecpar->sample_aspect_ratio.num = videoOutputInfo->sar[0];
    m_Mux.video.streamOut->codecpar->sample_aspect_ratio.den = videoOutputInfo->sar[1];
    m_Mux.video.streamOut->codecpar->chroma_location         = (AVChromaLocation)clamp(videoOutputInfo->vui.chromaloc, 0, 6);
    m_Mux.video.streamOut->codecpar->field_order             = picstrcut_rgy_to_avfieldorder(videoOutputInfo->picstruct);
    m_Mux.video.streamOut->codecpar->video_delay             = videoOutputInfo->videoDelay;
    m_Mux.video.streamOut->sample_aspect_ratio.num           = videoOutputInfo->sar[0]; //mkvではこちらの指定も必要
    m_Mux.video.streamOut->sample_aspect_ratio.den           = videoOutputInfo->sar[1];
    if (prm->videoCodecTag.length() > 0) {
        m_Mux.video.streamOut->codecpar->codec_tag           = tagFromStr(prm->videoCodecTag);
        AddMessage(RGY_LOG_DEBUG, _T("Set Video Codec Tag: %s\n"), char_to_tstring(tagToStr(m_Mux.video.streamOut->codecpar->codec_tag)).c_str());
    }
    if (videoOutputInfo->vui.descriptpresent) {
        m_Mux.video.streamOut->codecpar->color_space         = (AVColorSpace)videoOutputInfo->vui.matrix;
        m_Mux.video.streamOut->codecpar->color_primaries     = (AVColorPrimaries)videoOutputInfo->vui.colorprim;
        m_Mux.video.streamOut->codecpar->color_range         = (AVColorRange)videoOutputInfo->vui.colorrange;
        m_Mux.video.streamOut->codecpar->color_trc           = (AVColorTransferCharacteristic)videoOutputInfo->vui.transfer;
        AddMessage(RGY_LOG_DEBUG, _T("Set VUI Params: %s\n"), videoOutputInfo->vui.print_all().c_str());
    }
    if (0 > avcodec_open2(m_Mux.video.streamOut->codec, m_Mux.video.codec, NULL)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s for video.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("opened video avcodec\n"));

    m_Mux.video.bitstreamTimebase    = (av_isvalid_q(prm->bitstreamTimebase)) ? prm->bitstreamTimebase : HW_NATIVE_TIMEBASE;
    m_Mux.video.streamOut->time_base = (av_isvalid_q(prm->bitstreamTimebase)) ? prm->bitstreamTimebase : av_inv_q(m_Mux.video.outputFps);
    if (m_Mux.format.isMatroska) {
        m_Mux.video.streamOut->time_base = av_make_q(1, 1000);
    }
    m_Mux.video.streamOut->start_time          = 0;
    m_Mux.video.dtsUnavailable   = prm->bVideoDtsUnavailable;
    m_Mux.video.inputFirstKeyPts = prm->videoInputFirstKeyPts;
    m_Mux.video.timestamp        = prm->vidTimestamp;
    m_Mux.video.afs              = prm->afs;

    if (prm->videoInputStream) {
        m_Mux.video.inputStreamTimebase = prm->videoInputStream->time_base;
        m_Mux.video.streamOut->disposition = prm->videoInputStream->disposition;
        if (prm->videoInputStream->metadata) {
            auto language_data = av_dict_get(prm->videoInputStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
            if (language_data) {
                av_dict_set(&m_Mux.video.streamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Set Video language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
            }
        }
        int side_data_size = 0;
        auto side_data = av_stream_get_side_data(prm->videoInputStream, AV_PKT_DATA_DISPLAYMATRIX, &side_data_size);
        if (side_data) {
            unique_ptr<uint8_t, decltype(&av_freep)> side_data_copy((uint8_t *)av_malloc(side_data_size), av_freep);
            memcpy(side_data_copy.get(), side_data, side_data_size);
            auto rotation = av_display_rotation_get((const int *)side_data_copy.get());
            int err = av_stream_add_side_data(m_Mux.video.streamOut, AV_PKT_DATA_DISPLAYMATRIX, side_data_copy.get(), side_data_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rotation %d from input\n"), rotation);
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("copied rotation %d from input\n"), rotation);
            side_data_copy.release();
        }
#if 0
        if (videoOutputInfo->codec == RGY_CODEC_HEVC && prm->HEVCHdrSei == nullptr) {
            side_data = av_stream_get_side_data(prm->videoInputStream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, &side_data_size);
            if (side_data) {
                unique_ptr<uint8_t, decltype(&av_freep)> side_data_copy((uint8_t *)av_malloc(side_data_size), av_freep);
                memcpy(side_data_copy.get(), side_data, side_data_size);
                AVContentLightMetadata *coll = (AVContentLightMetadata *)side_data_copy.get();
                AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"), coll->MaxCLL, coll->MaxFALL);
                int err = av_stream_add_side_data(m_Mux.video.streamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, (uint8_t *)side_data_copy.get(), side_data_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_CONTENT_LIGHT_LEVEL from input\n"));
                side_data_copy.release();
            }
            side_data = av_stream_get_side_data(prm->videoInputStream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, &side_data_size);
            if (side_data && side_data_size == sizeof(AVMasteringDisplayMetadata)) {
                unique_ptr<uint8_t, decltype(&av_freep)> side_data_copy((uint8_t *)av_malloc(side_data_size), av_freep);
                memcpy(side_data_copy.get(), side_data, side_data_size);
                AVMasteringDisplayMetadata *mastering = (AVMasteringDisplayMetadata *)side_data_copy.get();
                AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
                    av_q2d(mastering->display_primaries[0][0]),
                    av_q2d(mastering->display_primaries[0][1]),
                    av_q2d(mastering->display_primaries[1][0]),
                    av_q2d(mastering->display_primaries[1][1]),
                    av_q2d(mastering->display_primaries[2][0]),
                    av_q2d(mastering->display_primaries[2][1]),
                    av_q2d(mastering->white_point[0]), av_q2d(mastering->white_point[1]),
                    av_q2d(mastering->min_luminance), av_q2d(mastering->max_luminance));

                int err = av_stream_add_side_data(m_Mux.video.streamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, (uint8_t *)side_data_copy.get(), side_data_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_MASTERING_DISPLAY_METADATA from input\n"));
                side_data_copy.release();
            }
        }
#endif
    }

    m_Mux.video.timestampList.clear();

    if (videoOutputInfo->codec == RGY_CODEC_HEVC && prm->HEVCHdrSei != nullptr) {
        auto seiNal = prm->HEVCHdrSei->gen_nal();
        if (seiNal.size() > 0) {
            m_Mux.video.seiNal.copy(seiNal.data(), (uint32_t)seiNal.size());
            AddMessage(RGY_LOG_DEBUG, char_to_tstring(prm->HEVCHdrSei->print()));

            const auto HEVCHdrSeiPrm = prm->HEVCHdrSei->getprm();

            if (false && HEVCHdrSeiPrm.masterdisplay_set) {
                unique_ptr<AVMasteringDisplayMetadata, decltype(&av_freep)> mastering(av_mastering_display_metadata_alloc(), av_freep);

                //streamのside dataとしてmasteringdisplay等を設定する
                mastering->display_primaries[1][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[0], 50000); //G
                mastering->display_primaries[1][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[1], 50000); //G
                mastering->display_primaries[2][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[2], 50000); //B
                mastering->display_primaries[2][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[3], 50000); //B
                mastering->display_primaries[0][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[4], 50000); //R
                mastering->display_primaries[0][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[5], 50000); //R
                mastering->white_point[0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[6], 50000);
                mastering->white_point[1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[7], 50000);
                mastering->max_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[8], 10000);
                mastering->min_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[9], 10000);
                mastering->has_primaries = 1;
                mastering->has_luminance = 1;

                AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
                    av_q2d(mastering->display_primaries[0][0]),
                    av_q2d(mastering->display_primaries[0][1]),
                    av_q2d(mastering->display_primaries[1][0]),
                    av_q2d(mastering->display_primaries[1][1]),
                    av_q2d(mastering->display_primaries[2][0]),
                    av_q2d(mastering->display_primaries[2][1]),
                    av_q2d(mastering->white_point[0]), av_q2d(mastering->white_point[1]),
                    av_q2d(mastering->max_luminance), av_q2d(mastering->min_luminance));

                int err = av_stream_add_side_data(m_Mux.video.streamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, (uint8_t *)mastering.get(), sizeof(mastering.get()[0]));
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                mastering.release(); //av_stream_add_side_dataされたデータはこちらで開放してはいけない
                AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
            }

            if (false && HEVCHdrSeiPrm.contentlight_set) {
                size_t coll_size = 0;
                unique_ptr<AVContentLightMetadata, decltype(&av_freep)> coll(av_content_light_metadata_alloc(&coll_size), av_freep);
                coll->MaxCLL = HEVCHdrSeiPrm.maxcll;
                coll->MaxFALL = HEVCHdrSeiPrm.maxfall;
                AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"),
                    coll->MaxCLL, coll->MaxFALL);
                int err = av_stream_add_side_data(m_Mux.video.streamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, (uint8_t *)coll.get(), coll_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                coll.release(); //av_stream_add_side_dataされたデータはこちらで開放してはいけない
                AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
            }
        }
    }

    if ((ENCODER_NVENC
        && (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC)
        && videoOutputInfo->sar[0] * videoOutputInfo->sar[1] > 0)
        || (ENCODER_VCEENC
            && (videoOutputInfo->vui.format != 5
                || videoOutputInfo->vui.colorprim != 2
                || videoOutputInfo->vui.transfer != 2
                || videoOutputInfo->vui.matrix != 2
                || videoOutputInfo->vui.chromaloc != 0))) {
        const char *bsf_name = nullptr;
        switch (videoOutputInfo->codec) {
        case RGY_CODEC_H264: bsf_name = "h264_metadata"; break;
        case RGY_CODEC_HEVC: bsf_name = "hevc_metadata"; break;
        default:
            break;
        }
        if (bsf_name == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("invalid codec to set metadata filter.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        const auto bsf_tname = char_to_tstring(bsf_name);
        AddMessage(RGY_LOG_DEBUG, _T("start initialize %s filter...\n"), bsf_tname.c_str());
        auto filter = av_bsf_get_by_name(bsf_name);
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find %s.\n"), bsf_tname.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>> codecpar(avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

        codecpar->codec_type              = AVMEDIA_TYPE_VIDEO;
        codecpar->codec_id                = getAVCodecId(videoOutputInfo->codec);
        codecpar->width                   = videoOutputInfo->dstWidth;
        codecpar->height                  = videoOutputInfo->dstHeight;
        codecpar->format                  = csp_rgy_to_avpixfmt(videoOutputInfo->csp);
        codecpar->level                   = videoOutputInfo->codecLevel;
        codecpar->profile                 = videoOutputInfo->codecProfile;
        codecpar->sample_aspect_ratio.num = videoOutputInfo->sar[0];
        codecpar->sample_aspect_ratio.den = videoOutputInfo->sar[1];
        codecpar->chroma_location         = (AVChromaLocation)videoOutputInfo->vui.chromaloc;
        codecpar->field_order             = picstrcut_rgy_to_avfieldorder(videoOutputInfo->picstruct);
        codecpar->video_delay             = videoOutputInfo->videoDelay;
        if (videoOutputInfo->vui.descriptpresent) {
            codecpar->color_space         = (AVColorSpace)videoOutputInfo->vui.matrix;
            codecpar->color_primaries     = (AVColorPrimaries)videoOutputInfo->vui.colorprim;
            codecpar->color_range         = (AVColorRange)videoOutputInfo->vui.colorrange;
            codecpar->color_trc           = (AVColorTransferCharacteristic)videoOutputInfo->vui.transfer;
        }
        int ret = 0;
        if (0 > (ret = av_bsf_alloc(filter, &m_Mux.video.bsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        if (0 > (ret = avcodec_parameters_copy(m_Mux.video.bsfc->par_in, codecpar.get()))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AVDictionary *bsfPrm = nullptr;
        unique_ptr<AVDictionary*, decltype(&av_dict_free)> bsfPrmDictDeleter(&bsfPrm, av_dict_free);
        if (ENCODER_NVENC) {
            char sar[128];
            sprintf_s(sar, "%d/%d", videoOutputInfo->sar[0], videoOutputInfo->sar[1]);
            av_dict_set(&bsfPrm, "sample_aspect_ratio", sar, 0);
            AddMessage(RGY_LOG_DEBUG, _T("set sar %d:%d by %s filter\n"), videoOutputInfo->sar[0], videoOutputInfo->sar[1], bsf_tname.c_str());
        }
        if (ENCODER_VCEENC) {
            if (videoOutputInfo->vui.format != 5 /*undef*/) {
                av_dict_set_int(&bsfPrm, "video_format", videoOutputInfo->vui.format, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set video_format %d by %s filter\n"), videoOutputInfo->vui.format, bsf_tname.c_str());
            }
            if (videoOutputInfo->vui.colorprim != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "colour_primaries", videoOutputInfo->vui.colorprim, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set colorprim %d by %s filter\n"), videoOutputInfo->vui.colorprim, bsf_tname.c_str());
            }
            if (videoOutputInfo->vui.transfer != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "transfer_characteristics", videoOutputInfo->vui.transfer, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set transfer %d by %s filter\n"), videoOutputInfo->vui.transfer, bsf_tname.c_str());
            }
            if (videoOutputInfo->vui.matrix != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "matrix_coefficients", videoOutputInfo->vui.matrix, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set matrix %d by %s filter\n"), videoOutputInfo->vui.matrix, bsf_tname.c_str());
            }
            if (videoOutputInfo->vui.chromaloc != 0) {
                av_dict_set_int(&bsfPrm, "chroma_sample_loc_type", videoOutputInfo->vui.chromaloc-1, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set chromaloc %d by %s filter\n"), videoOutputInfo->vui.chromaloc-1, bsf_tname.c_str());
            }
        }
        if (0 > (ret = av_opt_set_dict2(m_Mux.video.bsfc, &bsfPrm, AV_OPT_SEARCH_CHILDREN))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (0 > (ret = av_bsf_init(m_Mux.video.bsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter\n"), bsf_tname.c_str());
    }

#if ENCODER_VCEENC
    //parserを初期化 (VCEのみで必要)
    if (nullptr == (m_Mux.video.parserCtx = av_parser_init(m_Mux.format.formatCtx->video_codec_id))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init parser for %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.formatCtx->video_codec_id)).c_str());
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.video.parserCtx->flags |= PARSER_FLAG_COMPLETE_FRAMES;
    m_Mux.video.parserStreamPos = 0;
#endif //#if ENCODER_VCEENC

    if (prm->muxVidTsLogFile.length() > 0) {
        if (_tfopen_s(&m_Mux.video.fpTsLogFile, prm->muxVidTsLogFile.c_str(), _T("a"))) {
            AddMessage(RGY_LOG_WARN, _T("failed to open mux timestamp log file: \"%s\""), prm->muxVidTsLogFile.c_str());
            m_Mux.video.fpTsLogFile = NULL;
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Opened mux timestamp log file: \"%s\""), prm->muxVidTsLogFile.c_str());
            tstring strFileHeadSep;
            for (int i = 0; i < 78; i++) {
                strFileHeadSep += _T("-");
            }
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), m_Mux.format.filename);
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), strFileHeadSep.c_str());
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("output video stream timebase: %d/%d\n"), m_Mux.video.streamOut->time_base.num, m_Mux.video.streamOut->time_base.den);
    AddMessage(RGY_LOG_DEBUG, _T("bDtsUnavailable: %s\n"), (m_Mux.video.dtsUnavailable) ? _T("on") : _T("off"));
    return RGY_ERR_NONE;
}
#pragma warning (pop)

//音声フィルタの初期化
RGY_ERR RGYOutputAvcodec::InitAudioFilter(AVMuxAudio *muxAudio, int channels, uint64_t channel_layout, int sample_rate, AVSampleFormat sample_fmt) {
    //必要ならfilterを初期化
    if ((!muxAudio->filterGraph && (
        //フィルタが初期化されていない場合
        muxAudio->filter
        || bSplitChannelsEnabled(muxAudio->streamChannelSelect)
        || bSplitChannelsEnabled(muxAudio->streamChannelOut)
        || muxAudio->outCodecDecodeCtx->frame_size != muxAudio->outCodecEncodeCtx->frame_size
        || muxAudio->filterInChannels      != channels
        || muxAudio->filterInChannelLayout != channel_layout
        || muxAudio->filterInSampleRate    != sample_rate
        || muxAudio->filterInSampleFmt      != sample_fmt
        ))
        ||
        //フィルタがすでに初期化されている場合
        (  muxAudio->filterInChannels      != channels
        || muxAudio->filterInChannelLayout != channel_layout
        || muxAudio->filterInSampleRate    != sample_rate
        || muxAudio->filterInSampleFmt      != sample_fmt
        )) {
        if (muxAudio->filterGraph) {
            //filterをflush
            auto filteredFrames = AudioFilterFrameFlush(muxAudio);
            WriteNextPacketToAudioSubtracks(filteredFrames);

            //filterをclose
            avfilter_graph_free(&muxAudio->filterGraph);
        }
        muxAudio->filterInChannels      = channels;
        muxAudio->filterInChannelLayout = channel_layout;
        muxAudio->filterInSampleRate    = sample_rate;
        muxAudio->filterInSampleFmt      = sample_fmt;
        if (!channel_layout) {
            //時折channel_layoutが設定されていない場合がある
            channel_layout = av_get_default_channel_layout(channels);
        }

        int ret = 0;
        muxAudio->filterGraph = avfilter_graph_alloc();
        av_opt_set_int(muxAudio->filterGraph, "threads", 1, 0);

        auto filterchain = tchar_to_string(muxAudio->filter);

        //チャンネルレイアウトの変更
        if (bSplitChannelsEnabled(muxAudio->streamChannelSelect)
            && muxAudio->streamChannelSelect[muxAudio->inSubStream] != channel_layout
            && av_get_channel_layout_nb_channels(muxAudio->streamChannelSelect[muxAudio->inSubStream]) < channels) {
            //初期化
            for (int inChannel = 0; inChannel < _countof(muxAudio->channelMapping); inChannel++) {
                muxAudio->channelMapping[inChannel] = -1;
            }
            //時折channel_layoutが設定されていない場合がある
            const auto channelLayoutDec = (muxAudio->outCodecDecodeCtx->channel_layout) ? muxAudio->outCodecDecodeCtx->channel_layout : av_get_default_channel_layout(muxAudio->outCodecDecodeCtx->channels);
            //オプションによって指定されている、入力音声から抽出するべき音声レイアウト
            const auto select_channel_layout = muxAudio->streamChannelSelect[muxAudio->inSubStream];
            const int select_channel_count = av_get_channel_layout_nb_channels(select_channel_layout);
            std::string channel_map = "pan=" + getChannelLayoutChar(select_channel_count, av_get_default_channel_layout(select_channel_count));
            for (int inChannel = 0; inChannel < select_channel_count; inChannel++) {
                //オプションによって指定されているチャンネルレイアウトから、抽出する音声のチャンネルを順に取得する
                //実際には、「オプションによって指定されているチャンネルレイアウト」が入力音声に存在しない場合がある
                auto select_channel = av_channel_layout_extract_channel(select_channel_layout, std::min(inChannel, select_channel_count-1));
                //対象のチャンネルのインデックスを取得する
                auto select_channel_index = av_get_channel_layout_channel_index(channelLayoutDec, select_channel);
                if (select_channel_index < 0) {
                    //対応するチャンネルがもともとの入力音声ストリームにはない場合
                    const auto nChannels = (std::min)(inChannel, av_get_channel_layout_nb_channels(channelLayoutDec)-1);
                    //入力音声のストリームから、抽出する音声のチャンネルを順に取得する
                    select_channel = av_channel_layout_extract_channel(channelLayoutDec, nChannels);
                    //対象のチャンネルのインデックスを取得する
                    select_channel_index = av_get_channel_layout_channel_index(channelLayoutDec, select_channel);
                }
                muxAudio->channelMapping[inChannel] = select_channel_index;
                channel_map += "|c" + std::to_string(inChannel) + "=c" + std::to_string(select_channel_index);
            }
            if (filterchain.length() > 0) filterchain += ",";
            filterchain += channel_map;
            if (RGY_LOG_DEBUG >= m_printMes->getLogLevel()) {
                tstring channel_layout_str = strsprintf(_T("channel layout for track %d.%d:\n["), trackID(muxAudio->inTrackId), muxAudio->inSubStream);
                for (int inChannel = 0; inChannel < channels; inChannel++) {
                    channel_layout_str += strsprintf(_T("%4d"), muxAudio->channelMapping[inChannel]);
                }
                channel_layout_str += _T("]\n");
                AddMessage(RGY_LOG_DEBUG, channel_layout_str.c_str());
            }
        }

        if (filterchain.length() > 0) filterchain += ",";
        filterchain += strsprintf("aformat=sample_fmts=%s:sample_rates=%d:channel_layouts=0x%I64x",
            av_get_sample_fmt_name(muxAudio->outCodecEncodeCtx->sample_fmt),
            muxAudio->outCodecEncodeCtx->sample_rate,
            muxAudio->outCodecEncodeCtx->channel_layout);

        AVFilterInOut *filter_inputs = nullptr;
        AVFilterInOut *filter_outputs = nullptr;
        if (0 > (ret = avfilter_graph_parse2(muxAudio->filterGraph, filterchain.c_str(), &filter_inputs, &filter_outputs))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to parse filter description: %s: \"%s\"\n"), qsv_av_err2str(ret).c_str(), muxAudio->filter);
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
        const auto inargs = strsprintf("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%I64x",
            1, sample_rate,
            sample_rate, av_get_sample_fmt_name(sample_fmt), channel_layout);
        const AVFilter *abuffersrc  = avfilter_get_by_name("abuffer");
        const auto inName = strsprintf("in_track_%d.%d", trackID(muxAudio->inTrackId), muxAudio->inSubStream);
        if (0 > (ret = avfilter_graph_create_filter(&muxAudio->filterBufferSrcCtx, abuffersrc, inName.c_str(), inargs.c_str(), nullptr, muxAudio->filterGraph))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create abuffer: %s.\n"), qsv_av_err2str(ret).c_str());
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

RGY_ERR RGYOutputAvcodec::InitAudio(AVMuxAudio *muxAudio, AVOutputStreamPrm *inputAudio, uint32_t audioIgnoreDecodeError) {
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
    memcpy(muxAudio->streamChannelSelect, inputAudio->src.streamChannelSelect, sizeof(inputAudio->src.streamChannelSelect));
    memcpy(muxAudio->streamChannelOut,    inputAudio->src.streamChannelOut,    sizeof(inputAudio->src.streamChannelOut));

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
                    AddMessage(RGY_LOG_ERROR, _T("Unknown option to audio decoder[%s]: %s=%s\n"),
                        char_to_tstring(muxAudio->outCodecDecode->name).c_str(),
                        char_to_tstring(t->key).c_str(),
                        char_to_tstring(t->value).c_str());
                    m_Mux.format.streamError = true;
                    return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decoder opened\n"));
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decode Info: %s, %dch[0x%02x], %.1fkHz, %s, %d/%d %s\n"), char_to_tstring(avcodec_get_name(muxAudio->streamIn->codecpar->codec_id)).c_str(),
                muxAudio->outCodecDecodeCtx->channels, (uint32_t)muxAudio->outCodecDecodeCtx->channel_layout, muxAudio->outCodecDecodeCtx->sample_rate / 1000.0,
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

        //チャンネル選択の自動設定を反映
        for (int i = 0; i < _countof(muxAudio->streamChannelSelect); i++) {
            if (muxAudio->streamChannelSelect[i] == RGY_CHANNEL_AUTO) {
                muxAudio->streamChannelSelect[i] = (muxAudio->outCodecDecodeCtx->channel_layout)
                    ? muxAudio->outCodecDecodeCtx->channel_layout
                    : av_get_default_channel_layout(muxAudio->outCodecDecodeCtx->channels);
            }
        }

        auto enc_channel_layout = AutoSelectChannelLayout(muxAudio->outCodecEncode->channel_layouts, muxAudio->outCodecDecodeCtx);
        //もしチャンネルの分離・変更があれば、それを反映してエンコーダの入力とする
        if (bSplitChannelsEnabled(muxAudio->streamChannelOut)) {
            enc_channel_layout = muxAudio->streamChannelOut[muxAudio->inSubStream];
            if (enc_channel_layout == RGY_CHANNEL_AUTO) {
                auto channels = av_get_channel_layout_nb_channels(muxAudio->streamChannelSelect[muxAudio->inSubStream]);
                enc_channel_layout = av_get_default_channel_layout(channels);
            }
        }
        int enc_sample_rate = (inputAudio->samplingRate) ? inputAudio->samplingRate : muxAudio->outCodecDecodeCtx->sample_rate;
        //select samplefmt
        muxAudio->outCodecEncodeCtx->sample_fmt          = AutoSelectSampleFmt(muxAudio->outCodecEncode->sample_fmts, muxAudio->outCodecDecodeCtx);
        muxAudio->outCodecEncodeCtx->sample_rate         = AutoSelectSamplingRate(muxAudio->outCodecEncode->supported_samplerates, enc_sample_rate);
        muxAudio->outCodecEncodeCtx->channel_layout      = enc_channel_layout;
        muxAudio->outCodecEncodeCtx->channels            = av_get_channel_layout_nb_channels(enc_channel_layout);
        muxAudio->outCodecEncodeCtx->bits_per_raw_sample = muxAudio->outCodecDecodeCtx->bits_per_raw_sample;
        muxAudio->outCodecEncodeCtx->pkt_timebase        = av_make_q(1, muxAudio->outCodecDecodeCtx->sample_rate);
        if (!avcodecIsCopy(inputAudio->encodeCodec)) {
            muxAudio->outCodecEncodeCtx->bit_rate        = ((inputAudio->bitrate) ? inputAudio->bitrate : AVQSV_DEFAULT_AUDIO_BITRATE) * 1000;
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
            muxAudio->outCodecEncodeCtx->profile = selected_profile;
            AddMessage(RGY_LOG_DEBUG, _T("profile %d (%s) selected for codec %s (audio track %d)."),
                selected_profile, inputAudio->encodeCodecProfile.c_str(),
                char_to_tstring(muxAudio->outCodecEncode->name).c_str(), trackID(inputAudio->src.trackId));
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
        AddMessage(RGY_LOG_DEBUG, _T("Audio Encoder Param: %s, %dch[0x%02x], %.1fkHz, %s, %d (%s), %d/%d, %s\n"),
            char_to_tstring(muxAudio->outCodecEncode->name).c_str(),
            muxAudio->outCodecEncodeCtx->channels, (uint32_t)muxAudio->outCodecEncodeCtx->channel_layout, muxAudio->outCodecEncodeCtx->sample_rate / 1000.0,
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
                AddMessage(RGY_LOG_ERROR, _T("Unknown option to audio encoder[%s]: %s=%s\n"),
                    char_to_tstring(muxAudio->outCodecEncode->name).c_str(),
                    char_to_tstring(t->key).c_str(),
                    char_to_tstring(t->value).c_str());
                m_Mux.format.streamError = true;
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
        }

        muxAudio->filterInChannels      = av_get_channel_layout_nb_channels(muxAudio->outCodecEncodeCtx->channel_layout);
        muxAudio->filterInChannelLayout = muxAudio->outCodecEncodeCtx->channel_layout;
        muxAudio->filterInSampleRate    = muxAudio->outCodecEncodeCtx->sample_rate;
        muxAudio->filterInSampleFmt      = muxAudio->outCodecEncodeCtx->sample_fmt;
        //時折channel_layoutが設定されていない場合がある
        const auto channelLayoutDec = (muxAudio->outCodecDecodeCtx->channel_layout) ? muxAudio->outCodecDecodeCtx->channel_layout : av_get_default_channel_layout(muxAudio->outCodecDecodeCtx->channels);
        auto sts = InitAudioFilter(muxAudio,
            av_get_channel_layout_nb_channels(channelLayoutDec),
            channelLayoutDec,
            muxAudio->outCodecDecodeCtx->sample_rate,
            muxAudio->outCodecDecodeCtx->sample_fmt);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (muxAudio->streamIn->codecpar->codec_id == AV_CODEC_ID_AAC && muxAudio->streamIn->codecpar->extradata == NULL && m_Mux.video.streamOut) {
        AddMessage(RGY_LOG_DEBUG, _T("start initialize aac_adtstoasc filter...\n"));
        auto filter = av_bsf_get_by_name("aac_adtstoasc");
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find aac_adtstoasc.\n"));
            return RGY_ERR_NOT_FOUND;
        }
        int ret = 0;
        if (0 > (ret = av_bsf_alloc(filter, &muxAudio->AACBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        if (0 > (ret = avcodec_parameters_copy(muxAudio->AACBsfc->par_in, muxAudio->streamIn->codecpar))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        muxAudio->AACBsfc->time_base_in = muxAudio->streamIn->time_base;
        if (0 > (ret = av_bsf_init(muxAudio->AACBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (inputAudio->src.pktSample.data) {
            //mkvではavformat_write_headerまでにAVCodecContextにextradataをセットしておく必要がある
            for (AVPacket *inpkt = av_packet_clone(&inputAudio->src.pktSample); 0 == av_bsf_send_packet(muxAudio->AACBsfc, inpkt); inpkt = nullptr) {
                AVPacket outpkt = { 0 };
                av_init_packet(&outpkt);
                ret = av_bsf_receive_packet(muxAudio->AACBsfc, &outpkt);
                if (ret == 0) {
                    if (muxAudio->AACBsfc->par_out->extradata) {
                        SetExtraData(muxAudio->streamIn->codecpar, muxAudio->AACBsfc->par_out->extradata, muxAudio->AACBsfc->par_out->extradata_size);
                    }
                    break;
                }
                if (ret != AVERROR(EAGAIN) && !(inpkt && ret == AVERROR_EOF)) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to run aac_adtstoasc.\n"));
                    return RGY_ERR_UNKNOWN;
                }
                av_packet_unref(&outpkt);
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
        for (int i = 0; i < muxAudio->streamIn->nb_side_data; i++) {
            const AVPacketSideData *const sidedataSrc = &muxAudio->streamIn->side_data[i];
            uint8_t *const dst_data = av_stream_new_side_data(muxAudio->streamOut, sidedataSrc->type, sidedataSrc->size);
            memcpy(dst_data, sidedataSrc->data, sidedataSrc->size);
        }
        if (muxAudio->streamIn->codecpar->extradata_size) {
            //aac_adtstoascから得たヘッダをコピーする
            //これをしておかないと、avformat_write_headerで"Error parsing AAC extradata, unable to determine samplerate."という
            //意味不明なエラーメッセージが表示される
            AddMessage(RGY_LOG_DEBUG, _T("set extradata from original packet...\n"));
            SetExtraData(muxAudio->streamOut->codecpar, muxAudio->streamIn->codecpar->extradata, muxAudio->streamIn->codecpar->extradata_size);
        }
    }
    muxAudio->streamOut->time_base = av_make_q(1, muxAudio->streamOut->codecpar->sample_rate);
    muxAudio->streamOut->disposition = inputAudio->src.stream->disposition;

    if (inputAudio->src.subStreamId != 0) {
        //substream(--audio-filterなどによる複製stream)の場合はデフォルトstreamではない
        muxAudio->streamOut->disposition &= (~AV_DISPOSITION_DEFAULT);
    }
    if (inputAudio->src.stream->metadata) {
        for (AVDictionaryEntry *entry = nullptr;
        nullptr != (entry = av_dict_get(inputAudio->src.stream->metadata, "", entry, AV_DICT_IGNORE_SUFFIX));) {
            av_dict_set(&muxAudio->streamOut->metadata, entry->key, entry->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Copy Audio Metadata: key %s, value %s\n"), char_to_tstring(entry->key).c_str(), char_to_tstring(entry->value).c_str());
        }
        auto language_data = av_dict_get(inputAudio->src.stream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
        if (language_data) {
            av_dict_set(&muxAudio->streamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Set Audio language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::InitOther(AVMuxOther *muxSub, AVOutputStreamPrm *inputStream) {
    const auto mediaType = (inputStream->asdata) ? AVMEDIA_TYPE_UNKNOWN : trackMediaType(inputStream->src.trackId);
    const auto mediaTypeStr = char_to_tstring(av_get_media_type_string(mediaType));
    AddMessage(RGY_LOG_DEBUG, _T("start initializing %s ouput...\n"), mediaTypeStr.c_str());

    AVCodecID codecId = (inputStream->src.stream)
        ? inputStream->src.stream->codecpar->codec_id
        : (inputStream->src.caption2ass == FORMAT_ASS) ? AV_CODEC_ID_ASS : AV_CODEC_ID_SUBRIP;

    if (mediaType == AVMEDIA_TYPE_UNKNOWN) {
        codecId = AV_CODEC_ID_NONE;
    } else if (!avcodecIsCopy(inputStream->encodeCodec)) {
        auto codec = avcodec_find_decoder_by_name(tchar_to_string(inputStream->encodeCodec).c_str());
        if (codec == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find decoder for %s.\n"), inputStream->encodeCodec.c_str());
            return RGY_ERR_INVALID_CODEC;
        }
        codecId = codec->id;
    } else if (0 == strcmp(m_Mux.format.formatCtx->oformat->name, "mp4")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "mov")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "3gp")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "3g2")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "psp")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "ipod")
        || 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "f4v")) {
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

    auto copy_subtitle_header = [](AVCodecContext *dstCtx, const AVCodecContext *srcCtx) {
        if (srcCtx->subtitle_header_size) {
            dstCtx->subtitle_header_size = srcCtx->subtitle_header_size;
            dstCtx->subtitle_header = (uint8_t *)av_mallocz(dstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(dstCtx->subtitle_header, srcCtx->subtitle_header, srcCtx->subtitle_header_size);
        }
    };

    auto srcCodecParam = unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>>(
        avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

    if (inputStream->src.stream == nullptr) {
        //caption2assで生成した字幕を受け取る
        const auto src_codec_id = (inputStream->src.caption2ass == FORMAT_ASS) ? AV_CODEC_ID_ASS : AV_CODEC_ID_SUBRIP;
        const auto codec = avcodec_find_decoder(src_codec_id);
        if (nullptr == (muxSub->streamOut = avformat_new_stream(m_Mux.format.formatCtx, codec))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (src_codec_id == AV_CODEC_ID_ASS) {
            if (inputStream->src.subtitleHeader == nullptr || inputStream->src.subtitleHeaderSize == 0) {
                AddMessage(RGY_LOG_ERROR, _T("subtitle header unknown for track %d.\n"), trackID(inputStream->src.trackId));
                return RGY_ERR_NULL_PTR;
            }
            srcCodecParam->extradata_size = inputStream->src.subtitleHeaderSize;
            srcCodecParam->extradata = (uint8_t *)av_strdup((char *)inputStream->src.subtitleHeader);
        }
        srcCodecParam->codec_type = codec->type;
        srcCodecParam->codec_id   = codec->id;
    } else {
        avcodec_parameters_copy(srcCodecParam.get(), inputStream->src.stream->codecpar);

        if (nullptr == (muxSub->streamOut = avformat_new_stream(m_Mux.format.formatCtx, avcodec_find_decoder(codecId)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("output stream index %d, pkt_timebase %d/%d, trackId %d\n"),
            inputStream->src.index, inputStream->src.stream->time_base.num, inputStream->src.stream->time_base.den, trackID(inputStream->src.trackId));
    }
    if (inputStream->asdata) {
        srcCodecParam->codec_type = AVMEDIA_TYPE_UNKNOWN;
    } else if (mediaType == AVMEDIA_TYPE_DATA) {
        //なにもしない
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
        int ret;
        if (0 > (ret = avcodec_open2(muxSub->outCodecDecodeCtx, muxSub->outCodecDecode, nullptr))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(),
            muxSub->outCodecDecodeCtx->width, muxSub->outCodecDecodeCtx->height);

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
        if (inputStream->src.stream) {
            copy_subtitle_header(muxSub->outCodecEncodeCtx, inputStream->src.stream->codec);
        } else if (inputStream->src.subtitleHeader) {
            muxSub->outCodecEncodeCtx->subtitle_header = (uint8_t *)av_strdup((char *)inputStream->src.subtitleHeader);
            muxSub->outCodecEncodeCtx->subtitle_header_size = inputStream->src.subtitleHeaderSize;
        }

        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Encoder Param: %s, %dx%d\n"), char_to_tstring(muxSub->outCodecEncode->name).c_str(),
            muxSub->outCodecEncodeCtx->width, muxSub->outCodecEncodeCtx->height);
        if (muxSub->outCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
            //問答無用で使うのだ
            av_opt_set(muxSub->outCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (0 > (ret = avcodec_open2(muxSub->outCodecEncodeCtx, muxSub->outCodecEncode, nullptr))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
            AddMessage(RGY_LOG_ERROR, _T(" %s\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Opened Subtitle Encoder Param: %s\n"), char_to_tstring(muxSub->outCodecEncode->name).c_str());
        if (nullptr == (muxSub->bufConvert = (uint8_t *)av_malloc(SUB_ENC_BUF_MAX_SIZE))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate buffer memory for subtitle encoding.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        muxSub->streamOut->codec->codec = muxSub->outCodecEncodeCtx->codec;
    }

    muxSub->inTrackId     = inputStream->src.trackId;
    muxSub->streamIndexIn = inputStream->src.index;
    muxSub->streamIn      = inputStream->src.stream;
    muxSub->streamInTimebase = inputStream->src.timebase;

    if (muxSub->outCodecEncodeCtx) {
        avcodec_parameters_from_context(srcCodecParam.get(), muxSub->outCodecEncodeCtx);
    }
    avcodec_parameters_copy(muxSub->streamOut->codecpar, srcCodecParam.get());
    if (!muxSub->streamOut->codec->codec_tag) {
        uint32_t codec_tag = 0;
        if (!m_Mux.format.formatCtx->oformat->codec_tag
            || av_codec_get_id(m_Mux.format.formatCtx->oformat->codec_tag, srcCodecParam->codec_tag) == srcCodecParam->codec_id
            || !av_codec_get_tag2(m_Mux.format.formatCtx->oformat->codec_tag, srcCodecParam->codec_id, &codec_tag)) {
            muxSub->streamOut->codecpar->codec_tag = srcCodecParam->codec_tag;
        }
    }
    if (inputStream->src.stream) {
        copy_subtitle_header(muxSub->streamOut->codec, (muxSub->outCodecEncodeCtx) ?  muxSub->outCodecEncodeCtx : inputStream->src.stream->codec);
    } else if (inputStream->src.subtitleHeader != nullptr) {
        muxSub->streamOut->codec->subtitle_header = (uint8_t *)av_strdup((char *)inputStream->src.subtitleHeader);
        muxSub->streamOut->codec->subtitle_header_size = inputStream->src.subtitleHeaderSize;
    }
    muxSub->streamOut->time_base  = (mediaType == AVMEDIA_TYPE_SUBTITLE) ? av_make_q(1, 1000) : muxSub->streamInTimebase;
    muxSub->streamOut->start_time = 0;
    if (inputStream->src.stream) {
        muxSub->streamOut->disposition = inputStream->src.stream->disposition;
        if (inputStream->src.stream->metadata) {
            for (AVDictionaryEntry *entry = nullptr;
                nullptr != (entry = av_dict_get(inputStream->src.stream->metadata, "", entry, AV_DICT_IGNORE_SUFFIX));) {
                av_dict_set(&muxSub->streamOut->metadata, entry->key, entry->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Copy Subtitle Metadata: key %s, value %s\n"), char_to_tstring(entry->key).c_str(), char_to_tstring(entry->value).c_str());
            }
            auto language_data = av_dict_get(inputStream->src.stream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
            if (language_data) {
                av_dict_set(&muxSub->streamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Set Subtitle language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
            }
        }
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

    av_log_set_level((m_printMes->getLogLevel() == RGY_LOG_DEBUG) ?  AV_LOG_DEBUG : RGY_AV_LOG_LEVEL);
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
    int err = avformat_alloc_output_context2(&m_Mux.format.formatCtx, m_Mux.format.outputFmt, nullptr, filename.c_str());
    if (m_Mux.format.formatCtx == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate format context: %s.\n"), qsv_av_err2str(err).c_str());
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.format.isMatroska = 0 == strcmp(m_Mux.format.formatCtx->oformat->name, "matroska");
    m_Mux.format.disableMp4Opt = prm->disableMp4Opt;

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
            } else if (m_printMes->getLogLevel() == RGY_LOG_DEBUG) {
                AddMessage(RGY_LOG_DEBUG, _T("file name is %sunc path.\n"), (PathIsUNC(strFileName)) ? _T("") : _T("not "));
                if (PathFileExists(strFileName)) {
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
        if (NULL == (m_Mux.format.formatCtx->pb = avio_alloc_context(m_Mux.format.AVOutBuffer, m_Mux.format.AVOutBufferSize, 1, this, funcReadPacket, funcWritePacket, funcSeek))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return RGY_ERR_NULL_PTR;
        }
    }
#endif //#if USE_CUSTOM_IO

    m_Mux.trim = prm->trimList;

    if (videoOutputInfo) {
        RGY_ERR sts = InitVideo(videoOutputInfo, prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Initialized video output.\n"));
    }

    const int audioStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return trackMediaType(prm.src.trackId) == AVMEDIA_TYPE_AUDIO; });
    if (audioStreamCount) {
        m_Mux.audio.resize(audioStreamCount, { 0 });
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
                RGY_ERR sts = InitAudio(&m_Mux.audio[iAudioIdx], &prm->inputStreamList[iStream], prm->audioIgnoreDecodeError);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized audio output - #%d: track %d, substream %d.\n"),
                    iAudioIdx, trackID(prm->inputStreamList[iStream].src.trackId), prm->inputStreamList[iStream].src.subStreamId);
                iAudioIdx++;
            }
        }
    }
    const int otherStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return trackMediaType(prm.src.trackId) == AVMEDIA_TYPE_SUBTITLE || trackMediaType(prm.src.trackId) == AVMEDIA_TYPE_DATA; });
    if (otherStreamCount) {
        m_Mux.other.resize(otherStreamCount, { 0 });
        int iSubIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            const auto mediaType = trackMediaType(prm->inputStreamList[iStream].src.trackId);
            if (mediaType == AVMEDIA_TYPE_SUBTITLE || mediaType == AVMEDIA_TYPE_DATA) {
                RGY_ERR sts = InitOther(&m_Mux.other[iSubIdx], &prm->inputStreamList[iStream]);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized %s output - %d.\n"), char_to_tstring(av_get_media_type_string(mediaType)).c_str(), iSubIdx);
                iSubIdx++;
            }
        }
    }

    SetChapters(prm->chapterList, prm->chapterNoTrim);

    strcpy_s(m_Mux.format.formatCtx->filename, filename.c_str());
    if (m_Mux.format.outputFmt->flags & AVFMT_GLOBALHEADER) {
        if (m_Mux.video.streamOut) { m_Mux.video.streamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            if (m_Mux.audio[i].streamOut) { m_Mux.audio[i].streamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        }
        for (uint32_t i = 0; i < m_Mux.other.size(); i++) {
            if (m_Mux.other[i].streamOut) { m_Mux.other[i].streamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        }
    }

    if (m_Mux.format.formatCtx->metadata) {
        av_dict_copy(&m_Mux.format.formatCtx->metadata, prm->inputFormatMetadata, AV_DICT_DONT_OVERWRITE);
        av_dict_set(&m_Mux.format.formatCtx->metadata, "duration", NULL, 0);
        av_dict_set(&m_Mux.format.formatCtx->metadata, "creation_time", NULL, 0);
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

#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.streamOutMaxDts = 0;
    m_Mux.thread.queueInfo = prm->queueInfo;
    //スレッドの使用数を設定
    if (prm->threadOutput == RGY_OUTPUT_THREAD_AUTO) {
        prm->threadOutput = 1;
    }
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (prm->threadAudio == RGY_AUDIO_THREAD_AUTO) {
        prm->threadAudio = 0;
    }
    m_Mux.thread.enableAudProcessThread = prm->threadOutput > 0 && prm->threadAudio > 0;
    m_Mux.thread.enableAudEncodeThread  = prm->threadOutput > 0 && prm->threadAudio > 1;
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    m_Mux.thread.enableOutputThread     = prm->threadOutput > 0;
    if (m_Mux.thread.enableOutputThread) {
        AddMessage(RGY_LOG_DEBUG, _T("starting output thread...\n"));
        const int audioQueueCapacity = 4096;
        m_Mux.thread.abortOutput = false;
        m_Mux.thread.thAudProcessAbort = false;
        m_Mux.thread.thAudEncodeAbort = false;
        m_Mux.thread.qAudioPacketOut.init(16384, audioQueueCapacity * std::max(1, (int)m_Mux.audio.size())); //字幕のみコピーするときのため、最低でもある程度は確保する
        m_Mux.thread.qVideobitstream.init(4096, (std::max)(256, (m_Mux.video.outputFps.den) ? m_Mux.video.outputFps.num * 4 / m_Mux.video.outputFps.den : 0));
        m_Mux.thread.qVideobitstreamFreeI.init(256);
        m_Mux.thread.qVideobitstreamFreePB.init(3840);
        m_Mux.thread.heEventPktAddedOutput = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.heEventClosingOutput  = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.thOutput = std::thread(&RGYOutputAvcodec::WriteThreadFunc, this);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.enableAudProcessThread) {
            AddMessage(RGY_LOG_DEBUG, _T("starting audio process thread...\n"));
            m_Mux.thread.qAudioPacketProcess.init(16384, audioQueueCapacity * std::max(2, (int)m_Mux.audio.size()), 4);
            m_Mux.thread.heEventPktAddedAudProcess = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.heEventClosingAudProcess  = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.thAudProcess = std::thread(&RGYOutputAvcodec::ThreadFuncAudThread, this);
            if (m_Mux.thread.enableAudEncodeThread) {
                AddMessage(RGY_LOG_DEBUG, _T("starting audio encode thread...\n"));
                m_Mux.thread.qAudioFrameEncode.init(16384, audioQueueCapacity * std::max(2, (int)m_Mux.audio.size()), 4);
                m_Mux.thread.heEventPktAddedAudEncode = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.heEventClosingAudEncode  = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.thAudEncode = std::thread(&RGYOutputAvcodec::ThreadFuncAudEncodeThread, this);
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

RGY_ERR RGYOutputAvcodec::AddH264HeaderToExtraData(const RGYBitstream *bitstream) {
    std::vector<nal_info> nal_list = parse_nal_unit_h264(bitstream->data(), bitstream->size());
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
    return RGY_ERR_NONE;
}

//extradataにHEVCのヘッダーを追加する
RGY_ERR RGYOutputAvcodec::AddHEVCHeaderToExtraData(const RGYBitstream *bitstream) {
    std::vector<nal_info> nal_list = parse_nal_unit_hevc(bitstream->data(), bitstream->size());
    const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
    const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
    const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
    const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
    if (header_check) {
        m_Mux.video.streamOut->codecpar->extradata_size = (int)(hevc_vps_nal->size + hevc_sps_nal->size + hevc_pps_nal->size);
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.streamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, hevc_vps_nal->ptr, hevc_vps_nal->size);
        memcpy(new_ptr + hevc_vps_nal->size, hevc_sps_nal->ptr, hevc_sps_nal->size);
        memcpy(new_ptr + hevc_vps_nal->size + hevc_sps_nal->size, hevc_pps_nal->ptr, hevc_pps_nal->size);
        if (m_Mux.video.streamOut->codecpar->extradata) {
            av_free(m_Mux.video.streamOut->codecpar->extradata);
        }
        m_Mux.video.streamOut->codecpar->extradata = new_ptr;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteFileHeader(const RGYBitstream *bitstream) {
    if (m_Mux.video.streamOut && bitstream) {
        RGY_ERR sts = RGY_ERR_NONE;
        switch (m_Mux.video.streamOut->codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            sts = AddH264HeaderToExtraData(bitstream);
            break;
        case AV_CODEC_ID_HEVC:
            sts = AddHEVCHeaderToExtraData(bitstream);
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

    //なんらかの問題があると、ここでよく死ぬ
    int ret = 0;
    if (0 > (ret = avformat_write_header(m_Mux.format.formatCtx, &m_Mux.format.headerOptions))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write header for output file: %s\n"), qsv_av_err2str(ret).c_str());
        m_Mux.format.streamError = true;
        return RGY_ERR_UNKNOWN;
    }
    //不正なオプションを渡していないかチェック
    for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(m_Mux.format.headerOptions, "", t, AV_DICT_IGNORE_SUFFIX));) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown option to muxer: %s=%s\n"),
            char_to_tstring(t->key).c_str(),
            char_to_tstring(t->value).c_str());
        m_Mux.format.streamError = true;
        return RGY_ERR_INVALID_PARAM;
    }

    av_dump_format(m_Mux.format.formatCtx, 0, m_Mux.format.formatCtx->filename, 1);

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
                    getChannelLayoutChar(audioStream.outCodecDecodeCtx->channels, audioStream.outCodecDecodeCtx->channel_layout).c_str());
                if (audioStream.streamChannelSelect[audioStream.inSubStream] != 0) {
                    audiostr += strsprintf(":%s", getChannelLayoutChar(av_get_channel_layout_nb_channels(audioStream.streamChannelSelect[audioStream.inSubStream]), audioStream.streamChannelSelect[audioStream.inSubStream]).c_str());
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
                audiostr += strsprintf(" -> %s/%s/%dkbps",
                    audioStream.outCodecEncode->name,
                    getChannelLayoutChar(audioStream.outCodecEncodeCtx->channels, audioStream.outCodecEncodeCtx->channel_layout).c_str(),
                    audioStream.outCodecEncodeCtx->bit_rate / 1000);
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
    if (m_Mux.thread.thOutput.joinable()) {
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
        SetEvent(m_Mux.thread.heEventPktAddedOutput);
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }
#endif
    int64_t dts = 0;
    return WriteNextFrameInternal(bitstream, &dts);
}

#if ENCODER_VCEENC
RGY_ERR RGYOutputAvcodec::VidCheckStreamAVParser(RGYBitstream *pBitstream) {
    RGY_ERR err = RGY_ERR_NONE;
    m_Mux.video.parserStreamPos += pBitstream->size();
    AVPacket pkt;
    av_init_packet(&pkt);
    av_new_packet(&pkt, (int)pBitstream->size());
    memcpy(pkt.data, pBitstream->data(), pBitstream->size());
    pkt.size = (int)pBitstream->size();
    pkt.pts = pBitstream->pts();
    pkt.dts = pBitstream->dts();
    pkt.pos = m_Mux.video.parserStreamPos;
    uint8_t *dummy = nullptr;
    int dummy_size = 0;
    if (0 < av_parser_parse2(m_Mux.video.parserCtx, m_Mux.video.streamOut->codec, &dummy, &dummy_size, pkt.data, pkt.size, pkt.pts, pkt.dts, pkt.pos)) {
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
    av_packet_unref(&pkt);
    return err;
}
#endif

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputAvcodec::WriteNextFrameInternal(RGYBitstream *bitstream, int64_t *writtenDts) {
    if (!m_Mux.format.fileHeaderWritten) {
#if ENCODER_QSV
        //HEVCエンコードでは、DecodeTimeStampが正しく設定されない
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC && bitstream->dts() == MFX_TIMESTAMP_UNKNOWN) {
            m_Mux.video.dtsUnavailable = true;
        }
#else
        m_Mux.video.dtsUnavailable = true;
#endif
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC && m_Mux.video.seiNal.size() > 0) {
            RGYBitstream bsCopy = RGYBitstreamInit();
            bsCopy.copy(bitstream);
            std::vector<nal_info> nal_list = parse_nal_unit_hevc(bsCopy.data(), bsCopy.size());
            const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
            const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
            const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
            const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
            if (header_check) {
                bitstream->setSize(0);
                bitstream->setOffset(0);
                bitstream->append(hevc_vps_nal->ptr, hevc_vps_nal->size);
                bitstream->append(hevc_sps_nal->ptr, hevc_sps_nal->size);
                bitstream->append(hevc_pps_nal->ptr, hevc_pps_nal->size);
                bitstream->append(&m_Mux.video.seiNal);
                for (const auto& nal : nal_list) {
                    if (nal.type != NALU_HEVC_VPS && nal.type != NALU_HEVC_SPS && nal.type != NALU_HEVC_PPS) {
                        bitstream->append(nal.ptr, nal.size);
                    }
                }
            } else {
                AddMessage(RGY_LOG_ERROR, _T("Unexpected HEVC header.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            m_Mux.video.seiNal.clear();
        }
        RGY_ERR sts = WriteFileHeader(bitstream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        //dts生成を初期化
        //何フレーム前からにすればよいかは、b-pyramid次第で異なるので、可能な限りエンコーダの情報を使用する
        if (!m_Mux.video.dtsUnavailable) {
            const auto srcTimebase = (ENCODER_QSV) ? HW_NATIVE_TIMEBASE : m_Mux.video.bitstreamTimebase;
            m_VideoOutputInfo.videoDelay = -1 * (int)av_rescale_q(bitstream->dts(), srcTimebase, av_inv_q(m_Mux.video.outputFps));
        }
        m_Mux.video.fpsBaseNextDts = 0 - m_VideoOutputInfo.videoDelay;
        AddMessage(RGY_LOG_DEBUG, _T("calc dts, first dts %d x (timebase).\n"), m_Mux.video.fpsBaseNextDts);

        const AVRational fpsTimebase = (m_Mux.video.afs) ? av_inv_q(av_mul_q(m_Mux.video.outputFps, av_make_q(4, 5))) : av_inv_q(m_Mux.video.outputFps);
        const AVRational streamTimebase = m_Mux.video.streamOut->codec->pkt_timebase;
        for (int i = m_Mux.video.fpsBaseNextDts; i < 0; i++) {
            m_Mux.video.timestampList.add(av_rescale_q(i, fpsTimebase, streamTimebase));
        }
    }

#if ENCODER_QSV
    //QSVエンコーダでは、bitstreamからdurationの情報が取得できないので、別途取得する
    int64_t bs_duration = 0;
    if (m_Mux.video.timestamp) {
        while ((bs_duration = m_Mux.video.timestamp->get_and_pop(bitstream->pts())) < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
#endif

#if ENCODER_VCEENC
    VidCheckStreamAVParser(bitstream);
#endif //#if ENCODER_VCEENC

    std::vector<nal_info> nal_list;
    if (m_Mux.video.bsfc) {
        int target_nal = 0;
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
            target_nal = NALU_HEVC_SPS;
            nal_list = parse_nal_unit_hevc(bitstream->data(), bitstream->size());
        } else if (m_VideoOutputInfo.codec == RGY_CODEC_H264) {
            target_nal = NALU_H264_SPS;
            nal_list = parse_nal_unit_h264(bitstream->data(), bitstream->size());
        }
        auto sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [target_nal](nal_info info) { return info.type == target_nal; });
        if (sps_nal != nal_list.end()) {
            AVPacket pkt = { 0 };
            av_init_packet(&pkt);
            av_new_packet(&pkt, (int)sps_nal->size);
            memcpy(pkt.data, sps_nal->ptr, sps_nal->size);
            int ret = 0;
            if (0 > (ret = av_bsf_send_packet(m_Mux.video.bsfc, &pkt))) {
                av_packet_unref(&pkt);
                AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
                    char_to_tstring(m_Mux.video.bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            ret = av_bsf_receive_packet(m_Mux.video.bsfc, &pkt);
            if (ret == AVERROR(EAGAIN)) {
                return RGY_ERR_NONE;
            } else if ((ret < 0 && ret != AVERROR_EOF) || pkt.size < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
                    char_to_tstring(m_Mux.video.bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            const auto new_data_size = bitstream->size() + pkt.size - sps_nal->size;
            const auto sps_nal_offset = sps_nal->ptr - bitstream->data();
            const auto next_nal_orig_offset = sps_nal_offset + sps_nal->size;
            const auto next_nal_new_offset = sps_nal_offset + pkt.size;
            const auto stream_orig_length = bitstream->size();
            if ((decltype(new_data_size))bitstream->bufsize() < new_data_size) {
                bitstream->changeSize(new_data_size);
            } else if (pkt.size > (decltype(pkt.size))sps_nal->size) {
                bitstream->trim();
            }
            memmove(bitstream->data() + next_nal_new_offset, bitstream->data() + next_nal_orig_offset, stream_orig_length - next_nal_orig_offset);
            memcpy(bitstream->data() + sps_nal_offset, pkt.data, pkt.size);
            av_packet_unref(&pkt);
        }
    }

    //IDRかどうかのフラグ
    bool isIDR = (bitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) != 0;
    if (m_Mux.video.streamOut->codecpar->field_order != AV_FIELD_PROGRESSIVE) {
        if (m_VideoOutputInfo.codec == RGY_CODEC_H264) {
            if (nal_list.size() == 0) {
                nal_list = parse_nal_unit_h264(bitstream->data(), bitstream->size());
            }
            //インタレ保持の際、IDRかどうかのフラグが正しく設定されていないことがある
            //どちらかのフィールドがIDRならIDRのフラグを立てる
            isIDR = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_IDR; }) != nal_list.end();
        } else if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
            AddMessage(RGY_LOG_ERROR, _T("Interlaced HEVC encoding not supported!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    AVPacket pkt = { 0 };
    av_init_packet(&pkt);
    av_new_packet(&pkt, (int)bitstream->size());
    memcpy(pkt.data, bitstream->data(), bitstream->size());
    pkt.size = (int)bitstream->size();

    const AVRational streamTimebase = m_Mux.video.streamOut->codec->pkt_timebase;
    pkt.stream_index = m_Mux.video.streamOut->index;
    pkt.flags        = isIDR ? AV_PKT_FLAG_KEY : 0;
#if ENCODER_QSV
    //QSVエンコーダでは、bitstreamからdurationの情報が取得できないので、別途取得する
    pkt.duration = bs_duration;
#else
    pkt.duration = bitstream->duration();
#endif
    pkt.pts = bitstream->pts();
#if ENCODER_QSV
    //QSVエンコーダだけは、HW_NATIVE_TIMEBASEで送られてくる
    pkt.duration = av_rescale_q(pkt.duration, HW_NATIVE_TIMEBASE, m_Mux.video.bitstreamTimebase);
    pkt.pts = av_rescale_q(pkt.pts, HW_NATIVE_TIMEBASE, m_Mux.video.bitstreamTimebase);
#endif
    if (av_cmp_q(m_Mux.video.bitstreamTimebase, streamTimebase) != 0) {
        pkt.duration = av_rescale_q(pkt.duration, m_Mux.video.bitstreamTimebase, streamTimebase);
        pkt.pts      = av_rescale_q(pkt.pts, m_Mux.video.bitstreamTimebase, streamTimebase);
    }
    if (false && !m_Mux.video.dtsUnavailable) {
        pkt.dts = av_rescale_q(bitstream->dts(), m_Mux.video.bitstreamTimebase, streamTimebase);
    } else {
        m_Mux.video.timestampList.add(pkt.pts);
        pkt.dts = m_Mux.video.timestampList.get_min_pts();
    }
    const auto pts = pkt.pts, dts = pkt.dts, duration = pkt.duration;
    *writtenDts = av_rescale_q(pkt.dts, streamTimebase, QUEUE_DTS_TIMEBASE);
    m_Mux.format.streamError |= 0 != av_interleaved_write_frame(m_Mux.format.formatCtx, &pkt);

    //インタレ保持の際、IDRかどうかのフラグが正しく設定されていないことがある
    //どちらかのフィールドがIDRならIDRのフラグを立ててているので、それを参照する
    const auto frameType = (isIDR) ? RGY_FRAMETYPE_IDR : bitstream->frametype();
    if (m_Mux.video.fpTsLogFile) {
        const TCHAR *pFrameTypeStr =
            (frameType & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) ? _T("I") : (((frameType & RGY_FRAMETYPE_B) == 0) ? _T("P") : _T("B"));
        _ftprintf(m_Mux.video.fpTsLogFile, _T("%s, %20lld, %20lld, %20lld, %20lld, %d, %7zd\n"), pFrameTypeStr, (lls)bitstream->pts(), (lls)bitstream->dts(), (lls)pts, (lls)dts, (int)duration, bitstream->size());
    }
    m_encSatusInfo->SetOutputData(frameType, bitstream->size(), bitstream->avgQP());
#if ENABLE_AVCODEC_OUT_THREAD
    //最初のヘッダーを書いたパケットはコピーではないので、キューに入れない
    if (m_Mux.thread.thOutput.joinable()) {
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
#pragma warning (pop)

RGY_ERR RGYOutputAvcodec::WriteNextFrame(RGYFrame *surface) {
    UNREFERENCED_PARAMETER(surface);
    return RGY_ERR_UNSUPPORTED;
}

vector<int> RGYOutputAvcodec::GetStreamTrackIdList() {
    vector<int> streamTrackId;
    streamTrackId.reserve(m_Mux.audio.size());
    for (auto audio : m_Mux.audio) {
        streamTrackId.push_back(audio.inTrackId);
    }
    for (auto sub : m_Mux.other) {
        streamTrackId.push_back(sub.inTrackId);
    }
    return std::move(streamTrackId);
}

AVMuxAudio *RGYOutputAvcodec::getAudioPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (int)((uint32_t)pkt->flags >> 16);
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
    const int inTrackId = (int)((uint32_t)pkt->flags >> 16);
    for (int i = 0; i < (int)m_Mux.other.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.other[i].streamIndexIn == streamIndex
            && m_Mux.other[i].inTrackId == inTrackId) {
            return &m_Mux.other[i];
        }
    }
    return NULL;
}

RGY_ERR RGYOutputAvcodec::applyBitstreamFilterAAC(AVPacket *pkt, AVMuxAudio *muxAudio) {
    int ret = 0;
    //毎回bitstream filterを初期化して、extradataに新しいヘッダを供給する
    //動画とmuxする際に必須
    av_bsf_free(&muxAudio->AACBsfc);
    auto filter = av_bsf_get_by_name("aac_adtstoasc");
    if (filter == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find aac_adtstoasc.\n"));
        return RGY_ERR_NOT_FOUND;
    }
    if (0 > (ret = av_bsf_alloc(filter, &muxAudio->AACBsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    if (0 > (ret = avcodec_parameters_copy(muxAudio->AACBsfc->par_in, muxAudio->streamIn->codecpar))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    muxAudio->AACBsfc->time_base_in = muxAudio->streamIn->time_base;
    if (0 > (ret = av_bsf_init(muxAudio->AACBsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = av_bsf_send_packet(muxAudio->AACBsfc, pkt))) {
        av_packet_unref(pkt);
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to aac_adtstoasc bitstream filter: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(muxAudio->AACBsfc, pkt);
    if (ret == AVERROR(EAGAIN)) {
        pkt->size = 0;
        pkt->duration = 0;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
        //最初のフレームとかでなければ、エラーを許容し、単に処理しないようにする
        //多くの場合、ここでのエラーはtsなどの最終音声フレームが不完全なことで発生する
        //先頭から連続30回エラーとなった場合はおかしいのでエラー終了するようにする
        if (muxAudio->packetWritten == 0) {
            muxAudio->AACBsfErrorFromStart++;
            static int AACBSFFILTER_ERROR_THRESHOLD = 30;
            if (muxAudio->AACBsfErrorFromStart > AACBSFFILTER_ERROR_THRESHOLD) {
                m_Mux.format.streamError = true;
                AddMessage(RGY_LOG_ERROR, _T("failed to run aac_adtstoasc bitstream filter for %d times: %s.\n"), AACBSFFILTER_ERROR_THRESHOLD, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
        }
        AddMessage(RGY_LOG_WARN, _T("failed to run aac_adtstoasc bitstream filter: %s.\n"), qsv_av_err2str(ret).c_str());
        pkt->duration = 0; //書き込み処理が行われないように
        return RGY_WRN_FILTER_SKIPPED;
    }
    muxAudio->AACBsfErrorFromStart = 0;
    return RGY_ERR_NONE;
}

//音声/字幕パケットを実際に書き出す
// muxAudio ... [i]  pktに対応するストリーム情報
// pkt       ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// dts       ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVMuxAudio *muxAudio, AVPacket *pkt, int samples, int64_t *writtenDts) {
    if (pkt == nullptr || pkt->buf == nullptr) {
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            AudioFlushStream(&m_Mux.audio[i], writtenDts);
        }
        *writtenDts = INT64_MAX;
        AddMessage(RGY_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        return;
    }
    //durationについて、sample数から出力ストリームのtimebaseに変更する
    pkt->stream_index = muxAudio->streamOut->index;
    pkt->flags = AV_PKT_FLAG_KEY; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
    const AVRational samplerate = { 1, (muxAudio->outCodecEncodeCtx) ? muxAudio->outCodecEncodeCtx->sample_rate : muxAudio->streamIn->codecpar->sample_rate };
    if (!muxAudio->outCodecEncodeCtx) {
        if (samples > 0) {
            //av_rescale_deltaの入力ptsはAV_NOPTS_VALUEではない必要があるのでチェックする
            if (pkt->pts == AV_NOPTS_VALUE) {
                pkt->pts = muxAudio->lastPtsOut + (int)av_rescale_q(samples, samplerate, muxAudio->streamOut->time_base);
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
    if (m_Mux.video.streamOut && m_Mux.video.inputFirstKeyPts != 0) {
        pkt->pts -= av_rescale_q(m_Mux.video.inputFirstKeyPts, m_Mux.video.inputStreamTimebase, muxAudio->streamOut->time_base);
    }
    if (muxAudio->lastPtsOut != AV_NOPTS_VALUE) {
        //以前のptsより前になりそうになったら修正する
        const auto maxPts = muxAudio->lastPtsOut + !(m_Mux.format.formatCtx->oformat->flags & AVFMT_TS_NONSTRICT);
        if (pkt->pts < maxPts) {
            auto loglevel = (maxPts - pkt->pts > 2) ? RGY_LOG_WARN : RGY_LOG_DEBUG;
            if (loglevel < m_printMes->getLogLevel()) {
                AddMessage(loglevel, _T("Timestamp error in stream %d, previous: %lld, current: %lld [timebase: %d/%d].\n"),
                    muxAudio->streamOut->index,
                    (long long int)muxAudio->lastPtsOut,
                    (long long int)pkt->pts,
                    muxAudio->streamOut->time_base.num, muxAudio->streamOut->time_base.den);
                AddMessage(loglevel, _T("                              previous: %s current: %s\n"),
                    getTimestampString(muxAudio->lastPtsOut, muxAudio->streamOut->time_base).c_str(),
                    getTimestampString(pkt->pts, muxAudio->streamOut->time_base).c_str());
                AddMessage(loglevel, _T("Chaging timestamp to %lld(%s), this may corrupt av-synchronization.\n"),
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
    muxAudio->lastPtsOut = pkt->pts;
    *writtenDts = av_rescale_q(pkt->dts, muxAudio->streamOut->time_base, QUEUE_DTS_TIMEBASE);
    if (*writtenDts != AV_NOPTS_VALUE) {
        atomic_max(m_Mux.thread.streamOutMaxDts, *writtenDts);
    }
    //av_interleaved_write_frameに渡ったパケットは開放する必要がない
    m_Mux.format.streamError |= 0 != av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
    muxAudio->outputSamples += samples;
}

//音声/字幕パケットを実際に書き出す (構造体版)
// pktData->muxAudio ... [i]  pktに対応するストリーム情報
// &pktData->pkt      ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// pktData->samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// &pktData->dts      ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData) {
    WriteNextPacketProcessed(pktData->muxAudio, &pktData->pkt, pktData->samples, &pktData->dts);
}

void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData, int64_t *writtenDts) {
    WriteNextPacketProcessed(pktData->muxAudio, &pktData->pkt, pktData->samples, &pktData->dts);
    *writtenDts = pktData->dts;
}

vector<unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>> RGYOutputAvcodec::AudioDecodePacket(AVMuxAudio *muxAudio, AVPacket *pkt) {
    vector<unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>> decodedFrames;
    if (muxAudio->decodeError > muxAudio->ignoreDecodeError) {
        return std::move(decodedFrames);
    }
    AVPacket pktInInfo;
    av_packet_copy_props(&pktInInfo, pkt);

    bool sent_packet = false;

    //最終的な出力フレーム
    int64_t recieved_samples = 0;
    int recv_ret = 0;
    for (;;) {
        unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> receivedData(nullptr, RGYAVDeleter<AVFrame>(av_frame_free));
        int send_ret = 0;
        //必ず一度はパケットを送る
        if (!sent_packet || pkt->size > 0) {
            sent_packet = true;
            //ひとつのパケットをデコーダに送る
            send_ret = avcodec_send_packet(muxAudio->outCodecDecodeCtx, pkt);
            //AVERROR(EAGAIN) -> パケットを送る前に受け取る必要がある (パケットが受け取られていないので解放しない)
            if (send_ret != AVERROR(EAGAIN)) {
                av_packet_unref(pkt);
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
            receivedData = unique_ptr<AVFrame, RGYAVDeleter<AVFrame>>(av_frame_alloc(), RGYAVDeleter<AVFrame>(av_frame_free));
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
                    receivedData->pts = pktInInfo.pts;
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
                    muxAudio->decodeError, trackID(muxAudio->inTrackId), pktInInfo.pts, getTimestampString(pktInInfo.pts, muxAudio->streamIn->time_base).c_str());
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
            if (pktData.frame != nullptr) {
                //音声入力フォーマットに変更がないか確認し、もしあればresamplerを再初期化する
                auto sts = InitAudioFilter(muxAudio, pktData.frame->channels, pktData.frame->channel_layout, pktData.frame->sample_rate, (AVSampleFormat)pktData.frame->format);
                if (sts != RGY_ERR_NONE) {
                    m_Mux.format.streamError = true;
                    break;
                }
            }
            { //フィルターチェーンにフレームを追加
                auto ret = av_buffersrc_add_frame_flags(muxAudio->filterBufferSrcCtx, pktData.frame, AV_BUFFERSRC_FLAG_PUSH);
                // AVFrame構造体の破棄
                av_frame_free(&pktData.frame);
                if (ret < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to feed the audio filtergraph\n"));
                    m_Mux.format.streamError = true;
                    break;
                }
            }
            for (;;) {
                unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> filteredFrame(av_frame_alloc(), RGYAVDeleter<AVFrame>(av_frame_free));
                auto ret = av_buffersink_get_frame_flags(muxAudio->filterBufferSinkCtx, filteredFrame.get(), AV_BUFFERSINK_FLAG_NO_REQUEST);
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

    AVPktMuxData pktData;
    memset(&pktData.pkt, 0, sizeof(pktData.pkt)); //av_init_packetでsizeなどは初期化されないので0をセット
    pktData.type = MUX_DATA_TYPE_PACKET;
    pktData.muxAudio = muxAudio;
    while (ret >= 0) {
        av_init_packet(&pktData.pkt);
        ret = avcodec_receive_packet(muxAudio->outCodecEncodeCtx, &pktData.pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            AddMessage(RGY_LOG_WARN, _T("avcodec writer: failed to encode audio #%d: %s\n"), trackID(muxAudio->inTrackId), qsv_av_err2str(ret).c_str());
            muxAudio->encodeError = true;
        }
        pktData.samples = (int)av_rescale_q(pktData.pkt.duration, muxAudio->outCodecEncodeCtx->pkt_timebase, { 1, muxAudio->streamIn->codecpar->sample_rate });
        encPktDatas.push_back(pktData);
    }
    return encPktDatas;
}

void RGYOutputAvcodec::AudioFlushStream(AVMuxAudio *muxAudio, int64_t *writtenDts) {
    while (muxAudio->outCodecDecodeCtx && !muxAudio->encodeError) {
        AVPacket pkt = { 0 };
        auto decodedFrames = AudioDecodePacket(muxAudio, &pkt);
        if (decodedFrames.size() == 0) {
            break;
        }
        vector<AVPktMuxData> audioFrames;
        for (size_t i = 0; i < decodedFrames.size(); i++) {
            AVPktMuxData audPkt;
            av_init_packet(&audPkt.pkt);
            audPkt.dts = AV_NOPTS_VALUE;
            audPkt.samples = 0;
            audPkt.type = MUX_DATA_TYPE_FRAME;
            audPkt.frame = decodedFrames[i].release();
            audPkt.got_result = audPkt.frame && audPkt.frame->nb_samples > 0;
            audioFrames.push_back(audPkt);
        }
        //フィルタリングを行う
        WriteNextPacketToAudioSubtracks(std::move(audioFrames));
    }
    if (muxAudio->filterGraph) {
        WriteNextPacketAudioFrame(std::move(AudioFilterFrameFlush(muxAudio)));
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
}

RGY_ERR RGYOutputAvcodec::SubtitleTranscode(const AVMuxOther *muxSub, AVPacket *pkt) {
    //timescaleの変換が入ると、pts + duration > 次のpts となることがある
    //オリジナルのptsを使って再計算する
    const auto org_start_time = pkt->pts;
    const auto org_end_time = pkt->pts + pkt->duration;

    int got_sub = 0;
    AVSubtitle sub = { 0 };
    if (0 > avcodec_decode_subtitle2(muxSub->outCodecDecodeCtx, &sub, &got_sub, pkt)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
        m_Mux.format.streamError = true;
    }
    if (!muxSub->bufConvert) {
        AddMessage(RGY_LOG_ERROR, _T("No buffer for encoding subtitle.\n"));
        m_Mux.format.streamError = true;
    }
    av_packet_unref(pkt);
    if (m_Mux.format.streamError)
        return RGY_ERR_UNKNOWN;
    if (!got_sub || sub.num_rects == 0)
        return RGY_ERR_NONE;

    //AV_CODEC_ID_DVB_SUBTITLEははじめりと終わりで2パケット
    const int nOutPackets = 1 + (muxSub->outCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE);
    for (int i = 0; i < nOutPackets; i++) {
        sub.pts               += av_rescale_q(sub.start_display_time, av_make_q(1, 1000), av_make_q(1, AV_TIME_BASE));
        sub.end_display_time  -= sub.start_display_time;
        sub.start_display_time = 0;
        if (i > 0) {
            sub.num_rects = 0;
        }

        int sub_out_size = avcodec_encode_subtitle(muxSub->outCodecEncodeCtx, muxSub->bufConvert, SUB_ENC_BUF_MAX_SIZE, &sub);
        if (sub_out_size < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to encode subtitle.\n"));
            m_Mux.format.streamError = true;
            return RGY_ERR_UNKNOWN;
        }

        AVPacket pktOut;
        av_init_packet(&pktOut);
        pktOut.data = muxSub->bufConvert;
        pktOut.stream_index = muxSub->streamOut->index;
        pktOut.size = sub_out_size;
        // pts + duration <= 次のptsとなるよう、オリジナルのptsを使って再計算する
        auto end_ts = av_rescale_q(org_end_time,   muxSub->outCodecDecodeCtx->pkt_timebase, muxSub->streamOut->time_base);
        pktOut.pts  = av_rescale_q(org_start_time, muxSub->outCodecDecodeCtx->pkt_timebase, muxSub->streamOut->time_base);
        pktOut.duration = (int)av_rescale_q(end_ts - pktOut.pts, muxSub->outCodecDecodeCtx->pkt_timebase, muxSub->streamOut->time_base);
        if (muxSub->outCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {
            pktOut.pts += 90 * ((i == 0) ? sub.start_display_time : sub.end_display_time);
        }
        pktOut.dts = pktOut.pts;
        m_Mux.format.streamError |= 0 != av_interleaved_write_frame(m_Mux.format.formatCtx, &pktOut);
    }
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteOtherPacket(AVPacket *pkt) {
    //字幕を処理する
    const AVMuxOther *pMuxOther = getOtherPacketStreamData(pkt);
    const AVRational vid_pkt_timebase = av_isvalid_q(m_Mux.video.inputStreamTimebase) ? m_Mux.video.inputStreamTimebase : av_inv_q(m_Mux.video.outputFps);
    const int64_t pts_offset = av_rescale_q(m_Mux.video.inputFirstKeyPts, vid_pkt_timebase, pMuxOther->streamInTimebase);
    const AVRational timebase_conv = (pMuxOther->outCodecDecodeCtx) ? pMuxOther->outCodecDecodeCtx->pkt_timebase : pMuxOther->streamOut->time_base;
    pkt->pts = av_rescale_q(std::max<int64_t>(0, pkt->pts - pts_offset), pMuxOther->streamInTimebase, timebase_conv);
    pkt->dts = av_rescale_q(std::max<int64_t>(0, pkt->dts - pts_offset), pMuxOther->streamInTimebase, timebase_conv);
    pkt->flags &= 0x0000ffff; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
    pkt->duration = (int)av_rescale_q(pkt->duration, pMuxOther->streamInTimebase, pMuxOther->streamOut->time_base);
    pkt->stream_index = pMuxOther->streamOut->index;
    pkt->pos = -1;
    if (pkt->dts != AV_NOPTS_VALUE) {
        atomic_max(m_Mux.thread.streamOutMaxDts, av_rescale_q(pkt->dts, timebase_conv, QUEUE_DTS_TIMEBASE));
    }
    m_Mux.format.streamError |= 0 != av_interleaved_write_frame(m_Mux.format.formatCtx, pkt);
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

AVPktMuxData RGYOutputAvcodec::pktMuxData(const AVPacket *pkt) {
    AVPktMuxData data = { 0 };
    data.type = MUX_DATA_TYPE_PACKET;
    if (pkt) {
        data.pkt = *pkt;
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

RGY_ERR RGYOutputAvcodec::WriteNextPacket(AVPacket *pkt) {
    AVPktMuxData pktData = pktMuxData(pkt);
#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.thread.thOutput.joinable()) {
        auto& audioQueue   = (m_Mux.thread.thAudProcess.joinable()) ? m_Mux.thread.qAudioPacketProcess : m_Mux.thread.qAudioPacketOut;
        auto heEventPktAdd = (m_Mux.thread.thAudProcess.joinable()) ? m_Mux.thread.heEventPktAddedAudProcess : m_Mux.thread.heEventPktAddedOutput;
        //pkt = nullptrの代理として、pkt.buf == nullptrなパケットを投入
        AVPktMuxData zeroFilled = { 0 };
        if (!audioQueue.push((pkt == nullptr) ? zeroFilled : pktData)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for audio packet queue.\n"));
            m_Mux.format.streamError = true;
        }
        SetEvent(heEventPktAdd);
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }
#endif
    return WriteNextPacketInternal(&pktData, INT64_MAX);
}

//指定された音声キューに追加する
RGY_ERR RGYOutputAvcodec::AddAudQueue(AVPktMuxData *pktData, int type) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (m_Mux.thread.thAudProcess.joinable()) {
        //出力キューに追加する
        auto& qAudio       = (type == AUD_QUEUE_OUT) ? m_Mux.thread.qAudioPacketOut       : ((type == AUD_QUEUE_PROCESS) ? m_Mux.thread.qAudioPacketProcess       : m_Mux.thread.qAudioFrameEncode);
        auto& heEventAdded = (type == AUD_QUEUE_OUT) ? m_Mux.thread.heEventPktAddedOutput : ((type == AUD_QUEUE_PROCESS) ? m_Mux.thread.heEventPktAddedAudProcess : m_Mux.thread.heEventPktAddedAudEncode);
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
    if (!m_Mux.format.fileHeaderWritten) {
        //まだフレームヘッダーが書かれていなければ、パケットをキャッシュして終了
        m_AudPktBufFileHead.push_back(*pktData);
        return RGY_ERR_NONE;
    }
    //m_AudPktBufFileHeadにキャッシュしてあるパケットかどうかを調べる
    if (m_AudPktBufFileHead.end() == std::find_if(m_AudPktBufFileHead.begin(), m_AudPktBufFileHead.end(),
        [pktData](const AVPktMuxData& data) { return pktData->pkt.buf == data.pkt.buf; })) {
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

    if (pktData->pkt.data == nullptr) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.thAudProcess.joinable()) {
            //音声処理を別スレッドでやっている場合は、AddAudOutputQueueを後段の出力スレッドで行う必要がある
            //WriteNextPacketInternalでは音声キューに追加するだけにして、WriteNextPacketProcessedで対応する
            //ひとまず、ここでは処理せず、次のキューに回す
            return AddAudQueue(pktData, (m_Mux.thread.thAudEncode.joinable()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            AudioFlushStream(&m_Mux.audio[i], &pktData->dts);
        }
        pktData->dts = INT64_MAX;
        AddMessage(RGY_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }

    const int trackID = ((uint32_t)pktData->pkt.flags >> 16);
    if (trackMediaType(trackID) != AVMEDIA_TYPE_AUDIO) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.thAudProcess.joinable()) {
            //音声処理を別スレッドでやっている場合は、字幕パケットもその流れに乗せてやる必要がある
            //ひとまず、ここでは処理せず、次のキューに回す
            return AddAudQueue(pktData, (m_Mux.thread.thAudEncode.joinable()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        return WriteOtherPacket(&pktData->pkt);
    }
    return WriteNextPacketAudio(pktData);
}

//音声処理スレッドが存在する場合、この関数は音声処理スレッドによって処理される
//音声処理スレッドがなく、出力スレッドがあれば、出力スレッドにより処理される
//出力スレッドがなければメインエンコードスレッドが処理する
RGY_ERR RGYOutputAvcodec::WriteNextPacketAudio(AVPktMuxData *pktData) {
    pktData->samples = 0;
    AVMuxAudio *muxAudio = pktData->muxAudio;
    if (muxAudio == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get stream for input stream.\n"));
        m_Mux.format.streamError = true;
        av_packet_unref(&pktData->pkt);
        return RGY_ERR_NULL_PTR;
    }

    //AACBsfでのエラーを無音挿入で回避する(音声エンコード時のみ)
    bool bSetSilenceDueToAACBsfError = false;
    AVRational samplerate = { 1, muxAudio->streamIn->codecpar->sample_rate };
    //このパケットのサンプル数
    const int nSamples = (int)av_rescale_q(pktData->pkt.duration, muxAudio->streamIn->time_base, samplerate);
    if (muxAudio->AACBsfc) {
        auto sts = applyBitstreamFilterAAC(&pktData->pkt, muxAudio);
        //bitstream filterを正常に起動できなかった
        if (sts < RGY_ERR_NONE) {
            m_Mux.format.streamError = true;
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        //pktData->pkt.duration == 0 の場合はなにもせず終了する
        if (pktData->pkt.duration == 0) {
            av_packet_unref(&pktData->pkt);
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
        if (m_Mux.thread.thAudProcess.joinable()) {
            //ひとまず、ここでは処理せず、次のキューに回す
            AddAudQueue(pktData, (m_Mux.thread.thAudEncode.joinable()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        } else {
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
            WriteNextPacketProcessed(pktData);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    };
    if (!muxAudio->outCodecDecodeCtx) {
        pktData->samples = av_get_audio_frame_duration2(muxAudio->streamIn->codecpar, pktData->pkt.size);
        if (!pktData->samples) {
            pktData->samples = (int)av_rescale_q(pktData->pkt.duration, muxAudio->streamIn->time_base, samplerate);
            // 1/1000 timebaseは信じるに値しないので、frame_sizeがあればその値を使用する
            if (0 == av_cmp_q(muxAudio->streamIn->time_base, { 1, 1000 })
                && muxAudio->streamIn->codecpar->frame_size) {
                pktData->samples = muxAudio->streamIn->codecpar->frame_size;
            } else {
                //このdurationから計算したsampleが信頼できるか計算する
                //mkvではたまにptsの差分とdurationが一致しないことがある
                //ptsDiffが動画の1フレーム分より小さいときのみ対象とする (カット編集によるものを混同する可能性がある)
                int64_t ptsDiff = pktData->pkt.pts - muxAudio->lastPtsIn;
                if (0 < ptsDiff
                    && ptsDiff < av_rescale_q(1, av_inv_q(m_Mux.video.outputFps), samplerate)
                    && muxAudio->lastPtsIn != AV_NOPTS_VALUE
                    && 1 < std::abs(ptsDiff - pktData->pkt.duration)) {
                    //ptsの差分から計算しなおす
                    pktData->samples = (int)av_rescale_q(ptsDiff, muxAudio->streamIn->time_base, samplerate);
                }
            }
        }
        muxAudio->lastPtsIn = pktData->pkt.pts;
        writeOrSetNextPacketAudioProcessed(pktData);
    } else if (!(muxAudio->decodeError > muxAudio->ignoreDecodeError) && !muxAudio->encodeError) {
        vector<AVPktMuxData> audioFrames;
        if (bSetSilenceDueToAACBsfError) {
            //無音挿入
            AVFrame *silentFrame        = av_frame_alloc();
            silentFrame->nb_samples     = nSamples;
            silentFrame->channels       = muxAudio->outCodecDecodeCtx->channels;
            silentFrame->channel_layout = muxAudio->outCodecDecodeCtx->channel_layout;
            silentFrame->sample_rate    = muxAudio->outCodecDecodeCtx->sample_rate;
            silentFrame->format         = muxAudio->outCodecDecodeCtx->sample_fmt;
            silentFrame->pts            = pktData->pkt.pts;
            av_frame_get_buffer(silentFrame, 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
            av_samples_set_silence((uint8_t **)silentFrame->data, 0, silentFrame->nb_samples, silentFrame->channels, (AVSampleFormat)silentFrame->format);

            AVPktMuxData silentPkt = *pktData;
            silentPkt.type = MUX_DATA_TYPE_FRAME;
            silentPkt.frame = silentFrame;
            silentPkt.got_result = silentFrame && silentFrame->nb_samples > 0;
            audioFrames.push_back(silentPkt);
        } else {
            auto decodedFrames = AudioDecodePacket(muxAudio, &pktData->pkt);
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
    return WriteNextPacketAudioFrame(std::move(AudioFilterFrame(std::move(audioFrames))));
}

//フレームをresampleして後段に渡す
RGY_ERR RGYOutputAvcodec::WriteNextPacketAudioFrame(vector<AVPktMuxData> audioFrames) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    const bool bAudEncThread = m_Mux.thread.thAudEncode.joinable();
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
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.thAudEncode.joinable()) {
            //音声エンコードスレッドがこの関数を処理
            //AVPacketは字幕やnull終端パケットなどが流れてきたもの
            //これはそのまま出力キューに追加する
            AddAudQueue(pktData, AUD_QUEUE_OUT);
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        //音声エンコードスレッドが存在しない場合、ここにAVPacketは流れてこないはず
        return RGY_ERR_UNSUPPORTED;
    }
    auto encPktDatas = AudioEncodeFrame(pktData->muxAudio, pktData->frame);
    av_frame_free(&pktData->frame);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (m_Mux.thread.thAudProcess.joinable()) {
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

RGY_ERR RGYOutputAvcodec::ThreadFuncAudEncodeThread() {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    WaitForSingleObject(m_Mux.thread.heEventPktAddedAudEncode, INFINITE);
    while (!m_Mux.thread.thAudEncodeAbort) {
        if (!m_Mux.format.fileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (m_Mux.thread.qAudioFrameEncode.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_enc : nullptr)) {
                //音声エンコードを実行、出力キューに追加する
                WriteNextAudioFrame(&pktData);
            }
        }
        ResetEvent(m_Mux.thread.heEventPktAddedAudEncode);
        WaitForSingleObject(m_Mux.thread.heEventPktAddedAudEncode, 16);
    }
    {   //音声をすべてエンコード
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioFrameEncode.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_enc : nullptr)) {
            WriteNextAudioFrame(&pktData);
        }
    }
    SetEvent(m_Mux.thread.heEventClosingAudEncode);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::ThreadFuncAudThread() {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    WaitForSingleObject(m_Mux.thread.heEventPktAddedAudProcess, INFINITE);
    while (!m_Mux.thread.thAudProcessAbort) {
        if (!m_Mux.format.fileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (m_Mux.thread.qAudioPacketProcess.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_proc : nullptr)) {
                //音声処理を実行、出力キューに追加する
                WriteNextPacketInternal(&pktData, INT64_MAX);
            }
        }
        ResetEvent(m_Mux.thread.heEventPktAddedAudProcess);
        WaitForSingleObject(m_Mux.thread.heEventPktAddedAudProcess, 16);
    }
    {   //音声をすべて書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioPacketProcess.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_proc : nullptr)) {
            //音声処理を実行、出力キューに追加する
            WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    SetEvent(m_Mux.thread.heEventClosingAudProcess);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteThreadFunc() {
#if ENABLE_AVCODEC_OUT_THREAD
    //映像と音声の同期をとる際に、それをあきらめるまでの閾値
    const int nWaitThreshold = 32;
    //キューにデータが存在するか
    bool bAudioExists = false;
    bool bVideoExists = false;
    const auto fpsTimebase = av_inv_q(m_Mux.video.outputFps);
    const auto dtsThreshold = std::max<int64_t>(av_rescale_q(4, fpsTimebase, QUEUE_DTS_TIMEBASE), 4);
    //syncIgnoreDtsは映像と音声の同期を行う必要がないことを意味する
    //dtsThresholdを加算したときにオーバーフローしないよう、dtsThresholdを引いておく
    const int64_t syncIgnoreDts = INT64_MAX - dtsThreshold;
    int64_t audioDts = (m_Mux.audio.size()) ? 0 : syncIgnoreDts;
    int64_t videoDts = (m_Mux.video.streamOut) ? 0 : syncIgnoreDts;
    WaitForSingleObject(m_Mux.thread.heEventPktAddedOutput, INFINITE);
    //bThAudProcessは出力開始した後で取得する(この前だとまだ起動していないことがある)
    const bool bThAudProcess = m_Mux.thread.thAudProcess.joinable();
    auto writeProcessedPacket = [this](AVPktMuxData *pktData) {
        //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
        auto sts = RGY_ERR_NONE;
        const int trackFullID = ((uint32_t)pktData->pkt.flags >> 16);
        if (trackMediaType(trackFullID) == AVMEDIA_TYPE_AUDIO) {
            WriteNextPacketProcessed(pktData);
        } else {
            sts = WriteOtherPacket(&pktData->pkt);
        }
        return sts;
    };
    int audPacketsPerSec = 64;
    int nWaitAudio = 0;
    int nWaitVideo = 0;
    while (!m_Mux.thread.abortOutput) {
        do {
            if (!m_Mux.format.fileHeaderWritten) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //ヘッダー取得前に音声キューのサイズが足りず、エンコードが進まなくなってしまうことがある
                //キューのcapcityを増やすことでこれを回避する
                auto type = (m_Mux.thread.thAudEncode.joinable()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT;
                auto& qAudio = (type == AUD_QUEUE_OUT) ? m_Mux.thread.qAudioPacketOut : m_Mux.thread.qAudioFrameEncode;
                const auto nQueueCapacity = qAudio.capacity();
                if (qAudio.size() >= nQueueCapacity) {
                    qAudio.set_capacity(nQueueCapacity * 3 / 2);
                }
                //動画キューになにもなかったら再度待機する
                if (m_Mux.thread.qVideobitstream.size() == 0) {
                    break;
                }
            }
            //映像・音声の同期待ちが必要な場合、falseとなってループから抜けるよう、ここでfalseに設定する
            bAudioExists = false;
            bVideoExists = false;
            RGYBitstream bitstream = RGYBitstreamInit();
            while ((audioDts < 0 || videoDts <= audioDts + dtsThreshold)
                && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
                WriteNextFrameInternal(&bitstream, &videoDts);
                nWaitVideo = 0;
                const int log_level = RGY_LOG_TRACE;
                if (m_printMes && log_level >= m_printMes->getLogLevel()) {
                    AddMessage(log_level, _T("videoDts=%8lld: %s.\n"), videoDts, getTimestampString(videoDts, QUEUE_DTS_TIMEBASE).c_str());
                }
            }
            AVPktMuxData pktData = { 0 };
            while ((videoDts < 0 || audioDts <= videoDts + dtsThreshold)
                && false != (bAudioExists = m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr))) {
                if (pktData.muxAudio && pktData.muxAudio->streamIn) {
                    audPacketsPerSec = std::max(audPacketsPerSec, (int)(1.0 / (av_q2d(pktData.muxAudio->streamIn->time_base) * pktData.pkt.duration) + 0.5));
                    const auto videoDelay = (audioDts - videoDts) * av_q2d(QUEUE_DTS_TIMEBASE);
                    const auto streamQueueCapacity = (int)(audPacketsPerSec * std::max(5.0, videoDelay * 1.5) * std::max((int)m_Mux.audio.size(), 1) + 0.5);
                    if ((int)m_Mux.thread.qAudioPacketOut.capacity() < streamQueueCapacity) {
                        m_Mux.thread.qAudioPacketOut.set_capacity(streamQueueCapacity);
                    }
                }
                const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : syncIgnoreDts;
                //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
                (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
                //複数のstreamがあり得るので最大値をとる
                audioDts = (std::max)(audioDts, (std::max)(pktData.dts, m_Mux.thread.streamOutMaxDts.load()));
                nWaitAudio = 0;
                const int log_level = RGY_LOG_TRACE;
                if (m_printMes && log_level >= m_printMes->getLogLevel()) {
                    AddMessage(log_level, _T("audioDts=%8lld: %s, maxDst=%8lld.\n"), audioDts, getTimestampString(audioDts, QUEUE_DTS_TIMEBASE).c_str(), maxDts);
                }
            }
            //一定以上の動画フレームがキューにたまっており、音声キューになにもなければ、
            //音声を無視して動画フレームの処理を開始させる
            //音声が途中までしかなかったり、途中からしかなかったりする場合にこうした処理が必要
            const size_t videoPacketThreshold = std::min<size_t>(3072, m_Mux.thread.qVideobitstream.capacity()) - nWaitThreshold;
            if (m_Mux.thread.qAudioPacketOut.size() == 0 && m_Mux.thread.qVideobitstream.size() > videoPacketThreshold) {
                nWaitAudio++;
                if (nWaitAudio <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    //また、映像キューのサイズが足りないことが考えられるので、拡大する
                    m_Mux.thread.qVideobitstream.set_capacity(m_Mux.thread.qVideobitstream.capacity() + 50);
                    break;
                }
                audioDts = videoDts;
                AddMessage(RGY_LOG_TRACE, _T("audio not coming.\n"));
            }
            //一定以上の音声フレームがキューにたまっており、動画キューになにもなければ、
            //動画を無視して音声フレームの処理を開始させる
            const size_t audioPacketThreshold = std::min<size_t>(6144, m_Mux.thread.qAudioPacketOut.capacity()) - nWaitThreshold;
            if (m_Mux.thread.qVideobitstream.size() == 0 && m_Mux.thread.qAudioPacketOut.size() > audioPacketThreshold) {
                nWaitVideo++;
                if (nWaitVideo <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    //また、音声キューのサイズが足りないことが考えられるので、拡大する
                    m_Mux.thread.qAudioPacketOut.set_capacity(m_Mux.thread.qAudioPacketOut.capacity() * 3 / 2);
                    break;
                }
                videoDts = audioDts;
                AddMessage(RGY_LOG_TRACE, _T("video not coming.\n"));
            }
        } while (bAudioExists || bVideoExists); //両方のキューがひとまず空になるか、映像・音声の同期待ちが必要になるまで回す
                                                //次のフレーム・パケットが送られてくるまで待機する
        //キューの容量が両方とも半分以下なら、すこし待ってみる
        //一方、どちらかのキューが半分以上使われていれば、なるべく早く処理する必要がある
        if (   m_Mux.thread.qVideobitstream.size() / (double)m_Mux.thread.qVideobitstream.capacity() < 0.5
            && m_Mux.thread.qAudioPacketOut.size() / (double)m_Mux.thread.qAudioPacketOut.capacity() < 0.5) {
            ResetEvent(m_Mux.thread.heEventPktAddedOutput);
            WaitForSingleObject(m_Mux.thread.heEventPktAddedOutput, 16);
        } else {
            std::this_thread::yield();
        }
    }
    //メインループを抜けたことを通知する
    SetEvent(m_Mux.thread.heEventClosingOutput);
    m_Mux.thread.qAudioPacketOut.set_keep_length(0);
    m_Mux.thread.qVideobitstream.set_keep_length(0);
    bAudioExists = !m_Mux.thread.qAudioPacketOut.empty();
    bVideoExists = !m_Mux.thread.qVideobitstream.empty();
    //まずは映像と音声の同期をとって出力するためのループ
    while (bAudioExists && bVideoExists) {
        AVPktMuxData pktData = { 0 };
        while (audioDts <= videoDts + dtsThreshold
            && false != (bAudioExists = m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr))) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : INT64_MAX;
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
            //複数のstreamがあり得るので最大値をとる
            audioDts = (std::max)(audioDts, pktData.dts);
        }
        RGYBitstream bitstream = RGYBitstreamInit();
        while (videoDts <= audioDts + dtsThreshold
            && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr))) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
        bAudioExists = !m_Mux.thread.qAudioPacketOut.empty();
        bVideoExists = !m_Mux.thread.qVideobitstream.empty();
    }
    { //音声を書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_aud_out : nullptr)) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    { //動画を書き出す
        RGYBitstream bitstream = RGYBitstreamInit();
        while (m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.queueInfo) ? &m_Mux.thread.queueInfo->usage_vid_out : nullptr)) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
    }
#endif
    return (m_Mux.format.streamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

void RGYOutputAvcodec::WaitFin() {
    CloseThread();
}

HANDLE RGYOutputAvcodec::getThreadHandleOutput() {
#if ENABLE_AVCODEC_OUT_THREAD
    return (HANDLE)m_Mux.thread.thOutput.native_handle();
#else
    return NULL;
#endif
}

HANDLE RGYOutputAvcodec::getThreadHandleAudProcess() {
#if ENABLE_AVCODEC_OUT_THREAD && ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (HANDLE)m_Mux.thread.thAudProcess.native_handle();
#else
    return NULL;
#endif
}

HANDLE RGYOutputAvcodec::getThreadHandleAudEncode() {
#if ENABLE_AVCODEC_OUT_THREAD && ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (HANDLE)m_Mux.thread.thAudEncode.native_handle();
#else
    return NULL;
#endif
}

#if USE_CUSTOM_IO
int RGYOutputAvcodec::readPacket(uint8_t *buf, int buf_size) {
    return (int)_fread_nolock(buf, 1, buf_size, m_Mux.format.fpOutput);
}
int RGYOutputAvcodec::writePacket(uint8_t *buf, int buf_size) {
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
