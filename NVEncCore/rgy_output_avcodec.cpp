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

void RGYOutputAvcodec::CloseSubtitle(AVMuxSub *pMuxSub) {
    //close decoder
    if (pMuxSub->pOutCodecDecodeCtx) {
        avcodec_close(pMuxSub->pOutCodecDecodeCtx);
        av_free(pMuxSub->pOutCodecDecodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecDecodeCtx.\n"));
    }

    //close encoder
    if (pMuxSub->pOutCodecEncodeCtx) {
        avcodec_close(pMuxSub->pOutCodecEncodeCtx);
        av_free(pMuxSub->pOutCodecEncodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecEncodeCtx.\n"));
    }
    if (pMuxSub->pBuf) {
        av_free(pMuxSub->pBuf);
    }

    memset(pMuxSub, 0, sizeof(pMuxSub[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed subtitle.\n"));
}

void RGYOutputAvcodec::CloseAudio(AVMuxAudio *pMuxAudio) {
    //close decoder
    if (pMuxAudio->pOutCodecDecodeCtx
        && pMuxAudio->nInSubStream == 0) { //サブストリームのものは単なるコピーなので開放不要
        avcodec_close(pMuxAudio->pOutCodecDecodeCtx);
        av_free(pMuxAudio->pOutCodecDecodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecDecodeCtx.\n"));
    }

    //close encoder
    if (pMuxAudio->pOutCodecEncodeCtx) {
        avcodec_close(pMuxAudio->pOutCodecEncodeCtx);
        av_free(pMuxAudio->pOutCodecEncodeCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecEncodeCtx.\n"));
    }

    //close filter
    if (pMuxAudio->pFilterGraph) {
        avfilter_graph_free(&pMuxAudio->pFilterGraph);
    }

    if (pMuxAudio->pAACBsfc) {
        av_bsf_free(&pMuxAudio->pAACBsfc);
    }
    memset(pMuxAudio, 0, sizeof(pMuxAudio[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed audio.\n"));
}

void RGYOutputAvcodec::CloseVideo(AVMuxVideo *pMuxVideo) {
    if (m_Mux.video.fpTsLogFile) {
        fclose(m_Mux.video.fpTsLogFile);
    }
    m_Mux.video.timestampList.clear();
    if (m_Mux.video.pBsfc) {
        av_bsf_free(&m_Mux.video.pBsfc);
    }
    memset(pMuxVideo, 0, sizeof(pMuxVideo[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed video.\n"));
}

void RGYOutputAvcodec::CloseFormat(AVMuxFormat *pMuxFormat) {
    if (pMuxFormat->pFormatCtx) {
        if (!pMuxFormat->bStreamError) {
            av_write_trailer(pMuxFormat->pFormatCtx);
        }
#if USE_CUSTOM_IO
        if (!pMuxFormat->fpOutput) {
#endif
            avio_close(pMuxFormat->pFormatCtx->pb);
            AddMessage(RGY_LOG_DEBUG, _T("Closed AVIO Context.\n"));
#if USE_CUSTOM_IO
        }
#endif
        avformat_free_context(pMuxFormat->pFormatCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
#if USE_CUSTOM_IO
    if (pMuxFormat->fpOutput) {
        fflush(pMuxFormat->fpOutput);
        fclose(pMuxFormat->fpOutput);
        AddMessage(RGY_LOG_DEBUG, _T("Closed File Pointer.\n"));
    }

    if (pMuxFormat->pAVOutBuffer) {
        av_free(pMuxFormat->pAVOutBuffer);
    }

    if (pMuxFormat->pOutputBuffer) {
        free(pMuxFormat->pOutputBuffer);
    }
#endif //USE_CUSTOM_IO
    memset(pMuxFormat, 0, sizeof(pMuxFormat[0]));
    AddMessage(RGY_LOG_DEBUG, _T("Closed format.\n"));
}

void RGYOutputAvcodec::CloseQueues() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.bThAudEncodeAbort = true;
    m_Mux.thread.bThAudProcessAbort = true;
    m_Mux.thread.bAbortOutput = true;
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
    m_Mux.thread.bThAudEncodeAbort = true;
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
    m_Mux.thread.bThAudProcessAbort = true;
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
    m_Mux.thread.bAbortOutput = true;
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
    m_Mux.thread.bAbortOutput = false;
    m_Mux.thread.bThAudProcessAbort = false;
    m_Mux.thread.bThAudEncodeAbort = false;
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
    for (int i = 0; i < (int)m_Mux.sub.size(); i++) {
        CloseSubtitle(&m_Mux.sub[i]);
    }
    m_Mux.sub.clear();
    CloseVideo(&m_Mux.video);
    m_strOutputInfo.clear();
    m_pEncSatusInfo.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
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

AVCodecID RGYOutputAvcodec::PCMRequiresConversion(const AVCodecParameters *pCodecParm) {
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
        if (pcmConvertCodecs[i].first == pCodecParm->codec_id) {
            if (pcmConvertCodecs[i].second != AV_CODEC_ID_FIRST_AUDIO) {
                return pcmConvertCodecs[i].second;
            }
            switch (pCodecParm->bits_per_raw_sample) {
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

void RGYOutputAvcodec::SetExtraData(AVCodecParameters *pCodecParam, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (pCodecParam->extradata)
        av_free(pCodecParam->extradata);
    pCodecParam->extradata_size = size;
    pCodecParam->extradata      = (uint8_t *)av_malloc(pCodecParam->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(pCodecParam->extradata, data, size);
};

//音声のchannel_layoutを自動選択する
uint64_t RGYOutputAvcodec::AutoSelectChannelLayout(const uint64_t *pChannelLayout, const AVCodecContext *pSrcAudioCtx) {
    int srcChannels = av_get_channel_layout_nb_channels(pSrcAudioCtx->channel_layout);
    if (srcChannels == 0) {
        srcChannels = pSrcAudioCtx->channels;
    }
    if (pChannelLayout == nullptr) {
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

    for (int i = 0; pChannelLayout[i]; i++) {
        if (srcChannels == av_get_channel_layout_nb_channels(pChannelLayout[i])) {
            return pChannelLayout[i];
        }
    }
    return pChannelLayout[0];
}

int RGYOutputAvcodec::AutoSelectSamplingRate(const int *pSamplingRateList, int nSrcSamplingRate) {
    if (pSamplingRateList == nullptr) {
        return nSrcSamplingRate;
    }
    //一致するものがあれば、それを返す
    int i = 0;
    for (; pSamplingRateList[i]; i++) {
        if (nSrcSamplingRate == pSamplingRateList[i]) {
            return nSrcSamplingRate;
        }
    }
    //相対誤差が最も小さいものを選択する
    vector<double> diffrate(i);
    for (i = 0; pSamplingRateList[i]; i++) {
        diffrate[i] = std::abs(1 - pSamplingRateList[i] / (double)nSrcSamplingRate);
    }
    return pSamplingRateList[std::distance(diffrate.begin(), std::min_element(diffrate.begin(), diffrate.end()))];
}

AVSampleFormat RGYOutputAvcodec::AutoSelectSampleFmt(const AVSampleFormat *pSamplefmtList, const AVCodecContext *pSrcAudioCtx) {
    AVSampleFormat srcFormat = pSrcAudioCtx->sample_fmt;
    if (pSamplefmtList == nullptr) {
        return srcFormat;
    }
    if (srcFormat == AV_SAMPLE_FMT_NONE) {
        return pSamplefmtList[0];
    }
    for (int i = 0; pSamplefmtList[i] >= 0; i++) {
        if (srcFormat == pSamplefmtList[i]) {
            return pSamplefmtList[i];
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
        for (int i = 0; pSamplefmtList[i] >= 0; i++) {
            if (foundFormat->first == pSamplefmtList[i]) {
                return pSamplefmtList[i];
            }
        }
    }
    return pSamplefmtList[0];
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
RGY_ERR RGYOutputAvcodec::InitVideo(const VideoInfo *pVideoOutputInfo, const AvcodecWriterPrm *prm) {
    m_Mux.format.pFormatCtx->video_codec_id = getAVCodecId(pVideoOutputInfo->codec);
    if (m_Mux.format.pFormatCtx->video_codec_id == AV_CODEC_ID_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find codec id for video.\n"));
        return RGY_ERR_INVALID_CODEC;
    }
    m_Mux.format.pFormatCtx->oformat->video_codec = m_Mux.format.pFormatCtx->video_codec_id;
    if (NULL == (m_Mux.video.pCodec = avcodec_find_decoder(m_Mux.format.pFormatCtx->video_codec_id))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to codec for video codec %s.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.pFormatCtx->video_codec_id)).c_str());
        return RGY_ERR_INVALID_CODEC;
    }
    if (NULL == (m_Mux.video.pStreamOut = avformat_new_stream(m_Mux.format.pFormatCtx, m_Mux.video.pCodec))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for video.\n"));
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.video.nFPS = av_make_q(pVideoOutputInfo->fpsN, pVideoOutputInfo->fpsD);
    AddMessage(RGY_LOG_DEBUG, _T("output video stream fps: %d/%d\n"), m_Mux.video.nFPS.num, m_Mux.video.nFPS.den);

    m_Mux.video.pStreamOut->codecpar->codec_type              = AVMEDIA_TYPE_VIDEO;
    m_Mux.video.pStreamOut->codecpar->codec_id                = m_Mux.format.pFormatCtx->video_codec_id;
    m_Mux.video.pStreamOut->codecpar->width                   = pVideoOutputInfo->dstWidth;
    m_Mux.video.pStreamOut->codecpar->height                  = pVideoOutputInfo->dstHeight;
    m_Mux.video.pStreamOut->codecpar->format                  = csp_rgy_to_avpixfmt(pVideoOutputInfo->csp);
    m_Mux.video.pStreamOut->codecpar->level                   = pVideoOutputInfo->codecLevel;
    m_Mux.video.pStreamOut->codecpar->profile                 = pVideoOutputInfo->codecProfile;
    m_Mux.video.pStreamOut->codecpar->sample_aspect_ratio.num = pVideoOutputInfo->sar[0];
    m_Mux.video.pStreamOut->codecpar->sample_aspect_ratio.den = pVideoOutputInfo->sar[1];
    m_Mux.video.pStreamOut->codecpar->chroma_location         = (AVChromaLocation)clamp(pVideoOutputInfo->vui.chromaloc, 0, 6);
    m_Mux.video.pStreamOut->codecpar->field_order             = picstrcut_rgy_to_avfieldorder(pVideoOutputInfo->picstruct);
    m_Mux.video.pStreamOut->codecpar->video_delay             = pVideoOutputInfo->videoDelay;
    m_Mux.video.pStreamOut->sample_aspect_ratio.num           = pVideoOutputInfo->sar[0]; //mkvではこちらの指定も必要
    m_Mux.video.pStreamOut->sample_aspect_ratio.den           = pVideoOutputInfo->sar[1];
    if (pVideoOutputInfo->vui.descriptpresent) {
        m_Mux.video.pStreamOut->codecpar->color_space         = (AVColorSpace)pVideoOutputInfo->vui.matrix;
        m_Mux.video.pStreamOut->codecpar->color_primaries     = (AVColorPrimaries)pVideoOutputInfo->vui.colorprim;
        m_Mux.video.pStreamOut->codecpar->color_range         = (AVColorRange)(pVideoOutputInfo->vui.fullrange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG);
        m_Mux.video.pStreamOut->codecpar->color_trc           = (AVColorTransferCharacteristic)pVideoOutputInfo->vui.transfer;
    }
    if (0 > avcodec_open2(m_Mux.video.pStreamOut->codec, m_Mux.video.pCodec, NULL)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s for video.\n"), char_to_tstring(avcodec_get_name(m_Mux.format.pFormatCtx->video_codec_id)).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("opened video avcodec\n"));

    m_Mux.video.rBitstreamTimebase    = (av_isvalid_q(prm->rBitstreamTimebase)) ? prm->rBitstreamTimebase : HW_NATIVE_TIMEBASE;
    m_Mux.video.pStreamOut->time_base = (av_isvalid_q(prm->rBitstreamTimebase)) ? prm->rBitstreamTimebase : av_inv_q(m_Mux.video.nFPS);
    if (m_Mux.format.bIsMatroska) {
        m_Mux.video.pStreamOut->time_base = av_make_q(1, 1000);
    }

#if !ENCODER_NVENC
    if (pVideoOutputInfo->picstruct & RGY_PICSTRUCT_INTERLACED) {
        m_Mux.video.pStreamOut->time_base.den *= 2;
    }
#endif
    m_Mux.video.pStreamOut->start_time          = 0;
    m_Mux.video.bDtsUnavailable   = prm->bVideoDtsUnavailable;
    m_Mux.video.nInputFirstKeyPts = prm->nVideoInputFirstKeyPts;
    m_Mux.video.pTimestamp        = prm->pVidTimestamp;

    if (prm->pVideoInputStream) {
        m_Mux.video.inputStreamTimebase = prm->pVideoInputStream->time_base;
        m_Mux.video.pStreamOut->disposition = prm->pVideoInputStream->disposition;
        if (prm->pVideoInputStream->metadata) {
            auto language_data = av_dict_get(prm->pVideoInputStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
            if (language_data) {
                av_dict_set(&m_Mux.video.pStreamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Set Video language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
            }
        }
        int side_data_size = 0;
        auto side_data = av_stream_get_side_data(prm->pVideoInputStream, AV_PKT_DATA_DISPLAYMATRIX, &side_data_size);
        if (side_data) {
            unique_ptr<uint8_t, decltype(&av_freep)> side_data_copy((uint8_t *)av_malloc(side_data_size), av_freep);
            memcpy(side_data_copy.get(), side_data, side_data_size);
            auto rotation = av_display_rotation_get((const int *)side_data_copy.get());
            int err = av_stream_add_side_data(m_Mux.video.pStreamOut, AV_PKT_DATA_DISPLAYMATRIX, side_data_copy.get(), side_data_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rotation %d from input\n"), rotation);
                return RGY_ERR_INVALID_CALL;
            }
            AddMessage(RGY_LOG_DEBUG, _T("copied rotation %d from input\n"), rotation);
            side_data_copy.release();
        }
#if 0
        if (pVideoOutputInfo->codec == RGY_CODEC_HEVC && prm->pHEVCHdrSei == nullptr) {
            side_data = av_stream_get_side_data(prm->pVideoInputStream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, &side_data_size);
            if (side_data) {
                unique_ptr<uint8_t, decltype(&av_freep)> side_data_copy((uint8_t *)av_malloc(side_data_size), av_freep);
                memcpy(side_data_copy.get(), side_data, side_data_size);
                AVContentLightMetadata *coll = (AVContentLightMetadata *)side_data_copy.get();
                AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"), coll->MaxCLL, coll->MaxFALL);
                int err = av_stream_add_side_data(m_Mux.video.pStreamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, (uint8_t *)side_data_copy.get(), side_data_size);
                if (err < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                    return RGY_ERR_INVALID_CALL;
                }
                AddMessage(RGY_LOG_DEBUG, _T("copied AV_PKT_DATA_CONTENT_LIGHT_LEVEL from input\n"));
                side_data_copy.release();
            }
            side_data = av_stream_get_side_data(prm->pVideoInputStream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, &side_data_size);
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

                int err = av_stream_add_side_data(m_Mux.video.pStreamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, (uint8_t *)side_data_copy.get(), side_data_size);
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

    if (pVideoOutputInfo->codec == RGY_CODEC_HEVC && prm->pHEVCHdrSei != nullptr) {
        auto seiNal = prm->pHEVCHdrSei->gen_nal();
        if (seiNal.size() > 0) {
            m_Mux.video.seiNal.copy(seiNal.data(), (uint32_t)seiNal.size());

            const auto HEVCHdrSeiPrm = prm->pHEVCHdrSei->getprm();

            //streamのside dataとしてmasteringdisplay等を設定する
            //mkv等では、これをしておかないとヘッダ部分に情報が適切にセットされない場合がある
            size_t coll_size = 0;
            unique_ptr<AVContentLightMetadata, decltype(&av_freep)> coll(av_content_light_metadata_alloc(&coll_size), av_freep);
            unique_ptr<AVMasteringDisplayMetadata, decltype(&av_freep)> mastering(av_mastering_display_metadata_alloc(), av_freep);

            coll->MaxCLL = HEVCHdrSeiPrm.maxcll;
            coll->MaxFALL = HEVCHdrSeiPrm.maxfall;

            mastering->display_primaries[0][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[0], 50000);
            mastering->display_primaries[0][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[1], 50000);
            mastering->display_primaries[1][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[2], 50000);
            mastering->display_primaries[1][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[3], 50000);
            mastering->display_primaries[2][0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[4], 50000);
            mastering->display_primaries[2][1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[5], 50000);
            mastering->white_point[0] = av_make_q(HEVCHdrSeiPrm.masterdisplay[6], 50000);
            mastering->white_point[1] = av_make_q(HEVCHdrSeiPrm.masterdisplay[7], 50000);
            mastering->max_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[8], 10000);
            mastering->min_luminance = av_make_q(HEVCHdrSeiPrm.masterdisplay[9], 10000);
            mastering->has_primaries = 1;
            mastering->has_luminance = 1;

            AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"),
                coll->MaxCLL, coll->MaxFALL);
            AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
                av_q2d(mastering->display_primaries[0][0]),
                av_q2d(mastering->display_primaries[0][1]),
                av_q2d(mastering->display_primaries[1][0]),
                av_q2d(mastering->display_primaries[1][1]),
                av_q2d(mastering->display_primaries[2][0]),
                av_q2d(mastering->display_primaries[2][1]),
                av_q2d(mastering->white_point[0]), av_q2d(mastering->white_point[1]),
                av_q2d(mastering->max_luminance), av_q2d(mastering->min_luminance));

            int err = av_stream_add_side_data(m_Mux.video.pStreamOut, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, (uint8_t *)mastering.get(), sizeof(mastering.get()[0]));
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));
                return RGY_ERR_INVALID_CALL;
            }
            mastering.release(); //av_stream_add_side_dataされたデータはこちらで開放してはいけない
            AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_MASTERING_DISPLAY_METADATA\n"));

            err = av_stream_add_side_data(m_Mux.video.pStreamOut, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, (uint8_t *)coll.get(), coll_size);
            if (err < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
                return RGY_ERR_INVALID_CALL;
            }
            coll.release(); //av_stream_add_side_dataされたデータはこちらで開放してはいけない
            AddMessage(RGY_LOG_DEBUG, _T("set AV_PKT_DATA_CONTENT_LIGHT_LEVEL\n"));
        }
    }

    if ((ENCODER_NVENC
        && (pVideoOutputInfo->codec == RGY_CODEC_H264 || pVideoOutputInfo->codec == RGY_CODEC_HEVC)
        && pVideoOutputInfo->sar[0] * pVideoOutputInfo->sar[1] > 0)
        || (ENCODER_VCEENC
            && (pVideoOutputInfo->vui.format != 5
                || pVideoOutputInfo->vui.colorprim != 2
                || pVideoOutputInfo->vui.transfer != 2
                || pVideoOutputInfo->vui.matrix != 2
                || pVideoOutputInfo->vui.chromaloc != 0))) {
        const char *bsf_name = nullptr;
        switch (pVideoOutputInfo->codec) {
        case RGY_CODEC_H264: bsf_name = "h264_metadata"; break;
        case RGY_CODEC_HEVC: bsf_name = "hevc_metadata"; break;
        default:
            break;
        }
        if (bsf_name == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("invalid codec to set metadata filter.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        AddMessage(RGY_LOG_DEBUG, _T("start initialize %s filter...\n"), bsf_name);
        auto filter = av_bsf_get_by_name(bsf_name);
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find %s.\n"), bsf_name);
            return RGY_ERR_NOT_FOUND;
        }
        unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>> codecpar(avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

        codecpar->codec_type              = AVMEDIA_TYPE_VIDEO;
        codecpar->codec_id                = getAVCodecId(pVideoOutputInfo->codec);
        codecpar->width                   = pVideoOutputInfo->dstWidth;
        codecpar->height                  = pVideoOutputInfo->dstHeight;
        codecpar->format                  = csp_rgy_to_avpixfmt(pVideoOutputInfo->csp);
        codecpar->level                   = pVideoOutputInfo->codecLevel;
        codecpar->profile                 = pVideoOutputInfo->codecProfile;
        codecpar->sample_aspect_ratio.num = pVideoOutputInfo->sar[0];
        codecpar->sample_aspect_ratio.den = pVideoOutputInfo->sar[1];
        codecpar->chroma_location         = AVCHROMA_LOC_LEFT;
        codecpar->field_order             = picstrcut_rgy_to_avfieldorder(pVideoOutputInfo->picstruct);
        codecpar->video_delay             = pVideoOutputInfo->videoDelay;
        if (pVideoOutputInfo->vui.descriptpresent) {
            codecpar->color_space         = (AVColorSpace)pVideoOutputInfo->vui.matrix;
            codecpar->color_primaries     = (AVColorPrimaries)pVideoOutputInfo->vui.colorprim;
            codecpar->color_range         = (AVColorRange)(pVideoOutputInfo->vui.fullrange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG);
            codecpar->color_trc           = (AVColorTransferCharacteristic)pVideoOutputInfo->vui.transfer;
        }
        int ret = 0;
        if (0 > (ret = av_bsf_alloc(filter, &m_Mux.video.pBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        if (0 > (ret = avcodec_parameters_copy(m_Mux.video.pBsfc->par_in, codecpar.get()))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AVDictionary *bsfPrm = nullptr;
        unique_ptr<AVDictionary*, decltype(&av_dict_free)> bsfPrmDictDeleter(&bsfPrm, av_dict_free);
        if (ENCODER_NVENC) {
            char sar[128];
            sprintf_s(sar, "%d/%d", pVideoOutputInfo->sar[0], pVideoOutputInfo->sar[1]);
            av_dict_set(&bsfPrm, "sample_aspect_ratio", sar, 0);
            AddMessage(RGY_LOG_DEBUG, _T("set sar %d:%d by %s filter\n"), pVideoOutputInfo->sar[0], pVideoOutputInfo->sar[1], bsf_name);
        }
        if (ENCODER_VCEENC) {
            if (pVideoOutputInfo->vui.format != 5 /*undef*/) {
                av_dict_set_int(&bsfPrm, "video_format", pVideoOutputInfo->vui.format, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set video_format %d by %s filter\n"), pVideoOutputInfo->vui.format, bsf_name);
            }
            if (pVideoOutputInfo->vui.colorprim != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "colour_primaries", pVideoOutputInfo->vui.colorprim, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set colorprim %d by %s filter\n"), pVideoOutputInfo->vui.colorprim, bsf_name);
            }
            if (pVideoOutputInfo->vui.transfer != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "transfer_characteristics", pVideoOutputInfo->vui.transfer, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set transfer %d by %s filter\n"), pVideoOutputInfo->vui.transfer, bsf_name);
            }
            if (pVideoOutputInfo->vui.matrix != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "matrix_coefficients", pVideoOutputInfo->vui.matrix, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set matrix %d by %s filter\n"), pVideoOutputInfo->vui.matrix, bsf_name);
            }
            if (pVideoOutputInfo->vui.chromaloc != 0) {
                av_dict_set_int(&bsfPrm, "chroma_sample_loc_type", pVideoOutputInfo->vui.chromaloc, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set chromaloc %d by %s filter\n"), pVideoOutputInfo->vui.chromaloc, bsf_name);
            }
        }
        if (0 > (ret = av_opt_set_dict2(m_Mux.video.pBsfc, &bsfPrm, AV_OPT_SEARCH_CHILDREN))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (0 > (ret = av_bsf_init(m_Mux.video.pBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter\n"), bsf_name);
    }

    if (prm->pMuxVidTsLogFile) {
        if (_tfopen_s(&m_Mux.video.fpTsLogFile, prm->pMuxVidTsLogFile, _T("a"))) {
            AddMessage(RGY_LOG_WARN, _T("failed to open mux timestamp log file: \"%s\""), prm->pMuxVidTsLogFile);
            m_Mux.video.fpTsLogFile = NULL;
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Opened mux timestamp log file: \"%s\""), prm->pMuxVidTsLogFile);
            tstring strFileHeadSep;
            for (int i = 0; i < 78; i++) {
                strFileHeadSep += _T("-");
            }
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), strFileHeadSep.c_str());
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), m_Mux.format.pFilename);
            _ftprintf(m_Mux.video.fpTsLogFile, _T("%s\n"), strFileHeadSep.c_str());
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("output video stream timebase: %d/%d\n"), m_Mux.video.pStreamOut->time_base.num, m_Mux.video.pStreamOut->time_base.den);
    AddMessage(RGY_LOG_DEBUG, _T("bDtsUnavailable: %s\n"), (m_Mux.video.bDtsUnavailable) ? _T("on") : _T("off"));
    return RGY_ERR_NONE;
}
#pragma warning (pop)

//音声フィルタの初期化
RGY_ERR RGYOutputAvcodec::InitAudioFilter(AVMuxAudio *pMuxAudio, int channels, uint64_t channel_layout, int sample_rate, AVSampleFormat sample_fmt) {
    //必要ならfilterを初期化
    if ((!pMuxAudio->pFilterGraph && (
        //フィルタが初期化されていない場合
        pMuxAudio->pFilter
        || bSplitChannelsEnabled(pMuxAudio->pnStreamChannelSelect)
        || bSplitChannelsEnabled(pMuxAudio->pnStreamChannelOut)
        || pMuxAudio->pOutCodecDecodeCtx->frame_size != pMuxAudio->pOutCodecEncodeCtx->frame_size
        || pMuxAudio->nFilterInChannels      != channels
        || pMuxAudio->nFilterInChannelLayout != channel_layout
        || pMuxAudio->nFilterInSampleRate    != sample_rate
        || pMuxAudio->FilterInSampleFmt      != sample_fmt
        ))
        ||
        //フィルタがすでに初期化されている場合
        (  pMuxAudio->nFilterInChannels      != channels
        || pMuxAudio->nFilterInChannelLayout != channel_layout
        || pMuxAudio->nFilterInSampleRate    != sample_rate
        || pMuxAudio->FilterInSampleFmt      != sample_fmt
        )) {
        if (pMuxAudio->pFilterGraph) {
            //filterをflush
            auto filteredFrames = AudioFilterFrameFlush(pMuxAudio);
            WriteNextPacketToAudioSubtracks(filteredFrames);

            //filterをclose
            avfilter_graph_free(&pMuxAudio->pFilterGraph);
        }
        pMuxAudio->nFilterInChannels      = channels;
        pMuxAudio->nFilterInChannelLayout = channel_layout;
        pMuxAudio->nFilterInSampleRate    = sample_rate;
        pMuxAudio->FilterInSampleFmt      = sample_fmt;

        int ret = 0;
        pMuxAudio->pFilterGraph = avfilter_graph_alloc();
        av_opt_set_int(pMuxAudio->pFilterGraph, "threads", 1, 0);

        auto filterchain = tchar_to_string(pMuxAudio->pFilter);

        //チャンネルレイアウトの変更
        if (bSplitChannelsEnabled(pMuxAudio->pnStreamChannelSelect)
            && pMuxAudio->pnStreamChannelSelect[pMuxAudio->nInSubStream] != channel_layout
            && av_get_channel_layout_nb_channels(pMuxAudio->pnStreamChannelSelect[pMuxAudio->nInSubStream]) < channels) {
            //初期化
            for (int inChannel = 0; inChannel < _countof(pMuxAudio->channelMapping); inChannel++) {
                pMuxAudio->channelMapping[inChannel] = -1;
            }
            //オプションによって指定されている、入力音声から抽出するべき音声レイアウト
            const auto select_channel_layout = pMuxAudio->pnStreamChannelSelect[pMuxAudio->nInSubStream];
            const int select_channel_count = av_get_channel_layout_nb_channels(select_channel_layout);
            std::string channel_map = "pan=" + getChannelLayoutChar(select_channel_count, av_get_default_channel_layout(select_channel_count));
            for (int inChannel = 0; inChannel < select_channel_count; inChannel++) {
                //オプションによって指定されているチャンネルレイアウトから、抽出する音声のチャンネルを順に取得する
                //実際には、「オプションによって指定されているチャンネルレイアウト」が入力音声に存在しない場合がある
                auto select_channel = av_channel_layout_extract_channel(select_channel_layout, std::min(inChannel, select_channel_count-1));
                //対象のチャンネルのインデックスを取得する
                auto select_channel_index = av_get_channel_layout_channel_index(pMuxAudio->pOutCodecDecodeCtx->channel_layout, select_channel);
                if (select_channel_index < 0) {
                    //対応するチャンネルがもともとの入力音声ストリームにはない場合
                    const auto nChannels = (std::min)(inChannel, av_get_channel_layout_nb_channels(pMuxAudio->pOutCodecDecodeCtx->channel_layout)-1);
                    //入力音声のストリームから、抽出する音声のチャンネルを順に取得する
                    select_channel = av_channel_layout_extract_channel(pMuxAudio->pOutCodecDecodeCtx->channel_layout, nChannels);
                    //対象のチャンネルのインデックスを取得する
                    select_channel_index = av_get_channel_layout_channel_index(pMuxAudio->pOutCodecDecodeCtx->channel_layout, select_channel);
                }
                pMuxAudio->channelMapping[inChannel] = select_channel_index;
                channel_map += "|c" + std::to_string(inChannel) + "=c" + std::to_string(select_channel_index);
            }
            if (filterchain.length() > 0) filterchain += ",";
            filterchain += channel_map;
            if (RGY_LOG_DEBUG >= m_pPrintMes->getLogLevel()) {
                tstring channel_layout_str = strsprintf(_T("channel layout for track %d.%d:\n["), pMuxAudio->nInTrackId, pMuxAudio->nInSubStream);
                for (int inChannel = 0; inChannel < channels; inChannel++) {
                    channel_layout_str += strsprintf(_T("%4d"), pMuxAudio->channelMapping[inChannel]);
                }
                channel_layout_str += _T("]\n");
                AddMessage(RGY_LOG_DEBUG, channel_layout_str.c_str());
            }
        }

        if (filterchain.length() > 0) filterchain += ",";
        filterchain += strsprintf("aformat=sample_fmts=%s:sample_rates=%d:channel_layouts=0x%I64x",
            av_get_sample_fmt_name(pMuxAudio->pOutCodecEncodeCtx->sample_fmt),
            pMuxAudio->pOutCodecEncodeCtx->sample_rate,
            pMuxAudio->pOutCodecEncodeCtx->channel_layout);

        AVFilterInOut *filter_inputs = nullptr;
        AVFilterInOut *filter_outputs = nullptr;
        if (0 > (ret = avfilter_graph_parse2(pMuxAudio->pFilterGraph, filterchain.c_str(), &filter_inputs, &filter_outputs))) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to parse filter description: %s: \"%s\"\n"), qsv_av_err2str(ret).c_str(), pMuxAudio->pFilter);
            return RGY_ERR_INVALID_AUDIO_PARAM;
        }
        unique_ptr<AVFilterInOut, RGYAVDeleter<AVFilterInOut>> inputs(filter_inputs, RGYAVDeleter<AVFilterInOut>(avfilter_inout_free));
        unique_ptr<AVFilterInOut, RGYAVDeleter<AVFilterInOut>> outputs(filter_outputs, RGYAVDeleter<AVFilterInOut>(avfilter_inout_free));
        AddMessage(RGY_LOG_DEBUG, _T("Parsed filter: %s\n"), pMuxAudio->pFilter);
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
        const auto inName = strsprintf("in_track_%d.%d", pMuxAudio->nInTrackId, pMuxAudio->nInSubStream);
        if (0 > (ret = avfilter_graph_create_filter(&pMuxAudio->pFilterBufferSrcCtx, abuffersrc, inName.c_str(), inargs.c_str(), nullptr, pMuxAudio->pFilterGraph))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create abuffer: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNSUPPORTED;
        }
        if (0 > (ret = avfilter_link(pMuxAudio->pFilterBufferSrcCtx, 0, inputs->filter_ctx, inputs->pad_idx))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to link abuffer: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        inputs.reset();
        AddMessage(RGY_LOG_DEBUG, _T("filter linked with src buffer.\n"));

        //出力の設定
        const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
        const auto outName = strsprintf("out_track_%d.%d", pMuxAudio->nInTrackId, pMuxAudio->nInSubStream);
        if (0 > (ret = avfilter_graph_create_filter(&pMuxAudio->pFilterBufferSinkCtx, abuffersink, outName.c_str(), nullptr, nullptr, pMuxAudio->pFilterGraph))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNSUPPORTED;
        }
        if (0 > (ret = av_opt_set_int(pMuxAudio->pFilterBufferSinkCtx, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set channel counts to abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNSUPPORTED;
        }

        if (0 > (ret = avfilter_link(outputs->filter_ctx, outputs->pad_idx,
            pMuxAudio->pFilterBufferSinkCtx, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to link abuffersink: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("filter linked with sink buffer.\n"));
        outputs.reset();

        //パラメータ設定
        std::string swr_opts = "";
        if (pMuxAudio->nAudioResampler == RGY_RESAMPLER_SOXR) {
            swr_opts = "resampler=soxr";
        }
        pMuxAudio->pFilterGraph->scale_sws_opts = av_strdup(swr_opts.c_str());
        av_opt_set(pMuxAudio->pFilterGraph, "aresample_swr_opts", swr_opts.c_str(), 0);

        if (0 > (ret = avfilter_graph_config(pMuxAudio->pFilterGraph, nullptr))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to configure filter graph: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("filter config done, filter ready.\n"));

        av_buffersink_set_frame_size(pMuxAudio->pFilterBufferSinkCtx, pMuxAudio->pOutCodecEncodeCtx->frame_size);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::InitAudio(AVMuxAudio *pMuxAudio, AVOutputStreamPrm *pInputAudio, uint32_t nAudioIgnoreDecodeError) {
    pMuxAudio->pStreamIn = pInputAudio->src.pStream;
    AddMessage(RGY_LOG_DEBUG, _T("start initializing audio ouput...\n"));
    AddMessage(RGY_LOG_DEBUG, _T("output stream index %d, trackId %d.%d, delay %d, \n"), pInputAudio->src.nIndex, pInputAudio->src.nTrackId, pInputAudio->src.nSubStreamId, pMuxAudio->nDelaySamplesOfAudio);
    AddMessage(RGY_LOG_DEBUG, _T("samplerate %d, stream pkt_timebase %d/%d\n"), pMuxAudio->pStreamIn->codecpar->sample_rate, pMuxAudio->pStreamIn->time_base.num, pMuxAudio->pStreamIn->time_base.den);

    if (NULL == (pMuxAudio->pStreamOut = avformat_new_stream(m_Mux.format.pFormatCtx, NULL))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for audio.\n"));
        return RGY_ERR_NULL_PTR;
    }
    pMuxAudio->pDecodedFrameCache = nullptr;
    pMuxAudio->nIgnoreDecodeError = nAudioIgnoreDecodeError;
    pMuxAudio->nInTrackId = pInputAudio->src.nTrackId;
    pMuxAudio->nInSubStream = pInputAudio->src.nSubStreamId;
    pMuxAudio->nStreamIndexIn = pInputAudio->src.nIndex;
    pMuxAudio->nLastPtsIn = AV_NOPTS_VALUE;
    pMuxAudio->pFilter = pInputAudio->pFilter;
    memcpy(pMuxAudio->pnStreamChannelSelect, pInputAudio->src.pnStreamChannelSelect, sizeof(pInputAudio->src.pnStreamChannelSelect));
    memcpy(pMuxAudio->pnStreamChannelOut,    pInputAudio->src.pnStreamChannelOut,    sizeof(pInputAudio->src.pnStreamChannelOut));

    //音声がwavの場合、フォーマット変換が必要な場合がある
    AVCodecID codecId = AV_CODEC_ID_NONE;
    if (!avcodecIsCopy(pInputAudio->pEncodeCodec) || AV_CODEC_ID_NONE != (codecId = PCMRequiresConversion(pMuxAudio->pStreamIn->codecpar))) {
        //デコーダの作成は親ストリームのみ
        if (pMuxAudio->nInSubStream == 0) {
            //setup decoder
            if (NULL == (pMuxAudio->pOutCodecDecode = avcodec_find_decoder(pMuxAudio->pStreamIn->codecpar->codec_id))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), pInputAudio->src.pStream->codecpar->codec_id));
                AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
                return RGY_ERR_INVALID_CODEC;
            }
            if (NULL == (pMuxAudio->pOutCodecDecodeCtx = avcodec_alloc_context3(pMuxAudio->pOutCodecDecode))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), pInputAudio->src.pStream->codecpar->codec_id));
                return RGY_ERR_NULL_PTR;
            }
            int ret;
            if (0 > (ret = avcodec_parameters_to_context(pMuxAudio->pOutCodecDecodeCtx, pInputAudio->src.pStream->codecpar))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s\n"),
                    char_to_tstring(avcodec_get_name(pInputAudio->src.pStream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            }
            pMuxAudio->pOutCodecDecodeCtx->pkt_timebase = pInputAudio->src.pStream->time_base;
            SetExtraData(pMuxAudio->pOutCodecDecodeCtx, pInputAudio->src.pStream->codecpar->extradata, pInputAudio->src.pStream->codecpar->extradata_size);
            if (nullptr != strstr(pMuxAudio->pOutCodecDecode->name, "wma")) {
                pMuxAudio->pOutCodecDecodeCtx->block_align = pInputAudio->src.pStream->codecpar->block_align;
            }
            if (0 > (ret = avcodec_open2(pMuxAudio->pOutCodecDecodeCtx, pMuxAudio->pOutCodecDecode, NULL))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                    char_to_tstring(avcodec_get_name(pInputAudio->src.pStream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decoder opened\n"));
            AddMessage(RGY_LOG_DEBUG, _T("Audio Decode Info: %s, %dch[0x%02x], %.1fkHz, %s, %d/%d\n"), char_to_tstring(avcodec_get_name(pMuxAudio->pStreamIn->codecpar->codec_id)).c_str(),
                pMuxAudio->pOutCodecDecodeCtx->channels, (uint32_t)pMuxAudio->pOutCodecDecodeCtx->channel_layout, pMuxAudio->pOutCodecDecodeCtx->sample_rate / 1000.0,
                char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecDecodeCtx->sample_fmt)).c_str(),
                pMuxAudio->pOutCodecDecodeCtx->pkt_timebase.num, pMuxAudio->pOutCodecDecodeCtx->pkt_timebase.den);
        }

        if (codecId != AV_CODEC_ID_NONE) {
            //PCM encoder
            if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder(codecId))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
                return RGY_ERR_INVALID_CODEC;
            }
            pInputAudio->pEncodeCodec = RGY_AVCODEC_COPY;
        } else {
            if (avcodecIsAuto(pInputAudio->pEncodeCodec)) {
                //エンコーダを探す (自動)
                if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder(m_Mux.format.pOutputFmt->audio_codec))) {
                    AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), m_Mux.format.pOutputFmt->audio_codec));
                    AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return RGY_ERR_INVALID_CODEC;
                }
                AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s for audio track %d\n"), char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
            } else {
                //エンコーダを探す (指定のもの)
                if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder_by_name(tchar_to_string(pInputAudio->pEncodeCodec).c_str()))) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to find encoder for codec %s\n"), pInputAudio->pEncodeCodec);
                    AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return RGY_ERR_INVALID_CODEC;
                }
                AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s selected for audio track %d\n"), char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
            }
            pInputAudio->pEncodeCodec = _T("codec_something");
        }
        if (NULL == (pMuxAudio->pOutCodecEncodeCtx = avcodec_alloc_context3(pMuxAudio->pOutCodecEncode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return RGY_ERR_NULL_PTR;
        }

        //チャンネル選択の自動設定を反映
        for (int i = 0; i < _countof(pMuxAudio->pnStreamChannelSelect); i++) {
            if (pMuxAudio->pnStreamChannelSelect[i] == RGY_CHANNEL_AUTO) {
                pMuxAudio->pnStreamChannelSelect[i] = (pMuxAudio->pOutCodecDecodeCtx->channel_layout)
                    ? pMuxAudio->pOutCodecDecodeCtx->channel_layout
                    : av_get_default_channel_layout(pMuxAudio->pOutCodecDecodeCtx->channels);
            }
        }

        auto enc_channel_layout = AutoSelectChannelLayout(pMuxAudio->pOutCodecEncode->channel_layouts, pMuxAudio->pOutCodecDecodeCtx);
        //もしチャンネルの分離・変更があれば、それを反映してエンコーダの入力とする
        if (bSplitChannelsEnabled(pMuxAudio->pnStreamChannelOut)) {
            enc_channel_layout = pMuxAudio->pnStreamChannelOut[pMuxAudio->nInSubStream];
            if (enc_channel_layout == RGY_CHANNEL_AUTO) {
                auto channels = av_get_channel_layout_nb_channels(pMuxAudio->pnStreamChannelSelect[pMuxAudio->nInSubStream]);
                enc_channel_layout = av_get_default_channel_layout(channels);
            }
        }
        int enc_sample_rate = (pInputAudio->nSamplingRate) ? pInputAudio->nSamplingRate : pMuxAudio->pOutCodecDecodeCtx->sample_rate;
        //select samplefmt
        pMuxAudio->pOutCodecEncodeCtx->sample_fmt          = AutoSelectSampleFmt(pMuxAudio->pOutCodecEncode->sample_fmts, pMuxAudio->pOutCodecDecodeCtx);
        pMuxAudio->pOutCodecEncodeCtx->sample_rate         = AutoSelectSamplingRate(pMuxAudio->pOutCodecEncode->supported_samplerates, enc_sample_rate);
        pMuxAudio->pOutCodecEncodeCtx->channel_layout      = enc_channel_layout;
        pMuxAudio->pOutCodecEncodeCtx->channels            = av_get_channel_layout_nb_channels(enc_channel_layout);
        pMuxAudio->pOutCodecEncodeCtx->bits_per_raw_sample = pMuxAudio->pOutCodecDecodeCtx->bits_per_raw_sample;
        pMuxAudio->pOutCodecEncodeCtx->pkt_timebase        = av_make_q(1, pMuxAudio->pOutCodecDecodeCtx->sample_rate);
        if (!avcodecIsCopy(pInputAudio->pEncodeCodec)) {
            pMuxAudio->pOutCodecEncodeCtx->bit_rate        = ((pInputAudio->nBitrate) ? pInputAudio->nBitrate : AVQSV_DEFAULT_AUDIO_BITRATE) * 1000;
        }
        //音声プロファイルの設定
        if (pInputAudio->pEncodeCodecProfile != nullptr) {
            const int selected_profile = AudioGetCodecProfile(pInputAudio->pEncodeCodecProfile, pMuxAudio->pOutCodecEncodeCtx->codec_id);
            if (selected_profile == FF_PROFILE_UNKNOWN) {
                AddMessage(RGY_LOG_ERROR, _T("unknown profile \"%s\" for codec %s (audio track %d).\n"),
                    pInputAudio->pEncodeCodecProfile,
                    char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
            pMuxAudio->pOutCodecEncodeCtx->profile = selected_profile;
            AddMessage(RGY_LOG_DEBUG, _T("profile %d (%s) selected for codec %s (audio track %d)."),
                selected_profile, pInputAudio->pEncodeCodecProfile,
                char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
        }
        //音声エンコーダのオプションの設定
        AVDictionary *codecPrmDict = nullptr;
        unique_ptr<AVDictionary*, decltype(&av_dict_free)> codecPrmDictDeleter(&codecPrmDict, av_dict_free);
        unique_ptr<char, RGYAVDeleter<void>> prm_buf;
        if (pInputAudio->pEncodeCodecPrm) {
            int ret = av_dict_parse_string(&codecPrmDict, tchar_to_string(pInputAudio->pEncodeCodecPrm).c_str(), "=", ",", 0);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to parse param(s) for codec %s for audio track %d: %s\n"),
                    char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId, qsv_av_err2str(ret).c_str());
                AddMessage(RGY_LOG_ERROR, _T("  prm: %s\n"), pInputAudio->pEncodeCodecPrm);
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
            char *buf = nullptr;
            av_dict_get_string(codecPrmDict, &buf, '=', ',');
            prm_buf = unique_ptr<char, RGYAVDeleter<void>>(buf, RGYAVDeleter<void>(av_freep));
        }
        AddMessage(RGY_LOG_DEBUG, _T("Audio Encoder Param: %s, %dch[0x%02x], %.1fkHz, %s, %d (%s), %d/%d, %s\n"),
            char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(),
            pMuxAudio->pOutCodecEncodeCtx->channels, (uint32_t)pMuxAudio->pOutCodecEncodeCtx->channel_layout, pMuxAudio->pOutCodecEncodeCtx->sample_rate / 1000.0,
            char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecEncodeCtx->sample_fmt)).c_str(),
            pMuxAudio->pOutCodecEncodeCtx->profile, AudioGetCodecProfileStr(pMuxAudio->pOutCodecEncodeCtx->profile, pMuxAudio->pOutCodecEncodeCtx->codec_id).c_str(),
            pMuxAudio->pOutCodecEncodeCtx->pkt_timebase.num, pMuxAudio->pOutCodecEncodeCtx->pkt_timebase.den,
            char_to_tstring(prm_buf.get() ? prm_buf.get() : "default").c_str());
        if (pMuxAudio->pOutCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
            av_opt_set(pMuxAudio->pOutCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (0 > avcodec_open2(pMuxAudio->pOutCodecEncodeCtx, pMuxAudio->pOutCodecEncode, &codecPrmDict)) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
            return RGY_ERR_NULL_PTR;
        }
        if (codecPrmDict) {
            for (const AVDictionaryEntry *t = nullptr; (t = av_dict_get(codecPrmDict, "", t, AV_DICT_IGNORE_SUFFIX)) != nullptr;) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown option to audio encoder[%s]: %s=%s\n"),
                    char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(),
                    char_to_tstring(t->key).c_str(),
                    char_to_tstring(t->value).c_str());
                m_Mux.format.bStreamError = true;
                return RGY_ERR_INCOMPATIBLE_AUDIO_PARAM;
            }
        }

        pMuxAudio->nFilterInChannels      = av_get_channel_layout_nb_channels(pMuxAudio->pOutCodecEncodeCtx->channel_layout);
        pMuxAudio->nFilterInChannelLayout = pMuxAudio->pOutCodecEncodeCtx->channel_layout;
        pMuxAudio->nFilterInSampleRate    = pMuxAudio->pOutCodecEncodeCtx->sample_rate;
        pMuxAudio->FilterInSampleFmt      = pMuxAudio->pOutCodecEncodeCtx->sample_fmt;
        auto sts = InitAudioFilter(pMuxAudio,
            av_get_channel_layout_nb_channels(pMuxAudio->pOutCodecDecodeCtx->channel_layout),
            pMuxAudio->pOutCodecDecodeCtx->channel_layout,
            pMuxAudio->pOutCodecDecodeCtx->sample_rate,
            pMuxAudio->pOutCodecDecodeCtx->sample_fmt);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (pMuxAudio->pStreamIn->codecpar->codec_id == AV_CODEC_ID_AAC && pMuxAudio->pStreamIn->codecpar->extradata == NULL && m_Mux.video.pStreamOut) {
        AddMessage(RGY_LOG_DEBUG, _T("start initialize aac_adtstoasc filter...\n"));
        auto filter = av_bsf_get_by_name("aac_adtstoasc");
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find aac_adtstoasc.\n"));
            return RGY_ERR_NOT_FOUND;
        }
        int ret = 0;
        if (0 > (ret = av_bsf_alloc(filter, &pMuxAudio->pAACBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        if (0 > (ret = avcodec_parameters_copy(pMuxAudio->pAACBsfc->par_in, pMuxAudio->pStreamIn->codecpar))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        pMuxAudio->pAACBsfc->time_base_in = pMuxAudio->pStreamIn->time_base;
        if (0 > (ret = av_bsf_init(pMuxAudio->pAACBsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (pInputAudio->src.pktSample.data) {
            //mkvではavformat_write_headerまでにAVCodecContextにextradataをセットしておく必要がある
            for (AVPacket *inpkt = av_packet_clone(&pInputAudio->src.pktSample); 0 == av_bsf_send_packet(pMuxAudio->pAACBsfc, inpkt); inpkt = nullptr) {
                AVPacket outpkt = { 0 };
                av_init_packet(&outpkt);
                ret = av_bsf_receive_packet(pMuxAudio->pAACBsfc, &outpkt);
                if (ret == 0) {
                    if (pMuxAudio->pAACBsfc->par_out->extradata) {
                        SetExtraData(pMuxAudio->pStreamIn->codecpar, pMuxAudio->pAACBsfc->par_out->extradata, pMuxAudio->pAACBsfc->par_out->extradata_size);
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
    //下記のようにavcodec_copy_contextを使用するとavformat_write_header()が
    //Tag mp4a/0x6134706d incompatible with output codec id '86018' ([64][0][0][0])のようなエラーを出すことがある
    //そのため、必要な値だけをひとつづつコピーする
    //avcodec_copy_context(pMuxAudio->pStream->codec, srcCodecCtx);
    AVCodecParameters *srcCodecParam = avcodec_parameters_alloc();
    if (pMuxAudio->pOutCodecEncodeCtx) {
        avcodec_parameters_from_context(srcCodecParam, pMuxAudio->pOutCodecEncodeCtx);
    } else {
        avcodec_parameters_copy(srcCodecParam, pInputAudio->src.pStream->codecpar);
    }
    pMuxAudio->pStreamOut->codecpar->codec_type      = srcCodecParam->codec_type;
    pMuxAudio->pStreamOut->codecpar->codec_id        = srcCodecParam->codec_id;
    pMuxAudio->pStreamOut->codecpar->profile         = srcCodecParam->profile;
    pMuxAudio->pStreamOut->codecpar->frame_size      = srcCodecParam->frame_size;
    pMuxAudio->pStreamOut->codecpar->channels        = srcCodecParam->channels;
    pMuxAudio->pStreamOut->codecpar->channel_layout  = srcCodecParam->channel_layout;
    pMuxAudio->pStreamOut->codecpar->sample_rate     = srcCodecParam->sample_rate;
    pMuxAudio->pStreamOut->codecpar->format          = srcCodecParam->format;
    pMuxAudio->pStreamOut->codecpar->block_align     = srcCodecParam->block_align;
    if (srcCodecParam->extradata_size) {
        AddMessage(RGY_LOG_DEBUG, _T("set extradata from stream codec...\n"));
        SetExtraData(pMuxAudio->pStreamOut->codecpar, srcCodecParam->extradata, srcCodecParam->extradata_size);
    } else if (pMuxAudio->pStreamIn->codecpar->extradata_size) {
        //aac_adtstoascから得たヘッダをコピーする
        //これをしておかないと、avformat_write_headerで"Error parsing AAC extradata, unable to determine samplerate."という
        //意味不明なエラーメッセージが表示される
        AddMessage(RGY_LOG_DEBUG, _T("set extradata from original packet...\n"));
        SetExtraData(pMuxAudio->pStreamOut->codecpar, pMuxAudio->pStreamIn->codecpar->extradata, pMuxAudio->pStreamIn->codecpar->extradata_size);
    }
    pMuxAudio->pStreamOut->time_base = av_make_q(1, pMuxAudio->pStreamOut->codecpar->sample_rate);
    if (m_Mux.video.pStreamOut) {
        pMuxAudio->pStreamOut->start_time = (int)av_rescale_q(pInputAudio->src.nDelayOfStream, pMuxAudio->pStreamIn->time_base, pMuxAudio->pStreamOut->time_base);
        pMuxAudio->nDelaySamplesOfAudio = (int)pMuxAudio->pStreamOut->start_time;
        pMuxAudio->nLastPtsOut = pMuxAudio->pStreamOut->start_time;

        AddMessage(RGY_LOG_DEBUG, _T("delay      %6d (timabase %d/%d)\n"), pInputAudio->src.nDelayOfStream, pMuxAudio->pStreamIn->time_base.num, pMuxAudio->pStreamIn->time_base.den);
        AddMessage(RGY_LOG_DEBUG, _T("start_time %6d (timabase %d/%d)\n"), pMuxAudio->pStreamOut->start_time,  pMuxAudio->pStreamOut->codec->time_base.num, pMuxAudio->pStreamOut->codec->time_base.den);
    }
    if (srcCodecParam) {
        avcodec_parameters_free(&srcCodecParam);
    }
    pMuxAudio->pStreamOut->disposition = pInputAudio->src.pStream->disposition;
    if (pInputAudio->src.nSubStreamId != 0) {
        //substream(--audio-filterなどによる複製stream)の場合はデフォルトstreamではない
        pMuxAudio->pStreamOut->disposition &= (~AV_DISPOSITION_DEFAULT);
    }
    if (pInputAudio->src.pStream->metadata) {
        for (AVDictionaryEntry *pEntry = nullptr;
        nullptr != (pEntry = av_dict_get(pInputAudio->src.pStream->metadata, "", pEntry, AV_DICT_IGNORE_SUFFIX));) {
            av_dict_set(&pMuxAudio->pStreamOut->metadata, pEntry->key, pEntry->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Copy Audio Metadata: key %s, value %s\n"), char_to_tstring(pEntry->key).c_str(), char_to_tstring(pEntry->value).c_str());
        }
        auto language_data = av_dict_get(pInputAudio->src.pStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
        if (language_data) {
            av_dict_set(&pMuxAudio->pStreamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(RGY_LOG_DEBUG, _T("Set Audio language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::InitSubtitle(AVMuxSub *pMuxSub, AVOutputStreamPrm *pInputSubtitle) {
    AddMessage(RGY_LOG_DEBUG, _T("start initializing subtitle ouput...\n"));

    AVCodecID codecId = (pInputSubtitle->src.pStream)
        ? pInputSubtitle->src.pStream->codecpar->codec_id
        : (pInputSubtitle->src.caption2ass == FORMAT_ASS) ? AV_CODEC_ID_ASS : AV_CODEC_ID_SUBRIP;

    if (   0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mp4")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mov")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "3gp")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "3g2")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "psp")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "ipod")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "f4v")) {
        if (avcodec_descriptor_get(codecId)->props & AV_CODEC_PROP_TEXT_SUB) {
            //mp4はmov_text形式しか使用できない
            codecId = AV_CODEC_ID_MOV_TEXT;
            if (pInputSubtitle->src.pStream == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("--caption2ass is not supported when output format is mp4.\n"));
                return RGY_ERR_INVALID_FORMAT;
            }
            //if (pInputSubtitle->src.pStream == nullptr && pInputSubtitle->src.caption2ass != FORMAT_SRT) {
            //    AddMessage(RGY_LOG_ERROR, _T("When output format is mp4, please select \"srt\" for caption2ass format.\n"));
            //    return RGY_ERR_INVALID_FORMAT;
            //}
        }
    } else if (codecId == AV_CODEC_ID_MOV_TEXT) {
        codecId = AV_CODEC_ID_ASS;
    }

    auto copy_subtitle_header = [](AVCodecContext *pDstCtx, const AVCodecContext *pSrcCtx) {
        if (pSrcCtx->subtitle_header_size) {
            pDstCtx->subtitle_header_size = pSrcCtx->subtitle_header_size;
            pDstCtx->subtitle_header = (uint8_t *)av_mallocz(pDstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(pDstCtx->subtitle_header, pSrcCtx->subtitle_header, pSrcCtx->subtitle_header_size);
        }
    };

    auto srcCodecParam = unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>>(
        avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

    if (pInputSubtitle->src.pStream == nullptr) {
        //caption2assで生成した字幕を受け取る
        const auto src_codec_id = (pInputSubtitle->src.caption2ass == FORMAT_ASS) ? AV_CODEC_ID_ASS : AV_CODEC_ID_SUBRIP;
        const auto codec = avcodec_find_decoder(src_codec_id);
        if (nullptr == (pMuxSub->pStreamOut = avformat_new_stream(m_Mux.format.pFormatCtx, codec))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (src_codec_id == AV_CODEC_ID_ASS) {
            if (pInputSubtitle->src.subtitleHeader == nullptr || pInputSubtitle->src.subtitleHeaderSize == 0) {
                AddMessage(RGY_LOG_ERROR, _T("subtitle header unknown for track %d.\n"), pInputSubtitle->src.nTrackId);
                return RGY_ERR_NULL_PTR;
            }
            srcCodecParam->extradata_size = pInputSubtitle->src.subtitleHeaderSize;
            srcCodecParam->extradata = (uint8_t *)av_strdup((char *)pInputSubtitle->src.subtitleHeader);
        }
        srcCodecParam->codec_type = codec->type;
        srcCodecParam->codec_id   = codec->id;
    } else {
        avcodec_parameters_copy(srcCodecParam.get(), pInputSubtitle->src.pStream->codecpar);

        if (nullptr == (pMuxSub->pStreamOut = avformat_new_stream(m_Mux.format.pFormatCtx, avcodec_find_decoder(codecId)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("output stream index %d, pkt_timebase %d/%d, trackId %d\n"),
            pInputSubtitle->src.nIndex, pInputSubtitle->src.pStream->time_base.num, pInputSubtitle->src.pStream->time_base.den, pInputSubtitle->src.nTrackId);
    }
    if (srcCodecParam->codec_id != codecId || codecId == AV_CODEC_ID_MOV_TEXT) {
        //setup decoder
        if (nullptr == (pMuxSub->pOutCodecDecode = avcodec_find_decoder(srcCodecParam->codec_id))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), srcCodecParam->codec_id));
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
            return RGY_ERR_INVALID_CODEC;
        }
        if (nullptr == (pMuxSub->pOutCodecDecodeCtx = avcodec_alloc_context3(pMuxSub->pOutCodecDecode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), srcCodecParam->codec_id));
            return RGY_ERR_NULL_PTR;
        }
        //設定されていない必須情報があれば設定する
        pMuxSub->pOutCodecDecodeCtx->pkt_timebase = pInputSubtitle->src.timebase;
        SetExtraData(pMuxSub->pOutCodecDecodeCtx, srcCodecParam->extradata, srcCodecParam->extradata_size);
        int ret;
        if (0 > (ret = avcodec_open2(pMuxSub->pOutCodecDecodeCtx, pMuxSub->pOutCodecDecode, nullptr))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(srcCodecParam->codec_id)).c_str(),
            pMuxSub->pOutCodecDecodeCtx->width, pMuxSub->pOutCodecDecodeCtx->height);

        //エンコーダを探す
        if (nullptr == (pMuxSub->pOutCodecEncode = avcodec_find_encoder(codecId))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
            return RGY_ERR_INVALID_CODEC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s for subtitle track %d\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str(), pInputSubtitle->src.nTrackId);

        if (NULL == (pMuxSub->pOutCodecEncodeCtx = avcodec_alloc_context3(pMuxSub->pOutCodecEncode))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return RGY_ERR_NULL_PTR;
        }
        pMuxSub->pOutCodecEncodeCtx->time_base = av_make_q(1, 1000);

        //subtitle_headerをここで設定しないとavcodec_open2に失敗する
        //基本的にはass形式のヘッダーを設定する
        if (pInputSubtitle->src.pStream) {
            copy_subtitle_header(pMuxSub->pOutCodecEncodeCtx, pInputSubtitle->src.pStream->codec);
        } else if (pInputSubtitle->src.subtitleHeader) {
            pMuxSub->pOutCodecEncodeCtx->subtitle_header = (uint8_t *)av_strdup((char *)pInputSubtitle->src.subtitleHeader);
            pMuxSub->pOutCodecEncodeCtx->subtitle_header_size = pInputSubtitle->src.subtitleHeaderSize;
        }

        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Encoder Param: %s, %dx%d\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str(),
            pMuxSub->pOutCodecEncodeCtx->width, pMuxSub->pOutCodecEncodeCtx->height);
        if (pMuxSub->pOutCodecEncode->capabilities & AV_CODEC_CAP_EXPERIMENTAL) {
            //問答無用で使うのだ
            av_opt_set(pMuxSub->pOutCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (0 > (ret = avcodec_open2(pMuxSub->pOutCodecEncodeCtx, pMuxSub->pOutCodecEncode, nullptr))) {
            AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
            AddMessage(RGY_LOG_ERROR, _T(" %s\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Opened Subtitle Encoder Param: %s\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str());
        if (nullptr == (pMuxSub->pBuf = (uint8_t *)av_malloc(SUB_ENC_BUF_MAX_SIZE))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate buffer memory for subtitle encoding.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        pMuxSub->pStreamOut->codec->codec = pMuxSub->pOutCodecEncodeCtx->codec;
    }

    pMuxSub->nInTrackId     = pInputSubtitle->src.nTrackId;
    pMuxSub->nStreamIndexIn = pInputSubtitle->src.nIndex;
    pMuxSub->pStreamIn      = pInputSubtitle->src.pStream;
    pMuxSub->streamInTimebase = pInputSubtitle->src.timebase;

    if (pMuxSub->pOutCodecEncodeCtx) {
        avcodec_parameters_from_context(srcCodecParam.get(), pMuxSub->pOutCodecEncodeCtx);
    }
    avcodec_parameters_copy(pMuxSub->pStreamOut->codecpar, srcCodecParam.get());
    if (!pMuxSub->pStreamOut->codec->codec_tag) {
        uint32_t codec_tag = 0;
        if (!m_Mux.format.pFormatCtx->oformat->codec_tag
            || av_codec_get_id(m_Mux.format.pFormatCtx->oformat->codec_tag, srcCodecParam->codec_tag) == srcCodecParam->codec_id
            || !av_codec_get_tag2(m_Mux.format.pFormatCtx->oformat->codec_tag, srcCodecParam->codec_id, &codec_tag)) {
            pMuxSub->pStreamOut->codecpar->codec_tag = srcCodecParam->codec_tag;
        }
    }
    if (pInputSubtitle->src.pStream) {
        copy_subtitle_header(pMuxSub->pStreamOut->codec, (pMuxSub->pOutCodecEncodeCtx) ?  pMuxSub->pOutCodecEncodeCtx : pInputSubtitle->src.pStream->codec);
    } else if (pInputSubtitle->src.subtitleHeader != nullptr) {
        pMuxSub->pStreamOut->codec->subtitle_header = (uint8_t *)av_strdup((char *)pInputSubtitle->src.subtitleHeader);
        pMuxSub->pStreamOut->codec->subtitle_header_size = pInputSubtitle->src.subtitleHeaderSize;
    }
    pMuxSub->pStreamOut->time_base  = av_make_q(1, 1000);
    pMuxSub->pStreamOut->start_time = 0;
    if (pInputSubtitle->src.pStream) {
        pMuxSub->pStreamOut->disposition = pInputSubtitle->src.pStream->disposition;
        if (pInputSubtitle->src.pStream->metadata) {
            for (AVDictionaryEntry *pEntry = nullptr;
                nullptr != (pEntry = av_dict_get(pInputSubtitle->src.pStream->metadata, "", pEntry, AV_DICT_IGNORE_SUFFIX));) {
                av_dict_set(&pMuxSub->pStreamOut->metadata, pEntry->key, pEntry->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Copy Subtitle Metadata: key %s, value %s\n"), char_to_tstring(pEntry->key).c_str(), char_to_tstring(pEntry->value).c_str());
            }
            auto language_data = av_dict_get(pInputSubtitle->src.pStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
            if (language_data) {
                av_dict_set(&pMuxSub->pStreamOut->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
                AddMessage(RGY_LOG_DEBUG, _T("Set Subtitle language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::SetChapters(const vector<const AVChapter *>& pChapterList, bool bChapterNoTrim) {
    vector<AVChapter *> outChapters;
    for (int i = 0; i < (int)pChapterList.size(); i++) {
        int64_t start = (bChapterNoTrim) ? pChapterList[i]->start : AdjustTimestampTrimmed(pChapterList[i]->start, pChapterList[i]->time_base, pChapterList[i]->time_base, true);
        int64_t end   = (bChapterNoTrim) ? pChapterList[i]->end   : AdjustTimestampTrimmed(pChapterList[i]->end,   pChapterList[i]->time_base, pChapterList[i]->time_base, true);
        if (start < end) {
            AVChapter *pChap = (AVChapter *)av_mallocz(sizeof(pChap[0]));
            pChap->start     = start;
            pChap->end       = end;
            pChap->id        = pChapterList[i]->id;
            pChap->time_base = pChapterList[i]->time_base;
            av_dict_copy(&pChap->metadata, pChapterList[i]->metadata, 0);
            outChapters.push_back(pChap);
        }
    }
    if (outChapters.size() > 0) {
        m_Mux.format.pFormatCtx->nb_chapters = (uint32_t)outChapters.size();
        m_Mux.format.pFormatCtx->chapters = (AVChapter **)av_realloc_f(m_Mux.format.pFormatCtx->chapters, outChapters.size(), sizeof(m_Mux.format.pFormatCtx->chapters[0]) * outChapters.size());
        for (int i = 0; i < (int)outChapters.size(); i++) {
            m_Mux.format.pFormatCtx->chapters[i] = outChapters[i];

            AddMessage(RGY_LOG_DEBUG, _T("chapter #%d: id %d, start %I64d, end %I64d\n, timebase %d/%d\n"),
                outChapters[i]->id, outChapters[i]->start, outChapters[i]->end, outChapters[i]->time_base.num, outChapters[i]->time_base.den);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *option) {
    m_Mux.format.bStreamError = true;
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

    av_log_set_level((m_pPrintMes->getLogLevel() == RGY_LOG_DEBUG) ?  AV_LOG_DEBUG : RGY_AV_LOG_LEVEL);
    av_qsv_log_set(m_pPrintMes);
    for (const auto& stream : prm->inputStreamList) {
        if (stream.pFilter) {
            avfilter_register_all();
            break;
        }
    }

    if (prm->pOutputFormat != nullptr) {
        AddMessage(RGY_LOG_DEBUG, _T("output format specified: %s\n"), prm->pOutputFormat);
    }
    AddMessage(RGY_LOG_DEBUG, _T("output filename: \"%s\"\n"), strFileName);
    m_Mux.format.pFilename = strFileName;
    if (NULL == (m_Mux.format.pOutputFmt = av_guess_format((prm->pOutputFormat) ? tchar_to_string(prm->pOutputFormat).c_str() : NULL, filename.c_str(), NULL))) {
        AddMessage(RGY_LOG_ERROR,
            _T("failed to assume format from output filename.\n")
            _T("please set proper extension for output file, or specify format using option %s.\n"), (pVideoOutputInfo) ? _T("--format") : _T("--audio-file <format>:<filename>"));
        if (prm->pOutputFormat != nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Please use --check-formats to check available formats.\n"));
        }
        return RGY_ERR_INVALID_FORMAT;
    }
    if (0 == strcmp(filename.c_str(), "-")) {
        m_Mux.format.bIsPipe = true;
        m_bOutputIsStdout = true;
        filename = "pipe:1";
        AddMessage(RGY_LOG_DEBUG, _T("output is set to stdout\n"));
    } else if (filename.c_str() == strstr(filename.c_str(), R"(\\.\pipe\)")) {
        m_Mux.format.bIsPipe = true;
    }
    int err = avformat_alloc_output_context2(&m_Mux.format.pFormatCtx, m_Mux.format.pOutputFmt, nullptr, filename.c_str());
    if (m_Mux.format.pFormatCtx == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate format context: %s.\n"), qsv_av_err2str(err).c_str());
        return RGY_ERR_NULL_PTR;
    }
    m_Mux.format.bIsMatroska = 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "matroska");

#if USE_CUSTOM_IO
    if (m_Mux.format.bIsPipe || usingAVProtocols(filename, 1) || (m_Mux.format.pFormatCtx->oformat->flags & (AVFMT_NEEDNUMBER | AVFMT_NOFILE))) {
#endif //#if USE_CUSTOM_IO
        if (m_Mux.format.bIsPipe) {
            AddMessage(RGY_LOG_DEBUG, _T("output is pipe\n"));
#if defined(_WIN32) || defined(_WIN64)
            if (_setmode(_fileno(stdout), _O_BINARY) < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to switch stdout to binary mode.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
#endif //#if defined(_WIN32) || defined(_WIN64)
            if (0 == strcmp(filename.c_str(), "-")) {
                m_bOutputIsStdout = true;
                filename = "pipe:1";
                AddMessage(RGY_LOG_DEBUG, _T("output is set to stdout\n"));
            } else if (m_pPrintMes->getLogLevel() == RGY_LOG_DEBUG) {
                AddMessage(RGY_LOG_DEBUG, _T("file name is %sunc path.\n"), (PathIsUNC(strFileName)) ? _T("") : _T("not "));
                if (PathFileExists(strFileName)) {
                    AddMessage(RGY_LOG_DEBUG, _T("file already exists and will overwrite.\n"));
                }
            }
        }
        if (!(m_Mux.format.pFormatCtx->oformat->flags & AVFMT_NOFILE)) {
            if (0 > (err = avio_open2(&m_Mux.format.pFormatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to avio_open2 file \"%s\": %s\n"), char_to_tstring(filename, CP_UTF8).c_str(), qsv_av_err2str(err).c_str());
                return RGY_ERR_FILE_OPEN; // Couldn't open file
            }
        }
        AddMessage(RGY_LOG_DEBUG, _T("Opened file \"%s\".\n"), char_to_tstring(filename, CP_UTF8).c_str());
#if USE_CUSTOM_IO
    } else {
        m_Mux.format.nOutputBufferSize = clamp(prm->nBufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
        if (m_Mux.format.nOutputBufferSize == 0) {
            //出力バッファが0とされている場合、libavformat用の内部バッファも量を減らす
            m_Mux.format.nAVOutBufferSize = 128 * 1024;
            if (pVideoOutputInfo) {
                m_Mux.format.nAVOutBufferSize *= 4;
            }
        } else {
            m_Mux.format.nAVOutBufferSize = 1024 * 1024;
            if (pVideoOutputInfo) {
                m_Mux.format.nAVOutBufferSize *= 8;
            } else {
                //動画を出力しない(音声のみの場合)場合、バッファを減らす
                m_Mux.format.nOutputBufferSize /= 4;
            }
        }

        if (NULL == (m_Mux.format.pAVOutBuffer = (uint8_t *)av_malloc(m_Mux.format.nAVOutBufferSize))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate muxer buffer of %d MB.\n"), m_Mux.format.nAVOutBufferSize / (1024 * 1024));
            return RGY_ERR_MEMORY_ALLOC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("allocated internal buffer %d MB.\n"), m_Mux.format.nAVOutBufferSize / (1024 * 1024));
        CreateDirectoryRecursive(PathRemoveFileSpecFixed(strFileName).second.c_str());

        //"movflags:faststart"にするには、共有モードで開けるようにする必要がある
        m_Mux.format.fpOutput = _tfsopen(strFileName, _T("wb"), _SH_DENYWR);
        if (m_Mux.format.fpOutput == NULL) {
            errno_t error = errno;
            AddMessage(RGY_LOG_ERROR, _T("failed to open %soutput file \"%s\": %s.\n"), (pVideoOutputInfo) ? _T("") : _T("audio "), strFileName, _tcserror(error));
            return RGY_ERR_FILE_OPEN; // Couldn't open file
        }
        if (0 < (m_Mux.format.nOutputBufferSize = (uint32_t)malloc_degeneracy((void **)&m_Mux.format.pOutputBuffer, m_Mux.format.nOutputBufferSize, 1024 * 1024))) {
            setvbuf(m_Mux.format.fpOutput, m_Mux.format.pOutputBuffer, _IOFBF, m_Mux.format.nOutputBufferSize);
            AddMessage(RGY_LOG_DEBUG, _T("set external output buffer %d MB.\n"), m_Mux.format.nOutputBufferSize / (1024 * 1024));
        }
        if (NULL == (m_Mux.format.pFormatCtx->pb = avio_alloc_context(m_Mux.format.pAVOutBuffer, m_Mux.format.nAVOutBufferSize, 1, this, funcReadPacket, funcWritePacket, funcSeek))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return RGY_ERR_NULL_PTR;
        }
    }
#endif //#if USE_CUSTOM_IO

    m_Mux.trim = prm->trimList;

    if (pVideoOutputInfo) {
        RGY_ERR sts = InitVideo(pVideoOutputInfo, prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        AddMessage(RGY_LOG_DEBUG, _T("Initialized video output.\n"));
    }

    const int audioStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return prm.src.nTrackId > 0; });
    if (audioStreamCount) {
        m_Mux.audio.resize(audioStreamCount, { 0 });
        int iAudioIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            if (prm->inputStreamList[iStream].src.nTrackId > 0) {
                m_Mux.audio[iAudioIdx].nAudioResampler = prm->nAudioResampler;
                //サブストリームの場合は、デコーダ情報は親ストリームのものをコピーする
                if (prm->inputStreamList[iStream].src.nSubStreamId > 0) {
                    auto pAudioMuxStream = getAudioStreamData(prm->inputStreamList[iStream].src.nTrackId, 0);
                    if (pAudioMuxStream) {
                        //デコード情報をコピー
                        m_Mux.audio[iAudioIdx].pOutCodecDecode    = pAudioMuxStream->pOutCodecDecode;
                        m_Mux.audio[iAudioIdx].pOutCodecDecodeCtx = pAudioMuxStream->pOutCodecDecodeCtx;
                    } else {
                        AddMessage(RGY_LOG_ERROR, _T("Substream #%d found for track %d, but root stream not found.\n"),
                            prm->inputStreamList[iStream].src.nSubStreamId, prm->inputStreamList[iStream].src.nTrackId);
                        return RGY_ERR_UNDEFINED_BEHAVIOR;
                    }
                }
                RGY_ERR sts = InitAudio(&m_Mux.audio[iAudioIdx], &prm->inputStreamList[iStream], prm->nAudioIgnoreDecodeError);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized audio output - #%d: track %d, substream %d.\n"),
                    iAudioIdx, prm->inputStreamList[iStream].src.nTrackId, prm->inputStreamList[iStream].src.nSubStreamId);
                iAudioIdx++;
            }
        }
    }
    const int subStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return prm.src.nTrackId < 0; });
    if (subStreamCount) {
        m_Mux.sub.resize(subStreamCount, { 0 });
        int iSubIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            if (prm->inputStreamList[iStream].src.nTrackId < 0) {
                RGY_ERR sts = InitSubtitle(&m_Mux.sub[iSubIdx], &prm->inputStreamList[iStream]);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                AddMessage(RGY_LOG_DEBUG, _T("Initialized subtitle output - %d.\n"), iSubIdx);
                iSubIdx++;
            }
        }
    }

    SetChapters(prm->chapterList, prm->bChapterNoTrim);

    strcpy_s(m_Mux.format.pFormatCtx->filename, filename.c_str());
    if (m_Mux.format.pOutputFmt->flags & AVFMT_GLOBALHEADER) {
        if (m_Mux.video.pStreamOut) { m_Mux.video.pStreamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            if (m_Mux.audio[i].pStreamOut) { m_Mux.audio[i].pStreamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        }
        for (uint32_t i = 0; i < m_Mux.sub.size(); i++) {
            if (m_Mux.sub[i].pStreamOut) { m_Mux.sub[i].pStreamOut->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; }
        }
    }

    if (m_Mux.format.pFormatCtx->metadata) {
        av_dict_copy(&m_Mux.format.pFormatCtx->metadata, prm->pInputFormatMetadata, AV_DICT_DONT_OVERWRITE);
        av_dict_set(&m_Mux.format.pFormatCtx->metadata, "duration", NULL, 0);
        av_dict_set(&m_Mux.format.pFormatCtx->metadata, "creation_time", NULL, 0);
    }

    for (const auto& muxOpt : prm->vMuxOpt) {
        std::string optName = tchar_to_string(muxOpt.first);
        std::string optValue = tchar_to_string(muxOpt.second);
        if (0 > (err = av_dict_set(&m_Mux.format.pHeaderOptions, optName.c_str(), optValue.c_str(), 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set mux opt: %s = %s.\n"), muxOpt.first.c_str(), muxOpt.second.c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        AddMessage(RGY_LOG_DEBUG, _T("set mux opt: %s = %s.\n"), muxOpt.first.c_str(), muxOpt.second.c_str());
    }

    tstring mes = GetWriterMes();
    AddMessage(RGY_LOG_DEBUG, mes);
    m_strOutputInfo += mes;
    m_Mux.format.bStreamError = false;

#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.pQueueInfo = prm->pQueueInfo;
    //スレッドの使用数を設定
    if (prm->nOutputThread == RGY_OUTPUT_THREAD_AUTO) {
        prm->nOutputThread = 1;
    }
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    if (prm->nAudioThread == RGY_AUDIO_THREAD_AUTO) {
        prm->nAudioThread = 0;
    }
    m_Mux.thread.bEnableAudProcessThread = prm->nOutputThread > 0 && prm->nAudioThread > 0;
    m_Mux.thread.bEnableAudEncodeThread  = prm->nOutputThread > 0 && prm->nAudioThread > 1;
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    m_Mux.thread.bEnableOutputThread     = prm->nOutputThread > 0;
    if (m_Mux.thread.bEnableOutputThread) {
        AddMessage(RGY_LOG_DEBUG, _T("starting output thread...\n"));
        m_Mux.thread.bAbortOutput = false;
        m_Mux.thread.bThAudProcessAbort = false;
        m_Mux.thread.bThAudEncodeAbort = false;
        m_Mux.thread.qAudioPacketOut.init(8192, 256 * std::max(1, (int)m_Mux.audio.size())); //字幕のみコピーするときのため、最低でもある程度は確保する
        m_Mux.thread.qVideobitstream.init(4096, (std::max)(64, (m_Mux.video.nFPS.den) ? m_Mux.video.nFPS.num * 4 / m_Mux.video.nFPS.den : 0));
        m_Mux.thread.qVideobitstreamFreeI.init(256);
        m_Mux.thread.qVideobitstreamFreePB.init(3840);
        m_Mux.thread.heEventPktAddedOutput = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.heEventClosingOutput  = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_Mux.thread.thOutput = std::thread(&RGYOutputAvcodec::WriteThreadFunc, this);
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.bEnableAudProcessThread) {
            AddMessage(RGY_LOG_DEBUG, _T("starting audio process thread...\n"));
            m_Mux.thread.qAudioPacketProcess.init(8192, 512, 4);
            m_Mux.thread.heEventPktAddedAudProcess = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.heEventClosingAudProcess  = CreateEvent(NULL, TRUE, FALSE, NULL);
            m_Mux.thread.thAudProcess = std::thread(&RGYOutputAvcodec::ThreadFuncAudThread, this);
            if (m_Mux.thread.bEnableAudEncodeThread) {
                AddMessage(RGY_LOG_DEBUG, _T("starting audio encode thread...\n"));
                m_Mux.thread.qAudioFrameEncode.init(8192, 512, 4);
                m_Mux.thread.heEventPktAddedAudEncode = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.heEventClosingAudEncode  = CreateEvent(NULL, TRUE, FALSE, NULL);
                m_Mux.thread.thAudEncode = std::thread(&RGYOutputAvcodec::ThreadFuncAudEncodeThread, this);
            }
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    }
#endif //#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.video.pStreamOut == nullptr) {
        RGY_ERR sts = WriteFileHeader(nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_Mux.format.bFileHeaderWritten = true;
    }
    m_bInited = true;
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::AddH264HeaderToExtraData(const RGYBitstream *pBitstream) {
    std::vector<nal_info> nal_list = parse_nal_unit_h264(pBitstream->data(), pBitstream->size());
    const auto h264_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_SPS; });
    const auto h264_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_PPS; });
    const bool header_check = (nal_list.end() != h264_sps_nal) && (nal_list.end() != h264_pps_nal);
    if (header_check) {
        m_Mux.video.pStreamOut->codecpar->extradata_size = (int)(h264_sps_nal->size + h264_pps_nal->size);
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.pStreamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, h264_sps_nal->ptr, h264_sps_nal->size);
        memcpy(new_ptr + h264_sps_nal->size, h264_pps_nal->ptr, h264_pps_nal->size);
        if (m_Mux.video.pStreamOut->codecpar->extradata) {
            av_free(m_Mux.video.pStreamOut->codecpar->extradata);
        }
        m_Mux.video.pStreamOut->codecpar->extradata = new_ptr;
    }
    return RGY_ERR_NONE;
}

//extradataにHEVCのヘッダーを追加する
RGY_ERR RGYOutputAvcodec::AddHEVCHeaderToExtraData(const RGYBitstream *pBitstream) {
    std::vector<nal_info> nal_list = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
    const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
    const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
    const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
    const auto hevc_sei_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PREFIX_SEI; });
    const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
    if (header_check) {
        m_Mux.video.pStreamOut->codecpar->extradata_size = (int)(hevc_vps_nal->size + hevc_sps_nal->size + hevc_pps_nal->size);
        if (nal_list.end() != hevc_sei_nal) {
            m_Mux.video.pStreamOut->codecpar->extradata_size += (int)hevc_sei_nal->size;
        }
        uint8_t *new_ptr = (uint8_t *)av_malloc(m_Mux.video.pStreamOut->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(new_ptr, hevc_vps_nal->ptr, hevc_vps_nal->size);
        memcpy(new_ptr + hevc_vps_nal->size, hevc_sps_nal->ptr, hevc_sps_nal->size);
        memcpy(new_ptr + hevc_vps_nal->size + hevc_sps_nal->size, hevc_pps_nal->ptr, hevc_pps_nal->size);
        if (nal_list.end() != hevc_sei_nal) {
            memcpy(new_ptr + hevc_vps_nal->size + hevc_sps_nal->size + hevc_pps_nal->size, hevc_sei_nal->ptr, hevc_sei_nal->size);
        }
        if (m_Mux.video.pStreamOut->codecpar->extradata) {
            av_free(m_Mux.video.pStreamOut->codecpar->extradata);
        }
        m_Mux.video.pStreamOut->codecpar->extradata = new_ptr;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteFileHeader(const RGYBitstream *pBitstream) {
    if (m_Mux.video.pStreamOut && pBitstream) {
        RGY_ERR sts = RGY_ERR_NONE;
        switch (m_Mux.video.pStreamOut->codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            sts = AddH264HeaderToExtraData(pBitstream);
            break;
        case AV_CODEC_ID_HEVC:
            sts = AddHEVCHeaderToExtraData(pBitstream);
            break;
        default:
            break;
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to parse %s header.\n"), char_to_tstring(avcodec_get_name(m_Mux.video.pStreamOut->codecpar->codec_id)).c_str());
            return sts;
        }
    }

    //QSVEncCでエンコーダしたことを記録してみる
    //これは直接metadetaにセットする
    sprintf_s(m_Mux.format.metadataStr, ENCODER_NAME " (%s) %s", tchar_to_string(BUILD_ARCH_STR).c_str(), VER_STR_FILEVERSION);
    av_dict_set(&m_Mux.format.pFormatCtx->metadata, "encoding_tool", m_Mux.format.metadataStr, 0); //mp4
    //encoderではなく、encoding_toolを使用する。mp4はcomment, titleなどは設定可能, mkvではencode_byも可能

    //mp4のmajor_brandをisonからmp42に変更
    //これはmetadataではなく、avformat_write_headerのoptionsに渡す
    //この差ははっきり言って謎
    unique_ptr<AVDictionary*, decltype(&av_dict_free)> headerOptDeleter(&m_Mux.format.pHeaderOptions, av_dict_free);
    if (m_Mux.video.pStreamOut) {
        if (   0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mp4")
            || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mov")) {
            av_dict_set(&m_Mux.format.pHeaderOptions, "brand", "mp42", 0);
            AddMessage(RGY_LOG_DEBUG, _T("set format brand \"mp42\".\n"));

            //moovを先頭に
            av_dict_set(&m_Mux.format.pHeaderOptions, "movflags", "faststart", 0);
            AddMessage(RGY_LOG_DEBUG, _T("set faststart.\n"));
        }
    }

    //なんらかの問題があると、ここでよく死ぬ
    int ret = 0;
    if (0 > (ret = avformat_write_header(m_Mux.format.pFormatCtx, &m_Mux.format.pHeaderOptions))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write header for output file: %s\n"), qsv_av_err2str(ret).c_str());
        m_Mux.format.bStreamError = true;
        return RGY_ERR_UNKNOWN;
    }
    //不正なオプションを渡していないかチェック
    for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(m_Mux.format.pHeaderOptions, "", t, AV_DICT_IGNORE_SUFFIX));) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown option to muxer: %s=%s\n"),
            char_to_tstring(t->key).c_str(),
            char_to_tstring(t->value).c_str());
        m_Mux.format.bStreamError = true;
        return RGY_ERR_INVALID_PARAM;
    }

    av_dump_format(m_Mux.format.pFormatCtx, 0, m_Mux.format.pFormatCtx->filename, 1);

    //frame_sizeを表示
    for (const auto& audio : m_Mux.audio) {
        if (audio.pOutCodecDecodeCtx || audio.pOutCodecEncodeCtx) {
            tstring audioFrameSize = strsprintf(_T("audio track #%d:"), audio.nInTrackId);
            if (audio.pOutCodecDecodeCtx) {
                audioFrameSize += strsprintf(_T(" %s frame_size %d sample/byte"), char_to_tstring(audio.pOutCodecDecode->name).c_str(), audio.pOutCodecDecodeCtx->frame_size);
            }
            if (audio.pOutCodecEncodeCtx) {
                audioFrameSize += strsprintf(_T(" -> %s frame_size %d sample/byte"), char_to_tstring(audio.pOutCodecEncode->name).c_str(), audio.pOutCodecEncodeCtx->frame_size);
            }
            AddMessage(RGY_LOG_DEBUG, audioFrameSize);
        }
    }
    return RGY_ERR_NONE;
}

int64_t RGYOutputAvcodec::AdjustTimestampTrimmed(int64_t nTimeIn, AVRational timescaleIn, AVRational timescaleOut, bool lastValidFrame) {
    AVRational timescaleFps = av_inv_q(m_Mux.video.nFPS);
    const int vidFrameIdx = (int)av_rescale_q(nTimeIn, timescaleIn, timescaleFps);
    int cutFrames = 0;
    if (m_Mux.trim.size()) {
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

    if (m_Mux.video.pStreamOut) {
        add_mes(avcodec_get_name(m_Mux.video.pStreamOut->codecpar->codec_id));
        i_stream++;
    }
    for (const auto& audioStream : m_Mux.audio) {
        if (audioStream.pStreamOut) {
            std::string audiostr = (i_stream) ? ", " : "";
            if (audioStream.pOutCodecEncodeCtx) {
                //入力情報
                audiostr += strsprintf("#%d:%s/%s",
                    audioStream.nInTrackId,
                    audioStream.pOutCodecDecode->name,
                    getChannelLayoutChar(audioStream.pOutCodecDecodeCtx->channels, audioStream.pOutCodecDecodeCtx->channel_layout).c_str());
                if (audioStream.pnStreamChannelSelect[audioStream.nInSubStream] != 0) {
                    audiostr += strsprintf(":%s", getChannelLayoutChar(av_get_channel_layout_nb_channels(audioStream.pnStreamChannelSelect[audioStream.nInSubStream]), audioStream.pnStreamChannelSelect[audioStream.nInSubStream]).c_str());
                }
                //フィルタ情報
                if (audioStream.pFilter) {
                    audiostr += ":";
                    std::string filter_str;
                    auto filters = split(tchar_to_string(audioStream.pFilter, CP_UTF8), ",");
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
                    audioStream.pOutCodecEncode->name,
                    getChannelLayoutChar(audioStream.pOutCodecEncodeCtx->channels, audioStream.pOutCodecEncodeCtx->channel_layout).c_str(),
                    audioStream.pOutCodecEncodeCtx->bit_rate / 1000);
            } else {
                audiostr += strsprintf("%s", avcodec_get_name(audioStream.pStreamIn->codecpar->codec_id));
            }
            add_mes(audiostr);
            i_stream++;
        }
    }
    for (const auto& subtitleStream : m_Mux.sub) {
        if (subtitleStream.pStreamOut) {
            add_mes(std::string((i_stream) ? ", " : "") + strsprintf("sub#%d", std::abs(subtitleStream.nInTrackId)));
            i_stream++;
        }
    }
    if (m_Mux.format.pFormatCtx->nb_chapters > 0) {
        add_mes(std::string((i_stream) ? ", " : "") + "chap");
        i_stream++;
    }
    std::string output = " => ";
    output += m_Mux.format.pFormatCtx->oformat->name;
    //if (m_Mux.format.nOutputBufferSize) {
    //    output += strsprintf(" (%dMB buf)", m_Mux.format.nOutputBufferSize / (1024 * 1024));
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

RGY_ERR RGYOutputAvcodec::WriteNextFrame(RGYBitstream *pBitstream) {
#if ENABLE_AVCODEC_OUT_THREAD
    if (m_Mux.thread.thOutput.joinable()) {
        RGYBitstream copyStream = RGYBitstreamInit();
        bool bFrameI = (pBitstream->frametype() & RGY_FRAMETYPE_I) != 0;
        bool bFrameP = (pBitstream->frametype() & RGY_FRAMETYPE_P) != 0;
        //IフレームかPBフレームかでサイズが大きく違うため、空きのmfxBistreamは異なるキューで管理する
        auto& qVideoQueueFree = (bFrameI) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
        //空いているmfxBistreamを取り出す
        if (!qVideoQueueFree.front_copy_and_pop_no_lock(&copyStream) || copyStream.bufsize() < pBitstream->size()) {
            //空いているmfxBistreamがない、あるいはそのバッファサイズが小さい場合は、領域を取り直す
            const auto allocate_bytes = pBitstream->size() * ((bFrameI | bFrameP) ? 2 : 8);
            if (RGY_ERR_NONE != copyStream.init(allocate_bytes)) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for video bitstream output buffer, %sB.\n"), allocate_bytes);
                m_Mux.format.bStreamError = true;
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        //必要な情報をコピー
        copyStream.setDataflag(pBitstream->dataflag());
        copyStream.setPts(pBitstream->pts());
        copyStream.setDts(pBitstream->dts());
        copyStream.setDuration(pBitstream->duration());
        copyStream.setFrametype(pBitstream->frametype());
        copyStream.setSize(pBitstream->size());
        copyStream.setAvgQP(pBitstream->avgQP());
        copyStream.setOffset(0);
        memcpy(copyStream.bufptr(), pBitstream->data(), copyStream.size());
        //キューに押し込む
        if (!m_Mux.thread.qVideobitstream.push(copyStream)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory for video bitstream queue.\n"));
            m_Mux.format.bStreamError = true;
        }
        pBitstream->setSize(0);
        pBitstream->setOffset(0);
        SetEvent(m_Mux.thread.heEventPktAddedOutput);
        return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }
#endif
    int64_t dts = 0;
    return WriteNextFrameInternal(pBitstream, &dts);
}

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputAvcodec::WriteNextFrameInternal(RGYBitstream *pBitstream, int64_t *pWrittenDts) {
    if (!m_Mux.format.bFileHeaderWritten) {
#if ENCODER_QSV
        //HEVCエンコードでは、DecodeTimeStampが正しく設定されない
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC && pBitstream->dts() == MFX_TIMESTAMP_UNKNOWN) {
            m_Mux.video.bDtsUnavailable = true;
        }
#else
        m_Mux.video.bDtsUnavailable = true;
#endif
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC && m_Mux.video.seiNal.size() > 0) {
            RGYBitstream old = *pBitstream;
            std::vector<nal_info> nal_list = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
            const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
            const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
            const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
            const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
            if (header_check) {
                pBitstream->setSize(0);
                pBitstream->setOffset(0);
                pBitstream->append(hevc_vps_nal->ptr, hevc_vps_nal->size);
                pBitstream->append(hevc_sps_nal->ptr, hevc_sps_nal->size);
                pBitstream->append(hevc_pps_nal->ptr, hevc_pps_nal->size);
                pBitstream->append(&m_Mux.video.seiNal);
                for (const auto& nal : nal_list) {
                    if (nal.type != NALU_HEVC_VPS && nal.type != NALU_HEVC_SPS && nal.type != NALU_HEVC_PPS) {
                        pBitstream->append(nal.ptr, nal.size);
                    }
                }
            } else {
                AddMessage(RGY_LOG_ERROR, _T("Unexpected HEVC header.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            m_Mux.video.seiNal.clear();
        }
        RGY_ERR sts = WriteFileHeader(pBitstream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        //dts生成を初期化
        //何フレーム前からにすればよいかは、b-pyramid次第で異なるので、可能な限りエンコーダの情報を使用する
        if (!m_Mux.video.bDtsUnavailable) {
            m_VideoOutputInfo.videoDelay = -1 * (int)av_rescale_q(pBitstream->dts(), m_Mux.video.rBitstreamTimebase, av_inv_q(m_Mux.video.nFPS));
        }
        m_Mux.video.nFpsBaseNextDts = 0 - m_VideoOutputInfo.videoDelay;
        AddMessage(RGY_LOG_DEBUG, _T("calc dts, first dts %d x (timebase).\n"), m_Mux.video.nFpsBaseNextDts);

        const AVRational fpsTimebase = av_inv_q(m_Mux.video.nFPS);
        const AVRational streamTimebase = m_Mux.video.pStreamOut->codec->pkt_timebase;
        for (int i = m_Mux.video.nFpsBaseNextDts; i < 0; i++) {
            m_Mux.video.timestampList.add(av_rescale_q(i, fpsTimebase, streamTimebase));
        }
    }

#if ENCODER_QSV
    //QSVエンコーダでは、bitstreamからdurationの情報が取得できないので、別途取得する
    int64_t bs_duration = 0;
    if (m_Mux.video.pTimestamp) {
        while ((bs_duration = m_Mux.video.pTimestamp->get_and_pop(pBitstream->pts())) < 0) {
            Sleep(1);
        }
    }
#endif

    if (m_Mux.video.pBsfc) {
        int target_nal = 0;
        std::vector<nal_info> nal_list;
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
            target_nal = NALU_HEVC_SPS;
            nal_list = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
        } else if (m_VideoOutputInfo.codec == RGY_CODEC_H264) {
            target_nal = NALU_H264_SPS;
            nal_list = parse_nal_unit_h264(pBitstream->data(), pBitstream->size());
        }
        auto sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [target_nal](nal_info info) { return info.type == target_nal; });
        if (sps_nal != nal_list.end()) {
            AVPacket pkt = { 0 };
            av_init_packet(&pkt);
            av_new_packet(&pkt, (int)sps_nal->size);
            memcpy(pkt.data, sps_nal->ptr, sps_nal->size);
            int ret = 0;
            if (0 > (ret = av_bsf_send_packet(m_Mux.video.pBsfc, &pkt))) {
                av_packet_unref(&pkt);
                AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
                    char_to_tstring(m_Mux.video.pBsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            ret = av_bsf_receive_packet(m_Mux.video.pBsfc, &pkt);
            if (ret == AVERROR(EAGAIN)) {
                return RGY_ERR_NONE;
            } else if ((ret < 0 && ret != AVERROR_EOF) || pkt.size < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
                    char_to_tstring(m_Mux.video.pBsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            const auto new_data_size = pBitstream->size() + pkt.size - sps_nal->size;
            const auto sps_nal_offset = sps_nal->ptr - pBitstream->data();
            const auto next_nal_orig_offset = sps_nal_offset + sps_nal->size;
            const auto next_nal_new_offset = sps_nal_offset + pkt.size;
            const auto stream_orig_length = pBitstream->size();
            if ((decltype(new_data_size))pBitstream->bufsize() < new_data_size) {
                pBitstream->changeSize(new_data_size);
            } else if (pkt.size > (decltype(pkt.size))sps_nal->size) {
                pBitstream->trim();
            }
            memmove(pBitstream->data() + next_nal_new_offset, pBitstream->data() + next_nal_orig_offset, stream_orig_length - next_nal_orig_offset);
            memcpy(pBitstream->data() + sps_nal_offset, pkt.data, pkt.size);
            av_packet_unref(&pkt);
        }
    }

    AVPacket pkt = { 0 };
    av_init_packet(&pkt);
    av_new_packet(&pkt, (int)pBitstream->size());
    memcpy(pkt.data, pBitstream->data(), pBitstream->size());
    pkt.size = (int)pBitstream->size();

    const AVRational fpsTimebase = av_inv_q(m_Mux.video.nFPS);
    const AVRational streamTimebase = m_Mux.video.pStreamOut->codec->pkt_timebase;
    pkt.stream_index = m_Mux.video.pStreamOut->index;
    pkt.flags        = pBitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I) ? 1 : 0;
#if ENCODER_QSV
    //QSVエンコーダでは、bitstreamからdurationの情報が取得できないので、別途取得する
    pkt.duration = bs_duration;
#else
    pkt.duration = pBitstream->duration();
#endif
    pkt.pts = pBitstream->pts();
    if (av_cmp_q(m_Mux.video.rBitstreamTimebase, streamTimebase) != 0) {
        pkt.duration = av_rescale_q(pkt.duration, m_Mux.video.rBitstreamTimebase, streamTimebase);
        pkt.pts      = av_rescale_q(pkt.pts, m_Mux.video.rBitstreamTimebase, streamTimebase);
    }
    if (false && !m_Mux.video.bDtsUnavailable) {
        pkt.dts = av_rescale_q(av_rescale_q(pBitstream->dts(), m_Mux.video.rBitstreamTimebase, fpsTimebase), fpsTimebase, streamTimebase);
    } else {
        m_Mux.video.timestampList.add(pkt.pts);
        pkt.dts = m_Mux.video.timestampList.get_min_pts();
    }
    const auto pts = pkt.pts, dts = pkt.dts, duration = pkt.duration;
    *pWrittenDts = av_rescale_q(pkt.dts, streamTimebase, QUEUE_DTS_TIMEBASE);
    m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pkt);

    if (m_Mux.video.fpTsLogFile) {
        const uint32_t frameType = pBitstream->frametype();
        const TCHAR *pFrameTypeStr =
            (pBitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) ? _T("I") : (((pBitstream->frametype() & RGY_FRAMETYPE_B) == 0) ? _T("P") : _T("B"));
        _ftprintf(m_Mux.video.fpTsLogFile, _T("%s, %20lld, %20lld, %20lld, %20lld, %d, %7zd\n"), pFrameTypeStr, (lls)pBitstream->pts(), (lls)pBitstream->dts(), (lls)pts, (lls)dts, (int)duration, pBitstream->size());
    }
    m_pEncSatusInfo->SetOutputData(pBitstream->frametype(), pBitstream->size(), pBitstream->avgQP());
#if ENABLE_AVCODEC_OUT_THREAD
    //最初のヘッダーを書いたパケットはコピーではないので、キューに入れない
    if (m_Mux.thread.thOutput.joinable()) {
        //確保したメモリ領域を使いまわすためにキューに格納
        const auto frameI = (pBitstream->frametype() & (RGY_FRAMETYPE_IDR | RGY_FRAMETYPE_I)) != 0;
        auto& qVideoQueueFree = (frameI) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
        auto queueFavoredSize = (frameI) ? VID_BITSTREAM_QUEUE_SIZE_I : VID_BITSTREAM_QUEUE_SIZE_PB;
        if (qVideoQueueFree.size() > queueFavoredSize) {
            //あまり多すぎると無駄にメモリを使用するので減らす
            pBitstream->clear();
        } else {
            qVideoQueueFree.push(*pBitstream);
        }
    } else {
#endif
        pBitstream->setSize(0);
        pBitstream->setOffset(0);
#if ENABLE_AVCODEC_OUT_THREAD
    }
#endif
    m_Mux.format.bFileHeaderWritten = true;
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}
#pragma warning (pop)

RGY_ERR RGYOutputAvcodec::WriteNextFrame(RGYFrame *pSurface) {
    UNREFERENCED_PARAMETER(pSurface);
    return RGY_ERR_UNSUPPORTED;
}

vector<int> RGYOutputAvcodec::GetStreamTrackIdList() {
    vector<int> streamTrackId;
    streamTrackId.reserve(m_Mux.audio.size());
    for (auto audio : m_Mux.audio) {
        streamTrackId.push_back(audio.nInTrackId);
    }
    for (auto sub : m_Mux.sub) {
        streamTrackId.push_back(sub.nInTrackId);
    }
    return std::move(streamTrackId);
}

AVMuxAudio *RGYOutputAvcodec::getAudioPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (int16_t)(pkt->flags >> 16);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].nStreamIndexIn == streamIndex
            && m_Mux.audio[i].nInTrackId == inTrackId) {
            return &m_Mux.audio[i];
        }
    }
    return nullptr;
}

AVMuxAudio *RGYOutputAvcodec::getAudioStreamData(int nTrackId, int nSubStreamId) {
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].nInTrackId == nTrackId
            && m_Mux.audio[i].nInSubStream == nSubStreamId) {
            return &m_Mux.audio[i];
        }
    }
    return nullptr;
}

AVMuxSub *RGYOutputAvcodec::getSubPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (int16_t)(pkt->flags >> 16);
    for (int i = 0; i < (int)m_Mux.sub.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.sub[i].nStreamIndexIn == streamIndex
            && m_Mux.sub[i].nInTrackId == inTrackId) {
            return &m_Mux.sub[i];
        }
    }
    return NULL;
}

RGY_ERR RGYOutputAvcodec::applyBitstreamFilterAAC(AVPacket *pkt, AVMuxAudio *pMuxAudio) {
    int ret = 0;
    //毎回bitstream filterを初期化して、extradataに新しいヘッダを供給する
    //動画とmuxする際に必須
    av_bsf_free(&pMuxAudio->pAACBsfc);
    auto filter = av_bsf_get_by_name("aac_adtstoasc");
    if (filter == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find aac_adtstoasc.\n"));
        return RGY_ERR_NOT_FOUND;
    }
    if (0 > (ret = av_bsf_alloc(filter, &pMuxAudio->pAACBsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    if (0 > (ret = avcodec_parameters_copy(pMuxAudio->pAACBsfc->par_in, pMuxAudio->pStreamIn->codecpar))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    pMuxAudio->pAACBsfc->time_base_in = pMuxAudio->pStreamIn->time_base;
    if (0 > (ret = av_bsf_init(pMuxAudio->pAACBsfc))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init aac_adtstoasc: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = av_bsf_send_packet(pMuxAudio->pAACBsfc, pkt))) {
        av_packet_unref(pkt);
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to aac_adtstoasc bitstream filter: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(pMuxAudio->pAACBsfc, pkt);
    if (ret == AVERROR(EAGAIN)) {
        pkt->size = 0;
        pkt->duration = 0;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
        //最初のフレームとかでなければ、エラーを許容し、単に処理しないようにする
        //多くの場合、ここでのエラーはtsなどの最終音声フレームが不完全なことで発生する
        //先頭から連続30回エラーとなった場合はおかしいのでエラー終了するようにする
        if (pMuxAudio->nPacketWritten == 0) {
            pMuxAudio->nAACBsfErrorFromStart++;
            static int AACBSFFILTER_ERROR_THRESHOLD = 30;
            if (pMuxAudio->nAACBsfErrorFromStart > AACBSFFILTER_ERROR_THRESHOLD) {
                m_Mux.format.bStreamError = true;
                AddMessage(RGY_LOG_ERROR, _T("failed to run aac_adtstoasc bitstream filter for %d times: %s.\n"), AACBSFFILTER_ERROR_THRESHOLD, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
        }
        AddMessage(RGY_LOG_WARN, _T("failed to run aac_adtstoasc bitstream filter: %s.\n"), qsv_av_err2str(ret).c_str());
        pkt->duration = 0; //書き込み処理が行われないように
        return RGY_WRN_FILTER_SKIPPED;
    }
    pMuxAudio->nAACBsfErrorFromStart = 0;
    return RGY_ERR_NONE;
}

//音声/字幕パケットを実際に書き出す
// pMuxAudio ... [i]  pktに対応するストリーム情報
// pkt       ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// dts       ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVMuxAudio *pMuxAudio, AVPacket *pkt, int samples, int64_t *pWrittenDts) {
    if (pkt == nullptr || pkt->buf == nullptr) {
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            AudioFlushStream(&m_Mux.audio[i], pWrittenDts);
        }
        *pWrittenDts = INT64_MAX;
        AddMessage(RGY_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        return;
    }
    const AVRational samplerate = { 1, (pMuxAudio->pOutCodecEncodeCtx) ? pMuxAudio->pOutCodecEncodeCtx->sample_rate : pMuxAudio->pStreamIn->codecpar->sample_rate };
    if (!pMuxAudio->pOutCodecEncodeCtx) {
        samples = av_get_audio_frame_duration2(pMuxAudio->pStreamIn->codecpar, pkt->size);
        if (!samples) {
            samples = pMuxAudio->pOutCodecDecodeCtx->frame_size;
        }
        pkt->pts = av_rescale_delta(pMuxAudio->pStreamIn->time_base, pkt->pts, samplerate, samples, &pMuxAudio->dec_rescale_delta, pMuxAudio->pStreamOut->time_base);
    } else {
        pkt->pts = av_rescale_q(pkt->pts, pMuxAudio->pOutCodecEncodeCtx->time_base, pMuxAudio->pStreamOut->time_base);
    }
    if (samples) {
        //durationについて、sample数から出力ストリームのtimebaseに変更する
        pkt->stream_index = pMuxAudio->pStreamOut->index;
        pkt->flags        = AV_PKT_FLAG_KEY; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
        //pkt->pts          = //ここをそのままの処理にする av_rescale_q(pMuxAudio->nOutputSamples + pMuxAudio->nDelaySamplesOfAudio, samplerate, pMuxAudio->pStreamOut->time_base);
        if (m_Mux.video.pStreamOut) {
            pkt->pts -= av_rescale_q(m_Mux.video.nInputFirstKeyPts, m_Mux.video.inputStreamTimebase, pMuxAudio->pStreamOut->time_base);
        }
        pkt->dts          = pkt->pts;
        pkt->duration     = (int)av_rescale_q(samples, samplerate, pMuxAudio->pStreamOut->time_base);
        if (pkt->duration == 0)
            pkt->duration = (int)(pkt->pts - pMuxAudio->nLastPtsOut);
        pMuxAudio->nLastPtsOut = pkt->pts;
        *pWrittenDts = av_rescale_q(pkt->dts, pMuxAudio->pStreamOut->time_base, QUEUE_DTS_TIMEBASE);
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
        pMuxAudio->nOutputSamples += samples;
    } else {
        //av_interleaved_write_frameに渡ったパケットは開放する必要がないが、
        //それ以外は解放してやる必要がある
        av_packet_unref(pkt);
    }
}

//音声/字幕パケットを実際に書き出す (構造体版)
// pktData->pMuxAudio ... [i]  pktに対応するストリーム情報
// &pktData->pkt      ... [io] 書き出す音声/字幕パケット この関数でデータはav_interleaved_write_frameに渡されるか解放される
// pktData->samples   ... [i]  pktのsamples数 音声処理時のみ有効 / 字幕の際は0を渡すべき
// &pktData->dts      ... [o]  書き出したパケットの最終的なdtsをHW_NATIVE_TIMEBASEで返す
void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData) {
    WriteNextPacketProcessed(pktData->pMuxAudio, &pktData->pkt, pktData->samples, &pktData->dts);
}

void RGYOutputAvcodec::WriteNextPacketProcessed(AVPktMuxData *pktData, int64_t *pWrittenDts) {
    WriteNextPacketProcessed(pktData->pMuxAudio, &pktData->pkt, pktData->samples, &pktData->dts);
    *pWrittenDts = pktData->dts;
}

vector<unique_ptr<AVFrame, decltype(&av_frame_unref)>> RGYOutputAvcodec::AudioDecodePacket(AVMuxAudio *pMuxAudio, AVPacket *pkt) {
    vector<unique_ptr<AVFrame, decltype(&av_frame_unref)>> decodedFrames;
    if (pMuxAudio->nDecodeError > pMuxAudio->nIgnoreDecodeError) {
        return std::move(decodedFrames);
    }
    AVPacket pktInInfo;
    av_packet_copy_props(&pktInInfo, pkt);

    bool sent_packet = false;

    //最終的な出力フレーム
    int recv_ret = 0;
    for (;;) {
        unique_ptr<AVFrame, decltype(&av_frame_unref)> receivedData(nullptr, av_frame_unref);
        int send_ret = 0;
        //必ず一度はパケットを送る
        if (!sent_packet || pkt->size > 0) {
            sent_packet = true;
            //ひとつのパケットをデコーダに送る
            send_ret = avcodec_send_packet(pMuxAudio->pOutCodecDecodeCtx, pkt);
            //AVERROR(EAGAIN) -> パケットを送る前に受け取る必要がある (パケットが受け取られていないので解放しない)
            if (send_ret != AVERROR(EAGAIN)) {
                av_packet_unref(pkt);
            }
            if (send_ret == AVERROR_EOF) {
                AddMessage(RGY_LOG_DEBUG, _T("avcodec writer: failed to send packet to audio decoder, already flushed.\n"));
                break;
            }
        }
        if (send_ret < 0 && send_ret != AVERROR(EAGAIN)) {
            pMuxAudio->nDecodeError++;
        } else {
            receivedData = unique_ptr<AVFrame, decltype(&av_frame_unref)>(av_frame_alloc(), av_frame_unref);
            recv_ret = avcodec_receive_frame(pMuxAudio->pOutCodecDecodeCtx, receivedData.get());
            if (recv_ret == AVERROR(EAGAIN)   //もっとパケットを送る必要がある
                || recv_ret == AVERROR_EOF) { //最後まで読み込んだ
                break;
            }
            if (recv_ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to receive frame from audio decoder: %s.\n"), qsv_av_err2str(recv_ret).c_str());
                pMuxAudio->nDecodeError++;
            } else {
                pMuxAudio->nDecodeError = 0;
            }

            if (receivedData) {
                if (pkt && pkt->pts != AV_NOPTS_VALUE) {
                    receivedData->pts = pkt->pts;
                }
                if (receivedData->pts != AV_NOPTS_VALUE) {
                    //デコード後、samplerateを基準とする時間に変換する
                    auto timebase_samplerate = av_make_q(1, pMuxAudio->pOutCodecDecodeCtx->sample_rate);
                    const auto orig_pts = receivedData->pts;
                    receivedData->pts = av_rescale_delta(pMuxAudio->pStreamIn->time_base, orig_pts,
                        timebase_samplerate, receivedData->nb_samples, &pMuxAudio->dec_rescale_delta, timebase_samplerate);
                }
            }
        }

        if (pMuxAudio->nDecodeError) {
            if (pMuxAudio->nDecodeError <= pMuxAudio->nIgnoreDecodeError && receivedData && receivedData->pts != AV_NOPTS_VALUE) {
                //デコードエラーを無視する場合、入力パケットのサイズ分、無音を挿入する
                unique_ptr<AVFrame, decltype(&av_frame_unref)> silentFrame(av_frame_alloc(), av_frame_unref);
                AVRational samplerate = { 1, pMuxAudio->pOutCodecDecodeCtx->sample_rate };
                silentFrame->nb_samples     = (int)av_rescale_q(pktInInfo.duration, pMuxAudio->pStreamIn->time_base, samplerate);
                silentFrame->channels       = pMuxAudio->pOutCodecDecodeCtx->channels;
                silentFrame->channel_layout = pMuxAudio->pOutCodecDecodeCtx->channel_layout;
                silentFrame->sample_rate    = pMuxAudio->pOutCodecDecodeCtx->sample_rate;
                silentFrame->format         = pMuxAudio->pOutCodecDecodeCtx->sample_fmt;
                silentFrame->pts            = receivedData->pts;
                av_frame_get_buffer(silentFrame.get(), 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
                av_samples_set_silence((uint8_t **)silentFrame->data, 0, silentFrame->nb_samples, silentFrame->channels, (AVSampleFormat)silentFrame->format);
                decodedFrames.push_back(std::move(silentFrame));
            } else {
                AddMessage(RGY_LOG_ERROR, _T("avcodec writer: failed to decode audio #%d for %d times.\n"), pMuxAudio->nInTrackId, pMuxAudio->nDecodeError);
                m_Mux.format.bStreamError = true;
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
    for (const auto& pktData : inputFrames) {
        AVMuxAudio *pMuxAudio = pktData.pMuxAudio;
        if (pktData.pMuxAudio->pFilterGraph == nullptr) {
            //フィルタリングなし
            outputFrames.push_back(pktData);
        } else {
            if (pktData.pFrame != nullptr) {
                //音声入力フォーマットに変更がないか確認し、もしあればresamplerを再初期化する
                auto sts = InitAudioFilter(pMuxAudio, pktData.pFrame->channels, pktData.pFrame->channel_layout, pktData.pFrame->sample_rate, (AVSampleFormat)pktData.pFrame->format);
                if (sts != RGY_ERR_NONE) {
                    m_Mux.format.bStreamError = true;
                    break;
                }
            }
            //フィルターチェーンにフレームを追加
            if (av_buffersrc_add_frame_flags(pMuxAudio->pFilterBufferSrcCtx, pktData.pFrame, AV_BUFFERSRC_FLAG_PUSH) < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to feed the audio filtergraph\n"));
                m_Mux.format.bStreamError = true;
                av_frame_unref(pktData.pFrame);
                break;
            }
            for (;;) {
                unique_ptr<AVFrame, decltype(&av_frame_unref)> filteredFrame(av_frame_alloc(), av_frame_unref);
                int ret = av_buffersink_get_frame_flags(pMuxAudio->pFilterBufferSinkCtx, filteredFrame.get(), AV_BUFFERSINK_FLAG_NO_REQUEST);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    m_Mux.format.bStreamError = true;
                    break;
                }
                AVPktMuxData pktFiltered = pktData;
                pktFiltered.pFrame = filteredFrame.release();
                outputFrames.push_back(pktFiltered);
            }
            if (m_Mux.format.bStreamError) {
                break;
            }
        }
    }
    return outputFrames;
}

vector<AVPktMuxData> RGYOutputAvcodec::AudioFilterFrameFlush(AVMuxAudio *pMuxAudio) {
    vector<AVPktMuxData> flushFrame;
    AVPktMuxData pktData = { 0 };
    pktData.type = MUX_DATA_TYPE_FRAME;
    pktData.pFrame = nullptr;
    pktData.got_result = TRUE;
    pktData.pMuxAudio = pMuxAudio;
    flushFrame.push_back(pktData);
    return AudioFilterFrame(flushFrame);
}

//音声をエンコード
vector<AVPktMuxData> RGYOutputAvcodec::AudioEncodeFrame(AVMuxAudio *pMuxAudio, AVFrame *frame) {
    vector<AVPktMuxData> encPktDatas;

    if (frame) {
        //エンコーダのtimebaseに変換
        const auto timebase_filter = (pMuxAudio->pFilterGraph)
            ? av_buffersink_get_time_base(pMuxAudio->pFilterBufferSinkCtx)
            : av_make_q(1, pMuxAudio->pOutCodecDecodeCtx->sample_rate);
        frame->pts = av_rescale_q(frame->pts, timebase_filter, pMuxAudio->pOutCodecEncodeCtx->time_base);
    }
    int ret = avcodec_send_frame(pMuxAudio->pOutCodecEncodeCtx, frame);
    if (ret == AVERROR_EOF) {
        return encPktDatas;
    }
    if (ret < 0) {
        AddMessage(RGY_LOG_WARN, _T("avcodec writer: failed to send frame to audio encoder #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(ret).c_str());
        pMuxAudio->bEncodeError = true;
        return encPktDatas;
    }

    AVPktMuxData pktData;
    memset(&pktData.pkt, 0, sizeof(pktData.pkt)); //av_init_packetでsizeなどは初期化されないので0をセット
    pktData.type = MUX_DATA_TYPE_PACKET;
    pktData.pMuxAudio = pMuxAudio;
    while (ret >= 0) {
        av_init_packet(&pktData.pkt);
        ret = avcodec_receive_packet(pMuxAudio->pOutCodecEncodeCtx, &pktData.pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            AddMessage(RGY_LOG_WARN, _T("avcodec writer: failed to encode audio #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(ret).c_str());
            pMuxAudio->bEncodeError = true;
        }
        pktData.samples = (int)av_rescale_q(pktData.pkt.duration, pMuxAudio->pOutCodecEncodeCtx->pkt_timebase, { 1, pMuxAudio->pStreamIn->codecpar->sample_rate });
        encPktDatas.push_back(pktData);
    }
    return encPktDatas;
}

void RGYOutputAvcodec::AudioFlushStream(AVMuxAudio *pMuxAudio, int64_t *pWrittenDts) {
    while (pMuxAudio->pOutCodecDecodeCtx && !pMuxAudio->bEncodeError) {
        AVPacket pkt = { 0 };
        auto decodedFrames = AudioDecodePacket(pMuxAudio, &pkt);
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
            audPkt.pFrame = decodedFrames[i].release();
            audPkt.got_result = audPkt.pFrame && audPkt.pFrame->nb_samples > 0;
            audioFrames.push_back(audPkt);
        }
        //フィルタリングを行う
        WriteNextPacketToAudioSubtracks(std::move(audioFrames));
    }
    if (pMuxAudio->pFilterGraph) {
        WriteNextPacketAudioFrame(std::move(AudioFilterFrameFlush(pMuxAudio)));
    }
    while (pMuxAudio->pOutCodecEncodeCtx) {
        auto encPktDatas = AudioEncodeFrame(pMuxAudio, nullptr);
        if (encPktDatas.size() == 0) {
            break;
        }
        if (pMuxAudio->nDecodeError > pMuxAudio->nIgnoreDecodeError)
            break;
        for (auto& pktMux : encPktDatas) {
            WriteNextPacketProcessed(&pktMux, pWrittenDts);
        }
    }
}

RGY_ERR RGYOutputAvcodec::SubtitleTranscode(const AVMuxSub *pMuxSub, AVPacket *pkt) {
    //timescaleの変換が入ると、pts + duration > 次のpts となることがある
    //オリジナルのptsを使って再計算する
    const auto org_start_time = pkt->pts;
    const auto org_end_time = pkt->pts + pkt->duration;

    int got_sub = 0;
    AVSubtitle sub = { 0 };
    if (0 > avcodec_decode_subtitle2(pMuxSub->pOutCodecDecodeCtx, &sub, &got_sub, pkt)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
        m_Mux.format.bStreamError = true;
    }
    if (!pMuxSub->pBuf) {
        AddMessage(RGY_LOG_ERROR, _T("No buffer for encoding subtitle.\n"));
        m_Mux.format.bStreamError = true;
    }
    av_packet_unref(pkt);
    if (m_Mux.format.bStreamError)
        return RGY_ERR_UNKNOWN;
    if (!got_sub || sub.num_rects == 0)
        return RGY_ERR_NONE;

    //AV_CODEC_ID_DVB_SUBTITLEははじめりと終わりで2パケット
    const int nOutPackets = 1 + (pMuxSub->pOutCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE);
    for (int i = 0; i < nOutPackets; i++) {
        sub.pts               += av_rescale_q(sub.start_display_time, av_make_q(1, 1000), av_make_q(1, AV_TIME_BASE));
        sub.end_display_time  -= sub.start_display_time;
        sub.start_display_time = 0;
        if (i > 0) {
            sub.num_rects = 0;
        }

        int sub_out_size = avcodec_encode_subtitle(pMuxSub->pOutCodecEncodeCtx, pMuxSub->pBuf, SUB_ENC_BUF_MAX_SIZE, &sub);
        if (sub_out_size < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to encode subtitle.\n"));
            m_Mux.format.bStreamError = true;
            return RGY_ERR_UNKNOWN;
        }

        AVPacket pktOut;
        av_init_packet(&pktOut);
        pktOut.data = pMuxSub->pBuf;
        pktOut.stream_index = pMuxSub->pStreamOut->index;
        pktOut.size = sub_out_size;
        // pts + duration <= 次のptsとなるよう、オリジナルのptsを使って再計算する
        auto end_ts = av_rescale_q(org_end_time,   pMuxSub->pOutCodecDecodeCtx->pkt_timebase, pMuxSub->pStreamOut->time_base);
        pktOut.pts  = av_rescale_q(org_start_time, pMuxSub->pOutCodecDecodeCtx->pkt_timebase, pMuxSub->pStreamOut->time_base);
        pktOut.duration = (int)av_rescale_q(end_ts - pktOut.pts, pMuxSub->pOutCodecDecodeCtx->pkt_timebase, pMuxSub->pStreamOut->time_base);
        if (pMuxSub->pOutCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {
            pktOut.pts += 90 * ((i == 0) ? sub.start_display_time : sub.end_display_time);
        }
        pktOut.dts = pktOut.pts;
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pktOut);
    }
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::SubtitleWritePacket(AVPacket *pkt) {
    //字幕を処理する
    const AVMuxSub *pMuxSub = getSubPacketStreamData(pkt);
    const AVRational vid_pkt_timebase = av_isvalid_q(m_Mux.video.inputStreamTimebase) ? m_Mux.video.inputStreamTimebase : av_inv_q(m_Mux.video.nFPS);
    const int64_t pts_adjust = av_rescale_q(m_Mux.video.nInputFirstKeyPts, vid_pkt_timebase, pMuxSub->streamInTimebase);
    //ptsが存在しない場合はないものとすると、AdjustTimestampTrimmedの結果がAV_NOPTS_VALUEとなるのは、
    //Trimによりカットされたときのみ
    pkt->pts -= pts_adjust;
    const AVRational timebase_conv = (pMuxSub->pOutCodecDecodeCtx) ? pMuxSub->pOutCodecDecodeCtx->pkt_timebase : pMuxSub->pStreamOut->time_base;
    const int64_t pts_orig = pkt->pts;
    const int64_t pts_adj = AdjustTimestampTrimmed(std::max(INT64_C(0), pkt->pts), pMuxSub->streamInTimebase, pMuxSub->streamInTimebase, false);
    if (AV_NOPTS_VALUE != (pkt->pts = AdjustTimestampTrimmed(std::max(INT64_C(0), pkt->pts), pMuxSub->streamInTimebase, timebase_conv, false))) {
        //timescaleの変換を行い、負の値をとらないようにする
        pkt->dts = pkt->pts;
        if (pMuxSub->pOutCodecEncodeCtx) {
            return SubtitleTranscode(pMuxSub, pkt);
        }
        pkt->flags &= 0x0000ffff; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
        pkt->duration = (int)av_rescale_q(pkt->duration, pMuxSub->streamInTimebase, pMuxSub->pStreamOut->time_base);
        pkt->stream_index = pMuxSub->pStreamOut->index;
        pkt->pos = -1;
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
    }
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

AVPktMuxData RGYOutputAvcodec::pktMuxData(const AVPacket *pkt) {
    AVPktMuxData data = { 0 };
    data.type = MUX_DATA_TYPE_PACKET;
    if (pkt) {
        data.pkt = *pkt;
        data.pMuxAudio = getAudioPacketStreamData(pkt);
    }
    return data;
}

AVPktMuxData RGYOutputAvcodec::pktMuxData(AVFrame *pFrame) {
    AVPktMuxData data = { 0 };
    data.type = MUX_DATA_TYPE_FRAME;
    data.pFrame = pFrame;
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
            m_Mux.format.bStreamError = true;
        }
        SetEvent(heEventPktAdd);
        return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
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
            m_Mux.format.bStreamError = true;
        }
        SetEvent(heEventAdded);
        return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
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
    if (!m_Mux.format.bFileHeaderWritten) {
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
        return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
    }

    if (((int16_t)(pktData->pkt.flags >> 16)) < 0) {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        if (m_Mux.thread.thAudProcess.joinable()) {
            //音声処理を別スレッドでやっている場合は、字幕パケットもその流れに乗せてやる必要がある
            //ひとまず、ここでは処理せず、次のキューに回す
            return AddAudQueue(pktData, (m_Mux.thread.thAudEncode.joinable()) ? AUD_QUEUE_ENCODE : AUD_QUEUE_OUT);
        }
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
        return SubtitleWritePacket(&pktData->pkt);
    }
    return WriteNextPacketAudio(pktData);
}

//音声処理スレッドが存在する場合、この関数は音声処理スレッドによって処理される
//音声処理スレッドがなく、出力スレッドがあれば、出力スレッドにより処理される
//出力スレッドがなければメインエンコードスレッドが処理する
RGY_ERR RGYOutputAvcodec::WriteNextPacketAudio(AVPktMuxData *pktData) {
    pktData->samples = 0;
    AVMuxAudio *pMuxAudio = pktData->pMuxAudio;
    if (pMuxAudio == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get stream for input stream.\n"));
        m_Mux.format.bStreamError = true;
        av_packet_unref(&pktData->pkt);
        return RGY_ERR_NULL_PTR;
    }

    //AACBsfでのエラーを無音挿入で回避する(音声エンコード時のみ)
    bool bSetSilenceDueToAACBsfError = false;
    AVRational samplerate = { 1, pMuxAudio->pStreamIn->codecpar->sample_rate };
    //このパケットのサンプル数
    const int nSamples = (int)av_rescale_q(pktData->pkt.duration, pMuxAudio->pStreamIn->time_base, samplerate);
    if (pMuxAudio->pAACBsfc) {
        auto sts = applyBitstreamFilterAAC(&pktData->pkt, pMuxAudio);
        //bitstream filterを正常に起動できなかった
        if (sts < RGY_ERR_NONE) {
            m_Mux.format.bStreamError = true;
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        //pktData->pkt.duration == 0 の場合はなにもせず終了する
        if (pktData->pkt.duration == 0) {
            av_packet_unref(&pktData->pkt);
            //特にエラーでなければそのまま終了
            if (sts == RGY_ERR_NONE) {
                return RGY_ERR_NONE;
            }
            //先頭でエラーが出た場合は音声のDelayを増やすことで同期を保つ
            if (pMuxAudio->nPacketWritten == 0) {
                pMuxAudio->nDelaySamplesOfAudio += nSamples;
                return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
            }
            //音声エンコードしない場合はどうしようもないので終了
            if (!pMuxAudio->pOutCodecDecodeCtx || m_Mux.format.bStreamError) {
                return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
            }
            //音声エンコードの場合は無音挿入で時間をかせぐ
            bSetSilenceDueToAACBsfError = true;
        }
    }
    pMuxAudio->nPacketWritten++;
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
    if (!pMuxAudio->pOutCodecDecodeCtx) {
        pktData->samples = (int)av_rescale_q(pktData->pkt.duration, pMuxAudio->pStreamIn->time_base, samplerate);
        // 1/1000 timebaseは信じるに値しないので、frame_sizeがあればその値を使用する
        if (0 == av_cmp_q(pMuxAudio->pStreamIn->time_base, { 1, 1000 })
            && pMuxAudio->pStreamIn->codecpar->frame_size) {
            pktData->samples = pMuxAudio->pStreamIn->codecpar->frame_size;
        } else {
            //このdurationから計算したsampleが信頼できるか計算する
            //mkvではたまにptsの差分とdurationが一致しないことがある
            //ptsDiffが動画の1フレーム分より小さいときのみ対象とする (カット編集によるものを混同する可能性がある)
            int64_t ptsDiff = pktData->pkt.pts - pMuxAudio->nLastPtsIn;
            if (0 < ptsDiff
                && ptsDiff < av_rescale_q(1, av_inv_q(m_Mux.video.nFPS), samplerate)
                && pMuxAudio->nLastPtsIn != AV_NOPTS_VALUE
                && 1 < std::abs(ptsDiff - pktData->pkt.duration)) {
                //ptsの差分から計算しなおす
                pktData->samples = (int)av_rescale_q(ptsDiff, pMuxAudio->pStreamIn->time_base, samplerate);
            }
        }
        pMuxAudio->nLastPtsIn = pktData->pkt.pts;
        writeOrSetNextPacketAudioProcessed(pktData);
    } else if (!(pMuxAudio->nDecodeError > pMuxAudio->nIgnoreDecodeError) && !pMuxAudio->bEncodeError) {
        vector<AVPktMuxData> audioFrames;
        if (bSetSilenceDueToAACBsfError) {
            //無音挿入
            AVFrame *silentFrame        = av_frame_alloc();
            silentFrame->nb_samples     = nSamples;
            silentFrame->channels       = pMuxAudio->pOutCodecDecodeCtx->channels;
            silentFrame->channel_layout = pMuxAudio->pOutCodecDecodeCtx->channel_layout;
            silentFrame->sample_rate    = pMuxAudio->pOutCodecDecodeCtx->sample_rate;
            silentFrame->format         = pMuxAudio->pOutCodecDecodeCtx->sample_fmt;
            silentFrame->pts            = pktData->pkt.pts;
            av_frame_get_buffer(silentFrame, 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
            av_samples_set_silence((uint8_t **)silentFrame->data, 0, silentFrame->nb_samples, silentFrame->channels, (AVSampleFormat)silentFrame->format);

            AVPktMuxData silentPkt = *pktData;
            silentPkt.type = MUX_DATA_TYPE_FRAME;
            silentPkt.pFrame = silentFrame;
            silentPkt.got_result = silentFrame && silentFrame->nb_samples > 0;
            audioFrames.push_back(silentPkt);
        } else {
            auto decodedFrames = AudioDecodePacket(pMuxAudio, &pktData->pkt);
            for (size_t i = 0; i < decodedFrames.size(); i++) {
                AVPktMuxData audPkt = *pktData;
                audPkt.type = MUX_DATA_TYPE_FRAME;
                audPkt.pFrame = decodedFrames[i].release();
                audPkt.got_result = audPkt.pFrame && audPkt.pFrame->nb_samples > 0;
                audioFrames.push_back(audPkt);
            }
        }
        WriteNextPacketToAudioSubtracks(std::move(audioFrames));
    }
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

//フィルタリング後のパケットをサブトラックに分配する
RGY_ERR RGYOutputAvcodec::WriteNextPacketToAudioSubtracks(vector<AVPktMuxData> audioFrames) {
    const auto origPkts = audioFrames.size();
    for (size_t i = 0; i < origPkts; i++) {
        //サブストリームが存在すれば、frameをコピーしてそれぞれに渡す
        AVMuxAudio *pMuxAudioSubStream = nullptr;
        for (int iSubStream = 1; nullptr != (pMuxAudioSubStream = getAudioStreamData(audioFrames[i].pMuxAudio->nInTrackId, iSubStream)); iSubStream++) {
            auto pktDataCopy = audioFrames[i];
            pktDataCopy.pMuxAudio = pMuxAudioSubStream;
            pktDataCopy.pFrame = (audioFrames[i].pFrame) ? av_frame_clone(audioFrames[i].pFrame) : nullptr;
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
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
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
    auto encPktDatas = AudioEncodeFrame(pktData->pMuxAudio, pktData->pFrame);
    av_frame_unref(pktData->pFrame);
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
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::ThreadFuncAudEncodeThread() {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    WaitForSingleObject(m_Mux.thread.heEventPktAddedAudEncode, INFINITE);
    while (!m_Mux.thread.bThAudEncodeAbort) {
        if (!m_Mux.format.bFileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (m_Mux.thread.qAudioFrameEncode.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_enc : nullptr)) {
                //音声エンコードを実行、出力キューに追加する
                WriteNextAudioFrame(&pktData);
            }
        }
        ResetEvent(m_Mux.thread.heEventPktAddedAudEncode);
        WaitForSingleObject(m_Mux.thread.heEventPktAddedAudEncode, 16);
    }
    {   //音声をすべてエンコード
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioFrameEncode.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_enc : nullptr)) {
            WriteNextAudioFrame(&pktData);
        }
    }
    SetEvent(m_Mux.thread.heEventClosingAudEncode);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::ThreadFuncAudThread() {
#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    WaitForSingleObject(m_Mux.thread.heEventPktAddedAudProcess, INFINITE);
    while (!m_Mux.thread.bThAudProcessAbort) {
        if (!m_Mux.format.bFileHeaderWritten) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            AVPktMuxData pktData = { 0 };
            while (m_Mux.thread.qAudioPacketProcess.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_proc : nullptr)) {
                //音声処理を実行、出力キューに追加する
                WriteNextPacketInternal(&pktData, INT64_MAX);
            }
        }
        ResetEvent(m_Mux.thread.heEventPktAddedAudProcess);
        WaitForSingleObject(m_Mux.thread.heEventPktAddedAudProcess, 16);
    }
    {   //音声をすべて書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioPacketProcess.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_proc : nullptr)) {
            //音声処理を実行、出力キューに追加する
            WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    SetEvent(m_Mux.thread.heEventClosingAudProcess);
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
}

RGY_ERR RGYOutputAvcodec::WriteThreadFunc() {
#if ENABLE_AVCODEC_OUT_THREAD
    //映像と音声の同期をとる際に、それをあきらめるまでの閾値
    const int nWaitThreshold = 32;
    const size_t videoPacketThreshold = std::min<size_t>(3072, m_Mux.thread.qVideobitstream.capacity()) - nWaitThreshold;
    const size_t audioPacketThreshold = std::min<size_t>(6144, m_Mux.thread.qAudioPacketOut.capacity()) - nWaitThreshold;
    //キューにデータが存在するか
    bool bAudioExists = false;
    bool bVideoExists = false;
    const auto fpsTimebase = av_inv_q(m_Mux.video.nFPS);
    const auto dtsThreshold = std::max(av_rescale_q(4, fpsTimebase, QUEUE_DTS_TIMEBASE), 4ll);
    //syncIgnoreDtsは映像と音声の同期を行う必要がないことを意味する
    //dtsThresholdを加算したときにオーバーフローしないよう、dtsThresholdを引いておく
    const int64_t syncIgnoreDts = INT64_MAX - dtsThreshold;
    int64_t audioDts = (m_Mux.audio.size()) ? 0 : syncIgnoreDts;
    int64_t videoDts = (m_Mux.video.pStreamOut) ? 0 : syncIgnoreDts;
    WaitForSingleObject(m_Mux.thread.heEventPktAddedOutput, INFINITE);
    //bThAudProcessは出力開始した後で取得する(この前だとまだ起動していないことがある)
    const bool bThAudProcess = m_Mux.thread.thAudProcess.joinable();
    auto writeProcessedPacket = [this](AVPktMuxData *pktData) {
        //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
        auto sts = RGY_ERR_NONE;
        if (((int16_t)(pktData->pkt.flags >> 16)) < 0) {
            sts = SubtitleWritePacket(&pktData->pkt);
        } else {
            WriteNextPacketProcessed(pktData);
        }
        return sts;
    };
    int audPacketsPerSec = 64;
    int nWaitAudio = 0;
    int nWaitVideo = 0;
    while (!m_Mux.thread.bAbortOutput) {
        do {
            if (!m_Mux.format.bFileHeaderWritten) {
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
                && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_vid_out : nullptr))) {
                WriteNextFrameInternal(&bitstream, &videoDts);
                nWaitVideo = 0;
                //AddMessage(RGY_LOG_TRACE, _T("videoDts=%8lld.\n"), videoDts);
            }
            AVPktMuxData pktData = { 0 };
            while ((videoDts < 0 || audioDts <= videoDts + dtsThreshold)
                && false != (bAudioExists = m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_out : nullptr))) {
                if (pktData.pMuxAudio && pktData.pMuxAudio->pStreamIn) {
                    audPacketsPerSec = std::max(audPacketsPerSec, (int)(1.0 / (av_q2d(pktData.pMuxAudio->pStreamIn->time_base) * pktData.pkt.duration) + 0.5));
                    if ((int)m_Mux.thread.qAudioPacketOut.capacity() < audPacketsPerSec * 4) {
                        m_Mux.thread.qAudioPacketOut.set_capacity(audPacketsPerSec * 4);
                    }
                }
                const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : syncIgnoreDts;
                //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
                (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
                audioDts = (std::max)(audioDts, pktData.dts);
                nWaitAudio = 0;
                //AddMessage(RGY_LOG_TRACE, _T("audioDts=%8lld, maxDst=%8lld.\n"), audioDts, maxDts);
            }
            //一定以上の動画フレームがキューにたまっており、音声キューになにもなければ、
            //音声を無視して動画フレームの処理を開始させる
            //音声が途中までしかなかったり、途中からしかなかったりする場合にこうした処理が必要
            if (m_Mux.thread.qAudioPacketOut.size() == 0 && m_Mux.thread.qVideobitstream.size() > videoPacketThreshold) {
                nWaitAudio++;
                if (nWaitAudio <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    break;
                }
                audioDts = videoDts;
                //AddMessage(RGY_LOG_TRACE, _T("audio not coming.\n"));
            }
            //一定以上の音声フレームがキューにたまっており、動画キューになにもなければ、
            //動画を無視して音声フレームの処理を開始させる
            if (m_Mux.thread.qVideobitstream.size() == 0 && m_Mux.thread.qAudioPacketOut.size() > audioPacketThreshold) {
                nWaitVideo++;
                if (nWaitVideo <= nWaitThreshold) {
                    //時折まだパケットが来ているのにタイミングによってsize() == 0が成立することがある
                    //なのである程度連続でパケットが来ていないときのみ無視するようにする
                    //このようにすることで適切に同期がとれる
                    break;
                }
                videoDts = audioDts;
                //AddMessage(RGY_LOG_TRACE, _T("video not coming.\n"));
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
            && false != (bAudioExists = m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_out : nullptr))) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            const int64_t maxDts = (videoDts >= 0) ? videoDts + dtsThreshold : INT64_MAX;
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, maxDts);
            audioDts = (std::max)(audioDts, pktData.dts);
        }
        RGYBitstream bitstream = RGYBitstreamInit();
        while (videoDts <= audioDts + dtsThreshold
            && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_vid_out : nullptr))) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
        bAudioExists = !m_Mux.thread.qAudioPacketOut.empty();
        bVideoExists = !m_Mux.thread.qVideobitstream.empty();
    }
    { //音声を書き出す
        AVPktMuxData pktData = { 0 };
        while (m_Mux.thread.qAudioPacketOut.front_copy_and_pop_no_lock(&pktData, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_aud_out : nullptr)) {
            //音声処理スレッドが別にあるなら、出力スレッドがすべきことは単に出力するだけ
            (bThAudProcess) ? writeProcessedPacket(&pktData) : WriteNextPacketInternal(&pktData, INT64_MAX);
        }
    }
    { //動画を書き出す
        RGYBitstream bitstream = RGYBitstreamInit();
        while (m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream, (m_Mux.thread.pQueueInfo) ? &m_Mux.thread.pQueueInfo->usage_vid_out : nullptr)) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
    }
#endif
    return (m_Mux.format.bStreamError) ? RGY_ERR_UNKNOWN : RGY_ERR_NONE;
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
    return (int)_fwrite_nolock(buf, 1, buf_size, m_Mux.format.fpOutput);
}
int64_t RGYOutputAvcodec::seek(int64_t offset, int whence) {
    return _fseeki64(m_Mux.format.fpOutput, offset, whence);
}
#endif //USE_CUSTOM_IO

#endif //ENABLE_AVSW_READER
