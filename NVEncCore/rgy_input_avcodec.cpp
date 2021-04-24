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
#include <numeric>
#include <array>
#include <map>
#include <cctype>
#include <cmath>
#include <climits>
#include <limits>
#include <memory>
#include <cppcodec/base64_rfc4648.hpp>
#include "rgy_thread.h"
#include "rgy_input_avcodec.h"
#include "rgy_bitstream.h"
#include "rgy_avlog.h"
#include "rgy_language.h"


#if ENABLE_AVSW_READER
#if USE_CUSTOM_INPUT
static int funcReadPacket(void *opaque, uint8_t *buf, int buf_size) {
    RGYInputAvcodec *reader = reinterpret_cast<RGYInputAvcodec *>(opaque);
    return reader->readPacket(buf, buf_size);
}
static int funcWritePacket(void *opaque, uint8_t *buf, int buf_size) {
    RGYInputAvcodec *reader = reinterpret_cast<RGYInputAvcodec *>(opaque);
    return reader->writePacket(buf, buf_size);
}
static int64_t funcSeek(void *opaque, int64_t offset, int whence) {
    RGYInputAvcodec *reader = reinterpret_cast<RGYInputAvcodec *>(opaque);
    return reader->seek(offset, whence);
}
#endif //USE_CUSTOM_INPUT

static inline void extend_array_size(VideoFrameData *dataset) {
    static int default_capacity = 8 * 1024;
    int current_cap = dataset->capacity;
    dataset->capacity = (current_cap) ? current_cap * 2 : default_capacity;
    dataset->frame = (FramePos *)realloc(dataset->frame, dataset->capacity * sizeof(dataset->frame[0]));
    memset(dataset->frame + current_cap, 0, sizeof(dataset->frame[0]) * (dataset->capacity - current_cap));
}

RGYInputAvcodecPrm::RGYInputAvcodecPrm(RGYInputPrm base) :
    RGYInputPrm(base),
    memType(0),
    pInputFormat(nullptr),
    readVideo(false),
    videoTrack(0),
    videoStreamId(0),
    readAudio(0),
    readSubtitle(false),
    readData(false),
    readAttachment(false),
    readChapter(false),
    videoAvgFramerate(),
    analyzeSec(0),
    probesize(0),
    nTrimCount(0),
    pTrimList(nullptr),
    trackStartAudio(0),
    trackStartSubtitle(0),
    trackStartData(0),
    nAudioSelectCount(0),
    ppAudioSelect(nullptr),
    nSubtitleSelectCount(0),
    ppSubtitleSelect(nullptr),
    nDataSelectCount(0),
    ppDataSelect(nullptr),
    AVSyncMode(RGY_AVSYNC_ASSUME_CFR),
    procSpeedLimit(0),
    seekSec(0.0),
    logFramePosList(),
    logCopyFrameData(),
    threadInput(0),
    queueInfo(nullptr),
    HWDecCodecCsp(nullptr),
    videoDetectPulldown(false),
    caption2ass(FORMAT_INVALID),
    parseHDRmetadata(false),
    hdr10plusMetadataCopy(false),
    interlaceAutoFrame(false),
    qpTableListRef(nullptr),
    lowLatency(false),
    inputOpt() {

}

RGYInputAvcodec::RGYInputAvcodec() :
    m_Demux(),
    m_logFramePosList(),
    m_hevcMp42AnnexbBuffer(),
    m_cap2ass() {
    memset(&m_Demux.format, 0, sizeof(m_Demux.format));
    memset(&m_Demux.video,  0, sizeof(m_Demux.video));
    m_readerName = _T("av" DECODER_NAME "/avsw");
}

RGYInputAvcodec::~RGYInputAvcodec() {
    Close();
}

void RGYInputAvcodec::CloseThread() {
    m_Demux.thread.bAbortInput = true;
    if (m_Demux.thread.thInput.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Closing Input thread.\n"));
        m_Demux.qVideoPkt.set_capacity(SIZE_MAX);
        m_Demux.qVideoPkt.set_keep_length(0);
        m_Demux.thread.thInput.join();
        AddMessage(RGY_LOG_DEBUG, _T("Closed Input thread.\n"));
    }
    m_Demux.thread.bAbortInput = false;
}

void RGYInputAvcodec::CloseFormat(AVDemuxFormat *format) {
    //close video file
    if (format->fpInput) {
        AddMessage(RGY_LOG_DEBUG, _T("Closing file pointer...\n"));
        if (format->formatCtx) {
            if (format->formatCtx->pb) {
                if (format->formatCtx->pb->buffer) {
                    AddMessage(RGY_LOG_DEBUG, _T("Closing pb->buffer...\n"));
                    av_freep(&format->formatCtx->pb->buffer);
                    AddMessage(RGY_LOG_DEBUG, _T("Closed pb->buffer.\n"));
                }
                AddMessage(RGY_LOG_DEBUG, _T("Closing avio context...\n"));
                avio_context_free(&format->formatCtx->pb);
                AddMessage(RGY_LOG_DEBUG, _T("Closed avio context.\n"));
            }
        }
        fclose(format->fpInput);
        AddMessage(RGY_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    if (format->formatCtx) {
        AddMessage(RGY_LOG_DEBUG, _T("Closing avformat context...\n"));
        avformat_close_input(&format->formatCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
    if (format->formatOptions) {
        AddMessage(RGY_LOG_DEBUG, _T("Free formatOptions...\n"));
        av_dict_free(&format->formatOptions);
        AddMessage(RGY_LOG_DEBUG, _T("Freed formatOptions.\n"));
    }
    memset(format, 0, sizeof(format[0]));
}

void RGYInputAvcodec::CloseVideo(AVDemuxVideo *video) {
    //close parser
    if (video->pParserCtx) {
        AddMessage(RGY_LOG_DEBUG, _T("Close parser...\n"));
        av_parser_close(video->pParserCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed parser.\n"));
    }
    if (video->pCodecCtxParser) {
        AddMessage(RGY_LOG_DEBUG, _T("Close codecCtx for parser...\n"));
        avcodec_free_context(&video->pCodecCtxParser);
        AddMessage(RGY_LOG_DEBUG, _T("Closed codecCtx for parser.\n"));
    }
    if (video->codecCtxDecode) {
        AddMessage(RGY_LOG_DEBUG, _T("Close codecCtx...\n"));
        avcodec_free_context(&video->codecCtxDecode);
        AddMessage(RGY_LOG_DEBUG, _T("Closed codecCtx.\n"));
    }
    if (video->contentLight) {
        AddMessage(RGY_LOG_DEBUG, _T("Free content light metadata...\n"));
        av_freep(&video->contentLight);
        AddMessage(RGY_LOG_DEBUG, _T("Freed content light metadata.\n"));
    }
    if (video->masteringDisplay) {
        AddMessage(RGY_LOG_DEBUG, _T("Free mastering display metadata...\n"));
        av_freep(&video->masteringDisplay);
        AddMessage(RGY_LOG_DEBUG, _T("Freed mastering display metadata.\n"));
    }
    //close bitstreamfilter
    if (video->bsfcCtx) {
        AddMessage(RGY_LOG_DEBUG, _T("Free bsf...\n"));
        av_bsf_free(&video->bsfcCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Freed bsf.\n"));
    }
    if (video->frame) {
        AddMessage(RGY_LOG_DEBUG, _T("Free video frame...\n"));
        av_frame_free(&video->frame);
        AddMessage(RGY_LOG_DEBUG, _T("Freed video frame.\n"));
    }

    if (video->extradata) {
        AddMessage(RGY_LOG_DEBUG, _T("Free extra data...\n"));
        av_free(video->extradata);
        AddMessage(RGY_LOG_DEBUG, _T("Freed extra data.\n"));
    }

    memset(video, 0, sizeof(video[0]));
    video->index = -1;
}

void RGYInputAvcodec::CloseStream(AVDemuxStream *stream) {
    if (stream->pktSample.data) {
        AddMessage(RGY_LOG_DEBUG, _T("Free packet sample...\n"));
        av_packet_unref(&stream->pktSample);
        AddMessage(RGY_LOG_DEBUG, _T("Freed packet sample.\n"));
    }
    if (stream->subtitleHeader) {
        AddMessage(RGY_LOG_DEBUG, _T("Free subtitleHeader...\n"));
        av_free(stream->subtitleHeader);
        AddMessage(RGY_LOG_DEBUG, _T("Freed subtitleHeader.\n"));
    }
    memset(stream, 0, sizeof(stream[0]));
    stream->appliedTrimBlock = -1;
    stream->aud0_fin = AV_NOPTS_VALUE;
    stream->index = -1;
}

void RGYInputAvcodec::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    //リソースの解放
    CloseThread();
    m_Demux.qVideoPkt.close([](AVPacket *pkt) { av_packet_unref(pkt); });
    for (uint32_t i = 0; i < m_Demux.qStreamPktL1.size(); i++) {
        av_packet_unref(&m_Demux.qStreamPktL1[i]);
    }
    m_Demux.qStreamPktL1.clear();
    m_Demux.qStreamPktL2.close([](AVPacket *pkt) { av_packet_unref(pkt); });
    AddMessage(RGY_LOG_DEBUG, _T("Closed Stream Packet Buffer.\n"));

    m_cap2ass.close();
    AddMessage(RGY_LOG_DEBUG, _T("Closed caption handler.\n"));

    CloseFormat(&m_Demux.format); AddMessage(RGY_LOG_DEBUG, _T("Closed format.\n"));

    CloseVideo(&m_Demux.video); AddMessage(RGY_LOG_DEBUG, _T("Closed video.\n"));
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        AddMessage(RGY_LOG_DEBUG, _T("Closing Stream #%d...\n"), i);
        CloseStream(&m_Demux.stream[i]);
        AddMessage(RGY_LOG_DEBUG, _T("Closed Stream #%d.\n"), i);
    }
    m_Demux.stream.clear();
    m_Demux.chapter.clear();

    m_trimParam.list.clear();
    m_trimParam.offset = 0;

    m_hevcMp42AnnexbBuffer.clear();

    //free input buffer (使用していない)
    //if (buffer) {
    //    free(buffer);
    //    buffer = nullptr;
    //}
    m_encSatusInfo.reset();
    if (m_logFramePosList.length()) {
        m_Demux.frames.printList(m_logFramePosList.c_str());
        AddMessage(RGY_LOG_DEBUG, _T("Output logFramePosList.\n"));
    }
    m_Demux.frames.clear();
    AddMessage(RGY_LOG_DEBUG, _T("Cleared frame pos list.\n"));
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

RGY_ERR RGYInputAvcodec::initVideoBsfs() {
    if (m_Demux.video.bsfcCtx != nullptr) {
        AddMessage(RGY_LOG_DEBUG, _T("initVideoBsfs: Free old bsf...\n"));
        av_bsf_free(&m_Demux.video.bsfcCtx);
        AddMessage(RGY_LOG_DEBUG, _T("initVideoBsfs: Freed old bsf.\n"));
    }
    if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
        m_Demux.video.bUseHEVCmp42AnnexB = true;
    } else if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_H264 ||
        m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
        const char *filtername = nullptr;
        switch (m_Demux.video.stream->codecpar->codec_id) {
        case AV_CODEC_ID_H264: filtername = "h264_mp4toannexb"; break;
        case AV_CODEC_ID_HEVC: filtername = "hevc_mp4toannexb"; break;
        default: break;
        }
        if (filtername == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set bitstream filter.\n"));
            return RGY_ERR_NOT_FOUND;
        }
        auto filter = av_bsf_get_by_name(filtername);
        if (filter == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to find %s.\n"), char_to_tstring(filtername).c_str());
            return RGY_ERR_NOT_FOUND;
        }
        int ret = av_bsf_alloc(filter, &m_Demux.video.bsfcCtx);
        if (ret < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        m_Demux.video.bsfcCtx->time_base_in = av_stream_get_codec_timebase(m_Demux.video.stream);
        if (0 > (ret = avcodec_parameters_copy(m_Demux.video.bsfcCtx->par_in, m_Demux.video.stream->codecpar))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set parameter for %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        m_Demux.video.bsfcCtx->time_base_in = m_Demux.video.stream->time_base;
        if (0 > (ret = av_bsf_init(m_Demux.video.bsfcCtx))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter.\n"), char_to_tstring(filter->name).c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputAvcodec::initVideoParser() {
    if (m_Demux.video.pParserCtx) {
        AddMessage(RGY_LOG_DEBUG, _T("initVideoParser: Close old parser...\n"));
        av_parser_close(m_Demux.video.pParserCtx);
        AddMessage(RGY_LOG_DEBUG, _T("initVideoParser: Closed old parser.\n"));
        m_Demux.video.pParserCtx = nullptr;
    }
    if (m_Demux.video.stream->codecpar->extradata != nullptr
        && m_Demux.video.extradata == nullptr) {
        return RGY_ERR_MORE_DATA;
    }
    //parserはseek後に初期化すること
    //parserが使用されていれば、ここでも使用するようにする
    //たとえば、入力がrawcodecなどでは使用しない
    m_Demux.video.pParserCtx = av_parser_init(m_Demux.video.stream->codecpar->codec_id);
    if (m_Demux.video.pParserCtx) {
        m_Demux.video.pParserCtx->flags |= PARSER_FLAG_COMPLETE_FRAMES;
        if (nullptr == (m_Demux.video.pCodecCtxParser = avcodec_alloc_context3(avcodec_find_decoder(m_Demux.video.stream->codecpar->codec_id)))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate context for parser.\n"));
            return RGY_ERR_NULL_PTR;
        }
        unique_ptr_custom<AVCodecParameters> codecParamCopy(avcodec_parameters_alloc(), [](AVCodecParameters *pCodecPar) {
            avcodec_parameters_free(&pCodecPar);
            });
        int ret = 0;
        if (0 > (ret = avcodec_parameters_copy(codecParamCopy.get(), m_Demux.video.stream->codecpar))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (m_Demux.video.bsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB) {
            SetExtraData(codecParamCopy.get(), m_Demux.video.extradata, m_Demux.video.extradataSize);
        }
        if (0 > (ret = avcodec_parameters_to_context(m_Demux.video.pCodecCtxParser, codecParamCopy.get()))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        m_Demux.video.pCodecCtxParser->time_base = av_stream_get_codec_timebase(m_Demux.video.stream);
        m_Demux.video.pCodecCtxParser->pkt_timebase = m_Demux.video.stream->time_base;
        AddMessage(RGY_LOG_DEBUG, _T("initialized %s codec context for parser: time_base: %d/%d, pkt_timebase: %d/%d.\n"),
            char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(),
            m_Demux.video.pCodecCtxParser->time_base.num, m_Demux.video.pCodecCtxParser->time_base.den,
            m_Demux.video.pCodecCtxParser->pkt_timebase.num, m_Demux.video.pCodecCtxParser->pkt_timebase.den);
    } else if (m_Demux.video.HWDecodeDeviceId >= 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init parser for %s.\n"), char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str());
        return RGY_ERR_NULL_PTR;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputAvcodec::parseVideoExtraData(const AVPacket *pkt) {
    const char *bsf_name = "extract_extradata";
    const auto bsf = av_bsf_get_by_name(bsf_name);
    if (bsf == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to bsf %s.\n"), char_to_tstring(bsf_name).c_str());
        return RGY_ERR_NULL_PTR;
    }
    int ret = 0;
    AVBSFContext *bsfctmp = nullptr;
    if (0 > (ret = av_bsf_alloc(bsf, &bsfctmp))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> bsfc(bsfctmp, RGYAVDeleter<AVBSFContext>(av_bsf_free));
    bsfctmp = nullptr;

    if (0 > (ret = avcodec_parameters_copy(bsfc->par_in, m_Demux.video.stream->codecpar))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = av_bsf_init(bsfc.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), char_to_tstring(bsf_name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Initialized bsf %s\n"), char_to_tstring(bsf_name).c_str());

    unique_ptr<AVPacket, decltype(&av_packet_unref)> pktCopy(av_packet_clone(pkt), av_packet_unref);
    if (0 > (ret = av_bsf_send_packet(bsfc.get(), pktCopy.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(bsfc.get(), pktCopy.get());
    if (ret == AVERROR(EAGAIN)) {
        return RGY_ERR_NONE;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pktCopy->size < 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    int side_data_size = 0;
    auto side_data = av_packet_get_side_data(pktCopy.get(), AV_PKT_DATA_NEW_EXTRADATA, &side_data_size);
    if (side_data) {
        AddMessage(RGY_LOG_DEBUG, _T("Found extradata of codec %s: size %d\n"), char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(), side_data_size);
    }
    return RGY_ERR_NONE;
}

void RGYInputAvcodec::SetExtraData(AVCodecParameters *codecParam, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecParam->extradata)
        av_free(codecParam->extradata);
    codecParam->extradata_size = size;
    codecParam->extradata      = (uint8_t *)av_malloc(codecParam->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecParam->extradata, data, size);
    memset(codecParam->extradata + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
};

RGY_CODEC RGYInputAvcodec::checkHWDecoderAvailable(AVCodecID id, AVPixelFormat pixfmt, const CodecCsp *HWDecCodecCsp) {
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        if (HW_DECODE_LIST[i].avcodec_id == id) {
            auto rgy_codec = HW_DECODE_LIST[i].rgy_codec;
            if (HWDecCodecCsp->count(rgy_codec) > 0) {
                const auto rgy_csp = csp_avpixfmt_to_rgy(pixfmt);
                auto& csp_list = HWDecCodecCsp->at(rgy_codec);
                if (std::find(csp_list.begin(), csp_list.end(), rgy_csp) != csp_list.end()) {
                    return rgy_codec;
                }
            }
            return RGY_CODEC_UNKNOWN;
        }
    }
    return RGY_CODEC_UNKNOWN;
}

vector<int> RGYInputAvcodec::getStreamIndex(AVMediaType type, const vector<int> *vidStreamIndex) {
    vector<int> streams;
    const int n_streams = m_Demux.format.formatCtx->nb_streams;
    for (int i = 0; i < n_streams; i++) {
        const AVStream *stream = m_Demux.format.formatCtx->streams[i];
        if (type == AVMEDIA_TYPE_ATTACHMENT) {
            if (stream->codecpar->codec_type == type || (stream->disposition & AV_DISPOSITION_ATTACHED_PIC) != 0) {
                streams.push_back(i);
            }
        } else if (stream->codecpar->codec_type == type && (stream->disposition & AV_DISPOSITION_ATTACHED_PIC) == 0) {
            streams.push_back(i);
        }
    }
    if (type == AVMEDIA_TYPE_VIDEO) {
        std::sort(streams.begin(), streams.end(), [formatCtx = m_Demux.format.formatCtx](int streamIdA, int streamIdB) {
            auto pStreamA = formatCtx->streams[streamIdA];
            auto pStreamB = formatCtx->streams[streamIdB];
            if (pStreamA->codecpar == nullptr) {
                return false;
            }
            if (pStreamB->codecpar == nullptr) {
                return true;
            }
            const int resA = pStreamA->codecpar->width * pStreamA->codecpar->height;
            const int resB = pStreamB->codecpar->width * pStreamB->codecpar->height;
            return (resA > resB);
        });
    } else if (vidStreamIndex && vidStreamIndex->size()) {
        auto mostNearestVidStreamId = [formatCtx = m_Demux.format.formatCtx, vidStreamIndex](int streamId) {
            auto ret = std::make_pair(0, UINT32_MAX);
            for (uint32_t i = 0; i < vidStreamIndex->size(); i++) {
                uint32_t diff = (uint32_t)(streamId - formatCtx->streams[(*vidStreamIndex)[i]]->id);
                if (diff < ret.second) {
                    ret.second = diff;
                    ret.first = i;
                }
            }
            return ret;
        };
        std::sort(streams.begin(), streams.end(), [formatCtx = m_Demux.format.formatCtx, vidStreamIndex, mostNearestVidStreamId](int streamIdA, int streamIdB) {
            if (formatCtx->streams[streamIdA]->codecpar == nullptr) {
                return false;
            }
            if (formatCtx->streams[streamIdB]->codecpar == nullptr) {
                return true;
            }
            auto pStreamIdA = formatCtx->streams[streamIdA]->id;
            auto pStreamIdB = formatCtx->streams[streamIdB]->id;
            auto nearestVidA = mostNearestVidStreamId(pStreamIdA);
            auto nearestVidB = mostNearestVidStreamId(pStreamIdB);
            if (nearestVidA.first == nearestVidB.first) {
                return nearestVidA.second < nearestVidB.second;
            }
            return nearestVidA.first < nearestVidB.first;
        });
    }
    return streams;
}

bool RGYInputAvcodec::vc1StartCodeExists(uint8_t *ptr) {
    uint32_t code = readUB32(ptr);
    return check_range_unsigned(code, 0x010A, 0x010F) || check_range_unsigned(code, 0x011B, 0x011F);
}

void RGYInputAvcodec::vc1FixHeader(int nLengthFix) {
    if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_WMV3) {
        m_Demux.video.extradataSize += nLengthFix;
        uint32_t datasize = m_Demux.video.extradataSize;
        vector<uint8_t> buffer(20 + datasize, 0);
        uint32_t header = 0xC5000000;
        uint32_t width = m_Demux.video.stream->codecpar->width;
        uint32_t height = m_Demux.video.stream->codecpar->height;
        uint8_t *dataPtr = m_Demux.video.extradata - nLengthFix;
        memcpy(buffer.data() +  0, &header, sizeof(header));
        memcpy(buffer.data() +  4, &datasize, sizeof(datasize));
        memcpy(buffer.data() +  8, dataPtr, datasize);
        memcpy(buffer.data() +  8 + datasize, &height, sizeof(height));
        memcpy(buffer.data() + 12 + datasize, &width, sizeof(width));
        m_Demux.video.extradata = (uint8_t *)av_realloc(m_Demux.video.extradata, sizeof(buffer) + AV_INPUT_BUFFER_PADDING_SIZE);
        m_Demux.video.extradataSize = (int)buffer.size();
        memcpy(m_Demux.video.extradata, buffer.data(), buffer.size());
    } else {
        m_Demux.video.extradataSize += nLengthFix;
        memmove(m_Demux.video.extradata, m_Demux.video.extradata - nLengthFix, m_Demux.video.extradataSize);
    }
}

void RGYInputAvcodec::vc1AddFrameHeader(AVPacket *pkt) {
    uint32_t size = pkt->size;
    if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_WMV3) {
        av_grow_packet(pkt, 8);
        memmove(pkt->data + 8, pkt->data, size);
        memcpy(pkt->data, &size, sizeof(size));
        memset(pkt->data + 4, 0, 4);
    } else if (!vc1StartCodeExists(pkt->data)) {
        uint32_t startCode = 0x0D010000;
        av_grow_packet(pkt, sizeof(startCode));
        memmove(pkt->data + sizeof(startCode), pkt->data, size);
        memcpy(pkt->data, &startCode, sizeof(startCode));
    }
}

bool RGYInputAvcodec::isSelectedLangTrack(const std::string &lang, const AVStream *stream) {
    if (lang.length() == 0) return false;
    const auto &streamLang = getTrackLang(stream);
    if (streamLang.length() == 0) return false;
    return rgy_lang_equal(lang, streamLang);
}

void RGYInputAvcodec::hevcMp42Annexb(AVPacket *pkt) {
    static const uint8_t SC[] = { 0, 0, 0, 1 };
    const uint8_t *ptr, *ptr_fin;
    if (pkt == NULL) {
        m_hevcMp42AnnexbBuffer.reserve(m_Demux.video.extradataSize + 128);
        ptr = m_Demux.video.extradata;
        ptr_fin = ptr + m_Demux.video.extradataSize;
        ptr += 0x16;
    } else {
        m_hevcMp42AnnexbBuffer.reserve(pkt->size + 128);
        ptr = pkt->data;
        ptr_fin = ptr + pkt->size;
    }
    const int numOfArrays = *ptr;
    ptr += !!numOfArrays;

    while (ptr + 6 < ptr_fin) {
        ptr += !!numOfArrays;
        const int count = readUB16(ptr); ptr += 2;
        int units = (numOfArrays) ? count : 1;
        for (int i = (std::max)(1, units); i; i--) {
            uint32_t size = readUB16(ptr); ptr += 2;
            uint32_t uppper = count << 16;
            size += (numOfArrays) ? 0 : uppper;
            m_hevcMp42AnnexbBuffer.insert(m_hevcMp42AnnexbBuffer.end(), SC, SC+4);
            m_hevcMp42AnnexbBuffer.insert(m_hevcMp42AnnexbBuffer.end(), ptr, ptr+size); ptr += size;
        }
    }
    if (pkt) {
        if (pkt->buf->size < (int)m_hevcMp42AnnexbBuffer.size()) {
            av_grow_packet(pkt, (int)m_hevcMp42AnnexbBuffer.size());
        }
        memcpy(pkt->data, m_hevcMp42AnnexbBuffer.data(), m_hevcMp42AnnexbBuffer.size());
        pkt->size = (int)m_hevcMp42AnnexbBuffer.size();
    } else {
        if (m_Demux.video.extradata) {
            av_free(m_Demux.video.extradata);
        }
        m_Demux.video.extradata = (uint8_t *)av_malloc(m_hevcMp42AnnexbBuffer.size());
        m_Demux.video.extradataSize = (int)m_hevcMp42AnnexbBuffer.size();
        memcpy(m_Demux.video.extradata, m_hevcMp42AnnexbBuffer.data(), m_hevcMp42AnnexbBuffer.size());
    }
    m_hevcMp42AnnexbBuffer.clear();
}

const AVPacket *RGYInputAvcodec::findFirstAudioStreamPackets(const AVDemuxStream& streamInfo) {
    //まず、L2キューを探す
    for (int j = 0; j < (int)m_Demux.qStreamPktL2.size(); j++) {
        if (m_Demux.qStreamPktL2.get(j)->data.stream_index == streamInfo.index) {
            return &(m_Demux.qStreamPktL2.get(j)->data);
        }
    }
    //それで見つからなかったら、L1キューを探す
    for (int j = 0; j < (int)m_Demux.qStreamPktL1.size(); j++) {
        if (m_Demux.qStreamPktL1[j].stream_index == streamInfo.index) {
            return &m_Demux.qStreamPktL1[j];
        }
    }
    return nullptr;
}

RGY_ERR RGYInputAvcodec::getFirstFramePosAndFrameRate(const sTrim *pTrimList, int nTrimCount, bool bDetectpulldown, bool lowLatency, rgy_rational<int> fpsOverride) {
    AVRational fpsDecoder = m_Demux.video.stream->avg_frame_rate;
    const bool fpsDecoderInvalid = (fpsDecoder.den == 0 || fpsDecoder.num == 0);
    //timebaseが60で割り切れない場合には、ptsが完全には割り切れない値である場合があり、より多くのフレーム数を解析する必要がある

    int maxCheckFrames = 0;
    double maxCheckSec = 0.0;
    if (fpsOverride.is_valid()) { //あらかじめfpsが指定されていればそれを採用する
        maxCheckFrames = 1;
        maxCheckSec = 1e99;
    } else if (m_Demux.format.analyzeSec >= 0.0) {
        if (m_Demux.format.analyzeSec <= 1.0) { // analyzeが1秒以下の場合
            if (!fpsDecoderInvalid) { //fpsDecoderが有効な値ならそれを使用する
                maxCheckFrames = 1;
                maxCheckSec = 1e99;
                fpsOverride = rgy_rational<int>(fpsDecoder.num, fpsDecoder.den);
            } else { // なるべく短く判定を行う
                maxCheckFrames = (m_Demux.video.stream->time_base.den >= 1000 && m_Demux.video.stream->time_base.den % 60) ? 128 : 20;
                maxCheckSec = std::max(m_Demux.format.analyzeSec, 1.0);
            }
        } else {
            maxCheckFrames = 7200;
            maxCheckSec = m_Demux.format.analyzeSec;
        }
    } else {
        maxCheckFrames = (m_Demux.video.stream->time_base.den >= 1000 && m_Demux.video.stream->time_base.den % 60) ? 128 : ((lowLatency) ? 32 : 48);
        maxCheckSec = 1e99;
    }
    AddMessage(RGY_LOG_DEBUG, _T("fps decoder %d/%d, invalid: %s\n"), fpsDecoder.num, fpsDecoder.den, fpsDecoderInvalid ? _T("true") : _T("false"));

    AVPacket pkt;
    av_init_packet(&pkt);

    const bool bCheckDuration = m_Demux.video.stream->time_base.num * m_Demux.video.stream->time_base.den > 0;
    const double timebase = (bCheckDuration) ? m_Demux.video.stream->time_base.num / (double)m_Demux.video.stream->time_base.den : 1.0;
    m_Demux.video.streamFirstKeyPts = 0;
    int i_samples = 0;
    std::vector<int> frameDurationList;
    vector<std::pair<int, int>> durationHistgram;
    bool bPulldown = false;

    for (int i_retry = 0; ; i_retry++) {
        if (i_retry) {
            //フレームレート推定がうまくいかなそうだった場合、もう少しフレームを解析してみる
            maxCheckFrames <<= 1;
            if (maxCheckSec != 1e99) {
                maxCheckSec *= 2.0;
            }
            //ヒストグラム生成などは最初からやり直すので、一度クリアする
            durationHistgram.clear();
            frameDurationList.clear();
        }
        AddMessage(RGY_LOG_DEBUG, _T("maxCheckFrames %d, maxCheckSec: %.2f\n"), maxCheckFrames, maxCheckSec);

        int ret = 0;
        for (; i_samples < maxCheckFrames && ((ret = getSample(&pkt)) == 0); i_samples++) {
            m_Demux.qVideoPkt.push(pkt);
            if (bCheckDuration) {
                int64_t diff = 0;
                if (pkt.dts != AV_NOPTS_VALUE && m_Demux.frames.list(0).dts != AV_NOPTS_VALUE) {
                    diff = (int)(pkt.dts - m_Demux.frames.list(0).dts);
                } else if (pkt.pts != AV_NOPTS_VALUE && m_Demux.frames.list(0).pts != AV_NOPTS_VALUE) {
                    diff = (int)(pkt.pts - m_Demux.frames.list(0).pts);
                }
                const double duration = diff * timebase;
                if (duration >= maxCheckSec) {
                    break;
                }
            }
        }
        if (ret != 0 && ret != AVERROR_EOF) {
            return RGY_ERR_UNKNOWN;
        }
        if (m_Demux.qVideoPkt.size() == 0) {
            AddMessage(RGY_LOG_ERROR, _T("No video packets found!\n"));
            return RGY_ERR_UNKNOWN;
        }
#if _DEBUG && 0
        for (int i = 0; i < m_Demux.frames.frameNum(); i++) {
            fprintf(stderr, "%3d: pts:%lld, poc:%3d, duration:%5d, duration2:%5d, repeat:%d\n",
                i, (long long int)m_Demux.frames.list(i).pts, m_Demux.frames.list(i).poc,
                m_Demux.frames.list(i).duration, m_Demux.frames.list(i).duration2,
                m_Demux.frames.list(i).repeat_pict);
        }
#endif
        //ここまで集めたデータでpts, pocを確定させる
        double dEstFrameDurationByFpsDecoder = 0.0;
        if (av_isvalid_q(fpsDecoder) && av_isvalid_q(m_Demux.video.stream->time_base)) {
            dEstFrameDurationByFpsDecoder = av_q2d(av_inv_q(fpsDecoder)) * av_q2d(av_inv_q(m_Demux.video.stream->time_base));
        }
        m_Demux.frames.checkPtsStatus(dEstFrameDurationByFpsDecoder);

        const int nFramesToCheck = m_Demux.frames.fixedNum();
        AddMessage(RGY_LOG_DEBUG, _T("read %d packets.\n"), m_Demux.frames.frameNum());
        AddMessage(RGY_LOG_DEBUG, _T("checking %d frame samples.\n"), nFramesToCheck);
        if (fpsOverride.is_valid()) { //あらかじめfpsが指定されていればそれを採用するので、ここでちゃんと分析する必要はない
            //音声の最初のパケットが見つかっていればOK、そうでなければやり直す
            bool audioStreamPacketNotFound = false;
            for (const auto& streamInfo : m_Demux.stream) {
                if (streamInfo.stream
                    && avcodec_get_type(streamInfo.stream->codecpar->codec_id) == AVMEDIA_TYPE_AUDIO
                    && findFirstAudioStreamPackets(streamInfo) == nullptr) {
                    audioStreamPacketNotFound = true;
                    break;
                }
            }
            if (!audioStreamPacketNotFound) {
                break; //音声の最初のパケットが見つかっていればOK
            }
        } else if (nFramesToCheck > 0) {
            frameDurationList.reserve(nFramesToCheck);
            int rff_frames = 0;

            for (int i = 0; i < nFramesToCheck; i++) {
#if _DEBUG && 0
                fprintf(stderr, "%3d: pts:%lld, poc:%3d, duration:%5d, duration2:%5d, repeat:%d\n",
                    i, (long long int)m_Demux.frames.list(i).pts, m_Demux.frames.list(i).poc,
                    m_Demux.frames.list(i).duration, m_Demux.frames.list(i).duration2,
                    m_Demux.frames.list(i).repeat_pict);
#endif
                if (m_Demux.frames.list(i).poc != FRAMEPOS_POC_INVALID) {
                    int duration = m_Demux.frames.list(i).duration + m_Demux.frames.list(i).duration2;
                    auto repeat_pict = m_Demux.frames.list(i).repeat_pict;
                    //RFF用の補正
                    if (repeat_pict > 1) {
                        duration = (int)(duration * 2 / (double)(repeat_pict + 1) + 0.5);
                        rff_frames++;
                    }
                    frameDurationList.push_back(duration);
                }
            }
            bPulldown = (bDetectpulldown && ((rff_frames + 1/*たまたま切り捨てられることのないように*/) / (double)nFramesToCheck > 0.45));

            //durationのヒストグラムを作成
            std::for_each(frameDurationList.begin(), frameDurationList.end(), [&durationHistgram](const int& duration) {
                auto target = std::find_if(durationHistgram.begin(), durationHistgram.end(), [duration](const std::pair<int, int>& pair) { return pair.first == duration; });
                if (target != durationHistgram.end()) {
                    target->second++;
                } else {
                    durationHistgram.push_back(std::make_pair(duration, 1));
                }
                });
            //多い順にソートする
            std::sort(durationHistgram.begin(), durationHistgram.end(), [](const std::pair<int, int>& pairA, const std::pair<int, int>& pairB) { return pairA.second > pairB.second; });

            const auto codec_timebase = av_stream_get_codec_timebase(m_Demux.video.stream);
            AddMessage(RGY_LOG_DEBUG, _T("stream timebase %d/%d\n"), codec_timebase.num, codec_timebase.den);
            AddMessage(RGY_LOG_DEBUG, _T("decoder fps     %d/%d\n"), fpsDecoder.num, fpsDecoder.den);
            AddMessage(RGY_LOG_DEBUG, _T("duration histgram of %d frames\n"), durationHistgram.size());
            for (const auto& sample : durationHistgram) {
                AddMessage(RGY_LOG_DEBUG, _T("%3d [%3d frames]\n"), sample.first, sample.second);
            }

            //ここでやめてよいか判定する
            if (i_retry == 0) {
                //初回は、唯一のdurationが得られている場合を除き再解析する
                if (durationHistgram.size() <= 1) {
                    break;
                }
            } else if (durationHistgram.size() <= 1 //唯一のdurationが得られている
                || durationHistgram[0].second / (double)frameDurationList.size() > 0.95 //大半がひとつのdurationである
                || std::abs(durationHistgram[0].first - durationHistgram[1].first) <= 1) { //durationのブレが貧弱なtimebaseによる丸めによるもの(mkvなど)
                break;
            }
            if (i_retry >= 4) {
                break;
            }
        }
        //再度解析を行う場合は、音声がL2キューに入らないよう、一度fixedNumを0に戻す
        m_Demux.frames.clearPtsStatus();
    }

    if (fpsOverride.is_valid()) {
        m_Demux.video.nAvgFramerate = av_make_q(fpsOverride);
    } else {
        //durationが0でなく、最も頻繁に出てきたもの
        auto& mostPopularDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0];

        struct Rational64 {
            uint64_t num;
            uint64_t den;
        } estimatedAvgFps = { 0 }, nAvgFramerate64 = { 0 }, fpsDecoder64 = { (uint64_t)fpsDecoder.num, (uint64_t)fpsDecoder.den };
        if (mostPopularDuration.first == 0) {
            m_Demux.video.streamPtsInvalid |= RGY_PTS_ALL_INVALID;
        } else {
            //avgFpsとtargetFpsが近いかどうか
            auto fps_near = [](double avgFps, double targetFps) { return std::abs(1 - avgFps / targetFps) < 0.5; };
            //durationの平均を求める (ただし、先頭は信頼ならないので、cutoff分は計算に含めない)
            //std::accumulateの初期値に"(uint64_t)0"と与えることで、64bitによる計算を実行させ、桁あふれを防ぐ
            //大きすぎるtimebaseの時に必要
            double avgDuration = std::accumulate(frameDurationList.begin(), frameDurationList.end(), (uint64_t)0, [this](const uint64_t sum, const int& duration) { return sum + duration; }) / (double)(frameDurationList.size());
            if (bPulldown) {
                avgDuration *= 1.25;
            }
            double avgFps = m_Demux.video.stream->time_base.den / (double)(avgDuration * m_Demux.video.stream->time_base.num);
            double torrelance = (fps_near(avgFps, 25.0) || fps_near(avgFps, 50.0)) ? 0.05 : 0.0008; //25fps, 50fps近辺は基準が甘くてよい
            if (mostPopularDuration.second / (double)frameDurationList.size() > 0.95 && std::abs(1 - mostPopularDuration.first / avgDuration) < torrelance) {
                avgDuration = mostPopularDuration.first;
                AddMessage(RGY_LOG_DEBUG, _T("using popular duration...\n"));
            }
            //durationから求めた平均fpsを計算する
            const uint64_t mul = (uint64_t)ceil(1001.0 / m_Demux.video.stream->time_base.num);
            estimatedAvgFps.num = (uint64_t)(m_Demux.video.stream->time_base.den / avgDuration * (double)m_Demux.video.stream->time_base.num * mul + 0.5);
            estimatedAvgFps.den = (uint64_t)m_Demux.video.stream->time_base.num * mul;

            AddMessage(RGY_LOG_DEBUG, _T("fps mul:         %d\n"), mul);
            AddMessage(RGY_LOG_DEBUG, _T("raw avgDuration: %lf\n"), avgDuration);
            AddMessage(RGY_LOG_DEBUG, _T("estimatedAvgFps: %llu/%llu\n"), (long long int)estimatedAvgFps.num, (long long int)estimatedAvgFps.den);
        }

        if (m_Demux.video.streamPtsInvalid & RGY_PTS_ALL_INVALID) {
            //ptsとdurationをpkt_timebaseで適当に作成する
            nAvgFramerate64 = (fpsDecoderInvalid) ? estimatedAvgFps : fpsDecoder64;
        } else {
            if (fpsDecoderInvalid) {
                nAvgFramerate64 = estimatedAvgFps;
            } else {
                double dFpsDecoder = fpsDecoder.num / (double)fpsDecoder.den;
                double dEstimatedAvgFps = estimatedAvgFps.num / (double)estimatedAvgFps.den;
                //2フレーム分程度がもたらす誤差があっても許容する
                if (std::abs(dFpsDecoder / dEstimatedAvgFps - 1.0) < (2.0 / frameDurationList.size())) {
                    AddMessage(RGY_LOG_DEBUG, _T("use decoder fps...\n"));
                    nAvgFramerate64 = fpsDecoder64;
                } else {
                    double dEstimatedAvgFpsCompare = estimatedAvgFps.num / (double)(estimatedAvgFps.den + ((dFpsDecoder < dEstimatedAvgFps) ? 1 : -1));
                    //durationから求めた平均fpsがデコーダの出したfpsの近似値と分かれば、デコーダの出したfpsを採用する
                    nAvgFramerate64 = (std::abs(dEstimatedAvgFps - dFpsDecoder) < std::abs(dEstimatedAvgFpsCompare - dFpsDecoder)) ? fpsDecoder64 : estimatedAvgFps;
                }
            }
        }
        AddMessage(RGY_LOG_DEBUG, _T("final AvgFps (raw64): %llu/%llu\n"), (long long int)estimatedAvgFps.num, (long long int)estimatedAvgFps.den);

        //フレームレートが2000fpsを超えることは考えにくいので、誤判定
        //ほかのなにか使えそうな値で代用する
        const auto codec_timebase = av_stream_get_codec_timebase(m_Demux.video.stream);
        if (nAvgFramerate64.num / (double)nAvgFramerate64.den > 2000.0) {
            if (fpsDecoder.den > 0 && fpsDecoder.num > 0) {
                nAvgFramerate64.num = fpsDecoder.num;
                nAvgFramerate64.den = fpsDecoder.den;
            } else if (codec_timebase.den > 0
                && codec_timebase.num > 0) {
                const AVCodec *codec = avcodec_find_decoder(m_Demux.video.stream->codecpar->codec_id);
                AVCodecContext *pCodecCtx = avcodec_alloc_context3(codec);
                nAvgFramerate64.num = codec_timebase.den * pCodecCtx->ticks_per_frame;
                nAvgFramerate64.den = codec_timebase.num;
                avcodec_free_context(&pCodecCtx);
            }
        }

        rgy_reduce(nAvgFramerate64.num, nAvgFramerate64.den);
        m_Demux.video.nAvgFramerate = av_make_q((int)nAvgFramerate64.num, (int)nAvgFramerate64.den);
        AddMessage(RGY_LOG_DEBUG, _T("final AvgFps (gcd): %d/%d\n"), m_Demux.video.nAvgFramerate.num, m_Demux.video.nAvgFramerate.den);

        struct KnownFpsList {
            std::vector<int> base;
            std::vector<int> mul;
            int timebase_num;
        };
        const KnownFpsList knownFpsSmall = {
            std::vector<int>{1, 2, 3, 4, 5, 10},
            std::vector<int>{1},
            1
        };
        const KnownFpsList knownFps1 = {
            std::vector<int>{10, 12, 25},
            std::vector<int>{1, 2, 3, 4, 5, 6, 10, 12, 20},
            1
        };
        const KnownFpsList knownFps1001 = {
            std::vector<int>{12000, 15000},
            std::vector<int>{1, 2, 3, 4, 6, 8, 12, 16},
            1001
        };
        const double fpsAvg = av_q2d(m_Demux.video.nAvgFramerate);
        double fpsDiff = std::numeric_limits<double>::max();
        AVRational fpsNear = m_Demux.video.nAvgFramerate;
        auto round_fps = [&fpsDiff, &fpsNear, fpsAvg](const KnownFpsList& known_fps) {
            for (auto b : known_fps.base) {
                for (auto m : known_fps.mul) {
                    double fpsKnown = b * m / (double)known_fps.timebase_num;
                    double diff = std::abs(fpsKnown - fpsAvg);
                    if (diff < fpsDiff) {
                        fpsDiff = diff;
                        fpsNear = av_make_q(b * m, known_fps.timebase_num);
                    }
                }
            }
        };
        round_fps(knownFpsSmall);
        round_fps(knownFps1);
        round_fps(knownFps1001);
        if (fpsDiff / fpsAvg < 2.0 / 60.0) {
            m_Demux.video.nAvgFramerate = fpsNear;
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("final AvgFps (round): %d/%d\n\n"), m_Demux.video.nAvgFramerate.num, m_Demux.video.nAvgFramerate.den);

    auto trimList = make_vector(pTrimList, nTrimCount);
    //出力時の音声・字幕解析用に1パケットコピーしておく
    if (m_Demux.qStreamPktL1.size() > 0 || m_Demux.qStreamPktL2.size() > 0) {
        if (!m_Demux.frames.isEof() && m_Demux.qStreamPktL2.size() > 0) {
            //最後まで読み込んでいなかったら、すべてのパケットはqStreamPktL1にあるはず
            AddMessage(RGY_LOG_ERROR, _T("qStreamPktL2 > 0, this is internal error.\n"));
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        for (auto streamInfo = m_Demux.stream.begin(); streamInfo != m_Demux.stream.end(); streamInfo++) {
            if (streamInfo->stream && avcodec_get_type(streamInfo->stream->codecpar->codec_id) == AVMEDIA_TYPE_AUDIO) {
                AddMessage(RGY_LOG_DEBUG, _T("checking for stream #%d\n"), streamInfo->index);
                const AVPacket *pkt1 = nullptr; //最初のパケット
                const AVPacket *pkt2 = nullptr; //2番目のパケット
                //まず、L2キューを探す
                for (int j = 0; j < (int)m_Demux.qStreamPktL2.size(); j++) {
                    if (m_Demux.qStreamPktL2.get(j)->data.stream_index == streamInfo->index) {
                        if (pkt1) {
                            pkt2 = &(m_Demux.qStreamPktL2.get(j)->data);
                            break;
                        }
                        pkt1 = &(m_Demux.qStreamPktL2.get(j)->data);
                    }
                }
                if (pkt2 == nullptr) {
                    //それで見つからなかったら、L1キューを探す
                    for (int j = 0; j < (int)m_Demux.qStreamPktL1.size(); j++) {
                        if (m_Demux.qStreamPktL1[j].stream_index == streamInfo->index) {
                            if (pkt1) {
                                pkt2 = &m_Demux.qStreamPktL1[j];
                                break;
                            }
                            pkt1 = &m_Demux.qStreamPktL1[j];
                        }
                    }
                }
                if (pkt1 != nullptr) {
                    //1パケット目はたまにおかしいので、可能なら2パケット目を使用する
                    av_packet_ref(&streamInfo->pktSample, (pkt2) ? pkt2 : pkt1);
                } else {
                    //音声の最初のサンプルを取得できていない
                    AddMessage(RGY_LOG_WARN, _T("failed to find stream #%d in preread.\n"), streamInfo->index);
                    streamInfo = m_Demux.stream.erase(streamInfo) - 1;
                }
            }
        }
        if (m_Demux.stream.size() == 0) {
            //音声・字幕の最初のサンプルを取得できていないため、音声がすべてなくなってしまった
            AddMessage(RGY_LOG_ERROR, _T("failed to find audio/subtitle stream in preread.\n"));
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
    }

    return RGY_ERR_NONE;
}

RGY_ERR RGYInputAvcodec::parseHDRData() {
    //まずはstreamのside_dataを探す
    int size = 0;
    auto data = av_stream_get_side_data(m_Demux.video.stream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA, &size);
    if (data) {
        m_Demux.video.masteringDisplay = av_mastering_display_metadata_alloc();
        memcpy(m_Demux.video.masteringDisplay, data, size);
        AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[0][0]),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[0][1]),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[1][0]),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[1][1]),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[2][0]),
            av_q2d(m_Demux.video.masteringDisplay->display_primaries[2][1]),
            av_q2d(m_Demux.video.masteringDisplay->white_point[0]), av_q2d(m_Demux.video.masteringDisplay->white_point[1]),
            av_q2d(m_Demux.video.masteringDisplay->min_luminance), av_q2d(m_Demux.video.masteringDisplay->max_luminance));
    }
    data = av_stream_get_side_data(m_Demux.video.stream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL, &size);
    if (data) {
        size_t st_size = 0;
        m_Demux.video.contentLight = av_content_light_metadata_alloc(&st_size);
        memcpy(m_Demux.video.contentLight, data, st_size);
        AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"), m_Demux.video.contentLight->MaxCLL, m_Demux.video.contentLight->MaxFALL);
    }
    if (m_Demux.video.masteringDisplay || m_Demux.video.contentLight) {
        return RGY_ERR_NONE;
    }
    if (m_Demux.video.stream->codecpar->codec_id != AV_CODEC_ID_HEVC) {
        return RGY_ERR_NONE;
    }

    //codec側に格納されている値は、streamからは取得できない
    //HDR関連のmeta情報を取得するために、decoderをキックし、先頭のフレームにセットされる情報を取得する
    auto codecDecode = avcodec_find_decoder(m_Demux.video.stream->codecpar->codec_id);
    if (codecDecode == nullptr) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to find decoder"), m_Demux.video.stream->codecpar->codec_id).c_str());
        return RGY_ERR_NOT_FOUND;
    }
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecCtxDec(avcodec_alloc_context3(codecDecode), RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    if (!codecCtxDec) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to allocate decoder"), m_Demux.video.stream->codecpar->codec_id).c_str());
        return RGY_ERR_NULL_PTR;
    }
    unique_ptr_custom<AVCodecParameters> codecParamCopy(avcodec_parameters_alloc(), [](AVCodecParameters *pCodecPar) {
        avcodec_parameters_free(&pCodecPar);
        });
    auto ret = avcodec_parameters_copy(codecParamCopy.get(), m_Demux.video.stream->codecpar);
    if (ret < 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (m_Demux.video.bsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB) {
        SetExtraData(codecParamCopy.get(), m_Demux.video.extradata, m_Demux.video.extradataSize);
    }
    if (0 > (ret = avcodec_parameters_to_context(codecCtxDec.get(), codecParamCopy.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to set codec param to context for decoder: %s.\n"), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    AVDictionary *pDict = nullptr;
    av_dict_set_int(&pDict, "threads", 1, 0);
    if (0 > (ret = av_opt_set_dict(codecCtxDec.get(), &pDict))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set threads for decode (codec: %s): %s\n"),
            char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    av_dict_free(&pDict);

    codecCtxDec->time_base = av_stream_get_codec_timebase(m_Demux.video.stream);
    codecCtxDec->pkt_timebase = m_Demux.video.stream->time_base;
    if (0 > (ret = avcodec_open2(codecCtxDec.get(), codecDecode, nullptr))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to open decoder for %s: %s\n"), char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameDec(av_frame_alloc(), RGYAVDeleter<AVFrame>(av_frame_free));
    bool got_frame = false;
    for (uint32_t i = 0; i < m_Demux.qVideoPkt.size() && !got_frame; i++) {
        AVPacket *pkt = &m_Demux.qVideoPkt.get(i)->data;
        ret = avcodec_send_packet(codecCtxDec.get(), pkt);

        if (ret == AVERROR_EOF) { //これ以上パケットを送れない
            AddMessage(RGY_LOG_DEBUG, _T("failed to send packet to video decoder, already flushed: %s.\n"), qsv_av_err2str(ret).c_str());
        } else if (ret < 0 && ret != AVERROR(EAGAIN)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to send packet to video decoder: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        ret = avcodec_receive_frame(codecCtxDec.get(), frameDec.get());
        if (ret == AVERROR(EAGAIN)) { //もっとパケットを送る必要がある
            continue;
        }
        if (ret == AVERROR_EOF) {
            //最後まで読み込んだ
            return RGY_ERR_MORE_DATA;
        }
        if (ret < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to receive frame from video decoder: %s.\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        got_frame = true;
    }
    if (got_frame) {
        auto side_data = av_frame_get_side_data(frameDec.get(), AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        if (side_data) {
            m_Demux.video.masteringDisplay = av_mastering_display_metadata_alloc();
            memcpy(m_Demux.video.masteringDisplay, side_data->data, sizeof(AVMasteringDisplayMetadata));
            AddMessage(RGY_LOG_DEBUG, _T("Mastering Display: R(%f,%f) G(%f,%f) B(%f %f) WP(%f, %f) L(%f,%f)\n"),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[0][0]),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[0][1]),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[1][0]),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[1][1]),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[2][0]),
                av_q2d(m_Demux.video.masteringDisplay->display_primaries[2][1]),
                av_q2d(m_Demux.video.masteringDisplay->white_point[0]), av_q2d(m_Demux.video.masteringDisplay->white_point[1]),
                av_q2d(m_Demux.video.masteringDisplay->min_luminance), av_q2d(m_Demux.video.masteringDisplay->max_luminance));
        }
        side_data = av_frame_get_side_data(frameDec.get(), AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        if (side_data) {
            size_t st_size = 0;
            m_Demux.video.contentLight = av_content_light_metadata_alloc(&st_size);
            memcpy(m_Demux.video.contentLight, side_data->data, st_size);
            AddMessage(RGY_LOG_DEBUG, _T("MaxCLL=%d, MaxFALL=%d\n"), m_Demux.video.contentLight->MaxCLL, m_Demux.video.contentLight->MaxFALL);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputAvcodec::parseHDR10plus(AVPacket *pkt) {
    if (m_Demux.video.stream->codecpar->codec_id != AV_CODEC_ID_HEVC) {
        return RGY_ERR_NONE;
    }
    const auto nal_list = parse_nal_unit_hevc(pkt->data, pkt->size);
    for (const auto& nal_unit : nal_list) {
        if (nal_unit.type != NALU_HEVC_PREFIX_SEI) {
            continue;
        }
        const uint8_t *ptr = nal_unit.ptr;
        //nal header
        if (ptr[0] == 0x00
            && ptr[1] == 0x00
            && ptr[2] == 0x01) {
            ptr += 3;
        } else if (ptr[0] == 0x00
            && ptr[1] == 0x00
            && ptr[2] == 0x00
            && ptr[3] == 0x01) {
            ptr += 4;
        } else {
            continue;
        }
        if (ptr[0] == ((uint8_t)NALU_HEVC_PREFIX_SEI << 1)
            && ptr[1] == 0x01
            && ptr[2] == USER_DATA_REGISTERED_ITU_T_T35) {
            ptr += 3;
            const auto data_unnal = unnal(ptr, nal_unit.ptr + nal_unit.size - ptr);
            ptr = data_unnal.data();
            size_t size = 0;
            while (*ptr == 0xff) {
                size += *ptr++;
            }
            size += *ptr++;
            //AVDictionaryに格納するため、base64 encodeを行う
            const auto encoded = cppcodec::base64_rfc4648::encode(ptr, size);
            AVDictionary *frameDict = nullptr;
            int ret = av_dict_set(&frameDict, HDR10PLUS_METADATA_KEY, encoded.c_str(), 0);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to set dictionary for key=%s\n"), char_to_tstring(HDR10PLUS_METADATA_KEY).c_str());
                return RGY_ERR_UNKNOWN;
            }
            int frameDictSize = 0;
            uint8_t *frameDictData = av_packet_pack_dictionary(frameDict, &frameDictSize);
            if (frameDictData == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to pack dictionary for key=%s\n"), char_to_tstring(HDR10PLUS_METADATA_KEY).c_str());
                return RGY_ERR_UNKNOWN;
            }
            ret = av_packet_add_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA, frameDictData, frameDictSize);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to add side packet hdr10plus metadata.\n"), char_to_tstring(HDR10PLUS_METADATA_KEY).c_str());
                return RGY_ERR_UNKNOWN;
            }
            av_dict_free(&frameDict);
            AddMessage(RGY_LOG_TRACE, _T("Added hdr10plus metadata to packet timestamp %lld (%s), size %d, encoded size %d\n"), pkt->pts,
                getTimestampString(pkt->pts, m_Demux.format.formatCtx->streams[pkt->stream_index]->time_base).c_str(), size, (int)encoded.size());
        }
    }
    return RGY_ERR_NONE;
}

RGYFrameDataHDR10plus *RGYInputAvcodec::getHDR10plusMetaData(const AVFrame *frame) {
    auto ptrDictMetadata = av_dict_get(frame->metadata, HDR10PLUS_METADATA_KEY, nullptr, 0);
    if (ptrDictMetadata) {
        const auto ptrDictMetadataValLen = strlen(ptrDictMetadata->value);
        const auto decoded = cppcodec::base64_rfc4648::decode(ptrDictMetadata->value, ptrDictMetadataValLen);
        AddMessage(RGY_LOG_TRACE, _T("Got hdr10plus metadata to packet timestamp %lld (%s), size %d, decoded size %d\n"), frame->pts,
            getTimestampString(frame->pts, m_Demux.video.stream->time_base).c_str(), ptrDictMetadataValLen, (int)decoded.size());
        return new RGYFrameDataHDR10plus(decoded.data(), decoded.size(), frame->pts);
    }
    return nullptr;
}

RGYFrameDataHDR10plus *RGYInputAvcodec::getHDR10plusMetaData(const AVPacket *pkt) {
    int side_data_size = 0;
    auto side_data = av_packet_get_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA, &side_data_size);
    if (side_data) {
        AVDictionary *dict = nullptr;
        auto ret = av_packet_unpack_dictionary(side_data, side_data_size, &dict);
        if (ret == 0) {
            const auto ptrDictMetadata = av_dict_get(dict, HDR10PLUS_METADATA_KEY, nullptr, 0);
            if (ptrDictMetadata) {
                const auto ptrDictMetadataValLen = strlen(ptrDictMetadata->value);
                const auto decoded = cppcodec::base64_rfc4648::decode(ptrDictMetadata->value, ptrDictMetadataValLen);
                AddMessage(RGY_LOG_TRACE, _T("Got hdr10plus metadata to packet timestamp %lld (%s), size %d, decoded size %d\n"), pkt->pts,
                    getTimestampString(pkt->pts, m_Demux.video.stream->time_base).c_str(), ptrDictMetadataValLen, (int)decoded.size());
                return new RGYFrameDataHDR10plus(decoded.data(), decoded.size(), pkt->pts);
            }
        }
    }
    return nullptr;
}

#pragma warning(push)
#pragma warning(disable:4100)
#pragma warning(disable:4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYInputAvcodec::Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) {
    const RGYInputAvcodecPrm *input_prm = dynamic_cast<const RGYInputAvcodecPrm*>(prm);

    if (input_prm->readVideo) {
        if (inputInfo->type != RGY_INPUT_FMT_AVANY) {
            m_readerName = (inputInfo->type != RGY_INPUT_FMT_AVSW) ? _T("av" DECODER_NAME) : _T("avsw");
        }
    } else {
        m_readerName = _T("avsw");
    }

    m_Demux.video.readVideo = input_prm->readVideo;
    m_Demux.thread.queueInfo = input_prm->queueInfo;
    if (input_prm->readVideo) {
        m_inputVideoInfo = *inputInfo;
    } else {
        m_inputVideoInfo = VideoInfo();
    }

    if (!check_avcodec_dll()) {
        AddMessage(RGY_LOG_ERROR, error_mes_avcodec_dll_not_found());
        return RGY_ERR_NULL_PTR;
    }

    m_convert = std::unique_ptr<RGYConvertCSP>(new RGYConvertCSP(prm->threadCsp));

    for (int i = 0; i < input_prm->nAudioSelectCount; i++) {
        tstring audioLog = strsprintf(_T("select audio track %s, codec %s"),
            (input_prm->ppAudioSelect[i]->trackID) ? strsprintf(_T("#%d"), input_prm->ppAudioSelect[i]->trackID).c_str() : _T("all"),
            input_prm->ppAudioSelect[i]->encCodec.c_str());
        if (input_prm->ppAudioSelect[i]->extractFormat.length() > 0) {
            audioLog += tstring(_T("format ")) + input_prm->ppAudioSelect[i]->extractFormat;
        }
        if (input_prm->ppAudioSelect[i]->encCodec.length() > 0
            && !avcodecIsCopy(input_prm->ppAudioSelect[i]->encCodec)) {
            audioLog += strsprintf(_T("bitrate %d"), input_prm->ppAudioSelect[i]->encBitrate);
        }
        if (input_prm->ppAudioSelect[i]->extractFilename.length() > 0) {
            audioLog += tstring(_T("filename \"")) + input_prm->ppAudioSelect[i]->extractFilename + tstring(_T("\""));
        }
        AddMessage(RGY_LOG_DEBUG, audioLog);
    }

    av_log_set_level((m_printMes->getLogLevel() == RGY_LOG_DEBUG) ?  AV_LOG_DEBUG : RGY_AV_LOG_LEVEL);
    av_qsv_log_set(m_printMes);
    if (input_prm->caption2ass != FORMAT_INVALID) {
        auto err = m_cap2ass.init(m_printMes, input_prm->caption2ass);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    int ret = 0;
    std::string filename_char;
    if (0 == tchar_to_string(strFileName, filename_char, CP_UTF8)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to convert filename to utf-8 characters.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    m_Demux.format.isPipe = (0 == strcmp(filename_char.c_str(), "-"))
        || (0 == strncmp(filename_char.c_str(), "pipe:", strlen("pipe:")))
        || filename_char.c_str() == strstr(filename_char.c_str(), R"(\\.\pipe\)");
    m_Demux.format.formatCtx = avformat_alloc_context();
    m_Demux.format.analyzeSec = input_prm->analyzeSec;
    if (input_prm->probesize >= 0 || input_prm->analyzeSec >= 0) {
        const int64_t probesize = (input_prm->probesize > 0) ? input_prm->probesize : 1 << 29;
        if (0 != (ret = av_opt_set_int(m_Demux.format.formatCtx, "probesize", probesize, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set probesize to %s: error %d\n"), rgy_print_num_with_siprefix(probesize).c_str(), ret);
            return RGY_ERR_INVALID_PARAM;
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("set probesize: %s\n"), rgy_print_num_with_siprefix(probesize).c_str());
        }
    }
    if (0 == strcmp(filename_char.c_str(), "-")) {
#if defined(_WIN32) || defined(_WIN64)
        if (_setmode(_fileno(stdin), _O_BINARY) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to switch stdin to binary mode.\n"));
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        AddMessage(RGY_LOG_DEBUG, _T("input source set to stdin.\n"));
        filename_char = "pipe:0";
    }
    //ts向けの設定
    av_dict_set(&m_Demux.format.formatOptions, "scan_all_pmts", "1", 0);

    for (const auto& inputOpt : input_prm->inputOpt) {
        const std::string optName = tchar_to_string(inputOpt.first);
        const std::string optValue = tchar_to_string(inputOpt.second);
        const int err = av_dict_set(&m_Demux.format.formatOptions, optName.c_str(), optValue.c_str(), 0);
        if (err < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set input opt: %s = %s.\n"), inputOpt.first.c_str(), inputOpt.second.c_str());
            return RGY_ERR_INVALID_PARAM;
        }
        AddMessage(RGY_LOG_DEBUG, _T("set input opt: %s = %s.\n"), inputOpt.first.c_str(), inputOpt.second.c_str());
    }
    //入力フォーマットが指定されていれば、それを渡す
    AVInputFormat *inFormat = nullptr;
    if (input_prm->pInputFormat) {
        if (nullptr == (inFormat = av_find_input_format(tchar_to_string(input_prm->pInputFormat).c_str()))) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown Input format: %s.\n"), input_prm->pInputFormat);
            return RGY_ERR_INVALID_FORMAT;
        }
    }

#if USE_CUSTOM_INPUT
    if (m_Demux.format.isPipe || (!usingAVProtocols(filename_char, 0) && input_prm->inputOpt.size() == 0 && (inFormat == nullptr || !(inFormat->flags & (AVFMT_NEEDNUMBER | AVFMT_NOFILE))))) {
        if (0 == _tcscmp(strFileName, _T("-"))) {
            m_Demux.format.fpInput = stdin;
        } else {
            m_Demux.format.fpInput = _tfsopen(strFileName, _T("rb"), _SH_DENYWR);
            if (m_Demux.format.fpInput == NULL) {
                errno_t error = errno;
                AddMessage(RGY_LOG_ERROR, _T("failed to open input file \"%s\": %s.\n"), strFileName, _tcserror(error));
                return RGY_ERR_FILE_OPEN; // Couldn't open file
            }
            if (!m_Demux.format.isPipe) {
                if (rgy_get_filesize(strFileName, &m_Demux.format.inputFilesize) && m_Demux.format.inputFilesize == 0) {
                    AddMessage(RGY_LOG_ERROR, _T("size of input file \"%s\" is 0\n"), strFileName);
                    return RGY_ERR_FILE_OPEN;
                }
            }
        }
        m_Demux.format.inputBufferSize = 4 * 1024;
        m_Demux.format.inputBuffer = (char *)av_malloc(m_Demux.format.inputBufferSize);
        if (NULL == (m_Demux.format.formatCtx->pb = avio_alloc_context((unsigned char *)m_Demux.format.inputBuffer, m_Demux.format.inputBufferSize, 0, this, funcReadPacket, funcWritePacket, (m_Demux.format.isPipe) ? nullptr : funcSeek))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return RGY_ERR_NULL_PTR;
        }
    } else if (m_cap2ass.enabled()) {
        AddMessage(RGY_LOG_ERROR, (input_prm->inputOpt.size() > 0)
            ? _T("--caption2ass cannot be used with --input-option.\n")
            : _T("--caption2ass only supported when input is file or pipe.\n"));
        m_cap2ass.disable();
    }
#endif
    //ファイルのオープン
    if ((ret = avformat_open_input(&(m_Demux.format.formatCtx), filename_char.c_str(), inFormat, &m_Demux.format.formatOptions)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("error opening file \"%s\": %s\n"), char_to_tstring(filename_char, CP_UTF8).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_FILE_OPEN; // Couldn't open file
    }
    AddMessage(RGY_LOG_DEBUG, _T("opened file \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());

    //不正なオプションを渡していないかチェック
    for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(m_Demux.format.formatOptions, "", t, AV_DICT_IGNORE_SUFFIX));) {
        if (strcmp(t->key, "scan_all_pmts") != 0) {
            AddMessage(RGY_LOG_WARN, _T("Unknown input option: %s=%s, ignored.\n"),
                char_to_tstring(t->key).c_str(),
                char_to_tstring(t->value).c_str());
        }
    }

    if (m_Demux.format.analyzeSec >= 0) {
        const int64_t value = (int64_t)(m_Demux.format.analyzeSec * AV_TIME_BASE + 0.5);
        if (0 != (ret = av_opt_set_int(m_Demux.format.formatCtx, "analyzeduration", value, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set analyzeduration to %.2f sec, error %s\n"), m_Demux.format.analyzeSec, qsv_av_err2str(ret).c_str());
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("set analyzeduration: %.2f sec\n"), m_Demux.format.analyzeSec);
        }
    }

    if (avformat_find_stream_info(m_Demux.format.formatCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("error finding stream information.\n"));
        return RGY_ERR_UNKNOWN; // Couldn't find stream information
    }
    AddMessage(RGY_LOG_DEBUG, _T("got stream information.\n"));
    av_dump_format(m_Demux.format.formatCtx, 0, filename_char.c_str(), 0);
    //dump_format(dec.formatCtx, 0, argv[1], 0);

    //キュー関連初期化
    //getFirstFramePosAndFrameRateで大量にパケットを突っ込む可能性があるので、この段階ではcapacityは無限大にしておく
    m_Demux.qVideoPkt.init(4096, SIZE_MAX, 4);
    m_Demux.qVideoPkt.set_keep_length(1);
    m_Demux.qStreamPktL2.init(4096);

    //動画ストリームを探す
    //動画ストリームは動画を処理しなかったとしても同期のため必要
    auto videoStreams = getStreamIndex(AVMEDIA_TYPE_VIDEO);
    if (videoStreams.size()) {
        if (input_prm->videoTrack) {
            if (videoStreams.size() < (uint32_t)std::abs(input_prm->videoTrack)) {
                AddMessage(RGY_LOG_ERROR, _T("track %d was selected for video, but input only contains %d video tracks.\n"), input_prm->videoTrack, videoStreams.size());
                return RGY_ERR_INVALID_VIDEO_PARAM;
            } else if (input_prm->videoTrack < 0) {
                //逆順に並べ替え
                std::reverse(videoStreams.begin(), videoStreams.end());
            }
            m_Demux.video.index = videoStreams[std::abs(input_prm->videoTrack)-1];
        } else if (input_prm->videoStreamId) {
            auto streamIndexFound = std::find_if(videoStreams.begin(), videoStreams.end(), [formatCtx = m_Demux.format.formatCtx, nSearchId = input_prm->videoStreamId](int nStreamIndex) {
                return (formatCtx->streams[nStreamIndex]->id == nSearchId);
            });
            if (streamIndexFound == videoStreams.end()) {
                AddMessage(RGY_LOG_ERROR, _T("stream id %d (0x%x) not found in video tracks.\n"), input_prm->videoStreamId, input_prm->videoStreamId);
                return RGY_ERR_INVALID_VIDEO_PARAM;
            }
            m_Demux.video.index = *streamIndexFound;
        } else {
            m_Demux.video.index = videoStreams[0];
        }
        auto selectedStream = std::find(videoStreams.begin(), videoStreams.end(), m_Demux.video.index);
        if (selectedStream == videoStreams.end()) {
            AddMessage(RGY_LOG_ERROR, _T("video stream lost!\n"));
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        //もし、選択された動画ストリームが先頭にないのなら、先頭に入れ替える
        if (selectedStream != videoStreams.begin()) {
            int nSelectedStreamIndex = *selectedStream;
            videoStreams.erase(selectedStream);
            videoStreams.insert(videoStreams.begin(), nSelectedStreamIndex);
        }
        AddMessage(RGY_LOG_DEBUG, _T("found video stream, stream idx %d\n"), m_Demux.video.index);

        auto stream = m_Demux.format.formatCtx->streams[m_Demux.video.index];
        //HEVC入力の際に大量にメッセージが出て劇的に遅くなることがあるのを回避
#if 0
        if (stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            AVDictionary *pDict = nullptr;
            av_dict_set_int(&pDict, "log_level_offset", AV_LOG_ERROR, 0);
            if (0 > (ret = av_opt_set_dict(stream->codec, &pDict))) {
                AddMessage(RGY_LOG_WARN, _T("failed to set log_level_offset for HEVC codec reader.\n"));
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("set log_level_offset for HEVC codec reader.\n"));
            }
            av_dict_free(&pDict);
        }
#endif
        m_Demux.video.stream = stream;
    }

    //音声ストリームを探す
    if (input_prm->readAudio || input_prm->readSubtitle || input_prm->readData || input_prm->readAttachment) {
        vector<int> mediaStreams;
        if (input_prm->readAudio) {
            auto audioStreams = getStreamIndex(AVMEDIA_TYPE_AUDIO, &videoStreams);
            //他のファイルから音声を読み込む場合もあるので、ここでチェックはできない
            //if (audioStreams.size() == 0) {
            //    AddMessage(RGY_LOG_ERROR, _T("--audio-encode/--audio-copy/--audio-file is set, but no audio stream found.\n"));
            //    return RGY_ERR_NOT_FOUND;
            //}
            m_Demux.format.audioTracks = (int)audioStreams.size();
            vector_cat(mediaStreams, audioStreams);
        }
        if (input_prm->readSubtitle) {
            auto subStreams = getStreamIndex(AVMEDIA_TYPE_SUBTITLE, &videoStreams);
            if (subStreams.size() == 0 && !m_cap2ass.enabled()) {
                AddMessage(RGY_LOG_WARN, _T("--sub-copy%s is set, but no subtitle stream found.\n"), (ENCODER_NVENC) ? _T("/--vpp-subburn") : _T(""));
            } else {
                m_Demux.format.subtitleTracks = (int)subStreams.size();
                vector_cat(mediaStreams, subStreams);
            }
        }
        if (input_prm->readData) {
            auto dataStreams = getStreamIndex(AVMEDIA_TYPE_DATA, &videoStreams);
            m_Demux.format.dataTracks = (int)dataStreams.size();
            vector_cat(mediaStreams, dataStreams);
        }
        if (input_prm->readAttachment) {
            auto attachmentStreams = getStreamIndex(AVMEDIA_TYPE_ATTACHMENT, &videoStreams);
            m_Demux.format.attachmentTracks = (int)attachmentStreams.size();
            vector_cat(mediaStreams, attachmentStreams);
        }
        for (int iTrack = 0; iTrack < (int)mediaStreams.size(); iTrack++) {
            const AVStream *srcStream = m_Demux.format.formatCtx->streams[mediaStreams[iTrack]];
            const AVCodecID codecId = srcStream->codecpar->codec_id;
            AVMediaType mediaType = srcStream->codecpar->codec_type;
            bool useStream = false;
            AudioSelect *pAudioSelect = nullptr; //トラックに対応するAudioSelect (字幕ストリームの場合はnullptrのまま)
            if (mediaType == AVMEDIA_TYPE_ATTACHMENT
                || (srcStream->disposition & AV_DISPOSITION_ATTACHED_PIC) != 0) {
                //Attachmentの場合
                for (int i = 0; !useStream && i < input_prm->nAttachmentSelectCount; i++) {
                    if (input_prm->ppAttachmentSelect[i]->trackID == 0 //特に指定なし = 全指定かどうか
                        || input_prm->ppAttachmentSelect[i]->trackID - 1 == (iTrack - m_Demux.format.audioTracks - m_Demux.format.subtitleTracks - m_Demux.format.dataTracks)) {
                        useStream = true;
                        mediaType = AVMEDIA_TYPE_ATTACHMENT;
                    }
                }
            } else if (mediaType == AVMEDIA_TYPE_SUBTITLE) {
                //字幕・データの場合
                for (int i = 0; !useStream && i < input_prm->nSubtitleSelectCount; i++) {
                    if (input_prm->ppSubtitleSelect[i]->trackID == 0 //特に指定なし = 全指定かどうか
                        || (input_prm->ppSubtitleSelect[i]->trackID == TRACK_SELECT_BY_LANG && isSelectedLangTrack(input_prm->ppSubtitleSelect[i]->lang, srcStream))
                        || input_prm->ppSubtitleSelect[i]->trackID - 1 == (iTrack - m_Demux.format.audioTracks)) {
                        useStream = true;
                    }
                }
            } else if (mediaType == AVMEDIA_TYPE_DATA) {
                //データの場合
                for (int i = 0; !useStream && i < input_prm->nDataSelectCount; i++) {
                    if (input_prm->ppDataSelect[i]->trackID == 0 //特に指定なし = 全指定かどうか
                        || (input_prm->ppDataSelect[i]->trackID == TRACK_SELECT_BY_LANG && isSelectedLangTrack(input_prm->ppDataSelect[i]->lang, srcStream))
                        || input_prm->ppDataSelect[i]->trackID - 1 == (iTrack - m_Demux.format.audioTracks - m_Demux.format.subtitleTracks)) {
                        useStream = true;
                    }
                }
            } else {
                //音声の場合
                for (int i = 0; !useStream && i < input_prm->nAudioSelectCount; i++) {
                    if ((input_prm->ppAudioSelect[i]->trackID == TRACK_SELECT_BY_LANG && isSelectedLangTrack(input_prm->ppAudioSelect[i]->lang, srcStream))
                        || (input_prm->ppAudioSelect[i]->trackID - 1 == (iTrack))) {
                        useStream = true;
                        pAudioSelect = input_prm->ppAudioSelect[i];
                    }
                }
                if (pAudioSelect == nullptr) {
                    //見つからなかったら、全指定(trackID = 0)のものを使用する
                    for (int i = 0; !useStream && i < input_prm->nAudioSelectCount; i++) {
                        if (input_prm->ppAudioSelect[i]->trackID == 0) {
                            useStream = true;
                            pAudioSelect = input_prm->ppAudioSelect[i];
                        }
                    }
                }
            }
            if (useStream) {
                //存在するチャンネルまでのchannel_layoutのマスクを作成する
                //特に引数を指定せず--audio-channel-layoutを指定したときには、
                //pnStreamChannelsはchannelの存在しない不要なビットまで立っているのをここで修正
                if (pAudioSelect //字幕ストリームの場合は無視
                    && isSplitChannelAuto(pAudioSelect->streamChannelSelect)) {
                    const uint64_t channel_layout_mask = UINT64_MAX >> (sizeof(channel_layout_mask) * 8 - m_Demux.format.formatCtx->streams[mediaStreams[iTrack]]->codecpar->channels);
                    for (uint32_t iSubStream = 0; iSubStream < MAX_SPLIT_CHANNELS; iSubStream++) {
                        pAudioSelect->streamChannelSelect[iSubStream] &= channel_layout_mask;
                    }
                    for (uint32_t iSubStream = 0; iSubStream < MAX_SPLIT_CHANNELS; iSubStream++) {
                        pAudioSelect->streamChannelOut[iSubStream] &= channel_layout_mask;
                    }
                }

                //必要であれば、サブストリームを追加する
                for (uint32_t iSubStream = 0; iSubStream == 0 || //初回は字幕・音声含め、かならず登録する必要がある
                    (iSubStream < MAX_SPLIT_CHANNELS //最大サブストリームの上限
                        && pAudioSelect != nullptr //字幕ではない
                        && pAudioSelect->streamChannelSelect[iSubStream]); //audio-splitが指定されている
                    iSubStream++) {
                    AVDemuxStream stream = { 0 };
                    switch (mediaType) {
                    case AVMEDIA_TYPE_SUBTITLE:
                        stream.trackId = trackFullID(AVMEDIA_TYPE_SUBTITLE, iTrack - m_Demux.format.audioTracks + input_prm->trackStartSubtitle);
                        break;
                    case AVMEDIA_TYPE_DATA:
                        stream.trackId = trackFullID(AVMEDIA_TYPE_DATA, iTrack - m_Demux.format.audioTracks - m_Demux.format.subtitleTracks + input_prm->trackStartData);
                        break;
                    case AVMEDIA_TYPE_ATTACHMENT:
                        stream.trackId = trackFullID(AVMEDIA_TYPE_ATTACHMENT, iTrack - m_Demux.format.audioTracks - m_Demux.format.subtitleTracks - m_Demux.format.dataTracks + input_prm->trackStartData);
                        break;
                    case AVMEDIA_TYPE_AUDIO:
                        stream.trackId = trackFullID(AVMEDIA_TYPE_AUDIO, iTrack + input_prm->trackStartAudio);
                        break;
                    default:
                        AddMessage(RGY_LOG_ERROR, _T("Unknoen media type!"));
                        return RGY_ERR_INVALID_PARAM;
                    }
                    stream.index = mediaStreams[iTrack];
                    stream.subStreamId = iSubStream;
                    stream.sourceFileIndex = input_prm->fileIndex;
                    stream.stream = m_Demux.format.formatCtx->streams[stream.index];
                    stream.timebase = stream.stream->time_base;
                    stream.extractErrExcess = 0;
                    stream.appliedTrimBlock = -1;
                    stream.trimOffset = 0;
                    stream.aud0_fin = AV_NOPTS_VALUE;
                    strncpy_s(stream.lang, _countof(stream.lang), getTrackLang(m_Demux.format.formatCtx->streams[stream.index]).c_str(), 3);
                    if (pAudioSelect) {
                        stream.addDelayMs = pAudioSelect->addDelayMs;
                        memcpy(stream.streamChannelSelect, pAudioSelect->streamChannelSelect, sizeof(stream.streamChannelSelect));
                        memcpy(stream.streamChannelOut,    pAudioSelect->streamChannelOut,    sizeof(stream.streamChannelOut));
                    }
                    m_Demux.stream.push_back(stream);
                    AddMessage(RGY_LOG_DEBUG, _T("found %s stream, stream idx %d, trackID %d.%d, %s, frame_size %d, timebase %d/%d, delay %d ms\n"),
                        get_media_type_string(codecId).c_str(),
                        stream.index, trackID(stream.trackId), stream.subStreamId, char_to_tstring(avcodec_get_name(codecId)).c_str(),
                        stream.stream->codecpar->frame_size, stream.stream->time_base.num, stream.stream->time_base.den, stream.addDelayMs);
                }
            }
        }
        //指定されたすべての音声トラックが発見されたかを確認する
        for (int i = 0; i < input_prm->nAudioSelectCount; i++) {
            //全指定のトラック=0は無視
            if (input_prm->ppAudioSelect[i]->trackID > 0) {
                bool audioFound = false;
                for (const auto& stream : m_Demux.stream) {
                    if (trackID(stream.trackId) == input_prm->ppAudioSelect[i]->trackID) {
                        audioFound = true;
                        break;
                    }
                }
                if (!audioFound) {
                    AddMessage(RGY_LOG_WARN, _T("could not find audio track #%d\n"), input_prm->ppAudioSelect[i]->trackID);
                }
            }
        }
    }

    if (m_cap2ass.enabled()) {
        const int maxSubTrackId = std::accumulate(m_Demux.stream.begin(), m_Demux.stream.end(), 0, [](int a, const AVDemuxStream& b) {
            return trackMediaType(b.trackId) == AVMEDIA_TYPE_SUBTITLE ? std::max(a, trackID(b.trackId)) : a;
        });
        m_cap2ass.setIndex(m_Demux.format.formatCtx->nb_streams, trackFullID(AVMEDIA_TYPE_SUBTITLE, maxSubTrackId+1));
        m_Demux.stream.push_back(m_cap2ass.stream());
        m_Demux.format.subtitleTracks++;
    }

    if (input_prm->readChapter) {
        m_Demux.chapter = make_vector((const AVChapter **)m_Demux.format.formatCtx->chapters, m_Demux.format.formatCtx->nb_chapters);
    }

    //動画処理の初期化を行う
    if (input_prm->readVideo) {
        if (m_Demux.video.stream == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("unable to find video stream.\n"));
            return RGY_ERR_NOT_FOUND;
        }
        AddMessage(RGY_LOG_DEBUG, _T("use video stream #%d for input, codec %s, stream time_base %d/%d, codec_timebase %d/%d.\n"),
            m_Demux.video.stream->index,
            char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(),
            m_Demux.video.stream->time_base.num, m_Demux.video.stream->time_base.den,
            av_stream_get_codec_timebase(m_Demux.video.stream).num, av_stream_get_codec_timebase(m_Demux.video.stream).den);

        m_logFramePosList.clear();
        if (input_prm->logFramePosList.length() > 0) {
            m_logFramePosList = input_prm->logFramePosList;
            AddMessage(RGY_LOG_DEBUG, _T("Opened framepos log file: \"%s\"\n"), input_prm->logCopyFrameData.c_str());
        }

        m_Demux.video.hdr10plusMetadataCopy = input_prm->hdr10plusMetadataCopy;
        AddMessage(RGY_LOG_DEBUG, _T("hdr10plusMetadataCopy: %s\n"), m_Demux.video.hdr10plusMetadataCopy ? _T("on") : _T("off"));
        m_Demux.video.HWDecodeDeviceId = -1;
        if (m_inputVideoInfo.type != RGY_INPUT_FMT_AVSW) {
            for (const auto& devCodecCsp : *input_prm->HWDecCodecCsp) {
                //VC-1では、pixelFormatがAV_PIX_FMT_NONEとなっている場合があるので、その場合は試しにAV_PIX_FMT_YUV420Pとして処理してみる
                if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_VC1 && (AVPixelFormat)m_Demux.video.stream->codecpar->format == AV_PIX_FMT_NONE) {
                    AddMessage(RGY_LOG_WARN, _T("pixel format of input file reported as %s, will try decode as %s.\n"),
                        char_to_tstring(av_get_pix_fmt_name(AV_PIX_FMT_NONE)).c_str(),
                        char_to_tstring(av_get_pix_fmt_name((AVPixelFormat)m_Demux.video.stream->codecpar->format)).c_str());
                    m_Demux.video.stream->codecpar->format = AV_PIX_FMT_YUV420P;
                }
                m_inputVideoInfo.codec = checkHWDecoderAvailable(
                    m_Demux.video.stream->codecpar->codec_id, (AVPixelFormat)m_Demux.video.stream->codecpar->format, &devCodecCsp.second);
                if (m_inputVideoInfo.codec != RGY_CODEC_UNKNOWN) {
                    m_Demux.video.HWDecodeDeviceId = devCodecCsp.first;
                    break;
                }
                if (m_inputVideoInfo.type != RGY_INPUT_FMT_AVHW) {
                    break; //RGY_INPUT_FMT_AVANYのときは、HWデコードできなくても優先度の高いGPUを使用する
                }
            }
            if (m_inputVideoInfo.codec == RGY_CODEC_UNKNOWN
                //wmv3はAdvanced Profile (3)のみの対応
                || (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_WMV3 && m_Demux.video.stream->codecpar->profile != 3)) {
                if (m_inputVideoInfo.type == RGY_INPUT_FMT_AVHW) {
                    //HWデコードが指定されている場合にはエラー終了する
                    AddMessage(RGY_LOG_ERROR, _T("codec %s(%s) unable to decode by " DECODER_NAME ".\n"),
                        char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(),
                        char_to_tstring(av_get_pix_fmt_name((AVPixelFormat)m_Demux.video.stream->codecpar->format)).c_str());
                    return RGY_ERR_INVALID_CODEC;
                }
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("can be decoded by %s.\n"), _T(DECODER_NAME));
            }
        }
        m_readerName = (m_Demux.video.HWDecodeDeviceId >= 0) ? _T("av" DECODER_NAME) : _T("avsw");
        m_inputVideoInfo.type = (m_Demux.video.HWDecodeDeviceId >= 0) ? RGY_INPUT_FMT_AVHW : RGY_INPUT_FMT_AVSW;
        //念のため初期化
        m_trimParam.list.clear();
        m_trimParam.offset = 0;

        //必要ならbitstream filterを初期化
        if ((m_inputVideoInfo.codec != RGY_CODEC_UNKNOWN || m_Demux.video.hdr10plusMetadataCopy)
            && m_Demux.video.stream->codecpar->extradata && m_Demux.video.stream->codecpar->extradata[0] == 1) {
            RGY_ERR sts = initVideoBsfs();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init bsfs.\n"));
                return sts;
            }
        //HWデコードする場合には、ヘッダーが必要
        } else if (m_Demux.video.HWDecodeDeviceId >= 0
            && (m_inputVideoInfo.codec != RGY_CODEC_VP8 && m_inputVideoInfo.codec != RGY_CODEC_VP9)
            && m_Demux.video.stream->codecpar->extradata == nullptr
            && m_Demux.video.stream->codecpar->extradata_size == 0) {
            AddMessage(RGY_LOG_ERROR, _T("video header not extracted by libavcodec.\n"));
            return RGY_ERR_UNKNOWN;
        }
        if (m_Demux.video.stream->codecpar->extradata_size) {
            m_inputVideoInfo.codecExtra = m_Demux.video.stream->codecpar->extradata;
            m_inputVideoInfo.codecExtraSize = m_Demux.video.stream->codecpar->extradata_size;
        }

        AddMessage(RGY_LOG_DEBUG, _T("start predecode.\n"));

        //ヘッダーの取得を確認する
        RGY_ERR sts = RGY_ERR_NONE;
        RGYBitstream bitstream = RGYBitstreamInit();
        if (m_Demux.video.stream->codecpar->extradata) {
            sts = GetHeader(&bitstream);
            if (sts != RGY_ERR_NONE) {
                //bsfsがオンであるために失敗した可能性がある
                //一部のHEVCファイルでは、codecpar->extradataにヘッダの一部(たとえばextradata_size=23)しか
                //格納されていないため、正常にヘッダの取得が行えない。
                //またこのため、bsfs(HEVCmp42AnnexB)やparserが正常に動作しない。
                //一方、swデコーダではbsfsは不要であることから、とりあえずそちらに切り替えてしまって動作させることにした
                if (sts == RGY_ERR_MORE_DATA
                    && (m_Demux.video.bsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB)
                    && !m_Demux.video.hdr10plusMetadataCopy) {
                    AddMessage(RGY_LOG_WARN, _T("Failed to get header for hardware decoder, switching to software decoder...\n"));
                    m_inputVideoInfo.codec = RGY_CODEC_UNKNOWN; //hwデコードをオフにする
                    m_Demux.video.HWDecodeDeviceId = -1;
                    //close bitstreamfilter
                    if (m_Demux.video.bsfcCtx) {
                        AddMessage(RGY_LOG_DEBUG, _T("Free bsf...\n"));
                        av_bsf_free(&m_Demux.video.bsfcCtx);
                        AddMessage(RGY_LOG_DEBUG, _T("Freed bsf.\n"));
                    }
                    //bUseHEVCmp42AnnexBも無効化
                    m_Demux.video.bUseHEVCmp42AnnexB = false;
                    if (m_Demux.video.stream->codecpar->extradata_size) {
                        m_inputVideoInfo.codecExtra = m_Demux.video.stream->codecpar->extradata;
                        m_inputVideoInfo.codecExtraSize = m_Demux.video.stream->codecpar->extradata_size;
                    }
                    sts = GetHeader(&bitstream);
                }
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to get header.\n"));
                    return sts;
                }
            }
            m_inputVideoInfo.codecExtra = m_Demux.video.extradata;
            m_inputVideoInfo.codecExtraSize = m_Demux.video.extradataSize;
        }
        if (input_prm->seekSec > 0.0f) {
            AVPacket firstpkt;
            if (getSample(&firstpkt)) { //現在のtimestampを取得する
                AddMessage(RGY_LOG_ERROR, _T("Failed to get firstpkt of video!\n"));
                return RGY_ERR_UNKNOWN;
            }
            const auto seek_time = av_rescale_q(1, av_d2q((double)input_prm->seekSec, 1<<24), m_Demux.video.stream->time_base);
            int seek_ret = av_seek_frame(m_Demux.format.formatCtx, m_Demux.video.index, firstpkt.pts + seek_time, 0);
            if (0 > seek_ret) {
                seek_ret = av_seek_frame(m_Demux.format.formatCtx, m_Demux.video.index, firstpkt.pts + seek_time, AVSEEK_FLAG_ANY);
            }
            av_packet_unref(&firstpkt);
            if (0 > seek_ret) {
                AddMessage(RGY_LOG_ERROR, _T("failed to seek %s.\n"), print_time(input_prm->seekSec).c_str());
                return RGY_ERR_UNKNOWN;
            }
            //seekのために行ったgetSampleの結果は破棄する
            m_Demux.frames.clear();
        }

        //parserはseek後に初期化すること
        //parserが使用されていれば、ここでも使用するようにする
        //たとえば、入力がrawcodecなどでは使用しない
        sts = initVideoParser();
        if (sts != RGY_ERR_MORE_DATA && sts != RGY_ERR_NONE) {
            return sts;
        }
#if _DEBUG
        if (input_prm->logCopyFrameData.length() > 0) {
            if (m_Demux.frames.setLogCopyFrameData(input_prm->logCopyFrameData.c_str())) {
                AddMessage(RGY_LOG_WARN, _T("failed to open copy-framedata log file: \"%s\"\n"), input_prm->logCopyFrameData.c_str());
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("Opened copy-framedata log file: \"%s\"\n"), input_prm->logCopyFrameData.c_str());
            }
        }
#endif

        if (RGY_ERR_NONE != (sts = getFirstFramePosAndFrameRate(input_prm->pTrimList, input_prm->nTrimCount, input_prm->videoDetectPulldown, input_prm->lowLatency, input_prm->videoAvgFramerate))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get first frame position.\n"));
            return sts;
        }
        if (m_cap2ass.enabled()) {
            m_cap2ass.setVidFirstKeyPts(m_Demux.video.streamFirstKeyPts);
        }

        m_trimParam.list = make_vector(input_prm->pTrimList, input_prm->nTrimCount);
        //キーフレームに到達するまでQSVではフレームが出てこない
        //そのぶんのずれを記録しておき、Trim値などに補正をかける
        if (m_trimParam.offset) {
            for (int i = (int)m_trimParam.list.size() - 1; i >= 0; i--) {
                if (m_trimParam.list[i].fin - m_trimParam.offset < 0) {
                    m_trimParam.list.erase(m_trimParam.list.begin() + i);
                } else {
                    m_trimParam.list[i].start = (std::max)(0, m_trimParam.list[i].start - m_trimParam.offset);
                    if (m_trimParam.list[i].fin != TRIM_MAX) {
                        m_trimParam.list[i].fin = (std::max)(0, m_trimParam.list[i].fin - m_trimParam.offset);
                    }
                }
            }
            //ずれが存在し、範囲指定がない場合はダミーの全域指定を追加する
            //これにより、自動的に音声側との同期がとれるようになる
            if (m_trimParam.list.size() == 0) {
                m_trimParam.list.push_back({ 0, TRIM_MAX });
            }
            AddMessage(RGY_LOG_DEBUG, _T("adjust trim by offset %d.\n"), m_trimParam.offset);
        }

        struct pixfmtInfo {
            AVPixelFormat pix_fmt;
            int bit_depth;
            RGY_CHROMAFMT chroma_format;
            RGY_CSP output_csp;
        };

        static const pixfmtInfo pixfmtDataList[] = {
            { AV_PIX_FMT_YUV420P,      8, RGY_CHROMAFMT_YUV420, RGY_CSP_NV12 },
            { AV_PIX_FMT_YUVJ420P,     8, RGY_CHROMAFMT_YUV420, RGY_CSP_NV12 },
            { AV_PIX_FMT_NV12,         8, RGY_CHROMAFMT_YUV420, RGY_CSP_NV12 },
            { AV_PIX_FMT_NV21,         8, RGY_CHROMAFMT_YUV420, RGY_CSP_NV12 },
            { AV_PIX_FMT_YUVJ422P,     8, RGY_CHROMAFMT_YUV422, RGY_CSP_NA },
            { AV_PIX_FMT_YUYV422,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_YUY2 },
            { AV_PIX_FMT_UYVY422,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_NA },
#if ENCODER_QSV || ENCODER_VCEENC
            { AV_PIX_FMT_YUV422P,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV12 },
            { AV_PIX_FMT_NV16,         8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV12 },
#else
            { AV_PIX_FMT_YUV422P,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV16 },
            { AV_PIX_FMT_NV16,         8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV16 },
#endif
            { AV_PIX_FMT_YUV444P,      8, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444 },
            { AV_PIX_FMT_YUVJ444P,     8, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444 },
            { AV_PIX_FMT_YUV420P16LE, 16, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P14LE, 14, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P12LE, 12, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P10LE, 10, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P9LE,   9, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_NV20LE,      10, RGY_CHROMAFMT_YUV420, RGY_CSP_NA },
#if ENCODER_QSV || ENCODER_VCEENC
            { AV_PIX_FMT_YUV422P16LE, 16, RGY_CHROMAFMT_YUV422, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV422P14LE, 14, RGY_CHROMAFMT_YUV422, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV422P12LE, 12, RGY_CHROMAFMT_YUV422, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV422P10LE, 10, RGY_CHROMAFMT_YUV422, RGY_CSP_P010 },
#else
            { AV_PIX_FMT_YUV422P16LE, 16, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P14LE, 14, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P12LE, 12, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P10LE, 10, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
#endif
            { AV_PIX_FMT_YUV444P16LE, 16, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P14LE, 14, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P12LE, 12, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P10LE, 10, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P9LE,   9, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_RGB24,        8, RGY_CHROMAFMT_RGB_PACKED, (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
            { AV_PIX_FMT_BGR24,        8, RGY_CHROMAFMT_RGB_PACKED, (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
            { AV_PIX_FMT_RGBA,         8, RGY_CHROMAFMT_RGB_PACKED, (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
            { AV_PIX_FMT_BGRA,         8, RGY_CHROMAFMT_RGB_PACKED, (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
            { AV_PIX_FMT_GBRP,         8, RGY_CHROMAFMT_RGB,        (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
            { AV_PIX_FMT_GBRAP,        8, RGY_CHROMAFMT_RGB,        (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 },
        };

        const auto pixfmt = (AVPixelFormat)m_Demux.video.stream->codecpar->format;
        const auto pixfmtData = std::find_if(pixfmtDataList, pixfmtDataList + _countof(pixfmtDataList), [pixfmt](const pixfmtInfo& tableData) {
            return tableData.pix_fmt == pixfmt;
        });
        if (pixfmtData == (pixfmtDataList + _countof(pixfmtDataList)) || pixfmtData->output_csp == RGY_CSP_NA) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid pixel format \"%s\" from input file.\n"), char_to_tstring(av_get_pix_fmt_name(pixfmt)).c_str());
            return RGY_ERR_INVALID_COLOR_FORMAT;
        }

        const auto aspectRatio = m_Demux.video.stream->codecpar->sample_aspect_ratio;
        const bool bAspectRatioUnknown = aspectRatio.num * aspectRatio.den <= 0;

        if (!(m_Demux.video.HWDecodeDeviceId >= 0)) {
            if (nullptr == (m_Demux.video.codecDecode = avcodec_find_decoder(m_Demux.video.stream->codecpar->codec_id))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to find decoder"), m_Demux.video.stream->codecpar->codec_id).c_str());
                return RGY_ERR_NOT_FOUND;
            }
            if (nullptr == (m_Demux.video.codecCtxDecode = avcodec_alloc_context3(m_Demux.video.codecDecode))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to allocate decoder"), m_Demux.video.stream->codecpar->codec_id).c_str());
                return RGY_ERR_NULL_PTR;
            }
            unique_ptr_custom<AVCodecParameters> codecParamCopy(avcodec_parameters_alloc(), [](AVCodecParameters *pCodecPar) {
                avcodec_parameters_free(&pCodecPar);
            });
            if (0 > (ret = avcodec_parameters_copy(codecParamCopy.get(), m_Demux.video.stream->codecpar))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            if (m_Demux.video.bsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB) {
                SetExtraData(codecParamCopy.get(), m_Demux.video.extradata, m_Demux.video.extradataSize);
            }
            if (0 > (ret = avcodec_parameters_to_context(m_Demux.video.codecCtxDecode, codecParamCopy.get()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set codec param to context for decoder: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            cpu_info_t cpu_info;
            if (get_cpu_info(&cpu_info)) {
                AVDictionary *pDict = nullptr;
                av_dict_set_int(&pDict, "threads", std::min(cpu_info.logical_cores, 16u), 0);
                if (0 > (ret = av_opt_set_dict(m_Demux.video.codecCtxDecode, &pDict))) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to set threads for decode (codec: %s): %s\n"),
                        char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_UNKNOWN;
                }
                av_dict_free(&pDict);
            }
            m_Demux.video.codecCtxDecode->time_base = av_stream_get_codec_timebase(m_Demux.video.stream);
            m_Demux.video.codecCtxDecode->pkt_timebase = m_Demux.video.stream->time_base;
            if (0 > (ret = avcodec_open2(m_Demux.video.codecCtxDecode, m_Demux.video.codecDecode, nullptr))) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open decoder for %s: %s\n"), char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNSUPPORTED;
            }

            const auto pixCspConv = csp_avpixfmt_to_rgy(m_Demux.video.codecCtxDecode->pix_fmt);
            if (pixCspConv == RGY_CSP_NA) {
                AddMessage(RGY_LOG_ERROR, _T("invalid color format: %s\n"),
                    char_to_tstring(av_get_pix_fmt_name(m_Demux.video.codecCtxDecode->pix_fmt)).c_str());
                return RGY_ERR_INVALID_COLOR_FORMAT;
            }
            m_inputCsp = pixCspConv;
            //出力フォーマットへの直接変換を持たないものは、pixfmtDataListに従う
            const auto prefered_csp = m_inputVideoInfo.csp;
            if (prefered_csp == RGY_CSP_NA) {
                //ロスレスの場合は、入力側で出力フォーマットを決める
                m_inputVideoInfo.csp = pixfmtData->output_csp;
            } else {
                m_inputVideoInfo.csp = (m_convert->getFunc(m_inputCsp, prefered_csp, false, prm->simdCsp) != nullptr) ? prefered_csp : pixfmtData->output_csp;
                //QSVではNV16->P010がサポートされていない
                if (ENCODER_QSV && m_inputVideoInfo.csp == RGY_CSP_NV16 && prefered_csp == RGY_CSP_P010) {
                    m_inputVideoInfo.csp = RGY_CSP_P210;
                }
                //なるべく軽いフォーマットでGPUに転送するように
                if (ENCODER_NVENC
                    && RGY_CSP_BIT_PER_PIXEL[pixfmtData->output_csp] < RGY_CSP_BIT_PER_PIXEL[prefered_csp]
                    && m_convert->getFunc(m_inputCsp, pixfmtData->output_csp, false, prm->simdCsp) != nullptr) {
                    m_inputVideoInfo.csp = pixfmtData->output_csp;
                }
            }
            if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr && m_inputCsp == RGY_CSP_YUY2) {
                //YUY2用の特別処理
                m_inputVideoInfo.csp = RGY_CSP_CHROMA_FORMAT[pixfmtData->output_csp] == RGY_CHROMAFMT_YUV420 ? RGY_CSP_NV12 : RGY_CSP_YUV444;
                m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp);
            }
            if (m_convert->getFunc() == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
                     RGY_CSP_NAMES[pixCspConv], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
                return RGY_ERR_INVALID_COLOR_FORMAT;
            }
            m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp];
            if (cspShiftUsed(m_inputVideoInfo.csp) && RGY_CSP_BIT_DEPTH[m_inputVideoInfo.csp] > RGY_CSP_BIT_DEPTH[m_inputCsp]) {
                m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_inputCsp];
            }
            if (nullptr == (m_Demux.video.frame = av_frame_alloc())) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate frame for decoder.\n"));
                return RGY_ERR_NULL_PTR;
            }
            m_Demux.video.qpTableListRef = input_prm->qpTableListRef;
        } else {
            //HWデコードの場合は、色変換がかからないので、入力フォーマットがそのまま出力フォーマットとなる
            m_inputVideoInfo.csp = pixfmtData->output_csp;
            m_inputVideoInfo.bitdepth = pixfmtData->bit_depth;
        }

        m_Demux.format.AVSyncMode = input_prm->AVSyncMode;

        if (input_prm->parseHDRmetadata) {
            auto err = parseHDRData();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to parse HDR metadata header from input.\n"));
                return err;
            }
        }

        //情報を格納
        m_inputVideoInfo.srcWidth    = m_Demux.video.stream->codecpar->width;
        m_inputVideoInfo.srcHeight   = m_Demux.video.stream->codecpar->height;
        m_inputVideoInfo.sar[0]      = (bAspectRatioUnknown) ? 0 : m_Demux.video.stream->codecpar->sample_aspect_ratio.num;
        m_inputVideoInfo.sar[1]      = (bAspectRatioUnknown) ? 0 : m_Demux.video.stream->codecpar->sample_aspect_ratio.den;
        m_inputVideoInfo.picstruct   = (input_prm->interlaceAutoFrame) ? RGY_PICSTRUCT_AUTO : m_Demux.frames.getVideoPicStruct();
        m_inputVideoInfo.frames      = 0;
        //getFirstFramePosAndFrameRateをもとにfpsを決定
        m_inputVideoInfo.fpsN        = m_Demux.video.nAvgFramerate.num;
        m_inputVideoInfo.fpsD        = m_Demux.video.nAvgFramerate.den;
        m_inputVideoInfo.vui.chromaloc = (CspChromaloc)m_Demux.video.stream->codecpar->chroma_location;
        m_inputVideoInfo.vui.matrix    = (CspMatrix)m_Demux.video.stream->codecpar->color_space;
        m_inputVideoInfo.vui.colorprim = (CspColorprim)m_Demux.video.stream->codecpar->color_primaries;
        m_inputVideoInfo.vui.transfer  = (CspTransfer)m_Demux.video.stream->codecpar->color_trc;
        m_inputVideoInfo.vui.colorrange = (CspColorRange)m_Demux.video.stream->codecpar->color_range;
        m_inputVideoInfo.vui.descriptpresent = 1;

        if (m_Demux.video.HWDecodeDeviceId >= 0) {
            tstring mes = strsprintf(_T("av" DECODER_NAME ": %s, %dx%d, %d/%d fps"),
                CodecToStr(m_inputVideoInfo.codec).c_str(),
                m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
            if (input_prm->seekSec > 0.0f) {
                mes += strsprintf(_T("\n         seek: %s"), print_time(input_prm->seekSec).c_str());
            }
            AddMessage(RGY_LOG_DEBUG, mes);
            m_inputInfo += mes;
        } else {
            CreateInputInfo((tstring(_T("avsw: ")) + char_to_tstring(avcodec_get_name(m_Demux.video.stream->codecpar->codec_id))).c_str(),
                RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
            if (input_prm->seekSec > 0.0f) {
                m_inputInfo += strsprintf(_T("\n         seek: %s"), print_time(input_prm->seekSec).c_str());
            }
            AddMessage(RGY_LOG_DEBUG, m_inputInfo);
        }
        if (m_Demux.video.stream) {
            AddMessage(RGY_LOG_DEBUG, _T("streamFirstKeyPts: %lld\n"), (long long int)m_Demux.video.streamFirstKeyPts);
            AddMessage(RGY_LOG_DEBUG, m_inputVideoInfo.vui.print_all());
            AddMessage(RGY_LOG_DEBUG, _T("sar %d:%d, bitdepth %d\n"),
                m_inputVideoInfo.sar[0], m_inputVideoInfo.sar[1], m_inputVideoInfo.bitdepth);
        }

        *inputInfo = m_inputVideoInfo;

        //スレッド関連初期化
        m_Demux.thread.bAbortInput = false;
#if 1
        auto nPrmInputThread = input_prm->threadInput;
        m_Demux.thread.threadInput = ((nPrmInputThread == RGY_INPUT_THREAD_AUTO) | (m_Demux.video.stream != nullptr)) ? 0 : nPrmInputThread;
#else
        //NVEncではいまのところ、常に無効
        m_Demux.thread.threadInput = 0;
#endif
        if (m_Demux.thread.threadInput) {
            m_Demux.thread.thInput = std::thread(&RGYInputAvcodec::ThreadFuncRead, this);
            //はじめcapacityを無限大にセットしたので、この段階で制限をかける
            //入力をスレッド化しない場合には、自動的に同期が保たれるので、ここでの制限は必要ない
            m_Demux.qVideoPkt.set_capacity(256);
        }
    } else {
        //音声との同期とかに使うので、動画の情報を格納する
        m_Demux.video.nAvgFramerate = av_make_q(input_prm->videoAvgFramerate);

        if (input_prm->nTrimCount) {
            m_trimParam.list = vector<sTrim>(input_prm->pTrimList, input_prm->pTrimList + input_prm->nTrimCount);
        }

        if (m_Demux.video.stream) {
            //動画の最初のフレームを取得しておく
            AVPacket pkt;
            av_init_packet(&pkt);
            //音声のみ処理モードでは、動画の先頭をキーフレームとする必要はなく、
            //先頭がキーフレームでなくてもframePosListに追加するようにして、trimをoffsetなしで反映できるようにする
            //そこで、bTreatFirstPacketAsKeyframe=trueにして最初のパケットを処理する
            if (getSample(&pkt, true)) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to get first packet of the video!\n"));
                return RGY_ERR_UNKNOWN;
            }
            av_packet_unref(&pkt);

            m_Demux.frames.checkPtsStatus();
        }

        tstring mes;
        for (const auto& stream : m_Demux.stream) {
            if (mes.length()) mes += _T(", ");
            tstring codec_name = char_to_tstring(avcodec_get_name(stream.stream->codecpar->codec_id));
            mes += codec_name;
            AddMessage(RGY_LOG_DEBUG, _T("avcodec %s: %s from %s\n"),
                get_media_type_string(stream.stream->codecpar->codec_id).c_str(), codec_name.c_str(), strFileName);
        }
        m_inputInfo += _T("avcodec audio: ") + mes;
    }
    return RGY_ERR_NONE;
}
#pragma warning(pop)

vector<const AVStream *> RGYInputAvcodec::GetInputAttachmentStreams() {
    vector<const AVStream *> streams;
    if (m_Demux.format.formatCtx) {
        for (unsigned int ist = 0; ist < m_Demux.format.formatCtx->nb_streams; ist++) {
            const AVStream *stream = m_Demux.format.formatCtx->streams[ist];
            if (stream->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT) {
                streams.push_back(stream);
            }
        }
    }
    return streams;
}

vector<const AVChapter *> RGYInputAvcodec::GetChapterList() {
    return m_Demux.chapter;
}

int RGYInputAvcodec::GetSubtitleTrackCount() {
    return m_Demux.format.subtitleTracks;
}

int RGYInputAvcodec::GetDataTrackCount() {
    return m_Demux.format.dataTracks;
}

int RGYInputAvcodec::GetAudioTrackCount() {
    return m_Demux.format.audioTracks;
}

int64_t RGYInputAvcodec::GetVideoFirstKeyPts() {
    return m_Demux.video.streamFirstKeyPts;
}

FramePosList *RGYInputAvcodec::GetFramePosList() {
    return &m_Demux.frames;
}

int RGYInputAvcodec::getVideoFrameIdx(int64_t pts, AVRational timebase, int iStart) {
    const int framePosCount = m_Demux.frames.frameNum();
    const AVRational vid_pkt_timebase = (m_Demux.video.stream) ? m_Demux.video.stream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
    if (av_cmp_q(timebase, vid_pkt_timebase) == 0) {
        for (int i = (std::max)(0, iStart); i < framePosCount; i++) {
            if (pts == m_Demux.frames.list(i).pts) {
                return i;
            }
            //pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
            if (pts < m_Demux.frames.list(i).pts) {
                //0フレーム目なら、仮想的に -1 フレーム目を考えて、それよりも前かどうかを判定する
                //-2を返すことで、そのパケットは削除される
                if (i == 0 && pts < m_Demux.frames.list(i).pts - m_Demux.frames.list(i).duration) {
                    i--;
                }
                return i-1;
            }
        }
    } else {
        for (int i = (std::max)(0, iStart); i < framePosCount; i++) {
            //pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
            if (av_compare_ts(pts, timebase, m_Demux.frames.list(i).pts, vid_pkt_timebase) < 0) {
                //0フレーム目なら、仮想的に -1 フレーム目を考えて、それよりも前かどうかを判定する
                //-2を返すことで、そのパケットは削除される
                if (i == 0 && av_compare_ts(pts, timebase, m_Demux.frames.list(i).pts - m_Demux.frames.list(i).duration, vid_pkt_timebase) < 0) {
                    i--;
                }
                return i-1;
            }
        }
    }
    return framePosCount;
}

int64_t RGYInputAvcodec::convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *stream) {
    const AVRational vid_pkt_timebase = (m_Demux.video.stream) ? m_Demux.video.stream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
    return av_rescale_q(pts, vid_pkt_timebase, stream->timebase);
}

bool RGYInputAvcodec::checkStreamPacketToAdd(AVPacket *pkt, AVDemuxStream *stream) {
    if (pkt->pts != AV_NOPTS_VALUE) { //pkt->ptsがAV_NOPTS_VALUEの場合は、以前のフレームの継続とみなして更新しない
        stream->lastVidIndex = getVideoFrameIdx(pkt->pts, stream->timebase, stream->lastVidIndex);
    }

    //該当フレームが-1フレーム未満なら、その音声はこの動画には含まれない
    if (stream->lastVidIndex < -1) {
        return false;
    }

    //映像がないなら判定しない
    if (!m_Demux.video.stream) {
        return true;
    }

    const auto vidFramePos = &m_Demux.frames.list((std::max)(stream->lastVidIndex, 0));
    const int64_t vid1_fin = convertTimebaseVidToStream(vidFramePos->pts + ((stream->lastVidIndex >= 0) ? vidFramePos->duration : 0), stream);
    const int64_t vid2_start = convertTimebaseVidToStream(m_Demux.frames.list((std::max)(stream->lastVidIndex+1, 0)).pts, stream);

    int64_t aud1_start = pkt->pts;
    int64_t aud1_fin   = pkt->pts + pkt->duration;

    //block index (空白がtrimで削除された領域)
    //       #0       #0         #1         #1       #2    #2
    //   |        |----------|         |----------|     |------
    const auto frame_is_in_range = frame_inside_range(stream->lastVidIndex,     m_trimParam.list);
    const auto next_is_in_range  = frame_inside_range(stream->lastVidIndex + 1, m_trimParam.list);
    const auto frame_trim_block_index = frame_is_in_range.second;
    const auto next_trim_block_index  = next_is_in_range.second;

    bool result = true; //動画に含まれる音声かどうか

    if (frame_is_in_range.first) {
        if (aud1_fin < vid1_fin || next_is_in_range.first) {
            ; //完全に動画フレームの範囲内か、次のフレームも範囲内なら、その音声パケットは含まれる
              //              vid1_fin
              //動画 <-----------|
              //音声      |-----------|
              //     aud1_start     aud1_fin
        } else if (pkt->duration / 2 > (aud1_fin - vid1_fin + stream->extractErrExcess)) {
            //はみ出した領域が少ないなら、その音声パケットは含まれる
            if (stream->stream->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
                //字幕の場合は表示時間を調整する
                pkt->duration -= vid1_fin - aud1_fin;
                aud1_fin       = vid1_fin;
            } else {
                stream->extractErrExcess += aud1_fin - vid1_fin;
            }
        } else {
            //はみ出した領域が多いなら、その音声パケットは含まれない
            stream->extractErrExcess -= vid1_fin - aud1_start;
            result = false;
        }
    } else if (next_is_in_range.first && aud1_fin > vid2_start) {
        //             vid2_start
        //動画             |------------>
        //音声      |-----------|
        //     aud1_start     aud1_fin
        if (pkt->duration / 2 > (vid2_start - aud1_start + stream->extractErrExcess)) {
            if (stream->stream->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
                //字幕の場合は表示時間を調整する
                pkt->pts      += vid2_start - aud1_start;
                pkt->duration -= vid2_start - aud1_start;
                aud1_start     = vid2_start;
            } else {
                stream->extractErrExcess += vid2_start - aud1_start;
            }
        } else {
            stream->extractErrExcess -= aud1_fin - vid2_start;
            result = false;
        }
    } else {
        result = false;
    }
    if (result) {
        if (stream->appliedTrimBlock < frame_trim_block_index) {
            stream->appliedTrimBlock = frame_trim_block_index;
            if (stream->aud0_fin == AV_NOPTS_VALUE) {
                //まだ一度も音声のパケットが渡されていない
                //基本的には動画の情報を基準に情報を修正する
                const int first_vid_frame = (m_trimParam.list.size() > 0) ? m_trimParam.list[0].start : 0;
                const int64_t vid0_start = convertTimebaseVidToStream(m_Demux.frames.list(first_vid_frame).pts, stream);
                stream->trimOffset += std::min(aud1_start, vid0_start) - m_Demux.video.streamFirstKeyPts;
            } else {
                assert(frame_trim_block_index > 0);
                const int last_valid_vid_frame = m_trimParam.list[frame_trim_block_index-1].start;
                assert(last_valid_vid_frame >= 0);
                const int64_t vid0_fin = convertTimebaseVidToStream(m_Demux.frames.list(last_valid_vid_frame).pts, stream);
                const int64_t vid1_start = convertTimebaseVidToStream(vidFramePos->pts, stream);
                const int64_t vid_start = (frame_is_in_range.first) ? vid1_start : vid2_start;
                if (vid_start - vid0_fin > aud1_start - stream->aud0_fin) {
                    stream->trimOffset += aud1_start - stream->aud0_fin;
                } else {
                    stream->trimOffset += vid_start - vid0_fin - stream->extractErrExcess;
                    stream->extractErrExcess = 0;
                }
            }
        }
        stream->aud0_fin = aud1_fin;
        //最終的に時刻を補正
        pkt->pts -= stream->trimOffset;
        pkt->dts -= stream->trimOffset;
    }
    return result;
}

AVDemuxStream *RGYInputAvcodec::getPacketStreamData(const AVPacket *pkt) {
    int streamIndex = pkt->stream_index;
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        if (m_Demux.stream[i].index == streamIndex) {
            return &m_Demux.stream[i];
        }
    }
    return nullptr;
}

int RGYInputAvcodec::getSample(AVPacket *pkt, bool bTreatFirstPacketAsKeyframe) {
    av_init_packet(pkt);
    int i_samples = 0;
    int ret_read_frame = 0;
    while ((ret_read_frame = av_read_frame(m_Demux.format.formatCtx, pkt)) >= 0
        //trimからわかるフレーム数の上限値よりfixedNumがある程度の量の処理を進めたら読み込みを打ち切る
        && m_Demux.frames.fixedNum() - TRIM_OVERREAD_FRAMES < getVideoTrimMaxFramIdx()) {
        if (pkt->stream_index == m_Demux.video.index) {
            if (pkt->flags & AV_PKT_FLAG_CORRUPT) {
                const auto timestamp = (pkt->pts == AV_NOPTS_VALUE) ? pkt->dts : pkt->pts;
                AddMessage(RGY_LOG_WARN, _T("corrupt packet in video: %lld (%s)\n"), (long long int)timestamp, getTimestampString(timestamp, m_Demux.video.stream->time_base).c_str());
            }
            if (m_Demux.video.bsfcCtx) {
                auto ret = av_bsf_send_packet(m_Demux.video.bsfcCtx, pkt);
                if (ret < 0) {
                    av_packet_unref(pkt);
                    AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"), char_to_tstring(m_Demux.video.bsfcCtx->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return 1;
                }
                ret = av_bsf_receive_packet(m_Demux.video.bsfcCtx, pkt);
                if (ret == AVERROR(EAGAIN)) {
                    continue; //もっとpacketを送らないとダメ
                } else if (ret < 0 && ret != AVERROR_EOF) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"), char_to_tstring(m_Demux.video.bsfcCtx->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return 1;
                }
            }
            if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_VC1) {
                vc1AddFrameHeader(pkt);
            }
            if (m_Demux.video.bUseHEVCmp42AnnexB) {
                hevcMp42Annexb(pkt);
            }
            if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_HEVC && m_Demux.video.hdr10plusMetadataCopy) {
                parseHDR10plus(pkt);
            }
            FramePos pos = { 0 };
            pos.pts = pkt->pts;
            pos.dts = pkt->dts;
            pos.duration = (int)pkt->duration;
            pos.duration2 = 0;
            pos.poc = FRAMEPOS_POC_INVALID;
            pos.flags = (uint8_t)pkt->flags;
            pos.pict_type = AV_PICTURE_TYPE_NONE;
            if (m_Demux.video.pParserCtx) {
                uint8_t* dummy = nullptr;
                int dummy_size = 0;
                av_parser_parse2(m_Demux.video.pParserCtx, m_Demux.video.pCodecCtxParser, &dummy, &dummy_size, pkt->data, pkt->size, pkt->pts, pkt->dts, pkt->pos);
                pos.pict_type = (uint8_t)(std::max)(m_Demux.video.pParserCtx->pict_type, 0);
                switch (m_Demux.video.pParserCtx->picture_structure) {
                    //フィールドとして符号化されている
                case AV_PICTURE_STRUCTURE_TOP_FIELD:    pos.pic_struct = RGY_PICSTRUCT_FIELD_TOP; break;
                case AV_PICTURE_STRUCTURE_BOTTOM_FIELD: pos.pic_struct = RGY_PICSTRUCT_FIELD_BOTTOM; break;
                    //フレームとして符号化されている
                default:
                    switch (m_Demux.video.pParserCtx->field_order) {
                    case AV_FIELD_TT:
                    case AV_FIELD_TB: pos.pic_struct = RGY_PICSTRUCT_FRAME_TFF; break;
                    case AV_FIELD_BT:
                    case AV_FIELD_BB: pos.pic_struct = RGY_PICSTRUCT_FRAME_BFF; break;
                    default:          pos.pic_struct = RGY_PICSTRUCT_FRAME;     break;
                    }
                }
                pos.repeat_pict = (uint8_t)m_Demux.video.pParserCtx->repeat_pict;
            }
            //mkv入りのVC-1をカットしたものなど、動画によってはpkt->flagsにフラグがセットされていないことがある
            //parserの情報も活用してキーフレームかどうかを判定する
            const bool keyframe = (pkt->flags & AV_PKT_FLAG_KEY) != 0 || pos.pict_type == AV_PICTURE_TYPE_I;
            //最初のキーフレームを取得するまではスキップする
            //スキップした枚数はi_samplesでカウントし、trim時に同期を適切にとるため、m_trimParam.offsetに格納する
            //  ただし、bTreatFirstPacketAsKeyframeが指定されている場合には、キーフレームでなくてもframePosListへの追加を許可する
            //  このモードは、対象の入力ファイルから--audio-sourceなどで音声のみ拾ってくる場合に使用する
            if (!bTreatFirstPacketAsKeyframe && !m_Demux.video.gotFirstKeyframe && !keyframe) {
                av_packet_unref(pkt);
                i_samples++;
                continue;
            } else {
                if (!m_Demux.video.gotFirstKeyframe) {
                    if (pkt->flags & AV_PKT_FLAG_DISCARD) {
                        //timestampが正常に設定されておらず、移乗動作の原因となるので、
                        //AV_PKT_FLAG_DISCARDがついている最初のフレームは無視する
                        continue;
                    }
                    //ここに入った場合は、必ず最初のキーフレーム
                    m_Demux.video.streamFirstKeyPts = (pkt->pts == AV_NOPTS_VALUE) ? pkt->dts : pkt->pts;
                    if (m_Demux.video.streamFirstKeyPts == AV_NOPTS_VALUE) {
                        if (m_Demux.stream.size() > 0) {
                            AddMessage(RGY_LOG_WARN, _T("first key frame had timestamp AV_NOPTS_VALUE, this might lead to avsync error.\n"));
                        }
                        m_Demux.video.streamFirstKeyPts = 0;
                    }
                    m_Demux.video.gotFirstKeyframe = true;
                    //キーフレームに到達するまでQSVではフレームが出てこない
                    //そのため、getSampleでも最初のキーフレームを取得するまでパケットを出力しない
                    //だが、これが原因でtrimの値とずれを生じてしまう
                    //そこで、そのぶんのずれを記録しておき、Trim値などに補正をかける
                    m_trimParam.offset = i_samples;
                    AddMessage(RGY_LOG_DEBUG, _T("found first key frame: timestamp %lld (%s), offset %d\n"),
                        (long long int)m_Demux.video.streamFirstKeyPts, getTimestampString(m_Demux.video.streamFirstKeyPts, m_Demux.video.stream->time_base).c_str(),
                        m_trimParam.offset);
                }
#if ENCODER_NVENC
                //NVENCのhwデコーダでは、opengopなどでキーフレームのパケットよりあとにその前のフレームが来た場合、
                //フレーム位置がさらにずれるので補正する
                else if (!(pkt->flags & AV_PKT_FLAG_KEY)
                    && (pkt->pts != AV_NOPTS_VALUE)
                    && (m_Demux.video.streamFirstKeyPts != AV_NOPTS_VALUE)
                    && pkt->pts < m_Demux.video.streamFirstKeyPts
                    && m_Demux.video.HWDecodeDeviceId >= 0) { //trim調整の適用はavhwリーダーのみ
                    m_trimParam.offset++;
                }
#endif //#if ENCODER_NVENC
                m_Demux.frames.add(pos);
            }
            //ptsの確定したところまで、音声を出力する
            CheckAndMoveStreamPacketList();
            return 0;
        }
        const auto *stream = getPacketStreamData(pkt);
        if (stream != nullptr) {
            if (pkt->flags & AV_PKT_FLAG_CORRUPT) {
                const auto timestamp = (pkt->pts == AV_NOPTS_VALUE) ? pkt->dts : pkt->pts;
                AddMessage(RGY_LOG_WARN, _T("corrupt packet in stream %d: %lld (%s)\n"), pkt->stream_index, (long long int)timestamp, getTimestampString(timestamp, stream->stream->time_base).c_str());
            }
            //音声/字幕パケットはひとまずすべてバッファに格納する
            m_Demux.qStreamPktL1.push_back(*pkt);
        } else {
            av_packet_unref(pkt);
        }
    }
    //ファイルの終わりに到達
    if (ret_read_frame != AVERROR_EOF && ret_read_frame < 0) {
        AddMessage(RGY_LOG_ERROR, _T("error occured while reading file: %d frames, %s\n"), m_Demux.frames.frameNum(), qsv_av_err2str(ret_read_frame).c_str());
        return 1;
    }
    AddMessage(RGY_LOG_DEBUG, _T("%d frames, %s\n"), m_Demux.frames.frameNum(), qsv_av_err2str(ret_read_frame).c_str());
    pkt->data = nullptr;
    pkt->size = 0;
    //動画の終端を表す最後のptsを挿入する
    int64_t videoFinPts = 0;
    const int nFrameNum = m_Demux.frames.frameNum();
    if (m_Demux.video.streamPtsInvalid & RGY_PTS_ALL_INVALID) {
        videoFinPts = nFrameNum * m_Demux.frames.list(0).duration;
    } else if (nFrameNum) {
        const FramePos *lastFrame = &m_Demux.frames.list(nFrameNum - 1);
        videoFinPts = lastFrame->pts + lastFrame->duration;
    }
    //もし選択範囲が手動で決定されていないのなら、音声を最大限取得する
    if (m_trimParam.list.size() == 0 || m_trimParam.list.back().fin == TRIM_MAX) {
        for (uint32_t i = 0; i < m_Demux.qStreamPktL2.size(); i++) {
            videoFinPts = (std::max)(videoFinPts, m_Demux.qStreamPktL2[i].data.pts);
        }
        for (uint32_t i = 0; i < m_Demux.qStreamPktL1.size(); i++) {
            videoFinPts = (std::max)(videoFinPts, m_Demux.qStreamPktL1[i].pts);
        }
    }
    //最後のフレーム情報をセットし、m_Demux.framesの内部状態を終了状態に移行する
    m_Demux.frames.fin(framePos(videoFinPts, videoFinPts, 0), m_Demux.format.formatCtx->duration);
    //映像キューのサイズ維持制限を解除する → パイプラインに最後まで読み取らせる
    m_Demux.qVideoPkt.set_keep_length(0);
    //音声をすべて出力する
    //m_Demux.frames.finをしたので、ここで実行すれば、qAudioPktL1のデータがすべてqAudioPktL2に移される
    CheckAndMoveStreamPacketList();
    //音声のみ読み込みの場合はm_encSatusInfoはnullptrなので、nullチェックを行う
#if !FOR_AUO //auoでここからUpdateDisplay()してしまうと、メインスレッド以外からのGUI更新となり、例外で落ちる
    if (m_encSatusInfo) {
        m_encSatusInfo->UpdateDisplay(100.0);
    }
#endif
    return AVERROR_EOF;
}

//動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは消す)
RGY_ERR RGYInputAvcodec::GetNextBitstream(RGYBitstream *pBitstream) {
    AVPacket pkt;
    if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
        && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
        const int ret = getSample(&pkt);
        if (ret == 0) {
            m_Demux.qVideoPkt.push(pkt);
        } else if (ret != AVERROR_EOF) {
            return RGY_ERR_UNKNOWN;
        }
    }

    bool bGetPacket = false;
    for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_and_pop_no_lock(&pkt, (m_Demux.thread.queueInfo) ? &m_Demux.thread.queueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
        m_Demux.qVideoPkt.wait_for_push();
    }
    RGY_ERR sts = RGY_ERR_MORE_BITSTREAM;
    if (bGetPacket) {
        if (pkt.data) {
            auto pts = (0 == (m_Demux.frames.getStreamPtsStatus() & (~RGY_PTS_NORMAL))) ? pkt.pts : AV_NOPTS_VALUE;
            sts = pBitstream->copy(pkt.data, pkt.size, pkt.dts, pts);
        }
        if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_HEVC && m_Demux.video.hdr10plusMetadataCopy) {
            pBitstream->addFrameData(getHDR10plusMetaData(&pkt));
        }
        av_packet_unref(&pkt);
        m_Demux.video.nSampleGetCount++;
        m_encSatusInfo->m_sData.frameIn++;
    }
    return sts;
}

//動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは残す)
RGY_ERR RGYInputAvcodec::GetNextBitstreamNoDelete(RGYBitstream *pBitstream) {
    AVPacket pkt;
    if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
        && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
        const int ret = getSample(&pkt);
        if (ret == 0) {
            m_Demux.qVideoPkt.push(pkt);
        } else if (ret != AVERROR_EOF) {
            return RGY_ERR_UNKNOWN;
        }
    }

    bool bGetPacket = false;
    for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_no_lock(&pkt, (m_Demux.thread.queueInfo) ? &m_Demux.thread.queueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
        m_Demux.qVideoPkt.wait_for_push();
    }
    RGY_ERR sts = RGY_ERR_MORE_BITSTREAM;
    if (bGetPacket) {
        if (pkt.data) {
            auto pts = (0 == (m_Demux.frames.getStreamPtsStatus() & (~RGY_PTS_NORMAL))) ? pkt.pts : AV_NOPTS_VALUE;
            sts = pBitstream->copy(pkt.data, pkt.size, pkt.dts, pts);
        }
    }
    return sts;
}

void RGYInputAvcodec::GetAudioDataPacketsWhenNoVideoRead(int inputFrame) {

    if (m_Demux.video.nSampleGetCount >= inputFrame) {
        return;
    }
    m_Demux.video.nSampleGetCount = inputFrame;
    const double vidEstDurationSec = inputFrame * (double)m_Demux.video.nAvgFramerate.den / (double)m_Demux.video.nAvgFramerate.num; //1フレームの時間(秒)

    AVPacket pkt;
    av_init_packet(&pkt);
    if (m_Demux.video.stream) {
        //動画に映像がある場合、getSampleを呼んで1フレーム分の音声データをm_Demux.qStreamPktL1に取得する
        //同時に映像フレームをロードし、ロードしたptsデータを突っ込む
        if (!getSample(&pkt)) {
            //動画データ自体は不要なので解放
            av_packet_unref(&pkt);
            CheckAndMoveStreamPacketList();
        }
        return;
    }

    auto move_pkt = [this](double vidEstDurationSec) {
        while (!m_Demux.qStreamPktL1.empty()) {
            auto pkt2 = m_Demux.qStreamPktL1.front();
            AVDemuxStream *pStream2 = getPacketStreamData(&pkt2);
            const double pkt2timeSec = pkt2.pts * (double)pStream2->stream->time_base.num / (double)pStream2->stream->time_base.den;
            if (pkt2timeSec > vidEstDurationSec + 5.0) {
                break;
            }
            pkt2.flags = (pkt2.flags & 0xffff) | ((uint32_t)pStream2->trackId << 16); //flagsの上位16bitには、trackIdへのポインタを格納しておく
            m_Demux.qStreamPktL2.push(pkt2); //Writer側に渡したパケットはWriter側で開放する
            m_Demux.qStreamPktL1.pop_front();
        }
    };

    //動画に映像がない場合、
    //およそ1フレーム分のパケットを取得する
    while (av_read_frame(m_Demux.format.formatCtx, &pkt) >= 0) {
        const auto codec_type = m_Demux.format.formatCtx->streams[pkt.stream_index]->codecpar->codec_type;
        if (codec_type != AVMEDIA_TYPE_AUDIO && codec_type != AVMEDIA_TYPE_SUBTITLE) {
            av_packet_unref(&pkt);
        } else {
            AVDemuxStream *pStream = getPacketStreamData(&pkt);
            const auto delay_ts = av_rescale_q(pStream->addDelayMs, av_make_q(1, 1000), pStream->timebase);
            pkt.pts += delay_ts;
            if (checkStreamPacketToAdd(&pkt, pStream)) {
                m_Demux.qStreamPktL1.push_back(pkt);
            } else {
                av_packet_unref(&pkt); //Writer側に渡さないパケットはここで開放する
            }

            //最初のパケットは参照用にコピーしておく
            if (pStream->pktSample.data == nullptr) {
                av_packet_ref(&pStream->pktSample, &pkt);
            }
            auto pktt = (pkt.pts == AV_NOPTS_VALUE) ? pkt.dts : pkt.pts;
            auto pkt_dist = pktt - pStream->pktSample.pts;
            //1フレーム分のサンプルを取得したら終了
            if (pkt_dist * (double)pStream->stream->time_base.num / (double)pStream->stream->time_base.den > vidEstDurationSec + 2.5) {
                //およそ1フレーム分のパケットを設定する
                int64_t pts = inputFrame;
                m_Demux.frames.add(framePos(pts, pts, 1, 0, inputFrame, AV_PKT_FLAG_KEY));
                if (m_Demux.frames.getStreamPtsStatus() == RGY_PTS_UNKNOWN) {
                    m_Demux.frames.checkPtsStatus();
                }
                move_pkt(vidEstDurationSec);
                return;
            }
        }
    }
    move_pkt(vidEstDurationSec);
    if (!m_Demux.frames.isEof()) {
        //読み込みが終了
        int64_t pts = inputFrame;
        m_Demux.frames.fin(framePos(pts, pts, 1, 0, inputFrame, AV_PKT_FLAG_KEY), inputFrame);
    }
}

const AVDictionary *RGYInputAvcodec::GetInputFormatMetadata() {
    return m_Demux.format.formatCtx->metadata;
}

const AVStream *RGYInputAvcodec::GetInputVideoStream() {
    return m_Demux.video.stream;
}

double RGYInputAvcodec::GetInputVideoDuration() {
    return (m_Demux.format.formatCtx->duration * (1.0 / (double)AV_TIME_BASE));
}

rgy_rational<int> RGYInputAvcodec::getInputTimebase() {
    return to_rgy(GetInputVideoStream()->time_base);
}

//qStreamPktL1をチェックし、framePosListから必要な音声パケットかどうかを判定し、
//必要ならqStreamPktL2に移し、不要ならパケットを開放する
void RGYInputAvcodec::CheckAndMoveStreamPacketList() {
    if (m_Demux.frames.fixedNum() == 0) {
        return;
    }
    //出力するパケットを選択する
    const AVRational vid_pkt_timebase = (m_Demux.video.stream) ? m_Demux.video.stream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
    while (!m_Demux.qStreamPktL1.empty()) {
        auto pkt = m_Demux.qStreamPktL1.front();
        AVDemuxStream *pStream = getPacketStreamData(&pkt);
        const auto delay_ts = av_rescale_q(pStream->addDelayMs, av_make_q(1, 1000), pStream->timebase);
        //音声のptsが映像の終わりのptsを行きすぎたらやめる
        const auto fixedLastFrame = m_Demux.frames.list(m_Demux.frames.fixedNum() - 1);
        if (!m_Demux.frames.isEof() // 最後まで読み込んでいたらすべて転送するようにする
            && 0 < av_compare_ts(pkt.pts + delay_ts, pStream->timebase, fixedLastFrame.pts, vid_pkt_timebase)) {
            break;
        }
        pkt.pts += delay_ts;
        if (checkStreamPacketToAdd(&pkt, pStream)) {
            pkt.flags = (pkt.flags & 0xffff) | ((uint32_t)pStream->trackId << 16); //flagsの上位16bitには、trackIdへのポインタを格納しておく
            m_Demux.qStreamPktL2.push(pkt); //Writer側に渡したパケットはWriter側で開放する
        } else {
            av_packet_unref(&pkt); //Writer側に渡さないパケットはここで開放する
        }
        m_Demux.qStreamPktL1.pop_front();
    }
}

vector<AVPacket> RGYInputAvcodec::GetStreamDataPackets(int inputFrame) {
    if (!m_Demux.video.readVideo) {
        GetAudioDataPacketsWhenNoVideoRead(inputFrame);
    }

    //出力するパケットを選択する
    vector<AVPacket> packets;
    AVPacket pkt;
    while (m_Demux.qStreamPktL2.front_copy_and_pop_no_lock(&pkt, (m_Demux.thread.queueInfo) ? &m_Demux.thread.queueInfo->usage_aud_in : nullptr)) {
        packets.push_back(pkt);
    }
    return packets;
}

vector<AVDemuxStream> RGYInputAvcodec::GetInputStreamInfo() {
    return vector<AVDemuxStream>(m_Demux.stream.begin(), m_Demux.stream.end());
}

RGY_ERR RGYInputAvcodec::GetHeader(RGYBitstream *pBitstream) {
    if (pBitstream == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    if (pBitstream->bufptr() == nullptr) {
        auto sts = pBitstream->init(AVCODEC_READER_INPUT_BUF_SIZE);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    if (m_Demux.video.extradata == nullptr) {
        if (m_Demux.video.stream->codecpar->extradata == nullptr || m_Demux.video.stream->codecpar->extradata_size == 0) {
            pBitstream->clear();
            return RGY_ERR_NONE;
        }
        m_Demux.video.extradataSize = m_Demux.video.stream->codecpar->extradata_size;
        //ここでav_mallocを使用しないと正常に動作しない
        m_Demux.video.extradata = (uint8_t *)av_malloc(m_Demux.video.stream->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        //ヘッダのデータをコピーしておく
        memcpy(m_Demux.video.extradata, m_Demux.video.stream->codecpar->extradata, m_Demux.video.extradataSize);
        memset(m_Demux.video.extradata + m_Demux.video.extradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        if (m_Demux.video.bUseHEVCmp42AnnexB) {
            hevcMp42Annexb(nullptr);
        } else if (m_Demux.video.bsfcCtx && m_Demux.video.extradata[0] == 1) {
            if (m_Demux.video.extradataSize < m_Demux.video.bsfcCtx->par_out->extradata_size) {
                m_Demux.video.extradata = (uint8_t *)av_realloc(m_Demux.video.extradata, m_Demux.video.bsfcCtx->par_out->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
            }
            memcpy(m_Demux.video.extradata, m_Demux.video.bsfcCtx->par_out->extradata, m_Demux.video.bsfcCtx->par_out->extradata_size);
            AddMessage(RGY_LOG_DEBUG, _T("GetHeader: changed %d bytes -> %d bytes by %s.\n"),
                m_Demux.video.extradataSize, m_Demux.video.bsfcCtx->par_out->extradata_size,
                char_to_tstring(m_Demux.video.bsfcCtx->filter->name).c_str());
            m_Demux.video.extradataSize = m_Demux.video.bsfcCtx->par_out->extradata_size;
            memset(m_Demux.video.extradata + m_Demux.video.extradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        } else if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_VC1) {
            //int lengthFix = (0 == strcmp(m_Demux.format.formatCtx->iformat->name, "mpegts")) ? 0 : -1;
            //vc1FixHeader(lengthFix);
        }
        AddMessage(RGY_LOG_DEBUG, _T("GetHeader: %d bytes.\n"), m_Demux.video.extradataSize);
        if (m_Demux.video.extradataSize == 0 && m_Demux.video.extradata != nullptr) {
            av_free(m_Demux.video.extradata);
            m_Demux.video.extradata = nullptr;
            AddMessage(RGY_LOG_DEBUG, _T("Failed to get header: 0 byte."));
            return RGY_ERR_MORE_DATA;
        }

        //抽出されたextradataが大きすぎる場合、適当に縮める
        //NVEncのデコーダが受け取れるヘッダは1024byteまで
        if (m_Demux.video.extradataSize > 1024) {
            if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_H264) {
                std::vector<nal_info> nal_list = parse_nal_unit_h264(m_Demux.video.extradata, m_Demux.video.extradataSize);
                const auto h264_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_SPS; });
                const auto h264_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_PPS; });
                const bool header_check = (nal_list.end() != h264_sps_nal) && (nal_list.end() != h264_pps_nal);
                if (header_check) {
                    m_Demux.video.extradataSize = (int)(h264_sps_nal->size + h264_pps_nal->size);
                    uint8_t *new_ptr = (uint8_t *)av_malloc(m_Demux.video.extradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
                    memcpy(new_ptr, h264_sps_nal->ptr, h264_sps_nal->size);
                    memcpy(new_ptr + h264_sps_nal->size, h264_pps_nal->ptr, h264_pps_nal->size);
                    if (m_Demux.video.extradata) {
                        av_free(m_Demux.video.extradata);
                    }
                    m_Demux.video.extradata = new_ptr;
                }
            } else if (m_Demux.video.stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
                std::vector<nal_info> nal_list = parse_nal_unit_hevc(m_Demux.video.extradata, m_Demux.video.extradataSize);
                const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
                const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
                const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
                const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
                if (header_check) {
                    m_Demux.video.extradataSize = (int)(hevc_vps_nal->size + hevc_sps_nal->size + hevc_pps_nal->size);
                    uint8_t *new_ptr = (uint8_t *)av_malloc(m_Demux.video.extradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
                    memcpy(new_ptr, hevc_vps_nal->ptr, hevc_vps_nal->size);
                    memcpy(new_ptr + hevc_vps_nal->size, hevc_sps_nal->ptr, hevc_sps_nal->size);
                    memcpy(new_ptr + hevc_vps_nal->size + hevc_sps_nal->size, hevc_pps_nal->ptr, hevc_pps_nal->size);
                    if (m_Demux.video.extradata) {
                        av_free(m_Demux.video.extradata);
                    }
                    m_Demux.video.extradata = new_ptr;
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("GetHeader: shrinked header to %d bytes.\n"), m_Demux.video.extradataSize);
        }
    }
    pBitstream->copy(m_Demux.video.extradata, m_Demux.video.extradataSize);
    return RGY_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
RGY_ERR RGYInputAvcodec::LoadNextFrame(RGYFrame *pSurface) {
    if (m_Demux.video.codecCtxDecode) {
        //動画のデコードを行う
        int got_frame = 0;
        while (!got_frame) {
            AVPacket pkt;
            av_init_packet(&pkt);
            if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
                && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
                const int ret = getSample(&pkt);
                if (ret == 0) {
                    m_Demux.qVideoPkt.push(pkt);
                } else if (ret != AVERROR_EOF) {
                    return RGY_ERR_UNKNOWN;
                }
            }

            bool bGetPacket = false;
            for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_no_lock(&pkt, (m_Demux.thread.queueInfo) ? &m_Demux.thread.queueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
                m_Demux.qVideoPkt.wait_for_push();
            }
            if (!bGetPacket) {
                //flushするためのパケット
                pkt.data = nullptr;
                pkt.size = 0;
            }
            int ret = avcodec_send_packet(m_Demux.video.codecCtxDecode, &pkt);
            //AVERROR(EAGAIN) -> パケットを送る前に受け取る必要がある
            //パケットが受け取られていないのでpopしない
            if (ret != AVERROR(EAGAIN)) {
                m_Demux.qVideoPkt.pop();
                av_packet_unref(&pkt);
            }
            if (ret == AVERROR_EOF) { //これ以上パケットを送れない
                AddMessage(RGY_LOG_DEBUG, _T("failed to send packet to video decoder, already flushed: %s.\n"), qsv_av_err2str(ret).c_str());
            } else if (ret < 0 && ret != AVERROR(EAGAIN)) {
                AddMessage(RGY_LOG_ERROR, _T("failed to send packet to video decoder: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            ret = avcodec_receive_frame(m_Demux.video.codecCtxDecode, m_Demux.video.frame);
            if (ret == AVERROR(EAGAIN)) { //もっとパケットを送る必要がある
                continue;
            }
            if (ret == AVERROR_EOF) {
                //最後まで読み込んだ
                return RGY_ERR_MORE_DATA;
            }
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to receive frame from video decoder: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            got_frame = TRUE;
        }
        pSurface->setTimestamp(m_Demux.video.frame->pts);
        pSurface->setDuration(m_Demux.video.frame->pkt_duration);
        if (pSurface->picstruct() == RGY_PICSTRUCT_AUTO) { //autoの時は、frameのインタレ情報をセットする
            pSurface->setPicstruct(picstruct_avframe_to_rgy(m_Demux.video.frame));
        }
#if ENCODER_NVENC
        pSurface->dataList().clear();
        if (m_Demux.video.qpTableListRef != nullptr) {
            int qp_stride = 0;
            int qscale_type = 0;
            #pragma warning(push)
            #pragma warning(disable:4996) // warning C4996: 'av_frame_get_qp_table': が古い形式として宣言されました。
            RGY_DISABLE_WARNING_PUSH
            RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
            const auto qp_table = av_frame_get_qp_table(m_Demux.video.frame, &qp_stride, &qscale_type);
            RGY_DISABLE_WARNING_POP
            #pragma warning(pop)
            if (qp_table != nullptr) {
                auto table = m_Demux.video.qpTableListRef->get();
                const int qpw = (qp_stride) ? qp_stride : (pSurface->width() + 15) / 16;
                const int qph = (qp_stride) ? (pSurface->height() + 15) / 16 : 1;
                table->setQPTable(qp_table, qpw, qph, qp_stride, qscale_type, m_Demux.video.frame->pict_type, m_Demux.video.frame->pts);
                pSurface->dataList().push_back(table);
            }
        }
#endif //#if ENCODER_NVENC
#if ENABLE_DHDR10_INFO
        {
            auto hdr10plus = std::shared_ptr<RGYFrameData>(getHDR10plusMetaData(m_Demux.video.frame));
            if (hdr10plus) {
                pSurface->dataList().push_back(hdr10plus);
            }
        }
#endif //#if ENABLE_DHDR10_INFO
        //フレームデータをコピー
        void *dst_array[3];
        pSurface->ptrArray(dst_array, m_convert->getFunc()->csp_to == RGY_CSP_RGB24 || m_convert->getFunc()->csp_to == RGY_CSP_RGB32);
        m_convert->run(m_Demux.video.frame->interlaced_frame != 0,
            dst_array, (const void **)m_Demux.video.frame->data,
            m_inputVideoInfo.srcWidth, m_Demux.video.frame->linesize[0], m_Demux.video.frame->linesize[1], pSurface->pitch(),
            m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);
        if (got_frame) {
            av_frame_unref(m_Demux.video.frame);
        }
        m_encSatusInfo->m_sData.frameIn++;
    } else {
        if (m_Demux.qVideoPkt.size() == 0) {
            //m_Demux.qVideoPkt.size() == 0となるのは、最後まで読み込んだときか、中断した時しかありえない
            return RGY_ERR_MORE_DATA; //ファイルの終わりに到達
        }
    }
    //進捗表示
    double progressPercent = 0.0;
    if (m_Demux.format.formatCtx->duration) {
        progressPercent = m_Demux.frames.duration() * (m_Demux.video.stream->time_base.num / (double)m_Demux.video.stream->time_base.den);
    }
    return m_encSatusInfo->UpdateDisplayByCurrentDuration(progressPercent);
}
#pragma warning(pop)

int RGYInputAvcodec::GetHWDecDeviceID() {
    return m_Demux.video.HWDecodeDeviceId;
}

HANDLE RGYInputAvcodec::getThreadHandleInput() {
#if defined(WIN32) || defined(WIN64)
    return m_Demux.thread.thInput.native_handle();
#else
    return NULL;
#endif //#if defined(WIN32) || defined(WIN64)
}

//出力する動画の情報をセット
void RGYInputAvcodec::setOutputVideoInfo(int w, int h, int sar_x, int sar_y, bool mux) {
    if (m_cap2ass.enabled()) {
        if (mux) {
            m_cap2ass.setOutputResolution(w, h, sar_x, sar_y);
            m_cap2ass.printParam(RGY_LOG_DEBUG);
        } else {
            //muxしないなら処理する必要はない
            AddMessage(RGY_LOG_WARN, _T("caption2ass will be disabled as muxer is disabled.\n"));
            m_cap2ass.disable();
        }
    }
}

RGY_ERR RGYInputAvcodec::ThreadFuncRead() {
    while (!m_Demux.thread.bAbortInput) {
        AVPacket pkt;
        if (getSample(&pkt)) {
            break;
        }
        m_Demux.qVideoPkt.push(pkt);
    }
    return RGY_ERR_NONE;
}

const AVMasteringDisplayMetadata *RGYInputAvcodec::getMasteringDisplay() const {
    return m_Demux.video.masteringDisplay;
};
const AVContentLightMetadata *RGYInputAvcodec::getContentLight() const {
    return m_Demux.video.contentLight;
};

#if USE_CUSTOM_INPUT
int RGYInputAvcodec::readPacket(uint8_t *buf, int buf_size) {
    auto ret = (int)_fread_nolock(buf, 1, buf_size, m_Demux.format.fpInput);
    if (m_cap2ass.enabled()) {
        if (m_cap2ass.proc(buf, ret, m_Demux.qStreamPktL1) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to process ts caption.\n"));
            return AVERROR_INVALIDDATA;
        }
    }
    return (ret == 0) ? AVERROR_EOF : ret;
}
int RGYInputAvcodec::writePacket(uint8_t *buf, int buf_size) {
    return (int)fwrite(buf, 1, buf_size, m_Demux.format.fpInput);
}
int64_t RGYInputAvcodec::seek(int64_t offset, int whence) {
    if (whence == AVSEEK_SIZE) {
        //たまにwhence=65536(AVSEEK_SIZE)とかいう要求が来ることがある
        return (m_Demux.format.inputFilesize) ? m_Demux.format.inputFilesize : -1;
    }
    if (whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END) {
        return -1;
    }
    if (m_cap2ass.enabled() && whence == SEEK_SET && offset == 0) {
        m_cap2ass.reset();
    }
    return _fseeki64(m_Demux.format.fpInput, offset, whence);
}
#endif //USE_CUSTOM_INPUT

#endif //ENABLE_AVSW_READER

