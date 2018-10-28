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
#include "rgy_thread.h"
#include "rgy_input_avcodec.h"
#include "rgy_bitstream.h"
#include "rgy_avlog.h"

//#ifdef LIBVA_SUPPORT
//#include "qsv_hw_va.h"
//#include "qsv_allocator.h"
//#endif //#if LIBVA_SUPPORT


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

RGYInputAvcodec::RGYInputAvcodec() {
    memset(&m_Demux.format, 0, sizeof(m_Demux.format));
    memset(&m_Demux.video,  0, sizeof(m_Demux.video));
    m_strReaderName = _T("av" DECODER_NAME "/avsw");
}

RGYInputAvcodec::~RGYInputAvcodec() {
    Close();
}

void RGYInputAvcodec::CloseThread() {
    m_Demux.thread.bAbortInput = true;
    if (m_Demux.thread.thInput.joinable()) {
        m_Demux.qVideoPkt.set_capacity(SIZE_MAX);
        m_Demux.qVideoPkt.set_keep_length(0);
        m_Demux.thread.thInput.join();
        AddMessage(RGY_LOG_DEBUG, _T("Closed Input thread.\n"));
    }
    m_Demux.thread.bAbortInput = false;
}

void RGYInputAvcodec::CloseFormat(AVDemuxFormat *pFormat) {
    //close video file
    if (pFormat->pFormatCtx) {
        avformat_close_input(&pFormat->pFormatCtx);
        AddMessage(RGY_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
    if (m_Demux.format.pFormatOptions) {
        av_dict_free(&m_Demux.format.pFormatOptions);
    }
    memset(pFormat, 0, sizeof(pFormat[0]));
}

void RGYInputAvcodec::CloseVideo(AVDemuxVideo *pVideo) {
    //close parser
    if (pVideo->pParserCtx) {
        av_parser_close(pVideo->pParserCtx);
    }
    if (pVideo->pCodecCtxParser) {
        avcodec_free_context(&pVideo->pCodecCtxParser);
    }
    if (pVideo->pCodecCtxDecode) {
        avcodec_free_context(&pVideo->pCodecCtxDecode);
    }
    //close bitstreamfilter
    if (pVideo->pBsfcCtx) {
        av_bsf_free(&pVideo->pBsfcCtx);
    }
    if (pVideo->pFrame) {
        av_frame_free(&pVideo->pFrame);
    }

    if (pVideo->pExtradata) {
        av_free(pVideo->pExtradata);
    }

    memset(pVideo, 0, sizeof(pVideo[0]));
    pVideo->nIndex = -1;
}

void RGYInputAvcodec::CloseStream(AVDemuxStream *pStream) {
    if (pStream->pktSample.data) {
        av_packet_unref(&pStream->pktSample);
    }
    if (pStream->subtitleHeader) {
        av_free(pStream->subtitleHeader);
    }
    memset(pStream, 0, sizeof(pStream[0]));
    pStream->nIndex = -1;
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
    AddMessage(RGY_LOG_DEBUG, _T("Cleared Stream Packet Buffer.\n"));

    m_cap2ass.close();
    AddMessage(RGY_LOG_DEBUG, _T("Closed caption handler.\n"));

    CloseFormat(&m_Demux.format);
    CloseVideo(&m_Demux.video);   AddMessage(RGY_LOG_DEBUG, _T("Closed video.\n"));
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        CloseStream(&m_Demux.stream[i]);
        AddMessage(RGY_LOG_DEBUG, _T("Cleared Stream #%d.\n"), i);
    }
    m_Demux.stream.clear();
    m_Demux.chapter.clear();

    m_sTrimParam.list.clear();
    m_sTrimParam.offset = 0;

    m_hevcMp42AnnexbBuffer.clear();

    //free input buffer (使用していない)
    //if (buffer) {
    //    free(buffer);
    //    buffer = nullptr;
    //}
    m_pEncSatusInfo.reset();
    if (m_sFramePosListLog.length()) {
        m_Demux.frames.printList(m_sFramePosListLog.c_str());
    }
    m_Demux.frames.clear();

    memset(&m_inputVideoInfo, 0, sizeof(m_inputVideoInfo));
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

void RGYInputAvcodec::SetExtraData(AVCodecParameters *pCodecParam, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (pCodecParam->extradata)
        av_free(pCodecParam->extradata);
    pCodecParam->extradata_size = size;
    pCodecParam->extradata      = (uint8_t *)av_malloc(pCodecParam->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(pCodecParam->extradata, data, size);
    memset(pCodecParam->extradata + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
};

RGY_CODEC RGYInputAvcodec::checkHWDecoderAvailable(AVCodecID id, AVPixelFormat pixfmt, const CodecCsp *pHWDecCodecCsp) {
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        if (HW_DECODE_LIST[i].avcodec_id == id) {
            auto rgy_codec = HW_DECODE_LIST[i].rgy_codec;
            if (pHWDecCodecCsp->count(rgy_codec) > 0) {
                const auto rgy_csp = csp_avpixfmt_to_rgy(pixfmt);
                auto& csp_list = pHWDecCodecCsp->at(rgy_codec);
                if (std::find(csp_list.begin(), csp_list.end(), rgy_csp) != csp_list.end()) {
                    return rgy_codec;
                }
            }
            return RGY_CODEC_UNKNOWN;
        }
    }
    return RGY_CODEC_UNKNOWN;
}

vector<int> RGYInputAvcodec::getStreamIndex(AVMediaType type, const vector<int> *pVidStreamIndex) {
    vector<int> streams;
    const int n_streams = m_Demux.format.pFormatCtx->nb_streams;
    for (int i = 0; i < n_streams; i++) {
        if (m_Demux.format.pFormatCtx->streams[i]->codecpar->codec_type == type) {
            streams.push_back(i);
        }
    }
    if (type == AVMEDIA_TYPE_VIDEO) {
        std::sort(streams.begin(), streams.end(), [pFormatCtx = m_Demux.format.pFormatCtx](int streamIdA, int streamIdB) {
            auto pStreamA = pFormatCtx->streams[streamIdA];
            auto pStreamB = pFormatCtx->streams[streamIdB];
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
    } else if (pVidStreamIndex && pVidStreamIndex->size()) {
        auto mostNearestVidStreamId = [pFormatCtx = m_Demux.format.pFormatCtx, pVidStreamIndex](int streamId) {
            auto ret = std::make_pair(0, UINT32_MAX);
            for (uint32_t i = 0; i < pVidStreamIndex->size(); i++) {
                uint32_t diff = (uint32_t)(streamId - pFormatCtx->streams[(*pVidStreamIndex)[i]]->id);
                if (diff < ret.second) {
                    ret.second = diff;
                    ret.first = i;
                }
            }
            return ret;
        };
        std::sort(streams.begin(), streams.end(), [pFormatCtx = m_Demux.format.pFormatCtx, pVidStreamIndex, mostNearestVidStreamId](int streamIdA, int streamIdB) {
            if (pFormatCtx->streams[streamIdA]->codecpar == nullptr) {
                return false;
            }
            if (pFormatCtx->streams[streamIdB]->codecpar == nullptr) {
                return true;
            }
            auto pStreamIdA = pFormatCtx->streams[streamIdA]->id;
            auto pStreamIdB = pFormatCtx->streams[streamIdB]->id;
            auto nearestVidA = mostNearestVidStreamId(pStreamIdA);
            auto nearestVidB = mostNearestVidStreamId(pStreamIdB);
            if (nearestVidA.first == nearestVidB.first) {
                return nearestVidA.second < nearestVidB.second;
            }
            return nearestVidA.first < nearestVidB.first;
        });
    }
    return std::move(streams);
}

bool RGYInputAvcodec::vc1StartCodeExists(uint8_t *ptr) {
    uint32_t code = readUB32(ptr);
    return check_range_unsigned(code, 0x010A, 0x010F) || check_range_unsigned(code, 0x011B, 0x011F);
}

void RGYInputAvcodec::vc1FixHeader(int nLengthFix) {
    if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_WMV3) {
        m_Demux.video.nExtradataSize += nLengthFix;
        uint32_t datasize = m_Demux.video.nExtradataSize;
        vector<uint8_t> buffer(20 + datasize, 0);
        uint32_t header = 0xC5000000;
        uint32_t width = m_Demux.video.pStream->codecpar->width;
        uint32_t height = m_Demux.video.pStream->codecpar->height;
        uint8_t *dataPtr = m_Demux.video.pExtradata - nLengthFix;
        memcpy(buffer.data() +  0, &header, sizeof(header));
        memcpy(buffer.data() +  4, &datasize, sizeof(datasize));
        memcpy(buffer.data() +  8, dataPtr, datasize);
        memcpy(buffer.data() +  8 + datasize, &height, sizeof(height));
        memcpy(buffer.data() + 12 + datasize, &width, sizeof(width));
        m_Demux.video.pExtradata = (uint8_t *)av_realloc(m_Demux.video.pExtradata, sizeof(buffer) + AV_INPUT_BUFFER_PADDING_SIZE);
        m_Demux.video.nExtradataSize = (int)buffer.size();
        memcpy(m_Demux.video.pExtradata, buffer.data(), buffer.size());
    } else {
        m_Demux.video.nExtradataSize += nLengthFix;
        memmove(m_Demux.video.pExtradata, m_Demux.video.pExtradata - nLengthFix, m_Demux.video.nExtradataSize);
    }
}

void RGYInputAvcodec::vc1AddFrameHeader(AVPacket *pkt) {
    uint32_t size = pkt->size;
    if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_WMV3) {
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

void RGYInputAvcodec::hevcMp42Annexb(AVPacket *pkt) {
    static const uint8_t SC[] = { 0, 0, 0, 1 };
    const uint8_t *ptr, *ptr_fin;
    if (pkt == NULL) {
        m_hevcMp42AnnexbBuffer.reserve(m_Demux.video.nExtradataSize + 128);
        ptr = m_Demux.video.pExtradata;
        ptr_fin = ptr + m_Demux.video.nExtradataSize;
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
        if (m_Demux.video.pExtradata) {
            av_free(m_Demux.video.pExtradata);
        }
        m_Demux.video.pExtradata = (uint8_t *)av_malloc(m_hevcMp42AnnexbBuffer.size());
        m_Demux.video.nExtradataSize = (int)m_hevcMp42AnnexbBuffer.size();
        memcpy(m_Demux.video.pExtradata, m_hevcMp42AnnexbBuffer.data(), m_hevcMp42AnnexbBuffer.size());
    }
    m_hevcMp42AnnexbBuffer.clear();
}

RGY_ERR RGYInputAvcodec::getFirstFramePosAndFrameRate(const sTrim *pTrimList, int nTrimCount, bool bDetectpulldown) {
    AVRational fpsDecoder = m_Demux.video.pStream->avg_frame_rate;
    const bool fpsDecoderInvalid = (fpsDecoder.den == 0 || fpsDecoder.num == 0);
    //timebaseが60で割り切れない場合には、ptsが完全には割り切れない値である場合があり、より多くのフレーム数を解析する必要がある
    int maxCheckFrames = (m_Demux.format.nAnalyzeSec == 0) ? ((m_Demux.video.pStream->time_base.den >= 1000 && m_Demux.video.pStream->time_base.den % 60) ? 128 : 48) : 7200;
    int maxCheckSec = (m_Demux.format.nAnalyzeSec == 0) ? INT_MAX : m_Demux.format.nAnalyzeSec;
    AddMessage(RGY_LOG_DEBUG, _T("fps decoder invalid: %s\n"), fpsDecoderInvalid ? _T("true") : _T("false"));

    AVPacket pkt;
    av_init_packet(&pkt);

    const bool bCheckDuration = m_Demux.video.pStream->time_base.num * m_Demux.video.pStream->time_base.den > 0;
    const double timebase = (bCheckDuration) ? m_Demux.video.pStream->time_base.num / (double)m_Demux.video.pStream->time_base.den : 1.0;
    m_Demux.video.nStreamFirstKeyPts = 0;
    int i_samples = 0;
    std::vector<int> frameDurationList;
    vector<std::pair<int, int>> durationHistgram;
    bool bPulldown = bDetectpulldown;

    for (int i_retry = 0; ; i_retry++) {
        if (i_retry) {
            //フレームレート推定がうまくいかなそうだった場合、もう少しフレームを解析してみる
            maxCheckFrames <<= 1;
            if (maxCheckSec != INT_MAX) {
                maxCheckSec <<= 1;
            }
            //ヒストグラム生成などは最初からやり直すので、一度クリアする
            durationHistgram.clear();
            frameDurationList.clear();
        }
        for (; i_samples < maxCheckFrames && !getSample(&pkt); i_samples++) {
            m_Demux.qVideoPkt.push(pkt);
            if (bCheckDuration) {
                int64_t diff = 0;
                if (pkt.dts != AV_NOPTS_VALUE && m_Demux.frames.list(0).dts != AV_NOPTS_VALUE) {
                    diff = (int)(pkt.dts - m_Demux.frames.list(0).dts);
                } else if (pkt.pts != AV_NOPTS_VALUE && m_Demux.frames.list(0).pts != AV_NOPTS_VALUE) {
                    diff = (int)(pkt.pts - m_Demux.frames.list(0).pts);
                }
                const int duration = (int)((double)diff * timebase + 0.5);
                if (duration >= maxCheckSec) {
                    break;
                }
            }
        }
#if _DEBUG && 0
        for (int i = 0; i < m_Demux.frames.frameNum(); i++) {
            fprintf(stderr, "%3d: pts:%I64d, poc:%3d, duration:%5d, duration2:%5d, repeat:%d\n",
                i, m_Demux.frames.list(i).pts, m_Demux.frames.list(i).poc,
                m_Demux.frames.list(i).duration, m_Demux.frames.list(i).duration2,
                m_Demux.frames.list(i).repeat_pict);
        }
#endif
        //ここまで集めたデータでpts, pocを確定させる
        double dEstFrameDurationByFpsDecoder = 0.0;
        if (av_isvalid_q(fpsDecoder) && av_isvalid_q(m_Demux.video.pStream->time_base)) {
            dEstFrameDurationByFpsDecoder = av_q2d(av_inv_q(fpsDecoder)) * av_q2d(av_inv_q(m_Demux.video.pStream->time_base));
        }
        m_Demux.frames.checkPtsStatus(dEstFrameDurationByFpsDecoder);

        const int nFramesToCheck = m_Demux.frames.fixedNum();
        AddMessage(RGY_LOG_DEBUG, _T("read %d packets.\n"), m_Demux.frames.frameNum());
        AddMessage(RGY_LOG_DEBUG, _T("checking %d frame samples.\n"), nFramesToCheck);

        frameDurationList.reserve(nFramesToCheck);
        bPulldown = bDetectpulldown;
        int last_repeat_pict = -1;

        for (int i = 0; i < nFramesToCheck; i++) {
#if _DEBUG && 0
            fprintf(stderr, "%3d: pts:%I64d, poc:%3d, duration:%5d, duration2:%5d, repeat:%d\n",
                i, m_Demux.frames.list(i).pts, m_Demux.frames.list(i).poc,
                m_Demux.frames.list(i).duration, m_Demux.frames.list(i).duration2,
                m_Demux.frames.list(i).repeat_pict);
#endif
            if (m_Demux.frames.list(i).poc != FRAMEPOS_POC_INVALID) {
                int duration = m_Demux.frames.list(i).duration + m_Demux.frames.list(i).duration2;
                auto repeat_pict = m_Demux.frames.list(i).repeat_pict;
                //RFF用の補正
                if (repeat_pict > 1) {
                    duration = (int)(duration * 2 / (double)(repeat_pict + 1) + 0.5);
                }
                //puldownの検出
                if (repeat_pict == last_repeat_pict) {
                    bPulldown = false;
                }
                last_repeat_pict = repeat_pict;
                frameDurationList.push_back(duration);
            }
        }

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

        const auto codec_timebase = av_stream_get_codec_timebase(m_Demux.video.pStream);
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
        //再度解析を行う場合は、音声がL2キューに入らないよう、一度fixedNumを0に戻す
        m_Demux.frames.clearPtsStatus();
    }

    //durationが0でなく、最も頻繁に出てきたもの
    auto& mostPopularDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0];

    struct Rational64 {
        uint64_t num;
        uint64_t den;
    } estimatedAvgFps = { 0 }, nAvgFramerate64 = { 0 }, fpsDecoder64 = { (uint64_t)fpsDecoder.num, (uint64_t)fpsDecoder.den };
    if (mostPopularDuration.first == 0) {
        m_Demux.video.nStreamPtsInvalid |= RGY_PTS_ALL_INVALID;
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
        double avgFps = m_Demux.video.pStream->time_base.den / (double)(avgDuration * m_Demux.video.pStream->time_base.num);
        double torrelance = (fps_near(avgFps, 25.0) || fps_near(avgFps, 50.0)) ? 0.05 : 0.0008; //25fps, 50fps近辺は基準が甘くてよい
        if (mostPopularDuration.second / (double)frameDurationList.size() > 0.95 && std::abs(1 - mostPopularDuration.first / avgDuration) < torrelance) {
            avgDuration = mostPopularDuration.first;
            AddMessage(RGY_LOG_DEBUG, _T("using popular duration...\n"));
        }
        //durationから求めた平均fpsを計算する
        const uint64_t mul = (uint64_t)ceil(1001.0 / m_Demux.video.pStream->time_base.num);
        estimatedAvgFps.num = (uint64_t)(m_Demux.video.pStream->time_base.den / avgDuration * (double)m_Demux.video.pStream->time_base.num * mul + 0.5);
        estimatedAvgFps.den = (uint64_t)m_Demux.video.pStream->time_base.num * mul;

        AddMessage(RGY_LOG_DEBUG, _T("fps mul:         %d\n"),    mul);
        AddMessage(RGY_LOG_DEBUG, _T("raw avgDuration: %lf\n"),   avgDuration);
        AddMessage(RGY_LOG_DEBUG, _T("estimatedAvgFps: %I64u/%I64u\n"), estimatedAvgFps.num, estimatedAvgFps.den);
    }

    if (m_Demux.video.nStreamPtsInvalid & RGY_PTS_ALL_INVALID) {
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
    AddMessage(RGY_LOG_DEBUG, _T("final AvgFps (raw64): %I64u/%I64u\n"), estimatedAvgFps.num, estimatedAvgFps.den);

    //フレームレートが2000fpsを超えることは考えにくいので、誤判定
    //ほかのなにか使えそうな値で代用する
    const auto codec_timebase = av_stream_get_codec_timebase(m_Demux.video.pStream);
    if (nAvgFramerate64.num / (double)nAvgFramerate64.den > 2000.0) {
        if (fpsDecoder.den > 0 && fpsDecoder.num > 0) {
            nAvgFramerate64.num = fpsDecoder.num;
            nAvgFramerate64.den = fpsDecoder.den;
        } else if (codec_timebase.den > 0
                && codec_timebase.num > 0) {
            const AVCodec *pCodec = avcodec_find_decoder(m_Demux.video.pStream->codecpar->codec_id);
            AVCodecContext *pCodecCtx = avcodec_alloc_context3(pCodec);
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
            if (streamInfo->pStream && avcodec_get_type(streamInfo->pStream->codecpar->codec_id) == AVMEDIA_TYPE_AUDIO) {
                AddMessage(RGY_LOG_DEBUG, _T("checking for stream #%d\n"), streamInfo->nIndex);
                const AVPacket *pkt1 = nullptr; //最初のパケット
                const AVPacket *pkt2 = nullptr; //2番目のパケット
                //まず、L2キューを探す
                for (int j = 0; j < (int)m_Demux.qStreamPktL2.size(); j++) {
                    if (m_Demux.qStreamPktL2.get(j)->data.stream_index == streamInfo->nIndex) {
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
                        if (m_Demux.qStreamPktL1[j].stream_index == streamInfo->nIndex) {
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
                    av_copy_packet(&streamInfo->pktSample, (pkt2) ? pkt2 : pkt1);
                    if (m_Demux.video.nStreamPtsInvalid & RGY_PTS_ALL_INVALID) {
                        streamInfo->nDelayOfStream = 0;
                    } else {
                        //その音声の属する動画フレーム番号
                        const int vidIndex = getVideoFrameIdx(pkt1->pts, streamInfo->pStream->time_base, 0);
                        AddMessage(RGY_LOG_DEBUG, _T("audio track %d first pts: %I64d\n"), streamInfo->nTrackId, pkt1->pts);
                        AddMessage(RGY_LOG_DEBUG, _T("      first pts videoIdx: %d\n"), vidIndex);
                        if (vidIndex >= 0) {
                            //音声の遅れているフレーム数分のdurationを足し上げる
                            int delayOfStream = (frame_inside_range(vidIndex, trimList)) ? (int)(pkt1->pts - m_Demux.frames.list(vidIndex).pts) : 0;
                            for (int iFrame = m_sTrimParam.offset; iFrame < vidIndex; iFrame++) {
                                if (frame_inside_range(iFrame, trimList)) {
                                    delayOfStream += m_Demux.frames.list(iFrame).duration;
                                }
                            }
                            streamInfo->nDelayOfStream = delayOfStream;
                            AddMessage(RGY_LOG_DEBUG, _T("audio track %d delay: %d (timebase=%d/%d)\n"),
                                streamInfo->nIndex, streamInfo->nTrackId,
                                streamInfo->nDelayOfStream, streamInfo->pStream->time_base.num, streamInfo->pStream->time_base.den);
                        }
                    }
                } else {
                    //音声の最初のサンプルを取得できていない
                    AddMessage(RGY_LOG_WARN, _T("failed to find stream #%d in preread.\n"), streamInfo->nIndex);
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

#pragma warning(push)
#pragma warning(disable:4100)
RGY_ERR RGYInputAvcodec::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) {
    const AvcodecReaderPrm *input_prm = (const AvcodecReaderPrm *)prm;

    if (input_prm->bReadVideo) {
        if (pInputInfo->type != RGY_INPUT_FMT_AVANY) {
            m_strReaderName = (pInputInfo->type != RGY_INPUT_FMT_AVSW) ? _T("av" DECODER_NAME) : _T("avsw");
        }
    } else {
        m_strReaderName = _T("avsw");
    }

    m_Demux.video.bReadVideo = input_prm->bReadVideo;
    m_Demux.thread.pQueueInfo = input_prm->pQueueInfo;
    if (input_prm->bReadVideo) {
        memcpy(&m_inputVideoInfo, pInputInfo, sizeof(m_inputVideoInfo));
    } else {
        memset(&m_inputVideoInfo, 0, sizeof(m_inputVideoInfo));
    }

    if (!check_avcodec_dll()) {
        AddMessage(RGY_LOG_ERROR, error_mes_avcodec_dll_not_found());
        return RGY_ERR_NULL_PTR;
    }

    for (int i = 0; i < input_prm->nAudioSelectCount; i++) {
        tstring audioLog = strsprintf(_T("select audio track %s, codec %s"),
            (input_prm->ppAudioSelect[i]->nAudioSelect) ? strsprintf(_T("#%d"), input_prm->ppAudioSelect[i]->nAudioSelect).c_str() : _T("all"),
            input_prm->ppAudioSelect[i]->pAVAudioEncodeCodec);
        if (input_prm->ppAudioSelect[i]->pAudioExtractFormat) {
            audioLog += tstring(_T("format ")) + input_prm->ppAudioSelect[i]->pAudioExtractFormat;
        }
        if (input_prm->ppAudioSelect[i]->pAVAudioEncodeCodec != nullptr
            && 0 != _tcscmp(input_prm->ppAudioSelect[i]->pAVAudioEncodeCodec, RGY_AVCODEC_COPY)) {
            audioLog += strsprintf(_T("bitrate %d"), input_prm->ppAudioSelect[i]->nAVAudioEncodeBitrate);
        }
        if (input_prm->ppAudioSelect[i]->pAudioExtractFilename) {
            audioLog += tstring(_T("filename \"")) + input_prm->ppAudioSelect[i]->pAudioExtractFilename + tstring(_T("\""));
        }
        AddMessage(RGY_LOG_DEBUG, audioLog);
    }

    av_log_set_level((m_pPrintMes->getLogLevel() == RGY_LOG_DEBUG) ?  AV_LOG_DEBUG : RGY_AV_LOG_LEVEL);
    av_qsv_log_set(m_pPrintMes);
    if (input_prm->caption2ass != FORMAT_INVALID) {
        auto err = m_cap2ass.init(m_pPrintMes, input_prm->caption2ass);
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
    m_Demux.format.bIsPipe = (0 == strcmp(filename_char.c_str(), "-")) || filename_char.c_str() == strstr(filename_char.c_str(), R"(\\.\pipe\)");
    m_Demux.format.pFormatCtx = avformat_alloc_context();
    m_Demux.format.nAnalyzeSec = input_prm->nAnalyzeSec;
    if (m_Demux.format.nAnalyzeSec) {
        if (0 != (ret = av_opt_set_int(m_Demux.format.pFormatCtx, "probesize", 1 << 29, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set probesize to 0.5GB: error %d\n"), ret);
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("set probesize: 0.5GB\n"));
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
    av_dict_set(&m_Demux.format.pFormatOptions, "scan_all_pmts", "1", 0);
    //入力フォーマットが指定されていれば、それを渡す
    AVInputFormat *pInFormat = nullptr;
    if (input_prm->pInputFormat) {
        if (nullptr == (pInFormat = av_find_input_format(tchar_to_string(input_prm->pInputFormat).c_str()))) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown Input format: %s.\n"), input_prm->pInputFormat);
            return RGY_ERR_INVALID_FORMAT;
        }
    }

#if USE_CUSTOM_INPUT
    if (!m_Demux.format.bIsPipe && !usingAVProtocols(filename_char, 0) || !(pInFormat->flags & (AVFMT_NEEDNUMBER | AVFMT_NOFILE))) {
        m_Demux.format.fpInput = _tfopen(strFileName, _T("rb"));
        if (m_Demux.format.fpInput == NULL) {
            errno_t error = errno;
            AddMessage(RGY_LOG_ERROR, _T("failed to open input file \"%s\": %s.\n"), strFileName, _tcserror(error));
            return RGY_ERR_FILE_OPEN; // Couldn't open file
        }
        m_Demux.format.inputBufferSize = 512 * 1024;
        m_Demux.format.pInputBuffer = (char *)av_malloc(m_Demux.format.inputBufferSize);
        m_Demux.format.pFormatCtx->flags |= AVFMT_FLAG_CUSTOM_IO;
        if (NULL == (m_Demux.format.pFormatCtx->pb = avio_alloc_context((unsigned char *)m_Demux.format.pInputBuffer, m_Demux.format.inputBufferSize, 0, this, funcReadPacket, funcWritePacket, funcSeek))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return RGY_ERR_NULL_PTR;
        }
    }
#endif
    //ファイルのオープン
    if (avformat_open_input(&(m_Demux.format.pFormatCtx), filename_char.c_str(), pInFormat, &m_Demux.format.pFormatOptions)) {
        AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\"\n"), char_to_tstring(filename_char, CP_UTF8).c_str());
        return RGY_ERR_FILE_OPEN; // Couldn't open file
    }
    AddMessage(RGY_LOG_DEBUG, _T("opened file \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());

    if (m_Demux.format.nAnalyzeSec) {
        if (0 != (ret = av_opt_set_int(m_Demux.format.pFormatCtx, "analyzeduration", m_Demux.format.nAnalyzeSec * AV_TIME_BASE, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set analyzeduration to %d sec, error %d\n"), m_Demux.format.nAnalyzeSec, ret);
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("set analyzeduration: %d sec\n"), m_Demux.format.nAnalyzeSec);
        }
    }

    if (avformat_find_stream_info(m_Demux.format.pFormatCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("error finding stream information.\n"));
        return RGY_ERR_UNKNOWN; // Couldn't find stream information
    }
    AddMessage(RGY_LOG_DEBUG, _T("got stream information.\n"));
    av_dump_format(m_Demux.format.pFormatCtx, 0, filename_char.c_str(), 0);
    //dump_format(dec.pFormatCtx, 0, argv[1], 0);

    //キュー関連初期化
    //getFirstFramePosAndFrameRateで大量にパケットを突っ込む可能性があるので、この段階ではcapacityは無限大にしておく
    m_Demux.qVideoPkt.init(4096, SIZE_MAX, 4);
    m_Demux.qVideoPkt.set_keep_length(AV_FRAME_MAX_REORDER);
    m_Demux.qStreamPktL2.init(4096);

    //動画ストリームを探す
    //動画ストリームは動画を処理しなかったとしても同期のため必要
    auto videoStreams = getStreamIndex(AVMEDIA_TYPE_VIDEO);
    if (videoStreams.size()) {
        if (input_prm->nVideoTrack) {
            if (videoStreams.size() < (uint32_t)std::abs(input_prm->nVideoTrack)) {
                AddMessage(RGY_LOG_ERROR, _T("track %d was selected for video, but input only contains %d video tracks.\n"), input_prm->nVideoTrack, videoStreams.size());
                return RGY_ERR_INVALID_VIDEO_PARAM;
            } else if (input_prm->nVideoTrack < 0) {
                //逆順に並べ替え
                std::reverse(videoStreams.begin(), videoStreams.end());
            }
            m_Demux.video.nIndex = videoStreams[std::abs(input_prm->nVideoTrack)-1];
        } else if (input_prm->nVideoStreamId) {
            auto streamIndexFound = std::find_if(videoStreams.begin(), videoStreams.end(), [pFormatCtx = m_Demux.format.pFormatCtx, nSearchId = input_prm->nVideoStreamId](int nStreamIndex) {
                return (pFormatCtx->streams[nStreamIndex]->id == nSearchId);
            });
            if (streamIndexFound == videoStreams.end()) {
                AddMessage(RGY_LOG_ERROR, _T("stream id %d (0x%x) not found in video tracks.\n"), input_prm->nVideoStreamId, input_prm->nVideoStreamId);
                return RGY_ERR_INVALID_VIDEO_PARAM;
            }
            m_Demux.video.nIndex = *streamIndexFound;
        } else {
            m_Demux.video.nIndex = videoStreams[0];
        }
        auto selectedStream = std::find(videoStreams.begin(), videoStreams.end(), m_Demux.video.nIndex);
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
        AddMessage(RGY_LOG_DEBUG, _T("found video stream, stream idx %d\n"), m_Demux.video.nIndex);

        m_Demux.video.pStream = m_Demux.format.pFormatCtx->streams[m_Demux.video.nIndex];
    }

    //音声ストリームを探す
    if (input_prm->nReadAudio || input_prm->bReadSubtitle) {
        vector<int> mediaStreams;
        if (input_prm->nReadAudio) {
            auto audioStreams = getStreamIndex(AVMEDIA_TYPE_AUDIO, &videoStreams);
            //他のファイルから音声を読み込む場合もあるので、ここでチェックはできない
            //if (audioStreams.size() == 0) {
            //    AddMessage(RGY_LOG_ERROR, _T("--audio-encode/--audio-copy/--audio-file is set, but no audio stream found.\n"));
            //    return RGY_ERR_NOT_FOUND;
            //}
            m_Demux.format.nAudioTracks = (int)audioStreams.size();
            vector_cat(mediaStreams, audioStreams);
        }
        if (input_prm->bReadSubtitle) {
            auto subStreams = getStreamIndex(AVMEDIA_TYPE_SUBTITLE, &videoStreams);
            if (subStreams.size() == 0 && !m_cap2ass.enabled()) {
                AddMessage(RGY_LOG_WARN, _T("--sub-copy is set, but no subtitle stream found.\n"));
            } else {
                m_Demux.format.nSubtitleTracks = (int)subStreams.size();
                vector_cat(mediaStreams, subStreams);
            }
        }
        for (int iTrack = 0; iTrack < (int)mediaStreams.size(); iTrack++) {
            const AVCodecID codecId = m_Demux.format.pFormatCtx->streams[mediaStreams[iTrack]]->codecpar->codec_id;
            bool useStream = false;
            sAudioSelect *pAudioSelect = nullptr; //トラックに対応するsAudioSelect (字幕ストリームの場合はnullptrのまま)
            if (AVMEDIA_TYPE_SUBTITLE == avcodec_get_type(codecId)) {
                //字幕の場合
                for (int i = 0; !useStream && i < input_prm->nSubtitleSelectCount; i++) {
                    if (input_prm->pSubtitleSelect[i] == 0 //特に指定なし = 全指定かどうか
                        || input_prm->pSubtitleSelect[i] == (iTrack - m_Demux.format.nAudioTracks + 1 + input_prm->nSubtitleTrackStart)) {
                        useStream = true;
                    }
                }
            } else {
                //音声の場合
                for (int i = 0; !useStream && i < input_prm->nAudioSelectCount; i++) {
                    if (input_prm->ppAudioSelect[i]->nAudioSelect == (iTrack + input_prm->nAudioTrackStart)) {
                        useStream = true;
                        pAudioSelect = input_prm->ppAudioSelect[i];
                    }
                }
                if (pAudioSelect == nullptr) {
                    //見つからなかったら、全指定(trackID = 0)のものを使用する
                    for (int i = 0; !useStream && i < input_prm->nAudioSelectCount; i++) {
                        if (input_prm->ppAudioSelect[i]->nAudioSelect == 0) {
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
                    && isSplitChannelAuto(pAudioSelect->pnStreamChannelSelect)) {
                    const uint64_t channel_layout_mask = UINT64_MAX >> (sizeof(channel_layout_mask) * 8 - m_Demux.format.pFormatCtx->streams[mediaStreams[iTrack]]->codecpar->channels);
                    for (uint32_t iSubStream = 0; iSubStream < MAX_SPLIT_CHANNELS; iSubStream++) {
                        pAudioSelect->pnStreamChannelSelect[iSubStream] &= channel_layout_mask;
                    }
                    for (uint32_t iSubStream = 0; iSubStream < MAX_SPLIT_CHANNELS; iSubStream++) {
                        pAudioSelect->pnStreamChannelOut[iSubStream] &= channel_layout_mask;
                    }
                }

                //必要であれば、サブストリームを追加する
                for (uint32_t iSubStream = 0; iSubStream == 0 || //初回は字幕・音声含め、かならず登録する必要がある
                    (iSubStream < MAX_SPLIT_CHANNELS //最大サブストリームの上限
                        && pAudioSelect != nullptr //字幕ではない
                        && pAudioSelect->pnStreamChannelSelect[iSubStream]); //audio-splitが指定されている
                    iSubStream++) {
                    AVDemuxStream stream = { 0 };
                    stream.nTrackId = (AVMEDIA_TYPE_SUBTITLE == avcodec_get_type(codecId))
                        ? -(iTrack - m_Demux.format.nAudioTracks + 1 + input_prm->nSubtitleTrackStart) //字幕は -1, -2, -3
                        : iTrack + input_prm->nAudioTrackStart; //音声は1, 2, 3
                    stream.nIndex = mediaStreams[iTrack];
                    stream.nSubStreamId = iSubStream;
                    stream.pStream = m_Demux.format.pFormatCtx->streams[stream.nIndex];
                    stream.timebase = stream.pStream->time_base;
                    if (pAudioSelect) {
                        memcpy(stream.pnStreamChannelSelect, pAudioSelect->pnStreamChannelSelect, sizeof(stream.pnStreamChannelSelect));
                        memcpy(stream.pnStreamChannelOut,    pAudioSelect->pnStreamChannelOut,    sizeof(stream.pnStreamChannelOut));
                    }
                    m_Demux.stream.push_back(stream);
                    AddMessage(RGY_LOG_DEBUG, _T("found %s stream, stream idx %d, trackID %d.%d, %s, frame_size %d, timebase %d/%d\n"),
                        get_media_type_string(codecId).c_str(),
                        stream.nIndex, stream.nTrackId, stream.nSubStreamId, char_to_tstring(avcodec_get_name(codecId)).c_str(),
                        stream.pStream->codecpar->frame_size, stream.pStream->time_base.num, stream.pStream->time_base.den);
                }
            }
        }
        //指定されたすべての音声トラックが発見されたかを確認する
        for (int i = 0; i < input_prm->nAudioSelectCount; i++) {
            //全指定のトラック=0は無視
            if (input_prm->ppAudioSelect[i]->nAudioSelect > 0) {
                bool audioFound = false;
                for (const auto& stream : m_Demux.stream) {
                    if (stream.nTrackId == input_prm->ppAudioSelect[i]->nAudioSelect) {
                        audioFound = true;
                        break;
                    }
                }
                if (!audioFound) {
                    AddMessage(RGY_LOG_WARN, _T("could not find audio track #%d\n"), input_prm->ppAudioSelect[i]->nAudioSelect);
                }
            }
        }
    }

    if (m_cap2ass.enabled()) {
        //SubtitileのtrackIdは、-1, -2, -3
        const int minSubTrackId = std::accumulate(m_Demux.stream.begin(), m_Demux.stream.end(), 0, [](int a, const AVDemuxStream& b) {
            return std::min(a, b.nTrackId);
        });
        m_cap2ass.setIndex(m_Demux.format.pFormatCtx->nb_streams, minSubTrackId-1);
        m_Demux.stream.push_back(m_cap2ass.stream());
    }

    if (input_prm->bReadChapter) {
        m_Demux.chapter = make_vector((const AVChapter **)m_Demux.format.pFormatCtx->chapters, m_Demux.format.pFormatCtx->nb_chapters);
    }

    //動画処理の初期化を行う
    if (input_prm->bReadVideo) {
        if (m_Demux.video.pStream == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("unable to find video stream.\n"));
            return RGY_ERR_NOT_FOUND;
        }
        AddMessage(RGY_LOG_DEBUG, _T("use video stream #%d for input, codec %s, stream time_base %d/%d, codec_timebase %d/%d.\n"),
            m_Demux.video.pStream->index,
            char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str(),
            m_Demux.video.pStream->time_base.num, m_Demux.video.pStream->time_base.den,
            av_stream_get_codec_timebase(m_Demux.video.pStream).num, av_stream_get_codec_timebase(m_Demux.video.pStream).den);

        m_sFramePosListLog.clear();
        if (input_prm->pFramePosListLog) {
            m_sFramePosListLog = input_prm->pFramePosListLog;
        }

        m_Demux.video.nHWDecodeDeviceId = -1;
        if (m_inputVideoInfo.type != RGY_INPUT_FMT_AVSW) {
            for (const auto& devCodecCsp : *input_prm->pHWDecCodecCsp) {
                m_inputVideoInfo.codec = checkHWDecoderAvailable(
                    m_Demux.video.pStream->codecpar->codec_id, (AVPixelFormat)m_Demux.video.pStream->codecpar->format, &devCodecCsp.second);
                if (m_inputVideoInfo.codec != RGY_CODEC_UNKNOWN) {
                    m_Demux.video.nHWDecodeDeviceId = devCodecCsp.first;
                    break;
                }
                if (m_inputVideoInfo.type != RGY_INPUT_FMT_AVHW) {
                    break; //RGY_INPUT_FMT_AVANYのときは、HWデコードできなくても優先度の高いGPUを使用する
                }
            }
            if (RGY_CODEC_UNKNOWN == m_inputVideoInfo.codec
                //wmv3はAdvanced Profile (3)のみの対応
                || (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_WMV3 && m_Demux.video.pStream->codecpar->profile != 3)) {
                if (m_inputVideoInfo.type == RGY_INPUT_FMT_AVHW) {
                    //HWデコードが指定されている場合にはエラー終了する
                    AddMessage(RGY_LOG_ERROR, _T("codec %s(%s) unable to decode by " DECODER_NAME ".\n"),
                        char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str(),
                        char_to_tstring(av_get_pix_fmt_name((AVPixelFormat)m_Demux.video.pStream->codecpar->format)).c_str());
                    return RGY_ERR_INVALID_CODEC;
                }
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("can be decoded by %s.\n"), _T(DECODER_NAME));
            }
        }
        m_strReaderName = (m_Demux.video.nHWDecodeDeviceId >= 0) ? _T("av" DECODER_NAME) : _T("avsw");
        m_inputVideoInfo.type = (m_Demux.video.nHWDecodeDeviceId >= 0) ? RGY_INPUT_FMT_AVHW : RGY_INPUT_FMT_AVSW;
        //念のため初期化
        m_sTrimParam.list.clear();
        m_sTrimParam.offset = 0;

        //HEVC入力の際に大量にメッセージが出て劇的に遅くなることがあるのを回避
        if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            AVDictionary *pDict = nullptr;
            av_dict_set_int(&pDict, "log_level_offset", AV_LOG_ERROR, 0);
            if (0 > (ret = av_opt_set_dict(m_Demux.video.pStream->codec, &pDict))) {
                AddMessage(RGY_LOG_WARN, _T("failed to set log_level_offset for HEVC codec reader.\n"));
            } else {
                AddMessage(RGY_LOG_DEBUG, _T("set log_level_offset for HEVC codec reader.\n"));
            }
            av_dict_free(&pDict);
        }

        //必要ならbitstream filterを初期化
        if (m_Demux.video.nHWDecodeDeviceId >= 0 && m_Demux.video.pStream->codecpar->extradata && m_Demux.video.pStream->codecpar->extradata[0] == 1) {
            if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
                m_Demux.video.bUseHEVCmp42AnnexB = true;
            } else if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_H264 ||
                m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
                const char *filtername = nullptr;
                switch (m_Demux.video.pStream->codecpar->codec_id) {
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
                if (0 > (ret = av_bsf_alloc(filter, &m_Demux.video.pBsfcCtx))) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_NULL_PTR;
                }
                m_Demux.video.pBsfcCtx->time_base_in = av_stream_get_codec_timebase(m_Demux.video.pStream);
                if (0 > (ret = avcodec_parameters_from_context(m_Demux.video.pBsfcCtx->par_in, m_Demux.video.pStream->codec))) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to set parameter for %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_NULL_PTR;
                }
                m_Demux.video.pBsfcCtx->time_base_in = m_Demux.video.pStream->time_base;
                if (0 > (ret = av_bsf_init(m_Demux.video.pBsfcCtx))) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), char_to_tstring(filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_NULL_PTR;
                }
                AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter.\n"), char_to_tstring(filter->name).c_str());
            }
        //HWデコードする場合には、ヘッダーが必要
        } else if (m_Demux.video.nHWDecodeDeviceId >= 0
            && (m_inputVideoInfo.codec != RGY_CODEC_VP8 && m_inputVideoInfo.codec != RGY_CODEC_VP9)
            && m_Demux.video.pStream->codecpar->extradata == nullptr
            && m_Demux.video.pStream->codecpar->extradata_size == 0) {
            AddMessage(RGY_LOG_ERROR, _T("video header not extracted by libavcodec.\n"));
            return RGY_ERR_UNKNOWN;
        }
        if (m_Demux.video.pStream->codecpar->extradata_size) {
            m_inputVideoInfo.codecExtra = m_Demux.video.pStream->codecpar->extradata;
            m_inputVideoInfo.codecExtraSize = m_Demux.video.pStream->codecpar->extradata_size;
        }

        AddMessage(RGY_LOG_DEBUG, _T("start predecode.\n"));

        //ヘッダーの取得を確認する
        RGY_ERR sts = RGY_ERR_NONE;
        RGYBitstream bitstream = RGYBitstreamInit();
        if (m_Demux.video.pStream->codecpar->extradata) {
            if (RGY_ERR_NONE != (sts = GetHeader(&bitstream))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to get header.\n"));
                return sts;
            }
            m_inputVideoInfo.codecExtra = m_Demux.video.pExtradata;
            m_inputVideoInfo.codecExtraSize = m_Demux.video.nExtradataSize;
        }
        if (input_prm->fSeekSec > 0.0f) {
            AVPacket firstpkt;
            getSample(&firstpkt); //現在のtimestampを取得する
            const auto seek_time = av_rescale_q(1, av_d2q((double)input_prm->fSeekSec, 1<<24), m_Demux.video.pStream->time_base);
            int seek_ret = av_seek_frame(m_Demux.format.pFormatCtx, m_Demux.video.nIndex, firstpkt.pts + seek_time, 0);
            if (0 > seek_ret) {
                seek_ret = av_seek_frame(m_Demux.format.pFormatCtx, m_Demux.video.nIndex, firstpkt.pts + seek_time, AVSEEK_FLAG_ANY);
            }
            av_packet_unref(&firstpkt);
            if (0 > seek_ret) {
                AddMessage(RGY_LOG_ERROR, _T("failed to seek %s.\n"), print_time(input_prm->fSeekSec).c_str());
                return RGY_ERR_UNKNOWN;
            }
            //seekのために行ったgetSampleの結果は破棄する
            m_Demux.frames.clear();
        }

        //parserはseek後に初期化すること
        //parserが使用されていれば、ここでも使用するようにする
        //たとえば、入力がrawcodecなどでは使用しない
        m_Demux.video.pParserCtx = av_parser_init(m_Demux.video.pStream->codecpar->codec_id);
        if (m_Demux.video.pParserCtx) {
            m_Demux.video.pParserCtx->flags |= PARSER_FLAG_COMPLETE_FRAMES;
            if (nullptr == (m_Demux.video.pCodecCtxParser = avcodec_alloc_context3(avcodec_find_decoder(m_Demux.video.pStream->codecpar->codec_id)))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            unique_ptr_custom<AVCodecParameters> codecParamCopy(avcodec_parameters_alloc(), [](AVCodecParameters *pCodecPar) {
                avcodec_parameters_free(&pCodecPar);
            });
            if (0 > (ret = avcodec_parameters_from_context(codecParamCopy.get(), m_Demux.video.pStream->codec))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            if (m_Demux.video.pBsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB) {
                SetExtraData(codecParamCopy.get(), m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
            }
            if (0 > (ret = avcodec_parameters_to_context(m_Demux.video.pCodecCtxParser, codecParamCopy.get()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            m_Demux.video.pCodecCtxParser->time_base = av_stream_get_codec_timebase(m_Demux.video.pStream);
            m_Demux.video.pCodecCtxParser->pkt_timebase = m_Demux.video.pStream->time_base;
            AddMessage(RGY_LOG_DEBUG, _T("initialized %s codec context for parser: time_base: %d/%d, pkt_timebase: %d/%d.\n"),
                char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str(),
                m_Demux.video.pCodecCtxParser->time_base.num, m_Demux.video.pCodecCtxParser->time_base.den,
                m_Demux.video.pCodecCtxParser->pkt_timebase.num, m_Demux.video.pCodecCtxParser->pkt_timebase.den);
        } else if (m_Demux.video.nHWDecodeDeviceId >= 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init parser for %s.\n"), char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str());
            return RGY_ERR_NULL_PTR;
        }
#if _DEBUG
        if (m_Demux.frames.setLogCopyFrameData(input_prm->pLogCopyFrameData)) {
            AddMessage(RGY_LOG_WARN, _T("failed to open copy-framedata log file: \"%s\"\n"), input_prm->pLogCopyFrameData);
        }
#endif

        if (RGY_ERR_NONE != (sts = getFirstFramePosAndFrameRate(input_prm->pTrimList, input_prm->nTrimCount, input_prm->bVideoDetectPulldown))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get first frame position.\n"));
            return sts;
        }
        if (m_cap2ass.enabled()) {
            m_cap2ass.setVidFirstKeyPts(m_Demux.video.nStreamFirstKeyPts);
        }

        m_sTrimParam.list = make_vector(input_prm->pTrimList, input_prm->nTrimCount);
        //キーフレームに到達するまでQSVではフレームが出てこない
        //そのぶんのずれを記録しておき、Trim値などに補正をかける
        if (m_sTrimParam.offset) {
            for (int i = (int)m_sTrimParam.list.size() - 1; i >= 0; i--) {
                if (m_sTrimParam.list[i].fin - m_sTrimParam.offset < 0) {
                    m_sTrimParam.list.erase(m_sTrimParam.list.begin() + i);
                } else {
                    m_sTrimParam.list[i].start = (std::max)(0, m_sTrimParam.list[i].start - m_sTrimParam.offset);
                    if (m_sTrimParam.list[i].fin != TRIM_MAX) {
                        m_sTrimParam.list[i].fin = (std::max)(0, m_sTrimParam.list[i].fin - m_sTrimParam.offset);
                    }
                }
            }
            //ずれが存在し、範囲指定がない場合はダミーの全域指定を追加する
            //これにより、自動的に音声側との同期がとれるようになる
            if (m_sTrimParam.list.size() == 0) {
                m_sTrimParam.list.push_back({ 0, TRIM_MAX });
            }
            AddMessage(RGY_LOG_DEBUG, _T("adjust trim by offset %d.\n"), m_sTrimParam.offset);
        }

        //あらかじめfpsが指定されていればそれを採用する
        if (input_prm->nVideoAvgFramerate.first * input_prm->nVideoAvgFramerate.second > 0) {
            m_Demux.video.nAvgFramerate.num = input_prm->nVideoAvgFramerate.first;
            m_Demux.video.nAvgFramerate.den = input_prm->nVideoAvgFramerate.second;
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
            { AV_PIX_FMT_YUV422P,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV16 },
            { AV_PIX_FMT_YUVJ422P,     8, RGY_CHROMAFMT_YUV422, RGY_CSP_NA },
            { AV_PIX_FMT_YUYV422,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_YUY2 },
            { AV_PIX_FMT_UYVY422,      8, RGY_CHROMAFMT_YUV422, RGY_CSP_NA },
            { AV_PIX_FMT_NV16,         8, RGY_CHROMAFMT_YUV422, RGY_CSP_NV16 },
            { AV_PIX_FMT_YUV444P,      8, RGY_CHROMAFMT_YUV444, RGY_CSP_NV12 },
            { AV_PIX_FMT_YUVJ444P,     8, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444 },
            { AV_PIX_FMT_YUV420P16LE, 16, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P14LE, 14, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P12LE, 12, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P10LE, 10, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_YUV420P9LE,   9, RGY_CHROMAFMT_YUV420, RGY_CSP_P010 },
            { AV_PIX_FMT_NV20LE,      10, RGY_CHROMAFMT_YUV420, RGY_CSP_NA },
            { AV_PIX_FMT_YUV422P16LE, 16, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P14LE, 14, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P12LE, 12, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV422P10LE, 10, RGY_CHROMAFMT_YUV422, RGY_CSP_P210 },
            { AV_PIX_FMT_YUV444P16LE, 16, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P14LE, 14, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P12LE, 12, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P10LE, 10, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_YUV444P9LE,   9, RGY_CHROMAFMT_YUV444, RGY_CSP_YUV444_16 },
            { AV_PIX_FMT_RGB24,        8, RGY_CHROMAFMT_RGB,    (ENCODER_NVENC) ? RGY_CSP_RGB24 : RGY_CSP_RGB32 },
            { AV_PIX_FMT_RGBA,         8, RGY_CHROMAFMT_RGB,    RGY_CSP_RGB32 }
        };

        const auto pixfmt = (AVPixelFormat)m_Demux.video.pStream->codecpar->format;
        const auto pixfmtData = std::find_if(pixfmtDataList, pixfmtDataList + _countof(pixfmtDataList), [pixfmt](const pixfmtInfo& tableData) {
            return tableData.pix_fmt == pixfmt;
        });
        if (pixfmtData == (pixfmtDataList + _countof(pixfmtDataList)) || pixfmtData->output_csp == RGY_CSP_NA) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid pixel format \"%s\" from input file.\n"), char_to_tstring(av_get_pix_fmt_name(pixfmt)).c_str());
            return RGY_ERR_INVALID_COLOR_FORMAT;
        }

        const auto aspectRatio = m_Demux.video.pStream->codecpar->sample_aspect_ratio;
        const bool bAspectRatioUnknown = aspectRatio.num * aspectRatio.den <= 0;

        if (!(m_Demux.video.nHWDecodeDeviceId >= 0)) {
            if (nullptr == (m_Demux.video.pCodecDecode = avcodec_find_decoder(m_Demux.video.pStream->codecpar->codec_id))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to find decoder"), m_Demux.video.pStream->codecpar->codec_id).c_str());
                return RGY_ERR_NOT_FOUND;
            }
            if (nullptr == (m_Demux.video.pCodecCtxDecode = avcodec_alloc_context3(m_Demux.video.pCodecDecode))) {
                AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("Failed to allocate decoder"), m_Demux.video.pStream->codecpar->codec_id).c_str());
                return RGY_ERR_NULL_PTR;
            }
            unique_ptr_custom<AVCodecParameters> codecParamCopy(avcodec_parameters_alloc(), [](AVCodecParameters *pCodecPar) {
                avcodec_parameters_free(&pCodecPar);
            });
            if (0 > (ret = avcodec_parameters_copy(codecParamCopy.get(), m_Demux.video.pStream->codecpar))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy codec param to context for parser: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            if (m_Demux.video.pBsfcCtx || m_Demux.video.bUseHEVCmp42AnnexB) {
                SetExtraData(codecParamCopy.get(), m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
            }
            if (0 > (ret = avcodec_parameters_to_context(m_Demux.video.pCodecCtxDecode, codecParamCopy.get()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set codec param to context for decoder: %s.\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            cpu_info_t cpu_info;
            if (get_cpu_info(&cpu_info)) {
                AVDictionary *pDict = nullptr;
                av_dict_set_int(&pDict, "threads", std::min(cpu_info.logical_cores, 16u), 0);
                if (0 > (ret = av_opt_set_dict(m_Demux.video.pCodecCtxDecode, &pDict))) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to set threads for decode (codec: %s): %s\n"),
                        char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_UNKNOWN;
                }
                av_dict_free(&pDict);
            }
            m_Demux.video.pCodecCtxDecode->time_base = av_stream_get_codec_timebase(m_Demux.video.pStream);
            m_Demux.video.pCodecCtxDecode->pkt_timebase = m_Demux.video.pStream->time_base;
            if (0 > (ret = avcodec_open2(m_Demux.video.pCodecCtxDecode, m_Demux.video.pCodecDecode, nullptr))) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open decoder for %s: %s\n"), char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNSUPPORTED;
            }

            const auto pixCspConv = csp_avpixfmt_to_rgy(m_Demux.video.pCodecCtxDecode->pix_fmt);
            //出力フォーマットへの直接変換を持たないものは、pixfmtDataListに従う
            switch (pixCspConv) {
            case RGY_CSP_NV16:
            case RGY_CSP_YUV422:
            case RGY_CSP_YUV422_09:
            case RGY_CSP_YUV422_10:
            case RGY_CSP_YUV422_12:
            case RGY_CSP_YUV422_14:
            case RGY_CSP_YUV422_16:
            case RGY_CSP_P210:
            case RGY_CSP_RGB24:
            case RGY_CSP_RGB32:
                m_inputVideoInfo.csp = pixfmtData->output_csp;
                m_inputVideoInfo.shift = (RGY_CSP_BIT_DEPTH[pixCspConv] > 8) ? 16 - RGY_CSP_BIT_DEPTH[pixCspConv] : 0;
            default:
                break;
            }
            if (pixCspConv == RGY_CSP_NA
                || nullptr == (m_sConvert = get_convert_csp_func(pixCspConv, m_inputVideoInfo.csp, false))) {
                AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
                     RGY_CSP_NAMES[pixCspConv], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
                return RGY_ERR_INVALID_COLOR_FORMAT;
            }
            m_InputCsp = pixCspConv;
            m_inputVideoInfo.shift = (m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_P210) ? m_inputVideoInfo.shift : 0;
            if (nullptr == (m_Demux.video.pFrame = av_frame_alloc())) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to allocate frame for decoder.\n"));
                return RGY_ERR_NULL_PTR;
            }
        } else {
            //HWデコードの場合は、色変換がかからないので、入力フォーマットがそのまま出力フォーマットとなる
            m_inputVideoInfo.csp = pixfmtData->output_csp;
            m_inputVideoInfo.shift = (m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_P210) ? 16 - pixfmtData->bit_depth : 0;
        }

        m_Demux.format.nAVSyncMode = input_prm->nAVSyncMode;

        //情報を格納
        m_inputVideoInfo.srcWidth    = m_Demux.video.pStream->codecpar->width;
        m_inputVideoInfo.srcHeight   = m_Demux.video.pStream->codecpar->height;
        m_inputVideoInfo.codedWidth  = m_Demux.video.pStream->codec->coded_width;
        m_inputVideoInfo.codedHeight = m_Demux.video.pStream->codec->coded_height;
        m_inputVideoInfo.sar[0]      = (bAspectRatioUnknown) ? 0 : m_Demux.video.pStream->codecpar->sample_aspect_ratio.num;
        m_inputVideoInfo.sar[1]      = (bAspectRatioUnknown) ? 0 : m_Demux.video.pStream->codecpar->sample_aspect_ratio.den;
        m_inputVideoInfo.shift       = ((m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_P210) && m_inputVideoInfo.shift) ? m_inputVideoInfo.shift : 0;
        m_inputVideoInfo.picstruct   = m_Demux.frames.getVideoPicStruct();
        m_inputVideoInfo.frames      = 0;
        //getFirstFramePosAndFrameRateをもとにfpsを決定
        m_inputVideoInfo.fpsN        = m_Demux.video.nAvgFramerate.num;
        m_inputVideoInfo.fpsD        = m_Demux.video.nAvgFramerate.den;

        if (m_Demux.video.nHWDecodeDeviceId >= 0) {
            tstring mes = strsprintf(_T("av" DECODER_NAME ": %s, %dx%d, %d/%d fps"),
                CodecToStr(m_inputVideoInfo.codec).c_str(),
                m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
            if (input_prm->fSeekSec > 0.0f) {
                mes += strsprintf(_T("\n         seek: %s"), print_time(input_prm->fSeekSec).c_str());
            }
            AddMessage(RGY_LOG_DEBUG, mes);
            m_strInputInfo += mes;
        } else {
            CreateInputInfo((tstring(_T("avsw: ")) + char_to_tstring(avcodec_get_name(m_Demux.video.pStream->codecpar->codec_id))).c_str(),
                RGY_CSP_NAMES[m_sConvert->csp_from], RGY_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd), &m_inputVideoInfo);
            if (input_prm->fSeekSec > 0.0f) {
                m_strInputInfo += strsprintf(_T("\n         seek: %s"), print_time(input_prm->fSeekSec).c_str());
            }
            AddMessage(RGY_LOG_DEBUG, m_strInputInfo);
        }
        AddMessage(RGY_LOG_DEBUG, _T("sar %d:%d, shift %d\n"),
            m_inputVideoInfo.sar[0], m_inputVideoInfo.sar[1], m_inputVideoInfo.shift);

        *pInputInfo = m_inputVideoInfo;

        //スレッド関連初期化
        m_Demux.thread.bAbortInput = false;
#if ENCODER_QSV
        auto nPrmInputThread = input_prm->nInputThread;
        m_Demux.thread.nInputThread = ((nPrmInputThread == RGY_INPUT_THREAD_AUTO) | (m_Demux.video.pStream != nullptr)) ? 0 : nPrmInputThread;
#else
        //NVEncではいまのところ、常に無効
        m_Demux.thread.nInputThread = 0;
#endif
        if (m_Demux.thread.nInputThread) {
            m_Demux.thread.thInput = std::thread(&RGYInputAvcodec::ThreadFuncRead, this);
            //はじめcapacityを無限大にセットしたので、この段階で制限をかける
            //入力をスレッド化しない場合には、自動的に同期が保たれるので、ここでの制限は必要ない
            m_Demux.qVideoPkt.set_capacity(256);
        }
    } else {
        //音声との同期とかに使うので、動画の情報を格納する
        m_Demux.video.nAvgFramerate = av_make_q(input_prm->nVideoAvgFramerate.first, input_prm->nVideoAvgFramerate.second);

        if (input_prm->nTrimCount) {
            m_sTrimParam.list = vector<sTrim>(input_prm->pTrimList, input_prm->pTrimList + input_prm->nTrimCount);
        }

        if (m_Demux.video.pStream) {
            //動画の最初のフレームを取得しておく
            AVPacket pkt;
            av_init_packet(&pkt);
            //音声のみ処理モードでは、動画の先頭をキーフレームとする必要はなく、
            //先頭がキーフレームでなくてもframePosListに追加するようにして、trimをoffsetなしで反映できるようにする
            //そこで、bTreatFirstPacketAsKeyframe=trueにして最初のパケットを処理する
            getSample(&pkt, true);
            av_packet_unref(&pkt);

            m_Demux.frames.checkPtsStatus();
        }

        tstring mes;
        for (const auto& stream : m_Demux.stream) {
            if (mes.length()) mes += _T(", ");
            tstring codec_name = char_to_tstring(avcodec_get_name(stream.pStream->codecpar->codec_id));
            mes += codec_name;
            AddMessage(RGY_LOG_DEBUG, _T("avcodec %s: %s from %s\n"),
                get_media_type_string(stream.pStream->codecpar->codec_id).c_str(), codec_name.c_str(), strFileName);
        }
        m_strInputInfo += _T("avcodec audio: ") + mes;
    }
    return RGY_ERR_NONE;
}
#pragma warning(pop)

vector<const AVChapter *> RGYInputAvcodec::GetChapterList() {
    return m_Demux.chapter;
}

int RGYInputAvcodec::GetSubtitleTrackCount() {
    return m_Demux.format.nSubtitleTracks;
}

int RGYInputAvcodec::GetAudioTrackCount() {
    return m_Demux.format.nAudioTracks;
}

int64_t RGYInputAvcodec::GetVideoFirstKeyPts() {
    return m_Demux.video.nStreamFirstKeyPts;
}

FramePosList *RGYInputAvcodec::GetFramePosList() {
    return &m_Demux.frames;
}

int RGYInputAvcodec::getVideoFrameIdx(int64_t pts, AVRational timebase, int iStart) {
    const int framePosCount = m_Demux.frames.frameNum();
    const AVRational vid_pkt_timebase = (m_Demux.video.pStream) ? m_Demux.video.pStream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
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

int64_t RGYInputAvcodec::convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *pStream) {
    const AVRational vid_pkt_timebase = (m_Demux.video.pStream) ? m_Demux.video.pStream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
    return av_rescale_q(pts, vid_pkt_timebase, pStream->timebase);
}

bool RGYInputAvcodec::checkStreamPacketToAdd(const AVPacket *pkt, AVDemuxStream *pStream) {
    pStream->nLastVidIndex = getVideoFrameIdx(pkt->pts, pStream->timebase, pStream->nLastVidIndex);

    //該当フレームが-1フレーム未満なら、その音声はこの動画には含まれない
    if (pStream->nLastVidIndex < -1) {
        return false;
    }

    const auto vidFramePos = &m_Demux.frames.list((std::max)(pStream->nLastVidIndex, 0));
    const int64_t vid_fin = convertTimebaseVidToStream(vidFramePos->pts + ((pStream->nLastVidIndex >= 0) ? vidFramePos->duration : 0), pStream);

    const int64_t aud_start = pkt->pts;
    const int64_t aud_fin   = pkt->pts + pkt->duration;

    const bool frame_is_in_range = frame_inside_range(pStream->nLastVidIndex,     m_sTrimParam.list);
    const bool next_is_in_range  = frame_inside_range(pStream->nLastVidIndex + 1, m_sTrimParam.list);

    bool result = true; //動画に含まれる音声かどうか

    if (frame_is_in_range) {
        if (aud_fin < vid_fin || next_is_in_range) {
            ; //完全に動画フレームの範囲内か、次のフレームも範囲内なら、その音声パケットは含まれる
              //              vid_fin
              //動画 <-----------|
              //音声      |-----------|
              //     aud_start     aud_fin
        } else if (pkt->duration / 2 > (aud_fin - vid_fin + pStream->nExtractErrExcess)) {
            //はみ出した領域が少ないなら、その音声パケットは含まれる
            pStream->nExtractErrExcess += aud_fin - vid_fin;
        } else {
            //はみ出した領域が多いなら、その音声パケットは含まれない
            pStream->nExtractErrExcess -= vid_fin - aud_start;
            result = false;
        }
    } else if (next_is_in_range && aud_fin > vid_fin) {
        //             vid_fin
        //動画             |------------>
        //音声      |-----------|
        //     aud_start     aud_fin
        if (pkt->duration / 2 > (vid_fin - aud_start + pStream->nExtractErrExcess)) {
            pStream->nExtractErrExcess += vid_fin - aud_start;
        } else {
            pStream->nExtractErrExcess -= aud_fin - vid_fin;
            result = false;
        }
    } else {
        result = false;
    }
    return result;
}

AVDemuxStream *RGYInputAvcodec::getPacketStreamData(const AVPacket *pkt) {
    int streamIndex = pkt->stream_index;
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        if (m_Demux.stream[i].nIndex == streamIndex) {
            return &m_Demux.stream[i];
        }
    }
    return nullptr;
}

int RGYInputAvcodec::getSample(AVPacket *pkt, bool bTreatFirstPacketAsKeyframe) {
    av_init_packet(pkt);
    int i_samples = 0;
    int ret_read_frame = 0;
    while ((ret_read_frame = av_read_frame(m_Demux.format.pFormatCtx, pkt)) >= 0
        //trimからわかるフレーム数の上限値よりfixedNumがある程度の量の処理を進めたら読み込みを打ち切る
        && m_Demux.frames.fixedNum() - TRIM_OVERREAD_FRAMES < getVideoTrimMaxFramIdx()) {
        if (pkt->stream_index == m_Demux.video.nIndex) {
            if (m_Demux.video.pBsfcCtx) {
                auto ret = av_bsf_send_packet(m_Demux.video.pBsfcCtx, pkt);
                if (ret < 0) {
                    av_packet_unref(pkt);
                    AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"), char_to_tstring(m_Demux.video.pBsfcCtx->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return 1;
                }
                ret = av_bsf_receive_packet(m_Demux.video.pBsfcCtx, pkt);
                if (ret == AVERROR(EAGAIN)) {
                    continue; //もっとpacketを送らないとダメ
                } else if (ret < 0 && ret != AVERROR_EOF) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"), char_to_tstring(m_Demux.video.pBsfcCtx->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return 1;
                }
            }
            if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_VC1) {
                vc1AddFrameHeader(pkt);
            }
            if (m_Demux.video.bUseHEVCmp42AnnexB) {
                hevcMp42Annexb(pkt);
            }
            //最初のキーフレームを取得するまではスキップする
            //スキップした枚数はi_samplesでカウントし、trim時に同期を適切にとるため、m_sTrimParam.offsetに格納する
            //  ただし、bTreatFirstPacketAsKeyframeが指定されている場合には、キーフレームでなくてもframePosListへの追加を許可する
            //  このモードは、対象の入力ファイルから--audio-sourceなどで音声のみ拾ってくる場合に使用する
            if (!bTreatFirstPacketAsKeyframe && !m_Demux.video.bGotFirstKeyframe && !(pkt->flags & AV_PKT_FLAG_KEY)) {
                av_packet_unref(pkt);
                i_samples++;
                continue;
            } else {
                if (!m_Demux.video.bGotFirstKeyframe) {
                    if (pkt->flags & AV_PKT_FLAG_DISCARD) {
                        //timestampが正常に設定されておらず、移乗動作の原因となるので、
                        //AV_PKT_FLAG_DISCARDがついている最初のフレームは無視する
                        continue;
                    }
                    //ここに入った場合は、必ず最初のキーフレーム
                    m_Demux.video.nStreamFirstKeyPts = pkt->pts;
                    m_Demux.video.bGotFirstKeyframe = true;
                    //キーフレームに到達するまでQSVではフレームが出てこない
                    //そのため、getSampleでも最初のキーフレームを取得するまでパケットを出力しない
                    //だが、これが原因でtrimの値とずれを生じてしまう
                    //そこで、そのぶんのずれを記録しておき、Trim値などに補正をかける
                    m_sTrimParam.offset = i_samples;
                }
#if ENCODER_NVENC
                //NVENCのhwデコーダでは、opengopなどでキーフレームのパケットよりあとにその前のフレームが来た場合、
                //フレーム位置がさらにずれるので補正する
                else if (!(pkt->flags & AV_PKT_FLAG_KEY)
                    && (pkt->pts != AV_NOPTS_VALUE)
                    && (m_Demux.video.nStreamFirstKeyPts != AV_NOPTS_VALUE)
                    && pkt->pts < m_Demux.video.nStreamFirstKeyPts
                    && m_Demux.video.nHWDecodeDeviceId >= 0) { //trim調整の適用はavhwリーダーのみ
                    m_sTrimParam.offset++;
                }
#endif //#if ENCODER_NVENC
                FramePos pos = { 0 };
                pos.pts = pkt->pts;
                pos.dts = pkt->dts;
                pos.duration = (int)pkt->duration;
                pos.duration2 = 0;
                pos.poc = FRAMEPOS_POC_INVALID;
                pos.flags = (uint8_t)pkt->flags;
                if (m_Demux.video.pParserCtx) {
                    uint8_t *dummy = nullptr;
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
                m_Demux.frames.add(pos);
            }
            //ptsの確定したところまで、音声を出力する
            CheckAndMoveStreamPacketList();
            return 0;
        }
        if (getPacketStreamData(pkt) != NULL) {
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
    if (m_Demux.video.nStreamPtsInvalid & RGY_PTS_ALL_INVALID) {
        videoFinPts = nFrameNum * m_Demux.frames.list(0).duration;
    } else if (nFrameNum) {
        const FramePos *lastFrame = &m_Demux.frames.list(nFrameNum - 1);
        videoFinPts = lastFrame->pts + lastFrame->duration;
    }
    //もし選択範囲が手動で決定されていないのなら、音声を最大限取得する
    if (m_sTrimParam.list.size() == 0 || m_sTrimParam.list.back().fin == TRIM_MAX) {
        for (uint32_t i = 0; i < m_Demux.qStreamPktL2.size(); i++) {
            videoFinPts = (std::max)(videoFinPts, m_Demux.qStreamPktL2[i].data.pts);
        }
        for (uint32_t i = 0; i < m_Demux.qStreamPktL1.size(); i++) {
            videoFinPts = (std::max)(videoFinPts, m_Demux.qStreamPktL1[i].pts);
        }
    }
    //最後のフレーム情報をセットし、m_Demux.framesの内部状態を終了状態に移行する
    m_Demux.frames.fin(framePos(videoFinPts, videoFinPts, 0), m_Demux.format.pFormatCtx->duration);
    //映像キューのサイズ維持制限を解除する → パイプラインに最後まで読み取らせる
    m_Demux.qVideoPkt.set_keep_length(0);
    //音声をすべて出力する
    //m_Demux.frames.finをしたので、ここで実行すれば、qAudioPktL1のデータがすべてqAudioPktL2に移される
    CheckAndMoveStreamPacketList();
    //音声のみ読み込みの場合はm_pEncSatusInfoはnullptrなので、nullチェックを行う
#if !FOR_AUO //auoでここからUpdateDisplay()してしまうと、メインスレッド以外からのGUI更新となり、例外で落ちる
    if (m_pEncSatusInfo) {
        m_pEncSatusInfo->UpdateDisplay(100.0);
    }
#endif
    return 1;
}

//動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは消す)
RGY_ERR RGYInputAvcodec::GetNextBitstream(RGYBitstream *pBitstream) {
    AVPacket pkt;
    if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
        && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
        if (0 == getSample(&pkt)) {
            m_Demux.qVideoPkt.push(pkt);
        }
    }

    bool bGetPacket = false;
    for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_and_pop_no_lock(&pkt, (m_Demux.thread.pQueueInfo) ? &m_Demux.thread.pQueueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
        m_Demux.qVideoPkt.wait_for_push();
    }
    RGY_ERR sts = RGY_ERR_MORE_BITSTREAM;
    if (bGetPacket) {
        if (pkt.data) {
            auto pts = (0 == (m_Demux.frames.getStreamPtsStatus() & (~RGY_PTS_NORMAL))) ? pkt.pts : AV_NOPTS_VALUE;
            sts = pBitstream->copy(pkt.data, pkt.size, pkt.dts, pts);
        }
        av_packet_unref(&pkt);
        m_Demux.video.nSampleGetCount++;
        m_pEncSatusInfo->m_sData.frameIn++;
    }
    return sts;
}

//動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは残す)
RGY_ERR RGYInputAvcodec::GetNextBitstreamNoDelete(RGYBitstream *pBitstream) {
    AVPacket pkt;
    if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
        && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
        if (0 == getSample(&pkt)) {
            m_Demux.qVideoPkt.push(pkt);
        }
    }

    bool bGetPacket = false;
    for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_no_lock(&pkt, (m_Demux.thread.pQueueInfo) ? &m_Demux.thread.pQueueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
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

void RGYInputAvcodec::GetAudioDataPacketsWhenNoVideoRead() {
    m_Demux.video.nSampleGetCount++;

    AVPacket pkt;
    av_init_packet(&pkt);
    if (m_Demux.video.pStream) {
        //動画に映像がある場合、getSampleを呼んで1フレーム分の音声データをm_Demux.qStreamPktL1に取得する
        //同時に映像フレームをロードし、ロードしたptsデータを突っ込む
        if (!getSample(&pkt)) {
            //動画データ自体は不要なので解放
            av_packet_unref(&pkt);
            CheckAndMoveStreamPacketList();
        }
        return;
    } else {
        const double vidEstDurationSec = m_Demux.video.nSampleGetCount * (double)m_Demux.video.nAvgFramerate.den / (double)m_Demux.video.nAvgFramerate.num; //1フレームの時間(秒)
        //動画に映像がない場合、
        //およそ1フレーム分のパケットを取得する
        while (av_read_frame(m_Demux.format.pFormatCtx, &pkt) >= 0) {
            if (m_Demux.format.pFormatCtx->streams[pkt.stream_index]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
                av_packet_unref(&pkt);
            } else {
                AVDemuxStream *pStream = getPacketStreamData(&pkt);
                if (checkStreamPacketToAdd(&pkt, pStream)) {
                    m_Demux.qStreamPktL1.push_back(pkt);
                } else {
                    av_packet_unref(&pkt); //Writer側に渡さないパケットはここで開放する
                }

                //最初のパケットは参照用にコピーしておく
                if (pStream->pktSample.data == nullptr) {
                    av_copy_packet(&pStream->pktSample, &pkt);
                }
                uint64_t pktt = (pkt.pts == AV_NOPTS_VALUE) ? pkt.dts : pkt.pts;
                uint64_t pkt_dist = pktt - pStream->pktSample.pts;
                //1フレーム分のサンプルを取得したら終了
                if (pkt_dist * (double)pStream->pStream->time_base.num / (double)pStream->pStream->time_base.den > vidEstDurationSec) {
                    //およそ1フレーム分のパケットを設定する
                    int64_t pts = m_Demux.video.nSampleGetCount;
                    m_Demux.frames.add(framePos(pts, pts, 1, 0, m_Demux.video.nSampleGetCount, AV_PKT_FLAG_KEY));
                    if (m_Demux.frames.getStreamPtsStatus() == RGY_PTS_UNKNOWN) {
                        m_Demux.frames.checkPtsStatus();
                    }
                    CheckAndMoveStreamPacketList();
                    return;
                }
            }
        }
        //読み込みが終了
        int64_t pts = m_Demux.video.nSampleGetCount;
        m_Demux.frames.fin(framePos(pts, pts, 1, 0, m_Demux.video.nSampleGetCount, AV_PKT_FLAG_KEY), m_Demux.video.nSampleGetCount);
    }
}

const AVDictionary *RGYInputAvcodec::GetInputFormatMetadata() {
    return m_Demux.format.pFormatCtx->metadata;
}

const AVStream *RGYInputAvcodec::GetInputVideoStream() {
    return m_Demux.video.pStream;
}

double RGYInputAvcodec::GetInputVideoDuration() {
    return (m_Demux.format.pFormatCtx->duration * (1.0 / (double)AV_TIME_BASE));
}

//qStreamPktL1をチェックし、framePosListから必要な音声パケットかどうかを判定し、
//必要ならqStreamPktL2に移し、不要ならパケットを開放する
void RGYInputAvcodec::CheckAndMoveStreamPacketList() {
    if (m_Demux.frames.fixedNum() == 0) {
        return;
    }
    //出力するパケットを選択する
    const AVRational vid_pkt_timebase = (m_Demux.video.pStream) ? m_Demux.video.pStream->time_base : av_inv_q(m_Demux.video.nAvgFramerate);
    while (!m_Demux.qStreamPktL1.empty()) {
        auto pkt = m_Demux.qStreamPktL1.front();
        AVDemuxStream *pStream = getPacketStreamData(&pkt);
        //音声のptsが映像の終わりのptsを行きすぎたらやめる
        if (0 < av_compare_ts(pkt.pts, pStream->timebase, m_Demux.frames.list(m_Demux.frames.fixedNum()).pts, vid_pkt_timebase)) {
            break;
        }
        if (checkStreamPacketToAdd(&pkt, pStream)) {
            pkt.flags = (pkt.flags & 0xffff) | (pStream->nTrackId << 16); //flagsの上位16bitには、trackIdへのポインタを格納しておく
            m_Demux.qStreamPktL2.push(pkt); //Writer側に渡したパケットはWriter側で開放する
        } else {
            av_packet_unref(&pkt); //Writer側に渡さないパケットはここで開放する
        }
        m_Demux.qStreamPktL1.pop_front();
    }
}

vector<AVPacket> RGYInputAvcodec::GetStreamDataPackets() {
    if (!m_Demux.video.bReadVideo) {
        GetAudioDataPacketsWhenNoVideoRead();
    }

    //出力するパケットを選択する
    vector<AVPacket> packets;
    AVPacket pkt;
    while (m_Demux.qStreamPktL2.front_copy_and_pop_no_lock(&pkt, (m_Demux.thread.pQueueInfo) ? &m_Demux.thread.pQueueInfo->usage_aud_in : nullptr)) {
        packets.push_back(pkt);
    }
    return std::move(packets);
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

    if (m_Demux.video.pExtradata == nullptr) {
        m_Demux.video.nExtradataSize = m_Demux.video.pStream->codecpar->extradata_size;
        //ここでav_mallocを使用しないと正常に動作しない
        m_Demux.video.pExtradata = (uint8_t *)av_malloc(m_Demux.video.pStream->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        //ヘッダのデータをコピーしておく
        memcpy(m_Demux.video.pExtradata, m_Demux.video.pStream->codecpar->extradata, m_Demux.video.nExtradataSize);
        memset(m_Demux.video.pExtradata + m_Demux.video.nExtradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        if (m_Demux.video.bUseHEVCmp42AnnexB) {
            hevcMp42Annexb(nullptr);
        } else if (m_Demux.video.pBsfcCtx && m_Demux.video.pExtradata[0] == 1) {
            if (m_Demux.video.nExtradataSize < m_Demux.video.pBsfcCtx->par_out->extradata_size) {
                m_Demux.video.pExtradata = (uint8_t *)av_realloc(m_Demux.video.pExtradata, m_Demux.video.pBsfcCtx->par_out->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
            }
            memcpy(m_Demux.video.pExtradata, m_Demux.video.pBsfcCtx->par_out->extradata, m_Demux.video.pBsfcCtx->par_out->extradata_size);
            AddMessage(RGY_LOG_DEBUG, _T("GetHeader: changed %d bytes -> %d bytes by %s.\n"),
                m_Demux.video.nExtradataSize, m_Demux.video.pBsfcCtx->par_out->extradata_size,
                char_to_tstring(m_Demux.video.pBsfcCtx->filter->name).c_str());
            m_Demux.video.nExtradataSize = m_Demux.video.pBsfcCtx->par_out->extradata_size;
            memset(m_Demux.video.pExtradata + m_Demux.video.nExtradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        } else if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_VC1) {
            int lengthFix = (0 == strcmp(m_Demux.format.pFormatCtx->iformat->name, "mpegts")) ? 0 : -1;
            vc1FixHeader(lengthFix);
        }
        AddMessage(RGY_LOG_DEBUG, _T("GetHeader: %d bytes.\n"), m_Demux.video.nExtradataSize);

        //抽出されたextradataが大きすぎる場合、適当に縮める
        //NVEncのデコーダが受け取れるヘッダは1024byteまで
        if (m_Demux.video.nExtradataSize > 1024) {
            if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_H264) {
                std::vector<nal_info> nal_list = parse_nal_unit_h264(m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
                const auto h264_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_SPS; });
                const auto h264_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_H264_PPS; });
                const bool header_check = (nal_list.end() != h264_sps_nal) && (nal_list.end() != h264_pps_nal);
                if (header_check) {
                    m_Demux.video.nExtradataSize = h264_sps_nal->size + h264_pps_nal->size;
                    uint8_t *new_ptr = (uint8_t *)av_malloc(m_Demux.video.nExtradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
                    memcpy(new_ptr, h264_sps_nal->ptr, h264_sps_nal->size);
                    memcpy(new_ptr + h264_sps_nal->size, h264_pps_nal->ptr, h264_pps_nal->size);
                    if (m_Demux.video.pExtradata) {
                        av_free(m_Demux.video.pExtradata);
                    }
                    m_Demux.video.pExtradata = new_ptr;
                }
            } else if (m_Demux.video.pStream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
                std::vector<nal_info> nal_list = parse_nal_unit_hevc(m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
                const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
                const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
                const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
                const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
                if (header_check) {
                    m_Demux.video.nExtradataSize = hevc_vps_nal->size + hevc_sps_nal->size + hevc_pps_nal->size;
                    uint8_t *new_ptr = (uint8_t *)av_malloc(m_Demux.video.nExtradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
                    memcpy(new_ptr, hevc_vps_nal->ptr, hevc_vps_nal->size);
                    memcpy(new_ptr + hevc_vps_nal->size, hevc_sps_nal->ptr, hevc_sps_nal->size);
                    memcpy(new_ptr + hevc_vps_nal->size + hevc_sps_nal->size, hevc_pps_nal->ptr, hevc_pps_nal->size);
                    if (m_Demux.video.pExtradata) {
                        av_free(m_Demux.video.pExtradata);
                    }
                    m_Demux.video.pExtradata = new_ptr;
                }
            }
            AddMessage(RGY_LOG_DEBUG, _T("GetHeader: shrinked header to %d bytes.\n"), m_Demux.video.nExtradataSize);
        }
    }
    pBitstream->copy(m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
    return RGY_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
RGY_ERR RGYInputAvcodec::LoadNextFrame(RGYFrame *pSurface) {
    if (m_Demux.video.pCodecCtxDecode) {
        //動画のデコードを行う
        int got_frame = 0;
        while (!got_frame) {
            AVPacket pkt;
            av_init_packet(&pkt);
            if (!m_Demux.thread.thInput.joinable() //入力スレッドがなければ、自分で読み込む
                && m_Demux.qVideoPkt.get_keep_length() > 0) { //keep_length == 0なら読み込みは終了していて、これ以上読み込む必要はない
                if (0 == getSample(&pkt)) {
                    m_Demux.qVideoPkt.push(pkt);
                }
            }

            bool bGetPacket = false;
            for (int i = 0; false == (bGetPacket = m_Demux.qVideoPkt.front_copy_no_lock(&pkt, (m_Demux.thread.pQueueInfo) ? &m_Demux.thread.pQueueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
                m_Demux.qVideoPkt.wait_for_push();
            }
            if (!bGetPacket) {
                //flushするためのパケット
                pkt.data = nullptr;
                pkt.size = 0;
            }
            int ret = avcodec_send_packet(m_Demux.video.pCodecCtxDecode, &pkt);
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
            ret = avcodec_receive_frame(m_Demux.video.pCodecCtxDecode, m_Demux.video.pFrame);
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
        pSurface->setTimestamp(m_Demux.video.pFrame->pts);
        pSurface->setDuration(m_Demux.video.pFrame->pkt_duration);
        //フレームデータをコピー
        void *dst_array[3];
        pSurface->ptrArray(dst_array, m_sConvert->csp_to == RGY_CSP_RGB24 || m_sConvert->csp_to == RGY_CSP_RGB32);
        m_sConvert->func[m_Demux.video.pFrame->interlaced_frame != 0](
            dst_array, (const void **)m_Demux.video.pFrame->data,
            m_inputVideoInfo.srcWidth, m_Demux.video.pFrame->linesize[0], m_Demux.video.pFrame->linesize[1], pSurface->pitch(),
            m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);
        if (got_frame) {
            av_frame_unref(m_Demux.video.pFrame);
        }
        m_pEncSatusInfo->m_sData.frameIn++;
    } else {
        if (m_Demux.qVideoPkt.size() == 0) {
            //m_Demux.qVideoPkt.size() == 0となるのは、最後まで読み込んだときか、中断した時しかありえない
            return RGY_ERR_MORE_DATA; //ファイルの終わりに到達
        }
    }
    //進捗表示
    double progressPercent = 0.0;
    if (m_Demux.format.pFormatCtx->duration) {
        progressPercent = m_Demux.frames.duration() * (m_Demux.video.pStream->time_base.num / (double)m_Demux.video.pStream->time_base.den);
    }
    return m_pEncSatusInfo->UpdateDisplayByCurrentDuration(progressPercent);
}
#pragma warning(pop)

int RGYInputAvcodec::GetHWDecDeviceID() {
    return m_Demux.video.nHWDecodeDeviceId;
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
    if (mux) {
        m_cap2ass.setOutputResolution(w, h, sar_x, sar_y);
    } else {
        //muxしないなら処理する必要はない
        AddMessage(RGY_LOG_WARN, _T("caption2ass will be disabled as muxer is disabled.\n"));
        m_cap2ass.disable();
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

#if USE_CUSTOM_INPUT
int RGYInputAvcodec::readPacket(uint8_t *buf, int buf_size) {
    auto ret = (int)fread(buf, 1, buf_size, m_Demux.format.fpInput);
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
    if (whence != SEEK_SET || whence != SEEK_CUR || whence != SEEK_END) {
        //たまにwhence=65536とかいう要求が来ることがある
        return -1;
    }
    if (m_cap2ass.enabled() && whence == SEEK_SET && offset == 0) {
        m_cap2ass.reset();
    }
    return _fseeki64(m_Demux.format.fpInput, offset, whence);
}
#endif //USE_CUSTOM_INPUT

#endif //ENABLE_AVSW_READER

