//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <fcntl.h>
#include <io.h>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <cmath>
#include <climits>
#include <memory>
#include "avcodec_reader.h"

#ifdef LIBVA_SUPPORT
#include "hw_device.h"
#include "vaapi_device.h"
#include "vaapi_allocator.h"
#include "sample_utils.h"
#endif //#if LIBVA_SUPPORT

#if ENABLE_AVCUVID_READER
#include "NVEncCore.h"
#include "helper_cuda.h"

static inline void extend_array_size(VideoFrameData *dataset) {
    static int default_capacity = 8 * 1024;
    int current_cap = dataset->capacity;
    dataset->capacity = (current_cap) ? current_cap * 2 : default_capacity;
    dataset->frame = (FramePos *)realloc(dataset->frame, dataset->capacity * sizeof(dataset->frame[0]));
    memset(dataset->frame + current_cap, 0, sizeof(dataset->frame[0]) * (dataset->capacity - current_cap));
}

CAvcodecReader::CAvcodecReader() {
    memset(&m_Demux.format, 0, sizeof(m_Demux.format));
    memset(&m_Demux.video,  0, sizeof(m_Demux.video));
    memset(&m_sDecParam,    0, sizeof(m_sDecParam));
    m_StreamPacketsBufferL2Used = 0;
    m_strReaderName = _T("avcuvid");
}

CAvcodecReader::~CAvcodecReader() {

}

void CAvcodecReader::clearStreamPacketList(vector<AVPacket>& pktList) {
    for (uint32_t i_pkt = 0; i_pkt < pktList.size(); i_pkt++) {
        if (pktList[i_pkt].data) {
            av_free_packet(&pktList[i_pkt]);
        }
    }
    pktList.clear();
}

void CAvcodecReader::CloseFormat(AVDemuxFormat *pFormat) {
    //close video file
    if (pFormat->pFormatCtx) {
        avformat_close_input(&pFormat->pFormatCtx);
        AddMessage(NV_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
    memset(pFormat, 0, sizeof(pFormat[0]));
}

void CAvcodecReader::CloseVideo(AVDemuxVideo *pVideo) {
    //close bitstreamfilter
    if (pVideo->pH264Bsfc) {
        av_bitstream_filter_close(pVideo->pH264Bsfc);
    }
    
    if (pVideo->pExtradata) {
        av_free(pVideo->pExtradata);
    }

    memset(pVideo, 0, sizeof(pVideo[0]));
    pVideo->nIndex = -1;
}

void CAvcodecReader::CloseStream(AVDemuxStream *pStream) {
    if (pStream->pktSample.data) {
        av_free_packet(&pStream->pktSample);
    }
    memset(pStream, 0, sizeof(pStream[0]));
    pStream->nIndex = -1;
}

void CAvcodecReader::Close() {
    AddMessage(NV_LOG_DEBUG, _T("Closing...\n"));
    //リソースの解放
    for (int i = 0; i < _countof(m_StreamPacketsBufferL1); i++) {
        m_StreamPacketsBufferL1[i].clear();
    }
    if (m_StreamPacketsBufferL2Used) {
        //使用済みパケットを削除する
        //これらのパケットはすでにWriter側に渡っているか、解放されているので、av_free_packetは不要
        m_StreamPacketsBufferL2.erase(m_StreamPacketsBufferL2.begin(), m_StreamPacketsBufferL2.begin() + m_StreamPacketsBufferL2Used);
    }
    clearStreamPacketList(m_StreamPacketsBufferL2);
    m_StreamPacketsBufferL2Used = 0;
    AddMessage(NV_LOG_DEBUG, _T("Cleared Stream Packet Buffer.\n"));

    CloseFormat(&m_Demux.format);
    CloseVideo(&m_Demux.video);   AddMessage(NV_LOG_DEBUG, _T("Closed video.\n"));
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        CloseStream(&m_Demux.stream[i]);
        AddMessage(NV_LOG_DEBUG, _T("Cleared Stream #%d.\n"), i);
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

    memset(&m_sDecParam, 0, sizeof(m_sDecParam));
    AddMessage(NV_LOG_DEBUG, _T("Closed.\n"));
}

cudaVideoCodec CAvcodecReader::getCuvidcc(uint32_t id) {
    for (int i = 0; i < _countof(CUVID_DECODE_LIST); i++)
        if (CUVID_DECODE_LIST[i].codec_id == id)
            return CUVID_DECODE_LIST[i].cuvid_cc;
    return cudaVideoCodec_NumCodecs;
}

vector<int> CAvcodecReader::getStreamIndex(AVMediaType type) {
    vector<int> streams;
    const int n_streams = m_Demux.format.pFormatCtx->nb_streams;
    for (int i = 0; i < n_streams; i++) {
        if (m_Demux.format.pFormatCtx->streams[i]->codec->codec_type == type) {
            streams.push_back(i);
        }
    }
    return std::move(streams);
}

bool CAvcodecReader::vc1StartCodeExists(uint8_t *ptr) {
    uint32_t code = readUB32(ptr);
    return check_range_unsigned(code, 0x010A, 0x010F) || check_range_unsigned(code, 0x011B, 0x011F);
}

void CAvcodecReader::sortVideoPtsList() {
    //フレーム順序が確定していないところをソートする
    FramePos *ptr = m_Demux.video.frameData.frame;
    std::sort(ptr + m_Demux.video.frameData.fixed_num, ptr + m_Demux.video.frameData.num,
        [](const FramePos& posA, const FramePos& posB) {
        return ((uint32_t)std::abs(posA.pts - posB.pts) < 0xFFFFFFFF) ? posA.pts < posB.pts : posB.pts < posA.pts; });
}

void CAvcodecReader::addVideoPtsToList(FramePos pos) {
    if (m_Demux.video.frameData.capacity <= m_Demux.video.frameData.num+1) {
        std::lock_guard<std::mutex> lock(m_Demux.mtx);
        extend_array_size(&m_Demux.video.frameData);
    }
    if (pos.pts == AV_NOPTS_VALUE && 0 != (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_HALF_INVALID)) {
        //ptsがないのは音声抽出で、正常に抽出されない問題が生じる
        //半分PTSがないPAFFのような動画については、前のフレームからの補完を行う
        const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num-1];
        pos.dts = lastFrame->dts + lastFrame->duration;
        pos.pts = lastFrame->pts + lastFrame->duration;
    }
    m_Demux.video.frameData.frame[m_Demux.video.frameData.num] = pos;
    m_Demux.video.frameData.num++;

    if (m_Demux.video.frameData.fixed_num + 32 < m_Demux.video.frameData.num) {
        if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_SOMETIMES_INVALID) {
            //ptsがあてにならない時は、dtsから適当に生成する
            FramePos *frame = m_Demux.video.frameData.frame;
            const int i_fin = m_Demux.video.frameData.fixed_num + 16;
            for (int i = m_Demux.video.frameData.fixed_num + 1; i <= i_fin; i++) {
                if (frame[i].dts == AV_NOPTS_VALUE) {
                    //まずdtsがない場合は、前のフレームからコピーする
                    frame[i].dts = frame[i-1].dts;
                }
            }
            int64_t firstFramePtsDtsDiff = frame[0].pts - frame[0].dts;
            for (int i = m_Demux.video.frameData.fixed_num + 1; i <= i_fin; i++) {
                frame[i].pts = frame[i].dts + firstFramePtsDtsDiff;
            }
        }
        //m_Demux.video.frameData.fixed_numから16フレーム分、pts順にソートを行う
        sortVideoPtsList();
        //進捗表示用のdurationの計算を行う
        const FramePos *pos_fixed = m_Demux.video.frameData.frame + m_Demux.video.frameData.fixed_num;
        int64_t duration = pos_fixed[16].pts - pos_fixed[0].pts;
        if (duration < 0 || duration > 0xFFFFFFFF) {
            duration = 0;
            for (int i = 1; i < 16; i++) {
                int64_t diff = (std::max<int64_t>)(0, pos_fixed[i].pts - pos_fixed[i-1].pts);
                int64_t last_frame_dur = (std::max)(0, pos_fixed[i-1].duration);
                duration += (diff > 0xFFFFFFFF) ? last_frame_dur : diff;
            }
        }
        m_Demux.video.frameData.duration += duration;
        m_Demux.video.frameData.fixed_num += 16;
    }
}

void CAvcodecReader::hevcMp42Annexb(AVPacket *pkt) {
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

void CAvcodecReader::vc1FixHeader(int nLengthFix) {
    if (m_Demux.video.pCodecCtx->codec_id == AV_CODEC_ID_WMV3) {
        m_Demux.video.nExtradataSize += nLengthFix;
        uint32_t datasize = m_Demux.video.nExtradataSize;
        vector<uint8_t> buffer(20 + datasize, 0);
        uint32_t header = 0xC5000000;
        uint32_t width = m_Demux.video.pCodecCtx->width;
        uint32_t height = m_Demux.video.pCodecCtx->height;
        uint8_t *dataPtr = m_Demux.video.pExtradata - nLengthFix;
        memcpy(buffer.data() +  0, &header, sizeof(header));
        memcpy(buffer.data() +  4, &datasize, sizeof(datasize));
        memcpy(buffer.data() +  8, dataPtr, datasize);
        memcpy(buffer.data() +  8 + datasize, &height, sizeof(height));
        memcpy(buffer.data() + 12 + datasize, &width, sizeof(width));
        m_Demux.video.pExtradata = (uint8_t *)av_realloc(m_Demux.video.pExtradata, sizeof(buffer) + FF_INPUT_BUFFER_PADDING_SIZE);
        m_Demux.video.nExtradataSize = (int)buffer.size();
        memcpy(m_Demux.video.pExtradata, buffer.data(), buffer.size());
    } else {
        m_Demux.video.nExtradataSize += nLengthFix;
        memmove(m_Demux.video.pExtradata, m_Demux.video.pExtradata - nLengthFix, m_Demux.video.nExtradataSize);
    }
}

void CAvcodecReader::vc1AddFrameHeader(AVPacket *pkt) {
    uint32_t size = pkt->size;
    if (m_Demux.video.pCodecCtx->codec_id == AV_CODEC_ID_WMV3) {
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

int CAvcodecReader::getFirstFramePosAndFrameRate(CuvidDecode *dec, const sTrim *pTrimList, int nTrimCount) {
    AVRational fpsDecoder = m_Demux.video.pCodecCtx->framerate;
    const bool fpsDecoderInvalid = (0 == fpsDecoder.num);
    const int maxCheckFrames = (m_Demux.format.nAnalyzeSec == 0) ? 256 : 18000;
    const int maxCheckSec = (m_Demux.format.nAnalyzeSec == 0) ? INT_MAX : m_Demux.format.nAnalyzeSec;
    vector<FramePos> framePosList;
    framePosList.reserve(maxCheckFrames);
    AddMessage(NV_LOG_DEBUG, _T("fps decoder invalid: %s\n"), fpsDecoderInvalid ? _T("true") : _T("false"));

    int gotFrameCount = 0; //デコーダの出力フレーム
    int moreDataCount = 0; //出力が始まってから、デコーダが余剰にフレームを求めた回数
    bool gotFirstKeyframePos = false;
    FramePos firstKeyframePos = { 0 };
    AVPacket pkt;
    auto getTotalDuration =[&]() -> int {
        if (!gotFirstKeyframePos
            || m_Demux.video.pCodecCtx->pkt_timebase.num == 0
            || m_Demux.video.pCodecCtx->pkt_timebase.den == 0)
            return 0;
        int64_t diff = 0;
        if (pkt.dts != AV_NOPTS_VALUE && firstKeyframePos.dts != AV_NOPTS_VALUE) {
            diff = (int)(pkt.dts - firstKeyframePos.dts);
        } else if (pkt.pts != AV_NOPTS_VALUE && firstKeyframePos.pts != AV_NOPTS_VALUE) {
            diff = (int)(pkt.pts - firstKeyframePos.pts);
        }
        double timebase = m_Demux.video.pCodecCtx->pkt_timebase.num / (double)m_Demux.video.pCodecCtx->pkt_timebase.den;
        return (int)((double)diff * timebase + 0.5);
    };

    m_Demux.format.nPreReadBufferIdx = UINT_MAX; //PreReadBufferからの読み込みを禁止にする
    clearStreamPacketList(m_PreReadBuffer);
    m_PreReadBuffer.reserve(maxCheckFrames);

    m_Demux.video.nStreamFirstPts = 0;
    int i_samples = 0;
    for (; i_samples < maxCheckFrames && getTotalDuration() < maxCheckSec && !getSample(&pkt); i_samples++) {
        int64_t pts = pkt.pts, dts = pkt.dts;
        FramePos pos = { (pts == AV_NOPTS_VALUE) ? dts : pts, dts, (int)pkt.duration, pkt.flags };
        framePosList.push_back(pos);
        if (i_samples == 0 && pts != AV_NOPTS_VALUE) {
            m_Demux.video.nStreamFirstPts = pkt.pts;
        }
        if (!gotFirstKeyframePos && pkt.flags & AV_PKT_FLAG_KEY) {
            firstKeyframePos = pos;
            //キーフレームに到達するまでQSVではフレームが出てこない
            //そのぶんのずれを記録しておき、Trim値などに補正をかける
            m_sTrimParam.offset = i_samples;
            gotFirstKeyframePos = true;
        }
        ///キーフレーム取得済み
        if (gotFirstKeyframePos) {
            CUresult curesult = CUDA_SUCCESS;
            if (CUDA_SUCCESS != (curesult = dec->DecodePacket(pkt.data, pkt.size, pkt.pts, m_Demux.video.pCodecCtx->pkt_timebase))) {
                AddMessage(NV_LOG_ERROR, _T("Failed to decode packet # %d, %d (%s).\n"), i_samples, curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                return 1;
            }
            if (!dec->frameQueue()->isEmpty()) {
                CUVIDPARSERDISPINFO pInfo;
                if (dec->frameQueue()->dequeue(&pInfo)) {
                    gotFrameCount++;
                    dec->frameQueue()->releaseFrame(&pInfo);
                }
            } else {
                moreDataCount++;
            }
        }
        m_PreReadBuffer.push_back(pkt);
    }
    dec->FlushParser();

    //上記ループで最後まで行ってしまうと値が入ってしまうので初期化
    m_Demux.video.frameData.duration = 0;
    m_Demux.video.frameData.fixed_num = 0;
    m_Demux.video.frameData.num = 0;

    AddMessage(NV_LOG_DEBUG, _T("put %d frame samples for predecode.\n"), i_samples);

    if (!gotFirstKeyframePos) {
        AddMessage(NV_LOG_ERROR, _T("failed to get first frame pos.\n"));
        return 1;
    }

    //PAFFの場合、2フィールド目のpts, dtsが存在しないことがある
    uint32_t dts_pts_no_value_in_between = 0;
    uint32_t dts_pts_invalid_count = 0;
    uint32_t dts_pts_invalid_keyframe_count = 0;
    uint32_t keyframe_count = 0;
    for (uint32_t i = 0; i < framePosList.size(); i++) {
        //自分のpts/dtsがともにAV_NOPTS_VALUEで、それが前後ともにAV_NOPTS_VALUEでない場合に修正する
        if (i > 0
            && framePosList[i].dts == AV_NOPTS_VALUE
            && framePosList[i].pts == AV_NOPTS_VALUE) {
            if (   framePosList[i-1].dts != AV_NOPTS_VALUE
                && framePosList[i-1].pts != AV_NOPTS_VALUE
                && (i + 1 >= framePosList.size()
                    || (framePosList[i+1].pts != AV_NOPTS_VALUE
                     && framePosList[i+1].pts != AV_NOPTS_VALUE))) {
                framePosList[i].dts = framePosList[i-1].dts;
                framePosList[i].pts = framePosList[i-1].pts;
                dts_pts_no_value_in_between++;
            }
        }
        if (   framePosList[i].dts == AV_NOPTS_VALUE
            && framePosList[i].pts == AV_NOPTS_VALUE) {
            //自分のpts/dtsがともにAV_NOPTS_VALUEだが、規則的にはさまれているわけではなく、
            //データが無効な場合、そのままになる
            dts_pts_invalid_count++;
            dts_pts_invalid_keyframe_count += (framePosList[i].flags & 1);
        }
        keyframe_count += (framePosList[i].flags & 1);
    }

    //ほとんどのpts, dtsがあてにならない
    m_Demux.video.nStreamPtsInvalid  = (dts_pts_invalid_count >= framePosList.size() - 5 && dts_pts_invalid_keyframe_count >= (std::max)(1u, keyframe_count-2)) ? AVQSV_PTS_ALL_INVALID : 0;
    m_Demux.video.nStreamPtsInvalid |= (dts_pts_invalid_count >= (framePosList.size() - keyframe_count) - 5 && dts_pts_invalid_keyframe_count == 0)          ? AVQSV_PTS_NONKEY_INVALID : 0;

    //PAFFっぽさ (適当)
    const bool seemsLikePAFF =
        (framePosList.size() * 9 / 20 <= dts_pts_no_value_in_between)
        || (std::abs(1.0 - moreDataCount / (double)gotFrameCount) <= 0.2);
    m_Demux.video.nStreamPtsInvalid |= (seemsLikePAFF) ? AVQSV_PTS_HALF_INVALID : 0;

    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) {
        //キーフレームだけptsが得られている場合は、他のフレームのptsをdurationをもとに推定する
        for (int i = m_sTrimParam.offset + 1; i < (int)framePosList.size(); i++) {
            if (framePosList[i].dts == AV_NOPTS_VALUE && framePosList[i].pts == AV_NOPTS_VALUE) {
                framePosList[i].dts = framePosList[i-1].dts + framePosList[i-1].duration;
                framePosList[i].pts = framePosList[i-1].pts + framePosList[i-1].duration;
            }
        }
    } else if (dts_pts_invalid_count > framePosList.size() / 20) {
        //pts/dtsがともにAV_NOPTS_VALUEがある場合にはdurationの再計算を行わない
        m_Demux.video.nStreamPtsInvalid |= AVQSV_PTS_SOMETIMES_INVALID;
    } else {
        //durationを再計算する (主にmkvのくそな時間解像度への対策)
        std::sort(framePosList.begin(), framePosList.end(), [](const FramePos& posA, const FramePos& posB) { return posA.pts < posB.pts; });
        for (int i = 0; i < (int)framePosList.size() - 1; i++) {
            if (framePosList[i+1].pts != AV_NOPTS_VALUE && framePosList[i].pts != AV_NOPTS_VALUE) {
                int duration = (int)(framePosList[i+1].pts - framePosList[i].pts);
                if (duration >= 0) {
                    framePosList[i].duration = duration;
                }
            }
        }
    }
    
    AddMessage(NV_LOG_DEBUG, _T("first pts                      %I64d\n"),  m_Demux.video.nStreamFirstPts);
    AddMessage(NV_LOG_DEBUG, _T("first keyframe pts             %I64d\n"),  firstKeyframePos.pts);
    AddMessage(NV_LOG_DEBUG, _T("first key frame                %d\n"),     m_sTrimParam.offset);
    AddMessage(NV_LOG_DEBUG, _T("keyframe_count                 %d\n"),     keyframe_count);
    AddMessage(NV_LOG_DEBUG, _T("gotFrameCount                  %d\n"),     gotFrameCount);
    AddMessage(NV_LOG_DEBUG, _T("dts_pts_invalid_count          %d\n"),     dts_pts_invalid_count);
    AddMessage(NV_LOG_DEBUG, _T("dts_pts_no_value_in_between    %d\n"),     dts_pts_no_value_in_between);
    AddMessage(NV_LOG_DEBUG, _T("dts_pts_invalid_keyframe_count %d\n"),     dts_pts_invalid_keyframe_count);
    AddMessage(NV_LOG_DEBUG, _T("nStreamPtsInvalid flag         0x%02x\n"), m_Demux.video.nStreamPtsInvalid);

    //より正確なduration計算のため、最初と最後の数フレームは落とす
    //最初と最後のフレームはBフレームによりまだ並べ替えが必要な場合があり、正確なdurationを算出しない
    const int cutoff = (framePosList.size() >= 64) ? 16 : ((uint32_t)framePosList.size() / 4);
    AddMessage(NV_LOG_DEBUG, _T("cutoff                         %d\n"), cutoff);
    if (framePosList.size() >= 32) {
        vector<FramePos> newList;
        newList.reserve(framePosList.size() - cutoff - m_sTrimParam.offset);
        //最初のキーフレームからのフレームリストを構築する
        //最初のキーフレームからのリストであることは後段で、nDelayOfAudioを正確に計算するために重要
        //Bフレーム等の並べ替えを考慮し、キーフレームのpts以前のptsを持つものを削除、
        //また、最後の16フレームもBフレームを考慮してカット
        std::for_each(framePosList.begin() + m_sTrimParam.offset, framePosList.end() - cutoff,
            [&newList, firstKeyframePos](const FramePos& pos) {
            if (pos.pts == AV_NOPTS_VALUE || firstKeyframePos.pts <= pos.pts) newList.push_back(pos);
        });
        framePosList = std::move(newList);
    }

    //durationのヒストグラムを作成 (ただし、先頭は信頼ならないので、cutoff分は計算に含めない)
    vector<std::pair<int, int>> durationHistgram;
    std::for_each(framePosList.begin() + cutoff, framePosList.end(), [&durationHistgram](const FramePos& pos) {
        auto target = std::find_if(durationHistgram.begin(), durationHistgram.end(), [pos](const std::pair<int, int>& pair) { return pair.first == pos.duration; });
        if (target != durationHistgram.end()) {
            target->second++;
        } else {
            durationHistgram.push_back(std::make_pair(pos.duration, 1));
        }
    });
    //多い順にソートする
    std::sort(durationHistgram.begin(), durationHistgram.end(), [](const std::pair<int, int>& pairA, const std::pair<int, int>& pairB) { return pairA.second > pairB.second; });
    //durationが0でなく、最も頻繁に出てきたもの
    auto& mostPopularDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0];
    
    AddMessage(NV_LOG_DEBUG, _T("stream timebase %d/%d\n"), m_Demux.video.pCodecCtx->time_base.num, m_Demux.video.pCodecCtx->time_base.den);
    AddMessage(NV_LOG_DEBUG, _T("decoder fps     %d/%d\n"), fpsDecoder.num, fpsDecoder.den);
    AddMessage(NV_LOG_DEBUG, _T("duration histgram of %d frames\n"), framePosList.size() - cutoff);
    for (const auto& sample : durationHistgram) {
        AddMessage(NV_LOG_DEBUG, _T("%3d [%3d frames]\n"), sample.first, sample.second);
    }

    struct Rational64 {
        uint64_t num;
        uint64_t den;
    } estimatedAvgFps = { 0 }, nAvgFramerate64 = { 0 }, fpsDecoder64 = { (uint64_t)fpsDecoder.num, (uint64_t)fpsDecoder.den };
    if (mostPopularDuration.first == 0) {
        m_Demux.video.nStreamPtsInvalid |= AVQSV_PTS_ALL_INVALID;
    } else {
        //avgFpsとtargetFpsが近いかどうか
        auto fps_near = [](double avgFps, double targetFps) { return std::abs(1 - avgFps / targetFps) < 0.5; };
        //durationの平均を求める (ただし、先頭は信頼ならないので、cutoff分は計算に含めない)
        //std::accumulateの初期値に"(uint64_t)0"と与えることで、64bitによる計算を実行させ、桁あふれを防ぐ
        //大きすぎるtimebaseの時に必要
        double avgDuration = std::accumulate(framePosList.begin() + cutoff, framePosList.end(), (uint64_t)0, [](const uint64_t sum, const FramePos& pos) { return sum + pos.duration; }) / (double)(framePosList.size() - cutoff);
        double avgFps = m_Demux.video.pCodecCtx->pkt_timebase.den / (double)(avgDuration * m_Demux.video.pCodecCtx->time_base.num);
        double torrelance = (fps_near(avgFps, 25.0) || fps_near(avgFps, 50.0)) ? 0.01 : 0.0008; //25fps, 50fps近辺は基準が甘くてよい
        if (mostPopularDuration.second / (double)(framePosList.size() - cutoff) > 0.95 && std::abs(1 - mostPopularDuration.first / avgDuration) < torrelance) {
            avgDuration = mostPopularDuration.first;
            AddMessage(NV_LOG_DEBUG, _T("using popular duration...\n"));
        }
        //入力フレームに対し、出力フレームが半分程度なら、フレームのdurationを倍と見積もる
        avgDuration *= (seemsLikePAFF) ? 2.0 : 1.0;
        //durationから求めた平均fpsを計算する
        const uint64_t mul = (uint64_t)ceil(1001.0 / m_Demux.video.pCodecCtx->time_base.num);
        estimatedAvgFps.num = (uint64_t)(m_Demux.video.pCodecCtx->pkt_timebase.den / avgDuration * (double)m_Demux.video.pCodecCtx->time_base.num * mul + 0.5);
        estimatedAvgFps.den = (uint64_t)m_Demux.video.pCodecCtx->time_base.num * mul;
        
        AddMessage(NV_LOG_DEBUG, _T("fps mul:         %d\n"),    mul);
        AddMessage(NV_LOG_DEBUG, _T("raw avgDuration: %lf\n"),   avgDuration);
        AddMessage(NV_LOG_DEBUG, _T("estimatedAvgFps: %I64u/%I64u\n"), estimatedAvgFps.num, estimatedAvgFps.den);
    }

    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
        //ptsとdurationをpkt_timebaseで適当に作成する
        addVideoPtsToList({ 0, 0, (int)av_rescale_q(1, m_Demux.video.pCodecCtx->time_base, m_Demux.video.pCodecCtx->pkt_timebase), firstKeyframePos.flags });
        nAvgFramerate64 = (fpsDecoderInvalid) ? estimatedAvgFps : fpsDecoder64;
    } else {
        addVideoPtsToList(firstKeyframePos);
        if (fpsDecoderInvalid) {
            nAvgFramerate64 = estimatedAvgFps;
        } else {
            double dFpsDecoder = fpsDecoder.num / (double)fpsDecoder.den;
            double dEstimatedAvgFps = estimatedAvgFps.num / (double)estimatedAvgFps.den;
            //2フレーム分程度がもたらす誤差があっても許容する
            if (std::abs(dFpsDecoder / dEstimatedAvgFps - 1.0) < (2.0 / framePosList.size())) {
                AddMessage(NV_LOG_DEBUG, _T("use decoder fps...\n"));
                nAvgFramerate64 = fpsDecoder64;
            } else {
                double dEstimatedAvgFpsCompare = estimatedAvgFps.num / (double)(estimatedAvgFps.den + ((dFpsDecoder < dEstimatedAvgFps) ? 1 : -1));
                //durationから求めた平均fpsがデコーダの出したfpsの近似値と分かれば、デコーダの出したfpsを採用する
                nAvgFramerate64 = (std::abs(dEstimatedAvgFps - dFpsDecoder) < std::abs(dEstimatedAvgFpsCompare - dFpsDecoder)) ? fpsDecoder64 : estimatedAvgFps;
            }
        }
    }
    AddMessage(NV_LOG_DEBUG, _T("final AvgFps (raw64): %I64u/%I64u\n"), estimatedAvgFps.num, estimatedAvgFps.den);

    const uint64_t fps_gcd = nv_get_gcd(nAvgFramerate64.num, nAvgFramerate64.den);
    nAvgFramerate64.num /= fps_gcd;
    nAvgFramerate64.den /= fps_gcd;
    m_Demux.video.nAvgFramerate = av_make_q((int)nAvgFramerate64.num, (int)nAvgFramerate64.den);
    AddMessage(NV_LOG_DEBUG, _T("final AvgFps (gcd): %d/%d\n"), m_Demux.video.nAvgFramerate.num, m_Demux.video.nAvgFramerate.den);

    //近似値であれば、分母1001/分母1に合わせる
    double fps = m_Demux.video.nAvgFramerate.num / (double)m_Demux.video.nAvgFramerate.den;
    double fps_n = fps * 1001;
    int fps_n_int = (int)(fps + 0.5) * 1000;
    if (std::abs(fps_n / (double)fps_n_int - 1.0) < 1e-4) {
        m_Demux.video.nAvgFramerate.num = fps_n_int;
        m_Demux.video.nAvgFramerate.den = 1001;
    } else {
        fps_n = fps * 1000;
        fps_n_int = (int)(fps + 0.5) * 1000;
        if (std::abs(fps_n / (double)fps_n_int - 1.0) < 1e-4) {
            m_Demux.video.nAvgFramerate.num = fps_n_int / 1000;
            m_Demux.video.nAvgFramerate.den = 1;
        }
    }
    AddMessage(NV_LOG_DEBUG, _T("final AvgFps (round): %d/%d\n\n"), m_Demux.video.nAvgFramerate.num, m_Demux.video.nAvgFramerate.den);

    auto trimList = vector<sTrim>(pTrimList, pTrimList + nTrimCount);
    //出力時の音声・字幕解析用に1パケットコピーしておく
    auto& streamBuffer = m_StreamPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)];
    if (streamBuffer.size()) {
        for (auto streamInfo = m_Demux.stream.begin(); streamInfo != m_Demux.stream.end(); streamInfo++) {
            if (avcodec_get_type(streamInfo->pCodecCtx->codec_id) == AVMEDIA_TYPE_AUDIO) {
                AddMessage(NV_LOG_DEBUG, _T("checking for stream #%d\n"), streamInfo->nIndex);
                const AVPacket *pkt1 = NULL; //最初のパケット
                const AVPacket *pkt2 = NULL; //2番目のパケット
                for (int j = 0; j < (int)streamBuffer.size(); j++) {
                    if (streamBuffer[j].stream_index == streamInfo->nIndex) {
                        if (pkt1) {
                            pkt2 = &streamBuffer[j];
                            break;
                        }
                        pkt1 = &streamBuffer[j];
                    }
                }
                if (pkt1 != NULL) {
                    //1パケット目はたまにおかしいので、可能なら2パケット目を使用する
                    av_copy_packet(&streamInfo->pktSample, (pkt2) ? pkt2 : pkt1);
                    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
                        streamInfo->nDelayOfStream = 0;
                    } else {
                        //その音声の属する動画フレーム番号
                        const int vidIndex = getVideoFrameIdx(pkt1->pts, streamInfo->pCodecCtx->pkt_timebase, framePosList.data(), (int)framePosList.size(), 0);
                        AddMessage(NV_LOG_DEBUG, _T("audio track %d first pts: %I64d\n"), streamInfo->nTrackId, pkt1->pts);
                        AddMessage(NV_LOG_DEBUG, _T("      first pts videoIdx: %d\n"), vidIndex);
                        if (vidIndex >= 0) {
                            //音声の遅れているフレーム数分のdurationを足し上げる
                            int delayOfStream = (frame_inside_range(vidIndex, trimList)) ? (int)(pkt1->pts - framePosList[vidIndex].pts) : 0;
                            for (int iFrame = m_sTrimParam.offset; iFrame < vidIndex; iFrame++) {
                                if (frame_inside_range(iFrame, trimList)) {
                                    delayOfStream += framePosList[iFrame].duration;
                                }
                            }
                            streamInfo->nDelayOfStream = delayOfStream;
                            AddMessage(NV_LOG_DEBUG, _T("audio track %d delay: %d (timebase=%d/%d)\n"),
                                streamInfo->nIndex, streamInfo->nTrackId,
                                streamInfo->nDelayOfStream, streamInfo->pCodecCtx->pkt_timebase.num, streamInfo->pCodecCtx->pkt_timebase.den);
                        }
                    }
                } else {
                    //音声の最初のサンプルを取得できていない
                    streamInfo = m_Demux.stream.erase(streamInfo) - 1;
                    AddMessage(NV_LOG_WARN, _T("failed to find stream #%d in preread.\n"), streamInfo->nIndex);
                }
            }
        }
        if (streamBuffer.size() == 0) {
            //音声・字幕の最初のサンプルを取得できていないため、音声がすべてなくなってしまった
            AddMessage(NV_LOG_ERROR, _T("failed to find audio/subtitle stream in preread.\n"));
            return 1;
        }
    }
    //音声・字幕パケットもm_PreReadBufferに接続する
    //ここで、必ず音声パケットを先頭に挿入し、音声パケット->映像パケットになるようにする
    //そうしないと、音声が相当遅れてくることになり、映像がブロック状に割れるなどの支障をきたす
    m_PreReadBuffer.insert(m_PreReadBuffer.begin(), streamBuffer.begin(), streamBuffer.end());
    streamBuffer.clear();
    m_Demux.format.nPreReadBufferIdx = 0; //PreReadBufferからの読み込み優先にする

    return 0;
}

#pragma warning(push)
#pragma warning(disable:4100)
int CAvcodecReader::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {

    Close();

    m_pEncSatusInfo = pStatus;
    const AvcodecReaderPrm *input_prm = (const AvcodecReaderPrm *)inputPrm->otherPrm;

    if (!check_avcodec_dll()) {
        AddMessage(NV_LOG_ERROR, error_mes_avcodec_dll_not_found());
        return 1;
    }

    for (int i = 0; i < input_prm->nAudioSelectCount; i++) {
        AddMessage(NV_LOG_DEBUG, _T("select audio track %s, codec %s, format %s, bitrate %d, filename \"%s\"\n"),
            (input_prm->ppAudioSelect[i]->nAudioSelect) ? strsprintf(_T("#%d"), input_prm->ppAudioSelect[i]->nAudioSelect).c_str() : _T("all"),
            input_prm->ppAudioSelect[i]->pAVAudioEncodeCodec, input_prm->ppAudioSelect[i]->pAudioExtractFormat,
            input_prm->ppAudioSelect[i]->nAVAudioEncodeBitrate, input_prm->ppAudioSelect[i]->pAudioExtractFilename);
    }

    av_register_all();
    avcodec_register_all();
    av_log_set_level((m_pPrintMes->getLogLevel() == NV_LOG_DEBUG) ?  AV_LOG_DEBUG : NV_AV_LOG_LEVEL);

    int ret = 0;
    std::string filename_char;
    if (0 == tchar_to_string(inputPrm->filename, filename_char, CP_UTF8)) {
        AddMessage(NV_LOG_ERROR, _T("failed to convert filename to utf-8 characters.\n"));
        return 1;
    }
    m_Demux.format.bIsPipe = (0 == strcmp(filename_char.c_str(), "-")) || filename_char.c_str() == strstr(filename_char.c_str(), R"(\\.\pipe\)");
    m_Demux.format.pFormatCtx = avformat_alloc_context();
    m_Demux.format.nAnalyzeSec = input_prm->nAnalyzeSec;
    if (m_Demux.format.nAnalyzeSec) {
        if (0 != (ret = av_opt_set_int(m_Demux.format.pFormatCtx, "probesize", m_Demux.format.nAnalyzeSec * AV_TIME_BASE, 0))) {
            AddMessage(NV_LOG_ERROR, _T("failed to set probesize to %d sec: error %d\n"), m_Demux.format.nAnalyzeSec, ret);
        } else {
            AddMessage(NV_LOG_DEBUG, _T("set probesize: %d sec\n"), m_Demux.format.nAnalyzeSec);
        }
    }
    if (0 == strcmp(filename_char.c_str(), "-")) {
#if defined(_WIN32) || defined(_WIN64)
        if (_setmode(_fileno(stdin), _O_BINARY) < 0) {
            AddMessage(NV_LOG_ERROR, _T("failed to switch stdin to binary mode.\n"));
            return 1;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        AddMessage(NV_LOG_DEBUG, _T("input source set to stdin.\n"));
        filename_char = "pipe:0";
    }
    if (avformat_open_input(&(m_Demux.format.pFormatCtx), filename_char.c_str(), nullptr, nullptr)) {
        AddMessage(NV_LOG_ERROR, _T("error opening file: \"%s\"\n"), char_to_tstring(filename_char, CP_UTF8).c_str());
        return 1; // Couldn't open file
    }
    AddMessage(NV_LOG_DEBUG, _T("opened file \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());

    if (m_Demux.format.nAnalyzeSec) {
        if (0 != (ret = av_opt_set_int(m_Demux.format.pFormatCtx, "analyzeduration", m_Demux.format.nAnalyzeSec * AV_TIME_BASE, 0))) {
            AddMessage(NV_LOG_ERROR, _T("failed to set analyzeduration to %d sec, error %d\n"), m_Demux.format.nAnalyzeSec, ret);
        } else {
            AddMessage(NV_LOG_DEBUG, _T("set analyzeduration: %d sec\n"), m_Demux.format.nAnalyzeSec);
        }
    }
    if (avformat_find_stream_info(m_Demux.format.pFormatCtx, nullptr) < 0) {
        AddMessage(NV_LOG_ERROR, _T("error finding stream information.\n"));
        return 1; // Couldn't find stream information
    }
    AddMessage(NV_LOG_DEBUG, _T("got stream information.\n"));
    av_dump_format(m_Demux.format.pFormatCtx, 0, filename_char.c_str(), 0);
    //dump_format(dec.pFormatCtx, 0, argv[1], 0);

    //音声ストリームを探す
    if (input_prm->nReadAudio || input_prm->bReadSubtitle) {
        vector<int> mediaStreams;
        if (input_prm->nReadAudio) {
            auto audioStreams = getStreamIndex(AVMEDIA_TYPE_AUDIO);
            if (audioStreams.size() == 0) {
                AddMessage(NV_LOG_ERROR, _T("--audio-encode/--audio-copy/--audio-file is set, but no audio stream found.\n"));
                return 1;
            }
            m_Demux.format.nAudioTracks = (int)audioStreams.size();
            vector_cat(mediaStreams, audioStreams);
        }
        if (input_prm->bReadSubtitle) {
            auto subStreams = getStreamIndex(AVMEDIA_TYPE_SUBTITLE);
            if (subStreams.size() == 0) {
                AddMessage(NV_LOG_ERROR, _T("--sub-copy is set, but no subtitle stream found.\n"));
                return 1;
            }
            m_Demux.format.nSubtitleTracks = (int)subStreams.size();
            vector_cat(mediaStreams, subStreams);
        }
        for (int iTrack = 0; iTrack < (int)mediaStreams.size(); iTrack++) {
            const AVCodecID codecId = m_Demux.format.pFormatCtx->streams[mediaStreams[iTrack]]->codec->codec_id;
            bool useStream = false;
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
                    if (input_prm->ppAudioSelect[i]->nAudioSelect == 0 //特に指定なし = 全指定かどうか
                        || input_prm->ppAudioSelect[i]->nAudioSelect == (iTrack + input_prm->nAudioTrackStart)) {
                        useStream = true;
                    }
                }
            }
            if (useStream) {
                AVDemuxStream stream = { 0 };
                stream.nTrackId = (AVMEDIA_TYPE_SUBTITLE == avcodec_get_type(codecId))
                    ? -(iTrack - m_Demux.format.nAudioTracks + 1 + input_prm->nSubtitleTrackStart) //字幕は -1, -2, -3
                    : iTrack + input_prm->nAudioTrackStart; //音声は1, 2, 3
                stream.nIndex = mediaStreams[iTrack];
                stream.pCodecCtx = m_Demux.format.pFormatCtx->streams[stream.nIndex]->codec;
                stream.pStream = m_Demux.format.pFormatCtx->streams[stream.nIndex];
                m_Demux.stream.push_back(stream);
                AddMessage(NV_LOG_DEBUG, _T("found %s stream, stream idx %d, trackID %d, %s, frame_size %d, timebase %d/%d\n"),
                    get_media_type_string(codecId).c_str(),
                    stream.nIndex, stream.nTrackId, char_to_tstring(avcodec_get_name(codecId)).c_str(),
                    stream.pCodecCtx->frame_size, stream.pCodecCtx->pkt_timebase.num, stream.pCodecCtx->pkt_timebase.den);
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
                    AddMessage(NV_LOG_ERROR, _T("could not find audio track #%d\n"), input_prm->ppAudioSelect[i]->nAudioSelect);
                    return 1;
                }
            }
        }
    }

    if (input_prm->bReadChapter) {
        m_Demux.chapter = make_vector((const AVChapter **)m_Demux.format.pFormatCtx->chapters, m_Demux.format.pFormatCtx->nb_chapters);
    }

    //動画ストリームを探す
    if (input_prm->bReadVideo) {
        auto videoStreams = getStreamIndex(AVMEDIA_TYPE_VIDEO);
        if (videoStreams.size() == 0) {
            AddMessage(NV_LOG_ERROR, _T("unable to find video stream.\n"));
            return 1; // Didn't find a video stream
        }
        m_Demux.video.nIndex = videoStreams[0];
        AddMessage(NV_LOG_DEBUG, _T("found video stream, stream idx %d\n"), m_Demux.video.nIndex);

        m_Demux.video.pCodecCtx = m_Demux.format.pFormatCtx->streams[m_Demux.video.nIndex]->codec;

        //cuvidでデコード可能かチェック
        if (cudaVideoCodec_NumCodecs == (m_sDecParam.codec = getCuvidcc(m_Demux.video.pCodecCtx->codec_id))) {
            AddMessage(NV_LOG_ERROR, _T("codec "));
            if (m_Demux.video.pCodecCtx->codec && m_Demux.video.pCodecCtx->codec->name) {
                AddMessage(NV_LOG_ERROR, char_to_tstring(m_Demux.video.pCodecCtx->codec->name) + _T(" "));
            }
            AddMessage(NV_LOG_ERROR, _T("unable to decode by cuvid.\n"));
            return 1;
        }
        AddMessage(NV_LOG_DEBUG, _T("can be decoded by cuvid.\n"));
        //wmv3はAdvanced Profile (3)のみの対応
        if (m_Demux.video.pCodecCtx->codec_id == AV_CODEC_ID_WMV3 && m_Demux.video.pCodecCtx->profile != 3) {
            AddMessage(NV_LOG_ERROR, _T("unable to decode by cuvid.\n"));
            return 1;
        }

        //情報を格納
        inputPrm->codec       = m_sDecParam.codec;
        inputPrm->width       = m_Demux.video.pCodecCtx->width;
        inputPrm->height      = m_Demux.video.pCodecCtx->height;
        inputPrm->codedWidth  = m_Demux.video.pCodecCtx->coded_width;
        inputPrm->codedHeight = m_Demux.video.pCodecCtx->coded_height;
        inputPrm->sar[0]      = m_Demux.video.pCodecCtx->sample_aspect_ratio.num;
        inputPrm->sar[1]      = m_Demux.video.pCodecCtx->sample_aspect_ratio.den;

        //必要ならbitstream filterを初期化
        if (m_Demux.video.pCodecCtx->extradata && m_Demux.video.pCodecCtx->extradata[0] == 1) {
            if (m_sDecParam.codec == cudaVideoCodec_H264) {
                if (NULL == (m_Demux.video.pH264Bsfc = av_bitstream_filter_init("h264_mp4toannexb"))) {
                    AddMessage(NV_LOG_ERROR, _T("failed to init h264_mp4toannexb.\n"));
                    return 1;
                }
                AddMessage(NV_LOG_DEBUG, _T("initialized h264_mp4toannexb filter.\n"));
            } else if (m_sDecParam.codec == cudaVideoCodec_HEVC) {
                m_Demux.video.bUseHEVCmp42AnnexB = true;
                AddMessage(NV_LOG_DEBUG, _T("enabled HEVCmp42AnnexB filter.\n"));
            }
        } else if (m_Demux.video.pCodecCtx->extradata == NULL && m_Demux.video.pCodecCtx->extradata_size == 0) {
            AddMessage(NV_LOG_ERROR, _T("video header not extracted by libavcodec.\n"));
            return 1;
        }
        if (m_Demux.video.pCodecCtx->extradata_size) {
            inputPrm->codecExtra = m_Demux.video.pCodecCtx->extradata;
            inputPrm->codecExtraSize = m_Demux.video.pCodecCtx->extradata_size;
        }

        AddMessage(NV_LOG_DEBUG, _T("start predecode.\n"));

        int sts = 0;
        vector<uint8_t> bitstream = { 0 };
        if (0 != (sts = GetHeader(bitstream))) {
            AddMessage(NV_LOG_ERROR, _T("failed to get header.\n"));
            return sts;
        }

        if (m_sDecParam.codec == cudaVideoCodec_H264 || m_sDecParam.codec == cudaVideoCodec_HEVC) {
            //これを付加しないとMFXVideoDECODE_DecodeHeaderが成功しない
            const uint32_t IDR = 0x65010000;
            vector_cat(bitstream, (uint8_t *)&IDR, sizeof(IDR));
        }

        ////// デコード開始
        CUresult cuResult = CUDA_SUCCESS;
        CuvidDecode cuvidDec;
        if (CUDA_SUCCESS != (cuResult = cuvidDec.InitDecode(input_prm->ctxLock, inputPrm, m_pPrintMes, true))) {
            AddMessage(NV_LOG_ERROR, _T("Failed to init deocder.\n"));
            sts = 1;
        } else if (0 != (sts = getFirstFramePosAndFrameRate(&cuvidDec, input_prm->pTrimList, input_prm->nTrimCount))) {
            AddMessage(NV_LOG_ERROR, _T("failed to get first frame position.\n"));
        }
        auto decInfo = cuvidDec.GetDecodeInfo();
        inputPrm->codec       = m_sDecParam.codec;
        inputPrm->codedWidth  = decInfo.ulWidth;
        inputPrm->codedHeight = decInfo.ulHeight;
        cuvidDec.CloseDecoder();
        bitstream.clear();
        if (0 != sts) {
            AddMessage(NV_LOG_ERROR, _T("unable to decode by cuvid, please consider using other input method.\n"));
            return sts;
        }
        AddMessage(NV_LOG_DEBUG, _T("predecode success.\n"));

        m_sTrimParam.list = vector<sTrim>(input_prm->pTrimList, input_prm->pTrimList + input_prm->nTrimCount);
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
            AddMessage(NV_LOG_DEBUG, _T("adjust trim by offset %d.\n"), m_sTrimParam.offset);
        }

        //getFirstFramePosAndFrameRateをもとにfpsを決定
        inputPrm->scale = m_Demux.video.nAvgFramerate.den;
        inputPrm->rate = m_Demux.video.nAvgFramerate.num;

        memcpy(&m_sDecParam, inputPrm, sizeof(m_sDecParam));
        m_sDecParam.src_pitch = 0;

        tstring mes = strsprintf(_T("avcodec video: %s, %dx%d, %d/%d fps"), CodecIdToStr(inputPrm->codec).c_str(),
            inputPrm->width, inputPrm->height, inputPrm->rate, inputPrm->scale);
        AddMessage(NV_LOG_DEBUG, mes);
        m_strInputInfo += mes;
    } else {
        m_Demux.video.nAvgFramerate = av_make_q(input_prm->nVideoAvgFramerate.first, input_prm->nVideoAvgFramerate.second);

        tstring mes;
        for (const auto& stream : m_Demux.stream) {
            if (mes.length()) mes += _T(", ");
            tstring codec_name = char_to_tstring(avcodec_get_name(stream.pCodecCtx->codec_id));
            mes += codec_name;
            AddMessage(NV_LOG_DEBUG, _T("avcodec %s: %s from %s\n"),
                get_media_type_string(stream.pCodecCtx->codec_id).c_str(), codec_name.c_str(), inputPrm->filename);
        }
        m_strInputInfo += _T("avcodec audio: ") + mes;
    }

    m_tmLastUpdate = std::chrono::system_clock::now();
    return 0;
}
#pragma warning(pop)

vector<const AVChapter *> CAvcodecReader::GetChapterList() {
    return m_Demux.chapter;
}

//int CAvcodecReader::GetSubtitleTrackCount() {
//    return m_Demux.format.nSubtitleTracks;
//}
//
//int CAvcodecReader::GetAudioTrackCount() {
//    return m_Demux.format.nAudioTracks;
//}

int64_t CAvcodecReader::GetVideoFirstPts() {
    return m_Demux.video.nStreamFirstPts;
}

int CAvcodecReader::getVideoFrameIdx(int64_t pts, AVRational timebase, const FramePos *framePos, int framePosCount, int iStart) {
    for (int i = (std::max)(0, iStart); i < framePosCount; i++) {
        //pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
        if (0 > av_compare_ts(pts, timebase, framePos[i].pts, m_Demux.video.pCodecCtx->pkt_timebase)) {
            return i - 1;
        }
    }
    return framePosCount;
}

int64_t CAvcodecReader::convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *pStream) {
    return av_rescale_q(pts, m_Demux.video.pCodecCtx->pkt_timebase, pStream->pCodecCtx->pkt_timebase);
}

bool CAvcodecReader::checkStreamPacketToAdd(const AVPacket *pkt, AVDemuxStream *pStream) {
    pStream->nLastVidIndex = getVideoFrameIdx(pkt->pts, pStream->pCodecCtx->pkt_timebase, m_Demux.video.frameData.frame, m_Demux.video.frameData.num, pStream->nLastVidIndex);

    //該当フレームが-1フレーム未満なら、その音声はこの動画には含まれない
    if (pStream->nLastVidIndex < -1) {
        return false;
    }

    const FramePos *vidFramePos = &m_Demux.video.frameData.frame[(std::max)(pStream->nLastVidIndex, 0)];
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

AVDemuxStream *CAvcodecReader::getPacketStreamData(const AVPacket *pkt) {
    int streamIndex = pkt->stream_index;
    for (int i = 0; i < (int)m_Demux.stream.size(); i++) {
        if (m_Demux.stream[i].nIndex == streamIndex) {
            return &m_Demux.stream[i];
        }
    }
    return NULL;
}

int CAvcodecReader::getSample(AVPacket *pkt) {
    av_init_packet(pkt);
    auto get_sample = [this](AVPacket *pkt, bool *fromPreReadBuffer) {
        if (m_Demux.format.nPreReadBufferIdx < m_PreReadBuffer.size()) {
            *pkt = m_PreReadBuffer[m_Demux.format.nPreReadBufferIdx];
            m_Demux.format.nPreReadBufferIdx++;
            if (m_Demux.format.nPreReadBufferIdx >= m_PreReadBuffer.size()) {
                m_PreReadBuffer.clear();
                m_Demux.format.nPreReadBufferIdx = UINT_MAX;
            }
            *fromPreReadBuffer = true;
            return 0;
        } else {
            return av_read_frame(m_Demux.format.pFormatCtx, pkt);
        }
    };
    bool fromPreReadBuffer = false;
    while (get_sample(pkt, &fromPreReadBuffer) >= 0) {
        if (pkt->stream_index == m_Demux.video.nIndex) {
            if (!fromPreReadBuffer) {
                if (m_Demux.video.pH264Bsfc) {
                    uint8_t *data = NULL;
                    int dataSize = 0;
                    std::swap(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata);
                    std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
                    av_bitstream_filter_filter(m_Demux.video.pH264Bsfc, m_Demux.video.pCodecCtx, nullptr,
                        &data, &dataSize, pkt->data, pkt->size, 0);
                    std::swap(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata);
                    std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
                    av_free_packet(pkt); //メモリ解放を忘れない
                    av_packet_from_data(pkt, data, dataSize);
                }
                if (m_Demux.video.bUseHEVCmp42AnnexB) {
                    hevcMp42Annexb(pkt);
                }
                if (m_sDecParam.codec == cudaVideoCodec_VC1) {
                    vc1AddFrameHeader(pkt);
                }
            }
            //最初のptsが格納されていたら( = getFirstFramePosAndFrameRate()が実行済み)、後続のptsを格納していく
            if (m_Demux.video.frameData.num) {
                //最初のキーフレームを取得するまではスキップする
                if (!m_Demux.video.bGotFirstKeyframe && !(pkt->flags & AV_PKT_FLAG_KEY)) {
                    av_free_packet(pkt);
                    continue;
                } else {
                    m_Demux.video.bGotFirstKeyframe = true;
                    //AVPacketのもたらすptsが無効であれば、CFRを仮定して適当にptsとdurationを突っ込んでいく
                    //0フレーム目は格納されているので、その次からを格納する
                    if ((m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) && m_Demux.video.nSampleLoadCount) {
                        int duration = m_Demux.video.frameData.frame[0].duration;
                        int64_t pts = m_Demux.video.nSampleLoadCount * duration;
                        addVideoPtsToList({ pts, pts, duration, pkt->flags });
                        //最初のptsは格納されているので、その次からを格納する
                    } else {
                        int64_t pts = pkt->pts, dts = pkt->dts;
                        if ((m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) && pts == AV_NOPTS_VALUE) {
                            //キーフレーム以外のptsとdtsが無効な場合は、適当に推定する
                            const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num-1];
                            int duration = lastFrame->duration;
                            pts = lastFrame->pts + duration;
                            dts = lastFrame->dts + duration;
                        }
                        addVideoPtsToList({ (pts == AV_NOPTS_VALUE) ? dts : pts, dts, (int)pkt->duration, pkt->flags });
                    }
                }
            }
            return 0;
        }
        if (getPacketStreamData(pkt) != NULL) {
            //音声/字幕パケットはひとまずすべてバッファに格納する
            m_StreamPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)].push_back(*pkt);
        } else {
            av_free_packet(pkt);
        }
    }
    //ファイルの終わりに到達
    pkt->data = nullptr;
    pkt->size = 0;
    sortVideoPtsList();
    //動画の終端を表す最後のptsを挿入する
    int64_t videoFinPts = 0;
    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
        videoFinPts = m_Demux.video.nSampleLoadCount * m_Demux.video.frameData.frame[0].duration;
    } else if (m_Demux.video.frameData.num) {
        const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num - 1];
        videoFinPts = lastFrame->pts + lastFrame->duration;
    }
    //もし選択範囲が手動で決定されていないのなら、音声を最大限取得する
    if (m_sTrimParam.list.size() == 0 || m_sTrimParam.list.back().fin == TRIM_MAX) {
        if (m_StreamPacketsBufferL2.size()) {
            videoFinPts = (std::max)(videoFinPts, m_StreamPacketsBufferL2.back().pts + m_StreamPacketsBufferL2.back().duration);
        }
        for (auto packetsL1 : m_StreamPacketsBufferL1) {
            if (packetsL1.size()) {
                videoFinPts = (std::max)(videoFinPts, packetsL1.back().pts + packetsL1.back().duration);
            }
        }
    }
    addVideoPtsToList({ videoFinPts, videoFinPts, 0 });
    m_Demux.video.frameData.fixed_num = m_Demux.video.frameData.num - 1;
    m_Demux.video.frameData.duration = m_Demux.format.pFormatCtx->duration;
    m_pEncSatusInfo->UpdateDisplay(std::chrono::system_clock::now(), 100.0);
    return 1;
}

int CAvcodecReader::setToMfxBitstream(vector<uint8_t>& bitstream, AVPacket *pkt) {
    if (pkt->data && pkt->size) {
        bitstream = make_vector(pkt->data, pkt->size);
    } else {
        bitstream = vector<uint8_t>();
    }
    return 0;
}

int CAvcodecReader::GetNextBitstream(vector<uint8_t>& bitstream, int64_t *pts) {
    AVPacket *pkt = &m_Demux.video.packet[m_Demux.video.nSampleGetCount % _countof(m_Demux.video.packet)];
    int sts = setToMfxBitstream(bitstream, pkt);
    if (pts) {
        *pts = pkt->pts;
    }
    m_Demux.video.nSampleGetCount++;
    return sts;
}

vector<AVPacket> CAvcodecReader::GetAudioDataPacketsWhenNoVideoRead() {
    //音声の読み込みのみ行う場合は、LoadNextFrameが呼ばれないため、
    //音声データがm_StreamPacketsBufferL1に格納されていない
    //そこでここでデータを取得し、これを返すようにする
    vector<AVPacket> packets;

    m_Demux.video.nSampleGetCount++;

    const double vidEstDurationSec = m_Demux.video.nSampleGetCount * (double)m_Demux.video.nAvgFramerate.den / (double)m_Demux.video.nAvgFramerate.num; //1フレームの時間(秒)

    //およそ1フレーム分のパケットを取得する
    AVPacket pkt;
    av_init_packet(&pkt);
    while (av_read_frame(m_Demux.format.pFormatCtx, &pkt) >= 0) {
        AVStream *pStream = m_Demux.format.pFormatCtx->streams[pkt.stream_index];
        if (pStream->codec->codec_type != AVMEDIA_TYPE_AUDIO) {
            av_free_packet(&pkt);
        } else {
            AVDemuxStream *pAudio = getPacketStreamData(&pkt);
            pkt.flags = (pkt.flags & 0xffff) | (pAudio->nTrackId << 16);
            packets.push_back(pkt); //音声ストリームなら、すべて格納してしまう

            //最初のパケットは参照用にコピーしておく
            if (pAudio->pktSample.data == nullptr) {
                av_copy_packet(&pAudio->pktSample, &pkt);
            }
            uint64_t pktt = (pkt.pts == AV_NOPTS_VALUE) ? pkt.dts : pkt.pts;
            uint64_t pkt_dist = pktt - pAudio->pktSample.pts;
            //1フレーム分のサンプルを取得したら終了
            if (pkt_dist * (double)pAudio->pCodecCtx->pkt_timebase.num / (double)pAudio->pCodecCtx->pkt_timebase.den > vidEstDurationSec) {
                break;
            }
        }
    }
    return std::move(packets);
}

const AVDictionary *CAvcodecReader::GetInputFormatMetadata() {
    return m_Demux.format.pFormatCtx->metadata;
}

const AVCodecContext *CAvcodecReader::GetInputVideoCodecCtx() {
    return m_Demux.video.pCodecCtx;
}

vector<AVPacket> CAvcodecReader::GetStreamDataPackets() {
    if (m_Demux.video.pCodecCtx == nullptr) {
        return GetAudioDataPacketsWhenNoVideoRead();
    }
    //すでに使用した音声バッファはクリアする
    if (m_StreamPacketsBufferL2Used) {
        //使用済みパケットを削除する
        //これらのパケットはすでにWriter側に渡っているか、解放されているので、av_free_packetは不要
        m_StreamPacketsBufferL2.erase(m_StreamPacketsBufferL2.begin(), m_StreamPacketsBufferL2.begin() + m_StreamPacketsBufferL2Used);
    }
    m_StreamPacketsBufferL2Used = 0;

    //別スレッドで使用されていないほうを連結する
    const auto& packetsL1 = m_StreamPacketsBufferL1[m_Demux.video.nSampleGetCount % _countof(m_StreamPacketsBufferL1)];
    vector_cat(m_StreamPacketsBufferL2, packetsL1);

    //出力するパケットを選択する
    vector<AVPacket> packets;
    {
        std::lock_guard<std::mutex> lock(m_Demux.mtx);
        for (uint32_t i = 0; i < m_StreamPacketsBufferL2.size(); i++) {
            AVPacket *pkt = &m_StreamPacketsBufferL2[i];
            AVDemuxStream *pStream = getPacketStreamData(pkt);
            //音声のptsが映像の終わりのptsを行きすぎたらやめる
            if (0 < av_compare_ts(pkt->pts, pStream->pCodecCtx->pkt_timebase, m_Demux.video.frameData.frame[m_Demux.video.frameData.fixed_num].pts, m_Demux.video.pCodecCtx->pkt_timebase)) {
                break;
            }
            m_StreamPacketsBufferL2Used++;
            if (checkStreamPacketToAdd(pkt, pStream)) {
                pkt->flags = (pkt->flags & 0xffff) | (pStream->nTrackId << 16); //flagsの上位16bitには、trackIdへのポインタを格納しておく
                packets.push_back(*pkt); //Writer側に渡したパケットはWriter側で開放する
            } else {
                av_free_packet(pkt); //Writer側に渡さないパケットはここで開放する
                pkt->data = NULL;
                pkt->size = 0;
            }
        }
    }
    return std::move(packets);
}

vector<AVDemuxStream> CAvcodecReader::GetInputStreamInfo() {
    return vector<AVDemuxStream>(m_Demux.stream.begin(), m_Demux.stream.end());
}

int CAvcodecReader::GetHeader(vector<uint8_t>& bitstream) {
    if (m_Demux.video.pExtradata == nullptr) {
        m_Demux.video.nExtradataSize = m_Demux.video.pCodecCtx->extradata_size;
        //ここでav_mallocを使用しないと正常に動作しない
        m_Demux.video.pExtradata = (uint8_t *)av_malloc(m_Demux.video.pCodecCtx->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);
        //ヘッダのデータをコピーしておく
        memcpy(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata, m_Demux.video.nExtradataSize);
        memset(m_Demux.video.pExtradata + m_Demux.video.nExtradataSize, 0, FF_INPUT_BUFFER_PADDING_SIZE);

        if (m_Demux.video.bUseHEVCmp42AnnexB) {
            hevcMp42Annexb(NULL);
        } else if (m_Demux.video.pH264Bsfc && m_Demux.video.pExtradata[0] == 1) {
            uint8_t *dummy = NULL;
            int dummy_size = 0;
            std::swap(m_Demux.video.pExtradata,     m_Demux.video.pCodecCtx->extradata);
            std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
            av_bitstream_filter_filter(m_Demux.video.pH264Bsfc, m_Demux.video.pCodecCtx, nullptr, &dummy, &dummy_size, nullptr, 0, 0);
            std::swap(m_Demux.video.pExtradata,     m_Demux.video.pCodecCtx->extradata);
            std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
        } else if (m_sDecParam.codec == cudaVideoCodec_VC1) {
            int lengthFix = (0 == strcmp(m_Demux.format.pFormatCtx->iformat->name, "mpegts")) ? 0 : -1;
            vc1FixHeader(lengthFix);
        }
    }
    
    bitstream.resize(m_Demux.video.nExtradataSize);
    memcpy(bitstream.data(), m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
    return 0;
}

#pragma warning(push)
#pragma warning(disable:4100)
int CAvcodecReader::LoadNextFrame(void *dst, int dst_pitch) {
    AVPacket *pkt = &m_Demux.video.packet[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)];
    m_StreamPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_StreamPacketsBufferL1)].clear();

    if (pkt->data) {
        av_free_packet(pkt);
        pkt->data = nullptr;
        pkt->size = 0;
    }
    if (getSample(pkt)) {
        av_free_packet(pkt);
        pkt->data = nullptr;
        pkt->size = 0;
        return NVENC_THREAD_ERROR; //ファイルの終わりに到達
    }
    m_Demux.video.nSampleLoadCount++;
    m_pEncSatusInfo->m_sData.frameIn++;
    auto tm = std::chrono::system_clock::now();
    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastUpdate).count() > UPDATE_INTERVAL) {
        double progressPercent = 0.0;
        if (m_Demux.format.pFormatCtx->duration) {
            progressPercent = m_Demux.video.frameData.duration * (m_Demux.video.pCodecCtx->pkt_timebase.num / (double)m_Demux.video.pCodecCtx->pkt_timebase.den) / (m_Demux.format.pFormatCtx->duration * (1.0 / (double)AV_TIME_BASE)) * 100.0;
        }
        m_tmLastUpdate = tm;
        m_pEncSatusInfo->UpdateDisplay(tm, progressPercent);
    }
    return NV_ENC_SUCCESS;
}
#pragma warning(pop)

#endif //ENABLE_AVCUVID_READER
