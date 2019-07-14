// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#define _USE_MATH_DEFINES
#include <cmath>
#include "rgy_codepage.h"
#include "convert_csp.h"
#include "NVEncFilterSubburn.h"
#include "NVEncParam.h"

#if ENABLE_AVSW_READER

#pragma comment(lib, "libass-9.lib")

static bool check_libass_dll() {
#if defined(_WIN32) || defined(_WIN64)
    HMODULE hDll = LoadLibrary(_T("libass.dll"));
    if (hDll == NULL) {
        return false;
    }
    FreeLibrary(hDll);
    return true;
#else
    return true;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

//MSGL_FATAL 0 - RGY_LOG_ERROR  2
//MSGL_ERR   1 - RGY_LOG_ERROR  2
//MSGL_WARN  2 - RGY_LOG_WARN   1
//           3 - RGY_LOG_WARN   1
//MSGL_INFO  4 - RGY_LOG_MORE  -1 (いろいろ情報が出すぎるので)
//           5 - RGY_LOG_MORE  -1
//MSGL_V     6 - RGY_LOG_DEBUG -2
//MSGL_DBG2  7 - RGY_LOG_TRACE -3
static inline int log_level_ass2qsv(int level) {
    static const int log_level_map[] = {
        RGY_LOG_ERROR,
        RGY_LOG_ERROR,
        RGY_LOG_WARN,
        RGY_LOG_WARN,
        RGY_LOG_MORE,
        RGY_LOG_MORE,
        RGY_LOG_DEBUG,
        RGY_LOG_TRACE
    };
    return log_level_map[clamp(level, 0, _countof(log_level_map) - 1)];
}

static void ass_log(int ass_level, const char *fmt, va_list args, void *ctx) {
    ((RGYLog *)ctx)->write_line(log_level_ass2qsv(ass_level), fmt, args, CP_UTF8);
}

static void ass_log_error_only(int ass_level, const char *fmt, va_list args, void *ctx) {
    auto qsv_level = log_level_ass2qsv(ass_level);
    if (qsv_level >= RGY_LOG_ERROR) {
        ((RGYLog *)ctx)->write_line(qsv_level, fmt, args, CP_UTF8);
    }
}

NVEncFilterSubburn::NVEncFilterSubburn() :
    m_subType(0),
    m_formatCtx(),
    m_subtitleStreamIndex(-1),
    m_outCodecDecode(nullptr),
    m_outCodecDecodeCtx(unique_ptr<AVCodecContext, decltype(&avcodec_close)>(nullptr, avcodec_close)),
    m_subData(),
    m_assLibrary(unique_ptr<ASS_Library, decltype(&ass_library_done)>(nullptr, ass_library_done)),
    m_assRenderer(unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)>(nullptr, ass_renderer_done)),
    m_assTrack(unique_ptr<ASS_Track, decltype(&ass_free_track)>(nullptr, ass_free_track)) {
    m_sFilterName = _T("subburn");
}

NVEncFilterSubburn::~NVEncFilterSubburn() {
    close();
}

RGY_ERR NVEncFilterSubburn::checkParam(const std::shared_ptr<NVEncFilterParamSubburn> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.filename.length() > 0 && prm->subburn.trackId != 0) {
        AddMessage(RGY_LOG_ERROR, _T("track and filename should not be set at the same time.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.filename.length() > 0 && !PathFileExists(prm->subburn.filename.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("subtitle file \"prec\" does not exist\n"), prm->subburn.filename.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterSubburn::SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecCtx->extradata)
        av_free(codecCtx->extradata);
    codecCtx->extradata_size = size;
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecCtx->extradata, data, size);
};

RGY_ERR NVEncFilterSubburn::initAVCodec(const std::shared_ptr<NVEncFilterParamSubburn> prm) {
    auto inputCodecId = AV_CODEC_ID_NONE;
    if (prm->subburn.filename.length() > 0) {
        //ファイル読み込みの場合
        AddMessage(RGY_LOG_DEBUG, _T("trying to open subtitle file \"%s\""), prm->subburn.filename.c_str());

        std::string filename_char;
        if (0 == tchar_to_string(prm->subburn.filename.c_str(), filename_char, CP_UTF8)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to convert filename to utf-8 characters.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        {
            AVFormatContext *tmpFormatCtx = nullptr;
            int ret = avformat_open_input(&tmpFormatCtx, filename_char.c_str(), nullptr, nullptr);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\": %s\n"), char_to_tstring(filename_char, CP_UTF8).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_FILE_OPEN; // Couldn't open file
            }
            m_formatCtx = unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>>(tmpFormatCtx, RGYAVDeleter<AVFormatContext>(avformat_close_input));
        }

        if (avformat_find_stream_info(m_formatCtx.get(), nullptr) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("error finding stream information.\n"));
            return RGY_ERR_INVALID_FORMAT; // Couldn't find stream information
        }
        AddMessage(RGY_LOG_DEBUG, _T("got stream information.\n"));
        av_dump_format(m_formatCtx.get(), 0, filename_char.c_str(), 0);

        if (0 > (m_subtitleStreamIndex = av_find_best_stream(m_formatCtx.get(), AVMEDIA_TYPE_SUBTITLE, -1, -1, nullptr, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("no subtitle stream found in \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());
            return RGY_ERR_INVALID_FORMAT; // Couldn't open file
        }
        const auto pstream = m_formatCtx->streams[m_subtitleStreamIndex];
        inputCodecId = pstream->codec->codec_id;
        AddMessage(RGY_LOG_DEBUG, _T("found subtitle in stream #%d (%s), timebase %d/%d.\n"),
            m_subtitleStreamIndex, char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            pstream->time_base.num, pstream->time_base.den);
    } else {
        if (prm->streamIn.pStream == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("internal error: stream info not provided.\n"));
            return RGY_ERR_UNKNOWN;
        }
        inputCodecId = prm->streamIn.pStream->codecpar->codec_id;
        AddMessage(RGY_LOG_DEBUG, _T("using subtitle track #%d (%s), timebase %d/%d.\n"),
            prm->subburn.trackId, char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            prm->streamIn.pStream->time_base.num, prm->streamIn.pStream->time_base.den);
    }

    m_subType = avcodec_descriptor_get(inputCodecId)->props;
    AddMessage(RGY_LOG_DEBUG, _T("sub type: %s\n"), (m_subType & AV_CODEC_PROP_TEXT_SUB) ? _T("text") : _T("bitmap"));

    auto copy_subtitle_header = [](AVCodecContext *pDstCtx, const AVCodecContext *pSrcCtx) {
        if (pSrcCtx->subtitle_header_size) {
            pDstCtx->subtitle_header_size = pSrcCtx->subtitle_header_size;
            pDstCtx->subtitle_header = (uint8_t *)av_mallocz(pDstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(pDstCtx->subtitle_header, pSrcCtx->subtitle_header, pSrcCtx->subtitle_header_size);
        }
    };
    //decoderの初期化
    if (NULL == (m_outCodecDecode = avcodec_find_decoder(inputCodecId))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), inputCodecId));
        AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
        return RGY_ERR_NULL_PTR;
    }
    m_outCodecDecodeCtx = unique_ptr<AVCodecContext, decltype(&avcodec_close)>(avcodec_alloc_context3(m_outCodecDecode), avcodec_close);
    if (prm->streamIn.pStream) {
        //設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
        COPY_IF_ZERO(m_outCodecDecodeCtx->width, prm->streamIn.pStream->codecpar->width);
        COPY_IF_ZERO(m_outCodecDecodeCtx->height, prm->streamIn.pStream->codecpar->height);
#undef COPY_IF_ZERO
        m_outCodecDecodeCtx->pkt_timebase = prm->streamIn.pStream->time_base;
        SetExtraData(m_outCodecDecodeCtx.get(), prm->streamIn.pStream->codecpar->extradata, prm->streamIn.pStream->codecpar->extradata_size);
    } else {
        m_outCodecDecodeCtx->pkt_timebase = m_formatCtx->streams[m_subtitleStreamIndex]->time_base;
        auto *pCodecCtx = m_formatCtx->streams[m_subtitleStreamIndex]->codec;
        SetExtraData(m_outCodecDecodeCtx.get(), pCodecCtx->extradata, pCodecCtx->extradata_size);
    }

    int ret;
    AVDictionary *pCodecOpts = nullptr;
    if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        if (prm->subburn.filename.length() > 0) {
            if (prm->subburn.charcode.length() == 0) {
                FILE *fp = NULL;
                if (_tfopen_s(&fp, prm->subburn.filename.c_str(), _T("rb")) || fp == NULL) {
                    AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\"\n"), prm->subburn.filename.c_str());
                    return RGY_ERR_NULL_PTR; // Couldn't open file
                }

                std::vector<char> buffer(256 * 1024, 0);
                const auto readBytes = fread(buffer.data(), 1, sizeof(buffer[0]) * buffer.size(), fp);
                fclose(fp);

                const auto estCodePage = get_code_page(buffer.data(), (int)readBytes);
                std::map<uint32_t, std::string> codePageMap = {
                    { CODE_PAGE_SJIS,     "CP932"       },
                    { CODE_PAGE_JIS,      "ISO-2022-JP" },
                    { CODE_PAGE_EUC_JP,   "EUC-JP"      },
                    { CODE_PAGE_UTF8,     "UTF-8"       },
                    { CODE_PAGE_UTF16_LE, "UTF-16LE"    },
                    { CODE_PAGE_UTF16_BE, "UTF-16BE"    },
                    { CODE_PAGE_US_ASCII, "ASCII"       },
                    { CODE_PAGE_UNSET,    ""            },
                };
                if (codePageMap.find(estCodePage) != codePageMap.end()) {
                    prm->subburn.charcode = codePageMap[estCodePage];
                }
            }
        }
        if (prm->subburn.charcode.length() > 0) {
            if (0 > (ret = av_dict_set(&pCodecOpts, "sub_charenc", prm->subburn.charcode.c_str(), 0))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_charenc\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
        }
        AddMessage(RGY_LOG_DEBUG, _T("set \"sub_charenc\" to \"%s\""), char_to_tstring(prm->subburn.charcode).c_str());
        if (0 > (ret = av_dict_set(&pCodecOpts, "sub_text_format", "ass", 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_text_format\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        AddMessage(RGY_LOG_DEBUG, _T("set \"sub_text_format\" to \"ass\""));
    }
    if (0 > (ret = avcodec_open2(m_outCodecDecodeCtx.get(), m_outCodecDecode, &pCodecOpts))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
            char_to_tstring(avcodec_get_name(inputCodecId)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    if (prm->subburn.trackId == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            m_outCodecDecodeCtx->width, m_outCodecDecodeCtx->height);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSubburn::InitLibAss(const std::shared_ptr<NVEncFilterParamSubburn> prm) {
    //libassの初期化
    m_assLibrary = unique_ptr<ASS_Library, decltype(&ass_library_done)>(ass_library_init(), ass_library_done);
    if (!m_assLibrary) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass.\n"));
        return RGY_ERR_NULL_PTR;
    }
    ass_set_message_cb(m_assLibrary.get(), ass_log, m_pPrintMes.get());

    ass_set_extract_fonts(m_assLibrary.get(), 1);
    ass_set_style_overrides(m_assLibrary.get(), nullptr);

    m_assRenderer = unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)>(ass_renderer_init(m_assLibrary.get()), ass_renderer_done);
    if (!m_assRenderer) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass renderer.\n"));
        return RGY_ERR_NULL_PTR;
    }

    ass_set_use_margins(m_assRenderer.get(), 0);
    ass_set_hinting(m_assRenderer.get(), ASS_HINTING_LIGHT);
    ass_set_font_scale(m_assRenderer.get(), 1.0);
    ass_set_line_spacing(m_assRenderer.get(), 1.0);
    ass_set_shaper(m_assRenderer.get(), (ASS_ShapingLevel)prm->subburn.assShaping);

    const char *font = nullptr;
    const char *family = "Arial";
    ass_set_fonts(m_assRenderer.get(), font, family, 1, nullptr, 1);

    m_assTrack = unique_ptr<ASS_Track, decltype(&ass_free_track)>(ass_new_track(m_assLibrary.get()), ass_free_track);
    if (!m_assTrack) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass track.\n"));
        return RGY_ERR_NULL_PTR;
    }

    const int width = prm->videoInfo.srcWidth - prm->videoInfo.crop.e.left - prm->videoInfo.crop.e.right;
    const int height = prm->videoInfo.srcHeight - prm->videoInfo.crop.e.up - prm->videoInfo.crop.e.bottom;
    ass_set_frame_size(m_assRenderer.get(), width, height);

    const AVRational sar = { prm->videoInfo.sar[0], prm->videoInfo.sar[1] };
    double par = 1.0;
    if (sar.num * sar.den > 0) {
        par = (double)sar.num / sar.den;
    }
    ass_set_aspect_ratio(m_assRenderer.get(), 1, par);

    if (m_outCodecDecodeCtx && m_outCodecDecodeCtx->subtitle_header && m_outCodecDecodeCtx->subtitle_header_size > 0) {
        ass_process_codec_private(m_assTrack.get(), (char *)m_outCodecDecodeCtx->subtitle_header, m_outCodecDecodeCtx->subtitle_header_size);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSubburn::readSubFile() {
    AVPacket pkt;
    av_init_packet(&pkt);
    while (av_read_frame(m_formatCtx.get(), &pkt) >= 0) {
        if (pkt.stream_index == m_subtitleStreamIndex) {
            addStreamPacket(&pkt);
        } else {
            av_packet_unref(&pkt);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSubburn::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    //tweakは常に元のフレームを書き換え
    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, subburn will overwrite input frame.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;
    m_queueSubPackets.init();

    //字幕読み込み・デコーダの初期化
    if ((sts = initAVCodec(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        if ((sts = InitLibAss(prm)) != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (prm->streamIn.pStream == nullptr) {
        if ((sts = readSubFile()) != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (prm->subburn.scale <= 0.0f) {
        if (m_outCodecDecodeCtx->width > 0 && m_outCodecDecodeCtx->height > 0) {
            double scaleX = prm->frameOut.width / m_outCodecDecodeCtx->width;
            double scaleY = prm->frameOut.height / m_outCodecDecodeCtx->height;
            prm->subburn.scale = (float)std::sqrt(scaleX * scaleX + scaleY * scaleY);
            if (std::abs(prm->subburn.scale - 1.0f) <= 0.1f) {
                prm->subburn.scale = 1.0f;
            }
        } else {
            prm->subburn.scale = 1.0f;
        }
    } else if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        AddMessage(RGY_LOG_WARN, _T("manual scaling not available for text type fonts.\n"));
        prm->subburn.scale = 1.0f;
    }

    if (prm->subburn.filename.length() > 0) {
        m_sFilterInfo = strsprintf(_T("subburn: %s, scale %.2f"), prm->subburn.filename.c_str(), prm->subburn.scale);
    } else {
        m_sFilterInfo = strsprintf(_T("subburn: track #%d, scale %.2f"), prm->subburn.trackId, prm->subburn.scale);
    }

    //コピーを保存
    m_pParam = prm;
    return sts;
}

int NVEncFilterSubburn::targetTrackIdx() {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);
    if (!prm) {
        return 0;
    }
    return prm->streamIn.nTrackId;
}

RGY_ERR NVEncFilterSubburn::addStreamPacket(AVPacket *pkt) {
    m_queueSubPackets.push(*pkt);
    const int log_level = RGY_LOG_TRACE;
    if (m_pPrintMes != nullptr && log_level >= m_pPrintMes->getLogLevel()) {
        auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);
        if (!prm) {
            AddMessage(log_level, _T("Invalid parameter type.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        const auto inputSubStream = (prm->streamIn.pStream) ? prm->streamIn.pStream : m_formatCtx->streams[m_subtitleStreamIndex];
        const int64_t vidInputOffsetMs = av_rescale_q(prm->videoInputFirstKeyPts, prm->videoTimebase, { 1, 1000 });
        const auto pktTimeMs = av_rescale_q(pkt->pts, inputSubStream->time_base, { 1, 1000 }) - vidInputOffsetMs;
        AddMessage(log_level, _T("Add subtitle packet: %s\n"), getTimestampString(pktTimeMs, av_make_q(1, 1000)).c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSubburn::procFrame(FrameInfo *pOutputFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inputSubStream = (prm->streamIn.pStream) ? prm->streamIn.pStream : m_formatCtx->streams[m_subtitleStreamIndex];
    const int64_t nFrameTimeMs = av_rescale_q(pOutputFrame->timestamp, prm->videoTimebase, { 1, 1000 });
    const int64_t vidInputOffsetMs = av_rescale_q(prm->videoInputFirstKeyPts, prm->videoTimebase, { 1, 1000 });

    AVPacket pkt;
    while (m_queueSubPackets.front_copy_no_lock(&pkt)) {
        const auto pktTimeMs = av_rescale_q(pkt.pts, inputSubStream->time_base, { 1, 1000 }) - vidInputOffsetMs;
        if (!(m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            //字幕パケットのptsが、フレームのptsより古ければ、処理する必要がある
            if (nFrameTimeMs < pktTimeMs) {
                //取得したパケットが未来のパケットなら無視
                break;
            }
        }
        //字幕パケットをキューから取り除く
        m_queueSubPackets.pop();

        //新たに字幕構造体を確保(これまで構築していたデータは破棄される)
        m_subData = unique_ptr<AVSubtitle, subtitle_deleter>(new AVSubtitle(), subtitle_deleter());
        if (!(m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            m_subImages.clear();
        }

        //字幕パケットをデコードする
        int got_sub = 0;
        if (0 > avcodec_decode_subtitle2(m_outCodecDecodeCtx.get(), m_subData.get(), &got_sub, &pkt)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
            return RGY_ERR_NONE;
        }
        if (got_sub) {
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs;
            AddMessage(RGY_LOG_TRACE, _T("decoded subtitle chunk (%s - %s), Video frame (%s)"),
                getTimestampString(nStartTime, av_make_q(1, 1000)).c_str(),
                getTimestampString(nStartTime + m_subData->end_display_time, av_make_q(1, 1000)).c_str(),
                getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
        }
        if (got_sub && (m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs;
            const int64_t nDuration  = m_subData->end_display_time;
            for (uint32_t i = 0; i < m_subData->num_rects; i++) {
                auto *ass = m_subData->rects[i]->ass;
                if (!ass) {
                    break;
                }
                ass_process_chunk(m_assTrack.get(), ass, (int)strlen(ass), nStartTime, nDuration);
            }
        }
        av_packet_unref(&pkt);
    }

    if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        return procFrameText(pOutputFrame, nFrameTimeMs, stream);
    } else {
        if (m_subData) {
            //いまなんらかの字幕情報がデコード済みなら、その有効期限をチェックする
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs;
            const int64_t nDuration  = m_subData->end_display_time;
            if (nStartTime + nDuration < nFrameTimeMs) {
                //現在蓄えている字幕データを開放
                AddMessage(RGY_LOG_TRACE, _T("release subtitle chunk (%s - %s) [video frame (%s)]"),
                    getTimestampString(nStartTime, av_make_q(1, 1000)).c_str(),
                    getTimestampString(nStartTime + nDuration, av_make_q(1, 1000)).c_str(),
                    getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
                m_subData.reset();
                m_subImages.clear();
                return RGY_ERR_NONE;
            }
            AddMessage(RGY_LOG_TRACE, _T("burn subtitle into video frame (%s)"),
                getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
            return procFrameBitmap(pOutputFrame, prm->crop, stream);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSubburn::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!ppOutputFrames[0]->deivce_mem) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((sts = procFrame(ppOutputFrames[0], cudaStreamDefault)) != RGY_ERR_NONE) {
        return sts;
    }

    return sts;
}

void NVEncFilterSubburn::close() {
    m_assTrack.reset();
    m_assRenderer.reset();
    m_assLibrary.reset();
    m_queueSubPackets.clear();
    m_subData.reset();
    m_outCodecDecodeCtx.reset();
    m_formatCtx.reset();
    m_subType = 0;
    m_pFrameBuf.clear();
}

#endif //#if ENABLE_AVSW_READER
