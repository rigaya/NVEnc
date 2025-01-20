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

#include "rgy_output.h"
#include "rgy_filesystem.h"
#include "rgy_bitstream.h"
#include "rgy_language.h"
#include "convert_csp.h"
#include <filesystem>
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
#include <smmintrin.h>
#endif

#if ENCODER_QSV || ENCODER_NVENC

static RGY_ERR WriteY4MHeader(FILE *fp, const VideoInfo *info, const RGY_CSP csp) {
    char buffer[256] = { 0 };
    char *ptr = buffer;
    uint32_t len = 0;
    memcpy(ptr, "YUV4MPEG2 ", 10);
    len += 10;

    len += sprintf_s(ptr+len, sizeof(buffer)-len, "W%d H%d F%d:%d ", info->dstWidth, info->dstHeight, info->fpsN, info->fpsD);

    const char *picstruct = "Ip ";
    if (info->picstruct & RGY_PICSTRUCT_TFF) {
        picstruct = "It ";
    } else if (info->picstruct & RGY_PICSTRUCT_BFF) {
        picstruct = "Ib ";
    }
    strcpy_s(ptr+len, sizeof(buffer)-len, picstruct); len += 3;
    len += sprintf_s(ptr+len, sizeof(buffer)-len, "A%d:%d ", info->sar[0], info->sar[1]);
    const auto cspHeader = csp_rgy_to_y4mheader(csp);
    if (!cspHeader) return RGY_ERR_INVALID_COLOR_FORMAT;

    len += sprintf_s(ptr+len, sizeof(buffer)-len, "C%s\n", cspHeader);
    return (len == fwrite(buffer, 1, len, fp)) ? RGY_ERR_NONE : RGY_ERR_UNDEFINED_BEHAVIOR;
}

#endif //#if ENCODER_QSV || ENCODER_NVENC

#define WRITE_CHECK(writtenBytes, expected) { \
    if (writtenBytes != expected) { \
        AddMessage(RGY_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\n")); \
        return RGY_ERR_UNDEFINED_BEHAVIOR; \
    } }

const char *RGYOutput::OUT_DEBUG_FILE_HEADER = "size %d, pts %lld, dts %lld, duration %lld, frametype %d, frameidx %d, picstruct %d";

RGYOutput::RGYOutput() :
    m_outFilename(),
    m_encSatusInfo(),
    m_fDest(),
    m_fpDebug(),
    m_fpOutReplay(),
    m_outputIsStdout(false),
    m_inited(false),
    m_noOutput(false),
    m_OutType(OUT_TYPE_BITSTREAM),
    m_sourceHWMem(false),
    m_y4mHeaderWritten(false),
    m_enableHEVCAlphaChannelInfoSEIOverwrite(false),
    m_HEVCAlphaChannelMode(0),
    m_strWriterName(),
    m_strOutputInfo(),
    m_VideoOutputInfo(),
    m_printMes(),
    m_outputBuffer(),
    m_readBuffer(),
    m_UVBuffer(),
    m_bsf(),
    m_parse_nal_hevc(get_parse_nal_unit_hevc_func()) {
}

RGYOutput::~RGYOutput() {
    m_encSatusInfo.reset();
    m_printMes.reset();
    Close();
}

void RGYOutput::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing file \"%s\"...\n"), m_outFilename.c_str());
    if (m_fDest) {
        m_fDest.reset();
        AddMessage(RGY_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    m_fpOutReplay.reset();
    m_fpDebug.reset();
    m_encSatusInfo.reset();
    m_outputBuffer.reset();
    m_readBuffer.reset();
    m_UVBuffer.reset();
    m_bsf.reset();

    m_noOutput = false;
    m_inited = false;
    m_sourceHWMem = false;
    m_y4mHeaderWritten = false;
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
    m_printMes.reset();
}

RGY_ERR RGYOutput::writeRawDebug(RGYBitstream *pBitstream) {
    if (!m_fpDebug) return RGY_ERR_NONE;

    char frame_info[256] = { 0 };
    sprintf_s(frame_info, OUT_DEBUG_FILE_HEADER,
        (int)pBitstream->size(), pBitstream->pts(), pBitstream->dts(), pBitstream->duration(),
        pBitstream->frametype(), pBitstream->frameIdx(), pBitstream->picstruct());
    _fwrite_nolock(frame_info, 1, sizeof(frame_info), m_fpDebug.get());
    _fwrite_nolock(pBitstream->data(), 1, pBitstream->size(), m_fpDebug.get());
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutput::readRawDebug(RGYBitstream *pBitstream) {
    if (!m_fpOutReplay) return RGY_ERR_NONE;

    char frame_info[256] = { 0 };
    if (_fread_nolock(frame_info, 1, sizeof(frame_info), m_fpOutReplay.get()) != sizeof(frame_info)) {
        return RGY_ERR_MORE_DATA;
    }
    int size = 0, frameIdx = 0;
    int64_t pts = 0, dts = 0, duration = 0;
    RGY_FRAMETYPE frametype = RGY_FRAMETYPE_UNKNOWN;
    RGY_PICSTRUCT picstruct = RGY_PICSTRUCT_UNKNOWN;
    if (sscanf_s(frame_info, OUT_DEBUG_FILE_HEADER, &size, &pts, &dts, &duration, &frametype, &frameIdx, &picstruct) != 7) {
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    std::vector<uint8_t> buffer(size, 0);
    if (_fread_nolock(buffer.data(), 1, buffer.size(), m_fpOutReplay.get()) != buffer.size()) {
        return RGY_ERR_MORE_DATA;
    }
    pBitstream->setDuration(duration);
    pBitstream->setFrameIdx(frameIdx);
    pBitstream->setFrametype(frametype);
    pBitstream->setPicstruct(picstruct);
    pBitstream->copy(buffer.data(), buffer.size(), dts, pts);
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutput::InitVideoBsf(const VideoInfo *videoOutputInfo) {
    // bsfの作成
    std::unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>> codecpar(avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));

    codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    codecpar->codec_id = getAVCodecId(videoOutputInfo->codec);
    codecpar->width = videoOutputInfo->dstWidth;
    codecpar->height = videoOutputInfo->dstHeight;
    codecpar->format = csp_rgy_to_avpixfmt(videoOutputInfo->csp);
    codecpar->level = videoOutputInfo->codecLevel;
    codecpar->profile = videoOutputInfo->codecProfile;
    codecpar->sample_aspect_ratio.num = videoOutputInfo->sar[0];
    codecpar->sample_aspect_ratio.den = videoOutputInfo->sar[1];
    codecpar->chroma_location = (AVChromaLocation)videoOutputInfo->vui.chromaloc;
    codecpar->field_order = picstrcut_rgy_to_avfieldorder(videoOutputInfo->picstruct);
    codecpar->video_delay = (m_VideoOutputInfo.codec == RGY_CODEC_AV1 && AV1_TIMESTAMP_OVERRIDE) ? 0 : videoOutputInfo->videoDelay;
    if (videoOutputInfo->vui.descriptpresent) {
        codecpar->color_space = (AVColorSpace)videoOutputInfo->vui.matrix;
        codecpar->color_primaries = (AVColorPrimaries)videoOutputInfo->vui.colorprim;
        codecpar->color_range = (AVColorRange)videoOutputInfo->vui.colorrange;
        codecpar->color_trc = (AVColorTransferCharacteristic)videoOutputInfo->vui.transfer;
    }

    std::vector<std::unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>>> bsfList;

    // VUI情報設定用
    if ((ENCODER_NVENC
        && (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC)
        && videoOutputInfo->sar[0] * videoOutputInfo->sar[1] > 0)
        || (ENCODER_QSV
            && (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC || videoOutputInfo->codec == RGY_CODEC_AV1)
            && videoOutputInfo->vui.chromaloc != 0)
        || (ENCODER_VCEENC
            && (videoOutputInfo->codec == RGY_CODEC_HEVC // HEVCの時は常に上書き
                || (videoOutputInfo->vui.format != 5
                    || videoOutputInfo->vui.colorprim != 2
                    || videoOutputInfo->vui.transfer != 2
                    || videoOutputInfo->vui.matrix != 2
                    || videoOutputInfo->vui.chromaloc != 0)
                || (videoOutputInfo->codec == RGY_CODEC_AV1 && videoOutputInfo->vui.colorrange == RGY_COLORRANGE_FULL)))
        || (ENCODER_MPP
            && ((videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC) // HEVCの時は常に上書き)
                || (videoOutputInfo->sar[0] * videoOutputInfo->sar[1] > 0
                    || (videoOutputInfo->vui.format != 5
                        || videoOutputInfo->vui.colorprim != 2
                        || videoOutputInfo->vui.transfer != 2
                        || videoOutputInfo->vui.matrix != 2
                        || videoOutputInfo->vui.chromaloc != 0))))) {
        const char *bsf_name = nullptr;
        switch (videoOutputInfo->codec) {
        case RGY_CODEC_H264: bsf_name = "h264_metadata"; break;
        case RGY_CODEC_HEVC: bsf_name = "hevc_metadata"; break;
        case RGY_CODEC_AV1:  bsf_name = "av1_metadata"; break;
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
        AVBSFContext *bsfctx = nullptr;
        int ret = 0;
        if (0 > (ret = av_bsf_alloc(filter, &bsfctx))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_NULL_PTR;
        }
        std::unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> bsfCtx(bsfctx, RGYAVDeleter<AVBSFContext>(av_bsf_free));
        bsfctx = nullptr;

        if (0 > (ret = avcodec_parameters_copy(bsfCtx->par_in, codecpar.get()))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AVDictionary *bsfPrm = nullptr;
        std::unique_ptr<AVDictionary*, decltype(&av_dict_free)> bsfPrmDictDeleter(&bsfPrm, av_dict_free);
        if (ENCODER_MPP) {
            const auto level_str = get_cx_desc(get_level_list(videoOutputInfo->codec), videoOutputInfo->codecLevel);
            av_dict_set(&bsfPrm, "level", tchar_to_string(level_str).c_str(), 0);
            AddMessage(RGY_LOG_DEBUG, _T("set level %s by %s filter\n"), level_str, bsf_tname.c_str());
        }
        if ((ENCODER_NVENC || ENCODER_MPP) && videoOutputInfo->sar[0] * videoOutputInfo->sar[1] > 0) {
            char sar[128];
            sprintf_s(sar, "%d/%d", videoOutputInfo->sar[0], videoOutputInfo->sar[1]);
            av_dict_set(&bsfPrm, "sample_aspect_ratio", sar, 0);
            AddMessage(RGY_LOG_DEBUG, _T("set sar %d:%d by %s filter\n"), videoOutputInfo->sar[0], videoOutputInfo->sar[1], bsf_tname.c_str());
        }
        if (ENCODER_VCEENC || ENCODER_MPP) {
            // HEVCの10bitの時、エンコーダがおかしなVUIを設定することがあるのでこれを常に上書き
            const bool override_always = ENCODER_VCEENC && (videoOutputInfo->codec == RGY_CODEC_HEVC || videoOutputInfo->codec == RGY_CODEC_AV1);
            if (override_always || videoOutputInfo->vui.format != 5 /*undef*/) {
                if (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC) {
                    av_dict_set_int(&bsfPrm, "video_format", videoOutputInfo->vui.format, 0);
                    AddMessage(RGY_LOG_DEBUG, _T("set video_format %d by %s filter\n"), videoOutputInfo->vui.format, bsf_tname.c_str());
                }
            }
            if (override_always || videoOutputInfo->vui.colorprim != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, (videoOutputInfo->codec == RGY_CODEC_AV1) ? "color_primaries" : "colour_primaries", videoOutputInfo->vui.colorprim, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set colorprim %d by %s filter\n"), videoOutputInfo->vui.colorprim, bsf_tname.c_str());
            }
            if (override_always || videoOutputInfo->vui.transfer != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "transfer_characteristics", videoOutputInfo->vui.transfer, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set transfer %d by %s filter\n"), videoOutputInfo->vui.transfer, bsf_tname.c_str());
            }
            if (override_always || videoOutputInfo->vui.matrix != 2 /*undef*/) {
                av_dict_set_int(&bsfPrm, "matrix_coefficients", videoOutputInfo->vui.matrix, 0);
                AddMessage(RGY_LOG_DEBUG, _T("set matrix %d by %s filter\n"), videoOutputInfo->vui.matrix, bsf_tname.c_str());
            }
            if (override_always || videoOutputInfo->vui.colorrange != RGY_COLORRANGE_UNSPECIFIED /*undef*/) {
                if (videoOutputInfo->codec == RGY_CODEC_AV1) {
                    av_dict_set(&bsfPrm, "color_range", videoOutputInfo->vui.colorrange == RGY_COLORRANGE_FULL ? "pc" : "tv", 0);
                    AddMessage(RGY_LOG_DEBUG, _T("set color_range %s by %s filter\n"), videoOutputInfo->vui.colorrange == RGY_COLORRANGE_FULL ? _T("full") : _T("limited"), bsf_tname.c_str());
                } else if (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC) {
                    av_dict_set_int(&bsfPrm, "video_full_range_flag", videoOutputInfo->vui.colorrange == RGY_COLORRANGE_FULL ? 1 : 0, 0);
                    AddMessage(RGY_LOG_DEBUG, _T("set color_range %s by %s filter\n"), videoOutputInfo->vui.colorrange == RGY_COLORRANGE_FULL ? _T("full") : _T("limited"), bsf_tname.c_str());
                }
            }
        }
        if (ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP) {
            if (videoOutputInfo->vui.chromaloc != 0) {
                if (videoOutputInfo->codec == RGY_CODEC_AV1) {
                    if (videoOutputInfo->vui.chromaloc == RGY_CHROMALOC_TOPLEFT) {
                        av_dict_set(&bsfPrm, "chroma_sample_position", "colocated", 0);
                    } else {
                        av_dict_set(&bsfPrm, "chroma_sample_position", "vertical", 0);
                    }
                    AddMessage(RGY_LOG_DEBUG, _T("set chromaloc %d by %s filter\n"), videoOutputInfo->vui.chromaloc - 1, bsf_tname.c_str());
                } else if (videoOutputInfo->codec == RGY_CODEC_H264 || videoOutputInfo->codec == RGY_CODEC_HEVC) {
                    av_dict_set_int(&bsfPrm, "chroma_sample_loc_type", videoOutputInfo->vui.chromaloc - 1, 0);
                    AddMessage(RGY_LOG_DEBUG, _T("set chromaloc %d by %s filter\n"), videoOutputInfo->vui.chromaloc - 1, bsf_tname.c_str());
                }
            }
        }
        if (0 > (ret = av_opt_set_dict2(bsfCtx.get(), &bsfPrm, AV_OPT_SEARCH_CHILDREN))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        if (0 > (ret = av_bsf_init(bsfCtx.get()))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsf_tname.c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter\n"), bsf_tname.c_str());
        bsfList.push_back(std::move(bsfCtx));
    }

    if (bsfList.size() > 0) {
        int ret = 0;
        std::unique_ptr<AVBSFList, RGYAVDeleter<AVBSFList>> avBsfList(av_bsf_list_alloc(), RGYAVDeleter<AVBSFList>(av_bsf_list_free));
        for (auto& bsf : bsfList) {
            if (0 > (ret = av_bsf_list_append(avBsfList.get(), bsf.release()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to append bsf %s: %s.\n"), char_to_tstring(bsf->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
        }
        AVBSFContext *bsfc = nullptr;
        auto bsfPtr = avBsfList.release();
        if (0 > (ret = av_bsf_list_finalize(&bsfPtr, &bsfc))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to finalize bsf list: %s.\n"), qsv_av_err2str(ret).c_str());
            av_bsf_list_free(&bsfPtr);
            return RGY_ERR_UNKNOWN;
        }
        m_bsf = std::make_unique<RGYOutputBSF>(bsfc, videoOutputInfo->codec, m_strWriterName, m_printMes);
    }
    return RGY_ERR_NONE;
}

template<typename T>
std::pair<RGY_ERR, std::vector<uint8_t>> RGYOutput::getMetadata(const RGYFrameDataType metadataType, const RGYTimestampMapVal& bs_framedata, const RGYFrameDataMetadataConvertParam *convPrm) {
    const auto frameDataMetadata = std::find_if(bs_framedata.dataList.begin(), bs_framedata.dataList.end(), [metadataType](const std::shared_ptr<RGYFrameData>& data) {
        return data->dataType() == metadataType;
        });
    std::vector<uint8_t> metadata;
    if (frameDataMetadata != bs_framedata.dataList.end()) {
        auto frameDataPtr = dynamic_cast<T *>((*frameDataMetadata).get());
        if (!frameDataPtr) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid cast to %s metadata.\n"));
            return { RGY_ERR_UNSUPPORTED, metadata };
        }
        if (auto sts = frameDataPtr->convert(convPrm); sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to convert metadata: %s.\n"), get_err_mes(sts));
            return { sts, metadata };
        }
        if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
            metadata = frameDataPtr->gen_nal();
        } else if (m_VideoOutputInfo.codec == RGY_CODEC_AV1) {
            metadata = frameDataPtr->gen_obu();
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Setting %s metadata not supported in %s encoding.\n"), RGYFrameDataTypeToStr(metadataType), CodecToStr(m_VideoOutputInfo.codec).c_str());
            return { RGY_ERR_UNSUPPORTED, metadata };
        }
    }
    return { RGY_ERR_NONE, metadata };
}


RGY_ERR RGYOutput::OverwriteHEVCAlphaChannelInfoSEI(RGYBitstream *bitstream) {
    if (m_VideoOutputInfo.codec != RGY_CODEC_HEVC || !m_enableHEVCAlphaChannelInfoSEIOverwrite) {
        return RGY_ERR_NONE;
    }
    RGYBitstream bsCopy = RGYBitstreamInit();
    bsCopy.copy(bitstream);
    const auto nal_list = m_parse_nal_hevc(bsCopy.data(), bsCopy.size());
    const bool has_prefix_sei = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.nuh_layer_id == 0 && info.type == NALU_HEVC_PREFIX_SEI; }) != nal_list.end();
    if (!has_prefix_sei) {
        return RGY_ERR_NONE;
    }

    bitstream->setSize(0);
    bitstream->setOffset(0);
    for (const auto& nal : nal_list) {
        if (nal.nuh_layer_id == 0 && nal.type == NALU_HEVC_PREFIX_SEI) {
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
            if (sei_type == ALPHA_CHANNEL_INFO) { // alpha_channel_information
                const auto nalbuf = gen_hevc_alpha_channel_info_sei(m_HEVCAlphaChannelMode);
                bitstream->append(nalbuf.data(), nalbuf.size());
            } else {
                bitstream->append(nal.ptr, nal.size);
            }
        } else {
            bitstream->append(nal.ptr, nal.size);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutput::InsertMetadata(RGYBitstream *bitstream, std::vector<std::unique_ptr<RGYOutputInsertMetadata>>& metadataList) {
    if (metadataList.size() == 0) {
        return RGY_ERR_NONE;
    }
    if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
        RGYBitstream bsCopy = RGYBitstreamInit();
        bsCopy.copy(bitstream);
        const auto nal_list = m_parse_nal_hevc(bsCopy.data(), bsCopy.size());
        const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
        const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
        const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
        const bool header_check = (nal_list.end() != hevc_vps_nal) || (nal_list.end() != hevc_sps_nal) || (nal_list.end() != hevc_pps_nal);

        // onSequenceHeader = trueの場合、ヘッダーがない場合は、written=trueにして書き込まないようにする
        for (auto& metadata : metadataList) {
            if (metadata->onSequenceHeader && !header_check) {
                metadata->written = true;
            }
        }

        bitstream->setSize(0);
        bitstream->setOffset(0);
        if (!header_check) {
            for (auto& metadata : metadataList) {
                if (!metadata->written && !metadata->appendix) {
                    bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                    metadata->written = true;
                }
            }
        }
        for (int i = 0; i < (int)nal_list.size(); i++) {
            bitstream->append(nal_list[i].ptr, nal_list[i].size);
            if (nal_list[i].type == NALU_HEVC_VPS || nal_list[i].type == NALU_HEVC_SPS || nal_list[i].type == NALU_HEVC_PPS) {
                if (i + 1 < (int)nal_list.size()
                    && (nal_list[i + 1].type != NALU_HEVC_VPS && nal_list[i + 1].type != NALU_HEVC_SPS && nal_list[i + 1].type != NALU_HEVC_PPS)) {
                    for (auto& metadata : metadataList) {
                        if (!metadata->written && !metadata->appendix) {
                            bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                            metadata->written = true;
                        }
                    }
                }
            }
        }
        for (auto& metadata : metadataList) {
            if (!metadata->written && metadata->appendix) {
                bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                metadata->written = true;
            }
        }
        bsCopy.clear();
        for (auto& metadata : metadataList) {
            if (!metadata->written) {
                AddMessage(RGY_LOG_ERROR, _T("metadata not written, unexpected HEVC header.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
        }
    } else if (m_VideoOutputInfo.codec == RGY_CODEC_AV1) {
        const auto av1_units = parse_unit_av1(bitstream->data(), bitstream->size());
        bitstream->setSize(0);
        bitstream->setOffset(0);

        const auto has_seq_header = std::find_if(av1_units.begin(), av1_units.end(), [](const std::unique_ptr<unit_info>& info) { return info->type == OBU_SEQUENCE_HEADER; }) != av1_units.end();
        const auto has_td = std::find_if(av1_units.begin(), av1_units.end(), [](const std::unique_ptr<unit_info>& info) { return info->type == OBU_TEMPORAL_DELIMITER; }) != av1_units.end();

        // onSequenceHeader = trueの場合、ヘッダーがない場合は、written=trueにして書き込まないようにする
        for (auto& metadata : metadataList) {
            if (metadata->onSequenceHeader && !has_seq_header) {
                metadata->written = true;
            }
        }

        if (!has_seq_header && !has_td) {
            for (auto& metadata : metadataList) {
                if (!metadata->written && !metadata->appendix) {
                    bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                    metadata->written = true;
                }
            }
        }

        for (size_t i = 0; i < av1_units.size(); i++) {
            bitstream->append(av1_units[i]->unit_data.data(), av1_units[i]->unit_data.size());
            if (av1_units[i]->type == OBU_TEMPORAL_DELIMITER || av1_units[i]->type == OBU_SEQUENCE_HEADER) {
                if (i + 1 < av1_units.size()
                    && (av1_units[i + 1]->type != OBU_TEMPORAL_DELIMITER && av1_units[i + 1]->type != OBU_SEQUENCE_HEADER)) {
                    for (auto& metadata : metadataList) {
                        if (!metadata->written && !metadata->appendix) {
                            bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                            metadata->written = true;
                        }
                    }
                }
            }
        }
        for (auto& metadata : metadataList) {
            if (!metadata->written && metadata->appendix) {
                bitstream->append(metadata->mdata.data(), metadata->mdata.size());
                metadata->written = true;
            }
        }
        for (auto& metadata : metadataList) {
            if (!metadata->written) {
                AddMessage(RGY_LOG_ERROR, _T("metadata not written, unexpected AV1 frame.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("Setting metadata not supported in %s encoding.\n"), CodecToStr(m_VideoOutputInfo.codec).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGYOutputBSF::RGYOutputBSF(AVBSFContext *bsf, RGY_CODEC codec, tstring strWriterName, shared_ptr<RGYLog> log) :
    m_strWriterName(strWriterName),
    m_log(log),
    m_codec(codec),
    m_bsfc(std::unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>>(bsf, RGYAVDeleter<AVBSFContext>(av_bsf_free))),
    m_pkt(std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>>(av_packet_alloc(), RGYAVDeleter<AVPacket>(av_packet_free))),
    m_bsfBuffer(codec),
    m_parse_nal_h264(get_parse_nal_unit_h264_func()),
    m_parse_nal_hevc(get_parse_nal_unit_hevc_func()) {

}

RGYOutputBSF::~RGYOutputBSF() { }

RGY_ERR RGYOutputBSF::applyBitstreamFilter(RGYBitstream *bitstream) {
    if (m_codec == RGY_CODEC_H264 || m_codec == RGY_CODEC_HEVC) {
        int target_nal_start = -1;
        int target_nal_end = -1;
        std::vector<nal_info> nal_list;
        if (m_codec == RGY_CODEC_HEVC) {
            nal_list = m_parse_nal_hevc(bitstream->data(), bitstream->size());
            for (int i = 0; i < (int)nal_list.size(); i++) {
                if (nal_list[i].type == NALU_HEVC_VPS || nal_list[i].type == NALU_HEVC_SPS || nal_list[i].type == NALU_HEVC_PPS) {
                    if (target_nal_start < 0) target_nal_start = i;
                    target_nal_end = i;
                } else if (target_nal_start >= 0) {
                    break;
                }
            }
        } else if (m_codec == RGY_CODEC_H264) {
            nal_list = m_parse_nal_h264(bitstream->data(), bitstream->size());
            for (int i = 0; i < (int)nal_list.size(); i++) {
                if (nal_list[i].type == NALU_H264_SPS || nal_list[i].type == NALU_H264_PPS) {
                    if (target_nal_start < 0) target_nal_start = i;
                    target_nal_end = i;
                } else if (target_nal_start >= 0) {
                    break;
                }
            }
        }
        if (target_nal_start >= 0) {
            const ptrdiff_t header_size = (ptrdiff_t)(nal_list[target_nal_end].ptr - nal_list[target_nal_start].ptr) + nal_list[target_nal_end].size;
            if (header_size <= 0) {
                AddMessage(RGY_LOG_ERROR, _T("Unexpected error occured running bitstream filter.\n"));
                return RGY_ERR_UNKNOWN;
            }
            AVPacket *pkt = m_pkt.get();
            av_new_packet(pkt, (int)header_size);
            memcpy(pkt->data, nal_list[target_nal_start].ptr, header_size);
            int ret = 0;
            if (0 > (ret = av_bsf_send_packet(m_bsfc.get(), pkt))) {
                av_packet_unref(pkt);
                AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
                    char_to_tstring(m_bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            ret = av_bsf_receive_packet(m_bsfc.get(), pkt);
            if (ret == AVERROR(EAGAIN)) {
                return RGY_ERR_NONE;
            } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
                    char_to_tstring(m_bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            const auto new_data_size = bitstream->size() + pkt->size - header_size;
            m_bsfBuffer.resize(new_data_size);
            size_t offset = 0;
            for (int i = 0; i < target_nal_start; i++) {
                memcpy(m_bsfBuffer.data() + offset, nal_list[i].ptr, nal_list[i].size);
                offset += nal_list[i].size;
            }
            memcpy(m_bsfBuffer.data() + offset, pkt->data, pkt->size);
            offset += pkt->size;
            for (int i = target_nal_end+1; i < (int)nal_list.size(); i++) {
                memcpy(m_bsfBuffer.data() + offset, nal_list[i].ptr, nal_list[i].size);
                offset += nal_list[i].size;
            }
            if (new_data_size != offset) {
                AddMessage(RGY_LOG_ERROR, _T("Unexpected error occured after running bitstream filter.\n"));
                return RGY_ERR_UNKNOWN;
            }
            bitstream->copy(m_bsfBuffer.data(), new_data_size);
            av_packet_unref(pkt);

            av_bsf_flush(m_bsfc.get());
        }
    } else if (m_codec == RGY_CODEC_AV1) {
        AVPacket *pkt = m_pkt.get();
        av_new_packet(pkt, (int)bitstream->size());
        memcpy(pkt->data, bitstream->data(), bitstream->size());
        int ret = 0;
        if (0 > (ret = av_bsf_send_packet(m_bsfc.get(), pkt))) {
            av_packet_unref(pkt);
            AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
                char_to_tstring(m_bsfc.get()->filter->name).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        ret = av_bsf_receive_packet(m_bsfc.get(), pkt);
        if (ret == AVERROR(EAGAIN)) {
            return RGY_ERR_NONE;
        } else if ((ret < 0 && ret != AVERROR_EOF) || pkt->size < 0) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
                char_to_tstring(m_bsfc.get()->filter->name).c_str(), qsv_av_err2str(ret).c_str());
            return RGY_ERR_UNKNOWN;
        }
        bitstream->clear();
        bitstream->append(pkt->data, pkt->size);
        av_bsf_flush(m_bsfc.get());
        av_packet_unref(pkt);
    } else {
        AddMessage(RGY_LOG_ERROR, _T("bitstream filter not supported for %s.\n"), CodecToStr(m_codec).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGYOutputRaw::RGYOutputRaw() :
    m_outputBuf2(),
    m_hdrBitstream(),
    m_hdr10plusMetadataCopy(false),
    m_doviProfileDst(RGY_DOVI_PROFILE_UNSET),
    m_doviRpu(nullptr),
    m_doviRpuMetadataCopy(false),
    m_doviRpuConvertParam(),
    m_timestamp(nullptr),
    m_prevInputFrameId(-1),
    m_prevEncodeFrameId(-1),
    m_debugDirectAV1Out(false) {
    m_strWriterName = _T("bitstream");
    m_OutType = OUT_TYPE_BITSTREAM;
}

RGYOutputRaw::~RGYOutputRaw() {
    if (m_fpDebug) {
        m_fpDebug.reset();
    }
}

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputRaw::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(pVideoOutputInfo);
    RGYOutputRawPrm *rawPrm = (RGYOutputRawPrm *)prm;
    if (!rawPrm->benchmark && _tcslen(strFileName) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("output filename not set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (rawPrm->benchmark) {
        m_noOutput = true;
        AddMessage(RGY_LOG_DEBUG, _T("no output for benchmark mode.\n"));
    } else {
        if (_tcscmp(strFileName, _T("-")) == 0) {
            m_fDest.reset(stdout);
            m_outputIsStdout = true;
            AddMessage(RGY_LOG_DEBUG, _T("using stdout\n"));
        } else {
            CreateDirectoryRecursive(PathRemoveFileSpecFixed(strFileName).second.c_str());
            FILE *fp = NULL;
            int error = _tfopen_s(&fp, strFileName, _T("wb+"));
            if (error != 0 || fp == NULL) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open output file \"%s\": %s\n"), strFileName, _tcserror(error));
                return RGY_ERR_FILE_OPEN;
            }
            m_fDest.reset(fp);
            AddMessage(RGY_LOG_DEBUG, _T("Opened file \"%s\"\n"), strFileName);

            int bufferSizeByte = clamp(rawPrm->bufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
            if (bufferSizeByte) {
                void *ptr = nullptr;
                bufferSizeByte = (int)malloc_degeneracy(&ptr, bufferSizeByte, 1024 * 1024);
                if (bufferSizeByte) {
                    m_outputBuffer.reset((char*)ptr);
                    setvbuf(m_fDest.get(), m_outputBuffer.get(), _IOFBF, bufferSizeByte);
                    AddMessage(RGY_LOG_DEBUG, _T("Added %d MB output buffer.\n"), bufferSizeByte / (1024 * 1024));
                }
            }
        }

        if (auto sts = InitVideoBsf(pVideoOutputInfo); sts != RGY_ERR_NONE) {
            return sts;
        }

        if (rawPrm->hdrMetadataIn != nullptr && rawPrm->hdrMetadataIn->getprm().hasPrmSet()) {
            AddMessage(RGY_LOG_DEBUG, char_to_tstring(rawPrm->hdrMetadataIn->print()));
            if (rawPrm->codecId == RGY_CODEC_HEVC) {
                m_hdrBitstream = rawPrm->hdrMetadataIn->gen_nal();
            } else if (rawPrm->codecId == RGY_CODEC_AV1) {
                m_hdrBitstream = rawPrm->hdrMetadataIn->gen_obu();
            } else {
                AddMessage(RGY_LOG_ERROR, _T("Setting masterdisplay/contentlight not supported in %s encoding.\n"), CodecToStr(rawPrm->codecId).c_str());
                return RGY_ERR_UNSUPPORTED;
            }
        }
        m_hdr10plusMetadataCopy = rawPrm->hdr10plusMetadataCopy;
        m_doviProfileDst = rawPrm->doviProfile;
        m_doviRpu = rawPrm->doviRpu;
        m_doviRpuMetadataCopy = rawPrm->doviRpuMetadataCopy;
        m_doviRpuConvertParam = rawPrm->doviRpuConvertParam;
        m_timestamp = rawPrm->vidTimestamp;
        m_debugDirectAV1Out = rawPrm->debugDirectAV1Out;
        m_HEVCAlphaChannelMode = rawPrm->HEVCAlphaChannelMode;
        m_enableHEVCAlphaChannelInfoSEIOverwrite = rawPrm->codecId == RGY_CODEC_HEVC && rawPrm->HEVCAlphaChannel;
        if (m_enableHEVCAlphaChannelInfoSEIOverwrite) {
            AddMessage(RGY_LOG_DEBUG, _T("enableHEVCAlphaChannelInfoSEIFix : on\n"));
        }
        if (rawPrm->debugRawOut) {
            const auto filename_debug = m_outFilename + _T(".debug");
            m_fpDebug = std::unique_ptr<FILE, fp_deleter>(
                _tfopen(filename_debug.c_str(), _T("wb")), fp_deleter());
            if (!m_fpDebug) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open raw frame debug out file \"%s\".\n"), filename_debug.c_str());
                return RGY_ERR_FILE_OPEN;
            }
            AddMessage(RGY_LOG_INFO, _T("Raw frame debug out to file \"%s\".\n"), filename_debug.c_str());
        }
        if (!rawPrm->outReplayFile.empty()) {
            m_fpOutReplay = std::unique_ptr<FILE, fp_deleter>(
                _tfopen(rawPrm->outReplayFile.c_str(), _T("rb")), fp_deleter());
            if (!m_fpOutReplay) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open replay debug out from file \"%s\".\n"), rawPrm->outReplayFile.c_str());
                return RGY_ERR_FILE_OPEN;
            }

            AddMessage(RGY_LOG_WARN, _T("replay debug out from file \"%s\".\n"), rawPrm->outReplayFile.c_str());
            if (rawPrm->outReplayCodec != RGY_CODEC_UNKNOWN) {
                m_VideoOutputInfo.codec = rawPrm->outReplayCodec;
                AddMessage(RGY_LOG_WARN, _T("replay codec set to \"%s\".\n"), CodecToStr(m_VideoOutputInfo.codec).c_str());
            }
        }
    }
    m_inited = true;
    return RGY_ERR_NONE;
}
#pragma warning (pop)

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYBitstream *pBitstream) {
    if (pBitstream == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid call: WriteNextFrame\n"));
        return RGY_ERR_NULL_PTR;
    }

    readRawDebug(pBitstream);

    if (m_bsf) {
        auto sts = m_bsf->applyBitstreamFilter(pBitstream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    writeRawDebug(pBitstream);

    if (m_VideoOutputInfo.codec == RGY_CODEC_AV1) {
        const auto av1_units = parse_unit_av1(pBitstream->data(), pBitstream->size());
        const auto td_count = std::count_if(av1_units.begin(), av1_units.end(), [](const std::unique_ptr<unit_info>& info) { return info->type == OBU_TEMPORAL_DELIMITER; });
        if (td_count > 1) {
            RGYBitstream bsCopy = RGYBitstreamInit();
            for (int i = 0; i < (int)av1_units.size(); i++) {
                if (av1_units[i]->type == OBU_TEMPORAL_DELIMITER && bsCopy.size() > 0) {
                    WriteNextOneFrame(&bsCopy);
                }
                bsCopy.append(av1_units[i]->unit_data.data(), av1_units[i]->unit_data.size());
            }
            if (bsCopy.size() > 0) {
                return WriteNextOneFrame(&bsCopy);
            }
            return RGY_ERR_NONE;
        }
    }
    return WriteNextOneFrame(pBitstream);
}

RGY_ERR RGYOutputRaw::WriteNextOneFrame(RGYBitstream *pBitstream) {
    size_t nBytesWritten = 0;
    if (m_noOutput) {
        return RGY_ERR_NONE;
    }

    // NVENCのalpha_channel_info SEIの出力は変なので、適切なものに置き換える
    auto err = OverwriteHEVCAlphaChannelInfoSEI(pBitstream);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    RGYTimestampMapVal bs_framedata;
    if (m_timestamp) {
        bs_framedata = m_timestamp->get(pBitstream->pts());
        if (bs_framedata.inputFrameId < 0) {
            bs_framedata.inputFrameId = m_prevInputFrameId;
            AddMessage(RGY_LOG_WARN, _T("Failed to get frame ID for pts %lld, using %lld.\n"), pBitstream->pts(), bs_framedata.inputFrameId);
        }
        m_prevInputFrameId = bs_framedata.inputFrameId;
    }
    if (bs_framedata.inputFrameId < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get frame ID for pts %lld (%lld).\n"), pBitstream->pts(), bs_framedata.inputFrameId);
        return RGY_ERR_UNDEFINED_BEHAVIOR;
    }

    std::vector<std::unique_ptr<RGYOutputInsertMetadata>> metadataList;
    if (m_hdrBitstream.size() > 0) {
        std::vector<uint8_t> data(m_hdrBitstream.data(), m_hdrBitstream.data() + m_hdrBitstream.size());
        metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(data, (m_VideoOutputInfo.codec == RGY_CODEC_AV1) ? false : true, false));
    }
    if (m_hdr10plusMetadataCopy) {
        auto [err_hdr10plus, metadata_hdr10plus] = getMetadata<RGYFrameDataHDR10plus>(RGY_FRAME_DATA_HDR10PLUS, bs_framedata, nullptr);
        if (err_hdr10plus != RGY_ERR_NONE) {
            return err_hdr10plus;
        }
        if (metadata_hdr10plus.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(metadata_hdr10plus, false, false));
        }
    }
    if (m_doviRpu) {
        std::vector<uint8_t> dovi_nal;
        if (m_doviRpu->get_next_rpu(dovi_nal, m_doviProfileDst, &m_doviRpuConvertParam, bs_framedata.inputFrameId, m_VideoOutputInfo.codec) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get dovi rpu for %lld.\n"), bs_framedata.inputFrameId);
        }
        if (dovi_nal.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(dovi_nal, false, m_VideoOutputInfo.codec == RGY_CODEC_HEVC ? true : false));
        }
    } else if (m_doviRpuMetadataCopy) {
        auto doviRpuConvPrm = std::make_unique<RGYFrameDataDOVIRpuConvertParam>(m_doviProfileDst, m_doviRpuConvertParam);
        auto [err_dovirpu, metadata_dovi_rpu] = getMetadata<RGYFrameDataDOVIRpu>(RGY_FRAME_DATA_DOVIRPU, bs_framedata, doviRpuConvPrm.get());
        if (err_dovirpu != RGY_ERR_NONE) {
            return err_dovirpu;
        }
        if (metadata_dovi_rpu.size() > 0) {
            metadataList.push_back(std::make_unique<RGYOutputInsertMetadata>(metadata_dovi_rpu, false, m_VideoOutputInfo.codec == RGY_CODEC_HEVC ? true : false));
        }
    }

    err = InsertMetadata(pBitstream, metadataList);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    nBytesWritten = _fwrite_nolock(pBitstream->data(), 1, pBitstream->size(), m_fDest.get());
    WRITE_CHECK(nBytesWritten, pBitstream->size());

    m_encSatusInfo->SetOutputData(pBitstream->frametype(), nBytesWritten, 0);
    pBitstream->setSize(0);

    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYFrame *pSurface) {
    UNREFERENCED_PARAMETER(pSurface);
    return RGY_ERR_UNSUPPORTED;
}

#if ENCODER_QSV || ENCODER_NVENC

RGYOutFrame::RGYOutFrame() : m_bY4m(true) {
    m_strWriterName = _T("yuv writer");
    m_OutType = OUT_TYPE_SURFACE;
};

RGYOutFrame::~RGYOutFrame() {
};

RGY_ERR RGYOutFrame::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(pVideoOutputInfo);
    if (_tcscmp(strFileName, _T("-")) == 0) {
        m_fDest.reset(stdout);
        m_outputIsStdout = true;
        AddMessage(RGY_LOG_DEBUG, _T("using stdout\n"));
    } else {
        FILE *fp = NULL;
        int error = _tfopen_s(&fp, strFileName, _T("wb"));
        if (0 != error || fp == NULL) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to open file \"%s\": %s\n"), strFileName, _tcserror(error));
            return RGY_ERR_NULL_PTR;
        }
        m_fDest.reset(fp);
    }

    YUVWriterParam *writerParam = (YUVWriterParam *)prm;

    m_bY4m = writerParam->bY4m;
    m_sourceHWMem = true;
    m_inited = true;

    return RGY_ERR_NONE;
}

RGY_ERR RGYOutFrame::WriteNextFrame(RGYBitstream *pBitstream) {
    UNREFERENCED_PARAMETER(pBitstream);
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR RGYOutFrame::WriteNextFrame(RGYFrame *pSurface) {
    if (!m_fDest) {
        return RGY_ERR_NULL_PTR;
    }

    if (m_sourceHWMem) {
        if (m_readBuffer.get() == nullptr) {
            m_readBuffer.reset((uint8_t *)_aligned_malloc(pSurface->pitch() + 128, 16));
        }
    }

    if (m_bY4m) {
        if (!m_y4mHeaderWritten) {
            auto csp = pSurface->csp();
            if (csp == RGY_CSP_NV12) {
                csp = RGY_CSP_YV12;
            } else if (csp == RGY_CSP_P010) {
                csp = RGY_CSP_YV12_16;
            }
            WriteY4MHeader(m_fDest.get(), &m_VideoOutputInfo, csp);
            m_y4mHeaderWritten = true;
        }
        WRITE_CHECK(fwrite("FRAME\n", 1, strlen("FRAME\n"), m_fDest.get()), strlen("FRAME\n"));
    }

    auto loadLineToBuffer = [](uint8_t *ptrBuf, uint8_t *ptrSrc, const int pitch) {
#if (defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)) && ENCODER_QSV
        for (int i = 0; i < pitch; i += 128, ptrSrc += 128, ptrBuf += 128) {
            __m128i x0 = _mm_stream_load_si128((__m128i *)(ptrSrc +   0));
            __m128i x1 = _mm_stream_load_si128((__m128i *)(ptrSrc +  16));
            __m128i x2 = _mm_stream_load_si128((__m128i *)(ptrSrc +  32));
            __m128i x3 = _mm_stream_load_si128((__m128i *)(ptrSrc +  48));
            __m128i x4 = _mm_stream_load_si128((__m128i *)(ptrSrc +  64));
            __m128i x5 = _mm_stream_load_si128((__m128i *)(ptrSrc +  80));
            __m128i x6 = _mm_stream_load_si128((__m128i *)(ptrSrc +  96));
            __m128i x7 = _mm_stream_load_si128((__m128i *)(ptrSrc + 112));
            _mm_store_si128((__m128i *)(ptrBuf +   0), x0);
            _mm_store_si128((__m128i *)(ptrBuf +  16), x1);
            _mm_store_si128((__m128i *)(ptrBuf +  32), x2);
            _mm_store_si128((__m128i *)(ptrBuf +  48), x3);
            _mm_store_si128((__m128i *)(ptrBuf +  64), x4);
            _mm_store_si128((__m128i *)(ptrBuf +  80), x5);
            _mm_store_si128((__m128i *)(ptrBuf +  96), x6);
            _mm_store_si128((__m128i *)(ptrBuf + 112), x7);
        }
#else
        memcpy(ptrBuf, ptrSrc, pitch);
#endif
    };

    auto crop = initCrop();
#if ENCODER_QSV
    if (auto mfxsurf = dynamic_cast<RGYFrameMFXSurf*>(pSurface); mfxsurf) {
        crop = mfxsurf->crop();
    }
#endif
    const int pixSize = RGY_CSP_BIT_DEPTH[pSurface->csp()] > 8 ? 2 : 1;
    if (   RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV420
        || RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV444) {
        const uint32_t lumaWidthBytes = pSurface->width() * pixSize;
        const uint32_t cropOffset = crop.e.up * pSurface->pitch() + crop.e.left * pixSize;
        if (m_sourceHWMem) {
            for (decltype(pSurface->height()) j = 0; j < pSurface->height(); j++) {
                uint8_t *ptrBuf = m_readBuffer.get();
                uint8_t *ptrSrc = pSurface->ptrY() + (crop.e.up + j) * pSurface->pitch();
                loadLineToBuffer(ptrBuf, ptrSrc, pSurface->pitch());
                WRITE_CHECK(fwrite(ptrBuf + crop.e.left * pixSize, 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        } else {
            for (decltype(pSurface->height()) j = 0; j < pSurface->height(); j++) {
                WRITE_CHECK(fwrite(pSurface->ptrY() + cropOffset + j * pSurface->pitch(), 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("Unsupported colorspace %s.\n"), RGY_CSP_NAMES[pSurface->csp()]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    uint32_t frameSize = 0;
    if (   pSurface->csp() == RGY_CSP_NV12
        || pSurface->csp() == RGY_CSP_P010) {
        const uint32_t widthUV = pSurface->width() >> 1;
        const uint32_t heightUV = pSurface->height() >> 1;
        const uint32_t planeOffsetUV = ALIGN32((widthUV * heightUV + 32) * pixSize);
        if (!m_UVBuffer) {
            m_UVBuffer.reset((uint8_t *)_aligned_malloc(planeOffsetUV * 2, 32));
        }

        for (uint32_t j = 0; j < heightUV; j++) {
            uint8_t *ptrSrc = pSurface->ptrUV() + (crop.e.up + j) * pSurface->pitch();
            uint8_t *ptrBuf = ptrSrc;
            if (m_sourceHWMem) {
                loadLineToBuffer(m_readBuffer.get(), ptrSrc, pSurface->pitch());
                ptrBuf = m_readBuffer.get();
            }

            const void *ptrLineUV = ptrBuf + crop.e.left * pixSize;
            void *ptrLineU = m_UVBuffer.get() + j * widthUV * pixSize;
            void *ptrLineV = m_UVBuffer.get() + j * widthUV * pixSize + planeOffsetUV;
            if (pSurface->csp() == RGY_CSP_NV12) {
                const uint8_t *ptrUV = (const uint8_t *)ptrLineUV;
                uint8_t *ptrU = (uint8_t *)ptrLineU;
                uint8_t *ptrV = (uint8_t *)ptrLineV;
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
                alignas(16) static const uint16_t MASK_LOW8[] = {
                    0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff
                };
                const __m128i xMaskLow8 = _mm_load_si128((__m128i *)MASK_LOW8);

                for (uint32_t i = 0; i < widthUV; i += 16, ptrUV += 32, ptrU += 16, ptrV += 16) {
                    __m128i x0 = _mm_loadu_si128((const __m128i *)(ptrUV + 0));
                    __m128i x1 = _mm_loadu_si128((const __m128i *)(ptrUV + 16));
                    _mm_storeu_si128((__m128i *)ptrU, _mm_packus_epi16(_mm_and_si128(x0, xMaskLow8), _mm_and_si128(x1, xMaskLow8)));
                    _mm_storeu_si128((__m128i *)ptrV, _mm_packus_epi16(_mm_srli_epi16(x0, 8), _mm_srli_epi16(x1, 8)));
                }
#else
                convert_nv12_to_yv12_line_c<uint8_t, uint8_t, 8, 8>(ptrU, ptrV, ptrUV, widthUV);
#endif
            } else if (pSurface->csp() == RGY_CSP_P010) {
                const uint16_t *ptrUV = (const uint16_t *)ptrLineUV;
                uint16_t *ptrU = (uint16_t *)ptrLineU;
                uint16_t *ptrV = (uint16_t *)ptrLineV;
                switch (RGY_CSP_BIT_DEPTH[pSurface->csp()]) {
                case 10: convert_nv12_to_yv12_line_c<uint16_t, 10, uint16_t, 16>(ptrU, ptrV, ptrUV, widthUV); break;
                case 12: convert_nv12_to_yv12_line_c<uint16_t, 12, uint16_t, 16>(ptrU, ptrV, ptrUV, widthUV); break;
                case 14: convert_nv12_to_yv12_line_c<uint16_t, 14, uint16_t, 16>(ptrU, ptrV, ptrUV, widthUV); break;
                case 16:
                default: convert_nv12_to_yv12_line_c<uint16_t, 16, uint16_t, 16>(ptrU, ptrV, ptrUV, widthUV); break;
                }
            } else {
                return RGY_ERR_INVALID_COLOR_FORMAT;
            }
        }
        WRITE_CHECK(fwrite(m_UVBuffer.get(),                 1, widthUV * heightUV, m_fDest.get()), widthUV * heightUV);
        WRITE_CHECK(fwrite(m_UVBuffer.get() + planeOffsetUV, 1, widthUV * heightUV, m_fDest.get()), widthUV * heightUV);
    } else if (RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV420
            || RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV444) {
        uint8_t *const ptrBuf = m_readBuffer.get();

        for (int iplane = 1; iplane < RGY_CSP_PLANES[pSurface->csp()]; iplane++) {
            const uint32_t widthUV = pSurface->width() >> (RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV420 ? 1 : 0);
            const uint32_t heightUV = pSurface->height() >> (RGY_CSP_CHROMA_FORMAT[pSurface->csp()] == RGY_CHROMAFMT_YUV420 ? 1 : 0);
            for (uint32_t i = 0; i < heightUV; i++) {
                loadLineToBuffer(ptrBuf, pSurface->ptrPlane((RGY_PLANE)iplane) + (crop.e.up + i) * pSurface->pitch((RGY_PLANE)iplane), pSurface->pitch((RGY_PLANE)iplane));
                WRITE_CHECK(fwrite(ptrBuf + (crop.e.left * pixSize >> 1), 1, widthUV * pixSize, m_fDest.get()), widthUV * pixSize);
            }
        }
    } else {
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    m_encSatusInfo->SetOutputData(RGY_FRAMETYPE_IDR, frameSize, 0);
    return RGY_ERR_NONE;
}

#endif //#if ENCODER_QSV || ENCODER_NVENC

#include "rgy_input_sm.h"
#include "rgy_input_avcodec.h"
#include "rgy_output_avcodec.h"

std::unique_ptr<RGYHDRMetadata> createHEVCHDRSei(const std::string& maxCll, const std::string &masterDisplay, CspTransfer atcSei, const RGYInput *reader) {
    auto hdrMetadataIn = std::make_unique<RGYHDRMetadata>();
    const AVMasteringDisplayMetadata *masteringDisplaySrc = nullptr;
    const AVContentLightMetadata *contentLightSrc = nullptr;
    { auto avcodecReader = dynamic_cast<const RGYInputAvcodec *>(reader);
        if (avcodecReader != nullptr) {
            masteringDisplaySrc = avcodecReader->getMasteringDisplay();
            contentLightSrc = avcodecReader->getContentLight();
        }
    }
    int ret = 0;
    if (maxCll == maxCLLSource) {
        if (contentLightSrc != nullptr) {
            hdrMetadataIn->set_maxcll(contentLightSrc->MaxCLL, contentLightSrc->MaxFALL);
        }
    } else {
        ret = hdrMetadataIn->parse_maxcll(maxCll);
    }
    if (masterDisplay == masterDisplaySource) {
        if (masteringDisplaySrc != nullptr) {
            rgy_rational<int> masterdisplay[10];
            masterdisplay[RGYHDRMetadataPrmIndex::G_X] = to_rgy(masteringDisplaySrc->display_primaries[1][0]); //G
            masterdisplay[RGYHDRMetadataPrmIndex::G_Y] = to_rgy(masteringDisplaySrc->display_primaries[1][1]); //G
            masterdisplay[RGYHDRMetadataPrmIndex::B_X] = to_rgy(masteringDisplaySrc->display_primaries[2][0]); //B
            masterdisplay[RGYHDRMetadataPrmIndex::B_Y] = to_rgy(masteringDisplaySrc->display_primaries[2][1]); //B
            masterdisplay[RGYHDRMetadataPrmIndex::R_X] = to_rgy(masteringDisplaySrc->display_primaries[0][0]); //R
            masterdisplay[RGYHDRMetadataPrmIndex::R_Y] = to_rgy(masteringDisplaySrc->display_primaries[0][1]); //R
            masterdisplay[RGYHDRMetadataPrmIndex::WP_X] = to_rgy(masteringDisplaySrc->white_point[0]);
            masterdisplay[RGYHDRMetadataPrmIndex::WP_Y] = to_rgy(masteringDisplaySrc->white_point[1]);
            masterdisplay[RGYHDRMetadataPrmIndex::L_Max] = to_rgy(masteringDisplaySrc->max_luminance);
            masterdisplay[RGYHDRMetadataPrmIndex::L_Min] = to_rgy(masteringDisplaySrc->min_luminance);
            hdrMetadataIn->set_masterdisplay(masterdisplay);
        }
    } else {
        ret = hdrMetadataIn->parse_masterdisplay(masterDisplay);
    }
    if (atcSei != RGY_TRANSFER_UNKNOWN) {
        hdrMetadataIn->set_atcsei(atcSei);
    }
    if (ret) {
        hdrMetadataIn.reset();
    }
    return hdrMetadataIn;
}

static bool audioSelected(const AudioSelect *sel, const AVDemuxStream *stream) {
    if (sel->trackID == trackID(stream->trackId)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_LANG && rgy_lang_equal(sel->lang, stream->lang)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_CODEC && stream->stream != nullptr && avcodec_equal(sel->selectCodec, stream->stream->codecpar->codec_id)) {
        return true;
    }
    return false;
};
static bool subSelected(const SubtitleSelect *sel, const AVDemuxStream *stream) {
    if (sel->trackID == trackID(stream->trackId)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_LANG && rgy_lang_equal(sel->lang, stream->lang)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_CODEC && stream->stream != nullptr && avcodec_equal(sel->selectCodec, stream->stream->codecpar->codec_id)) {
        return true;
    }
    return false;
};
static bool dataSelected(const DataSelect *sel, const AVDemuxStream *stream) {
    if (sel->trackID == trackID(stream->trackId)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_LANG && rgy_lang_equal(sel->lang, stream->lang)) {
        return true;
    }
    if (sel->trackID == TRACK_SELECT_BY_CODEC && stream->stream != nullptr && avcodec_equal(sel->selectCodec, stream->stream->codecpar->codec_id)) {
        return true;
    }
    return false;
};

RGY_ERR initWriters(
    shared_ptr<RGYOutput> &pFileWriter,
    vector<shared_ptr<RGYOutput>>& pFileWriterListAudio,
    shared_ptr<RGYInput> &pFileReader,
    vector<shared_ptr<RGYInput>> &otherReaders,
    RGYParamCommon *common,
    const VideoInfo *input,
    const RGYParamControl *ctrl,
    const VideoInfo outputVideoInfo,
    const sTrimParam& trimParam,
    const rgy_rational<int> outputTimebase,
#if ENABLE_AVSW_READER
    const vector<unique_ptr<AVChapter>>& chapters,
#endif //#if ENABLE_AVSW_READER
    const RGYHDRMetadata *hdrMetadataIn,
    DOVIRpu *doviRpu,
    RGYTimestamp *vidTimestamp,
    const bool videoDtsUnavailable,
    const bool benchmark,
    const bool HEVCAlphaChannel,
    const int HEVCAlphaChannelMode,
    RGYPoolAVPacket *poolPkt,
    RGYPoolAVFrame *poolFrame,
    shared_ptr<EncodeStatus> pStatus,
    shared_ptr<CPerfMonitor> pPerfMonitor,
    shared_ptr<RGYLog> log
) {
    bool stdoutUsed = false;
#if ENABLE_AVSW_READER
    vector<int> streamTrackUsed; //使用した音声/字幕のトラックIDを保存する
    bool useH264ESOutput =
        ((common->muxOutputFormat.length() > 0 && 0 == _tcscmp(common->muxOutputFormat.c_str(), _T("raw")))) //--formatにrawが指定されている
        || std::filesystem::path(common->outputFilename).extension().empty() //拡張子がない
        || check_ext(common->outputFilename.c_str(), { ".m2v", ".264", ".h264", ".avc", ".avc1", ".x264", ".265", ".h265", ".hevc", ".vp9", ".av1", ".raw" }); //特定の拡張子
    if (!useH264ESOutput && outputVideoInfo.codec != RGY_CODEC_RAW) {
        common->AVMuxTarget |= RGY_MUX_VIDEO;
    }

    double inputFileDuration = 0.0;
    if (auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(pFileReader); pAVCodecReader != nullptr) {
        inputFileDuration = pAVCodecReader->GetInputVideoDuration();
    }
    bool isAfs = false;
#if ENABLE_SM_READER
    { auto pReaderSM = std::dynamic_pointer_cast<RGYInputSM>(pFileReader);
    if (pReaderSM) {
        isAfs = pReaderSM->isAfs();
    }
    }
#endif //#if ENABLE_SM_READER
    //if (inputParams->CodecId == MFX_CODEC_RAW) {
    //    inputParams->AVMuxTarget &= ~RGY_MUX_VIDEO;
    //}
    pStatus->Init(outputVideoInfo.fpsN, outputVideoInfo.fpsD, input->frames, inputFileDuration, trimParam, log, pPerfMonitor);
    if (ctrl->perfMonitorSelect || ctrl->perfMonitorSelectMatplot) {
        pPerfMonitor->SetEncStatus(pStatus);
    }

    bool audioCopyAll = false;
    if (common->AVMuxTarget & RGY_MUX_VIDEO) {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Using avformat writer.\n"));
        pFileWriter = std::make_shared<RGYOutputAvcodec>();
        AvcodecWriterPrm writerPrm;
        writerPrm.outputFormat            = common->muxOutputFormat;
        writerPrm.allowOtherNegativePts   = common->allowOtherNegativePts;
        writerPrm.timestampPassThrough    = common->timestampPassThrough;
        writerPrm.trimList                = trimParam.list;
        writerPrm.bVideoDtsUnavailable    = videoDtsUnavailable;
        writerPrm.threadOutput            = ctrl->threadOutput;
        writerPrm.threadAudio             = ctrl->threadAudio;
        writerPrm.threadParamOutput       = ctrl->threadParams.get(RGYThreadType::OUTUT);
        writerPrm.threadParamAudio        = ctrl->threadParams.get(RGYThreadType::AUDIO);
        writerPrm.bufSizeMB               = ctrl->outputBufSizeMB;
        writerPrm.audioResampler          = common->audioResampler;
        writerPrm.audioIgnoreDecodeError  = common->audioIgnoreDecodeError;
        writerPrm.queueInfo = (pPerfMonitor) ? pPerfMonitor->GetQueueInfoPtr() : nullptr;
        writerPrm.muxVidTsLogFile         = ctrl->logMuxVidTs.getFilename(common->outputFilename, _T(".muxts.log"));
        writerPrm.bitstreamTimebase       = av_make_q(outputTimebase);
        writerPrm.chapterNoTrim           = common->chapterNoTrim;
        writerPrm.attachments             = common->attachmentSource;
        writerPrm.hdrMetadataIn           = hdrMetadataIn;
        writerPrm.hdr10plusMetadataCopy   = common->hdr10plusMetadataCopy || common->dynamicHdr10plusJson.length() > 0;
        writerPrm.doviRpu                 = doviRpu;
        writerPrm.doviRpuMetadataCopy     = common->doviRpuMetadataCopy;
        writerPrm.doviProfile             = common->doviProfile;
        writerPrm.doviRpuConvertParam     = common->doviRpuParams;
        writerPrm.vidTimestamp            = vidTimestamp;
        writerPrm.videoCodecTag           = common->videoCodecTag;
        writerPrm.videoMetadata           = common->videoMetadata;
        writerPrm.formatMetadata          = common->formatMetadata;
        writerPrm.afs                     = isAfs;
        writerPrm.disableMp4Opt           = common->disableMp4Opt;
        writerPrm.lowlatency              = ctrl->lowLatency;
        writerPrm.debugDirectAV1Out       = common->debugDirectAV1Out;
        writerPrm.HEVCAlphaChannel        = HEVCAlphaChannel;
        writerPrm.HEVCAlphaChannelMode    = HEVCAlphaChannelMode;
        writerPrm.muxOpt                  = common->muxOpt;
        writerPrm.poolPkt                 = poolPkt;
        writerPrm.poolFrame               = poolFrame;
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(pFileReader);
        if (pAVCodecReader != nullptr) {
            writerPrm.inputFormatMetadata = pAVCodecReader->GetInputFormatMetadata();
            writerPrm.videoInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
            writerPrm.videoInputStream = pAVCodecReader->GetInputVideoStream();
        }
        if (chapters.size() > 0 && (common->copyChapter || common->chapterFile.length() > 0)) {
            writerPrm.chapterList.clear();
            for (uint32_t i = 0; i < chapters.size(); i++) {
                writerPrm.chapterList.push_back(chapters[i].get());
            }
        }
        if (common->AVMuxTarget & (RGY_MUX_AUDIO | RGY_MUX_SUBTITLE)) {
            log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Audio/Subtitle muxing enabled.\n"));
            for (int i = 0; !audioCopyAll && i < common->nAudioSelectCount; i++) {
                //トラック"0"が指定されていれば、すべてのトラックをコピーするということ
                audioCopyAll = (common->ppAudioSelectList[i]->trackID == 0);
            }
            log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: CopyAll=%s\n"), (audioCopyAll) ? _T("true") : _T("false"));
            pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(pFileReader);
            vector<AVDemuxStream> streamList = pFileReader->GetInputStreamInfo();

            for (auto& stream : streamList) {
                const auto streamMediaType = trackMediaType(stream.trackId);
                //audio-fileで別ファイルとして抽出するものは除く
                bool usedInAudioFile = false;
                for (int i = 0; i < (int)common->nAudioSelectCount; i++) {
                    if (audioSelected(common->ppAudioSelectList[i], &stream)
                        && common->ppAudioSelectList[i]->extractFilename.length() > 0) {
                        usedInAudioFile = true;
                    }
                }
                if (usedInAudioFile) {
                    continue;
                }
                const AudioSelect *pAudioSelect = nullptr;
                if (streamMediaType == AVMEDIA_TYPE_AUDIO) {
                    for (int i = 0; i < (int)common->nAudioSelectCount; i++) {
                        if (audioSelected(common->ppAudioSelectList[i], &stream)
                            && common->ppAudioSelectList[i]->extractFilename.length() == 0) {
                            pAudioSelect = common->ppAudioSelectList[i];
                            break;
                        }
                    }
                    if (pAudioSelect == nullptr) {
                        //一致するTrackIDがなければ、trackID = 0 (全指定)を探す
                        for (int i = 0; i < common->nAudioSelectCount; i++) {
                            if (common->ppAudioSelectList[i]->trackID == 0
                                && common->ppAudioSelectList[i]->extractFilename.length() == 0) {
                                pAudioSelect = common->ppAudioSelectList[i];
                                break;
                            }
                        }
                    }
                }
                const SubtitleSelect *pSubtitleSelect = nullptr;
                if (streamMediaType == AVMEDIA_TYPE_SUBTITLE) {
                    for (int i = 0; i < common->nSubtitleSelectCount; i++) {
                        if (subSelected(common->ppSubtitleSelectList[i], &stream)) {
                            pSubtitleSelect = common->ppSubtitleSelectList[i];
                            break;
                        }
                    }
                    if (pSubtitleSelect == nullptr) {
                        //一致するTrackIDがなければ、trackID = 0 (全指定)を探す
                        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
                            if (common->ppSubtitleSelectList[i]->trackID == 0) {
                                pSubtitleSelect = common->ppSubtitleSelectList[i];
                                break;
                            }
                        }
                    }
                }
                const DataSelect *pDataSelect = nullptr;
                if (streamMediaType == AVMEDIA_TYPE_DATA) {
                    for (int i = 0; i < common->nDataSelectCount; i++) {
                        if (dataSelected(common->ppDataSelectList[i], &stream)) {
                            pDataSelect = common->ppDataSelectList[i];
                        }
                    }
                    if (pSubtitleSelect == nullptr) {
                        //一致するTrackIDがなければ、trackID = 0 (全指定)を探す
                        for (int i = 0; i < common->nDataSelectCount; i++) {
                            if (common->ppDataSelectList[i]->trackID == 0) {
                                pDataSelect = common->ppDataSelectList[i];
                                break;
                            }
                        }
                    }
                }
                if (pAudioSelect != nullptr || audioCopyAll || streamMediaType != AVMEDIA_TYPE_AUDIO) {
                    streamTrackUsed.push_back(stream.trackId);
                    if (pSubtitleSelect == nullptr && streamMediaType == AVMEDIA_TYPE_SUBTITLE) {
                        continue;
                    }
                    AVOutputStreamPrm prm;
                    prm.src = stream;
                    //pAudioSelect == nullptrは "copyAllStreams" か 字幕ストリーム によるもの
                    if (pAudioSelect != nullptr) {
                        prm.decodeCodecPrm = pAudioSelect->decCodecPrm;
                        prm.bitrate = pAudioSelect->encBitrate;
                        prm.quality = pAudioSelect->encQuality;
                        prm.samplingRate = pAudioSelect->encSamplingRate;
                        prm.encodeCodec = pAudioSelect->encCodec;
                        prm.encodeCodecPrm = pAudioSelect->encCodecPrm;
                        prm.encodeCodecProfile = pAudioSelect->encCodecProfile;
                        prm.filter = pAudioSelect->filter;
                        prm.bsf = pAudioSelect->bsf;
                        prm.disposition = pAudioSelect->disposition;
                        prm.metadata = pAudioSelect->metadata;
                        prm.resamplerPrm = pAudioSelect->resamplerPrm;
                    }
                    if (pSubtitleSelect != nullptr) {
                        prm.decodeCodecPrm = pSubtitleSelect->decCodecPrm;
                        prm.encodeCodec = pSubtitleSelect->encCodec;
                        prm.encodeCodecPrm = pSubtitleSelect->encCodecPrm;
                        prm.asdata = pSubtitleSelect->asdata;
                        prm.bsf = pSubtitleSelect->bsf;
                        prm.disposition = pSubtitleSelect->disposition;
                        prm.metadata = pSubtitleSelect->metadata;
                    }
                    if (pDataSelect != nullptr) {
                        prm.disposition = pDataSelect->disposition;
                        prm.metadata = pDataSelect->metadata;
                    }
                    log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Added %s track#%d (stream idx %d) for mux, bitrate %d, quality %s, codec: %s %s %s, bsf: %s, disposition: %s, metadata %s\n"),
                        char_to_tstring(av_get_media_type_string(streamMediaType)).c_str(),
                        stream.trackId, stream.index, prm.bitrate,
                        prm.quality.first ? strsprintf("%d", prm.quality.second).c_str() : "unset",
                        prm.encodeCodec.c_str(),
                        prm.encodeCodecProfile.c_str(),
                        prm.encodeCodecPrm.c_str(),
                        prm.bsf.length() > 0 ? prm.bsf.c_str() : _T("<none>"),
                        prm.disposition.length() > 0 ? prm.disposition.c_str() : _T("<copy>"),
                        prm.metadata.size() > 0 ? print_metadata(prm.metadata).c_str() : _T("<copy>"));
                    writerPrm.inputStreamList.push_back(std::move(prm));
                }
            }
            vector<AVDemuxStream> otherSrcStreams;
            for (const auto &reader : otherReaders) {
                if (reader->GetAudioTrackCount() > 0 || reader->GetSubtitleTrackCount() > 0) {
                    auto pAVCodecAudioReader = std::dynamic_pointer_cast<RGYInputAvcodec>(reader);
                    if (pAVCodecAudioReader) {
                        vector_cat(otherSrcStreams, pAVCodecAudioReader->GetInputStreamInfo());
                    }
                    //もしavqsvリーダーでないなら、音声リーダーから情報を取得する必要がある
                    if (pAVCodecReader == nullptr) {
                        writerPrm.videoInputFirstKeyPts = pAVCodecAudioReader->GetVideoFirstKeyPts();
                        writerPrm.videoInputStream = pAVCodecAudioReader->GetInputVideoStream();
                    }
                }
            }
            for (auto &stream : otherSrcStreams) {
                const auto streamMediaType = trackMediaType(stream.trackId);
                if (stream.sourceFileIndex < 0) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Internal Error, Invalid file index %d set for %s-source.\n"),
                        stream.sourceFileIndex, char_to_tstring(av_get_media_type_string(streamMediaType)).c_str());
                    return RGY_ERR_UNKNOWN;
                }
                //audio-fileで別ファイルとして抽出するものは除く
                if (streamMediaType == AVMEDIA_TYPE_AUDIO) {
                    bool usedInAudioFile = false;
                    const auto& audsrc = common->audioSource[stream.sourceFileIndex];
                    for (const auto& audsel : audsrc.select) {
                        if (audioSelected(&audsel.second, &stream)
                            && audsel.second.extractFilename.length() > 0) {
                            usedInAudioFile = true;
                        }
                    }
                    if (usedInAudioFile) {
                        continue;
                    }
                }
                const AudioSelect *pAudioSelect = nullptr;
                if (streamMediaType == AVMEDIA_TYPE_AUDIO) {
                    if (stream.sourceFileIndex >= (int)common->audioSource.size()) {
                        log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Internal Error, Invalid file index %d set for audio-source.\n"), stream.sourceFileIndex);
                        return RGY_ERR_UNKNOWN;
                    }
                    const auto& audsrc = common->audioSource[stream.sourceFileIndex];
                    for (const auto &audsel : audsrc.select) {
                        if (audioSelected(&audsel.second, &stream)) {
                            pAudioSelect = &audsel.second;
                            break;
                        }
                    }
                    if (pAudioSelect == nullptr) {
                        //一致するTrackIDがなければ、trackID = 0 (全指定)を探す
                        for (const auto& audsel : audsrc.select) {
                            if (audsel.first == 0) {
                                pAudioSelect = &audsel.second;
                                break;
                            }
                        }
                    }
                }
                const SubtitleSelect *pSubtitleSelect = nullptr;
                if (streamMediaType == AVMEDIA_TYPE_SUBTITLE) {
                    if (stream.sourceFileIndex >= (int)common->subSource.size()) {
                        log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Internal Error, Invalid file index %d set for audio-source.\n"), stream.sourceFileIndex);
                        return RGY_ERR_UNKNOWN;
                    }
                    const auto& subsrc = common->subSource[stream.sourceFileIndex];
                    for (const auto &subsel : subsrc.select) {
                        if (subSelected(&subsel.second, &stream)) {
                            pSubtitleSelect = &subsel.second;
                            break;
                        }
                    }
                    if (pSubtitleSelect == nullptr) {
                        //一致するTrackIDがなければ、trackID = 0 (全指定)を探す
                        for (const auto &subsel : subsrc.select) {
                            if (subsel.first == 0) {
                                pSubtitleSelect = &subsel.second;
                                break; //2重ループをbreak
                            }
                        }
                    }
                }
                if (pAudioSelect != nullptr || audioCopyAll || streamMediaType != AVMEDIA_TYPE_AUDIO) {
                    streamTrackUsed.push_back(stream.trackId);
                    AVOutputStreamPrm prm;
                    prm.src = stream;
                    //pAudioSelect == nullptrは "copyAllStreams" か 字幕ストリーム によるもの
                    if (pAudioSelect != nullptr) {
                        prm.decodeCodecPrm = pAudioSelect->decCodecPrm;
                        prm.bitrate = pAudioSelect->encBitrate;
                        prm.quality = pAudioSelect->encQuality;
                        prm.samplingRate = pAudioSelect->encSamplingRate;
                        prm.encodeCodec = pAudioSelect->encCodec;
                        prm.encodeCodecPrm = pAudioSelect->encCodecPrm;
                        prm.encodeCodecProfile = pAudioSelect->encCodecProfile;
                        prm.filter = pAudioSelect->filter;
                        prm.bsf = pAudioSelect->bsf;
                        prm.disposition = pAudioSelect->disposition;
                        prm.metadata = pAudioSelect->metadata;
                        prm.resamplerPrm = pAudioSelect->resamplerPrm;
                    }
                    if (pSubtitleSelect != nullptr) {
                        prm.decodeCodecPrm = pSubtitleSelect->decCodecPrm;
                        prm.encodeCodec = pSubtitleSelect->encCodec;
                        prm.encodeCodecPrm = pSubtitleSelect->encCodecPrm;
                        prm.asdata = pSubtitleSelect->asdata;
                        prm.bsf = pSubtitleSelect->bsf;
                        prm.disposition = pSubtitleSelect->disposition;
                        prm.metadata = pSubtitleSelect->metadata;
                    }
                    log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Added %s track#%d (stream idx %d) for mux, bitrate %d, quality %s, codec: %s %s %s, bsf: %s, disposition: %s, metadata: %s\n"),
                        char_to_tstring(av_get_media_type_string(streamMediaType)).c_str(),
                        stream.trackId, stream.index, prm.bitrate,
                        prm.quality.first ? strsprintf("%d", prm.quality.second).c_str() : "unset",
                        prm.encodeCodec.c_str(),
                        prm.encodeCodecProfile.c_str(),
                        prm.encodeCodecPrm.c_str(),
                        prm.bsf.length() > 0 ? prm.bsf.c_str() : _T("<none>"),
                        prm.disposition.length() > 0 ? prm.disposition.c_str() : _T("<copy>"),
                        prm.metadata.size() > 0 ? print_metadata(prm.metadata).c_str() : _T("<copy>"));
                    writerPrm.inputStreamList.push_back(std::move(prm));
                }
            }
            vector_cat(streamList, otherSrcStreams);
        }
        auto sts = pFileWriter->Init(common->outputFilename.c_str(), &outputVideoInfo, &writerPrm, log, pStatus);
        if (sts != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, pFileWriter->GetOutputMessage());
            return sts;
        } else if (common->AVMuxTarget & (RGY_MUX_AUDIO | RGY_MUX_SUBTITLE)) {
            pFileWriterListAudio.push_back(pFileWriter);
        }
        stdoutUsed = pFileWriter->outputStdout();
        log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Initialized avformat writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
    } else if (common->AVMuxTarget & (RGY_MUX_AUDIO | RGY_MUX_SUBTITLE)) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Audio mux cannot be used alone, should be use with video mux.\n"));
        return RGY_ERR_UNKNOWN;
    } else {
#endif //ENABLE_AVSW_READER
#if ENCODER_QSV || ENCODER_NVENC
        if (outputVideoInfo.codec == RGY_CODEC_RAW) {
            pFileWriter = std::make_shared<RGYOutFrame>();
            YUVWriterParam param;
            param.bY4m = common->muxOutputFormat != _T("raw");
            auto sts = pFileWriter->Init(common->outputFilename.c_str(), &outputVideoInfo, &param, log, pStatus);
            if (sts != RGY_ERR_NONE) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, pFileWriter->GetOutputMessage());
                return sts;
            }
            stdoutUsed = pFileWriter->outputStdout();
            log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Initialized yuv frame writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
        } else
#endif
        {
            pFileWriter = std::make_shared<RGYOutputRaw>();
            RGYOutputRawPrm rawPrm;
            rawPrm.bufSizeMB = ctrl->outputBufSizeMB;
            rawPrm.benchmark = benchmark;
            rawPrm.codecId = outputVideoInfo.codec;
            rawPrm.hdrMetadataIn = hdrMetadataIn;
            rawPrm.hdr10plusMetadataCopy = common->hdr10plusMetadataCopy || common->dynamicHdr10plusJson.length() > 0;
            rawPrm.doviProfile = common->doviProfile;
            rawPrm.doviRpu = doviRpu;
            rawPrm.doviRpuMetadataCopy = common->doviRpuMetadataCopy;
            rawPrm.doviRpuConvertParam = common->doviRpuParams;
            rawPrm.vidTimestamp = vidTimestamp;
            rawPrm.debugDirectAV1Out = common->debugDirectAV1Out;
            rawPrm.HEVCAlphaChannel = HEVCAlphaChannel;
            rawPrm.HEVCAlphaChannelMode = HEVCAlphaChannelMode;
            rawPrm.debugRawOut = common->debugRawOut;
            rawPrm.outReplayFile = common->outReplayFile;
            rawPrm.outReplayCodec = common->outReplayCodec;
            auto sts = pFileWriter->Init(common->outputFilename.c_str(), &outputVideoInfo, &rawPrm, log, pStatus);
            if (sts != RGY_ERR_NONE) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, pFileWriter->GetOutputMessage());
                return sts;
            }
            stdoutUsed = pFileWriter->outputStdout();
            log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Initialized bitstream writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
        }
#if ENABLE_AVSW_READER
    }

    //音声の抽出(--audio-file)
    const bool hasAudioExtract = std::find_if(common->ppAudioSelectList, common->ppAudioSelectList + common->nAudioSelectCount,
        [](const AudioSelect *pAudioSelect) { return pAudioSelect->extractFilename.length() > 0; }) != common->ppAudioSelectList + common->nAudioSelectCount;
    if (hasAudioExtract
        && common->nAudioSelectCount + common->nSubtitleSelectCount - (audioCopyAll ? 1 : 0) > (int)streamTrackUsed.size()) {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Audio file output enabled.\n"));
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(pFileReader);
        if (pAVCodecReader == nullptr) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Audio output is only supported with transcoding (avhw/avsw reader).\n"));
            return RGY_ERR_INVALID_PARAM;
        } else {
            auto inutAudioInfoList = pAVCodecReader->GetInputStreamInfo();
            for (auto& audioTrack : inutAudioInfoList) {
                bool bTrackAlreadyUsed = false;
                for (auto usedTrack : streamTrackUsed) {
                    if (usedTrack == audioTrack.trackId) {
                        bTrackAlreadyUsed = true;
                        log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Audio track #%d is already set to be muxed, so cannot be extracted to file.\n"), trackID(audioTrack.trackId));
                        break;
                    }
                }
                if (bTrackAlreadyUsed) {
                    continue;
                }
                const AudioSelect *pAudioSelect = nullptr;
                for (int i = 0; i < (int)common->nAudioSelectCount; i++) {
                    if (audioSelected(common->ppAudioSelectList[i], &audioTrack)
                        && common->ppAudioSelectList[i]->extractFilename.length() > 0) {
                        pAudioSelect = common->ppAudioSelectList[i];
                    }
                }
                if (pAudioSelect == nullptr) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Audio track #%d is not used anyware, this should not happen.\n"), trackID(audioTrack.trackId));
                    return RGY_ERR_UNKNOWN;
                }
                log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Output audio track #%d (stream index %d) to \"%s\", format: %s, codec %s, bitrate %d\n"),
                    trackID(audioTrack.trackId), audioTrack.index, pAudioSelect->extractFilename.c_str(), pAudioSelect->extractFormat.c_str(), pAudioSelect->encCodec.c_str(), pAudioSelect->encBitrate);

                AVOutputStreamPrm prm;
                prm.src = audioTrack;
                //pAudioSelect == nullptrは "copyAll" によるもの
                prm.decodeCodecPrm = pAudioSelect->decCodecPrm;
                prm.bitrate = pAudioSelect->encBitrate;
                prm.quality = pAudioSelect->encQuality;
                prm.samplingRate = pAudioSelect->encSamplingRate;
                prm.encodeCodec = pAudioSelect->encCodec;
                prm.encodeCodecPrm = pAudioSelect->encCodecPrm;
                prm.encodeCodecProfile = pAudioSelect->encCodecProfile;
                prm.filter = pAudioSelect->filter;
                prm.bsf = pAudioSelect->bsf;
                prm.disposition = pAudioSelect->disposition;
                prm.metadata = pAudioSelect->metadata;
                prm.resamplerPrm = pAudioSelect->resamplerPrm;

                AvcodecWriterPrm writerAudioPrm;
                writerAudioPrm.threadOutput   = ctrl->threadOutput;
                writerAudioPrm.threadAudio    = ctrl->threadAudio;
                writerAudioPrm.threadParamOutput = ctrl->threadParams.get(RGYThreadType::OUTUT);
                writerAudioPrm.threadParamAudio  = ctrl->threadParams.get(RGYThreadType::AUDIO);
                writerAudioPrm.bufSizeMB      = ctrl->outputBufSizeMB;
                writerAudioPrm.outputFormat   = pAudioSelect->extractFormat;
                writerAudioPrm.audioIgnoreDecodeError = common->audioIgnoreDecodeError;
                writerAudioPrm.lowlatency = ctrl->lowLatency;
                writerAudioPrm.audioResampler = common->audioResampler;
                writerAudioPrm.inputStreamList.push_back(prm);
                writerAudioPrm.trimList = trimParam.list;
                writerAudioPrm.videoInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
                writerAudioPrm.videoInputStream = pAVCodecReader->GetInputVideoStream();
                writerAudioPrm.bitstreamTimebase = av_make_q(outputTimebase);
                writerAudioPrm.poolPkt = poolPkt;
                writerAudioPrm.poolFrame = poolFrame;

                shared_ptr<RGYOutput> pWriter = std::make_shared<RGYOutputAvcodec>();
                auto sts = pWriter->Init(pAudioSelect->extractFilename.c_str(), nullptr, &writerAudioPrm, log, pStatus);
                if (sts != RGY_ERR_NONE) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, pWriter->GetOutputMessage());
                    return sts;
                }
                log->write(RGY_LOG_DEBUG, RGY_LOGT_OUT, _T("Output: Intialized audio output for track #%d.\n"), trackID(audioTrack.trackId));
                bool audioStdout = pWriter->outputStdout();
                if (stdoutUsed && audioStdout) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_OUT, _T("Multiple stream outputs are set to stdout, please remove conflict.\n"));
                    return RGY_ERR_UNKNOWN;
                }
                stdoutUsed |= audioStdout;
                pFileWriterListAudio.push_back(std::move(pWriter));
            }
        }
    }
#endif //ENABLE_AVSW_READER
    return RGY_ERR_NONE;
}
