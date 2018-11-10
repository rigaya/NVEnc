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
#include "rgy_bitstream.h"

#define WRITE_CHECK(writtenBytes, expected) { \
    if (writtenBytes != expected) { \
        AddMessage(RGY_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\n")); \
        return RGY_ERR_UNDEFINED_BEHAVIOR; \
    } }

RGYOutput::RGYOutput() :
    m_pEncSatusInfo(),
    m_fDest(),
    m_bOutputIsStdout(false),
    m_bInited(false),
    m_bNoOutput(false),
    m_OutType(OUT_TYPE_BITSTREAM),
    m_bSourceHWMem(false),
    m_bY4mHeaderWritten(false),
    m_strWriterName(),
    m_strOutputInfo(),
    m_VideoOutputInfo(),
    m_pPrintMes(),
    m_pOutputBuffer(),
    m_pReadBuffer(),
    m_pUVBuffer() {
    memset(&m_VideoOutputInfo, 0, sizeof(m_VideoOutputInfo));
}

RGYOutput::~RGYOutput() {
    m_pEncSatusInfo.reset();
    m_pPrintMes.reset();
    Close();
}

void RGYOutput::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    if (m_fDest) {
        m_fDest.reset();
        AddMessage(RGY_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    m_pEncSatusInfo.reset();
    m_pOutputBuffer.reset();
    m_pReadBuffer.reset();
    m_pUVBuffer.reset();

    m_bNoOutput = false;
    m_bInited = false;
    m_bSourceHWMem = false;
    m_bY4mHeaderWritten = false;
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
    m_pPrintMes.reset();
}

RGYOutputRaw::RGYOutputRaw() :
    m_seiNal()
#if ENABLE_AVSW_READER
    , m_pBsfc()
#endif //#if ENABLE_AVSW_READER
{
    m_strWriterName = _T("bitstream");
    m_OutType = OUT_TYPE_BITSTREAM;
}

RGYOutputRaw::~RGYOutputRaw() {
#if ENABLE_AVSW_READER
    m_pBsfc.reset();
#endif //#if ENABLE_AVSW_READER
}

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYOutputRaw::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(pVideoOutputInfo);
    RGYOutputRawPrm *rawPrm = (RGYOutputRawPrm *)prm;
    if (!rawPrm->bBenchmark && _tcslen(strFileName) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("output filename not set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (rawPrm->bBenchmark) {
        m_bNoOutput = true;
        AddMessage(RGY_LOG_DEBUG, _T("no output for benchmark mode.\n"));
    } else {
        if (_tcscmp(strFileName, _T("-")) == 0) {
            m_fDest.reset(stdout);
            m_bOutputIsStdout = true;
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

            int bufferSizeByte = clamp(rawPrm->nBufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
            if (bufferSizeByte) {
                void *ptr = nullptr;
                bufferSizeByte = (int)malloc_degeneracy(&ptr, bufferSizeByte, 1024 * 1024);
                if (bufferSizeByte) {
                    m_pOutputBuffer.reset((char*)ptr);
                    setvbuf(m_fDest.get(), m_pOutputBuffer.get(), _IOFBF, bufferSizeByte);
                    AddMessage(RGY_LOG_DEBUG, _T("Added %d MB output buffer.\n"), bufferSizeByte / (1024 * 1024));
                }
            }
        }
#if ENABLE_AVSW_READER
        if (ENCODER_NVENC
            && (pVideoOutputInfo->codec == RGY_CODEC_H264 || pVideoOutputInfo->codec == RGY_CODEC_HEVC)
            && pVideoOutputInfo->sar[0] * pVideoOutputInfo->sar[1] > 0) {
            if (!check_avcodec_dll()) {
                AddMessage(RGY_LOG_ERROR, error_mes_avcodec_dll_not_found());
                return RGY_ERR_NULL_PTR;
            }

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
            AVBSFContext *bsfc = nullptr;
            if (0 > (ret = av_bsf_alloc(filter, &bsfc))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            if (0 > (ret = avcodec_parameters_copy(bsfc->par_in, codecpar.get()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            m_pBsfc = unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>>(bsfc, RGYAVDeleter<AVBSFContext>(av_bsf_free));
            AVDictionary *bsfPrm = nullptr;
            char sar[128];
            sprintf_s(sar, "%d/%d", pVideoOutputInfo->sar[0], pVideoOutputInfo->sar[1]);
            av_dict_set(&bsfPrm, "sample_aspect_ratio", sar, 0);
            AddMessage(RGY_LOG_DEBUG, _T("set sar %d:%d by %s filter\n"), pVideoOutputInfo->sar[0], pVideoOutputInfo->sar[1], bsf_name);
            if (0 > (ret = av_opt_set_dict2(m_pBsfc.get(), &bsfPrm, AV_OPT_SEARCH_CHILDREN))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set parameters for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            if (0 > (ret = av_bsf_init(m_pBsfc.get()))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
                return RGY_ERR_UNKNOWN;
            }
            AddMessage(RGY_LOG_DEBUG, _T("initialized %s filter\n"), bsf_name);
        }
#endif //#if ENABLE_AVSW_READER
        if (rawPrm->codecId == RGY_CODEC_HEVC) {
            m_seiNal = rawPrm->seiNal;
        }
    }
    m_bInited = true;
    return RGY_ERR_NONE;
}
#pragma warning (pop)

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYBitstream *pBitstream) {
    if (pBitstream == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid call: WriteNextFrame\n"));
        return RGY_ERR_NULL_PTR;
    }

    size_t nBytesWritten = 0;
    if (!m_bNoOutput) {
#if ENABLE_AVSW_READER
        if (m_pBsfc) {
            std::vector<nal_info> nal_list;
            if (m_VideoOutputInfo.codec == RGY_CODEC_HEVC) {
                nal_list = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
            } else if (m_VideoOutputInfo.codec == RGY_CODEC_H264) {
                nal_list = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
            }
            auto sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
            if (sps_nal != nal_list.end()) {
                AVPacket pkt = { 0 };
                av_init_packet(&pkt);
                av_new_packet(&pkt, (int)sps_nal->size);
                memcpy(pkt.data, sps_nal->ptr, sps_nal->size);
                int ret = 0;
                if (0 > (ret = av_bsf_send_packet(m_pBsfc.get(), &pkt))) {
                    av_packet_unref(&pkt);
                    AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
                        char_to_tstring(m_pBsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_UNKNOWN;
                }
                ret = av_bsf_receive_packet(m_pBsfc.get(), &pkt);
                if (ret == AVERROR(EAGAIN)) {
                    return RGY_ERR_NONE;
                } else if ((ret < 0 && ret != AVERROR_EOF) || pkt.size < 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
                        char_to_tstring(m_pBsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
                    return RGY_ERR_UNKNOWN;
                }
                const auto new_data_size = pBitstream->size() + pkt.size - sps_nal->size;
                const auto sps_nal_offset = (int)(sps_nal->ptr - pBitstream->data());
                const auto next_nal_orig_offset = sps_nal_offset + sps_nal->size;
                const auto next_nal_new_offset = sps_nal_offset + pkt.size;
                const auto stream_orig_length = pBitstream->size();
                if ((decltype(new_data_size))pBitstream->bufsize() < new_data_size) {
                    pBitstream->changeSize(new_data_size);
                } else if (pkt.size > sps_nal->size) {
                    pBitstream->trim();
                }
                memmove(pBitstream->data() + next_nal_new_offset, pBitstream->data() + next_nal_orig_offset, stream_orig_length - next_nal_orig_offset);
                memcpy(pBitstream->data() + sps_nal_offset, pkt.data, pkt.size);
                av_packet_unref(&pkt);
            }
        }
#endif //#if ENABLE_AVSW_READER
        if (m_seiNal.size()) {
            const auto nal_list     = parse_nal_unit_hevc(pBitstream->data(), pBitstream->size());
            const auto hevc_vps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_VPS; });
            const auto hevc_sps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_SPS; });
            const auto hevc_pps_nal = std::find_if(nal_list.begin(), nal_list.end(), [](nal_info info) { return info.type == NALU_HEVC_PPS; });
            const bool header_check = (nal_list.end() != hevc_vps_nal) && (nal_list.end() != hevc_sps_nal) && (nal_list.end() != hevc_pps_nal);
            if (header_check) {
                nBytesWritten  = _fwrite_nolock(hevc_vps_nal->ptr, 1, hevc_vps_nal->size, m_fDest.get());
                nBytesWritten += _fwrite_nolock(hevc_sps_nal->ptr, 1, hevc_sps_nal->size, m_fDest.get());
                nBytesWritten += _fwrite_nolock(hevc_pps_nal->ptr, 1, hevc_pps_nal->size, m_fDest.get());
                nBytesWritten += _fwrite_nolock(m_seiNal.data(),   1, m_seiNal.size(),    m_fDest.get());
                for (const auto& nal : nal_list) {
                    if (nal.type != NALU_HEVC_VPS && nal.type != NALU_HEVC_SPS && nal.type != NALU_HEVC_PPS) {
                        nBytesWritten += _fwrite_nolock(nal.ptr, 1, nal.size, m_fDest.get());
                    }
                }
            } else {
                AddMessage(RGY_LOG_ERROR, _T("Unexpected HEVC header.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            m_seiNal.clear();
        } else {
            nBytesWritten = _fwrite_nolock(pBitstream->data(), 1, pBitstream->size(), m_fDest.get());
            WRITE_CHECK(nBytesWritten, pBitstream->size());
        }
    }

    m_pEncSatusInfo->SetOutputData(pBitstream->frametype(), pBitstream->size(), 0);
    pBitstream->setSize(0);

    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYFrame *pSurface) {
    UNREFERENCED_PARAMETER(pSurface);
    return RGY_ERR_UNSUPPORTED;
}
