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

#include "rgy_input_avs.h"
#if ENABLE_AVISYNTH_READER

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR *avisynth_dll_name = _T("avisynth.dll");
#else
static const TCHAR *avisynth_dll_name = _T("libavisynth.so");
#endif

static const int RGY_AVISYNTH_INTERFACE_25 = 2;

int AVSC_CC rgy_avs_get_pitch_p(const AVS_VideoFrame * p, int plane) {
    switch (plane) {
    case AVS_PLANAR_U:
    case AVS_PLANAR_V:
        return p->pitchUV;
    }
    return p->pitch;
}

const uint8_t* AVSC_CC rgy_avs_get_read_ptr_p(const AVS_VideoFrame * p, int plane) {
    switch (plane) {
    case AVS_PLANAR_U: return p->vfb->data + p->offsetU;
    case AVS_PLANAR_V: return p->vfb->data + p->offsetV;
    default:           return p->vfb->data + p->offset;
    }
}

RGYInputAvsPrm::RGYInputAvsPrm(RGYInputPrm base) :
    RGYInputPrm(base),
    readAudio(false),
    avsdll() {

}

RGYInputAvs::RGYInputAvs() :
    m_sAVSenv(nullptr),
    m_sAVSclip(nullptr),
    m_sAVSinfo(nullptr),
    m_sAvisynth(),
#if ENABLE_AVSW_READER
    m_audio(),
    m_format(unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(nullptr, &avformat_free_context)),
    m_audioCurrentSample(0)
#endif //#if ENABLE_AVSW_READER
{
    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
    m_readerName = _T("avs");
}

RGYInputAvs::~RGYInputAvs() {
    Close();
}

void RGYInputAvs::release_avisynth() {
    if (m_sAvisynth.h_avisynth)
#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(m_sAvisynth.h_avisynth);
#else
        dlclose(m_sAvisynth.h_avisynth);
#endif

    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
}

RGY_ERR RGYInputAvs::load_avisynth(const tstring &avsdll) {
    release_avisynth();

    const TCHAR *avs_dll_target = nullptr;
    if (avsdll.length() > 0) {
        avs_dll_target = avsdll.c_str();
    }
    if (avs_dll_target == nullptr) {
        avs_dll_target = avisynth_dll_name;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Load Avisynth DLL \"%s\".\n"), avs_dll_target);

#if defined(_WIN32) || defined(_WIN64)
    if (nullptr == (m_sAvisynth.h_avisynth = (HMODULE)LoadLibrary(avs_dll_target)))
#else
    if (nullptr == (m_sAvisynth.h_avisynth = dlopen(avs_dll_target, RTLD_LAZY)))
#endif
        return RGY_ERR_INVALID_HANDLE;

#define LOAD_FUNC(x, required, altern_func) {\
    if (nullptr == (m_sAvisynth.f_ ## x = (func_avs_ ## x)RGY_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_" #x))) { \
        if (required) return RGY_ERR_INVALID_HANDLE; \
        if (altern_func != nullptr) { m_sAvisynth.f_ ## x = (altern_func); }; \
    } \
}
#pragma warning(push)
#pragma warning(disable:4127) //warning C4127: 条件式が定数です。
    LOAD_FUNC(invoke, true, nullptr);
    LOAD_FUNC(take_clip, true, nullptr);
    LOAD_FUNC(release_value, true, nullptr);
    LOAD_FUNC(create_script_environment, true, nullptr);
    LOAD_FUNC(get_video_info, true, nullptr);
    LOAD_FUNC(get_audio, true, nullptr);
    LOAD_FUNC(get_frame, true, nullptr);
    LOAD_FUNC(release_video_frame, true, nullptr);
    LOAD_FUNC(release_clip, true, nullptr);
    LOAD_FUNC(delete_script_environment, true, nullptr);
    LOAD_FUNC(get_version, true, nullptr);
    LOAD_FUNC(get_pitch_p, false, rgy_avs_get_pitch_p);
    LOAD_FUNC(get_read_ptr_p, false, rgy_avs_get_read_ptr_p);
    LOAD_FUNC(clip_get_error, true, nullptr);
    LOAD_FUNC(is_420, false, nullptr);
    LOAD_FUNC(is_422, false, nullptr);
    LOAD_FUNC(is_444, false, nullptr);
#pragma warning(pop)
#undef LOAD_FUNC
    return RGY_ERR_NONE;
}

#if ENABLE_AVSW_READER
RGY_ERR RGYInputAvs::InitAudio() {
    auto format = avformat_alloc_context();
    if (format == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to alloc format context.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    m_format = unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(format, &avformat_free_context);

    AVDemuxStream st = { 0 };

    st.stream = avformat_new_stream(m_format.get(), NULL);

    st.stream->codecpar->codec_type  = AVMEDIA_TYPE_AUDIO;
    st.stream->codecpar->sample_rate = m_sAVSinfo->audio_samples_per_second;
    st.stream->codecpar->channels    = m_sAVSinfo->nchannels;
    st.stream->duration              = m_sAVSinfo->num_audio_samples;
    st.stream->time_base             = av_make_q(1, m_sAVSinfo->audio_samples_per_second);

    switch (m_sAVSinfo->sample_type) {
    case AVS_SAMPLE_INT8:
        st.stream->codecpar->codec_id = AV_CODEC_ID_PCM_U8;
        st.stream->codecpar->bits_per_coded_sample = 8;
        st.stream->codecpar->bits_per_raw_sample = 8;
        st.stream->codecpar->format = AV_SAMPLE_FMT_U8;
        break;
    case AVS_SAMPLE_INT16:
        st.stream->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
        st.stream->codecpar->bits_per_coded_sample = 16;
        st.stream->codecpar->bits_per_raw_sample = 16;
        st.stream->codecpar->format = AV_SAMPLE_FMT_S16;
        break;
    case AVS_SAMPLE_INT24:
        st.stream->codecpar->codec_id = AV_CODEC_ID_PCM_S24LE;
        st.stream->codecpar->bits_per_coded_sample = 24;
        st.stream->codecpar->bits_per_raw_sample = 24;
        break;
    case AVS_SAMPLE_INT32:
        st.stream->codecpar->codec_id = AV_CODEC_ID_PCM_S32LE;
        st.stream->codecpar->bits_per_coded_sample = 32;
        st.stream->codecpar->bits_per_raw_sample = 32;
        st.stream->codecpar->format = AV_SAMPLE_FMT_S32;
        break;
    case AVS_SAMPLE_FLOAT:
        st.stream->codecpar->codec_id = AV_CODEC_ID_PCM_F32LE;
        st.stream->codecpar->bits_per_coded_sample = 32;
        st.stream->codecpar->bits_per_raw_sample = 32;
        st.stream->codecpar->format = AV_SAMPLE_FMT_FLT;
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("Unknown AviSynth sample type %d.\n"), m_sAVSinfo->sample_type);
        return RGY_ERR_INVALID_AUDIO_PARAM;
    }
    st.index = 0;
    st.timebase = st.stream->time_base;
    st.trackId = trackFullID(AVMEDIA_TYPE_AUDIO, (int)m_audio.size() + 1);
    m_audio.push_back(st);
    return RGY_ERR_NONE;
}

vector<AVPacket> RGYInputAvs::GetStreamDataPackets(int inputFrame) {
    UNREFERENCED_PARAMETER(inputFrame);

    vector<AVPacket> pkts;
    if (m_audio.size() == 0) {
        return pkts;
    }

    const auto samplerate = av_make_q(m_sAVSinfo->audio_samples_per_second, 1);
    const auto fps = av_make_q(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);
    auto samples = (int)(av_rescale_q(m_encSatusInfo->m_sData.frameIn, samplerate, fps) - m_audioCurrentSample);
    if (samples <= 0) {
        return pkts;
    }
    if (m_audioCurrentSample + samples > m_sAVSinfo->num_audio_samples) {
        samples = (int)(m_sAVSinfo->num_audio_samples - m_audioCurrentSample);
    }

    const int size = avs_bytes_per_channel_sample(m_sAVSinfo) * samples * m_sAVSinfo->nchannels;
    AVPacket pkt;
    if (av_new_packet(&pkt, size) < 0) {
        return pkts;
    }
    pkt.pts = m_audioCurrentSample;
    pkt.dts = m_audioCurrentSample;
    pkt.duration = samples;
    pkt.stream_index = m_audio.begin()->index;
    pkt.flags = (pkt.flags & 0xffff) | ((uint32_t)m_audio.begin()->trackId << 16); //flagsの上位16bitには、trackIdへのポインタを格納しておく

    m_sAvisynth.f_get_audio(m_sAVSclip, pkt.data, m_audioCurrentSample, samples);
    const auto avs_err = m_sAvisynth.f_clip_get_error(m_sAVSclip);
    if (avs_err) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown error when reading audio frame from avisynth: %d.\n"), avs_err);
        return pkts;
    }
    pkts.push_back(pkt);
    m_audioCurrentSample += samples;
    return pkts;
}
#endif //#if ENABLE_AVSW_READER

#pragma warning(push)
#pragma warning(disable:4127) //warning C4127: 条件式が定数です。
RGY_ERR RGYInputAvs::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const RGYInputPrm *prm) {
    m_inputVideoInfo = *pInputInfo;

    auto avsPrm = reinterpret_cast<const RGYInputAvsPrm *>(prm);
    if (load_avisynth(avsPrm->avsdll) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load %s.\n"), avisynth_dll_name);
        return RGY_ERR_INVALID_HANDLE;
    }

    m_convert = std::make_unique<RGYConvertCSP>(prm->threadCsp);

    const auto interface_ver = (m_sAvisynth.f_is_420 && m_sAvisynth.f_is_422 && m_sAvisynth.f_is_444) ? AVISYNTH_INTERFACE_VERSION : RGY_AVISYNTH_INTERFACE_25;
    if (nullptr == (m_sAVSenv = m_sAvisynth.f_create_script_environment(interface_ver))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init avisynth enviroment.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    std::string filename_char;
    if (0 == tchar_to_string(strFileName, filename_char)) {
        AddMessage(RGY_LOG_ERROR,  _T("failed to convert to ansi characters.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVS_Value val_filename = avs_new_value_string(filename_char.c_str());
    AVS_Value val_res = m_sAvisynth.f_invoke(m_sAVSenv, "Import", val_filename, nullptr);
    m_sAvisynth.f_release_value(val_filename);
    AddMessage(RGY_LOG_DEBUG,  _T("opened avs file: \"%s\"\n"), char_to_tstring(filename_char).c_str());
    if (!avs_is_clip(val_res)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid clip.\n"));
        if (avs_is_error(val_res)) {
            AddMessage(RGY_LOG_ERROR, char_to_tstring(avs_as_string(val_res)) + _T("\n"));
        }
        m_sAvisynth.f_release_value(val_res);
        return RGY_ERR_INVALID_HANDLE;
    }
    m_sAVSclip = m_sAvisynth.f_take_clip(val_res, m_sAVSenv);
    m_sAvisynth.f_release_value(val_res);

    if (nullptr == (m_sAVSinfo = m_sAvisynth.f_get_video_info(m_sAVSclip))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get avs info.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }

    if (!avs_has_video(m_sAVSinfo)) {
        AddMessage(RGY_LOG_ERROR, _T("avs has no video.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    AddMessage(RGY_LOG_DEBUG, _T("found video from avs file, pixel type 0x%x.\n"), m_sAVSinfo->pixel_type);

    struct CSPMap {
        int fmtID;
        RGY_CSP in, out;
        constexpr CSPMap(int fmt, RGY_CSP i, RGY_CSP o) : fmtID(fmt), in(i), out(o) {};
    };

    static constexpr auto valid_csp_list = make_array<CSPMap>(
        CSPMap( AVS_CS_YV12,       RGY_CSP_YV12,      RGY_CSP_NV12 ),
        CSPMap( AVS_CS_I420,       RGY_CSP_YV12,      RGY_CSP_NV12 ),
        CSPMap( AVS_CS_IYUV,       RGY_CSP_YV12,      RGY_CSP_NV12 ),
        CSPMap( AVS_CS_YUV420P10,  RGY_CSP_YV12_10,   RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV420P12,  RGY_CSP_YV12_12,   RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV420P14,  RGY_CSP_YV12_14,   RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV420P16,  RGY_CSP_YV12_16,   RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUY2,       RGY_CSP_YUY2,      RGY_CSP_NV16 ),
#if ENCODER_QSV || ENCODER_VCEENC
        CSPMap( AVS_CS_YV16,       RGY_CSP_YUV422,    RGY_CSP_NV12 ),
        CSPMap( AVS_CS_YUV422P10,  RGY_CSP_YUV422_10, RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV422P12,  RGY_CSP_YUV422_12, RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV422P14,  RGY_CSP_YUV422_14, RGY_CSP_P010 ),
        CSPMap( AVS_CS_YUV422P16,  RGY_CSP_YUV422_16, RGY_CSP_P010 ),
#else
        CSPMap( AVS_CS_YV16,       RGY_CSP_YUV422,    RGY_CSP_NV16 ),
        CSPMap( AVS_CS_YUV422P10,  RGY_CSP_YUV422_10, RGY_CSP_P210 ),
        CSPMap( AVS_CS_YUV422P12,  RGY_CSP_YUV422_12, RGY_CSP_P210 ),
        CSPMap( AVS_CS_YUV422P14,  RGY_CSP_YUV422_14, RGY_CSP_P210 ),
        CSPMap( AVS_CS_YUV422P16,  RGY_CSP_YUV422_16, RGY_CSP_P210 ),
#endif
        CSPMap( AVS_CS_YV24,       RGY_CSP_YUV444,    RGY_CSP_YUV444 ),
        CSPMap( AVS_CS_YUV444P10,  RGY_CSP_YUV444_10, RGY_CSP_YUV444_16 ),
        CSPMap( AVS_CS_YUV444P12,  RGY_CSP_YUV444_12, RGY_CSP_YUV444_16 ),
        CSPMap( AVS_CS_YUV444P14,  RGY_CSP_YUV444_14, RGY_CSP_YUV444_16 ),
        CSPMap( AVS_CS_YUV444P16,  RGY_CSP_YUV444_16, RGY_CSP_YUV444_16 ),
        CSPMap( AVS_CS_BGR24,      RGY_CSP_RGB24R,    (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 ),
        CSPMap( AVS_CS_BGR32,      RGY_CSP_RGB32R,    (ENCODER_NVENC) ? RGY_CSP_RGB : RGY_CSP_RGB32 )
    );

    const RGY_CSP prefered_csp = m_inputVideoInfo.csp;
    m_inputCsp = RGY_CSP_NA;
    for (const auto& csp : valid_csp_list) {
        if (csp.fmtID == m_sAVSinfo->pixel_type) {
            m_inputCsp = csp.in;
            if (prefered_csp == RGY_CSP_NA) {
                //ロスレスの場合は、入力側で出力フォーマットを決める
                m_inputVideoInfo.csp = csp.out;
            } else {
                m_inputVideoInfo.csp = (m_convert->getFunc(m_inputCsp, prefered_csp, false, prm->simdCsp) != nullptr) ? prefered_csp : csp.out;
                //csp.outがYUV422に関しては可能ならcsp.outを優先する
                if (RGY_CSP_CHROMA_FORMAT[csp.out] == RGY_CHROMAFMT_YUV422
                    && m_convert->getFunc(m_inputCsp, csp.out, false, prm->simdCsp) != nullptr) {
                    m_inputVideoInfo.csp = csp.out;
                }
                //QSVではNV16->P010がサポートされていない
                if (ENCODER_QSV && m_inputVideoInfo.csp == RGY_CSP_NV16 && prefered_csp == RGY_CSP_P010) {
                    m_inputVideoInfo.csp = RGY_CSP_P210;
                }
                //なるべく軽いフォーマットでGPUに転送するように
                if (ENCODER_NVENC
                    && RGY_CSP_BIT_PER_PIXEL[csp.out] < RGY_CSP_BIT_PER_PIXEL[prefered_csp]
                    && m_convert->getFunc(m_inputCsp, csp.out, false, prm->simdCsp) != nullptr) {
                    m_inputVideoInfo.csp = csp.out;
                }
            }
            if (m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp) == nullptr && m_inputCsp == RGY_CSP_YUY2) {
                //YUY2用の特別処理
                m_inputVideoInfo.csp = (RGY_CSP_CHROMA_FORMAT[csp.out] == RGY_CHROMAFMT_YUV420 || ENCODER_QSV) ? RGY_CSP_NV12 : RGY_CSP_YUV444;
                m_convert->getFunc(m_inputCsp, m_inputVideoInfo.csp, false, prm->simdCsp);
            }
            break;
        }
    }

    if (m_inputCsp == RGY_CSP_NA) {
        AddMessage(RGY_LOG_ERROR, _T("invalid colorformat.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (m_convert->getFunc() == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("color conversion not supported: %s -> %s.\n"),
            RGY_CSP_NAMES[m_inputCsp], RGY_CSP_NAMES[m_inputVideoInfo.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    if (m_inputVideoInfo.csp != prefered_csp) {
        //入力フォーマットを変えた場合、m_inputVideoInfo.shiftは、出力フォーマットに対応する値ではなく、
        //入力フォーマットに対応する値とする必要がある
        m_inputVideoInfo.shift = (RGY_CSP_BIT_DEPTH[m_inputCsp] > 8) ? 16 - RGY_CSP_BIT_DEPTH[m_inputCsp] : 0;
    }

    m_inputVideoInfo.srcWidth = m_sAVSinfo->width;
    m_inputVideoInfo.srcHeight = m_sAVSinfo->height;
    m_inputVideoInfo.fpsN = m_sAVSinfo->fps_numerator;
    m_inputVideoInfo.fpsD = m_sAVSinfo->fps_denominator;
    m_inputVideoInfo.shift = ((m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_P210) && m_inputVideoInfo.shift) ? m_inputVideoInfo.shift : 0;
    m_inputVideoInfo.frames = m_sAVSinfo->num_frames;
    rgy_reduce(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);

    if (avsPrm != nullptr && avsPrm->readAudio) {
        if (!avs_has_audio(m_sAVSinfo)) {
            AddMessage(RGY_LOG_WARN, _T("avs has no audio.\n"));
        } else {
            auto err = InitAudio();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_DEBUG, _T("failed to initialize audio.\n"));
                return err;
            }
        }
    }

    tstring avisynth_version = (m_sAvisynth.f_is_420 && m_sAvisynth.f_is_422 && m_sAvisynth.f_is_444) ? _T("Avisynth+ ") : _T("Avisynth ");
    AVS_Value val_version = m_sAvisynth.f_invoke(m_sAVSenv, "VersionString", avs_new_value_array(nullptr, 0), nullptr);
    if (avs_is_error(val_version) || avs_as_string(val_version) == nullptr) {
        val_version = m_sAvisynth.f_invoke(m_sAVSenv, "VersionNumber", avs_new_value_array(nullptr, 0), nullptr);
        if (avs_is_float(val_version)) {
            avisynth_version += strsprintf(_T("%.2f"), avs_as_float(val_version));
        }
    } else {
        //VersionStringの短縮 (ちょっと長い)
        avisynth_version = char_to_tstring(avs_as_string(val_version));
        auto pos1 = avisynth_version.find(_T("("));
        if (pos1 != std::string::npos) {
            auto pos2 = avisynth_version.find(_T(","), pos1);
            if (pos2 != std::string::npos) {
                avisynth_version = avisynth_version.substr(0, pos1) + avisynth_version.substr(pos1 + 1, pos2 - pos1 - 1);
            }
        }
        avisynth_version = str_replace(avisynth_version, _T("Avisynth "), _T("Avisynth"));
        avisynth_version = str_replace(avisynth_version, _T("AviSynth "), _T("AviSynth"));
        avisynth_version = str_replace(avisynth_version, _T(", "), _T(","));
    }
    m_sAvisynth.f_release_value(val_version);

    CreateInputInfo(avisynth_version.c_str(), RGY_CSP_NAMES[m_convert->getFunc()->csp_from], RGY_CSP_NAMES[m_convert->getFunc()->csp_to], get_simd_str(m_convert->getFunc()->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_inputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}
#pragma warning(pop)

void RGYInputAvs::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
#if ENABLE_AVSW_READER
    m_format.reset();
#endif //#if ENABLE_AVSW_READER
    if (m_sAVSclip)
        m_sAvisynth.f_release_clip(m_sAVSclip);
    if (m_sAVSenv)
        m_sAvisynth.f_delete_script_environment(m_sAVSenv);

    release_avisynth();

    m_sAVSenv = nullptr;
    m_sAVSclip = nullptr;
    m_sAVSinfo = nullptr;
    m_encSatusInfo.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

RGY_ERR RGYInputAvs::LoadNextFrame(RGYFrame *pSurface) {
    if ((int)m_encSatusInfo->m_sData.frameIn >= m_inputVideoInfo.frames
        //m_encSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)m_encSatusInfo->m_sData.frameIn - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }

    AVS_VideoFrame *frame = m_sAvisynth.f_get_frame(m_sAVSclip, m_encSatusInfo->m_sData.frameIn);
    if (frame == nullptr) {
        return RGY_ERR_MORE_DATA;
    }
    auto avs_err = m_sAvisynth.f_clip_get_error(m_sAVSclip);
    if (avs_err) {
        AddMessage(RGY_LOG_ERROR, _T("Unknown error when reading video frame from avisynth: %d.\n"), avs_err);
        return RGY_ERR_UNKNOWN;
    }

    void *dst_array[3];
    pSurface->ptrArray(dst_array, m_convert->getFunc()->csp_to == RGY_CSP_RGB24 || m_convert->getFunc()->csp_to == RGY_CSP_RGB32);
    const void *src_array[3] = { m_sAvisynth.f_get_read_ptr_p(frame, AVS_PLANAR_Y), m_sAvisynth.f_get_read_ptr_p(frame, AVS_PLANAR_U), m_sAvisynth.f_get_read_ptr_p(frame, AVS_PLANAR_V) };

    m_convert->run((m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0,
        dst_array, src_array,
        m_inputVideoInfo.srcWidth, m_sAvisynth.f_get_pitch_p(frame, AVS_PLANAR_Y), m_sAvisynth.f_get_pitch_p(frame, AVS_PLANAR_U),
        pSurface->pitch(), m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

    m_sAvisynth.f_release_video_frame(frame);

    m_encSatusInfo->m_sData.frameIn++;
    return m_encSatusInfo->UpdateDisplay();
}

#endif //ENABLE_AVISYNTH_READER
