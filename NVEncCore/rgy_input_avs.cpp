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
static const TCHAR *avisynth_dll_name = _T("libavxsynth.so");
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

const BYTE* AVSC_CC rgy_avs_get_read_ptr_p(const AVS_VideoFrame * p, int plane) {
    switch (plane) {
    case AVS_PLANAR_U: return p->vfb->data + p->offsetU;
    case AVS_PLANAR_V: return p->vfb->data + p->offsetV;
    default:           return p->vfb->data + p->offset;
    }
}

RGYInputAvs::RGYInputAvs() :
    m_sAVSenv(nullptr),
    m_sAVSclip(nullptr),
    m_sAVSinfo(nullptr),
    m_sAvisynth() {
    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
    m_strReaderName = _T("avs");
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

RGY_ERR RGYInputAvs::load_avisynth() {
    release_avisynth();

#if defined(_WIN32) || defined(_WIN64)
    if (   nullptr == (m_sAvisynth.h_avisynth = (HMODULE)LoadLibrary(avisynth_dll_name)))
#else
    if (nullptr == (m_sAvisynth.h_avisynth = dlopen(avisynth_dll_name, RTLD_LAZY)))
#endif
        return RGY_ERR_INVALID_HANDLE;

#define LOAD_FUNC(x, required, altern_func) {\
    if (nullptr == (m_sAvisynth. ## x = (func_avs_ ## x)RGY_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_" #x))) { \
        if (required) return RGY_ERR_INVALID_HANDLE; \
        if (altern_func) { m_sAvisynth. ## x = (altern_func); }; \
    } \
}
    LOAD_FUNC(invoke, true, nullptr);
    LOAD_FUNC(take_clip, true, nullptr);
    LOAD_FUNC(release_value, true, nullptr);
    LOAD_FUNC(create_script_environment, true, nullptr);
    LOAD_FUNC(get_video_info, true, nullptr);
    LOAD_FUNC(get_frame, true, nullptr);
    LOAD_FUNC(release_video_frame, true, nullptr);
    LOAD_FUNC(release_clip, true, nullptr);
    LOAD_FUNC(delete_script_environment, true, nullptr);
    LOAD_FUNC(get_version, true, nullptr);
    LOAD_FUNC(get_pitch_p, false, rgy_avs_get_pitch_p);
    LOAD_FUNC(get_read_ptr_p, false, rgy_avs_get_read_ptr_p);
    LOAD_FUNC(is_420, false, nullptr);
    LOAD_FUNC(is_422, false, nullptr);
    LOAD_FUNC(is_444, false, nullptr);

#undef LOAD_FUNC
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputAvs::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(prm);
    memcpy(&m_inputVideoInfo, pInputInfo, sizeof(m_inputVideoInfo));

    if (load_avisynth() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load %s.\n"), avisynth_dll_name);
        return RGY_ERR_INVALID_HANDLE;
    }

    const auto interface_ver = (m_sAvisynth.is_420 && m_sAvisynth.is_422 && m_sAvisynth.is_444) ? AVISYNTH_INTERFACE_VERSION : 5;
    if (nullptr == (m_sAVSenv = m_sAvisynth.create_script_environment(interface_ver))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init avisynth enviroment.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    std::string filename_char;
    if (0 == tchar_to_string(strFileName, filename_char)) {
        AddMessage(RGY_LOG_ERROR,  _T("failed to convert to ansi characters.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVS_Value val_filename = avs_new_value_string(filename_char.c_str());
    AVS_Value val_res = m_sAvisynth.invoke(m_sAVSenv, "Import", val_filename, nullptr);
    m_sAvisynth.release_value(val_filename);
    AddMessage(RGY_LOG_DEBUG,  _T("opened avs file: \"%s\"\n"), char_to_tstring(filename_char).c_str());
    if (!avs_is_clip(val_res)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid clip.\n"));
        if (avs_is_error(val_res)) {
            AddMessage(RGY_LOG_ERROR, char_to_tstring(avs_as_string(val_res)) + _T("\n"));
        }
        m_sAvisynth.release_value(val_res);
        return RGY_ERR_INVALID_HANDLE;
    }
    m_sAVSclip = m_sAvisynth.take_clip(val_res, m_sAVSenv);
    m_sAvisynth.release_value(val_res);

    if (nullptr == (m_sAVSinfo = m_sAvisynth.get_video_info(m_sAVSclip))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get avs info.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }

    if (!avs_has_video(m_sAVSinfo)) {
        AddMessage(RGY_LOG_ERROR, _T("avs has no video.\n"));
        return RGY_ERR_INVALID_HANDLE;
    }
    AddMessage(RGY_LOG_DEBUG, _T("found video from avs file, pixel type 0x%x.\n"), m_sAVSinfo->pixel_type);

    const RGY_CSP prefered_csp = (m_inputVideoInfo.csp != RGY_CSP_NA) ? m_inputVideoInfo.csp : RGY_CSP_NV12;

    struct CSPMap {
        int fmtID;
        RGY_CSP in, out;
    };

    const std::vector<CSPMap> valid_csp_list = {
        { AVS_CS_YV12,       RGY_CSP_YV12,      prefered_csp },
        { AVS_CS_I420,       RGY_CSP_YV12,      prefered_csp },
        { AVS_CS_IYUV,       RGY_CSP_YV12,      prefered_csp },
        { AVS_CS_YUY2,       RGY_CSP_YUY2,      prefered_csp },
        { AVS_CS_YV16,       RGY_CSP_YUV422,    RGY_CSP_NV16 },
        { AVS_CS_YV24,       RGY_CSP_YUV444,    prefered_csp },
        { AVS_CS_YUV420P10,  RGY_CSP_YV12_10,   prefered_csp },
        { AVS_CS_YUV422P10,  RGY_CSP_YUV422_10, RGY_CSP_P210 },
        { AVS_CS_YUV444P10,  RGY_CSP_YUV444_10, prefered_csp },
        { AVS_CS_YUV420P12,  RGY_CSP_YV12_12,   prefered_csp },
        { AVS_CS_YUV422P12,  RGY_CSP_YUV422_12, RGY_CSP_P210 },
        { AVS_CS_YUV444P12,  RGY_CSP_YUV444_12, prefered_csp },
        { AVS_CS_YUV420P14,  RGY_CSP_YV12_14,   prefered_csp },
        { AVS_CS_YUV422P14,  RGY_CSP_YUV422_14, RGY_CSP_P210 },
        { AVS_CS_YUV444P14,  RGY_CSP_YUV444_14, prefered_csp },
        { AVS_CS_YUV420P16,  RGY_CSP_YV12_16,   prefered_csp },
        { AVS_CS_YUV422P16,  RGY_CSP_YUV422_16, RGY_CSP_P210 },
        { AVS_CS_YUV444P16,  RGY_CSP_YUV444_16, prefered_csp },
        { AVS_CS_BGR24,      RGY_CSP_RGB24R,    RGY_CSP_RGB32 },
        { AVS_CS_BGR32,      RGY_CSP_RGB32R,    RGY_CSP_RGB32 },
    };

    m_InputCsp = RGY_CSP_NA;
    for (auto csp : valid_csp_list) {
        if (csp.fmtID == m_sAVSinfo->pixel_type) {
            m_InputCsp = csp.in;
            m_inputVideoInfo.csp = csp.out;
            m_sConvert = get_convert_csp_func(csp.in, csp.out, false);
            break;
        }
    }

    if (m_InputCsp == RGY_CSP_NA || m_sConvert == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("invalid colorformat.\n"));
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    m_inputVideoInfo.srcWidth = m_sAVSinfo->width;
    m_inputVideoInfo.srcHeight = m_sAVSinfo->height;
    m_inputVideoInfo.fpsN = m_sAVSinfo->fps_numerator;
    m_inputVideoInfo.fpsD = m_sAVSinfo->fps_denominator;
    m_inputVideoInfo.shift = ((m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_P210) && m_inputVideoInfo.shift) ? m_inputVideoInfo.shift : 0;
    m_inputVideoInfo.frames = m_sAVSinfo->num_frames;
    rgy_reduce(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);

    tstring avisynth_version = (m_sAvisynth.is_420 && m_sAvisynth.is_422 && m_sAvisynth.is_444) ? _T("Avisynth+ ") : _T("Avisynth ");
    AVS_Value val_version = m_sAvisynth.invoke(m_sAVSenv, "VersionNumber", avs_new_value_array(nullptr, 0), nullptr);
    if (avs_is_float(val_version)) {
        avisynth_version += strsprintf(_T("%.2f"), avs_as_float(val_version));
    }
    m_sAvisynth.release_value(val_version);

    CreateInputInfo(avisynth_version.c_str(), RGY_CSP_NAMES[m_sConvert->csp_from], RGY_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_strInputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}

void RGYInputAvs::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    if (m_sAVSclip)
        m_sAvisynth.release_clip(m_sAVSclip);
    if (m_sAVSenv)
        m_sAvisynth.delete_script_environment(m_sAVSenv);

    release_avisynth();

    m_sAVSenv = nullptr;
    m_sAVSclip = nullptr;
    m_sAVSinfo = nullptr;
    m_pEncSatusInfo.reset();
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
}

RGY_ERR RGYInputAvs::LoadNextFrame(RGYFrame *pSurface) {
    if ((int)m_pEncSatusInfo->m_sData.frameIn >= m_inputVideoInfo.frames
        //m_pEncSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)m_pEncSatusInfo->m_sData.frameIn - TRIM_OVERREAD_FRAMES) {
        return RGY_ERR_MORE_DATA;
    }

    AVS_VideoFrame *frame = m_sAvisynth.get_frame(m_sAVSclip, m_pEncSatusInfo->m_sData.frameIn);
    if (frame == nullptr) {
        return RGY_ERR_MORE_DATA;
    }

    void *dst_array[3];
    pSurface->ptrArray(dst_array, m_sConvert->csp_to == RGY_CSP_RGB24 || m_sConvert->csp_to == RGY_CSP_RGB32);
    const void *src_array[3] = { m_sAvisynth.get_read_ptr_p(frame, AVS_PLANAR_Y), m_sAvisynth.get_read_ptr_p(frame, AVS_PLANAR_U), m_sAvisynth.get_read_ptr_p(frame, AVS_PLANAR_V) };

    m_sConvert->func[(m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0](
        dst_array, src_array,
        m_inputVideoInfo.srcWidth, m_sAvisynth.get_pitch_p(frame, AVS_PLANAR_Y), m_sAvisynth.get_pitch_p(frame, AVS_PLANAR_U),
        pSurface->pitch(), m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

    m_sAvisynth.release_video_frame(frame);

    m_pEncSatusInfo->m_sData.frameIn++;
    return m_pEncSatusInfo->UpdateDisplay();
}

#endif //ENABLE_AVISYNTH_READER
