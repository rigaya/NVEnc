//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <io.h>
#include <fcntl.h>
#include <string>
#include <sstream>
#include "nvEncodeAPI.h"
#include "NVEncStatus.h"
#include "NVEncInput.h"
#include "ConvertCSP.h"
#include "NVEncCore.h"
#include "NVEncInputAvs.h"

#if AVS_READER

NVEncInputAvs::NVEncInputAvs() {
    m_nFrame = 0;
    m_nMaxFrame = 0;
    m_sAVSenv = nullptr;
    m_sAVSclip = nullptr;
    m_sAVSinfo = nullptr;
}

NVEncInputAvs::~NVEncInputAvs() {
    Close();
}

void NVEncInputAvs::release_avisynth() {
    if (m_sAvisynth.h_avisynth)
        FreeLibrary(m_sAvisynth.h_avisynth);

    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
}

int NVEncInputAvs::load_avisynth() {
    release_avisynth();

    if (   NULL == (m_sAvisynth.h_avisynth = (HMODULE)LoadLibrary(_T("avisynth.dll")))
        || NULL == (m_sAvisynth.invoke = (func_avs_invoke)GetProcAddress(m_sAvisynth.h_avisynth, "avs_invoke"))
        || NULL == (m_sAvisynth.take_clip = (func_avs_take_clip)GetProcAddress(m_sAvisynth.h_avisynth, "avs_take_clip"))
        || NULL == (m_sAvisynth.create_script_environment = (func_avs_create_script_environment)GetProcAddress(m_sAvisynth.h_avisynth, "avs_create_script_environment"))
        || NULL == (m_sAvisynth.delete_script_environment = (func_avs_delete_script_environment)GetProcAddress(m_sAvisynth.h_avisynth, "avs_delete_script_environment"))
        || NULL == (m_sAvisynth.get_frame = (func_avs_get_frame)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_frame"))
        || NULL == (m_sAvisynth.get_version = (func_avs_get_version)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_version"))
        || NULL == (m_sAvisynth.get_video_info = (func_avs_get_video_info)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_video_info"))
        || NULL == (m_sAvisynth.release_clip = (func_avs_release_clip)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_clip"))
        || NULL == (m_sAvisynth.release_value = (func_avs_release_value)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_value"))
        || NULL == (m_sAvisynth.release_video_frame = (func_avs_release_video_frame)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_video_frame")))
        return 1;
    return 0;
}

int NVEncInputAvs::Init(InputVideoInfo *inputPrm, EncodeStatus *pStatus) {
    Close();

    m_pStatus = pStatus;
    InputInfoAvs *info = reinterpret_cast<InputInfoAvs *>(inputPrm->otherPrm);
    m_bInterlaced = info->interlaced;
    
    if (load_avisynth()) {
        m_inputMes += _T("avisynth: failed to load avisynth.dll.\n");
        return 1;
    }

    if (nullptr == (m_sAVSenv = m_sAvisynth.create_script_environment(AVISYNTH_INTERFACE_VERSION))) {
        m_inputMes += _T("avisynth: failed to init avisynth enviroment.\n");
        return 1;
    }
#if UNICODE
    char *filename_char = nullptr;
    {
        const uint32_t buffer_length = (uint32_t)(wcslen(inputPrm->filename.c_str()) + 1) * 2;
        BOOL error = FALSE;
        if (nullptr == (filename_char = (char *)calloc(buffer_length, sizeof(char)))) {
            m_inputMes += _T("avisynth: failed to allocate memory for character conversion.\n");
            return 1;
        } else if (0 == WideCharToMultiByte(CP_THREAD_ACP, WC_NO_BEST_FIT_CHARS, inputPrm->filename.c_str(), -1, filename_char, buffer_length, NULL, &error) || error) {
            m_inputMes += _T("avisynth: failed to convert to ansi characters.\n");
            free(filename_char);
            return 1;
        }
    }
    fprintf(stderr, "%s\n", filename_char);
    AVS_Value val_filename = avs_new_value_string(filename_char);
#else
    AVS_Value val_filename = avs_new_value_string(inputPrm->filename.c_str());
#endif
    AVS_Value val_res = m_sAvisynth.invoke(m_sAVSenv, "Import", val_filename, nullptr);
    m_sAvisynth.release_value(val_filename);
    if (!avs_is_clip(val_res)) {
        m_inputMes += _T("avisynth: invalid clip.\n");
        if (avs_is_error(val_res)) {
#if UNICODE
            WCHAR buf[1024];
            MultiByteToWideChar(CP_THREAD_ACP, MB_PRECOMPOSED, avs_as_string(val_res), -1, buf, _countof(buf));
            m_inputMes += buf;
#else
            m_inputMes += avs_as_string(val_res);
#endif
            m_inputMes += _T("\n");
        }
        m_sAvisynth.release_value(val_res);
        return 1;
    }
    m_sAVSclip = m_sAvisynth.take_clip(val_res, m_sAVSenv);
    m_sAvisynth.release_value(val_res);

#if UNICODE    
    free(filename_char);
#endif

    if (nullptr == (m_sAVSinfo = m_sAvisynth.get_video_info(m_sAVSclip))) {
        m_inputMes += _T("avisynth: failed to get avs info.\n");
        return 1;
    }

    if (!avs_has_video(m_sAVSinfo)) {
        m_inputMes += _T("avisynth: avs has no video.\n");
        return 1;
    }

    typedef struct CSPMap {
        int fmtID;
        NV_ENC_CSP in, out;
    } CSPMap;

    static const std::vector<CSPMap> valid_csp_list = {
        { AVS_CS_YV12,  NV_ENC_CSP_YV12, NV_ENC_CSP_NV12},
        { AVS_CS_I420,  NV_ENC_CSP_YV12, NV_ENC_CSP_NV12},
        { AVS_CS_IYUV,  NV_ENC_CSP_YV12, NV_ENC_CSP_NV12},
        { AVS_CS_YUY2,  NV_ENC_CSP_YUY2, NV_ENC_CSP_NV12},
        //{ AVS_CS_BGR24, MFX_FOURCC_RGB3, MFX_FOURCC_RGB4},
        //{ AVS_CS_BGR32, MFX_FOURCC_RGB4, MFX_FOURCC_RGB4},
    };

    for (auto csp : valid_csp_list) {
        if (csp.fmtID == m_sAVSinfo->pixel_type) {
            m_pConvCSPInfo = get_convert_csp_func(csp.in, csp.out, false);
            break;
        }
    }

    if (nullptr == m_pConvCSPInfo) {
        m_inputMes += _T("avisynth: invalid colorformat.\n");
        return 1;
    }
    
    uint32_t fps_gcd = nv_get_gcd(m_sAVSinfo->fps_numerator, m_sAVSinfo->fps_denominator);
    pStatus->m_sData.frameTotal = m_sAVSinfo->num_frames;
    m_nMaxFrame = m_sAVSinfo->num_frames;
    inputPrm->width = m_sAVSinfo->width;
    inputPrm->height = m_sAVSinfo->height;
    inputPrm->rate = m_sAVSinfo->fps_numerator / fps_gcd;
    inputPrm->scale = m_sAVSinfo->fps_denominator / fps_gcd;
    
    TCHAR avisynth_version[32] = { 0 };
    AVS_Value val_version = m_sAvisynth.invoke(m_sAVSenv, "VersionNumber", avs_new_value_array(nullptr, 0), nullptr);
    if (avs_is_float(val_version)) {
        _stprintf_s(avisynth_version, _T("%s %.2f"), _T("Avisynth"), avs_as_float(val_version));
    }
    m_sAvisynth.release_value(val_version);

    setSurfaceInfo(inputPrm);
    m_stSurface.src_pitch = 0;
    CreateInputInfo(avisynth_version, NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_from], NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_to], get_simd_str(m_pConvCSPInfo->simd), inputPrm);

    return 0;
}

void NVEncInputAvs::Close() {
    if (m_sAVSclip)
        m_sAvisynth.release_clip(m_sAVSclip);
    if (m_sAVSenv)
        m_sAvisynth.delete_script_environment(m_sAVSenv);

    release_avisynth();

    m_sAVSenv = nullptr;
    m_sAVSclip = nullptr;
    m_sAVSinfo = nullptr;
    m_nFrame = 0;
}

int NVEncInputAvs::LoadNextFrame(void *dst, int dst_pitch) {
    if (m_nFrame >= m_nMaxFrame) {
        return NVENC_THREAD_FINISHED;
    }

    AVS_VideoFrame *frame = m_sAvisynth.get_frame(m_sAVSclip, m_nFrame);
    if (frame == nullptr) {
        return NVENC_THREAD_ERROR;
    }

    void *dst_array[3];
    dst_array[0] = dst;
    dst_array[1] = (uint8_t *)dst_array[0] + dst_pitch * (m_stSurface.height - m_stSurface.crop[1] - m_stSurface.crop[3]);

    const void *src_array[3] = { avs_get_read_ptr_p(frame, AVS_PLANAR_Y), avs_get_read_ptr_p(frame, AVS_PLANAR_U), avs_get_read_ptr_p(frame, AVS_PLANAR_V) };
    //if (MFX_FOURCC_RGB4 == m_sConvert->csp_to) {
    //    dst_ptr[0] = min(min(pData->R, pData->G), pData->B);
    //}
    m_pConvCSPInfo->func[!!m_bInterlaced](dst_array, src_array, m_stSurface.width, avs_get_pitch_p(frame, AVS_PLANAR_Y), avs_get_pitch_p(frame, AVS_PLANAR_U), dst_pitch, m_stSurface.height, m_stSurface.height, m_stSurface.crop);
    
    m_sAvisynth.release_video_frame(frame);

    m_nFrame++;
    m_pStatus->m_sData.frameIn++;
    
    uint32_t tm = timeGetTime();
    if (tm - m_tmLastUpdate > 800) {
        m_tmLastUpdate = tm;
        m_pStatus->UpdateDisplay();
    }

    return 0;
}

#endif
