// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#include <Windows.h>
#include <Process.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib") 
#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>

#include "output.h"
#include "vphelp_client.h"

#pragma warning( push )
#pragma warning( disable: 4127 )
#include "afs_client.h"
#pragma warning( pop )

#include "convert.h"

#include "auo.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "auo_version.h"

#include "auo_encode.h"
#include "auo_video.h"
#include "auo_audio_parallel.h"

#include "auo_nvenc.h"
#include "ConvertCSP.h"

AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe);

static int calc_input_frame_size(int width, int height, int color_format) {
    width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
    return width * height * COLORFORMATS[color_format].size;
}

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe) {
    //すでに初期化してある または 必要ない
    if (pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED || !conf->vid.afs)
        return TRUE;

    //high444出力ならAviutlからYC48をもらう
    const int color_format = (0 == memcmp(&conf->nvenc.enc_config.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(NV_ENC_H264_PROFILE_HIGH_444_GUID))) ? CF_YC48 : CF_YUY2;
    const int frame_size = calc_input_frame_size(oip->w, oip->h, color_format);
    //Aviutl(自動フィールドシフト)からの映像入力
    if (afs_vbuf_setup((OUTPUT_INFO *)oip, conf->vid.afs, frame_size, COLORFORMATS[color_format].FOURCC)) {
        pe->afs_init = TRUE;
        return TRUE;
    } else if (conf->vid.afs && sys_dat->exstg->s_local.auto_afs_disable) {
        afs_vbuf_release(); //一度解放
        warning_auto_afs_disable();
        conf->vid.afs = FALSE;
        //再度使用するmuxerをチェックする
        pe->muxer_to_be_used = check_muxer_to_be_used(conf, sys_dat, pe->temp_filename, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
        return TRUE;
    }
    //エラー
    error_afs_setup(conf->vid.afs, sys_dat->exstg->s_local.auto_afs_disable);
    return FALSE;
}

void close_afsvideo(PRM_ENC *pe) {
    if (!pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return;

    afs_vbuf_release();

    pe->afs_init = FALSE;
}

AuoEncodeStatus::AuoEncodeStatus() {
    m_tmLastLogUpdate = m_sData.tmLastUpdate;
    m_pause = false;
}

AuoEncodeStatus::~AuoEncodeStatus() {
    m_pause = false;
}

void AuoEncodeStatus::UpdateDisplay(const TCHAR *mes, double progressPercent) {
    set_log_title_and_progress(mes, progressPercent * 0.01);
    m_auoData.oip->func_rest_time_disp(m_sData.frameOut, m_auoData.oip->n);
    m_auoData.oip->func_update_preview();
}

void AuoEncodeStatus::SetPrivData(void *pPrivateData) {
    m_auoData = *(InputInfoAuo *)pPrivateData;
    enable_enc_control(&m_pause, m_auoData.pe->afs_init, FALSE, timeGetTime(), m_auoData.oip->n);
}

void AuoEncodeStatus::WriteLine(const TCHAR *mes) {
    const char *HEADER = "nvenc [info]: ";
    int buf_len = strlen(mes) + 1 + strlen(HEADER);
    char *buf = (char *)calloc(buf_len, sizeof(buf[0]));
    if (buf) {
        memcpy(buf, HEADER, strlen(HEADER));
        memcpy(buf + strlen(HEADER), mes, strlen(mes) + 1);
        write_log_line(LOG_INFO, buf);
        free(buf);
    }
}
int AuoEncodeStatus::UpdateDisplay(double progressPercent) {
    auto tm = std::chrono::system_clock::now();

    if (m_auoData.oip->func_is_abort())
        return NVENC_THREAD_ABORT;

    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastLogUpdate).count() >= LOG_UPDATE_INTERVAL) {
        log_process_events();

        while (m_pause) {
            Sleep(LOG_UPDATE_INTERVAL);
            if (m_auoData.oip->func_is_abort())
                return NVENC_THREAD_ABORT;
            log_process_events();
        }
        m_tmLastLogUpdate = tm;
    }
    return EncodeStatus::UpdateDisplay(progressPercent);
}

AuoInput::AuoInput() {
    oip = NULL;
    conf = NULL;
    pe = NULL;
}

AuoInput::~AuoInput() {
    Close();
}

void AuoInput::Close() {
    if (pe)
        close_afsvideo(pe);
    oip = NULL;
    conf = NULL;
    pe = NULL;
    m_iFrame = 0;
    disable_enc_control();
}
int AuoInput::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {
    Close();
    
    m_pEncSatusInfo = pStatus;
    auto *info = reinterpret_cast<InputInfoAuo *>(inputPrm->otherPrm);

    oip = info->oip;
    conf = info->conf;
    pe = info->pe;
    jitter = info->jitter;
    m_interlaced = info->interlaced;

    int fps_gcd = nv_get_gcd(oip->rate, oip->scale);

    pStatus->m_sData.frameTotal = oip->n;
    inputPrm->width = oip->w;
    inputPrm->height = oip->h;
    inputPrm->rate = oip->rate / fps_gcd;
    inputPrm->scale = oip->scale / fps_gcd;

    //high444出力ならAviutlからYC48をもらう
    m_pConvCSPInfo = get_convert_csp_func((inputPrm->csp == NV_ENC_CSP_YUV444) ? NV_ENC_CSP_YC48 : NV_ENC_CSP_YUY2, inputPrm->csp, false);

    if (nullptr == m_pConvCSPInfo) {
        AddMessage(NV_LOG_ERROR, "invalid colorformat.\n");
        return 1;
    }

    if (conf->vid.afs) {
        if (!setup_afsvideo(oip, info->sys_dat, conf, pe)) {
            AddMessage(NV_LOG_ERROR, "自動フィールドシフトの初期化に失敗しました。\n");
            return 1;
        }
    }

    memcpy(&m_sDecParam, inputPrm, sizeof(m_sDecParam));
    m_sDecParam.src_pitch = m_sDecParam.width;
    CreateInputInfo(_T("auo"), NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_from], NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_to], get_simd_str(m_pConvCSPInfo->simd), inputPrm);
    AddMessage(NV_LOG_DEBUG, m_strInputInfo);
    return 0;
}
int AuoInput::LoadNextFrame(void *dst, int dst_pitch) {
    if (FALSE != (pe->aud_parallel.abort = oip->func_is_abort()))
        return NVENC_THREAD_ABORT;

    if (m_iFrame >= oip->n) {
        oip->func_rest_time_disp(m_iFrame-1, oip->n);
        release_audio_parallel_events(pe);
        return NVENC_THREAD_FINISHED;
    }

    void *frame = NULL;
    if (conf->vid.afs) {
        BOOL drop = FALSE;
        for (;;) {
            if ((frame = afs_get_video((OUTPUT_INFO *)oip, m_iFrame, &drop, &jitter[m_iFrame + 1])) == NULL) {
                error_afs_get_frame();
                return false;
            }
            if (!drop)
                break;
            jitter[m_iFrame] = DROP_FRAME_FLAG;
            pe->drop_count++;
            m_pEncSatusInfo->m_sData.frameDrop++;
            m_iFrame++;
            if (m_iFrame >= oip->n) {
                oip->func_rest_time_disp(m_iFrame, oip->n);
                release_audio_parallel_events(pe);
                return false;
            }
        }
    } else {
        //high444出力ならAviutlからYC48をもらう
        if ((frame = oip->func_get_video_ex(m_iFrame, COLORFORMATS[m_pConvCSPInfo->csp_from == NV_ENC_CSP_YC48 ? CF_YC48 : CF_YUY2].FOURCC)) == NULL) {
            error_afs_get_frame();
            return false;
        }
    }
    void *dst_array[3];
    dst_array[0] = dst;
    dst_array[1] = (uint8_t *)dst_array[0] + dst_pitch * m_sDecParam.height;
    dst_array[2] = (uint8_t *)dst_array[1] + dst_pitch * m_sDecParam.height; //YUV444出力時
    int src_pitch = m_sDecParam.src_pitch * ((m_pConvCSPInfo->csp_from == NV_ENC_CSP_YC48) ? 6 : 2); //high444出力ならAviutlからYC48をもらう
    m_pConvCSPInfo->func[!!m_interlaced](dst_array, (const void **)&frame, m_sDecParam.width, src_pitch, 0, dst_pitch, m_sDecParam.height, m_sDecParam.height, m_sDecParam.crop.c);

    m_iFrame++;
    if (!(m_iFrame & 7))
        aud_parallel_task(oip, pe);

    m_pEncSatusInfo->m_sData.frameIn++;
    m_pEncSatusInfo->UpdateDisplay();

    return NVENC_THREAD_RUNNING;
}

CAuoNvEnc::CAuoNvEnc() {

}

CAuoNvEnc::~CAuoNvEnc() {

}

NVENCSTATUS CAuoNvEnc::InitLog(const InEncodeVideoParam *inputParam) {
    m_pNVLog.reset(new CAuoLog(inputParam->logfile.c_str(), inputParam->loglevel));
    return NV_ENC_SUCCESS;
}

NVENCSTATUS CAuoNvEnc::InitInput(InEncodeVideoParam *inputParam) {
    m_pStatus.reset(new AuoEncodeStatus());
    m_pStatus->SetPrivData(inputParam->input.otherPrm);
    m_pStatus->init(m_pNVLog);
    m_pFileReader.reset(new AuoInput());
    m_pFileReader->SetNVEncLogPtr(m_pNVLog);
    int ret = m_pFileReader->Init(&inputParam->input, m_pStatus);
    m_pStatus->m_nOutputFPSRate = inputParam->input.rate;
    m_pStatus->m_nOutputFPSScale = inputParam->input.scale;
    return (ret) ? NV_ENC_ERR_GENERIC : NV_ENC_SUCCESS;
}

void CAuoLog::write(int logLevel, const TCHAR *format, ... ) {
    if (logLevel < m_nLogLevel) {
        return;
    }

    logLevel = clamp(logLevel, LOG_INFO, LOG_ERROR);

    va_list args;
    va_start(args, format);

    int len = _vscprintf(format, args);
    char *const buffer = (char*)malloc((len+1) * sizeof(buffer[0])); // _vscprintf doesn't count terminating '\0'

    vsprintf_s(buffer, len+1, format, args);

    static const char *const LOG_LEVEL_STR[] = { "info", "warning", "error" };
    const int mes_line_len = len+1 + strlen("nvenc [warning]: ");
    char *const mes_line = (char *)malloc(mes_line_len * sizeof(mes_line[0]));

    char *a, *b, *mes = buffer;
    char *const fin = mes + len+1; //null文字の位置
    while ((a = strchr(mes, '\n')) != NULL) {
        if ((b = strrchr(mes, '\r', a - mes - 2)) != NULL)
            mes = b + 1;
        *a = '\0';
        sprintf_s(mes_line, mes_line_len, "nvenc [%s]: %s", LOG_LEVEL_STR[logLevel], mes);
        write_log_line(logLevel, mes_line);
        mes = a + 1;
    }
    if ((a = strrchr(mes, '\r', fin - mes - 1)) != NULL) {
        b = a - 1;
        while (*b == ' ' || *b == '\r')
            b--;
        *(b+1) = '\0';
        if ((b = strrchr(mes, '\r', b - mes - 2)) != NULL)
            mes = b + 1;
        set_window_title(mes);
        mes = a + 1;
    }

    free(buffer);
    free(mes_line);
}
