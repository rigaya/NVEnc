﻿// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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
// --------------------------------------------------------------------------------------------

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#pragma comment(lib, "user32.lib") //WaitforInputIdle
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>

#include "auo.h"
#include "vphelp_client.h"

#pragma warning( push )
#pragma warning( disable: 4127 )
#include "afs_client.h"
#pragma warning( pop )

#include "convert.h"

#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "auo_version.h"

#include "auo_encode.h"
#include "auo_convert.h"
#include "auo_video.h"
#include "auo_audio.h"
#include "auo_audio_parallel.h"
#include "auo_mes.h"

#include "cpu_info.h"
#include "rgy_input_sm.h"
#include "NVEncCmd.h"

static const int MAX_CONV_THREADS = 4;

int get_aviutl_color_format(int use_highbit, RGY_CSP csp) {
    //Aviutlからの入力に使用するフォーマット
    RGY_CHROMAFMT chromafmt = RGY_CSP_CHROMA_FORMAT[csp];

    if (use_highbit) {
        return CF_YC48;
    }

    switch (chromafmt) {
    case RGY_CHROMAFMT_RGB:
        return CF_RGB;
    case RGY_CHROMAFMT_YUV444:
        return CF_YC48;
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    default:
        return CF_YUY2;
    }
}

void get_csp_and_bitdepth(bool& use_highbit, RGY_CSP& csp, const CONF_GUIEX *conf) {
    const bool isAviutl2 = is_aviutl2();
    InEncodeVideoParam enc_prm;
    NV_ENC_CODEC_CONFIG codec_prm[RGY_CODEC_NUM] = { 0 };
    codec_prm[RGY_CODEC_H264] = DefaultParamH264();
    codec_prm[RGY_CODEC_HEVC] = DefaultParamHEVC();
    codec_prm[RGY_CODEC_AV1]  = DefaultParamAV1();
    parse_cmd(&enc_prm, codec_prm, conf->enc.cmd);
    enc_prm.encConfig.encodeCodecConfig = codec_prm[enc_prm.codec_rgy];
    if (enc_prm.lossless) {
        enc_prm.outputCsp = RGY_CSP_YUV444;
    }
    use_highbit = !isAviutl2 && enc_prm.outputDepth > 8;
    if (RGY_CSP_CHROMA_FORMAT[enc_prm.outputCsp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[enc_prm.outputCsp] == RGY_CHROMAFMT_RGB_PACKED) {
        csp = RGY_CSP_RGB;
    } else {
        if (use_highbit) {
            csp = (RGY_CSP_CHROMA_FORMAT[enc_prm.outputCsp] == RGY_CHROMAFMT_YUV444) ? RGY_CSP_YUV444_16 : RGY_CSP_P010;
        } else {
            csp = (RGY_CSP_CHROMA_FORMAT[enc_prm.outputCsp] == RGY_CHROMAFMT_YUV444) ? ((isAviutl2) ? RGY_CSP_RGB : RGY_CSP_YUV444) : RGY_CSP_NV12;
        }
    }
}

static int calc_input_frame_size(int width, int height, int color_format, int& buf_size) {
    width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
    //widthが割り切れない場合、多めにアクセスが発生するので、そのぶんを確保しておく
    const DWORD pixel_size = COLORFORMATS[color_format].size;
    const RGY_SIMD simd_check = get_availableSIMD();
    const DWORD align_size = ((simd_check & RGY_SIMD::SSE2) == RGY_SIMD::SSE2) ? (((simd_check & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) ? 64 : 32) : 1;
#define ALIGN_NEXT(i, align) (((i) + (align-1)) & (~(align-1))) //alignは2の累乗(1,2,4,8,16,32...)
    buf_size = ALIGN_NEXT(width * height * pixel_size + (ALIGN_NEXT(width, align_size / pixel_size) - width) * 2 * pixel_size, align_size);
#undef ALIGN_NEXT
    return width * height * pixel_size;
}

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe) {
    //すでに初期化してある または 必要ない
    if (pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED || !conf->vid.afs)
        return TRUE;

    bool use_highbit;
    RGY_CSP csp;
    get_csp_and_bitdepth(use_highbit, csp, conf);
    const int color_format = get_aviutl_color_format(use_highbit, csp);
    int buf_size;
    const int frame_size = calc_input_frame_size(oip->w, oip->h, color_format, buf_size);
    //Aviutl(自動フィールドシフト)からの映像入力
    if (afs_vbuf_setup((OUTPUT_INFO *)oip, conf->vid.afs, frame_size, buf_size, COLORFORMATS[color_format].FOURCC)) {
        pe->afs_init = TRUE;
        return TRUE;
    } else if (conf->vid.afs && sys_dat->exstg->s_local.auto_afs_disable) {
        afs_vbuf_release(); //一度解放
        warning_auto_afs_disable();
        conf->vid.afs = FALSE;
        //再度使用するmuxerをチェックする
        pe->muxer_to_be_used = check_muxer_to_be_used(conf, pe, sys_dat, pe->temp_filename, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
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

DWORD tcfile_out(int *jitter, int frame_n, double fps, BOOL afs, const PRM_ENC *pe) {
    DWORD ret = AUO_RESULT_SUCCESS;
    TCHAR auotcfile[MAX_PATH_LEN];
    FILE *tcfile = NULL;

    if (afs)
        fps *= 4; //afsなら4倍精度
    const double tm_multi = 1000.0 / fps;

    //ファイル名作成
    apply_appendix(auotcfile, _countof(auotcfile), pe->temp_filename, pe->append.tc);

    if (NULL != _tfopen_s(&tcfile, auotcfile, _T("wb"))) {
        ret |= AUO_RESULT_ERROR; warning_auo_tcfile_failed();
    } else {
        fprintf(tcfile, "# timecode format v2\r\n");
        if (afs) {
            int time_additional_frame = 0;
            //オーディオディレイカットのために映像フレームを追加したらその分を考慮したタイムコードを出力する
            if (pe->delay_cut_additional_vframe) {
                //24fpsと30fpsどちらに近いかを考慮する
                const int multi_for_additional_vframe = 4 + !!fps_after_afs_is_24fps(frame_n, pe);
                for (int i = 0; i < pe->delay_cut_additional_vframe; i++)
                    fprintf(tcfile, "%.6lf\r\n", i * multi_for_additional_vframe * tm_multi);

                time_additional_frame = pe->delay_cut_additional_vframe * multi_for_additional_vframe;
            }
            for (int i = 0; i < frame_n; i++)
                if (jitter[i] != DROP_FRAME_FLAG)
                    fprintf(tcfile, "%.6lf\r\n", (i * 4 + jitter[i] + time_additional_frame) * tm_multi);
        } else {
            frame_n += pe->delay_cut_additional_vframe;
            for (int i = 0; i < frame_n; i++)
                fprintf(tcfile, "%.6lf\r\n", i * tm_multi);
        }
        fclose(tcfile);
    }
    return ret;
}

static void build_full_cmd(TCHAR *cmd, size_t nSize, const CONF_GUIEX *conf, const InEncodeVideoParam *encPrm, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const TCHAR *input) {
    //cliモードでない
    //自動設定の適用
    //apply_guiEx_auto_settings(&prm.x264, oip->w, oip->h, oip->rate, oip->scale, sys_dat->exstg->s_local.auto_ref_limit_by_level);
    //GUI部のコマンドライン生成
    _tcscpy_s(cmd, nSize, gen_cmd(encPrm, nullptr, false).c_str());
    //cmdexの処理
    TCHAR cmdex[sizeof(conf->vid.cmdex)];
    _tcscpy_s(cmdex, conf->vid.cmdex);
    cmd_replace(cmdex, _countof(cmdex), pe, sys_dat, conf, oip);
    replace_cmd_CRLF_to_Space(cmdex, _countof(cmdex) - 1);
    _stprintf_s(cmd + _tcslen(cmd), nSize - _tcslen(cmd), _T(" %s"), cmdex);

    //メッセージの発行
    if ((encPrm->encConfig.rcParams.vbvBufferSize != 0 || encPrm->encConfig.rcParams.vbvInitialDelay != 0) && conf->vid.afs) {
        write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_VIDEO_AFS_VBV_WARN));
    }
    //出力ファイル
    _stprintf_s(cmd + _tcslen(cmd), nSize - _tcslen(cmd), _T(" -o \"%s\""), pe->temp_filename);
    //入力
    _stprintf_s(cmd + _tcslen(cmd), nSize - _tcslen(cmd), _T(" --sm --parent-pid %08x -i -"), GetCurrentProcessId());
}

//並列処理時に音声データを取得する
AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe, BOOL use_internal) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    AUD_PARALLEL_ENC *aud_p = &pe->aud_parallel; //長いんで省略したいだけ
    if (aud_p->th_aud) {
        //---   排他ブロック 開始  ---> 音声スレッドが止まっていなければならない
        if (aud_p->he_vid_start && WaitForSingleObject(aud_p->he_vid_start, (use_internal) ? 0 : INFINITE) == WAIT_OBJECT_0) {
            if (aud_p->he_vid_start && aud_p->get_length) {
                DWORD required_buf_size = aud_p->get_length * (DWORD)get_audio_size(oip, 1);
                if (aud_p->buf_max_size < required_buf_size) {
                    //メモリ不足なら再確保
                    if (aud_p->buffer) free(aud_p->buffer);
                    aud_p->buf_max_size = required_buf_size;
                    if (NULL == (aud_p->buffer = malloc(aud_p->buf_max_size)))
                        aud_p->buf_max_size = 0; //ここのmallocエラーは次の分岐でAUO_RESULT_ERRORに設定
                }
                void *data_ptr = NULL;
                if (NULL == aud_p->buffer ||
                    NULL == (data_ptr = oip_func_get_audio(oip, aud_p->start, aud_p->get_length, &aud_p->get_length, 1))) {
                    ret = AUO_RESULT_ERROR; //mallocエラーかget_audioのエラー
                } else {
                    //自前のバッファにコピーしてdata_ptrが破棄されても良いようにする
                    memcpy(aud_p->buffer, data_ptr, aud_p->get_length * (DWORD)get_audio_size(oip, 1));
                }
                //すでにTRUEなら変更しないようにする
            aud_p->abort |= (oip->func_is_abort() ? TRUE : FALSE);
            }
            flush_audio_log();
            if_valid_set_event(aud_p->he_aud_start);
            //---   排他ブロック 終了  ---> 音声スレッドを開始
        }
        //内蔵エンコーダを使う場合、エンコーダが終了していたら意味がないので終了する
        if (use_internal && pe->h_p_videnc && WaitForSingleObject(pe->h_p_videnc, 0) != WAIT_TIMEOUT) {
            aud_p->abort |= TRUE;
        }
    }
    return ret;
}

//音声処理をどんどん回して終了させる
static AUO_RESULT finish_aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe, BOOL use_internal, AUO_RESULT vid_ret) {
    //エラーが発生していたら音声出力ループをとめる
    pe->aud_parallel.abort |= (vid_ret != AUO_RESULT_SUCCESS);
    if (pe->aud_parallel.th_aud) {
        write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_VIDEO_AUDIO_PROC_WAIT));
        set_window_title(g_auo_mes.get(AUO_VIDEO_AUDIO_PROC_WAIT), PROGRESSBAR_MARQUEE);
        while (pe->aud_parallel.he_vid_start)
            vid_ret |= aud_parallel_task(oip, pe, use_internal);
        set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);
    }
    return vid_ret;
}

//並列処理スレッドの終了を待ち、終了コードを回収する
static AUO_RESULT exit_audio_parallel_control(const OUTPUT_INFO *oip, PRM_ENC *pe, BOOL use_internal, AUO_RESULT vid_ret) {
    vid_ret |= finish_aud_parallel_task(oip, pe, use_internal, vid_ret); //wav出力を完了させる
    release_audio_parallel_events(pe);
    if (pe->aud_parallel.buffer) free(pe->aud_parallel.buffer);
    if (pe->aud_parallel.th_aud) {
        //音声エンコードを完了させる
        //2passエンコードとかだと音声エンコーダの終了を待機する必要あり
        BOOL wait_for_audio = FALSE;
        while (WaitForSingleObject(pe->aud_parallel.th_aud, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (!wait_for_audio) {
                set_window_title(g_auo_mes.get(AUO_VIDEO_AUDIO_PROC_WAIT), PROGRESSBAR_MARQUEE);
                wait_for_audio = !wait_for_audio;
            }
            pe->aud_parallel.abort |= (oip->func_is_abort() ? TRUE : FALSE);
            log_process_events();
        }
        flush_audio_log();
        if (wait_for_audio)
            set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);

        DWORD exit_code = 0;
        //GetExitCodeThreadの返り値がNULLならエラー
        vid_ret |= (NULL == GetExitCodeThread(pe->aud_parallel.th_aud, &exit_code)) ? AUO_RESULT_ERROR : exit_code;
        CloseHandle(pe->aud_parallel.th_aud);
    }
    //初期化 (重要!!!)
    ZeroMemory(&pe->aud_parallel, sizeof(pe->aud_parallel));
    return vid_ret;
}

//auo_pipe.cppのread_from_pipeの特別版
static int ReadLogEnc(PIPE_SET *pipes, int total_drop, int current_frames) {
    DWORD pipe_read = 0;
    if (!PeekNamedPipe(pipes->stdErr.h_read, NULL, 0, NULL, &pipe_read, NULL))
        return -1;
    if (pipe_read) {
        ReadFile(pipes->stdErr.h_read, pipes->read_buf + pipes->buf_len, sizeof(pipes->read_buf) - pipes->buf_len - 1, &pipe_read, NULL);
        pipes->buf_len += pipe_read;
        write_log_enc_mes(pipes->read_buf, &pipes->buf_len, total_drop, current_frames, NULL);
    } else {
        log_process_events();
    }
    return pipe_read;
}

static std::unique_ptr<RGYSharedMemWin> video_create_param_mem(const OUTPUT_INFO *oip, bool afs, RGY_CSP out_csp, RGY_PICSTRUCT picstruct, HANDLE heBufEmpty[2], HANDLE heBufFilled[2]) {
    char sm_name[256];
    sprintf_s(sm_name, "%s_%08x", RGYInputSMPrmSM, GetCurrentProcessId());
    auto PrmSm = std::unique_ptr<RGYSharedMemWin>(new RGYSharedMemWin(sm_name, sizeof(RGYInputSMSharedData)));
    if (PrmSm->is_open()) {
        RGYInputSMSharedData *prmsm = (RGYInputSMSharedData *)PrmSm->ptr();
        prmsm->w = oip->w;
        prmsm->h = oip->h;
        prmsm->fpsN = oip->rate;
        prmsm->fpsD = oip->scale;
        prmsm->frames = (afs) ? 0 : oip->n;
        prmsm->picstruct = picstruct;
        prmsm->csp = out_csp;
        prmsm->abort = false;
        prmsm->afs = afs;
        for (int i = 0; i < 2; i++) {
            prmsm->heBufEmpty[i] = (uint64_t)heBufEmpty[i];
            prmsm->heBufFilled[i] = (uint64_t)heBufFilled[i];
            prmsm->duration[i] = 0;
            prmsm->timestamp[i] = 0;
            prmsm->dropped[i] = 0;
        }
    }
    return std::move(PrmSm);
}

static int video_create_event(HANDLE heBufEmpty[2], HANDLE heBufFilled[2]) {
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    for (int i = 0; i < 2; i++) {
        heBufEmpty[i] = CreateEventA(&sa, FALSE, FALSE, nullptr);
        //if (heBufEmpty[i] != NULL) {
        //    write_log_auo_line_fmt(LOG_MORE, "Created Event: %s", buf_event_name);
        //}
        heBufFilled[i] = CreateEventA(&sa, FALSE, FALSE, nullptr);
        //if (heBufFilled[i] != NULL) {
        //    write_log_auo_line_fmt(LOG_MORE, "Created Event: %s", buf_event_name);
        //}
    }
    return heBufEmpty[0] == NULL || heBufEmpty[1] == NULL || heBufFilled[0] == NULL || heBufFilled[1] == NULL;
}

static int send_frame(
    std::unique_ptr<RGYSharedMemWin>& inputbuf, void *const frame,
    const int i, const int sendFrame, const BOOL copy_frame, int *const next_jitter,
    const OUTPUT_INFO *oip, const PRM_ENC *pe,
    const RGY_CSP input_csp,
    const InEncodeVideoParam& enc_prm,
    RGYInputSMSharedData* const prmsm,
    RGYConvertCSP *const convert,
    std::unique_ptr<uint8_t, aligned_malloc_deleter>& tempBufForNonModWidth, int& tempBufForNonModWidthPitch) {
    void* dst_array[3];
    if (!inputbuf) {
        char sm_name[256];
        sprintf_s(sm_name, "%s_%08x_%d", RGYInputSMBuffer, GetCurrentProcessId(), sendFrame & 1);
        inputbuf = std::unique_ptr<RGYSharedMemWin>(new RGYSharedMemWin(sm_name, prmsm->bufSize));
        if (!inputbuf) {
            error_video_open_shared_input_buf();
            return AUO_RESULT_ERROR;
        }
        RGY_SIMD simd = RGY_SIMD::SIMD_ALL;
        if (prmsm->w % ((input_csp == RGY_CSP_YC48) ? 16 : 32) != 0) {
            if (prmsm->w % ((input_csp == RGY_CSP_YC48) ? 8 : 16) == 0) { //SSEで割り切れるならそちらを使う
                simd = RGY_SIMD::AVX | RGY_SIMD::POPCNT | RGY_SIMD::SSE42 | RGY_SIMD::SSE41 | RGY_SIMD::SSSE3 | RGY_SIMD::SSE2;
            } else {
                //SIMDの要求する値で割り切れない場合は、一時バッファを使用してpitchがあるようにする
                tempBufForNonModWidthPitch = ALIGN(oip->w, 128) * ((input_csp == RGY_CSP_YC48) ? 6 : ((input_csp == RGY_CSP_BGR24R) ? 3 : 2));
                tempBufForNonModWidth = std::unique_ptr<uint8_t, aligned_malloc_deleter>(
                    (uint8_t*)_aligned_malloc(tempBufForNonModWidthPitch * oip->h, 128), aligned_malloc_deleter());
            }
        }
        if (convert->getFunc(input_csp, prmsm->csp, false, simd) == nullptr) {
            error_video_get_conv_func();
            return AUO_RESULT_ERROR;
        }
        if (sendFrame == 0) {
            write_log_auo_line_fmt(RGY_LOG_INFO, L"Convert %s -> %s [%s]",
                tchar_to_wstring(RGY_CSP_NAMES[convert->getFunc()->csp_from]).c_str(),
                tchar_to_wstring(RGY_CSP_NAMES[convert->getFunc()->csp_to]).c_str(),
                tchar_to_wstring(get_simd_str(convert->getFunc()->simd)).c_str());
        }
    }
    dst_array[0] = inputbuf->ptr();
    dst_array[1] = (uint8_t*)dst_array[0] + prmsm->pitch * prmsm->h;
    switch (prmsm->csp) {
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        dst_array[2] = (uint8_t*)dst_array[1] + prmsm->pitch * prmsm->h / 4;
        break;
    case RGY_CSP_YUV422:
    case RGY_CSP_YUV422_09:
    case RGY_CSP_YUV422_10:
    case RGY_CSP_YUV422_12:
    case RGY_CSP_YUV422_14:
    case RGY_CSP_YUV422_16:
        dst_array[2] = (uint8_t*)dst_array[1] + prmsm->pitch * prmsm->h / 2;
        break;
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_09:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_12:
    case RGY_CSP_YUV444_14:
    case RGY_CSP_YUV444_16:
    case RGY_CSP_RGB:
        dst_array[2] = (uint8_t*)dst_array[1] + prmsm->pitch * prmsm->h;
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    default:
        break;
    }
    //コピーフレームの場合は、映像バッファの中身を更新せず、そのままパイプに流す
    if (!copy_frame) {
        uint8_t* ptr_src = (uint8_t*)frame;
        int src_pitch = (input_csp == RGY_CSP_YC48) ? oip->w * 6 : ((input_csp == RGY_CSP_BGR24R) ? oip->w * 3 : oip->w * 2);
        if (tempBufForNonModWidth) { //SIMDの要求する値で割り切れない場合は、一時バッファを使用してpitchがあるようにする
            for (int j = 0; j < oip->h; j++) {
                auto dst = tempBufForNonModWidth.get() + tempBufForNonModWidthPitch * j;
                auto src = (uint8_t*)frame + src_pitch * j;
                memcpy(dst, src, src_pitch);
            }
            src_pitch = tempBufForNonModWidthPitch;
            ptr_src = tempBufForNonModWidth.get();
        }
        const int src_uv_pitch = (input_csp == RGY_CSP_YC48) ? src_pitch : ((input_csp == RGY_CSP_BGR24R) ? src_pitch : src_pitch >> 1);
        int dummy[4] = { 0 };
        convert->run((enc_prm.input.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0,
            dst_array, (const void**)&ptr_src, oip->w,
            src_pitch,
            src_uv_pitch,
            prmsm->pitch, prmsm->pitch, oip->h, oip->h, dummy);
    }
    prmsm->timestamp[sendFrame & 1] = (int64_t)i * 4;
    prmsm->duration[sendFrame & 1] = 0;
    if (next_jitter) {
        prmsm->timestamp[sendFrame & 1] += next_jitter[-1];
    }
    prmsm->dropped[sendFrame & 1] = pe->drop_count;
    return AUO_RESULT_SUCCESS;
}

static void error_videnc_failed(const PRM_ENC *pe) {
    ULARGE_INTEGER temp_drive_avail_space = { 0 };
    const uint64_t disk_warn_threshold = 4 * 1024 * 1024; //4MB
    //指定されたドライブが存在するかどうか
    TCHAR temp_root[MAX_PATH_LEN];
    if (PathGetRoot(pe->temp_filename, temp_root, _countof(temp_root))
        && PathIsDirectory(temp_root)
        && GetDiskFreeSpaceEx(temp_root, &temp_drive_avail_space, NULL, NULL)
        && temp_drive_avail_space.QuadPart <= disk_warn_threshold) {
        TCHAR driveLetter[MAX_PATH_LEN];
        _tcscpy_s(driveLetter, temp_root);
        if (_tcslen(driveLetter) > 1 && driveLetter[_tcslen(driveLetter) - 1] == '\\') {
            driveLetter[_tcslen(driveLetter) - 1] = '\0';
        }
        if (_tcslen(driveLetter) > 1 && driveLetter[_tcslen(driveLetter) - 1] == ':') {
            driveLetter[_tcslen(driveLetter) - 1] = '\0';
        }
        error_videnc_dead_and_nodiskspace(driveLetter, temp_drive_avail_space.QuadPart);
    } else {
        error_videnc_dead();
    }
}

#pragma warning( push )
#pragma warning( disable: 4100 )
static DWORD video_output_inside(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //動画エンコードの必要がなければ終了
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return AUO_RESULT_SUCCESS;

    InEncodeVideoParam enc_prm;
    NV_ENC_CODEC_CONFIG codec_prm[RGY_CODEC_NUM] = { 0 };
    codec_prm[RGY_CODEC_H264] = DefaultParamH264();
    codec_prm[RGY_CODEC_HEVC] = DefaultParamHEVC();
    codec_prm[RGY_CODEC_AV1]  = DefaultParamAV1();
    parse_cmd(&enc_prm, codec_prm, conf->enc.cmd);
    enc_prm.encConfig.encodeCodecConfig = codec_prm[enc_prm.codec_rgy];

    if (!conf->enc.resize_enable) {
        enc_prm.input.dstWidth = 0;
        enc_prm.input.dstHeight = 0;
    }

    enc_prm.common.AVSyncMode = (conf->vid.afs) ? RGY_AVSYNC_VFR : RGY_AVSYNC_AUTO;
    enc_prm.common.disableMp4Opt = pe->muxer_to_be_used != MUXER_DISABLED;
    if (conf->vid.afs && enc_prm.vpp.afs.enable) {
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_VIDEO_AFS_AVIUTL_AND_VPP_CONFLICT1));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_VIDEO_AFS_AVIUTL_AND_VPP_CONFLICT2));
        return AUO_RESULT_ERROR;
    }

    if ((oip->flag & OUTPUT_INFO_FLAG_AUDIO)) {
        if_valid_wait_for_single_object(pe->aud_parallel.he_vid_start, INFINITE);
        auto common = &enc_prm.common;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        //音声
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
            TCHAR aud_filename[MAX_PATH_LEN];
            if (conf->aud.use_internal) {
                get_audio_pipe_name(aud_filename, _countof(aud_filename), i_aud);
            } else {
                get_aud_filename(aud_filename, _countof(aud_filename), pe, i_aud);
            }

            AudioSource src;
            src.filename = aud_filename;
            AudioSelect &chSel = src.select[0];
            if (!conf->aud.use_internal) {
                chSel.encCodec = _T("copy");
            } else {
                const CONF_AUDIO_BASE *cnf_aud = &conf->aud.in;
                const AUDIO_SETTINGS *aud_stg = &sys_dat->exstg->s_aud_int[cnf_aud->encoder];
                if (sys_dat->exstg->is_faw(aud_stg)) {
                    chSel.encCodec = _T("copy");
                } else {
                    if (aud_stg->mode[cnf_aud->enc_mode].bitrate) {
                        chSel.encBitrate = cnf_aud->bitrate;
                    }
                    chSel.encCodec = aud_stg->codec;
                }
            }
            common->audioSource.push_back(src);
        }
    }

    //chapterファイル
    TCHAR chap_file[MAX_PATH_LEN];
    TCHAR chap_apple[MAX_PATH_LEN];
    const MUXER_CMD_EX *muxer_mode = get_muxer_mode(conf, sys_dat, pe->muxer_to_be_used);
    if (muxer_mode && _tcslen(muxer_mode->chap_file) > 0) {
        set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), muxer_mode->chap_file, pe, sys_dat, conf, oip);
        if (!PathFileExists(chap_file)) {
            warning_mux_no_chapter_file();
        } else {
            enc_prm.common.chapterFile = chap_file;
        }
    }

    TCHAR exe_cmd[MAX_CMD_LEN] = { 0 };
    TCHAR exe_args[MAX_CMD_LEN] = { 0 };
    TCHAR exe_dir[MAX_PATH_LEN] = { 0 };

    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    PIPE_SET pipes = { 0 };
    PROCESS_INFORMATION pi_enc = { 0 };
    const BOOL afs = conf->vid.afs != 0;

    //プロセス用情報準備
    if (!PathFileExists(sys_dat->exstg->s_enc.fullpath)) {
        ret |= AUO_RESULT_ERROR; error_no_exe_file(ENCODER_NAME_W, sys_dat->exstg->s_enc.fullpath);
        return ret;
    }
    PathGetDirectory(exe_dir, _countof(exe_dir), sys_dat->exstg->s_enc.fullpath);

    //output csp
    RGY_CSP rgy_output_csp;
    bool output_highbit_depth;
    get_csp_and_bitdepth(output_highbit_depth, rgy_output_csp, conf);
    const bool interlaced = (enc_prm.input.picstruct & RGY_PICSTRUCT_INTERLACED) != 0;

    //YUY2/YC48->NV12/YUV444, RGBコピー用関数
    const int input_csp_idx = get_aviutl_color_format(output_highbit_depth, rgy_output_csp);
    const RGY_CSP input_csp = (input_csp_idx == CF_YC48) ? RGY_CSP_YC48 : ((input_csp_idx == CF_RGB) ? RGY_CSP_BGR24R : RGY_CSP_YUY2);

    //自動フィールドシフト関連
    if (pe->muxer_to_be_used != MUXER_DISABLED) {
        enc_prm.vpp.afs.timecode = false;
    }

    //コマンドライン生成
    build_full_cmd(exe_cmd, _countof(exe_cmd), conf, &enc_prm, oip, pe, sys_dat, PIPE_FN);
    write_log_auo_line_fmt(LOG_INFO, L"%s options...", ENCODER_APP_NAME_W);
    write_args(exe_cmd);
    _stprintf_s(exe_args, _countof(exe_args), _T("\"%s\" %s"), sys_dat->exstg->s_enc.fullpath, exe_cmd);
    _tremove(pe->temp_filename); //ファイルサイズチェックの時に旧ファイルを参照してしまうのを回避

    //パイプの設定
    pipes.stdErr.mode = AUO_PIPE_ENABLE;

    set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_CONTINUOUS);
    log_process_events();

    HANDLE heBufEmpty[2] = { NULL, NULL }, heBufFilled[2] = { NULL, NULL };
    std::unique_ptr<RGYSharedMemWin> prmSM;

    int *jitter = NULL;
    int rp_ret = 0;

    if (conf->vid.afs && interlaced) {
        ret |= AUO_RESULT_ERROR; error_afs_interlace_stg();
    } else if (conf->vid.afs && NULL == (jitter = (int *)calloc(oip->n + 1, sizeof(int)))) {
        ret |= AUO_RESULT_ERROR; error_malloc_tc();
    //Aviutl(afs)からのフレーム読み込み
    } else if (!setup_afsvideo(oip, sys_dat, conf, pe)) {
        ret |= AUO_RESULT_ERROR; //Aviutl(afs)からのフレーム読み込みに失敗
    //x264プロセス開始
    } else if (video_create_event(heBufEmpty, heBufFilled)) {
        ret |= AUO_RESULT_ERROR; error_video_create_event();
    } else if ((prmSM = video_create_param_mem(oip, afs, rgy_output_csp, enc_prm.input.picstruct, heBufEmpty, heBufFilled)) == nullptr || !prmSM->is_open()) {
        ret |= AUO_RESULT_ERROR; error_video_create_param_mem();
    } else if ((rp_ret = RunProcess(exe_args, exe_dir, &pi_enc, &pipes, GetPriorityClass(pe->h_p_aviutl), TRUE, FALSE)) != RP_SUCCESS) {
        ret |= AUO_RESULT_ERROR; error_run_process(ENCODER_APP_NAME_W, rp_ret);
    } else {
        //全て正常
        int i = 0;
        void *frame = NULL;
        int *next_jitter = NULL;
        bool enc_pause = false;
        BOOL copy_frame = false, drop = false;
        const DWORD aviutl_color_fmt = COLORFORMATS[get_aviutl_color_format(output_highbit_depth, rgy_output_csp)].FOURCC;
        double time_get_frame = 0.0;
        int64_t qp_freq, qp_start, qp_end;
        QueryPerformanceFrequency((LARGE_INTEGER *)&qp_freq);
        const double qp_freq_sec = 1.0 / (double)qp_freq;
        RGYInputSMSharedData *const prmsm = (RGYInputSMSharedData *)prmSM->ptr();
        std::unique_ptr<uint8_t, aligned_malloc_deleter> tempBufForNonModWidth;
        int tempBufForNonModWidthPitch = 0;
        int sendFrames = 0;
        std::array<std::unique_ptr<RGYSharedMemWin>, 2> inputbuf;
        auto convert = std::unique_ptr<RGYConvertCSP>(new RGYConvertCSP(std::min(MAX_CONV_THREADS, ((int)get_cpu_info().physical_cores + 3) / 4), RGYParamThread()));

        //Aviutlの時間を取得
        PROCESS_TIME time_aviutl;
        GetProcessTime(pe->h_p_aviutl, &time_aviutl);
        pe->h_p_videnc = pi_enc.hProcess;

        //x264が待機に入るまでこちらも待機
        while (WaitForInputIdle(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            log_process_events();
        if (conf->aud.use_internal)
            if_valid_set_event(pe->aud_parallel.he_aud_start);

        //ログウィンドウ側から制御を可能に
        DWORD tm_vid_enc_start = timeGetTime();
        enable_enc_control(nullptr, &enc_pause, afs, TRUE, tm_vid_enc_start, oip->n);

        //------------メインループ------------
        for (i = 0, next_jitter = jitter + 1, pe->drop_count = 0; i < oip->n; i++, next_jitter++) {
            //中断を確認
            ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;

            //x264が実行中なら、メッセージを取得・ログウィンドウに表示
            if (ReadLogEnc(&pipes, pe->drop_count, i) < 0) {
                //勝手に死んだ...
                ret |= AUO_RESULT_ERROR; error_videnc_failed(pe);
                break;
            }

            if (!(i & 7)) {
                //Aviutlの進捗表示を更新
                oip->func_rest_time_disp(i, oip->n);
                if (!conf->aud.use_internal) {
                    //音声同時処理
                    ret |= aud_parallel_task(oip, pe, conf->aud.use_internal);
                }
            }

            //一時停止
            while (enc_pause & !ret) {
                Sleep(LOG_UPDATE_INTERVAL);
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                ReadLogEnc(&pipes, pe->drop_count, i);
                log_process_events();
            }

            //中断・エラー等をチェック
            if (AUO_RESULT_SUCCESS != ret)
                break;

#if AVIUTL_TARGET_VER == 1
            //コピーフレームフラグ処理
            copy_frame = (!!i & (oip->func_get_flag(i) & OUTPUT_INFO_FRAME_FLAG_COPYFRAME));
#endif

            DWORD wait_err = WAIT_TIMEOUT;
            do {
                if (conf->aud.use_internal) {
                    //音声同時処理
                    ret |= aud_parallel_task(oip, pe, conf->aud.use_internal);
                }

                if (wait_err == WAIT_FAILED) {
                    error_video_wait_event();
                    ret |= AUO_RESULT_ABORT;
                    break;
                }

                //x264が実行中なら、メッセージを取得・ログウィンドウに表示
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                if (ReadLogEnc(&pipes, pe->drop_count, i) < 0) {
                    //勝手に死んだ...
                    ret |= AUO_RESULT_ERROR; error_videnc_failed(pe);
                    break;
                }
            } while ((wait_err = WaitForSingleObject(heBufEmpty[sendFrames & 1], LOG_UPDATE_INTERVAL)) != WAIT_OBJECT_0);
            if (ret != AUO_RESULT_SUCCESS)
                break;

            //Aviutl(afs)からフレームをもらう
            QueryPerformanceCounter((LARGE_INTEGER *)&qp_start);
            if (NULL == (frame = ((afs) ? afs_get_video((OUTPUT_INFO *)oip, i, &drop, next_jitter) : oip->func_get_video_ex(i, aviutl_color_fmt)))) {
                ret |= AUO_RESULT_ERROR; error_afs_get_frame();
                break;
            }
            QueryPerformanceCounter((LARGE_INTEGER *)&qp_end);
            time_get_frame += (qp_end - qp_start) * qp_freq_sec;

            drop |= (afs & copy_frame);

            if (!drop) {
                ret |= send_frame(inputbuf[sendFrames&1], frame, i, sendFrames, copy_frame, (jitter) ? next_jitter : nullptr,
                    oip, pe, input_csp, enc_prm, prmsm, convert.get(), tempBufForNonModWidth, tempBufForNonModWidthPitch);
                if (ret != AUO_RESULT_SUCCESS)
                    break;

                //完了通知
                if (SetEvent(heBufFilled[sendFrames & 1]) == FALSE) {
                    error_video_set_event();
                    ret |= AUO_RESULT_ERROR;
                    break;
                }
                sendFrames++;
            } else {
                *(next_jitter - 1) = DROP_FRAME_FLAG;
                pe->drop_count++;

                //もう一度取得するために
                if (SetEvent(heBufEmpty[sendFrames & 1]) == FALSE) {
                    error_video_set_event();
                    ret |= AUO_RESULT_ERROR;
                    break;
                }
            }

#if AVIUTL_TARGET_VER == 1
            // 「表示 -> セーブ中もプレビュー表示」がチェックされていると
            // func_update_preview() の呼び出しによって func_get_video_ex() の
            // 取得したバッファが書き換えられてしまうので、呼び出し位置を移動 (拡張AVI出力 plus より)
            oip->func_update_preview();
#endif
        }
        //------------メインループここまで--------------

        while ((WAIT_TIMEOUT == WaitForSingleObject(heBufEmpty[(sendFrames-1) & 1], LOG_UPDATE_INTERVAL))) {
            //x264が実行中なら、メッセージを取得・ログウィンドウに表示
            if (ReadLogEnc(&pipes, pe->drop_count, i) < 0) {
                //勝手に死んだ...
                ret |= AUO_RESULT_ERROR; error_videnc_failed(pe);
                break;
            }
            log_process_events();
        }
        prmsm->abort = true;
        SetEvent(heBufFilled[sendFrames & 1]);

        //ログウィンドウからのx264制御を無効化
        disable_enc_control();

        if (!ret) oip->func_rest_time_disp(oip->n, oip->n);

        //音声の同時処理を終了させる
        ret |= finish_aud_parallel_task(oip, pe, conf->aud.use_internal, ret);
        //音声との同時処理が終了
        release_audio_parallel_events(pe);

        //タイムコード出力
        if (!ret && ((afs && pe->muxer_to_be_used != MUXER_DISABLED) || conf->vid.auo_tcfile_out))
            tcfile_out(jitter, oip->n, (double)oip->rate / (double)oip->scale, afs, pe);

        //エンコーダ終了待機
        while (WaitForSingleObject(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            ReadLogEnc(&pipes, pe->drop_count, i);

        DWORD tm_vid_enc_fin = timeGetTime();

        //最後にメッセージを取得
        while (ReadLogEnc(&pipes, pe->drop_count, i) > 0);

        if (!(ret & AUO_RESULT_ERROR) && afs)
            write_log_auo_line_fmt(LOG_INFO, L"drop %d / %d frames", pe->drop_count, i);

        write_log_auo_line_fmt(LOG_INFO, L"%s: Aviutl: %.2f%% / %s: %.2f%%", g_auo_mes.get(AUO_VIDEO_CPU_USAGE), GetProcessAvgCPUUsage(pe->h_p_aviutl, &time_aviutl), ENCODER_APP_NAME_W, GetProcessAvgCPUUsage(pi_enc.hProcess));
        if (i > 0) write_log_auo_line_fmt(LOG_INFO, L"Aviutl %s: %.3f ms", g_auo_mes.get(AUO_VIDEO_AVIUTL_PROC_AVG_TIME), time_get_frame * 1000.0 / i);
        write_log_auo_enc_time(g_auo_mes.get(AUO_VIDEO_ENCODE_TIME), tm_vid_enc_fin - tm_vid_enc_start);
    }
    set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);

    for (int i = 0; i < 2; i++) {
        if (heBufEmpty[i]) CloseHandle(heBufEmpty[i]);
        if (heBufFilled[i]) CloseHandle(heBufFilled[i]);
    }

    //解放処理
    if (pipes.stdErr.mode)
        CloseHandle(pipes.stdErr.h_read);
    CloseHandle(pi_enc.hProcess);
    CloseHandle(pi_enc.hThread);
    pe->h_p_videnc = NULL;

    if (jitter) free(jitter);

    return ret;
}
#pragma warning( pop )

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    return exit_audio_parallel_control(oip, pe, conf->aud.use_internal, video_output_inside(conf, oip, pe, sys_dat));
}
