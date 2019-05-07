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
#pragma comment(lib, "user32.lib") //WaitforInputIdle
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>

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
#include "auo_convert.h"
#include "auo_video.h"
#include "auo_audio_parallel.h"

#include "NVEncParam.h"
#include "NVEncCmd.h"

static const int MAX_CONV_THREADS = 4;

struct alignas(64) video_convert_thread_t {
    void *frame;
    CONVERT_CF_DATA *pixel_data;
    func_convert_frame convert_frame;
    int w, h;
    int thread_id;
    int thread_n;
    BOOL abort;
    HANDLE thread;
    HANDLE he_conv_start;
    HANDLE he_conv_fin;
};

struct video_output_thread_t {
    CONVERT_CF_DATA *pixel_data;
    FILE *f_out;
    BOOL abort;
    HANDLE thread;
    HANDLE he_out_start;
    HANDLE he_out_fin;
    int repeat;
};

int get_aviutl_color_format(int use_highbit, RGY_CSP csp) {
    //Aviutlからの入力に使用するフォーマット
    RGY_CHROMAFMT chromafmt = RGY_CSP_CHROMA_FORMAT[csp];

    if (use_highbit) {
        return CF_YC48;
    }

    switch (chromafmt) {
    case RGY_CHROMAFMT_YUV444:
        return CF_YC48;
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    default:
        return CF_YUY2;
    }
}

void get_csp_and_bitdepth(bool& use_highbit, RGY_CSP& csp, const CONF_GUIEX *conf) {
    ParseCmdError err;
    InEncodeVideoParam enc_prm;
    NV_ENC_CODEC_CONFIG codec_prm[2] = { 0 };
    codec_prm[NV_ENC_H264] = DefaultParamH264();
    codec_prm[NV_ENC_HEVC] = DefaultParamHEVC();
    parse_cmd(&enc_prm, codec_prm, conf->nvenc.cmd, err);
    enc_prm.encConfig.encodeCodecConfig = codec_prm[enc_prm.codec];
    if (enc_prm.lossless) {
        enc_prm.yuv444 = true;
    }
    use_highbit = enc_prm.codec == NV_ENC_HEVC && enc_prm.encConfig.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 > 0;
    if (use_highbit) {
        csp = (enc_prm.yuv444) ? RGY_CSP_YUV444_16 : RGY_CSP_P010;
    } else {
        csp = (enc_prm.yuv444) ? RGY_CSP_YUV444 : RGY_CSP_NV12;
    }
}

static int calc_input_frame_size(int width, int height, int color_format, int& buf_size) {
    width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
    //widthが割り切れない場合、多めにアクセスが発生するので、そのぶんを確保しておく
    const DWORD pixel_size = COLORFORMATS[color_format].size;
    const DWORD simd_check = get_availableSIMD();
    const DWORD align_size = (simd_check & SSE2) ? ((simd_check & AVX2) ? 64 : 32) : 1;
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
    char auotcfile[MAX_PATH_LEN];
    FILE *tcfile = NULL;

    if (afs)
        fps *= 4; //afsなら4倍精度
    double tm_multi = 1000.0 / fps;

    //ファイル名作成
    apply_appendix(auotcfile, sizeof(auotcfile), pe->temp_filename, pe->append.tc);

    if (fopen_s(&tcfile, auotcfile, "wb") == NULL) {
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
            for (int i = 0; i < frame_n; i++)
                fprintf(tcfile, "%.6lf\r\n", i * tm_multi);
        }
        fclose(tcfile);
    } else {
        ret |= AUO_RESULT_ERROR; warning_auo_tcfile_failed();
    }
    return ret;
}

static void build_full_cmd(char *cmd, size_t nSize, const CONF_GUIEX *conf, const InEncodeVideoParam *encPrm, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const char *input) {
    //cliモードでない
    //自動設定の適用
    //apply_guiEx_auto_settings(&prm.x264, oip->w, oip->h, oip->rate, oip->scale, sys_dat->exstg->s_local.auto_ref_limit_by_level);
    //GUI部のコマンドライン生成
    strcpy_s(cmd, nSize, gen_cmd(encPrm, nullptr, false).c_str());
    //メッセージの発行
    if ((encPrm->encConfig.rcParams.vbvBufferSize != 0 || encPrm->encConfig.rcParams.vbvInitialDelay != 0) && conf->vid.afs) {
        write_log_auo_line(LOG_INFO, "自動フィールドシフト使用時はvbv設定は正確に反映されません。");
    }
    //キーフレーム検出を行い、そのQPファイルが存在し、かつ--qpfileの指定がなければ、それをqpfileで読み込む
    //出力ファイル
    sprintf_s(cmd + strlen(cmd), nSize - strlen(cmd), " -o \"%s\"", pe->temp_filename);
    //入力
    sprintf_s(cmd + strlen(cmd), nSize - strlen(cmd), " --y4m -i -");
}

static void set_pixel_data(CONVERT_CF_DATA *pixel_data, const CONF_GUIEX *conf, int w, int h, bool output_highbit_depth, RGY_CSP rgy_output_csp) {
    const int byte_per_pixel = (output_highbit_depth) ? sizeof(short) : sizeof(BYTE);
    ZeroMemory(pixel_data, sizeof(CONVERT_CF_DATA));
    switch (rgy_output_csp) {
    case OUT_CSP_YUY2: //yuy2 (YUV422)
        pixel_data->count = 1;
        pixel_data->size[0] = w * h * byte_per_pixel * 2;
        break;
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_16: //i444 (YUV444 planar)
        pixel_data->count = 3;
        pixel_data->size[0] = w * h * byte_per_pixel;
        pixel_data->size[1] = pixel_data->size[0];
        pixel_data->size[2] = pixel_data->size[0];
        break;
    case RGY_CSP_NV12: //nv12 (YUV420)
    case RGY_CSP_P010: //nv12 (YUV420)
    default:
        pixel_data->count = 2;
        pixel_data->size[0] = w * h * byte_per_pixel;
        pixel_data->size[1] = pixel_data->size[0] / 2;
        break;
    }
    //サイズの総和計算
    for (int i = 0; i < pixel_data->count; i++)
        pixel_data->total_size += pixel_data->size[i];
}

//並列処理時に音声データを取得する
AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    AUD_PARALLEL_ENC *aud_p = &pe->aud_parallel; //長いんで省略したいだけ
    if (aud_p->th_aud) {
        //---   排他ブロック 開始  ---> 音声スレッドが止まっていなければならない
        if_valid_wait_for_single_object(aud_p->he_vid_start, INFINITE);
        if (aud_p->he_vid_start && aud_p->get_length) {
            DWORD required_buf_size = aud_p->get_length * (DWORD)oip->audio_size;
            if (aud_p->buf_max_size < required_buf_size) {
                //メモリ不足なら再確保
                if (aud_p->buffer) free(aud_p->buffer);
                aud_p->buf_max_size = required_buf_size;
                if (NULL == (aud_p->buffer = malloc(aud_p->buf_max_size)))
                    aud_p->buf_max_size = 0; //ここのmallocエラーは次の分岐でAUO_RESULT_ERRORに設定
            }
            void *data_ptr = NULL;
            if (NULL == aud_p->buffer ||
                NULL == (data_ptr = oip->func_get_audio(aud_p->start, aud_p->get_length, &aud_p->get_length))) {
                ret = AUO_RESULT_ERROR; //mallocエラーかget_audioのエラー
            } else {
                //自前のバッファにコピーしてdata_ptrが破棄されても良いようにする
                memcpy(aud_p->buffer, data_ptr, aud_p->get_length * oip->audio_size);
            }
            //すでにTRUEなら変更しないようにする
            aud_p->abort |= oip->func_is_abort();
        }
        flush_audio_log();
        if_valid_set_event(aud_p->he_aud_start);
        //---   排他ブロック 終了  ---> 音声スレッドを開始
    }
    return ret;
}

//音声処理をどんどん回して終了させる
static AUO_RESULT finish_aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe, AUO_RESULT vid_ret) {
    //エラーが発生していたら音声出力ループをとめる
    pe->aud_parallel.abort |= (vid_ret != AUO_RESULT_SUCCESS);
    if (pe->aud_parallel.th_aud) {
        write_log_auo_line(LOG_INFO, "音声処理の終了を待機しています...");
        set_window_title("音声処理の終了を待機しています...", PROGRESSBAR_MARQUEE);
        while (pe->aud_parallel.he_vid_start)
            vid_ret |= aud_parallel_task(oip, pe);
        set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);
    }
    return vid_ret;
}

//並列処理スレッドの終了を待ち、終了コードを回収する
static AUO_RESULT exit_audio_parallel_control(const OUTPUT_INFO *oip, PRM_ENC *pe, AUO_RESULT vid_ret) {
    vid_ret |= finish_aud_parallel_task(oip, pe, vid_ret); //wav出力を完了させる
    release_audio_parallel_events(pe);
    if (pe->aud_parallel.buffer) free(pe->aud_parallel.buffer);
    if (pe->aud_parallel.th_aud) {
        //音声エンコードを完了させる
        //2passエンコードとかだと音声エンコーダの終了を待機する必要あり
        BOOL wait_for_audio = FALSE;
        while (WaitForSingleObject(pe->aud_parallel.th_aud, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (!wait_for_audio) {
                set_window_title("音声処理の終了を待機しています...", PROGRESSBAR_MARQUEE);
                wait_for_audio = !wait_for_audio;
            }
            pe->aud_parallel.abort |= oip->func_is_abort();
            log_process_events();
        }
        flush_audio_log();
        if (wait_for_audio)
            set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

        DWORD exit_code = 0;
        //GetExitCodeThreadの返り値がNULLならエラー
        vid_ret |= (NULL == GetExitCodeThread(pe->aud_parallel.th_aud, &exit_code)) ? AUO_RESULT_ERROR : exit_code;
        CloseHandle(pe->aud_parallel.th_aud);
    }
    //初期化 (重要!!!)
    ZeroMemory(&pe->aud_parallel, sizeof(pe->aud_parallel));
    return vid_ret;
}

static void write_y4m_header(FILE *fp, const OUTPUT_INFO *oip, RGY_CSP rgy_output_csp) {
    const char *y4m_csp = nullptr;
    switch (rgy_output_csp) {
    case RGY_CSP_YUV444:    y4m_csp = "444"; break;
    case RGY_CSP_YUV444_16: y4m_csp = "444p16"; break;
    case RGY_CSP_P010:      y4m_csp = "p010"; break;
    case RGY_CSP_NV12:
    default:                y4m_csp = "nv12"; break;
    }
    fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d C%s\n", oip->w, oip->h, oip->rate, oip->scale, y4m_csp);
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

static unsigned __stdcall video_convert_thread_func(void *prm) {
    video_convert_thread_t *th = reinterpret_cast<video_convert_thread_t *>(prm);
    CONVERT_CF_DATA *pixel_data = th->pixel_data;
    WaitForSingleObject(th->he_conv_start, INFINITE);
    while (false == th->abort) {
        th->convert_frame(th->frame, th->pixel_data, th->w, th->h, th->thread_id, th->thread_n);  /// YUY2/YC48->NV12/YUV444変換, RGBコピー
        SetEvent(th->he_conv_fin);
        WaitForSingleObject(th->he_conv_start, INFINITE);
    }
    return 0;
}

static int video_convert_create_thread(std::vector<video_convert_thread_t>& thread_data, CONVERT_CF_DATA *pixel_data, const OUTPUT_INFO *oip, const func_convert_frame convert_frame) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    const auto thread_n = min(MAX_CONV_THREADS, ((int)get_cpu_info().physical_cores + 3) / 4);
    thread_data.resize(thread_n-1);
    for (int ith = 1; ith < thread_n; ith++) {
        video_convert_thread_t& th = thread_data[ith-1];
        th.thread_id = ith;
        th.thread_n = thread_n;
        th.pixel_data = pixel_data;
        th.convert_frame = convert_frame;
        th.frame = nullptr;
        th.w = oip->w;
        th.h = oip->h;
        th.abort = false;
        if (   NULL == (th.he_conv_start = (HANDLE)CreateEvent(NULL, false, false, NULL))
            || NULL == (th.he_conv_fin   = (HANDLE)CreateEvent(NULL, false, true, NULL))
            || NULL == (th.thread        = (HANDLE)_beginthreadex(NULL, 0, video_convert_thread_func, &th, 0, NULL))) {
            ret = AUO_RESULT_ERROR;
            break;
        }
    }
    return ret;
}

static void video_convert_close_thread(std::vector<video_convert_thread_t> &thread_data, AUO_RESULT ret) {
    for (int ith = 0; ith < thread_data.size(); ith++) {
        video_convert_thread_t &th = thread_data[ith];
        th.abort = true;
        SetEvent(th.he_conv_start);
        WaitForSingleObject(th.thread, INFINITE);
        CloseHandle(th.thread);
        CloseHandle(th.he_conv_start);
        CloseHandle(th.he_conv_fin);
    }
}

static void convert_frame_threads(void *frame, std::vector<video_convert_thread_t> &thread_data, CONVERT_CF_DATA *pixel_data, const OUTPUT_INFO *oip, const func_convert_frame convert_frame) {
    const int thread_n = (int)thread_data.size() + 1;
    HANDLE heConvfinCopy[MAX_CONV_THREADS];
    for (int ith = 1; ith < thread_n; ith++) {
        thread_data[ith-1].frame = frame;
        SetEvent(thread_data[ith-1].he_conv_start);
        heConvfinCopy[ith-1] = thread_data[ith-1].he_conv_fin;
    }
    convert_frame(frame, pixel_data, oip->w, oip->h, 0, thread_n);  /// YUY2/YC48->NV12/YUV444変換, RGBコピー
    if (thread_n > 1) {
        WaitForMultipleObjects(thread_n-1, heConvfinCopy, TRUE, INFINITE);
    }
}

static unsigned __stdcall video_output_thread_func(void *prm) {
    video_output_thread_t *thread_data = reinterpret_cast<video_output_thread_t *>(prm);
    CONVERT_CF_DATA *pixel_data = thread_data->pixel_data;
    WaitForSingleObject(thread_data->he_out_start, INFINITE);
    while (false == thread_data->abort) {
        //映像データをパイプに
        for (int i = 0; i < 1 + thread_data->repeat; i++) {
            const char *FRAME_HEADER = "FRAME\n";
            _fwrite_nolock(FRAME_HEADER, 1, strlen(FRAME_HEADER), thread_data->f_out);
            for (int j = 0; j < pixel_data->count; j++) {
                _fwrite_nolock((void *)pixel_data->data[j], 1, pixel_data->size[j], thread_data->f_out);
            }
        }

        thread_data->repeat = 0;
        SetEvent(thread_data->he_out_fin);
        WaitForSingleObject(thread_data->he_out_start, INFINITE);
    }
    return 0;
}

static int video_output_create_thread(video_output_thread_t *thread_data, CONVERT_CF_DATA *pixel_data, FILE *pipe_stdin) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    thread_data->abort = false;
    thread_data->pixel_data = pixel_data;
    thread_data->f_out = pipe_stdin;
    if (NULL == (thread_data->he_out_start = (HANDLE)CreateEvent(NULL, false, false, NULL))
        || NULL == (thread_data->he_out_fin   = (HANDLE)CreateEvent(NULL, false, true, NULL))
        || NULL == (thread_data->thread       = (HANDLE)_beginthreadex(NULL, 0, video_output_thread_func, thread_data, 0, NULL))) {
        ret = AUO_RESULT_ERROR;
    }
    return ret;
}

static void video_output_close_thread(video_output_thread_t *thread_data, AUO_RESULT ret) {
    if (thread_data->thread) {
        if (!ret)
            while (WAIT_TIMEOUT == WaitForSingleObject(thread_data->he_out_fin, LOG_UPDATE_INTERVAL))
                log_process_events();
        thread_data->abort = true;
        SetEvent(thread_data->he_out_start);
        WaitForSingleObject(thread_data->thread, INFINITE);
        CloseHandle(thread_data->thread);
        CloseHandle(thread_data->he_out_start);
        CloseHandle(thread_data->he_out_fin);
    }
    memset(thread_data, 0, sizeof(thread_data[0]));
}

#pragma warning( push )
#pragma warning( disable: 4100 )
static DWORD video_output_inside(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //動画エンコードの必要がなければ終了
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return AUO_RESULT_SUCCESS;

    ParseCmdError err;
    InEncodeVideoParam enc_prm;
    NV_ENC_CODEC_CONFIG codec_prm[2] = { 0 };
    codec_prm[NV_ENC_H264] = DefaultParamH264();
    codec_prm[NV_ENC_HEVC] = DefaultParamHEVC();
    parse_cmd(&enc_prm, codec_prm, conf->nvenc.cmd, err);
    enc_prm.encConfig.encodeCodecConfig = codec_prm[enc_prm.codec];

    if (conf->vid.afs && enc_prm.vpp.afs.enable) {
        write_log_auo_line(LOG_ERROR, "Aviutlの自動フィールドシフトとNVEnc Vppの自動フィールドシフトは併用できません。");
        write_log_auo_line(LOG_ERROR, "どちらかを選択してからやり直してください。");
        return AUO_RESULT_ERROR;
    }

    char exe_cmd[MAX_CMD_LEN] = { 0 };
    char exe_args[MAX_CMD_LEN] = { 0 };
    char exe_dir[MAX_PATH_LEN] = { 0 };

    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    PIPE_SET pipes = { 0 };
    PROCESS_INFORMATION pi_enc = { 0 };
    const BOOL afs = conf->vid.afs != 0;

    //プロセス用情報準備
    if (!PathFileExists(sys_dat->exstg->s_vid.fullpath)) {
        ret |= AUO_RESULT_ERROR; error_no_exe_file(ENCODER_NAME, sys_dat->exstg->s_vid.fullpath);
        return ret;
    }
    PathGetDirectory(exe_dir, _countof(exe_dir), sys_dat->exstg->s_vid.fullpath);

    //output csp
    const int output_csp = (enc_prm.yuv444) ? OUT_CSP_YUV444 : OUT_CSP_NV12;
    RGY_CSP rgy_output_csp;
    bool output_highbit_depth;
    get_csp_and_bitdepth(output_highbit_depth, rgy_output_csp, conf);
    const bool interlaced = (enc_prm.input.picstruct & RGY_PICSTRUCT_INTERLACED) != 0;

    CONVERT_CF_DATA pixel_data = { 0 };
    set_pixel_data(&pixel_data, conf, oip->w, oip->h, output_highbit_depth, rgy_output_csp);

    //YUY2/YC48->NV12/YUV444, RGBコピー用関数
    const int input_csp_idx = get_aviutl_color_format(output_highbit_depth, rgy_output_csp);
    const func_convert_frame convert_frame = get_convert_func(oip->w, input_csp_idx, (output_highbit_depth) ? 16 : 8, interlaced, output_csp);
    if (convert_frame == NULL) {
        ret |= AUO_RESULT_ERROR; error_select_convert_func(oip->w, oip->h, output_highbit_depth, interlaced, output_csp);
        return ret;
    }
    //映像バッファ用メモリ確保
    if (!malloc_pixel_data(&pixel_data, oip->w, oip->h, output_csp, (output_highbit_depth) ? 16 : 8)) {
        ret |= AUO_RESULT_ERROR; error_malloc_pixel_data();
        return ret;
    }

    //コマンドライン生成
    build_full_cmd(exe_cmd, _countof(exe_cmd), conf, &enc_prm, oip, pe, sys_dat, PIPE_FN);
    write_log_auo_line(LOG_INFO, "NVEncC options...");
    write_args(exe_cmd);
    sprintf_s(exe_args, _countof(exe_args), "\"%s\" %s", sys_dat->exstg->s_vid.fullpath, exe_cmd);
    remove(pe->temp_filename); //ファイルサイズチェックの時に旧ファイルを参照してしまうのを回避

    //パイプの設定
    pipes.stdIn.mode = AUO_PIPE_ENABLE;
    pipes.stdErr.mode = AUO_PIPE_ENABLE;
    pipes.stdIn.bufferSize = pixel_data.total_size * 2;

    DWORD tm_start = timeGetTime();
    set_window_title("NVEnc エンコード", PROGRESSBAR_CONTINUOUS);
    log_process_events();

    std::vector<video_convert_thread_t> thread_conv;
    video_output_thread_t thread_data = { 0 };

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
    } else if ((rp_ret = RunProcess(exe_args, exe_dir, &pi_enc, &pipes, GetPriorityClass(pe->h_p_aviutl), TRUE, FALSE)) != RP_SUCCESS) {
        ret |= AUO_RESULT_ERROR; error_run_process("NVEncC", rp_ret);
    //変換スレッドを開始
    } else if (video_convert_create_thread(thread_conv, &pixel_data, oip, convert_frame)) {
        ret |= AUO_RESULT_ERROR; error_video_convert_thread_start();
    //書き込みスレッドを開始
    } else if (video_output_create_thread(&thread_data, &pixel_data, pipes.f_stdin)) {
        ret |= AUO_RESULT_ERROR; error_video_output_thread_start();
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

        //Aviutlの時間を取得
        PROCESS_TIME time_aviutl;
        GetProcessTime(pe->h_p_aviutl, &time_aviutl);

        //x264が待機に入るまでこちらも待機
        while (WaitForInputIdle(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            log_process_events();

        write_y4m_header(pipes.f_stdin, oip, rgy_output_csp);

        //ログウィンドウ側から制御を可能に
        DWORD tm_vid_enc_start = timeGetTime();
        enable_enc_control(&enc_pause, afs, TRUE, tm_vid_enc_start, oip->n);

        //------------メインループ------------
        for (i = 0, next_jitter = jitter + 1, pe->drop_count = 0; i < oip->n; i++, next_jitter++) {
            //中断を確認
            ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;

            //x264が実行中なら、メッセージを取得・ログウィンドウに表示
            if (ReadLogEnc(&pipes, pe->drop_count, i) < 0) {
                //勝手に死んだ...
                ret |= AUO_RESULT_ERROR; error_x264_dead();
                break;
            }

            if (!(i & 7)) {
                //Aviutlの進捗表示を更新
                oip->func_rest_time_disp(i, oip->n);

                //音声同時処理
                ret |= aud_parallel_task(oip, pe);
            }

            //一時停止
            while (enc_pause & !ret) {
                Sleep(LOG_UPDATE_INTERVAL);
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                ReadLogEnc(&pipes, pe->drop_count, i);
                log_process_events();
            }

            //標準入力への書き込み完了をチェック
            while (WAIT_TIMEOUT == WaitForSingleObject(thread_data.he_out_fin, LOG_UPDATE_INTERVAL)) {
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                ReadLogEnc(&pipes, pe->drop_count, i);
                log_process_events();
            }

            //中断・エラー等をチェック
            if (AUO_RESULT_SUCCESS != ret)
                break;

            //コピーフレームフラグ処理
            copy_frame = (!!i & (oip->func_get_flag(i) & OUTPUT_INFO_FRAME_FLAG_COPYFRAME));

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
                //コピーフレームの場合は、映像バッファの中身を更新せず、そのままパイプに流す
                if (!copy_frame)
                    convert_frame_threads(frame, thread_conv, &pixel_data, oip, convert_frame);
                //標準入力への書き込みを開始
                SetEvent(thread_data.he_out_start);
            } else {
                *(next_jitter - 1) = DROP_FRAME_FLAG;
                pe->drop_count++;
                //次のフレームの変換を許可
                SetEvent(thread_data.he_out_fin);
            }

            // 「表示 -> セーブ中もプレビュー表示」がチェックされていると
            // func_update_preview() の呼び出しによって func_get_video_ex() の
            // 取得したバッファが書き換えられてしまうので、呼び出し位置を移動 (拡張AVI出力 plus より)
            oip->func_update_preview();
        }
        //------------メインループここまで--------------

        //書き込みスレッドを終了
        video_output_close_thread(&thread_data, ret);
        //変換用スレッドを終了
        video_convert_close_thread(thread_conv, ret);

        //ログウィンドウからのx264制御を無効化
        disable_enc_control();

        //パイプを閉じる
        CloseStdIn(&pipes);

        if (!ret) oip->func_rest_time_disp(oip->n, oip->n);

        //音声の同時処理を終了させる
        ret |= finish_aud_parallel_task(oip, pe, ret);
        //音声との同時処理が終了
        release_audio_parallel_events(pe);

        //タイムコード出力
        if (!ret && (afs || conf->vid.auo_tcfile_out))
            tcfile_out(jitter, oip->n, (double)oip->rate / (double)oip->scale, afs, pe);

        //エンコーダ終了待機
        while (WaitForSingleObject(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            ReadLogEnc(&pipes, pe->drop_count, i);

        DWORD tm_vid_enc_fin = timeGetTime();

        //最後にメッセージを取得
        while (ReadLogEnc(&pipes, pe->drop_count, i) > 0);

        if (!(ret & AUO_RESULT_ERROR) && afs)
            write_log_auo_line_fmt(LOG_INFO, "drop %d / %d frames", pe->drop_count, i);

        write_log_auo_line_fmt(LOG_INFO, "CPU使用率: Aviutl: %.2f%% / NVEnc: %.2f%%", GetProcessAvgCPUUsage(pe->h_p_aviutl, &time_aviutl), GetProcessAvgCPUUsage(pi_enc.hProcess));
        write_log_auo_line_fmt(LOG_INFO, "Aviutl 平均フレーム取得時間: %.3f ms", time_get_frame * 1000.0 / i);
        write_log_auo_enc_time("NVEncエンコード時間", tm_vid_enc_fin - tm_vid_enc_start);
    }
    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

    if (jitter) free(jitter);

    return ret;
}
#pragma warning( pop )

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    return exit_audio_parallel_control(oip, pe, video_output_inside(conf, oip, pe, sys_dat));
}
