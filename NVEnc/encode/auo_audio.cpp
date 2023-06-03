// -----------------------------------------------------------------------------------------
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
#include <string>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <process.h>
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "user32.lib") //WaitforInputIdle
#include <fcntl.h>
#include <io.h>
#include <mutex>

#include "output.h"
#include "auo.h"
#include "auo_version.h"
#include "auo_convert.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "fawcheck.h"
#include "auo_faw2aac.h"
#include "auo_runbat.h"
#include "auo_mes.h"

#include "auo_audio_parallel.h"
#include "auo_encode.h"
#include "exe_version.h"
#include "cpu_info.h"

const int WAVE_HEADER_SIZE = 44;
const int RIFF_SIZE_POS    = 4;
const int WAVE_SIZE_POS    = WAVE_HEADER_SIZE - 4;

inline void *get_audio_data(const OUTPUT_INFO *oip, PRM_ENC *pe, int start, int length, int *readed) {
    if (pe->aud_parallel.th_aud) {
        pe->aud_parallel.start = start;
        pe->aud_parallel.get_length = length;
        if_valid_set_event(pe->aud_parallel.he_vid_start);
        if_valid_wait_for_single_object(pe->aud_parallel.he_aud_start, INFINITE);
        if (pe->aud_parallel.he_aud_start) {
            *readed = pe->aud_parallel.get_length;
            return pe->aud_parallel.buffer;
        }
    }
    return oip->func_get_audio(start, length, readed);
}

void auo_faw_check(CONF_AUDIO *aud, const OUTPUT_INFO *oip, PRM_ENC *pe, const guiEx_settings *ex_stg) {
    if (!(oip->flag & OUTPUT_INFO_FLAG_AUDIO))
        return;
    const int faw_index = ex_stg->get_faw_index(aud->use_internal);
    if (faw_index == FAW_INDEX_ERROR) {
        write_log_auo_line_fmt(LOG_WARNING, L"FAWCheck : %s", g_auo_mes.get(AUO_AUDIO_FAW_INDEX_ERR));
        return;
    }
    int n = 0;
    short *dat = (short *)get_audio_data(oip, pe, 0, std::min(oip->audio_n, 10 * oip->audio_rate), &n);
    int ret = FAWCheck(dat, n, oip->audio_rate, oip->audio_size);
    switch (ret) {
        case NON_FAW:
            write_log_auo_line(LOG_INFO, L"FAWCheck : non-FAW");
            break;
        case FAW_FULL:
        case FAW_HALF:
        case FAW_MIX: {
            const AUDIO_SETTINGS *aud_stg = nullptr;
            if (aud->use_internal) {
                aud_stg = &ex_stg->s_aud_int[faw_index];
                aud->in.encoder = faw_index;
                aud->in.enc_mode = ret - FAW_FULL;
            } else {
                aud_stg = &ex_stg->s_aud_ext[faw_index];
                aud->ext.encoder = faw_index;
                aud->ext.enc_mode = ret - FAW_FULL;
                aud->ext.use_2pass = aud_stg->mode[aud->ext.enc_mode].enc_2pass;
                aud->ext.use_wav = !aud_stg->pipe_input;
            }
            write_log_auo_line_fmt(LOG_INFO, L"FAWCheck : FAW, %s", char_to_wstring(FAW_TYPE_NAME[ret]).c_str());
            } break;
        case FAWCHECK_ERROR_TOO_SHORT:
            write_log_auo_line_fmt(LOG_WARNING, L"FAWCheck : %s", g_auo_mes.get(AUO_AUDIO_ERR_TOO_SHORT));
            break;
        case FAWCHECK_ERROR_OTHER:
        default:
            write_log_auo_line_fmt(LOG_WARNING, L"FAWCheck : %s", g_auo_mes.get(AUO_AUDIO_ERR_OTHER));
            break;
    }
}

void check_audio_length(OUTPUT_INFO *oip) {
    const double video_length = oip->n * (double)oip->scale / oip->rate;
    const double audio_length = oip->audio_n / (double)oip->audio_rate;
    if (video_length <= 1.0) { // 1秒未満はチェックしない
        return;
    }
    const double audio_ratio = audio_length / video_length;
    if (!check_range(audio_ratio, 0.95, 1.05)) { // 5%以上 差がある場合
        const BOOL exedit_is_used = check_if_exedit_is_used();
        int selected_audio_rate = 0;
        if (exedit_is_used
            && audio_ratio > 1.0) { // 音声の長さが長い場合にはカットできるが、音声が短い場合には調整できない
            //拡張編集で音声を読み込ませたあと、異なるサンプリングレートの音声をAviutl本体に読み込ませると、
            //音声のサンプル数はそのままに、サンプリングレートだけが変わってしまい、音声の時間が変わってしまうことがある
            //拡張編集使用時に、映像と音声の長さにずれがある場合、これを疑ってサンプリングレートのずれの可能性がある場合は
            //音声のサンプル数を修正する
            const int estimated_audio_rate = (int)div_round((int64_t)oip->audio_n * oip->rate, (int64_t)oip->n * oip->scale);
            static const int known_audio_rate[] = { 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000, 176400, 192000 };
            for (int i = 0; i < _countof(known_audio_rate); i++) {
                if (estimated_audio_rate == known_audio_rate[i]) {
                    selected_audio_rate = estimated_audio_rate;
                }
            }
        }
        if (selected_audio_rate != 0) {
            oip->audio_n = (int)div_round((int64_t)oip->audio_n * (int64_t)oip->audio_rate, (int64_t)selected_audio_rate);
            info_audio_length_changed(video_length, audio_length, exedit_is_used);
        } else { // 5%以上差がある場合
            warning_audio_length(video_length, audio_length, exedit_is_used);
        }
    }
}

static void build_wave_header(BYTE *head, const OUTPUT_INFO *oip, BOOL use_8bit, int sample_n) {
    static const char * const RIFF_HEADER = "RIFF";
    static const char * const WAVE_HEADER = "WAVE";
    static const char * const FMT_CHUNK   = "fmt ";
    static const char * const DATA_CHUNK  = "data";
    const DWORD FMT_SIZE    = 16;
    const short FMT_ID      = 1;
    const int   size        = (use_8bit) ? sizeof(BYTE) : sizeof(short);

    memcpy(   head +  0, RIFF_HEADER, strlen(RIFF_HEADER));
    *(DWORD*)(head +  4) = sample_n * (size * oip->audio_ch) + WAVE_HEADER_SIZE - 8;
    memcpy(   head +  8, WAVE_HEADER, strlen(WAVE_HEADER));
    memcpy(   head + 12, FMT_CHUNK, strlen(FMT_CHUNK));
    *(DWORD*)(head + 16) = FMT_SIZE;
    *(short*)(head + 20) = FMT_ID;
    *(short*)(head + 22) = (short)oip->audio_ch;
    *(DWORD*)(head + 24) = oip->audio_rate;
    *(DWORD*)(head + 28) = oip->audio_rate * oip->audio_ch * size;
    *(short*)(head + 32) = (short)(size * oip->audio_ch);
    *(short*)(head + 34) = (short)(size * 8);
    memcpy(   head + 36, DATA_CHUNK, strlen(DATA_CHUNK));
    *(DWORD*)(head + 40) = sample_n * (size * oip->audio_ch);
    //計44byte(WAVE_HEADER_SIZE)
}

static void correct_header(FILE *f_out, int data_size) {
    //2箇所の出力データサイズ部分を書き換え
    int riff_size = data_size + (WAVE_SIZE_POS - RIFF_SIZE_POS);
    _fseeki64(f_out, RIFF_SIZE_POS, SEEK_SET);
    fwrite(&riff_size, sizeof(int), 1, f_out);
    _fseeki64(f_out, WAVE_SIZE_POS - RIFF_SIZE_POS, SEEK_CUR);
    fwrite(&data_size, sizeof(int), 1, f_out);
}

typedef struct {
    int id;
    char wavfile[MAX_PATH_LEN];
    char audfile[MAX_PATH_LEN];
    char cmd[MAX_CMD_LEN];
    char args[MAX_CMD_LEN];
    char append[MAX_APPENDIX_LEN];

    BOOL is_internal;
    HANDLE h_aud_namedpipe;
    HANDLE he_ov_aud_namedpipe;
    FILE *fp_out;
    PIPE_SET pipes;
    PROCESS_INFORMATION pi_aud;
    LOG_CACHE log_line_cache;
} aud_data_t;

static size_t write_file(aud_data_t *aud_dat, const PRM_ENC *pe, const void *buf, size_t size) {
    if (aud_dat->is_internal) {
        OVERLAPPED overlapped;
        memset(&overlapped, 0, sizeof(overlapped));
        overlapped.hEvent = aud_dat->he_ov_aud_namedpipe;
        DWORD sizeWritten = 0;
        //非同期処理中は0を返すことがある
        WriteFile(aud_dat->h_aud_namedpipe, buf, size, &sizeWritten, &overlapped);
        while (WaitForSingleObject(aud_dat->he_ov_aud_namedpipe, 1000) != WAIT_OBJECT_0) {
            if (pe->aud_parallel.abort) {
                return 0;
            }
        }
        return sizeWritten;
    } else {
        return _fwrite_nolock(buf, 1, size, aud_dat->fp_out);
    }
}

static void write_wav_header(aud_data_t *aud_dat, const OUTPUT_INFO *oip, const PRM_ENC *pe, BOOL use_8bit) {
    BYTE head[WAVE_HEADER_SIZE];
    build_wave_header(head, oip, use_8bit, oip->audio_n);
    write_file(aud_dat, pe, &head, sizeof(head));
}

static void make_wavfilename(aud_data_t *aud_dat, BOOL use_pipe, const char *tempfilename, const char *append_wav) {
    if (use_pipe)
        strcpy_s(aud_dat->wavfile, _countof(aud_dat->wavfile), PIPE_FN);
    else {
        apply_appendix(aud_dat->wavfile, _countof(aud_dat->wavfile), tempfilename, append_wav);
        if (aud_dat->id)
            insert_before_ext(aud_dat->wavfile, _countof(aud_dat->wavfile), aud_dat->id);
    }
}

static void build_audcmd(aud_data_t *aud_dat, const CONF_GUIEX *conf, const AUDIO_SETTINGS *aud_stg,
                         const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const OUTPUT_INFO *oip) {
    const CONF_AUDIO_BASE *aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;
    const DWORD nSize = _countof(aud_dat->cmd);
    strcpy_s(aud_dat->cmd, nSize, aud_stg->cmd_base);
    //%{2pass_cmd}
    replace(aud_dat->cmd, nSize, "%{2pass_cmd}", (aud->use_2pass) ? aud_stg->cmd_2pass : "");
    //%{raw_cmd}
    replace(aud_dat->cmd, nSize, "%{raw_cmd}", (aud->delay_cut == AUDIO_DELAY_CUT_EDTS) ? aud_stg->cmd_raw : "");
    //%{mode}
    replace(aud_dat->cmd, nSize, "%{mode}", aud_stg->mode[aud->enc_mode].cmd);
    //%{wavpath}
    replace(aud_dat->cmd, nSize, "%{wavpath}", aud_dat->wavfile);
    //%{rate}
    char tmp[22];
    sprintf_s(tmp, _countof(tmp), "%d", aud->bitrate);
    replace(aud_dat->cmd, nSize, "%{rate}", tmp);

    //音声番号に合わせ、置換キーを調整
    if (aud_dat->id) {
        char aud_key[128] = "%{audpath}";
        insert_num_to_replace_key(aud_key, _countof(aud_key), aud_dat->id);
        replace(aud_dat->cmd, nSize, "%{audpath}", aud_key);
    }

    cmd_replace(aud_dat->cmd, nSize, pe, sys_dat, conf, oip);
}

static void show_progressbar(BOOL use_pipe, const wchar_t *enc_name, int progress_mode) {
    wchar_t mes[1024];
    if (use_pipe)
        swprintf_s(mes, _countof(mes), L"%s %s", enc_name, g_auo_mes.get(AUO_AUDIO_PROGRESS_BAR_ENCODE));
    else
        wcscpy_s(mes, _countof(mes), g_auo_mes.get(AUO_AUDIO_PROGRESS_BAR_WAV_OUT));
    set_window_title(mes, progress_mode);
}

static void show_audio_delay_cut_info(int delay_cut, const PRM_ENC *pe) {
    if (AUDIO_DELAY_CUT_EDTS == delay_cut) {
        write_log_auo_line_fmt(LOG_INFO, L"%s - %s", g_auo_mes.get(AUO_AUDIO_DELAY_CUT), AUDIO_DELAY_CUT_MODE[AUDIO_DELAY_CUT_EDTS].desc);
    } else if (0 != pe->delay_cut_additional_aframe || 0 != pe->delay_cut_additional_vframe) {
        wchar_t message[1024] = { 0 };
        int mes_len = 0;
        mes_len += swprintf_s(message, _countof(message), L"%s - ", g_auo_mes.get(AUO_AUDIO_DELAY_CUT));
        if (pe->delay_cut_additional_vframe) {
            mes_len += swprintf_s(message + mes_len, _countof(message) - mes_len, L"%s%dframe%s",
                (0  < pe->delay_cut_additional_vframe) ? L"+" : L"",
                pe->delay_cut_additional_vframe,
                (1 < abs(pe->delay_cut_additional_vframe)) ? L"s" : L"");
        }
        if (pe->delay_cut_additional_vframe && pe->delay_cut_additional_aframe) {
            mes_len += swprintf_s(message + mes_len, _countof(message) - mes_len, L", ");
        }
        if (pe->delay_cut_additional_aframe) {
            mes_len += swprintf_s(message + mes_len, _countof(message) - mes_len, L"%s%dsample%s",
                (0  < pe->delay_cut_additional_aframe) ? L"+" : L"",
                pe->delay_cut_additional_aframe,
                (1 < abs(pe->delay_cut_additional_aframe)) ? L"s" : L"");
        }
        write_log_auo_line(LOG_INFO, message);
    }
}

static void show_audio_enc_info(const AUDIO_SETTINGS *aud_stg, const CONF_AUDIO_BASE *cnf_aud, const PRM_ENC *pe, const aud_data_t *aud_dat) {
    std::string ver_str = "";
    int version[4] = { 0 };
    if (str_has_char(aud_stg->cmd_ver) && 0 == get_exe_version_from_cmd(aud_stg->fullpath, aud_stg->cmd_ver, version)) {
        ver_str = " (" + ver_string(version) + ")";
    }

    wchar_t bitrate[128] = { 0 };
    if (aud_stg->mode[cnf_aud->enc_mode].bitrate)
        swprintf_s(bitrate, _countof(bitrate), L", %dkbps", cnf_aud->bitrate);
    const wchar_t *use2pass = (cnf_aud->use_2pass) ? L", 2pass" : L"";
    write_log_auo_line_fmt(LOG_INFO, L"%s%s %s%s%s%s",
        aud_stg->dispname, char_to_wstring(ver_str).c_str(),
        g_auo_mes.get(AUO_AUDIO_START_ENCODE), aud_stg->mode[cnf_aud->enc_mode].name, bitrate, use2pass);
    show_audio_delay_cut_info(cnf_aud->delay_cut, pe);
    if (strlen(aud_dat->args) > 0) {
        write_log_auo_line(LOG_DEBUG, char_to_wstring(aud_dat->args).c_str());
    }
}

static void recalculate_audio_delay_cut_for_afs(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const AUDIO_SETTINGS *aud_stg, const CONF_AUDIO_BASE *cnf_aud) {
    if (pe->delay_cut_additional_aframe > 0 || pe->delay_cut_additional_vframe > 0) { //ディレイカットの動画追加モード
        if (   conf->vid.afs                      //自動フィールドシフト
            && 0 == cnf_aud->audio_encode_timing //音声エンコ順が「後」
            && fps_after_afs_is_24fps(oip->n, pe)) { //推定fpsが24fpsで修正が必要
            //追加した動画フレーム数を指定して再計算
            const int audio_delay = aud_stg->mode[cnf_aud->enc_mode].delay;
            double fps_after_afs = oip->rate / (double)oip->scale * 0.8;
            pe->delay_cut_additional_aframe = additional_silence_for_aud_delay_cut(fps_after_afs, oip->audio_rate, audio_delay, pe->delay_cut_additional_vframe);
        }
    }
}

static AUO_RESULT silent_wav_output(aud_data_t *aud_dat, const PRM_ENC *pe, int samples, int wav_8bit, int audio_ch) {
    if (NULL == aud_dat)
        return AUO_RESULT_ERROR;

    if (0 >= samples)
        return AUO_RESULT_SUCCESS;

    int silent_bytes = samples * (2 - !!wav_8bit) * audio_ch;
    BYTE *buffer = (BYTE *)calloc(silent_bytes, 1);
    if (NULL == buffer)
        return AUO_RESULT_ERROR;

    if (wav_8bit)
        for (int i = 0; i < silent_bytes; i++)
            buffer[i] = 128;

    write_file(aud_dat, pe, buffer, silent_bytes);
    free(buffer);
    return AUO_RESULT_SUCCESS;
}

static AUO_RESULT wav_file_open(aud_data_t *aud_dat, const OUTPUT_INFO *oip, const PRM_ENC *pe, BOOL use_pipe, BOOL wav_8bit, int bufsize,
                                const wchar_t *auddispname, const char *auddir, DWORD encoder_priority, DWORD disable_log) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    if (use_pipe) {
        //パイプ準備
        aud_dat->pipes.stdIn.mode = AUO_PIPE_ENABLE;
        aud_dat->pipes.stdIn.bufferSize = bufsize * 2;
        if (!(disable_log & DISABLE_LOG_PIPE_INPUT)) {
            aud_dat->pipes.stdOut.mode = AUO_PIPE_ENABLE;
            aud_dat->pipes.stdErr.mode = AUO_PIPE_MUXED;
        }
        //エンコーダ準備
        int rp_ret;
        if (RP_SUCCESS != (rp_ret = RunProcess(aud_dat->args, auddir, &aud_dat->pi_aud, &aud_dat->pipes, encoder_priority, FALSE, TRUE))) {
            ret |= AUO_RESULT_ERROR; error_run_process(auddispname, rp_ret);
        } else {
            aud_dat->fp_out = aud_dat->pipes.f_stdin;
            while (WaitForInputIdle(aud_dat->pi_aud.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
                log_process_events();
        }
    } else if (fopen_s(&aud_dat->fp_out, aud_dat->wavfile, "wbS")) {
        ret |= AUO_RESULT_ERROR; error_open_wavfile();
    }
    //wavヘッダ出力
    if (!ret)
        write_wav_header(aud_dat, oip, pe, wav_8bit);
    return ret;
}

static AUO_RESULT wav_file_close(aud_data_t *aud_dat, const OUTPUT_INFO *oip, int samples_read, int wav_sample_size, BOOL use_pipe) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    if (aud_dat->is_internal) return ret;

    //終了処理
    if (!use_pipe && oip->audio_n != samples_read)
        correct_header(aud_dat->fp_out, samples_read * wav_sample_size);

    //ファイルを閉じる
    (use_pipe) ? CloseStdIn(&aud_dat->pipes) : fclose(aud_dat->fp_out);

    //wavファイル出力が成功したか確認
    if (!use_pipe && !FileExistsAndHasSize(aud_dat->wavfile)) {
        ret |= AUO_RESULT_ERROR; error_no_wavefile();
    }
    return ret;
}

static AUO_RESULT wav_output(aud_data_t *aud_dat, const OUTPUT_INFO *oip, PRM_ENC *pe, int wav_8bit, int bufsize,
                        const wchar_t *auddispname, const char *auddir, DWORD encoder_priority, DWORD disable_log)
{
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    BYTE *buf8bit = NULL;
    const func_audio_16to8 audio_16to8 = get_audio_16to8_func(wav_8bit == 2);
    const BOOL use_pipe = (strcmp(aud_dat->wavfile, PIPE_FN) == NULL);

    //並列時は8フレーム分
    if (pe->aud_parallel.th_aud) {
        bufsize = ceil_div_int((int)(oip->audio_rate * (double)oip->scale / (double)oip->rate), 16) * 16 * 8;
        //あとから音声エンコーダを回す必要が有る場合、wav出力を倍速で
        if (!use_pipe && str_has_char(auddir))
            bufsize *= 2;
    }
    //8bitを使用する場合のメモリ確保
    if (wav_8bit && NULL == (buf8bit = (BYTE *)_aligned_malloc(bufsize * oip->audio_ch * sizeof(BYTE) * wav_8bit, 32))) {
        ret |= AUO_RESULT_ERROR; error_malloc_8bit();
        return ret;
    }

    //パイプ or ファイルオープン
    if (aud_dat->is_internal) {
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            char pipename[MAX_PATH_LEN];
            get_audio_pipe_name(pipename, _countof(pipename), i_aud);
            aud_dat[i_aud].h_aud_namedpipe = CreateNamedPipeA(pipename, PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
            aud_dat[i_aud].he_ov_aud_namedpipe = CreateEvent(NULL, FALSE, FALSE, NULL);
        }
    }

    //確実なfcloseのために何故か一度ここで待機する必要あり
    if_valid_set_event(pe->aud_parallel.he_vid_start);
    if_valid_wait_for_single_object(pe->aud_parallel.he_aud_start, INFINITE);

    //パイプ or ファイルオープン
    if (aud_dat->is_internal) {
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            OVERLAPPED overlapped;
            memset(&overlapped, 0, sizeof(overlapped));
            overlapped.hEvent = aud_dat[i_aud].he_ov_aud_namedpipe;
            ConnectNamedPipe(aud_dat[i_aud].h_aud_namedpipe, &overlapped);
            while (WaitForSingleObject(overlapped.hEvent, 50) != WAIT_OBJECT_0) {
                if (pe->aud_parallel.abort) {
                    ret |= AUO_RESULT_ABORT;
                    break;
                }
            }
            write_wav_header(&aud_dat[i_aud], oip, pe, wav_8bit);
        }
    } else {
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++)
            ret |= wav_file_open(&aud_dat[i_aud], oip, pe, use_pipe, wav_8bit, bufsize, auddispname, auddir, encoder_priority, disable_log);
    }

    if (!ret) {
        //メッセージ
        show_progressbar(use_pipe, auddispname, PROGRESSBAR_CONTINUOUS);

        //wav出力
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
            silent_wav_output(&aud_dat[i_aud], pe, pe->delay_cut_additional_aframe, wav_8bit, oip->audio_ch);

        const int wav_sample_size = oip->audio_ch * ((wav_8bit) ? sizeof(BYTE) : sizeof(short));
        void *audio_dat = NULL;
        int samples_read = (pe->delay_cut_additional_aframe < 0) ? -1 * pe->delay_cut_additional_aframe : 0;
        int samples_get = bufsize;
        //wav出力ループ
        while (oip->audio_n - samples_read > 0 && samples_get) {
            //中断
            if ((pe->aud_parallel.he_aud_start) ? pe->aud_parallel.abort : oip->func_is_abort()) {
                ret |= AUO_RESULT_ABORT;
                break;
            }
            audio_dat = get_audio_data(oip, pe, samples_read, std::min(oip->audio_n - samples_read, bufsize), &samples_get);
            samples_read += samples_get;
            set_log_progress(samples_read / (double)oip->audio_n);

            while (0 < ReadLogExe(&aud_dat->pipes, nullptr, &aud_dat->log_line_cache));

            if (wav_8bit)
                audio_16to8(buf8bit, (short*)audio_dat, samples_get * oip->audio_ch);

            const int write_bytes = samples_get * wav_sample_size;
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
                write_file(&aud_dat[i_aud], pe, (wav_8bit) ? buf8bit + i_aud * write_bytes : audio_dat, write_bytes);
        }

        //動画との音声との同時処理が終了
        release_audio_parallel_events(pe);

        //ファイルクローズ
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
            ret |= wav_file_close(&aud_dat[i_aud], oip, samples_read, wav_sample_size, use_pipe);
    } else {
        //これをやっておかないとプラグインがフリーズしてしまう
        //動画との音声との同時処理が終了
        release_audio_parallel_events(pe);
    }

    if (buf8bit) _aligned_free(buf8bit);
    if (aud_dat->he_ov_aud_namedpipe) {
        CloseHandle(aud_dat->he_ov_aud_namedpipe);
    }
    if (aud_dat->h_aud_namedpipe) {
        DisconnectNamedPipe(aud_dat->h_aud_namedpipe);
        CloseHandle(aud_dat->h_aud_namedpipe);
    }

    return ret;
}

static AUO_RESULT init_aud_dat(aud_data_t *aud_dat, PRM_ENC *pe, BOOL use_pipe, const CONF_GUIEX *conf,
                         const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, const AUDIO_SETTINGS *aud_stg) {
    //ログキャッシュの初期化
    if (init_log_cache(&aud_dat->log_line_cache)) {
        error_log_line_cache();
        return AUO_RESULT_ERROR;
    }

    aud_dat->is_internal = aud_stg->is_internal;
    const CONF_AUDIO_BASE *aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;

    //wavfile名作成
    make_wavfilename(aud_dat, use_pipe, pe->temp_filename, pe->append.wav);

    //pe一時パラメータにコピーしておく
    strcpy_s(pe->append.aud[aud_dat->id], _countof(pe->append.aud[0]), (aud->delay_cut == AUDIO_DELAY_CUT_EDTS) ? aud_stg->raw_appendix : aud_stg->aud_appendix);
    if (aud_dat->id)
        insert_before_ext(pe->append.aud[aud_dat->id], _countof(pe->append.aud[0]), aud_dat->id);

    //audfile名作成
    get_aud_filename(aud_dat->audfile, _countof(aud_dat->audfile), pe, aud_dat->id);

    //コマンドライン作成
    build_audcmd(aud_dat, conf, aud_stg, pe, sys_dat, oip);
    sprintf_s(aud_dat->args, _countof(aud_dat->args), "\"%s\" %s", aud_stg->fullpath, aud_dat->cmd);

    return AUO_RESULT_SUCCESS;
}

static AUO_RESULT audio_run_enc_wavfile(aud_data_t *aud_dat, const AUDIO_SETTINGS *aud_stg, const CONF_GUIEX *conf, const char *auddir, DWORD encoder_priority) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    //パイプの設定
    if (!(aud_stg->disable_log & DISABLE_LOG_NORMAL)) {
        aud_dat->pipes.stdOut.mode = AUO_PIPE_ENABLE;
        aud_dat->pipes.stdErr.mode = AUO_PIPE_MUXED;
    }
    show_progressbar(TRUE, aud_stg->dispname, PROGRESSBAR_MARQUEE);
    int rp_ret;
    if (RP_SUCCESS != (rp_ret = RunProcess(aud_dat->args, auddir, &aud_dat->pi_aud, &aud_dat->pipes, encoder_priority, TRUE, conf->aud.ext.minimized))) {
        ret |= AUO_RESULT_ERROR; error_run_process(aud_stg->dispname, rp_ret);
    }
    return ret;
}

static AUO_RESULT audio_finish_enc(AUO_RESULT ret, aud_data_t *aud_dat, const AUDIO_SETTINGS *aud_stg) {
    if (!ret && str_has_char(aud_stg->filename)) {
        while (WaitForSingleObject(aud_dat->pi_aud.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (0 == ReadLogExe(&aud_dat->pipes, aud_stg->dispname, &aud_dat->log_line_cache))
                log_process_events();
        }
        //最後のメッセージを回収
        while (ReadLogExe(&aud_dat->pipes, aud_stg->dispname, &aud_dat->log_line_cache) > 0);

        UINT64 audfilesize = 0;
        if (!PathFileExists(aud_dat->audfile) ||
            (GetFileSizeUInt64(aud_dat->audfile, &audfilesize) && audfilesize == 0)) {
                //エラーが発生した場合
                ret |= AUO_RESULT_ERROR; error_audenc_failed(aud_stg->dispname, aud_dat->args);
                write_cached_lines(LOG_ERROR, aud_stg->dispname, &aud_dat->log_line_cache);
        } else {
            if (FileExistsAndHasSize(aud_dat->audfile))
                remove(aud_dat->wavfile); //ゴミ掃除
            write_cached_lines(LOG_MORE, aud_stg->dispname, &aud_dat->log_line_cache);
        }
    }
    write_log_auo_line_fmt(LOG_MORE, L"%s %s: %.2f%%", aud_stg->dispname, g_auo_mes.get(AUO_AUDIO_CPU_USAGE), GetProcessAvgCPUUsage(aud_dat->pi_aud.hProcess));

    CloseHandle(aud_dat->pi_aud.hProcess);
    CloseHandle(aud_dat->pi_aud.hThread);
    release_log_cache(&aud_dat->log_line_cache);
    return ret;
}

AUO_RESULT audio_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    //音声エンコードの必要がなければ終了
    if (!(oip->flag & OUTPUT_INFO_FLAG_AUDIO))
        return ret;

    //使用するエンコーダの設定を選択
    const CONF_AUDIO_BASE *cnf_aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;
    const AUDIO_SETTINGS *aud_stg = (conf->aud.use_internal) ? &sys_dat->exstg->s_aud_int[cnf_aud->encoder] : &sys_dat->exstg->s_aud_ext[cnf_aud->encoder];
    pe->aud_count = (aud_stg->mode[cnf_aud->enc_mode].use_8bit == 2) ? 2 : 1;

    //もし必要なら、オーディオディレイカット用の追加sample数を再計算する
    recalculate_audio_delay_cut_for_afs(conf, oip, pe, aud_stg, cnf_aud);

    //可能ならfaw2aacを使用
    if (sys_dat->exstg->is_faw(aud_stg)) {
        if ((ret = audio_faw2aac(conf, oip, pe, sys_dat)) == AUO_RESULT_SUCCESS) {
            return run_bat_file(conf, oip, pe, sys_dat, RUN_BAT_AFTER_AUDIO);
        } else if (ret == AUO_RESULT_ERROR) {
            return ret;
        }
    }

    aud_data_t aud_dat[2] = { { 0, 0 }, { 1, 0 } };
    char auddir[MAX_PATH_LEN]  = { 0 };
    const BOOL use_pipe = (!cnf_aud->use_wav && !cnf_aud->use_2pass) ? TRUE : FALSE;
    DWORD encoder_priority = GetExePriority(cnf_aud->priority, pe->h_p_aviutl);

    //実行ファイルチェック(filenameが空文字列なら実行しない)
    if (str_has_char(aud_stg->filename) && !PathFileExists(aud_stg->fullpath)) {
        error_no_exe_file(aud_stg->dispname, aud_stg->fullpath);
        return AUO_RESULT_ERROR;
    }

    //wav、音声ファイル名、音声エンココマンド等作成
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
        init_aud_dat(&aud_dat[i_aud], pe, use_pipe, conf, oip, sys_dat, aud_stg);

    //情報表示
    show_audio_enc_info(aud_stg, cnf_aud, pe, aud_dat);

    //auddir作成
    PathGetDirectory(auddir, _countof(auddir), aud_stg->fullpath);

    //wav出力
    ret |= wav_output(aud_dat, oip, pe, aud_stg->mode[cnf_aud->enc_mode].use_8bit, sys_dat->exstg->s_local.audio_buffer_size, aud_stg->dispname, auddir, encoder_priority, aud_stg->disable_log);

    //音声エンコード前バッチ処理
    if (!ret) ret |= run_bat_file(conf, oip, pe, sys_dat, RUN_BAT_BEFORE_AUDIO);

    //音声エンコード(filenameが空文字列なら実行しない)
    if (!aud_stg->is_internal && !use_pipe && str_has_char(aud_stg->filename))
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++)
            ret |= audio_run_enc_wavfile(&aud_dat[i_aud], aud_stg, conf, auddir, encoder_priority);

    //終了待機、メッセージ取得(filenameが空文字列なら実行しない)
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
        ret |= audio_finish_enc(ret, &aud_dat[i_aud], aud_stg);

    set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);

    //音声エンコード後バッチ処理
    if (!ret) ret |= run_bat_file(conf, oip, pe, sys_dat, RUN_BAT_AFTER_AUDIO);

    return ret;
}
