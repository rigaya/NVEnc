//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <Math.h>
#include <stdio.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_pipe.h"

#include "auo_frm.h"
#include "auo_encode.h"
#include "auo_error.h"
#include "auo_audio.h"
#include "auo_faw2aac.h"

static BOOL check_muxer_exist(MUXER_SETTINGS *muxer_stg) {
    if (PathFileExists(muxer_stg->fullpath)) 
        return TRUE;
    error_no_exe_file(muxer_stg->filename, muxer_stg->fullpath);
    return FALSE;
}

BOOL check_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const guiEx_settings *exstg) {
    BOOL check = TRUE;
    //ファイル名長さ
    if (strlen(oip->savefile) > (MAX_PATH_LEN - MAX_APPENDIX_LEN - 1)) {
        error_filename_too_long();
        check = FALSE;
    }

    //出力するもの
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED && !(oip->flag & OUTPUT_INFO_FLAG_AUDIO)) {
        error_nothing_to_output();
        check = FALSE;
    }

    //解像度
    int w_mul = 2, h_mul = 2;
    if (is_interlaced(conf->nvenc.pic_struct))
        h_mul *= 2;
    if (oip->w % w_mul) {
        error_invalid_resolution(TRUE,  w_mul, oip->w, oip->h);
        check = FALSE;
    }
    if (oip->h % h_mul) {
        error_invalid_resolution(FALSE, h_mul, oip->w, oip->h);
        check = FALSE;
    }

    if (conf->oth.out_audio_only)
        write_log_auo_line(LOG_INFO, "音声のみ出力を行います。");

    //音声エンコーダ
    if (oip->flag & OUTPUT_INFO_FLAG_AUDIO) {
        AUDIO_SETTINGS *aud_stg = &exstg->s_aud[conf->aud.encoder];
        if (str_has_char(aud_stg->filename) && !PathFileExists(aud_stg->fullpath)) {
            //fawの場合はfaw2aacがあればOKだが、それもなければエラー
            if (!(conf->aud.encoder == exstg->s_aud_faw_index && check_if_faw2aac_exists())) {
                error_no_exe_file(aud_stg->filename, aud_stg->fullpath);
                check = FALSE;
            }
        }
    }

    //muxer
    switch (pe->muxer_to_be_used) {
        case MUXER_TC2MP4:
            check &= check_muxer_exist(&exstg->s_mux[MUXER_MP4]); //tc2mp4使用時は追加でmp4boxも必要
            //下へフォールスルー
        case MUXER_MP4:
        case MUXER_MKV:
            check &= check_muxer_exist(&exstg->s_mux[pe->muxer_to_be_used]);
            break;
        default:
            break;
    }

    //オーディオディレイカット
    if (conf->vid.afs && AUDIO_DELAY_CUT_ADD_VIDEO == conf->aud.delay_cut) {
        info_afs_audio_delay_confliction();
        conf->aud.audio_encode_timing = 0;
    }

    return check;
}

void open_log_window(const char *savefile, const SYSTEM_DATA *sys_dat, int current_pass, int total_pass) {
    char mes[MAX_PATH_LEN + 512];
    char *newLine = (get_current_log_len(current_pass)) ? "\r\n\r\n" : ""; //必要なら行送り
    static const char *SEPARATOR = "------------------------------------------------------------------------------------------------------------------------------";
    if (total_pass < 2)
        sprintf_s(mes, sizeof(mes), "%s%s\r\n[%s]\r\n%s", newLine, SEPARATOR, savefile, SEPARATOR);
    else
        sprintf_s(mes, sizeof(mes), "%s%s\r\n[%s] (%d / %d pass)\r\n%s", newLine, SEPARATOR, savefile, current_pass, total_pass, SEPARATOR);
    
    show_log_window(sys_dat->aviutl_dir, sys_dat->exstg->s_local.disable_visual_styles);
    write_log_line(LOG_INFO, mes);
}

static void set_tmpdir(PRM_ENC *pe, int tmp_dir_index, const char *savefile, const SYSTEM_DATA *sys_dat) {
    if (tmp_dir_index < TMP_DIR_OUTPUT || TMP_DIR_CUSTOM < tmp_dir_index)
        tmp_dir_index = TMP_DIR_OUTPUT;

    if (tmp_dir_index == TMP_DIR_SYSTEM) {
        //システムの一時フォルダを取得
        if (GetTempPath(_countof(pe->temp_filename), pe->temp_filename) != NULL) {
            PathRemoveBackslash(pe->temp_filename);
            write_log_auo_line_fmt(LOG_INFO, "一時フォルダ : %s", pe->temp_filename);
        } else {
            warning_failed_getting_temp_path();
            tmp_dir_index = TMP_DIR_OUTPUT;
        }
    }
    if (tmp_dir_index == TMP_DIR_CUSTOM) {
        //指定されたフォルダ
        if (DirectoryExistsOrCreate(sys_dat->exstg->s_local.custom_tmp_dir)) {
            strcpy_s(pe->temp_filename, _countof(pe->temp_filename), sys_dat->exstg->s_local.custom_tmp_dir);
            PathRemoveBackslash(pe->temp_filename);
            write_log_auo_line_fmt(LOG_INFO, "一時フォルダ : %s", pe->temp_filename);
        } else {
            warning_no_temp_root(sys_dat->exstg->s_local.custom_tmp_dir);
            tmp_dir_index = TMP_DIR_OUTPUT;
        }
    }
    if (tmp_dir_index == TMP_DIR_OUTPUT) {
        //出力フォルダと同じ("\"なし)
        strcpy_s(pe->temp_filename, _countof(pe->temp_filename), savefile);
        PathRemoveFileSpecFixed(pe->temp_filename);
    }
}

static void set_aud_delay_cut(CONF_GUIEX *conf, PRM_ENC *pe, const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat) {
    pe->delay_cut_additional_vframe = 0;
    pe->delay_cut_additional_aframe = 0;
    if (oip->flag & OUTPUT_INFO_FLAG_AUDIO) {
        int audio_delay = sys_dat->exstg->s_aud[conf->aud.encoder].mode[conf->aud.enc_mode].delay;
        if (audio_delay) {
            const double fps = oip->rate / (double)oip->scale;
            const int audio_rate = oip->audio_rate;
            switch (conf->aud.delay_cut) {
            case AUDIO_DELAY_CUT_DELETE_AUDIO:
                pe->delay_cut_additional_aframe = -1 * audio_delay;
                break;
            case AUDIO_DELAY_CUT_ADD_VIDEO:
                pe->delay_cut_additional_vframe = additional_vframe_for_aud_delay_cut(fps, audio_rate, audio_delay);
                pe->delay_cut_additional_aframe = additional_silence_for_aud_delay_cut(fps, audio_rate, audio_delay);
                break;
            case AUDIO_DELAY_CUT_NONE:
            default:
                break;
            }
        } else {
            conf->aud.delay_cut = AUDIO_DELAY_CUT_NONE;
        }
    }
}

void set_enc_prm(CONF_GUIEX *conf, PRM_ENC *pe, const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat) {
    //初期化
    ZeroMemory(pe, sizeof(PRM_ENC));
    //設定更新
    sys_dat->exstg->load_encode_stg();
    sys_dat->exstg->load_append();
    sys_dat->exstg->load_fn_replace();

    pe->video_out_type = check_video_ouput(conf, oip);
    memcpy(&pe->append, &sys_dat->exstg->s_append, sizeof(FILE_APPENDIX));

    char filename_replace[MAX_PATH_LEN];

    //一時フォルダの決定
    set_tmpdir(pe, conf->oth.temp_dir, oip->savefile, sys_dat);

    //音声一時フォルダの決定
    char *cus_aud_tdir = pe->temp_filename;
    if (conf->aud.aud_temp_dir) {
        if (DirectoryExistsOrCreate(sys_dat->exstg->s_local.custom_audio_tmp_dir)) {
            cus_aud_tdir = sys_dat->exstg->s_local.custom_audio_tmp_dir;
            write_log_auo_line_fmt(LOG_INFO, "音声一時フォルダ : %s", cus_aud_tdir);
        } else {
            warning_no_aud_temp_root(sys_dat->exstg->s_local.custom_audio_tmp_dir);
        }
    }
    strcpy_s(pe->aud_temp_dir, _countof(pe->aud_temp_dir), cus_aud_tdir);

    //ファイル名置換を行い、一時ファイル名を作成
    strcpy_s(filename_replace, _countof(filename_replace), PathFindFileName(oip->savefile));
    sys_dat->exstg->apply_fn_replace(filename_replace, _countof(filename_replace));
    PathCombineLong(pe->temp_filename, _countof(pe->temp_filename), pe->temp_filename, filename_replace);

    //ESしか出せないので拡張子を変更
    change_ext(pe->temp_filename, _countof(pe->temp_filename), (conf->nvenc.codec == NV_ENC_H264) ? ".264" : ".265");

    pe->muxer_to_be_used = check_muxer_to_be_used(conf, sys_dat, pe->temp_filename, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
    
    //FAWチェックとオーディオディレイの修正
    if (conf->aud.faw_check)
        auo_faw_check(&conf->aud, oip, pe, sys_dat->exstg);
    set_aud_delay_cut(conf, pe, oip, sys_dat);
}

void auto_save_log(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    guiEx_settings ex_stg(true);
    ex_stg.load_log_win();
    if (!ex_stg.s_log.auto_save_log)
        return;
    char log_file_path[MAX_PATH_LEN];
    if (AUO_RESULT_SUCCESS != getLogFilePath(log_file_path, _countof(log_file_path), pe, sys_dat, conf, oip))
        warning_no_auto_save_log_dir();
    auto_save_log_file(log_file_path);
    return;
}

int additional_vframe_for_aud_delay_cut(double fps, int audio_rate, int audio_delay) {
    double delay_sec = audio_delay / (double)audio_rate;
    return (int)ceil(delay_sec * fps);
}

int additional_silence_for_aud_delay_cut(double fps, int audio_rate, int audio_delay, int vframe_added) {
    vframe_added = (vframe_added >= 0) ? vframe_added : additional_vframe_for_aud_delay_cut(fps, audio_rate, audio_delay);
    return (int)(vframe_added / (double)fps * audio_rate + 0.5) - audio_delay;
}

BOOL fps_after_afs_is_24fps(const int frame_n, const PRM_ENC *pe) {
    return (pe->drop_count > (frame_n * 0.10));
}

void get_aud_filename(char *audfile, size_t nSize, const PRM_ENC *pe, int i_aud) {
    PathCombineLong(audfile, nSize, pe->aud_temp_dir, PathFindFileName(pe->temp_filename));
    apply_appendix(audfile, nSize, audfile, pe->append.aud[i_aud]);
}

static void get_muxout_appendix(char *muxout_appendix, size_t nSize, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe) {
    static const char * const MUXOUT_APPENDIX = "_muxout";
    strcpy_s(muxout_appendix, nSize, MUXOUT_APPENDIX);
    const char *ext = (pe->muxer_to_be_used >= 0 && str_has_char(sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext)) ?
        sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext : PathFindExtension(pe->temp_filename);
    strcat_s(muxout_appendix, nSize, ext);
}

void get_muxout_filename(char *filename, size_t nSize, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe) {
    char muxout_appendix[MAX_APPENDIX_LEN];
    get_muxout_appendix(muxout_appendix, sizeof(muxout_appendix), sys_dat, pe);
    apply_appendix(filename, nSize, pe->temp_filename, muxout_appendix);
}

//チャプターファイル名とapple形式のチャプターファイル名を同時に作成する
void set_chap_filename(char *chap_file, size_t cf_nSize, char *chap_apple, size_t ca_nSize, const char *chap_base, 
                       const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    strcpy_s(chap_file, cf_nSize, chap_base);
    cmd_replace(chap_file, cf_nSize, pe, sys_dat, conf, oip);
    apply_appendix(chap_apple, ca_nSize, chap_file, pe->append.chap_apple);
    sys_dat->exstg->apply_fn_replace(PathFindFileName(chap_apple), ca_nSize - (PathFindFileName(chap_apple) - chap_apple));
}

void insert_num_to_replace_key(char *key, size_t nSize, int num) {
    char tmp[128];
    int key_len = strlen(key);
    sprintf_s(tmp, _countof(tmp), "%d%s", num, &key[key_len-1]);
    key[key_len-1] = '\0';
    strcat_s(key, nSize, tmp);
}

static void set_guiEx_auto_sar(int *sar_x, int *sar_y, int width, int height) {
    if (width > 0 && height > 0 && *sar_x < 0 && *sar_y < 0) {
        int x = -1 * *sar_x * height;
        int y = -1 * *sar_y * width;
        if (abs(y - x) > -16 * *sar_y) {
            int gcd = get_gcd(x, y);
            *sar_x = x / gcd;
            *sar_y = y / gcd;
        } else {
            *sar_x = *sar_y = 1;
        }
    } else if (*sar_x * *sar_y < 0) {
        *sar_x = *sar_y = 0;
    }
}
/*
static void replace_aspect_ratio(char *cmd, size_t nSize, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    const int w = oip->w;
    const int h = oip->h;
 
    int sar_x = conf->qsv.nPAR[0];
    int sar_y = conf->qsv.nPAR[1];
    int dar_x = 0;
    int dar_y = 0;
    if (sar_x * sar_y > 0) {
        if (sar_x < 0) {
            dar_x = -1 * sar_x;
            dar_y = -1 * sar_y;
            set_guiEx_auto_sar(&sar_x, &sar_y, w, h);
        } else {
            dar_x = sar_x * w;
            dar_y = sar_y * h;
            const int gcd = get_gcd(dar_x, dar_y);
            dar_x /= gcd;
            dar_y /= gcd;
        }
    }
    if (sar_x * sar_y <= 0)
        sar_x = sar_y = 1;
    if (dar_x * dar_y <= 0)
        dar_x = dar_y = 1;

    char buf[32];
    //%{sar_x} / %{par_x}
    sprintf_s(buf, _countof(buf), "%d", sar_x);
    replace(cmd, nSize, "%{sar_x}", buf);
    replace(cmd, nSize, "%{par_x}", buf);
    //%{sar_x} / %{sar_y}
    sprintf_s(buf, _countof(buf), "%d", sar_y);
    replace(cmd, nSize, "%{sar_y}", buf);
    replace(cmd, nSize, "%{par_y}", buf);
    //%{dar_x}
    sprintf_s(buf, _countof(buf), "%d", dar_x);
    replace(cmd, nSize, "%{dar_x}", buf);
    //%{dar_y}
    sprintf_s(buf, _countof(buf), "%d", dar_y);
    replace(cmd, nSize, "%{dar_y}", buf);
}
*/
void cmd_replace(char *cmd, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    char tmp[MAX_PATH_LEN] = { 0 };
    //置換操作の実行
    //%{vidpath}
    replace(cmd, nSize, "%{vidpath}", pe->temp_filename);
    //%{audpath}
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
        if (str_has_char(pe->append.aud[i_aud])) {
            get_aud_filename(tmp, _countof(tmp), pe, i_aud);
            char aud_key[128] = "%{audpath}";
            if (i_aud)
                insert_num_to_replace_key(aud_key, _countof(aud_key), i_aud);
            replace(cmd, nSize, aud_key, tmp);
        }
    }
    //%{tmpdir}
    strcpy_s(tmp, _countof(tmp), pe->temp_filename);
    PathRemoveFileSpecFixed(tmp);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{tmpdir}", tmp);
    //%{tmpfile}
    strcpy_s(tmp, _countof(tmp), pe->temp_filename);
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{tmpfile}", tmp);
    //%{tmpname}
    strcpy_s(tmp, _countof(tmp), PathFindFileName(pe->temp_filename));
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{tmpname}", tmp);
    //%{savpath}
    replace(cmd, nSize, "%{savpath}", oip->savefile);
    //%{savfile}
    strcpy_s(tmp, _countof(tmp), oip->savefile);
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{savfile}", tmp);
    //%{savname}
    strcpy_s(tmp, _countof(tmp), PathFindFileName(oip->savefile));
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{savname}", tmp);
    //%{savdir}
    strcpy_s(tmp, _countof(tmp), oip->savefile);
    PathRemoveFileSpecFixed(tmp);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{savdir}", tmp);
    //%{aviutldir}
    strcpy_s(tmp, _countof(tmp), sys_dat->aviutl_dir);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{aviutldir}", tmp);
    //%{chpath}
    apply_appendix(tmp, _countof(tmp), oip->savefile, pe->append.chap);
    replace(cmd, nSize, "%{chpath}", tmp);
    //%{tcpath}
    apply_appendix(tmp, _countof(tmp), pe->temp_filename, pe->append.tc);
    replace(cmd, nSize, "%{tcpath}", tmp);
    //%{muxout}
    get_muxout_filename(tmp, _countof(tmp), sys_dat, pe);
    replace(cmd, nSize, "%{muxout}", tmp);
    //%{fps_rate}
    int fps_rate = oip->rate;
    int fps_scale = oip->scale;
#ifdef MSDK_SAMPLE_VERSION
    if (conf->qsv.vpp.nDeinterlace == MFX_DEINTERLACE_IT)
        fps_rate = (fps_rate * 4) / 5;
#endif
    const int fps_gcd = get_gcd(fps_rate, fps_scale);
    fps_rate /= fps_gcd;
    fps_scale /= fps_gcd;
    sprintf_s(tmp, sizeof(tmp), "%d", fps_rate);
    replace(cmd, nSize, "%{fps_rate}", tmp);
    //%{fps_rate_times_4}
    fps_rate *= 4;
    sprintf_s(tmp, sizeof(tmp), "%d", fps_rate);
    replace(cmd, nSize, "%{fps_rate_times_4}", tmp);
    //%{fps_scale}
    sprintf_s(tmp, sizeof(tmp), "%d", fps_scale);
    replace(cmd, nSize, "%{fps_scale}", tmp);
    //アスペクト比
    //replace_aspect_ratio(cmd, nSize, conf, oip);

    char fullpath[MAX_PATH_LEN];
    replace(cmd, nSize, "%{audencpath}",   GetFullPath(sys_dat->exstg->s_aud[conf->aud.encoder].fullpath, fullpath, _countof(fullpath)));
    replace(cmd, nSize, "%{mp4muxerpath}", GetFullPath(sys_dat->exstg->s_mux[MUXER_MP4].fullpath,         fullpath, _countof(fullpath)));
    replace(cmd, nSize, "%{mkvmuxerpath}", GetFullPath(sys_dat->exstg->s_mux[MUXER_MKV].fullpath,         fullpath, _countof(fullpath)));
}

//一時ファイルの移動・削除を行う 
// move_from -> move_to
// temp_filename … 動画ファイルの一時ファイル名。これにappendixをつけてmove_from を作る。
//                  appndixがNULLのときはこれをそのままmove_fromとみなす。
// appendix      … ファイルの後修飾子。NULLも可。
// savefile      … 保存動画ファイル名。これにappendixをつけてmove_to を作る。NULLだと move_to に移動できない。
// ret, erase    … これまでのエラーと一時ファイルを削除するかどうか。エラーがない場合にのみ削除できる
// name          … 一時ファイルの種類の名前
// must_exist    … trueのとき、移動するべきファイルが存在しないとエラーを返し、ファイルが存在しないことを伝える
static BOOL move_temp_file(const char *appendix, const char *temp_filename, const char *savefile, DWORD ret, BOOL erase, const char *name, BOOL must_exist) {
    char move_from[MAX_PATH_LEN] = { 0 };
    if (appendix)
        apply_appendix(move_from, _countof(move_from), temp_filename, appendix);
    else
        strcpy_s(move_from, _countof(move_from), temp_filename);

    if (!PathFileExists(move_from)) {
        if (must_exist)
            write_log_auo_line_fmt(LOG_WARNING, "%sファイルが見つかりませんでした。", name);
        return (must_exist) ? FALSE : TRUE;
    }
    if (ret == AUO_RESULT_SUCCESS && erase) {
        remove(move_from);
        return TRUE;
    }
    if (savefile == NULL || appendix == NULL)
        return TRUE;
    char move_to[MAX_PATH_LEN] = { 0 };
    apply_appendix(move_to, _countof(move_to), savefile, appendix);
    if (_stricmp(move_from, move_to) != NULL) {
        if (PathFileExists(move_to))
            remove(move_to);
        if (rename(move_from, move_to))
            write_log_auo_line_fmt(LOG_WARNING, "%sファイルの移動に失敗しました。", name);
    }
    return TRUE;
}

AUO_RESULT move_temporary_files(const CONF_GUIEX *conf, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const OUTPUT_INFO *oip, DWORD ret) {
    //動画ファイル
    if (!conf->oth.out_audio_only)
        if (!move_temp_file(PathFindExtension((pe->muxer_to_be_used >= 0) ? oip->savefile : pe->temp_filename), pe->temp_filename, oip->savefile, ret, FALSE, "出力", !ret))
            ret |= AUO_RESULT_ERROR;
    //動画のみファイル
    if (str_has_char(pe->muxed_vid_filename) && PathFileExists(pe->muxed_vid_filename))
        remove(pe->muxed_vid_filename);
    //mux後ファイル
    if (pe->muxer_to_be_used >= 0) {
        char muxout_appendix[MAX_APPENDIX_LEN];
        get_muxout_appendix(muxout_appendix, _countof(muxout_appendix), sys_dat, pe);
        move_temp_file(muxout_appendix, pe->temp_filename, oip->savefile, ret, FALSE, "mux後ファイル", FALSE);
    }
    //qpファイル
    move_temp_file(pe->append.qp,   pe->temp_filename, oip->savefile, ret, TRUE, "qp", FALSE);
    //tcファイル
    BOOL erase_tc = conf->vid.afs && !conf->vid.auo_tcfile_out && pe->muxer_to_be_used != MUXER_DISABLED;
    move_temp_file(pe->append.tc,   pe->temp_filename, oip->savefile, ret, erase_tc, "タイムコード", FALSE);
    //チャプターファイル
    if (pe->muxer_to_be_used >= 0 && sys_dat->exstg->s_local.auto_del_chap) {
        char chap_file[MAX_PATH_LEN];
        char chap_apple[MAX_PATH_LEN];
        const MUXER_CMD_EX *muxer_mode = &sys_dat->exstg->s_mux[pe->muxer_to_be_used].ex_cmd[(pe->muxer_to_be_used == MUXER_MKV) ? conf->mux.mkv_mode : conf->mux.mp4_mode];
        set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), muxer_mode->chap_file, pe, sys_dat, conf, oip);
        move_temp_file(NULL, chap_file,  NULL, ret, TRUE, "チャプター",        FALSE);
        move_temp_file(NULL, chap_apple, NULL, ret, TRUE, "チャプター(Apple)", FALSE);
    }
    //音声ファイル(wav)
    if (strcmp(pe->append.aud[0], pe->append.wav)) //「wav出力」ならここでは処理せず下のエンコード後ファイルとして扱う
        move_temp_file(pe->append.wav,  pe->temp_filename, oip->savefile, ret, TRUE, "wav", FALSE);
    //音声ファイル(エンコード後ファイル)
    char aud_tempfile[MAX_PATH_LEN];
    PathCombineLong(aud_tempfile, _countof(aud_tempfile), pe->aud_temp_dir, PathFindFileName(pe->temp_filename));
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
        if (!move_temp_file(pe->append.aud[i_aud], aud_tempfile, oip->savefile, ret, !conf->oth.out_audio_only && pe->muxer_to_be_used != MUXER_DISABLED, "音声", conf->oth.out_audio_only))
            ret |= AUO_RESULT_ERROR;
    return ret;
}

DWORD GetExePriority(DWORD set, HANDLE h_aviutl) {
    if (set == AVIUTLSYNC_PRIORITY_CLASS)
        return (h_aviutl) ? GetPriorityClass(h_aviutl) : NORMAL_PRIORITY_CLASS;
    else
        return priority_table[set].value;
}

int check_video_ouput(const char *filename) {
    if (check_ext(filename, ".mp4"))  return VIDEO_OUTPUT_MP4;
    if (check_ext(filename, ".mkv"))  return VIDEO_OUTPUT_MKV;
    return VIDEO_OUTPUT_RAW;
}

int check_video_ouput(const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    if ((oip->flag & OUTPUT_INFO_FLAG_VIDEO) && !conf->oth.out_audio_only) {
        return check_video_ouput(oip->savefile);
    }
    return VIDEO_OUTPUT_DISABLED;
}

BOOL check_output_has_chapter(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, int muxer_to_be_used) {
    BOOL has_chapter = FALSE;
    if (muxer_to_be_used == MUXER_MKV || muxer_to_be_used == MUXER_TC2MP4 || muxer_to_be_used == MUXER_MP4) {
        const MUXER_CMD_EX *muxer_mode = &sys_dat->exstg->s_mux[muxer_to_be_used].ex_cmd[(muxer_to_be_used == MUXER_MKV) ? conf->mux.mkv_mode : conf->mux.mp4_mode];
        has_chapter = str_has_char(muxer_mode->chap_file);
    }
    return has_chapter;
}

int check_muxer_to_be_used(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, const char *temp_filename, int video_output_type, BOOL audio_output) {
    int muxer_to_be_used = MUXER_DISABLED;
    if (video_output_type == VIDEO_OUTPUT_MP4 && !conf->mux.disable_mp4ext)
        muxer_to_be_used = (conf->vid.afs) ? MUXER_TC2MP4 : MUXER_MP4;
    else if (video_output_type == VIDEO_OUTPUT_MKV && !conf->mux.disable_mkvext)
        muxer_to_be_used = MUXER_MKV;
    else if (video_output_type == VIDEO_OUTPUT_MPEG2 && !conf->mux.disable_mpgext)
        muxer_to_be_used = MUXER_MPG;
    
    //muxerが必要ないかどうかチェック
    BOOL no_muxer = TRUE;
    no_muxer &= !audio_output;
    no_muxer &= !conf->vid.afs;
    no_muxer &= video_output_type == check_video_ouput(temp_filename);
    no_muxer &= !check_output_has_chapter(conf, sys_dat, muxer_to_be_used);
    return (no_muxer) ? MUXER_DISABLED : muxer_to_be_used;
}

AUO_RESULT getLogFilePath(char *log_file_path, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    guiEx_settings stg(TRUE); //ログウィンドウの保存先設定は最新のものを使用する
    stg.load_log_win();
    switch (stg.s_log.auto_save_log_mode) {
        case AUTO_SAVE_LOG_CUSTOM:
            char log_file_dir[MAX_PATH_LEN];
            strcpy_s(log_file_path, nSize, stg.s_log.auto_save_log_path);
            cmd_replace(log_file_path, nSize, pe, sys_dat, conf, oip);
            PathGetDirectory(log_file_dir, _countof(log_file_dir), log_file_path);
            if (DirectoryExistsOrCreate(log_file_dir))
                break;
            ret = AUO_RESULT_WARNING;
            //下へフォールスルー
        case AUTO_SAVE_LOG_OUTPUT_DIR:
        default:
            apply_appendix(log_file_path, nSize, oip->savefile, "_log.txt"); 
            break;
    }
    return ret;
}

double get_duration(const OUTPUT_INFO *oip, const PRM_ENC *pe) {
    //Aviutlから再生時間情報を取得
    return ((double)(oip->n + pe->delay_cut_additional_vframe) * (double)oip->scale) / (double)oip->rate;
}

int ReadLogExe(PIPE_SET *pipes, const char *exename, LOG_CACHE *log_line_cache) {
    DWORD pipe_read = 0;
    if (pipes->stdOut.h_read) {
        if (!PeekNamedPipe(pipes->stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
            return -1;
        if (pipe_read) {
            ReadFile(pipes->stdOut.h_read, pipes->read_buf + pipes->buf_len, sizeof(pipes->read_buf) - pipes->buf_len - 1, &pipe_read, NULL);
            pipes->buf_len += pipe_read;
            pipes->read_buf[pipes->buf_len] = '\0';
            write_log_exe_mes(pipes->read_buf, &pipes->buf_len, exename, log_line_cache);
        }
    }
    return (int)pipe_read;
}

void write_cached_lines(int log_level, const char *exename, LOG_CACHE *log_line_cache) {
    static const char *const LOG_LEVEL_STR[] = { "info", "warning", "error" };
    static const char *MESSAGE_FORMAT = "%s [%s]: %s";
    char *buffer = NULL;
    int buffer_len = 0;
    const int log_level_idx = clamp(log_level, LOG_INFO, LOG_ERROR);
    const int additional_length = strlen(exename) + strlen(LOG_LEVEL_STR[log_level_idx]) + strlen(MESSAGE_FORMAT) - strlen("%s") * 3 + 1;
    for (int i = 0; i < log_line_cache->idx; i++) {
        const int required_buffer_len = strlen(log_line_cache->lines[i]) + additional_length;
        if (buffer_len < required_buffer_len) {
            if (buffer) free(buffer);
            buffer = (char *)malloc(required_buffer_len * sizeof(buffer[0]));
            buffer_len = required_buffer_len;
        }
        if (buffer) {
            sprintf_s(buffer, buffer_len, MESSAGE_FORMAT, exename, LOG_LEVEL_STR[log_level_idx], log_line_cache->lines[i]);
            write_log_line(log_level, buffer, true);
        }
    }
    if (buffer) free(buffer);
}
