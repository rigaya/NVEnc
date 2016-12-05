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
#include <Math.h>
#include <stdlib.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "output.h"
#include "auo.h"
#include "auo_version.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_encode.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_chapter.h"
#include "auo_system.h"
#include "auo_mux.h"
#include "auo_encode.h"
#include "exe_version.h"
#include "cpu_info.h"

static void show_mux_info(const MUXER_SETTINGS *mux_stg, BOOL vidmux, BOOL audmux, BOOL tcmux, BOOL chapmux, const char *muxer_mode_name) {
    char mes[1024];
    static const char * const ON_OFF_INFO[] = { "off", " on" };

    std::string ver_str = "";
    int version[4] = { 0 };
    if (str_has_char(mux_stg->ver_cmd) && 0 == get_exe_version_from_cmd(mux_stg->fullpath, mux_stg->ver_cmd, version)) {
        ver_str = " (" + ver_string(version) + ")";
    }

    sprintf_s(mes, _countof(mes), "%s%sでmuxを行います。映像:%s, 音声:%s, tc:%s, chap:%s, 拡張モード:%s", 
        mux_stg->dispname,
        ver_str.c_str(),
        ON_OFF_INFO[vidmux != 0],
        ON_OFF_INFO[audmux != 0],
        ON_OFF_INFO[tcmux != 0],
        ON_OFF_INFO[chapmux != 0],
        muxer_mode_name);
    write_log_auo_line_fmt(LOG_INFO, mes);

    sprintf_s(mes, _countof(mes), "%s で mux中...", mux_stg->dispname);
    set_window_title(mes, PROGRESSBAR_MARQUEE);
}

//muxの空き容量などを計算し、行えるかを確認する
static AUO_RESULT check_mux_disk_space(const MUXER_SETTINGS *mux_stg, const char *mux_tmpdir, const CONF_GUIEX *conf, const PRM_ENC *pe, UINT64 expected_filesize) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    UINT64 required_space = (UINT64)(expected_filesize * 1.01); //ちょい多め
    //出力先ドライブ
    char vid_root[MAX_PATH_LEN];
    strcpy_s(vid_root, _countof(vid_root), pe->temp_filename);
    PathStripToRoot(vid_root);
    //一時フォルダ指定用のコマンドがあれば、一時フォルダの指定について検証
    if (str_has_char(mux_stg->tmp_cmd) && conf->mux.mp4_temp_dir) {
        ULARGE_INTEGER temp_drive_avail_space = { 0 };
        BOOL tmp_same_drive_as_out = FALSE;
        //指定されたドライブが存在するかどうか
        char temp_root[MAX_PATH_LEN];
        if (!PathGetRoot(mux_tmpdir, temp_root, _countof(temp_root)) ||
            !PathIsDirectory(temp_root) ||
            !DirectoryExistsOrCreate(mux_tmpdir)) {
            ret = AUO_RESULT_WARNING; warning_no_mux_tmp_root(temp_root);
        //空き容量を取得できていなければ、チェックを終了する
        } else if (expected_filesize <= 0) {
            return AUO_RESULT_SUCCESS;
        //ドライブの空き容量取得
        } else if (!GetDiskFreeSpaceEx(temp_root, &temp_drive_avail_space, NULL, NULL)) {
            ret = AUO_RESULT_WARNING; warning_failed_mux_tmp_drive_space();
        //一時フォルダと出力先が同じフォルダかどうかで、一時フォルダの必要とされる空き領域が変わる
        } else {
            tmp_same_drive_as_out = (_stricmp(vid_root, temp_root) == NULL);
            if ((UINT64)temp_drive_avail_space.QuadPart < required_space * (1 + tmp_same_drive_as_out)) {
                ret = AUO_RESULT_WARNING; warning_mux_tmp_not_enough_space();
            }
        }
        //一時フォルダと出力先が同じフォルダならさらなる検証の必要はない
        if (tmp_same_drive_as_out && ret == AUO_RESULT_SUCCESS)
            return ret;
    }
    //空き容量を取得できていなければ、チェックを終了する
    if (expected_filesize <= 0)
        return AUO_RESULT_SUCCESS;
    //mp4boxの場合、一時ファイルが指定されていないときには
    //カレントディレクトリのあるドライブ(muxerのあるドライブ)に一時ファイルが作られる
    //その一時フォルダのドライブについて検証
    if ((!conf->mux.mp4_temp_dir || ret == AUO_RESULT_WARNING) && stristr(mux_stg->filename, "mp4box")) {
        char muxer_root[MAX_PATH_LEN];
        //ドライブの空き容量取得
        ULARGE_INTEGER muxer_drive_avail_space = { 0 };
        if (!PathGetRoot(mux_stg->fullpath, muxer_root, _countof(muxer_root)) ||
            !GetDiskFreeSpaceEx(muxer_root, &muxer_drive_avail_space, NULL, NULL)) {
            error_failed_muxer_drive_space(); return AUO_RESULT_ERROR;
        }
        //一時フォルダと出力先が同じフォルダかどうかで、一時フォルダの必要とされる空き領域が変わる
        BOOL muxer_same_drive_as_out = (_stricmp(vid_root, muxer_root) == NULL);
        if ((UINT64)muxer_drive_avail_space.QuadPart < required_space * (1 + muxer_same_drive_as_out)) {
            error_muxer_drive_not_enough_space(); return AUO_RESULT_ERROR;
        }
        //一時フォルダと出力先が同じフォルダならさらなる検証の必要はない
        if (muxer_same_drive_as_out && ret == AUO_RESULT_SUCCESS)
            return ret;
    }
    //出力先のドライブの空き容量
    //ドライブの空き容量取得
    ULARGE_INTEGER out_drive_avail_space = { 0 };
    if (!GetDiskFreeSpaceEx(vid_root, &out_drive_avail_space, NULL, NULL)) {
        error_failed_out_drive_space(); return AUO_RESULT_ERROR;
    }
    if ((UINT64)out_drive_avail_space.QuadPart < required_space) {
        error_out_drive_not_enough_space(); return AUO_RESULT_ERROR;
    }
    return ret;
}

//muxする動画ファイルと音声ファイルからmux後ファイルの推定サイズを取得する
static AUO_RESULT get_expected_filesize(const PRM_ENC *pe, BOOL enable_vid_mux, DWORD enable_aud_mux, UINT64 *_expected_filesize) {
    *_expected_filesize = 0;
    //動画ファイルのサイズ
    if (enable_vid_mux) {
        UINT64 vid_size = 0;
        if (!PathFileExists(pe->temp_filename)) {
            error_no_vid_file(); return AUO_RESULT_ERROR;
        }
        if (!GetFileSizeUInt64(pe->temp_filename, &vid_size)) {
            warning_failed_get_vid_size(); return AUO_RESULT_WARNING;
        }
        if (vid_size == 0) {
            error_vid_file_zero_byte(); return AUO_RESULT_ERROR;
        }
        *_expected_filesize += vid_size;
    }
    //音声ファイルのサイズ
    if (enable_aud_mux) {
        UINT64 aud_size = 0;
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
            if (enable_aud_mux & (0x01<<i_aud)) {
                char audfile[MAX_PATH_LEN] = { 0 };
                get_aud_filename(audfile, _countof(audfile), pe, i_aud);
                if (!PathFileExists(audfile)) {
                    error_no_aud_file(); return AUO_RESULT_ERROR;
                }
                if (!GetFileSizeUInt64(audfile, &aud_size)) {
                    warning_failed_get_aud_size(); return AUO_RESULT_WARNING;
                }
                if (aud_size == 0) {
                    error_aud_file_zero_byte(); return AUO_RESULT_ERROR;
                }
                *_expected_filesize += aud_size;
            }
        }
    }
    return AUO_RESULT_SUCCESS;
}

//mux後ファイルが存在する他とファイルサイズをチェック
//大丈夫そうならTRUEを返す
static AUO_RESULT check_muxout_filesize(const char *muxout, UINT64 expected_filesize) {
    const double FILE_SIZE_THRESHOLD_MULTI = 0.95;
    UINT64 muxout_filesize = 0;
    if (!PathFileExists(muxout)) {
        error_check_muxout_exist();
        return AUO_RESULT_ERROR;
    }
    //推定ファイルサイズの取得に失敗していたら終了
    if (expected_filesize <= 0)
        return AUO_RESULT_WARNING;
    if (GetFileSizeUInt64(muxout, &muxout_filesize)) {
        //ファイルサイズの取得に成功したら、予想サイズとの比較を行う
        if (((double)muxout_filesize) <= ((double)expected_filesize * FILE_SIZE_THRESHOLD_MULTI * (1.0 - exp(-1.0 * (double)expected_filesize / (128.0 * 1024.0))))) {
            error_check_muxout_too_small((int)(expected_filesize / 1024), (int)(muxout_filesize / 1024));
            return AUO_RESULT_ERROR;
        }
        return AUO_RESULT_SUCCESS;
    }
    warning_failed_check_muxout_filesize();
    return AUO_RESULT_WARNING;
}

//不必要なチャプターコマンドを削除する
static void del_chap_cmd(char *cmd, BOOL apple_type_only) {
    if (!apple_type_only)
        del_arg(cmd, "%{chap_apple}", -1);
    del_arg(cmd, "%{chapter}", -1);
}

static int get_excmd_mode(const CONF_GUIEX *conf, const PRM_ENC *pe) {
    int mode = 0;
    switch (pe->muxer_to_be_used) {
        case MUXER_MKV:     mode = conf->mux.mkv_mode; break;
        case MUXER_MPG:     mode = conf->mux.mpg_mode; break;
        case MUXER_MP4:
        case MUXER_TC2MP4: 
        case MUXER_MP4_RAW: mode = conf->mux.mp4_mode; break;
    }
    return mode;
}

static void build_aud_mux_cmd(char *audstr, size_t nSize, const char *aud_cmd, DWORD enable_aud_mux, const PRM_ENC *pe) {
    *audstr = '\0';
    if (enable_aud_mux == 0x00)
        return;

    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
        if (enable_aud_mux & (0x01 << i_aud)) {
            char audcmd_tmp[1024];
            strcpy_s(audcmd_tmp, _countof(audcmd_tmp), aud_cmd);
            if (i_aud) {
                char audkey[128] = "%{audpath}";
                insert_num_to_replace_key(audkey, _countof(audkey), i_aud);
                replace(audcmd_tmp, _countof(audcmd_tmp), "%{audpath}", audkey);
            }
            strcat_s(audstr, nSize, audcmd_tmp);
            strcat_s(audstr, nSize, " ");
        }
    }
}

static AUO_RESULT build_mux_cmd(char *cmd, size_t nSize, const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, 
                          const SYSTEM_DATA *sys_dat, const MUXER_SETTINGS *mux_stg, UINT64 expected_filesize,
                          BOOL enable_vid_mux, DWORD enable_aud_mux, BOOL enable_chap_mux) {
    strcpy_s(cmd, nSize, mux_stg->base_cmd);
    const BOOL enable_tc_mux = ((conf->vid.afs) != 0) && str_has_char(mux_stg->tc_cmd);
    const MUXER_CMD_EX *muxer_mode = &mux_stg->ex_cmd[get_excmd_mode(conf, pe)];
    const char *vidstr = (enable_vid_mux) ? mux_stg->vid_cmd : "";
    const char *tcstr  = (enable_tc_mux) ? mux_stg->tc_cmd : "";
    const char *exstr  = (conf->mux.apple_mode && str_has_char(muxer_mode->cmd_apple)) ? muxer_mode->cmd_apple : muxer_mode->cmd;
    char audstr[MAX_CMD_LEN];
    build_aud_mux_cmd(audstr, _countof(audstr), mux_stg->aud_cmd, enable_aud_mux, pe);
    //映像用コマンド
    replace(cmd, nSize, "%{vd_cmd}",  vidstr);
    //音声用コマンド
    replace(cmd, nSize, "%{au_cmd}",  audstr);
    //タイムコード用
    replace(cmd, nSize, "%{tc_cmd}",  tcstr);
    //一時ファイル(空き容量等)のチェックを行う
    AUO_RESULT mux_check = check_mux_disk_space(mux_stg, sys_dat->exstg->s_local.custom_mp4box_tmp_dir, conf, pe, expected_filesize);
    switch (mux_check) {
        case AUO_RESULT_SUCCESS:
            if (conf->mux.mp4_temp_dir) {
                //一時フォルダ指定を行う
                replace(cmd, nSize, "%{tmp_cmd}", mux_stg->tmp_cmd);
                char m_tmp_dir[MAX_PATH_LEN];
                strcpy_s(m_tmp_dir, _countof(m_tmp_dir), sys_dat->exstg->s_local.custom_mp4box_tmp_dir);
                PathForceRemoveBackSlash(m_tmp_dir);
                replace(cmd, nSize, "%{m_tmpdir}", m_tmp_dir);
                break;
            }
            //下へフォールスルー(一時フォルダ指定を行わない)
        case AUO_RESULT_WARNING: //一時フォルダ指定を行えない
            replace(cmd, nSize, "%{tmp_cmd}", "");
            break;
        case AUO_RESULT_ERROR://一時ファイル関連のチェックでエラー
        default:
            return AUO_RESULT_ERROR;
    }
    //拡張オプションとチャプター処理
    //とりあえず必要なくてもチャプターファイル名を作る
    char chap_file[MAX_PATH_LEN];
    char chap_apple[MAX_PATH_LEN];
    set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), 
        muxer_mode->chap_file, pe, sys_dat, conf, oip);
    replace(cmd, nSize, "%{ex_cmd}", exstr);
    if (!enable_chap_mux) {
        del_chap_cmd(cmd, FALSE); //チャプター用コマンドとパラメータを削除
    } else if (strstr(cmd, "%{chapter}") || strstr(cmd, "%{chap_apple}")) {
        //もし、チャプターファイル名への置換があるなら、チャプターファイルの存在をチェックする
        if (!PathFileExists(chap_file)) {
            //チャプターファイルが存在しない
            warning_mux_no_chapter_file();
            del_chap_cmd(cmd, FALSE);
            enable_chap_mux = FALSE;
        } else {
            replace(cmd, nSize, "%{chapter}", chap_file);
            chapter_file chapter;
            if (AUO_CHAP_ERR_NONE != (chapter.read_file(chap_file, CODE_PAGE_UNSET, get_duration(oip, pe)))) {
                warning_mux_chapter(chapter.get_result());
            } else {
                chapter.add_dummy_chap_zero_pos();
                //オーディオディレイのカットを映像追加で行ったら、チャプター位置の修正も必要
                if (0 < pe->delay_cut_additional_vframe) {
                    const double fps = oip->rate / (double)oip->scale * (fps_after_afs_is_24fps(oip->n, pe) ? 0.8 : 1.0);
                    const int vid_delay_ms = (int)(pe->delay_cut_additional_vframe * 1000.0 / fps + 0.5);
                    chapter.delay_chapter(vid_delay_ms);
                }
                //必要ならnero形式をUTF-8に変換
                chapter.overwrite_file(CHAP_TYPE_UNKNOWN, (sys_dat->exstg->s_local.chap_nero_convert_to_utf8 && CHAP_TYPE_NERO == chapter.file_chapter_type()));

                //mp4系ならapple形式チャプター追加も考慮する
                if (pe->muxer_to_be_used == MUXER_MP4 || 
                    pe->muxer_to_be_used == MUXER_TC2MP4 || 
                    pe->muxer_to_be_used == MUXER_MP4_RAW) {
                    //apple形式チャプターファイルへの置換が行われたら、apple形式チャプターファイルを作成する
                    if (strstr(cmd, "%{chap_apple}")) {
                        if (AUO_CHAP_ERR_NONE != chapter.write_file(chap_apple, CHAP_TYPE_ANOTHER, false)) {
                            warning_mux_chapter(chapter.get_result());
                            del_chap_cmd(cmd, TRUE);
                        } else {
                            replace(cmd, nSize, "%{chap_apple}", chap_apple);
                            if (CHAP_TYPE_APPLE == chapter.file_chapter_type()) {
                                swap_file(chap_apple, chap_file);
                            }
                        }
                    }
                }
            }
        }
    } else {
        enable_chap_mux = FALSE;
    }
    //音声ディレイ修正用コマンド %{delay_cmd}
    const AUDIO_SETTINGS *aud_stg = &sys_dat->exstg->s_aud[conf->aud.encoder];
    if (aud_stg->mode[conf->aud.enc_mode].delay
        && AUDIO_DELAY_CUT_EDTS == conf->aud.delay_cut
        && str_has_char(mux_stg->delay_cmd)) {
        char str[128] = { 0 };
        sprintf_s(str, "%d", aud_stg->mode[conf->aud.enc_mode].delay);
        replace(cmd, nSize, "%{delay_cmd}", mux_stg->delay_cmd);
        replace(cmd, nSize, "%{delay}", str);
    } else {
        replace(cmd, nSize, "%{delay_cmd}", "");
    }
    //その他の置換を実行
    cmd_replace(cmd, nSize, pe, sys_dat, conf, oip);
    //情報表示
    show_mux_info(mux_stg, enable_vid_mux, enable_aud_mux, enable_tc_mux, enable_chap_mux, muxer_mode->name);
    return AUO_RESULT_SUCCESS;
}

static void change_mux_vid_filename(const char *muxout, const PRM_ENC *pe) {
    char vidfile_append[MAX_APPENDIX_LEN];
    strcpy_s(vidfile_append, _countof(vidfile_append), "_video");
    strcat_s(vidfile_append, _countof(vidfile_append), PathFindExtension(pe->temp_filename));
    char vidfile_newname[MAX_PATH_LEN];
    apply_appendix(vidfile_newname, _countof(vidfile_newname), pe->temp_filename, vidfile_append);
    rename(pe->temp_filename, vidfile_newname);
    rename(muxout, pe->temp_filename);
}

//rawなのかmp4なのか (拡張子による判定)
static inline BOOL video_to_mux_is_raw(const PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    return !check_ext(pe->temp_filename, sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext);
}

//audio_to_mux_is_rawのモード決定用
enum {
    MODE_ONE = 0,                
    MODE_ALL = 0x80000000,
    MASK_ALL = ~MODE_ALL,
    ONE = MODE_ONE | MASK_ALL,  //ひとつでもrawならTRUEを返す
    ALL = MODE_ALL | MASK_ALL,  //すべてrawならTRUEを返す
};

//rawなのかmp4なのか (拡張子による判定)
static inline BOOL audio_to_mux_is_raw(const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, int target) {
    const DWORD mask = (DWORD)target & MASK_ALL;
    BOOL result = !!((DWORD)target & MODE_ALL);
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
        if (mask & (0x01 << i_aud)) {
            BOOL is_raw = (!check_ext(pe->append.aud[i_aud], ".m4a")
                        && !check_ext(pe->append.aud[i_aud], sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext));
            (((DWORD)target & MODE_ALL)) ? result &= is_raw : result |= is_raw;
        }
    }
    return result;
}

//mp4同士のmux専用のmuxerかどうか
static inline BOOL muxer_is_remux_only(const PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //mp4であり、かつMUXER_MP4_RAWが存在する
    return (pe->muxer_to_be_used == MUXER_MP4
        && str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].base_cmd));
}

//raw同士のmux専用のmuxerかどうか
static inline BOOL muxer_is_for_raw_only(const PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //MUXER_MP4_RAWであることと、その存在の確認
    return (pe->muxer_to_be_used == MUXER_MP4_RAW
        && str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].base_cmd));
}

//異なるmuxerを駆動し、最後にpe->muxer_to_be_usedを戻す
//PRM_ENCをコピーして使用しないのは、PRM_ENCのファイル名(拡張子)が
//muxによって変更されることがあるため(その情報を用いて、mp4かrawか判定していて、重要な情報)
static AUO_RESULT run_mux_as(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat, int run_as) {
    const int last_muxer = pe->muxer_to_be_used;
    pe->muxer_to_be_used = run_as;
    AUO_RESULT ret = mux(conf, oip, pe, sys_dat);
    pe->muxer_to_be_used = last_muxer;
    return ret;
}

static DWORD check_for_aud_mux(int oip_flag, const char *aud_cmd, const PRM_ENC *pe) {
    BOOL check = ((oip_flag != 0) && str_has_char(aud_cmd));
    DWORD flag = 0x00;
    for (int i = 0; i < pe->aud_count; i++)
        flag |= check << i;
    return flag;
}

AUO_RESULT mux(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    //muxの必要がなければ終了
    if (pe->muxer_to_be_used == MUXER_DISABLED)
        return ret;

    //映像・音声のmux判定
    BOOL  enable_vid_mux = TRUE;
    DWORD enable_aud_mux = check_for_aud_mux(oip->flag, sys_dat->exstg->s_mux[pe->muxer_to_be_used].aud_cmd, pe);
    BOOL  aud_use_remuxer = !!enable_aud_mux && sys_dat->exstg->s_aud[conf->aud.encoder].mode[conf->aud.enc_mode].use_remuxer;
    BOOL  enable_chap_mux = TRUE;
    //事前muxが必要なら実行 (L-SMASH remuxerの前のmuxer)
    if (pe->muxer_to_be_used == MUXER_TC2MP4 && video_to_mux_is_raw(pe, sys_dat)) {
        //mp4に格納された動画が必要
        if (AUO_RESULT_SUCCESS != (ret |= run_mux_as(conf, oip, pe, sys_dat, MUXER_MP4_RAW)))
            return ret;
    } else if (muxer_is_for_raw_only(pe, sys_dat)) {
        //raw用muxerに切り替えられていたら、コンテナへの格納が必要なものを処理する
        //必ずひとつひとつ格納するようにする
        //チャプターは(remuxerに戻ってから)後で処理するので、ここでは無効化しておく
        enable_chap_mux = FALSE;
        enable_vid_mux = (enable_vid_mux && video_to_mux_is_raw(pe, sys_dat));
        //raw用muxerに切り替えられていたら、muxerでは音声と映像は同時にmuxしない
        //特に、afs時はmuxerではmuxしない)
        enable_aud_mux = 0x00;
        if (!enable_vid_mux)
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
                if (0 != (enable_aud_mux = audio_to_mux_is_raw(pe, sys_dat, MODE_ONE | (0x01 << i_aud)) << i_aud))
                    break;
    } else if ((conf->vid.afs //自動フィールドシフト(timelineeditor)使用時のみ、個別のmuxが必要となる (timelienedtior実行後、post_muxでここを通る)
                || aud_use_remuxer)
        && muxer_is_remux_only(pe, sys_dat)) {
        //mp4用muxer(初期状態)で、動画・音声ともrawなら、raw用muxerに完全に切り替える
        if ((enable_vid_mux && video_to_mux_is_raw(pe, sys_dat)) && 
            (enable_aud_mux && audio_to_mux_is_raw(pe, sys_dat, ALL))) {
            pe->muxer_to_be_used = MUXER_MP4_RAW;
        } else {
            //mp4用muxer(初期状態)で、動画・音声のどちらかがrawなら、rawのものを事前にmuxerでmp4に格納する。
            if (enable_vid_mux && video_to_mux_is_raw(pe, sys_dat))
                if (AUO_RESULT_SUCCESS != (ret |= run_mux_as(conf, oip, pe, sys_dat, MUXER_MP4_RAW)))
                    return ret;
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
                if ((enable_aud_mux & (0x01 << i_aud)) && audio_to_mux_is_raw(pe, sys_dat, MODE_ONE | (0x01 << i_aud)))
                    if (AUO_RESULT_SUCCESS != (ret |= run_mux_as(conf, oip, pe, sys_dat, MUXER_MP4_RAW)))
                        return ret;
        }
    } else if (pe->muxer_to_be_used == MUXER_MP4 && !(conf->vid.afs || aud_use_remuxer) && str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].base_cmd)) {
        //自動フィールドシフト(timelineeditor)使用時以外は、remuxerを使用しなくても良くなった
        //なので、単純に使用するmuxerをmuxer.exeに切り替え
        pe->muxer_to_be_used = MUXER_MP4_RAW;
    }

    //mux処理の開始
    const MUXER_SETTINGS *mux_stg = &sys_dat->exstg->s_mux[pe->muxer_to_be_used];

    if (!PathFileExists(mux_stg->fullpath)) {
        ret |= AUO_RESULT_ERROR; error_no_exe_file(mux_stg->dispname, mux_stg->fullpath);
        return ret;
    }
    if (pe->muxer_to_be_used == MUXER_TC2MP4 && !PathFileExists(sys_dat->exstg->s_mux[MUXER_MP4].fullpath)) {        
        ret |= AUO_RESULT_ERROR; error_no_exe_file(sys_dat->exstg->s_mux[MUXER_MP4].dispname, sys_dat->exstg->s_mux[MUXER_MP4].fullpath);
        return ret;
    }
    UINT64 expected_filesize = 0;
    char muxcmd[MAX_CMD_LEN]  = { 0 };
    char muxargs[MAX_CMD_LEN] = { 0 };
    char muxdir[MAX_PATH_LEN] = { 0 };
    char muxout[MAX_PATH_LEN] = { 0 };
    DWORD mux_priority = GetExePriority(conf->mux.priority, pe->h_p_aviutl);
    get_muxout_filename(muxout, _countof(muxout), sys_dat, pe);

    PIPE_SET pipes = { 0 };
    LOG_CACHE log_line_cache = { 0 };
    PROCESS_INFORMATION pi_mux = { 0 };
    int rp_ret;

    //ログキャッシュの初期化
    if (init_log_cache(&log_line_cache)) {
        error_log_line_cache();
        return AUO_RESULT_ERROR;
    }

    PathGetDirectory(muxdir, _countof(muxdir), mux_stg->fullpath);

    //mux終了後の予想サイズを取得
    ret |= get_expected_filesize(pe, enable_vid_mux, enable_aud_mux, &expected_filesize);
    if (ret & AUO_RESULT_ERROR)
        return AUO_RESULT_ERROR;

    //コマンドライン生成・情報表示
    ret |= build_mux_cmd(muxcmd, _countof(muxcmd), conf, oip, pe, sys_dat, mux_stg, expected_filesize, enable_vid_mux, enable_aud_mux, enable_chap_mux);
    if (ret & AUO_RESULT_ERROR)
        return AUO_RESULT_ERROR; //エラーメッセージはbuild_mux_cmd関数内で吐かれる
    sprintf_s(muxargs, _countof(muxargs), "\"%s\" %s", mux_stg->fullpath, muxcmd);
    write_log_auo_line(LOG_MORE, muxargs);
    //パイプの設定
    pipes.stdOut.mode = AUO_PIPE_ENABLE;
    pipes.stdErr.mode = AUO_PIPE_MUXED;

    if ((rp_ret = RunProcess(muxargs, muxdir, &pi_mux, &pipes, mux_priority, TRUE, conf->mux.minimized)) != RP_SUCCESS) {
        //エラー
        ret |= AUO_RESULT_ERROR; error_run_process(mux_stg->dispname, rp_ret);
    } else {
        while (WaitForSingleObject(pi_mux.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (0 == ReadLogExe(&pipes, mux_stg->dispname, &log_line_cache))
                log_process_events();
        }
        //最後のメッセージを回収
        while (ReadLogExe(&pipes, mux_stg->dispname, &log_line_cache) > 0);

        ret |= check_muxout_filesize(muxout, expected_filesize);
        int muxer_log_level = LOG_MORE;
        if (ret == AUO_RESULT_SUCCESS) {
            if (enable_vid_mux) {
                if (str_has_char(pe->muxed_vid_filename) && PathFileExists(pe->muxed_vid_filename)) remove(pe->muxed_vid_filename);
                apply_appendix(pe->muxed_vid_filename, _countof(pe->muxed_vid_filename), pe->temp_filename, VID_FILE_APPENDIX);
                strcat_s(pe->muxed_vid_filename, _countof(pe->muxed_vid_filename), PathFindExtension(pe->temp_filename));
                if (PathFileExists(pe->muxed_vid_filename)) remove(pe->muxed_vid_filename);
                rename(pe->temp_filename, pe->muxed_vid_filename);
                change_ext(pe->temp_filename, _countof(pe->temp_filename), mux_stg->out_ext); //拡張子を変更
                if (PathFileExists(pe->temp_filename)) remove(pe->temp_filename);
                rename(muxout, pe->temp_filename);
            } else {
                //音声のみmuxなら、一時音声ファイルの情報を変更する
                char aud_file[MAX_PATH_LEN] = { 0 };
                for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                    if (enable_aud_mux & (0x01 << i_aud)) {
                        get_aud_filename(aud_file, _countof(aud_file), pe, i_aud);
                        remove(aud_file);
                        change_ext(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), mux_stg->out_ext); //拡張子を変更
                        get_aud_filename(aud_file, _countof(aud_file), pe, i_aud);
                        if (PathFileExists(aud_file)) remove(aud_file);
                        rename(muxout, aud_file);
                    }
                }
            }
        } else if (ret & AUO_RESULT_ERROR) {
            muxer_log_level = LOG_ERROR;
            error_mux_failed(mux_stg->dispname, muxargs);
            if (PathFileExists(muxout))
                remove(muxout);
        } else {
            //AUO_RESULT_WARNING
            change_mux_vid_filename(muxout, pe);
        }
        write_cached_lines(muxer_log_level, mux_stg->dispname, &log_line_cache);
        write_log_auo_line_fmt(LOG_MORE, "%s CPU使用率: %.2f%%", mux_stg->dispname, GetProcessAvgCPUUsage(pi_mux.hProcess));
        CloseHandle(pi_mux.hProcess);
        CloseHandle(pi_mux.hThread);
    }

    release_log_cache(&log_line_cache);
    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

    //さらにmuxの必要があれば、それを行う(L-SMASH系 timelineeditor のあとの remuxer を想定)
    if (!ret && mux_stg->post_mux >= MUXER_MP4) {
        if (AUO_RESULT_SUCCESS != (ret |= run_mux_as(conf, oip, pe, sys_dat, mux_stg->post_mux)))
            return ret;
    }

    return ret;
}
