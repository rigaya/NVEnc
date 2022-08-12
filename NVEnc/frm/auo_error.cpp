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

#include "auo.h"
#include "auo_version.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_chapter.h"
#include "auo_convert.h"

void warning_conf_not_initialized() {
    write_log_auo_line(LOG_WARNING, "設定が行われていないため、デフォルト設定でエンコードを行います。");
    write_log_auo_line(LOG_WARNING, "設定を変更するには[プラグイン出力]>[" AUO_FULL_NAME "]の画面で「ビデオ圧縮」をクリックし、");
    write_log_auo_line(LOG_WARNING, "設定画面で設定を変更したのち [ OK ] をクリックしてください。");
}

void warning_failed_getting_temp_path() {
    write_log_auo_line(LOG_WARNING, "一時フォルダ名取得に失敗しました。一時フォルダ指定を解除しました。");
}

void warning_unable_to_open_tempfile(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING,
        "指定された一時フォルダ \"%s\" にファイルを作成できません。一時フォルダ指定を解除しました。",
        dir);
}

void warning_no_temp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING,
        "指定された一時フォルダ \"%s\" が存在しません。一時フォルダ指定を解除しました。",
        dir);
}

void warning_no_aud_temp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING,
        "指定された音声用一時フォルダ \"%s\" が存在しません。一時フォルダ指定を解除しました。",
        dir);
}

void error_filename_too_long() {
    write_log_auo_line(LOG_ERROR, "出力ファイル名が長すぎます。もっと短くしてください。");
}

void error_savdir_do_not_exist(const char *savname, const char *savedir) {
    write_log_auo_line(    LOG_ERROR, "出力先フォルダを認識できないため、出力できません。");
    write_log_auo_line_fmt(LOG_ERROR, "  出力ファイル名: \"%s\"", savname);
    write_log_auo_line_fmt(LOG_ERROR, "  出力先フォルダ: \"%s\"", savedir);
    if (strchr(savedir, '?') != nullptr) {
        write_log_auo_line(LOG_ERROR, "このエラーは、上記出力先のフォルダ名に環境依存文字を含む場合に発生することがあります。");
        write_log_auo_line(LOG_ERROR, "  該当文字は、\"?\"で表示されています。");
        write_log_auo_line(LOG_ERROR, "  環境依存文字を含まないフォルダに出力先に変更して出力しなおしてください。");
    }
}

void error_file_is_already_opened_by_aviutl() {
    write_log_auo_line(LOG_ERROR, "出力ファイルはすでにAviutlで開かれているため、出力できません。");
    write_log_auo_line(LOG_ERROR, "異なるファイル名を指定してやり直してください。");
}

void warning_no_auo_check_fileopen() {
    write_log_auo_line_fmt(LOG_WARNING, "映像の出力ファイルチェック用のサブプロセス %s が %s 以下に存在しません。",
        AUO_CHECK_FILEOPEN_NAME, DEFAULT_EXE_DIR);
    write_log_auo_line_fmt(LOG_WARNING, "同梱の %s フォルダをAviutlフォルダ内にすべてコピーできているか、再確認してください。", DEFAULT_EXE_DIR);
}

static void error_failed_to_open_temp_file_dir(const char *temp_filename, const char *mesBuffer, const DWORD err, const BOOL target_is_dir) {
    if (target_is_dir) {
        write_log_auo_line_fmt(LOG_ERROR, "出力先 \"%s\" にファイルを作成できません。", temp_filename);
    } else {
        write_log_auo_line_fmt(LOG_ERROR, "映像の出力ファイル \"%s\" を開くことができません。", temp_filename);
    }
    write_log_auo_line_fmt(LOG_ERROR, "  %s", mesBuffer);
    if (strchr(temp_filename, '?') != nullptr) {
        write_log_auo_line(LOG_ERROR, "このエラーは、出力%s名に環境依存文字を含む場合に発生することがあります。", (target_is_dir) ? "フォルダ" : "ファイル");
        write_log_auo_line(LOG_ERROR, "  該当文字は、\"?\"で表示されていますので該当文字を避けた%sに出力しなおしてください。", (target_is_dir) ? "フォルダ" : "ファイル");
    } else if (err == ERROR_ACCESS_DENIED) {
        char systemdrive_dir[MAX_PATH_LEN] = { 0 };
        char systemroot_dir[MAX_PATH_LEN] = { 0 };
        char programdata_dir[MAX_PATH_LEN] = { 0 };
        char programfiles_dir[MAX_PATH_LEN] = { 0 };
        //char programfilesx86_dir[MAX_PATH_LEN];
        ExpandEnvironmentStrings("%SystemDrive%", systemdrive_dir, _countof(systemdrive_dir));
        ExpandEnvironmentStrings("%SystemRoot%", systemroot_dir, _countof(systemroot_dir));
        ExpandEnvironmentStrings("%PROGRAMDATA%", programdata_dir, _countof(programdata_dir));
        ExpandEnvironmentStrings("%PROGRAMFILES%", programfiles_dir, _countof(programfiles_dir));
        //ExpandEnvironmentStrings("%PROGRAMFILES(X86)%", programfilesx86_dir, _countof(programfilesx86_dir));
        write_log_auo_line(LOG_ERROR, "このエラーは、アクセス権のないフォルダ、あるいはWindowsにより保護されたフォルダに");
        write_log_auo_line(LOG_ERROR, "出力しようとすると発生することがあります。");
        write_log_auo_line(LOG_ERROR, "出力先のフォルダを変更して出力しなおしてください。");
        write_log_auo_line(LOG_ERROR, "なお、下記はWindowsにより保護されたフォルダですので、ここへの出力は避けてください。");
        write_log_auo_line_fmt(LOG_ERROR, "例: %s ドライブ直下", systemdrive_dir);
        write_log_auo_line_fmt(LOG_ERROR, "    %s 以下", systemroot_dir);
        write_log_auo_line_fmt(LOG_ERROR, "    %s 以下", programdata_dir);
        write_log_auo_line_fmt(LOG_ERROR, "    %s 以下", programfiles_dir);
        //write_log_auo_line_fmt(LOG_ERROR, "    %s 以下", programfilesx86_dir);
        write_log_auo_line(LOG_ERROR, "    など");
        write_log_auo_line(LOG_ERROR, "");
    } else {
        write_log_auo_line(LOG_ERROR, "出力先のフォルダ・ファイルを変更して出力しなおしてください。");
    }
}

void error_failed_to_open_tempdir(const char *temp_dir, const char *mesBuffer, const DWORD err) {
    error_failed_to_open_temp_file_dir(temp_dir, mesBuffer, err, true);
}

void error_failed_to_open_tempfile(const char *temp_filename, const char *mesBuffer, const DWORD err) {
    error_failed_to_open_temp_file_dir(temp_filename, mesBuffer, err, false);
}

void error_nothing_to_output() {
    write_log_auo_line(LOG_ERROR, "出力すべきものがありません。");
}

void error_output_zero_frames() {
    write_log_auo_line(LOG_ERROR, "出力フレーム数が 0 フレームのため、エンコードできません。");
    write_log_auo_line(LOG_ERROR, "選択範囲が適切になっているか確認して出力しなおしてください。");
}

void info_afs_audio_delay_confliction() {
    write_log_auo_line(LOG_INFO, "自動フィールドシフト、音声ディレイカット[動画追加]が同時に指定されている場合には、音声エンコードは後で行います。");
}

void error_invalid_resolution(BOOL width, int mul, int w, int h) {
    write_log_auo_line_fmt(LOG_ERROR, "%s入力解像度が %d で割りきれません。エンコードできません。入力解像度:%dx%d",
        (width) ? "横" : "縦", mul, w, h);
}

void error_log_line_cache() {
    write_log_auo_line(LOG_ERROR, "ログ保存キャッシュ用メモリ確保に失敗しました。");
}

void error_no_exe_file(const char *name, const char *path) {
    if (strlen(path))
        write_log_auo_line_fmt(LOG_ERROR, "指定された %s が %s にありません。", name, path);
    else
        write_log_auo_line_fmt(LOG_ERROR, "%s の場所が指定されていません。", name);
    write_log_auo_line_fmt(LOG_ERROR, "%s を用意し、その場所を設定画面から正しく指定してください。", name);
}

void warning_use_default_audio_encoder(const char *name) {
    write_log_auo_line_fmt(LOG_WARNING, "音声エンコーダが適切に設定されていないため、デフォルトの音声エンコーダ %s を使用します。", name);
}

void info_use_exe_found(const char *target, const char *path) {
    write_log_auo_line_fmt(LOG_INFO, "%sとして \"%s\" を使用します。", target, path);
}

void error_invalid_ini_file() {
    write_log_auo_line(LOG_ERROR, "プラグイン(auo)とiniファイルの音声エンコーダの記述が一致しません。");
}

void error_unsupported_audio_format_by_muxer(const int video_out_type, const char *selected_aud, const char *default_aud) {
    if (video_out_type < _countof(OUTPUT_FILE_EXT)) {
        write_log_auo_line_fmt(LOG_ERROR, "音声エンコーダ %s は、%s 形式での出力に対応していません。", selected_aud, OUTPUT_FILE_EXT[video_out_type] + 1);
        if (default_aud) {
            write_log_auo_line_fmt(LOG_ERROR, "%s 等の他の音声エンコーダを選択して出力してください。", default_aud);
        } else {
            write_log_auo_line(LOG_ERROR, "他の音声エンコーダを選択して出力してください。");
        }
    }
}

void warning_auto_afs_disable() {
    write_log_line(LOG_WARNING, ""
        "auo [warning]: Aviutlからの映像入力の初期化に失敗したため、\n"
        "               自動フィールドシフト(afs)をオフにして再初期化を行いました。\n"
        "               この問題は、Aviutlでafsを使用していないにも関わらず、\n"
        "               x264guiEx側でafsをオンにしていると発生します。\n"
        "               他のエラーの可能性も考えられます。afsがオフになっている点に注意してください。"
        );
}

void error_afs_setup(BOOL afs, BOOL auto_afs_disable) {
    if (afs && !auto_afs_disable) {
        write_log_line(LOG_ERROR, ""
            "auo [error]: Aviutlからの映像入力の初期化に失敗しました。以下のような原因が考えられます。\n"
            "             ・自動フィールドシフト(afs)をAviutlで使用していないにもかかわらず、\n"
            "               x264guiExの設定画面で自動フィールドシフトにチェックを入れていたり、\n"
            "               自動フィールドシフト非対応の動画(60fps読み込み等)を入力したりしている。\n"
            "             ・メモリ不足による、メモリ確保の失敗。"
            );
    } else
        write_log_auo_line(LOG_ERROR, "Aviutlからの映像入力の初期化に失敗しました。メモリを確保できませんでした。");
}

void error_open_pipe() {
    write_log_auo_line(LOG_ERROR, "パイプの作成に失敗しました。");
}

void error_get_pipe_handle() {
    write_log_auo_line(LOG_ERROR, "パイプハンドルの取得に失敗しました。");
}

void error_run_process(const char *exe_name, int rp_ret) {
    switch (rp_ret) {
        case RP_ERROR_OPEN_PIPE:
            write_log_auo_line(LOG_ERROR, "パイプの作成に失敗しました。");
            break;
        case RP_ERROR_GET_STDIN_FILE_HANDLE:
            write_log_auo_line(LOG_ERROR, "パイプハンドルの取得に失敗しました。");
            break;
        case RP_ERROR_CREATE_PROCESS:
        default:
            write_log_auo_line_fmt(LOG_ERROR, "%s の実行に失敗しました。", exe_name);
            break;
    }
}

void error_video_create_param_mem() {
    write_log_auo_line(LOG_ERROR, "パラメータ保持用のメモリ確保に失敗しました。");
}

void error_video_create_event() {
    write_log_auo_line(LOG_ERROR, "読み込み用のイベント作成に失敗しました。");
}

void error_video_wait_event() {
    write_log_auo_line(LOG_ERROR, "読み込み用のイベント待機に失敗しました。");
}

void error_video_set_event() {
    write_log_auo_line(LOG_ERROR, "読み込み用のイベントセットに失敗しました。");
}

void error_video_open_shared_input_buf() {
    write_log_auo_line(LOG_ERROR, "読み込み用のメモリのオープンに失敗しました。");
}

void error_video_get_conv_func() {
    write_log_auo_line(LOG_ERROR, "色変換用の関数選択に失敗しました。");
}

void warning_auto_qpfile_failed() {
    write_log_auo_line(LOG_WARNING, "Aviutlのキーフレーム検出用 qpfileの自動作成に失敗しました。");
}

void warning_auo_tcfile_failed() {
    write_log_auo_line(LOG_WARNING, "タイムコードファイル作成に失敗しました。");
}

void error_malloc_pixel_data() {
    write_log_auo_line(LOG_ERROR, "映像バッファ用メモリ確保に失敗しました。");
}

void error_malloc_tc() {
    write_log_auo_line(LOG_ERROR, "タイムコード用メモリ確保に失敗しました。");
}

void error_malloc_8bit() {
    write_log_auo_line(LOG_ERROR, "音声16bit→8bit変換用メモリ確保に失敗しました。");
}

void error_afs_interlace_stg() {
    write_log_line(LOG_ERROR,
        "auo [error]: 自動フィールドシフトとインターレース設定が両方オンになっており、設定が矛盾しています。\n"
        "             設定を見なおしてください。");
}

void error_videnc_dead() {
    write_log_auo_line_fmt(LOG_ERROR, "%sが予期せず途中終了しました。%sに不正なパラメータ(オプション)が渡された可能性があります。", ENCODER_NAME, ENCODER_NAME);
}

void error_videnc_dead_and_nodiskspace(const char *drive, uint64_t diskspace) {
    write_log_auo_line_fmt(LOG_ERROR, "%sが予期せず途中終了しました。", ENCODER_NAME);
    write_log_auo_line_fmt(LOG_ERROR, "%sドライブの空き容量が残り %.2f MBしかありません。", drive, (double)diskspace / (1024 * 1024));
    write_log_auo_line_fmt(LOG_ERROR, "%sドライブの空き容量不足で失敗した可能性があります。", drive);
    write_log_auo_line_fmt(LOG_ERROR, "%sドライブの空きをつくり、再度実行しなおしてください。", drive);
}

void error_videnc_version(const char *required_ver, const char *current_ver) {
    write_log_line_fmt(LOG_ERROR, ""
        "auo [error]: %sのバージョンが古く、エンコードできません。\n"
        "             最新の%sをダウンロードし、設定画面で最新版に指定しなおしてください。\n"
        "             必要なバージョン:         %s\n"
        "             実行ファイルのバージョン: %s\n",
        ENCODER_NAME, required_ver, current_ver);
}

void error_x264_version() {
    write_log_line(LOG_ERROR, ""
        "auo [error]: NVEncCのバージョンが古く、エンコードできません。\n"
        "             最新のNVEncCをダウンロードし、設定画面で最新版に指定しなおしてください。");
}

void error_afs_get_frame() {
    write_log_auo_line(LOG_ERROR, "Aviutlからのフレーム読み込みに失敗しました。");
}

void error_open_wavfile() {
    write_log_auo_line(LOG_ERROR, "wavファイルのオープンに失敗しました。");
}

void error_no_wavefile() {
    write_log_auo_line(LOG_ERROR, "wavファイルがみつかりません。音声エンコードに失敗しました。");
}

static void message_audio_length_different(const double video_length, const double audio_length, const BOOL exedit_is_used, const BOOL audio_length_changed) {
    const int vid_h = (int)(video_length / 3600);
    const int vid_m = (int)(video_length - vid_h * 3600) / 60;
    const int vid_s = (int)(video_length - vid_h * 3600 - vid_m * 60);
    const int vid_ms = std::min((int)((video_length - (double)(vid_h * 3600 + vid_m * 60 + vid_s)) * 1000.0), 999);

    const int aud_h = (int)audio_length / 3600;
    const int aud_m = (int)(audio_length - aud_h * 3600) / 60;
    const int aud_s = (int)(audio_length - aud_h * 3600 - aud_m * 60);
    const int aud_ms = std::min((int)((audio_length - (double)(aud_h * 3600 + aud_m * 60 + aud_s)) * 1000.0), 999);

    if (audio_length_changed) {
        write_log_line_fmt(LOG_INFO,
            "auo [info]: 音声の長さが映像の長さと異なるようです。\n"
            "            映像: %d:%02d:%02d.%03d, 音声: %d:%02d:%02d.%03d\n",
            vid_h, vid_m, vid_s, vid_ms,
            aud_h, aud_m, aud_s, aud_ms);
        write_log_line_fmt(LOG_INFO, ""
            "            音声の長さが映像の長さに一致するよう、自動的に調整しました。\n");
        if (exedit_is_used) {
            write_log_line_fmt(LOG_INFO, ""
                "            拡張編集の音声トラックとAviutl本体の音声トラックが競合している可能性があります。\n"
                "            拡張編集使用時には、Aviutl本体の音声トラック読み込みを使用しないようご注意ください。\n");
        }
    } else {
        write_log_line_fmt(LOG_WARNING,
            "auo [warning]: 音声の長さが映像の長さと異なるようです。\n"
            "               映像: %d:%02d:%02d.%03d, 音声: %d:%02d:%02d.%03d\n",
            vid_h, vid_m, vid_s, vid_ms,
            aud_h, aud_m, aud_s, aud_ms);
        if (exedit_is_used) {
            write_log_line_fmt(LOG_WARNING, ""
                "               拡張編集の音声トラックとAviutl本体の音声トラックが競合している可能性があります。\n"
                "               拡張編集使用時には、Aviutl本体の音声トラック読み込みを使用しないようご注意ください。\n");
        } else {
            write_log_line_fmt(LOG_WARNING, ""
                "               これが意図したものでない場合、音声が正常に出力されていないかもしれません。\n"
                "               この問題は圧縮音声をソースとしていると発生することがあります。\n"
                "               一度音声をデコードし、「音声読み込み」から無圧縮wavとして別に読み込むか、\n"
                "               異なる入力プラグインを利用して読み込むといった方法を試してみてください。");
        }
    }
}

void info_audio_length_changed(const double video_length, const double audio_length, const BOOL exedit_is_used) {
    message_audio_length_different(video_length, audio_length, exedit_is_used, TRUE);
}

void warning_audio_length(const double video_length, const double audio_length, const BOOL exedit_is_used) {
    message_audio_length_different(video_length, audio_length, exedit_is_used, FALSE);
}

void error_audenc_failed(const char *name, const char *args) {
    write_log_auo_line_fmt(LOG_ERROR, "出力音声ファイルがみつかりません。%s での音声のエンコードに失敗しました。", name);
    if (args) {
        write_log_auo_line(    LOG_ERROR, "音声エンコードのコマンドラインは…");
        write_log_auo_line(    LOG_ERROR, args);
    }
}

void error_mux_failed(const char *name, const char *args) {
    write_log_auo_line_fmt(LOG_ERROR, "%s でのmuxに失敗しました。", name);
    write_log_auo_line(    LOG_ERROR, "muxのコマンドラインは…");
    write_log_auo_line(    LOG_ERROR, args);
}

void warning_no_mux_tmp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING,
        "指定されたmux用一時ドライブ \"%s\" が存在しません。一時フォルダ指定を解除しました。",
        dir);
}

void warning_failed_mux_tmp_drive_space() {
    write_log_auo_line(LOG_WARNING, "指定されたmux用一時フォルダのあるドライブの空き容量取得に失敗しました。mux用一時フォルダ指定を解除しました。");
}

void warning_failed_muxer_drive_space() {
    write_log_auo_line(LOG_WARNING, "muxerのあるドライブの空き容量取得に失敗しました。容量不足によりmuxが失敗する可能性があります。");
}

void warning_failed_out_drive_space() {
    write_log_auo_line(LOG_WARNING, "出力先のあるドライブの空き容量取得に失敗しました。容量不足によりmuxが失敗する可能性があります。");
}

void warning_failed_get_aud_size() {
    write_log_auo_line(LOG_WARNING, "音声一時ファイルのサイズ取得に失敗しました。muxが正常に行えるか確認できません。");
}

void warning_failed_get_vid_size() {
    write_log_auo_line(LOG_WARNING, "映像一時ファイルのサイズ取得に失敗しました。muxが正常に行えるか確認できません。");
}

void error_no_aud_file() {
    write_log_auo_line(LOG_ERROR, "音声一時ファイルが見つかりません。muxを行えません。");
}

void error_no_vid_file() {
    write_log_auo_line(LOG_ERROR, "映像一時ファイルが見つかりません。muxを行えません。");
}

void error_aud_file_zero_byte() {
    write_log_auo_line(LOG_ERROR, "音声一時ファイルが 0 byteです。muxを行えません。");
}

void error_vid_file_zero_byte() {
    write_log_auo_line(LOG_ERROR, "映像一時ファイルが 0 byteです。muxを行えません。");
}

void warning_mux_tmp_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line_fmt(LOG_WARNING, "mux一時フォルダのある%sドライブに十分な空きがありません。", drive);
    write_log_auo_line_fmt(LOG_WARNING, "  必要サイズ %.2f MB, 残り空き容量 %.2f MB", (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
    write_log_auo_line(LOG_WARNING, "mux用一時フォルダ指定を解除しました。");
}

void error_muxer_drive_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line(LOG_ERROR, "muxerのある%sドライブに十分な空きがありません。muxを行えません。", drive);
    write_log_auo_line_fmt(LOG_ERROR, "  必要サイズ %.2f MB, 残り空き容量 %.2f MB", (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
}

void error_out_drive_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line(LOG_ERROR, "出力先の%sドライブに十分な空きがありません。muxを行えません。", drive);
    write_log_auo_line_fmt(LOG_ERROR, "  必要サイズ %.2f MB, 残り空き容量 %.2f MB", (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
}

void warning_failed_to_get_duration_from_timecode() {
    write_log_auo_line(LOG_WARNING, "タイムコードからの動画長さの取得に失敗しました。");
    write_log_auo_line(LOG_WARNING, "Apple形式チャプターに記述する動画長さはAviutlから取得したものを使用します。");
    write_log_auo_line(LOG_WARNING, "そのため、チャプターストリームの長さが実際の動画と異なる恐れがあります。");
}

void error_check_muxout_exist() {
    write_log_auo_line(LOG_ERROR, "mux後ファイルが見つかりませんでした。");
}

void error_check_muxout_too_small(int expected_filesize_KB, int muxout_filesize_KB) {
    write_log_auo_line    (LOG_ERROR, "mux後ファイルが小さすぎます。muxに失敗したものと思われます。");
    write_log_auo_line_fmt(LOG_ERROR, "推定ファイルサイズ %d KB,  出力ファイルサイズ %d KB", expected_filesize_KB, muxout_filesize_KB);
}

void warning_failed_check_muxout_filesize() {
    write_log_auo_line(LOG_WARNING, "mux後ファイルのファイルサイズ確認に失敗しました。正常にmuxされていない可能性があります。");
}

void warning_no_auto_save_log_dir() {
    write_log_auo_line(LOG_WARNING, "指定した自動ログ保存先が存在しません。動画出力先に保存します。");
}

void info_encoding_aborted() {
    write_log_auo_line(LOG_INFO, "エンコードを中断しました。");
}

void warning_mux_no_chapter_file() {
    write_log_auo_line(LOG_WARNING, "指定されたチャプターファイルが存在しません。チャプターはmuxされません。");
}

void warning_mux_chapter(int sts) {
    switch (sts) {
        case AUO_CHAP_ERR_NONE: break;
        case AUO_CHAP_ERR_FILE_OPEN:        write_log_auo_line(LOG_WARNING, "チャプターファイルのオープンに失敗しました。"); break;
        case AUO_CHAP_ERR_FILE_READ:        write_log_auo_line(LOG_WARNING, "チャプターファイルの読み込みに失敗しました。"); break;
        case AUO_CHAP_ERR_FILE_WRITE:       write_log_auo_line(LOG_WARNING, "チャプターファイルの書き込みに失敗しました。"); break;
        case AUO_CHAP_ERR_FILE_SWAP:        write_log_auo_line(LOG_WARNING, "チャプターファイル名の交換に失敗しました。"); break;
        case AUO_CHAP_ERR_CP_DETECT:        write_log_auo_line(LOG_WARNING, "チャプターファイルのコードページの判定に失敗しました。"); break;
        case AUO_CHAP_ERR_INIT_IMUL2:       write_log_auo_line(LOG_WARNING, "コードページ変換の初期化に失敗しました。"); break;
        case AUO_CHAP_ERR_INVALID_FMT:      write_log_auo_line(LOG_WARNING, "指定されたチャプターファイルの書式が不正です。"); break;
        case AUO_CHAP_ERR_NULL_PTR:         write_log_auo_line(LOG_WARNING, "ぬるぽ。"); break;
        case AUO_CHAP_ERR_INIT_XML_PARSER:  write_log_auo_line(LOG_WARNING, "Xml Parserの初期化に失敗しました。"); break;
        case AUO_CHAP_ERR_INIT_READ_STREAM: write_log_auo_line(LOG_WARNING, "チャプターファイルのオープンに失敗しました。"); break;
        case AUO_CHAP_ERR_FAIL_SET_STREAM:  write_log_auo_line(LOG_WARNING, "Xml Parserと入力ストリームの接続に失敗しました。"); break;
        case AUO_CHAP_ERR_PARSE_XML:        write_log_auo_line(LOG_WARNING, "チャプターファイルの読み取りに失敗しました。"); break;
        default:                            write_log_auo_line(LOG_WARNING, "チャプターmux: 不明なエラーが発生しました。"); break;
    }
    return;
}

void warning_chapter_convert_to_utf8(int sts) {
    write_log_auo_line_fmt(LOG_WARNING, "チャプターファイルのUTF-8への変換に失敗しました。");
    warning_mux_chapter(sts);
}

void error_select_convert_func(int width, int height, BOOL use16bit, BOOL interlaced, int output_csp) {
    write_log_auo_line(LOG_ERROR, "色形式変換関数の取得に失敗しました。");
    write_log_auo_line_fmt(LOG_ERROR, "%dx%d%s, output-csp %s%s%s",
        width, height,
        (interlaced) ? "i" : "p",
        specify_csp[output_csp],
        (use16bit) ? "(16bit)" : ""
    );
}

void warning_no_batfile(const char *batfile) {
    write_log_auo_line_fmt(LOG_WARNING, "指定されたバッチファイル \"%s\"が存在しません。", batfile);
}

void warning_malloc_batfile_tmp() {
    write_log_auo_line(LOG_WARNING, "一時バッチファイル作成用バッファの確保に失敗しました。");
}

void warning_failed_open_bat_orig() {
    write_log_auo_line(LOG_WARNING, "バッチファイルを開けませんでした。");
}

void warning_failed_open_bat_new() {
    write_log_auo_line(LOG_WARNING, "一時バッチファイルを作成できませんでした。");
}
