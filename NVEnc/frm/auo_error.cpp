//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "auo.h"
#include "auo_version.h"
#include "auo_frm.h" 
#include "auo_pipe.h"
#include "auo_chapter.h"

void warning_failed_getting_temp_path() {
    write_log_auo_line(LOG_WARNING, "一時フォルダ名取得に失敗しました。一時フォルダ指定を解除しました。");
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

void error_nothing_to_output() {
    write_log_auo_line(LOG_ERROR, "出力すべきものがありません。");
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

void error_x264_dead() {
    write_log_auo_line(LOG_ERROR, "x264が予期せず途中終了しました。x264に不正なパラメータ(オプション)が渡された可能性があります。");
}

void error_x264_version() {
    write_log_line(LOG_ERROR, ""
        "auo [error]: x264のバージョンが古く、エンコードできません。\n"
        "             最新のx264をダウンロードし、設定画面で最新版に指定しなおしてください。");
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

void warning_audio_length() {
    write_log_line(LOG_WARNING, 
        "auo [warning]: 音声の長さが動画の長さと大きく異なるようです。\n"
        "               これが意図したものでない場合、音声が正常に出力されていないかもしれません。\n"
        "               この問題は圧縮音声をソースとしていると発生することがあります。\n"
        "               一度音声をデコードし、「音声読み込み」から無圧縮wavとして別に読み込むか、\n"
        "               異なる入力プラグインを利用して読み込むといった方法を試してみてください。");
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

void error_failed_muxer_drive_space() {
    write_log_auo_line(LOG_ERROR, "muxerのあるドライブの空き容量取得に失敗しました。muxを行えません。");
}

void error_failed_out_drive_space() {
    write_log_auo_line(LOG_ERROR, "出力先のあるドライブの空き容量取得に失敗しました。muxを行えません。");
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

void warning_mux_tmp_not_enough_space() {
    write_log_auo_line(LOG_WARNING, "mux一時フォルダのあるドライブに十分な空きがありません。mux用一時フォルダ指定を解除しました。");
}

void error_muxer_drive_not_enough_space() {
    write_log_auo_line(LOG_ERROR, "muxerのあるドライブに十分な空きがありません。muxを行えません。");
}

void error_out_drive_not_enough_space() {
    write_log_auo_line(LOG_ERROR, "出力先のドライブに十分な空きがありません。muxを行えません。");
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
