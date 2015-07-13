//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_ENCODE_H_
#define _AUO_ENCODE_H_

#include <Windows.h>
#include <stdio.h>

#include "auo.h"
#include "output.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_frm.h"
#include "auo_pipe.h"

static const char * const PIPE_FN = "-";

static const char * const VID_FILE_APPENDIX = "_vid";

typedef AUO_RESULT (*encode_task) (CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat);

BOOL check_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const guiEx_settings *exstg);
void open_log_window(const char *savefile, const SYSTEM_DATA *sys_dat, int current_pass, int total_pass);
void auto_save_log(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat);
void set_enc_prm(CONF_GUIEX *conf, PRM_ENC *pe, const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat);

int additional_vframe_for_aud_delay_cut(double fps, int audio_rate, int audio_delay);
int additional_silence_for_aud_delay_cut(double fps, int audio_rate, int audio_delay, int vframe_added = -1);
BOOL fps_after_afs_is_24fps(const int frame_n, const PRM_ENC *pe);

void get_aud_filename(char *audfile, size_t nSize, const PRM_ENC *pe, int i_aud); //音声一時ファイル名を作成
void get_muxout_filename(char *filename, size_t nSize, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe); //mux出力ファイル名を作成
void set_chap_filename(char *chap_file, size_t cf_nSize, char *chap_apple, size_t ca_nSize, const char *chap_base, 
                       const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip); //チャプターファイルのパスを生成
void insert_num_to_replace_key(char *key, size_t nSize, int num);
void cmd_replace(char *cmd, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip); //コマンドラインの共通置換を実行
AUO_RESULT move_temporary_files(const CONF_GUIEX *conf, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const OUTPUT_INFO *oip, DWORD ret); //一時ファイルの最終的な移動・削除を実行
DWORD GetExePriority(DWORD set, HANDLE h_aviutl); //実行ファイルに指定すべき優先度を取得

AUO_RESULT getLogFilePath(char *log_file_path, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip);

int check_video_ouput(const CONF_GUIEX *conf, const OUTPUT_INFO *oip);
int check_muxer_to_be_used(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, const char *temp_filename, int video_output_type, BOOL audio_output);

double get_duration(const OUTPUT_INFO *oip, const PRM_ENC *pe);

int ReadLogExe(PIPE_SET *pipes, const char *exename, LOG_CACHE *log_line_cache);
void write_cached_lines(int log_level, const char *exename, LOG_CACHE *log_line_cache);

#endif //_AUO_ENCODE_H_