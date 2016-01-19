// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 1999-2016 rigaya
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

#ifndef _AUO_FRM_H_
#define _AUO_FRM_H_

#include <Windows.h>
#include "auo_conf.h"
#include "auo_system.h"

const int LOG_UPDATE_INTERVAL = 50;

enum {
    LOG_MORE    = -1,
    LOG_INFO    = 0,
    LOG_WARNING = 1,
    LOG_ERROR   = 2,
};

enum {
    PROGRESSBAR_DISABLED   = 0,
    PROGRESSBAR_CONTINUOUS = 1,
    PROGRESSBAR_MARQUEE    = 2,
};

typedef struct {
    int max_line; //格納できる最大の行数
    int idx;      //現在の行数
    char **lines; //格納している一行
} LOG_CACHE;

//設定ウィンドウ
void ShowfrmConfig(CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat);

//ログウィンドウ制御
void show_log_window(const char *aviutl_dir, BOOL disable_visual_styles);
void set_window_title(const char *chr);
void set_window_title(const char *chr, int progress_mode);
void set_window_title_enc_mes(const char *chr, int total_drop, int frame_n);
void set_task_name(const char *chr);
void set_log_progress(double progress);
void set_log_title_and_progress(const char * chr, double progress);
void write_log_auo_line(int log_type_index, const char *chr, bool from_utf8 = false);
void write_log_line(int log_type_index, const char *chr, bool from_utf8 = false);
void flush_audio_log();
void enable_enc_control(BOOL *enc_pause, BOOL afs, BOOL add_progress, DWORD start_time, int _total_frame);
void disable_enc_control();
void set_prevent_log_close(BOOL prevent);
void auto_save_log_file(const char *log_filepath);
void log_process_events();
int  get_current_log_len(int current_pass);
void log_reload_settings();

int init_log_cache(LOG_CACHE *log_cache); //LOG_CACHEの初期化、linesのメモリ確保、成功->0, 失敗->1
void release_log_cache(LOG_CACHE *log_cache); //LOG_CACHEで使用しているメモリの開放

void write_log_enc_mes(char * const mes, DWORD *log_len, int total_drop, int current_frames);
void write_log_exe_mes(char *const msg, DWORD *log_len, const char *exename, LOG_CACHE *cache_line);
void write_args(const char *args);

#endif //_AUO_FRM_H_