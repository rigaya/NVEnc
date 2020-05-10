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

#ifndef _AUO_SETTINGS_H_
#define _AUO_SETTINGS_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <vector>
#include "auo.h"

//----    デフォルト値    ---------------------------------------------------

static const BOOL   DEFAULT_LARGE_CMD_BOX         = 0;
static const BOOL   DEFAULT_AUTO_AFS_DISABLE      = 0;
static const int    DEFAULT_OUTPUT_EXT            = 0;
static const BOOL   DEFAULT_AUTO_DEL_CHAP         = 1;
static const BOOL   DEFAULT_DISABLE_TOOLTIP_HELP  = 0;
static const BOOL   DEFAULT_DISABLE_VISUAL_STYLES = 0;
static const BOOL   DEFAULT_ENABLE_STG_ESC_KEY    = 0;
static const BOOL   DEFAULT_SAVE_RELATIVE_PATH    = 0;
static const BOOL   DEFAULT_CHAP_NERO_TO_UTF8     = 0;
static const BOOL   DEFAULT_AUDIO_ENCODER_EXT     = 15;
static const BOOL   DEFAULT_AUDIO_ENCODER_IN      = 1;
static const BOOL   DEFAULT_AUDIO_ENCODER_USE_IN  = 1;
static const BOOL   DEFAULT_THREAD_TUNING         = 0;

static const BOOL   DEFAULT_RUN_BAT_MINIMIZED     = 0;

static const int    DEFAULT_LOG_LEVEL            = 0;
static const BOOL   DEFAULT_LOG_WINE_COMPAT      = 0;
static const BOOL   DEFAULT_LOG_START_MINIMIZED  = 0;
static const BOOL   DEFAULT_LOG_TRANSPARENT      = 1;
static const BOOL   DEFAULT_LOG_AUTO_SAVE        = 0;
static const int    DEFAULT_LOG_AUTO_SAVE_MODE   = 0;
static const BOOL   DEFAULT_LOG_SHOW_STATUS_BAR  = 1;
static const BOOL   DEFAULT_LOG_TASKBAR_PROGRESS = 1;
static const BOOL   DEFAULT_LOG_SAVE_SIZE        = 0;
static const int    DEFAULT_LOG_WIDTH            = 0;
static const int    DEFAULT_LOG_HEIGHT           = 0;
static const int    DEFAULT_LOG_TRANSPARENCY     = 28;
static const int    DEFAULT_LOG_POS[2]           = { 100, 100 };

///ログ表示で使う色                                        R    G    B
static const int    DEFAULT_LOG_COLOR_BACKGROUND[3] =   {   0,   0,   0 };
static const int    DEFAULT_LOG_COLOR_TEXT[3][3]    = { { 198, 253, 226 },   //LOG_INFO
                                                        { 245, 218,  90 },   //LOG_WARNING
                                                        { 253,  83, 121 } }; //LOG_ERROR

static const BOOL   DEFAULT_FBC_CALC_BITRATE         = 1;
static const BOOL   DEFAULT_FBC_CALC_TIME_FROM_FRAME = 0;
static const int    DEFAULT_FBC_LAST_FRAME_NUM       = 0;
static const double DEFAULT_FBC_LAST_FPS             = 29.970;
static const int    DEFAULT_FBC_LAST_TIME_IN_SEC     = 0;
static const double DEFAULT_FBC_INITIAL_SIZE         = 39.8;

typedef struct {
    char *name; //x264でのオプション名
    WCHAR *desc; //GUIでの表示用
} X264_OPTION_STR;

const int FAW_INDEX_ERROR = -1;

const int AUTO_SAVE_LOG_OUTPUT_DIR = 0;
const int AUTO_SAVE_LOG_CUSTOM = 1;

enum {
    DISABLE_LOG_PIPE_INPUT = 0x01,
    DISABLE_LOG_NORMAL     = 0x02,
    DISABLE_LOG_ALL        = DISABLE_LOG_PIPE_INPUT | DISABLE_LOG_NORMAL,
};

//メモリーを切り刻みます。
class mem_cutter {
private:
    char *init_ptr;
    char *mp;
    size_t mp_init_size;
    size_t mp_size;
public:
    mem_cutter() {
        mp = NULL;
        init_ptr = NULL;
        mp_init_size = 0;
        mp_size = mp_init_size;
    };
    ~mem_cutter() {
        clear();
    };
    void init(size_t size) {
        clear();
        mp_init_size = size;
        mp_size = mp_init_size;
        init_ptr = (char*)calloc(mp_init_size, 1);
        mp = init_ptr;
    };
    void clear() {
        if (init_ptr) free(init_ptr); init_ptr = NULL;
        mp = NULL;
        mp_size = 0;
    };
    void *CutMem(size_t size) {
        if (mp_size - size < 0)
            return NULL;
        void *ptr = mp;
        mp += size;
        mp_size -= size;
        return ptr;
    };
    char *SetPrivateProfileString(const char *section, const char *keyname, const char *defaultString, const char *ini_file) {
        char *ptr = NULL;
        if (mp_size > 0) {
            size_t len = GetPrivateProfileString(section, keyname, defaultString, mp, (DWORD)mp_size, ini_file);
            ptr = mp;
            mp += len + 1;
            mp_size -= len + 1;
        }
        return ptr;
    };
    void *GetPtr() {
        return mp;
    };
    size_t GetRemain() {
        return mp_size;
    };
    void CutString(int sizeof_chr) {
        size_t cut_size = (strlen(mp) + 1) * sizeof_chr;
        mp += cut_size;
        mp_size -= cut_size;
    };
};

typedef struct {
    char *name;          //名前
    char *cmd;           //コマンドライン
    BOOL bitrate;        //ビットレート指定モード
    int bitrate_min;     //ビットレートの最小値
    int bitrate_max;     //ビットレートの最大値
    int bitrate_default; //ビットレートのデフォルト値
    int bitrate_step;    //クリックでの変化幅
    int delay;           //エンコード遅延 (音声が映像に対し遅れるsample数)
    int enc_2pass;       //2passエンコを行う
    int use_8bit;        //8bitwavを入力する
    int use_remuxer;     //remuxerが必要
    char *disp_list;     //表示名のリスト
    char *cmd_list;      //コマンドラインのリスト
} AUDIO_ENC_MODE;

typedef struct {
    BOOL is_internal;            //内蔵エンコーダかどうか
    int  auolink_only;           //auo link時のみ有効
    char *keyName;               //iniファイルでのセクション名
    char *dispname;              //名前
    char *codec;                 //コーデック名
    char *filename;              //拡張子付き名前
    char fullpath[MAX_PATH_LEN]; //エンコーダの場所(フルパス)
    char *aud_appendix;          //作成する音声ファイル名に追加する文字列
    char *raw_appendix;          //作成する音声ファイル名に追加する文字列 (raw出力時)
    int pipe_input;              //パイプ入力が可能
    DWORD disable_log;           //ログ表示を禁止 (DISABLE_LOG_xxx)
    char *cmd_base;              //1st pass用コマンドライン
    char *cmd_2pass;             //2nd pass用コマンドライン
    char *cmd_raw;               //raw出力用コマンドライン
    char *cmd_help;              //ヘルプ表示用コマンドライン
    char *cmd_ver;               //バージョン表示用のコマンドライン
    int mode_count;              //エンコードモードの数
    AUDIO_ENC_MODE *mode;        //エンコードモードの設定
} AUDIO_SETTINGS;

typedef struct {
    char *name;      //拡張オプションの名前
    char *cmd;       //拡張オプションのコマンドライン
    char *cmd_apple; //Apple用モードの時のコマンドライン
    char *chap_file; //チャプターファイル
} MUXER_CMD_EX;

typedef struct {
    char *keyName;                //iniファイルでのセクション名
    char *dispname;               //名前
    char *filename;               //拡張子付き名前
    char fullpath[MAX_PATH_LEN];  //エンコーダの場所(フルパス)
    char *out_ext;                //mux後ファイルの拡張子
    char *base_cmd;               //もととなるコマンドライン
    char *vid_cmd;                //映像mux用のコマンドライン
    char *aud_cmd;                //音声mux用のコマンドライン
    char *tc_cmd;                 //タイムコードmux用のコマンドライン
    char *delay_cmd;              //音声エンコーダディレイ指定用のコマンドライン
    char *tmp_cmd;                //一時フォルダ指定用コマンドライン
    char *help_cmd;               //ヘルプ表示用コマンドライン
    char *ver_cmd;                //バージョン表示用のコマンドライン
    int ex_count;                 //拡張オプションの数
    MUXER_CMD_EX *ex_cmd;         //拡張オプション
    int post_mux;                 //muxerを実行したあとに別のmuxerを実行する
} MUXER_SETTINGS;

typedef struct {
    char *filename;                      //動画エンコーダのファイル名
    char fullpath[MAX_PATH_LEN];         //動画エンコーダの場所(フルパス)
    char *default_cmd;                   //デフォルト設定用コマンドライン
    char *help_cmd;                      //ヘルプ表示用cmd
} VIDEO_SETTINGS;

typedef struct {
    char *from; //置換元文字列
    char *to;   //置換先文字列
} FILENAME_REPLACE;

typedef struct {
    BOOL minimized;                        //最小化で起動
    BOOL wine_compat;                      //wine互換モード
    int  log_level;                        //ログ出力のレベル
    BOOL transparent;                      //半透明で表示
    int  transparency;                     //透過度
    BOOL auto_save_log;                    //ログ自動保存を行うかどうか
    int  auto_save_log_mode;               //ログ自動保存のモード
    char auto_save_log_path[MAX_PATH_LEN]; //ログ自動保存ファイル名
    BOOL show_status_bar;                  //ステータスバーの表示
    BOOL taskbar_progress;                 //タスクバーに進捗を表示
    BOOL save_log_size;                    //ログの大きさを保存する
    int  log_width;                        //ログ幅
    int  log_height;                       //ログ高さ
    int  log_pos[2];                       //ログ位置
    int  log_color_background[3];          //ログ背景色
    int  log_color_text[3][3];             //ログ文字色
    AUO_FONT_INFO log_font;                //ログフォント
} LOG_WINDOW_SETTINGS;

typedef struct {
    BOOL   calc_bitrate;          //ビットレート計算モード
    BOOL   calc_time_from_frame;  //フレーム数とフレームレートから動画時間を計算
    int    last_frame_num;        //最後に指定したフレーム数
    double last_fps;              //最後に指定したフレームレート
    DWORD  last_time_in_sec;      //最後に指定した時間
    double initial_size;          //初期サイズ
} BITRATE_CALC_SETTINGS;

typedef struct {
    BOOL   large_cmdbox;                        //拡大サイズでコマンドラインプレビューを行う
    DWORD  audio_buffer_size;                   //音声用バッファサイズ
    BOOL   auto_afs_disable;                    //自動的にafsを無効化
    int    default_output_ext;                  //デフォルトで使用する拡張子
    BOOL   auto_del_chap;                       //チャプターファイルの自動削除
    BOOL   disable_tooltip_help;                //ポップアップヘルプを抑制する
    BOOL   disable_visual_styles;               //視覚効果をオフにする
    BOOL   enable_stg_esc_key;                  //設定画面でEscキーを有効化する
    AUO_FONT_INFO conf_font;                    //設定画面のフォント
    BOOL   chap_nero_convert_to_utf8;           //nero形式のチャプターをUTF-8に変換する
    BOOL   default_audenc_use_in;               //デフォルトの音声エンコーダとして、内蔵エンコーダを選択する
    int    default_audio_encoder_ext;           //デフォルトの音声エンコーダ
    int    default_audio_encoder_in;            //デフォルトの音声エンコーダ
    BOOL   get_relative_path;                   //相対パスで保存する
    BOOL   thread_tuning;                       //スレッドチューニング

    BOOL   run_bat_minimized;                   //エンコ前後バッチ処理を最小化で実行
    char   custom_tmp_dir[MAX_PATH_LEN];        //一時フォルダ
    char   custom_audio_tmp_dir[MAX_PATH_LEN];  //音声用一時フォルダ
    char   custom_mp4box_tmp_dir[MAX_PATH_LEN]; //mp4box用一時フォルダ
    char   stg_dir[MAX_PATH_LEN];               //プロファイル設定ファイル保存フォルダ
    char   app_dir[MAX_PATH_LEN];               //実行ファイルのフォルダ
    char   bat_dir[MAX_PATH_LEN];               //バッチファイルのフォルダ
} LOCAL_SETTINGS;

typedef struct {
    char aud[2][MAX_APPENDIX_LEN];     //音声ファイル名に追加する文字列...音声エンコード段階で設定する
    char tc[MAX_APPENDIX_LEN];         //タイムコードファイル名に追加する文字列
    char qp[MAX_APPENDIX_LEN];         //qpファイル名に追加する文字列
    char chap[MAX_APPENDIX_LEN];       //チャプターファイル名に追加する文字列
    char chap_apple[MAX_APPENDIX_LEN]; //Apple形式のチャプター名に追加する文字列
    char wav[MAX_APPENDIX_LEN];        //一時wavファイル名に追加する文字列
} FILE_APPENDIX;

class guiEx_settings {
private:
    mem_cutter fn_rep_mc;
    mem_cutter s_vid_mc;
    mem_cutter s_aud_mc;
    mem_cutter s_mux_mc;

    static BOOL  init;                        //静的確保したものが初期化
    static char  ini_section_main[256];       //メインセクション
    static char  auo_path[MAX_PATH_LEN];      //自分(auo)のフルパス
    static char  ini_fileName[MAX_PATH_LEN];  //iniファイル(読み込み用)の場所
    static char  conf_fileName[MAX_PATH_LEN]; //configファイル(読み書き用)の場所
    static DWORD ini_filesize;                //iniファイル(読み込み用)のサイズ

    void load_vid();          //動画エンコーダ関連の設定の読み込み・更新
    void load_aud();          //音声エンコーダ関連の設定の読み込み・更新
    void load_aud(BOOL internal); //音声エンコーダ関連の設定の読み込み・更新
    void load_mux();          //muxerの設定の読み込み・更新
    void load_local();        //ファイルの場所等の設定の読み込み・更新
    BOOL s_vid_refresh;              //動画設定の再ロード

    void make_default_stg_dir(char *default_stg_dir, DWORD nSize); //プロファイル設定ファイルの保存場所の作成
    BOOL check_inifile();            //iniファイルが読めるかテスト

public:
    static char blog_url[MAX_PATH_LEN];      //ブログページのurl
    int s_aud_ext_count;                 //音声エンコーダの数
    int s_aud_int_count;                 //音声エンコーダの数
    int s_mux_count;                 //muxerの数 (基本3固定)
    VIDEO_SETTINGS s_vid;            //動画エンコーダの設定
    AUDIO_SETTINGS *s_aud_ext;       //音声エンコーダの設定
    AUDIO_SETTINGS *s_aud_int;       //音声エンコーダの設定
    MUXER_SETTINGS *s_mux;           //muxerの設定
    LOCAL_SETTINGS s_local;          //ファイルの場所等
    std::vector<FILENAME_REPLACE> fn_rep;  //一時ファイル名置換
    LOG_WINDOW_SETTINGS s_log;       //ログウィンドウ関連の設定
    FILE_APPENDIX s_append;          //各種ファイルに追加する名前
    BITRATE_CALC_SETTINGS s_fbc;    //簡易ビットレート計算機設定

    guiEx_settings();
    guiEx_settings(BOOL disable_loading);
    guiEx_settings(BOOL disable_loading, const char *_auo_path, const char *main_section);
    ~guiEx_settings();

    BOOL get_init_success();                 //iniファイルが存在し、正しいバージョンだったか
    BOOL get_init_success(BOOL no_message);  //iniファイルが存在し、正しいバージョンだったか
    void load_encode_stg();                  //映像・音声・動画関連の設定の読み込み・更新
    void load_fn_replace();                  //一時ファイル名置換等の設定の読み込み・更新
    void load_log_win();                     //ログウィンドウ等の設定の読み込み・更新
    void load_append();                      //各種ファイルの設定の読み込み・更新
    void load_fbc();                         //簡易ビットレート計算機設定の読み込み・更新

    void save_local();        //ファイルの場所等の設定の保存
    void save_log_win();      //ログウィンドウ等の設定の保存
    void save_fbc();          //簡易ビットレート計算機設定の保存

    void apply_fn_replace(char *target_filename, DWORD nSize);  //一時ファイル名置換の適用

    BOOL is_faw(const AUDIO_SETTINGS *aud_stg) const;
    int get_faw_index(BOOL internal) const; //FAWのインデックスを取得する

private:
    void initialize(BOOL disable_loading);
    void initialize(BOOL disable_loading, const char *_auo_path, const char *main_section);

    void clear_vid();         //動画エンコーダ関連の設定の消去
    void clear_aud();         //音声エンコーダ関連の設定の消去
    void clear_mux();         //muxerの設定の消去
    void clear_local();       //ファイルの場所等の設定の消去
    void clear_fn_replace();  //一時ファイル名置換等の消去
    void clear_log_win();     //ログウィンドウ等の設定の消去
    void clear_append();      //各種ファイルの設定の消去
    void clear_fbc();         //簡易ビットレート計算機設定のクリア
};

#endif //_AUO_SETTINGS_H_
