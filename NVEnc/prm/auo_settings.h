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

#ifndef _AUO_SETTINGS_H_
#define _AUO_SETTINGS_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <vector>
#include "auo.h"
#include "auo_mes.h"
#include "auo_util.h"

//----    デフォルト値    ---------------------------------------------------

static const BOOL   DEFAULT_LARGE_CMD_BOX         = 0;
static const BOOL   DEFAULT_AUTO_AFS_DISABLE      = 0;
static const int    DEFAULT_OUTPUT_EXT            = 0;
static const BOOL   DEFAULT_AUTO_DEL_STATS        = 1;
static const BOOL   DEFAULT_AUTO_DEL_CHAP         = 1;
static const BOOL   DEFAULT_KEEP_QP_FILE          = 0;
static const BOOL   DEFAULT_AUD_DELAY_CUT_MODE    = 0;
static const BOOL   DEFAULT_DISABLE_TOOLTIP_HELP  = 0;
static const BOOL   DEFAULT_DISABLE_VISUAL_STYLES = 0;
static const BOOL   DEFAULT_ENABLE_STG_ESC_KEY    = 0;
static const BOOL   DEFAULT_SAVE_RELATIVE_PATH    = 1;
static const BOOL   DEFAULT_CHAP_NERO_TO_UTF8     = 0;
#if ENCODER_FFMPEG
static const int    DEFAULT_AUDIO_ENCODER_EXT     = 0;
#else
static const int    DEFAULT_AUDIO_ENCODER_EXT     = 15;
#endif
static const int    DEFAULT_AUDIO_ENCODER_IN      = 1;
static const BOOL   DEFAULT_AUDIO_ENCODER_USE_IN  = (ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1 || ENCODER_VVENC) ? 0 : 1;
static const int    DEFAULT_THREAD_PTHROTTLING    = 0;
static const int    DEFAULT_AMP_RETRY_LIMIT       = 2;
static const double DEFAULT_AMP_MARGIN            = 0.05;
static const double DEFAULT_AMP_REENC_AUDIO_MULTI = 0.15;
static const BOOL   DEFAULT_AMP_KEEP_OLD_FILE     = 0;
static const BOOL   DEFAULT_RUN_BAT_MINIMIZED     = 0;
static const BOOL   DEFAULT_SET_KEYFRAME_AFS24FPS = 0;
static const BOOL   DEFAULT_AUTO_REFLIMIT_BYLEVEL = 0;
#if ENCODER_QSV
static const BOOL   DEFAULT_FORCE_BLURAY          = 0;
static const BOOL   DEFAULT_PERF_MONITOR          = 0;
static const BOOL   DEFAULT_PERF_MONITOR_PLOT     = 0;
#endif
static const double DEFAULT_AV_LENGTH_DIFF_THRESOLD = 0.05;

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

enum class AuoTheme : int {
    DefaultLight,
    DarkenWindowLight,
    DarkenWindowDark,
    CountMax
};

///ログ表示で使う色                                        R    G    B
static const int    DEFAULT_LOG_COLOR_BACKGROUND[3] =   {   0,   0,   0 };
static const int    DEFAULT_LOG_COLOR_TEXT[3][3]    = { { 198, 253, 226 },   //LOG_INFO
                                                        { 245, 218,  90 },   //LOG_WARNING
                                                        { 253,  83, 121 } }; //LOG_ERROR
static const int    DEFAULT_UI_COLOR_BASE_LIGHT[3]  =  { 240, 240, 240 };
static const int    DEFAULT_UI_COLOR_TEXT_LIGHT[3]  =  {   0,   0,   0 };
static const int    DEFAULT_UI_COLOR_BASE_DARK[3]   =  {  51,  51,  51 };
static const int    DEFAULT_UI_COLOR_TEXT_DARK[3]   =  { 255, 255, 255 };
static const int    DEFAULT_UI_COLOR_TEXT_DARK_DISABLED[3] = { 153, 153, 153 };

static const BOOL   DEFAULT_FBC_CALC_BITRATE         = 1;
static const BOOL   DEFAULT_FBC_CALC_TIME_FROM_FRAME = 0;
static const int    DEFAULT_FBC_LAST_FRAME_NUM       = 0;
static const double DEFAULT_FBC_LAST_FPS             = 29.970;
static const int    DEFAULT_FBC_LAST_TIME_IN_SEC     = 0;
static const double DEFAULT_FBC_INITIAL_SIZE         = 39.8;

static const TCHAR *DEFAULT_EXE_DIR                  = _T("exe_files");
static const TCHAR *AUO_CHECK_FILEOPEN_NAME          = _T("auo_check_fileopen.exe");

typedef struct ENC_OPTION_STR {
    const TCHAR *name; //エンコーダでのオプション名
    AuoMes mes;  //GUIでの表示用
    const wchar_t *desc; //GUIでの表示用
} ENC_OPTION_STR;

typedef struct ENC_OPTION_STR2 {
    AuoMes mes;  //GUIでの表示用
    const wchar_t *desc; //GUIでの表示用
    int value;
} ENC_OPTION_STR2;

typedef struct ENC_OPTION_STR3 {
    AuoMes mes;  //GUIでの表示用
    const wchar_t *desc; //GUIでの表示用
    int64_t value;
} ENC_OPTION_STR3;

const int FAW_INDEX_ERROR = -1;

const int AUTO_SAVE_LOG_OUTPUT_DIR = 0;
const int AUTO_SAVE_LOG_CUSTOM = 1;

enum {
    DISABLE_LOG_PIPE_INPUT = 0x01,
    DISABLE_LOG_NORMAL     = 0x02,
    DISABLE_LOG_ALL        = DISABLE_LOG_PIPE_INPUT | DISABLE_LOG_NORMAL,
};

static size_t GetPrivateProfileSectionStg(const char *section, char *buf, size_t bufSize, const char *ini_file, const DWORD codepage) {
    if (bufSize == 0) {
        return 0;
    }
    size_t len = GetPrivateProfileSectionA(section, buf, (DWORD)bufSize, ini_file);
    if (codepage == CP_THREAD_ACP) {
        return len;
    }
    const auto str_thread_acp = wstring_to_string(char_to_wstring(buf, codepage), CP_THREAD_ACP);
    strcpy_s(buf, bufSize, str_thread_acp.c_str());
    return str_thread_acp.length();
}
static size_t GetPrivateProfileSectionStgW(const char *section, TCHAR *buf, size_t bufSize, const char *ini_file, const DWORD codepage) {
    if (bufSize == 0) {
        return 0;
    }
    std::vector<char> buftmp(bufSize * 4, 0);
    size_t len = GetPrivateProfileSectionA(section, buftmp.data(), (DWORD)buftmp.size(), ini_file);
    if (codepage == CP_THREAD_ACP) {
        return len;
    }
    const auto strw = char_to_wstring(buftmp.data(), codepage);
    wcscpy_s(buf, bufSize, strw.c_str());
    return strw.length();
}
static size_t GetPrivateProfileStringStg(const char *section, const char *keyname, const char *defaultString, char *buf, size_t bufSize, const char *ini_file, const DWORD codepage) {
    if (bufSize == 0) {
        return 0;
    }
    size_t len = GetPrivateProfileStringA(section, keyname, defaultString, buf, (DWORD)bufSize, ini_file);
    if (codepage == CP_THREAD_ACP) {
        return len;
    }
    const auto str_thread_acp = wstring_to_string(char_to_wstring(buf, codepage), CP_THREAD_ACP);
    strcpy_s(buf, bufSize, str_thread_acp.c_str());
    return str_thread_acp.length();
}
static size_t GetPrivateProfileWStringStg(const char *section, const char *keyname, const char *defaultString, char *buf, size_t bufSize, const char *ini_file, const DWORD codepage) {
    if (bufSize == 0) {
        return 0;
    }
    size_t len = GetPrivateProfileStringA(section, keyname, defaultString, buf, (DWORD)bufSize, ini_file);
    if (codepage == CP_THREAD_ACP) {
        return len;
    }
    const auto wstr = char_to_wstring(buf, codepage);
    wcscpy_s((TCHAR *)buf, bufSize / sizeof(TCHAR), wstr.c_str());
    return wstr.length();
}
static size_t GetPrivateProfileTStg(const char *section, const char *keyname, const TCHAR *defaultString, TCHAR *buf, size_t bufSize, const char *ini_file, const DWORD codepage) {
    if (bufSize == 0) {
        return 0;
    }
    auto defaultStrChar = tchar_to_string(defaultString, codepage);
    std::vector<char> buftmp(bufSize * 4, 0);
    GetPrivateProfileStringA(section, keyname, defaultStrChar.c_str(), buftmp.data(), (int)buftmp.size(), ini_file);
    const auto tstr = char_to_tstring(buftmp.data(), codepage);
    _tcscpy_s(buf, bufSize, tstr.c_str());
    return tstr.length();
}

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
    char *SetPrivateProfileString(const char *section, const char *keyname, const char *defaultString, const char *ini_file, const DWORD codepage) {
        char *ptr = NULL;
        if (mp_size > 0) {
            size_t len = GetPrivateProfileStringStg(section, keyname, defaultString, mp, (DWORD)mp_size, ini_file, codepage);
            ptr = mp;
            mp += len + 1;
            mp_size -= len + 1;
        }
        return ptr;
    };
    TCHAR *SetPrivateProfileWString(const char *section, const char *keyname, const char *defaultString, const char *ini_file, const DWORD codepage) {
        TCHAR *ptr = NULL;
        if (mp_size > 0) {
            size_t len = GetPrivateProfileWStringStg(section, keyname, defaultString, mp, (DWORD)mp_size, ini_file, codepage);
            ptr = (TCHAR *)mp;
            mp += (len + 1) * sizeof(TCHAR);
            mp_size -= (len + 1) * sizeof(TCHAR);
        }
        return ptr;
    };
    TCHAR *SetPrivateProfileT(const char *section, const char *keyname, const TCHAR *defaultString, const char *ini_file, const DWORD codepage) {
        TCHAR *ptr = NULL;
        if (mp_size > 0) {
            size_t len = GetPrivateProfileTStg(section, keyname, defaultString, (TCHAR *)mp, (DWORD)mp_size / sizeof(TCHAR), ini_file, codepage);
            ptr = (TCHAR *)mp;
            mp += (len + 1) * sizeof(TCHAR);
            mp_size -= (len + 1) * sizeof(TCHAR);
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
        size_t cut_size = 0;
        if (sizeof_chr == sizeof(char)) {
            cut_size = (strlen(mp) + 1) * sizeof_chr;
        } else if (sizeof_chr == sizeof(TCHAR)) {
            cut_size = (wcslen((TCHAR*)mp) + 1) * sizeof_chr;
        }
        mp += cut_size;
        mp_size -= cut_size;
    };
};

typedef struct AUDIO_ENC_MODE {
    TCHAR *name;       //名前
    TCHAR *cmd;        //コマンドライン
    BOOL bitrate;        //ビットレート指定モード
    int bitrate_min;     //ビットレートの最小値
    int bitrate_max;     //ビットレートの最大値
    int bitrate_default; //ビットレートのデフォルト値
    int bitrate_step;    //クリックでの変化幅
    int delay;           //エンコード遅延 (音声が映像に対し遅れるsample数)
    int enc_2pass;       //2passエンコを行う
    int use_8bit;        //8bitwavを入力する
    int use_remuxer;     //remuxerが必要
    TCHAR *disp_list;  //表示名のリスト
    TCHAR *cmd_list;   //コマンドラインのリスト
} AUDIO_ENC_MODE;

typedef struct AUDIO_SETTINGS {
    BOOL is_internal;            //内蔵エンコーダかどうか
    char *keyName;               //iniファイルでのセクション名
    TCHAR *dispname;           //名前
    TCHAR *codec;              //コーデック名
    TCHAR *filename;           //拡張子付き名前
    TCHAR fullpath[MAX_PATH_LEN]; //エンコーダの場所(フルパス)
    TCHAR *aud_appendix;       //作成する音声ファイル名に追加する文字列
    TCHAR *raw_appendix;       //作成する音声ファイル名に追加する文字列 (raw出力時)
    int pipe_input;              //パイプ入力が可能
    DWORD disable_log;           //ログ表示を禁止 (DISABLE_LOG_xxx)
    BOOL  unsupported_mp4;       //mp4非対応
    BOOL  enable_rf64;           //必要に応じてRF64出力を行う
    BOOL  pcm_fp32;              //PCM(float)32bit(WAVE_FORMAT_IEEE_FLOAT)対応
    TCHAR *cmd_base;           //1st pass用コマンドライン
    TCHAR *cmd_2pass;          //2nd pass用コマンドライン
    TCHAR *cmd_raw;            //raw出力用コマンドライン
    TCHAR *cmd_help;           //ヘルプ表示用コマンドライン
    TCHAR *cmd_ver;            //バージョン表示用のコマンドライン
    int mode_count;              //エンコードモードの数
    AUDIO_ENC_MODE *mode;        //エンコードモードの設定
} AUDIO_SETTINGS;

typedef struct MUXER_CMD_EX {
    TCHAR *name;      //拡張オプションの名前
    TCHAR *cmd;       //拡張オプションのコマンドライン
    TCHAR *cmd_apple; //Apple用モードの時のコマンドライン
    TCHAR *chap_file; //チャプターファイル
} MUXER_CMD_EX;

typedef struct MUXER_SETTINGS {
    char *keyName;                  //iniファイルでのセクション名
    TCHAR *dispname;              //名前
    TCHAR *filename;              //拡張子付き名前
    TCHAR fullpath[MAX_PATH_LEN]; //エンコーダの場所(フルパス)
    TCHAR *out_ext;               //mux後ファイルの拡張子
    TCHAR *base_cmd;              //もととなるコマンドライン
    TCHAR *vid_cmd;               //映像mux用のコマンドライン
    TCHAR *aud_cmd;               //音声mux用のコマンドライン
    TCHAR *tc_cmd;                //タイムコードmux用のコマンドライン
    TCHAR *tmp_cmd;               //一時フォルダ指定用コマンドライン
    TCHAR *delay_cmd;             //音声エンコーダディレイ指定用のコマンドライン
    TCHAR *help_cmd;              //ヘルプ表示用コマンドライン
    TCHAR *ver_cmd;               //バージョン表示用のコマンドライン
    int ex_count;                   //拡張オプションの数
    MUXER_CMD_EX *ex_cmd;           //拡張オプション
    int post_mux;                   //muxerを実行したあとに別のmuxerを実行する
} MUXER_SETTINGS;

typedef struct ENC_CMD {
    ENC_OPTION_STR *name;  //各種設定用動画エンコーダのコマンドの名前(配列、最後はnull)
    TCHAR **cmd;         //各種設定用動画エンコーダのコマンド(配列、最後はnull)
} ENC_CMD;

typedef struct ENC_SETTINGS {
    TCHAR *filename;                   //動画エンコーダのファイル名
    TCHAR fullpath[MAX_PATH_LEN];      //動画エンコーダの場所(フルパス)
    TCHAR *default_cmd;                //デフォルト設定用コマンドライン
    TCHAR *default_cmd_highbit;        //highbit depthデフォルト設定用追加コマンドライン
    TCHAR *help_cmd;                   //ヘルプ表示用cmd
    int preset_count;                    //presetの数
    int tune_count;                      //tuneの数
    int profile_count;                   //profileの数
    int default_preset;                  //デフォルトpresetのインデックス
    int default_tune;                    //デフォルトtuneのインデックス
    int default_profile;                 //デフォルトprofileのインデックス
    ENC_CMD preset;                     //presetコマンドライン集
    ENC_CMD tune;                       //tuneコマンドライン集
    ENC_CMD profile;                    //profileコマンドライン集
    float *profile_vbv_multi;            //profileによるvbv倍率
} ENC_SETTINGS;

typedef struct FILENAME_REPLACE {
    TCHAR *from; //置換元文字列
    TCHAR *to;   //置換先文字列
} FILENAME_REPLACE;

typedef struct LOG_WINDOW_SETTINGS {
    BOOL minimized;                        //最小化で起動
    int  log_level;                        //ログ出力のレベル
    BOOL transparent;                      //半透明で表示
    int  transparency;                     //透過度
    BOOL auto_save_log;                    //ログ自動保存を行うかどうか
    int  auto_save_log_mode;               //ログ自動保存のモード
    TCHAR auto_save_log_path[MAX_PATH_LEN]; //ログ自動保存ファイル名
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

typedef struct BITRATE_CALC_SETTINGS {
    BOOL   calc_bitrate;          //ビットレート計算モード
    BOOL   calc_time_from_frame;  //フレーム数とフレームレートから動画時間を計算
    int    last_frame_num;        //最後に指定したフレーム数
    double last_fps;              //最後に指定したフレームレート
    DWORD  last_time_in_sec;      //最後に指定した時間
    double initial_size;          //初期サイズ
} BITRATE_CALC_SETTINGS;

typedef struct LOCAL_SETTINGS {
    BOOL   large_cmdbox;                        //拡大サイズでコマンドラインプレビューを行う
    DWORD  audio_buffer_size;                   //音声用バッファサイズ
    BOOL   auto_afs_disable;                    //自動的にafsを無効化
    int    default_output_ext;                  //デフォルトで使用する拡張子
    BOOL   auto_del_stats;                      //自動マルチパス時、ステータスファイルを自動的に削除
    BOOL   auto_del_chap;                       //チャプターファイルの自動削除
    BOOL   keep_qp_file;                        //キーフレーム検出で作成したqpファイルを削除しない
    BOOL   disable_tooltip_help;                //ポップアップヘルプを抑制する
    BOOL   disable_visual_styles;               //視覚効果をオフにする
    BOOL   enable_stg_esc_key;                  //設定画面でEscキーを有効化する
    AUO_FONT_INFO conf_font;                    //設定画面のフォント
    int    thread_pthrottling_mode;             //スレッドの電力スロットリングモード
    int    amp_retry_limit;                     //自動マルチパス試行回数制限
    double amp_bitrate_margin_multi;            //自動マルチパスで、上限ファイルサイズからビットレートを再計算するときの倍率
    double amp_reenc_audio_multi;               //自動マルチパスで、音声側を再エンコしてビットレート調整をする上限倍率
    BOOL   amp_keep_old_file;                   //自動マルチパスで、上限を超えてしまったファイルを削除しない
    BOOL   chap_nero_convert_to_utf8;           //nero形式のチャプターをUTF-8に変換する
    BOOL   default_audenc_use_in;               //デフォルトの音声エンコーダとして、内蔵エンコーダを選択する
    int    default_audio_encoder_ext;           //デフォルトの音声エンコーダ
    int    default_audio_encoder_in;            //デフォルトの音声エンコーダ
    double av_length_threshold;                 //音声と映像の長さの差の割合がこの値を超える場合、エラー・警告を表示する
    BOOL   get_relative_path;                   //相対パスで保存する
    BOOL   run_bat_minimized;                   //エンコ前後バッチ処理を最小化で実行
    BOOL   set_keyframe_as_afs_24fps;           //自動フィールドシフト使用時にも24fps化としてキーフレーム設定を強制的に行う
    BOOL   auto_ref_limit_by_level;             //参照フレーム数をLevelにより自動的に制限する
#if ENCODER_QSV
    BOOL   force_bluray;                        //VBR,CBR以外でもBluray用出力を許可する
    BOOL   perf_monitor;                        //パフォーマンスモニタリングを有効にする
#endif
    TCHAR custom_tmp_dir[MAX_PATH_LEN];        //一時フォルダ
    TCHAR custom_audio_tmp_dir[MAX_PATH_LEN];  //音声用一時フォルダ
    TCHAR custom_mp4box_tmp_dir[MAX_PATH_LEN]; //mp4box用一時フォルダ
    TCHAR stg_dir[MAX_PATH_LEN];               //プロファイル設定ファイル保存フォルダ
    TCHAR app_dir[MAX_PATH_LEN];               //実行ファイルのフォルダ
    TCHAR bat_dir[MAX_PATH_LEN];               //バッチファイルのフォルダ
} LOCAL_SETTINGS;

typedef struct FILE_APPENDIX {
    TCHAR aud[2][MAX_APPENDIX_LEN];     //音声ファイル名に追加する文字列...音声エンコード段階で設定する
    TCHAR tc[MAX_APPENDIX_LEN];         //タイムコードファイル名に追加する文字列
    TCHAR qp[MAX_APPENDIX_LEN];         //qpファイル名に追加する文字列
    TCHAR chap[MAX_APPENDIX_LEN];       //チャプターファイル名に追加する文字列
    TCHAR chap_apple[MAX_APPENDIX_LEN]; //Apple形式のチャプター名に追加する文字列
    TCHAR wav[MAX_APPENDIX_LEN];        //一時wavファイル名に追加する文字列
} FILE_APPENDIX;

class guiEx_settings {
private:
    mem_cutter s_enc_mc;
    mem_cutter fn_rep_mc;
    mem_cutter s_aud_mc;
    mem_cutter s_mux_mc;

    static BOOL  init;                        //静的確保したものが初期化
    static char  ini_section_main[256];       //メインセクション
    static TCHAR auo_path[MAX_PATH_LEN];      //自分(auo)のフルパス
    static char  ini_fileName[MAX_PATH_LEN];  //iniファイル(読み込み用)の場所
    static char  conf_fileName[MAX_PATH_LEN]; //configファイル(読み書き用)の場所
    static DWORD ini_filesize;                //iniファイル(読み込み用)のサイズ
    static TCHAR default_lang[4];             //デフォルトの言語
    static int  ini_ver;                      //iniファイルのバージョン
    static DWORD codepage_ini;                //iniファイルの文字コード
    static DWORD codepage_cnf;                //confファイルの文字コード
    static TCHAR language[MAX_PATH_LEN];       //言語設定
    TCHAR last_out_stg[MAX_PATH_LEN];          //前回出力のstgファイル(多言語対応のため、デフォルトのCONF_LAST_OUTに加え、これも探す)

    void load_enc_cmd(ENC_CMD *vidcmd, int *count, int *default_index, const char *section);  //動画エンコーダコマンドライン設定の読み込み

    void load_aud();          //音声エンコーダ関連の設定の読み込み・更新
    void load_aud(BOOL internal); //音声エンコーダ関連の設定の読み込み・更新
    void load_mux();          //muxerの設定の読み込み・更新
    void load_enc();          //動画エンコーダ関連の設定の読み込み・更新
    void load_local();        //ファイルの場所等の設定の読み込み・更新

    int get_faw_index();             //FAWのインデックスを取得する
    BOOL s_enc_refresh;             //動画エンコーダ関連設定の再ロード

    void make_default_stg_dir(TCHAR *default_stg_dir, DWORD nSize); //プロファイル設定ファイルの保存場所の作成
    BOOL check_inifile();             //iniファイルが読めるかテスト

public:
    static TCHAR blog_url[MAX_PATH_LEN];      //ブログページのurl
    int s_aud_ext_count;                 //音声エンコーダの数
    int s_aud_int_count;                 //音声エンコーダの数
    int s_mux_count;                 //muxerの数 (基本3固定)
    AUDIO_SETTINGS *s_aud_ext;       //音声エンコーダの設定
    AUDIO_SETTINGS *s_aud_int;       //音声エンコーダの設定
    MUXER_SETTINGS *s_mux;           //muxerの設定
    ENC_SETTINGS  s_enc;             //動画エンコーダ関連の設定
    LOCAL_SETTINGS s_local;          //ファイルの場所等
    std::vector<FILENAME_REPLACE> fn_rep;  //一時ファイル名置換
    LOG_WINDOW_SETTINGS s_log;       //ログウィンドウ関連の設定
    FILE_APPENDIX s_append;          //各種ファイルに追加する名前
    BITRATE_CALC_SETTINGS s_fbc;    //簡易ビットレート計算機設定

    guiEx_settings();
    guiEx_settings(BOOL disable_loading);
    guiEx_settings(BOOL disable_loading, const TCHAR *_auo_path, const char *main_section);
    ~guiEx_settings();
    void clear_all();

    BOOL get_init_success();                 //iniファイルが存在し、正しいバージョンだったか
    BOOL get_init_success(BOOL no_message);  //iniファイルが存在し、正しいバージョンだったか
    void load_encode_stg();                  //映像・音声・動画関連の設定の読み込み・更新
    void load_fn_replace();                  //一時ファイル名置換等の設定の読み込み・更新
    void load_log_win();                     //ログウィンドウ等の設定の読み込み・更新
    void load_append();                      //各種ファイルの設定の読み込み・更新
    void load_fbc();                         //簡易ビットレート計算機設定の読み込み・更新
    void load_lang();                        //言語設定をロード
    void load_last_out_stg();                //last_out_stgのロード

    void save_local();        //ファイルの場所等の設定の保存
    void save_log_win();      //ログウィンドウ等の設定の保存
    void save_fbc();          //簡易ビットレート計算機設定の保存
    void save_lang();         //言語設定の保存
    void save_last_out_stg(); //last_out_stgの保存

    void apply_fn_replace(TCHAR *target_filename, DWORD nSize);  //一時ファイル名置換の適用

    BOOL is_faw(const AUDIO_SETTINGS *aud_stg) const;
    int get_faw_index(BOOL internal) const; //FAWのインデックスを取得する

    BOOL get_reset_s_enc_referesh(); //s_encが更新されたか
    const TCHAR *get_lang() const;
    const TCHAR *get_last_out_stg() const;
    void set_and_save_lang(const TCHAR *lang);
    void get_default_lang();
    void set_last_out_stg(const TCHAR *stg);
private:
    void initialize(BOOL disable_loading);
    void initialize(BOOL disable_loading, const TCHAR *_auo_path, const char *main_section);

    void clear_enc();         //動画エンコーダ関連の設定の消去
    void clear_aud();         //音声エンコーダ関連の設定の消去
    void clear_mux();         //muxerの設定の消去
    void clear_local();       //ファイルの場所等の設定の消去
    void clear_fn_replace();  //一時ファイル名置換等の消去
    void clear_log_win();     //ログウィンドウ等の設定の消去
    void clear_append();      //各種ファイルの設定の消去
    void clear_fbc();         //簡易ビットレート計算機設定のクリア
};

struct ColorRGB {
    int r, g, b;
    ColorRGB() : r(0), g(0), b(0) {};
    ColorRGB(int r_, int g_, int b_) : r(r_), g(g_), b(b_) {};
    std::string print() const {
        char buf[32];
        sprintf_s(buf, "(%3d, %3d, %3d)", r, g, b);
        return buf;
    }
    std::string printHex() const {
        char buf[32];
        sprintf_s(buf, "#%02x%02x%02x", r, g, b);
        return buf;
    }
};

class DarkenWindowStgNamedColor {
public:
    DarkenWindowStgNamedColor(const char *name, const char * fillColor, const char * edgeColor, const char * textForeColor, const char * textBackColor) :
        name_((name) ? name : ""),
        fillColor_((fillColor) ? fillColor : ""),
        edgeColor_((edgeColor) ? edgeColor : ""),
        textForeColor_((textForeColor) ? textForeColor : ""),
        textBackColor_((textBackColor) ? textBackColor : "") {}
    ~DarkenWindowStgNamedColor() {};
    const std::string& name() const { return name_; };
    ColorRGB fillColor() const { return parseColor(fillColor_); };
    ColorRGB edgeColor() const { return parseColor(edgeColor_); };
    ColorRGB textForeColor() const { return parseColor(textForeColor_); };
    ColorRGB textBackColor() const { return parseColor(textBackColor_); };
    ColorRGB parseColor(const std::string& colorStr) const;
protected:
    std::string name_;
    std::string fillColor_;
    std::string edgeColor_;
    std::string textForeColor_;
    std::string textBackColor_;
};

enum class DarkenWindowState {
    Normal,
    Hot,
    Disabled,
    MaxCout
};

static const char * DWSTATE_NAMES[] = {
    "normal",
    "hot",
    "disabled"
};

class DarkenWindowStgReader {
public:
    DarkenWindowStgReader(const std::wstring& dir);
    ~DarkenWindowStgReader();
    int parseStg();

    std::wstring getSelectedThemeXml() const { return selectedThemeXml; }
    std::wstring getSelectedTheme() const;
    const DarkenWindowStgNamedColor *getColorStatic(const DarkenWindowState state = DarkenWindowState::Normal) const;
    const DarkenWindowStgNamedColor *getColorButton(const DarkenWindowState state = DarkenWindowState::Normal) const;
    const DarkenWindowStgNamedColor *getColorCheckBox(const DarkenWindowState state = DarkenWindowState::Normal) const;
    const DarkenWindowStgNamedColor *getColorTextBox(const DarkenWindowState state = DarkenWindowState::Normal) const;
    const DarkenWindowStgNamedColor *getColorMenu(const DarkenWindowState state = DarkenWindowState::Normal) const;
    const DarkenWindowStgNamedColor *getColorToolTip(const DarkenWindowState state = DarkenWindowState::Normal) const;
    bool isDarkTheme() const;
protected:
    const DarkenWindowStgNamedColor *getColor(const char *name) const;
    const DarkenWindowStgNamedColor *getColorFromNameAndState(const char *name, const DarkenWindowState state) const;
    int parseRootStg();
    int parseSelectedStg();
    int parseSelectedStg2();
    std::wstring rootDir;
    std::wstring selectedThemeXml;
    std::vector<std::wstring> selectedThemeXmlFileSets;
    std::vector<DarkenWindowStgNamedColor> namedColors;

    static std::string readWcharXml(const std::wstring& path);
};

DarkenWindowStgReader *createDarkenWindowStgReader(const TCHAR *aviutl_dir);
std::tuple<AuoTheme, DarkenWindowStgReader *> check_current_theme(const TCHAR *aviutl_dir);

#endif //_AUO_SETTINGS_H_
