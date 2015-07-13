//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_CONF_H_
#define _AUO_CONF_H_

#include <Windows.h>
#include "auo.h"
#include "NVEncCore.h"


const int CONF_INITIALIZED = 1;

enum {
    TMP_DIR_OUTPUT = 0,
    TMP_DIR_SYSTEM = 1,
    TMP_DIR_CUSTOM = 2,
};

enum : DWORD {
    RUN_BAT_NONE           = 0x00,
    RUN_BAT_BEFORE_PROCESS = 0x01,
    RUN_BAT_AFTER_PROCESS  = 0x02,
    RUN_BAT_BEFORE_AUDIO   = 0x04,
    RUN_BAT_AFTER_AUDIO    = 0x08,
};

static inline int get_run_bat_idx(DWORD flag) {
    DWORD ret;
    _BitScanForward(&ret, flag);
    return (int)ret;
}

static const char *const CONF_NAME_OLD_1 = "NVEnc ConfigFile";
static const char *const CONF_NAME_OLD_2 = "NVEnc ConfigFile v2";
static const char *const CONF_NAME_OLD_3 = "NVEnc ConfigFile v3";
static const char *const CONF_NAME       = CONF_NAME_OLD_3;
const int CONF_NAME_BLOCK_LEN            = 32;
const int CONF_BLOCK_MAX                 = 32;
const int CONF_BLOCK_COUNT               = 6; //最大 CONF_BLOCK_MAXまで
const int CONF_HEAD_SIZE                 = (3 + CONF_BLOCK_MAX) * sizeof(int) + CONF_BLOCK_MAX * sizeof(size_t) + CONF_NAME_BLOCK_LEN;

enum {
    CONF_ERROR_NONE = 0,
    CONF_ERROR_FILE_OPEN,
    CONF_ERROR_BLOCK_SIZE,
    CONF_ERROR_INVALID_FILENAME,
};

enum {
    AUDIO_DELAY_CUT_NONE         = 0, //音声エンコード遅延の削除を行わない
    AUDIO_DELAY_CUT_DELETE_AUDIO = 1, //音声エンコード遅延の削除を音声の先頭を削除することで行う
    AUDIO_DELAY_CUT_ADD_VIDEO    = 2, //音声エンコード遅延の削除を映像を先頭に追加することで行う
    AUDIO_DELAY_CUT_EDTS         = 3, //音声エンコード遅延の削除をedtsを用いて行う
};

static const char *const AUDIO_DELAY_CUT_MODE[] = {
    "補正なし",
    "音声カット",
    "----------",
    "edts",
    NULL
};

const int CMDEX_MAX_LEN = 2048;    //追加コマンドラインの最大長

typedef struct CONF_NVENC {
    NV_ENC_CONFIG enc_config;
    NV_ENC_PIC_STRUCT pic_struct;
    int preset;
    int deviceID;
    int inputBuffer;
    int par[2];
    int codec;
    NV_ENC_CODEC_CONFIG codecConfig[2];
    int bluray;
} CONF_NVENC;

typedef struct {
    BOOL afs;                      //自動フィールドシフトの使用
    BOOL auo_tcfile_out;           //auo側でタイムコードを出力する
} CONF_VIDEO; //動画用設定

typedef struct {
    int  encoder;             //使用する音声エンコーダ
    int  enc_mode;            //使用する音声エンコーダの設定
    int  bitrate;             //ビットレート指定モード
    BOOL use_2pass;           //音声2passエンコードを行う
    BOOL use_wav;             //パイプを使用せず、wavを出力してエンコードを行う
    BOOL faw_check;           //FAWCheckを行う
    int  priority;            //音声エンコーダのCPU優先度(インデックス)
    BOOL minimized;           //音声エンコーダを最小化で実行
    int  aud_temp_dir;        //音声専用一時フォルダ
    int  audio_encode_timing; //音声を先にエンコード
    int  delay_cut;           //エンコード遅延の削除
} CONF_AUDIO; //音声用設定

typedef struct {
    BOOL disable_mp4ext;  //mp4出力時、外部muxerを使用する
    BOOL disable_mkvext;  //mkv出力時、外部muxerを使用する
    int  mp4_mode;        //mp4 外部muxer用追加コマンドの設定
    int  mkv_mode;        //mkv 外部muxer用追加コマンドの設定
    BOOL minimized;       //muxを最小化で実行
    int  priority;        //mux優先度(インデックス)
    int  mp4_temp_dir;    //mp4box用一時ディレクトリ
    BOOL apple_mode;      //Apple用モード(mp4系専用)
    BOOL disable_mpgext;  //mpg出力時、外部muxerを使用する
    int  mpg_mode;        //mpg 外部muxer用追加コマンドの設定
} CONF_MUX; //muxer用設定

typedef struct {
    //BOOL disable_guicmd;         //GUIによるコマンドライン生成を停止(CLIモード)
    int  temp_dir;               //一時ディレクトリ
    BOOL out_audio_only;         //音声のみ出力
    char notes[128];             //メモ
    DWORD run_bat;                //バッチファイルを実行するかどうか
    DWORD dont_wait_bat_fin;      //バッチファイルの処理終了待機をするかどうか
    union {
        char batfiles[4][512];        //バッチファイルのパス
        struct {
            char before_process[512]; //エンコ前バッチファイルのパス
            char after_process[512];  //エンコ後バッチファイルのパス
            char before_audio[512];   //音声エンコ前バッチファイルのパス
            char after_audio[512];    //音声エンコ後バッチファイルのパス
        } batfile;
    };
} CONF_OTHER;

typedef struct {
    char        conf_name[CONF_NAME_BLOCK_LEN];  //保存時に使用
    int         size_all;                        //保存時: CONF_VCEOUTの全サイズ / 設定中、エンコ中: CONF_INITIALIZED
    int         head_size;                       //ヘッダ部分の全サイズ
    int         block_count;                     //ヘッダ部を除いた設定のブロック数
    int         block_size[CONF_BLOCK_MAX];      //各ブロックのサイズ
    size_t      block_head_p[CONF_BLOCK_MAX];    //各ブロックのポインタ位置
    CONF_NVENC  nvenc;                           //nvencについての設定
    CONF_VIDEO  vid;                             //その他動画についての設定
    CONF_AUDIO  aud;                             //音声についての設定
    CONF_MUX    mux;                             //muxについての設定
    CONF_OTHER  oth;                             //その他の設定
} CONF_GUIEX;

class guiEx_config {
private:
    static const size_t conf_block_pointer[CONF_BLOCK_COUNT];
    static const int conf_block_data[CONF_BLOCK_COUNT];
    static void convert_nvencstg_to_nvencstgv3(CONF_GUIEX *conf, const void *dat);
    static void convert_nvencstgv2_to_nvencstgv3(CONF_GUIEX *conf);
public:
    guiEx_config();
    static void write_conf_header(CONF_GUIEX *conf);
    static int  adjust_conf_size(CONF_GUIEX *conf_buf, void *old_data, int old_size);
    int  load_guiex_conf(CONF_GUIEX *conf, const char *stg_file);       //設定をstgファイルから読み込み
    int  save_guiex_conf(const CONF_GUIEX *conf, const char *stg_file); //設定をstgファイルとして保存
};

//定義はNVEnc.cpp
void init_CONF_GUIEX(CONF_GUIEX *conf, BOOL use_10bit); //初期化し、デフォルトを設定

//出力ファイルの拡張子フィルタを作成
//filterがNULLならauoのOUTPUT_PLUGIN_TABLE用のフィルタを書き換える
void make_file_filter(char *filter, size_t nSize, int default_index);

void overwrite_aviutl_ini_file_filter(int idx);
void overwrite_aviutl_ini_name();

#endif //_AUO_CONF_H_
