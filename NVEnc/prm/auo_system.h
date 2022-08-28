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

#ifndef _AUO_SYSTEM_H_
#define _AUO_SYSTEM_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "auo.h"
#include "auo_settings.h"
#include "auo_conf.h"

#if _M_IX86
#define ALIGN_PTR __declspec(align(4))
#else
#define ALIGN_PTR __declspec(align(8))
#endif

class std::mutex;

struct faw2aacbuf {
    void *buffer;
    DWORD  buf_len;
    DWORD threadid;
};

typedef struct ALIGN_PTR {
    HANDLE ALIGN_PTR he_aud_start; //InterlockedExchangeを使用するため、__declspec(align(4))が必要
    HANDLE ALIGN_PTR he_vid_start; //InterlockedExchangeを使用するため、__declspec(align(4))が必要
    HANDLE th_aud;
    std::mutex *mtx_aud;
    faw2aacbuf faw2aac[2];
    void  *buffer;
    DWORD  buf_len;
    DWORD  buf_max_size;
    int    start;
    int    get_length;
    BOOL   abort;
} AUD_PARALLEL_ENC;

typedef struct {
    AUD_PARALLEL_ENC aud_parallel;         //音声並列処理の管理
    int video_out_type;                    //出力する動画のフォーマット(拡張子により判断)
    int muxer_to_be_used;                  //使用するmuxerのインデックス
    int drop_count;                        //ドロップ数
    BOOL afs_init;                         //動画入力の準備ができているか
    HANDLE h_p_aviutl;                     //優先度取得用のAviutlのハンドル
    HANDLE h_p_videnc;                     //動画エンコーダのハンドル
    char **opened_aviutl_files;            //Aviutlの開いているファイルリスト
    int n_opened_aviutl_files;             //Aviutlの開いているファイルリストの数
    char temp_filename[MAX_PATH_LEN];      //一時ファイル名
    char muxed_vid_filename[MAX_PATH_LEN]; //mux後に退避された動画のみファイル
    int  aud_count;                        //音声ファイル数...音声エンコード段階で設定する
                                           //auo_mux.cppのenable_aud_muxの制限から31以下
    char aud_temp_dir[MAX_PATH_LEN];       //音声一時ディレクトリ
    FILE_APPENDIX append;                  //ファイル名に追加する文字列のリスト
    int delay_cut_additional_vframe;       //音声エンコード遅延解消のための追加の動画フレーム (負値なら先頭を削ることを意味する)
    int delay_cut_additional_aframe;       //音声エンコード遅延解消のための追加の音声フレーム (負値なら先頭を削ることを意味する)
} PRM_ENC;

typedef struct {
    BOOL init;
    char auo_path[MAX_PATH_LEN];    //auoのフルパス
    char aviutl_dir[MAX_PATH_LEN];  //Aviutlのディレクトリ(\無し)
    guiEx_settings *exstg;          //ini設定
} SYSTEM_DATA;

void init_SYSTEM_DATA(SYSTEM_DATA *_sys_dat);
void delete_SYSTEM_DATA(SYSTEM_DATA *_sys_dat);

#endif //_AUO_SYSTEM_H_