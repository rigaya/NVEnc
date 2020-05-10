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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <process.h>
#include <mutex>
#pragma comment(lib, "winmm.lib")
#include "auo.h"
#include "auo_version.h"
#include "auo_system.h"
#include "auo_audio.h"
#include "auo_frm.h"

typedef struct {
    CONF_GUIEX *_conf;
    const OUTPUT_INFO *_oip;
    PRM_ENC *_pe;
    const SYSTEM_DATA *_sys_dat;
} AUDIO_OUTPUT_PRM;

static inline void if_valid_close_handle(HANDLE *p_hnd) {
    if (*p_hnd) {
        CloseHandle(*p_hnd);
        *p_hnd = NULL;
    }
}

//音声並列処理スレッド用関数
static unsigned __stdcall audio_output_parallel_func(void *prm) {
    AUDIO_OUTPUT_PRM *aud_prm = (AUDIO_OUTPUT_PRM *)prm;
    CONF_GUIEX *conf = aud_prm->_conf;
    const OUTPUT_INFO *oip = aud_prm->_oip;
    PRM_ENC *pe = aud_prm->_pe;
    const SYSTEM_DATA *sys_dat = aud_prm->_sys_dat;
    free(prm); //audio_output_parallel関数内で確保したものをここで解放

    //_endthreadexは明示的なCloseHandleが必要 (exit_audio_parallel_control内で実行)
    _endthreadex(audio_output(conf, oip, pe, sys_dat));
    return 0;
}

//音声並列処理を開始する
AUO_RESULT audio_output_parallel(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    //音声エンコードの必要がなければ終了
    if (!(oip->flag & OUTPUT_INFO_FLAG_AUDIO))
        return ret;
    AUDIO_OUTPUT_PRM *parameters = (AUDIO_OUTPUT_PRM *)malloc(sizeof(AUDIO_OUTPUT_PRM)); //スレッド関数(audio_output_parallel_func)内で解放
    if (parameters == NULL) return AUO_RESULT_ERROR;
    parameters->_conf = conf;
    parameters->_oip = oip;
    parameters->_pe = pe;
    parameters->_sys_dat = sys_dat;

    ZeroMemory(&pe->aud_parallel, sizeof(pe->aud_parallel));
    if        (NULL == (pe->aud_parallel.he_aud_start = CreateEvent(NULL, FALSE, FALSE, NULL))) {
        ret = AUO_RESULT_ERROR;
    } else if (NULL == (pe->aud_parallel.he_vid_start = CreateEvent(NULL, FALSE, FALSE, NULL))) {
        ret = AUO_RESULT_ERROR;
    } else if (NULL == (pe->aud_parallel.th_aud = (HANDLE)_beginthreadex(NULL, 0, audio_output_parallel_func, (void *)parameters, 0, NULL))) {
        ret = AUO_RESULT_ERROR;
    }
    pe->aud_parallel.mtx_aud = new std::mutex();

    if (ret == AUO_RESULT_ERROR) {
        if_valid_close_handle(&(pe->aud_parallel.he_aud_start));
        if_valid_close_handle(&(pe->aud_parallel.he_vid_start));
    }
    return ret;
}

//並列処理制御用のイベントをすべて解放する
//映像・音声どちらかのAviutlからのデータ取得が必要なくなった時点で呼ぶ
//呼び出しは映像・音声スレッドどちらでもよい
//この関数が呼ばれたあとは、映像・音声どちらも自由に動くようにする
void release_audio_parallel_events(PRM_ENC *pe) {
    if (pe->aud_parallel.he_aud_start) {
        //この関数が同時に呼ばれた場合のことを考え、InterlockedExchangePointerを使用してHANDLEを処理する
        HANDLE he_aud_start_copy = InterlockedExchangePointer(&(pe->aud_parallel.he_aud_start), NULL);
        SetEvent(he_aud_start_copy); //もし止まっていたら動かしてやる
        CloseHandle(he_aud_start_copy);
    }
    if (pe->aud_parallel.he_vid_start) {
        //この関数が同時に呼ばれた場合のことを考え、InterlockedExchangePointerを使用してHANDLEを処理する
        HANDLE he_vid_start_copy = InterlockedExchangePointer(&(pe->aud_parallel.he_vid_start), NULL);
        SetEvent(he_vid_start_copy); //もし止まっていたら動かしてやる
        CloseHandle(he_vid_start_copy);
    }
    std::mutex *mtx_aud = (std::mutex *)InterlockedExchangePointer((void **)&pe->aud_parallel.mtx_aud, nullptr);
    if (mtx_aud) {
        delete mtx_aud;
    }
}
