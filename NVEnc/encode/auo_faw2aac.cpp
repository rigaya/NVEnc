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
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <emmintrin.h>

#include "output.h"
#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_frm.h"
#include "auo_error.h"
#include "auo_encode.h"
#include "auo_audio.h"
#include "auo_audio_parallel.h"
#include "auo_faw2aac.h"

typedef OUTPUT_PLUGIN_TABLE* (*func_get_auo_table)(void);

typedef void *(*auo_func_get_audio)( int start,int length,int *readed );

BOOL check_if_faw2aac_exists() {
    char aviutl_dir[MAX_PATH_LEN];
    get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));

    for (int i = 0; i < _countof(FAW2AAC_NAME); i++) {
        char faw2aac_path[MAX_PATH_LEN];
        PathCombineLong(faw2aac_path, _countof(faw2aac_path), aviutl_dir, FAW2AAC_NAME[i]);
        if (PathFileExists(faw2aac_path))
            return TRUE;
    }
    return FALSE;
}

static const OUTPUT_INFO *g_oip;
static PRM_ENC *g_pe;
static BOOL auo_rest_time_disp(int now, int total) {
    if (!g_pe || !g_pe->aud_parallel.th_aud) { //並列処理時には進捗表示をスキップ
        if (g_oip)
            g_oip->func_rest_time_disp(now, total);
        //進捗表示
        static DWORD tm_last = timeGetTime();
        DWORD tm;
        if ((tm = timeGetTime()) - tm_last > LOG_UPDATE_INTERVAL * 5) {
            set_log_progress(now / (double)total);
            tm_last = tm;
        }
    }
    return TRUE;
};

static void __forceinline audio_pass_upper8bit(short *data, int length) {
    for (int i = 0; i < length; i++)
        data[i] &= 0xff00;
}
static void __forceinline audio_pass_lower8bit(short *data, int length) {
    for (int i = 0; i < length; i++)
        data[i] <<= 8;
}
static void __forceinline audio_pass_upper8bit_sse2(short *data, int length) {
    short *data_fin = (short *)(((size_t)data + 15) & ~15);
    length -= (data_fin - data);
    for ( ; data < data_fin; data++)
        *data &= 0xff00;
    //メインループ
    data_fin = data + (length & ~15);
    __m128i x0, x1;
    __m128i xMask = _mm_slli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8); //0xff00
    for ( ; data < data_fin; data += 16) {
        x0 = _mm_load_si128((__m128i*)(data + 0));
        x1 = _mm_load_si128((__m128i*)(data + 8));
        x0 = _mm_and_si128(x0, xMask);
        x1 = _mm_and_si128(x1, xMask);
        _mm_store_si128((__m128i*)(data + 0), x0);
        _mm_store_si128((__m128i*)(data + 8), x1);
    }
    data = data_fin + (length & 15) - 16;
    x0 = _mm_loadu_si128((__m128i*)(data + 0));
    x1 = _mm_loadu_si128((__m128i*)(data + 8));
    x0 = _mm_and_si128(x0, xMask);
    x1 = _mm_and_si128(x1, xMask);
    _mm_storeu_si128((__m128i*)(data + 0), x0);
    _mm_storeu_si128((__m128i*)(data + 8), x1);
}
static void __forceinline audio_pass_lower8bit_sse2(short *data, int length) {
    short *data_fin = (short *)(((size_t)data + 15) & ~15);
    length -= (data_fin - data);
    for ( ; data < data_fin; data++)
        *data <<= 8;
    //メインループ
    data_fin = data + (length & ~15);
    __m128i x0, x1;
    for ( ; data < data_fin; data += 16) {
        x0 = _mm_load_si128((__m128i*)(data + 0));
        x1 = _mm_load_si128((__m128i*)(data + 8));
        x0 = _mm_slli_epi16(x0, 8);
        x1 = _mm_slli_epi16(x1, 8);
        _mm_store_si128((__m128i*)(data + 0), x0);
        _mm_store_si128((__m128i*)(data + 8), x1);
    }
    data_fin += (length & 15);
    for ( ; data < data_fin; data++)
        *data <<= 8;
}

//音声通常処理用
static void *auo_get_audio_normal_upper8bit(int start, int length, int *readed) {
    short *dat = (short *)g_oip->func_get_audio(start, length, readed);
    audio_pass_upper8bit(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_lower8bit(int start, int length, int *readed) {
    short *dat = (short *)g_oip->func_get_audio(start, length, readed);
    audio_pass_lower8bit(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_upper8bit_sse2(int start, int length, int *readed) {
    short *dat = (short *)g_oip->func_get_audio(start, length, readed);
    audio_pass_upper8bit_sse2(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_lower8bit_sse2(int start, int length, int *readed) {
    short *dat = (short *)g_oip->func_get_audio(start, length, readed);
    audio_pass_lower8bit_sse2(dat, *readed * g_oip->audio_ch);
    return dat;
}
//音声並列処理用
static void *auo_get_audio_parallel(int start, int length, int *readed) {
    return get_audio_data(g_oip, g_pe, start, length, readed);
}
static void *auo_get_audio_parallel_upper8bit(int start, int length, int *readed) {
    short *dat = (short *)get_audio_data(g_oip, g_pe, start, length, readed);
    audio_pass_upper8bit(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_parallel_lower8bit(int start, int length, int *readed) {
    short *dat = (short *)get_audio_data(g_oip, g_pe, start, length, readed);
    audio_pass_lower8bit(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_parallel_upper8bit_sse2(int start, int length, int *readed) {
    short *dat = (short *)get_audio_data(g_oip, g_pe, start, length, readed);
    audio_pass_upper8bit_sse2(dat, *readed * g_oip->audio_ch);
    return dat;
}
static void *auo_get_audio_parallel_lower8bit_sse2(int start, int length, int *readed) {
    short *dat = (short *)get_audio_data(g_oip, g_pe, start, length, readed);
    audio_pass_lower8bit_sse2(dat, *readed * g_oip->audio_ch);
    return dat;
}
static const auo_func_get_audio FAW2AAC_AUDIO_NORMAL[][2] = {
    { auo_get_audio_normal_upper8bit, auo_get_audio_normal_upper8bit_sse2 },
    { auo_get_audio_normal_lower8bit, auo_get_audio_normal_lower8bit_sse2 },
};
static const auo_func_get_audio FAW2AAC_AUDIO_PARALLEL[][2] = {
    { auo_get_audio_parallel_upper8bit, auo_get_audio_parallel_upper8bit_sse2 },
    { auo_get_audio_parallel_lower8bit, auo_get_audio_parallel_lower8bit_sse2 },
};


static BOOL auo_get_if_abort() {
    return (g_pe) ? g_pe->aud_parallel.abort : FALSE;
}
static int auo_kill_update_preview() {
    return TRUE;
}

static AUO_RESULT audio_faw2aac_check(const char *audfile) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    UINT64 audfilesize = 0;
    if (!PathFileExists(audfile) ||
        (GetFileSizeUInt64(audfile, &audfilesize) && audfilesize == 0)) {
            //エラーが発生した場合
        ret |= AUO_RESULT_ERROR; error_audenc_failed("faw2aac.auo", NULL);
    }
    return ret;
}

AUO_RESULT audio_faw2aac(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    HMODULE hModule = NULL;
    func_get_auo_table getFAW2AACTable = NULL;
    OUTPUT_PLUGIN_TABLE *opt = NULL;
    char aviutl_dir[MAX_PATH_LEN];
    get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));

    for (int i = 0; i < _countof(FAW2AAC_NAME); i++) {
        char faw2aac_path[MAX_PATH_LEN];
        PathCombineLong(faw2aac_path, _countof(faw2aac_path), aviutl_dir, FAW2AAC_NAME[i]);
        if (PathFileExists(faw2aac_path)) {
            hModule = LoadLibrary(faw2aac_path);
            break;
        }
    }

    if (hModule == NULL) {
        ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_INFO, "faw2aac.auoが見つかりませんでした。");
    } else if (
           NULL == (getFAW2AACTable = (func_get_auo_table)GetProcAddress(hModule, "GetOutputPluginTable"))
        || NULL == (opt = getFAW2AACTable())
        || NULL ==  opt->func_output) {
        ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_WARNING, "faw2aac.auoのロードに失敗しました。");
    } else {
        OUTPUT_INFO oip_faw2aac = *oip;
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            //audfile名作成
            char audfile[MAX_PATH_LEN];
            const AUDIO_SETTINGS *aud_stg = &sys_dat->exstg->s_aud[conf->aud.encoder];
            strcpy_s(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), aud_stg->aud_appendix); //pe一時パラメータにコピーしておく
            if (i_aud)
                insert_before_ext(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), i_aud);
            get_aud_filename(audfile, _countof(audfile), pe, i_aud);
            oip_faw2aac.savefile = audfile;
            //進捗表示の取り込み
            g_oip = oip;
            g_pe = pe;
            oip_faw2aac.func_rest_time_disp = auo_rest_time_disp;
            //並列処理制御用
            if (pe->aud_parallel.th_aud) {
                oip_faw2aac.func_get_audio = (pe->aud_count > 1) ? FAW2AAC_AUDIO_PARALLEL[!!i_aud][!!check_sse2()] : auo_get_audio_parallel;
                oip_faw2aac.func_is_abort = auo_get_if_abort;
                oip_faw2aac.func_update_preview = auo_kill_update_preview;
            //通常処理用
            } else if (pe->aud_count > 1)
                oip_faw2aac.func_get_audio = FAW2AAC_AUDIO_NORMAL[!!i_aud][!!check_sse2()];

            //開始
            if (opt->func_init && !opt->func_init()) {
                ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_WARNING, "faw2aac.auoの初期化に失敗しました。");
            } else {
                set_window_title("faw2aac", PROGRESSBAR_CONTINUOUS);
                write_log_auo_line(LOG_INFO, "faw2aac で音声エンコードを行います。");
                if (FALSE == opt->func_output(&oip_faw2aac)) {
                    ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_WARNING, "faw2aac.auoの実行に失敗しました。");
                }
                if (opt->func_exit)
                    opt->func_exit();
                if (!ret)
                    ret |= audio_faw2aac_check(audfile);
            }
        }
        release_audio_parallel_events(pe);
        g_oip = NULL;
        g_pe = NULL;
    }

    if (hModule)
        FreeLibrary(hModule);

    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

    return ret;
}
