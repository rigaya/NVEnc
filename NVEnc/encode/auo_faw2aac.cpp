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
#include <fcntl.h>
#include <io.h>
#include <thread>
#include <future>

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
    auto oip = g_oip;
    if (oip == nullptr) {
        throw std::exception();
    }
    short *dat = (short *)oip->func_get_audio(start, length, readed);
    audio_pass_upper8bit(dat, *readed * oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_lower8bit(int start, int length, int *readed) {
    auto oip = g_oip;
    if (oip == nullptr) {
        throw std::exception();
    }
    short *dat = (short *)oip->func_get_audio(start, length, readed);
    audio_pass_lower8bit(dat, *readed * oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_upper8bit_sse2(int start, int length, int *readed) {
    auto oip = g_oip;
    if (oip == nullptr) {
        throw std::exception();
    }
    short *dat = (short *)oip->func_get_audio(start, length, readed);
    audio_pass_upper8bit_sse2(dat, *readed * oip->audio_ch);
    return dat;
}
static void *auo_get_audio_normal_lower8bit_sse2(int start, int length, int *readed) {
    auto oip = g_oip;
    if (oip == nullptr) {
        throw std::exception();
    }
    short *dat = (short *)oip->func_get_audio(start, length, readed);
    audio_pass_lower8bit_sse2(dat, *readed * oip->audio_ch);
    return dat;
}
//音声並列処理用
static void *auo_get_audio_parallel_buf(int length) {
    const auto thid = GetCurrentThreadId();
    int thidx = -1;
    for (int i = 0; i < 2 && thidx < 0; i++) {
        if (g_pe->aud_parallel.faw2aac[i].threadid == 0) {
            g_pe->aud_parallel.faw2aac[i].threadid = thid;
        }
        if (g_pe->aud_parallel.faw2aac[i].threadid == thid) {
            thidx = i;
        }
    }
    if (thidx < 0) {
        return nullptr;
    }
    if ((int)g_pe->aud_parallel.faw2aac[thidx].buf_len < length) {
        if (g_pe->aud_parallel.faw2aac[thidx].buffer) {
            _aligned_free(g_pe->aud_parallel.faw2aac[thidx].buffer);
        }
        g_pe->aud_parallel.faw2aac[thidx].buffer = _aligned_malloc(length, 32);
        g_pe->aud_parallel.faw2aac[thidx].buf_len = length;
    }
    return g_pe->aud_parallel.faw2aac[thidx].buffer;
}
static void *auo_get_audio_parallel(int start, int length, int *readed) {
    auto pe = g_pe;
    auto oip = g_oip;
    if (oip == nullptr || pe == nullptr) {
        throw std::exception();
    }
    std::lock_guard<std::mutex> lock(*pe->aud_parallel.mtx_aud);
    short *dat = (short *)get_audio_data(oip, pe, start, length, readed);
    short *buf = (short *)auo_get_audio_parallel_buf(*readed * oip->audio_size);
    memcpy(buf, dat, *readed * oip->audio_size);
    return buf;
}
static void *auo_get_audio_parallel_upper8bit(int start, int length, int *readed) {
    auto pe = g_pe;
    auto oip = g_oip;
    if (oip == nullptr || pe == nullptr) {
        throw std::exception();
    }
    std::lock_guard<std::mutex> lock(*pe->aud_parallel.mtx_aud);
    short *dat = (short *)get_audio_data(oip, pe, start, length, readed);
    short *buf = (short *)auo_get_audio_parallel_buf(*readed * oip->audio_size);
    memcpy(buf, dat, *readed * oip->audio_size);
    audio_pass_upper8bit(buf, *readed * oip->audio_ch);
    return buf;
}
static void *auo_get_audio_parallel_lower8bit(int start, int length, int *readed) {
    auto pe = g_pe;
    auto oip = g_oip;
    if (oip == nullptr || pe == nullptr) {
        throw std::exception();
    }
    std::lock_guard<std::mutex> lock(*pe->aud_parallel.mtx_aud);
    short *dat = (short *)get_audio_data(g_oip, pe, start, length, readed);
    short *buf = (short *)auo_get_audio_parallel_buf(*readed * oip->audio_size);
    memcpy(buf, dat, *readed * oip->audio_size);
    audio_pass_lower8bit(buf, *readed * oip->audio_ch);
    return buf;
}
static void *auo_get_audio_parallel_upper8bit_sse2(int start, int length, int *readed) {
    auto pe = g_pe;
    auto oip = g_oip;
    if (oip == nullptr || pe == nullptr) {
        throw std::exception();
    }
    std::lock_guard<std::mutex> lock(*pe->aud_parallel.mtx_aud);
    short *dat = (short *)get_audio_data(g_oip, pe, start, length, readed);
    short *buf = (short *)auo_get_audio_parallel_buf(*readed * oip->audio_size);
    memcpy(buf, dat, *readed * oip->audio_size);
    audio_pass_upper8bit_sse2(buf, *readed * oip->audio_ch);
    return buf;
}
static void *auo_get_audio_parallel_lower8bit_sse2(int start, int length, int *readed) {
    auto pe = g_pe;
    auto oip = g_oip;
    if (oip == nullptr || pe == nullptr) {
        throw std::exception();
    }
    std::lock_guard<std::mutex> lock(*pe->aud_parallel.mtx_aud);
    short *dat = (short *)get_audio_data(g_oip, pe, start, length, readed);
    short *buf = (short *)auo_get_audio_parallel_buf(*readed * oip->audio_size);
    memcpy(buf, dat, *readed * oip->audio_size);
    audio_pass_lower8bit_sse2(buf, *readed * oip->audio_ch);
    return buf;
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

typedef struct {
    char name[MAX_PATH_LEN];
    HANDLE h_pipe;
    HANDLE he_ov_aud_namedpipe;
} faw2aac_named_pipeset_t;

typedef struct {
    OUTPUT_INFO oip;
    std::future<int> th_faw2aac;
    std::future<int> th_transfer;
    bool th_transfer_started;
    faw2aac_named_pipeset_t from_auo;
    faw2aac_named_pipeset_t to_exe;
    char audfile[MAX_PATH_LEN];
} faw2aac_data_t;

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
        ret = AUO_RESULT_WARNING; write_log_auo_line(LOG_INFO, "faw2aac.auoが見つかりませんでした。");
    } else if (
           NULL == (getFAW2AACTable = (func_get_auo_table)GetProcAddress(hModule, "GetOutputPluginTable"))
        || NULL == (opt = getFAW2AACTable())
        || NULL ==  opt->func_output) {
        ret = AUO_RESULT_WARNING; write_log_auo_line(LOG_WARNING, "faw2aac.auoのロードに失敗しました。");
    } else {
        //進捗表示の取り込み
        g_oip = oip;
        g_pe = pe;

        set_window_title("faw2aac", PROGRESSBAR_CONTINUOUS);
        write_log_auo_line(LOG_INFO, "faw2aac で音声エンコードを行います。");
        static const int PIPE_BUF = 4096;

        faw2aac_data_t aud_dat[2];
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            aud_dat[i_aud].oip = *oip;
            aud_dat[i_aud].from_auo.h_pipe = NULL;
            aud_dat[i_aud].to_exe.h_pipe = NULL;
            aud_dat[i_aud].th_transfer_started = false;
            pe->aud_parallel.faw2aac[i_aud].threadid = 0;
            pe->aud_parallel.faw2aac[i_aud].buffer = 0;
            pe->aud_parallel.faw2aac[i_aud].buf_len = 0;
            if (conf->aud.use_internal) {
                static const char *const FAW2AAC_NAMED_PIPE_BASE = "\\\\.\\pipe\\Aviutl%08x_AuoFAW2AACPipe%d.aac";
                sprintf_s(aud_dat[i_aud].from_auo.name, FAW2AAC_NAMED_PIPE_BASE, GetCurrentProcessId(), i_aud);
                aud_dat[i_aud].from_auo.h_pipe = CreateNamedPipeA(aud_dat[i_aud].from_auo.name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, PIPE_BUF, PIPE_BUF, 0, NULL);
                aud_dat[i_aud].oip.savefile = aud_dat[i_aud].from_auo.name;

                get_audio_pipe_name(aud_dat[i_aud].to_exe.name, _countof(aud_dat[i_aud].to_exe.name), i_aud);
                aud_dat[i_aud].to_exe.h_pipe = CreateNamedPipeA(aud_dat[i_aud].to_exe.name, PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, PIPE_BUF, PIPE_BUF, 0, NULL);
                aud_dat[i_aud].to_exe.he_ov_aud_namedpipe = CreateEvent(NULL, FALSE, FALSE, NULL);
            } else {
                //audfile名作成
                const CONF_AUDIO_BASE *cnf_aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;
                const AUDIO_SETTINGS *aud_stg = (conf->aud.use_internal) ? &sys_dat->exstg->s_aud_int[cnf_aud->encoder] : &sys_dat->exstg->s_aud_ext[cnf_aud->encoder];
                strcpy_s(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), aud_stg->aud_appendix); //pe一時パラメータにコピーしておく
                if (i_aud)
                    insert_before_ext(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), i_aud);
                get_aud_filename(aud_dat[i_aud].audfile, _countof(aud_dat[i_aud].audfile), pe, i_aud);
                aud_dat[i_aud].oip.savefile = aud_dat[i_aud].audfile;
            }
            aud_dat[i_aud].oip.func_rest_time_disp = auo_rest_time_disp;
            //並列処理制御用
            if (pe->aud_parallel.th_aud) {
                aud_dat[i_aud].oip.func_get_audio = (pe->aud_count > 1) ? FAW2AAC_AUDIO_PARALLEL[!!i_aud][!!check_sse2()] : auo_get_audio_parallel;
                aud_dat[i_aud].oip.func_is_abort = auo_get_if_abort;
                aud_dat[i_aud].oip.func_update_preview = auo_kill_update_preview;
                //通常処理用
            } else if (pe->aud_count > 1) {
                aud_dat[i_aud].oip.func_get_audio = FAW2AAC_AUDIO_NORMAL[!!i_aud][!!check_sse2()];
            }
        }
        if_valid_set_event(pe->aud_parallel.he_vid_start);
        if_valid_wait_for_single_object(pe->aud_parallel.he_aud_start, INFINITE);
        HANDLE threadStarted[2] = { NULL, NULL };
        auto run_faw2aac = [&](int audio_idx) {
            int ret = AUO_RESULT_SUCCESS;
            //開始
            if (opt->func_init && !opt->func_init()) {
                ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_WARNING, "faw2aac.auoの初期化に失敗しました。");
            } else {
                //faw2aac用の処理スレッドが処理を開始したことを通知
                if (threadStarted[audio_idx]) SetEvent(threadStarted[audio_idx]);
                if (FALSE == opt->func_output(&aud_dat[audio_idx].oip)) {
                    ret = AUO_RESULT_ERROR; write_log_auo_line(LOG_WARNING, "faw2aac.auoの実行に失敗しました。");
                }
            }
            if (opt->func_exit)
                opt->func_exit();
            return ret;
        };

        if (conf->aud.use_internal) {
            auto th_faw2aac_finished = [&]() {
                for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                    if (aud_dat[i_aud].th_faw2aac.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                        return false;
                    }
                }
                return true;
            };
            auto th_transfer_finished = [&]() {
                for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                    if (aud_dat[i_aud].th_transfer.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                        return false;
                    }
                }
                return true;
            };
            auto run_transfer_pipe = [&](int audio_idx) {
                int ret = 0;
                auto aud_track = &aud_dat[audio_idx];
                //faw2aacを実行するスレッドの起動を確認する
                while (WaitForSingleObject(threadStarted[audio_idx], 10) == WAIT_TIMEOUT) {
                    if (pe->aud_parallel.abort || aud_track->th_faw2aac.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                        return 1; //faw2aacを実行するスレッドが異常終了した場合
                    }
                }
                //少し待って様子を見る(func_outputが失敗しないかどうか)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (aud_track->th_faw2aac.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    return 1; //faw2aacを実行するスレッドが異常終了した場合
                }
                //エンコーダプロセスの起動を確認
                OVERLAPPED overlapped;
                memset(&overlapped, 0, sizeof(overlapped));
                overlapped.hEvent = aud_track->to_exe.he_ov_aud_namedpipe;
                ConnectNamedPipe(aud_track->to_exe.h_pipe, &overlapped);
                while ((ret = WaitForSingleObject(aud_track->to_exe.he_ov_aud_namedpipe, 50)) != WAIT_OBJECT_0) {
                    if (pe->aud_parallel.abort || aud_track->th_faw2aac.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                        return 1;
                    }
                }
                //転送を実行
                while (!pe->aud_parallel.abort) {
                    const bool faw2aac_finished = th_faw2aac_finished();
                    bool readFinished = true;
                    DWORD sizeRead = 0;
                    char buffer[PIPE_BUF];
                    if (ReadFile(aud_track->from_auo.h_pipe, buffer, sizeof(buffer), &sizeRead, NULL) == 0) {
                        return 1;
                    } else if (sizeRead > 0) {
                        readFinished = false;
                        DWORD sizeWritten = 0;
                        memset(&overlapped, 0, sizeof(overlapped));
                        overlapped.hEvent = aud_track->to_exe.he_ov_aud_namedpipe;
                        //非同期処理中は0を返すことがある
                        WriteFile(aud_track->to_exe.h_pipe, buffer, sizeRead, &sizeWritten, &overlapped);
                        while (WaitForSingleObject(aud_track->to_exe.he_ov_aud_namedpipe, 50) != WAIT_OBJECT_0) {
                            if (pe->aud_parallel.abort) {
                                return 1;
                            }
                        }
                    }
                    //faw2aacが終了し、かつ転送するものがなくなったら終了
                    if (faw2aac_finished && readFinished) {
                        break;
                    }
                }
                return pe->aud_parallel.abort ? 1 : 0;
            };
            //スレッドを起動
            for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
                threadStarted[i_aud] = CreateEvent(NULL, FALSE, FALSE, NULL);
                aud_dat[i_aud].th_faw2aac = std::async(run_faw2aac, i_aud);
                aud_dat[i_aud].th_transfer = std::async(run_transfer_pipe, i_aud);
                aud_dat[i_aud].th_transfer_started = true;
            }
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                if (aud_dat[i_aud].th_transfer_started) {
                    if (aud_dat[i_aud].th_transfer.get() != AUO_RESULT_SUCCESS) {
                        ret = AUO_RESULT_ERROR;
                    }
                }
            }
            //あと片付け
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                if (aud_dat[i_aud].from_auo.h_pipe) {
                    DisconnectNamedPipe(aud_dat[i_aud].from_auo.h_pipe);
                    CloseHandle(aud_dat[i_aud].from_auo.h_pipe);
                }
                if (aud_dat[i_aud].to_exe.h_pipe) {
                    DisconnectNamedPipe(aud_dat[i_aud].to_exe.h_pipe);
                    CloseHandle(aud_dat[i_aud].to_exe.h_pipe);
                }
                if (aud_dat[i_aud].to_exe.he_ov_aud_namedpipe) {
                    CloseHandle(aud_dat[i_aud].to_exe.he_ov_aud_namedpipe);
                }
            }
        } else {
            for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
                ret = run_faw2aac(i_aud);
                if (!ret)
                    ret |= audio_faw2aac_check(aud_dat[i_aud].audfile);
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
