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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <emmintrin.h>
#include <chrono>
#include <fcntl.h>
#include <io.h>
#include <thread>
#include <future>

#include "rgy_faw.h"
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
#include "auo_mes.h"

struct faw2aac_data_t {
    int id;
    TCHAR audfile[MAX_PATH_LEN];
    BOOL is_internal;
    HANDLE h_aud_namedpipe;
    HANDLE he_ov_aud_namedpipe;
    std::vector<uint8_t> outBuffer;
    std::future<int> thOut;
    bool thAbort;
    HANDLE heOutputDataPushed;
    HANDLE heOutputDataWritten;
    FILE *fp_out;
};

static size_t write_file(faw2aac_data_t *aud_dat, const PRM_ENC *pe, const void *buf, size_t size) {
    if (aud_dat->is_internal) {
        while (WaitForSingleObject(aud_dat->heOutputDataWritten, 50) != WAIT_OBJECT_0) {
            if (pe->aud_parallel.abort) {
                return 0;
            }
        }
        const auto origSize = aud_dat->outBuffer.size();
        aud_dat->outBuffer.resize(origSize + size);
        memcpy(aud_dat->outBuffer.data() + origSize, buf, size);
        SetEvent(aud_dat->heOutputDataPushed);
        return size;
    } else {
        return _fwrite_nolock(buf, 1, size, aud_dat->fp_out);
    }
}

AUO_RESULT audio_faw2aac(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;

    set_window_title(L"faw2aac", PROGRESSBAR_CONTINUOUS);
    write_log_auo_line_fmt(LOG_INFO, L"faw2aac %s", g_auo_mes.get(AUO_AUDIO_START_ENCODE));
    const int bufsize = sys_dat->exstg->s_local.audio_buffer_size;

    faw2aac_data_t aud_dat[2];
    //パイプ or ファイルオープン
    for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
        // 初期化
        aud_dat[i_aud].id = i_aud;
        memset(aud_dat[i_aud].audfile, 0, sizeof(aud_dat[i_aud].audfile));
        aud_dat[i_aud].is_internal = conf->aud.use_internal;
        aud_dat[i_aud].h_aud_namedpipe = nullptr;
        aud_dat[i_aud].he_ov_aud_namedpipe = nullptr;
        aud_dat[i_aud].thAbort = false;
        aud_dat[i_aud].heOutputDataPushed = nullptr;
        aud_dat[i_aud].heOutputDataWritten = nullptr;
        aud_dat[i_aud].fp_out = nullptr;
        if (conf->aud.use_internal) {
            TCHAR pipename[MAX_PATH_LEN];
            get_audio_pipe_name(pipename, _countof(pipename), i_aud);
            aud_dat[i_aud].h_aud_namedpipe = CreateNamedPipe(pipename, PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
            aud_dat[i_aud].he_ov_aud_namedpipe = CreateEvent(NULL, FALSE, FALSE, NULL);
            aud_dat[i_aud].heOutputDataPushed = CreateEvent(NULL, FALSE, FALSE, NULL);
            aud_dat[i_aud].heOutputDataWritten = CreateEvent(NULL, FALSE, TRUE, NULL);
        }
    }

    //確実なfcloseのために何故か一度ここで待機する必要あり
    if_valid_set_event(pe->aud_parallel.he_vid_start);
    if_valid_wait_for_single_object(pe->aud_parallel.he_aud_start, INFINITE);

    //パイプ or ファイルオープン
    if (conf->aud.use_internal) {
        auto run_transfer_pipe = [&](int audio_idx) {
            int ret = 0;
            auto aud_track = &aud_dat[audio_idx];
            //エンコーダプロセスの起動を確認
            {
                OVERLAPPED overlapped;
                memset(&overlapped, 0, sizeof(overlapped));
                overlapped.hEvent = aud_track->he_ov_aud_namedpipe;
                ConnectNamedPipe(aud_track->h_aud_namedpipe, &overlapped);
                while ((ret = WaitForSingleObject(overlapped.hEvent, 50)) != WAIT_OBJECT_0) {
                    if (pe->aud_parallel.abort) {
                        return 1;
                    }
                }
            }
            //転送を実行
            while (!pe->aud_parallel.abort) {
                if (WaitForSingleObject(aud_track->heOutputDataPushed, 50) == WAIT_OBJECT_0) {
                    if (aud_track->outBuffer.size() > 0) {
                        OVERLAPPED overlapped;
                        memset(&overlapped, 0, sizeof(overlapped));
                        overlapped.hEvent = aud_track->he_ov_aud_namedpipe;
                        DWORD sizeWritten = 0;
                        //非同期処理中は0を返すことがある
                        WriteFile(aud_track->h_aud_namedpipe, aud_track->outBuffer.data(), (DWORD)aud_track->outBuffer.size(), &sizeWritten, &overlapped);
                        while (WaitForSingleObject(overlapped.hEvent, 1000) != WAIT_OBJECT_0) {
                            if (pe->aud_parallel.abort) {
                                return 0;
                            }
                        }
                        aud_track->outBuffer.clear();
                    }
                }
                SetEvent(aud_track->heOutputDataWritten);
                if (aud_track->thAbort && aud_track->outBuffer.size() == 0) {
                    break;
                }
            }
            return pe->aud_parallel.abort ? 1 : 0;
        };
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            aud_dat[i_aud].thOut = std::async(run_transfer_pipe, i_aud);
        }
    } else {
        for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
            const CONF_AUDIO_BASE *cnf_aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;
            const AUDIO_SETTINGS *aud_stg = (conf->aud.use_internal) ? &sys_dat->exstg->s_aud_int[cnf_aud->encoder] : &sys_dat->exstg->s_aud_ext[cnf_aud->encoder];
            _tcscpy_s(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), aud_stg->aud_appendix); //pe一時パラメータにコピーしておく
            if (i_aud)
                insert_before_ext(pe->append.aud[i_aud], _countof(pe->append.aud[i_aud]), i_aud);
            get_aud_filename(aud_dat[i_aud].audfile, _countof(aud_dat[i_aud].audfile), pe, i_aud);
            if (_tfopen_s(&aud_dat[i_aud].fp_out, aud_dat[i_aud].audfile, _T("wbS")) != NULL) {
                ret |= AUO_RESULT_ABORT;
                break;
            }
        }
    }

    if (!ret) {
        const int elemsize = sizeof(short);
        const int wav_sample_size = oip->audio_ch * elemsize;

        RGYWAVHeader wavheader = { 0 };
        wavheader.file_size = 0;
        wavheader.subchunk_size = 16;
        wavheader.audio_format = 1;
        wavheader.number_of_channels = (uint16_t)oip->audio_ch;
        wavheader.sample_rate = oip->audio_rate;
        wavheader.byte_rate = oip->audio_rate * oip->audio_ch * elemsize;
        wavheader.block_align = (uint16_t)wav_sample_size;
        wavheader.bits_per_sample = elemsize * 8;
        wavheader.data_size = oip->audio_n * wavheader.number_of_channels * elemsize;

        RGYFAWDecoder fawdec;
        fawdec.init(&wavheader);

        RGYFAWDecoderOutput output;
        int samples_read = 0;
        int samples_get = bufsize;

        //wav出力ループ
        while (oip->audio_n - samples_read > 0 && samples_get) {
            //中断
            if ((pe->aud_parallel.he_aud_start) ? pe->aud_parallel.abort : oip->func_is_abort()) {
                ret |= AUO_RESULT_ABORT;
                break;
            }
            uint8_t *audio_dat = (uint8_t *)get_audio_data(oip, pe, samples_read, std::min(oip->audio_n - samples_read, bufsize), &samples_get, 1); // FAWは16bitのみ
            samples_read += samples_get;
            set_log_progress(samples_read / (double)oip->audio_n);

            fawdec.decode(output, audio_dat, samples_get * wav_sample_size);
            for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
                if (output[i_aud].size() > 0) {
                    if (write_file(&aud_dat[i_aud], pe, output[i_aud].data(), output[i_aud].size()) == 0) {
                        ret |= AUO_RESULT_ABORT;
                        break;
                    }
                }
            }
        }

        fawdec.fin(output);
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
            if (output[i_aud].size() > 0) {
                if (write_file(&aud_dat[i_aud], pe, output[i_aud].data(), output[i_aud].size()) == 0) {
                    ret |= AUO_RESULT_ABORT;
                    break;
                }
            }
            aud_dat[i_aud].thAbort = true;
        }
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
            if (conf->aud.use_internal) {
                if (aud_dat[i_aud].thOut.get() != AUO_RESULT_SUCCESS) {
                    ret = AUO_RESULT_ERROR;
                }
            }
        }

        //動画との音声との同時処理が終了
        release_audio_parallel_events(pe);

        //ファイルクローズ
        for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
            if (aud_dat[i_aud].fp_out) {
                fclose(aud_dat[i_aud].fp_out);
                aud_dat[i_aud].fp_out = nullptr;
            }
        }
    } else {
        //これをやっておかないとプラグインがフリーズしてしまう
        //動画との音声との同時処理が終了
        release_audio_parallel_events(pe);
    }

    for (int i_aud = 0; !ret && i_aud < pe->aud_count; i_aud++) {
        if (aud_dat[i_aud].he_ov_aud_namedpipe) {
            CloseHandle(aud_dat[i_aud].he_ov_aud_namedpipe);
        }
        if (aud_dat[i_aud].h_aud_namedpipe) {
            FlushFileBuffers(aud_dat->h_aud_namedpipe);
            //DisconnectNamedPipe(aud_dat->h_aud_namedpipe); //これをするとなぜかInvalid argumentというメッセージが出てしまう
            CloseHandle(aud_dat[i_aud].h_aud_namedpipe);
        }
        if (aud_dat[i_aud].heOutputDataPushed) {
            CloseHandle(aud_dat[i_aud].heOutputDataPushed);
        }
        if (aud_dat[i_aud].heOutputDataWritten) {
            CloseHandle(aud_dat[i_aud].heOutputDataWritten);
        }
    }

    set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);
    return ret;
}
