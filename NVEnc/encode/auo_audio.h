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

#ifndef _AUO_AUDIO_H_
#define _AUO_AUDIO_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "auo.h"
#include "auo_conf.h"
#include "auo_system.h"

int get_audio_size(const OUTPUT_INFO *oip, const int audio_format);
void *oip_func_get_audio(const OUTPUT_INFO *oip, int start, int length, int* readed, int audio_format);
void *get_audio_data(const OUTPUT_INFO *oip, PRM_ENC *pe, int start, int length, int *readed);

void auo_faw_check(CONF_AUDIO *aud, const OUTPUT_INFO *oip, PRM_ENC *pe, const guiEx_settings *ex_stg);
int check_audio_length(OUTPUT_INFO *oip, double av_length_threshold);

AUO_RESULT audio_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat); //音声処理を実行
AUO_RESULT audio_output_parallel(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat);

BOOL check_audenc_output(const AUDIO_SETTINGS *aud_stg, std::wstring& exe_message);

#endif //_AUO_AUDIO_H_