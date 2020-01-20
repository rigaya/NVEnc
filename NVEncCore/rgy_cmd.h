// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#pragma once
#ifndef __RGY_CMD_H__
#define __RGY_CMD_H__

#include "rgy_prm.h"

struct ParseCmdError {
    tstring strAppName;
    tstring strErrorMessage;
    tstring strOptionName;
    tstring strErrorValue;
};

struct sArgsData {
#if ENCODER_QSV
    int outputDepth = 8;
#endif
    tstring cachedlevel, cachedprofile;
#if !ENCODER_NVENC
    tstring cachedtier;
#endif
    uint32_t nParsedAudioFile = 0;
    uint32_t nParsedAudioEncode = 0;
    uint32_t nParsedAudioCopy = 0;
    uint32_t nParsedAudioBitrate = 0;
    uint32_t nParsedAudioSamplerate = 0;
    uint32_t nParsedAudioSplit = 0;
    uint32_t nParsedAudioFilter = 0;
    uint32_t nTmpInputBuf = 0;
};

#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))
#define CMD_PARSE_SET_ERR(app_name, errmes, opt_name, err_val) \
    err.strAppName = (app_name) ? app_name : _T(""); \
    err.strErrorMessage = (errmes) ? errmes : _T(""); \
    err.strOptionName = (opt_name) ? opt_name : _T(""); \
    err.strErrorValue = (err_val) ? err_val : _T("");

int parse_one_input_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, VideoInfo *input, sArgsData *argData, ParseCmdError &err);
int parse_one_common_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamCommon *common, sArgsData *argData, ParseCmdError &err);
int parse_one_ctrl_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamControl *ctrl, sArgsData *argData, ParseCmdError &err);

tstring print_list_options(const TCHAR *option_name, const CX_DESC *list, int default_index);

tstring gen_cmd(const VideoInfo *common, const VideoInfo *defaultPrm, bool save_disabled_prm);
tstring gen_cmd(const RGYParamCommon *common, const RGYParamCommon *defaultPrm, bool save_disabled_prm);
tstring gen_cmd(const RGYParamControl *ctrl, const RGYParamControl *defaultPrm, bool save_disabled_prm);

tstring gen_cmd_help_input();
tstring gen_cmd_help_common();
tstring gen_cmd_help_ctrl();

#endif //__RGY_CMD_H__
