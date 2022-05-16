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

struct sArgsData {
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

#if defined(_WIN32) || defined(_WIN64)
static const auto CODEPAGE_CMDARG = _T("--process-codepage");
static const auto CODEPAGE_CMDARG_APPLIED = _T("--process-codepage-applied");
#endif //#if defined(_WIN32) || defined(_WIN64)

tstring encoder_help();
const TCHAR *cmd_short_opt_to_long(TCHAR short_opt);
int cmd_string_to_bool(bool *b, const tstring &str);
int parse_qp(int a[3], const TCHAR *str);

std::vector<tstring> cmd_from_config_file(const tstring& filename);
std::vector<std::pair<std::string, std::string>> createOptionList();

void print_cmd_error_unknown_opt(tstring strErrorValue);
void print_cmd_error_unknown_opt_param(tstring option, tstring strErrorValue, const std::vector<std::string> &optionParamsList);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<std::pair<RGY_CODEC, const CX_DESC *>>& codec_list);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const CX_DESC *list);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const FEATURE_DESC *list);

template<typename T>
void print_cmd_error_invalid_value(tstring strErrorMessage, tstring strOptionName, tstring strErrorValue, const T *list, int list_length = std::numeric_limits<int>::max());

int parse_one_vpp_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamVpp *vpp, sArgsData *argData);
int parse_one_input_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, VideoInfo *input, RGYParamInput *inprm, sArgsData *argData);
int parse_one_common_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamCommon *common, sArgsData *argData);
int parse_one_ctrl_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamControl *ctrl, sArgsData *argData);

tstring print_list_options(const TCHAR *option_name, const CX_DESC *list, int default_index);

tstring gen_cmd(const RGYParamVpp *common, const RGYParamVpp *defaultPrm, bool save_disabled_prm);
tstring gen_cmd(const VideoInfo *param, const VideoInfo *defaultPrm, const RGYParamInput *inprm, const RGYParamInput *inprmDefault, bool save_disabled_prm);
tstring gen_cmd(const RGYParamCommon *common, const RGYParamCommon *defaultPrm, bool save_disabled_prm);
tstring gen_cmd(const RGYParamControl *ctrl, const RGYParamControl *defaultPrm, bool save_disabled_prm);

tstring gen_cmd_help_input();
tstring gen_cmd_help_common();
tstring gen_cmd_help_vpp();
tstring gen_cmd_help_ctrl();

#endif //__RGY_CMD_H__
