// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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

#pragma once
#ifndef __NVENC_CMD_H__
#define __NVENC_CMD_H__

#include <sstream>
#include "rgy_cmd.h"
#include "NVEncParam.h"

tstring GetNVEncVersion();
tstring encoder_help();
const TCHAR *cmd_short_opt_to_long(TCHAR short_opt);

template <size_t size> void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const guid_desc(&list)[size]);
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage, const std::vector<std::pair<RGY_CODEC, std::vector<guid_desc>>>& codec_list);

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, int nArgNum, const TCHAR **strInput, bool ignore_parse_err = false);
int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, const char *cmda, bool ignore_parse_err = false);

tstring gen_cmd(const InEncodeVideoParam *pParams, const NV_ENC_CODEC_CONFIG codecPrm[2], bool save_disabled_prm);

#endif //__NVENC_PARSE_CMD_H__