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

#include <set>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "rgy_util.h"
#include "rgy_avutil.h"
#include "rgy_prm.h"
#include "rgy_cmd.h"
#include "rgy_language.h"
#include "rgy_perf_monitor.h"
#include "rgy_osdep.h"

#if !FOR_AUO
#if ENABLE_CPP_REGEX
#include <regex>
#endif //#if ENABLE_CPP_REGEX
#if ENABLE_DTL
#include <dtl/dtl.hpp>
#endif //#if ENABLE_DTL

#if ENABLE_CPP_REGEX
std::vector<std::pair<std::string, std::string>> createOptionList() {
    vector<std::pair<std::string, std::string>> optionList;
    const auto helpLines = split(tchar_to_string(encoder_help()), "\n");
    std::regex re1(R"(^\s{2,6}--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re2(R"(^\s{0,3}-[A-Za-z0-9],--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re3(R"(^\s{0,3}--\(no-\)([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::pair<std::string, std::string> lastOpt;
    int lastHit = -1;
    for (int i = 0; i < (int)helpLines.size(); i++) {
        const std::string &line = helpLines[i];
        bool matched = false;
        std::smatch match;
        if (std::regex_match(line, match, re1) && match.size() == 2) {
            matched = true;
        } else if (std::regex_match(line, match, re2) && match.size() == 2) {
            matched = true;
        } else if (std::regex_match(line, match, re3) && match.size() == 2) {
            matched = true;
        } else if (trim(line).length() == 0) {
            if (lastHit >= 0) {
                for (int j = lastHit; j < i; j++) {
                    optionList.back().second += helpLines[j];
                }
            }
            lastHit = -1;
        }
        if (matched) {
            if (lastHit >= 0) {
                for (int j = lastHit; j < i; j++) {
                    optionList.back().second += helpLines[j];
                }
            }
            optionList.push_back(std::make_pair(match[1], ""));
            lastHit = i;
        }
    }
    if (lastHit >= 0) {
        for (int j = lastHit; j < (int)helpLines.size(); j++) {
            optionList.back().second += helpLines[j];
        }
    }
    return optionList;
}
#endif //#if ENABLE_CPP_REGEX

#if (ENABLE_CPP_REGEX && ENABLE_DTL)
std::vector<std::pair<std::string, int>> searchNearString(const std::string &target, const std::vector<std::string> &candidateList) {
    //入力文字列を"-"で区切り、その組み合わせをすべて試す
    const auto target_words = split(target, "-", true);
    CombinationGenerator generator((int)target_words.size());
    const auto combinationList = generator.generate();
    vector<std::pair<std::string, int>> editDistList;
    for (const auto &candidate : candidateList) {
        int nMinEditDist = INT_MAX;
        for (const auto &combination : combinationList) {
            std::string check_key;
            for (auto i : combination) {
                if (check_key.length() > 0) {
                    check_key += "-";
                }
                check_key += target_words[i];
            }
            dtl::Diff<char, std::string> diff(check_key, candidate);
            diff.onOnlyEditDistance();
            diff.compose();
            nMinEditDist = (std::min)(nMinEditDist, (int)diff.getEditDistance());
        }
        editDistList.push_back(std::make_pair(candidate, nMinEditDist));
    }
    std::sort(editDistList.begin(), editDistList.end(), [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
        return b.second > a.second;
        });
    return editDistList;
}
#endif //#if (ENABLE_CPP_REGEX && ENABLE_DTL)
#endif //#if !FOR_AUO

void print_cmd_error_unknown_opt(tstring strErrorValue) {
#if !FOR_AUO
    _ftprintf(stderr, _T("Error: Unknown option: %s\n\n"), strErrorValue.c_str());
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
    if (strErrorValue.length() > 0) {
        //どのオプション名に近いか検証する
        const auto optHelpList = createOptionList();
        std::vector<std::string> optList;
        for (const auto &optHelp : optHelpList) {
            optList.push_back(optHelp.first);
        }
        const auto editDistList = searchNearString(tchar_to_string(strErrorValue.c_str()), optList);
        const int nMinEditDist = editDistList[0].second;
        _ftprintf(stderr, _T("Did you mean option(s) below?\n"));
        for (const auto &editDist : editDistList) {
            if (editDist.second != nMinEditDist) {
                break;
            }
            _ftprintf(stderr, _T("  --%s\n"), char_to_tstring(editDist.first).c_str());
        }
    }
#endif //#if ENABLE_DTL
#endif //#if !FOR_AUO
}


void print_cmd_error_unknown_opt_param(tstring option, tstring strErrorValue, const std::vector<std::string>& optionParamsList) {
#if !FOR_AUO
    _ftprintf(stderr, _T("Error: Unknown param \"%s\" for option \"--%s\"\n"), strErrorValue.c_str(), option.c_str());
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
    if (strErrorValue.length() > 0) {
        //どのオプション名に近いか検証する
        const auto editDistList = searchNearString(tchar_to_string(strErrorValue.c_str()), optionParamsList);
        const int nMinEditDist = editDistList[0].second;
        _ftprintf(stderr, _T("Did you mean param(s) below?\n"));
        for (const auto &editDist : editDistList) {
            if (editDist.second != nMinEditDist) {
                break;
            }
            _ftprintf(stderr, _T("  %s\n"), char_to_tstring(editDist.first).c_str());
        }
    }
#endif //#if ENABLE_DTL
#endif //#if !FOR_AUO
}

template<typename T>
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage, const T *list, int list_length) {
    if (!FOR_AUO && strOptionName.length() > 0) {
        if (strErrorValue.length() > 0) {
            if (0 == _tcsnccmp(strErrorValue.c_str(), _T("--"), _tcslen(_T("--")))
                || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && cmd_short_opt_to_long(strErrorValue[1]) != nullptr)) {
                _ftprintf(stderr, _T("Error: \"--%s\" requires value.\n\n"), strOptionName.c_str());
            } else {
                tstring str = _T("Error: Invalid value \"") + strErrorValue + _T("\" for \"--") + strOptionName + _T("\"");
                if (strErrorMessage.length() > 0) {
                    str += _T(": ") + strErrorMessage;
                }
                _ftprintf(stderr, _T("%s\n"), str.c_str());
            }
            if (list) {
                _ftprintf(stderr, _T("  Option value should be one of below...\n"));
                tstring str = _T("    ");
                for (int i = 0; list[i].desc && i < list_length; i++) {
                    str += tstring(list[i].desc) + _T(", ");
                    if (str.length() > 70) {
                        _ftprintf(stderr, _T("%s\n"), str.c_str());
                        str = _T("    ");
                    }
                }
                _ftprintf(stderr, _T("%s\n"), str.substr(0, str.length()-2).c_str());
            }
        } else {
            _ftprintf(stderr, _T("Error: %s for --%s\n\n"), strErrorMessage.c_str(), strOptionName.c_str());
        }
    }
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const CX_DESC *list) {
    print_cmd_error_invalid_value(strOptionName, strErrorValue, _T(""), list);
}
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const FEATURE_DESC *list) {
    print_cmd_error_invalid_value(strOptionName, strErrorValue, _T(""), list);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue) {
    print_cmd_error_invalid_value(strOptionName, strErrorValue, _T(""), (const CX_DESC *)nullptr);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage) {
    print_cmd_error_invalid_value(strOptionName, strErrorValue, strErrorMessage, (const CX_DESC *)nullptr);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<std::pair<RGY_CODEC, const CX_DESC *>>& codec_list) {
    if (!FOR_AUO && strOptionName.length() > 0) {
        if (strErrorValue.length() > 0) {
            if (0 == _tcsnccmp(strErrorValue.c_str(), _T("--"), _tcslen(_T("--")))
                || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && cmd_short_opt_to_long(strErrorValue[1]) != nullptr)) {
                _ftprintf(stderr, _T("Error: \"--%s\" requires value.\n\n"), strOptionName.c_str());
            } else {
                tstring str = _T("Error: Invalid value \"") + strErrorValue + _T("\" for \"--") + strOptionName + _T("\"");
                _ftprintf(stderr, _T("%s\n"), str.c_str());
            }
            _ftprintf(stderr, _T("  Option value should be one of below...\n"));
            for (const auto& codec : codec_list) {
                _ftprintf(stderr, _T("    For %s\n"), CodecToStr(codec.first).c_str());
                tstring str = _T("      ");
                for (int i = 0; codec.second[i].desc; i++) {
                    str += tstring(codec.second[i].desc) + _T(", ");
                    if (str.length() > 70) {
                        _ftprintf(stderr, _T("%s\n"), str.c_str());
                        str = _T("      ");
                    }
                }
                _ftprintf(stderr, _T("%s\n\n"), str.substr(0, str.length() - 2).c_str());
            }
        }
    }
}

std::vector<tstring> cmd_from_config_file(const tstring& filename) {
#if defined(_WIN32) || defined(_WIN64)
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        _ftprintf(stderr, _T("Failed to open option file!\n"));
        return std::vector<tstring>();
    }
    std::string configstr;
    std::string str;
    while (getline(ifs, str)) {
        str = trim(str);
        //行頭が"#"の場合はコメントとする
        if (str[0] == '#') continue;
        if (str.length() > 0) {
            if (configstr.length() > 0) {
                configstr += " ";
            }
            configstr += trim(str);
        }
    }
    //configstrが空文字列の場合、sep_cmdに渡すと先頭に実行ファイルへのパスが付与されてしまう
    //エラーを避けるため、空のvectorを返すようにする
    if (configstr.length() == 0) {
        _ftprintf(stderr, _T("Option file is empty!\n"));
        return std::vector<tstring>();
    }
    return sep_cmd(char_to_tstring(configstr));
#else
    _ftprintf(stderr, _T("--option-file not supported on linux systems!\n"));
    exit(1);
    return std::vector<tstring>();
#endif
}

int cmd_string_to_bool(bool *b, const tstring &str) {
    if (str == _T("true") || str == _T("on")) {
        *b = true;
        return 0;
    } else if (str == _T("false") || str == _T("off")) {
        *b = false;
        return 0;
    } else {
        return 1;
    }
}

static int getAudioTrackIdx(const RGYParamCommon *common, const int iTrack, const std::string& lang, const std::string& selectCodec) {
    if (iTrack == TRACK_SELECT_BY_LANG) {
        if (lang.length() == 0) return -1;
        for (int i = 0; i < common->nAudioSelectCount; i++) {
            if (lang == common->ppAudioSelectList[i]->lang) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_CODEC) {
        if (selectCodec.length() == 0) return -1;
        for (int i = 0; i < common->nAudioSelectCount; i++) {
            if (selectCodec == common->ppAudioSelectList[i]->selectCodec) {
                return i;
            }
        }
    } else {
        for (int i = 0; i < common->nAudioSelectCount; i++) {
            if (iTrack == common->ppAudioSelectList[i]->trackID) {
                return i;
            }
        }
    }
    return -1;
}

static int getFreeAudioTrack(const RGYParamCommon *common) {
    for (int iTrack = 1;; iTrack++) {
        if (0 > getAudioTrackIdx(common, iTrack, "", "")) {
            return iTrack;
        }
    }
#ifndef _MSC_VER
    return -1;
#endif //_MSC_VER
}

static int getSubTrackIdx(const RGYParamCommon *common, const int iTrack, const std::string& lang, const std::string& selectCodec) {
    if (iTrack == TRACK_SELECT_BY_LANG) {
        if (lang.length() == 0) return -1;
        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
            if (lang == common->ppSubtitleSelectList[i]->lang) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_CODEC) {
        if (selectCodec.length() == 0) return -1;
        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
            if (selectCodec == common->ppSubtitleSelectList[i]->selectCodec) {
                return i;
            }
        }
    } else {
        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
            if (iTrack == common->ppSubtitleSelectList[i]->trackID) {
                return i;
            }
        }
    }
    return -1;
}

static int getDataTrackIdx(const RGYParamCommon *common, const int iTrack, const std::string& lang, const std::string& selectCodec) {
    if (iTrack == TRACK_SELECT_BY_LANG) {
        if (lang.length() == 0) return -1;
        for (int i = 0; i < common->nDataSelectCount; i++) {
            if (lang == common->ppDataSelectList[i]->lang) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_CODEC) {
        if (selectCodec.length() == 0) return -1;
        for (int i = 0; i < common->nDataSelectCount; i++) {
            if (selectCodec == common->ppDataSelectList[i]->selectCodec) {
                return i;
            }
        }
    } else {
        for (int i = 0; i < common->nDataSelectCount; i++) {
            if (iTrack == common->ppDataSelectList[i]->trackID) {
                return i;
            }
        }
    }
    return -1;
}

static int getAttachmentTrackIdx(const RGYParamCommon *common, const int iTrack) {
    for (int i = 0; i < common->nAttachmentSelectCount; i++) {
        if (iTrack == common->ppAttachmentSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}

#pragma warning(disable: 4100) //warning C4100: 'argData': 引数は関数の本体部で 1 度も参照されません。
#pragma warning(disable: 4127) //warning C4127: 条件式が定数です。

int parse_one_vpp_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamVpp *vpp, sArgsData *argData) {
    if (IS_OPTION("vpp-resize")
        || (IS_OPTION("vpp-scaling") && ENCODER_QSV)) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_vpp_resize, strInput[i]))) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_resize_help);
            return 1;
        }
        vpp->resize_algo = (RGY_VPP_RESIZE_ALGO)value;
        return 0;
    }
    if (IS_OPTION("vpp-resize-mode") && ENCODER_QSV) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_vpp_resize_mode, strInput[i]))) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_resize_mode);
            return 1;
        }
        vpp->resize_mode = (RGY_VPP_RESIZE_MODE)value;
        return 0;
    }
    if (IS_OPTION("vpp-colorspace") && ENABLE_VPP_FILTER_COLORSPACE) {
        vpp->colorspace.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr + 1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        const auto paramList = std::vector<std::string>{
            "matrix", "colormatrix", "colorprim", "transfer", "range", "colorrange", "source_peak", "approx_gamma",
            "hdr2sdr", "ldr_nits", "a", "b", "c", "d", "e", "f", "contrast", "peak",
            "desat_base", "desat_strength", "desat_exp", "lut3d", "lut3d_interp" };

        for (const auto &param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto parse = [](int *from, int *to, tstring param_val, const CX_DESC *list) {
                    auto from_to = split(param_val, _T(":"));
                    if (from_to.size() == 2
                        && get_list_value(list, from_to[0].c_str(), from)
                        && get_list_value(list, from_to[1].c_str(), to)) {
                        return true;
                    }
                    return false;
                };
                if (vpp->colorspace.convs.size() == 0) {
                    vpp->colorspace.convs.push_back(ColorspaceConv());
                }
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("matrix") || param_arg == _T("colormatrix")) {
                    auto& conv = vpp->colorspace.convs.back();
                    if (conv.from.matrix != conv.to.matrix) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                        conv = vpp->colorspace.convs.back();
                    }
                    if (!parse((int *)&conv.from.matrix, (int *)&conv.to.matrix, param_val, list_colormatrix)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_colormatrix);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colorprim")) {
                    auto &conv = vpp->colorspace.convs.back();
                    if (conv.from.colorprim != conv.to.colorprim) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                        conv = vpp->colorspace.convs.back();
                    }
                    if (!parse((int *)&conv.from.colorprim, (int *)&conv.to.colorprim, param_val, list_colorprim)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_colorprim);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("transfer")) {
                    auto &conv = vpp->colorspace.convs.back();
                    if (conv.from.transfer != conv.to.transfer) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                        conv = vpp->colorspace.convs.back();
                    }
                    if (!parse((int *)&conv.from.transfer, (int *)&conv.to.transfer, param_val, list_transfer)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_transfer);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("range") || param_arg == _T("colorrange")) {
                    auto &conv = vpp->colorspace.convs.back();
                    if (conv.from.colorrange != conv.to.colorrange) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                        conv = vpp->colorspace.convs.back();
                    }
                    if (!parse((int *)&conv.from.colorrange, (int *)&conv.to.colorrange, param_val, list_colorrange)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_colorrange);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("source_peak")) {
                    try {
                        vpp->colorspace.hdr2sdr.hdr_source_peak = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("approx_gamma")) {
                    auto &conv = vpp->colorspace.convs.back();
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        conv.approx_gamma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scene_ref")) {
                    auto &conv = vpp->colorspace.convs.back();
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        conv.scene_ref = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("hdr2sdr")) {
                    int value = 0;
                    if (get_list_value(list_vpp_hdr2sdr, param_val.c_str(), &value)) {
                        vpp->colorspace.hdr2sdr.tonemap = (HDR2SDRToneMap)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_hdr2sdr);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ldr_nits")) {
                    try {
                        vpp->colorspace.hdr2sdr.ldr_nits = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("a")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.a = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("b")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.b = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("c")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.c = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("d")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.d = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("e")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.e = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("f")) {
                    try {
                        vpp->colorspace.hdr2sdr.hable.f = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("w")) {
                    continue;
                }
                if (param_arg == _T("transition")) {
                    try {
                        vpp->colorspace.hdr2sdr.mobius.transition = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        vpp->colorspace.hdr2sdr.reinhard.contrast = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("peak")) {
                    try {
                        float peak = std::stof(param_val);
                        vpp->colorspace.hdr2sdr.mobius.peak = peak;
                        vpp->colorspace.hdr2sdr.reinhard.peak = peak;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("desat_base")) {
                    try {
                        float desat_base = std::stof(param_val);
                        vpp->colorspace.hdr2sdr.desat_base = desat_base;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("desat_strength")) {
                    try {
                        float desat_strength = std::stof(param_val);
                        vpp->colorspace.hdr2sdr.desat_strength = desat_strength;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("desat_exp")) {
                    try {
                        float desat_exp = std::stof(param_val);
                        vpp->colorspace.hdr2sdr.desat_exp = desat_exp;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lut3d")) {
                    vpp->colorspace.lut3d.table_file = param_val;
                    continue;
                }
                if (param_arg == _T("lut3d_interp")) {
                    int value = 0;
                    if (get_list_value(list_vpp_colorspace_lut3d_interp, param_val.c_str(), &value)) {
                        vpp->colorspace.lut3d.interp = (LUT3DInterp)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_colorspace_lut3d_interp);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("hdr2sdr")) {
                    vpp->colorspace.hdr2sdr.tonemap = HDR2SDR_HABLE;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-delogo")) {
        vpp->delogo.enable = true;

        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr+1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        const auto paramList = std::vector<std::string>{
            "file", "select", "add", "pos", "depth", "y", "cb", "cr"
#if ENCODER_NVENC
            , "auto_nr", "auto_fade", "nr_area", "nr_value", "log"
#endif
        };

        for (const auto& param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->delogo.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("file")) {
                    try {
                        vpp->delogo.logoFilePath = trim(param_val, _T("\""));
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("select")) {
                    try {
                        vpp->delogo.logoSelect = param_val;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("add")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->delogo.mode = (b) ? DELOGO_MODE_ADD : DELOGO_MODE_REMOVE;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (ENABLE_VPP_FILTER_DELOGO_MULTIADD && param_arg == _T("multi_add")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->delogo.mode = (b) ? DELOGO_MODE_ADD_MULTI : DELOGO_MODE_REMOVE;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("pos")) {
                    int posOffsetX, posOffsetY;
                    if (   2 != _stscanf_s(param_val.c_str(), _T("%dx%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d,%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d/%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d:%d"), &posOffsetX, &posOffsetY)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    vpp->delogo.posX = posOffsetX;
                    vpp->delogo.posY = posOffsetY;
                    continue;
                }
                if (param_arg == _T("depth")) {
                    try {
                        vpp->delogo.depth = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("y")) {
                    try {
                        vpp->delogo.Y = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cb")) {
                    try {
                        vpp->delogo.Cb = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cr")) {
                    try {
                        vpp->delogo.Cr = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (ENCODER_NVENC) {
                    if (param_arg == _T("auto_nr")) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->delogo.autoNR = b;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("auto_fade")) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->delogo.autoFade = b;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("nr_area")) {
                        try {
                            vpp->delogo.NRArea = std::stoi(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("nr_value")) {
                        try {
                            vpp->delogo.NRValue = std::stoi(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("log")) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->delogo.log = b;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                }
                if (ENABLE_VPP_FILTER_DELOGO_MULTIADD) {
                    if (param_arg == _T("multi_add_depth_min")) {
                        try {
                            vpp->delogo.multiaddDepthMin = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("multi_add_depth_max")) {
                        try {
                            vpp->delogo.multiaddDepthMax = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                vpp->delogo.logoFilePath = param;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-file")) {
        i++;
        vpp->delogo.enable = true;
        vpp->delogo.logoFilePath = strInput[i];
        return 0;
    }
    if (IS_OPTION("vpp-delogo-select")) {
        i++;
        vpp->delogo.enable = true;
        vpp->delogo.logoSelect = strInput[i];
        return 0;
    }
    if (IS_OPTION("vpp-delogo-pos")) {
        i++;
        int posOffsetx = 0, posOffsety = 0;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &posOffsetx, &posOffsety)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        vpp->delogo.posX = posOffsetx;
        vpp->delogo.posY = posOffsety;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-depth")) {
        i++;
        try {
            vpp->delogo.depth = clamp(std::stoi(strInput[i]), 0, 255);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-y")) {
        i++;
        try {
            vpp->delogo.Y = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cb")) {
        i++;
        try {
            vpp->delogo.Cb = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cr")) {
        i++;
        try {
            vpp->delogo.Cr = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-add")) {
        vpp->delogo.mode = DELOGO_MODE_ADD;
        return 0;
    }
    if (IS_OPTION("no-vpp-delogo-add")) {
        vpp->delogo.mode = DELOGO_MODE_REMOVE;
        return 0;
    }
    if (IS_OPTION("vpp-afs") && ENABLE_VPP_FILTER_AFS) {
        vpp->afs.enable = true;

        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr+1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        const auto paramList = std::vector<std::string>{
            "top", "bottom", "left", "right",
            "method_switch", "coeff_shift", "thre_shift", "thre_deint", "thre_motion_y", "thre_motion_c",
            "level", "shift", "drop", "smooth", "24fps", "tune", "timecode", "ini", "preset", "rff", "log"
        };

        for (const auto &param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("ini")) {
                    if (vpp->afs.read_afs_inifile(trim(param_val, _T("\"")).c_str())) {
                        print_cmd_error_invalid_value(option_name, strInput[i], _T("ini file does not exist."));
                        return 1;
                    }
                }
                if (param_arg == _T("preset")) {
                    try {
                        int value = 0;
                        if (get_list_value(list_afs_preset, param_val.c_str(), &value)) {
                            vpp->afs.set_preset(value);
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_afs_preset);
                            return 1;
                        }
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
            }
        }
        for (const auto &param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->afs.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->afs.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("top")) {
                    try {
                        vpp->afs.clip.top = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("bottom")) {
                    try {
                        vpp->afs.clip.bottom = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("left")) {
                    try {
                        vpp->afs.clip.left = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("right")) {
                    try {
                        vpp->afs.clip.right = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("method_switch")) {
                    try {
                        vpp->afs.method_switch = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("coeff_shift")) {
                    try {
                        vpp->afs.coeff_shift = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_shift")) {
                    try {
                        vpp->afs.thre_shift = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_deint")) {
                    try {
                        vpp->afs.thre_deint = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_y")) {
                    try {
                        vpp->afs.thre_Ymotion = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_c")) {
                    try {
                        vpp->afs.thre_Cmotion = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("level")) {
                    try {
                        vpp->afs.analyze = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("shift")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.shift = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("drop")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.drop = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smooth")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.smooth = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("24fps")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.force24 = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tune")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.tune = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rff")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.rff = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("timecode")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.timecode = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.log = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ini")) {
                    continue;
                }
                if (param_arg == _T("preset")) {
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("shift")) {
                    vpp->afs.shift = true;
                    continue;
                }
                if (param == _T("drop")) {
                    vpp->afs.drop = true;
                    continue;
                }
                if (param == _T("smooth")) {
                    vpp->afs.smooth = true;
                    continue;
                }
                if (param == _T("24fps")) {
                    vpp->afs.force24 = true;
                    continue;
                }
                if (param == _T("tune")) {
                    vpp->afs.tune = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-nnedi") && ENABLE_VPP_FILTER_NNEDI) {
        vpp->nnedi.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "field", "nns", "nsize", "quality", "prescreen", "errortype", "prec", "weightfile" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->nnedi.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("field")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_field, param_val.c_str(), &value)) {
                        vpp->nnedi.field = (VppNnediField)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_field);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nns")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_nns, param_val.c_str(), &value)) {
                        vpp->nnedi.nns = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_nns);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nsize")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_nsize, param_val.c_str(), &value)) {
                        vpp->nnedi.nsize = (VppNnediNSize)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_nsize);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("quality")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_quality, param_val.c_str(), &value)) {
                        vpp->nnedi.quality = (VppNnediQuality)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_quality);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("prescreen")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_pre_screen, param_val.c_str(), &value)) {
                        vpp->nnedi.pre_screen = (VppNnediPreScreen)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_pre_screen);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("errortype")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nnedi_error_type, param_val.c_str(), &value)) {
                        vpp->nnedi.errortype = (VppNnediErrorType)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_error_type);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("prec")) {
                    int value = 0;
                    if (get_list_value(list_vpp_fp_prec, param_val.c_str(), &value)) {
                        vpp->nnedi.precision = (VppFpPrecision)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_fp_prec);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("weightfile")) {
                    vpp->nnedi.weightfile = trim(param_val, _T("\"")).c_str();
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-yadif") && ENABLE_VPP_FILTER_YADIF) {
        vpp->yadif.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "mode" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->yadif.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_yadif_mode, param_val.c_str(), &value)) {
                        vpp->yadif.mode = (VppYadifMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_yadif_mode);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-rff") && ENABLE_VPP_FILTER_RFF) {
        vpp->rff.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "log" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rff.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rff.log = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-select-every") && ENABLE_VPP_FILTER_SELECT_EVERY) {
        vpp->selectevery.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "offset", "step" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->selectevery.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("offset")) {
                    try {
                        vpp->selectevery.offset = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("step")) {
                    try {
                        vpp->selectevery.step = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                try {
                    vpp->selectevery.step = std::stoi(strInput[i]);
                } catch (...) {
                    print_cmd_error_invalid_value(option_name, strInput[i]);
                    return 1;
                }
                continue;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-decimate") && ENABLE_VPP_FILTER_DECIMATE) {
        vpp->decimate.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "cycle", "drop", "thresc", "thredup", "blockx", "blocky", "chroma", "log" /*, "pp"*/ };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decimate.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cycle")) {
                    try {
                        vpp->decimate.cycle = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("drop")) {
                    try {
                        vpp->decimate.drop = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thresc")) {
                    try {
                        vpp->decimate.threSceneChange = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thredup")) {
                    try {
                        vpp->decimate.threDuplicate = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blockx")) {
                    int value = 0;
                    if (get_list_value(list_vpp_decimate_block, param_val.c_str(), &value)) {
                        vpp->decimate.blockX = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_decimate_block);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blocky")) {
                    int value = 0;
                    if (get_list_value(list_vpp_decimate_block, param_val.c_str(), &value)) {
                        vpp->decimate.blockY = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_decimate_block);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("pp")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decimate.preProcessed = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decimate.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decimate.log = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("log")) {
                    vpp->decimate.log = true;
                    continue;
                }
                if (param == _T("chroma")) {
                    vpp->decimate.chroma = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-mpdecimate") && ENABLE_VPP_FILTER_MPDECIMATE) {
        vpp->mpdecimate.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "lo", "hi", "max", "frac", "log" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->mpdecimate.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lo")) {
                    try {
                        vpp->mpdecimate.lo = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("hi")) {
                    try {
                        vpp->mpdecimate.hi = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("max")) {
                    try {
                        vpp->mpdecimate.max = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("frac")) {
                    try {
                        vpp->mpdecimate.frac = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->mpdecimate.log = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("log")) {
                    vpp->decimate.log = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-pad") && ENABLE_VPP_FILTER_PAD) {
        vpp->pad.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "r", "l", "t", "b" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->pad.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("r")) {
                    try {
                        vpp->pad.right = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("l")) {
                    try {
                        vpp->pad.left = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("t")) {
                    try {
                        vpp->pad.top = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("b")) {
                    try {
                        vpp->pad.bottom = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                int val[4] = { 0 };
                if (   4 == _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &val[0], &val[1], &val[2], &val[3])
                    || 4 == _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &val[0], &val[1], &val[2], &val[3])) {
                    vpp->pad.left   = val[0];
                    vpp->pad.top    = val[1];
                    vpp->pad.right  = val[2];
                    vpp->pad.bottom = val[3];
                    return 0;
                }
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-convolution3d") && ENABLE_VPP_FILTER_CONVOLUTION3D) {
        vpp->convolution3d.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "matrix", "fast", "ythresh", "cthresh", "t_ythresh", "t_cthresh" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->convolution3d.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->convolution3d.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("matrix")) {
                    int value = 0;
                    if (get_list_value(list_vpp_convolution3d_matrix, param_val.c_str(), &value)) {
                        vpp->convolution3d.matrix = (VppConvolution3dMatrix)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_fp_prec);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("fast")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->convolution3d.fast = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ythresh")) {
                    try {
                        vpp->convolution3d.threshYspatial = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cthresh")) {
                    try {
                        vpp->convolution3d.threshCspatial = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("t_ythresh")) {
                    try {
                        vpp->convolution3d.threshYtemporal = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("t_cthresh")) {
                    try {
                        vpp->convolution3d.threshCtemporal = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-knn")) {
        vpp->knn.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            vpp->knn.radius = FILTER_DEFAULT_KNN_RADIUS;
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "radius", "strength", "lerp", "th_weight", "th_lerp" };

        int radius = FILTER_DEFAULT_KNN_RADIUS;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &radius)) {
            for (const auto& param : split(strInput[i], _T(","))) {
                auto pos = param.find_first_of(_T("="));
                if (pos != std::string::npos) {
                    auto param_arg = param.substr(0, pos);
                    auto param_val = param.substr(pos+1);
                    param_arg = tolowercase(param_arg);
                    if (param_arg == _T("enable")) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->knn.enable = b;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("radius")) {
                        try {
                            vpp->knn.radius = std::stoi(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("strength")) {
                        try {
                            vpp->knn.strength = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("lerp")) {
                        try {
                            vpp->knn.lerpC = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_weight")) {
                        try {
                            vpp->knn.weight_threshold = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_lerp")) {
                        try {
                            vpp->knn.lerp_threshold = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                    return 1;
                } else {
                    print_cmd_error_unknown_opt_param(option_name, param, paramList);
                    return 1;
                }
            }
        } else {
            vpp->knn.radius = radius;
        }
        return 0;
    }
    if (IS_OPTION("vpp-pmd") && ENABLE_VPP_FILTER_PMD) {
        vpp->pmd.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "apply_count", "strength", "threshold", "useexp" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->pmd.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("apply_count")) {
                    try {
                        vpp->pmd.applyCount = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->pmd.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->pmd.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("useexp")) {
                    try {
                        vpp->pmd.useExp = std::stoi(param_val) != 0;
                    } catch (...) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->pmd.useExp = b;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-smooth") && ENABLE_VPP_FILTER_SMOOTH) {
        vpp->smooth.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "quality", "qp", "use_qp_table" /*, "strength", "threshold", "bratio", "prec", "max_error"*/ };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->smooth.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->smooth.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("quality")) {
                    try {
                        vpp->smooth.quality = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("qp")) {
                    try {
                        vpp->smooth.qp = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("use_qp_table")) {
                    vpp->smooth.useQPTable = (param_val == _T("on") || param_val == _T("true"));
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->smooth.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->smooth.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("bratio")) {
                    try {
                        vpp->smooth.bratio = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("prec")) {
                    int value = 0;
                    if (get_list_value(list_vpp_fp_prec, param_val.c_str(), &value)) {
                        vpp->smooth.prec = (VppFpPrecision)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_fp_prec);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("max_error")) {
                    try {
                        vpp->smooth.maxQPTableErrCount = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-subburn")) {
        VppSubburn subburn;
        subburn.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            vpp->subburn.push_back(subburn);
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr+1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        const auto paramList = std::vector<std::string>{ "track", "filename", "charcode", "shaping", "scale", "transparency", "brightness", "contrast", "vid_ts_offset", "ts_offset", "fontsdir", "forced_subs_only" };

        for (const auto &param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        subburn.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("track")) {
                    try {
                        subburn.trackId = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("filename")) {
                    subburn.filename = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("charcode")) {
                    subburn.charcode = trim(tchar_to_string(param_val), "\"");
                    continue;
                }
                if (param_arg == _T("shaping")) {
                    int value = 0;
                    if (get_list_value(list_vpp_ass_shaping, param_val.c_str(), &value)) {
                        subburn.assShaping = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_ass_shaping);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scale")) {
                    try {
                        subburn.scale = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("transparency")) {
                    try {
                        subburn.transparency_offset = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("brightness")) {
                    try {
                        subburn.brightness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        subburn.contrast = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("vid_ts_offset")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        subburn.vid_ts_offset = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ts_offset")) {
                    try {
                        subburn.ts_offset = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("fontsdir")) {
                    subburn.fontsdir = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("forced_subs_only")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        subburn.forced_subs_only = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            } else {
                try {
                    subburn.trackId = std::stoi(param);
                } catch (...) {
                    subburn.filename = param;
                }
            }
        }
        vpp->subburn.push_back(subburn);
        return 0;
    }

    if (IS_OPTION("vpp-unsharp") && ENABLE_VPP_FILTER_UNSHARP) {
        vpp->unsharp.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            vpp->unsharp.radius = FILTER_DEFAULT_UNSHARP_RADIUS;
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "radius", "weight", "threshold" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->unsharp.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->unsharp.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("radius")) {
                    try {
                        vpp->unsharp.radius = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("weight")) {
                    try {
                        vpp->unsharp.weight = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->unsharp.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-edgelevel") && ENABLE_VPP_FILTER_EDGELEVEL) {
        vpp->edgelevel.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "strength", "threshold", "black", "white" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->edgelevel.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->edgelevel.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->edgelevel.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("black")) {
                    try {
                        vpp->edgelevel.black = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("white")) {
                    try {
                        vpp->edgelevel.white = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-warpsharp") && ENABLE_VPP_FILTER_WARPSHARP) {
        vpp->warpsharp.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "threshold", "blur", "type", "depth", "chroma" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->warpsharp.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->warpsharp.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blur")) {
                    try {
                        vpp->warpsharp.blur = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("type")) {
                    try {
                        vpp->warpsharp.type = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("depth")) {
                    try {
                        vpp->warpsharp.depth = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    try {
                        vpp->warpsharp.chroma = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-curves") && ENABLE_VPP_FILTER_CURVES) {
        vpp->curves.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "r", "g", "b", "m", "red", "green", "blue", "master", "all", "preset" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->curves.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("r") || param_arg == _T("red")) {
                    vpp->curves.prm.r = param_val;
                    continue;
                }
                if (param_arg == _T("g") || param_arg == _T("green")) {
                    vpp->curves.prm.g = param_val;
                    continue;
                }
                if (param_arg == _T("b") || param_arg == _T("blue")) {
                    vpp->curves.prm.b = param_val;
                    continue;
                }
                if (param_arg == _T("m") || param_arg == _T("master")) {
                    vpp->curves.prm.m = param_val;
                    continue;
                }
                if (param_arg == _T("all")) {
                    vpp->curves.prm.m = param_val;
                    continue;
                }
                if (param_arg == _T("preset")) {
                    int value = 0;
                    if (get_list_value(list_vpp_curves_preset, param_val.c_str(), &value)) {
                        vpp->curves.preset = (VppCurvesPreset)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_curves_preset);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-tweak") && ENABLE_VPP_FILTER_TWEAK) {
        vpp->tweak.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "contrast", "brightness", "gamma", "saturation", "swapuv", "hue" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->tweak.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("brightness")) {
                    try {
                        vpp->tweak.brightness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        vpp->tweak.contrast = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gamma")) {
                    try {
                        vpp->tweak.gamma = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("saturation")) {
                    try {
                        vpp->tweak.saturation = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("hue")) {
                    try {
                        vpp->tweak.hue = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("swapuv")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->tweak.swapuv = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("swapuv")) {
                    vpp->tweak.swapuv = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-rotate")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        int value = 0;
        if (get_list_value(list_vpp_rotate, strInput[i], &value)) {
            vpp->transform.enable = true;
            if (!vpp->transform.setRotate(value)) {
                print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_rotate);
                return 1;
            }
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_rotate);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-mirror")) { // QSVEncでの互換性維持のため
        vpp->transform.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        int value = 0;
        if (get_list_value(list_vpp_mirroring, strInput[i], &value)) {
            switch (value) {
            case 1: /*horizontal 水平方向*/
                vpp->transform.enable = true;
                vpp->transform.flipX = true;
                vpp->transform.flipY = false;
                break;
            case 2: /*horizontal 垂直方向*/
                vpp->transform.enable = true;
                vpp->transform.flipX = false;
                vpp->transform.flipY = true;
                break;
            case 0:
            default:
                vpp->transform.enable = false;
                vpp->transform.flipX = false;
                vpp->transform.flipY = false;
                break;
            }
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_mirroring);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-transform")) {
        vpp->transform.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "flip_x", "flip_y", "transpose" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->transform.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("flip_x")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->transform.flipX = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("flip_y")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->transform.flipY = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("transpose")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->transform.transpose = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("flip_x")) {
                    vpp->transform.flipX = true;
                    continue;
                }
                if (param == _T("flip_y")) {
                    vpp->transform.flipY = true;
                    continue;
                }
                if (param == _T("transpose")) {
                    vpp->transform.transpose = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-deband") && ENABLE_VPP_FILTER_DEBAND) {
        vpp->deband.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "range", "thre", "thre_y", "thre_cb",
            "thre_cr", "dither", "dither_y", "dither_c", "sample", "seed",
            "blurfirst", "rand_each_frame" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deband.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("range")) {
                    try {
                        vpp->deband.range = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre")) {
                    try {
                        vpp->deband.threY = std::stoi(param_val);
                        vpp->deband.threCb = vpp->deband.threY;
                        vpp->deband.threCr = vpp->deband.threY;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_y")) {
                    try {
                        vpp->deband.threY = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cb")) {
                    try {
                        vpp->deband.threCb = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cr")) {
                    try {
                        vpp->deband.threCr = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dither")) {
                    try {
                        vpp->deband.ditherY = std::stoi(param_val);
                        vpp->deband.ditherC = vpp->deband.ditherY;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_y")) {
                    try {
                        vpp->deband.ditherY = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_c")) {
                    try {
                        vpp->deband.ditherC = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sample")) {
                    try {
                        vpp->deband.sample = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("seed")) {
                    try {
                        vpp->deband.seed = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blurfirst")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deband.blurFirst = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rand_each_frame")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deband.randEachFrame = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("blurfirst")) {
                    vpp->deband.blurFirst = true;
                    continue;
                }
                if (param == _T("rand_each_frame")) {
                    vpp->deband.randEachFrame = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-overlay") && ENABLE_VPP_FILTER_OVERLAY) {
        VppOverlay overlay;
        overlay.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr + 1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        const auto paramList = std::vector<std::string>{
            "pos", "posx", "posy",
            "size", "width", "height",
            "alpha", "alpha_mode", "loop", "file",
            "lumakey_threshold", "lumakey_tolerance", "lumakey_softness"};

        for (const auto& param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        overlay.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("file")) {
                    overlay.inputFile = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("pos")) {
                    int x = 0, y = 0;
                    if (   _stscanf_s(param_val.c_str(), _T("%dx%d"), &x, &y) == 2
                        || _stscanf_s(param_val.c_str(), _T("%d/%d"), &x, &y) == 2) {
                        overlay.posX = x;
                        overlay.posY = y;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("posx")) {
                    try {
                        overlay.posX = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("posy")) {
                    try {
                        overlay.posY = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("size")) {
                    int w = 0, h = 0;
                    if (   _stscanf_s(param_val.c_str(), _T("%dx%d"), &w, &h) == 2
                        || _stscanf_s(param_val.c_str(), _T("%d/%d"), &w, &h) == 2) {
                        overlay.width = w;
                        overlay.height = h;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("width")) {
                    try {
                        overlay.width = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("height")) {
                    try {
                        overlay.height = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("alpha")) {
                    try {
                        overlay.alpha = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("alpha_mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_overlay_alpha_mode, param_val.c_str(), &value)) {
                        overlay.alphaMode = (VppOverlayAlphaMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_overlay_alpha_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lumakey_threshold")) {
                    try {
                        overlay.lumaKey.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lumakey_tolerance")) {
                    try {
                        overlay.lumaKey.tolerance = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lumakey_softness")) {
                    try {
                        overlay.lumaKey.shoftness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("loop")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        overlay.loop = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("loop")) {
                    overlay.loop = true;
                    continue;
                }
                overlay.inputFile = param;
                return 0;
            }
        }
        vpp->overlay.push_back(overlay);
        return 0;
    }
    if (IS_OPTION("vpp-perf-monitor")) {
        vpp->checkPerformance = true;
        return 0;
    }
    if (IS_OPTION("no-vpp-perf-monitor")) {
        vpp->checkPerformance = false;
        return 0;
    }
    return -1;
}

int parse_one_input_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, VideoInfo *input, RGYParamInput *inprm, sArgsData *argData) {
    if (IS_OPTION("frames")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        input->frames = v;
        return 0;
    }
    if (IS_OPTION("fps")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            input->fpsN = a[0];
            input->fpsD = a[1];
        } else {
            double d;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    input->fpsN = rate;
                    input->fpsD = 1001;
                } else {
                    input->fpsD = 100000;
                    input->fpsN = (int)(d * input->fpsD + 0.5);
                    rgy_reduce(input->fpsN, input->fpsD);
                }
            } else {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-res")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])) {
            input->srcWidth  = a[0];
            input->srcHeight = a[1];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("output-res")) {
        if (i + 1 >= nArgNum) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "preserve_aspect_ratio" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("preserve_aspect_ratio")) {
                    int value = 0;
                    if (get_list_value(list_vpp_resize_res_mode, param_val.c_str(), &value)) {
                        inprm->resizeResMode = (RGYResizeResMode)value;
                        continue;
                    } else {
                        print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_resize_res_mode);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                int a[2] = { 0 };
                if (   2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
                    || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])) {
                    input->dstWidth  = a[0];
                    input->dstHeight = a[1];
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("crop")) {
        i++;
        sInputCrop a = initCrop();
        if (   4 == _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])
            || 4 == _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])) {
            input->crop = a;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("input-csp")) {
        i++;
        int value = 0;
        if (get_list_value(list_rgy_csp, strInput[i], &value)) {
            input->csp = (RGY_CSP)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_rgy_csp);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("raw")) {
        input->type = RGY_INPUT_FMT_RAW;
        return 0;
    }
    if (IS_OPTION("y4m")) {
        input->type = RGY_INPUT_FMT_Y4M;
        return 0;
    }
    if (IS_OPTION("sm")) {
#if ENABLE_SM_READER
        input->type = RGY_INPUT_FMT_SM;
        return 0;
#else
        _ftprintf(stderr, _T("sm reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("avi")) {
#if ENABLE_AVI_READER
        input->type = RGY_INPUT_FMT_AVI;
        return 0;
#else
        _ftprintf(stderr, _T("avi reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("avs")) {
#if ENABLE_AVISYNTH_READER
        input->type = RGY_INPUT_FMT_AVS;
        return 0;
#else
        _ftprintf(stderr, _T("avs reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("vpy")) {
#if ENABLE_VAPOURSYNTH_READER
        input->type = RGY_INPUT_FMT_VPY;
        return 0;
#else
        _ftprintf(stderr, _T("vpy reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("vpy-mt")) {
#if ENABLE_VAPOURSYNTH_READER
        input->type = RGY_INPUT_FMT_VPY_MT;
        return 0;
#else
        _ftprintf(stderr, _T("vpy-mt reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (   IS_OPTION("avcuvid")
        || IS_OPTION("avqsv")
        || IS_OPTION("avvce")
        || IS_OPTION("avhw")) {
#if ENABLE_AVSW_READER
        input->type = RGY_INPUT_FMT_AVHW;
        return 0;
#else
        _ftprintf(stderr, _T("avhw reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("avsw")) {
#if ENABLE_AVSW_READER
        input->type = RGY_INPUT_FMT_AVSW;
        return 0;
#else
        _ftprintf(stderr, _T("avsw reader not supported in this build.\n"));
        return 1;
#endif
    }
    if (IS_OPTION("tff")) {
        input->picstruct = RGY_PICSTRUCT_FRAME_TFF;
        return 0;
    }
    if (IS_OPTION("bff")) {
        input->picstruct = RGY_PICSTRUCT_FRAME_BFF;
        return 0;
    }
    if (IS_OPTION("interlace") || IS_OPTION("interlaced")) {
        i++;
        int value = 0;
        if (get_list_value(list_interlaced, strInput[i], &value)) {
            input->picstruct = (RGY_PICSTRUCT)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], _T(""), list_interlaced, _countof(list_interlaced));
            return 1;
        }
        return 0;
    }
    return -1;
}

int parse_log_level_param(const TCHAR *option_name, const TCHAR *arg_value, RGYParamLogLevel& loglevel) {
    std::vector<std::string> paramList;
    for (const auto& param : RGY_LOG_TYPE_STR) {
        paramList.push_back(tchar_to_string(param.second));
    }
    std::vector<CX_DESC> logLevelList;
    for (const auto& param : RGY_LOG_LEVEL_STR) {
        logLevelList.push_back({ param.second, param.first });
    }
    logLevelList.push_back({ nullptr, 0 });

    for (const auto &param : split(arg_value, _T(","))) {
        auto pos = param.find_first_of(_T("="));
        if (pos != std::string::npos) {
            auto param_arg = param.substr(0, pos);
            auto param_val = param.substr(pos + 1);
            param_arg = tolowercase(param_arg);
            int value = 0;
            if (get_list_value(logLevelList.data(), param_val.c_str(), &value)) {
                auto type_ret = std::find_if(RGY_LOG_TYPE_STR.begin(), RGY_LOG_TYPE_STR.end(), [param_arg](decltype(RGY_LOG_TYPE_STR[0])& type) {
                    return param_arg == type.second;
                    });
                if (type_ret != RGY_LOG_TYPE_STR.end()) {
                    loglevel.set((RGYLogLevel)value, type_ret->first);
                    continue;
                } else {
                    print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                    return 1;
                }
            } else {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, logLevelList.data());
                return 1;
            }
        } else {
            int value = 0;
            if (get_list_value(logLevelList.data(), param.c_str(), &value)) {
                loglevel.set((RGYLogLevel)value, RGY_LOGT_ALL);
                continue;
            } else {
                print_cmd_error_invalid_value(option_name, arg_value, logLevelList.data());
                return 1;
            }
        }
    }
    return 0;
}

int parse_one_audio_param(AudioSelect& chSel, const tstring& str, const TCHAR *option_name) {
    const auto paramList = std::vector<std::string>{ "codec", "bitrate", "samplerate", "delay", "profile", "disposition", "filter", "dec_prm", "enc_prm", "lang", "select-codec", "metadata", "bsf", "copy" };
    for (const auto &param : split(str, _T(";"))) {
        auto pos = param.find_first_of(_T("="));
        if (pos != std::string::npos) {
            auto param_arg = param.substr(0, pos);
            auto param_val = param.substr(pos + 1);
            if (param_arg == _T("codec")) {
                chSel.encCodec = param_val;
            } else if (param_arg == _T("bitrate")) {
                try {
                    chSel.encBitrate = std::stoi(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("samplerate")) {
                try {
                    chSel.encSamplingRate = std::stoi(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("delay")) {
                try {
                    chSel.addDelayMs = std::stoi(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("profile")) {
                chSel.encCodecProfile = param_val;
            } else if (param_arg == _T("disposition")) {
                chSel.disposition = param_val;
            } else if (param_arg == _T("filter")) {
                chSel.filter = param_val;
            } else if (param_arg == _T("dec_prm")) {
                chSel.decCodecPrm = param_val;
            } else if (param_arg == _T("enc_prm")) {
                chSel.encCodecPrm = param_val;
            } else if (param_arg == _T("lang")) {
                chSel.lang = tchar_to_string(param_val);
            } else if (param_arg == _T("select-codec")) {
                chSel.selectCodec = tchar_to_string(param_val);
            } else if (param_arg == _T("metadata")) {
                chSel.metadata.push_back(param_val);
            } else if (param_arg == _T("bsf")) {
                chSel.bsf = param_val;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_AUTO;
            }
            continue;
        } else {
            if (param == _T("copy")) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
    }
    return 0;
}

int parse_one_subtitle_param(SubtitleSelect& chSel, const tstring& str, const TCHAR *option_name) {
    const auto paramList = std::vector<std::string>{ "codec", "dec_prm", "enc_prm", "disposition", "select-codec", "metadata", "lang", "bsf", "copy", "asdata" };
    for (const auto &param : split(str, _T(";"))) {
        auto pos = param.find_first_of(_T("="));
        if (pos != std::string::npos) {
            auto param_arg = param.substr(0, pos);
            auto param_val = param.substr(pos + 1);
            if (param_arg == _T("codec")) {
                chSel.encCodec = param_val;
            } else if (param_arg == _T("dec_prm")) {
                chSel.decCodecPrm = param_val;
            } else if (param_arg == _T("enc_prm")) {
                chSel.encCodecPrm = param_val;
            } else if (param_arg == _T("disposition")) {
                chSel.disposition = param_val;
            } else if (param_arg == _T("select-codec")) {
                chSel.selectCodec = tchar_to_string(param_val);
            } else if (param_arg == _T("metadata")) {
                chSel.metadata.push_back(param_val);
            } else if (param_arg == _T("lang")) {
                chSel.lang = tchar_to_string(param_val);
            } else if (param_arg == _T("bsf")) {
                chSel.bsf = param_val;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
            continue;
        } else {
            if (param == _T("copy")) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            } else if (param == _T("asdata")) {
                chSel.asdata = true;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
    }
    return 0;
}

int parse_one_common_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamCommon *common, sArgsData *argData) {

    if (IS_OPTION("input") || IS_OPTION("input-file")) {
        i++;
        common->inputFilename = strInput[i];
        return 0;
    }
    if (IS_OPTION("output") || IS_OPTION("output-file")) {
        i++;
        common->outputFilename = strInput[i];
        return 0;
    }

    if (   IS_OPTION("input-analyze")
        || IS_OPTION("avcuvid-analyze")) {
        i++;
        double value = 0.0;
        if (rgy_parse_num(value, strInput[i]) != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        } else if (value < 0.0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("input-analyze requires non-negative value."));
            return 1;
        } else {
            common->demuxAnalyzeSec = value;
        }
        return 0;
    }
    if (IS_OPTION("input-probesize")) {
        i++;
        int64_t value = 0;
        if (rgy_parse_num(value, strInput[i]) != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        } else if (value < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("input-probesize requires non-negative value."));
            return 1;
        } else {
            common->demuxProbesize = value;
        }
        return 0;
    }
    if (IS_OPTION("input-retry")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->inputRetry = v;
        return 0;
    }
    if (IS_OPTION("video-track")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->videoTrack = v;
        return 0;
    }
    if (IS_OPTION("video-streamid")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->videoStreamId = v;
        return 0;
    }
    if (IS_OPTION("video-tag")) {
        i++;
        common->videoCodecTag = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("video-metadata")) {
        i++;
        common->videoMetadata.push_back(strInput[i]);
        return 0;
    }
    if (IS_OPTION("trim")) {
        i++;
        auto trim_str_list = split(strInput[i], _T(","));
        std::vector<sTrim> trim_list;
        for (auto trim_str : trim_str_list) {
            sTrim trim;
            if (2 != _stscanf_s(trim_str.c_str(), _T("%d:%d"), &trim.start, &trim.fin) || (trim.fin > 0 && trim.fin < trim.start)) {
                print_cmd_error_invalid_value(option_name, trim_str.c_str());
                return 1;
            }
            if (trim.fin == 0) {
                trim.fin = TRIM_MAX;
            } else if (trim.fin < 0) {
                trim.fin = trim.start - trim.fin - 1;
            }
            trim_list.push_back(trim);
        }
        if (trim_list.size()) {
            std::sort(trim_list.begin(), trim_list.end(), [](const sTrim& trimA, const sTrim& trimB) { return trimA.start < trimB.start; });
            for (int j = (int)trim_list.size() - 2; j >= 0; j--) {
                if (trim_list[j].fin > trim_list[j+1].start) {
                    trim_list[j].fin = trim_list[j+1].fin;
                    trim_list.erase(trim_list.begin() + j+1);
                }
            }
            common->nTrimCount = (int)trim_list.size();
            common->pTrimList = (sTrim *)malloc(sizeof(common->pTrimList[0]) * trim_list.size());
            memcpy(common->pTrimList, &trim_list[0], sizeof(common->pTrimList[0]) * trim_list.size());
        }
        return 0;
    }
    if (IS_OPTION("seek") || IS_OPTION("seekto")) {
        const bool seekTo = IS_OPTION("seekto");
        i++;
        int ret = 0;
        int hh = 0, mm = 0;
        float sec = 0.0f;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%f"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%f"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%f"),              &sec))) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (hh > 0 && mm >= 60) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (seekTo) {
            common->seekToSec = sec + mm * 60;
        } else {
            common->seekSec = sec + mm * 60;
        }
        return 0;
    }
#if ENABLE_AVSW_READER && !FOR_AUO
    if (IS_OPTION("audio-source")) {
        i++;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        AudioSource src;
        const TCHAR *ptr = strInput[i];
        const TCHAR *qtr = _tcsrchr(ptr, _T(':'));
#if defined(_WIN32) || defined(_WIN64)
        if (qtr
            && (qtr - ptr == 1 || qtr - ptr == 2)
            && qtr[1] == _T('\\')
            && _istalpha(qtr[-1])
            && (qtr - ptr == 1 || qtr[-2] == _T('\"'))) {
            qtr = _tcsrchr(qtr + 1, _T(':'));
        }
#endif
        if (qtr == nullptr) {
            src.filename = strInput[i];
            src.select[0].encCodec = RGY_AVCODEC_COPY;
            common->audioSource.push_back(src);
            return 0;
        }
        src.filename = tstring(strInput[i]).substr(0, qtr - ptr);
        auto channel_select_list = split(qtr+1, _T("/"));
        for (size_t ichannel = 0; ichannel < channel_select_list.size(); ichannel++) {
            auto& channel = channel_select_list[ichannel];
            {
                auto option_split = channel.find(_T('='));
                if (option_split != std::string::npos) {
                    if (channel.substr(0, option_split) == _T("format")) {
                        src.format = channel.substr(option_split + 1);
                        continue;
                    } else if (channel.substr(0, option_split) == _T("input_opt")) {
                        src.inputOpt.push_back(std::make_pair<tstring, tstring>(tstring(channel.substr(option_split + 1)), tstring(channel_select_list[ichannel + 1])));
                        ichannel++;
                        continue;
                    }
                }
            }
            int trackId = 0;
            auto channel_id_split = channel.find(_T('?'));
            if (channel_id_split != std::string::npos) {
                try {
                    trackId = std::stoi(channel.substr(0, channel_id_split));
                } catch (...) {
                    print_cmd_error_invalid_value(option_name, strInput[i]);
                    return 1;
                }
                channel = channel.substr(channel_id_split+1);
            }
            AudioSelect &chSel = src.select[trackId];
            chSel.trackID = trackId;
            int ret = parse_one_audio_param(chSel, channel, option_name);
            if (ret != 0) return ret;
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
        }
        common->audioSource.push_back(src);
        return 0;
    }
    if (IS_OPTION("audio-file")) {
        i++;
        const TCHAR *ptr = strInput[i];
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = -1;
        int trackId = 0;
        if (_tcschr(ptr, '?') == nullptr || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
            //トラック番号を適当に発番する (カウントは1から)
            trackId = argData->nParsedAudioFile+1;
            audioIdx = getAudioTrackIdx(common, trackId, "", "");
            if (audioIdx < 0 || common->ppAudioSelectList[audioIdx]->extractFilename.length() > 0) {
                trackId = getFreeAudioTrack(common);
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
        } else if (i <= 0) {
            //トラック番号は1から連番で指定
            print_cmd_error_invalid_value(option_name, strInput[i], _T("track number should be positive value."));
            return 1;
        } else {
            audioIdx = getAudioTrackIdx(common, trackId, "", "");
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
            ptr = _tcschr(ptr, '?') + 1;
        }
        assert(pAudioSelect != nullptr);
        const TCHAR *qtr = _tcschr(ptr, ':');
        if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
            pAudioSelect->extractFormat = ptr;
            ptr = qtr + 1;
        }
        size_t filename_len = _tcslen(ptr);
        //ファイル名が""でくくられてたら取り除く
        if (ptr[0] == _T('\"') && ptr[filename_len-1] == _T('\"')) {
            filename_len -= 2;
            ptr++;
        }
        //ファイル名が重複していないかを確認する
        for (int j = 0; j < common->nAudioSelectCount; j++) {
            if (common->ppAudioSelectList[j]->extractFilename.length() > 0
                && 0 == _tcsicmp(common->ppAudioSelectList[j]->extractFilename.c_str(), ptr)) {
                print_cmd_error_invalid_value(option_name, _T(""), _T("Same output file name is used more than twice."));
                return 1;
            }
        }

        if (audioIdx < 0) {
            audioIdx = common->nAudioSelectCount;
            //新たに要素を追加
            common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
            common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
            common->nAudioSelectCount++;
        }
        common->ppAudioSelectList[audioIdx]->extractFilename = ptr;
        argData->nParsedAudioFile++;
        return 0;
    }
    if (IS_OPTION("format") || IS_OPTION("output-format")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->muxOutputFormat = strInput[i];
            if (0 != _tcsicmp(strInput[i], _T("raw"))) {
                common->AVMuxTarget |= RGY_MUX_VIDEO;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-format")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->AVInputFormat = _tcsdup(strInput[i]);
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    auto set_audio_prm = [&](std::function<void(AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        if (i+1 < nArgNum) {
            int test_val = 0;
            if ((strInput[i+1][0] != _T('-') || (_stscanf_s(strInput[i+1], _T("%d"), &test_val) == 1 && test_val < 0)) && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                const tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                try {
                    trackId = std::stoi(temp);
                } catch (...) {
                    auto tempc = tchar_to_string(temp);
                    if (rgy_lang_exist(tempc)) {
                        trackId = TRACK_SELECT_BY_LANG;
                        lang = tempc;
                    } else if (avcodec_exists(tempc, AVMEDIA_TYPE_AUDIO)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    }
                }
            }
        }
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(common, trackId, lang, selectCodec);
        if (audioIdx < 0) {
            pAudioSelect = new AudioSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                AudioSelect *pAudioSelectAll = nullptr;
                for (int itrack = 0; itrack < common->nAudioSelectCount; itrack++) {
                    if (common->ppAudioSelectList[itrack]->trackID == 0) {
                        pAudioSelectAll = common->ppAudioSelectList[itrack];
                    }
                }
                if (pAudioSelectAll) {
                    *pAudioSelect = *pAudioSelectAll;
                }
            }
            pAudioSelect->trackID = trackId;
        } else {
            pAudioSelect = common->ppAudioSelectList[audioIdx];
        }
        pAudioSelect->lang = lang;
        func_set(pAudioSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < common->nAudioSelectCount; itrack++) {
                func_set(common->ppAudioSelectList[itrack], trackId, ptr);
            }
        }

        if (audioIdx < 0) {
            audioIdx = common->nAudioSelectCount;
            //新たに要素を追加
            common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
            common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
            common->nAudioSelectCount++;
        }
        return 0;
    };
    auto set_sub_prm = [&](std::function<void(SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        if (i+1 < nArgNum) {
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                const tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                try {
                    trackId = std::stoi(temp);
                } catch (...) {
                    auto tempc = tchar_to_string(temp);
                    if (rgy_lang_exist(tempc)) {
                        trackId = TRACK_SELECT_BY_LANG;
                        lang = tempc;
                    } else if (avcodec_exists(tempc, AVMEDIA_TYPE_SUBTITLE)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    }
                }
            }
        }
        SubtitleSelect *pSubSelect = nullptr;
        int subIdx = getSubTrackIdx(common, trackId, lang, selectCodec);
        if (subIdx < 0) {
            pSubSelect = new SubtitleSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                SubtitleSelect *pSubSelectAll = nullptr;
                for (int itrack = 0; itrack < common->nSubtitleSelectCount; itrack++) {
                    if (common->ppSubtitleSelectList[itrack]->trackID == 0) {
                        pSubSelectAll = common->ppSubtitleSelectList[itrack];
                    }
                }
                if (pSubSelectAll) {
                    *pSubSelect = *pSubSelectAll;
                }
            }
            pSubSelect->trackID = trackId;
        } else {
            pSubSelect = common->ppSubtitleSelectList[subIdx];
        }
        pSubSelect->lang = lang;
        func_set(pSubSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < common->nSubtitleSelectCount; itrack++) {
                func_set(common->ppSubtitleSelectList[itrack], trackId, ptr);
            }
        }

        if (subIdx < 0) {
            subIdx = common->nSubtitleSelectCount;
            //新たに要素を追加
            common->ppSubtitleSelectList = (SubtitleSelect **)realloc(common->ppSubtitleSelectList, sizeof(common->ppSubtitleSelectList[0]) * (common->nSubtitleSelectCount + 1));
            common->ppSubtitleSelectList[common->nSubtitleSelectCount] = pSubSelect;
            common->nSubtitleSelectCount++;
        }
        return 0;
    };
    auto set_data_prm = [&](std::function<void(DataSelect *pSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        if (i + 1 < nArgNum) {
            if (strInput[i + 1][0] != _T('-') && strInput[i + 1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim + 1;
            }
            if (ptrDelim != nullptr) {
                const tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                try {
                    trackId = std::stoi(temp);
                } catch (...) {
                    auto tempc = tchar_to_string(temp);
                    if (rgy_lang_exist(tempc)) {
                        trackId = TRACK_SELECT_BY_LANG;
                        lang = tempc;
                    } else if (avcodec_exists(tempc, AVMEDIA_TYPE_DATA)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    }
                }
            }
        }
        DataSelect *pSelect = nullptr;
        int dataIdx = getDataTrackIdx(common, trackId, lang, selectCodec);
        if (dataIdx < 0) {
            pSelect = new DataSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                DataSelect *pDataSelectAll = nullptr;
                for (int itrack = 0; itrack < common->nDataSelectCount; itrack++) {
                    if (common->ppDataSelectList[itrack]->trackID == 0) {
                        pDataSelectAll = common->ppDataSelectList[itrack];
                    }
                }
                if (pDataSelectAll) {
                    *pSelect = *pDataSelectAll;
                }
            }
            pSelect->trackID = trackId;
        } else {
            pSelect = common->ppDataSelectList[dataIdx];
        }
        pSelect->lang = lang;
        func_set(pSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < common->nDataSelectCount; itrack++) {
                func_set(common->ppDataSelectList[itrack], trackId, ptr);
            }
        }

        if (dataIdx < 0) {
            dataIdx = common->nDataSelectCount;
            //新たに要素を追加
            common->ppDataSelectList = (DataSelect **)realloc(common->ppDataSelectList, sizeof(common->ppDataSelectList[0]) * (common->nDataSelectCount + 1));
            common->ppDataSelectList[common->nDataSelectCount] = pSelect;
            common->nDataSelectCount++;
        }
        return 0;
    };
    if (IS_OPTION("audio-copy") || IS_OPTION("copy-audio")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        using trackID_Lang = std::pair<int, std::string>;
        std::set<trackID_Lang> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    if (rgy_lang_exist(tchar_to_string(str))) {
                        trackSet.insert(std::make_pair(TRACK_SELECT_BY_LANG, tchar_to_string(str)));
                    } else if (avcodec_exists(tchar_to_string(str), AVMEDIA_TYPE_AUDIO)) {
                        trackSet.insert(std::make_pair(TRACK_SELECT_BY_CODEC, tchar_to_string(str)));
                    } else {
                        print_cmd_error_invalid_value(option_name, strInput[i]);
                        return 1;
                    }
                } else {
                    trackSet.insert(std::make_pair(iTrack, ""));
                }
            }
        } else {
            trackSet.insert(std::make_pair(0, ""));
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            AudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(common, it->first, it->second, it->second);
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = it->first;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->lang = it->second;
            pAudioSelect->encCodec = RGY_AVCODEC_COPY;

            if (audioIdx < 0) {
                audioIdx = common->nAudioSelectCount;
                //新たに要素を追加
                common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
                common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
                common->nAudioSelectCount++;
            }
            argData->nParsedAudioCopy++;
        }
        return 0;
    }
    if (IS_OPTION("audio-codec")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pAudioSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pAudioSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pAudioSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pAudioSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("audio-profile")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodecProfile.length() == 0) {
                pAudioSelect->encCodecProfile = prmstr;
            }
        });
        if (ret) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("audio-bitrate")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encBitrate == 0) {
                    pAudioSelect->encBitrate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-delay")) {
        try {
            auto ret = set_audio_prm([](AudioSelect* pAudioSelect, int trackId, const TCHAR* prmstr) {
                if (trackId != 0 || pAudioSelect->addDelayMs == 0) {
                    pAudioSelect->addDelayMs = std::stod(prmstr);
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-ignore-decode-error")) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->audioIgnoreDecodeError = value;
        return 0;
    }
    //互換性のため残す
    if (IS_OPTION("audio-ignore-notrack-error")) {
        return 0;
    }
    if (IS_OPTION("video-ignore-timestamp-error")) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->videoIgnoreTimestampError = value;
        return 0;
    }
    if (IS_OPTION("audio-samplerate")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encSamplingRate == 0) {
                    pAudioSelect->encSamplingRate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-resampler")) {
        i++;
        int v = 0;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            common->audioResampler = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            common->audioResampler = v;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_resampler);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("audio-stream")) {

        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || (pAudioSelect->streamChannelSelect[0].empty() && pAudioSelect->streamChannelOut[0].empty())) {
                    auto streamSelectList = split(tchar_to_string(prmstr), ",");
                    if (streamSelectList.size() > pAudioSelect->streamChannelSelect.size()) {
                        return 1;
                    }
                    static const char *DELIM = ":";
                    for (uint32_t j = 0; j < streamSelectList.size(); j++) {
                        auto selectPtr = streamSelectList[j].c_str();
                        auto selectDelimPos = strstr(selectPtr, DELIM);
                        if (selectDelimPos == nullptr) {
                            pAudioSelect->streamChannelSelect[j] = selectPtr;
                            pAudioSelect->streamChannelOut[j]    = RGY_CHANNEL_AUTO; //自動
                        } else if (selectPtr == selectDelimPos) {
                            pAudioSelect->streamChannelSelect[j] = RGY_CHANNEL_AUTO;
                            pAudioSelect->streamChannelOut[j]    = selectDelimPos + strlen(DELIM);
                        } else {
                            pAudioSelect->streamChannelSelect[j] = streamSelectList[j].substr(0, selectDelimPos - selectPtr);
                            pAudioSelect->streamChannelOut[j]    = selectDelimPos + strlen(DELIM);
                        }
                    }
                }
                return 0;
            });
            if (ret) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("Too much streams splitted."));
                return ret;
            }
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-filter")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->filter.length() == 0) {
                    pAudioSelect->filter = prmstr;
                }
            });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-bsf")) {
        try {
            auto ret = set_audio_prm([](AudioSelect* pAudioSelect, int trackId, const TCHAR* prmstr) {
                if (trackId != 0 || pAudioSelect->bsf.length() == 0) {
                    pAudioSelect->bsf = prmstr;
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-disposition")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->disposition.length() == 0) {
                    pAudioSelect->disposition = prmstr;
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-metadata")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->metadata.size() == 0) {
                    pAudioSelect->metadata.push_back(prmstr);
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
#endif //#if ENABLE_AVCODEC_QSV_READER
    if (IS_OPTION("chapter-copy") || IS_OPTION("copy-chapter")) {
        common->copyChapter = true;
        return 0;
    }
    if (IS_OPTION("chapter")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->chapterFile = strInput[i];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("chapter-no-trim")) {
        common->chapterNoTrim = true;
        return 0;
    }
#if ENABLE_KEYFRAME_INSERT
    if (IS_OPTION("key-on-chapter")) {
        common->keyOnChapter = true;
        return 0;
    }
    if (IS_OPTION("keyfile")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->keyFile = strInput[i];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
#endif // #if ENABLE_KEYFRAME_INSERT
#if ENABLE_AVSW_READER && !FOR_AUO
    if (IS_OPTION("sub-copy") || IS_OPTION("copy-sub")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        const auto paramList = std::vector<std::string>{ "asdata" };
        using trackID_Lang = std::pair<int, std::string>;
        std::map<trackID_Lang, SubtitleSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    if (str == _T("asdata")) {
                        auto track = std::make_pair(0, "");
                        trackSet[track].trackID = iTrack;
                        trackSet[track].encCodec = RGY_AVCODEC_COPY;
                        trackSet[track].asdata = true;
                    } else if (rgy_lang_exist(tchar_to_string(str))) {
                        auto track = std::make_pair(TRACK_SELECT_BY_LANG, tchar_to_string(str));
                        trackSet[track].trackID = TRACK_SELECT_BY_LANG;
                        trackSet[track].encCodec = RGY_AVCODEC_COPY;
                        trackSet[track].lang = tchar_to_string(str);
                    } else if (avcodec_exists(tchar_to_string(str), AVMEDIA_TYPE_SUBTITLE)) {
                        auto track = std::make_pair(TRACK_SELECT_BY_CODEC, tchar_to_string(str));
                        trackSet[track].trackID = TRACK_SELECT_BY_CODEC;
                        trackSet[track].encCodec = RGY_AVCODEC_COPY;
                        trackSet[track].selectCodec = tchar_to_string(str);
                    } else {
                        print_cmd_error_unknown_opt_param(option_name, str, paramList);
                        return 1;
                    }
                } else {
                    auto track = std::make_pair(iTrack, "");
                    trackSet[track].trackID = iTrack;
                    trackSet[track].encCodec = RGY_AVCODEC_COPY;
                    auto options = str.find(_T('?'));
                    if (str.substr(options+1) == _T("asdata")) {
                        trackSet[track].asdata = true;
                    }
                }
            }
        } else {
            auto track = std::make_pair(0, "");
            trackSet[track].trackID = 0;
            trackSet[track].encCodec = RGY_AVCODEC_COPY;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            auto& track = it->first;
            SubtitleSelect *pSubtitleSelect = nullptr;
            int subIdx = getSubTrackIdx(common, track.first, track.second, track.second);
            if (subIdx < 0) {
                pSubtitleSelect = new SubtitleSelect();
            } else {
                pSubtitleSelect = common->ppSubtitleSelectList[subIdx];
            }
            pSubtitleSelect[0] = it->second;

            if (subIdx < 0) {
                subIdx = common->nSubtitleSelectCount;
                //新たに要素を追加
                common->ppSubtitleSelectList = (SubtitleSelect **)realloc(common->ppSubtitleSelectList, sizeof(common->ppSubtitleSelectList[0]) * (common->nSubtitleSelectCount + 1));
                common->ppSubtitleSelectList[common->nSubtitleSelectCount] = pSubtitleSelect;
                common->nSubtitleSelectCount++;
            }
        }
        return 0;
    }
    if (IS_OPTION("sub-codec")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_sub_prm([](SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pSubSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pSubSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pSubSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pSubSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pSubSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("sub-disposition")) {
        try {
            auto ret = set_sub_prm([](SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pSubSelect->disposition.length() == 0) {
                    pSubSelect->disposition = prmstr;
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("sub-metadata")) {
        try {
            auto ret = set_sub_prm([](SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pSubSelect->metadata.size() == 0) {
                    pSubSelect->metadata.push_back(prmstr);
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("sub-source")) {
        i++;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        SubSource src;
        const TCHAR *ptr = strInput[i];
        const TCHAR *qtr = _tcsrchr(ptr, _T(':'));
#if defined(_WIN32) || defined(_WIN64)
        if (qtr
            && (qtr - ptr == 1 || qtr - ptr == 2)
            && qtr[1] == _T('\\')
            && _istalpha(qtr[-1])
            && (qtr - ptr == 1 || qtr[-2] == _T('\"'))) {
            qtr = _tcsrchr(qtr + 1, _T(':'));
        }
#endif
        if (qtr == nullptr) {
            src.filename = strInput[i];
            src.select[0].encCodec = RGY_AVCODEC_COPY;
            common->subSource.push_back(src);
            return 0;
        }
        src.filename = tstring(strInput[i]).substr(0, qtr - ptr);
        const auto paramList = std::vector<std::string>{ "codec", "enc_prm", "copy", "disposition", "select-codec", "bsf" };
        auto channel_select_list = split(qtr+1, _T("/"));

        for (size_t ichannel = 0; ichannel < channel_select_list.size(); ichannel++) {
            auto& channel = channel_select_list[ichannel];
            {
                auto option_split = channel.find(_T('='));
                if (option_split != std::string::npos) {
                    if (channel.substr(0, option_split) == _T("format")) {
                        src.format = channel.substr(option_split + 1);
                        continue;
                    } else if (channel.substr(0, option_split) == _T("input_opt")) {
                        src.inputOpt.push_back(std::make_pair<tstring, tstring>(tstring(channel.substr(option_split + 1)), tstring(channel_select_list[ichannel + 1])));
                        ichannel++;
                        continue;
                    }
                }
            }
            int trackId = 0;
            auto channel_id_split = channel.find(_T('?'));
            if (channel_id_split != std::string::npos) {
                try {
                    trackId = std::stoi(channel.substr(0, channel_id_split));
                } catch (...) {
                    print_cmd_error_invalid_value(option_name, strInput[i], _T("invalid track ID."));
                    return 1;
                }
                channel = channel.substr(channel_id_split+1);
            }
            SubtitleSelect &chSel = src.select[trackId];
            chSel.trackID = trackId;
            int ret = parse_one_subtitle_param(chSel, channel, option_name);
            if (ret != 0) return ret;
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
        }
        common->subSource.push_back(src);
        return 0;
    }
    if (IS_OPTION("sub-bsf")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_sub_prm([](SubtitleSelect* pSubSelect, int trackId, const TCHAR* prmstr) {
            if (trackId != 0 || pSubSelect->bsf.length() == 0) {
                pSubSelect->bsf = prmstr;
            }
            });
        if (ret) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("data-copy")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        using trackID_Lang = std::pair<int, std::string>;
        std::map<trackID_Lang, DataSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    if (rgy_lang_exist(tchar_to_string(str))) {
                        auto track = std::make_pair(TRACK_SELECT_BY_LANG, tchar_to_string(str));
                        trackSet[track].trackID = TRACK_SELECT_BY_LANG;
                        trackSet[track].lang = tchar_to_string(str);
                    } else if (avcodec_exists(tchar_to_string(str), AVMEDIA_TYPE_DATA)) {
                        auto track = std::make_pair(TRACK_SELECT_BY_CODEC, tchar_to_string(str));
                        trackSet[track].trackID = TRACK_SELECT_BY_CODEC;
                        trackSet[track].selectCodec = tchar_to_string(str);
                    } else {
                        print_cmd_error_invalid_value(option_name, strInput[i], _T("invalid track ID."));
                        return 1;
                    }
                } else {
                    auto track = std::make_pair(iTrack, "");
                    trackSet[track].trackID = iTrack;
                }
            }
        } else {
            auto track = std::make_pair(0, "");
            trackSet[track].trackID = 0;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            const auto track = it->first;
            DataSelect *pDataSelect = nullptr;
            int dataIdx = getDataTrackIdx(common, track.first, track.second, track.second);
            if (dataIdx < 0) {
                pDataSelect = new DataSelect();
            } else {
                pDataSelect = common->ppDataSelectList[dataIdx];
            }
            pDataSelect[0] = it->second;

            if (dataIdx < 0) {
                dataIdx = common->nDataSelectCount;
                //新たに要素を追加
                common->ppDataSelectList = (DataSelect **)realloc(common->ppDataSelectList, sizeof(common->ppDataSelectList[0]) * (common->nDataSelectCount + 1));
                common->ppDataSelectList[common->nDataSelectCount] = pDataSelect;
                common->nDataSelectCount++;
            }
        }
        return 0;
    }
    if (IS_OPTION("data-disposition")) {
        try {
            auto ret = set_data_prm([](DataSelect *pDataSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pDataSelect->disposition.length() == 0) {
                    pDataSelect->disposition = prmstr;
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("data-metadata")) {
        try {
            auto ret = set_data_prm([](DataSelect *pDataSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pDataSelect->metadata.size() == 0) {
                    pDataSelect->metadata.push_back(prmstr);
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("attachment-copy")) {
        common->AVMuxTarget |= RGY_MUX_VIDEO;
        std::map<int, DataSelect> trackSet; //重複しないように
        if (i + 1 < nArgNum && (strInput[i + 1][0] != _T('-') && strInput[i + 1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    print_cmd_error_invalid_value(option_name, strInput[i], _T("invalid track ID."));
                    return 1;
                } else {
                    trackSet[iTrack].trackID = iTrack;
                }
            }
        } else {
            trackSet[0].trackID = 0;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = it->first;
            AttachmentSelect *pAttachmentSelect = nullptr;
            int dataIdx = getAttachmentTrackIdx(common, trackId);
            if (dataIdx < 0) {
                pAttachmentSelect = new AttachmentSelect();
            } else {
                pAttachmentSelect = common->ppAttachmentSelectList[dataIdx];
            }
            pAttachmentSelect[0] = it->second;

            if (dataIdx < 0) {
                dataIdx = common->nAttachmentSelectCount;
                //新たに要素を追加
                common->ppAttachmentSelectList = (AttachmentSelect **)realloc(common->ppAttachmentSelectList, sizeof(common->ppAttachmentSelectList[0]) * (common->nAttachmentSelectCount + 1));
                common->ppAttachmentSelectList[common->nAttachmentSelectCount] = pAttachmentSelect;
                common->nAttachmentSelectCount++;
            }
        }
        return 0;
    }
    if (IS_OPTION("attachment-source")) {
        i++;
        common->AVMuxTarget |= RGY_MUX_VIDEO;
        SubSource src;
        const TCHAR *ptr = strInput[i];
        const TCHAR *qtr = _tcsrchr(ptr, _T(':'));
#if defined(_WIN32) || defined(_WIN64)
        if (qtr
            && (qtr - ptr == 1 || qtr - ptr == 2)
            && qtr[1] == _T('\\')
            && _istalpha(qtr[-1])
            && (qtr - ptr == 1 || qtr[-2] == _T('\"'))) {
            qtr = _tcsrchr(qtr + 1, _T(':'));
        }
#endif
        if (qtr == nullptr) {
            src.filename = strInput[i];
            src.select[0].encCodec = RGY_AVCODEC_COPY;
            common->attachmentSource.push_back(src);
            return 0;
        }
        src.filename = tstring(strInput[i]).substr(0, qtr - ptr);
        auto channel_select_list = split(qtr + 1, _T(":"));
        for (auto channel : channel_select_list) {
            int trackId = 0;
            auto channel_id_split = channel.find(_T('?'));
            if (channel_id_split != std::string::npos) {
                try {
                    trackId = std::stoi(channel.substr(0, channel_id_split));
                } catch (...) {
                    print_cmd_error_invalid_value(option_name, strInput[i], _T("invalid track ID."));
                    return 1;
                }
                channel = channel.substr(channel_id_split + 1);
            }
            SubtitleSelect &chSel = src.select[trackId];
            chSel.trackID = trackId;
            int ret = parse_one_subtitle_param(chSel, channel, option_name);
            if (ret != 0) return ret;
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
        }
        common->attachmentSource.push_back(src);
        return 0;
    }
#endif //#if ENABLE_AVSW_READER
    if (IS_OPTION("avsync")) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            common->AVSyncMode = (RGYAVSync)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_avsync);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("timestamp-passthrough")) {
        common->timestampPassThrough = true;
        common->AVSyncMode = RGY_AVSYNC_VFR;
        return 0;
    }
    if (IS_OPTION("input-option")) {
        if (i + 1 < nArgNum && strInput[i + 1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            } else {
                common->inputOpt.push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr + 1)));
            }
        } else {
            print_cmd_error_invalid_value(option_name, _T(""));
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("mux-option")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            } else {
                common->muxOpt.push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            print_cmd_error_invalid_value(option_name, _T(""));
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("metadata")) {
        if (i + 1 < nArgNum && strInput[i + 1][0] != _T('-')) {
            i++;
            common->formatMetadata.push_back(strInput[i]);
        } else {
            print_cmd_error_invalid_value(option_name, _T(""));
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("no-mp4opt")) {
        common->disableMp4Opt = true;
        return 0;
    }
    if (IS_OPTION("fullrange") || IS_OPTION("fullrange:h264") || IS_OPTION("fullrange:hevc")) {
        common->out_vui.colorrange = RGY_COLORRANGE_FULL;
        return 0;
    }
    if (IS_OPTION("colorrange")) {
        i++;
        int value = 0;
        if (get_list_value(list_colorrange, strInput[i], &value)) {
            common->out_vui.colorrange = (CspColorRange)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_colorrange);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("videoformat") || IS_OPTION("videoformat:h264") || IS_OPTION("videoformat:hevc")) {
        i++;
        int value = 0;
        if (get_list_value(list_videoformat, strInput[i], &value)) {
            common->out_vui.format = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_videoformat);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("colormatrix") || IS_OPTION("colormatrix:h264") || IS_OPTION("colormatrix:hevc")) {
        i++;
        int value = 0;
        if (get_list_value(list_colormatrix, strInput[i], &value)) {
            common->out_vui.descriptpresent = 1;
            common->out_vui.matrix = (CspMatrix)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_colormatrix);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("colorprim") || IS_OPTION("colorprim:h264") || IS_OPTION("colorprim:hevc")) {
        i++;
        int value = 0;
        if (get_list_value(list_colorprim, strInput[i], &value)) {
            common->out_vui.descriptpresent = 1;
            common->out_vui.colorprim = (CspColorprim)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_colorprim);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("transfer") || IS_OPTION("transfer:h264") || IS_OPTION("transfer:hevc")) {
        i++;
        int value = 0;
        if (get_list_value(list_transfer, strInput[i], &value)) {
            common->out_vui.descriptpresent = 1;
            common->out_vui.transfer = (CspTransfer)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_transfer);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("chromaloc") || IS_OPTION("chromaloc:h264") || IS_OPTION("chromaloc:hevc")) {
        i++;
        int value = 0;
        if (get_list_value(list_chromaloc, strInput[i], &value)) {
            common->out_vui.chromaloc = (CspChromaloc)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_chromaloc);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("max-cll")) {
        i++;
        common->maxCll = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("master-display")) {
        i++;
        common->masterDisplay = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("atc-sei")) {
        i++;
        int value = 0;
        if (get_list_value(list_transfer, strInput[i], &value)) {
            common->atcSei = (CspTransfer)value;
        } else if (_stscanf_s(strInput[i], _T("%d"), &value) == 1) {
            common->atcSei = (CspTransfer)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_transfer);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("dhdr10-info")) {
        i++;
        if (strInput[i] == tstring(_T("copy"))) {
            common->hdr10plusMetadataCopy = true;
        } else {
            common->dynamicHdr10plusJson = strInput[i];
        }
        return 0;
    }
#if ENABLE_DOVI_METADATA_OPTIONS
    if (IS_OPTION("dolby-vision-profile")) {
        i++;
        int value = 0;
        if (get_list_value(list_dovi_profile, strInput[i], &value)) {
            common->doviProfile = value;
        } else if (_stscanf_s(strInput[i], _T("%d"), &value) == 1) {
            common->doviProfile = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_colorprim);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("dolby-vision-rpu")) {
        i++;
        common->doviRpuFile = strInput[i];
        return 0;
    }
#endif //#if ENABLE_DOVI_METADATA_OPTIONS
    if (IS_OPTION("timecode")) {
        common->timecode = true;
        if (i + 1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->timecodeFile = strInput[i];
        }
        return 0;
    }
    if (IS_OPTION("no-timecode")) {
        common->timecode = false;
        if (i + 1 < nArgNum && strInput[i + 1][0] != _T('-')) {
            i++;
            common->timecodeFile = strInput[i];
        }
        return 0;
    }
    if (IS_OPTION("tcfile-in")) {
        i++;
        common->tcfileIn = strInput[i];
        return 0;
    }
    if (IS_OPTION("timebase")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            common->timebase = rgy_rational<int>(a[0], a[1]);
        } else {
            double d = 0.0;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    common->timebase = rgy_rational<int>(rate, 1001);
                } else {
                    common->timebase = rgy_rational<int>((int)(d * 100000 + 0.5), 100000);
                }
            } else {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-hevc-bsf")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_bsf_mode, strInput[i], &value)) {
            common->hevcbsf = (RGYHEVCBsf)value;
        }
        else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_bsf_mode);
            return -1;
        }
        return 0;
    }
#if !ENCODER_MPP
    if (IS_OPTION("ssim")) {
        common->metric.ssim = true;
        return 0;
    }
    if (IS_OPTION("no-ssim")) {
        common->metric.ssim = false;
        return 0;
    }
    if (IS_OPTION("psnr")) {
        common->metric.psnr = true;
        return 0;
    }
    if (IS_OPTION("no-psnr")) {
        common->metric.psnr = false;
        return 0;
    }
#endif
#if ENABLE_VMAF
    if (IS_OPTION("no-vmaf")) {
        common->metric.vmaf.enable = false;
        return 0;
    }
    if (IS_OPTION("vmaf")) {
        common->metric.vmaf.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "model", "threads", "subsample", "phone_model", "enable_transform" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vmaf.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("model")) {
                    common->metric.vmaf.model = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("threads")) {
                    try {
                        common->metric.vmaf.threads = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("subsample")) {
                    try {
                        common->metric.vmaf.subsample = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("phone_model")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vmaf.phone_model = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("enable_transform")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vmaf.enable_transform = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("enable_transform")) {
                    common->metric.vmaf.enable_transform = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
#endif
    if (IS_OPTION("disable-av1-write-parser")) {
        common->debugDirectAV1Out = true;
        return 0;
    }
    if (IS_OPTION("debug-raw-out")) {
        common->debugRawOut = true;
        return 0;
    }
    if (IS_OPTION("allow-other-negative-pts")) {
        common->allowOtherNegativePts = true;
        return 0;
    }
    if (IS_OPTION("out-replay")) {
        i++;
        common->outReplayFile = strInput[i];
        return 0;
    }
    if (IS_OPTION("out-replay-codec")) {
        i++;
        int value = 0;
        if (get_list_value(list_rgy_codec, strInput[i], &value)) {
            common->outReplayCodec = (RGY_CODEC)value;
        }
        else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_rgy_codec);
            return 1;
        }
        return 0;
    }
    return -10;
}

int parse_one_ctrl_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamControl *ctrl, sArgsData *argData) {
    if (IS_OPTION("log")) {
        i++;
        ctrl->logfile = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-opt")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "addtime", "framelist", "packets" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("addtime")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logAddTime = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("framelist")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logFramePosList.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("packets")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logPacketsList.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mux-ts")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logMuxVidTs.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("addtime")) {
                    ctrl->logAddTime = true;
                    continue;
                } else if (param == _T("framelist")) {
                    ctrl->logFramePosList.enable = true;
                    continue;
                } else if (param == _T("packets")) {
                    ctrl->logPacketsList.enable = true;
                    continue;
                } else if (param == _T("mux-ts")) {
                    ctrl->logMuxVidTs.enable = true;
                    continue;
                } else {
                    print_cmd_error_unknown_opt_param(option_name, param, paramList);
                    return 1;
                }
            }
        }
        return 0;
    }
    if (IS_OPTION("log-level")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        return parse_log_level_param(option_name, strInput[i], ctrl->loglevel);
    }
    if (IS_OPTION("log-framelist")) {
        ctrl->logFramePosList.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        ctrl->logFramePosList.filename = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-packets")) {
        ctrl->logPacketsList.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        ctrl->logPacketsList.filename = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-mux-ts")) {
        ctrl->logMuxVidTs.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        ctrl->logMuxVidTs.filename = strInput[i];
        return 0;
    }
    if (IS_OPTION("max-procfps")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return -1;
        }
        if (value < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return -0;
        }
        ctrl->procSpeedLimit = (std::min)(value, std::numeric_limits<decltype(ctrl->procSpeedLimit)>::max());
        return 0;
    }
    if (IS_OPTION("lowlatency")) {
        ctrl->lowLatency = true;
        return 0;
    }
    if (IS_OPTION("input-thread") || IS_OPTION("thread-input")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("shoule be 0 or 1"));
            return 1;
        }
        ctrl->threadInput = value;
        return 0;
    }
    if (IS_OPTION("no-output-thread")) {
        ctrl->threadOutput = 0;
        return 0;
    }
    if (IS_OPTION("output-thread") || IS_OPTION("thread-output")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("shoule be 0 or 1"));
            return 1;
        }
        ctrl->threadOutput = value;
        return 0;
    }
    if (IS_OPTION("audio-thread") || IS_OPTION("thread-audio")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 4) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("shoule be in range: 0 - 3"));
            return 1;
        }
        ctrl->threadAudio = value;
        return 0;
    }
    if (IS_OPTION("thread-affinity")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        std::array<CX_DESC, RGY_THREAD_AFFINITY_MODE_STR.size() + 1> list_thread_affinity_mode;
        for (size_t ia = 0; ia < RGY_THREAD_AFFINITY_MODE_STR.size(); ia++) {
            list_thread_affinity_mode[ia].value = (int)RGY_THREAD_AFFINITY_MODE_STR[ia].second;
            list_thread_affinity_mode[ia].desc = RGY_THREAD_AFFINITY_MODE_STR[ia].first;
        }
        list_thread_affinity_mode[RGY_THREAD_AFFINITY_MODE_STR.size()].value = 0;
        list_thread_affinity_mode[RGY_THREAD_AFFINITY_MODE_STR.size()].desc = nullptr;

        auto parse_val = [option_name, &list_thread_affinity_mode](RGYThreadAffinity& affinity, const tstring& param_arg, const tstring& param_val) {
            if (param_val.substr(0, 2) == _T("0x")) {
                try {
                    uint64_t affintyValue = std::strtoull(tchar_to_string(param_val).c_str(), nullptr, 16);
                    affinity = RGYThreadAffinity(RGYThreadAffinityMode::CUSTOM, affintyValue);
                    return 0;
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            }

            uint64_t affintyValue = std::numeric_limits<decltype(affintyValue)>::max();
            auto mode = param_val;
            auto pos = param_val.find_first_of(_T("#"));
            if (pos != std::string::npos) {
                mode = param_val.substr(0, pos);
                affintyValue = 0u;
                for (auto item : split(param_val.substr(pos + 1), _T(":"))) {
                    int v0 = 0, v1 = 0;
                    if (_stscanf_s(item.c_str(), _T("%d-%d"), &v0, &v1) == 2) {
                        for (int id = v0; id <= v1; id++) {
                            affintyValue |= (1llu << id);
                        }
                    } else if (_stscanf_s(item.c_str(), _T("%d"), &v0) == 1) {
                        affintyValue |= (1llu << v0);
                    } else {
                        return 1;
                    }
                }
            }

            const auto affinity_mode = rgy_str_to_thread_affnity_mode(mode.c_str());
            if (affinity_mode != RGYThreadAffinityMode::END) {
                affinity = RGYThreadAffinity(affinity_mode, affintyValue);
            } else {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_thread_affinity_mode.data());
                return 1;
            }
            return 0;
        };

        std::vector<std::string> paramList;
        for (const auto& param : RGY_THREAD_TYPE_STR) {
            paramList.push_back(tchar_to_string(param.second));
        }

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);

                RGYThreadAffinity affinity;
                if (parse_val(affinity, param_arg, tolowercase(param_val)) == 0) {
                    auto type_ret = std::find_if(RGY_THREAD_TYPE_STR.begin(), RGY_THREAD_TYPE_STR.end(), [param_arg](decltype(RGY_THREAD_TYPE_STR[0])& type) {
                        return param_arg == type.second;
                        });
                    if (type_ret != RGY_THREAD_TYPE_STR.end()) {
                        ctrl->threadParams.set(affinity, type_ret->first);
                    } else {
                        print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                        return 1;
                    }
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_thread_affinity_mode.data());
                    return 1;
                }
            } else {
                RGYThreadAffinity affinity;
                if (parse_val(affinity, _T(""), tolowercase(param)) == 0) {
                    ctrl->threadParams.set(affinity, RGYThreadType::ALL);
                } else {
                    print_cmd_error_invalid_value(option_name, strInput[i]);
                    return 1;
                }
            }
        }
        return 0;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (IS_OPTION("thread-priority")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        std::vector<std::string> paramList;
        for (const auto& param : RGY_THREAD_TYPE_STR) {
            paramList.push_back(tchar_to_string(param.second));
        }

        std::array<CX_DESC, RGY_THREAD_PRIORITY_STR.size() + 1> list_thread_priority;
        for (size_t j = 0; j < RGY_THREAD_PRIORITY_STR.size(); j++) {
            list_thread_priority[j].value = (int)RGY_THREAD_PRIORITY_STR[j].first;
            list_thread_priority[j].desc = RGY_THREAD_PRIORITY_STR[j].second;
        }
        list_thread_priority[RGY_THREAD_PRIORITY_STR.size()].value = 0;
        list_thread_priority[RGY_THREAD_PRIORITY_STR.size()].desc = nullptr;

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);

                const RGYThreadPriority priority = rgy_str_to_thread_priority_mode(tolowercase(param_val).c_str());
                if (priority != RGYThreadPriority::Unknwon) {
                    auto type_ret = std::find_if(RGY_THREAD_TYPE_STR.begin(), RGY_THREAD_TYPE_STR.end(), [param_arg](decltype(RGY_THREAD_TYPE_STR[0])& type) {
                        return param_arg == type.second;
                        });
                    if (type_ret != RGY_THREAD_TYPE_STR.end()) {
                        ctrl->threadParams.set(priority, type_ret->first);
                    } else {
                        print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                        return 1;
                    }
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_thread_priority.data());
                    return 1;
                }
            } else {
                const RGYThreadPriority priority = rgy_str_to_thread_priority_mode(tolowercase(param).c_str());
                if (priority != RGYThreadPriority::Unknwon) {
                    ctrl->threadParams.set(priority, RGYThreadType::ALL);
                } else {
                    print_cmd_error_invalid_value(tstring(option_name), param, list_thread_priority.data());
                    return 1;
                }
            }
        }
        return 0;
    }
    if (IS_OPTION("thread-throttling")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        std::vector<std::string> paramList;
        for (const auto& param : RGY_THREAD_TYPE_STR) {
            paramList.push_back(tchar_to_string(param.second));
        }

        std::array<CX_DESC, RGY_THREAD_POWER_THROTTOLING_MODE_STR.size() + 1> list_thread_throttoling;
        for (size_t j = 0; j < RGY_THREAD_POWER_THROTTOLING_MODE_STR.size(); j++) {
            list_thread_throttoling[j].value = (int)RGY_THREAD_POWER_THROTTOLING_MODE_STR[j].first;
            list_thread_throttoling[j].desc = RGY_THREAD_POWER_THROTTOLING_MODE_STR[j].second;
        }
        list_thread_throttoling[RGY_THREAD_POWER_THROTTOLING_MODE_STR.size()].value = 0;
        list_thread_throttoling[RGY_THREAD_POWER_THROTTOLING_MODE_STR.size()].desc = nullptr;

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);

                const RGYThreadPowerThrottlingMode throttling = rgy_str_to_thread_power_throttoling_mode(tolowercase(param_val).c_str());
                if (throttling != RGYThreadPowerThrottlingMode::END) {
                    auto type_ret = std::find_if(RGY_THREAD_TYPE_STR.begin(), RGY_THREAD_TYPE_STR.end(), [param_arg](decltype(RGY_THREAD_TYPE_STR[0])& type) {
                        return param_arg == type.second;
                        });
                    if (type_ret != RGY_THREAD_TYPE_STR.end()) {
                        ctrl->threadParams.set(throttling, type_ret->first);
                    } else {
                        print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                        return 1;
                    }
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_thread_throttoling.data());
                    return 1;
                }
            } else {
                const RGYThreadPowerThrottlingMode throttling = rgy_str_to_thread_power_throttoling_mode(tolowercase(param).c_str());
                if (throttling != RGYThreadPowerThrottlingMode::END) {
                    ctrl->threadParams.set(throttling, RGYThreadType::ALL);
                } else {
                    print_cmd_error_invalid_value(tstring(option_name), param, list_thread_throttoling.data());
                    return 1;
                }
            }
        }
        return 0;
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
    if (IS_OPTION("output-buf")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("--output-buf should be set in positive value."));
            return 1;
        }
        ctrl->outputBufSizeMB = (std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
        return 0;
    }
    if (IS_OPTION("thread-csp")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < -1) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        ctrl->threadCsp = value;
        return 0;
    }
    if (IS_OPTION("simd-csp")) {
        i++;
        uint64_t value = 0;
        if (get_list_value(list_simd, strInput[i], &value)) {
            ctrl->simdCsp = (RGY_SIMD)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_simd);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("avsdll")) {
        i++;
        ctrl->avsdll = strInput[i];
        return 0;
    }
    if (IS_OPTION("perf-monitor")) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            ctrl->perfMonitorSelect = (int)PERF_MONITOR_ALL;
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    print_cmd_error_invalid_value(option_name, item.c_str(), list_pref_monitor);
                    return 1;
                }
                ctrl->perfMonitorSelect |= value;
            }
        }
        return 0;
    }
    if (IS_OPTION("perf-monitor-interval")) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        ctrl->perfMonitorInterval = std::max(50, v);
        return 0;
    }
    if (IS_OPTION("parent-pid")) {
        i++;
        try {
            ctrl->parentProcessID = std::stoul(strInput[i], nullptr, 16);
            if (ctrl->parentProcessID < 0) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("parent-pid should be positive value.\n"));
            }
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("gpu-select")) {
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "cores", "gen", "ve", "gpu" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("cores")) {
                    try {
                        ctrl->gpuSelect.cores = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gen")) {
                    try {
                        ctrl->gpuSelect.gen = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ve")) {
                    try {
                        ctrl->gpuSelect.ve = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gpu")) {
                    try {
                        ctrl->gpuSelect.gpu = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("skip-hwenc-check")) {
        ctrl->skipHWEncodeCheck = true;
        return 0;
    }
    if (IS_OPTION("skip-hwdec-check")) {
        ctrl->skipHWDecodeCheck = true;
        return 0;
    }
    if (IS_OPTION("debug-cmd-parser")) {
        return 0;
    }
    if (IS_OPTION("option-file")) {
        i++;
        return 0;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (IS_OPTION("process-codepage")) {
        i++;
        return 0;
    }
    if (IS_OPTION("process-codepage-applied")) {
        i++;
        return 0;
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    if (IS_OPTION("disable-opencl")) {
        ctrl->enableOpenCL = false;
        return 0;
    }
    if (IS_OPTION("enable-opencl")) {
        ctrl->enableOpenCL = true;
        return 0;
    }
#endif
    return -10;
}

#define OPT_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (param->opt);
#define OPT_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(param->opt);
#define OPT_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (decltype(list->value))(param->opt));
#define OPT_BOOL(str_true, str_false, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << ((param->opt) ? (str_true) : (str_false));

#define OPT_TCHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << (param->opt);
#define OPT_TSTR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << param->opt.c_str();
#define OPT_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt);
#define OPT_STR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt).c_str();
#define OPT_CHAR_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" \"") << (param->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (param->opt.c_str()) << _T("\"");

#define ADD_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (param->opt);
#define ADD_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << (param->opt);
#define ADD_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (int)(param->opt));
#define ADD_BOOL(str, opt) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << ((param->opt) ? (_T("true")) : (_T("false")));
#define ADD_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) tmp << _T(",") << (str) << _T("=") << (param->opt);
#define ADD_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) tmp << _T(",") << (str) << _T("=\"") << (param->opt) << _T("\"");
#define ADD_STR(str, opt) if (param->opt.length() > 0) tmp << _T(",") << (str) << _T("=") << (param->opt.c_str());

#define ADD_FLOAT2(str, prm, def, opt, prec) if ((prm.opt) != (def.opt)) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (prm.opt);
#define ADD_NUM2(str, prm, def, opt) if ((prm.opt) != (def.opt)) tmp << _T(",") << (str) << _T("=") << (prm.opt);
#define ADD_LST2(str, prm, def, opt, list) if ((prm.opt) != (def.opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (int)(prm.opt));
#define ADD_BOOL2(str, prm, def, opt) if ((prm.opt) != (def.opt)) tmp << _T(",") << (str) << _T("=") << ((prm.opt) ? (_T("true")) : (_T("false")));
#define ADD_PATH2(str, prm, opt) if ((prm.opt) && _tcslen(prm.opt)) tmp << _T(",") << (str) << _T("=\"") << (prm.opt) << _T("\"");
#define ADD_STR2(str, prm, opt) if (prm.opt.length() > 0) tmp << _T(",") << (str) << _T("=") << (prm.opt.c_str());

tstring gen_cmd(const VideoInfo *param, const VideoInfo *defaultPrm, const RGYParamInput *inprm, const RGYParamInput *inprmDefault, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    switch (param->type) {
    case RGY_INPUT_FMT_RAW:    cmd << _T(" --raw"); break;
    case RGY_INPUT_FMT_Y4M:    cmd << _T(" --y4m"); break;
    case RGY_INPUT_FMT_AVI:    cmd << _T(" --avi"); break;
    case RGY_INPUT_FMT_AVS:    cmd << _T(" --avs"); break;
    case RGY_INPUT_FMT_VPY:    cmd << _T(" --vpy"); break;
    case RGY_INPUT_FMT_VPY_MT: cmd << _T(" --vpy-mt"); break;
    case RGY_INPUT_FMT_AVHW:   cmd << _T(" --avhw"); break;
    case RGY_INPUT_FMT_AVSW:   cmd << _T(" --avsw"); break;
    default: break;
    }
    if (param->csp != RGY_CSP_NA) {
        OPT_LST(_T("--input-csp"), csp, list_rgy_csp);
    }
    if (save_disabled_prm || param->picstruct != RGY_PICSTRUCT_FRAME) {
        OPT_LST(_T("--interlace"), picstruct, list_interlaced);
    }
    if (cropEnabled(param->crop)) {
        cmd << _T(" --crop ") << param->crop.e.left << _T(",") << param->crop.e.up
            << _T(",") << param->crop.e.right << _T(",") << param->crop.e.bottom;
    }
    if (param->frames > 0) {
        cmd << _T(" --frames ") << param->frames;
    }
    if (param->fpsN * param->fpsD > 0) {
        cmd << _T(" --fps ") << param->fpsN << _T("/") << param->fpsD;
    }
    if (param->srcWidth * param->srcHeight > 0) {
        cmd << _T(" --input-res ") << param->srcWidth << _T("x") << param->srcHeight;
    }
    if (param->dstWidth * param->dstHeight != 0) {
        cmd << _T(" --output-res ") << param->dstWidth << _T("x") << param->dstHeight;
        if (inprm->resizeResMode != inprmDefault->resizeResMode) {
            cmd << _T(",preserve_aspect_ratio=") << get_chr_from_value(list_vpp_resize_res_mode, (int)(inprm->resizeResMode));
        }
    }
    return cmd.str();
}

tstring printTrack(const AudioSelect *sel) {
    return sel->trackID == TRACK_SELECT_BY_LANG ? char_to_tstring(sel->lang) : std::to_tstring(sel->trackID);
};
tstring printTrack(const SubtitleSelect *sel) {
    return sel->trackID == TRACK_SELECT_BY_LANG ? char_to_tstring(sel->lang) : std::to_tstring(sel->trackID);
};
tstring printTrack(const DataSelect *sel) {
    return sel->trackID == TRACK_SELECT_BY_LANG ? char_to_tstring(sel->lang) : std::to_tstring(sel->trackID);
};


tstring gen_cmd(const RGYParamVpp *param, const RGYParamVpp *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    std::basic_stringstream<TCHAR> tmp;

    OPT_LST(_T("--vpp-resize"), resize_algo, list_vpp_resize);
#if ENCODER_QSV
    OPT_LST(_T("--vpp-resize-mode"), resize_mode, list_vpp_resize_mode);
#endif

    if (param->colorspace != defaultPrm->colorspace) {
        tmp.str(tstring());
        if (!param->colorspace.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->colorspace.enable || save_disabled_prm) {
            for (size_t i = 0; i < param->colorspace.convs.size(); i++) {
                auto from = param->colorspace.convs[i].from;
                auto to = param->colorspace.convs[i].to;
                if (from.matrix != to.matrix) {
                    tmp << _T(",matrix=");
                    tmp << get_cx_desc(list_colormatrix, from.matrix);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colormatrix, to.matrix);
                }
                if (from.colorprim != to.colorprim) {
                    tmp << _T(",colorprim=");
                    tmp << get_cx_desc(list_colorprim, from.colorprim);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorprim, to.colorprim);
                }
                if (from.transfer != to.transfer) {
                    tmp << _T(",transfer=");
                    tmp << get_cx_desc(list_transfer, from.transfer);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_transfer, to.transfer);
                }
                if (from.colorrange != to.colorrange) {
                    tmp << _T(",range=");
                    tmp << get_cx_desc(list_colorrange, from.colorrange);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorrange, to.colorrange);
                }
                ADD_BOOL(_T("approx_gamma"), colorspace.convs[i].approx_gamma);
                ADD_BOOL(_T("scene_ref"), colorspace.convs[i].scene_ref);
                ADD_PATH(_T("lut3d"), colorspace.lut3d.table_file.c_str());
                ADD_LST(_T("lut3d_interp"), colorspace.lut3d.interp, list_vpp_colorspace_lut3d_interp);
                ADD_LST(_T("hdr2sdr"), colorspace.hdr2sdr.tonemap, list_vpp_hdr2sdr);
                ADD_FLOAT(_T("ldr_nits"), colorspace.hdr2sdr.ldr_nits, 1);
                ADD_FLOAT(_T("source_peak"), colorspace.hdr2sdr.hdr_source_peak, 1);
                ADD_FLOAT(_T("a"), colorspace.hdr2sdr.hable.a, 3);
                ADD_FLOAT(_T("b"), colorspace.hdr2sdr.hable.b, 3);
                ADD_FLOAT(_T("c"), colorspace.hdr2sdr.hable.c, 3);
                ADD_FLOAT(_T("d"), colorspace.hdr2sdr.hable.d, 3);
                ADD_FLOAT(_T("e"), colorspace.hdr2sdr.hable.e, 3);
                ADD_FLOAT(_T("f"), colorspace.hdr2sdr.hable.f, 3);
                ADD_FLOAT(_T("transition"), colorspace.hdr2sdr.mobius.transition, 3);
                ADD_FLOAT(_T("peak"), colorspace.hdr2sdr.mobius.peak, 3);
                ADD_FLOAT(_T("contrast"), colorspace.hdr2sdr.reinhard.contrast, 3);
                ADD_FLOAT(_T("desat_base"), colorspace.hdr2sdr.desat_base, 3);
                ADD_FLOAT(_T("desat_strength"), colorspace.hdr2sdr.desat_strength, 3);
                ADD_FLOAT(_T("desat_exp"), colorspace.hdr2sdr.desat_exp, 3);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-colorspace ") << tmp.str().substr(1);
        } else if (param->colorspace.enable) {
            cmd << _T(" --vpp-colorspace");
        }
    }
    if (param->delogo != defaultPrm->delogo) {
        tmp.str(tstring());
        if (!param->delogo.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->delogo.enable || save_disabled_prm) {
            ADD_PATH(_T("file"), delogo.logoFilePath.c_str());
            ADD_PATH(_T("select"), delogo.logoSelect.c_str());
            if (param->delogo.posX != defaultPrm->delogo.posX
                || param->delogo.posY != defaultPrm->delogo.posY) {
                tmp << _T(",pos=") << param->delogo.posX << _T("x") << param->delogo.posY;
            }
            ADD_NUM(_T("depth"), delogo.depth);
            ADD_NUM(_T("y"),  delogo.Y);
            ADD_NUM(_T("cb"), delogo.Cb);
            ADD_NUM(_T("cr"), delogo.Cr);
            if (param->delogo.mode == DELOGO_MODE_ADD) {
                ADD_BOOL(_T("add"), delogo.mode);
            } else if (param->delogo.mode == DELOGO_MODE_ADD_MULTI) {
                ADD_BOOL(_T("multi_add"), delogo.mode);
            }
            ADD_BOOL(_T("auto_fade"), delogo.autoFade);
            ADD_BOOL(_T("auto_nr"), delogo.autoNR);
            ADD_NUM(_T("nr_area"), delogo.NRArea);
            ADD_NUM(_T("nr_value"), delogo.NRValue);
            ADD_BOOL(_T("log"), delogo.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-delogo ") << tmp.str().substr(1);
        }
    }
    if (param->afs != defaultPrm->afs) {
        tmp.str(tstring());
        if (!param->afs.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->afs.enable || save_disabled_prm) {
            ADD_NUM(_T("top"), afs.clip.top);
            ADD_NUM(_T("bottom"), afs.clip.bottom);
            ADD_NUM(_T("left"), afs.clip.left);
            ADD_NUM(_T("right"), afs.clip.right);
            ADD_NUM(_T("method_switch"), afs.method_switch);
            ADD_NUM(_T("coeff_shift"), afs.coeff_shift);
            ADD_NUM(_T("thre_shift"), afs.thre_shift);
            ADD_NUM(_T("thre_deint"), afs.thre_deint);
            ADD_NUM(_T("thre_motion_y"), afs.thre_Ymotion);
            ADD_NUM(_T("thre_motion_c"), afs.thre_Cmotion);
            ADD_NUM(_T("level"), afs.analyze);
            ADD_BOOL(_T("shift"), afs.shift);
            ADD_BOOL(_T("drop"), afs.drop);
            ADD_BOOL(_T("smooth"), afs.smooth);
            ADD_BOOL(_T("24fps"), afs.force24);
            ADD_BOOL(_T("tune"), afs.tune);
            ADD_BOOL(_T("rff"), afs.rff);
            ADD_BOOL(_T("timecode"), afs.timecode);
            ADD_BOOL(_T("log"), afs.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-afs ") << tmp.str().substr(1);
        } else if (param->afs.enable) {
            cmd << _T(" --vpp-afs");
        }
    }
    if (param->nnedi != defaultPrm->nnedi) {
        tmp.str(tstring());
        if (!param->nnedi.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nnedi.enable || save_disabled_prm) {
            ADD_LST(_T("field"), nnedi.field, list_vpp_nnedi_field);
            ADD_LST(_T("nns"), nnedi.nns, list_vpp_nnedi_nns);
            ADD_LST(_T("nsize"), nnedi.nsize, list_vpp_nnedi_nsize);
            ADD_LST(_T("quality"), nnedi.quality, list_vpp_nnedi_quality);
            ADD_LST(_T("prec"), nnedi.precision, list_vpp_fp_prec);
            ADD_LST(_T("prescreen"), nnedi.pre_screen, list_vpp_nnedi_pre_screen);
            ADD_LST(_T("errortype"), nnedi.errortype, list_vpp_nnedi_error_type);
            ADD_PATH(_T("weightfile"), nnedi.weightfile.c_str());
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nnedi ") << tmp.str().substr(1);
        } else if (param->nnedi.enable) {
            cmd << _T(" --vpp-nnedi");
        }
    }
    if (param->yadif != defaultPrm->yadif) {
        tmp.str(tstring());
        if (!param->yadif.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->yadif.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), yadif.mode, list_vpp_yadif_mode);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-yadif ") << tmp.str().substr(1);
        } else if (param->yadif.enable) {
            cmd << _T(" --vpp-yadif");
        }
    }
    if (param->rff != defaultPrm->rff) {
        tmp.str(tstring());
        if (!param->rff.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rff.enable || save_disabled_prm) {
            ADD_BOOL(_T("log"), rff.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rff ") << tmp.str().substr(1);
        }
    }

    if (param->decimate != defaultPrm->decimate) {
        tmp.str(tstring());
        if (!param->decimate.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->decimate.enable || save_disabled_prm) {
            ADD_NUM(_T("cycle"), decimate.cycle);
            ADD_NUM(_T("drop"), decimate.drop);
            ADD_FLOAT(_T("thredup"), decimate.threDuplicate, 3);
            ADD_FLOAT(_T("thresc"), decimate.threSceneChange, 2);
            ADD_LST(_T("blockx"), decimate.blockX, list_vpp_decimate_block);
            ADD_LST(_T("blocky"), decimate.blockY, list_vpp_decimate_block);
            ADD_BOOL(_T("pp"), decimate.preProcessed);
            ADD_BOOL(_T("chroma"), decimate.chroma);
            ADD_BOOL(_T("log"), decimate.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-decimate ") << tmp.str().substr(1);
        }
    }
    if (param->mpdecimate != defaultPrm->mpdecimate) {
        tmp.str(tstring());
        if (!param->mpdecimate.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->mpdecimate.enable || save_disabled_prm) {
            ADD_NUM(_T("lo"), mpdecimate.lo);
            ADD_NUM(_T("hi"), mpdecimate.hi);
            ADD_NUM(_T("max"), mpdecimate.max);
            ADD_FLOAT(_T("frac"), mpdecimate.frac, 3);
            ADD_BOOL(_T("log"), decimate.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-mpdecimate ") << tmp.str().substr(1);
        }
    }
    if (param->selectevery != defaultPrm->selectevery) {
        tmp.str(tstring());
        if (!param->selectevery.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->selectevery.enable || save_disabled_prm) {
            ADD_NUM(_T("step"), selectevery.step);
            ADD_NUM(_T("offset"), selectevery.offset);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-select-every ") << tmp.str().substr(1);
        }
    }
    if (param->pad != defaultPrm->pad) {
        tmp.str(tstring());
        if (!param->pad.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->pad.enable || save_disabled_prm) {
            ADD_NUM(_T("r"), pad.right);
            ADD_NUM(_T("l"), pad.left);
            ADD_NUM(_T("t"), pad.top);
            ADD_NUM(_T("b"), pad.bottom);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-pad ") << tmp.str().substr(1);
        } else if (param->pad.enable) {
            cmd << _T(" --vpp-pad");
        }
    }
    if (param->convolution3d != defaultPrm->convolution3d) {
        tmp.str(tstring());
        if (!param->convolution3d.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->convolution3d.enable || save_disabled_prm) {
            ADD_LST(_T("matrix"),    convolution3d.matrix, list_vpp_convolution3d_matrix);
            ADD_BOOL(_T("fast"),     convolution3d.fast);
            ADD_NUM(_T("ythresh"),   convolution3d.threshYspatial);
            ADD_NUM(_T("cthresh"),   convolution3d.threshCspatial);
            ADD_NUM(_T("t_ythresh"), convolution3d.threshYtemporal);
            ADD_NUM(_T("t_cthresh"), convolution3d.threshCtemporal);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-convolution3d ") << tmp.str().substr(1);
        } else if (param->convolution3d.enable) {
            cmd << _T(" --vpp-convolution3d");
        }
    }
    if (param->knn != defaultPrm->knn) {
        tmp.str(tstring());
        if (!param->knn.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->knn.enable || save_disabled_prm) {
            ADD_NUM(_T("radius"), knn.radius);
            ADD_FLOAT(_T("strength"), knn.strength, 3);
            ADD_FLOAT(_T("lerp"), knn.lerpC, 3);
            ADD_FLOAT(_T("th_weight"), knn.weight_threshold, 3);
            ADD_FLOAT(_T("th_lerp"), knn.lerp_threshold, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-knn ") << tmp.str().substr(1);
        } else if (param->knn.enable) {
            cmd << _T(" --vpp-knn");
        }
    }
    if (param->pmd != defaultPrm->pmd) {
        tmp.str(tstring());
        if (!param->pmd.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->pmd.enable || save_disabled_prm) {
            ADD_NUM(_T("apply_count"), pmd.applyCount);
            ADD_FLOAT(_T("strength"), pmd.strength, 3);
            ADD_FLOAT(_T("threshold"), pmd.threshold, 3);
            ADD_NUM(_T("useexp"), pmd.useExp);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-pmd ") << tmp.str().substr(1);
        } else if (param->pmd.enable) {
            cmd << _T(" --vpp-pmd");
        }
    }
    if (param->smooth != defaultPrm->smooth) {
        tmp.str(tstring());
        if (!param->smooth.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->smooth.enable || save_disabled_prm) {
            ADD_NUM(_T("quality"), smooth.quality);
            ADD_NUM(_T("qp"), smooth.qp);
            ADD_LST(_T("prec"), smooth.prec, list_vpp_fp_prec);
            ADD_BOOL(_T("use_qp_table"), smooth.useQPTable);
            ADD_FLOAT(_T("strength"), smooth.strength, 3);
            ADD_FLOAT(_T("threshold"), smooth.threshold, 3);
            ADD_FLOAT(_T("bratio"), smooth.bratio, 3);
            ADD_NUM(_T("max_error"), smooth.maxQPTableErrCount);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-smooth ") << tmp.str().substr(1);
        } else if (param->smooth.enable) {
            cmd << _T(" --vpp-smooth");
        }
    }
    for (size_t i = 0; i < param->subburn.size(); i++) {
        const auto subburnDefault = VppSubburn();
        if (param->subburn[i] != subburnDefault) {
            tmp.str(tstring());
            if (!param->subburn[i].enable && save_disabled_prm) {
                tmp << _T(",enable=false");
            }
            if (param->subburn[i].enable || save_disabled_prm) {
                ADD_NUM2(_T("track"), param->subburn[i], subburnDefault, trackId);
                ADD_PATH2(_T("filename"), param->subburn[i], filename.c_str());
                ADD_STR2(_T("charcode"), param->subburn[i], charcode);
                ADD_LST2(_T("shaping"), param->subburn[i], subburnDefault, assShaping, list_vpp_ass_shaping);
                ADD_FLOAT2(_T("scale"), param->subburn[i], subburnDefault, scale, 4);
                ADD_FLOAT2(_T("transparency"), param->subburn[i], subburnDefault, transparency_offset, 4);
                ADD_FLOAT2(_T("brightness"), param->subburn[i], subburnDefault, brightness, 4);
                ADD_FLOAT2(_T("contrast"), param->subburn[i], subburnDefault, contrast, 4);
                ADD_BOOL2(_T("vid_ts_offset"), param->subburn[i], subburnDefault, vid_ts_offset);
                ADD_FLOAT2(_T("ts_offset"), param->subburn[i], subburnDefault, ts_offset, 4);
                ADD_PATH2(_T("fontsdir"), param->subburn[i], fontsdir.c_str());
                ADD_BOOL2(_T("forced_subs_only"), param->subburn[i], subburnDefault, forced_subs_only);
            }
            if (!tmp.str().empty()) {
                cmd << _T(" --vpp-subburn ") << tmp.str().substr(1);
            } else if (param->subburn[i].enable) {
                cmd << _T(" --vpp-subburn");
            }
        }
    }
    if (param->unsharp != defaultPrm->unsharp) {
        tmp.str(tstring());
        if (!param->unsharp.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->unsharp.enable || save_disabled_prm) {
            ADD_NUM(_T("radius"), unsharp.radius);
            ADD_FLOAT(_T("weight"), unsharp.weight, 3);
            ADD_FLOAT(_T("threshold"), unsharp.threshold, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-unsharp ") << tmp.str().substr(1);
        } else if (param->unsharp.enable) {
            cmd << _T(" --vpp-unsharp");
        }
    }
    if (param->edgelevel != defaultPrm->edgelevel) {
        tmp.str(tstring());
        if (!param->edgelevel.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->edgelevel.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), edgelevel.strength, 3);
            ADD_FLOAT(_T("threshold"), edgelevel.threshold, 3);
            ADD_FLOAT(_T("black"), edgelevel.black, 3);
            ADD_FLOAT(_T("white"), edgelevel.white, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-edgelevel ") << tmp.str().substr(1);
        } else if (param->edgelevel.enable) {
            cmd << _T(" --vpp-edgelevel");
        }
    }
    if (param->warpsharp != defaultPrm->warpsharp) {
        tmp.str(tstring());
        if (!param->warpsharp.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->warpsharp.enable || save_disabled_prm) {
            ADD_FLOAT(_T("threshold"), warpsharp.threshold, 3);
            ADD_NUM(_T("blur"), warpsharp.blur);
            ADD_NUM(_T("type"), warpsharp.type);
            ADD_FLOAT(_T("depth"), warpsharp.depth, 3);
            ADD_NUM(_T("chroma"), warpsharp.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-warpsharp ") << tmp.str().substr(1);
        } else if (param->warpsharp.enable) {
            cmd << _T(" --vpp-warpsharp");
        }
    }
    if (param->curves != defaultPrm->curves) {
        tmp.str(tstring());
        if (!param->curves.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->curves.enable || save_disabled_prm) {
            ADD_LST(_T("preset"), curves.preset, list_vpp_curves_preset);
            ADD_STR(_T("r"), curves.prm.r);
            ADD_STR(_T("g"), curves.prm.g);
            ADD_STR(_T("b"), curves.prm.b);
            ADD_STR(_T("all"), curves.all);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-curves ") << tmp.str().substr(1);
        } else if (param->curves.enable) {
            cmd << _T(" --vpp-curves");
        }
    }
    if (param->tweak != defaultPrm->tweak) {
        tmp.str(tstring());
        if (!param->tweak.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->tweak.enable || save_disabled_prm) {
            ADD_FLOAT(_T("brightness"), tweak.brightness, 3);
            ADD_FLOAT(_T("contrast"), tweak.contrast, 3);
            ADD_FLOAT(_T("gamma"), tweak.gamma, 3);
            ADD_FLOAT(_T("saturation"), tweak.saturation, 3);
            ADD_FLOAT(_T("hue"), tweak.hue, 3);
            ADD_BOOL(_T("swapuv"), tweak.swapuv);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-tweak ") << tmp.str().substr(1);
        } else if (param->tweak.enable) {
            cmd << _T(" --vpp-tweak");
        }
    }
    OPT_LST(_T("--vpp-rotate"), transform.rotate(), list_vpp_rotate);
    if (!param->transform.rotate()) {
        if (param->transform != defaultPrm->transform) {
            tmp.str(tstring());
            if (!param->transform.enable && save_disabled_prm) {
                tmp << _T(",enable=false");
            }
            if (param->transform.enable || save_disabled_prm) {
                ADD_BOOL(_T("flip_x"), transform.flipX);
                ADD_BOOL(_T("flip_y"), transform.flipY);
                ADD_BOOL(_T("transpose"), transform.transpose);
            }
            if (!tmp.str().empty()) {
                cmd << _T(" --vpp-transform ") << tmp.str().substr(1);
            } else if (param->transform.enable) {
                cmd << _T(" --vpp-transform");
            }
        }
    }
    if (param->deband != defaultPrm->deband) {
        tmp.str(tstring());
        if (!param->deband.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->deband.enable || save_disabled_prm) {
            ADD_NUM(_T("range"), deband.range);
            if (param->deband.threY == param->deband.threCb
                && param->deband.threY == param->deband.threCr) {
                ADD_NUM(_T("thre"), deband.threY);
            } else {
                ADD_NUM(_T("thre_y"), deband.threY);
                ADD_NUM(_T("thre_cb"), deband.threCb);
                ADD_NUM(_T("thre_cr"), deband.threCr);
            }
            if (param->deband.ditherY == param->deband.ditherC) {
                ADD_NUM(_T("dither"), deband.ditherY);
            } else {
                ADD_NUM(_T("dither_y"), deband.ditherY);
                ADD_NUM(_T("dither_c"), deband.ditherC);
            }
            ADD_NUM(_T("sample"), deband.sample);
            ADD_BOOL(_T("blurfirst"), deband.blurFirst);
            ADD_BOOL(_T("rand_each_frame"), deband.randEachFrame);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-deband ") << tmp.str().substr(1);
        } else if (param->deband.enable) {
            cmd << _T(" --vpp-deband");
        }
    }
    for (size_t i = 0; i < param->overlay.size(); i++) {
        const auto overlayDefault = VppOverlay();
        if (param->overlay[i] != overlayDefault) {
            tmp.str(tstring());
            if (!param->overlay[i].enable && save_disabled_prm) {
                tmp << _T(",enable=false");
            }
            if (param->overlay[i].enable || save_disabled_prm) {
                ADD_PATH(_T("file"), overlay[i].inputFile.c_str());
                if (   param->overlay[i].posX != overlayDefault.posX
                    || param->overlay[i].posY != overlayDefault.posY) {
                    tmp << _T(",pos=") << param->overlay[i].posX << _T("x") << param->overlay[i].posY;
                }
                if (   param->overlay[i].width  != overlayDefault.width
                    || param->overlay[i].height != overlayDefault.height) {
                    tmp << _T(",size=") << param->overlay[i].width << _T("x") << param->overlay[i].height;
                }
                ADD_FLOAT2(_T("alpha"), param->overlay[i], overlayDefault, alpha, 3);
                ADD_LST2(_T("alpha_mode"), param->overlay[i], overlayDefault, alphaMode, list_vpp_overlay_alpha_mode);
                ADD_FLOAT2(_T("lumakey_threshold"), param->overlay[i], overlayDefault, lumaKey.threshold, 3);
                ADD_FLOAT2(_T("lumakey_tolerance"), param->overlay[i], overlayDefault, lumaKey.tolerance, 3);
                ADD_FLOAT2(_T("lumakey_shoftness"), param->overlay[i], overlayDefault, lumaKey.shoftness, 3);
                ADD_BOOL2(_T("loop"), param->overlay[i], overlayDefault, loop);
            }
            if (!tmp.str().empty()) {
                cmd << _T(" --vpp-overlay ") << tmp.str().substr(1);
            } else if (param->deband.enable) {
                cmd << _T(" --vpp-overlay");
            }
        }
    }
    OPT_BOOL(_T("--vpp-perf-monitor"), _T("--no-vpp-perf-monitor"), checkPerformance);
    return cmd.str();
}

tstring gen_cmd(const RGYParamCommon *param, const RGYParamCommon *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;

    OPT_STR_PATH(_T("-i"), inputFilename);
    OPT_STR_PATH(_T("-o"), outputFilename);

    std::basic_stringstream<TCHAR> tmp;

    OPT_FLOAT(_T("--input-analyze"), demuxAnalyzeSec, 6);
    OPT_NUM(_T("--input-probesize"), demuxProbesize);
    OPT_NUM(_T("--input-retry"), inputRetry);
    if (param->nTrimCount > 0) {
        cmd << _T(" --trim ");
        for (int i = 0; i < param->nTrimCount; i++) {
            if (i > 0) cmd << _T(",");
            cmd << param->pTrimList[i].start << _T(":") << param->pTrimList[i].fin;
        }
    }
    OPT_FLOAT(_T("--seek"), seekSec, 2);
    OPT_FLOAT(_T("--seekto"), seekToSec, 2);
    OPT_TCHAR(_T("--input-format"), AVInputFormat);
    OPT_TSTR(_T("--output-format"), muxOutputFormat);
    OPT_STR(_T("--video-tag"), videoCodecTag);
    for (auto &m : param->videoMetadata) {
        cmd << _T(" --video-metadata ") << m;
    }
    OPT_NUM(_T("--video-track"), videoTrack);
    OPT_NUM(_T("--video-streamid"), videoStreamId);
    for (uint32_t i = 0; i < param->inputOpt.size(); i++) {
        cmd << _T(" --input-option ") << param->inputOpt.at(i).first << _T(":") << param->inputOpt.at(i).second;
    }
    for (uint32_t i = 0; i < param->muxOpt.size(); i++) {
        cmd << _T(" -m ") << param->muxOpt.at(i).first << _T(":") << param->muxOpt.at(i).second;
    }
    tmp.str(tstring());
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec == RGY_AVCODEC_COPY) {
            if (pAudioSelect->trackID == 0) {
                tmp << _T(","); // --audio-copy のみの指定 (トラックIDを省略)
            } else {
                tmp << _T(",") << printTrack(pAudioSelect);
            }
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --audio-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY) {
            cmd << _T(" --audio-codec ") << printTrack(pAudioSelect);
            if (pAudioSelect->encCodec != RGY_AVCODEC_AUTO) {
                cmd << _T("?") << pAudioSelect->encCodec;
            }
            if (pAudioSelect->encCodecPrm.length() > 0) {
                cmd << _T(":") << pAudioSelect->encCodecPrm;
            }
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encCodecProfile.length() > 0) {
            cmd << _T(" --audio-profile ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->encCodecProfile;
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encBitrate > 0) {
            cmd << _T(" --audio-bitrate ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->encBitrate;
        }
    }
#if !FOR_AUO
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        tmp.str(tstring());
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
            if (pAudioSelect->streamChannelSelect[j].empty()) {
                break;
            }
            if (j > 0) tmp << _T(",");
            if (pAudioSelect->streamChannelSelect[j] != RGY_CHANNEL_AUTO) {
                tmp << char_to_tstring(pAudioSelect->streamChannelOut[j]);
            }
            if (pAudioSelect->streamChannelOut[j] != RGY_CHANNEL_AUTO) {
                tmp << _T(":");
                tmp << char_to_tstring(pAudioSelect->streamChannelOut[j]);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << printTrack(pAudioSelect) << _T("?") << tmp.str();
        }
    }
#endif
    tmp.str(tstring());

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encSamplingRate > 0) {
            cmd << _T(" --audio-samplerate ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->encSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), audioResampler, list_resampler);

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->filter.length() > 0) {
            cmd << _T(" --audio-filter ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->filter;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->addDelayMs != 0.0) {
            cmd << _T(" --audio-delay ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->addDelayMs;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->bsf.length() > 0) {
            cmd << _T(" --audio-bsf ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->bsf;
        }
        if (pAudioSelect->disposition.length() > 0) {
            cmd << _T(" --audio-disposition ") << printTrack(pAudioSelect) << _T("?") << pAudioSelect->disposition;
        }
        for (auto &m : pAudioSelect->metadata) {
            cmd << _T(" --audio-metadata ") << printTrack(pAudioSelect) << _T("?") << m;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->extractFilename.length() > 0) {
            cmd << _T(" --audio-file ") << printTrack(pAudioSelect) << _T("?");
            if (pAudioSelect->extractFormat.length() > 0) {
                cmd << pAudioSelect->extractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->extractFilename << _T("\"");
        }
    }
    for (const auto &src : param->audioSource) {
        if (src.filename.length() > 0) {
            cmd << _T(" --audio-source ") << _T("\"") << src.filename << _T("\"");
            auto source_delim = _T(":");
            if (src.format.length() > 0) {
                cmd << source_delim << _T("format=") << src.format;
                source_delim = _T("/");
            }
            for (auto& opt : src.inputOpt) {
                cmd << source_delim << _T("input_opt=") << opt.first << _T("=") << opt.second;
                source_delim = _T("/");
            }
            for (const auto& channel : src.select) {
                cmd << source_delim;
                source_delim = _T("/");
                if (channel.first == TRACK_SELECT_BY_LANG) {
                    cmd << char_to_tstring(channel.second.lang) << _T("?");
                } else if (channel.first > 0) {
                    cmd << channel.first << _T("?");
                }
                const auto &sel = channel.second;
                if (sel.encCodec.length() == 0) {
                    ; //何もしない
                } else if (sel.encCodec == RGY_AVCODEC_COPY) {
                    cmd << _T("copy");
                } else {
                    tmp.str(tstring());
                    tmp << _T(";codec=") << sel.encCodec;
                    if (sel.encBitrate > 0) {
                        tmp << _T(";bitrate=") << sel.encBitrate;
                    }
                    if (sel.addDelayMs > 0) {
                        tmp << _T(";delay=") << sel.addDelayMs;
                    }
                    if (sel.decCodecPrm.length() > 0) {
                        tmp << _T(";dec_prm=") << sel.decCodecPrm;
                    }
                    if (sel.encCodecPrm.length() > 0) {
                        tmp << _T(";enc_prm=") << sel.encCodecPrm;
                    }
                    if (sel.encCodecProfile.length() > 0) {
                        tmp << _T(";profile=") << sel.encCodecProfile;
                    }
                    if (sel.encSamplingRate > 0) {
                        tmp << _T(";samplerate=") << sel.encSamplingRate;
                    }
                    if (sel.filter.length() > 0) {
                        tmp << _T(";filter=") << _T("\"") << sel.filter << _T("\"");
                    }
                    if (sel.disposition.length() > 0) {
                        tmp << _T(";disposition=") << sel.disposition;
                    }
                    for (const auto& metadata : sel.metadata) {
                        tmp << _T(";metadata=") << metadata;
                    }
                    if (sel.bsf.length() > 0) {
                        tmp << _T(";bsf=") << sel.bsf;
                    }
                }
                if (!tmp.str().empty()) {
                    cmd << tmp.str().substr(1);
                }
            }
        }
    }
    OPT_NUM(_T("--audio-ignore-decode-error"), audioIgnoreDecodeError);
    OPT_NUM(_T("--video-ignore-timestamp-error"), videoIgnoreTimestampError);

    tmp.str(tstring());
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        tmp << _T(",") << param->ppSubtitleSelectList[i]->trackID;
        if (param->ppSubtitleSelectList[i]->asdata) {
            tmp << _T("?asdata");
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --sub-copy ") << tmp.str().substr(1);
    }
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        const SubtitleSelect *pSubSelect = param->ppSubtitleSelectList[i];
        if (pSubSelect->disposition.length() > 0) {
            cmd << _T(" --sub-disposition ") << printTrack(pSubSelect) << _T("?") << pSubSelect->disposition;
        }
        for (auto &m : pSubSelect->metadata) {
            cmd << _T(" --sub-metadata ") << printTrack(pSubSelect) << _T("?") << m;
        }
    }
    tmp.str(tstring());
    for (const auto &src : param->subSource) {
        if (src.filename.length() > 0) {
            cmd << _T(" --sub-source ") << _T("\"") << src.filename << _T("\"");
            auto source_delim = _T(":");
            if (src.format.length() > 0) {
                cmd << source_delim << _T("format=") << src.format;
                source_delim = _T("/");
            }
            for (auto& opt : src.inputOpt) {
                cmd << source_delim << _T("input_opt=") << opt.first << _T("=") << opt.second;
                source_delim = _T("/");
            }
            for (const auto& channel : src.select) {
                cmd << source_delim;
                source_delim = _T("/");
                if (channel.first == TRACK_SELECT_BY_LANG) {
                    cmd << char_to_tstring(channel.second.lang) << _T("?");
                } else if (channel.first > 0) {
                    cmd << channel.first << _T("?");
                }
                const auto &sel = channel.second;
                if (sel.encCodec.length() == 0) {
                    ; //何もしない
                } else if (sel.encCodec == RGY_AVCODEC_COPY) {
                    cmd << _T("copy");
                } else {
                    tmp.str(tstring());
                    tmp << _T(";codec=") << sel.encCodec;
                    if (sel.encCodecPrm.length() > 0) {
                        tmp << _T(";prm=") << sel.encCodecPrm;
                    }
                    if (sel.disposition.length() > 0) {
                        tmp << _T(";disposition=") << sel.disposition;
                    }
                    for (const auto& metadata : sel.metadata) {
                        tmp << _T(";metadata=") << metadata;
                    }
                    if (sel.bsf.length() > 0) {
                        tmp << _T(";bsf=") << sel.bsf;
                    }
                }
                if (!tmp.str().empty()) {
                    cmd << tmp.str().substr(1);
                }
            }
        }
    }
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        if (param->ppSubtitleSelectList[i]->bsf.length() > 0) {
            cmd << _T(" --sub-bsf ") << printTrack(param->ppSubtitleSelectList[i]) << _T("?") << param->ppSubtitleSelectList[i]->bsf;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < param->nDataSelectCount; i++) {
        tmp << _T(",") << param->ppDataSelectList[i]->trackID;
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --data-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());
    for (int i = 0; i < param->nDataSelectCount; i++) {
        const DataSelect *pDataSelect = param->ppDataSelectList[i];
        if (pDataSelect->disposition.length() > 0) {
            cmd << _T(" --data-disposition ") << printTrack(pDataSelect) << _T("?") << pDataSelect->disposition;
        }
        for (auto &m : pDataSelect->metadata) {
            cmd << _T(" --data-metadata ") << printTrack(pDataSelect) << _T("?") << m;
        }
    }

    for (int i = 0; i < param->nAttachmentSelectCount; i++) {
        tmp << _T(",") << param->ppAttachmentSelectList[i]->trackID;
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --attachment-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (const auto &src : param->attachmentSource) {
        if (src.filename.length() > 0) {
            cmd << _T(" --attachment-source ") << _T("\"") << src.filename << _T("\"");
            for (const auto &channel : src.select) {
                cmd << _T(":");
                tmp.str(tstring());
                for (const auto& metadata : channel.second.metadata) {
                    tmp << _T(";metadata=") << metadata;
                }
                if (!tmp.str().empty()) {
                    cmd << tmp.str().substr(1);
                }
            }
        }
    }

    OPT_STR_PATH(_T("--chapter"), chapterFile);
    OPT_BOOL(_T("--chapter-copy"), _T(""), copyChapter);
    OPT_BOOL(_T("--chapter-no-trim"), _T(""), chapterNoTrim);
    OPT_BOOL(_T("--key-on-chapter"), _T(""), keyOnChapter);
    OPT_STR_PATH(_T("--keyfile"), keyFile);

    OPT_BOOL(_T("--no-mp4opt"), _T(""), disableMp4Opt);
    OPT_LST(_T("--avsync"), AVSyncMode, list_avsync);
    OPT_BOOL(_T("--timestamp-passthrough"), _T(""), timestampPassThrough);
    for (auto &m : param->formatMetadata) {
        cmd << _T(" --metadata ") << m;
    }

    OPT_LST(_T("--chromaloc"), out_vui.chromaloc, list_chromaloc);
    OPT_LST(_T("--colorrange"), out_vui.colorrange, list_colorrange);
    OPT_LST(_T("--colormatrix"), out_vui.matrix, list_colormatrix);
    OPT_LST(_T("--colorprim"), out_vui.colorprim, list_colorprim);
    OPT_LST(_T("--transfer"), out_vui.transfer, list_transfer);
    OPT_LST(_T("--videoformat"), out_vui.format, list_videoformat);
    OPT_STR(_T("--max-cll"), maxCll);
    OPT_STR(_T("--master-display"), masterDisplay);
    OPT_LST(_T("--atc-sei"), atcSei, list_transfer);
    if (param->hdr10plusMetadataCopy) {
        cmd << _T("--dhdr10-info copy");
    } else {
        OPT_TSTR(_T("--dhdr10-info"), dynamicHdr10plusJson);
    }
    OPT_LST(_T("--dolby-vision-profile"), doviProfile, list_dovi_profile);
    OPT_STR_PATH(_T("--dolby-vision-rpu"), doviRpuFile);
    if (param->timecode || param->timecodeFile.length() > 0) {
        cmd << (param->timecode ? _T("--timecode ") : _T("--no-timecode "));
        if (param->timecodeFile.length() > 0) {
            cmd << param->timecodeFile;
        }
    }

    OPT_LST(_T("--input-hevc-bsf"), hevcbsf, list_hevc_bsf_mode);
    OPT_STR_PATH(_T("--tcfile-in"), tcfileIn);
    if (param->timebase != defaultPrm->timebase) {
        cmd << _T("--timebase ") << param->timebase.n() << _T("/") << param->timebase.d();
    }

    OPT_BOOL(_T("--ssim"), _T("--no-ssim"), metric.ssim);
    OPT_BOOL(_T("--psnr"), _T("--no-psnr"), metric.psnr);

    if (param->metric.vmaf != defaultPrm->metric.vmaf) {
        tmp.str(tstring());
        if (!param->metric.vmaf.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->metric.vmaf.enable || save_disabled_prm) {
            ADD_PATH(_T("model"), metric.vmaf.model.c_str());
            ADD_NUM(_T("threads"), metric.vmaf.threads);
            ADD_NUM(_T("subsample"), metric.vmaf.subsample);
            ADD_BOOL(_T("phone_model"), metric.vmaf.phone_model);
            ADD_BOOL(_T("enable_transform"), metric.vmaf.enable_transform);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vmaf ") << tmp.str().substr(1);
        }
    }
    OPT_BOOL(_T("--allow-other-negative-pts"), _T(""), allowOtherNegativePts);
    OPT_BOOL(_T("--disable-av1-write-parser"), _T("--no-disable-av1-write-parser"), debugDirectAV1Out);
    OPT_BOOL(_T("--debug-raw-out"), _T("--no-debug-raw-out"), debugRawOut);
    return cmd.str();
}

tstring gen_cmd(const RGYParamControl *param, const RGYParamControl *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    OPT_NUM(_T("--output-buf"), outputBufSizeMB);
    OPT_NUM(_T("--thread-output"), threadOutput);
    OPT_NUM(_T("--thread-input"), threadInput);
    OPT_NUM(_T("--thread-audio"), threadAudio);
    OPT_NUM(_T("--thread-csp"), threadCsp);
    if (param->threadParams != defaultPrm->threadParams) {
        cmd << _T(" --thread-affinity ")    << param->threadParams.to_string(RGYParamThreadType::affinity);
        cmd << _T(" --thread-priority ")    << param->threadParams.to_string(RGYParamThreadType::priority);
        cmd << _T(" --thread-throttling ") << param->threadParams.to_string(RGYParamThreadType::throttling);
    }
    OPT_LST(_T("--simd-csp"), simdCsp, list_simd);
    OPT_NUM(_T("--max-procfps"), procSpeedLimit);
    OPT_BOOL(_T("--lowlatency"), _T(""), lowLatency);
    OPT_STR_PATH(_T("--log"), logfile);
    if (param->loglevel != defaultPrm->loglevel) {
        cmd << _T(" --log-level ") << param->loglevel.to_string();
    }

    if (param->logAddTime != defaultPrm->logAddTime) {
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        if (param->logAddTime != defaultPrm->logAddTime) {
            tmp << _T(",addtime");
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --log-opt ") << tmp.str().substr(1);
        }
    }
    if (param->logFramePosList.enable) {
        cmd << _T(" --log-framelist");
        if (param->logFramePosList.filename.length() > 0) {
            cmd << _T(" \"") << param->logFramePosList.filename << _T("\"");
        }
    }
    if (param->logPacketsList.enable) {
        cmd << _T(" --log-packets");
        if (param->logPacketsList.filename.length() > 0) {
            cmd << _T(" \"") << param->logPacketsList.filename << _T("\"");
        }
    }
    if (param->logMuxVidTs.enable) {
        cmd << _T(" --log-mux-ts");
        if (param->logMuxVidTs.filename.length() > 0) {
            cmd << _T(" \"") << param->logMuxVidTs.filename << _T("\"");
        }
    }
    OPT_BOOL(_T("--skip-hwenc-check"), _T(""), skipHWEncodeCheck);
    OPT_BOOL(_T("--skip-hwdec-check"), _T(""), skipHWDecodeCheck);
    OPT_STR_PATH(_T("--avsdll"), avsdll);
    if (param->perfMonitorSelect != defaultPrm->perfMonitorSelect) {
        auto select = (int)param->perfMonitorSelect;
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor");
        } else {
            cmd << _T(" --perf-monitor ") << tmp.str().substr(1);
        }
    }
    OPT_NUM(_T("--perf-monitor-interval"), perfMonitorInterval);
    OPT_NUM(_T("--parent-pid"), parentProcessID);
    if (param->gpuSelect != defaultPrm->gpuSelect) {
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        ADD_FLOAT(_T("cores"), gpuSelect.cores, 6);
        ADD_FLOAT(_T("gen"), gpuSelect.gen, 3);
        ADD_FLOAT(_T("ve"), gpuSelect.ve, 3);
        ADD_FLOAT(_T("gpu"), gpuSelect.gpu, 3);
        if (!tmp.str().empty()) {
            cmd << _T(" --gpu-select ") << tmp.str().substr(1);
        }
    }
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    OPT_BOOL(_T("--enable-opencl"), _T("--disable-opencl"), enableOpenCL);
#endif
    return cmd.str();
}


//適当に改行しながら表示する
tstring print_list(const CX_DESC *list) {
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;

    tstring str = indent_space;
    int line_len = (int)str.length();
    for (int i = 0; list[i].desc; i++) {
        if (line_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            str += strsprintf(_T("\n%s"), indent_space);
            line_len = indent_len;
        } else {
            if (i) {
                str += strsprintf(_T(", "));
                line_len += 2;
            }
        }
        str += strsprintf(_T("%s"), list[i].desc);
        line_len += (int)_tcslen(list[i].desc);
    }
    return str;
}

tstring print_list_options(const TCHAR *option_name, const CX_DESC *list, int default_index) {
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;
    tstring str = strsprintf(_T("   %s "), option_name);
    while ((int)str.length() < indent_len)
        str += _T(" ");
    int line_len = (int)str.length();
    for (int i = 0; list[i].desc; i++) {
        if (line_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            str += strsprintf(_T("\n%s"), indent_space);
            line_len = indent_len;
        } else {
            if (i) {
                str += strsprintf(_T(", "));
                line_len += 2;
            }
        }
        str += strsprintf(_T("%s"), list[i].desc);
        line_len += (int)_tcslen(list[i].desc);
    }
    str += strsprintf(_T("\n%s default: %s\n"), indent_space, list[default_index].desc);
    return str;
}

tstring gen_cmd_help_input() {
    tstring str =
        _T("\n")
        _T("-i,--input <filename>           set input filename\n")
        _T("-o,--output <filename>          set output filename\n")
        _T("\n")
        _T(" Input formats (auto detected from extension of not set)\n")
        _T("   --raw                        set input as raw format\n")
        _T("   --y4m                        set input as y4m format\n")
#if ENABLE_AVI_READER
        _T("   --avi                        set input as avi format\n")
#endif
#if ENABLE_AVISYNTH_READER
        _T("   --avs                        set input as avs format\n")
#endif
#if ENABLE_VAPOURSYNTH_READER
        _T("   --vpy                        set input as vpy format\n")
        _T("   --vpy-mt                     set input as vpy(mt) format\n")
#endif
#if ENABLE_AVSW_READER
        _T("   --avhw                       use libavformat + hw decode for input\n")
        _T("   --avsw                       set input to use avcodec + sw decoder\n")
#endif
        _T("   --input-res <int>x<int>        set input resolution\n")
        _T("   --crop <int>,<int>,<int>,<int> crop pixels from left,top,right,bottom\n")
        _T("                                    left crop is unavailable with avhw reader\n")
        _T("   --output-res <int>x<int>[,<string>=<string>]...\n")
        _T("                                set output resolution\n")
        _T("    params\n")
        _T("      preserve_aspect_ratio=<string>   preserve input aspect ratio.\n")
        _T("        decrease ... preserve aspect ratio by decreasing resolution specified.\n")
        _T("        increase ... preserve aspect ratio by increasing resolution specified.\n")
        _T("\n")
        _T("   --frames <int>               frames to encode (based on input frames)\n")
        _T("   --fps <int>/<int> or <float> set framerate\n")
        _T("   --interlace <string>         set input as interlaced\n")
        _T("                                  tff, bff\n");
    str += print_list_options(_T("--input-csp <string>           set input colorspace for raw reader"),
        list_rgy_csp, get_cx_index(list_rgy_csp, _T("yv12")));
    return str;
}

tstring gen_cmd_help_common() {
    tstring str =
        _T("   --chromaloc <int>            set chroma location flag [ 0 ... 5 ]\n")
        _T("                                  default: 0 = unspecified\n");
    str += print_list_options(_T("--videoformat <string>"), list_videoformat, 0);
    str += print_list_options(_T("--colormatrix <string>"), list_colormatrix, 0);
    str += print_list_options(_T("--colorprim <string>"), list_colorprim, 0);
    str += print_list_options(_T("--transfer <string>"), list_transfer, 0);
    str += print_list_options(_T("--colorrange <string>"), list_colorrange, 0);
#if ENABLE_AVSW_READER
    str += strsprintf(
        _T("   --max-cll <int>,<int>        set MaxCLL/MaxFall in nits. e.g. \"1000,300\"\n")
        _T("   --master-display <string>    set Mastering display data.\n")
        _T("   e.g. \"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)\"\n"));
    str += print_list_options(_T("--atc-sei <string> or <int>"), list_transfer, 1);
    str += strsprintf(
        _T("   --dhdr10-info <string>       apply dynamic HDR10+ metadata from json file.\n")
        _T("   --dhdr10-info copy           Copy dynamic HDR10+ metadata from input file.\n"));
#if ENABLE_DOVI_METADATA_OPTIONS
    str += print_list_options(_T("--dolby-vision-profile <int>"), list_dovi_profile, 0);
    str += strsprintf(
        _T("   --dolby-vision-rpu <string>  Copy dolby vision metadata from input rpu file.\n"));
#endif //#if ENABLE_DOVI_METADATA_OPTIONS
    str += strsprintf(
        _T("   --input-analyze <int>        set time (sec) which reader analyze input file.\n")
        _T("                                 default: 5 (seconds).\n")
        _T("                                 could be only used with avhw/avsw reader.\n")
        _T("                                 use if reader fails to detect audio stream.\n")
        _T("   --input-probesize <int>      set size in bytes which reader analyze input file.\n")
        //_T("   --input-retry <int>          set retry count for openning input file.\n")
        //_T("                                 could useful for streaming input.\n")
        //_T("                                  default: disabled.\n")
        _T("   --video-track <int>          set video track to encode in track id\n")
        _T("                                 1 (default)  highest resolution video track\n")
        _T("                                 2            next high resolution video track\n")
        _T("                                   ... \n")
        _T("                                 -1           lowest resolution video track\n")
        _T("                                 -2           next low resolution video track\n")
        _T("                                   ... \n")
        _T("   --video-streamid <int>       set video track to encode in stream id\n")
        _T("   --video-tag <string>         specify video tag\n")
        _T("   --video-metadata <string>    set metadata for video track.\n")
        _T("                                 - copy ... copy metadata from input\n")
        _T("                                 - clear ... do not set metadata (default)\n")
        _T("   --audio-source <string>      input extra audio file.\n")
        _T("   --audio-file [<int>?][<string>:]<string>\n")
        _T("                                extract audio into file.\n")
        _T("                                 could be only used with avhw/avsw reader.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to extract.\n")
        _T("                                  in [<string>?], specify output format.\n")
        _T("   --trim <int>:<int>[,<int>:<int>]...\n")
        _T("                                trim video for the frame range specified.\n")
        _T("                                 frame range should not overwrap each other.\n")
        _T("   --seek [<int>:][<int>:]<int>[.<int>] (hh:mm:ss.ms)\n")
        _T("                                skip video for the time specified,\n")
        _T("                                 seek will be inaccurate but fast.\n")
        _T("   --seekto [<int>:][<int>:]<int>[.<int>] (hh:mm:ss.ms)\n")
        _T("                                time to end encoding.\n")
        _T("   --input-format <string>      set input format of input file.\n")
        _T("                                 this requires use of avhw/avsw reader.\n")
        _T("-f,--output-format <string>     set output format of output file.\n")
        _T("                                 if format is not specified, output format will\n")
        _T("                                 be guessed from output file extension.\n")
        _T("                                 set \"raw\" for H.264/ES output.\n")
        _T("   --audio-copy [<int>[,...]]   mux audio with video during output.\n")
        _T("                                 could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 by default copies all audio tracks.\n")
        _T("                                 \"--audio-copy 1,2\" will extract\n")
        _T("                                 audio track #1 and #2.\n")
        _T("   --audio-codec [<int>?]<string>\n")
        _T("                                encode audio to specified format.\n")
        _T("                                  in [<int>?], specify track number to encode.\n")
        _T("   --audio-profile [<int>?]<string>\n")
        _T("                                specify audio profile.\n")
        _T("                                  in [<int>?], specify track number to apply.\n")
        _T("   --audio-bitrate [<int>?]<int>\n")
        _T("                                set encode bitrate for audio (kbps).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-ignore-decode-error <int>  (default: %d)\n")
        _T("                                set numbers of continuous packets of audio decode\n")
        _T("                                 error to ignore, replaced by silence.\n")
        _T("   --audio-samplerate [<int>?]<int>\n")
        _T("                                set sampling rate for audio (Hz).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-resampler <string>   set audio resampler.\n")
        _T("                                  swr (swresampler: default), soxr (libsoxr)\n")
        _T("   --audio-delay [<int>?]<float>  set audio delay (ms).\n")
        _T("   --audio-stream [<int>?][<string1>][:<string2>][,[<string1>][:<string2>]][..\n")
        _T("       set audio streams in channels.\n")
        _T("         in [<int>?], specify track number to split.\n")
        _T("         in <string1>, set input channels to use from source stream.\n")
        _T("           if unset, all input channels will be used.\n")
        _T("         in <string2>, set output channels to mix.\n")
        _T("           if unset, all input channels will be copied without mixing.\n")
        _T("       example1: --audio-stream FL,FR\n")
        _T("         splitting dual mono audio to each stream.\n")
        _T("       example2: --audio-stream :stereo\n")
        _T("         mixing input channels to stereo.\n")
        _T("       example3: --audio-stream 5.1,5.1:stereo\n")
        _T("         keeping 5.1ch audio and also adding downmixed stereo stream.\n")
        _T("       usable symbols\n")
        _T("         mono       = FC\n")
        _T("         stereo     = FL + FR\n")
        _T("         2.1        = FL + FR + LFE\n")
        _T("         3.0        = FL + FR + FC\n")
        _T("         3.0(back)  = FL + FR + BC\n")
        _T("         3.1        = FL + FR + FC + LFE\n")
        _T("         4.0        = FL + FR + FC + BC\n")
        _T("         quad       = FL + FR + BL + BR\n")
        _T("         quad(side) = FL + FR + SL + SR\n")
        _T("         5.0        = FL + FR + FC + SL + SR\n")
        _T("         5.1        = FL + FR + FC + LFE + SL + SR\n")
        _T("         6.0        = FL + FR + FC + BC + SL + SR\n")
        _T("         6.0(front) = FL + FR + FLC + FRC + SL + SR\n")
        _T("         hexagonal  = FL + FR + FC + BL + BR + BC\n")
        _T("         6.1        = FL + FR + FC + LFE + BC + SL + SR\n")
        _T("         6.1(front) = FL + FR + LFE + FLC + FRC + SL + SR\n")
        _T("         7.0        = FL + FR + FC + BL + BR + SL + SR\n")
        _T("         7.0(front) = FL + FR + FC + FLC + FRC + SL + SR\n")
        _T("         7.1        = FL + FR + FC + LFE + BL + BR + SL + SR\n")
        _T("         7.1(wide)  = FL + FR + FC + LFE + FLC + FRC + SL + SR\n")
        _T("   --audio-filter [<int>?]<string>\n")
        _T("                                set audio filter.\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-disposition [<int>?]<string>\n")
        _T("                                set disposition for the specified audio track.\n")
        _T("                                disposition for the unspecified tracks will be reset.\n")
        _T("   --audio-metadata [<int>?]<string>\n")
        _T("                                set metadata for the specified audio track.\n")
        _T("                                 - copy ... copy metadata from input (default)\n")
        _T("                                 - clear ... do not set metadata\n")
        _T("   --audio-bsf [<int>?]<string> set bitstream filter to audio track.\n")
        _T("   --chapter-copy               copy chapter to output file.\n")
        _T("   --chapter <string>           set chapter from file specified.\n")
        _T("   --chapter-no-trim            do not apply --trim to --chapter.\n")
#if ENABLE_KEYFRAME_INSERT
        _T("   --key-on-chapter             set key frame on chapter.\n")
        _T("   --keyfile <string>           set keyframes on frames specified in the file.\n")
        _T("                                  frame num should start from 0.\n")
#endif //#if ENABLE_KEYFRAME_INSERT
        _T("   --sub-source <string>        input extra subtitle file.\n")
        _T("   --sub-copy [<int>[,...]]     copy subtitle to output file.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("   --sub-disposition [<int>?]<string>\n")
        _T("                                set disposition for the specified subtitle track.\n")
        _T("                                disposition for the unspecified tracks will be reset.\n")
        _T("   --sub-metadata [<int>?]<string>\n")
        _T("                                set metadata for the specified audio track.\n")
        _T("                                 - copy ... copy metadata from input (default)\n")
        _T("                                 - clear ... do not set metadata\n")
        _T("   --sub-bsf [<int>?]<string>   set bitstream filter to subtitle track.\n")
        _T("   --data-copy [<int>[,...]]       copy data stream to output file.\n")
        _T("   --attachment-copy [<int>[,...]] copy attachment stream to output file.\n")
        _T("   --attachment-source <string> add file as attachment stream.\n")
        _T("\n")
        _T("   --avsync <string>            method for AV sync (default: cfr)\n")
        _T("                                 cfr      ... assume cfr\n")
        _T("                                 forcecfr ... check timestamp and force cfr\n")
        _T("                                 vfr      ... honor source timestamp and enable vfr output.\n")
        _T("                                              only available for avsw/avhw reader,\n")
        _T("                                              and could not be used with --trim.\n")
        _T("  --timestamp-passthrough       passthrough original timestamp\n")
        _T("  --input-option <string1>:<string2>\n")
        _T("                                set input option name and value.\n")
        _T("                                 these could be only used with avhw/avsw reader.\n")
        _T("-m,--mux-option <string1>:<string2>\n")
        _T("                                set muxer option name and value.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("   --metadata <string>          set metadata for output file.\n")
        _T("                                 - copy ... copy metadata from input (default)\n")
        _T("                                 - clear ... do not set metadata\n")
        _T("\n")
        _T("   --timecode [<string>]        output timecode file.\n")
        _T("\n")
        _T("   --tcfile-in <string>         input timecode file, will not work with --avhw.\n")
        _T("   --tc-timebase <int>/<int>    timebase of input timecode.\n")
        _T("\n")
        _T("   --input-hevc-bsf <string>    switch hevc bitstream filter used for hw decoder input\n")
        _T("                                 - internal   ... use internal implementation (default)\n")
        _T("                                 - libavcodec ... use hevc_mp4toannexb bsf\n"),
        DEFAULT_IGNORE_DECODE_ERROR);
    str += _T("\n")
        _T("   --allow-other-negative-pts  for debug\n")
        _T("\n");
#endif
#if !ENCODER_MPP
    str += _T("\n")
        _T("   --ssim                       calc ssim\n")
        _T("   --psnr                       calc psnr\n")
        _T("\n");
#endif //#if !ENCODER_MPP
#if ENABLE_VMAF
    str += strsprintf(_T("")
        _T("   --vmaf [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Calc vmaf. Please note that this is very CPU intensive and likely to \n")
        _T("     become bottleneck, strongly affecting encoding perfromance.\n")
        _T("    params\n")
        _T("      model=<string>            set model version/filepath [default:%s].\n")
        _T("      threads=<int>             cpu thread(s) to calculate vmaf score.\n")
        _T("      subsample=<int>           interval for frame subsampling calculating vmaf score.\n")
        _T("      phone_model=<bool>        use phone model which generate higher vmaf score.\n")
        _T("      enable_transform=<bool>   enable transform when calculating vmaf score.\n"),
        VMAF_DEFAULT_MODEL_VERSION);
#endif //#if ENABLE_VMAF
    return str;
}

tstring gen_cmd_help_vpp() {
    tstring str;
#if ENABLE_VPP_FILTER_COLORSPACE
    str += strsprintf(_T("\n")
        _T("   --vpp-colorspace [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Converts colorspace of the video.\n")
        _T("    params\n")
        _T("      matrix=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR,\n")
        _T("        bt2020nc, bt2020c\n")
        _T("      colorprim=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020\n")
        _T("      transfer=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,\n")
        _T("        log100, log316, iec61966-2-4, iec61966-2-1,\n")
        _T("        bt2020-10, bt2020-12, smpte2084, arib-srd-b67\n")
        _T("      range=<from>:<to>\n")
        _T("        limited, full\n")
        _T("      lut3d=<path>\n")
        _T("      lut3d_interp=<string>\n")
        _T("        nearest, trilinear, tetrahedral, pyramid, prism\n")
        _T("      hdr2sdr=<string>     Enables HDR10 to SDR.\n")
        _T("                             hable, mobius, reinhard, bt2390, none\n")
        _T("      source_peak=<float>     (default: %.1f)\n")
        _T("      ldr_nits=<float>        (default: %.1f)\n")
        _T("      desat_base=<float>      (default: %.2f)\n")
        _T("      desat_strength=<float>  (default: %.2f)\n")
        _T("      desat_exp=<float>       (default: %.2f)\n"),
        FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK,
        FILTER_DEFAULT_COLORSPACE_LDRNITS,
        FILTER_DEFAULT_HDR2SDR_DESAT_BASE,
        FILTER_DEFAULT_HDR2SDR_DESAT_STRENGTH,
        FILTER_DEFAULT_HDR2SDR_DESAT_EXP
    );
#endif
    str += strsprintf(_T("")
        _T("   --vpp-delogo <string>[,<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     remove half-transparent logo with the specified logo file.\n")
        _T("     the logo file should be created by logoscan.auf.\n")
        _T("    params\n")
        _T("      select=<string>           set target logo name or auto select file\n")
        _T("                                 or logo index starting from 1.\n")
        _T("      pos=<int>:<int>           set delogo pos offset.\n")
        _T("      depth=<int>               set delogo depth. [default:%d]\n")
        _T("      y=<int>                   set delogo y  param.\n")
        _T("      cb=<int>                  set delogo cb param.\n")
        _T("      cr=<int>                  set delogo cr param.\n")
#if ENCODER_NVENC
        _T("      auto_fade=<bool>          adjust fade value dynamically.\n")
        _T("      auto_nr=<bool>            adjust strength of noise reduction dynamically.\n")
        _T("      nr_area=<int>             area of noise reduction near logo.\n")
        _T("      nr_value=<int>            strength of noise reduction near logo.\n")
        _T("      log=<bool>                output log for auto_fade/auto_nr.\n")
#endif
        ,
        FILTER_DEFAULT_DELOGO_DEPTH);
#if ENABLE_VPP_FILTER_AFS
    str += strsprintf(_T("\n")
        _T("   --vpp-afs [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable auto field shift deinterlacer\n")
        _T("    params\n")
        _T("      preset=<string>\n")
        _T("          default, triple, double, anime, cinema, min_afterimg,\n")
        _T("          24fps, 24fps_sd, 30fps\n")
        _T("      ini=<string>\n")
        _T("          read setting from ini file specified (output of afs.auf)\n")
        _T("\n")
        _T("      !! params from preset & ini will be overrided by user settings below !!\n")
        _T("\n")
        _T("                   Aviutlでのパラメータ名\n")
        _T("      top=<int>           (上)         clip range to scan (default=%d)\n")
        _T("      bottom=<int>        (下)         clip range to scan (default=%d)\n")
        _T("      left=<int>          (左)         clip range to scan (default=%d)\n")
        _T("      right=<int>         (右)         clip range to scan (default=%d)\n")
        _T("                                        left & right must be muitiple of 4\n")
        _T("      method_switch=<int> (切替点)     (default=%d, 0-256)\n")
        _T("      coeff_shift=<int>   (判定比)     (default=%d, 0-256)\n")
        _T("      thre_shift=<int>    (縞(シフト)) stripe(shift)thres (default=%d, 0-1024)\n")
        _T("      thre_deint=<int>    (縞(解除))   stripe(deint)thres (default=%d, 0-1024)\n")
        _T("      thre_motion_y=<int> (Y動き)      Y motion threshold (default=%d, 0-1024)\n")
        _T("      thre_motion_c=<int> (C動き)      C motion threshold (default=%d, 0-1024)\n")
        _T("      level=<int>         (解除Lv)     set deint level    (default=%d, 0-4\n")
        _T("      shift=<bool>  (フィールドシフト) enable field shift (default=%s)\n")
        _T("      drop=<bool>   (ドロップ)         enable frame drop  (default=%s)\n")
        _T("      smooth=<bool> (スムージング)     enable smoothing   (default=%s)\n")
        _T("      24fps=<bool>  (24fps化)          force 30fps->24fps (default=%s)\n")
        _T("      tune=<bool>   (調整モード)       show scan result   (default=%s)\n")
        _T("      rff=<bool>                       rff flag aware     (default=%s)\n")
        _T("      timecode=<bool>                  output timecode    (default=%s)\n")
        _T("      log=<bool>                       output log         (default=%s)\n"),
        FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_TB,
        FILTER_DEFAULT_AFS_CLIP_LR, FILTER_DEFAULT_AFS_CLIP_LR,
        FILTER_DEFAULT_AFS_METHOD_SWITCH, FILTER_DEFAULT_AFS_COEFF_SHIFT,
        FILTER_DEFAULT_AFS_THRE_SHIFT, FILTER_DEFAULT_AFS_THRE_DEINT,
        FILTER_DEFAULT_AFS_THRE_YMOTION, FILTER_DEFAULT_AFS_THRE_CMOTION,
        FILTER_DEFAULT_AFS_ANALYZE,
        FILTER_DEFAULT_AFS_SHIFT   ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_DROP    ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_SMOOTH  ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_FORCE24 ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_TUNE    ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_RFF     ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_TIMECODE ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_LOG      ? _T("on") : _T("off"));
#endif
#if ENABLE_VPP_FILTER_NNEDI
    str += strsprintf(_T("\n")
        _T("   --vpp-nnedi [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable nnedi deinterlacer\n")
        _T("    params\n")
        _T("      field=<string>\n")
        _T("          auto (default)    Generate latter field from first field.\n")
        _T("          top               Generate bottom field using top field.\n")
        _T("          bottom            Generate top field using bottom field.\n")
        _T("      nns=<int>             Neurons of neural net (default: 32)\n")
        _T("                              16, 32, 64, 128, 256\n")
        _T("      nszie=<int>x<int>     Area size neural net uses to generate a pixel.\n")
        _T("                              8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4(default)\n")
        _T("      quality=<string>      quality settings\n")
        _T("                              fast (default), slow\n")
        _T("      prescreen=<string>    (default: new_block)\n")
        _T("          none              No pre-screening is done and all pixels will be\n")
        _T("                            generated by neural net.\n")
        _T("          original          Runs prescreener to determine which pixel to apply\n")
        _T("          new               neural net, other pixels will be generated from\n")
        _T("                            simple interpolation.\n")
        _T("          original_block    GPU optimized ver of original/new.\n")
        _T("          new_block\n")
        _T("      errortype=<string>    Select weight parameter for neural net.\n")
        _T("                              abs (default), square\n")
        _T("      prec=<string>         Select calculation precision.\n")
        _T("                              auto (default), fp16, fp32\n")
        _T("      weightfile=<string>   Set path of weight file. By default (not specified),\n")
        _T("                              internal weight params will be used.\n"));
#endif
#if ENABLE_VPP_FILTER_YADIF
    str += strsprintf(_T("\n")
        _T("   --vpp-yadif [<param1>=<value>]\n")
        _T("     enable yadif deinterlacer\n")
        _T("    params\n")
        _T("      mode=<string>\n")
        _T("          auto (default)    Generate latter field using first field.\n")
        _T("          tff               Generate bottom field using top field.\n")
        _T("          bff               Generate top field using bottom field.\n")
        _T("          bob               Generate one frame from each field.\n")
        _T("          bob_tff           Generate one frame from each field assuming tff.\n")
        _T("          bob_bff           Generate one frame from each field assuming bff.\n"));
#endif
#if ENABLE_VPP_FILTER_RFF
    str += strsprintf(_T("\n")
        _T("   --vpp-rff                    apply rff flag, with %savsw reader only.\n"),
            ENABLE_VPP_FILTER_RFF_AVHW ? _T("avhw/") : _T(""));
#endif
#if ENABLE_VPP_FILTER_SELECT_EVERY
    str += strsprintf(_T("\n")
        _T("   --vpp-select-every <int>[,offset=<int>]\n")
        _T("     select one frame per specified frames and create output.\n"));
#endif
#if ENABLE_VPP_FILTER_DECIMATE
    str += strsprintf(_T("\n")
        _T("   --vpp-decimate [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     drop duplicated frame.\n")
        _T("    params\n")
        _T("      cycle=<int>               num of frame to select frame(s) to be droppped.\n")
        _T("                                  (default=%d)\n")
        _T("      drop=<int>                num of frame(s) to drop within a cycle.\n")
        _T("                                  (default=%d)\n")
        _T("      thredup=<float>           duplicate threshold. (default=%.1f, 0 - 100)\n")
        _T("      thresc=<float>            scene change threshold. (default=%.1f, 0 - 100)\n")
        _T("      blockx=<int>              block size of x direction (default=%d).\n")
        _T("      blocky=<int>              block size of y direction (default=%d).\n")
        _T("                                  block size could be 4, 8, 16, 32, 64.\n")
        _T("      chroma=<bool>             consdier chroma (default: %s)\n")
        _T("      log=<bool>                output log file (default: %s).\n"),
        FILTER_DEFAULT_DECIMATE_CYCLE, FILTER_DEFAULT_DECIMATE_DROP,
        FILTER_DEFAULT_DECIMATE_THRE_DUP, FILTER_DEFAULT_DECIMATE_THRE_SC,
        FILTER_DEFAULT_DECIMATE_BLOCK_X, FILTER_DEFAULT_DECIMATE_BLOCK_Y,
        FILTER_DEFAULT_DECIMATE_PREPROCESSED ? _T("on") : _T("off"),
        FILTER_DEFAULT_DECIMATE_CHROMA ? _T("on") : _T("off"),
        FILTER_DEFAULT_DECIMATE_LOG ? _T("on") : _T("off"));
#endif
#if ENABLE_VPP_FILTER_MPDECIMATE
    str += strsprintf(_T("\n")
        _T("   --vpp-mpdecimate [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     drop duplicated frame.\n")
        _T("    params\n")
        _T("      hi=<int>                  the frame might be dropped if no 8x8 block difference\n")
        _T("                                is more than \"hi\" (default=%d (8x8x%d)).\n")
        _T("      lo=<int>                  the frame might be dropped if the fraction of 8x8 blocks\n")
        _T("      frac=<float>              with difference smaller than \"lo\" is more than \"frac\".\n")
        _T("                                  (lo default=%d (8x8x%d), frac default=%.3f)\n")
        _T("      max=<bool>                Max consecutive frames which can be dropped (positive)\n")
        _T("                                min interval between dropped frames (if negative)\n")
        _T("                                  (default: %d)\n")
        _T("      log=<bool>                output log file (default: %s).\n"),
        FILTER_DEFAULT_MPDECIMATE_HI, FILTER_DEFAULT_MPDECIMATE_HI / (8 * 8),
        FILTER_DEFAULT_MPDECIMATE_LO, FILTER_DEFAULT_MPDECIMATE_LO / (8 * 8),
        FILTER_DEFAULT_MPDECIMATE_FRAC, FILTER_DEFAULT_MPDECIMATE_MAX,
        FILTER_DEFAULT_DECIMATE_LOG ? _T("on") : _T("off"));
#endif
#if ENABLE_NVVFX
    {
        str += strsprintf(_T("\n")
            _T("--vpp-resize <string> or [<param1>=<value>][,<param2>=<value>][...]")
            _T("    params\n")
            _T("      algo=<string>             select algorithm"));
        const TCHAR *indent = _T("        ");
        int length = 80;
        for (int ia = 0; list_vpp_resize[ia].desc; ia++) {
            if (length > 77) {
                length = _tcslen(indent);
                str += tstring(_T("\n")) + indent;
            } else {
                length += _tcslen(_T(", "));
                str += _T(", ");
            }
            length += _tcslen(list_vpp_resize[ia].desc);
            str += list_vpp_resize[ia].desc;
        }
        str += _T("        default: auto\n");
        str += strsprintf(_T("\n")
            _T("      superres-mode=<int>\n")
            _T("        mode for nvvfx-superres     0 ... conservative (default)\n")
            _T("                                    1 ... aggressive \n")
            _T("      superres-strength=<float>\n")
            _T("        strength for nvvfx-superres (0.0 - 1.0)\n"));
    }
#else
    str += print_list_options(_T("--vpp-resize <string>"), list_vpp_resize_help, 0);
#endif
#if ENCODER_QSV
    str += print_list_options(_T("--vpp-resize-mode <string>"), list_vpp_resize_mode, 0);
#endif
#if ENABLE_VPP_FILTER_CONVOLUTION3D
    str += strsprintf(_T("\n")
        _T("   --vpp-convolution3d [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable denoise filter by convolution3d.\n")
        _T("    params\n")
        _T("      matrix=<string>  standard(default), simple\n")
        _T("      fast=<bool>      use fast mode\n")
        _T("      ythresh=<int>    spatial  luma threshold   (default=%d, 0 - 255)\n")
        _T("      cthresh=<int>    spatial  chroma threshold (default=%d, 0 - 255)\n")
        _T("      t_ythresh=<int>  temporal luma threshold   (default=%d, 0 - 255)\n")
        _T("      t_cthresh=<int>  temporal chroma threshold (default=%d, 0 - 255)\n"),
        FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_SPATIAL, FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_SPATIAL,
        FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_TEMPORAL, FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_TEMPORAL);
#endif
    str += strsprintf(_T("\n")
        _T("   --vpp-knn [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable denoise filter by K-nearest neighbor.\n")
        _T("    params\n")
        _T("      radius=<int>              radius of knn (default=%d)\n")
        _T("      strength=<float>          strength of knn (default=%.2f, 0.0-1.0)\n")
        _T("      lerp=<float>              balance of orig & blended pixel (default=%.2f)\n")
        _T("                                  lower value results strong denoise.\n")
        _T("      th_lerp=<float>           edge detect threshold (default=%.2f, 0.0-1.0)\n")
        _T("                                  higher value will preserve edge.\n"),
        FILTER_DEFAULT_KNN_RADIUS, FILTER_DEFAULT_KNN_STRENGTH, FILTER_DEFAULT_KNN_LERPC,
        FILTER_DEFAULT_KNN_LERPC_THRESHOLD);
#if ENABLE_VPP_FILTER_PMD
    str += strsprintf(_T("\n")
        _T("   --vpp-pmd [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable denoise filter by pmd.\n")
        _T("    params\n")
        _T("      apply_count=<int>         count to apply pmd denoise (default=%d)\n")
        _T("      strength=<float>          strength of pmd (default=%.2f, 0.0-100.0)\n")
        _T("      threshold=<float>         threshold of pmd (default=%.2f, 0.0-255.0)\n")
        _T("                                  lower value will preserve edge.\n"),
        FILTER_DEFAULT_PMD_APPLY_COUNT, FILTER_DEFAULT_PMD_STRENGTH, FILTER_DEFAULT_PMD_THRESHOLD);
#endif
#if ENABLE_VPP_FILTER_SMOOTH
    str += strsprintf(_T("\n")
        _T("   --vpp-smooth [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable smooth filter.\n")
        _T("    params\n")
        _T("      quality=<int>         quality of filter (high=higher quality but slow)\n")
        _T("                             (default=%d, 1-6)\n")
        _T("      qp=<float>            strength of filter (default=%.2f, 0.0-100.0)\n")
        _T("      prec=<string>         Select calculation precision.\n")
        _T("                              auto (default), fp16, fp32\n"),
        FILTER_DEFAULT_SMOOTH_QUALITY, FILTER_DEFAULT_SMOOTH_QP);
#endif
    str += strsprintf(_T("\n")
        _T("   --vpp-subburn [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Burn in specified subtitle to the video.\n")
        _T("    params\n")
        _T("      track=<int>               subtitle track of the input file to burn in.\n")
        _T("      filename=<string>         subtitle file path to burn in.\n")
        _T("      charcode=<string>         subtitle charcter code.\n")
        _T("      shaping=<string>          rendering quality of text.\n")
        _T("      scale=<float>             scaling multiplizer for bitmap subtitles.\n")
        _T("      transparency=<float>      adds additional transparency.\n")
        _T("                                  (default=0.0, 0.0 - 1.0)\n")
        _T("      brightness=<float>        modifies brightness of the subtitle.\n")
        _T("                                  (default=%.1f, -1.0 - 1.0)\n")
        _T("      contrast=<float>          modifies contrast of the subtitle.\n")
        _T("                                  (default=%.1f, -2.0 - 2.0)\n")
        _T("      vid_ts_offset=<bool>      add timestamp offset to match the first timestamp of\n")
        _T("                                  the video file (default: on)\n")
        _T("                                  (when \"track\" is used this options is always on)\n")
        _T("      ts_offset=<float>         add offset in seconds to subtitle timestamps.\n")
        _T("      fontsdir=<string>         directory with fonts used.\n")
        _T("      forced_subs_only=<bool>   render forced subs only.\n"),
        FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST);
#if ENABLE_VPP_FILTER_UNSHARP
    str += strsprintf(_T("\n")
        _T("   --vpp-unsharp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable unsharp filter.\n")
        _T("    params\n")
        _T("      radius=<int>              filter range for edge detection (default=%d, 1-9)\n")
        _T("      weight=<float>            strength of filter (default=%.2f, 0-10)\n")
        _T("      threshold=<float>         min brightness change to be sharpened (default=%.2f, 0-255)\n"),
        FILTER_DEFAULT_UNSHARP_RADIUS, FILTER_DEFAULT_UNSHARP_WEIGHT, FILTER_DEFAULT_UNSHARP_THRESHOLD);
#endif
#if ENABLE_VPP_FILTER_EDGELEVEL
    str += strsprintf(_T("\n")
        _T("   --vpp-edgelevel [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     edgelevel filter to enhance edge.\n")
        _T("    params\n")
        _T("      strength=<float>          strength (default=%d, -31 - 31)\n")
        _T("      threshold=<float>         threshold to ignore noise (default=%.1f, 0-255)\n")
        _T("      black=<float>             allow edge to be darker on edge enhancement\n")
        _T("                                  (default=%.1f, 0-31)\n")
        _T("      white=<float>             allow edge to be brighter on edge enhancement\n")
        _T("                                  (default=%.1f, 0-31)\n"),
        FILTER_DEFAULT_EDGELEVEL_STRENGTH, FILTER_DEFAULT_EDGELEVEL_THRESHOLD, FILTER_DEFAULT_EDGELEVEL_BLACK, FILTER_DEFAULT_EDGELEVEL_WHITE);
#endif
#if ENABLE_VPP_FILTER_WARPSHARP
    str += strsprintf(_T("\n")
        _T("   --vpp-warpsharp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     warpsharp filter to enhance edge.\n")
        _T("    params\n")
        _T("      threshold=<float>         edge mask threshold (default=%.1f, 0-255)\n")
        _T("      blur=<int>                number of times to blur the edge mask.\n")//Increase for weaker sharpening
        _T("                                  (default=%d, 0-)\n")
        _T("      type=<int>                blur type, 0...13x13, 1...5x5 (default=%d)\n")
        _T("      depth=<float>             how far to warp (default=%.1f, -128 - 128)\n")
        _T("      chroma=<int>              0...use luma mask, 1...create chroma mask\n")
        _T("                                  (default=%d)\n"),
        FILTER_DEFAULT_WARPSHARP_THRESHOLD, FILTER_DEFAULT_WARPSHARP_BLUR, FILTER_DEFAULT_WARPSHARP_TYPE,
        FILTER_DEFAULT_WARPSHARP_DEPTH, FILTER_DEFAULT_WARPSHARP_CHROMA);
#endif
#if ENABLE_VPP_FILTER_CURVES
    str += strsprintf(_T("\n")
        _T("   --vpp-curves [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     apply color adjustments using curves.\n")
        _T("    params\n")
        _T("      preset=<string>\n")
        _T("        color_negative, process, darker, lighter, increase_contrast\n")
        _T("        linear_contrast, medium_contrast, strong_contrast\n")
        _T("        negative, vintage\n")
        _T("      m=<string>\n")
        _T("        set master curve points, post process for luminance.\n")
        _T("      r=<string>\n")
        _T("        set curve points for red. Will override preset settings.\n")
        _T("      g=<string>\n")
        _T("        set curve points for green. Will override preset settings.\n")
        _T("      b=<string>\n")
        _T("        set curve points for blue. Will override preset settings.\n")
        _T("      all=<string>\n")
        _T("        set curve points for r,g,b when not specified. Will override preset settings.\n"));
#endif
#if ENABLE_VPP_FILTER_TWEAK
    str += strsprintf(_T("\n")
        _T("   --vpp-tweak [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     apply brightness, constrast, gamma, hue adjustment.\n")
        _T("    params\n")
        _T("      brightness=<float>        (default=%.1f, -1.0 - 1.0)\n")
        _T("      contrast=<float>          (default=%.1f, -2.0 - 2.0)\n")
        _T("      gamma=<float>             (default=%.1f,  0.1 - 10.0)\n")
        _T("      saturation=<float>        (default=%.1f,  0.0 - 3.0)\n")
        _T("      hue=<float>               (default=%.1f, -180 - 180)\n"),
        FILTER_DEFAULT_TWEAK_BRIGHTNESS,
        FILTER_DEFAULT_TWEAK_CONTRAST,
        FILTER_DEFAULT_TWEAK_GAMMA,
        FILTER_DEFAULT_TWEAK_SATURATION,
        FILTER_DEFAULT_TWEAK_HUE);
#endif
    str += strsprintf(_T("\n")
        _T("   --vpp-rotate <int>           rotate video (90, 180, 270)\n")
    );
    str += strsprintf(_T("\n")
        _T("   --vpp-transform [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("    params\n")
        _T("      flip_x=<bool>\n")
        _T("      flip_y=<bool>\n")
        _T("      transpose=<bool>\n")
    );
#if ENABLE_VPP_FILTER_DEBAND
    str += strsprintf(_T("\n")
        _T("   --vpp-deband [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable deband filter.\n")
        _T("    params\n")
        _T("      range=<int>               range (default=%d, 0-127)\n")
        _T("      sample=<int>              sample (default=%d, 0-2)\n")
        _T("      thre=<int>                threshold for y, cb & cr\n")
        _T("      thre_y=<int>              threshold for y (default=%d, 0-31)\n")
        _T("      thre_cb=<int>             threshold for cb (default=%d, 0-31)\n")
        _T("      thre_cr=<int>             threshold for cr (default=%d, 0-31)\n")
        _T("      dither=<int>              strength of dither for y, cb & cr\n")
        _T("      dither_y=<int>            strength of dither for y (default=%d, 0-31)\n")
        _T("      dither_c=<int>            strength of dither for cb/cr (default=%d, 0-31)\n")
        _T("      seed=<int>                rand seed (default=%d)\n")
        _T("      blurfirst                 blurfirst (default=%s)\n")
        _T("      rand_each_frame           generate rand for each frame (default=%s)\n"),
        FILTER_DEFAULT_DEBAND_RANGE, FILTER_DEFAULT_DEBAND_MODE,
        FILTER_DEFAULT_DEBAND_THRE_Y, FILTER_DEFAULT_DEBAND_THRE_CB, FILTER_DEFAULT_DEBAND_THRE_CR,
        FILTER_DEFAULT_DEBAND_DITHER_Y, FILTER_DEFAULT_DEBAND_DITHER_C,
        FILTER_DEFAULT_DEBAND_SEED,
        FILTER_DEFAULT_DEBAND_BLUR_FIRST ? _T("on") : _T("off"),
        FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME ? _T("on") : _T("off"));
#endif
#if ENABLE_VPP_FILTER_PAD
    str += strsprintf(_T("\n")
        _T("   --vpp-pad <int>,<int>,<int>,<int>\n")
        _T("     add padding to left,top,right,bottom (in pixels)\n"));
#endif
#if ENABLE_VPP_FILTER_OVERLAY
    str += strsprintf(_T("\n")
        _T("   --vpp-overlay [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("    params\n")
        _T("      file=<string>             src file path of the image\n")
        _T("      pos=<int>x<int>           position to add image\n")
        _T("      size=<int>x<int>          size of image  (default: 0x0 = no resize)\n")
        _T("      alpha=<float>             alpha value of overlay\n")
        _T("                                  default: 1.0 (0.0 - 1.0)\n")
        _T("      alpha_mode=<string>       override ... set value of alpha\n")
        _T("                                mul      ... multiple original value\n")
        _T("                                lumakey  ... set alpha depending on luma\n")
        _T("      lumakey_threshold=<float> luma used for tranparency.\n")
        _T("                                  default: 0.0 (dark: 0.0 - 1.0 :bright)\n")
        _T("      lumakey_tolerance=<float> set luma range to be keyed out.\n")
        _T("                                  default: 0.1 (0.0 - 1.0)\n")
        _T("      lumakey_threshold=<float> set the range of softness for lumakey\n")
        _T("      loop=<bool>\n")
    );
#endif
    str += strsprintf(_T("\n")
        _T("   --vpp-perf-monitor           check vpp perfromance (for debug)\n")
    );
    return str;
}

tstring gen_cmd_help_ctrl() {
    tstring str = strsprintf(_T("\n")
        _T("   --log <string>               set log file name\n")
        _T("   --log-level <string>         set log level\n")
        _T("                                  debug, info(default), warn, error, quiet\n")
        _T("   --log-opt [<param1>][,<param2>][]...\n")
        _T("     additional options for log output.\n")
        _T("    params\n")
        _T("      addtime                   add time to log lines.\n")
        _T("   --log-framelist [<string>]   output debug info for avsw/avhw reader.\n")
        _T("   --log-packets [<string>]     output debug info for avsw/avhw reader.\n")
        _T("   --log-mux-ts [<string>]      output debug info for avsw/avhw reader.\n"));

    str += strsprintf(_T("\n")
        _T("   --option-file <string>       read commanline options written in file.\n"));
    str += strsprintf(_T("")
        _T("   --max-procfps <int>          limit encoding speed for lower utilization.\n")
        _T("                                 default:0 (no limit)\n")
        _T("   --lowlatency                 minimize latency (might have lower throughput).\n"));
    str += strsprintf(_T("")
        _T("   --output-buf <int>           buffer size for output in MByte\n")
        _T("                                 default %d MB (0-%d)\n"),
        RGY_OUTPUT_BUF_MB_DEFAULT, RGY_OUTPUT_BUF_MB_MAX
    );
#if ENABLE_AVCODEC_OUT_THREAD
    str += strsprintf(_T("")
        _T("   --output-thread <int>        set output thread num\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
#if 0
        _T("   --audio-thread <int>         set audio thread num, available only with output thread\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
        _T("                                  2: use two thread\n")
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    );
    {
        std::array<CX_DESC, RGY_THREAD_TYPE_STR.size() + 1> list_rgy_thread_type;
        for (size_t i = 0; i < RGY_THREAD_TYPE_STR.size(); i++) {
            list_rgy_thread_type[i].value = (int)RGY_THREAD_TYPE_STR[i].first;
            list_rgy_thread_type[i].desc = RGY_THREAD_TYPE_STR[i].second;
        }
        list_rgy_thread_type[RGY_THREAD_TYPE_STR.size()].value = 0;
        list_rgy_thread_type[RGY_THREAD_TYPE_STR.size()].desc = nullptr;

        std::array<CX_DESC, RGY_THREAD_AFFINITY_MODE_STR.size()+1> list_thread_affinity_mode;
        for (size_t i = 0; i < RGY_THREAD_AFFINITY_MODE_STR.size(); i++) {
            list_thread_affinity_mode[i].value = (int)RGY_THREAD_AFFINITY_MODE_STR[i].second;
            list_thread_affinity_mode[i].desc = RGY_THREAD_AFFINITY_MODE_STR[i].first;
        }
        list_thread_affinity_mode[RGY_THREAD_AFFINITY_MODE_STR.size()].value = 0;
        list_thread_affinity_mode[RGY_THREAD_AFFINITY_MODE_STR.size()].desc = nullptr;

        std::array<CX_DESC, RGY_THREAD_PRIORITY_STR.size() + 1> list_thread_priority;
        for (size_t i = 0; i < RGY_THREAD_PRIORITY_STR.size(); i++) {
            list_thread_priority[i].value = (int)RGY_THREAD_PRIORITY_STR[i].first;
            list_thread_priority[i].desc = RGY_THREAD_PRIORITY_STR[i].second;
        }
        list_thread_priority[RGY_THREAD_PRIORITY_STR.size()].value = 0;
        list_thread_priority[RGY_THREAD_PRIORITY_STR.size()].desc = nullptr;

        std::array<CX_DESC, RGY_THREAD_POWER_THROTTOLING_MODE_STR.size() + 1> list_thread_throttoling;
        for (size_t i = 0; i < RGY_THREAD_POWER_THROTTOLING_MODE_STR.size(); i++) {
            list_thread_throttoling[i].value = (int)RGY_THREAD_POWER_THROTTOLING_MODE_STR[i].first;
            list_thread_throttoling[i].desc = RGY_THREAD_POWER_THROTTOLING_MODE_STR[i].second;
        }
        list_thread_throttoling[RGY_THREAD_POWER_THROTTOLING_MODE_STR.size()].value = 0;
        list_thread_throttoling[RGY_THREAD_POWER_THROTTOLING_MODE_STR.size()].desc = nullptr;

        str += strsprintf(_T("")
            _T("   --thread-affinity [<string1>=](<string2>[#<int>[:<int>][]...] or 0x<hex>)\n"));
        str += strsprintf(_T("")
            _T("     target (string1)  (default: %s)\n"), RGY_THREAD_TYPE_STR[(int)RGYThreadType::ALL].second
        ) + print_list(list_rgy_thread_type.data()) + _T("\n");
        str += strsprintf(_T("")
            _T("     thread type (string2)  (default: %s)\n"), rgy_thread_affnity_mode_to_str(RGYThreadAffinityMode::ALL)
        ) + print_list(list_thread_affinity_mode.data()) + _T("\n");
#if defined(_WIN32) || defined(_WIN64)
        str += strsprintf(_T("")
            _T("   --thread-priority [<string1>=](<string2>[#<int>[:<int>][]...] or 0x<hex>)\n"));
        str += strsprintf(_T("")
            _T("     target (string1)  (default: %s)\n"), RGY_THREAD_TYPE_STR[(int)RGYThreadType::ALL].second
        ) + print_list(list_rgy_thread_type.data()) + _T("\n");
        str += strsprintf(_T("")
            _T("     priority (string2)  (default: %s)\n"), rgy_thread_priority_mode_to_str(RGYThreadPriority::Normal)
        ) + print_list(list_thread_priority.data()) + _T("\n");

        str += strsprintf(_T("")
            _T("   --thread-throttling [<string1>=](<string2>[#<int>[:<int>][]...] or 0x<hex>)\n"));
        str += strsprintf(_T("")
            _T("     target (string1)  (default: %s)\n"), RGY_THREAD_TYPE_STR[(int)RGYThreadType::ALL].second
        ) + print_list(list_rgy_thread_type.data()) + _T("\n");
        str += strsprintf(_T("")
            _T("     throttling mode (string2)  (default: %s)\n"), rgy_thread_power_throttoling_mode_to_str(RGYThreadPowerThrottlingMode::Auto)
        ) + print_list(list_thread_throttoling.data()) + _T("\n");
#endif //#if defined(_WIN32) || defined(_WIN64)
    }
#endif //#if ENABLE_AVCODEC_OUT_THREAD
    str += strsprintf(_T("\n")
        _T("   --avsdll <string>            specifies AviSynth DLL location to use.\n"));
#if defined(_WIN32) || defined(_WIN64)
    str += strsprintf(_T("\n")
        _T("   --process-codepage <string>  utf8 ... use UTF-8 (default)\n")
        _T("                                os   ... use the codepage set in Operating System.\n"));
#endif //#if defined(_WIN32) || defined(_WIN64)
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    str += strsprintf(_T("\n")
        _T("   --disable-opencl             disable opencl features.\n"));
#endif
    str += strsprintf(_T("\n")
        _T("   --perf-monitor [<string>][,<string>]...\n")
        _T("       check performance info of encoder and output to log file\n")
        _T("       select counter from below, default = all\n")
        _T("                                 \n")
        _T("     counters for perf-monitor\n")
        _T("                                 all          ... monitor all info\n")
        _T("                                 cpu_total    ... cpu total usage (%%)\n")
        _T("                                 cpu_kernel   ... cpu kernel usage (%%)\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu_main     ... cpu main thread usage (%%)\n")
        _T("                                 cpu_enc      ... cpu encode thread usage (%%)\n")
        _T("                                 cpu_in       ... cpu input thread usage (%%)\n")
        _T("                                 cpu_out      ... cpu output thread usage (%%)\n")
        _T("                                 cpu_aud_proc ... cpu aud proc thread usage (%%)\n")
        _T("                                 cpu_aud_enc  ... cpu aud enc thread usage (%%)\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu          ... monitor all cpu info\n")
        _T("                                 gpu_load    ... gpu usage (%%)\n")
        _T("                                 gpu_clock   ... gpu avg clock\n")
        _T("                                 vee_load    ... gpu video encoder usage (%%)\n")
        _T("                                 ved_load    ... gpu video decoder usage (%%)\n")
#if ENABLE_NVML
        _T("                                 ve_clock    ... gpu video engine clock\n")
#endif
        _T("                                 gpu         ... monitor all gpu info\n")
        _T("                                 queue       ... queue usage\n")
        _T("                                 mem_private ... private memory (MB)\n")
        _T("                                 mem_virtual ... virtual memory (MB)\n")
        _T("                                 mem         ... monitor all memory info\n")
        _T("                                 io_read     ... io read  (MB/s)\n")
        _T("                                 io_write    ... io write (MB/s)\n")
        _T("                                 io          ... monitor all io info\n")
        _T("                                 fps         ... encode speed (fps)\n")
        _T("                                 fps_avg     ... encode avg. speed (fps)\n")
        _T("                                 bitrate     ... encode bitrate (kbps)\n")
        _T("                                 bitrate_avg ... encode avg. bitrate (kbps)\n")
        _T("                                 frame_out   ... written_frames\n")
        _T("                                 \n")
        _T("   --perf-monitor-interval <int> set perf monitor check interval (millisec)\n")
        _T("                                 default 500, must be 50 or more\n"));
    return str;
}
