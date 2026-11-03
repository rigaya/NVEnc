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
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "rgy_util.h"
#include "rgy_avutil.h"
#if ENCODER_NVENC
#include "NVEncFilterRtgmcSearchPrefilter.h"
#else
#include "rgy_filter_rtgmc_repair_profile.h"
#endif
#include "rgy_prm.h"
#include "rgy_cmd.h"
#include "rgy_language.h"
#include "rgy_perf_monitor.h"
#include "rgy_osdep.h"
#if ENABLE_VPP_FILTER_ONNX
#include "rgy_model_registry.h"
#include "rgy_log.h"
#include "rgy_filesystem.h"
#endif

std::vector<tstring> splitCommandLine(const TCHAR *cmd) {
    std::vector<tstring> result;
    if (cmd == nullptr || _tcslen(cmd) == 0) {
        // configstrが空文字列の場合、CommandLineToArgvWに渡すと先頭に実行ファイルへのパスが付与されてしまう
        // 混乱を避けるため、ここでreturnする
        return result;
    }
#if defined(_WIN32) || defined(_WIN64)
    std::wstring cmdw = tchar_to_wstring(cmd);
    int argc = 0;
    auto argvw = CommandLineToArgvW(cmdw.c_str(), &argc);
    if (argc <= 1) {
        return result;
    }
    for (int i = 0; i < argc; i++) {
        result.push_back(wstring_to_tstring(argvw[i]));
    }
    LocalFree(argvw);
#else
    tstring token;
    bool inDoubleQuotes = false;
    bool inSingleQuotes = false;
    bool escape = false;
    std::basic_stringstream<TCHAR> ss;

    while (*cmd) {
        TCHAR c = *cmd++;
        if (escape) {
            ss << c;
            escape = false;
        } else if (c == _T('\\')) {
            escape = true;
        } else if (c == _T('"')) {
            if (!inSingleQuotes) {
                inDoubleQuotes = !inDoubleQuotes;
            } else {
                ss << c;
            }
        } else if (c == _T('\'')) {
            if (!inDoubleQuotes) {
                inSingleQuotes = !inSingleQuotes;
            } else {
                ss << c;
            }
        } else if (c == _T(' ') && !inDoubleQuotes && !inSingleQuotes) {
            if (!ss.str().empty() || escape) {
                result.push_back(ss.str());
                ss.str(_T(""));
                ss.clear();
            }
        } else {
            ss << c;
        }
    }

    if (!ss.str().empty() || escape) {
        result.push_back(ss.str());
    }
#endif
    return result;
}

static bool is_valid_rtgmc_rep_thin(const int value) {
    return rgy_rtgmc_repair_thin_level_is_valid(value);
}

static bool is_valid_rtgmc_rep_pad(const int value) {
    return rgy_rtgmc_repair_pad_level_is_valid(value);
}

static int parse_degrain_search_early_sad(int *dst, const tstring& value, const bool allowAuto) {
    const auto normalized = tolowercase(trim(value, _T("\"")));
    if (normalized == _T("off")) {
        *dst = FILTER_DEFAULT_DEGRAIN_SEARCH_EARLY_SAD;
        return 0;
    }
    if (allowAuto && normalized == _T("auto")) {
        *dst = FILTER_DEFAULT_KFM_SEARCH_EARLY_SAD_OVERRIDE;
        return 0;
    }
    try {
        size_t idx = 0;
        const auto parsed = std::stoll(normalized, &idx);
        if (idx != normalized.length()
            || parsed < FILTER_MIN_DEGRAIN_SEARCH_EARLY_SAD
            || parsed > FILTER_MAX_DEGRAIN_SEARCH_EARLY_SAD) {
            return 1;
        }
        *dst = (int)parsed;
    } catch (...) {
        return 1;
    }
    return 0;
}

#if FOR_AUO
#pragma warning (disable: 4100)
#else
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

RGYCmdErrorInfo& rgy_cmd_error_info() {
    static RGYCmdErrorInfo info;
    return info;
}

void print_cmd_error_unknown_opt(tstring strErrorValue) {
    rgy_cmd_error_info().errorOccurred = true;
    if (rgy_cmd_error_info().quiet) return;
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
    rgy_cmd_error_info().errorOccurred = true;
    rgy_cmd_error_info().paramList = optionParamsList;
    rgy_cmd_error_info().unknownParam = true;
    if (rgy_cmd_error_info().quiet) return;
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
    rgy_cmd_error_info().errorOccurred = true;
    if (list) {
        rgy_cmd_error_info().valueList.clear();
        for (int i = 0; list[i].desc && i < list_length; i++) {
            rgy_cmd_error_info().valueList.push_back(tchar_to_string(list[i].desc));
        }
    }
    if (rgy_cmd_error_info().quiet) return;
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

// 文字列のみで候補が決まる(CX_DESCを持たない)オプション向けのオーバーロード
// CX_DESC版に委譲するため、終端(desc==nullptr)付きの一時配列を組み立てる
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<tstring>& valueList) {
    std::vector<CX_DESC> list;
    for (const auto& value : valueList) {
        list.push_back({ value.c_str(), (int)list.size() });
    }
    list.push_back({ nullptr, 0 });
    print_cmd_error_invalid_value(strOptionName, strErrorValue, _T(""), list.data());
}

// パース後の整合性チェックで見つかったエラーを出力する
// 自己診断(--check-cmd-parse)実行中は出力を抑止する
void print_cmd_error_msg(const TCHAR *msg) {
    rgy_cmd_error_info().errorOccurred = true;
    if (rgy_cmd_error_info().quiet) return;
    _ftprintf(stderr, _T("%s\n"), msg);
}

// param_val が候補リストに含まれるか確認する
// 含まれない場合は候補を提示してエラーとし、std::nullopt を返す
static std::optional<tstring> check_str_list(const tstring& option_name, const tstring& param_arg, const tstring& param_val, const std::vector<tstring>& valueList) {
    const auto value = tolowercase(param_val);
    if (std::find(valueList.begin(), valueList.end(), value) == valueList.end()) {
        print_cmd_error_invalid_value(option_name + _T(" ") + param_arg + _T("="), param_val, valueList);
        return std::nullopt;
    }
    return value;
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage) {
    print_cmd_error_invalid_value(strOptionName, strErrorValue, strErrorMessage, (const CX_DESC *)nullptr);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<std::pair<RGY_CODEC, const CX_DESC *>>& codec_list) {
    rgy_cmd_error_info().errorOccurred = true;
    rgy_cmd_error_info().valueList.clear();
    for (const auto& codec : codec_list) {
        for (int i = 0; codec.second[i].desc; i++) {
            rgy_cmd_error_info().valueList.push_back(tchar_to_string(codec.second[i].desc));
        }
    }
    if (rgy_cmd_error_info().quiet) return;
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
    //エラーを避けるため、空のvectorを返すようにする
    if (configstr.length() == 0) {
        _ftprintf(stderr, _T("Option file is empty!\n"));
        return std::vector<tstring>();
    }
    return splitCommandLine(char_to_tstring(configstr).c_str());
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

static void mergeTrackExclusions(std::vector<int>& trackIDs, const std::vector<int>& additionalTrackIDs) {
    trackIDs.insert(trackIDs.end(), additionalTrackIDs.begin(), additionalTrackIDs.end());
    std::sort(trackIDs.begin(), trackIDs.end());
    trackIDs.erase(std::unique(trackIDs.begin(), trackIDs.end()), trackIDs.end());
}

static bool parseTrackExclusion(const std::string& str, std::vector<int>& trackIDs) {
    if (str.empty() || str[0] != '!') {
        return false;
    }
    std::vector<int> parsedTrackIDs;
    for (const auto& token : split(str, ",")) {
        if (token.size() <= 1 || token[0] != '!') {
            return false;
        }
        try {
            size_t parsedLength = 0;
            const auto trackID = std::stoi(token.substr(1), &parsedLength);
            if (trackID < 1 || parsedLength != token.size() - 1) {
                return false;
            }
            parsedTrackIDs.push_back(trackID);
        } catch (...) {
            return false;
        }
    }
    mergeTrackExclusions(trackIDs, parsedTrackIDs);
    return !trackIDs.empty();
}

static int getAudioTrackIdx(const RGYParamCommon *common, const int iTrack, const std::string& lang, const std::string& selectCodec) {
    if (iTrack == TRACK_SELECT_BY_LANG_EXCLUDE || iTrack == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        for (int i = 0; i < common->nAudioSelectCount; i++) {
            if (common->ppAudioSelectList[i]->trackID == iTrack) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_LANG) {
        if (lang.length() == 0) return -1;
        for (int i = 0; i < common->nAudioSelectCount; i++) {
            if (iTrack == common->ppAudioSelectList[i]->trackID && lang == common->ppAudioSelectList[i]->lang) {
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
    if (iTrack == TRACK_SELECT_BY_LANG_EXCLUDE || iTrack == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
            if (common->ppSubtitleSelectList[i]->trackID == iTrack) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_LANG) {
        if (lang.length() == 0) return -1;
        for (int i = 0; i < common->nSubtitleSelectCount; i++) {
            if (iTrack == common->ppSubtitleSelectList[i]->trackID && lang == common->ppSubtitleSelectList[i]->lang) {
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
    if (iTrack == TRACK_SELECT_BY_LANG_EXCLUDE || iTrack == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        for (int i = 0; i < common->nDataSelectCount; i++) {
            if (common->ppDataSelectList[i]->trackID == iTrack) {
                return i;
            }
        }
    } else if (iTrack == TRACK_SELECT_BY_LANG) {
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

std::vector<CX_DESC> get_libplacebo_only_resize_list() {
    std::vector<CX_DESC> libplacebo_resize_list;
    for (size_t i = 0; list_vpp_resize[i].desc; i++) {
        if (_tcsncmp(list_vpp_resize[i].desc, _T("libplacebo-"), _tcslen(_T("libplacebo-"))) == 0) {
            libplacebo_resize_list.push_back(list_vpp_resize[i]);
        }
    }
    libplacebo_resize_list.push_back({ nullptr, 0 });
    return libplacebo_resize_list;
}

#pragma warning(disable: 4100) //warning C4100: 'argData': 引数は関数の本体部で 1 度も参照されません。
#pragma warning(disable: 4127) //warning C4127: 条件式が定数です。

int handle_vpp_onnx_list_models(const RGYParamVpp *vpp) {
#if ENABLE_VPP_FILTER_ONNX
    if (vpp->onnxModelDir.empty()) {
        _ftprintf(stderr, _T("Error: --vpp-onnx-model-dir must be specified with --vpp-onnx list.\n"));
        return 1;
    }
    const auto jsonPath = PathCombineS(vpp->onnxModelDir, _T("models.json"));
    RGYModelRegistry reg;
    auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET);
    if (reg.load(jsonPath, log) != RGY_ERR_NONE) {
        _ftprintf(stderr, _T("Error: failed to load models.json from %s\n"), vpp->onnxModelDir.c_str());
        return 1;
    }
    _ftprintf(stdout, _T("Available ONNX models:\n"));
    for (const auto& [name, entry] : reg.models()) {
        const auto fullPath = PathCombineS(reg.baseDir(), entry.path);
        const bool exists = rgy_file_exists(fullPath);
        const auto frames = entry.frames > 1 ? strsprintf(_T(" [frames=%d]"), entry.frames) : tstring();
        _ftprintf(stdout, _T("  %-24s  %s%s%s%s\n"), name.c_str(), entry.path.c_str(),
            entry.fp32 ? _T(" [fp32]") : _T(""),
            frames.c_str(),
            exists ? _T("") : _T(" [not found]"));
    }
#endif // ENABLE_VPP_FILTER_ONNX
    return 1;
}

static const CX_DESC list_vpp_nnedi_upscale_shift[] = {
    { _T("linear"), 0 },
    { _T("cubic"),  1 },
    { nullptr, 0 }
};

// NNEDIのネットワーク設定は通常版と2倍拡大版で共通化する。
// 戻り値: 0=処理済み、1=エラー、-1=未知のパラメータ。
static int parse_one_nnedi_param(VppNnedi& nnedi, const tstring& param_arg, const tstring& param_val,
    const TCHAR *option_name, const bool allowFieldPlanes) {
    const auto invalid = [&]() {
        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
        return 1;
    };
    const auto parse_int = [&](int *dst) {
        try {
            size_t idx = 0;
            *dst = std::stoi(param_val, &idx);
            return (idx == param_val.length()) ? 0 : invalid();
        } catch (...) {
            return invalid();
        }
    };
    const auto parse_bool = [&](bool *dst) {
        bool value = false;
        if (cmd_string_to_bool(&value, param_val)) return invalid();
        *dst = value;
        return 0;
    };

    if (allowFieldPlanes && param_arg == _T("enable")) {
        return parse_bool(&nnedi.enable);
    }
    if (allowFieldPlanes && param_arg == _T("planes")) {
        nnedi.planes = { false, false, false };
        if (param_val == _T("all")) {
            nnedi.planes = { true, true, true };
        } else {
            for (const auto& plane : split(param_val, _T(":"))) {
                if      (plane == _T("y")) nnedi.planes[0] = true;
                else if (plane == _T("u")) nnedi.planes[1] = true;
                else if (plane == _T("v")) nnedi.planes[2] = true;
                else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                        _T("Supported values: all, or \":\"-separated list of y, u, v (e.g. y:u:v)."));
                    return 1;
                }
            }
        }
        return 0;
    }
    if (allowFieldPlanes && param_arg == _T("field")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_field, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_field);
            return 1;
        }
        nnedi.field = (VppNnediField)value;
        return 0;
    }
    if (param_arg == _T("nsize")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_nsize, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_nsize);
            return 1;
        }
        nnedi.nsize = (VppNnediNSize)value;
        return 0;
    }
    if (param_arg == _T("nns")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_nns, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_nns);
            return 1;
        }
        nnedi.nns = value;
        return 0;
    }
    if (param_arg == _T("quality")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_quality, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_quality);
            return 1;
        }
        nnedi.quality = (VppNnediQuality)value;
        return 0;
    }
    if (param_arg == _T("prescreen")) {
        if (parse_int(&nnedi.prescreen)) return 1;
        if (nnedi.prescreen < 2 || nnedi.prescreen > 4) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                _T("prescreen should be 2, 3, or 4."));
            return 1;
        }
        return 0;
    }
    if (param_arg == _T("errortype")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_error_type, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nnedi_error_type);
            return 1;
        }
        nnedi.errortype = (VppNnediErrorType)value;
        return 0;
    }
    if (param_arg == _T("prec")) {
        return 0;
    }
    if (param_arg == _T("clamp")) {
        if (parse_int(&nnedi.clamp)) return 1;
        if (nnedi.clamp < 0 || nnedi.clamp > 4) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                _T("clamp should be 0 - 4."));
            return 1;
        }
        return 0;
    }
    if (allowFieldPlanes && param_arg == _T("double_height")) {
        return parse_bool(&nnedi.doubleHeight);
    }
    if (param_arg == _T("weightfile")) {
        nnedi.weightfile = trim(param_val, _T("\"")).c_str();
        if (nnedi.weightfile.empty()) return invalid();
        return 0;
    }
    return -1;
}

static int parse_one_nnedi_upscale_param(VppNnediUpscale& upscale, const tstring& param_arg,
    const tstring& param_val, const TCHAR *option_name) {
    if (param_arg == _T("enable")) {
        bool value = false;
        if (cmd_string_to_bool(&value, param_val)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" enable="), param_val);
            return 1;
        }
        upscale.enable = value;
        return 0;
    }
    if (param_arg == _T("shift")) {
        int value = 0;
        if (!get_list_value(list_vpp_nnedi_upscale_shift, param_val.c_str(), &value)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" shift="), param_val,
                list_vpp_nnedi_upscale_shift);
            return 1;
        }
        upscale.shiftCubic = value != 0;
        return 0;
    }
    return parse_one_nnedi_param(upscale.nnedi, param_arg, param_val, option_name, false);
}

int parse_one_vpp_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamVpp *vpp, sArgsData *argData) {
    if (IS_OPTION("vpp-order") && ENABLE_VPP_ORDER) {
        i++;
        const auto list_vpp_filters = get_list_vpp_filter();
        const auto vpp_list = split(strInput[i], _T(","));
        vpp->filterOrder.clear();
        for (auto& vppstr : vpp_list) {
            const auto vpptype = vppfilter_str_to_type(vppstr);
            if (vpptype == VppType::VPP_NONE) {
                print_cmd_error_invalid_value(option_name, vppstr.c_str(), list_vpp_filters.data());
                return 1;
            }
            vpp->filterOrder.push_back(vpptype);
        }
        return 0;
    }
    if (IS_OPTION("vpp-deint-csp")) {
        i++;
        int value = 0;
        if (get_list_value(list_vpp_deint_csp, strInput[i], &value)) {
            vpp->deintCsp = (VppDeintCsp)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_deint_csp);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-resize")
        || (IS_OPTION("vpp-scaling") && ENCODER_QSV)) {
        i++;
        if (ENABLE_LIBPLACEBO || ENABLE_OPENCL || ENCODER_NVENC) {
            ///////////////////////////////////////////////////////////////////////////////////////
            // パラメータを追加したら、paramsResizeLibPlaceboにも追加する！！
            ///////////////////////////////////////////////////////////////////////////////////////
            std::vector<std::string> paramListResizeNVEnc;
            for (size_t ielem = 0; ielem < _countof(paramsResizeNVEnc); ielem++) {
                paramListResizeNVEnc.push_back(paramsResizeNVEnc[ielem]);
            }
            std::vector<std::string> paramListResizeQSVEnc;
            for (size_t ielem = 0; ielem < _countof(paramsResizeQSVEnc); ielem++) {
                paramListResizeQSVEnc.push_back(paramsResizeQSVEnc[ielem]);
            }
            std::vector<std::string> paramListResizeFsr1;
            for (size_t ielem = 0; ielem < _countof(paramsResizeFsr1); ielem++) {
                paramListResizeFsr1.push_back(paramsResizeFsr1[ielem]);
            }
            std::vector<std::string> paramListResizeNis;
            for (size_t ielem = 0; ielem < _countof(paramsResizeNis); ielem++) {
                paramListResizeNis.push_back(paramsResizeNis[ielem]);
            }
            // ビルド構成で無効なサブパラメータは候補に出さない(パーサ側は他エンコーダのコマンドライン互換のため受理して無視する)
            std::vector<std::string> paramList;
            if (ENCODER_NVENC && (ENABLE_NVVFX || FOR_AUO)) vector_cat(paramList, paramListResizeNVEnc);
            if (ENCODER_QSV) vector_cat(paramList, paramListResizeQSVEnc);
            vector_cat(paramList, paramListResizeFsr1);
            for (const auto& prm : paramListResizeNis) {
                if (ENABLE_OPENCL || prm != "opt") paramList.push_back(prm); // optはOpenCLフィルタ実装でのみ対応
            }
            for (size_t ielem = 0; ielem < _countof(paramsResizeDpid); ielem++) {
                paramList.push_back(paramsResizeDpid[ielem]);
            }
            for (size_t ielem = 0; ielem < _countof(paramsResizeBicubic); ielem++) {
                paramList.push_back(paramsResizeBicubic[ielem]);
            }
            for (size_t ielem = 0; ielem < _countof(paramsResizeLibPlacebo); ielem++) {
                paramList.push_back(paramsResizeLibPlacebo[ielem]);
            }
            RGY_VPP_RESIZE_ALGO preScanAlgo = vpp->resize_algo;
            for (const auto& param : split(strInput[i], _T(","))) {
                auto pos = param.find_first_of(_T("="));
                int probe = 0;
                if (pos != std::string::npos) {
                    auto k = tolowercase(param.substr(0, pos));
                    auto v = param.substr(pos + 1);
                    if (k == _T("algo") && get_list_value(list_vpp_resize, v.c_str(), &probe)) {
                        preScanAlgo = (RGY_VPP_RESIZE_ALGO)probe;
                    }
                } else if (get_list_value(list_vpp_resize, param.c_str(), &probe)) {
                    preScanAlgo = (RGY_VPP_RESIZE_ALGO)probe;
                }
            }
            for (const auto& param : split(strInput[i], _T(","))) {
                auto pos = param.find_first_of(_T("="));
                if (pos != std::string::npos) {
                    auto param_arg = param.substr(0, pos);
                    auto param_val = param.substr(pos + 1);
                    param_arg = tolowercase(param_arg);
                    if (param_arg == _T("algo")) {
                        int value = 0;
                        if (get_list_value(list_vpp_resize, param_val.c_str(), &value)) {
                            vpp->resize_algo = (RGY_VPP_RESIZE_ALGO)value;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-radius")) {
                        try {
                            vpp->resize_libplacebo.radius = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-clamp")) {
                        try {
                            vpp->resize_libplacebo.clamp_ = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-taper")) {
                        try {
                            vpp->resize_libplacebo.taper = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-blur")) {
                        try {
                            vpp->resize_libplacebo.blur = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-antiring")) {
                        try {
                            vpp->resize_libplacebo.antiring = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("pl-cplace")) {
                        try {
                            vpp->resize_libplacebo.cplace = std::stoi(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("sharpness")) {
                        try {
                            const float val = std::stof(param_val);
                            if (preScanAlgo == RGY_VPP_RESIZE_NIS) {
                                vpp->resize_nis.sharpness = val;
                            } else {
                                vpp->resize_fsr1.sharpness = val;
                            }
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        const float val = (preScanAlgo == RGY_VPP_RESIZE_NIS) ? vpp->resize_nis.sharpness : vpp->resize_fsr1.sharpness;
                        if (val < 0.0f || val > 1.0f) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sharpness should be 0.0 - 1.0."));
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("dpid_lambda")) {
                        try {
                            size_t idx = 0;
                            const float value = std::stof(param_val, &idx);
                            if (idx != param_val.length() || !std::isfinite(value)
                                || value < FILTER_MIN_RESIZE_DPID_LAMBDA || value > FILTER_MAX_RESIZE_DPID_LAMBDA) {
                                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                                    _T("dpid_lambda should be 0.0 - 4.0."));
                                return 1;
                            }
                            vpp->resize_dpid.lambda = value;
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("cascade")) {
                        int value = 0;
                        if (get_list_value(list_vpp_resize_nis_cascade, param_val.c_str(), &value)) {
                            vpp->resize_nis.cascade = value;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize_nis_cascade);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("hdr")) {
                        int value = 0;
                        if (get_list_value(list_vpp_resize_nis_hdr, param_val.c_str(), &value)) {
                            vpp->resize_nis.hdrMode = value;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize_nis_hdr);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("opt") && ENABLE_OPENCL) {
                        int value = 0;
                        if (get_list_value(list_vpp_resize_nis_opt, param_val.c_str(), &value)) {
                            vpp->resize_nis.opt = value;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize_nis_opt);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("b")) {
                        try {
                            vpp->resize_bicubic.b = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (param_arg == _T("c")) {
                        try {
                            vpp->resize_bicubic.c = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    }
                    if (ENCODER_NVENC && std::find_if(paramListResizeNVEnc.begin(), paramListResizeNVEnc.end(), [param_arg](const std::string& str) {
                        return param_arg == char_to_tstring(str);
                        }) != paramListResizeNVEnc.end()) {
                        continue;
                    }
                    if (ENCODER_QSV && std::find_if(paramListResizeQSVEnc.begin(), paramListResizeQSVEnc.end(), [param_arg](const std::string& str) {
                        return param_arg == char_to_tstring(str);
                        }) != paramListResizeQSVEnc.end()) {
                        continue;
                    }
                    if (std::find_if(paramListResizeNis.begin(), paramListResizeNis.end(), [param_arg](const std::string& str) {
                        return param_arg == char_to_tstring(str);
                        }) != paramListResizeNis.end()) {
                        continue;
                    }
                    print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                    return 1;
                } else {
                    int value = 0;
                    if (get_list_value(list_vpp_resize, param.c_str(), &value)) {
                        vpp->resize_algo = (RGY_VPP_RESIZE_ALGO)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name), param, list_vpp_resize_help);
                        return 1;
                    }
                }
            }
        } else {
            int value;
            if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_vpp_resize, strInput[i]))) {
                print_cmd_error_invalid_value(option_name, strInput[i], list_vpp_resize_help);
                return 1;
            }
            vpp->resize_algo = (RGY_VPP_RESIZE_ALGO)value;
        }
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
                // 参照は付け替えられないので、push_backを終えてから末尾の参照を取る。
                if (param_arg == _T("matrix") || param_arg == _T("colormatrix")) {
                    if (vpp->colorspace.convs.back().from.matrix != vpp->colorspace.convs.back().to.matrix) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                    }
                    auto& conv = vpp->colorspace.convs.back();
                    if (!parse((int *)&conv.from.matrix, (int *)&conv.to.matrix, param_val, list_colormatrix)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_colormatrix);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colorprim")) {
                    if (vpp->colorspace.convs.back().from.colorprim != vpp->colorspace.convs.back().to.colorprim) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                    }
                    auto& conv = vpp->colorspace.convs.back();
                    if (!parse((int *)&conv.from.colorprim, (int *)&conv.to.colorprim, param_val, list_colorprim)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_colorprim);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("transfer")) {
                    if (vpp->colorspace.convs.back().from.transfer != vpp->colorspace.convs.back().to.transfer) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                    }
                    auto& conv = vpp->colorspace.convs.back();
                    if (!parse((int *)&conv.from.transfer, (int *)&conv.to.transfer, param_val, list_transfer)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("should be specified by <string>:<string>."), list_transfer);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("range") || param_arg == _T("colorrange")) {
                    if (vpp->colorspace.convs.back().from.colorrange != vpp->colorspace.convs.back().to.colorrange) {
                        vpp->colorspace.convs.push_back(ColorspaceConv());
                    }
                    auto& conv = vpp->colorspace.convs.back();
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
                        double desat_base = std::stod(param_val);
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

    if (IS_OPTION("vpp-libplacebo-tonemapping")) {
        vpp->libplacebo_tonemapping.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "src_csp", "dst_csp", "src_max", "src_min", "dst_max", "dst_min", "dynamic_peak_detection", "smooth_period",
            "scene_threshold_low", "scene_threshold_high", "percentile", "black_cutoff", "gamut_mapping",
            "tonemapping_function", "contrast_recovery", "contrast_smoothness", "inverse_tone_mapping", "visualize_lut", "show_clipping",
            "use_dovi", "lut_path", "lut_type", "dst_pl_transfer", "dst_pl_colorprim",
            "knee_adaptation", "knee_min", "knee_max", "knee_default", "knee_offset", "slope_tuning",
            "slope_offset", "spline_contrast", "reinhard_contrast", "linear_knee", "exposure"
        };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->libplacebo_tonemapping.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("src_csp")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_csp, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.src_csp = (VppLibplaceboToneMappingCSP)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_csp);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dst_csp")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_csp, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.dst_csp = (VppLibplaceboToneMappingCSP)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_csp);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("src_max")) {
                    try {
                        vpp->libplacebo_tonemapping.src_max = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("src_min")) {
                    try {
                        vpp->libplacebo_tonemapping.src_min = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dst_max")) {
                    try {
                        vpp->libplacebo_tonemapping.dst_max = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dst_min")) {
                    try {
                        vpp->libplacebo_tonemapping.dst_min = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dynamic_peak_detection")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->libplacebo_tonemapping.dynamic_peak_detection = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smooth_period")) {
                    try {
                        vpp->libplacebo_tonemapping.smooth_period = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scene_threshold_low")) {
                    try {
                        vpp->libplacebo_tonemapping.scene_threshold_low = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scene_threshold_high")) {
                    try {
                        vpp->libplacebo_tonemapping.scene_threshold_high = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("percentile")) {
                    try {
                        vpp->libplacebo_tonemapping.percentile = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("black_cutoff")) {
                    try {
                        vpp->libplacebo_tonemapping.black_cutoff = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gamut_mapping")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_gamut_mapping, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.gamut_mapping = (VppLibplaceboToneMappingGamutMapping)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_gamut_mapping);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tonemapping_function")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_function, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.tonemapping_function = (VppLibplaceboToneMappingFunction)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_function);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("metadata")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_metadata, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.metadata = (VppLibplaceboToneMappingMetadata)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_metadata);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast_recovery")) {
                    try {
                        vpp->libplacebo_tonemapping.contrast_recovery = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast_smoothness")) {
                    try {
                        vpp->libplacebo_tonemapping.contrast_smoothness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("inverse_tone_mapping")) {
                    bool b = false;
                    if (cmd_string_to_bool(&b, param_val) != 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    vpp->libplacebo_tonemapping.inverse_tone_mapping = b;
                    continue;
                }
                if (param_arg == _T("visualize_lut")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->libplacebo_tonemapping.visualize_lut = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("show_clipping")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->libplacebo_tonemapping.show_clipping = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("use_dovi")) {
                    try {
                        vpp->libplacebo_tonemapping.use_dovi = std::stoi(param_val);
                    } catch (...) {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            vpp->libplacebo_tonemapping.use_dovi = (b) ? 1 : 0;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("lut_path")) {
                    vpp->libplacebo_tonemapping.lut_path = param_val;
                    continue;
                }
                if (param_arg == _T("lut_type")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_lut_type, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.lut_type = (VppLibplaceboToneMappingLUTType)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_lut_type);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dst_pl_transfer")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_transfer, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.dst_pl_transfer = (VppLibplaceboToneMappingTransfer)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_transfer);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dst_pl_colorprim")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_colorprim, param_val.c_str(), &value)) {
                        vpp->libplacebo_tonemapping.dst_pl_colorprim = (VppLibplaceboToneMappingColorprim)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_colorprim);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("knee_adaptation")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.st2094.knee_adaptation = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("knee_min")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.st2094.knee_min = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("knee_max")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.st2094.knee_max = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("knee_default")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.st2094.knee_default = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("knee_offset")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.bt2390.knee_offset = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("slope_tuning")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.spline.slope_tuning = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("slope_offset")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.spline.slope_offset = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("spline_contrast")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.spline.spline_contrast = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("reinhard_contrast")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.reinhard.contrast = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }

                if (param_arg == _T("linear_knee")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.mobius.linear_knee = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("exposure")) {
                    try {
                        vpp->libplacebo_tonemapping.tone_constants.linear.exposure = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("dynamic_peak_detection")) {
                    vpp->libplacebo_tonemapping.dynamic_peak_detection = true;
                    continue;
                }
                if (param == _T("visualize_lut")) {
                    vpp->libplacebo_tonemapping.visualize_lut = true;
                    continue;
                }
                if (param == _T("show_clipping")) {
                    vpp->libplacebo_tonemapping.show_clipping = true;
                    continue;
                }
                if (param == _T("use_dovi")) {
                    vpp->libplacebo_tonemapping.use_dovi = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-libplacebo-tonemapping-lut")) {
        i++;
        vpp->libplacebo_tonemapping.lut_path = strInput[i];
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
                    bool b = false; int value = 0;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->afs.tune = b ? AFS_TUNE_MODE_FINAL : AFS_TUNE_MODE_NONE;
                    } else if (get_list_value(list_afs_tune_mode, param_val.c_str(), &value)) {
                        vpp->afs.tune = (AFS_TUNE_MODE)value;
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
                    vpp->afs.tune = AFS_TUNE_MODE_FINAL;
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
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "planes", "field", "nsize", "nns", "quality", "prescreen", "errortype", "prec", "clamp", "double_height", "weightfile" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == std::string::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto param_arg = tolowercase(param.substr(0, pos));
            const auto param_val = param.substr(pos + 1);
            const auto result = parse_one_nnedi_param(vpp->nnedi, param_arg, param_val, option_name, true);
            if (result > 0) return 1;
            if (result < 0) {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-nnedi-upscale") && ENABLE_VPP_FILTER_NNEDI_UPSCALE) {
        vpp->nnediUpscale.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) return 0;
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "nsize", "nns", "quality", "prescreen", "errortype", "prec", "clamp", "shift", "weightfile" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == std::string::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto param_arg = tolowercase(param.substr(0, pos));
            const auto param_val = param.substr(pos + 1);
            const auto result = parse_one_nnedi_upscale_param(vpp->nnediUpscale, param_arg, param_val, option_name);
            if (result > 0) return 1;
            if (result < 0) {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
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
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->yadif.log = b;
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
                    vpp->yadif.log = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-ivtc") && ENABLE_VPP_FILTER_IVTC) {
        vpp->ivtc.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "guide", "post", "cycle", "drop", "combthresh", "cleanfrac",
            "dthresh", "chroma", "back", "y0", "y1", "nt", "cthresh", "combpel", "scthresh", "cadlock", "gthresh", "vthresh", "expand", "mixed", "hysteresis", "tff", "log" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->ivtc.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("guide")) {
                    try {
                        const int g = std::stoi(param_val);
                        if (g < 0 || g > 2) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: 0 (min-combing), 1 (2-way + combed-override), 2 (PAL 2:2)."));
                            return 1;
                        }
                        vpp->ivtc.guide = g;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("post")) {
                    try {
                        const int p = std::stoi(param_val);
                        if (p != 0 && p != 2) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: 0 (off), 2 (adaptive blend)."));
                            return 1;
                        }
                        vpp->ivtc.post = p;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cycle")) {
                    if (param_val == _T("auto") || param_val == _T("-1")) {
                        vpp->ivtc.cycle = -1;
                        continue;
                    }
                    try {
                        const int c = std::stoi(param_val);
                        if (c != 0 && (c < 2 || c > 16)) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: auto, 0 (disabled), or 2..16."));
                            return 1;
                        }
                        vpp->ivtc.cycle = c;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("drop")) {
                    try {
                        const int d = std::stoi(param_val);
                        if (d != 1) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Only drop=1 is supported in this build."));
                            return 1;
                        }
                        vpp->ivtc.drop = d;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("combthresh")) {
                    try {
                        vpp->ivtc.combThresh = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cleanfrac")) {
                    try {
                        vpp->ivtc.cleanFrac = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dthresh")) {
                    try {
                        const int d = std::stoi(param_val);
                        if (d < 0 || d > 255) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported range: 0..255 (0 = disable gate)."));
                            return 1;
                        }
                        vpp->ivtc.dthresh = d;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool bb = false;
                    if (!cmd_string_to_bool(&bb, param_val)) {
                        vpp->ivtc.chroma = bb;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("back")) {
                    try {
                        const int b = std::stoi(param_val);
                        if (b != 0 && b != 1) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: 0 (always test P), 1 (only test P when combed)."));
                            return 1;
                        }
                        vpp->ivtc.back = b;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("y0")) {
                    try {
                        vpp->ivtc.y0 = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("y1")) {
                    try {
                        vpp->ivtc.y1 = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nt")) {
                    try {
                        vpp->ivtc.nt = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cthresh")) {
                    try {
                        vpp->ivtc.cthresh = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("combpel")) {
                    try {
                        vpp->ivtc.combPel = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scthresh")) {
                    try {
                        vpp->ivtc.scThresh = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cadlock")) {
                    // Accept: auto|-1 (auto), on/true/1 (on), off/false/0 (off).
                    if (param_val == _T("auto") || param_val == _T("-1")) {
                        vpp->ivtc.cadenceLock = -1;
                        continue;
                    }
                    bool bb = false;
                    if (!cmd_string_to_bool(&bb, param_val)) {
                        vpp->ivtc.cadenceLock = bb ? 1 : 0;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: auto, on, off (auto enables when guide >= 1)."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("hysteresis")) {
                    try {
                        vpp->ivtc.hysteresis = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gthresh")) {
                    try {
                        const int g = std::stoi(param_val);
                        if (g < 0 || g > 100) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported range: 0..100 (percent). 0 disables cadence override."));
                            return 1;
                        }
                        vpp->ivtc.gthresh = g;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("vthresh")) {
                    try {
                        const int v = std::stoi(param_val);
                        if (v < 0 || v > 256) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported range: 0..256 (per-WG zigzag pixel count). 0 disables post-assembly combing gate."));
                            return 1;
                        }
                        vpp->ivtc.vthresh = v;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("expand")) {
                    // Accept: auto|-1 (auto), on/true/1 (on), off/false/0 (off).
                    if (param_val == _T("auto") || param_val == _T("-1")) {
                        vpp->ivtc.expand = -1;
                        continue;
                    }
                    bool bb = false;
                    if (!cmd_string_to_bool(&bb, param_val)) {
                        vpp->ivtc.expand = bb ? 1 : 0;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: auto, on, off (auto enables DGDecode-style RFF expansion when guide >= 1 and input is soft-telecine)."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mixed")) {
                    bool bb = false;
                    if (!cmd_string_to_bool(&bb, param_val)) {
                        vpp->ivtc.mixed = bb ? 1 : 0;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: on, off."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tff")) {
                    if (param_val == _T("auto") || param_val == _T("-1")) {
                        vpp->ivtc.tff = -1;
                    } else {
                        bool bb = false;
                        if (!cmd_string_to_bool(&bb, param_val)) {
                            vpp->ivtc.tff = bb ? 1 : 0;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool bb = false;
                    if (!cmd_string_to_bool(&bb, param_val)) {
                        vpp->ivtc.log = bb;
                        if (bb) {
                            vpp->ivtc.logPath = _T("");
                        }
                    } else {
                        vpp->ivtc.log = true;
                        vpp->ivtc.logPath = param_val;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("log")) {
                    vpp->ivtc.log = true;
                    vpp->ivtc.logPath = _T("");
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-bwdif") && ENABLE_VPP_FILTER_BWDIF) {
        vpp->bwdif.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mode", "order", "thr", "deint", "log" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->bwdif.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->bwdif.log = b;
                        if (b) {
                            vpp->bwdif.logPath = _T("");
                        }
                    } else {
                        vpp->bwdif.log = true;
                        vpp->bwdif.logPath = param_val;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    if (param_val == _T("frame")) {
                        vpp->bwdif.mode = VppBwdifMode::Frame;
                    } else if (param_val == _T("bob")) {
                        vpp->bwdif.mode = VppBwdifMode::Bob;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: frame, bob."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("order")) {
                    if (param_val == _T("auto") || param_val == _T("-1")) {
                        vpp->bwdif.order = VppBwdifOrder::Auto;
                    } else if (param_val == _T("tff")) {
                        vpp->bwdif.order = VppBwdifOrder::TFF;
                    } else if (param_val == _T("bff")) {
                        vpp->bwdif.order = VppBwdifOrder::BFF;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: auto, tff, bff."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thr")) {
                    try {
                        vpp->bwdif.thr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("deint")) {
                    if (param_val == _T("all")) {
                        vpp->bwdif.deint = VppBwdifDeint::All;
                    } else if (param_val == _T("interlaced")) {
                        vpp->bwdif.deint = VppBwdifDeint::Interlaced;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: all, interlaced."));
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("log")) {
                    vpp->bwdif.log = true;
                    vpp->bwdif.logPath = _T("");
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-maa") && ENABLE_VPP_FILTER_MAA) {
        vpp->maa.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "ss", "aa", "aac", "mask", "mthresh", "chroma", "show", "edge" };

        // Track whether the user explicitly set aac so we can default it to
        // max(0, aa - 8) per analysis/maa2_investigation/05_parameter_design.md § 2
        // when only aa is specified.
        bool aacUserSet = false;
        bool aaUserSet  = false;

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->maa.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ss")) {
                    try {
                        vpp->maa.ss = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("aa")) {
                    try {
                        vpp->maa.aa = std::stoi(param_val);
                        aaUserSet = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("aac")) {
                    try {
                        vpp->maa.aac = std::stoi(param_val);
                        aacUserSet = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mask")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->maa.mask = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: on, off, true, false."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mthresh")) {
                    try {
                        vpp->maa.mthresh = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->maa.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: on, off, true, false."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("show")) {
                    try {
                        vpp->maa.show = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("edge")) {
                    const auto edge = check_str_list(option_name, param_arg, param_val,
                        { _T("sobel"), _T("prewitt"), _T("sobel_full"), _T("scharr"), _T("kirsch"), _T("laplacian") });
                    if (!edge) return 1;
                    vpp->maa.edge = *edge;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }

        // Default aac = max(0, aa - 8) when the user set aa but not aac.
        // Matches the reference MAA2 script's `aac = aa - 8` default behaviour.
        if (aaUserSet && !aacUserSet) {
            vpp->maa.aac = std::max(0, vpp->maa.aa - 8);
        }
        return 0;
    }


    if (IS_OPTION("vpp-rtgmc") && ENABLE_VPP_FILTER_RTGMC) {
        VppRtgmc parsedRtgmc;
        parsedRtgmc.enable = true;

        const auto paramList = std::vector<std::string>{
            "enable", "preset", "tuning", "border", "lossless", "input_type", "prog_sad_mask", "prog_sad_mask_gamma", "source_match", "match_tr1", "match_tr2",
            "match_preset", "match_preset2", "match_edi", "match_enhance",
            "order", "tr0", "rep0-thin", "rep0-pad", "search_refine", "edi",
            "nnsize", "nneurons", "ediqual", "chroma_edi",
            "mv_spatial_refine",
            "tv_range", "chroma_motion",
            "noise_process", "ezdenoise", "ezkeepgrain", "denoiser", "sigma", "chroma_noise",
            "noise_deint", "denoise_mc", "noise_tr", "grain_restore", "noise_restore",
            "sharpness", "limit", "smode", "slmode", "slrad", "sovs", "svthin", "sbb", "precise",
            "rep1-thin", "rep1-pad",
            "rep2-thin", "rep2-pad", "rep_chroma",
            "blksize", "search", "thsad", "thsad1", "thsad2", "thscd1", "thscd2", "pel", "levels", "overlap", "delta",
            "subpelinterp", "searchparam", "pelsearch", "search_early_sad", "spatial_early_sad",
            "truemotion", "lambda", "lsad", "pnew", "plevel", "globalmotion", "dct", "useflag",
            "delta_analyze", "delta_tr1", "delta_tr2"
        };
        bool userSetTr2 = false;
        bool userSetSharpness = false;
        bool userSetSlmode = false;
        bool userSetNoiseProcess = false;
        bool userSetNoiseSigma = false;
        bool userSetMatchEdi = false;
        int rtgmcPresetNum = (int)parsedRtgmc.preset;
        int rtgmcMatchPresetNum = -1;
        struct RtgmcPresetOverride {
            bool tr0 = false;
            bool rep0Thin = false;
            bool rep0Pad = false;
            bool searchRefine = false;
            bool chromaMotion = false;
            bool edi = false;
            bool nnsize = false;
            bool nneurons = false;
            bool ediqual = false;
            bool matchEdi = false;
            bool analyzeDelta = false;
            bool tr1Delta = false;
            bool tr2Delta = false;
            bool rep1Thin = false;
            bool rep1Pad = false;
            bool rep2Thin = false;
            bool rep2Pad = false;
            bool smode = false;
            bool slmode = false;
            bool slrad = false;
            bool sbb = false;
            bool precise = false;
            bool blksize = false;
            bool overlap = false;
            bool search = false;
            bool pel = false;
            bool searchParam = false;
            bool pelSearch = false;
            bool searchEarlySad = false;
            bool lambda = false;
            bool noiseProcess = false;
            bool denoiser = false;
            bool noiseDeint = false;
            bool denoiseMC = false;
            bool noiseTR = false;
            bool grainRestore = false;
            bool noiseRestore = false;
            bool progSADMask = false;
        } presetOverride;
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        const auto parse_float = [&](float *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stof(param_val, &idx);
                if (idx != param_val.length() || !std::isfinite(*dst)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        const auto check_fixed_int = [&](const tstring& param_arg, const tstring& param_val, const int fixedValue) {
            int value = 0;
            if (parse_int(&value, param_arg, param_val)) return 1;
            if (value != fixedValue) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                    strsprintf(_T("%s is fixed to %d in --vpp-rtgmc; use --vpp-degrain for standalone debug."), param_arg.c_str(), fixedValue).c_str());
                return 1;
            }
            return 0;
        };
        const auto check_fixed_bool = [&](const tstring& param_arg, const tstring& param_val, const bool fixedValue) {
            bool value = false;
            if (cmd_string_to_bool(&value, param_val)) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            if (value != fixedValue) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                    strsprintf(_T("%s is fixed to %s in --vpp-rtgmc; use --vpp-degrain for standalone debug."),
                        param_arg.c_str(), fixedValue ? _T("true") : _T("false")).c_str());
                return 1;
            }
            return 0;
        };
        const auto apply_rtgmc_motion_common = [&]() {
            parsedRtgmc.analyze.tr0 = parsedRtgmc.searchPrefilter.tr0;
            parsedRtgmc.tr1.tr0 = parsedRtgmc.searchPrefilter.tr0;
            parsedRtgmc.tr2.tr0 = parsedRtgmc.searchPrefilter.tr0;
            parsedRtgmc.analyze.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
            parsedRtgmc.tr1.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
            parsedRtgmc.tr2.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
            parsedRtgmc.analyze.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
            parsedRtgmc.tr1.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
            parsedRtgmc.tr2.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
            parsedRtgmc.analyze.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
            parsedRtgmc.tr1.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
            parsedRtgmc.tr2.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
        };
        const auto apply_rtgmc_preset = [&](const VppRtgmcPreset preset, const VppRtgmcTuning tuning, const RtgmcPresetOverride& override) {
            static const int tr0[]         = { 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0 };
            static const int tr1[]         = { 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1 };
            static const int tr2[]         = { 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
            static const int rep0Thin[]    = { 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 0 };
            static const int rep1Thin[]    = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            static const int rep2Thin[]    = { 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 0 };
            static const VppRtgmcEdiMode edi[] = {
                VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3,
                VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3,
                VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3, VppRtgmcEdiMode::NNEDI3,
                VppRtgmcEdiMode::RepYadif, VppRtgmcEdiMode::Bob
            };
            static const int nnsize[]      = { 1, 1, 1, 1, 5, 5, 4, 4, 4, 4, 4 };
            static const int nneurons[]    = { 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
            static const int ediqual[]     = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            static const int smode[]       = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0 };
            static const int slmode[]      = { 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0 };
            static const int slrad[]       = { 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            static const int sbb[]         = { 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
            static const int searchRefine[]  = { 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 0 };
            static const int subpel[]      = { 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1 };
            static const int search[]      = { 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0 };
            static const int searchparam[] = { 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1 };
            static const int pelsearch[]   = { 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1 };
            static const bool chroma[]     = { true, true, true, false, false, false, false, false, false, false, false };
            static const bool precise[]    = { true, true, false, false, false, false, false, false, false, false, false };
            static const float progSADMask[] = { 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

            const int p = (int)preset;
            const int bs = tuning == VppRtgmcTuning::DVHD ? 32 : 16;
            const int bs2 = 32;
            const int blocksize = (p <= (int)VppRtgmcPreset::Fast) ? bs : bs2;
            const int overlap = (p <= (int)VppRtgmcPreset::Faster) ? blocksize / 2 : blocksize / 4;
            const int searchEarlySad = get_vpp_rtgmc_search_early_sad(preset);
            const auto defaultLambda = [](const VppDegrain& prm) {
                return ((prm.trueMotion ? 1000 : 100) * prm.blksize * prm.blksize) / (8 * 8);
            };
            parsedRtgmc.preset = preset;
            parsedRtgmc.tuning = tuning;
            if (!override.progSADMask) {
                parsedRtgmc.progSADMask = progSADMask[p];
            }
            if (!override.tr0) {
                parsedRtgmc.searchPrefilter.tr0 = tr0[p];
            }
            if (!override.rep0Thin) {
                parsedRtgmc.searchPrefilter.rep0Thin = rep0Thin[p];
            }
            if (!override.rep0Pad) {
                parsedRtgmc.searchPrefilter.rep0Pad = 0;
            }
            if (!override.searchRefine) {
                parsedRtgmc.searchPrefilter.searchRefine = searchRefine[p];
            }
            if (!override.chromaMotion) {
                parsedRtgmc.searchPrefilter.chromaMotion = chroma[p];
            }
            if (override.tr0 && !override.rep0Thin && parsedRtgmc.searchPrefilter.tr0 < 1) {
                parsedRtgmc.searchPrefilter.rep0Thin = 0;
                if (!override.rep0Pad) {
                    parsedRtgmc.searchPrefilter.rep0Pad = 0;
                }
            }
            apply_rtgmc_motion_common();
            if (!override.edi) {
                parsedRtgmc.edi.mode = edi[p];
            }
            if (!override.nnsize) {
                parsedRtgmc.edi.nnsize = nnsize[p];
            }
            if (!override.nneurons) {
                parsedRtgmc.edi.nneurons = nneurons[p];
            }
            if (!override.ediqual) {
                parsedRtgmc.edi.ediqual = ediqual[p];
            }
            if (!override.matchEdi) {
                parsedRtgmc.matchEdi.mode = parsedRtgmc.edi.mode;
            }
            parsedRtgmc.matchEdi.nnsize = parsedRtgmc.edi.nnsize;
            parsedRtgmc.matchEdi.nneurons = parsedRtgmc.edi.nneurons;
            parsedRtgmc.matchEdi.ediqual = parsedRtgmc.edi.ediqual;
            if (!override.analyzeDelta) {
                parsedRtgmc.analyze.delta = tr0[p];
            }
            if (!override.tr1Delta) {
                parsedRtgmc.tr1.delta = tr1[p];
            }
            if (!override.tr2Delta) {
                parsedRtgmc.tr2.delta = tr2[p];
            }
            if (!override.rep1Thin) {
                parsedRtgmc.rep1.repThin = rep1Thin[p];
            }
            if (!override.rep1Pad) {
                parsedRtgmc.rep1.repPad = 0;
            }
            if (!override.rep2Thin) {
                parsedRtgmc.rep2.repThin = rep2Thin[p];
            }
            if (!override.rep2Pad) {
                parsedRtgmc.rep2.repPad = 0;
            }
            if (!override.smode) {
                parsedRtgmc.retouch.smode = smode[p];
            }
            if (!override.slmode) {
                parsedRtgmc.retouch.slmode = slmode[p];
            }
            if (!override.slrad) {
                parsedRtgmc.retouch.slrad = slrad[p];
            }
            if (!override.sbb) {
                parsedRtgmc.retouch.sbb = sbb[p];
            }
            if (!override.precise) {
                parsedRtgmc.retouch.precise = precise[p];
            }
            for (auto *stagePrm : { &parsedRtgmc.analyze, &parsedRtgmc.tr1, &parsedRtgmc.tr2 }) {
                if (!override.blksize) {
                    stagePrm->blksize = blocksize;
                }
                if (!override.overlap) {
                    stagePrm->overlap = overlap;
                }
                if (!override.search) {
                    stagePrm->search = search[p];
                }
                if (!override.pel) {
                    stagePrm->pel = subpel[p];
                }
                if (!override.searchParam) {
                    stagePrm->searchParam = searchparam[p];
                }
                if (!override.pelSearch) {
                    stagePrm->pelSearch = pelsearch[p];
                }
                if (!override.searchEarlySad) {
                    stagePrm->searchEarlySad = searchEarlySad;
                }
                if (!override.lambda) {
                    stagePrm->lambda = defaultLambda(*stagePrm);
                }
            }
            if (!override.denoiser) {
                parsedRtgmc.noise.denoiser = VppRtgmcNoiseDenoiser::FFT3D;
            }
            if (!override.denoiseMC) {
                parsedRtgmc.noise.denoiseMC = false;
            }
            if (!override.noiseTR) {
                parsedRtgmc.noise.noiseTR = 1;
            }
            if (!override.noiseDeint) {
                parsedRtgmc.noise.noiseDeint = VppRtgmcNoiseDeint::None;
            }
            if (!override.noiseProcess) {
                parsedRtgmc.noise.noiseProcess = (preset == VppRtgmcPreset::Placebo || preset == VppRtgmcPreset::VerySlow) ? 2 : 0;
            }
            if (!override.grainRestore) {
                parsedRtgmc.noise.grainRestore = parsedRtgmc.noise.noiseProcess == 2 ? 0.3f : 0.0f;
            }
            if (!override.noiseRestore) {
                parsedRtgmc.noise.noiseRestore = parsedRtgmc.noise.noiseProcess == 2 ? 0.1f : 0.0f;
            }
        };
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            vpp->rtgmc = parsedRtgmc;
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("preset")) {
                    int value = 0;
                    const auto preset = tolowercase(trim(param_val, _T("\"")));
                    if (get_list_value(list_vpp_rtgmc_preset, preset.c_str(), &value)) {
                        rtgmcPresetNum = value;
                        parsedRtgmc.preset = (VppRtgmcPreset)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_preset);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tuning")) {
                    int value = 0;
                    const auto tuning = tolowercase(trim(param_val, _T("\"")));
                    if (get_list_value(list_vpp_rtgmc_tuning, tuning.c_str(), &value)) {
                        parsedRtgmc.tuning = (VppRtgmcTuning)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_tuning);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("border")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.border = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lossless")) {
                    if (parse_int(&parsedRtgmc.lossless, param_arg, param_val)) return 1;
                    if (parsedRtgmc.lossless < 0 || parsedRtgmc.lossless > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("lossless should be 0 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("input_type")) {
                    if (parse_int(&parsedRtgmc.inputType, param_arg, param_val)) return 1;
                    if (parsedRtgmc.inputType < 0 || parsedRtgmc.inputType > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("input_type should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("prog_sad_mask")) {
                    if (parse_float(&parsedRtgmc.progSADMask, param_arg, param_val)) return 1;
                    if (parsedRtgmc.progSADMask < 0.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("prog_sad_mask should be 0.0 or larger."));
                        return 1;
                    }
                    presetOverride.progSADMask = true;
                    continue;
                }
                if (param_arg == _T("prog_sad_mask_gamma")) {
                    if (parse_float(&parsedRtgmc.progSADMaskGamma, param_arg, param_val)) return 1;
                    if (parsedRtgmc.progSADMaskGamma <= 0.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("prog_sad_mask_gamma should be larger than 0.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("source_match")) {
                    if (parse_int(&parsedRtgmc.sourceMatch, param_arg, param_val)) return 1;
                    if (parsedRtgmc.sourceMatch < 0 || parsedRtgmc.sourceMatch > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("source_match should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("match_tr1")) {
                    if (parse_int(&parsedRtgmc.matchTR1, param_arg, param_val)) return 1;
                    if (parsedRtgmc.matchTR1 < 0 || parsedRtgmc.matchTR1 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("match_tr1 should be 0 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("match_tr2")) {
                    if (parse_int(&parsedRtgmc.matchTR2, param_arg, param_val)) return 1;
                    if (parsedRtgmc.matchTR2 < 0 || parsedRtgmc.matchTR2 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("match_tr2 should be 0 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("match_preset") || param_arg == _T("match_preset2")) {
                    int value = 0;
                    const auto preset = tolowercase(trim(param_val, _T("\"")));
                    if (!get_list_value(list_vpp_rtgmc_preset, preset.c_str(), &value)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_preset);
                        return 1;
                    }
                    if ((VppRtgmcPreset)value == VppRtgmcPreset::Draft) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("Draft MatchPreset is unsupported in the current source refinement path."));
                        return 1;
                    }
                    rtgmcMatchPresetNum = (rtgmcMatchPresetNum < 0) ? value : std::min(rtgmcMatchPresetNum, value);
                    if (value < rtgmcPresetNum) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("MatchPreset cannot use a slower setting than Preset."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("match_edi")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_edi_mode, param_val.c_str(), &value)) {
                        parsedRtgmc.matchEdi.mode = (VppRtgmcEdiMode)value;
                        userSetMatchEdi = true;
                        presetOverride.matchEdi = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_edi_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("match_enhance")) {
                    try {
                        parsedRtgmc.matchEnhance = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (parsedRtgmc.matchEnhance < 0.0f || parsedRtgmc.matchEnhance > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("match_enhance should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("order")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_bob_order, param_val.c_str(), &value)) {
                        parsedRtgmc.bob.order = (VppRtgmcBobOrder)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_bob_order);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr0")) {
                    if (parse_int(&parsedRtgmc.searchPrefilter.tr0, param_arg, param_val)) return 1;
                    if (parsedRtgmc.searchPrefilter.tr0 < -1 || parsedRtgmc.searchPrefilter.tr0 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr0 should be -1, 0, 1, or 2."));
                        return 1;
                    }
                    const int analyzeTr0 = std::max(parsedRtgmc.searchPrefilter.tr0, 0);
                    parsedRtgmc.analyze.tr0 = analyzeTr0;
                    parsedRtgmc.tr1.tr0 = analyzeTr0;
                    parsedRtgmc.tr2.tr0 = analyzeTr0;
                    presetOverride.tr0 = true;
                    continue;
                }
                if (param_arg == _T("rep0-thin")) {
                    if (parse_int(&parsedRtgmc.searchPrefilter.rep0Thin, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_thin(parsedRtgmc.searchPrefilter.rep0Thin)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep0-thin should be 0 - 7."));
                        return 1;
                    }
                    parsedRtgmc.analyze.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
                    parsedRtgmc.tr1.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
                    parsedRtgmc.tr2.rep0 = parsedRtgmc.searchPrefilter.rep0Thin;
                    presetOverride.rep0Thin = true;
                    continue;
                }
                if (param_arg == _T("rep0-pad")) {
                    if (parse_int(&parsedRtgmc.searchPrefilter.rep0Pad, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_pad(parsedRtgmc.searchPrefilter.rep0Pad)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep0-pad should be 0 - 3."));
                        return 1;
                    }
                    presetOverride.rep0Pad = true;
                    continue;
                }
                if (param_arg == _T("search_refine")) {
                    if (parse_int(&parsedRtgmc.searchPrefilter.searchRefine, param_arg, param_val)) return 1;
                    if (parsedRtgmc.searchPrefilter.searchRefine < 0 || parsedRtgmc.searchPrefilter.searchRefine > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("search_refine should be 0 - 3."));
                        return 1;
                    }
                    parsedRtgmc.analyze.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
                    parsedRtgmc.tr1.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
                    parsedRtgmc.tr2.searchRefine = parsedRtgmc.searchPrefilter.searchRefine;
                    presetOverride.searchRefine = true;
                    continue;
                }
                if (param_arg == _T("mv_spatial_refine")) {
                    if (tolowercase(param_val) == _T("auto")) {
                        parsedRtgmc.mvSpatialRefine = FILTER_DEFAULT_RTGMC_MV_SPATIAL_REFINE;
                    } else if (parse_int(&parsedRtgmc.mvSpatialRefine, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedRtgmc.mvSpatialRefine < -1) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("mv_spatial_refine should be auto, -1, or greater."));
                        return 1;
                    }
                    parsedRtgmc.analyze.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
                    parsedRtgmc.tr1.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
                    parsedRtgmc.tr2.mvSpatialRefine = parsedRtgmc.mvSpatialRefine;
                    continue;
                }
                if (param_arg == _T("tv_range")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.searchPrefilter.tvRange = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma_motion")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.searchPrefilter.chromaMotion = b;
                        presetOverride.chromaMotion = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("noise_process")) {
                    if (parse_int(&parsedRtgmc.noise.noiseProcess, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.noiseProcess < 0 || parsedRtgmc.noise.noiseProcess > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("noise_process should be 0 - 2."));
                        return 1;
                    }
                    userSetNoiseProcess = true;
                    presetOverride.noiseProcess = true;
                    continue;
                }
                if (param_arg == _T("ezdenoise")) {
                    if (parse_float(&parsedRtgmc.noise.ezDenoise, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.ezDenoise < 0.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("ezdenoise should be 0.0 or larger."));
                        return 1;
                    }
                    if (!userSetNoiseProcess) {
                        parsedRtgmc.noise.noiseProcess = 1;
                    }
                    presetOverride.noiseProcess = true;
                    if (!userSetNoiseSigma) {
                        parsedRtgmc.noise.sigma = parsedRtgmc.noise.ezDenoise;
                    }
                    continue;
                }
                if (param_arg == _T("ezkeepgrain")) {
                    if (parse_float(&parsedRtgmc.noise.ezKeepGrain, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.ezKeepGrain < 0.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("ezkeepgrain should be 0.0 or larger."));
                        return 1;
                    }
                    if (!userSetNoiseProcess) {
                        parsedRtgmc.noise.noiseProcess = 2;
                    }
                    presetOverride.noiseProcess = true;
                    continue;
                }
                if (param_arg == _T("denoiser")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_noise_denoiser, param_val.c_str(), &value)) {
                        parsedRtgmc.noise.denoiser = (VppRtgmcNoiseDenoiser)value;
                        presetOverride.denoiser = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_noise_denoiser);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("noise_deint")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_noise_deint, param_val.c_str(), &value)) {
                        parsedRtgmc.noise.noiseDeint = (VppRtgmcNoiseDeint)value;
                        presetOverride.noiseDeint = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_noise_deint);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    if (parse_float(&parsedRtgmc.noise.sigma, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.sigma < 0.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sigma should be 0.0 or larger."));
                        return 1;
                    }
                    userSetNoiseSigma = true;
                    continue;
                }
                if (param_arg == _T("chroma_noise")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.noise.chromaNoise = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("denoise_mc")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.noise.denoiseMC = b;
                        presetOverride.denoiseMC = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("noise_tr")) {
                    if (parse_int(&parsedRtgmc.noise.noiseTR, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.noiseTR < 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("noise_tr should be 0 or larger."));
                        return 1;
                    }
                    presetOverride.noiseTR = true;
                    continue;
                }
                if (param_arg == _T("grain_restore")) {
                    if (parse_float(&parsedRtgmc.noise.grainRestore, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.grainRestore < 0.0f || parsedRtgmc.noise.grainRestore > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("grain_restore should be 0.0 - 1.0."));
                        return 1;
                    }
                    presetOverride.grainRestore = true;
                    continue;
                }
                if (param_arg == _T("noise_restore")) {
                    if (parse_float(&parsedRtgmc.noise.noiseRestore, param_arg, param_val)) return 1;
                    if (parsedRtgmc.noise.noiseRestore < 0.0f || parsedRtgmc.noise.noiseRestore > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("noise_restore should be 0.0 - 1.0."));
                        return 1;
                    }
                    presetOverride.noiseRestore = true;
                    continue;
                }
                if (param_arg == _T("edi")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_edi_mode, param_val.c_str(), &value)) {
                        parsedRtgmc.edi.mode = (VppRtgmcEdiMode)value;
                        presetOverride.edi = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_edi_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nnsize")) {
                    if (parse_int(&parsedRtgmc.edi.nnsize, param_arg, param_val)) return 1;
                    if (parsedRtgmc.edi.nnsize < 0 || parsedRtgmc.edi.nnsize > 6) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("nnsize should be 0 - 6."));
                        return 1;
                    }
                    presetOverride.nnsize = true;
                    continue;
                }
                if (param_arg == _T("nneurons")) {
                    if (parse_int(&parsedRtgmc.edi.nneurons, param_arg, param_val)) return 1;
                    if (parsedRtgmc.edi.nneurons < 0 || parsedRtgmc.edi.nneurons > 4) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("nneurons should be 0 - 4."));
                        return 1;
                    }
                    presetOverride.nneurons = true;
                    continue;
                }
                if (param_arg == _T("ediqual")) {
                    if (parse_int(&parsedRtgmc.edi.ediqual, param_arg, param_val)) return 1;
                    if (parsedRtgmc.edi.ediqual < 1 || parsedRtgmc.edi.ediqual > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("ediqual should be 1 - 2."));
                        return 1;
                    }
                    presetOverride.ediqual = true;
                    continue;
                }
                if (param_arg == _T("chroma_edi")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_chroma_edi_mode, param_val.c_str(), &value)) {
                        parsedRtgmc.edi.chromaEdi = (VppRtgmcChromaEdiMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_chroma_edi_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sharpness")) {
                    userSetSharpness = true;
                    try {
                        parsedRtgmc.retouch.sharpness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (parsedRtgmc.retouch.sharpness < 0.0f || parsedRtgmc.retouch.sharpness > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sharpness should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("limit")) {
                    try {
                        parsedRtgmc.retouch.limit = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (parsedRtgmc.retouch.limit < 0.0f || parsedRtgmc.retouch.limit > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("limit should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smode")) {
                    if (parse_int(&parsedRtgmc.retouch.smode, param_arg, param_val)) return 1;
                    if (parsedRtgmc.retouch.smode < 0 || parsedRtgmc.retouch.smode > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("smode should be 0 - 2."));
                        return 1;
                    }
                    presetOverride.smode = true;
                    continue;
                }
                if (param_arg == _T("slmode")) {
                    userSetSlmode = true;
                    if (parse_int(&parsedRtgmc.retouch.slmode, param_arg, param_val)) return 1;
                    if (parsedRtgmc.retouch.slmode < 0 || parsedRtgmc.retouch.slmode > 4) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("slmode should be 0 - 4."));
                        return 1;
                    }
                    presetOverride.slmode = true;
                    continue;
                }
                if (param_arg == _T("slrad")) {
                    if (parse_int(&parsedRtgmc.retouch.slrad, param_arg, param_val)) return 1;
                    if (parsedRtgmc.retouch.slrad < 0 || parsedRtgmc.retouch.slrad > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("slrad should be 0 - 3."));
                        return 1;
                    }
                    presetOverride.slrad = true;
                    continue;
                }
                if (param_arg == _T("sovs")) {
                    if (parse_int(&parsedRtgmc.retouch.sovs, param_arg, param_val)) return 1;
                    if (parsedRtgmc.retouch.sovs < 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sovs should be 0 or greater."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("svthin")) {
                    try {
                        parsedRtgmc.retouch.svthin = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (parsedRtgmc.retouch.svthin < 0.0f || parsedRtgmc.retouch.svthin > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("svthin should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sbb")) {
                    if (parse_int(&parsedRtgmc.retouch.sbb, param_arg, param_val)) return 1;
                    if (parsedRtgmc.retouch.sbb < 0 || parsedRtgmc.retouch.sbb > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sbb should be 0 - 3."));
                        return 1;
                    }
                    presetOverride.sbb = true;
                    continue;
                }
                if (param_arg == _T("precise")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.retouch.precise = b;
                        presetOverride.precise = true;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep1-thin")) {
                    if (parse_int(&parsedRtgmc.rep1.repThin, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_thin(parsedRtgmc.rep1.repThin)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep1-thin should be 0 - 7."));
                        return 1;
                    }
                    presetOverride.rep1Thin = true;
                    continue;
                }
                if (param_arg == _T("rep1-pad")) {
                    if (parse_int(&parsedRtgmc.rep1.repPad, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_pad(parsedRtgmc.rep1.repPad)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep1-pad should be 0 - 3."));
                        return 1;
                    }
                    presetOverride.rep1Pad = true;
                    continue;
                }
                if (param_arg == _T("rep2-thin")) {
                    if (parse_int(&parsedRtgmc.rep2.repThin, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_thin(parsedRtgmc.rep2.repThin)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep2-thin should be 0 - 7."));
                        return 1;
                    }
                    presetOverride.rep2Thin = true;
                    continue;
                }
                if (param_arg == _T("rep2-pad")) {
                    if (parse_int(&parsedRtgmc.rep2.repPad, param_arg, param_val)) return 1;
                    if (!is_valid_rtgmc_rep_pad(parsedRtgmc.rep2.repPad)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep2-pad should be 0 - 3."));
                        return 1;
                    }
                    presetOverride.rep2Pad = true;
                    continue;
                }
                if (param_arg == _T("rep_chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedRtgmc.rep1.repChroma = b;
                        parsedRtgmc.rep2.repChroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thsad1") || param_arg == _T("thsad2")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    if (param_arg == _T("thsad1")) {
                        parsedRtgmc.tr1.thsad = value;
                    } else {
                        parsedRtgmc.tr2.thsad = value;
                    }
                    continue;
                }
                if (param_arg == _T("searchparam")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    if (value < 1 || value > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("searchparam should be 1 or 2 in --vpp-rtgmc."));
                        return 1;
                    }
                    parsedRtgmc.analyze.searchParam = value;
                    parsedRtgmc.tr1.searchParam = value;
                    parsedRtgmc.tr2.searchParam = value;
                    presetOverride.searchParam = true;
                    continue;
                }
                if (param_arg == _T("pelsearch")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    if (value < 1 || value > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("pelsearch should be 1 or 2 in --vpp-rtgmc."));
                        return 1;
                    }
                    parsedRtgmc.analyze.pelSearch = value;
                    parsedRtgmc.tr1.pelSearch = value;
                    parsedRtgmc.tr2.pelSearch = value;
                    presetOverride.pelSearch = true;
                    continue;
                }
                if (param_arg == _T("search_early_sad")) {
                    int value = 0;
                    if (parse_degrain_search_early_sad(&value, param_val, false)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("search_early_sad should be off or -1 - 65535."));
                        return 1;
                    }
                    parsedRtgmc.analyze.searchEarlySad = value;
                    parsedRtgmc.tr1.searchEarlySad = value;
                    parsedRtgmc.tr2.searchEarlySad = value;
                    presetOverride.searchEarlySad = true;
                    continue;
                }
                if (param_arg == _T("spatial_early_sad")) {
                    int value = 0;
                    if (parse_degrain_search_early_sad(&value, param_val, false)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("spatial_early_sad should be off or -1 - 65535."));
                        return 1;
                    }
                    parsedRtgmc.analyze.spatialEarlySad = value;
                    parsedRtgmc.tr1.spatialEarlySad = value;
                    parsedRtgmc.tr2.spatialEarlySad = value;
                    continue;
                }
                if (param_arg == _T("subpelinterp")) {
                    if (check_fixed_int(param_arg, param_val, 2)) return 1;
                    continue;
                }
                if (param_arg == _T("dct")) {
                    if (check_fixed_int(param_arg, param_val, 0)) return 1;
                    continue;
                }
                if (param_arg == _T("truemotion")) {
                    if (check_fixed_bool(param_arg, param_val, false)) return 1;
                    continue;
                }
                if (param_arg == _T("blksize") || param_arg == _T("search") || param_arg == _T("thsad")
                    || param_arg == _T("thscd1") || param_arg == _T("thscd2") || param_arg == _T("pel")
                    || param_arg == _T("levels") || param_arg == _T("overlap") || param_arg == _T("delta")
                    || param_arg == _T("lambda") || param_arg == _T("lsad")
                    || param_arg == _T("pnew") || param_arg == _T("plevel")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    auto set_degrain_value = [&](VppDegrain& target) {
                        if (param_arg == _T("blksize")) target.blksize = value;
                        if (param_arg == _T("search")) target.search = value;
                        if (param_arg == _T("thsad")) target.thsad = value;
                        if (param_arg == _T("thscd1")) target.thscd1 = value;
                        if (param_arg == _T("thscd2")) target.thscd2 = value;
                        if (param_arg == _T("pel")) target.pel = value;
                        if (param_arg == _T("levels")) target.levels = value;
                        if (param_arg == _T("overlap")) target.overlap = value;
                        if (param_arg == _T("delta")) target.delta = value;
                        if (param_arg == _T("lambda")) target.lambda = value;
                        if (param_arg == _T("lsad")) target.lsad = value;
                        if (param_arg == _T("pnew")) target.pnew = value;
                        if (param_arg == _T("plevel")) target.plevel = value;
                    };
                    set_degrain_value(parsedRtgmc.analyze);
                    set_degrain_value(parsedRtgmc.tr1);
                    set_degrain_value(parsedRtgmc.tr2);
                    if (param_arg == _T("blksize")) presetOverride.blksize = true;
                    if (param_arg == _T("search")) presetOverride.search = true;
                    if (param_arg == _T("pel")) presetOverride.pel = true;
                    if (param_arg == _T("overlap")) presetOverride.overlap = true;
                    if (param_arg == _T("lambda")) presetOverride.lambda = true;
                    if (param_arg == _T("delta")) {
                        presetOverride.analyzeDelta = true;
                        presetOverride.tr1Delta = true;
                        presetOverride.tr2Delta = true;
                        userSetTr2 = true;
                    }
                    continue;
                }
                if (param_arg == _T("useflag")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    if (value < 0 || value > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("useFlag should be 0 - 2."));
                        return 1;
                    }
                    parsedRtgmc.tr1.useFlag = value;
                    parsedRtgmc.tr2.useFlag = value;
                    continue;
                }
                if (param_arg == _T("globalmotion")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        auto set_degrain_flag = [&](VppDegrain& target) {
                            target.globalMotion = b;
                        };
                        set_degrain_flag(parsedRtgmc.analyze);
                        set_degrain_flag(parsedRtgmc.tr1);
                        set_degrain_flag(parsedRtgmc.tr2);
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("delta_analyze")
                    || param_arg == _T("delta_tr1")
                    || param_arg == _T("delta_tr2")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) return 1;
                    if (param_arg == _T("delta_analyze")) {
                        parsedRtgmc.analyze.delta = value;
                        presetOverride.analyzeDelta = true;
                    } else if (param_arg == _T("delta_tr1")) {
                        parsedRtgmc.tr1.delta = value;
                        presetOverride.tr1Delta = true;
                    } else {
                        parsedRtgmc.tr2.delta = value;
                        userSetTr2 = true;
                        presetOverride.tr2Delta = true;
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
        apply_rtgmc_preset(parsedRtgmc.preset, parsedRtgmc.tuning, presetOverride);
        if (rtgmcMatchPresetNum >= 0 && rtgmcMatchPresetNum < rtgmcPresetNum) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" match_preset="),
                get_cx_desc(list_vpp_rtgmc_preset, rtgmcMatchPresetNum),
                _T("MatchPreset cannot use a slower setting than Preset."));
            return 1;
        }
        if (parsedRtgmc.sourceMatch > 0) {
            if (!userSetMatchEdi) {
                parsedRtgmc.matchEdi.mode = parsedRtgmc.edi.mode;
            }
            if (!userSetTr2) {
                parsedRtgmc.tr2.delta = std::max(parsedRtgmc.tr2.delta, 1);
            }
            if (!userSetSharpness) {
                parsedRtgmc.retouch.sharpness = 0.2f;
            }
            if (!userSetSlmode) {
                parsedRtgmc.retouch.slmode = 0;
            }
            parsedRtgmc.analyze.delta = std::max(parsedRtgmc.analyze.delta, std::max(parsedRtgmc.matchTR1, parsedRtgmc.matchTR2));
        }
        parsedRtgmc.matchEdi.nnsize = parsedRtgmc.edi.nnsize;
        parsedRtgmc.matchEdi.nneurons = parsedRtgmc.edi.nneurons;
        parsedRtgmc.matchEdi.ediqual = parsedRtgmc.edi.ediqual;
        parsedRtgmc.retouch.tr1 = parsedRtgmc.tr1.delta;
        parsedRtgmc.retouch.tr2 = parsedRtgmc.tr2.delta;
        if ((parsedRtgmc.retouch.smode == 0 || parsedRtgmc.retouch.sharpness == 0.0f || parsedRtgmc.retouch.slrad <= 0)
            && parsedRtgmc.retouch.slmode < 3) {
            parsedRtgmc.retouch.slmode = 0;
        }
        if (!userSetNoiseProcess) {
            if (parsedRtgmc.noise.ezKeepGrain > 0.0f) {
                parsedRtgmc.noise.noiseProcess = 2;
            } else if (parsedRtgmc.noise.noiseProcess != 2
                && (parsedRtgmc.noise.ezDenoise > 0.0f
                || parsedRtgmc.noise.grainRestore > 0.0f || parsedRtgmc.noise.noiseRestore > 0.0f)) {
                parsedRtgmc.noise.noiseProcess = 1;
            }
        }
        if (!userSetNoiseSigma && parsedRtgmc.noise.ezDenoise > 0.0f) {
            parsedRtgmc.noise.sigma = parsedRtgmc.noise.ezDenoise;
        }
        if (parsedRtgmc.noise.noiseProcess == 0) {
            if (!presetOverride.noiseTR) {
                parsedRtgmc.noise.noiseTR = 0;
            }
            if (!presetOverride.grainRestore) {
                parsedRtgmc.noise.grainRestore = 0.0f;
            }
            if (!presetOverride.noiseRestore) {
                parsedRtgmc.noise.noiseRestore = 0.0f;
            }
        }
        for (auto *stagePrm : { &parsedRtgmc.analyze, &parsedRtgmc.tr1, &parsedRtgmc.tr2 }) {
            stagePrm->chroma = parsedRtgmc.searchPrefilter.chromaMotion;
        }
        vpp->rtgmc = parsedRtgmc;
        return 0;
    }

    if (IS_OPTION("vpp-kfm") && ENABLE_VPP_FILTER_KFM) {
        vpp->kfm.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "enable", "mode", "preset", "timing", "past_cycles",
            "thswitch", "ucf", "nr", "is120", "rff", "debug", "debug_stage", "timecode", "search_early_sad", "spatial_early_sad"
        };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != tstring::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    const auto value = get_value_from_chr(list_vpp_kfm_mode, param_val.c_str());
                    if (value == PARSE_ERROR_FLAG) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_kfm_mode);
                        return 1;
                    }
                    vpp->kfm.mode = (VppKfmMode)value;
                    continue;
                }
                if (param_arg == _T("preset")) {
                    const auto value = get_value_from_chr(list_vpp_rtgmc_preset, param_val.c_str());
                    if (value == PARSE_ERROR_FLAG) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_preset);
                        return 1;
                    }
                    vpp->kfm.preset = (VppRtgmcPreset)value;
                    continue;
                }
                if (param_arg == _T("search_early_sad")) {
                    if (parse_degrain_search_early_sad(&vpp->kfm.searchEarlySadOverride, param_val, true)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("search_early_sad should be auto, off, or -1 - 65535."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("spatial_early_sad")) {
                    if (parse_degrain_search_early_sad(&vpp->kfm.spatialEarlySadOverride, param_val, true)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("spatial_early_sad should be auto, off, or -1 - 65535."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("timing")) {
                    const auto value = get_value_from_chr(list_vpp_kfm_timing, param_val.c_str());
                    if (value == PARSE_ERROR_FLAG) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_kfm_timing);
                        return 1;
                    }
                    vpp->kfm.timing = (VppKfmTiming)value;
                    continue;
                }
                if (param_arg == _T("past_cycles")) {
                    try {
                        vpp->kfm.pastCycles = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->kfm.pastCycles < 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thswitch")) {
                    try {
                        vpp->kfm.thswitch = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ucf")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.ucf = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nr")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.nr = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("is120")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.is120 = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rff")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.rff = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("debug")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->kfm.debug = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("debug_stage")) {
                    const auto value = get_value_from_chr(list_vpp_kfm_debug_stage, param_val.c_str());
                    if (value == PARSE_ERROR_FLAG) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_kfm_debug_stage);
                        return 1;
                    }
                    vpp->kfm.debugStage = (VppKfmDebugStage)value;
                    continue;
                }
                if (param_arg == _T("timecode")) {
                    vpp->kfm.timecode = param_val;
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

    if (IS_OPTION("vpp-rtgmc-bob") && ENABLE_VPP_FILTER_RTGMC_BOB) {
        vpp->rtgmc_bob.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "order" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_bob.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("order")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_bob_order, param_val.c_str(), &value)) {
                        vpp->rtgmc_bob.order = (VppRtgmcBobOrder)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_bob_order);
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

    if (IS_OPTION("vpp-rtgmc-search-prefilter") && ENABLE_VPP_FILTER_RTGMC_SEARCH_PREFILTER) {
        vpp->rtgmc_search_prefilter.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "enable", "tr0", "rep0-thin", "rep0-pad",
            "search_refine", "tv_range", "chroma_motion",
            "dump_y4m", "dump_stage", "dump_max_frames"
        };
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_search_prefilter.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr0")) {
                    if (parse_int(&vpp->rtgmc_search_prefilter.tr0, param_arg, param_val)) {
                        return 1;
                    }
                    if (vpp->rtgmc_search_prefilter.tr0 < -1 || vpp->rtgmc_search_prefilter.tr0 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr0 should be -1, 0, 1, or 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep0-thin")) {
                    if (parse_int(&vpp->rtgmc_search_prefilter.rep0Thin, param_arg, param_val)) {
                        return 1;
                    }
                    if (!is_valid_rtgmc_rep_thin(vpp->rtgmc_search_prefilter.rep0Thin)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep0-thin should be 0 - 7."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep0-pad")) {
                    if (parse_int(&vpp->rtgmc_search_prefilter.rep0Pad, param_arg, param_val)) {
                        return 1;
                    }
                    if (!is_valid_rtgmc_rep_pad(vpp->rtgmc_search_prefilter.rep0Pad)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep0-pad should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_refine")) {
                    if (parse_int(&vpp->rtgmc_search_prefilter.searchRefine, param_arg, param_val)) {
                        return 1;
                    }
                    if (vpp->rtgmc_search_prefilter.searchRefine < 0 || vpp->rtgmc_search_prefilter.searchRefine > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("search_refine should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tv_range")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_search_prefilter.tvRange = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma_motion")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_search_prefilter.chromaMotion = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dump_y4m")) {
                    vpp->rtgmc_search_prefilter.dumpY4m = param_val;
                    continue;
                }
                if (param_arg == _T("dump_stage")) {
                    const auto stage = check_str_list(option_name, param_arg, param_val,
                        { _T("final"), _T("search_refine"), _T("finalyuv"), _T("temporal_candidate"), _T("deshimmered_search"),
                          _T("search_correction_delta"), _T("positive_correction_gate"), _T("negative_correction_gate"),
                          _T("corrected_search_base"), _T("half_search_base"), _T("half_search_smoothed"),
                          _T("search_smoothed3x3"), _T("edge_softened_search"), _T("softened_search_blend"),
                          _T("pre_stabilized_search"), _T("stabilized_search") });
                    if (!stage) return 1;
                    vpp->rtgmc_search_prefilter.dumpStage = *stage;
                    continue;
                }
                if (param_arg == _T("dump_max_frames")) {
                    if (parse_int(&vpp->rtgmc_search_prefilter.dumpMaxFrames, param_arg, param_val)) {
                        return 1;
                    }
                    if (vpp->rtgmc_search_prefilter.dumpMaxFrames < 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("dump_max_frames should be >= 0."));
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

    if (IS_OPTION("vpp-rtgmc-edi") && ENABLE_VPP_FILTER_RTGMC_EDI) {
        vpp->rtgmc_edi.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "mode", "nnsize", "nneurons", "ediqual", "chroma_edi" };
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_edi.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_edi_mode, param_val.c_str(), &value)) {
                        vpp->rtgmc_edi.mode = (VppRtgmcEdiMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_edi_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nnsize")) {
                    if (parse_int(&vpp->rtgmc_edi.nnsize, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_edi.nnsize < 0 || vpp->rtgmc_edi.nnsize > 6) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("nnsize should be 0 - 6."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("nneurons")) {
                    if (parse_int(&vpp->rtgmc_edi.nneurons, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_edi.nneurons < 0 || vpp->rtgmc_edi.nneurons > 4) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("nneurons should be 0 - 4."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ediqual")) {
                    if (parse_int(&vpp->rtgmc_edi.ediqual, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_edi.ediqual < 1 || vpp->rtgmc_edi.ediqual > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("ediqual should be 1 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma_edi")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_chroma_edi_mode, param_val.c_str(), &value)) {
                        vpp->rtgmc_edi.chromaEdi = (VppRtgmcChromaEdiMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_chroma_edi_mode);
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

    if (IS_OPTION("vpp-rtgmc-retouch") && ENABLE_VPP_FILTER_RTGMC_RETOUCH) {
        vpp->rtgmc_retouch.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "sharpness", "limit", "smode", "slmode", "slrad", "sovs", "svthin", "sbb", "precise", "tr1", "tr2" };
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_retouch.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sharpness")) {
                    try {
                        vpp->rtgmc_retouch.sharpness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->rtgmc_retouch.sharpness < 0.0f || vpp->rtgmc_retouch.sharpness > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sharpness should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("limit")) {
                    try {
                        vpp->rtgmc_retouch.limit = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->rtgmc_retouch.limit < 0.0f || vpp->rtgmc_retouch.limit > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("limit should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smode")) {
                    if (parse_int(&vpp->rtgmc_retouch.smode, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.smode < 0 || vpp->rtgmc_retouch.smode > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("smode should be 0 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("slmode")) {
                    if (parse_int(&vpp->rtgmc_retouch.slmode, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.slmode < 0 || vpp->rtgmc_retouch.slmode > 4) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("slmode should be 0 - 4."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("slrad")) {
                    if (parse_int(&vpp->rtgmc_retouch.slrad, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.slrad < 0 || vpp->rtgmc_retouch.slrad > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("slrad should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sovs")) {
                    if (parse_int(&vpp->rtgmc_retouch.sovs, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.sovs < 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sovs should be 0 or greater."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("svthin")) {
                    try {
                        vpp->rtgmc_retouch.svthin = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->rtgmc_retouch.svthin < 0.0f || vpp->rtgmc_retouch.svthin > 1.0f) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("svthin should be 0.0 - 1.0."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sbb")) {
                    if (parse_int(&vpp->rtgmc_retouch.sbb, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.sbb < 0 || vpp->rtgmc_retouch.sbb > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("sbb should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("precise")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->rtgmc_retouch.precise = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr1")) {
                    if (parse_int(&vpp->rtgmc_retouch.tr1, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.tr1 < 0 || vpp->rtgmc_retouch.tr1 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr1 should be 0 - 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr2")) {
                    if (parse_int(&vpp->rtgmc_retouch.tr2, param_arg, param_val)) return 1;
                    if (vpp->rtgmc_retouch.tr2 < 0 || vpp->rtgmc_retouch.tr2 > 5) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr2 should be 0 - 5."));
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

    if (IS_OPTION("vpp-rtgmc-shimmer-repair") && ENABLE_VPP_FILTER_RTGMC_SHIMMER_REPAIR) {
        VppRtgmcShimmerRepair parsedShimmerRepair;
        parsedShimmerRepair.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            vpp->rtgmc_shimmer_repair = parsedShimmerRepair;
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "enable", "stage", "mode", "rep-thin", "rep-pad", "rep_chroma"
        };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedShimmerRepair.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("stage")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_shimmer_repair_stage, param_val.c_str(), &value)) {
                        parsedShimmerRepair.stage = (VppRtgmcShimmerRepairStage)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_shimmer_repair_stage);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    // Step2 では passthrough のみ対応
                    if (!check_str_list(option_name, param_arg, param_val, { _T("passthrough") })) return 1;
                    continue;
                }
                if (param_arg == _T("rep-thin")) {
                    try {
                        size_t idx = 0;
                        parsedShimmerRepair.repThin = std::stoi(param_val, &idx);
                        if (idx != param_val.length()) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (!is_valid_rtgmc_rep_thin(parsedShimmerRepair.repThin)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep-thin should be 0 - 7."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep-pad")) {
                    try {
                        size_t idx = 0;
                        parsedShimmerRepair.repPad = std::stoi(param_val, &idx);
                        if (idx != param_val.length()) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (!is_valid_rtgmc_rep_pad(parsedShimmerRepair.repPad)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep-pad should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep_chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedShimmerRepair.repChroma = b;
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
        if (parsedShimmerRepair.stage == VppRtgmcShimmerRepairStage::Rep1) {
            vpp->rtgmc_shimmer_repairRep1 = parsedShimmerRepair;
        } else {
            vpp->rtgmc_shimmer_repairRep2 = parsedShimmerRepair;
        }
        return 0;
    }

    if (IS_OPTION("vpp-rtgmc-primitive") && ENABLE_VPP_FILTER_RTGMC_PRIMITIVE) {
        VppRtgmcPrimitive parsedPrimitive;
        parsedPrimitive.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            vpp->rtgmc_primitive = parsedPrimitive;
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "op", "ref", "mode", "weight", "chroma" };
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        const auto parse_float = [&](float *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stof(param_val, &idx);
                if (idx != param_val.length() || !std::isfinite(*dst)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedPrimitive.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("op")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_primitive_op, param_val.c_str(), &value)) {
                        parsedPrimitive.op = (VppRtgmcPrimitiveOp)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_primitive_op);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ref")) {
                    int value = 0;
                    if (get_list_value(list_vpp_rtgmc_primitive_ref, param_val.c_str(), &value)) {
                        parsedPrimitive.ref = (VppRtgmcPrimitiveRef)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_rtgmc_primitive_ref);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    if (parse_int(&parsedPrimitive.mode, param_arg, param_val)) return 1;
                    continue;
                }
                if (param_arg == _T("weight")) {
                    if (parse_float(&parsedPrimitive.weight, param_arg, param_val)) return 1;
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedPrimitive.chroma = b;
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
        if (parsedPrimitive.op == VppRtgmcPrimitiveOp::Merge
            && (parsedPrimitive.weight < 0.0f || parsedPrimitive.weight > 1.0f)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" weight="), strsprintf(_T("%.3f"), parsedPrimitive.weight),
                _T("merge weight should be 0.0 - 1.0."));
            return 1;
        }
        if (parsedPrimitive.op == VppRtgmcPrimitiveOp::AddWeightedDiff
            && (parsedPrimitive.weight < -1.0f || parsedPrimitive.weight > 1.0f)) {
            print_cmd_error_invalid_value(tstring(option_name) + _T(" weight="), strsprintf(_T("%.3f"), parsedPrimitive.weight),
                _T("addweighteddiff weight should be -1.0 - 1.0."));
            return 1;
        }
        vpp->rtgmc_primitive = parsedPrimitive;
        return 0;
    }


    if (IS_OPTION("vpp-decomb") && ENABLE_VPP_FILTER_DECOMB) {
        vpp->decomb.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "full", "threshold", "dthreshold", "blend" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decomb.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("full")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decomb.full = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->decomb.threshold = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dthreshold")) {
                    try {
                        vpp->decomb.dthreshold = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blend")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->decomb.blend = b;
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
        const auto paramList = std::vector<std::string>{ "lo", "hi", "max", "keep", "frac", "log" };

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
                if (param_arg == _T("keep")) {
                    try {
                        vpp->mpdecimate.keep = std::stoi(param_val);
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
                    vpp->mpdecimate.log = true;
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
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_convolution3d_matrix);
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

        const auto paramList = std::vector<std::string>{ "radius", "d", "strength", "lerp", "th_weight", "th_lerp" };

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
                    if (param_arg == _T("d")) {
                        try {
                            vpp->knn.d = std::stoi(param_val);
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
    if (IS_OPTION("vpp-nlmeans")) {
        vpp->nlmeans.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "sigma", "patch", "search", "h", "d", "search_t", "fp16", "shared_mem" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->nlmeans.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("patch")) {
                    try {
                        vpp->nlmeans.patchSize = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search")) {
                    try {
                        vpp->nlmeans.searchSize = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    try {
                        vpp->nlmeans.sigma = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("h")) {
                    try {
                        vpp->nlmeans.h = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("d")) {
                    try {
                        vpp->nlmeans.d = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_t")) {
                    try {
                        vpp->nlmeans.searchSizeT = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("fp16")) {
                    int value = 0;
                    if (get_list_value(list_vpp_nlmeans_fp16, param_val.c_str(), &value)) {
                        vpp->nlmeans.fp16 = (VppNLMeansFP16Opt)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_nlmeans_fp16);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("shared_mem")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->nlmeans.sharedMem = b;
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
    if (IS_OPTION("vpp-bm3d") && ENABLE_VPP_FILTER_DENOISE_BM3D) {
        vpp->bm3d.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "profile", "sigma", "block_step", "group_size", "bm_range", "radius", "chroma" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->bm3d.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("profile")) {
                    // Presets for block_step / group_size / bm_range. Anything
                    // given after profile= overrides the preset.
                    static const CX_DESC bm3d_profile_list[] = {
                        { _T("fast"), 0 }, { _T("lc"), 1 }, { _T("np"), 2 }, { _T("high"), 3 }, { NULL, 0 }
                    };
                    int p = 0;
                    if (!get_list_value(bm3d_profile_list, param_val.c_str(), &p)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, bm3d_profile_list);
                        return 1;
                    }
                    switch (p) {
                    case 1: vpp->bm3d.block_step = 6; vpp->bm3d.group_size = 8;  vpp->bm3d.bm_range = 9;  break;
                    case 2: vpp->bm3d.block_step = 4; vpp->bm3d.group_size = 16; vpp->bm3d.bm_range = 16; break;
                    case 3: vpp->bm3d.block_step = 3; vpp->bm3d.group_size = 16; vpp->bm3d.bm_range = 16; break;
                    default: vpp->bm3d.block_step = 8; vpp->bm3d.group_size = 8; vpp->bm3d.bm_range = 9;  break;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    try {
                        size_t parsed = 0;
                        const float value = std::stof(param_val, &parsed);
                        if (parsed != param_val.length() || !std::isfinite(value)) {
                            throw std::invalid_argument("invalid float");
                        }
                        vpp->bm3d.sigma = value;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->bm3d.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                int *ivalue = nullptr;
                if      (param_arg == _T("block_step")) ivalue = &vpp->bm3d.block_step;
                else if (param_arg == _T("group_size")) ivalue = &vpp->bm3d.group_size;
                else if (param_arg == _T("bm_range"))   ivalue = &vpp->bm3d.bm_range;
                else if (param_arg == _T("radius"))     ivalue = &vpp->bm3d.radius;
                if (ivalue != nullptr) {
                    try {
                        size_t parsed = 0;
                        const int value = std::stoi(param_val, &parsed);
                        if (parsed != param_val.length()) {
                            throw std::invalid_argument("trailing characters");
                        }
                        *ivalue = value;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
            print_cmd_error_unknown_opt_param(option_name, param, paramList);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-hqdn3d") && ENABLE_VPP_FILTER_HQDN3D) {
        vpp->hqdn3d.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "luma_spatial", "chroma_spatial", "luma_temporal", "chroma_temporal" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->hqdn3d.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("luma_spatial")) {
                    try {
                        vpp->hqdn3d.luma_spatial = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma_spatial")) {
                    try {
                        vpp->hqdn3d.chroma_spatial = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("luma_temporal")) {
                    try {
                        vpp->hqdn3d.luma_temporal = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma_temporal")) {
                    try {
                        vpp->hqdn3d.chroma_temporal = std::stof(param_val);
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
    if (IS_OPTION("vpp-descale") && ENABLE_VPP_FILTER_DESCALE) {
        vpp->descale.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "enable", "kernel", "width", "height", "b", "c", "src_left", "src_top", "src_width", "src_height",
            "border_handling", "border", "auto", "search_min", "search_max", "search_step",
            "detect_frames", "show_scores"
        };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos == std::string::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            auto param_arg = tolowercase(param.substr(0, pos));
            auto param_val = param.substr(pos + 1);
            if (param_arg == _T("enable")) {
                bool b = false;
                if (!cmd_string_to_bool(&b, param_val)) {
                    vpp->descale.enable = b;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("kernel")) {
                int value = 0;
                if (get_list_value(list_vpp_descale_kernel, param_val.c_str(), &value)) {
                    vpp->descale.kernel = (VppDescaleKernel)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_descale_kernel);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("border_handling") || param_arg == _T("border")) {
                int value = 0;
                if (get_list_value(list_vpp_descale_border, param_val.c_str(), &value)) {
                    vpp->descale.border = (VppDescaleBorder)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_descale_border);
                    return 1;
                }
                continue;
            }
            int *int_target = nullptr;
            float *float_target = nullptr;
            bool *bool_target = nullptr;
            if      (param_arg == _T("width"))         int_target = &vpp->descale.width;
            else if (param_arg == _T("height"))        int_target = &vpp->descale.height;
            else if (param_arg == _T("search_min"))    int_target = &vpp->descale.search_min;
            else if (param_arg == _T("search_max"))    int_target = &vpp->descale.search_max;
            else if (param_arg == _T("search_step"))   int_target = &vpp->descale.search_step;
            else if (param_arg == _T("detect_frames")) int_target = &vpp->descale.detect_frames;
            else if (param_arg == _T("b"))             float_target = &vpp->descale.b;
            else if (param_arg == _T("c"))             float_target = &vpp->descale.c;
            else if (param_arg == _T("src_left"))      float_target = &vpp->descale.src_left;
            else if (param_arg == _T("src_top"))       float_target = &vpp->descale.src_top;
            else if (param_arg == _T("src_width"))     float_target = &vpp->descale.src_width;
            else if (param_arg == _T("src_height"))    float_target = &vpp->descale.src_height;
            else if (param_arg == _T("auto"))          bool_target = &vpp->descale.autoDetect;
            else if (param_arg == _T("show_scores"))   bool_target = &vpp->descale.show_scores;
            if (int_target) {
                try { *int_target = std::stoi(param_val); }
                catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                continue;
            }
            if (float_target) {
                try { *float_target = std::stof(param_val); }
                catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                continue;
            }
            if (bool_target) {
                bool b = false;
                if (!cmd_string_to_bool(&b, param_val)) {
                    *bool_target = b;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
            return 1;
        }
        if (vpp->descale.autoDetect) {
            vpp->descale.kernel = VppDescaleKernel::Auto;
        }
        return 0;
    }
    if (IS_OPTION("vpp-anime4k-shader") && ENABLE_VPP_FILTER_ANIME4K) {
        vpp->anime4k.enable = true;
        // Whether strength= was given explicitly. The deblur default below is
        // derived after the whole option string is parsed, so it cannot depend
        // on the order the parameters happen to be written in.
        bool anime4kStrengthGiven = false;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "enable", "mode", "scale", "strength", "chroma_resize", "chroma",
            "darken", "thin",
            "denoise", "denoise_intensity", "denoise_spatial", "denoise_curve", "denoise_hist_reg",
            "prefilter_denoise", "clamp_highlights", "antiring",
            "out_res", "resize" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos == std::string::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            auto param_arg = param.substr(0, pos);
            auto param_val = param.substr(pos + 1);
            param_arg = tolowercase(param_arg);
            if (param_arg == _T("enable")) {
                bool b = false;
                if (!cmd_string_to_bool(&b, param_val)) {
                    vpp->anime4k.enable = b;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("mode")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_mode, param_val.c_str(), &value)) {
                    vpp->anime4k.mode = (VppAnime4kMode)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_mode);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("scale")) {
                try {
                    vpp->anime4k.scale = std::stoi(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("strength")) {
                try {
                    vpp->anime4k.strength = std::stof(param_val);
                    anime4kStrengthGiven = true;
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("chroma_resize")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_chroma_resize, param_val.c_str(), &value)) {
                    vpp->anime4k.chromaResize = (VppAnime4kChromaResize)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_chroma_resize);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("chroma")) {
                bool b = false;
                if (!cmd_string_to_bool(&b, param_val)) {
                    vpp->anime4k.chroma = b;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("out_res")) {
                auto xpos = param_val.find_first_of(_T("xX"));
                int w = 0, h = 0;
                bool ok = false;
                if (xpos != tstring::npos) {
                    try {
                        w = std::stoi(param_val.substr(0, xpos));
                        h = std::stoi(param_val.substr(xpos + 1));
                        ok = true;
                    } catch (...) {
                        ok = false;
                    }
                }
                // A negative value on ONE axis keeps the source aspect (magnitude =
                // rounding step), matching --output-res (e.g. -2x1080 -> auto-even width).
                // Both negative, or any zero, is invalid.
                if (!ok || w == 0 || h == 0 || (w < 0 && h < 0)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                        _T("expected WxH; a negative value keeps aspect (e.g. -2x1080); both cannot be negative"));
                    return 1;
                }
                vpp->anime4k.postResizeW = w;
                vpp->anime4k.postResizeH = h;
                continue;
            }
            if (param_arg == _T("resize")) {
                int value = 0;
                if (get_list_value(list_vpp_resize, param_val.c_str(), &value)) {
                    vpp->anime4k.postResizeAlgo = (RGY_VPP_RESIZE_ALGO)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("darken")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_darken, param_val.c_str(), &value)) {
                    vpp->anime4k.darken = (VppAnime4kDarken)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_darken);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("thin")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_thin, param_val.c_str(), &value)) {
                    vpp->anime4k.thin = (VppAnime4kThin)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_thin);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("denoise")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_denoise, param_val.c_str(), &value)) {
                    vpp->anime4k.denoise = (VppAnime4kDenoise)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_denoise);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("denoise_intensity")) {
                try {
                    vpp->anime4k.denoiseIntensity = std::stof(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("denoise_spatial")) {
                try {
                    vpp->anime4k.denoiseSpatial = std::stof(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("denoise_curve")) {
                try {
                    vpp->anime4k.denoiseCurve = std::stof(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("denoise_hist_reg")) {
                try {
                    vpp->anime4k.denoiseHistReg = std::stof(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("prefilter_denoise")) {
                int value = 0;
                if (get_list_value(list_vpp_anime4k_denoise, param_val.c_str(), &value)) {
                    vpp->anime4k.prefilterDenoise = (VppAnime4kDenoise)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_anime4k_denoise);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("clamp_highlights")) {
                bool b = false;
                if (!cmd_string_to_bool(&b, param_val)) {
                    vpp->anime4k.clampHighlights = b;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            if (param_arg == _T("antiring")) {
                try {
                    vpp->anime4k.antiring = std::stof(param_val);
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                continue;
            }
            print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
            return 1;
        }
        // deblur uses REFINE_STRENGTH=1.0 in the reference shader while original
        // uses 0.5. Applied here, once the final mode is known, so that
        // strength=0.5,mode=ani4k_deblur keeps the 0.5 the user asked for.
        if (vpp->anime4k.mode == VppAnime4kMode::Deblur && !anime4kStrengthGiven) {
            vpp->anime4k.strength = 1.0f;
        }
        return 0;
    }
    if (IS_OPTION("vpp-onnx") && ENABLE_VPP_FILTER_ONNX) {
        vpp->onnx.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "enable", "model", "modelfile", "provider", "device",
#if ENCODER_NVENC
            "interop", "prec",
#endif
            "precision",
            "colormatrix", "colormatrix_out", "colorrange", "colorspace", "noise", "frames", "mask", "out_res", "resize"
        };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->onnx.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("model") || param_arg == _T("modelfile")) {
                    vpp->onnx.modelFile = param_val;
                    continue;
                }
                if (param_arg == _T("provider")) {
                    const tstring v = tolowercase(param_val);
                    if (v == _T("auto") || v == _T("cuda") || v == _T("tensorrt") || v == _T("trt")) {
                        vpp->onnx.provider = (v == _T("trt")) ? _T("tensorrt") : v;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("device")) {
                    //OpenVINOのデバイス名は大文字 (GPU.0/GPU/CPU/AUTO/NPU) - 小文字入力も受け付ける
                    vpp->onnx.device = touppercase(param_val);
                    continue;
                }
#if ENCODER_NVENC
                if (param_arg == _T("interop")) {
                    const tstring v = tolowercase(param_val);
                    if (v == _T("auto") || v == _T("ocl") || v == _T("host")) {
                        vpp->onnx.interop = v;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
#endif
                if (param_arg == _T("prec") || param_arg == _T("precision")) {
                    const tstring v = tolowercase(param_val);
                    if (v == _T("auto") || v == _T("fp16") || v == _T("f16") || v == _T("fp32") || v == _T("f32")) {
                        vpp->onnx.precision = (v == _T("f16")) ? _T("fp16") : (v == _T("f32")) ? _T("fp32") : v;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colormatrix")) {
                    int value = 0;
                    if (get_list_value(list_colormatrix, param_val.c_str(), &value)) {
                        vpp->onnx.colormatrix = (CspMatrix)value;
                    } else {
                        const auto compatMatrix = tolowercase(param_val);
                        // 互換性のため、公開済みの旧指定名だけは --vpp-onnx colormatrix で吸収する。
                        if (compatMatrix == _T("bt601")) {
                            vpp->onnx.colormatrix = RGY_MATRIX_ST170_M;
                        } else if (compatMatrix == _T("bt2020")) {
                            vpp->onnx.colormatrix = RGY_MATRIX_BT2020_NCL;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_colormatrix);
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("colormatrix_out")) {
                    int value = 0;
                    if (get_list_value(list_colormatrix, param_val.c_str(), &value)) {
                        vpp->onnx.colormatrixOut = (CspMatrix)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_colormatrix);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colorrange")) {
                    int value = 0;
                    if (get_list_value(list_colorrange, param_val.c_str(), &value)) {
                        vpp->onnx.colorrange = (CspColorRange)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_colorrange);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colorspace")) {
                    const auto v = check_str_list(option_name, param_arg, param_val, { _T("rgb"), _T("ycbcr") });
                    if (!v) return 1;
                    vpp->onnx.colorspace = *v;
                    continue;
                }
                if (param_arg == _T("noise")) {
                    try {
                        vpp->onnx.noise = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("frames")) {
                    try {
                        vpp->onnx.frames = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->onnx.frames < 1 || (vpp->onnx.frames % 2) == 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("frames must be a positive odd number"));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mask")) {
                    vpp->onnx.maskFile = param_val;
                    continue;
                }
                if (param_arg == _T("out_res")) {
                    auto xpos = param_val.find_first_of(_T("xX"));
                    int w = 0, h = 0;
                    bool ok = false;
                    if (xpos != tstring::npos) {
                        try {
                            w = std::stoi(param_val.substr(0, xpos));
                            h = std::stoi(param_val.substr(xpos + 1));
                            ok = true;
                        } catch (...) {
                            ok = false;
                        }
                    }
                    if (!ok || w == 0 || h == 0 || (w < 0 && h < 0)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("expected WxH; a negative value keeps aspect (e.g. -2x1080); both cannot be negative"));
                        return 1;
                    }
                    vpp->onnx.postResizeW = w;
                    vpp->onnx.postResizeH = h;
                    continue;
                }
                if (param_arg == _T("resize")) {
                    int value = 0;
                    if (get_list_value(list_vpp_resize, param_val.c_str(), &value)) {
                        vpp->onnx.postResizeAlgo = (RGY_VPP_RESIZE_ALGO)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (tolowercase(param) == _T("list")) {
                    vpp->onnxListModels = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-onnx-model-dir") && ENABLE_VPP_FILTER_ONNX) {
        i++;
        vpp->onnxModelDir = tstring(strInput[i]);
        return 0;
    }
    if (IS_OPTION("vpp-rife-ov") && ENABLE_VPP_FILTER_RIFE_OV) {
        vpp->rife_ov.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "model", "device", "multi", "fps", "colormatrix", "colorrange" };
        bool multiSpecified = false;
        bool fpsSpecified = false;
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == tstring::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto name = tolowercase(param.substr(0, pos));
            const auto value = param.substr(pos + 1);
            if (name == _T("enable")) {
                if (cmd_string_to_bool(&vpp->rife_ov.enable, value)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value);
                    return 1;
                }
            } else if (name == _T("model")) {
                vpp->rife_ov.modelFile = value;
            } else if (name == _T("device")) {
                vpp->rife_ov.device = touppercase(value);
            } else if (name == _T("multi")) {
                multiSpecified = true;
                try {
                    vpp->rife_ov.multi = std::stoi(value);
                } catch (...) {
                    vpp->rife_ov.multi = 0;
                }
                if (vpp->rife_ov.multi < 2) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value, _T("multi must be an integer >= 2"));
                    return 1;
                }
            } else if (name == _T("fps")) {
                fpsSpecified = true;
                int fpsN = 0, fpsD = 0;
                const auto parseInt = [](const tstring& str, int& result) {
                    size_t parsed = 0;
                    try {
                        result = std::stoi(str, &parsed);
                    } catch (...) {
                        return false;
                    }
                    return parsed == str.size();
                };
                const auto slash = value.find(_T('/'));
                if (slash != tstring::npos) {
                    if (slash == 0 || slash + 1 >= value.size() || value.find(_T('/'), slash + 1) != tstring::npos
                        || !parseInt(value.substr(0, slash), fpsN) || !parseInt(value.substr(slash + 1), fpsD)) {
                        fpsN = 0;
                        fpsD = 0;
                    }
                } else {
                    double fpsValue = 0.0;
                    size_t parsed = 0;
                    try {
                        fpsValue = std::stod(value, &parsed);
                    } catch (...) {
                        fpsValue = 0.0;
                    }
                    if (parsed == value.size() && std::isfinite(fpsValue) && fpsValue > 0.0
                        && fpsValue <= (double)std::numeric_limits<int>::max() / 1000.0) {
                        if (std::abs(fpsValue - 24000.0 / 1001.0) < 0.01) { fpsN = 24000; fpsD = 1001; }
                        else if (std::abs(fpsValue - 30000.0 / 1001.0) < 0.01) { fpsN = 30000; fpsD = 1001; }
                        else if (std::abs(fpsValue - 60000.0 / 1001.0) < 0.01) { fpsN = 60000; fpsD = 1001; }
                        else if (std::abs(fpsValue - 120000.0 / 1001.0) < 0.01) { fpsN = 120000; fpsD = 1001; }
                        else { fpsN = (int)(fpsValue * 1000.0 + 0.5); fpsD = 1000; }
                    }
                }
                if (fpsN <= 0 || fpsD <= 0) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value,
                        _T("fps must be a positive rate, given as a ratio or a decimal"));
                    return 1;
                }
                vpp->rife_ov.fps = rgy_rational<int>(fpsN, fpsD);
            } else if (name == _T("colormatrix")) {
                const auto normalized = tolowercase(value);
                if (normalized != _T("auto") && normalized != _T("bt601") && normalized != _T("bt709") && normalized != _T("bt2020")) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value);
                    return 1;
                }
                vpp->rife_ov.colormatrix = normalized;
            } else if (name == _T("colorrange")) {
                const auto normalized = tolowercase(value);
                if (normalized != _T("auto") && normalized != _T("tv") && normalized != _T("limited") && normalized != _T("pc") && normalized != _T("full")) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value);
                    return 1;
                }
                vpp->rife_ov.colorrange = (normalized == _T("limited")) ? _T("tv") : (normalized == _T("full")) ? _T("pc") : normalized;
            } else {
                print_cmd_error_unknown_opt_param(option_name, name, paramList);
                return 1;
            }
        }
        if (multiSpecified && fpsSpecified) {
            _ftprintf(stderr, _T("Warning: --vpp-rife-ov fps= overrides multi=.\n"));
        }
        return 0;
    }
    if ((IS_OPTION("vpp-onnx-deint") || IS_OPTION("vpp-vhs-restore")) && ENABLE_VPP_FILTER_ONNX_DEINT) {
        const auto operation = IS_OPTION("vpp-vhs-restore") ? VppOnnxDeintOperation::VhsRestore : VppOnnxDeintOperation::Deinterlace;
        // enable=falseの指定も含めて併記を拒否し、後の指定で他方の設定を流用しない。
        if (vpp->onnxDeint.operation != VppOnnxDeintOperation::Unspecified && vpp->onnxDeint.operation != operation) {
            _ftprintf(stderr, _T("--vpp-onnx-deintと--vpp-vhs-restoreは同時に指定できません。\n"));
            return 1;
        }
        vpp->onnxDeint.operation = operation;
        vpp->onnxDeint.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "model", "device", "precision", "mode", "colormatrix", "colorrange" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == tstring::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto name = tolowercase(param.substr(0, pos));
            const auto value = param.substr(pos + 1);
            if (name == _T("enable")) {
                if (cmd_string_to_bool(&vpp->onnxDeint.enable, value)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value);
                    return 1;
                }
            } else if (name == _T("model")) {
                vpp->onnxDeint.modelFile = value;
            } else if (name == _T("device")) {
                vpp->onnxDeint.device = touppercase(value);
            } else if (name == _T("precision")) {
                const auto normalized = tolowercase(value);
                if (normalized != _T("fp32") && normalized != _T("auto")) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value);
                    return 1;
                }
                vpp->onnxDeint.precision = normalized;
            } else if (name == _T("mode")) {
                int mode = 0;
                if (!get_list_value(list_vpp_onnx_deint_mode, value.c_str(), &mode)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value, list_vpp_onnx_deint_mode);
                    return 1;
                }
                vpp->onnxDeint.mode = (VppOnnxDeintMode)mode;
            } else if (name == _T("colormatrix")) {
                int matrix = 0;
                if (!get_list_value(list_colormatrix, value.c_str(), &matrix)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value, list_colormatrix);
                    return 1;
                }
                vpp->onnxDeint.colormatrix = (CspMatrix)matrix;
            } else if (name == _T("colorrange")) {
                int range = 0;
                if (!get_list_value(list_colorrange, value.c_str(), &range)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + name + _T("="), value, list_colorrange);
                    return 1;
                }
                vpp->onnxDeint.colorrange = (CspColorRange)range;
            } else {
                print_cmd_error_unknown_opt_param(option_name, name, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-onnx-cache-dir")
        && (ENABLE_VPP_FILTER_ONNX || ENABLE_VPP_FILTER_ONNX_DEINT)
        && (ENABLE_OPENVINO || ENCODER_NVENC)) {
        i++;
        // autoは既定のキャッシュ先(ユーザーごとのフォルダ)を使う。helpの既定値と同じ指定を受け付ける
        vpp->onnx.cacheDir = (_tcsicmp(strInput[i], _T("auto")) == 0) ? tstring() : tstring(strInput[i]);
        return 0;
    }
    if (IS_OPTION("vpp-denoise-dct") && ENABLE_VPP_FILTER_DENOISE_DCT) {
        vpp->dct.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "sigma", "sigma2", "sigma3", "sigma4", "step", "block_size" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->dct.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->dct.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    try {
                        vpp->dct.sigma = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma2")) {
                    try {
                        vpp->dct.sigma2 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma3")) {
                    try {
                        vpp->dct.sigma3 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma4")) {
                    try {
                        vpp->dct.sigma4 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("step")) {
                    int value = 0;
                    if (get_list_value(list_vpp_denoise_dct_step, param_val.c_str(), &value)) {
                        vpp->dct.step = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_denoise_dct_step);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("block_size")) {
                    int value = 0;
                    if (get_list_value(list_vpp_denoise_dct_block_size, param_val.c_str(), &value)) {
                        vpp->dct.block_size = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_denoise_dct_block_size);
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
    if (IS_OPTION("vpp-fft3d") && ENABLE_VPP_FILTER_FFT3D) {
        vpp->fft3d.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "sigma", "sigma2", "sigma3", "sigma4", "amount", "block_size", "overlap",/*"overlap2",*/ "method", "temporal",
            "bt", "sharpen", "scutoff", "svr", "smin", "smax", "degrid", "signorm", "prec"};
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        vpp->fft3d.enable = true;
                    } else if (param_val == _T("false")) {
                        vpp->fft3d.enable = false;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    try {
                        vpp->fft3d.sigma = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma2")) {
                    try {
                        vpp->fft3d.sigma2 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma3")) {
                    try {
                        vpp->fft3d.sigma3 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma4")) {
                    try {
                        vpp->fft3d.sigma4 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("amount")) {
                    try {
                        vpp->fft3d.amount = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("block_size")) {
                    int value = 0;
                    if (get_list_value(list_vpp_fft3d_block_size, param_val.c_str(), &value)) {
                        vpp->fft3d.block_size = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_fft3d_block_size);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("overlap")) {
                    try {
                        vpp->fft3d.overlap = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("overlap2")) {
                    try {
                        vpp->fft3d.overlap2 = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("method")) {
                    try {
                        vpp->fft3d.method = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("temporal")) {
                    try {
                        vpp->fft3d.temporal = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("bt")) {
                    try {
                        vpp->fft3d.bt = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sharpen")) {
                    try {
                        vpp->fft3d.sharpen = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scutoff")) {
                    try {
                        vpp->fft3d.scutoff = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("svr")) {
                    try {
                        vpp->fft3d.svr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smin")) {
                    try {
                        vpp->fft3d.smin = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("smax")) {
                    try {
                        vpp->fft3d.smax = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("degrid")) {
                    try {
                        vpp->fft3d.degrid = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("signorm")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->fft3d.signorm = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("prec")) {
                    int value = 0;
                    if (get_list_value(list_vpp_fp_prec, param_val.c_str(), &value)) {
                        vpp->fft3d.precision = (VppFpPrecision)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_fp_prec);
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
    if (IS_OPTION("vpp-degrain") && ENABLE_VPP_FILTER_DEGRAIN) {
        VppDegrain parsedDegrain;
        parsedDegrain.enable = true;
        const auto apply_auto_preset = [](VppDegrain& target) {
            target.preset = VppDegrainPreset::Auto;
            target.mode = VppDegrainMode::Degrain;
            target.stage = VppDegrainStage::TR2;
            target.delta = 2;
            target.search = 4;
            target.thsad = 300;
            target.thsadc = 150;
            target.thscd1 = 1600;
            target.thscd2 = 130;
            target.pel = 1;
            target.blksize = 16;
            target.overlap = 8;
            target.levels = 2;
            target.chroma = true;
            target.binomial = 0;
            target.tvRange = true;
        };
        const auto apply_tr = [&](VppDegrain& target, const int tr, const tstring& param_arg, const tstring& param_val) {
            if (tr != 1 && tr != 2) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr should be 1 or 2."));
                return 1;
            }
            target.mode = VppDegrainMode::Degrain;
            target.stage = (tr == 1) ? VppDegrainStage::TR1 : VppDegrainStage::TR2;
            target.delta = tr;
            return 0;
        };
        const auto set_degrain = [&](const VppDegrain& value) {
            if (value.mode == VppDegrainMode::Analyze) {
                auto parsed = value;
                if (parsed.stage == VppDegrainStage::Auto) {
                    parsed.stage = VppDegrainStage::TR1;
                }
                vpp->degrainAnalyze = parsed;
            } else if (value.mode == VppDegrainMode::Degrain && value.stage == VppDegrainStage::TR1) {
                vpp->degrainTR1 = value;
            } else if (value.mode == VppDegrainMode::Degrain && value.stage == VppDegrainStage::TR2) {
                vpp->degrainTR2 = value;
            } else {
                vpp->degrain = value;
            }
        };
        int parsedTr = 0;
        bool parsedOverlap = false;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            set_degrain(parsedDegrain);
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "enable", "preset", "mode", "stage", "tr", "blksize", "search", "thsad", "thsadc", "thscd1", "thscd2", "pel", "levels", "overlap", "delta", "tr0", "rep0", "search_refine",
            "subpelinterp", "searchparam", "pelsearch", "search_early_sad", "spatial_early_sad",
            "truemotion", "lambda", "lsad", "pnew", "plevel", "globalmotion", "dct", "useflag",
            "mv_spatial_refine", "chroma", "binomial", "tv_range"
        };
        const auto parse_int = [&](int *dst, const tstring& param_arg, const tstring& param_val) {
            try {
                size_t idx = 0;
                *dst = std::stoi(param_val, &idx);
                if (idx != param_val.length()) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } catch (...) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                return 1;
            }
            return 0;
        };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedDegrain.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("preset")) {
                    int value = 0;
                    if (get_list_value(list_vpp_degrain_preset, param_val.c_str(), &value)) {
                        parsedDegrain.preset = (VppDegrainPreset)value;
                        if (parsedDegrain.preset == VppDegrainPreset::Auto) {
                            apply_auto_preset(parsedDegrain);
                        }
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_degrain_preset);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_degrain_mode, param_val.c_str(), &value)) {
                        parsedDegrain.mode = (VppDegrainMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_degrain_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("stage")) {
                    int value = 0;
                    if (get_list_value(list_vpp_degrain_stage, param_val.c_str(), &value)) {
                        parsedDegrain.stage = (VppDegrainStage)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_degrain_stage);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr")) {
                    int value = 0;
                    if (parse_int(&value, param_arg, param_val)) {
                        return 1;
                    }
                    if (apply_tr(parsedDegrain, value, param_arg, param_val)) {
                        return 1;
                    }
                    parsedTr = value;
                    continue;
                }
                if (param_arg == _T("blksize")) {
                    if (parse_int(&parsedDegrain.blksize, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search")) {
                    if (parse_int(&parsedDegrain.search, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thsad")) {
                    if (parse_int(&parsedDegrain.thsad, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thsadc")) {
                    if (parse_int(&parsedDegrain.thsadc, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thscd1")) {
                    if (parse_int(&parsedDegrain.thscd1, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thscd2")) {
                    if (parse_int(&parsedDegrain.thscd2, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("pel")) {
                    if (parse_int(&parsedDegrain.pel, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("levels")) {
                    if (parse_int(&parsedDegrain.levels, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("overlap")) {
                    if (parse_int(&parsedDegrain.overlap, param_arg, param_val)) {
                        return 1;
                    }
                    parsedOverlap = true;
                    continue;
                }
                if (param_arg == _T("delta")) {
                    if (parse_int(&parsedDegrain.delta, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tr0")) {
                    if (parse_int(&parsedDegrain.tr0, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedDegrain.tr0 < 0 || parsedDegrain.tr0 > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("tr0 should be 0, 1, or 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rep0")) {
                    if (parse_int(&parsedDegrain.rep0, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedDegrain.rep0 < 0 || parsedDegrain.rep0 > 7) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("rep0 should be 0 - 7."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_refine")) {
                    if (parse_int(&parsedDegrain.searchRefine, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedDegrain.searchRefine < 0 || parsedDegrain.searchRefine > 3) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("search_refine should be 0 - 3."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("subpelinterp")) {
                    if (parse_int(&parsedDegrain.subpelInterp, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedDegrain.subpelInterp < 0 || parsedDegrain.subpelInterp > 2) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("subpelinterp should be 0, 1, or 2."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("searchparam")) {
                    if (parse_int(&parsedDegrain.searchParam, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("pelsearch")) {
                    if (parse_int(&parsedDegrain.pelSearch, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_early_sad")) {
                    if (parse_degrain_search_early_sad(&parsedDegrain.searchEarlySad, param_val, false)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("search_early_sad should be off or -1 - 65535."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("spatial_early_sad")) {
                    if (parse_degrain_search_early_sad(&parsedDegrain.spatialEarlySad, param_val, false)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("spatial_early_sad should be off or -1 - 65535."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("truemotion")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedDegrain.trueMotion = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lambda")) {
                    if (parse_int(&parsedDegrain.lambda, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lsad")) {
                    if (parse_int(&parsedDegrain.lsad, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("pnew")) {
                    if (parse_int(&parsedDegrain.pnew, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("plevel")) {
                    if (parse_int(&parsedDegrain.plevel, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("globalmotion")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedDegrain.globalMotion = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedDegrain.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("binomial")) {
                    if (tolowercase(param_val) == _T("auto")) {
                        parsedDegrain.binomial = -1;
                    } else {
                        bool b = false;
                        if (!cmd_string_to_bool(&b, param_val)) {
                            parsedDegrain.binomial = b ? 1 : 0;
                        } else {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("binomial should be true, false, or auto."));
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("mv_spatial_refine")) {
                    if (tolowercase(param_val) == _T("auto")) {
                        parsedDegrain.mvSpatialRefine = -1;
                    } else if (parse_int(&parsedDegrain.mvSpatialRefine, param_arg, param_val)) {
                        return 1;
                    }
                    if (parsedDegrain.mvSpatialRefine < -1) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("mv_spatial_refine should be auto, -1, or greater."));
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("tv_range")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        parsedDegrain.tvRange = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dct")) {
                    if (parse_int(&parsedDegrain.dct, param_arg, param_val)) {
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("useflag")) {
                    if (parse_int(&parsedDegrain.useFlag, param_arg, param_val)) {
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
        if (parsedTr > 0) {
            const auto expectedStage = (parsedTr == 1) ? VppDegrainStage::TR1 : VppDegrainStage::TR2;
            if (parsedDegrain.mode != VppDegrainMode::Degrain
                || parsedDegrain.stage != expectedStage
                || parsedDegrain.delta != parsedTr) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" tr="), strsprintf(_T("%d"), parsedTr),
                    _T("tr requires matching mode=degrain, stage, and delta."));
                return 1;
            }
        }
        if (parsedDegrain.preset == VppDegrainPreset::Auto) {
            if (!parsedOverlap) {
                parsedDegrain.overlap = parsedDegrain.blksize / 2;
            }
            const auto print_auto_error = [&](const TCHAR *message) {
                print_cmd_error_invalid_value(tstring(option_name) + _T(" preset=auto"), _T(""), message);
            };
            if (parsedDegrain.mode != VppDegrainMode::Degrain
                || (parsedDegrain.stage != VppDegrainStage::TR1 && parsedDegrain.stage != VppDegrainStage::TR2)
                || (parsedDegrain.delta != 1 && parsedDegrain.delta != 2)) {
                print_auto_error(_T("Auto preset supports only tr=1/2 degrain output."));
                return 1;
            }
            if ((parsedDegrain.stage == VppDegrainStage::TR1) != (parsedDegrain.delta == 1)) {
                print_auto_error(_T("Auto preset requires stage and delta to match tr=1/2."));
                return 1;
            }
            if (parsedDegrain.search != 4) {
                print_auto_error(_T("Auto preset requires search=4."));
                return 1;
            }
            if (parsedDegrain.levels != 2) {
                print_auto_error(_T("Auto preset requires levels=2."));
                return 1;
            }
            if (parsedDegrain.blksize != 8 && parsedDegrain.blksize != 16 && parsedDegrain.blksize != 32) {
                print_auto_error(_T("Auto preset supports blksize=8, 16, or 32."));
                return 1;
            }
            if (parsedDegrain.overlap != parsedDegrain.blksize / 2) {
                print_auto_error(_T("Auto preset requires overlap=blksize/2."));
                return 1;
            }
            if (parsedDegrain.pel != 1 && parsedDegrain.pel != 2) {
                print_auto_error(_T("Auto preset supports pel=1 or 2."));
                return 1;
            }
            if (!parsedDegrain.tvRange) {
                print_auto_error(_T("Auto preset requires tv_range=true for the search reference."));
                return 1;
            }
        }
        set_degrain(parsedDegrain);
        return 0;
    }
    if (IS_OPTION("vpp-msmooth") && ENABLE_VPP_FILTER_MSMOOTH) {
        vpp->msmooth.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "strength", "threshold", "threshold_c", "highq", "mask" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msmooth.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->msmooth.strength = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->msmooth.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold_c")) {
                    try {
                        vpp->msmooth.threshold_c = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("highq")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msmooth.highq = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mask")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msmooth.mask = b;
                    } else {
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

    if (IS_OPTION("vpp-libplacebo-shader") && ENABLE_VPP_FILTER_LIBPLACEBO) {
        VppLibplaceboShader shader;
        shader.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            vpp->libplacebo_shader.push_back(shader);
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{
            "shader", "res", "width", "height", "csp", "chromaloc", "colorsystem", "transfer", "resampler",
            "radius", "clamp", "taper", "blur", "antiring", "linear", "sigmoid", "sigmoid_center", "sigmoid_slope", "custom"
        };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        shader.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("shader")) {
                    shader.shader = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("res")) {
                    int width = 0, height = 0;
                    if (2 == _stscanf_s(param_val.c_str(), _T("%dx%d"), &width, &height)) {
                        shader.width = width;
                        shader.height = height;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("width")) {
                    try {
                        shader.width = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("height")) {
                    try {
                        shader.height = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("csp")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_shader_csp, param_val.c_str(), &value)) {
                        shader.csp = (VppLibplaceboInputCSP)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_shader_csp);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chromaloc")) {
                    int value = 0;
                    if (get_list_value(list_chromaloc_str, param_val.c_str(), &value)) {
                        shader.chromaloc = (CspChromaloc)value;
                    } else if (get_list_value(list_chromaloc, param_val.c_str(), &value)) {
                        shader.chromaloc = (CspChromaloc)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_chromaloc_str);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("colorsystem")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_colorsystem, param_val.c_str(), &value)) {
                        shader.colorsystem = (VppLibplaceboColorsystem)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_colorsystem);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("transfer")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_tone_mapping_transfer, param_val.c_str(), &value)) {
                        shader.transfer = (VppLibplaceboToneMappingTransfer)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_tone_mapping_transfer);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("resampler")) {
                    int value = 0;
                    const auto libplacebo_resize_list = get_libplacebo_only_resize_list();
                    if (get_list_value(libplacebo_resize_list.data(), param_val.c_str(), &value)) {
                        shader.resize_algo = (RGY_VPP_RESIZE_ALGO)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, libplacebo_resize_list.data());
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("radius")) {
                    try {
                        shader.radius = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("clamp")) {
                    try {
                        shader.clamp_ = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("taper")) {
                    try {
                        shader.taper = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("blur")) {
                    try {
                        shader.blur = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("antiring")) {
                    try {
                        shader.antiring = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("linear")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        shader.linear = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigmoid")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        shader.sigmoid = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigmoid_center")) {
                    try {
                        shader.sigmoid_center = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigmoid_slope")) {
                    try {
                        shader.sigmoid_slope = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("custom")) {
                    auto eqpos = param_val.find_first_of(_T("="));
                    if (eqpos == tstring::npos) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" custom="), param_val);
                        return 1;
                    }
                    shader.custom_params.push_back(std::make_pair(param_val.substr(0, eqpos), param_val.substr(eqpos + 1)));
                    continue;
                }
                shader.params.push_back(std::make_pair(param.substr(0, pos), param_val));
                continue;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        vpp->libplacebo_shader.push_back(shader);
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
    if (IS_OPTION("vpp-vinverse") && ENABLE_VPP_FILTER_VINVERSE) {
        vpp->vinverse.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mode", "sstr", "amnt", "scl", "thr", "chroma" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->vinverse.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_vinverse_mode, param_val.c_str(), &value)) {
                        vpp->vinverse.mode = (VppVinverseMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_vinverse_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->vinverse.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sstr")) {
                    try {
                        vpp->vinverse.sstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("amnt")) {
                    try {
                        vpp->vinverse.amnt = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scl")) {
                    try {
                        vpp->vinverse.scl = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thr")) {
                    try {
                        vpp->vinverse.thr = std::stof(param_val);
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
    if (IS_OPTION("vpp-chromashift") && ENABLE_VPP_FILTER_CHROMASHIFT) {
        vpp->chromashift.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "x", "y", "show", "auto", "auto_detect", "auto_frames", "auto_min_pairs" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->chromashift.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("x")) {
                    try {
                        vpp->chromashift.x = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("y")) {
                    try {
                        vpp->chromashift.y = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("show")) {
                    const auto show = tolowercase(param_val);
                    if (show == _T("normal") || show == _T("none") || show == _T("0")) {
                        vpp->chromashift.show = 0;
                    } else if (show == _T("laplacian") || show == _T("lap") || show == _T("1")) {
                        vpp->chromashift.show = 1;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("auto") || param_arg == _T("auto_detect")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->chromashift.auto_detect = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("auto_frames")) {
                    try {
                        vpp->chromashift.auto_frames = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("auto_min_pairs")) {
                    try {
                        vpp->chromashift.auto_min_pairs = std::stoi(param_val);
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
    if (IS_OPTION("vpp-deblock") && ENABLE_VPP_FILTER_DEBLOCK) {
        vpp->deblock.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "qp", "alpha", "beta", "chroma", "grid" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deblock.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("qp")) {
                    try {
                        vpp->deblock.qp = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("alpha")) {
                    try {
                        vpp->deblock.alpha = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("beta")) {
                    try {
                        vpp->deblock.beta = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("grid")) {
                    static const CX_DESC deblock_grid_list[] = {
                        { _T("4"), 4 }, { _T("8"), 8 },
                        { _T("h264"), 4 }, { _T("avc"), 4 },
                        { _T("mpeg2"), 8 }, { _T("mpeg4"), 8 },
                        { NULL, 0 }
                    };
                    int v = 0;
                    if (!get_list_value(deblock_grid_list, param_val.c_str(), &v)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, deblock_grid_list);
                        return 1;
                    }
                    vpp->deblock.grid = v;
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deblock.chroma = b;
                    } else {
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
    if (IS_OPTION("vpp-deflicker") && ENABLE_VPP_FILTER_DEFLICKER) {
        vpp->deflicker.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "strength", "damping", "scene_threshold", "frames", "predictor", "chroma" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deflicker.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->deflicker.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("damping")) {
                    try {
                        vpp->deflicker.damping = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("scene_threshold")) {
                    try {
                        vpp->deflicker.scene_threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("frames")) {
                    try {
                        vpp->deflicker.frames = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("predictor")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deflicker.predictor = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->deflicker.chroma = b;
                    } else {
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
    if (IS_OPTION("vpp-stab") && ENABLE_VPP_FILTER_STAB) {
        vpp->stab.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "strength", "damping", "trust", "trust_threshold", "max_shift", "border" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->stab.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->stab.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("damping")) {
                    try {
                        vpp->stab.damping = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("trust") || param_arg == _T("trust_threshold")) {
                    try {
                        vpp->stab.trust_threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("max_shift")) {
                    try {
                        vpp->stab.max_shift = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("border")) {
                    const auto value = tolowercase(param_val);
                    if (value == _T("black")) {
                        vpp->stab.border = VPP_STAB_BORDER_BLACK;
                    } else if (value == _T("clamp")) {
                        vpp->stab.border = VPP_STAB_BORDER_CLAMP;
                    } else if (value == _T("mirror")) {
                        vpp->stab.border = VPP_STAB_BORDER_MIRROR;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val,
                            _T("Supported values: black, clamp, mirror."));
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
    if (IS_OPTION("vpp-colorfix") && ENABLE_VPP_FILTER_COLORFIX) {
        vpp->colorfix.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mode", "space", "matrix", "white", "black", "white_r", "white_g", "white_b", "black_r", "black_g", "black_b", "frames", "strength", "variance_threshold", "temperature" };
        auto parse_color = [&](const tstring& value, int& r, int& g, int& b) {
            auto v = tolowercase(value);
            if (v.size() == 7 && v[0] == _T('#')) {
                v = v.substr(1);
            }
            if (v.size() == 6 && v.find_first_not_of(_T("0123456789abcdef")) == tstring::npos) {
                const int rgb = std::stoi(v, nullptr, 16);
                r = (rgb >> 16) & 0xff;
                g = (rgb >> 8) & 0xff;
                b = rgb & 0xff;
                return true;
            }
            const auto values = split(value, _T(":"));
            if (values.size() == 3) {
                try {
                    r = std::stoi(values[0]);
                    g = std::stoi(values[1]);
                    b = std::stoi(values[2]);
                } catch (...) {
                    return false;
                }
                return 0 <= r && r <= 255 && 0 <= g && g <= 255 && 0 <= b && b <= 255;
            }
            return false;
        };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->colorfix.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    const auto mode = tolowercase(param_val);
                    if (mode == _T("manual") || mode == _T("0")) {
                        vpp->colorfix.mode = VPP_COLORFIX_MODE_MANUAL;
                    } else if (mode == _T("auto") || mode == _T("1")) {
                        vpp->colorfix.mode = VPP_COLORFIX_MODE_AUTO;
                    } else if (mode == _T("gray") || mode == _T("grayworld") || mode == _T("2")) {
                        vpp->colorfix.mode = VPP_COLORFIX_MODE_GRAY;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("space")) {
                    const auto space = tolowercase(param_val);
                    if (space == _T("auto") || space == _T("0")) {
                        vpp->colorfix.space = VPP_COLORFIX_SPACE_AUTO;
                    } else if (space == _T("rgb") || space == _T("1")) {
                        vpp->colorfix.space = VPP_COLORFIX_SPACE_RGB;
                    } else if (space == _T("yuv") || space == _T("2")) {
                        vpp->colorfix.space = VPP_COLORFIX_SPACE_YUV;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("matrix")) {
                    const auto matrix = tolowercase(param_val);
                    if (matrix == _T("auto") || matrix == _T("0")) {
                        vpp->colorfix.matrix = VPP_COLORFIX_MATRIX_AUTO;
                    } else if (matrix == _T("bt601") || matrix == _T("601") || matrix == _T("1")) {
                        vpp->colorfix.matrix = VPP_COLORFIX_MATRIX_BT601;
                    } else if (matrix == _T("bt709") || matrix == _T("709") || matrix == _T("2")) {
                        vpp->colorfix.matrix = VPP_COLORFIX_MATRIX_BT709;
                    } else if (matrix == _T("bt2020") || matrix == _T("2020") || matrix == _T("3")) {
                        vpp->colorfix.matrix = VPP_COLORFIX_MATRIX_BT2020;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("white")) {
                    if (!parse_color(param_val, vpp->colorfix.whiteR, vpp->colorfix.whiteG, vpp->colorfix.whiteB)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("black")) {
                    if (!parse_color(param_val, vpp->colorfix.blackR, vpp->colorfix.blackG, vpp->colorfix.blackB)) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("white_r") || param_arg == _T("white_g") || param_arg == _T("white_b")
                    || param_arg == _T("black_r") || param_arg == _T("black_g") || param_arg == _T("black_b")
                    || param_arg == _T("frames") || param_arg == _T("temperature")) {
                    try {
                        size_t idx = 0;
                        const auto value = std::stoi(param_val, &idx);
                        if (idx != param_val.length()
                            || (param_arg == _T("temperature") && value != FILTER_DEFAULT_COLORFIX_TEMPERATURE
                                && (value < FILTER_MIN_COLORFIX_TEMPERATURE || value > FILTER_MAX_COLORFIX_TEMPERATURE))) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        if (param_arg == _T("white_r")) vpp->colorfix.whiteR = value;
                        if (param_arg == _T("white_g")) vpp->colorfix.whiteG = value;
                        if (param_arg == _T("white_b")) vpp->colorfix.whiteB = value;
                        if (param_arg == _T("black_r")) vpp->colorfix.blackR = value;
                        if (param_arg == _T("black_g")) vpp->colorfix.blackG = value;
                        if (param_arg == _T("black_b")) vpp->colorfix.blackB = value;
                        if (param_arg == _T("frames"))  vpp->colorfix.frames = value;
                        if (param_arg == _T("temperature")) vpp->colorfix.temperature = value;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->colorfix.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("variance_threshold")) {
                    try {
                        vpp->colorfix.varianceThreshold = std::stof(param_val);
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
    if (IS_OPTION("vpp-dehalo") && ENABLE_VPP_FILTER_DEHALO) {
        vpp->dehalo.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mode", "rx", "ry", "darkstr", "brightstr", "lowsens", "highsens", "ss", "search_rade", "search_radi" };
        bool searchRadeSet = false;
        bool searchRadiSet = false;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->dehalo.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_dehalo_mode, param_val.c_str(), &value)) {
                        vpp->dehalo.mode = (VppDehaloMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_dehalo_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rx")) {
                    try {
                        vpp->dehalo.rx = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ry")) {
                    try {
                        vpp->dehalo.ry = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("darkstr")) {
                    try {
                        vpp->dehalo.darkstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("brightstr")) {
                    try {
                        vpp->dehalo.brightstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lowsens")) {
                    try {
                        vpp->dehalo.lowsens = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("highsens")) {
                    try {
                        vpp->dehalo.highsens = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ss")) {
                    try {
                        vpp->dehalo.ss = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_rade")) {
                    try {
                        vpp->dehalo.searchRade = (tolowercase(param_val) == _T("auto")) ? FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO : std::stoi(param_val);
                        searchRadeSet = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_radi")) {
                    try {
                        vpp->dehalo.searchRadi = (tolowercase(param_val) == _T("auto")) ? FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO : std::stoi(param_val);
                        searchRadiSet = true;
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
        if (searchRadeSet && !searchRadiSet) {
            vpp->dehalo.searchRadi = vpp->dehalo.searchRade;
        }
        return 0;
    }
    if (IS_OPTION("vpp-finedehalo") && ENABLE_VPP_FILTER_FINEDEHALO) {
        vpp->finedehalo.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mode", "rx", "ry", "darkstr", "brightstr", "lowsens", "highsens", "ss", "search_rade", "search_radi", "thmi", "thma", "thlimi", "thlima", "showmask", "excl", "edgeproc", "edge" };
        bool searchRadeSet = false;
        bool searchRadiSet = false;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->finedehalo.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_dehalo_mode, param_val.c_str(), &value)) {
                        vpp->finedehalo.mode = (VppDehaloMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_dehalo_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("rx")) {
                    try {
                        vpp->finedehalo.rx = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ry")) {
                    try {
                        vpp->finedehalo.ry = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("darkstr")) {
                    try {
                        vpp->finedehalo.darkstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("brightstr")) {
                    try {
                        vpp->finedehalo.brightstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lowsens")) {
                    try {
                        vpp->finedehalo.lowsens = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("highsens")) {
                    try {
                        vpp->finedehalo.highsens = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ss")) {
                    try {
                        vpp->finedehalo.ss = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_rade")) {
                    try {
                        vpp->finedehalo.searchRade = (tolowercase(param_val) == _T("auto")) ? FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO : std::stoi(param_val);
                        searchRadeSet = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("search_radi")) {
                    try {
                        vpp->finedehalo.searchRadi = (tolowercase(param_val) == _T("auto")) ? FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO : std::stoi(param_val);
                        searchRadiSet = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thmi")) {
                    try {
                        vpp->finedehalo.thmi = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thma")) {
                    try {
                        vpp->finedehalo.thma = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thlimi")) {
                    try {
                        vpp->finedehalo.thlimi = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("thlima")) {
                    try {
                        vpp->finedehalo.thlima = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("showmask")) {
                    try {
                        vpp->finedehalo.showmask = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("excl")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->finedehalo.excl = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("edgeproc")) {
                    try {
                        vpp->finedehalo.edgeproc = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("edge")) {
                    const auto edge = check_str_list(option_name, param_arg, param_val,
                        { _T("prewitt"), _T("sobel"), _T("scharr"), _T("kirsch"), _T("laplacian") });
                    if (!edge) return 1;
                    vpp->finedehalo.edge = *edge;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        if (searchRadeSet && !searchRadiSet) {
            vpp->finedehalo.searchRadi = vpp->finedehalo.searchRade;
        }
        return 0;
    }
    if (IS_OPTION("vpp-guidedfilter") && ENABLE_VPP_FILTER_GUIDEDFILTER) {
        vpp->guidedfilter.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "radius", "eps", "chroma" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == tstring::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto param_arg = tolowercase(param.substr(0, pos));
            const auto param_val = param.substr(pos + 1);
            if (param_arg == _T("enable")) {
                bool value = false;
                if (cmd_string_to_bool(&value, param_val)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                vpp->guidedfilter.enable = value;
            } else if (param_arg == _T("radius")) {
                try {
                    size_t parsed = 0;
                    vpp->guidedfilter.radius = std::stoi(param_val, &parsed);
                    if (parsed != param_val.length()) {
                        throw std::invalid_argument("trailing characters");
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("eps")) {
                try {
                    size_t parsed = 0;
                    vpp->guidedfilter.eps = std::stof(param_val, &parsed);
                    if (parsed != param_val.length() || !std::isfinite(vpp->guidedfilter.eps)) {
                        throw std::invalid_argument("invalid float");
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("chroma")) {
                bool value = false;
                if (cmd_string_to_bool(&value, param_val)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                vpp->guidedfilter.chroma = value;
            } else {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-clahe") && ENABLE_VPP_FILTER_CLAHE) {
        vpp->clahe.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "tiles_x", "tiles_y", "slope" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == tstring::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto param_arg = tolowercase(param.substr(0, pos));
            const auto param_val = param.substr(pos + 1);
            if (param_arg == _T("enable")) {
                bool value = false;
                if (cmd_string_to_bool(&value, param_val)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
                vpp->clahe.enable = value;
            } else if (param_arg == _T("tiles_x") || param_arg == _T("tiles_y")) {
                try {
                    size_t parsed = 0;
                    const int value = std::stoi(param_val, &parsed);
                    if (parsed != param_val.length()) {
                        throw std::invalid_argument("trailing characters");
                    }
                    if (param_arg == _T("tiles_x")) {
                        vpp->clahe.tiles_x = value;
                    } else {
                        vpp->clahe.tiles_y = value;
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("slope")) {
                try {
                    size_t parsed = 0;
                    vpp->clahe.slope = std::stof(param_val, &parsed);
                    if (parsed != param_val.length() || !std::isfinite(vpp->clahe.slope)) {
                        throw std::invalid_argument("invalid float");
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else {
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-dehaze") && ENABLE_VPP_FILTER_DEHAZE) {
        vpp->dehaze.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "patch_radius", "omega", "t_floor", "atm_light" };
        for (const auto& param : split(strInput[i], _T(","))) {
            const auto pos = param.find_first_of(_T("="));
            if (pos == tstring::npos) {
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
            const auto paramArg = tolowercase(param.substr(0, pos));
            const auto paramValue = param.substr(pos + 1);
            if (paramArg == _T("enable")) {
                bool value = false;
                if (cmd_string_to_bool(&value, paramValue)) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + paramArg + _T("="), paramValue);
                    return 1;
                }
                vpp->dehaze.enable = value;
            } else if (paramArg == _T("patch_radius")) {
                try {
                    size_t parsed = 0;
                    vpp->dehaze.patch_radius = std::stoi(paramValue, &parsed);
                    if (parsed != paramValue.length()) {
                        throw std::invalid_argument("trailing characters");
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + paramArg + _T("="), paramValue);
                    return 1;
                }
            } else if (paramArg == _T("omega") || paramArg == _T("t_floor") || paramArg == _T("atm_light")) {
                float value = 0.0f;
                try {
                    size_t parsed = 0;
                    value = std::stof(paramValue, &parsed);
                    if (parsed != paramValue.length() || !std::isfinite(value)) {
                        throw std::invalid_argument("invalid float");
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + paramArg + _T("="), paramValue);
                    return 1;
                }
                if (paramArg == _T("omega")) {
                    vpp->dehaze.omega = value;
                } else if (paramArg == _T("t_floor")) {
                    vpp->dehaze.t_floor = value;
                } else {
                    vpp->dehaze.atm_light = value;
                }
            } else {
                print_cmd_error_unknown_opt_param(option_name, paramArg, paramList);
                return 1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-hqdering") && ENABLE_VPP_FILTER_HQDERING) {
        vpp->dering.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "mrad", "mthr", "sigma", "showmask", "protect", "edge" , "thr", "elast", "darkthr", "minp", "msmooth", "drrep", "sharp", "planes" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->dering.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mrad")) {
                    try {
                        vpp->dering.mrad = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mthr")) {
                    try {
                        vpp->dering.mthr = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sigma")) {
                    try {
                        vpp->dering.sigma = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("showmask")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->dering.showmask = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("protect")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->dering.protect = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("edge")) {
                    const auto edge = check_str_list(option_name, param_arg, param_val,
                        { _T("log"), _T("sobel"), _T("prewitt"), _T("scharr"), _T("kirsch"), _T("laplacian") });
                    if (!edge) return 1;
                    vpp->dering.edge = *edge;
                    continue;
                }
                if (param_arg == _T("thr")) {
                    try {
                        vpp->dering.thr = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("darkthr")) {
                    try {
                        vpp->dering.darkthr = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("minp")) {
                    try {
                        vpp->dering.minp = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("msmooth")) {
                    try {
                        vpp->dering.msmooth = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("drrep")) {
                    try {
                        vpp->dering.drrep = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sharp")) {
                    try {
                        vpp->dering.sharp = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("planes")) {
                    vpp->dering.planes = { false, false, false };
                    if (param_val == _T("all")) {
                        vpp->dering.planes = { true, true, true };
                    } else {
                        for (const auto& plane : split(param_val, _T(":"))) {
                            if      (plane == _T("y")) vpp->dering.planes[0] = true;
                            else if (plane == _T("u")) vpp->dering.planes[1] = true;
                            else if (plane == _T("v")) vpp->dering.planes[2] = true;
                            else {
                                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, _T("Supported values: all, or \":\"-separated list of y, u, v (e.g. y:u:v)."));
                                return 1;
                            }
                        }
                    }
                    continue;
                }
                if (param_arg == _T("elast")) {
                    try {
                        vpp->dering.elast = std::stof(param_val);
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
    if (IS_OPTION("vpp-msharpen") && ENABLE_VPP_FILTER_MSHARPEN) {
        vpp->msharpen.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "strength", "threshold", "slope", "luma_limit", "block_protect", "highq", "mask" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msharpen.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vpp->msharpen.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->msharpen.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("slope")) {
                    try {
                        vpp->msharpen.slope = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("luma_limit")) {
                    try {
                        vpp->msharpen.luma_limit = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("block_protect")) {
                    try {
                        vpp->msharpen.block_protect = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("highq")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msharpen.highq = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mask")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->msharpen.mask = b;
                    } else {
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
    if (IS_OPTION("vpp-cas") && ENABLE_VPP_FILTER_CAS) {
        vpp->cas.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "sharpness", "chroma", "hdr" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->cas.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sharpness")) {
                    try {
                        vpp->cas.sharpness = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chroma")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->cas.chroma = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("hdr")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->cas.hdr = b;
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
    if (IS_OPTION("vpp-warpsharp") && ENABLE_VPP_FILTER_WARPSHARP) {
        vpp->warpsharp.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "threshold", "blur", "type", "depth", "chroma", "depth_min", "depth_max", "edge_thr", "gamma" };
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
                if (param_arg == _T("depth_min")) {
                    try {
                        vpp->warpsharp.depth_min = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("depth_max")) {
                    try {
                        vpp->warpsharp.depth_max = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("edge_thr")) {
                    try {
                        vpp->warpsharp.edge_thr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("gamma")) {
                    try {
                        vpp->warpsharp.gamma = std::stof(param_val);
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
    if (IS_OPTION("vpp-detailsharpen") && ENABLE_VPP_FILTER_DETAILSHARPEN) {
        vpp->detailsharpen.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "enable", "z", "sstr", "power", "ldmp", "mode", "med" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->detailsharpen.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("z")) {
                    try {
                        vpp->detailsharpen.z = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("sstr")) {
                    try {
                        vpp->detailsharpen.sstr = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("power")) {
                    try {
                        vpp->detailsharpen.power = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("ldmp")) {
                    try {
                        vpp->detailsharpen.ldmp = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    try {
                        size_t parsed = 0;
                        vpp->detailsharpen.mode = std::stoi(param_val, &parsed);
                        if (parsed != param_val.length()) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    if (vpp->detailsharpen.mode != 0 && vpp->detailsharpen.mode != 1) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("med")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->detailsharpen.med = b;
                    } else {
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

        const auto paramList = std::vector<std::string>{ "r", "g", "b", "m", "red", "green", "blue", "master", "all", "preset", "interp" };

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
                    vpp->curves.all = param_val;
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
                if (param_arg == _T("interp")) {
                    int value = 0;
                    if (get_list_value(list_vpp_curves_interp, param_val.c_str(), &value)) {
                        vpp->curves.interp = (VppCurvesInterp)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_curves_interp);
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

    if (IS_OPTION("vpp-softlight") && ENABLE_VPP_FILTER_SOFTLIGHT) {
        vpp->softlight.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "mode", "formula", "skipblack" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->softlight.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    param_val = tolowercase(param_val);
                    if (get_list_value(list_vpp_softlight_mode, param_val.c_str(), &value)) {
                        vpp->softlight.mode = (VppSoftLightMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_softlight_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("formula")) {
                    int value = 0;
                    param_val = tolowercase(param_val);
                    if (get_list_value(list_vpp_softlight_formula, param_val.c_str(), &value)) {
                        vpp->softlight.formula = (VppSoftLightFormula)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_softlight_formula);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("skipblack")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->softlight.skipblack = b;
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

    if (IS_OPTION("vpp-tweak") && ENABLE_VPP_FILTER_TWEAK) {
        vpp->tweak.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;

        auto paramList = std::vector<std::string>{ "contrast", "brightness", "gamma", "saturation", "vibrance", "swapuv", "hue", "coring", "start_hue", "end_hue" };
        for (auto& channel : { "y", "cb", "cr", "r", "g", "b" }) {
            paramList.push_back(std::string(channel) + "_offset");
            paramList.push_back(std::string(channel) + "_gain");
        }
        for (auto& channel : { "r", "g", "b" }) { // gammaに対応するのは r, g, b のみ
            paramList.push_back(std::string(channel) + "_gamma");
        }

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
                if (param_arg == _T("vibrance")) {
                    try {
                        size_t idx = 0;
                        vpp->tweak.vibrance = std::stof(param_val, &idx);
                        if (idx != param_val.length() || !std::isfinite(vpp->tweak.vibrance)
                            || vpp->tweak.vibrance < FILTER_MIN_TWEAK_VIBRANCE
                            || vpp->tweak.vibrance > FILTER_MAX_TWEAK_VIBRANCE) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
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
                if (param_arg == _T("coring")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->tweak.coring = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("start_hue")) {
                    try {
                        vpp->tweak.startHue = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("end_hue")) {
                    try {
                        vpp->tweak.endHue = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                bool parse_gamma = false; // gammaのパラメータをパースのは r, g, b のみで、 y, cb, crは対象外
                auto param_subopt = param_arg;
                VppTweakChannel *tweak_channel = nullptr;
                if (param_arg.length() > 2 && param_arg[1] == _T('_')) {
                    param_subopt = param_arg.substr(2);
                    switch (param_arg[0]) {
                    case _T('r'): tweak_channel = &vpp->tweak.r; parse_gamma = true; break;
                    case _T('g'): tweak_channel = &vpp->tweak.g; parse_gamma = true; break;
                    case _T('b'): tweak_channel = &vpp->tweak.b; parse_gamma = true; break;
                    case _T('y'): tweak_channel = &vpp->tweak.y; break;
                    default: break;
                    }
                }
                if (!tweak_channel && param_arg.length() > 3 && param_arg[2] == _T('_')) {
                    param_subopt = param_arg.substr(3);
                    if (param_arg.substr(0, 2) == _T("cb")) {
                        tweak_channel = &vpp->tweak.cb;
                    } else if (param_arg.substr(0, 2) == _T("cr")) {
                        tweak_channel = &vpp->tweak.cr;
                    }
                }
                if (tweak_channel) {
                    // param_subopt から、 offset, gain, gammaを取り出す
                    if (param_subopt == _T("offset")) {
                        try {
                            tweak_channel->offset = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    } else if (param_subopt == _T("gain")) {
                        try {
                            tweak_channel->gain = std::stof(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                        continue;
                    } else if (param_subopt == _T("gamma") && parse_gamma) {
                        try {
                            tweak_channel->gamma = std::stof(param_val);
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
                if (param == _T("swapuv")) {
                    vpp->tweak.swapuv = true;
                    continue;
                }
                if (param == _T("coring")) {
                    vpp->tweak.coring = true;
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
    if (IS_OPTION("vpp-v360")) {
        vpp->v360.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "in", "out", "yaw", "pitch", "roll", "in_hfov", "h_fov", "w", "h" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) { vpp->v360.enable = b; }
                    else { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("in")) {
                    int value = 0;
                    if (get_list_value(list_vpp_v360_proj, param_val.c_str(), &value)) { vpp->v360.in_proj = value; }
                    else { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_v360_proj); return 1; }
                    continue;
                }
                if (param_arg == _T("out")) {
                    int value = 0;
                    if (get_list_value(list_vpp_v360_proj, param_val.c_str(), &value)) { vpp->v360.out_proj = value; }
                    else { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_v360_proj); return 1; }
                    continue;
                }
                if (param_arg == _T("yaw")) {
                    try { vpp->v360.yaw = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("pitch")) {
                    try { vpp->v360.pitch = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("roll")) {
                    try { vpp->v360.roll = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("h_fov")) {
                    try { vpp->v360.out_hfov = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("in_hfov")) {
                    try { vpp->v360.in_hfov = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("w")) {
                    try { vpp->v360.w = std::stoi(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("h")) {
                    try { vpp->v360.h = std::stoi(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
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
    if (IS_OPTION("vpp-lenscorrection")) {
        vpp->lenscorrection.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "k1", "k2", "cx", "cy", "vignette" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->lenscorrection.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("k1")) {
                    try { vpp->lenscorrection.k1 = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("k2")) {
                    try { vpp->lenscorrection.k2 = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("cx")) {
                    try { vpp->lenscorrection.cx = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("cy")) {
                    try { vpp->lenscorrection.cy = std::stof(param_val); } catch (...) { print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val); return 1; }
                    continue;
                }
                if (param_arg == _T("vignette")) {
                    try {
                        size_t idx = 0;
                        vpp->lenscorrection.vignette = std::stof(param_val, &idx);
                        if (idx != param_val.length() || !std::isfinite(vpp->lenscorrection.vignette)
                            || vpp->lenscorrection.vignette < -1.0f || vpp->lenscorrection.vignette > 4.0f) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
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
    if (IS_OPTION("vpp-libplacebo-deband") && ENABLE_VPP_FILTER_LIBPLACEBO) {
        vpp->libplacebo_deband.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{
            "iterations", "threshold", "radius",
            "grain_y", "grain_c", "dither", "lut_size" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->libplacebo_deband.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("iterations")) {
                    try {
                        vpp->libplacebo_deband.iterations = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        vpp->libplacebo_deband.threshold = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("radius")) {
                    try {
                        vpp->libplacebo_deband.radius = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("grain_y")) {
                    try {
                        vpp->libplacebo_deband.grainY = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("grain_c")) {
                    try {
                        vpp->libplacebo_deband.grainC = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("dither")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_deband_dither_mode, param_val.c_str(), &value)) {
                        vpp->libplacebo_deband.dither = (VppLibplaceboDebandDitherMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_libplacebo_deband_dither_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("lut_size")) {
                    int value = 0;
                    if (get_list_value(list_vpp_libplacebo_deband_lut_size, param_val.c_str(), &value)) {
                        vpp->libplacebo_deband.lut_size = value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_ass_shaping);
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
    if (IS_OPTION("vpp-fruc") && ENABLE_VPP_FILTER_FRUC) {
        vpp->fruc.enable = true;
        if (ENCODER_NVENC) {
            vpp->fruc.mode = VppFrucMode::NVOFFRUCx2;
        }
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "fps" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vpp->fruc.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("fps")) {
                    int a[2] = { 0 };
                    if (   2 == _stscanf_s(param_val.c_str(), _T("%d/%d"), &a[0], &a[1])
                        || 2 == _stscanf_s(param_val.c_str(), _T("%d:%d"), &a[0], &a[1])
                        || 2 == _stscanf_s(param_val.c_str(), _T("%d,%d"), &a[0], &a[1])) {
                        vpp->fruc.targetFps = rgy_rational<int>(a[0], a[1]);
                        vpp->fruc.mode = VppFrucMode::NVOFFRUCFps;
                    } else {
                        double d;
                        if (1 == _stscanf_s(param_val.c_str(), _T("%lf"), &d)) {
                            int rate = (int)(d * 1001.0 + 0.5);
                            if (rate % 1000 == 0) {
                                vpp->fruc.targetFps = rgy_rational<int>(rate, 1001);
                            } else {
                                vpp->fruc.targetFps = rgy_rational<int>((int)(d * 100000 + 0.5), 100000);
                            }
                            vpp->fruc.mode = VppFrucMode::NVOFFRUCFps;
                        } else {
                            print_cmd_error_invalid_value(option_name, strInput[i]);
                            return 1;
                        }
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("double")) {
                    vpp->fruc.mode = VppFrucMode::NVOFFRUCx2;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
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
                if (param_arg == _T("ignore_sar")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        inprm->ignoreSAR = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else if (param == _T("ignore_sar")) {
                inprm->ignoreSAR = true;
                continue;
            } else {
                int a[2] = { 0 };
                if (   2 == _stscanf_s(param.c_str(), _T("%dx%d"), &a[0], &a[1])
                    || 2 == _stscanf_s(param.c_str(), _T("%d:%d"), &a[0], &a[1])) {
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
        if (i + 1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            inprm->avswDecoder = strInput[i];
        }
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

int parse_audio_bitrate(std::vector<AudioBitrate>& ret, const tstring& str) {
    auto prms = split(tchar_to_string(str), ",");
    for (const auto& prm : prms) {
        auto delim = prm.find(":");
        try {
            if (delim == std::string::npos) {
                ret.push_back(AudioBitrate(std::stoi(prm)));
            } else {
                ret.push_back(AudioBitrate(prm.substr(0, delim), std::stoi(prm.substr(delim + 1))));
            }
        } catch (...) {
            print_cmd_error_invalid_value(_T("--audio-bitrate "), char_to_tstring(prm));
            return 1;
        }
    }
    return 0;
}

int parse_one_audio_param(AudioSelect& chSel, const tstring& str, const TCHAR *option_name) {
    const auto paramList = std::vector<std::string>{ "codec", "bitrate", "quality", "samplerate", "delay", "profile", "disposition", "filter", "dec_prm", "enc_prm", "lang", "select-codec", "metadata", "bsf", "resampler", "copy" };
    for (const auto &param : split(str, _T(";"))) {
        auto pos = param.find_first_of(_T("="));
        if (pos != std::string::npos) {
            auto param_arg = param.substr(0, pos);
            auto param_val = param.substr(pos + 1);
            if (param_arg == _T("codec")) {
                chSel.encCodec = param_val;
            } else if (param_arg == _T("bitrate")) {
                try {
                    if (parse_audio_bitrate(chSel.encBitrate, param_val)) {
                        return 1;
                    }
                } catch (...) {
                    print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                    return 1;
                }
            } else if (param_arg == _T("quality")) {
                try {
                    chSel.encQuality = { true, std::stoi(param_val) };
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
            } else if (param_arg == _T("resampler")) {
                chSel.resamplerPrm = tchar_to_string(param_val);
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
    if (IS_OPTION("input-pixel-format")) {
        i++;
        common->inputPixFmtStr = strInput[i];
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
    if (IS_OPTION("avcodec-prms")) {
        i++;
        common->avcodec_videnc_prms = strInput[i];
        return 0;
    }
    if (IS_OPTION("video-metadata")) {
        i++;
        common->videoMetadata.push_back(strInput[i]);
        return 0;
    }
    if (IS_OPTION("trim")) {
        i++;
        tstring trim_arg_str = strInput[i];
        // strInput[i]が "Trim" で始まるかで判定し、AviSynthのTrim形式を変換する
        if (0 == _tcsncmp(tolowercase(trim_arg_str).c_str(), _T("trim"), _tcslen(_T("trim")))) {
            trim_arg_str = tolowercase(trim_arg_str);
            trim_arg_str = str_replace(trim_arg_str, _T(" "), _T("")); // 空白除去
            trim_arg_str = str_replace(trim_arg_str, _T(","), _T(":")); // "," -> ":"
            trim_arg_str = str_replace(trim_arg_str, _T(")++trim("), _T(",")); // ")++trim(" -> ","
            trim_arg_str = str_replace(trim_arg_str, _T("trim("), _T(""));
            trim_arg_str = str_replace(trim_arg_str, _T(")"), _T(""));
        }
        auto trim_str_list = split(trim_arg_str, _T(","));
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
        double sec = 0.0;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%lf"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%lf"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%lf"),              &sec))) {
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
            common->seekSec = (float)sec + mm * 60;
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
                        if (ichannel + 1 < channel_select_list.size()) {
                            src.inputOpt.push_back(std::make_pair<tstring, tstring>(tstring(channel.substr(option_split + 1)), tstring(channel_select_list[ichannel + 1])));
                            ichannel++;
                        }
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
    if (IS_OPTION("y4m-timestamp")) {
        common->y4mTimestamp = true;
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
    // allowNoValue: 値が省略された場合に func_set へ nullptr を渡してよいかどうか
    auto set_audio_prm = [&](std::function<void(AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set, const bool allowNoValue = false) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        std::vector<int> excludeTrackIDs;
        bool invalidTrackSelector = false;
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
                    const bool excludeLang = !tempc.empty() && tempc[0] == '!';
                    auto langCode = excludeLang ? tempc.substr(1) : tempc;
                    if (excludeLang) {
                        langCode = str_replace(langCode, ",!", ",");
                    }
                    const auto langList = split(langCode, ",");
                    if (std::all_of(langList.begin(), langList.end(), [](const auto& lang) { return rgy_lang_exist(lang); })) {
                        trackId = excludeLang ? TRACK_SELECT_BY_LANG_EXCLUDE : TRACK_SELECT_BY_LANG;
                        lang = langCode;
                    } else if (parseTrackExclusion(tempc, excludeTrackIDs)) {
                        trackId = TRACK_SELECT_BY_TRACK_EXCLUDE;
                    } else if (avcodec_exists_audio(tempc)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    } else if (excludeLang) {
                        invalidTrackSelector = true;
                    }
                }
            }
        }
        if (invalidTrackSelector) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if ((trackId == TRACK_SELECT_BY_TRACK_EXCLUDE && getAudioTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0)
            || (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && getAudioTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0)) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
        }
        if (ptr == nullptr && !allowNoValue) {
            print_cmd_error_invalid_value(option_name, tstring(), _T("value not specified"));
            return 1;
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
        if (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && !pAudioSelect->lang.empty()) {
            pAudioSelect->lang += "," + lang;
        } else {
            pAudioSelect->lang = lang;
        }
        if (trackId == TRACK_SELECT_BY_TRACK_EXCLUDE) {
            mergeTrackExclusions(pAudioSelect->excludeTrackIDs, excludeTrackIDs);
        }
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
    // allowNoValue: 値が省略された場合に func_set へ nullptr を渡してよいかどうか
    auto set_sub_prm = [&](std::function<void(SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr)> func_set, const bool allowNoValue = false) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        std::vector<int> excludeTrackIDs;
        bool invalidTrackSelector = false;
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
                    const bool excludeLang = !tempc.empty() && tempc[0] == '!';
                    auto langCode = excludeLang ? tempc.substr(1) : tempc;
                    if (excludeLang) {
                        langCode = str_replace(langCode, ",!", ",");
                    }
                    const auto langList = split(langCode, ",");
                    if (std::all_of(langList.begin(), langList.end(), [](const auto& lang) { return rgy_lang_exist(lang); })) {
                        trackId = excludeLang ? TRACK_SELECT_BY_LANG_EXCLUDE : TRACK_SELECT_BY_LANG;
                        lang = langCode;
                    } else if (parseTrackExclusion(tempc, excludeTrackIDs)) {
                        trackId = TRACK_SELECT_BY_TRACK_EXCLUDE;
                    } else if (avcodec_exists_subtitle(tempc)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    } else if (excludeLang) {
                        invalidTrackSelector = true;
                    }
                }
            }
        }
        if (invalidTrackSelector) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if ((trackId == TRACK_SELECT_BY_TRACK_EXCLUDE && getSubTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0)
            || (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && getSubTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0)) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
        }
        if (ptr == nullptr && !allowNoValue) {
            print_cmd_error_invalid_value(option_name, tstring(), _T("value not specified"));
            return 1;
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
        if (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && !pSubSelect->lang.empty()) {
            pSubSelect->lang += "," + lang;
        } else {
            pSubSelect->lang = lang;
        }
        if (trackId == TRACK_SELECT_BY_TRACK_EXCLUDE) {
            mergeTrackExclusions(pSubSelect->excludeTrackIDs, excludeTrackIDs);
        }
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
    // allowNoValue: 値が省略された場合に func_set へ nullptr を渡してよいかどうか
    auto set_data_prm = [&](std::function<void(DataSelect *pSelect, int trackId, const TCHAR *prmstr)> func_set, const bool allowNoValue = false) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        std::string lang;
        std::string selectCodec;
        std::vector<int> excludeTrackIDs;
        bool invalidTrackSelector = false;
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
                    const bool excludeLang = !tempc.empty() && tempc[0] == '!';
                    auto langCode = excludeLang ? tempc.substr(1) : tempc;
                    if (excludeLang) {
                        langCode = str_replace(langCode, ",!", ",");
                    }
                    const auto langList = split(langCode, ",");
                    if (std::all_of(langList.begin(), langList.end(), [](const auto& lang) { return rgy_lang_exist(lang); })) {
                        trackId = excludeLang ? TRACK_SELECT_BY_LANG_EXCLUDE : TRACK_SELECT_BY_LANG;
                        lang = langCode;
                    } else if (parseTrackExclusion(tempc, excludeTrackIDs)) {
                        trackId = TRACK_SELECT_BY_TRACK_EXCLUDE;
                    } else if (avcodec_exists_data(tempc)) {
                        trackId = TRACK_SELECT_BY_CODEC;
                        selectCodec = tempc;
                    } else if (excludeLang) {
                        invalidTrackSelector = true;
                    }
                }
            }
        }
        if (invalidTrackSelector) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if ((trackId == TRACK_SELECT_BY_TRACK_EXCLUDE && getDataTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0)
            || (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && getDataTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0)) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
        }
        if (ptr == nullptr && !allowNoValue) {
            print_cmd_error_invalid_value(option_name, tstring(), _T("value not specified"));
            return 1;
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
        if (trackId == TRACK_SELECT_BY_LANG_EXCLUDE && !pSelect->lang.empty()) {
            pSelect->lang += "," + lang;
        } else {
            pSelect->lang = lang;
        }
        if (trackId == TRACK_SELECT_BY_TRACK_EXCLUDE) {
            mergeTrackExclusions(pSelect->excludeTrackIDs, excludeTrackIDs);
        }
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
        std::string excludeLangs;
        std::vector<int> excludeTrackIDs;
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    const auto strTrack = tchar_to_string(str);
                    const bool excludeLang = !strTrack.empty() && strTrack[0] == '!';
                    const auto langCode = excludeLang ? strTrack.substr(1) : strTrack;
                    if (rgy_lang_exist(langCode)) {
                        if (excludeLang) {
                            excludeLangs += (excludeLangs.empty() ? "" : ",") + langCode;
                        } else {
                            trackSet.insert(std::make_pair(TRACK_SELECT_BY_LANG, langCode));
                        }
                    } else if (parseTrackExclusion(strTrack, excludeTrackIDs)) {
                    } else if (avcodec_exists_audio(tchar_to_string(str))) {
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
        if (!excludeLangs.empty()) {
            trackSet.insert(std::make_pair(TRACK_SELECT_BY_LANG_EXCLUDE, excludeLangs));
        }
        if (!excludeTrackIDs.empty()) {
            if (!excludeLangs.empty() || getAudioTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
                return 1;
            }
            trackSet.insert(std::make_pair(TRACK_SELECT_BY_TRACK_EXCLUDE, ""));
        }
        if (!excludeLangs.empty() && getAudioTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
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
            if (it->first == TRACK_SELECT_BY_TRACK_EXCLUDE) {
                mergeTrackExclusions(pAudioSelect->excludeTrackIDs, excludeTrackIDs);
            }
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
        }, true /*値省略時は自動選択とするため許可する*/);
        if (ret) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("audio-encode-other-codec-only")) {
        common->audioEncodeOtherCodecOnly = true;
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
    if (IS_OPTION("audio-quality")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || !pAudioSelect->encQuality.first) {
                    pAudioSelect->encQuality = { true, std::stoi(prmstr) };
                }
                });
            return ret;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-bitrate")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encBitrate.size() == 0) {
                    if (parse_audio_bitrate(pAudioSelect->encBitrate, prmstr)) {
                        throw std::runtime_error("invalid bitrate");
                    }
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
        int v = 0;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i+1]))) {
            i++;
            common->audioResampler = v;
        } else {
            try {
                auto ret = set_audio_prm([](AudioSelect* pAudioSelect, int trackId, const TCHAR* prmstr) {
                    if (trackId != 0 || pAudioSelect->resamplerPrm.length() == 0) {
                        pAudioSelect->resamplerPrm = tchar_to_string(prmstr);
                    }
                    });
                return ret;
            } catch (...) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
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
        std::string excludeLangs;
        std::vector<int> excludeTrackIDs;
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
                    } else {
                        const auto strTrack = tchar_to_string(str);
                        const bool excludeLang = !strTrack.empty() && strTrack[0] == '!';
                        const auto langCode = excludeLang ? strTrack.substr(1) : strTrack;
                        if (rgy_lang_exist(langCode)) {
                            if (excludeLang) {
                                excludeLangs += (excludeLangs.empty() ? "" : ",") + langCode;
                            } else {
                                auto track = std::make_pair(TRACK_SELECT_BY_LANG, langCode);
                                trackSet[track].trackID = track.first;
                                trackSet[track].encCodec = RGY_AVCODEC_COPY;
                                trackSet[track].lang = langCode;
                            }
                        } else if (parseTrackExclusion(strTrack, excludeTrackIDs)) {
                        } else if (avcodec_exists(strTrack, AVMEDIA_TYPE_SUBTITLE)) {
                            auto track = std::make_pair(TRACK_SELECT_BY_CODEC, strTrack);
                            trackSet[track].trackID = TRACK_SELECT_BY_CODEC;
                            trackSet[track].encCodec = RGY_AVCODEC_COPY;
                            trackSet[track].selectCodec = strTrack;
                        } else {
                            print_cmd_error_unknown_opt_param(option_name, str, paramList);
                            return 1;
                        }
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
        if (!excludeLangs.empty()) {
            auto track = std::make_pair(TRACK_SELECT_BY_LANG_EXCLUDE, excludeLangs);
            trackSet[track].trackID = TRACK_SELECT_BY_LANG_EXCLUDE;
            trackSet[track].encCodec = RGY_AVCODEC_COPY;
            trackSet[track].lang = excludeLangs;
        }
        if (!excludeTrackIDs.empty()) {
            if (!excludeLangs.empty() || getSubTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
                return 1;
            }
            auto track = std::make_pair(TRACK_SELECT_BY_TRACK_EXCLUDE, "");
            trackSet[track].trackID = TRACK_SELECT_BY_TRACK_EXCLUDE;
            trackSet[track].encCodec = RGY_AVCODEC_COPY;
            trackSet[track].excludeTrackIDs = excludeTrackIDs;
        }
        if (!excludeLangs.empty() && getSubTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
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
            if (track.first == TRACK_SELECT_BY_TRACK_EXCLUDE) {
                mergeTrackExclusions(pSubtitleSelect->excludeTrackIDs, it->second.excludeTrackIDs);
                pSubtitleSelect->trackID = track.first;
                pSubtitleSelect->encCodec = it->second.encCodec;
            } else {
                pSubtitleSelect[0] = it->second;
            }

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
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
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
        }, true /*値省略時は自動選択とするため許可する*/);
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
                        if (ichannel + 1 < channel_select_list.size()) {
                            src.inputOpt.push_back(std::make_pair<tstring, tstring>(tstring(channel.substr(option_split + 1)), tstring(channel_select_list[ichannel + 1])));
                            ichannel++;
                        }
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
        std::string excludeLangs;
        std::vector<int> excludeTrackIDs;
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    const auto strTrack = tchar_to_string(str);
                    const bool excludeLang = !strTrack.empty() && strTrack[0] == '!';
                    const auto langCode = excludeLang ? strTrack.substr(1) : strTrack;
                    if (rgy_lang_exist(langCode)) {
                        if (excludeLang) {
                            excludeLangs += (excludeLangs.empty() ? "" : ",") + langCode;
                        } else {
                            auto track = std::make_pair(TRACK_SELECT_BY_LANG, langCode);
                            trackSet[track].trackID = TRACK_SELECT_BY_LANG;
                            trackSet[track].lang = langCode;
                            trackSet[track].encCodec = RGY_AVCODEC_COPY;
                        }
                    } else if (parseTrackExclusion(strTrack, excludeTrackIDs)) {
                    } else if (avcodec_exists(strTrack, AVMEDIA_TYPE_DATA)) {
                        auto track = std::make_pair(TRACK_SELECT_BY_CODEC, strTrack);
                        trackSet[track].trackID = TRACK_SELECT_BY_CODEC;
                        trackSet[track].selectCodec = strTrack;
                        trackSet[track].encCodec = RGY_AVCODEC_COPY;
                    } else {
                        print_cmd_error_invalid_value(option_name, strInput[i], _T("invalid track ID."));
                        return 1;
                    }
                } else {
                    auto track = std::make_pair(iTrack, "");
                    trackSet[track].trackID = iTrack;
                    trackSet[track].encCodec = RGY_AVCODEC_COPY;
                }
            }
        } else {
            auto track = std::make_pair(0, "");
            trackSet[track].trackID = 0;
        }
        if (!excludeLangs.empty()) {
            auto track = std::make_pair(TRACK_SELECT_BY_LANG_EXCLUDE, excludeLangs);
            trackSet[track].trackID = TRACK_SELECT_BY_LANG_EXCLUDE;
            trackSet[track].lang = excludeLangs;
            trackSet[track].encCodec = RGY_AVCODEC_COPY;
        }
        if (!excludeTrackIDs.empty()) {
            if (!excludeLangs.empty() || getDataTrackIdx(common, TRACK_SELECT_BY_LANG_EXCLUDE, "", "") >= 0) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
                return 1;
            }
            auto track = std::make_pair(TRACK_SELECT_BY_TRACK_EXCLUDE, "");
            trackSet[track].trackID = TRACK_SELECT_BY_TRACK_EXCLUDE;
            trackSet[track].excludeTrackIDs = excludeTrackIDs;
            trackSet[track].encCodec = RGY_AVCODEC_COPY;
        }
        if (!excludeLangs.empty() && getDataTrackIdx(common, TRACK_SELECT_BY_TRACK_EXCLUDE, "", "") >= 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("language and track exclusions cannot be mixed"));
            return 1;
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
            if (track.first == TRACK_SELECT_BY_TRACK_EXCLUDE) {
                mergeTrackExclusions(pDataSelect->excludeTrackIDs, it->second.excludeTrackIDs);
                pDataSelect->trackID = track.first;
                pDataSelect->encCodec = RGY_AVCODEC_COPY;
            } else {
                pDataSelect[0] = it->second;
                pDataSelect->encCodec = RGY_AVCODEC_COPY;
            }

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
                    trackSet[iTrack].encCodec = RGY_AVCODEC_COPY;
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
            pAttachmentSelect->encCodec = RGY_AVCODEC_COPY;

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
    if (IS_OPTION("muxer-add-cmd")) {
        common->muxerAddCmd = true;
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
        if (get_list_value(list_chromaloc_str, strInput[i], &value)) {
            common->out_vui.chromaloc = (CspChromaloc)value;
        } else if (get_list_value(list_chromaloc, strInput[i], &value)) {
            common->out_vui.chromaloc = (CspChromaloc)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_chromaloc_str);
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
        if (get_list_value(list_dovi_profile_parse, strInput[i], &value)) {
            common->doviProfile = (RGYDOVIProfile)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_dovi_profile);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("dolby-vision-rpu")) {
        i++;
        if (strInput[i] == tstring(_T("copy"))) {
            common->doviRpuMetadataCopy = true;
        } else {
            common->doviRpuFile = strInput[i];
        }
        return 0;
    }
    if (IS_OPTION("dolby-vision-rpu-prm")) {
        i++;
        const auto paramList = std::vector<std::string>{ "crop" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = tolowercase(param.substr(0, pos));
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("crop")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->doviRpuParams.activeAreaOffsets.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("crop")) {
                    common->doviRpuParams.activeAreaOffsets.enable = true;
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
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
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_bsf_mode);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("adapt-resolution")) {
        if (i + 1 >= nArgNum) {
            print_cmd_error_invalid_value(option_name, _T(""));
            return 1;
        }
        i++;
        int resolution[2] = { 0, 0 };
        // 共通パーサでは形式と正数であることだけを確認する。サーフェス型の上限や初期入力との大小関係は、
        // encoderとヘッダ解析結果に依存するため、reader初期化後のpipeline側で検証する。
        if (2 != _stscanf_s(strInput[i], _T("%dx%d"), &resolution[0], &resolution[1])
            || resolution[0] <= 0 || resolution[1] <= 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        common->adaptResolution = std::make_pair(resolution[0], resolution[1]);
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
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('\0') || strInput[i + 1][0] == _T('-')) {
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
#if ENABLE_LIBVSHIP
    if (IS_OPTION("no-vship-ssimulacra2")) {
        common->metric.vshipSsimu2.enable = false;
        return 0;
    }
    if (IS_OPTION("vship-ssimulacra2")) {
        common->metric.vshipSsimu2.enable = true;
        return 0;
    }
    if (IS_OPTION("no-vship-butteraugli")) {
        common->metric.vshipButteraugli.enable = false;
        return 0;
    }
    if (IS_OPTION("vship-butteraugli")) {
        common->metric.vshipButteraugli.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('\0') || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "Qnorm", "intensity_multiplier" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vshipButteraugli.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("Qnorm")) {
                    try {
                        common->metric.vshipButteraugli.Qnorm = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("intensity_multiplier")) {
                    try {
                        common->metric.vshipButteraugli.intensity_multiplier = std::stof(param_val);
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
    if (IS_OPTION("no-vship-cvvdp")) {
        common->metric.vshipCvvdp.enable = false;
        return 0;
    }
    if (IS_OPTION("vship-cvvdp")) {
        common->metric.vshipCvvdp.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('\0') || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "enable", "model", "model_config_json", "resize" };

        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vshipCvvdp.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("model")) {
                    common->metric.vshipCvvdp.model = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("model_config_json")) {
                    common->metric.vshipCvvdp.model_config_json = trim(param_val, _T("\""));
                    continue;
                }
                if (param_arg == _T("resize")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        common->metric.vshipCvvdp.resize = b;
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
#endif //#if ENABLE_LIBVSHIP
    if (IS_OPTION("disable-av1-write-parser")) {
        common->debugDirectAV1Out = true;
        return 0;
    }
    if (IS_OPTION("debug-raw-out")) {
        common->debugRawOut = true;
        return 0;
    }
    if (IS_OPTION("offset-video-dts-advance")) {
        common->offsetVideoDtsAdvance = true;
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
                        ctrl->logOpt.addTime = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("addlevel")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logOpt.addLogLevel = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("color")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        ctrl->logOpt.disableColor = !b;
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
                    ctrl->logOpt.addTime = true;
                    continue;
                } else if (param == _T("addlevel")) {
                    ctrl->logOpt.addLogLevel = true;
                    continue;
                } else if (param == _T("color")) {
                    ctrl->logOpt.disableColor = true;
                    continue;
                } else if (param == _T("no-color")) {
                    ctrl->logOpt.disableColor = true;
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
    if (IS_OPTION("no-avoid-idle-clock")) {
        ctrl->avoidIdleClock.mode = RGYParamAvoidIdleClockMode::Disabled;
    }
    if (IS_OPTION("avoid-idle-clock")) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        i++;
        auto params = split(strInput[i], _T("="));
        int value = 0;
        if (get_list_value(list_avoid_idle_clock, params[0].c_str(), &value)) {
            ctrl->avoidIdleClock.mode = (RGYParamAvoidIdleClockMode)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_avoid_idle_clock);
            return -1;
        }
        if (params.size() > 1) {
            float load = 0.0f;
            if (1 != _stscanf_s(params[1].c_str(), _T("%f"), &load)) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return -1;
            }
            if (value < 0) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return -1;
            }
            ctrl->avoidIdleClock.loadPercent = load;
        }
        return 0;
    }
    if (IS_OPTION("task-perf-monitor")) {
        ctrl->taskPerfMonitor = true;
        return 0;
    }
    if (IS_OPTION("lowlatency")) {
        ctrl->lowLatency = true;
        return 0;
    }
    if (IS_OPTION("fallback-bitdepth")) {
        ctrl->fallbackBitdepth = true;
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
#if defined(_WIN32) || defined(_WIN64)
    if (IS_OPTION("vsdir")) {
        i++;
        ctrl->vsdir = strInput[i];
        return 0;
    }
#endif
    if (IS_OPTION("vpy-assume-script-dir")) {
        ctrl->vpyAssumeScriptDir = true;
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
    if (IS_OPTION("python")) {
        i++;
        ctrl->pythonPath = strInput[i];
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
    if (IS_OPTION("disable-vulkan")) {
        ctrl->enableVulkan = RGYParamInitVulkan::Disable;
        return 0;
    }
    if (IS_OPTION("enable-vulkan")) {
        ctrl->enableVulkan = RGYParamInitVulkan::TargetVendor;
        return 0;
    }
    if (IS_OPTION("enable-vulkan-all")) {
        ctrl->enableVulkan = RGYParamInitVulkan::All;
        return 0;
    }
    if (IS_OPTION("opencl-build-threads")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        ctrl->openclBuildThreads = value;
        return 0;
    }
#if ENCODER_QSV
    if (IS_OPTION("opencl-task-threads")) {
        i++;
        int value = -1;
        if (_tcsicmp(strInput[i], _T("auto")) == 0) {
            value = -1;
        } else if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value != -1 && value != 0 && value != 2 && value != 3) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("opencl-task-threads should be auto, 0, 2, or 3."));
            return 1;
        }
        ctrl->openclTaskThreads = value;
        return 0;
    }
#endif
    if (IS_OPTION("cl-perf-dump")) {
        i++;
        ctrl->clPerfDumpDir = strInput[i];
        return 0;
    }
    if (IS_OPTION("cl-perf-timeline")) {
        if (i + 1 < nArgNum && strInput[i + 1][0] != _T('-')) {
            i++;
            try {
                ctrl->clPerfTimelineSec = std::stod(tchar_to_string(strInput[i]));
            } catch (...) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        } else {
            ctrl->clPerfTimelineSec = 10.0;
        }
        if (ctrl->clPerfTimelineSec == 0.0) ctrl->clPerfTimelineSec = 10.0;
        return 0;
    }
    if (IS_OPTION("cl-perf-disasm-tool")) {
        i++;
        const auto value = tolowercase(strInput[i]);
        if (value != _T("auto") && value != _T("ocloc") && value != _T("rga") && value != _T("none")) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        ctrl->clPerfDisasmTool = value;
        return 0;
    }
    if (IS_OPTION("ocloc-path")) {
        i++;
        ctrl->clPerfOclocPath = strInput[i];
        return 0;
    }
    if (IS_OPTION("rga-path")) {
        i++;
        ctrl->clPerfRgaPath = strInput[i];
        return 0;
    }
    if (IS_OPTION("parallel") && ENABLE_PARALLEL_ENC) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            ctrl->parallelEnc.parallelCount = -1;
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "id" };
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("mp")) {
                    if (param == _T("auto")) {
                        ctrl->parallelEnc.parallelCount = -1;
                    } else {
                        try {
                            ctrl->parallelEnc.parallelCount = std::stoi(param_val);
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("id")) {
                    try {
                        ctrl->parallelEnc.parallelId = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chunks")) {
                    try {
                        ctrl->parallelEnc.chunks = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("chunk-handles")) {
                    auto handles = split(param_val, _T(":"));
                    for (const auto& handle : handles) {
                        try {
                            auto handle_frame_id = split(handle, _T("#"));
                            if (handle_frame_id.size() != 2) {
                                print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                                return 1;
                            }
                            ctrl->parallelEnc.chunkPipeHandles.push_back(RGYParamParallelEncPipeHandle(std::stoull(handle_frame_id[0]), std::stoi(handle_frame_id[1])));
                        } catch (...) {
                            print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                            return 1;
                        }
                    }
                    continue;
                }
                if (param_arg == _T("cache")) {
                    int value = 0;
                    if (get_list_value(list_parallel_enc_cache, param_val.c_str(), &value)) {
                        ctrl->parallelEnc.cacheMode = (RGYParamParallelEncCache)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_parallel_enc_cache);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                if (param == _T("auto")) {
                    ctrl->parallelEnc.parallelCount = -1;
                } else {
                    try {
                        ctrl->parallelEnc.parallelCount = std::stoi(param);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name), param);
                        return 1;
                    }
                }
                continue;
            }
        }
        return 0;
    }
    if (IS_OPTION("parallel-force-large-memory-filters") && ENABLE_PARALLEL_ENC) {
        ctrl->parallelEnc.forceLargeMemoryFilters = true;
        return 0;
    }
    if (IS_OPTION("process-monitor-dev-usage")) {
        ctrl->processMonitorDevUsage = true;
        return 0;
    }
    if (IS_OPTION("process-monitor-dev-usage-reset")) {
        ctrl->processMonitorDevUsageReset = true;
        return 0;
    }
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
#define OPT_CHAR_PATH(str, opt) if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && (param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" \"") << (param->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && param->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (param->opt.c_str()) << _T("\"");

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

tstring gen_cmd(const VideoInfo *param, const VideoInfo *defaultPrm, const RGYParamInput *inprm, const RGYParamInput *inprmDefault, bool save_disabled_prm, RGYDisableGenCmdFlags disable_flags) {
    std::basic_stringstream<TCHAR> cmd;
    if (rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::InputPrms)) {
        return _T("");
    }
    switch (param->type) {
    case RGY_INPUT_FMT_RAW:    cmd << _T(" --raw"); break;
    case RGY_INPUT_FMT_Y4M:    cmd << _T(" --y4m"); break;
    case RGY_INPUT_FMT_AVI:    cmd << _T(" --avi"); break;
    case RGY_INPUT_FMT_AVS:    cmd << _T(" --avs"); break;
    case RGY_INPUT_FMT_VPY:    cmd << _T(" --vpy"); break;
    case RGY_INPUT_FMT_VPY_MT: cmd << _T(" --vpy-mt"); break;
    case RGY_INPUT_FMT_AVHW:   cmd << _T(" --avhw"); break;
    case RGY_INPUT_FMT_AVSW:   cmd << _T(" --avsw"); if (!inprm->avswDecoder.empty()) cmd << _T(" ") << inprm->avswDecoder; break;
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
        if (inprm->ignoreSAR != inprmDefault->ignoreSAR) {
            cmd << _T(",ignore_sar=") << (inprm->ignoreSAR ? (_T("true")) : (_T("false")));
        }
    }
    return cmd.str();
}

static tstring printTrackExclusions(const std::vector<int>& trackIDs) {
    tstring trackList;
    for (const auto trackID : trackIDs) {
        trackList += strsprintf(_T(",!%d"), trackID);
    }
    return trackList.empty() ? tstring() : trackList.substr(1);
}

tstring printTrack(const AudioSelect *sel) {
    if (sel->trackID == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        return printTrackExclusions(sel->excludeTrackIDs);
    }
    auto lang = char_to_tstring(sel->lang);
    if (sel->trackID == TRACK_SELECT_BY_LANG_EXCLUDE) {
        lang = str_replace(lang, _T(","), _T(",!"));
        return _T("!") + lang;
    }
    return sel->trackID == TRACK_SELECT_BY_LANG ? lang : std::to_tstring(sel->trackID);
};
tstring printTrack(const SubtitleSelect *sel) {
    if (sel->trackID == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        return printTrackExclusions(sel->excludeTrackIDs);
    }
    auto lang = char_to_tstring(sel->lang);
    if (sel->trackID == TRACK_SELECT_BY_LANG_EXCLUDE) {
        lang = str_replace(lang, _T(","), _T(",!"));
        return _T("!") + lang;
    }
    return sel->trackID == TRACK_SELECT_BY_LANG ? lang : std::to_tstring(sel->trackID);
};
tstring printTrack(const DataSelect *sel) {
    if (sel->trackID == TRACK_SELECT_BY_TRACK_EXCLUDE) {
        return printTrackExclusions(sel->excludeTrackIDs);
    }
    auto lang = char_to_tstring(sel->lang);
    if (sel->trackID == TRACK_SELECT_BY_LANG_EXCLUDE) {
        lang = str_replace(lang, _T(","), _T(",!"));
        return _T("!") + lang;
    }
    return sel->trackID == TRACK_SELECT_BY_LANG ? lang : std::to_tstring(sel->trackID);
};

// "<トラック指定>?" を出力する
// トラックID 0 はトラック指定なしを意味するので、そのまま出力すると
// パースできない "0?" になってしまう。この場合は何も出力しない
template<typename T>
tstring printTrackPrm(const T *sel) {
    return (sel->trackID == 0) ? tstring() : printTrack(sel) + _T("?");
};


tstring gen_cmd(const RGYParamVpp *param, const RGYParamVpp *defaultPrm, bool save_disabled_prm, RGYDisableGenCmdFlags disable_flags) {
    std::basic_stringstream<TCHAR> cmd;
    std::basic_stringstream<TCHAR> tmp;

    tmp.str(tstring());
    for (auto& vpptype : param->filterOrder) {
        tmp << _T(",") << vppfilter_type_to_str(vpptype);
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --vpp-order ") << tmp.str().substr(1);
    }
    if (param->deintCsp == VppDeintCsp::Output) {
        cmd << _T(" --vpp-deint-csp output");
    }

    if (!isNgxResizeFiter(param->resize_algo) && !isQSVMFXResizeFiter(param->resize_algo)) {
        if (isLibplaceboResizeFiter(param->resize_algo)) {
            OPT_LST(_T("--vpp-resize"), resize_algo, list_vpp_resize);
            if (param->resize_libplacebo.radius != defaultPrm->resize_libplacebo.radius) {
                cmd << _T(",pl-radius=") << std::setprecision(3) << param->resize_libplacebo.radius;
            }
            if (param->resize_libplacebo.clamp_ != defaultPrm->resize_libplacebo.clamp_) {
                cmd << _T(",pl-clamp=") << std::setprecision(3) << param->resize_libplacebo.clamp_;
            }
            if (param->resize_libplacebo.taper != defaultPrm->resize_libplacebo.taper) {
                cmd << _T(",pl-taper=") << std::setprecision(3) << param->resize_libplacebo.taper;
            }
            if (param->resize_libplacebo.blur != defaultPrm->resize_libplacebo.blur) {
                cmd << _T(",pl-blur=") << std::setprecision(3) << param->resize_libplacebo.blur;
            }
            if (param->resize_libplacebo.antiring != defaultPrm->resize_libplacebo.antiring) {
                cmd << _T(",pl-antiring=") << std::setprecision(3) << param->resize_libplacebo.antiring;
            }
            if (param->resize_libplacebo.cplace != defaultPrm->resize_libplacebo.cplace) {
                cmd << _T(",pl-cplace=") << param->resize_libplacebo.cplace;
            }
        } else {
            OPT_LST(_T("--vpp-resize"), resize_algo, list_vpp_resize);
            if (param->resize_algo == RGY_VPP_RESIZE_DPID) {
                tmp.str(tstring());
                tmp.clear();
                ADD_FLOAT2(_T("dpid_lambda"), param->resize_dpid, defaultPrm->resize_dpid, lambda, 3);
                cmd << tmp.str();
            }
            if (param->resize_algo == RGY_VPP_RESIZE_FSR1
                && param->resize_fsr1.sharpness != defaultPrm->resize_fsr1.sharpness) {
                cmd << _T(",sharpness=") << std::setprecision(3) << param->resize_fsr1.sharpness;
            }
            if (param->resize_algo == RGY_VPP_RESIZE_NIS) {
                if (param->resize_nis.cascade != defaultPrm->resize_nis.cascade) {
                    cmd << _T(",cascade=") << get_chr_from_value(list_vpp_resize_nis_cascade, param->resize_nis.cascade);
                }
                if (param->resize_nis.sharpness != defaultPrm->resize_nis.sharpness) {
                    cmd << _T(",sharpness=") << std::setprecision(3) << param->resize_nis.sharpness;
                }
                if (param->resize_nis.hdrMode != defaultPrm->resize_nis.hdrMode) {
                    cmd << _T(",hdr=") << get_chr_from_value(list_vpp_resize_nis_hdr, param->resize_nis.hdrMode);
                }
                if (param->resize_nis.opt != defaultPrm->resize_nis.opt) {
                    cmd << _T(",opt=") << get_chr_from_value(list_vpp_resize_nis_opt, param->resize_nis.opt);
                }
            }
            if (param->resize_algo == RGY_VPP_RESIZE_BICUBIC) {
                if (param->resize_bicubic.b != defaultPrm->resize_bicubic.b) {
                    cmd << _T(",b=") << std::setprecision(3) << param->resize_bicubic.b;
                }
                if (param->resize_bicubic.c != defaultPrm->resize_bicubic.c) {
                    cmd << _T(",c=") << std::setprecision(3) << param->resize_bicubic.c;
                }
            }
        }
    }
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
                const auto& from = param->colorspace.convs[i].from;
                const auto& to = param->colorspace.convs[i].to;
                // CLFILTERS_AUFではエントリを省略してしまうと入力側の情報が不明になるため、省略しない
                if (from.matrix != to.matrix || CLFILTERS_AUF) {
                    tmp << _T(",matrix=");
                    tmp << get_cx_desc(list_colormatrix, from.matrix);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colormatrix, to.matrix);
                }
                if (from.colorprim != to.colorprim || CLFILTERS_AUF) {
                    tmp << _T(",colorprim=");
                    tmp << get_cx_desc(list_colorprim, from.colorprim);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorprim, to.colorprim);
                }
                if (from.transfer != to.transfer || CLFILTERS_AUF) {
                    tmp << _T(",transfer=");
                    tmp << get_cx_desc(list_transfer, from.transfer);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_transfer, to.transfer);
                }
                if (from.colorrange != to.colorrange || CLFILTERS_AUF) {
                    tmp << _T(",range=");
                    tmp << get_cx_desc(list_colorrange, from.colorrange);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorrange, to.colorrange);
                }
                ColorspaceConv convDefault;
                if (param->colorspace.convs[i].approx_gamma != convDefault.approx_gamma) {
                    tmp << _T(",approx_gamma=");
                    tmp << (param->colorspace.convs[i].approx_gamma ? _T("true") : _T("false"));
                }
                if (param->colorspace.convs[i].scene_ref != convDefault.scene_ref) {
                    tmp << _T(",scene_ref=");
                    tmp << (param->colorspace.convs[i].scene_ref ? _T("true") : _T("false"));
                }
            }
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
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-colorspace ") << tmp.str().substr(1);
        } else if (param->colorspace.enable) {
            cmd << _T(" --vpp-colorspace");
        }
    }
    if (param->libplacebo_tonemapping != defaultPrm->libplacebo_tonemapping) {
        tmp.str(tstring());
        if (!param->libplacebo_tonemapping.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->libplacebo_tonemapping.enable || save_disabled_prm) {
            ADD_LST(_T("src_csp"), libplacebo_tonemapping.src_csp, list_vpp_libplacebo_tone_mapping_csp);
            ADD_LST(_T("dst_csp"), libplacebo_tonemapping.dst_csp, list_vpp_libplacebo_tone_mapping_csp);
            ADD_FLOAT(_T("src_max"), libplacebo_tonemapping.src_max, 3);
            ADD_FLOAT(_T("src_min"), libplacebo_tonemapping.src_min, 3);
            ADD_FLOAT(_T("dst_max"), libplacebo_tonemapping.dst_max, 3);
            ADD_FLOAT(_T("dst_min"), libplacebo_tonemapping.dst_min, 3);
            ADD_BOOL(_T("dynamic_peak_detection"), libplacebo_tonemapping.dynamic_peak_detection);
            ADD_FLOAT(_T("smooth_period"), libplacebo_tonemapping.smooth_period, 3);
            ADD_FLOAT(_T("scene_threshold_low"), libplacebo_tonemapping.scene_threshold_low, 3);
            ADD_FLOAT(_T("scene_threshold_high"), libplacebo_tonemapping.scene_threshold_high, 3);
            ADD_FLOAT(_T("percentile"), libplacebo_tonemapping.percentile, 3);
            ADD_FLOAT(_T("black_cutoff"), libplacebo_tonemapping.black_cutoff, 3);
            ADD_LST(_T("gamut_mapping"), libplacebo_tonemapping.gamut_mapping, list_vpp_libplacebo_tone_mapping_gamut_mapping);
            ADD_LST(_T("tonemapping_function"), libplacebo_tonemapping.tonemapping_function, list_vpp_libplacebo_tone_mapping_function);
            ADD_LST(_T("metadata"), libplacebo_tonemapping.metadata, list_vpp_libplacebo_tone_mapping_metadata);
            ADD_FLOAT(_T("contrast_recovery"), libplacebo_tonemapping.contrast_recovery, 3);
            ADD_FLOAT(_T("contrast_smoothness"), libplacebo_tonemapping.contrast_smoothness, 3);
            ADD_BOOL(_T("inverse_tone_mapping"), libplacebo_tonemapping.inverse_tone_mapping);
            ADD_BOOL(_T("visualize_lut"), libplacebo_tonemapping.visualize_lut);
            ADD_BOOL(_T("show_clipping"), libplacebo_tonemapping.show_clipping);
            ADD_NUM(_T("use_dovi"), libplacebo_tonemapping.use_dovi);
            ADD_PATH(_T("lut_path"), libplacebo_tonemapping.lut_path.c_str());
            ADD_LST(_T("lut_type"), libplacebo_tonemapping.lut_type, list_vpp_libplacebo_tone_mapping_lut_type);
            ADD_LST(_T("dst_pl_transfer"), libplacebo_tonemapping.dst_pl_transfer, list_vpp_libplacebo_tone_mapping_transfer);
            ADD_LST(_T("dst_pl_colorprim"), libplacebo_tonemapping.dst_pl_colorprim, list_vpp_libplacebo_tone_mapping_colorprim);
            ADD_FLOAT(_T("knee_adaptation"), libplacebo_tonemapping.tone_constants.st2094.knee_adaptation, 3);
            ADD_FLOAT(_T("knee_min"), libplacebo_tonemapping.tone_constants.st2094.knee_min, 3);
            ADD_FLOAT(_T("knee_max"), libplacebo_tonemapping.tone_constants.st2094.knee_max, 3);
            ADD_FLOAT(_T("knee_default"), libplacebo_tonemapping.tone_constants.st2094.knee_default, 3);
            ADD_FLOAT(_T("knee_offset"), libplacebo_tonemapping.tone_constants.bt2390.knee_offset, 3);
            ADD_FLOAT(_T("slope_tuning"), libplacebo_tonemapping.tone_constants.spline.slope_tuning, 3);
            ADD_FLOAT(_T("slope_offset"), libplacebo_tonemapping.tone_constants.spline.slope_offset, 3);
            ADD_FLOAT(_T("spline_contrast"), libplacebo_tonemapping.tone_constants.spline.spline_contrast, 3);
            ADD_FLOAT(_T("reinhard_contrast"), libplacebo_tonemapping.tone_constants.reinhard.contrast, 3);
            ADD_FLOAT(_T("linear_knee"), libplacebo_tonemapping.tone_constants.mobius.linear_knee, 3);
            ADD_FLOAT(_T("exposure"), libplacebo_tonemapping.tone_constants.linear.exposure, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-libplacebo-tonemapping ") << tmp.str().substr(1);
        } else if (param->libplacebo_tonemapping.enable) {
            cmd << _T(" --vpp-libplacebo-tonemapping");
        }
    }
    OPT_STR_PATH(_T("--vpp-libplacebo-tonemapping-lut"), libplacebo_tonemapping.lut_path);

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
            ADD_LST(_T("tune"), afs.tune, list_afs_tune_mode);
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
            if (param->nnedi.planes != defaultPrm->nnedi.planes) {
                tstring p;
                if (param->nnedi.planes[0]) p += _T(":y");
                if (param->nnedi.planes[1]) p += _T(":u");
                if (param->nnedi.planes[2]) p += _T(":v");
                tmp << _T(",planes=") << (p.length() > 0 ? p.substr(1) : _T(""));
            }
            ADD_LST(_T("nsize"), nnedi.nsize, list_vpp_nnedi_nsize);
            ADD_LST(_T("nns"), nnedi.nns, list_vpp_nnedi_nns);
            ADD_LST(_T("quality"), nnedi.quality, list_vpp_nnedi_quality);
            ADD_NUM(_T("prescreen"), nnedi.prescreen);
            ADD_LST(_T("errortype"), nnedi.errortype, list_vpp_nnedi_error_type);
            ADD_NUM(_T("clamp"), nnedi.clamp);
            ADD_BOOL(_T("double_height"), nnedi.doubleHeight);
            ADD_PATH(_T("weightfile"), nnedi.weightfile.c_str());
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nnedi ") << tmp.str().substr(1);
        } else if (param->nnedi.enable) {
            cmd << _T(" --vpp-nnedi");
        }
    }
    if (param->nnediUpscale != defaultPrm->nnediUpscale) {
        tmp.str(tstring());
        if (!param->nnediUpscale.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nnediUpscale.enable || save_disabled_prm) {
            ADD_LST(_T("nsize"), nnediUpscale.nnedi.nsize, list_vpp_nnedi_nsize);
            ADD_LST(_T("nns"), nnediUpscale.nnedi.nns, list_vpp_nnedi_nns);
            ADD_LST(_T("quality"), nnediUpscale.nnedi.quality, list_vpp_nnedi_quality);
            ADD_NUM(_T("prescreen"), nnediUpscale.nnedi.prescreen);
            ADD_LST(_T("errortype"), nnediUpscale.nnedi.errortype, list_vpp_nnedi_error_type);
            ADD_NUM(_T("clamp"), nnediUpscale.nnedi.clamp);
            if (param->nnediUpscale.shiftCubic != defaultPrm->nnediUpscale.shiftCubic) {
                tmp << _T(",shift=") << (param->nnediUpscale.shiftCubic ? _T("cubic") : _T("linear"));
            }
            ADD_PATH(_T("weightfile"), nnediUpscale.nnedi.weightfile.c_str());
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nnedi-upscale ") << tmp.str().substr(1);
        } else if (param->nnediUpscale.enable) {
            cmd << _T(" --vpp-nnedi-upscale");
        }
    }
    if (param->yadif != defaultPrm->yadif) {
        tmp.str(tstring());
        if (!param->yadif.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->yadif.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), yadif.mode, list_vpp_yadif_mode);
            ADD_BOOL(_T("log"), yadif.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-yadif ") << tmp.str().substr(1);
        } else if (param->yadif.enable) {
            cmd << _T(" --vpp-yadif");
        }
    }
    if (param->bwdif != defaultPrm->bwdif) {
        tmp.str(tstring());
        if (!param->bwdif.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->bwdif.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), bwdif.mode, list_vpp_bwdif_mode);
            ADD_LST(_T("order"), bwdif.order, list_vpp_bwdif_order);
            ADD_FLOAT(_T("thr"), bwdif.thr, 1);
            if (param->bwdif.deint != defaultPrm->bwdif.deint) {
                tmp << _T(",deint=") << ((param->bwdif.deint == VppBwdifDeint::Interlaced) ? _T("interlaced") : _T("all"));
            }
            if (param->bwdif.logPath.length() > 0) {
                tmp << _T(",log=\"") << param->bwdif.logPath << _T("\"");
            } else {
                ADD_BOOL(_T("log"), bwdif.log);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-bwdif ") << tmp.str().substr(1);
        } else if (param->bwdif.enable) {
            cmd << _T(" --vpp-bwdif");
        }
    }
    if (param->rtgmc != defaultPrm->rtgmc) {
        tmp.str(tstring());
        if (!param->rtgmc.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc.enable || save_disabled_prm) {
            ADD_LST(_T("preset"), rtgmc.preset, list_vpp_rtgmc_preset);
            ADD_LST(_T("tuning"), rtgmc.tuning, list_vpp_rtgmc_tuning);
            ADD_BOOL(_T("border"), rtgmc.border);
            ADD_NUM(_T("lossless"), rtgmc.lossless);
            ADD_NUM(_T("input_type"), rtgmc.inputType);
            ADD_FLOAT(_T("prog_sad_mask"), rtgmc.progSADMask, 3);
            ADD_FLOAT(_T("prog_sad_mask_gamma"), rtgmc.progSADMaskGamma, 3);
            ADD_NUM(_T("source_match"), rtgmc.sourceMatch);
            ADD_NUM(_T("match_tr1"), rtgmc.matchTR1);
            ADD_NUM(_T("match_tr2"), rtgmc.matchTR2);
            ADD_LST(_T("match_edi"), rtgmc.matchEdi.mode, list_vpp_rtgmc_edi_mode);
            ADD_FLOAT(_T("match_enhance"), rtgmc.matchEnhance, 3);
            ADD_LST(_T("order"), rtgmc.bob.order, list_vpp_rtgmc_bob_order);
            ADD_NUM(_T("tr0"), rtgmc.searchPrefilter.tr0);
            ADD_NUM(_T("rep0-thin"), rtgmc.searchPrefilter.rep0Thin);
            ADD_NUM(_T("rep0-pad"), rtgmc.searchPrefilter.rep0Pad);
            ADD_NUM(_T("search_refine"), rtgmc.searchPrefilter.searchRefine);
            ADD_BOOL(_T("tv_range"), rtgmc.searchPrefilter.tvRange);
            ADD_BOOL(_T("chroma_motion"), rtgmc.searchPrefilter.chromaMotion);
            ADD_NUM(_T("mv_spatial_refine"), rtgmc.mvSpatialRefine);
            ADD_NUM(_T("noise_process"), rtgmc.noise.noiseProcess);
            ADD_FLOAT(_T("ezdenoise"), rtgmc.noise.ezDenoise, 3);
            ADD_FLOAT(_T("ezkeepgrain"), rtgmc.noise.ezKeepGrain, 3);
            ADD_LST(_T("denoiser"), rtgmc.noise.denoiser, list_vpp_rtgmc_noise_denoiser);
            ADD_LST(_T("noise_deint"), rtgmc.noise.noiseDeint, list_vpp_rtgmc_noise_deint);
            ADD_FLOAT(_T("sigma"), rtgmc.noise.sigma, 3);
            ADD_BOOL(_T("chroma_noise"), rtgmc.noise.chromaNoise);
            ADD_BOOL(_T("denoise_mc"), rtgmc.noise.denoiseMC);
            ADD_NUM(_T("noise_tr"), rtgmc.noise.noiseTR);
            ADD_FLOAT(_T("grain_restore"), rtgmc.noise.grainRestore, 3);
            ADD_FLOAT(_T("noise_restore"), rtgmc.noise.noiseRestore, 3);
            ADD_LST(_T("edi"), rtgmc.edi.mode, list_vpp_rtgmc_edi_mode);
            ADD_NUM(_T("nnsize"), rtgmc.edi.nnsize);
            ADD_NUM(_T("nneurons"), rtgmc.edi.nneurons);
            ADD_NUM(_T("ediqual"), rtgmc.edi.ediqual);
            ADD_LST(_T("chroma_edi"), rtgmc.edi.chromaEdi, list_vpp_rtgmc_chroma_edi_mode);
            ADD_NUM(_T("blksize"), rtgmc.tr1.blksize);
            ADD_NUM(_T("search"), rtgmc.tr1.search);
            ADD_NUM(_T("thsad1"), rtgmc.tr1.thsad);
            ADD_NUM(_T("thsad2"), rtgmc.tr2.thsad);
            ADD_NUM(_T("thscd1"), rtgmc.tr1.thscd1);
            ADD_NUM(_T("thscd2"), rtgmc.tr1.thscd2);
            ADD_NUM(_T("pel"), rtgmc.tr1.pel);
            ADD_NUM(_T("levels"), rtgmc.tr1.levels);
            ADD_NUM(_T("overlap"), rtgmc.tr1.overlap);
            ADD_NUM(_T("lambda"), rtgmc.analyze.lambda);
            ADD_NUM(_T("lsad"), rtgmc.analyze.lsad);
            ADD_NUM(_T("pnew"), rtgmc.analyze.pnew);
            ADD_NUM(_T("plevel"), rtgmc.analyze.plevel);
            ADD_BOOL(_T("globalmotion"), rtgmc.analyze.globalMotion);
            ADD_NUM(_T("searchparam"), rtgmc.analyze.searchParam);
            ADD_NUM(_T("pelsearch"), rtgmc.analyze.pelSearch);
            if (param->rtgmc.analyze.searchEarlySad != defaultPrm->rtgmc.analyze.searchEarlySad) {
                tmp << _T(",search_early_sad=");
                if (param->rtgmc.analyze.searchEarlySad < 0) {
                    tmp << _T("off");
                } else {
                    tmp << param->rtgmc.analyze.searchEarlySad;
                }
            }
            if (param->rtgmc.analyze.spatialEarlySad != defaultPrm->rtgmc.analyze.spatialEarlySad) {
                tmp << _T(",spatial_early_sad=");
                if (param->rtgmc.analyze.spatialEarlySad < 0) {
                    tmp << _T("off");
                } else {
                    tmp << param->rtgmc.analyze.spatialEarlySad;
                }
            }
            ADD_NUM(_T("delta_analyze"), rtgmc.analyze.delta);
            ADD_NUM(_T("delta_tr1"), rtgmc.tr1.delta);
            ADD_NUM(_T("delta_tr2"), rtgmc.tr2.delta);
            ADD_NUM(_T("useflag"), rtgmc.tr2.useFlag);
            ADD_NUM(_T("rep1-thin"), rtgmc.rep1.repThin);
            ADD_NUM(_T("rep1-pad"), rtgmc.rep1.repPad);
            ADD_FLOAT(_T("sharpness"), rtgmc.retouch.sharpness, 3);
            ADD_FLOAT(_T("limit"), rtgmc.retouch.limit, 3);
            ADD_NUM(_T("smode"), rtgmc.retouch.smode);
            ADD_NUM(_T("slmode"), rtgmc.retouch.slmode);
            ADD_NUM(_T("slrad"), rtgmc.retouch.slrad);
            ADD_NUM(_T("sovs"), rtgmc.retouch.sovs);
            ADD_FLOAT(_T("svthin"), rtgmc.retouch.svthin, 3);
            ADD_NUM(_T("sbb"), rtgmc.retouch.sbb);
            ADD_BOOL(_T("precise"), rtgmc.retouch.precise);
            ADD_NUM(_T("rep2-thin"), rtgmc.rep2.repThin);
            ADD_NUM(_T("rep2-pad"), rtgmc.rep2.repPad);
            ADD_BOOL(_T("rep_chroma"), rtgmc.rep1.repChroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc ") << tmp.str().substr(1);
        } else if (param->rtgmc.enable) {
            cmd << _T(" --vpp-rtgmc");
        }
    }
    if (param->kfm != defaultPrm->kfm) {
        tmp.str(tstring());
        if (!param->kfm.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->kfm.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), kfm.mode, list_vpp_kfm_mode);
            ADD_LST(_T("preset"), kfm.preset, list_vpp_rtgmc_preset);
            ADD_LST(_T("timing"), kfm.timing, list_vpp_kfm_timing);
            ADD_NUM(_T("past_cycles"), kfm.pastCycles);
            ADD_FLOAT(_T("thswitch"), kfm.thswitch, 3);
            ADD_BOOL(_T("ucf"), kfm.ucf);
            ADD_BOOL(_T("nr"), kfm.nr);
            ADD_BOOL(_T("is120"), kfm.is120);
            ADD_BOOL(_T("rff"), kfm.rff);
            ADD_BOOL(_T("debug"), kfm.debug);
            ADD_LST(_T("debug_stage"), kfm.debugStage, list_vpp_kfm_debug_stage);
            ADD_PATH(_T("timecode"), kfm.timecode.c_str());
            if (param->kfm.searchEarlySadOverride != defaultPrm->kfm.searchEarlySadOverride) {
                tmp << _T(",search_early_sad=");
                if (param->kfm.searchEarlySadOverride == FILTER_DEFAULT_KFM_SEARCH_EARLY_SAD_OVERRIDE) {
                    tmp << _T("auto");
                } else if (param->kfm.searchEarlySadOverride < 0) {
                    tmp << _T("off");
                } else {
                    tmp << param->kfm.searchEarlySadOverride;
                }
            }
            if (param->kfm.spatialEarlySadOverride != defaultPrm->kfm.spatialEarlySadOverride) {
                tmp << _T(",spatial_early_sad=");
                if (param->kfm.spatialEarlySadOverride == FILTER_DEFAULT_KFM_SPATIAL_EARLY_SAD_OVERRIDE) {
                    tmp << _T("auto");
                } else if (param->kfm.spatialEarlySadOverride < 0) {
                    tmp << _T("off");
                } else {
                    tmp << param->kfm.spatialEarlySadOverride;
                }
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-kfm ") << tmp.str().substr(1);
        } else if (param->kfm.enable) {
            cmd << _T(" --vpp-kfm");
        }
    }
    if (param->rtgmc_bob != defaultPrm->rtgmc_bob) {
        tmp.str(tstring());
        if (!param->rtgmc_bob.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc_bob.enable || save_disabled_prm) {
            ADD_LST(_T("order"), rtgmc_bob.order, list_vpp_rtgmc_bob_order);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-bob ") << tmp.str().substr(1);
        } else if (param->rtgmc_bob.enable) {
            cmd << _T(" --vpp-rtgmc-bob");
        }
    }
    if (param->rtgmc_search_prefilter != defaultPrm->rtgmc_search_prefilter) {
        tmp.str(tstring());
        if (!param->rtgmc_search_prefilter.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc_search_prefilter.enable || save_disabled_prm) {
            ADD_NUM(_T("tr0"), rtgmc_search_prefilter.tr0);
            ADD_NUM(_T("rep0-thin"), rtgmc_search_prefilter.rep0Thin);
            ADD_NUM(_T("rep0-pad"), rtgmc_search_prefilter.rep0Pad);
            ADD_NUM(_T("search_refine"), rtgmc_search_prefilter.searchRefine);
            ADD_BOOL(_T("tv_range"), rtgmc_search_prefilter.tvRange);
            ADD_BOOL(_T("chroma_motion"), rtgmc_search_prefilter.chromaMotion);
            ADD_STR(_T("dump_y4m"), rtgmc_search_prefilter.dumpY4m);
            ADD_STR(_T("dump_stage"), rtgmc_search_prefilter.dumpStage);
            ADD_NUM(_T("dump_max_frames"), rtgmc_search_prefilter.dumpMaxFrames);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-search-prefilter ") << tmp.str().substr(1);
        } else if (param->rtgmc_search_prefilter.enable) {
            cmd << _T(" --vpp-rtgmc-search-prefilter");
        }
    }
    if (param->rtgmc_edi != defaultPrm->rtgmc_edi) {
        tmp.str(tstring());
        if (!param->rtgmc_edi.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc_edi.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), rtgmc_edi.mode, list_vpp_rtgmc_edi_mode);
            ADD_NUM(_T("nnsize"), rtgmc_edi.nnsize);
            ADD_NUM(_T("nneurons"), rtgmc_edi.nneurons);
            ADD_NUM(_T("ediqual"), rtgmc_edi.ediqual);
            ADD_LST(_T("chroma_edi"), rtgmc_edi.chromaEdi, list_vpp_rtgmc_chroma_edi_mode);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-edi ") << tmp.str().substr(1);
        } else if (param->rtgmc_edi.enable) {
            cmd << _T(" --vpp-rtgmc-edi");
        }
    }
    if (param->maa != defaultPrm->maa) {
        tmp.str(tstring());
        if (!param->maa.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->maa.enable || save_disabled_prm) {
            ADD_FLOAT(_T("ss"), maa.ss, 2);
            ADD_NUM(_T("aa"), maa.aa);
            ADD_NUM(_T("aac"), maa.aac);
            ADD_BOOL(_T("mask"), maa.mask);
            ADD_NUM(_T("mthresh"), maa.mthresh);
            ADD_BOOL(_T("chroma"), maa.chroma);
            ADD_NUM(_T("show"), maa.show);
            tmp << _T(",edge=") << param->maa.edge;
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-maa ") << tmp.str().substr(1);
        } else if (param->maa.enable) {
            cmd << _T(" --vpp-maa");
        }
    }
    if (param->decomb != defaultPrm->decomb) {
        tmp.str(tstring());
        if (!param->decomb.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->decomb.enable || save_disabled_prm) {
            ADD_BOOL(_T("full"), decomb.full);
            ADD_NUM(_T("threshold"), decomb.threshold);
            ADD_NUM(_T("dthreshold"), decomb.dthreshold);
            ADD_BOOL(_T("blend"), decomb.blend);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-decomb ") << tmp.str().substr(1);
        } else if (param->decomb.enable) {
            cmd << _T(" --vpp-decomb");
        }
    }
    if (param->ivtc != defaultPrm->ivtc) {
        tmp.str(tstring());
        if (!param->ivtc.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->ivtc.enable || save_disabled_prm) {
            ADD_NUM(_T("guide"), ivtc.guide);
            ADD_NUM(_T("post"), ivtc.post);
            if (param->ivtc.cycle != defaultPrm->ivtc.cycle) {
                tmp << _T(",cycle=") << ((param->ivtc.cycle < 0) ? _T("auto") : std::to_tstring(param->ivtc.cycle));
            }
            ADD_NUM(_T("drop"), ivtc.drop);
            ADD_FLOAT(_T("combthresh"), ivtc.combThresh, 3);
            ADD_FLOAT(_T("cleanfrac"), ivtc.cleanFrac, 3);
            ADD_NUM(_T("dthresh"), ivtc.dthresh);
            ADD_BOOL(_T("chroma"), ivtc.chroma);
            ADD_NUM(_T("back"), ivtc.back);
            ADD_NUM(_T("y0"), ivtc.y0);
            ADD_NUM(_T("y1"), ivtc.y1);
            ADD_NUM(_T("nt"), ivtc.nt);
            ADD_NUM(_T("cthresh"), ivtc.cthresh);
            ADD_NUM(_T("combpel"), ivtc.combPel);
            ADD_FLOAT(_T("scthresh"), ivtc.scThresh, 3);
            if (param->ivtc.cadenceLock != defaultPrm->ivtc.cadenceLock) {
                tmp << _T(",cadlock=") << ((param->ivtc.cadenceLock < 0) ? _T("auto") : (param->ivtc.cadenceLock ? _T("on") : _T("off")));
            }
            ADD_NUM(_T("gthresh"), ivtc.gthresh);
            ADD_NUM(_T("vthresh"), ivtc.vthresh);
            if (param->ivtc.expand != defaultPrm->ivtc.expand) {
                tmp << _T(",expand=") << ((param->ivtc.expand < 0) ? _T("auto") : (param->ivtc.expand ? _T("on") : _T("off")));
            }
            ADD_BOOL(_T("mixed"), ivtc.mixed);
            ADD_FLOAT(_T("hysteresis"), ivtc.hysteresis, 3);
            if (param->ivtc.tff != defaultPrm->ivtc.tff) {
                tmp << _T(",tff=") << ((param->ivtc.tff < 0) ? _T("auto") : (param->ivtc.tff ? _T("on") : _T("off")));
            }
            if (param->ivtc.logPath.length() > 0) {
                tmp << _T(",log=\"") << param->ivtc.logPath << _T("\"");
            } else {
                ADD_BOOL(_T("log"), ivtc.log);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-ivtc ") << tmp.str().substr(1);
        } else if (param->ivtc.enable) {
            cmd << _T(" --vpp-ivtc");
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
            ADD_NUM(_T("keep"), mpdecimate.keep);
            ADD_FLOAT(_T("frac"), mpdecimate.frac, 3);
            ADD_BOOL(_T("log"), mpdecimate.log);
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
            ADD_NUM(_T("d"), knn.d);
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
    if (param->nlmeans != defaultPrm->nlmeans) {
        tmp.str(tstring());
        if (!param->nlmeans.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nlmeans.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sigma"), nlmeans.sigma, 3);
            ADD_NUM(_T("patch"), nlmeans.patchSize);
            ADD_NUM(_T("search"), nlmeans.searchSize);
            ADD_FLOAT(_T("h"), nlmeans.h, 3);
            ADD_NUM(_T("d"), nlmeans.d);
            ADD_NUM(_T("search_t"), nlmeans.searchSizeT);
            ADD_LST(_T("fp16"), nlmeans.fp16, list_vpp_nlmeans_fp16);
            ADD_BOOL(_T("shared_mem"), nlmeans.sharedMem);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nlmeans ") << tmp.str().substr(1);
        } else if (param->nlmeans.enable) {
            cmd << _T(" --vpp-nlmeans");
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
    if (param->bm3d != defaultPrm->bm3d) {
        tmp.str(tstring());
        if (!param->bm3d.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->bm3d.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sigma"), bm3d.sigma, 3);
            ADD_NUM(_T("block_step"), bm3d.block_step);
            ADD_NUM(_T("group_size"), bm3d.group_size);
            ADD_NUM(_T("bm_range"), bm3d.bm_range);
            ADD_NUM(_T("radius"), bm3d.radius);
            ADD_BOOL(_T("chroma"), bm3d.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-bm3d ") << tmp.str().substr(1);
        } else if (param->bm3d.enable) {
            cmd << _T(" --vpp-bm3d");
        }
    }
    if (param->hqdn3d != defaultPrm->hqdn3d) {
        tmp.str(tstring());
        if (!param->hqdn3d.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->hqdn3d.enable || save_disabled_prm) {
            ADD_FLOAT(_T("luma_spatial"), hqdn3d.luma_spatial, 3);
            ADD_FLOAT(_T("chroma_spatial"), hqdn3d.chroma_spatial, 3);
            ADD_FLOAT(_T("luma_temporal"), hqdn3d.luma_temporal, 3);
            ADD_FLOAT(_T("chroma_temporal"), hqdn3d.chroma_temporal, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-hqdn3d ") << tmp.str().substr(1);
        } else if (param->hqdn3d.enable) {
            cmd << _T(" --vpp-hqdn3d");
        }
    }
    if (param->descale != defaultPrm->descale) {
        tmp.str(tstring());
        if (!param->descale.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->descale.enable || save_disabled_prm) {
            ADD_LST(_T("kernel"), descale.kernel, list_vpp_descale_kernel);
            ADD_NUM(_T("width"), descale.width);
            ADD_NUM(_T("height"), descale.height);
            ADD_FLOAT(_T("b"), descale.b, 3);
            ADD_FLOAT(_T("c"), descale.c, 3);
            ADD_FLOAT(_T("src_left"), descale.src_left, 3);
            ADD_FLOAT(_T("src_top"), descale.src_top, 3);
            ADD_FLOAT(_T("src_width"), descale.src_width, 3);
            ADD_FLOAT(_T("src_height"), descale.src_height, 3);
            ADD_LST(_T("border_handling"), descale.border, list_vpp_descale_border);
            ADD_BOOL(_T("auto"), descale.autoDetect);
            ADD_NUM(_T("search_min"), descale.search_min);
            ADD_NUM(_T("search_max"), descale.search_max);
            ADD_NUM(_T("search_step"), descale.search_step);
            ADD_NUM(_T("detect_frames"), descale.detect_frames);
            ADD_BOOL(_T("show_scores"), descale.show_scores);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-descale ") << tmp.str().substr(1);
        } else if (param->descale.enable) {
            cmd << _T(" --vpp-descale");
        }
    }
    if (param->anime4k != defaultPrm->anime4k) {
        tmp.str(tstring());
        if (!param->anime4k.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->anime4k.enable || save_disabled_prm) {
            ADD_LST2(_T("mode"), param->anime4k, defaultPrm->anime4k, mode, list_vpp_anime4k_mode);
            ADD_NUM2(_T("scale"), param->anime4k, defaultPrm->anime4k, scale);
            ADD_FLOAT2(_T("strength"), param->anime4k, defaultPrm->anime4k, strength, 3);
            ADD_LST2(_T("chroma_resize"), param->anime4k, defaultPrm->anime4k, chromaResize, list_vpp_anime4k_chroma_resize);
            ADD_BOOL2(_T("chroma"), param->anime4k, defaultPrm->anime4k, chroma);
            ADD_LST2(_T("darken"), param->anime4k, defaultPrm->anime4k, darken, list_vpp_anime4k_darken);
            ADD_LST2(_T("thin"), param->anime4k, defaultPrm->anime4k, thin, list_vpp_anime4k_thin);
            ADD_LST2(_T("denoise"), param->anime4k, defaultPrm->anime4k, denoise, list_vpp_anime4k_denoise);
            ADD_FLOAT2(_T("denoise_intensity"), param->anime4k, defaultPrm->anime4k, denoiseIntensity, 3);
            ADD_FLOAT2(_T("denoise_spatial"), param->anime4k, defaultPrm->anime4k, denoiseSpatial, 3);
            ADD_FLOAT2(_T("denoise_curve"), param->anime4k, defaultPrm->anime4k, denoiseCurve, 3);
            ADD_FLOAT2(_T("denoise_hist_reg"), param->anime4k, defaultPrm->anime4k, denoiseHistReg, 3);
            ADD_LST2(_T("prefilter_denoise"), param->anime4k, defaultPrm->anime4k, prefilterDenoise, list_vpp_anime4k_denoise);
            ADD_BOOL2(_T("clamp_highlights"), param->anime4k, defaultPrm->anime4k, clampHighlights);
            ADD_FLOAT2(_T("antiring"), param->anime4k, defaultPrm->anime4k, antiring, 2);
            if (param->anime4k.postResizeW != defaultPrm->anime4k.postResizeW
             || param->anime4k.postResizeH != defaultPrm->anime4k.postResizeH) {
                tmp << _T(",out_res=") << param->anime4k.postResizeW << _T("x") << param->anime4k.postResizeH;
            }
            ADD_LST2(_T("resize"), param->anime4k, defaultPrm->anime4k, postResizeAlgo, list_vpp_resize);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-anime4k-shader ") << tmp.str().substr(1);
        } else if (param->anime4k.enable) {
            cmd << _T(" --vpp-anime4k-shader");
        }
    }
    if (param->onnx != defaultPrm->onnx) {
        tmp.str(tstring());
        if (!param->onnx.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->onnx.enable || save_disabled_prm) {
            if (!param->onnx.modelFile.empty()) {
                tmp << _T(",model=") << param->onnx.modelFile;
            }
            tmp << _T(",device=") << param->onnx.device;
#if ENCODER_NVENC
            tmp << _T(",interop=") << param->onnx.interop;
#endif
            tmp << _T(",prec=") << param->onnx.precision;
            tmp << _T(",colormatrix=") << get_cx_desc(list_colormatrix, param->onnx.colormatrix);
            if (param->onnx.colormatrixOut != RGY_MATRIX_AUTO) {
                tmp << _T(",colormatrix_out=") << get_cx_desc(list_colormatrix, param->onnx.colormatrixOut);
            }
            tmp << _T(",colorrange=") << get_cx_desc(list_colorrange, param->onnx.colorrange);
            tmp << _T(",colorspace=") << param->onnx.colorspace;
            tmp << _T(",noise=") << param->onnx.noise;
            if (param->onnx.frames > 1) {
                tmp << _T(",frames=") << param->onnx.frames;
            }
            if (!param->onnx.maskFile.empty()) {
                tmp << _T(",mask=") << param->onnx.maskFile;
            }
            if (param->onnx.postResizeW != 0 && param->onnx.postResizeH != 0) {
                tmp << _T(",out_res=") << param->onnx.postResizeW << _T("x") << param->onnx.postResizeH;
                tmp << _T(",resize=") << get_cx_desc(list_vpp_resize, param->onnx.postResizeAlgo);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-onnx ") << tmp.str().substr(1);
        } else if (param->onnx.enable) {
            cmd << _T(" --vpp-onnx");
        }
    }
    if (!param->onnxModelDir.empty()) {
        cmd << _T(" --vpp-onnx-model-dir ") << param->onnxModelDir;
    }
    if (param->rife_ov != defaultPrm->rife_ov) {
        tmp.str(tstring());
        if (!param->rife_ov.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rife_ov.enable || save_disabled_prm) {
            if (!param->rife_ov.modelFile.empty()) tmp << _T(",model=") << param->rife_ov.modelFile;
            tmp << _T(",device=") << param->rife_ov.device;
            if (param->rife_ov.fps.is_valid() && param->rife_ov.fps.n() > 0) {
                tmp << _T(",fps=") << param->rife_ov.fps.n() << _T("/") << param->rife_ov.fps.d();
            } else {
                tmp << _T(",multi=") << param->rife_ov.multi;
            }
            tmp << _T(",colormatrix=") << param->rife_ov.colormatrix;
            tmp << _T(",colorrange=") << param->rife_ov.colorrange;
        }
        cmd << _T(" --vpp-rife-ov ") << tmp.str().substr(1);
    }
    if (param->onnxDeint != defaultPrm->onnxDeint) {
        const auto option = param->onnxDeint.operation == VppOnnxDeintOperation::VhsRestore ? _T(" --vpp-vhs-restore") : _T(" --vpp-onnx-deint");
        tmp.str(tstring());
        if (!param->onnxDeint.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->onnxDeint.enable || save_disabled_prm) {
            if (!param->onnxDeint.modelFile.empty()) tmp << _T(",model=") << param->onnxDeint.modelFile;
            tmp << _T(",device=") << param->onnxDeint.device;
            tmp << _T(",precision=") << param->onnxDeint.precision;
            tmp << _T(",mode=") << get_cx_desc(list_vpp_onnx_deint_mode, (int)param->onnxDeint.mode);
            tmp << _T(",colormatrix=") << get_cx_desc(list_colormatrix, param->onnxDeint.colormatrix);
            tmp << _T(",colorrange=") << get_cx_desc(list_colorrange, param->onnxDeint.colorrange);
        }
        if (!tmp.str().empty()) {
            cmd << option << _T(" ") << tmp.str().substr(1);
        } else if (param->onnxDeint.enable) {
            cmd << option;
        }
    }
    if (!param->onnx.cacheDir.empty()) {
        cmd << _T(" --vpp-onnx-cache-dir ") << param->onnx.cacheDir;
    }
    if (param->dct != defaultPrm->dct) {
        tmp.str(tstring());
        if (!param->dct.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->dct.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sigma"), dct.sigma, 3);
            ADD_FLOAT(_T("sigma2"), dct.sigma2, 3);
            ADD_FLOAT(_T("sigma3"), dct.sigma3, 3);
            ADD_FLOAT(_T("sigma4"), dct.sigma4, 3);
            ADD_LST(_T("step"), dct.step, list_vpp_denoise_dct_step);
            ADD_LST(_T("block_size"), dct.block_size, list_vpp_denoise_dct_block_size);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-denoise-dct ") << tmp.str().substr(1);
        } else if (param->dct.enable) {
            cmd << _T(" --vpp-denoise-dct");
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
    if (param->fft3d != defaultPrm->fft3d) {
        tmp.str(tstring());
        if (!param->fft3d.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->fft3d.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sigma"), fft3d.sigma, 3);
            ADD_FLOAT(_T("sigma2"), fft3d.sigma2, 3);
            ADD_FLOAT(_T("sigma3"), fft3d.sigma3, 3);
            ADD_FLOAT(_T("sigma4"), fft3d.sigma4, 3);
            ADD_FLOAT(_T("amount"), fft3d.amount, 3);
            ADD_NUM(_T("block_size"), fft3d.block_size);
            ADD_FLOAT(_T("overlap"), fft3d.overlap, 3);
            ADD_FLOAT(_T("overlap2"), fft3d.overlap2, 3);
            ADD_NUM(_T("method"), fft3d.method);
            ADD_NUM(_T("temporal"), fft3d.temporal);
            ADD_NUM(_T("bt"), fft3d.bt);
            ADD_FLOAT(_T("sharpen"), fft3d.sharpen, 3);
            ADD_FLOAT(_T("scutoff"), fft3d.scutoff, 3);
            ADD_FLOAT(_T("svr"), fft3d.svr, 3);
            ADD_FLOAT(_T("smin"), fft3d.smin, 3);
            ADD_FLOAT(_T("smax"), fft3d.smax, 3);
            ADD_FLOAT(_T("degrid"), fft3d.degrid, 3);
            ADD_BOOL(_T("signorm"), fft3d.signorm);
            ADD_LST(_T("prec"), fft3d.precision, list_vpp_fp_prec);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-fft3d ") << tmp.str().substr(1);
        } else if (param->fft3d.enable) {
            cmd << _T(" --vpp-fft3d");
        }
    }
    const auto add_degrain_cmd = [&](const VppDegrain& degrain, const VppDegrain& defaultDegrain) {
        tmp.str(tstring());
        if (!degrain.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (degrain.enable || save_disabled_prm) {
            if (degrain.preset != defaultDegrain.preset) tmp << _T(",preset=") << get_chr_from_value(list_vpp_degrain_preset, (int)degrain.preset);
            if (degrain.mode != defaultDegrain.mode) tmp << _T(",mode=") << get_chr_from_value(list_vpp_degrain_mode, (int)degrain.mode);
            if (degrain.stage != defaultDegrain.stage) tmp << _T(",stage=") << get_chr_from_value(list_vpp_degrain_stage, (int)degrain.stage);
            if (degrain.blksize != defaultDegrain.blksize) tmp << _T(",blksize=") << degrain.blksize;
            if (degrain.search != defaultDegrain.search) tmp << _T(",search=") << degrain.search;
            if (degrain.thsad != defaultDegrain.thsad) tmp << _T(",thsad=") << degrain.thsad;
            if (degrain.thsadc != defaultDegrain.thsadc) tmp << _T(",thsadc=") << degrain.thsadc;
            if (degrain.thscd1 != defaultDegrain.thscd1) tmp << _T(",thscd1=") << degrain.thscd1;
            if (degrain.thscd2 != defaultDegrain.thscd2) tmp << _T(",thscd2=") << degrain.thscd2;
            if (degrain.pel != defaultDegrain.pel) tmp << _T(",pel=") << degrain.pel;
            if (degrain.subpelInterp != defaultDegrain.subpelInterp) tmp << _T(",subpelinterp=") << degrain.subpelInterp;
            if (degrain.levels != defaultDegrain.levels) tmp << _T(",levels=") << degrain.levels;
            if (degrain.overlap != defaultDegrain.overlap) tmp << _T(",overlap=") << degrain.overlap;
            if (degrain.delta != defaultDegrain.delta) tmp << _T(",delta=") << degrain.delta;
            if (degrain.tr0 != defaultDegrain.tr0) tmp << _T(",tr0=") << degrain.tr0;
            if (degrain.rep0 != defaultDegrain.rep0) tmp << _T(",rep0=") << degrain.rep0;
            if (degrain.searchRefine != defaultDegrain.searchRefine) tmp << _T(",search_refine=") << degrain.searchRefine;
            if (degrain.searchParam != defaultDegrain.searchParam) tmp << _T(",searchparam=") << degrain.searchParam;
            if (degrain.pelSearch != defaultDegrain.pelSearch) tmp << _T(",pelsearch=") << degrain.pelSearch;
            if (degrain.searchEarlySad != defaultDegrain.searchEarlySad) {
                tmp << _T(",search_early_sad=");
                if (degrain.searchEarlySad < 0) {
                    tmp << _T("off");
                } else {
                    tmp << degrain.searchEarlySad;
                }
            }
            if (degrain.spatialEarlySad != defaultDegrain.spatialEarlySad) {
                tmp << _T(",spatial_early_sad=");
                if (degrain.spatialEarlySad < 0) {
                    tmp << _T("off");
                } else {
                    tmp << degrain.spatialEarlySad;
                }
            }
            if (degrain.trueMotion != defaultDegrain.trueMotion) tmp << _T(",truemotion=") << (degrain.trueMotion ? _T("true") : _T("false"));
            if (degrain.lambda != defaultDegrain.lambda) tmp << _T(",lambda=") << degrain.lambda;
            if (degrain.lsad != defaultDegrain.lsad) tmp << _T(",lsad=") << degrain.lsad;
            if (degrain.pnew != defaultDegrain.pnew) tmp << _T(",pnew=") << degrain.pnew;
            if (degrain.plevel != defaultDegrain.plevel) tmp << _T(",plevel=") << degrain.plevel;
            if (degrain.globalMotion != defaultDegrain.globalMotion) tmp << _T(",globalmotion=") << (degrain.globalMotion ? _T("true") : _T("false"));
            if (degrain.dct != defaultDegrain.dct) tmp << _T(",dct=") << degrain.dct;
            if (degrain.useFlag != defaultDegrain.useFlag) tmp << _T(",useflag=") << degrain.useFlag;
            if (degrain.mvSpatialRefine != defaultDegrain.mvSpatialRefine) tmp << _T(",mv_spatial_refine=") << degrain.mvSpatialRefine;
            if (degrain.chroma != defaultDegrain.chroma) tmp << _T(",chroma=") << (degrain.chroma ? _T("true") : _T("false"));
            if (degrain.binomial != defaultDegrain.binomial) {
                tmp << _T(",binomial=") << (degrain.binomial < 0 ? _T("auto") : (degrain.binomial ? _T("true") : _T("false")));
            }
            if (degrain.tvRange != defaultDegrain.tvRange) tmp << _T(",tv_range=") << (degrain.tvRange ? _T("true") : _T("false"));
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-degrain ") << tmp.str().substr(1);
        } else if (degrain.enable) {
            cmd << _T(" --vpp-degrain");
        }
    };
    if (param->degrain != defaultPrm->degrain) {
        add_degrain_cmd(param->degrain, defaultPrm->degrain);
    }
    if (param->degrainAnalyze != defaultPrm->degrainAnalyze) {
        add_degrain_cmd(param->degrainAnalyze, defaultPrm->degrainAnalyze);
    }
    if (param->degrainTR1 != defaultPrm->degrainTR1) {
        add_degrain_cmd(param->degrainTR1, defaultPrm->degrainTR1);
    }
    if (param->degrainTR2 != defaultPrm->degrainTR2) {
        add_degrain_cmd(param->degrainTR2, defaultPrm->degrainTR2);
    }
    if (param->rtgmc_retouch != defaultPrm->rtgmc_retouch) {
        tmp.str(tstring());
        if (!param->rtgmc_retouch.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc_retouch.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sharpness"), rtgmc_retouch.sharpness, 3);
            ADD_FLOAT(_T("limit"), rtgmc_retouch.limit, 3);
            ADD_NUM(_T("smode"), rtgmc_retouch.smode);
            ADD_NUM(_T("slmode"), rtgmc_retouch.slmode);
            ADD_NUM(_T("slrad"), rtgmc_retouch.slrad);
            ADD_NUM(_T("sovs"), rtgmc_retouch.sovs);
            ADD_FLOAT(_T("svthin"), rtgmc_retouch.svthin, 3);
            ADD_NUM(_T("sbb"), rtgmc_retouch.sbb);
            ADD_BOOL(_T("precise"), rtgmc_retouch.precise);
            ADD_NUM(_T("tr1"), rtgmc_retouch.tr1);
            ADD_NUM(_T("tr2"), rtgmc_retouch.tr2);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-retouch ") << tmp.str().substr(1);
        } else if (param->rtgmc_retouch.enable) {
            cmd << _T(" --vpp-rtgmc-retouch");
        }
    }
    // stage の指定によってパース時の格納先が Rep1 / Rep2 に振り分けられるため、生成時も格納先ごとに出力する
    // (格納先を復元できるよう、Rep1 / Rep2 側は stage を必ず出力する)
    auto add_rtgmc_shimmer_repair_cmd = [&](const VppRtgmcShimmerRepair& prm, const VppRtgmcShimmerRepair& prmDefault, const bool addStage) {
        if (prm == prmDefault) {
            return;
        }
        tmp.str(tstring());
        if (!prm.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (prm.enable || save_disabled_prm) {
            if (addStage) {
                tmp << _T(",stage=") << get_chr_from_value(list_vpp_rtgmc_shimmer_repair_stage, (int)prm.stage);
            }
            if (prm.repThin != prmDefault.repThin) tmp << _T(",rep-thin=") << prm.repThin;
            if (prm.repPad != prmDefault.repPad) tmp << _T(",rep-pad=") << prm.repPad;
            if (prm.repChroma != prmDefault.repChroma) tmp << _T(",rep_chroma=") << (prm.repChroma ? _T("true") : _T("false"));
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-shimmer-repair ") << tmp.str().substr(1);
        } else if (prm.enable) {
            cmd << _T(" --vpp-rtgmc-shimmer-repair");
        }
    };
    // 値を指定せずに --vpp-rtgmc-shimmer-repair のみを指定した場合の格納先。stageは出力しない
    add_rtgmc_shimmer_repair_cmd(param->rtgmc_shimmer_repair, defaultPrm->rtgmc_shimmer_repair, false);
    add_rtgmc_shimmer_repair_cmd(param->rtgmc_shimmer_repairRep1, defaultPrm->rtgmc_shimmer_repairRep1, true);
    add_rtgmc_shimmer_repair_cmd(param->rtgmc_shimmer_repairRep2, defaultPrm->rtgmc_shimmer_repairRep2, true);
    if (param->rtgmc_primitive != defaultPrm->rtgmc_primitive) {
        tmp.str(tstring());
        if (!param->rtgmc_primitive.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->rtgmc_primitive.enable || save_disabled_prm) {
            ADD_LST(_T("op"), rtgmc_primitive.op, list_vpp_rtgmc_primitive_op);
            ADD_LST(_T("ref"), rtgmc_primitive.ref, list_vpp_rtgmc_primitive_ref);
            ADD_NUM(_T("mode"), rtgmc_primitive.mode);
            ADD_FLOAT(_T("weight"), rtgmc_primitive.weight, 3);
            ADD_BOOL(_T("chroma"), rtgmc_primitive.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-rtgmc-primitive ") << tmp.str().substr(1);
        } else if (param->rtgmc_primitive.enable) {
            cmd << _T(" --vpp-rtgmc-primitive");
        }
    }
    if (param->msmooth != defaultPrm->msmooth) {
        tmp.str(tstring());
        if (!param->msmooth.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->msmooth.enable || save_disabled_prm) {
            ADD_NUM(_T("strength"), msmooth.strength);
            ADD_FLOAT(_T("threshold"), msmooth.threshold, 3);
            ADD_FLOAT(_T("threshold_c"), msmooth.threshold_c, 3);
            ADD_BOOL(_T("highq"), msmooth.highq);
            ADD_BOOL(_T("mask"), msmooth.mask);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-msmooth ") << tmp.str().substr(1);
        } else if (param->msmooth.enable) {
            cmd << _T(" --vpp-msmooth");
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
    for (size_t i = 0; i < param->libplacebo_shader.size(); i++) {
        const auto shaderDefault = VppLibplaceboShader();
        if (param->libplacebo_shader[i] != shaderDefault) {
            tmp.str(tstring());
            if (!param->libplacebo_shader[i].enable && save_disabled_prm) {
                tmp << _T(",enable=false");
            }
            if (param->libplacebo_shader[i].enable || save_disabled_prm) {
                ADD_PATH2(_T("shader"), param->libplacebo_shader[i], shader.c_str());
                if (param->libplacebo_shader[i].width > 0 && param->libplacebo_shader[i].height > 0) {
                    tmp << _T(",res=") << param->libplacebo_shader[i].width << _T("x") << param->libplacebo_shader[i].height;
                }
                ADD_LST2(_T("csp"), param->libplacebo_shader[i], shaderDefault, csp, list_vpp_libplacebo_shader_csp);
                ADD_LST2(_T("chromaloc"), param->libplacebo_shader[i], shaderDefault, chromaloc, list_chromaloc_str);
                ADD_LST2(_T("colorsystem"), param->libplacebo_shader[i], shaderDefault, colorsystem, list_vpp_libplacebo_colorsystem);
                ADD_LST2(_T("transfer"), param->libplacebo_shader[i], shaderDefault, transfer, list_vpp_libplacebo_tone_mapping_transfer);
                ADD_LST2(_T("filter"), param->libplacebo_shader[i], shaderDefault, resize_algo, list_vpp_resize);
                ADD_FLOAT2(_T("radius"), param->libplacebo_shader[i], shaderDefault, radius, 3);
                ADD_FLOAT2(_T("clamp"), param->libplacebo_shader[i], shaderDefault, clamp_, 3);
                ADD_FLOAT2(_T("taper"), param->libplacebo_shader[i], shaderDefault, taper, 3);
                ADD_FLOAT2(_T("blur"), param->libplacebo_shader[i], shaderDefault, blur, 3);
                ADD_FLOAT2(_T("antiring"), param->libplacebo_shader[i], shaderDefault, antiring, 3);
                ADD_BOOL2(_T("linear"), param->libplacebo_shader[i], shaderDefault, linear);
                ADD_BOOL2(_T("sigmoid"), param->libplacebo_shader[i], shaderDefault, sigmoid);
                if (param->libplacebo_shader[i].sigmoid_center != shaderDefault.sigmoid_center
                    && param->libplacebo_shader[i].sigmoid_center) {
                    tmp << _T(",sigmoid_center=") << std::setprecision(3) << *param->libplacebo_shader[i].sigmoid_center;
                }
                if (param->libplacebo_shader[i].sigmoid_slope != shaderDefault.sigmoid_slope
                    && param->libplacebo_shader[i].sigmoid_slope) {
                    tmp << _T(",sigmoid_slope=") << std::setprecision(3) << *param->libplacebo_shader[i].sigmoid_slope;
                }
                for (const auto& custom : param->libplacebo_shader[i].custom_params) {
                    tmp << _T(",custom=") << custom.first << _T("=") << custom.second;
                }
            }
            if (!tmp.str().empty()) {
                cmd << _T(" --vpp-libplacebo-shader \"") << tmp.str().substr(1) << _T("\"");
            } else if (param->libplacebo_shader[i].enable) {
                cmd << _T(" --vpp-libplacebo-shader");
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
    if (param->vinverse != defaultPrm->vinverse) {
        tmp.str(tstring());
        if (!param->vinverse.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->vinverse.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), vinverse.mode, list_vpp_vinverse_mode);
            ADD_FLOAT(_T("sstr"), vinverse.sstr, 3);
            ADD_FLOAT(_T("amnt"), vinverse.amnt, 3);
            ADD_FLOAT(_T("scl"), vinverse.scl, 3);
            ADD_FLOAT(_T("thr"), vinverse.thr, 3);
            ADD_BOOL(_T("chroma"), vinverse.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-vinverse ") << tmp.str().substr(1);
        } else if (param->vinverse.enable) {
            cmd << _T(" --vpp-vinverse");
        }
    }
    if (param->chromashift != defaultPrm->chromashift) {
        tmp.str(tstring());
        if (!param->chromashift.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->chromashift.enable || save_disabled_prm) {
            ADD_FLOAT(_T("x"), chromashift.x, 3);
            ADD_FLOAT(_T("y"), chromashift.y, 3);
            ADD_NUM(_T("show"), chromashift.show);
            ADD_BOOL(_T("auto"), chromashift.auto_detect);
            ADD_NUM(_T("auto_frames"), chromashift.auto_frames);
            ADD_NUM(_T("auto_min_pairs"), chromashift.auto_min_pairs);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-chromashift ") << tmp.str().substr(1);
        } else if (param->chromashift.enable) {
            cmd << _T(" --vpp-chromashift");
        }
    }
    if (param->deblock != defaultPrm->deblock) {
        tmp.str(tstring());
        if (!param->deblock.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->deblock.enable || save_disabled_prm) {
            ADD_NUM(_T("qp"), deblock.qp);
            ADD_NUM(_T("alpha"), deblock.alpha);
            ADD_NUM(_T("beta"), deblock.beta);
            ADD_BOOL(_T("chroma"), deblock.chroma);
            ADD_NUM(_T("grid"), deblock.grid);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-deblock ") << tmp.str().substr(1);
        } else if (param->deblock.enable) {
            cmd << _T(" --vpp-deblock");
        }
    }
    if (param->deflicker != defaultPrm->deflicker) {
        tmp.str(tstring());
        if (!param->deflicker.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->deflicker.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), deflicker.strength, 3);
            ADD_FLOAT(_T("damping"), deflicker.damping, 3);
            ADD_FLOAT(_T("scene_threshold"), deflicker.scene_threshold, 3);
            ADD_NUM(_T("frames"), deflicker.frames);
            ADD_BOOL(_T("predictor"), deflicker.predictor);
            ADD_BOOL(_T("chroma"), deflicker.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-deflicker ") << tmp.str().substr(1);
        } else if (param->deflicker.enable) {
            cmd << _T(" --vpp-deflicker");
        }
    }
    if (param->stab != defaultPrm->stab) {
        tmp.str(tstring());
        if (!param->stab.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->stab.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), stab.strength, 3);
            ADD_FLOAT(_T("damping"), stab.damping, 3);
            ADD_FLOAT(_T("trust"), stab.trust_threshold, 3);
            ADD_FLOAT(_T("max_shift"), stab.max_shift, 1);
            const TCHAR *borderStr = _T("black");
            switch (param->stab.border) {
            case VPP_STAB_BORDER_CLAMP:  borderStr = _T("clamp");  break;
            case VPP_STAB_BORDER_MIRROR: borderStr = _T("mirror"); break;
            case VPP_STAB_BORDER_BLACK:
            default: break;
            }
            tmp << _T(",border=") << borderStr;
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-stab ") << tmp.str().substr(1);
        } else if (param->stab.enable) {
            cmd << _T(" --vpp-stab");
        }
    }
    if (param->colorfix != defaultPrm->colorfix) {
        tmp.str(tstring());
        if (!param->colorfix.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->colorfix.enable || save_disabled_prm) {
            const auto modeStr = (param->colorfix.mode == VPP_COLORFIX_MODE_AUTO) ? _T("auto")
                : (param->colorfix.mode == VPP_COLORFIX_MODE_GRAY) ? _T("gray") : _T("manual");
            const auto spaceStr = (param->colorfix.space == VPP_COLORFIX_SPACE_RGB) ? _T("rgb")
                : (param->colorfix.space == VPP_COLORFIX_SPACE_YUV) ? _T("yuv") : _T("auto");
            const auto matrixStr = (param->colorfix.matrix == VPP_COLORFIX_MATRIX_BT601) ? _T("bt601")
                : (param->colorfix.matrix == VPP_COLORFIX_MATRIX_BT709) ? _T("bt709")
                : (param->colorfix.matrix == VPP_COLORFIX_MATRIX_BT2020) ? _T("bt2020") : _T("auto");
            tmp << _T(",mode=") << modeStr;
            tmp << _T(",space=") << spaceStr;
            tmp << _T(",matrix=") << matrixStr;
            tmp << strsprintf(_T(",white=%02x%02x%02x"), param->colorfix.whiteR, param->colorfix.whiteG, param->colorfix.whiteB);
            tmp << strsprintf(_T(",black=%02x%02x%02x"), param->colorfix.blackR, param->colorfix.blackG, param->colorfix.blackB);
            ADD_NUM(_T("frames"), colorfix.frames);
            ADD_FLOAT(_T("strength"), colorfix.strength, 3);
            ADD_FLOAT(_T("variance_threshold"), colorfix.varianceThreshold, 3);
            ADD_NUM(_T("temperature"), colorfix.temperature);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-colorfix ") << tmp.str().substr(1);
        } else if (param->colorfix.enable) {
            cmd << _T(" --vpp-colorfix");
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
    if (param->dehalo != defaultPrm->dehalo) {
        tmp.str(tstring());
        if (!param->dehalo.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->dehalo.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), dehalo.mode, list_vpp_dehalo_mode);
            ADD_FLOAT(_T("rx"), dehalo.rx, 3);
            ADD_FLOAT(_T("ry"), dehalo.ry, 3);
            ADD_FLOAT(_T("darkstr"), dehalo.darkstr, 3);
            ADD_FLOAT(_T("brightstr"), dehalo.brightstr, 3);
            ADD_NUM(_T("lowsens"), dehalo.lowsens);
            ADD_NUM(_T("highsens"), dehalo.highsens);
            ADD_FLOAT(_T("ss"), dehalo.ss, 3);
            ADD_NUM(_T("search_rade"), dehalo.searchRade);
            ADD_NUM(_T("search_radi"), dehalo.searchRadi);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-dehalo ") << tmp.str().substr(1);
        } else if (param->dehalo.enable) {
            cmd << _T(" --vpp-dehalo");
        }
    }
    if (param->finedehalo != defaultPrm->finedehalo) {
        tmp.str(tstring());
        if (!param->finedehalo.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->finedehalo.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), finedehalo.mode, list_vpp_dehalo_mode);
            ADD_FLOAT(_T("rx"), finedehalo.rx, 3);
            ADD_FLOAT(_T("ry"), finedehalo.ry, 3);
            ADD_FLOAT(_T("darkstr"), finedehalo.darkstr, 3);
            ADD_FLOAT(_T("brightstr"), finedehalo.brightstr, 3);
            ADD_NUM(_T("lowsens"), finedehalo.lowsens);
            ADD_NUM(_T("highsens"), finedehalo.highsens);
            ADD_FLOAT(_T("ss"), finedehalo.ss, 3);
            ADD_NUM(_T("search_rade"), finedehalo.searchRade);
            ADD_NUM(_T("search_radi"), finedehalo.searchRadi);
            ADD_NUM(_T("thmi"), finedehalo.thmi);
            ADD_NUM(_T("thma"), finedehalo.thma);
            ADD_NUM(_T("thlimi"), finedehalo.thlimi);
            ADD_NUM(_T("thlima"), finedehalo.thlima);
            ADD_NUM(_T("showmask"), finedehalo.showmask);
            ADD_BOOL(_T("excl"), finedehalo.excl);
            ADD_FLOAT(_T("edgeproc"), finedehalo.edgeproc, 3);
            if (param->finedehalo.edge != defaultPrm->finedehalo.edge) {
                tmp << _T(",edge=") << param->finedehalo.edge.c_str();
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-finedehalo ") << tmp.str().substr(1);
        } else if (param->finedehalo.enable) {
            cmd << _T(" --vpp-finedehalo");
        }
    }
    if (param->guidedfilter != defaultPrm->guidedfilter) {
        tmp.str(tstring());
        if (!param->guidedfilter.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->guidedfilter.enable || save_disabled_prm) {
            ADD_NUM(_T("radius"), guidedfilter.radius);
            ADD_FLOAT(_T("eps"), guidedfilter.eps, 4);
            ADD_BOOL(_T("chroma"), guidedfilter.chroma);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-guidedfilter ") << tmp.str().substr(1);
        } else if (param->guidedfilter.enable) {
            cmd << _T(" --vpp-guidedfilter");
        }
    }
    if (param->clahe != defaultPrm->clahe) {
        tmp.str(tstring());
        if (!param->clahe.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->clahe.enable || save_disabled_prm) {
            ADD_NUM(_T("tiles_x"), clahe.tiles_x);
            ADD_NUM(_T("tiles_y"), clahe.tiles_y);
            ADD_FLOAT(_T("slope"), clahe.slope, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-clahe ") << tmp.str().substr(1);
        } else if (param->clahe.enable) {
            cmd << _T(" --vpp-clahe");
        }
    }
    if (param->dehaze != defaultPrm->dehaze) {
        tmp.str(tstring());
        if (!param->dehaze.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->dehaze.enable || save_disabled_prm) {
            ADD_NUM(_T("patch_radius"), dehaze.patch_radius);
            ADD_FLOAT(_T("omega"), dehaze.omega, 6);
            ADD_FLOAT(_T("t_floor"), dehaze.t_floor, 6);
            ADD_FLOAT(_T("atm_light"), dehaze.atm_light, 6);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-dehaze ") << tmp.str().substr(1);
        } else if (param->dehaze.enable) {
            cmd << _T(" --vpp-dehaze");
        }
    }
    if (param->dering != defaultPrm->dering) {
        tmp.str(tstring());
        if (!param->dering.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->dering.enable || save_disabled_prm) {
            ADD_NUM(_T("mrad"), dering.mrad);
            ADD_NUM(_T("mthr"), dering.mthr);
            ADD_FLOAT(_T("sigma"), dering.sigma, 3);
            ADD_NUM(_T("thr"), dering.thr);
            ADD_FLOAT(_T("elast"), dering.elast, 2);
            ADD_NUM(_T("darkthr"), dering.darkthr);
            ADD_NUM(_T("minp"), dering.minp);
            ADD_NUM(_T("msmooth"), dering.msmooth);
            ADD_NUM(_T("drrep"), dering.drrep);
            ADD_NUM(_T("sharp"), dering.sharp);
            if (param->dering.planes != defaultPrm->dering.planes) {
                tstring p;
                if (param->dering.planes[0]) p += _T(":y");
                if (param->dering.planes[1]) p += _T(":u");
                if (param->dering.planes[2]) p += _T(":v");
                tmp << _T(",planes=") << ((p.length() > 0) ? p.substr(1) : _T(""));
            }
            ADD_BOOL(_T("showmask"), dering.showmask);
            ADD_BOOL(_T("protect"), dering.protect);
            if (param->dering.edge != defaultPrm->dering.edge) {
                tmp << _T(",edge=") << param->dering.edge.c_str();
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-hqdering ") << tmp.str().substr(1);
        } else if (param->dering.enable) {
            cmd << _T(" --vpp-hqdering");
        }
    }
    if (param->msharpen != defaultPrm->msharpen) {
        tmp.str(tstring());
        if (!param->msharpen.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->msharpen.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), msharpen.strength, 3);
            ADD_FLOAT(_T("threshold"), msharpen.threshold, 3);
            ADD_FLOAT(_T("slope"), msharpen.slope, 3);
            ADD_FLOAT(_T("luma_limit"), msharpen.luma_limit, 3);
            ADD_FLOAT(_T("block_protect"), msharpen.block_protect, 3);
            ADD_BOOL(_T("highq"), msharpen.highq);
            ADD_BOOL(_T("mask"), msharpen.mask);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-msharpen ") << tmp.str().substr(1);
        } else if (param->msharpen.enable) {
            cmd << _T(" --vpp-msharpen");
        }
    }
    if (param->cas != defaultPrm->cas) {
        tmp.str(tstring());
        if (!param->cas.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->cas.enable || save_disabled_prm) {
            ADD_FLOAT(_T("sharpness"), cas.sharpness, 3);
            ADD_BOOL(_T("chroma"), cas.chroma);
            ADD_BOOL(_T("hdr"), cas.hdr);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-cas ") << tmp.str().substr(1);
        } else if (param->cas.enable) {
            cmd << _T(" --vpp-cas");
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
            ADD_FLOAT(_T("depth_min"), warpsharp.depth_min, 3);
            ADD_FLOAT(_T("depth_max"), warpsharp.depth_max, 3);
            ADD_FLOAT(_T("edge_thr"), warpsharp.edge_thr, 3);
            ADD_FLOAT(_T("gamma"), warpsharp.gamma, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-warpsharp ") << tmp.str().substr(1);
        } else if (param->warpsharp.enable) {
            cmd << _T(" --vpp-warpsharp");
        }
    }
    if (param->detailsharpen != defaultPrm->detailsharpen) {
        tmp.str(tstring());
        if (!param->detailsharpen.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->detailsharpen.enable || save_disabled_prm) {
            ADD_FLOAT(_T("z"), detailsharpen.z, 3);
            ADD_FLOAT(_T("sstr"), detailsharpen.sstr, 3);
            ADD_FLOAT(_T("power"), detailsharpen.power, 3);
            ADD_FLOAT(_T("ldmp"), detailsharpen.ldmp, 3);
            ADD_NUM(_T("mode"), detailsharpen.mode);
            ADD_BOOL(_T("med"), detailsharpen.med);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-detailsharpen ") << tmp.str().substr(1);
        } else if (param->detailsharpen.enable) {
            cmd << _T(" --vpp-detailsharpen");
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
            ADD_LST(_T("interp"), curves.interp, list_vpp_curves_interp);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-curves ") << tmp.str().substr(1);
        } else if (param->curves.enable) {
            cmd << _T(" --vpp-curves");
        }
    }
    if (param->softlight != defaultPrm->softlight) {
        tmp.str(tstring());
        if (!param->softlight.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->softlight.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), softlight.mode, list_vpp_softlight_mode);
            ADD_LST(_T("formula"), softlight.formula, list_vpp_softlight_formula);
            ADD_BOOL(_T("skipblack"), softlight.skipblack);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-softlight ") << tmp.str().substr(1);
        } else if (param->softlight.enable) {
            cmd << _T(" --vpp-softlight");
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
            ADD_FLOAT(_T("vibrance"), tweak.vibrance, 3);
            ADD_FLOAT(_T("hue"), tweak.hue, 3);
            ADD_BOOL(_T("coring"), tweak.coring);
            ADD_FLOAT(_T("start_hue"), tweak.startHue, 3);
            ADD_FLOAT(_T("end_hue"), tweak.endHue, 3);
            ADD_BOOL(_T("swapuv"), tweak.swapuv);
            ADD_FLOAT(_T("y_offset"),  tweak.y.offset, 3);
            ADD_FLOAT(_T("y_gain"),    tweak.y.gain, 3);
            //ADD_FLOAT(_T("y_gamma"),   tweak.y.gamma, 3);
            ADD_FLOAT(_T("cb_offset"), tweak.cb.offset, 3);
            ADD_FLOAT(_T("cb_gain"),   tweak.cb.gain, 3);
            //ADD_FLOAT(_T("cb_gamma"),  tweak.cb.gamma, 3);
            ADD_FLOAT(_T("cr_offset"), tweak.cr.offset, 3);
            ADD_FLOAT(_T("cr_gain"),   tweak.cr.gain, 3);
            //ADD_FLOAT(_T("cr_gamma"),  tweak.cr.gamma, 3);
            ADD_FLOAT(_T("r_offset"),  tweak.r.offset, 3);
            ADD_FLOAT(_T("r_gain"),    tweak.r.gain, 3);
            ADD_FLOAT(_T("r_gamma"),   tweak.r.gamma, 3);
            ADD_FLOAT(_T("g_offset"),  tweak.g.offset, 3);
            ADD_FLOAT(_T("g_gain"),    tweak.g.gain, 3);
            ADD_FLOAT(_T("g_gamma"),   tweak.g.gamma, 3);
            ADD_FLOAT(_T("b_offset"),  tweak.b.offset, 3);
            ADD_FLOAT(_T("b_gain"),    tweak.b.gain, 3);
            ADD_FLOAT(_T("b_gamma"),   tweak.b.gamma, 3);
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
    if (param->v360 != defaultPrm->v360) {
        tmp.str(tstring());
        if (!param->v360.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->v360.enable || save_disabled_prm) {
            ADD_LST(_T("in"), v360.in_proj, list_vpp_v360_proj);
            ADD_LST(_T("out"), v360.out_proj, list_vpp_v360_proj);
            ADD_FLOAT(_T("yaw"), v360.yaw, 3);
            ADD_FLOAT(_T("pitch"), v360.pitch, 3);
            ADD_FLOAT(_T("roll"), v360.roll, 3);
            ADD_FLOAT(_T("in_hfov"), v360.in_hfov, 3);
            ADD_FLOAT(_T("h_fov"), v360.out_hfov, 3);
            ADD_NUM(_T("w"), v360.w);
            ADD_NUM(_T("h"), v360.h);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-v360 ") << tmp.str().substr(1);
        } else if (param->v360.enable) {
            cmd << _T(" --vpp-v360");
        }
    }
    if (param->lenscorrection != defaultPrm->lenscorrection) {
        tmp.str(tstring());
        if (!param->lenscorrection.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->lenscorrection.enable || save_disabled_prm) {
            ADD_FLOAT(_T("k1"), lenscorrection.k1, 3);
            ADD_FLOAT(_T("k2"), lenscorrection.k2, 3);
            ADD_FLOAT(_T("cx"), lenscorrection.cx, 3);
            ADD_FLOAT(_T("cy"), lenscorrection.cy, 3);
            ADD_FLOAT(_T("vignette"), lenscorrection.vignette, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-lenscorrection ") << tmp.str().substr(1);
        } else if (param->lenscorrection.enable) {
            cmd << _T(" --vpp-lenscorrection");
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
            ADD_NUM(_T("seed"), deband.seed);
            ADD_BOOL(_T("blurfirst"), deband.blurFirst);
            ADD_BOOL(_T("rand_each_frame"), deband.randEachFrame);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-deband ") << tmp.str().substr(1);
        } else if (param->deband.enable) {
            cmd << _T(" --vpp-deband");
        }
    }
    if (param->libplacebo_deband != defaultPrm->libplacebo_deband) {
        tmp.str(tstring());
        if (!param->libplacebo_deband.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->libplacebo_deband.enable || save_disabled_prm) {
            ADD_NUM(_T("iterations"), libplacebo_deband.iterations);
            ADD_FLOAT(_T("threshold"), libplacebo_deband.threshold, 3);
            ADD_FLOAT(_T("radius"), libplacebo_deband.radius, 3);
            ADD_FLOAT(_T("grain_y"), libplacebo_deband.grainY, 3);
            if (param->libplacebo_deband.grainC >= 0.0f && param->libplacebo_deband.grainY != param->libplacebo_deband.grainC) {
                ADD_FLOAT(_T("grain_c"), libplacebo_deband.grainC, 3);
            }
            ADD_LST(_T("dither"), libplacebo_deband.dither, list_vpp_libplacebo_deband_dither_mode);
            if (param->libplacebo_deband.lut_size != defaultPrm->libplacebo_deband.lut_size) {
                tmp << _T(",lut_size=") << rgy_pow_int<int>(2, param->libplacebo_deband.lut_size);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-libplacebo-deband ") << tmp.str().substr(1);
        } else if (param->libplacebo_deband.enable) {
            cmd << _T(" --vpp-libplacebo-deband");
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
                ADD_FLOAT2(_T("lumakey_softness"), param->overlay[i], overlayDefault, lumaKey.shoftness, 3);
                ADD_BOOL2(_T("loop"), param->overlay[i], overlayDefault, loop);
            }
            if (!tmp.str().empty()) {
                cmd << _T(" --vpp-overlay ") << tmp.str().substr(1);
            } else if (param->overlay[i].enable) {
                // 値なし指定ではパーサが要素を追加しないため、既定値だけでも有効な要素を復元する。
                cmd << _T(" --vpp-overlay alpha_mode=") << get_chr_from_value(list_vpp_overlay_alpha_mode, (int)param->overlay[i].alphaMode);
            }
        }
    }
    if (param->fruc != defaultPrm->fruc) {
        tmp.str(tstring());
        if (!param->fruc.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->fruc.enable || save_disabled_prm) {
            if (param->fruc.mode == VppFrucMode::NVOFFRUCx2) {
                tmp << _T(",double");
            } else if (param->fruc.mode == VppFrucMode::NVOFFRUCFps) {
                if (param->fruc.targetFps.is_valid()) {
                    tmp << _T(",fps=") << param->fruc.targetFps.printt();
                }
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-fruc ") << tmp.str().substr(1);
        } else if (param->fruc.enable) {
            cmd << _T(" --vpp-fruc");
        }
    }
    OPT_BOOL(_T("--vpp-perf-monitor"), _T("--no-vpp-perf-monitor"), checkPerformance);
    return cmd.str();
}

tstring gen_cmd(const RGYParamCommon *param, const RGYParamCommon *defaultPrm, bool save_disabled_prm, RGYDisableGenCmdFlags disable_flags) {
    std::basic_stringstream<TCHAR> cmd;

    OPT_STR_PATH(_T("-i"), inputFilename);
    OPT_STR_PATH(_T("-o"), outputFilename);

    std::basic_stringstream<TCHAR> tmp;

    OPT_FLOAT(_T("--input-analyze"), demuxAnalyzeSec, 6);
    OPT_NUM(_T("--input-probesize"), demuxProbesize);
    OPT_TSTR(_T("--input-pixel-format"), inputPixFmtStr);
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
    OPT_BOOL(_T("--y4m-timestamp"), _T(""), y4mTimestamp);
    OPT_STR(_T("--video-tag"), videoCodecTag);
    OPT_TSTR(_T("--avcodec-prms"), avcodec_videnc_prms);
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
            // トラックID 0 (トラック指定なし)、コーデック自動選択の場合は、それぞれ出力しない
            tstring prm;
            if (pAudioSelect->trackID != 0) {
                prm += printTrack(pAudioSelect);
            }
            if (pAudioSelect->encCodec != RGY_AVCODEC_AUTO) {
                if (prm.length() > 0) prm += _T("?");
                prm += pAudioSelect->encCodec;
            }
            if (pAudioSelect->encCodecPrm.length() > 0) {
                prm += _T(":") + pAudioSelect->encCodecPrm;
            }
            cmd << _T(" --audio-codec");
            if (prm.length() > 0) {
                cmd << _T(" ") << prm;
            }
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encCodecProfile.length() > 0) {
            cmd << _T(" --audio-profile ") << printTrackPrm(pAudioSelect) << pAudioSelect->encCodecProfile;
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encBitrate.size() > 0) {
            tstring bitratestr;
            for (const auto& bitrate : pAudioSelect->encBitrate) {
                if (!bitratestr.empty()) bitratestr += _T(",");
                bitratestr += bitrate.print();
            }
            cmd << _T(" --audio-bitrate ") << printTrackPrm(pAudioSelect) << bitratestr;
        }
    }
    OPT_BOOL(_T("--audio-encode-other-codec-only"), _T(""), audioEncodeOtherCodecOnly);
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encQuality.first) {
            cmd << _T(" --audio-quality ") << printTrackPrm(pAudioSelect) << pAudioSelect->encQuality.second;
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
                tmp << char_to_tstring(pAudioSelect->streamChannelSelect[j]);
            }
            if (pAudioSelect->streamChannelOut[j] != RGY_CHANNEL_AUTO) {
                tmp << _T(":");
                tmp << char_to_tstring(pAudioSelect->streamChannelOut[j]);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << printTrackPrm(pAudioSelect) << tmp.str();
        }
    }
#endif
    tmp.str(tstring());

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encSamplingRate > 0) {
            cmd << _T(" --audio-samplerate ") << printTrackPrm(pAudioSelect) << pAudioSelect->encSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), audioResampler, list_resampler);
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->resamplerPrm.length() > 0) {
            cmd << _T(" --audio-resampler ") << printTrackPrm(pAudioSelect) << char_to_tstring(pAudioSelect->resamplerPrm);
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->filter.length() > 0) {
            cmd << _T(" --audio-filter ") << printTrackPrm(pAudioSelect) << pAudioSelect->filter;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->addDelayMs != 0.0) {
            cmd << _T(" --audio-delay ") << printTrackPrm(pAudioSelect) << pAudioSelect->addDelayMs;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->bsf.length() > 0) {
            cmd << _T(" --audio-bsf ") << printTrackPrm(pAudioSelect) << pAudioSelect->bsf;
        }
        if (pAudioSelect->disposition.length() > 0) {
            cmd << _T(" --audio-disposition ") << printTrackPrm(pAudioSelect) << pAudioSelect->disposition;
        }
        for (auto &m : pAudioSelect->metadata) {
            cmd << _T(" --audio-metadata ") << printTrackPrm(pAudioSelect) << m;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && pAudioSelect->extractFilename.length() > 0) {
            cmd << _T(" --audio-file ") << printTrackPrm(pAudioSelect);
            if (pAudioSelect->extractFormat.length() > 0) {
                cmd << pAudioSelect->extractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->extractFilename << _T("\"");
        }
    }
    for (const auto &src : param->audioSource) {
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && src.filename.length() > 0) {
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
                    if (sel.encBitrate.size() > 0) {
                        tstring bitratestr;
                        for (const auto& bitrate : sel.encBitrate) {
                            if (!bitratestr.empty()) bitratestr += _T(",");
                            bitratestr += bitrate.print();
                        }
                        tmp << _T(";bitrate=") << bitratestr;
                    }
                    if (sel.encQuality.first) {
                        tmp << _T(";quality=") << sel.encQuality.second;
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
    bool hasSubCopy = false;
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        const SubtitleSelect *pSubSelect = param->ppSubtitleSelectList[i];
        if (pSubSelect->encCodec != RGY_AVCODEC_COPY) {
            continue;
        }
        hasSubCopy = true;
        // トラックID 0 はトラック指定なしを意味するので、トラック番号としては出力しない
        if (pSubSelect->trackID != 0) {
            tmp << _T(",") << printTrack(pSubSelect);
            if (pSubSelect->asdata) {
                tmp << _T("?asdata");
            }
        } else if (pSubSelect->asdata) {
            tmp << _T(",asdata");
        }
    }
    if (hasSubCopy) {
        cmd << _T(" --sub-copy");
        if (!tmp.str().empty()) {
            cmd << _T(" ") << tmp.str().substr(1);
        }
    }
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        const SubtitleSelect *pSubSelect = param->ppSubtitleSelectList[i];
        if (pSubSelect->encCodec.empty() || pSubSelect->encCodec == RGY_AVCODEC_COPY) {
            continue;
        }
        tstring prm;
        if (pSubSelect->trackID != 0) {
            prm += printTrack(pSubSelect);
        }
        if (pSubSelect->encCodec != RGY_AVCODEC_AUTO) {
            if (!prm.empty()) prm += _T("?");
            prm += pSubSelect->encCodec;
        }
        if (!pSubSelect->encCodecPrm.empty()) {
            prm += _T(":") + pSubSelect->encCodecPrm;
        }
        if (!pSubSelect->decCodecPrm.empty()) {
            prm += _T("#") + pSubSelect->decCodecPrm;
        }
        cmd << _T(" --sub-codec");
        if (!prm.empty()) {
            cmd << _T(" ") << prm;
        }
    }
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        const SubtitleSelect *pSubSelect = param->ppSubtitleSelectList[i];
        if (pSubSelect->disposition.length() > 0) {
            cmd << _T(" --sub-disposition ") << printTrackPrm(pSubSelect) << pSubSelect->disposition;
        }
        for (auto &m : pSubSelect->metadata) {
            cmd << _T(" --sub-metadata ") << printTrackPrm(pSubSelect) << m;
        }
    }
    tmp.str(tstring());
    for (const auto &src : param->subSource) {
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && src.filename.length() > 0) {
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
            cmd << _T(" --sub-bsf ") << printTrackPrm(param->ppSubtitleSelectList[i]) << param->ppSubtitleSelectList[i]->bsf;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < param->nDataSelectCount; i++) {
        if (param->ppDataSelectList[i]->trackID != 0) { // トラックID 0 はトラック指定なしなので出力しない
            tmp << _T(",") << printTrack(param->ppDataSelectList[i]);
        }
    }
    if (param->nDataSelectCount > 0) {
        cmd << _T(" --data-copy");
        if (!tmp.str().empty()) {
            cmd << _T(" ") << tmp.str().substr(1);
        }
    }
    tmp.str(tstring());
    for (int i = 0; i < param->nDataSelectCount; i++) {
        const DataSelect *pDataSelect = param->ppDataSelectList[i];
        if (pDataSelect->disposition.length() > 0) {
            cmd << _T(" --data-disposition ") << printTrackPrm(pDataSelect) << pDataSelect->disposition;
        }
        for (auto &m : pDataSelect->metadata) {
            cmd << _T(" --data-metadata ") << printTrackPrm(pDataSelect) << m;
        }
    }

    for (int i = 0; i < param->nAttachmentSelectCount; i++) {
        if (param->ppAttachmentSelectList[i]->trackID != 0) { // トラックID 0 はトラック指定なしなので出力しない
            tmp << _T(",") << param->ppAttachmentSelectList[i]->trackID;
        }
    }
    if (param->nAttachmentSelectCount > 0) {
        cmd << _T(" --attachment-copy");
        if (!tmp.str().empty()) {
            cmd << _T(" ") << tmp.str().substr(1);
        }
    }
    tmp.str(tstring());

    for (const auto &src : param->attachmentSource) {
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && src.filename.length() > 0) {
            cmd << _T(" --attachment-source ") << _T("\"") << src.filename << _T("\"");
            for (const auto &channel : src.select) {
                tmp.str(tstring());
                for (const auto& metadata : channel.second.metadata) {
                    tmp << _T(";metadata=") << metadata;
                }
                if (tmp.str().empty()) continue; // 出力する内容が無い場合は区切りの ":" も出力しない
                cmd << _T(":");
                {
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
    OPT_BOOL(_T("--muxer-add-cmd"), _T(""), muxerAddCmd);
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
    if (param->doviRpuMetadataCopy) {
        cmd << _T("--dolby-vision-rpu copy");
    } else {
        OPT_STR_PATH(_T("--dolby-vision-rpu"), doviRpuFile);
    }

    if (param->doviRpuParams != defaultPrm->doviRpuParams) {
        tmp.str(tstring());
        if (param->doviRpuParams.activeAreaOffsets.enable) {
            tmp << ",crop=on";
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --dolby-vision-rpu-prm ") << tmp.str().substr(1);
        }
    }
    if (param->timecode || param->timecodeFile.length() > 0) {
        cmd << (param->timecode ? _T(" --timecode") : _T(" --no-timecode"));
        if (param->timecodeFile.length() > 0) {
            cmd << _T(" \"") << param->timecodeFile << _T("\"");
        }
    }

    OPT_LST(_T("--input-hevc-bsf"), hevcbsf, list_hevc_bsf_mode);
    if (param->adaptResolution != defaultPrm->adaptResolution) {
        cmd << _T(" --adapt-resolution ") << param->adaptResolution.first << _T("x") << param->adaptResolution.second;
    }
    OPT_STR_PATH(_T("--tcfile-in"), tcfileIn);
    if (param->timebase != defaultPrm->timebase) {
        cmd << _T(" --timebase ") << param->timebase.n() << _T("/") << param->timebase.d();
    }

    OPT_BOOL(_T("--ssim"), _T("--no-ssim"), metric.ssim);
    OPT_BOOL(_T("--psnr"), _T("--no-psnr"), metric.psnr);

#if ENABLE_VMAF
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
        } else if (param->metric.vmaf.enable) {
            cmd << _T(" --vmaf");
        }
    }
#endif //#if ENABLE_VMAF
#if ENABLE_LIBVSHIP
    OPT_BOOL(_T("--vship-ssimulacra2"), _T("--no-vship-ssimulacra2"), metric.vshipSsimu2.enable);
    if (param->metric.vshipButteraugli != defaultPrm->metric.vshipButteraugli) {
        tmp.str(tstring());
        if (!param->metric.vshipButteraugli.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->metric.vshipButteraugli.enable || save_disabled_prm) {
            ADD_NUM(_T("Qnorm"), metric.vshipButteraugli.Qnorm);
            ADD_NUM(_T("intensity_multiplier"), metric.vshipButteraugli.intensity_multiplier);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vship-butteraugli ") << tmp.str().substr(1);
        } else if (param->metric.vshipButteraugli.enable) {
            cmd << _T(" --vship-butteraugli");
        }
    }
    if (param->metric.vshipCvvdp != defaultPrm->metric.vshipCvvdp) {
        tmp.str(tstring());
        if (!param->metric.vshipCvvdp.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->metric.vshipCvvdp.enable || save_disabled_prm) {
            ADD_STR(_T("model"), metric.vshipCvvdp.model);
            ADD_PATH(_T("model_config_json"), metric.vshipCvvdp.model_config_json.c_str());
            ADD_BOOL(_T("resize"), metric.vshipCvvdp.resize);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vship-cvvdp ") << tmp.str().substr(1);
        } else if (param->metric.vshipCvvdp.enable) {
            cmd << _T(" --vship-cvvdp");
        }
    }
#endif //#if ENABLE_LIBVSHIP
    OPT_BOOL(_T("--offset-video-dts-advance"), _T(""), offsetVideoDtsAdvance);
    OPT_BOOL(_T("--allow-other-negative-pts"), _T(""), allowOtherNegativePts);
    OPT_BOOL(_T("--disable-av1-write-parser"), _T("--no-disable-av1-write-parser"), debugDirectAV1Out);
    OPT_BOOL(_T("--debug-raw-out"), _T("--no-debug-raw-out"), debugRawOut);
    return cmd.str();
}

tstring gen_cmd(const RGYParamControl *param, const RGYParamControl *defaultPrm, bool save_disabled_prm, RGYDisableGenCmdFlags disable_flags) {
    if (rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::CtrlPrms)) {
        return _T("");
    }
    std::basic_stringstream<TCHAR> cmd;
    OPT_NUM(_T("--output-buf"), outputBufSizeMB);
    OPT_NUM(_T("--thread-output"), threadOutput);
    OPT_NUM(_T("--thread-input"), threadInput);
    OPT_NUM(_T("--thread-audio"), threadAudio);
    OPT_NUM(_T("--thread-csp"), threadCsp);
    if (param->threadParams != defaultPrm->threadParams) {
        cmd << _T(" --thread-affinity ")    << param->threadParams.to_string(RGYParamThreadType::affinity);
#if defined(_WIN32) || defined(_WIN64)
        // --thread-priority / --thread-throttling はWindowsのみのオプション
        cmd << _T(" --thread-priority ")    << param->threadParams.to_string(RGYParamThreadType::priority);
        cmd << _T(" --thread-throttling ") << param->threadParams.to_string(RGYParamThreadType::throttling);
#endif
    }
    OPT_LST(_T("--simd-csp"), simdCsp, list_simd);
    OPT_NUM(_T("--max-procfps"), procSpeedLimit);
    if (param->avoidIdleClock != defaultPrm->avoidIdleClock) {
        cmd << _T(" --avoid-idle-clock ") << get_chr_from_value(list_avoid_idle_clock, (int)param->avoidIdleClock.mode);
        if (param->avoidIdleClock.mode != RGYParamAvoidIdleClockMode::Disabled
            && param->avoidIdleClock.loadPercent != defaultPrm->avoidIdleClock.loadPercent) {
            cmd << _T("=") << std::setprecision(2) << param->avoidIdleClock.loadPercent;
        }
    }
    OPT_BOOL(_T("--task-perf-monitor"), _T(""), taskPerfMonitor);
    OPT_BOOL(_T("--lowlatency"), _T(""), lowLatency);
    OPT_BOOL(_T("--fallback-bitdepth"), _T(""), fallbackBitdepth);
    OPT_STR_PATH(_T("--log"), logfile);
    if (param->loglevel != defaultPrm->loglevel) {
        cmd << _T(" --log-level ") << param->loglevel.to_string();
    }

    if (param->logOpt != defaultPrm->logOpt) {
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        ADD_BOOL(_T("addtime"), logOpt.addTime);
        ADD_BOOL(_T("addlevel"), logOpt.addLogLevel);
        ADD_BOOL(_T("color"), logOpt.disableColor);
        if (!tmp.str().empty()) {
            cmd << _T(" --log-opt ") << tmp.str().substr(1);
        }
    }
    if (param->logFramePosList.enable) {
        cmd << _T(" --log-framelist");
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && param->logFramePosList.filename.length() > 0) {
            cmd << _T(" \"") << param->logFramePosList.filename << _T("\"");
        }
    }
    if (param->logPacketsList.enable) {
        cmd << _T(" --log-packets");
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && param->logPacketsList.filename.length() > 0) {
            cmd << _T(" \"") << param->logPacketsList.filename << _T("\"");
        }
    }
    if (param->logMuxVidTs.enable) {
        cmd << _T(" --log-mux-ts");
        if (!rgy_disable_gen_cmd(disable_flags, RGYDisableGenCmdFlags::FilePath) && param->logMuxVidTs.filename.length() > 0) {
            cmd << _T(" \"") << param->logMuxVidTs.filename << _T("\"");
        }
    }
    OPT_BOOL(_T("--skip-hwenc-check"), _T(""), skipHWEncodeCheck);
    OPT_BOOL(_T("--skip-hwdec-check"), _T(""), skipHWDecodeCheck);
    OPT_STR_PATH(_T("--avsdll"), avsdll);
    OPT_STR_PATH(_T("--vsdir"), vsdir);
    OPT_BOOL(_T("--vpy-assume-script-dir"), _T(""), vpyAssumeScriptDir);
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
    OPT_STR_PATH(_T("--python"), pythonPath);
    if (param->parentProcessID != defaultPrm->parentProcessID) {
        cmd << strsprintf(_T(" --parent-pid %x"), param->parentProcessID);
    }
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
    if (param->enableVulkan != defaultPrm->enableVulkan) {
        if (param->enableVulkan == RGYParamInitVulkan::Disable) {
            cmd << _T(" --disable-vulkan");
        } else if (param->enableVulkan == RGYParamInitVulkan::TargetVendor) {
            cmd << _T(" --enable-vulkan");
        } else if (param->enableVulkan == RGYParamInitVulkan::All) {
            cmd << _T(" --enable-vulkan-all");
        }
    }
    OPT_NUM(_T("--opencl-build-threads"), openclBuildThreads);
#if ENCODER_QSV
    OPT_NUM(_T("--opencl-task-threads"), openclTaskThreads);
#endif
    OPT_TSTR(_T("--cl-perf-dump"), clPerfDumpDir);
    if (param->clPerfTimelineSec != defaultPrm->clPerfTimelineSec && param->clPerfTimelineSec != 0.0) {
        cmd << _T(" --cl-perf-timeline ") << param->clPerfTimelineSec;
    }
    OPT_TSTR(_T("--cl-perf-disasm-tool"), clPerfDisasmTool);
    OPT_TSTR(_T("--ocloc-path"), clPerfOclocPath);
    OPT_TSTR(_T("--rga-path"), clPerfRgaPath);
    OPT_BOOL(_T("--process-monitor-dev-usage"), _T(""), processMonitorDevUsage);
    OPT_BOOL(_T("--process-monitor-dev-usage-reset"), _T(""), processMonitorDevUsageReset);

    if (param->parallelEnc != defaultPrm->parallelEnc) {
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        ADD_NUM(_T("mp"), parallelEnc.parallelCount);
        ADD_NUM(_T("id"), parallelEnc.parallelId);
        ADD_NUM(_T("chunks"), parallelEnc.chunks);
        if (param->parallelEnc.chunkPipeHandles.size() > 0) {
            tmp << _T(",chunk-handles=");
            for (size_t i = 0; i < param->parallelEnc.chunkPipeHandles.size(); i++) {
                if (i > 0) {
                    tmp << _T(":");
                }
                tmp << param->parallelEnc.chunkPipeHandles[i].handle << _T("#") << param->parallelEnc.chunkPipeHandles[i].startFrameId;
            }
        }
        ADD_LST(_T("cache"), parallelEnc.cacheMode, list_parallel_enc_cache);
        if (!tmp.str().empty()) {
            cmd << _T(" --parallel ") << tmp.str().substr(1);
        }
        OPT_BOOL(_T("--parallel-force-large-memory-filters"), _T(""), parallelEnc.forceLargeMemoryFilters);
    }
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
        _T("   --avsw [<string>]            set input to use avcodec + sw decoder\n")
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
        _T("      ignore_sar=<bool>                ignore sar when using negative value.\n")
        _T("\n")
        _T("   --frames <int>               frames to encode (based on input frames)\n")
        _T("   --fps <int>/<int> or <float> set framerate when --raw is used.\n")
        _T("                                  not recommended for other readers.\n");
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
        _T("   --dolby-vision-rpu <string>  Copy dolby vision metadata from input rpu file.\n")
        _T("   --dolby-vision-rpu copy      Copy dolby vision metadata from input file.\n")
        _T("   --dolby-vision-rpu-prm <param1>=<value>[,<param2>=<value>][...]\n")
        _T("     parameters for --dolby-vision-rpu.\n")
        _T("    params\n")
        _T("      crop=<bool>               Set active area offsets to 0 (no letterbox bars).\n")
    );
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
        _T("   --avcodec-prms <string>      set parameters for avcodec video encoder.\n")
        _T("                                 available when avcodec encoder is enabled by -c av_xxx.\n")
        _T("   --audio-source <string>      input extra audio file.\n")
        _T("   --audio-file [<int>?][<string>:]<string>\n")
        _T("                                extract audio into file.\n")
        _T("                                 could be only used with avhw/avsw reader.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to extract.\n")
        _T("                                  in [<string>?], specify output format.\n")
        _T("   --trim <int>:<int>[,<int>:<int>]...\n")
        _T("                                trim video for the frame range specified.\n")
        _T("                                 frame range should not overlap each other.\n")
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
        _T("   --y4m-timestamp             add XTIMEBASE to y4m header and timestamps to FRAME lines.\n")
        _T("   --audio-copy [<int/string>[,...]]\n")
        _T("                                mux audio with video during output.\n")
        _T("                                 could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 by default copies all audio tracks.\n")
        _T("                                 prefix track number with ! to exclude it.\n")
        _T("                                 prefix language with ! to exclude it.\n")
        _T("                                 \"--audio-copy 1,2\" will extract\n")
        _T("                                 audio track #1 and #2.\n")
        _T("   --audio-codec [<int>?]<string>\n")
        _T("                                encode audio to specified format.\n")
        _T("                                  in [<int>?], specify track number to encode.\n")
        _T("   --audio-encode-other-codec-only\n")
        _T("                                with --audio-codec, copy if input audio codec\n")
        _T("                                 equals the specified codec, and encode only\n")
        _T("                                 when different.\n")
        _T("   --audio-profile [<int>?]<string>\n")
        _T("                                specify audio profile.\n")
        _T("                                  in [<int>?], specify track number to apply.\n")
        _T("   --audio-bitrate [<int>?]<int> or [<int>?]<string>:<int>[,<string>:<int>][...]\n")
        _T("                                set encode bitrate for audio (kbps).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("                                  in [<string>?], specify channel symbols of audio.\n")
        _T("   --audio-quality [<int>?]<int>\n")
        _T("                                set encode quality for audio.\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-ignore-decode-error <int>  (default: %d)\n")
        _T("                                set numbers of continuous packets of audio decode\n")
        _T("                                 error to ignore, replaced by silence.\n")
        _T("   --audio-samplerate [<int>?]<int>\n")
        _T("                                set sampling rate for audio (Hz).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-resampler [<int>?]<string>\n")
        _T("                                set audio resampler or resampler params.\n")
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
        _T("   --sub-copy [<int/string>[,...]]\n")
        _T("                                copy subtitle to output file.\n")
        _T("                                 prefix track number with ! to exclude it.\n")
        _T("                                 prefix language with ! to exclude it.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("   --sub-codec [<int/string>?]<string>\n")
        _T("                                convert subtitle to specified format.\n")
        _T("   --sub-disposition [<int>?]<string>\n")
        _T("                                set disposition for the specified subtitle track.\n")
        _T("                                disposition for the unspecified tracks will be reset.\n")
        _T("   --sub-metadata [<int>?]<string>\n")
        _T("                                set metadata for the specified audio track.\n")
        _T("                                 - copy ... copy metadata from input (default)\n")
        _T("                                 - clear ... do not set metadata\n")
        _T("   --sub-bsf [<int>?]<string>   set bitstream filter to subtitle track.\n")
        _T("   --data-copy [<int/string>[,...]]\n")
        _T("                                copy data stream to output file.\n")
        _T("                                 prefix track number or language with ! to exclude it.\n")
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
        _T("  --muxer-add-cmd               add input command line to muxer metadata (encoding_tool)\n")
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
        _T("   --timebase <int>/<int>       timebase of input timecode.\n")
        _T("\n")
        _T("   --input-hevc-bsf <string>    switch hevc bitstream filter used for hw decoder input\n")
        _T("                                 - internal   ... use internal implementation (default)\n")
        _T("                                 - libavcodec ... use hevc_mp4toannexb bsf\n"),
        DEFAULT_IGNORE_DECODE_ERROR);
    str += _T("\n")
        _T("   --adapt-resolution <int>x<int>\n")
        _T("                                入力途中の解像度変更で許容する最大解像度を指定する。\n")
        _T("                                avhwではデコーダ、avswでは入力サーフェスを指定値で確保する。\n");
    str += _T("\n")
        _T("   --input-pixel-format <string>  set input pixel format for avdevice\n")
        _T("   --offset-video-dts-advance  offset timestamp to cancel bframe delay\n")
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
#if ENABLE_LIBVSHIP
    str += _T("")
        _T("   --vship-ssimulacra2          calc SSIMULACRA2 score using libvship.\n")
        _T("   --no-vship-ssimulacra2       disable SSIMULACRA2 score calculation.\n")
        _T("\n")
        _T("   --vship-butteraugli [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Calc Butteraugli score using libvship.\n")
        _T("    params\n")
        _T("      Qnorm=<int>               norm for Butteraugli score aggregation [default:2].\n")
        _T("      intensity_multiplier=<float>\n")
        _T("                                intensity multiplier for Butteraugli [default:80.0].\n")
        _T("\n")
        _T("   --vship-cvvdp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Calc ColorVideoVDP score using libvship.\n")
        _T("    params\n")
        _T("      model=<string>            set cvvdp model name [default:standard_4k].\n")
        _T("      model_config_json=<string> set path to cvvdp model config json.\n")
        _T("      resize=<bool>             resize frames before cvvdp calculation [default:false].\n")
        _T("\n");
#endif //#if ENABLE_LIBVSHIP
    return str;
}

// 表示候補はパーサと同じ列挙表から取り、help専用の値テーブルを二重管理しない。
tstring rgy_cmd_help_values(const CX_DESC *list) {
    tstring result = _T("        values: ");
    std::vector<tstring> seen;
    if (list != nullptr) {
        for (size_t i = 0; list[i].desc != nullptr; ++i) {
            const tstring value = list[i].desc;
            if (std::find(seen.begin(), seen.end(), value) != seen.end()) continue;
            if (!seen.empty()) result += _T(", ");
            result += value;
            seen.push_back(value);
        }
    }
    return seen.empty() ? tstring() : result + _T("\n");
}

// 共通部とエンコーダ固有部の見出しをまとめ、同じ用途を分断せず表示する。
// 見出しのない行は先頭へ残し、分類の記載漏れを検証器から隠さない。
tstring rgy_cmd_help_vpp_sections(const tstring& help) {
    struct Section { const TCHAR *id; const TCHAR *name; };
    static const Section sections[] = {
        { _T("deinterlace"), _T("Deinterlace") }, { _T("denoise"), _T("Denoise") },
        { _T("edge"), _T("Edge enhancement") }, { _T("deband"), _T("Deband") },
        { _T("color"), _T("Color adjustment") }, { _T("resize"), _T("Resize and geometry") },
        { _T("framerate"), _T("Frame rate and interpolation") }, { _T("overlay"), _T("Overlay") },
        { _T("ai"), _T("AI") }, { _T("other"), _T("Other settings") }
    };
    // 説明は29桁目へ揃え、長い宣言だけ1空白にする。二度まとめても同じ配置を保つ。
    const auto alignDescription = [](const tstring& source) {
        const auto first = source.find_first_not_of(_T(" \t"));
        if (first == tstring::npos) return source;
        const auto text = source.substr(first);
        if (text.compare(0, 7, _T("values:")) == 0) return tstring(28, _T(' ')) + text;
        size_t end = tstring::npos;
        if (first == 6 && text.find(_T("=<")) != tstring::npos) {
            end = source.find_first_of(_T(" \t"), first);
        } else if (first == 3 && text.compare(0, 6, _T("--vpp-")) == 0) {
            end = source.find_first_of(_T(" \t"), first);
            if (end != tstring::npos) {
                const auto argument = source.find_first_not_of(_T(" \t"), end);
                if (argument != tstring::npos && (source[argument] == _T('<') || source[argument] == _T('[')))
                    end = source.find_first_of(_T(" \t"), argument);
            }
        } else if (first >= 8) {
            return tstring(28, _T(' ')) + text;
        }
        if (end == tstring::npos) return source;
        const auto description = source.find_first_not_of(_T(" \t"), end);
        if (description == tstring::npos) return source.substr(0, end);
        return source.substr(0, end) + tstring(end < 28 ? 28 - end : 1, _T(' ')) + source.substr(description);
    };
    std::vector<tstring> content(_countof(sections));
    tstring unclassified;
    int current = -1;
    for (auto line : split(help, _T("\n"))) {
        if (!line.empty() && line.back() == _T('\r')) line.pop_back();
        bool heading = false;
        for (size_t i = 0; i < _countof(sections); ++i) {
            const auto prefix = tstring(_T("  [")) + sections[i].id + _T("] ");
            if (line.compare(0, prefix.size(), prefix) == 0) {
                current = (int)i;
                heading = true;
                break;
            }
        }
        if (!heading) (current < 0 ? unclassified : content[current]) += alignDescription(line) + _T("\n");
    }
    if (unclassified.find_first_not_of(_T(" \t\r\n")) == tstring::npos) unclassified.clear();
    tstring result = unclassified;
    for (size_t i = 0; i < _countof(sections); ++i) {
        auto& block = content[i];
        while (!block.empty() && (block.front() == _T('\n') || block.front() == _T('\r'))) block.erase(block.begin());
        while (!block.empty() && (block.back() == _T('\n') || block.back() == _T('\r'))) block.pop_back();
        if (block.empty()) continue;
        result += tstring(_T("\n  [")) + sections[i].id + _T("] ") + sections[i].name + _T("\n") + block + _T("\n");
    }
    return result;
}

tstring gen_cmd_help_vpp() {
    tstring str;
#if ENABLE_VPP_ORDER
    str += _T("\n  [other] Other settings\n");
    str += _T("   --vpp-order <filter[,filter...]> Filter order; unspecified filters retain their normal order. (default: derived)\n");
    str += rgy_cmd_help_values(get_list_vpp_filter().data());
#endif
    str += _T("\n  [other] Other settings\n");
    str += _T("   --vpp-deint-csp <string> Select the colorspace used for deinterlacing. (default: input)\n");
    str += rgy_cmd_help_values(list_vpp_deint_csp);
#if ENABLE_VPP_FILTER_COLORSPACE
    str += _T("\n  [color] Color adjustment\n");
    str += _T("   --vpp-colorspace [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Convert colorspace, transfer characteristics, or HDR tone mapping.\n    params\n");
    str += _T("      matrix=<from>:<to>          Matrix conversion; retain the input matrix when unspecified. (default: derived)\n");
    str += _T("        Names: bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c.\n");
    str += _T("      colorprim=<from>:<to>       Primaries conversion; retain the input primaries when unspecified. (default: derived)\n");
    str += _T("        Names: bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020.\n");
    str += _T("      transfer=<from>:<to>        Transfer conversion; retain the input transfer when unspecified. (default: derived)\n");
    str += _T("        Names: bt709, smpte170m, bt470m, bt470bg, smpte240m, linear, log100, log316,\n");
    str += _T("        iec61966-2-4, iec61966-2-1, bt2020-10, bt2020-12, smpte2084, arib-srd-b67.\n");
    str += _T("      range=<from>:<to>           Signal-range conversion (limited or full); retain the input range when unspecified. (default: derived)\n");
    str += _T("      lut3d=<path>                3D LUT file.\n");
    str += _T("      lut3d_interp=<string>       3D LUT interpolation. (default: tetrahedral)\n");
    str += rgy_cmd_help_values(list_vpp_colorspace_lut3d_interp);
    str += _T("      hdr2sdr=<string>            HDR10-to-SDR tone-mapping operator. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_hdr2sdr);
    str += strsprintf(_T("      source_peak=<float>         HDR source peak luminance in nits. (default: %.1f)\n"), FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK);
    str += strsprintf(_T("      ldr_nits=<float>            Target SDR peak luminance in nits. (default: %.1f)\n"), FILTER_DEFAULT_COLORSPACE_LDRNITS);
    str += strsprintf(_T("      desat_base=<float>          Base luminance for highlight desaturation. (default: %.2f)\n"), FILTER_DEFAULT_HDR2SDR_DESAT_BASE);
    str += strsprintf(_T("      desat_strength=<float>      Highlight desaturation strength. (default: %.2f)\n"), FILTER_DEFAULT_HDR2SDR_DESAT_STRENGTH);
    str += strsprintf(_T("      desat_exp=<float>           Highlight desaturation exponent. (default: %.2f)\n"), FILTER_DEFAULT_HDR2SDR_DESAT_EXP);
#endif
#if ENABLE_LIBPLACEBO
    str += _T("\n  [color] Color adjustment\n");
    str += _T("   --vpp-libplacebo-tonemapping [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Convert HDR/SDR colorspace with libplacebo tone mapping.\n    params\n");
    str += _T("      src_csp=<string>            Input colorspace. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_csp);
    str += _T("      dst_csp=<string>            Output colorspace. (default: sdr)\n");
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_csp);
    str += _T("      src_max=<float>             Maximum input luminance; inferred from HDR/SDR input. (default: derived)\n");
    str += _T("      src_min=<float>             Minimum input luminance; inferred from HDR/SDR input. (default: derived)\n");
    str += _T("      dst_max=<float>             Maximum output luminance; inferred from output HDR/SDR colorspace. (default: derived)\n");
    str += _T("      dst_min=<float>             Minimum output luminance; inferred from output HDR/SDR colorspace. (default: derived)\n");
    str += strsprintf(_T("      dynamic_peak_detection=<bool>  Detect changing signal peak luminance. (default: %s)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DYNAMIC_PEAK_DETECTION ? _T("true") : _T("false"));
    str += strsprintf(_T("      smooth_period=<float>       Peak-detection smoothing period. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SMOOTH_PERIOD);
    str += strsprintf(_T("      scene_threshold_low=<float> Low scene-change threshold in dB. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SCENE_THRESHOLD_LOW);
    str += strsprintf(_T("      scene_threshold_high=<float>  High scene-change threshold in dB. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SCENE_THRESHOLD_HIGH);
    str += strsprintf(_T("      percentile=<float>          Peak-detection luminance percentile. (default: %.3f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_PERCENTILE);
    str += strsprintf(_T("      black_cutoff=<float>        Peak-detection black cutoff in PQ. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_BLACK_CUTOFF);
    str += strsprintf(_T("      gamut_mapping=<string>      Gamut-mapping algorithm. (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_gamut_mapping, FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_GAMUT_MAPPING));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_gamut_mapping);
    str += strsprintf(_T("      tonemapping_function=<string>  Tone-mapping curve. (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_function, FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_TONEMAPPING_FUNCTION));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_function);
    str += strsprintf(_T("      metadata=<string>           Source of signal metadata. (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_metadata, FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_METADATA));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_metadata);
    str += strsprintf(_T("      contrast_recovery=<float>   Recover local contrast. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_CONTRAST_RECOVERY);
    str += strsprintf(_T("      contrast_smoothness=<float> Contrast-recovery lowpass kernel size. (default: %.1f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_CONTRAST_SMOOTHNESS);
    str += strsprintf(_T("      inverse_tone_mapping=<bool> Expand SDR content into HDR. (default: %s)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_INVERSE_TONE_MAPPING ? _T("true") : _T("false"));
    str += strsprintf(_T("      visualize_lut=<bool>        Show the tone-mapping LUT. (default: %s)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_VISUALIZE_LUT ? _T("true") : _T("false"));
    str += strsprintf(_T("      show_clipping=<bool>        Highlight clipped pixels. (default: %s)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SHOW_CLIPPING ? _T("true") : _T("false"));
    str += strsprintf(_T("      use_dovi=<int>              Dolby Vision RPU: -1=automatic, 0=ignore, 1=use. (default: %d, range: -1..1)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_USE_DOVI);
    str += _T("      lut_path=<path>             LUT file for tone mapping.\n");
    str += _T("      lut_type=<string>           LUT interpretation; use the libplacebo default when unspecified. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_lut_type);
    str += strsprintf(_T("      dst_pl_transfer=<string>    Output transfer function; pair with dst_pl_colorprim. (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_transfer, FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_PL_TRANSFER));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_transfer);
    str += strsprintf(_T("      dst_pl_colorprim=<string>   Output primaries; pair with dst_pl_transfer. (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_colorprim, FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_PL_COLORPRIM));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_colorprim);
    str += strsprintf(_T("      knee_adaptation=<float>     Knee adaptation speed. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_ADAPTATION);
    str += strsprintf(_T("      knee_min=<float>            Minimum knee point. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_MIN);
    str += strsprintf(_T("      knee_max=<float>            Maximum knee point. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_MAX);
    str += strsprintf(_T("      knee_default=<float>        Default knee point. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_DEFAULT);
    str += strsprintf(_T("      knee_offset=<float>         Knee offset. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_OFFSET);
    str += strsprintf(_T("      slope_tuning=<float>        Spline slope tuning. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SLOPE_TUNING);
    str += strsprintf(_T("      slope_offset=<float>        Spline slope offset. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SLOPE_OFFSET);
    str += strsprintf(_T("      spline_contrast=<float>     Spline contrast. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SPLINE_CONTRAST);
    str += strsprintf(_T("      reinhard_contrast=<float>   Reinhard contrast. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_REINHARD_CONTRAST);
    str += strsprintf(_T("      linear_knee=<float>         Linear knee point. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_LINEAR_KNEE);
    str += strsprintf(_T("      exposure=<float>            Exposure adjustment. (default: %.2f)\n"), FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_EXPOSURE);
    str += _T("\n  [color] Color adjustment\n");
    str += _T("   --vpp-libplacebo-tonemapping-lut <path> LUT file for tone mapping.\n");
#endif
    str += _T("\n  [overlay] Overlay and subtitles\n");
    str += _T("   --vpp-delogo <path>[,<param1>=<value>][,<param2>=<value>][...] Remove a translucent logo using a logo file created by logoscan.auf.\n");
    str += _T("    params\n");
    str += _T("      select=<string>             Logo name, automatic-selection file, or one-based logo index.\n");
    str += _T("      pos=<int>:<int>             Horizontal and vertical logo offset. (default: 0:0)\n");
    str += strsprintf(_T("      depth=<int>                 Logo depth. (default: %d)\n"), FILTER_DEFAULT_DELOGO_DEPTH);
    str += _T("      y=<int>                     Luma offset. (default: 0)\n");
    str += _T("      cb=<int>                    Cb offset. (default: 0)\n");
    str += _T("      cr=<int>                    Cr offset. (default: 0)\n");
#if ENCODER_NVENC
    str += _T("      auto_fade=<bool>            Adjust fade dynamically. (default: false)\n");
    str += _T("      auto_nr=<bool>              Adjust noise reduction dynamically. (default: false)\n");
    str += _T("      nr_area=<int>               Noise-reduction area around the logo. (default: 0)\n");
    str += _T("      nr_value=<int>              Noise-reduction strength near the logo. (default: 0)\n");
    str += _T("      log=<bool>                  Write automatic fade/noise-reduction logs. (default: false)\n");
#endif
#if ENABLE_VPP_FILTER_AFS
    str += _T("\n  [deinterlace] Deinterlace\n");
    str += _T("   --vpp-afs [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Automatic field-shift deinterlacing.\n    params\n");
    str += _T("      preset=<string>             Load an AFS preset before applying parameters below. (default: default)\n");
    str += rgy_cmd_help_values(list_afs_preset);
    str += _T("      ini=<path>                  Load settings exported by afs.auf; explicit parameters override loaded settings.\n");
    str += strsprintf(_T("      top=<int>                   Top scan clipping. (default: %d)\n"), FILTER_DEFAULT_AFS_CLIP_TB);
    str += strsprintf(_T("      bottom=<int>                Bottom scan clipping. (default: %d)\n"), FILTER_DEFAULT_AFS_CLIP_TB);
    str += strsprintf(_T("      left=<int>                  Left scan clipping; use a multiple of 4. (default: %d)\n"), FILTER_DEFAULT_AFS_CLIP_LR);
    str += strsprintf(_T("      right=<int>                 Right scan clipping; use a multiple of 4. (default: %d)\n"), FILTER_DEFAULT_AFS_CLIP_LR);
    str += strsprintf(_T("      method_switch=<int>         Field-shift method switching threshold. (default: %d, range: 0..256)\n"), FILTER_DEFAULT_AFS_METHOD_SWITCH);
    str += strsprintf(_T("      coeff_shift=<int>           Field-shift decision coefficient. (default: %d, range: 0..256)\n"), FILTER_DEFAULT_AFS_COEFF_SHIFT);
    str += strsprintf(_T("      thre_shift=<int>            Stripe threshold for field shifting. (default: %d, range: 0..1024)\n"), FILTER_DEFAULT_AFS_THRE_SHIFT);
    str += strsprintf(_T("      thre_deint=<int>            Stripe threshold for deinterlacing. (default: %d, range: 0..1024)\n"), FILTER_DEFAULT_AFS_THRE_DEINT);
    str += strsprintf(_T("      thre_motion_y=<int>         Luma motion threshold. (default: %d, range: 0..1024)\n"), FILTER_DEFAULT_AFS_THRE_YMOTION);
    str += strsprintf(_T("      thre_motion_c=<int>         Chroma motion threshold. (default: %d, range: 0..1024)\n"), FILTER_DEFAULT_AFS_THRE_CMOTION);
    str += strsprintf(_T("      level=<int>                 Deinterlacing analysis level. (default: %d, range: 0..4)\n"), FILTER_DEFAULT_AFS_ANALYZE);
    str += strsprintf(_T("      shift=<bool>                Enable field shifting. (default: %s)\n"), FILTER_DEFAULT_AFS_SHIFT ? _T("true") : _T("false"));
    str += strsprintf(_T("      drop=<bool>                 Enable frame dropping. (default: %s)\n"), FILTER_DEFAULT_AFS_DROP ? _T("true") : _T("false"));
    str += strsprintf(_T("      smooth=<bool>               Smooth dropped-frame timing. (default: %s)\n"), FILTER_DEFAULT_AFS_SMOOTH ? _T("true") : _T("false"));
    str += strsprintf(_T("      24fps=<bool>                Force 30fps-to-24fps conversion. (default: %s)\n"), FILTER_DEFAULT_AFS_FORCE24 ? _T("true") : _T("false"));
    str += strsprintf(_T("      tune=<string>               Visualize AFS analysis. (default: %s)\n"), get_cx_desc(list_afs_tune_mode, FILTER_DEFAULT_AFS_TUNE));
    str += rgy_cmd_help_values(list_afs_tune_mode);
    str += strsprintf(_T("      rff=<bool>                  Honor repeat-first-field flags. (default: %s)\n"), FILTER_DEFAULT_AFS_RFF ? _T("true") : _T("false"));
    str += strsprintf(_T("      timecode=<bool>             Write timecodes. (default: %s)\n"), FILTER_DEFAULT_AFS_TIMECODE ? _T("true") : _T("false"));
    str += strsprintf(_T("      log=<bool>                  Write an AFS log. (default: %s)\n"), FILTER_DEFAULT_AFS_LOG ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_NNEDI_UPSCALE
    str += _T("\n  [resize] Resize and geometry\n");
    str += _T("   --vpp-nnedi-upscale [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Edge-directed 2x upscale with the NNEDI network.\n    params\n");
    str += _T("     Output is exactly 2x the input. Even-sized progressive 4:2:0, 4:4:4 or monochrome input is required.\n");
    str += _T("      nsize=<string>              NNEDI neighborhood size. (default: 32x4)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_nsize);
    str += _T("      nns=<int>                   Number of neurons. (default: 32)\n");
    str += _T("        values: 16, 32, 64, 128, 256\n");
    str += _T("      quality=<string>            Calculation quality. (default: fast)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_quality);
    str += _T("      prescreen=<int>             Prescreening mode; modes 0 and 1 are unsupported. (default: 2, range: 2..4)\n");
    str += _T("      errortype=<string>          Network error metric. (default: abs)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_error_type);
    str += _T("      clamp=<int>                 Output clamping mode. (default: 1, range: 0..4)\n");
    str += _T("      shift=<string>              Half-pixel correction interpolation. (default: linear)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_upscale_shift);
    str += _T("      weightfile=<path>           Weights file; Windows searches nnedi3_weights.bin, Linux uses embedded weights.\n");
#endif
#if ENABLE_VPP_FILTER_NNEDI
    str += _T("\n  [deinterlace] Deinterlace\n");
    str += _T("   --vpp-nnedi [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     NNEDI deinterlacing.\n    params\n");
    str += _T("      planes=<string>             Target planes; colon-separated combinations are also accepted. (default: all)\n");
    str += _T("        values: all, y, u, v, y:u:v\n");
    str += _T("      field=<string>              Target output field. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_field);
    str += _T("      nsize=<string>              NNEDI neighborhood size. (default: 32x4)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_nsize);
    str += _T("      nns=<int>                   Number of neurons. (default: 32)\n");
    str += _T("        values: 16, 32, 64, 128, 256\n");
    str += _T("      quality=<string>            Calculation quality. (default: fast)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_quality);
    str += _T("      prescreen=<int>             Prescreening mode; modes 0 and 1 are unsupported. (default: 2, range: 2..4)\n");
    str += _T("      errortype=<string>          Network error metric. (default: abs)\n");
    str += rgy_cmd_help_values(list_vpp_nnedi_error_type);
    str += _T("      prec=<string>               Accepted for compatibility and ignored.\n");
    str += _T("      clamp=<int>                 Output clamping mode. (default: 1, range: 0..4)\n");
    str += _T("      double_height=<bool>        Double height; supported with field=auto, top or bottom. (default: false)\n");
    str += _T("      weightfile=<path>           Weights file; Windows searches nnedi3_weights.bin, Linux uses embedded weights.\n");
#endif
#if ENABLE_VPP_FILTER_YADIF
    str += _T("\n  [deinterlace] Deinterlace\n");
    str += _T("   --vpp-yadif [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     YADIF deinterlacing.\n    params\n");
    str += _T("      mode=<string>               Frame output or one output frame per field (bob). (default: auto)\n");
    str += _T("        values: auto, tff, bff, bob, bob_tff, bob_bff\n");
#endif
#if ENABLE_VPP_FILTER_DECOMB
    str += _T("\n  [deinterlace] Deinterlace\n");
    str += _T("   --vpp-decomb [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Decomb deinterlacing.\n    params\n");
    str += strsprintf(_T("      full=<bool>                 Deinterlace every frame. (default: %s)\n"), FILTER_DEFAULT_DECOMB_FULL ? _T("true") : _T("false"));
    str += strsprintf(_T("      threshold=<int>             Combing threshold. (default: %d, range: 0..255)\n"), FILTER_DEFAULT_DECOMB_THRESHOLD);
    str += strsprintf(_T("      dthreshold=<int>            Deinterlacing threshold. (default: %d, range: 0..255)\n"), FILTER_DEFAULT_DECOMB_DTHRESHOLD);
    str += strsprintf(_T("      blend=<bool>                Blend rather than interpolate. (default: %s)\n"), FILTER_DEFAULT_DECOMB_BLEND ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_BWDIF
    str += _T("\n  [deinterlace] Deinterlace\n");
    str += _T("   --vpp-bwdif [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Motion-adaptive deinterlacing with w3fdif and cubic interpolation.\n    params\n");
    str += _T("      mode=<string>               Same-rate frames or double-rate bob output. (default: frame)\n");
    str += rgy_cmd_help_values(list_vpp_bwdif_mode);
    str += _T("      order=<string>              Input field order. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_bwdif_order);
    str += strsprintf(_T("      thr=<float>                 Motion threshold. (default: %.1f, range: 0..100)\n"), FILTER_DEFAULT_BWDIF_THR);
#endif
#if ENABLE_VPP_FILTER_MAA
    str += _T("\n  [edge] Edge enhancement\n");
    str += _T("   --vpp-maa [<param1>=<value>][,<param2>=<value>][...]\n");
    str += _T("     Masked anti-aliasing for animated and cel-shaded content.\n    params\n");
    str += _T("      ss=<float>                  Supersampling factor. (default: 2.0, range: 1..4)\n");
    str += _T("      aa=<int>                    Luma anti-aliasing strength. (default: 48, range: 0..255)\n");
    str += _T("      aac=<int>                   Chroma strength; follows max(0, aa-8) when aa is explicitly set. (default: derived, range: 0..255)\n");
    str += _T("      mask=<bool>                 Enable edge masking. (default: true)\n");
    str += _T("      mthresh=<int>               Edge threshold; higher values detect fewer edges. (default: 7, range: 1..255)\n");
    str += _T("      chroma=<bool>               Process chroma planes; increases processing cost. (default: false)\n");
    str += _T("      show=<int>                  Debug display: 0=normal, 1=mask, 2=mask with anti-aliasing. (default: 0, range: 0..2)\n");
    str += _T("      edge=<string>               Edge operator. (default: sobel)\n");
    str += _T("        values: sobel, prewitt, sobel_full, scharr, kirsch, laplacian\n");
#endif
#if ENABLE_VPP_FILTER_RTGMC
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-rtgmc [<param1>=<value>][,<param2>=<value>][...]\n     High-quality motion-compensated deinterlacing.\n    params\n");
    str += _T("      preset=<string> Speed and quality preset. (default: faster)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_preset);
    str += _T("      tuning=<string> Preset block-size tuning. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_tuning);
    str += _T("      border=<bool> Clamp four border rows before filtering, then crop them. (default: false)\n");
    str += _T("      lossless=<int> Restore source fields after TR2 (1) or before retouch (2). (default: 0, range: 0..2)\n");
    str += _T("      input_type=<int> Input frame type; progressive SAD masking applies to 2 or 3. (default: 0, range: 0..3)\n");
    str += _T("      prog_sad_mask=<float> Progressive SAD mask strength chosen by preset. (default: derived)\n");
    str += _T("      prog_sad_mask_gamma=<float> Progressive SAD mask gamma; must be positive. (default: 1.0)\n");
    str += _T("      source_match=<int> Number of source-refinement stages. (default: 0, range: 0..3)\n");
    str += _T("      match_tr1=<int> Temporal radius for the first refinement stage. (default: 1, range: 0..2)\n");
    str += _T("      match_tr2=<int> Temporal radius for later refinement stages. (default: 1, range: 0..2)\n");
    str += _T("      match_edi=<string> Refinement interpolation inherited from the selected EDI preset. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_edi_mode);
    str += _T("      match_enhance=<float> Refinement detail boost strength. (default: 0.5, range: 0.0..1.0)\n");
    str += _T("      order=<string> Input field order. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_bob_order);
    str += _T("      tr0=<int> Search-prefilter temporal radius chosen by preset; -1 uses an external EDI reference. (default: derived, range: -1..2)\n");
    str += _T("      rep0-thin=<int> Search-prefilter thin repair chosen by preset. (default: derived, range: 0..7)\n");
    str += _T("      rep0-pad=<int> Search-prefilter padding repair. (default: 0, range: 0..3)\n");
    str += _T("      search_refine=<int> Search-prefilter mode chosen by preset. (default: derived, range: 0..3)\n");
    str += _T("      mv_spatial_refine=<int|auto> Spatial refinement passes; auto limits refinement to the inner level. (default: auto, range: -1..)\n");
    str += _T("        values: auto\n");
    str += _T("      tv_range=<bool> Expand the search reference from TV to PC range. (default: true)\n");
    str += _T("      chroma_motion=<bool> Chroma motion analysis chosen by preset. (default: derived)\n");
    str += _T("      noise_process=<int> Noise processing mode (0=off, 1=denoise, 2=grain restoration), chosen by preset. (default: derived, range: 0..2)\n");
    str += _T("      ezdenoise=<float> Omission preserves preset noise settings; an explicit value selects noise_process and sigma. (default: derived, range: 0.0..)\n");
    str += _T("      ezkeepgrain=<float> Omission preserves preset noise settings; an explicit value selects noise_process. (default: derived, range: 0.0..)\n");
    str += _T("      denoiser=<string> Noise filter chosen by preset. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_noise_denoiser);
    str += _T("      noise_deint=<string> Noise interpolation chosen by preset; generate requires backend support. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_noise_deint);
    str += _T("      sigma=<float> Noise-filter sigma; an explicit ezdenoise value overrides the initial sigma. (default: derived, range: 0.0..)\n");
    str += _T("      chroma_noise=<bool> Denoise chroma planes. (default: true)\n");
    str += _T("      denoise_mc=<bool> Motion-compensated noise processing, chosen by preset. (default: derived)\n");
    str += _T("      noise_tr=<int> Noise temporal radius chosen by preset. (default: derived, range: 0..)\n");
    str += _T("      grain_restore=<float> Grain restoration before TR2, chosen by preset. (default: derived, range: 0.0..1.0)\n");
    str += _T("      noise_restore=<float> Noise restoration after lossless processing, chosen by preset. (default: derived, range: 0.0..1.0)\n");
    str += _T("      edi=<string> EDI interpolation chosen by preset. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_edi_mode);
    str += _T("      nnsize=<int> NNEDI window-size index chosen by preset. (default: derived, range: 0..6)\n");
    str += _T("      nneurons=<int> NNEDI neuron-count index chosen by preset. (default: derived, range: 0..4)\n");
    str += _T("      ediqual=<int> NNEDI quality chosen by preset. (default: derived, range: 1..2)\n");
    str += _T("      chroma_edi=<string> Chroma interpolation. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_chroma_edi_mode);
    str += _T("      rep1-thin=<int> First shimmer thin repair chosen by preset. (default: derived, range: 0..7)\n");
    str += _T("      rep1-pad=<int> First shimmer padding repair. (default: 0, range: 0..3)\n");
    str += _T("      rep2-thin=<int> Second shimmer thin repair chosen by preset. (default: derived, range: 0..7)\n");
    str += _T("      rep2-pad=<int> Second shimmer padding repair. (default: 0, range: 0..3)\n");
    str += _T("      delta_analyze=<int> Analysis temporal delta chosen by preset. (default: derived)\n");
    str += _T("      delta_tr1=<int> TR1 temporal delta chosen by preset. (default: derived)\n");
    str += _T("      delta_tr2=<int> TR2 temporal delta chosen by preset. (default: derived)\n");
    str += _T("      thsad1=<int> TR1 degrain SAD threshold. (default: 640)\n");
    str += _T("      thsad2=<int> TR2 degrain SAD threshold. (default: 256)\n");
    str += strsprintf(_T("      useflag=<int> Reference directions (0=both, 1=backward, 2=forward). (default: %d, range: 0..2)\n"), FILTER_DEFAULT_DEGRAIN_USEFLAG);
    str += _T("      pel=<int> Motion subpixel precision chosen by preset. (default: derived)\n");
    str += _T("        values: 1, 2, 4\n");
    str += _T("      levels=<int> Motion analysis levels. (default: 2, range: 1..2)\n");
    str += _T("      lambda=<int> Motion penalty derived from preset block size and true-motion mode. (default: derived)\n");
    str += _T("      lsad=<int> Large SAD threshold. (default: 400)\n");
    str += _T("      pnew=<int> New-vector penalty. (default: 25)\n");
    str += _T("      plevel=<int> Predictor level. (default: 0)\n");
    str += _T("      globalmotion=<bool> Global-motion tuning. (default: true)\n");
    str += _T("      searchparam=<int> Motion search parameter chosen by preset. (default: derived, range: 1..2)\n");
    str += _T("      pelsearch=<int> Subpixel motion search parameter chosen by preset. (default: derived, range: 1..2)\n");
    str += _T("      search_early_sad=<int|off> Level-0 early SAD threshold in 8x8-block, 8-bit units, chosen by preset. (default: derived, range: -1..65535)\n");
    str += _T("        values: off\n");
    str += _T("      spatial_early_sad=<int|off> Level-1 spatial-refinement SAD threshold in 8x8-block, 8-bit units. (default: off, range: -1..65535)\n");
    str += _T("        values: off\n");
    str += _T("      sharpness=<float> Retouch sharpness; source_match selects the initial strength. (default: derived, range: 0.0..1.0)\n");
    str += strsprintf(_T("      limit=<float> Legacy retouch limit. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_LIMIT);
    str += _T("      smode=<int> Resharpen mode chosen by preset. (default: derived, range: 0..2)\n");
    str += _T("      slmode=<int> Sharpness-limit mode chosen by preset; 1 or 2 runs before TR2. (default: derived, range: 0..4)\n");
    str += _T("      slrad=<int> Sharpness-limit radius chosen by preset. (default: derived, range: 0..3)\n");
    str += strsprintf(_T("      sovs=<int> Sharpness-limit overshoot. (default: %d)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SOVS);
    str += strsprintf(_T("      svthin=<float> Vertical thin strength. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SVTHIN);
    str += _T("      sbb=<int> Back-blend mode chosen by preset. (default: derived, range: 0..3)\n");
    str += _T("      precise=<bool> Precise retouch mode chosen by preset. (default: derived)\n");
#endif
#if ENABLE_VPP_FILTER_KFM
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-kfm [<param1>=<value>][,<param2>=<value>][...]\n     Adaptive inverse telecine.\n    params\n");
    str += _T("      mode=<string> Output cadence. (default: vfr)\n");
    str += rgy_cmd_help_values(list_vpp_kfm_mode);
    str += _T("      preset=<string> Motion-search speed preset. (default: faster)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_preset);
    str += _T("      search_early_sad=<int|auto|off> Level-0 early SAD threshold in 8x8-block, 8-bit units. (default: auto, range: -1..65535)\n");
    str += _T("        values: auto, off\n");
    str += _T("      spatial_early_sad=<int|auto|off> Level-1 spatial-refinement SAD threshold in 8x8-block, 8-bit units. (default: auto, range: -1..65535)\n");
    str += _T("        values: auto, off\n");
    str += _T("      timing=<string> Cadence decision timing. (default: realtime+)\n");
    str += rgy_cmd_help_values(list_vpp_kfm_timing);
    str += _T("      past_cycles=<int> Commit delay in cycles for realtime+. (default: 30, range: 0..)\n");
    str += _T("      thswitch=<float> Threshold for switching to 60p output. (default: 0.5, range: 0.0..1.0)\n");
    str += _T("      ucf=<bool> Use the placeholder UCF copy stage. (default: false)\n");
    str += _T("      nr=<bool> Noise reduction on 24p and 30p branches after UCF, before patching. (default: false)\n");
    str += _T("        Disabled for mode=60; uses block size 32 at height 720 or above, otherwise 16.\n        Inherits early-SAD settings; preserved RFF frames skip UCF/patch but receive 24p NR.\n");
    str += _T("      is120=<bool> Reserve the 120fps duration-correction flag. (default: true)\n");
    str += _T("      rff=<bool> Preserve progressive RFF frames. (default: true)\n");
    str += _T("      debug=<bool> Reserve the stage-dump flag. (default: false)\n");
    str += _T("      debug_stage=<string> 24p debug-output stage. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_kfm_debug_stage);
    str += _T("      timecode=<path> Timecode-v2 output path.\n");
#endif
#if ENABLE_VPP_FILTER_RTGMC_BOB
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-rtgmc-bob [<param1>=<value>][,<param2>=<value>][...]\n     RTGMC bob deinterlacing with double-rate output.\n    params\n");
    str += _T("      order=<string> Input field order. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_bob_order);
#endif
#if ENABLE_VPP_FILTER_RTGMC_SEARCH_PREFILTER
    str += _T("\n  [other] Other settings\n   --vpp-rtgmc-search-prefilter [<param1>=<value>][,<param2>=<value>][...]\n     Search-reference prefilter scaffold.\n    params\n");
    str += strsprintf(_T("      tr0=<int> Temporal radius; -1 uses an external EDI reference. (default: %d, range: -1..2)\n"), FILTER_DEFAULT_DEGRAIN_TR0);
    str += strsprintf(_T("      rep0-thin=<int> Thin repair level. (default: %d, range: 0..7)\n"), FILTER_DEFAULT_DEGRAIN_REP0);
    str += _T("      rep0-pad=<int> Padding repair level. (default: 0, range: 0..3)\n");
    str += strsprintf(_T("      search_refine=<int> Search-prefilter mode. (default: %d, range: 0..3)\n"), FILTER_DEFAULT_DEGRAIN_SEARCH_REFINE);
    str += _T("      tv_range=<bool> Expand the search reference from TV to PC range. (default: true)\n");
    str += _T("      chroma_motion=<bool> Prefilter chroma planes. (default: false)\n");
#endif
#if ENABLE_VPP_FILTER_RTGMC_EDI
    str += _T("\n  [other] Other settings\n   --vpp-rtgmc-edi [<param1>=<value>][,<param2>=<value>][...]\n     Standalone EDI scaffold.\n    params\n");
    str += _T("      mode=<string> Interpolation mode. (default: bob)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_edi_mode);
    str += _T("      nnsize=<int> NNEDI window-size index. (default: 1, range: 0..6)\n");
    str += _T("      nneurons=<int> NNEDI neuron-count index. (default: 1, range: 0..4)\n");
    str += _T("      ediqual=<int> NNEDI quality. (default: 1, range: 1..2)\n");
    str += _T("      chroma_edi=<string> Chroma interpolation. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_chroma_edi_mode);
#endif
#if ENABLE_VPP_FILTER_RTGMC_RETOUCH
    str += _T("\n  [other] Other settings\n   --vpp-rtgmc-retouch [<param1>=<value>][,<param2>=<value>][...]\n     Standalone retouch resharpening.\n    params\n");
    str += strsprintf(_T("      sharpness=<float> Vertical resharpen strength. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SHARPNESS);
    str += strsprintf(_T("      limit=<float> Legacy sharpness limit. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_LIMIT);
    str += strsprintf(_T("      smode=<int> Resharpen mode. (default: %d, range: 0..2)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SMODE);
    str += strsprintf(_T("      slmode=<int> Sharpness-limit mode; 1 or 2 runs before TR2. (default: %d, range: 0..4)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SLMODE);
    str += strsprintf(_T("      slrad=<int> Sharpness-limit radius. (default: %d, range: 0..3)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SLRAD);
    str += strsprintf(_T("      sovs=<int> Sharpness-limit overshoot. (default: %d)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SOVS);
    str += strsprintf(_T("      svthin=<float> Vertical thin strength. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SVTHIN);
    str += strsprintf(_T("      sbb=<int> Back-blend mode. (default: %d, range: 0..3)\n"), FILTER_DEFAULT_RTGMC_RETOUCH_SBB);
    str += _T("      precise=<bool> Use the precise retouch path. (default: false)\n");
    str += strsprintf(_T("      tr1=<int> Effective detail-gain TR1 value. (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_DELTA);
    str += strsprintf(_T("      tr2=<int> Effective detail-gain TR2 value. (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_DELTA);
#endif
#if ENABLE_VPP_FILTER_RTGMC_SHIMMER_REPAIR
    str += _T("\n  [other] Other settings\n   --vpp-rtgmc-shimmer-repair [<param1>=<value>][,<param2>=<value>][...]\n     Standalone shimmer repair scaffold.\n    params\n");
    str += _T("      stage=<string> Bare invocation is standalone; parameterized invocation defaults to rep1. (default: derived)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_shimmer_repair_stage);
    str += _T("      mode=<string> Processing mode; currently passthrough only. (default: passthrough)\n");
    str += _T("        values: passthrough\n");
    str += _T("      rep-thin=<int> Thin repair level. (default: 0, range: 0..7)\n");
    str += _T("      rep-pad=<int> Padding repair level. (default: 0, range: 0..3)\n");
    str += _T("      rep_chroma=<bool> Process chroma planes. (default: true)\n");
#endif
#if ENABLE_VPP_FILTER_RTGMC_PRIMITIVE
    str += _T("\n  [other] Other settings\n   --vpp-rtgmc-primitive [<param1>=<value>][,<param2>=<value>][...]\n     Standalone primitive and debug operations.\n    params\n");
    str += _T("      op=<string> Primitive operation. (default: copy)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_primitive_op);
    str += _T("      ref=<string> Second-input reference for primitive tests. (default: none)\n");
    str += rgy_cmd_help_values(list_vpp_rtgmc_primitive_ref);
    str += strsprintf(_T("      mode=<int> Operation-specific mode; Gaussian p for gaussresize. (default: %d)\n"), FILTER_DEFAULT_RTGMC_PRIMITIVE_MODE);
    str += strsprintf(_T("      weight=<float> Merge or weighted-difference blend weight; limits depend on op. (default: %.3f)\n"), FILTER_DEFAULT_RTGMC_PRIMITIVE_WEIGHT);
    str += strsprintf(_T("      chroma=<bool> Process chroma planes. (default: %s)\n"), FILTER_DEFAULT_RTGMC_PRIMITIVE_CHROMA ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_IVTC
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-ivtc [<param1>=<value>][,<param2>=<value>][...]\n     Inverse telecine with field matching and decimation.\n    params\n");
    str += strsprintf(_T("      guide=<int> Matching mode (0=min-combing, 1=clean-frame preference, 2=PAL 2:2). (default: %d, range: 0..2)\n"), FILTER_DEFAULT_IVTC_GUIDE);
    str += strsprintf(_T("      post=<int> Residual-combing postprocessing (0=off, 2=adaptive bob). (default: %d)\n"), FILTER_DEFAULT_IVTC_POST);
    str += _T("        values: 0, 2\n");
    str += _T("      cycle=<int|auto> Decimation cycle: 0 disables, auto uses 5 for approximately 30fps input. (default: auto, range: 2..16)\n");
    str += _T("        values: auto\n");
    str += strsprintf(_T("      drop=<int> Frames dropped per cycle; only 1 is supported. (default: %d)\n"), FILTER_DEFAULT_IVTC_DROP);
    str += _T("        values: 1\n");
    str += strsprintf(_T("      combthresh=<float> Per-pixel combing threshold. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_IVTC_COMB_THRESH);
    str += strsprintf(_T("      cleanfrac=<float> Allowed fraction of combed pixels in a block for guide=1/post=2. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_IVTC_CLEAN_FRAC);
    str += strsprintf(_T("      dthresh=<int> Per-pixel deinterlace gate (8-bit scale). (default: %d, range: 0..255)\n"), FILTER_DEFAULT_IVTC_DTHRESH);
    str += _T("      chroma=<bool> Include chroma in match-quality scoring. (default: false)\n");
    str += strsprintf(_T("      back=<int> When to test the previous-frame match. (default: %d, range: 0..1)\n"), FILTER_DEFAULT_IVTC_BACK);
    str += strsprintf(_T("      y0=<int> First row of the ignored band; y0=y1=0 disables the band. (default: %d)\n"), FILTER_DEFAULT_IVTC_Y0);
    str += strsprintf(_T("      y1=<int> Last row of the ignored band. (default: %d)\n"), FILTER_DEFAULT_IVTC_Y1);
    str += strsprintf(_T("      nt=<int> Match-metric noise tolerance (8-bit scale). (default: %d)\n"), FILTER_DEFAULT_IVTC_NT);
    str += strsprintf(_T("      cthresh=<int> Per-pixel comb threshold for scoring. (default: %d)\n"), FILTER_DEFAULT_IVTC_CTHRESH);
    str += strsprintf(_T("      combpel=<int> Combed pixels per block needed to mark it combed. (default: %d)\n"), FILTER_DEFAULT_IVTC_COMBPEL);
    str += strsprintf(_T("      scthresh=<float> Scene-change threshold as a fraction of max SAD; 0 selects adaptive mode. (default: %.2f)\n"), FILTER_DEFAULT_IVTC_SCTHRESH);
    str += _T("      cadlock=<string> Cadence pattern lock and match override. (default: auto)\n");
    str += _T("        values: auto, on, off\n");
    str += strsprintf(_T("      gthresh=<int> Cadence-override tolerance in percent. (default: %d, range: 0..100)\n"), FILTER_DEFAULT_IVTC_GTHRESH);
    str += strsprintf(_T("      vthresh=<int> Post-assembly combing-veto threshold. (default: %d, range: 0..256)\n"), FILTER_DEFAULT_IVTC_VTHRESH);
    str += _T("      expand=<string> RFF expansion; auto uses detected pulldown. (default: auto)\n");
    str += _T("        values: auto, on, off\n");
    str += _T("      mixed=<bool> Allow mixed RFF/progressive and interlaced input. (default: false)\n");
    str += strsprintf(_T("      hysteresis=<float> Match hysteresis. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_IVTC_HYSTERESIS);
    str += _T("      tff=<string> Field order; auto reads input picture structure. (default: auto)\n");
    str += _T("        values: auto, on, off\n");
    str += _T("      log=<path|bool> Write per-frame match logs; true derives the log path from output filename. (default: false)\n");
#endif
#if ENABLE_VPP_FILTER_RFF
    str += strsprintf(_T("\n  [framerate] Frame rate and interpolation\n   --vpp-rff Apply RFF flags; available with the %savsw reader.\n"), ENABLE_VPP_FILTER_RFF_AVHW ? _T("avhw/") : _T(""));
#endif
#if ENABLE_VPP_FILTER_SELECT_EVERY
    str += _T("\n  [framerate] Frame rate and interpolation\n   --vpp-select-every [<param1>=<value>][,<param2>=<value>][...]\n     Select one frame from each input-frame group.\n    params\n");
    str += _T("      step=<int> Number of input frames per output frame. (default: 1)\n");
    str += _T("      offset=<int> Offset of the selected frame within each group. (default: 0)\n");
    str += _T("     Short form: --vpp-select-every <step>[,offset=<int>].\n");
#endif
#if ENABLE_VPP_FILTER_DECIMATE
    str += _T("\n  [framerate] Frame rate and interpolation\n   --vpp-decimate [<param1>=<value>][,<param2>=<value>][...]\n     Drop duplicate frames from each cycle.\n    params\n");
    str += strsprintf(_T("      cycle=<int> Frames per decimation cycle. (default: %d)\n"), FILTER_DEFAULT_DECIMATE_CYCLE);
    str += strsprintf(_T("      drop=<int> Frames to drop per cycle. (default: %d)\n"), FILTER_DEFAULT_DECIMATE_DROP);
    str += strsprintf(_T("      thredup=<float> Duplicate threshold in percent. (default: %.1f, range: 0..100)\n"), FILTER_DEFAULT_DECIMATE_THRE_DUP);
    str += strsprintf(_T("      thresc=<float> Scene-change threshold in percent. (default: %.1f, range: 0..100)\n"), FILTER_DEFAULT_DECIMATE_THRE_SC);
    str += strsprintf(_T("      blockx=<int> Horizontal block size. (default: %d)\n"), FILTER_DEFAULT_DECIMATE_BLOCK_X);
    str += rgy_cmd_help_values(list_vpp_decimate_block);
    str += strsprintf(_T("      blocky=<int> Vertical block size. (default: %d)\n"), FILTER_DEFAULT_DECIMATE_BLOCK_Y);
    str += rgy_cmd_help_values(list_vpp_decimate_block);
    str += strsprintf(_T("      chroma=<bool> Consider chroma when detecting duplicates. (default: %s)\n"), FILTER_DEFAULT_DECIMATE_CHROMA ? _T("true") : _T("false"));
    str += strsprintf(_T("      log=<bool> Write a frame-decision log. (default: %s)\n"), FILTER_DEFAULT_DECIMATE_LOG ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_MPDECIMATE
    str += _T("\n  [framerate] Frame rate and interpolation\n   --vpp-mpdecimate [<param1>=<value>][,<param2>=<value>][...]\n     Drop near-duplicate frames using block differences.\n    params\n");
    str += strsprintf(_T("      hi=<int> A frame may be dropped if every 8x8 block difference is below this threshold. (default: %d)\n"), FILTER_DEFAULT_MPDECIMATE_HI);
    str += strsprintf(_T("      lo=<int> Low 8x8-block difference threshold used with frac. (default: %d)\n"), FILTER_DEFAULT_MPDECIMATE_LO);
    str += strsprintf(_T("      frac=<float> Required fraction of blocks with difference below lo. (default: %.3f)\n"), FILTER_DEFAULT_MPDECIMATE_FRAC);
    str += strsprintf(_T("      max=<int> Maximum consecutive drops; negative values set the minimum drop interval. (default: %d)\n"), FILTER_DEFAULT_MPDECIMATE_MAX);
    str += strsprintf(_T("      keep=<int> Consecutive similar frames kept before dropping begins. (default: %d)\n"), FILTER_DEFAULT_MPDECIMATE_KEEP);
    str += strsprintf(_T("      log=<bool> Write a frame-decision log. (default: %s)\n"), FILTER_DEFAULT_MPDECIMATE_LOG ? _T("true") : _T("false"));
#endif
#if ENABLE_NVVFX || ENABLE_NVSDKNGX || ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC
    {
    str += _T("\n  [resize] Resize and geometry\n   --vpp-resize [<param1>=<value>][,<param2>=<value>][...]\n     Select a resizing algorithm and its parameters.\n    params\n      algo=<string> Resizing algorithm. (default: auto)\n");
    // OpenCLとハードウェアで同じ綴りの候補があっても、表示は一度だけにする。
    str += rgy_cmd_help_values(list_vpp_resize);
    str += _T("        The short form is --vpp-resize <algorithm>[,<param>=<value>].\n        Gaussian resizing uses p=2.0.\n");
    str += strsprintf(_T("      sharpness=<float> RCAS strength for fsr1 or USM strength for nis. (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_RESIZE_FSR1_SHARPNESS);
    str += _T("        The fsr1 and nis parameters have the same initial strength.\n");
    str += strsprintf(_T("      dpid_lambda=<float> Detail-preservation strength for dpid; 0 is area-equivalent. (default: %.2f, range: 0.0..4.0)\n"), FILTER_DEFAULT_RESIZE_DPID_LAMBDA);
    str += _T("      cascade=<string> Cascade mode for nis. (default: auto)\n");
    str += _T("        values: auto, on, off\n");
    str += _T("      hdr=<string> HDR mode for nis. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_resize_nis_hdr);
#if ENABLE_OPENCL
    str += _T("      opt=<string> Implementation for nis. (default: default)\n");
    str += rgy_cmd_help_values(list_vpp_resize_nis_opt);
#endif
    str += strsprintf(_T("      b=<float> Mitchell-Netravali B parameter for bicubic. (default: %.2f)\n"), FILTER_DEFAULT_RESIZE_BICUBIC_B);
    str += strsprintf(_T("      c=<float> Mitchell-Netravali C parameter for bicubic. (default: %.2f)\n"), FILTER_DEFAULT_RESIZE_BICUBIC_C);
    str += _T("        Bicubic aliases: mitchell, catmull-rom, hermite.\n");
#if ENABLE_NVSDKNGX
    str += _T("      vsr-quality=<int> NGX video-super-resolution model/mode. (default: 1)\n");
    str += _T("        values: 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23\n");
    str += _T("        0 uses bicubic; 8-11 denoise, 12-15 deblur, 16-19 restore detail.\n");
    str += _T("        Modes 0 and 8-19 require VFX SDK 1.2 or later; streaming modes 21/23\n");
    str += _T("        require VFX SDK 1.3 or later and an Ampere or newer GPU.\n");
    str += _T("      vsr-strength=<float> NGX VSR strength; requires VFX SDK 1.3 or later. (default: 1.0, range: 0.0..1.0)\n");
#endif
#if ENCODER_QSV
    str += _T("      superres-mode=<string> QSV AI super-resolution mode. (default: default)\n");
    str += _T("        values: disabled, default, sharpen, artifactremoval\n");
    str += _T("      superres-algo=<string> QSV AI super-resolution algorithm. (default: default)\n");
    str += _T("        values: default, 1, 2\n");
#endif
        if (ENABLE_LIBPLACEBO) {
    str += _T("      pl-radius=<float> Libplacebo resampling radius; selected filter defines the unspecified radius. (default: derived, range: 0.0..16.0)\n");
    str += strsprintf(_T("      pl-clamp=<float> Clamp strength. (default: %.1f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_CLAMP);
    str += strsprintf(_T("      pl-taper=<float> Taper strength. (default: %.1f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_TAPER);
    str += strsprintf(_T("      pl-blur=<float> Blur strength. (default: %.1f, range: 0.0..100.0)\n"), FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_BLUR);
    str += strsprintf(_T("      pl-antiring=<float> Antiringing strength. (default: %.1f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_ANTIRING);
        }
    }
#else
    str += _T("\n  [resize] Resize and geometry\n   --vpp-resize <string> Resizing algorithm. (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_resize_help);
#endif
#if ENCODER_QSV
    str += _T("\n  [other] Other settings\n   --vpp-resize-mode <string> QSV hardware resizing mode. (default: auto)\n     values: auto, lowpower, quality\n");
#endif
#if ENABLE_VPP_FILTER_CONVOLUTION3D
    str += _T("\n  [denoise] Denoise\n   --vpp-convolution3d [<param1>=<value>][,<param2>=<value>][...]\n     Spatial and temporal convolution denoising.\n    params\n");
    str += _T("      matrix=<string> Convolution matrix (default: standard)\n");
    str += _T("        values: standard, simple\n");
    str += _T("      fast=<bool> Use the faster implementation (default: false)\n");
    str += strsprintf(_T("      ythresh=<int> Spatial luma threshold (default: %d, range: 0..255)\n"), FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_SPATIAL);
    str += strsprintf(_T("      cthresh=<int> Spatial chroma threshold (default: %d, range: 0..255)\n"), FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_SPATIAL);
    str += strsprintf(_T("      t_ythresh=<int> Temporal luma threshold (default: %d, range: 0..255)\n"), FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_TEMPORAL);
    str += strsprintf(_T("      t_cthresh=<int> Temporal chroma threshold (default: %d, range: 0..255)\n"), FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_TEMPORAL);
#endif
    str += _T("\n  [denoise] Denoise\n   --vpp-knn [<param1>=<value>][,<param2>=<value>][...]\n     K-nearest-neighbor denoising.\n    params\n");
    str += strsprintf(_T("      radius=<int> Spatial radius (default: %d)\n"), FILTER_DEFAULT_KNN_RADIUS);
    str += strsprintf(_T("      d=<int> Previous/next frames in the weighting window (default: %d, range: 0..2)\n"), FILTER_DEFAULT_KNN_D);
    str += strsprintf(_T("      strength=<float> Denoising strength (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_KNN_STRENGTH);
    str += strsprintf(_T("      lerp=<float> Original/blended balance; lower is stronger (default: %.2f)\n"), FILTER_DEFAULT_KNN_LERPC);
    str += strsprintf(_T("      th_lerp=<float> Edge preservation threshold (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_KNN_LERPC_THRESHOLD);
#if ENABLE_VPP_FILTER_NLMEANS
    str += _T("\n  [denoise] Denoise\n   --vpp-nlmeans [<param1>=<value>][,<param2>=<value>][...]\n     Non-local-means denoising.\n    params\n");
    str += strsprintf(_T("      sigma=<float> Noise standard deviation (default: %.3f, range: 0.0..)\n"), FILTER_DEFAULT_NLMEANS_FILTER_SIGMA);
    str += strsprintf(_T("      h=<float> Positive filtering strength (default: %.3f)\n"), FILTER_DEFAULT_NLMEANS_H);
#if ENCODER_NVENC
    str += strsprintf(_T("      patch=<int> Odd patch size (default: %d, range: 3..21)\n"), FILTER_DEFAULT_NLMEANS_PATCH_SIZE);
    str += strsprintf(_T("      search=<int> Odd search size (default: %d, range: 3..21)\n"), FILTER_DEFAULT_NLMEANS_SEARCH_SIZE);
#else
    str += strsprintf(_T("      patch=<int> Odd patch size (default: %d, range: 3..)\n"), FILTER_DEFAULT_NLMEANS_PATCH_SIZE);
    str += strsprintf(_T("      search=<int> Odd search size (default: %d, range: 3..)\n"), FILTER_DEFAULT_NLMEANS_SEARCH_SIZE);
#endif
    str += _T("      fp16=<string> Half-precision usage (default: blockdiff)\n");
    str += _T("        values: none, blockdiff, all\n");
    str += strsprintf(_T("      d=<int> Temporal radius (default: %d)\n"), FILTER_DEFAULT_NLMEANS_D);
    str += strsprintf(_T("      search_t=<int> Odd temporal search size (default: %d, range: 3..)\n"), FILTER_DEFAULT_NLMEANS_SEARCH_SIZE);
#endif
#if ENABLE_VPP_FILTER_PMD
    str += _T("\n  [denoise] Denoise\n   --vpp-pmd [<param1>=<value>][,<param2>=<value>][...]\n     Perona-Malik diffusion denoising.\n    params\n");
    str += strsprintf(_T("      apply_count=<int> Iteration count (default: %d)\n"), FILTER_DEFAULT_PMD_APPLY_COUNT);
    str += strsprintf(_T("      strength=<float> Denoising strength (default: %.2f, range: 0.0..100.0)\n"), FILTER_DEFAULT_PMD_STRENGTH);
    str += strsprintf(_T("      threshold=<float> Edge preservation threshold (default: %.2f, range: 0.0..255.0)\n"), FILTER_DEFAULT_PMD_THRESHOLD);
    str += strsprintf(_T("      useexp=<bool> Use the exponential diffusion function (default: %s)\n"), FILTER_DEFAULT_PMD_USE_EXP ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_DENOISE_BM3D
    str += _T("\n  [denoise] Denoise\n   --vpp-bm3d [<param1>=<value>][,<param2>=<value>][...]\n     Block matching and 3D collaborative denoising; a basic estimate followed by Wiener filtering.\n    params\n");
    str += _T("      profile=<string> Preset for block_step/group_size/bm_range; later parameters override it (default: fast)\n");
    str += _T("        values: fast, lc, np, high\n");
    str += strsprintf(_T("      sigma=<float> Noise standard deviation; 0 disables noise removal (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_BM3D_SIGMA);
    str += strsprintf(_T("      block_step=<int> Reference-patch stride (default: %d, range: 1..8)\n"), FILTER_DEFAULT_DENOISE_BM3D_BLOCK_STEP);
    str += strsprintf(_T("      group_size=<int> Maximum blocks per group (default: %d, range: 1..32)\n"), FILTER_DEFAULT_DENOISE_BM3D_GROUP_SIZE);
    str += strsprintf(_T("      bm_range=<int> Block-match window half-side (default: %d, range: 1..32)\n"), FILTER_DEFAULT_DENOISE_BM3D_BM_RANGE);
    str += strsprintf(_T("      radius=<int> Temporal radius; 0 is spatial-only (default: %d, range: 0..4)\n"), FILTER_DEFAULT_DENOISE_BM3D_RADIUS);
    str += strsprintf(_T("      chroma=<bool> Also denoise chroma (default: %s)\n"), FILTER_DEFAULT_DENOISE_BM3D_CHROMA ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_HQDN3D
    str += _T("\n  [denoise] Denoise\n   --vpp-hqdn3d [<param1>=<value>][,<param2>=<value>][...]\n     Spatial and temporal denoising.\n    params\n");
    str += strsprintf(_T("      luma_spatial=<float> Luma spatial strength (default: %.2f, range: 0.0..255.0)\n"), FILTER_DEFAULT_HQDN3D_LUMA_SPATIAL);
    str += strsprintf(_T("      chroma_spatial=<float> Chroma spatial strength (default: %.2f, range: 0.0..255.0)\n"), FILTER_DEFAULT_HQDN3D_CHROMA_SPATIAL);
    str += strsprintf(_T("      luma_temporal=<float> Luma temporal strength (default: %.2f, range: 0.0..255.0)\n"), FILTER_DEFAULT_HQDN3D_LUMA_TEMPORAL);
    str += strsprintf(_T("      chroma_temporal=<float> Chroma temporal strength (default: %.2f, range: 0.0..255.0)\n"), FILTER_DEFAULT_HQDN3D_CHROMA_TEMPORAL);
#endif
#if ENABLE_VPP_FILTER_DESCALE
    str += _T("\n  [resize] Resize and geometry\n   --vpp-descale [<param1>=<value>][,<param2>=<value>][...]\n     Undo upscaling by solving the inverse system for the original scaler kernel.\n    params\n");
    str += _T("      kernel=<string> Original upscaler kernel (default: bicubic)\n");
    str += rgy_cmd_help_values(list_vpp_descale_kernel);
    str += _T("      width=<int> Native width; 0 leaves it unspecified (default: 0)\n");
    str += _T("      height=<int> Native height; 0 leaves it unspecified (default: 0)\n");
    str += strsprintf(_T("      b=<float> Bicubic B parameter (default: %.2f)\n"), FILTER_DEFAULT_DESCALE_BICUBIC_B);
    str += strsprintf(_T("      c=<float> Bicubic C parameter (default: %.2f)\n"), FILTER_DEFAULT_DESCALE_BICUBIC_C);
    str += strsprintf(_T("      src_left=<float> Horizontal sub-pixel offset (default: %.2f)\n"), FILTER_DEFAULT_DESCALE_SRC_LEFT);
    str += strsprintf(_T("      src_top=<float> Vertical sub-pixel offset (default: %.2f)\n"), FILTER_DEFAULT_DESCALE_SRC_TOP);
    str += _T("      src_width=<float> Fractional active source width; 0 disables it (default: 0.0)\n");
    str += _T("      src_height=<float> Fractional active source height; 0 disables it (default: 0.0)\n");
    str += _T("      border_handling=<string> Border extension method (default: mirror)\n");
    str += rgy_cmd_help_values(list_vpp_descale_border);
    str += _T("      auto=<bool> Enable kernel and native-resolution search (default: false)\n");
    str += _T("      search_min=<int> Minimum candidate height is half the input height (default: derived)\n");
    str += _T("      search_max=<int> Maximum candidate height is input height minus one (default: derived)\n");
    str += strsprintf(_T("      search_step=<int> Fine-pass step in pixels (default: %d)\n"), FILTER_DEFAULT_DESCALE_SEARCH_STEP);
    str += strsprintf(_T("      detect_frames=<int> Frames averaged before locking (default: %d)\n"), FILTER_DEFAULT_DESCALE_DETECT_FRAMES);
    str += _T("      show_scores=<bool> Log candidate error scores (default: false)\n");
#endif
#if ENABLE_VPP_FILTER_ANIME4K
    str += _T("\n  [resize] Resize and geometry\n   --vpp-anime4k-shader [<param1>=<value>][,<param2>=<value>][...]\n     Anime4K GLSL luma refinement and upscaling.\n    params\n");
    str += _T("      mode=<string> Processing variant (default: ani4k_original)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_mode);
    str += strsprintf(_T("      scale=<int> 1 refines, 2 upscales; some modes select their own scale (default: %d)\n"), FILTER_DEFAULT_ANIME4K_SCALE);
    str += _T("        values: 1, 2\n");
    str += strsprintf(_T("      strength=<float> Refinement strength; deblur promotes an omitted value to 1.0 (default: %.2f)\n"), FILTER_DEFAULT_ANIME4K_STRENGTH);
    str += _T("      prefilter_denoise=<string> Denoise luma before refinement (default: off)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_denoise);
    str += _T("      darken=<string> Line darkening pass (default: off)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_darken);
    str += _T("      thin=<string> Line thinning pass (default: off)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_thin);
    str += _T("      denoise=<string> Denoise pass after refinement (default: off)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_denoise);
    str += _T("      denoise_intensity=<float> Advanced denoising adjustment (default: 0.1)\n");
    str += _T("      denoise_spatial=<float> Advanced denoising adjustment (default: 1.0)\n");
    str += _T("      denoise_curve=<float> Advanced denoising adjustment (default: 1.0)\n");
    str += _T("      denoise_hist_reg=<float> Advanced denoising adjustment (default: -1.0)\n");
    str += _T("      clamp_highlights=<bool> Clamp highlights to the local source maximum (default: false)\n");
    str += _T("      antiring=<float> Clamp upscaled luma to the source min/max envelope (default: 0.0, range: 0.0..1.0)\n");
    str += _T("      chroma_resize=<string> Chroma scaler; joint is luma-guided (default: spline36)\n");
    str += rgy_cmd_help_values(list_vpp_anime4k_chroma_resize);
    str += _T("      chroma=<bool> Resize chroma at scale=2; scale=1 passes it through (default: true)\n");
    str += _T("      out_res=<WxH> Final resolution defaults to the scaled source; one negative axis preserves aspect (default: derived)\n");
    str += _T("      resize=<string> Final-resolution resampler (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_resize);
#endif
#if ENABLE_VPP_FILTER_ONNX
    str += _T("\n  [resize] Resize and geometry\n   --vpp-lenscorrection [<param1>=<value>][,<param2>=<value>][...]\n     Correct radial lens distortion and corner brightness.\n    params\n");
    str += _T("      k1=<float> First radial distortion coefficient (default: 0.0)\n");
    str += _T("      k2=<float> Second radial distortion coefficient (default: 0.0)\n");
    str += _T("      cx=<float> Horizontal correction centre (default: 0.5)\n");
    str += _T("      cy=<float> Vertical correction centre (default: 0.5)\n");
    str += _T("      vignette=<float> Corner gain is 1+vignette; positive removes falloff (default: 0.0, range: -1.0..4.0)\n");
    str += _T("\n  [resize] Resize and geometry\n   --vpp-v360 [<param1>=<value>][,<param2>=<value>][...]\n     Convert between panorama and perspective projections.\n    params\n");
    str += _T("      in=<string> Input projection (default: equirect)\n");
    str += rgy_cmd_help_values(list_vpp_v360_proj);
    str += _T("      out=<string> Output projection (default: flat)\n");
    str += rgy_cmd_help_values(list_vpp_v360_proj);
    str += _T("      yaw=<float> Yaw angle in degrees (default: 0.0)\n");
    str += _T("      pitch=<float> Pitch angle in degrees (default: 0.0)\n");
    str += _T("      roll=<float> Roll angle in degrees (default: 0.0)\n");
    str += _T("      in_hfov=<float> Input horizontal field of view (default: 90.0)\n");
    str += _T("      h_fov=<float> Output horizontal field of view (default: 90.0)\n");
    str += _T("      w=<int> Output width; 0 follows the input (default: 0)\n");
    str += _T("      h=<int> Output height; 0 follows the input (default: 0)\n");
    str += _T("\n  [ai] AI\n   --vpp-onnx [<param1>=<value>][,<param2>=<value>][...]\n     Run an ONNX model directly with ONNX Runtime.\n    params\n");
    str += _T("      model=<path> Registered model name or ONNX model path\n");
#if ENABLE_OPENVINO
    str += _T("      device=<string> OpenVINO device; NPU requires a compatible runtime (default: GPU.0)\n");
    str += _T("        values: GPU.0, GPU, CPU, AUTO, NPU\n");
    str += _T("      precision=<string> Inference precision (default: auto)\n");
    str += _T("        values: auto, fp16, fp32\n");
#endif
#if ENCODER_NVENC
    str += _T("      provider=<string> Execution provider; auto prefers TensorRT and falls back to CUDA (default: auto)\n");
    str += _T("        values: auto, cuda, tensorrt\n");
    str += _T("      prec=<string> TensorRT engine precision; the CUDA provider runs fp32 (default: auto)\n");
    str += _T("        values: auto, fp16, fp32\n");
#endif
    str += _T("      colormatrix=<string> Input YUV/RGB conversion matrix (default: auto)\n");
    str += _T("        values: auto, auto_res, smpte170m, bt470bg, bt709, bt2020nc\n");
    str += _T("      colormatrix_out=<string> Output RGB/YUV matrix; auto follows the input (default: auto)\n");
    str += _T("        values: auto, auto_res, smpte170m, bt470bg, bt709, bt2020nc\n");
    str += _T("      colorrange=<string> Input color range; auto uses limited range (default: auto)\n");
    str += _T("        values: auto, tv, limited, pc, full\n");
    str += _T("      colorspace=<string> Model color space for 3-channel inputs (default: rgb)\n");
    str += _T("        values: rgb, ycbcr\n");
    str += _T("      noise=<int> Noise-model sigma (default: 15, range: 0..255)\n");
    str += _T("      frames=<int> Positive odd temporal window size (default: 1, range: 1..)\n");
    str += _T("      mask=<path> Grayscale mask for a two-input model; white is processed\n");
    str += _T("      out_res=<WxH> Final resolution defaults to the network output; one negative axis preserves aspect (default: derived)\n");
    str += _T("      resize=<string> Final-resolution resampler (default: auto)\n");
    str += rgy_cmd_help_values(list_vpp_resize);
    str += _T("\n  [other] Other settings\n   --vpp-onnx-model-dir <path> Directory containing models.json for registered ONNX models.\n");
#if ENABLE_VPP_FILTER_RIFE_OV
    str += _T("\n  [framerate] Frame rate and interpolation\n   --vpp-rife-ov [<param1>=<value>][,<param2>=<value>][...]\n     RIFE v4.x frame interpolation.\n    params\n");
    str += _T("      model=<path> Registered RIFE model name or ONNX path; names use rife_ov_models.json\n");
    str += _T("      multi=<int> Frame-rate multiplier (default: 2, range: 2..)\n");
    str += _T("      fps=<rate> Positive ratio or decimal; overrides multi and otherwise follows input rate times multi (default: derived)\n");
#if ENABLE_OPENVINO
    str += _T("      device=<string> OpenVINO inference device (default: GPU.0)\n");
    str += _T("        values: GPU.0, GPU, CPU, AUTO, NPU\n");
#endif
    str += _T("      colormatrix=<string> Conversion matrix (default: auto)\n");
    str += _T("        values: auto, bt601, bt709, bt2020\n");
    str += _T("      colorrange=<string> Conversion range (default: auto)\n");
    str += _T("        values: auto, tv, limited, pc, full\n");
#endif
#if ENABLE_VPP_FILTER_ONNX_DEINT
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-onnx-deint [<param1>=<value>][,<param2>=<value>][...]\n     Model-based deinterlacing; cannot be combined with vhs-restore.\n    params\n");
    str += _T("      enable=<bool> Enable the filter (default: true)\n");
    str += _T("      model=<path> Registered model name from onnx_deint_models.json\n");
    str += _T("      mode=<string> bob doubles the frame rate (default: bob)\n");
    str += rgy_cmd_help_values(list_vpp_onnx_deint_mode);
    str += _T("      device=<string> Inference device (default: GPU.0)\n");
    str += _T("      precision=<string> Inference precision (default: fp32)\n");
    str += _T("        values: fp32, auto\n");
    str += _T("      colormatrix=<string> Conversion matrix (default: auto)\n");
    str += _T("        values: auto, auto_res, smpte170m, bt470bg, bt709, bt2020nc\n");
    str += _T("      colorrange=<string> Conversion range (default: auto)\n");
    str += _T("        values: auto, tv, limited, pc, full\n");
    str += _T("\n  [deinterlace] Deinterlace\n   --vpp-vhs-restore [<param1>=<value>][,<param2>=<value>][...]\n     Restore VHS fields, then apply GPU Bwdif; cannot be combined with onnx-deint.\n    params\n");
    str += _T("      enable=<bool> Enable the filter (default: true)\n");
    str += _T("      model=<path> Registered vhs_restore_pal or vhs_restore_secam model; names use onnx_deint_models.json\n");
    str += _T("      mode=<string> bob doubles the frame rate (default: bob)\n");
    str += rgy_cmd_help_values(list_vpp_onnx_deint_mode);
#if ENABLE_OPENVINO
    str += _T("      device=<string> OpenVINO device; shared-GPU operation is required (default: GPU.0)\n");
    str += _T("        values: GPU.0, GPU, AUTO\n");
#endif
    str += _T("      precision=<string> Inference precision (default: fp32)\n");
    str += _T("        values: fp32, auto\n");
    str += _T("      colormatrix=<string> Conversion matrix (default: auto)\n");
    str += _T("        values: auto, auto_res, smpte170m, bt470bg, bt709, bt2020nc\n");
    str += _T("      colorrange=<string> Conversion range (default: auto)\n");
    str += _T("        values: auto, tv, limited, pc, full\n");
#endif
#if ENABLE_OPENVINO || ENCODER_NVENC
    str += _T("\n  [other] Other settings\n   --vpp-onnx-cache-dir <path> Cache compiled OpenVINO models or TensorRT engines in this directory. (default: auto)\n");
#endif
#endif
#if ENABLE_VPP_FILTER_SMOOTH
    str += _T("\n  [denoise] Denoise\n   --vpp-smooth [<param1>=<value>][,<param2>=<value>][...]\n     Smooth denoising.\n    params\n");
    str += strsprintf(_T("      quality=<int> Quality; larger values are slower (default: %d, range: 1..6)\n"), FILTER_DEFAULT_SMOOTH_QUALITY);
    str += strsprintf(_T("      qp=<float> Filter strength (default: %.2f, range: 0.0..63.0)\n"), static_cast<double>(FILTER_DEFAULT_SMOOTH_QP));
    str += _T("      prec=<string> Calculation precision (default: auto)\n");
    str += _T("        values: auto, fp16, fp32\n");
#endif
#if ENABLE_VPP_FILTER_DENOISE_DCT
    str += _T("\n  [denoise] Denoise\n   --vpp-denoise-dct [<param1>=<value>][,<param2>=<value>][...]\n     DCT-domain denoising.\n    params\n");
    str += _T("      step=<int> Calculation step; smaller is higher quality (default: 2)\n");
    str += _T("        values: 1, 2, 4, 8\n");
    str += strsprintf(_T("      sigma=<float> Noise level for the highest-frequency bins (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_DCT_SIGMA);
    str += _T("      sigma2=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += _T("      sigma3=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += _T("      sigma4=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += _T("      block_size=<int> Calculation block size (default: 8)\n");
    str += _T("        values: 8, 16\n");
#endif
#if ENABLE_VPP_FILTER_FFT3D
    str += _T("\n  [denoise] Denoise\n   --vpp-fft3d [<param1>=<value>][,<param2>=<value>][...]\n     FFT-domain spatial and temporal denoising.\n    params\n");
    str += strsprintf(_T("      sigma=<float> Noise level for the highest frequencies (default: %.2f, range: 0..100)\n"), FILTER_DEFAULT_DENOISE_FFT3D_SIGMA);
    str += _T("      sigma2=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += _T("      sigma3=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += _T("      sigma4=<float> Noise level follows sigma when omitted (default: derived)\n");
    str += strsprintf(_T("      amount=<float> Denoising amount (default: %.2f, range: 0..1)\n"), FILTER_DEFAULT_DENOISE_FFT3D_AMOUNT);
    str += _T("      block_size=<int> Calculation block size (default: 32)\n");
    str += _T("        values: 8, 16, 32, 64\n");
    str += strsprintf(_T("      overlap=<float> Block overlap (default: %.2f, range: 0.2..0.8)\n"), FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP);
    str += _T("      method=<int> Denoising method (default: 0)\n");
    str += _T("        values: 0, 1\n");
    str += strsprintf(_T("      temporal=<int> Enable temporal filtering (default: %d)\n"), FILTER_DEFAULT_DENOISE_FFT3D_TEMPORAL);
    str += _T("      bt=<int> Temporal radius follows temporal; 1 is spatial, -1 is sharpen-only (default: derived)\n");
    str += _T("        values: -1, 1, 2, 3, 4\n");
    str += _T("      sharpen=<float> Frequency-domain luma sharpening; 0 disables it (default: 0.0)\n");
    str += strsprintf(_T("      scutoff=<float> Relative sharpening cutoff (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_FFT3D_SCUTOFF);
    str += strsprintf(_T("      svr=<float> Vertical sharpening ratio; 0 disables it (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_FFT3D_SVR);
    str += strsprintf(_T("      smin=<float> Noise-margin limit (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_FFT3D_SMIN);
    str += strsprintf(_T("      smax=<float> Halo-margin limit (default: %.2f)\n"), FILTER_DEFAULT_DENOISE_FFT3D_SMAX);
    str += _T("      degrid=<float> Grid compensation; 0 disables it, 1 is full compensation (default: 0.0)\n");
    str += _T("      signorm=<bool> Normalize sigma and limits to noise-power units (default: false)\n");
    str += _T("      prec=<string> Calculation precision (default: auto)\n");
    str += _T("        values: auto, fp16, fp32\n");
#endif
#if ENABLE_VPP_FILTER_DEGRAIN
    str += _T("\n  [denoise] Denoise\n   --vpp-degrain [<param1>=<value>][,<param2>=<value>][...]\n     Motion-compensated degraining and motion-analysis output.\n    params\n");
    str += _T("      preset=<string> Surface preset (default: custom)\n");
    str += _T("        values: custom, auto\n");
    str += strsprintf(_T("      mode=<string> Output mode (default: %s)\n"), get_cx_desc(list_vpp_degrain_mode, (int)FILTER_DEFAULT_DEGRAIN_MODE));
    str += rgy_cmd_help_values(list_vpp_degrain_mode);
    str += _T("      stage=<string> Processing stage (default: auto)\n");
    str += _T("        values: auto, tr1, tr2\n");
    str += _T("      tr=<int> Auto-preset radius follows stage and delta when omitted (default: derived)\n");
    str += _T("        values: 1, 2\n");
    str += strsprintf(_T("      blksize=<int> Motion block size (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_BLKSIZE);
    str += strsprintf(_T("      search=<int> Search radius (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_SEARCH);
    str += strsprintf(_T("      thsad=<int> Luma SAD threshold (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_THSAD);
    str += strsprintf(_T("      thsadc=<int> Chroma SAD threshold (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_THSADC);
    str += strsprintf(_T("      thscd1=<int> Scene-change SAD threshold (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_THSCD1);
    str += strsprintf(_T("      thscd2=<int> Scene-change ratio in AviSynth units (default: %d, range: 0..255)\n"), FILTER_DEFAULT_DEGRAIN_THSCD2);
    str += strsprintf(_T("      pel=<int> Subpixel precision (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_PEL);
    str += _T("        values: 1, 2, 4\n");
    str += strsprintf(_T("      subpelinterp=<int> Subpixel interpolation mode (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_SUBPEL_INTERP);
    str += strsprintf(_T("      levels=<int> Analysis levels (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_LEVELS);
    str += strsprintf(_T("      overlap=<int> Overlap in pixels (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_OVERLAP);
    str += strsprintf(_T("      delta=<int> Temporal radius (default: %d, range: 1..5)\n"), FILTER_DEFAULT_DEGRAIN_DELTA);
    str += strsprintf(_T("      tr0=<int> Reference temporal prefilter radius (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_TR0);
    str += _T("        values: 0, 1, 2\n");
    str += strsprintf(_T("      rep0=<int> Reference repair mode (default: %d, range: 0..7)\n"), FILTER_DEFAULT_DEGRAIN_REP0);
    str += strsprintf(_T("      search_refine=<int> Reference pre/postprocessing mode (default: %d, range: 0..3)\n"), FILTER_DEFAULT_DEGRAIN_SEARCH_REFINE);
    str += _T("      mv_spatial_refine=<int|auto> Spatial motion-vector refinement count (default: auto)\n");
    str += _T("        values: auto\n");
    str += strsprintf(_T("      searchparam=<int> Motion search parameter (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_SEARCHPARAM);
    str += strsprintf(_T("      pelsearch=<int> Motion search parameter (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_PELSEARCH);
    str += _T("      search_early_sad=<int|off> Level-0 early SAD threshold in 8x8/8-bit units (default: off)\n");
    str += _T("        values: off\n");
    str += _T("      spatial_early_sad=<int|off> Level-1 refinement skip threshold in 8x8/8-bit units (default: off)\n");
    str += _T("        values: off\n");
    str += strsprintf(_T("      truemotion=<bool> Use true-motion tuning (default: %s)\n"), FILTER_DEFAULT_DEGRAIN_TRUEMOTION ? _T("true") : _T("false"));
    str += strsprintf(_T("      lambda=<int> Motion penalty strength (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_LAMBDA);
    str += strsprintf(_T("      lsad=<int> Large-SAD threshold (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_LSAD);
    str += strsprintf(_T("      pnew=<int> New-vector penalty (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_PNEW);
    str += strsprintf(_T("      plevel=<int> Predictor level (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_PLEVEL);
    str += strsprintf(_T("      globalmotion=<bool> Enable global motion tuning (default: %s)\n"), FILTER_DEFAULT_DEGRAIN_GLOBALMOTION ? _T("true") : _T("false"));
    str += strsprintf(_T("      dct=<int> DCT search mode (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_DCT);
    str += strsprintf(_T("      useflag=<int> Reference direction: 0 both, 1 backward, 2 forward (default: %d)\n"), FILTER_DEFAULT_DEGRAIN_USEFLAG);
    str += strsprintf(_T("      chroma=<bool> Include chroma in motion/SAD analysis (default: %s)\n"), FILTER_DEFAULT_DEGRAIN_CHROMA ? _T("true") : _T("false"));
    str += _T("      binomial=<bool|auto> Binomial prefilter path (default: auto)\n");
    str += _T("        values: auto, false, true\n");
    str += strsprintf(_T("      tv_range=<bool> Expand motion analysis from TV to PC range (default: %s)\n"), FILTER_DEFAULT_DEGRAIN_TV_RANGE ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_MSMOOTH
    str += _T("\n  [denoise] Denoise\n   --vpp-msmooth [<param1>=<value>][,<param2>=<value>][...]\n     Detail-preserving smoothing.\n    params\n");
    str += strsprintf(_T("      strength=<int> Smoothing iterations (default: %d, range: 0..20)\n"), FILTER_DEFAULT_MSMOOTH_STRENGTH);
    str += strsprintf(_T("      threshold=<float> Luma edge threshold (default: %.1f, range: 0.0..255.0)\n"), FILTER_DEFAULT_MSMOOTH_THRESHOLD);
    str += _T("      threshold_c=<float> Chroma edge threshold; follows threshold when omitted (default: derived)\n");
    str += strsprintf(_T("      highq=<bool> High-quality edge detection (default: %s)\n"), FILTER_DEFAULT_MSMOOTH_HIGHQ ? _T("true") : _T("false"));
    str += strsprintf(_T("      mask=<bool> Output only the edge mask (default: %s)\n"), FILTER_DEFAULT_MSMOOTH_MASK ? _T("true") : _T("false"));
#endif
    str += _T("\n  [overlay] Overlay\n   --vpp-subburn [<param1>=<value>][,<param2>=<value>][...]\n     Burn subtitles into the video.\n    params\n");
    str += _T("      track=<int> Input subtitle track ID (default: 0)\n");
    str += _T("      filename=<path> Subtitle file path\n");
    str += _T("      charcode=<string> Character encoding\n");
    str += _T("      shaping=<string> Text shaping quality (default: complex)\n");
    str += rgy_cmd_help_values(list_vpp_ass_shaping);
    str += _T("      scale=<float> Bitmap subtitle scale is chosen automatically when omitted (default: derived)\n");
    str += _T("      transparency=<float> Additional transparency (default: 0.0, range: 0.0..1.0)\n");
    str += strsprintf(_T("      brightness=<float> Subtitle brightness (default: %.1f, range: -1.0..1.0)\n"), FILTER_DEFAULT_TWEAK_BRIGHTNESS);
    str += strsprintf(_T("      contrast=<float> Subtitle contrast (default: %.1f, range: -2.0..2.0)\n"), FILTER_DEFAULT_TWEAK_CONTRAST);
    str += _T("      vid_ts_offset=<bool> Apply the first-video-timestamp offset; always on with track (default: true)\n");
    str += _T("      ts_offset=<float> Additional subtitle timestamp offset in seconds (default: 0.0)\n");
    str += _T("      fontsdir=<path> Directory containing subtitle fonts\n");
    str += _T("      forced_subs_only=<bool> Render only forced subtitles (default: false)\n");
#if ENABLE_LIBPLACEBO
    str += _T("\n  [resize] Resize and geometry\n   --vpp-libplacebo-shader [<param1>=<value>][,<param2>=<value>][...]\n     Apply a custom libplacebo shader.\n    params\n");
    str += _T("      shader=<path> Shader file path\n");
    str += _T("      custom=<name=value> Value for a shader //!PARAM\n");
    str += _T("      res=<int>x<int> Output resolution; positive dimensions\n");
    str += strsprintf(_T("      csp=<string> Input CSP passed to libplacebo (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_shader_csp, FILTER_DEFAULT_LIBPLACEBO_SHADER_CSP));
    str += rgy_cmd_help_values(list_vpp_libplacebo_shader_csp);
    str += strsprintf(_T("      chromaloc=<string> Chroma location used to derive chroma shift (default: %s)\n"), get_cx_desc(list_chromaloc_str, FILTER_DEFAULT_LIBPLACEBO_SHADER_CHROMALOC));
    str += rgy_cmd_help_values(list_chromaloc_str);
    str += strsprintf(_T("      colorsystem=<string> Color matrix (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_colorsystem, FILTER_DEFAULT_LIBPLACEBO_SHADER_COLORSYSTEM));
    str += rgy_cmd_help_values(list_vpp_libplacebo_colorsystem);
    str += strsprintf(_T("      transfer=<string> Output transfer function (default: %s)\n"), get_cx_desc(list_vpp_libplacebo_tone_mapping_transfer, FILTER_DEFAULT_LIBPLACEBO_SHADER_TRANSFER));
    str += rgy_cmd_help_values(list_vpp_libplacebo_tone_mapping_transfer);
    str += strsprintf(_T("      resampler=<string> Scaling filter function (default: %s)\n"), FILTER_DEFAULT_LIBPLACEBO_SHADER_RESAMPLER_NAME);
    str += rgy_cmd_help_values(get_libplacebo_only_resize_list().data());
    str += _T("      radius=<float> Function radius follows the chosen libplacebo filter (default: derived)\n");
    str += strsprintf(_T("      clamp=<float> Clamp coefficient (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_SHADER_CLAMP);
    str += strsprintf(_T("      taper=<float> Taper coefficient (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_SHADER_TAPER);
    str += strsprintf(_T("      blur=<float> Blur coefficient (default: %.2f, range: 0.0..100.0)\n"), FILTER_DEFAULT_LIBPLACEBO_SHADER_BLUR);
    str += strsprintf(_T("      antiring=<float> Antiring coefficient (default: %.2f, range: 0.0..1.0)\n"), FILTER_DEFAULT_LIBPLACEBO_SHADER_ANTIRING);
    str += _T("      linear=<bool> Linearize the image before processing (default: false)\n");
    str += _T("      sigmoid=<bool> Use sigmoidization with linear=true on the upscaling path (default: false)\n");
    str += _T("      sigmoid_center=<float> Uses the libplacebo default centre (default: derived)\n");
    str += _T("      sigmoid_slope=<float> Uses the libplacebo default slope (default: derived)\n");
#endif
#if ENABLE_VPP_FILTER_UNSHARP
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-unsharp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable unsharp filter.\n")
        _T("    params\n")
        _T("      radius=<int>              filter range for edge detection (default: %d, range: 1..9)\n")
        _T("      weight=<float>            strength of filter (default: %.2f, range: 0..10)\n")
        _T("      threshold=<float>         min brightness change to be sharpened (default: %.2f, range: 0..255)\n"),
        FILTER_DEFAULT_UNSHARP_RADIUS, FILTER_DEFAULT_UNSHARP_WEIGHT, FILTER_DEFAULT_UNSHARP_THRESHOLD);
#endif
#if ENABLE_VPP_FILTER_VINVERSE
    str += strsprintf(_T("\n")
        _T("  [deinterlace] Deinterlace\n")
        _T("   --vpp-vinverse [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     remove residual interlaced combing after deinterlace.\n")
        _T("    params\n")
        _T("      mode=<string>             filter mode (default: vinverse)\n")
        _T("        values: vinverse, vinverse2\n")
        _T("      sstr=<float>              contra reference strength (default: %.2f, range: 0.0..8.0)\n")
        _T("      amnt=<float>              max delta per pixel in 8-bit scale (default: %.1f, range: 0.0..255.0)\n")
        _T("      scl=<float>               soft clip scale (default: %.2f, range: 0.0..4.0)\n")
        _T("      thr=<float>               untouched residual threshold in 8-bit scale (default: %.1f, range: 0.0..255.0)\n")
        _T("      chroma=<bool>             process chroma planes (default: %s)\n"),
        FILTER_DEFAULT_VINVERSE_SSTR, FILTER_DEFAULT_VINVERSE_AMNT,
        FILTER_DEFAULT_VINVERSE_SCL, FILTER_DEFAULT_VINVERSE_THR,
        FILTER_DEFAULT_VINVERSE_CHROMA ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_CHROMASHIFT
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-chromashift [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     shift chroma planes for chroma/luma alignment correction.\n")
        _T("    params\n")
        _T("      x=<float>                 horizontal shift in luma pixels (default: %.2f, range: -4.0..4.0)\n")
        _T("      y=<float>                 vertical shift in luma pixels (default: %.2f, range: -4.0..4.0)\n")
        _T("      show=<string>   diagnostic output (default: normal)\n")
        _T("        values: normal, laplacian\n")
        _T("      auto=<bool>               auto detect shift from early frames (default: %s)\n")
        _T("      auto_frames=<int>         accepted analysis frames (default: %d, range: 1..100)\n")
        _T("      auto_min_pairs=<int>      minimum zero-crossing pairs per frame (default: %d, range: 10..10000)\n"),
        FILTER_DEFAULT_CHROMASHIFT_X, FILTER_DEFAULT_CHROMASHIFT_Y,
        FILTER_DEFAULT_CHROMASHIFT_AUTO ? _T("true") : _T("false"),
        FILTER_DEFAULT_CHROMASHIFT_AUTO_FRAMES, FILTER_DEFAULT_CHROMASHIFT_AUTO_MIN_PAIRS);
#endif
#if ENABLE_VPP_FILTER_DEBLOCK
    str += strsprintf(_T("\n")
        _T("  [denoise] Denoise\n")
        _T("   --vpp-deblock [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     H.264 style spatial deblocking filter.\n")
        _T("    params\n")
        _T("      qp=<int>                  filter strength QP (default: %d, range: 0..51)\n")
        _T("      alpha=<int>               alpha offset (default: %d, range: -6..6)\n")
        _T("      beta=<int>                beta offset (default: %d, range: -6..6)\n")
        _T("      chroma=<bool>             process planar chroma planes (default: %s)\n")
        _T("      grid=<int>                block boundary spacing (default: %d)\n")
        _T("                                  4: H.264/AVC, 8: MPEG-2/MPEG-4 Part 2\n")
        _T("                                  aliases: h264, avc, mpeg2, mpeg4\n"),
        FILTER_DEFAULT_DEBLOCK_QP, FILTER_DEFAULT_DEBLOCK_ALPHA, FILTER_DEFAULT_DEBLOCK_BETA,
        FILTER_DEFAULT_DEBLOCK_CHROMA ? _T("true") : _T("false"),
        FILTER_DEFAULT_DEBLOCK_GRID);
#endif
#if ENABLE_VPP_FILTER_DEFLICKER
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-deflicker [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     temporal brightness stabilization filter.\n")
        _T("    params\n")
        _T("      strength=<float>          correction blend strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      damping=<float>           temporal damping of correction (default: %.2f, range: 0.0..1.0)\n")
        _T("      scene_threshold=<float>   scene change threshold (default: %.2f, range: 0.5..5.0)\n")
        _T("      frames=<int>              rolling analysis window (default: %d, range: 5..300)\n")
        _T("      predictor=<bool>          use predictor-corrector pass (default: %s)\n")
        _T("      chroma=<bool>             process chroma planes (default: %s)\n"),
        FILTER_DEFAULT_DEFLICKER_STRENGTH, FILTER_DEFAULT_DEFLICKER_DAMPING,
        FILTER_DEFAULT_DEFLICKER_SCENE_THRESHOLD, FILTER_DEFAULT_DEFLICKER_FRAMES,
        FILTER_DEFAULT_DEFLICKER_PREDICTOR ? _T("true") : _T("false"),
        FILTER_DEFAULT_DEFLICKER_CHROMA ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_STAB
    str += strsprintf(_T("\n")
        _T("  [resize] Resize and geometry\n")
        _T("   --vpp-stab [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     camera-shake stabilisation via phase correlation.\n")
        _T("    params\n")
        _T("      strength=<float>          correction strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      damping=<float>           temporal smoothing of detected shift (default: %.2f, range: 0.0..1.0)\n")
        _T("      trust=<float>             minimum normalized correlation peak (default: %.2f, range: 0.0..1.0)\n")
        _T("      max_shift=<float>         maximum shift in luma pixels (default: %.1f, range: 1.0..256.0)\n")
        _T("      border=<string> Border mode for warped pixels (default: black)\n")
        _T("        values: black, clamp, mirror\n")
        _T("\n"),
        FILTER_DEFAULT_STAB_STRENGTH,
        FILTER_DEFAULT_STAB_DAMPING,
        FILTER_DEFAULT_STAB_TRUST_THRESHOLD,
        FILTER_DEFAULT_STAB_MAX_SHIFT);
#endif
#if ENABLE_VPP_FILTER_COLORFIX
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-colorfix [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Correct color balance manually or from frame statistics.\n")
        _T("    params\n")
        _T("      mode=<string> Correction mode (default: manual)\n")
        _T("        values: manual, auto, gray\n")
        _T("      space=<string> Working color space (default: auto)\n")
        _T("        values: auto, rgb, yuv\n")
        _T("      matrix=<string> YUV/RGB conversion matrix (default: auto)\n")
        _T("        values: auto, bt601, bt709, bt2020\n")
        _T("      white=<rrggbb> Manual white point (default: ffffff)\n")
        _T("      black=<rrggbb> Manual black point (default: 000000)\n")
        _T("      frames=<int> Analysis frames for auto/gray (default: %d, range: 10..5000)\n")
        _T("      strength=<float> Correction strength for auto/gray (default: %.2f, range: 0.0..1.0)\n")
        _T("      variance_threshold=<float> Positive flash/fade rejection threshold (default: %.2f)\n")
        _T("      temperature=<int> Correct source light to %d K in manual RGB mode; 0 disables correction (default: %d, range: %d..%d)\n"),
        FILTER_DEFAULT_COLORFIX_FRAMES, FILTER_DEFAULT_COLORFIX_STRENGTH, FILTER_DEFAULT_COLORFIX_VARIANCE_THRESHOLD, FILTER_REF_COLORFIX_TEMPERATURE, FILTER_DEFAULT_COLORFIX_TEMPERATURE, FILTER_MIN_COLORFIX_TEMPERATURE, FILTER_MAX_COLORFIX_TEMPERATURE);
#endif
#if ENABLE_VPP_FILTER_EDGELEVEL
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-edgelevel [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     edgelevel filter to enhance edge.\n")
        _T("    params\n")
        _T("      strength=<float>          strength (default: %.1f, range: -31..31)\n")
        _T("      threshold=<float>         threshold to ignore noise (default: %.1f, range: 0..255)\n")
        _T("      black=<float>             allow edge to be darker on edge enhancement (default: %.1f, range: 0..31)\n")
        _T("\n")
        _T("      white=<float>             allow edge to be brighter on edge enhancement (default: %.1f, range: 0..31)\n")
        _T("\n"),
        FILTER_DEFAULT_EDGELEVEL_STRENGTH, FILTER_DEFAULT_EDGELEVEL_THRESHOLD, FILTER_DEFAULT_EDGELEVEL_BLACK, FILTER_DEFAULT_EDGELEVEL_WHITE);
#endif
#if ENABLE_VPP_FILTER_DEHALO
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-dehalo [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     halo removal filter. Applies correction to luma and copies chroma unchanged.\n")
        _T("    params\n")
        _T("      mode=<string>             filter mode (default: legacy)\n")
        _T("        values: legacy, alpha\n")
        _T("      rx=<float>                horizontal halo radius (default: %.2f, range: 0.5..10.0)\n")
        _T("      ry=<float>                vertical halo radius (default: %.2f, range: 0.5..10.0)\n")
        _T("      darkstr=<float>           bright-halo darkening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      brightstr=<float>         dark-halo brightening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      lowsens=<int>             sensitivity ramp lower anchor (default: %d, range: 0..100)\n")
        _T("      highsens=<int>            sensitivity ramp upper anchor (default: %d, range: 0..100)\n")
        _T("      ss=<float>                supersampling ratio (default: %.2f, range: 1.0..4.0)\n")
        _T("      search_rade=<int|auto>         alpha mode range expand radius (default: auto, range: 1..10)\n")
        _T("        values: auto\n")
        _T("      search_radi=<int>         alpha mode range inpand radius; follows search_rade (default: derived)\n"),
        FILTER_DEFAULT_DEHALO_RX, FILTER_DEFAULT_DEHALO_RY,
        FILTER_DEFAULT_DEHALO_DARKSTR, FILTER_DEFAULT_DEHALO_BRIGHTSTR,
        FILTER_DEFAULT_DEHALO_LOWSENS, FILTER_DEFAULT_DEHALO_HIGHSENS,
        FILTER_DEFAULT_DEHALO_SS);
#endif
#if ENABLE_VPP_FILTER_FINEDEHALO
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-finedehalo [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     fine halo removal filter. Applies correction to luma and copies chroma unchanged.\n")
        _T("    params\n")
        _T("      mode=<string>             internal dehalo mode (default: alpha)\n")
        _T("        values: legacy, alpha\n")
        _T("      rx=<float>                horizontal halo radius (default: %.2f, range: 0.5..10.0)\n")
        _T("      ry=<float>                vertical halo radius (default: %.2f, range: 0.5..10.0)\n")
        _T("      darkstr=<float>           bright-halo darkening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      brightstr=<float>         dark-halo brightening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      lowsens=<int>             dehalo sensitivity lower anchor (default: %d, range: 0..100)\n")
        _T("      highsens=<int>            dehalo sensitivity upper anchor (default: %d, range: 0..100)\n")
        _T("      ss=<float>                supersampling ratio (default: %.2f, range: 1.0..4.0)\n")
        _T("      search_rade=<int>         alpha mode range expand radius (default: %d, range: 1..10)\n")
        _T("      search_radi=<int>         alpha mode range inpand radius; follows search_rade (default: derived)\n")
        _T("      thmi=<int>                edge mask lower threshold (default: %d, range: 0..255)\n")
        _T("      thma=<int>                edge mask upper threshold (default: %d, range: 0..255)\n")
        _T("      thlimi=<int>              light edge mask lower threshold (default: %d, range: 0..255)\n")
        _T("      thlima=<int>              light edge mask upper threshold (default: %d, range: 0..255)\n")
        _T("      showmask=<int>            debug mask output (default: %d, range: 0..4)\n")
        _T("                                1=outside, 2=shrink, 3=edges, 4=strong\n")
        _T("      excl=<bool>               enable exclusion zone (default: %s)\n")
        _T("      edgeproc=<float>          add strong edge mask to outside mask (default: %.2f, range: 0.0..1.0)\n")
        _T("      edge=<string>             edge operator (default: %s)\n")
        _T("        values: prewitt, sobel, scharr, kirsch, laplacian\n"),
        FILTER_DEFAULT_DEHALO_RX, FILTER_DEFAULT_DEHALO_RY,
        FILTER_DEFAULT_DEHALO_DARKSTR, FILTER_DEFAULT_FINEDEHALO_BRIGHTSTR,
        FILTER_DEFAULT_DEHALO_LOWSENS, FILTER_DEFAULT_DEHALO_HIGHSENS,
        FILTER_DEFAULT_DEHALO_SS,
        FILTER_DEFAULT_FINEDEHALO_SEARCH_RADIUS,
        FILTER_DEFAULT_FINEDEHALO_THMI, FILTER_DEFAULT_FINEDEHALO_THMA,
        FILTER_DEFAULT_FINEDEHALO_THLIMI, FILTER_DEFAULT_FINEDEHALO_THLIMA,
        FILTER_DEFAULT_FINEDEHALO_SHOWMASK,
        FILTER_DEFAULT_FINEDEHALO_EXCL ? _T("true") : _T("false"),
        FILTER_DEFAULT_FINEDEHALO_EDGEPROC,
        FILTER_DEFAULT_FINEDEHALO_EDGE);
#endif
#if ENABLE_VPP_FILTER_GUIDEDFILTER
    str += strsprintf(_T("\n")
        _T("  [denoise] Denoise\n")
        _T("   --vpp-guidedfilter [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     edge-preserving smoothing by self-guided image filtering.\n")
        _T("    params\n")
        _T("      radius=<int>              box-filter radius (default: %d, range: 1..32)\n")
        _T("      eps=<float>               smoothing strength (default: %.4f, range: 0.0001..1.0)\n")
        _T("      chroma=<bool>             filter chroma planes (default: false)\n"),
        FILTER_DEFAULT_GUIDEDFILTER_RADIUS, FILTER_DEFAULT_GUIDEDFILTER_EPS);
#endif
#if ENABLE_VPP_FILTER_CLAHE
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-clahe [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     contrast limited adaptive histogram equalization.\n")
        _T("    params\n")
        _T("      tiles_x=<int>             horizontal tile count (default: %d, range: 2..32)\n")
        _T("      tiles_y=<int>             vertical tile count (default: %d, range: 2..32)\n")
        _T("      slope=<float>             contrast clip ratio (default: %.1f, range: 1.0..40.0)\n"),
        FILTER_DEFAULT_CLAHE_TILES_X, FILTER_DEFAULT_CLAHE_TILES_Y, FILTER_DEFAULT_CLAHE_SLOPE);
#endif
#if ENABLE_VPP_FILTER_DEHAZE
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-dehaze [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     luminance dehazing based on the dark channel prior.\n")
        _T("    params\n")
        _T("      patch_radius=<int>        local minimum radius (default: %d, range: 1..15)\n")
        _T("      omega=<float>             haze removal strength (default: %.2f, range: 0.5..1.0)\n")
        _T("      t_floor=<float>           transmission lower bound (default: %.2f, range: 0.01..0.5)\n")
        _T("      atm_light=<float>         atmospheric light (default: %.2f, range: 0.1..1.0)\n"),
        FILTER_DEFAULT_DEHAZE_PATCH_RADIUS, FILTER_DEFAULT_DEHAZE_OMEGA,
        FILTER_DEFAULT_DEHAZE_T_FLOOR, FILTER_DEFAULT_DEHAZE_ATM_LIGHT);
#endif
#if ENABLE_VPP_FILTER_HQDERING
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-hqdering [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     DCT ringing reduction filter. Applies correction to luma and copies chroma unchanged.\n")
        _T("    params\n")
        _T("      mrad=<int>                ring mask expansion radius (default: %d, range: 1..3)\n")
        _T("      mthr=<int>                edge mask threshold (default: %d, range: 0..255)\n")
        _T("      sigma=<float>             Gaussian blur sigma (default: %.2f, range: 0.5..5.0)\n")
        _T("      showmask=<bool>           output effective mask only (default: %s)\n")
        _T("      protect=<bool>            protect original edge pixels (default: %s)\n")
        _T("      edge=<string>             edge operator (default: %s)\n")
        _T("        values: log, sobel, prewitt, scharr, kirsch, laplacian\n")
        _T("      thr=<int>                 limit for the change per pixel, 8-bit scale (default: 0)\n")
        _T("\n")
        _T("      elast=<float>             elastic falloff of thr (default: 2.0, range: 1.0..3.0)\n")
        _T("      darkthr=<int>             separate limit for darkening follows thr (default: derived)\n")
        _T("      minp=<int>                edge-core inpand iterations excluded from the (default: 0, range: 0..3)\n")
        _T("                                  ring mask\n")
        _T("      msmooth=<int>             ring mask smoothing iterations (default: 0, range: 0..3)\n")
        _T("      drrep=<int>               repair blurred clip: 0=off, 1=clamp to 3x3 min/max (default: 0)\n")
        _T("                                  of the source\n")
        _T("      sharp=<int>               contra-sharpening level (default: 0, range: 0..3)\n")
        _T("                                  restores line strength lost to the blur, limited\n")
        _T("                                  so it cannot re-introduce ringing.\n")
        _T("      planes=<string>           target planes (default: y)\n")
        _T("        values: y, u, v, y:u, y:v, u:v, y:u:v, all\n")
        _T("                                  all, or \":\"-separated list of y, u, v.\n"),
        FILTER_DEFAULT_HQDERING_MRAD, FILTER_DEFAULT_HQDERING_MTHR,
        FILTER_DEFAULT_HQDERING_SIGMA,
        FILTER_DEFAULT_HQDERING_SHOWMASK ? _T("true") : _T("false"),
        FILTER_DEFAULT_HQDERING_PROTECT ? _T("true") : _T("false"),
        FILTER_DEFAULT_HQDERING_EDGE);
#endif
#if ENABLE_VPP_FILTER_MSHARPEN
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-msharpen [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     edge-selective sharpening filter.\n")
        _T("    params\n")
        _T("      strength=<float>          sharpening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      threshold=<float>         edge detection threshold (default: %.1f, range: 0.0..255.0)\n")
        _T("      slope=<float>             sigmoid soft mask slope (default: %.1f, range: 0.0..)\n")
        _T("      luma_limit=<float>        attenuate sharpening in dark luma areas (default: %.1f, range: 0.0..255.0)\n")
        _T("      block_protect=<float>     attenuate near DCT block boundaries (default: %.1f, range: 0.0..1.0)\n")
        _T("      highq=<bool>              high quality edge detection (default: %s)\n")
        _T("      mask=<bool>               output edge mask only (default: %s)\n"),
        FILTER_DEFAULT_MSHARPEN_STRENGTH, FILTER_DEFAULT_MSHARPEN_THRESHOLD,
        FILTER_DEFAULT_MSHARPEN_SLOPE,
        FILTER_DEFAULT_MSHARPEN_LUMA_LIMIT,
        FILTER_DEFAULT_MSHARPEN_BLOCK_PROTECT,
        FILTER_DEFAULT_MSHARPEN_HIGHQ ? _T("true") : _T("false"),
        FILTER_DEFAULT_MSHARPEN_MASK  ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_CAS
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-cas [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     luma-only Contrast Adaptive Sharpening filter.\n")
        _T("    params\n")
        _T("      sharpness=<float>         sharpening strength (default: %.2f, range: 0.0..1.0)\n")
        _T("      hdr=<bool>                skip SDR gamma 2.0 luma approximation (default: %s)\n")
        _T("      chroma=<bool>             also sharpen chroma planes (default: false)\n"),
        FILTER_DEFAULT_CAS_SHARPNESS,
        FILTER_DEFAULT_CAS_HDR ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_WARPSHARP
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-warpsharp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     warpsharp filter to enhance edge.\n")
        _T("    params\n")
        _T("      threshold=<float>         edge mask threshold (default: %.1f, range: 0..255)\n")
        _T("      blur=<int>                number of times to blur the edge mask. (default: %d, range: 0..)\n")// 強くぼかす場合は回数を増やす。
        _T("\n")
        _T("      type=<int>                blur type, 0...13x13, 1...5x5 (default: %d)\n")
        _T("      depth=<float>             how far to warp (default: %.1f, range: -128..128)\n")
        _T("      chroma=<int>              0...use luma mask, 1...create chroma mask (default: %d)\n")
        _T("\n")
        _T("      depth_min=<float>         Weak-edge depth follows depth (default: derived)\n")
        _T("\n")
        _T("      depth_max=<float>         Strong-edge depth follows depth (default: derived)\n")
        _T("\n")
        _T("      edge_thr=<float>          mask level reaching depth_max (default: %.1f, range: 1..255)\n")
        _T("\n")
        _T("      gamma=<float>             adaptive depth response curve (default: %.2f, range: 0.01..8.0)\n")
        _T("\n"),
        FILTER_DEFAULT_WARPSHARP_THRESHOLD, FILTER_DEFAULT_WARPSHARP_BLUR, FILTER_DEFAULT_WARPSHARP_TYPE,
        FILTER_DEFAULT_WARPSHARP_DEPTH, FILTER_DEFAULT_WARPSHARP_CHROMA,
        FILTER_DEFAULT_WARPSHARP_EDGE_THR, FILTER_DEFAULT_WARPSHARP_GAMMA);
#endif
#if ENABLE_VPP_FILTER_DETAILSHARPEN
    str += strsprintf(_T("\n")
        _T("  [edge] Edge enhancement\n")
        _T("   --vpp-detailsharpen [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     detail sharpening filter to enhance fine texture.\n")
        _T("    params\n")
        _T("      z=<float>                 zero point (default: %.3f, range: 0.001..64.0)\n")
        _T("      sstr=<float>              sharpening strength (default: %.3f, range: 0.0..16.0)\n")
        _T("      power=<float>             nonlinear power (default: %.3f, range: 1.0..16.0)\n")
        _T("      ldmp=<float>              low-amplitude damping (default: %.3f, range: 0.0..1000.0)\n")
        _T("      mode=<int>                blur mode, 0...gauss, 1...box (default: %d)\n")
        _T("      med=<bool>                apply 3x3 median to blur (default: %s)\n"),
        FILTER_DEFAULT_DETAILSHARPEN_Z, FILTER_DEFAULT_DETAILSHARPEN_SSTR,
        FILTER_DEFAULT_DETAILSHARPEN_POWER, FILTER_DEFAULT_DETAILSHARPEN_LDMP,
        FILTER_DEFAULT_DETAILSHARPEN_MODE, FILTER_DEFAULT_DETAILSHARPEN_MED ? _T("true") : _T("false"));
#endif
#if ENABLE_VPP_FILTER_CURVES
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-curves [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Adjust colors using control-point curves.\n")
        _T("    params\n")
        _T("      preset=<string> Color preset (default: none)\n")
        _T("        values: none, color_negative, process, darker, lighter, increase_contrast, linear_contrast, medium_contrast, strong_contrast, negative, vintage\n")
        _T("      interp=<string> Interpolation between points (default: spline)\n")
        _T("        values: spline, pchip\n")
        _T("      all=<points> Fallback points for channels without an explicit curve.\n")
        _T("      m=<points> Master luminance curve applied after the channel curves.\n")
        _T("      r=<points> Red curve; overrides the preset.\n")
        _T("      g=<points> Green curve; overrides the preset.\n")
        _T("      b=<points> Blue curve; overrides the preset.\n"));
#endif
#if ENABLE_VPP_FILTER_SOFTLIGHT
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-softlight [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Correct color cast or boost contrast and saturation.\n")
        _T("    params\n")
        _T("      mode=<string> Correction mode (default: neutralize)\n")
        _T("        values: neutralize, lightness, neutralize_boost_sat, neutralize_full, neutralize_boost, boost, saturation\n")
        _T("      formula=<string> Blend formula (default: pegtop)\n")
        _T("        values: pegtop, illusionshu, w3c\n")
        _T("      skipblack=<bool> Exclude pure black pixels from averages (default: false)\n"));
#endif
#if ENABLE_VPP_FILTER_TWEAK
    str += strsprintf(_T("\n")
        _T("  [color] Color adjustment\n")
        _T("   --vpp-tweak [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Adjust brightness, contrast, gamma, saturation and hue.\n")
        _T("    params\n")
        _T("      brightness=<float> Brightness offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      contrast=<float> Contrast multiplier (default: %.1f, range: -2.0..2.0)\n")
        _T("      gamma=<float> Gamma correction (default: %.1f, range: 0.1..10.0)\n")
        _T("      saturation=<float> Saturation multiplier (default: %.1f, range: 0.0..3.0)\n")
        _T("      vibrance=<float> Selective saturation adjustment (default: %.1f, range: -1.0..2.0)\n")
        _T("      hue=<float> Hue rotation in degrees (default: %.1f, range: -180..180)\n")
        _T("      coring=<bool> Clamp output to TV range (default: false)\n")
        _T("      start_hue=<float> Start of the adjusted hue range in degrees (default: 0.0)\n")
        _T("      end_hue=<float> End of the adjusted hue range in degrees (default: 360.0)\n")
        _T("      y_offset=<float> Y offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      y_gain=<float> Y gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      cb_offset=<float> CB offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      cb_gain=<float> CB gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      cr_offset=<float> CR offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      cr_gain=<float> CR gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      r_offset=<float> R offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      r_gain=<float> R gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      r_gamma=<float> R gamma (default: %.1f, range: 0.1..10.0)\n")
        _T("      g_offset=<float> G offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      g_gain=<float> G gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      g_gamma=<float> G gamma (default: %.1f, range: 0.1..10.0)\n")
        _T("      b_offset=<float> B offset (default: %.1f, range: -1.0..1.0)\n")
        _T("      b_gain=<float> B gain (default: %.1f, range: -2.0..2.0)\n")
        _T("      b_gamma=<float> B gamma (default: %.1f, range: 0.1..10.0)\n"),
        FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_GAMMA, FILTER_DEFAULT_TWEAK_SATURATION, FILTER_DEFAULT_TWEAK_VIBRANCE, FILTER_DEFAULT_TWEAK_HUE, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_GAMMA, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_GAMMA, FILTER_DEFAULT_TWEAK_BRIGHTNESS, FILTER_DEFAULT_TWEAK_CONTRAST, FILTER_DEFAULT_TWEAK_GAMMA);
#endif
    str += strsprintf(_T("\n")
        _T("  [resize] Resize and geometry\n")
        _T("   --vpp-rotate <int>  Rotation is derived from transform flags when omitted (default: derived)\n")
        _T("     values: 90, 180, 270\n"));
    str += strsprintf(_T("\n")
        _T("  [resize] Resize and geometry\n")
        _T("   --vpp-transform [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Flip or transpose the image.\n")
        _T("    params\n")
        _T("      flip_x=<bool> Flip horizontally (default: false)\n")
        _T("      flip_y=<bool> Flip vertically (default: false)\n")
        _T("      transpose=<bool> Transpose rows and columns (default: false)\n"));
#if ENABLE_VPP_FILTER_DEBAND
    str += strsprintf(_T("\n")
        _T("  [deband] Deband\n")
        _T("   --vpp-deband [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable deband filter.\n")
        _T("    params\n")
        _T("      range=<int>               range (default: %d, range: 0..127)\n")
        _T("      sample=<int>              sample (default: %d, range: 0..2)\n")
        _T("      thre=<int>                Set thresholds for all planes (default: derived)\n")
        _T("      thre_y=<int>              threshold for y (default: %d, range: 0..31)\n")
        _T("      thre_cb=<int>             threshold for cb (default: %d, range: 0..31)\n")
        _T("      thre_cr=<int>             threshold for cr (default: %d, range: 0..31)\n")
        _T("      dither=<int>              Set dither strength for all planes (default: derived)\n")
        _T("      dither_y=<int>            strength of dither for y (default: %d, range: 0..31)\n")
        _T("      dither_c=<int>            strength of dither for cb/cr (default: %d, range: 0..31)\n")
        _T("      seed=<int>                Random seed (default: %d)\n")
        _T("      blurfirst=<bool>          Blur before debanding (default: %s)\n")
        _T("      rand_each_frame=<bool>    Generate a new seed per frame (default: %s)\n"),
        FILTER_DEFAULT_DEBAND_RANGE, FILTER_DEFAULT_DEBAND_MODE,
        FILTER_DEFAULT_DEBAND_THRE_Y, FILTER_DEFAULT_DEBAND_THRE_CB, FILTER_DEFAULT_DEBAND_THRE_CR,
        FILTER_DEFAULT_DEBAND_DITHER_Y, FILTER_DEFAULT_DEBAND_DITHER_C,
        FILTER_DEFAULT_DEBAND_SEED,
        FILTER_DEFAULT_DEBAND_BLUR_FIRST ? _T("on") : _T("off"),
        FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME ? _T("on") : _T("off"));
#endif
#if ENABLE_LIBPLACEBO
    str += strsprintf(_T("\n")
        _T("  [deband] Deband\n")
        _T("   --vpp-libplacebo-deband [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable deband filter.\n")
        _T("    params\n")
        _T("      iterations=<int>          iterations (default: %d, range: 0..)\n")
        _T("      threshold=<float>         cut-off threshold (default: %.1f, range: 0..)\n")
        _T("      radius=<float>            initial radius (default: %.1f, range: 0..)\n")
        _T("      grain_y=<float>           extra noise for luma (default: %.1f, range: 0..)\n")
        _T("      grain_c=<float>           Chroma grain follows grain_y (default: derived, range: 0..)\n")
        _T("      dither=<string>           Dither mode for 8-bit output (default: blue_noise)\n")
        _T("        values: none, blue_noise, ordered_lut, ordered_fixed, white_noise\n")
        _T("      lut_size=<int>            LUT size (default: %d)\n")
        _T("        values: 2, 4, 8, 16, 32, 64, 128, 256\n"),
        FILTER_DEFAULT_LIBPLACEBO_DEBAND_ITERATIONS, FILTER_DEFAULT_LIBPLACEBO_DEBAND_THRESHOLD, FILTER_DEFAULT_LIBPLACEBO_DEBAND_RADIUS,
        FILTER_DEFAULT_LIBPLACEBO_DEBAND_GRAINY, rgy_pow_int<FILTER_DEFAULT_LIBPLACEBO_DEBAND_LUT_SIZE>(2));
#endif
#if ENABLE_VPP_FILTER_PAD
    str += strsprintf(_T("\n")
        _T("  [resize] Resize and geometry\n")
        _T("   --vpp-pad <int>,<int>,<int>,<int>\n")
        _T("     add padding to left,top,right,bottom (in pixels)\n"));
#endif
#if ENABLE_VPP_FILTER_OVERLAY
    str += strsprintf(_T("\n")
        _T("  [overlay] Overlay\n")
        _T("   --vpp-overlay [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Composite an image or video over the input.\n")
        _T("    params\n")
        _T("      file=<path> Source image or video file.\n")
        _T("      pos=<int>x<int> Image position (default: 0x0)\n")
        _T("      size=<int>x<int> Image size; 0x0 disables resizing (default: 0x0)\n")
        _T("      alpha=<float> Overlay alpha (default: 0.0, range: 0.0..1.0)\n")
        _T("      alpha_mode=<string> How to combine source alpha (default: override)\n")
        _T("        values: override, mul, lumakey\n")
        _T("      lumakey_threshold=<float> Luma key center (default: 0.0, range: 0.0..1.0)\n")
        _T("      lumakey_tolerance=<float> Luma key range (default: 0.1, range: 0.0..1.0)\n")
        _T("      lumakey_softness=<float> Luma key edge softness (default: 0.0, range: 0.0..1.0)\n")
        _T("      loop=<bool> Loop the source (default: false)\n"));
#endif
#if ENABLE_VPP_FILTER_FRUC
    str += strsprintf(_T("\n")
        _T("  [framerate] Frame rate and interpolation\n")
        _T("   --vpp-fruc [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Interpolate frames. Use the bare value double to double the input rate.\n")
        _T("    params\n")
        _T("      fps=<int>/<int> Target frame rate; decimal rates are also accepted (default: derived)\n"));
#endif
    str += strsprintf(_T("\n")
        _T("  [other] Other settings\n")
        _T("   --vpp-perf-monitor           check vpp performance (for debug)\n")
    );
    return rgy_cmd_help_vpp_sections(str);
}

tstring gen_cmd_help_ctrl() {
    tstring str = strsprintf(_T("\n")
#if ENABLE_PARALLEL_ENC
        _T("   --parallel <int> or auto     Enable parallel encoding by file splitting.\n")
        _T("   --parallel-force-large-memory-filters\n")
        _T("                                Disable large memory filter parallel count limit.\n")
#endif
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
        _T("   --avoid-idle-clock <string>[=<float>]\n")
        _T("                                add dummy load to avoid idle GPU clock.\n")
        _T("                                - off\n")
        _T("                                - auto [default]\n")
        _T("                                - on\n")
        _T("                                 the optional value is target dummy load percentage,\n")
        _T("                                 when number omitted: %.2f percent load\n"),
        DEFAULT_DUMMY_LOAD_PERCENT);
    str += strsprintf(_T("")
        _T("   --task-perf-monitor          enable task performance monitoring.\n")
        _T("   --lowlatency                 minimize latency (might have lower throughput).\n")
        _T("   --fallback-bitdepth          fallback to 8-bit when 10-bit encode unsupported by all GPUs.\n"));
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
        _T("   --vsdir <string>            specifies VapourSynth portable directory to use.\n"));
#endif //#if defined(_WIN32) || defined(_WIN64)
    str += strsprintf(_T("\n")
        _T("   --vpy-assume-script-dir     resolves relative paths in .vpy from the script directory.\n"));
#if defined(_WIN32) || defined(_WIN64)
    str += strsprintf(_T("\n")
        _T("   --process-codepage <string>  utf8 ... use UTF-8 (default)\n")
        _T("                                os   ... use the codepage set in Operating System.\n"));
#endif //#if defined(_WIN32) || defined(_WIN64)
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    str += strsprintf(_T("\n")
        _T("   --disable-opencl             disable opencl features.\n"));
#endif
#if ENABLE_OPENCL
    str += strsprintf(_T("\n")
#if ENCODER_QSV
        _T("   --opencl-task-threads <auto|int>  set OpenCL task thread mode.\n")
        _T("                                  auto: 2 on IceLake or HEVC FF capable GPUs, otherwise 0 (default)\n")
        _T("                                  0: legacy single-thread path\n")
        _T("                                  2: acquire + release workers\n")
#endif
        _T("   --cl-perf-dump <dir>         dump OpenCL kernel performance data to <dir>.\n")
        _T("                                 enables CL_QUEUE_PROFILING_ENABLE automatically.\n")
        _T("                                 output: programs.jsonl, launches.jsonl, meta.json,\n")
        _T("                                         binaries/<name>__<hash>.bin,\n")
        _T("                                         build_logs/<name>__<hash>.log\n")
#if 0        
        _T("   --cl-perf-disasm-tool <str>  set disasm tool for --cl-perf-dump report generation.\n")
        _T("                                 auto, ocloc, rga, none (default: auto).\n")
#endif
#if ENCODER_QSV
        _T("   --ocloc-path <path>          set ocloc executable path for Intel GPU disasm.\n")
#elif ENCODER_VCEENC
        _T("   --rga-path <path>            set Radeon GPU Analyzer path for AMD GPU disasm.\n")
#endif
        _T("   --cl-perf-timeline [=<sec>]  enable per-event timeline capture for <sec> seconds (default 10).\n")
        _T("                                requires --cl-perf-dump. output: timeline.jsonl\n"));
#endif
#if ENCODER_QSV || ENCODER_VCEENC || ENCODER_MPP
    str += strsprintf(_T("\n")
        _T("   --python <string>            set python path for --perf-monitor-plot\n")
        _T("                                 and --cl-perf-dump report generation.\n"));
#endif
    str += strsprintf(_T("\n")
        _T("   --disable-vulkan             disable vulkan features.\n"));
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
