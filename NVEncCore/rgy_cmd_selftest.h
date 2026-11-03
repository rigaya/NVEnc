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
#ifndef __RGY_CMD_SELFTEST_H__
#define __RGY_CMD_SELFTEST_H__

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "rgy_cmd.h"
#include "rgy_util.h"
#if ENABLE_CPP_REGEX
#include <regex>
#endif

// コマンドラインのパーサ(parse_cmd)と生成(gen_cmd)の対応関係を自己診断する。
//
// 診断の考え方
//   構造体を直接比較しようとするとメンバごとの比較コードが必要になるため、
//   コマンドライン文字列の「不動点」で判定する。
//     C0 --parse--> P1 --gen--> C1 --parse--> P2 --gen--> C2
//   parse と gen が対応していれば C1 == C2 になる。ここが崩れれば
//   単位の食い違い、書式の誤り、値の取りこぼしのいずれかが起きている。
//
//   これとは別に、パースはできるが gen_cmd 側に書き忘れがある場合は
//   C1 == C2 では検出できない (両方で消えるため)。そこで、値を変えても
//   gen_cmd の出力がデフォルトから変化しないオプションを生成漏れとして報告する。
//
// テストケースの作り方
//   オプション名は encoder_help() の解析 (createOptionList()) から得る。
//   与える値はパーサ自身から引き出す。わざと不正な値・不正なサブパラメータ名を
//   渡すと、パーサはエラー時に候補一覧を提示するので、それを RGYCmdErrorInfo 経由で
//   回収する。これによりテスト側に値のテーブルを二重持ちしなくて済む。

// args を parse_cmd に渡せる argv 形式に変換する
// (先頭は実行ファイル名の位置なので空、末尾はパーサが i+1 を参照するための番兵)
static inline std::vector<const TCHAR *> rgy_cmd_selftest_argv(const std::vector<tstring>& args) {
    std::vector<const TCHAR *> argv;
    argv.push_back(_T(""));
    for (const auto& arg : args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(_T(""));
    return argv;
}

struct RGYCmdSelfTestResult {
    int total = 0;
    int ok = 0;
    int ngRoundTrip = 0;    // parse と gen の往復で結果が変わる
    int ngNotGenerated = 0; // parse は通るが gen に反映されない
    int ngUnknownParam = 0; // パーサが候補として提示するサブパラメータ名を自分で受理しない
    int noValue = 0;        // 有効な値を見つけられず診断できなかった(診断側の限界)
    int except = 0;         // 一致しないのが正しいと分かっている既知の例外
    int skipped = 0;        // 診断対象外のオプション
};

// help検証は判定不能も別計上し、検証を飛ばしてNG 0になったように見せない。
struct RGYCmdHelpSelfTestResult {
    int ng[7] = {};
    int checked[7] = {};
    int except[7] = {};
    int unsupported[7] = {};
    int options = 0;
    int parameters = 0;
    int derived = 0;
    int structured = 0;
};

// 値の組み合わせを指定して確認する往復テスト用ケース。
// expected は再生成結果に必ず含まれる文字列、unexpected は含まれてはいけない文字列を表す。
struct RGYCmdSelfTestRoundTripCase {
    tstring name;
    std::vector<tstring> args;
    bool saveDisabledPrm;
    std::vector<tstring> expected;
    std::vector<tstring> unexpected;
};

template<typename T>
class RGYCmdSelfTest {
public:
    using ParseFunc = std::function<int(T *, const std::vector<tstring>&)>;
    using GenFunc = std::function<tstring(const T *)>;
    using GenWithSaveDisabledFunc = std::function<tstring(const T *, bool)>;

    // argsPrefix は入出力ファイル名など、パースを通すために常に必要な引数
    // codecList はコーデックによって有効・無効が変わるオプションのために順に試すコーデック
    // extraList はレート制御などの追加前提。Bフレーム非対応のエンコーダでも指定できる。
    RGYCmdSelfTest(ParseFunc parse, GenFunc gen,
        const std::vector<tstring>& argsPrefix = { _T("-i"), _T("rgy_selftest_input.yuv"), _T("-o"), _T("rgy_selftest_output.raw") },
        const std::vector<tstring>& codecList = { _T("h264"), _T("hevc"), _T("av1") },
        const std::vector<std::vector<tstring>>& extraList = { {}, { _T("--vbr"), _T("5000"), _T("-b"), _T("3") } }) :
        m_parse(parse), m_gen(gen), m_argsPrefixList(), m_cmdDefaultList(), m_prefixIdx(0) {
        // レート制御やBフレームの有無によって生成されるかどうかが変わるオプションがあるため、それらを指定した前提も用意する
        // パースできない組み合わせはrun()側で対象から外す。
        for (const auto& extra : extraList) {
            for (const auto& codec : codecList) {
                auto prefix = argsPrefix;
                prefix.push_back(_T("-c"));
                prefix.push_back(codec);
                prefix.insert(prefix.end(), extra.begin(), extra.end());
                m_argsPrefixList.push_back(prefix);
            }
        }
    };

    // filter が指定された場合、その文字列を名前に含むオプションのみ診断する
    int run(const tstring& filter) {
#if !ENABLE_CPP_REGEX
        _ftprintf(stderr, _T("--check-cmd-parse requires ENABLE_CPP_REGEX.\n"));
        return -1;
#else
        auto& errInfo = rgy_cmd_error_info();
        errInfo.quiet = true; // 診断中は大量に発生するパースエラーの表示を抑止する

        // 何も指定しない場合の生成結果。各オプションの効果はこれとの差分で判定する
        m_cmdDefaultList.clear();
        for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size(); ) {
            tstring cmdDefault;
            if (!parseAndGen({}, cmdDefault)) {
                // そのエンコーダに存在しないオプションを含む前提は診断対象から外す
                m_argsPrefixList.erase(m_argsPrefixList.begin() + m_prefixIdx);
                continue;
            }
            m_cmdDefaultList.push_back(cmdDefault);
            m_prefixIdx++;
        }
        if (m_cmdDefaultList.empty()) {
            _ftprintf(stderr, _T("failed to parse default options.\n"));
            errInfo.quiet = false;
            return -1;
        }
        m_prefixIdx = 0;

        RGYCmdSelfTestResult res;
        for (const auto& optHelp : createOptionList()) {
            const auto opt = char_to_tstring(optHelp.first);
            if (isSkipOption(opt)) {
                res.skipped++;
                continue;
            }
            if (filter.length() > 0 && opt.find(filter) == tstring::npos) {
                continue;
            }
            const auto subParams = querySubParams(opt);
            if (subParams.empty()) {
                testCase(opt, tstring(), res);
            } else {
                // サブパラメータを持つオプションは、どのキーが壊れているか分かるようキー単位で診断する
                for (const auto& key : subParams) {
                    testCase(opt, key, res);
                }
            }
        }

        errInfo.quiet = false;
        _ftprintf(stdout, _T("\n----------------------------------------------------------------\n"));
        _ftprintf(stdout, _T("total %d, ok %d, roundtrip NG %d, no-gen NG %d, unknown-param NG %d, no-value %d, except %d, skipped(option) %d\n"),
            res.total, res.ok, res.ngRoundTrip, res.ngNotGenerated, res.ngUnknownParam, res.noValue, res.except, res.skipped);
        return (res.ngRoundTrip + res.ngNotGenerated + res.ngUnknownParam > 0) ? -1 : 1;
#endif
    }

    // 通常の自動探索では表現しにくい、複数パラメータ・空白を含む値・無効化状態を指定して確認する。
    int runRoundTripCases(const std::vector<RGYCmdSelfTestRoundTripCase>& cases, const GenWithSaveDisabledFunc& gen) {
        auto& errInfo = rgy_cmd_error_info();
        const auto quiet = errInfo.quiet;
        errInfo.quiet = true;
        int ng = 0;
        for (const auto& test : cases) {
            bool ok = false;
            tstring detail;
            for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size(); m_prefixIdx++) {
                T prm;
                errInfo.reset();
                const auto args = withPrefix(test.args);
                if (m_parse(&prm, args) != 0 || errInfo.errorOccurred) {
                    continue;
                }
                const auto cmd1 = gen(&prm, test.saveDisabledPrm);
                for (const auto& expected : test.expected) {
                    if (cmd1.find(expected) == tstring::npos) {
                        detail = _T("再生成結果に必要な指定がない: ") + expected + _T("\n    in :") + concatArgs(args) + _T("\n    gen:") + cmd1;
                        break;
                    }
                }
                if (detail.empty()) {
                    for (const auto& unexpected : test.unexpected) {
                        if (cmd1.find(unexpected) != tstring::npos) {
                            detail = _T("再生成結果に無効化されるべき指定が残っている: ") + unexpected + _T("\n    in :") + concatArgs(args) + _T("\n    gen:") + cmd1;
                            break;
                        }
                    }
                }
                if (!detail.empty()) {
                    break;
                }
                T prm2;
                errInfo.reset();
                const auto genArgs = splitCommandLine(trim(cmd1).c_str());
                if (m_parse(&prm2, genArgs) != 0 || errInfo.errorOccurred) {
                    detail = _T("再生成したコマンドラインを再度パースできない\n    in :") + concatArgs(args) + _T("\n    gen:") + cmd1;
                    break;
                }
                const auto cmd2 = gen(&prm2, test.saveDisabledPrm);
                if (cmd1 != cmd2) {
                    detail = _T("再生成したコマンドラインが一致しない\n    in  :") + concatArgs(args) + _T("\n    gen1:") + cmd1 + _T("\n    gen2:") + cmd2;
                    break;
                }
                ok = true;
                break;
            }
            if (ok) {
                printResult(_T("OK"), test.name, _T("指定値を含む往復に成功"));
            } else {
                ng++;
                printResult(_T("NG"), test.name, detail.empty() ? _T("テスト前提のコマンドラインをパースできない") : detail);
            }
        }
        m_prefixIdx = 0;
        errInfo.quiet = quiet;
        _ftprintf(stdout, _T("metric roundtrip total %d, NG %d\n"), (int)cases.size(), ng);
        return (ng == 0) ? 1 : -1;
    }

    const RGYCmdHelpSelfTestResult& lastHelpValidationResult() const { return m_helpResult; }

    // 構文はhelpから、受理可能な値と生成の前提は既存のパーサ/probeから取得する。
    int runHelpValidation(const tstring& help, const tstring& filter = tstring()) {
        m_helpResult = RGYCmdHelpSelfTestResult();
#if !ENABLE_CPP_REGEX
        _ftprintf(stdout, _T("Help validation requires ENABLE_CPP_REGEX.\n"));
        m_helpResult.unsupported[0]++;
        return -1;
#else
        auto& error = rgy_cmd_error_info();
        struct QuietRestore {
            RGYCmdErrorInfo& error;
            bool quiet;
            ~QuietRestore() { error.quiet = quiet; }
        } restore{ error, error.quiet };
        error.quiet = true;
        m_cmdDefaultList.clear();
        for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size();) {
            tstring generated;
            if (!helpParse({}, generated)) {
                m_argsPrefixList.erase(m_argsPrefixList.begin() + m_prefixIdx);
            } else {
                m_cmdDefaultList.push_back(generated);
                ++m_prefixIdx;
            }
        }
        m_prefixIdx = 0;
        if (m_cmdDefaultList.empty()) {
            helpUnsupported(0, _T("help"), _T("no valid parser premise"));
            return -1;
        }
        auto options = parseHelp(help);
        for (auto& option : options) {
            if (!filter.empty() && option.name.find(filter) == tstring::npos) continue;
            ++m_helpResult.options;
            const auto optionName = _T("--") + option.name;
            helpCheck(0, option.canonical, optionName, _T("option must start at column 4"));
            helpCheck(0, option.classified, optionName, _T("option has no valid category heading"));
            helpCheck(0, !option.duplicate, optionName, _T("duplicate option declaration"));
            // mainの名前はparams内の宣言だけを参照し、位置と重複も独立に調べる。
            std::set<tstring> parameterNames, mainNames;
            for (const auto& field : option.fields) {
                if (!field.key.empty() && option.paramsLine != 0 && field.line > option.paramsLine) parameterNames.insert(field.key);
            }
            for (const auto line : option.mainLines) {
                helpCheck(5, option.paramsLine != 0 && line < option.paramsLine, optionName, _T("main: must precede params"));
            }
            for (const auto& entry : option.mainParameters) {
                helpCheck(5, parameterNames.count(entry) != 0, optionName, _T("main: parameter is not declared in params: ") + entry);
                helpCheck(5, mainNames.insert(entry).second, optionName, _T("duplicate parameter in main: ") + entry);
            }
            std::set<tstring> names;
            for (auto& field : option.fields) {
                if (!field.key.empty()) ++m_helpResult.parameters;
                const auto name = optionName + (field.key.empty() ? tstring() : _T(" ") + field.key);
                helpCheck(0, field.canonical, name, _T("parameter must use six spaces and name=<type>"));
                helpCheck(0, names.insert(field.key).second, name, _T("duplicate parameter declaration"));
                // 条件の参照先と候補は、同じオプションの宣言済みパラメータだけで解決する。
                if (!field.conditions.empty()) {
                    helpCheck(6, field.conditionLast, name, _T("when: must follow the description and values:"));
                    helpCheck(6, field.conditions.size() == 1, name, _T("only one when: line is allowed per parameter"));
                    for (const auto& condition : field.conditions) {
                        HelpMatch match;
                        static const HelpRegex whenRow(_T("^( {8,})when: ([A-Za-z0-9_][A-Za-z0-9_.-]*)(?:(=|!=)(.+))?$"));
                        const bool valid = std::regex_match(condition, match, whenRow);
                        helpCheck(6, valid, name, _T("malformed when: condition"));
                        if (!valid) continue;
                        helpCheck(6, !field.hasValues || match[1].str().size() == field.valuesIndent, name,
                            _T("when: must use the same indentation as values:"));
                        const auto reference = match[2].str();
                        const auto referenced = std::find_if(option.fields.begin(), option.fields.end(), [&](const HelpField& item) {
                            return item.key == reference && option.paramsLine != 0 && item.line > option.paramsLine;
                        });
                        helpCheck(6, referenced != option.fields.end(), name, _T("when: reference is not declared in params: ") + reference);
                        if (referenced == option.fields.end() || match[3].str().empty()) continue;
                        for (const auto& rawValue : split(match[4].str(), _T(","))) {
                            const auto value = trim(rawValue);
                            const auto wildcard = value.find(_T('*'));
                            const bool prefix = wildcard != tstring::npos && wildcard == value.size() - 1;
                            const bool validValue = !value.empty() && (wildcard == tstring::npos
                                || (prefix && value.size() > 1 && value.find(_T('*'), wildcard + 1) == tstring::npos));
                            const bool found = validValue && std::any_of(referenced->values.begin(), referenced->values.end(), [&](const tstring& candidate) {
                                return prefix ? candidate.compare(0, value.size() - 1, value, 0, value.size() - 1) == 0 : candidate == value;
                            });
                            helpCheck(6, found, name, _T("when: candidate does not match any referenced values: ") + value);
                        }
                    }
                }
                const auto types = split(field.type, _T("|"));
                helpCheck(0, field.validTypeSyntax, name, _T("malformed type declaration"));
                if (field.structured) ++m_helpResult.structured;
                const bool numeric = !field.structured && (std::find(types.begin(), types.end(), _T("int")) != types.end()
                    || std::find(types.begin(), types.end(), _T("float")) != types.end()
                    || std::find(types.begin(), types.end(), _T("bool")) != types.end());
                const bool stringType = !field.structured && std::find(types.begin(), types.end(), _T("string")) != types.end();
                m_prefixIdx = 0;
                const auto parserValues = queryValueList(option.name, field.key);
                // 任意文字列の拒否だけでは、列挙型と構造化された自由入力を区別できない。
                const bool constrainedString = stringType && (!parserValues.empty() || field.hasValues);
                const bool unknownString = stringType && !constrainedString
                    && !helpAccepts(option.name, field.key, _T("rgy_selftest_invalid_value"));
                if (unknownString) helpUnsupported(0, name, _T("cannot distinguish an enum from a structured string without parser candidates or values:"));
                const bool needsDefault = numeric || constrainedString;
                if (needsDefault) helpCheck(0, field.hasDefault, name, _T("required default is missing"));
                if (field.hasDefault) helpCheck(0, field.canonicalDefault, name, _T("default must be on the declaration line as (default: X[, range: A..B])"));
                if (constrainedString) helpCheck(0, field.hasValues, name, _T("constrained string requires values:"));
                if (field.hasValues) {
                    helpCheck(0, field.canonicalValues, name, _T("values: must immediately follow the declaration"));
                    // 候補を正規化した後で比較し、引用符や空白の違いで重複を見逃さない。
                    const std::set<tstring> uniqueValues(field.values.begin(), field.values.end());
                    helpCheck(0, uniqueValues.size() == field.values.size(), name, _T("duplicate value in values:"));
                }
                if (!field.structured) for (const auto& type : types) {
                    if (type == _T("int") || type == _T("float") || type == _T("bool") || type == _T("string") || type == _T("path")) continue;
                    helpCheck(0, std::find(field.values.begin(), field.values.end(), type) != field.values.end(), name, _T("literal type alternative is missing from values:"));
                }
                if (field.hasRange) {
                    helpCheck(0, field.canonicalRange && (!field.low.empty() || !field.high.empty()), name, _T("range must use A..B in the default annotation"));
                }
                const bool derived = field.hasDefault && field.defaultValue == _T("derived");
                if (derived) ++m_helpResult.derived;
                if (const auto reason = helpException(option.name, field.key)) {
                    for (int category = 1; category < 5; ++category) {
                        if ((category <= 2 && field.hasDefault && !derived) || (category == 3 && field.hasValues) || (category == 4 && field.hasRange))
                            helpExcept(category, name, reason);
                    }
                    continue;
                }
                if (field.hasDefault && !derived) {
                    helpCheck(1, helpAccepts(option.name, field.key, field.defaultValue), name,
                        _T("parser rejected default: ") + field.defaultValue);
                    if (const auto reason = helpException(option.name, field.key, 2)) helpExcept(2, name, reason);
                    else helpDefault(option.name, field, name);
                } else if (needsDefault && !derived) {
                    helpUnsupported(1, name, _T("default unavailable because help declaration is incomplete"));
                    helpUnsupported(2, name, _T("default unavailable because help declaration is incomplete"));
                }
                for (const auto& value : field.values) {
                    // 変換元と変換先の候補は、両側に同じ値を置いて構造全体をパーサに渡す。
                    const auto input = field.type == _T("<from>:<to>") ? value + _T(":") + value : value;
                    helpCheck(3, helpAccepts(option.name, field.key, input), name,
                        _T("parser rejected value: ") + input);
                }
                if (constrainedString && !field.hasValues) helpUnsupported(3, name, _T("values unavailable because help declaration is incomplete"));
                if (unknownString) helpUnsupported(3, name, _T("enum versus structured string is unresolved; no declared values are available"));
                if (field.hasRange) {
                    for (const auto& endpoint : { field.low, field.high }) {
                        if (endpoint.empty()) continue;
                        helpCheck(4, helpAccepts(option.name, field.key, endpoint), name,
                            _T("parser rejected range endpoint: ") + endpoint);
                    }
                }
            }
        }
        helpCheck(0, m_helpResult.options > 0, _T("VPP Options"), _T("no VPP options were discovered"));
        m_prefixIdx = 0;
        int failures = 0;
        _ftprintf(stdout, _T("derived defaults %d (C2/C3 excluded), structured types %d\n"), m_helpResult.derived, m_helpResult.structured);
        _ftprintf(stdout, _T("Help validation: options %d, parameters %d\n"), m_helpResult.options, m_helpResult.parameters);
        for (int category = 0; category < 7; ++category) {
            _ftprintf(stdout, _T("C%d checked %d, NG %d, except %d, unsupported %d\n"), category + 1,
                m_helpResult.checked[category], m_helpResult.ng[category], m_helpResult.except[category], m_helpResult.unsupported[category]);
            failures += m_helpResult.ng[category] + m_helpResult.unsupported[category];
        }
        return failures == 0 ? 1 : -1;
#endif
    }

private:
#if ENABLE_CPP_REGEX
    using HelpRegex = std::basic_regex<TCHAR>;
    using HelpMatch = std::match_results<tstring::const_iterator>;
    struct HelpField {
        tstring key, type, defaultValue, low, high;
        std::vector<tstring> values;
        std::vector<tstring> conditions;
        size_t line = 0, valuesIndent = 0;
        bool canonical = false;
        bool validTypeSyntax = false, structured = false;
        bool hasDefault = false, canonicalDefault = false;
        bool hasValues = false, canonicalValues = false;
        bool conditionLast = true;
        bool hasRange = false, canonicalRange = false;
    };
    struct HelpOption {
        tstring name;
        std::vector<HelpField> fields;
        std::vector<tstring> mainParameters;
        std::vector<size_t> mainLines;
        size_t paramsLine = 0;
        bool classified = false, canonical = false, duplicate = false;
    };

    void helpCheck(int category, bool ok, const tstring& name, const tstring& reason) {
        ++m_helpResult.checked[category];
        if (!ok) {
            ++m_helpResult.ng[category];
            _ftprintf(stdout, _T("[C%d NG] %s: %s\n"), category + 1, name.c_str(), reason.c_str());
        }
    }
    void helpUnsupported(int category, const tstring& name, const tstring& reason) {
        ++m_helpResult.unsupported[category];
        _ftprintf(stdout, _T("[C%d UNVERIFIED] %s: %s\n"), category + 1, name.c_str(), reason.c_str());
    }
    void helpExcept(int category, const tstring& name, const tstring& reason) {
        ++m_helpResult.except[category];
        _ftprintf(stdout, _T("[C%d EXCEPT] %s: %s\n"), category + 1, name.c_str(), reason.c_str());
    }
    static const TCHAR *helpException(const tstring& option, const tstring& key, int category = -1) {
        struct Exception { tstring name; const TCHAR *reason; int category; };
        static const std::vector<Exception> exceptions = {
            { _T("vpp-afs ini"), _T("the parser reads an existing AFS ini file; no fixture file is supplied"), -1 },
            // 固定値の受理はC2/C4で調べ、非既定値を作れないC3だけを恒久例外にする。
            { _T("vpp-ivtc drop"), _T("only 1 is accepted; no non-default value exists for a generated-output comparison"), 2 },
            { _T("vpp-rtgmc-shimmer-repair mode"), _T("only passthrough is accepted; no non-default mode exists for a generated-output comparison"), 2 }
        };
        const auto name = option + _T(" ") + key;
        for (const auto& exception : exceptions)
            if (name == exception.name && (exception.category < 0 || exception.category == category)) return exception.reason;
        return nullptr;
    }

    // 省略可能な値の直後を空の番兵にせず、既存の-i等を置いて裸の指定を正しく調べる。
    bool helpParse(const std::vector<tstring>& args, tstring& generated) {
        auto all = args;
        const auto& prefix = m_argsPrefixList[m_prefixIdx];
        all.insert(all.end(), prefix.begin(), prefix.end());
        return parseAndGen(all, generated, false);
    }
    static tstring helpUnquote(const tstring& source) {
        auto value = trim(source);
        if (value.size() >= 2 && ((value.front() == _T('"') && value.back() == _T('"'))
            || (value.front() == _T('\'') && value.back() == _T('\'')))) return value.substr(1, value.size() - 2);
        return value;
    }
    // クォート内のカンマは、主値やパスの区切りと区別する。
    static std::vector<tstring> helpSplitValues(const tstring& value) {
        std::vector<tstring> result;
        tstring part;
        TCHAR quote = 0;
        for (const auto ch : value) {
            if ((ch == _T('"') || ch == _T('\'')) && (quote == 0 || quote == ch)) quote = quote == 0 ? ch : 0;
            if (ch == _T(',') && quote == 0) { result.push_back(trim(part)); part.clear(); }
            else part += ch;
        }
        if (!part.empty()) result.push_back(trim(part));
        return result;
    }
    static tstring helpWithoutKey(const tstring& value, const tstring& key) {
        tstring result;
        for (const auto& token : helpSplitValues(value)) {
            const auto equals = token.find(_T('='));
            if (equals != tstring::npos && trim(token.substr(0, equals)) == key) continue;
            if (!result.empty()) result += _T(',');
            result += token;
        }
        return result;
    }
    std::vector<tstring> helpContexts(const tstring& option, const tstring& key, bool extended) {
        std::vector<tstring> contexts = { tstring() };
        if (key.empty()) return contexts;
        auto add = [&](const tstring& value) {
            const auto context = helpWithoutKey(value, key);
            if (std::find(contexts.begin(), contexts.end(), context) == contexts.end()) contexts.push_back(context);
        };
        if (m_probeHint.count(option)) add(m_probeHint[option]);
        if (!extended) return contexts;
        const auto result = probe(option, key);
        if (result.args.size() == 2) add(result.args[1]);
        for (const auto& main : queryValueList(option, tstring())) add(main);
        for (const auto& main : genericValues()) add(main);
        for (const auto& other : querySubParams(option)) {
            if (other == key) continue;
            auto values = queryValueList(option, other);
            if (values.empty()) {
                // 数値だけでなく2x2等の構造値も試す。通らない値で前提の数を膨らませない。
                int accepted = 0;
                for (const auto& value : genericValues()) {
                    tstring generated;
                    if (!helpParse(buildArgs(option, other, value), generated)) continue;
                    add(other + _T("=") + value);
                    if (++accepted >= 2) break;
                }
            } else {
                for (const auto& value : values) add(other + _T("=") + value);
            }
        }
        return contexts;
    }
    bool helpAccepts(const tstring& option, const tstring& key, const tstring& value) {
        for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size(); ++m_prefixIdx) {
            for (int extended = 0; extended < 2; ++extended) {
                for (const auto& context : helpContexts(option, key, extended != 0)) {
                    tstring generated;
                    if (helpParse(buildArgs(option, key, value, context), generated)) {
                        m_prefixIdx = 0;
                        return true;
                    }
                }
            }
        }
        m_prefixIdx = 0;
        return false;
    }
    bool helpObservable(const tstring& option, const tstring& key, const tstring& context, const tstring& baseline, const HelpField& field, tstring *alternative = nullptr) {
        auto values = queryValueList(option, key);
        values.insert(values.end(), field.values.begin(), field.values.end());
        if (!field.low.empty()) values.push_back(field.low);
        if (!field.high.empty()) values.push_back(field.high);
        const auto& generic = genericValues();
        values.insert(values.end(), generic.begin(), generic.end());
        for (const auto& value : values) {
            tstring generated;
            if (helpParse(buildArgs(option, key, value, context), generated) && generated != baseline) {
                if (alternative) *alternative = generated;
                return true;
            }
        }
        return false;
    }
    static std::vector<tstring> helpGeneratedContexts(const tstring& generated, const tstring& option, const tstring& key) {
        const auto args = splitCommandLine(trim(generated).c_str());
        std::vector<tstring> contexts;
        for (size_t index = 0; index < args.size(); ++index) {
            if (args[index] != _T("--") + option) continue;
            if (index + 1 >= args.size() || args[index + 1].empty() || args[index + 1][0] == _T('-')) contexts.push_back(tstring());
            else contexts.push_back(helpWithoutKey(args[++index], key));
        }
        return contexts;
    }
    bool helpNeutralContext(const tstring& option, const tstring& key, const tstring& baseline) {
        const auto contexts = helpGeneratedContexts(baseline, option, key);
        if (contexts.empty()) return baseline == m_cmdDefaultList[m_prefixIdx];
        std::vector<tstring> args;
        for (const auto& context : contexts) {
            args.push_back(_T("--") + option);
            if (!context.empty()) args.push_back(context);
        }
        tstring repeated;
        // 前提が対象keyの別名を設定していれば、生成から対象keyを除くと状態が変わる。
        return helpParse(args, repeated) && repeated == baseline;
    }
    bool helpRecoverGeneratedContext(const tstring& option, const HelpField& field,
        const tstring& generated, bool& observable) {
        observable = false;
        if (field.key.empty()) return false;
        const auto contexts = helpGeneratedContexts(generated, option, field.key);
        bool matched = false;
        for (size_t occurrence = 0; occurrence < contexts.size(); ++occurrence) {
            const auto& context = contexts[occurrence];
            if (context.empty()) continue;
            tstring repeated, baseline;
            // 明示既定値の生成結果を変えない前提だけを使い、別slotの既定値へ逃げない。
            if (!helpParse(buildArgs(option, field.key, field.defaultValue, context), repeated) || repeated != generated) continue;
            if (!helpParse({ _T("--") + option, context }, baseline) || baseline != generated) continue;
            tstring alternative;
            if (!helpObservable(option, field.key, context, baseline, field, &alternative)) {
                matched = true;
                continue;
            }
            // 別名や複合表現へ対象値が残る前提は採用しない。代替値でも前提部分が不変なことを確認する。
            const auto alternateContexts = helpGeneratedContexts(alternative, option, field.key);
            if (occurrence >= alternateContexts.size() || alternateContexts[occurrence] != context) continue;
            matched = true;
            observable = true;
            m_probeHint[option] = context;
            return true;
        }
        return matched;
    }

    void helpDefault(const tstring& option, const HelpField& field, const tstring& name) {
        bool parsed = false, unobservable = false, activation = false, noInstance = false;
        for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size(); ++m_prefixIdx) {
            for (int extended = 0; extended < 2; ++extended) {
                for (const auto& context : helpContexts(option, field.key, extended != 0)) {
                    tstring baseline, generated;
                    const auto bare = context.empty() ? std::vector<tstring>{ _T("--") + option }
                        : std::vector<tstring>{ _T("--") + option, context };
                    if (helpParse(bare, baseline) && helpParse(buildArgs(option, field.key, field.defaultValue, context), generated)) {
                        parsed = true;
                        if (!context.empty() && !field.key.empty() && !helpNeutralContext(option, field.key, baseline)) continue;
                        if (generated != baseline) {
                            bool observable = false;
                            if (helpRecoverGeneratedContext(option, field, generated, observable)) {
                                if (observable) {
                                    helpCheck(2, true, name, tstring());
                                    m_prefixIdx = 0;
                                    return;
                                }
                                unobservable = true;
                                continue;
                            }
                            // 裸の指定がインスタンスを作らない場合だけ、有効な共通前提を探索する。
                            if (baseline == m_cmdDefaultList[m_prefixIdx]) {
                                noInstance = true;
                                continue;
                            }
                            helpCheck(2, false, name, _T("help default differs from parser default: ") + field.defaultValue);
                            m_prefixIdx = 0;
                            return;
                        }
                        if (helpObservable(option, field.key, context, baseline, field)) {
                            if (!field.key.empty() && !context.empty()) m_probeHint[option] = context;
                            helpCheck(2, true, name, tstring());
                            m_prefixIdx = 0;
                            return;
                        }
                        unobservable = true;
                    }
                }
            }
            if (field.key.empty()) {
                // 必須短形式は、無効化側があれば同じ有効状態で値だけ比較する。
                const auto negative = _T("no-") + option;
                tstring baseline, generated;
                if (helpParse({ _T("--") + negative }, baseline)
                    && helpParse(buildArgs(negative, tstring(), field.defaultValue), generated)) {
                    if (generated != baseline || helpObservable(negative, tstring(), tstring(), baseline, field)) {
                        helpCheck(2, generated == baseline, name, _T("help default differs from disabled parser default: ") + field.defaultValue);
                        m_prefixIdx = 0;
                        return;
                    }
                    unobservable = true;
                }
                // 省略と同じ生成になり、かつ別の値が生成を変える場合だけ省略基準を認める。
                if (helpParse({}, baseline) && helpParse(buildArgs(option, tstring(), field.defaultValue), generated)) {
                    if (generated == baseline && helpObservable(option, tstring(), tstring(), baseline, field)) {
                        helpCheck(2, true, name, tstring());
                        m_prefixIdx = 0;
                        return;
                    }
                    activation = generated != baseline;
                }
            }
        }
        m_prefixIdx = 0;
        helpUnsupported(2, name, activation ? _T("required short form has no comparable activation state")
            : noInstance ? _T("bare option creates no instance and no comparable parameter premise was found")
            : unobservable ? _T("no alternate value changes generated output in a comparable premise")
            : parsed ? _T("no comparable generated default") : _T("default or baseline could not be parsed"));
    }

    static void helpSetType(HelpField& field, const tstring& type) {
        field.type = type;
        field.validTypeSyntax = std::regex_match(type, HelpRegex(_T("<[^<>]+>([^ \\t<>=]*<[^<>]+>)*")));
        if (!field.validTypeSyntax) return;
        const bool single = type.find(_T('>')) == type.size() - 1 && type.find(_T('<'), 1) == tstring::npos;
        const auto inner = single ? type.substr(1, type.size() - 2) : type;
        const auto parts = split(inner, _T("|"));
        const bool base = std::any_of(parts.begin(), parts.end(), [](const tstring& part) {
            return part == _T("int") || part == _T("float") || part == _T("bool") || part == _T("string") || part == _T("path");
        });
        const bool atoms = std::all_of(parts.begin(), parts.end(), [](const tstring& part) {
            return std::regex_match(part, HelpRegex(_T("[A-Za-z0-9_.+-]+")));
        });
        field.structured = !single || !base || !atoms;
        if (!field.structured) field.type = inner;
    }

    static tstring helpLegacyDefaultValue(const HelpField& field, const tstring& source) {
        auto value = helpUnquote(source);
        if (value.empty() || (value.back() != _T(';') && value.back() != _T('.'))) return value;
        const auto candidate = value.substr(0, value.size() - 1);
        const auto types = split(field.type, _T("|"));
        const bool numeric = std::find(types.begin(), types.end(), _T("int")) != types.end()
            || std::find(types.begin(), types.end(), _T("float")) != types.end();
        const bool boolean = std::find(types.begin(), types.end(), _T("bool")) != types.end();
        // 説明の句読点であると型から確定できる場合だけ除く。パスや複合値は加工しない。
        if (numeric && std::regex_match(candidate, HelpRegex(_T("[+-]?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][+-]?[0-9]+)?")))) return candidate;
        if (boolean && std::regex_match(candidate, HelpRegex(_T("true|false|on|off|0|1"), std::regex_constants::icase))) return candidate;
        return value;
    }

    static void helpReadMetadata(HelpField& field, const tstring& line, size_t number) {
        if (trim(line).compare(0, 5, _T("when:")) == 0) {
            field.conditions.push_back(line);
            return;
        }
        if (!field.conditions.empty() && !trim(line).empty()) field.conditionLast = false;
        HelpMatch match;
        static const HelpRegex canonicalDefault(_T("\\(default: ([^,()]+)(, range: ([^(),]*)\\.\\.([^(),]*))?\\)$"));
        if (std::regex_search(line, match, canonicalDefault)) {
            if (!field.hasDefault) {
                field.hasDefault = true;
                field.defaultValue = helpUnquote(match[1].str());
                field.canonicalDefault = number == field.line;
            }
            if (match[2].matched) {
                field.hasRange = true;
                field.low = trim(match[3].str()); field.high = trim(match[4].str());
                field.canonicalRange = number == field.line;
            }
        } else if (!field.hasDefault) {
            // 旧書式も値だけは回収する。key名に含まれるdefault=や、別keyの既定値は混同しない。
            static const HelpRegex legacyDefault(_T("(^|[ \\t(,])([A-Za-z0-9_][A-Za-z0-9_.-]*[ \\t]+)?default[ \\t]*[:=][ \\t]*([^ \\t,)]+)"), std::regex_constants::icase);
            static const HelpRegex legacyMarked(_T("([A-Za-z0-9_.+-]+)\\(default\\)"), std::regex_constants::icase);
            tstring named, anonymous;
            bool hasNamed = false;
            // 宣言のkey=<type>自体は説明に含めず、defaultというkey名も値扱いしない。
            const auto typeEnd = number == field.line ? line.find(_T('>')) : tstring::npos;
            const auto metadata = typeEnd == tstring::npos ? line : line.substr(typeEnd + 1);
            auto begin = metadata.cbegin();
            while (std::regex_search(begin, metadata.cend(), match, legacyDefault)) {
                const auto key = trim(match[2].str());
                if (key.empty()) anonymous = match[3].str();
                else {
                    hasNamed = true;
                    if (key == field.key) named = match[3].str();
                }
                begin = match[0].second;
            }
            if (!named.empty() || (!hasNamed && !anonymous.empty())) {
                field.hasDefault = true;
                field.defaultValue = helpLegacyDefaultValue(field, named.empty() ? anonymous : named);
            } else if (!hasNamed && std::regex_search(metadata, match, legacyMarked)) {
                field.hasDefault = true; field.defaultValue = helpLegacyDefaultValue(field, match[1].str());
            }
        }
        static const HelpRegex values(_T("^[ \\t]*values: (.+)$"));
        if (std::regex_match(line, match, values)) {
            field.hasValues = true;
            field.valuesIndent = line.find_first_not_of(_T(" \t"));
            field.canonicalValues = number == field.line + 1;
            for (const auto& value : helpSplitValues(match[1].str())) field.values.push_back(helpUnquote(value));
        }
        if (!field.hasRange) {
            static const HelpRegex range(_T("range: ([^,()]*)\\.\\.([^,()]*)"));
            if (std::regex_search(line, match, range)) {
                field.hasRange = true; field.low = trim(match[1].str()); field.high = trim(match[2].str());
            }
        }
    }
    std::vector<HelpOption> parseHelp(const tstring& help) {
        std::vector<HelpOption> options;
        const std::vector<tstring> categories = { _T("deinterlace"), _T("denoise"), _T("edge"), _T("deband"), _T("color"),
            _T("resize"), _T("framerate"), _T("overlay"), _T("ai"), _T("other") };
        static const HelpRegex heading(_T("^  \\[([a-z]+)\\] [A-Za-z].*$"));
        static const HelpRegex headingCandidate(_T("^ *\\[[A-Za-z][A-Za-z0-9_-]*\\].*$"));
        static const HelpRegex optionRow(_T("^( *)(--[A-Za-z0-9_-]+)([ \\t].*)?$"));
        static const HelpRegex fieldRow(_T("^( *)([A-Za-z0-9_][A-Za-z0-9_./-]*)[ \\t]*=[ \\t]*(<[^<>]+>([^ \\t<>=]*<[^<>]+>)*)(.*)$"));
        static const HelpRegex fieldName(_T("[A-Za-z0-9_][A-Za-z0-9_.-]*"));
        static const HelpRegex shortType(_T("<[^<>]+>([^ \\t<>=]*<[^<>]+>)*"));
        bool inVpp = false, classified = false, inParams = false;
        int current = -1, lastField = -1;
        std::set<tstring> names;
        size_t number = 0;
        for (auto line : split(help, _T("\n"))) {
            ++number;
            if (!line.empty() && line.back() == _T('\r')) line.pop_back();
            if (!inVpp) { if (trim(line) == _T("VPP Options:")) inVpp = true; continue; }
            HelpMatch match;
            if (std::regex_match(line, match, heading)) {
                classified = std::find(categories.begin(), categories.end(), match[1].str()) != categories.end();
                current = lastField = -1; inParams = false;
                continue;
            }
            if (std::regex_match(line, headingCandidate)) {
                classified = false; current = lastField = -1; inParams = false;
                continue;
            }
            if (std::regex_match(line, match, optionRow) && match[1].length() <= 4) {
                const auto option = match[2].str().substr(2);
                current = lastField = -1; inParams = false;
                if (option.compare(0, 4, _T("vpp-")) != 0) continue;
                HelpOption item;
                item.name = option; item.classified = classified; item.canonical = match[1].length() == 3;
                item.duplicate = !names.insert(option).second;
                options.push_back(item); current = (int)options.size() - 1;
                const auto suffix = match[3].str();
                if (std::regex_search(suffix, match, shortType)) {
                    const auto type = match[0].str();
                    if (type.find(_T("<param")) != 0 && type != _T("<value>")) {
                        HelpField field; helpSetType(field, type); field.line = number; field.canonical = true;
                        helpReadMetadata(field, line, number);
                        options.back().fields.push_back(field); lastField = 0;
                    }
                }
                continue;
            }
            if (current < 0) continue;
            const auto trimmedLine = trim(line);
            if (trimmedLine.compare(0, 5, _T("main:")) == 0) {
                options[current].mainLines.push_back(number);
                for (const auto& name : split(trimmedLine.substr(5), _T(","))) options[current].mainParameters.push_back(trim(name));
                continue;
            }
            if (trimmedLine == _T("params")) {
                inParams = true; lastField = -1;
                if (options[current].paramsLine == 0) options[current].paramsLine = number;
                continue;
            }
            if (std::regex_match(line, match, fieldRow)) {
                HelpField field;
                field.key = match[2].str(); helpSetType(field, match[3].str()); field.line = number;
                field.canonical = match[1].length() == 6 && std::regex_match(field.key, fieldName)
                    && (match[5].str().empty() || match[5].str().front() == _T(' ') || match[5].str().front() == _T('\t'));
                field.canonical = field.canonical && !std::regex_search(match[5].str(),
                    HelpRegex(_T(",[ \\t]*[^ \\t=,]+[ \\t]*=[ \\t]*<")));
                helpReadMetadata(field, line, number);
                options[current].fields.push_back(field); lastField = (int)options[current].fields.size() - 1;
            } else if (inParams && (std::regex_match(line, match, HelpRegex(_T("^ {6}([^ \\t=]+)[ \\t]*=.*$")))
                || std::regex_match(line, match, HelpRegex(_T("^ {1,7}([A-Za-z0-9_][A-Za-z0-9_./-]*)[ \\t]*=.*$"))))) {
                // 型やkeyの括弧が壊れた行も捨てず、書式NGとして残す。
                HelpField field; field.key = match[1].str(); field.line = number;
                HelpMatch typeMatch;
                if (std::regex_search(line, typeMatch, shortType)) helpSetType(field, typeMatch[0].str());
                helpReadMetadata(field, line, number);
                options[current].fields.push_back(field); lastField = (int)options[current].fields.size() - 1;
            } else if (lastField >= 0) {
                helpReadMetadata(options[current].fields[lastField], line, number);
            }
        }
        return options;
    }
#endif

    // probe の結果。args が空の場合、なぜ判定できなかったのかを parseOk / unknownParam で表す
    struct ProbeResult {
        std::vector<tstring> args;   // gen_cmd の出力を変化させた引数列
        bool parseOk = false;        // パースだけは通る値が1つでもあった
        bool unknownParam = false;   // サブパラメータ名自体が受理されなかった
    };

    // 引数列をパースし、コマンドラインを再生成する。パースに失敗した場合 false
    // addPrefix は再パース時(既に必須引数を含む)には false にする
    bool parseAndGen(const std::vector<tstring>& args, tstring& cmd, bool addPrefix = true) {
        T prm;
        auto& errInfo = rgy_cmd_error_info();
        errInfo.reset();
        if (m_parse(&prm, addPrefix ? withPrefix(args) : args) != 0 || errInfo.errorOccurred) {
            return false;
        }
        cmd = m_gen(&prm);
        return true;
    }

    std::vector<tstring> withPrefix(const std::vector<tstring>& args) const {
        auto argsAll = m_argsPrefixList[m_prefixIdx];
        argsAll.insert(argsAll.end(), args.begin(), args.end());
        return argsAll;
    }

    // わざと不正なサブパラメータ名を渡し、パーサが提示するサブパラメータ名の一覧を回収する
    std::vector<tstring> querySubParams(const tstring& opt) {
        T prm;
        auto& errInfo = rgy_cmd_error_info();
        errInfo.reset();
        m_parse(&prm, withPrefix({ _T("--") + opt, _T("rgy_selftest_invalid_param=1") }));
        std::vector<tstring> params;
        for (const auto& param : errInfo.paramList) {
            params.push_back(char_to_tstring(param));
        }
        return params;
    }

    // わざと不正な値を渡し、パーサが提示する値の候補一覧を回収する
    std::vector<tstring> queryValueList(const tstring& opt, const tstring& subKey) {
        static const auto INVALID_VALUE = tstring(_T("rgy_selftest_invalid_value"));
        T prm;
        auto& errInfo = rgy_cmd_error_info();
        errInfo.reset();
        m_parse(&prm, withPrefix(buildArgs(opt, subKey, INVALID_VALUE)));
        std::vector<tstring> values;
        for (const auto& value : errInfo.valueList) {
            values.push_back(char_to_tstring(value));
        }
        return values;
    }

    // mainValue はサブパラメータの前に置く主値 ("--vpp-resize <algo>,<subKey>=<value>" の <algo>)
    static std::vector<tstring> buildArgs(const tstring& opt, const tstring& subKey, const tstring& value, const tstring& mainValue = tstring()) {
        const auto optArg = _T("--") + opt;
        if (subKey.length() == 0) {
            return { optArg, value };
        }
        const auto prm = subKey + _T("=") + value;
        return { optArg, (mainValue.length() == 0) ? prm : (mainValue + _T(",") + prm) };
    }

    // 値の型が判明しないオプションに対して順に試す汎用の値
    // 候補一覧をコード上に持たず、エラーメッセージ中に直接書いているオプションが一定数あるため、
    // そうしたオプションでもパースが通るよう、頻出する列挙値もここに含めておく
    static const std::vector<tstring>& genericValues() {
        static const std::vector<tstring> values = {
            _T("2"), _T("1"), _T("3"), _T("0"), _T("0.5"), _T("true"), _T("false"),
            _T("16"), _T("2x2"), _T("2/1"), _T("2,2,2,2"), _T("2:2"), _T("2:2:2"), _T("rgy_selftest"),
            _T("on"), _T("off"), _T("auto"), _T("all"), _T("y:u:v"),
            _T("bob"), _T("frame"), _T("tff"), _T("interlaced"), _T("clamp"), _T("fp32")
        };
        return values;
    }

    // gen_cmd の出力がデフォルトから変化する引数列を探す
    // (パースが通るのに生成に反映されない = 生成漏れ、そもそもパースが通らない = 診断側の値の問題)
    ProbeResult probe(const tstring& opt, const tstring& subKey) {
        ProbeResult result;
        result.unknownParam = subKey.length() > 0;
        const auto& cmdDefault = m_cmdDefaultList[m_prefixIdx];
        auto& errInfo = rgy_cmd_error_info();
        tstring cmd;
        if (subKey.length() == 0) {
            // 値を取らないフラグ形式のオプションを先に試す
            const std::vector<tstring> args = { _T("--") + opt };
            if (parseAndGen(args, cmd)) {
                result.parseOk = true;
                if (cmd != cmdDefault) { result.args = args; return result; }
                // 既定で有効なフラグはそのまま指定しても差分が出ないので、対になるオプションを試す
                const auto optPair = (opt.compare(0, 3, _T("no-")) == 0) ? opt.substr(3) : (_T("no-") + opt);
                const std::vector<tstring> argsPair = { _T("--") + optPair };
                if (parseAndGen(argsPair, cmd) && cmd != cmdDefault) { result.args = argsPair; return result; }
            }
        }
        // パーサが知っている候補を優先して試し、無ければ汎用の値を試す
        auto values = queryValueList(opt, subKey);
        // "<string>:<string>" のように値を2つ要求するオプションがあるため、候補を2つ繋げた形も試す
        for (const auto& value : std::vector<tstring>(values)) {
            values.push_back(value + _T(":") + value);
        }
        const auto& generic = genericValues();
        values.insert(values.end(), generic.begin(), generic.end());
        for (const auto& value : values) {
            const auto args = buildArgs(opt, subKey, value);
            if (parseAndGen(args, cmd)) {
                result.parseOk = true;
                result.unknownParam = false;
                if (cmd != cmdDefault) { result.args = args; return result; }
            } else if (!errInfo.unknownParam) {
                result.unknownParam = false; // 値が拒否されただけならサブパラメータ名自体は通っている
            }
        }
        // 主値(リサイズアルゴリズム等)が特定の値のときだけ生成に反映されるサブパラメータがあるため、
        // ここまでで差分が出なかった場合は主値を変えながら再試行する
        // 主値を指定した時点で出力は変わるので、比較対象は「主値のみを指定した場合の出力」とする
        if (subKey.length() > 0 && !result.unknownParam) {
            auto mainValues = queryValueList(opt, tstring());
            // 同じオプションで既にパースの通った引数列があれば、それを前置の第一候補にする
            // (--dynamic-rc の end のように、複数のサブパラメータの同時指定が必要なものに対応するため)
            if (m_probeHint.count(opt) > 0) mainValues.insert(mainValues.begin(), m_probeHint[opt]);
            mainValues.insert(mainValues.end(), generic.begin(), generic.end());
            // "--dynamic-rc start=0,cbr=1000" のように、他のサブパラメータの同時指定が必須のものがあるため、
            // 他のサブパラメータも主値と同じ扱いで前置する
            for (const auto& subKeyOther : querySubParams(opt)) {
                if (subKeyOther == subKey) continue;
                const auto valuesOther = queryValueList(opt, subKeyOther);
                mainValues.push_back(subKeyOther + _T("=") + (valuesOther.empty() ? generic[0] : valuesOther[0]));
            }
            for (const auto& mainValue : mainValues) {
                tstring cmdRef;
                // 前置のみではパースが通らないことがあるため、その場合は最初にパースの通った値の出力を基準にする
                if (!parseAndGen({ _T("--") + opt, mainValue }, cmdRef)) cmdRef.clear();
                for (const auto& value : values) {
                    const auto args = buildArgs(opt, subKey, value, mainValue);
                    if (!parseAndGen(args, cmd)) continue;
                    if (cmdRef.length() == 0) { cmdRef = cmd; continue; }
                    if (cmd != cmdRef) { result.args = args; return result; }
                }
            }
        }
        return result;
    }

    void testCase(const tstring& opt, const tstring& subKey, RGYCmdSelfTestResult& res) {
        res.total++;
        const auto name = (subKey.length() == 0) ? (_T("--") + opt) : (_T("--") + opt + _T(" ") + subKey);
        // コーデックによって受け付けるかどうかが変わるオプションがあるため、順に試す
        ProbeResult probeRes;
        for (m_prefixIdx = 0; m_prefixIdx < m_argsPrefixList.size(); m_prefixIdx++) {
            const auto probeThis = probe(opt, subKey);
            probeRes.args = probeThis.args;
            probeRes.parseOk |= probeThis.parseOk;
            probeRes.unknownParam |= probeThis.unknownParam;
            if (!probeRes.args.empty()) break;
        }
        const auto args = probeRes.args;
        // 同一オプションの他のサブパラメータを診断する際の前置として使えるよう覚えておく
        if (subKey.length() > 0 && args.size() == 2) m_probeHint[opt] = args.back();
        if (args.empty()) {
            m_prefixIdx = 0;
            if (const auto reason = knownException(name); reason != nullptr) {
                res.except++;
                printResult(_T("EXCEPT"), name, tstring(_T("一致しないのが正しい既知の例外: ")) + reason);
            } else if (probeRes.parseOk) {
                res.ngNotGenerated++;
                printResult(_T("NOGEN"), name, _T("パースは通るがgen_cmdの出力に反映されない(生成漏れの疑い)"));
            } else if (probeRes.unknownParam) {
                res.ngUnknownParam++;
                printResult(_T("NOPARAM"), name, _T("パーサが候補として提示するサブパラメータ名を、パーサ自身が受理しない"));
            } else {
                res.noValue++;
                printResult(_T("NOVALUE"), name, _T("パースの通る値を見つけられず診断できなかった"));
            }
            return;
        }
        tstring cmd1;
        parseAndGen(args, cmd1); // probe で成功済みなので必ず成功する
        tstring cmd2;
        // gen_cmd の出力は先頭に空白が入るが、Windowsの分割では先頭の空白がそのまま空の引数になるため、両端を除いてから分割する
        if (!parseAndGen(splitCommandLine(trim(cmd1).c_str()), cmd2, false)) {
            res.ngRoundTrip++;
            printResult(_T("NOPARSE"), name, _T("生成したコマンドラインを再度パースできない\n    in  :") + concatArgs(withPrefix(args)) + _T("\n    gen :") + cmd1);
            return;
        }
        if (cmd1 != cmd2) {
            res.ngRoundTrip++;
            printResult(_T("DIFF"), name, _T("再生成したコマンドラインが一致しない\n    in  :") + concatArgs(withPrefix(args)) + _T("\n    gen1:") + cmd1 + _T("\n    gen2:") + cmd2);
            return;
        }
        res.ok++;
    }

    static tstring concatArgs(const std::vector<tstring>& args) {
        tstring str;
        for (const auto& arg : args) {
            str += _T(" ") + arg;
        }
        return str;
    }

    static void printResult(const TCHAR *type, const tstring& name, const tstring& detail) {
        _ftprintf(stdout, _T("[%-7s] %s\n    %s\n"), type, name.c_str(), detail.c_str());
    }

    // parse と gen が一致しないのが正しい(あるいは診断側から判定できない)ことが分かっているケース
    // 戻り値は除外する理由。除外しない場合は nullptr
    static const TCHAR *knownException(const tstring& name) {
        static const std::vector<std::pair<const TCHAR *, const TCHAR *>> list = {
            { _T("--avbr-unitsize"),                    _T("レート制御がAVBRの場合のみ生成されるため") },
            { _T("--strict-gop"),                       _T("既定で有効であり、無効化するオプションが存在しないため") },
            { _T("--bench-quality"),                    _T("ベンチマーク専用でエンコード設定ではないため") },
            { _T("--sub-copy asdata"),                  _T("サブパラメータではなくトラック指定に付ける修飾子であるため") },
            { _T("--vpp-afs ini"),                      _T("実在するiniファイルのパスが必要で、診断側から値を用意できないため") },
            { _T("--cabac"),                            _T("既定値と同じ指定であり差分が出ないため(対となる--cavlcで診断済み)") },
            { _T("--d3d"),                              _T("既定値と同じ指定であり差分が出ないため(--d3d9/--d3d11/--disable-d3dで診断済み)") },
        };
        for (const auto& except : list) {
            if (name == except.first) return except.second;
        }
        return nullptr;
    }

    // パースに副作用があるもの、情報表示専用で構造体に値を持たないものは診断対象外とする
    static bool isSkipOption(const tstring& opt) {
        static const std::vector<tstring> skipList = {
            _T("help"), _T("version"), _T("option-list"), _T("option-file"),
            _T("process-codepage"), _T("process-codepage-applied"),
            _T("debug-cmd-parser")
        };
        if (opt.compare(0, 6, _T("check-")) == 0) {
            return true;
        }
        return std::find(skipList.begin(), skipList.end(), opt) != skipList.end();
    }

    RGYCmdHelpSelfTestResult m_helpResult;
    ParseFunc m_parse;
    GenFunc m_gen;
    std::vector<std::vector<tstring>> m_argsPrefixList; // 診断時に前提として付与する引数列(コーデックごと)
    std::vector<tstring> m_cmdDefaultList;              // m_argsPrefixList の各要素に対応するデフォルトの生成結果
    size_t m_prefixIdx;                                 // 現在使用中の m_argsPrefixList のインデックス
    std::map<tstring, tstring> m_probeHint;             // オプションごとに、パースの通った引数列を覚えておく
};

#endif //__RGY_CMD_SELFTEST_H__
