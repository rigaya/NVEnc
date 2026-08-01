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
#include "rgy_cmd.h"
#include "rgy_util.h"

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

template<typename T>
class RGYCmdSelfTest {
public:
    using ParseFunc = std::function<int(T *, const std::vector<tstring>&)>;
    using GenFunc = std::function<tstring(const T *)>;

    // argsPrefix は入出力ファイル名など、パースを通すために常に必要な引数
    // codecList はコーデックによって有効・無効が変わるオプションのために順に試すコーデック
    RGYCmdSelfTest(ParseFunc parse, GenFunc gen,
        const std::vector<tstring>& argsPrefix = { _T("-i"), _T("rgy_selftest_input.yuv"), _T("-o"), _T("rgy_selftest_output.raw") },
        const std::vector<tstring>& codecList = { _T("h264"), _T("hevc"), _T("av1") }) :
        m_parse(parse), m_gen(gen), m_argsPrefixList(), m_cmdDefaultList(), m_prefixIdx(0) {
        // レート制御やBフレームの有無によって生成されるかどうかが変わるオプションがあるため、それらを指定した前提も用意する
        // (パースできない組み合わせは run() 側で対象から外すので、エンコーダごとに変える必要はない)
        static const std::vector<std::vector<tstring>> extraList = { {}, { _T("--vbr"), _T("5000"), _T("-b"), _T("3") } };
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

private:
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

    ParseFunc m_parse;
    GenFunc m_gen;
    std::vector<std::vector<tstring>> m_argsPrefixList; // 診断時に前提として付与する引数列(コーデックごと)
    std::vector<tstring> m_cmdDefaultList;              // m_argsPrefixList の各要素に対応するデフォルトの生成結果
    size_t m_prefixIdx;                                 // 現在使用中の m_argsPrefixList のインデックス
    std::map<tstring, tstring> m_probeHint;             // オプションごとに、パースの通った引数列を覚えておく
};

#endif //__RGY_CMD_SELFTEST_H__
