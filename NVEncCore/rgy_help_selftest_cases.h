#ifndef __RGY_HELP_SELFTEST_CASES_H__
#define __RGY_HELP_SELFTEST_CASES_H__

#include <cmath>
#include "rgy_cmd_selftest.h"

// 実フィルタの知識を持たない合成パーサで、help検証器の検出能力を試験する。
struct RGYHelpFixtureParams {
    bool enable = false;
    bool routed = false;
    bool lazy = false;
    std::map<tstring, tstring> values = {
        { _T("strength"), _T("2") }, { _T("strength_default"), _T("2") }, { _T("threshold"), _T("0.5") },
        { _T("mode"), _T("auto") }, { _T("toggle"), _T("false") }, { _T("24fps"), _T("false") },
        { _T("route"), _T("bare") }, { _T("file"), _T("") }, { _T("shape"), _T("2x2") }, { _T("opaque"), _T("2") }
    };
};

static inline int rgy_help_fixture_parse(RGYHelpFixtureParams *prm, const std::vector<tstring>& args) {
    auto& error = rgy_cmd_error_info();
    const auto fail = [&](bool unknown = false) {
        error.errorOccurred = true;
        error.unknownParam = unknown;
        return 1;
    };
    const auto setValue = [&](const tstring& key, const tstring& value) {
        const tstring target = key == _T("alias") ? _T("strength") : key;
        if (prm->values.count(target) == 0) {
            error.paramList = { "strength", "alias", "strength_default", "threshold", "mode", "toggle", "24fps", "route", "file", "shape", "opaque" };
            return fail(true);
        }
        if (target == _T("route")) {
            if (value != _T("bare") && value != _T("one")) return fail();
        } else if (target == _T("shape")) {
            const auto dimensions = split(value, _T("x"));
            if (dimensions.size() != 2) return fail();
            for (const auto& dimension : dimensions) {
                if (dimension.empty() || !std::all_of(dimension.begin(), dimension.end(), [](TCHAR c) { return c >= _T('0') && c <= _T('9'); })) return fail();
                try { if (std::stoul(dimension) == 0) return fail(); } catch (...) { return fail(); }
            }
        } else if (target == _T("mode")) {
            if (value != _T("auto") && value != _T("fast") && value != _T("slow")) {
                error.valueList = { "auto", "fast", "slow" };
                return fail();
            }
        } else if (target == _T("toggle") || target == _T("24fps")) {
            if (value != _T("true") && value != _T("false")) {
                error.valueList = { "true", "false" };
                return fail();
            }
        } else if (target == _T("strength") || target == _T("strength_default") || target == _T("threshold") || target == _T("opaque")) {
            try {
                size_t end = 0;
                const auto number = std::stod(value, &end);
                if (end != value.size() || !std::isfinite(number)) return fail();
                if (target == _T("strength") && (number < 0 || number > 4 || number != std::floor(number))) return fail();
                if (target == _T("threshold") && (number < 0 || number > 1)) return fail();
            } catch (...) { return fail(); }
        }
        prm->values[target] = value;
        return 0;
    };
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == _T("-i") || args[i] == _T("-o") || args[i] == _T("-c") || args[i] == _T("--vbr") || args[i] == _T("-b")) {
            if (++i >= args.size()) return fail();
        } else if (args[i] == _T("--vpp-strength") || args[i] == _T("--no-vpp-strength")) {
            const bool positive = args[i] == _T("--vpp-strength");
            prm->enable = positive;
            if (i + 1 >= args.size() || args[i + 1].find(_T("-")) == 0) {
                if (positive) return fail();
            } else if (setValue(_T("strength"), args[++i])) return 1;
        } else if (args[i] == _T("--vpp-mode")) {
            if (i + 1 >= args.size() || args[i + 1].find(_T("-")) == 0) return fail();
            if (setValue(_T("mode"), args[++i])) return 1;
        } else if (args[i] == _T("--vpp-fixture") || args[i] == _T("--no-vpp-fixture") || args[i] == _T("--vpp-routed") || args[i] == _T("--vpp-lazy")) {
            prm->routed = args[i] == _T("--vpp-routed");
            prm->enable = args[i] != _T("--no-vpp-fixture");
            prm->lazy = args[i] == _T("--vpp-lazy");
            if (prm->lazy) { prm->values[_T("strength")] = _T("0"); prm->enable = false; }
            if (i + 1 >= args.size() || args[i + 1].find(_T("-")) == 0) continue;
            if (prm->lazy) prm->enable = true;
            if (prm->routed) prm->values[_T("route")] = _T("one");
            for (const auto& item : split(args[++i], _T(","))) {
                const auto eq = item.find(_T("="));
                if (eq == tstring::npos) {
                    error.valueList = { "auto", "fast", "slow" };
                    return fail();
                }
                if (setValue(item.substr(0, eq), item.substr(eq + 1))) return 1;
            }
        } else { return fail(); }
    }
    return 0;
}

static inline tstring rgy_help_fixture_gen(const RGYHelpFixtureParams *prm) {
    RGYHelpFixtureParams defaults;
    if (prm->lazy) defaults.values[_T("strength")] = _T("0");
    tstring value;
    for (const auto& pair : prm->values) {
        // opaqueは生成漏れを意図的に作り、C3が見かけの一致を合格にしないことを試す。
        if (pair.first == _T("opaque") || pair.second == defaults.values.at(pair.first)) continue;
        if (!value.empty()) value += _T(",");
        value += pair.first + _T("=") + pair.second;
    }
    if (!prm->enable && value.empty()) return tstring();
    const tstring option = prm->lazy ? _T(" --vpp-lazy") : prm->routed ? _T(" --vpp-routed") : prm->enable ? _T(" --vpp-fixture") : _T(" --no-vpp-fixture");
    return option + (value.empty() ? tstring() : _T(" ") + value);
}

static inline int rgy_help_selftest_cases() {
    const tstring prefix = _T("VPP Options:\n\n  [denoise] Noise reduction\n   --vpp-fixture [<param1>=<value>][,...]\n    params\n");
    const tstring numeric = _T("      strength=<int> Strength (default: 2, range: 0..4)\n");
    const tstring mode = _T("      mode=<string> Mode (default: auto)\n        values: auto, fast, slow\n");
    const tstring golden = prefix + numeric + mode
        + _T("      threshold=<float> Threshold (default: 0.5, range: 0..1)\n")
        + _T("      toggle=<bool> Toggle (default: false)\n")
        + _T("      file=<path> Output path\n");
    struct Case { const TCHAR *name; tstring help; int category; bool unsupported; bool noSemanticNg = false; int derived = -1; int structured = -1; };
    const std::vector<Case> cases = {
        { _T("valid schema"), golden, -1, false },
        { _T("no-op activation ignores target-changing alias premises"), _T("VPP Options:\n  [other] Settings\n   --vpp-lazy [<param1>=<value>]\n    params\n      strength=<int> Strength (default: 0)\n"), -1, false },
        { _T("no-op activation still rejects wrong default"), _T("VPP Options:\n  [other] Settings\n   --vpp-lazy [<param1>=<value>]\n    params\n      strength=<int> Strength (default: 3)\n"), 2, false },
        { _T("routed default preserves activation"), _T("VPP Options:\n  [other] Settings\n   --vpp-routed [<param1>=<value>]\n    params\n      strength=<int> Strength (default: 2)\n"), -1, false },
        { _T("routed alias wrong default is not hidden by serialized context"), _T("VPP Options:\n  [other] Settings\n   --vpp-routed [<param1>=<value>]\n    params\n      alias=<int> Strength alias (default: 3)\n"), 2, false },
        { _T("routed wrong default is still rejected"), _T("VPP Options:\n  [other] Settings\n   --vpp-routed [<param1>=<value>]\n    params\n      strength=<int> Strength (default: 3)\n"), 2, false },
        { _T("derived numeric default"), prefix + _T("      strength=<int> Strength (default: derived, range: 0..4)\n"), -1, false, false, 1, 0 },
        { _T("structured named type"), prefix + _T("      shape=<WxH> Shape (default: derived)\n"), -1, false, false, 1, 1 },
        { _T("structured composite type"), prefix + _T("      shape=<int>x<int> Shape (default: derived)\n"), -1, false, false, 1, 1 },
        { _T("numeric parameter name"), prefix + _T("      24fps=<bool> Toggle (default: false)\n"), -1, false },
        { _T("structured string is not assumed to be enum"), prefix + _T("      shape=<string> Shape (default: 2x2)\n"), 0, true },
        { _T("valid required short enum"), _T("VPP Options:\n  [other] Settings\n   --vpp-mode <string> Mode (default: auto)\n     values: auto, fast, slow\n"), -1, false },
        { _T("valid activating required short"), _T("VPP Options:\n  [edge] Edge\n   --vpp-strength <int> Strength (default: 2, range: 0..4)\n"), -1, false },
        { _T("C1 option indentation"), golden + _T("    --vpp-mode <string> Mode (default: auto)\n      values: auto, fast, slow\n"), 0, false },
        { _T("C1 grouped parameter names"), prefix + _T("      [r,g,b]_offset=<float> Offset\n"), 0, false },
        { _T("C1 multiple declarations"), prefix + _T("      strength=<int>,threshold=<float> Strength (default: 2)\n"), 0, false },
        { _T("C1 malformed field"), prefix + _T("      strength=int Strength (default: 2)\n"), 0, false },
        { _T("C1 malformed category"), golden + _T("   [other] Settings\n   --vpp-mode <string> Mode (default: auto)\n     values: auto, fast, slow\n"), 0, false },
        { _T("legacy metadata key name"), prefix + _T("      strength_default=<int> Strength (default:2)\n"), 0, false, true },
        { _T("legacy metadata multiple keys"), prefix + _T("      strength=<int> Strength\n        (threshold default=0.5, strength default=2)\n"), 0, false, true },
        { _T("legacy metadata punctuation"), prefix + _T("      toggle=<bool> Toggle (default=false.)\n"), 0, false, true },
        { _T("C1 missing category"), _T("VPP Options:\n   --vpp-fixture\n    params\n") + numeric, 0, false },
        { _T("C1 missing default"), prefix + _T("      strength=<int> Strength\n"), 0, false },
        { _T("C1 missing enum values"), prefix + _T("      mode=<string> Mode (default: auto)\n"), 0, false },
        { _T("C1 duplicate option"), golden + _T("   --vpp-fixture\n"), 0, false },
        { _T("C1 duplicate named enum value"), prefix + _T("      mode=<string> Mode (default: auto)\n        values: auto, fast, slow, fast\n"), 0, false, true },
        { _T("C1 duplicate quoted enum value"), prefix + _T("      mode=<string> Mode (default: auto)\n        values: auto, fast, slow, \"fast\"\n"), 0, false, true },
        { _T("C1 duplicate short enum value"), _T("VPP Options:\n  [other] Settings\n   --vpp-mode <string> Mode (default: auto)\n     values: auto, fast, slow, auto\n"), 0, false, true },
        { _T("C2 rejected default"), prefix + _T("      strength=<int> Strength (default: 9)\n"), 1, false },
        { _T("C3 wrong default"), prefix + _T("      strength=<int> Strength (default: 3)\n"), 2, false },
        { _T("C4 rejected enum value"), prefix + _T("      mode=<string> Mode (default: auto)\n        values: auto, snail\n"), 3, false },
        { _T("C5 rejected endpoint"), prefix + _T("      strength=<int> Strength (default: 2, range: 0..9)\n"), 4, false },
        { _T("C3 generation omission"), prefix + _T("      opaque=<int> Opaque (default: 2)\n"), 2, true }
    };
    int failures = 0;
    for (const auto& test : cases) {
        RGYCmdSelfTest<RGYHelpFixtureParams> checker(rgy_help_fixture_parse, rgy_help_fixture_gen, {}, { _T("h264") });
        const int status = checker.runHelpValidation(test.help);
        const auto& result = checker.lastHelpValidationResult();
        bool ok = status > 0;
        if (test.category >= 0) {
            ok = status < 0 && (test.unsupported ? result.unsupported[test.category] > 0 : result.ng[test.category] > 0);
        }
        // 書式だけが旧式でも、回収した正しい既定値を受理失敗・既定値誤記にしてはいけない。
        if (test.noSemanticNg) ok = ok && result.ng[1] == 0 && result.ng[2] == 0;
        if (test.derived >= 0) ok = ok && result.derived == test.derived;
        if (test.structured >= 0) ok = ok && result.structured == test.structured;
        _ftprintf(stdout, _T("[HELP-FIXTURE %s] %s\n"), ok ? _T("OK") : _T("NG"), test.name);
        if (!ok) ++failures;
    }
    const CX_DESC values[] = { { _T("a"), 0 }, { _T("b"), 1 }, { _T("a"), 2 }, { nullptr, 0 } };
    const bool valuesOk = rgy_cmd_help_values(values) == _T("        values: a, b\n");
    const auto grouped = rgy_cmd_help_vpp_sections(_T("  [color] Color\n   --vpp-first\n  [denoise] Noise\n   --vpp-noise\n  [color] Color\n   --vpp-second\n"));
    const auto color = grouped.find(_T("  [color]"));
    const bool sectionsOk = grouped.find(_T("  [denoise]")) < color
        && grouped.find(_T("  [color]"), color + 1) == tstring::npos
        && grouped.find(_T("--vpp-first")) < grouped.find(_T("--vpp-second"));
    const auto layout = rgy_cmd_help_vpp_sections(_T("  [other] Settings\n   --vpp-layout\n    params\n      mode=<string> Output cadence.\n        values: auto, fast\n        Continued description.\n      unnecessarily_long_parameter_name=<int> Long description.\n"));
    const auto rows = split(layout, _T("\n"));
    const auto atColumn = [&](const tstring& text, size_t column) {
        return std::any_of(rows.begin(), rows.end(), [&](const tstring& row) { return row.find(text) == column; });
    };
    const bool layoutOk = atColumn(_T("Output cadence."), 28) && atColumn(_T("values:"), 28)
        && atColumn(_T("Continued description."), 28)
        && layout.find(_T("      unnecessarily_long_parameter_name=<int> Long description.")) != tstring::npos
        && rgy_cmd_help_vpp_sections(layout) == layout;
    if (!layoutOk) {
        ++failures;
        _ftprintf(stdout, _T("layout columns: description=%d values=%d continuation=%d long=%d stable=%d\n"),
            atColumn(_T("Output cadence."), 28), atColumn(_T("values:"), 28), atColumn(_T("Continued description."), 28),
            layout.find(_T("      unnecessarily_long_parameter_name=<int> Long description.")) != tstring::npos,
            rgy_cmd_help_vpp_sections(layout) == layout);
    }
    _ftprintf(stdout, _T("[HELP-FIXTURE %s] aligned descriptions\n"), layoutOk ? _T("OK") : _T("NG"));
    if (!valuesOk) ++failures;
    if (!sectionsOk) ++failures;
    _ftprintf(stdout, _T("[HELP-FIXTURE %s] shared values\n"), valuesOk ? _T("OK") : _T("NG"));
    _ftprintf(stdout, _T("[HELP-FIXTURE %s] grouped sections\n"), sectionsOk ? _T("OK") : _T("NG"));
    _ftprintf(stdout, _T("help fixtures total %d, NG %d\n"), (int)cases.size() + 3, failures);
    return failures == 0 ? 1 : -1;
}

#endif
