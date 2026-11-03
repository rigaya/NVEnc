#ifndef __RGY_HELP_CODEGEN_CASES_H__
#define __RGY_HELP_CODEGEN_CASES_H__

#include "NVEncCmd.h"
#include "rgy_cmd_selftest.h"

// help整形とは独立に、今回見つかったパース精度と生成漏れの回帰を確認する。
static inline int rgy_help_codegen_cases() {
    RGYCmdSelfTest<InEncodeVideoParam> checker(
        [](InEncodeVideoParam *prm, const std::vector<tstring>& args) {
            auto argv = rgy_cmd_selftest_argv(args);
            return parse_cmd(prm, (int)argv.size() - 1, argv.data());
        },
        [](const InEncodeVideoParam *prm) { return gen_cmd(prm, false); });
    const auto roundTrips = checker.runRoundTripCases({
        {
            _T("overlay preserves explicit default alpha mode"),
            { _T("--vpp-overlay"), _T("alpha_mode=override") },
            false,
            { _T("--vpp-overlay"), _T("alpha_mode=override") }, {}
        },
        {
            _T("overlay preserves multiply alpha mode"),
            { _T("--vpp-overlay"), _T("alpha_mode=mul") },
            false,
            { _T("alpha_mode=mul") }, {}
        },
        {
            _T("overlay preserves luma key alpha mode"),
            { _T("--vpp-overlay"), _T("alpha_mode=lumakey") },
            false,
            { _T("alpha_mode=lumakey") }, {}
        },
        {
            _T("overlay preserves all canonical parameters"),
            { _T("--vpp-overlay"), _T("file=overlay.png,pos=12x34,size=320x180,alpha=0.625,alpha_mode=lumakey,lumakey_threshold=0.3,lumakey_tolerance=0.15,lumakey_softness=0.2,loop=true") },
            false,
            { _T("file=\"overlay.png\""), _T("pos=12x34"), _T("size=320x180"), _T("alpha=0.625"), _T("alpha_mode=lumakey"),
              _T("lumakey_threshold=0.3"), _T("lumakey_tolerance=0.15"), _T("lumakey_softness=0.2"), _T("loop=true") }, {}
        },
        {
            _T("overlay normalizes position and size aliases"),
            { _T("--vpp-overlay"), _T("posx=12,posy=34,width=320,height=180") },
            false,
            { _T("pos=12x34"), _T("size=320x180") }, {}
        },
        {
            _T("overlay preserves multiple default elements"),
            { _T("--vpp-overlay"), _T("alpha_mode=override"), _T("--vpp-overlay"), _T("alpha_mode=override") },
            false,
            { _T("--vpp-overlay alpha_mode=override --vpp-overlay alpha_mode=override") }, {}
        },
        {
            _T("overlay saves disabled alpha mode"),
            { _T("--vpp-overlay"), _T("enable=false,alpha_mode=mul") },
            true,
            { _T("--vpp-overlay"), _T("enable=false"), _T("alpha_mode=mul") }, {}
        },
        {
            _T("overlay omits disabled alpha mode without save flag"),
            { _T("--vpp-overlay"), _T("enable=false,alpha_mode=mul") },
            false,
            {}, { _T("--vpp-overlay") }
        },
        {
            _T("colorspace preserves chained matrix operations"),
            { _T("--vpp-colorspace"), _T("matrix=bt709:smpte170m,matrix=smpte170m:bt709") },
            false,
            { _T("matrix=bt709:smpte170m"), _T("matrix=smpte170m:bt709") }, {}
        },
        {
            _T("colorspace preserves chained colorprim operations"),
            { _T("--vpp-colorspace"), _T("colorprim=bt709:bt2020,colorprim=bt2020:bt709") },
            false,
            { _T("colorprim=bt709:bt2020"), _T("colorprim=bt2020:bt709") }, {}
        },
        {
            _T("colorspace preserves chained transfer operations"),
            { _T("--vpp-colorspace"), _T("transfer=bt709:linear,transfer=linear:bt709") },
            false,
            { _T("transfer=bt709:linear"), _T("transfer=linear:bt709") }, {}
        },
        {
            _T("colorspace preserves chained range operations"),
            { _T("--vpp-colorspace"), _T("range=limited:full,range=full:limited") },
            false,
            { _T("range=limited:full"), _T("range=full:limited") }, {}
        },
        {
            _T("colorspace default desat_base keeps double precision"),
            { _T("--vpp-colorspace"), _T("desat_base=0.18") },
            false,
            { _T("--vpp-colorspace") },
            { _T("desat_base=") }
        },
        {
            _T("deband preserves a non-default seed"),
            { _T("--vpp-deband"), _T("seed=9876") },
            false,
            { _T("--vpp-deband"), _T("seed=9876") },
            {}
        },
        {
            _T("deband saves a disabled filter seed"),
            { _T("--vpp-deband"), _T("enable=false,seed=9876") },
            true,
            { _T("--vpp-deband"), _T("enable=false"), _T("seed=9876") },
            {}
        },
        {
            _T("deband omits disabled seed without save flag"),
            { _T("--vpp-deband"), _T("enable=false,seed=9876") },
            false,
            {},
            { _T("--vpp-deband"), _T("seed=9876") }
        }
    }, [](const InEncodeVideoParam *prm, bool saveDisabledPrm) { return gen_cmd(prm, saveDisabledPrm); });

    int stateTotal = 0, stateNg = 0;
    const auto check = [&](bool ok, const TCHAR *name) {
        ++stateTotal;
        if (!ok) ++stateNg;
        _ftprintf(stdout, _T("[CODEGEN-STATE %s] %s\n"), ok ? _T("OK") : _T("NG"), name);
    };
    {
        InEncodeVideoParam prm;
        const std::vector<tstring> args = {
            _T("--vpp-colorspace"), _T("desat_base=0.18"),
            _T("-i"), _T("rgy_codegen_input.yuv"), _T("-o"), _T("rgy_codegen_output.raw")
        };
        auto argv = rgy_cmd_selftest_argv(args);
        auto& error = rgy_cmd_error_info();
        const bool quiet = error.quiet;
        error.quiet = true;
        error.reset();
        const bool parsed = parse_cmd(&prm, (int)argv.size() - 1, argv.data()) == 0 && !error.errorOccurred;
        error.quiet = quiet;
        check(parsed && prm.vpp.colorspace.hdr2sdr.desat_base == FILTER_DEFAULT_HDR2SDR_DESAT_BASE,
            _T("parsed desat_base equals the double default exactly"));
    }
    {
        InEncodeVideoParam prm;
        prm.vpp.overlay.emplace_back();
        prm.vpp.overlay.back().enable = true;
        // 引数なしoverlayのパーサは状態を追加しないため、生成分岐の独立性は状態を直接作って確認する。
        // ここではエンコードを行わず、debandの状態がoverlayの出力を左右しないことだけを試す。
        prm.vpp.deband.enable = false;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) != tstring::npos,
            _T("enabled default overlay is emitted with deband disabled"));
        prm.vpp.deband.enable = true;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) != tstring::npos,
            _T("enabled default overlay is emitted with deband enabled"));
        prm.vpp.overlay.back().enable = false;
        prm.vpp.overlay.back().alpha = 0.5f;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) == tstring::npos,
            _T("disabled overlay is not emitted by enabled deband"));
        const auto saved = gen_cmd(&prm, true);
        check(saved.find(_T("--vpp-overlay")) != tstring::npos
            && saved.find(_T("enable=false")) != tstring::npos
            && saved.find(_T("alpha=")) != tstring::npos,
            _T("disabled overlay parameters are kept with save flag"));
    }
    _ftprintf(stdout, _T("codegen state total %d, NG %d\n"), stateTotal, stateNg);
    return roundTrips > 0 && stateNg == 0 ? 1 : -1;
}

#endif
