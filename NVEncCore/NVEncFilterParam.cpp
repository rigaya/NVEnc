// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#include <sstream>
#include <numeric>
#include <iomanip>
#include "NVEncFilterParam.h"
#include "rgy_cmd.h"

VppCustom::VppCustom() :
    enable(false),
    filter_name(),
    kernel_name(FILTER_DEFAULT_CUSTOM_KERNEL_NAME),
    kernel_path(),
    kernel(),
    dev_params(nullptr),
    compile_options(),
    kernel_interface(VPP_CUSTOM_INTERFACE_PER_PLANE),
    interlace(VPP_CUSTOM_INTERLACE_UNSUPPORTED),
    threadPerBlockX(FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_X),
    threadPerBlockY(FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_Y),
    pixelPerThreadX(FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_X),
    pixelPerThreadY(FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_Y),
    dstWidth(0),
    dstHeight(0),
    params() {

}

bool VppCustom::operator==(const VppCustom &x) const {
    return enable == x.enable
        && filter_name == x.filter_name
        && kernel_name == x.kernel_name
        && kernel_path == x.kernel_path
        && kernel == x.kernel
        && dev_params == x.dev_params
        && compile_options == x.compile_options
        && kernel_interface == x.kernel_interface
        && interlace == x.interlace
        && threadPerBlockX == x.threadPerBlockX
        && threadPerBlockY == x.threadPerBlockY
        && pixelPerThreadX == x.pixelPerThreadX
        && pixelPerThreadY == x.pixelPerThreadY
        && dstWidth == x.dstWidth
        && dstHeight == x.dstHeight
        && params == x.params;
}
bool VppCustom::operator!=(const VppCustom &x) const {
    return !(*this == x);
}

tstring VppCustom::print() const {
    return strsprintf(_T("%s: %s, interface %s, interlace %s\n")
        _T("                    thread/block (%d,%d), pixel/thread (%d,%d)\n"),
        filter_name.c_str(), kernel_path.c_str(),
        get_cx_desc(list_vpp_custom_interface, kernel_interface),
        get_cx_desc(list_vpp_custom_interlace, interlace),
        threadPerBlockX, threadPerBlockY,
        pixelPerThreadX, pixelPerThreadY);
}

VppNvvfxDenoise::VppNvvfxDenoise() :
    enable(false),
    strength(FILTER_DEFAULT_NVVFX_DENOISE_STRENGTH) {

}

bool VppNvvfxDenoise::operator==(const VppNvvfxDenoise &x) const {
    return enable == x.enable
        && strength == x.strength;
}
bool VppNvvfxDenoise::operator!=(const VppNvvfxDenoise &x) const {
    return !(*this == x);
}

tstring VppNvvfxDenoise::print() const {
    return strsprintf(_T("nvvfx-denoise: strength %.0f"),
        strength);
}

VppNvvfxArtifactReduction::VppNvvfxArtifactReduction() :
    enable(false),
    mode(FILTER_DEFAULT_NVVFX_ARTIFACT_REDUCTION_MODE) {

}

bool VppNvvfxArtifactReduction::operator==(const VppNvvfxArtifactReduction &x) const {
    return enable == x.enable
        && mode == x.mode;
}
bool VppNvvfxArtifactReduction::operator!=(const VppNvvfxArtifactReduction &x) const {
    return !(*this == x);
}

tstring VppNvvfxArtifactReduction::print() const {
    return strsprintf(_T("nvvfx-artifact-reduction: mode %d (%s)"),
        mode, get_cx_desc(list_vpp_nvvfx_mode, mode));
}

VppNvvfxSuperRes::VppNvvfxSuperRes() :
    enable(false),
    mode(FILTER_DEFAULT_NVVFX_SUPER_RES_MODE),
    strength(FILTER_DEFAULT_NVVFX_SUPER_RES_STRENGTH) {

}

bool VppNvvfxSuperRes::operator==(const VppNvvfxSuperRes &x) const {
    return enable == x.enable
        && mode == x.mode
        && strength == x.strength;
}
bool VppNvvfxSuperRes::operator!=(const VppNvvfxSuperRes &x) const {
    return !(*this == x);
}

tstring VppNvvfxSuperRes::print() const {
    return strsprintf(_T("nvvfx-superres: mode: %d (%s), strength %.2f"),
        mode, get_cx_desc(list_vpp_nvvfx_mode, mode), strength);
}

VppNvvfxUpScaler::VppNvvfxUpScaler() :
    enable(false),
    strength(FILTER_DEFAULT_NVVFX_UPSCALER_STRENGTH) {

}

bool VppNvvfxUpScaler::operator==(const VppNvvfxUpScaler &x) const {
    return enable == x.enable
        && strength == x.strength;
}
bool VppNvvfxUpScaler::operator!=(const VppNvvfxUpScaler &x) const {
    return !(*this == x);
}

tstring VppNvvfxUpScaler::print() const {
    return strsprintf(_T("nvvfx-upscaler: strength %.2f"),
        strength);
}

VppNGXVSR::VppNGXVSR() :
    enable(false),
    quality(FILTER_DEFAULT_NGX_VSR_QUALITY) {
}

bool VppNGXVSR::operator==(const VppNGXVSR& x) const {
    return (enable == x.enable && quality == x.quality);
}

bool VppNGXVSR::operator!=(const VppNGXVSR& x) const {
    return !(*this == x);
}

tstring VppNGXVSR::print() const {
    return strsprintf(_T("nvsdk-ngx vsr: quality: %d"), quality);
}

VppNGXTrueHDR::VppNGXTrueHDR() :
    enable(false),
    contrast(FILTER_DEFAULT_NGX_TRUEHDR_CONTRAST),
    saturation(FILTER_DEFAULT_NGX_TRUEHDR_SATURATION),
    middleGray(FILTER_DEFAULT_NGX_TRUEHDR_MIDDLE_GRAY),
    maxLuminance(FILTER_DEFAULT_NGX_TRUEHDR_MAX_LUMINANCE) {
}

bool VppNGXTrueHDR::operator==(const VppNGXTrueHDR &x) const {
    return enable == x.enable &&
        contrast == x.contrast &&
        saturation == x.saturation &&
        middleGray == x.middleGray &&
        maxLuminance == x.maxLuminance;
}

bool VppNGXTrueHDR::operator!=(const VppNGXTrueHDR &x) const {
    return !(*this == x);
}

tstring VppNGXTrueHDR::print() const {
    return strsprintf(_T("ngx-truehdr:")
        _T(" contrast: %d,")
        _T(" saturation: %d,")
        _T(" middleGray: %d,")
        _T(" maxLuminance: %d"),
        contrast, saturation, middleGray, maxLuminance);
}

VppParam::VppParam() :
#if ENCODER_NVENC
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    gaussMaskSize((NppiMaskSize)0),
#endif //#if ENCODER_NVENC
    nvvfxDenoise(),
    nvvfxArtifactReduction(),
    nvvfxSuperRes(),
    nvvfxUpScaler(),
    nvvfxModelDir(),
    ngxVSR(),
    ngxTrueHDR() {
}


bool VppParam::operator==(const VppParam &x) const {
        return
#if ENCODER_NVENC
            deinterlace == x.deinterlace &&
           gaussMaskSize == x.gaussMaskSize &&
#endif //#if ENCODER_NVENC
           nvvfxDenoise == x.nvvfxDenoise
        && nvvfxArtifactReduction == x.nvvfxArtifactReduction
        && nvvfxSuperRes == x.nvvfxSuperRes
        && nvvfxUpScaler == x.nvvfxUpScaler
        && nvvfxModelDir == x.nvvfxModelDir;

}
bool VppParam::operator!=(const VppParam &x) const {
        return !(*this == x);
}

int parse_one_vppnv_option(const TCHAR* option_name, const TCHAR* strInput[], int& i, [[maybe_unused]] int nArgNum, VppParam* vppnv, [[maybe_unused]] sArgsData* argData, RGY_VPP_RESIZE_ALGO& resize_algo) {
#if ENCODER_NVENC
    if (IS_OPTION("vpp-deinterlace")) {
        i++;
        int value = 0;
        if (get_list_value(list_deinterlace, strInput[i], &value)) {
            vppnv->deinterlace = (cudaVideoDeinterlaceMode)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_deinterlace);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-gauss")) {
        i++;
        int value = 0;
        if (get_list_value(list_nppi_gauss, strInput[i], &value)) {
            vppnv->gaussMaskSize = (NppiMaskSize)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_nppi_gauss);
            return 1;
        }
        return 0;
    }
#endif //#if ENCODER_NVENC
    if (IS_OPTION("vpp-nvvfx-model-dir") && (ENABLE_NVVFX || FOR_AUO)) {
        i++;
        vppnv->nvvfxModelDir = strInput[i];
        return 0;
    }
    if (IS_OPTION("vpp-resize") && (ENABLE_NVVFX || FOR_AUO)) {
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "superres-mode", "superres-strength", "vsr-quality" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vppnv->nvvfxSuperRes.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("algo")) {
                    int value = 0;
                    if (get_list_value(list_vpp_resize, param_val.c_str(), &value)) {
                        resize_algo = (RGY_VPP_RESIZE_ALGO)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_resize);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("superres-mode")) {
                    try {
                        vppnv->nvvfxSuperRes.mode = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("superres-strength")) {
                    try {
                        vppnv->nvvfxSuperRes.strength = std::stof(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("vsr-quality")) {
                    try {
                        vppnv->ngxVSR.quality = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {

                int value = 0;
                if (get_list_value(list_vpp_resize, param.c_str(), &value)) {
                    resize_algo = (RGY_VPP_RESIZE_ALGO)value;
                } else {
                    print_cmd_error_invalid_value(tstring(option_name), param, list_vpp_resize);
                    return 1;
                }
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-nvvfx-denoise") && (ENABLE_NVVFX || FOR_AUO)) {
        vppnv->nvvfxDenoise.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "strength" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vppnv->nvvfxDenoise.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vppnv->nvvfxDenoise.strength = std::stof(param_val);
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

    if (IS_OPTION("vpp-nvvfx-artifact-reduction") && (ENABLE_NVVFX || FOR_AUO)) {
        vppnv->nvvfxArtifactReduction.enable = true;
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
                        vppnv->nvvfxArtifactReduction.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    try {
                        vppnv->nvvfxArtifactReduction.mode = std::stoi(param_val);
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

    if (IS_OPTION("vpp-nvvfx-upscaler") && (ENABLE_NVVFX || FOR_AUO)) {
        vppnv->nvvfxUpScaler.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "strength" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vppnv->nvvfxUpScaler.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vppnv->nvvfxUpScaler.strength = std::stof(param_val);
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

    if (IS_OPTION("vpp-ngx-truehdr") && (ENABLE_NVSDKNGX || FOR_AUO)) {
        vppnv->ngxTrueHDR.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;
        const auto paramList = std::vector<std::string>{ "contrast", "saturation", "middlegray", "maxluminance" };
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vppnv->ngxTrueHDR.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        vppnv->ngxTrueHDR.contrast = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("saturation")) {
                    try {
                        vppnv->ngxTrueHDR.saturation = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("middlegray")) {
                    try {
                        vppnv->ngxTrueHDR.middleGray = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("maxluminance")) {
                    try {
                        vppnv->ngxTrueHDR.maxLuminance = std::stoi(param_val);
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
    return -1;
}

tstring gen_cmd(const VppParam *param, const VppParam *defaultPrm, RGY_VPP_RESIZE_ALGO resize_algo, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
#define OPT_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (param->opt);
#define OPT_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(param->opt);
#define OPT_GUID(str, opt, list) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << get_name_from_guid((param->opt), list);
#define OPT_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (param->opt));

#define OPT_TCHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << (param->opt);
#define OPT_TSTR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << param->opt.c_str();
#define OPT_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt);
#define OPT_STR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt).c_str();
#define OPT_CHAR_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" \"") << (param->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (param->opt.c_str()) << _T("\"");

#define ADD_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (param->opt);
#define ADD_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << (param->opt);
#define ADD_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (param->opt));
#define ADD_BOOL(str, opt) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << ((param->opt) ? (_T("true")) : (_T("false")));
#define ADD_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) tmp << _T(",") << (str) << _T("=") << (param->opt);
#define ADD_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) tmp << _T(",") << (str) << _T("=\"") << (param->opt) << _T("\"");
#define ADD_STR(str, opt) if (param->opt.length() > 0) tmp << _T(",") << (str) << _T("=") << (param->opt.c_str());

#if ENCODER_NVENC
    OPT_LST(_T("--vpp-deinterlace"), deinterlace, list_deinterlace);
    OPT_LST(_T("--vpp-gauss"), gaussMaskSize, list_nppi_gauss);
#endif

#if (ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    if (resize_algo == RGY_VPP_RESIZE_NGX_VSR) {
        cmd << _T(" --vpp-resize ") << get_chr_from_value(list_vpp_resize, resize_algo);
        if (param->ngxVSR.quality != defaultPrm->ngxVSR.quality) {
            cmd << _T(",quality=") << param->ngxVSR.quality;
        }
    } else if (resize_algo == RGY_VPP_RESIZE_NVVFX_SUPER_RES) {
        cmd << _T(" --vpp-resize ") << get_chr_from_value(list_vpp_resize, resize_algo);
        if (param->nvvfxSuperRes.mode != defaultPrm->nvvfxSuperRes.mode) {
            cmd << _T(",superres-mode=") << param->nvvfxSuperRes.mode;
        }
        if (param->nvvfxSuperRes.strength != defaultPrm->nvvfxSuperRes.strength) {
            cmd << _T(",superres-strength=") << param->nvvfxSuperRes.strength;
        }
    }
#endif

    std::basic_stringstream<TCHAR> tmp;
    if (param->nvvfxDenoise != defaultPrm->nvvfxDenoise) {
        tmp.str(tstring());
        if (!param->nvvfxDenoise.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nvvfxDenoise.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), nvvfxDenoise.strength, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nvvfx-denoise ") << tmp.str().substr(1);
        } else if (param->nvvfxDenoise.enable) {
            cmd << _T(" --vpp-nvvfx-denoise");
        }
    }

    if (param->nvvfxArtifactReduction != defaultPrm->nvvfxArtifactReduction) {
        tmp.str(tstring());
        if (!param->nvvfxArtifactReduction.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nvvfxArtifactReduction.enable || save_disabled_prm) {
            ADD_NUM(_T("mode"), nvvfxArtifactReduction.mode);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nvvfx-artifact-reduction ") << tmp.str().substr(1);
        } else if (param->nvvfxArtifactReduction.enable) {
            cmd << _T(" --vpp-nvvfx-artifact-reduction");
        }
    }

#if (ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    if (param->nvvfxSuperRes != defaultPrm->nvvfxSuperRes && resize_algo == RGY_VPP_RESIZE_NVVFX_SUPER_RES) {
        tmp.str(tstring());
        //if (!param->nvvfxSuperRes.enable && save_disabled_prm) {
        //    tmp << _T(",enable=false");
        //}
        //if (param->nvvfxSuperRes.enable || save_disabled_prm) {
            ADD_NUM(_T("superres-mode"), nvvfxSuperRes.mode);
            ADD_FLOAT(_T("superres-strength"), nvvfxSuperRes.strength, 3);
        //}
        cmd << _T(" --vpp-resize algo=nvvfx-superres");
        if (!tmp.str().empty()) {
            cmd << tmp.str();
        }
    }
#endif

    if (param->nvvfxUpScaler != defaultPrm->nvvfxUpScaler) {
        tmp.str(tstring());
        if (!param->nvvfxUpScaler.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->nvvfxDenoise.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), nvvfxUpScaler.strength, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-nvvfx-upscale ") << tmp.str().substr(1);
        } else if (param->nvvfxUpScaler.enable) {
            cmd << _T(" --vpp-nvvfx-upscale");
        }
    }

    OPT_STR_PATH(_T("--vpp-nvvfx-model-dir"), nvvfxModelDir);

    if (param->ngxTrueHDR != defaultPrm->ngxTrueHDR) {
        tmp.str(tstring());
        if (!param->ngxTrueHDR.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->ngxTrueHDR.enable || save_disabled_prm) {
            ADD_NUM(_T("contrast"), ngxTrueHDR.contrast);
            ADD_NUM(_T("saturation"), ngxTrueHDR.saturation);
            ADD_NUM(_T("middlegray"), ngxTrueHDR.middleGray);
            ADD_NUM(_T("maxluminance"), ngxTrueHDR.maxLuminance);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-ngx-truehdr ") << tmp.str().substr(1);
        } else if (param->ngxTrueHDR.enable) {
            cmd << _T(" --vpp-ngx-truehdr");
        }
    }
    return cmd.str();
}
