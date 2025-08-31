// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <tinyxml2.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "auo_util.h"
#include "auo_settings.h"
#include "auo_version.h"

static const int INI_SECTION_BUFSIZE = 32768;
static const int INI_KEY_MAX_LEN = 256;

static const int INI_VER_MIN = 2;
static const int INI_VER_UTF8 = 3;
static const int CNF_VER_UTF8 = 1;

static const TCHAR * const INI_APPENDIX  = _T(".ini");
static const TCHAR * const CONF_APPENDIX = _T(".conf");

static const char * const STG_DEFAULT_DIRECTORY_APPENDIX = "_stg";

//----    セクション名    ---------------------------------------------------

#if ENCODER_X264
static const char * const INI_SECTION_MAIN         = "X264GUIEX";
static const char * const INI_SECTION_ENC          = "X264";
static const char * const INI_SECTION_ENC_DEFAULT  = "X264_DEFAULT";
static const char * const INI_SECTION_ENC_PRESET   = "X264_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "X264_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "X264_PROFILE";
#elif ENCODER_X265
static const char * const INI_SECTION_MAIN         = "X265GUIEX"; //CONF_VER
static const char * const INI_SECTION_MAIN_OLD     = "X26XGUIEX"; //CONF_VER_OLD
static const char * const INI_SECTION_ENC          = "X265";
static const char * const INI_SECTION_ENC_DEFAULT  = "X265_DEFAULT";
static const char * const INI_SECTION_ENC_PRESET   = "X265_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "X265_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "X265_PROFILE";
#elif ENCODER_SVTAV1
static const char * const INI_SECTION_MAIN         = "SVTAV1GUIEX";
static const char * const INI_SECTION_ENC          = "ENC";
static const char * const INI_SECTION_ENC_DEFAULT  = "ENC_DEFAULT";
static const char * const INI_SECTION_ENC_PRESET   = "ENC_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "ENC_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "ENC_PROFILE";
#elif ENCODER_QSV
static const char * const INI_SECTION_MAIN         = "QSVENC";
static const char * const INI_SECTION_ENC          = "VIDEO";
static const char * const INI_SECTION_ENC_DEFAULT  = "VIDEO";
static const char * const INI_VID_FILENAME         = "qsvencc";
static const char * const INI_SECTION_ENC_PRESET   = "ENC_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "ENC_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "ENC_PROFILE";
#elif ENCODER_NVENC
static const char * const INI_SECTION_MAIN         = "NVENC";
static const char * const INI_SECTION_ENC          = "VIDEO";
static const char * const INI_SECTION_ENC_DEFAULT  = "VIDEO";
static const char * const INI_VID_FILENAME         = "nvencc";
static const char * const INI_SECTION_ENC_PRESET   = "ENC_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "ENC_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "ENC_PROFILE";
#elif ENCODER_VCEENC
static const char * const INI_SECTION_MAIN         = "VCEENC";
static const char * const INI_SECTION_ENC          = "VIDEO";
static const char * const INI_SECTION_ENC_DEFAULT  = "VIDEO";
static const char * const INI_VID_FILENAME         = "vceencc";
static const char * const INI_SECTION_ENC_PRESET   = "ENC_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "ENC_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "ENC_PROFILE";
#elif ENCODER_FFMPEG
static const char * const INI_SECTION_MAIN         = "FFMPEGOUT";
static const char * const INI_SECTION_ENC          = "FFMPEGOUT";
static const char * const INI_SECTION_ENC_DEFAULT  = "FFMPEGOUT";
static const char * const INI_VID_FILENAME         = "ffmpeg";
static const char * const INI_SECTION_ENC_PRESET   = "ENC_PRESET";
static const char * const INI_SECTION_ENC_TUNE     = "ENC_TUNE";
static const char * const INI_SECTION_ENC_PROFILE  = "ENC_PROFILE";
#else
static_assert(false);
#endif

static const char * const INI_SECTION_APPENDIX     = "APPENDIX";
static const char * const INI_SECTION_AUD          = "AUDIO";
static const char * const INI_SECTION_AUD_INTERNAL = "AUDIO_INTERNAL";
static const char * const INI_SECTION_MUX          = "MUXER";
static const char * const INI_SECTION_FN           = "FILENAME_REPLACE";
static const char * const INI_SECTION_PREFIX       = "SETTING_";
static const char * const INI_SECTION_MODE         = "MODE_";
static const char * const INI_SECTION_FBC          = "BITRATE_CALC";
static const char * const INI_SECTION_AMP          = "AUTO_MULTI_PASS";

static inline double GetPrivateProfileDouble(const char *section, const char *keyname, double defaultValue, const char *ini_file) {
    char buf[INI_KEY_MAX_LEN], str_default[64], *eptr;
    double d;
    sprintf_s(str_default, _countof(str_default), "%f", defaultValue);
    GetPrivateProfileStringA(section, keyname, str_default, buf, _countof(buf), ini_file);
    d = strtod(buf, &eptr);
    if (*eptr == '\0') return d;
    return defaultValue;
}

static inline void GetFontInfo(const char *section, const char *keyname_base, AUO_FONT_INFO *font_info, const char *ini_file) {
    const size_t keyname_base_len = strlen(keyname_base);
    if (keyname_base_len >= INI_KEY_MAX_LEN)
        return;
    char key[INI_KEY_MAX_LEN];
    memcpy(key, keyname_base, sizeof(key[0]) * (keyname_base_len + 1));
    strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_name");
    GetPrivateProfileTStg(section, key, _T(""), font_info->name, _countof(font_info->name), ini_file, CP_THREAD_ACP);
    strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_size");
    font_info->size = GetPrivateProfileDouble(section, key, 0.0, ini_file);
    strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_style");
    font_info->style = GetPrivateProfileIntA(section, key, 0, ini_file);
}

static inline void GetColorInfo(const char *section, const char *keyname, int *color_rgb, const int *default_color_rgb, const char *ini_file) {
    char buf[INI_KEY_MAX_LEN], str_default[64];
    sprintf_s(str_default, _countof(str_default), "%d,%d,%d", default_color_rgb[0], default_color_rgb[1], default_color_rgb[2]);
    GetPrivateProfileStringA(section, keyname, str_default, buf, _countof(buf), ini_file);
    if (3 != sscanf_s(buf, "%d,%d,%d", &color_rgb[0], &color_rgb[1], &color_rgb[2]))
        memcpy(color_rgb, default_color_rgb, sizeof(color_rgb[0]) * 3);
    for (int i = 0; i < 3; i++)
        color_rgb[i] = clamp(color_rgb[i], 0, 255);
}

static inline void WritePrivateProfileInt(const char *section, const char *keyname, int value, const char *ini_file) {
    char tmp[22];
    sprintf_s(tmp, _countof(tmp), "%d", value);
    WritePrivateProfileStringA(section, keyname, tmp, ini_file);
}

static inline void WritePrivateProfileIntWithDefault(const char *section, const char *keyname, int value, int _default, const char *ini_file) {
    if (value != (int)GetPrivateProfileIntA(section, keyname, _default, ini_file))
        WritePrivateProfileInt(section, keyname, value, ini_file);
}

static inline void WritePrivateProfileDouble(const char *section, const char *keyname, double value, const char *ini_file) {
    char tmp[32];
    sprintf_s(tmp, _countof(tmp), "%lf", value);
    WritePrivateProfileStringA(section, keyname, tmp, ini_file);
}

static inline void WritePrivateProfileDoubleWithDefault(const char *section, const char *keyname, double value, double _default, const char *ini_file) {
    if (abs(value - GetPrivateProfileDouble(section, keyname, _default, ini_file)) > 1.0e-6)
        WritePrivateProfileDouble(section, keyname, value, ini_file);
}

static inline void WritePrivateProfileW(const char *section, const char *keyname, const TCHAR *value, const char *ini_file, const DWORD codepage) {
    const auto wstr = wstring_to_string(value, codepage);
    WritePrivateProfileStringA(section, keyname, wstr.c_str(), ini_file);
}

static inline void WriteFontInfo(const char *section, const char *keyname_base, AUO_FONT_INFO *font_info, const char *ini_file) {
    const size_t keyname_base_len = strlen(keyname_base);
    if (keyname_base_len >= INI_KEY_MAX_LEN)
        return;

    AUO_FONT_INFO current_info = { 0 };
    GetFontInfo(section, keyname_base, &current_info, ini_file);

    char key[INI_KEY_MAX_LEN];
    memcpy(key, keyname_base, sizeof(key[0]) * (keyname_base_len + 1));
    if (str_has_char(font_info->name)) {
        strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_name");
        WritePrivateProfileW(section, key, font_info->name, ini_file, CP_THREAD_ACP);
    }
    if (font_info->size > 0.0 || font_info->size != current_info.size) {
        strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_size");
        WritePrivateProfileDouble(section, key, font_info->size, ini_file);
    }
    if (font_info->style != 0 || font_info->style != current_info.style) {
        strcpy_s(key + keyname_base_len, _countof(key) - keyname_base_len, "_style");
        WritePrivateProfileInt(section, key, font_info->style, ini_file);
    }
}

static inline void WriteColorInfo(const char *section, const char *keyname, int *color_rgb, const int *default_color_rgb, const char *ini_file) {
    int current_color[3] = { 0 };
    GetColorInfo(section, keyname, current_color, default_color_rgb, ini_file);
    if (0 != memcmp(color_rgb, current_color, sizeof(color_rgb[0]) * 3)) {
        char buf[256];
        sprintf_s(buf, _countof(buf), "%d,%d,%d", color_rgb[0], color_rgb[1], color_rgb[2]);
        WritePrivateProfileStringA(section, keyname, buf, ini_file);
    }
}



BOOL  guiEx_settings::init = FALSE;
char  guiEx_settings::ini_section_main[256] = { 0 };
TCHAR guiEx_settings::auo_path[MAX_PATH_LEN] = { 0 };
char  guiEx_settings::ini_fileName[MAX_PATH_LEN] = { 0 };
char  guiEx_settings::conf_fileName[MAX_PATH_LEN] = { 0 };
DWORD guiEx_settings::ini_filesize = 0;
TCHAR guiEx_settings::default_lang[4] = { 0 };
int   guiEx_settings::ini_ver = 0;
DWORD guiEx_settings::codepage_ini = 0;
DWORD guiEx_settings::codepage_cnf = 0;
TCHAR  guiEx_settings::language[MAX_PATH_LEN] = { 0 };
TCHAR  guiEx_settings::blog_url[MAX_PATH_LEN] = { 0 };

guiEx_settings::guiEx_settings() {
    initialize(false);
}

guiEx_settings::guiEx_settings(BOOL disable_loading) {
    initialize(disable_loading);
}

guiEx_settings::guiEx_settings(BOOL disable_loading, const TCHAR *_auo_path, const char *main_section) {
    initialize(disable_loading, _auo_path, main_section);
}

void guiEx_settings::initialize(BOOL disable_loading) {
    initialize(disable_loading, NULL, NULL);
}

void guiEx_settings::initialize(BOOL disable_loading, const TCHAR *_auo_path, const char *main_section) {
    s_aud_ext_count = 0;
    s_aud_int_count = 0;
    s_mux_count = 0;
    s_aud_int = NULL;
    s_aud_ext = NULL;
    s_mux = NULL;
    ZeroMemory(&s_enc, sizeof(s_enc));
    ZeroMemory(&s_local, sizeof(s_local));
    ZeroMemory(&s_log, sizeof(s_log));
    ZeroMemory(&s_append, sizeof(s_append));
    ZeroMemory(&language, sizeof(language));
    ZeroMemory(&last_out_stg, sizeof(last_out_stg));
    if (!init) {
        if (_tcslen(default_lang) == 0) {
            get_default_lang();
        }
        if (_auo_path == NULL) {
            get_auo_path(auo_path, _countof(auo_path));
        } else {
            _tcscpy_s(auo_path, _countof(auo_path), _auo_path);
        }
        TCHAR conf_fileName_tstr[MAX_PATH_LEN] = { 0 };
        apply_appendix(conf_fileName_tstr, _countof(conf_fileName_tstr), auo_path, CONF_APPENDIX);
        if (!canbe_converted_to(conf_fileName_tstr, CP_THREAD_ACP)) {
            // CP_THREAD_ACP = sjisに変換できない場合は、相対パスにする
            TCHAR conf_fileName_tstr_relative[MAX_PATH_LEN];
            GetRelativePathTo(conf_fileName_tstr_relative, _countof(conf_fileName_tstr_relative), conf_fileName_tstr, NULL);
            _tcscpy_s(conf_fileName_tstr, conf_fileName_tstr_relative);
        }
        strcpy_s(conf_fileName, tchar_to_string(conf_fileName_tstr, CP_THREAD_ACP).c_str());
        strcpy_s(ini_section_main, _countof(ini_section_main), (main_section == NULL) ? INI_SECTION_MAIN : main_section);
        const int cnf_ver = GetPrivateProfileIntA(ini_section_main, "cnf_ver", 0, conf_fileName);
        codepage_cnf = cnf_ver >= CNF_VER_UTF8 ? CP_UTF8 : CP_THREAD_ACP;

        load_lang();
        TCHAR ini_fileName_tstr[MAX_PATH_LEN] = { 0 };
        bool language_ini_selected = false;
        const auto language_str = wstring_to_string(language);
        for (const auto& auo_lang : list_auo_languages) {
            if (   _tcscmp(language, auo_lang.code) == 0
                && _tcscmp(_T("ja"), auo_lang.code) != 0) { // 日本語用x264guiEx.iniはx264guiEx.iniのまま
                TCHAR ini_append[64];
                _stprintf_s(ini_append, _T(".%s%s"), language, INI_APPENDIX);
                apply_appendix(ini_fileName_tstr, _countof(ini_fileName_tstr), auo_path, ini_append);
                if (PathFileExists(ini_fileName_tstr)) {
                    language_ini_selected = true;
                }
            }
        }
        if (!language_ini_selected) {
            TCHAR auo_dir[MAX_PATH_LEN];
            get_auo_path(auo_dir, _countof(auo_dir));
            PathRemoveFileSpecFixed(auo_dir);
            TCHAR lng_path[MAX_PATH_LEN];
            PathCombineLong(lng_path, _countof(lng_path), auo_dir, language);
            if (PathFileExists(lng_path)) {
                const auto lang_code = get_file_lang_code(lng_path);
                TCHAR ini_append[64];
                _stprintf_s(ini_append, _T(".%s%s"), lang_code.c_str(), INI_APPENDIX);
                apply_appendix(ini_fileName_tstr, _countof(ini_fileName_tstr), auo_path, ini_append);
                if (PathFileExists(ini_fileName_tstr)) {
                    language_ini_selected = true;
                }
            }
        }
        if (!language_ini_selected) {
            apply_appendix(ini_fileName_tstr, _countof(ini_fileName_tstr), auo_path, INI_APPENDIX);
        }
        if (!canbe_converted_to(ini_fileName_tstr, CP_THREAD_ACP)) {
            // CP_THREAD_ACP = sjisに変換できない場合は、相対パスにする
            TCHAR ini_fileName_tstr_relative[MAX_PATH_LEN];
            GetRelativePathTo(ini_fileName_tstr_relative, _countof(ini_fileName_tstr_relative), ini_fileName_tstr, NULL);
            _tcscpy_s(ini_fileName_tstr, ini_fileName_tstr_relative);
        }
        strcpy_s(ini_fileName, tchar_to_string(ini_fileName_tstr, CP_THREAD_ACP).c_str());
        init = check_inifile() && !disable_loading;
        GetPrivateProfileTStg(ini_section_main, "blog_url", _T(""), blog_url, _countof(blog_url), ini_fileName, codepage_ini);
        if (init) {
            load_encode_stg();
            load_fn_replace();
            load_log_win();
            load_append();
            load_last_out_stg();
        }
    }
}

guiEx_settings::~guiEx_settings() {
    clear_all();
}

void guiEx_settings::clear_all() {
    clear_aud();
    clear_mux();
    clear_enc();
    clear_local();
    clear_fn_replace();
    clear_log_win();
    clear_append();
    clear_fbc();
}

BOOL guiEx_settings::check_inifile() {
    ini_ver = GetPrivateProfileIntA(ini_section_main, "ini_ver", 0, ini_fileName);
    BOOL ret = (ini_ver >= INI_VER_MIN);
    uint64_t filesize = 0;
    if (ret && !rgy_get_filesize(ini_fileName, &filesize))
        ret = FALSE;
    ini_filesize = (DWORD)filesize;
    codepage_ini = ini_ver >= INI_VER_UTF8 ? CP_UTF8 : CP_THREAD_ACP;
    
    if (!PathFileExistsA(conf_fileName)) {
        codepage_cnf = CP_UTF8;
        WritePrivateProfileInt(ini_section_main, "cnf_ver", CNF_VER_UTF8, conf_fileName);
    } else {
        const int cnf_ver = GetPrivateProfileIntA(ini_section_main, "cnf_ver", 0, conf_fileName);
        codepage_cnf = cnf_ver >= CNF_VER_UTF8 ? CP_UTF8 : CP_THREAD_ACP;
        if (cnf_ver < CNF_VER_UTF8) {
            codepage_cnf = CP_THREAD_ACP;
            load_encode_stg();
            load_log_win();
            load_last_out_stg();
            codepage_cnf = CP_UTF8;
            save_local();
            save_log_win();
            save_last_out_stg();
            WritePrivateProfileW(ini_section_main, "theme", _T(""), conf_fileName, codepage_cnf);
            WritePrivateProfileInt(ini_section_main, "cnf_ver", CNF_VER_UTF8, conf_fileName);
        }
        codepage_cnf = CP_UTF8;
    }
    return ret;
}

BOOL guiEx_settings::get_init_success() {
    return get_init_success(FALSE);
}

BOOL guiEx_settings::get_init_success(BOOL no_message) {
    if (!init && !no_message) {
        TCHAR mes[1024];
        TCHAR title[256];
        _tcscpy_s(mes, AUO_NAME_W);
        _stprintf_s(PathFindExtension(mes), _countof(mes) - wcslen(mes),
            L".iniが存在しないか、iniファイルが古いです。\n%s を開始できません。\n"
            L"iniファイルを更新してみてください。", 
            AUO_FULL_NAME_W);
        _stprintf_s(title, _countof(title), L"%s - エラー", AUO_FULL_NAME_W);
        MessageBox(NULL, mes, title, MB_ICONERROR);
    }
    return init;
}

void guiEx_settings::get_default_lang() {
    WCHAR userSysLangW[LOCALE_NAME_MAX_LENGTH] = { 0 };
    GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SISO639LANGNAME, userSysLangW, _countof(userSysLangW));
    const auto userSysLang = wstring_to_tstring(userSysLangW);
    const TCHAR *defaultLanguage = AUO_LANGUAGE_DEFAULT;
    for (const auto& auo_lang : list_auo_languages) {
        if (_tcsicmp(userSysLang.c_str(), auo_lang.code) == 0) {
            defaultLanguage = auo_lang.code;
            break;
        }
    }
    //日本語のみ重ねてチェック
    WCHAR userSysCountryW[LOCALE_NAME_MAX_LENGTH] = { 0 };
    GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SISO3166CTRYNAME, userSysCountryW, _countof(userSysCountryW));
    const auto userSysCountry = wstring_to_tstring(userSysCountryW);
    if (_tcsicmp(userSysCountry.c_str(), _T("JP")) == 0) {
        defaultLanguage = AUO_LANGUAGE_JA;
    }
    _tcscpy_s(default_lang, defaultLanguage);
}

const TCHAR *guiEx_settings::get_lang() const {
    return language;
}

void guiEx_settings::set_and_save_lang(const TCHAR *lang) {
    if (_tcscmp(language, lang) != 0) {
        _tcscpy_s(language, lang);
        save_lang();

        //強制リロード
        TCHAR tmp_auo_path[_countof(auo_path)];
        char tmp_section_main[_countof(ini_section_main)];
        _tcscpy_s(tmp_auo_path, auo_path);
        strcpy_s(tmp_section_main, ini_section_main);
        clear_all();

        init = FALSE;
        initialize(FALSE, tmp_auo_path, tmp_section_main);
    }
}

const TCHAR *guiEx_settings::get_last_out_stg() const {
    return last_out_stg;
}

void guiEx_settings::set_last_out_stg(const TCHAR *stg) {
    _tcscpy_s(last_out_stg, _countof(last_out_stg), stg);
}

BOOL guiEx_settings::is_faw(const AUDIO_SETTINGS *aud_stg) const {
    if (_tcsstr(aud_stg->codec, _T("faw"))) {
        return TRUE;
    }
    if (!aud_stg->is_internal) {
        return _tcsstr(aud_stg->dispname, _T("FAW")) ? TRUE : FALSE;
    }
    return FALSE;
}

int guiEx_settings::get_faw_index(BOOL internal) const {
    if (internal) {
        for (int i = 0; i < s_aud_int_count; i++)
            if (is_faw(&s_aud_int[i]))
                return i;
    } else {
        for (int i = 0; i < s_aud_ext_count; i++)
            if (is_faw(&s_aud_ext[i]))
                return i;
    }
    return FAW_INDEX_ERROR;
}

void guiEx_settings::load_encode_stg() {
    load_aud();
    load_mux();
    load_enc();
    load_local(); //fullpathの情報がきちんと格納されるよう、最後に呼ぶ
}

void guiEx_settings::load_lang() {
    GetPrivateProfileTStg(ini_section_main, "language", default_lang, language, _countof(language), conf_fileName, codepage_cnf);
}

void guiEx_settings::load_last_out_stg() {
    GetPrivateProfileTStg(ini_section_main, "last_out_stg", _T(""), last_out_stg, _countof(last_out_stg), conf_fileName, codepage_cnf);
}

void guiEx_settings::load_aud() {
    clear_aud();

    s_aud_ext_count = GetPrivateProfileIntA(INI_SECTION_AUD,          "count", 0, ini_fileName);
    s_aud_int_count = GetPrivateProfileIntA(INI_SECTION_AUD_INTERNAL, "count", 0, ini_fileName);
    s_aud_mc.init(ini_filesize + (s_aud_ext_count + s_aud_int_count) * (sizeof(AUDIO_SETTINGS) + 1024));
    load_aud(TRUE);
    load_aud(FALSE);
}

void guiEx_settings::load_aud(BOOL internal) {
    int i, j, k;
    char encoder_section[INI_KEY_MAX_LEN];
    char key[INI_KEY_MAX_LEN];

    const auto ini_section = (internal) ? INI_SECTION_AUD_INTERNAL : INI_SECTION_AUD;
    const int s_aud_count = (internal) ? s_aud_int_count : s_aud_ext_count;
    AUDIO_SETTINGS *s_aud = (AUDIO_SETTINGS *)s_aud_mc.CutMem(s_aud_count * sizeof(AUDIO_SETTINGS));
    for (i = 0; i < s_aud_count; i++) {
        s_aud[i].is_internal = internal;
        sprintf_s(key, _countof(key), "audio_encoder_%d", i+1);
        s_aud[i].keyName = s_aud_mc.SetPrivateProfileString(ini_section, key, "key", ini_fileName, codepage_ini);
        sprintf_s(encoder_section, _countof(encoder_section), "%s%s", INI_SECTION_PREFIX, s_aud[i].keyName);
        s_aud[i].dispname     = s_aud_mc.SetPrivateProfileT(encoder_section, "dispname",     _T(""), ini_fileName, codepage_ini);
        s_aud[i].codec        = s_aud_mc.SetPrivateProfileT(encoder_section, "codec",        _T(""), ini_fileName, codepage_ini);
        s_aud[i].filename     = s_aud_mc.SetPrivateProfileT(encoder_section, "filename",     _T(""), ini_fileName, codepage_ini);
        s_aud[i].aud_appendix = s_aud_mc.SetPrivateProfileT(encoder_section, "aud_appendix", _T(""), ini_fileName, codepage_ini);
        s_aud[i].raw_appendix = s_aud_mc.SetPrivateProfileT(encoder_section, "raw_appendix", _T(""), ini_fileName, codepage_ini);
        s_aud[i].cmd_base     = s_aud_mc.SetPrivateProfileT(encoder_section, "base_cmd",     _T(""), ini_fileName, codepage_ini);
        s_aud[i].cmd_2pass    = s_aud_mc.SetPrivateProfileT(encoder_section, "2pass_cmd",    _T(""), ini_fileName, codepage_ini);
        s_aud[i].cmd_help     = s_aud_mc.SetPrivateProfileT(encoder_section, "help_cmd",     _T(""), ini_fileName, codepage_ini);
        s_aud[i].cmd_ver      = s_aud_mc.SetPrivateProfileT(encoder_section, "ver_cmd",      _T(""), ini_fileName, codepage_ini);
        s_aud[i].cmd_raw      = s_aud_mc.SetPrivateProfileT(encoder_section, "raw_cmd",      _T(""), ini_fileName, codepage_ini);
        s_aud[i].pipe_input   = GetPrivateProfileIntA(            encoder_section, "pipe_input",    0, ini_fileName);
        s_aud[i].disable_log  = GetPrivateProfileIntA(            encoder_section, "disable_log",   0, ini_fileName);
        s_aud[i].unsupported_mp4  = GetPrivateProfileIntA(    encoder_section, "unsupported_mp4",   0, ini_fileName);
        s_aud[i].enable_rf64      = GetPrivateProfileIntA(    encoder_section, "enable_rf64",       0, ini_fileName);
        s_aud[i].pcm_fp32         = GetPrivateProfileIntA(    encoder_section, "pcm_fp32",          0, ini_fileName);

        sprintf_s(encoder_section, _countof(encoder_section), "%s%s", INI_SECTION_MODE, s_aud[i].keyName);
        int tmp_count = GetPrivateProfileIntA(encoder_section, "count", 0, ini_fileName);
        //置き換えリストの影響で、この段階ではAUDIO_ENC_MODEが最終的に幾つになるのかわからない
        //とりあえず、一時的に読み込んでみる
        s_aud[i].mode_count = tmp_count;
        AUDIO_ENC_MODE *tmp_mode = (AUDIO_ENC_MODE *)s_aud_mc.CutMem(tmp_count * sizeof(AUDIO_ENC_MODE));
        for (j = 0; j < tmp_count; j++) {
            sprintf_s(key, _countof(key), "mode_%d", j+1);
            tmp_mode[j].name = s_aud_mc.SetPrivateProfileT(encoder_section, key, _T(""), ini_fileName, codepage_ini);
            const size_t keybase_len = strlen(key);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_cmd");
            tmp_mode[j].cmd = s_aud_mc.SetPrivateProfileT(encoder_section, key, _T(""), ini_fileName, codepage_ini);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_2pass");
            tmp_mode[j].enc_2pass = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_convert8bit");
            tmp_mode[j].use_8bit = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_use_remuxer");
            tmp_mode[j].use_remuxer = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_delay");
            tmp_mode[j].delay = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_bitrate");
            tmp_mode[j].bitrate = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            if (tmp_mode[j].bitrate) {
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_bitrate_min");
                tmp_mode[j].bitrate_min = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_bitrate_max");
                tmp_mode[j].bitrate_max = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_bitrate_step");
                tmp_mode[j].bitrate_step = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_bitrate_default");
                tmp_mode[j].bitrate_default = GetPrivateProfileIntA(encoder_section, key, 0, ini_fileName);
            } else {
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_dispList");
                tmp_mode[j].disp_list = s_aud_mc.SetPrivateProfileT(encoder_section, key, _T(""), ini_fileName, codepage_ini);
                s_aud_mc.CutMem(sizeof(key[0]));
                strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_cmdList");
                tmp_mode[j].cmd_list = s_aud_mc.SetPrivateProfileT(encoder_section, key, _T(""), ini_fileName, codepage_ini);
                s_aud_mc.CutMem(sizeof(key[0]));
                //リストのcmd置き換えリストの","の数分AUDIO_ENC_MODEは増える
                if (!tmp_mode[j].bitrate)
                    s_aud[i].mode_count += countchr(tmp_mode[j].cmd_list, ',');
            }
        }
        s_aud[i].mode = (AUDIO_ENC_MODE *)s_aud_mc.CutMem(s_aud[i].mode_count * sizeof(AUDIO_ENC_MODE));
        j = 0;
        for (int tmp_index = 0; tmp_index < tmp_count; tmp_index++) {
            if (tmp_mode[tmp_index].bitrate) {
                memcpy(&s_aud[i].mode[j], &tmp_mode[tmp_index], sizeof(AUDIO_ENC_MODE));
                j++;
            } else {
                //置き換えリストを分解する
                TCHAR *p, *q;
                int list_count = countchr(tmp_mode[tmp_index].cmd_list, ',') + 1;
                //分解した先頭へのポインタへのポインタ用領域を確保
                TCHAR **cmd_list  = (TCHAR**)s_aud_mc.CutMem(sizeof(TCHAR*) * list_count);
                TCHAR **disp_list = (TCHAR**)s_aud_mc.CutMem(sizeof(TCHAR*) * list_count);
                //cmdの置き換えリストを","により分解
                cmd_list[0] = tmp_mode[tmp_index].cmd_list;
                for (k = 0, p = cmd_list[0];  (cmd_list[k] = wcstok_s(p, L",", &q))  != NULL; k++)
                    p = NULL;
                //同様に表示用リストを分解
                disp_list[0] = tmp_mode[tmp_index].disp_list;
                TCHAR *wp, *wq;
                for (k = 0, wp = disp_list[0]; (disp_list[k] = wcstok_s(wp, L",", &wq)) != NULL; k++)
                    wp = NULL;
                //リストの個数分、置き換えを行ったAUDIO_ENC_MODEを作成する
                for (k = 0; k < list_count; j++, k++) {
                    memcpy(&s_aud[i].mode[j], &tmp_mode[tmp_index], sizeof(AUDIO_ENC_MODE));

                    if (cmd_list[k]) {
                        _tcscpy_s((TCHAR *)s_aud_mc.GetPtr(), s_aud_mc.GetRemain() / sizeof(s_aud[i].mode[j].cmd[0]), s_aud[i].mode[j].cmd);
                        replace((TCHAR *)s_aud_mc.GetPtr(), s_aud_mc.GetRemain() / sizeof(s_aud[i].mode[j].cmd[0]), L"%{cmdList}", cmd_list[k]);
                        s_aud[i].mode[j].cmd = (TCHAR *)s_aud_mc.GetPtr();
                        s_aud_mc.CutString(sizeof(s_aud[i].mode[j].cmd[0]));
                    }

                    if (disp_list[k]) {
                        _tcscpy_s((TCHAR *)s_aud_mc.GetPtr(), s_aud_mc.GetRemain() / sizeof(s_aud[i].mode[j].name[0]), s_aud[i].mode[j].name);
                        replace((TCHAR *)s_aud_mc.GetPtr(), s_aud_mc.GetRemain() / sizeof(s_aud[i].mode[j].name[0]), L"%{dispList}", disp_list[k]);
                        s_aud[i].mode[j].name = (TCHAR *)s_aud_mc.GetPtr();
                        s_aud_mc.CutString(sizeof(s_aud[i].mode[j].name[0]));
                    }
                }
            }
        }
    }
    if (s_aud_count > 0) {
        if (internal) {
            s_aud_int = s_aud;
        } else {
            s_aud_ext = s_aud;
        }
    }
}

void guiEx_settings::load_mux() {
    int i, j;
    size_t len, keybase_len;
    char muxer_section[INI_KEY_MAX_LEN];
    char key[INI_KEY_MAX_LEN];

    static const char * MUXER_TYPE[MUXER_MAX_COUNT]    = { "MUXER_MP4", "MUXER_MKV", "MUXER_TC2MP4", "MUXER_MP4_RAW", "MUXER_INTERNAL" };
    static const TCHAR * MUXER_OUT_EXT[MUXER_MAX_COUNT] = {      L".mp4",      L".mkv",         L".mp4",          L".mp4",             L".*" };

    clear_mux();


    s_mux_count = MUXER_MAX_COUNT;
    s_mux_mc.init(ini_filesize + s_mux_count * sizeof(MUXER_SETTINGS));
    s_mux = (MUXER_SETTINGS *)s_mux_mc.CutMem(s_mux_count * sizeof(MUXER_SETTINGS));
    for (i = 0; i < s_mux_count; i++) {
        sprintf_s(muxer_section, _countof(muxer_section), "%s%s", INI_SECTION_PREFIX, MUXER_TYPE[i]);
        len = strlen(MUXER_TYPE[i]);
        s_mux[i].keyName  = (char *)s_mux_mc.CutMem((len + 1) * sizeof(s_mux[i].keyName[0]));
        memcpy(s_mux[i].keyName, MUXER_TYPE[i], (len + 1) * sizeof(s_mux[i].keyName[0]));
        s_mux[i].dispname  = s_mux_mc.SetPrivateProfileWString(muxer_section, "dispname",  "", ini_fileName, codepage_ini);
        s_mux[i].filename  = s_mux_mc.SetPrivateProfileWString(muxer_section, "filename",  "", ini_fileName, codepage_ini);
        s_mux[i].base_cmd  = s_mux_mc.SetPrivateProfileWString(muxer_section, "base_cmd",  "", ini_fileName, codepage_ini);
        s_mux[i].out_ext   = (TCHAR *)s_mux_mc.GetPtr();
        _tcscpy_s(s_mux[i].out_ext, s_mux_mc.GetRemain() / sizeof(s_mux[i].out_ext[0]), MUXER_OUT_EXT[i]);
        s_mux_mc.CutString(sizeof(s_mux[i].out_ext[0]));
        s_mux[i].vid_cmd   = s_mux_mc.SetPrivateProfileWString(muxer_section, "vd_cmd",    "", ini_fileName, codepage_ini);
        s_mux[i].aud_cmd   = s_mux_mc.SetPrivateProfileWString(muxer_section, "au_cmd",    "", ini_fileName, codepage_ini);
        s_mux[i].tc_cmd    = s_mux_mc.SetPrivateProfileWString(muxer_section, "tc_cmd",    "", ini_fileName, codepage_ini);
        s_mux[i].delay_cmd = s_mux_mc.SetPrivateProfileWString(muxer_section, "delay_cmd", "", ini_fileName, codepage_ini);
        s_mux[i].tmp_cmd   = s_mux_mc.SetPrivateProfileWString(muxer_section, "tmp_cmd",   "", ini_fileName, codepage_ini);
        s_mux[i].help_cmd  = s_mux_mc.SetPrivateProfileWString(muxer_section, "help_cmd",  "", ini_fileName, codepage_ini);
        s_mux[i].ver_cmd   = s_mux_mc.SetPrivateProfileWString(muxer_section, "ver_cmd",   "", ini_fileName, codepage_ini);
        s_mux[i].post_mux  = GetPrivateProfileIntA(muxer_section, "post_mux", MUXER_DISABLED,  ini_fileName);

        sprintf_s(muxer_section, _countof(muxer_section), "%s%s", INI_SECTION_MODE, s_mux[i].keyName);
        s_mux[i].ex_count = GetPrivateProfileIntA(muxer_section, "count", 0, ini_fileName);
        s_mux[i].ex_cmd = (MUXER_CMD_EX *)s_mux_mc.CutMem(s_mux[i].ex_count * sizeof(MUXER_CMD_EX));
        for (j = 0; j < s_mux[i].ex_count; j++) {
            sprintf_s(key, _countof(key), "ex_cmd_%d", j+1);
            s_mux[i].ex_cmd[j].cmd  = s_mux_mc.SetPrivateProfileWString(muxer_section, key, "", ini_fileName, codepage_ini);
            keybase_len = strlen(key);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_name");
            s_mux[i].ex_cmd[j].name = s_mux_mc.SetPrivateProfileWString(muxer_section, key, "", ini_fileName, codepage_ini);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_apple");
            s_mux[i].ex_cmd[j].cmd_apple = s_mux_mc.SetPrivateProfileWString(muxer_section, key, "", ini_fileName, codepage_ini);
            strcpy_s(key + keybase_len, _countof(key) - keybase_len, "_chap");
            s_mux[i].ex_cmd[j].chap_file = s_mux_mc.SetPrivateProfileWString(muxer_section, key, "", ini_fileName, codepage_ini);
        }
    }
}

void guiEx_settings::load_fn_replace() {
    clear_fn_replace();

    fn_rep_mc.init(ini_filesize);

    TCHAR *ptr = (TCHAR *)fn_rep_mc.GetPtr();
    size_t len = GetPrivateProfileSectionStgW(INI_SECTION_FN, ptr, (DWORD)fn_rep_mc.GetRemain() / sizeof(ptr[0]), ini_fileName, codepage_ini);
    fn_rep_mc.CutMem((len + 1) * sizeof(ptr[0]));
    for (; *ptr != NULL; ptr += wcslen(ptr) + 1) {
        FILENAME_REPLACE rep = { 0 };
        TCHAR *p = wcschr(ptr, L'=');
        rep.from = (p) ? p + 1 : ptr - 1;
        p = wcschr(ptr, L':');
        if (p) *p = '\0';
        rep.to   = (p) ? p + 1 : ptr - 1;
        fn_rep.push_back(rep);
    }
}

void guiEx_settings::load_enc_cmd(ENC_CMD *x264cmd, int *count, int *default_index, const char *section) {
    char key[INI_KEY_MAX_LEN];
    TCHAR *desc = s_enc_mc.SetPrivateProfileWString(section, "name", "", ini_fileName, codepage_ini);
    s_enc_mc.CutMem(sizeof(desc[0]));
    *count = countchr(desc, ',') + 1;
    x264cmd->name = (ENC_OPTION_STR *)s_enc_mc.CutMem(sizeof(ENC_OPTION_STR) * (*count + 1));
    ZeroMemory(x264cmd->name, sizeof(ENC_OPTION_STR) * (*count + 1));
    x264cmd->cmd = (TCHAR **)s_enc_mc.CutMem(sizeof(TCHAR *) * (*count + 1));

    x264cmd->name[0].desc = desc;
    TCHAR *p = desc, *q;
    for (int i = 0; (x264cmd->name[i].desc = wcstok_s(p, L",", &q)) != NULL; i++)
        p = NULL;

    for (int i = 0; x264cmd->name[i].desc; i++) {
        TCHAR *name = (TCHAR *)s_enc_mc.GetPtr();
        x264cmd->name[i].name = name;
        _tcscpy_s(name, s_enc_mc.GetRemain() / sizeof(x264cmd->name[i].name[0]), x264cmd->name[i].desc);
        s_enc_mc.CutMem((wcslen(x264cmd->name[i].desc) + 1) * sizeof(x264cmd->name[i].name[0]));
    }

    *default_index = 0;
    TCHAR *def = s_enc_mc.SetPrivateProfileWString(section, "disp", "", ini_fileName, codepage_ini);
    sprintf_s(key,  _countof(key), "cmd_");
    size_t keybase_len = strlen(key);
    for (int i = 0; x264cmd->name[i].desc; i++) {
        auto str_utf8 = wstring_to_string(x264cmd->name[i].desc, CP_UTF8);
        strcpy_s(key + keybase_len, _countof(key) - keybase_len, str_utf8.c_str());
        x264cmd->cmd[i] = s_enc_mc.SetPrivateProfileWString(section, key, "", ini_fileName, codepage_ini);
        if (_wcsicmp(x264cmd->name[i].desc, def) == NULL)
            *default_index = i;
    }
}

void guiEx_settings::load_enc() {
    char key[INI_KEY_MAX_LEN];

    clear_enc();

    s_enc_mc.init(ini_filesize);

    s_enc.filename            = s_enc_mc.SetPrivateProfileWString(INI_SECTION_ENC_DEFAULT, "filename",      "x264", ini_fileName, codepage_ini);
    s_enc.default_cmd         = s_enc_mc.SetPrivateProfileWString(INI_SECTION_ENC_DEFAULT, "cmd_default",       "", ini_fileName, codepage_ini);
    s_enc.default_cmd_highbit = s_enc_mc.SetPrivateProfileWString(INI_SECTION_ENC_DEFAULT, "cmd_default_10bit", "", ini_fileName, codepage_ini);
    s_enc.help_cmd            = s_enc_mc.SetPrivateProfileWString(INI_SECTION_ENC_DEFAULT, "cmd_help",          "", ini_fileName, codepage_ini);

    load_enc_cmd(&s_enc.preset,  &s_enc.preset_count,  &s_enc.default_preset,  INI_SECTION_ENC_PRESET);
    load_enc_cmd(&s_enc.tune,    &s_enc.tune_count,    &s_enc.default_tune,    INI_SECTION_ENC_TUNE);
    load_enc_cmd(&s_enc.profile, &s_enc.profile_count, &s_enc.default_profile, INI_SECTION_ENC_PROFILE);

    s_enc.profile_vbv_multi = (float *)s_enc_mc.CutMem(sizeof(float) * s_enc.profile_count);
    for (int i = 0; i < s_enc.profile_count; i++) {
        sprintf_s(key, _countof(key), "vbv_multi_%s", tchar_to_string(s_enc.profile.name[i].name).c_str());
        s_enc.profile_vbv_multi[i] = (float)GetPrivateProfileDouble(INI_SECTION_ENC_PROFILE, key, 1.0, ini_fileName);
    }

    s_enc_refresh = TRUE;
}

void guiEx_settings::make_default_stg_dir(TCHAR *default_stg_dir, DWORD nSize) {
    //絶対パスで作成
    //_tcscpy_s(default_stg_dir, nSize, char_to_wstring(auo_path).c_str());
    //相対パスで作成
    TCHAR temp_dir[MAX_PATH_LEN];
    GetRelativePathTo(temp_dir, _countof(temp_dir), auo_path, NULL);
    _tcscpy_s(default_stg_dir, nSize, temp_dir);

    TCHAR *filename_ptr = PathFindExtension(default_stg_dir);
    _tcscpy_s(filename_ptr, nSize - (filename_ptr - default_stg_dir), char_to_tstring(STG_DEFAULT_DIRECTORY_APPENDIX).c_str());
}

void guiEx_settings::load_local() {
    TCHAR default_stg_dir[MAX_PATH_LEN];
    make_default_stg_dir(default_stg_dir, _countof(default_stg_dir));

    clear_local();

    s_local.large_cmdbox              = GetPrivateProfileIntA(   ini_section_main, "large_cmdbox",              DEFAULT_LARGE_CMD_BOX,         conf_fileName);
    s_local.auto_afs_disable          = GetPrivateProfileIntA(   ini_section_main, "auto_afs_disable",          DEFAULT_AUTO_AFS_DISABLE,      conf_fileName);
    s_local.default_output_ext        = GetPrivateProfileIntA(   ini_section_main, "default_output_ext",        DEFAULT_OUTPUT_EXT,            conf_fileName);
    s_local.auto_del_stats            = GetPrivateProfileIntA(   ini_section_main, "auto_del_stats",            DEFAULT_AUTO_DEL_STATS,        conf_fileName);
    s_local.auto_del_chap             = GetPrivateProfileIntA(   ini_section_main, "auto_del_chap",             DEFAULT_AUTO_DEL_CHAP,         conf_fileName);
    s_local.keep_qp_file              = GetPrivateProfileIntA(   ini_section_main, "keep_qp_file",              DEFAULT_KEEP_QP_FILE,          conf_fileName);
    s_local.disable_tooltip_help      = GetPrivateProfileIntA(   ini_section_main, "disable_tooltip_help",      DEFAULT_DISABLE_TOOLTIP_HELP,  conf_fileName);
    s_local.disable_visual_styles     = GetPrivateProfileIntA(   ini_section_main, "disable_visual_styles",     DEFAULT_DISABLE_VISUAL_STYLES, conf_fileName);
    s_local.enable_stg_esc_key        = GetPrivateProfileIntA(   ini_section_main, "enable_stg_esc_key",        DEFAULT_ENABLE_STG_ESC_KEY,    conf_fileName);
    s_local.chap_nero_convert_to_utf8 = GetPrivateProfileIntA(   ini_section_main, "chap_nero_convert_to_utf8", DEFAULT_CHAP_NERO_TO_UTF8,     conf_fileName);
    s_local.get_relative_path         = GetPrivateProfileIntA(   ini_section_main, "get_relative_path",         DEFAULT_SAVE_RELATIVE_PATH,    conf_fileName);
    s_local.run_bat_minimized         = GetPrivateProfileIntA(   ini_section_main, "run_bat_minimized",         DEFAULT_RUN_BAT_MINIMIZED,     conf_fileName);
    s_local.set_keyframe_as_afs_24fps = GetPrivateProfileIntA(   ini_section_main, "set_keyframe_as_afs_24fps", DEFAULT_SET_KEYFRAME_AFS24FPS, conf_fileName);
    s_local.auto_ref_limit_by_level   = GetPrivateProfileIntA(   ini_section_main, "auto_ref_limit_by_level",   DEFAULT_AUTO_REFLIMIT_BYLEVEL, conf_fileName);
    s_local.default_audio_encoder_ext = GetPrivateProfileIntA(   ini_section_main, "default_audio_encoder",     DEFAULT_AUDIO_ENCODER_EXT,     conf_fileName);
    s_local.default_audio_encoder_in  = GetPrivateProfileIntA(   ini_section_main, "default_audio_encoder_in",  DEFAULT_AUDIO_ENCODER_IN,      conf_fileName);
    s_local.default_audenc_use_in     = GetPrivateProfileIntA(   ini_section_main, "default_audenc_use_in",     DEFAULT_AUDIO_ENCODER_USE_IN,  conf_fileName);
    s_local.av_length_threshold       = GetPrivateProfileDouble(ini_section_main, "av_length_threshold",       DEFAULT_AV_LENGTH_DIFF_THRESOLD, conf_fileName);
    s_local.thread_pthrottling_mode   = GetPrivateProfileIntA(   ini_section_main, "thread_pthrottling_mode",   DEFAULT_THREAD_PTHROTTLING,    conf_fileName);
#if ENCODER_QSV
    s_local.force_bluray              = GetPrivateProfileIntA(   ini_section_main, "force_bluray",              DEFAULT_FORCE_BLURAY,          conf_fileName);
    s_local.perf_monitor              = GetPrivateProfileIntA(   ini_section_main, "perf_monitor",              DEFAULT_PERF_MONITOR,          conf_fileName);
#endif

    s_local.amp_retry_limit           = GetPrivateProfileIntA(   INI_SECTION_AMP,  "amp_retry_limit",           DEFAULT_AMP_RETRY_LIMIT,       conf_fileName);
    s_local.amp_bitrate_margin_multi  = GetPrivateProfileDouble(INI_SECTION_AMP,  "amp_bitrate_margin_multi",  DEFAULT_AMP_MARGIN,            conf_fileName);
    s_local.amp_reenc_audio_multi     = GetPrivateProfileDouble(INI_SECTION_AMP,  "amp_reenc_audio_multi",     DEFAULT_AMP_REENC_AUDIO_MULTI, conf_fileName);
    s_local.amp_keep_old_file         = GetPrivateProfileIntA(   INI_SECTION_AMP,  "amp_keep_old_file",         DEFAULT_AMP_KEEP_OLD_FILE,     conf_fileName);
    s_local.amp_bitrate_margin_multi  = clamp(s_local.amp_bitrate_margin_multi, 0.0, 1.0);

    GetFontInfo(ini_section_main, "conf_font", &s_local.conf_font, conf_fileName);

    GetPrivateProfileTStg(ini_section_main, "custom_tmp_dir",        _T(""), s_local.custom_tmp_dir,        _countof(s_local.custom_tmp_dir),        conf_fileName, codepage_cnf);
    GetPrivateProfileTStg(ini_section_main, "custom_audio_tmp_dir",  _T(""), s_local.custom_audio_tmp_dir,  _countof(s_local.custom_audio_tmp_dir),  conf_fileName, codepage_cnf);
    GetPrivateProfileTStg(ini_section_main, "custom_mp4box_tmp_dir", _T(""), s_local.custom_mp4box_tmp_dir, _countof(s_local.custom_mp4box_tmp_dir), conf_fileName, codepage_cnf);
    GetPrivateProfileTStg(ini_section_main, "stg_dir",  default_stg_dir, s_local.stg_dir,               _countof(s_local.stg_dir),               conf_fileName, codepage_cnf);
    GetPrivateProfileTStg(ini_section_main, "last_app_dir",          _T(""), s_local.app_dir,               _countof(s_local.app_dir),               conf_fileName, codepage_cnf);
    GetPrivateProfileTStg(ini_section_main, "last_bat_dir",          _T(""), s_local.bat_dir,               _countof(s_local.bat_dir),               conf_fileName, codepage_cnf);

    //設定ファイル保存場所をチェックする
    if (!str_has_char(s_local.stg_dir) || !PathRootExists(s_local.stg_dir))
        _tcscpy_s(s_local.stg_dir, _countof(s_local.stg_dir), default_stg_dir);

    s_local.large_cmdbox = 0;
    s_local.audio_buffer_size   = std::min((decltype(s_local.audio_buffer_size))GetPrivateProfileIntA(ini_section_main, "audio_buffer",        AUDIO_BUFFER_DEFAULT, conf_fileName), AUDIO_BUFFER_MAX);

    GetPrivateProfileTStg(INI_SECTION_ENC,     INI_SECTION_ENC,  _T(""), s_enc.fullpath,         _countof(s_enc.fullpath),         conf_fileName, codepage_cnf);
    for (int i = 0; i < s_aud_ext_count; i++)
    GetPrivateProfileTStg(INI_SECTION_AUD, s_aud_ext[i].keyName, _T(""), s_aud_ext[i].fullpath, _countof(s_aud_ext[i].fullpath), conf_fileName, codepage_cnf);
    for (int i = 0; i < s_mux_count; i++)
    GetPrivateProfileTStg(INI_SECTION_MUX, s_mux[i].keyName, _T(""), s_mux[i].fullpath,       _countof(s_mux[i].fullpath),       conf_fileName, codepage_cnf);
}

void guiEx_settings::load_log_win() {
    clear_log_win();
    s_log.minimized          = GetPrivateProfileIntA(   ini_section_main, "log_start_minimized",  DEFAULT_LOG_START_MINIMIZED,  conf_fileName);
    s_log.log_level          = GetPrivateProfileIntA(   ini_section_main, "log_level",            DEFAULT_LOG_LEVEL,            conf_fileName);
    s_log.transparent        = GetPrivateProfileIntA(   ini_section_main, "log_transparent",      DEFAULT_LOG_TRANSPARENT,      conf_fileName);
    s_log.transparency       = GetPrivateProfileIntA(   ini_section_main, "log_transparency",     DEFAULT_LOG_TRANSPARENCY,     conf_fileName);
    s_log.auto_save_log      = GetPrivateProfileIntA(   ini_section_main, "log_auto_save",        DEFAULT_LOG_AUTO_SAVE,        conf_fileName);
    s_log.auto_save_log_mode = GetPrivateProfileIntA(   ini_section_main, "log_auto_save_mode",   DEFAULT_LOG_AUTO_SAVE_MODE,   conf_fileName);
    GetPrivateProfileTStg(ini_section_main, "log_auto_save_path", _T(""), s_log.auto_save_log_path, _countof(s_log.auto_save_log_path), conf_fileName, codepage_cnf);
    s_log.show_status_bar    = GetPrivateProfileIntA(   ini_section_main, "log_show_status_bar",  DEFAULT_LOG_SHOW_STATUS_BAR,  conf_fileName);
    s_log.taskbar_progress   = GetPrivateProfileIntA(   ini_section_main, "log_taskbar_progress", DEFAULT_LOG_TASKBAR_PROGRESS, conf_fileName);
    s_log.save_log_size      = GetPrivateProfileIntA(   ini_section_main, "save_log_size",        DEFAULT_LOG_SAVE_SIZE,        conf_fileName);
    s_log.log_width          = GetPrivateProfileIntA(   ini_section_main, "log_width",            DEFAULT_LOG_WIDTH,            conf_fileName);
    s_log.log_height         = GetPrivateProfileIntA(   ini_section_main, "log_height",           DEFAULT_LOG_HEIGHT,           conf_fileName);
    s_log.log_pos[0]         = GetPrivateProfileIntA(   ini_section_main, "log_pos_x",            DEFAULT_LOG_POS[0],           conf_fileName);
    s_log.log_pos[1]         = GetPrivateProfileIntA(   ini_section_main, "log_pos_y",            DEFAULT_LOG_POS[1],           conf_fileName);
    GetColorInfo(ini_section_main, "log_color_background",   s_log.log_color_background, DEFAULT_LOG_COLOR_BACKGROUND, conf_fileName);
    GetColorInfo(ini_section_main, "log_color_text_info",    s_log.log_color_text[0],    DEFAULT_LOG_COLOR_TEXT[0],    conf_fileName);
    GetColorInfo(ini_section_main, "log_color_text_warning", s_log.log_color_text[1],    DEFAULT_LOG_COLOR_TEXT[1],    conf_fileName);
    GetColorInfo(ini_section_main, "log_color_text_error",   s_log.log_color_text[2],    DEFAULT_LOG_COLOR_TEXT[2],    conf_fileName);
    GetFontInfo(ini_section_main,  "log_font", &s_log.log_font, conf_fileName);
}

void guiEx_settings::load_append() {
    clear_append();
    GetPrivateProfileTStg(INI_SECTION_APPENDIX, "tc_appendix",         _T("_tc.txt"),      s_append.tc,         _countof(s_append.tc),         ini_fileName, codepage_ini);
    GetPrivateProfileTStg(INI_SECTION_APPENDIX, "qp_appendix",         _T("_qp.txt"),      s_append.qp,         _countof(s_append.qp),         ini_fileName, codepage_ini);
    GetPrivateProfileTStg(INI_SECTION_APPENDIX, "chap_appendix",       _T("_chapter.txt"), s_append.chap,       _countof(s_append.chap),       ini_fileName, codepage_ini);
    GetPrivateProfileTStg(INI_SECTION_APPENDIX, "chap_apple_appendix", _T("_chapter.txt"), s_append.chap_apple, _countof(s_append.chap_apple), ini_fileName, codepage_ini);
    GetPrivateProfileTStg(INI_SECTION_APPENDIX, "wav_appendix",        _T("_tmp.wav"),     s_append.wav,        _countof(s_append.wav),        ini_fileName, codepage_ini);
}

void guiEx_settings::load_fbc() {
    clear_fbc();
    s_fbc.calc_bitrate         = GetPrivateProfileIntA(   INI_SECTION_FBC, "calc_bitrate",         DEFAULT_FBC_CALC_BITRATE,         conf_fileName);
    s_fbc.calc_time_from_frame = GetPrivateProfileIntA(   INI_SECTION_FBC, "calc_time_from_frame", DEFAULT_FBC_CALC_TIME_FROM_FRAME, conf_fileName);
    s_fbc.last_frame_num       = GetPrivateProfileIntA(   INI_SECTION_FBC, "last_frame_num",       DEFAULT_FBC_LAST_FRAME_NUM,       conf_fileName);
    s_fbc.last_fps             = GetPrivateProfileDouble(INI_SECTION_FBC, "last_fps",             DEFAULT_FBC_LAST_FPS,             conf_fileName);
    s_fbc.last_time_in_sec     = GetPrivateProfileIntA(   INI_SECTION_FBC, "last_time_in_sec",     DEFAULT_FBC_LAST_TIME_IN_SEC,     conf_fileName);
    s_fbc.initial_size         = GetPrivateProfileDouble(INI_SECTION_FBC, "initial_size",         DEFAULT_FBC_INITIAL_SIZE,         conf_fileName);
}

void guiEx_settings::save_local() {
    WritePrivateProfileIntWithDefault(   ini_section_main, "large_cmdbox",              s_local.large_cmdbox,              DEFAULT_LARGE_CMD_BOX,         conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "auto_afs_disable",          s_local.auto_afs_disable,          DEFAULT_AUTO_AFS_DISABLE,      conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "default_output_ext",        s_local.default_output_ext,        DEFAULT_OUTPUT_EXT,            conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "auto_del_stats",            s_local.auto_del_stats,            DEFAULT_AUTO_DEL_STATS,        conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "auto_del_chap",             s_local.auto_del_chap,             DEFAULT_AUTO_DEL_CHAP,         conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "keep_qp_file",              s_local.keep_qp_file,              DEFAULT_KEEP_QP_FILE,          conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "disable_tooltip_help",      s_local.disable_tooltip_help,      DEFAULT_DISABLE_TOOLTIP_HELP,  conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "disable_visual_styles",     s_local.disable_visual_styles,     DEFAULT_DISABLE_VISUAL_STYLES, conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "enable_stg_esc_key",        s_local.enable_stg_esc_key,        DEFAULT_ENABLE_STG_ESC_KEY,    conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "chap_nero_convert_to_utf8", s_local.chap_nero_convert_to_utf8, DEFAULT_CHAP_NERO_TO_UTF8,     conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "get_relative_path",         s_local.get_relative_path,         DEFAULT_SAVE_RELATIVE_PATH,    conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "run_bat_minimized",         s_local.run_bat_minimized,         DEFAULT_RUN_BAT_MINIMIZED,     conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "set_keyframe_as_afs_24fps", s_local.set_keyframe_as_afs_24fps, DEFAULT_SET_KEYFRAME_AFS24FPS, conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "auto_ref_limit_by_level",   s_local.auto_ref_limit_by_level,   DEFAULT_AUTO_REFLIMIT_BYLEVEL, conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "default_audio_encoder",     s_local.default_audio_encoder_ext, DEFAULT_AUDIO_ENCODER_EXT,     conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "default_audio_encoder_in",  s_local.default_audio_encoder_in,  DEFAULT_AUDIO_ENCODER_IN,      conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "default_audenc_use_in",     s_local.default_audenc_use_in,     DEFAULT_AUDIO_ENCODER_USE_IN,  conf_fileName);
    WritePrivateProfileDoubleWithDefault(ini_section_main, "av_length_threshold",       s_local.av_length_threshold,       DEFAULT_AV_LENGTH_DIFF_THRESOLD,conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "thread_pthrottling_mode",   s_local.thread_pthrottling_mode, DEFAULT_THREAD_PTHROTTLING,      conf_fileName);
#if ENCODER_QSV
    WritePrivateProfileIntWithDefault(   ini_section_main, "force_bluray",              s_local.force_bluray,              DEFAULT_FORCE_BLURAY,          conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "perf_monitor",              s_local.perf_monitor,              DEFAULT_PERF_MONITOR,          conf_fileName);
#endif

    WritePrivateProfileIntWithDefault(   INI_SECTION_AMP,  "amp_retry_limit",           s_local.amp_retry_limit,           DEFAULT_AMP_RETRY_LIMIT,       conf_fileName);
    WritePrivateProfileDoubleWithDefault(INI_SECTION_AMP,  "amp_bitrate_margin_multi",  s_local.amp_bitrate_margin_multi,  DEFAULT_AMP_MARGIN,            conf_fileName);
    WritePrivateProfileDoubleWithDefault(INI_SECTION_AMP,  "amp_reenc_audio_multi",     s_local.amp_reenc_audio_multi,     DEFAULT_AMP_REENC_AUDIO_MULTI, conf_fileName);
    WritePrivateProfileIntWithDefault(   INI_SECTION_AMP,  "amp_keep_old_file",         s_local.amp_keep_old_file,         DEFAULT_AMP_KEEP_OLD_FILE,     conf_fileName);

    WriteFontInfo(ini_section_main, "conf_font", &s_local.conf_font, conf_fileName);

    PathRemoveBlanksW(s_local.custom_tmp_dir);
    PathRemoveBackslashW(s_local.custom_tmp_dir);
    WritePrivateProfileW(ini_section_main, "custom_tmp_dir",        s_local.custom_tmp_dir,        conf_fileName, codepage_cnf);

    PathRemoveBlanksW(s_local.custom_audio_tmp_dir);
    PathRemoveBackslashW(s_local.custom_audio_tmp_dir);
    WritePrivateProfileW(ini_section_main, "custom_audio_tmp_dir",  s_local.custom_audio_tmp_dir,  conf_fileName, codepage_cnf);

    PathRemoveBlanksW(s_local.custom_mp4box_tmp_dir);
    PathRemoveBackslashW(s_local.custom_mp4box_tmp_dir);
    WritePrivateProfileW(ini_section_main, "custom_mp4box_tmp_dir", s_local.custom_mp4box_tmp_dir, conf_fileName, codepage_cnf);

    PathRemoveBlanksW(s_local.stg_dir);
    PathRemoveBackslashW(s_local.stg_dir);
    WritePrivateProfileW(ini_section_main, "stg_dir",               s_local.stg_dir,               conf_fileName, codepage_cnf);

    PathRemoveBlanksW(s_local.app_dir);
    PathRemoveBackslashW(s_local.app_dir);
    WritePrivateProfileW(ini_section_main, "last_app_dir",          s_local.app_dir,               conf_fileName, codepage_cnf);

    PathRemoveBlanksW(s_local.bat_dir);
    PathRemoveBackslashW(s_local.bat_dir);
    WritePrivateProfileW(ini_section_main, "last_bat_dir",          s_local.bat_dir,               conf_fileName, codepage_cnf);

    for (int i = 0; i < s_aud_ext_count; i++) {
        PathRemoveBlanksW(s_aud_ext[i].fullpath);
        WritePrivateProfileW(INI_SECTION_AUD, s_aud_ext[i].keyName, s_aud_ext[i].fullpath, conf_fileName, codepage_cnf);
    }
    for (int i = 0; i < s_mux_count; i++) {
        PathRemoveBlanksW(s_mux[i].fullpath);
        WritePrivateProfileW(INI_SECTION_MUX, s_mux[i].keyName, s_mux[i].fullpath, conf_fileName, codepage_cnf);
    }

    PathRemoveBlanksW(s_enc.fullpath);
    WritePrivateProfileW(INI_SECTION_ENC,   INI_SECTION_ENC,        s_enc.fullpath,                conf_fileName, codepage_cnf);
}

void guiEx_settings::save_log_win() {
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_start_minimized",   s_log.minimized,          DEFAULT_LOG_START_MINIMIZED,  conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_level",             s_log.log_level,          DEFAULT_LOG_LEVEL,            conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_transparent",       s_log.transparent,        DEFAULT_LOG_TRANSPARENT,      conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_transparency",      s_log.transparency,       DEFAULT_LOG_TRANSPARENCY,     conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_auto_save",         s_log.auto_save_log,      DEFAULT_LOG_AUTO_SAVE,        conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_auto_save_mode",    s_log.auto_save_log_mode, DEFAULT_LOG_AUTO_SAVE_MODE,   conf_fileName);
    WritePrivateProfileW(ini_section_main, "log_auto_save_path", s_log.auto_save_log_path, conf_fileName, codepage_cnf);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_show_status_bar",   s_log.show_status_bar,    DEFAULT_LOG_SHOW_STATUS_BAR,  conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_taskbar_progress",  s_log.taskbar_progress,   DEFAULT_LOG_TASKBAR_PROGRESS, conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "save_log_size",         s_log.save_log_size,      DEFAULT_LOG_SAVE_SIZE,        conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_width",             s_log.log_width,          DEFAULT_LOG_WIDTH,            conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_height",            s_log.log_height,         DEFAULT_LOG_HEIGHT,           conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_pos_x",             s_log.log_pos[0],         DEFAULT_LOG_POS[0],           conf_fileName);
    WritePrivateProfileIntWithDefault(   ini_section_main, "log_pos_y",             s_log.log_pos[1],         DEFAULT_LOG_POS[1],           conf_fileName);
    WriteColorInfo(ini_section_main, "log_color_background",   s_log.log_color_background, DEFAULT_LOG_COLOR_BACKGROUND, conf_fileName);
    WriteColorInfo(ini_section_main, "log_color_text_info",    s_log.log_color_text[0],    DEFAULT_LOG_COLOR_TEXT[0],    conf_fileName);
    WriteColorInfo(ini_section_main, "log_color_text_warning", s_log.log_color_text[1],    DEFAULT_LOG_COLOR_TEXT[1],    conf_fileName);
    WriteColorInfo(ini_section_main, "log_color_text_error",   s_log.log_color_text[2],    DEFAULT_LOG_COLOR_TEXT[2],    conf_fileName);
    WriteFontInfo(ini_section_main,  "log_font", &s_log.log_font, conf_fileName);
}

void guiEx_settings::save_fbc() {
    WritePrivateProfileIntWithDefault(   INI_SECTION_FBC, "calc_bitrate",         s_fbc.calc_bitrate,         DEFAULT_FBC_CALC_BITRATE,         conf_fileName);
    WritePrivateProfileIntWithDefault(   INI_SECTION_FBC, "calc_time_from_frame", s_fbc.calc_time_from_frame, DEFAULT_FBC_CALC_TIME_FROM_FRAME, conf_fileName);
    WritePrivateProfileIntWithDefault(   INI_SECTION_FBC, "last_frame_num",       s_fbc.last_frame_num,       DEFAULT_FBC_LAST_FRAME_NUM,       conf_fileName);
    WritePrivateProfileDoubleWithDefault(INI_SECTION_FBC, "last_fps",             s_fbc.last_fps,             DEFAULT_FBC_LAST_FPS,             conf_fileName);
    WritePrivateProfileDoubleWithDefault(INI_SECTION_FBC, "last_time_in_sec",     s_fbc.last_time_in_sec,     DEFAULT_FBC_LAST_TIME_IN_SEC,     conf_fileName);
    WritePrivateProfileDoubleWithDefault(INI_SECTION_FBC, "initial_size",         s_fbc.initial_size,         DEFAULT_FBC_INITIAL_SIZE,         conf_fileName);
}

void guiEx_settings::save_lang() {
    WritePrivateProfileW(ini_section_main, "language", language, conf_fileName, codepage_cnf);
}

void guiEx_settings::save_last_out_stg() {
    WritePrivateProfileW(ini_section_main, "last_out_stg", last_out_stg, conf_fileName, codepage_cnf);
}

BOOL guiEx_settings::get_reset_s_enc_referesh() {
    BOOL refresh = s_enc_refresh;
    s_enc_refresh = FALSE;
    return refresh;
}

void guiEx_settings::clear_aud() {
    s_aud_mc.clear();
    s_aud_ext_count = 0;
    s_aud_int_count = 0;
}

void guiEx_settings::clear_mux() {
    s_mux_mc.clear();
    s_mux_count = 0;
}

void guiEx_settings::clear_enc() {
    s_enc_mc.clear();
    s_enc_refresh = TRUE;
}

void guiEx_settings::clear_local() {
    ZeroMemory(&s_local, sizeof(s_local));
}

void guiEx_settings::clear_fn_replace() {
    fn_rep_mc.clear();
    fn_rep.clear();
}

void guiEx_settings::clear_log_win() {
    ZeroMemory(&s_log, sizeof(s_log));
}

void guiEx_settings::clear_fbc() {
    ZeroMemory(&s_fbc, sizeof(s_fbc));
}

void guiEx_settings::clear_append() {
    ZeroMemory(&s_append, sizeof(s_append));
}

void guiEx_settings::apply_fn_replace(TCHAR *target_filename, DWORD nSize) {
    for (auto i_rep : fn_rep) {
        replace(target_filename, nSize, i_rep.from, i_rep.to);
    }
}

ColorRGB DarkenWindowStgNamedColor::parseColor(const std::string& colorStr) const {
    if (colorStr.length() == 0) return ColorRGB();
    {
        int r = 0, g = 0, b = 0;
        if (sscanf_s(colorStr.c_str(), "%d, %d, %d", &r, &g, &b) == 3) {
            return ColorRGB(r, g, b);
        }
    }
    const bool withSharp = (colorStr[0] == '#');
    const int colorLength = (int)colorStr.length() - (withSharp ? 1 : 0);
    int value = 0;
    if (colorLength == 3 && sscanf_s(colorStr.c_str(), "#%x", &value) == 1) {
        const auto R = colorStr.substr(withSharp ? 1 : 0, 1);
        const auto G = colorStr.substr(withSharp ? 2 : 1, 1);
        const auto B = colorStr.substr(withSharp ? 3 : 2, 1);
        return ColorRGB(
            std::stoi(R + R, nullptr, 16),
            std::stoi(G + G, nullptr, 16),
            std::stoi(B + B, nullptr, 16));
    } else if (colorLength == 6 && sscanf_s(colorStr.c_str(), "#%x", &value) == 1) {
        const auto R = colorStr.substr(withSharp ? 1 : 0, 2);
        const auto G = colorStr.substr(withSharp ? 3 : 2, 2);
        const auto B = colorStr.substr(withSharp ? 5 : 4, 2);
        return ColorRGB(
            std::stoi(R, nullptr, 16),
            std::stoi(G, nullptr, 16),
            std::stoi(B, nullptr, 16));
    }
    return ColorRGB();
}

const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColor(const char *name) const {
    for (const auto& c : namedColors) {
        if (_stricmp(c.name().c_str(), name) == 0) {
            return &c;
        }
    }
    return nullptr;
}

const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorFromNameAndState(const char *name, const DarkenWindowState state) const {
    if (namedColors.size() == 0) return nullptr;
    const int stateId = clamp((int)state, 0, (int)DarkenWindowState::MaxCout - 1);
    char key[256];
    sprintf_s(key, "%s_%s", name, DWSTATE_NAMES[stateId]);
    const auto color = getColor(key);
    if (color) return color;
    return nullptr;
}

const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorStatic(const DarkenWindowState state) const {
    static const char *STATIC_COLOR_NAMES[] = { "c5", "c2" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(STATIC_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(STATIC_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}
const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorButton(const DarkenWindowState state) const {
    static const char *BUTTON_COLOR_NAMES[] = { "c2" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(BUTTON_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(BUTTON_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}
const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorCheckBox(const DarkenWindowState state) const {
    static const char *BUTTON_COLOR_NAMES[] = { "c5", "c2" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(BUTTON_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(BUTTON_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}
const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorTextBox(const DarkenWindowState state) const {
    static const char *TEXT_BOX_COLOR_NAMES[] = { "c1" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(TEXT_BOX_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(TEXT_BOX_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}

const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorMenu(const DarkenWindowState state) const {
    static const char *STATIC_COLOR_NAMES[] = { "c4" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(STATIC_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(STATIC_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}
const DarkenWindowStgNamedColor *DarkenWindowStgReader::getColorToolTip(const DarkenWindowState state) const {
    static const char *BUTTON_COLOR_NAMES[] = { "c2" };
    if (namedColors.size() == 0) return nullptr;
    for (int i = 0; i < _countof(BUTTON_COLOR_NAMES); i++) {
        const auto color = getColorFromNameAndState(BUTTON_COLOR_NAMES[i], state);
        if (color) return color;
    }
    return nullptr;
}

bool DarkenWindowStgReader::isDarkTheme() const {
    const auto color = getColorStatic();
    if (color) {
        //輝度に変換して、暗い系の色かをチェックする
        const double y = color->fillColor().r * 0.299 + color->fillColor().g * 0.587 + color->fillColor().b * 0.114;
        return y < 128.0;
    }
    return false;
}

DarkenWindowStgReader::DarkenWindowStgReader(const std::wstring& dir) : rootDir(dir), selectedThemeXml(), selectedThemeXmlFileSets(), namedColors() {};
DarkenWindowStgReader::~DarkenWindowStgReader() {};

std::string DarkenWindowStgReader::readWcharXml(const std::wstring& path) {
    std::ifstream ifs(path, std::ios_base::in | std::ios_base::binary);
    ifs.seekg(0, std::ios::end);
    const auto size = (size_t)ifs.tellg();
    ifs.seekg(0);

    std::vector<char> buffer(size + sizeof(TCHAR), '\0');
    ifs.read(buffer.data(), size);
    char *ptr = buffer.data();
    if ((ptr[0] == 0xFE && ptr[1] == 0xFF) || ptr[0] == 0xFF && ptr[1] == 0xFE) {
        ptr += 2;
    }
    return wstring_to_string((TCHAR*)buffer.data(), CP_UTF8);
}

int DarkenWindowStgReader::parseRootStg() {
    const auto rootStgPath = std::filesystem::path(rootDir) / L"DarkenWindowSettings.xml";
    if (!std::filesystem::exists(rootStgPath)) {
        return 1;
    }
    const auto xml_utf8 = DarkenWindowStgReader::readWcharXml(rootStgPath.wstring());
    if (xml_utf8.length() == 0) {
        return 1;
    }

    tinyxml2::XMLDocument xml;
    if (xml.Parse(xml_utf8.c_str(), xml_utf8.length()) != tinyxml2::XML_NO_ERROR) {
        return 1;
    }
    static const char *ELEM_NAME_SETTINGS = "Settings";
    static const char *ATTR_NAME_SKIN = "skin";
    auto root = xml.FirstChildElement(ELEM_NAME_SETTINGS);
    if (root != nullptr) {
        if (auto attr = root->Attribute(ATTR_NAME_SKIN); attr != nullptr) {
            selectedThemeXml = char_to_wstring(attr, CP_UTF8);
        }
    }
    return 0;
}

std::wstring DarkenWindowStgReader::getSelectedTheme() const {
    return std::filesystem::path(selectedThemeXml).stem().wstring();
}

int DarkenWindowStgReader::parseSelectedStg() {
    static const char *ELEM_NAME_SETTINGS = "Settings";
    static const char *ELEM_NAME_SKIN = "Skin";
    selectedThemeXmlFileSets.clear();
    const auto selectedStg = std::filesystem::path(rootDir) / selectedThemeXml;
    if (!std::filesystem::exists(selectedStg)) {
        return 1;
    }
    const auto xml_utf8 = DarkenWindowStgReader::readWcharXml(selectedStg.wstring());
    if (xml_utf8.length() == 0) {
        return 1;
    }
    tinyxml2::XMLDocument xml;
    if (xml.Parse(xml_utf8.c_str(), xml_utf8.length()) != tinyxml2::XML_NO_ERROR) {
        return 1;
    }
    auto elem = xml.FirstChildElement(ELEM_NAME_SETTINGS);
    if (elem == nullptr) {
        return 1;
    }
    for (auto skinXml = elem->FirstChildElement(ELEM_NAME_SKIN);
        skinXml != nullptr; skinXml = skinXml->NextSiblingElement(ELEM_NAME_SKIN)) {
        auto filename = skinXml->Attribute("fileName");
        if (filename != nullptr) {
            selectedThemeXmlFileSets.push_back(char_to_wstring(filename, CP_UTF8));
        }
    }
    return 0;
}

int DarkenWindowStgReader::parseSelectedStg2() {
    static const char *ELEM_NAME_SKIN = "Skin";
    static const char *ELEM_NAME_ATTRIBUTES = "Attributes";
    static const char *ELEM_NAME_NAMED_COLORS = "NamedColors";
    static const char *ELEM_NAME_NAMED_COLOR = "NamedColor";
    namedColors.clear();

    const auto selectedStg = (std::filesystem::path(rootDir) / selectedThemeXml).remove_filename() / selectedThemeXmlFileSets.front();
    if (!std::filesystem::exists(selectedStg)) {
        return 1;
    }
    const auto xml_utf8 = DarkenWindowStgReader::readWcharXml(selectedStg.wstring());
    if (xml_utf8.length() == 0) {
        return 1;
    }
    tinyxml2::XMLDocument xml;
    if (xml.Parse(xml_utf8.c_str(), xml_utf8.length()) != tinyxml2::XML_NO_ERROR) {
        return 1;
    }
    auto elem = xml.FirstChildElement(ELEM_NAME_SKIN);
    if (elem == nullptr) {
        return 1;
    }
    elem = elem->FirstChildElement(ELEM_NAME_ATTRIBUTES);
    if (elem == nullptr) {
        return 1;
    }
    auto namedColorsElem = elem->FirstChildElement(ELEM_NAME_NAMED_COLORS);
    if (namedColorsElem == nullptr) {
        return 1;
    }
    for (auto namedColorElem = namedColorsElem->FirstChildElement(ELEM_NAME_NAMED_COLOR);
        namedColorElem != nullptr; namedColorElem = namedColorElem->NextSiblingElement(ELEM_NAME_NAMED_COLOR)) {
        DarkenWindowStgNamedColor color(
            namedColorElem->Attribute("name"),
            namedColorElem->Attribute("fillColor"),
            namedColorElem->Attribute("edgeColor"),
            namedColorElem->Attribute("textForeColor"),
            namedColorElem->Attribute("textBackColor"));
        if (false) {
            fprintf(stderr, "color=%s, fillColor=%s, edgeColor=%s, textForeColor=%s, textBackColor=%s\n",
                color.name().c_str(),
                color.fillColor().printHex().c_str(),
                color.edgeColor().printHex().c_str(),
                color.textForeColor().printHex().c_str(),
                color.textBackColor().printHex().c_str());
        }
        if (color.name().length() > 0) {
            namedColors.push_back(color);
        }
    }
    return 0;
}

int DarkenWindowStgReader::parseStg() {
    if (parseRootStg()) {
        return 1;
    }
    if (parseSelectedStg()) {
        return 1;
    }
    if (parseSelectedStg2()) {
        return 1;
    }
    return 0;
}

DarkenWindowStgReader *createDarkenWindowStgReader(const TCHAR *aviutl_dir) {
    TCHAR pluginDir[MAX_PATH_LEN];
    PathCombineLong(pluginDir, _countof(pluginDir), aviutl_dir, _T("plugins"));
    TCHAR DWStgDir[MAX_PATH_LEN];
    PathCombineLong(DWStgDir, _countof(DWStgDir), pluginDir, _T("DarkenWindow"));
    return new DarkenWindowStgReader(DWStgDir);
}

std::tuple<AuoTheme, DarkenWindowStgReader *> check_current_theme(const TCHAR *aviutl_dir) {
    const bool DWLoaded = checkIfModuleLoaded(L"DarkenWindow.aul");
    DarkenWindowStgReader *dwStg = nullptr;
    if (DWLoaded) {
        dwStg = createDarkenWindowStgReader(aviutl_dir);
    }
    //DarkenWindowが使用されていれば設定をロードする
    return {
        (DWLoaded && dwStg->parseStg() == 0) ? ((dwStg->isDarkTheme()) ? AuoTheme::DarkenWindowDark : AuoTheme::DarkenWindowLight) : AuoTheme::DefaultLight,
        dwStg
    };
}
