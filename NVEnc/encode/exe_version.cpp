﻿// -----------------------------------------------------------------------------------------
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
#include <string>
#include "auo_pipe.h"
#include "rgy_util.h"
#include "rgy_filesystem.h"
#include "auo_version.h"
#include "exe_version.h"
#if ENCODER_X265
#include <regex>
#endif

static const int RC_VER_ADD = -1000000;
static const int RC_VER_MUL =    10000;

int version_a_larger_than_b(const int a[4], const int b[4]) {
    for (int i = 0; i < 4; i++) {
        if (a[i] > b[i]) return +1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

std::string ver_string(int ver[4]) {
    const int VER_LENGTH = 4;
    if (nullptr == ver)
        return "";

    bool allZero = (0 == ver[0]);
    bool isRev = !allZero;
    for (int i = 1; i < VER_LENGTH; i++) {
        allZero &= (0 == ver[i]);
        isRev   &= (0 == ver[i]);
    }
    if (allZero)
        return "";
    if (isRev)
        return strsprintf("r%d", ver[0]);

    auto str = strsprintf("v%d", ver[0]);
    int loop_fin = VER_LENGTH-1;
    for (int i = 1; i < loop_fin; i++) {
        if (ver[i]) {
            str += strsprintf(".%d", ver[i]);
        }
    }
    if (ver[loop_fin]) {
        str += strsprintf("+%d", ver[loop_fin]);
    }
    return str;
}

#if ENCODER_X264
int get_x264_version_from_filename(const TCHAR *exe_path, int version[4]) {
    const auto filename = PathGetFilename(exe_path);

    int rev = 0;
    if (_stscanf_s(filename.c_str(), _T("x264_%d_x64.exe"), &rev) == 1) {
        version[2] = rev;
        return 0;
    }
    if (_stscanf_s(filename.c_str(), _T("x264_%d_x86.exe"), &rev) == 1) {
        version[2] = rev;
        return 0;
    }
    return -1;
}
#endif

#if ENCODER_X265
int get_x265_version_from_filename(const TCHAR *exe_path, int version[4]) {
    const auto filename = PathGetFilename(exe_path);

    int value[4] = { 0 };
    memset(version, 0, sizeof(value));

    if (   _stscanf_s(filename.c_str(), _T("x265_%d.%d+%d_x64.exe"), &value[0], &value[1], &value[3]) == 3
        || _stscanf_s(filename.c_str(), _T("x265_%d.%d+%d_x86.exe"), &value[0], &value[1], &value[3]) == 3
        || _stscanf_s(filename.c_str(), _T("x265_%d.%d_x64.exe"),    &value[0], &value[1]) == 2
        || _stscanf_s(filename.c_str(), _T("x265_%d.%d_x86.exe"),    &value[0], &value[1]) == 2) {
        memcpy(version, value, sizeof(value));
        return 0;
    }
    return -1;
}

int get_x265ver_regex(const TCHAR *txt, int v[4]) {
    tstring str = txt;
    while (!str.empty() && (str.back() == _T('\r') || str.back() == _T('\n') || str.back() == _T(' '))) {
        str.pop_back();
    }
    // 正規表現
    std::basic_regex<TCHAR> re(_T("(\\d+).(\\d+)(_ [0-9A-Za-z]+)*(\\+(\\d+))?(([-][A-Za-z0-9]+)+([ ].*)*)*"));
    // 上記の "_ [0-9A-Za-z]+" は実際には "_[0-9A-Za-z]+" を意味する
    // スペースを除去
    tstring pattern = _T("(\\d+).(\\d+)(_[0-9A-Za-z]+)*(\\+(\\d+))?(([-][A-Za-z0-9]+)+([ ].*)*)*");
    std::basic_regex<TCHAR> re2(pattern);
    std::match_results<tstring::const_iterator> match;
    int ret = 1;
    memset(v, 0, sizeof(v[0]) * 4);
    if (std::regex_match(str.cbegin(), str.cend(), match, re2)) {
        try {
            v[0] = std::stoi(match[1].str());
            v[1] = std::stoi(match[2].str());
            ret = 0;
            if (match[5].length() > 0) {
                v[3] = std::stoi(match[5].str());
            }
        } catch (...) {
            // 次に進む
        }
    }
    return ret;
}
#endif

#if ENCODER_SVTAV1
int get_svtav1_version_from_filename(const TCHAR *exe_path, int version[4]) {
    const auto filename = PathGetFilename(exe_path);

    int value[4] = { 0 };
    memset(version, 0, sizeof(value));
    
    int value4 = 0;
    if (   _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d-%d_x64.exe"), &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d-%d_x86.exe"), &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d+%d_x64.exe"), &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d+%d_x86.exe"), &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d-%d_x64.exe"),  &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d-%d_x86.exe"),  &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d+%d_x64.exe"),  &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d+%d_x86.exe"),  &value[0], &value[1], &value[2], &value[3], &value4) == 5
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d_x64.exe"),    &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-rc%d_x86.exe"),    &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d_x64.exe"),     &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-rc%d_x86.exe"),     &value[0], &value[1], &value[2], &value[3]) == 4) {
        value[3] *= RC_VER_MUL;
        value[3] += RC_VER_ADD;
        value[3] += value4;
        memcpy(version, value, sizeof(value));
        return 0;
    }
    if (   _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-%d_x64.exe"), &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d-%d_x86.exe"), &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d+%d_x64.exe"), &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d+%d_x86.exe"), &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-%d_x64.exe"),  &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d-%d_x86.exe"),  &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d+%d_x64.exe"),  &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d+%d_x86.exe"),  &value[0], &value[1], &value[2], &value[3]) == 4
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d_x64.exe"),    &value[0], &value[1], &value[2]) == 3
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_v%d.%d.%d_x86.exe"),    &value[0], &value[1], &value[2]) == 3
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d_x64.exe"),     &value[0], &value[1], &value[2]) == 3
        || _stscanf_s(filename.c_str(), _T("SvtAv1EncApp_%d.%d.%d_x86.exe"),     &value[0], &value[1], &value[2]) == 3) {
        memcpy(version, value, sizeof(value));
        return 0;
    }
    return -1;
}
#endif

int get_exe_version_info(const TCHAR *exe_path, int version[4]) {
    #pragma comment(lib, "version.lib")
    int ret = -1;
    BYTE *data_ver_info = nullptr;
    DWORD ver_info_size, dummy = 0;
    if (         0 == (ver_info_size = GetFileVersionInfoSize(exe_path, &dummy))
        || nullptr == (data_ver_info = (BYTE *)malloc(ver_info_size)))
        return ret;

    if (GetFileVersionInfo(exe_path, 0, ver_info_size, (void*)data_ver_info)) {
        TCHAR *buf;
        UINT buf_len;
        typedef struct {
            WORD wLanguage, wCodePage;
        } LANGANDCODEPAGE;
        LANGANDCODEPAGE *ptr_translate = nullptr;
        UINT translate_len = 0;
        if (VerQueryValue(data_ver_info, L"\\VarFileInfo\\Translation", (void**)&ptr_translate, &translate_len)) {
            for (DWORD i = 0; i < (translate_len/sizeof(LANGANDCODEPAGE)); i++) {
                TCHAR sub_block[256];
                _stprintf_s(sub_block, _countof(sub_block), _T("\\StringFileInfo\\%04x%04x\\FileVersion"), ptr_translate[i].wLanguage, ptr_translate[i].wCodePage);
                buf = nullptr;
                buf_len = 0;
                int ver[4] = { 0 };
                if (VerQueryValue(data_ver_info, sub_block, (void **)&buf, &buf_len) && buf) {
                    if (   4 == _stscanf_s(buf, _T("%d.%d.%d.%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                        || 4 == _stscanf_s(buf, _T("%d.%d.%d+%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                        || 4 == _stscanf_s(buf, _T("%d.%d.%d-%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                        || 3 == _stscanf_s(buf, _T("%d.%d.%d"),    &ver[0], &ver[1], &ver[2]         )
                        || 3 == _stscanf_s(buf, _T("%d.%d+%d"),    &ver[0], &ver[1],          &ver[3])
                        || 2 == _stscanf_s(buf, _T("%d.%d"),       &ver[0], &ver[1]                  )
                        || 2 == _stscanf_s(buf, _T("%d+%d"),       &ver[0],                   &ver[3])
                        || 1 == _stscanf_s(buf, _T("%d"),                                     &ver[0]) ) {
                        memcpy(version, ver, sizeof(int) * 4);
                        ret = 0;
#if ENCODER_X265
                        if (ver[3] == 0 && get_x265ver_regex(buf, ver) == 0) {
                            memcpy(version, ver, sizeof(int) * 4);
                        }
#endif
                        break;
                    }
#if ENCODER_X265
                    if ((ret = get_x265ver_regex(buf, ver)) == 0) {
                        memcpy(version, ver, sizeof(int) * 4);
                        break;
                    }
#endif
                }
            }
            static const WORD wCodePageID[] = { 0, 932, 949, 950, 1200, 1250, 1251, 1252, 1253, 1254, 1255, 1256 };
            static const WORD wLanguageID[] = {
                0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407, 0x0408, 0x0409, 0x040A, 0x040B, 0x040C, 0x040D, 0x040E, 0x040F,
                0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
                0x0420, 0x0421, 0x0804, 0x0807, 0x0809, 0x080A, 0x080C, 0x0810, 0x0813, 0x0814, 0x0816, 0x081A, 0x0C0C, 0x100C
            };
            for (int i = 0; ret < 0 && i < _countof(wCodePageID); i++) {
                for (int j = 0; ret < 0 && j < _countof(wLanguageID); j++) {
                    TCHAR sub_block[256];
                    _stprintf_s(sub_block, _countof(sub_block), _T("\\StringFileInfo\\%04x%04x\\FileVersion"), wLanguageID[j], wCodePageID[i]);
                    buf = nullptr;
                    buf_len = 0;
                    int ver[4] = { 0 };
                    if (VerQueryValue(data_ver_info, sub_block, (void **)&buf, &buf_len) && buf) {
                        if (   4 == _stscanf_s(buf, _T("%d.%d.%d.%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                            || 4 == _stscanf_s(buf, _T("%d.%d.%d+%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                            || 4 == _stscanf_s(buf, _T("%d.%d.%d-%d"), &ver[0], &ver[1], &ver[2], &ver[3])
                            || 3 == _stscanf_s(buf, _T("%d.%d.%d"),    &ver[0], &ver[1], &ver[2]         )
                            || 3 == _stscanf_s(buf, _T("%d.%d+%d"),    &ver[0], &ver[1],          &ver[3])
                            || 2 == _stscanf_s(buf, _T("%d.%d"),       &ver[0], &ver[1]                  )
                            || 2 == _stscanf_s(buf, _T("%d+%d"),       &ver[0],                   &ver[3])
                            || 1 == _stscanf_s(buf, _T("%d"),                                     &ver[0]) ) {
                            memcpy(version, ver, sizeof(int) * 4);
                            ret = 0;
#if ENCODER_X265
                            if (ver[3] == 0 && get_x265ver_regex(buf, ver) == 0) {
                                memcpy(version, ver, sizeof(int) * 4);
                            }
#endif
                            break;
                        }
#if ENCODER_X265
                        if ((ret = get_x265ver_regex(buf, ver)) == 0) {
                            memcpy(version, ver, sizeof(int) * 4);
                            break;
                        }
#endif
                    }
                }
            }
        }
    }

    free(data_ver_info);
    return ret;
}

int get_exe_version_from_cmd(const TCHAR *exe_path, const TCHAR *cmd_ver, int version[4]) {
    int ret = -1;
    if (nullptr == version || nullptr == exe_path || !rgy_file_exists(exe_path))
        return ret;

    memset(version, 0, sizeof(int) * 4);
    const int BUFFER_LEN = 128 * 1024;
    char *buffer = (char *)malloc(BUFFER_LEN);
    if (nullptr == buffer)
        return ret;
    if (nullptr == cmd_ver)
        cmd_ver = _T("-h");
    if (get_exe_message(exe_path, cmd_ver, buffer, BUFFER_LEN / sizeof(buffer[0]), AUO_PIPE_MUXED) == RP_SUCCESS) {
        char *str = nullptr;
#if ENCODER_X264
        int core = 0;
        if (1 == sscanf_s(buffer, "x264 core:%d", &core)) {
            str = buffer + (strlen("x264 core:") + get_intlen(core));
        } else
#endif
#if ENCODER_SVTAV1
        if (strstr(buffer, "SVT-AV1 Encoder Lib v") != nullptr) {
            str = strstr(buffer, "SVT-AV1 Encoder Lib v") + strlen("SVT-AV1 Encoder Lib v");
        } else
#endif
        {
            str = buffer;
        }
        for (char *rtr = nullptr; 0 != ret && nullptr != (str = strtok_s(str, "\n", &rtr)); ) {
            char *ptr = str;
            static const char *PREFIX[] = { "fdkaac", "flac", "qaac", "refalac", "opus-tools", "version", "revision.", "revision", "rev.", "rev", " r.", " r", " v" };
            for (int i = 0; i < _countof(PREFIX); i++) {
                char *qtr = NULL;
                if (NULL != (qtr = stristr(ptr, PREFIX[i]))) {
                    ptr = qtr + strlen(PREFIX[i]);

                    char *const ptr_fin = ptr + strlen(ptr);
                    while (!isdigit(*ptr) && ptr < ptr_fin)
                        ptr++;

                    int ver[4] = { 0 };
                    int value4 = 0;
                    if (   5 == sscanf_s(ptr, "%d.%d.%d-rc%d-%d", &ver[0], &ver[1], &ver[2], &ver[3], &value4)
                        || 5 == sscanf_s(ptr, "%d.%d.%d-rc%d+%d", &ver[0], &ver[1], &ver[2], &ver[3], &value4)
                        || 4 == sscanf_s(ptr, "%d.%d.%d-rc%d",    &ver[0], &ver[1], &ver[2], &ver[3])) {
                        ver[3] *= RC_VER_MUL;
                        ver[3] += RC_VER_ADD;
                        ver[3] += value4;
                        memcpy(version, ver, sizeof(int) * 4);
                        ret = 0;
                        break;
                    }
                    if (   4 == sscanf_s(ptr, "%d.%d.%d.%d", &ver[0], &ver[1], &ver[2], &ver[3])
                        || 4 == sscanf_s(ptr, "%d.%d.%d-%d", &ver[0], &ver[1], &ver[2], &ver[3])
                        || 4 == sscanf_s(ptr, "%d.%d.%d+%d", &ver[0], &ver[1], &ver[2], &ver[3])
                        || 3 == sscanf_s(ptr, "%d.%d.%d",    &ver[0], &ver[1], &ver[2]         )
                        || 3 == sscanf_s(ptr, "%d.%d+%d",    &ver[0], &ver[1],          &ver[3])
                        || 2 == sscanf_s(ptr, "%d.%d",       &ver[0], &ver[1]                  )
                        || 2 == sscanf_s(ptr, "%d+%d",       &ver[0],                   &ver[3])
                        || 1 == sscanf_s(ptr, "%d",          &ver[0]                           )) {
                        memcpy(version, ver, sizeof(int) * 4);
                        ret = 0;
#if ENCODER_X265
                        const auto linetstr = char_to_tstring(ptr);
                        if (ver[3] == 0 && get_x265ver_regex(linetstr.c_str(), ver) == 0) {
                            memcpy(version, ver, sizeof(int) * 4);
                        }
#endif
                        break;
                    }
#if ENCODER_X265
                    const auto linetstr = char_to_tstring(ptr);
                    if ((ret = get_x265ver_regex(linetstr.c_str(), ver)) == 0) {
                        memcpy(version, ver, sizeof(int) * 4);
                        break;
                    }
#endif
                }
            }
            str = nullptr;
        }
    }
    free(buffer);
    return ret;
}

#if ENCODER_X264
int get_x264_rev(const TCHAR *x264fullpath) {
    int ret = -1;
    if (!rgy_file_exists(x264fullpath))
        return ret;

    int version[4] = { 0 };
    if (   ((ret = get_x264_version_from_filename(x264fullpath, version))            != -1 && version[2] != 0)
        || ((ret = get_exe_version_info(x264fullpath, version))                      != -1 && version[2] != 0)
        || ((ret = get_exe_version_from_cmd(x264fullpath, _T("--version"), version)) != -1 && version[2] != 0)) {
        return version[2];
    }
    return 0;
}
#endif

#if ENCODER_X265
int get_x265_rev(const TCHAR *x265fullpath, int version[4]) {
    int ret = -1;
    if (!rgy_file_exists(x265fullpath))
        return ret;

    int value[4] = { 0 };
    memset(value, 0, sizeof(value));

    if (((ret = get_x265_version_from_filename(x265fullpath, value))            != -1 && memcmp(version, value, sizeof(value)) != 0)
    ||  ((ret = get_exe_version_info(x265fullpath, value))                      != -1 && memcmp(version, value, sizeof(value)) != 0)
    ||  ((ret = get_exe_version_from_cmd(x265fullpath, _T("--version"), value)) != -1 && memcmp(version, value, sizeof(value)) != 0)) {
        memcpy(version, value, sizeof(value));
        return 0;
    }
    return -1;
}

int get_x265ver_from_txt(const TCHAR *txt, int v[4]) {
    int ret = 1;
    memset(v, 0, sizeof(v[0]) * 4);
    if (   4 != _stscanf_s(txt, _T("%d.%d.%d.%d"), &v[0], &v[1], &v[2], &v[3])
        && 4 != _stscanf_s(txt, _T("%d.%d.%d+%d"), &v[0], &v[1], &v[2], &v[3])
        && 3 != _stscanf_s(txt, _T("%d.%d.%d"),    &v[0], &v[1], &v[2]       )
        && 3 != _stscanf_s(txt, _T("%d.%d+%d"),    &v[0], &v[1],        &v[3])
        && 2 != _stscanf_s(txt, _T("%d.%d"),       &v[0], &v[1]              )
        && 2 != _stscanf_s(txt, _T("%d+%d"),       &v[0],               &v[3])
        && 1 != _stscanf_s(txt, _T("%d"),          &v[0]                     )) {
        if ((ret = get_x265ver_regex(txt, v)) != 0) {
            memset(v, 0, sizeof(v[0]) * 4);
        }
    } else {
        ret = 0;
        int test[4] = { 0 };
        if (v[3] == 0 && get_x265ver_regex(txt, test) == 0) {
            memcpy(v, test, sizeof(int) * 4);
        }
    }
    return ret;
}
#endif

#if ENCODER_SVTAV1
int get_svtav1_rev(const TCHAR *svtav1fullpath, int version[4]) {
    int ret = -1;
    if (!rgy_file_exists(svtav1fullpath))
        return ret;

    int value[4] = { 0 };
    if (   ((ret = get_svtav1_version_from_filename(svtav1fullpath, value))          != -1 && memcmp(version, value, sizeof(value)) != 0)
        || ((ret = get_exe_version_info(svtav1fullpath, value))                      != -1 && memcmp(version, value, sizeof(value)) != 0)
        || ((ret = get_exe_version_from_cmd(svtav1fullpath, _T("--version"), value)) != -1 && memcmp(version, value, sizeof(value)) != 0)) {
        memcpy(version, value, sizeof(value));
        return 0;
    }
    return -1;
}
#endif

static BOOL qaac_dll_available() {
    //Apple Application Supportのレジストリをチェック
    #pragma comment(lib, "Advapi32.lib")
    static const TCHAR *CHECK_KEY = _T("SOFTWARE\\Apple Inc.\\Apple Application Support");
    HKEY hKey = NULL;
    if (ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, CHECK_KEY, 0, KEY_QUERY_VALUE, &hKey)) {
        RegCloseKey(hKey);
        return TRUE;
    }
    return FALSE;
}

static BOOL qaac_dll_available(const TCHAR *dir) {
    if (nullptr == dir || !str_has_char(dir))
        return FALSE;

    static const TCHAR *QAAC_DLL[] = { _T("CoreAudioToolbox.dll"), _T("CoreFoundation.dll") };
    for (int i = 0; i < _countof(QAAC_DLL); i++) {
        if (!rgy_file_exists(PathCombineS(dir, QAAC_DLL[i])))
            return FALSE;
    }
    return TRUE;
}

QTDLL check_if_apple_dll_required_for_qaac(const TCHAR *exe_dir, const TCHAR *current_fullpath) {
    if (qaac_dll_available())
        return QAAC_APPLEDLL_IN_EXEDIR;
    if (qaac_dll_available(exe_dir))
        return QAAC_APPLEDLL_IN_EXEDIR;
    if (nullptr != current_fullpath && str_has_char(current_fullpath)) {
        auto [ ret , dir ] = PathRemoveFileSpecFixed(current_fullpath);
        if (qaac_dll_available(dir.c_str()))
            return QAAC_APPLEDLL_IN_CURRENTDIR;
    }
    return QAAC_APPLEDLL_UNAVAILABLE;
}
