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
#pragma once
#ifndef _EXE_VERSION_H_
#define _EXE_VERSION_H_

#include <string>

int version_a_larger_than_b(const int a[4], const int b[4]);
std::string ver_string(int ver[4]);

int get_exe_version_info(const char *exe_path, int version[4]);
int get_exe_version_from_cmd(const char *exe_path, const char *cmd_ver, int version[4]);

int get_x264_rev(const char *x264fullpath);
int get_x265_rev(const char *x265fullpath, int version[4]);
int get_svtav1_rev(const char *svtav1fullpath, int version[4]);

int get_x265ver_from_txt(const char *txt, int v[4]);

enum QTDLL {
    QAAC_APPLEDLL_UNAVAILABLE = 0,
    QAAC_APPLEDLL_IN_EXEDIR = 1,
    QAAC_APPLEDLL_IN_CURRENTDIR = 2
};

QTDLL check_if_apple_dll_required_for_qaac(const char *exe_dir, const char *current_fullpath);

#endif //_EXE_VERSION_H_
