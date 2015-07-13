#pragma once
#ifndef _EXE_VERSION_H_
#define _EXE_VERSION_H_

#include <string>

std::string ver_string(int ver[4]);

int get_exe_version_info(const char *exe_path, int version[4]);
int get_exe_version_from_cmd(const char *exe_path, const char *cmd_ver, int version[4]);

int get_x264_rev(const char *x264fullpath);

int get_x265ver_from_txt(const char *txt, int v[4]);

enum QTDLL {
    QAAC_APPLEDLL_UNAVAILABLE = 0,
    QAAC_APPLEDLL_IN_EXEDIR = 1,
    QAAC_APPLEDLL_IN_CURRENTDIR = 2
};

QTDLL check_if_apple_dll_required_for_qaac(const char *exe_dir, const char *current_fullpath);

#endif //_EXE_VERSION_H_
