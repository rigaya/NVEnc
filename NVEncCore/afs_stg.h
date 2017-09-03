#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

static const char *const AFS_STG_UP               = "up";
static const char *const AFS_STG_BOTTOM           = "bottom";
static const char *const AFS_STG_LEFT             = "left";
static const char *const AFS_STG_RIGHT            = "right";
static const char *const AFS_STG_METHOD_WATERSHED = "method_watershed";
static const char *const AFS_STG_COEFF_SHIFT      = "coeff_shift";
static const char *const AFS_STG_THRE_SHIFT       = "thre_shift";
static const char *const AFS_STG_THRE_DEINT       = "thre_deint";
static const char *const AFS_STG_THRE_Y_MOTION    = "thre_Ymotion";
static const char *const AFS_STG_THRE_C_MOTION    = "thre_Cmotion";
static const char *const AFS_STG_MODE             = "mode";
static const char *const AFS_STG_THREADS          = "threads";
static const char *const AFS_STG_SUB_THREADS      = "sub_threads";

static const char *const AFS_STG_FIELD_SHIFT      = "field_shift";
static const char *const AFS_STG_DROP             = "drop";
static const char *const AFS_STG_SMOOTH           = "smooth";
static const char *const AFS_STG_FORCE24          = "force24";
static const char *const AFS_STG_DETECT_SC        = "detect_sc";
static const char *const AFS_STG_TUNE_MODE        = "tune_mode";
static const char *const AFS_STG_LOG_SAVE         = "log_save";
static const char *const AFS_STG_TRACE_MODE       = "trace_mode";
static const char *const AFS_STG_REPLAY_MODE      = "replay_mode";
static const char *const AFS_STG_YUY2UPSAMPLE     = "yuy2upsample";
static const char *const AFS_STG_THROUGH_MODE     = "through_mode";
static const char *const AFS_STG_PROC_MODE        = "proc_mode";
static const char *const AFS_STG_RFF              = "rff";
static const char *const AFS_STG_LOG              = "log";

#define AFS_STG_SECTION   "AFS_STG"
#define AFSVF_STG_SECTION "AFSVF_STG"
#define AFS_STG_FILTER  "設定ファイル (*.ini)\0*.ini\0" "全てのファイル (*.*)\0*.*\0"

#if 0
static inline void WritePrivateProfileInt(const char *section, const char *keyname, int value, const char *ini_file) {
    char tmp[22];
    sprintf_s(tmp, _countof(tmp), "%d", value);
    WritePrivateProfileStringA(section, keyname, tmp, ini_file);
}
static void write_stg_file(bool bForVF, const char *filename, int *track, int track_n, int *check, int check_n, int proc_mode) {
    auto section = (bForVF) ? AFSVF_STG_SECTION : AFS_STG_SECTION;
    WritePrivateProfileInt(section, AFS_STG_UP,               track[0],  filename);
    WritePrivateProfileInt(section, AFS_STG_BOTTOM,           track[1],  filename);
    WritePrivateProfileInt(section, AFS_STG_LEFT,             track[2],  filename);
    WritePrivateProfileInt(section, AFS_STG_RIGHT,            track[3],  filename);
    WritePrivateProfileInt(section, AFS_STG_METHOD_WATERSHED, track[4],  filename);
    WritePrivateProfileInt(section, AFS_STG_COEFF_SHIFT,      track[5],  filename);
    WritePrivateProfileInt(section, AFS_STG_THRE_SHIFT,       track[6],  filename);
    WritePrivateProfileInt(section, AFS_STG_THRE_DEINT,       track[7],  filename);
    WritePrivateProfileInt(section, AFS_STG_THRE_Y_MOTION,    track[8],  filename);
    WritePrivateProfileInt(section, AFS_STG_THRE_C_MOTION,    track[9],  filename);
    WritePrivateProfileInt(section, AFS_STG_MODE,             track[10], filename);
    WritePrivateProfileInt(section, AFS_STG_THREADS,          track[11], filename);
    if (track_n >= 13) {
        WritePrivateProfileInt(section, AFS_STG_SUB_THREADS,  track[12], filename);
    }
    WritePrivateProfileInt(section, AFS_STG_FIELD_SHIFT,      check[0],  filename);
    WritePrivateProfileInt(section, AFS_STG_DROP,             check[1],  filename);
    WritePrivateProfileInt(section, AFS_STG_SMOOTH,           check[2],  filename);
    WritePrivateProfileInt(section, AFS_STG_FORCE24,          check[3],  filename);
    WritePrivateProfileInt(section, AFS_STG_DETECT_SC,        check[4],  filename);
    WritePrivateProfileInt(section, AFS_STG_TUNE_MODE,        check[5],  filename);
    WritePrivateProfileInt(section, AFS_STG_LOG_SAVE,         check[6],  filename);
    WritePrivateProfileInt(section, AFS_STG_TRACE_MODE,       check[7],  filename);
    WritePrivateProfileInt(section, AFS_STG_REPLAY_MODE,      check[8],  filename);
    WritePrivateProfileInt(section, AFS_STG_YUY2UPSAMPLE,     check[9],  filename);
    WritePrivateProfileInt(section, AFS_STG_THROUGH_MODE,     check[10], filename);

    WritePrivateProfileInt(section, AFS_STG_PROC_MODE,        proc_mode, filename);
}
#endif
