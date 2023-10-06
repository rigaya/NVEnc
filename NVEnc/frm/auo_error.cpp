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

#include <algorithm>
#include "auo.h"
#include "auo_frm.h"
#include "auo_mes.h"
#include "auo_version.h"
#include "auo_pipe.h"
#include "auo_chapter.h"
#include "auo_settings.h"
#include "auo_util.h"

void warning_conf_not_initialized(const char *default_stg_file) {
    if (default_stg_file && strlen(default_stg_file) > 0) {
        write_log_auo_line_fmt(LOG_WARNING, L"%s: %s", g_auo_mes.get(AUO_ERR_CONF_NOT_INIT0), char_to_wstring(default_stg_file).c_str());
    } else {
        write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_CONF_NOT_INIT1));
    }
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_CONF_NOT_INIT2));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_CONF_NOT_INIT3));
}

void warning_failed_getting_temp_path() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_GET_TEMP_PATH));
}

void warning_unable_to_open_tempfile(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_UNABLE_OPEM_TEMP_FILE), char_to_wstring(dir).c_str());
}

void warning_no_temp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_NO_TEMP_ROOT), char_to_wstring(dir).c_str());
}

void warning_no_aud_temp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_NO_AUD_TEMP_ROOT), char_to_wstring(dir).c_str());
}

void error_filename_too_long() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FILENAME_TOO_LONG));
}

void error_savdir_do_not_exist(const char *savname, const char *savedir) {
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST1));
    write_log_auo_line_fmt(LOG_ERROR, L"%s: \"%s\"", g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST2), char_to_wstring(savname).c_str());
    write_log_auo_line_fmt(LOG_ERROR, L"%s: \"%s\"", g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST3), char_to_wstring(savedir).c_str());
    if (strchr(savedir, '?') != nullptr) {
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST4));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST5));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_SAVDIR_DO_NOT_EXIST6));
    }
}

void error_file_is_already_opened_by_aviutl() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FILE_ALREADY_OPENED1));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FILE_ALREADY_OPENED2));
}

void warning_no_auo_check_fileopen() {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_WARN_NO_AUO_CHECK_FILEOPEN1), char_to_wstring(AUO_CHECK_FILEOPEN_NAME).c_str(), char_to_wstring(DEFAULT_EXE_DIR).c_str());
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_WARN_NO_AUO_CHECK_FILEOPEN2), char_to_wstring(DEFAULT_EXE_DIR).c_str());
}

static void error_failed_to_open_temp_file_dir(const char *temp_filename, const char *mesBuffer, const DWORD err, const BOOL target_is_dir) {
    if (target_is_dir) {
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR1), char_to_wstring(temp_filename).c_str());
    } else {
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR2), char_to_wstring(temp_filename).c_str());
    }
    write_log_auo_line_fmt(LOG_ERROR, L"  %s", char_to_wstring(mesBuffer).c_str());
    if (strchr(temp_filename, '?') != nullptr) {
        const wchar_t *target_name = (target_is_dir) ? g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_FOLDER) : g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_FILE);
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_SPECIAL_CHAR1), target_name);
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_SPECIAL_CHAR2), target_name);
    } else if (err == ERROR_ACCESS_DENIED) {
        wchar_t systemdrive_dir[MAX_PATH_LEN] = { 0 };
        wchar_t systemroot_dir[MAX_PATH_LEN] = { 0 };
        wchar_t programdata_dir[MAX_PATH_LEN] = { 0 };
        wchar_t programfiles_dir[MAX_PATH_LEN] = { 0 };
        //char programfilesx86_dir[MAX_PATH_LEN];
        ExpandEnvironmentStringsW(L"%SystemDrive%", systemdrive_dir, _countof(systemdrive_dir));
        ExpandEnvironmentStringsW(L"%SystemRoot%", systemroot_dir, _countof(systemroot_dir));
        ExpandEnvironmentStringsW(L"%PROGRAMDATA%", programdata_dir, _countof(programdata_dir));
        ExpandEnvironmentStringsW(L"%PROGRAMFILES%", programfiles_dir, _countof(programfiles_dir));
        //ExpandEnvironmentStrings("%PROGRAMFILES(X86)%", programfilesx86_dir, _countof(programfilesx86_dir));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED1));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED2));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED3));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED4));
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED5), systemdrive_dir);
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED6), systemroot_dir);
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED6), programdata_dir);
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED6), programfiles_dir);
        //write_log_auo_line_fmt(LOG_ERROR, "    %s 以下", programfilesx86_dir);
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_ACCESS_DENIED7));
        write_log_auo_line(LOG_ERROR, L"");
    } else {
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_OPEN_TEMP_FILE_DIR_OTHER));
    }
}

void error_failed_to_open_tempdir(const char *temp_dir, const char *mesBuffer, const DWORD err) {
    error_failed_to_open_temp_file_dir(temp_dir, mesBuffer, err, true);
}

void error_failed_to_open_tempfile(const char *temp_filename, const char *mesBuffer, const DWORD err) {
    error_failed_to_open_temp_file_dir(temp_filename, mesBuffer, err, false);
}

void error_nothing_to_output() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_NOTHING_TO_OUTPUT));
}

void error_output_zero_frames() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_OUTPUT_ZERO_FRAMES1));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_OUTPUT_ZERO_FRAMES2));
}

void warning_amp_bitrate_confliction(int lower, int upper) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_BITRATE_CONFLICT1), upper, lower);
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_BITRATE_CONFLICT2));
}

void error_amp_bitrate_confliction() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_BITRATE_CONFLICT3));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_BITRATE_CONFLICT4));
}

void error_amp_afs_audio_delay_confliction() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_AFS_AUDIO_DELAY_CONFLICT1));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_AFS_AUDIO_DELAY_CONFLICT2));
}

void info_afs_audio_delay_confliction() {
    write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ERR_AFS_AUDIO_DELAY_CONFLICT));
}

static const wchar_t *get_target_limit_name(DWORD target_limit) {
    const wchar_t *str_limit = L"";
    switch (target_limit) {
        case AMPLIMIT_BITRATE_UPPER:
        case AMPLIMIT_BITRATE_LOWER:
            str_limit = g_auo_mes.get(AUO_ERR_AMP_TARGET_LIMIT_NAME_BITRATE);   break;
        case AMPLIMIT_FILE_SIZE:
            str_limit = g_auo_mes.get(AUO_ERR_AMP_TARGET_LIMIT_NAME_FILESIZE); break;
        default:
            str_limit = g_auo_mes.get(AUO_ERR_AMP_TARGET_LIMIT_NAME_BITRATE_FILESIZE);   break;
    }
    return str_limit;
}

void info_amp_do_aud_enc_first(DWORD target_limit) {
    write_log_auo_line_fmt(LOG_INFO, g_auo_mes.get(AUO_ERR_AMP_DO_AUD_ENC_FIRST), get_target_limit_name(target_limit));
}

void error_amp_aud_too_big(DWORD target_limit) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_AUD_TOO_BIG1), get_target_limit_name(target_limit));
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_AUD_TOO_BIG2));
}

void error_amp_target_bitrate_too_small(DWORD target_limit) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_TARGET_BITRATE_TOO_SMALL1), get_target_limit_name(target_limit));
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_AMP_TARGET_BITRATE_TOO_SMALL2));
}

void warning_amp_change_bitrate(int bitrate_old, int bitrate_new, DWORD target_limit) {
    if (bitrate_old > 0) {
        write_log_auo_line_fmt(LOG_WARNING, (bitrate_old > bitrate_new)
            ? g_auo_mes.get(AUO_ERR_AMP_CHANGE_BITRATE_TOO_BIG)
            : g_auo_mes.get(AUO_ERR_AMP_CHANGE_BITRATE_TOO_SMALL),
            get_target_limit_name(target_limit));
        write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_CHANGE_BITRATE_FROM_TO), bitrate_old, bitrate_new);
    } else {
        //-1は上限確認付crfで使用する
        write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_CHANGE_BITRATE_AUTO), bitrate_new);
    }
}

void error_invalid_resolution(BOOL width, int mul, int w, int h) {
    const wchar_t *resolution_x_y = (width) ? g_auo_mes.get(AUO_ERR_INVALID_RESOLUTION_WIDTH) : g_auo_mes.get(AUO_ERR_INVALID_RESOLUTION_HEIGHT);
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_INVALID_RESOLUTION), resolution_x_y, mul, w, h);
}

void error_log_line_cache() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_LOG_LINE_CACHE));
}

void error_tc2mp4_afs_not_supported() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_TC2MP4_AFS_NOT_SUPPORTED1));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_TC2MP4_AFS_NOT_SUPPORTED2));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_TC2MP4_AFS_NOT_SUPPORTED3));
}

void error_no_exe_file(const wchar_t *name, const char *path) {
    if (strlen(path))
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_EXE_FILE1), name, char_to_wstring(path).c_str());
    else
        write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_EXE_FILE2), name);
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_EXE_FILE3), name);
}

void warning_use_default_audio_encoder(const wchar_t *name) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_USE_DEFAULT_AUDIO_ENCODER), name);
}

void info_use_exe_found(const wchar_t *target, const char *path) {
    write_log_auo_line_fmt(LOG_INFO, g_auo_mes.get(AUO_ERR_INFO_USE_EXE_FOUND), target, char_to_wstring(path).c_str());
}

void error_invalid_ini_file() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_INVALID_INI_FILE));
}

void error_unsupported_audio_format_by_muxer(const int video_out_type, const wchar_t *selected_aud, const wchar_t *default_aud) {
    if (video_out_type < _countof(OUTPUT_FILE_EXT)) {
        write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_UNSUPPORTED_AUDIO_FORMAT_BY_MUXER1), selected_aud, char_to_wstring(OUTPUT_FILE_EXT[video_out_type] + 1).c_str());
        if (default_aud) {
            write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_UNSUPPORTED_AUDIO_FORMAT_BY_MUXER2), default_aud);
        } else {
            write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_UNSUPPORTED_AUDIO_FORMAT_BY_MUXER3));
        }
    }
}

void error_failed_to_run_audio_encoder(const wchar_t *selected_aud, const wchar_t *error_mes, const wchar_t *default_aud) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TO_RUN_AUDIO_ENCODER1), selected_aud);
    write_log_auo_line(LOG_WARNING, error_mes);
    if (default_aud) {
        write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TO_RUN_AUDIO_ENCODER2), default_aud);
    } else {
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_FAILED_TO_RUN_AUDIO_ENCODER3));
    }
}

void error_mp4box_ini() {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_MP4_MUXER_ERROR));
}

void warning_auto_afs_disable() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AFS_AUTO_DISABLE1));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AFS_AUTO_DISABLE2));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AFS_AUTO_DISABLE3));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AFS_AUTO_DISABLE4));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AFS_AUTO_DISABLE5));
}

void error_afs_setup(BOOL afs, BOOL auto_afs_disable) {
    if (afs && !auto_afs_disable) {
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP1));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP2));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP3));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP4));
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP5));
    } else
        write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_SETUP6));
}

void error_open_pipe() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_OPEN_PIPE));
}

void error_get_pipe_handle() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_GET_PIPE_HANDLE));
}

void error_run_process(const wchar_t *exe_name, int rp_ret) {
    switch (rp_ret) {
        case RP_ERROR_OPEN_PIPE:
            write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_OPEN_PIPE));
            break;
        case RP_ERROR_GET_STDIN_FILE_HANDLE:
            write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_GET_PIPE_HANDLE));
            break;
        case RP_ERROR_CREATE_PROCESS:
        default:
            write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_RUN_PROCESS), exe_name);
            break;
    }
}

void error_video_output_thread_start() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_OUTPUT_THREAD_START));
}

void error_video_create_param_mem() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_CREATE_PARAM_MEM));
}

void error_video_create_event() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_CREATE_EVENT));
}

void error_video_wait_event() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_WAIT_EVENT));
}

void error_video_set_event() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_SET_EVENT));
}

void error_video_open_shared_input_buf() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_OPEN_SHARED_INPUT_BUF));
}

void error_video_get_conv_func() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDEO_GET_CONV_FUNC));
}

void warning_auto_qpfile_failed() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_QPFILE_FAILED));
}

void warning_auo_tcfile_failed() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_TCFILE_FAILED));
}

void error_malloc_pixel_data() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_MALLOC_PIXEL_DATA));
}

void error_malloc_tc() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_MALLOC_TC));
}

void error_malloc_8bit() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_MALLOC_8BIT));
}

void error_afs_interlace_stg() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_INTERLACE_STG));
}

void warning_x264_mp4_output_not_supported() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_X264_MP4_OUTPUT_NOT_SUPPORTED1));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_X264_MP4_OUTPUT_NOT_SUPPORTED2));
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_X264_MP4_OUTPUT_NOT_SUPPORTED3));
}

void error_videnc_dead() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_DEAD));
}

void error_videnc_dead_and_nodiskspace(const char *drive, uint64_t diskspace) {
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_DEAD_AND_NODISKSPACE1));
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_DEAD_AND_NODISKSPACE2), char_to_wstring(drive).c_str(), (double)diskspace / (1024 * 1024));
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_DEAD_AND_NODISKSPACE3), char_to_wstring(drive).c_str());
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_DEAD_AND_NODISKSPACE4), char_to_wstring(drive).c_str());
}
void error_videnc_version(const char *required_ver, const char *current_ver) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_VERSION1));
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_VIDENC_VERSION2));
    write_log_auo_line_fmt(LOG_ERROR, L"%s: %s", g_auo_mes.get(AUO_ERR_VIDENC_VERSION3), char_to_wstring(required_ver).c_str());
    write_log_auo_line_fmt(LOG_ERROR, L"%s: %s", g_auo_mes.get(AUO_ERR_VIDENC_VERSION4), char_to_wstring(current_ver).c_str());
}

void error_afs_get_frame() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_AFS_GET_FRAME));
}

void error_open_wavfile() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_OPEN_WAVFILE));
}

void error_no_wavefile() {
    write_log_auo_line(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_WAVFILE));
}

static void message_audio_length_different(const double video_length, const double audio_length, const BOOL exedit_is_used, const BOOL audio_length_changed) {
    const int vid_h = (int)(video_length / 3600);
    const int vid_m = (int)(video_length - vid_h * 3600) / 60;
    const int vid_s = (int)(video_length - vid_h * 3600 - vid_m * 60);
    const int vid_ms = std::min((int)((video_length - (double)(vid_h * 3600 + vid_m * 60 + vid_s)) * 1000.0), 999);

    const int aud_h = (int)audio_length / 3600;
    const int aud_m = (int)(audio_length - aud_h * 3600) / 60;
    const int aud_s = (int)(audio_length - aud_h * 3600 - aud_m * 60);
    const int aud_ms = std::min((int)((audio_length - (double)(aud_h * 3600 + aud_m * 60 + aud_s)) * 1000.0), 999);

    if (audio_length_changed) {
        write_log_auo_line(    LOG_INFO, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT1));
        write_log_auo_line_fmt(LOG_INFO, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT2),
            vid_h, vid_m, vid_s, vid_ms,
            aud_h, aud_m, aud_s, aud_ms);
        write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT3));
        if (exedit_is_used) {
            write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT4));
            write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT5));
        }
    } else {
        write_log_auo_line(    LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT1));
        write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT2),
            vid_h, vid_m, vid_s, vid_ms,
            aud_h, aud_m, aud_s, aud_ms);
        if (exedit_is_used) {
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT4));
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT5));
        } else {
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT6));
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT7));
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT8));
            write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AUDIO_LENGTH_DIFFERENT9));
        }
    }
}

void info_audio_length_changed(const double video_length, const double audio_length, const BOOL exedit_is_used) {
    message_audio_length_different(video_length, audio_length, exedit_is_used, TRUE);
}

void warning_audio_length(const double video_length, const double audio_length, const BOOL exedit_is_used) {
    message_audio_length_different(video_length, audio_length, exedit_is_used, FALSE);
}

void error_audenc_failed(const wchar_t *name, const char *args) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_AUDENC_FAILED1), name);
    if (args) {
        write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_AUDENC_FAILED2));
        write_log_auo_line(    LOG_ERROR, char_to_wstring(args).c_str());
    }
}

void error_mux_failed(const wchar_t *name, const char *args) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_MUX_FAILED1), name);
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_MUX_FAILED2));
    write_log_auo_line(    LOG_ERROR, char_to_wstring(args).c_str());
}

void warning_no_mux_tmp_root(const char *dir) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_NO_MUX_TMP_ROOT), char_to_wstring(dir).c_str());
}

void warning_failed_mux_tmp_drive_space(const char *drivename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TMP_DRIVE_SPACE), char_to_wstring(drivename).c_str());
}

void warning_failed_muxer_drive_space(const char *drivename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_MUX_DRIVE_SPACE), char_to_wstring(drivename).c_str());
}

void warning_failed_out_drive_space(const char *drivename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_OUT_DRIVE_SPACE), char_to_wstring(drivename).c_str());
}

void warning_failed_get_aud_size(const char *filename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_GET_AUD_SIZE), char_to_wstring(filename).c_str());
}

void warning_failed_get_vid_size(const char *filename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_GET_VID_SIZE), char_to_wstring(filename).c_str());
}

void error_no_aud_file(const char *filename) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_AUD_FILE), char_to_wstring(filename).c_str());
}

void error_no_vid_file(const char *filename) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_VID_FILE), char_to_wstring(filename).c_str());
}

void error_aud_file_zero_byte(const char *filename) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_AUD_FILE_ZERO_BYTE), char_to_wstring(filename).c_str());
}

void error_vid_file_zero_byte(const char *filename) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_VID_FILE_ZERO_BYTE), char_to_wstring(filename).c_str());
}

void warning_mux_tmp_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_TMP_NO_ENOUGH_SPACE1), char_to_wstring(drive).c_str());
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_NO_ENOUGH_SPACE_SHOW_SIZE), (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
    write_log_auo_line    (LOG_WARNING, g_auo_mes.get(AUO_ERR_TMP_NO_ENOUGH_SPACE2));
}

void error_muxer_drive_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_MUX_NO_ENOUGH_SPACE), char_to_wstring(drive).c_str());
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_ENOUGH_SPACE_SHOW_SIZE), (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
}

void error_out_drive_not_enough_space(const char *drive, const uint64_t free_diskspace, const uint64_t required_diskspace) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_OUT_NO_ENOUGH_SPACE), char_to_wstring(drive).c_str());
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_NO_ENOUGH_SPACE_SHOW_SIZE), (double)required_diskspace / (1024 * 1024), (double)free_diskspace / (1024 * 1024));
}

void warning_failed_to_get_duration_from_timecode() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TO_GET_DURATION_FROM_TIMECODE1));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TO_GET_DURATION_FROM_TIMECODE2));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_TO_GET_DURATION_FROM_TIMECODE3));
}

void error_check_muxout_exist(const char *filename) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_CHECK_MUXOUT_EXIST), char_to_wstring(filename).c_str());
}

void error_check_muxout_too_small(const char *filename, int expected_filesize_KB, int muxout_filesize_KB) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_CHECK_MUXOUT_TO_SMALL1), char_to_wstring(filename).c_str());
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_CHECK_MUXOUT_TO_SMALL2), expected_filesize_KB, muxout_filesize_KB);
}

void warning_failed_check_muxout_filesize(const char *filename) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_CHECK_MUXOUT_GET_SIZE), char_to_wstring(filename).c_str());
}

std::wstring getLastErrorStr(DWORD err) {
    std::wstring message;
    char *mesBuffer = nullptr;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&mesBuffer, 0, NULL);
    if (mesBuffer != nullptr) {
        message = char_to_wstring(mesBuffer);
        LocalFree(mesBuffer);
    }
    return message;
}

void error_failed_remove_file(const char *filename, const DWORD err) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_REMOVE_FILE), getLastErrorStr(err).c_str(), char_to_wstring(filename).c_str());
}

void error_failed_rename_file(const char *filename, const DWORD err) {
    write_log_auo_line_fmt(LOG_ERROR, g_auo_mes.get(AUO_ERR_RENAME_FILE), getLastErrorStr(err).c_str(), char_to_wstring(filename).c_str());
}

void warning_amp_failed() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_FAILED));
}

void warning_amp_filesize_over_limit() {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_FILESIZE_OVER_LIMIT1));
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_AMP_FILESIZE_OVER_LIMIT2));
}

void warning_no_auto_save_log_dir() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_NO_AUTO_SAVE_LOG_DIR));
}

void info_encoding_aborted() {
    write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ERR_ABORT));
}

void warning_mux_no_chapter_file() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_MUX_NO_CHAPTER_FILE));
}

void info_amp_result(DWORD status, int amp_result, UINT64 filesize, double file_bitrate, double limit_filesize, double limit_filebitrate_upper, double limit_filebitrate_lower, int retry_count, int new_bitrate) {
    int log_index = (status) ? ((amp_result) ? LOG_WARNING : LOG_ERROR) : LOG_INFO;
    write_log_auo_line_fmt(    log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT1), filesize / (double)(1024*1024), file_bitrate);
    if (status & AMPLIMIT_FILE_SIZE)
        write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT2), limit_filesize);
    if (status & AMPLIMIT_BITRATE_UPPER)
        write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT3), limit_filebitrate_upper);
    if (status & AMPLIMIT_BITRATE_LOWER)
        write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT4), limit_filebitrate_lower);
    if (status && amp_result)
        if (amp_result == 2)
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT5), new_bitrate);
        else if (new_bitrate > 0) //-1, 0は上限確認付crfで使用する
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT6), new_bitrate);

    if (!status)
        write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT7));
    else if (!amp_result) {
        if (status & (AMPLIMIT_BITRATE_UPPER | AMPLIMIT_FILE_SIZE)) {
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT8), retry_count);
        } else if (status & AMPLIMIT_BITRATE_LOWER) {
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT9), retry_count);
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT10));
            write_log_auo_line_fmt(log_index, g_auo_mes.get(AUO_ERR_AMP_RESULT11));
        }
    }
}

void warning_mux_chapter(int sts) {
    AuoMes id = AUO_MES_UNKNOWN;
    switch (sts) {
        case AUO_CHAP_ERR_NONE: break;
        case AUO_CHAP_ERR_FILE_OPEN:        id = AUO_ERR_MUX_CHPATER_OPEN; break;
        case AUO_CHAP_ERR_FILE_READ:        id = AUO_ERR_MUX_CHPATER_READ; break;
        case AUO_CHAP_ERR_FILE_WRITE:       id = AUO_ERR_MUX_CHPATER_WRITE; break;
        case AUO_CHAP_ERR_FILE_SWAP:        id = AUO_ERR_MUX_CHPATER_SWAP; break;
        case AUO_CHAP_ERR_CP_DETECT:        id = AUO_ERR_MUX_CHPATER_CP_DETECT; break;
        case AUO_CHAP_ERR_INIT_IMUL2:       id = AUO_ERR_MUX_CHPATER_INIT_IMUL2; break;
        case AUO_CHAP_ERR_INVALID_FMT:      id = AUO_ERR_MUX_CHPATER_INVALID_FMT; break;
        case AUO_CHAP_ERR_NULL_PTR:         id = AUO_ERR_MUX_CHPATER_NULL_PTR; break;
        case AUO_CHAP_ERR_INIT_XML_PARSER:  id = AUO_ERR_MUX_CHPATER_INIT_XML_PARSE; break;
        case AUO_CHAP_ERR_INIT_READ_STREAM: id = AUO_ERR_MUX_CHPATER_INIT_READ_STREAM; break;
        case AUO_CHAP_ERR_FAIL_SET_STREAM:  id = AUO_ERR_MUX_CHPATER_SET_STREAM; break;
        case AUO_CHAP_ERR_PARSE_XML:        id = AUO_ERR_MUX_CHPATER_PARSE_XML; break;
        default:                            id = AUO_ERR_MUX_CHPATER_UNKNOWN; break;
    }
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(id));
    return;
}

void warning_chapter_convert_to_utf8(int sts) {
    write_log_auo_line_fmt(LOG_WARNING, g_auo_mes.get(AUO_ERR_CHPATER_CONVERT));
    warning_mux_chapter(sts);
}

void error_select_convert_func(int width, int height, int bit_depth, BOOL interlaced, int output_csp) {
    const wchar_t *bit_depth_str = L"";
    switch (bit_depth) {
    case 16: bit_depth_str = L"(16bit)"; break;
    case 12: bit_depth_str = L"(12bit)"; break;
    case 10: bit_depth_str = L"(10bit)"; break;
    default: break;
    }
    write_log_auo_line(    LOG_ERROR, g_auo_mes.get(AUO_ERR_SEL_CONVERT_FUNC));
    write_log_auo_line_fmt(LOG_ERROR, L"%dx%d%s, output-csp %s%s",
        width, height,
        (interlaced) ? L"i" : L"p",
        char_to_wstring(specify_csp[output_csp]).c_str(),
        bit_depth_str
        );
}

void warning_no_batfile(const char *batfile) {
    write_log_auo_line_fmt(LOG_WARNING, L"%s: %s", g_auo_mes.get(AUO_ERR_NO_BAT_FILE), char_to_wstring(batfile).c_str());
}

void warning_malloc_batfile_tmp() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_MALLOC_BAT_FILE_TMP));
}

void warning_failed_open_bat_orig() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_OPEN_BAT_ORG));
}

void warning_failed_open_bat_new() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_FAILED_OPEN_BAT_NEW));
}

void warning_video_very_short() {
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_VIDEO_VERY_SHORT1));
    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ERR_VIDEO_VERY_SHORT2));
}
