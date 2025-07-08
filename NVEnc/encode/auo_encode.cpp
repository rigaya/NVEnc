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
#include <cmath>
#include <float.h>
#include <stdio.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <functional>

#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_pipe.h"
#include "auo_mes.h"

#include "auo_frm.h"
#include "auo_video.h"
#include "auo_encode.h"
#include "auo_error.h"
#include "auo_audio.h"
#include "auo_faw2aac.h"
#include "rgy_env.h"
#include "rgy_filesystem.h"
#include "exe_version.h"

using unique_handle = std::unique_ptr<std::remove_pointer<HANDLE>::type, std::function<void(HANDLE)>>;

static void create_aviutl_opened_file_list(PRM_ENC *pe);
static bool check_file_is_aviutl_opened_file(const char *filepath, const PRM_ENC *pe);

static void avoid_exsisting_tmp_file(char *buf, size_t size) {
    if (!PathFileExists(buf)) {
        return;
    }
    char tmp[MAX_PATH_LEN];
    for (int i = 0; i < 1000000; i++) {
        char new_ext[32];
        sprintf_s(new_ext, ".%d%s", i, PathFindExtension(buf));
        strcpy_s(tmp, buf);
        change_ext(tmp, size, new_ext);
        if (!PathFileExists(tmp)) {
            strcpy_s(buf, size, tmp);
            return;
        }
    }
}

#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1
#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4996)
static inline std::string tolowercase(const std::string& str) {
    std::string str_copy = str;
    std::transform(str_copy.cbegin(), str_copy.cend(), str_copy.begin(), tolower);
    return str_copy;
}
#pragma warning (pop)
#endif

static std::vector<std::filesystem::path> find_exe_files(const char *target_dir) {
    std::vector<std::filesystem::path> ret;
    try {
        for (const std::filesystem::directory_entry& x : std::filesystem::recursive_directory_iterator(target_dir)) {
            if (x.path().extension() == ".exe") {
                ret.push_back(x.path());
            }
        }
    } catch (...) {}
    return ret;
}

static std::vector<std::filesystem::path> find_exe_files(const char *target_dir, const char *target_dir2) {
    auto list1 = find_exe_files(target_dir);
    auto list2 = find_exe_files(target_dir2);
    list1.insert(list1.end(), list2.begin(), list2.end());
    return list1;
}

static std::vector<std::filesystem::path> find_target_exe_files(const char *target_name, const std::vector<std::filesystem::path>& exe_files) {
    std::vector<std::filesystem::path> ret;
    const auto targetNameLower = tolowercase(std::filesystem::path(target_name).stem().string());
    for (const auto& path : exe_files) {
        if (tolowercase(path.stem().string()).substr(0, targetNameLower.length()) == targetNameLower) {
            ret.push_back(path);
        }
    }
    return ret;
}

static bool ends_with(const std::string& s, const std::string& check) {
    if (s.size() < check.size()) return false;
    return std::equal(std::rbegin(check), std::rend(check), std::rbegin(s));
}

static std::vector<std::filesystem::path> select_exe_file(const std::vector<std::filesystem::path>& pathList) {
    if (pathList.size() <= 1) {
        return pathList;
    }
    std::vector<std::filesystem::path> exe32bit;
    std::vector<std::filesystem::path> exe64bit;
    std::vector<std::filesystem::path> exeUnknown;
    for (const auto& path : pathList) {
        if (ends_with(tolowercase(path.filename().string()), "_x64.exe")) {
            exe64bit.push_back(path);
            continue;
        } else if (ends_with(tolowercase(path.filename().string()), "_x86.exe")) {
            exe32bit.push_back(path);
            continue;
        }
        bool checked = false;
        std::filesystem::path p = path;
        for (int i = 0; p.string().length() > 0 && i < 10000; i++) {
            auto parent = p.parent_path();
            if (parent == p) {
                break;
            }
            if (p.filename().string() == "x64") {
                exe64bit.push_back(path);
                checked = true;
                break;
            } else if (p.filename().string() == "x86") {
                exe32bit.push_back(path);
                checked = true;
                break;
            }
        }
        if (!checked) {
            if (ends_with(tolowercase(path.filename().string()), "64.exe")) {
                exe64bit.push_back(path);
            } else {
                exeUnknown.push_back(path);
            }
        }
    }
    if (rgy_is_64bit_os()) {
        return (exe64bit.size() > 0) ? exe64bit : exeUnknown;
    } else {
        return (exe32bit.size() > 0) ? exe32bit : exeUnknown;
    }
}

std::filesystem::path find_latest_videnc(const std::vector<std::filesystem::path>& pathList) {
    if (pathList.size() == 0) {
        return std::filesystem::path();
    }
    auto selectedPathList = select_exe_file(pathList);
    if (selectedPathList.size() == 1) {
        return selectedPathList.front();
    }
    int version[4] = { 0 };
    std::filesystem::path ret;
    for (auto& path : selectedPathList) {
        int value[4] = { 0 };
#if ENCODER_X264
        value[0] = get_x264_rev(path.string().c_str());
        if (value[0] >= version[0]) {
            version[0] = value[0];
            ret = path;
    	}
#elif ENCODER_X265
        if (get_x265_rev(path.string().c_str(), value) == 0) {
            if (version_a_larger_than_b(value, version) > 0) {
                memcpy(version, value, sizeof(version));
                ret = path;
            }
        }
#elif ENCODER_SVTAV1
        if (get_svtav1_rev(path.string().c_str(), value) == 0) {
            if (version_a_larger_than_b(value, version) > 0) {
                memcpy(version, value, sizeof(version));
                ret = path;
            }
        }
#elif ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
        if (get_exe_version_info(path.string().c_str(), value) == 0) {
            if (version_a_larger_than_b(value, version) > 0) {
                memcpy(version, value, sizeof(version));
                ret = path;
            }
        }
#else
		static_assert(false);
#endif
    }
    return ret;
}

std::string find_latest_videnc_for_frm() {
    char aviutl_dir[MAX_PATH_LEN] = { 0 };
    get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));

    char defaultExeDir[MAX_PATH_LEN] = { 0 };
    PathCombineLong(defaultExeDir, _countof(defaultExeDir), aviutl_dir, DEFAULT_EXE_DIR);

    const auto exeFiles = find_exe_files(defaultExeDir);
    const auto targetExes = find_target_exe_files(ENCODER_NAME, exeFiles);
    if (targetExes.size() > 0) {
        const auto latestVidEnc = find_latest_videnc(targetExes);
        return latestVidEnc.string().c_str();
    }
    return "";
}

void get_audio_pipe_name(char *pipename, size_t nSize, int audIdx) {
    sprintf_s(pipename, nSize, AUO_NAMED_PIPE_BASE, GetCurrentProcessId(), audIdx);
}

static BOOL check_muxer_exist(MUXER_SETTINGS *muxer_stg, const char *aviutl_dir, const BOOL get_relative_path, const std::vector<std::filesystem::path>& exe_files) {
    if (PathFileExists(muxer_stg->fullpath)) {
        info_use_exe_found(muxer_stg->dispname, muxer_stg->fullpath);
        return TRUE;
    }
    const auto targetExes = select_exe_file(find_target_exe_files(muxer_stg->filename, exe_files));
    if (targetExes.size() > 0) {
        if (get_relative_path) {
            GetRelativePathTo(muxer_stg->fullpath, _countof(muxer_stg->fullpath), targetExes.front().string().c_str(), FILE_ATTRIBUTE_NORMAL, aviutl_dir);
        } else {
            strcpy_s(muxer_stg->fullpath, targetExes.front().string().c_str());
        }
    }
    if (PathFileExists(muxer_stg->fullpath)) {
        info_use_exe_found(muxer_stg->dispname, muxer_stg->fullpath);
        return TRUE;
    }
    error_no_exe_file(muxer_stg->dispname, muxer_stg->fullpath);
    return FALSE;
}

const MUXER_CMD_EX *get_muxer_mode(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, int muxer_to_be_used) {
    int mode = -1;
    switch (muxer_to_be_used) {
    case MUXER_TC2MP4:
    case MUXER_MP4:      mode = conf->mux.mp4_mode; break;
    case MUXER_MKV:      mode = conf->mux.mkv_mode; break;
    case MUXER_INTERNAL: mode = conf->mux.internal_mode; break;
    default: break;
    }
    return (mode >= 0) ? &sys_dat->exstg->s_mux[muxer_to_be_used].ex_cmd[mode] : nullptr;
}

static BOOL check_if_exe_is_mp4box(const char *exe_path, const char *version_arg) {
    BOOL ret = FALSE;
    char exe_message[8192] = { 0 };
    if (   PathFileExists(exe_path)
        && RP_SUCCESS == get_exe_message(exe_path, version_arg, exe_message, _countof(exe_message), AUO_PIPE_MUXED)
        && (stristr(exe_message, "mp4box") || stristr(exe_message, "GPAC"))) {
        ret = TRUE;
    }
    return ret;
}

static BOOL check_if_exe_is_lsmash(const char *exe_path, const char *version_arg) {
    BOOL ret = FALSE;
    char exe_message[8192] = { 0 };
    if (   PathFileExists(exe_path)
        && RP_SUCCESS == get_exe_message(exe_path, version_arg, exe_message, _countof(exe_message), AUO_PIPE_MUXED)
        && stristr(exe_message, "L-SMASH")) {
        ret = TRUE;
    }
    return ret;
}

static BOOL check_muxer_matched_with_ini(const MUXER_SETTINGS *mux_stg) {
    BOOL ret = TRUE;
    //不確定な場合は"0", mp4boxなら"-1", L-SMASHなら"1"
    if (ENCODER_X264 || ENCODER_X265) {
        bool mp4box_ini = stristr(mux_stg[MUXER_MP4].filename, "mp4box") != nullptr;
        if (mp4box_ini) {
            error_mp4box_ini();
            ret = FALSE;
        }
    }
    return ret;
}

bool is_afsvfr(const CONF_GUIEX *conf) {
    return conf->vid.afs != 0;
}

static BOOL check_amp(CONF_GUIEX *conf) {
    BOOL check = TRUE;
#if ENABLE_AMP
    if (!conf->enc.use_auto_npass)
        return check;
    if (conf->vid.amp_check & AMPLIMIT_BITRATE_UPPER) {
        //if (conf->x264.bitrate > conf->vid.amp_limit_bitrate_upper) {
        //    check = FALSE; error_amp_bitrate_confliction();
        //} else if (conf->vid.amp_limit_bitrate_upper <= 0.0)
        //    conf->vid.amp_check &= ~AMPLIMIT_BITRATE; //フラグを折る
        if (conf->vid.amp_limit_bitrate_upper <= 0.0)
            conf->vid.amp_check &= ~AMPLIMIT_BITRATE_UPPER; //フラグを折る
    }
    if (conf->vid.amp_check & AMPLIMIT_FILE_SIZE) {
        if (conf->vid.amp_limit_file_size <= 0.0)
            conf->vid.amp_check &= ~AMPLIMIT_FILE_SIZE; //フラグを折る
    }
    if (conf->vid.amp_check && conf->vid.afs && AUDIO_DELAY_CUT_ADD_VIDEO == conf->aud.delay_cut) {
        check = FALSE; error_amp_afs_audio_delay_confliction();
    }
#endif
    return check;
}

static BOOL muxer_supports_audio_format(const int muxer_to_be_used, const AUDIO_SETTINGS *aud_stg) {
    switch (muxer_to_be_used) {
    case MUXER_TC2MP4:
    case MUXER_MP4_RAW:
    case MUXER_MP4:
        return aud_stg->unsupported_mp4 == 0;
    case MUXER_MKV:
    case MUXER_DISABLED:
    case MUXER_INTERNAL:
        return TRUE;
    default:
        return FALSE;
    }
}

BOOL check_if_exedit_is_used() {
    char name[256];
    wsprintf(name, "exedit_%d_%d", '01', GetCurrentProcessId());
    auto handle = unique_handle(OpenFileMapping(FILE_MAP_WRITE, FALSE, name),
        [](HANDLE h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); });

    return handle != nullptr;
}

static std::string find_auo_check_fileopen(const char *defaultExeDir, const char *defaultExeDir2) {
    char exe_path[MAX_PATH_LEN] = { 0 };
    PathCombine(exe_path, defaultExeDir, AUO_CHECK_FILEOPEN_NAME);
    if (PathFileExists(exe_path)) {
        return exe_path;
    }
    PathCombine(exe_path, defaultExeDir2, AUO_CHECK_FILEOPEN_NAME);
    if (PathFileExists(exe_path)) {
        return exe_path;
    }
    return "";
}

static BOOL check_temp_file_open(const char *target, const std::string& auo_check_fileopen_path, const bool check_dir, const bool auo_check_fileopen_warning) {
    DWORD err = ERROR_SUCCESS;

    if (is_64bit_os() && (auo_check_fileopen_path.length() == 0 || !PathFileExists(auo_check_fileopen_path.c_str())) && auo_check_fileopen_warning) {
        warning_no_auo_check_fileopen();
    }

    char test_filename[MAX_PATH_LEN];
    if (check_dir) {
        PathCombineLong(test_filename, _countof(test_filename), target, "auo_test_tempfile.tmp");
        avoid_exsisting_tmp_file(test_filename, _countof(test_filename));
    } else {
        strcpy_s(test_filename, target);
    }

    if (is_64bit_os() && auo_check_fileopen_path.length() > 0 && PathFileExists(auo_check_fileopen_path.c_str())) {
        //64bit OSでは、32bitアプリに対してはVirtualStoreが働く一方、
        //64bitアプリに対してはVirtualStoreが働かない
        //x264を64bitで実行することを考慮すると、
        //Aviutl(32bit)からチェックしても意味がないので、64bitプロセスからのチェックを行う
        PROCESS_INFORMATION pi;
        PIPE_SET pipes;
        InitPipes(&pipes);

        char fullargs[4096] = { 0 };
        sprintf_s(fullargs, "\"%s\" \"%s\"", auo_check_fileopen_path.c_str(), test_filename);

        char exeDir[MAX_PATH_LEN];
        strcpy_s(exeDir, auo_check_fileopen_path.c_str());
        PathRemoveFileSpecFixed(exeDir);

        int ret = 0;
        if ((ret = RunProcess(fullargs, exeDir, &pi, &pipes, NORMAL_PRIORITY_CLASS, TRUE, FALSE)) == RP_SUCCESS) {
            WaitForSingleObject(pi.hProcess, INFINITE);
            GetExitCodeProcess(pi.hProcess, &err);
            CloseHandle(pi.hProcess);
        }
        if (err == ERROR_SUCCESS) {
            return TRUE;
        }
    } else {
        auto handle = unique_handle(CreateFile(test_filename, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL),
            [](HANDLE h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); });
        if (handle.get() != INVALID_HANDLE_VALUE) {
            handle.reset();
            DeleteFile(test_filename);
            return TRUE;
        }
        err = GetLastError();
    }
    if (err != ERROR_ALREADY_EXISTS) {
        char *mesBuffer = nullptr;
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR)&mesBuffer, 0, NULL);
        if (check_dir) {
            error_failed_to_open_tempdir(target, mesBuffer, err);
        } else {
            error_failed_to_open_tempfile(target, mesBuffer, err);
        }
        if (mesBuffer != nullptr) {
            LocalFree(mesBuffer);
        }
    }
    return FALSE;
}

BOOL audio_encoder_exe_exists(const CONF_GUIEX *conf, const guiEx_settings *exstg) {
    const BOOL use_internal = conf->aud.use_internal;
    const int aud_idx = (use_internal) ? conf->aud.in.encoder : conf->aud.ext.encoder;
    const AUDIO_SETTINGS *aud_list = (use_internal) ? exstg->s_aud_int : exstg->s_aud_ext;
    const AUDIO_SETTINGS *aud_stg = &aud_list[aud_idx];
    if (!str_has_char(aud_stg->filename)) {
        return TRUE;
    }
    if (aud_idx == exstg->get_faw_index(use_internal)) {
        return TRUE;
    }
    return PathFileExists(aud_stg->fullpath);
}

BOOL check_output(CONF_GUIEX *conf, OUTPUT_INFO *oip, const PRM_ENC *pe, guiEx_settings *exstg) {
    BOOL check = TRUE;
    //ファイル名長さ
    if (strlen(oip->savefile) > (MAX_PATH_LEN - MAX_APPENDIX_LEN - 1)) {
        error_filename_too_long();
        check = FALSE;
    }

    char aviutl_dir[MAX_PATH_LEN] = { 0 };
    get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));

    char defaultExeDir[MAX_PATH_LEN] = { 0 };
    PathCombineLong(defaultExeDir, _countof(defaultExeDir), aviutl_dir, DEFAULT_EXE_DIR);

    char pluginsDir[MAX_PATH_LEN] = { 0 };
    char defaultExeDir2[MAX_PATH_LEN] = { 0 };
    PathCombineLong(pluginsDir, _countof(pluginsDir), aviutl_dir, "plugins");
    PathCombineLong(defaultExeDir2, _countof(defaultExeDir2), pluginsDir, DEFAULT_EXE_DIR);

    const auto auo_check_fileopen_path = find_auo_check_fileopen(defaultExeDir, defaultExeDir2);

    //ダメ文字・環境依存文字チェック
    char savedir[MAX_PATH_LEN] = { 0 };
    strcpy_s(savedir, oip->savefile);
    PathRemoveFileSpecFixed(savedir);
    if (!PathIsDirectory(savedir)) {
        error_savdir_do_not_exist(oip->savefile, savedir);
        check = FALSE;
    //出力フォルダにファイルを開けるかどうか
    } else if (!check_temp_file_open(savedir, auo_check_fileopen_path, true, true)) {
        check = FALSE;
    //一時ファイルを開けるかどうか
    } else if (!check_temp_file_open(pe->temp_filename, auo_check_fileopen_path, false, false)) {
        check = FALSE;
    }

    if (check_file_is_aviutl_opened_file(oip->savefile, pe)) {
        error_file_is_already_opened_by_aviutl();
        check = FALSE;
    }

    //解像度
    int w_mul = 2, h_mul = 2;
    if (oip->w % w_mul) {
        error_invalid_resolution(TRUE,  w_mul, oip->w, oip->h);
        check = FALSE;
    }
    if (oip->h % h_mul) {
        //error_invalid_resolution(FALSE, h_mul, oip->w, oip->h);
        //check = FALSE;
        //切り捨て
        oip->h = (int)(oip->h / h_mul) * h_mul;
    }

    //出力するもの
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED && !(oip->flag & OUTPUT_INFO_FLAG_AUDIO)) {
        error_nothing_to_output();
        check = FALSE;
    }
    if (pe->video_out_type != VIDEO_OUTPUT_DISABLED && oip->n <= 0) {
        error_output_zero_frames();
        check = FALSE;
    }

    if (conf->oth.out_audio_only)
        write_log_auo_line(LOG_INFO, g_auo_mes.get(AUO_ENCODE_AUDIO_ONLY));

    const auto exeFiles = find_exe_files(defaultExeDir, defaultExeDir2);

    //必要な実行ファイル
    if (pe->video_out_type != VIDEO_OUTPUT_DISABLED) {
        if (!PathFileExists(exstg->s_vid.fullpath)) {
            const auto targetExes = find_target_exe_files(ENCODER_APP_NAME, exeFiles);
            if (targetExes.size() > 0) {
                const auto latestVidEnc = find_latest_videnc(targetExes);
                if (exstg->s_local.get_relative_path) {
                    GetRelativePathTo(exstg->s_vid.fullpath, _countof(exstg->s_vid.fullpath), latestVidEnc.string().c_str(), FILE_ATTRIBUTE_NORMAL, aviutl_dir);
                } else {
                    strcpy_s(exstg->s_vid.fullpath, latestVidEnc.string().c_str());
                }
            }
            if (!PathFileExists(exstg->s_vid.fullpath)) {
                error_no_exe_file(ENCODER_APP_NAME_W, exstg->s_vid.fullpath);
                check = FALSE;
            }
        }
        info_use_exe_found(ENCODER_NAME_W, exstg->s_vid.fullpath);
    }

    //音声エンコーダ
    if (oip->flag & OUTPUT_INFO_FLAG_AUDIO) {
        //音声長さチェック
        if (check_audio_length(oip, exstg->s_local.av_length_threshold)) {
            check = FALSE;
        }

        if (conf->aud.use_internal) {
            CONF_AUDIO_BASE *cnf_aud = &conf->aud.in;
            cnf_aud->audio_encode_timing = 2;
            cnf_aud->delay_cut = AUDIO_DELAY_CUT_NONE;

            const bool default_audenc_cnf_avail = (exstg->s_local.default_audio_encoder_in < exstg->s_aud_int_count
                && str_has_char(exstg->s_aud_int[exstg->s_local.default_audio_encoder_in].filename));
            const bool default_audenc_auo_avail = (DEFAULT_AUDIO_ENCODER_IN < exstg->s_aud_int_count
                && str_has_char(exstg->s_aud_int[DEFAULT_AUDIO_ENCODER_IN].filename));
            if (cnf_aud->encoder < 0 || exstg->s_aud_int_count <= cnf_aud->encoder) {
                if (default_audenc_cnf_avail) {
                    cnf_aud->encoder = exstg->s_local.default_audio_encoder_ext;
                    warning_use_default_audio_encoder(exstg->s_aud_int[cnf_aud->encoder].dispname);
                } else if (default_audenc_auo_avail) {
                    cnf_aud->encoder = DEFAULT_AUDIO_ENCODER_IN;
                    warning_use_default_audio_encoder(exstg->s_aud_int[cnf_aud->encoder].dispname);
                }
            }
            if (cnf_aud->encoder < 0 || exstg->s_aud_int_count <= cnf_aud->encoder) {
                error_invalid_ini_file();
                check = FALSE;
            }
            AUDIO_SETTINGS *aud_stg = &exstg->s_aud_int[cnf_aud->encoder];
            if (!muxer_supports_audio_format(pe->muxer_to_be_used, aud_stg)) {
                AUDIO_SETTINGS *aud_default = nullptr;
                if (default_audenc_cnf_avail) {
                    aud_default = &exstg->s_aud_ext[exstg->s_local.default_audio_encoder_ext];
                } else if (default_audenc_auo_avail) {
                    aud_default = &exstg->s_aud_ext[DEFAULT_AUDIO_ENCODER_EXT];
                }
                error_unsupported_audio_format_by_muxer(pe->video_out_type, aud_stg->dispname, (aud_default) ? aud_default->dispname : nullptr);
                check = FALSE;
            }
        } else {
            CONF_AUDIO_BASE *cnf_aud = &conf->aud.ext;
            const bool default_audenc_cnf_avail = (exstg->s_local.default_audio_encoder_ext < exstg->s_aud_ext_count
                && str_has_char(exstg->s_aud_ext[exstg->s_local.default_audio_encoder_ext].filename));
            const bool default_audenc_auo_avail = (DEFAULT_AUDIO_ENCODER_EXT < exstg->s_aud_ext_count
                && str_has_char(exstg->s_aud_ext[DEFAULT_AUDIO_ENCODER_EXT].filename));
            if ((cnf_aud->encoder < 0 || exstg->s_aud_ext_count <= cnf_aud->encoder)) {
                if (default_audenc_cnf_avail) {
                    cnf_aud->encoder = exstg->s_local.default_audio_encoder_ext;
                    warning_use_default_audio_encoder(exstg->s_aud_ext[cnf_aud->encoder].dispname);
                } else if (default_audenc_auo_avail) {
                    cnf_aud->encoder = DEFAULT_AUDIO_ENCODER_EXT;
                    warning_use_default_audio_encoder(exstg->s_aud_ext[cnf_aud->encoder].dispname);
                }
            }
            for (;;) {
                if (cnf_aud->encoder < 0 || exstg->s_aud_ext_count <= cnf_aud->encoder) {
                    error_invalid_ini_file();
                    check = FALSE;
                    break;
                }
                AUDIO_SETTINGS *aud_stg = &exstg->s_aud_ext[cnf_aud->encoder];
                if (!muxer_supports_audio_format(pe->muxer_to_be_used, aud_stg)) {
                    const bool retry_with_default_audenc = false; // ffmpeg_audencを配布していないQSV/NV/VCEEncではここのretryは無効化する
                    const int orig_encoder = cnf_aud->encoder;
                    if (retry_with_default_audenc) {
                        if (default_audenc_cnf_avail
                            && orig_encoder != exstg->s_local.default_audio_encoder_ext
                            && 0 <= exstg->s_local.default_audio_encoder_ext && exstg->s_local.default_audio_encoder_ext < exstg->s_aud_ext_count
                            && muxer_supports_audio_format(pe->muxer_to_be_used, &exstg->s_aud_ext[exstg->s_local.default_audio_encoder_ext])) {
                            cnf_aud->encoder = exstg->s_local.default_audio_encoder_ext;
                        } else if (default_audenc_auo_avail) {
                            cnf_aud->encoder = DEFAULT_AUDIO_ENCODER_EXT;
                        }
                    }
                    error_unsupported_audio_format_by_muxer(pe->video_out_type,
                        exstg->s_aud_ext[orig_encoder].dispname,
                        (orig_encoder != cnf_aud->encoder) ? exstg->s_aud_ext[cnf_aud->encoder].dispname : nullptr);
                    // 同じエンコーダあるいはデフォルトエンコーダがうまく取得できな場合は再チェックしても意味がない
                    if (orig_encoder == cnf_aud->encoder) {
                        check = FALSE;
                        break;
                    }
                    // デフォルトエンコーダに戻して再チェック
                    warning_use_default_audio_encoder(exstg->s_aud_ext[cnf_aud->encoder].dispname);
                    continue;
                }
                if (!audio_encoder_exe_exists(conf, exstg)) {
                    //とりあえず、exe_filesを探す
                    {
                        const auto targetExes = select_exe_file(find_target_exe_files(aud_stg->filename, exeFiles));
                        if (targetExes.size() > 0) {
                            if (exstg->s_local.get_relative_path) {
                                GetRelativePathTo(aud_stg->fullpath, _countof(aud_stg->fullpath), targetExes.front().string().c_str(), FILE_ATTRIBUTE_NORMAL, aviutl_dir);
                            } else {
                                strcpy_s(aud_stg->fullpath, targetExes.front().string().c_str());
                            }
                        }
                    }
                    //みつからなければ、デフォルトエンコーダを探す
                    if (!PathFileExists(aud_stg->fullpath) && default_audenc_cnf_avail) {
                        cnf_aud->encoder = exstg->s_local.default_audio_encoder_ext;
                        aud_stg = &exstg->s_aud_ext[cnf_aud->encoder];
                        if (!PathFileExists(aud_stg->fullpath)) {
                            const auto targetExes = select_exe_file(find_target_exe_files(aud_stg->filename, exeFiles));
                            if (targetExes.size() > 0) {
                                if (exstg->s_local.get_relative_path) {
                                    GetRelativePathTo(aud_stg->fullpath, _countof(aud_stg->fullpath), targetExes.front().string().c_str(), FILE_ATTRIBUTE_NORMAL, aviutl_dir);
                                } else {
                                    strcpy_s(aud_stg->fullpath, targetExes.front().string().c_str());
                                }
                                warning_use_default_audio_encoder(aud_stg->dispname);
                            }
                        }
                    }
                    if (!PathFileExists(aud_stg->fullpath) && default_audenc_auo_avail) {
                        cnf_aud->encoder = DEFAULT_AUDIO_ENCODER_EXT;
                        aud_stg = &exstg->s_aud_ext[cnf_aud->encoder];
                        if (!PathFileExists(aud_stg->fullpath)) {
                            const auto targetExes = select_exe_file(find_target_exe_files(aud_stg->filename, exeFiles));
                            if (targetExes.size() > 0) {
                                if (exstg->s_local.get_relative_path) {
                                    GetRelativePathTo(aud_stg->fullpath, _countof(aud_stg->fullpath), targetExes.front().string().c_str(), FILE_ATTRIBUTE_NORMAL, aviutl_dir);
                                } else {
                                    strcpy_s(aud_stg->fullpath, targetExes.front().string().c_str());
                                }
                                warning_use_default_audio_encoder(aud_stg->dispname);
                            }
                        }
                    }
                    if (!PathFileExists(aud_stg->fullpath)) {
                        //fawの場合はOK
                        if (cnf_aud->encoder != exstg->get_faw_index(conf->aud.use_internal)) {
                            error_no_exe_file(aud_stg->dispname, aud_stg->fullpath);
                            check = FALSE;
                            break;
                        }
                    }
                }
                if (str_has_char(aud_stg->filename) && (cnf_aud->encoder != exstg->get_faw_index(conf->aud.use_internal))) {
                    std::wstring exe_message;
                    if (!check_audenc_output(aud_stg, exe_message)) {
                        const bool retry_with_default_audenc = false; // ffmpeg_audencを配布していないQSV/NV/VCEEncではここのretryは無効化する
                        const int orig_encoder = cnf_aud->encoder;
                        if (retry_with_default_audenc) {
                            if (default_audenc_cnf_avail
                                && orig_encoder != exstg->s_local.default_audio_encoder_ext
                                && 0 <= exstg->s_local.default_audio_encoder_ext && exstg->s_local.default_audio_encoder_ext < exstg->s_aud_ext_count
                                && muxer_supports_audio_format(pe->muxer_to_be_used, &exstg->s_aud_ext[exstg->s_local.default_audio_encoder_ext])) {
                                cnf_aud->encoder = exstg->s_local.default_audio_encoder_ext;
                            } else if (default_audenc_auo_avail) {
                                cnf_aud->encoder = DEFAULT_AUDIO_ENCODER_EXT;
                            }
                            error_failed_to_run_audio_encoder(
                                exstg->s_aud_ext[orig_encoder].dispname,
                                exe_message.c_str(),
                                (orig_encoder != cnf_aud->encoder) ? exstg->s_aud_ext[cnf_aud->encoder].dispname : nullptr);
                        }
                        // 同じエンコーダあるいはデフォルトエンコーダがうまく取得できな場合は再チェックしても意味がない
                        if (orig_encoder == cnf_aud->encoder) {
                            check = FALSE;
                            break;
                        }
                        // デフォルトエンコーダに戻して再チェック
                        warning_use_default_audio_encoder(exstg->s_aud_ext[cnf_aud->encoder].dispname);
                        continue;
                    }
                    info_use_exe_found(aud_stg->dispname, aud_stg->fullpath);
                }
                // ここまで来たらエンコーダの確認終了なのでbreak
                break;
            }
        }
    }

    //muxer
    switch (pe->muxer_to_be_used) {
    case MUXER_TC2MP4:
        check &= check_muxer_exist(&exstg->s_mux[MUXER_MP4], aviutl_dir, exstg->s_local.get_relative_path, exeFiles); //tc2mp4使用時は追加でmp4boxも必要
        //下へフォールスルー
    case MUXER_MP4:
        check &= check_muxer_exist(&exstg->s_mux[MUXER_MP4], aviutl_dir, exstg->s_local.get_relative_path, exeFiles);
        if (str_has_char(exstg->s_mux[MUXER_MP4_RAW].base_cmd)) {
            check &= check_muxer_exist(&exstg->s_mux[MUXER_MP4_RAW], aviutl_dir, exstg->s_local.get_relative_path, exeFiles);
        }
        //check &= check_muxer_matched_with_ini(exstg->s_mux);
        break;
    case MUXER_MKV:
        check &= check_muxer_exist(&exstg->s_mux[pe->muxer_to_be_used], aviutl_dir, exstg->s_local.get_relative_path, exeFiles);
        break;
    default:
        break;
    }

    return check;
}

void open_log_window(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, int current_pass, int total_pass, bool amp_crf_reenc) {
    wchar_t mes[MAX_PATH_LEN + 512];
    const wchar_t *newLine = (get_current_log_len(current_pass == 1 && !amp_crf_reenc)) ? L"\r\n\r\n" : L""; //必要なら行送り
    static const wchar_t *SEPARATOR = L"------------------------------------------------------------------------------------------------------------------------------";
    const std::wstring savefile_w = char_to_wstring(oip->savefile);
    if (total_pass < 2 || current_pass > total_pass)
        swprintf_s(mes, L"%s%s\r\n[%s]\r\n%s", newLine, SEPARATOR, savefile_w.c_str(), SEPARATOR);
    else
        swprintf_s(mes, L"%s%s\r\n[%s] (%d / %d pass)\r\n%s", newLine, SEPARATOR, savefile_w.c_str(), current_pass, total_pass, SEPARATOR);

    show_log_window(sys_dat->aviutl_dir, sys_dat->exstg->s_local.disable_visual_styles);
    write_log_line(LOG_INFO, mes);

    if (oip->flag & OUTPUT_INFO_FLAG_VIDEO) {
        const double video_length = oip->n * (double)oip->scale / oip->rate;

        const int vid_h = (int)(video_length / 3600);
        const int vid_m = (int)(video_length - vid_h * 3600) / 60;
        const int vid_s = (int)(video_length - vid_h * 3600 - vid_m * 60);
        const int vid_ms = std::min((int)((video_length - (double)(vid_h * 3600 + vid_m * 60 + vid_s)) * 1000.0), 999);

        write_log_auo_line_fmt(LOG_INFO, L"video: %d:%02d:%02d.%03d %d/%d(%.3f) fps",
            vid_h, vid_m, vid_s, vid_ms, oip->rate, oip->scale, oip->rate / (double)oip->scale);
    }

    if (oip->flag & OUTPUT_INFO_FLAG_AUDIO) {
        const double audio_length = oip->audio_n / (double)oip->audio_rate;

        const int aud_h = (int)audio_length / 3600;
        const int aud_m = (int)(audio_length - aud_h * 3600) / 60;
        const int aud_s = (int)(audio_length - aud_h * 3600 - aud_m * 60);
        const int aud_ms = std::min((int)((audio_length - (double)(aud_h * 3600 + aud_m * 60 + aud_s)) * 1000.0), 999);

        write_log_auo_line_fmt(LOG_INFO, L"audio: %d:%02d:%02d.%03d %dch %.1fkHz %d samples",
            aud_h, aud_m, aud_s, aud_ms, oip->audio_ch, oip->audio_rate / 1000.0, oip->audio_n);
    }
}

static void set_tmpdir(PRM_ENC *pe, int tmp_dir_index, const char *savefile, const SYSTEM_DATA *sys_dat) {
    if (tmp_dir_index < TMP_DIR_OUTPUT || TMP_DIR_CUSTOM < tmp_dir_index)
        tmp_dir_index = TMP_DIR_OUTPUT;

    if (tmp_dir_index == TMP_DIR_SYSTEM) {
        //システムの一時フォルダを取得
        if (GetTempPath(_countof(pe->temp_filename), pe->temp_filename) != NULL) {
            PathRemoveBackslash(pe->temp_filename);
            write_log_auo_line_fmt(LOG_INFO, L"%s : %s", g_auo_mes.get(AUO_ENCODE_TMP_FOLDER), char_to_wstring(pe->temp_filename).c_str());
        } else {
            warning_failed_getting_temp_path();
            tmp_dir_index = TMP_DIR_OUTPUT;
        }
    }
    if (tmp_dir_index == TMP_DIR_CUSTOM) {
        //指定されたフォルダ
        if (DirectoryExistsOrCreate(sys_dat->exstg->s_local.custom_tmp_dir)) {
            strcpy_s(pe->temp_filename, GetFullPathFrom(sys_dat->exstg->s_local.custom_tmp_dir, sys_dat->aviutl_dir).c_str());
            PathRemoveBackslash(pe->temp_filename);

            //指定された一時フォルダにファイルを作成できるか確認する
            char defaultExeDir[MAX_PATH_LEN] = { 0 };
            PathCombineLong(defaultExeDir, _countof(defaultExeDir), sys_dat->aviutl_dir, DEFAULT_EXE_DIR);

            char pluginsDir[MAX_PATH_LEN] = { 0 };
            char defaultExeDir2[MAX_PATH_LEN] = { 0 };
            PathCombineLong(pluginsDir, _countof(pluginsDir), sys_dat->aviutl_dir, "plugins");
            PathCombineLong(defaultExeDir2, _countof(defaultExeDir2), pluginsDir, DEFAULT_EXE_DIR);

            const auto auo_check_fileopen_path = find_auo_check_fileopen(defaultExeDir, defaultExeDir2);

            if (check_temp_file_open(pe->temp_filename, auo_check_fileopen_path, true, false)) {
                write_log_auo_line_fmt(LOG_INFO, L"%s : %s", g_auo_mes.get(AUO_ENCODE_TMP_FOLDER), char_to_wstring(pe->temp_filename).c_str());
            } else {
                warning_unable_to_open_tempfile(sys_dat->exstg->s_local.custom_tmp_dir);
                tmp_dir_index = TMP_DIR_OUTPUT;
            }
        } else {
            warning_no_temp_root(sys_dat->exstg->s_local.custom_tmp_dir);
            tmp_dir_index = TMP_DIR_OUTPUT;
        }
    }
    if (tmp_dir_index == TMP_DIR_OUTPUT) {
        //出力フォルダと同じ("\"なし)
        strcpy_s(pe->temp_filename, _countof(pe->temp_filename), savefile);
        PathRemoveFileSpecFixed(pe->temp_filename);
    }
}

static void set_aud_delay_cut(CONF_GUIEX *conf, PRM_ENC *pe, const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat) {
    pe->delay_cut_additional_vframe = 0;
    pe->delay_cut_additional_aframe = 0;
    if (oip->flag & OUTPUT_INFO_FLAG_AUDIO) {
        if (conf->aud.use_internal) {
            conf->aud.in.delay_cut = AUDIO_DELAY_CUT_NONE;
        } else {
            CONF_AUDIO_BASE *cnf_aud = &conf->aud.ext;
            const AUDIO_SETTINGS *aud_stg = &sys_dat->exstg->s_aud_ext[cnf_aud->encoder];
            int audio_delay = aud_stg->mode[cnf_aud->enc_mode].delay;
            if (audio_delay) {
                const double fps = oip->rate / (double)oip->scale;
                const int audio_rate = oip->audio_rate;
                switch (cnf_aud->delay_cut) {
                case AUDIO_DELAY_CUT_DELETE_AUDIO:
                    pe->delay_cut_additional_aframe = -1 * audio_delay;
                    break;
                case AUDIO_DELAY_CUT_ADD_VIDEO:
                    pe->delay_cut_additional_vframe = additional_vframe_for_aud_delay_cut(fps, audio_rate, audio_delay);
                    pe->delay_cut_additional_aframe = additional_silence_for_aud_delay_cut(fps, audio_rate, audio_delay);
                    break;
                case AUDIO_DELAY_CUT_NONE:
                default:
                    break;
                }
            } else {
                cnf_aud->delay_cut = AUDIO_DELAY_CUT_NONE;
            }
        }
    }
}

bool use_auto_npass(const CONF_GUIEX *conf) {
    return false;
}

int get_total_path(const CONF_GUIEX *conf) {
    return 1;
}

void free_enc_prm(PRM_ENC *pe) {
    if (pe->opened_aviutl_files) {
        for (int i = 0; i < pe->n_opened_aviutl_files; i++) {
            if (pe->opened_aviutl_files[i]) {
                free(pe->opened_aviutl_files[i]);
            }
        }
        free(pe->opened_aviutl_files);
        pe->opened_aviutl_files = nullptr;
        pe->n_opened_aviutl_files = 0;
    }
}

void init_enc_prm(const CONF_GUIEX *conf, PRM_ENC *pe, OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat) {
    //初期化
    ZeroMemory(pe, sizeof(PRM_ENC));
    //設定更新
    sys_dat->exstg->load_encode_stg();
    sys_dat->exstg->load_append();
    sys_dat->exstg->load_fn_replace();

    strcpy_s(pe->save_file_name, oip->savefile);
    pe->video_out_type = check_video_ouput(conf, oip);

    // 不明な拡張子だった場合、デフォルトの出力拡張子を付与する
    if (pe->video_out_type == VIDEO_OUTPUT_UNKNOWN) {
        int out_ext_idx = sys_dat->exstg->s_local.default_output_ext;
        if (out_ext_idx < 0 || out_ext_idx >= _countof(OUTPUT_FILE_EXT)) {
            out_ext_idx = 0;
        }
        // 拡張子を付与
        strcat_s(pe->save_file_name, OUTPUT_FILE_EXT[out_ext_idx]);
        // ファイル名が重複していた場合、連番を付与する
        if (PathFileExists(pe->save_file_name)) {
            char tmp[MAX_PATH_LEN];
            for (int i = 0; i < 1000000; i++) {
                char new_ext[32];
                sprintf_s(new_ext, ".%d%s", i, OUTPUT_FILE_EXT[out_ext_idx]);
                strcpy_s(tmp, pe->save_file_name);
                change_ext(tmp, _countof(tmp), new_ext);
                if (!PathFileExists(tmp)) {
                    strcpy_s(pe->save_file_name, tmp);
                    break;
                }
            }
        }
        // オリジナルのsavefileのポインタを保存
        pe->org_save_file_name = oip->savefile;
        // 保存先のファイル名を変更
        oip->savefile = pe->save_file_name;
        // 再度チェック
        pe->video_out_type = check_video_ouput(conf, oip);
    }
}

void set_enc_prm(CONF_GUIEX *conf, PRM_ENC *pe, const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat) {
    pe->video_out_type = check_video_ouput(conf, oip);
    pe->drop_count = 0;
    memcpy(&pe->append, &sys_dat->exstg->s_append, sizeof(FILE_APPENDIX));
    ZeroMemory(&pe->append.aud, sizeof(pe->append.aud));
    create_aviutl_opened_file_list(pe);

    char filename_replace[MAX_PATH_LEN];

    //一時フォルダの決定
    set_tmpdir(pe, conf->oth.temp_dir, oip->savefile, sys_dat);

    //音声一時フォルダの決定
    char *cus_aud_tdir = pe->temp_filename;
    if (!conf->aud.use_internal) {
        if (conf->aud.ext.aud_temp_dir) {
            if (DirectoryExistsOrCreate(sys_dat->exstg->s_local.custom_audio_tmp_dir)) {
                cus_aud_tdir = sys_dat->exstg->s_local.custom_audio_tmp_dir;
                write_log_auo_line_fmt(LOG_INFO, L"%s : %s", g_auo_mes.get(AUO_ENCODE_TMP_FOLDER_AUDIO), char_to_wstring(GetFullPathFrom(cus_aud_tdir, sys_dat->aviutl_dir)).c_str());
            } else {
                warning_no_aud_temp_root(sys_dat->exstg->s_local.custom_audio_tmp_dir);
            }
        }
        strcpy_s(pe->aud_temp_dir, GetFullPathFrom(cus_aud_tdir, sys_dat->aviutl_dir).c_str());
    }

    //ファイル名置換を行い、一時ファイル名を作成
    strcpy_s(filename_replace, _countof(filename_replace), PathFindFileName(oip->savefile));
    sys_dat->exstg->apply_fn_replace(filename_replace, _countof(filename_replace));
    PathCombineLong(pe->temp_filename, _countof(pe->temp_filename), pe->temp_filename, filename_replace);
    //ファイルの上書きを避ける
    avoid_exsisting_tmp_file(pe->temp_filename, _countof(pe->temp_filename));

    pe->muxer_to_be_used = check_muxer_to_be_used(conf, pe, sys_dat, pe->temp_filename, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
    if (pe->muxer_to_be_used >= 0) {
        const MUXER_CMD_EX *muxer_mode = &sys_dat->exstg->s_mux[pe->muxer_to_be_used].ex_cmd[get_mux_excmd_mode(conf, pe)];
        if (str_has_char(muxer_mode->chap_file) && strstr(muxer_mode->chap_file, "chapter.%{pid}.txt")) {
            char move_to[MAX_PATH_LEN] = { 0 };
            char move_from[MAX_PATH_LEN] = { 0 };
            strcpy_s(move_to, muxer_mode->chap_file);
            strcpy_s(move_from, muxer_mode->chap_file);
            replace(move_from, sizeof(move_from), "%{pid}.", "");
            cmd_replace(move_to, sizeof(move_to), pe, sys_dat, conf, oip);
            cmd_replace(move_from, sizeof(move_from), pe, sys_dat, conf, oip);
            if (PathFileExists(move_from)) {
                if (PathFileExists(move_to))
                    remove(move_to);
                if (rename(move_from, move_to))
                    write_log_auo_line(LOG_WARNING, g_auo_mes.get(AUO_ENCODE_ERROR_MOVE_CHAPTER_FILE));
            }
        }
    }

    //FAWチェックとオーディオディレイの修正
    const CONF_AUDIO_BASE *cnf_aud = (conf->aud.use_internal) ? &conf->aud.in : &conf->aud.ext;
    if (cnf_aud->faw_check)
        auo_faw_check(&conf->aud, oip, pe, sys_dat->exstg);
    set_aud_delay_cut(conf, pe, oip, sys_dat);
}

void auto_save_log(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const bool force_save) {
    guiEx_settings ex_stg(true);
    ex_stg.load_log_win();
    if (!force_save && !ex_stg.s_log.auto_save_log)
        return;
    char log_file_path[MAX_PATH_LEN];
    if (AUO_RESULT_SUCCESS != getLogFilePath(log_file_path, _countof(log_file_path), pe, sys_dat, conf, oip))
        warning_no_auto_save_log_dir();
    auto_save_log_file(log_file_path);
    return;
}

void warn_video_length(const OUTPUT_INFO *oip) {
    const double fps = oip->rate / (double)oip->scale;
    if (oip->n <= (int)(fps + 0.5)) {
        warning_video_very_short();
    }
}

int additional_vframe_for_aud_delay_cut(double fps, int audio_rate, int audio_delay) {
    double delay_sec = audio_delay / (double)audio_rate;
    return (int)ceil(delay_sec * fps);
}

int additional_silence_for_aud_delay_cut(double fps, int audio_rate, int audio_delay, int vframe_added) {
    vframe_added = (vframe_added >= 0) ? vframe_added : additional_vframe_for_aud_delay_cut(fps, audio_rate, audio_delay);
    return (int)(vframe_added / (double)fps * audio_rate + 0.5) - audio_delay;
}

BOOL fps_after_afs_is_24fps(const int frame_n, const PRM_ENC *pe) {
    return (pe->drop_count > (frame_n * 0.10));
}

int get_mux_excmd_mode(const CONF_GUIEX *conf, const PRM_ENC *pe) {
    int mode = 0;
    switch (pe->muxer_to_be_used) {
    case MUXER_INTERNAL: mode = conf->mux.internal_mode; break;
    case MUXER_MKV:      mode = conf->mux.mkv_mode; break;
    case MUXER_MP4:
    case MUXER_TC2MP4:
    case MUXER_MP4_RAW:  mode = conf->mux.mp4_mode; break;
    }
    return mode;
}

void get_aud_filename(char *audfile, size_t nSize, const PRM_ENC *pe, int i_aud) {
    PathCombineLong(audfile, nSize, pe->aud_temp_dir, PathFindFileName(pe->temp_filename));
    apply_appendix(audfile, nSize, audfile, pe->append.aud[i_aud]);
}

static void get_muxout_appendix(char *muxout_appendix, size_t nSize, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe) {
    static const char * const MUXOUT_APPENDIX = "_muxout";
    strcpy_s(muxout_appendix, nSize, MUXOUT_APPENDIX);
    const char *ext = (pe->muxer_to_be_used >= 0 && str_has_char(sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext)) ?
        sys_dat->exstg->s_mux[pe->muxer_to_be_used].out_ext : PathFindExtension(pe->temp_filename);
    strcat_s(muxout_appendix, nSize, ext);
}

void get_muxout_filename(char *filename, size_t nSize, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe) {
    char muxout_appendix[MAX_APPENDIX_LEN];
    get_muxout_appendix(muxout_appendix, sizeof(muxout_appendix), sys_dat, pe);
    apply_appendix(filename, nSize, pe->temp_filename, muxout_appendix);
}

//チャプターファイル名とapple形式のチャプターファイル名を同時に作成する
void set_chap_filename(char *chap_file, size_t cf_nSize, char *chap_apple, size_t ca_nSize, const char *chap_base,
                       const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    strcpy_s(chap_file, cf_nSize, chap_base);
    cmd_replace(chap_file, cf_nSize, pe, sys_dat, conf, oip);
    apply_appendix(chap_apple, ca_nSize, chap_file, pe->append.chap_apple);
    sys_dat->exstg->apply_fn_replace(PathFindFileName(chap_apple), ca_nSize - (PathFindFileName(chap_apple) - chap_apple));
}

void insert_num_to_replace_key(char *key, size_t nSize, int num) {
    char tmp[128];
    int key_len = strlen(key);
    sprintf_s(tmp, _countof(tmp), "%d%s", num, &key[key_len-1]);
    key[key_len-1] = '\0';
    strcat_s(key, nSize, tmp);
}

static void set_guiEx_auto_sar(int *sar_x, int *sar_y, int width, int height) {
    if (width > 0 && height > 0 && *sar_x < 0 && *sar_y < 0) {
        int x = -1 * *sar_x * height;
        int y = -1 * *sar_y * width;
        if (abs(y - x) > -16 * *sar_y) {
            int gcd = get_gcd(x, y);
            *sar_x = x / gcd;
            *sar_y = y / gcd;
        } else {
            *sar_x = *sar_y = 1;
        }
    } else if (*sar_x * *sar_y < 0) {
        *sar_x = *sar_y = 0;
    }
}
/*
static void replace_aspect_ratio(char *cmd, size_t nSize, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    const int w = oip->w;
    const int h = oip->h;

    int sar_x = conf->enc.nPAR[0];
    int sar_y = conf->enc.nPAR[1];
    int dar_x = 0;
    int dar_y = 0;
    if (sar_x * sar_y > 0) {
        if (sar_x < 0) {
            dar_x = -1 * sar_x;
            dar_y = -1 * sar_y;
            set_guiEx_auto_sar(&sar_x, &sar_y, w, h);
        } else {
            dar_x = sar_x * w;
            dar_y = sar_y * h;
            const int gcd = get_gcd(dar_x, dar_y);
            dar_x /= gcd;
            dar_y /= gcd;
        }
    }
    if (sar_x * sar_y <= 0)
        sar_x = sar_y = 1;
    if (dar_x * dar_y <= 0)
        dar_x = dar_y = 1;

    char buf[32];
    //%{sar_x} / %{par_x}
    sprintf_s(buf, _countof(buf), "%d", sar_x);
    replace(cmd, nSize, "%{sar_x}", buf);
    replace(cmd, nSize, "%{par_x}", buf);
    //%{sar_x} / %{sar_y}
    sprintf_s(buf, _countof(buf), "%d", sar_y);
    replace(cmd, nSize, "%{sar_y}", buf);
    replace(cmd, nSize, "%{par_y}", buf);
    //%{dar_x}
    sprintf_s(buf, _countof(buf), "%d", dar_x);
    replace(cmd, nSize, "%{dar_x}", buf);
    //%{dar_y}
    sprintf_s(buf, _countof(buf), "%d", dar_y);
    replace(cmd, nSize, "%{dar_y}", buf);
}
*/
void cmd_replace(char *cmd, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    char tmp[MAX_PATH_LEN] = { 0 };
    //置換操作の実行
    //%{vidpath}
    replace(cmd, nSize, "%{vidpath}", pe->temp_filename);
    //%{audpath}
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++) {
        if (str_has_char(pe->append.aud[i_aud])) {
            get_aud_filename(tmp, _countof(tmp), pe, i_aud);
            char aud_key[128] = "%{audpath}";
            if (i_aud)
                insert_num_to_replace_key(aud_key, _countof(aud_key), i_aud);
            replace(cmd, nSize, aud_key, tmp);
        }
    }
    //%{tmpdir}
    strcpy_s(tmp, _countof(tmp), pe->temp_filename);
    PathRemoveFileSpecFixed(tmp);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{tmpdir}", tmp);
    //%{tmpfile}
    strcpy_s(tmp, _countof(tmp), pe->temp_filename);
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{tmpfile}", tmp);
    //%{tmpname}
    strcpy_s(tmp, _countof(tmp), PathFindFileName(pe->temp_filename));
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{tmpname}", tmp);
    //%{savpath}
    replace(cmd, nSize, "%{savpath}", oip->savefile);
    //%{savfile}
    strcpy_s(tmp, _countof(tmp), oip->savefile);
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{savfile}", tmp);
    //%{savname}
    strcpy_s(tmp, _countof(tmp), PathFindFileName(oip->savefile));
    PathRemoveExtension(tmp);
    replace(cmd, nSize, "%{savname}", tmp);
    //%{savdir}
    strcpy_s(tmp, _countof(tmp), oip->savefile);
    PathRemoveFileSpecFixed(tmp);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{savdir}", tmp);
    //%{aviutldir}
    strcpy_s(tmp, _countof(tmp), sys_dat->aviutl_dir);
    PathForceRemoveBackSlash(tmp);
    replace(cmd, nSize, "%{aviutldir}", tmp);
    //%{chpath}
    apply_appendix(tmp, _countof(tmp), oip->savefile, pe->append.chap);
    replace(cmd, nSize, "%{chpath}", tmp);
    //%{tcpath}
    apply_appendix(tmp, _countof(tmp), pe->temp_filename, pe->append.tc);
    replace(cmd, nSize, "%{tcpath}", tmp);
    //%{muxout}
    get_muxout_filename(tmp, _countof(tmp), sys_dat, pe);
    replace(cmd, nSize, "%{muxout}", tmp);
    //%{fps_rate}
    int fps_rate = oip->rate;
    int fps_scale = oip->scale;
    const int fps_gcd = get_gcd(fps_rate, fps_scale);
    fps_rate /= fps_gcd;
    fps_scale /= fps_gcd;
    sprintf_s(tmp, sizeof(tmp), "%d", fps_rate);
    replace(cmd, nSize, "%{fps_rate}", tmp);
    //%{fps_rate_times_4}
    fps_rate *= 4;
    sprintf_s(tmp, sizeof(tmp), "%d", fps_rate);
    replace(cmd, nSize, "%{fps_rate_times_4}", tmp);
    //%{fps_scale}
    sprintf_s(tmp, sizeof(tmp), "%d", fps_scale);
    replace(cmd, nSize, "%{fps_scale}", tmp);
    //アスペクト比
    //replace_aspect_ratio(cmd, nSize, conf, oip);
    //%{pid}
    sprintf_s(tmp, sizeof(tmp), "%d", GetCurrentProcessId());
    replace(cmd, nSize, "%{pid}", tmp);

    if (conf->aud.use_internal) {
        replace(cmd, nSize, "%{audencpath}", "");
    } else {
        const CONF_AUDIO_BASE *cnf_aud = &conf->aud.ext;
        const AUDIO_SETTINGS *aud_stg = &sys_dat->exstg->s_aud_ext[cnf_aud->encoder];
        replace(cmd, nSize, "%{audencpath}", GetFullPathFrom(aud_stg->fullpath, sys_dat->aviutl_dir).c_str());
    }
    replace(cmd, nSize, "%{mp4muxerpath}", GetFullPathFrom(sys_dat->exstg->s_mux[MUXER_MP4].fullpath, sys_dat->aviutl_dir).c_str());
    replace(cmd, nSize, "%{mkvmuxerpath}", GetFullPathFrom(sys_dat->exstg->s_mux[MUXER_MKV].fullpath, sys_dat->aviutl_dir).c_str());
}

static void remove_file(const char *target, const wchar_t *name) {
    if (!DeleteFile(target)) {
        auto errstr = getLastErrorStr(GetLastError());
        write_log_auo_line_fmt(LOG_WARNING, L"%s%s: %s (\"%s\")", name, g_auo_mes.get(AUO_ENCODE_FILE_REMOVE_FAILED), errstr.c_str(), char_to_wstring(target).c_str());
    }
}

static void move_file(const char *move_from, const char *move_to, const wchar_t *name) {
    if (!MoveFile(move_from, move_to)) {
        auto errstr = getLastErrorStr(GetLastError());
        write_log_auo_line_fmt(LOG_WARNING, L"%s%s: %s (\"%s\")", name, g_auo_mes.get(AUO_ENCODE_FILE_MOVE_FAILED), errstr.c_str(), char_to_wstring(move_to).c_str());
    }
}

//一時ファイルの移動・削除を行う
// move_from -> move_to
// temp_filename … 動画ファイルの一時ファイル名。これにappendixをつけてmove_from を作る。
//                  appndixがNULLのときはこれをそのままmove_fromとみなす。
// appendix      … ファイルの後修飾子。NULLも可。
// savefile      … 保存動画ファイル名。これにappendixをつけてmove_to を作る。NULLだと move_to に移動できない。
// ret, erase    … これまでのエラーと一時ファイルを削除するかどうか。エラーがない場合にのみ削除できる
// name          … 一時ファイルの種類の名前
// must_exist    … trueのとき、移動するべきファイルが存在しないとエラーを返し、ファイルが存在しないことを伝える
static BOOL move_temp_file(const char *appendix, const char *temp_filename, const char *savefile, DWORD ret, BOOL erase, const wchar_t *name, BOOL must_exist) {
    char move_from[MAX_PATH_LEN] = { 0 };
    if (appendix)
        apply_appendix(move_from, _countof(move_from), temp_filename, appendix);
    else
        strcpy_s(move_from, _countof(move_from), temp_filename);

    if (!PathFileExists(move_from)) {
        if (must_exist)
            write_log_auo_line_fmt(LOG_WARNING, L"%s%s", name, g_auo_mes.get(AUO_ENCODE_FILE_NOT_FOUND));
        return (must_exist) ? FALSE : TRUE;
    }
    if (ret == AUO_RESULT_SUCCESS && erase) {
        remove_file(move_from, name);
        return TRUE;
    }
    if (savefile == NULL || appendix == NULL)
        return TRUE;
    char move_to[MAX_PATH_LEN] = { 0 };
    apply_appendix(move_to, _countof(move_to), savefile, appendix);
    if (_stricmp(move_from, move_to) != NULL) {
        if (PathFileExists(move_to)) {
            remove_file(move_to, name);
        }
        move_file(move_from, move_to, name);
    }
    return TRUE;
}

AUO_RESULT move_temporary_files(const CONF_GUIEX *conf, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const OUTPUT_INFO *oip, DWORD ret) {
    //動画ファイル
    if (!conf->oth.out_audio_only) {
        if (!move_temp_file(PathFindExtension((pe->muxer_to_be_used >= 0) ? oip->savefile : pe->temp_filename), pe->temp_filename, oip->savefile, ret, FALSE, L"出力", !ret)) {
            ret |= AUO_RESULT_ERROR;
        }
    }
    //動画のみファイル
    if (str_has_char(pe->muxed_vid_filename) && PathFileExists(pe->muxed_vid_filename))
        remove_file(pe->muxed_vid_filename, L"映像一時ファイル");
    //mux後ファイル
    if (pe->muxer_to_be_used >= 0) {
        char muxout_appendix[MAX_APPENDIX_LEN];
        get_muxout_appendix(muxout_appendix, _countof(muxout_appendix), sys_dat, pe);
        move_temp_file(muxout_appendix, pe->temp_filename, oip->savefile, ret, FALSE, g_auo_mes.get(AUO_ENCODE_AFTER_MUX), FALSE);
    }
    //qpファイル
    //move_temp_file(pe->append.qp,   pe->temp_filename, oip->savefile, ret, TRUE, L"qp", FALSE);
    //tcファイル
    BOOL erase_tc = is_afsvfr(conf) && !conf->vid.auo_tcfile_out && pe->muxer_to_be_used != MUXER_DISABLED;
    move_temp_file(pe->append.tc,   pe->temp_filename, oip->savefile, ret, erase_tc, g_auo_mes.get(AUO_ENCODE_TC_FILE), FALSE);
    //チャプターファイル
    if (pe->muxer_to_be_used >= 0) {
        const MUXER_CMD_EX *muxer_mode = &sys_dat->exstg->s_mux[pe->muxer_to_be_used].ex_cmd[get_mux_excmd_mode(conf, pe)];
        bool chapter_auf = strstr(muxer_mode->chap_file, "chapter.%{pid}.txt") != nullptr;
        if (sys_dat->exstg->s_local.auto_del_chap || chapter_auf) {
            char chap_file[MAX_PATH_LEN];
            char chap_apple[MAX_PATH_LEN];
            set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), muxer_mode->chap_file, pe, sys_dat, conf, oip);
            move_temp_file(NULL, chap_file,  NULL, chapter_auf ? AUO_RESULT_SUCCESS : ret, TRUE, g_auo_mes.get(AUO_ENCODE_CHAPTER_FILE), FALSE);
            move_temp_file(NULL, chap_apple, NULL, chapter_auf ? AUO_RESULT_SUCCESS : ret, TRUE, g_auo_mes.get(AUO_ENCODE_CHAPTER_APPLE_FILE), FALSE);
        }
    }
    //音声ファイル(wav)
    if (strcmp(pe->append.aud[0], pe->append.wav)) //「wav出力」ならここでは処理せず下のエンコード後ファイルとして扱う
        move_temp_file(pe->append.wav,  pe->temp_filename, oip->savefile, ret, TRUE, L"wav", FALSE);
    //音声ファイル(エンコード後ファイル)
    char aud_tempfile[MAX_PATH_LEN];
    PathCombineLong(aud_tempfile, _countof(aud_tempfile), pe->aud_temp_dir, PathFindFileName(pe->temp_filename));
    for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
        if (!move_temp_file(pe->append.aud[i_aud], aud_tempfile, oip->savefile, ret, !conf->oth.out_audio_only && pe->muxer_to_be_used != MUXER_DISABLED, g_auo_mes.get(AUO_ENCODE_AUDIO_FILE), conf->oth.out_audio_only))
            ret |= AUO_RESULT_ERROR;
    return ret;
}

DWORD GetExePriority(DWORD set, HANDLE h_aviutl) {
    if (set == AVIUTLSYNC_PRIORITY_CLASS)
        return (h_aviutl) ? GetPriorityClass(h_aviutl) : NORMAL_PRIORITY_CLASS;
    else
        return priority_table[set].value;
}

int check_video_ouput(const char *filename) {
    if (check_ext(filename, ".mp4"))  return VIDEO_OUTPUT_MP4;
    if (check_ext(filename, ".mkv"))  return VIDEO_OUTPUT_MKV;
    return VIDEO_OUTPUT_RAW;
}

int check_video_ouput(const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    if ((oip->flag & OUTPUT_INFO_FLAG_VIDEO) && !conf->oth.out_audio_only) {
        return check_video_ouput(oip->savefile);
    }
    return VIDEO_OUTPUT_DISABLED;
}

BOOL check_output_has_chapter(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, int muxer_to_be_used) {
    const MUXER_CMD_EX *muxer_mode = get_muxer_mode(conf, sys_dat, muxer_to_be_used);
    return (muxer_mode != nullptr) ? str_has_char(muxer_mode->chap_file) : FALSE;
}

int check_muxer_to_be_used(const CONF_GUIEX *conf, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const char *temp_filename, int video_output_type, BOOL audio_output) {
    int muxer_to_be_used = MUXER_DISABLED;
    if (conf->mux.use_internal)
        muxer_to_be_used = MUXER_INTERNAL;
    else if (video_output_type == VIDEO_OUTPUT_MP4 && !conf->mux.disable_mp4ext)
        muxer_to_be_used = MUXER_INTERNAL; // MUXER_MP4;
    else if (video_output_type == VIDEO_OUTPUT_MKV && !conf->mux.disable_mkvext)
        muxer_to_be_used = MUXER_INTERNAL; // MUXER_MKV;

    //muxerが必要ないかどうかチェック
    BOOL no_muxer = TRUE;
    no_muxer &= !conf->mux.use_internal;
    no_muxer &= !audio_output;
    no_muxer &= !conf->vid.afs;
    no_muxer &= video_output_type == check_video_ouput(temp_filename);
    no_muxer &= !check_output_has_chapter(conf, sys_dat, muxer_to_be_used);
    return (no_muxer) ? MUXER_DISABLED : muxer_to_be_used;
}

AUO_RESULT getLogFilePath(char *log_file_path, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    guiEx_settings stg(TRUE); //ログウィンドウの保存先設定は最新のものを使用する
    stg.load_log_win();
    switch (stg.s_log.auto_save_log_mode) {
        case AUTO_SAVE_LOG_CUSTOM:
            char log_file_dir[MAX_PATH_LEN];
            strcpy_s(log_file_path, nSize, stg.s_log.auto_save_log_path);
            cmd_replace(log_file_path, nSize, pe, sys_dat, conf, oip);
            PathGetDirectory(log_file_dir, _countof(log_file_dir), log_file_path);
            if (DirectoryExistsOrCreate(log_file_dir))
                break;
            ret = AUO_RESULT_WARNING;
            //下へフォールスルー
        case AUTO_SAVE_LOG_OUTPUT_DIR:
        default:
            apply_appendix(log_file_path, nSize, oip->savefile, "_log.txt");
            break;
    }
    return ret;
}

double get_duration(const CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat, const PRM_ENC *pe, const OUTPUT_INFO *oip) {
    //Aviutlから再生時間情報を取得
    return ((double)(oip->n + pe->delay_cut_additional_vframe) * (double)oip->scale) / (double)oip->rate;
}

int ReadLogExe(PIPE_SET *pipes, const wchar_t *exename, LOG_CACHE *log_line_cache) {
    DWORD pipe_read = 0;
    if (pipes->stdOut.h_read) {
        if (!PeekNamedPipe(pipes->stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
            return -1;
        if (pipe_read) {
            ReadFile(pipes->stdOut.h_read, pipes->read_buf + pipes->buf_len, sizeof(pipes->read_buf) - pipes->buf_len - 1, &pipe_read, NULL);
            pipes->buf_len += pipe_read;
            pipes->read_buf[pipes->buf_len] = '\0';
            write_log_exe_mes(pipes->read_buf, &pipes->buf_len, exename, log_line_cache);
        }
    }
    return (int)pipe_read;
}

void write_cached_lines(int log_level, const wchar_t *exename, LOG_CACHE *log_line_cache) {
    static const wchar_t *const LOG_LEVEL_STR[] = { L"info", L"warning", L"error" };
    static const wchar_t *MESSAGE_FORMAT = L"%s [%s]: %s";
    wchar_t *buffer = NULL;
    int buffer_len = 0;
    const int log_level_idx = clamp(log_level, LOG_INFO, LOG_ERROR);
    const int additional_length = wcslen(exename) + wcslen(LOG_LEVEL_STR[log_level_idx]) + wcslen(MESSAGE_FORMAT) - wcslen(L"%s") * 3 + 1;
    for (int i = 0; i < log_line_cache->idx; i++) {
        const int required_buffer_len = wcslen(log_line_cache->lines[i]) + additional_length;
        if (buffer_len < required_buffer_len) {
            if (buffer) free(buffer);
            buffer = (wchar_t *)malloc(required_buffer_len * sizeof(buffer[0]));
            buffer_len = required_buffer_len;
        }
        if (buffer) {
            swprintf_s(buffer, buffer_len, MESSAGE_FORMAT, exename, LOG_LEVEL_STR[log_level_idx], log_line_cache->lines[i]);
            write_log_line(log_level, buffer);
        }
    }
    if (buffer) free(buffer);
}


static std::vector<std::wstring> createProcessModuleList() {
    std::vector<std::wstring> moduleList;
    const auto currentPID = GetCurrentProcessId();
    std::unique_ptr<std::remove_pointer<HANDLE>::type, decltype(&CloseHandle)> hProcess(OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, currentPID), CloseHandle);
    HMODULE hMods[1024];
    DWORD cbNeeded = 0;
    if (EnumProcessModules(hProcess.get(), hMods, sizeof(hMods), &cbNeeded)) {
        for (size_t i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            wchar_t moduleName[MAX_PATH_LEN] = { 0 };
            if (GetModuleFileNameExW(hProcess.get(), hMods[i], moduleName, _countof(moduleName))) {
                moduleList.push_back(moduleName);
            }
        }
    }
    return moduleList;
}

bool checkIfModuleLoaded(const wchar_t *moduleName) {
    const auto moduleList = createProcessModuleList();
    for (const auto& modulePath : moduleList) {
        const auto moduleFilename = std::filesystem::path(modulePath).filename().wstring();
        if (_wcsicmp(moduleName, moduleFilename.c_str()) == 0) {
            return true;
        }
    }
    return false;
}

#include "rgy_filesystem.h"
#include "rgy_env.h"

static void create_aviutl_opened_file_list(PRM_ENC *pe) {
    const auto pid_aviutl = GetCurrentProcessId();
    auto list_pid = createChildProcessIDList(pid_aviutl);
    list_pid.push_back(pid_aviutl);

    const auto list_file = createProcessOpenedFileList(list_pid);
    pe->n_opened_aviutl_files = (int)list_file.size();
    if (pe->n_opened_aviutl_files > 0) {
        pe->opened_aviutl_files = (char **)calloc(1, sizeof(char *) * pe->n_opened_aviutl_files);
        for (int i = 0; i < pe->n_opened_aviutl_files; i++) {
            pe->opened_aviutl_files[i] = _strdup(list_file[i].c_str());
        }
    }
}

static bool check_file_is_aviutl_opened_file(const char *filepath, const PRM_ENC *pe) {
    for (int i = 0; i < pe->n_opened_aviutl_files; i++) {
        if (rgy_path_is_same(filepath, pe->opened_aviutl_files[i])) {
            return true;
        }
    }
    return false;
}
