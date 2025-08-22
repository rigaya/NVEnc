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
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>
#include <string>
#include <fstream>

#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_pipe.h"
#include "auo_encode.h"

#include "auo_mes.h"
#include "auo_error.h"
#include "auo_frm.h"

#include "rgy_util.h"
#include "rgy_codepage.h"

static void bat_replace(TCHAR *cmd, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
    TCHAR log_path[MAX_PATH_LEN];
    getLogFilePath(log_path, _countof(log_path), pe, sys_dat, conf, oip);
    replace(cmd, nSize, _T("%{logpath}"), log_path);

    TCHAR chap_file[MAX_PATH_LEN] = { 0 };
    TCHAR chap_apple[MAX_PATH_LEN] = { 0 };
    if (pe->muxer_to_be_used >= 0) {
        const MUXER_SETTINGS *mux_stg = &sys_dat->exstg->s_mux[pe->muxer_to_be_used];
        const MUXER_CMD_EX *muxer_mode = &mux_stg->ex_cmd[(pe->muxer_to_be_used == MUXER_MKV) ? conf->mux.mkv_mode : conf->mux.mp4_mode];
        set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple),
            muxer_mode->chap_file, pe, sys_dat, conf, oip);
    }
    replace(cmd, nSize, _T("%{chapter}"),    chap_file);
    replace(cmd, nSize, _T("%{chap_apple}"), chap_apple);
}

AUO_RESULT run_bat_file(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, DWORD run_bat_mode) {
    if (!(conf->oth.run_bat & run_bat_mode))
        return AUO_RESULT_SUCCESS;

    const TCHAR *batfile = conf->oth.batfiles[get_run_bat_idx(run_bat_mode)];
    if (!PathFileExists(batfile)) {
        warning_no_batfile(batfile); return AUO_RESULT_ERROR;
    }
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    TCHAR bat_tmp[MAX_PATH_LEN];
    apply_appendix(bat_tmp, _countof(bat_tmp), batfile, _T("_tmp.bat"));

    try {
        // ファイルをバイナリモードで読み込み
        std::ifstream input_file(batfile, std::ios::binary);
        if (!input_file.is_open()) {
            ret = AUO_RESULT_ERROR; warning_failed_open_bat_orig();
            return ret;
        }

        // ファイル全体を読み込み
        std::vector<char> file_data((std::istreambuf_iterator<char>(input_file)),
                                   std::istreambuf_iterator<char>());
        input_file.close();

        if (file_data.empty()) {
            ret = AUO_RESULT_ERROR; warning_failed_open_bat_orig();
            return ret;
        }

        // コードページを判定
        uint32_t input_codepage = CP_THREAD_ACP;  // デフォルト
        const void* data_ptr = file_data.data();
        
        // UTF-8 BOMをチェック
        if (file_data.size() >= 3 && 
            memcmp(data_ptr, UTF8_BOM, sizeof(UTF8_BOM)) == 0) {
            input_codepage = CODE_PAGE_UTF8;
        }

        // char型データをTCHAR型に変換
        std::string file_content(file_data.begin(), file_data.end());
        
        // BOMがある場合はスキップ
        const char* content_start = file_content.c_str();
        if (input_codepage == CODE_PAGE_UTF8 && file_content.size() >= 3 &&
            memcmp(content_start, UTF8_BOM, sizeof(UTF8_BOM)) == 0) {
            content_start += sizeof(UTF8_BOM);
        }

        tstring tstr_content = char_to_tstring(content_start, input_codepage);

        // 行ごとに分割して処理
        std::vector<tstring> lines;
        tstring current_line;
        for (size_t i = 0; i < tstr_content.length(); i++) {
            TCHAR ch = tstr_content[i];
            if (ch == _T('\n')) {
                lines.push_back(current_line);
                current_line.clear();
            } else if (ch != _T('\r')) {
                current_line += ch;
            }
        }
        if (!current_line.empty()) {
            lines.push_back(current_line);
        }

        // 一時ファイルを作成（UTF-8 BOM付き）
        std::ofstream output_file(bat_tmp, std::ios::binary);
        if (!output_file.is_open()) {
            ret = AUO_RESULT_ERROR; warning_failed_open_bat_new();
            return ret;
        }

        // UTF-8 BOMを書き込み & chcp 65001を書き込み
        output_file.write(reinterpret_cast<const char*>(UTF8_BOM), sizeof(UTF8_BOM));
        output_file << "\r\nchcp 65001\r\n\r\n" << std::endl;

        const int BAT_REPLACE_MARGIN = 4096;
        int buf_len = BAT_REPLACE_MARGIN * 2;

        // 各行を処理
        std::vector<TCHAR> line_buf(buf_len);
        for (auto& line : lines) {
            // 十分なバッファサイズを確保
            while ((int)line.length() + BAT_REPLACE_MARGIN > buf_len) {
                buf_len *= 2;
            }
            line_buf.resize(buf_len);
            std::fill(line_buf.begin(), line_buf.end(), _T('\0'));
            _tcscpy_s(line_buf.data(), buf_len, line.c_str());
            
            // 行末の空白文字を削除
            deleteCRLFSpace_at_End(line_buf.data());
            
            // rem行をスキップ
            const TCHAR* trimmed = line_buf.data();
            while (*trimmed == _T(' ') || *trimmed == _T('\t')) trimmed++;
            if (_tcsnicmp(trimmed, _T("rem"), 3) == 0 && 
                (trimmed[3] == _T(' ') || trimmed[3] == _T('\t') || trimmed[3] == _T('\0'))) {
                continue; // rem行をスキップ
            }
            
            // 空行もスキップ
            if (_tcslen(trimmed) == 0) {
                continue;
            }
            
            // 置換を実行
            cmd_replace(line_buf.data(), buf_len, pe, sys_dat, conf, oip);
            bat_replace(line_buf.data(), buf_len, pe, sys_dat, conf, oip);
            
            // UTF-8に変換して出力
            std::string output_line = tchar_to_string(line_buf.data(), CODE_PAGE_UTF8);
            output_file << output_line << "\r\n";
        }
        
        output_file.close();
    } catch (...) {
        ret = AUO_RESULT_ERROR;
    }
    //エラーが発生していたら終了
    if (ret)
        return ret;

    //バッチ処理の実行
    PROCESS_INFORMATION pi_bat;
    int rp_ret;
    TCHAR bat_args[MAX_PATH_LEN];
    TCHAR bat_dir[MAX_PATH_LEN];
    _stprintf_s(bat_args, _T("\"%s\""), bat_tmp);
    _stprintf_s(bat_dir, _T("\"%s\""), sys_dat->aviutl_dir);
    set_window_title(g_auo_mes.get(AUO_BAT_RUN), PROGRESSBAR_MARQUEE);
    if (RP_SUCCESS != (rp_ret = RunProcess(bat_args, sys_dat->aviutl_dir, &pi_bat, NULL, NORMAL_PRIORITY_CLASS, FALSE, sys_dat->exstg->s_local.run_bat_minimized))) {
        ret |= AUO_RESULT_ERROR; error_run_process(g_auo_mes.get(AUO_BAT_RUN), rp_ret);
    }
    if (!ret && !(conf->oth.dont_wait_bat_fin & run_bat_mode))
        while (WaitForSingleObject(pi_bat.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            log_process_events();

    set_window_title(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);

    return ret;
}