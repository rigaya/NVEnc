//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_pipe.h"
#include "auo_encode.h"

#include "auo_error.h"
#include "auo_frm.h"

static void bat_replace(char *cmd, size_t nSize, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
	char log_path[MAX_PATH_LEN];
	getLogFilePath(log_path, _countof(log_path), pe, sys_dat, conf, oip);
	replace(cmd, nSize, "%{logpath}", log_path);

	char chap_file[MAX_PATH_LEN] = { 0 };
	char chap_apple[MAX_PATH_LEN] = { 0 };
	if (pe->muxer_to_be_used >= 0) {
		const MUXER_SETTINGS *mux_stg = &sys_dat->exstg->s_mux[pe->muxer_to_be_used];
		const MUXER_CMD_EX *muxer_mode = &mux_stg->ex_cmd[(pe->muxer_to_be_used == MUXER_MKV) ? conf->mux.mkv_mode : conf->mux.mp4_mode];
		set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), 
			muxer_mode->chap_file, pe, sys_dat, conf, oip);
	}
	replace(cmd, nSize, "%{chapter}",    chap_file);
	replace(cmd, nSize, "%{chap_apple}", chap_apple);
}

AUO_RESULT run_bat_file(const CONF_GUIEX *conf, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, DWORD run_bat_mode) {
	if (!(conf->oth.run_bat & run_bat_mode))
		return AUO_RESULT_SUCCESS;

	const char *batfile = (run_bat_mode & RUN_BAT_BEFORE) ? conf->oth.batfile_before : conf->oth.batfile_after;
	if (!PathFileExists(batfile)) {
		warning_no_batfile(batfile); return AUO_RESULT_ERROR;
	}
	AUO_RESULT ret = AUO_RESULT_SUCCESS;
	char bat_tmp[MAX_PATH_LEN];
	apply_appendix(bat_tmp, _countof(bat_tmp), batfile, "_tmp.bat");

	const int BAT_REPLACE_MARGIN = 4096;
	int buf_len = BAT_REPLACE_MARGIN * 2;
	FILE *fp_orig = NULL, *fp_tmp = NULL;
	char *line_buf = NULL;
	if        (fopen_s(&fp_orig, batfile, "r" ) != NULL) {
		ret = AUO_RESULT_ERROR; warning_failed_open_bat_orig();
	} else if (fopen_s(&fp_tmp,  bat_tmp, "wb") != NULL) {
		ret = AUO_RESULT_ERROR; warning_failed_open_bat_new();
	} else if ((line_buf = (char *)calloc(buf_len, sizeof(line_buf[0]))) == NULL) {
		ret = AUO_RESULT_ERROR; warning_malloc_batfile_tmp();
	} else {
		//一行づつ処理
		while (fgets(line_buf + strlen(line_buf), buf_len - (int)strlen(line_buf), fp_orig) != NULL) {
			//十分なバッファがなければ拡張する
			BOOL buf_not_enough = ((int)strlen(line_buf) + BAT_REPLACE_MARGIN > buf_len);
			if (buf_not_enough) {
				buf_len *= 2;
				if (NULL == (line_buf = (char *)realloc(line_buf, buf_len * sizeof(line_buf[0])))) {
					ret = AUO_RESULT_ERROR; warning_malloc_batfile_tmp(); break;
				}
			}
			char *ptr = NULL;
			//置換を実行し出力(最終行 || 行末まで読めてる)
			if (!buf_not_enough || (ptr = strrchr(line_buf, '\n')) != NULL) {
				if (ptr)
					*ptr = '\0';
				deleteCRLFSpace_at_End(line_buf);
				cmd_replace(line_buf, buf_len, pe, sys_dat, conf, oip);
				bat_replace(line_buf, buf_len, pe, sys_dat, conf, oip);
				fprintf(fp_tmp, "%s\r\n", line_buf);
				*line_buf = '\0'; //line_bufの長さを0に
			}
		}
		free(line_buf); line_buf = NULL;
	}

	if (fp_orig) { fclose(fp_orig); fp_orig = NULL; }
	if (fp_tmp)  { fclose(fp_tmp);  fp_tmp  = NULL; }
	//エラーが発生していたら終了
	if (ret)
		return ret;

	//バッチ処理の実行
	PROCESS_INFORMATION pi_bat;
	int rp_ret;
	char bat_args[MAX_PATH_LEN];
	char bat_dir[MAX_PATH_LEN];
	sprintf_s(bat_args, _countof(bat_args), "\"%s\"", bat_tmp);
	sprintf_s(bat_dir, _countof(bat_dir), "\"%s\"", sys_dat->aviutl_dir);
	set_window_title("バッチファイル処理", PROGRESSBAR_MARQUEE);
	if (RP_SUCCESS != (rp_ret = RunProcess(bat_args, sys_dat->aviutl_dir, &pi_bat, NULL, NORMAL_PRIORITY_CLASS, FALSE, sys_dat->exstg->s_local.run_bat_minimized))) {
		ret |= AUO_RESULT_ERROR; error_run_process("バッチファイル処理", rp_ret);
	}
	if (!ret && !(conf->oth.dont_wait_bat_fin & run_bat_mode))
		while (WaitForSingleObject(pi_bat.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
			log_process_events();

	set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

	return ret;
}