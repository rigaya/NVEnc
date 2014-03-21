//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <Math.h>
#include <stdio.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "auo.h"
#include "auo_version.h"
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_pipe.h"

#include "auo_frm.h"
#include "auo_encode.h"
#include "auo_error.h"

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
 
	int sar_x = conf->qsv.nPAR[0];
	int sar_y = conf->qsv.nPAR[1];
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
#ifdef MSDK_SAMPLE_VERSION
	if (conf->qsv.vpp.nDeinterlace == MFX_DEINTERLACE_IT)
		fps_rate = (fps_rate * 4) / 5;
#endif
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

	char fullpath[MAX_PATH_LEN];
	replace(cmd, nSize, "%{audencpath}",   GetFullPath(sys_dat->exstg->s_aud[conf->aud.encoder].fullpath, fullpath, _countof(fullpath)));
	replace(cmd, nSize, "%{mp4muxerpath}", GetFullPath(sys_dat->exstg->s_mux[MUXER_MP4].fullpath,         fullpath, _countof(fullpath)));
	replace(cmd, nSize, "%{mkvmuxerpath}", GetFullPath(sys_dat->exstg->s_mux[MUXER_MKV].fullpath,         fullpath, _countof(fullpath)));
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
static BOOL move_temp_file(const char *appendix, const char *temp_filename, const char *savefile, DWORD ret, BOOL erase, const char *name, BOOL must_exist) {
	char move_from[MAX_PATH_LEN] = { 0 };
	if (appendix)
		apply_appendix(move_from, _countof(move_from), temp_filename, appendix);
	else
		strcpy_s(move_from, _countof(move_from), temp_filename);

	if (!PathFileExists(move_from)) {
		if (must_exist)
			write_log_auo_line_fmt(LOG_WARNING, "%sファイルが見つかりませんでした。", name);
		return (must_exist) ? FALSE : TRUE;
	}
	if (ret == AUO_RESULT_SUCCESS && erase) {
		remove(move_from);
		return TRUE;
	}
	if (savefile == NULL || appendix == NULL)
		return TRUE;
	char move_to[MAX_PATH_LEN] = { 0 };
	apply_appendix(move_to, _countof(move_to), savefile, appendix);
	if (_stricmp(move_from, move_to) != NULL) {
		if (PathFileExists(move_to))
			remove(move_to);
		if (rename(move_from, move_to))
			write_log_auo_line_fmt(LOG_WARNING, "%sファイルの移動に失敗しました。", name);
	}
	return TRUE;
}

AUO_RESULT move_temporary_files(const CONF_GUIEX *conf, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const OUTPUT_INFO *oip, DWORD ret) {
	//動画ファイル
	if (!conf->oth.out_audio_only)
		if (!move_temp_file(PathFindExtension((pe->muxer_to_be_used >= 0) ? oip->savefile : pe->temp_filename), pe->temp_filename, oip->savefile, ret, FALSE, "出力", !ret))
			ret |= AUO_RESULT_ERROR;
	//動画のみファイル
	if (str_has_char(pe->muxed_vid_filename) && PathFileExists(pe->muxed_vid_filename))
		remove(pe->muxed_vid_filename);
	//mux後ファイル
	if (pe->muxer_to_be_used >= 0) {
		char muxout_appendix[MAX_APPENDIX_LEN];
		get_muxout_appendix(muxout_appendix, _countof(muxout_appendix), sys_dat, pe);
		move_temp_file(muxout_appendix, pe->temp_filename, oip->savefile, ret, FALSE, "mux後ファイル", FALSE);
	}
	//qpファイル
	move_temp_file(pe->append.qp,   pe->temp_filename, oip->savefile, ret, TRUE, "qp", FALSE);
	//tcファイル
	BOOL erase_tc = conf->vid.afs && !conf->vid.auo_tcfile_out && pe->muxer_to_be_used != MUXER_DISABLED;
	move_temp_file(pe->append.tc,   pe->temp_filename, oip->savefile, ret, erase_tc, "タイムコード", FALSE);
	//チャプターファイル
	if (pe->muxer_to_be_used >= 0 && sys_dat->exstg->s_local.auto_del_chap) {
		char chap_file[MAX_PATH_LEN];
		char chap_apple[MAX_PATH_LEN];
		const MUXER_CMD_EX *muxer_mode = &sys_dat->exstg->s_mux[pe->muxer_to_be_used].ex_cmd[(pe->muxer_to_be_used == MUXER_MKV) ? conf->mux.mkv_mode : conf->mux.mp4_mode];
		set_chap_filename(chap_file, _countof(chap_file), chap_apple, _countof(chap_apple), muxer_mode->chap_file, pe, sys_dat, conf, oip);
		move_temp_file(NULL, chap_file,  NULL, ret, TRUE, "チャプター",        FALSE);
		move_temp_file(NULL, chap_apple, NULL, ret, TRUE, "チャプター(Apple)", FALSE);
	}
	//音声ファイル(wav)
	if (strcmp(pe->append.aud[0], pe->append.wav)) //「wav出力」ならここでは処理せず下のエンコード後ファイルとして扱う
		move_temp_file(pe->append.wav,  pe->temp_filename, oip->savefile, ret, TRUE, "wav", FALSE);
	//音声ファイル(エンコード後ファイル)
	char aud_tempfile[MAX_PATH_LEN];
	PathCombineLong(aud_tempfile, _countof(aud_tempfile), pe->aud_temp_dir, PathFindFileName(pe->temp_filename));
	for (int i_aud = 0; i_aud < pe->aud_count; i_aud++)
		if (!move_temp_file(pe->append.aud[i_aud], aud_tempfile, oip->savefile, ret, !conf->oth.out_audio_only && pe->muxer_to_be_used != MUXER_DISABLED, "音声", conf->oth.out_audio_only))
			ret |= AUO_RESULT_ERROR;
	return ret;
}

DWORD GetExePriority(DWORD set, HANDLE h_aviutl) {
	if (set == AVIUTLSYNC_PRIORITY_CLASS)
		return (h_aviutl) ? GetPriorityClass(h_aviutl) : NORMAL_PRIORITY_CLASS;
	else
		return priority_table[set].value;
}

int check_video_ouput(const CONF_GUIEX *conf, const OUTPUT_INFO *oip) {
	if ((oip->flag & OUTPUT_INFO_FLAG_VIDEO) && !conf->oth.out_audio_only) {
		if (check_ext(oip->savefile, ".mp4"))  return VIDEO_OUTPUT_MP4;
		if (check_ext(oip->savefile, ".mkv"))  return VIDEO_OUTPUT_MKV;
		return VIDEO_OUTPUT_RAW;
	}
	return VIDEO_OUTPUT_DISABLED;
}
#pragma warning( push )
#pragma warning( disable: 4100 )
int check_muxer_to_be_used(const CONF_GUIEX *conf, int video_output_type, BOOL audio_output) {
	if (video_output_type == VIDEO_OUTPUT_MP4 && !conf->mux.disable_mp4ext)
		return (conf->vid.afs) ? MUXER_TC2MP4 : MUXER_MP4;
	else if (video_output_type == VIDEO_OUTPUT_MKV && !conf->mux.disable_mkvext)
		return MUXER_MKV;
	else if (video_output_type == VIDEO_OUTPUT_MPEG2 && !conf->mux.disable_mpgext)
		return MUXER_MPG;
	else
		return MUXER_DISABLED;
}
#pragma warning( pop )
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

double get_duration(const OUTPUT_INFO *oip, const PRM_ENC *pe) {
	//Aviutlから再生時間情報を取得
	return ((double)(oip->n + pe->delay_cut_additional_vframe) * (double)oip->scale) / (double)oip->rate;
}

int ReadLogExe(PIPE_SET *pipes, const char *exename, LOG_CACHE *log_line_cache) {
	DWORD pipe_read = 0;
	if (!PeekNamedPipe(pipes->stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
		return -1;
	if (pipe_read) {
		ReadFile(pipes->stdOut.h_read, pipes->read_buf + pipes->buf_len, sizeof(pipes->read_buf) - pipes->buf_len - 1, &pipe_read, NULL);
		pipes->buf_len += pipe_read;
		pipes->read_buf[pipes->buf_len] = '\0';
		write_log_exe_mes(pipes->read_buf, &pipes->buf_len, exename, log_line_cache);
	}
	return (int)pipe_read;
}

void write_cached_lines(int log_level, const char *exename, LOG_CACHE *log_line_cache) {
	static const char *const LOG_LEVEL_STR[] = { "info", "warning", "error" };
	static const char *MESSAGE_FORMAT = "%s [%s]: %s";
	char *buffer = NULL;
	int buffer_len = 0;
	log_level = clamp(log_level, LOG_INFO, LOG_ERROR);
	const int additional_length = strlen(exename) + strlen(LOG_LEVEL_STR[log_level]) + strlen(MESSAGE_FORMAT) - strlen("%s") * 3 + 1;
	for (int i = 0; i < log_line_cache->idx; i++) {
		const int required_buffer_len = strlen(log_line_cache->lines[i]) + additional_length;
		if (buffer_len < required_buffer_len) {
			if (buffer) free(buffer);
			buffer = (char *)malloc(required_buffer_len * sizeof(buffer[0]));
			buffer_len = required_buffer_len;
		}
		if (buffer) {
			sprintf_s(buffer, buffer_len, MESSAGE_FORMAT, exename, LOG_LEVEL_STR[log_level], log_line_cache->lines[i]);
			write_log_line(log_level, buffer);
		}
	}
	if (buffer) free(buffer);
}
