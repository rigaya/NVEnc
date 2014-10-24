#include <Windows.h>
#include <string>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "auo_pipe.h"
#include "auo_util.h"
#include "exe_version.h"

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
		return strprintf("r%d", ver[0]);

	auto str = strprintf("v%d", ver[0]);
	int loop_fin = VER_LENGTH-1;
	for (int i = 1; i < loop_fin; i++) {
		if (ver[i]) {
			str += strprintf(".%d", ver[i]);
		}
	}
	if (ver[loop_fin]) {
		str += strprintf("+%d", ver[loop_fin]);
	}
	return str;
}

int get_exe_version_info(const char *exe_path, int version[4]) {
	#pragma comment(lib, "version.lib")
	int ret = -1;
	BYTE *data_ver_info = nullptr;
	DWORD ver_info_size, dummy = 0;
	if (         0 == (ver_info_size = GetFileVersionInfoSize(exe_path, &dummy))
		|| nullptr == (data_ver_info = (BYTE *)malloc(ver_info_size)))
		return ret;

	if (GetFileVersionInfo(exe_path, 0, ver_info_size, (void*)data_ver_info)) {
		char *buf;
		UINT buf_len;
		typedef struct {
			WORD wLanguage, wCodePage;
		} LANGANDCODEPAGE;
		LANGANDCODEPAGE *ptr_translate = nullptr;
		UINT translate_len = 0;
		if (VerQueryValue(data_ver_info, "\\VarFileInfo\\Translation", (void**)&ptr_translate, &translate_len)) {
			for (DWORD i = 0; i < (translate_len/sizeof(LANGANDCODEPAGE)); i++) {
				char sub_block[256];
				sprintf_s(sub_block, _countof(sub_block), "\\StringFileInfo\\%04x%04x\\FileVersion", ptr_translate[i].wLanguage, ptr_translate[i].wCodePage);
				buf = nullptr;
				buf_len = 0;
				int ver[4] = { 0 };
				if (VerQueryValue(data_ver_info, sub_block, (void **)&buf, &buf_len)
					&& buf
					&& (   4 == sscanf_s(buf, "%d.%d.%d.%d", &ver[0], &ver[1], &ver[2], &ver[3])
						|| 4 == sscanf_s(buf, "%d.%d.%d+%d", &ver[0], &ver[1], &ver[2], &ver[3])
						|| 3 == sscanf_s(buf, "%d.%d.%d",    &ver[0], &ver[1], &ver[2]         )
						|| 3 == sscanf_s(buf, "%d.%d+%d",    &ver[0], &ver[1],          &ver[3])
						|| 2 == sscanf_s(buf, "%d.%d",       &ver[0], &ver[1]                  )
						|| 2 == sscanf_s(buf, "%d+%d",       &ver[0],                   &ver[3])
						|| 1 == sscanf_s(buf, "%d",                                     &ver[0]) ) ) {
					memcpy(version, ver, sizeof(int) * 4);
					ret = 0;
					break;
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
					char sub_block[256];
					sprintf_s(sub_block, _countof(sub_block), "\\StringFileInfo\\%04x%04x\\FileVersion", wLanguageID[j], wCodePageID[i]);
					buf = nullptr;
					buf_len = 0;
					int ver[4] = { 0 };
					if (VerQueryValue(data_ver_info, sub_block, (void **)&buf, &buf_len)
						&& buf
						&& (   4 == sscanf_s(buf, "%d.%d.%d.%d", &ver[0], &ver[1], &ver[2], &ver[3])
							|| 4 == sscanf_s(buf, "%d.%d.%d+%d", &ver[0], &ver[1], &ver[2], &ver[3])
							|| 3 == sscanf_s(buf, "%d.%d.%d",    &ver[0], &ver[1], &ver[2]         )
							|| 3 == sscanf_s(buf, "%d.%d+%d",    &ver[0], &ver[1],          &ver[3])
							|| 2 == sscanf_s(buf, "%d.%d",       &ver[0], &ver[1]                  )
							|| 2 == sscanf_s(buf, "%d+%d",       &ver[0],                   &ver[3])
							|| 1 == sscanf_s(buf, "%d",                                     &ver[0]) ) ) {
						memcpy(version, ver, sizeof(int) * 4);
						ret = 0;
						break;
					}
				}
			}
		}
	}

	free(data_ver_info);
	return ret;
}

int get_exe_version_from_cmd(const char *exe_path, const char *cmd_ver, int version[4]) {
	int ret = -1;
	if (nullptr == version || nullptr == exe_path || !PathFileExists(exe_path))
		return ret;

	memset(version, 0, sizeof(int) * 4);
	const int BUFFER_LEN = 128 * 1024;
	char *buffer = (char *)malloc(BUFFER_LEN);
	if (nullptr == buffer)
		return ret;
	if (nullptr == cmd_ver)
		cmd_ver = "-h";
	if (get_exe_message(exe_path, cmd_ver, buffer, BUFFER_LEN / sizeof(buffer[0]), AUO_PIPE_MUXED) == RP_SUCCESS) {
		char *str;
		int core;
		if (1 == sscanf_s(buffer, "x264 core:%d", &core)) {
			str = buffer + (strlen("x264 core:") + get_intlen(core));
		} else {
			str = buffer;
		}
		for (char *rtr = nullptr; 0 != ret && nullptr != (str = strtok_s(str, "\n", &rtr)); ) {
			char *ptr = str;
			static const char *PREFIX[] = { "fdkaac", "flac", "qaac", "refalac", "version", "revision.", "revision", "rev.", "rev", " r.", " r", " v" };
			for (int i = 0; i < _countof(PREFIX); i++) {
				char *qtr = NULL;
				if (NULL != (qtr = stristr(ptr, PREFIX[i]))) {
					ptr = qtr + strlen(PREFIX[i]);

					char *const ptr_fin = ptr + strlen(ptr);
					while (!isdigit(*ptr) && ptr < ptr_fin)
						ptr++;

					int ver[4] = { 0 };
					if (   4 == sscanf_s(ptr, "%d.%d.%d.%d", &ver[0], &ver[1], &ver[2], &ver[3])
						|| 4 == sscanf_s(ptr, "%d.%d.%d+%d", &ver[0], &ver[1], &ver[2], &ver[3])
						|| 3 == sscanf_s(ptr, "%d.%d.%d",    &ver[0], &ver[1], &ver[2]         )
						|| 3 == sscanf_s(ptr, "%d.%d+%d",    &ver[0], &ver[1],          &ver[3])
						|| 2 == sscanf_s(ptr, "%d.%d",       &ver[0], &ver[1]                  )
						|| 2 == sscanf_s(ptr, "%d+%d",       &ver[0],                   &ver[3])
						|| 1 == sscanf_s(ptr, "%d",          &ver[0]                           )) {
						memcpy(version, ver, sizeof(int) * 4);
						ret = 0;
						break;
					}
				}
			}
			str = nullptr;
		}
	}
	free(buffer);
	return ret;
}

int get_x264_rev(const char *x264fullpath) {
	int ret = -1;
	if (!PathFileExists(x264fullpath))
		return ret;

	int version[4] = { 0 };
	if (-1 == (ret = get_exe_version_info(x264fullpath, version)) || version[2] == 0) {
		if (-1 == get_exe_version_from_cmd(x264fullpath, "--version", version) || version[2] == 0) {
			version[2] = -1;
		}
	}
	return version[2];
}

int get_x265ver_from_txt(const char *txt, int v[4]) {
	int ret = 1;
	memset(v, 0, sizeof(v[0]) * 4);
	if (   4 != sscanf_s(txt, "%d.%d.%d.%d", &v[0], &v[1], &v[2], &v[3])
		&& 4 != sscanf_s(txt, "%d.%d.%d+%d", &v[0], &v[1], &v[2], &v[3])
		&& 3 != sscanf_s(txt, "%d.%d.%d",    &v[0], &v[1], &v[2]       )
		&& 3 != sscanf_s(txt, "%d.%d+%d",    &v[0], &v[1],        &v[3])
		&& 2 != sscanf_s(txt, "%d.%d",       &v[0], &v[1]              )
		&& 2 != sscanf_s(txt, "%d+%d",       &v[0],               &v[3])
		&& 1 != sscanf_s(txt, "%d",          &v[0]                     )) {
		v[0] = v[1] = v[2] = v[3] = 0;
	} else {
		ret = 0;
	}
	return ret;
}

static BOOL qaac_dll_available() {
	//Apple Application Supportのレジストリをチェック
	#pragma comment(lib, "Advapi32.lib")
	static const char *CHECK_KEY = "SOFTWARE\\Apple Inc.\\Apple Application Support";
	HKEY hKey = NULL;
	if (ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE, CHECK_KEY, 0, KEY_QUERY_VALUE, &hKey)) {
		RegCloseKey(hKey);
		return TRUE;
	}
	return FALSE;
}

static BOOL qaac_dll_available(const char *dir) {
	if (nullptr == dir || !str_has_char(dir))
		return FALSE;

	char temp[1024] = { 0 };
	static const char *QAAC_DLL[] = { "CoreAudioToolbox.dll", "CoreFoundation.dll" };
	for (int i = 0; i < _countof(QAAC_DLL); i++) {
		PathCombineLong(temp, _countof(temp), dir, QAAC_DLL[i]);
		if (!PathFileExists(temp))
			return FALSE;
	}
	return TRUE;
}

QTDLL check_if_apple_dll_required_for_qaac(const char *exe_dir, const char *current_fullpath) {
	if (qaac_dll_available())
		return QAAC_APPLEDLL_IN_EXEDIR;
	if (qaac_dll_available(exe_dir))
		return QAAC_APPLEDLL_IN_EXEDIR;
	if (nullptr != current_fullpath && str_has_char(current_fullpath)) {
		char temp[1024] = { 0 };
		strcpy_s(temp, _countof(temp), current_fullpath);
		PathRemoveFileSpecFixed(temp);
		if (qaac_dll_available(temp))
			return QAAC_APPLEDLL_IN_CURRENTDIR;
	}
	return QAAC_APPLEDLL_UNAVAILABLE;
}
