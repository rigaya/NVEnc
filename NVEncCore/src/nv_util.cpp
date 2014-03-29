//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <stdio.h>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <tchar.h>
#include "cpu_info.h"
#include "gpu_info.h"
#include "nv_util.h"

void adjust_sar(int *sar_w, int *sar_h, int width, int height) {
	int aspect_w = *sar_w;
	int aspect_h = *sar_h;
	//正負チェック
	if (aspect_w * aspect_h <= 0)
		aspect_w = aspect_h = 0;
	else if (aspect_w < 0) {
		//負で与えられている場合はDARでの指定
		//SAR比に変換する
		int dar_x = -1 * aspect_w;
		int dar_y = -1 * aspect_h;
		int x = dar_x * height;
		int y = dar_y * width;
		//多少のづれは容認する
		if (abs(y - x) > 16 * dar_y) {
			//gcd
			int a = x, b = y, c;
			while ((c = a % b) != 0)
				a = b, b = c;
			*sar_w = x / b;
			*sar_h = y / b;
		} else {
			 *sar_w = *sar_h = 1;
		}
	} else {
		//sarも一応gcdをとっておく
		int a = aspect_w, b = aspect_h, c;
		while ((c = a % b) != 0)
			a = b, b = c;
		*sar_w = aspect_w / b;
		*sar_h = aspect_h / b;
	}
}

static const std::map<int, std::pair<int, int>> sar_list = {
	{  0, {  0,  0 } },
	{  1, {  1,  1 } },
	{  2, { 12, 11 } },
	{  3, { 10, 11 } },
	{  4, { 16, 11 } },
	{  5, { 40, 33 } },
	{  6, { 24, 11 } },
	{  7, { 20, 11 } },
	{  8, { 32, 11 } },
	{  9, { 80, 33 } },
	{ 10, { 18, 11 } },
	{ 11, { 15, 11 } },
	{ 12, { 64, 33 } },
	{ 13, {160, 99 } },
	{ 14, {  4,  3 } },
	{ 15, {  3,  2 } },
	{ 16, {  2,  1 } }
};

std::pair<int, int> get_h264_sar(int idx) {
	for (auto i_sar : sar_list) {
		if (i_sar.first == idx)
			return i_sar.second;
	}
	return std::make_pair(0, 0);
}

int get_h264_sar_idx(std::pair<int, int> sar) {

	if (0 != sar.first && 0 != sar.second) {
		const int gcd = nv_get_gcd(sar);
		sar.first  /= gcd;
		sar.second /= gcd;
	}

	for (auto i_sar : sar_list) {
		if (i_sar.second == sar)
			return i_sar.first;
	}
	return -1;
}

/*
int ParseY4MHeader(char *buf, mfxFrameInfo *info) {
	char *p, *q = NULL;
	memset(info, 0, sizeof(mfxFrameInfo));
	for (p = buf; (p = strtok_s(p, " ", &q)) != NULL; ) {
		switch (*p) {
			case 'W':
				{
					char *eptr = NULL;
					int w = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && w)
						info->Width = (mfxU16)w;
				}
				break;
			case 'H':
				{
					char *eptr = NULL;
					int h = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && h)
						info->Height = (mfxU16)h;
				}
				break;
			case 'F':
				{
					int rate = 0, scale = 0;
					if (   (info->FrameRateExtN == 0 || info->FrameRateExtD == 0)
						&& sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
							if (rate && scale) {
								info->FrameRateExtN = rate;
								info->FrameRateExtD = scale;
							}
					}
				}
				break;
			case 'A':
				{
					int sar_x = 0, sar_y = 0;
					if (   (info->AspectRatioW == 0 || info->AspectRatioH == 0)
						&& sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
							if (sar_x && sar_y) {
								info->AspectRatioW = (mfxU16)sar_x;
								info->AspectRatioH = (mfxU16)sar_y;
							}
					}
				}
				break;
			case 'I':
				switch (*(p+1)) {
			case 'b':
				info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
				break;
			case 't':
			case 'm':
				info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
				break;
			default:
				break;
				}
				break;
			case 'C':
				if (   0 != _strnicmp(p+1, "420",      strlen("420"))
					&& 0 != _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
					&& 0 != _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
					&& 0 != _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
					return MFX_PRINT_OPTION_ERR;
				}
				break;
			default:
				break;
		}
		p = NULL;
	}
	return MFX_ERR_NONE;
}
*/
#include <Windows.h>
#include <process.h>

BOOL check_OS_Win8orLater() {
	OSVERSIONINFO osvi = { 0 };
	osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&osvi);
	return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
}

const TCHAR *getOSVersion() {
	const TCHAR *ptr = _T("Unknown");
	OSVERSIONINFO info = { 0 };
	info.dwOSVersionInfoSize = sizeof(info);
	GetVersionEx(&info);
	switch (info.dwPlatformId) {
	case VER_PLATFORM_WIN32_WINDOWS:
		if (4 <= info.dwMajorVersion) {
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows 95"); break;
			case 10: ptr = _T("Windows 98"); break;
			case 90: ptr = _T("Windows Me"); break;
			default: break;
			}
		}
		break;
	case VER_PLATFORM_WIN32_NT:
		switch (info.dwMajorVersion) {
		case 3:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows NT 3"); break;
			case 1:  ptr = _T("Windows NT 3.1"); break;
			case 5:  ptr = _T("Windows NT 3.5"); break;
			case 51: ptr = _T("Windows NT 3.51"); break;
			default: break;
			}
			break;
		case 4:
			if (0 == info.dwMinorVersion)
				ptr = _T("Windows NT 4.0");
			break;
		case 5:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows 2000"); break;
			case 1:  ptr = _T("Windows XP"); break;
			case 2:  ptr = _T("Windows Server 2003"); break;
			default: break;
			}
			break;
		case 6:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows Vista"); break;
			case 1:  ptr = _T("Windows 7"); break;
			case 2:  ptr = _T("Windows 8"); break;
			case 3:  ptr = _T("Windows 8.1"); break;
			default:
				if (4 <= info.dwMinorVersion) {
					ptr = _T("Later than Windows 8.1");
				}
				break;
			}
			break;
		default:
			if (7 <= info.dwPlatformId) {
				ptr = _T("Later than Windows 8.1");
			}
			break;
		}
		break;
	default:
		break;
	}
	return ptr;
}

BOOL is_64bit_os() {
	SYSTEM_INFO sinfo = { 0 };
	GetNativeSystemInfo(&sinfo);
	return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
}

UINT64 getPhysicalRamSize(UINT64 *ramUsed) {
	MEMORYSTATUSEX msex ={ 0 };
	msex.dwLength = sizeof(msex);
	GlobalMemoryStatusEx(&msex);
	if (NULL != ramUsed)
		*ramUsed = msex.ullTotalPhys - msex.ullAvailPhys;
	return msex.ullTotalPhys;
}

void getEnviromentInfo(TCHAR *buf, unsigned int buffer_size) {
	ZeroMemory(buf, sizeof(buf[0]) * buffer_size);

	TCHAR cpu_info[1024] = { 0 };
	getCPUInfo(cpu_info, _countof(cpu_info));

	TCHAR gpu_info[1024] = { 0 };
	getGPUInfo("NVIDIA", gpu_info, _countof(gpu_info));

	UINT64 UsedRamSize = 0;
	UINT64 totalRamsize = getPhysicalRamSize(&UsedRamSize);

	auto add_tchar_to_buf = [buf, buffer_size](const TCHAR *fmt, ...) {
		unsigned int buf_length = (unsigned int)_tcslen(buf);
		va_list args = { 0 };
		va_start(args, fmt);
		_vstprintf_s(buf + buf_length, buffer_size - buf_length, fmt, args);
		va_end(args);
	};

	add_tchar_to_buf(_T("Environment Info\n"));
	add_tchar_to_buf(_T("OS : %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
	add_tchar_to_buf(_T("CPU: %s\n"), cpu_info);
	add_tchar_to_buf(_T("GPU: %s\n"), gpu_info);
	add_tchar_to_buf(_T("RAM: Total %d MB / Used %d MB\n"), (UINT)(totalRamsize >> 20), (UINT)(UsedRamSize >> 20));
}
