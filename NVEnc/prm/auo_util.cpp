//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <algorithm>
#include <tlhelp32.h>
#include <vector>

#include "auo_util.h"
#include "auo_version.h"

//ボム文字かどうか、コードページの判定
DWORD check_bom(const void* chr) {
	if (chr == NULL) return CODE_PAGE_UNSET;
	if (memcmp(chr, UTF16_LE_BOM, sizeof(UTF16_LE_BOM)) == NULL) return CODE_PAGE_UTF16_LE;
	if (memcmp(chr, UTF16_BE_BOM, sizeof(UTF16_BE_BOM)) == NULL) return CODE_PAGE_UTF16_BE;
	if (memcmp(chr, UTF8_BOM,     sizeof(UTF8_BOM))     == NULL) return CODE_PAGE_UTF8;
	return CODE_PAGE_UNSET;
}

static BOOL isJis(const void *str, DWORD size_in_byte) {
	static const BYTE ESCAPE[][7] = {
		//先頭に比較すべきバイト数
		{ 3, 0x1B, 0x28, 0x42, 0x00, 0x00, 0x00 },
		{ 3, 0x1B, 0x28, 0x4A, 0x00, 0x00, 0x00 },
		{ 3, 0x1B, 0x28, 0x49, 0x00, 0x00, 0x00 },
		{ 3, 0x1B, 0x24, 0x40, 0x00, 0x00, 0x00 },
		{ 3, 0x1B, 0x24, 0x42, 0x00, 0x00, 0x00 },
		{ 6, 0x1B, 0x26, 0x40, 0x1B, 0x24, 0x42 },
		{ 4, 0x1B, 0x24, 0x28, 0x44, 0x00, 0x00 },
		{ 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } //終了
	};
	const BYTE * const str_fin = (const BYTE *)str + size_in_byte;
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin; chr++) {
		if (*chr > 0x7F)
			return FALSE;
		for (int i = 0; ESCAPE[i][0]; i++) {
			if (str_fin - chr > ESCAPE[i][0] && 
				memcmp(chr, &ESCAPE[i][1], ESCAPE[i][0]) == NULL)
				return TRUE;
		}
	}
	return FALSE;
}

static DWORD isUTF16(const void *str, DWORD size_in_byte) {
	const BYTE * const str_fin = (const BYTE *)str + size_in_byte;
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin; chr++) {
		if (chr[0] == 0x00 && str_fin - chr > 1 && chr[1] <= 0x7F)
			return ((chr - (const BYTE *)str) % 2 == 1) ? CODE_PAGE_UTF16_LE : CODE_PAGE_UTF16_BE;
	}
	return CODE_PAGE_UNSET;
}

static BOOL isASCII(const void *str, DWORD size_in_byte) {
	const BYTE * const str_fin = (const BYTE *)str + size_in_byte;
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin; chr++) {
		if (*chr == 0x1B || *chr >= 0x80)
			return FALSE;
	}
	return TRUE;
}

static int jpn_check(const void *str, DWORD size_in_byte) {
	int score_sjis = 0;
	int score_euc = 0;
	int score_utf8 = 0;
	const BYTE * const str_fin = (const BYTE *)str + size_in_byte;
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin - 1; chr++) {
		if ((0x81 <= chr[0] && chr[0] <= 0x9F) ||
			(0xE0 <= chr[0] && chr[0] <= 0xFC) ||
			(0x40 <= chr[1] && chr[1] <= 0x7E) ||
			(0x80 <= chr[1] && chr[1] <= 0xFC)) {
				score_sjis += 2; chr++;
		}
	}
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin - 1; chr++) {
		if ((0xC0 <= chr[0] && chr[0] <= 0xDF) &&
			(0x80 <= chr[1] && chr[1] <= 0xBF)) {
				score_utf8 += 2; chr++;
		} else if (
			str_fin - chr > 2 &&
			(0xE0 <= chr[0] && chr[0] <= 0xEF) &&
			(0x80 <= chr[1] && chr[1] <= 0xBF) &&
			(0x80 <= chr[2] && chr[2] <= 0xBF)) {
				score_utf8 += 3; chr++;
		}
	}
	for (const BYTE *chr = (const BYTE *)str; chr < str_fin - 1; chr++) {
		if (((0xA1 <= chr[0] && chr[0] <= 0xFE) && (0xA1 <= chr[1] && chr[1] <= 0xFE)) ||
			(chr[0] == 0x8E                     && (0xA1 <= chr[1] && chr[1] <= 0xDF))) {
				score_euc += 2; chr++;
		} else if (
			str_fin - chr > 2 &&
			chr[0] == 0x8F && 
			(0xA1 <= chr[1] && chr[1] <= 0xFE) && 
			(0xA1 <= chr[2] && chr[2] <= 0xFE)) {
				score_euc += 3; chr += 2;
		}
	}
	if (score_sjis > score_euc && score_sjis > score_utf8)
		return CODE_PAGE_SJIS;
	if (score_utf8 > score_euc && score_utf8 > score_sjis)
		return CODE_PAGE_UTF8;
	if (score_euc > score_sjis && score_euc > score_utf8)
		return CODE_PAGE_EUC_JP;
	return CODE_PAGE_UNSET;
}

DWORD get_code_page(const void *str, DWORD size_in_byte) {
	int ret = CODE_PAGE_UNSET;
	if ((ret = check_bom(str)) != CODE_PAGE_UNSET)
		return ret;

	if (isJis(str, size_in_byte))
		return CODE_PAGE_JIS;

	if ((ret = isUTF16(str, size_in_byte)) != CODE_PAGE_UNSET)
		return ret;

	if (isASCII(str, size_in_byte))
		return CODE_PAGE_US_ASCII;

	return jpn_check(str, size_in_byte);
}

BOOL fix_ImulL_WesternEurope(UINT *code_page) {
	//IMultiLanguage2 の DetectInputCodepage はよく西ヨーロッパ言語と誤判定しやがる
	if (*code_page == CODE_PAGE_WEST_EUROPE)
		*code_page = CODE_PAGE_SJIS;
	return TRUE;
}

static inline BOOL is_space_or_crlf(int c) {
	return (c == ' ' || c == '\r' || c == '\n');
}
BOOL del_arg(char *cmd, char *target_arg, int del_arg_delta) {
	char *p_start, *ptr;
	char * const cmd_fin = cmd + strlen(cmd);
	del_arg_delta = clamp(del_arg_delta, -1, 1);
	//指定された文字列を検索
	if ((p_start = strstr(cmd, target_arg)) == NULL)
		return FALSE;
	//指定された文字列の含まれる部分の先頭を検索
	for ( ; cmd < p_start; p_start--)
		if (is_space_or_crlf(*(p_start-1)))
			break;
	//指定された文字列の含まれる部分の最後尾を検索
	ptr = p_start;
	{
		BOOL dQB = FALSE;
		while (is_space_or_crlf(*ptr))
			ptr++;

		while (cmd < ptr && ptr < cmd_fin) {
			if (*ptr == '"') dQB = !dQB;
			if (!dQB && is_space_or_crlf(*ptr))
				break;
			ptr++;
		}
	}
	if (del_arg_delta < 0)
		std::swap(p_start, ptr);

	//次の値を検索
	if (del_arg_delta) {
		do {
			ptr += del_arg_delta;
		} while (is_space_or_crlf(*ptr));

		BOOL dQB = FALSE;
		while (cmd < ptr && ptr < cmd_fin) {
			if (*ptr == '"') dQB = !dQB;
			if (!dQB && is_space_or_crlf(*ptr))
				break;
			ptr += del_arg_delta;
		}
	}
	//文字列の移動
	if (del_arg_delta < 0)
		std::swap(p_start, ptr);
		
	memmove(p_start, ptr, (cmd_fin - ptr + 1) * sizeof(cmd[0]));
	return TRUE;
}

static const int ThreadQuerySetWin32StartAddress = 9;
typedef int (WINAPI *typeNtQueryInformationThread)(HANDLE, int, PVOID, ULONG, PULONG);

static void *GetThreadBeginAddress(DWORD TargetProcessId) {
	HMODULE hNtDll = NULL;
	typeNtQueryInformationThread NtQueryInformationThread = NULL;
	HANDLE hThread = NULL;
	ULONG length = 0;
	void *BeginAddress = NULL;

	if (   NULL != (hNtDll = LoadLibrary("ntdll.dll"))
		&& NULL != (NtQueryInformationThread = (typeNtQueryInformationThread)GetProcAddress(hNtDll, "NtQueryInformationThread"))
		&& NULL != (hThread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, TargetProcessId)) ) {
		NtQueryInformationThread(hThread, ThreadQuerySetWin32StartAddress, &BeginAddress, sizeof(BeginAddress), &length );
	}
	if (hNtDll)
		FreeLibrary(hNtDll);
	if (hThread)
		CloseHandle(hThread);
	return BeginAddress;
}

static inline std::vector<DWORD> GetThreadList(DWORD TargetProcessId) {
	std::vector<DWORD> ThreadList;
	HANDLE hSnapshot;

	if (INVALID_HANDLE_VALUE != (hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0x00))) {
		THREADENTRY32 te32 = { 0 };
		te32.dwSize = sizeof(THREADENTRY32);

		if (Thread32First(hSnapshot, &te32)) {
			do {
				if (te32.th32OwnerProcessID == TargetProcessId)
					ThreadList.push_back(te32.th32ThreadID);
			} while (Thread32Next(hSnapshot, &te32));
		}
		CloseHandle(hSnapshot);
	}
	return ThreadList;
}

static inline std::vector<MODULEENTRY32> GetModuleList(DWORD TargetProcessId) {
	std::vector<MODULEENTRY32> ModuleList;
	HANDLE hSnapshot;

	if (INVALID_HANDLE_VALUE != (hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, TargetProcessId))) {
		MODULEENTRY32 me32 = { 0 };
		me32.dwSize = sizeof(MODULEENTRY32);

		if (Module32First(hSnapshot, &me32)) {
			do {
				ModuleList.push_back(me32);
			} while (Module32Next(hSnapshot, &me32));
		}
		CloseHandle(hSnapshot);
	}
	return ModuleList;
}

static BOOL SetThreadPriorityFromThreadId(DWORD TargetThreadId, int ThreadPriority) {
	HANDLE hThread = OpenThread(THREAD_SET_INFORMATION, FALSE, TargetThreadId);
	if (hThread == NULL)
		return FALSE;
	BOOL ret = SetThreadPriority(hThread, ThreadPriority);
	CloseHandle(hThread);
	return ret;
}

BOOL SetThreadPriorityForModule(DWORD TargetProcessId, const char *TargetModule, int ThreadPriority) {
	BOOL ret = TRUE;
	std::vector<DWORD> thread_list = GetThreadList(TargetProcessId);
	std::vector<MODULEENTRY32> module_list = GetModuleList(TargetProcessId);
	foreach(std::vector<DWORD>, it_tid, &thread_list) {
		void *thread_address = GetThreadBeginAddress(*it_tid);
		if (!thread_address) {
			ret = FALSE;
		} else {
			foreach(std::vector<MODULEENTRY32>, it_module, &module_list) {
				if (   check_range(thread_address, it_module->modBaseAddr, it_module->modBaseAddr + it_module->modBaseSize - 1)
					&& (NULL == TargetModule || NULL == _strnicmp(TargetModule, it_module->szModule, strlen(TargetModule)))) {
					ret &= !!SetThreadPriorityFromThreadId(*it_tid, ThreadPriority);
					break;
				}
			}
		}
	}
	return ret;
}

static BOOL SetThreadAffinityFromThreadId(DWORD TargetThreadId, DWORD_PTR ThreadAffinityMask) {
	HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, TargetThreadId);
	if (hThread == NULL)
		return FALSE;
	DWORD_PTR ret = SetThreadAffinityMask(hThread, ThreadAffinityMask);
	CloseHandle(hThread);
	return (ret != 0);
}

BOOL SetThreadAffinityForModule(DWORD TargetProcessId, const char *TargetModule, DWORD_PTR ThreadAffinityMask) {
	BOOL ret = TRUE;
	std::vector<DWORD> thread_list = GetThreadList(TargetProcessId);
	std::vector<MODULEENTRY32> module_list = GetModuleList(TargetProcessId);
	foreach(std::vector<DWORD>, it_tid, &thread_list) {
		void *thread_address = GetThreadBeginAddress(*it_tid);
		if (!thread_address) {
			ret = FALSE;
		} else {
			foreach(std::vector<MODULEENTRY32>, it_module, &module_list) {
				if (   check_range(thread_address, it_module->modBaseAddr, it_module->modBaseAddr + it_module->modBaseSize - 1)
					&& (NULL == TargetModule || NULL == _strnicmp(TargetModule, it_module->szModule, strlen(TargetModule)))) {
					ret &= !!SetThreadAffinityFromThreadId(*it_tid, ThreadAffinityMask);
					break;
				}
			}
		}
	}
	return ret;
}


typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

static DWORD CountSetBits(ULONG_PTR bitMask) {
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    for (ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT; bitTest; bitTest >>= 1)
        bitSetCount += ((bitMask & bitTest) != 0);

    return bitSetCount;
}

BOOL getProcessorCount(DWORD *physical_processor_core, DWORD *logical_processor_core) {
	*physical_processor_core = 0;
	*logical_processor_core = 0;

    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle("kernel32"), "GetLogicalProcessorInformation");
    if (NULL == glpi)
		return FALSE;

    DWORD returnLength = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
	while (FALSE == glpi(buffer, &returnLength)) {
		if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
			if (buffer) 
				free(buffer);
			if (NULL == (buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength)))
				return FALSE;
		}
	}

    DWORD logicalProcessorCount = 0;
    DWORD numaNodeCount = 0;
    DWORD processorCoreCount = 0;
    DWORD processorL1CacheCount = 0;
    DWORD processorL2CacheCount = 0;
    DWORD processorL3CacheCount = 0;
    DWORD processorPackageCount = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    for (DWORD byteOffset = 0; byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength;
		byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
        switch (ptr->Relationship) {
        case RelationNumaNode:
            // Non-NUMA systems report a single record of this type.
            numaNodeCount++;
            break;
        case RelationProcessorCore:
            processorCoreCount++;
            // A hyperthreaded core supplies more than one logical processor.
            logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
            break;

        case RelationCache:
			{
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache. 
            PCACHE_DESCRIPTOR Cache = &ptr->Cache;
			processorL1CacheCount += (Cache->Level == 1);
			processorL2CacheCount += (Cache->Level == 2);
			processorL3CacheCount += (Cache->Level == 3);
            break;
			}
        case RelationProcessorPackage:
            // Logical processors share a physical package.
            processorPackageCount++;
            break;

        default:
            //Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.
            break;
        }
        ptr++;
    }

	*physical_processor_core = processorCoreCount;
	*logical_processor_core = logicalProcessorCount;

    return TRUE;
}

int getCPUName(char *buf, size_t nSize) {
    int CPUInfo[4] = {-1};
    __cpuid(CPUInfo, 0x80000000);
    unsigned int nExIds = CPUInfo[0];
	if (nSize < 0x40)
		return 1;
	memset(buf, 0, 0x40);
    for (unsigned int i = 0x80000000; i <= nExIds; i++) {
        __cpuid(CPUInfo, i);
		int offset = 0;
		switch (i) {
			case 0x80000002: offset =  0; break;
			case 0x80000003: offset = 16; break;
			case 0x80000004: offset = 32; break;
		}
		memcpy(buf + offset, CPUInfo, sizeof(CPUInfo)); 
	}
	const int str_len = strlen(buf);
	for (int i = 0; i < str_len; i++) {
		if (buf[i] != ' ') {
			if (i)
				memmove(buf, buf + i, str_len + 1 - i);
			break;
		}
	}
	return 0;
}

