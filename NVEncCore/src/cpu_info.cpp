//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <vector>
#include <string>
#include <vector>
#include <algorithm>
#include <intrin.h>
#include <tchar.h>
#include "cpu_info.h"

static int getCPUName(char *buffer, size_t nSize) {
	int CPUInfo[4] = {-1};
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];
	if (nSize < 0x40)
		return 1;

	memset(buffer, 0, 0x40);
	for (unsigned int i = 0x80000000; i <= nExIds; i++) {
		__cpuid(CPUInfo, i);
		int offset = 0;
		switch (i) {
			case 0x80000002: offset =  0; break;
			case 0x80000003: offset = 16; break;
			case 0x80000004: offset = 32; break;
			default:
				continue;
		}
		memcpy(buffer + offset, CPUInfo, sizeof(CPUInfo)); 
	}
	auto remove_string =[](char *target_str, const char *remove_str) {
		char *ptr = strstr(target_str, remove_str);
		if (nullptr != ptr) {
			memmove(ptr, ptr + strlen(remove_str), (strlen(ptr) - strlen(remove_str) + 1) *  sizeof(target_str[0]));
		}
	};
	remove_string(buffer, "(R)");
	remove_string(buffer, "(TM)");
	remove_string(buffer, "CPU");
	//crop space beforce string
	for (int i = 0; buffer[i]; i++) {
		if (buffer[i] != ' ') {
			if (i)
				memmove(buffer, buffer + i, strlen(buffer + i) + 1);
			break;
		}
	}
	//remove space which continues.
	for (int i = 0; buffer[i]; i++) {
		if (buffer[i] == ' ') {
			int space_idx = i;
			while (buffer[i+1] == ' ')
				i++;
			if (i != space_idx)
				memmove(buffer + space_idx + 1, buffer + i + 1, strlen(buffer + i + 1) + 1);
		}
	}
	//delete last blank
	if (0 < strlen(buffer)) {
		char *last_ptr = buffer + strlen(buffer) - 1;
		if (' ' == *last_ptr)
			last_ptr = '\0';
	}
	return 0;
}
static int getCPUName(wchar_t *buffer, size_t nSize) {
	int ret = 0;
	char *buf = (char *)calloc(nSize, sizeof(char));
	if (NULL == buf) {
		buffer[0] = L'\0';
		ret = 1;
	} else {
		if (0 == (ret = getCPUName(buf, nSize)))
			MultiByteToWideChar(CP_ACP, 0, buf, -1, buffer, (DWORD)nSize);
		free(buf);
	}
	return ret;
}

double getCPUDefaultClockFromCPUName() {
	double defaultClock = 0.0;
	TCHAR buffer[1024] = { 0 };
	getCPUName(buffer, _countof(buffer));
	TCHAR *ptr_mhz = _tcsstr(buffer, _T("MHz"));
	TCHAR *ptr_ghz = _tcsstr(buffer, _T("GHz"));
	TCHAR *ptr = _tcschr(buffer, _T('@'));
	bool clockInfoAvailable = (NULL != ptr_mhz || ptr_ghz != NULL) && NULL != ptr;
	if (clockInfoAvailable && 1 == _stscanf_s(ptr+1, _T("%lf"), &defaultClock)) {
		return defaultClock * ((NULL == ptr_ghz) ? 1000.0 : 1.0);
	}
	return 0.0;
}

#include <Windows.h>
#include <process.h>

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

	LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(_T("kernel32")), "GetLogicalProcessorInformation");
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

const int LOOP_COUNT = 5000;
const int CLOCKS_FOR_2_INSTRUCTION = 2;
const int COUNT_OF_REPEAT = 4; //以下のようにCOUNT_OF_REPEAT分マクロ展開する
#define REPEAT4(instruction) \
	instruction \
	instruction \
	instruction \
	instruction

static UINT64 __fastcall repeatFunc(int *test) {
	__m128i x0 = _mm_sub_epi32(_mm_setzero_si128(), _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
	__m128i x1 = _mm_add_epi32(x0, x0);
	//計算結果を強引に使うことで最適化による計算の削除を抑止する
	__m128i x2 = _mm_add_epi32(x1, _mm_set1_epi32(*test));
	UINT dummy;
	UINT64 start = __rdtscp(&dummy);

	for (int i = LOOP_COUNT; i; i--) {
		//2重にマクロを使うことでCOUNT_OF_REPEATの2乗分ループ内で実行する
		//これでループカウンタの影響はほぼ無視できるはず
		//ここでのPXORは依存関係により、1クロックあたり1回に限定される
		REPEAT4(REPEAT4(
		x0 = _mm_xor_si128(x0, x1);
		x0 = _mm_xor_si128(x0, x2);))
	}
	
	UINT64 fin = __rdtscp(&dummy); //終了はrdtscpで受ける
	
	//計算結果を強引に使うことで最適化による計算の削除を抑止する
	x0 = _mm_add_epi32(x0, x1);
	x0 = _mm_add_epi32(x0, x2);
	*test = x0.m128i_i32[0];

	return fin - start;
}

static unsigned int __stdcall getCPUClockMaxSubFunc(void *arg) {
	UINT64 *prm = (UINT64 *)arg;
	//渡されたスレッドIDからスレッドAffinityを決定
	//特定のコアにスレッドを縛り付ける
	SetThreadAffinityMask(GetCurrentThread(), 1 << (int)*prm);
	//高優先度で実行
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

	int test = 0;
	UINT64 result = MAXUINT64;
	
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 800; i++) {
			//連続で大量に行うことでTurboBoostを働かせる
			//何回か行って最速値を使用する
			result = min(result, repeatFunc(&test));
		}
		Sleep(1); //一度スレッドを休ませて、仕切りなおす (Sleep(0)でもいいかも)
	}

	*prm = result;

	return 0;
}

//rdtscpを使うと0xc0000096例外 (一般ソフトウェア例外)を発する場合があるらしい
//そこでそれを検出する
bool check_rdtscp_available() {
	__try {
		UINT dummy;
		__rdtscp(&dummy);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return false;
	}
	return true;
}

//__rdtscが定格クロックに基づいた値を返すのを利用して、実際の動作周波数を得る
//やや時間がかかるので注意
double getCPUMaxTurboClock(unsigned int num_thread) {
	double resultClock = 0;
	double defaultClock = getCPUDefaultClock();
	if (0.0 >= defaultClock) {
		return 0.0;
	}

	//http://instlatx64.atw.hu/
	//によれば、Sandy/Ivy/Haswell/Silvermont
	//いずれでもサポートされているのでノーチェックでも良い気がするが...
	//固定クロックのタイマーを持つかチェック (Fn:8000_0007:EDX8)
	int CPUInfo[4] = {-1};
	__cpuid(CPUInfo, 0x80000007);
	if (0 == (CPUInfo[3] & (1<<8))) {
		return defaultClock;
	}
	//rdtscp命令のチェック (Fn:8000_0001:EDX27)
	__cpuid(CPUInfo, 0x80000001);
	if (0 == (CPUInfo[3] & (1<<27))) {
		return defaultClock;
	}
	//例外が発生するなら処理を中断する
	if (!check_rdtscp_available()) {
		return defaultClock;
	}

	DWORD processorCoreCount = 0, logicalProcessorCount = 0;
	getProcessorCount(&processorCoreCount, &logicalProcessorCount);
	//ハーパースレッディングを考慮してスレッドIDを渡す
	int thread_id_multi = (logicalProcessorCount > processorCoreCount) ? logicalProcessorCount / processorCoreCount : 1;
	//上限は物理プロセッサ数、0なら自動的に物理プロセッサ数に設定
	num_thread = (0 == num_thread) ? max(1, processorCoreCount - (logicalProcessorCount == processorCoreCount)) : min(num_thread, processorCoreCount);

	std::vector<HANDLE> list_of_threads(num_thread, NULL);
	std::vector<UINT64> list_of_result(num_thread, 0);
	DWORD thread_loaded = 0;
	for ( ; thread_loaded < num_thread; thread_loaded++) {
		list_of_result[thread_loaded] = thread_loaded * thread_id_multi; //スレッドIDを渡す
		list_of_threads[thread_loaded] = (HANDLE)_beginthreadex(NULL, 0, getCPUClockMaxSubFunc, &list_of_result[thread_loaded], CREATE_SUSPENDED, NULL);
		if (NULL == list_of_threads[thread_loaded]) {
			break; //失敗したらBreak
		}
	}
	
	if (thread_loaded) {
		for (DWORD i_thread = 0; i_thread < thread_loaded; i_thread++) {
			ResumeThread(list_of_threads[i_thread]);
		}
		WaitForMultipleObjects(thread_loaded, &list_of_threads[0], TRUE, INFINITE);
	}

	if (thread_loaded < num_thread) {
		resultClock = defaultClock;
	} else {
		UINT64 min_result = *std::min_element(list_of_result.begin(), list_of_result.end());
		resultClock = (min_result) ? defaultClock * (double)(LOOP_COUNT * COUNT_OF_REPEAT * COUNT_OF_REPEAT * 2) / (double)min_result : defaultClock;
		resultClock = max(resultClock, defaultClock);
	}

	for (auto thread : list_of_threads) {
		if (NULL != thread) {
			CloseHandle(thread);
		}
	}

	return resultClock;
}

#if ENABLE_OPENCL
#include "cl_func.h"
#endif

#pragma warning (push)
#pragma warning (disable: 4100)
double getCPUDefaultClockOpenCL() {
#if !ENABLE_OPENCL
	return 0.0;
#else
	int frequency = 0;
	char buffer[1024] = { 0 };
	getCPUName(buffer, _countof(buffer));
	const std::vector<const char*> vendorNameList = { "Intel", "NVIDIA", "AMD" };
	
	const char *vendorName = NULL;
	for (auto vendor : vendorNameList) {
		if (cl_check_vendor_name(buffer, vendor)) {
			vendorName = vendor;
		}
	}
	if (NULL != vendorName) {
		cl_func_t cl = { 0 };
		cl_data_t data = { 0 };
		if (CL_SUCCESS == cl_get_func(&cl)
			&& CL_SUCCESS == cl_get_platform_and_device(vendorName, CL_DEVICE_TYPE_CPU, &data, &cl)) {
			frequency = cl_get_device_max_clock_frequency_mhz(&data, &cl);
		}
		cl_release(&data, &cl);
	}
	return frequency / 1000.0;
#endif // !ENABLE_OPENCL
}
#pragma warning (pop)

double getCPUDefaultClock() {
	double defautlClock = getCPUDefaultClockFromCPUName();
	if (0 >= defautlClock)
		defautlClock = getCPUDefaultClockOpenCL();
	return defautlClock;
}

int getCPUInfo(TCHAR *buffer, size_t nSize) {
	int ret = 0;
	buffer[0] = _T('\0');
	DWORD processorCoreCount = 0, logicalProcessorCount = 0;
	if (   getCPUName(buffer, nSize)
		|| FALSE == getProcessorCount(&processorCoreCount, &logicalProcessorCount)) {
		ret = 1;
	} else {
		double defaultClock = getCPUDefaultClockFromCPUName();
		bool noDefaultClockInCPUName = (0.0 >= defaultClock);
		if (noDefaultClockInCPUName)
			defaultClock = getCPUDefaultClockOpenCL();
		if (defaultClock > 0.0) {
			if (noDefaultClockInCPUName) {
				_stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" @ %.2fGHz"), defaultClock);
			}
			double maxFrequency = getCPUMaxTurboClock(0);
			//大きな違いがなければ、TurboBoostはないものとして表示しない
			if (maxFrequency / defaultClock > 1.01) {
				_stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" [TB: %.2fGHz]"), maxFrequency);
			}
			_stprintf_s(buffer + _tcslen(buffer), nSize - _tcslen(buffer), _T(" (%dC/%dT)"), processorCoreCount, logicalProcessorCount);
		}
	}
	return ret;
}

BOOL GetProcessTime(HANDLE hProcess, PROCESS_TIME *time) {
	SYSTEMTIME systime;
	GetSystemTime(&systime);
	return (NULL != hProcess
		&& GetProcessTimes(hProcess, (FILETIME *)&time->creation, (FILETIME *)&time->exit, (FILETIME *)&time->kernel, (FILETIME *)&time->user)
		&& (WAIT_OBJECT_0 == WaitForSingleObject(hProcess, 0) || SystemTimeToFileTime(&systime, (FILETIME *)&time->exit)));
}

double GetProcessAvgCPUUsage(HANDLE hProcess, PROCESS_TIME *start) {
	PROCESS_TIME current = { 0 };
	DWORD physicalProcessors = 0, logicalProcessors = 0;
	double result = 0;
	if (NULL != hProcess
		&& getProcessorCount(&physicalProcessors, &logicalProcessors)
		&& GetProcessTime(hProcess, &current)) {
		UINT64 current_total_time = current.kernel + current.user;
		UINT64 start_total_time = (nullptr == start) ? 0 : start->kernel + start->user;
		result = (current_total_time - start_total_time) * 100.0 / (double)(logicalProcessors * (current.exit - ((nullptr == start) ? current.creation : start->exit)));
	}
	return result;
}
