// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// ------------------------------------------------------------------------------------------

#include "rgy_osdep.h"
#include "rgy_version.h"
#include "rgy_env.h"
#include "cpu_info.h"
#if ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
#include "gpu_info.h"
#endif
#include "rgy_codepage.h"
#include <unordered_map>

#if defined(_WIN32) || defined(_WIN64)

#include <process.h>
#include <VersionHelpers.h>

typedef void (WINAPI *RtlGetVersion_FUNC)(OSVERSIONINFOEXW*);

static int getRealWindowsVersion(DWORD *major, DWORD *minor, DWORD *build) {
    *major = 0;
    *minor = 0;
    OSVERSIONINFOEXW osver;
    HMODULE hModule = NULL;
    RtlGetVersion_FUNC func = NULL;
    int ret = 1;
    if (   NULL != (hModule = LoadLibrary(_T("ntdll.dll")))
        && NULL != (func = (RtlGetVersion_FUNC)GetProcAddress(hModule, "RtlGetVersion"))) {
        func(&osver);
        *major = osver.dwMajorVersion;
        *minor = osver.dwMinorVersion;
        *build = osver.dwBuildNumber;
        ret = 0;
    }
    if (hModule) {
        FreeLibrary(hModule);
    }
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

BOOL check_OS_Win8orLater() {
#if defined(_WIN32) || defined(_WIN64)
#if (_MSC_VER >= 1800)
    return IsWindows8OrGreater();
#else
    OSVERSIONINFO osvi = { 0 };
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osvi);
    return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
#endif //(_MSC_VER >= 1800)
#else //#if defined(_WIN32) || defined(_WIN64)
    return FALSE;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
#pragma warning(push)
#pragma warning(disable:4996) // warning C4996: 'GetVersionExW': が古い形式として宣言されました。
tstring getOSVersion(OSVERSIONINFOEXW *osinfo) {
    const TCHAR *ptr = _T("Unknown");
    OSVERSIONINFOW info = { 0 };
    OSVERSIONINFOEXW infoex = { 0 };
    info.dwOSVersionInfoSize = sizeof(info);
    infoex.dwOSVersionInfoSize = sizeof(infoex);
    GetVersionExW(&info);
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
        if (info.dwMajorVersion >= 6 || (info.dwMajorVersion == 5 && info.dwMinorVersion >= 2)) {
            GetVersionExW((OSVERSIONINFOW *)&infoex);
        } else {
            memcpy(&infoex, &info, sizeof(info));
        }
        if (info.dwMajorVersion == 6) {
            getRealWindowsVersion(&infoex.dwMajorVersion, &infoex.dwMinorVersion, &infoex.dwBuildNumber);
        }
        if (osinfo) {
            memcpy(osinfo, &infoex, sizeof(infoex));
        }
        switch (infoex.dwMajorVersion) {
        case 3:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = _T("Windows NT 3"); break;
            case 1:  ptr = _T("Windows NT 3.1"); break;
            case 5:  ptr = _T("Windows NT 3.5"); break;
            case 51: ptr = _T("Windows NT 3.51"); break;
            default: break;
            }
            break;
        case 4:
            if (0 == infoex.dwMinorVersion)
                ptr = _T("Windows NT 4.0");
            break;
        case 5:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = _T("Windows 2000"); break;
            case 1:  ptr = _T("Windows XP"); break;
            case 2:  ptr = _T("Windows Server 2003"); break;
            default: break;
            }
            break;
        case 6:
            switch (infoex.dwMinorVersion) {
            case 0:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows Vista") : _T("Windows Server 2008");    break;
            case 1:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 7")     : _T("Windows Server 2008 R2"); break;
            case 2:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 8")     : _T("Windows Server 2012");    break;
            case 3:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 8.1")   : _T("Windows Server 2012 R2"); break;
            case 4:  ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? _T("Windows 10")    : _T("Windows Server 2016");    break;
            default:
                if (5 <= infoex.dwMinorVersion) {
                    ptr = _T("Later than Windows 10");
                }
                break;
            }
            break;
        case 10:
            ptr = (infoex.wProductType == VER_NT_WORKSTATION) ? ((infoex.dwBuildNumber >= 21996) ? _T("Windows 11") : _T("Windows 10")) : _T("Windows Server 2016"); break;
        default:
            if (10 < infoex.dwMajorVersion) {
                ptr = _T("Windows 11 or later");
            }
            break;
        }
        break;
    default:
        break;
    }
    return tstring(ptr);
}
#pragma warning(pop)

tstring getOSVersion() {
    OSVERSIONINFOEXW osversioninfo = { 0 };
    tstring osversionstr = getOSVersion(&osversioninfo);
    osversionstr += strsprintf(_T(" %s (%d)"), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber);
    return osversionstr;
}
#else //#if defined(_WIN32) || defined(_WIN64)

#include <sys/utsname.h>
#include <sys/sysinfo.h>

tstring getOSVersion() {
    std::string str = "";
    FILE *fp = fopen("/etc/os-release", "r");
    if (fp != NULL) {
        char buffer[2048];
        while (fgets(buffer, _countof(buffer), fp) != NULL) {
            if (strncmp(buffer, "PRETTY_NAME=", strlen("PRETTY_NAME=")) == 0) {
                str = trim(std::string(buffer + strlen("PRETTY_NAME=")), " \"\t\n");
                break;
            }
        }
        fclose(fp);
    }
    if (str.length() == 0) {
        struct stat buffer;
        if (stat ("/usr/bin/lsb_release", &buffer) == 0) {
            FILE *fp = popen("/usr/bin/lsb_release -a", "r");
            if (fp != NULL) {
                char buffer[2048];
                while (NULL != fgets(buffer, _countof(buffer), fp)) {
                    str += buffer;
                }
                pclose(fp);
                if (str.length() > 0) {
                    auto sep = split(str, "\n");
                    for (auto line : sep) {
                        if (line.find("Description") != std::string::npos) {
                            std::string::size_type pos = line.find(":");
                            if (pos == std::string::npos) {
                                pos = std::string("Description").length();
                            }
                            pos++;
                            str = line.substr(pos);
                            break;
                        }
                    }
                }
            }
        }
    }
    struct utsname buf;
    uname(&buf);
    if (str.length() == 0) {
        str += buf.sysname;
        str += " ";
    }
    str += " (";
    str += buf.release; //kernelのバージョン
    str += ")";
    return char_to_tstring(trim(str));
}
#endif //#if defined(_WIN32) || defined(_WIN64)

BOOL rgy_is_64bit_os() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sinfo = { 0 };
    GetNativeSystemInfo(&sinfo);
    return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct utsname buf;
    uname(&buf);
    return NULL != strstr(buf.machine, "x64")
        || NULL != strstr(buf.machine, "x86_64")
        || NULL != strstr(buf.machine, "amd64");
#endif //#if defined(_WIN32) || defined(_WIN64)
}

uint64_t getPhysicalRamSize(uint64_t *ramUsed) {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX msex ={ 0 };
    msex.dwLength = sizeof(msex);
    GlobalMemoryStatusEx(&msex);
    if (NULL != ramUsed) {
        *ramUsed = msex.ullTotalPhys - msex.ullAvailPhys;
    }
    return msex.ullTotalPhys;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct sysinfo info;
    sysinfo(&info);
    if (NULL != ramUsed) {
        *ramUsed = info.totalram - info.freeram;
    }
    return info.totalram;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
tstring getACPCodepageStr() {
    const auto codepage = GetACP();
    auto codepage_ptr = codepage_str((uint32_t)codepage);
    if (codepage_ptr != nullptr) {
        return char_to_tstring(codepage_ptr);
    }
    return _T("CP") + char_to_tstring(std::to_string(codepage));
}
#endif

tstring getEnviromentInfo([[maybe_unused]] int device_id) {
    tstring buf;

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));
    uint64_t UsedRamSize = 0;
    uint64_t totalRamsize = getPhysicalRamSize(&UsedRamSize);

    buf += _T("Environment Info\n");
#if defined(_WIN32) || defined(_WIN64)
    OSVERSIONINFOEXW osversioninfo = { 0 };
    tstring osversionstr = getOSVersion(&osversioninfo);
    buf += strsprintf(_T("OS : %s %s (%d) [%s]\n"), osversionstr.c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber, getACPCodepageStr().c_str());
#elif defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    buf += strsprintf(_T("OS : %s %s\n"), getOSVersion().c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"));
#else
    buf += strsprintf(_T("OS : %s\n"), getOSVersion().c_str());
#endif
    buf += strsprintf(_T("CPU: %s\n"), cpu_info);
    buf += strsprintf(_T("RAM: Used %d MB, Total %d MB\n"), (uint32_t)(UsedRamSize >> 20), (uint32_t)(totalRamsize >> 20));

#if ENCODER_QSV
    TCHAR gpu_info[1024] = { 0 };
    getGPUInfo(GPU_VENDOR, gpu_info, _countof(gpu_info), device_id);
    buf += strsprintf(_T("GPU: %s\n"), gpu_info);
#endif //#if ENCODER_QSV
    return buf;
}

#if defined(_WIN32) || defined(_WIN64)

#include <tlhelp32.h>

static bool check_parent(size_t check_pid, const size_t target_pid, const std::unordered_map<size_t, size_t>& map_pid) {
    for (size_t i = 0; i < map_pid.size(); i++) { // 最大でもmap_pid.size()を超えてチェックする必要はないはず
        if (check_pid == target_pid) return true;
        if (check_pid == 0) return false;
        auto key = map_pid.find(check_pid);
        if (key == map_pid.end() || key->second == 0 || key->second == key->first) return false;
        check_pid = key->second;
    }
    return false;
};

std::vector<size_t> createChildProcessIDList(const size_t target_pid) {
    auto h = unique_handle(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0), [](HANDLE h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); });
    if (h.get() == INVALID_HANDLE_VALUE) {
        return std::vector<size_t>();
    }

    PROCESSENTRY32 pe = { 0 };
    pe.dwSize = sizeof(PROCESSENTRY32);

    std::unordered_map<size_t, size_t> map_pid;
    if (Process32First(h.get(), &pe)) {
        do {
            map_pid[pe.th32ProcessID] = pe.th32ParentProcessID;
        } while (Process32Next(h.get(), &pe));
    }

    std::vector<size_t> list_childs;
    for (auto& [pid, parentpid] : map_pid) {
        if (check_parent(parentpid, target_pid, map_pid)) {
            list_childs.push_back(pid);
        }
    }
    return list_childs;
}

#include <winternl.h>

typedef __kernel_entry NTSYSCALLAPI NTSTATUS(NTAPI *NtQueryObject_t)(HANDLE Handle, OBJECT_INFORMATION_CLASS ObjectInformationClass, PVOID ObjectInformation, ULONG ObjectInformationLength, PULONG ReturnLength);
typedef __kernel_entry NTSTATUS(NTAPI *NtQuerySystemInformation_t)(SYSTEM_INFORMATION_CLASS SystemInformationClass, PVOID SystemInformation, ULONG SystemInformationLength, PULONG ReturnLength);

typedef struct _SYSTEM_HANDLE_TABLE_ENTRY_INFO_EX {
    PVOID Object;
    ULONG_PTR UniqueProcessId;
    ULONG_PTR HandleValue;
    ULONG GrantedAccess;
    USHORT CreatorBackTraceIndex;
    USHORT ObjectTypeIndex;
    ULONG HandleAttributes;
    ULONG Reserved;
} SYSTEM_HANDLE_TABLE_ENTRY_INFO_EX, *PSYSTEM_HANDLE_TABLE_ENTRY_INFO_EX;

typedef struct _SYSTEM_HANDLE_INFORMATION_EX {
    ULONG_PTR  NumberOfHandles;
    ULONG_PTR  Reserved;
    SYSTEM_HANDLE_TABLE_ENTRY_INFO_EX Handles[1];
} SYSTEM_HANDLE_INFORMATION_EX, * PSYSTEM_HANDLE_INFORMATION_EX;

#pragma warning(push)
#pragma warning(disable: 4200) //C4200: 非標準の拡張機能が使用されています: 構造体または共用体中にサイズが 0 の配列があります。
typedef struct _OBJECT_NAME_INFORMATION {
    UNICODE_STRING          Name;
    WCHAR                   NameBuffer[0];
} OBJECT_NAME_INFORMATION, * POBJECT_NAME_INFORMATION;
#pragma warning(pop)

typedef struct _OBJECT_BASIC_INFORMATION {
    ULONG Attributes;
    ACCESS_MASK GrantedAccess;
    ULONG HandleCount;
    ULONG PointerCount;
    ULONG PagedPoolCharge;
    ULONG NonPagedPoolCharge;
    ULONG Reserved[3];
    ULONG NameInfoSize;
    ULONG TypeInfoSize;
    ULONG SecurityDescriptorSize;
    LARGE_INTEGER CreationTime;
} OBJECT_BASIC_INFORMATION, *POBJECT_BASIC_INFORMATION;

typedef struct _OBJECT_TYPE_INFORMATION {
    UNICODE_STRING TypeName;
    ULONG TotalNumberOfObjects;
    ULONG TotalNumberOfHandles;
    ULONG TotalPagedPoolUsage;
    ULONG TotalNonPagedPoolUsage;
    ULONG TotalNamePoolUsage;
    ULONG TotalHandleTableUsage;
    ULONG HighWaterNumberOfObjects;
    ULONG HighWaterNumberOfHandles;
    ULONG HighWaterPagedPoolUsage;
    ULONG HighWaterNonPagedPoolUsage;
    ULONG HighWaterNamePoolUsage;
    ULONG HighWaterHandleTableUsage;
    ULONG InvalidAttributes;
    GENERIC_MAPPING GenericMapping;
    ULONG ValidAccessMask;
    BOOLEAN SecurityRequired;
    BOOLEAN MaintainHandleCount;
    UCHAR TypeIndex; // since WINBLUE
    CHAR ReservedByte;
    ULONG PoolType;
    ULONG DefaultPagedPoolCharge;
    ULONG DefaultNonPagedPoolCharge;
} OBJECT_TYPE_INFORMATION, *POBJECT_TYPE_INFORMATION;

typedef struct _OBJECT_TYPES_INFORMATION {
    ULONG NumberOfTypes;
} OBJECT_TYPES_INFORMATION, *POBJECT_TYPES_INFORMATION;

#ifndef STATUS_INFO_LENGTH_MISMATCH
#define STATUS_INFO_LENGTH_MISMATCH ((NTSTATUS)0xC0000004L)
#endif

#define CEIL_INT(x, div) (((x + div - 1) / div) * div)

using unique_handle = std::unique_ptr<std::remove_pointer<HANDLE>::type, std::function<void(HANDLE)>>;

std::vector<unique_handle> createProcessHandleList(const std::vector<size_t>& list_pid, const wchar_t *handle_type) {
    std::vector<unique_handle> handle_list;
    std::unique_ptr<std::remove_pointer<HMODULE>::type, decltype(&FreeLibrary)> hNtDll(LoadLibrary(_T("ntdll.dll")), FreeLibrary);
    if (hNtDll == NULL) return handle_list;

    auto fNtQueryObject = (decltype(NtQueryObject) *)GetProcAddress(hNtDll.get(), "NtQueryObject");
    auto fNtQuerySystemInformation = (decltype(NtQuerySystemInformation) *)GetProcAddress(hNtDll.get(), "NtQuerySystemInformation");
    if (fNtQueryObject == nullptr || fNtQuerySystemInformation == nullptr) {
        return handle_list;
    }

    //auto getObjectTypeNumber = [fNtQueryObject](wchar_t * TypeName) {
    //    static const auto ObjectTypesInformation = (OBJECT_INFORMATION_CLASS)3;
    //    std::vector<char> data(1024, 0);
    //    NTSTATUS status = STATUS_INFO_LENGTH_MISMATCH;
    //    do {
    //        data.resize(data.size() * 2);
    //        ULONG size = 0;
    //        status = fNtQueryObject(NULL, ObjectTypesInformation, data.data(), data.size(), &size);
    //    } while (status == STATUS_INFO_LENGTH_MISMATCH);

    //    POBJECT_TYPES_INFORMATION objectTypes = (POBJECT_TYPES_INFORMATION)data.data();
    //    char *ptr = data.data() + CEIL_INT(sizeof(OBJECT_TYPES_INFORMATION), sizeof(ULONG_PTR));
    //    for (size_t i = 0; i < objectTypes->NumberOfTypes; i++) {
    //        POBJECT_TYPE_INFORMATION objectType = (POBJECT_TYPE_INFORMATION)ptr;
    //        if (wcsicmp(objectType->TypeName.Buffer, TypeName) == 0) {
    //            return (int)objectType->TypeIndex;
    //        }
    //        ptr += sizeof(OBJECT_TYPE_INFORMATION) + CEIL_INT(objectType->TypeName.MaximumLength, sizeof(ULONG_PTR));
    //    }
    //    return -1;
    //};
    //const int fileObjectTypeIndex = getObjectTypeNumber(L"File");

    static const SYSTEM_INFORMATION_CLASS SystemExtendedHandleInformation = (SYSTEM_INFORMATION_CLASS)0x40;
    ULONG size = 0;
    fNtQuerySystemInformation(SystemExtendedHandleInformation, NULL, 0, &size);
    std::vector<char> shibuffer;
    NTSTATUS status = STATUS_INFO_LENGTH_MISMATCH;
    do {
        shibuffer.resize(size + 16*1024);
        status = fNtQuerySystemInformation(SystemExtendedHandleInformation, shibuffer.data(), (ULONG)shibuffer.size(), &size);
    } while (status == STATUS_INFO_LENGTH_MISMATCH);

    if (NT_SUCCESS(status)) {
        const auto currentPID = GetCurrentProcessId();
        const auto currentProcessHandle = GetCurrentProcess();
        const auto shi = (PSYSTEM_HANDLE_INFORMATION_EX)shibuffer.data();
        for (decltype(shi->NumberOfHandles) i = 0; i < shi->NumberOfHandles; i++) {
            const auto handlePID = shi->Handles[i].UniqueProcessId;
            if (std::find(list_pid.begin(), list_pid.end(), handlePID) != list_pid.end()) {
                auto handle = unique_handle((HANDLE)shi->Handles[i].HandleValue, []([[maybe_unused]] HANDLE h) { /*Do nothing*/ });
                // handleValue はプロセスごとに存在する
                // 自プロセスでなければ、DuplicateHandle で自プロセスでの調査用のhandleをつくる
                // その場合は新たに作ったhandleなので CloseHandle が必要
                if (shi->Handles[i].UniqueProcessId != currentPID) {
                    const auto hProcess = std::unique_ptr<std::remove_pointer<HANDLE>::type, decltype(&CloseHandle)>(OpenProcess(PROCESS_DUP_HANDLE, FALSE, (DWORD)handlePID), CloseHandle);
                    if (hProcess) {
                        HANDLE handleDup = NULL;
                        const BOOL ret = DuplicateHandle(hProcess.get(), (HANDLE)shi->Handles[i].HandleValue, currentProcessHandle, &handleDup, 0, FALSE, DUPLICATE_SAME_ACCESS);
                        if (ret) {
                            handle = unique_handle((HANDLE)handleDup, [](HANDLE h) { CloseHandle(h); });
                        }
                    }
                }
                if (handle_type) {
                    // handleの種類を確認する
                    size = 0;
                    status = fNtQueryObject(handle.get(), ObjectTypeInformation, NULL, 0, &size);
                    if (status == STATUS_INFO_LENGTH_MISMATCH) { // 問題なければ、STATUS_INFO_LENGTH_MISMATCHが返る
                        std::vector<char> otibuffer(size, 0);
                        status = fNtQueryObject(handle.get(), ObjectTypeInformation, otibuffer.data(), (ULONG)otibuffer.size(), &size);
                        const auto oti = (PPUBLIC_OBJECT_TYPE_INFORMATION)otibuffer.data();
                        if (NT_SUCCESS(status) && oti->TypeName.Buffer && _wcsicmp(oti->TypeName.Buffer, handle_type) == 0) {
                            //static const OBJECT_INFORMATION_CLASS ObjectNameInformation = (OBJECT_INFORMATION_CLASS)1;
                            //status = fNtQueryObject(handle, ObjectNameInformation, NULL, 0, &size);
                            //std::vector<char> buffer3(size, 0);
                            //status = fNtQueryObject(handle, ObjectNameInformation, buffer3.data(), buffer3.size(), &size);
                            //POBJECT_NAME_INFORMATION oni = (POBJECT_NAME_INFORMATION)buffer3.data();
                            handle_list.push_back(std::move(handle));
                        }
                    }
                } else {
                    handle_list.push_back(std::move(handle));
                }
            }
        }
    }
    return handle_list;
}

#endif //#if defined(_WIN32) || defined(_WIN64)
