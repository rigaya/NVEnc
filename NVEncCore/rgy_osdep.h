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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_OSDEP_H__
#define __RGY_OSDEP_H__

#if defined(_MSC_VER)
#ifndef RGY_FORCEINLINE
#define RGY_FORCEINLINE __forceinline
#endif
#ifndef RGY_NOINLINE
#define RGY_NOINLINE __declspec(noinline)
#endif
#else
#ifndef RGY_FORCEINLINE
#define RGY_FORCEINLINE inline
#endif
#ifndef RGY_NOINLINE
#define RGY_NOINLINE __attribute__ ((noinline))
#endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <process.h>
#include <io.h>
#include <conio.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#include <shellapi.h>
#define RGY_LOAD_LIBRARY(x) LoadLibrary(x)
#define RGY_GET_PROC_ADDRESS GetProcAddress
#define RGY_FREE_LIBRARY FreeLibrary

static bool RGYThreadStillActive(HANDLE handle) {
    DWORD exitCode = 0;
    return GetExitCodeThread(handle, &exitCode) == STILL_ACTIVE;
}

static int getStdInKey() {
    static HANDLE hStdInHandle = NULL;
    static bool stdin_from_console = false;
    if (hStdInHandle == NULL) {
        hStdInHandle = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode = 0;
        stdin_from_console = GetConsoleMode(hStdInHandle, &mode) != 0;
    }
    if (stdin_from_console) {
        if (_kbhit()) {
            return _getch();
        }
    }
    return 0;
}

#else //#if defined(_WIN32) || defined(_WIN64)
#include <sys/stat.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/select.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <pthread.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <dlfcn.h>

static inline void *_aligned_malloc(size_t size, size_t alignment) {
    void *p = nullptr;
    int ret = posix_memalign(&p, alignment, size);
    return (ret == 0) ? p : 0;
}
#define _aligned_free free

typedef wchar_t WCHAR;
typedef int BOOL;
typedef void* HANDLE;
typedef void* HMODULE;
typedef void* HINSTANCE;
typedef int errno_t;

#define RGY_LOAD_LIBRARY(x) dlopen((x), RTLD_LAZY)
#define RGY_GET_PROC_ADDRESS dlsym
#define RGY_FREE_LIBRARY dlclose

static uint32_t CP_THREAD_ACP = 0;
static uint32_t CP_UTF8 = 0;

#define __stdcall
#define __fastcall

template <typename _CountofType, size_t _SizeOfArray>
char (*__countof_helper(_CountofType (&_Array)[_SizeOfArray]))[_SizeOfArray];
#define _countof(_Array) (int)sizeof(*__countof_helper(_Array))

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

struct LUID {
  int LowPart;
  int HighPart;
};

static inline char *strtok_s(char *strToken, const char *strDelimit, char **context) {
    return strtok(strToken, strDelimit);
}
static inline char *strcpy_s(char *dst, size_t size, const char *src) {
    return strcpy(dst, src);
}
static inline char *strcpy_s(char *dst, const char *src) {
    return strcpy(dst, src);
}
static inline char *strncpy_s(char *dst, size_t numberOfElements, const char *src, size_t count) {
    return strncpy(dst, src, count);
}
static inline char *strncpy_s(char *dst, const char *src, size_t count) {
    return strncpy(dst, src, count);
}
static inline char *strcat_s(char *dst, size_t size, const char *src) {
    return strcat(dst, src);
}
static inline int _vsprintf_s(char *buffer, size_t size, const char *format, va_list argptr) {
    return vsprintf(buffer, format, argptr);
}
#define sscanf_s sscanf
#define swscanf_s swscanf
#define vsprintf_s(buf, size, fmt, va)  vsprintf(buf, fmt, va)
#define vswprintf_s vswprintf
#define _strnicmp strncasecmp
#define stricmp strcasecmp
#define _stricmp stricmp
#define wcsicmp wcscasecmp
#define _wcsicmp wcsicmp

static short _InterlockedIncrement16(volatile short *pVariable) {
    return __sync_add_and_fetch((volatile short*)pVariable, 1);
}

static short _InterlockedDecrement16(volatile short *pVariable) {
    return __sync_sub_and_fetch((volatile short*)pVariable, 1);
}

static int32_t _InterlockedIncrement(volatile int32_t *pVariable) {
    return __sync_add_and_fetch((volatile int32_t*)pVariable, 1);
}

static int32_t _InterlockedDecrement(volatile int32_t *pVariable) {
    return __sync_sub_and_fetch((volatile int32_t*)pVariable, 1);
}

static inline int _vscprintf(const char * format, va_list pargs) {
    int retval;
    va_list argcopy;
    va_copy(argcopy, pargs);
    retval = vsnprintf(NULL, 0, format, argcopy);
    va_end(argcopy);
    return retval;
}

static inline int _vscwprintf(const WCHAR * format, va_list pargs) {
    int retval;
    va_list argcopy;
    va_copy(argcopy, pargs);
    retval = vswprintf(NULL, 0, format, argcopy);
    va_end(argcopy);
    return retval;
}

static inline int sprintf_s(char *dst, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int ret = vsprintf(dst, format, args);
    va_end(args);
    return ret;
}
static inline int sprintf_s(char *dst, size_t size, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int ret = vsprintf(dst, format, args);
    va_end(args);
    return ret;
}

static inline int fopen_s(FILE **pfp, const char *filename, const char *mode) {
    FILE *fp = fopen(filename, mode);
    *pfp = fp;
    return (fp == NULL) ? 1 : 0;
}

static uint32_t GetCurrentProcessId() {
    pid_t pid = getpid();
    return (uint32_t)pid;
}

static pid_t GetCurrentProcess() {
    return getpid();
}

static pthread_t GetCurrentThread() {
    return pthread_self();
}

static size_t SetProcessAffinityMask(pid_t process, size_t mask) {
    cpu_set_t cpuset_org;
    CPU_ZERO(&cpuset_org);
    sched_getaffinity(process, sizeof(cpu_set_t), &cpuset_org);
    size_t mask_org = 0x00;
    for (uint32_t j = 0; j < sizeof(mask_org) * 8; j++) {
        if (CPU_ISSET(j, &cpuset_org)) {
            mask_org |= ((size_t)1u << j);
        }
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (uint32_t j = 0; j < sizeof(mask) * 8; j++) {
        if (mask & (1 << j)) {
            CPU_SET(j, &cpuset);
        }
    }
    sched_setaffinity(process, sizeof(cpu_set_t), &cpuset);
    return mask_org;
}

static size_t SetThreadAffinityMask(pthread_t thread, size_t mask) {
    cpu_set_t cpuset_org;
    CPU_ZERO(&cpuset_org);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset_org);
    size_t mask_org = 0x00;
    for (uint32_t j = 0; j < sizeof(mask_org) * 8; j++) {
        if (CPU_ISSET(j, &cpuset_org)) {
            mask_org |= ((size_t)1u << j);
        }
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (uint32_t j = 0; j < sizeof(mask) * 8; j++) {
        if (mask & (1 << j)) {
            CPU_SET(j, &cpuset);
        }
    }
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return mask_org;
}

static bool RGYThreadStillActive(pthread_t thread) {
    return pthread_tryjoin_np(thread, nullptr) != 0;
}

enum {
    THREAD_PRIORITY_NORMAL,
    THREAD_PRIORITY_HIGHEST,
    THREAD_PRIORITY_ABOVE_NORMAL,
    THREAD_PRIORITY_BELOW_NORMAL,
    THREAD_PRIORITY_LOWEST,
    THREAD_PRIORITY_IDLE,
};

static void SetThreadPriority(pthread_t thread, int priority) {
    return; //何もしない
}

static void SetPriorityClass(pid_t thread, int priority) {
    return; //何もしない
}

static int getStdInKey() {
#if 0 // stdinで読み込む場合と干渉してしまうので、無効化する
    const int stdInFd = 0; // 0 = stdin
    fd_set fdStdIn;
    FD_ZERO(&fdStdIn);
    FD_SET(stdInFd, &fdStdIn);

    struct timeval timeout = { 0 };
    if (select(stdInFd+1, &fdStdIn, NULL, NULL, &timeout) > 0) {
        char key = 0;
        if (read(0, &key, 1) == 1) {
            return key;
        }
    }
#endif
    return 0;
}

#define _fread_nolock fread
#define _fwrite_nolock fwrite
#define _fgetc_nolock fgetc
#define _fseeki64 fseek
#define _ftelli64 ftell

#endif //#if defined(_WIN32) || defined(_WIN64)

static bool stdInAbort() {
    const auto key = getStdInKey();
    return (key == 'q' || key == 'Q');
}

#endif //__RGY_OSDEP_H__
