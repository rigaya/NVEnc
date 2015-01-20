//
// Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef NVUTILS_H
#define NVUTILS_H

#include "nvCPUOPSys.h"
#include "nvFileIO.h"

#if defined (NV_WINDOWS)
#include <windows.h>
#elif defined NV_UNIX
#include <sys/time.h>
#include <limits.h>

#define FALSE 0
#define TRUE  1
#define INFINITE UINT_MAX
#define stricmp strcasecmp
#define FILE_BEGIN               SEEK_SET
#define INVALID_SET_FILE_POINTER (-1)
#define INVALID_HANDLE_VALUE     ((void *)(-1))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

typedef void* HANDLE;
typedef void* HINSTANCE;
typedef unsigned long DWORD, *LPWORD;
typedef DWORD FILE_SIZE;
#endif

inline bool NvSleep(unsigned int mSec)
{
#if defined (NV_WINDOWS)
    Sleep(mSec);
#elif defined NV_UNIX
    usleep(mSec * 1000);
#else
#error NvSleep function unknown for this platform.
#endif
    return true;
}

inline bool NvQueryPerformanceFrequency(unsigned long long *freq)
{
    *freq = 0;
#if defined (NV_WINDOWS)
    LARGE_INTEGER lfreq;
    if (!QueryPerformanceFrequency(&lfreq)) {
        return false;
    }
    *freq = lfreq.QuadPart;
#elif defined NV_UNIX
    // We use system's  gettimeofday() to return timer ticks in uSec
    *freq = 1000000000;
#else
#error NvQueryPerformanceFrequency function not defined for this platform.
#endif

    return true;
}

#define SEC_TO_NANO_ULL(sec)    ((unsigned long long)sec * 1000000000)
#define MICRO_TO_NANO_ULL(sec)  ((unsigned long long)sec * 1000)

inline bool NvQueryPerformanceCounter(unsigned long long *counter)
{
    *counter = 0;
#if defined (NV_WINDOWS)
    LARGE_INTEGER lcounter;
    if (!QueryPerformanceCounter(&lcounter)) {
        return false;
    }
    *counter = lcounter.QuadPart;
#elif defined NV_UNIX
    struct timeval tv;
    int ret;

    ret = gettimeofday(&tv, NULL);
    if (ret != 0) {
        return false;
    }

    *counter = SEC_TO_NANO_ULL(tv.tv_sec) + MICRO_TO_NANO_ULL(tv.tv_usec);
#else
#error NvQueryPerformanceCounter function not defined for this platform.
#endif
    return true;
}

#if defined NV_UNIX
__inline bool operator==(const GUID &guid1, const GUID &guid2)
{
     if (guid1.Data1    == guid2.Data1 &&
         guid1.Data2    == guid2.Data2 &&
         guid1.Data3    == guid2.Data3 &&
         guid1.Data4[0] == guid2.Data4[0] &&
         guid1.Data4[1] == guid2.Data4[1] &&
         guid1.Data4[2] == guid2.Data4[2] &&
         guid1.Data4[3] == guid2.Data4[3] &&
         guid1.Data4[4] == guid2.Data4[4] &&
         guid1.Data4[5] == guid2.Data4[5] &&
         guid1.Data4[6] == guid2.Data4[6] &&
         guid1.Data4[7] == guid2.Data4[7])
    {
        return true;
    }

    return false;
}
__inline bool operator!=(const GUID &guid1, const GUID &guid2)
{
    return !(guid1 == guid2);
}
#endif
#endif

#define PRINTERR(message, ...) \
    fprintf(stderr, "%s line %d: " message, __FILE__, __LINE__, ##__VA_ARGS__)
