///
// Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef NVFILE_IO_H
#define NVFILE_IO_H

#if defined __linux__
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>

typedef void * HANDLE;
typedef void             *HINSTANCE;
typedef unsigned long     DWORD, *LPDWORD;
typedef DWORD             FILE_SIZE;

#define FALSE   0
#define TRUE    1
#define INFINITE UINT_MAX

#define FILE_BEGIN               SEEK_SET
#define INVALID_SET_FILE_POINTER (-1)
#define INVALID_HANDLE_VALUE     ((void *)(-1))

#else
#include <stdio.h>
#include <windows.h>
#endif

#include "nvCPUOPSys.h"

typedef unsigned long long  U64;
typedef unsigned int        U32;

#pragma warning(push)
#pragma warning(disable:4100)

inline U32 nvSetFilePointer(HANDLE hInputFile, U32 fileOffset, U32 *moveFilePointer, U32 flag)
{
#if defined (NV_WINDOWS)
    return SetFilePointer(hInputFile, fileOffset, NULL, flag);
#elif defined __linux || defined __APPLE_ || defined __MACOSX
    return fseek((FILE *)hInputFile, fileOffset, flag);
#endif
}

inline U32 nvSetFilePointer64(HANDLE hInputFile, U64 fileOffset, U64 *moveFilePointer, U32 flag)
{
#if defined (NV_WINDOWS)
    return SetFilePointer(hInputFile, ((U32 *)&fileOffset)[0], (PLONG)&((U32 *)&fileOffset)[1], flag);
#elif defined __linux || defined __APPLE__ || defined __MACOSX
    return fseek((FILE *)hInputFile, (long int)fileOffset, flag);
#endif
}

inline bool nvReadFile(HANDLE hInputFile, void *buf, U32 bytes_to_read, U32 *bytes_read, void *operlapped)
{
#if defined (NV_WINDOWS)
    ReadFile(hInputFile, buf, bytes_to_read, (LPDWORD)bytes_read, NULL);
    return true;
#elif defined __linux || defined __APPLE__ || defined __MACOSX
    U32 num_bytes_read;
    num_bytes_read = fread(buf, bytes_to_read, 1, (FILE *)hInputFile);

    if (bytes_read)
    {
        *bytes_read = num_bytes_read;
    }
    return true;
#endif
}

inline void nvGetFileSize(HANDLE hInputFile, DWORD *pFilesize)
{
#if defined (NV_WINDOWS)
    LARGE_INTEGER file_size;

    if (hInputFile != INVALID_HANDLE_VALUE)
    {
        file_size.LowPart = GetFileSize(hInputFile, (LPDWORD)&file_size.HighPart);
        printf("[ Input Filesize] : %I64d bytes\n", ((LONGLONG) file_size.HighPart << 32) + (LONGLONG)file_size.LowPart);

        if (pFilesize != NULL) *pFilesize = file_size.LowPart;
    }

#elif defined __linux || defined __APPLE__ || defined __MACOSX
    FILE_SIZE file_size;

    if (hInputFile != NULL)
    {
        nvSetFilePointer64(hInputFile, 0, NULL, SEEK_END);
        file_size = ftell((FILE *)hInputFile);
        nvSetFilePointer64(hInputFile, 0, NULL, SEEK_SET);
        printf("Input Filesize: %ld bytes\n", file_size);

        if (pFilesize != NULL) *pFilesize = file_size;
    }

#endif
}

inline HANDLE nvOpenFile(const char *input_file)
{
    HANDLE hInput = NULL;

#if defined (NV_WINDOWS)
    hInput = CreateFileA(input_file, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING , FILE_ATTRIBUTE_NORMAL, NULL);

    if (hInput == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "nvOpenFile Failed to open \"%s\"\n", input_file);
        exit(EXIT_FAILURE);
    }

#elif defined __linux || defined __APPLE_ || defined __MACOSX
    hInput = fopen(input_file, "rb");

    if (hInput == NULL)
    {
        fprintf(stderr, "nvOpenFile Failed to open \"%s\"\n", input_file);
        exit(EXIT_FAILURE);
    }

#endif
    return hInput;
}

inline HANDLE nvOpenFileWrite(const char *output_file)
{
    HANDLE hOutput = NULL;

#if defined (NV_WINDOWS)
    hOutput = CreateFileA(output_file, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING , FILE_ATTRIBUTE_NORMAL, NULL);

    if (hOutput == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "nvOpenFileWrite Failed to open \"%s\"\n", output_file);
        exit(EXIT_FAILURE);
    }

#elif defined __linux || defined __APPLE_ || defined __MACOSX
    hOutput = fopen(output_file, "wb+");

    if (hOutput == NULL)
    {
        fprintf(stderr, "nvOpenFileWrite Failed to open \"%s\"\n", output_file);
        exit(EXIT_FAILURE);
    }

#endif
    return hOutput;
}

inline void nvCloseFile(HANDLE hFileHandle)
{
    if (hFileHandle)
    {
#if defined (NV_WINDOWS)
        CloseHandle(hFileHandle);
#else
        fclose((FILE *)hFileHandle);
#endif
    }
}

#pragma warning(pop)

#endif
