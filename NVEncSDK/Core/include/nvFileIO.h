/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NVFILE_IO_H
#define NVFILE_IO_H

#if defined __linux__
typedef void * HANDLE;
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#endif
#include <stdarg.h>
#pragma warning(push)
#pragma warning(disable:4100)
/*
inline void NvPrintf(const char* format, ...)
{
#ifndef _QUIET
	va_list valist;
	va_start(valist, format);

	vprintf(format, valist);
	va_end(valist);
#endif
}
*/
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

#endif
}

inline void nvGetFileSize(HANDLE hInputFile, DWORD *pFilesize)
{
#if defined (NV_WINDOWS)
    LARGE_INTEGER file_size;

    if (hInputFile != INVALID_HANDLE_VALUE)
    {
        file_size.LowPart = GetFileSize(hInputFile, (LPDWORD)&file_size.HighPart);
        //NvPrintf("[ Input Filesize] : %ld bytes\n", ((LONGLONG) file_size.HighPart << 32) + (LONGLONG)file_size.LowPart);

        if (pFilesize != NULL) *pFilesize = file_size.LowPart;
    }

#elif defined __linux || defined __APPLE__ || defined __MACOSX
    FILE_SIZE file_size;

    if (hInputFile != NULL)
    {
        nvSetFilePointer64(hInputFile, 0, NULL, SEEK_END);
        file_size = ftell((FILE *)hInputFile);
        nvSetFilePointer64(hInputFile, 0, NULL, SEEK_SET);
        NvPrintf("Input Filesize: %ld bytes\n", file_size);

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
        //NvPrintf("nvOpenFile Failed to open \"%s\"\n", input_file);
        exit(EXIT_FAILURE);
    }

#elif defined __linux || defined __APPLE_ || defined __MACOSX
    hInput = fopen(input_file, "rb");

    if (hInput == NULL)
    {
        NvPrintf("nvOpenFile Failed to open \"%s\"\n", input_file);
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
        //NvPrintf("nvOpenFileWrite Failed to open \"%s\"\n", output_file);
        exit(EXIT_FAILURE);
    }

#elif defined __linux || defined __APPLE_ || defined __MACOSX
    hOutput = fopen(output_file, "wb+");

    if (hOutput == NULL)
    {
        NvPrintf("nvOpenFileWrite Failed to open \"%s\"\n", output_file);
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