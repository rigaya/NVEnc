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


//---------------------------------------------------------------------------
//! \file Types.h
//! \brief Platform independent types.
//!
//! Types.h contains the base types used to construct more abstract data
//! types. This file is necessary because different machines and compilers
//! may apply a different number of bits precision to the C/C++ types \c int,
//! \c unsigned, \c short, \c signed and \c char. Use the platform independent
//! types defined in this module in most if not all places. Use of \c int, and
//! other built in types should only be used within an algorithm where compiled
//! speed is more important.
//---------------------------------------------------------------------------

#ifndef _NV_TYPES_H
#define _NV_TYPES_H

#if defined LINUX || defined __linux || defined NV_UNIX || defined INTEGRITY || defined __CC_ARM || defined TENSILICA

typedef unsigned long long  U64;
typedef signed long long    S64;
typedef unsigned int        U32;
typedef signed int          S32;
typedef unsigned short      U16;
typedef signed short        S16;
typedef unsigned char       U8;
typedef signed char         S8;

// Remapping Windows Definition Types for Linux
typedef void             *HINSTANCE;
typedef int               FILE_HANDLE;
typedef off64_t           FILE_SIZE;
typedef off64_t           FILE_64BIT_HANDLE;
typedef long long         LONGLONG;
typedef unsigned long     DWORD, *LPDWORD;
typedef int               HRESULT;
typedef enum
{
    E_FAIL = 0,
    S_OK   = 1
} eRESULT;

#define FILE_ERROR        -1
#define FILE_ERROR_SET_FP -1

#define U64_MAX 18446744073709551615ULL
#define U64_MIN 0
#define S64_MAX 9223372036854775807LL
#define S64_MIN (-S64_MAX - 1)
#define U32_MAX 4294967295UL
#define U32_MIN 0
#define S32_MAX 2147483647L
#define S32_MIN (-S32_MAX - 1)
#define U16_MAX 65535
#define U16_MIN 0
#define S16_MAX 32767
#define S16_MIN (-S16_MAX - 1)
#define U8_MAX  255
#define U8_MIN  0
#define S8_MAX  127
#define S8_MIN  (-S8_MAX - 1)


#elif defined QNX || defined VXWORKS //FIXME: verify all the data types

typedef unsigned long long int U64;
typedef signed long long int   S64;
typedef unsigned int        U32;
typedef signed int          S32;
typedef unsigned short      U16;
typedef signed short        S16;
typedef unsigned char       U8;
typedef signed char         S8;

// File Definition Types for Linux
typedef int               FILE_HANDLE;
typedef off64_t           FILE_SIZE;
typedef off64_t           FILE_64BIT_HANDLE;
typedef long long         LONGLONG;
typedef unsigned long     DWORD, *LPDWORD;
typedef int               HRESULT;
typedef enum
{
    E_FAIL = 0,
    S_OK   = 1
} eRESULT;

#define FILE_ERROR        -1
#define FILE_ERROR_SET_FP -1

#define U64_MAX 18446744073709551615ULL
#define U64_MIN 0
#define S64_MAX 9223372036854775807LL
#define S64_MIN (-S64_MAX - 1)
#define U32_MAX 4294967295UL
#define U32_MIN 0
#define S32_MAX 2147483647L
#define S32_MIN (-S32_MAX - 1)
#define U16_MAX 65535
#define U16_MIN 0
#define S16_MAX 32767
#define S16_MIN (-S16_MAX - 1)
#define U8_MAX  255
#define U8_MIN  0
#define S8_MAX  127
#define S8_MIN  (-S8_MAX - 1)

#elif defined UNDER_CE

typedef unsigned __int64    U64;
typedef signed __int64      S64;
typedef unsigned int        U32;
typedef signed int          S32;
typedef unsigned short      U16;
typedef signed short        S16;
typedef unsigned char       U8;
typedef signed char         S8;

typedef HANDLE            FILE_HANDLE;
typedef DWORD             FILE_SIZE;
typedef LARGE_INTEGER     FILE_64BIT_HANDLE;
// Unified file definitions
#define FILE_ERROR        INVALID_HANDLE_VALUE
#define FILE_ERROR_SET_FP INVALID_SET_FILE_POINTER

#define U64_MAX 18446744073709551615 /* WARNING! WINCE doesn't allow "ULL" suffix - this value may be truncated */
#define U64_MIN 0
#define S64_MAX 9223372036854775807LL
#define S64_MIN (-S64_MAX - 1)
#define U32_MAX 4294967295UL
#define U32_MIN 0
#define S32_MAX 2147483647L
#define S32_MIN (-S32_MAX - 1)
#define U16_MAX 65535
#define U16_MIN 0
#define S16_MAX 32767
#define S16_MIN (-S16_MAX - 1)
#define U8_MAX  255
#define U8_MIN  0
#define S8_MAX  127
#define S8_MIN  (-S8_MAX - 1)

#elif defined WIN32 || defined _WIN32

#include <windows.h>

typedef unsigned __int64    U64;
typedef signed __int64      S64;
typedef unsigned int        U32;
typedef signed int          S32;
typedef unsigned short      U16;
typedef signed short        S16;
typedef unsigned char       U8;
typedef signed char         S8;

typedef HANDLE            FILE_HANDLE;
typedef DWORD             FILE_SIZE;
typedef LARGE_INTEGER     FILE_64BIT_HANDLE;
// Unified file definitions
#define FILE_ERROR        INVALID_HANDLE_VALUE
#define FILE_ERROR_SET_FP INVALID_SET_FILE_POINTER

#define U64_MAX 18446744073709551615ULL
#define U64_MIN 0
#define S64_MAX 9223372036854775807LL
#define S64_MIN (-S64_MAX - 1)
#define U32_MAX 4294967295UL
#define U32_MIN 0
#define S32_MAX 2147483647L
#define S32_MIN (-S32_MAX - 1)
#define U16_MAX 65535
#define U16_MIN 0
#define S16_MAX 32767
#define S16_MIN (-S16_MAX - 1)
#define U8_MAX  255
#define U8_MIN  0
#define S8_MAX  127
#define S8_MIN  (-S8_MAX - 1)

#elif defined __APPLE__ || defined __MACOSX

typedef unsigned long long  U64;
typedef signed long long    S64;
typedef unsigned int        U32;
typedef signed int          S32;
typedef unsigned short      U16;
typedef signed short        S16;
typedef unsigned char       U8;
typedef signed char         S8;

#define U64_MAX 18446744073709551615ULL
#define U64_MIN 0
#define S64_MAX 9223372036854775807LL
#define S64_MIN (-S64_MAX - 1)
#define U32_MAX 4294967295UL
#define U32_MIN 0
#define S32_MAX 2147483647L
#define S32_MIN (-S32_MAX - 1)
#define U16_MAX 65535
#define U16_MIN 0
#define S16_MAX 32767
#define S16_MIN (-S16_MAX - 1)
#define U8_MAX  255
#define U8_MIN  0
#define S8_MAX  127
#define S8_MIN  (-S8_MAX - 1)

#else

#error Unknown platform.

// The following is for documentation only.

//! \brief Unsigned 64 bits.
//! Use sparingly since some platforms may emulate this.
typedef unsigned long long  U64;

//! \brief Signed 64 bits.
//! Use sparingly since some platforms may emulate this.
typedef signed long long    S64;

//! \brief Unsigned 32 bits.
typedef unsigned int        U32;

//! \brief Signed 32 bits.
typedef signed int          S32;

//! \brief Unsigned 16 bits.
typedef unsigned short      U16;

//! \brief Signed 16 bits.
typedef signed short        S16;

//! \brief Unsigned 8 bits.
//! Note that some platforms char is signed and on other it is unsigned. Use
//! of U8 or S8 is highly recommended.
typedef unsigned char       U8;

//! \brief Signed 8 bits.
//! Note that some platforms char is signed and on other it is unsigned. Use
//! of U8 or S8 is highly recommended.
typedef signed char         S8;

//! \brief Maximum value of a U64.
#define U64_MAX 18446744073709551615

//! \brief Minimum value of a U64.
#define U64_MIN 0

//! \brief Maximum value of an S64.
#define S64_MAX 9223372036854775807

//! \brief Minimum value of an S64.
#define S64_MIN -9223372036854775808

//! \brief Maximum value of a U32.
#define U32_MAX 4294967295

//! \brief Minimum value of a U32.
#define U32_MIN 0

//! \brief Maximum value of an S32.
#define S32_MAX 2147483647

//! \brief Minimum value of an S32.
#define S32_MIN -2147483648

//! \brief Maxmimum value of a U16.
#define U16_MAX 65535

//! \brief Minimum value of a U16.
#define U16_MIN 0

//! \brief Maximum value of an S16.
#define S16_MAX 32767

//! \brief Minimum value of an S16.
#define S16_MIN -32768

//! \brief Maximum value of a U8.
#define U8_MAX  255

//! \brief Minimum value of a U8.
#define U8_MIN  0

//! \brief Maximum value of an S8.
#define S8_MAX  127

//! \brief Minimum value of an S8.
#define S8_MIN  -128

#endif

//! \brief UTF-8 single character.
//! Used for Unicode strings formatted in UTF-8.
typedef U8  UTF8;

//! \brief UTF-16 single character.
//! Used for Unicode strings formatted in UTF-16.
typedef U16 UTF16;

typedef int BOOL;

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#endif
