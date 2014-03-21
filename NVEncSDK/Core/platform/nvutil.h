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

#ifndef _CODECS_INCLUDE_NVUTIL_H_
#define _CODECS_INCLUDE_NVUTIL_H_

#include "NvCallingConventions.h"
#include "NvCriticalSection.h"
#include "NvSystemClock.h"

#ifdef __cplusplus
extern "C" {
#endif

    // General purpose
#ifndef FALSE
#define TRUE    1
#define FALSE   0
#endif

    // 64 bit reference time type
#ifdef _MSC_VER
    typedef __int64 NVTIME;
#else
    typedef long long NVTIME;
#endif

    // Enum's for error values
    typedef enum
    {
        NV_SUCCESS = 0,
        NV_FAIL = 1,
        NV_NOT_FINISHED,
        NV_ERROR_ALLOC = 0x10000,
        NV_ERROR_FREE,
    } NvStatus;

    // Enum's for NVDebugOut client field
    typedef enum
    {
        NV_CLIENT_AUDDEC = 0,
        NV_CLIENT_VIDDEC,
        NV_CLIENT_SPDEC,
        NV_CLIENT_L21DEC,
        NV_CLIENT_AC3ADEC,
        NV_CLIENT_MPGADEC,
        NV_CLIENT_MPGVDEC,
        NV_CLIENT_DVDSPDEC,
        NV_CLIENT_PCMADEC,
        NV_CLIENT_DXVA,
        NV_CLIENT_DTSADEC,
        NV_CLIENT_DHPAENC,
        NV_CLIENT_DPLADEC,
        NV_CLIENT_NAVMAIN,
        NV_CLIENT_NAVAUDIO,
        NV_CLIENT_NAVVIDEO,
        NV_CLIENT_NAVSP,
        NV_CLIENT_MPGDEMUX,
        NV_CLIENT_MPGMUX,
        NV_CLIENT_AUDENC,
        NV_CLIENT_MPGAENC,
        NV_CLIENT_VIDENC,
        NV_CLIENT_MPGVENC,
        NV_CLIENT_AUDIOFX,
        NV_CLIENT_ASCALER,
        NV_CLIENT_MUXFILTR,
        NV_CLIENT_NOTRACE,
        NV_CLIENT_DVDVIDEO,
        NV_CLIENT_NVAVENC,
        NV_CLIENT_TVRATING,
        NV_CLIENT_TSINFO,
        NV_CLIENT_MP3ADEC,
        NV_CLIENT_WAVADEC,
        NV_CLIENT_WMAADEC,
        NV_CLIENT_H264DEC,
        NV_CLIENT_CRYPT,
        NV_CLIENT_DEBLOCK,
        NV_CLIENT_DVDNAV,
        NV_NUM_CLIENTS
    } NV_CLIENTS;

    // Enum's for NVDebugOut type field
    typedef enum
    {
        NV_TYPE_TIMING,  // Timing and performance measurements
        NV_TYPE_TRACE,   // General step point call tracing
        NV_TYPE_MEMORY,  // Memory and object allocation/destruction
        NV_TYPE_LOCKING, // Locking/unlocking of critical sections
        NV_TYPE_ERROR,   // Debug error notification
        NV_TYPE_CUSTOM1,
        NV_TYPE_CUSTOM2,
        NV_TYPE_CUSTOM3,

        NV_NUM_TYPES
    } NV_TYPES;

    // Enum's for NVDebugOut level field
    typedef enum
    {
        NV_LEVEL_MAJOR = 1,
        NV_LEVEL_MINOR = 2,
        NV_LEVEL_DETAIL = 3
    } NV_LEVELS;

    // Logs debug messages to either a file or debug window and/or a console window
    void NVDebugRegister(char *modulename, char *clientname, NV_CLIENTS client);
    void NVDebugReRegister(char *modulename, char *clientname, NV_CLIENTS client);
    void NVDebugUnregister(char *modulename, char *clientname, NV_CLIENTS client);
    void NVDebugOut(NV_CLIENTS client, NV_TYPES type, NV_LEVELS level, char *szFormat, ...);
    void NVTimeOut1(NV_CLIENTS client, NV_TYPES type, NV_LEVELS level, char *text, int ms1);
    void NVTimeOut2(NV_CLIENTS client, NV_TYPES type, NV_LEVELS level, char *text, int ms1, int ms2);
    void NVTimeOut3(NV_CLIENTS client, NV_TYPES type, NV_LEVELS level, char *text, int ms1, int ms2, int ms3);

    // Trace functions
#ifdef NVTRACE
#define NVDRC(_x_) NVDebugRegister _x_
#define NVDRRC(_x_) NVDebugReRegister _x_
#define NVDUC(_x_) NVDebugUnregister _x_
#define NVDPF(_x_) NVDebugOut _x_
#define NVDTS1(_x_) NVTimeOut1 _x_
#define NVDTS2(_x_) NVTimeOut2 _x_
#define NVDTS3(_x_) NVTimeOut3 _x_
#else
#define NVDRC(_x_)
#define NVDRRC(_x_)
#define NVDUC(_x_)
#define NVDPF(_x_)
#define NVDTS1(_x_)
#define NVDTS2(_x_)
#define NVDTS3(_x_)
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#define ARRAY_ELEMENT_COUNT(_array) (sizeof(_array) / sizeof(_array[0]))

    typedef void *CORE_HANDLE;

#ifdef __cplusplus
}
#endif

#endif
