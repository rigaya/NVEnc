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

#include "xcodeutil.h"
#if defined WIN32 || defined _WIN32
#include <threads/NvThreadingWin32.h>
#elif defined __APPLE__ || defined __MACOSX || defined __linux || defined NV_UNIX
#include <sys/time.h>
#endif
#include <platform/NvStrings.h>

#if defined WIN32 || defined _WIN32
// This is defined in <intrin.h>, but somehow it conflicts with other headers
#ifndef _ARM_
extern "C" void __cpuid(int a[4], int b);
#pragma intrinsic(__cpuid)
#endif

#pragma warning(disable: 4996)

// CPU Identification
unsigned long XCODEAPI GetCPUIdentification()
{
#ifdef _ARM_
    return 0;
#else
    int cpuinfo[4];
    unsigned long cpu_identification = 0;
    __cpuid(cpuinfo, 1);
    cpu_identification = cpuinfo[0];
    return cpu_identification;
#endif
}
// CPU Features
unsigned long XCODEAPI GetCPUFeatures()
{
#ifdef _ARM_
    return 0;
#else
    int cpuinfo[4];
    unsigned long cpu_features = 0;
    __cpuid(cpuinfo, 1);

    if (cpuinfo[3] & (1<<23))
    {
        cpu_features |= CPU_FEATURE_MMX;

        if (cpuinfo[3] & (1<<25))
        {
            cpu_features |= (CPU_FEATURE_ISSE|CPU_FEATURE_SSE);

            if (cpuinfo[3] & (1<<26))
            {
                cpu_features |= CPU_FEATURE_SSE2;

                if (cpuinfo[2] & (1<<0))
                {
                    cpu_features |= CPU_FEATURE_SSE3;

                    if (cpuinfo[2] & (1<<9))
                    {
                        cpu_features |= CPU_FEATURE_SSSE3;  // Supplemental SSE3

                        if (cpuinfo[2] & (1<<19))
                        {
                            cpu_features |= CPU_FEATURE_SSE41;  // SSE4.1

                            if (cpuinfo[2] & (1<<20))
                            {
                                cpu_features |= CPU_FEATURE_SSE42;  // SSE4.2
                            }
                        }
                    }
                }
            }
            else
            {
                __cpuid(cpuinfo, 0x80000000);

                if ((unsigned long)cpuinfo[0] >= 0x80000001)
                {
                    __cpuid(cpuinfo, 0x80000001);

                    if (cpuinfo[3] & (1<<22)) // AMD MMX extensions
                        cpu_features |= CPU_FEATURE_ISSE;
                }
            }
        }
    }

    return cpu_features;
#endif
}

/*
* Function
*               GetCPUBrand()
* Parameters
*               pszCPUBrand [in/out] :   Pointer to char array to store the string
*               iCPUBrandBytes [in]  :   size in bytes of above array (min 0x40 req)
* Return
*               true    :   if call succeeded
                false   :   if call failed
*/
bool XCODEAPI GetCPUBrand(char *pszCPUBrand, int iCPUBrandBytes)
{
#ifdef _ARM_
    return false;
#else
    int             iCPUInfo[4] = {-1};
    unsigned int    uiExIds     = 0;
    unsigned int    uiIter      = 0;

    // pszCPUBrand should be atleast 0x40 bytes
    if ((!pszCPUBrand) || (iCPUBrandBytes < 0x40))
    {
        return false;
    }

    memset(pszCPUBrand, 0, iCPUBrandBytes);

    // get max value of extended ids
    __cpuid(iCPUInfo, 0x80000000);
    uiExIds = iCPUInfo[0];

    // traverse till CPU brand string ids
    for (uiIter=0x80000000; uiIter<=uiExIds; ++uiIter)
    {
        __cpuid(iCPUInfo, uiIter);

        // CPU brand string is spread through in the following ids
        if (uiIter == 0x80000002)
        {
            memcpy(pszCPUBrand, iCPUInfo, sizeof(iCPUInfo));
        }
        else if (uiIter == 0x80000003)
        {
            memcpy(pszCPUBrand + 16, iCPUInfo, sizeof(iCPUInfo));
        }
        else if (uiIter == 0x80000004)
        {
            memcpy(pszCPUBrand + 32, iCPUInfo, sizeof(iCPUInfo));
            break;
        }
    }

    // validate that CPU brand string was extracted
    if (uiExIds >= 0x80000004)
    {
        return true;
    }
    else
    {
        return false;
    }

#endif
}

// limited to 32 bits
unsigned int PopCnt(unsigned int bitMask)
{
    unsigned int LSHIFT = sizeof(unsigned int)*8 - 1;
    unsigned int bitSetCount = 0;
    unsigned int bitTest = (unsigned int)1 << LSHIFT;
    unsigned int i;

    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}

/*
* Function
*               GetCPUCoreCount()
* Parameters
*               none
* Return
*               Number of physical processors :   if call succeeded
                0                             :   if call failed
*/
unsigned int XCODEAPI GetCPUCoreCount()
{
    BOOL done = FALSE;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
    DWORD returnLength = 0;
    DWORD logicalProcessorCount = 0;
    DWORD processorCoreCount = 0;
    DWORD processorPackageCount = 0;
    DWORD byteOffset = 0;
    unsigned int uNumPhysProcessors = 0;

    typedef BOOL (WINAPI *LogicalProcessorFunc)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

    LogicalProcessorFunc fnptrLogicalProcessorFunc =
        (LogicalProcessorFunc)GetProcAddress(GetModuleHandle(TEXT("kernel32")),
                                             "GetLogicalProcessorInformation");

    if (fnptrLogicalProcessorFunc == NULL)
    {
        return 0;
    }

    while (!done)
    {
        DWORD rc = fnptrLogicalProcessorFunc(buffer, &returnLength);

        if (FALSE == rc)
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            {
                if (buffer)
                    free(buffer);

                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                             returnLength);

                if (NULL == buffer)
                {
                    return 0;
                }
            }
            else
            {
                return 0;
            }
        }
        else
        {
            done = TRUE;
        }
    }

    ptr = buffer;

    while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength)
    {
        switch (ptr->Relationship)
        {

            case RelationProcessorCore:
                processorCoreCount++;

                // A hyperthreaded core supplies more than one logical processor.
                logicalProcessorCount += PopCnt((unsigned int)ptr->ProcessorMask);
                break;

            case RelationProcessorPackage:
                // Logical processors share a physical package.
                processorPackageCount++;
                break;

            default:
                break;
        }

        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

    free(buffer);

    uNumPhysProcessors = processorCoreCount;

    return uNumPhysProcessors;
}

#elif defined __APPLE__ || defined __MACOSX

// Intel Macs always have MMX, SSE, SSE2.
unsigned long XCODEAPI
GetCPUFeatures()
{
    unsigned long cpu_features = CPU_FEATURE_MMX |
                                 // CPU_FEATURE_ISSE |
                                 CPU_FEATURE_SSE | CPU_FEATURE_SSE2 |
#if defined __SSE3__
                                 CPU_FEATURE_SSE3 |
#endif
#if defined __SSE4_1__
                                 CPU_FEATURE_SSE41 |
#endif
#if defined __SSE4_2__
                                 CPU_FEATURE_SSE42 |
#endif
                                 0;

    return cpu_features;
}

bool XCODEAPI
GetCPUBrand(char *pszCPUBrand, int iCPUBrandBytes)
{
    return false;
}

#else
unsigned long XCODEAPI GetCPUFeatures()
{
    return 0;
}
bool XCODEAPI GetCPUBrand(char *pszCPUBrand, int iCPUBrandBytes)
{
    return false;
}
unsigned int XCODEAPI GetCPUCoreCount()
{
    return 0;
}

#endif // if defined(_WIN32)

#if defined WIN32 || defined _WIN32
extern "C" U64 NvGetSystemClockMicrosecs()
{
    LARGE_INTEGER li, liFreq;

    QueryPerformanceCounter(&li);
    QueryPerformanceFrequency(&liFreq);
    double d = ((double)li.QuadPart/(double)liFreq.QuadPart) * 1000000.0;
    return (U64)d;
}
#endif

// ReadConfigString:
// Returns a string from a config file of the form "keyname = keyvalue"
//
// Not exactly efficient (scans the entire file until a match is found every time),
// but this is assumed to be called very infrequently (startup for internal test tools).
//
bool XCODEAPI ReadConfigString(const char *pszCfgFileName, const char *pszKeyName, char *pszStr, size_t ccMax)
{
    char s[256];
    FILE *f = fopen(pszCfgFileName, "r");

    pszStr[0] = 0;

    if (f)
    {
        while (fgets(s, sizeof(s), f))
        {
            char *pval = strchr(s, '=');

            if (pval)
            {
                char *pkeystart=s, *pkeyend = pval-1;
                *pval++ = 0;

                // remove leading spaces from the key name
                while ((pkeystart < pkeyend) && (pkeystart[0] > 0) && (pkeystart[0] <= ' ')) pkeystart++;

                // remove trailing spaces from key name
                while ((pkeyend >= pkeystart) && (pkeyend[0] > 0) && (pkeyend[0] <= ' ')) *pkeyend-- = 0;

                // Check for a match
                if ((pkeyend > pkeystart) && (!NvStringCaseCmp(pkeystart, pszKeyName)))
                {
                    // Remove leading spaces from the key value
                    while ((pval[0] > 0) && (pval[0] <= ' ')) pval++;

                    if (pval+ccMax-1 < s+sizeof(s))
                        pval[ccMax-1] = 0;

                    strcpy(pszStr, pval);
                    break;
                }
            }
        }

        fclose(f);
    }

    return (pszStr[0] != 0);
}


int XCODEAPI ReadConfigInt(const char *pszCfgFileName, const char *pszKeyName, int lDefaultValue)
{
    char s[128];
    int val = lDefaultValue;

    if (ReadConfigString(pszCfgFileName, pszKeyName, s, sizeof(s)))
    {
        int sign = (s[0] == '-');

        if ((s[sign] >= '0') && (s[sign] <= '9'))
            val = atol(s);
    }

    return val;
}


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Frame rate / Aspect Ratio handling
//

const NVFrameRateDesc g_FrameRateDesc[NV_NUM_FRAME_RATES+2] =
{
    { 12000,  1000, 833333 },   // 12
    { 12500,  1000, 800000 },   // 12.5
    { 15000,  1001, 667333 },   // 14.985
    { 15000,  1000, 666666 },   // 15
    { 24000,  1001, 417083 },   // 23.976
    { 24000,  1000, 416666 },   // 24
    { 25000,  1000, 400000 },   // 25
    { 30000,  1001, 333666 },   // 29.97
    { 30000,  1000, 333333 },   // 30
    { 50000,  1000, 200000 },   // 50
    { 60000,  1001, 166833 },   // 59.94
    { 60000,  1000, 166666 },   // 60
    // 2 dummy entries in case someone attempts to index the array with NV_FRAME_RATE_UNKNOWN
    {     0,     0,      0 },
    {     0,     0,      0 },
};

// Convert AvgTimePerFrame to the closest frame rate code
NvFrameRate XCODEAPI FindClosestFrameRate(S64 llAvgTimePerFrame, S32 lUnits)
{
    S32 llBestErr = (S32)g_FrameRateDesc[0].llAvgTimePerFrame;
    int nBestMatch = 0;

    if ((lUnits != 10000000) && (lUnits > 0))
    {
        llAvgTimePerFrame = (llAvgTimePerFrame * 10000000LL) / lUnits;
    }

    for (int i=0; i<NV_NUM_FRAME_RATES; i++) if (g_FrameRateDesc[i].llAvgTimePerFrame != 0)
        {
            S32 llErr = abs((S32)(llAvgTimePerFrame - g_FrameRateDesc[i].llAvgTimePerFrame));

            if ((!i) || (llErr < llBestErr))
            {
                llBestErr = llErr;
                nBestMatch = i;
            }
        }

    return (NvFrameRate)nBestMatch;
}


// Simplify an aspect ratio fraction (both inputs must be positive)
void XCODEAPI SimplifyAspectRatio(S32 *pARWidth, S32 *pARHeight)
{
    U32 a = abs(*pARWidth), b = abs(*pARHeight);

    while (a)
    {
        U32 tmp = a;
        a = b % tmp;
        b = tmp;
    }

    if (b)
    {
        *pARWidth /= (S32)b;
        *pARHeight /= (S32)b;
    }
}


// set 'nbits' bits starting at bit offset 'offset' from buffer location 'ptr'
// this is totally inefficient, but is useful for basic bitstream support.
void XCODEAPI NvSetBits(U8 *p, S32 offset, U32 val, U32 nbits)
{
    while (nbits > 0)
    {
        U32 bit = (val>>--nbits) & 1;
        U32 bitpos = (7 - offset) & 7;
        p[offset>>3] = (U8)((p[offset>>3] & ~(1<<bitpos)) | (bit<<bitpos));
        offset++;
    }
}


// Output formatted debug string (only here to make debugging easier)
void XCODEAPI NvDbgPrint(const char *pszFormat, ...)
{
#if defined  WIN32 || defined _WIN32
    char s[1024];
    va_list va;
    va_start(va, pszFormat);
    wvsprintfA(s, pszFormat, va);
    OutputDebugStringA(s);
    va_end(va);
#elif defined __APPLE__ || defined __MACOSX || defined __linux
    va_list argptr;
    va_start(argptr, pszFormat);
    vfprintf(stderr, pszFormat, argptr);
    va_end(argptr);
#else
#error fomatted debug print manipulation functions unknown for this platform.
#endif
}

// sleep for time milli-seconds
void XCODEAPI NvSleep(U32 mSec)
{
#if defined  WIN32 || defined _WIN32
    Sleep(mSec);
#elif defined __APPLE__ || defined __MACOSX || defined __linux
    usleep(mSec * 1000);
#else
#error NvSleep function unknown for this platform.
#endif
}

bool XCODEAPI NvQueryPerformanceFrequency(U64 *freq)
{
    *freq = 0;
#if defined  WIN32 || defined _WIN32
    LARGE_INTEGER lfreq;

    if (!QueryPerformanceFrequency(&lfreq))
    {
        return false;
    }

    *freq = lfreq.QuadPart;
#elif defined __APPLE__ || defined __MACOSX || defined __linux
    // We use system's  gettimeofday() to return timer ticks in uSec
    *freq = 1000000000;
#else
#error NvQueryPerformanceFrequency function not defined for this platform.
#endif

    return true;
}

#define SEC_TO_NANO_ULL(sec)    ((U64)sec * 1000000000)
#define MICRO_TO_NANO_ULL(sec)  ((U64)sec * 1000)

bool XCODEAPI NvQueryPerformanceCounter(U64 *counter)
{
    *counter = 0;
#if defined  WIN32 || defined _WIN32
    LARGE_INTEGER lcounter;

    if (!QueryPerformanceCounter(&lcounter))
    {
        return false;
    }

    *counter = lcounter.QuadPart;
#elif defined __APPLE__ || defined __MACOSX || defined __linux
    struct timeval tv;
    int ret;

    ret = gettimeofday(&tv, NULL);

    if (ret != 0)
    {
        return false;
    }

    *counter = SEC_TO_NANO_ULL(tv.tv_sec) + MICRO_TO_NANO_ULL(tv.tv_usec);
#else
#error NvQueryPerformanceCounter function not defined for this platform.
#endif
    return true;
}

