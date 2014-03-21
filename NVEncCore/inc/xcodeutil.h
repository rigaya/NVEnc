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

#ifndef _XCODEUTIL_H_
#define _XCODEUTIL_H_

#if defined(_MSC_VER)
#pragma warning(push)
#endif // _MSC_VER

#if defined(_WIN32) || defined(_WIN64)
#if defined(_WIN32_WINNT)
#  if _WIN32_WINNT < 0x0403
#    undef _WIN32_WINNT
#  endif
#endif
#if !defined(_WIN32_WINNT)
#  define _WIN32_WINNT 0x500
#endif
#include <windows.h>
#if !defined(_ARM_)
#include <emmintrin.h>
#endif
#pragma warning(disable:4512; disable:4100)
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include <include/NvTypes.h>
#include <include/NvAssert.h>
#include <threads/NvThreadingClasses.h>
#include <platform/NvCriticalSection.h>
#include <platform/NvCallingConventions.h>
#include <platform/NvRefCount.h>
#include <platform/NvSystemClock.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Debug
//

#define XCODEAPI            NV_CALL_CONV_COM

void XCODEAPI NvDbgPrint(const char *pszFormat, ...);
void XCODEAPI NvSleep(U32 mSec);
bool XCODEAPI NvQueryPerformanceFrequency(U64 *freq);
bool XCODEAPI NvQueryPerformanceCounter(U64 *counter);

#define RPRINTF(_exp_) NvDbgPrint _exp_

// Debug-only assert
#ifdef _DEBUG
#define NVASSERT    assert
#define TPRINTF     RPRINTF
#else
#define NVASSERT(expr)
#define TPRINTF(_exp_) {}
#endif

typedef S64 NVTIME;

#if defined MACOSX || defined NV_UNIX
#ifndef min
#define min    MIN
#endif
#ifndef max
#define max    MAX
#endif
#endif

#ifndef INFINITE
#define INFINITE UINT_MAX
#endif

// Spinlock class
class CNvSpinLock
{
    private:
        volatile U32 *m_pLock;
    public:
        CNvSpinLock(volatile U32 *pLock):m_pLock(pLock)
        {
            while (NVInterlockedIncrement(m_pLock) != 1)
            {
                NVInterlockedDecrement(m_pLock);
                NVSleep(0);
            }
        }
        ~CNvSpinLock()
        {
            NVInterlockedDecrement(m_pLock);
        }
};


//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple INvRefCOunt implementation (no-dependency alternative to NvRefCountImpl)
//

class CNvRefCount: public virtual INvRefCount
{
    protected:
        U32 m_ulRefCount;
    public:
        CNvRefCount():m_ulRefCount(1) {}
};

#define IMPL_INVREFCOUNT\
    virtual unsigned long NV_CALL_CONV_COM AddRef() { return NVInterlockedIncrement(&m_ulRefCount); }\
    virtual unsigned long NV_CALL_CONV_COM Release() { unsigned long ret=NVInterlockedDecrement(&m_ulRefCount); if (!ret) delete this; return ret; }\
     

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Inline helpers
//

#if 0
static inline S32 lmin(S32 a, S32 b)
{
    return (a<b) ? a:b;
}
static inline S32 lmax(S32 a, S32 b)
{
    return (a>b) ? a:b;
}
#else
static inline S32 lmin(S32 a, S32 b)
{
    return a + (((b-a)>>31) & (b-a));
}
static inline S32 lmax(S32 a, S32 b)
{
    return a + (((a-b)>>31) & (b-a));
}
#endif
static inline U8 clipusb(S32 x)
{
    return (U8)((x>=0)?(x<=255)?x:255:0);
}
static inline U8 avgusb(U32 a, U32 b)
{
    return (U8)((a+b+1)>>1);
}
static inline S16 clipss(S32 x)
{
return (S16)((x>=-32768)?(x<=32767)?x:32767:-32768);
}

#if defined(_WIN32)

#pragma intrinsic(_byteswap_ulong)  // force intrinsic version
#define SWAP_U32(x) _byteswap_ulong(x)

#if defined(_X86_)
__forceinline S32 ftolr(double d)  // float to int with rounding
{
    S32 l;
    _asm
    {
        fld     d
        fistp   l
    }
    return l;
}
__forceinline S64 ftollr(double d)  // float to int64 with rounding
{
    S64 l;
    _asm
    {
        fld     d
        fistp   l
    }
    return l;
}
#else
#if defined(_ARM_)
static inline S32 ftolr(double d)
{
    return (S32)(d+0.5);
}
#else
static inline S32 ftolr(double d)
{
    return _mm_cvtsd_si32(_mm_set_sd(d));
}
#endif
static inline S64 ftollr(double d)
{
    return (S64)(d+0.5);
}
#endif

// Return the number of bits necessary to represent n (n must be positive)
// (return k so that 2^k > n)
__forceinline S32 Log2U31(S32 n)
{
#if defined(_X86_)
    _asm
    {
        mov     eax, n
        lea     eax, [eax+eax+1]
        bsr     eax, eax
    }
#else
    DWORD ndx;
    _BitScanReverse(&ndx, n*2+1);
    return ndx;
#endif
}

#ifdef _ARM_
// TODO: Use intrinsic for PLD
#define MM_PREFETCH_T0(p)
#define MM_PREFETCH_T1(p)
#define MM_PREFETCH_T2(p)
#define MM_PREFETCH_NTA(p)
#define MM_EMMS
#else
#define MM_PREFETCH_T0(p)   _mm_prefetch(((const char *)(p)), _MM_HINT_T0)
#define MM_PREFETCH_T1(p)   _mm_prefetch(((const char *)(p)), _MM_HINT_T1)
#define MM_PREFETCH_T2(p)   _mm_prefetch(((const char *)(p)), _MM_HINT_T2)
#define MM_PREFETCH_NTA(p)  _mm_prefetch(((const char *)(p)), _MM_HINT_NTA)

#if !defined(_WIN64)
#define MM_EMMS             _mm_empty()
#else
#define MM_EMMS
#endif
#endif

#else // if defined(_WIN32)

// byte-swap for unix
#if defined(NV_UNIX)
#include <include/std/byteswap.h>
#define SWAP_U32(x)     bswap_32(x)
#endif

static inline S32 ftolr(double d)
{
    return (S32)(d+0.5);    // NOTE: not identical for negative values
}
static inline S64 ftollr(double d)
{
    return (S64)(d+0.5);
}

// Return the number of bits necessary to represent n (n must be positive)
static inline S32 Log2U31(S32 n)
{
    NVASSERT(n >= 0);
    S32 sz = 0;

    while (n)
    {
        sz++;
        n >>= 1;
    }

    return sz;
}
#define MM_EMMS
#define MM_PREFETCH_T0(p)
#define MM_PREFETCH_T1(p)
#define MM_PREFETCH_T2(p)
#define MM_PREFETCH_NTA(p)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Create magic number macros
//
#define MAGIC(x,y,z,w)  ((x << 24)|(y << 16)|(z << 8)|(w))

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Pointer alignment macros
//

#define ALIGN16(p)      ((unsigned char *) (( ((size_t)(p)) + 15 ) & ~15 ))
#define ALIGN32(p)      ((unsigned char *) (( ((size_t)(p)) + 31 ) & ~31 ))
#define ALIGN64(p)      ((unsigned char *) (( ((size_t)(p)) + 63 ) & ~63 ))
#define ALIGN128(p)     ((unsigned char *) (( ((size_t)(p)) + 127 ) & ~127 ))

#if defined(_MSC_VER)
#define ALIGNED(x)  __declspec(align(x))
#else
#define ALIGNED(x)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Misc common data types
//

typedef struct _NVRect16
{
    S16 left, top, right, bottom;
} NVRect16;

typedef struct _NVRect32
{
    S32 left, top, right, bottom;
} NVRect32;

typedef struct _NVVector16
{
    S16 x,y;

    // Operators
    U32 xy() const
    {
        return *(const U32 *)this;
    }
    bool operator==(const struct _NVVector16 &b) const
    {
        return xy() == b.xy();
    }
    bool operator!=(const struct _NVVector16 &b) const
    {
        return xy() != b.xy();
    }
    bool operator!() const
    {
        return !xy();
    }
    struct _NVVector16 &operator=(const struct _NVVector16 &b)
    {
        *(U32 *)this = *(const U32 *) &b;
        return *this;
    }
    struct _NVVector16 &operator=(U32 xy32)
    {
        *(U32 *)this = xy32;
        return *this;
    }
} NVVector16;


// Frame rate description (as a fraction and as an aproximate time per frame in 100ns units)
typedef struct _NVFrameRateDesc
{
    S32 lNumerator;
    S32 lDenominator;
    NVTIME llAvgTimePerFrame;
} NVFrameRateDesc;

enum NvFrameRate
{
    NV_FRAME_RATE_12 = 0,
    NV_FRAME_RATE_12_5,
    NV_FRAME_RATE_14_98,
    NV_FRAME_RATE_15,
    NV_FRAME_RATE_23_97,
    NV_FRAME_RATE_24,
    NV_FRAME_RATE_25,
    NV_FRAME_RATE_29_97,
    NV_FRAME_RATE_30,
    NV_FRAME_RATE_50,
    NV_FRAME_RATE_59_94,
    NV_FRAME_RATE_60,
    NV_FRAME_RATE_NUMDEN,    // Frame rate in numerator and denominator format (To specify the frame rates other than the predefined values)
    NV_NUM_FRAME_RATES,
    NV_FRAME_RATE_UNKNOWN    // Unknown/unspecified frame rate (or variable)
};

typedef struct _NV_FrameRateDescriptor
{
    NvFrameRate eFrameRate;
    int lNumerator;
    int lDenominator;
} NVFrameRateDescriptor;

extern const NVFrameRateDesc g_FrameRateDesc[];

// Convert AvgTimePerFrame to the closest frame rate code
NvFrameRate XCODEAPI FindClosestFrameRate(S64 llAvgTimePerFrame, S32 lUnits=10000000);
// Simplify an aspect ratio fraction (both inputs must be positive)
void XCODEAPI SimplifyAspectRatio(S32 *pARWidth, S32 *pARHeight);

//framework
enum XcodeFrameWork
{
    XCODE_FRAMEWORK_DSHOW    = MAGIC('D','S','H','O'),
    XCODE_FRAMEWORK_CLIB     = MAGIC('C','L','I','B'),
    XCODE_FRAMEWORK_MFT      = MAGIC('M','F','T','-'),
    XCODE_FRAMEWORK_UNKNOWN
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Video Frame Interface
//

// Surface pixel formats
enum NvSurfaceFormat
{
    NVSF_UYVY=0,
    NVSF_YUY2,
    NVSF_YV12,
    NVSF_NV12,
    NVSF_IYUV,
    NVSF_RGB24,
    NVSF_RGB32,
    NVSF_L10R,   // 10-bit RGB (10.10.10.2) (DPX)
    NVSF_YUV444,
};

// picture structure
enum NvPicStruct
{
    NV_TOP_FIELD          = 0x01,
    NV_BOTTOM_FIELD       = 0x02,
    NV_FRAME_PICTURE      = 0x03
};

class INvCudaAlloc;

// Generic interface for picture buffers
typedef struct _PICBUF_PROPERTIES
{
    int             width;
    int             height;
    int             pitch;
    NvSurfaceFormat format;
    NvPicStruct     structure;          // top-field, bottom-field, frame, etc.
    bool            topFieldFirst;
    bool            repeatFirstField;
    bool            progressiveFrame;
    int             bufferId;           // application-specific buffer id
    unsigned char   *pBuf;              // pointer to yuv buffer
    void            *pUserData;         // user data (proprietary information)
    INvCudaAlloc    *pCudaAlloc;        // interface to associated cuda allocation, if any (NULL otherwise)
} PICBUF_PROPERTIES;


class IPicBuf
{
        public:
        virtual void AddRef() = 0;
        virtual void Release() = 0;
        virtual void GetProperties(PICBUF_PROPERTIES *pPicBufProp) = 0;
        virtual void SetProperties(PICBUF_PROPERTIES *pPicBufProp) = 0;

        protected:
        virtual ~IPicBuf() {}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Misc useful utility functions
//

// Return CPU features
#define CPU_FEATURE_MMX     0x01    // MMX
#define CPU_FEATURE_ISSE    0x02    // Integer SSE support (AMD Athlon)
#define CPU_FEATURE_SSE     0x04    // Full SSE support (P3, AthlonXP)
#define CPU_FEATURE_SSE2    0x08    // SSE2 (P4, Athlon64)
#define CPU_FEATURE_SSE3    0x10    // SSE3
#define CPU_FEATURE_SSSE3   0x20    // Supplemental SSE3
#define CPU_FEATURE_SSE41   0x40    // SSE4.1
#define CPU_FEATURE_SSE42   0x80    // SSE4.2

unsigned long XCODEAPI GetCPUFeatures();
unsigned long XCODEAPI GetCPUIdentification();
bool XCODEAPI GetCPUBrand(char *pszCPUBrand, int iCPUBrandBytes);
unsigned int XCODEAPI GetCPUCoreCount();

// set 'nbits' bits starting at bit offset 'offset' from buffer location 'ptr'
void XCODEAPI NvSetBits(U8 *p, S32 offset, U32 val, U32 nbits);

// Reads a string value from a configuration file
bool XCODEAPI ReadConfigString(const char *pszCfgFileName, const char *pszKeyName, char *pszStr, size_t ccMax);
int XCODEAPI ReadConfigInt(const char *pszCfgFileName, const char *pszKeyName, int lDefaultValue);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

#endif // _XCODEUTIL_H_
