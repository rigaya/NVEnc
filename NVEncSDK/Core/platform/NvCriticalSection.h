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
// Platform independent critical section and related function decorations
//---------------------------------------------------------------------------

#ifndef _NV_CRITICAL_SECTION_H
#define _NV_CRITICAL_SECTION_H

#if defined __unix || defined __linux
#define NV_UNIX
#endif

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define NVCPU_X86_64
#else
#define NVCPU_X86
#endif

#if defined WIN32 || defined _WIN32

#include <windows.h>
typedef CRITICAL_SECTION NVCriticalSection;

__inline void NVCreateCriticalSection(NVCriticalSection *cs)
{
    InitializeCriticalSection(cs);
}
__inline void NVDestroyCriticalSection(NVCriticalSection *cs)
{
    DeleteCriticalSection(cs);
}
__inline void NVLockCriticalSection(NVCriticalSection *cs)
{
    EnterCriticalSection(cs);
}
__inline void NVUnlockCriticalSection(NVCriticalSection *cs)
{
    LeaveCriticalSection(cs);
}

#elif defined __APPLE__ || defined __MACOSX

#include <pthread.h>

typedef struct
{
    pthread_mutex_t     m_mutex;
    pthread_mutexattr_t m_attr;
} NVCriticalSection;

__inline void NVCreateCriticalSection(NVCriticalSection *cs)
{
    pthread_mutexattr_init(&cs->m_attr);
    pthread_mutexattr_settype(&cs->m_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&cs->m_mutex, &cs->m_attr);
}
__inline void NVDestroyCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_destroy(&cs->m_mutex);
    pthread_mutexattr_destroy(&cs->m_attr);
}
__inline void NVLockCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_lock(&cs->m_mutex);
}
__inline void NVUnlockCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_unlock(&cs->m_mutex);
}

#elif defined NV_UNIX

#include "threads/NvPthreadABI.h"

typedef struct
{
    pthread_mutex_t     m_mutex;
    pthread_mutexattr_t m_attr;
} NVCriticalSection;

__inline void NVCreateCriticalSection(NVCriticalSection *cs)
{
    pthread_mutexattr_init(&cs->m_attr);
    pthread_mutexattr_settype(&cs->m_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&cs->m_mutex, &cs->m_attr);
}
__inline void NVDestroyCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_destroy(&cs->m_mutex);
    pthread_mutexattr_destroy(&cs->m_attr);
}
__inline void NVLockCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_lock(&cs->m_mutex);
}
__inline void NVUnlockCriticalSection(NVCriticalSection *cs)
{
    pthread_mutex_unlock(&cs->m_mutex);
}

#else

#error Critical section functions unknown for this platform.

#endif


#if defined WIN32 || defined _WIN32

#include <windows.h>

__inline U32 NVInterlockedIncrement(volatile U32 *p)
{
    return InterlockedIncrement((volatile LONG *)p);
}
__inline U32 NVInterlockedDecrement(volatile U32 *p)
{
    return InterlockedDecrement((volatile LONG *)p);
}

__inline void NVInterlockedAdd(volatile long *pDestination, long value)
{
    long cur;

    do
    {
        cur = *pDestination;
    }
    while (InterlockedCompareExchange(pDestination, cur + value, cur) != cur);
}

#elif defined __APPLE__ || defined __MACOSX

#include <CoreServices/CoreServices.h>
#include <libkern/OSAtomic.h>

__inline U32 NVInterlockedIncrement(volatile U32 *p)
{
    return OSAtomicIncrement32((volatile int32_t *)p);
}
__inline U32 NVInterlockedDecrement(volatile U32 *p)
{
    return OSAtomicDecrement32((volatile int32_t *)p);
}

#elif defined NV_UNIX

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)

/* replace _count with _count+1 and set _acquire=_count */
#define NV_ATOMIC_INCREMENT(_count, _acquire)            \
    {                                                        \
        U32 _release;                                        \
        char _fail;                                          \
        do {                                                 \
            U32 _dummy;                                      \
            _acquire = _count;                               \
            _release = _acquire + 1;                         \
            __asm__ __volatile__(                            \
                                                             "lock ; cmpxchgl %4,%1\n\t"               \
                                                             "setnz %0"                                \
                                                             : "=d" (_fail),                           \
                                                             "=m" (_count),                            \
                                                             "=a" (_dummy)                             \
                                                             : "2" (_acquire),                         \
                                                             "r" (_release));                          \
        } while (_fail);                                     \
    }

/* replace _count with _count-1 and set _acquire=_count */
#define NV_ATOMIC_DECREMENT(_count, _acquire)            \
    {                                                        \
        U32 _release;                                        \
        char _fail;                                          \
        do {                                                 \
            U32 _dummy;                                      \
            _acquire = _count;                               \
            _release = _acquire - 1;                         \
            __asm__ __volatile__(                            \
                                                             "lock ; cmpxchgl %4,%1\n\t"               \
                                                             "setnz %0"                                \
                                                             : "=d" (_fail),                           \
                                                             "=m" (_count),                            \
                                                             "=a" (_dummy)                             \
                                                             : "2" (_acquire),                         \
                                                             "r" (_release));                          \
        } while (_fail);                                     \
    }

#elif defined(NVCPU_ARM)

#define NV_ATOMIC_OPERATION(_count, _acquire, _op)              \
    {                                                               \
        U32 newval;                                                 \
        char fail = 0;                                              \
        \
        do {                                                        \
            _acquire = _count;                                      \
            newval = (_op);                                         \
            \
            __sync_synchronize();                                   \
            __asm__ __volatile(                                     \
                                                                    "ldrex r0, [%[counter]]\n\t"                        \
                                                                    "cmp r0, %[oldval]\n\t"                             \
                                                                    "it eq\n\t"                                         \
                                                                    "strexeq %[ret], %[newval], [%[counter]]\n\t"       \
                                                                    "it ne\n\t"                                         \
                                                                    "movne %[ret], #1\n\t"                              \
                                                                    : [ret] "=&r" (fail)                                \
                                                                    : [counter] "r" (&_count),                          \
                                                                    [oldval] "r" (_acquire),                          \
                                                                    [newval] "r" (newval)                             \
                                                                    : "r0");                                            \
            __sync_synchronize();                                   \
        } while (fail);                                             \
    }

#define NV_ATOMIC_INCREMENT(_count, _acquire)   \
    NV_ATOMIC_OPERATION(_count, _acquire, _count + 1)

#define NV_ATOMIC_DECREMENT(_count, _acquire)   \
    NV_ATOMIC_OPERATION(_count, _acquire, _count - 1)

#else
#error "NV_ATOMIC_{INCREMENT,DECREMENT} undefined for this architecture."
#endif

// NVInterlockedIncrement wants to return the value that *p was
// incremented to.
__inline U32 NVInterlockedIncrement(volatile U32 *p)
{
    U32 val;
    NV_ATOMIC_INCREMENT(*p, val);
    return val + 1;
}
__inline U32 NVInterlockedDecrement(volatile U32 *p)
{
    U32 val;
    NV_ATOMIC_DECREMENT(*p, val);
    return val - 1;
}

#else

#error NVInterlockedIncrement functions unknown for this platform.

#endif

#endif // _NV_CRITICAL_SECTION_H
