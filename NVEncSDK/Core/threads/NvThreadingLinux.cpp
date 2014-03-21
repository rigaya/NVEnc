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
// Linux implementation of the NvThreading::INvThreading interface.
//
// Copyright(c) 2003 NVIDIA Corporation.
//---------------------------------------------------------------------------

#include <threads/NvThreadingLinux.h>
#include <include/NvAssert.h>
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include "threads/NvPthreadABI.h"

void *const INvThreading::NV_HANDLE_INVALID = 0;

static CNvThreadingLinux soThreading;

INvThreading *INvThreading::GetThreading()
{
    return &soThreading;
}

CNvThreadingLinux::CNvThreadingLinux() :
    m_iSchedPolicy(SCHED_OTHER),
    m_iSchedPriorityMin(0),
    m_iSchedPriorityMax(0),
    m_iSchedPriorityBase(0)
{
    initialTime = GetTime();

    srand((unsigned int)initialTime);
}

CNvThreadingLinux::~CNvThreadingLinux()
{
}

NvResult CNvThreadingLinux::MutexCreate(Handle *puMutexHandle)
{
    return _MutexCreate(puMutexHandle, true);
}

NvResult CNvThreadingLinux::MutexAcquire(Handle uMutexHandle)
{
    if (uMutexHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CMutexData *pMutexData = (CMutexData *)uMutexHandle;

    if (pthread_mutex_lock(&pMutexData->mutex))
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::MutexTryAcquire(Handle uMutexHandle)
{
    if (uMutexHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CMutexData *pMutexData = (CMutexData *)uMutexHandle;

    int ret = pthread_mutex_trylock(&pMutexData->mutex);

    if (ret == EBUSY)
    {
        return RESULT_FALSE;
    }
    else if (ret)
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::MutexRelease(Handle uMutexHandle)
{
    if (uMutexHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CMutexData *pMutexData = (CMutexData *)uMutexHandle;

    if (pthread_mutex_unlock(&pMutexData->mutex))
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::MutexDestroy(Handle *puMutexHandle)
{
    if (*puMutexHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CMutexData *pMutexData = (CMutexData *)*puMutexHandle;

    pthread_mutex_destroy(&pMutexData->mutex);
    pthread_mutexattr_destroy(&pMutexData->mutexattr);

    delete pMutexData;

    *puMutexHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

NvResult CNvThreadingLinux::EventCreate(Handle *puEventHandle, bool bManual, bool bSet)
{
    *puEventHandle = NV_HANDLE_INVALID;

    CConditionData *pCond = new CConditionData;

    if (!pCond)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    pCond->manual = bManual;

    if (pthread_mutex_init(&pCond->mutex, NULL))
    {
        delete pCond;
        return RESULT_OUT_OF_HANDLES;
    }

    if (pthread_cond_init(&pCond->condition, NULL))
    {
        pthread_mutex_destroy(&pCond->mutex);
        delete pCond;
        return RESULT_OUT_OF_HANDLES;
    }

    *puEventHandle = (Handle)pCond;

    if (bSet)
    {
        EventSet(*puEventHandle);
    }
    else
    {
        EventReset(*puEventHandle);
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::EventWait(Handle uEventHandle, U32 uTimeoutMs)
{
    if (uEventHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CConditionData *pCond = (CConditionData *)uEventHandle;
    time_ms_t       timetmp;
    struct timespec timeout;
    int             retcode;

    timetmp = GetTime() + uTimeoutMs;
    timeout.tv_sec = timetmp / 1000;
    timeout.tv_nsec = (timetmp % 1000) * 1000000;

    pthread_mutex_lock(&pCond->mutex);

    if (uTimeoutMs == 0)
    {
        if (pCond->signaled)
        {
            if (!pCond->manual)
            {
                pCond->signaled = false;
            }
        }
        else
        {
            pthread_mutex_unlock(&pCond->mutex);
            return RESULT_TIMEOUT;
        }
    }
    else if (uTimeoutMs == NV_TIMEOUT_INFINITE)
    {
        while (!pCond->signaled)
        {
            pthread_cond_wait(&pCond->condition, &pCond->mutex);
        }

        if (!pCond->manual)
        {
            pCond->signaled = false;
        }
    }
    else
    {
        while (!pCond->signaled)
        {
            retcode = pthread_cond_timedwait(&pCond->condition, &pCond->mutex, &timeout);

            if (retcode == ETIMEDOUT)
            {
                pthread_mutex_unlock(&pCond->mutex);
                return RESULT_TIMEOUT;
            }
        }

        if (!pCond->manual)
        {
            pCond->signaled = false;
        }
    }

    pthread_mutex_unlock(&pCond->mutex);

    return RESULT_OK;
}

NvResult CNvThreadingLinux::EventSet(Handle uEventHandle)
{
    if (uEventHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CConditionData *pCond = (CConditionData *)uEventHandle;

    pthread_mutex_lock(&pCond->mutex);
    pCond->signaled = true;
    pthread_cond_signal(&pCond->condition);
    pthread_mutex_unlock(&pCond->mutex);

    return RESULT_OK;
}

NvResult CNvThreadingLinux::EventReset(Handle uEventHandle)
{
    if (uEventHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CConditionData *pCond = (CConditionData *)uEventHandle;

    pthread_mutex_lock(&pCond->mutex);
    pCond->signaled = false;
    pthread_mutex_unlock(&pCond->mutex);

    return RESULT_OK;
}

NvResult CNvThreadingLinux::EventDestroy(Handle *puEventHandle)
{
    if (*puEventHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CConditionData *pCond = (CConditionData *)(*puEventHandle);

    // Destroying this condition will fail if threads are still waiting on it.
    pthread_mutex_lock(&pCond->mutex);
    pthread_cond_destroy(&pCond->condition);
    pthread_mutex_destroy(&pCond->mutex);

    delete pCond;

    *puEventHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

NvResult CNvThreadingLinux::SemaphoreCreate(Handle *puSemaphoreHandle, U32 uInitCount, U32 uMaxCount)
{
    *puSemaphoreHandle = NV_HANDLE_INVALID;

    CNvSemaphoreData *pSem = new CNvSemaphoreData;

    if (!pSem)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    if (uInitCount > uMaxCount)
    {
        uInitCount = uMaxCount;
    }

    pSem->maxCount = uMaxCount;
    pSem->count = uInitCount;

    if (pthread_mutex_init(&pSem->mutex, NULL))
    {
        delete pSem;
        return RESULT_OUT_OF_HANDLES;
    }

    if (pthread_cond_init(&pSem->condition, NULL))
    {
        pthread_mutex_destroy(&pSem->mutex);
        delete pSem;
        return RESULT_OUT_OF_HANDLES;
    }

    *puSemaphoreHandle = (Handle)pSem;
    return RESULT_OK;
}

NvResult CNvThreadingLinux::SemaphoreIncrement(Handle uSemaphoreHandle)
{
    if (uSemaphoreHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CNvSemaphoreData *pSem = (CNvSemaphoreData *)uSemaphoreHandle;

    pthread_mutex_lock(&pSem->mutex);
    pSem->count++;

    if (pSem->count > pSem->maxCount)
    {
        pSem->count = pSem->maxCount;
    }
    else
    {
        pthread_cond_broadcast(&pSem->condition);
    }

    pthread_mutex_unlock(&pSem->mutex);

    return RESULT_OK;
}

NvResult CNvThreadingLinux::SemaphoreDecrement(Handle uSemaphoreHandle, U32 uTimeoutMs)
{
    if (uSemaphoreHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CNvSemaphoreData *pSem = (CNvSemaphoreData *)uSemaphoreHandle;
    time_ms_t       timetmp;
    struct timespec timeout;
    int             retcode;

    timetmp = GetTime() + uTimeoutMs;
    timeout.tv_sec = timetmp / 1000;
    timeout.tv_nsec = (timetmp % 1000) * 1000000;

    while (true)
    {
        pthread_mutex_lock(&pSem->mutex);

        if (pSem->count > 0)
        {
            pSem->count--;
            pthread_mutex_unlock(&pSem->mutex);
            break;
        }

        if (uTimeoutMs == 0)
        {
            pthread_mutex_unlock(&pSem->mutex);
            return RESULT_TIMEOUT;
        }
        else if (uTimeoutMs == NV_TIMEOUT_INFINITE)
        {
            pthread_cond_wait(&pSem->condition, &pSem->mutex);
        }
        else
        {
            retcode = pthread_cond_timedwait(&pSem->condition, &pSem->mutex, &timeout);

            if (retcode == ETIMEDOUT)
            {
                pthread_mutex_unlock(&pSem->mutex);
                return RESULT_TIMEOUT;
            }
        }

        pthread_mutex_unlock(&pSem->mutex);
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::SemaphoreDestroy(Handle *puSemaphoreHandle)
{
    if (*puSemaphoreHandle == NV_HANDLE_INVALID)
    {
        return RESULT_INVALID_HANDLE;
    }

    CNvSemaphoreData *pSem = (CNvSemaphoreData *)(*puSemaphoreHandle);

    // Destroying this condition will fail if threads are still waiting on it.

    pthread_mutex_lock(&pSem->mutex);
    pthread_cond_destroy(&pSem->condition);
    pthread_mutex_destroy(&pSem->mutex);

    delete pSem;

    *puSemaphoreHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

void *CNvThreadingLinux::TimerFunc(void *lpParameter)
{
    CNvTimerData     *pTimer = (CNvTimerData *)(lpParameter);
    struct timespec timeout;
    time_ms_t       currentTime;

    while (!pTimer->exit)
    {
        timeout.tv_sec = pTimer->nextTime / 1000;
        timeout.tv_nsec = (pTimer->nextTime % 1000) * 1000000;

        pthread_mutex_lock(&pTimer->mutex);

        if (pTimer->exit)
        {
            pthread_mutex_unlock(&pTimer->mutex);
            break;
        }

        int ret = pthread_cond_timedwait(&pTimer->condition, &pTimer->mutex, &timeout);
        pthread_mutex_unlock(&pTimer->mutex);

        if (ret != ETIMEDOUT)
        {
            break;
        }

        if (pTimer->exit)
        {
            break;
        }

        if (!(*pTimer->pFunc)(pTimer->pParam))
        {
            break;
        }

        if (!pTimer->period)
        {
            break;
        }

        pTimer->nextTime += pTimer->period;

        currentTime = GetTime();

        if (currentTime > pTimer->nextTime)
        {
            pTimer->nextTime = currentTime;
        }
    }

    return NULL;
}

NvResult CNvThreadingLinux::TimerCreate(Handle *puTimerHandle, bool (*pFunc)(void *pParam), void *pParam, U32 uTimeMs, U32 uPeriodMs)
{
    CNvTimerData *pTimer;

    pTimer = new CNvTimerData;

    if (!pTimer)
    {
        *puTimerHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    *puTimerHandle = reinterpret_cast<Handle>(pTimer);

    pTimer->nextTime = GetTime() + uTimeMs;
    pTimer->period = uPeriodMs;
    pTimer->pFunc = pFunc;
    pTimer->pParam = pParam;
    pTimer->exit = false;

    pthread_mutex_init(&pTimer->mutex, NULL);
    pthread_cond_init(&pTimer->condition, NULL);

    pthread_attr_init(&pTimer->thread_attr);
    pthread_attr_setinheritsched(&pTimer->thread_attr, PTHREAD_INHERIT_SCHED);

    if (pthread_create(&pTimer->thread, &pTimer->thread_attr, TimerFunc, (void *)pTimer))
    {
        pthread_mutex_destroy(&pTimer->mutex);
        pthread_cond_destroy(&pTimer->condition);
        delete pTimer;
        *puTimerHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    return RESULT_OK;
}

NvResult CNvThreadingLinux::TimerDestroy(Handle *puTimerHandle)
{
    CNvTimerData *pTimer = (CNvTimerData *)(*puTimerHandle);

    pthread_mutex_lock(&pTimer->mutex);
    pTimer->exit = true;
    pthread_cond_signal(&pTimer->condition);
    pthread_mutex_unlock(&pTimer->mutex);
    pthread_join(pTimer->thread, NULL);
    pthread_attr_destroy(&pTimer->thread_attr);

    pthread_mutex_destroy(&pTimer->mutex);
    pthread_cond_destroy(&pTimer->condition);

    delete pTimer;
    *puTimerHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

void *CNvThreadingLinux::ThreadFunc(void *lpParameter)
{
    CNvThreadData *pThreadData = (CNvThreadData *)(lpParameter);

    // Set thread priority?
    if (pThreadData->pid == 0)
    {
        // Store thread identifier and signal thread initialization
        pthread_mutex_lock(&pThreadData->mutex);
        pThreadData->pid = getpid();
        pthread_cond_signal(&pThreadData->condition);
        pthread_mutex_unlock(&pThreadData->mutex);
    }

#if (NV_PROFILE==1)
    struct itimerval oldValue;
    setitimer(ITIMER_PROF, &pThreadData->profTimer, &oldValue);
#endif

    U32 result = (*pThreadData->pFunc)(pThreadData->pParam);

    return (void *)result;
}

NvResult CNvThreadingLinux::ThreadCreate(Handle *puThreadHandle, U32(*pFunc)(void *pParam), void *pParam, S32 sPriority)
{
    // Assume invalid handle
    *puThreadHandle = NV_HANDLE_INVALID;

    // Allocate thread handle
    CNvThreadData *pThreadData = new CNvThreadData;

    if (!pThreadData)
    {
        return RESULT_OUT_OF_HANDLES;
    }

#if (NV_PROFILE==1)
    getitimer(ITIMER_PROF, &pThreadData->profTimer);
#endif

    // Store thread parameters
    pThreadData->pFunc = pFunc;
    pThreadData->pParam = pParam;
    pThreadData->pid = 0;

    // Query policy and priority from calling thread
    struct sched_param oSched;

    if (pthread_getschedparam(pthread_self(), &m_iSchedPolicy, &oSched) == 0)
    {
        // Set base priority from calling thread
        if (m_iSchedPolicy == SCHED_OTHER)
            m_iSchedPriorityBase = getpriority(PRIO_PROCESS, 0);
        else
            m_iSchedPriorityBase = oSched.sched_priority;

        // Set minimum and maximum priority limits
        m_iSchedPriorityMin = sched_get_priority_min(m_iSchedPolicy);
        m_iSchedPriorityMax = sched_get_priority_max(m_iSchedPolicy);

        if (m_iSchedPolicy == SCHED_OTHER)
        {
            m_iSchedPriorityMin = -20; // setpriority minimum
            m_iSchedPriorityMax =  19; // setpriority maximum
        }
    }

    // Initialize thread attributes
    pthread_attr_init(&pThreadData->thread_attr);

    // Always inherit scheduling!
    pthread_attr_setinheritsched(&pThreadData->thread_attr, PTHREAD_INHERIT_SCHED);

    // Create thread mutex and condition
    pthread_mutex_init(&pThreadData->mutex, NULL);
    pthread_cond_init(&pThreadData->condition, NULL);

    // Create actual thread and add to thread map
    if (pthread_create(&pThreadData->thread, &pThreadData->thread_attr, ThreadFunc, (void *)pThreadData))
    {
        delete pThreadData;
        return RESULT_OUT_OF_HANDLES;
    }

    // Wait for thread initialization
    pthread_mutex_lock(&pThreadData->mutex);

    while (pThreadData->pid == 0)
        pthread_cond_wait(&pThreadData->condition, &pThreadData->mutex);

    pthread_mutex_unlock(&pThreadData->mutex);

    // Set thread priority
    ThreadPrioritySet(pThreadData, sPriority);

    // Return pThreadData as casting to Handle
    *puThreadHandle = reinterpret_cast<Handle>(pThreadData);

    return RESULT_OK;
}

NvResult CNvThreadingLinux::ThreadPriorityGet(Handle uThreadHandle, S32 &rsPriority)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(uThreadHandle);

    if (pThreadData)
    {
        if (m_iSchedPolicy == SCHED_OTHER)
            rsPriority = m_iSchedPriorityBase - pThreadData->priority;
        else
            rsPriority = pThreadData->priority - m_iSchedPriorityBase;

        return RESULT_OK;
    }
    else
    {
        rsPriority = 0;
        return RESULT_INVALID_HANDLE;
    }
}

NvResult CNvThreadingLinux::ThreadPrioritySet(Handle uThreadHandle, S32 sPriority)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(uThreadHandle);

    if (pThreadData)
    {
        // Compute priority is relative to base
        if (m_iSchedPolicy == SCHED_OTHER)
            pThreadData->priority = m_iSchedPriorityBase - sPriority;
        else
            pThreadData->priority = m_iSchedPriorityBase + sPriority;

        // Ensure priority is within limits
        if (pThreadData->priority < m_iSchedPriorityMin)
            pThreadData->priority = m_iSchedPriorityMin;
        else if (pThreadData->priority > m_iSchedPriorityMax)
            pThreadData->priority = m_iSchedPriorityMax;

        if (m_iSchedPolicy == SCHED_OTHER)
        {
            if (pThreadData->pid == 0)
                return RESULT_FAIL;  // Impossible since ThreadCreate waited on ThreadFunc

            if (setpriority(PRIO_PROCESS, pThreadData->pid, pThreadData->priority))
                return RESULT_FAIL;
        }
        else
        {
            struct sched_param oSched;
            oSched.sched_priority = pThreadData->priority;

            if (pthread_setschedparam(pThreadData->thread, m_iSchedPolicy, &oSched))
                return RESULT_FAIL;
        }

        return RESULT_OK;
    }
    else
    {
        return RESULT_INVALID_HANDLE;
    }
}

NvResult CNvThreadingLinux::ThreadDestroy(Handle *puThreadHandle)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(*puThreadHandle);

    if (!pThreadData)
    {
        return RESULT_INVALID_HANDLE;
    }

    if (pthread_join(pThreadData->thread, NULL))
    {
        return RESULT_INVALID_HANDLE;
    }

    pthread_attr_destroy(&pThreadData->thread_attr);
    pthread_cond_destroy(&pThreadData->condition);
    pthread_mutex_destroy(&pThreadData->mutex);

    delete pThreadData;
    *puThreadHandle = NV_HANDLE_INVALID;

    return RESULT_OK;
}

bool CNvThreadingLinux::ThreadIsCurrent(Handle uThreadHandle)
{
    Handle hCurrent = reinterpret_cast<Handle>(pthread_self());
    return (hCurrent == uThreadHandle);
}

#define TOLERANCE 1000 // 1 second

CNvThreadingLinux::time_ms_t CNvThreadingLinux::GetTime()
{
    struct timeval tv;
    time_ms_t currentTime;
    static time_ms_t previousTime=0;

    // Get current time
    gettimeofday(&tv, 0);
    currentTime = (time_ms_t(tv.tv_sec) * 1000) + (time_ms_t(tv.tv_usec) / 1000);

    // Time going backwards?
    if (previousTime > currentTime)
    {
        // Real time change?
        if ((previousTime - currentTime) < TOLERANCE)
        {
            currentTime = previousTime;
        }
    }

    // Store previous time
    previousTime = currentTime;

    // Return current time
    return (currentTime);
}

U32 CNvThreadingLinux::GetTicksMs()
{
    time_ms_t relativeTime;

    relativeTime = GetTime() - initialTime;

    return (U32)(relativeTime);
}

void CNvThreadingLinux::ResetTimeBase(U32 uOffsetMs)
{
    initialTime = GetTime() - uOffsetMs;
}

#if defined(NV_BSD) || defined(NV_SUNOS) || defined __APPLE__ || defined __MACOSX
#define PTHREAD_MUTEX_RECURSIVE_NP PTHREAD_MUTEX_RECURSIVE
#endif

NvResult CNvThreadingLinux::_MutexCreate(Handle *puMutexHandle, bool bIsRecursive)
{
    *puMutexHandle = NV_HANDLE_INVALID;

    CMutexData *pMutexData = new CMutexData;

    if (!pMutexData)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    if (pthread_mutexattr_init(&pMutexData->mutexattr))
    {
        delete pMutexData;
        return RESULT_OUT_OF_HANDLES;
    }

    if (bIsRecursive)
    {
        if (pthread_mutexattr_settype(&pMutexData->mutexattr, PTHREAD_MUTEX_RECURSIVE_NP))
        {
            pthread_mutexattr_destroy(&pMutexData->mutexattr);
            delete pMutexData;
            return RESULT_OUT_OF_HANDLES;
        }
    }

    if (pthread_mutex_init(&pMutexData->mutex, &pMutexData->mutexattr))
    {
        pthread_mutexattr_destroy(&pMutexData->mutexattr);
        delete pMutexData;
        return RESULT_OUT_OF_HANDLES;
    }

    *puMutexHandle = (Handle)pMutexData;

    return RESULT_OK;
}

U32 CNvThreadingLinux::GetThreadID(Handle hThreadHandle)
{
    return 0;
}
