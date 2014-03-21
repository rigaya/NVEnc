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
// NvThreadingClasses.cpp
//
// Some convenience classes for the NvThreading::INvThreading.
//
// Copyright(c) 2003 NVIDIA Corporation.
//---------------------------------------------------------------------------

#include <include/NvAssert.h>
#include <threads/NvThreadingClasses.h>

CNvMutex::CNvMutex() :
    m_pThreading(INvThreading::GetThreading())
{
    NV_ASSERT(m_pThreading->MutexCreate(&m_uHandle) == RESULT_OK);
}

CNvMutex::~CNvMutex()
{
    NV_ASSERT(m_pThreading->MutexDestroy(&m_uHandle) == RESULT_OK);
}

void CNvMutex::Acquire() const
{
    NV_ASSERT(m_pThreading->MutexAcquire(m_uHandle) == RESULT_OK);
}

bool CNvMutex::TryAcquire() const
{
    NvResult er = m_pThreading->MutexTryAcquire(m_uHandle);
    NV_ASSERT(er == RESULT_OK || er == RESULT_FALSE);

    return (er == RESULT_OK);
}

void CNvMutex::Release() const
{
    NV_ASSERT(m_pThreading->MutexRelease(m_uHandle) == RESULT_OK);
}

CNvMutex::operator INvThreading::Handle() const
{
    return m_uHandle;
}

CNvAutoMutex::CNvAutoMutex(const CNvMutex &rMutex) :
    m_Mutex(rMutex),
    m_bReleased(false)
{
    m_Mutex.Acquire();
}

CNvAutoMutex::~CNvAutoMutex()
{
    if (!m_bReleased)
    {
        m_Mutex.Release();
    }
}

void CNvAutoMutex::Release()
{
    m_Mutex.Release();
    m_bReleased = true;
}

CNvEvent::CNvEvent(bool bManual, bool bSet) :
    m_pThreading(INvThreading::GetThreading())
{
    NV_ASSERT(m_pThreading->EventCreate(&m_uHandle, bManual, bSet) == RESULT_OK);
}

CNvEvent::~CNvEvent()
{
    NV_ASSERT(m_pThreading->EventDestroy(&m_uHandle) == RESULT_OK);
}

bool CNvEvent::Wait(U32 uTimeoutMs)
{
    NvResult ret = m_pThreading->EventWait(m_uHandle, uTimeoutMs);

    switch (ret)
    {
        case RESULT_OK:
            return true;

        case RESULT_TIMEOUT:
            return false;

        default:
            NV_ASSERT(false);
            break;
    }

    return false;
}

void CNvEvent::Set()
{
    NV_ASSERT(m_pThreading->EventSet(m_uHandle) == RESULT_OK);
}

void CNvEvent::Reset()
{
    NV_ASSERT(m_pThreading->EventReset(m_uHandle) == RESULT_OK);
}

CNvEvent::operator INvThreading::Handle() const
{
    return m_uHandle;
}

CNvSemaphore::CNvSemaphore(U32 uInitCount, U32 uMaxCount) :
    m_pThreading(INvThreading::GetThreading())
{
    NV_ASSERT(m_pThreading->SemaphoreCreate(&m_uHandle, uInitCount, uMaxCount) == RESULT_OK);
}

CNvSemaphore::~CNvSemaphore()
{
    NV_ASSERT(m_pThreading->SemaphoreDestroy(&m_uHandle) == RESULT_OK);
}

void CNvSemaphore::Increment()
{
    NV_ASSERT(m_pThreading->SemaphoreIncrement(m_uHandle) == RESULT_OK);
}

bool CNvSemaphore::Decrement(U32 uTimeoutMs)
{
    NvResult ret = m_pThreading->SemaphoreDecrement(m_uHandle, uTimeoutMs);

    switch (ret)
    {
        case RESULT_OK:
            return true;

        case RESULT_TIMEOUT:
            return false;

        default:
            NV_ASSERT(false);
            break;
    }

    return false;
}

CNvSemaphore::operator INvThreading::Handle() const
{
    return m_uHandle;
}

CNvTimer::CNvTimer(bool (*pFunc)(void *), void *pUserData) :
    m_pThreading(INvThreading::GetThreading()),
    m_uHandle(INvThreading::NV_HANDLE_INVALID),
    m_bPaused(false),
    m_pFunc(pFunc),
    m_pUserData(pUserData)
{
}

CNvTimer::~CNvTimer()
{
    if (m_uHandle != INvThreading::NV_HANDLE_INVALID)
    {
        NV_ASSERT(m_pThreading->TimerDestroy(&m_uHandle) == RESULT_OK);
    }
}

void CNvTimer::TimerConfig(bool (*pFunc)(void *pUserData), void *pUserData)
{
    CNvAutoMutex oLock(m_Mutex);
    m_pFunc = pFunc;
    m_pUserData = pUserData;
}

void CNvTimer::TimerStart(U32 uPeriodMs)
{
    TimerStart(uPeriodMs, uPeriodMs);
}

void CNvTimer::TimerStart(U32 uTimeMs, U32 uPeriodMs)
{
    if (m_uHandle != INvThreading::NV_HANDLE_INVALID)
    {
        NV_ASSERT(m_pThreading->TimerDestroy(&m_uHandle) == RESULT_OK);
    }

    m_uStartTimeMs = m_pThreading->GetTicksMs();
    m_uTimeMs = uTimeMs;
    m_uPeriodMs = uPeriodMs;
    m_bPaused = false;
    NV_ASSERT(m_pThreading->TimerCreate(&m_uHandle, m_FuncWrapperStatic, this, uTimeMs, uPeriodMs) == RESULT_OK);
}

void CNvTimer::TimerPause(bool bPause)
{
    if (bPause)
    {
        if (m_uHandle == INvThreading::NV_HANDLE_INVALID)
        {
            return;
        }

        NV_ASSERT(m_pThreading->TimerDestroy(&m_uHandle) == RESULT_OK);
        m_uStopTimeMs = m_pThreading->GetTicksMs();
        m_bPaused = true;
    }
    else if (m_bPaused)
    {
        U32 uUsedTime = m_uStopTimeMs - m_uStartTimeMs;

        U32 uRemainingTime;

        if (uUsedTime < m_uTimeMs)
        {
            uRemainingTime = m_uTimeMs - uUsedTime;
        }
        else
        {
            uUsedTime = m_uTimeMs;
            uRemainingTime = 0;
        }

        m_uStartTimeMs = m_pThreading->GetTicksMs() - uUsedTime;
        m_bPaused = false;
        NV_ASSERT(m_pThreading->TimerCreate(&m_uHandle, m_FuncWrapperStatic, this, uRemainingTime, m_uPeriodMs) == RESULT_OK);
    }
}

void CNvTimer::TimerStop()
{
    if (m_uHandle == INvThreading::NV_HANDLE_INVALID)
    {
        return;
    }

    NV_ASSERT(m_pThreading->TimerDestroy(&m_uHandle) == RESULT_OK);
}

CNvTimer::operator INvThreading::Handle() const
{
    return m_uHandle;
}

bool CNvTimer::TimerFunc()
{
    CNvAutoMutex oLock(m_Mutex);

    if (m_pFunc)
    {
        return m_pFunc(m_pUserData);
    }
    else
    {
        return false;
    }
}

bool CNvTimer::m_FuncWrapperStatic(void *pParam)
{
    CNvTimer *p_Timer = static_cast<CNvTimer *>(pParam);
    return p_Timer->m_FuncWrapper();
}

bool CNvTimer::m_FuncWrapper()
{
    if (m_uHandle == INvThreading::NV_HANDLE_INVALID)
    {
        return false;
    }

    m_uStartTimeMs = m_pThreading->GetTicksMs();
    m_uTimeMs = m_uPeriodMs;

    return TimerFunc();
}

CNvThread::CNvThread(const char *szThreadName, S32 sPriority, bool bOneShot) :
    m_pThreading(INvThreading::GetThreading()),
    m_hThread(INvThreading::NV_HANDLE_INVALID),
    m_bQuit(true),
    m_bSyncStart(false),
    m_sPriority(sPriority),
    m_bOneShot(bOneShot),
    m_pFunc(0),
    m_pUserData(0),
    m_szThreadName(szThreadName)
{
}

CNvThread::CNvThread(const char *szThreadName, bool (*pFunc)(void *pUserData), void *pUserData, S32 sPriority) :
    m_pThreading(INvThreading::GetThreading()),
    m_hThread(INvThreading::NV_HANDLE_INVALID),
    m_bQuit(true),
    m_bSyncStart(false),
    m_bOneShot(false),
    m_sPriority(sPriority),
    m_pFunc(pFunc),
    m_pUserData(pUserData),
    m_szThreadName(szThreadName)
{
}

CNvThread::~CNvThread()
{
}

CNvThread::CAutoLock::CAutoLock(const CNvThread *pThread) :
    m_pThread(pThread),
    m_bReleased(false)
{
    m_pThread->ThreadLock();
}

CNvThread::CAutoLock::~CAutoLock()
{
    if (!m_bReleased)
    {
        m_pThread->ThreadUnlock();
    }
}

void CNvThread::CAutoLock::Release()
{
    m_pThread->ThreadUnlock();
    m_bReleased = true;
}

bool CNvThread::ThreadInit()
{
    return true;
}

bool CNvThread::ThreadFunc()
{
    if (m_pFunc)
    {
        return m_pFunc(m_pUserData);
    }
    else
    {
        return false;
    }
}

bool CNvThread::ThreadFini()
{
    return true;
}

void CNvThread::ThreadStart(bool bSync)
{
    if (m_hThread != INvThreading::NV_HANDLE_INVALID)
    {
        return;
    }

    m_bQuit = false;
    m_bSyncStart = bSync;
    NV_ASSERT(m_pThreading->ThreadCreate(&m_hThread, m_FuncStatic, this, m_sPriority) == RESULT_OK);

    if (bSync)
    {
        m_EventStart.Wait((U32)INvThreading::NV_TIMEOUT_INFINITE);
    }
}

void CNvThread::ThreadQuit()
{
    if (m_hThread == INvThreading::NV_HANDLE_INVALID)
    {
        return;
    }

    m_bQuit = true;
    ThreadTrigger();
    NV_ASSERT(m_pThreading->ThreadDestroy(&m_hThread) == RESULT_OK);
}

void CNvThread::ThreadTrigger()
{
    if (m_hThread == INvThreading::NV_HANDLE_INVALID)
    {
        return;
    }

    m_Event.Set();
}

void CNvThread::ThreadLock() const
{
    m_Mutex.Acquire();
}

void CNvThread::ThreadUnlock() const
{
    m_Mutex.Release();
}

void CNvThread::ThreadPriority(S32 sPriority)
{
    m_sPriority = sPriority;

    if (m_hThread != INvThreading::NV_HANDLE_INVALID)
    {
        NV_ASSERT(m_pThreading->ThreadPrioritySet(m_hThread, sPriority) == RESULT_OK);
    }
}

const char *CNvThread::ThreadName() const
{
    return m_szThreadName;
}

U32 CNvThread::m_FuncStatic(void *vpParam)
{
    CNvThread *pThis = reinterpret_cast<CNvThread *>(vpParam);
    return pThis->m_Func();
}

U32 CNvThread::m_Func()
{
    NV_ASSERT(ThreadInit());

    if (m_bSyncStart)
    {
        m_EventStart.Set();
    }

    if (m_bOneShot)
    {
        m_Mutex.Acquire();
        ThreadFunc();
        m_Mutex.Release();
    }
    else
    {
        while (!ThreadIsQuit())
        {
            m_Mutex.Acquire();
            bool bDoMore = ThreadFunc();
            m_Mutex.Release();

            if (bDoMore || ThreadIsQuit())
            {
                continue;
            }

            ThreadBlock((U32)INvThreading::NV_TIMEOUT_INFINITE);
        }
    }

    NV_ASSERT(ThreadFini());

    return 0;
}

bool CNvThread::ThreadIsQuit() const
{
    return m_bQuit;
}

bool CNvThread::ThreadBlock(U32 uTickMs)
{
    return m_Event.Wait(uTickMs);
}

CNvThread::operator INvThreading::Handle() const
{
    return m_hThread;
}

