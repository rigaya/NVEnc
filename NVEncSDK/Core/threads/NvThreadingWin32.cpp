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
// Win32 implementation of the INvThreading interface.
//
// Copyright(c) 2003 NVIDIA Corporation.
//---------------------------------------------------------------------------

#include <threads/NvThreadingWin32.h>
//#include <include/NvMemory.h>
#include <include/NvAssert.h>


// Disable warnings about casting between DWORD and pointer types. We know both are 32bits.
#pragma warning(disable : 4311 4312)

#define _TIMER_TARGET_RESOLUTION 100

void *const INvThreading::NV_HANDLE_INVALID = 0;

INvThreading *INvThreading::GetThreading()
{
    static CNvThreadingWin32 soThreading;
    return &soThreading;
}

CNvThreadingWin32::CNvThreadingWin32()
{
    TIMECAPS TimeCaps;
    MMRESULT MMResult;
    MMResult = timeGetDevCaps(&TimeCaps, sizeof TimeCaps);
    _ASSERT(MMResult == TIMERR_NOERROR);
    m_uTimerResolution = min(max(TimeCaps.wPeriodMin, _TIMER_TARGET_RESOLUTION), TimeCaps.wPeriodMax);
    timeBeginPeriod(m_uTimerResolution);
}

CNvThreadingWin32::~CNvThreadingWin32()
{
    timeEndPeriod(m_uTimerResolution);
}

NvResult CNvThreadingWin32::MutexCreate(Handle *puMutexHandle)
{
    CRITICAL_SECTION *pCriticalSection = new CRITICAL_SECTION;

    if (!pCriticalSection)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    BOOL ret = InitializeCriticalSectionAndSpinCount(pCriticalSection, 0x8000000fU);

    if (!ret)
    {
#if _DEBUG
        DWORD err = GetLastError();
        NV_ASSERT(err == 0);
#endif
        InitializeCriticalSection(pCriticalSection);
    }

    *puMutexHandle = reinterpret_cast<Handle>(pCriticalSection);
    return RESULT_OK;
}

NvResult CNvThreadingWin32::MutexAcquire(Handle uMutexHandle)
{
    CRITICAL_SECTION *pCriticalSection = reinterpret_cast<CRITICAL_SECTION *>(uMutexHandle);

    EnterCriticalSection(pCriticalSection);

    return RESULT_OK;
}

NvResult CNvThreadingWin32::MutexTryAcquire(Handle uMutexHandle)
{
    CRITICAL_SECTION *pCriticalSection = reinterpret_cast<CRITICAL_SECTION *>(uMutexHandle);

    if (!TryEnterCriticalSection(pCriticalSection))
    {
        return RESULT_FALSE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingWin32::MutexRelease(Handle uMutexHandle)
{
    CRITICAL_SECTION *pCriticalSection = reinterpret_cast<CRITICAL_SECTION *>(uMutexHandle);

    LeaveCriticalSection(pCriticalSection);

    return RESULT_OK;
}

NvResult CNvThreadingWin32::MutexDestroy(Handle *puMutexHandle)
{
    CRITICAL_SECTION *pCriticalSection = reinterpret_cast<CRITICAL_SECTION *>(*puMutexHandle);

    DeleteCriticalSection(pCriticalSection);

    delete pCriticalSection;
    *puMutexHandle = NV_HANDLE_INVALID;

    return RESULT_OK;
}

NvResult CNvThreadingWin32::EventCreate(Handle *puEventHandle, bool bManual, bool bSet)
{
    HANDLE hEvent = CreateEvent(NULL, bManual, bSet, NULL);

    if (hEvent == NULL)
    {
        *puEventHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    *puEventHandle = reinterpret_cast<Handle>(hEvent);
    return RESULT_OK;
}

NvResult CNvThreadingWin32::EventWait(Handle uEventHandle, U32 uTimeoutMs)
{
    HANDLE hEvent = reinterpret_cast<HANDLE>(uEventHandle);

    switch (WaitForSingleObject(hEvent, uTimeoutMs))
    {
        case WAIT_OBJECT_0:
            return RESULT_OK;

        case WAIT_TIMEOUT:
            return RESULT_TIMEOUT;

        default:
            return RESULT_INVALID_HANDLE;
    }
}

NvResult CNvThreadingWin32::EventSet(Handle uEventHandle)
{
    HANDLE hEvent = reinterpret_cast<HANDLE>(uEventHandle);

    if (!SetEvent(hEvent))
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingWin32::EventReset(Handle uEventHandle)
{
    HANDLE hEvent = reinterpret_cast<HANDLE>(uEventHandle);

    if (!ResetEvent(hEvent))
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingWin32::EventDestroy(Handle *puEventHandle)
{
    HANDLE hEvent = reinterpret_cast<HANDLE>(*puEventHandle);

    if (!CloseHandle(hEvent))
    {
        return RESULT_INVALID_HANDLE;
    }

    *puEventHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

NvResult CNvThreadingWin32::SemaphoreCreate(Handle *puSemaphoreHandle, U32 uInitCount, U32 uMaxCount)
{
    HANDLE hSemaphore = CreateSemaphore(NULL, uInitCount, uMaxCount, NULL);

    if (hSemaphore == NULL)
    {
        *puSemaphoreHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    *puSemaphoreHandle = reinterpret_cast<Handle>(hSemaphore);
    return RESULT_OK;
}

NvResult CNvThreadingWin32::SemaphoreIncrement(Handle uSemaphoreHandle)
{
    HANDLE hSemaphore = reinterpret_cast<HANDLE>(uSemaphoreHandle);

    if (!ReleaseSemaphore(hSemaphore, 1, NULL))
    {
        return RESULT_INVALID_HANDLE;
    }

    return RESULT_OK;
}

NvResult CNvThreadingWin32::SemaphoreDecrement(Handle uSemaphoreHandle, U32 timeout_ms)
{
    HANDLE hSemaphore = reinterpret_cast<HANDLE>(uSemaphoreHandle);

    switch (WaitForSingleObject(hSemaphore, timeout_ms))
    {
        case WAIT_OBJECT_0:
            return RESULT_OK;

        case WAIT_TIMEOUT:
            return RESULT_TIMEOUT;

        default:
            return RESULT_INVALID_HANDLE;
    }
}

NvResult CNvThreadingWin32::SemaphoreDestroy(Handle *puSemaphoreHandle)
{
    HANDLE hSemaphore = reinterpret_cast<HANDLE>(*puSemaphoreHandle);

    if (!CloseHandle(hSemaphore))
    {
        return RESULT_INVALID_HANDLE;
    }

    *puSemaphoreHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

NvResult CNvThreadingWin32::TimerCreate(Handle *puTimerHandle, bool (*pFunc)(void *pParam), void *pParam, U32 uTimeMs, U32 uPeriodMs)
{
    CNvTimerData *pTimerData = new CNvTimerData;

    if (!pTimerData)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    pTimerData->pFunc = pFunc;
    pTimerData->pParam = pParam;
    pTimerData->bFirstEvent = true;
    pTimerData->uPeriodMs = uPeriodMs;
    pTimerData->uTimerResolution = m_uTimerResolution;

    pTimerData->hTimer = timeSetEvent(uTimeMs, m_uTimerResolution, TimerFunc, reinterpret_cast<DWORD>(pTimerData), TIME_ONESHOT);

    if (!pTimerData->hTimer)
    {
        delete pTimerData;
        *puTimerHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    *puTimerHandle = reinterpret_cast<Handle>(pTimerData);
    return RESULT_OK;
}

NvResult CNvThreadingWin32::TimerDestroy(Handle *puTimerHandle)
{
    if (!*puTimerHandle)
    {
        return RESULT_INVALID_HANDLE;
    }

    CNvTimerData *pTimerData = reinterpret_cast<CNvTimerData *>(*puTimerHandle);

    if (pTimerData->hTimer != NULL)
    {
        timeKillEvent(pTimerData->hTimer);
    }

    delete pTimerData;
    *puTimerHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

void CALLBACK CNvThreadingWin32::TimerFunc(UINT, UINT, DWORD_PTR dwUser, DWORD_PTR, DWORD_PTR)
{
    CNvTimerData *pTimerData = reinterpret_cast<CNvTimerData *>(dwUser);

    if (pTimerData->bFirstEvent)
    {
        pTimerData->bFirstEvent = false;

        if (pTimerData->uPeriodMs)
        {
            pTimerData->hTimer = timeSetEvent(pTimerData->uPeriodMs, pTimerData->uTimerResolution, TimerFunc, dwUser, TIME_PERIODIC);
            _ASSERT(pTimerData->hTimer != NULL);
        }
        else
        {
            pTimerData->hTimer = NULL;
        }
    }

    if (!pTimerData->pFunc(pTimerData->pParam))
    {
        timeKillEvent(pTimerData->hTimer);
    }
}

NvResult CNvThreadingWin32::ThreadCreate(Handle *puThreadHandle, U32(*pFunc)(void *pParam), void *pParam, S32 sPriority)
{
    CNvThreadData *pThreadData = new CNvThreadData;

    if (!pThreadData)
    {
        return RESULT_OUT_OF_HANDLES;
    }

    pThreadData->pFunc = pFunc;
    pThreadData->pParam = pParam;

    pThreadData->hThread = CreateThread(NULL, 0, ThreadFunc, pThreadData, 0, &pThreadData->dwThreadId);

    if (pThreadData->hThread == NULL)
    {
        delete pThreadData;
        *puThreadHandle = NV_HANDLE_INVALID;
        return RESULT_OUT_OF_HANDLES;
    }

    if (sPriority != NV_THREAD_PRIORITY_NORMAL)
    {
        ThreadPrioritySet(pThreadData, sPriority);
    }

    *puThreadHandle = reinterpret_cast<Handle>(pThreadData);
    return RESULT_OK;
}

NvResult CNvThreadingWin32::ThreadPriorityGet(Handle uThreadHandle, S32 &rsPriority)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(uThreadHandle);

    int iPriority = GetThreadPriority(pThreadData->hThread);

    if (iPriority == THREAD_PRIORITY_ERROR_RETURN)
    {
        return RESULT_FAIL;
    }

    rsPriority = iPriority;

    return RESULT_OK;
}

NvResult CNvThreadingWin32::ThreadPrioritySet(Handle uThreadHandle, S32 sPriority)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(uThreadHandle);

    if (!SetThreadPriority(pThreadData->hThread, sPriority))
    {
        return RESULT_FAIL;
    }

    return RESULT_OK;
}

NvResult CNvThreadingWin32::ThreadDestroy(Handle *puThreadHandle)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(*puThreadHandle);

    if (WaitForSingleObject(pThreadData->hThread, INFINITE) != WAIT_OBJECT_0)
    {
        return RESULT_INVALID_HANDLE;
    }

    if (!CloseHandle(pThreadData->hThread))
    {
        return RESULT_INVALID_HANDLE;
    }

    delete pThreadData;
    *puThreadHandle = NV_HANDLE_INVALID;
    return RESULT_OK;
}

bool CNvThreadingWin32::ThreadIsCurrent(Handle uThreadHandle)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(uThreadHandle);

    return (pThreadData->dwThreadId == GetCurrentThreadId());
}

DWORD WINAPI CNvThreadingWin32::ThreadFunc(LPVOID lpParameter)
{
    CNvThreadData *pThreadData = static_cast<CNvThreadData *>(lpParameter);
    U32 uResult = pThreadData->pFunc(pThreadData->pParam);
    return static_cast<DWORD>(uResult);
}

HANDLE CNvThreadingWin32::ThreadGetHandle(Handle hThreadHandle) const
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(hThreadHandle);
    return pThreadData->hThread;
}

U32 CNvThreadingWin32::GetTicksMs()
{
    return static_cast<U32>(timeGetTime());
}

U32 CNvThreadingWin32::GetThreadID(Handle hThreadHandle)
{
    CNvThreadData *pThreadData = reinterpret_cast<CNvThreadData *>(hThreadHandle);
    return pThreadData->dwThreadId;
}

