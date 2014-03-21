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
// NvThreadingWin32.cpp
//
// Win32 implementation of the IThread interface.
//---------------------------------------------------------------------------

#ifndef _NV_OSAL_NV_THREADING_WIN32_H
#define _NV_OSAL_NV_THREADING_WIN32_H

#if defined(_WIN32_WINNT)
#  if _WIN32_WINNT < 0x0403
#    undef _WIN32_WINNT
#  endif
#endif
#if !defined(_WIN32_WINNT)
#  define _WIN32_WINNT 0x0403
#endif

#include <windows.h>
#include <crtdbg.h>
#include <threads/NvThreading.h>

class CNvThreadingWin32 : public INvThreading
{
    public:
        CNvThreadingWin32();
        virtual ~CNvThreadingWin32();

        //////////////////////////////////////////////////////////////////////
        // Mutex.
        //////////////////////////////////////////////////////////////////////

        virtual NvResult MutexCreate(Handle *puMutexHandle);
        virtual NvResult MutexAcquire(Handle uMutexHandle);
        virtual NvResult MutexTryAcquire(Handle uMutexHandle);
        virtual NvResult MutexRelease(Handle uMutexHandle);
        virtual NvResult MutexDestroy(Handle *puMutexHandle);

        //////////////////////////////////////////////////////////////////////
        // Events.
        //////////////////////////////////////////////////////////////////////

        virtual NvResult EventCreate(Handle *puEventHandle, bool bManual, bool bSet);
        virtual NvResult EventWait(Handle uEventHandle, U32 uTimeoutMs);
        virtual NvResult EventSet(Handle uEventHandle);
        virtual NvResult EventReset(Handle uEventHandle);
        virtual NvResult EventDestroy(Handle *puEventHandle);

        //////////////////////////////////////////////////////////////////////
        // Semaphores.
        //////////////////////////////////////////////////////////////////////

        virtual NvResult SemaphoreCreate(Handle *puSemaphoreHandle, U32 uInitCount, U32 uMaxCount);
        virtual NvResult SemaphoreIncrement(Handle uSemaphoreHandle);
        virtual NvResult SemaphoreDecrement(Handle uSemaphoreHandle, U32 uTimeoutMs);
        virtual NvResult SemaphoreDestroy(Handle *puSemaphoreHandle);

        //////////////////////////////////////////////////////////////////////
        // Timers.
        //////////////////////////////////////////////////////////////////////

        virtual NvResult TimerCreate(Handle *puTimerHandle, bool (*pFunc)(void *pParam), void *pParam, U32 uTimeMs, U32 uPeriodMs);
        virtual NvResult TimerDestroy(Handle *puTimerHandle);

        //////////////////////////////////////////////////////////////////////
        // Threads.
        //////////////////////////////////////////////////////////////////////

        virtual NvResult ThreadCreate(Handle *puThreadHandle, U32(*pFunc)(void *pParam), void *pParam, S32 sPriority);
        virtual NvResult ThreadPriorityGet(Handle uThreadHandle, S32 &rsPriority);
        virtual NvResult ThreadPrioritySet(Handle uThreadHandle, S32 sPriority);
        virtual NvResult ThreadDestroy(Handle *puThreadHandle);

        virtual bool ThreadIsCurrent(Handle uThreadHandle);

        //////////////////////////////////////////////////////////////////////
        // Misc.
        //////////////////////////////////////////////////////////////////////

        virtual U32 GetTicksMs();

        virtual U32 GetThreadID(Handle hThreadHandle);

        //////////////////////////////////////////////////////////////////////
        // Win32 specific.
        //////////////////////////////////////////////////////////////////////

        HANDLE ThreadGetHandle(Handle hThreadHandle) const;

    private:
        UINT m_uTimerResolution;

        struct CNvTimerData
        {
            bool (*pFunc)(void *);
            void    *pParam;
            MMRESULT hTimer;
            bool     bFirstEvent;
            UINT     uPeriodMs;
            UINT     uTimerResolution;
        };

        static void CALLBACK TimerFunc(UINT uID, UINT uMsg, DWORD_PTR dwUser, DWORD_PTR dw1, DWORD_PTR dw2);

        struct CNvThreadData
        {
            U32(*pFunc)(void *);
            void        *pParam;
            HANDLE       hThread;
            DWORD        dwThreadId;
        };

        static DWORD WINAPI ThreadFunc(LPVOID lpParameter);
};

#endif
