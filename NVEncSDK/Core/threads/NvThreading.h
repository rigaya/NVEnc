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
//! \file NvThreading.h
//! \brief Platform independent threading and thread synchronization.
//!
//! The \e INvThreading interface is an operating system abstraction for
//! threading and synchronization facilities. This allows a single application
//! that uses the features to be compiled to run on a number of platforms that
//! implement this interface.
//! The developer may use these methods directly or refer to
//! \e NvThreadingClasses.h for C++ wrappers that make using \e INvThreading
//! much more convenient.
//---------------------------------------------------------------------------

#ifndef _NV_OSAL_NV_THREADING_H
#define _NV_OSAL_NV_THREADING_H

#include <limits.h>

#include <include/NvTypes.h>
#include <include/NvResult.h>

//! \brief Interface for operating system threading abstraction.
//!
//! The methods in this interface provide threading and thread
//! syncrhonization facilities.
class INvThreading
{
    public:
        //! Get a pointer to the \e INvThreading class.
        static INvThreading *GetThreading();

        //! Class constants.
        enum
        {
            //! Used to specify an infinite timeout on blocking methods.
            NV_TIMEOUT_INFINITE = UINT_MAX,

            //! Used to specify a thread priority of normal. A lower priority
            //! thread is negative and a higher priority thread is positive.
            //! All priorities are level and only relevant within the context
            //! of the running application. Choose wisely.
            NV_THREAD_PRIORITY_NORMAL = 0
        };

        //! Generic handle used for Mutex, Event, Semaphore, etc.
        typedef void *Handle;

        //! Platform defined constant for an invalid handle.
        static Handle const NV_HANDLE_INVALID;

        //////////////////////////////////////////////////////////////////////
        //! \name Mutexes
        //! Mutexes can be used to provide mutual exclusion of some data. For
        //! example, with more than one thread in the system running, it is
        //! possible that both thread could simultaneously try to access a
        //! data structure. In this case, the data within the data structure
        //! could be partially modified by one thread while another thread
        //! is reading. To avoid getting corrupted data, access to this
        //! data structure should be protected so a thread can complete
        //! its change before another accesses the data. It is common to use
        //! mutex for this. Access to the data structure should be protected
        //! by the thread Acquiring a lock beforing accessing the data. Once
        //! the changes have been made, the lock should be Released.
        //////////////////////////////////////////////////////////////////////
        //@{

        //! Create a mutex and assigned to \e Handle.
        virtual NvResult MutexCreate(Handle *puMutexHandle) = 0;

        //! Acquire the mutex lock. This function will block until
        //! another thread gives up its lock. The mutex allows
        //! recursive locking. In other words, if this is called
        //! from the same thread that has already acquired this lock
        //! then this function will increment the lock count. The
        //! lock needs to be released the same number of times it
        //! has been acquired.
        virtual NvResult MutexAcquire(Handle uMutexHandle) = 0;

        //! Try to acquire the lock. If another thread has this locked
        //! the function will return \e RESULT_FALSE. Otherwise it will
        //! acquire the lock in the same way as \e MutexAcquire.
        virtual NvResult MutexTryAcquire(Handle uMutexHandle) = 0;

        //! Release a lock already acquired.
        virtual NvResult MutexRelease(Handle uMutexHandle) = 0;

        //! Destroy the mutex.
        virtual NvResult MutexDestroy(Handle *puMutexHandle) = 0;

        //@}

        //////////////////////////////////////////////////////////////////////
        //! \name Events
        //! Events are used for one thread to signal another. A thread may
        //! wait on the event. It will sleep if the event hasn't been signalled
        //! yet. When the event is signalled the waiting thread will wake up.
        //////////////////////////////////////////////////////////////////////
        //@{

        //! Create an event.
        //! \param puEventHandle Event handle returned by the function.
        //! \param bManual Flag specifies whether the signalled state of the event
        //! is reset automatically to non-signalled when another thread successfully
        //! called \e EventWait.
        //! \param bSet Flag specifies the default signalled state of the event on
        //! creation.
        virtual NvResult EventCreate(Handle *puEventHandle, bool bManual, bool bSet) = 0;

        //! Wait for the event to be signalled.
        //! \param uEventHandle Handle provided be \e EventCreate.
        //! \param uTimeoutMs How long should the thread wait for the event
        //! to be signalled. Units are in milliseconds.
        //! \return NvResult Returns an error or the state of the event.
        //! \retval RESULT_OK The event was signalled.
        //! \retval RESULT_TIMEOUT The event was not signalled within the
        //! \e uTimeoutMs period.
        //! \retval RESULT_INVALID_HANDLE The \e uEventHandle was invalid.
        virtual NvResult EventWait(Handle uEventHandle, U32 uTimeoutMs) = 0;

        //! Signal the event.
        virtual NvResult EventSet(Handle uEventHandle) = 0;

        //! Clear the signalled state of the event.
        virtual NvResult EventReset(Handle uEventHandle) = 0;

        //! Destroy the event.
        virtual NvResult EventDestroy(Handle *puEventHandle) = 0;

        //@}

        //////////////////////////////////////////////////////////////////////
        //! \name Semaphores
        //! Semaphores are similar to \e Mutexes, except they allow more than
        //! one thread to "acquire" the lock. A sempahore is created with an
        //! initial count. One ore more threads can "acquire" the semphore
        //! by called \e SemaphoreDecrement. When the initial count of the
        //! sempahore has been decremented to zero, subsequent calls to
        //! \e SemaphoreDecrement will block the caller. A thread that has
        //! previously called \e SempaphoreDecrement can call \e SemaphoreIncrement
        //! to "release" their lock.
        //////////////////////////////////////////////////////////////////////
        //@{

        //! Create a semaphore.
        //! \param puSemaphoreHandle Semaphore handle returned by the function.
        //! \param uInitCount Initial count of the semaphore.
        //! \param uMaxCount Maximum number of the semaphore count. In other words
        //! the maximum number of "locks" that can be acuired. A value of 1 would
        //! make the semaphore exactly like a mutex.
        virtual NvResult SemaphoreCreate(Handle *puSemaphoreHandle, U32 uInitCount, U32 uMaxCount) = 0;

        //! Increment the semaphore count by 1. This is like "releasing" a lock
        //! that was previously acquired by \e SemaphoreDecrement.
        virtual NvResult SemaphoreIncrement(Handle uSemaphoreHandle) = 0;

        //! Decrement the sempahore count by 1. This is like "acquiring" a lock.
        //! If the semaphore count is 0, this function may block as determined by
        //! the \e uTimeoutMs parameter.
        //! \param uSemaphoreHandle Handle of the semaphore to decrement.
        //! \param uTimeoutMs Specifies the maximum time to wait for another
        //! thread to increment the semaphore count above 0.
        virtual NvResult SemaphoreDecrement(Handle uSemaphoreHandle, U32 uTimeoutMs) = 0;

        //! Destroy the semaphore.
        virtual NvResult SemaphoreDestroy(Handle *puSemaphoreHandle) = 0;

        //@}

        //////////////////////////////////////////////////////////////////////
        //! \name Timers
        //! Timers can be used to trigger an event or perform some action at
        //! a specified time or period.
        //////////////////////////////////////////////////////////////////////
        //@{

        //! Create a timer.
        //! \param puTimerHandle The new timer handle returned by this method.
        //! \param pFunc The function the timer should call when the timer expires.
        //! The implementation of this function should return true to allow the
        //! timer to keep firing every \e uPeriosMs. If the function returns false
        //! the timer will no longer call back into \e pFunc.
        //! \param pParam Pointer to client specific data.
        //! \param uTimeMs The initial expire time of the timer.
        //! \param uPeriodMs The incremental expire time of the timer. In other words,
        //! when the initial timer expires, \e uPeriosMs will be used to set the
        //! new expire time. A \e uPeriodMs of 0 means that the timer should only
        //! fire once.
        virtual NvResult TimerCreate(Handle *puTimerHandle, bool (*pFunc)(void *pParam), void *pParam, U32 uTimeMs, U32 uPeriodMs) = 0;

        //! Destroy the timer.
        virtual NvResult TimerDestroy(Handle *puTimerHandle) = 0;

        //@}

        //////////////////////////////////////////////////////////////////////
        //! \name Threads.
        //! Threads are at the heart of the module. A thread allows code to
        //! be run simultaneously with other threads. So, if one thread blocks
        //! or stalls, the other threads will continue running.
        //////////////////////////////////////////////////////////////////////
        //@{

        //! Create a thread.
        //! \param puThreadHandle Thread handle returned by method.
        //! \param pFunc Entry function to be called by new thread.
        //! \param pParam Client data that is passed back on the thread function.
        //! \param sPriority Thread priority. Use \e NV_THREAD_PRIORITY_NORMAL or
        //! some offset from \e NV_THREAD_PRIORITY NORMAL. Lower numbers represent
        //! lower thread priority. Higher numbers represent higher thread priority. Don't
        //! forget that it's the relative numbers that matter, not the absolute
        //! (relative meaning relative to the other threads you've already created).
        virtual NvResult ThreadCreate(Handle *puThreadHandle, U32(*pFunc)(void *pParam), void *pParam, S32 sPriority) = 0;

        //! Get the thread priority of the thread specified in \e uThreadHandle.
        virtual NvResult ThreadPriorityGet(Handle uThreadHandle, S32 &rsPriority) = 0;

        //! Set the thread priority of the thread specified in \e uThreadHandle.
        virtual NvResult ThreadPrioritySet(Handle uThreadHandle, S32 sPriority) = 0;

        //! Destroy the thread.
        //! \warning This function will block until the thread to be destroyed
        //! had completed execution. Make sure you design your software so this
        //! can happen. This method will \b not forcibly destroy a running thread.
        virtual NvResult ThreadDestroy(Handle *puThreadHandle) = 0;

        //! Return true if the given thread is the currently running thread.
        virtual bool ThreadIsCurrent(Handle uThreadHandle) = 0;

        //@}

        //////////////////////////////////////////////////////////////////////
        // Misc.
        //////////////////////////////////////////////////////////////////////

        //! Get the number of milliseconds that has passed since the application
        //! was started.
        //! \warning This function will wrap every 49 days. Make sure your math
        //! handles this correctly.
        virtual U32 GetTicksMs() = 0;

        //! Get thread ID from Handle
        virtual U32 GetThreadID(Handle hThreadHandle) = 0;
    protected:
        virtual ~INvThreading() { }
};

#endif
