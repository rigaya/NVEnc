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
//! \file NvThreadingClasses.h
//! \brief Wrapper classes for the \e INvThreading interface methods.
//!
//! The classes in this module provide convenient ways to use the
//! \e INvThreading methods so that you don't need to remember to Create
//! and Destroy them. It is highly recommended you use these methods rather
//! than the \e INvThreading methods directly.
//---------------------------------------------------------------------------

#ifndef _NV_OSAL_NV_THREADING_CLASSES_H
#define _NV_OSAL_NV_THREADING_CLASSES_H

#include <threads/NvThreading.h>

//! Mutex class.
class CNvMutex
{
    public:
        CNvMutex();
        ~CNvMutex();

        //! Acquire the mutex lock. Will block if already locked by
        //! another thread.
        void Acquire() const;

        //! Try to acquire the mutex. Will never block.
        //! \retval true Lock was acquired.
        //! \retval false Lock wasn't acquired.
        bool TryAcquire() const;

        //! Release the mutex lock.
        void Release() const;

        //! Get the mutex handle.
        operator INvThreading::Handle() const;

    private:
        INvThreading        *m_pThreading;
        INvThreading::Handle m_uHandle;
};

//! \brief Auto mutex class.
///
/// Creating this object will automatically
/// acquire the lock. When the object is destoyed the lock will
/// be released.
class CNvAutoMutex
{
    public:
        //! Automatically locks the mutex parameter.
        CNvAutoMutex(const CNvMutex &);

        //! Automatically releases the mutex.
        ~CNvAutoMutex();

        //! Release the mutex manually.
        void Release();

    private:
        const CNvMutex &m_Mutex;

        bool m_bReleased;
};

//! Event class.
class CNvEvent
{
    public:
        //! Create an event class.
        //! \param bManual Flag specifies whether this event will automatically clear
        //! the signalled state when a thread has a successful \e Wait().
        //! \param bSet Flag specifies the initial signal condition of the event.
        CNvEvent(bool bManual = false, bool bSet = false);
        ~CNvEvent();

        //! Wait for the signal. Block for a maximum of \e uTimeoutMs.
        //! \retval false Signal was not set.
        //! \retval true Signal was set.
        bool Wait(U32 uTimeoutMs);

        //! Set the signal.
        void Set();

        //! Clear the signal.
        void Reset();

        //! Get the event handle.
        operator INvThreading::Handle() const;

    private:
        INvThreading        *m_pThreading;
        INvThreading::Handle m_uHandle;
};

//! Semaphore class.
class CNvSemaphore
{
    public:
        //! Create a semaphore with initial count \e uInitCount and
        //! maximum count \e uMaxCount.
        CNvSemaphore(U32 uInitCount, U32 uMaxCount);
        ~CNvSemaphore();

        //! Increment the semaphore count.
        void Increment();

        //! Decrement the semaphore count. This will block
        //! for up to \e uTimeoutMs milliseconds if the count is
        //! already 0.
        //! \retval false The semaphore wasn't decremented.
        //! \retval true The sempahore was decremented.
        bool Decrement(U32 uTimeoutMs);

        //! Get the semaphore handle.
        operator INvThreading::Handle() const;

    private:
        INvThreading        *m_pThreading;
        INvThreading::Handle m_uHandle;
};

//! Timer class.
class CNvTimer
{
    public:
        //! Create a timer.
        //! \param pFunc If non-zero, this function will be called when
        //! the timer expires. If zero, use the \e TimerFunc() virtual
        //! method instead.
        //! The implementation of this function should return true to allow the
        //! timer to keep firing every \e uPeriosMs. If the function returns false
        //! the timer will no longer call back into \e pFunc.
        //! \param pUserData Client data returned back to the callback
        //! \e pFunc function.
        CNvTimer(bool (*pFunc)(void *pUserData) = 0, void *pUserData = 0);
        virtual ~CNvTimer();

        //! Configure the timer with a new pFunc and pUserData.
        void TimerConfig(bool (*pFunc)(void *pUserData), void *pUserData = 0);

        //! Start the timer. The timer will call back every \e uPeriodMs.
        void TimerStart(U32 uPeriodMs);

        //! Start the timer. The first timer will fire at \e uTimeMs and
        //! all subequent callbacks will be called at \e uPeriosMs.
        void TimerStart(U32 uTimeMs, U32 uPeriodMs);

        //! Pause a currently running timer.
        //! \param bPause true to pause the timer. false to unpause.
        void TimerPause(bool bPause);

        //! Stop the running timer.
        void TimerStop();

        //! Get the handle to the running timer.
        operator INvThreading::Handle() const;

    protected:
        //! Optional virtual function to overload. This function
        //! is called each time the timer fires.
        //! The implementation of this function should return true to allow the
        //! timer to keep firing every \e uPeriosMs. If the function returns false
        //! the timer will no longer call back into \e TimerFunc.
        virtual bool TimerFunc();

    private:
        INvThreading        *m_pThreading;
        INvThreading::Handle m_uHandle;

        CNvMutex m_Mutex;

        U32 m_uStartTimeMs;
        U32 m_uStopTimeMs;
        U32 m_uPeriodMs;
        U32 m_uTimeMs;

        bool m_bPaused;

        bool (*m_pFunc)(void *pUserData);
        void *m_pUserData;

        static bool m_FuncWrapperStatic(void *);
        bool        m_FuncWrapper();
};

//! Thread class.
class CNvThread
{
    public:
        //! Create a thread that calls the \e ThreadFunc() virtual method.
        //! \param pszThreadName Name of the thread for debugging purposes.
        //! \param sPriority Priority of the new thread.
        //! \param bOneShot Controls how this thread class operates.
        //! If set to true, then \e ThreadFunc() will only be called
        //! once. If set to false (default) then \e ThreadFunc() will
        //! be called repeatedly whenever the thread wakes on \e ThreadTrigger()
        //! or if \e ThreadFunc() returns true.
        CNvThread(
            const char *pszThreadName,
            S32 sPriority = INvThreading::NV_THREAD_PRIORITY_NORMAL,
            bool bOneShot = false
        );

        //! Create a thread that calls \e pFunc once.
        //! \param pszThreadName Name of the thread for debugging purposes.
        //! \param pFunc Thread function to be called.
        //! \param pUserData Client data passed to \e pFunc.
        //! \param sPriority Priority of the new thread.
        CNvThread(
            const char *pszThreadName,
            bool (*pFunc)(void *pUserData),
            void *pUserData,
            S32 sPriority = INvThreading::NV_THREAD_PRIORITY_NORMAL
        );

        virtual ~CNvThread();

        //! Auto lock class.
        class CAutoLock
        {
            public:
                //! Automatically calls \e ThreadLock().
                CAutoLock(const CNvThread *pThread);

                //! Automatically calls \e ThreadUnlock().
                ~CAutoLock();

                //! Release the lock manually.
                void Release();

            private:
                const CNvThread *m_pThread;

                bool m_bReleased;
        };

        //! Start the thread running.
        //! \param bSync If true, this function will block until
        //! the thread has started running. If false, the function
        //! will return immediately.
        virtual void ThreadStart(bool bSync = false);

        //! Quit the thread. This will block until the thread is quit.
        virtual void ThreadQuit();

        //! Wake up a sleeping thread.
        virtual void ThreadTrigger();

        //! Lock the thread from running. This can be used to temporarily
        //! stop the thread running while another thread accesses shared data.
        //! This call will block until the lock is acquired.
        virtual void ThreadLock() const;

        //! Unlock a previously locked thread.
        virtual void ThreadUnlock() const;

        //! Adjust the thread priority.
        virtual void ThreadPriority(S32 sPriority);

        //! Get the name of the thread.
        virtual const char *ThreadName() const;

        //! Get the handle to the thread.
        operator INvThreading::Handle() const;

        // return the priority
        S32 GetThreadPriority()
        {
            return m_sPriority;
        }

    protected:
        //! Overload this function to be called when the thread is started.
        virtual bool ThreadInit();

        //! Overload this function to be called when the thread runs.
        //! \retval true I have more work to do. Call me back immediately.
        //! \retval false Call me back when someone calls \e ThreadTrigger().
        virtual bool ThreadFunc();

        //! Overload this function to be called when the thread quits.
        virtual bool ThreadFini();

        //! Has someone asked for the thread to be quit?
        //! \retval true Someone has called \e ThreadQuit().
        //! \retval false Thread should continue running.
        virtual bool ThreadIsQuit() const;

        //! Put the thread to sleep for up to \e uTickMs or until someone calls
        //! \e ThreadTrigger().
        virtual bool ThreadBlock(U32 uTickMs);

    private:
        INvThreading        *m_pThreading;
        INvThreading::Handle m_hThread;

        bool     m_bQuit;
        CNvMutex m_Mutex;
        CNvEvent m_Event;
        CNvEvent m_EventStart;
        bool     m_bSyncStart;
        bool     m_bOneShot;
        S32      m_sPriority;

        bool (*m_pFunc)(void *pUserData);
        void *m_pUserData;

        const char *m_szThreadName;

        static U32 m_FuncStatic(void *vpParam);
        U32        m_Func();
};

//! Queue class. Thread safe queuing.
template<class T, U32 L = 1>
class CNvQueue
{
    public:
        //! Optional callback interface.
        class ICallback
        {
            public:
                virtual ~ICallback() { }

                //! The queue is no longer empty.
                virtual void OnQueueNotEmpty(const CNvQueue *pNvQueue) = 0;

                //! The queue is no longer full.
                virtual void OnQueueNotFull(const CNvQueue *pNvQueue) = 0;
        };

        //! Create a queue of size \e uSize.
        CNvQueue(U32 uSize = L);

        //! Create a queue of size \e uSize and call back \e pCallback
        //! on queue not empty and queue not full conditions.
        CNvQueue(ICallback *pCallback, U32 uSize = L);

        ~CNvQueue();

        //! Add an element to the queue. This call will block for up to
        //! \e uTimeoutMs if the queue is full.
        //! \retval true The item was added to the queue.
        //! \retval false The item was not added to the queue (timeout).
        bool Add(const T &, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE);

        //! Add a new queue item to the front of the queue. Useful for high priority
        //! commands. This call will block for up to \e uTimeoutMs if the queue is full.
        //! \retval true The item was added to the queue.
        //! \retval false The item was not added to the queue (timeout).
        bool AddFront(const T &, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE);

        //! Remove an item from the queue. This call will block for up to
        //! \e uTimeoutMs or until a new item arrives in the queue.
        //! \retval true An item was removed from the queue.
        //! \retval false A timeout occured and no item was removed.
        bool Remove(T &, U32 uTimeoutMs = 0);

        //! Peek at the item at the head of the queue. Warning, this function
        //! should only be used by designs where there is only one consumer.
        //! Using Peek() in a multi-consumer design will cause problems because
        //! the result of Peek() is not likely to be accurate for long.
        //! \retval true There was something at the head.
        //! \retval false The queue is empty.
        bool Peek(T &);

        //! Pop the first item off the queue. This function is useful in conjunction
        //! with the \e Peek() call. The same warning applies. Do not use this
        //! function in a multi-consumer design because what you last Peek()ed is.
        //! not necessarily what you're going to Pop() and that could lead to all
        //! kinds of trouble.
        //! \retval true An item was popped of the front of the queue.
        //! \retval false The queue is empty.
        bool Pop();

        //! Clear all items in the queue.
        void Clear();

        //! Get the number of items in the queue. Warning, this function
        //! should be used carefully since the result is only accurate at
        //! the point of asking. In a multi-threaded design the result is
        //! unlikely to be accurate for long. Do not use this result for
        //! logic descision making. Useful only for logging and debugging.
        U32 GetCount();

    private:
        T *m_pBuffer;

        CNvMutex m_Mutex;

        CNvSemaphore m_SemQueueFull;
        CNvSemaphore m_SemQueueEmpty;

        U32 m_uSize;
        U32 m_uCount;
        U32 m_uReadIndex;
        U32 m_uWriteIndex;

        ICallback *m_pCallback;
};

template<class T, U32 L>
CNvQueue<T, L>::CNvQueue(U32 uSize) :
    m_Mutex(),
    m_SemQueueFull(uSize, uSize),
    m_SemQueueEmpty(0, uSize),
    m_uSize(uSize),
    m_uCount(0),
    m_uReadIndex(0),
    m_uWriteIndex(0),
    m_pCallback(0)
{
    assert(m_uSize);
    m_pBuffer = new T[m_uSize];
}

template<class T, U32 L>
CNvQueue<T, L>::CNvQueue(ICallback *pCallback, U32 uSize) :
    m_Mutex(),
    m_SemQueueFull(uSize, uSize),
    m_SemQueueEmpty(0, uSize),
    m_uSize(uSize),
    m_uCount(0),
    m_uReadIndex(0),
    m_uWriteIndex(0),
    m_pCallback(pCallback)
{
    assert(m_uSize);
    m_pBuffer = new T[m_uSize];
}

template<class T, U32 L>
CNvQueue<T, L>::~CNvQueue()
{
    delete[] m_pBuffer;
}

template<class T, U32 L>
bool CNvQueue<T, L>::Add(const T &Item, U32 uTimeoutMs)
{
    if (!m_SemQueueFull.Decrement(uTimeoutMs))
    {
        return false;
    }

    bool bSignalNotEmpty = false;

    m_Mutex.Acquire();
    m_pBuffer[m_uWriteIndex] = Item;
    m_uWriteIndex++;

    if (m_uWriteIndex >= m_uSize)
    {
        m_uWriteIndex -= m_uSize;
    }

    if (m_uCount == 0)
    {
        bSignalNotEmpty = true;
    }

    m_uCount += 1;
    m_SemQueueEmpty.Increment();
    m_Mutex.Release();

    if (m_pCallback && bSignalNotEmpty)
    {
        m_pCallback->OnQueueNotEmpty(this);
    }

    return true;
}

template<class T, U32 L>
bool CNvQueue<T, L>::AddFront(const T &Item, U32 uTimeoutMs)
{
    if (!m_SemQueueFull.Decrement(uTimeoutMs))
    {
        return false;
    }

    bool bSignalNotEmpty = false;

    m_Mutex.Acquire();
    m_uReadIndex = (m_uReadIndex + m_uSize - 1);

    if (m_uReadIndex >= m_uSize)
    {
        m_uReadIndex -= m_uSize;
    }

    m_pBuffer[m_uReadIndex] = Item;

    if (m_uCount == 0)
    {
        bSignalNotEmpty = true;
    }

    m_uCount += 1;
    m_SemQueueEmpty.Increment();
    m_Mutex.Release();

    if (m_pCallback && bSignalNotEmpty)
    {
        m_pCallback->OnQueueNotEmpty(this);
    }

    return true;
}

template<class T, U32 L>
bool CNvQueue<T, L>::Remove(T &Item, U32 uTimeoutMs)
{
    if (!m_SemQueueEmpty.Decrement(uTimeoutMs))
    {
        return false;
    }

    bool bSignalNotFull = false;

    m_Mutex.Acquire();
    Item = m_pBuffer[m_uReadIndex];
    m_uReadIndex++;

    if (m_uReadIndex >= m_uSize)
    {
        m_uReadIndex -= m_uSize;
    }

    if (m_uCount == m_uSize)
    {
        bSignalNotFull = true;
    }

    m_uCount -= 1;
    m_SemQueueFull.Increment();
    m_Mutex.Release();

    if (m_pCallback && bSignalNotFull)
    {
        m_pCallback->OnQueueNotFull(this);
    }

    return true;
}

template<class T, U32 L>
bool CNvQueue<T, L>::Peek(T &Item)
{
    CNvAutoMutex lock(m_Mutex);

    if (!m_uCount)
    {
        return false;
    }

    Item = m_pBuffer[m_uReadIndex];
    return true;
}

template<class T, U32 L>
bool CNvQueue<T, L>::Pop()
{
    if (!m_SemQueueEmpty.Decrement(0))
    {
        return false;
    }

    bool bSignalNotFull = false;

    m_Mutex.Acquire();
    m_uReadIndex++;

    if (m_uReadIndex >= m_uSize)
    {
        m_uReadIndex -= m_uSize;
    }

    if (m_uCount == m_uSize)
    {
        bSignalNotFull = true;
    }

    m_uCount -= 1;
    m_SemQueueFull.Increment();
    m_Mutex.Release();

    if (m_pCallback && bSignalNotFull)
    {
        m_pCallback->OnQueueNotFull(this);
    }

    return true;
}

template<class T, U32 L>
void CNvQueue<T, L>::Clear()
{
    while (m_SemQueueEmpty.Decrement(0))
    {
        m_Mutex.Acquire();
        m_uReadIndex++;

        if (m_uReadIndex >= m_uSize)
        {
            m_uReadIndex -= m_uSize;
        }

        m_uCount -= 1;
        m_SemQueueFull.Increment();
        m_Mutex.Release();
    }
}

template<class T, U32 L>
U32 CNvQueue<T, L>::GetCount()
{
    CNvAutoMutex lock(m_Mutex);
    return m_uCount;
}

//! Worker thread class that binds a thread and a single queue.
template<class T, U32 L>
class CNvWorkerThread : private CNvThread
{
    public:
        //! Create a worker thread.
        //! \param pszThreadName Name of the thread for debugging purposes.
        //! \param sPriority Priority of the new thread.
        CNvWorkerThread(const char *pszThreadName, S32 sPriority = INvThreading::NV_THREAD_PRIORITY_NORMAL) : CNvThread(pszThreadName, sPriority, true) { }

        //! Start the worker thread.
        void Create()
        {
            ThreadStart();
        }

        //! Check on the worker thread.
        bool IsQuit()
        {
            return ThreadIsQuit();
        }

        //! Stop the worker thread.
        void Destroy()
        {
            ThreadQuit();
        }

        //! Queue a new task for the worker thread.
        bool QueueTask(const T &task, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE)
        {
            bool bResult = m_oQueue.Add(task, uTimeoutMs);

            if (bResult)
            {
                ThreadTrigger();
            }

            return bResult;
        }

        //! Get the next task to process from the task queue.
        bool GetTask(T &task, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE)
        {
            return m_oQueue.Remove(task, uTimeoutMs);
        }

        //! Empty the task queue.
        void EmptyCommandQueue()
        {
            m_oQueue.Clear();
        }

        //! ThreadFunc.
        virtual bool ThreadFunc() = 0;

    protected:
        virtual ~CNvWorkerThread() { }

    private:
        CNvQueue<T, L> m_oQueue;
};

//! Worker thread class that binds a thread and 2 queues.
template<class Ti, U32 Li, class To, U32 Lo>
class CNvWorkerThreadRQ : public CNvWorkerThread<Ti, Li>
{
    public:
        //! Create the worker thread.
        //! \param pszThreadName Name of the thread for debugging purposes.
        //! \param sPriority Priority of the new thread.
        CNvWorkerThreadRQ(const char *pszThreadName, S32 sPriority = INvThreading::NV_THREAD_PRIORITY_NORMAL) : CNvWorkerThread<Ti, Li>(pszThreadName, sPriority) { }

        //! Queue a response message.
        bool QueueResponse(const To &task, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE)
        {
            bool bResult = m_oResponseQueue.Add(task, uTimeoutMs);

            if (bResult)
            {
                CNvWorkerThreadRQ<Ti, Li, To, Lo>::ThreadTrigger();
            }

            return bResult;
        }

        //! Get the next response message to process.
        bool GetResponse(To &task, U32 uTimeoutMs = INvThreading::NV_TIMEOUT_INFINITE)
        {
            return m_oResponseQueue.Remove(task, uTimeoutMs);
        }

        //! Empty all the responses.
        void EmptyResponseQueue()
        {
            m_oResponseQueue.Clear();
        }

        //! ThreadFunc.
        virtual bool ThreadFunc() = 0;

    protected:
        virtual ~CNvWorkerThreadRQ() { }

    private:
        CNvQueue<To, Lo> m_oResponseQueue;
};

#endif
