#ifndef _FRAME_QUEUE
#define _FRAME_QUEUE
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma warning(push)
#pragma warning(disable: 4201)
#include "dynlink_nvcuvid.h"
#pragma warning(pop)

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>

#else
  #include <unistd.h>
  #include <string.h>
  #include <pthread.h>
  typedef pthread_mutex_t CRITICAL_SECTION;
  typedef void* HANDLE;

  #define Sleep(x) usleep(1000*x);
#endif

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

class FrameQueue
{
public:
    static const int cnMaximumSize = 24; // MAX_FRM_CNT;

    FrameQueue(CUvideoctxlock ctxLock);

    virtual
   ~FrameQueue();

    void
    waitForQueueUpdate();

    void
    enter_CS(CRITICAL_SECTION *pCS);

    void
    leave_CS(CRITICAL_SECTION *pCS);

    void
    set_event(HANDLE event);

    void
    reset_event(HANDLE event);

#pragma warning(push)
#pragma warning(disable: 4100)
    virtual void
    init(int frameWidth, int frameHeight) { }
#pragma warning(pop)

    virtual void
    enqueue(const void * pData) = 0;

    // Deque the next frame.
    // Parameters:
    //      pDisplayInfo - New frame info gets placed into this object.
    //          Note: This pointer must point to a valid struct. The method
    //          does not create memory for this.
    // Returns:
    //      true, if a new frame was returned,
    //      false, if the queue was empty and no new frame could be returned.
    //          In that case, pPicParams doesn't contain valid data.
    virtual bool
    dequeue(void * pData) = 0;

    virtual void
    releaseFrame(const void * pPicParams) = 0;

    bool
    isInUse(int nPictureIndex)
    const;

    bool
    isEndOfDecode()
    const;

    void
    endDecode();

    // Spins until frame becomes available or decoding
    // gets canceled.
    // If the requested frame is available the method returns true.
    // If decoding was interupted before the requested frame becomes
    // available, the method returns false.
    bool
    waitUntilFrameAvailable(int nPictureIndex);


    size_t getPitch() { return nPitch; }

    bool isEmpty() { return nFramesInQueue_ == 0; }

    bool nearFull() { return nFramesInQueue_ >= cnMaximumSize - 8; }

protected:
    void
    signalStatusChange();

    HANDLE hEvent_;
    CRITICAL_SECTION    oCriticalSection_;
    volatile int        nReadPosition_;
    volatile int        nWritePosition_;

    volatile int        nFramesInQueue_;
    volatile int        aIsFrameInUse_[cnMaximumSize];
    volatile int        bEndOfDecode_;

    CUvideoctxlock      m_ctxLock;
    size_t              nPitch;
};

class CUVIDFrameQueue: public FrameQueue {

public:
    CUVIDFrameQueue(CUvideoctxlock ctxLock);
    ~CUVIDFrameQueue();

    virtual void enqueue(const void * pData);
    virtual bool dequeue(void * pData);
    virtual void releaseFrame(const void * pPicParams);

protected:
    CUVIDPARSERDISPINFO aDisplayQueue_[cnMaximumSize];
};


#endif
