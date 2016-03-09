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

#include "FrameQueue.h"
#include <stdio.h>
#include <assert.h>

FrameQueue::FrameQueue(CUvideoctxlock ctxLock): hEvent_(0)
    , nReadPosition_(0), nWritePosition_(0), nFramesInQueue_(0)
    , bEndOfDecode_(0), m_ctxLock(ctxLock)
{
#ifdef _WIN32
    hEvent_ = CreateEvent(NULL, false, false, NULL);
    InitializeCriticalSection(&oCriticalSection_);
#else
    pthread_mutex_init(&oCriticalSection_, NULL);
#endif

    memset((void*)aIsFrameInUse_, 0, cnMaximumSize * sizeof(int));
}

FrameQueue::~FrameQueue()
{
#ifdef _WIN32
    DeleteCriticalSection(&oCriticalSection_);
    CloseHandle(hEvent_);
#else
    pthread_mutex_destroy(&oCriticalSection_);
#endif
}

void
FrameQueue::waitForQueueUpdate()
{
#ifdef _WIN32
    WaitForSingleObject(hEvent_, 10);
#endif
}

void
FrameQueue::enter_CS(CRITICAL_SECTION *pCS)
{
#ifdef _WIN32
    EnterCriticalSection(pCS);
#else
    pthread_mutex_lock(pCS);
#endif
}


void
FrameQueue::leave_CS(CRITICAL_SECTION *pCS)
{
#ifdef _WIN32
    LeaveCriticalSection(pCS);
#else
    pthread_mutex_unlock(pCS);
#endif
}

void
FrameQueue::set_event(HANDLE event)
{
#ifdef _WIN32
   SetEvent(event);
#endif
}

void
FrameQueue::reset_event(HANDLE event)
{
#ifdef _WIN32
   ResetEvent(event);
#endif
}

bool
FrameQueue::isInUse(int nPictureIndex)
const
{
    assert(nPictureIndex >= 0);
    assert(nPictureIndex < (int)cnMaximumSize);

    return (0 != aIsFrameInUse_[nPictureIndex]);
}

bool
FrameQueue::isEndOfDecode()
const
{
    return (0 != bEndOfDecode_);
}

void
FrameQueue::endDecode()
{
    bEndOfDecode_ = true;
    signalStatusChange();  // Signal for the display thread
}

// Spins until frame becomes available or decoding
// gets canceled.
// If the requested frame is available the method returns true.
// If decoding was interupted before the requested frame becomes
// available, the method returns false.
bool
FrameQueue::waitUntilFrameAvailable(int nPictureIndex)
{
    while (isInUse(nPictureIndex))
    {
        Sleep(1);   // Decoder is getting too far ahead from display
        if (isEndOfDecode())
            return false;
    }

    return true;
}

void
FrameQueue::signalStatusChange()
{
    set_event(hEvent_);
}

CUVIDFrameQueue::CUVIDFrameQueue(CUvideoctxlock ctxLock): FrameQueue(ctxLock)
{
    memset(aDisplayQueue_, 0, cnMaximumSize * sizeof(CUVIDPARSERDISPINFO));
}

CUVIDFrameQueue::~CUVIDFrameQueue()
{}

void
CUVIDFrameQueue::enqueue(const void * pData)
{
    // Mark the frame as 'in-use' so we don't re-use it for decoding until it is no longer needed
    // for display
    const CUVIDPARSERDISPINFO* pPicParams = (const CUVIDPARSERDISPINFO*)(pData);
    aIsFrameInUse_[pPicParams->picture_index] = true;
    // Wait until we have a free entry in the display queue (should never block if we have enough entries)
    do
    {
        bool bPlacedFrame = false;
        enter_CS(&oCriticalSection_);
        if (nFramesInQueue_ < (int)FrameQueue::cnMaximumSize)
        {
            int iWritePosition = (nReadPosition_ + nFramesInQueue_) % cnMaximumSize;
            aDisplayQueue_[iWritePosition] = *pPicParams;
            nFramesInQueue_++;
            bPlacedFrame = true;
        }

        leave_CS(&oCriticalSection_);
        if (bPlacedFrame) // Done
            break;
        Sleep(1);   // Wait a bit
    } while (!bEndOfDecode_);
    signalStatusChange();  // Signal for the display thread
}

// if no valid picture can be return the pic-info's picture_index will
// be -1.
bool
CUVIDFrameQueue::dequeue(void* pData)
{
    CUVIDPARSERDISPINFO* pDisplayInfo = (CUVIDPARSERDISPINFO*)(pData);
    pDisplayInfo->picture_index = -1;
    bool bHaveNewFrame = false;

    enter_CS(&oCriticalSection_);
    if (nFramesInQueue_ > 0)
    {
        int iEntry = nReadPosition_;
        *pDisplayInfo = aDisplayQueue_[iEntry];
        nReadPosition_ = (iEntry+1) % cnMaximumSize;
        nFramesInQueue_--;
        bHaveNewFrame = true;
    }
    leave_CS(&oCriticalSection_);

    return bHaveNewFrame;
}

void
CUVIDFrameQueue::releaseFrame(const void * pPicParams)  {

    const CUVIDPARSERDISPINFO* pInfo = (const CUVIDPARSERDISPINFO*)(pPicParams);
    aIsFrameInUse_[pInfo->picture_index] = false;
}
