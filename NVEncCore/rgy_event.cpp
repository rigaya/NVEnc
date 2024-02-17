// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#include "rgy_event.h"
#if !(defined(_WIN32) || defined(_WIN64))

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <climits>
#include <chrono>
#include <algorithm>

class Event {
public:
    bool bManualReset;
    bool bReady;
    std::mutex mtx;
    std::condition_variable cv;

    Event() : bManualReset(false), bReady(false), mtx(), cv() {

    };
    Event(bool manualReset) : Event() {
        bManualReset = manualReset;
    };
};

void ResetEvent(HANDLE ev) {
    Event *event = (Event *)ev;
    {
        std::lock_guard<std::mutex> lock(event->mtx);
        if (event->bReady) {
            event->bReady = false;
        }
    }
}

void SetEvent(HANDLE ev) {
    Event *event = (Event *)ev;
    {
        std::lock_guard<std::mutex> lock(event->mtx);
        if (!event->bReady) {
            event->bReady = true;
            (event->bManualReset) ? event->cv.notify_all() : event->cv.notify_one();
        }
    }
}

HANDLE CreateEvent(void *pDummy, int bManualReset, int bInitialState, void *pDummy2) {
    Event *event = new Event(!!bManualReset);
    if (bInitialState) {
        SetEvent(event);
    }
    return event;
}

void CloseEvent(HANDLE ev) {
    if (ev != NULL) {
        Event *event = (Event *)ev;
       delete event;
    }
}

uint32_t WaitForSingleObject(HANDLE ev, uint32_t millisec) {
    Event *event = (Event *)ev;
    {
        std::unique_lock<std::mutex> uniq_lk(event->mtx);
        if (millisec == INFINITE) {
            event->cv.wait(uniq_lk, [&event]{ return event->bReady;});
            if (!event->bManualReset) {
                event->bReady = false;
            }
        } else {
            event->cv.wait_for(uniq_lk, std::chrono::milliseconds(millisec), [&event]{ return event->bReady;});
            if (!event->bReady) {
                return WAIT_TIMEOUT;
            }
            if (!event->bManualReset) {
                event->bReady = false;
            }
        }
    }
    return WAIT_OBJECT_0;
}

uint32_t WaitForMultipleObjects(uint32_t count, HANDLE *pev, int dummy, uint32_t millisec) {
    Event **pevent = (Event **)pev;
    int success = 0;
    bool bTimeout = false;
    for (uint32_t i = 0; i < count; i++) {
        if (WAIT_TIMEOUT == WaitForSingleObject(pevent[i], (bTimeout) ? 0 : millisec)) {
            bTimeout = true;
        } else {
            success++;
        }
    }
    return (bTimeout) ? WAIT_TIMEOUT : (WAIT_OBJECT_0 + success);
}

unique_event CreateEventUnique(void *pDummy, int bManualReset, int bInitialState, void *pDummy2) {
    return unique_event(CreateEvent(pDummy, bManualReset, bInitialState, pDummy2), CloseEvent);
}
unique_event CreateEventUnique(void* pDummy, int bManualReset, int bInitialState) {
    return unique_event(CreateEvent(pDummy, bManualReset, bInitialState, nullptr), CloseEvent);
}

#else

unique_event CreateEventUnique(void* pDummy, int bManualReset, int bInitialState, const wchar_t* name) {
    return unique_event(CreateEventW((LPSECURITY_ATTRIBUTES)pDummy, bManualReset, bInitialState, name), CloseEvent);
}
unique_event CreateEventUnique(void* pDummy, int bManualReset, int bInitialState, const char* name) {
    return unique_event(CreateEventA((LPSECURITY_ATTRIBUTES)pDummy, bManualReset, bInitialState, name), CloseEvent);
}
unique_event CreateEventUnique(void* pDummy, int bManualReset, int bInitialState) {
    return unique_event(CreateEventA((LPSECURITY_ATTRIBUTES)pDummy, bManualReset, bInitialState, nullptr), CloseEvent);
}

#endif //#if !(defined(_WIN32) || defined(_WIN64))

