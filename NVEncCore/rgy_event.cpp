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

struct Waiter;

struct Event {
    bool manual;
    unsigned count; // 0 or 1
    std::mutex mtx;
    std::vector<Waiter*> waiters;

    Event(bool manualReset, bool initial)
        : manual(manualReset),
          count(initial ? 1u : 0u) {}
};

struct Waiter {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<Event*> events;
};

HANDLE CreateEvent(void*, int manualReset, int initialState, void*) {
    return new Event(!!manualReset, !!initialState);
}

void CloseEvent(HANDLE ev) {
    delete static_cast<Event*>(ev);
}

void ResetEvent(HANDLE ev) {
    auto* e = static_cast<Event*>(ev);
    std::lock_guard<std::mutex> lk(e->mtx);
    e->count = 0;
}

void SetEvent(HANDLE ev) {
    auto* e = static_cast<Event*>(ev);

    std::lock_guard<std::mutex> lk(e->mtx);

    if (e->count == 1)
        return;

    e->count = 1;

    for (auto* w : e->waiters) {
        std::lock_guard<std::mutex> wl(w->mtx);
        w->cv.notify_one();
    }
}

static bool all_signaled(const Waiter& w) {
    for (auto* e : w.events)
        if (e->count == 0)
            return false;
    return true;
}

uint32_t WaitForSingleObject(HANDLE ev, uint32_t timeout_ms) {
    Event* e = static_cast<Event*>(ev);
    Waiter w;
    w.events.push_back(e);

    {
        std::lock_guard<std::mutex> lk(e->mtx);
        e->waiters.push_back(&w);
    }

    std::unique_lock<std::mutex> lk(w.mtx);

    auto pred = [&]() { return e->count == 1; };

    bool ok;
    if (timeout_ms == INFINITE) {
        w.cv.wait(lk, pred);
        ok = true;
    } else {
        ok = w.cv.wait_for(
            lk,
            std::chrono::milliseconds(timeout_ms),
            pred
        );
    }

    {
        std::lock_guard<std::mutex> lk2(e->mtx);
        e->waiters.erase(
            std::remove(e->waiters.begin(), e->waiters.end(), &w),
            e->waiters.end()
        );
    }

    if (!ok)
        return WAIT_TIMEOUT;

    if (!e->manual)
        e->count = 0;

    return WAIT_OBJECT_0;
}

uint32_t WaitForMultipleObjects(
    uint32_t count,
    HANDLE* handles,
    int /*waitAll*/,
    uint32_t timeout_ms
) {
    Waiter w;

    for (uint32_t i = 0; i < count; ++i) {
        auto* e = static_cast<Event*>(handles[i]);
        w.events.push_back(e);

        std::lock_guard<std::mutex> lk(e->mtx);
        e->waiters.push_back(&w);
    }

    std::unique_lock<std::mutex> lk(w.mtx);

    auto pred = [&]() { return all_signaled(w); };

    bool ok;
    if (timeout_ms == INFINITE) {
        w.cv.wait(lk, pred);
        ok = true;
    } else {
        ok = w.cv.wait_for(
            lk,
            std::chrono::milliseconds(timeout_ms),
            pred
        );
    }

    // unregister
    for (auto* e : w.events) {
        std::lock_guard<std::mutex> lk2(e->mtx);
        e->waiters.erase(
            std::remove(e->waiters.begin(), e->waiters.end(), &w),
            e->waiters.end()
        );
    }

    if (!ok)
        return WAIT_TIMEOUT;

    // auto-reset consume
    for (auto* e : w.events)
        if (!e->manual)
            e->count = 0;

    return WAIT_OBJECT_0;
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

