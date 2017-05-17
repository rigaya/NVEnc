// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
// ------------------------------------------------------------------------------------------
#ifndef __NVENC_THREAD_H__
#define __NVENC_THREAD_H__

#include <thread>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

static void __forceinline sleep_hybrid(int count) {
    _mm_pause();
    if ((count & 4095) == 4095) {
        std::this_thread::sleep_for(std::chrono::milliseconds((count & 65535) == 65535));
    }
}

#if defined(_WIN32) || defined(_WIN64)

static inline bool CheckThreadAlive(std::thread& thread) {
    DWORD exit_code = 0;
    return (0 != GetExitCodeThread(thread.native_handle(), &exit_code)) && exit_code == STILL_ACTIVE;
}

#else //#if defined(_WIN32) || defined(_WIN64)
#include <pthread.h>
#include <signal.h>

static inline bool CheckThreadAlive(std::thread& thread) {
    uint32_t exit_code = 0;
    return pthread_kill(thread.native_handle(), 0) != ESRCH;
}

#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__NVENC_THREAD_H__