// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
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

#pragma once
#ifndef __RGY_EVENT_H__
#define __RGY_EVENT_H__

#include <cstdint>
#include <climits>
#include <memory>
#include "rgy_osdep.h"

#if defined(_WIN32) || defined(_WIN64)
#define CloseEvent CloseHandle
#else //#if defined(_WIN32) || defined(_WIN64)

enum : uint32_t {
    WAIT_OBJECT_0 = 0,
    WAIT_TIMEOUT = 258L,
    WAIT_ABANDONED_0 = 0x00000080L
};

#ifndef INFINITE
#define INFINITE (UINT_MAX)
#endif

void ResetEvent(HANDLE ev);

void SetEvent(HANDLE ev);

HANDLE CreateEvent(void *pDummy, int bManualReset, int bInitialState, void *pDummy2);

void CloseEvent(HANDLE ev);

uint32_t WaitForSingleObject(HANDLE ev, uint32_t millisec);

uint32_t WaitForMultipleObjects(uint32_t count, HANDLE *pev, int dummy, uint32_t millisec);

#endif //#if defined(_WIN32) || defined(_WIN64)

using unique_event = std::unique_ptr<std::remove_pointer<HANDLE>::type, decltype(&CloseEvent)>;

unique_event CreateEventUnique(void *pDummy, int bManualReset, int bInitialState, const char* name);
unique_event CreateEventUnique(void *pDummy, int bManualReset, int bInitialState, const wchar_t* name);
unique_event CreateEventUnique(void *pDummy, int bManualReset, int bInitialState);

#endif //__RGY_EVENT_H__
