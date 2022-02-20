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

#pragma once
#ifndef __RGY_ENV_H__
#define __RGY_ENV_H__

#include <memory>
#include <vector>
#include <functional>
#include "rgy_util.h"

#if defined(_WIN32) || defined(_WIN64)
tstring getOSVersion(OSVERSIONINFOEXW *osinfo);
tstring getOSVersion();
#else
tstring getOSVersion();
#endif
BOOL rgy_is_64bit_os();
uint64_t getPhysicalRamSize(uint64_t *ramUsed);
tstring getEnviromentInfo(int device_id = 0);

BOOL check_OS_Win8orLater();

#if defined(_WIN32) || defined(_WIN64)
using unique_handle = std::unique_ptr<std::remove_pointer<HANDLE>::type, std::function<void(HANDLE)>>;

std::vector<size_t> createChildProcessIDList(const size_t target_pid);
std::vector<unique_handle> createProcessHandleList(const std::vector<size_t>& list_pid, const wchar_t *handle_type);
#endif

#endif //__RGY_ENV_H__
