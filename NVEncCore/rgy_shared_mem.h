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
#ifndef __RGY_SHARED_MEM_H__
#define __RGY_SHARED_MEM_H__

#include "rgy_osdep.h"
#include <cstdint>
#include <string>

class RGYSharedMem {
protected:
    uint64_t shared_size;
    void *handle;
    void *buffer;
    std::string mem_name;

public:
    RGYSharedMem() : shared_size(0), handle(nullptr), buffer(nullptr) {
    }

#pragma warning(push)
#pragma warning(disable:4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
    RGYSharedMem(const char *pipename, uint64_t size) : RGYSharedMem() {
    }
#pragma warning(pop)

    virtual ~RGYSharedMem() {
    }

    void *ptr() const { return buffer; }
    bool is_open() { return (buffer != nullptr) && (handle != nullptr); }
    void setSize(uint64_t size) { shared_size = size; }
    uint64_t size() const { return shared_size; }
    const std::string &name() const { return mem_name; }

    virtual void open(const char *pipename, uint64_t size) = 0;
    virtual void close() = 0;
};

#if defined(_WIN32) || defined(_WIN64)
class RGYSharedMemWin : public RGYSharedMem {
public:
    RGYSharedMemWin() {
        shared_size = 0;
        handle = nullptr;
        buffer = nullptr;
    };
    RGYSharedMemWin(const char *pipename, uint64_t size) : RGYSharedMemWin() {
        open(pipename, size);
    };
    virtual ~RGYSharedMemWin() {
        close();
    };

    void open(const char *pipename, uint64_t size) override {
        close();
        shared_size = size;
        mem_name = pipename;
        handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(size >> 32), (DWORD)(size & 0xffffffffu), pipename);
        if (handle == nullptr) {
            return;
        }
        buffer = nullptr;
        for (int i = 0; i < 10 && buffer == nullptr; i++) {
            //空いているメモリ空間を探す
            void *ptr = VirtualAlloc(NULL, (size_t)size, MEM_RESERVE, PAGE_READWRITE);
            if (!ptr) {
                break;
            }
            //探せたらすぐに開放
            VirtualFree(ptr, 0, MEM_RELEASE);
            //アライメントを考慮してメモリをマッピングする
            buffer = MapViewOfFileEx(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0, ptr);
        }
        if (buffer == nullptr) {
            CloseHandle(handle);
            handle = nullptr;
            return;
        }
    }
    void close() override {
        if (buffer != nullptr) {
            UnmapViewOfFile(buffer);
            buffer = nullptr;
        }
        if (handle != nullptr) {
            CloseHandle(handle);
            handle = nullptr;
        }
        shared_size = 0;
        mem_name.clear();
    }
};
#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__RGY_SHARED_MEM_H__
