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

#if defined(_WIN32) || defined(_WIN64)
using SMHandle = HANDLE;
#else
#include <sys/shm.h>
using SMHandle = key_t;
#endif

class RGYSharedMem {
protected:
    uint64_t shared_size;
    SMHandle handle;
    void *buffer;
    std::string mem_name;

public:
    RGYSharedMem() : shared_size(0), handle(0), buffer(nullptr) {
    }

#pragma warning(push)
#pragma warning(disable:4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
    RGYSharedMem(const char *pipename, uint64_t size) : RGYSharedMem() {
    }
#pragma warning(pop)

    virtual ~RGYSharedMem() {
    }

    void *ptr() const { return buffer; }
    virtual bool is_open() { return (buffer != nullptr) && (handle != 0); }
    void setSize(uint64_t size) { shared_size = size; }
    uint64_t size() const { return shared_size; }
    const std::string &name() const { return mem_name; }

    virtual int open(const char *pipename, uint64_t size) = 0;
    virtual int open(const int id, uint64_t size) = 0;
    virtual void close() = 0;
    virtual void detach() = 0;
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

    virtual int open(const char *pipename, uint64_t size) override {
        close();
        shared_size = size;
        mem_name = pipename;
        handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(size >> 32), (DWORD)(size & 0xffffffffu), pipename);
        if (handle == nullptr) {
            return 1;
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
            return 1;
        }
        return 0;
    }
#pragma warning(push)
#pragma warning(disable:4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
    virtual int open(const int id, uint64_t size) override {
        return 1;
    }
#pragma warning(pop)
    void detach() override {
        close();
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
#else
class RGYSharedMemLinux : public RGYSharedMem {
public:
    RGYSharedMemLinux() {
        shared_size = 0;
        handle = -1;
        buffer = nullptr;
    };
    RGYSharedMemLinux(const char *pipename, uint64_t size) : RGYSharedMem() {
        shared_size = 0;
        handle = -1;
        buffer = nullptr;
        open(pipename, size);
    };
    RGYSharedMemLinux(const int id, uint64_t size) : RGYSharedMem() {
        shared_size = 0;
        handle = -1;
        buffer = nullptr;
        open(id, size);
    };
    virtual ~RGYSharedMemLinux() {
        close();
    };
    virtual bool is_open() override { return buffer != nullptr; }

    virtual int open(const char *pipename, uint64_t size) override {
        return 1;
    }
    virtual int open(const int id, uint64_t size) override {
        handle = -1;
        const auto getExePath = []() {
            char path[4096];
            memset(path, 0, sizeof(path));
            if (readlink("/proc/self/exe", path, sizeof(path) - 1) == -1) {
                return std::string();
            }
            return std::string(path);
        };
        const auto exePath = getExePath();
        auto key = ftok(exePath.c_str(), id);
        if (key == -1) {
            return 1;
        }

        // まずは共有メモリがあるかを確認する
        auto segment_id = shmget(key, 0, 0);
        if (segment_id == -1) {
            // 共有メモリがない場合、作成する
            segment_id = shmget(key, size, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR);
            if (segment_id == -1) {
                return 1;
            }
            // 作成者であることを示すために、キーを保存しておく
            handle = key;
        }
        buffer = shmat(segment_id, 0, 0);
        shared_size = size;
    }
    void detach() override {
        if (buffer != nullptr) {
            shmdt(buffer);
            buffer = nullptr;
        }
        handle = -1;
        shared_size = 0;
    }
    void close() override {
        if (buffer != nullptr) {
            shmdt(buffer);
            buffer = nullptr;
        }
        if (handle != -1) {
            shmctl(handle, IPC_RMID, 0);
            handle = -1;
        }
        shared_size = 0;
    }
};
#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__RGY_SHARED_MEM_H__
