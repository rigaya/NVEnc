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

#ifndef __GPUZ_INFO_H__
#define __GPUZ_INFO_H__

#if (defined(_WIN32) || defined(_WIN64))
#include <Windows.h>
#include <tchar.h>

#define SHMEM_NAME _T("GPUZShMem")
#define MAX_RECORDS 128

#pragma pack(push, 1)
struct GPUZ_RECORD {
    WCHAR key[256];
    WCHAR value[256];
};

struct GPUZ_SENSOR_RECORD {
    WCHAR name[256];
    WCHAR unit[8];
    UINT32 digits;
    double value;
};

struct GPUZ_SH_MEM {
    UINT32 version;
    volatile LONG busy;
    UINT32 lastUpdate;
    GPUZ_RECORD data[MAX_RECORDS];
    GPUZ_SENSOR_RECORD sensors[MAX_RECORDS];
};
#pragma pack(pop)

int get_gpuz_info(GPUZ_SH_MEM *data);
double gpu_core_clock(GPUZ_SH_MEM *data);
double gpu_load(GPUZ_SH_MEM *data);

#endif //#if (defined(_WIN32) || defined(_WIN64))

#endif //__GPUZ_INFO_H__
