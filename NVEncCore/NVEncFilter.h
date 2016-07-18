// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdint.h>
#include <tchar.h>
#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#pragma warning (pop)
#include <memory>
#include <vector>
#include "cuda.h"
#include "helper_cuda.h"
#include "NVEncUtil.h"
#include "NVEncLog.h"
#include "ConvertCsp.h"

#pragma comment(lib, "cudart.lib")

using std::vector;

static inline int divCeil(int value, int radix) {
    return (value + radix - 1) / radix;
}

class NVEncFilterParam {
public:
    int nInWidth;
    int nInHeight;
    int nInPitch;
    int nOutWidth;
    int nOutHeight;
    int nOutPitch;
    bool bOutOverwrite;

    NVEncFilterParam() : nInWidth(0), nInHeight(0), nOutWidth(0), nOutHeight(0), bOutOverwrite(false) {};
    virtual ~NVEncFilterParam() {};
};

struct CUFrameBuf {
    uint8_t *ptr;
    int pitch;
    int width;
    int height;
    NV_ENC_CSP csp;
    cudaEvent_t event;

    CUFrameBuf()
        : ptr(0), pitch(0), width(0), height(0), csp(NV_ENC_CSP_NA), event() {
        cudaEventCreate(&event);
    };
    CUFrameBuf(uint8_t *_ptr, int _pitch, int _width, int _height, NV_ENC_CSP _csp = NV_ENC_CSP_NV12)
        : ptr(0), pitch(0), width(0), height(0), csp(NV_ENC_CSP_NA), event() {
        ptr = _ptr;
        pitch = _pitch;
        width = _width;
        height = _height;
        csp = _csp;
        cudaEventCreate(&event);
    };
    CUFrameBuf(int _width, int _height, NV_ENC_CSP _csp) 
        : ptr(0), pitch(0), width(0), height(0), csp(NV_ENC_CSP_NA), event() {
        ptr = nullptr;
        width = _width;
        height = _height;
        csp = _csp;
        cudaEventCreate(&event);
    };
    cudaError_t alloc() {
        if (ptr) {
            cudaFree(ptr);
        }
        size_t memPitch = 0;
        cudaError_t ret = cudaSuccess;
        switch (csp) {
        case NV_ENC_CSP_NV12:
            ret = cudaMallocPitch(&ptr, &memPitch, width, height * 3 / 2);
            break;
        default:
            ret = cudaErrorNotSupported;
            break;
        }
        pitch = (int)memPitch;
        return ret;
    }
    ~CUFrameBuf() {
        if (ptr) {
            cudaFree(ptr);
        }
        cudaEventDestroy(event);
    }
};

class NVEncFilter {
public:
    NVEncFilter();
    virtual ~NVEncFilter();
    tstring name() {
        return m_sFilterName;
    }
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) = 0;
    virtual NVENCSTATUS filter(const CUFrameBuf *pInputFrame, CUFrameBuf **ppOutputFrames, int *pOutputFrameNum) = 0;
protected:
    virtual void close() = 0;

    const tstring GetInputMessage() {
        return m_sFilterInfo;
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_sFilterName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    cudaError_t AllocFrameBuf(int width, int height, NV_ENC_CSP csp, int frames);

    tstring m_sFilterName;
    tstring m_sFilterInfo;
    shared_ptr<CNVEncLog> m_pPrintMes;  //ログ出力
    vector<unique_ptr<CUFrameBuf>> m_pFrameBuf;
    int m_nFrameIdx;
};

class NVEncFilterParamCrop : public NVEncFilterParam {
public:
    sInputCrop crop;

    virtual ~NVEncFilterParamCrop() {};
};

class NVEncFilterCrop : public NVEncFilter {
public:
    NVEncFilterCrop();
    virtual ~NVEncFilterCrop();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<CNVEncLog> pPrintMes) override;
    virtual NVENCSTATUS filter(const CUFrameBuf *pInputFrame, CUFrameBuf **ppOutputFrames, int *pOutputFrameNum) override;
protected:
    virtual void close() override;

    NVEncFilterParamCrop m_filterParam;
};
