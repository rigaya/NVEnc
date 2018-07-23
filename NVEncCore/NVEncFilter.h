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
#include <npp.h>
#include <cuda.h>
#pragma warning (pop)
#include <memory>
#include <vector>
#include "helper_cuda.h"
#include "NVEncUtil.h"
#include "NVEncParam.h"
#include "rgy_log.h"
#include "convert_csp.h"
#include "NVEncFrameInfo.h"

#pragma comment(lib, "cudart_static.lib")
#ifndef _M_IX86
#pragma comment(lib, "nppi.lib")
#endif

static const TCHAR *NPPI_DLL_NAME = _T("nppi64_80.dll");
bool check_if_nppi_dll_available();

using std::vector;

struct cudaevent_deleter {
    void operator()(cudaEvent_t *pEvent) const {
        cudaEventDestroy(*pEvent);
        delete pEvent;
    }
};

struct cudastream_deleter {
    void operator()(cudaStream_t *pStream) const {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

struct cudahost_deleter {
    void operator()(void *ptr) const {
        cudaFreeHost(ptr);
    }
};

struct cudadevice_deleter {
    void operator()(void *ptr) const {
        cudaFree(ptr);
    }
};

static inline int divCeil(int value, int radix) {
    return (value + radix - 1) / radix;
}

static NppiSize nppisize(const FrameInfo *pFrame) {
    NppiSize size;
    size.width = pFrame->width;
    size.height = pFrame->height;
    return size;
}

static NppiRect nppiroi(const FrameInfo *pFrame) {
    NppiRect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = pFrame->width;
    rect.height = pFrame->height;
    return rect;
}

static inline cudaMemcpyKind getCudaMemcpyKind(bool inputDevice, bool outputDevice) {
    if (inputDevice) {
        return (outputDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    } else {
        return (outputDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    }
}

static const TCHAR *getCudaMemcpyKindStr(cudaMemcpyKind kind) {
    switch (kind) {
    case cudaMemcpyDeviceToDevice:
        return _T("copyDtoD");
    case cudaMemcpyDeviceToHost:
        return _T("copyDtoH");
    case cudaMemcpyHostToDevice:
        return _T("copyHtoD");
    case cudaMemcpyHostToHost:
        return _T("copyHtoH");
    default:
        return _T("copyUnknown");
    }
}

static const TCHAR *getCudaMemcpyKindStr(bool inputDevice, bool outputDevice) {
    return getCudaMemcpyKindStr(getCudaMemcpyKind(inputDevice, outputDevice));
}

class NVEncFilterParam {
public:
    FrameInfo frameIn;
    FrameInfo frameOut;
    bool bOutOverwrite;

    NVEncFilterParam() : frameIn({ 0 }), frameOut({ 0 }), bOutOverwrite(false) {};
    virtual ~NVEncFilterParam() {};
};

struct CUFrameBuf {
public:
    FrameInfo frame;
    cudaEvent_t event;
    CUFrameBuf()
        : frame({ 0 }), event() {
        cudaEventCreate(&event);
    };
    CUFrameBuf(uint8_t *ptr, int pitch, int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frame({ 0 }), event() {
        frame.ptr = ptr;
        frame.pitch = pitch;
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.deivce_mem = true;
        cudaEventCreate(&event);
    };
    CUFrameBuf(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frame({ 0 }), event() {
        frame.ptr = nullptr;
        frame.pitch = 0;
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.deivce_mem = true;
        cudaEventCreate(&event);
    };
    CUFrameBuf(const FrameInfo& _info) 
        : frame(_info), event() {
        cudaEventCreate(&event);
    };
protected:
    CUFrameBuf(const CUFrameBuf &) = delete;
    void operator =(const CUFrameBuf &) = delete;
public:
    cudaError_t alloc() {
        if (frame.ptr) {
            cudaFree(frame.ptr);
        }
        size_t memPitch = 0;
        cudaError_t ret = cudaSuccess;
        const auto infoEx = getFrameInfoExtra(&frame);
        if (infoEx.width_byte) {
            ret = cudaMallocPitch(&frame.ptr, &memPitch, infoEx.width_byte, infoEx.height_total);
        } else {
            ret = cudaErrorNotSupported;
        }
        frame.pitch = (int)memPitch;
        return ret;
    }
    void clear() {
        if (frame.ptr) {
            cudaFree(frame.ptr);
            frame.ptr = nullptr;
        }
    }
    ~CUFrameBuf() {
        clear();
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
};

struct CUMemBuf {
    void *ptr;
    size_t nSize;

    CUMemBuf() : ptr(nullptr), nSize(0) {

    };
    CUMemBuf(void *_ptr, size_t _nSize) : ptr(_ptr), nSize(_nSize) {

    };
    CUMemBuf(size_t _nSize) : ptr(nullptr), nSize(_nSize) {

    }
    cudaError_t alloc() {
        if (ptr) {
            cudaFree(ptr);
        }
        cudaError_t ret = cudaSuccess;
        if (nSize > 0) {
            ret = cudaMalloc(&ptr, nSize);
        } else {
            ret = cudaErrorNotSupported;
        }
        return ret;
    }
    void clear() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        nSize = 0;
    }
    ~CUMemBuf() {
        clear();
    }
};

struct CUMemBufPair {
    void *ptrDevice;
    void *ptrHost;
    size_t nSize;

    CUMemBufPair() : ptrDevice(nullptr), ptrHost(nullptr), nSize(0) {

    };
    CUMemBufPair(size_t _nSize) : ptrDevice(nullptr), ptrHost(nullptr), nSize(_nSize) {

    }
    cudaError_t alloc() {
        if (ptrDevice) {
            cudaFree(ptrDevice);
        }
        cudaError_t ret = cudaSuccess;
        if (nSize > 0) {
            ret = cudaMalloc(&ptrDevice, nSize);
            if (ret == cudaSuccess) {
                ret = cudaMallocHost(&ptrHost, nSize);
            }
        } else {
            ret = cudaErrorNotSupported;
        }
        return ret;
    }
    cudaError_t alloc(size_t _nSize) {
        nSize = _nSize;
        return alloc();
    }
    cudaError_t copyDtoHAsync(cudaStream_t stream = 0) {
        return cudaMemcpyAsync(ptrHost, ptrDevice, nSize, cudaMemcpyDeviceToHost, stream);
    }
    cudaError_t copyDtoH() {
        return cudaMemcpy(ptrHost, ptrDevice, nSize, cudaMemcpyDeviceToHost);
    }
    cudaError_t copyHtoDAsync(cudaStream_t stream = 0) {
        return cudaMemcpyAsync(ptrDevice, ptrHost, nSize, cudaMemcpyHostToDevice, stream);
    }
    cudaError_t copyHtoD() {
        return cudaMemcpy(ptrDevice, ptrHost, nSize, cudaMemcpyHostToDevice);
    }
    void clear() {
        if (ptrDevice) {
            cudaFree(ptrDevice);
            ptrDevice = nullptr;
        }
        if (ptrHost) {
            cudaFreeHost(ptrHost);
            ptrDevice = nullptr;
        }
        nSize = 0;
    }
    ~CUMemBufPair() {
        clear();
    }
};

enum FILTER_PATHTHROUGH_FRAMEINFO : uint32_t {
    FILTER_PATHTHROUGH_NONE      = 0x00u,
    FILTER_PATHTHROUGH_TIMESTAMP = 0x01u,
    FILTER_PATHTHROUGH_FLAGS     = 0x02u,
    FILTER_PATHTHROUGH_PICSTRUCT = 0x04u,

    FILTER_PATHTHROUGH_ALL       = 0x07u,
};

static FILTER_PATHTHROUGH_FRAMEINFO operator|(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a | (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator|=(FILTER_PATHTHROUGH_FRAMEINFO& a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a | b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a & (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&=(FILTER_PATHTHROUGH_FRAMEINFO& a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a & b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator~(FILTER_PATHTHROUGH_FRAMEINFO a) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)(~((uint32_t)a));
}

class NVEncFilter {
public:
    NVEncFilter();
    virtual ~NVEncFilter();
    tstring name() {
        return m_sFilterName;
    }
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) = 0;
    NVENCSTATUS filter(FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum);
    const tstring GetInputMessage() {
        return m_sFilterInfo;
    }
    const NVEncFilterParam *GetFilterParam() {
        return m_pParam.get();
    }
    void CheckPerformance(bool flag);
    double GetAvgTimeElapsed();
protected:
    NVENCSTATUS filter_as_interlaced_pair(const FrameInfo *pInputFrame, FrameInfo *pOutputFrame, cudaStream_t stream);
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) = 0;
    virtual void close() = 0;

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
    cudaError_t AllocFrameBuf(const FrameInfo& frame, int frames);

    tstring m_sFilterName;
    tstring m_sFilterInfo;
    shared_ptr<RGYLog> m_pPrintMes;  //ログ出力
    vector<unique_ptr<CUFrameBuf>> m_pFrameBuf;
    int m_nFrameIdx;
    unique_ptr<CUFrameBuf> m_pFieldPairIn;
    unique_ptr<CUFrameBuf> m_pFieldPairOut;
    shared_ptr<NVEncFilterParam> m_pParam;
    FILTER_PATHTHROUGH_FRAMEINFO m_nPathThrough;
private:
    bool m_bCheckPerformance;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterStart;
    unique_ptr<cudaEvent_t, cudaevent_deleter> m_peFilterFin;
    double m_dFilterTimeMs;
    int m_nFilterRunCount;
};

class NVEncFilterParamCrop : public NVEncFilterParam {
public:
    sInputCrop crop;

    virtual ~NVEncFilterParamCrop() {};
};

class NVEncFilterCspCrop : public NVEncFilter {
public:
    NVEncFilterCspCrop();
    virtual ~NVEncFilterCspCrop();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    NVENCSTATUS convertYBitDepth(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS convertCspFromNV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS convertCspFromYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS convertCspFromNV16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS convertCspFromRGB(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS convertCspFromYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;
};

class NVEncFilterParamResize : public NVEncFilterParam {
public:
    int interp;
    virtual ~NVEncFilterParamResize() {};
};

class NVEncFilterResize : public NVEncFilter {
public:
    NVEncFilterResize();
    virtual ~NVEncFilterResize();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    NVENCSTATUS resizeYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS resizeYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;

    bool m_bInterlacedWarn;
    CUMemBuf m_weightSpline36;
};


class NVEncFilterParamGaussDenoise : public NVEncFilterParam {
public:
    NppiMaskSize masksize;
    virtual ~NVEncFilterParamGaussDenoise() {};
};

class NVEncFilterDenoiseGauss : public NVEncFilter {
public:
    NVEncFilterDenoiseGauss();
    virtual ~NVEncFilterDenoiseGauss();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    NVENCSTATUS denoiseYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    NVENCSTATUS denoiseYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;
    bool m_bInterlacedWarn;
};

class NVEncFilterParamPad : public NVEncFilterParam {
public:
    VppPad pad;
    virtual ~NVEncFilterParamPad() {};
};

class NVEncFilterPad : public NVEncFilter {
public:
    NVEncFilterPad();
    virtual ~NVEncFilterPad();
    virtual NVENCSTATUS init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual NVENCSTATUS run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;

    NVENCSTATUS padPlane(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, int pad_color, const VppPad *pad);
    virtual void close() override;
};
