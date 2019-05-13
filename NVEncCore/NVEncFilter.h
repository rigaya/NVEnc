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

struct AVPacket;

extern const TCHAR *NPPI_DLL_NAME_TSTR;
extern const TCHAR *NVRTC_DLL_NAME_TSTR;
extern const TCHAR *NVRTC_BUILTIN_DLL_NAME_TSTR;

bool check_if_nppi_dll_available();
#if ENABLE_NVRTC
bool check_if_nvrtc_dll_available();
bool check_if_nvrtc_builtin_dll_available();
#endif

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

static cudaError_t copyFrameData(FrameInfo *dst, const FrameInfo *src) {
    auto dstInfoEx = getFrameInfoExtra(dst);
    const auto srcInfoEx = getFrameInfoExtra(src);
    if (dst->pitch == 0
        || srcInfoEx.width_byte > dst->pitch
        || srcInfoEx.height_total > dstInfoEx.height_total) {
        if (dst->ptr) {
            cudaFree(dst->ptr);
            dst->ptr = nullptr;
        }
        dst->pitch = 0;
    }
    dst->width = src->width;
    dst->height = src->height;
    dst->csp = src->csp;
    dst->picstruct = src->picstruct;
    dst->timestamp = src->timestamp;
    dst->duration = src->duration;
    dst->flags = src->flags;
    if (dst->ptr == nullptr) {
        dstInfoEx = getFrameInfoExtra(dst);
        if (!dstInfoEx.width_byte) {
            return cudaErrorNotSupported;
        }
        if (dst->deivce_mem) {
            size_t memPitch = 0;
            auto ret = cudaMallocPitch(&dst->ptr, &memPitch, dstInfoEx.width_byte, dstInfoEx.height_total);
            if (ret != cudaSuccess) {
                return ret;
            }
            dst->pitch = (int)memPitch;
        } else {
            dst->pitch = ALIGN(dstInfoEx.width_byte, 64);
            dstInfoEx = getFrameInfoExtra(dst);
            auto ret = cudaMallocHost(&dst->ptr, dstInfoEx.frame_size);
            if (ret != cudaSuccess) {
                return ret;
            }
        }
    }
    //更新
    dstInfoEx = getFrameInfoExtra(dst);
    return cudaMemcpy2D(dst->ptr, dst->pitch, src->ptr, src->pitch, dstInfoEx.width_byte, dstInfoEx.height_total, getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem));
}

static cudaError_t copyFrameDataAsync(FrameInfo *dst, const FrameInfo *src, cudaStream_t stream) {
    auto dstInfoEx = getFrameInfoExtra(dst);
    const auto srcInfoEx = getFrameInfoExtra(src);
    if (dst->pitch == 0
        || srcInfoEx.width_byte > dst->pitch
        || srcInfoEx.height_total > dstInfoEx.height_total) {
        if (dst->ptr) {
            cudaFree(dst->ptr);
            dst->ptr = nullptr;
        }
        dst->pitch = 0;
    }
    dst->width = src->width;
    dst->height = src->height;
    dst->csp = src->csp;
    dst->picstruct = src->picstruct;
    dst->timestamp = src->timestamp;
    dst->duration = src->duration;
    dst->flags = src->flags;
    if (dst->ptr == nullptr) {
        dstInfoEx = getFrameInfoExtra(dst);
        if (!dstInfoEx.width_byte) {
            return cudaErrorNotSupported;
        }
        if (dst->deivce_mem) {
            size_t memPitch = 0;
            auto ret = cudaMallocPitch(&dst->ptr, &memPitch, dstInfoEx.width_byte, dstInfoEx.height_total);
            if (ret != cudaSuccess) {
                return ret;
            }
            dst->pitch = (int)memPitch;
        } else {
            dst->pitch = ALIGN(dstInfoEx.width_byte, 64);
            dstInfoEx = getFrameInfoExtra(dst);
            auto ret = cudaMallocHost(&dst->ptr, dstInfoEx.frame_size);
            if (ret != cudaSuccess) {
                return ret;
            }
        }
    }
    //更新
    dstInfoEx = getFrameInfoExtra(dst);
    return cudaMemcpy2DAsync(dst->ptr, dst->pitch, src->ptr, src->pitch, dstInfoEx.width_byte, dstInfoEx.height_total, getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem), stream);
}

class NVEncFilterParam {
public:
    FrameInfo frameIn;
    FrameInfo frameOut;
    rgy_rational<int> baseFps;
    bool bOutOverwrite;

    NVEncFilterParam() : frameIn({ 0 }), frameOut({ 0 }), baseFps(), bOutOverwrite(false) {};
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
    cudaError_t copyFrame(const FrameInfo *src) {
        return copyFrameData(&frame, src);
    }
    cudaError_t copyFrameAsync(const FrameInfo *src, cudaStream_t stream) {
        return copyFrameDataAsync(&frame, src, stream);
    }
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
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) = 0;
    RGY_ERR filter(FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum);
    const tstring GetInputMessage() {
        return m_sFilterInfo;
    }
    const NVEncFilterParam *GetFilterParam() {
        return m_pParam.get();
    }
    void CheckPerformance(bool flag);
    double GetAvgTimeElapsed();
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) { UNREFERENCED_PARAMETER(pkt); return RGY_ERR_UNSUPPORTED; };
    virtual int targetTrackIdx() { return 0; };
protected:
    RGY_ERR filter_as_interlaced_pair(const FrameInfo *pInputFrame, FrameInfo *pOutputFrame, cudaStream_t stream);
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) = 0;
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
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    RGY_ERR convertYBitDepth(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR convertCspFromNV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR convertCspFromYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR convertCspFromNV16(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR convertCspFromRGB(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR convertCspFromYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
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
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    RGY_ERR resizeNppiYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR resizeNppiYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    virtual void close() override;

    bool m_bInterlacedWarn;
    CUMemBuf m_weightSpline;
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
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;
    RGY_ERR denoiseYV12(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
    RGY_ERR denoiseYUV444(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame);
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
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) override;

    RGY_ERR padPlane(FrameInfo *pOutputFrame, const FrameInfo *pInputFrame, int pad_color, const VppPad *pad);
    virtual void close() override;
};
