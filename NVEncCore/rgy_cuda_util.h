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
#ifndef __RGY_CUDA_UTIL_H__
#define __RGY_CUDA_UTIL_H__

#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#include <npp.h>
#include <cuda.h>
#pragma warning (pop)
#include "rgy_tchar.h"
#include "rgy_util.h"

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

static cudaError_t copyPlane(FrameInfo *dst, const FrameInfo *src) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemcpy2D(dst->ptr, dst->pitch, src->ptr, src->pitch, width_byte, dst->height, getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem));
}

static cudaError_t copyPlaneAsync(FrameInfo *dst, const FrameInfo *src, cudaStream_t stream) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemcpy2DAsync(dst->ptr, dst->pitch, src->ptr, src->pitch, width_byte, dst->height, getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem), stream);
}

static cudaError_t copyPlaneField(FrameInfo *dst, const FrameInfo *src, const bool dstTopField, const bool srcTopField) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemcpy2D(
        dst->ptr + ((dstTopField) ? 0 : dst->pitch),
        dst->pitch << 1,
        src->ptr + ((srcTopField) ? 0 : src->pitch),
        src->pitch << 1,
        width_byte,
        dst->height >> 1,
        getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem));
}

static cudaError_t copyPlaneFieldAsync(FrameInfo *dst, const FrameInfo *src, const bool dstTopField, const bool srcTopField, cudaStream_t stream) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemcpy2DAsync(
        dst->ptr + ((dstTopField) ? 0 : dst->pitch),
        dst->pitch << 1,
        src->ptr + ((srcTopField) ? 0 : src->pitch),
        src->pitch << 1,
        width_byte,
        dst->height >> 1,
        getCudaMemcpyKind(src->deivce_mem, dst->deivce_mem), stream);
}

static cudaError_t setPlane(FrameInfo *dst, int value) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemset2D(
        dst->ptr,
        dst->pitch,
        value,
        width_byte,
        dst->height);
}

static cudaError_t setPlaneAsync(FrameInfo *dst, int value, cudaStream_t stream) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemset2DAsync(
        dst->ptr,
        dst->pitch,
        value,
        width_byte,
        dst->height, stream);
}

static cudaError_t setPlaneField(FrameInfo *dst, int value, bool topField) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemset2D(
        dst->ptr + ((topField) ? 0 : dst->pitch),
        dst->pitch << 1,
        value,
        width_byte,
        dst->height >> 1);
}

static cudaError_t setPlaneFieldAsync(FrameInfo *dst, int value, bool topField, cudaStream_t stream) {
    const int width_byte = dst->width * (RGY_CSP_BIT_DEPTH[dst->csp] > 8 ? 2 : 1);
    return cudaMemset2DAsync(
        dst->ptr + ((topField) ? 0 : dst->pitch),
        dst->pitch << 1,
        value,
        width_byte,
        dst->height >> 1, stream);
}

static cudaError_t checkCopyFrame(FrameInfo *dst, const FrameInfo *src) {
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
    return cudaSuccess;
}

static cudaError_t copyFrame(FrameInfo *dst, const FrameInfo *src) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlane(&dstPlane, &srcPlane);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    return cudaSuccess;
}

static cudaError_t copyFrameAsync(FrameInfo *dst, const FrameInfo *src, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneAsync(&dstPlane, &srcPlane, stream);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    return cudaSuccess;
}

static cudaError_t copyFrameField(FrameInfo *dst, const FrameInfo *src, const bool dstTopField, const bool srcTopField) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneField(&dstPlane, &srcPlane, dstTopField, srcTopField);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    return cudaSuccess;
}

static cudaError_t copyFrameFieldAsync(FrameInfo *dst, const FrameInfo *src, const bool dstTopField, const bool srcTopField, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneFieldAsync(&dstPlane, &srcPlane, dstTopField, srcTopField, stream);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    return cudaSuccess;
}

static cudaError_t copyFrameData(FrameInfo *dst, const FrameInfo *src) {
    {   auto ret = checkCopyFrame(dst, src);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    auto ret = copyFrame(dst, src);
    if (ret == cudaSuccess) {
        copyFrameProp(dst, src);
    }
    return ret;
}

static cudaError_t copyFrameDataAsync(FrameInfo *dst, const FrameInfo *src, cudaStream_t stream) {
    {   auto ret = checkCopyFrame(dst, src);
        if (ret != cudaSuccess) {
            return ret;
        }
    }
    auto ret = copyFrameAsync(dst, src, stream);
    if (ret == cudaSuccess) {
        copyFrameProp(dst, src);
    }
    return ret;
}

struct CUFrameBuf {
public:
    FrameInfo frame;
    cudaEvent_t event;
    CUFrameBuf()
        : frame(), event() {
        cudaEventCreate(&event);
    };
    CUFrameBuf(uint8_t *ptr, int pitch, int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frame(), event() {
        frame.ptr = ptr;
        frame.pitch = pitch;
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.deivce_mem = true;
        cudaEventCreate(&event);
    };
    CUFrameBuf(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frame(), event() {
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
    cudaError_t alloc(int width, int height, RGY_CSP csp = RGY_CSP_NV12) {
        if (frame.ptr) {
            cudaFree(frame.ptr);
        }
        frame.ptr = nullptr;
        frame.pitch = 0;
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.deivce_mem = true;
        return alloc();
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

struct CUFrameBufPair {
public:
    FrameInfo frameDev;
    FrameInfo frameHost;
    cudaEvent_t event;
    CUFrameBufPair()
        : frameDev(), frameHost(), event() {
        cudaEventCreate(&event);
    };
    CUFrameBufPair(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frameDev(), frameHost(), event() {
        frameDev.ptr = nullptr;
        frameDev.pitch = 0;
        frameDev.width = width;
        frameDev.height = height;
        frameDev.csp = csp;
        frameDev.deivce_mem = true;

        frameHost = frameDev;
        frameHost.deivce_mem = false;

        cudaEventCreate(&event);
    };
protected:
    CUFrameBufPair(const CUFrameBufPair &) = delete;
    void operator =(const CUFrameBufPair &) = delete;
public:
    cudaError_t allocHost() {
        if (frameHost.ptr) {
            cudaFree(frameHost.ptr);
        }
        cudaError_t ret = cudaSuccess;
        const auto infoEx = getFrameInfoExtra(&frameHost);
        frameHost.pitch = ALIGN(infoEx.width_byte, 256);
        if (infoEx.width_byte) {
            ret = cudaMallocHost(&frameHost.ptr, frameHost.pitch * infoEx.height_total);
        } else {
            ret = cudaErrorNotSupported;
        }
        return ret;
    }
    cudaError_t allocDev() {
        if (frameDev.ptr) {
            cudaFree(frameDev.ptr);
        }
        size_t memPitch = 0;
        cudaError_t ret = cudaSuccess;
        const auto infoEx = getFrameInfoExtra(&frameDev);
        if (infoEx.width_byte) {
            ret = cudaMallocPitch(&frameDev.ptr, &memPitch, infoEx.width_byte, infoEx.height_total);
        } else {
            ret = cudaErrorNotSupported;
        }
        frameDev.pitch = (int)memPitch;
        return ret;
    }
    cudaError_t alloc() {
        clearHost();
        clearDev();
        auto err = allocDev();
        if (err != cudaSuccess) return err;
        return allocHost();
    }
    cudaError_t alloc(int width, int height, RGY_CSP csp = RGY_CSP_NV12) {
        clearHost();
        clearDev();

        frameDev.ptr = nullptr;
        frameDev.pitch = 0;
        frameDev.width = width;
        frameDev.height = height;
        frameDev.csp = csp;
        frameDev.deivce_mem = true;

        frameHost = frameDev;
        frameHost.deivce_mem = false;

        return alloc();
    }
    cudaError_t copyDtoHAsync(cudaStream_t stream = 0) {
        const auto infoEx = getFrameInfoExtra(&frameDev);
        return cudaMemcpy2DAsync(frameHost.ptr, frameHost.pitch, frameDev.ptr, frameDev.pitch, infoEx.width_byte, infoEx.height_total, cudaMemcpyDeviceToHost, stream);
    }
    cudaError_t copyDtoH() {
        const auto infoEx = getFrameInfoExtra(&frameDev);
        return cudaMemcpy2D(frameHost.ptr, frameHost.pitch, frameDev.ptr, frameDev.pitch, infoEx.width_byte, infoEx.height_total, cudaMemcpyDeviceToHost);
    }
    cudaError_t copyHtoDAsync(cudaStream_t stream = 0) {
        const auto infoEx = getFrameInfoExtra(&frameDev);
        return cudaMemcpy2DAsync(frameDev.ptr, frameDev.pitch, frameHost.ptr, frameHost.pitch, infoEx.width_byte, infoEx.height_total, cudaMemcpyHostToDevice, stream);
    }
    cudaError_t copyHtoD() {
        const auto infoEx = getFrameInfoExtra(&frameDev);
        return cudaMemcpy2D(frameDev.ptr, frameDev.pitch, frameHost.ptr, frameHost.pitch, infoEx.width_byte, infoEx.height_total, cudaMemcpyHostToDevice);
    }

    void clearHost() {
        if (frameDev.ptr) {
            cudaFree(frameDev.ptr);
            frameDev.ptr = nullptr;
        }
    }
    void clearDev() {
        if (frameDev.ptr) {
            cudaFree(frameDev.ptr);
            frameDev.ptr = nullptr;
        }
    }
    void clear() {
        clearDev();
        clearHost();
    }
    ~CUFrameBufPair() {
        clearDev();
        clearHost();
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

#endif //__RGY_CUDA_UTIL_H__
