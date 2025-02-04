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
#include "rgy_err.h"
#include "rgy_frame.h"
#include "rgy_frame_info.h"
#include "convert_csp.h"

#ifdef _DEBUG
#define ENABLE_CUDA_DEBUG_SYNC (1)
#else
#define ENABLE_CUDA_DEBUG_SYNC (0)
#endif

#if ENABLE_CUDA_DEBUG_SYNC
#define CUDA_DEBUG_SYNC { \
    cudaError_t cudaDebugSyncErr = cudaDeviceSynchronize(); \
    if (cudaDebugSyncErr != cudaSuccess) { \
        PrintMes(RGY_LOG_ERROR, _T("CUDA error(1): %s(%d): %s: %s\n"), char_to_tstring(__FILE__).c_str(), __LINE__, char_to_tstring(__func__).c_str(), get_err_mes(err_to_rgy(cudaDebugSyncErr))); \
        return err_to_rgy(cudaDebugSyncErr); \
    } \
    cudaDebugSyncErr = cudaGetLastError(); \
    if (cudaDebugSyncErr != cudaSuccess) { \
        PrintMes(RGY_LOG_ERROR, _T("CUDA error(2): %s(%d): %s: %s\n"), char_to_tstring(__FILE__).c_str(), __LINE__, char_to_tstring(__func__).c_str(), get_err_mes(err_to_rgy(cudaDebugSyncErr))); \
        return err_to_rgy(cudaDebugSyncErr); \
    } \
}
#define CUDA_DEBUG_SYNC_ERR { \
    cudaError_t cudaDebugSyncErr = cudaDeviceSynchronize(); \
    if (cudaDebugSyncErr != cudaSuccess) { \
        _ftprintf(stderr, _T("CUDA error(1): %s(%d): %s: %s\n"), char_to_tstring(__FILE__).c_str(), __LINE__, char_to_tstring(__func__).c_str(), get_err_mes(err_to_rgy(cudaDebugSyncErr))); \
        return err_to_rgy(cudaDebugSyncErr); \
    } \
    cudaDebugSyncErr = cudaGetLastError(); \
    if (cudaDebugSyncErr != cudaSuccess) { \
        _ftprintf(stderr, _T("CUDA error(2): %s(%d): %s: %s\n"), char_to_tstring(__FILE__).c_str(), __LINE__, char_to_tstring(__func__).c_str(), get_err_mes(err_to_rgy(cudaDebugSyncErr))); \
        return err_to_rgy(cudaDebugSyncErr); \
    } \
}
#else
#define CUDA_DEBUG_SYNC 
#define CUDA_DEBUG_SYNC_ERR
#endif


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

static inline cudaMemcpyKind getCudaMemcpyKind(RGY_MEM_TYPE inputDevice, RGY_MEM_TYPE outputDevice) {
    if (inputDevice != RGY_MEM_TYPE_CPU) {
        return (outputDevice != RGY_MEM_TYPE_CPU) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    } else {
        return (outputDevice != RGY_MEM_TYPE_CPU) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
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

static const TCHAR *getCudaMemcpyKindStr(RGY_MEM_TYPE inputDevice, RGY_MEM_TYPE outputDevice) {
    return getCudaMemcpyKindStr(getCudaMemcpyKind(inputDevice, outputDevice));
}

static RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemcpy2D(dst->ptr[0], dst->pitch[0], src->ptr[0], src->pitch[0], width_byte, dst->height, getCudaMemcpyKind(src->mem_type, dst->mem_type)));
}

static RGY_ERR copyPlaneAsync(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemcpy2DAsync(dst->ptr[0], dst->pitch[0], src->ptr[0], src->pitch[0], width_byte, dst->height, getCudaMemcpyKind(src->mem_type, dst->mem_type), stream));
}

RGY_ERR convertPlaneBitDepth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const sInputCrop *pCrop, cudaStream_t stream);

RGY_ERR copyPlaneAlphaAsync(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream);

static RGY_ERR copyPlaneAsyncWithCrop(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *crop, cudaStream_t stream) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    uint8_t *srcPtr = src->ptr[0] + crop->e.up * src->pitch[0] + crop->e.left * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemcpy2DAsync(dst->ptr[0], dst->pitch[0], srcPtr, src->pitch[0], width_byte, dst->height, getCudaMemcpyKind(src->mem_type, dst->mem_type), stream));
}

static RGY_ERR copyPlaneField(RGYFrameInfo *dst, const RGYFrameInfo *src, const bool dstTopField, const bool srcTopField) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemcpy2D(
        dst->ptr[0] + ((dstTopField) ? 0 : dst->pitch[0]),
        dst->pitch[0] << 1,
        src->ptr[0] + ((srcTopField) ? 0 : src->pitch[0]),
        src->pitch[0] << 1,
        width_byte,
        dst->height >> 1,
        getCudaMemcpyKind(src->mem_type, dst->mem_type)));
}

static RGY_ERR copyPlaneFieldAsync(RGYFrameInfo *dst, const RGYFrameInfo *src, const bool dstTopField, const bool srcTopField, cudaStream_t stream) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemcpy2DAsync(
        dst->ptr[0] + ((dstTopField) ? 0 : dst->pitch[0]),
        dst->pitch[0] << 1,
        src->ptr[0] + ((srcTopField) ? 0 : src->pitch[0]),
        src->pitch[0] << 1,
        width_byte,
        dst->height >> 1,
        getCudaMemcpyKind(src->mem_type, dst->mem_type), stream));
}

static RGY_ERR setPlane(RGYFrameInfo *dst, int value) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemset2D(
        dst->ptr[0],
        dst->pitch[0],
        value,
        width_byte,
        dst->height));
}

static RGY_ERR setPlaneAsync(RGYFrameInfo *dst, int value, cudaStream_t stream) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemset2DAsync(
        dst->ptr[0],
        dst->pitch[0],
        value,
        width_byte,
        dst->height, stream));
}

static RGY_ERR setPlaneField(RGYFrameInfo *dst, int value, bool topField) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemset2D(
        dst->ptr[0] + ((topField) ? 0 : dst->pitch[0]),
        dst->pitch[0] << 1,
        value,
        width_byte,
        dst->height >> 1));
}

static RGY_ERR setPlaneFieldAsync(RGYFrameInfo *dst, int value, bool topField, cudaStream_t stream) {
    const int width_byte = dst->width * bytesPerPix(dst->csp);
    return err_to_rgy(cudaMemset2DAsync(
        dst->ptr[0] + ((topField) ? 0 : dst->pitch[0]),
        dst->pitch[0] << 1,
        value,
        width_byte,
        dst->height >> 1, stream));
}

static RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlane(&dstPlane, &srcPlane);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR copyFrameAsync(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneAsync(&dstPlane, &srcPlane, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR copyFrameField(RGYFrameInfo *dst, const RGYFrameInfo *src, const bool dstTopField, const bool srcTopField) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneField(&dstPlane, &srcPlane, dstTopField, srcTopField);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR copyFrameFieldAsync(RGYFrameInfo *dst, const RGYFrameInfo *src, const bool dstTopField, const bool srcTopField, cudaStream_t stream) {
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        const auto srcPlane = getPlane(src, (RGY_PLANE)i);
        auto dstPlane = getPlane(dst, (RGY_PLANE)i);
        auto ret = copyPlaneFieldAsync(&dstPlane, &srcPlane, dstTopField, srcTopField, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
    }
    return RGY_ERR_NONE;
}

enum class CUFrameBufType {
    Unknown,
    devPtr,      // cudaMalloc, cudaMallocPitch, cudaAllocHost, cudaFree, cudaFreeHost
    CUdevivePtr, // cuMemAlloc, cuMemAllocPitch, cuMemAllocHost, cuMemFree, cuMemFreeHost
    CUVID,       // CUVIDPARSERDISPINFO
    EncDevWrap,
    EncHostWrap,
};

struct CUFrameBufBase : public RGYFrame {
public:
    RGYFrameInfo frame;
    cudaEvent_t event;
    CUFrameBufType framebuftype;
    std::unique_ptr<CUFrameBufBase> refFrameHost;
    CUFrameBufBase()
        : frame(), event(), framebuftype(CUFrameBufType::Unknown), refFrameHost() {
        cudaEventCreate(&event);
    };
    CUFrameBufBase(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frame(), event(), framebuftype(CUFrameBufType::Unknown), refFrameHost() {
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.mem_type = RGY_MEM_TYPE_GPU;
        cudaEventCreate(&event);
    };
    CUFrameBufBase(const RGYFrameInfo& _info)
        : frame(_info), event(), framebuftype(CUFrameBufType::Unknown), refFrameHost() {
        cudaEventCreate(&event);
    };
    virtual ~CUFrameBufBase() {
        refFrameHost.reset();
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
    cudaEvent_t getEvent() const { return event; }
    void releasePtr() {
        memset(frame.ptr, 0, sizeof(frame.ptr));
        memset(frame.pitch, 0, sizeof(frame.pitch));
    }
    RGY_ERR alloc(bool singleAlloc = false, int align = 0) {
        frame.mem_type = RGY_MEM_TYPE_GPU;
        frame.singleAlloc = singleAlloc;
        return allocMemory(align);
    }
    RGY_ERR alloc(int width, int height, RGY_CSP csp = RGY_CSP_NV12, bool singleAlloc = false, int align = 0) {
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.mem_type = RGY_MEM_TYPE_GPU;
        return alloc(singleAlloc, align);
    }
    RGY_ERR allocHost(bool singleAlloc = false, int align = 0) {
        frame.mem_type = RGY_MEM_TYPE_CPU;
        frame.singleAlloc = singleAlloc;
        return allocMemory(align);
    }
    RGY_ERR allocHost(int width, int height, RGY_CSP csp = RGY_CSP_NV12, bool singleAlloc = false, int align = 0) {
        frame.width = width;
        frame.height = height;
        frame.csp = csp;
        frame.mem_type = RGY_MEM_TYPE_CPU;
        return allocHost(singleAlloc, align);
    }
    virtual std::pair<RGY_ERR, std::unique_ptr<CUFrameBufBase>> createHost(const RGYFrameInfo& info) const = 0;
    RGY_ERR allocRefHost(const bool singleAlloc = false, int align = 0) {
        if (frame.mem_type == RGY_MEM_TYPE_CPU) {
            return RGY_ERR_INVALID_CALL;
        }
        auto ret = createHost(frame);
        if (ret.first != RGY_ERR_NONE) {
            return ret.first;
        }
        refFrameHost = std::move(ret.second);
        return refFrameHost->allocHost(singleAlloc, align);
    }
    RGYFrame *getRefHostFrame() { return refFrameHost.get(); }
    RGY_ERR copyFrameFromHostRef(cudaStream_t stream) {
        if (!refFrameHost) {
            return RGY_ERR_INVALID_PARAM;
        }
        auto ret = copyFrameAsync(&frame, &refFrameHost->frame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        setPropertyFrom(refFrameHost.get());
        return RGY_ERR_NONE;
    }
    RGY_ERR copyFrameToHostRef(cudaStream_t stream) {
        if (!refFrameHost) {
            return RGY_ERR_INVALID_PARAM;
        }
        auto ret = copyFrameAsync(&refFrameHost->frame, &frame, stream);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        refFrameHost->setPropertyFrom(this);
        return RGY_ERR_NONE;
    }
    void clear() {
        clearMemory();
    }
    CUFrameBufType bufType() const { return framebuftype; };
    virtual bool isempty() const { return !frame.ptr[0]; }
    virtual void setTimestamp(uint64_t timestamp) override { frame.timestamp = timestamp; if (refFrameHost) refFrameHost->setTimestamp(timestamp); }
    virtual void setDuration(uint64_t frame_duration) override { frame.duration = frame_duration; if (refFrameHost) refFrameHost->setDuration(frame_duration); }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; if (refFrameHost) refFrameHost->setPicstruct(picstruct); }
    virtual void setInputFrameId(int inputFrameId) override { frame.inputFrameId = inputFrameId; if (refFrameHost) refFrameHost->setInputFrameId(inputFrameId); }
    virtual void setFlags(RGY_FRAME_FLAGS flag) override { frame.flags = flag; if (refFrameHost) refFrameHost->setFlags(flag); };
    virtual void clearDataList() override { frame.dataList.clear(); if (refFrameHost) refFrameHost->clearDataList(); };
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return frame.dataList; };
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return frame.dataList; };
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { frame.dataList = dataList; if (refFrameHost) refFrameHost->setDataList(dataList); };
    virtual RGYFrameInfo getInfo() const { return frame; }
    void setSingleAlloc(bool singleAlloc) { frame.singleAlloc = singleAlloc; }
protected:
    CUFrameBufBase(const CUFrameBufBase &) = delete;
    void operator =(const CUFrameBufBase &) = delete;

    virtual RGY_ERR memmalloc(void **mem, size_t& memPitch, const int align, const int widthByte, const int totalHeight) = 0;
    virtual RGY_ERR memfree(uint8_t **mem) = 0;
    RGY_ERR allocMemory(const int align) {
        const int pixsize = bytesPerPix(frame.csp);
        clearMemory();
        if (frame.singleAlloc) {
            int totalHeight = 0;
            for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
                totalHeight += getPlane(&frame, (RGY_PLANE)i).height;
            }
            const int widthByte = frame.width * pixsize;
            size_t memPitch = ALIGN(widthByte, (align) ? align : 128); //このアライメントはRGY_MEM_TYPE_CPUのとき、読み込み時の色変換の並列化のために必要
            void *mem = nullptr;
            auto sts = memmalloc(&mem, memPitch, align, widthByte, totalHeight);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            frame.pitch[0] = (int)memPitch;
            frame.ptr[0] = (uint8_t *)mem;
            return RGY_ERR_NONE;
        }

        for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
            const auto plane = getPlane(&frame, (RGY_PLANE)i);
            const int widthByte = plane.width * pixsize;
            size_t memPitch = ALIGN(widthByte, (align) ? align : 128); //このアライメントはRGY_MEM_TYPE_CPUのとき、読み込み時の色変換の並列化のために必要
            void *mem = nullptr;
            auto sts = memmalloc(&mem, memPitch, align, widthByte, plane.height);
            if (sts != RGY_ERR_NONE) {
                for (int j = i - 1; j >= 0; j--) {
                    memfree(&frame.ptr[i]);
                }
                return sts;
            }
            frame.pitch[i] = (int)memPitch;
            frame.ptr[i] = (uint8_t *)mem;
        }
        return RGY_ERR_NONE;
    }
    void clearMemory() {
        for (int i = 0; i < ((frame.singleAlloc) ? 1 : RGY_CSP_PLANES[frame.csp]); i++) {
            memfree(&frame.ptr[i]);
        }
        memset(frame.ptr, 0, sizeof(frame.ptr));
        memset(frame.pitch, 0, sizeof(frame.pitch));
    }
};

struct CUFrameBuf : public CUFrameBufBase {
public:
    CUFrameBuf()
        : CUFrameBufBase() {
        framebuftype = CUFrameBufType::devPtr;
    };
    CUFrameBuf(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : CUFrameBufBase(width, height, csp) {
        framebuftype = CUFrameBufType::devPtr;
    };
    CUFrameBuf(const RGYFrameInfo& _info)
        : CUFrameBufBase(_info) {
        framebuftype = CUFrameBufType::devPtr;
    };
    virtual ~CUFrameBuf() {
        clear();
    }
    virtual std::pair<RGY_ERR, std::unique_ptr<CUFrameBufBase>> createHost(const RGYFrameInfo& info) const override {
        return std::make_pair(RGY_ERR_NONE, std::unique_ptr<CUFrameBufBase>(new CUFrameBuf(info)));
    }
    RGY_ERR copyFrame(const RGYFrameInfo *src) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrame(&frame, src);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR copyFrameAsync(const RGYFrameInfo *src, cudaStream_t stream) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrameAsync(&frame, src, stream);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
protected:
    CUFrameBuf(const CUFrameBuf &) = delete;
    void operator =(const CUFrameBuf &) = delete;
    virtual RGY_ERR memmalloc(void **mem, size_t& memPitch, const int align, const int widthByte, const int totalHeight) override {
        auto sts = RGY_ERR_NONE;
        if (frame.mem_type == RGY_MEM_TYPE_CPU) {
            sts = err_to_rgy(cudaMallocHost(mem, totalHeight * memPitch));
        } else if (align) {
            sts = err_to_rgy(cudaMalloc(mem, totalHeight * memPitch));
        } else {
            memPitch = 0;
            sts = err_to_rgy(cudaMallocPitch(mem, &memPitch, widthByte, totalHeight));
        }
        return sts;
    }
    virtual RGY_ERR memfree(uint8_t **mem) override {
        auto sts = RGY_ERR_NONE;
        if (mem && *mem) {
            if (frame.mem_type == RGY_MEM_TYPE_CPU) {
                cudaFreeHost(*mem);
            } else {
                cudaFree(*mem);
            }
            *mem = nullptr;
        }
        return sts;
    }
public:
};

struct CUFrameDevPtr : public CUFrameBufBase {
public:
    CUFrameDevPtr()
        : CUFrameBufBase() {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    CUFrameDevPtr(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : CUFrameBufBase(width, height, csp) {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    CUFrameDevPtr(const RGYFrameInfo& _info)
        : CUFrameBufBase(_info) {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    virtual ~CUFrameDevPtr() {
        clear();
    }
    virtual std::pair<RGY_ERR, std::unique_ptr<CUFrameBufBase>> createHost(const RGYFrameInfo& info) const override {
        return std::make_pair(RGY_ERR_NONE, std::unique_ptr<CUFrameBufBase>(new CUFrameDevPtr(info)));
    }
    RGY_ERR copyFrame(const RGYFrameInfo *src) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrame(&frame, src);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR copyFrameAsync(const RGYFrameInfo *src, cudaStream_t stream) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrameAsync(&frame, src, stream);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
protected:
    CUFrameDevPtr(const CUFrameDevPtr &) = delete;
    void operator =(const CUFrameDevPtr &) = delete;
    virtual RGY_ERR memmalloc(void **mem, size_t& memPitch, const int align, const int widthByte, const int totalHeight) override {
        auto sts = RGY_ERR_NONE;
        if (frame.mem_type == RGY_MEM_TYPE_CPU) {
            sts = err_to_rgy(cuMemAllocHost(mem, totalHeight * memPitch));
        } else if (align) {
            CUdeviceptr ptr;
            sts = err_to_rgy(cuMemAlloc(&ptr, totalHeight * memPitch));
            *mem = (void *)ptr;
        } else {
            memPitch = 0;
            CUdeviceptr ptr;
            sts = err_to_rgy(cuMemAllocPitch(&ptr, &memPitch, widthByte, totalHeight, 1));
            *mem = (void *)ptr;
        }
        return sts;
    }
    virtual RGY_ERR memfree(uint8_t **mem) override {
        auto sts = RGY_ERR_NONE;
        if (mem && *mem) {
            if (frame.mem_type == RGY_MEM_TYPE_CPU) {
                cuMemFreeHost(*mem);
            } else {
                cuMemFree((CUdeviceptr)(*mem));
            }
            *mem = nullptr;
        }
        return sts;
    }
};


#if 0
struct CUFrameCUArray : public CUFrameBufBase {
public:
    CUFrameCUArray()
        : CUFrameBufBase() {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    CUFrameCUArray(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : CUFrameBufBase(width, height, csp) {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    CUFrameCUArray(const RGYFrameInfo& _info)
        : CUFrameBufBase(_info) {
        framebuftype = CUFrameBufType::CUdevivePtr;
    };
    virtual ~CUFrameCUArray() {
        clear();
    }
    RGY_ERR copyFrame(const RGYFrameInfo *src) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrame(&frame, src);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR copyFrameAsync(const RGYFrameInfo *src, cudaStream_t stream) {
        if (frame.ptr[0] == nullptr || !cmpFrameInfoCspResolution(&frame, src)) {

        }
        auto ret = ::copyFrameAsync(&frame, src, stream);
        if (ret == RGY_ERR_NONE) {
            copyFrameProp(&frame, src);
        }
        return RGY_ERR_NONE;
    }
protected:
    CUFrameCUArray(const CUFrameCUArray &) = delete;
    void operator =(const CUFrameCUArray &) = delete;
    virtual RGY_ERR memmalloc(void **mem, size_t& memPitch, const int align, const int widthByte, const int totalHeight) override {
        auto sts = RGY_ERR_NONE;
        if (frame.mem_type == RGY_MEM_TYPE_CPU || align) {
            return RGY_ERR_UNSUPPORTED;
        } else {
            CUDA_ARRAY_DESCRIPTOR cuArrayDesc = { 0 };
            cuArrayDesc.Format = RGY_CSP_BIT_DEPTH[frame.csp] == 32 ? CU_AD_FORMAT_FLOAT : (RGY_CSP_BIT_DEPTH[frame.csp] > 8 ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8);
            cuArrayDesc.Height = frame.height;
            cuArrayDesc.NumChannels = 4;
            cuArrayDesc.Width = frame.width;

            CUarray ptr;
            sts = err_to_rgy(cuArrayCreate(&ptr, &cuArrayDesc));
            *mem = (void *)ptr;
        }
        return sts;
    }
    virtual RGY_ERR memfree(uint8_t **mem) override {
        auto sts = RGY_ERR_NONE;
        if (mem && *mem) {
            if (frame.mem_type == RGY_MEM_TYPE_CPU) {
                return RGY_ERR_UNSUPPORTED;
            } else {
                cuArrayDestroy((CUarray)(*mem));
            }
            *mem = nullptr;
        }
        return sts;
    }
};
#endif

struct CUFrameBufPair {
public:
    CUFrameBuf frameDev;
    CUFrameBuf frameHost;
    cudaEvent_t event;
    CUFrameBufPair()
        : frameDev(), frameHost(), event() {
        cudaEventCreate(&event);
    };
    CUFrameBufPair(int width, int height, RGY_CSP csp = RGY_CSP_NV12)
        : frameDev(), frameHost(), event() {
        frameDev.frame.width = width;
        frameDev.frame.height = height;
        frameDev.frame.csp = csp;
        frameDev.frame.mem_type = RGY_MEM_TYPE_GPU;

        frameHost.frame = frameDev.frame;
        frameHost.frame.mem_type = RGY_MEM_TYPE_CPU;

        cudaEventCreate(&event);
    };
protected:
    CUFrameBufPair(const CUFrameBufPair &) = delete;
    void operator =(const CUFrameBufPair &) = delete;
public:
    RGY_ERR allocHost() {
        return frameHost.allocHost();
    }
    RGY_ERR allocDev() {
        return frameDev.alloc();
    }
    RGY_ERR alloc() {
        clearHost();
        clearDev();
        auto err = allocDev();
        if (err != RGY_ERR_NONE) return err;
        return allocHost();
    }
    RGY_ERR alloc(int width, int height, RGY_CSP csp = RGY_CSP_NV12) {
        clearHost();
        clearDev();

        frameDev.frame.width = width;
        frameDev.frame.height = height;
        frameDev.frame.csp = csp;
        frameDev.frame.mem_type = RGY_MEM_TYPE_GPU;

        frameHost.frame = frameDev.frame;
        frameHost.frame.mem_type = RGY_MEM_TYPE_CPU;

        return alloc();
    }
    RGY_ERR copyDtoHAsync(cudaStream_t stream = 0) {
        return copyFrameAsync(&frameHost.frame, &frameDev.frame, stream);
    }
    RGY_ERR copyDtoH() {
        return copyFrame(&frameHost.frame, &frameDev.frame);
    }
    RGY_ERR copyHtoDAsync(cudaStream_t stream = 0) {
        return copyFrameAsync(&frameDev.frame, &frameHost.frame, stream);
    }
    RGY_ERR copyHtoD() {
        return copyFrame(&frameDev.frame, &frameHost.frame);
    }
public:
    void clearHost() {
        frameHost.clear();
    }
    void clearDev() {
        frameDev.clear();
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
    RGY_ERR alloc() {
        if (ptr) {
            cudaFree(ptr);
        }
        auto ret = RGY_ERR_NONE;
        if (nSize > 0) {
            ret = err_to_rgy(cudaMalloc(&ptr, nSize));
        } else {
            ret = RGY_ERR_UNSUPPORTED;
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
    RGY_ERR alloc() {
        if (ptrDevice) {
            cudaFree(ptrDevice);
        }
        auto ret = RGY_ERR_NONE;
        if (nSize > 0) {
            ret = err_to_rgy(cudaMalloc(&ptrDevice, nSize));
            if (ret == RGY_ERR_NONE) {
                ret = err_to_rgy(cudaMallocHost(&ptrHost, nSize));
            }
        } else {
            ret = RGY_ERR_UNSUPPORTED;
        }
        return ret;
    }
    RGY_ERR alloc(size_t _nSize) {
        nSize = _nSize;
        return alloc();
    }
    RGY_ERR copyDtoHAsync(cudaStream_t stream = 0) {
        return err_to_rgy(cudaMemcpyAsync(ptrHost, ptrDevice, nSize, cudaMemcpyDeviceToHost, stream));
    }
    RGY_ERR copyDtoH() {
        return err_to_rgy(cudaMemcpy(ptrHost, ptrDevice, nSize, cudaMemcpyDeviceToHost));
    }
    RGY_ERR copyHtoDAsync(cudaStream_t stream = 0) {
        return err_to_rgy(cudaMemcpyAsync(ptrDevice, ptrHost, nSize, cudaMemcpyHostToDevice, stream));
    }
    RGY_ERR copyHtoD() {
        return err_to_rgy(cudaMemcpy(ptrDevice, ptrHost, nSize, cudaMemcpyHostToDevice));
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
