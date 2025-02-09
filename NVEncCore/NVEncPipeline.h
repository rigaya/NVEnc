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
#ifndef __NVENC_PIPELINE_H__
#define __NVENC_PIPELINE_H__

#include <thread>
#include <future>
#include <atomic>
#include <deque>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>

#include "rgy_osdep.h"
#include "rgy_version.h"
#include "rgy_err.h"
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_input.h"
#include "rgy_input_avcodec.h"
#include "rgy_input_sm.h"
#include "rgy_output.h"
#include "rgy_output_avcodec.h"
#include "rgy_filter.h"
#include "rgy_thread.h"
#include "rgy_timecode.h"

#include "NVEncParam.h"
#include "NVEncDevice.h"
#include "NVEncUtil.h"
#include "NVEncFilter.h"
#include "NVEncFilterSsim.h"
#include "NvHWEncoder.h"
#include "dynlink_nvcuvid.h"

#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4834)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wunused-result")
RGY_DISABLE_WARNING_STR("-Wtautological-compare")
#define TTMATH_NOASM
#include "ttmath/ttmath.h"
#if _M_IX86
typedef ttmath::Int<4> ttint128;
#else
typedef ttmath::Int<2> ttint128;
#endif
RGY_DISABLE_WARNING_POP
#pragma warning(pop)

static const int RGY_WAIT_INTERVAL = 60000;

using unique_cuevent = unique_ptr<cudaEvent_t, cudaevent_deleter>;

// cuvidフレームのラッパー
struct CUFrameCuvid : public CUFrameBufBase {
    CUFrameCuvid() : CUFrameBufBase(),
        m_decoder(nullptr),
        m_dispInfo(),
        m_oVPP() {
        framebuftype = CUFrameBufType::CUVID;
        frame.singleAlloc = true;
        frame.ptr[0] = nullptr;
        frame.pitch[0] = 0;
    };
    CUFrameCuvid(void *decoder, const RGYFrameInfo& _info, std::shared_ptr<CUVIDPARSERDISPINFO> dispInfo) : CUFrameBufBase(_info),
        m_decoder(decoder),
        m_dispInfo(dispInfo),
        m_oVPP() {
        framebuftype = CUFrameBufType::CUVID;
        frame.singleAlloc = true;
        frame.ptr[0] = nullptr;
        frame.pitch[0] = 0;
    };
    virtual ~CUFrameCuvid() {
        clear();
        m_dispInfo.reset();
    }
    virtual std::pair<RGY_ERR, std::unique_ptr<CUFrameBufBase>> createHost([[maybe_unused]] const RGYFrameInfo& info) const override {
        return std::make_pair(RGY_ERR_UNSUPPORTED, nullptr);
    }
    void *decoder() { return m_decoder; }
    std::shared_ptr<CUVIDPARSERDISPINFO>& dispInfo() { return m_dispInfo; }
    const CUVIDPROCPARAMS& oVPP() const { return m_oVPP; }
    void setOVPP(const CUVIDPROCPARAMS& oVPP) { m_oVPP = oVPP; }
    RGY_ERR mapFrame() {
        CUdeviceptr dMappedFrame = 0;
        uint32_t pitch = 0;
        // cuvidMapVideoFrameは複数のフレーム対して連続で実行できないので注意
        // 必ず mapFrameを読んだら次にmapFrameを呼ぶ前にunmapFrameを呼ぶこと
        auto sts = err_to_rgy(cuvidMapVideoFrame(m_decoder, m_dispInfo->picture_index, &dMappedFrame, &pitch, &m_oVPP));
        if (sts == RGY_ERR_NONE) {
            frame.singleAlloc = true;
            frame.ptr[0] = (uint8_t *)dMappedFrame;
            frame.pitch[0] = pitch;
        }
        return sts;
    }
    // unmapする
    // 基本的にはmap前に明示的に呼ぶように
    // あるいはデストラクタ->memfree経由でも呼ばれる
    void unmapFrame() {
        memfree(frame.ptr);
    }
protected:
    CUFrameCuvid(const CUFrameCuvid &) = delete;
    void operator =(const CUFrameCuvid &) = delete;
    virtual RGY_ERR memmalloc([[maybe_unused]] void **mem, [[maybe_unused]] size_t& memPitch, [[maybe_unused]] const int align, [[maybe_unused]] const int widthByte, [[maybe_unused]] const int totalHeight) override {
        return RGY_ERR_UNSUPPORTED;
    }
    virtual RGY_ERR memfree(uint8_t **mem) override {
        if (mem[0]) {
            auto sts = err_to_rgy(cuvidUnmapVideoFrame(m_decoder, (CUdeviceptr)mem[0]));
            if (sts != RGY_ERR_NONE) {
                fprintf(stderr, "Failed to unamp cuvid frame.\n");
            }
            mem[0] = nullptr;
        }
        return RGY_ERR_NONE;
    }

    void *m_decoder;
    std::shared_ptr<CUVIDPARSERDISPINFO> m_dispInfo;
    CUVIDPROCPARAMS m_oVPP;
};

struct CUFrameEnc : public CUFrameBufBase {
    CUFrameEnc() : CUFrameBufBase(),
        m_encoder(nullptr),
        m_encBuffer(nullptr) {
        framebuftype = CUFrameBufType::EncDevWrap;
        frame.singleAlloc = true;
    };
    CUFrameEnc(EncodeBuffer *encBuffer, NVEncoder *encoder, bool rgbAsYUV444) : CUFrameBufBase(),
        m_encoder(encoder),
        m_encBuffer(encBuffer) {
        framebuftype = CUFrameBufType::EncDevWrap;
        frame.singleAlloc = true;
        frame.width = encBuffer->stInputBfr.dwWidth;
        frame.height = encBuffer->stInputBfr.dwHeight;
        frame.pitch[0] = encBuffer->stInputBfr.uNV12Stride;
        frame.mem_type = RGY_MEM_TYPE_GPU;
        frame.csp = getEncCsp(encBuffer->stInputBfr.bufferFmt, encBuffer->stInputBfrAlpha.nvRegisteredResource != nullptr, rgbAsYUV444);
    };
    virtual ~CUFrameEnc() {
        clear();
    }
    virtual std::pair<RGY_ERR, std::unique_ptr<CUFrameBufBase>> createHost([[maybe_unused]] const RGYFrameInfo& info) const override {
        return std::make_pair(RGY_ERR_UNSUPPORTED, nullptr);
    }
    virtual RGY_ERR map() = 0;
    virtual void unmap() = 0;
    EncodeBuffer *encBuffer() { return m_encBuffer; }
protected:
    CUFrameEnc(const CUFrameEnc &) = delete;
    void operator =(const CUFrameEnc &) = delete;
    virtual RGY_ERR memmalloc([[maybe_unused]] void **mem, [[maybe_unused]] size_t& memPitch, [[maybe_unused]] const int align, [[maybe_unused]] const int widthByte, [[maybe_unused]] const int totalHeight) override {
        return RGY_ERR_UNSUPPORTED;
    }
    virtual RGY_ERR memfree(uint8_t **mem) override {
        if (mem[0]) {
            unmap();
            mem[0] = nullptr;
        }
        return RGY_ERR_NONE;
    }

    NVEncoder *m_encoder;
    EncodeBuffer *m_encBuffer;
};

struct CUFrameEncDevWrap : public CUFrameEnc {
    CUFrameEncDevWrap() : CUFrameEnc() {
        framebuftype = CUFrameBufType::EncDevWrap;
        frame.mem_type = RGY_MEM_TYPE_GPU;
        frame.singleAlloc = true;
    };
    CUFrameEncDevWrap(EncodeBuffer *encBuffer, NVEncoder *encoder, bool rgbAsYUV444) : CUFrameEnc(encBuffer, encoder, rgbAsYUV444) {
        framebuftype = CUFrameBufType::EncDevWrap;
        frame.mem_type = RGY_MEM_TYPE_GPU;
        frame.ptr[0] = (uint8_t *)encBuffer->stInputBfr.pNV12devPtr;
        frame.pitch[0] = m_encBuffer->stInputBfr.uNV12Stride;
        frame.singleAlloc = true;
    };
    virtual ~CUFrameEncDevWrap() {
        clear();
    }
    virtual RGY_ERR map() override {
        auto nvencret = m_encoder->NvEncMapInputResource(m_encBuffer->stInputBfr.nvRegisteredResource, &m_encBuffer->stInputBfr.hInputSurface);
        if (nvencret != NV_ENC_SUCCESS) {
            return err_to_rgy(nvencret);
        }
        if (m_encBuffer->stInputBfrAlpha.nvRegisteredResource) {
            nvencret = m_encoder->NvEncMapInputResource(m_encBuffer->stInputBfrAlpha.nvRegisteredResource, &m_encBuffer->stInputBfrAlpha.hInputSurface);
                if (nvencret != NV_ENC_SUCCESS) {
                    return err_to_rgy(nvencret);
            }
        }
        frame.ptr[0] = (uint8_t *)m_encBuffer->stInputBfr.pNV12devPtr;
        frame.pitch[0] = m_encBuffer->stInputBfr.uNV12Stride;
        return RGY_ERR_NONE;
    }
    virtual void unmap() override {
        m_encoder->NvEncUnmapInputResource(m_encBuffer->stInputBfr.hInputSurface);
        if (m_encBuffer->stInputBfrAlpha.nvRegisteredResource) {
            m_encoder->NvEncUnmapInputResource(m_encBuffer->stInputBfrAlpha.hInputSurface);
        }
        frame.ptr[0] = nullptr;
        frame.pitch[0] = 0;
    }
protected:
    CUFrameEncDevWrap(const CUFrameEncDevWrap &) = delete;
    void operator =(const CUFrameEncDevWrap &) = delete;
};

struct CUFrameEncHostWrap : public CUFrameEnc {
    CUFrameEncHostWrap() : CUFrameEnc() {
        framebuftype = CUFrameBufType::EncHostWrap;
        frame.mem_type = RGY_MEM_TYPE_CPU;
        frame.singleAlloc = true;
    };
    CUFrameEncHostWrap(EncodeBuffer *encBuffer, NVEncoder *encoder, bool rgbAsYUV444) : CUFrameEnc(encBuffer, encoder, rgbAsYUV444) {
        framebuftype = CUFrameBufType::EncHostWrap;
        frame.mem_type = RGY_MEM_TYPE_CPU;
        frame.singleAlloc = true;
    };
    virtual ~CUFrameEncHostWrap() {
        clear();
    }
    virtual RGY_ERR map() override {
        //インタレ保持の場合は、NvEncCreateInputBuffer経由でフレームを渡さないと正常にエンコードできない
        uint32_t lockedPitch = 0;
        unsigned char *pInputSurface = nullptr;
        m_encoder->NvEncLockInputBuffer(m_encBuffer->stInputBfr.hInputSurface, (void**)&pInputSurface, &lockedPitch);
        frame.ptr[0] = (uint8_t *)pInputSurface;
        frame.pitch[0] = lockedPitch;
        return RGY_ERR_NONE;
    }
    virtual void unmap() override {
        m_encoder->NvEncUnlockInputBuffer(m_encBuffer->stInputBfr.hInputSurface);
        if (m_encBuffer->stInputBfrAlpha.nvRegisteredResource) {
            m_encoder->NvEncUnlockInputBuffer(m_encBuffer->stInputBfrAlpha.hInputSurface);
        }
    }
protected:
    CUFrameEncHostWrap(const CUFrameEncHostWrap &) = delete;
    void operator =(const CUFrameEncHostWrap &) = delete;
};


enum RGYRunState {
    RGY_STATE_STOPPED,
    RGY_STATE_RUNNING,
    RGY_STATE_ERROR,
    RGY_STATE_ABORT,
    RGY_STATE_EOF
};

struct VppVilterBlock {
    VppFilterType type;
    std::vector<std::unique_ptr<NVEncFilter>> vppnv;

    VppVilterBlock(std::vector<std::unique_ptr<NVEncFilter>>& filter) : type(VppFilterType::FILTER_CUDA), vppnv(std::move(filter)) {};
};

enum class PipelineTaskOutputType {
    UNKNOWN,
    SURFACE,
    BITSTREAM
};

enum class PipelineTaskSurfaceType {
    UNKNOWN,
    CUBUF,
    CUDEV,
    CUVID,
    ENCDEV,
    ENCHOST
};

class PipelineTaskSurface {
private:
    RGYFrame *surf;
    std::atomic<int> *ref;
public:
    PipelineTaskSurface() : surf(nullptr), ref(nullptr) {};
    PipelineTaskSurface(std::pair<RGYFrame *, std::atomic<int> *> surf_) : PipelineTaskSurface(surf_.first, surf_.second) {};
    PipelineTaskSurface(RGYFrame *surf_, std::atomic<int> *ref_) : surf(surf_), ref(ref_) { if (surf) (*ref)++; };
    PipelineTaskSurface(const PipelineTaskSurface& obj) : surf(obj.surf), ref(obj.ref) { if (surf) (*ref)++; }
    PipelineTaskSurface &operator=(const PipelineTaskSurface &obj) {
        if (this != &obj) { // 自身の代入チェック
            surf = obj.surf;
            ref = obj.ref;
            if (surf) (*ref)++;
        }
        return *this;
    }
    ~PipelineTaskSurface() { reset(); }
    void reset() { if (surf) (*ref)--; surf = nullptr; ref = nullptr; }
    bool operator !() const {
        return frame() == nullptr;
    }
    bool operator !=(const PipelineTaskSurface& obj) const { return frame() != obj.frame(); }
    bool operator ==(const PipelineTaskSurface& obj) const { return frame() == obj.frame(); }
    bool operator !=(std::nullptr_t) const { return frame() != nullptr; }
    bool operator ==(std::nullptr_t) const { return frame() == nullptr; }
    const CUFrameBuf *cubuf() const { return dynamic_cast<const CUFrameBuf*>(surf); }
    CUFrameBuf *cubuf() { return dynamic_cast<CUFrameBuf*>(surf); }
    const CUFrameDevPtr *cudev() const { return dynamic_cast<const CUFrameDevPtr*>(surf); }
    CUFrameDevPtr *cudev() { return dynamic_cast<CUFrameDevPtr*>(surf); }
    const CUFrameCuvid *cuvid() const { return dynamic_cast<const CUFrameCuvid*>(surf); }
    CUFrameCuvid *cuvid() { return dynamic_cast<CUFrameCuvid*>(surf); }
    const CUFrameEnc *enc() const { return dynamic_cast<const CUFrameEnc*>(surf); }
    CUFrameEnc *enc() { return dynamic_cast<CUFrameEnc*>(surf); }
    const RGYFrame *frame() const { return surf; }
    RGYFrame *frame() { return surf; }
};

// アプリ用の独自参照カウンタと組み合わせたクラス
class PipelineTaskSurfaces {
private:
    class PipelineTaskSurfacesPair {
    private:
        std::unique_ptr<RGYFrame> surf_;
        std::atomic<int> ref;
    public:
        PipelineTaskSurfacesPair(std::unique_ptr<RGYFrame> s) : surf_(std::move(s)), ref(0) {};
        
        bool isFree() const { return ref == 0; } // 使用されていないフレームかを返す
        PipelineTaskSurface getRef() { return PipelineTaskSurface(surf_.get(), &ref); };
        const RGYFrame *surf() const { return surf_.get(); }
        RGYFrame *surf() { return surf_.get(); }
        PipelineTaskSurfaceType type() const {
            if (!surf_) return PipelineTaskSurfaceType::UNKNOWN;
            if (dynamic_cast<const CUFrameBuf*>(surf_.get())) return PipelineTaskSurfaceType::CUBUF;
            if (dynamic_cast<const CUFrameDevPtr*>(surf_.get())) return PipelineTaskSurfaceType::CUDEV;
            if (dynamic_cast<const CUFrameCuvid*>(surf_.get())) return PipelineTaskSurfaceType::CUVID;
            if (dynamic_cast<const CUFrameEncDevWrap*>(surf_.get())) return PipelineTaskSurfaceType::ENCDEV;
            if (dynamic_cast<const CUFrameEncHostWrap*>(surf_.get())) return PipelineTaskSurfaceType::ENCHOST;
            return PipelineTaskSurfaceType::UNKNOWN;
        }
    };
    std::vector<std::unique_ptr<PipelineTaskSurfacesPair>> m_surfaces; // フレームと参照カウンタ
public:
    PipelineTaskSurfaces() : m_surfaces() {};
    virtual ~PipelineTaskSurfaces() {}

    void clear() {
        m_surfaces.clear();
    }
    template<typename T>
    void setSurfaces(std::vector<std::unique_ptr<T>>& frames) {
        clear();
        m_surfaces.resize(frames.size());
        for (size_t i = 0; i < m_surfaces.size(); i++) {
            m_surfaces[i] = std::make_unique<PipelineTaskSurfacesPair>(std::move(frames[i]));
        }
    }
    PipelineTaskSurface addSurface(std::unique_ptr<CUFrameBuf>& surf) {
        deleteFreedSurface();
        m_surfaces.push_back(std::move(std::unique_ptr<PipelineTaskSurfacesPair>(new PipelineTaskSurfacesPair(std::move(surf)))));
        return m_surfaces.back()->getRef();
    }
    PipelineTaskSurface addSurface(std::unique_ptr<CUFrameCuvid>& surf) {
        deleteFreedSurface();
        m_surfaces.push_back(std::move(std::unique_ptr<PipelineTaskSurfacesPair>(new PipelineTaskSurfacesPair(std::move(surf)))));
        return m_surfaces.back()->getRef();
    }

    PipelineTaskSurface getFreeSurf() {
        for (auto& s : m_surfaces) {
            if (s->isFree()) {
                return s->getRef();
            }
        }
        return PipelineTaskSurface();
    }
    PipelineTaskSurface get(CUFrameBufBase *frame) {
        auto s = findSurf(frame);
        if (s != nullptr) {
            return s->getRef();
        }
        return PipelineTaskSurface();
    }
    size_t bufCount() const { return m_surfaces.size(); }

    bool isAllFree() const {
        for (const auto& s : m_surfaces) {
            if (!s->isFree()) {
                return false;
            }
        }
        return true;
    }
    PipelineTaskSurfaceType type() const {
        if (m_surfaces.size() == 0) return PipelineTaskSurfaceType::UNKNOWN;
        return m_surfaces.front()->type();
    }
protected:
    void deleteFreedSurface() {
        for (auto it = m_surfaces.begin(); it != m_surfaces.end();) {
            if ((*it)->isFree()) {
                it = m_surfaces.erase(it);
            } else {
                it++;
            }
        }
    }
    PipelineTaskSurfacesPair *findSurf(RGYFrame *surf) {
        for (auto& s : m_surfaces) {
            if (s->surf() == surf) {
                return s.get();
            }
        }
        return nullptr;
    }
};

class PipelineTaskOutputDataCustom {
    int type;
public:
    PipelineTaskOutputDataCustom() {};
    virtual ~PipelineTaskOutputDataCustom() {};
};

class PipelineTaskOutputDataCheckPts : public PipelineTaskOutputDataCustom {
private:
    int64_t timestamp;
public:
    PipelineTaskOutputDataCheckPts() : PipelineTaskOutputDataCustom() {};
    PipelineTaskOutputDataCheckPts(int64_t timestampOverride) : PipelineTaskOutputDataCustom(), timestamp(timestampOverride) {};
    virtual ~PipelineTaskOutputDataCheckPts() {};
    int64_t timestampOverride() const { return timestamp; }
};

class PipelineTaskOutput {
protected:
    PipelineTaskOutputType m_type;
    std::unique_ptr<PipelineTaskOutputDataCustom> m_customData;
public:
    PipelineTaskOutput() : m_type(PipelineTaskOutputType::UNKNOWN), m_customData() {};
    PipelineTaskOutput(PipelineTaskOutputType type) : m_type(type), m_customData() {};
    PipelineTaskOutput(PipelineTaskOutputType type, std::unique_ptr<PipelineTaskOutputDataCustom>& customData) : m_type(type), m_customData(std::move(customData)) {};
    RGY_ERR waitsync([[maybe_unused]] uint32_t wait = RGY_WAIT_INTERVAL) {
        return RGY_ERR_NONE;
    }
    virtual void depend_clear() {};
    PipelineTaskOutputType type() const { return m_type; }
    const PipelineTaskOutputDataCustom *customdata() const { return m_customData.get(); }
    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] NVEncFilterSsim *videoQualityMetric) {
        return RGY_ERR_UNSUPPORTED;
    }
    virtual ~PipelineTaskOutput() {};
#if ENCODER_NVENC
    virtual RGY_ERR setDependCUStream([[maybe_unused]] cudaStream_t stream) { return RGY_ERR_NONE; }
#endif
};

class PipelineTaskOutputSurf : public PipelineTaskOutput {
protected:
    PipelineTaskSurface m_surf;
    std::unique_ptr<PipelineTaskOutput> m_dependencyFrame;
    std::vector<std::shared_ptr<cudaEvent_t>> m_cuevents;
    CUvideoctxlock m_vidCtxLock;
public:
    PipelineTaskOutputSurf(CUvideoctxlock vidCtxLock, PipelineTaskSurface surf) :
        PipelineTaskOutput(PipelineTaskOutputType::SURFACE), m_vidCtxLock(vidCtxLock), m_surf(surf), m_dependencyFrame(), m_cuevents() {
    };
    PipelineTaskOutputSurf(CUvideoctxlock vidCtxLock, PipelineTaskSurface surf, std::unique_ptr<PipelineTaskOutputDataCustom>& customData) :
        PipelineTaskOutput(PipelineTaskOutputType::SURFACE, customData), m_vidCtxLock(vidCtxLock), m_surf(surf), m_dependencyFrame(), m_cuevents() {
    };
    PipelineTaskOutputSurf(CUvideoctxlock vidCtxLock, PipelineTaskSurface surf, std::shared_ptr<cudaEvent_t>& cuevent) :
        PipelineTaskOutput(PipelineTaskOutputType::SURFACE),
        m_vidCtxLock(vidCtxLock), m_surf(surf), m_dependencyFrame(), m_cuevents() {
        m_cuevents.push_back(cuevent);
    };
    PipelineTaskOutputSurf(CUvideoctxlock vidCtxLock, PipelineTaskSurface surf, std::unique_ptr<PipelineTaskOutput>& dependencyFrame, std::shared_ptr<cudaEvent_t>& cuevent) :
        PipelineTaskOutput(PipelineTaskOutputType::SURFACE),
        m_vidCtxLock(vidCtxLock), m_surf(surf), m_dependencyFrame(std::move(dependencyFrame)), m_cuevents() {
        m_cuevents.push_back(cuevent);
    };
    virtual ~PipelineTaskOutputSurf() {
        depend_clear();
        m_surf.reset();
    };

    PipelineTaskSurface& surf() { return m_surf; }

    void addCUEvent(std::shared_ptr<cudaEvent_t>& cuevent) {
        m_cuevents.push_back(cuevent);
    }
    
    virtual RGY_ERR setDependCUStream(cudaStream_t stream) {
        auto sts = m_dependencyFrame->setDependCUStream(stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        for (auto& cuevent : m_cuevents) {
            if (cuevent != nullptr) {
                NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                auto cuerr = cudaStreamWaitEvent(stream, *cuevent.get(), 0);
                if (cuerr != cudaSuccess) {
                    return err_to_rgy(cuerr);
                }
            }
        }
        return RGY_ERR_NONE;
    }

    virtual void depend_clear() override {
        // m_cuevents内のイベントを待つ
        for (auto& cuevent : m_cuevents) {
            if (cuevent != nullptr) {
                NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                cudaEventSynchronize(*cuevent.get());
            }
        }
        m_cuevents.clear();
        m_dependencyFrame.reset();
    }
    
    void streamWaitCuEvents(cudaStream_t stream) {
        NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
        for (auto& cuevent : m_cuevents) {
            if (cuevent != nullptr) {
                cudaStreamWaitEvent(stream, *cuevent.get(), 0);
            }
        }
        m_cuevents.clear();
    }

    RGY_ERR writeCU(RGYOutput *writer) {
        auto cubuf = m_surf.cubuf();
        if (!cubuf) {
            return RGY_ERR_NULL_PTR;
        }
        if (cubuf->mem_type() != RGY_MEM_TYPE_CPU) {
            return RGY_ERR_UNSUPPORTED;
        }
        return writer->WriteNextFrame(cubuf);
    }
#if 0

    RGY_ERR writeCL(RGYOutput *writer, RGYOpenCLQueue *clqueue) {
        if (clqueue == nullptr) {
            return RGY_ERR_NULL_PTR;
        }
        auto clframe = m_surf.cl();
        auto err = clframe->queueMapBuffer(*clqueue, CL_MAP_READ); // CPUが読み込むためにmapする
        if (err != RGY_ERR_NONE) {
            return err;
        }
        clframe->mapWait();
        auto mappedframe = clframe->mappedHost();
        err = writer->WriteNextFrame(mappedframe);
        clframe->unmapBuffer();
        return err;
    }
#endif
    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] NVEncFilterSsim *videoQualityMetric) override {
        if (!writer || writer->getOutType() == OUT_TYPE_NONE) {
            return RGY_ERR_NOT_INITIALIZED;
        }
        if (writer->getOutType() != OUT_TYPE_SURFACE) {
            return RGY_ERR_INVALID_OPERATION;
        }
        auto err = writeCU(writer);
        return err;
    }
};

class PipelineTaskOutputBitstream : public PipelineTaskOutput {
protected:
    std::shared_ptr<RGYBitstream> m_bs;
public:
    PipelineTaskOutputBitstream(std::shared_ptr<RGYBitstream> bs) : PipelineTaskOutput(PipelineTaskOutputType::BITSTREAM), m_bs(bs) {};
    virtual ~PipelineTaskOutputBitstream() {};

    std::shared_ptr<RGYBitstream>& bitstream() { return m_bs; }

    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] NVEncFilterSsim *videoQualityMetric) override {
        if (!writer || writer->getOutType() == OUT_TYPE_NONE) {
            return RGY_ERR_NOT_INITIALIZED;
        }
        if (writer->getOutType() != OUT_TYPE_BITSTREAM) {
            return RGY_ERR_INVALID_OPERATION;
        }
        if (videoQualityMetric) {
            if (!videoQualityMetric->decodeStarted()) {
                videoQualityMetric->initDecode(m_bs.get());
            }
            videoQualityMetric->addBitstream(m_bs.get());
        }
        return writer->WriteNextFrame(m_bs.get());
    }
};

template<typename T>
class FrameReleaseData {
    using TaskOutputEvent = std::pair<std::unique_ptr<PipelineTaskOutput>, std::shared_ptr<T>>;
    CUvideoctxlock m_vidCtxLock;
    std::deque<TaskOutputEvent> m_prevInputFrame; //前回投入されたフレーム、完了通知を待ってから解放するため、参照を保持する
    std::mutex m_mtx;
    unique_event m_heFrameAdded;
    unique_event m_heQueueEmpty;
    std::atomic<int> m_queueSize;
    std::thread m_thread;
    bool m_abort;
public:

    FrameReleaseData(CUvideoctxlock vidCtxLock) : m_vidCtxLock(vidCtxLock), m_prevInputFrame(), m_mtx(),
        m_heFrameAdded(std::move(CreateEventUnique(nullptr, FALSE, FALSE))),
        m_heQueueEmpty(std::move(CreateEventUnique(nullptr, FALSE, FALSE))),
        m_queueSize(0), m_thread(), m_abort(false) {}

    ~FrameReleaseData() {
        finish();
    }
    void finish() {
        m_abort = true;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
    void waitUntilEmpty() {
        while (m_queueSize > 0) {
            WaitForSingleObject(m_heQueueEmpty.get(), 10);
        }
    }

    void start() {
        m_thread = std::thread([&]() {
            while (!m_abort) {
                int queueSize = -1;
                TaskOutputEvent prevframe;
                { // m_mtx のロックを取得
                    std::lock_guard<std::mutex> lock(m_mtx);
                    if ((queueSize = (int)m_prevInputFrame.size()) > 0) {
                        prevframe = std::move(m_prevInputFrame.front());
                        m_prevInputFrame.pop_front();
                    }
                }
                if (prevframe.first) {
                    prevframe.first->depend_clear();
                    if (auto surfVppInCuvid = dynamic_cast<PipelineTaskOutputSurf *>(prevframe.first.get())->surf().cuvid(); surfVppInCuvid != nullptr) {
                        // cuvidでは、cuvidのmap/unmapが同時に多重にできないので、まず前のフレームを解放(unmap)する
                        NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                        surfVppInCuvid->unmapFrame();
                    }
                    if ((m_queueSize = queueSize) == 0) {
                        SetEvent(m_heQueueEmpty.get());
                    }
                } else {
                    if ((m_queueSize = queueSize) == 0) {
                        SetEvent(m_heQueueEmpty.get());
                    }
                    WaitForSingleObject(m_heFrameAdded.get(), 100);
                }
            }
        });
    }
    void addFrame(std::unique_ptr<PipelineTaskOutput>& frame, std::shared_ptr<T> event) {
        dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->addCUEvent(event);
        m_queueSize++;
        std::lock_guard<std::mutex> lock(m_mtx);
        m_prevInputFrame.push_back(std::make_pair(std::move(frame), event));
        SetEvent(m_heFrameAdded.get());
    }
};

enum class PipelineTaskType {
    UNKNOWN,
    NVDEC,
    NVENC,
    INPUT,
    INPUTCU,
    CHECKPTS,
    TRIM,
    AUDIO,
    OUTPUTRAW,
    CUDA,
    VIDEOMETRIC,
};

static const TCHAR *getPipelineTaskTypeName(PipelineTaskType type) {
    switch (type) {
    case PipelineTaskType::NVDEC:       return _T("NVDEC");
    case PipelineTaskType::NVENC:       return _T("NVENC");
    case PipelineTaskType::INPUT:       return _T("INPUT");
    case PipelineTaskType::INPUTCU:     return _T("INPUTCU");
    case PipelineTaskType::CHECKPTS:    return _T("CHECKPTS");
    case PipelineTaskType::TRIM:        return _T("TRIM");
    case PipelineTaskType::CUDA:        return _T("CUDA");
    case PipelineTaskType::AUDIO:       return _T("AUDIO");
    case PipelineTaskType::VIDEOMETRIC: return _T("VIDEOMETRIC");
    case PipelineTaskType::OUTPUTRAW:   return _T("OUTRAW");
    default: return _T("UNKNOWN");
    }
}

// Alllocするときの優先度 値が高い方が優先
static const int getPipelineTaskAllocPriority(PipelineTaskType type) {
    switch (type) {
    case PipelineTaskType::NVENC:    return 3;
    case PipelineTaskType::NVDEC:    return 2;
    case PipelineTaskType::INPUT:
    case PipelineTaskType::INPUTCU:
    case PipelineTaskType::CHECKPTS:
    case PipelineTaskType::TRIM:
    case PipelineTaskType::CUDA:
    case PipelineTaskType::AUDIO:
    case PipelineTaskType::OUTPUTRAW:
    case PipelineTaskType::VIDEOMETRIC:
    default: return 0;
    }
}

class PipelineTask {
protected:
    PipelineTaskType m_type;
    NVGPUInfo *m_dev;
    std::deque<std::unique_ptr<PipelineTaskOutput>> m_outQeueue;
    PipelineTaskSurfaces m_workSurfs;
    int m_inFrames;
    int m_outFrames;
    int m_outMaxQueueSize;
    std::unique_ptr<std::mutex> m_outQeueueMtx;
    std::shared_ptr<RGYLog> m_log;
public:
    PipelineTask() : m_type(PipelineTaskType::UNKNOWN), m_dev(nullptr), m_outQeueue(), m_workSurfs(), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(0), m_log() {};
    PipelineTask(PipelineTaskType type, NVGPUInfo *dev, int outMaxQueueSize, bool useOutQueueMtx, std::shared_ptr<RGYLog> log) :
        m_type(type), m_dev(dev), m_outQeueue(), m_workSurfs(), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(outMaxQueueSize), m_log(log), m_outQeueueMtx(useOutQueueMtx ? std::make_unique<std::mutex>() : nullptr) {
    };
    virtual ~PipelineTask() {
        m_workSurfs.clear();
    }
    virtual bool isPassThrough() const { return false; }
    virtual tstring print() const { return getPipelineTaskTypeName(m_type); }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() = 0;
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() = 0;
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) = 0;
    virtual RGY_ERR getOutputFrameInfo(RGYFrameInfo& info) { info = RGYFrameInfo(); return RGY_ERR_NONE; }
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) {
        std::vector<std::unique_ptr<PipelineTaskOutput>> output;
        for (;;) {
            std::unique_ptr<PipelineTaskOutput> out;
            { // m_outQeueueにアクセスする場合、必要なら m_outQeueueMtx のロックを取得
                std::optional<std::lock_guard<std::mutex>> lock;
                if (m_outQeueueMtx) {
                    lock.emplace(*m_outQeueueMtx);
                }
                if (m_outQeueue.size() <= m_outMaxQueueSize) {
                    break;
                }
                out = std::move(m_outQeueue.front());
                m_outQeueue.pop_front();
            }
            if (sync) {
                out->waitsync();
            }
            out->depend_clear();
            m_outFrames++;
            output.push_back(std::move(out));
        }
        return output;
    }
    bool isNVTask() const { return isNVTask(m_type); }
    bool isNVTask(const PipelineTaskType task) const {
        return task == PipelineTaskType::NVENC
            || task == PipelineTaskType::NVDEC;
    }
    // mfx関連とそうでないtaskのやり取りでロックが必要
    bool requireSync(const PipelineTaskType nextTaskType) const {
        return (ENCODER_NVENC) ? nextTaskType == PipelineTaskType::NVENC : isNVTask(m_type) != isNVTask(nextTaskType);
    }
    int workSurfacesAllocPriority() const {
        return getPipelineTaskAllocPriority(m_type);
    }
    size_t workSurfacesCount() const {
        return m_workSurfs.bufCount();
    }

    void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log.get() == nullptr) {
            if (log_level <= RGY_LOG_INFO) {
                return;
            }
        } else if (log_level < m_log->getLogLevel(RGY_LOGT_CORE)) {
            return;
        }

        va_list args;
        va_start(args, format);

        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        vector<TCHAR> buffer(len, 0);
        _vstprintf_s(buffer.data(), len, format, args);
        va_end(args);

        tstring mes = getPipelineTaskTypeName(m_type) + tstring(_T(": ")) + buffer.data();

        if (m_log.get() != nullptr) {
            m_log->write(log_level, RGY_LOGT_CORE, mes.c_str());
        } else {
            _ftprintf(stderr, _T("%s"), mes.c_str());
        }
    }
protected:
    RGY_ERR workSurfacesClear() {
        if (m_outQeueue.size() != 0) {
            return RGY_ERR_UNSUPPORTED;
        }
        if (!m_workSurfs.isAllFree()) {
            return RGY_ERR_UNSUPPORTED;
        }
        return RGY_ERR_NONE;
    }
public:
    virtual RGY_ERR workSurfacesAllocCUBuf(const int numFrames, const RGYFrameInfo &frame) {
        auto sts = workSurfacesClear();
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to clear old surfaces: %s.\n"), get_err_mes(sts));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("allocWorkSurfaces:   cleared old surfaces: %s.\n"), get_err_mes(sts));

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        CUDA_DEBUG_SYNC_ERR;

        // フレームの確保
        std::vector<std::unique_ptr<CUFrameBuf>> frames;
        for (int i = 0; i < numFrames; i++) {
            auto uptr = std::make_unique<CUFrameBuf>(frame);
            auto ret = (frame.mem_type == RGY_MEM_TYPE_CPU) ? uptr->allocHost() : uptr->alloc();
            if (ret != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("failed to alloc host frame: %s.\n"), get_err_mes(ret));
                return ret;
            }
            CUDA_DEBUG_SYNC_ERR;
            frames.push_back(std::move(uptr));
        }
        m_workSurfs.setSurfaces(frames);
        return RGY_ERR_NONE;
    }
    RGY_ERR setWorkSurfaces(std::vector<std::unique_ptr<EncodeBuffer>>& m_stEncodeBuffer, RGYQueueMPMP<CUFrameEnc *>& qEncodeBufferFree, NVEncoder *encoder, const bool rgbAsYUV444) {
        std::vector<std::unique_ptr<CUFrameEnc>> frames;
        for (auto& bfr : m_stEncodeBuffer) {
            frames.push_back(std::unique_ptr<CUFrameEnc>((bfr->stInputBfr.pNV12devPtr)
                ? (CUFrameEnc *)new CUFrameEncDevWrap(bfr.get(), encoder, rgbAsYUV444)
                : (CUFrameEnc *)new CUFrameEncHostWrap(bfr.get(), encoder, rgbAsYUV444)));
        }
        // qEncodeBufferFreeにフレームのポインタのみを登録しておく
        for (auto& f : frames) {
            qEncodeBufferFree.push(f.get());
        }
        // フレームをm_workSurfsに登録する
        // 実体としてはこっちで、qEncodeBufferFreeから取得したポインタから実体をPipelineTaskSurfaces::getで取得する
        m_workSurfs.setSurfaces(frames);
        return RGY_ERR_NONE;
    }
    PipelineTaskSurfaceType workSurfaceType() const {
        if (m_workSurfs.bufCount() == 0) {
            return PipelineTaskSurfaceType::UNKNOWN;
        }
        return m_workSurfs.type();
    }
    // 使用中でないフレームを探してきて、参照カウンタを加算したものを返す
    // 破棄時にアプリ側の参照カウンタを減算するようにshared_ptrで設定してある
    PipelineTaskSurface getWorkSurf() {
        if (m_workSurfs.bufCount() == 0) {
            PrintMes(RGY_LOG_ERROR, _T("getWorkSurf:   No buffer allocated!\n"));
            return PipelineTaskSurface();
        }
        for (int i = 0; i < RGY_WAIT_INTERVAL; i++) {
            PipelineTaskSurface s = m_workSurfs.getFreeSurf();
            if (s != nullptr) {
                return s;
            }
            sleep_hybrid(i);
        }
        PrintMes(RGY_LOG_ERROR, _T("getWorkSurf:   Failed to get work surface, all %d frames used.\n"), m_workSurfs.bufCount());
        return PipelineTaskSurface();
    }

    void setOutputMaxQueueSize(int size) { m_outMaxQueueSize = size; }

    PipelineTaskType taskType() const { return m_type; }
    int inputFrames() const { return m_inFrames; }
    int outputFrames() const { return m_outFrames; }
    int outputMaxQueueSize() const { return m_outMaxQueueSize; }
};

class PipelineTaskInput : public PipelineTask {
    RGYInput *m_input;
    cudaStream_t m_streamUpload;
    RGYListRef<cudaEvent_t> m_frameUseFinEvent;
public:
    PipelineTaskInput(NVGPUInfo *dev, int outMaxQueueSize, RGYInput *input, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::INPUT, dev, outMaxQueueSize, false, log), m_input(input), m_streamUpload(nullptr), m_frameUseFinEvent() {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        auto ret = cudaStreamCreateWithFlags(&m_streamUpload, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create upload stream: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
        }
    };
    virtual ~PipelineTaskInput() {
        if (m_streamUpload) {
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            cudaStreamDestroy(m_streamUpload);
        }
        m_streamUpload = nullptr;
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        m_outQeueue.clear(); // m_frameUseFinEvent解放前に行うこと
        m_frameUseFinEvent.clear([](cudaEvent_t *event) { cudaEventDestroy(*event); });
    };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override {
        const auto inputFrameInfo = m_input->GetInputFrameInfo();
        RGYFrameInfo info(inputFrameInfo.srcWidth, inputFrameInfo.srcHeight, inputFrameInfo.csp, inputFrameInfo.bitdepth, inputFrameInfo.picstruct, RGY_MEM_TYPE_GPU);
        return std::make_pair(info, m_outMaxQueueSize);
    };
    virtual RGY_ERR workSurfacesAllocCUBuf(const int numFrames, const RGYFrameInfo &frame) override {
        auto sts = workSurfacesClear();
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to clear old surfaces: %s.\n"), get_err_mes(sts));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("allocWorkSurfaces:   cleared old surfaces: %s.\n"), get_err_mes(sts));

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        CUDA_DEBUG_SYNC_ERR;

        // フレームの確保
        std::vector<std::unique_ptr<CUFrameBuf>> frames;
        for (int i = 0; i < numFrames; i++) {
            auto uptr = std::make_unique<CUFrameBuf>(frame);
            // 先にhost側のフレームを確保してしまうこと
            // そうしないと cudaGetLastError()等で謎のエラーが発生する
            auto ret = uptr->allocRefHost(true);
            if (ret != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("failed to alloc host frame: %s.\n"), get_err_mes(ret));
                return ret;
            }
            CUDA_DEBUG_SYNC_ERR;
            frames.push_back(std::move(uptr));
        }

        for (auto& f : frames) {
            // 後でデバイス側のフレームを確保する
            auto ret = f->alloc(true);
            if (ret != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("failed to alloc frame: %s.\n"), get_err_mes(ret));
                return ret;
            }
            CUDA_DEBUG_SYNC_ERR;
        }
        m_workSurfs.setSurfaces(frames);
        return RGY_ERR_NONE;
    }
    RGY_ERR LoadNextFrame() {
        auto surfWork = getWorkSurf();
        if (surfWork == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get work surface for input.\n"));
            return RGY_ERR_NOT_ENOUGH_BUFFER;
        }
        auto cuframe = surfWork.cubuf();
        auto hostFrame = cuframe->getRefHostFrame(); // CPUが書き込むための領域を取得
        hostFrame = cuframe->getRefHostFrame();
        if (!hostFrame) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get host frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        auto err = m_input->LoadNextFrame(hostFrame);
        if (err != RGY_ERR_NONE) {
            //Unlockする必要があるので、ここに入ってもすぐにreturnしてはいけない
            if (err == RGY_ERR_MORE_DATA) { // EOF
                err = RGY_ERR_MORE_BITSTREAM; // EOF を PipelineTaskMFXDecode のreturnコードに合わせる
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(err));
            }
            return err;
        }
        hostFrame->setInputFrameId(m_inFrames++);

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        err = cuframe->copyFrameFromHostRef(m_streamUpload);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to copy frame from host: %s.\n"), get_err_mes(err));
            return err;
        }
        if (err == RGY_ERR_NONE) {
            auto cudaEvent = m_frameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
            if (!cudaEvent) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get cuda event.\n"));
                return RGY_ERR_UNKNOWN;
            }
            //eventを入力フレームを使用し終わったことの合図として登録する
            auto cuerr = cudaEventRecord(*cudaEvent, m_streamUpload);
            if (cuerr != cudaSuccess) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to record event for video metric calcualtion: %s.\n"), char_to_tstring(cudaGetErrorString(cuerr)).c_str());
                return err_to_rgy(cuerr);
            }
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), surfWork, cudaEvent));
        }
        return err;
    }
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        return LoadNextFrame();
    }
};

class PipelineTaskNVDecode : public PipelineTask {
protected:
    struct FrameFlags {
        int64_t timestamp;
        RGY_FRAME_FLAGS flags;

        FrameFlags() : timestamp(AV_NOPTS_VALUE), flags(RGY_FRAME_FLAG_NONE) {};
        FrameFlags(int64_t pts, RGY_FRAME_FLAGS f) : timestamp(pts), flags(f) {};
    };
    RGYInput *m_input;
    CuvidDecode *m_dec;
    RGYQueueMPMP<RGYFrameDataMetadata*> m_queueHDR10plusMetadata;
    RGYQueueMPMP<FrameFlags> m_dataFlag;
    RGYRunState m_state;
    int m_decOutFrames;
    int64_t m_hwDecFirstPts;
#if THREAD_DEC_USE_FUTURE
    std::future m_thDecoder;
#else
    std::thread m_thDecoder;
#endif //#if THREAD_DEC_USE_FUTURE
public:
    PipelineTaskNVDecode(NVGPUInfo *dev, CuvidDecode *dec, int outMaxQueueSize, RGYInput *input, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::NVDEC, dev, outMaxQueueSize, false, log), m_input(input), m_dec(dec),
        m_queueHDR10plusMetadata(), m_dataFlag(),
        m_state(RGY_STATE_STOPPED), m_decOutFrames(0), m_hwDecFirstPts(AV_NOPTS_VALUE), m_thDecoder() {
        m_queueHDR10plusMetadata.init(256);
        m_dataFlag.init();
    };
    virtual ~PipelineTaskNVDecode() {
        m_state = RGY_STATE_ABORT;
        closeThread();
        m_queueHDR10plusMetadata.close([](RGYFrameDataMetadata **ptr) { if (*ptr) { delete *ptr; *ptr = nullptr; }; });
    };
    void setDec(CuvidDecode *dec) { m_dec = dec; };

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override {
        const auto inputFrameInfo = m_input->GetInputFrameInfo();
        RGYFrameInfo info(inputFrameInfo.srcWidth, inputFrameInfo.srcHeight, inputFrameInfo.csp, inputFrameInfo.bitdepth, inputFrameInfo.picstruct, RGY_MEM_TYPE_GPU);
        return std::make_pair(info, 0);
    };

    void closeThread() {
        PrintMes(RGY_LOG_DEBUG, _T("Flushing Decoder\n"));
#if THREAD_DEC_USE_FUTURE
        if (m_thDecoder.valid()) {
            while (m_thDecoder.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
#else
        if (m_thDecoder.joinable()) {
            //エンコード中断時の処理
            //ここでフレームをすべて吐き出し切らないと、中断時にデコードスレッドが終了しない
            while (!m_dec->GetError()
                && !(m_dec->frameQueue()->isEndOfDecode() && m_dec->frameQueue()->isEmpty())) {
                m_dec->frameQueue()->endDecode(); //デコーダの待機ループから強制的に出る
                CUVIDPARSERDISPINFO pInfo;
                if (m_dec->frameQueue()->dequeue(&pInfo)) {
                    m_dec->frameQueue()->releaseFrame(&pInfo);
                }
            }
            while (RGYThreadStillActive(m_thDecoder.native_handle())) {
#endif
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
#if !THREAD_DEC_USE_FUTURE
            // linuxでは、これがRGYThreadStillActiveのwhile文を抜けるときに行われるため、
            // これを呼ぶとエラーになってしまう
            m_thDecoder.join();
#endif
        }
    }
    RGY_ERR startThread() {
        m_state = RGY_STATE_RUNNING;
        m_thDecoder = std::thread([this]() {
            CUresult curesult = CUDA_SUCCESS;
            RGYBitstream bitstream = RGYBitstreamInit();
            RGY_ERR sts = RGY_ERR_NONE;
            for (int i = 0; sts == RGY_ERR_NONE && m_state == RGY_STATE_RUNNING && !m_dec->GetError(); i++) {
                if ((  (sts = m_input->LoadNextFrame(nullptr)) != RGY_ERR_NONE //進捗表示のため
                    || (sts = m_input->GetNextBitstream(&bitstream)) != RGY_ERR_NONE)) {
                    if (sts != RGY_ERR_MORE_DATA && sts != RGY_ERR_MORE_BITSTREAM) {
                        m_state = RGY_STATE_ERROR;
                    }
                    break; // エラーないしEOFなら終了
                }

                for (auto& frameData : bitstream.getFrameDataList()) {
                    if (frameData->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameData);
                        if (ptr) {
                            m_queueHDR10plusMetadata.push(new RGYFrameDataHDR10plus(*ptr));
                        }
                    } else if (frameData->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameData);
                        if (ptr) {
                            m_queueHDR10plusMetadata.push(new RGYFrameDataDOVIRpu(*ptr));
                        }
                    }
                }
                const auto flags = FrameFlags(bitstream.pts(), (RGY_FRAME_FLAGS)bitstream.dataflag());
                m_dataFlag.push(flags);
                if (m_hwDecFirstPts == AV_NOPTS_VALUE) {
                    m_hwDecFirstPts = bitstream.pts();
                }
                PrintMes(RGY_LOG_TRACE, _T("Set packet #%d, size %zu, pts %lld (%s)\n"), i, bitstream.size(),
                    (long long int)bitstream.pts(), getTimestampString(bitstream.pts(), av_make_q(m_input->getInputTimebase())).c_str());
                if (CUDA_SUCCESS != (curesult = m_dec->DecodePacket(bitstream.bufptr() + bitstream.offset(), bitstream.size(), bitstream.pts(), av_make_q(m_input->getInputTimebase())))) {
                    PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacket: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                    m_state = RGY_STATE_ERROR;
                    return err_to_rgy(curesult);
                }
                bitstream.setSize(0);
                bitstream.setOffset(0);
                bitstream.clearFrameDataList();
            }
            // flush
            PrintMes(RGY_LOG_DEBUG, _T("Decode thread: flush.\n"));
            if (CUDA_SUCCESS != (curesult = m_dec->DecodePacket(nullptr, 0, AV_NOPTS_VALUE, av_make_q(m_input->getInputTimebase())))) {
                PrintMes(RGY_LOG_ERROR, _T("Error in DecodePacketFin: %d (%s).\n"), curesult, char_to_tstring(_cudaGetErrorEnum(curesult)).c_str());
                m_state = RGY_STATE_ERROR;
                sts = err_to_rgy(curesult);
            }
            PrintMes(RGY_LOG_DEBUG, _T("Decode thread: finished: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MORE_BITSTREAM;
        });
        return RGY_ERR_NONE;
    }

    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        return getOutput();
    }
    PipelineTaskSurface addTaskSurface(std::unique_ptr<CUFrameCuvid>& surf) {
        return m_workSurfs.addSurface(surf);
    }
    
protected:
    RGY_ERR getOutput() {
        auto ret = RGY_ERR_NONE;
        if (m_state == RGY_STATE_STOPPED) {
            m_state = RGY_STATE_RUNNING;
            if ((ret = startThread()) != RGY_ERR_NONE) {
                m_state = RGY_STATE_ERROR;
                PrintMes(RGY_LOG_ERROR, _T("Failed to start Decode thread: %s.\n"), get_err_mes(ret));
                return ret;
            }
        }
        if (m_state != RGY_STATE_RUNNING) {
            return (m_state == RGY_STATE_EOF) ? RGY_ERR_MORE_BITSTREAM : RGY_ERR_UNKNOWN;
        }

        CUVIDPARSERDISPINFO dispInfo = { 0 };
        while (m_state == RGY_STATE_RUNNING) {
            if (m_dec->GetError()) {
                m_state = RGY_STATE_ERROR;
                return RGY_ERR_UNKNOWN;
            }
            if (m_dec->frameQueue()->isEndOfDecode() && m_dec->frameQueue()->isEmpty()) {
                m_state = RGY_STATE_EOF;
                return RGY_ERR_MORE_BITSTREAM;
            }

            dispInfo = CUVIDPARSERDISPINFO{ 0 };
            if (!m_dec->frameQueue()->dequeue(&dispInfo)) {
                m_dec->frameQueue()->waitForQueueUpdate();
                continue;
            }
            // OpenGOP等でキーフレームより前のフレームが出てくることがあるのを削除
            if (dispInfo.timestamp < m_hwDecFirstPts) {
                m_dec->frameQueue()->releaseFrame(&dispInfo);
                continue;
            }
            break;
        }

        //cuvidのtimestampはかならず分子が1になっているのでもとに戻す
        auto cuvidTimebase = rgy_rational<int>(1, m_input->getInputTimebase().d());
        dispInfo.timestamp = rational_rescale(dispInfo.timestamp, cuvidTimebase, m_input->getInputTimebase());
        PrintMes(RGY_LOG_TRACE, _T("input frame (dev) #%d, pic_idx %d, timestamp %lld\n"), m_decOutFrames, dispInfo.picture_index, dispInfo.timestamp);

        auto inputPicstruct = (dispInfo.progressive_frame) ? RGY_PICSTRUCT_FRAME : ((dispInfo.top_field_first) ? RGY_PICSTRUCT_FRAME_TFF : RGY_PICSTRUCT_FRAME_BFF);
        auto flags = RGY_FRAME_FLAG_NONE;
        if (dispInfo.repeat_first_field == 1) {
            flags |= (dispInfo.top_field_first) ? RGY_FRAME_FLAG_RFF_TFF : RGY_FRAME_FLAG_RFF_BFF;
        }

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        auto surfDecOut = std::make_unique<CUFrameCuvid>(m_dec->GetDecoder(), m_dec->GetDecFrameInfo(),
            std::shared_ptr<CUVIDPARSERDISPINFO>(new CUVIDPARSERDISPINFO(dispInfo), [&](CUVIDPARSERDISPINFO *ptr) {
            // CUFrameCuvidのデストラクト時にreleaseFrameが呼ばれるようにしておく
            PrintMes(RGY_LOG_TRACE, _T("Free input frame pic_idx %d, timestamp %lld\n"), ptr->picture_index, ptr->timestamp);
            m_dec->frameQueue()->releaseFrame(ptr);
            delete ptr;
        }));
        surfDecOut->setFlags(flags);
        surfDecOut->setPicstruct(inputPicstruct);
        surfDecOut->setTimestamp(dispInfo.timestamp);
        surfDecOut->setInputFrameId(m_decOutFrames++);
        surfDecOut->setFlags(getDataFlag(surfDecOut->timestamp()));

        surfDecOut->clearDataList();
        if (auto data = getMetadata(RGY_FRAME_DATA_HDR10PLUS, surfDecOut->timestamp()); data) {
            surfDecOut->dataList().push_back(data);
        }
        if (auto data = getMetadata(RGY_FRAME_DATA_DOVIRPU, surfDecOut->timestamp()); data) {
            surfDecOut->dataList().push_back(data);
        }
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), m_workSurfs.addSurface(surfDecOut)));
        return ret;
    }
    RGY_FRAME_FLAGS getDataFlag(const int64_t timestamp) {
        FrameFlags pts_flag;
        while (m_dataFlag.front_copy_no_lock(&pts_flag)) {
            if (pts_flag.timestamp < timestamp || pts_flag.timestamp == AV_NOPTS_VALUE) {
                m_dataFlag.pop();
            } else {
                break;
            }
        }
        size_t queueSize = m_dataFlag.size();
        for (uint32_t i = 0; i < queueSize; i++) {
            if (m_dataFlag.copy(&pts_flag, i, &queueSize)) {
                if (pts_flag.timestamp == timestamp) {
                    return pts_flag.flags;
                }
            }
        }
        return RGY_FRAME_FLAG_NONE;
    }
    std::shared_ptr<RGYFrameData> getMetadata(const RGYFrameDataType datatype, const int64_t timestamp) {
        std::shared_ptr<RGYFrameData> frameData;
        RGYFrameDataMetadata *frameDataPtr = nullptr;
        while (m_queueHDR10plusMetadata.front_copy_no_lock(&frameDataPtr)) {
            if (frameDataPtr->timestamp() < timestamp) {
                m_queueHDR10plusMetadata.pop();
                delete frameDataPtr;
            } else {
                break;
            }
        }
        size_t queueSize = m_queueHDR10plusMetadata.size();
        for (uint32_t i = 0; i < queueSize; i++) {
            if (m_queueHDR10plusMetadata.copy(&frameDataPtr, i, &queueSize)) {
                if (frameDataPtr->timestamp() == timestamp && frameDataPtr->dataType() == datatype) {
                    if (frameDataPtr->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameDataPtr);
                        if (ptr) {
                            frameData = std::make_shared<RGYFrameDataHDR10plus>(*ptr);
                        }
                    } else if (frameDataPtr->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameDataPtr);
                        if (ptr) {
                            frameData = std::make_shared<RGYFrameDataDOVIRpu>(*ptr);
                        }
                    }
                    break;
                }
            }
        }
        return frameData;
    };
};

class PipelineTaskCheckPTS : public PipelineTask {
protected:
    rgy_rational<int> m_srcTimebase;
    rgy_rational<int> m_streamTimebase;
    rgy_rational<int> m_outputTimebase;
    RGYAVSync m_avsync;
    bool m_timestampPassThrough;
    bool m_vpp_rff;
    bool m_vpp_afs_rff_aware;
    bool m_interlaceAuto;
    cudaVideoDeinterlaceMode m_deinterlaceMode;
    int64_t m_outFrameDuration; //(m_outputTimebase基準)
    int64_t m_tsOutFirst;     //(m_outputTimebase基準)
    int64_t m_tsOutEstimated; //(m_outputTimebase基準)
    int64_t m_tsPrev;         //(m_outputTimebase基準)
    uint32_t m_inputFramePosIdx;
    FramePosList *m_framePosList;
    CuvidDecode *m_dec;
    PipelineTaskNVDecode *m_taskNVDec;
public:
    PipelineTaskCheckPTS(NVGPUInfo *dev, CuvidDecode *dec, PipelineTaskNVDecode *taskNVDec, rgy_rational<int> srcTimebase, rgy_rational<int> streamTimebase, rgy_rational<int> outputTimebase, int64_t outFrameDuration, RGYAVSync avsync, cudaVideoDeinterlaceMode deinterlaceMode,
        bool timestampPassThrough, bool vpp_rff, bool vpp_afs_rff_aware, bool interlaceAuto, FramePosList *framePosList, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CHECKPTS, dev, /*outMaxQueueSize = */ 0 /*常に0である必要がある*/, false, log),
        m_srcTimebase(srcTimebase), m_streamTimebase(streamTimebase), m_outputTimebase(outputTimebase), m_avsync(avsync),
        m_timestampPassThrough(timestampPassThrough), m_vpp_rff(vpp_rff), m_vpp_afs_rff_aware(vpp_afs_rff_aware), m_interlaceAuto(interlaceAuto), m_deinterlaceMode(deinterlaceMode),
        m_outFrameDuration(outFrameDuration),
        m_tsOutFirst(-1), m_tsOutEstimated(0), m_tsPrev(-1), m_inputFramePosIdx(std::numeric_limits<decltype(m_inputFramePosIdx)>::max()), m_framePosList(framePosList), m_dec(dec), m_taskNVDec(taskNVDec) {
    };
    virtual ~PipelineTaskCheckPTS() {};

    virtual bool isPassThrough() const override {
        // そのまま渡すのでpaththrough
        return true;
    }
    static const int MAX_FORCECFR_INSERT_FRAMES = 1024; //事実上の無制限
public:
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            //PipelineTaskCheckPTSは、getOutputで1フレームずつしか取り出さない
            //そのためm_outQeueueにまだフレームが残っている可能性がある
            return (m_outQeueue.size() > 0) ? RGY_ERR_MORE_SURFACE : RGY_ERR_MORE_DATA;
        }
        int64_t outPtsSource = m_tsOutEstimated; //(m_outputTimebase基準)
        int64_t outDuration = m_outFrameDuration; //入力fpsに従ったduration

        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (taskSurf == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type: failed to cast to PipelineTaskOutputSurf.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        if ((m_srcTimebase.n() > 0 && m_srcTimebase.is_valid())
            && ((m_avsync & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) || m_vpp_rff || m_vpp_afs_rff_aware || m_timestampPassThrough)) {
            const auto srcTimestamp = taskSurf->surf().frame()->timestamp();
            if (srcTimestamp < 0) {
                // timestampを修正
                outPtsSource = m_tsOutEstimated;
                taskSurf->surf().frame()->setTimestamp(rational_rescale(m_tsOutEstimated, m_outputTimebase, m_srcTimebase));
                taskSurf->surf().frame()->setDuration(rational_rescale(m_outFrameDuration, m_outputTimebase, m_srcTimebase));
                PrintMes(RGY_LOG_WARN, _T("check_pts: Invalid timestamp from input frame #%d: timestamp %lld, timebase %d/%d, duration %lld.\n"),
                         taskSurf->surf().frame()->inputFrameId(), taskSurf->surf().frame()->timestamp(), m_srcTimebase.n(), m_srcTimebase.d(), taskSurf->surf().frame()->duration());
                PrintMes(RGY_LOG_WARN, _T("           use estimated timestamp: timestamp %lld, timebase %d/%d, duration %lld.\n"),
                    outPtsSource, m_outputTimebase.n(), m_outputTimebase.d(), m_outFrameDuration);
            } else {
                //CFR仮定ではなく、オリジナルの時間を見る
                if (srcTimestamp == AV_NOPTS_VALUE) {
                    outPtsSource = m_tsPrev + m_outFrameDuration + m_tsOutFirst/*あとでm_tsOutFirstが引かれるので*/;
                } else {
                    outPtsSource = rational_rescale(srcTimestamp, m_srcTimebase, m_outputTimebase);
                }
                if (taskSurf->surf().frame()->duration() > 0) {
                    outDuration = rational_rescale(taskSurf->surf().frame()->duration(), m_srcTimebase, m_outputTimebase);
                    taskSurf->surf().frame()->setDuration(outDuration);
                }
            }
        }
        PrintMes(RGY_LOG_TRACE, _T("check_pts(%d/%d): nOutEstimatedPts %lld, outPtsSource %lld, outDuration %d\n"), taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsOutEstimated, outPtsSource, outDuration);
        if (m_tsOutFirst < 0) {
            m_tsOutFirst = outPtsSource; //最初のpts
            PrintMes(RGY_LOG_TRACE, _T("check_pts: m_tsOutFirst %lld\n"), outPtsSource);
        }
        //最初のptsを0に修正
        if (!m_timestampPassThrough) {
            //最初のptsを0に修正
            outPtsSource -= m_tsOutFirst;
        }

        if ((m_avsync & RGY_AVSYNC_VFR) || m_vpp_rff || m_vpp_afs_rff_aware) {
            if (m_vpp_rff || m_vpp_afs_rff_aware) {
                if (std::abs(outPtsSource - m_tsOutEstimated) >= 32 * m_outFrameDuration) {
                    PrintMes(RGY_LOG_TRACE, _T("check_pts: detected gap %lld, changing offset.\n"), outPtsSource, std::abs(outPtsSource - m_tsOutEstimated));
                    //timestampに一定以上の差があればそれを無視する
                    m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                    outPtsSource = m_tsOutEstimated;
                    PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
                }
                auto ptsDiff = outPtsSource - m_tsOutEstimated;
                if (ptsDiff <= std::min<int64_t>(-1, -1 * m_outFrameDuration * 7 / 8)) {
                    //間引きが必要
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   skipping frame (vfr)\n"), taskSurf->surf().frame()->inputFrameId());
                    return RGY_ERR_MORE_SURFACE;
                }
                // 少しのずれはrffによるものとみなし、基準値を修正する
                m_tsOutEstimated = outPtsSource;
            }
            if ((ENCODER_VCEENC || ENCODER_NVENC) && m_framePosList) {
                //cuvidデコード時は、timebaseの分子はかならず1なので、streamIn->time_baseとズレているかもしれないのでオリジナルを計算
                const auto orig_pts = rational_rescale(taskSurf->surf().frame()->timestamp(), m_srcTimebase, m_streamTimebase);
                //ptsからフレーム情報を取得する
                const auto framePos = m_framePosList->findpts(orig_pts, &m_inputFramePosIdx);
                PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   estimetaed orig_pts %lld, framePos %d\n"), taskSurf->surf().frame()->inputFrameId(), orig_pts, framePos.poc);
                if (framePos.poc != FRAMEPOS_POC_INVALID && framePos.duration > 0) {
                    //有効な値ならオリジナルのdurationを使用する
                    outDuration = rational_rescale(framePos.duration, m_streamTimebase, m_outputTimebase);
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   changing duration to original: %d\n"), taskSurf->surf().frame()->inputFrameId(), outDuration);
                }
            }
        }
        if (m_avsync & RGY_AVSYNC_FORCE_CFR) {
            if (std::abs(outPtsSource - m_tsOutEstimated) >= CHECK_PTS_MAX_INSERT_FRAMES * m_outFrameDuration) {
                //timestampに一定以上の差があればそれを無視する
                m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                outPtsSource = m_tsOutEstimated;
                PrintMes(RGY_LOG_WARN, _T("Big Gap was found between 2 frames, avsync might be corrupted.\n"));
                PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
            }
            auto ptsDiff = outPtsSource - m_tsOutEstimated;
            if (ptsDiff <= std::min<int64_t>(-1, -1 * m_outFrameDuration * 7 / 8)) {
                //間引きが必要
                PrintMes(RGY_LOG_DEBUG, _T("Drop frame: framepts %lld, estimated next %lld, diff %lld [%.1f]\n"), outPtsSource, m_tsOutEstimated, ptsDiff, ptsDiff / (double)m_outFrameDuration);
                return RGY_ERR_MORE_SURFACE;
            }
            while (ptsDiff >= std::max<int64_t>(1, m_outFrameDuration * 7 / 8)) {
                PrintMes(RGY_LOG_DEBUG, _T("Insert frame: framepts %lld, estimated next %lld, diff %lld [%.1f]\n"), outPtsSource, m_tsOutEstimated, ptsDiff, ptsDiff / (double)m_outFrameDuration);
                add_dec_vpp_param(taskSurf, m_tsOutEstimated, m_outFrameDuration, true /* timestamp等を別として登録するため、新しいフレームを作成する*/);
                m_tsOutEstimated += m_outFrameDuration;
                ptsDiff = outPtsSource - m_tsOutEstimated;
            }
            outPtsSource = m_tsOutEstimated;
        }
        if (m_tsPrev >= outPtsSource) {
            if (m_tsPrev - outPtsSource >= MAX_FORCECFR_INSERT_FRAMES * m_outFrameDuration) {
                PrintMes(RGY_LOG_DEBUG, _T("check_pts: previous pts %lld, current pts %lld, estimated pts %lld, m_tsOutFirst %lld, changing offset.\n"), m_tsPrev, outPtsSource, m_tsOutEstimated, m_tsOutFirst);
                m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                outPtsSource = m_tsOutEstimated;
                PrintMes(RGY_LOG_DEBUG, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
            } else {
                if (m_avsync & RGY_AVSYNC_FORCE_CFR) {
                    //間引きが必要
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d/%d): timestamp of video frame is smaller than previous frame, skipping frame: previous pts %lld, current pts %lld.\n"),
                        taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsPrev, outPtsSource);
                    return RGY_ERR_MORE_SURFACE;
                } else {
                    const auto origPts = outPtsSource;
                    outPtsSource = m_tsPrev + std::max<int64_t>(1, m_outFrameDuration / 4);
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d/%d): timestamp of video frame is smaller than previous frame, changing pts: %lld -> %lld (previous pts %lld).\n"),
                        taskSurf->surf().frame()->inputFrameId(), m_inFrames, origPts, outPtsSource, m_tsPrev);
                }
            }
        }

        //次のフレームのptsの予想
        m_inFrames++;
        m_tsOutEstimated += outDuration;
        m_tsPrev = outPtsSource;
        add_dec_vpp_param(taskSurf, outPtsSource, outDuration, false);
        return RGY_ERR_NONE;
    }

    RGY_ERR add_dec_vpp_param(PipelineTaskOutputSurf *taskSurf, const int64_t outPts, const int64_t outDuration, const bool createCopy) {
        if (auto surf = taskSurf->surf().cuvid(); surf != nullptr) {
            if (!m_taskNVDec) {
                PrintMes(RGY_LOG_ERROR, _T("detected cuvid frame, but null pointer for taskNVDec.\n"));
                return RGY_ERR_NULL_PTR;
            }
            auto frameinfo = surf->getInfo();
            PipelineTaskSurface outSurf = taskSurf->surf();
            CUVIDPROCPARAMS oVPP = { 0 };
            oVPP.top_field_first = surf->dispInfo()->top_field_first;
            switch (m_deinterlaceMode) {
            case cudaVideoDeinterlaceMode_Weave:
                oVPP.progressive_frame = surf->dispInfo()->progressive_frame;
                oVPP.unpaired_field = 0;// oVPP.progressive_frame;
                PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[dev](%d): idx %d, outPtsSource %lld, outDuration %d, progressive %d\n"), outSurf.frame()->inputFrameId(), surf->dispInfo()->picture_index, outSurf.frame()->timestamp(), outSurf.frame()->duration(), oVPP.progressive_frame);
                if (createCopy) { // timestamp等を別として登録するため、新しいフレームを作成する
                    auto surfCopy = std::make_unique<CUFrameCuvid>(m_dec->GetDecoder(), outSurf.cuvid()->getInfo(), outSurf.cuvid()->dispInfo());
                    surfCopy->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    surfCopy->setTimestamp(outPts);
                    surfCopy->setDuration(outDuration);
                    surfCopy->setOVPP(oVPP);
                    // m_taskNVDecのtask surfaceとして登録する (そうしないとデコーダで必要なタイミングでdeleteFreedSurfaceが呼ばれず、フリーズしてしまう場合がある)
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), m_taskNVDec->addTaskSurface(surfCopy)));
                } else {
                    outSurf.cuvid()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    outSurf.cuvid()->setTimestamp(outPts);
                    outSurf.cuvid()->setDuration(outDuration);
                    outSurf.cuvid()->setOVPP(oVPP);
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), outSurf));
                }
                break;
            case cudaVideoDeinterlaceMode_Bob:
                oVPP.progressive_frame = (m_interlaceAuto) ? surf->dispInfo()->progressive_frame : 0;
                oVPP.second_field = 0;
                if (createCopy) { // timestamp等を別として登録するため、新しいフレームを作成する
                    auto surfCopy = std::make_unique<CUFrameCuvid>(m_dec->GetDecoder(), outSurf.cuvid()->getInfo(), outSurf.cuvid()->dispInfo());
                    surfCopy->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    surfCopy->setTimestamp(outPts);
                    surfCopy->setDuration(outDuration);
                    surfCopy->setOVPP(oVPP);
                    //RFFに関するフラグを念のためクリア
                    surfCopy->setFlags(outSurf.frame()->flags() & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)));
                    surfCopy->setPicstruct(RGY_PICSTRUCT_FRAME);
                    surfCopy->setDuration(outDuration >> 1);
                    // m_taskNVDecのtask surfaceとして登録する (そうしないとデコーダで必要なタイミングでdeleteFreedSurfaceが呼ばれず、フリーズしてしまう場合がある)
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), m_taskNVDec->addTaskSurface(surfCopy)));
                } else {
                    outSurf.cuvid()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    outSurf.cuvid()->setTimestamp(outPts);
                    outSurf.cuvid()->setDuration(outDuration);
                    outSurf.cuvid()->setOVPP(oVPP);
                    //RFFに関するフラグを念のためクリア
                    outSurf.cuvid()->setFlags(outSurf.frame()->flags() & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)));
                    outSurf.cuvid()->setPicstruct(RGY_PICSTRUCT_FRAME);
                    outSurf.cuvid()->setDuration(outDuration >> 1);
                    PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[bob](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), outSurf.frame()->inputFrameId(), outSurf.frame()->timestamp(), outSurf.frame()->duration(), oVPP.progressive_frame);
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), outSurf));
                }
                { // timestamp等を別として登録するため、新しいフレームを作成する
                    auto surfCopy = std::make_unique<CUFrameCuvid>(m_dec->GetDecoder(), outSurf.cuvid()->getInfo(), outSurf.cuvid()->dispInfo());
                    surfCopy->setPropertyFrom(outSurf.cuvid());
                    surfCopy->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    surfCopy->setTimestamp(outPts + (outDuration >> 1));
                    surfCopy->setDuration(taskSurf->surf().frame()->duration() - (outDuration >> 1));
                    oVPP.second_field = 1;
                    surfCopy->setOVPP(oVPP);
                    PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[bob](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), surfCopy->inputFrameId(), surfCopy->timestamp(), surfCopy->duration(), oVPP.progressive_frame);
                    // m_taskNVDecのtask surfaceとして登録する (そうしないとデコーダで必要なタイミングでdeleteFreedSurfaceが呼ばれず、フリーズしてしまう場合がある)
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), m_taskNVDec->addTaskSurface(surfCopy)));
                }
                break;
            case cudaVideoDeinterlaceMode_Adaptive:
                if (createCopy) { // timestamp等を別として登録するため、新しいフレームを作成する
                    auto surfCopy = std::make_unique<CUFrameCuvid>(m_dec->GetDecoder(), outSurf.cuvid()->getInfo(), outSurf.cuvid()->dispInfo());
                    surfCopy->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    surfCopy->setTimestamp(outPts);
                    surfCopy->setDuration(outDuration);
                    oVPP.progressive_frame = (m_interlaceAuto) ? surf->dispInfo()->progressive_frame : 0;
                    //RFFに関するフラグを念のためクリア
                    surfCopy->setFlags(outSurf.frame()->flags() & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)));
                    surfCopy->setPicstruct(RGY_PICSTRUCT_FRAME);
                    surfCopy->setOVPP(oVPP);
                    PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[adp](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), outSurf.frame()->inputFrameId(), outSurf.frame()->timestamp(), outSurf.frame()->duration(), oVPP.progressive_frame);
                    // m_taskNVDecのtask surfaceとして登録する (そうしないとデコーダで必要なタイミングでdeleteFreedSurfaceが呼ばれず、フリーズしてしまう場合がある)
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), m_taskNVDec->addTaskSurface(surfCopy)));
                } else {
                    outSurf.cuvid()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                    outSurf.cuvid()->setTimestamp(outPts);
                    outSurf.cuvid()->setDuration(outDuration);
                    oVPP.progressive_frame = (m_interlaceAuto) ? surf->dispInfo()->progressive_frame : 0;
                    //RFFに関するフラグを念のためクリア
                    outSurf.cuvid()->setFlags(outSurf.frame()->flags() & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)));
                    outSurf.cuvid()->setPicstruct(RGY_PICSTRUCT_FRAME);
                    outSurf.cuvid()->setOVPP(oVPP);
                    PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[adp](%d): outPtsSource %lld, outDuration %d, progressive %d\n"), outSurf.frame()->inputFrameId(), outSurf.frame()->timestamp(), outSurf.frame()->duration(), oVPP.progressive_frame);
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), outSurf));
                }
                break;
            default:
                PrintMes(RGY_LOG_ERROR, _T("Unknown Deinterlace mode\n"));
                break;
            }
        } else if (createCopy) {
            PrintMes(RGY_LOG_ERROR, _T("Not implmented yet!\n"));
            return RGY_ERR_UNSUPPORTED;
        } else {
            PipelineTaskSurface outSurf = taskSurf->surf();
            outSurf.frame()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
            outSurf.frame()->setTimestamp(outPts);
            outSurf.frame()->setDuration(outDuration);
            PrintMes(RGY_LOG_TRACE, _T("add_dec_vpp_param[dev](%d): outPtsSource %lld, outDuration %d\n"), taskSurf->surf().frame()->inputFrameId(), outPts, outDuration);
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), outSurf));
        }
        return RGY_ERR_NONE;
    };
#if 0
    //checkptsではtimestampを上書きするため特別に常に1フレームしか取り出さない
    //これは--avsync frocecfrでフレームを参照コピーする際、
    //mfxSurface1自体は同じデータを指すため、複数のタイムスタンプを持つことができないため、
    //1フレームずつgetOutputし、都度タイムスタンプを上書きしてすぐに後続のタスクに投入してタイムスタンプを反映させる必要があるため
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) override {
        std::vector<std::unique_ptr<PipelineTaskOutput>> output;
        if ((int)m_outQeueue.size() > m_outMaxQueueSize) {
            auto out = std::move(m_outQeueue.front());
            m_outQeueue.pop_front();
            if (sync) {
                out->waitsync();
            }
            out->depend_clear();
            if (out->customdata() != nullptr) {
                const auto dataCheckPts = dynamic_cast<const PipelineTaskOutputDataCheckPts *>(out->customdata());
                if (dataCheckPts == nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get timestamp data, timestamp might be inaccurate!\n"));
                } else {
                    PipelineTaskOutputSurf *outSurf = dynamic_cast<PipelineTaskOutputSurf *>(out.get());
                    outSurf->surf().frame()->setTimestamp(dataCheckPts->timestampOverride());
                }
            }
            m_outFrames++;
            output.push_back(std::move(out));
        }
        if (output.size() > 1) {
            PrintMes(RGY_LOG_ERROR, _T("output queue more than 1, invalid!\n"));
        }
        return output;
    }
#endif
};

class PipelineTaskTrim : public PipelineTask {
protected:
    const sTrimParam &m_trimParam;
    RGYInput *m_input;
    rgy_rational<int> m_srcTimebase;
    rgy_rational<int> m_outTimebase;
    int64_t m_trimTimestampOffset;
    int64_t m_lastTrimFramePts;
public:
    PipelineTaskTrim(NVGPUInfo *dev, const sTrimParam &trimParam, RGYInput *input, const rgy_rational<int>& srcTimebase, const rgy_rational<int>& outTimebase, int outMaxQueueSize, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::TRIM, dev, outMaxQueueSize, false, log),
        m_trimParam(trimParam), m_input(input), m_srcTimebase(srcTimebase), m_outTimebase(outTimebase), m_trimTimestampOffset(0), m_lastTrimFramePts(AV_NOPTS_VALUE) {
    };
    virtual ~PipelineTaskTrim() {};

    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    int64_t trimTimestampOffset() const { return m_trimTimestampOffset; }

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }
        m_inFrames++;
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        const auto trimSts = frame_inside_range(taskSurf->surf().frame()->inputFrameId(), m_trimParam.list);
        const auto inputFramePts = rational_rescale(taskSurf->surf().frame()->timestamp(), m_srcTimebase, m_outTimebase);
        if ((trimSts.second > 0) //check_pts内で最初のフレームのptsを0とするようnOutFirstPtsが設定されるので、先頭のtrim blockについてはここでは処理しない
            && (m_lastTrimFramePts != AV_NOPTS_VALUE)) { //前のフレームがtrimで脱落させたフレームなら
            m_trimTimestampOffset += inputFramePts - m_lastTrimFramePts; //trimで脱落させたフレームの分の時間を加算
        }
        if (!trimSts.first) {
            m_lastTrimFramePts = inputFramePts; //脱落させたフレームの時間を記憶
            return RGY_ERR_NONE;
        }
        m_lastTrimFramePts = AV_NOPTS_VALUE;
        if (!m_input->checkTimeSeekTo(taskSurf->surf().frame()->timestamp(), m_srcTimebase)) {
            return RGY_ERR_NONE; //seektoにより脱落させるフレーム
        }
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), taskSurf->surf()));
        return RGY_ERR_NONE;
    }
};

class PipelineTaskAudio : public PipelineTask {
protected:
    RGYInput *m_input;
    std::map<int, std::shared_ptr<RGYOutputAvcodec>> m_pWriterForAudioStreams;
    std::map<int, NVEncFilter *> m_filterForStreams;
    std::vector<std::shared_ptr<RGYInput>> m_audioReaders;
public:
    PipelineTaskAudio(NVGPUInfo *dev, RGYInput *input, std::vector<std::shared_ptr<RGYInput>>& audioReaders, std::vector<std::shared_ptr<RGYOutput>>& fileWriterListAudio, std::vector<VppVilterBlock>& vpFilters, int outMaxQueueSize, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::AUDIO, dev, outMaxQueueSize, false, log),
        m_input(input), m_audioReaders(audioReaders) {
        //streamのindexから必要なwriteへのポインタを返すテーブルを作成
        for (auto writer : fileWriterListAudio) {
            auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
            if (pAVCodecWriter) {
                auto trackIdList = pAVCodecWriter->GetStreamTrackIdList();
                for (auto trackID : trackIdList) {
                    m_pWriterForAudioStreams[trackID] = pAVCodecWriter;
                }
            }
        }
        //streamのtrackIdからパケットを送信するvppフィルタへのポインタを返すテーブルを作成
        for (auto& filterBlock : vpFilters) {
            if (filterBlock.type == VppFilterType::FILTER_OPENCL) {
                for (auto& filter : filterBlock.vppnv) {
                    const auto targetTrackId = filter->targetTrackIdx();
                    if (targetTrackId != 0) {
                        m_filterForStreams[targetTrackId] = filter.get();
                    }
                }
            }
        }
    };
    virtual ~PipelineTaskAudio() {};

    virtual bool isPassThrough() const override { return true; }

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };


    void flushAudio() {
        PrintMes(RGY_LOG_DEBUG, _T("Clear packets in writer...\n"));
        std::set<RGYOutputAvcodec*> writers;
        for (const auto& [ streamid, writer ] : m_pWriterForAudioStreams) {
            auto pWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
            if (pWriter != nullptr) {
                writers.insert(pWriter.get());
            }
        }
        for (const auto& writer : writers) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            writer->WriteNextPacket(nullptr);
        }
    }

    RGY_ERR extractAudio(int inputFrames) {
        RGY_ERR ret = RGY_ERR_NONE;
#if ENABLE_AVSW_READER
        if (m_pWriterForAudioStreams.size() > 0) {
#if ENABLE_SM_READER
            RGYInputSM *pReaderSM = dynamic_cast<RGYInputSM *>(m_input);
            const int droppedInAviutl = (pReaderSM != nullptr) ? pReaderSM->droppedFrames() : 0;
#else
            const int droppedInAviutl = 0;
#endif

            auto packetList = m_input->GetStreamDataPackets(inputFrames + droppedInAviutl);

            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_audioReaders) {
                vector_cat(packetList, reader->GetStreamDataPackets(inputFrames + droppedInAviutl));
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                AVPacket *pkt = packetList[i];
                const int nTrackId = pktFlagGetTrackID(pkt);
                const bool sendToFilter = m_filterForStreams.count(nTrackId) > 0;
                const bool sendToWriter = m_pWriterForAudioStreams.count(nTrackId) > 0;
                if (sendToFilter) {
                    AVPacket *pktToFilter = nullptr;
                    if (sendToWriter) {
                        pktToFilter = av_packet_clone(pkt);
                    } else {
                        std::swap(pktToFilter, pkt);
                    }
                    auto err = m_filterForStreams[nTrackId]->addStreamPacket(pktToFilter);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                }
                if (sendToWriter) {
                    auto pWriter = m_pWriterForAudioStreams[nTrackId];
                    if (pWriter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid writer found for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                        return RGY_ERR_NOT_FOUND;
                    }
                    auto err = pWriter->WriteNextPacket(pkt);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                    pkt = nullptr;
                }
                if (pkt != nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to find writer for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                    return RGY_ERR_NOT_FOUND;
                }
            }
        }
#endif //ENABLE_AVSW_READER
        return ret;
    };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        m_inFrames++;
        auto err = extractAudio(m_inFrames);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (!frame) {
            flushAudio();
            return RGY_ERR_MORE_DATA;
        }
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), taskSurf->surf()));
        return RGY_ERR_NONE;
    }
};

class PipelineTaskVideoQualityMetric : public PipelineTask {
private:
    NVEncFilterSsim *m_videoMetric;
    cudaStream_t m_stream;
    RGYListRef<cudaEvent_t> m_frameUseFinEvent;
public:
    PipelineTaskVideoQualityMetric(NVGPUInfo *dev, NVEncFilterSsim *videoMetric, int outMaxQueueSize, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::VIDEOMETRIC, dev, outMaxQueueSize, false, log), m_videoMetric(videoMetric), m_stream(nullptr), m_frameUseFinEvent() {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        auto ret = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create vqm stream: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
        }
    };
    virtual ~PipelineTaskVideoQualityMetric() {
        if (m_stream) {
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
        }
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        m_frameUseFinEvent.clear([](cudaEvent_t *event) { cudaEventDestroy(*event); });
    };

    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }

        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (taskSurf == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
            return RGY_ERR_NULL_PTR;
        }
        cudaEvent_t filterFinishEvent; // このフィルタの終了を待つためのイベント
        RGYFrameInfo inputFrame;
        if (auto surfVppIn = taskSurf->surf().cubuf(); surfVppIn != nullptr) {
            inputFrame = surfVppIn->getInfo();
            filterFinishEvent = surfVppIn->getEvent();
        } else if (auto surfVppInDev = taskSurf->surf().cudev(); surfVppInDev != nullptr) {
            inputFrame = surfVppInDev->getInfo();
            filterFinishEvent = surfVppInDev->getEvent();
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Invalid input frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        // m_videoMetric->filterを実行するm_streamが、taskSurfの依存する処理を待つように指示する
        taskSurf->setDependCUStream(m_stream);

        { //フレームを転送
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            int dummy = 0;
            auto err = m_videoMetric->filter(&inputFrame, nullptr, &dummy, m_stream);
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to send frame for video metric calcualtion: %s.\n"), get_err_mes(err));
                return err;
            }
            auto cudaEvent = m_frameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
            if (!cudaEvent) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get cuda event.\n"));
                return RGY_ERR_UNKNOWN;
            }
            //eventを入力フレームを使用し終わったことの合図として登録する
            auto cuerr = cudaEventRecord(*cudaEvent, m_stream);
            if (cuerr != cudaSuccess) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to record event for video metric calcualtion: %s.\n"), char_to_tstring(cudaGetErrorString(cuerr)).c_str());
                return err_to_rgy(cuerr);
            }
            taskSurf->addCUEvent(cudaEvent);
        }
        m_outQeueue.push_back(std::move(frame));
        return RGY_ERR_NONE;
    }
};

class NVEncRunCtx {
protected:
    const NVGPUInfo *m_dev;
    cudaStream_t m_streamIn;
    cudaStream_t m_streamOut;
    RGYQueueMPMP<CUFrameEnc *> m_qEncodeBufferUsed;
    RGYQueueMPMP<CUFrameEnc *> m_qEncodeBufferFree;
    EncodeBuffer               m_stEOSOutputBfr;
    std::vector<std::unique_ptr<EncodeBuffer>> m_stEncodeBuffer;  //エンコーダへのフレームバッファ
    RGYLog *log;
public:
    NVEncRunCtx(NVGPUInfo *dev_, RGYLog *log_) : m_dev(dev_), m_streamIn(nullptr), m_streamOut(nullptr), m_qEncodeBufferUsed(), m_qEncodeBufferFree(), m_stEOSOutputBfr(), m_stEncodeBuffer(), log(log_) {}
    ~NVEncRunCtx() {
        releaseEncodeBuffer();
        {
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            if (m_streamIn) {
                cudaStreamDestroy(m_streamIn);
                m_streamIn = nullptr;
            }
            if (m_streamOut) {
                cudaStreamDestroy(m_streamOut);
                m_streamOut = nullptr;
            }
        }
        m_stEncodeBuffer.clear();
        m_qEncodeBufferUsed.clear();
        m_qEncodeBufferFree.clear();
    }
    const cudaStream_t& streamIn() const { return m_streamIn; }
    const cudaStream_t& streamOut() const { return m_streamOut; }
    cudaStream_t& streamIn() { return m_streamIn; }
    cudaStream_t& streamOut() { return m_streamOut; }
    RGYQueueMPMP<CUFrameEnc *> &qEncodeBufferUsed() { return m_qEncodeBufferUsed; }
    RGYQueueMPMP<CUFrameEnc *> &qEncodeBufferFree() { return m_qEncodeBufferFree; }
    EncodeBuffer &stEOSOutputBfr() { return m_stEOSOutputBfr; }
    std::vector<std::unique_ptr<EncodeBuffer>> &stEncodeBuffer() { return m_stEncodeBuffer; }

    // 初期化
    RGY_ERR init() {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        auto ret = cudaStreamCreateWithFlags(&m_streamIn, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            return err_to_rgy(ret);
        }
        ret = cudaStreamCreateWithFlags(&m_streamOut, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            return err_to_rgy(ret);
        }
        m_qEncodeBufferUsed.init(128);
        m_qEncodeBufferFree.init(128);
        return RGY_ERR_NONE;
    }
    RGY_ERR allocEncodeBuffer(const uint32_t uInputWidth, const uint32_t uInputHeight, const NV_ENC_BUFFER_FORMAT inputFormat, const NV_ENC_PIC_STRUCT picStruct, const bool alphaChannel, const int numFrames) {
        uint32_t uInputWidthByte = 0;
        uint32_t uInputHeightTotal = 0;
        switch (inputFormat) {
        case NV_ENC_BUFFER_FORMAT_UNDEFINED: /**< Undefined buffer format */
        case NV_ENC_BUFFER_FORMAT_YV12:      /**< Planar YUV [Y plane followed by V and U planes] */
        case NV_ENC_BUFFER_FORMAT_IYUV:      /**< Planar YUV [Y plane followed by U and V planes] */
            return RGY_ERR_UNSUPPORTED;
        case NV_ENC_BUFFER_FORMAT_YUV444:    /**< Planar YUV [Y plane followed by U and V planes] */
            uInputWidthByte = uInputWidth;
            uInputHeightTotal = uInputHeight * 3;
            break;
        case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: /**< 10 bit Semi-Planar YUV [Y plane followed by interleaved UV plane]. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data. */
            uInputWidthByte = uInputWidth * 2;
            uInputHeightTotal = uInputHeight * 3 / 2;
            break;
        case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: /**< 10 bit Planar YUV444 [Y plane followed by U and V planes]. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
            uInputWidthByte = uInputWidth * 2;
            uInputHeightTotal = uInputHeight * 3;
            break;
        case NV_ENC_BUFFER_FORMAT_ARGB:    /**< 8 bit Packed A8R8G8B8 */
        case NV_ENC_BUFFER_FORMAT_ARGB10:  /**< 10 bit Packed A2R10G10B10. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
        case NV_ENC_BUFFER_FORMAT_AYUV:    /**< 8 bit Packed A8Y8U8V8 */
        case NV_ENC_BUFFER_FORMAT_ABGR:    /**< 8 bit Packed A8B8G8R8 */
        case NV_ENC_BUFFER_FORMAT_ABGR10:  /**< 10 bit Packed A2B10G10R10. Each pixel of size 2 bytes. Most Significant 10 bits contain pixel data.  */
            return RGY_ERR_UNSUPPORTED;
        case NV_ENC_BUFFER_FORMAT_NV12:    /**< Semi-Planar YUV [Y plane followed by interleaved UV plane] */
            uInputWidthByte = uInputWidth;
            uInputHeightTotal = uInputHeight * 3 / 2;
            break;
        default:
            return RGY_ERR_UNSUPPORTED;
        }
        log->write(RGY_LOG_DEBUG, RGY_LOGT_CORE, _T("AllocateIOBuffers: %s %dx%d (width byte %d, height total %d), buffer count %d\n"),
            RGY_CSP_NAMES[csp_enc_to_rgy(inputFormat)], uInputWidth, uInputHeight, uInputWidthByte, uInputHeightTotal, numFrames);

        for (int i = 0; i < numFrames; i++) {
            auto bfr = std::make_unique<EncodeBuffer>();
            if (ENABLE_INTERLACE_FROM_HWMEM && picStruct == NV_ENC_PIC_STRUCT_FRAME) {
                auto sts = allocateEncodeBufferFrame(bfr.get(), uInputWidth, uInputHeight, uInputWidthByte, uInputHeightTotal, inputFormat, alphaChannel);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            } else {
                //インタレ保持の場合は、NvEncCreateInputBuffer経由でフレームを渡さないと正常にエンコードできない
                if (alphaChannel) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("alpha channel encoding not supported with interlaced encoding.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                auto sts = err_to_rgy(m_dev->encoder()->NvEncCreateInputBuffer(uInputWidth, uInputHeight, &bfr->stInputBfr.hInputSurface, inputFormat));
                if (sts != RGY_ERR_NONE) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to allocate Input Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
                    return sts;
                }
            }

            bfr->stInputBfr.bufferFmt = inputFormat;
            bfr->stInputBfr.dwWidth = uInputWidth;
            bfr->stInputBfr.dwHeight = uInputHeight;

            auto sts = err_to_rgy(m_dev->encoder()->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &bfr->stOutputBfr.hBitstreamBuffer));
            if (sts != RGY_ERR_NONE) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to allocate Output Buffer, Please reduce MAX_FRAMES_TO_PRELOAD\n"));
                return sts;
            }
            bfr->stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

            sts = err_to_rgy(m_dev->encoder()->NvEncRegisterAsyncEvent(&bfr->stOutputBfr.hOutputEvent));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            bfr->stOutputBfr.bWaitOnEvent = ENABLE_ASYNC != 0;

            m_stEncodeBuffer.push_back(std::move(bfr));
        }
        
        auto sts = err_to_rgy(m_dev->encoder()->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.stOutputBfr.hOutputEvent));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }
    void releaseEncodeBuffer() {
        for (int i = 0; i < m_stEncodeBuffer.size(); i++) {
            releaseEncodeBufferFrame(m_stEncodeBuffer[i].get());
        }
        m_stEncodeBuffer.clear();
        
        if (m_stEOSOutputBfr.stOutputBfr.hOutputEvent) {
            m_dev->encoder()->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.stOutputBfr.hOutputEvent);
            CloseEvent(m_stEOSOutputBfr.stOutputBfr.hOutputEvent);
            m_stEOSOutputBfr.stOutputBfr.hOutputEvent = NULL;
        }
    }
protected:
    RGY_ERR allocateEncodeBufferFrame(EncodeBuffer *bfr, const uint32_t uInputWidth, const uint32_t uInputHeight,
        const uint32_t uInputWidthByte, const uint32_t uInputHeightTotal, const NV_ENC_BUFFER_FORMAT inputFormat, const bool alphaChannel) {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        const auto allocHeight = uInputHeightTotal * (alphaChannel ? 2 : 1);
        size_t allocPitch = 0;
        auto cudaerr = cudaMallocPitch((void **)&bfr->stInputBfr.pNV12devPtr, &allocPitch, uInputWidthByte, allocHeight);
        if (cudaerr != cudaSuccess) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to cuMemAllocPitch, %d (%s)\n"), cudaerr, char_to_tstring(_cudaGetErrorEnum(cudaerr)).c_str());
            return err_to_rgy(cudaerr);
        }
        bfr->stInputBfr.uNV12Stride = (uint32_t)allocPitch;
        //初期化
        auto sts = err_to_rgy(cudaMemset2D((void *)bfr->stInputBfr.pNV12devPtr, bfr->stInputBfr.uNV12Stride, -128, uInputWidthByte, allocHeight));
        if (sts != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to init buffer: %s\n"), get_err_mes(sts));
            return sts;
        }

        auto nvStatus = m_dev->encoder()->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            (void*)bfr->stInputBfr.pNV12devPtr,
            uInputWidth, uInputHeight, bfr->stInputBfr.uNV12Stride, inputFormat,
            &bfr->stInputBfr.nvRegisteredResource);
        if (nvStatus != NV_ENC_SUCCESS) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to register input device memory.\n"));
            return err_to_rgy(nvStatus);
        }
        // alpha channelが必要な場合、メモリ確保は連続で行い、NvEncRegisterResourceを分割する
        if (alphaChannel) {
            bfr->stInputBfrAlpha.pNV12devPtr = (CUdeviceptr)nullptr;
            bfr->stInputBfrAlpha.dwHeight = uInputWidth;
            bfr->stInputBfrAlpha.dwHeight = uInputHeight;
            bfr->stInputBfrAlpha.bufferFmt = inputFormat;
            bfr->stInputBfrAlpha.uNV12Stride = bfr->stInputBfr.uNV12Stride;
            uint8_t *ptr = (uint8_t *)bfr->stInputBfr.pNV12devPtr;
            ptr += bfr->stInputBfr.uNV12Stride * uInputHeightTotal;
            nvStatus = m_dev->encoder()->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                (void*)ptr, uInputWidth, uInputHeight, bfr->stInputBfrAlpha.uNV12Stride, inputFormat,
                &bfr->stInputBfrAlpha.nvRegisteredResource);
            if (nvStatus != NV_ENC_SUCCESS) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to register input device memory.\n"));
                return err_to_rgy(nvStatus);
            }
        }
        return RGY_ERR_NONE;
    }
    void releaseEncodeBufferFrame(EncodeBuffer *bfr) {
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        if (bfr->stInputBfr.pNV12devPtr) {
            cuMemFree(bfr->stInputBfr.pNV12devPtr);
            bfr->stInputBfr.pNV12devPtr = 0;
        } else {
            if (bfr->stInputBfr.hInputSurface) {
                m_dev->encoder()->NvEncDestroyInputBuffer(bfr->stInputBfr.hInputSurface);
                bfr->stInputBfr.hInputSurface = NULL;
            }
        }

        if (bfr->stOutputBfr.hBitstreamBuffer) {
            m_dev->encoder()->NvEncDestroyBitstreamBuffer(bfr->stOutputBfr.hBitstreamBuffer);
            bfr->stOutputBfr.hBitstreamBuffer = NULL;
        }
        if (bfr->stOutputBfr.hOutputEvent) {
            m_dev->encoder()->NvEncUnregisterAsyncEvent(bfr->stOutputBfr.hOutputEvent);
            CloseEvent(bfr->stOutputBfr.hOutputEvent);
            bfr->stOutputBfr.hOutputEvent = NULL;
        }
    }
};

class PipelineTaskNVEncode : public PipelineTask {
protected:
    NVEncRunCtx *m_runCtx;
    const RGY_CODEC m_encCodec;
    const int m_encWidth;
    const int m_encHeight;
    const RGY_CSP m_encCsp;
    const int m_encBitdepth;
    const RGY_PICSTRUCT m_encPicStruct;
    const NV_ENC_CONFIG& m_stEncConfig;
    const NV_ENC_INITIALIZE_PARAMS& m_stCreateEncodeParams;
    RGYTimecode *m_timecode;
    RGYTimestamp *m_encTimestamp;
    rgy_rational<int> m_outputTimebase;
    RGYListRef<RGYBitstream> m_bitStreamOut;
    RGYHDR10Plus *m_hdr10plus;
    const DOVIRpu *m_doviRpu;
    std::vector<NVEncRCParam>& m_dynamicRC;
    int m_appliedDynamicRC;
    std::vector<int>& m_keyFile;
    bool m_keyOnChapter;
    std::vector<std::unique_ptr<AVChapter>>& m_Chapters;
    std::thread m_threadOutput;
    bool m_threadOutputAbort;
public:
    PipelineTaskNVEncode(
        NVGPUInfo *dev, NVEncRunCtx *runCtx, RGY_CODEC encCodec, int encWidth, int encHeight, RGY_CSP encCsp, int encBitdepth, RGY_PICSTRUCT encPicStruct,
        const NV_ENC_CONFIG& stEncConfig, const NV_ENC_INITIALIZE_PARAMS& stCreateEncodeParams,
        RGYTimecode *timecode, RGYTimestamp *encTimestamp, rgy_rational<int> outputTimebase, RGYHDR10Plus *hdr10plus, const DOVIRpu *doviRpu,
        std::vector<NVEncRCParam>& dynamicRC, std::vector<int>& keyFile, bool keyOnChapter, std::vector<std::unique_ptr<AVChapter>>& chapters,
         int outMaxQueueSize, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::NVENC, dev, outMaxQueueSize, true, log),
        m_runCtx(runCtx), m_encCodec(encCodec), m_encWidth(encWidth), m_encHeight(encHeight), m_encCsp(encCsp), m_encBitdepth(encBitdepth), m_encPicStruct(encPicStruct),
        m_stEncConfig(stEncConfig), m_stCreateEncodeParams(stCreateEncodeParams),
        m_timecode(timecode), m_encTimestamp(encTimestamp), m_outputTimebase(outputTimebase),
        m_bitStreamOut(), m_hdr10plus(hdr10plus), m_doviRpu(doviRpu), m_dynamicRC(dynamicRC), m_appliedDynamicRC(-1), m_keyFile(keyFile), m_keyOnChapter(keyOnChapter), m_Chapters(chapters),
        m_threadOutput(), m_threadOutputAbort(false) {
        runThreadOutput();
    };
    virtual ~PipelineTaskNVEncode() {
        flushEncoder();
        m_outQeueue.clear(); // m_bitStreamOutが解放されるより前にこちらを解放する
        m_threadOutputAbort = true;
        if (m_threadOutput.joinable()) {
            m_threadOutput.join();
        }
    };

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override {
        RGYFrameInfo info(m_encWidth, m_encHeight, m_encCsp, m_encBitdepth, m_encPicStruct, RGY_MEM_TYPE_GPU);
        return std::make_pair(info, m_outMaxQueueSize);
    }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    std::pair<RGY_ERR, std::shared_ptr<RGYBitstream>> getOutputBitstream(const EncodeBuffer *pEncodeBuffer) {
        if (!pEncodeBuffer->stOutputBfr.hBitstreamBuffer && !pEncodeBuffer->stOutputBfr.bEOSFlag) {
            return { RGY_ERR_INVALID_PARAM, nullptr };
        }

        if (pEncodeBuffer->stOutputBfr.bWaitOnEvent) {
            if (!pEncodeBuffer->stOutputBfr.hOutputEvent) {
                return { RGY_ERR_INVALID_PARAM, nullptr };
            }
            WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
        }

        if (pEncodeBuffer->stOutputBfr.bEOSFlag) {
            return { RGY_ERR_MORE_DATA, nullptr };
        }
        auto output = m_bitStreamOut.get([](RGYBitstream *bs) {
            *bs = RGYBitstreamInit();
            return 0;
        });
        if (!output) {
            return { RGY_ERR_NULL_PTR, nullptr };
        }

        NV_ENC_LOCK_BITSTREAM lockBitstreamData = { 0 };
        m_dev->encoder()->setStructVer(lockBitstreamData);
        lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
        lockBitstreamData.doNotWait = false;

        auto nvStatus = m_dev->encoder()->NvEncLockBitstream(&lockBitstreamData);
        if (nvStatus != NV_ENC_SUCCESS) {
            return { err_to_rgy(nvStatus), nullptr };
        }
        output->copy((uint8_t *)lockBitstreamData.bitstreamBufferPtr, lockBitstreamData.bitstreamSizeInBytes, (int64_t)0, (int64_t)lockBitstreamData.outputTimeStamp);
        output->setAvgQP(lockBitstreamData.frameAvgQP);
        output->setFrametype(frametype_enc_to_rgy(lockBitstreamData.pictureType));
        output->setPicstruct(picstruct_enc_to_rgy(lockBitstreamData.pictureStruct));
        output->setFrameIdx(lockBitstreamData.frameIdx);
        output->setDuration(lockBitstreamData.outputDuration);
        nvStatus = m_dev->encoder()->NvEncUnlockBitstream(pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
        return { RGY_ERR_NONE, output };
    }

    RGY_ERR runThreadOutput() {
        m_threadOutput = std::thread([this]() {
            while (!m_threadOutputAbort) {
                CUFrameEnc *frameEnc = nullptr;
                while (!m_runCtx->qEncodeBufferUsed().front_copy_and_pop_no_lock(&frameEnc)) {
                    m_runCtx->qEncodeBufferUsed().wait_for_push(); // 最大16ms待機
                    if (m_threadOutputAbort) {
                        return RGY_ERR_ABORTED;
                    }
                }
                if (!frameEnc) {
                    continue;
                }
                auto outBs = getOutputBitstream(frameEnc->encBuffer());
                if (outBs.first != RGY_ERR_NONE) {
                    if (outBs.first == RGY_ERR_MORE_DATA) {
                        PrintMes(RGY_LOG_DEBUG, _T("Output thread reached EOS.\n"));
                        return outBs.first;
                    }
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get output bitstream: %s.\n"), get_err_mes(outBs.first));
                    return outBs.first;
                }
                m_runCtx->qEncodeBufferFree().push(frameEnc);
                {
                    // m_outQeueueへのロックが必要ならロックを取得
                    std::optional<std::lock_guard<std::mutex>> lock;
                    if (m_outQeueueMtx) {
                        lock.emplace(*m_outQeueueMtx);
                    }
                    m_outQeueue.push_back(std::make_unique<PipelineTaskOutputBitstream>(outBs.second));
                }
            }
            return RGY_ERR_NONE;
        });
        return RGY_ERR_NONE;
    }

    RGY_ERR encodeFrame(EncodeBuffer *pEncodeBuffer, const int id, const int64_t timestamp, const int64_t duration, const int inputFrameId, const std::vector<std::shared_ptr<RGYFrameData>>& frameDataList) {
        PrintMes((inputFrameId < 0 || timestamp < 0 || duration < 0) ? RGY_LOG_WARN : RGY_LOG_TRACE, _T("Sending frame #%d to encoder: timestamp %lld, duration %lld\n"), inputFrameId, timestamp, duration);
        NV_ENC_PIC_PARAMS encPicParams = { 0 };
        m_dev->encoder()->setStructVer(encPicParams);

        if (m_dynamicRC.size() > 0) {
            int selectedIdx = DYNAMIC_PARAM_NOT_SELECTED;
            for (int i = 0; i < (int)m_dynamicRC.size(); i++) {
                if (m_dynamicRC[i].start <= id && id <= m_dynamicRC[i].end) {
                    selectedIdx = i;
                }
                if (m_dynamicRC[i].start > id) {
                    break;
                }
            }
            if (m_appliedDynamicRC != selectedIdx) {
                NV_ENC_CONFIG encConfig = m_stEncConfig; //エンコード設定
                NV_ENC_RECONFIGURE_PARAMS reconf_params = { 0 };
                m_dev->encoder()->setStructVer(reconf_params);
                reconf_params.resetEncoder = 1;
                reconf_params.forceIDR = 1;
                reconf_params.reInitEncodeParams = m_stCreateEncodeParams;
                reconf_params.reInitEncodeParams.encodeConfig = &encConfig;
                if (selectedIdx >= 0) {
                    const auto &selectedPrms = m_dynamicRC[selectedIdx];
                    encConfig.rcParams.rateControlMode = selectedPrms.rc_mode;
                    // API v10.0で追加されたmultipass関係の互換性維持
                    if (m_dev->encoder()->checkAPIver(10, 0)) {
                        if (encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CBR_HQ) {
                            encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
                            encConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
                        } else if (encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_VBR_HQ) {
                            encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
                            encConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
                        }
                    } else {
                        if (encConfig.rcParams.multiPass != NV_ENC_MULTI_PASS_DISABLED) {
                            encConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
                            if (encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CBR) {
                                encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_HQ;
                            } else if (encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_VBR) {
                                encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR_HQ;
                            }
                        }
                    }
                    int averageBitRateUsed = 0;
                    if (encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CONSTQP) {
                        setQP(encConfig.rcParams.constQP, selectedPrms.qp);
                    } else {
                        encConfig.rcParams.averageBitRate = selectedPrms.avg_bitrate;
                        averageBitRateUsed = encConfig.rcParams.averageBitRate;
                        if (selectedPrms.targetQuality >= 0 && selectedPrms.targetQualityLSB >= 0) {
                            encConfig.rcParams.targetQuality    = (uint8_t)selectedPrms.targetQuality;
                            encConfig.rcParams.targetQualityLSB = (uint8_t)selectedPrms.targetQualityLSB;
                        }
                    }
                    if (selectedPrms.max_bitrate > 0) {
                        encConfig.rcParams.maxBitRate = std::max(selectedPrms.max_bitrate, averageBitRateUsed);
                    }
                }
                NVENCSTATUS nvStatus = m_dev->encoder()->NvEncReconfigureEncoder(&reconf_params);
                if (nvStatus != NV_ENC_SUCCESS) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to reconfigure the encoder.\n"));
                    return err_to_rgy(nvStatus);
                }
                m_appliedDynamicRC = selectedIdx;
                PrintMes(RGY_LOG_DEBUG, _T("Reconfigured encoder (%d).\n"), selectedIdx);
            }
        }

        if (m_Chapters.size() > 0 && m_keyOnChapter) {
            for (const auto& chap : m_Chapters) {
                //av_cmopare_tsを使うと、timebaseが粗く端数が出る場合に厳密に比較できないことがある
                //そこで、ここでは、最小公倍数をとって厳密な比較を行う
                const auto timebase_lcm = std::lcm<int64_t, int64_t>(chap->time_base.den, m_outputTimebase.d());
                ttint128 ts_frame = timestamp;
                ts_frame *= m_outputTimebase.n();
                ts_frame *= timebase_lcm / m_outputTimebase.d();

                ttint128 ts_chap = chap->start;
                ts_chap *= chap->time_base.num;
                ts_chap *= timebase_lcm / chap->time_base.den;

                if (chap->id >= 0 && ts_chap <= ts_frame) {
                    PrintMes(RGY_LOG_DEBUG, _T("Insert Keyframe on chapter %d: %s at frame #%d: %s (timebase: %lld).\n"),
                        chap->id,
                        wstring_to_tstring(ts_chap.ToWString()).c_str(),
                        id,
                        wstring_to_tstring(ts_frame.ToWString()).c_str(),
                        timebase_lcm);
                    chap->id = -1;
                    encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
                    break;
                }
            }
        }
        if (std::find(m_keyFile.begin(), m_keyFile.end(), id) != m_keyFile.end()) {
            PrintMes(RGY_LOG_DEBUG, _T("Insert Keyframe on frame #%d.\n"), id);
            encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        std::vector<std::shared_ptr<RGYFrameData>> metadatalist;
        if (m_encCodec == RGY_CODEC_HEVC || m_encCodec == RGY_CODEC_AV1) {
            metadatalist = frameDataList;
            if (m_hdr10plus) {
                // 外部からHDR10+を読み込む場合、metadatalist 内のHDR10+の削除
                for (auto it = metadatalist.begin(); it != metadatalist.end(); ) {
                    if ((*it)->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        it = metadatalist.erase(it);
                    } else {
                        it++;
                    }
                }
                // 外部からHDR10+を読み込む
                if (const auto data = m_hdr10plus->getData(inputFrameId); data.size() > 0) {
                    metadatalist.push_back(std::make_shared<RGYFrameDataHDR10plus>(data.data(), data.size(), timestamp));
                }
            }
            if (m_doviRpu) {
                // 外部からdoviを読み込む場合、metadatalist 内のdovi rpuの削除
                for (auto it = metadatalist.begin(); it != metadatalist.end(); ) {
                    if ((*it)->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        it = metadatalist.erase(it);
                    } else {
                        it++;
                    }
                }
            }
        }

        if (m_timecode) {
            m_timecode->write(timestamp, m_outputTimebase);
        }

        encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
        encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
        encPicParams.inputWidth = m_encWidth;
        encPicParams.inputHeight = m_encHeight;
        encPicParams.inputPitch = pEncodeBuffer->stInputBfr.uNV12Stride;
        encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
        encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
        encPicParams.inputTimeStamp = timestamp;
        encPicParams.inputDuration = duration;
        encPicParams.pictureStruct = picstruct_rgy_to_enc(m_encPicStruct);
        encPicParams.alphaBuffer = pEncodeBuffer->stInputBfrAlpha.hInputSurface;
        //encPicParams.qpDeltaMap = qpDeltaMapArray;
        //encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;

        //if (encPicCommand)
        //{
        //    if (encPicCommand->bForceIDR)
        //    {
        //        encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        //    }

        //    if (encPicCommand->bForceIntraRefresh)
        //    {
        //        if (codecGUID == NV_ENC_CODEC_HEVC_GUID)
        //        {
        //            encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
        //        }
        //        else
        //        {
        //            encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
        //        }
        //    }
        //}

        if (inputFrameId < 0) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid input frame ID %d sent to encoder.\n"), inputFrameId);
            return RGY_ERR_INVALID_CALL;
        }
        m_encTimestamp->add(timestamp, inputFrameId, (encPicParams.frameIdx = id), duration, metadatalist);

        NVENCSTATUS nvStatus = m_dev->encoder()->NvEncEncodePicture(&encPicParams);
        if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to add frame into the encoder.\n"));
            return err_to_rgy(nvStatus);
        }
        PrintMes(RGY_LOG_TRACE, _T("  Sent frame %d to encoder\n"), inputFrameId);

        return RGY_ERR_NONE;
    }

    virtual RGY_ERR flushEncoder() {
        m_runCtx->stEOSOutputBfr().stOutputBfr.bEOSFlag = true;
        CUFrameEncHostWrap flushBuffer(&m_runCtx->stEOSOutputBfr(), m_dev->encoder(), false);

        auto sts = err_to_rgy(m_dev->encoder()->NvEncFlushEncoderQueue(m_runCtx->stEOSOutputBfr().stOutputBfr.hOutputEvent));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_runCtx->qEncodeBufferUsed().push(&flushBuffer);

        //if (m_runCtx->stEOSOutputBfr().stOutputBfr.hOutputEvent && WaitForSingleObject(m_runCtx->stEOSOutputBfr().stOutputBfr.hOutputEvent, 1000) != WAIT_OBJECT_0) {
        //    PrintMes(RGY_LOG_ERROR, _T("m_stEOSOutputBfr.hOutputEvent%s"), (FOR_AUO) ? _T("が終了しません。") : _T(" does not finish within proper time."));
        //    return RGY_ERR_UNKNOWN;
        //}
        // flushしたらそれを受けて出力スレッドが終了するのを待つ
        if (m_threadOutput.joinable()) {
            m_threadOutput.join();
        }
        return RGY_ERR_MORE_DATA;
    }

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (frame && frame->type() != PipelineTaskOutputType::SURFACE) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        auto surfEncodeIn = (frame) ? dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().enc() : nullptr;
        if (surfEncodeIn) {
            //前の同期アイテムをm_streamInが待機するようにし、cueventsをクリア(NVEncCtxAutoLockはこの中)
            dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->streamWaitCuEvents(m_runCtx->streamIn());

            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            if (surfEncodeIn->bufType() == CUFrameBufType::EncDevWrap) {
                auto sts = surfEncodeIn->map();
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to map frame: %s.\n"), get_err_mes(sts));
                    return sts;
                }
            } else if (surfEncodeIn->bufType() == CUFrameBufType::EncHostWrap) {
                surfEncodeIn->unmap();
            }
        }
        
        if (surfEncodeIn) {
            auto sts = encodeFrame(surfEncodeIn->encBuffer(), m_inFrames++, surfEncodeIn->timestamp(), surfEncodeIn->duration(), surfEncodeIn->inputFrameId(), surfEncodeIn->dataList());
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to encode frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
            surfEncodeIn->clearDataList();

            m_runCtx->qEncodeBufferUsed().push(surfEncodeIn);
        } else {
            // フレームがない場合は、エンコーダのキューをフラッシュ
            auto sts = flushEncoder();
            if (sts != RGY_ERR_NONE) {
                if (sts != RGY_ERR_MORE_DATA) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to flush encoder queue: %s.\n"), get_err_mes(sts));
                }
                return sts;
            }
        }
        return RGY_ERR_NONE;
    }
};

class PipelineTaskCUDAVpp : public PipelineTask {
protected:
    std::vector<std::unique_ptr<NVEncFilter>>& m_vpFilters;
    NVEncFilterSsim *m_videoMetric;
    FrameReleaseData<cudaEvent_t> m_frameReleaseData;
    RGYListRef<cudaEvent_t> m_inFrameUseFinEvent;
    RGYQueueMPMP<CUFrameEnc *>& m_qEncodeBufferFree;
    bool m_rgbAsYUV444;
    cudaEvent_t m_eventDefaultToFilter;
    cudaStream_t m_streamFilter;
    cudaStream_t m_streamDownload;
    std::unique_ptr<PipelineTaskOutput> m_cuvidPrev;
public:
    PipelineTaskCUDAVpp(NVGPUInfo *dev, std::vector<std::unique_ptr<NVEncFilter>>& vppfilters, NVEncFilterSsim *videoMetric,
    RGYQueueMPMP<CUFrameEnc *>& qEncodeBufferFree, bool rgbAsYUV444, int outMaxQueueSize, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CUDA, dev, outMaxQueueSize, false, log), m_vpFilters(vppfilters), m_videoMetric(videoMetric),
        m_frameReleaseData(dev->vidCtxLock()), m_inFrameUseFinEvent(), m_qEncodeBufferFree(qEncodeBufferFree), m_rgbAsYUV444(rgbAsYUV444),
        m_eventDefaultToFilter(nullptr),m_streamFilter(nullptr), m_streamDownload(nullptr), m_cuvidPrev() {

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        auto ret = cudaStreamCreateWithFlags(&m_streamFilter, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create upload stream: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
        }
        m_streamFilter = cudaStreamPerThread;
        ret = cudaStreamCreateWithFlags(&m_streamDownload, cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create download stream: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
        }
        ret = cudaEventCreateWithFlags(&m_eventDefaultToFilter, cudaEventDefault);
        if (ret != cudaSuccess) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create event for default to filter: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
        }
        runFrameReleaseThread();
    };


    virtual ~PipelineTaskCUDAVpp() {
        m_frameReleaseData.finish();
        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        if (m_streamFilter) {
            cudaStreamDestroy(m_streamFilter);
            m_streamFilter = nullptr;
        }
        if (m_streamDownload) {
            cudaStreamDestroy(m_streamDownload);
            m_streamDownload = nullptr;
        }
        if (m_eventDefaultToFilter) {
            cudaEventDestroy(m_eventDefaultToFilter);
            m_eventDefaultToFilter = nullptr;
        }
        m_outQeueue.clear(); // m_inFrameUseFinEvent解放前に行うこと
        m_inFrameUseFinEvent.clear([](cudaEvent_t *event) { cudaEventDestroy(*event); });
    };


    void setVideoQualityMetricFilter(NVEncFilterSsim *videoMetric) {
        m_videoMetric = videoMetric;
    }

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override {
        auto lastFilterFrame = m_vpFilters.back()->GetFilterParam()->frameOut;
        return std::make_pair(lastFilterFrame, 0);
    };

    virtual void runFrameReleaseThread() {
        m_frameReleaseData.start();
    }

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        std::deque<std::pair<RGYFrameInfo, uint32_t>> filterframes;
        bool drain = !frame;
        cudaStream_t streamFilter = m_streamFilter;
        if (!frame) {
            filterframes.push_back(std::make_pair(RGYFrameInfo(), 0u));
        } else {
            auto taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
            if (taskSurf == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
                return RGY_ERR_NULL_PTR;
            }
            if (auto surfVppInCuvid = taskSurf->surf().cuvid(); surfVppInCuvid != nullptr) {
                // cuvidでは、cuvidのmap/unmapが同時に多重にできないので、まず前のフレームを解放を待つ
                m_frameReleaseData.waitUntilEmpty();
                PrintMes(RGY_LOG_TRACE, _T("filter_frame: map video frame: %d, %lld.\n"), surfVppInCuvid->dispInfo()->picture_index, surfVppInCuvid->dispInfo()->timestamp);
                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                auto sts = surfVppInCuvid->mapFrame();
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to map frame: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                // cuvidでmapする際に、CUDAのdefault streamを使用するようで、これと同期する必要がある
                // これを行わないと、cuvidの処理が終わる前に(?)CUDAの処理が行われておかしくなるようである
                streamFilter = cudaStreamPerThread;
                filterframes.push_back(std::make_pair(surfVppInCuvid->getInfo(), 0u));
            } else if (auto surfVppInCU = taskSurf->surf().cubuf(); surfVppInCU != nullptr) {
                filterframes.push_back(std::make_pair(surfVppInCU->getInfo(), 0u));
            } else if (auto surfVppInCUDev = taskSurf->surf().cudev(); surfVppInCUDev != nullptr) {
                // filterを実行するm_streamFilterが、taskSurfの依存する処理を待つように指示する
                taskSurf->setDependCUStream(m_streamFilter);
                filterframes.push_back(std::make_pair(surfVppInCUDev->getInfo(), 0u));
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid task surface (not opencl or amf).\n"));
                return RGY_ERR_NULL_PTR;
            }
        }
#define FRAME_COPY_ONLY 0
#if !FRAME_COPY_ONLY
        std::vector<std::unique_ptr<PipelineTaskOutputSurf>> outputSurfs;
        while (filterframes.size() > 0 || drain) {
            //フィルタリングするならここ
            for (uint32_t ifilter = filterframes.front().second; ifilter < m_vpFilters.size() - 1; ifilter++) {
                // コピーを作ってそれをfilter関数に渡す
                // vpp-rffなどoverwirteするフィルタのときに、filterframes.pop_front -> push がうまく動作しない
                RGYFrameInfo input = filterframes.front().first;

                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                auto sts_filter = m_vpFilters[ifilter]->filter(&input, (RGYFrameInfo **)&outInfo, &nOutFrames, streamFilter);
                if (sts_filter != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_vpFilters[ifilter]->name().c_str());
                    return sts_filter;
                }
                if (nOutFrames == 0) {
                    if (drain) {
                        filterframes.front().second++;
                        continue;
                    }
                    return RGY_ERR_NONE;
                }
                if (ifilter == 0) { //最初のフィルタなら転送なので、イベントをここでセットする
                    auto cudaEvent = m_inFrameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
                    if (!cudaEvent) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to get cuda event.\n"));
                        return RGY_ERR_UNKNOWN;
                    }
                    cudaEventRecord(*cudaEvent, streamFilter);
                    //ここでinput frameの参照を m_prevInputFrame で保持するようにして、CUDAによるフレームの処理が完了しているかを確認できるようにする
                    //これを行わないとこのフレームが再度使われてしまうことになる
                    m_frameReleaseData.addFrame(frame, cudaEvent);
                }
                // cuvidとの同期のため、cudaStreamPerThreadを最初に使った場合でも、その次のフィルタはm_streamFilterを使用するようにする
                if (streamFilter != m_streamFilter) {
                    auto err = cudaEventRecord(m_eventDefaultToFilter, streamFilter);
                    if (err != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to record cuda event for default to filter.\n"));
                        return err_to_rgy(err);
                    }
                    err = cudaStreamWaitEvent(m_streamFilter, m_eventDefaultToFilter, 0);
                    if (err != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to set wait for cuda event for streamFilter -> m_streamFilter.\n"));
                        return err_to_rgy(err);
                    }
                    streamFilter = m_streamFilter;
                }
                drain = false; //途中でフレームが出てきたら、drain完了していない
                filterframes.pop_front();
                //最初に出てきたフレームは先頭に追加する
                for (int jframe = nOutFrames - 1; jframe >= 0; jframe--) {
                    filterframes.push_front(std::make_pair(*outInfo[jframe], ifilter + 1));
                }
            }
            if (drain) {
                return RGY_ERR_MORE_DATA; //最後までdrain = trueなら、drain完了
            }

            PipelineTaskSurface frameVppOut; 
            if (m_dev->encoder()) {
                // 使用していないエンコードバッファを取得
                CUFrameEnc *encBuffer = nullptr;
                while (!m_qEncodeBufferFree.front_copy_and_pop_no_lock(&encBuffer)) {
                    m_qEncodeBufferFree.wait_for_push(); // 最大16ms待機
                }
                frameVppOut = m_workSurfs.get(encBuffer);
                if (!frameVppOut) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get work surface for vpp output.\n"));
                    return RGY_ERR_NULL_PTR;
                }
                if (!frameVppOut.enc()) {
                    PrintMes(RGY_LOG_ERROR, _T("Encoder enabled but frame type for vpp output is not for encoder.\n"));
                    return RGY_ERR_NULL_PTR;
                }
                if (frameVppOut.enc()->bufType() == CUFrameBufType::EncHostWrap) {
                    frameVppOut.enc()->map(); // NvEncLockInputBuffer
                }

            } else {
                frameVppOut = m_workSurfs.getFreeSurf();
            }

            //エンコードバッファにコピー
            auto &lastFilter = m_vpFilters[m_vpFilters.size() - 1];
            //最後のフィルタはNVEncFilterCspCropでなければならない
            if (typeid(*lastFilter.get()) != typeid(NVEncFilterCspCrop)) {
                PrintMes(RGY_LOG_ERROR, _T("Last filter setting invalid.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
            NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
            int nOutFrames = 0;
            RGYFrameInfo *outInfo[16] = { 0 };
            //エンコードバッファの情報を設定
            RGYFrameInfo frameVppOutInfo;
            if (frameVppOut.enc()) {
                frameVppOutInfo = frameVppOut.enc()->getInfo();
            } else if (frameVppOut.cubuf()) {
                frameVppOutInfo = frameVppOut.cubuf()->getInfo();
            } else if (frameVppOut.cudev()) {
                frameVppOutInfo = frameVppOut.cudev()->getInfo();
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid frame type for vpp output.\n"));
                return RGY_ERR_NULL_PTR;
            }
            std::shared_ptr<cudaEvent_t> cudaEventFilterToDownload;
            auto streamLastFilter = streamFilter;
            RGYFrameInfo& ssimTarget = (frameVppOutInfo.mem_type == RGY_MEM_TYPE_CPU) ? filterframes.front().first : frameVppOutInfo;
            if (frameVppOutInfo.mem_type == RGY_MEM_TYPE_CPU) {
                if (m_vpFilters.size() > 1) {
                    // 最後のフィルタ(=copyDtoH)だけではに場合は、専用のstream(m_streamDownload)を使用するようにして高速化
                    if (streamLastFilter != cudaStreamPerThread) {
                        streamLastFilter = m_streamDownload;
                    }
                    // 最後のフィルタではm_streamFilterではなくm_streamDownloadを使用するため、
                    // メモリをダウンロードするためのイベントを作成する

                    cudaEventFilterToDownload = m_inFrameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
                    if (!cudaEventFilterToDownload) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to get cuda event .\n"));
                        return RGY_ERR_UNKNOWN;
                    }
                    auto err = cudaEventRecord(*cudaEventFilterToDownload, m_streamFilter);
                    if (err != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to record cuda event for m_streamFilter -> m_streamDownload.\n"));
                        return err_to_rgy(err);
                    }
                    err = cudaStreamWaitEvent(m_streamDownload, *cudaEventFilterToDownload, 0);
                    if (err != cudaSuccess) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to set wait for cuda event for m_streamFilter -> m_streamDownload.\n"));
                        return err_to_rgy(err);
                    }
                }
            }
            // エンコードバッファのポインタを渡す
            outInfo[0] = &frameVppOutInfo;
            auto sts_filter = lastFilter->filter(&filterframes.front().first, (RGYFrameInfo **)&outInfo, &nOutFrames, streamLastFilter);
            if (sts_filter != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), lastFilter->name().c_str());
                return sts_filter;
            }
            if (m_videoMetric) {
                //フレームを転送
                int dummy = 0;
                auto err = m_videoMetric->filter(&ssimTarget, nullptr, &dummy, m_streamFilter);
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to send frame for video metric calcualtion: %s.\n"), get_err_mes(err));
                    return err;
                }
            }
            // 処理の終了を示すイベント
            auto cudaEvent = m_inFrameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
            if (!cudaEvent) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get filter finish cuda event.\n"));
                return RGY_ERR_UNKNOWN;
            }
            auto err = cudaEventRecord(*cudaEvent, streamLastFilter);
            if (err != cudaSuccess) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to record filter finish cuda event.\n"));
                return err_to_rgy(err);
            }
            if (frame) {
                //ここでinput frameの参照を m_prevInputFrame で保持するようにして、CUDAによるフレームの処理が完了しているかを確認できるようにする
                //これを行わないとこのフレームが再度使われてしまうことになる
                m_frameReleaseData.addFrame(frame, cudaEvent);
            }
            filterframes.pop_front();

            frameVppOut.frame()->setDuration(frameVppOutInfo.duration);
            frameVppOut.frame()->setTimestamp(frameVppOutInfo.timestamp);
            frameVppOut.frame()->setInputFrameId(frameVppOutInfo.inputFrameId);
            frameVppOut.frame()->setPicstruct(frameVppOutInfo.picstruct);
            frameVppOut.frame()->setFlags(frameVppOutInfo.flags);
            frameVppOut.frame()->setDataList(frameVppOutInfo.dataList);

            auto outputSurf = std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), frameVppOut, frame, cudaEvent);
            if (cudaEventFilterToDownload) {
                outputSurf->addCUEvent(cudaEventFilterToDownload);
            }
            outputSurfs.push_back(std::move(outputSurf));

            #undef clFrameOutInteropRelease

        }
        m_outQeueue.insert(m_outQeueue.end(),
            std::make_move_iterator(outputSurfs.begin()),
            std::make_move_iterator(outputSurfs.end())
        );
#else
        auto surfVppOut = getWorkSurf();
        if (m_surfVppOutInterop.count(surfVppOut.get()) == 0) {
            m_surfVppOutInterop[surfVppOut.get()] = getOpenCLFrameInterop(surfVppOut.get(), m_memType, CL_MEM_WRITE_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_vpFilters.front()->GetFilterParam()->frameIn);
        }
        auto clFrameOutInterop = m_surfVppOutInterop[surfVppOut.get()].get();
        if (!clFrameOutInterop) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [out].\n"));
            return RGY_ERR_NULL_PTR;
        }
        auto err = clFrameOutInterop->acquire(m_cl->queue());
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [out]: %s.\n"), get_err_mes(err));
            return RGY_ERR_NULL_PTR;
        }
        auto inputSurface = clFrameInInterop->frameInfo();
        surfVppOut->Data.TimeStamp = inputSurface.timestamp;
        surfVppOut->Data.FrameOrder = inputSurface.inputFrameId;
        surfVppOut->Info.PicStruct = picstruct_rgy_to_enc(inputSurface.picstruct);
        surfVppOut->Data.DataFlag = (mfxU16)inputSurface.flags;

        auto encSurfaceInfo = clFrameOutInterop->frameInfo();
        RGYOpenCLEvent clevent;
        m_cl->copyFrame(&encSurfaceInfo, &inputSurface, nullptr, m_cl->queue(), &clevent);
        if (clFrameInInterop) {
            clFrameInInterop->release(&clevent);
            if (!m_prevInputFrame.empty() && m_prevInputFrame.back()) {
                dynamic_cast<PipelineTaskOutputSurf *>(m_prevInputFrame.back().get())->addClEvent(clevent);
            }
        }
        clFrameOutInterop->release(&clevent);
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfVppOut, frame, clevent));
#endif
        return RGY_ERR_NONE;
    }
};

#endif //__NVENC_PIPELINE_H__
