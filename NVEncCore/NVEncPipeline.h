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
#include "rgy_thread_affinity.h"
#include "rgy_timecode.h"
#include "rgy_parallel_enc.h"

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


#if defined(_WIN32) || defined(_WIN64)
#define THREAD_DEC_USE_FUTURE 0
#else
// linuxではスレッド周りの使用の違いにより、従来の実装ではVCECore解放時に異常終了するので、
// std::futureを使った実装に切り替える
// std::threadだとtry joinのようなことができないのが問題
#define THREAD_DEC_USE_FUTURE 1
#endif

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

class PipelineTaskStopWatch {
    std::array<std::vector<std::pair<tstring, int64_t>>, 2> m_ticks;
    std::array<std::chrono::high_resolution_clock::time_point, 2> m_prevTimepoints;
public:
    PipelineTaskStopWatch(const std::vector<tstring>& tickSend, const std::vector<tstring>& tickGet) : m_ticks(), m_prevTimepoints() {
        for (size_t i = 0; i < tickSend.size(); i++) {
            m_ticks[0].push_back({ tickSend[i], 0 });
        }
        for (size_t i = 0; i < tickGet.size(); i++) {
            m_ticks[1].push_back({ tickGet[i], 0 });
        }
    };
    void set(const int type) {
        m_prevTimepoints[type] = std::chrono::high_resolution_clock::now();
    }
    void add(const int type, const int idx) {
        auto now = std::chrono::high_resolution_clock::now();
        m_ticks[type][idx].second += std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_prevTimepoints[type]).count();
        m_prevTimepoints[type] = now;
    }
    int64_t totalTicks() const {
        int64_t total = 0;
        for (int itype = 0; itype < 2; itype++) {
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                total += m_ticks[itype][i].second;
            }
        }
        return total;
    }
    size_t maxWorkStrLen() const {
        size_t maxLen = 0;
        for (size_t itype = 0; itype < m_ticks.size(); itype++) {
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                maxLen = (std::max)(maxLen, m_ticks[itype][i].first.length());
            }
        }
        return maxLen;
    }
    tstring print(const int64_t totalTicks, const size_t maxLen) {
        const TCHAR *type[] = {_T("send"), _T("get ")};
        tstring str;
        for (size_t itype = 0; itype < m_ticks.size(); itype++) {
            int64_t total = 0;
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                str += type[itype] + tstring(_T(":"));
                str += m_ticks[itype][i].first;
                str += tstring(maxLen - m_ticks[itype][i].first.length(), _T(' '));
                str += strsprintf(_T(" : %8d ms [%5.1f]\n"), ((m_ticks[itype][i].second + 500000) / 1000000), m_ticks[itype][i].second * 100.0 / totalTicks);
                total += m_ticks[itype][i].second;
            }
            if (m_ticks[itype].size() > 1) {
                str += type[itype] + tstring(_T(":"));
                str += _T("total");
                str += tstring(maxLen - _tcslen(_T("total")), _T(' '));
                str += strsprintf(_T(" : %8d ms [%5.1f]\n"), ((total + 500000) / 1000000), total * 100.0 / totalTicks);
            }
        }
        return str;
    }
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
    void deleteFreedSurface() {
        for (auto it = m_surfaces.begin(); it != m_surfaces.end();) {
            if ((*it)->isFree()) {
                it = m_surfaces.erase(it);
            } else {
                it++;
            }
        }
    }
protected:
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

    bool hasDependencyFrame() const { return m_dependencyFrame != nullptr; }
    
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
#if 1
                NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                cudaEventSynchronize(*cuevent.get());
#else
                const int MAX_LOOP = 1000;
                for (int i = 0; ; i++) {
                    {
                        NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                        if (cudaEventQuery(*cuevent.get()) != cudaErrorNotReady) {
                            break;
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds((i % MAX_LOOP == (MAX_LOOP - 1)) ? 1 : 0));
                };
#endif
            }
        }
        m_cuevents.clear();
        m_dependencyFrame.reset();
    }
    
    void streamWaitCuEvents(cudaStream_t stream) {
        for (auto& cuevent : m_cuevents) {
            if (cuevent != nullptr) {
                NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
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
    int m_queueSizeMax;
    std::thread m_thread;
    RGYParamThread m_threadParam;
    bool m_abort;
public:
    FrameReleaseData(CUvideoctxlock vidCtxLock, int queueSizeMax, RGYParamThread threadParam) : m_vidCtxLock(vidCtxLock), m_prevInputFrame(), m_mtx(),
        m_heFrameAdded(std::move(CreateEventUnique(nullptr, FALSE, FALSE))),
        m_heQueueEmpty(std::move(CreateEventUnique(nullptr, FALSE, FALSE))),
        m_queueSize(0), m_queueSizeMax(std::max(1, queueSizeMax)), m_thread(), m_threadParam(threadParam), m_abort(false) {}

    ~FrameReleaseData() {
        finish();
    }
    void finish() {
        m_abort = true;
        if (m_thread.joinable()) {
            SetEvent(m_heFrameAdded.get());
            m_thread.join();
        } else {
            waitFrameSingleThread(0);
        }
    }
    void waitUntilEmptyMultiThread() {
        if (m_thread.joinable()) {
            // マルチスレッド動作の場合
            while (m_queueSize > 0) {
                WaitForSingleObject(m_heQueueEmpty.get(), 10);
            }
        }
    }
    int queueSizeMax() const { return m_queueSizeMax; }

    void start() {
        m_thread = std::thread([&]() {
            m_threadParam.apply(GetCurrentThread());
            while (!m_abort) {
                int queueSize = -1;
                TaskOutputEvent prevframe;
                { // m_mtx のロックを取得
                    std::lock_guard<std::mutex> lock(m_mtx);
                    if ((queueSize = (int)m_prevInputFrame.size()) > 0) {
                        prevframe = std::move(m_prevInputFrame.front());
                        m_prevInputFrame.pop_front();
                        queueSize--;
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
                        WaitForSingleObject(m_heFrameAdded.get(), 16);
                    }
                }
            }
        });
    }

    void waitFrameSingleThread(int queueSizeMin) { // シングルスレッド動作での待機用
        if (m_thread.joinable()) {
            // マルチスレッド動作の場合
            return;
        }
        int queueSize = -1;
        TaskOutputEvent prevframe;
        while ((queueSize = (int)m_prevInputFrame.size()) > queueSizeMin) {
            prevframe = std::move(m_prevInputFrame.front());
            m_prevInputFrame.pop_front();
            queueSize--;
            
            if (prevframe.first) {
                prevframe.first->depend_clear();
                if (auto surfVppInCuvid = dynamic_cast<PipelineTaskOutputSurf *>(prevframe.first.get())->surf().cuvid(); surfVppInCuvid != nullptr) {
                    // cuvidでは、cuvidのmap/unmapが同時に多重にできないので、まず前のフレームを解放(unmap)する
                    NVEncCtxAutoLock(ctxlock(m_vidCtxLock));
                    surfVppInCuvid->unmapFrame();
                }
                prevframe.first.reset();
            }
        }
    }

    void addFrame(std::unique_ptr<PipelineTaskOutput>& frame, std::shared_ptr<T> event) {
        dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->addCUEvent(event);
        if (m_thread.joinable()) {
            // マルチスレッド動作の場合
            while (m_queueSize >= m_queueSizeMax) {
                SetEvent(m_heFrameAdded.get());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            m_queueSize++;
            std::lock_guard<std::mutex> lock(m_mtx);
            m_prevInputFrame.push_back(std::make_pair(std::move(frame), event));
            SetEvent(m_heFrameAdded.get());
        } else {
            m_prevInputFrame.push_back(std::make_pair(std::move(frame), event));
        }
    }

    size_t queueSize() {
        std::lock_guard<std::mutex> lock(m_mtx);
        size_t total = 0;
        for (auto& f : m_prevInputFrame) {
            total++;
            if (auto surf = dynamic_cast<PipelineTaskOutputSurf *>(f.first.get()); surf != nullptr) {
                total += surf->hasDependencyFrame() ? 1 : 0;
            }
        }
        return total;
    }
};

enum class PipelineTaskType {
    UNKNOWN,
    NVDEC,
    NVENC,
    INPUT,
    INPUTCU,
    CHECKPTS,
    PECOLLECT,
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
    case PipelineTaskType::PECOLLECT:   return _T("PECOLLECT");
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
    case PipelineTaskType::PECOLLECT:
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
    RGYParamThread m_threadParam;
    std::shared_ptr<RGYLog> m_log;
    RGYLogType m_logType;
    std::unique_ptr<PipelineTaskStopWatch> m_stopwatch;
public:
    PipelineTask() : m_type(PipelineTaskType::UNKNOWN), m_dev(nullptr), m_outQeueue(), m_workSurfs(), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(0), m_log() {};
    PipelineTask(PipelineTaskType type, NVGPUInfo *dev, int outMaxQueueSize, bool useOutQueueMtx, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        m_type(type), m_dev(dev), m_outQeueue(), m_workSurfs(), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(outMaxQueueSize),
        m_outQeueueMtx(useOutQueueMtx ? std::make_unique<std::mutex>() : nullptr), m_threadParam(threadParam), m_log(log), m_logType(RGY_LOGT_CORE) {
    };
    virtual ~PipelineTask() {
        m_workSurfs.clear();
    }
    virtual void setStopWatch() {};
    virtual void printStopWatch(const int64_t totalTicks, const size_t maxLen) {
        if (m_stopwatch) {
            const auto strlines = split(m_stopwatch->print(totalTicks, maxLen), _T("\n"));
            for (auto& str : strlines) {
                if (str.length() > 0) {
                    PrintMes(RGY_LOG_INFO, _T("%s\n"), str.c_str());
                }
            }
        }
    }
    virtual int64_t getStopWatchTotal() const {
        return (m_stopwatch) ? m_stopwatch->totalTicks() : 0ll;
    }
    virtual size_t getStopWatchMaxWorkStrLen() const {
        return (m_stopwatch) ? m_stopwatch->maxWorkStrLen() : 0u;
    }
    virtual size_t getOutQueueFrames() const {
        size_t total = 0;
        { // m_outQeueueにアクセスする場合、必要なら m_outQeueueMtx のロックを取得
            std::optional<std::lock_guard<std::mutex>> lock;
            if (m_outQeueueMtx) {
                lock.emplace(*m_outQeueueMtx);
            }
            for (auto& f : m_outQeueue) {
                total++;
                if (auto surf = dynamic_cast<PipelineTaskOutputSurf *>(f.get()); surf != nullptr) {
                    total += surf->hasDependencyFrame() ? 1 : 0;
                }
            }
        }
        return total;
    }
    virtual void printStatus() {
        PrintMes(RGY_LOG_INFO, _T("in %d, out %d, outQeueue size: %d.\n"), m_inFrames, m_outFrames, (int)getOutQueueFrames());
    }
    virtual bool isPassThrough() const { return false; }
    virtual tstring print() const { return getPipelineTaskTypeName(m_type); }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() = 0;
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() = 0;
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) = 0;
    virtual RGY_ERR getOutputFrameInfo(RGYFrameInfo& info) { info = RGYFrameInfo(); return RGY_ERR_NONE; }
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) {
        if (m_stopwatch) m_stopwatch->set(1);
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
        if (m_stopwatch) m_stopwatch->add(1, 0);
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
        } else if (log_level < m_log->getLogLevel(m_logType)) {
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
            m_log->write(log_level, m_logType, mes.c_str());
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
    int64_t m_endPts; // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    cudaStream_t m_streamUpload;
    RGYListRef<cudaEvent_t> m_frameUseFinEvent;
public:
    PipelineTaskInput(NVGPUInfo *dev, int outMaxQueueSize, RGYInput *input, int64_t endPts, RGYParamThread threadParam, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::INPUT, dev, outMaxQueueSize, false, threadParam, log), m_input(input), m_endPts(endPts), m_streamUpload(nullptr), m_frameUseFinEvent() {
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
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("getWorkSurf"), _T("getRefHostFrame"), _T("LoadNextFrame"), _T("LoadNextFrame"), _T("copyFrameFromHostRef") },
            std::vector<tstring>{_T("")}
        );
    }
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
        if (m_stopwatch) m_stopwatch->set(0);
        auto surfWork = getWorkSurf();
        if (surfWork == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get work surface for input.\n"));
            return RGY_ERR_NOT_ENOUGH_BUFFER;
        }
        if (m_stopwatch) m_stopwatch->add(0, 1);
        auto cuframe = surfWork.cubuf();
        auto hostFrame = cuframe->getRefHostFrame(); // CPUが書き込むための領域を取得
        if (!hostFrame) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get host frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (m_stopwatch) m_stopwatch->add(0, 2);
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
        if (m_stopwatch) m_stopwatch->add(0, 3);
        hostFrame->setInputFrameId(m_inFrames++);
        if (m_endPts >= 0
            && (int64_t)hostFrame->timestamp() != AV_NOPTS_VALUE // timestampが設定されていない場合は無視
            && (int64_t)hostFrame->timestamp() >= m_endPts) { // m_endPtsは含まないようにする(重要)
            return RGY_ERR_MORE_BITSTREAM; //入力ビットストリームは終了
        }

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
        if (m_stopwatch) m_stopwatch->add(0, 4);
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
    int64_t m_endPts; // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    CuvidDecode *m_dec;
    RGYQueueMPMP<RGYFrameDataMetadata*> m_queueHDR10plusMetadata;
    RGYQueueMPMP<FrameFlags> m_dataFlag;
    RGYRunState m_state;
    int m_decOutFrames;
    int64_t m_hwDecFirstPts;
    bool m_gotFrameAfterFirstPts; //最初のフレームより前のptsで出てきたフレームのカウント
#if THREAD_DEC_USE_FUTURE
    std::future<RGY_ERR> m_thDecoder;
#else
    std::thread m_thDecoder;
#endif //#if THREAD_DEC_USE_FUTURE
    FrameReleaseData<cudaEvent_t> *m_frameReleaseData;
public:
    PipelineTaskNVDecode(NVGPUInfo *dev, CuvidDecode *dec, int outMaxQueueSize, RGYInput *input, int64_t endPts, RGYParamThread threadParam, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::NVDEC, dev, outMaxQueueSize, false, threadParam, log), m_input(input), m_endPts(endPts), m_dec(dec),
        m_queueHDR10plusMetadata(), m_dataFlag(),
        m_state(RGY_STATE_STOPPED), m_decOutFrames(0), m_hwDecFirstPts(AV_NOPTS_VALUE), m_gotFrameAfterFirstPts(false), m_thDecoder(), m_frameReleaseData(nullptr) {
        m_queueHDR10plusMetadata.init(256);
        m_dataFlag.init();
    };
    virtual ~PipelineTaskNVDecode() {
        m_state = RGY_STATE_ABORT;
        closeThread();
        m_queueHDR10plusMetadata.close([](RGYFrameDataMetadata **ptr) { if (*ptr) { delete *ptr; *ptr = nullptr; }; });
    };
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("waitForQueueUpdate"), _T("dequeue"), _T("DecodeFrameAsync"), _T("PushQueue") },
            std::vector<tstring>{_T("")}
        );
    }
    void setDec(CuvidDecode *dec) { m_dec = dec; };

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override {
        const auto inputFrameInfo = m_input->GetInputFrameInfo();
        RGYFrameInfo info(inputFrameInfo.srcWidth, inputFrameInfo.srcHeight, inputFrameInfo.csp, inputFrameInfo.bitdepth, inputFrameInfo.picstruct, RGY_MEM_TYPE_GPU);
        return std::make_pair(info, 0);
    };

    void setVppFrameReleaseData(FrameReleaseData<cudaEvent_t> *frameReleaseData) {
        m_frameReleaseData = frameReleaseData;
    }

    void closeThread() {
        PrintMes(RGY_LOG_DEBUG, _T("Flushing Decoder\n"));
#if THREAD_DEC_USE_FUTURE
        if (m_thDecoder.valid()) {
#else
        if (m_thDecoder.joinable()) {
#endif
            //エンコード中断時の処理
            //ここでフレームをすべて吐き出し切らないと、中断時にデコードスレッドが終了しない
            while (!m_dec->GetError()
                && !(m_dec->frameQueue()->isEndOfDecode() && m_dec->frameQueue()->isEmpty())) {
                m_dec->frameQueue()->endDecode(); //デコーダの待機ループから強制的に出る
                CUVIDPARSERDISPINFO pInfo;
                if (m_dec->frameQueue()->dequeue(&pInfo)) {
                    m_dec->frameQueue()->releaseFrame(&pInfo);
                }
                if (m_frameReleaseData) {
                    m_frameReleaseData->waitFrameSingleThread(0);
                    m_workSurfs.deleteFreedSurface(); // これを呼ばないとフレームが解放されず、デコードが止まってしまうことがある
                }
            }
#if THREAD_DEC_USE_FUTURE
            while (m_thDecoder.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
#else
            while (m_thDecoder.native_handle() && RGYThreadStillActive(m_thDecoder.native_handle())) {
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
#if THREAD_DEC_USE_FUTURE
        m_thDecoder = std::async(std::launch::async, [this]() {
#else
        m_thDecoder = std::thread([this]() {
#endif //#if THREAD_DEC_USE_FUTURE
            CUresult curesult = CUDA_SUCCESS;
            RGYBitstream bitstream = RGYBitstreamInit();
            m_threadParam.apply(GetCurrentThread());
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
        return getOutputFrame();
    }
    PipelineTaskSurface addTaskSurface(std::unique_ptr<CUFrameCuvid>& surf) {
        return m_workSurfs.addSurface(surf);
    }
    
protected:
    RGY_ERR getOutputFrame() {
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

        int istart = 0; // dispInfoListのうち、デコードを行う(m_outQeueueに追加する)フレームの開始idx
        std::vector<CUVIDPARSERDISPINFO> dispInfoList;
        while (m_state == RGY_STATE_RUNNING) {
            if (m_stopwatch) m_stopwatch->set(0);
            if (m_dec->GetError()) {
                m_state = RGY_STATE_ERROR;
                return RGY_ERR_UNKNOWN;
            }
            if (m_dec->frameQueue()->isEndOfDecode() && m_dec->frameQueue()->isEmpty()) {
                m_state = RGY_STATE_EOF;
                return RGY_ERR_MORE_BITSTREAM;
            }

            auto dispInfo = CUVIDPARSERDISPINFO{ 0 };
            if (!m_dec->frameQueue()->dequeue(&dispInfo)) {
                //転送の終了状況を確認、可能ならリソースの開放を行う
                if (m_frameReleaseData) {
                    m_frameReleaseData->waitFrameSingleThread(0);
                    m_workSurfs.deleteFreedSurface(); // これを呼ばないとフレームが解放されず、デコードが止まってしまうことがある
                }
                m_dec->frameQueue()->waitForQueueUpdate();
#if THREAD_DEC_USE_FUTURE
                if (m_thDecoder.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
#else
                if (m_thDecoder.native_handle() && !RGYThreadStillActive(m_thDecoder.native_handle())) {
#endif
                    PrintMes(RGY_LOG_ERROR, _T("Decode thread is not responding.\n"));
                    m_state = RGY_STATE_ERROR;
                    return RGY_ERR_UNKNOWN;
                }
                if (m_stopwatch) m_stopwatch->add(0, 0);
                continue;
            }
            //cuvidのtimestampはかならず分子が1になっているのでもとに戻す
            auto cuvidTimebase = rgy_rational<int>(1, m_input->getInputTimebase().d());
            dispInfo.timestamp = rational_rescale(dispInfo.timestamp, cuvidTimebase, m_input->getInputTimebase());
            PrintMes(RGY_LOG_TRACE, _T("input frame (dev) #%d, pic_idx %d, timestamp %lld\n"), m_decOutFrames, dispInfo.picture_index, dispInfo.timestamp);

            if (m_endPts >= 0
                && dispInfo.timestamp >= m_endPts) { // m_endPtsは含まないようにする(重要)
                m_state = RGY_STATE_EOF;
                return RGY_ERR_MORE_BITSTREAM;
            }
            dispInfoList.push_back(dispInfo);
            // 一度でもフレームが出ている場合は、それ以降のフレームはチェックをskipする (wrap対策)
            if (m_gotFrameAfterFirstPts) {
                break;
            }
            // OpenGOP等でキーフレームより前のフレームのptsで出てくることがあるのを調整
            // 実際に前のフレームが出ているのではなく、前のフレームのptsで出てきているだけの場合(パターンA[例: "Beauty_3840x2160_120fps_420_8bit_HEVC_MP4.mp4"の--seek 6.66667])もあるので注意が必要
            // そうでなくキーフレームより前のフレームがたくさん出てきてしまう場合もある(パターンB [例: "720p - AVC - MP2 2.0 - ZDF HD.ts"の先頭から])
            if (dispInfo.timestamp >= m_hwDecFirstPts || m_hwDecFirstPts == AV_NOPTS_VALUE) {
                if (dispInfoList.size() > 1) {
                    // 最終フレームがFirstPtsに一致していたらそこからデコード
                    // 最終フレームがFirstPtsを超えていたらそのひとつ前からデコード
                    const bool lastFrameIsOverFirstPts = dispInfo.timestamp > m_hwDecFirstPts;
                    const int targetStart = (int)dispInfoList.size() - (lastFrameIsOverFirstPts ? 2 : 1);
                    for (; istart < targetStart; istart++) {
                        m_dec->frameQueue()->releaseFrame(&dispInfoList[istart]);
                    }
                    if (targetStart >= 0 && lastFrameIsOverFirstPts) {
                        // パターンAなので、最初のフレームのtimestampを修正する
                        dispInfoList[targetStart].timestamp = m_hwDecFirstPts;
                    }
                }
                m_gotFrameAfterFirstPts = true;
                break;
            }
            // m_hwDecFirstPtsより前のフレームがたくさん出てきてしまうことがある
            // m_hwDecFirstPtsより前のフレームはdropするしかない (そうしないとデコードがフレームバッファ不足で止まってしまう)
            // 最後のフレームは、m_hwDecFirstPtsが出てこない場合に備えて残しておく
            for (; istart < (int)dispInfoList.size() - 1; istart++) {
                m_dec->frameQueue()->releaseFrame(&dispInfoList[istart]);
            }
            if (m_stopwatch) m_stopwatch->add(0, 1);
        }
        if (m_stopwatch) m_stopwatch->set(0);
        for (; istart < (int)dispInfoList.size(); istart++) {
            const auto& dispInfo = dispInfoList[istart];
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
        }
        if (m_stopwatch) m_stopwatch->add(0, 2);
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

class PipelineTaskAudio : public PipelineTask {
protected:
    RGYInput *m_input;
    std::map<int, std::shared_ptr<RGYOutputAvcodec>> m_pWriterForAudioStreams;
    std::map<int, NVEncFilter *> m_filterForStreams;
    std::vector<std::shared_ptr<RGYInput>> m_audioReaders;
public:
    PipelineTaskAudio(NVGPUInfo *dev, RGYInput *input, std::vector<std::shared_ptr<RGYInput>>& audioReaders, std::vector<std::shared_ptr<RGYOutput>>& fileWriterListAudio, std::vector<VppVilterBlock>& vpFilters, int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::AUDIO, dev, outMaxQueueSize, false, threadParam, log),
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
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }

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
        if (m_stopwatch) m_stopwatch->set(0);
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
        if (m_stopwatch) m_stopwatch->add(0, 0);
        return RGY_ERR_NONE;
    }
};

class PipelineTaskParallelEncBitstream : public PipelineTask {
protected:
    RGYInput *m_input;
    int m_currentChunk; // いま並列処理の何番目を処理中か
    RGYTimestamp *m_encTimestamp;
    RGYTimecode *m_timecode;
    RGYParallelEnc *m_parallelEnc;
    EncodeStatus *m_encStatus;
    rgy_rational<int> m_encFps;
    rgy_rational<int> m_outputTimebase;
    std::unique_ptr<PipelineTaskAudio> m_taskAudio;
    std::unique_ptr<FILE, fp_deleter> m_fReader;
    int64_t m_firstPts; //最初のpts
    int64_t m_maxPts; // 最後のpts
    int64_t m_ptsOffset; // 分割出力間の(2分割目以降の)ptsのオフセット
    int64_t m_encFrameOffset; // 分割出力間の(2分割目以降の)エンコードフレームのオフセット
    int64_t m_inputFrameOffset; // 分割出力間の(2分割目以降の)エンコードフレームのオフセット
    int64_t m_maxEncFrameIdx; // 最後にエンコードしたフレームのindex
    int64_t m_maxInputFrameIdx; // 最後にエンコードしたフレームのindex
    RGYBitstream m_decInputBitstream; // 映像読み込み (ダミー)
    bool m_inputBitstreamEOF; // 映像側の読み込み終了フラグ (音声処理の終了も確認する必要があるため)
    RGYListRef<RGYBitstream> m_bitStreamOut;
    RGYDurationCheck m_durationCheck;
    bool m_tsDebug;
public:
    PipelineTaskParallelEncBitstream(NVGPUInfo *dev, RGYInput *input, RGYTimestamp *encTimestamp, RGYTimecode *timecode, RGYParallelEnc *parallelEnc, EncodeStatus *encStatus,
        rgy_rational<int> encFps, rgy_rational<int> outputTimebase,
        std::unique_ptr<PipelineTaskAudio>& taskAudio, int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::PECOLLECT, dev, outMaxQueueSize, false, threadParam, log),
        m_input(input), m_currentChunk(-1), m_encTimestamp(encTimestamp), m_timecode(timecode),
        m_parallelEnc(parallelEnc), m_encStatus(encStatus), m_encFps(encFps), m_outputTimebase(outputTimebase),
        m_taskAudio(std::move(taskAudio)), m_fReader(std::unique_ptr<FILE, fp_deleter>(nullptr, fp_deleter())),
        m_firstPts(-1), m_maxPts(-1), m_ptsOffset(0), m_encFrameOffset(0), m_inputFrameOffset(0), m_maxEncFrameIdx(-1), m_maxInputFrameIdx(-1),
        m_decInputBitstream(), m_inputBitstreamEOF(false), m_bitStreamOut(), m_durationCheck(), m_tsDebug(false) {
        m_decInputBitstream.init(AVCODEC_READER_INPUT_BUF_SIZE);
        auto reader = dynamic_cast<RGYInputAvcodec*>(input);
        if (reader) {
            // 親側で不要なデコーダを終了させる、こうしないとavsw使用時に映像が無駄にデコードされてしまう
            reader->CloseVideoDecoder();
        }
        m_logType = RGY_LOGT_CORE_PARALLEL;
    };
    virtual ~PipelineTaskParallelEncBitstream() {
        m_decInputBitstream.clear();
    };

    virtual bool isPassThrough() const override { return true; }

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };
protected:
    RGY_ERR checkEncodeResult() {
        // まずそのエンコーダの終了を待機
        while (m_parallelEnc->waitProcessFinished(m_currentChunk, UPDATE_INTERVAL) != WAIT_OBJECT_0) {
            // 進捗表示の更新
            auto currentData = m_encStatus->GetEncodeData();
            m_encStatus->UpdateDisplay(currentData.progressPercent);
        }
        // 戻り値を確認
        auto procsts = m_parallelEnc->processReturnCode(m_currentChunk);
        if (!procsts.has_value()) { // そんなはずはないのだが、一応
            PrintMes(RGY_LOG_ERROR, _T("Unknown error in parallel enc: %d.\n"), m_currentChunk);
            return RGY_ERR_UNKNOWN;
        }
        if (procsts.value() != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts.value()));
            return procsts.value();
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR openNextFile() {
        if (m_currentChunk >= 0 && m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::Mem) {
            // メモリモードの場合は、まだそのエンコーダの戻り値をチェックしていないので、ここでチェック
            auto procsts = checkEncodeResult();
            if (procsts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts));
                return procsts;
            }
        }

        m_currentChunk++;
        if (m_currentChunk >= (int)m_parallelEnc->parallelCount()) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        
        if (m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::File) {
            // 戻り値を確認
            auto procsts = checkEncodeResult();
            if (procsts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts));
                return procsts;
            }
            // ファイルを開く
            auto tmpPath = m_parallelEnc->tmpPath(m_currentChunk);
            if (tmpPath.empty()) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get tmp path for parallel enc %d.\n"), m_currentChunk);
                return RGY_ERR_UNKNOWN;
            }
            {
                FILE *fp = nullptr;
                if (_tfopen_s(&fp, tmpPath.c_str(), _T("rb")) != 0 || fp == nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to open file: %s\n"), tmpPath.c_str());
                    return RGY_ERR_FILE_OPEN;
                }
                m_fReader = std::unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
            }
        }
        //最初のファイルに対するptsの差を取り、それをtimebaseを変換して適用する
        const auto inputFrameInfo = m_input->GetInputFrameInfo();
        const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
        const auto srcTimebase = (m_input->getInputTimebase().n() > 0 && m_input->getInputTimebase().is_valid()) ? m_input->getInputTimebase() : inputFpsTimebase;
        // seek結果による入力ptsを用いて計算した本来のpts offset
        const auto ptsOffsetOrig = (m_firstPts < 0) ? 0 : rational_rescale(m_parallelEnc->getVideofirstKeyPts(m_currentChunk), srcTimebase, m_outputTimebase) - m_firstPts;
        // 直前のフレームから計算したpts offset(-1フレーム分) 最低でもこれ以上のoffsetがないといけない
        const auto ptsOffsetMax = (m_firstPts < 0) ? 0 : m_maxPts - m_firstPts;
        // フレームの長さを決める
        int64_t lastDuration = 0;
        const auto frameDuration = m_durationCheck.getDuration(lastDuration);
        // frameDuration のうち、登場回数が最も多いものを探す
        int mostFrequentDuration = 0;
        int64_t mostFrequentDurationCount = 0;
        int64_t totalFrameCount = 0;
        for (const auto& [duration, count] : frameDuration) {
            if (count > mostFrequentDurationCount) {
                mostFrequentDuration = duration;
                mostFrequentDurationCount = count;
            }
            totalFrameCount += count;
        }
        // フレーム長が1つしかない場合、あるいは登場頻度の高いフレーム長がある場合、そのフレーム長を採用する
        if (frameDuration.size() == 1 || ((totalFrameCount * 9 / 10) < mostFrequentDurationCount)) {
            m_ptsOffset = ptsOffsetMax + mostFrequentDuration;
        } else if (frameDuration.size() == 2) {
            if ((totalFrameCount * 7 / 10) < mostFrequentDurationCount || lastDuration != mostFrequentDuration) {
                m_ptsOffset = ptsOffsetMax + mostFrequentDuration;
            } else {
                int otherDuration = mostFrequentDuration;
                for (auto itr = frameDuration.begin(); itr != frameDuration.end(); itr++) {
                    if (itr->first != mostFrequentDuration) {
                        otherDuration = itr->first;
                        break;
                    }
                }
                m_ptsOffset = ptsOffsetMax + otherDuration;
            }
        } else {
            // ptsOffsetOrigが必要offsetの最小値(ptsOffsetMax)より大きく、そのずれが2フレーム以内ならそれを採用する
            // そうでなければ、ptsOffsetMaxに1フレーム分の時間を足した時刻にする
            m_ptsOffset = (m_firstPts < 0) ? 0 :
                ((ptsOffsetOrig - ptsOffsetMax > 0 && ptsOffsetOrig - ptsOffsetMax <= rational_rescale(2, m_encFps.inv(), m_outputTimebase))
                    ? ptsOffsetOrig : (ptsOffsetMax + rational_rescale(1, m_encFps.inv(), m_outputTimebase)));
        }
        m_encFrameOffset = (m_currentChunk > 0) ? m_maxEncFrameIdx + 1 : 0;
        m_inputFrameOffset = (m_currentChunk > 0) ? m_maxInputFrameIdx + 1 : 0;
        PrintMes(m_tsDebug ? RGY_LOG_ERROR : RGY_LOG_DEBUG, _T("Switch to next file: pts offset %lld, frame offset %d.\n")
            _T("  firstKeyPts 0: % lld, %d : % lld.\n")
            _T("  ptsOffsetOrig: %lld, ptsOffsetMax: %lld, m_maxPts: %lld\n"),
            m_ptsOffset, m_encFrameOffset,
            m_firstPts, m_currentChunk, rational_rescale(m_parallelEnc->getVideofirstKeyPts(m_currentChunk), srcTimebase, m_outputTimebase),
            ptsOffsetOrig, ptsOffsetMax, m_maxPts);
        return RGY_ERR_NONE;
    }

    void updateAndSetHeaderProperties(RGYBitstream *bsOut, RGYOutputRawPEExtHeader *header) {
        header->pts += m_ptsOffset;
        header->dts += m_ptsOffset;
        header->encodeFrameIdx += m_encFrameOffset;
        header->inputFrameIdx += m_inputFrameOffset;
        bsOut->setPts(header->pts);
        bsOut->setDts(header->dts);
        bsOut->setDuration(header->duration);
        bsOut->setFrametype(header->frameType);
        bsOut->setPicstruct(header->picstruct);
        bsOut->setFrameIdx(header->encodeFrameIdx);
        bsOut->setDataflag((RGY_FRAME_FLAGS)header->flags);
    }

    RGY_ERR getBitstreamOneFrameFromQueue(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        RGYOutputRawPEExtHeader *packet = nullptr;
        auto err = m_parallelEnc->getNextPacket(m_currentChunk, &packet);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (packet == nullptr) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        updateAndSetHeaderProperties(bsOut, packet);
        if (packet->size <= 0) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        } else {
            bsOut->resize(packet->size);
            memcpy(&header, packet, sizeof(header));
            memcpy(bsOut->data(), (void *)(packet + 1), packet->size);
        }
        // メモリを使いまわすため、使い終わったパケットを回収する
        m_parallelEnc->putFreePacket(m_currentChunk, packet);
        PrintMes(RGY_LOG_TRACE, _T("Q: pts %08lld, dts %08lld, size %d.\n"), bsOut->pts(), bsOut->dts(), bsOut->size());
        return RGY_ERR_NONE;
    }

    RGY_ERR getBitstreamOneFrameFromFile(FILE *fp, RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (header.size <= 0) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        updateAndSetHeaderProperties(bsOut, &header);
        bsOut->resize(header.size);
        PrintMes(RGY_LOG_TRACE, _T("F: pts %08lld, dts %08lld, size %d.\n"), bsOut->pts(), bsOut->dts(), bsOut->size());

        if (fread(bsOut->data(), 1, bsOut->size(), fp) != bsOut->size()) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR getBitstreamOneFrame(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        return (m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::File)
            ? getBitstreamOneFrameFromFile(m_fReader.get(), bsOut, header)
            : getBitstreamOneFrameFromQueue(bsOut, header);
    }

    virtual RGY_ERR getBitstream(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        if (m_currentChunk < 0) {
            if (auto err = openNextFile(); err != RGY_ERR_NONE) {
                return err;
            }
        } else if (m_currentChunk >= (int)m_parallelEnc->parallelCount()) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        auto err = getBitstreamOneFrame(bsOut, header);
        if (err == RGY_ERR_MORE_BITSTREAM) {
            if ((err = openNextFile()) != RGY_ERR_NONE) {
                return err;
            }
            err = getBitstreamOneFrame(bsOut, header);
        }
        return err;
    }
public:
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        m_inFrames++;
        auto ret = m_input->LoadNextFrame(nullptr); // 進捗表示用のダミー
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_DATA && ret != RGY_ERR_MORE_BITSTREAM) {
            PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(ret));
            return ret;
        }
        m_inputBitstreamEOF |= (ret == RGY_ERR_MORE_DATA || ret == RGY_ERR_MORE_BITSTREAM);

        // 音声等抽出のため、入力ファイルの読み込みを進める
        //この関数がMFX_ERR_NONE以外を返せば、入力ビットストリームは終了
        ret = m_input->GetNextBitstream(&m_decInputBitstream);
        m_inputBitstreamEOF |= (ret == RGY_ERR_MORE_DATA || ret == RGY_ERR_MORE_BITSTREAM);
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_DATA && ret != RGY_ERR_MORE_BITSTREAM) {
            PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(ret));
            return ret; //エラー
        }
        m_decInputBitstream.clear();

        if (m_taskAudio) {
            ret = m_taskAudio->extractAudio(m_inFrames);
            if (ret != RGY_ERR_NONE) {
                return ret;
            }
        }

        // 定期的に全スレッドでエラー終了したものがないかチェックする
        if ((m_inFrames & 15) == 0) {
            if ((ret = m_parallelEnc->checkAllProcessErrors()) != RGY_ERR_NONE) {
                return ret; //エラー
            }
        }

        auto bsOut = m_bitStreamOut.get([](RGYBitstream *bs) {
            *bs = RGYBitstreamInit();
            return 0;
            });
        if (!bsOut) {
            return RGY_ERR_NULL_PTR;
        }
        RGYOutputRawPEExtHeader header;
        ret = getBitstream(bsOut.get(), header);
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_BITSTREAM) {
            return ret;
        }
        if (ret == RGY_ERR_NONE && bsOut->size() > 0) {
            std::vector<std::shared_ptr<RGYFrameData>> metadatalist;
            const auto duration = (ENCODER_QSV) ? header.duration : bsOut->duration(); // QSVの場合、Bitstreamにdurationの値がないため、durationはheaderから取得する
            m_encTimestamp->add(bsOut->pts(), header.inputFrameIdx, header.encodeFrameIdx, duration, metadatalist);
            if (m_firstPts < 0) m_firstPts = bsOut->pts();
            m_maxPts = std::max(m_maxPts, bsOut->pts());
            m_maxEncFrameIdx = std::max(m_maxEncFrameIdx, header.encodeFrameIdx);
            m_maxInputFrameIdx = std::max(m_maxInputFrameIdx, header.inputFrameIdx);
            PrintMes(m_tsDebug ? RGY_LOG_ERROR : RGY_LOG_DEBUG, _T("Packet: pts %lld, dts: %lld, duration: %d, input idx: %lld, encode idx: %lld, size %lld.\n"), bsOut->pts(), bsOut->dts(), duration, header.inputFrameIdx, header.encodeFrameIdx, bsOut->size());
            if (m_timecode) {
                m_timecode->write(bsOut->pts(), m_outputTimebase);
            }
            m_durationCheck.add(bsOut->pts());
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputBitstream>(bsOut));
        }
        if (m_inputBitstreamEOF && ret == RGY_ERR_MORE_BITSTREAM && m_taskAudio) {
            m_taskAudio->flushAudio();
        }
        return (m_inputBitstreamEOF && ret == RGY_ERR_MORE_BITSTREAM) ? RGY_ERR_MORE_BITSTREAM : RGY_ERR_NONE;
    }
};

class PipelineTaskTrim : public PipelineTask {
protected:
    const sTrimParam &m_trimParam;
    RGYInput *m_input;
    RGYParallelEnc *m_parallelEnc;
    rgy_rational<int> m_srcTimebase;
    rgy_rational<int> m_outTimebase;
    int64_t m_trimTimestampOffset;
    int64_t m_lastTrimFramePts;
public:
    PipelineTaskTrim(NVGPUInfo *dev, const sTrimParam &trimParam, RGYInput *input, RGYParallelEnc *parallelEnc, const rgy_rational<int>& srcTimebase, const rgy_rational<int>& outTimebase, int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::TRIM, dev, outMaxQueueSize, false, threadParam, log),
        m_trimParam(trimParam), m_input(input), m_parallelEnc(parallelEnc), m_srcTimebase(srcTimebase), m_outTimebase(outTimebase), m_trimTimestampOffset(0), m_lastTrimFramePts(AV_NOPTS_VALUE) {};
    virtual ~PipelineTaskTrim() {};

    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }

    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    int64_t trimTimestampOffset() const { return m_trimTimestampOffset; }

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
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

        if (m_parallelEnc) {
            auto finKeyPts = m_parallelEnc->getVideoEndKeyPts();
            if (finKeyPts >= 0 && inputFramePts >= finKeyPts) {
                m_parallelEnc->setVideoFinished();
                return RGY_ERR_NONE;
            }
        }

        if (!m_input->checkTimeSeekTo(taskSurf->surf().frame()->timestamp(), m_srcTimebase)) {
            return RGY_ERR_NONE; //seektoにより脱落させるフレーム
        }
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_dev->vidCtxLock(), taskSurf->surf()));
        if (m_stopwatch) m_stopwatch->add(0, 0);
        return RGY_ERR_NONE;
    }
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
    const PipelineTaskTrim *m_taskTrim;
public:
    PipelineTaskCheckPTS(NVGPUInfo *dev, CuvidDecode *dec, PipelineTaskNVDecode *taskNVDec, const PipelineTaskTrim *taskTrim, rgy_rational<int> srcTimebase, rgy_rational<int> streamTimebase, rgy_rational<int> outputTimebase, int64_t outFrameDuration, RGYAVSync avsync, cudaVideoDeinterlaceMode deinterlaceMode,
        bool timestampPassThrough, bool vpp_rff, bool vpp_afs_rff_aware, bool interlaceAuto, FramePosList *framePosList, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CHECKPTS, dev, /*outMaxQueueSize = */ 0 /*常に0である必要がある*/, false, threadParam, log),
        m_srcTimebase(srcTimebase), m_streamTimebase(streamTimebase), m_outputTimebase(outputTimebase), m_avsync(avsync),
        m_timestampPassThrough(timestampPassThrough), m_vpp_rff(vpp_rff), m_vpp_afs_rff_aware(vpp_afs_rff_aware), m_interlaceAuto(interlaceAuto), m_deinterlaceMode(deinterlaceMode),
        m_outFrameDuration(outFrameDuration),
        m_tsOutFirst(-1), m_tsOutEstimated(0), m_tsPrev(-1), m_inputFramePosIdx(std::numeric_limits<decltype(m_inputFramePosIdx)>::max()), m_framePosList(framePosList), m_dec(dec), m_taskNVDec(taskNVDec), m_taskTrim(taskTrim) {
    };
    virtual ~PipelineTaskCheckPTS() {};

    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }

    virtual bool isPassThrough() const override {
        // そのまま渡すのでpaththrough
        return true;
    }
    static const int MAX_FORCECFR_INSERT_FRAMES = 1024; //事実上の無制限
public:
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
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
            const int64_t srcTimestamp = taskSurf->surf().frame()->timestamp();
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
                if (m_taskTrim) {
                    outPtsSource -= m_taskTrim->trimTimestampOffset();
                }
            }
        }
        PrintMes(RGY_LOG_TRACE, _T("check_pts(%d/%d): nOutEstimatedPts %lld, outPtsSource %lld, outDuration %d\n"), taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsOutEstimated, outPtsSource, outDuration);
        if (m_tsOutFirst < 0) {
            m_tsOutFirst = outPtsSource; //最初のpts
            PrintMes(RGY_LOG_DEBUG, _T("check_pts: m_tsOutFirst %lld\n"), outPtsSource);
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
        if (m_stopwatch) m_stopwatch->add(0, 0);
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

class PipelineTaskVideoQualityMetric : public PipelineTask {
private:
    NVEncFilterSsim *m_videoMetric;
    cudaStream_t m_stream;
    RGYListRef<cudaEvent_t> m_frameUseFinEvent;
public:
    PipelineTaskVideoQualityMetric(NVGPUInfo *dev, NVEncFilterSsim *videoMetric, int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::VIDEOMETRIC, dev, outMaxQueueSize, false, threadParam, log), m_videoMetric(videoMetric), m_stream(nullptr), m_frameUseFinEvent() {
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
        case NV_ENC_BUFFER_FORMAT_NV16:
            uInputWidthByte = uInputWidth;
            uInputHeightTotal = uInputHeight * 2;
            break;
        case NV_ENC_BUFFER_FORMAT_P210:
            uInputWidthByte = uInputWidth * 2;
            uInputHeightTotal = uInputHeight * 2;
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
            if (ENABLE_INTERLACE_FROM_HWMEM || picStruct == NV_ENC_PIC_STRUCT_FRAME) {
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
        for (size_t i = 0; i < m_stEncodeBuffer.size(); i++) {
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
    const RGYHDR10Plus *m_hdr10plus;
    const DOVIRpu *m_doviRpu;
    std::vector<NVEncRCParam>& m_dynamicRC;
    int m_appliedDynamicRC;
    std::vector<int>& m_keyFile;
    bool m_keyOnChapter;
    std::vector<std::unique_ptr<AVChapter>>& m_Chapters;
    std::thread m_threadOutput;
    std::promise<RGY_ERR> m_threadOutputPromise;
    std::future<RGY_ERR> m_threadOutputFuture;
    std::optional<RGY_ERR> m_threadOutputResult;
    bool m_threadOutputAbort;
    // Linux (ENABLE_ASYNC=0)の場合にデータの取り出しをマルチスレッドで行うと、
    // NvEncLockBitstreamがInvalid(8)を返して異常終了してしまう
    // そのため、スレッドを立てずに従来通りPipelineTaskCUDAVppの中でエンコーダに渡すフレームが足りなくなったときに
    // outputThreadFunc(true)を読んで出力するようにする
    static const bool m_bEnableOutputThread = ENABLE_ASYNC != 0;
public:
    PipelineTaskNVEncode(
        NVGPUInfo *dev, NVEncRunCtx *runCtx, RGY_CODEC encCodec, int encWidth, int encHeight, RGY_CSP encCsp, int encBitdepth, RGY_PICSTRUCT encPicStruct,
        const NV_ENC_CONFIG& stEncConfig, const NV_ENC_INITIALIZE_PARAMS& stCreateEncodeParams,
        RGYTimecode *timecode, RGYTimestamp *encTimestamp, rgy_rational<int> outputTimebase, const RGYHDR10Plus *hdr10plus, const DOVIRpu *doviRpu,
        std::vector<NVEncRCParam>& dynamicRC, std::vector<int>& keyFile, bool keyOnChapter, std::vector<std::unique_ptr<AVChapter>>& chapters,
         int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::NVENC, dev, outMaxQueueSize, m_bEnableOutputThread, threadParam, log),
        m_runCtx(runCtx), m_encCodec(encCodec), m_encWidth(encWidth), m_encHeight(encHeight), m_encCsp(encCsp), m_encBitdepth(encBitdepth), m_encPicStruct(encPicStruct),
        m_stEncConfig(stEncConfig), m_stCreateEncodeParams(stCreateEncodeParams),
        m_timecode(timecode), m_encTimestamp(encTimestamp), m_outputTimebase(outputTimebase),
        m_bitStreamOut(), m_hdr10plus(hdr10plus), m_doviRpu(doviRpu), m_dynamicRC(dynamicRC), m_appliedDynamicRC(-1), m_keyFile(keyFile), m_keyOnChapter(keyOnChapter), m_Chapters(chapters),
        m_threadOutput(), m_threadOutputPromise(), m_threadOutputFuture(), m_threadOutputResult(), m_threadOutputAbort(false) {
        runThreadOutput();
    };
    virtual ~PipelineTaskNVEncode() {
        flushEncoder();
        if (m_bEnableOutputThread) {
            m_threadOutputAbort = true;
            getOutputThreadResult(30 * 1000);
            if (m_threadOutput.joinable()) {
                m_threadOutput.join();
            }
        }
        m_outQeueue.clear(); // m_bitStreamOutが解放されるより前にこちらを解放する
    };
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("streamWaitCuEvents"), _T("encodeFrame"), _T("flushEncoder") },
            std::vector<tstring>{_T("")}
        );
    }

    bool useOutputThread() const {
        return m_bEnableOutputThread;
    }

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override {
        RGYFrameInfo info(m_encWidth, m_encHeight, m_encCsp, m_encBitdepth, m_encPicStruct, RGY_MEM_TYPE_GPU);
        return std::make_pair(info, m_outMaxQueueSize);
    }
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override { return std::nullopt; };

    std::optional<RGY_ERR> getOutputThreadResult(int timeout) {
        if (!m_bEnableOutputThread || m_threadOutputResult.has_value()) return m_threadOutputResult;
        const auto status = m_threadOutputFuture.wait_for(std::chrono::milliseconds(timeout));
        if (status == std::future_status::ready) {
            m_threadOutputResult = m_threadOutputFuture.get();
        } else if (m_threadOutput.native_handle()) {
            const bool threadActive = RGYThreadStillActive(m_threadOutput.native_handle());
            if (!threadActive) {
                m_threadOutputResult = RGY_ERR_UNKNOWN;
            }
        }
        return m_threadOutputResult;
    }
protected:
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
public:
    RGY_ERR outputThreadFunc(const bool getOneFrame) {
        while (!m_threadOutputAbort) {
            struct CUFrameEncAutoDelete {
                RGYQueueMPMP<CUFrameEnc *>& qEncodeBufferFree;
                CUFrameEncAutoDelete(RGYQueueMPMP<CUFrameEnc *>& q) : qEncodeBufferFree(q) {};;
                void operator()(CUFrameEnc* p) { if (p) qEncodeBufferFree.push(p); }
            };
            std::unique_ptr<CUFrameEnc, CUFrameEncAutoDelete> frameEnc(nullptr, CUFrameEncAutoDelete(m_runCtx->qEncodeBufferFree()));
            {
                CUFrameEnc *frameEncPtr = nullptr;
                while (!m_runCtx->qEncodeBufferUsed().front_copy_and_pop_no_lock(&frameEncPtr)) {
                    if (getOneFrame) {
                        return RGY_ERR_NONE;
                    }
                    m_runCtx->qEncodeBufferUsed().wait_for_push(); // 最大16ms待機
                    if (m_threadOutputAbort) {
                        return RGY_ERR_ABORTED;
                    }
                }
                if (!frameEncPtr) {
                    continue;
                }
                frameEnc.reset(frameEncPtr);
            }
            auto outBs = getOutputBitstream(frameEnc->encBuffer());
            if (outBs.first != RGY_ERR_NONE) {
                if (outBs.first == RGY_ERR_MORE_DATA) {
                    PrintMes(RGY_LOG_DEBUG, _T("Output thread reached EOS.\n"));
                } else {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get output bitstream: %s.\n"), get_err_mes(outBs.first));
                }
                return outBs.first;
            }
            frameEnc.reset();
            {
                // m_outQeueueへのロックが必要ならロックを取得
                std::optional<std::lock_guard<std::mutex>> lock;
                if (m_outQeueueMtx) {
                    lock.emplace(*m_outQeueueMtx);
                }
                m_outQeueue.push_back(std::make_unique<PipelineTaskOutputBitstream>(outBs.second));
            }
            if (getOneFrame) {
                return RGY_ERR_NONE;
            }
        }
        return RGY_ERR_NONE;
    }
protected:
    RGY_ERR runThreadOutput() {
        if (!m_bEnableOutputThread) return RGY_ERR_NONE;
        m_threadOutputFuture = m_threadOutputPromise.get_future();
        m_threadOutput = std::thread([this]() {
            auto err = RGY_ERR_NONE;
            m_threadParam.apply(GetCurrentThread());
            try {
                err = outputThreadFunc(false);
            } catch (const std::exception &e) {
                PrintMes(RGY_LOG_ERROR, _T("Output thread failed: %s.\n"), e.what());
                err = RGY_ERR_UNKNOWN;
            } catch (...) {
                PrintMes(RGY_LOG_ERROR, _T("Output thread failed.\n"));
                err = RGY_ERR_UNKNOWN;
            }
            if (err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_BITSTREAM) {
                err = RGY_ERR_NONE;
            }
            m_threadOutputPromise.set_value(err);
            PrintMes(err == RGY_ERR_NONE ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("Output thread finished: %s.\n"), get_err_mes(err));
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
                const int end = (m_dynamicRC[i].end < 0) ? std::numeric_limits<decltype(id)>::max() : m_dynamicRC[i].end;
                if (m_dynamicRC[i].start <= id && id <= end) {
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

        if (m_bEnableOutputThread) {
            // flushしたらそれを受けて出力スレッドが終了するのを待つ
            getOutputThreadResult(300 * 1000);
        } else {
            // 出力スレッドがない場合は、自分で出力処理を行う必要がある
            sts = outputThreadFunc(false);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return RGY_ERR_MORE_DATA;
    }
public:
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (frame && frame->type() != PipelineTaskOutputType::SURFACE) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (auto err = getOutputThreadResult(0); err.has_value()) {
            return err.value();
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
            if (m_stopwatch) m_stopwatch->add(0, 0);
        }
        
        if (surfEncodeIn) {
            auto sts = encodeFrame(surfEncodeIn->encBuffer(), m_inFrames++, surfEncodeIn->timestamp(), surfEncodeIn->duration(), surfEncodeIn->inputFrameId(), surfEncodeIn->dataList());
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to encode frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
            surfEncodeIn->clearDataList();

            m_runCtx->qEncodeBufferUsed().push(surfEncodeIn);
            if (m_stopwatch) m_stopwatch->add(0, 1);
        } else {
            // フレームがない場合は、エンコーダのキューをフラッシュ
            auto sts = flushEncoder();
            if (sts != RGY_ERR_NONE) {
                if (sts != RGY_ERR_MORE_DATA) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to flush encoder queue: %s.\n"), get_err_mes(sts));
                }
                return sts;
            }
            if (m_stopwatch) m_stopwatch->add(0, 2);
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
    int m_cudaStreamOpt;
    int m_cudaMT;
    cudaEvent_t m_eventDefaultToFilter;
    cudaStream_t m_streamFilter;
    cudaStream_t m_streamDownload;
    std::unique_ptr<PipelineTaskOutput> m_cuvidPrev;
    PipelineTaskNVEncode *m_encode;
public:
    PipelineTaskCUDAVpp(NVGPUInfo *dev, std::vector<std::unique_ptr<NVEncFilter>>& vppfilters, NVEncFilterSsim *videoMetric,
    RGYQueueMPMP<CUFrameEnc *>& qEncodeBufferFree, bool rgbAsYUV444, int cudaStreamOpt, int cudaMT, int outMaxQueueSize, RGYParamThread threadParam, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CUDA, dev, outMaxQueueSize, false, threadParam, log), m_vpFilters(vppfilters), m_videoMetric(videoMetric),
        m_frameReleaseData(dev->vidCtxLock(), 4, threadParam), m_inFrameUseFinEvent(), m_qEncodeBufferFree(qEncodeBufferFree), m_rgbAsYUV444(rgbAsYUV444),
        m_cudaStreamOpt(cudaStreamOpt), m_cudaMT(cudaMT), m_eventDefaultToFilter(nullptr),m_streamFilter(nullptr), m_streamDownload(nullptr), m_cuvidPrev(), m_encode(nullptr) {

        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
        if (cudaStreamOpt > 1) {
            auto ret = cudaStreamCreateWithFlags(&m_streamFilter, cudaStreamNonBlocking);
            if (ret != cudaSuccess) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to create upload stream: %s.\n"), char_to_tstring(cudaGetErrorString(ret)).c_str());
            }
        } else {
            m_streamFilter = cudaStreamPerThread;
        }
        auto ret = cudaStreamCreateWithFlags(&m_streamDownload, cudaStreamNonBlocking);
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
        if (m_streamFilter != cudaStreamPerThread) {
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
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("waitDepends"), _T("filter"), _T("getEncodeBuffer"), _T("filter2") },
            std::vector<tstring>{_T("")}
        );
    }

    virtual void printStatus() override {
        PrintMes(RGY_LOG_INFO, _T("in %d, out %d, outQeueue size: %d, frame release: %d.\n"),
            m_inFrames, m_outFrames, getOutQueueFrames(), m_frameReleaseData.queueSize());
    }

    void setEncodeTask(PipelineTaskNVEncode *encode) {
        m_encode = encode;
    }

    void setVideoQualityMetricFilter(NVEncFilterSsim *videoMetric) {
        m_videoMetric = videoMetric;
    }

    FrameReleaseData<cudaEvent_t> *cuvidFrameReleaseData() {
        return &m_frameReleaseData;
    }

    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfIn() override {
        auto firstFilterFrame = m_vpFilters.front()->GetFilterParam()->frameIn;
        return std::make_pair(firstFilterFrame, m_outMaxQueueSize + ((m_cudaMT) ? m_frameReleaseData.queueSizeMax() : 0));
    };
    virtual std::optional<std::pair<RGYFrameInfo, int>> requiredSurfOut() override {
        auto lastFilterFrame = m_vpFilters.back()->GetFilterParam()->frameOut;
        return std::make_pair(lastFilterFrame, m_outMaxQueueSize);
    };

    virtual void runFrameReleaseThread() {
        if (m_cudaMT) {
            m_frameReleaseData.start();
        }
    }

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
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
            // cudaをマルチスレッドで使用しない場合(cudaMT=0)は、ここで待機する (cudaMTが1のときはこれはなにもしない)
            m_frameReleaseData.waitFrameSingleThread(0);
            if (auto surfVppInCuvid = taskSurf->surf().cuvid(); surfVppInCuvid != nullptr) {
                // cuvidでは、cuvidのmap/unmapが同時に多重にできないので、まず前のフレームを解放を待つ (cudaMTがtrueのとき)
                m_frameReleaseData.waitUntilEmptyMultiThread();
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
        if (m_stopwatch) m_stopwatch->add(0, 0);
        while (filterframes.size() > 0 || drain) {
            if (m_stopwatch) m_stopwatch->set(0);
            //フィルタリングするならここ
            for (uint32_t ifilter = filterframes.front().second; ifilter < m_vpFilters.size() - 1; ifilter++) {
                // コピーを作ってそれをfilter関数に渡す
                // vpp-rffなどoverwirteするフィルタのときに、filterframes.pop_front -> push がうまく動作しない
                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                RGYFrameInfo input = filterframes.front().first;
                {
                    NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                    auto sts_filter = m_vpFilters[ifilter]->filter(&input, (RGYFrameInfo **)&outInfo, &nOutFrames, streamFilter);
                    if (sts_filter != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_vpFilters[ifilter]->name().c_str());
                        return sts_filter;
                    }
                }
                if (nOutFrames == 0) {
                    if (drain) {
                        filterframes.front().second++;
                        continue;
                    }
                    return RGY_ERR_NONE;
                }
                if (ifilter == 0) { //最初のフィルタなら転送なので、イベントをここでセットする
                    std::shared_ptr<cudaEvent_t> cudaEvent;
                    {
                        NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
                        cudaEvent = m_inFrameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
                        if (!cudaEvent) {
                            PrintMes(RGY_LOG_ERROR, _T("Failed to get cuda event.\n"));
                            return RGY_ERR_UNKNOWN;
                        }
                        cudaEventRecord(*cudaEvent, streamFilter);
                    }
                    //ここでinput frameの参照を m_prevInputFrame で保持するようにして、CUDAによるフレームの処理が完了しているかを確認できるようにする
                    //これを行わないとこのフレームが再度使われてしまうことになる
                    //NVEncCtxAutoLockは内部で行われるため不要
                    m_frameReleaseData.addFrame(frame, cudaEvent);
                }
                // cuvidとの同期のため、cudaStreamPerThreadを最初に使った場合でも、その次のフィルタはm_streamFilterを使用するようにする
                if (streamFilter != m_streamFilter) {
                    NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
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
            if (m_stopwatch) m_stopwatch->add(0, 1);
            if (drain) {
                return RGY_ERR_MORE_DATA; //最後までdrain = trueなら、drain完了
            }
            struct CUFrameEncAutoDelete {
                RGYQueueMPMP<CUFrameEnc *>& qEncodeBufferFree;
                CUFrameEncAutoDelete(RGYQueueMPMP<CUFrameEnc *>& q) : qEncodeBufferFree(q) {};;
                void operator()(CUFrameEnc* p) { if (p) qEncodeBufferFree.push(p); }
            };
            std::unique_ptr<CUFrameEnc, CUFrameEncAutoDelete> encBuffer(nullptr, CUFrameEncAutoDelete(m_qEncodeBufferFree));
            PipelineTaskSurface frameVppOut; 
            if (m_dev->encoder()) {
                { // 使用していないエンコードバッファを取得
                    CUFrameEnc *encBufferPtr = nullptr;
                    while (!m_qEncodeBufferFree.front_copy_and_pop_no_lock(&encBufferPtr)) {
                        if (m_encode) {
                            if (m_encode->useOutputThread()) {
                                m_qEncodeBufferFree.wait_for_push(); // 最大16ms待機
                                // エンコーダの出力スレッドがここで終了してしまっているのは想定外なのでエラー終了
                                // ここで検知しておかないとずっとここで待ち続けてしまう
                                if (auto err = m_encode->getOutputThreadResult(0); err.has_value()) {
                                    return err.value() == RGY_ERR_NONE ? RGY_ERR_ABORTED : err.value();
                                }
                            } else {
                                // 出力スレッドが有効でない場合、ここで出力を行う
                                if (auto err = m_encode->outputThreadFunc(true); err != RGY_ERR_NONE) {
                                    return err;
                                }
                            }
                        }
                    }
                    encBuffer.reset(encBufferPtr);
                }
                frameVppOut = m_workSurfs.get(encBuffer.get());

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
            if (m_stopwatch) m_stopwatch->add(0, 2);

            //エンコードバッファにコピー
            auto &lastFilter = m_vpFilters[m_vpFilters.size() - 1];
            //最後のフィルタはNVEncFilterCspCropでなければならない
            if (typeid(*lastFilter.get()) != typeid(NVEncFilterCspCrop)) {
                PrintMes(RGY_LOG_ERROR, _T("Last filter setting invalid.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
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
                    NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
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
            std::shared_ptr<cudaEvent_t> cudaEvent;
            {
                NVEncCtxAutoLock(ctxlock(m_dev->vidCtxLock()));
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
                cudaEvent = m_inFrameUseFinEvent.get([](cudaEvent_t *event) { return cudaEventCreateWithFlags(event, cudaEventDefault) != cudaSuccess ? 1 : 0; });
                if (!cudaEvent) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get filter finish cuda event.\n"));
                    return RGY_ERR_UNKNOWN;
                }
                auto err = cudaEventRecord(*cudaEvent, streamLastFilter);
                if (err != cudaSuccess) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to record filter finish cuda event.\n"));
                    return err_to_rgy(err);
                }
            }
            if (frame) {
                //ここでinput frameの参照を m_prevInputFrame で保持するようにして、CUDAによるフレームの処理が完了しているかを確認できるようにする
                //これを行わないとこのフレームが再度使われてしまうことになる
                //NVEncCtxAutoLockは内部で行われるため不要
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
            m_outQeueue.push_back(std::move(outputSurf));
            encBuffer.release();
            if (m_stopwatch) m_stopwatch->add(0, 3);
        }
        return RGY_ERR_NONE;
    }
};

#endif //__NVENC_PIPELINE_H__
