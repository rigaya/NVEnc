// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterRtgmcBob.h"
#include "NVEncFilterRtgmcCommon.h"
#include "NVEncFilterRtgmcSearchPrefilter.h"
#include "NVEncFilterRtgmcEdi.h"
#include "NVEncFilterDenoiseNLMeans.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "NVEncFilterDegrain.h"
#include "NVEncFilterRtgmcRetouch.h"
#include "NVEncFilterRtgmcShimmerRepair.h"
#include "NVEncFilterRtgmcLossless.h"
#include "NVEncFilterRtgmcPrimitive.h"
#if __has_include("NVEncFilterRtgmcMMask.h")
#include "NVEncFilterRtgmcMMask.h"
#define RGY_HAS_RTGMC_MMASK_FILTER 1
#endif
#include "rgy_prm.h"

#ifndef RGY_HAS_RTGMC_MMASK_FILTER
#define RGY_HAS_RTGMC_MMASK_FILTER 0
#endif

class NVRtgmcSharedFramePool {
public:
    NVRtgmcSharedFramePool(const char *allocStatsTag = "RTGMC shared frame pool");
    std::shared_ptr<CUFrameBuf> acquire(const RGYFrameInfo *frame);
    void clear();
private:
    std::vector<std::shared_ptr<CUFrameBuf>> m_pool;
    const char *m_allocStatsTag;
};

class NVEncFilterParamRtgmc : public NVEncFilterParam {
public:
    VppRtgmc rtgmc;
    rgy_rational<int> timebase;
    bool sharedAnalysisMode;

    NVEncFilterParamRtgmc() : rtgmc(), timebase(), sharedAnalysisMode(false) {}
    virtual ~NVEncFilterParamRtgmc() {}
    virtual tstring print() const override { return rtgmc.print(); }
};

class NVEncFilterRtgmc : public NVEncFilter {
public:
    NVEncFilterRtgmc();
    virtual ~NVEncFilterRtgmc();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    using NVEncFilter::filter;
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    bool draining() const;
    bool drainComplete() const;

protected:
    struct RtgmcFrameKey {
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;

        RtgmcFrameKey() : inputFrameId(-1), timestamp(0), duration(0) {}
        explicit RtgmcFrameKey(const RGYFrameInfo *frame) :
            inputFrameId(frame ? frame->inputFrameId : -1),
            timestamp(frame ? frame->timestamp : 0),
            duration(frame ? frame->duration : 0) {}
        bool matches(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp
                && duration == frame->duration;
        }
        bool matchesFrameIdentity(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp;
        }
    };

    struct RtgmcPendingCompRef {
        RtgmcFrameKey key;
        std::shared_ptr<RGYFrameData> backward;
        std::shared_ptr<RGYFrameData> forward;
        RGYCudaEvent backwardEvent;
        RGYCudaEvent forwardEvent;
        bool hasInlineParams;
        std::array<RGYDegrainCompensateInlineParams, 3> backwardInlineParams;
        std::array<RGYDegrainCompensateInlineParams, 3> forwardInlineParams;

        RtgmcPendingCompRef() :
            key(),
            backward(),
            forward(),
            backwardEvent(),
            forwardEvent(),
            hasInlineParams(false),
            backwardInlineParams(),
            forwardInlineParams() {
        }
    };

    struct RtgmcPendingEdiRef {
        RtgmcFrameKey key;
        std::shared_ptr<RGYFrameData> edi;
        RGYCudaEvent event;
    };

    struct RtgmcSourceCacheFrame {
        RtgmcFrameKey key;
        std::unique_ptr<CUFrameBuf> frame;
        RGYCudaEvent event;
    };

    struct RtgmcPendingFrameRef {
        RtgmcFrameKey key;
        std::shared_ptr<CUFrameBuf> frame;
        RGYCudaEvent event;
    };

    struct RtgmcMatchCorrectionPass {
        std::unique_ptr<NVEncFilterRtgmcEdi> interpolator;
        std::unique_ptr<NVEncFilterRtgmcPrimitive> correctionBuild;
        std::unique_ptr<NVEncFilterRtgmcPrimitive> correctionSpatialPrepass;
        bool fusedCorrectionBuild = false;
        bool fusedCorrectionApply = false;
        std::unique_ptr<NVEncFilterDegrain> correctionTemporalFilter;
        std::unique_ptr<NVEncFilterRtgmcPrimitive> correctionApply;
        std::unique_ptr<NVEncFilterRtgmcRetouch> correctionEnhance;
        std::deque<RtgmcPendingFrameRef> composeBaseRefs;
    };

public:
    struct RtgmcSharedAnalysisData {
        NVEncFilterDegrain *analyzeFilter;
        std::deque<RtgmcPendingEdiRef> *pendingEdiRefs;
        std::array<RtgmcSourceCacheFrame, 256> *sourceCache;
        std::deque<RtgmcPendingCompRef> *pendingCompRefs;
        std::deque<RtgmcPendingFrameRef> *pendingNoiseRefs;
        std::shared_ptr<NVRtgmcSharedFramePool> sharedFramePool;

        RtgmcSharedAnalysisData() : analyzeFilter(nullptr), pendingEdiRefs(nullptr),
            sourceCache(nullptr), pendingCompRefs(nullptr), pendingNoiseRefs(nullptr), sharedFramePool() {}
    };

    void setSharedAnalysisData(const RtgmcSharedAnalysisData &data);
    RtgmcSharedAnalysisData getSharedAnalysisData();

    struct RtgmcCapturedIntermediate {
        std::shared_ptr<CUFrameBuf> frame;
        RGYFrameInfo frameInfo;
        RGYCudaEvent event;
    };

    void enableIntermediateCapture(bool enable);
    const std::vector<RtgmcCapturedIntermediate>& getCapturedIntermediates() const;
    void clearCapturedIntermediates();
    void pushIntermediateInput(const RtgmcCapturedIntermediate &input);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
    int requiredPrimingSourceFrames() const;
protected:

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmc> &prm);
    RGY_ERR initFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm);
    RGY_ERR initRetouchCompFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm, const RGYFrameInfo &frameInfo, const rgy_rational<int> &baseFps);
    RGY_ERR initSourceMatchCorrectionFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm, const RGYFrameInfo &sourceFrameIn,
        const RGYFrameInfo &frameInfo, const rgy_rational<int> &sourceBaseFps, const rgy_rational<int> &sourceTimebase,
        const rgy_rational<int> &baseFps);
    RGY_ERR runNestedFilter(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runSourceMatchCorrectionPasses(int firstStage, int lastStage, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runSourceMatchCorrectionPass(int stage, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runThrough(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, bool storePending);
    RGY_ERR drainFrom(size_t filterIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR returnPendingFrames(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);
    RGY_ERR initBorderFrame(const RGYFrameInfo &frameInfo);
    RGY_ERR buildBorderCopyProgram(const RGYFrameInfo &frameInfo);
    RGY_ERR runBorderCopy(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, bool crop,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR addBorderToInput(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR copyFinalOutput(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    std::shared_ptr<CUFrameBuf> getSharedFrameBuffer(const RGYFrameInfo *frame);
    bool noiseRestoreEnabled() const;
    RGY_ERR storeNoiseReference(const RGYFrameInfo *baseFrame, RGYFrameInfo *denoisedFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RtgmcPendingFrameRef *findNoiseReference(const RGYFrameInfo *frame);
    void clearNoiseReference(const RGYFrameInfo *frame);
    RGY_ERR cacheSourceFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    const RtgmcSourceCacheFrame *findCachedSourceEntry(const RGYFrameInfo *frame) const;
    const RGYFrameInfo *findCachedSourceFrame(const RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events);
    int sourceFieldForFrame(const RGYFrameInfo *frame) const;
    RGY_ERR storePostLimitBaseReference(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RtgmcPendingFrameRef *findPostLimitBaseReference(const RGYFrameInfo *frame);
    void clearPostLimitBaseReference(const RGYFrameInfo *frame);
    RGY_ERR storeMatchCorrectionBaseReference(int stage, const RGYFrameInfo *keyFrame, const RGYFrameInfo *baseFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RtgmcPendingFrameRef *findMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame);
    void clearMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame);
    void enqueueSourceMatchFrameProp(const RGYFrameInfo *frame);
    RGY_ERR applySourceMatchFrameProp(RGYFrameInfo *frame);
    RGY_ERR attachEdiReference(RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR updateCompReferenceStore(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR drainCompReferenceStore(cudaStream_t stream);
    void storeEdiReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameDataRtgmcEdi> &edi, const RGYCudaEvent &event);
    RtgmcPendingEdiRef *findStoredEdiReference(const RGYFrameInfo *frame);
    void clearStoredEdiReference(const RGYFrameInfo *frame);
    void storeCompReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameData> &backward, const std::shared_ptr<RGYFrameData> &forward,
        const RGYCudaEvent &backwardEvent, const RGYCudaEvent &forwardEvent);
    RtgmcPendingCompRef *findStoredCompReference(const RGYFrameInfo *frame);
    void clearStoredCompReferences(const RGYFrameInfo *frame);
    void attachStoredEdiReference(RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events);
    void attachStoredCompReferences(RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events);

    std::vector<std::unique_ptr<NVEncFilter>> m_filters;
    NVEncFilter *m_noiseFilter;
    std::array<std::unique_ptr<NVEncFilterDegrain>, 2> m_retouchCompFilters;
    std::array<RtgmcMatchCorrectionPass, 3> m_matchCorrectionPasses;
    std::deque<RtgmcPendingEdiRef> m_pendingEdiRefs;
    std::deque<RtgmcPendingCompRef> m_pendingCompRefs;
    std::deque<RtgmcPendingFrameRef> m_pendingPostLimitBaseRefs;
    std::deque<int> m_pendingOutputFrames;
    std::array<RtgmcSourceCacheFrame, 256> m_sourceCache;
    std::shared_ptr<NVRtgmcSharedFramePool> m_sharedFramePool;
    std::shared_ptr<NVRtgmcSharedFramePool> m_ediSideDataFramePool;
    std::unique_ptr<CUFrameBuf> m_borderFrame;
    std::unique_ptr<NVEncFilterRtgmcPrimitive> m_noiseDiffFilter;
    std::deque<RtgmcPendingFrameRef> m_pendingNoiseRefs;
    RGYFrameInfo m_inputFrame;
    int m_sourceCacheNext;
    int m_outputBufferIndex;
    std::deque<RtgmcFrameKey> m_pendingSourceMatchFrameProps;
    size_t m_drainFilterIdx;
    bool m_draining;
    bool m_drainComplete;
    int m_debugResetAtFrame; // 0 = disabled; set from NVENC_RTGMC_DEBUG_RESET_AT env var
    int m_nFrame;            // count of valid input frames processed (used by debug hook)
    bool m_attachRetouchCompRefs;
    bool m_enablePostTR2Limit;
    bool m_sharedAnalysisMode;
    RtgmcSharedAnalysisData m_sharedData;
    bool m_captureIntermediate;
    std::vector<RtgmcCapturedIntermediate> m_capturedIntermediates;
    std::deque<RtgmcCapturedIntermediate> m_pendingIntermediateInputs;
};
