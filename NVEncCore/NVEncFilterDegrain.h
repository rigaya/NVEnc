// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterDegrainMV.h"
#include "rgy_prm.h"

using NVEncDegrainKernelProgram = std::string;

class NVEncFilterParamDegrain : public NVEncFilterParam {
public:
    VppDegrain degrain;
    bool attachAnalysisData;
    bool zeroCopyCache;

    NVEncFilterParamDegrain() : degrain(), attachAnalysisData(true), zeroCopyCache(false) {};
    virtual ~NVEncFilterParamDegrain() {};
    virtual tstring print() const override {
        auto str = degrain.print();
        if (!attachAnalysisData && degrain.mode == VppDegrainMode::Analyze) {
            str += _T(", direct-result");
        }
        if (zeroCopyCache) {
            str += _T(", zero-copy-cache");
        }
        return str;
    };
};

class NVEncFilterDegrain : public NVEncFilter {
public:
    static constexpr int DEGRAIN_CACHE_SIZE = 16;
    static constexpr int SCENE_CHANGE_PIPELINE_DEPTH = 2;
    static constexpr int SCENE_CHANGE_READBACK_POOL_SIZE = SCENE_CHANGE_PIPELINE_DEPTH + 1;

    NVEncFilterDegrain();
    virtual ~NVEncFilterDegrain();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGYDegrainAnalyzeResult analyzeResult() const;
    RGYDegrainAnalyzeResultSet analyzeResultSet() const;
    bool setDirectAnalyzeResult(const RGYDegrainAnalyzeResult &result);
    bool setDirectAnalyzeResultSet(const RGYDegrainAnalyzeResultSet &resultSet);
    void clearDirectAnalyzeResult();

    RGY_ERR feedFrameOnly(const RGYFrameInfo *pInputFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event = nullptr);
    bool outputReady() const;
    RGY_ERR buildCompensateInlineParams(std::array<RGYDegrainCompensateInlineParams, 3> &paramsOut, RGYFrameInfo *outputFrameIdentity, cudaStream_t stream);
    bool drainReady() const;
    RGY_ERR drainBuildInlineParams(std::array<RGYDegrainCompensateInlineParams, 3> &paramsOut, RGYFrameInfo *outputFrameIdentity, cudaStream_t stream);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
protected:

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamDegrain> &prm);
    RGY_ERR allocCacheFrames(const RGYFrameInfo &frameInfo);
    RGY_ERR pushCacheFrame(const RGYFrameInfo *pInputFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event = nullptr);
    const RGYFrameInfo *cacheFrame(int frame) const;
    RGY_ERR emitSourceFrame(const RGYFrameInfo *pCurrentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR emitDebugFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR emitCompensateFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
        const RGYDegrainRefDisableArray &disableRefs,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR emitDegrainFrame(const RGYFilterDegrainFrameSet &frames,
        const RGYDegrainRefDisableArray &disableRefs,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR attachAnalysisData(const RGYFrameInfo *sourceFrame, RGYFrameInfo *outputFrame,
        int currentFrame, cudaStream_t stream, const RGYCudaEvent &frameCopyEvent, RGYCudaEvent *event);
    RGY_ERR prepareAnalysisState(const RGYFilterDegrainFrameSet &frames, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR prepareFallbackAnalysisState(const RGYFilterDegrainProcessFrameSet &frames, int currentFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR prepareAnalysisStateMotionSearch(const RGYFrameInfo &planeCur, const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> &refPlanes,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR runSourceMode(const RGYFilterDegrainFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR runAnalyzeMode(const RGYFilterDegrainProcessFrameSet &frames, int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runDebugMode(const RGYFilterDegrainProcessFrameSet &frames, int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runCompensateMode(const RGYFilterDegrainProcessFrameSet &frames, int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runDegrainMode(const RGYFilterDegrainProcessFrameSet &frames, int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runResolvedFrames(const RGYFilterDegrainProcessFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    struct PendingSceneChange {
        std::shared_ptr<NVEncFilterParamDegrain> prm;
        RGYFilterDegrainProcessFrameSet frames;
        std::shared_ptr<RGYFrameDataDegrain> frameAnalysisData;
        RGYDegrainAnalyzeResult boundAnalyzeResult;
        RGYDegrainBlockLayout frameAnalysisLayout;
        RGYDegrainBlockLayout layout;
        RGYDegrainRefDisableArray availabilityDisableRefs;
        RGYDegrainRefDisableArray useFlagDisableRefs;
        RGYDegrainRefDisableArray disableRefs;
        std::array<uint32_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> sceneChangeBlockCounts;
        uint32_t scaledThSad;
        uint32_t scaledThSCD1;
        uint64_t scaledThSCD2;
        uint32_t disableMask;
        CUMemBuf *sad;
        RGYCudaEvent mapEvent;
        std::vector<RGYDegrainSAD> *sadHost;
        bool mapSubmitted;

        PendingSceneChange() :
            prm(),
            frames(),
            frameAnalysisData(),
            boundAnalyzeResult(),
            frameAnalysisLayout(),
            layout(),
            availabilityDisableRefs(),
            useFlagDisableRefs(),
            disableRefs(),
            sceneChangeBlockCounts(),
            scaledThSad(0),
            scaledThSCD1(0),
            scaledThSCD2(0),
            disableMask(0),
            sad(nullptr),
            mapEvent(),
            sadHost(nullptr),
            mapSubmitted(false) {
            availabilityDisableRefs.fill(true);
            useFlagDisableRefs.fill(false);
            disableRefs.fill(true);
            sceneChangeBlockCounts.fill(0);
        }
    };
    RGY_ERR submitSceneChangeReadback(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
        cudaStream_t stream, PendingSceneChange *pending);
    std::vector<RGYDegrainSAD> *acquireSceneChangeReadbackSAD(size_t count);
    RGY_ERR resolveSceneChangeReadback(PendingSceneChange &pending, cudaStream_t stream, RGYDegrainRefDisableArray *disableRefs);
    RGY_ERR emitResolvedSceneChangeFrame(PendingSceneChange &pending, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR resolvePendingSceneChangeFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    void applyPendingSceneChangeAnalysisContext(const PendingSceneChange &pending);
    void clearPendingSceneChange();
    RGY_ERR resolveSceneChangeRefs(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, cudaStream_t stream,
        RGYDegrainRefDisableArray *disableRefs);
    RGY_ERR resolveSceneChange(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, cudaStream_t stream,
        bool *disableBackward, bool *disableForward);
    RGY_ERR unsupportedModeError(VppDegrainMode mode);
    void loadDebugEnv();
    RGY_ERR buildKernels(const std::shared_ptr<NVEncFilterParamDegrain> &prm);
    NVEncDegrainKernelProgram *getDegrainMotionSearchProgram(const std::string &normalizedBuildOptions);
    NVEncDegrainKernelProgram *degrainRenderProgram(RGY_PLANE plane);
    RGY_ERR allocAnalysisBuffers(const std::shared_ptr<NVEncFilterParamDegrain> &prm);
    bool modeImplemented(VppDegrainMode mode) const;
    bool modeRequiresAnalysis(VppDegrainMode mode) const;
    bool hasDirectAnalyzeResult() const;
    bool useAnalysisLumaCache() const;
    bool prefetchAnalysisLumaCache() const;
    RGYFilterDegrainFrameSet resolveFrameSet(int currentFrame) const;
    const RGYFrameInfo *resolveAnalysisLumaSourceFrame(int frameIndex) const;
    RGYFilterDegrainFrameSet resolveAnalysisFrameSet(int currentFrame) const;
    RGYFilterDegrainProcessFrameSet resolveFrames(bool hasInput) const;
    RGY_ERR generateAnalysisLumaFrame(int centerFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR ensureAnalysisLumaGenerated(int targetFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    int outputDelay() const;
    int drainFrameCount() const;
    int cacheIndex(int frame) const;
    int analysisCacheIndex(int frame) const;
    void clearFrameAnalysisData();
    bool degrainDebugLogEnabled() const;
    void logAnalyzeBinding(const TCHAR *sourceName, const RGYFrameInfo *frame, const RGYDegrainAnalyzeResult &result);
    void logLocalAnalysis(const TCHAR *sourceName, const RGYFilterDegrainFrameSet &frames);
    void logAnalysisSamples(const TCHAR *sourceName, const RGYFrameInfo *frame, cudaStream_t stream);
    void logReferenceGate(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames,
        const RGYDegrainRefDisableArray &availabilityDisableRefs, const RGYDegrainRefDisableArray &useFlagDisableRefs,
        const RGYDegrainRefDisableArray &disableRefs,
        const std::array<size_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> *sceneChangeBlockCounts,
        uint32_t scaledThSad, uint32_t scaledThSCD1, uint64_t scaledThSCD2);
    bool degrainApplyTraceEnabled() const;
    void logApplyTrace(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
        const RGYDegrainRefDisableArray &disableRefs, cudaStream_t stream);
    int requestedDelta() const;
    bool validateAnalyzeResultFrame(const RGYDegrainAnalyzeResult &result, const RGYFrameInfo *frame, int currentFrame, const TCHAR *sourceName, bool requireFrameIndex);
    bool bindAnalyzeResult(const RGYDegrainAnalyzeResult &result, const RGYFrameInfo *frame, int currentFrame, const TCHAR *sourceName, bool requireFrameIndex, cudaStream_t stream);
    bool bindDirectAnalyzeResult(const RGYFrameInfo *frame, int currentFrame, cudaStream_t stream);
    bool bindFrameAnalysisData(const RGYFrameInfo *frame, int currentFrame, cudaStream_t stream);
    CUMemBuf *analysisMV() const;
    CUMemBuf *analysisSAD() const;
    const RGYDegrainBlockLayout &analysisLayout() const;
    const RGYCudaEvent &analysisEvent() const;
    bool analysisSADIncludesChroma(const std::shared_ptr<NVEncFilterParamDegrain> &prm) const;
    RGYDegrainRefDisableArray analysisAvailabilityDisableRefs(const RGYFilterDegrainFrameSet &frames) const;

    struct DebugEnv {
        bool applyTrace;
        int applyTraceBlock;
        bool forceDegrainCopy;
        bool pixelTrace;
        int pixelTraceX;
        int pixelTraceY;
        int pixelTraceFrame;

        DebugEnv() :
            applyTrace(false),
            applyTraceBlock(-1),
            forceDegrainCopy(false),
            pixelTrace(false),
            pixelTraceX(0),
            pixelTraceY(0),
            pixelTraceFrame(-1) {
        }
    };

    std::array<std::unique_ptr<CUFrameBuf>, DEGRAIN_CACHE_SIZE> m_cacheFrames;
    std::array<RGYFrameInfo, DEGRAIN_CACHE_SIZE> m_cacheFrameRefs;
    std::array<std::shared_ptr<CUFrameBuf>, DEGRAIN_CACHE_SIZE> m_cacheFrameOwners;
    NVEncDegrainKernelProgram m_degrain;
    NVEncDegrainKernelProgram m_degrainChroma;
    NVEncDegrainKernelProgram m_degrainPel1;
    std::unordered_map<std::string, NVEncDegrainKernelProgram> m_degrainMotionSearchPrograms;
    RGYDegrainAnalysisState m_analysis;
    RGYDegrainAnalyzeResultSet m_directAnalyzeResultSet;
    RGYDegrainAnalyzeResult m_boundAnalyzeResult;
    std::shared_ptr<RGYFrameDataDegrain> m_frameAnalysisData;
    RGYDegrainBlockLayout m_frameAnalysisLayout;
    std::deque<std::unique_ptr<PendingSceneChange>> m_pendingSceneChange;
    std::shared_ptr<RGYDegrainBufferPool> m_sideDataBufferPool;
    std::array<std::vector<RGYDegrainSAD>, SCENE_CHANGE_READBACK_POOL_SIZE> m_sceneChangeReadbackSAD;
    std::unique_ptr<CUMemBuf> m_sceneChangeCounts;
    std::unique_ptr<CUMemBuf> m_sceneChangeDisableMask;
    int m_sceneChangeReadbackSADIndex;
    int m_inputCount;
    int m_drainCount;
    bool m_bInterlacedWarn;
    bool m_lastAnalysisUsedSearchLuma;
    bool m_lastAnalysisIncludedChroma;
    bool m_useDegrainChromaProgram;
    DebugEnv m_debugEnv;
};
