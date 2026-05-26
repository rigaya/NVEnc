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
#include <cstdio>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterDegrainMV.h"
#include "rgy_filter_kfm_analyze.h"

class NVEncFilterDegrain;
class NVEncFilterRtgmc;

class NVEncFilterParamKfm : public NVEncFilterParam {
public:
    VppKfm kfm;
    rgy_rational<int> timebase;

    NVEncFilterParamKfm() : kfm(), timebase() {};
    virtual ~NVEncFilterParamKfm() {};
    virtual tstring print() const override;
};

RGY_ERR run_kfm_pad_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int vpad, cudaStream_t stream);
RGY_ERR run_kfm_init_fmcount(RGYKFM::FMCount *dst, cudaStream_t stream);
RGY_ERR run_kfm_analyze_count_cmflags_clean(
    RGYKFM::FMCount *dst,
    const RGYFrameInfo *prevSrc0,
    const RGYFrameInfo *prevSrc1,
    const RGYFrameInfo *curSrc0,
    const RGYFrameInfo *curSrc1,
    int width,
    int height,
    int prevParity,
    int curParity,
    int countParity,
    int pixelStep,
    int pixelOffset,
    int threshM,
    int threshS,
    int threshLS,
    int cleanThresh,
    cudaStream_t stream);
RGY_ERR run_kfm_zero_plane(RGYFrameInfo *pOutputFrame, cudaStream_t stream);
RGY_ERR run_kfm_static_calc_combe_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int srcYOffset, cudaStream_t stream);
RGY_ERR run_kfm_temporal_min_diff5_3_plane(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *src0,
    const RGYFrameInfo *src1,
    const RGYFrameInfo *src2,
    const RGYFrameInfo *src3,
    const RGYFrameInfo *src4,
    const RGYFrameInfo *src5,
    const RGYFrameInfo *src6,
    cudaStream_t stream);
RGY_ERR run_kfm_merge_uv_coefs_plane(RGYFrameInfo *flagY, const RGYFrameInfo *flagU, const RGYFrameInfo *flagV, int logUVx, int logUVy, cudaStream_t stream);
RGY_ERR run_kfm_extend_coefs_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
RGY_ERR run_kfm_and_coefs_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDiffFrame, float invcombe, float invdiff, cudaStream_t stream);
RGY_ERR run_kfm_apply_uv_coefs_420_plane(RGYFrameInfo *flagU, RGYFrameInfo *flagV, const RGYFrameInfo *flagY, cudaStream_t stream);
RGY_ERR run_kfm_merge_static_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pFlagFrame, cudaStream_t stream);
RGY_ERR run_kfm_telecine_weave_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *src0, const RGYFrameInfo *src1, const RGYFrameInfo *src2,
    int srcYOffset, int fieldStart, int fieldCount, int parity, cudaStream_t stream);
RGY_ERR run_kfm_analyze_plane(uint8_t *dst, int dstPitch,
    const RGYFrameInfo *src0, const RGYFrameInfo *src1,
    int width, int height, int parity, int pixelStep, int pixelOffset, cudaStream_t stream);
RGY_ERR run_kfm_clean_super_direct_max_plane(RGYFrameInfo *dst,
    const RGYFrameInfo *prevSrc0, const RGYFrameInfo *prevSrc1, int prevParity,
    const RGYFrameInfo *curSrc0, const RGYFrameInfo *curSrc1, int curParity,
    int widthPairs, int height, int field, int cleanThresh, int maxMode,
    int dstStep, int dstOffset, int pixelStep, int pixelOffset, cudaStream_t stream);
RGY_ERR run_kfm_clean_separated_super_max_plane(RGYFrameInfo *dst,
    const uint8_t *prevSuper, const uint8_t *curSuper, int superPitch,
    int widthPairs, int height, int field, int cleanThresh, int maxMode,
    int dstStep, int dstOffset, cudaStream_t stream);
RGY_ERR run_kfm_remove_combe_binomial_plane(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pCombeFrame,
    const RGYFrameInfo *teleSrc0, const RGYFrameInfo *teleSrc1, const RGYFrameInfo *teleSrc2,
    int threshold, int srcStep, int srcOffset, int combeStep, int combeOffset,
    int teleSrcYOffset, int teleFieldStart, int teleFieldCount, int teleParity, cudaStream_t stream);
RGY_ERR run_kfm_patch_combe_plane(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame,
    int threshold, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_combe_min(uint8_t *dstY, int dstYPitch, uint8_t *dstC, int dstCPitch,
    const RGYFrameInfo *superPrevY, const RGYFrameInfo *superY, const RGYFrameInfo *superNextY,
    const RGYFrameInfo *superPrevUV, const RGYFrameInfo *superUV, const RGYFrameInfo *superNextUV,
    const RGYFrameInfo *superPrevV, const RGYFrameInfo *superV, const RGYFrameInfo *superNextV,
    int combeWidth, int combeHeight, int combeCWidth, int combeCHeight,
    int hasUV, int interleavedUV, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_from_combe_min(uint8_t *dstY, int dstYPitch, uint8_t *dstC, int dstCPitch,
    const uint8_t *combeY, int combeYPitch, const uint8_t *combeC, int combeCPitch,
    int flagWidth, int flagHeight, int innerWidth, int innerHeight,
    int combeWidth, int combeHeight, int combeCWidth, int combeCHeight, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_box3x3_min(uint8_t *dst, int dstPitch, const uint8_t *src, int srcPitch,
    int width, int height, int innerWidth, int innerHeight, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_binary_extend_hv_min(RGYFrameInfo *dst,
    const uint8_t *srcY, int srcYPitch, const uint8_t *srcC, int srcCPitch,
    int innerWidth, int innerHeight, int thY, int thC, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_binary_min(RGYFrameInfo *dst,
    const uint8_t *srcY, int srcYPitch, const uint8_t *srcC, int srcCPitch,
    int innerWidth, int innerHeight, int thY, int thC, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_extend_h_min(uint8_t *dst, int dstPitch, const RGYFrameInfo *src,
    int width, int height, int offsetX, int offsetY, cudaStream_t stream);
RGY_ERR run_kfm_switch_flag_extend_v_min(RGYFrameInfo *dst, const uint8_t *src, int srcPitch,
    int width, int height, int offsetX, int offsetY, cudaStream_t stream);
RGY_ERR run_kfm_contains_combe_init(uint32_t *count, cudaStream_t stream);
RGY_ERR run_kfm_contains_combe_count(const RGYFrameInfo *mask, uint32_t *count, int threshold, cudaStream_t stream);
RGY_ERR run_kfm_contains_combe_mark(RGYFrameInfo *dst, const uint32_t *count, cudaStream_t stream);
RGY_ERR run_kfm_combe_mask_resize_bilinear_min_plane(RGYFrameInfo *dst, const RGYFrameInfo *flag,
    int srcStep, int srcOffset, int scaleX, int shiftX, int scaleY, int shiftY,
    int innerWidth, int innerHeight, cudaStream_t stream);
RGY_ERR run_kfm_copy_u8_buffer_to_plane(RGYFrameInfo *dst, const uint8_t *src, int srcPitch,
    int width, int height, cudaStream_t stream);
RGY_ERR run_kfm_ucf_copy_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream);
RGY_ERR run_kfm_ucf_field_crop_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    int srcXOffset, int srcYOffset, int srcYStep, cudaStream_t stream);
RGY_ERR run_kfm_ucf_gaussresize_v_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int *offset, const float *coeff, int filterSize, cudaStream_t stream);
RGY_ERR run_kfm_ucf_field_crop_gaussresize_v_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    int srcXOffset, int srcYOffset, int srcYStep, const int *offset, const float *coeff, int filterSize, cudaStream_t stream);
RGY_ERR run_kfm_ucf_gaussresize_h_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int *offset, const float *coeff, int filterSize, cudaStream_t stream);
RGY_ERR run_kfm_ucf_gaussresize_h_uv_interleaved_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    int chromaWidth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream);
RGY_ERR run_kfm_ucf_analyze_noise_partial(RGYKFM::NoiseResult *dst, int dstOffset,
    const RGYFrameInfo *src0, const RGYFrameInfo *src1, const RGYFrameInfo *src2,
    int width4, int height, cudaStream_t stream);
RGY_ERR run_kfm_ucf_analyze_diff_partial(RGYKFM::NoiseResult *dst, int dstOffset,
    const RGYFrameInfo *src0, const RGYFrameInfo *src1,
    int width4, int height, int srcYOffset, cudaStream_t stream);
RGY_ERR run_kfm_ucf_noise_limit_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int nmin, int range, cudaStream_t stream);
RGY_ERR run_kfm_ucf_source_crop_noise_limit_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int srcXOffset, int srcYOffset, int srcYStep, int nmin, int range, cudaStream_t stream);

class RGYFrameDataKfmSwitch : public RGYFrameData {
public:
    RGYFrameDataKfmSwitch(int n60, int n24, int baseType, int sourceStart, int numSourceFrames, int duration60, int duration120, int pattern, float cost) :
        m_n60(n60),
        m_n24(n24),
        m_baseType(baseType),
        m_sourceStart(sourceStart),
        m_numSourceFrames(numSourceFrames),
        m_duration60(duration60),
        m_duration120(duration120),
        m_pattern(pattern),
        m_cost(cost) {
        m_dataType = RGY_FRAME_DATA_KFM_SWITCH;
    }
    virtual ~RGYFrameDataKfmSwitch() {};

    int n60() const { return m_n60; }
    int n24() const { return m_n24; }
    int baseType() const { return m_baseType; }
    int sourceStart() const { return m_sourceStart; }
    int numSourceFrames() const { return m_numSourceFrames; }
    int duration60() const { return m_duration60; }
    int duration120() const { return m_duration120; }
    int pattern() const { return m_pattern; }
    float cost() const { return m_cost; }

protected:
    int m_n60;
    int m_n24;
    int m_baseType;
    int m_sourceStart;
    int m_numSourceFrames;
    int m_duration60;
    int m_duration120;
    int m_pattern;
    float m_cost;
};

class NVEncFilterKfm : public NVEncFilter {
public:
    NVEncFilterKfm();
    virtual ~NVEncFilterKfm();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual int requiredOutputFrames() const;

protected:
    struct KfmCachedSource;
    struct KfmCachedDeint60;
    struct KfmCachedUcfNoise;
    struct KfmUcfNoiseDumpRecord;
    struct KfmPendingUcfNoiseResult;
    struct KfmUcfGaussProgram;
    struct KfmPendingFMCount;
    struct KfmPendingVfrOutput;
    enum KfmFrameType {
        KFM_FRAME_60 = 1,
        KFM_FRAME_30 = 2,
        KFM_FRAME_24 = 3,
        KFM_FRAME_UCF = 4,
    };
    enum KfmUcf60Flag {
        KFM_UCF60_NONE = 0,
        KFM_UCF60_NR = 1,
        KFM_UCF60_PREV = 2,
        KFM_UCF60_NEXT = 3,
    };
    enum KfmUcf24SelectType {
        KFM_UCF24_SELECT_DEINT24 = 0,
        KFM_UCF24_SELECT_FRAME = 1,
        KFM_UCF24_SELECT_DWEAVE = 2,
    };
    struct KfmUcf24Selection {
        KfmUcf24SelectType type;
        int n60;
        const RGYFrameInfo *frame;

        KfmUcf24Selection() : type(KFM_UCF24_SELECT_DEINT24), n60(-1), frame(nullptr) {};
    };
    struct KfmSwitchTiming {
        int start60;
        int start120;
        int sourceIndex;
        int frame24Index;
        int baseType;
        int sourceStart;
        int numSourceFrames;
        int duration60;
        int duration120;
        bool isFrame24;
        bool isFrame60;

        KfmSwitchTiming() : start60(0), start120(0), sourceIndex(0), frame24Index(-1), baseType(KFM_FRAME_60), sourceStart(0), numSourceFrames(1), duration60(1), duration120(2), isFrame24(false), isFrame60(false) {};
    };
    class SharedFramePool {
    public:
        std::shared_ptr<CUFrameBuf> acquire(const RGYFrameInfo& info);
        void clear();
    private:
        std::deque<std::shared_ptr<CUFrameBuf>> m_pool;
    };

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR initRtgmc(const std::shared_ptr<NVEncFilterParamKfm>& prm, std::unique_ptr<NVEncFilterRtgmc>& rtgmc, bool updateOutputParam, int useFlag = 0);
    RGY_ERR initAnalyzer(const NVEncFilterParamKfm& prm);
    RGY_ERR initNrFilter(const std::shared_ptr<NVEncFilterParamKfm>& prm);
    void initStageDumpConfig(const NVEncFilterParamKfm& prm);
    bool stageDumpRequested(int frame24Index) const;
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, int frame24Index,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR padSourceFrame(RGYFrameInfo *pPaddedFrame, const RGYFrameInfo *pSourceFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR cacheSourceFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR runDeint60Branch(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, int *cachedFrames = nullptr);
    RGY_ERR drainDeint60Branch(cudaStream_t stream, int *cachedFrames = nullptr);
    RGY_ERR cacheDeint60Frame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runUcfRtgmcBranches(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR runUcfRtgmcBranch(NVEncFilterRtgmc *rtgmc, const char *stage, const RGYFrameInfo *frame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
        std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYCudaEvent& cacheCopyEvent);
    RGY_ERR drainUcfRtgmcBranch(NVEncFilterRtgmc *rtgmc, const char *stage, cudaStream_t stream,
        std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYCudaEvent& cacheCopyEvent);
    RGY_ERR cacheUcfRtgmcFrame(const char *stage, const RGYFrameInfo *frame, cudaStream_t stream,
        const std::vector<RGYCudaEvent> &wait_events, std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYCudaEvent *event);
    std::shared_ptr<CUFrameBuf> acquireKfmFrame(const RGYFrameInfo& info, const TCHAR *label);
    RGY_ERR allocWorkFrameBuf(const RGYFrameInfo& frame, int frames);
    RGYFrameInfo *nextOutputFrame();
    RGYFrameInfo *nextWorkFrame();
    size_t sourceCacheLimit() const;
    size_t deint60CacheLimit() const;
    int sourceCacheTrimFloor() const;
    int deint60CacheTrimFloor() const;
    const RGYFrameInfo *findDeint60Frame(int n60, std::vector<RGYCudaEvent> *wait_events) const;
    const RGYFrameInfo *findSourceFrame(const RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events);
    const KfmCachedSource *findSourceByIndex(int sourceIndex) const;
    const KfmCachedSource *findSourceByIndexExact(int sourceIndex) const;
    const KfmCachedDeint60 *findCachedDeint60Frame(const std::deque<KfmCachedDeint60>& cache, int n60, std::vector<RGYCudaEvent> *wait_events) const;
    const KfmUcfNoiseDumpRecord *findUcfNoiseResult(int sourceIndex) const;
    RGY_ERR copyUcfFrame(const NVEncFilterParamKfm& prm, RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR createUcfGaussProgram(KfmUcfGaussProgram& program, int sourceSize, double cropSize, int targetSize, double p);
    RGY_ERR prepareUcfNoiseFieldCropFrame(RGYFrameInfo **ppFieldFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR prepareUcfNoiseGaussFrameFromSource(RGYFrameInfo **ppGaussFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR prepareUcfNoiseGaussFrame(RGYFrameInfo **ppGaussFrame, int parity, const RGYFrameInfo *pInputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR runUcfNoiseLimitStageFromSource(const NVEncFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
        int fieldIndex, int parity, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR runUcfNoiseLimitStage(const NVEncFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
        int fieldIndex, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);
    RGY_ERR analyzeUcfNoiseDebug(cudaStream_t stream);
    RGY_ERR submitUcfNoiseResult(const KfmCachedUcfNoise& noise0, const KfmCachedUcfNoise& noise1, const KfmCachedUcfNoise& noise2,
        const KfmCachedSource& source0, const KfmCachedSource& source1, cudaStream_t stream);
    RGY_ERR resolveUcfNoiseResult(KfmPendingUcfNoiseResult& pending, cudaStream_t stream);
    RGY_ERR resolveUcfNoiseResults(int sourceIndex, cudaStream_t stream);
    RGY_ERR resolveAllUcfNoiseResults(cudaStream_t stream);
    std::unique_ptr<CUMemBufPair> acquireUcfNoiseResultBuf(size_t requiredBytes);
    void releaseUcfNoiseResultBuf(std::unique_ptr<CUMemBufPair>&& buf);
    void pushUcfNoiseResultDump(int sourceIndex, const RGYKFM::NoiseResult (&results)[2], const RGYKFM::UCFNoiseMeta& meta);
    void writeUcfNoiseResultDump(const KfmUcfNoiseDumpRecord& record, const KfmUcfNoiseDumpRecord *nextRecord);
    const RGYFrameInfo *selectUcfDecomb30Frame(int sourceIndex, const RGYFrameInfo *deint30, std::vector<RGYCudaEvent> *wait_events) const;
    bool getUcf60FieldDiff(int nstart, double (&diff)[4]) const;
    KfmUcf60Flag calcUcf60Flag(int n60) const;
    const RGYFrameInfo *selectUcfDecomb60Frame(int n60, const RGYFrameInfo *deint60, std::vector<RGYCudaEvent> *wait_events) const;
    KfmUcf24Selection selectUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo, const RGYFrameInfo *deint24, std::vector<RGYCudaEvent> *wait_events) const;
    void finalizeAnalyzerResults(VppKfmTiming timing);
    std::vector<RGYKFM::KFMResult> analyzerResultsSnapshot(bool mark60p) const;
    void appendAnalyzerResults(size_t resultCount, bool dump, bool mark60p);
    std::vector<KfmSwitchTiming> deriveSwitchTimings(int total60) const;
    int64_t sourceFrameDuration(const KfmCachedSource *source) const;
    bool isSwitchSingleFrameN60(int n60) const;
    void markSwitchSingleFrameN60Range(int start60, int duration60);
    bool switchSingleFrameDurationEnabled() const;
    void writeSwitchTimingDump();
    void writeTelecine24DurationDump();
    void writeFMCountDump(const std::array<RGYKFM::FMCount, 18>& counts, int cycle);
    void writeAnalyzerResult(const RGYKFM::KFMResult& result, bool dump);
    void writeAnalyzerResultsFinal(size_t resultCount, bool mark60p);
    void writeFrameTimecode(const RGYFrameInfo *frame);
    void writeFrameInfoDump(const char *stage, const RGYFrameInfo *frame, const RGYKFM::KFMResult *result = nullptr);
    void writeContainsCombeDump(const char *stage, const KfmSwitchTiming& timing, uint32_t containsCombeCount, bool durationApplied, const RGYKFM::KFMResult *result);
    void attachSwitchFrameData(RGYFrameInfo *frame, const KfmSwitchTiming& timing, const RGYKFM::KFMResult *result) const;
    int telecine24FrameCount(bool drain) const;
    RGY_ERR runNrFilter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR processMainRtgmcOutputs(const NVEncFilterParamKfm& prm, RGYFrameInfo **rtgmcOutFrames, int rtgmcOutNum,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR drainMainRtgmcBranch(const NVEncFilterParamKfm& prm, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR emitOutputFrame(RGYFrameInfo *pFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const RGYCudaEvent &frameEvent, RGYCudaEvent *event);
    RGY_ERR queueVfrOutputFrame(const RGYFrameInfo *pFrame, cudaStream_t stream, const RGYCudaEvent &frameEvent);
    RGY_ERR emitPendingVfrOutput(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR emitPendingVfrOutputs(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event, int keepFrames);
    RGY_ERR drainNrFilter(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, RGYCudaEvent *event);
    RGY_ERR renderTelecine24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR renderDoubleWeaveFrame(RGYFrameInfo *pOutputFrame, int firstField, int fieldCount, bool drain,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR renderCleanSuperFields(RGYFrameInfo *pOutputFrame, int firstSuperField, int lastSuperField, int propSourceIndex, int outputFrameId, bool drain,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR renderTelecineSuper24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR renderSuper30(RGYFrameInfo *pOutputFrame, int frame30Index, bool drain,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR removeCombeFields(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeintFrame, const RGYFrameInfo *pTelecineSuperFrame,
        int firstField, int fieldCount, int stageFrameIndex, const char *stageName,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR removeCombe24(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint24Frame, const RGYFrameInfo *pTelecineSuperFrame, int frame24Index,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR ensureMaskBranchFrames(RGYFrameInfo **ppSwitchFlagFrame, RGYFrameInfo **ppContainsCombeFrame, RGYFrameInfo **ppCombeMaskFrame,
        const RGYFrameInfo *pTelecineSuperFrame, const TCHAR *stageLabel);
    RGY_ERR renderMaskBranch(RGYFrameInfo *pSwitchFlagFrame, RGYFrameInfo *pContainsCombeFrame, RGYFrameInfo *pCombeMaskFrame,
        const RGYFrameInfo *pTelecineSuperPrevFrame, const RGYFrameInfo *pTelecineSuperFrame, const RGYFrameInfo *pTelecineSuperNextFrame,
        const char *switchFlagStage, const char *containsCombeStage, const char *combeMaskStage,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, uint32_t *containsCombeCount = nullptr);
    RGY_ERR patchCombe(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame,
        int frameIndex, const char *stageName, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR submitFMCounts(int cycle, bool drain, cudaStream_t stream);
    RGY_ERR readbackFMCounts(std::array<RGYKFM::FMCount, 18>& counts, int cycle, bool drain, cudaStream_t stream);
    RGY_ERR analyzeAvailableSource(bool drain, cudaStream_t stream);
    RGY_ERR clearStaticFlag(cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event = nullptr);
    RGY_ERR analyzeStaticFlag(cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR analyzeStaticFlag(int sourceIndex, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    RGY_ERR mergeStatic(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    void flushUcfNoiseResultDump();

    struct KfmCachedSource {
        int sourceIndex;
        int inputFrameId;
        int64_t timestamp;
        std::shared_ptr<CUFrameBuf> frame;
        std::shared_ptr<CUFrameBuf> paddedFrame;
        RGYCudaEvent event;
        RGYCudaEvent paddedEvent;

        KfmCachedSource() : sourceIndex(-1), inputFrameId(-1), timestamp(0), frame(), paddedFrame(), event(), paddedEvent() {};
    };

    struct KfmCachedDeint60 {
        int n60;
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;
        std::shared_ptr<CUFrameBuf> frame;
        RGYCudaEvent event;

        KfmCachedDeint60() : n60(-1), inputFrameId(-1), timestamp(0), duration(0), frame(), event() {};
    };

    struct KfmCachedUcfNoise {
        int fieldIndex;
        int inputFrameId;
        int64_t timestamp;
        std::shared_ptr<CUFrameBuf> frame;
        RGYCudaEvent event;

        KfmCachedUcfNoise() : fieldIndex(-1), inputFrameId(-1), timestamp(0), frame(), event() {};
    };

    struct KfmUcfNoiseDumpRecord {
        int sourceIndex;
        RGYKFM::NoiseResult results[2];
        RGYKFM::UCFNoiseMeta meta;
        bool valid;

        KfmUcfNoiseDumpRecord() : sourceIndex(-1), results(), meta(), valid(false) {};
    };

    struct KfmPendingUcfNoiseResult {
        struct Segment {
            int offset;
            int count;
            int plane;
        };
        int sourceIndex;
        std::unique_ptr<CUMemBufPair> resultBuf;
        std::vector<Segment> segments;
        RGYCudaEvent event;
        RGYKFM::UCFNoiseMeta meta;

        KfmPendingUcfNoiseResult() : sourceIndex(-1), resultBuf(), segments(), event(), meta() {};
    };

    struct KfmUcfGaussProgram {
        int sourceSize;
        int targetSize;
        int filterSize;
        std::unique_ptr<CUMemBuf> offset;
        std::unique_ptr<CUMemBuf> coeff;

        KfmUcfGaussProgram() : sourceSize(0), targetSize(0), filterSize(0), offset(), coeff() {};
    };

    struct KfmPendingFMCount {
        int cycle;
        std::vector<std::unique_ptr<CUMemBufPair>> pairCounts;
        std::vector<RGYCudaEvent> pairEvents;

        KfmPendingFMCount() : cycle(-1), pairCounts(), pairEvents() {};
    };

    struct KfmPendingVfrOutput {
        std::shared_ptr<CUFrameBuf> frame;
        RGYCudaEvent event;

        KfmPendingVfrOutput() : frame(), event() {};
    };

    std::unique_ptr<NVEncFilterRtgmc> m_rtgmc;
    std::unique_ptr<NVEncFilterRtgmc> m_deint60Rtgmc;
    std::unique_ptr<NVEncFilterRtgmc> m_before60Rtgmc;
    std::unique_ptr<NVEncFilterRtgmc> m_after60Rtgmc;
    std::unique_ptr<NVEncFilterDegrain> m_nrFilter;
    std::unique_ptr<RGYKFM::KFMAnalyze> m_analyzer;
    std::shared_ptr<SharedFramePool> m_kfmFramePool;
    std::deque<KfmCachedSource> m_sourceCache;
    std::deque<KfmCachedDeint60> m_deint60Cache;
    std::deque<KfmCachedDeint60> m_before60Cache;
    std::deque<KfmCachedDeint60> m_after60Cache;
    std::deque<KfmCachedUcfNoise> m_ucfNoiseCache;
    std::deque<KfmPendingUcfNoiseResult> m_pendingUcfNoiseResults;
    std::deque<std::unique_ptr<CUMemBuf>> m_fmCountBufPool;
    std::deque<std::unique_ptr<CUMemBufPair>> m_ucfNoiseResultBufPool;
    std::deque<KfmUcfNoiseDumpRecord> m_ucfNoiseResultCache;
    KfmUcfNoiseDumpRecord m_pendingUcfNoiseDump;
    int m_deint60SubmittedSourceFrames;
    int m_before60SubmittedSourceFrames;
    int m_after60SubmittedSourceFrames;
    RGYCudaEvent m_deint60CacheCopyEvent;
    RGYCudaEvent m_before60CacheCopyEvent;
    RGYCudaEvent m_after60CacheCopyEvent;
    std::unique_ptr<CUFrameBuf> m_staticFlag;
    std::array<std::unique_ptr<CUFrameBuf>, 5> m_staticWorkFrames;
    std::array<std::unique_ptr<CUMemBuf>, 2> m_analyzeFlags;
    std::deque<KfmPendingFMCount> m_pendingFMCounts;
    std::deque<KfmPendingVfrOutput> m_pendingVfrOutputs;
    std::array<std::unique_ptr<CUMemBuf>, 2> m_telecineSuperRaw;
    std::array<std::unique_ptr<CUFrameBuf>, 2> m_telecineSuperFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 2> m_telecineSuperNeighborFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 4> m_switchFlagFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 4> m_containsCombeFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 4> m_combeMaskFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 4> m_patchCombeFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 2> m_ucfNoiseFieldFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 2> m_ucfNoiseGaussTmpFrames;
    std::array<std::unique_ptr<CUFrameBuf>, 2> m_ucfNoiseGaussFrames;
    std::array<std::array<KfmUcfGaussProgram, 2>, 2> m_ucfNoiseGaussVert;
    std::array<std::array<KfmUcfGaussProgram, 2>, 2> m_ucfNoiseGaussHori;
    std::array<std::unique_ptr<CUMemBuf>, 4> m_switchFlagWork;
    RGYCudaEvent m_switchFlagWorkEvent;
    std::unique_ptr<CUMemBuf> m_containsCombeCount;
    FILE *m_fpResult;
    FILE *m_fpFMCount;
    FILE *m_fpTimecode;
    FILE *m_fpFrameInfo;
    FILE *m_fpContainsCombe;
    FILE *m_fpUcfNoise;
    tstring m_switchDurationPath;
    tstring m_switchTimecodePath;
    std::string m_stageDumpDir;
    RGYKFM::KFMResult m_lastAnalyzeResult;
    std::vector<RGYKFM::KFMResult> m_analyzerOutputResults;
    bool m_hasLastAnalyzeResult;
    bool m_analyzerFinalized;
    bool m_switchTimingDumped;
    int m_analyzeSourceFrames;
    int m_nextAnalyzeCycle;
    int m_nextFMCountSubmitCycle;
    int m_nextFMCountDumpFrame;
    int m_cachedSourceFrames;
    int m_nextSwitchN60;
    int64_t m_nextSwitchPts;
    bool m_hasLastSwitchTiming;
    int m_lastSwitchStart60;
    int m_lastSwitchDuration60;
    int64_t m_lastSwitchStart120;
    bool m_lastSwitchIsFrame24;
    std::vector<int> m_switchSingleFrameN60;
    std::unordered_map<std::string, int> m_stageDumpFrameCounts;
    std::unordered_map<std::string, std::unordered_set<int>> m_stageDumpFrameIndices;
    std::unordered_set<int> m_stageDumpTargetFrames;
    int m_nextTelecine24Frame;
    int64_t m_nextTelecine24Pts;
    int m_telecineSuperBufferIndex;
    int m_maskBranchBufferIndex;
    int m_patchCombeBufferIndex;
    int m_stageDumpMaxFrames;
    int m_timecodeFrameIndex;
    int m_outputBufferIndex;
    std::vector<std::unique_ptr<CUFrameBuf>> m_workFrameBuf;
    int m_workBufferIndex;
};
