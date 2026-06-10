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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "NVEncFilter.h"
#include "rgy_cuda_util.h"
#include "rgy_prm.h"

static constexpr int RGY_DEGRAIN_MAX_DELTA = 5;

class RGYCudaEvent {
public:
    RGYCudaEvent() : m_event() {}
    RGYCudaEvent(const std::shared_ptr<cudaEvent_t>& event) : m_event(event) {}
    ~RGYCudaEvent() {}

    cudaEvent_t operator()() const { return m_event ? *m_event : nullptr; }
    void reset() { m_event.reset(); }
    void set(const std::shared_ptr<cudaEvent_t>& event) { m_event = event; }
    std::shared_ptr<cudaEvent_t> shared() const { return m_event; }

private:
    std::shared_ptr<cudaEvent_t> m_event;
};

struct RGYFilterDegrainFrameSet {
    std::array<const RGYFrameInfo *, RGY_DEGRAIN_MAX_DELTA> backward;
    const RGYFrameInfo *cur;
    std::array<const RGYFrameInfo *, RGY_DEGRAIN_MAX_DELTA> forward;
    std::array<bool, RGY_DEGRAIN_MAX_DELTA> backwardInRange;
    std::array<bool, RGY_DEGRAIN_MAX_DELTA> forwardInRange;

    RGYFilterDegrainFrameSet() :
        backward(),
        cur(nullptr),
        forward(),
        backwardInRange(),
        forwardInRange() {
        backward.fill(nullptr);
        forward.fill(nullptr);
        backwardInRange.fill(false);
        forwardInRange.fill(false);
    }

    const RGYFrameInfo *backwardRef(const int delta) const {
        return (delta >= 1 && delta <= RGY_DEGRAIN_MAX_DELTA) ? backward[delta - 1] : nullptr;
    }
    const RGYFrameInfo *forwardRef(const int delta) const {
        return (delta >= 1 && delta <= RGY_DEGRAIN_MAX_DELTA) ? forward[delta - 1] : nullptr;
    }
    bool backwardRefInRange(const int delta) const {
        return (delta >= 1 && delta <= RGY_DEGRAIN_MAX_DELTA) ? backwardInRange[delta - 1] : false;
    }
    bool forwardRefInRange(const int delta) const {
        return (delta >= 1 && delta <= RGY_DEGRAIN_MAX_DELTA) ? forwardInRange[delta - 1] : false;
    }
};

struct RGYFilterDegrainProcessFrameSet {
    RGYFilterDegrainFrameSet render;
    RGYFilterDegrainFrameSet analysis;
    int currentFrame;
};

enum class RGYDegrainRefDir : uint16_t {
    Backward = 0,
    Forward = 1,
    Backward2 = 2,
    Forward2 = 3,
    Backward3 = 4,
    Forward3 = 5,
    Backward4 = 6,
    Forward4 = 7,
    Backward5 = 8,
    Forward5 = 9,
};

struct RGYDegrainMV {
    int16_t dx;
    int16_t dy;
    uint16_t sad;
    uint16_t refdir;
    uint32_t flags;
    uint32_t reserved;
};

struct RGYDegrainSAD {
    uint32_t sad;
    uint32_t srcAvg;
    uint32_t refAvg;
    uint32_t reserved;
};

static_assert(sizeof(RGYDegrainMV) == 16, "RGYDegrainMV ABI mismatch.");
static_assert(sizeof(RGYDegrainSAD) == 16, "RGYDegrainSAD ABI mismatch.");

// Future analyze/apply split will need to attach MV/SAD analysis to frames.
// Keep this header self-contained so frame side-data can reuse it unchanged.
static constexpr uint32_t RGY_DEGRAIN_FRAME_META_SIGNATURE = 0x4d434447u; // "MCDG"
static constexpr uint16_t RGY_DEGRAIN_FRAME_META_VERSION = 0x0001u;

enum RGYDegrainFrameMetaFlags : uint32_t {
    RGY_DEGRAIN_FRAME_META_FLAG_NONE          = 0x00000000u,
    RGY_DEGRAIN_FRAME_META_FLAG_ANALYSIS_LUMA = 0x00000001u,
    RGY_DEGRAIN_FRAME_META_FLAG_LEVEL1_REFINE = 0x00000002u,
    RGY_DEGRAIN_FRAME_META_FLAG_CHROMA_SAD    = 0x00000004u,
};

struct RGYDegrainFrameMetaLayout {
    uint32_t blockSize;
    uint32_t overlap;
    uint32_t step;
    uint32_t search;
    uint32_t blocksX;
    uint32_t blocksY;
    uint32_t coveredWidth;
    uint32_t coveredHeight;
    uint32_t temporalDirections;
};

struct RGYDegrainFrameMetaHeader {
    uint32_t signature;
    uint16_t version;
    uint16_t headerBytes;
    uint32_t flags;
    uint32_t payloadBytes;
    uint32_t mvOffsetBytes;
    uint32_t mvCount;
    uint32_t sadOffsetBytes;
    uint32_t sadCount;
    RGYDegrainFrameMetaLayout layout;
    uint32_t reserved[7];
};

struct RGYDegrainFrameMetaView {
    const RGYDegrainFrameMetaHeader *header;
    const RGYDegrainMV *mv;
    const RGYDegrainSAD *sad;
    size_t bytes;

    bool valid() const {
        return header != nullptr && mv != nullptr && sad != nullptr;
    }
};

static_assert(sizeof(RGYDegrainFrameMetaLayout) == 36, "RGYDegrainFrameMetaLayout ABI mismatch.");
static_assert(sizeof(RGYDegrainFrameMetaHeader) == 96, "RGYDegrainFrameMetaHeader ABI mismatch.");

static constexpr int RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS = RGY_DEGRAIN_MAX_DELTA * 2;
static constexpr int RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE = 16;
static constexpr int RGY_DEGRAIN_MAX_LEVEL1_LUMA_FRAMES = RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS + 1;
using RGYDegrainRefDisableArray = std::array<bool, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS>;

struct RGYDegrainBlockLayout {
    int blockSize;
    int overlap;
    int step;
    int search;
    int blocksX;
    int blocksY;
    int coveredWidth;
    int coveredHeight;
    int temporalDirections;

    size_t blockCount() const {
        return (size_t)blocksX * (size_t)blocksY;
    }
    size_t mvCount() const {
        return blockCount() * (size_t)temporalDirections;
    }
    size_t sadCount() const {
        return blockCount() * (size_t)temporalDirections;
    }
};

struct RGYDegrainAnalyzeResult {
    uint32_t flags;
    RGYDegrainBlockLayout layout;
    CUMemBuf *mv;
    CUMemBuf *sad;
    RGYCudaEvent event;
    int frameIndex;
    int inputFrameId;
    int64_t timestamp;
    int64_t duration;
    RGYDegrainRefDisableArray availabilityDisableRefs;

    RGYDegrainAnalyzeResult() :
        flags(RGY_DEGRAIN_FRAME_META_FLAG_NONE),
        layout(),
        mv(nullptr),
        sad(nullptr),
        event(),
        frameIndex(-1),
        inputFrameId(-1),
        timestamp(0),
        duration(0),
        availabilityDisableRefs() {
        availabilityDisableRefs.fill(true);
    }

    bool valid() const {
        return mv != nullptr
            && sad != nullptr
            && layout.blockSize > 0
            && layout.step > 0
            && layout.blocksX > 0
            && layout.blocksY > 0
            && layout.temporalDirections > 0;
    }

    bool hasFrameIdentity() const {
        return frameIndex >= 0 && inputFrameId >= 0;
    }
};

struct RGYDegrainAnalyzeResultSet {
    std::array<RGYDegrainAnalyzeResult, RGY_DEGRAIN_MAX_DELTA + 1> slots;

    RGYDegrainAnalyzeResultSet() : slots() {}

    bool valid(const int delta) const {
        return delta >= 1 && delta <= RGY_DEGRAIN_MAX_DELTA && slots[delta].valid();
    }
    const RGYDegrainAnalyzeResult *get(const int delta) const {
        return valid(delta) ? &slots[delta] : nullptr;
    }
};

struct RGYDegrainWindowRampState {
    std::unique_ptr<CUMemBuf> ramp;
    size_t bytes;
    int overlapX;
    int overlapY;

    RGYDegrainWindowRampState() :
        ramp(),
        bytes(0),
        overlapX(0),
        overlapY(0) {
    }

    void reset() {
        ramp.reset();
        bytes = 0;
        overlapX = 0;
        overlapY = 0;
    }

    bool reusable(const int cmpOverlapX, const int cmpOverlapY, const size_t cmpBytes) const {
        return ramp != nullptr
            && bytes == cmpBytes
            && overlapX == cmpOverlapX
            && overlapY == cmpOverlapY;
    }
};

struct RGYDegrainCompensateInlineParams {
    const uint8_t *cur;
    int cur_pitch;
    const uint8_t *refBack;
    const uint8_t *refForw;
    int refDirBack;
    int refDirForw;
    const RGYDegrainMV *mv;
    const RGYDegrainSAD *sad;
    int blocksX;
    int blocksY;
    int blockSize;
    int overlap;
    int step;
    int coveredWidth;
    int coveredHeight;
    int planeScaleX;
    int planeScaleY;
    uint32_t thsad;
    uint32_t disableMask;
    const float *windowRamp;
    int width;
    int height;
    int refs;
    int pel;
    int subpelInterp;
};

struct RGYDegrainTemporalMixPlanState {
    std::unique_ptr<CUMemBuf> plan;
    RGYCudaEvent event;
    size_t bytes;
    uint32_t thsad;
    uint32_t disableMask;

    RGYDegrainTemporalMixPlanState() :
        plan(),
        event(),
        bytes(0),
        thsad(0),
        disableMask(0) {
    }

    void reset() {
        plan.reset();
        event.reset();
        bytes = 0;
        thsad = 0;
        disableMask = 0;
    }

    bool reusable(const size_t cmpBytes, const uint32_t cmpThsad, const uint32_t cmpDisableMask) const {
        return plan != nullptr
            && event() != nullptr
            && bytes == cmpBytes
            && thsad == cmpThsad
            && disableMask == cmpDisableMask;
    }
};

struct RGYDegrainMotionSearchConfig {
    int pixelBytes;
    int bitDepth;
    int pixelMax;
    int blockSize;
    int searchMode;
    int pel;
    int chroma;
    int cpuEmu;
    int searchParam;
    int refs;
    int trueMotion;
    int globalMotion;
    int subpelInterp;
    int width;
    int height;
    int blocksX;
    int blocksY;
    int step;
    int overlap;
    int pad;
    int motionCostScale;
    int lowSadWeightScale;
    int zeroCandidateCostScale;
    int frameAverageCandidateCostScale;
    int predictorCandidateCostScale;
    int newCandidateCostScale;
    int level;

    RGYDegrainMotionSearchConfig() :
        pixelBytes(0),
        bitDepth(0),
        pixelMax(0),
        blockSize(0),
        searchMode(0),
        pel(0),
        chroma(0),
        cpuEmu(0),
        searchParam(0),
        refs(0),
        trueMotion(0),
        globalMotion(0),
        subpelInterp(0),
        width(0),
        height(0),
        blocksX(0),
        blocksY(0),
        step(0),
        overlap(0),
        pad(0),
        motionCostScale(0),
        lowSadWeightScale(0),
        zeroCandidateCostScale(0),
        frameAverageCandidateCostScale(0),
        predictorCandidateCostScale(0),
        newCandidateCostScale(0),
        level(0) {
    }
};

struct RGYDegrainMotionSearchLevelWorkspace {
    static constexpr size_t VECTOR_BYTES = sizeof(int16_t) * 2 + sizeof(uint32_t) * 2;
    static constexpr size_t SAD_BYTES = sizeof(uint32_t);

    std::unique_ptr<CUMemBuf> vectors;
    std::unique_ptr<CUMemBuf> vectorsPrev;
    std::unique_ptr<CUMemBuf> vectorsFinal;
    std::unique_ptr<CUMemBuf> sads;
    size_t vectorsBytes;
    size_t vectorsPrevBytes;
    size_t vectorsFinalBytes;
    size_t sadsBytes;

    RGYDegrainMotionSearchLevelWorkspace() :
        vectors(),
        vectorsPrev(),
        vectorsFinal(),
        sads(),
        vectorsBytes(0),
        vectorsPrevBytes(0),
        vectorsFinalBytes(0),
        sadsBytes(0) {
    }

    void reset() {
        vectors.reset();
        vectorsPrev.reset();
        vectorsFinal.reset();
        sads.reset();
        vectorsBytes = 0;
        vectorsPrevBytes = 0;
        vectorsFinalBytes = 0;
        sadsBytes = 0;
    }
};

struct RGYDegrainMotionSearchWorkspace {
    static constexpr size_t VECTOR_BYTES = RGYDegrainMotionSearchLevelWorkspace::VECTOR_BYTES;
    static constexpr size_t SAD_BYTES = RGYDegrainMotionSearchLevelWorkspace::SAD_BYTES;
    static constexpr size_t FRAME_AVERAGE_MV_BYTES = sizeof(int32_t) * 2;

    RGYDegrainMotionSearchLevelWorkspace level0;
    RGYDegrainMotionSearchLevelWorkspace level1;
    std::unique_ptr<CUMemBuf> frameAverageMV;
    std::string buildOptionsLevel0;
    std::string buildOptionsLevel1;
    size_t frameAverageMVBytes;

    RGYDegrainMotionSearchWorkspace() :
        level0(),
        level1(),
        frameAverageMV(),
        buildOptionsLevel0(),
        buildOptionsLevel1(),
        frameAverageMVBytes(0) {
    }

    void reset() {
        level0.reset();
        level1.reset();
        frameAverageMV.reset();
        buildOptionsLevel0.clear();
        buildOptionsLevel1.clear();
        frameAverageMVBytes = 0;
    }
};

struct RGYDegrainAnalysisState {
    VppDegrainMode mode;
    RGYDegrainBlockLayout layout;
    RGYDegrainBlockLayout layoutLevel1;
    std::unique_ptr<CUMemBuf> mv;
    std::unique_ptr<CUMemBuf> sad;
    RGYDegrainWindowRampState windowRampY;
    RGYDegrainWindowRampState windowRampC;
    RGYDegrainTemporalMixPlanState temporalMixPlanY;
    RGYDegrainTemporalMixPlanState temporalMixPlanC;
    std::unique_ptr<CUMemBuf> temporalMixPrior;
    RGYDegrainMotionSearchWorkspace motionSearchWorkspace;
    std::array<std::unique_ptr<CUMemBuf>, RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE> analysisLuma;
    std::array<RGYFrameInfo, RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE> analysisLumaFrame;
    std::array<int, RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE> analysisLumaFrameNumbers;
    std::array<std::unique_ptr<CUMemBuf>, RGY_DEGRAIN_MAX_LEVEL1_LUMA_FRAMES> lumaLevel1;
    std::array<RGYCudaEvent, RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE> analysisLumaEvents;
    RGYDegrainRefDisableArray lastAvailabilityDisableRefs;
    RGYCudaEvent analysisLumaEvent;
    RGYCudaEvent event;
    size_t mvBytes;
    size_t sadBytes;
    size_t analysisLumaBytes;
    size_t lumaLevel1Bytes;
    int analysisLumaWidth;
    int analysisLumaHeight;
    int analysisLumaPitch;
    int analysisLumaGeneratedUntil;
    int lumaLevel1Width;
    int lumaLevel1Height;
    int lumaLevel1Pitch;
    int lastFrameIndex;
    int lastInputFrameId;
    int64_t lastTimestamp;
    int64_t lastDuration;

    RGYDegrainAnalysisState() :
        mode(VppDegrainMode::Source),
        layout(),
        layoutLevel1(),
        mv(),
        sad(),
        windowRampY(),
        windowRampC(),
        temporalMixPlanY(),
        temporalMixPlanC(),
        temporalMixPrior(),
        motionSearchWorkspace(),
        analysisLuma(),
        analysisLumaFrame(),
        analysisLumaFrameNumbers(),
        lumaLevel1(),
        analysisLumaEvents(),
        analysisLumaEvent(),
        event(),
        mvBytes(0),
        sadBytes(0),
        analysisLumaBytes(0),
        lumaLevel1Bytes(0),
        analysisLumaWidth(0),
        analysisLumaHeight(0),
        analysisLumaPitch(0),
        analysisLumaGeneratedUntil(-1),
        lumaLevel1Width(0),
        lumaLevel1Height(0),
        lumaLevel1Pitch(0),
        lastFrameIndex(-1),
        lastInputFrameId(-1),
        lastTimestamp(0),
        lastDuration(0) {
        analysisLumaFrameNumbers.fill(-1);
        lastAvailabilityDisableRefs.fill(true);
    }
};

class RGYDegrainBufferPool;

class RGYFrameDataDegrain : public RGYFrameData {
public:
    RGYFrameDataDegrain();
    RGYFrameDataDegrain(const RGYDegrainFrameMetaHeader &header, std::unique_ptr<CUMemBuf> mv, std::unique_ptr<CUMemBuf> sad, const RGYCudaEvent &event,
        int frameIndex, int inputFrameId, int64_t timestamp, int64_t duration, const RGYDegrainRefDisableArray &availabilityDisableRefs,
        const std::weak_ptr<RGYDegrainBufferPool> &bufferPool = {});
    virtual ~RGYFrameDataDegrain();

    const RGYDegrainFrameMetaHeader &header() const { return m_header; }
    RGYDegrainBlockLayout layout() const;
    uint32_t flags() const { return m_header.flags; }
    const CUMemBuf *mv() const { return m_mv.get(); }
    CUMemBuf *mv() { return m_mv.get(); }
    const CUMemBuf *sad() const { return m_sad.get(); }
    CUMemBuf *sad() { return m_sad.get(); }
    const RGYCudaEvent &event() const { return m_event; }
    bool valid() const;
    RGYDegrainAnalyzeResult analyzeResult() const;

private:
    RGYDegrainFrameMetaHeader m_header;
    std::unique_ptr<CUMemBuf> m_mv;
    std::unique_ptr<CUMemBuf> m_sad;
    RGYCudaEvent m_event;
    int m_frameIndex;
    int m_inputFrameId;
    int64_t m_timestamp;
    int64_t m_duration;
    RGYDegrainRefDisableArray m_availabilityDisableRefs;
    std::weak_ptr<RGYDegrainBufferPool> m_bufferPool;
};

class RGYDegrainBufferPool {
public:
    static constexpr size_t MAX_POOL_BUFFERS = 128;

    RGYDegrainBufferPool();
    ~RGYDegrainBufferPool();
    std::unique_ptr<CUMemBuf> acquire(size_t size);
    void recycle(std::unique_ptr<CUMemBuf>&& buf, const RGYCudaEvent &readyEvent);
    void clear();

private:
    struct Entry {
        std::unique_ptr<CUMemBuf> buf;
        RGYCudaEvent readyEvent;
    };

    void waitAndDropFront();

    std::deque<Entry> m_buffers;
};

uint32_t rgy_degrain_scale_sad_threshold(const VppDegrain &degrain, const RGYFrameInfo &frameInfo, int prmThreshold, bool includeChroma = false);
uint64_t rgy_degrain_scale_scene_change_block_threshold(size_t blockCount, int thscd2);
RGYDegrainBlockLayout rgy_degrain_make_block_layout(const RGYFrameInfo &frameInfo, const VppDegrain &degrain);
RGYDegrainBlockLayout rgy_degrain_make_pyramid_block_layout(const RGYFrameInfo &frameInfo, const VppDegrain &degrain);
RGYDegrainBlockLayout rgy_degrain_make_block_layout(const RGYDegrainFrameMetaLayout &layout);
int rgy_degrain_refdir_index(RGYDegrainRefDir refdir);
int rgy_degrain_temporal_direction_count(int delta);
bool rgy_degrain_refdir_from_mode(VppDegrainMode mode, RGYDegrainRefDir *refdir);
int rgy_degrain_ref_index(int delta, bool forward);
int rgy_degrain_delta_from_ref_index(int refIndex);
bool rgy_degrain_ref_index_is_forward(int refIndex);
RGYDegrainFrameMetaLayout rgy_degrain_make_frame_meta_layout(const RGYDegrainBlockLayout &layout);
size_t rgy_degrain_mv_bytes(const RGYDegrainBlockLayout &layout);
size_t rgy_degrain_sad_bytes(const RGYDegrainBlockLayout &layout);
size_t rgy_degrain_frame_meta_payload_bytes(const RGYDegrainBlockLayout &layout);
size_t rgy_degrain_frame_meta_total_bytes(const RGYDegrainBlockLayout &layout);
RGYDegrainFrameMetaHeader rgy_degrain_make_frame_meta_header(const RGYDegrainBlockLayout &layout, uint32_t flags);
RGYDegrainFrameMetaView rgy_degrain_resolve_frame_meta(const void *metaData, size_t metaBytes);
bool rgy_degrain_layout_equal(const RGYDegrainBlockLayout &lhs, const RGYDegrainBlockLayout &rhs);
std::shared_ptr<RGYFrameDataDegrain> rgy_degrain_get_frame_data(const RGYFrameInfo *frame);
RGYDegrainAnalyzeResult rgy_degrain_get_analyze_result(const RGYFrameInfo *frame);
void rgy_degrain_erase_frame_data(std::vector<std::shared_ptr<RGYFrameData>> &dataList);
RGYDegrainMotionSearchConfig rgy_degrain_make_motion_search_config(const RGYFrameInfo &frameInfo, const VppDegrain &degrain, const RGYDegrainBlockLayout &layout, int level, int pad);
std::string makeDegrainMotionSearchBuildOptions(const RGYDegrainMotionSearchConfig &cfg);
