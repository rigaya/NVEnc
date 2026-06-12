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
#include <cstdint>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "NVEncFilter.h"

static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_WIDE_CORE = 1 << 0;
static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_CORE_BLEND = 1 << 1;
static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_RANK_LIMIT = 1 << 2;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE = 1 << 0;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH = 1 << 1;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_ENABLED = 1 << 2;
static constexpr int RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL = 0;
static constexpr int RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL = 7;
static constexpr int RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL = 0;
static constexpr int RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL = 3;
static constexpr int RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT =
    RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL - RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL + 1;
static constexpr int RGY_RTGMC_REPAIR_RESTORE_PADDING_LEVEL_COUNT =
    RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL - RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL + 1;

struct RGYRtgmcRepairProfile {
    uint8_t thinRejectLevel;
    uint8_t restorePaddingLevel;
    uint8_t thinRejectFlags;
    uint8_t restoreFlags;
};
static_assert(sizeof(RGYRtgmcRepairProfile) == sizeof(uint32_t), "RGYRtgmcRepairProfile must fit in one 32-bit value.");

inline int rgy_rtgmc_repair_clamp(const int value, const int minValue, const int maxValue) {
    return (value < minValue) ? minValue : (value > maxValue) ? maxValue : value;
}

inline bool rgy_rtgmc_repair_thin_level_is_valid(const int thinRejectLevel) {
    return thinRejectLevel >= RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL
        && thinRejectLevel <= RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL;
}

inline bool rgy_rtgmc_repair_pad_level_is_valid(const int restorePaddingLevel) {
    return restorePaddingLevel >= RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL
        && restorePaddingLevel <= RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL;
}

inline bool rgy_rtgmc_repair_levels_are_valid(const int thinRejectLevel, const int restorePaddingLevel) {
    return rgy_rtgmc_repair_thin_level_is_valid(thinRejectLevel)
        && rgy_rtgmc_repair_pad_level_is_valid(restorePaddingLevel);
}

inline RGYRtgmcRepairProfile rgy_rtgmc_repair_profile_from_levels(const int thinLevel, const int padLevel) {
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> thinFlagsByLevel = {
        0,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND | RGY_RTGMC_REPAIR_THIN_RANK_LIMIT,
        0,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND | RGY_RTGMC_REPAIR_THIN_RANK_LIMIT,
        RGY_RTGMC_REPAIR_THIN_WIDE_CORE,
        RGY_RTGMC_REPAIR_THIN_WIDE_CORE | RGY_RTGMC_REPAIR_THIN_CORE_BLEND
    };
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> restoreFlagsByLevel = {
        0,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE
    };
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> noPaddingRestoreFlagsByLevel = {
        0,
        0,
        0,
        0,
        RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH,
        0,
        0,
        0
    };

    const int thinRejectLevel = rgy_rtgmc_repair_clamp(
        thinLevel,
        RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL,
        RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL);
    const int restorePaddingLevel = rgy_rtgmc_repair_clamp(
        padLevel,
        RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL,
        RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL);
    return RGYRtgmcRepairProfile {
        (uint8_t)thinRejectLevel,
        (uint8_t)restorePaddingLevel,
        thinFlagsByLevel[thinRejectLevel],
        (uint8_t)(restoreFlagsByLevel[thinRejectLevel]
            | ((restorePaddingLevel == 0) ? noPaddingRestoreFlagsByLevel[thinRejectLevel] : 0))
    };
}

inline uint32_t rgy_rtgmc_repair_profile_pack(const RGYRtgmcRepairProfile& profile) {
    return (uint32_t)profile.thinRejectLevel
        | ((uint32_t)profile.restorePaddingLevel << 8)
        | ((uint32_t)profile.thinRejectFlags << 16)
        | ((uint32_t)profile.restoreFlags << 24);
}

class NVEncFilterResizePlaneProxy;

class NVEncFilterParamRtgmcSearchPrefilter : public NVEncFilterParam {
public:
    int tr0;
    int searchRefine;
    int rep0Thin;
    int rep0Pad;
    RGYRtgmcRepairProfile repairProfile;
    bool tvRange;
    bool chromaMotion;
    bool attachSearchLuma;
    tstring dumpY4m;
    tstring dumpStage;
    int dumpMaxFrames;

    NVEncFilterParamRtgmcSearchPrefilter() : tr0(1), searchRefine(1), rep0Thin(1), rep0Pad(0), repairProfile(), tvRange(true), chromaMotion(false), attachSearchLuma(false), dumpY4m(), dumpStage(), dumpMaxFrames(0) {}
    virtual ~NVEncFilterParamRtgmcSearchPrefilter() {}
    virtual tstring print() const override;
};

class RGYFrameDataRtgmcSearchLuma : public RGYFrameData {
public:
    RGYFrameDataRtgmcSearchLuma(std::shared_ptr<CUFrameBuf> frame, int bitdepth);
    virtual ~RGYFrameDataRtgmcSearchLuma() {}
    const RGYFrameInfo *frame() const;
    CUFrameBuf *cuFrame() const { return m_frame.get(); }
    int bitdepth() const { return m_bitdepth; }
protected:
    std::shared_ptr<CUFrameBuf> m_frame;
    int m_bitdepth;
};

class NVEncFilterRtgmcSearchPrefilter : public NVEncFilter {
public:
    static constexpr int RTGMC_SEARCH_PREFILTER_CACHE_SIZE = 5;

    NVEncFilterRtgmcSearchPrefilter();
    virtual ~NVEncFilterRtgmcSearchPrefilter();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

protected:
    struct SearchRefine1PlaneResources {
        std::unique_ptr<CUFrameBuf> motionGuide;
        std::unique_ptr<CUFrameBuf> halfSearchBase;
        std::unique_ptr<CUFrameBuf> halfSearchSmoothed;

        void clear() {
            motionGuide.reset();
            halfSearchBase.reset();
            halfSearchSmoothed.reset();
        }
    };
    struct SearchRefine2PlaneResources {
        std::unique_ptr<CUFrameBuf> motionGuide;
        std::unique_ptr<CUFrameBuf> searchSmoothed3x3;
        std::unique_ptr<CUFrameBuf> edgeSoftenedSearch;
        std::unique_ptr<CUFrameBuf> preStabilizedSearch;

        void clear() {
            motionGuide.reset();
            searchSmoothed3x3.reset();
            edgeSoftenedSearch.reset();
            preStabilizedSearch.reset();
        }
    };

    struct SharedFramePool : public std::enable_shared_from_this<SharedFramePool> {
        struct Entry {
            std::unique_ptr<CUFrameBuf> frame;
            std::unique_ptr<cudaEvent_t, cudaevent_deleter> readyEvent;
        };
        std::deque<Entry> frames;

        SharedFramePool() : frames() {};
        std::shared_ptr<CUFrameBuf> get(const RGYFrameInfo &frameInfo);
        void recycle(CUFrameBuf *frame);
        void clear();
    };
    struct PendingSceneChangePlane {
        RGY_PLANE plane;
        tstring planeName;
        int smoothRadius;
        int groupCount;
        uint64_t sceneThreshold;
        std::array<int, 4> flags;
        std::unique_ptr<CUMemBufPair> partial;
        std::unique_ptr<cudaEvent_t, cudaevent_deleter> mapEvent;
        bool mapSubmitted;

        PendingSceneChangePlane() :
            plane(RGY_PLANE_Y), planeName(), smoothRadius(0), groupCount(0), sceneThreshold(0), flags(),
            partial(), mapEvent(), mapSubmitted(false) {
            flags.fill(0);
        }
    };
    struct PendingSearchPrefilterFrame {
        int currentFrame;
        std::array<std::shared_ptr<CUFrameBuf>, RTGMC_SEARCH_PREFILTER_CACHE_SIZE> refs;
        std::vector<PendingSceneChangePlane> scenePlanes;

        PendingSearchPrefilterFrame() : currentFrame(-1), refs(), scenePlanes() {}
    };

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
protected:

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcSearchPrefilter> &prm);
    RGY_ERR buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcSearchPrefilter> &prm);
    RGY_ERR allocCacheFrames(const RGYFrameInfo &frameInfo);
    RGY_ERR setupSearchRefine1Resources(const RGYFrameInfo &frameInfo, bool processChroma);
    RGY_ERR setupSearchRefine2Resources(const RGYFrameInfo &frameInfo, bool processChroma);
    std::shared_ptr<CUFrameBuf> createSearchLumaFrame(const RGYFrameInfo &frameInfo, bool includeChroma);
    std::unique_ptr<CUFrameBuf> createPlaneFrame(const RGYFrameInfo &frameInfo);
    RGY_ERR pushCacheFrame(const RGYFrameInfo *pInputFrame, cudaStream_t stream);
    RGY_ERR checkSameResolutionPlanePitches(const TCHAR *stageName, const std::vector<const RGYFrameInfo *> &planes);
    RGY_ERR checkTemporalPlanePitches(const TCHAR *planeName,
        const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2);
    std::unique_ptr<CUMemBufPair> getSceneChangeBuffer(size_t requiredSize);
    void recycleSceneChangeBuffer(std::unique_ptr<CUMemBufPair> &&buf);
    RGY_ERR submitSceneChangePlane(PendingSceneChangePlane *pending,
        const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2,
        RGY_PLANE plane, const TCHAR *planeName, int smoothRadius, cudaStream_t stream);
    RGY_ERR resolveSceneChangePlane(PendingSceneChangePlane *pending);
    RGY_ERR submitPendingSearchPrefilterFrame(int currentFrame, cudaStream_t stream);
    RGY_ERR resolvePendingSearchPrefilterFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream);
    void clearPendingSearchPrefilterFrames();
    std::array<int, 4> sceneChangeFlagsForPlane(const PendingSearchPrefilterFrame &pending, RGY_PLANE plane) const;
    RGY_ERR emitPrefilteredFrame(PendingSearchPrefilterFrame &pending, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream);
    RGY_ERR initSearchLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcSearchPrefilter &prm);
    RGY_ERR dumpSearchLumaFrame(CUFrameBuf *searchLuma, const RGYFrameInfo &sourceFrame,
        cudaStream_t stream);
    RGY_ERR dumpSearchYuvFrame(const RGYFrameInfo &yFrame, const RGYFrameInfo *chromaFrame,
        cudaStream_t stream);
    const RGYFrameInfo *resolveCacheFrame(int frameIndex) const;
    std::shared_ptr<CUFrameBuf> resolveCacheFrameShared(int frameIndex) const;
    int cacheIndex(int frame) const;
    int outputDelay() const;
    int drainFrameCount() const;

    std::array<std::shared_ptr<CUFrameBuf>, RTGMC_SEARCH_PREFILTER_CACHE_SIZE> m_cacheFrames;
    std::deque<std::unique_ptr<CUMemBufPair>> m_sceneChangeBufferPool;
    std::deque<PendingSearchPrefilterFrame> m_pendingSearchPrefilterFrames;
    std::array<SearchRefine1PlaneResources, 2> m_searchRefine1PlaneResources;
    std::array<SearchRefine2PlaneResources, 2> m_searchRefine2PlaneResources;
    std::array<std::unique_ptr<NVEncFilterResizePlaneProxy>, 2> m_searchRefine1ResizeDown;
    std::array<std::unique_ptr<NVEncFilterResizePlaneProxy>, 2> m_searchRefine1ResizeUp;
    std::array<std::unique_ptr<NVEncFilterResizePlaneProxy>, 2> m_searchRefine2ResizeEdgeSoftenedSearch;
    std::shared_ptr<SharedFramePool> m_cacheFramePool;
    std::shared_ptr<SharedFramePool> m_searchLumaPool;
    std::string m_buildOptions;
    std::ofstream m_searchLumaDump;
    std::string m_searchLumaDumpPath;
    std::string m_searchLumaDumpStage;
    int m_searchLumaDumpMaxFrames;
    int m_searchLumaDumpFrameCount;
    bool m_searchLumaDumpEnabled;
    bool m_searchLumaDumpHeaderWritten;
    int m_inputCount;
    int m_drainCount;
};
