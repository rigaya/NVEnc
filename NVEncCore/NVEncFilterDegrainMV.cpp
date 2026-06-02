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

#include "NVEncFilterDegrainMV.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sstream>

#include "rgy_frame_info.h"

namespace {
static uint32_t degrainClampMetaU32(const size_t value) {
    return (uint32_t)std::min<size_t>(value, std::numeric_limits<uint32_t>::max());
}

static bool degrainEnvFlagNotDisabled(const char *name) {
    const auto value = std::getenv(name);
    return !(value && value[0] == '0' && value[1] == '\0');
}

static bool degrainMotionSearchRefLocalCacheEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("NVENC_DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE");
    return enabled;
}

static bool degrainMotionSearchLazyRefWindowEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("NVENC_DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW");
    return enabled;
}

static bool degrainMotionSearchSpatialReusePreviousSadEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("NVENC_DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD");
    return enabled;
}
}

RGYFrameDataDegrain::RGYFrameDataDegrain() :
    RGYFrameData(),
    m_header(),
    m_mv(),
    m_sad(),
    m_event(),
    m_frameIndex(-1),
    m_inputFrameId(-1),
    m_timestamp(0),
    m_duration(0),
    m_availabilityDisableRefs(),
    m_bufferPool() {
    m_dataType = RGY_FRAME_DATA_DEGRAIN;
    m_availabilityDisableRefs.fill(true);
}

RGYFrameDataDegrain::RGYFrameDataDegrain(const RGYDegrainFrameMetaHeader &header, std::unique_ptr<CUMemBuf> mv, std::unique_ptr<CUMemBuf> sad, const RGYCudaEvent &event,
    int frameIndex, int inputFrameId, int64_t timestamp, int64_t duration, const RGYDegrainRefDisableArray &availabilityDisableRefs,
    const std::weak_ptr<RGYDegrainBufferPool> &bufferPool) :
    RGYFrameData(),
    m_header(header),
    m_mv(std::move(mv)),
    m_sad(std::move(sad)),
    m_event(event),
    m_frameIndex(frameIndex),
    m_inputFrameId(inputFrameId),
    m_timestamp(timestamp),
    m_duration(duration),
    m_availabilityDisableRefs(availabilityDisableRefs),
    m_bufferPool(bufferPool) {
    m_dataType = RGY_FRAME_DATA_DEGRAIN;
}

RGYFrameDataDegrain::~RGYFrameDataDegrain() {
    if (auto pool = m_bufferPool.lock()) {
        pool->recycle(std::move(m_mv), m_event);
        pool->recycle(std::move(m_sad), m_event);
    }
}

RGYDegrainBufferPool::RGYDegrainBufferPool() :
    m_buffers() {
}

RGYDegrainBufferPool::~RGYDegrainBufferPool() {
    clear();
}

std::unique_ptr<CUMemBuf> RGYDegrainBufferPool::acquire(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    auto pooled = std::find_if(m_buffers.begin(), m_buffers.end(), [size](const Entry& entry) {
        return entry.buf && entry.buf->nSize == size;
    });
    if (pooled != m_buffers.end()) {
        if (pooled->readyEvent() != nullptr) {
            cudaEventSynchronize(pooled->readyEvent());
            pooled->readyEvent.reset();
        }
        auto buf = std::move(pooled->buf);
        m_buffers.erase(pooled);
        return buf;
    }
    auto buf = std::make_unique<CUMemBuf>(size);
    if (buf->alloc() != RGY_ERR_NONE) {
        return nullptr;
    }
    return buf;
}

void RGYDegrainBufferPool::recycle(std::unique_ptr<CUMemBuf>&& buf, const RGYCudaEvent &readyEvent) {
    if (!buf) {
        return;
    }
    Entry entry;
    entry.buf = std::move(buf);
    entry.readyEvent = readyEvent;
    m_buffers.emplace_back(std::move(entry));
    while (m_buffers.size() > MAX_POOL_BUFFERS) {
        waitAndDropFront();
    }
}

void RGYDegrainBufferPool::waitAndDropFront() {
    if (m_buffers.empty()) {
        return;
    }
    if (m_buffers.front().readyEvent() != nullptr) {
        cudaEventSynchronize(m_buffers.front().readyEvent());
        m_buffers.front().readyEvent.reset();
    }
    m_buffers.pop_front();
}

void RGYDegrainBufferPool::clear() {
    while (!m_buffers.empty()) {
        waitAndDropFront();
    }
}

RGYDegrainBlockLayout RGYFrameDataDegrain::layout() const {
    return rgy_degrain_make_block_layout(m_header.layout);
}

bool RGYFrameDataDegrain::valid() const {
    const auto layoutValue = layout();
    return m_header.signature == RGY_DEGRAIN_FRAME_META_SIGNATURE
        && m_header.version == RGY_DEGRAIN_FRAME_META_VERSION
        && m_header.headerBytes == sizeof(RGYDegrainFrameMetaHeader)
        && m_mv != nullptr
        && m_sad != nullptr
        && m_mv->nSize == rgy_degrain_mv_bytes(layoutValue)
        && m_sad->nSize == rgy_degrain_sad_bytes(layoutValue);
}

RGYDegrainAnalyzeResult RGYFrameDataDegrain::analyzeResult() const {
    RGYDegrainAnalyzeResult result;
    if (!valid()) {
        return result;
    }
    result.flags = flags();
    result.layout = layout();
    result.mv = m_mv.get();
    result.sad = m_sad.get();
    result.event = m_event;
    result.frameIndex = m_frameIndex;
    result.inputFrameId = m_inputFrameId;
    result.timestamp = m_timestamp;
    result.duration = m_duration;
    result.availabilityDisableRefs = m_availabilityDisableRefs;
    return result;
}

uint32_t rgy_degrain_scale_sad_threshold(const VppDegrain &degrain, const RGYFrameInfo &frameInfo, const int prmThreshold, const bool includeChroma) {
    (void)includeChroma; // keep thSAD/thSCD unchanged when chroma motion is enabled
    if (prmThreshold <= 0 || degrain.blksize <= 0) {
        return 0;
    }
    const int bitdepth = std::max<int>((int)RGY_CSP_BIT_DEPTH[frameInfo.csp], 8);
    const uint64_t blockScale = (uint64_t)degrain.blksize * (uint64_t)degrain.blksize;
    const uint64_t depthScale = 1ull << std::max(bitdepth - 8, 0);
    const uint64_t scaled = ((uint64_t)prmThreshold * blockScale * depthScale) / 64ull;
    return (uint32_t)std::min<uint64_t>(scaled, std::numeric_limits<uint32_t>::max());
}

uint64_t rgy_degrain_scale_scene_change_block_threshold(const size_t blockCount, const int thscd2) {
    return ((uint64_t)std::max(thscd2, 0) * (uint64_t)blockCount) / 256ull;
}

RGYDegrainBlockLayout rgy_degrain_make_block_layout(const RGYFrameInfo &frameInfo, const VppDegrain &degrain) {
    RGYDegrainBlockLayout layout = {};
    layout.blockSize = degrain.blksize;
    layout.overlap = std::max(degrain.overlap, 0);
    layout.step = std::max(layout.blockSize - layout.overlap, 1);
    layout.search = degrain.search;
    layout.blocksX = std::max(1, std::max(frameInfo.width - layout.overlap, 0) / layout.step);
    layout.blocksY = std::max(1, std::max(frameInfo.height - layout.overlap, 0) / layout.step);
    layout.coveredWidth = std::min(frameInfo.width, layout.blocksX * layout.step + layout.overlap);
    layout.coveredHeight = std::min(frameInfo.height, layout.blocksY * layout.step + layout.overlap);
    layout.temporalDirections = rgy_degrain_temporal_direction_count(degrain.delta);
    return layout;
}

RGYDegrainBlockLayout rgy_degrain_make_pyramid_block_layout(const RGYFrameInfo &frameInfo, const VppDegrain &degrain) {
    auto levelDegrain = degrain;
    levelDegrain.search = degrain.search;

    auto levelFrame = frameInfo;
    levelFrame.width = std::max(1, (frameInfo.width + 1) / 2);
    levelFrame.height = std::max(1, (frameInfo.height + 1) / 2);
    return rgy_degrain_make_block_layout(levelFrame, levelDegrain);
}

RGYDegrainBlockLayout rgy_degrain_make_block_layout(const RGYDegrainFrameMetaLayout &layout) {
    RGYDegrainBlockLayout blockLayout = {};
    blockLayout.blockSize = (int)layout.blockSize;
    blockLayout.overlap = (int)layout.overlap;
    blockLayout.step = (int)layout.step;
    blockLayout.search = (int)layout.search;
    blockLayout.blocksX = (int)layout.blocksX;
    blockLayout.blocksY = (int)layout.blocksY;
    blockLayout.coveredWidth = (int)layout.coveredWidth;
    blockLayout.coveredHeight = (int)layout.coveredHeight;
    blockLayout.temporalDirections = (int)layout.temporalDirections;
    return blockLayout;
}

int rgy_degrain_refdir_index(const RGYDegrainRefDir refdir) {
    return static_cast<int>(refdir);
}

int rgy_degrain_temporal_direction_count(const int delta) {
    return clamp(delta, 1, RGY_DEGRAIN_MAX_DELTA) * 2;
}

bool rgy_degrain_refdir_from_mode(const VppDegrainMode mode, RGYDegrainRefDir *refdir) {
    if (!refdir) {
        return false;
    }
    switch (mode) {
    case VppDegrainMode::MotionBack:
        *refdir = RGYDegrainRefDir::Backward;
        return true;
    case VppDegrainMode::MotionForw:
        *refdir = RGYDegrainRefDir::Forward;
        return true;
    case VppDegrainMode::MotionBack2:
        *refdir = static_cast<RGYDegrainRefDir>(2);
        return true;
    case VppDegrainMode::MotionForw2:
        *refdir = static_cast<RGYDegrainRefDir>(3);
        return true;
    default:
        return false;
    }
}

int rgy_degrain_ref_index(const int delta, const bool forward) {
    if (delta < 1) {
        return 0;
    }
    return (clamp(delta, 1, RGY_DEGRAIN_MAX_DELTA) - 1) * 2 + (forward ? 1 : 0);
}

int rgy_degrain_delta_from_ref_index(const int refIndex) {
    return clamp(refIndex / 2 + 1, 1, RGY_DEGRAIN_MAX_DELTA);
}

bool rgy_degrain_ref_index_is_forward(const int refIndex) {
    return (refIndex & 1) != 0;
}

RGYDegrainFrameMetaLayout rgy_degrain_make_frame_meta_layout(const RGYDegrainBlockLayout &layout) {
    RGYDegrainFrameMetaLayout metaLayout = {};
    metaLayout.blockSize = (uint32_t)std::max(layout.blockSize, 0);
    metaLayout.overlap = (uint32_t)std::max(layout.overlap, 0);
    metaLayout.step = (uint32_t)std::max(layout.step, 0);
    metaLayout.search = (uint32_t)std::max(layout.search, 0);
    metaLayout.blocksX = (uint32_t)std::max(layout.blocksX, 0);
    metaLayout.blocksY = (uint32_t)std::max(layout.blocksY, 0);
    metaLayout.coveredWidth = (uint32_t)std::max(layout.coveredWidth, 0);
    metaLayout.coveredHeight = (uint32_t)std::max(layout.coveredHeight, 0);
    metaLayout.temporalDirections = (uint32_t)std::max(layout.temporalDirections, 0);
    return metaLayout;
}

size_t rgy_degrain_mv_bytes(const RGYDegrainBlockLayout &layout) {
    return layout.mvCount() * sizeof(RGYDegrainMV);
}

size_t rgy_degrain_sad_bytes(const RGYDegrainBlockLayout &layout) {
    return layout.sadCount() * sizeof(RGYDegrainSAD);
}

size_t rgy_degrain_frame_meta_payload_bytes(const RGYDegrainBlockLayout &layout) {
    return rgy_degrain_mv_bytes(layout) + rgy_degrain_sad_bytes(layout);
}

size_t rgy_degrain_frame_meta_total_bytes(const RGYDegrainBlockLayout &layout) {
    return sizeof(RGYDegrainFrameMetaHeader) + rgy_degrain_frame_meta_payload_bytes(layout);
}

RGYDegrainFrameMetaHeader rgy_degrain_make_frame_meta_header(const RGYDegrainBlockLayout &layout, const uint32_t flags) {
    RGYDegrainFrameMetaHeader header = {};
    const auto mvBytes = rgy_degrain_mv_bytes(layout);
    const auto sadBytes = rgy_degrain_sad_bytes(layout);
    header.signature = RGY_DEGRAIN_FRAME_META_SIGNATURE;
    header.version = RGY_DEGRAIN_FRAME_META_VERSION;
    header.headerBytes = (uint16_t)sizeof(RGYDegrainFrameMetaHeader);
    header.flags = flags;
    header.payloadBytes = degrainClampMetaU32(mvBytes + sadBytes);
    header.mvOffsetBytes = (uint32_t)sizeof(RGYDegrainFrameMetaHeader);
    header.mvCount = degrainClampMetaU32(layout.mvCount());
    header.sadOffsetBytes = degrainClampMetaU32((size_t)header.mvOffsetBytes + mvBytes);
    header.sadCount = degrainClampMetaU32(layout.sadCount());
    header.layout = rgy_degrain_make_frame_meta_layout(layout);
    return header;
}

RGYDegrainFrameMetaView rgy_degrain_resolve_frame_meta(const void *metaData, const size_t metaBytes) {
    RGYDegrainFrameMetaView view = {};
    if (metaData == nullptr || metaBytes < sizeof(RGYDegrainFrameMetaHeader)) {
        return view;
    }

    const auto *header = reinterpret_cast<const RGYDegrainFrameMetaHeader *>(metaData);
    if (header->signature != RGY_DEGRAIN_FRAME_META_SIGNATURE
        || header->version != RGY_DEGRAIN_FRAME_META_VERSION
        || header->headerBytes != sizeof(RGYDegrainFrameMetaHeader)) {
        return view;
    }

    const auto mvBytes = (size_t)header->mvCount * sizeof(RGYDegrainMV);
    const auto sadBytes = (size_t)header->sadCount * sizeof(RGYDegrainSAD);
    if ((size_t)header->mvOffsetBytes != sizeof(RGYDegrainFrameMetaHeader)
        || (size_t)header->sadOffsetBytes != (size_t)header->mvOffsetBytes + mvBytes
        || (size_t)header->payloadBytes != mvBytes + sadBytes) {
        return view;
    }

    const size_t requiredBytes = (size_t)header->headerBytes + (size_t)header->payloadBytes;
    const size_t expectedCount = (size_t)header->layout.blocksX * (size_t)header->layout.blocksY * (size_t)header->layout.temporalDirections;
    if (metaBytes < requiredBytes
        || (size_t)header->mvCount != expectedCount
        || (size_t)header->sadCount != expectedCount) {
        return view;
    }

    const auto *base = reinterpret_cast<const uint8_t *>(metaData);
    view.header = header;
    view.mv = reinterpret_cast<const RGYDegrainMV *>(base + header->mvOffsetBytes);
    view.sad = reinterpret_cast<const RGYDegrainSAD *>(base + header->sadOffsetBytes);
    view.bytes = requiredBytes;
    return view;
}

bool rgy_degrain_layout_equal(const RGYDegrainBlockLayout &lhs, const RGYDegrainBlockLayout &rhs) {
    return lhs.blockSize == rhs.blockSize
        && lhs.overlap == rhs.overlap
        && lhs.step == rhs.step
        && lhs.search == rhs.search
        && lhs.blocksX == rhs.blocksX
        && lhs.blocksY == rhs.blocksY
        && lhs.coveredWidth == rhs.coveredWidth
        && lhs.coveredHeight == rhs.coveredHeight
        && lhs.temporalDirections == rhs.temporalDirections;
}

std::shared_ptr<RGYFrameDataDegrain> rgy_degrain_get_frame_data(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        auto degrain = std::dynamic_pointer_cast<RGYFrameDataDegrain>(data);
        return degrain != nullptr && degrain->valid();
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<RGYFrameDataDegrain>(*frameData);
}

RGYDegrainAnalyzeResult rgy_degrain_get_analyze_result(const RGYFrameInfo *frame) {
    const auto frameData = rgy_degrain_get_frame_data(frame);
    return frameData ? frameData->analyzeResult() : RGYDegrainAnalyzeResult();
}

void rgy_degrain_erase_frame_data(std::vector<std::shared_ptr<RGYFrameData>> &dataList) {
    dataList.erase(std::remove_if(dataList.begin(), dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataDegrain>(data) != nullptr;
    }), dataList.end());
}

RGYDegrainMotionSearchConfig rgy_degrain_make_motion_search_config(
    const RGYFrameInfo &frameInfo,
    const VppDegrain &degrain,
    const RGYDegrainBlockLayout &layout,
    const int level,
    const int pad) {
    RGYDegrainMotionSearchConfig cfg;
    cfg.pixelBytes = (RGY_CSP_BIT_DEPTH[frameInfo.csp] > 8) ? 2 : 1;
    cfg.bitDepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    cfg.pixelMax = (cfg.bitDepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << cfg.bitDepth) - 1);
    cfg.blockSize = layout.blockSize;
    cfg.searchMode = layout.search;
    cfg.pel = degrain.pel;
    cfg.chroma = 0;
    cfg.cpuEmu = 1;
    cfg.searchParam = degrain.searchParam;
    cfg.refs = layout.temporalDirections;
    cfg.trueMotion = degrain.trueMotion ? 1 : 0;
    cfg.globalMotion = degrain.globalMotion ? 1 : 0;
    cfg.subpelInterp = degrain.subpelInterp;
    cfg.width = frameInfo.width;
    cfg.height = frameInfo.height;
    cfg.blocksX = layout.blocksX;
    cfg.blocksY = layout.blocksY;
    cfg.step = layout.step;
    cfg.overlap = layout.overlap;
    cfg.pad = pad;
    cfg.motionCostScale = degrain.lambda;
    auto scaledDegrain = degrain;
    scaledDegrain.blksize = layout.blockSize;
    scaledDegrain.overlap = layout.overlap;
    cfg.lowSadWeightScale = (int)rgy_degrain_scale_sad_threshold(scaledDegrain, frameInfo, degrain.lsad);
    cfg.zeroCandidateCostScale = degrain.pnew;
    cfg.frameAverageCandidateCostScale = 0;
    cfg.predictorCandidateCostScale = degrain.plevel;
    cfg.newCandidateCostScale = degrain.pnew;
    cfg.level = level;
    return cfg;
}

std::string makeDegrainMotionSearchBuildOptions(const RGYDegrainMotionSearchConfig &cfg) {
    std::ostringstream options;
    options
        << "-D TypePixel=" << ((cfg.pixelBytes > 1) ? "ushort" : "uchar")
        << " -D DEGRAIN_PIXEL_TYPE=" << ((cfg.pixelBytes > 1) ? "ushort" : "uchar")
        << " -D DEGRAIN_PIXEL_BYTES=" << cfg.pixelBytes
        << " -D DEGRAIN_BIT_DEPTH=" << cfg.bitDepth
        << " -D DEGRAIN_PIXEL_MAX=" << cfg.pixelMax
        << " -D DEGRAIN_BLK_SIZE=" << cfg.blockSize
        << " -D DEGRAIN_BLOCK_SIZE=" << cfg.blockSize
        << " -D DEGRAIN_SEARCH=" << cfg.searchMode
        << " -D DEGRAIN_SEARCH_MODE=" << cfg.searchMode
        << " -D DEGRAIN_PEL=" << cfg.pel
        << " -D DEGRAIN_NPEL=" << cfg.pel
        << " -D DEGRAIN_CHROMA=" << (cfg.chroma ? 1 : 0)
        << " -D DEGRAIN_CPU_EMU=" << (cfg.cpuEmu ? 1 : 0)
        << " -D DEGRAIN_SEARCH_PARAM=" << cfg.searchParam
        << " -D DEGRAIN_REFS=" << cfg.refs
        << " -D DEGRAIN_TRUE_MOTION=" << (cfg.trueMotion ? 1 : 0)
        << " -D DEGRAIN_GLOBAL_MOTION=" << (cfg.globalMotion ? 1 : 0)
        << " -D DEGRAIN_SUBPEL_INTERP=" << cfg.subpelInterp
        << " -D DEGRAIN_WIDTH=" << cfg.width
        << " -D DEGRAIN_HEIGHT=" << cfg.height
        << " -D DEGRAIN_BLOCKS_X=" << cfg.blocksX
        << " -D DEGRAIN_BLOCKS_Y=" << cfg.blocksY
        << " -D DEGRAIN_STEP=" << cfg.step
        << " -D DEGRAIN_OVERLAP=" << cfg.overlap
        << " -D DEGRAIN_PAD=" << cfg.pad
        << " -D DEGRAIN_MOTION_COST_SCALE=" << cfg.motionCostScale
        << " -D DEGRAIN_LOW_SAD_WEIGHT_SCALE=" << cfg.lowSadWeightScale
        << " -D DEGRAIN_ZERO_CANDIDATE_COST_SCALE=" << cfg.zeroCandidateCostScale
        << " -D DEGRAIN_FRAME_AVERAGE_CANDIDATE_COST_SCALE=" << cfg.frameAverageCandidateCostScale
        << " -D DEGRAIN_PREDICTOR_CANDIDATE_COST_SCALE=" << cfg.predictorCandidateCostScale
        << " -D DEGRAIN_NEW_CANDIDATE_COST_SCALE=" << cfg.newCandidateCostScale
        << " -D DEGRAIN_LEVEL=" << cfg.level
        << " -D DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE=" << (degrainMotionSearchRefLocalCacheEnabled() ? 1 : 0)
        << " -D DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW=" << (degrainMotionSearchLazyRefWindowEnabled() ? 1 : 0)
        << " -D DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD=" << (degrainMotionSearchSpatialReusePreviousSadEnabled() ? 1 : 0);
    return options.str();
}
