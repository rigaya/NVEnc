#include "NVEncFilterRtgmcShimmerRepair.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

#include "rgy_cuda_util.h"

namespace {
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_X = 32;
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_Y = 8;
static constexpr int RTGMC_SHIMMER_REPAIR_STAGE_Y_OFFSET = 2;
static constexpr int RTGMC_SHIMMER_REPAIR_STAGE_BUFFER_COUNT = 4;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT = 0;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_DELTA = 1;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE = 2;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE = 3;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP = 6;
static constexpr int RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP = 7;

enum RtgmcShimmerRepairStageBuffer : int {
    RTGMC_SHIMMER_REPAIR_STAGE_VC_POS = 0,
    RTGMC_SHIMMER_REPAIR_STAGE_VC_NEG,
    RTGMC_SHIMMER_REPAIR_STAGE_LC_POS,
    RTGMC_SHIMMER_REPAIR_STAGE_LC_NEG,
};

static bool rtgmcShimmerRepairEnvFlagNotDisabled(const char *name) {
    const auto value = std::getenv(name);
    return !(value && value[0] == '0' && value[1] == '\0');
}

static bool rtgmcShimmerRepairStagedEnabled() {
    static const bool enabled = rtgmcShimmerRepairEnvFlagNotDisabled("NVENC_RTGMC_SHIMMER_REPAIR_STAGED");
    return enabled;
}

static const char *rtgmcShimmerRepairTargetName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? "rep1" : "rep2";
}

static const TCHAR *rtgmcShimmerRepairStageName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? _T("pre-retouch") : _T("post-tr2");
}

static void rtgmcShimmerRepairLoadProfile(NVEncFilterParamRtgmcShimmerRepair *prm) {
    prm->repairProfile = rgy_rtgmc_repair_profile_from_levels(prm->repairThin, prm->repairPad);
}

static RGY_ERR rtgmcShimmerRepairWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto& waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            const auto sts = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR rtgmcShimmerRepairRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto sts = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}
}

tstring NVEncFilterParamRtgmcShimmerRepair::print() const {
    return strsprintf(_T("rtgmc-shimmer-repair: stage=%s repair-thin=%d repair-pad=%d process_chroma=%s"),
        rtgmcShimmerRepairStageName(stage), repairThin, repairPad, processChroma ? _T("true") : _T("false"));
}

NVEncFilterRtgmcShimmerRepair::NVEncFilterRtgmcShimmerRepair() :
    NVEncFilter(),
    m_buildOptions(),
    m_lumaDump(),
    m_lumaDumpPath(),
    m_lumaDumpStage("shimmer_corrected"),
    m_lumaDumpTarget(),
    m_lumaDumpMaxFrames(0),
    m_lumaDumpFrameCount(0),
    m_lumaDumpEnabled(false),
    m_lumaDumpHeaderWritten(false),
    m_lumaDumpFullYuv(false),
    m_useKernel(false),
    m_useStagedThin4Pad0(false),
    m_stagedBuffers(),
    m_stagedPitch(0),
    m_stagedHeight(0) {
    m_name = _T("rtgmc-shimmer-repair");
}

NVEncFilterRtgmcShimmerRepair::~NVEncFilterRtgmcShimmerRepair() {
    close();
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto dataType = RGY_CSP_DATA_TYPE[prm->frameOut.csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (!rgy_rtgmc_repair_thin_level_is_valid(prm->repairThin)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-thin must be 0-7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!rgy_rtgmc_repair_pad_level_is_valid(prm->repairPad)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-pad must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcShimmerRepair> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm->repairProfile;
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d -D rtgmc_shimmer_repair_block_x=%d -D rtgmc_shimmer_repair_block_y=%d"
        " -D RTGMC_SHIMMER_REPAIR_THIN_LEVEL=%d -D RTGMC_SHIMMER_REPAIR_PAD_LEVEL=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf,
        RTGMC_SHIMMER_REPAIR_BLOCK_X,
        RTGMC_SHIMMER_REPAIR_BLOCK_Y,
        profile.thinRejectLevel,
        profile.restorePaddingLevel);
    AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernel for rtgmc-shimmer-repair: %s\n"),
        char_to_tstring(m_buildOptions).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcShimmerRepair &prm) {
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpFullYuv = false;
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_YUV_DUMP_Y4M");
    if (dumpPathEnv != nullptr && dumpPathEnv[0] != '\0') {
        m_lumaDumpFullYuv = true;
    } else {
        dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M");
    }
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        return RGY_ERR_NONE;
    }
    m_lumaDumpPath = dumpPathEnv;

    if (const char *stageEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_STAGE"); stageEnv != nullptr && stageEnv[0] != '\0') {
        m_lumaDumpStage = stageEnv;
        std::transform(m_lumaDumpStage.begin(), m_lumaDumpStage.end(), m_lumaDumpStage.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    if (m_lumaDumpStage != "correction_delta" && m_lumaDumpStage != "positive_correction_gate"
        && m_lumaDumpStage != "negative_correction_gate" && m_lumaDumpStage != "shimmer_corrected") {
        AddMessage(RGY_LOG_ERROR, _T("unsupported rtgmc rep luma dump stage: %s.\n"),
            char_to_tstring(m_lumaDumpStage).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (const char *targetEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_TARGET"); targetEnv != nullptr && targetEnv[0] != '\0') {
        m_lumaDumpTarget = targetEnv;
        std::transform(m_lumaDumpTarget.begin(), m_lumaDumpTarget.end(), m_lumaDumpTarget.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    const char *activeTarget = (prm.repairThin > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";
    if (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != activeTarget) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc rep luma dump target %s skipped for inactive %s instance.\n"),
            char_to_tstring(m_lumaDumpTarget).c_str(), char_to_tstring(activeTarget).c_str());
        return RGY_ERR_NONE;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep stage dump supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_YUV_DUMP_Y4M supports only 4:2:0 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frameInfo.csp] != 1) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M supports only 4:2:0/Y8 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    const char *maxFrames = std::getenv(m_lumaDumpFullYuv
        ? "QSVENC_RTGMC_REP_YUV_DUMP_MAX_FRAMES"
        : "QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    if (maxFrames == nullptr || maxFrames[0] == '\0') {
        maxFrames = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    }
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_lumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }

    m_lumaDump.open(m_lumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc rep %s dump enabled: %s (target=%s, stage=%s).\n"),
        m_lumaDumpFullYuv ? _T("yuv") : _T("luma"),
        char_to_tstring(m_lumaDumpPath).c_str(), char_to_tstring(activeTarget).c_str(), char_to_tstring(m_lumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_lumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpMaxFrames > 0 && m_lumaDumpFrameCount >= m_lumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    if (frame == nullptr || frame->ptr[0] == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    if (bitdepth > 8 || (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420)
        || (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep luma dump disabled by unsupported frame csp: %s.\n"),
            RGY_CSP_NAMES[frame->csp]);
        m_lumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }

    auto sts = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    CUFrameBuf hostFrame(frame->width, frame->height, frame->csp);
    hostFrame.frame.mem_type = RGY_MEM_TYPE_CPU;
    sts = hostFrame.allocHost();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc rep luma dump host buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = copyFrameAsync(&hostFrame.frame, frame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc rep luma dump frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = err_to_rgy(cudaStreamSynchronize(stream));
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc rep luma dump read: %s.\n"), get_err_mes(sts));
        return sts;
    }

    if (!m_lumaDumpHeaderWritten) {
        m_lumaDump << "YUV4MPEG2 W" << hostFrame.frame.width << " H" << hostFrame.frame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_lumaDumpHeaderWritten = true;
    }
    m_lumaDump << "FRAME\n";
    for (int y = 0; y < hostFrame.frame.height; y++) {
        m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[0] + (size_t)y * hostFrame.frame.pitch[0]), hostFrame.frame.width);
    }
    const int chromaWidth = (hostFrame.frame.width + 1) >> 1;
    const int chromaHeight = (hostFrame.frame.height + 1) >> 1;
    if (m_lumaDumpFullYuv) {
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[1] + (size_t)y * hostFrame.frame.pitch[1]), chromaWidth);
        }
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.frame.ptr[2] + (size_t)y * hostFrame.frame.pitch[2]), chromaWidth);
        }
    } else {
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_lumaDumpEnabled || m_lumaDumpStage != stage || (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != target)) {
        return RGY_ERR_NONE;
    }
    return dumpLumaFrame(frame, stream, wait_events);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairFused(
    RGYFrameInfo *pOutputFrame,
    RGYFrameInfo *pCorrectionDeltaFrame,
    RGYFrameInfo *pPositiveCorrectionGateFrame,
    RGYFrameInfo *pNegativeCorrectionGateFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    int iplane, cudaStream_t stream) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto deltaPlane = getPlane(pCorrectionDeltaFrame, (RGY_PLANE)iplane);
    const auto positivePlane = getPlane(pPositiveCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto negativePlane = getPlane(pNegativeCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm.repairProfile;
    const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    const dim3 gridSize(divCeil(outPlane.width, blockSize.x), divCeil(outPlane.height, blockSize.y));
    const bool launched = (bitdepth <= 8)
        ? launchRtgmcShimmerRepairFusedU8(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (uint8_t *)deltaPlane.ptr[0], deltaPlane.pitch[0],
            (uint8_t *)positivePlane.ptr[0], positivePlane.pitch[0],
            (uint8_t *)negativePlane.ptr[0], negativePlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal)
        : launchRtgmcShimmerRepairFusedU16(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (uint8_t *)deltaPlane.ptr[0], deltaPlane.pitch[0],
            (uint8_t *)positivePlane.ptr[0], positivePlane.pitch[0],
            (uint8_t *)negativePlane.ptr[0], negativePlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal);
    if (!launched) {
        AddMessage(RGY_LOG_ERROR, _T("invalid rtgmc-shimmer-repair profile: thin=%d pad=%d.\n"),
            profile.thinRejectLevel, profile.restorePaddingLevel);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto err = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply_fused"), iplane, get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairApply(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    int iplane, cudaStream_t stream) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm.repairProfile;
    const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    const dim3 gridSize(divCeil(outPlane.width, blockSize.x), divCeil(outPlane.height, blockSize.y));
    const bool launched = (bitdepth <= 8)
        ? launchRtgmcShimmerRepairApplyU8(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal)
        : launchRtgmcShimmerRepairApplyU16(
            profile.thinRejectLevel, profile.restorePaddingLevel, gridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            outPlane.width, outPlane.height,
            rangeHalf, maxVal);
    if (!launched) {
        AddMessage(RGY_LOG_ERROR, _T("invalid rtgmc-shimmer-repair profile: thin=%d pad=%d.\n"),
            profile.thinRejectLevel, profile.restorePaddingLevel);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        auto err = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply"), iplane, get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairStaged(
    RGYFrameInfo *pOutputFrame,
    RGYFrameInfo *pCorrectionDeltaFrame,
    RGYFrameInfo *pPositiveCorrectionGateFrame,
    RGYFrameInfo *pNegativeCorrectionGateFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    int iplane, cudaStream_t stream) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    const int stagedHeight = outPlane.height + RTGMC_SHIMMER_REPAIR_STAGE_Y_OFFSET * 2;
    const int pixelBytes = (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) ? 2 : 1;
    if (m_stagedPitch < outPlane.width * pixelBytes || m_stagedHeight < stagedHeight) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc shimmer repair staged buffer is too small for plane %d.\n"), iplane);
        return RGY_ERR_MEMORY_ALLOC;
    }

    uint8_t *correctionDelta = nullptr;
    uint8_t *positiveCorrectionGate = nullptr;
    uint8_t *negativeCorrectionGate = nullptr;
    int correctionDeltaPitch = 0;
    int positiveCorrectionGatePitch = 0;
    int negativeCorrectionGatePitch = 0;
    if (pCorrectionDeltaFrame != nullptr
        && pPositiveCorrectionGateFrame != nullptr
        && pNegativeCorrectionGateFrame != nullptr) {
        const auto deltaPlane = getPlane(pCorrectionDeltaFrame, (RGY_PLANE)iplane);
        const auto positivePlane = getPlane(pPositiveCorrectionGateFrame, (RGY_PLANE)iplane);
        const auto negativePlane = getPlane(pNegativeCorrectionGateFrame, (RGY_PLANE)iplane);
        correctionDelta = (uint8_t *)deltaPlane.ptr[0];
        correctionDeltaPitch = deltaPlane.pitch[0];
        positiveCorrectionGate = (uint8_t *)positivePlane.ptr[0];
        positiveCorrectionGatePitch = positivePlane.pitch[0];
        negativeCorrectionGate = (uint8_t *)negativePlane.ptr[0];
        negativeCorrectionGatePitch = negativePlane.pitch[0];
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    const dim3 gridSize(divCeil(outPlane.width, blockSize.x), divCeil(outPlane.height, blockSize.y));
    const dim3 stagedGridSize(divCeil(outPlane.width, blockSize.x), divCeil(stagedHeight, blockSize.y));
    const bool launched = (bitdepth <= 8)
        ? launchRtgmcShimmerRepairStagedU8(
            gridSize, stagedGridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            correctionDelta, correctionDeltaPitch,
            positiveCorrectionGate, positiveCorrectionGatePitch,
            negativeCorrectionGate, negativeCorrectionGatePitch,
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_VC_POS].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_VC_NEG].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_LC_POS].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_LC_NEG].ptr,
            m_stagedPitch, outPlane.width, outPlane.height,
            RTGMC_SHIMMER_REPAIR_STAGE_Y_OFFSET, rangeHalf, maxVal)
        : launchRtgmcShimmerRepairStagedU16(
            gridSize, stagedGridSize, blockSize, stream,
            (uint8_t *)outPlane.ptr[0], outPlane.pitch[0],
            correctionDelta, correctionDeltaPitch,
            positiveCorrectionGate, positiveCorrectionGatePitch,
            negativeCorrectionGate, negativeCorrectionGatePitch,
            (const uint8_t *)inputPlane.ptr[0], inputPlane.pitch[0],
            (const uint8_t *)refPlane.ptr[0], refPlane.pitch[0],
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_VC_POS].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_VC_NEG].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_LC_POS].ptr,
            (uint8_t *)m_stagedBuffers[RTGMC_SHIMMER_REPAIR_STAGE_LC_NEG].ptr,
            m_stagedPitch, outPlane.width, outPlane.height,
            RTGMC_SHIMMER_REPAIR_STAGE_Y_OFFSET, rangeHalf, maxVal);
    if (!launched) {
        return RGY_ERR_INVALID_PARAM;
    }
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        const auto err = err_to_rgy(cudaerr);
        AddMessage(RGY_LOG_ERROR, _T("error at rtgmc shimmer repair staged apply (plane %d): %s.\n"),
            iplane, get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    rtgmcShimmerRepairLoadProfile(prm.get());

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(m_param);
    if (m_useKernel
        && (!m_param
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->repairThin != prm->repairThin
        || prmPrev->repairPad != prm->repairPad)) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-shimmer-repair kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 8);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    m_useStagedThin4Pad0 = m_useKernel
        && prm->repairThin == 4
        && prm->repairPad == 0
        && rtgmcShimmerRepairStagedEnabled();
    if (m_useStagedThin4Pad0) {
        const int pixelBytes = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? 2 : 1;
        m_stagedPitch = ALIGN(prm->frameOut.width * pixelBytes, 128);
        m_stagedHeight = prm->frameOut.height + RTGMC_SHIMMER_REPAIR_STAGE_Y_OFFSET * 2;
        const size_t stagedBytes = (size_t)m_stagedPitch * (size_t)m_stagedHeight;
        for (int i = 0; i < RTGMC_SHIMMER_REPAIR_STAGE_BUFFER_COUNT; i++) {
            auto& buffer = m_stagedBuffers[i];
            if (buffer.ptr == nullptr || buffer.nSize != stagedBytes) {
                buffer.clear();
                buffer.nSize = stagedBytes;
                sts = buffer.alloc();
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc shimmer repair staged buffer %d.\n"), i);
                    return RGY_ERR_MEMORY_ALLOC;
                }
            }
        }
    } else {
        for (auto& buffer : m_stagedBuffers) {
            buffer.clear();
        }
        m_stagedPitch = 0;
        m_stagedHeight = 0;
    }
    sts = initLumaDump(prm->frameOut, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
    const NVEncFilterParamRtgmcShimmerRepair &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int repair = prm.repairThin;
    const char *target = (repair > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";
    auto sts = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto launchCopy = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, int iplane) -> RGY_ERR {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const dim3 blockSize(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        const bool launched = (RGY_CSP_BIT_DEPTH[dstFrame->csp] <= 8)
            ? launchRtgmcShimmerRepairCopyU8(
                gridSize, blockSize, stream,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height, 255)
            : launchRtgmcShimmerRepairCopyU16(
                gridSize, blockSize, stream,
                (uint8_t *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const uint8_t *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height, 65535);
        if (!launched) {
            return RGY_ERR_INVALID_PARAM;
        }
        const auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            auto err = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_shimmer_repair_copy"), iplane, get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    };
    auto dumpProcessedStages = [&](const char *stageTarget, RGYFrameInfo *deltaFrame, RGYFrameInfo *positiveGateFrame,
        RGYFrameInfo *negativeGateFrame, RGYFrameInfo *outputFrame, bool dumpNow) -> RGY_ERR {
        if (!dumpNow) {
            return RGY_ERR_NONE;
        }
        auto err = dumpStageFrame("correction_delta", deltaFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("positive_correction_gate", positiveGateFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("negative_correction_gate", negativeGateFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("shimmer_corrected", outputFrame, stageTarget, stream, {});
        if (err != RGY_ERR_NONE) return err;
        return RGY_ERR_NONE;
    };

    for (int iplane = 0; iplane < planes; iplane++) {
        const bool processPlane = (iplane == 0 || prm.processChroma);
        if (!processPlane || repair == 0) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (m_lumaDumpEnabled && m_lumaDumpFullYuv && repair != 0) {
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame, pInputFrame, iplane);
                if (err != RGY_ERR_NONE) return err;
            }
            continue;
        }

        RGYFrameInfo *correctionDelta = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame;
        RGYFrameInfo *positiveCorrectionGate = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame;
        RGYFrameInfo *negativeCorrectionGate = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame;

        const auto err = m_useStagedThin4Pad0 && iplane == 0
            ? launchRtgmcShimmerRepairStaged(
                pOutputFrame,
                m_lumaDumpEnabled ? correctionDelta : nullptr,
                m_lumaDumpEnabled ? positiveCorrectionGate : nullptr,
                m_lumaDumpEnabled ? negativeCorrectionGate : nullptr,
                pInputFrame, pRefFrame, iplane, stream)
            : (m_lumaDumpEnabled
                ? launchRtgmcShimmerRepairFused(
                    pOutputFrame, correctionDelta, positiveCorrectionGate, negativeCorrectionGate,
                    pInputFrame, pRefFrame, prm, iplane, stream)
                : launchRtgmcShimmerRepairApply(
                    pOutputFrame, pInputFrame, pRefFrame, prm, iplane, stream));
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (iplane == 0 && !m_lumaDumpFullYuv) {
            sts = dumpProcessedStages(target, correctionDelta, positiveCorrectionGate, negativeCorrectionGate, pOutputFrame, true);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }
    if (m_lumaDumpFullYuv) {
        RGY_ERR err = RGY_ERR_NONE;
        if (m_lumaDumpStage == "correction_delta") {
            err = dumpStageFrame("correction_delta", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_DELTA]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "positive_correction_gate") {
            err = dumpStageFrame("positive_correction_gate", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_POS_GATE]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "negative_correction_gate") {
            err = dumpStageFrame("negative_correction_gate", &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_NEG_GATE]->frame, target, stream, {});
        } else if (m_lumaDumpStage == "shimmer_corrected") {
            err = dumpStageFrame("shimmer_corrected", pOutputFrame, target, stream, {});
        }
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return rtgmcShimmerRepairRecordEvent(stream, event);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0] || !pRefFrame || !pRefFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (m_useKernel && !m_frameBuf.size()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load rtgmc-shimmer-repair kernel (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcShimmerRepair>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT]->frame.mem_type);
        const auto refMemcpyKind = getCudaMemcpyKind(pRefFrame->mem_type, m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_OUTPUT]->frame.mem_type);
        if (memcpyKind == cudaMemcpyDeviceToDevice && refMemcpyKind == cudaMemcpyDeviceToDevice) {
            auto err = processFrame(&pOutFrame->frame, pInputFrame, pRefFrame, *prm, stream, wait_events, event);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            return RGY_ERR_NONE;
        }
    }

    if (m_useKernel) {
        auto pInputTmp = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP]->frame;
        auto pRefTmp = &m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP]->frame;
        auto waitErr = rtgmcShimmerRepairWaitEvents(stream, wait_events);
        if (waitErr != RGY_ERR_NONE) {
            return waitErr;
        }
        auto copyErr = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_INPUT_TMP]->copyFrameAsync(pInputFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair input frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        copyErr = m_frameBuf[RTGMC_SHIMMER_REPAIR_FRAME_REF_TMP]->copyFrameAsync(pRefFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair reference frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        auto err = processFrame(&pOutFrame->frame, pInputTmp, pRefTmp, *prm, stream, {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        return RGY_ERR_NONE;
    }

    auto waitErr = rtgmcShimmerRepairWaitEvents(stream, wait_events);
    if (waitErr != RGY_ERR_NONE) {
        return waitErr;
    }
    auto copyErr = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    return rtgmcShimmerRepairRecordEvent(stream, event);
}

RGY_ERR NVEncFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    return run_filter(pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

void NVEncFilterRtgmcShimmerRepair::close() {
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFullYuv = false;
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
    m_useStagedThin4Pad0 = false;
    for (auto& buffer : m_stagedBuffers) {
        buffer.clear();
    }
    m_stagedPitch = 0;
    m_stagedHeight = 0;
}
