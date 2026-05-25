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

#include "NVEncFilterDegrain.h"
#include "NVEncFilterDegrainCommon.h"

#include <algorithm>
#include <limits>

#include "rgy_frame_info.h"
#include "rgy_util.h"

namespace {
bool degrainChromaBuildEnabled(const RGY_CSP csp, const VppDegrain &degrain) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return degrain.chroma;
    default:
        return false;
    }
}

int degrainChromaScaleX(const RGY_CSP csp) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
        return 2;
    case RGY_CHROMAFMT_YUV444:
        return 1;
    default:
        return 1;
    }
}

int degrainChromaScaleY(const RGY_CSP csp) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420:
        return 2;
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return 1;
    default:
        return 1;
    }
}

RGY_ERR degrainWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto &waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            auto err = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR degrainRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto err = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}

}

NVEncFilterDegrain::NVEncFilterDegrain() :
    NVEncFilter(),
    m_cacheFrames(),
    m_degrain(),
    m_degrainChroma(),
    m_degrainPel1(),
    m_degrainMotionSearchPrograms(),
    m_analysis(),
    m_directAnalyzeResultSet(),
    m_boundAnalyzeResult(),
    m_frameAnalysisData(),
    m_frameAnalysisLayout(),
    m_pendingSceneChange(),
    m_inputCount(0),
    m_drainCount(0),
    m_bInterlacedWarn(false),
    m_lastAnalysisUsedSearchLuma(false),
    m_lastAnalysisIncludedChroma(false),
    m_useDegrainChromaProgram(false),
    m_debugEnv() {
    m_name = _T("degrain");
}

NVEncFilterDegrain::~NVEncFilterDegrain() {
    close();
}

bool NVEncFilterDegrain::modeImplemented(VppDegrainMode mode) const {
    switch (mode) {
    case VppDegrainMode::Source:
    case VppDegrainMode::Analyze:
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
    case VppDegrainMode::Degrain:
        return true;
    default:
        return false;
    }
}

bool NVEncFilterDegrain::modeRequiresAnalysis(VppDegrainMode mode) const {
    switch (mode) {
    case VppDegrainMode::Analyze:
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
    case VppDegrainMode::Degrain:
        return true;
    case VppDegrainMode::Source:
    default:
        return false;
    }
}

RGY_ERR NVEncFilterDegrain::checkParam(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("degrain requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.blksize <= 0 || prm->degrain.search < 0
        || prm->degrain.thsad < 0 || prm->degrain.thsadc < 0 || prm->degrain.thscd1 < 0 || prm->degrain.thscd2 < 0) {
        AddMessage(RGY_LOG_ERROR, _T("degrain requires non-negative block/search/threshold parameters.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.thscd2 > 255) {
        AddMessage(RGY_LOG_ERROR, _T("degrain thscd2 must satisfy 0 <= thscd2 <= 255.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.pel != 1 && prm->degrain.pel != 2 && prm->degrain.pel != 4) {
        AddMessage(RGY_LOG_ERROR, _T("degrain supports only pel=1, pel=2, or pel=4.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.levels < 1 || prm->degrain.levels > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain Step2c currently supports only levels=1 or levels=2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode) && prm->degrain.levels != 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis requires levels=2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp];
    if (bitDepth == 16 && chromaFormat != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_ERROR, _T("degrain 16-bit input currently supports only YUV420.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode)
        && bitDepth != 8
        && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis supports only 8-bit or 16-bit YUV420 input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (modeRequiresAnalysis(prm->degrain.mode)
        && prm->degrain.blksize != 8 && prm->degrain.blksize != 16 && prm->degrain.blksize != 32) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis supports only blksize=8, 16, or 32.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.overlap < 0 || prm->degrain.overlap >= prm->degrain.blksize) {
        AddMessage(RGY_LOG_ERROR, _T("degrain overlap must satisfy 0 <= overlap < blksize.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.overlap != 0 && prm->degrain.overlap != prm->degrain.blksize / 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain Step2a currently supports only overlap=0 or overlap=blksize/2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.delta < 1 || prm->degrain.delta > 5) {
        AddMessage(RGY_LOG_ERROR, _T("degrain delta must satisfy 1 <= delta <= 5.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const bool allowExtendedDelta =
        prm->degrain.mode == VppDegrainMode::Analyze
        || (prm->degrain.mode == VppDegrainMode::Degrain
            && prm->degrain.stage == VppDegrainStage::TR2);
    if (prm->degrain.delta > 2 && !allowExtendedDelta) {
        AddMessage(RGY_LOG_ERROR, _T("degrain delta>2 is currently supported only for analyze or degrain stage=tr2.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->degrain.tr0 < 0 || prm->degrain.tr0 > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain tr0 must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.rep0 < 0 || prm->degrain.rep0 > 7) {
        AddMessage(RGY_LOG_ERROR, _T("degrain rep0 must be 0 - 7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.searchRefine < 0 || prm->degrain.searchRefine > 3) {
        AddMessage(RGY_LOG_ERROR, _T("degrain search_refine must be 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.mvSpatialRefine < -1) {
        AddMessage(RGY_LOG_ERROR, _T("degrain mv_spatial_refine must be -1 or greater.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.subpelInterp < 0 || prm->degrain.subpelInterp > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain subpelinterp must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.searchParam < 0 || prm->degrain.pelSearch < 0
        || prm->degrain.lambda < 0 || prm->degrain.lsad < 0
        || prm->degrain.pnew < 0 || prm->degrain.plevel < 0
        || prm->degrain.dct < 0) {
        AddMessage(RGY_LOG_ERROR, _T("degrain searchparam/pelsearch/lambda/lsad/pnew/plevel/dct must be non-negative.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.useFlag < 0 || prm->degrain.useFlag > 2) {
        AddMessage(RGY_LOG_ERROR, _T("degrain useFlag must be 0, 1, or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->degrain.binomial < -1 || prm->degrain.binomial > 1) {
        AddMessage(RGY_LOG_ERROR, _T("degrain binomial must be auto, true, or false.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::allocCacheFrames(const RGYFrameInfo &frameInfo) {
    bool reuse = true;
    for (const auto &frame : m_cacheFrames) {
        if (!frame || cmpFrameInfoCspResolution(&frame->frame, &frameInfo)) {
            reuse = false;
            break;
        }
        for (int i = 0; i < RGY_CSP_PLANES[frame->frame.csp]; i++) {
            if (frame->frame.ptr[i] == nullptr) {
                reuse = false;
                break;
            }
        }
        if (!reuse) {
            break;
        }
    }
    if (reuse) {
        return RGY_ERR_NONE;
    }

    for (auto &frame : m_cacheFrames) {
        frame.reset();
    }
    for (auto &frame : m_cacheFrames) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        if (!frame || frame->alloc() != RGY_ERR_NONE) {
            for (auto &clearFrame : m_cacheFrames) {
                clearFrame.reset();
            }
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::buildKernels(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
    const auto layout = rgy_degrain_make_block_layout(prm->frameOut, prm->degrain);
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int binomial = (prm->degrain.binomial < 0)
        ? ((prm->degrain.stage != VppDegrainStage::TR2) ? 1 : 0)
        : prm->degrain.binomial;
    const auto makeOptions = [&](const int pel, const int planeScaleX, const int planeScaleY) {
        return strsprintf(
        "-D TypePixel=%s"
        " -D DEGRAIN_BLOCK_SIZE=%d"
        " -D DEGRAIN_OVERLAP=%d"
        " -D DEGRAIN_STEP=%d"
        " -D DEGRAIN_BLOCKS_X=%d"
        " -D DEGRAIN_BLOCKS_Y=%d"
        " -D DEGRAIN_COVERED_WIDTH=%d"
        " -D DEGRAIN_COVERED_HEIGHT=%d"
        " -D DEGRAIN_SEARCH=%d"
        " -D DEGRAIN_REFS=%d"
        " -D DEGRAIN_PEL=%d"
        " -D DEGRAIN_SUBPEL_INTERP=%d"
        " -D DEGRAIN_TR0=%d"
        " -D DEGRAIN_SEARCH_REFINE=%d"
        " -D DEGRAIN_REP0=%d"
        " -D DEGRAIN_PIXEL_MAX=%d"
        " -D DEGRAIN_BINOMIAL=%d"
        " -D DEGRAIN_CHROMA=%d"
        " -D DEGRAIN_CHROMA_SCALE_X=%d"
        " -D DEGRAIN_CHROMA_SCALE_Y=%d"
        " -D DEGRAIN_PLANE_SCALE_X=%d"
        " -D DEGRAIN_PLANE_SCALE_Y=%d"
        " -D DEGRAIN_TV_RANGE=%d"
        " -D DEGRAIN_ANALYSIS_LUMA_PRECONVERTED=%d"
        " -D DEGRAIN_FAST_OVERLAP_TRIG=1",
        bitdepth > 8 ? "ushort" : "uchar",
        layout.blockSize,
        layout.overlap,
        layout.step,
        layout.blocksX,
        layout.blocksY,
        layout.coveredWidth,
        layout.coveredHeight,
        layout.search,
        layout.temporalDirections,
        pel,
        prm->degrain.subpelInterp,
        prm->degrain.tr0,
        prm->degrain.searchRefine,
        prm->degrain.rep0,
        pixelMax,
        binomial,
        degrainChromaBuildEnabled(prm->frameOut.csp, prm->degrain) ? 1 : 0,
        degrainChromaScaleX(prm->frameOut.csp),
        degrainChromaScaleY(prm->frameOut.csp),
        planeScaleX,
        planeScaleY,
        prm->degrain.tvRange ? 1 : 0,
        prm->degrain.tvRange ? 1 : 0);
    };
    const int chromaScaleX = degrainChromaScaleX(prm->frameOut.csp);
    const int chromaScaleY = degrainChromaScaleY(prm->frameOut.csp);
    const bool buildChromaProgram = chromaScaleX != 1 || chromaScaleY != 1;

    const auto options = makeOptions(prm->degrain.pel, 1, 1);
    m_degrain = options;
    if (buildChromaProgram) {
        const auto optionsChroma = makeOptions(prm->degrain.pel, chromaScaleX, chromaScaleY);
        m_degrainChroma = optionsChroma;
        m_useDegrainChromaProgram = true;
    } else {
        m_degrainChroma.clear();
        m_useDegrainChromaProgram = false;
    }
    const auto optionsPel1 = makeOptions(1, 1, 1);
    m_degrainPel1 = optionsPel1;
    return RGY_ERR_NONE;
}

NVEncDegrainKernelProgram *NVEncFilterDegrain::degrainRenderProgram(RGY_PLANE plane) {
    return (plane != RGY_PLANE_Y && m_useDegrainChromaProgram) ? &m_degrainChroma : &m_degrain;
}

NVEncDegrainKernelProgram *NVEncFilterDegrain::getDegrainMotionSearchProgram(const std::string &normalizedBuildOptions) {
    if (normalizedBuildOptions.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search build options are empty.\n"));
        return nullptr;
    }
    auto program = m_degrainMotionSearchPrograms.find(normalizedBuildOptions);
    if (program == m_degrainMotionSearchPrograms.end()) {
        auto inserted = m_degrainMotionSearchPrograms.emplace(normalizedBuildOptions, normalizedBuildOptions);
        return &inserted.first->second;
    }
    return &program->second;
}

int NVEncFilterDegrain::cacheIndex(int frame) const {
    return frame % DEGRAIN_CACHE_SIZE;
}

int NVEncFilterDegrain::analysisCacheIndex(int frame) const {
    return frame % RGY_DEGRAIN_ANALYSIS_LUMA_CACHE_SIZE;
}

bool NVEncFilterDegrain::useAnalysisLumaCache() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    return prm && modeRequiresAnalysis(prm->degrain.mode) && degrainRequiresAnalysisLumaCache(prm->degrain);
}

bool NVEncFilterDegrain::degrainDebugLogEnabled() const {
    return m_pLog != nullptr && m_pLog->getLogLevel(RGY_LOGT_VPP) <= RGY_LOG_DEBUG;
}

void NVEncFilterDegrain::logAnalyzeBinding(const TCHAR *sourceName, const RGYFrameInfo *frame, const RGYDegrainAnalyzeResult &result) {
    if (!degrainDebugLogEnabled() || !sourceName || !frame || !result.valid()) {
        return;
    }
    const int slotDelta = std::max(1, result.layout.temporalDirections / 2);
    AddMessage(RGY_LOG_DEBUG,
        _T("degrain bind %s slot=d%d cur{id=%d,pts=%lld,dur=%lld} meta{frame=%d,id=%d,pts=%lld,dur=%lld,dirs=%d,flags=0x%08x}.\n"),
        sourceName,
        slotDelta,
        frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
        result.frameIndex, result.inputFrameId, (long long)result.timestamp, (long long)result.duration,
        result.layout.temporalDirections, result.flags);
}

void NVEncFilterDegrain::logLocalAnalysis(const TCHAR *sourceName, const RGYFilterDegrainFrameSet &frames) {
    if (!degrainDebugLogEnabled() || !sourceName || !frames.cur) {
        return;
    }
    const int slotDelta = std::max(1, analysisLayout().temporalDirections / 2);
    AddMessage(RGY_LOG_DEBUG,
        _T("degrain bind %s slot=d%d cur{id=%d,pts=%lld,dur=%lld} analysis{dirs=%d,search-luma=%s}.\n"),
        sourceName,
        slotDelta,
        frames.cur->inputFrameId, (long long)frames.cur->timestamp, (long long)frames.cur->duration,
        analysisLayout().temporalDirections,
        m_lastAnalysisUsedSearchLuma ? _T("attached") : _T("source"));
}

void NVEncFilterDegrain::logAnalysisSamples(const TCHAR *sourceName, const RGYFrameInfo *frame, cudaStream_t stream) {
    if (!degrainDebugLogEnabled() || !sourceName || !frame) {
        return;
    }

    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    const auto &layout = analysisLayout();
    if (!mv || !sad || layout.blockCount() == 0 || layout.temporalDirections <= 0) {
        return;
    }

    const auto entryCount = layout.blockCount() * (size_t)layout.temporalDirections;
    if (layout.mvCount() < entryCount || layout.sadCount() < entryCount) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain sample src=%s skipped: invalid entry count (blocks=%llu, stride=%d).\n"),
            sourceName, (unsigned long long)layout.blockCount(), layout.temporalDirections);
        return;
    }

    if (analysisEvent()() != nullptr) {
        cudaStreamWaitEvent(stream, analysisEvent()(), 0);
    }
    std::vector<RGYDegrainMV> mvValues(layout.mvCount());
    std::vector<RGYDegrainSAD> sadValues(layout.sadCount());
    auto err = err_to_rgy(cudaMemcpyAsync(mvValues.data(), mv->ptr, rgy_degrain_mv_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to copy MV buffer: %s.\n"), sourceName, get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaMemcpyAsync(sadValues.data(), sad->ptr, rgy_degrain_sad_bytes(layout), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to copy SAD buffer: %s.\n"), sourceName, get_err_mes(err));
        return;
    }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain sample src=%s failed to synchronize MV/SAD copy: %s.\n"), sourceName, get_err_mes(err));
        return;
    }

    std::array<size_t, 3> sampleBlocks = {
        0,
        layout.blockCount() / 2,
        layout.blockCount() - 1
    };
    static const TCHAR *sampleNames[] = { _T("first"), _T("mid"), _T("last") };
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    const int reqDelta = requestedDelta();
    const auto stage = (prm) ? prm->degrain.stage : VppDegrainStage::Auto;
    for (int dir = 0; dir < std::min(layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); dir++) {
        const int delta = rgy_degrain_delta_from_ref_index(dir);
        const bool forward = rgy_degrain_ref_index_is_forward(dir);
        for (int sample = 0; sample < (int)sampleBlocks.size(); sample++) {
            const size_t block = sampleBlocks[sample];
            if ((sample > 0 && block == sampleBlocks[sample - 1])
                || (sample > 1 && block == sampleBlocks[sample - 2])) {
                continue;
            }
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)dir;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            AddMessage(RGY_LOG_DEBUG,
                _T("degrain sample src=%s cur{id=%d,pts=%lld,dur=%lld} request{delta=%d,stage=%s} layout{blocks=%llu,stride=%d,directions=%d} slot=d%d%s sample=%s block=%llu entry=%llu motion{dx=%d,dy=%d,sad=%u,ref_slot=%u} sad_stats{sad=%u,src=%u,ref=%u} diag{base_sad=%u,score=%u,sad_score=%u}.\n"),
                sourceName,
                frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
                reqDelta, get_cx_desc(list_vpp_degrain_stage, (int)stage),
                (unsigned long long)layout.blockCount(), layout.temporalDirections, layout.temporalDirections,
                delta, forward ? _T("f") : _T("b"),
                sampleNames[sample],
                (unsigned long long)block,
                (unsigned long long)entry,
                (int)mvValue.dx, (int)mvValue.dy, (unsigned int)mvValue.sad, (unsigned int)mvValue.refdir,
                (unsigned int)sadValue.sad, (unsigned int)sadValue.srcAvg, (unsigned int)sadValue.refAvg,
                (unsigned int)mvValue.flags, (unsigned int)mvValue.reserved, (unsigned int)sadValue.reserved);
        }
    }

}

void NVEncFilterDegrain::logReferenceGate(const std::shared_ptr<NVEncFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames,
    const RGYDegrainRefDisableArray &availabilityDisableRefs, const RGYDegrainRefDisableArray &useFlagDisableRefs,
    const RGYDegrainRefDisableArray &disableRefs,
    const std::array<size_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> *sceneChangeBlockCounts,
    const uint32_t scaledThSad, const uint32_t scaledThSCD1, const uint64_t scaledThSCD2) {
    if (!degrainDebugLogEnabled() || !prm || !frames.cur) {
        return;
    }

    auto line = strsprintf(
        _T("degrain gate cur{id=%d,pts=%lld,dur=%lld} slot=d%d mode=%s stage=%s sad_limit=%u scene_limit1=%u scene_limit2=%llu reference_policy=%d"),
        frames.cur->inputFrameId, (long long)frames.cur->timestamp, (long long)frames.cur->duration,
        std::max(1, analysisLayout().temporalDirections / 2),
        get_cx_desc(list_vpp_degrain_mode, (int)prm->degrain.mode),
        get_cx_desc(list_vpp_degrain_stage, (int)prm->degrain.stage),
        scaledThSad,
        scaledThSCD1,
        (unsigned long long)scaledThSCD2,
        prm->degrain.useFlag);
    for (int dir = 0; dir < std::min(analysisLayout().temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); dir++) {
        const int delta = rgy_degrain_delta_from_ref_index(dir);
        const bool forward = rgy_degrain_ref_index_is_forward(dir);
        const bool inRange = forward ? frames.forwardRefInRange(delta) : frames.backwardRefInRange(delta);
        const auto *ref = forward ? frames.forwardRef(delta) : frames.backwardRef(delta);
        const bool availabilityDisabled = availabilityDisableRefs[dir];
        const bool useFlagDisabled = useFlagDisableRefs[dir] && !availabilityDisabled;
        const bool sceneChangeDisabled = disableRefs[dir] && !availabilityDisabled && !useFlagDisabled;
        const auto sceneChangeText = sceneChangeBlockCounts
            ? strsprintf(_T("%llu/%llu"), (unsigned long long)(*sceneChangeBlockCounts)[dir], (unsigned long long)scaledThSCD2)
            : tstring(_T("-"));
        line += strsprintf(
            _T(" d%d%s{id=%d,in=%d,policy=%d,scene=%s,disabled=%d}"),
            delta, forward ? _T("f") : _T("b"),
            ref ? ref->inputFrameId : -1,
            inRange ? 1 : 0,
            useFlagDisabled ? 1 : 0,
            sceneChangeText.c_str(),
            (availabilityDisabled || sceneChangeDisabled || useFlagDisabled) ? 1 : 0);
    }
    AddMessage(RGY_LOG_DEBUG, _T("%s.\n"), line.c_str());
}

int NVEncFilterDegrain::requestedDelta() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    return prm ? clamp(prm->degrain.delta, 1, RGY_DEGRAIN_MAX_DELTA) : 1;
}

int NVEncFilterDegrain::outputDelay() const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        return FILTER_DEFAULT_DEGRAIN_DELTA;
    }
    return prm->degrain.delta + (modeRequiresAnalysis(prm->degrain.mode) ? prm->degrain.tr0 : 0);
}

int NVEncFilterDegrain::drainFrameCount() const {
    return std::min(outputDelay(), m_inputCount);
}

RGY_ERR NVEncFilterDegrain::pushCacheFrame(const RGYFrameInfo *pInputFrame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto pCacheFrame = &m_cacheFrames[cacheIndex(m_inputCount)]->frame;
    auto err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyFrameAsync(pCacheFrame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy input to degrain cache slot %d: %s.\n"),
            cacheIndex(m_inputCount), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain cache copy event: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFrameProp(pCacheFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGYFilterDegrainFrameSet NVEncFilterDegrain::resolveFrameSet(const int currentFrame) const {
    RGYFilterDegrainFrameSet frames;
    if (m_inputCount <= 0) {
        return frames;
    }
    if (currentFrame < 0) {
        return frames;
    }

    frames.cur = &m_cacheFrames[cacheIndex(currentFrame)]->frame;
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        frames.backwardInRange[delta - 1] = (currentFrame + delta) < m_inputCount;
        frames.forwardInRange[delta - 1] = (currentFrame - delta) >= 0;
        const int backwardFrame = frames.backwardInRange[delta - 1] ? (currentFrame + delta) : currentFrame;
        const int forwardFrame = frames.forwardInRange[delta - 1] ? (currentFrame - delta) : currentFrame;
        frames.backward[delta - 1] = &m_cacheFrames[cacheIndex(backwardFrame)]->frame;
        frames.forward[delta - 1] = &m_cacheFrames[cacheIndex(forwardFrame)]->frame;
    }
    return frames;
}

RGYFilterDegrainProcessFrameSet NVEncFilterDegrain::resolveFrames(bool hasInput) const {
    RGYFilterDegrainProcessFrameSet frames = {};
    frames.currentFrame = -1;
    if (m_inputCount <= 0) {
        return frames;
    }

    const int currentFrame = hasInput
        ? (m_inputCount - outputDelay() - 1)
        : (std::max(0, m_inputCount - drainFrameCount()) + m_drainCount);
    if (currentFrame < 0) {
        return frames;
    }

    frames.currentFrame = currentFrame;
    frames.render = resolveFrameSet(currentFrame);
    frames.analysis = (degrainGetAttachedSearchLuma(frames.render.cur) || !useAnalysisLumaCache())
        ? frames.render
        : resolveAnalysisFrameSet(currentFrame);
    return frames;
}

RGY_ERR NVEncFilterDegrain::emitSourceFrame(const RGYFrameInfo *pCurrentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!pCurrentFrame || pCurrentFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getCudaMemcpyKind(pCurrentFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto err = degrainWaitEvents(stream, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = copyFrameAsync(ppOutputFrames[0], pCurrentFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain source output: %s.\n"), get_err_mes(err));
        return err;
    }
    err = degrainRecordEvent(stream, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to record degrain source copy event: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pCurrentFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::runSourceMode(const RGYFilterDegrainFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    return emitSourceFrame(frames.cur, ppOutputFrames, pOutputFrameNum, stream, {}, event);
}

RGY_ERR NVEncFilterDegrain::unsupportedModeError(VppDegrainMode mode) {
    AddMessage(RGY_LOG_ERROR,
        _T("degrain mode %s is not implemented. Supported today: source, analyze, mv(debug scaffold), sad(debug scaffold), motionback, motionforw, motionback2, motionforw2, degrain.\n"),
        get_cx_desc(list_vpp_degrain_mode, (int)mode));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    close();
    loadDebugEnv();

    sts = buildKernels(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = allocCacheFrames(prm->frameIn);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain cache: %s.\n"), get_err_mes(sts));
        return sts;
    }

    sts = allocAnalysisBuffers(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_inputCount = 0;
    m_drainCount = 0;
    m_bInterlacedWarn = false;
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    clearDirectAnalyzeResult();
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_DATA);

    if (modeRequiresAnalysis(prm->degrain.mode)) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain %s mode uses MV analysis output.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)prm->degrain.mode));
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDegrain::runResolvedFrames(const RGYFilterDegrainProcessFrameSet &frames, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    switch (prm->degrain.mode) {
    case VppDegrainMode::Source:
        return runSourceMode(frames.render, ppOutputFrames, pOutputFrameNum, stream, event);
    case VppDegrainMode::Analyze:
        return runAnalyzeMode(frames, frames.currentFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::MV:
    case VppDegrainMode::SAD:
        return runDebugMode(frames, frames.currentFrame, prm->degrain.mode, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::MotionBack:
    case VppDegrainMode::MotionForw:
    case VppDegrainMode::MotionBack2:
    case VppDegrainMode::MotionForw2:
        return runCompensateMode(frames, frames.currentFrame, prm->degrain.mode, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    case VppDegrainMode::Degrain:
        return runDegrainMode(frames, frames.currentFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    default:
        return unsupportedModeError(prm->degrain.mode);
    }
}

RGY_ERR NVEncFilterDegrain::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    const std::vector<RGYCudaEvent> wait_events;
    RGYCudaEvent filterEvent;
    auto *event = &filterEvent;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!modeImplemented(prm->degrain.mode)) {
        return unsupportedModeError(prm->degrain.mode);
    }

    const bool hasInput = (pInputFrame != nullptr && pInputFrame->ptr[0] != nullptr);
    if (hasInput) {
        if (!m_bInterlacedWarn && (pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED)) {
            AddMessage(RGY_LOG_WARN, _T("degrain Step1 skeleton is frame-domain only; current modes ignore interlaced-specific processing.\n"));
            m_bInterlacedWarn = true;
        }

        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_cacheFrames[0]->frame.mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        m_drainCount = 0;
        RGYCudaEvent cacheCopyEvent;
        auto sts = pushCacheFrame(pInputFrame, stream, wait_events, &cacheCopyEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        m_inputCount++;
        if (useAnalysisLumaCache()) {
            sts = ensureAnalysisLumaGenerated(m_inputCount - 1 - prm->degrain.tr0, stream, { cacheCopyEvent });
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (m_inputCount < outputDelay() + 1) {
            return RGY_ERR_NONE;
        }

        return runResolvedFrames(resolveFrames(true), ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    }

    if (m_drainCount < drainFrameCount()) {
        if (useAnalysisLumaCache()) {
            const int currentFrame = std::max(0, m_inputCount - drainFrameCount()) + m_drainCount;
            const auto sts = ensureAnalysisLumaGenerated(currentFrame + prm->degrain.delta, stream, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto frames = resolveFrames(false);
        m_drainCount++;
        return runResolvedFrames(frames, ppOutputFrames, pOutputFrameNum, stream, {}, event);
    }

    if (prm->degrain.mode == VppDegrainMode::Degrain && m_pendingSceneChange) {
        return resolvePendingSceneChangeFrame(ppOutputFrames, pOutputFrameNum, stream, event);
    }

    return RGY_ERR_NONE;
}

void NVEncFilterDegrain::close() {
    m_degrain.clear();
    m_degrainChroma.clear();
    m_degrainPel1.clear();
    m_degrainMotionSearchPrograms.clear();
    m_useDegrainChromaProgram = false;
    clearPendingSceneChange();
    m_analysis.mv.reset();
    m_analysis.sad.reset();
    m_analysis.windowRampY.reset();
    m_analysis.windowRampC.reset();
    m_analysis.temporalMixPlanY.reset();
    m_analysis.temporalMixPlanC.reset();
    m_analysis.temporalMixPrior.reset();
    clearDirectAnalyzeResult();
    clearFrameAnalysisData();
    for (auto &luma : m_analysis.analysisLuma) {
        luma.reset();
    }
    for (auto &frame : m_analysis.analysisLumaFrame) {
        frame = RGYFrameInfo();
    }
    m_analysis.analysisLumaFrameNumbers.fill(-1);
    for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
        analysisLumaEvent.reset();
    }
    m_analysis.analysisLumaEvent.reset();
    for (auto &luma : m_analysis.lumaLevel1) {
        luma.reset();
    }
    m_analysis.event.reset();
    m_analysis.mvBytes = 0;
    m_analysis.sadBytes = 0;
    m_analysis.analysisLumaBytes = 0;
    m_analysis.lumaLevel1Bytes = 0;
    m_analysis.analysisLumaWidth = 0;
    m_analysis.analysisLumaHeight = 0;
    m_analysis.analysisLumaPitch = 0;
    m_analysis.analysisLumaGeneratedUntil = -1;
    m_analysis.lumaLevel1Width = 0;
    m_analysis.lumaLevel1Height = 0;
    m_analysis.lumaLevel1Pitch = 0;
    m_analysis.lastFrameIndex = -1;
    m_analysis.lastInputFrameId = -1;
    m_analysis.lastTimestamp = 0;
    m_analysis.lastDuration = 0;
    m_analysis.layout = {};
    m_analysis.layoutLevel1 = {};
    m_analysis.mode = VppDegrainMode::Source;
    for (auto &frame : m_cacheFrames) {
        frame.reset();
    }
    m_inputCount = 0;
    m_drainCount = 0;
    m_bInterlacedWarn = false;
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    m_frameBuf.clear();
}

RGY_ERR NVEncFilterDegrain::emitDebugFrame(const RGYFilterDegrainFrameSet&, VppDegrainMode,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::emitCompensateFrame(const RGYFilterDegrainFrameSet&, VppDegrainMode,
    const RGYDegrainRefDisableArray&, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::emitDegrainFrame(const RGYFilterDegrainFrameSet&,
    const RGYDegrainRefDisableArray&, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::attachAnalysisData(const RGYFrameInfo *, RGYFrameInfo *, int, cudaStream_t, const RGYCudaEvent &, RGYCudaEvent *) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::prepareAnalysisState(const RGYFilterDegrainFrameSet&, cudaStream_t, const std::vector<RGYCudaEvent>&) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::prepareAnalysisStateMotionSearch(const RGYFrameInfo&, const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS>&,
    cudaStream_t, const std::vector<RGYCudaEvent>&) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::runAnalyzeMode(const RGYFilterDegrainProcessFrameSet&, int, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t, const std::vector<RGYCudaEvent>&, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::runDebugMode(const RGYFilterDegrainProcessFrameSet&, int, VppDegrainMode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t, const std::vector<RGYCudaEvent>&, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::runCompensateMode(const RGYFilterDegrainProcessFrameSet&, int, VppDegrainMode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t, const std::vector<RGYCudaEvent>&, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::runDegrainMode(const RGYFilterDegrainProcessFrameSet&, int, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t, const std::vector<RGYCudaEvent>&, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::submitSceneChangeReadback(const std::shared_ptr<NVEncFilterParamDegrain>&, const RGYFilterDegrainProcessFrameSet&,
    cudaStream_t, PendingSceneChange *) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::resolveSceneChangeReadback(PendingSceneChange&, cudaStream_t, RGYDegrainRefDisableArray *) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::resolvePendingSceneChangeFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t, RGYCudaEvent *) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_UNSUPPORTED;
}

void NVEncFilterDegrain::applyPendingSceneChangeAnalysisContext(const PendingSceneChange&) {
}

void NVEncFilterDegrain::clearPendingSceneChange() {
    m_pendingSceneChange.reset();
}

RGY_ERR NVEncFilterDegrain::resolveSceneChangeRefs(const std::shared_ptr<NVEncFilterParamDegrain>&, const RGYFilterDegrainFrameSet&, cudaStream_t,
    RGYDegrainRefDisableArray *) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::resolveSceneChange(const std::shared_ptr<NVEncFilterParamDegrain>&, const RGYFilterDegrainFrameSet&, cudaStream_t,
    bool *, bool *) {
    return RGY_ERR_UNSUPPORTED;
}

void NVEncFilterDegrain::loadDebugEnv() {
    m_debugEnv = DebugEnv();
}

RGY_ERR NVEncFilterDegrain::allocAnalysisBuffers(const std::shared_ptr<NVEncFilterParamDegrain> &prm) {
    if (modeRequiresAnalysis(prm->degrain.mode)) {
        return RGY_ERR_UNSUPPORTED;
    }
    m_analysis = RGYDegrainAnalysisState();
    return RGY_ERR_NONE;
}

const RGYFrameInfo *NVEncFilterDegrain::resolveAnalysisLumaSourceFrame(int) const {
    return nullptr;
}

RGYFilterDegrainFrameSet NVEncFilterDegrain::resolveAnalysisFrameSet(int currentFrame) const {
    return resolveFrameSet(currentFrame);
}

RGY_ERR NVEncFilterDegrain::generateAnalysisLumaFrame(int, cudaStream_t, const std::vector<RGYCudaEvent>&) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterDegrain::ensureAnalysisLumaGenerated(int, cudaStream_t, const std::vector<RGYCudaEvent>&) {
    return RGY_ERR_UNSUPPORTED;
}

void NVEncFilterDegrain::clearFrameAnalysisData() {
    m_boundAnalyzeResult = RGYDegrainAnalyzeResult();
    m_frameAnalysisData.reset();
    m_frameAnalysisLayout = {};
}

bool NVEncFilterDegrain::degrainApplyTraceEnabled() const {
    return false;
}

void NVEncFilterDegrain::logApplyTrace(const std::shared_ptr<NVEncFilterParamDegrain>&, const RGYFilterDegrainProcessFrameSet&,
    const RGYDegrainRefDisableArray&, cudaStream_t) {
}

bool NVEncFilterDegrain::validateAnalyzeResultFrame(const RGYDegrainAnalyzeResult&, const RGYFrameInfo *, int, const TCHAR *, bool) {
    return false;
}

bool NVEncFilterDegrain::bindDirectAnalyzeResult(const RGYFrameInfo *, int, cudaStream_t) {
    return false;
}

bool NVEncFilterDegrain::bindFrameAnalysisData(const RGYFrameInfo *, int, cudaStream_t) {
    return false;
}

CUMemBuf *NVEncFilterDegrain::analysisMV() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.mv : m_analysis.mv.get();
}

CUMemBuf *NVEncFilterDegrain::analysisSAD() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.sad : m_analysis.sad.get();
}

const RGYDegrainBlockLayout &NVEncFilterDegrain::analysisLayout() const {
    return m_boundAnalyzeResult.valid() ? m_frameAnalysisLayout : m_analysis.layout;
}

const RGYCudaEvent &NVEncFilterDegrain::analysisEvent() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.event : m_analysis.event;
}

bool NVEncFilterDegrain::analysisSADIncludesChroma(const std::shared_ptr<NVEncFilterParamDegrain>&) const {
    return false;
}

RGYDegrainRefDisableArray NVEncFilterDegrain::analysisAvailabilityDisableRefs(const RGYFilterDegrainFrameSet &frames) const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.availabilityDisableRefs : degrainReferenceAvailability(frames);
}

RGYDegrainAnalyzeResult NVEncFilterDegrain::analyzeResult() const {
    return RGYDegrainAnalyzeResult();
}

RGYDegrainAnalyzeResultSet NVEncFilterDegrain::analyzeResultSet() const {
    return RGYDegrainAnalyzeResultSet();
}

bool NVEncFilterDegrain::setDirectAnalyzeResult(const RGYDegrainAnalyzeResult &result) {
    RGYDegrainAnalyzeResultSet resultSet;
    if (result.valid() && result.hasFrameIdentity()) {
        const int maxDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, result.layout.temporalDirections / 2));
        for (int delta = 1; delta <= maxDelta; delta++) {
            auto slot = result;
            slot.layout.temporalDirections = rgy_degrain_temporal_direction_count(delta);
            resultSet.slots[delta] = slot;
        }
    }
    return setDirectAnalyzeResultSet(resultSet);
}

bool NVEncFilterDegrain::setDirectAnalyzeResultSet(const RGYDegrainAnalyzeResultSet &resultSet) {
    m_directAnalyzeResultSet = resultSet;
    return m_directAnalyzeResultSet.valid(requestedDelta());
}

void NVEncFilterDegrain::clearDirectAnalyzeResult() {
    m_directAnalyzeResultSet = RGYDegrainAnalyzeResultSet();
}
