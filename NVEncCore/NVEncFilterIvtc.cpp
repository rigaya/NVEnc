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

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include "convert_csp.h"
#include "rgy_avutil.h"
#include "rgy_filter_input_probe.h"
#include "NVEncFilterIvtc.h"

// Decode the cadence-tracker tag int (set per-frame by updateCadence +
// the override gate) into a short log string. Matches the 0..11 enum
// documented in processInputToCycle's cadence block.
static const char *ivtc_cadence_tag_str(int t) {
    switch (t) {
        case 0:  return "none";
        case 1:  return "warmup";
        case 2:  return "disabled";
        case 3:  return "locked/C/nochange";
        case 4:  return "locked/P/nochange";
        case 5:  return "locked/N/nochange";
        case 6:  return "locked/C/override";
        case 7:  return "locked/P/override";
        case 8:  return "locked/N/override";
        case 9:  return "locked/C/reject";
        case 10: return "locked/P/reject";
        case 11: return "locked/N/reject";
        default: return "?";
    }
}

static const int IVTC_BLOCK_X = 32;
static const int IVTC_BLOCK_Y = 8;
// リングバッファ: prev2/prev/cur/next/next2 の5枚.
// Full BWDIF temporal window for the post-processing path; the field-match
// step still only consults prev/cur/next but the extra
// prev2/next2 slots are free because flushCycle already owns the larger frame
// buffer for staging. Latency is 3 frames (process cur once next2 has arrived).
static const int IVTC_CACHE_SIZE = 5;

static std::unique_ptr<CUFrameBuf> createCUFrameBuffer(const RGYFrameInfo& frame) {
    auto buf = std::make_unique<CUFrameBuf>();
    buf->frame = frame;
    if (buf->alloc() != RGY_ERR_NONE) {
        return nullptr;
    }
    return buf;
}

// ============================================================================
//  Selection / decision-layer constants (Phase 2 / Phase 3, 2026-04-25).
// ============================================================================
// Pulled out as named constants during the Sub-Phase 7 cleanup pass so the
// matcher / rescue / confidence / plateau / temporal blocks share a single
// source of truth for their thresholds. Values are NOT changed — every
// constant equals the literal it replaced.
//
// All combScore values are 8-bit-domain trimmed-average per-WG pixel counts
// (the units produced by scoreCandidates). They are NOT bit-depth-scaled
// because the scoring kernel produces fixed-bound counts (max 256 per WG).
namespace {
constexpr uint64_t COMB_BLEND_THRESHOLD     = 65;  // primary blend-gate floor (== combThreshProg defined in processInputToCycle)
constexpr uint64_t COMB_CLEAN_BOUND         = 30;  // upper bound of CLEAN frame class (confidence layer)
constexpr uint64_t PLATEAU_DELTA            = 10;  // max combScore spread for plateau detection (Sub-Phase 6)
constexpr uint64_t PLATEAU_MAX_DELTA        = 10;  // max combMax spread for confidence-plateau detection
constexpr uint64_t TEMPORAL_TOLERANCE       = 6;   // Sub-Phase 7: how close prev match must be to plateau winner
constexpr uint64_t CONFIDENCE_PLATEAU_SCORE = 6;   // confidence-layer plateau scoreGap bound
constexpr uint64_t CONFIDENCE_GAP_THRESHOLD = 15;  // confidence: scoreGap below this -> LOW classification
constexpr uint64_t CONFIDENCE_SEPARATION    = 8;   // confidence: clear-winner gap that overrides to HIGH
constexpr uint64_t RESCUE_TRIGGER           = 40;  // rescue activates only when primaryBest.combScore >= this
constexpr uint64_t RESCUE_IMPROVEMENT       = 8;   // rescue accepts swap when resc.combScore + this < orig.combScore
constexpr int      ALT_DECISIVE_RATIO       = 3;   // v5 alt-guard: alt is decisive when 3x cleaner than primary
}  // anonymous namespace

// Forward declaration for the init()-time RFF pre-scan. Definition is
// further down in this file, after the ring-buffer helpers.
static RGY_ERR ivtcPreScanInput(const tstring &inputPath,
                                 std::vector<IvtcPreScanFrame> &frames,
                                 std::shared_ptr<RGYLog> log);

static bool ivtcMixedIsRffSection(const RGYFrameInfo *frame) {
    if (!frame) return false;
    if (frame->flags & RGY_FRAME_FLAG_RFF) return true;
    if ((frame->flags & (RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF)) == 0) return false;
    const auto ps = frame->picstruct;
    return ps == RGY_PICSTRUCT_FRAME;
}

static bool ivtcMixedIsInterlacedSection(const RGYFrameInfo *frame) {
    if (!frame || ivtcMixedIsRffSection(frame)) return false;
    const auto ps = frame->picstruct;
    return ps == RGY_PICSTRUCT_FRAME_TFF
        || ps == RGY_PICSTRUCT_FRAME_BFF
        || ps == RGY_PICSTRUCT_TFF
        || ps == RGY_PICSTRUCT_BFF
        || ps == RGY_PICSTRUCT_FIELD_TOP
        || ps == RGY_PICSTRUCT_FIELD_BOTTOM
        || ps == RGY_PICSTRUCT_INTERLACED;
}

static IvtcMixedSection ivtcMixedClassify(const RGYFrameInfo *frame) {
    if (ivtcMixedIsRffSection(frame)) return IvtcMixedSection::Rff;
    if (ivtcMixedIsInterlacedSection(frame)) return IvtcMixedSection::Interlaced;
    return IvtcMixedSection::Passthrough;
}

NVEncFilterIvtc::NVEncFilterIvtc() :
    NVEncFilter(),
    m_cacheFrames(),
    m_scoreBuf(),
    m_scoreHost(),
    m_diffBuf(),
    m_diffHost(),
    m_cycleInPts(),
    m_cycleInDur(),
    m_cycleInputIds(),
    m_cycleMatchScore(),
    m_cycleCombScore(),
    m_cycleCombMax(),
    m_cycleCombBlocks(),
    m_cycleCombScorePrim(),
    m_cycleCombMaxPrim(),
    m_cycleCombBlocksPrim(),
    m_cycleCombScoreAlt(),
    m_cycleCombMaxAlt(),
    m_cycleCombBlocksAlt(),
    m_cycleMatchAltParity(),
    m_cycleConfidence(),
    m_cycleBlendTrigger(),
    m_blendTriggerCounts{},
    m_cycleMatchType(),
    m_cycleApplyBlend(),
    m_cycleDecTag(),
    m_cycleCadenceTag(),
    m_cycleDiffPrev(),
    m_cycleSceneSAD(),
    m_cycleIsSynth(),
    m_emitQueue(),
    m_stagingBase(0),
    m_mixedDirectStagingBase(0),
    m_mixedDirectStagingCount(0),
    m_mixedDirectStagingNext(0),
    m_mixedActive(false),
    m_mixedLastInputPts(AV_NOPTS_VALUE),
    m_mixedLastInputDur(0),
    m_mixedLastEmitEndPts(AV_NOPTS_VALUE),
    m_mixedLastInputValid(false),
    m_mixedRffPendingTopFrame(),
    m_mixedRffPendingBottomFrame(),
    m_mixedRffPendingTopInfo(),
    m_mixedRffPendingBottomInfo(),
    m_mixedRffPendingTopValid(false),
    m_mixedRffPendingBottomValid(false),
    m_nPts(0),
    m_nPtsInit(false),
    m_cfrBaseDur(0),
    m_cfrEmitIdx(0),
    m_hasSaveSlot(false),
    m_cycleFilled(0),
    m_outputFrameCount(0),
    m_inputCount(0),
    m_processedCount(0),
    m_lastSceneChange(false),
    m_lastSceneSAD(0),
    m_tffFixed(-1),
    m_lastMatch(-1),
    m_lastMatchScore(0),
    m_cadenceHistory(),
    m_cadenceFill(0),
    m_cadenceIndex(0),
    m_cadenceLockedPhase(-1),
    m_cadenceConfidence(0),
    m_cadenceLastPrediction(-1),
    m_expandActive(false),
    m_skipBaseFpsMultiplier(false),
    m_displayFrameList(),
    m_displayFrameCount(0),
    m_nextDisplayIdx(0),
    m_expandBuf(),
    m_expandBufMeta(),
    m_expandBufCount(0),
    m_expandBufBase(0),
    m_expandCarryFrame(),
    m_expandCarryMeta(),
    m_expandCarryValid(false),
    m_expandCarryCodedIdx(-1),
    m_expandSynth(),
    m_expandDecLogIdx(0),
    m_expandSynthCount(0),
    m_expandCodedEmitCount(0),
    m_expandRffSeen(0),
    m_prescanRffRatio(0.0),
    m_prescanInterlacedCount(0),
    m_prescanRffTffCount(0),
    m_prescanRffBffCount(0),
    m_prescanTotalFrames(0),
    m_fpLog() {
    m_name = _T("ivtc");
    m_cadenceHistory.fill(0);
    m_cacheIsSynth.fill(false);
    m_expandCarryMeta = ExpandBufMeta{0, RGY_PICSTRUCT_UNKNOWN, AV_NOPTS_VALUE, 0, -1};
    for (auto &meta : m_expandBufMeta) {
        meta = ExpandBufMeta{0, RGY_PICSTRUCT_UNKNOWN, AV_NOPTS_VALUE, 0, -1};
    }
}

NVEncFilterIvtc::~NVEncFilterIvtc() {
    close();
}

RGY_ERR NVEncFilterIvtc::checkParam(const std::shared_ptr<NVEncFilterParamIvtc> pParam) {
    if (pParam->frameOut.height <= 0 || pParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // Field-parity processing (iy & 1 mask in score / synthesize / BWDIF
    // kernels) assumes an even-height plane. An odd height would leave a
    // stray parity-mixed row at the bottom. MPEG-2 / H.264 / HEVC always
    // produce even-height interlaced streams; only raw-input users could
    // hit this. Reject at config time rather than silently mis-process.
    if (pParam->frameIn.height % 2 != 0) {
        AddMessage(RGY_LOG_ERROR,
            _T("ivtc: input height (%d) must be even for field-parity processing.\n"),
            pParam->frameIn.height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (pParam->ivtc.combThresh < 0.0f || pParam->ivtc.combThresh > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid combthresh %.3f: must be in [0.0, 1.0].\n"), pParam->ivtc.combThresh);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cleanFrac < 0.0f || pParam->ivtc.cleanFrac > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid cleanfrac %.3f: must be in [0.0, 1.0].\n"), pParam->ivtc.cleanFrac);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.guide < 0 || pParam->ivtc.guide > 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid guide=%d: 0=min-combing, 1=2-way, 2=PAL 2:2.\n"), pParam->ivtc.guide);
        return RGY_ERR_INVALID_PARAM;
    }
    // post=1 (metrics-only) not implemented; use post=0 or post=2
    if (pParam->ivtc.post != 0 && pParam->ivtc.post != 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid post=%d: post=1 (metrics-only) not implemented; use post=0 or post=2.\n"), pParam->ivtc.post);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cycle > 16 || (pParam->ivtc.cycle > 0 && pParam->ivtc.cycle < 2)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid cycle=%d: must be -1 (auto), 0 (disabled) or in [2, 16].\n"), pParam->ivtc.cycle);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cycle > 0 && pParam->ivtc.drop != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid drop=%d: only drop=1 is supported in this build.\n"), pParam->ivtc.drop);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.mixed != 0 && pParam->ivtc.mixed != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mixed=%d: must be 0(off) or 1(on).\n"), pParam->ivtc.mixed);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.mixed) {
        if (!pParam->inputIsAvcodecReader) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed=on requires avcodec input reader (--avsw/--avhw) to preserve RFF/picstruct metadata.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (pParam->ivtc.expand > 0) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed=on cannot be used with expand=on.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (pParam->ivtc.cycle >= 0) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed=on does not allow user cycle; use cycle=auto or omit cycle.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (pParam->ivtc.drop != 1) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed=on requires drop=1.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (pParam->ivtc.back != 0 && pParam->ivtc.back != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid back=%d: 0=always test P, 1=only when C looks combed.\n"), pParam->ivtc.back);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.y0 < 0 || pParam->ivtc.y1 < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid y0=%d y1=%d: must be >= 0.\n"), pParam->ivtc.y0, pParam->ivtc.y1);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.y1 != 0 && pParam->ivtc.y1 <= pParam->ivtc.y0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid y0=%d y1=%d: y1 must be greater than y0 (or both 0 to disable).\n"), pParam->ivtc.y0, pParam->ivtc.y1);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.hysteresis < 0.0f || pParam->ivtc.hysteresis > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid hysteresis=%.3f: must be in [0.0, 1.0].\n"), pParam->ivtc.hysteresis);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    // 出力は progressive 扱い (field-matching の結果として)
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);

    // -- RFF expansion resolution (libavcodec pre-scan driven).
    //
    // At init time (right here), we run a lightweight pass over the
    // ENTIRE input file using av_parser_parse2 to extract per-frame
    // rff/tff/picture-structure flags. The DGDecode vfapidec.cpp:461-499
    // FrameList state machine then runs over those flags to produce a
    // complete display-order schedule (m_displayFrameList). At runtime
    // flushExpandBuffer just walks the schedule -- no state machine
    // decisions in the hot path.
    //
    // Because the pre-scan decodes the same input file that the demux
    // reader uses during encoding, per-frame flag alignment follows the
    // runtime decoded stream.
    //
    // Authority chain:
    //   user explicit expand=on                    -> active (pre-scan mandatory)
    //   user explicit expand=off                   -> off
    //   expand=auto (default) + guide>=1 +
    //     demuxer pulldown hint                    -> active (pre-scan then confirms)
    //   otherwise                                  -> off
    //
    // Once the pre-scan completes and the schedule is built we compute
    // the actual expansion ratio and pick cycle/drop to cancel it at
    // the external output rate (typical clean 3:2 pulldown: 1.25x ->
    // cycle=5 drop=1, net 1.0x).
    m_mixedActive = prm->ivtc.mixed != 0;
    if (m_mixedActive) {
        prm->ivtc.expand = 0;
        prm->ivtc.cycle = 5;
        prm->ivtc.drop = 1;
        prm->baseFps = rgy_rational<int>(24000, 1001);
        pParam->baseFps = prm->baseFps;
        m_pathThrough &= ~FILTER_PATHTHROUGH_TIMESTAMP;
    }

    bool expandRequested;
    if (m_mixedActive) {
        expandRequested = false;
    } else if (prm->ivtc.expand > 0) {
        expandRequested = true;
    } else if (prm->ivtc.expand < 0) {
        expandRequested = (prm->ivtc.guide >= 1)
                        && prm->inputBPulldownDetected;
    } else {
        expandRequested = false;
    }

    m_expandActive = false;
    if (expandRequested) {
        // Run the pre-scan unconditionally when requested. If it fails
        // and the user explicitly said expand=on, that's an error; if
        // auto, we silently disable expansion.
        std::vector<IvtcPreScanFrame> scanFrames;
        const auto scanErr = ivtcPreScanInput(prm->inputFilePath, scanFrames, m_pLog);
        // DIAG #5: mirror the pre-scan summary through the direct
        // m_pLog->write path (bypasses AddMessage's _vsctprintf +
        // _vstprintf_s variadic layer, which invoked the CRT invalid-
        // parameter handler -> STATUS_STACK_BUFFER_OVERRUN in the
        // prescan4 crash). All numeric args pre-extracted to strict
        // int/double locals so there's no ambiguity at the va_list boundary.
        // CHANGE 2 + CHANGE 3 (2026-04-24): compute the pre-scan summary
        // ONCE over scanFrames. Used later by the RFF-driven cycle
        // auto-resolve (CHANGE 2) and the pure-soft-telecine auto-disable
        // heuristic (CHANGE 3). These stats are only available in this
        // scope (scanFrames goes out of scope at end of this branch), so
        // they must be latched into class members now.
        {
            int rffC = 0, tffC = 0, bffC = 0, interC = 0;
            for (const auto &f : scanFrames) {
                if (f.rff) { rffC++; if (f.tff) tffC++; else bffC++; }
                if (!f.progressive) interC++;
            }
            m_prescanTotalFrames     = (int)scanFrames.size();
            m_prescanRffTffCount     = tffC;
            m_prescanRffBffCount     = bffC;
            m_prescanInterlacedCount = interC;
            m_prescanRffRatio        = (m_prescanTotalFrames > 0)
                                       ? (double)rffC / (double)m_prescanTotalFrames
                                       : 0.0;
        }
        if (m_pLog) {
            const int    frameCount = m_prescanTotalFrames;
            const int    rff        = m_prescanRffTffCount + m_prescanRffBffCount;
            const double pct        = m_prescanRffRatio * 100.0;
            const TCHAR *stsStr     = get_err_mes(scanErr);
            m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
                _T("ivtc DIAG: pre-scan returned sts=%s frames=%d RFF=%d (%.3f%%) RFF-TFF=%d RFF-BFF=%d\n"),
                stsStr, frameCount, rff, pct, m_prescanRffTffCount, m_prescanRffBffCount);
        }
        if (scanErr != RGY_ERR_NONE) {
            if (prm->ivtc.expand > 0) {
                AddMessage(RGY_LOG_ERROR,
                    _T("ivtc: expand=on requested but pre-scan failed (%s).\n"),
                    get_err_mes(scanErr));
                return scanErr;
            }
            AddMessage(RGY_LOG_WARN,
                _T("ivtc: expand=auto pre-scan failed; expansion disabled.\n"));
        } else if (scanFrames.empty()) {
            if (prm->ivtc.expand > 0) {
                AddMessage(RGY_LOG_ERROR,
                    _T("ivtc: expand=on requested but pre-scan returned 0 frames.\n"));
                return RGY_ERR_INVALID_DATA_TYPE;
            }
        } else {
            const int trimOff = std::max(0, prm->trimOffset);
            const int trimLen = (prm->trimFrameCount > 0)
                                ? prm->trimFrameCount
                                : std::max(0, (int)scanFrames.size() - trimOff);
            buildScheduleFromScan(scanFrames, trimOff, trimLen);

            if (m_displayFrameCount == 0) {
                AddMessage(RGY_LOG_WARN,
                    _T("ivtc: pre-scan produced empty schedule; expansion disabled.\n"));
            } else if (m_expandRffSeen == 0 && prm->ivtc.expand <= 0) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("ivtc: pre-scan found 0 RFF frames; expand=auto stays off.\n"));
                m_displayFrameList.clear();
                m_displayFrameCount = 0;
            } else {
                const int    codedInRange = trimLen;
                const int    synthInSched = m_displayFrameCount - codedInRange;
                const double ratio        = (codedInRange > 0)
                                            ? (double)m_displayFrameCount / (double)codedInRange
                                            : 1.0;

                // If the pre-scan produced zero synth entries (schedule
                // == coded 1:1), expansion is a no-op: there's no
                // pulldown to undo. Running the buffered flush path
                // would add 10-frame latency and burn memory for no
                // output benefit, so we disable expansion entirely --
                // even when the user explicitly asked for expand=on.
                // The upstream cycle-auto logic will then pick cycle=0
                // or cycle=5 the normal way based on baseFps.
                if (synthInSched <= 0) {
                    AddMessage(RGY_LOG_INFO,
                        _T("ivtc: pre-scan found no expansion needed ")
                        _T("(%d coded -> %d display, 1.000x). Disabling expansion.\n"),
                        codedInRange, m_displayFrameCount);
                    m_displayFrameList.clear();
                    m_displayFrameCount = 0;
                } else {
                    // Dynamic cycle/drop from measured ratio. We want
                    // the internal decimation ((cycle-drop)/cycle) to
                    // be 1/ratio so that net output rate == coded rate.
                    //
                    //   ratio=1.25  -> cycle=5  drop=1
                    //   ratio=1.20  -> cycle=6  drop=1
                    //   ratio=1.50  -> cycle=3  drop=1
                    //   ratio<=1.05 -> cycle=0 (no decimation; hits the
                    //                  disable branch above via
                    //                  synthInSched<=0)
                    int newCycle, newDrop;
                    if (ratio > 1.40) {
                        newCycle = 3; newDrop = 1;   // ~1.50x
                    } else if (ratio > 1.22) {
                        newCycle = 5; newDrop = 1;   // ~1.25x
                    } else {
                        newCycle = 6; newDrop = 1;   // ~1.20x
                    }

                    m_expandActive          = true;
                    m_skipBaseFpsMultiplier = true;
                    m_pathThrough          &= ~FILTER_PATHTHROUGH_TIMESTAMP;
                    m_expandBufCount        = 0;
                    m_expandBufBase         = 0;
                    m_expandCarryValid      = false;
                    m_expandCarryCodedIdx   = -1;
                    m_expandDecLogIdx       = 0;
                    m_nextDisplayIdx        = 0;

                    prm->ivtc.cycle = newCycle;
                    prm->ivtc.drop  = newDrop;

                    // DIAG #2: confirm the dispatch actually assigned. Uses
                    // m_pLog->write directly to bypass AddMessage's variadic
                    // layer (see DIAG #5 comment). All numeric args pre-
                    // extracted to local int/double.
                    if (m_pLog) {
                        const double d_ratio = ratio;
                        const int    i_cycle = newCycle;
                        const int    i_drop  = newDrop;
                        const int    i_act   = (int)m_expandActive;
                        const int    i_skip  = (int)m_skipBaseFpsMultiplier;
                        m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
                            _T("ivtc DIAG: cycle/drop dispatch: ratio=%.4fx -> cycle=%d drop=%d (m_expandActive=%d m_skipBaseFpsMultiplier=%d)\n"),
                            d_ratio, i_cycle, i_drop, i_act, i_skip);
                    }

                    // Diagnostic dump of the first 30 schedule entries
                    // so the user can cross-check against dgdecode_trace.txt.
                    const int dumpN = std::min(30, m_displayFrameCount);
                    AddMessage(RGY_LOG_INFO,
                        _T("ivtc: RFF expansion ENABLED (pre-scan). ")
                        _T("%d coded -> %d display (%.3fx, %d synth). cycle=%d drop=%d.\n"),
                        codedInRange, m_displayFrameCount, ratio, synthInSched,
                        newCycle, newDrop);
                    for (int i = 0; i < dumpN; i++) {
                        const auto &df = m_displayFrameList[i];
                        AddMessage(RGY_LOG_DEBUG,
                            _T("ivtc: schedule[%3d]: top=%d bot=%d %s\n"),
                            i, df.topSource, df.bottomSource,
                            df.isSynth ? _T("SYNTH") : _T("normal"));
                    }
                }
            }
        }
    }

    // CHANGE 3 (2026-04-24): pure-soft-telecine auto-disable.
    //   When expand was auto-enabled (prm->ivtc.expand < 0) and the
    //   pre-scan reveals a clean 3:2 pulldown (high RFF, balanced
    //   TFF/BFF, negligible interlaced content), the decoded stream
    //   already delivers one progressive film frame per coded
    //   picture. Expansion would add cross-time synth composites for
    //   no benefit -- structural judder and wasted cycles. Disable
    //   expansion here and fall through to the RFF-driven cycle
    //   auto-resolve (CHANGE 2), which will pick cycle=0 for this
    //   material via the rff_ratio>0.35 branch (already at coded
    //   rate, no decimation needed).
    //
    //   Gate on prm->ivtc.expand < 0 so explicit expand=on is NEVER
    //   auto-disabled -- the user is telling us to run the expansion
    //   path regardless of heuristics.
    if (m_expandActive && prm->ivtc.expand < 0 && m_prescanRffRatio > 0.45) {
        const double tffBffBalance =
            (m_prescanRffTffCount > 0 && m_prescanRffBffCount > 0)
            ? (double)std::min(m_prescanRffTffCount, m_prescanRffBffCount)
              / (double)std::max(m_prescanRffTffCount, m_prescanRffBffCount)
            : 0.0;
        const double interlacedFraction =
            (m_prescanTotalFrames > 0)
            ? (double)m_prescanInterlacedCount / (double)m_prescanTotalFrames
            : 0.0;
        if (tffBffBalance > 0.80 && interlacedFraction < 0.10) {
            m_expandActive = false;
            // Force cycle=0 (no decimation) directly. The decoded stream is
            // already at film rate (one clean film frame per coded picture),
            // so a cycle=5 drop=1 decimator would drop 20% of real film
            // frames. Do NOT leave cycle=-1 to fall through CHANGE 2's
            // auto-resolve -- that branch reads m_prescanRffRatio which is
            // still > 0.45 here and would re-select cycle=5 drop=1.
            // CHANGE 2's `if (prm->ivtc.cycle >= 0)` guard honours this
            // explicit choice.
            prm->ivtc.cycle = 0;
            prm->ivtc.drop  = 0;
            if (m_pLog) {
                const double pctRff    = m_prescanRffRatio * 100.0;
                const double pctTffBff = tffBffBalance * 100.0;
                const double pctInt    = interlacedFraction * 100.0;
                m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
                    _T("ivtc: pure soft-telecine detected (RFF=%.1f%%, TFF/BFF=%.1f%%, interlaced=%.1f%%). Expansion auto-disabled.\n"),
                    pctRff, pctTffBff, pctInt);
            }
        }
    }

    if (!m_expandActive) {
        m_skipBaseFpsMultiplier = false;
        m_expandBufCount        = 0;
        m_expandBufBase         = 0;
        m_expandCarryValid      = false;
        m_expandCarryCodedIdx   = -1;
        m_displayFrameList.clear();
        m_displayFrameCount     = 0;
        m_nextDisplayIdx        = 0;
    }

    // Cycle auto-resolution.
    //
    // CHANGE 2 (2026-04-24): prefer the pre-scan's RFF statistic over the
    // reader-reported baseFps. The reader's bPulldown heuristic can rewrite
    // avgDuration *= 1.25 (see rgy_input_avcodec.cpp:922-1003), producing a
    // 23.976 label on a stream whose decoder still emits 29.97 frames. Reading
    // baseFps in that state led to cycle=0 when expand=on actually needed
    // cycle=5. The pre-scan's rff_frames/total_frames ratio is a direct
    // structural measurement and is unaffected by any upstream relabelling.
    //
    // Routing priority:
    //   1. user explicit cycle >= 0                -> honour it
    //   2. m_expandActive (pre-scan dispatched)    -> pre-scan picked
    //                                                 cycle/drop at :422-442
    //   3. guide=2 (PAL 2:2)                       -> cycle=2
    //   4. prescan ratio > 0.35                    -> cycle=5 drop=1 (strong
    //                                                 telecine signal)
    //   5. prescan ratio < 0.10                    -> cycle=0 (clean
    //                                                 progressive)
    //   6. prescan ratio in (0.0, 0.10..0.35]      -> cycle=5 drop=1
    //                                                 (ambiguous / mixed;
    //                                                 attempt IVTC)
    //   7. no prescan data                         -> fall back to baseFps
    //                                                 (legacy branch)
    if (prm->ivtc.cycle >= 0) {
        // user explicitly set cycle -- nothing to do
    } else if (m_expandActive) {
        // pre-scan set cycle/drop already at :422-442
    } else if (prm->ivtc.guide == 2) {
        AddMessage(RGY_LOG_DEBUG, _T("ivtc: guide=2 (PAL 2:2), forcing cycle=2 (no decimation).\n"));
        prm->ivtc.cycle = 2;
    } else if (m_prescanTotalFrames > 0 && m_prescanRffRatio > 0.35) {
        AddMessage(RGY_LOG_INFO,
            _T("ivtc: pre-scan RFF ratio %.1f%% > 35%% -- strong telecine signal. cycle=5 drop=1.\n"),
            m_prescanRffRatio * 100.0);
        prm->ivtc.cycle = 5;
        prm->ivtc.drop  = 1;
    } else if (m_prescanTotalFrames > 0 && m_prescanRffRatio < 0.10) {
        AddMessage(RGY_LOG_INFO,
            _T("ivtc: pre-scan RFF ratio %.1f%% < 10%% -- clean progressive. cycle=0 (field-match only).\n"),
            m_prescanRffRatio * 100.0);
        prm->ivtc.cycle = 0;
    } else if (m_prescanTotalFrames > 0 && m_prescanRffRatio > 0.0) {
        AddMessage(RGY_LOG_INFO,
            _T("ivtc: pre-scan RFF ratio %.1f%% -- ambiguous zone, attempting IVTC. cycle=5 drop=1.\n"),
            m_prescanRffRatio * 100.0);
        prm->ivtc.cycle = 5;
        prm->ivtc.drop  = 1;
    } else {
        // No pre-scan was performed (expand=off and no RFF hint path).
        // Fall back to the legacy baseFps-threshold heuristic so behaviour
        // on inputs without a pre-scan is unchanged. NOTE: this branch is
        // still sensitive to upstream bPulldown rewriting baseFps; CHANGE 1
        // addresses that only for expand=auto/on.
        const double inputFps = (prm->baseFps.d() > 0) ? (double)prm->baseFps.n() / (double)prm->baseFps.d() : 0.0;
        if (inputFps > 0.0 && inputFps < 26.0) {
            AddMessage(RGY_LOG_INFO,
                _T("ivtc: no pre-scan data; baseFps=%.3f < 26 -> cycle=0 (field-match only).\n"),
                inputFps);
            prm->ivtc.cycle = 0;
        } else {
            AddMessage(RGY_LOG_DEBUG,
                _T("ivtc: no pre-scan data; baseFps=%.3f >= 26 -> cycle=5 drop=1.\n"),
                inputFps);
            prm->ivtc.cycle = 5;
            prm->ivtc.drop  = 1;
        }
    }

    // cadenceLock auto-resolution: cadenceLock=-1 (auto) enables the 5-frame
    // pattern tracker automatically when guide mode is active (guide >= 1).
    // The tracker is inert on non-pulldown content (no phase ever reaches
    // the 4/5 fit threshold), so enabling it for guide-mode users is safe
    // and matches user intent. Explicit cadlock=off overrides when needed.
    if (prm->ivtc.cadenceLock < 0) {
        prm->ivtc.cadenceLock = (prm->ivtc.guide >= 1) ? 1 : 0;
        AddMessage(RGY_LOG_DEBUG,
            _T("ivtc: cadenceLock auto-resolved to %s (guide=%d).\n"),
            prm->ivtc.cadenceLock ? _T("on") : _T("off"),
            prm->ivtc.guide);
    }

    // デシメーションが有効な場合は 1-in / 1-out ではなくなるので、
    // 基底クラスのタイムスタンプ path-through を外す (出力側で手動管理)。
    // 同時に出力 baseFps を (cycle-drop)/cycle 倍しておく (デコーダ側の
    // 期待フレームレートを正しく 24fps に更新する)。
    //
    // FIX A (2026-04-24): the decimation multiplier always applies when
    // cycle>0 and drop>0, regardless of m_skipBaseFpsMultiplier. The old
    // guard assumed the internal 4->5 expansion cancelled the 5->4
    // decimation and left the rate unchanged, but that only affects the
    // synth-insertion step -- the decimation itself genuinely removes
    // frames from the output stream, so the output rate must be reduced.
    // Without this adjustment DH2 expand=on produced 8,001 frames at
    // 29.97fps (267s) instead of the expected 23.976fps (334s), a 25%
    // playback speed-up. See analysis/vpp_ivtc_progress.txt.
    if (m_mixedActive) {
        m_pathThrough &= ~(FILTER_PATHTHROUGH_TIMESTAMP);
    } else if (prm->ivtc.cycle > 0) {
        m_pathThrough &= ~(FILTER_PATHTHROUGH_TIMESTAMP);
        if (prm->ivtc.drop > 0) {
            pParam->baseFps *= rgy_rational<int>(prm->ivtc.cycle - prm->ivtc.drop, prm->ivtc.cycle);
        }
    } else {
        m_pathThrough |= FILTER_PATHTHROUGH_TIMESTAMP;
    }

    // m_frameBuf レイアウト (cycle>0 時):
    //   [0 .. cycleLen-1]            : サイクル蓄積バッファ
    //   [cycleLen]                   : 前サイクル末尾の保存スロット (SAD 比較用)
    //   [cycleLen+1 .. stagingEnd-1] : emit-staging (cycleLen - drop 枚)
    // flushCycle がここで決まった emit 候補を staging にコピーして m_emitQueue に積み、
    // popEmit が同一call内でまとめて下流へ返す。次サイクルは cycle[0..cycleLen-1] を
    // 安心して上書きできる。
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int stagingCount = (cycleLen > 0) ? (cycleLen - prm->ivtc.drop) : 0;
    const int mixedDirectStagingCount = (m_mixedActive && cycleLen > 0) ? std::max(stagingCount, 1) : 0;
    const int bufCount = (cycleLen > 0) ? (cycleLen + 1 + stagingCount + mixedDirectStagingCount) : 1;
    m_stagingBase = (cycleLen > 0) ? (cycleLen + 1) : 0;
    m_mixedDirectStagingBase = m_stagingBase + stagingCount;
    m_mixedDirectStagingCount = mixedDirectStagingCount;
    m_mixedDirectStagingNext = 0;
    sts = AllocFrameBuf(prm->frameOut, bufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // prev/cur/next 用にリングバッファを確保 (元フレームはインタレ扱いのまま保持)
    if ((int)m_cacheFrames.size() != IVTC_CACHE_SIZE
        || !prmPrev
        || cmpFrameInfoCspResolution(&m_cacheFrames[0]->frame, &prm->frameIn)) {
        m_cacheFrames.clear();
        for (int i = 0; i < IVTC_CACHE_SIZE; i++) {
            auto clframe = createCUFrameBuffer(prm->frameIn);
            if (!clframe) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame %d.\n"), i);
                return RGY_ERR_MEMORY_ALLOC;
            }
            m_cacheFrames.push_back(std::move(clframe));
        }
    }

    // RFF expansion buffers. Allocate one CUFrameBuf per buffer slot
    // (EXPAND_BUF_SIZE) plus the carry frame and synth scratch. The
    // buffer holds raw decoded frames verbatim; the carry frame holds
    // an open-slot frame that crosses a flush boundary.
    if (m_expandActive) {
        for (int i = 0; i < EXPAND_BUF_SIZE; i++) {
            if (!m_expandBuf[i] || cmpFrameInfoCspResolution(&m_expandBuf[i]->frame, &prm->frameIn)) {
                m_expandBuf[i] = createCUFrameBuffer(prm->frameIn);
                if (!m_expandBuf[i]) {
                    AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate expand buffer slot %d.\n"), i);
                    return RGY_ERR_MEMORY_ALLOC;
                }
            }
            m_expandBufMeta[i] = ExpandBufMeta{0, RGY_PICSTRUCT_UNKNOWN, AV_NOPTS_VALUE, 0, -1};
        }
        if (!m_expandCarryFrame || cmpFrameInfoCspResolution(&m_expandCarryFrame->frame, &prm->frameIn)) {
            m_expandCarryFrame = createCUFrameBuffer(prm->frameIn);
            if (!m_expandCarryFrame) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate expand carry frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        if (!m_expandSynth || cmpFrameInfoCspResolution(&m_expandSynth->frame, &prm->frameIn)) {
            m_expandSynth = createCUFrameBuffer(prm->frameIn);
            if (!m_expandSynth) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate expand synth buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        m_expandSynthCount     = 0;
        m_expandCodedEmitCount = 0;
        // m_expandRffSeen is set during buildScheduleFromScan(); don't
        // zero it here or the close() stats line would be wrong.
    } else {
        for (auto &buf : m_expandBuf) buf.reset();
        m_expandCarryFrame.reset();
        m_expandSynth.reset();
    }
    if (m_mixedActive) {
        if (!m_mixedRffPendingTopFrame || cmpFrameInfoCspResolution(&m_mixedRffPendingTopFrame->frame, &prm->frameIn)) {
            m_mixedRffPendingTopFrame = createCUFrameBuffer(prm->frameIn);
            if (!m_mixedRffPendingTopFrame) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate mixed RFF pending top frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        if (!m_mixedRffPendingBottomFrame || cmpFrameInfoCspResolution(&m_mixedRffPendingBottomFrame->frame, &prm->frameIn)) {
            m_mixedRffPendingBottomFrame = createCUFrameBuffer(prm->frameIn);
            if (!m_mixedRffPendingBottomFrame) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate mixed RFF pending bottom frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        if (!m_expandSynth || cmpFrameInfoCspResolution(&m_expandSynth->frame, &prm->frameIn)) {
            m_expandSynth = createCUFrameBuffer(prm->frameIn);
            if (!m_expandSynth) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: failed to allocate mixed RFF synth buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        resetMixedRffState();
    } else {
        m_mixedRffPendingTopFrame.reset();
        m_mixedRffPendingBottomFrame.reset();
        resetMixedRffState();
    }

    // スコア集計バッファ: WG ごとに 9 uints = [mC, mP, mN, cC, cP, cN, bC, bP, bN]
    //   mX = match-quality   (3-top vs 2-bot pattern diff sum per block)
    //   cX = combing-count   (zigzag pixels per block — per-pixel sum)
    //   bX = combed-block flag (SUB-PHASE 1, 2026-04-24; 0/1 binary per WG,
    //        indicating whether that WG's cX sum met BLOCK_COMB_THRESH)
    // CPU 側:
    //   mX -> trimmed-average across WGs
    //   cX -> trimmed-average (primary) + MAX across WGs (Priority 1)
    //   bX -> SUM across WGs = frame-wide count of combed blocks
    const int wg_count_x = (prm->frameIn.width  + IVTC_BLOCK_X - 1) / IVTC_BLOCK_X;
    const int wg_count_y = (prm->frameIn.height + IVTC_BLOCK_Y - 1) / IVTC_BLOCK_Y;
    const size_t wg_count = (size_t)wg_count_x * wg_count_y;
    const size_t score_count = wg_count * 9;
    if (!m_scoreBuf || m_scoreHost.size() != score_count) {
        m_scoreBuf = std::make_unique<CUMemBuf>(score_count * sizeof(uint32_t));
        if (m_scoreBuf->alloc() != RGY_ERR_NONE) m_scoreBuf.reset();
        if (!m_scoreBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate score buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_scoreHost.assign(score_count, 0u);
    }
    // SAD 集計用バッファ (WGごとに uint×1)。
    // NOTE: allocated unconditionally (not gated on cycleLen > 0) because
    // processInputToCycle's scene-change detection (added 2026-04-21) uses
    // computePairDiff EVERY frame regardless of decimation. Previously this
    // buffer was only created when cycle-decimation was on, which caused a
    // null-deref crash on ~24fps inputs (auto-cycle=0) — the crash cascaded
    // into encoder output init failing with AVERROR_INVALIDDATA.
    if (!m_diffBuf || m_diffHost.size() != wg_count) {
        m_diffBuf = std::make_unique<CUMemBuf>(wg_count * sizeof(uint32_t));
        if (m_diffBuf->alloc() != RGY_ERR_NONE) m_diffBuf.reset();
        if (!m_diffBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate diff buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_diffHost.assign(wg_count, 0u);
    }
    // サイクルのメタデータ用ベクトル
    if (cycleLen > 0) {
        m_cycleInPts.assign(cycleLen, 0);
        m_cycleInDur.assign(cycleLen, 0);
        m_cycleInputIds.assign(cycleLen, 0);
        m_cycleMatchScore.assign(cycleLen, 0);
        m_cycleCombScore.assign(cycleLen, 0);
        m_cycleCombMax.assign(cycleLen, 0);
        m_cycleCombBlocks.assign(cycleLen, 0);
        // SUB-PHASE 2: dual-parity per-candidate metric vectors.
        // Default-init each slot to {0,0,0} via std::array{} value-init.
        m_cycleCombScorePrim.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleCombMaxPrim.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleCombBlocksPrim.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleCombScoreAlt.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleCombMaxAlt.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleCombBlocksAlt.assign(cycleLen, std::array<uint64_t, 3>{});
        m_cycleMatchAltParity.assign(cycleLen, 0);
        m_cycleConfidence.assign(cycleLen, 0);
        m_cycleBlendTrigger.assign(cycleLen, 0);
        m_cycleMatchType.assign(cycleLen, 0);
        m_cycleApplyBlend.assign(cycleLen, 0);
        m_cycleDecTag.assign(cycleLen, 0);
        m_cycleCadenceTag.assign(cycleLen, 0);
        m_cycleDiffPrev.assign(cycleLen, 0);
        m_cycleSceneSAD.assign(cycleLen, 0);
        m_cycleIsSynth.assign(cycleLen, 0);
    }

    // 入力の picstruct から TFF/BFF を決定。明示指定があればそれを優先。
    if (prm->ivtc.tff >= 0) {
        m_tffFixed = prm->ivtc.tff;
    } else if (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) {
        m_tffFixed = 0;
    } else {
        m_tffFixed = 1; // デフォルトは TFF
    }

    if (prm->ivtc.log && prm->ivtc.logPath.length() > 0 && !m_fpLog) {
#if defined(_WIN32) || defined(_WIN64)
        FILE *fp = nullptr;
        _wfopen_s(&fp, prm->ivtc.logPath.c_str(), L"w");
        m_fpLog.reset(fp);
#else
        m_fpLog.reset(fopen(tchar_to_string(prm->ivtc.logPath).c_str(), "w"));
#endif
        if (m_fpLog) {
            fprintf(m_fpLog.get(),
                "# DEC: decoder-output view (picstruct + RGY flags, the bitstream-level\n"
                "#      progressive/tff/rff bits are not preserved through the RGY path,\n"
                "#      so DEC uses the RGY flags that QSVEnc actually received)\n"
                "# IN:  per-input-frame record as seen by IVTC (framenum tracks inputCount)\n"
                "# OUT: per-output-frame IVTC decision (header fields below)\n"
                "#out_idx\tin_id\tmatch\tmatchParity\tconf\tpost\tstatus\tdec\tcadence\tmQ\tcComb\tcCombMax\tcCombBlocks\tcComb_c\tcComb_p\tcComb_n\tcComb_cA\tcComb_pA\tcComb_nA\tcMax_c\tcMax_p\tcMax_n\tcMax_cA\tcMax_pA\tcMax_nA\tcBlk_c\tcBlk_p\tcBlk_n\tcBlk_cA\tcBlk_pA\tcBlk_nA\tbtrig\tpostComb\tdiff_to_prev\tscene_sad%s\n"
                "%s"
                "# cComb_{c,p,n}     = primary-parity per-candidate trimmed-avg combing\n"
                "# cComb_{cA,pA,nA}  = alternate-parity (!tff) per-candidate trimmed-avg combing\n"
                "# cMax_*  / cBlk_*  = same naming for block-MAX (Priority 1) and combed-block-count (SUB-PHASE 1) signals\n"
                "# matchParity: which parity comb-first selection landed on (SUB-PHASE 3)\n"
                "#   pri = primary tff    alt = !tff (decisive-alt guard cleared)\n"
                "# conf: confidence-based classifier (SUB-PHASE 5)\n"
                "#   0=HIGH 1=MEDIUM 2=LOW 3=VERY_LOW\n"
                "# btrig: blend trigger classifier (INSTRUMENTATION)\n"
                "#   0=none 1=mislabeledBySat 2=mislabeledByDual 3=unknownCombed\n"
                "#   4=progressiveCombed 5=strongMatch 6=vthresh_vetoed 7=confidenceForced\n",
                m_mixedActive ? "\tsection" : "",
                m_mixedActive ? "# section: mixed section tag; rff/pass/interlaced\n" : "");
            fflush(m_fpLog.get());
        }
    }

    // 状態リセット (ストリーム切り替え時など)
    if (!prmPrev || prmPrev->ivtc != prm->ivtc) {
        m_inputCount = 0;
        m_processedCount = 0;
        m_lastSceneChange = false;
        m_lastSceneSAD = 0;
        m_cycleFilled = 0;
        m_hasSaveSlot = false;
        m_outputFrameCount = 0;
        m_emitQueue.clear();
        m_nPts = 0;
        m_nPtsInit = false;
        m_cfrBaseDur = 0;
        m_cfrEmitIdx = 0;
        resetMixedTemporalState();
        resetMixedRffState();
        resetCadenceState();
    }

    // DIAG #3: end-of-init state. If this disagrees with DIAG #2,
    // something between the dispatch and here is mutating cycle/drop.
    // All numeric args pre-extracted to strict int locals; %08x replaced
    // with %u to eliminate any enum-to-unsigned-hex format quirk that
    // the prior /GS-cookie crash (0xC0000409) hinted at.
    if (m_pLog) {
        const int cycleVal     = prm->ivtc.cycle;
        const int dropVal      = prm->ivtc.drop;
        const int baseN        = prm->baseFps.n();
        const int baseD        = prm->baseFps.d();
        const int pathVal      = (int)(uint32_t)m_pathThrough;  // cast to int, use %d (not %u)
        const int actVal       = (int)m_expandActive;
        const int dispCountVal = m_displayFrameCount;
        m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
            _T("ivtc DIAG: init complete: cycle=%d drop=%d baseFps=%d/%d pathThrough=%d m_expandActive=%d m_displayFrameCount=%d mixed=%d\n"),
            cycleVal, dropVal, baseN, baseD, pathVal, actVal, dispCountVal, (int)m_mixedActive);
    }

    setFilterInfo(prm->print() + _T("\n                         tff=")
        + (m_tffFixed ? _T("on") : _T("off"))
        + _T(", expand=") + (m_expandActive ? _T("active") : _T("off"))
        + _T(", mixed=") + (m_mixedActive ? _T("active") : _T("off")));
    m_param = prm;
    return sts;
}

// Format picstruct as a human-readable tag. Multiple bits can be set; priority
// order here is: explicit field order > plane-level FRAME/INTERLACED. Returns
// short strings suitable for TSV log columns.
static const char *picstructToTag(RGY_PICSTRUCT ps) {
    const bool tff = (ps & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF;
    const bool bff = (ps & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF;
    const bool interlaced = (ps & RGY_PICSTRUCT_INTERLACED) != 0;
    const bool frame = (ps & RGY_PICSTRUCT_FRAME) != 0;
    if (tff)             return "TFF";
    if (bff)             return "BFF";
    if (interlaced)      return "INTERLACED";
    if (frame)           return "FRAME";
    return "UNKNOWN";
}

void NVEncFilterIvtc::logInputFrame(const RGYFrameInfo *pInputFrame, int frameNum) {
    if (!m_fpLog || !pInputFrame) return;
    const char *picTag = picstructToTag(pInputFrame->picstruct);
    const int rff      = (pInputFrame->flags & RGY_FRAME_FLAG_RFF)     ? 1 : 0;
    const int rff_tff  = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_TFF) ? 1 : 0;
    const int rff_bff  = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_BFF) ? 1 : 0;
    const int rff_copy = (pInputFrame->flags & RGY_FRAME_FLAG_RFF_COPY)? 1 : 0;

    // DEC: the view QSVEnc received from the demux/decode + RGY CSP conversion.
    // All RGY flags that derive from the MPEG-2 parser's repeat_pict + field_order
    // are shown. The underlying bitstream bits (progressive_frame, top_field_first,
    // repeat_first_field) are NOT preserved individually through the RGY pipeline
    // -- they are collapsed into these flags by rgy_input_avcodec.cpp:2954-3513.
    fprintf(m_fpLog.get(),
        "DEC:\tframe=%d\tpicstruct=%s\trff=%d\trff_tff=%d\trff_bff=%d\trff_copy=%d\tpts=%lld\tdur=%lld\n",
        frameNum, picTag, rff, rff_tff, rff_bff, rff_copy,
        (long long)pInputFrame->timestamp, (long long)pInputFrame->duration);

    // IN: the same frame, IVTC-centric view (dropped columns that IVTC doesn't use).
    fprintf(m_fpLog.get(),
        "IN:\tframe=%d\tpicstruct=%s\trff=%d\tpts=%lld\tdur=%lld\n",
        frameNum, picTag, rff,
        (long long)pInputFrame->timestamp, (long long)pInputFrame->duration);
    fflush(m_fpLog.get());
}

void NVEncFilterIvtc::resetCadenceState() {
    m_lastMatch             = -1;
    m_lastMatchScore        = 0;
    m_cadenceFill           = 0;
    m_cadenceIndex          = 0;
    m_cadenceLockedPhase    = -1;
    m_cadenceConfidence     = 0;
    m_cadenceLastPrediction = -1;
    m_cadenceHistory.fill(0);
}

// 5-frame 3:2 cadence tracker. Classic NTSC pulldown produces one of 5 canonical
// C/P/N patterns depending on phase:
//   phase 0 : C C C N P
//   phase 1 : C C N P C
//   phase 2 : C N P C C
//   phase 3 : N P C C C
//   phase 4 : P C C C N
// (derived from the 3:2 field-repeat pattern with TFF semantics).
// Match P is rare in well-detected streams; typical phases show CCCNC or CCCCN
// due to per-frame argmin picking C when motion is low. We accept the two
// dominant patterns (NP and PN segments) as the disambiguator for phase lock.
//
// observedMatch: 0=C, 1=P, 2=N
// returns: -1 if no prediction available (not locked), else predicted match for
//          the NEXT frame.
int NVEncFilterIvtc::updateCadence(int observedMatch) {
    // Phase B3 rewrite (2026-04-22): CCN-only patterns ported from TFM's
    // PredictHardYUY2 (Telecide.h:432-464). Previously we tracked C/P/N
    // rotations with P in them; TFM's patterns are pure C/N because P is
    // only produced via back-on-combed fallback, not by clean pulldown.
    //
    // Encoding: IvtcMatch::C=0, IvtcMatch::P=1, IvtcMatch::N=2. This is
    // different from TFM's P=0/C=1/N=2 encoding, but the patterns
    // themselves are the same shape.
    //
    // guide=1 (GUIDE_32, NTSC 3:2): base cycle "C C C N N" — 5 rotations.
    // guide=2 (GUIDE_22, PAL 2:2): base cycle "C N" — 2 rotations, each
    //   repeated to fill the 5-frame window as "C N C N C" or "N C N C N".
    // guide=0 or any other: no cadence tracking (caller should skip
    //   updateCadence or pass guide=1 to keep the existing behaviour).
    //
    // Record the observation into the ring buffer.
    m_cadenceHistory[m_cadenceIndex] = observedMatch;
    m_cadenceIndex = (m_cadenceIndex + 1) % IVTC_CADENCE_LEN;
    if (m_cadenceFill < IVTC_CADENCE_LEN) m_cadenceFill++;

    // Verify last call's prediction BEFORE the warmup early-return so the
    // ledger ticks as soon as predictions start flowing. Even during
    // warmup (m_cadenceFill < LEN), if we have a stored prediction from a
    // previous locked state (e.g. immediately after history clear but
    // with carry-over state), verify and clear it.
    constexpr int MAX_CONFIDENCE = 255;
    constexpr int LOCK_THRESHOLD = 4;
    if (m_cadenceLastPrediction >= 0) {
        if (m_cadenceLastPrediction == observedMatch) {
            m_cadenceConfidence = std::min(MAX_CONFIDENCE, m_cadenceConfidence + 2);
        } else {
            m_cadenceConfidence = std::max(0, m_cadenceConfidence - 1);
        }
        m_cadenceLastPrediction = -1;   // cleared; re-set below if still locked
    }

    // Need a full 5-frame window before attempting to lock.
    if (m_cadenceFill < IVTC_CADENCE_LEN) {
        return -1;
    }

    // Read history in chronological order (oldest first).
    int ordered[IVTC_CADENCE_LEN];
    for (int i = 0; i < IVTC_CADENCE_LEN; i++) {
        ordered[i] = m_cadenceHistory[(m_cadenceIndex + i) % IVTC_CADENCE_LEN];
    }

    // Determine which pattern table to use based on the guide parameter.
    // The param is cached via m_param; reacquire the shared_ptr once.
    int guide = 1;  // default to 3:2 pulldown tracking
    if (auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param)) {
        guide = prm->ivtc.guide;
    }

    // GUIDE_32 (NTSC 3:2): 5 rotations of "C C C N N".
    //   Encoding: C=0, N=2 (IvtcMatch enum).
    // GUIDE_22 (PAL 2:2): 2 rotations of "C N", repeated to 5-frame window.
    //
    // Each row of the pattern table is what the 5-frame observation window
    // SHOULD look like at a given phase offset.
    //
    // Phase 0 means "observed = row 0"; the next frame's expected match is
    // pattern[row 0 rotated by +1 slot][last slot], i.e. phase 1's last
    // observation.
    static constexpr int PATTERNS_32[IVTC_CADENCE_LEN][IVTC_CADENCE_LEN] = {
        { 0, 0, 0, 2, 2 },  // phase 0 : CCCNN
        { 0, 0, 2, 2, 0 },  // phase 1 : CCNNC
        { 0, 2, 2, 0, 0 },  // phase 2 : CNNCC
        { 2, 2, 0, 0, 0 },  // phase 3 : NNCCC
        { 2, 0, 0, 0, 2 },  // phase 4 : NCCCN
    };
    // GUIDE_22 table. Only two distinct phases exist; the other three
    // rows are duplicates to match the fixed IVTC_CADENCE_LEN array
    // shape. Best-phase scoring still works — the duplicates can't
    // beat their originals.
    static constexpr int PATTERNS_22[IVTC_CADENCE_LEN][IVTC_CADENCE_LEN] = {
        { 0, 2, 0, 2, 0 },  // phase 0 : CNCNC
        { 2, 0, 2, 0, 2 },  // phase 1 : NCNCN
        { 0, 2, 0, 2, 0 },  // duplicate of phase 0
        { 2, 0, 2, 0, 2 },  // duplicate of phase 1
        { 0, 2, 0, 2, 0 },  // duplicate of phase 0
    };

    const int (*PATTERNS)[IVTC_CADENCE_LEN] = PATTERNS_32;
    if (guide == 2) {
        PATTERNS = PATTERNS_22;
    } else if (guide == 0) {
        // Disabled: do not emit predictions. History is still tracked for
        // debug logging, but the caller will not consult the prediction.
        m_cadenceLastPrediction = -1;
        return -1;
    }

    // Score each phase hypothesis by counting matches to observed history.
    int bestPhase = -1;
    int bestScore = -1;
    for (int p = 0; p < IVTC_CADENCE_LEN; p++) {
        int s = 0;
        for (int i = 0; i < IVTC_CADENCE_LEN; i++) {
            if (ordered[i] == PATTERNS[p][i]) s++;
        }
        if (s > bestScore) { bestScore = s; bestPhase = p; }
    }

    // Lock / unlock logic (history-based fit, orthogonal to the prediction
    // ledger above). Enters a lock when 4+/5 history slots fit the phase;
    // drops the lock when the ledger falls below LOCK_THRESHOLD — the
    // history-fit still steers which phase we're on, but the ledger now
    // decides how much to trust that phase.
    if (bestScore >= LOCK_THRESHOLD) {
        if (m_cadenceLockedPhase != bestPhase) {
            // New lock or phase slip: reset ledger to just-above-threshold.
            m_cadenceLockedPhase = bestPhase;
            m_cadenceConfidence  = std::max(m_cadenceConfidence, LOCK_THRESHOLD);
        }
    } else if (m_cadenceConfidence < LOCK_THRESHOLD) {
        // History stopped fitting AND ledger too weak to hold the lock.
        m_cadenceLockedPhase = -1;
    }

    // If locked, emit a prediction for next call's observation. The
    // predicted NEXT observation is the LAST slot of the phase-rotated-by-1
    // row (equivalent to pattern continuing one position forward).
    int prediction = -1;
    if (m_cadenceLockedPhase >= 0) {
        const int nextPhase = (m_cadenceLockedPhase + 1) % IVTC_CADENCE_LEN;
        prediction = PATTERNS[nextPhase][IVTC_CADENCE_LEN - 1];
    }
    m_cadenceLastPrediction = prediction;
    return prediction;
}

RGY_ERR NVEncFilterIvtc::scoreCandidates(const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, uint64_t matchScoreOut[3], uint64_t combScoreOut[3], uint64_t combMaxOut[3], uint64_t combBlocksOut[3], const int tffForScoring, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[cur->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int nt = std::max(1, (maxVal * 10) / 255);
    const int T  = std::max(1, (maxVal *  4) / 255);
    const int y0 = std::max(0, prm->ivtc.y0);
    const uint32_t CLIP_THRESH = std::max<uint32_t>(4u, (uint32_t)(nt * 4));

    auto scorePlane = [&](const RGYFrameInfo &planePrev, const RGYFrameInfo &planeCur, const RGYFrameInfo &planeNext,
                          uint64_t out[6], uint64_t outMax[3], uint64_t outBlocks[3]) -> RGY_ERR {
        const int y1local = (prm->ivtc.y1 > 0) ? std::min(prm->ivtc.y1, planeCur.height - 1) : 0;
        auto err = run_ivtc_score_candidates(&planePrev, &planeCur, &planeNext, tffForScoring, nt, T, y0, y1local, (uint32_t *)m_scoreBuf->ptr, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at run_ivtc_score_candidates: %s.\n"), get_err_mes(err));
            return err;
        }

        const int wg_count_x = divCeil(planeCur.width,  IVTC_BLOCK_X);
        const int wg_count_y = divCeil(planeCur.height, IVTC_BLOCK_Y);
        const size_t wg_count = (size_t)wg_count_x * (size_t)wg_count_y;
        const size_t score_bytes = wg_count * 9 * sizeof(uint32_t);
        auto cudaerr = cudaMemcpyAsync(m_scoreHost.data(), m_scoreBuf->ptr, score_bytes, cudaMemcpyDeviceToHost, stream);
        if (cudaerr == cudaSuccess) cudaerr = cudaStreamSynchronize(stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

        uint64_t sum[6] = { 0, 0, 0, 0, 0, 0 };
        uint32_t cnt[6] = { 0, 0, 0, 0, 0, 0 };
        uint32_t maxC[3] = { 0, 0, 0 };
        uint64_t blocksC[3] = { 0, 0, 0 };
        for (size_t i = 0; i < wg_count; i++) {
            const uint32_t *e = &m_scoreHost[i * 9];
            for (int k = 0; k < 6; k++) {
                if (e[k] > CLIP_THRESH) { sum[k] += e[k]; cnt[k]++; }
            }
            maxC[0] = std::max(maxC[0], e[3]);
            maxC[1] = std::max(maxC[1], e[4]);
            maxC[2] = std::max(maxC[2], e[5]);
            blocksC[0] += e[6];
            blocksC[1] += e[7];
            blocksC[2] += e[8];
        }
        for (int k = 0; k < 6; k++) out[k] = (cnt[k] >= 2) ? (sum[k] / (uint64_t)cnt[k]) : 0ULL;
        outMax[0] = maxC[0]; outMax[1] = maxC[1]; outMax[2] = maxC[2];
        outBlocks[0] = blocksC[0]; outBlocks[1] = blocksC[1]; outBlocks[2] = blocksC[2];
        return RGY_ERR_NONE;
    };

    const auto planePrevY = getPlane(prev, RGY_PLANE_Y);
    const auto planeCurY  = getPlane(cur,  RGY_PLANE_Y);
    const auto planeNextY = getPlane(next, RGY_PLANE_Y);
    uint64_t result[6] = {}, resultMax[3] = {}, resultBlocks[3] = {};
    auto err = scorePlane(planePrevY, planeCurY, planeNextY, result, resultMax, resultBlocks);
    if (err != RGY_ERR_NONE) return err;

    const bool planesSeparate = (RGY_CSP_PLANES[cur->csp] >= 3);
    if (prm->ivtc.chroma && planesSeparate) {
        uint64_t scoreU[6] = {}, scoreUMax[3] = {}, scoreUBlocks[3] = {};
        uint64_t scoreV[6] = {}, scoreVMax[3] = {}, scoreVBlocks[3] = {};
        const auto planePrevU = getPlane(prev, RGY_PLANE_U);
        const auto planeCurU  = getPlane(cur,  RGY_PLANE_U);
        const auto planeNextU = getPlane(next, RGY_PLANE_U);
        err = scorePlane(planePrevU, planeCurU, planeNextU, scoreU, scoreUMax, scoreUBlocks);
        if (err != RGY_ERR_NONE) return err;
        const auto planePrevV = getPlane(prev, RGY_PLANE_V);
        const auto planeCurV  = getPlane(cur,  RGY_PLANE_V);
        const auto planeNextV = getPlane(next, RGY_PLANE_V);
        err = scorePlane(planePrevV, planeCurV, planeNextV, scoreV, scoreVMax, scoreVBlocks);
        if (err != RGY_ERR_NONE) return err;
        for (int k = 0; k < 6; k++) result[k] += (scoreU[k] + scoreV[k]) >> 2;
        for (int k = 0; k < 3; k++) {
            const uint64_t chromaMax = std::max<uint64_t>(scoreUMax[k], scoreVMax[k]) >> 2;
            resultMax[k] = std::max<uint64_t>(resultMax[k], chromaMax);
            resultBlocks[k] += (scoreUBlocks[k] + scoreVBlocks[k]) >> 2;
        }
    }

    matchScoreOut[0] = result[0]; matchScoreOut[1] = result[1]; matchScoreOut[2] = result[2];
    combScoreOut[0] = result[3]; combScoreOut[1] = result[4]; combScoreOut[2] = result[5];
    combMaxOut[0] = resultMax[0]; combMaxOut[1] = resultMax[1]; combMaxOut[2] = resultMax[2];
    combBlocksOut[0] = resultBlocks[0]; combBlocksOut[1] = resultBlocks[1]; combBlocksOut[2] = resultBlocks[2];
    return RGY_ERR_NONE;
}


RGY_ERR NVEncFilterIvtc::synthesizeToCycle(int cycleSlot, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const IvtcMatch match, const bool applyBlend, const int dthresh, const int tffForSynth, cudaStream_t stream) {
    auto err = run_ivtc_synthesize_frame(&m_frameBuf[cycleSlot]->frame, prev, cur, next, tffForSynth, (int)match, applyBlend ? 1 : 0, dthresh, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at run_ivtc_synthesize_frame: %s.\n"), get_err_mes(err));
    }
    return err;
}


// Full BWDIF path: fires when the post-processing gate decides the
// frame needs reconstruction. Reconstructs missing-
// field rows via motion-adaptive w3fdif using the full 5-frame temporal
// window (prev2/prev/cur/next/next2) from the IVTC ring. Caller passes
// aliased pointers (prev2==prev / next2==next) during ring-startup or
// drain when the real frames aren't available — BWDIF degrades cleanly
// to the 3-frame approximation in that case.
// Preserved rows come straight from cur.
RGY_ERR NVEncFilterIvtc::synthesizeToCycleBwdif(int cycleSlot, const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2, const int streamTff, const int sceneChange, const int dthresh, cudaStream_t stream) {
    auto err = run_ivtc_bwdif_frame(&m_frameBuf[cycleSlot]->frame, prev2, prev, cur, next, next2, streamTff ? 1 : 0, sceneChange, dthresh, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at run_ivtc_bwdif_frame: %s.\n"), get_err_mes(err));
    }
    return err;
}


RGY_ERR NVEncFilterIvtc::computePairDiff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint64_t &diffOut, cudaStream_t stream) {
    auto err = run_ivtc_frame_diff(pA, pB, (uint32_t *)m_diffBuf->ptr, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at run_ivtc_frame_diff: %s.\n"), get_err_mes(err));
        return err;
    }
    const size_t bytes = m_diffHost.size() * sizeof(uint32_t);
    auto cudaerr = cudaMemcpyAsync(m_diffHost.data(), m_diffBuf->ptr, bytes, cudaMemcpyDeviceToHost, stream);
    if (cudaerr == cudaSuccess) cudaerr = cudaStreamSynchronize(stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    uint64_t sum = 0;
    for (const uint32_t v : m_diffHost) sum += v;
    diffOut = sum;
    return RGY_ERR_NONE;
}


// Overlay one field's rows from src onto dst via stride-doubled kernel.
// DGDecode vfapidec.cpp:1027-1059 equivalent (CopyBot for tff=1, CopyTop for tff=0).
// Applies to every plane with the plane's own dimensions.
RGY_ERR NVEncFilterIvtc::overlayField(CUFrameBuf *dst, const CUFrameBuf *src, int tff, cudaStream_t stream) {
    if (!dst || !src) return RGY_ERR_NULL_PTR;
    auto err = run_ivtc_field_overlay(&dst->frame, &src->frame, tff, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at run_ivtc_field_overlay: %s.\n"), get_err_mes(err));
    }
    return err;
}


// Separate libavcodec pre-scan of the entire input file. Opens a
// PRIVATE AVFormatContext + decoder (no shared state with QSVEnc's
// main reader) and does a FULL decode of every video frame -- reading
// repeat_pict / top_field_first / interlaced_frame directly off the
// resulting AVFrame. This matches exactly what the runtime reader
// does at rgy_input_avcodec.cpp:3507-3513 so per-frame flag alignment
// is guaranteed frame-for-frame.
//
// WHY FULL DECODE (not parser-only)? The MPEG-2 bitstream parser's
// `field_order` field is typically reported as AV_FIELD_PROGRESSIVE
// for soft-telecine pictures (progressive_frame=1) -- it does NOT
// expose top_field_first reliably. Only the decoded AVFrame carries
// that bit in a form that matches the runtime. Tried parser-only
// first; every RFF frame stashed into the same pending slot (all TFF
// or all BFF according to a fallback heuristic), producing either 0
// synths or all-synth runaway. Full decode is ~20x slower per frame
// than parser-only but still finishes in <30 s for a 2-hour DVD, and
// correctness trumps the speed gap at init.
//
// Returns a vector of per-frame metadata in DECODE order (the same
// order NVEncFilterIvtc::run_filter will see). The caller is
// responsible for applying the trim offset when walking the vector.
static RGY_ERR ivtcPreScanInput(const tstring &inputPath,
                                 std::vector<IvtcPreScanFrame> &frames,
                                 std::shared_ptr<RGYLog> log) {
    frames.clear();
    if (inputPath.empty()) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("ivtc prescan: input path is empty\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    std::string filenameUtf8;
    // The 3-arg tchar_to_string overload takes a const TCHAR*, so we
    // pass inputPath.c_str() (matches the existing pattern at
    // rgy_input_avcodec.cpp:1502).
    if (0 == tchar_to_string(inputPath.c_str(), filenameUtf8, CP_UTF8)) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("ivtc prescan: failed to convert filename to utf-8\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (const auto protocol = unsupportedProbeProtocol(filenameUtf8); protocol != nullptr) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: input \"%s\" uses %s protocol, which cannot be pre-scanned safely. ")
            _T("--vpp-ivtc expand requires a re-openable local file input.\n"),
            inputPath.c_str(), char_to_tstring(protocol).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    // Silence libav "[mpeg2video] ac-tex damaged" / "Invalid mb type"
    // clutter for the duration of the pre-scan. DVD m2v streams often
    // have start-of-GOP decoder slop that's normally absorbed by
    // QSVEnc's main reader (which sees more context). For the pre-scan
    // we only care about the picture-header bits (repeat_pict / tff),
    // which the decoder resolves even when macroblock coefficients are
    // mangled -- so these error-level messages are cosmetic here.
    //
    // Scope the suppression to this function: save the current level,
    // set AV_LOG_FATAL (silences ERROR/WARNING/INFO; keeps genuine
    // fatals visible), restore via RAII when we return.
    const int savedAvLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    struct AvLogLevelRestorer {
        int prev;
        ~AvLogLevelRestorer() { av_log_set_level(prev); }
    } avLogGuard{savedAvLogLevel};

    AVFormatContext *fmtCtxRaw = nullptr;
    int avret = avformat_open_input(&fmtCtxRaw, filenameUtf8.c_str(), nullptr, nullptr);
    if (avret < 0) {
        // qsv_av_err2str returns tstring already -- no char_to_tstring wrap
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: avformat_open_input failed for \"%s\": %s\n"),
            inputPath.c_str(), qsv_av_err2str(avret).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    // RGYAVDeleter pattern (rgy_filter_subburn.cpp:511) -- canonical
    // for avformat contexts in this codebase.
    std::unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> fmtCtxGuard(
        fmtCtxRaw, RGYAVDeleter<AVFormatContext>(avformat_close_input));
    AVFormatContext *fmtCtx = fmtCtxGuard.get();

    if ((avret = avformat_find_stream_info(fmtCtx, nullptr)) < 0) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: avformat_find_stream_info failed: %s\n"),
            qsv_av_err2str(avret).c_str());
        return RGY_ERR_UNKNOWN;
    }

    const int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIdx < 0) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: no video stream found\n"));
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    AVStream *vst = fmtCtx->streams[videoIdx];

    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: avcodec_find_decoder failed for codec %s\n"),
            char_to_tstring(avcodec_get_name(vst->codecpar->codec_id)).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    AVCodecContext *codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecCtxGuard(
        codecCtxRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    AVCodecContext *codecCtx = codecCtxGuard.get();
    if ((avret = avcodec_parameters_to_context(codecCtx, vst->codecpar)) < 0) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: avcodec_parameters_to_context failed: %s\n"),
            qsv_av_err2str(avret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    codecCtx->time_base    = vst->time_base;
    codecCtx->pkt_timebase = vst->time_base;
    // Open the decoder for real (avcodec_open2) -- we need AVFrame
    // output to read top_field_first reliably.
    if ((avret = avcodec_open2(codecCtx, codec, nullptr)) < 0) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
            _T("ivtc prescan: avcodec_open2 failed: %s\n"),
            qsv_av_err2str(avret).c_str());
        return RGY_ERR_UNKNOWN;
    }

    AVPacket *pktRaw = av_packet_alloc();
    if (!pktRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> pktGuard(
        pktRaw, RGYAVDeleter<AVPacket>(av_packet_free));
    AVPacket *pkt = pktGuard.get();

    AVFrame *frameRaw = av_frame_alloc();
    if (!frameRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameGuard(
        frameRaw, RGYAVDeleter<AVFrame>(av_frame_free));
    AVFrame *frame = frameGuard.get();

    // Drain any frames the decoder already has queued before or after a
    // send_packet call. Reads out to `frames` until EAGAIN / EOF.
    auto drainDecoder = [&]() -> int {
        for (;;) {
            int ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return ret;
            if (ret < 0) return ret;
            IvtcPreScanFrame f;
            // AVFrame::repeat_pict encodes the number of *extra* field
            // half-periods the display should be delayed by:
            //   0 : normal 2-field frame
            //   1 : MPEG-2 soft-telecine RFF on interlaced sequence
            //       (+ progressive_frame=1), i.e. 3-field display
            //   2 / 4 : other RFF combos (progressive-sequence RFFs)
            // Any non-zero value means "repeat a field" -> RFF. The
            // runtime demux reader at rgy_input_avcodec.cpp:3507
            // compares against the PARSER's repeat_pict (not the
            // AVFrame's) using `> 1`, which works because MPEG-2
            // parser values are 2/4 for RFF. For the AVFrame path we
            // must use `> 0` or interlaced MPEG-2 RFFs (repeat_pict=1)
            // would be missed.
            // rgy_avframe_tff_flag / rgy_avframe_interlaced wrap the
            // FFmpeg-version-dependent AV_FRAME_FLAG_TOP_FIELD_FIRST /
            // AV_FRAME_FLAG_INTERLACED vs. the legacy struct fields.
            f.rff         = (frame->repeat_pict > 0);
            f.tff         = rgy_avframe_tff_flag(frame);
            f.progressive = !rgy_avframe_interlaced(frame);
            frames.push_back(f);
            av_frame_unref(frame);
        }
    };

    const auto scanStart = std::chrono::steady_clock::now();
    while (av_read_frame(fmtCtx, pkt) >= 0) {
        if (pkt->stream_index != videoIdx) {
            av_packet_unref(pkt);
            continue;
        }
        int sendRet = avcodec_send_packet(codecCtx, pkt);
        if (sendRet < 0 && sendRet != AVERROR(EAGAIN)) {
            // Non-fatal send failure (e.g. corrupt frame) -- skip packet.
            av_packet_unref(pkt);
            continue;
        }
        int drainRet = drainDecoder();
        av_packet_unref(pkt);
        if (drainRet < 0 && drainRet != AVERROR(EAGAIN) && drainRet != AVERROR_EOF) {
            if (log) log->write(RGY_LOG_WARN, RGY_LOGT_VPP,
                _T("ivtc prescan: avcodec_receive_frame failed: %s\n"),
                qsv_av_err2str(drainRet).c_str());
            return RGY_ERR_UNKNOWN;
        }
    }
    // Flush decoder: send NULL to signal EOF, then drain any buffered frames.
    avcodec_send_packet(codecCtx, nullptr);
    drainDecoder();

    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - scanStart).count();

    int rffCount = 0;
    int tffCount = 0;
    int bffCount = 0;
    for (const auto &f : frames) {
        if (f.rff) {
            rffCount++;
            if (f.tff) tffCount++;
            else       bffCount++;
        }
    }
    if (log) log->write(RGY_LOG_INFO, RGY_LOGT_VPP,
        _T("ivtc prescan: %d frames scanned (%d RFF = %.3f%%, RFF-TFF=%d RFF-BFF=%d) in %.1f ms\n"),
        (int)frames.size(), rffCount,
        frames.empty() ? 0.0 : (100.0 * rffCount / (double)frames.size()),
        tffCount, bffCount, elapsedMs);
    return RGY_ERR_NONE;
}

// Ring-buffer insertion helper. Copies pFrame into m_cacheFrames[slot],
// mirrors metadata, clamps negative inputFrameId, optionally writes the
// DEC:/IN: log lines, increments m_inputCount.
RGY_ERR NVEncFilterIvtc::pushFrameToRing(const RGYFrameInfo *pFrame, cudaStream_t stream,
                                       int logFrameNum, bool isSynth,
                                       bool isSynthFrame) {
    const int slot = m_inputCount % IVTC_CACHE_SIZE;
    RGYFrameInfo *pSlot = &m_cacheFrames[slot]->frame;
    auto copyErr = copyFrameAsync(pSlot, pFrame, stream);
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy %s frame to cache slot %d: %s.\n"),
            isSynth ? _T("synth") : _T("input"), slot, get_err_mes(copyErr));
        return copyErr;
    }
    pSlot->timestamp    = pFrame->timestamp;
    pSlot->duration     = pFrame->duration;
    pSlot->inputFrameId = pFrame->inputFrameId;
    pSlot->picstruct    = pFrame->picstruct;
    pSlot->flags        = pFrame->flags;
    if (pSlot->inputFrameId < 0) {
        pSlot->inputFrameId = m_inputCount;
    }
    // PATCH 1: mark the cache slot so processInputToCycle can bypass
    // score/match/cadence/blend for cross-time synth frames.
    m_cacheIsSynth[slot] = isSynthFrame;
    // DEC:/IN: log lines only for real decoder inputs — synth frames
    // are anonymous to the log (they show up as m_inputCount-indexed
    // entries without a DEC source, which matches DGDecode's convention
    // of making synth frames indistinguishable downstream).
    if (!isSynth) {
        logInputFrame(pFrame, logFrameNum);
    }
    m_inputCount++;
    return RGY_ERR_NONE;
}

// Port of DGDecode vfapidec.cpp:461-499 FrameList state machine. Operates
// on the pre-scanned per-frame RFF flags (see ivtcPreScanInput). Output
// indices in m_displayFrameList are POST-TRIM, i.e. 0-based against the
// stream of frames run_filter will receive at runtime. See
// analysis/dgdecode_trace.txt for an annotated 10-frame example.
void NVEncFilterIvtc::buildScheduleFromScan(
    const std::vector<IvtcPreScanFrame> &frames,
    int trimOffset,
    int trimFrameCount)
{
    m_displayFrameList.clear();
    m_displayFrameCount = 0;
    m_nextDisplayIdx    = 0;
    m_expandRffSeen     = 0;

    const int total     = (int)frames.size();
    const int firstFilm = std::max(0, trimOffset);
    const int lastFilm  = (trimFrameCount > 0)
                          ? std::min(total, firstFilm + trimFrameCount)
                          : total;

    // DGDecode state: "top" and "bottom" flags track whether the
    // currently-open slot has its .top or .bottom field already
    // committed but is still waiting for the other side. Both persist
    // across non-RFF frames (see dgdecode_trace.txt). pendingTopFilm /
    // pendingBotFilm record the POST-TRIM coded index that supplied the
    // committed side.
    bool top    = false;
    bool bottom = false;
    int  pendingTopFilm = -1;
    int  pendingBotFilm = -1;

    m_displayFrameList.reserve((size_t)(lastFilm - firstFilm) * 5 / 4 + 4);

    for (int film = firstFilm; film < lastFilm; film++) {
        const auto &f = frames[film];
        const int  localFilm = film - firstFilm;
        if (f.rff) m_expandRffSeen++;

        // --- STEP 1: complete pending half-open slot OR emit normal ---
        if (top) {
            IvtcDisplayFrame de;
            de.topSource    = pendingTopFilm;
            de.bottomSource = localFilm;
            de.isSynth      = (de.topSource != de.bottomSource);
            m_displayFrameList.push_back(de);
            // Open NEW half-open slot with .top = this film (bottom TBD).
            pendingTopFilm = localFilm;
        } else if (bottom) {
            IvtcDisplayFrame de;
            de.topSource    = localFilm;
            de.bottomSource = pendingBotFilm;
            de.isSynth      = (de.topSource != de.bottomSource);
            m_displayFrameList.push_back(de);
            pendingBotFilm = localFilm;
        } else {
            IvtcDisplayFrame de;
            de.topSource    = localFilm;
            de.bottomSource = localFilm;
            de.isSynth      = false;
            m_displayFrameList.push_back(de);
        }

        // --- STEP 2: RFF stash + possible close ---
        if (f.rff) {
            if (f.tff) { pendingTopFilm = localFilm; top    = true; }
            else       { pendingBotFilm = localFilm; bottom = true; }

            if (top && bottom) {
                IvtcDisplayFrame de;
                de.topSource    = pendingTopFilm;
                de.bottomSource = pendingBotFilm;
                de.isSynth      = (de.topSource != de.bottomSource);
                m_displayFrameList.push_back(de);
                top = bottom = false;
                pendingTopFilm = pendingBotFilm = -1;
            }
        }
    }
    // Any trailing half-open slot is dropped (DGDecode vfapidec.cpp:534-538
    // "fringe slot past ntsc is not emitted").

    m_displayFrameCount = (int)m_displayFrameList.size();

    // DIAG #1: what the builder produced, independent of later cycle-
    // dispatch decisions. Uses m_pLog->write directly to bypass
    // AddMessage's variadic layer. "synths" counts schedule slots whose
    // top/bottom coded indices differ; for a pure 3:2 stream this
    // should be ~20% of m_displayFrameCount (one synth per 4 coded ->
    // 5 display).
    if (m_pLog) {
        const int cirVal = lastFilm - firstFilm;
        int synthSlots = 0;
        for (const auto &df : m_displayFrameList) if (df.isSynth) synthSlots++;
        const int    dcVal    = m_displayFrameCount;
        const double schedVal = (cirVal > 0)
                                ? (double)m_displayFrameCount / (double)cirVal
                                : 0.0;
        m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
            _T("ivtc DIAG: schedule built: codedInRange=%d display=%d synths=%d ratio=%.4fx\n"),
            cirVal, dcVal, synthSlots, schedVal);
    }
}

// Flush the 10-slot staging buffer by walking the pre-computed schedule
// (m_displayFrameList). Emits each schedule entry whose coded sources are
// present in the buffer (or in the single carry slot from the previous
// flush). Between pushes the cycle-processing drain loop runs inline so
// the 5-slot IVTC cache (IVTC_CACHE_SIZE) never gets overwritten before
// its data is consumed -- a single flush can push 12+ display frames.
// On EOS (isFinal=true) we emit all remaining schedule entries that are
// in range and drop any trailing half-open slot.
RGY_ERR NVEncFilterIvtc::flushExpandBuffer(cudaStream_t stream,
                                          int cycleLen, bool isFinal) {
    if (m_expandBufCount == 0) return RGY_ERR_NONE;

    // Helper: resolve a POST-TRIM coded index to its buffer entry or the
    // single-frame carry. Returns nullptr if the frame is not currently
    // resident (i.e. lives in a future buffer -- caller stops emission).
    auto resolveSrc = [&](int codedIdx, CUFrameBuf **outFrame,
                           const ExpandBufMeta **outMeta) -> bool {
        if (codedIdx >= m_expandBufBase
            && codedIdx <  m_expandBufBase + m_expandBufCount) {
            const int slot = codedIdx - m_expandBufBase;
            *outFrame = m_expandBuf[slot].get();
            *outMeta  = &m_expandBufMeta[slot];
            return true;
        }
        if (m_expandCarryValid && codedIdx == m_expandCarryCodedIdx) {
            *outFrame = m_expandCarryFrame.get();
            *outMeta  = &m_expandCarryMeta;
            return true;
        }
        return false;
    };

    const int lastBufCodedIdx = m_expandBufBase + m_expandBufCount - 1;
    int scheduleSynth = 0;
    int scheduleCoded = 0;

    while (m_nextDisplayIdx < m_displayFrameCount) {
        const auto &df = m_displayFrameList[m_nextDisplayIdx];
        // Don't emit this entry yet if either source is beyond the
        // buffer -- unless we're at EOS, in which case out-of-range
        // sources are silently dropped. Crossing a future-buffer
        // boundary should never happen with a 10-frame buffer in
        // practice (max back-ref distance is 1 in DGDecode 3:2), but
        // we guard against it to keep the walker total-correct.
        const int maxIdxUsed = std::max(df.topSource, df.bottomSource);
        if (maxIdxUsed > lastBufCodedIdx) {
            if (!isFinal) break;
            // isFinal: the entry references a non-existent frame;
            // silently drop the rest (matches DGDecode's trailing-slot
            // drop behaviour at EOS).
            break;
        }

        CUFrameBuf          *topFrame = nullptr;
        const ExpandBufMeta *topMeta  = nullptr;
        CUFrameBuf          *botFrame = nullptr;
        const ExpandBufMeta *botMeta  = nullptr;
        if (!resolveSrc(df.topSource,    &topFrame, &topMeta)
         || !resolveSrc(df.bottomSource, &botFrame, &botMeta)) {
            // At least one source was evicted (should be impossible
            // with a 1-back carry + 10-slot buffer unless someone
            // changed EXPAND_BUF_SIZE without revisiting the max
            // backward-reference invariant). Stop and let future
            // flushes re-try; if this is EOS, drop the schedule tail.
            if (!isFinal) break;
            AddMessage(RGY_LOG_WARN,
                _T("ivtc: schedule entry %d references un-resident coded idx (top=%d bot=%d buf=[%d..%d] carry=%d); dropping.\n"),
                m_nextDisplayIdx, df.topSource, df.bottomSource,
                m_expandBufBase, lastBufCodedIdx,
                m_expandCarryValid ? m_expandCarryCodedIdx : -1);
            m_nextDisplayIdx++;
            continue;
        }

        if (!df.isSynth) {
            auto err = pushFrameToRing(&topFrame->frame, stream,
                                        m_inputCount, /*isSynth=*/true);
            if (err != RGY_ERR_NONE) return err;
            scheduleCoded++;
        } else {
            auto cpErr = copyFrameAsync(&m_expandSynth->frame, &topFrame->frame, stream);
            if (cpErr != RGY_ERR_NONE) return cpErr;
            // Kernel tff=1 overwrites odd rows -- always the right
            // choice for "overlay bottomSource's bottom field onto the
            // topSource copy" since libavcodec decoded frames always
            // place the top field in even rows regardless of stream TFF.
            auto ovErr = overlayField(m_expandSynth.get(), botFrame, /*tff=*/1, stream);
            if (ovErr != RGY_ERR_NONE) return ovErr;

            RGYFrameInfo si = m_expandSynth->frame;
            si.picstruct    = RGY_PICSTRUCT_FRAME;
            si.flags        = (RGY_FRAME_FLAGS)(topMeta->flags & ~(
                                    RGY_FRAME_FLAG_RFF
                                  | RGY_FRAME_FLAG_RFF_COPY
                                  | RGY_FRAME_FLAG_RFF_TFF
                                  | RGY_FRAME_FLAG_RFF_BFF));
            si.duration     = topMeta->duration;
            si.inputFrameId = topMeta->inputFrameId;
            const int64_t tTop = topMeta->timestamp;
            const int64_t tBot = botMeta->timestamp;
            si.timestamp    = (tTop != AV_NOPTS_VALUE && tBot != AV_NOPTS_VALUE)
                              ? (tTop + tBot) / 2
                              : tTop;
            // PATCH 1: isSynth=true suppresses DEC:/IN: logging (real decoder
            // input already logged at BUFFER time); isSynthFrame=true marks
            // m_cacheIsSynth[slot]=true so processInputToCycle skips the
            // scorer/matcher/cadence/blend pipeline for this cross-time synth.
            auto pushErr = pushFrameToRing(&si, stream,
                                            m_inputCount, /*isSynth=*/true,
                                            /*isSynthFrame=*/true);
            if (pushErr != RGY_ERR_NONE) return pushErr;
            scheduleSynth++;
        }
        m_nextDisplayIdx++;

        // Drain IVTC processing between pushes so the 5-slot cache
        // never gets overwritten mid-flush.
        if (cycleLen == 0) {
            if (m_processedCount + 3 <= m_inputCount) {
                const int center    = m_processedCount;
                const int idx_cur   =  center      % IVTC_CACHE_SIZE;
                const int idx_next  = (center + 1) % IVTC_CACHE_SIZE;
                const int idx_next2 = (center + 2) % IVTC_CACHE_SIZE;
                const int idx_prev  = (center >= 1) ? (center - 1) % IVTC_CACHE_SIZE : idx_cur;
                const int idx_prev2 = (center >= 2) ? (center - 2) % IVTC_CACHE_SIZE : idx_prev;
                auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2,
                                                center, stream);
                if (err != RGY_ERR_NONE) return err;
                m_processedCount++;
            }
        } else {
            while (m_processedCount + 3 <= m_inputCount) {
                const int center    = m_processedCount;
                const int idx_cur   =  center      % IVTC_CACHE_SIZE;
                const int idx_next  = (center + 1) % IVTC_CACHE_SIZE;
                const int idx_next2 = (center + 2) % IVTC_CACHE_SIZE;
                const int idx_prev  = (center >= 1) ? (center - 1) % IVTC_CACHE_SIZE : idx_cur;
                const int idx_prev2 = (center >= 2) ? (center - 2) % IVTC_CACHE_SIZE : idx_prev;
                auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2,
                                                center, stream);
                if (err != RGY_ERR_NONE) return err;
                m_processedCount++;
                if (m_cycleFilled >= cycleLen) {
                    auto ferr = flushCycle(false, topMeta->timestamp, stream);
                    if (ferr != RGY_ERR_NONE) return ferr;
                }
            }
        }
    }

    // Save the last coded frame of this buffer as the carry for the
    // next flush -- future schedule entries may reference it as a
    // synth source (top-pending or bottom-pending chain straddling
    // the boundary). One carry frame suffices because DGDecode's
    // algorithm only references the previous film in any synth slot.
    if (!isFinal && m_expandBufCount > 0) {
        const int lastSlot = m_expandBufCount - 1;
        auto cpErr = copyFrameAsync(&m_expandCarryFrame->frame,
                                      &m_expandBuf[lastSlot]->frame, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc: carry-frame copy failed: %s\n"), get_err_mes(cpErr));
            return cpErr;
        }
        m_expandCarryMeta     = m_expandBufMeta[lastSlot];
        m_expandCarryValid    = true;
        m_expandCarryCodedIdx = m_expandBufBase + lastSlot;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("ivtc: expand flush: %d coded -> %d display (%d synth)%s\n"),
        m_expandBufCount, scheduleCoded + scheduleSynth, scheduleSynth,
        isFinal ? _T(" [final]") : _T(""));

    m_expandSynthCount     += scheduleSynth;
    m_expandCodedEmitCount += scheduleCoded;
    // Advance base past the consumed buffer. Runtime buffer is now empty.
    m_expandBufBase        += m_expandBufCount;
    m_expandBufCount        = 0;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::processInputToCycle(int idx_prev2, int idx_prev, int idx_cur, int idx_next, int idx_next2, int centerDisplayIdx, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int slot = (cycleLen > 0) ? m_cycleFilled : 0;

    // =========================================================================
    //  PATCH 1: isSynth bypass for cross-time synth frames
    // -------------------------------------------------------------------------
    //  Synth frames pushed by flushExpandBuffer were ALREADY correctly
    //  assembled by kernel_ivtc_field_overlay (topSource's copy + bottomSource's
    //  odd rows). Running them through scoreCandidates is structurally wrong:
    //    - Their fields come from different coded times -> match=C looks
    //      heavily combed (cComb ~130 vs normal ~55).
    //    - This triggers applyBlend unnecessarily on every synth frame.
    //    - Poisoning the cadence tracker with these spurious "combed" observations
    //      prevents phase lock and cascades into coded frames blending too.
    //  The bypass below:
    //    - Direct-copies m_cacheFrames[idx_cur] into m_frameBuf[slot]
    //      (skips synthesizeToCycle / synthesizeToCycleBwdif kernels).
    //    - Fills the m_cycle* bookkeeping arrays with match=C, applyBlend=0,
    //      zero scores, decTag=16 ("synth-passthru").
    //    - Still runs computePairDiff so cycle decimation can rank this slot's
    //      SAD vs its neighbour -- synths are temporal duplicates and SHOULD be
    //      preferentially dropped by the 5->4 decimation.
    //    - Does NOT touch the cadence tracker / lastSceneChange state.
    //  Synths still participate in cycle decimation downstream.
    // =========================================================================
    if (m_cacheIsSynth[idx_cur]) {
        const RGYFrameInfo &curInfoS = m_cacheFrames[idx_cur]->frame;

        // Direct copy cache -> cycle staging slot.
        auto cpErr = copyFrameAsync(&m_frameBuf[slot]->frame, &curInfoS, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc synth passthrough copy failed: %s\n"),
                       get_err_mes(cpErr));
            return cpErr;
        }
        m_frameBuf[slot]->frame.picstruct = RGY_PICSTRUCT_FRAME;
        m_frameBuf[slot]->frame.flags     = curInfoS.flags;

        if (cycleLen > 0) {
            m_cycleInPts[slot]         = curInfoS.timestamp;
            m_cycleInDur[slot]         = curInfoS.duration;
            m_cycleInputIds[slot]      = curInfoS.inputFrameId;
            m_cycleMatchScore[slot]    = 0;
            m_cycleCombScore[slot]     = 0;
            m_cycleCombMax[slot]       = 0;   // synth passthru: no scoring ran
            m_cycleCombBlocks[slot]    = 0;   // synth passthru: no scoring ran
            // SUB-PHASE 2: synth passthru — no scoring ran for either parity.
            m_cycleCombScorePrim[slot]   = std::array<uint64_t, 3>{};
            m_cycleCombMaxPrim[slot]     = std::array<uint64_t, 3>{};
            m_cycleCombBlocksPrim[slot]  = std::array<uint64_t, 3>{};
            m_cycleCombScoreAlt[slot]    = std::array<uint64_t, 3>{};
            m_cycleCombMaxAlt[slot]      = std::array<uint64_t, 3>{};
            m_cycleCombBlocksAlt[slot]   = std::array<uint64_t, 3>{};
            m_cycleMatchAltParity[slot]  = 0;   // synth always primary
            m_cycleConfidence[slot]      = 0;   // synth: HIGH confidence (passthru, no scoring ran)
            m_cycleBlendTrigger[slot]  = 0;   // synth never blends
            m_cycleMatchType[slot]     = (int)IvtcMatch::C;   // clean passthru
            m_cycleApplyBlend[slot]    = 0;                   // no blend on synth
            m_cycleDecTag[slot]        = 16;                  // SYNTH_PASSTHRU (new)
            m_cycleCadenceTag[slot]    = 0;                   // "none"
            m_cycleSceneSAD[slot]      = 0;
            m_cycleIsSynth[slot]       = 1;                   // FIX B: prefer dropping this slot

            // Pair-wise SAD for the decimation argmin. Synth frames are temporal
            // duplicates of their top-source neighbour; their diff-to-prev will
            // typically be LOWER than coded neighbours' diffs, so the 5->4
            // decimator preferentially drops them -- exactly what we want.
            uint64_t diff = 0;
            RGY_ERR derr = RGY_ERR_NONE;
            if (slot > 0) {
                derr = computePairDiff(&m_frameBuf[slot]->frame,
                                        &m_frameBuf[slot - 1]->frame, diff, stream);
            } else if (m_hasSaveSlot) {
                derr = computePairDiff(&m_frameBuf[slot]->frame,
                                        &m_frameBuf[cycleLen]->frame, diff, stream);
            } else {
                diff = std::numeric_limits<uint64_t>::max();
            }
            if (derr != RGY_ERR_NONE) return derr;
            m_cycleDiffPrev[slot] = diff;
            m_cycleFilled++;
        } else {
            // cycle=0 path: emit inline (no flushCycle will run for this slot).
            m_frameBuf[slot]->frame.timestamp    = curInfoS.timestamp;
            m_frameBuf[slot]->frame.duration     = curInfoS.duration;
            m_frameBuf[slot]->frame.inputFrameId = curInfoS.inputFrameId;
            if (m_fpLog) {
                fprintf(m_fpLog.get(),
                    "OUT:\t#%d\tin_id=%d\tmatch=c\tmatchParity=pri\tconf=0\tpost=none \tstatus=emit \t"
                    "dec=SYNTH_PASSTHRU\tcadence=none\t"
                    "mQ=0\tcComb=0\tcCombMax=0\tcCombBlocks=0\t"
                    "cComb_c=0\tcComb_p=0\tcComb_n=0\tcComb_cA=0\tcComb_pA=0\tcComb_nA=0\t"
                    "cMax_c=0\tcMax_p=0\tcMax_n=0\tcMax_cA=0\tcMax_pA=0\tcMax_nA=0\t"
                    "cBlk_c=0\tcBlk_p=0\tcBlk_n=0\tcBlk_cA=0\tcBlk_pA=0\tcBlk_nA=0\t"
                    "btrig=0\tpostComb=0\tdiff=0\tscene_sad=0\n",
                    m_outputFrameCount, curInfoS.inputFrameId);
                fflush(m_fpLog.get());
            }
            m_outputFrameCount++;
        }
        // Intentionally do NOT touch m_lastSceneChange / m_lastSceneSAD -- those
        // track the coded-frame scene-change history, and synth frames would
        // corrupt it (a synth at a scene boundary has bizarre SAD that would
        // mislead the next coded frame's adaptive threshold).
        return RGY_ERR_NONE;
    }

    // ============================================================================
    //  PIPELINE STAGE 1 — CANDIDATE SCORING
    // ============================================================================
    // 1. 候補スコア計算 (match-quality と combing-count を独立に算出 = block-max メトリック)
    //    Per-frame tff derivation: prefer the center frame's picstruct field-order
    //    bit when present; otherwise fall back to the init-time m_tffFixed.
    //    This unifies the tff logic across score / synthesize / BWDIF kernels
    //    (BWDIF already does the same derivation at ~line 1588). For streams
    //    with mixed-picstruct frames (some FRAME_TFF / some FRAME_BFF / some
    //    FRAME-only), the per-frame value is authoritative; the init-time
    //    m_tffFixed serves only as the fallback when the decoder doesn't
    //    emit a field-order bit.
    const auto centerPicstruct = m_cacheFrames[idx_cur]->frame.picstruct;
    int tffForScoring = m_tffFixed ? 1 : 0;
    if      (centerPicstruct == RGY_PICSTRUCT_TFF || centerPicstruct == RGY_PICSTRUCT_FRAME_TFF) tffForScoring = 1;
    else if (centerPicstruct == RGY_PICSTRUCT_BFF || centerPicstruct == RGY_PICSTRUCT_FRAME_BFF) tffForScoring = 0;

    uint64_t matchScore [3] = { 0, 0, 0 };
    uint64_t combScore  [3] = { 0, 0, 0 };
    uint64_t combMax    [3] = { 0, 0, 0 };   // PRIORITY 1: TFM-style block-MAX per candidate
    uint64_t combBlocks [3] = { 0, 0, 0 };   // SUB-PHASE 1: count of blocks with cX >= BLOCK_COMB_THRESH
    auto err = scoreCandidates(
        &m_cacheFrames[idx_prev]->frame,
        &m_cacheFrames[idx_cur ]->frame,
        &m_cacheFrames[idx_next]->frame,
        matchScore, combScore, combMax, combBlocks, tffForScoring, stream);

    // SUB-PHASE 2 (2026-04-24) — DUAL-PARITY DIAGNOSTIC SCORING.
    // Re-dispatch the scoring kernel with the inverted tff to obtain
    // the alternate field-pair combinations (C/P/N at !tff). The kernel
    // produces semantically distinct outputs for tff vs !tff (verified
    // by reading rgy_filter_ivtc.cl: thread firing pivots on
    // first_parity = tff?0:1 at line 154, and pix_match's source select
    // is_first_field_row at line 92 inverts which physical rows the
    // C/P/N candidates pull from). Each dispatch fully overwrites
    // m_scoreBuf so reusing the buffer is safe.
    //
    // The alt scoring is gated on the primary scoring succeeding —
    // otherwise the original err return below will fire first and we
    // never use the alt buffers. Alt failure is non-fatal: we log a
    // warning and leave the alt buffers zero-initialized so the TSV
    // log columns simply show 0 for that frame.
    //
    // Selection logic still consumes ONLY the primary triplets. The
    // alt triplets are diagnostic-only in this sub-phase — used by the
    // 18 new TSV columns for offline analysis of which SG1 frames
    // would benefit from comb-first dual-parity selection in
    // SUB-PHASE 3.
    uint64_t matchScoreAlt [3] = { 0, 0, 0 };
    uint64_t combScoreAlt  [3] = { 0, 0, 0 };
    uint64_t combMaxAlt    [3] = { 0, 0, 0 };
    uint64_t combBlocksAlt [3] = { 0, 0, 0 };
    if (err == RGY_ERR_NONE) {
        const auto errAlt = scoreCandidates(
            &m_cacheFrames[idx_prev]->frame,
            &m_cacheFrames[idx_cur ]->frame,
            &m_cacheFrames[idx_next]->frame,
            matchScoreAlt, combScoreAlt, combMaxAlt, combBlocksAlt, tffForScoring ? 0 : 1, stream);
        if (errAlt != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN,
                _T("ivtc SUB-PHASE 2: alt-parity scoring failed (%s); zeroing alt metrics for this frame.\n"),
                get_err_mes(errAlt));
            // Alt buffers stay zero per the initializer above.
        }
    }
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const RGYFrameInfo &curInfo = m_cacheFrames[idx_cur]->frame;
    // block-max ベースの「クリーン閾値」: 1 ブロック内で何ピクセルが combing 判定されたら汚いとするか。
    // 旧実装は全フレーム総和ピクセル数に対して cleanFrac を掛けていたが、block-max 化に合わせて
    // 1 ブロックあたり IVTC_BLOCK_X*IVTC_BLOCK_Y * cleanFrac ピクセルを許容上限とする。
    const uint64_t blockPixels = (uint64_t)IVTC_BLOCK_X * (uint64_t)IVTC_BLOCK_Y;
    const uint64_t cleanBlockThresh = std::max<uint64_t>(1, (uint64_t)((double)blockPixels * prm->ivtc.cleanFrac));

    // 2. マッチ選択: match-quality の argmin で C / P / N を選ぶ
    //    (based on published 3:2 pulldown detection algorithm)。
    //    guide=1 のとき、C の combing-count が cleanBlockThresh 以下なら既にクリーンなので
    //    無駄な P/N への切り替えを避ける (pattern hint の簡易版)。
    //    さらに「C に 1 ピクセルの combing も検出されない (= 既に完全 progressive)」
    //    ケースは、ソフトテレシネ素材のデコーダ出力に当てはまる状況で、ここで P/N に
    //    切り替えると別時刻のフィールドを混ぜて逆に combing を生成してしまう。
    //    guide に依存しない hard-guard として C をロックする。

    // -- 2a. Scene change detection ------------------------------------------
    // Compute a lightweight cur-vs-prev SAD every frame (centerDisplayIdx >= 1).
    // Used for two purposes:
    //   (a) Force match=C + reset cadence state when a true scene cut is
    //       detected, so we don't lock a stale phase across the cut.
    //   (b) Trigger the BWDIF kernel's spatial-only fallback for the scene
    //       change frame AND the frame after it (via m_lastSceneChange) to
    //       avoid prev/next temporal ghosting across the cut.
    //
    // Threshold: 15% of the theoretical per-pixel max SAD. Empirically:
    //   - static scene             : SAD ~ 1-3 %
    //   - camera pan / motion      : SAD ~ 3-10 %
    //   - real scene cut           : SAD ~ 15-50 %
    // The previous 40% value was tight enough that many real cuts didn't
    // trip it (frame 5624 ghosting regression observed 2026-04-22); 15% is
    // the lowest conservative value that cleanly separates cuts from high-
    // motion pans.
    //
    // sceneSAD is HOISTED out of the firing block so it's visible to the
    // per-frame logger below; otherwise the log always shows 0 in cycle=0
    // mode and the user can't tell whether detection is firing at all.
    //
    // Guard conditions:
    //   centerDisplayIdx >= 1 : skip first frame (no prev to compare against)
    //   m_diffBuf             : defensive null-check (allocated unconditionally
    //                           in init but keep the guard for degenerate cases)
    //   idx_prev != idx_cur   : prev aliases to cur at stream start; a self-SAD
    //                           is always 0 and would misrepresent a still frame
    //                           as a scene cut-if-0. Skip rather than compute 0.
    const int bitDepth = RGY_CSP_BIT_DEPTH[curInfo.csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const uint64_t frameMaxSAD   = (uint64_t)curInfo.width * (uint64_t)curInfo.height * (uint64_t)maxVal;

    // Adaptive threshold: scale to 1.5x the previous frame's SAD (captures
    // "a few-times-harder cut than normal motion"), clamped to [0.12, 0.30]
    // of frameMaxSAD. Startup fallback: when m_lastSceneSAD is near-zero
    // (first call OR effectively-static previous frame), use the historical
    // 0.15 fixed value so the detector still behaves on still-opener content.
    double sceneFrac = 0.15;
    if (m_lastSceneSAD > 1) {
        const double prevFrac = (double)m_lastSceneSAD / (double)frameMaxSAD;
        sceneFrac = prevFrac * 1.5;
        if (sceneFrac < 0.12) sceneFrac = 0.12;
        if (sceneFrac > 0.30) sceneFrac = 0.30;
    }
    const uint64_t sceneThresh = (uint64_t)((double)frameMaxSAD * sceneFrac);
    bool sceneChange = false;
    uint64_t sceneSAD = 0;   // visible to logger tail; 0 on the first frame is correct
    if (centerDisplayIdx >= 1 && m_diffBuf && idx_prev != idx_cur) {
        auto diffErr = computePairDiff(
            &m_cacheFrames[idx_cur ]->frame,
            &m_cacheFrames[idx_prev]->frame,
            sceneSAD, stream);
        if (diffErr == RGY_ERR_NONE && sceneSAD > sceneThresh) {
            sceneChange = true;
            AddMessage(RGY_LOG_DEBUG, _T("ivtc: scene change detected (SAD %llu > %llu, frac %.3f), forcing C.\n"),
                (unsigned long long)sceneSAD, (unsigned long long)sceneThresh, sceneFrac);
            resetCadenceState();
        }
    }

    // -- 2b. PAL 2:2 (guide=2): alternate C/N by input-frame parity ----------
    // 2:2 cadence has no P borrow; just pick C for even-indexed inputs, N for odd.
    IvtcMatch match = IvtcMatch::C;
    uint64_t chosenMatchScore = matchScore[(int)IvtcMatch::C];
    // SUB-PHASE 3 (2026-04-24): tracks whether the comb-first selector
    // chose an alt-parity candidate. true = !tff parity used for both
    // matching metrics AND frame assembly. Scene-change, guide=2, and
    // cadence-override paths force this back to false (primary parity).
    bool matchAltParity = false;
    // Cadence tag for per-frame log (set by the cadence-override block
    // below when it runs). Defaults to "disabled" for paths that skip
    // the override (scene change, guide=2). See encoding below.
    int cadenceTag = 2;   // 2 = "disabled"

    if (sceneChange) {
        match = IvtcMatch::C;
        chosenMatchScore = matchScore[(int)IvtcMatch::C];
    } else if (prm->ivtc.guide == 2) {
        const bool useN = ((centerDisplayIdx & 1) != 0);
        match = useN ? IvtcMatch::N : IvtcMatch::C;
        chosenMatchScore = matchScore[(int)match];
    } else {
        // ========================================================================
        //  PIPELINE STAGE 2 — PRIMARY SELECTION (COMB-FIRST 6-CANDIDATE)
        // ========================================================================
        // SUB-PHASE 3 (2026-04-24) — COMB-FIRST 6-CANDIDATE SELECTION.
        //
        // Replaces the previous argmin-SAD + hysteresis block. Builds a
        // unified array over (C, P, N) at primary tff and (cA, pA, nA) at
        // alternate tff, ranks lexicographically by:
        //
        //     combBlocks (count of combed blocks)        — primary
        //     combMax    (worst block's combed pixels)   — secondary
        //     combScore  (trimmed-avg combed pixels)     — tertiary
        //     matchScore (interp-error / SAD)            — final tiebreak
        //
        // Then applies the decisive-alt guard: alt-parity wins only when
        // its combScore is dramatically lower than the best primary's
        // (zero-vs-noisy or half-or-better). This prevents marginal
        // parity flips that could introduce field-inversion artifacts on
        // borderline frames.
        //
        // Hysteresis from the prior path is removed — the comb-first
        // comparator's matchScore tiebreak provides natural temporal
        // stability (when nothing changes, scores stay equal and the
        // last-arbitrary-tied winner stays winner). The cadence override
        // block below still applies on the primary-parity result.
        //
        // The `back` parameter no longer gates P-test: with comb-first
        // ranking the comparator considers all 3 types regardless of
        // C's combing. `back` is preserved as a no-op for back-compat.
        struct MatchCandidate {
            int      type;        // 0=c, 1=p, 2=n
            bool     altParity;   // false=primary tff, true=!tff
            uint64_t combBlocks;
            uint64_t combMax;
            uint64_t combScore;
            uint64_t matchScore;
        };
        MatchCandidate candidates[6];
        for (int i = 0; i < 3; i++) {
            candidates[i]     = { i, false, combBlocks    [i], combMax    [i], combScore    [i], matchScore    [i] };
            candidates[i + 3] = { i, true,  combBlocksAlt [i], combMaxAlt [i], combScoreAlt [i], matchScoreAlt [i] };
        }

        auto combFirstLess = [](const MatchCandidate &a, const MatchCandidate &b) -> bool {
            if (a.combBlocks != b.combBlocks) return a.combBlocks < b.combBlocks;
            if (a.combMax    != b.combMax)    return a.combMax    < b.combMax;
            if (a.combScore  != b.combScore)  return a.combScore  < b.combScore;
            return a.matchScore < b.matchScore;
        };

        // Best across all 6.
        int bestIdx = 0;
        for (int i = 1; i < 6; i++) {
            if (combFirstLess(candidates[i], candidates[bestIdx])) bestIdx = i;
        }
        // Best primary-only (used by the decisive-alt guard).
        int bestPriIdx = 0;
        for (int i = 1; i < 3; i++) {
            if (combFirstLess(candidates[i], candidates[bestPriIdx])) bestPriIdx = i;
        }

        // ========================================================================
        //  PIPELINE STAGE 3 — PLATEAU HANDLING (Sub-Phase 6)
        //  PIPELINE STAGE 4 — TEMPORAL CONSISTENCY (Sub-Phase 7, nested below)
        // ========================================================================
        // -------- SUB-PHASE 6 (2026-04-25): Separation-aware re-ranking --------
        // The default lex order (combBlocks → combMax → combScore) prioritises
        // structural cleanliness — the right call when one candidate is
        // structurally distinct from the others. On PLATEAU frames where all
        // primary candidates are within 10 combScore of each other, no
        // candidate is structurally distinct, and the matchScore tiebreak
        // can pick a winner whose average comb level is noticeably worse
        // than an alternative with similar block counts.
        //
        // In that narrow case, re-rank the 3 primaries with a swapped
        // priority — combScore first, then combBlocks, then combMax. This
        // selects the candidate with the lowest average comb level on
        // plateau frames where the structural metrics are essentially noise.
        //
        // Scoped tightly to avoid disturbing strong matches:
        //   - bestScore < 65 : not in blend territory (existing gates handle
        //                      that); below this floor we trust matcher
        //                      defaults.
        //   - plateau (worst - best <= 10) : no decisive structural winner.
        //
        // Affects ONLY bestPriIdx. bestIdx (6-cand) is left untouched —
        // alt parity is globally clamped so bestIdx doesn't drive output,
        // but leaving it on the original lex order keeps the v5 alt-guard's
        // diagnostic-search semantics intact.
        {
            const uint64_t s0 = candidates[0].combScore;
            const uint64_t s1 = candidates[1].combScore;
            const uint64_t s2 = candidates[2].combScore;
            const uint64_t bestScore  = std::min({s0, s1, s2});
            const uint64_t worstScore = std::max({s0, s1, s2});
            const bool isPlateau = ((worstScore - bestScore) <= PLATEAU_DELTA);
            if (isPlateau && bestScore < COMB_BLEND_THRESHOLD) {
                auto plateauLess = [](const MatchCandidate &a, const MatchCandidate &b) -> bool {
                    if (a.combScore  != b.combScore)  return a.combScore  < b.combScore;
                    if (a.combBlocks != b.combBlocks) return a.combBlocks < b.combBlocks;
                    if (a.combMax    != b.combMax)    return a.combMax    < b.combMax;
                    return a.matchScore < b.matchScore;
                };
                int plateauBest = 0;
                for (int i = 1; i < 3; i++) {
                    if (plateauLess(candidates[i], candidates[plateauBest])) plateauBest = i;
                }
                bestPriIdx = plateauBest;

                // -------- SUB-PHASE 7 (2026-04-25): Temporal consistency --------
                // Plateau-only tiebreaker: prefer the previous frame's match
                // type when it sits close to the new plateau winner. Reduces
                // visible "flicker" between equally-scored candidates across
                // adjacent frames in pulldown-stable content where the matcher
                // would otherwise oscillate between (effectively-equal) C / P
                // / N picks.
                //
                // m_lastMatch already exists (declared in rgy_filter_ivtc.h),
                // is initialized to -1, reset to -1 on stream change, and is
                // updated to the post-cadence chosen match at line ~2771. So
                // it carries the previous emitted-frame's match decision —
                // exactly the right signal for temporal stability.
                //
                // Conditions all required:
                //   - isPlateau (already in scope)            : ambiguous case
                //   - bestScore < 65 (already in scope)       : not blending
                //   - m_lastMatch in [0, 2]                   : prev exists
                //   - prev's combScore <= bestScore + 6       : prev is close
                //     (do NOT override if prev is clearly worse)
                if (m_lastMatch >= 0 && m_lastMatch <= 2) {
                    const uint64_t prevScore = candidates[m_lastMatch].combScore;
                    if (prevScore <= bestScore + TEMPORAL_TOLERANCE) {
                        bestPriIdx = m_lastMatch;
                    }
                }
            }
        }

        // ========================================================================
        //  PIPELINE STAGE 5 — ALT GUARD (currently disabled via global clamp)
        // ========================================================================
        // The v5 alt-guard logic is preserved for auditability but its output
        // is neutralised below by the matchAltParity = false clamp. See the
        // INVARIANT note at that clamp site.
        //
        // Decisive-alt guard (2026-04-25, fifth revision).
        //
        // History:
        //   v1 (combScore-only)            — 174 SG1 alt-flips; trivially
        //                                    cleared by clean-frame noise.
        //   v2 (broadened primaryIsCombed) — 171; combMax>=100 padded the
        //                                    OR with a clause that already
        //                                    fired on clean content.
        //   v3 (combScore>=65 OR combMax>=100) — still ~171; standalone
        //                                    combMax>=100 fired on clean
        //                                    texture (combMax is a raw
        //                                    per-WG pixel count, not a
        //                                    combed-block flag).
        //   v4 (combScore>=65 OR (combMax>=100 AND combScore>=50)) —
        //                                    alt count dropped to ~20 and
        //                                    blend counts came in correct
        //                                    (SG1 ~47, DH2 = 1), but a
        //                                    visual regression remained:
        //                                    frame 4058 showed mixed-
        //                                    field discoloration even
        //                                    though its scores looked
        //                                    clean.
        //
        // Root cause exposed by v4: P-alt and N-alt are NOT parity-flipped
        // versions of P-pri / N-pri — they are SEMANTICALLY DIFFERENT
        // film-frame pairings that combine fields from different time
        // instants:
        //
        //   primary tff (TFF source):
        //     P-pri = cur.top  + prev.bot     (cur frame, borrow bot from prev)
        //     N-pri = next.top + cur.bot      (cur frame, borrow top from next)
        //
        //   alt tff (TFF source, !tff = BFF interpretation):
        //     P-alt = prev.top + cur.bot      (prev's top + cur's bot)
        //     N-alt = cur.top  + next.bot     (cur's top + next's bot)
        //
        // P-alt / N-alt can score very low (their assembled frame
        // interpolates cleanly enough to fool the combing metric) while
        // physically combining fields from two different time instants.
        // Any inter-frame chroma drift, panning, or color rotation between
        // those time instants then surfaces as discoloration in the
        // output frame — chroma is half-resolution and exposes temporal
        // mismatches that luma's high-frequency texture can mask.
        //
        // C-alt, by contrast, is parity-invariant: pix_match returns pCur
        // for every row regardless of tff, so C-alt and C-pri produce
        // identical assembled output. Allowing C-alt is harmless (it just
        // logs matchParity=alt for diagnostic value) but allowing P-alt /
        // N-alt is unsafe.
        //
        // v5 restricts alt selection to C-only (altIsSafe = matchType==0)
        // and reverts primaryIsCombed back to the bare blend-gate floor
        // (combScore >= 65) — the strongLocalCombing OR clause was added
        // in v4 to recover frames that just-missed the blend floor, but
        // with C as the only allowed alt that recovery has no effect
        // (C-alt produces the same output as C-pri).
        //
        // Single-source-of-truth: only finalIdx is read after this point;
        // bestIdx and bestPriIdx are not referenced past line below.
        const auto &best        = candidates[bestIdx];
        const auto &primaryBest = candidates[bestPriIdx];

        bool useAlt = false;
        if (best.altParity) {
            const bool primaryWouldBlend =
                (primaryBest.combScore >= COMB_BLEND_THRESHOLD);

            const bool altIsClearlyBetter =
                (best.combScore == 0) ||
                (primaryBest.combScore > 0 &&
                 best.combScore * ALT_DECISIVE_RATIO < primaryBest.combScore);

            // Restrict alt to SAME-FRAME (C) matches only. P-alt / N-alt
            // mix fields from different time instants and cause visible
            // chroma artifacts — see the rationale block above.
            const bool altIsSafe = (best.type == 0);   // C only

            useAlt = primaryWouldBlend && altIsClearlyBetter && altIsSafe;
        }

        int finalIdx = useAlt ? bestIdx : bestPriIdx;

        // ========================================================================
        //  PIPELINE STAGE 6 — RESCUE PASS (Sub-Phase 4 / 4b)
        // ========================================================================
        // SUB-PHASE 4 (2026-04-25) — Rescue Pass (comb-only fallback).
        // --------------------------------------------------------------
        // TFM-style additive rescue: re-rank the existing 6 candidates
        // by combing only (no matchScore tiebreak) when the chosen
        // candidate has structural combing but isn't bad enough to
        // trigger the blend gate. Targets near-threshold failures
        // (e.g. SG1 ~4285) where the comb-first comparator's matchScore
        // tiebreak picked a slightly-combed candidate over a cleaner
        // alternative that happened to have a marginally higher SAD.
        //
        // Hard rules (mirror the parent task's CRITICAL constraints):
        //   - Does not modify selection, thresholds, blend gates,
        //     assembly, or the candidate array. Only mutates finalIdx.
        //   - Trigger uses combBlocks AND combScore (combMax never
        //     standalone — would re-introduce the v3 false-positive
        //     mode where clean texture cleared the threshold).
        //   - Parity safety enforces the v5 invariant: P-alt / N-alt
        //     are NEVER selectable; only primary parity or C-alt.
        //   - Single source of truth: finalIdx is mutated in place,
        //     then the existing `chosen` extraction below sees the
        //     rescued value. All downstream reads (match,
        //     matchAltParity, chosenCombScore/Max/Blocks, assemblyTff,
        //     TSV log, m_cycleMatchAltParity) are driven by finalIdx
        //     transitively through `chosen`/match/matchAltParity.
        //   - Toggleable via enableRescuePass for one-flag rollback.
        const bool enableRescuePass = true;
        {
            // CRITICAL GATE (2026-04-25 correction): rescue must ONLY run
            // when the BEST PRIMARY candidate is in the combed zone.
            // Without this gate, the original combBlocks > 0 trigger fires
            // on essentially every frame (noisy block-flag floor at
            // THRESH=8), turning rescue into an unconditional re-rank that
            // flipped 2814/8000 SG1 frames to C-alt and pushed blend
            // counts from 47 → 64.
            //
            // Threshold revision (2026-04-25 v3): lowered from 65 to 40.
            // The original 65 mirrored the blend-gate floor (combThreshProg)
            // so rescue only fired on frames that would already blend —
            // but the visible regressions (e.g. frame 4285) sit in the
            // mid-comb 40-65 band, where primary's combScore is high
            // enough to leave structural combing in the output but not
            // high enough to trigger the blend gate. Lowering to 40
            // brings these recoverable frames into rescue range without
            // re-opening the noise-floor floodgates: the decisive-better
            // gate (Case 1: combBlocks==0 victory, Case 2: 3x combScore
            // improvement) and parity-safe gate still filter out
            // marginal swaps.
            //
            // primaryBest is already in scope from the v5 alt-guard
            // above; reused here.
            const bool primaryWouldBlendForRescue =
                (primaryBest.combScore >= RESCUE_TRIGGER);

            const auto &origForRescue = candidates[finalIdx];
            const bool needsRescue =
                (origForRescue.combBlocks > 0)                       // structural combing exists
                && (origForRescue.combScore < COMB_BLEND_THRESHOLD); // below blend threshold

            if (enableRescuePass && primaryWouldBlendForRescue && needsRescue) {
                // Comb-only re-rank across all 6 candidates. Lexicographic:
                //   combBlocks → combMax → combScore. matchScore is
                //   intentionally NOT a tiebreak here — that's what makes
                //   this a rescue rather than a re-run of the comb-first
                //   selection.
                int rescueIdx = finalIdx;
                for (int i = 0; i < 6; i++) {
                    const auto &a = candidates[i];
                    const auto &b = candidates[rescueIdx];
                    const bool better =
                        (a.combBlocks <  b.combBlocks)
                        || (a.combBlocks == b.combBlocks && a.combMax <  b.combMax)
                        || (a.combBlocks == b.combBlocks && a.combMax == b.combMax
                            && a.combScore < b.combScore);
                    if (better) rescueIdx = i;
                }

                const auto &orig = candidates[finalIdx];
                const auto &resc = candidates[rescueIdx];

                // Decisive-better guard (2026-04-25, third revision).
                //
                // History:
                //   v1: combBlocks<  OR 2x combScore — accepted marginal
                //       noise-driven swaps. Blend counts 47 → 64;
                //       2814 spurious matchParity=alt flips on SG1.
                //   v2: combBlocks==0 victory OR 3x combScore — too
                //       strict. The 3x threshold only catches extreme
                //       improvements, which don't exist between
                //       similarly-scored mid-comb candidates. Frame
                //       4285 (~combScore 40-50) had no candidate
                //       achieving 3x reduction, so rescue never fired.
                //       Blends stuck at 63.
                //
                // v3 (Sub-Phase 4b "soft rescue"): swap the
                // multiplicative 3x for an ADDITIVE delta of 8. In the
                // 40-65 mid-comb band an absolute +8 improvement
                // represents real cleanup (e.g. orig=48 → resc<=39 is
                // a meaningful drop) while still excluding the noise-
                // floor variance that v1 was accepting (typical
                // candidate-to-candidate jitter on clean content is
                // 1-3 combScore units).
                //
                // The combBlocks==0 fast-path is kept — it's the
                // unambiguous "rescue candidate is structurally
                // clean" win and doesn't need a magnitude rule.
                //
                // Standalone combMax and standalone `combBlocks <`
                // comparisons are still NOT used — combMax is a
                // per-WG raw pixel count (the noise-prone signal that
                // drove the v3 alt-guard regression), and a bare
                // combBlocks decrease is satisfied by ordinary noise
                // variance between candidates.
                //
                // Underflow safety: the additive form
                // `resc.combScore + 8 < orig.combScore` evaluates as
                // uint64 addition on the LHS — never underflows. An
                // alternative subtractive form
                // `orig.combScore - 8 > resc.combScore` would underflow
                // if orig.combScore < 8 (it's uint64_t), so the
                // additive form is the safe choice.
                const bool rescueIsBetter =
                    // Case 1: eliminate all combing (unchanged)
                    (orig.combBlocks > 0 && resc.combBlocks == 0)
                    ||
                    // Case 2: meaningful additive improvement (RESCUE_IMPROVEMENT margin)
                    (orig.combScore > 0 &&
                     resc.combScore + RESCUE_IMPROVEMENT < orig.combScore);

                // Parity safety (2026-04-25 tightened revision).
                //
                // Prior gate allowed C-alt because C is parity-invariant
                // (pix_match returns pCur regardless of tff, so the
                // assembled output is identical to C-pri). However, when
                // rescue had broad activation (combScore >= 40), the
                // C-alt allowance let rescue pick C-alt on ~146 SG1
                // frames per run, re-flipping matchParity=alt on
                // structurally-identical output. Visually safe but
                // effectively re-enabled alt parity outside the v5
                // guard's control — and the matchParity TSV column
                // stopped meaning what it was supposed to mean.
                //
                // New rule: rescue selects PRIMARY PARITY ONLY. Alt
                // parity (including C-alt) can only be introduced by
                // the v5 alt-guard at the top of the matcher, never by
                // the rescue path.
                const bool rescueAltSafe = (!resc.altParity);

                // Sub-Phase 4c (plateau detection) was reverted on
                // 2026-04-25: it produced an alt-parity explosion
                // (0 → 481), a small blend regression (63 → 64), and no
                // visible improvement on plateau frames such as 4285.
                // The plateau swap accepted candidates with strictly
                // lower combBlocks even when combScore was higher,
                // which both pushed some frames over the blend gate
                // and over-flipped to C-alt on plateau-symmetric
                // candidate populations. Restored to the rescueIsBetter
                // gate alone.
                if (rescueIsBetter && rescueAltSafe) {
                    finalIdx = rescueIdx;
                }
            }
        }

        // A primary N immediately followed by primary P assembles the exact
        // same field pair:
        //   prev N = cur.top + prev.bot
        //   cur  P = cur.top + prev.bot
        // Leaving both for cycle decimation is fragile because another
        // duplicate-like pair in the same 5-frame cycle can consume the single
        // drop slot. Prefer the best C/N primary candidate instead; cadence
        // override below can still steer to the locked C/N pattern.
        if (prm->ivtc.guide == 1
            && m_lastMatch == (int)IvtcMatch::N
            && candidates[finalIdx].type == (int)IvtcMatch::P
            && !candidates[finalIdx].altParity) {
            const int bestNonP = combFirstLess(candidates[0], candidates[2]) ? 0 : 2;
            finalIdx = bestNonP;
        }

        const MatchCandidate &chosen = candidates[finalIdx];
        match            = (IvtcMatch)chosen.type;
        chosenMatchScore = chosen.matchScore;
        matchAltParity   = chosen.altParity;

        // ====================================================================
        //  INVARIANT: matchAltParity MUST remain false after this point.
        //  Alt parity is globally disabled. Downstream code (assembly,
        //  blend gates, logging, cycle-slot bookkeeping) keys off this
        //  flag and depends on it being false for every emitted frame.
        // ====================================================================
        // GLOBAL PARITY INVARIANT (2026-04-25, final clamp).
        //
        // After several iterations of the v5 alt-guard / rescue / plateau
        // experiments, the residual alt-parity selections (e.g. 12
        // C-alt frames on DH2 from the v5 guard satisfying primary>=65
        // && altIsClearlyBetter on heavily-textured progressive frames)
        // were determined to provide no measurable benefit while
        // continuing to make the matchParity TSV column noisy.
        //
        // Final policy: matchAltParity is permanently FALSE. The v5
        // alt-guard's selection LOGIC is preserved (it can still pick
        // bestIdx over bestPriIdx for diagnostic-search reasons) but
        // the resulting parity flag is clamped here. Because the v5
        // guard restricts alt to C-only (altIsSafe = best.type == 0)
        // and C is parity-invariant (pix_match returns pCur regardless
        // of tff), forcing the flag to false produces the SAME
        // assembled output that the alt selection would have — only
        // the metric channel and TSV log change.
        //
        // Done as a single chokepoint clamp rather than removing the
        // alt branches because:
        //   - keeps the v5 guard / rescue logic auditable as-is
        //   - one-line revert if alt parity is ever re-enabled
        //   - downstream paths (chosenCombScore via combScore[match]
        //     instead of combScoreAlt, assemblyTff = bwdifTff,
        //     m_cycleMatchAltParity[slot] = 0, TSV matchParity=pri)
        //     all key off matchAltParity and automatically follow.
        matchAltParity = false;

        // -- 2c. Cadence pattern-predicted match override (Phase B3) -------
        // Port of TFM's PredictHardYUY2 + gthresh validation
        // (Telecide.cpp:276-304). Flow:
        //
        //   1. Feed argmin winner into 5-frame ring, get prediction (or -1).
        //   2. If prediction available AND gthresh > 0:
        //        mismatch% = |pred_score - argmin_score| / pred_score * 100
        //        if mismatch% < gthresh  → override match with prediction
        //        else                    → reject prediction
        //   3. gthresh=0 disables override entirely (pure argmin behaviour).
        //   4. cadenceLock=off also disables — pattern history still tracked
        //      for diagnostic logging but never applied.
        //
        // TFM uses gthresh=10 as default; same here. The gate is symmetric
        // — requires pred_score within 10% of argmin in EITHER direction.
        //
        // Cadence tag encoding (for log):
        //   0 = "none"       no pattern locked (common for non-pulldown content)
        //   1 = "warmup"     ring not yet full (first 4 frames post-reset)
        //   2 = "disabled"   guide=0 or cadenceLock=off
        //   3/4/5 = "locked/C|P|N/nochange"  pattern locked, prediction == argmin
        //   6/7/8 = "locked/C|P|N/override"  pattern locked, override fired
        //   9/10/11 = "locked/C|P|N/reject"  pattern locked, gthresh rejected
        // cadenceTag is declared at match-selection scope (above); set here.
        cadenceTag = 0; // "none"
        if (prm->ivtc.guide == 0 || !prm->ivtc.cadenceLock) {
            // Still feed the tracker so history stays current in case
            // cadenceLock gets turned on mid-stream.
            updateCadence((int)match);
            cadenceTag = 2; // "disabled"
        } else {
            const int predicted = updateCadence((int)match);
            if (m_cadenceFill < IVTC_CADENCE_LEN) {
                cadenceTag = 1; // "warmup"
            } else if (predicted < 0 || predicted >= 3) {
                cadenceTag = 0; // "none" — ring full but no pattern locked
            } else {
                // Pattern is locked; evaluate override gate.
                const int predBase = predicted;  // 0=C, 1=P, 2=N
                if ((int)match == predicted) {
                    // No-op: argmin already agrees with the pattern.
                    cadenceTag = 3 + predBase;  // locked/X/nochange
                } else {
                    const int gthresh = std::max(0, std::min(100, prm->ivtc.gthresh));
                    if (gthresh == 0) {
                        // Override disabled by gthresh=0.
                        cadenceTag = 9 + predBase;  // locked/X/reject
                    } else {
                        const uint64_t predScore   = matchScore[predicted];
                        const uint64_t chosenScore = chosenMatchScore;
                        // Mismatch percent = |pred - chosen| / pred * 100.
                        // Integer-safe: compute |a-b| first, then * 100, then / pred.
                        // Guard against predScore == 0 (degenerate case).
                        uint64_t diffAbs = (predScore > chosenScore)
                                         ? (predScore - chosenScore)
                                         : (chosenScore - predScore);
                        bool accept = false;
                        if (predScore == 0) {
                            // Pred has zero score — ideal match; always accept.
                            accept = true;
                        } else {
                            const uint64_t diffScaled = diffAbs * 100ULL;
                            const uint64_t threshScaled = predScore * (uint64_t)gthresh;
                            accept = (diffScaled < threshScaled);
                        }
                        if (accept) {
                            AddMessage(RGY_LOG_DEBUG,
                                _T("ivtc: cadence override: argmin=%d -> predicted=%d (pred=%llu argmin=%llu)\n"),
                                (int)match, predicted,
                                (unsigned long long)predScore, (unsigned long long)chosenScore);
                            match = (IvtcMatch)predicted;
                            chosenMatchScore = matchScore[predicted];
                            // SUB-PHASE 3: cadence patterns are calibrated on
                            // primary-parity match history. When cadence
                            // override fires we always land on a primary-parity
                            // candidate, regardless of what comb-first chose.
                            matchAltParity = false;
                            cadenceTag = 6 + predBase;  // locked/X/override
                        } else {
                            cadenceTag = 9 + predBase;  // locked/X/reject
                        }
                    }
                }
            }
        }
    }

    m_lastMatch      = (int)match;
    m_lastMatchScore = chosenMatchScore;

    // 3. post=2 ブレンド発火判定は「選択マッチ後も残っている combing 量」で判定する。
    //    match-quality ではなく combing-count を使うのがポイント (post=2 はピクセル単位の
    //    凸凹を平滑するための後処理で、residual comb に反応すべきなので)。
    //    発火時は synthesize カーネル側で second-field 行を unconditional に bob 補間する
    //    (per-pixel 閾値判定は、デコーダでフィールドマージされて弱く均された combing に対して
    //     反応しないことが多く、ベイクドイン combing が残るケースを救えないため)。
    // SUB-PHASE 3 (2026-04-24): pull the chosen match's combing metrics
    // from the matching-parity arrays. Primary parity uses combScore /
    // combMax / combBlocks (the originals); alt parity uses the *Alt
    // versions populated by the second scoreCandidates call. The blend
    // gate downstream consumes chosenCombScore / chosenCombMax /
    // chosenCombBlocks so it ALSO sees the alt metrics when alt was
    // chosen — keeping the gate aligned with the actually-selected
    // field combination.
    const uint64_t chosenCombScore  = matchAltParity ? combScoreAlt [(int)match] : combScore [(int)match];
    const uint64_t chosenCombMax    = matchAltParity ? combMaxAlt   [(int)match] : combMax   [(int)match];
    const uint64_t chosenCombBlocks = matchAltParity ? combBlocksAlt[(int)match] : combBlocks[(int)match];

    // ============================================================
    //  Picstruct-driven applyBlend gate (2026-04-22 rewrite).
    // ============================================================
    //
    // The per-frame routing decision reads the decoder's picstruct flags
    // (curInfo.picstruct). The decoder's own output is authoritative
    // because it is what the pixel pipeline actually processes.
    //
    // picstruct classification:
    //   RGY_PICSTRUCT_FRAME               -> progressive (no action unless mislabeled)
    //   RGY_PICSTRUCT_TFF / _BFF          -> genuinely interlaced (RAW-input edge case only;
    //                                        see comment on treatAsInterlaced below)
    //   RGY_PICSTRUCT_FRAME_TFF / _BFF    -> ambiguous (progressive flag + field order);
    //                                        treat as unknown — fire conservatively
    //   anything else (UNKNOWN, 0, etc.)  -> treat as unknown
    const uint64_t MAX_CCOMB_PER_BLOCK = (uint64_t)IVTC_BLOCK_X * (uint64_t)IVTC_BLOCK_Y;
    // picstruct comes straight from the decoder. Earlier experiment with
    // RFF_TFF/RFF_BFF promotion was reverted: those flags are set on
    // progressive 3:2-pulldown content, and promoting them would over-blend
    // frames the existing strongMatch path already handles correctly.
    const auto picstruct = curInfo.picstruct;

    const bool picstructValid =
        (picstruct == RGY_PICSTRUCT_FRAME ||
         picstruct == RGY_PICSTRUCT_TFF ||
         picstruct == RGY_PICSTRUCT_BFF ||
         picstruct == RGY_PICSTRUCT_FRAME_TFF ||
         picstruct == RGY_PICSTRUCT_FRAME_BFF);

    // treatAsInterlaced — RAW-INPUT EDGE CASE ONLY.
    //
    // Per the decoder pipeline investigation in
    // analysis/decoder_pipeline_analysis.txt (2026-04-22), pure
    // RGY_PICSTRUCT_TFF / RGY_PICSTRUCT_BFF is NEVER emitted by any
    // real video decoder:
    //   - avsw (libavcodec): picstruct_avframe_to_rgy at
    //     rgy_avutil.cpp:156-161 maps only to FRAME / FRAME_TFF /
    //     FRAME_BFF — never pure TFF/BFF.
    //   - avhw (Intel MFX): picstruct_enc_to_rgy at
    //     qsv_util.cpp:145-149 and the per-frame MFX mapping at
    //     qsv_pipeline_ctrl.h:1032-1048 — same three-value output set.
    //   - Intel media-driver always writes one merged surface per
    //     MPEG-2 picture (confirmed in media-driver source); no
    //     field-separated output mode exists.
    //
    // The ONLY write of pure RGY_PICSTRUCT_TFF in the entire codebase
    // is rgy_input_raw.cpp:87, which fires when the user passes raw
    // interlaced YUV via --raw or --y4m with a forced TFF hint. This
    // is the path treatAsInterlaced is kept for — to allow the user
    // to bypass the pixel-based classification on known-interlaced
    // raw feeds. Do NOT expect this branch to fire under normal
    // avsw/avhw operation; for those, interlaced content arrives as
    // FRAME_TFF/FRAME_BFF and is routed through the treatAsUnknown
    // path (with strongMatch / unknownCombed bypasses) instead.
    //
    // Kept rather than removed so raw-YUV interlaced users still get
    // the direct classification without relying on combing detection.
    const bool treatAsInterlaced =
        (picstruct == RGY_PICSTRUCT_TFF ||
         picstruct == RGY_PICSTRUCT_BFF);
    const bool treatAsProgressive =
        (picstruct == RGY_PICSTRUCT_FRAME);
    const bool treatAsUnknown =
        !picstructValid ||
        !(treatAsInterlaced || treatAsProgressive);

    // bestOther + strongMatch computed FIRST so the mislabel dual-condition
    // check can reuse the "chosen is a clear winner" signal. Match scores
    // are argmin ("lower is better"), so chosenMatchScore ≤ every
    // alternative. bestOther is computed via an explicit loop over the
    // three candidates (skipping selected) to avoid manual index juggling;
    // UINT64_MAX sentinel fallback defends against the theoretical
    // "no non-selected" case (shouldn't happen — there are always 3).
    uint64_t bestOther = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < 3; i++) {
        if (i != (int)match) {
            bestOther = std::min(bestOther, matchScore[i]);
        }
    }
    if (bestOther == std::numeric_limits<uint64_t>::max()) {
        bestOther = matchScore[(int)match];   // defensive fallback
    }
    const uint64_t marginPost  = std::max<uint64_t>(1ULL, bestOther / 10ULL);
    // Clear-winner test: chosen is lower than bestOther by at least margin.
    // NOTE: the user spec wrote this as (chosen > bestOther) — under
    // argmin that's never true. We use (bestOther > chosen + margin)
    // which matches the INTENT ("chosen is clearly the best match").
    const bool strongMatch =
        (bestOther > chosenMatchScore) &&
        ((bestOther - chosenMatchScore) > marginPost);

    // ----------------------------------------------------------------
    //  Dual-condition mislabel detection (2026-04-22 follow-up fix):
    //  the saturation-only gate (cComb ≥ MAX_CCOMB) misses genuinely
    //  combed frames at cComb = 185..248. Empirical data:
    //    - DH2 texture false positives: mQ maxes at ~1129
    //    - SG-1 genuinely combed frames: mQ starts at ~1719
    //  Gap centre ≈ 1500 → mQ_threshold = 1500 (8-bit base).
    //
    //  The dual condition fires when ALL of:
    //    1. cComb ≥ 3 × cleanBlockThresh (residual combing well above noise)
    //    2. mQ > mQ_threshold (interpolation residual above texture ceiling)
    //    3. chosenMatchScore >= bestOther (minimal ordering guard — allows
    //       flat high scores where all C/P/N are equally bad, which is the
    //       SG-1 combed case; blocks pathological inversions only when the
    //       selected candidate has been overridden to worse than
    //       alternatives by hysteresis or cadence lock).
    //
    //  strongMatch is NOT required here. Genuinely combed mislabeled
    //  frames have all three candidates at similarly high scores —
    //  there is no clear argmin winner when the content is truly combed
    //  regardless of field combination. Requiring strongMatch would
    //  block the exact cases we want to catch.
    //
    //  mQ_threshold scales linearly with pixel range via bit-depth shift
    //  (clamped to [0, 8]): 1500 at 8-bit, 6000 at 10-bit, 384000 at 16-bit.
    // ----------------------------------------------------------------
    const int      mQshift           = std::max(0, std::min(8, bitDepth - 8));
    const uint64_t mQ_threshold      = (uint64_t)1500 << mQshift;
    // Separate, looser mQ gate for the progressive path only — sw-decoder
    // weaved output has inherently suppressed mQ on real combing (DH2
    // #7769-#7777 cluster at mQ=1008..1491, cComb=86..113, all visually
    // combed but below the 1500 threshold). Interlaced/unknown paths
    // keep the stricter 1500 floor because those classes have more
    // headroom in the metric space.
    const uint64_t mQ_threshold_prog = (uint64_t)1000 << mQshift;
    const uint64_t combThresh3       = (uint64_t)cleanBlockThresh * 3ULL;
    // Fixed-absolute cComb floor used by BOTH unknownCombed and
    // progressiveCombed. cComb is a per-WG zigzag pixel count (max 256)
    // and is NOT bit-depth-scaled, so the absolute threshold works at
    // all bit depths. combThresh3 (~153 at cleanFrac=0.20) is too high
    // empirically — SG-1 avhw frames #4460/4461 (cComb=117-119) and
    // DH2 cases fail the combThresh3 gate despite real combing.
    const uint64_t combThreshProg    = 65ULL;

    const bool mislabeledBySaturation =
        treatAsProgressive && (chosenCombScore >= MAX_CCOMB_PER_BLOCK);
    const bool mislabeledByDual =
        treatAsProgressive
        && !mislabeledBySaturation
        && (chosenCombScore >= combThresh3)
        && (chosenMatchScore > mQ_threshold)
        && (chosenMatchScore >= bestOther);
    const bool mislabeled = mislabeledBySaturation || mislabeledByDual;

    // Strong-combing escape hatch for treatAsUnknown. Observed avhw frames
    // like mQ=12891/cComb=194 and SG-1 #4460 (mQ=1928/cComb=119) have flat
    // candidate scores where strongMatch fails even though the combing is
    // unambiguously real.
    //
    // 2026-04-24: DUAL-GATE (MAX + AVG). The trimmed-average cComb alone
    // hides localised combing — e.g. SG-1 #2741 has cComb=62 (below the
    // old 65 threshold) but cCombMax=128 because one hot block is diluted
    // by surrounding clean blocks. Relying on cCombMax alone over-triggers
    // on noise spikes (~138 clean frames in the test corpus have cCombMax
    // > 200 from isolated-block anomalies).
    //
    // Dual gate requires:
    //   cCombMax  > 100  — at least one block has strong localised comb
    //                      (catches SG-1 #2741's hot block; above most
    //                       noise-spike outliers)
    //   cComb     > 45   — non-trivial frame-level activity (below the
    //                      old 65 to give the MAX-driven path headroom,
    //                      but above the clean-frame noise floor)
    //
    // Only applies to treatAsUnknown — other classes (progressive, inter-
    // laced) keep their original gates unchanged. mQ gate retained as a
    // secondary guard against texture false positives.
    const uint64_t combMaxThreshUnknown = 100ULL;
    const uint64_t combAvgThreshUnknown = 45ULL;
    const bool unknownCombed =
        treatAsUnknown
        && (chosenCombMax   > combMaxThreshUnknown)
        && (chosenCombScore > combAvgThreshUnknown)
        && (chosenMatchScore > mQ_threshold);

    // Progressive-combed bypass. Sw decoder weaves interlaced content to
    // picstruct=FRAME with no RFF/TFF flags at all — there is no picstruct
    // signal to route on. DH2 is the canonical case (100% picstruct=FRAME).
    //
    // Uses the LOOSER mQ gate (mQ_threshold_prog = 1000, declared above)
    // because sw-weaved real combing suppresses mQ: DH2 #7769-#7777 sit
    // at mQ=1008..1491, which would all miss a 1500 threshold. Calibration
    // frames (8-bit):
    //   DH2 #686   mQ=1816  cComb=70   — needs catch
    //   DH2 #7769  mQ=1491  cComb=113  — needs catch (was missed at mQ>1500)
    //   DH2 #7777  mQ=1008  cComb=86   — needs catch (was missed at mQ>1500)
    //   DH2 clean  mQ=450   cComb=52   — must stay clean (below mQ=1000)
    //
    // combThreshProg=65 declared above is shared with unknownCombed.
    // mQ_threshold_prog IS bit-depth-scaled via mQshift; combThreshProg
    // is an absolute per-WG pixel count so stays fixed at 65.
    //
    // We deliberately drop the "chosen >= bestOther" ordering guard here
    // because on legitimate progressive content a clear C-winner is
    // normal. The mQ + cComb combined threshold is strict enough on
    // its own to block texture false positives.
    //
    // 2026-04-24 Option D — dual-gate defense via block-MAX:
    //   Add cCombMax > 85 as an AND condition mirroring unknownCombed's
    //   dual-gate pattern. Rationale:
    //   - Guards against frame-level noise pushing cComb over 65 with
    //     NO single block strongly combed (no cCombMax spike). Such a
    //     case would be a texture false positive; requiring cCombMax>85
    //     ensures at least one 16x16 block has a concentrated combing
    //     signature, matching TFM's highest_sumc semantics.
    //   - Threshold 85 (not 100) calibrated to preserve DH2 frame #712
    //     which has cCombMax=88 — a legitimate catch on sw-weaved
    //     content. Raising to 100 would lose that catch.
    //   - Zero impact on the 36 SG1 progressiveCombed fires measured in
    //     ivtc_sg1_dualgate_log.txt; all have cCombMax in [117, 128].
    //   See analysis/dominant_gate_analysis.txt for the measurement.
    const uint64_t combMaxThreshProg = 85ULL;
    const bool progressiveCombed =
        treatAsProgressive
        && !mislabeled
        && (chosenMatchScore > mQ_threshold_prog)
        && (chosenCombScore  >= combThreshProg)
        && (chosenCombMax    > combMaxThreshProg);

    // applyBlend gate — Phase B4 (post-assembly veto), 2026-04-23 revision.
    //
    // Two-stage decision:
    //
    //   (1) applyBlendGate: picstruct-class classifier. Unchanged from the
    //       pre-B4 (fix5) build. The mQ+cComb combinations here are what
    //       distinguish genuinely-combed frames from normal film texture —
    //       dropping them caused a 4.4x over-blend regression (DH2: 157 →
    //       692 blends) because progressive texture at cComb=66..70 with
    //       mQ=395..700 passed a cComb-only gate.
    //
    //   (2) vthresh veto (TFM vmetric analogue, Telecide.cpp:376-397): on
    //       top of (1), require chosenCombScore >= vthresh. Default 50 —
    //       below combThreshProg (65) so it only filters frames that
    //       reached (1) via the strongMatch branch (which has no cComb
    //       threshold of its own beyond cleanBlockThresh). The cComb-gated
    //       branches (mislabeled*/unknownCombed/progressiveCombed) already
    //       demand cComb >= 65 or higher, so vthresh=50 never overrides
    //       them. vthresh=0 disables the veto entirely (pure fix5 behaviour).
    //
    // Key property: vthresh can only REMOVE blends relative to fix5, never
    // ADD them. With default 50 the effect is minimal — a safety net for
    // the strongMatch branch. Users who want TFM-style post-assembly
    // discipline can raise vthresh toward 65; this will start trimming
    // blends that the classifier gate flagged but the assembled frame
    // doesn't actually need.
    //
    // Option C still applies to the veto check: scoreCandidates in
    // rgy_filter_ivtc.cl uses pix_match(...) which assembles each
    // candidate's field pair per-row before measuring combing, so
    // chosenCombScore IS the post-assembly cComb. No separate re-scoring.
    // ============================================================================
    //  PIPELINE STAGE 7 — CONFIDENCE EVALUATION (Sub-Phase 5)
    // ============================================================================
    //  SUB-PHASE 5 (2026-04-25) — Confidence-Based Decision Layer
    // ----------------------------------------------------------------------------
    //
    // Computed AFTER cadence override and immediately BEFORE the
    // applyBlend gate so the classifier sees the post-cadence chosen
    // state. Adds a single OR-clause to applyBlendGate when confidence
    // is LOW (with a tightened secondary check) or VERY_LOW; otherwise
    // existing gate logic stands. See:
    //   analysis/Confidence-Based_Matcher_TFM-style.txt
    //
    // Reuses existing thresholds:
    //   - mQ_threshold_prog (line ~2873) for VERY_LOW + LOW gating
    //   - cleanBlockThresh as absolute lower bound (no blend below it)
    //   - LOW lowered floor = 55 (combThreshProg=65 minus 10)
    //
    // One-flag rollback: enableConfidenceLayer = false collapses the
    // OR-clause to a no-op while still emitting the TSV column for
    // diagnostic A/B comparison.
    int  confidenceLevel       = 0;     // 0=HIGH 1=MEDIUM 2=LOW 3=VERY_LOW
    bool confidenceForcesBlend = false;
    {
        const bool enableConfidenceLayer = true;

        // Re-derive primaryBest / secondBest by the same lex order the
        // matcher used (combBlocks → combMax → combScore → matchScore),
        // restricted to the 3 PRIMARY candidates only. The matcher's
        // local MatchCandidate struct is out of scope here; operate on
        // the underlying combScore[]/combMax[]/combBlocks[]/matchScore[]
        // arrays which are in outer-function scope.
        auto priLess = [&](int a, int b) -> bool {
            if (combBlocks[a] != combBlocks[b]) return combBlocks[a] < combBlocks[b];
            if (combMax   [a] != combMax   [b]) return combMax   [a] < combMax   [b];
            if (combScore [a] != combScore [b]) return combScore [a] < combScore [b];
            return matchScore[a] < matchScore[b];
        };
        int bestPri = 0;
        for (int i = 1; i < 3; i++) if (priLess(i, bestPri)) bestPri = i;
        int secondPri = (bestPri == 0) ? 1 : 0;
        for (int i = 0; i < 3; i++) if (i != bestPri && priLess(i, secondPri)) secondPri = i;

        const uint64_t primaryBestScore = combScore[bestPri];
        const uint64_t secondScore      = combScore[secondPri];
        const uint64_t secondMax        = combMax  [secondPri];

        // Saturating gaps: chosen may have higher score than secondBest
        // when cadence overrode the matcher to a lower-priority candidate
        // or when rescue swapped to a fewer-combBlocks-but-higher-score
        // alternative.
        const uint64_t scoreGap = (secondScore > chosenCombScore)
            ? (secondScore - chosenCombScore) : 0;
        const uint64_t maxGap   = (secondMax   > chosenCombMax)
            ? (secondMax   - chosenCombMax)   : 0;

        // ---- Separation-by-combScore winner check (Sub-Phase 5 v2 fix) ----
        // Independent of the matcher's lex order. If one candidate is a
        // CLEAR winner by combScore alone — i.e. the second-lowest score
        // is at least 8 above the lowest — classify as HIGH regardless of
        // the absolute combScore band, plateau check, or scoreGap-by-lex.
        //
        // Why: the matcher's lex order is combBlocks → combMax → combScore,
        // so secondPri (lex-second) can have a LOWER combScore than the
        // lex-best. The original scoreGap = secondScore - chosenCombScore
        // then saturated to 0, falsely flagging "no decisive winner". On
        // DH2 this misclassified ~45 mid-comb film frames as LOW/VERY_LOW
        // and forced spurious blends. A direct sort over the 3 primary
        // combScores recovers the actual score-spread independent of the
        // lex pivot.
        const uint64_t cs0 = combScore[0];
        const uint64_t cs1 = combScore[1];
        const uint64_t cs2 = combScore[2];
        const uint64_t scoreMin = std::min({cs0, cs1, cs2});
        // Second-min: walk the three values, skip exactly one occurrence
        // of the min, take min of the remainder.
        uint64_t scoreMin2 = (uint64_t)-1;
        bool minSkipped = false;
        const uint64_t triple[3] = { cs0, cs1, cs2 };
        for (int i = 0; i < 3; i++) {
            if (triple[i] == scoreMin && !minSkipped) { minSkipped = true; continue; }
            if (triple[i] < scoreMin2) scoreMin2 = triple[i];
        }
        const bool clearScoreWinner =
            (scoreMin2 != (uint64_t)-1) && (scoreMin2 >= scoreMin + CONFIDENCE_SEPARATION);

        // Frame class from PRIMARY BEST (matcher's intrinsic difficulty).
        // Not from chosen — chosen may have shifted under cadence/rescue
        // and we want to label the frame, not the post-mutation outcome.
        enum { CLEAN_CLS = 0, WEAK_CLS = 1, STRONG_CLS = 2 };
        int frameCls;
        if      (primaryBestScore <  COMB_CLEAN_BOUND)     frameCls = CLEAN_CLS;
        else if (primaryBestScore <  COMB_BLEND_THRESHOLD) frameCls = WEAK_CLS;
        else                                                frameCls = STRONG_CLS;

        // Confidence assignment.
        enum { CONF_HIGH = 0, CONF_MEDIUM = 1, CONF_LOW = 2, CONF_VERY_LOW = 3 };
        int conf;
        if (frameCls == STRONG_CLS) {
            conf = CONF_HIGH;   // strong-class blend gates already cover
        } else if (frameCls == CLEAN_CLS) {
            conf = (chosenCombBlocks == 0) ? CONF_HIGH : CONF_MEDIUM;
        } else {  // WEAK_CLS — the decision band
            const bool plateau = (scoreGap <= CONFIDENCE_PLATEAU_SCORE && maxGap <= PLATEAU_MAX_DELTA);
            if (plateau && chosenCombBlocks > 0 && chosenMatchScore > mQ_threshold_prog) {
                conf = CONF_VERY_LOW;   // no-win: candidates similar AND mQ confirms combing
            } else if (plateau || scoreGap < CONFIDENCE_GAP_THRESHOLD) {
                conf = CONF_LOW;        // no decisive winner
            } else {
                conf = CONF_MEDIUM;     // clear winner in mid-comb; rescue may help
            }
        }

        // Separation override: a clear combScore winner (gap >= CONFIDENCE_SEPARATION)
        // trumps the WEAK-class plateau/scoreGap heuristics that would otherwise
        // flag the frame as LOW/VERY_LOW.
        if (clearScoreWinner) {
            conf = CONF_HIGH;
        }
        confidenceLevel = conf;

        // confidenceForcesBlend: single OR-clause for the existing gate.
        // Hard guard (Sub-Phase 5 v2 fix): require chosenCombScore >= 65.
        // The original LOW path had a 55 floor and VERY_LOW had no floor,
        // which let confidence force-blend mid-comb frames where one
        // candidate was actually clean — DH2 went 1 → 45 blends. Lifting
        // the floor to 65 ensures confidence only adds blends in the band
        // where the existing progressiveCombed/unknownCombed gates may
        // miss due to combMax not crossing 85/100. Mid-comb (40-65) film
        // frames are now NEVER blended via the confidence path; they
        // either pass through (if no other gate fires) or blend via the
        // explicit-predicate gates.
        if (enableConfidenceLayer
            && (prm->ivtc.post >= 2)
            && (chosenCombScore > cleanBlockThresh)
            && (chosenCombScore >= COMB_BLEND_THRESHOLD)) {   // hard guard against false blends
            if (conf == CONF_VERY_LOW) {
                confidenceForcesBlend = true;
            } else if (conf == CONF_LOW
                    && chosenMatchScore >  mQ_threshold_prog) {
                confidenceForcesBlend = true;
            }
        }
    }

    // ============================================================================
    //  PIPELINE STAGE 8 — FINAL SELECTION (applyBlend gate)
    // ============================================================================
    // applyBlendGate combines the explicit-predicate gates (mislabeled /
    // unknownCombed / progressiveCombed / strongMatch on interlaced+unknown)
    // with the confidence layer's OR-clause (Stage 7). vthresh veto runs
    // after to optionally trim borderline blends. The final `applyBlend`
    // boolean determines which kernel the assembly stage dispatches to.
    const bool applyBlendGate =
        (prm->ivtc.post >= 2)
        && (chosenCombScore > cleanBlockThresh)
        && (mislabeled
            || unknownCombed
            || progressiveCombed
            || ((treatAsInterlaced || treatAsUnknown) && strongMatch)
            || confidenceForcesBlend);

    const uint64_t vthreshU = (uint64_t)std::max(0, prm->ivtc.vthresh);
    const bool vthreshVeto  = (vthreshU > 0) && (chosenCombScore < vthreshU);

    const bool applyBlend = applyBlendGate && !vthreshVeto;

    // INSTRUMENTATION: classify which specific predicate caused blend.
    // Priority order (most specific first): mislabeledBySat > mislabeledByDual
    // > unknownCombed > progressiveCombed > confidenceForcesBlend > strongMatch.
    // Codes:
    //    0 = no blend
    //    1 = mislabeledBySaturation
    //    2 = mislabeledByDual
    //    3 = unknownCombed
    //    4 = progressiveCombed
    //    5 = strongMatch (on treatAsInterlaced or treatAsUnknown)
    //    6 = vthresh_vetoed (gate fired but veto canceled blend)
    //    7 = confidenceForcesBlend (Sub-Phase 5 — fired only when no
    //        explicit-predicate gate did, so this counts the NEW blend
    //        population introduced by the confidence layer)
    uint8_t blendTrigger = 0;
    if (applyBlendGate && vthreshVeto) {
        blendTrigger = 6;  // gated but vetoed — final applyBlend=false
    } else if (applyBlend) {
        if      (mislabeledBySaturation)                                       blendTrigger = 1;
        else if (mislabeledByDual)                                             blendTrigger = 2;
        else if (unknownCombed)                                                blendTrigger = 3;
        else if (progressiveCombed)                                            blendTrigger = 4;
        else if ((treatAsInterlaced || treatAsUnknown) && strongMatch)         blendTrigger = 5;
        else if (confidenceForcesBlend)                                        blendTrigger = 7;
        else                                                                   blendTrigger = 5;  // defensive fallback
    }
    if (blendTrigger < m_blendTriggerCounts.size()) {
        m_blendTriggerCounts[blendTrigger]++;
    }

    // bwdifSpatialOnly covers the scene-change frame AND the frame after:
    // the latter's prev is still from the outgoing scene so temporal
    // context would ghost two scenes. m_lastSceneChange carries the flag
    // forward by one frame; it's updated unconditionally at the tail of
    // this function so the state ticks regardless of BWDIF firing.
    const bool bwdifSpatialOnly = sceneChange || m_lastSceneChange;

    // Build the per-frame dec= log tag from the ROUTING DECISION, not
    // raw picstruct — this guarantees logs always reflect what the
    // filter actually did. "_combed" suffix appended when applyBlend
    // fires.
    const char *decTag;
    if (treatAsProgressive) {
        if      (mislabeled)        decTag = "FRAME_mislabeled";
        else if (progressiveCombed) decTag = "FRAME_combed";
        else                         decTag = "FRAME_clean";
    } else if (treatAsInterlaced) {
        // Raw-YUV-only branch (see treatAsInterlaced comment block above).
        // "TFF" / "BFF" tags in the log indicate the user passed raw
        // interlaced YUV via --raw/--y4m — never fires for avsw/avhw.
        decTag = (picstruct == RGY_PICSTRUCT_TFF) ? "TFF" : "BFF";
    } else {
        // The COMMON path for avsw/avhw decoded interlaced content:
        // real interlaced MPEG-2 arrives here as FRAME_TFF/FRAME_BFF.
        if (picstruct == RGY_PICSTRUCT_FRAME_TFF)      decTag = "FRAME_TFF_unknown";
        else if (picstruct == RGY_PICSTRUCT_FRAME_BFF) decTag = "FRAME_BFF_unknown";
        else                                            decTag = "UNKNOWN";
    }
    if (applyBlend) {
        // Build the combed variant. Each non-combed tag has a parallel
        // _combed tag; map here via string compare (kept simple rather
        // than factoring through an enum — the combined set is small).
        if      (strcmp(decTag, "FRAME_clean")        == 0) decTag = "FRAME_clean_combed";
        else if (strcmp(decTag, "FRAME_mislabeled")   == 0) decTag = "FRAME_mislabeled_combed";
        else if (strcmp(decTag, "FRAME_combed")       == 0) decTag = "FRAME_combed_blend";
        else if (strcmp(decTag, "TFF")                == 0) decTag = "TFF_combed";
        else if (strcmp(decTag, "BFF")                == 0) decTag = "BFF_combed";
        else if (strcmp(decTag, "FRAME_TFF_unknown")  == 0) decTag = "FRAME_TFF_unknown_combed";
        else if (strcmp(decTag, "FRAME_BFF_unknown")  == 0) decTag = "FRAME_BFF_unknown_combed";
        else if (strcmp(decTag, "UNKNOWN")            == 0) decTag = "UNKNOWN_combed";
    }

    // 4. マッチ結果を cycle スロット (= m_frameBuf[slot]) に合成。
    //    applyBlend フレームは常に 5-frame BWDIF カーネル経由で処理。
    //    applyBlend フレームは BWDIF に一本化。 TFF 情報は m_tffFixed から取る (init で
    //    picstruct から決定、必要なら per-frame で上書き可能)。
    // Per-pixel deinterlace gate: scale the user's 8-bit-domain dthresh
    // (rgy_prm.h default 7) to the input bit depth. 0 disables the gate
    // kernel-side.
    const int dthreshScaled = (prm->ivtc.dthresh > 0)
        ? std::max(1, (maxVal * prm->ivtc.dthresh) / 255)
        : 0;

    // BWDIF kernel TFF: prefer the frame's picstruct field when it
    // carries a valid TFF/BFF bit; otherwise fall back to m_tffFixed
    // (init-time default from input picstruct).
    int bwdifTff = m_tffFixed ? 1 : 0;
    if (picstruct == RGY_PICSTRUCT_TFF || picstruct == RGY_PICSTRUCT_FRAME_TFF)      bwdifTff = 1;
    else if (picstruct == RGY_PICSTRUCT_BFF || picstruct == RGY_PICSTRUCT_FRAME_BFF) bwdifTff = 0;

    // SUB-PHASE 3 (2026-04-25): when comb-first selection chose an
    // alternate-parity candidate, the field assembly must be performed
    // with the inverted tff so the synthesize kernel pulls the same
    // physical field rows that scoreCandidates measured at !tff.
    // Without this flip the gate would have approved an alt-parity
    // metric while the assembled output still used the primary
    // parity — re-introducing the very combing the alt selection
    // was supposed to avoid. The global bwdifTff is left untouched
    // (BWDIF is a pure deinterlace path that does not take a match
    // type, and m_tffFixed must remain the stream-level invariant).
    // Hoisted out of the dispatch below because the P/N+blend branch
    // (added 2026-04-25) also needs it.
    const int assemblyTff = matchAltParity ? (bwdifTff ? 0 : 1) : bwdifTff;

    // ============================================================================
    //  PIPELINE STAGE 9 — SYNTHESIS DISPATCH
    // ============================================================================
    // HARD REVERT (2026-04-25): the hybrid-deinterlace experiment was
    // removed in full. Both branches now call the match-aware
    // synthesizeToCycle kernel. The kernel's apply_blend parameter
    // selects between pure pix_match assembly (apply_blend=false) and
    // per-row motion-adaptive blend on combed rows (apply_blend=true).
    //
    // BWDIF (synthesizeToCycleBwdif) is NOT called from this dispatch.
    // synth-with-blend always honours the matcher's chosen P / N
    // cross-frame pairing, avoiding the frame-4058-class cadence-shift
    // artifact that BWDIF can introduce on P/N+blend frames.
    //
    // The TSV `post=blend` / `post=none` label tracks applyBlend
    // exactly, as before. assemblyTff carries the post-cadence parity
    // (matchAltParity is globally clamped to false, so this currently
    // equals bwdifTff for every frame).
    if (applyBlend) {
        err = synthesizeToCycle(slot,
            &m_cacheFrames[idx_prev]->frame,
            &m_cacheFrames[idx_cur ]->frame,
            &m_cacheFrames[idx_next]->frame,
            match, /*applyBlend=*/true, dthreshScaled, assemblyTff,
            stream);
    } else {
        err = synthesizeToCycle(slot,
            &m_cacheFrames[idx_prev]->frame,
            &m_cacheFrames[idx_cur ]->frame,
            &m_cacheFrames[idx_next]->frame,
            match, /*applyBlend=*/false, dthreshScaled, assemblyTff,
            stream);
    }
    if (err != RGY_ERR_NONE) {
        return err;
    }

    // 5. 入力メタデータを保持 (出力時に使うため)
    m_frameBuf[slot]->frame.picstruct = RGY_PICSTRUCT_FRAME;
    m_frameBuf[slot]->frame.flags     = curInfo.flags;
    if (cycleLen > 0) {
        m_cycleInPts[slot]      = curInfo.timestamp;
        m_cycleInDur[slot]      = curInfo.duration;
        m_cycleInputIds[slot]   = curInfo.inputFrameId;
        m_cycleMatchScore[slot] = chosenMatchScore;
        m_cycleCombScore[slot]  = chosenCombScore;
        m_cycleCombMax[slot]    = chosenCombMax;
        m_cycleCombBlocks[slot] = chosenCombBlocks;
        // SUB-PHASE 2 (2026-04-24) — store full primary + alternate triplets
        // for offline analysis. C/P/N at primary tff first, then C/P/N at !tff.
        m_cycleCombScorePrim[slot]   = { combScore[0],     combScore[1],     combScore[2]     };
        m_cycleCombMaxPrim[slot]     = { combMax[0],       combMax[1],       combMax[2]       };
        m_cycleCombBlocksPrim[slot]  = { combBlocks[0],    combBlocks[1],    combBlocks[2]    };
        m_cycleCombScoreAlt[slot]    = { combScoreAlt[0],  combScoreAlt[1],  combScoreAlt[2]  };
        m_cycleCombMaxAlt[slot]      = { combMaxAlt[0],    combMaxAlt[1],    combMaxAlt[2]    };
        m_cycleCombBlocksAlt[slot]   = { combBlocksAlt[0], combBlocksAlt[1], combBlocksAlt[2] };
        m_cycleBlendTrigger[slot] = blendTrigger;
        m_cycleMatchType[slot]  = (int)match;
        m_cycleApplyBlend[slot] = applyBlend ? 1 : 0;
        // SUB-PHASE 3 (2026-04-25): persist comb-first parity choice so
        // flushCycle can emit the matchParity TSV column. Stored as a
        // uint8_t flag (0=primary, 1=alt). Mirrors the synth-passthru
        // assignment in the early-return branch (line ~1997) which
        // forces 0 because synth never re-scores.
        m_cycleMatchAltParity[slot] = matchAltParity ? 1 : 0;
        // SUB-PHASE 5 (2026-04-25): persist confidence level so flushCycle
        // can emit the confidence TSV column. 0=HIGH, 1=MEDIUM, 2=LOW,
        // 3=VERY_LOW. Synth-passthru bypass at line ~1999 sets this to 0
        // (HIGH) because no scoring ran on those slots.
        m_cycleConfidence[slot] = (uint8_t)confidenceLevel;

        // Encode decTag as an int (decoder-driven routing decision — the
        // actual processing path taken). String-to-int lookup; symmetric
        // _combed variants have index = base_index + 7. Keep the mapping
        // compact for log compactness:
        //    0 = FRAME_clean                 7 = FRAME_clean_combed       (unreachable: progressive + !mislabeled never fires applyBlend)
        //    1 = FRAME_mislabeled            8 = FRAME_mislabeled_combed
        //    2 = TFF                         9 = TFF_combed
        //    3 = BFF                        10 = BFF_combed
        //    4 = FRAME_TFF_unknown          11 = FRAME_TFF_unknown_combed
        //    5 = FRAME_BFF_unknown          12 = FRAME_BFF_unknown_combed
        //    6 = UNKNOWN                    13 = UNKNOWN_combed
        // Indices 0..13 are the original set; 14..15 were appended for
        // Bug 2 (sw-decoder FRAME-with-combing) without renumbering so
        // existing log analysis scripts continue to decode correctly.
        // Index 16 (SYNTH_PASSTHRU) was appended by PATCH 1 (isSynth
        // bypass) for cross-time synth frames pushed from
        // flushExpandBuffer; they skip score/match/cadence/blend and
        // emit unchanged. Never reached by this line (the bypass
        // returns early before decTag is computed); documented here
        // for reference.
        // "FRAME_combed_blend" uses "_blend" rather than "_combed" as the
        // applyBlend-suffix to avoid the doubled "_combed_combed" name.
        m_cycleDecTag[slot] = (strcmp(decTag, "FRAME_clean") == 0)                ? 0
                            : (strcmp(decTag, "FRAME_mislabeled") == 0)           ? 1
                            : (strcmp(decTag, "TFF") == 0)                        ? 2
                            : (strcmp(decTag, "BFF") == 0)                        ? 3
                            : (strcmp(decTag, "FRAME_TFF_unknown") == 0)          ? 4
                            : (strcmp(decTag, "FRAME_BFF_unknown") == 0)          ? 5
                            : (strcmp(decTag, "UNKNOWN") == 0)                    ? 6
                            : (strcmp(decTag, "FRAME_clean_combed") == 0)         ? 7
                            : (strcmp(decTag, "FRAME_mislabeled_combed") == 0)    ? 8
                            : (strcmp(decTag, "TFF_combed") == 0)                 ? 9
                            : (strcmp(decTag, "BFF_combed") == 0)                 ? 10
                            : (strcmp(decTag, "FRAME_TFF_unknown_combed") == 0)   ? 11
                            : (strcmp(decTag, "FRAME_BFF_unknown_combed") == 0)   ? 12
                            : (strcmp(decTag, "UNKNOWN_combed") == 0)             ? 13
                            : (strcmp(decTag, "FRAME_combed") == 0)               ? 14
                            : (strcmp(decTag, "FRAME_combed_blend") == 0)         ? 15
                            :                                                       6 /* UNKNOWN */;

        // 6. 直前フレームとの SAD を計算。slot==0 の場合のみ前サイクル末尾 (保存スロット) と比較。
        uint64_t diff = 0;
        if (slot > 0) {
            err = computePairDiff(&m_frameBuf[slot]->frame, &m_frameBuf[slot - 1]->frame, diff, stream);
        } else if (m_hasSaveSlot) {
            err = computePairDiff(&m_frameBuf[slot]->frame, &m_frameBuf[cycleLen]->frame, diff, stream);
        } else {
            // 最初のサイクルの 1 フレーム目。比較対象がないので、decimate 候補から除外するため
            // 最大値を入れておく (最小値で選ばれる drop 候補にならない)。
            diff = std::numeric_limits<uint64_t>::max();
        }
        if (err != RGY_ERR_NONE) {
            return err;
        }
        m_cycleDiffPrev[slot] = diff;
        m_cycleSceneSAD[slot] = sceneSAD;
        m_cycleCadenceTag[slot] = cadenceTag;
        m_cycleIsSynth[slot] = 0;  // FIX B: coded path — not a synth
        m_cycleFilled++;
    } else {
        // cycle=0 の場合は即座にこの1枚を出力するため、メタをそのまま残す。
        m_frameBuf[slot]->frame.timestamp    = curInfo.timestamp;
        m_frameBuf[slot]->frame.duration     = curInfo.duration;
        m_frameBuf[slot]->frame.inputFrameId = curInfo.inputFrameId;

        // cycle=0 パスでは flushCycle を通らないので、per-frame ログもここで書く。
        // diff_to_prev カラムは cycle=0 では decimation SAD が存在しないため 0。
        // scene_sad カラムにシーンチェンジ検出の素値を載せるので、0 が続くのは
        // 本当にフレーム差がない場合 (静止) か最初のフレーム (比較対象なし)。
        if (m_fpLog) {
            const char *matchStr = ((int)match == 0) ? "c" : ((int)match == 1) ? "p" : "n";
            fprintf(m_fpLog.get(),
                "OUT:\t#%d\tin_id=%d\tmatch=%s\tmatchParity=%s\tconf=%d\tpost=%s\tstatus=%s\tdec=%s\tcadence=%s\tmQ=%llu\tcComb=%llu\tcCombMax=%llu\tcCombBlocks=%llu\t"
                "cComb_c=%llu\tcComb_p=%llu\tcComb_n=%llu\tcComb_cA=%llu\tcComb_pA=%llu\tcComb_nA=%llu\t"
                "cMax_c=%llu\tcMax_p=%llu\tcMax_n=%llu\tcMax_cA=%llu\tcMax_pA=%llu\tcMax_nA=%llu\t"
                "cBlk_c=%llu\tcBlk_p=%llu\tcBlk_n=%llu\tcBlk_cA=%llu\tcBlk_pA=%llu\tcBlk_nA=%llu\t"
                "btrig=%u\tpostComb=%llu\tdiff=%llu\tscene_sad=%llu\n",
                m_outputFrameCount,
                curInfo.inputFrameId,
                matchStr,
                matchAltParity ? "alt" : "pri",
                confidenceLevel,
                applyBlend ? "blend" : "none ",
                "emit ",
                decTag,
                ivtc_cadence_tag_str(cadenceTag),
                (unsigned long long)chosenMatchScore,
                (unsigned long long)chosenCombScore,
                (unsigned long long)chosenCombMax,
                (unsigned long long)chosenCombBlocks,
                (unsigned long long)combScore   [0], (unsigned long long)combScore   [1], (unsigned long long)combScore   [2],
                (unsigned long long)combScoreAlt[0], (unsigned long long)combScoreAlt[1], (unsigned long long)combScoreAlt[2],
                (unsigned long long)combMax     [0], (unsigned long long)combMax     [1], (unsigned long long)combMax     [2],
                (unsigned long long)combMaxAlt  [0], (unsigned long long)combMaxAlt  [1], (unsigned long long)combMaxAlt  [2],
                (unsigned long long)combBlocks  [0], (unsigned long long)combBlocks  [1], (unsigned long long)combBlocks  [2],
                (unsigned long long)combBlocksAlt[0],(unsigned long long)combBlocksAlt[1],(unsigned long long)combBlocksAlt[2],
                (unsigned)blendTrigger,
                (unsigned long long)chosenCombScore,
                0ull,
                (unsigned long long)sceneSAD);
            fflush(m_fpLog.get());
        }
        m_outputFrameCount++;
    }

    // Tick the scene-change carry-forward flag unconditionally. The BWDIF
    // kernel uses (sceneChange || m_lastSceneChange) to pick spatial-only
    // mode; overwriting with THIS frame's sceneChange makes the NEXT frame
    // see "previous was scene change" correctly. Runs for every frame
    // regardless of whether BWDIF fired this call.
    m_lastSceneChange = sceneChange;
    // Also snapshot the raw SAD for the next frame's adaptive threshold.
    m_lastSceneSAD    = sceneSAD;

    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::flushCycle(bool finalFlush, int64_t nextInputPts, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int filled = m_cycleFilled;
    if (filled <= 0) {
        return RGY_ERR_NONE;
    }

    // Drop 対象の決定: SAD の最小を持つ frame を落とす (3:2 プルダウン由来の重複)。
    // 完全サイクル (filled == cycleLen) 時のみ drop する。部分サイクル (finalFlush) は
    // shippable な情報だけ出す。
    //
    // FIX B (2026-04-24): two-phase selection. When expand=on, each cycle
    // contains a mix of coded frames and synth frames (the expansion-added
    // duplicates). Synths are structurally redundant — the decimator should
    // drop them before touching coded content. Phase 1 searches only synth
    // slots for the lowest-SAD candidate; Phase 2 falls back to normal
    // SAD-minimum over all slots when no synth is present in the cycle.
    // expand=off cycles have m_cycleIsSynth all zero, so Phase 2 is reached
    // immediately and behavior matches the pre-FIX-B path.
    int dropIdx = -1;
    if (cycleLen > 0 && filled == cycleLen && prm->ivtc.drop >= 1) {
        uint64_t minDiff = std::numeric_limits<uint64_t>::max();
        // Phase 1: prefer a synth slot. Among synths, still pick the
        // lowest-SAD one so the drop is the most duplicate-like synth
        // (least information loss).
        for (int i = 0; i < filled; i++) {
            if (m_cycleIsSynth[i] && m_cycleDiffPrev[i] < minDiff) {
                minDiff = m_cycleDiffPrev[i];
                dropIdx = i;
            }
        }
        // Phase 2: no synth in this cycle — fall back to global SAD argmin.
        if (dropIdx < 0) {
            minDiff = std::numeric_limits<uint64_t>::max();
            for (int i = 0; i < filled; i++) {
                if (m_cycleDiffPrev[i] < minDiff) {
                    minDiff = m_cycleDiffPrev[i];
                    dropIdx = i;
                }
            }
        }
    }
    const int dropCount = (dropIdx >= 0) ? 1 : 0;
    const int emitCount = filled - dropCount;

    // CFR output timestamp strategy (drift-free, rescale-based):
    //   emit[N].pts      = seed + rational_rescale(N, 1/baseFps, timebase)
    //   emit[N].duration = rational_rescale(N+1, ...) - rational_rescale(N, ...)
    // where baseFps is the OUTPUT frame rate (already adjusted by cycle*(cycle-drop)/cycle
    // in init) and timebase is the output stream timebase.
    //
    // This is strictly CFR: pts and duration depend only on the global emit index,
    // not on any per-cycle input-timestamp jitter. Fixes VFR-output regression
    // introduced when --vpp-rff (duration splitting 2/3+1/3) feeds IVTC — previously
    // IVTC computed baseDur = totalCycleDur / emitCount and latched on the first
    // full cycle, which produced a fixed baseDur but DIDN'T compensate against
    // rff-jittered input timestamps accumulating drift between cycles. The rescale
    // approach uses only the output baseFps and stream timebase, so input jitter
    // never influences the output timeline. MediaInfo will report CFR.
    bool ptsInvalid = false;
    for (int i = 0; i < filled; i++) {
        if (m_cycleInPts[i] == AV_NOPTS_VALUE) { ptsInvalid = true; break; }
    }
    if (!m_nPtsInit && emitCount > 0) {
        m_nPts = ptsInvalid ? 0 : m_cycleInPts[0];
        m_nPtsInit = true;
    }

    // Derive the period rational (1/baseFps) once per flush. Cached base dur is
    // kept for log/print consistency but the actual emit math uses rescale.
    const rgy_rational<int> fpsPeriod = prm->baseFps.inv();
    const rgy_rational<int> outTb     = prm->timebase;
    if (m_cfrBaseDur <= 0 && fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0) {
        m_cfrBaseDur = rational_rescale(1, fpsPeriod, outTb);
        if (m_cfrBaseDur <= 0) m_cfrBaseDur = 1;
    }
    // Fallback if rational data is degenerate: degrade to the old avg-from-input
    // computation, matching pre-fix behavior. This path should never run for a
    // normally-configured pipeline but protects against surprising param state.
    int64_t fallbackBaseDur = 0;
    const bool cfrReady = (fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0);
    if (!cfrReady) {
        int64_t totalCycleDur = 0;
        if (!ptsInvalid && nextInputPts != AV_NOPTS_VALUE && filled >= 1) {
            totalCycleDur = nextInputPts - m_cycleInPts[0];
        }
        if (totalCycleDur <= 0) {
            for (int i = 0; i < filled; i++) totalCycleDur += m_cycleInDur[i];
        }
        fallbackBaseDur = (emitCount > 0 && totalCycleDur > 0) ? (totalCycleDur / emitCount) : 1;
        if (fallbackBaseDur <= 0) fallbackBaseDur = 1;
    }

    // emit されるフレームを staging にコピーし、IvtcEmitEntry を m_emitQueue に積む。
    // staging 領域は m_frameBuf[m_stagingBase .. m_stagingBase+stagingCount-1] (cycleLen-drop 枚)。
    int emitted = 0;
    for (int i = 0; i < filled; i++) {
        if (i == dropIdx) continue;
        const int stagingIdx = m_stagingBase + emitted;
        if (stagingIdx >= (int)m_frameBuf.size()) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc staging overflow: idx=%d size=%lld.\n"), stagingIdx, (long long)m_frameBuf.size());
            return RGY_ERR_UNKNOWN;
        }
        auto cpErr = copyFrameAsync(&m_frameBuf[stagingIdx]->frame, &m_frameBuf[i]->frame, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy emit frame to staging[%d]: %s.\n"), stagingIdx, get_err_mes(cpErr));
            return cpErr;
        }
        IvtcEmitEntry e{};
        e.stagingIdx    = stagingIdx;
        e.inputFrameId  = m_cycleInputIds[i];
        // Drift-free CFR: each emit gets pts = seed + rescale(idx, period, tb),
        // duration = rescale(idx+1, ...) - rescale(idx, ...). Input timestamp
        // jitter (from --vpp-rff's 2/3+1/3 duration splitting, etc.) is
        // ignored for the output timeline.
        if (cfrReady) {
            const int64_t ptsCur  = rational_rescale((int64_t)m_cfrEmitIdx,     fpsPeriod, outTb);
            const int64_t ptsNext = rational_rescale((int64_t)m_cfrEmitIdx + 1, fpsPeriod, outTb);
            e.timestamp = m_nPts + ptsCur;
            e.duration  = ptsNext - ptsCur;
            if (e.duration <= 0) e.duration = m_cfrBaseDur > 0 ? m_cfrBaseDur : 1;
        } else {
            e.timestamp = m_nPts;
            e.duration  = fallbackBaseDur;
            m_nPts     += fallbackBaseDur;  // accumulator mode for the fallback
        }
        m_cfrEmitIdx++;
        m_emitQueue.push_back(e);
        AddMessage(RGY_LOG_DEBUG, _T("ivtc enqueue[%d]: cycleSlot=%d staging=%d pts=%lld dur=%lld inputId=%d match=%d blend=%d (drop=%d mQ=%llu cComb=%llu diff=%llu)\n"),
            emitted, i, stagingIdx,
            (long long)e.timestamp, (long long)e.duration,
            e.inputFrameId, m_cycleMatchType[i], m_cycleApplyBlend[i],
            dropIdx,
            (unsigned long long)m_cycleMatchScore[i],
            (unsigned long long)m_cycleCombScore[i],
            (unsigned long long)m_cycleDiffPrev[i]);
        emitted++;
    }

    // 次サイクルの SAD[0] 用に、今サイクル末尾を保存スロットにコピーしておく。
    if (cycleLen > 0 && filled >= 1 && !finalFlush) {
        const RGYFrameInfo *pLast = &m_frameBuf[filled - 1]->frame;
        RGYFrameInfo *pSave = &m_frameBuf[cycleLen]->frame;
        auto cpErr = copyFrameAsync(pSave, pLast, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy cycle tail to save slot: %s.\n"), get_err_mes(cpErr));
            return cpErr;
        }
        m_hasSaveSlot = true;
    }

    // per-frame ログ (emit / DROP 両方)。out_idx は emit 順で付番。
    if (m_fpLog) {
        for (int i = 0; i < filled; i++) {
            const char *matchStr = (m_cycleMatchType[i] == 0) ? "c" : (m_cycleMatchType[i] == 1) ? "p" : "n";
            const char *status = (i == dropIdx) ? "DROP " : "emit ";
            const char *decCycleTag = (m_cycleDecTag[i] ==  0) ? "FRAME_clean"
                                    : (m_cycleDecTag[i] ==  1) ? "FRAME_mislabeled"
                                    : (m_cycleDecTag[i] ==  2) ? "TFF"
                                    : (m_cycleDecTag[i] ==  3) ? "BFF"
                                    : (m_cycleDecTag[i] ==  4) ? "FRAME_TFF_unknown"
                                    : (m_cycleDecTag[i] ==  5) ? "FRAME_BFF_unknown"
                                    : (m_cycleDecTag[i] ==  6) ? "UNKNOWN"
                                    : (m_cycleDecTag[i] ==  7) ? "FRAME_clean_combed"
                                    : (m_cycleDecTag[i] ==  8) ? "FRAME_mislabeled_combed"
                                    : (m_cycleDecTag[i] ==  9) ? "TFF_combed"
                                    : (m_cycleDecTag[i] == 10) ? "BFF_combed"
                                    : (m_cycleDecTag[i] == 11) ? "FRAME_TFF_unknown_combed"
                                    : (m_cycleDecTag[i] == 12) ? "FRAME_BFF_unknown_combed"
                                    : (m_cycleDecTag[i] == 13) ? "UNKNOWN_combed"
                                    : (m_cycleDecTag[i] == 14) ? "FRAME_combed"
                                    : (m_cycleDecTag[i] == 15) ? "FRAME_combed_blend"
                                    : (m_cycleDecTag[i] == 16) ? "SYNTH_PASSTHRU"
                                                               : "UNKNOWN";
            fprintf(m_fpLog.get(),
                "OUT:\t#%d\tin_id=%d\tmatch=%s\tmatchParity=%s\tconf=%u\tpost=%s\tstatus=%s\tdec=%s\tcadence=%s\tmQ=%llu\tcComb=%llu\tcCombMax=%llu\tcCombBlocks=%llu\t"
                "cComb_c=%llu\tcComb_p=%llu\tcComb_n=%llu\tcComb_cA=%llu\tcComb_pA=%llu\tcComb_nA=%llu\t"
                "cMax_c=%llu\tcMax_p=%llu\tcMax_n=%llu\tcMax_cA=%llu\tcMax_pA=%llu\tcMax_nA=%llu\t"
                "cBlk_c=%llu\tcBlk_p=%llu\tcBlk_n=%llu\tcBlk_cA=%llu\tcBlk_pA=%llu\tcBlk_nA=%llu\t"
                "btrig=%u\tpostComb=%llu\tdiff=%llu\tscene_sad=%llu\n",
                m_outputFrameCount + ((i < dropIdx || dropIdx < 0) ? i : i - 1),
                m_cycleInputIds[i],
                matchStr,
                m_cycleMatchAltParity[i] ? "alt" : "pri",
                (unsigned)m_cycleConfidence[i],
                m_cycleApplyBlend[i] ? "blend" : "none ",
                status,
                decCycleTag,
                ivtc_cadence_tag_str(m_cycleCadenceTag[i]),
                (unsigned long long)m_cycleMatchScore[i],
                (unsigned long long)m_cycleCombScore[i],
                (unsigned long long)m_cycleCombMax[i],
                (unsigned long long)m_cycleCombBlocks[i],
                (unsigned long long)m_cycleCombScorePrim [i][0], (unsigned long long)m_cycleCombScorePrim [i][1], (unsigned long long)m_cycleCombScorePrim [i][2],
                (unsigned long long)m_cycleCombScoreAlt  [i][0], (unsigned long long)m_cycleCombScoreAlt  [i][1], (unsigned long long)m_cycleCombScoreAlt  [i][2],
                (unsigned long long)m_cycleCombMaxPrim   [i][0], (unsigned long long)m_cycleCombMaxPrim   [i][1], (unsigned long long)m_cycleCombMaxPrim   [i][2],
                (unsigned long long)m_cycleCombMaxAlt    [i][0], (unsigned long long)m_cycleCombMaxAlt    [i][1], (unsigned long long)m_cycleCombMaxAlt    [i][2],
                (unsigned long long)m_cycleCombBlocksPrim[i][0], (unsigned long long)m_cycleCombBlocksPrim[i][1], (unsigned long long)m_cycleCombBlocksPrim[i][2],
                (unsigned long long)m_cycleCombBlocksAlt [i][0], (unsigned long long)m_cycleCombBlocksAlt [i][1], (unsigned long long)m_cycleCombBlocksAlt [i][2],
                (unsigned)m_cycleBlendTrigger[i],
                (unsigned long long)m_cycleCombScore[i],
                // diff: the first frame of the very first cycle has no prev
                // to compare against; sentinel UINT64_MAX is stored to keep
                // it out of the decimation-argmin, but print 0 for
                // readability.
                (unsigned long long)((m_cycleDiffPrev[i] == std::numeric_limits<uint64_t>::max()) ? 0ULL : m_cycleDiffPrev[i]),
                (unsigned long long)m_cycleSceneSAD[i]);
        }
        fflush(m_fpLog.get());
    }
    m_cycleFilled = 0;
    return RGY_ERR_NONE;
}

void NVEncFilterIvtc::resetMixedTemporalState() {
    m_mixedLastInputPts = AV_NOPTS_VALUE;
    m_mixedLastInputDur = 0;
    m_mixedLastEmitEndPts = AV_NOPTS_VALUE;
    m_mixedLastInputValid = false;
}

void NVEncFilterIvtc::resetMixedRffState() {
    m_mixedRffPendingTopValid = false;
    m_mixedRffPendingBottomValid = false;
    m_mixedRffPendingTopInfo = RGYFrameInfo();
    m_mixedRffPendingBottomInfo = RGYFrameInfo();
}

RGY_ERR NVEncFilterIvtc::pushMixedEmitEntry(int stagingIdx, const RGYFrameInfo *srcInfo) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm || !srcInfo) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_nPtsInit) {
        m_nPts = (srcInfo->timestamp == AV_NOPTS_VALUE) ? 0 : srcInfo->timestamp;
        m_nPtsInit = true;
    }
    const rgy_rational<int> fpsPeriod = prm->baseFps.inv();
    const rgy_rational<int> outTb = prm->timebase;
    if (m_cfrBaseDur <= 0 && fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0) {
        m_cfrBaseDur = rational_rescale(1, fpsPeriod, outTb);
        if (m_cfrBaseDur <= 0) m_cfrBaseDur = 1;
    }

    IvtcEmitEntry e{};
    e.stagingIdx = stagingIdx;
    e.inputFrameId = srcInfo->inputFrameId;
    if (fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0) {
        const int64_t ptsCur = rational_rescale((int64_t)m_cfrEmitIdx, fpsPeriod, outTb);
        const int64_t ptsNext = rational_rescale((int64_t)m_cfrEmitIdx + 1, fpsPeriod, outTb);
        e.timestamp = m_nPts + ptsCur;
        e.duration = ptsNext - ptsCur;
        if (e.duration <= 0) e.duration = m_cfrBaseDur > 0 ? m_cfrBaseDur : 1;
    } else {
        const int64_t fallbackDur = (srcInfo->duration > 0) ? srcInfo->duration : 1;
        e.timestamp = m_nPts;
        e.duration = fallbackDur;
        m_nPts += fallbackDur;
    }
    m_cfrEmitIdx++;
    m_mixedLastEmitEndPts = e.timestamp + e.duration;
    m_emitQueue.push_back(e);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::enqueueMixedDirectFrame(const CUFrameBuf *srcFrame, const RGYFrameInfo *srcInfo, const char *decTag, const char *section, cudaStream_t stream) {
    if (!srcFrame || !srcInfo || m_mixedDirectStagingCount <= 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int outIdx = m_outputFrameCount + (int)m_emitQueue.size();
    const int stagingIdx = m_mixedDirectStagingBase + m_mixedDirectStagingNext;
    m_mixedDirectStagingNext = (m_mixedDirectStagingNext + 1) % m_mixedDirectStagingCount;

    auto cpErr = copyFrameAsync(&m_frameBuf[stagingIdx]->frame, &srcFrame->frame, stream);
    if (cpErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("ivtc mixed direct copy failed: %s.\n"), get_err_mes(cpErr));
        return cpErr;
    }
    auto &dst = m_frameBuf[stagingIdx]->frame;
    dst.picstruct = RGY_PICSTRUCT_FRAME;
    dst.flags = (RGY_FRAME_FLAGS)(srcInfo->flags & ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF));
    dst.timestamp = srcInfo->timestamp;
    dst.duration = srcInfo->duration;
    dst.inputFrameId = srcInfo->inputFrameId;

    auto err = pushMixedEmitEntry(stagingIdx, srcInfo);
    if (err != RGY_ERR_NONE) return err;

    if (m_fpLog) {
        fprintf(m_fpLog.get(),
            "OUT:\t#%d\tin_id=%d\tmatch=c\tmatchParity=pri\tconf=0\tpost=none \tstatus=emit \tdec=%s\tcadence=direct\tmQ=0\tcComb=0\tcCombMax=0\tcCombBlocks=0\t"
            "cComb_c=0\tcComb_p=0\tcComb_n=0\tcComb_cA=0\tcComb_pA=0\tcComb_nA=0\t"
            "cMax_c=0\tcMax_p=0\tcMax_n=0\tcMax_cA=0\tcMax_pA=0\tcMax_nA=0\t"
            "cBlk_c=0\tcBlk_p=0\tcBlk_n=0\tcBlk_cA=0\tcBlk_pA=0\tcBlk_nA=0\t"
            "btrig=0\tpostComb=0\tdiff=0\tscene_sad=0\tsection=%s\n",
            outIdx, srcInfo->inputFrameId, decTag ? decTag : "DIRECT", section ? section : "pass");
        fflush(m_fpLog.get());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::enqueueMixedPassthrough(const RGYFrameInfo *frame, int cacheIdx, int64_t nextPts, cudaStream_t stream) {
    (void)nextPts;
    if (!frame || cacheIdx < 0 || cacheIdx >= (int)m_cacheFrames.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
    const auto section = ivtcMixedClassify(frame);
    return enqueueMixedDirectFrame(m_cacheFrames[cacheIdx].get(), frame, (section == IvtcMixedSection::Rff) ? "RFF_DIRECT_COPY" : "PROG_PASSTHRU", (section == IvtcMixedSection::Rff) ? "rff" : "pass", stream);
}

RGY_ERR NVEncFilterIvtc::appendMixedRffDisplayFrame(const CUFrameBuf *topFrame, const RGYFrameInfo *topInfo, const CUFrameBuf *bottomFrame, const RGYFrameInfo *bottomInfo, int decTag, cudaStream_t stream) {
    if (!topFrame || !topInfo || !bottomFrame || !bottomInfo || m_mixedDirectStagingCount <= 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int outIdx = m_outputFrameCount + (int)m_emitQueue.size();
    const int stagingIdx = m_mixedDirectStagingBase + m_mixedDirectStagingNext;
    m_mixedDirectStagingNext = (m_mixedDirectStagingNext + 1) % m_mixedDirectStagingCount;

    auto cpErr = copyFrameAsync(&m_frameBuf[stagingIdx]->frame, &topFrame->frame, stream);
    if (cpErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("ivtc mixed RFF display copy failed: %s.\n"), get_err_mes(cpErr));
        return cpErr;
    }
    if (topFrame != bottomFrame) {
        auto ovErr = overlayField(m_frameBuf[stagingIdx].get(), bottomFrame, /*tff=*/1, stream);
        if (ovErr != RGY_ERR_NONE) return ovErr;
    }
    auto &dst = m_frameBuf[stagingIdx]->frame;
    dst.picstruct = RGY_PICSTRUCT_FRAME;
    dst.flags = (RGY_FRAME_FLAGS)(topInfo->flags & ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF));
    const bool synth = topFrame != bottomFrame;
    int64_t pts = topInfo->timestamp;
    if (synth && topInfo->timestamp != AV_NOPTS_VALUE && bottomInfo->timestamp != AV_NOPTS_VALUE) {
        pts = std::max(topInfo->timestamp, bottomInfo->timestamp);
    }
    dst.timestamp = pts;
    dst.duration = topInfo->duration;
    dst.inputFrameId = topInfo->inputFrameId;

    auto err = pushMixedEmitEntry(stagingIdx, &dst);
    if (err != RGY_ERR_NONE) return err;

    if (m_fpLog) {
        fprintf(m_fpLog.get(),
            "OUT:\t#%d\tin_id=%d\tmatch=c\tmatchParity=pri\tconf=0\tpost=none \tstatus=emit \tdec=%s\tcadence=direct\tmQ=0\tcComb=0\tcCombMax=0\tcCombBlocks=0\t"
            "cComb_c=0\tcComb_p=0\tcComb_n=0\tcComb_cA=0\tcComb_pA=0\tcComb_nA=0\t"
            "cMax_c=0\tcMax_p=0\tcMax_n=0\tcMax_cA=0\tcMax_pA=0\tcMax_nA=0\t"
            "cBlk_c=0\tcBlk_p=0\tcBlk_n=0\tcBlk_cA=0\tcBlk_pA=0\tcBlk_nA=0\t"
            "btrig=0\tpostComb=0\tdiff=0\tscene_sad=0\tsection=rff\n",
            outIdx, dst.inputFrameId, (decTag == 19) ? "RFF_RECON_FIELD" : "RFF_RECON_COPY");
        fflush(m_fpLog.get());
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::setMixedRffPending(const CUFrameBuf *srcFrame, const RGYFrameInfo *srcInfo, bool pendingTop, cudaStream_t stream) {
    auto &pendingFrame = pendingTop ? m_mixedRffPendingTopFrame : m_mixedRffPendingBottomFrame;
    auto &pendingInfo = pendingTop ? m_mixedRffPendingTopInfo : m_mixedRffPendingBottomInfo;
    auto &pendingValid = pendingTop ? m_mixedRffPendingTopValid : m_mixedRffPendingBottomValid;
    if (!pendingFrame || !srcFrame || !srcInfo) {
        return RGY_ERR_INVALID_PARAM;
    }
    auto cpErr = copyFrameAsync(&pendingFrame->frame, &srcFrame->frame, stream);
    if (cpErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("ivtc mixed RFF pending copy failed: %s.\n"), get_err_mes(cpErr));
        return cpErr;
    }
    pendingInfo = *srcInfo;
    pendingValid = true;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::enqueueMixedRffFrame(int cacheIdx, int64_t nextPts, cudaStream_t stream) {
    (void)nextPts;
    if (cacheIdx < 0 || cacheIdx >= (int)m_cacheFrames.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
    auto *curFrame = m_cacheFrames[cacheIdx].get();
    auto *curInfo = &curFrame->frame;

    auto err = RGY_ERR_NONE;
    if (m_mixedRffPendingTopValid || m_mixedRffPendingBottomValid) {
        const bool pendingTop = m_mixedRffPendingTopValid;
        const auto *topFrame = pendingTop ? m_mixedRffPendingTopFrame.get() : curFrame;
        const auto *topInfo = pendingTop ? &m_mixedRffPendingTopInfo : curInfo;
        const auto *bottomFrame = pendingTop ? curFrame : m_mixedRffPendingBottomFrame.get();
        const auto *bottomInfo = pendingTop ? curInfo : &m_mixedRffPendingBottomInfo;

        auto cpErr = copyFrameAsync(&m_expandSynth->frame, &topFrame->frame, stream);
        if (cpErr != RGY_ERR_NONE) return cpErr;
        if (topFrame != bottomFrame) {
            auto ovErr = overlayField(m_expandSynth.get(), bottomFrame, /*tff=*/1, stream);
            if (ovErr != RGY_ERR_NONE) return ovErr;
        }
        auto synthInfo = m_expandSynth->frame;
        synthInfo.picstruct = RGY_PICSTRUCT_FRAME;
        synthInfo.flags = (RGY_FRAME_FLAGS)(topInfo->flags & ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF));
        synthInfo.timestamp = (topInfo->timestamp != AV_NOPTS_VALUE && bottomInfo->timestamp != AV_NOPTS_VALUE) ? std::max(topInfo->timestamp, bottomInfo->timestamp) : topInfo->timestamp;
        synthInfo.duration = topInfo->duration;
        synthInfo.inputFrameId = topInfo->inputFrameId;

        struct CombMetric {
            uint64_t blocks;
            uint64_t max;
            uint64_t score;
        };
        auto measureComb = [&](const RGYFrameInfo *frame, const int tff, CombMetric &metric) -> RGY_ERR {
            uint64_t matchScore[3] = {};
            uint64_t combScore[3] = {};
            uint64_t combMax[3] = {};
            uint64_t combBlocks[3] = {};
            auto scoreErr = scoreCandidates(frame, frame, frame, matchScore, combScore, combMax, combBlocks, tff, stream);
            if (scoreErr != RGY_ERR_NONE) return scoreErr;
            metric = { combBlocks[0], combMax[0], combScore[0] };
            return RGY_ERR_NONE;
        };
        auto combLess = [](const CombMetric &a, const CombMetric &b) {
            if (a.blocks != b.blocks) return a.blocks < b.blocks;
            if (a.max != b.max) return a.max < b.max;
            return a.score < b.score;
        };
        auto measureBestComb = [&](const RGYFrameInfo *frame, CombMetric &metric) -> RGY_ERR {
            CombMetric tff0 = {};
            CombMetric tff1 = {};
            auto scoreErr = measureComb(frame, 0, tff0);
            if (scoreErr != RGY_ERR_NONE) return scoreErr;
            scoreErr = measureComb(frame, 1, tff1);
            if (scoreErr != RGY_ERR_NONE) return scoreErr;
            metric = combLess(tff1, tff0) ? tff1 : tff0;
            return RGY_ERR_NONE;
        };
        auto combClean = [](const CombMetric &m) {
            return m.blocks == 0 && m.max <= COMB_CLEAN_BOUND && m.score <= COMB_CLEAN_BOUND;
        };

        CombMetric copyComb = {};
        CombMetric synthComb = {};
        err = measureBestComb(curInfo, copyComb);
        if (err != RGY_ERR_NONE) return err;
        err = measureBestComb(&synthInfo, synthComb);
        if (err != RGY_ERR_NONE) return err;

        if (!combClean(copyComb) && combLess(synthComb, copyComb)) {
            err = enqueueMixedDirectFrame(m_expandSynth.get(), &synthInfo, "RFF_RECON_FIELD", "rff", stream);
        } else {
            err = enqueueMixedDirectFrame(curFrame, curInfo, "RFF_RECON_COPY", "rff", stream);
        }
        resetMixedRffState();
    } else {
        err = enqueueMixedDirectFrame(curFrame, curInfo, "RFF_RECON_COPY", "rff", stream);
    }
    if (err != RGY_ERR_NONE) return err;

    if (curInfo->flags & RGY_FRAME_FLAG_RFF) {
        bool repeatTop = m_tffFixed != 0;
        if (curInfo->flags & RGY_FRAME_FLAG_RFF_TFF) repeatTop = true;
        else if (curInfo->flags & RGY_FRAME_FLAG_RFF_BFF) repeatTop = false;
        else if (curInfo->picstruct == RGY_PICSTRUCT_FRAME_TFF || curInfo->picstruct == RGY_PICSTRUCT_TFF) repeatTop = true;
        else if (curInfo->picstruct == RGY_PICSTRUCT_FRAME_BFF || curInfo->picstruct == RGY_PICSTRUCT_BFF) repeatTop = false;

        err = setMixedRffPending(curFrame, curInfo, repeatTop, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::flushCycleMixed(bool finalFlush, int64_t cycleEndPts, bool allowDrop, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int filled = m_cycleFilled;
    if (filled <= 0) return RGY_ERR_NONE;

    int dropIdx = -1;
    if (!finalFlush && allowDrop && cycleLen > 0 && filled == cycleLen && prm->ivtc.drop >= 1) {
        uint64_t minDiff = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < filled; i++) {
            if (m_cycleIsSynth[i] && m_cycleDiffPrev[i] < minDiff) {
                minDiff = m_cycleDiffPrev[i];
                dropIdx = i;
            }
        }
        if (dropIdx < 0) {
            minDiff = std::numeric_limits<uint64_t>::max();
            for (int i = 0; i < filled; i++) {
                if (m_cycleDiffPrev[i] < minDiff) {
                    minDiff = m_cycleDiffPrev[i];
                    dropIdx = i;
                }
            }
        }
    }
    const int dropCount = (dropIdx >= 0) ? 1 : 0;
    const int emitCount = filled - dropCount;

    (void)cycleEndPts;
    bool ptsInvalid = false;
    for (int i = 0; i < filled; i++) {
        if (m_cycleInPts[i] == AV_NOPTS_VALUE) {
            ptsInvalid = true;
            break;
        }
    }
    if (!m_nPtsInit && emitCount > 0) {
        m_nPts = ptsInvalid ? 0 : m_cycleInPts[0];
        m_nPtsInit = true;
    }

    const rgy_rational<int> fpsPeriod = prm->baseFps.inv();
    const rgy_rational<int> outTb = prm->timebase;
    if (m_cfrBaseDur <= 0 && fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0) {
        m_cfrBaseDur = rational_rescale(1, fpsPeriod, outTb);
        if (m_cfrBaseDur <= 0) m_cfrBaseDur = 1;
    }
    const bool cfrReady = fpsPeriod.n() > 0 && fpsPeriod.d() > 0 && outTb.n() > 0 && outTb.d() > 0;
    int64_t fallbackBaseDur = m_cfrBaseDur > 0 ? m_cfrBaseDur : 1;
    if (!cfrReady) {
        int64_t sumDur = 0;
        for (int i = 0; i < filled; i++) {
            if (m_cycleInDur[i] > 0) sumDur += m_cycleInDur[i];
        }
        fallbackBaseDur = (emitCount > 0 && sumDur > 0) ? (sumDur / emitCount) : 1;
        if (fallbackBaseDur <= 0) fallbackBaseDur = 1;
    }

    int emitted = 0;
    for (int i = 0; i < filled; i++) {
        if (i == dropIdx) continue;
        const int stagingIdx = m_stagingBase + emitted;
        if (stagingIdx >= (int)m_frameBuf.size()) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed cycle staging overflow: idx=%d size=%lld.\n"), stagingIdx, (long long)m_frameBuf.size());
            return RGY_ERR_UNKNOWN;
        }
        auto cpErr = copyFrameAsync(&m_frameBuf[stagingIdx]->frame, &m_frameBuf[i]->frame, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc mixed cycle copy failed: %s.\n"), get_err_mes(cpErr));
            return cpErr;
        }
        IvtcEmitEntry e{};
        e.stagingIdx = stagingIdx;
        e.inputFrameId = m_cycleInputIds[i];
        if (cfrReady) {
            const int64_t ptsCur = rational_rescale((int64_t)m_cfrEmitIdx, fpsPeriod, outTb);
            const int64_t ptsNext = rational_rescale((int64_t)m_cfrEmitIdx + 1, fpsPeriod, outTb);
            e.timestamp = m_nPts + ptsCur;
            e.duration = ptsNext - ptsCur;
            if (e.duration <= 0) e.duration = m_cfrBaseDur > 0 ? m_cfrBaseDur : 1;
        } else {
            e.timestamp = m_nPts;
            e.duration = fallbackBaseDur;
            m_nPts += fallbackBaseDur;
        }
        m_cfrEmitIdx++;
        m_mixedLastEmitEndPts = e.timestamp + e.duration;
        m_emitQueue.push_back(e);
        emitted++;
    }

    if (cycleLen > 0 && filled >= 1 && !finalFlush) {
        const RGYFrameInfo *pLast = &m_frameBuf[filled - 1]->frame;
        RGYFrameInfo *pSave = &m_frameBuf[cycleLen]->frame;
        auto cpErr = copyFrameAsync(pSave, pLast, stream);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy mixed cycle tail to save slot: %s.\n"), get_err_mes(cpErr));
            return cpErr;
        }
        m_hasSaveSlot = true;
    } else if (finalFlush) {
        m_hasSaveSlot = false;
    }

    if (m_fpLog) {
        for (int i = 0; i < filled; i++) {
            const char *matchStr = (m_cycleMatchType[i] == 0) ? "c" : (m_cycleMatchType[i] == 1) ? "p" : "n";
            const char *mixedDec = (m_cycleDecTag[i] == 17) ? "RFF_RECON_COPY"
                                 : (m_cycleDecTag[i] == 18) ? "PROG_PASSTHRU"
                                 : (m_cycleDecTag[i] == 19) ? "RFF_RECON_FIELD"
                                                            : "MIXED_INTERLACED";
            const char *mixedSection = (m_cycleDecTag[i] == 17 || m_cycleDecTag[i] == 19) ? "rff"
                                     : (m_cycleDecTag[i] == 18) ? "pass"
                                                                : "interlaced";
            fprintf(m_fpLog.get(),
                "OUT:\t#%d\tin_id=%d\tmatch=%s\tmatchParity=%s\tconf=%u\tpost=%s\tstatus=%s\tdec=%s\tcadence=%s\tmQ=%llu\tcComb=%llu\tcCombMax=%llu\tcCombBlocks=%llu\t"
                "cComb_c=%llu\tcComb_p=%llu\tcComb_n=%llu\tcComb_cA=%llu\tcComb_pA=%llu\tcComb_nA=%llu\t"
                "cMax_c=%llu\tcMax_p=%llu\tcMax_n=%llu\tcMax_cA=%llu\tcMax_pA=%llu\tcMax_nA=%llu\t"
                "cBlk_c=%llu\tcBlk_p=%llu\tcBlk_n=%llu\tcBlk_cA=%llu\tcBlk_pA=%llu\tcBlk_nA=%llu\t"
                "btrig=%u\tpostComb=%llu\tdiff=%llu\tscene_sad=%llu\tsection=%s\n",
                m_outputFrameCount + ((i < dropIdx || dropIdx < 0) ? i : i - 1),
                m_cycleInputIds[i],
                matchStr,
                m_cycleMatchAltParity[i] ? "alt" : "pri",
                (unsigned)m_cycleConfidence[i],
                m_cycleApplyBlend[i] ? "blend" : "none ",
                (i == dropIdx) ? "DROP " : "emit ",
                mixedDec,
                ivtc_cadence_tag_str(m_cycleCadenceTag[i]),
                (unsigned long long)m_cycleMatchScore[i],
                (unsigned long long)m_cycleCombScore[i],
                (unsigned long long)m_cycleCombMax[i],
                (unsigned long long)m_cycleCombBlocks[i],
                (unsigned long long)m_cycleCombScorePrim [i][0], (unsigned long long)m_cycleCombScorePrim [i][1], (unsigned long long)m_cycleCombScorePrim [i][2],
                (unsigned long long)m_cycleCombScoreAlt  [i][0], (unsigned long long)m_cycleCombScoreAlt  [i][1], (unsigned long long)m_cycleCombScoreAlt  [i][2],
                (unsigned long long)m_cycleCombMaxPrim   [i][0], (unsigned long long)m_cycleCombMaxPrim   [i][1], (unsigned long long)m_cycleCombMaxPrim   [i][2],
                (unsigned long long)m_cycleCombMaxAlt    [i][0], (unsigned long long)m_cycleCombMaxAlt    [i][1], (unsigned long long)m_cycleCombMaxAlt    [i][2],
                (unsigned long long)m_cycleCombBlocksPrim[i][0], (unsigned long long)m_cycleCombBlocksPrim[i][1], (unsigned long long)m_cycleCombBlocksPrim[i][2],
                (unsigned long long)m_cycleCombBlocksAlt [i][0], (unsigned long long)m_cycleCombBlocksAlt [i][1], (unsigned long long)m_cycleCombBlocksAlt [i][2],
                (unsigned)m_cycleBlendTrigger[i],
                (unsigned long long)m_cycleCombScore[i],
                (unsigned long long)((m_cycleDiffPrev[i] == std::numeric_limits<uint64_t>::max()) ? 0ULL : m_cycleDiffPrev[i]),
                (unsigned long long)m_cycleSceneSAD[i],
                mixedSection);
        }
        fflush(m_fpLog.get());
    }
    m_cycleFilled = 0;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::partialFlushMixed(cudaStream_t stream, int64_t cycleEndPts) {
    if (m_cycleFilled <= 0) return RGY_ERR_NONE;
    auto err = flushCycleMixed(true, cycleEndPts, false, stream);
    if (err != RGY_ERR_NONE) return err;
    m_hasSaveSlot = false;
    resetCadenceState();
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::popEmit(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    if (m_emitQueue.empty()) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    constexpr int maxFilterOutputFrames = 16; // qsv_pipeline_ctrl.h uses RGYFrameInfo *outInfo[16].
    int nOut = 0;
    while (!m_emitQueue.empty() && nOut < maxFilterOutputFrames) {
        const IvtcEmitEntry e = m_emitQueue.front();
        m_emitQueue.pop_front();
        RGYFrameInfo *pOut = &m_frameBuf[e.stagingIdx]->frame;
        pOut->timestamp    = e.timestamp;
        pOut->duration     = e.duration;
        pOut->inputFrameId = e.inputFrameId;
        pOut->picstruct    = RGY_PICSTRUCT_FRAME;
        // Final belt-and-suspenders guard: encoder (qsv_pipeline_ctrl.h:2297)
        // aborts on inputFrameId < 0. Every upstream step should have sanitized
        // this already (processInputToCycle stores from a cache slot that's
        // sanitized at input time), but any defect above here would leak a -1
        // through. Clamp to a synthesized monotonic id on the very last path.
        if (pOut->inputFrameId < 0) pOut->inputFrameId = m_outputFrameCount;
        ppOutputFrames[nOut++] = pOut;
        m_outputFrameCount++;
    }
    *pOutputFrameNum = nOut;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterIvtc::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);

    // DIAG #4: fire once on the first run_filter call. Uses
    // m_pLog->write directly. All numeric args pre-extracted to local
    // int so the va_list boundary sees no type surprises.
    if (m_inputCount == 0 && m_outputFrameCount == 0 && m_pLog) {
        const int actVal      = (int)m_expandActive;
        const int cycleVal    = prm->ivtc.cycle;
        const int dropVal     = prm->ivtc.drop;
        const int cycleLenVal = cycleLen;
        const int filledVal   = m_cycleFilled;
        const int dispCountVal= m_displayFrameCount;
        const int nextIdxVal  = m_nextDisplayIdx;
        m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
            _T("ivtc DIAG: runtime first-call: m_expandActive=%d cycle=%d drop=%d cycleLen=%d m_cycleFilled=%d m_displayFrameCount=%d m_nextDisplayIdx=%d\n"),
            actVal, cycleVal, dropVal, cycleLenVal, filledVal, dispCountVal, nextIdxVal);
    }

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    // 1. 入力を消費する (ある場合)。cycle=0 なら即時 1 emit、cycle>0 なら cycle が満タン
    //    になったタイミングで flushCycle が emit キューを積む。
    if (hasInput) {
        // Log the input frame ONCE (DEC: + IN: lines) against the current
        // m_inputCount index. pushFrameToRing() is called either once (no
        // expansion / non-RFF frame) or twice (expansion + RFF set on
        // cur); logInputFrame is invoked from within pushFrameToRing
        // with isSynth=false to ensure exactly one log record per real
        // decoder input.
        const int logFrameNum = m_inputCount;

        // =============================================================
        //  RFF EXPANSION -- buffered. We accumulate EXPAND_BUF_SIZE coded
        //  frames along with their per-frame RGY_FRAME_FLAG_RFF /
        //  RFF_TFF / RFF_BFF flags (stamped by libavcodec's parser in
        //  rgy_input_avcodec.cpp:3507-3513). When the buffer fills, we
        //  run the DGDecode vfapidec.cpp:461-499 FrameList state machine
        //  over the buffered flags and emit each scheduled display frame
        //  (normal coded-passthrough OR cross-time synth) into the main
        //  IVTC ring. Buffering 10 frames gives us the random-access
        //  lookahead DGDecode needs to correctly handle unclean 3:2
        //  cadences (e.g. DH2's first GOP with consecutive same-parity
        //  RFFs that drag pending state through several non-RFF frames).
        //
        //  DEC:/IN: logs are written on BUFFER (not flush) so the log
        //  index maps 1:1 to decoder-input order; synth frames emitted
        //  at flush time are isSynth=true to suppress their own DEC
        //  lines.
        //
        //  If expansion is inactive, cur is pushed directly to the main
        //  ring (the 1-in-1-out path).
        // =============================================================
        if (!m_expandActive) {
            auto err0 = pushFrameToRing(pInputFrame, stream, logFrameNum, /*isSynth=*/false);
            if (err0 != RGY_ERR_NONE) return err0;
        } else {
            // Buffer the frame. Copy the pixel data into m_expandBuf[slot]
            // and stash its metadata (flags, picstruct, pts, dur, id) in
            // m_expandBufMeta[slot] for the FrameList-builder to consult.
            const int slot = m_expandBufCount;
            auto cpErr = copyFrameAsync(&m_expandBuf[slot]->frame, pInputFrame, stream);
            if (cpErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("ivtc: expand buffer copy failed (slot=%d): %s\n"),
                    slot, get_err_mes(cpErr));
                return cpErr;
            }
            m_expandBuf[slot]->frame.timestamp    = pInputFrame->timestamp;
            m_expandBuf[slot]->frame.duration     = pInputFrame->duration;
            m_expandBuf[slot]->frame.inputFrameId = pInputFrame->inputFrameId;
            m_expandBuf[slot]->frame.picstruct    = pInputFrame->picstruct;
            m_expandBuf[slot]->frame.flags        = pInputFrame->flags;
            m_expandBufMeta[slot].flags        = pInputFrame->flags;
            m_expandBufMeta[slot].picstruct    = pInputFrame->picstruct;
            m_expandBufMeta[slot].timestamp    = pInputFrame->timestamp;
            m_expandBufMeta[slot].duration     = pInputFrame->duration;
            m_expandBufMeta[slot].inputFrameId = pInputFrame->inputFrameId;
            m_expandBufCount++;

            // Log the real decoder input exactly once, at buffer time.
            logInputFrame(pInputFrame, m_expandDecLogIdx);
            m_expandDecLogIdx++;

            // Flush when full. flushExpandBuffer pushes frames to the
            // main ring AND runs processInputToCycle/flushCycle inline
            // so the cache drains between pushes.
            if (m_expandBufCount >= EXPAND_BUF_SIZE) {
                auto ferr = flushExpandBuffer(stream,
                                                cycleLen, /*isFinal=*/false);
                if (ferr != RGY_ERR_NONE) return ferr;
            }
        }

        // -- Lookahead processing. With expansion active the ring can
        //    advance by 2 per input, so we run the processInputToCycle
        //    loop as many times as the lookahead condition permits
        //    (cycleLen > 0 only; cycleLen == 0 keeps the original 1-in-
        //    1-out contract).
        if (cycleLen == 0) {
            if (m_processedCount + 3 <= m_inputCount) {
                const int center     = m_processedCount;
                const int idx_cur    = center % IVTC_CACHE_SIZE;
                const int idx_next   = (center + 1) % IVTC_CACHE_SIZE;
                const int idx_next2  = (center + 2) % IVTC_CACHE_SIZE;
                const int idx_prev   = (center >= 1) ? (center - 1) % IVTC_CACHE_SIZE : idx_cur;
                const int idx_prev2  = (center >= 2) ? (center - 2) % IVTC_CACHE_SIZE : idx_prev;
                auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2, center, stream);
                if (err != RGY_ERR_NONE) return err;
                m_processedCount++;
                // decimation オフ: 1 in 1 out。m_frameBuf[0] がこの call の出力。
                if (m_frameBuf[0]->frame.inputFrameId < 0) m_frameBuf[0]->frame.inputFrameId = m_outputFrameCount;
                ppOutputFrames[0] = &m_frameBuf[0]->frame;
                *pOutputFrameNum = 1;
                return RGY_ERR_NONE;
            }
        } else {
            while (m_processedCount + 3 <= m_inputCount) {
                const int center     = m_processedCount;
                const int idx_cur    = center % IVTC_CACHE_SIZE;
                const int idx_next   = (center + 1) % IVTC_CACHE_SIZE;
                const int idx_next2  = (center + 2) % IVTC_CACHE_SIZE;
                const int idx_prev   = (center >= 1) ? (center - 1) % IVTC_CACHE_SIZE : idx_cur;
                const int idx_prev2  = (center >= 2) ? (center - 2) % IVTC_CACHE_SIZE : idx_prev;
                if (m_mixedActive) {
                    auto *cur = &m_cacheFrames[idx_cur]->frame;
                    const int64_t curPts = cur->timestamp;
                    const int64_t curDur = cur->duration;
                    int64_t discontinuityLimit = 0;
                    if (prm->timebase.n() > 0 && prm->timebase.d() > 0) {
                        discontinuityLimit = rational_rescale(2, rgy_rational<int>(1001, 30000), prm->timebase);
                    }
                    if (discontinuityLimit <= 0) discontinuityLimit = (m_mixedLastInputDur > 0) ? m_mixedLastInputDur * 2 : 2;
                    if (m_mixedLastInputValid && curPts != AV_NOPTS_VALUE && m_mixedLastInputPts != AV_NOPTS_VALUE) {
                        const int64_t diff = curPts - m_mixedLastInputPts;
                        if (diff <= 0 || diff > discontinuityLimit) {
                            auto ferr = partialFlushMixed(stream, curPts);
                            if (ferr != RGY_ERR_NONE) return ferr;
                            m_lastSceneChange = false;
                            m_lastSceneSAD = 0;
                            resetMixedTemporalState();
                            resetMixedRffState();
                        }
                    }
                    const auto section = ivtcMixedClassify(cur);
                    if (section == IvtcMixedSection::Interlaced) {
                        resetMixedRffState();
                        auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2, center, stream);
                        if (err != RGY_ERR_NONE) return err;
                        if (m_cycleFilled >= cycleLen) {
                            auto ferr = flushCycleMixed(false, m_cacheFrames[idx_next]->frame.timestamp, true, stream);
                            if (ferr != RGY_ERR_NONE) return ferr;
                        }
                    } else if (section == IvtcMixedSection::Rff || m_mixedRffPendingTopValid || m_mixedRffPendingBottomValid) {
                        if (m_cycleFilled > 0) {
                            auto ferr = partialFlushMixed(stream, curPts);
                            if (ferr != RGY_ERR_NONE) return ferr;
                        }
                        auto err = enqueueMixedRffFrame(idx_cur, m_cacheFrames[idx_next]->frame.timestamp, stream);
                        if (err != RGY_ERR_NONE) return err;
                    } else {
                        if (m_cycleFilled > 0) {
                            auto ferr = partialFlushMixed(stream, curPts);
                            if (ferr != RGY_ERR_NONE) return ferr;
                        }
                        auto err = enqueueMixedPassthrough(cur, idx_cur, m_cacheFrames[idx_next]->frame.timestamp, stream);
                        if (err != RGY_ERR_NONE) return err;
                    }
                    m_mixedLastInputPts = curPts;
                    if (curDur > 0) m_mixedLastInputDur = curDur;
                    m_mixedLastInputValid = true;
                } else {
                    auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2, center, stream);
                    if (err != RGY_ERR_NONE) return err;
                    if (m_cycleFilled >= cycleLen) {
                        auto ferr = flushCycle(false, pInputFrame->timestamp, stream);
                        if (ferr != RGY_ERR_NONE) return ferr;
                    }
                }
                m_processedCount++;
            }
        }
        // cycleLen > 0 の通常入力時はここで 1 popEmit (キューにあれば) する。
        // 入力がキュー生成をトリガした (flushCycle) ケースも、1 call 1 emit なので 1 枚だけ取り出す。
        return popEmit(ppOutputFrames, pOutputFrameNum);
    }

    // 2. EOS drain. Before running the processInputToCycle tail drain,
    //    flush any still-buffered expansion frames + emit any remaining
    //    schedule entries that reference the final buffer (isFinal=true
    //    drops the trailing half-open schedule slot per DGDecode's
    //    "fringe slot past ntsc is not emitted" behaviour).
    if (m_expandActive && m_expandBufCount > 0) {
        auto ferr = flushExpandBuffer(stream, cycleLen, /*isFinal=*/true);
        if (ferr != RGY_ERR_NONE) return ferr;
        // If the flush pushed frames to the ring, fall through to the
        // standard drain path -- it'll run processInputToCycle / flushCycle
        // on the trailing 3 frames that the flush's inline drain left behind.
    }

    // ドレイン: 未処理フレームを 1 call 1 frame ずつ処理 (末尾 2 フレーム分)。
    //   5-frame lookahead では末尾 2 フレームが normal path 中には処理されていない。
    //   cur に対して next/next2 が欠ける場合、idx_next/idx_next2 を idx_cur に
    //   エイリアスして processInputToCycle に渡す (末尾フィールドの dup 相当)。
    if (m_processedCount < m_inputCount) {
        const int center     = m_processedCount;
        const int idx_cur    = center % IVTC_CACHE_SIZE;
        const int idx_next   = (center + 1 < m_inputCount) ? (center + 1) % IVTC_CACHE_SIZE : idx_cur;
        const int idx_next2  = (center + 2 < m_inputCount) ? (center + 2) % IVTC_CACHE_SIZE : idx_next;
        const int idx_prev   = (center >= 1) ? (center - 1) % IVTC_CACHE_SIZE : idx_cur;
        const int idx_prev2  = (center >= 2) ? (center - 2) % IVTC_CACHE_SIZE : idx_prev;

        if (m_mixedActive) {
            auto *cur = &m_cacheFrames[idx_cur]->frame;
            const auto section = ivtcMixedClassify(cur);
            if (section == IvtcMixedSection::Interlaced) {
                resetMixedRffState();
                auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2, center, stream);
                if (err != RGY_ERR_NONE) return err;
            } else if (section == IvtcMixedSection::Rff || m_mixedRffPendingTopValid || m_mixedRffPendingBottomValid) {
                if (m_cycleFilled > 0) {
                    auto ferr = partialFlushMixed(stream, cur->timestamp);
                    if (ferr != RGY_ERR_NONE) return ferr;
                }
                const int64_t nextPts = (idx_next != idx_cur) ? m_cacheFrames[idx_next]->frame.timestamp : AV_NOPTS_VALUE;
                auto err = enqueueMixedRffFrame(idx_cur, nextPts, stream);
                if (err != RGY_ERR_NONE) return err;
            } else {
                if (m_cycleFilled > 0) {
                    auto ferr = partialFlushMixed(stream, cur->timestamp);
                    if (ferr != RGY_ERR_NONE) return ferr;
                }
                const int64_t nextPts = (idx_next != idx_cur) ? m_cacheFrames[idx_next]->frame.timestamp : AV_NOPTS_VALUE;
                auto err = enqueueMixedPassthrough(cur, idx_cur, nextPts, stream);
                if (err != RGY_ERR_NONE) return err;
            }
            if (cur->timestamp != AV_NOPTS_VALUE) m_mixedLastInputPts = cur->timestamp;
            if (cur->duration > 0) m_mixedLastInputDur = cur->duration;
            m_mixedLastInputValid = true;
        } else {
            auto err = processInputToCycle(idx_prev2, idx_prev, idx_cur, idx_next, idx_next2, center, stream);
            if (err != RGY_ERR_NONE) return err;
        }
        m_processedCount++;

        if (cycleLen == 0) {
            // Drain cycle=0 emit: same belt-and-suspenders as the normal
            // path above — clamp inputFrameId so the encoder's non-negative
            // check always passes.
            if (m_frameBuf[0]->frame.inputFrameId < 0) m_frameBuf[0]->frame.inputFrameId = m_outputFrameCount;
            ppOutputFrames[0] = &m_frameBuf[0]->frame;
            *pOutputFrameNum = 1;
            return RGY_ERR_NONE;
        }

        // ドレイン時は nextInputPts なし。途中で cycle が満タンなら full flush、
        // 最終 pass で partial なら finalFlush=true。どちらも emit キューに積む。
        const bool noMoreDrain   = (m_processedCount >= m_inputCount);
        const bool cycleJustFull = (m_cycleFilled >= cycleLen);
        const bool finalFlush    = noMoreDrain && (m_cycleFilled > 0) && (m_cycleFilled < cycleLen);
        if (cycleJustFull || finalFlush) {
            auto ferr = m_mixedActive
                ? flushCycleMixed(finalFlush, AV_NOPTS_VALUE, !finalFlush, stream)
                : flushCycle(finalFlush, AV_NOPTS_VALUE, stream);
            if (ferr != RGY_ERR_NONE) {
                return ferr;
            }
        }
        return popEmit(ppOutputFrames, pOutputFrameNum);
    }

    // 3. drain 済み: キューに残りがあれば 1 枚返す、空なら 0 emit。
    return popEmit(ppOutputFrames, pOutputFrameNum);
}

void NVEncFilterIvtc::close() {
    // RFF expansion stats + teardown. The pre-scan gives us the
    // authoritative expansion ratio up front; the close() report
    // confirms what actually made it to the ring at runtime.
    if (m_expandActive) {
        const long long totalIn  = m_expandCodedEmitCount;
        const long long totalOut = m_expandCodedEmitCount + m_expandSynthCount;
        const double ratio = (totalIn > 0) ? (double)totalOut / (double)totalIn : 0.0;
        AddMessage(RGY_LOG_INFO,
            _T("ivtc: expansion: %lld synth frames, %lld coded frames, %lld total RFF inputs\n"),
            (long long)m_expandSynthCount,
            (long long)m_expandCodedEmitCount,
            (long long)m_expandRffSeen);
        AddMessage(RGY_LOG_INFO,
            _T("ivtc: expansion ratio: %.3fx\n"), ratio);
    }

    // INSTRUMENTATION summary: per-gate blend classification.
    // Counters tallied at the applyBlendGate site in processInputToCycle.
    // Codes: 1=mislabeledBySat 2=mislabeledByDual 3=unknownCombed
    //        4=progressiveCombed 5=strongMatch 6=vthresh_vetoed
    //
    // 2026-04-24 HEAP-CRASH REMEDIATION: converted from AddMessage() to
    // m_pLog->write() direct calls, following the PATCH 1 playbook. The
    // AddMessage variadic wrapper (_vsctprintf + _vstprintf_s) had a
    // documented history of emitting silent corruption under certain
    // argument patterns — notably the STATUS_STACK_BUFFER_OVERRUN crash
    // during pre-scan summary. All numeric args pre-extracted to strict
    // const int/uint64_t/double locals so no cast expressions evaluate
    // inside the varargs promotion.
    {
        const uint64_t triggerSat    = m_blendTriggerCounts[1];
        const uint64_t triggerDual   = m_blendTriggerCounts[2];
        const uint64_t triggerUnk    = m_blendTriggerCounts[3];
        const uint64_t triggerProg   = m_blendTriggerCounts[4];
        const uint64_t triggerStrong = m_blendTriggerCounts[5];
        const uint64_t triggerVeto   = m_blendTriggerCounts[6];
        const uint64_t triggerConf   = m_blendTriggerCounts[7];   // SUB-PHASE 5: confidence-forced
        const uint64_t triggerNone   = m_blendTriggerCounts[0];
        const uint64_t totalBlend    = triggerSat + triggerDual + triggerUnk + triggerProg + triggerStrong + triggerConf;
        if (m_pLog) {
            m_pLog->write(RGY_LOG_INFO, RGY_LOGT_VPP,
                _T("ivtc blend-gate breakdown: total=%llu  mislabeledBySat=%llu  mislabeledByDual=%llu  unknownCombed=%llu  progressiveCombed=%llu  strongMatch=%llu  confidenceForced=%llu  (vthresh_vetoed=%llu, no_blend=%llu)\n"),
                (unsigned long long)totalBlend,
                (unsigned long long)triggerSat,
                (unsigned long long)triggerDual,
                (unsigned long long)triggerUnk,
                (unsigned long long)triggerProg,
                (unsigned long long)triggerStrong,
                (unsigned long long)triggerConf,
                (unsigned long long)triggerVeto,
                (unsigned long long)triggerNone);
            // The "blend-gate share: ..." percentage line was removed
            // 2026-04-24 after two separate format-corruption fixes both
            // failed. The raw counts in the breakdown line above provide
            // the same information; percentages can be computed offline
            // from the counts when needed.
        }
    }
    m_blendTriggerCounts.fill(0);
    m_expandActive          = false;
    m_skipBaseFpsMultiplier = false;
    m_displayFrameList.clear();
    m_displayFrameCount     = 0;
    m_nextDisplayIdx        = 0;
    m_expandBufCount        = 0;
    m_expandBufBase         = 0;
    m_expandCarryValid      = false;
    m_expandCarryCodedIdx   = -1;
    m_expandDecLogIdx       = 0;
    for (auto &buf : m_expandBuf) buf.reset();
    m_expandCarryFrame.reset();
    m_expandSynth.reset();
    m_expandSynthCount      = 0;
    m_expandCodedEmitCount  = 0;
    m_expandRffSeen         = 0;

    m_cacheFrames.clear();
    m_scoreBuf.reset();
    m_scoreHost.clear();
    m_diffBuf.reset();
    m_diffHost.clear();
    m_cycleInPts.clear();
    m_cycleInDur.clear();
    m_cycleInputIds.clear();
    m_cycleMatchScore.clear();
    m_cycleCombScore.clear();
    m_cycleCombMax.clear();
    m_cycleCombBlocks.clear();
    m_cycleCombScorePrim.clear();
    m_cycleCombMaxPrim.clear();
    m_cycleCombBlocksPrim.clear();
    m_cycleCombScoreAlt.clear();
    m_cycleCombMaxAlt.clear();
    m_cycleCombBlocksAlt.clear();
    m_cycleMatchAltParity.clear();
    m_cycleConfidence.clear();
    m_cycleBlendTrigger.clear();
    m_cycleMatchType.clear();
    m_cycleApplyBlend.clear();
    m_cycleDecTag.clear();
    m_cycleCadenceTag.clear();
    m_cycleDiffPrev.clear();
    m_cycleSceneSAD.clear();
    m_cycleIsSynth.clear();
    m_emitQueue.clear();
    m_stagingBase = 0;
    m_mixedDirectStagingBase = 0;
    m_mixedDirectStagingCount = 0;
    m_mixedDirectStagingNext = 0;
    m_mixedActive = false;
    resetMixedTemporalState();
    resetMixedRffState();
    m_mixedRffPendingTopFrame.reset();
    m_mixedRffPendingBottomFrame.reset();
    m_nPts = 0;
    m_nPtsInit = false;
    m_cfrBaseDur = 0;
    m_cfrEmitIdx = 0;
    m_hasSaveSlot = false;
    m_cycleFilled = 0;
    m_outputFrameCount = 0;
    m_inputCount = 0;
    m_processedCount = 0;
    m_lastSceneChange = false;
    m_lastSceneSAD = 0;
    resetCadenceState();
    m_fpLog.reset();
    m_frameBuf.clear();
}
