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
#include <array>
#include <deque>
#include <memory>
#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamIvtc : public NVEncFilterParam {
public:
    VppIvtc ivtc;
    rgy_rational<int> timebase;  // output stream timebase; used by flushCycle for drift-free CFR emit pts via rational_rescale
    bool inputBPulldownDetected; // upstream demuxer's bPulldown result from
                                 // rgy_input_avcodec.cpp:937. Read by IVTC init
                                 // to resolve expand=auto. Defaults to false
                                 // until qsv_pipeline populates it; users can
                                 // force-enable via --vpp-ivtc expand=on.
    bool inputIsAvcodecReader;   // true when input reader preserves avcodec
                                 // per-frame RFF/picstruct metadata.
    tstring inputFilePath;       // absolute path to input media file. Consumed by
                                 // the RFF-expansion pre-scan (ivtcPreScanInput)
                                 // at init time to build the complete DGDecode
                                 // FrameList schedule from every coded frame's
                                 // RFF/TFF/picture_structure flags. Not required
                                 // when expand is off.
    int  trimOffset;             // m_trimParam.offset from the avcodec reader:
                                 // number of coded frames the decoder discards
                                 // before the first keyframe output. The pre-scan
                                 // sees ALL coded frames (from the first packet),
                                 // so we use this to skip the leading entries that
                                 // IVTC will never see at runtime.
    int  trimFrameCount;         // total number of coded frames IVTC expects to
                                 // receive at runtime (post-trim). Used to bound
                                 // the schedule and compute the expansion ratio.
                                 // 0 means "scan to end".

    NVEncFilterParamIvtc() : ivtc(), timebase(), inputBPulldownDetected(false), inputIsAvcodecReader(false),
                             inputFilePath(), trimOffset(0), trimFrameCount(0) {};
    virtual ~NVEncFilterParamIvtc() {};
    virtual tstring print() const override { return ivtc.print(); };
};

enum class IvtcMatch : int {
    C = 0, // 現在フレームをそのまま使用
    P = 1, // TFF: [cur.top, prev.bot]  match-with-previous (bot field borrowed from prev)
    N = 2, // TFF: [next.top, cur.bot]  match-with-next (top field borrowed from next)
};

// One display-order slot in the RFF-expansion schedule. topSource and
// bottomSource are 0-based POST-TRIM coded-frame indices (i.e. indexed
// against the frames that run_filter will receive at runtime, which is
// what m_expandBufBase counts against). For a normal coded passthrough
// topSource == bottomSource. For a synthesised cross-time slot (top
// field from one coded frame, bottom field from another) topSource !=
// bottomSource and isSynth is true.
struct IvtcDisplayFrame {
    int  topSource;
    int  bottomSource;
    bool isSynth;
};

// Per-frame bitstream metadata captured by the pre-scan helper.
// Populated by ivtcPreScanInput() from the libavcodec parser; consumed
// by NVEncFilterIvtc::buildScheduleFromScan() to drive the DGDecode
// vfapidec.cpp:461-499 FrameList state machine.
struct IvtcPreScanFrame {
    bool rff;          // repeat_first_field (from parser->repeat_pict > 1)
    bool tff;          // true if top field displayed first
    bool progressive;  // parser->picture_structure is FRAME (not FIELD)
};

// 出力キューに積む単位。m_frameBuf[stagingIdx] の CLFrame を指し、
// 下流に渡す際のメタデータを個別に保持する。
struct IvtcEmitEntry {
    int stagingIdx;
    int64_t timestamp;
    int64_t duration;
    int inputFrameId;
};

enum class IvtcMixedSection : uint8_t {
    Passthrough = 0,
    Rff = 1,
    Interlaced = 2,
};


RGY_ERR run_ivtc_score_candidates(const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int nt, int T, int y0, int y1, uint32_t *scoreDev, cudaStream_t stream);
RGY_ERR run_ivtc_synthesize_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, int tff, int match, int applyBlend, int dthresh, cudaStream_t stream);
RGY_ERR run_ivtc_bwdif_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrev2, const RGYFrameInfo *pPrev, const RGYFrameInfo *pCur, const RGYFrameInfo *pNext, const RGYFrameInfo *pNext2, int tff, int sceneChange, int dthresh, cudaStream_t stream);
RGY_ERR run_ivtc_frame_diff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint32_t *diffDev, cudaStream_t stream);
RGY_ERR run_ivtc_field_overlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int tff, cudaStream_t stream);

class NVEncFilterIvtc : public NVEncFilter {
public:
    NVEncFilterIvtc();
    virtual ~NVEncFilterIvtc();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamIvtc> pParam);

    // Ring-buffer insertion helper (shared by normal-input path and RFF
    // expansion synth-frame path). Copies pFrame into m_cacheFrames[slot],
    // updates metadata, clamps negative inputFrameId, optionally writes the
    // DEC:/IN: log lines, and increments m_inputCount. Must be called under
    // the same CUDA stream as the caller would have used.
    // isSynth: legacy name -- when true, suppresses the DEC:/IN: log
    //   line for this push. Set to true for all schedule-driven pushes
    //   from flushExpandBuffer (both coded passthrough and cross-time
    //   synths) because the real decoder-input logging already happened
    //   at BUFFER time in run_filter's expansion branch.
    // isSynthFrame: when true, marks m_cacheIsSynth[slot]=true so the
    //   scoring pipeline in processInputToCycle knows to bypass
    //   score/match/cadence/blend for this frame. Set to true ONLY for
    //   cross-time synths (schedule entries with topSource != bottomSource).
    //   Defaults to false for backward compat with existing callers.
    RGY_ERR pushFrameToRing(const RGYFrameInfo *pFrame, cudaStream_t stream,
                                                        int logFrameNum, bool isSynth,
                            bool isSynthFrame = false);
    // Overlay one field (alternate rows) from src onto dst via stride*2 BitBlt.
    // Follows DGDecode vfapidec.cpp:1027-1059 CopyBot/CopyTop. tff=1: overlay
    // BOT field (odd rows) from src (for TFF source). tff=0: overlay TOP field
    // (even rows) from src. Applies to all planes with plane-local dimensions.
    RGY_ERR overlayField(CUFrameBuf *dst, const CUFrameBuf *src, int tff,
                         cudaStream_t stream);
    // Flush the 10-frame staging buffer by following the pre-computed
    // m_displayFrameList schedule. Emits each schedule entry whose source
    // coded frames are in the buffer (normal coded-passthrough OR
    // cross-time synth built via overlayField); between pushes the
    // cycle-processing drain loop runs inline so the 5-slot cache never
    // gets overwritten before its data is consumed. A single carry frame
    // caches the last coded frame of the buffer for the (rare) synth
    // whose bottomSource lives in the previous buffer.
    RGY_ERR flushExpandBuffer(cudaStream_t stream, int cycleLen, bool isFinal);
    // Build the complete display-order schedule (m_displayFrameList) from
    // the pre-scan's per-frame RFF/TFF flags. Implements the DGDecode
    // vfapidec.cpp:461-499 FrameList state machine exactly as traced in
    // analysis/dgdecode_trace.txt. Coded indices in the schedule are
    // POST-TRIM (0-based from the first frame run_filter will receive).
    void buildScheduleFromScan(const std::vector<IvtcPreScanFrame> &frames,
                                int trimOffset, int trimFrameCount);
    RGY_ERR scoreCandidates(const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, uint64_t matchScoreOut[3], uint64_t combScoreOut[3], uint64_t combMaxOut[3], uint64_t combBlocksOut[3], const int tffForScoring, cudaStream_t stream);
    RGY_ERR synthesizeToCycle(int cycleSlot, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const IvtcMatch match, const bool applyBlend, const int dthresh, const int tffForSynth, cudaStream_t stream);
    // Full BWDIF deinterlacer for frames selected by the post-processing gate.
    // Fires instead of the simpler synthesize+apply_blend path and uses the full
    // 5-frame temporal window (prev2/prev/cur/next/next2) held by the IVTC
    // ring buffer; startup aliasing (prev2=prev, next2=next) is handled by
    // the caller when the ring hasn't filled.
    // sceneChange: when non-zero, the kernel skips ALL temporal reads and
    // uses spatial-only SP cubic interpolation instead. Set to 1 on the
    // scene-change frame itself AND the frame after, because the latter's
    // prev is still from the outgoing scene and would produce ghosting.
    RGY_ERR synthesizeToCycleBwdif(int cycleSlot, const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2, const int streamTff, const int sceneChange, const int dthresh, cudaStream_t stream);
    RGY_ERR computePairDiff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint64_t &diffOut, cudaStream_t stream);
    // Process one "center" frame with full 5-frame temporal context. centerDisplayIdx
    // is the 0-based input-stream index of the frame at idx_cur (used for
    // per-frame logging). idx_prev2/idx_prev may alias to idx_cur at stream start;
    // idx_next2/idx_next may alias to idx_cur during drain.
    RGY_ERR processInputToCycle(int idx_prev2, int idx_prev, int idx_cur, int idx_next, int idx_next2, int centerDisplayIdx, cudaStream_t stream);
    RGY_ERR flushCycle(bool finalFlush, int64_t nextInputPts, cudaStream_t stream);
    RGY_ERR enqueueMixedPassthrough(const RGYFrameInfo *frame, int cacheIdx, int64_t nextPts, cudaStream_t stream);
    RGY_ERR appendMixedRffDisplayFrame(const CUFrameBuf *topFrame, const RGYFrameInfo *topInfo, const CUFrameBuf *bottomFrame, const RGYFrameInfo *bottomInfo, int decTag, cudaStream_t stream);
    RGY_ERR enqueueMixedDirectFrame(const CUFrameBuf *srcFrame, const RGYFrameInfo *srcInfo, const char *decTag, const char *section, cudaStream_t stream);
    RGY_ERR pushMixedEmitEntry(int stagingIdx, const RGYFrameInfo *srcInfo);
    RGY_ERR setMixedRffPending(const CUFrameBuf *srcFrame, const RGYFrameInfo *srcInfo, bool pendingTop, cudaStream_t stream);
    RGY_ERR enqueueMixedRffFrame(int cacheIdx, int64_t nextPts, cudaStream_t stream);
    RGY_ERR flushCycleMixed(bool finalFlush, int64_t cycleEndPts, bool allowDrop, cudaStream_t stream);
    RGY_ERR partialFlushMixed(cudaStream_t stream, int64_t cycleEndPts);
    void resetMixedTemporalState();
    void resetMixedRffState();
    RGY_ERR popEmit(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);

    // Record a match decision, return cadence-predicted match if pattern is locked
    // (else -1). Also updates the internal 5-entry ring buffer and confidence counter.
    int  updateCadence(int observedMatch);
    // Reset cadence + hysteresis state (on scene change or re-init).
    void resetCadenceState();
    // Emit the per-input DEC: + IN: TSV lines to m_fpLog (if enabled).
    // Called from run_filter on every input frame including pass-through / drain.
    void logInputFrame(const RGYFrameInfo *pInputFrame, int frameNum);

        std::vector<std::unique_ptr<CUFrameBuf>> m_cacheFrames; // リングバッファ: prev/cur/next の3枚
    std::unique_ptr<CUMemBuf> m_scoreBuf;                   // WGごとのスコア集計バッファ
    std::vector<uint32_t> m_scoreHost;                      // ホスト側リードバック用
    std::unique_ptr<CUMemBuf> m_diffBuf;                    // WGごとの SAD 集計バッファ (reused)
    std::vector<uint32_t> m_diffHost;                       // ホスト側リードバック用
    std::vector<int64_t> m_cycleInPts;                      // サイクル内フレームの入力タイムスタンプ
    std::vector<int64_t> m_cycleInDur;                      // サイクル内フレームの入力 duration
    std::vector<int> m_cycleInputIds;                       // サイクル内フレームの inputFrameId (ログ用)
    std::vector<uint64_t> m_cycleMatchScore;                // 選択マッチの match-quality (ログ用)
    std::vector<uint64_t> m_cycleCombScore;                 // 選択マッチの combing-count (ログ用)
    std::vector<uint64_t> m_cycleCombMax;                   // PRIORITY 1: TFM-style block-MAX across WGs (ログ用; Priority 2 will wire into blend gate)
    std::vector<uint64_t> m_cycleCombBlocks;                // SUB-PHASE 1: count of combed blocks (cX sum >= BLOCK_COMB_THRESH) per chosen candidate (ログ用; SUB-PHASE 3 will wire into selection)

    // SUB-PHASE 2 (2026-04-24) — DUAL-PARITY DIAGNOSTIC.
    // Per-slot full triplets (C,P,N) of cComb / cCombMax / cCombBlocks for
    // BOTH the primary parity (matches m_tffFixed) and the alternate parity
    // (!m_tffFixed). Logged as 18 new TSV columns at flushCycle. Selection
    // logic still uses the primary triplets only — this is diagnostic-only
    // until SUB-PHASE 3 wires comb-first selection across the full 6-set.
    std::vector<std::array<uint64_t, 3>> m_cycleCombScorePrim;   // [c, p, n]   primary tff
    std::vector<std::array<uint64_t, 3>> m_cycleCombMaxPrim;
    std::vector<std::array<uint64_t, 3>> m_cycleCombBlocksPrim;
    std::vector<std::array<uint64_t, 3>> m_cycleCombScoreAlt;    // [cA, pA, nA] alternate tff
    std::vector<std::array<uint64_t, 3>> m_cycleCombMaxAlt;
    std::vector<std::array<uint64_t, 3>> m_cycleCombBlocksAlt;
    // SUB-PHASE 3 (2026-04-24): per-slot flag indicating whether the
    // comb-first 6-candidate selection chose an alt-parity (!tff)
    // candidate over the best primary-parity one. 0 = primary tff,
    // 1 = alt tff. Logged as matchParity TSV column.
    std::vector<uint8_t> m_cycleMatchAltParity;
    // SUB-PHASE 5 (2026-04-25): per-slot confidence level emitted by the
    // confidence-based decision layer. 0=HIGH, 1=MEDIUM, 2=LOW, 3=VERY_LOW.
    // Logged as confidence TSV column. See analysis/Confidence-Based_Matcher_TFM-style.txt.
    std::vector<uint8_t> m_cycleConfidence;
    std::vector<uint8_t>  m_cycleBlendTrigger;              // INSTRUMENTATION: which blend gate caused applyBlend=true for this slot. 0=none, 1=mislabeledBySat, 2=mislabeledByDual, 3=unknownCombed, 4=progressiveCombed, 5=strongMatch, 6=vthresh_vetoed, 7=confidenceForced
    std::array<uint64_t, 8> m_blendTriggerCounts;           // cumulative counter across the run; reported at close()
    std::vector<int> m_cycleMatchType;                      // 選択マッチ種別 (0=C, 1=P, 2=N) ログ用
    std::vector<int> m_cycleApplyBlend;                     // post=2 の blend 適用フラグ (ログ用)
    std::vector<int> m_cycleDecTag;                         // Decoder-driven routing decision per slot (see encoding in processInputToCycle)
    std::vector<int> m_cycleCadenceTag;                     // Cadence tracker state + override action per slot (see encoding in processInputToCycle)
    std::vector<uint64_t> m_cycleDiffPrev;                  // SAD(cycle[i], cycle[i-1]) for i>=1; [0] uses m_saveSlot
    std::vector<uint64_t> m_cycleSceneSAD;                  // scene-change detector SAD per slot (ログ用)
    // Per-cycle-slot synth flag (FIX B, 2026-04-24). Mirrors m_cacheIsSynth
    // at the moment processInputToCycle fills a cycle slot. Consumed by
    // flushCycle's decimation-drop selection to prefer dropping synth
    // frames over coded frames — synths are the expansion-added
    // duplicates, so dropping them first preserves as many coded frames
    // as possible in the decimated output. See analysis/vpp_ivtc_progress.txt.
    std::vector<uint8_t>  m_cycleIsSynth;                   // 1=synth (from expand=on), 0=coded; indexed like m_cycleDiffPrev
    std::deque<IvtcEmitEntry> m_emitQueue;                  // 1 call 1 emit 用の出力キュー (AFS/Decimate 方式)
    int m_stagingBase;                                      // m_frameBuf のうち emit-staging 領域の先頭 index
    int m_mixedDirectStagingBase;                           // mixed direct emit staging ring
    int m_mixedDirectStagingCount;
    int m_mixedDirectStagingNext;
    bool m_mixedActive;                                     // resolved mixed=on state
    int64_t m_mixedLastInputPts;
    int64_t m_mixedLastInputDur;
    int64_t m_mixedLastEmitEndPts;
    bool m_mixedLastInputValid;
    std::unique_ptr<CUFrameBuf> m_mixedRffPendingTopFrame;
    std::unique_ptr<CUFrameBuf> m_mixedRffPendingBottomFrame;
    RGYFrameInfo m_mixedRffPendingTopInfo;
    RGYFrameInfo m_mixedRffPendingBottomInfo;
    bool m_mixedRffPendingTopValid;
    bool m_mixedRffPendingBottomValid;
    int64_t m_nPts;                                         // 出力 PTS seed (first input pts after init; constant thereafter)
    bool m_nPtsInit;                                        // m_nPts has been seeded
    int64_t m_cfrBaseDur;                                   // reference duration per emit (kept for --log and drain display; live emit math uses rational_rescale)
    int64_t m_cfrEmitIdx;                                   // running global emit index; pts_N = seed + rescale(N, 1/baseFps, timebase). Drift-free.
    bool m_hasSaveSlot;                                     // 前サイクル末尾を m_frameBuf[cycle] に保存済みか
    int m_cycleFilled;                                      // 現サイクルに蓄積されたフレーム数
    int m_outputFrameCount;                                 // 出力済みフレーム数 (ログ用)
    int m_inputCount;                                       // 入力フレーム数の累計
    int m_processedCount;                                   // processInputToCycle 経由で処理済みフレーム数 (0-based next-to-process index)
    bool m_lastSceneChange;                                 // true if the previous processed frame triggered scene-change SAD
    uint64_t m_lastSceneSAD;                                // previous frame's cur-vs-prev SAD (used for adaptive threshold)
    int m_tffFixed;                                         // -1: 入力picstruct由来で自動, 0: BFF, 1: TFF

    // Per-cache-slot synth flag. True when the frame in m_cacheFrames[slot]
    // was assembled by flushExpandBuffer() via kernel_ivtc_field_overlay()
    // (i.e. a cross-time synth combining fields from two different coded
    // pictures). Synth frames are ALREADY correctly assembled and must
    // NOT be re-scored by the field-matcher -- running them through
    // scoreCandidates would always find match=c with massive combing
    // (their two fields come from different film times), poisoning the
    // cadence tracker and triggering unnecessary blends.
    // Set by pushFrameToRing() from its isSynthFrame parameter; consumed
    // by processInputToCycle() to bypass score/match/cadence/blend when
    // the center slot holds a synth. Synths still participate in cycle
    // decimation.
    std::array<bool, 5> m_cacheIsSynth;                     // indexed by m_cacheFrames slot (IVTC_CACHE_SIZE=5)

    // Cadence + hysteresis state (new 2026-04-21):
    static constexpr int IVTC_CADENCE_LEN = 5;
    int                            m_lastMatch;                 // previous frame's chosen match (-1 = none)
    uint64_t                       m_lastMatchScore;            // previous frame's chosen-match raw score (for integer hysteresis)
    std::array<int, IVTC_CADENCE_LEN> m_cadenceHistory;         // ring buffer of the last 5 match decisions
    int                            m_cadenceFill;               // how many entries currently in history (0..5)
    int                            m_cadenceIndex;              // circular write position
    int                            m_cadenceLockedPhase;        // -1 = unlocked; else 0..4
    int                            m_cadenceConfidence;         // weighted correct/wrong ledger (+2/-1, clamped to [0, 255])
    int                            m_cadenceLastPrediction;     // prediction made LAST call; -1 = none (verify against THIS obs)

    // RFF expansion state (libavcodec pre-scan driven).
    //
    // At init time we run a separate libavcodec pre-scan
    // (ivtcPreScanInput) over the entire input file, extract per-frame
    // RFF/TFF/picture_type flags, then run DGDecode's vfapidec.cpp
    // :461-499 FrameList state machine over the complete flag sequence
    // to produce m_displayFrameList -- the authoritative display-order
    // schedule. The pre-scan uses the same parser libavcodec will use
    // at decode time (PARSER_FLAG_COMPLETE_FRAMES), so the flag
    // alignment with the frames the IVTC filter receives at runtime is
    // guaranteed.
    //
    // At runtime each coded frame is copied into a 10-slot staging
    // buffer; flushExpandBuffer walks the pre-computed schedule and
    // emits each entry whose source coded indices are present in the
    // buffer (plus an optional carry frame from the previous flush).
    // See analysis/dgdecode_trace.txt and analysis/prescan_expansion.txt.
    static constexpr int EXPAND_BUF_SIZE = 10;
    struct ExpandBufMeta {
        uint32_t flags;         // RGY_FRAME_FLAG_* bitmask captured at buffer time
        RGY_PICSTRUCT picstruct;
        int64_t  timestamp;
        int64_t  duration;
        int      inputFrameId;
    };
    bool                                       m_expandActive;           // resolved from prm->ivtc.expand + scan RFF count
    bool                                       m_skipBaseFpsMultiplier;  // expand=on flag; retained for diagnostic logging. Post-FIX-A (2026-04-24) this no longer gates the baseFps decimation multiplier — the multiplier always fires when cycle>0 drop>0.
    std::vector<IvtcDisplayFrame>              m_displayFrameList;       // complete schedule built by buildScheduleFromScan
    int                                        m_displayFrameCount;      // == (int)m_displayFrameList.size()
    int                                        m_nextDisplayIdx;         // cursor into m_displayFrameList (0..m_displayFrameCount)
    std::array<std::unique_ptr<CUFrameBuf>, EXPAND_BUF_SIZE> m_expandBuf;        // 10-slot buffer of recent coded frames
    std::array<ExpandBufMeta, EXPAND_BUF_SIZE>              m_expandBufMeta;    // per-slot decoder metadata (flags/pts/dur/id)
    int                                        m_expandBufCount;         // number of buffered frames (0..EXPAND_BUF_SIZE)
    int                                        m_expandBufBase;          // 0-based POST-TRIM coded index of m_expandBuf[0]
    std::unique_ptr<CUFrameBuf>                m_expandCarryFrame;       // last coded frame of previous buffer, kept for boundary-spanning synths
    ExpandBufMeta                              m_expandCarryMeta;        // metadata matching m_expandCarryFrame
    bool                                       m_expandCarryValid;       // m_expandCarryFrame holds a frame (true after first flush)
    int                                        m_expandCarryCodedIdx;    // POST-TRIM coded index of m_expandCarryFrame (-1 when invalid)
    std::unique_ptr<CUFrameBuf>                m_expandSynth;            // scratch buffer for a synthesised display frame
    int                                        m_expandDecLogIdx;        // monotonic decoder-input counter used for DEC/IN log lines
    int64_t                                    m_expandSynthCount;       // diagnostic: total synth frames emitted
    int64_t                                    m_expandCodedEmitCount;   // diagnostic: total coded frames passed through unchanged
    int64_t                                    m_expandRffSeen;          // diagnostic: coded frames with RFF flag observed in the pre-scan

    // Pre-scan summary (CHANGE 2 + CHANGE 3, 2026-04-24). Computed after
    // ivtcPreScanInput returns, consumed by the cycle auto-resolve and
    // the pure-soft-telecine auto-disable checks in init().
    double                                     m_prescanRffRatio;        // rff_frames / total_frames over the pre-scan
    int                                        m_prescanInterlacedCount; // count of f.progressive==false across the pre-scan
    int                                        m_prescanRffTffCount;     // RFF frames with top-field-first
    int                                        m_prescanRffBffCount;     // RFF frames with bottom-field-first
    int                                        m_prescanTotalFrames;     // size of the pre-scan frame vector

    unique_ptr<FILE, fp_deleter> m_fpLog;
};
