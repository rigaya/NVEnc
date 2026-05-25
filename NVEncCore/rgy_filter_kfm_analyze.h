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
#include <string>
#include <vector>

namespace RGYKFM {

enum {
    OVERLAP = 8,
    VPAD = 4,

    MOVE = 1,
    SHIMA = 2,
    LSHIMA = 4,

    NUM_PATTERNS = 21,
};

struct FMCount {
    int move;
    int shima;
    int lshima;
};

struct PulldownPatternField {
    bool split;
    bool merge;
    bool shift;
};

struct PulldownPattern {
    std::array<PulldownPatternField, 10 * 4> fields;
    int cycle;

    PulldownPattern(int nf0, int nf1, int nf2, int nf3);
    PulldownPattern();

    const PulldownPatternField *getPattern(int n) const {
        return fields.data() + 10 + n - 2;
    }
    int getCycleLength() const {
        return cycle;
    }
};

struct Frame24Info {
    int cycleIndex;
    int frameIndex;
    int fieldStartIndex;
    int numFields;
    int fieldShift;
};

struct FMData {
    std::array<float, 14> mft;
    std::array<float, 14> mftr;
    std::array<float, 14> mftcost;
};

struct FMMatch {
    std::array<float, NUM_PATTERNS> shima;
    std::array<float, NUM_PATTERNS> costs;
    std::array<float, NUM_PATTERNS> reliability;
};

struct KFMResult {
    int pattern;
    int is60p;
    float score;
    float cost;
    float reliability;

    KFMResult();
    KFMResult(int pattern, float score, float cost, float reliability);
    KFMResult(const FMMatch& match, int pattern);
};

struct KFMAnalyzeParam {
    float lscale = 5.0f;
    float costth = -1.0f;
    float adj2224 = 0.5f;
    float adj30 = 1.5f;
    int cycleRange = 5;
    float NGThresh = 1.0f;
    int pastCycles = 180;
    float th60 = 3.0f;
    float th24 = 0.1f;
    float rel24 = 0.2f;
};

struct NoiseResult {
    uint64_t noise0;
    uint64_t noise1;
    uint64_t noiseR0;
    uint64_t noiseR1;
    uint64_t diff0;
    uint64_t diff1;
};

struct UCFNoiseMeta {
    enum {
        VERSION = 1,
        MAGIC_KEY = 0x39EDF8,
    };
    int nMagicKey = MAGIC_KEY;
    int nVersion = VERSION;

    int srcw = 0;
    int srch = 0;
    int srcUVw = 0;
    int srcUVh = 0;
    int noisew = 0;
    int noiseh = 0;
    int noiseUVw = 0;
    int noiseUVh = 0;
};

enum DECOMB_UCF_RESULT {
    DECOMB_UCF_CLEAN_1,
    DECOMB_UCF_CLEAN_2,
    DECOMB_UCF_USE_0,
    DECOMB_UCF_USE_1,
    DECOMB_UCF_NOISY,
};

struct DecombUCFThreshScore {
    double y1 = 0.0;
    double y2 = 0.0;
    double y3 = 0.0;
    double y4 = 0.0;
    double y5 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;
    double x4 = 0.0;
    double x5 = 0.0;

    double calc(double x) const;
};

struct DecombUCFParam {
    enum {
        VERSION = 1,
        MAGIC_KEY = 0x40EDF8,
    };
    int nMagicKey = MAGIC_KEY;
    int nVersion = VERSION;

    int chroma = 1;
    double fd_thresh = 128.0;
    int th_mode = 0;
    double off_t = 0.0;
    double off_b = 0.0;
    int namax_thresh = 82;
    int namax_diff = 38;
    double nrt1y = 28.0;
    double nrt2y = 36.0;
    double nrt2x = 53.5;
    double nrw = 2.0;
    bool show = false;

    DecombUCFThreshScore th_score;

    DecombUCFParam();
};

const DecombUCFThreshScore& decombUCFDefaultThreshScore(int thMode);
const char *decombUCFResultToString(DECOMB_UCF_RESULT result);
DECOMB_UCF_RESULT CalcDecombUCF(const UCFNoiseMeta *meta, const DecombUCFParam *param,
    const NoiseResult *result0, const NoiseResult *result1, bool second, std::string *message = nullptr);

class PulldownPatterns {
    PulldownPattern m_p2323;
    PulldownPattern m_p2233;
    PulldownPattern m_p2224;
    PulldownPattern m_p30;
    std::array<int, 5> m_patternOffsets;
    std::array<const PulldownPatternField *, NUM_PATTERNS> m_allPatterns;

public:
    PulldownPatterns();

    const PulldownPatternField *getPattern(int patternIndex) const;
    const char *patternToString(int patternIndex, int& index) const;

    Frame24Info getFrame24(int patternIndex, int n24) const;
    Frame24Info getFrame60(int patternIndex, int n60) const;

    FMMatch matching(const FMData& data, int width, int height, float costth, float adj2224, float adj30) const;

    static bool is30p(int patternIndex) {
        return patternIndex == NUM_PATTERNS - 1;
    }
    static bool is60p(int patternIndex) {
        return patternIndex == NUM_PATTERNS;
    }
};

class KFMAnalyze {
    PulldownPatterns m_patterns;
    KFMAnalyzeParam m_param;
    std::vector<KFMResult> m_results;
    std::deque<KFMResult> m_recentBest;
    FMMatch m_lastMatch;
    bool m_hasLastMatch;
    int m_pattern;

public:
    explicit KFMAnalyze(const KFMAnalyzeParam& param = KFMAnalyzeParam());

    void reset();

    const PulldownPatterns& patterns() const {
        return m_patterns;
    }
    const KFMAnalyzeParam& param() const {
        return m_param;
    }
    const std::vector<KFMResult>& results() const {
        return m_results;
    }

    FMData makeFMData(const FMCount *counts18, int width, int height) const;
    FMMatch matching(const FMData& data, int width, int height) const;
    KFMResult realtime(const FMData& data, int width, int height) const;
    KFMResult realtimeFromCounts(const FMCount *counts18, int width, int height) const;
    KFMResult analyzeMatch(const FMMatch& match);
    KFMResult analyzeCycle(const FMData& data, int width, int height);
    KFMResult analyzeCycleFromCounts(const FMCount *counts18, int width, int height);
    void analyzeTrailingCycles(int cycles);

    void mark60p();

    static float defaultCostth(int height);
    static int bestPattern(const FMMatch& match);
};

std::vector<uint8_t> serializeResults(const std::vector<KFMResult>& results);

} // namespace RGYKFM

static_assert(sizeof(RGYKFM::KFMResult) == 20);
static_assert(sizeof(RGYKFM::FMCount) == 12);
static_assert(sizeof(RGYKFM::NoiseResult) == 48);
static_assert(sizeof(RGYKFM::UCFNoiseMeta) == 40);
