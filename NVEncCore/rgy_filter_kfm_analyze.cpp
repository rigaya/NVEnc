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

#include "rgy_filter_kfm_analyze.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace RGYKFM {

static const DecombUCFThreshScore THRESH_SCORE_PARAM_TABLE[] = {
    {},
    { 13, 17, 17, 20, 50, 20, 28, 32, 37, 50 },
    { 14, 18, 20, 40, 50, 19, 28, 36, 42, 50 },
    { 15, 19, 21, 43, 63, 20, 28, 36, 41, 53 },
    { 15, 20, 23, 43, 63, 20, 28, 36, 41, 53 },
    { 15, 20, 23, 45, 63, 20, 28, 36, 41, 50 },
    { 15, 21, 23, 45, 63, 20, 28, 36, 41, 50 },
    { 15, 22, 24, 45, 63, 20, 28, 35, 41, 50 },
    { 17, 25, 28, 47, 64, 20, 28, 33, 41, 48 },
    { 20, 32, 38, 52, 66, 21, 30, 36, 40, 48 },
    { 22, 37, 44, 52, 66, 22, 32, 35, 40, 48 },
};

static float splitScore(const PulldownPatternField *pattern, const float *fv) {
    float sumsplit = 0.0f;
    float sumnsplit = 0.0f;

    for (int i = 0; i < 14; ++i) {
        if (pattern[i].split) {
            sumsplit += fv[i];
        } else {
            sumnsplit += fv[i];
        }
    }

    return sumsplit - sumnsplit;
}

static float splitCost(const PulldownPatternField *pattern, const float *fv, const float *fvcost, float costth) {
    int nsplit = 0;
    float sumcost = 0.0f;

    for (int i = 0; i < 14; ++i) {
        if (pattern[i].split) {
            nsplit++;
            if (fv[i] < costth) {
                sumcost += (costth - fv[i]) * std::log2(fvcost[i] + 1.0f);
            }
        }
    }

    return sumcost / nsplit;
}

static float splitReliability(const PulldownPatternField *pattern, const float *fv, float costth) {
    int nsplit = 0;
    float sumcost = 0.0f;

    for (int i = 0; i < 14; ++i) {
        if (pattern[i].split) {
            nsplit++;
            if (fv[i] < costth) {
                sumcost += costth - fv[i];
            }
        }
    }

    return sumcost / nsplit;
}

static void checkPatternIndex(int patternIndex, bool allow60p = false) {
    const int maxPattern = allow60p ? NUM_PATTERNS : NUM_PATTERNS - 1;
    if (patternIndex < 0 || patternIndex > maxPattern) {
        throw std::out_of_range("KFM pattern index is out of range.");
    }
}

double DecombUCFThreshScore::calc(double x) const {
    return (x < x1) ? y1
        : (x < x2) ? ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)
        : (x < x3) ? ((y3 - y2) * x + x3 * y2 - x2 * y3) / (x3 - x2)
        : (x < x4) ? ((y4 - y3) * x + x4 * y3 - x3 * y4) / (x4 - x3)
        : (x < x5) ? ((y5 - y4) * x + x5 * y4 - x4 * y5) / (x5 - x4)
        : y5;
}

DecombUCFParam::DecombUCFParam()
    : th_score(decombUCFDefaultThreshScore(5)) {
}

const DecombUCFThreshScore& decombUCFDefaultThreshScore(int thMode) {
    if (thMode < 0 || thMode >= static_cast<int>(sizeof(THRESH_SCORE_PARAM_TABLE) / sizeof(THRESH_SCORE_PARAM_TABLE[0]))) {
        throw std::out_of_range("DecombUCF threshold mode is out of range.");
    }
    return THRESH_SCORE_PARAM_TABLE[thMode];
}

const char *decombUCFResultToString(DECOMB_UCF_RESULT result) {
    switch (result) {
    case DECOMB_UCF_CLEAN_1: return "CLEAN_1";
    case DECOMB_UCF_CLEAN_2: return "CLEAN_2";
    case DECOMB_UCF_USE_0: return "USE_0";
    case DECOMB_UCF_USE_1: return "USE_1";
    case DECOMB_UCF_NOISY: return "NOISY";
    default: return "UNKNOWN";
    }
}

DECOMB_UCF_RESULT CalcDecombUCF(const UCFNoiseMeta *meta, const DecombUCFParam *param,
    const NoiseResult *result0, const NoiseResult *result1, bool second, std::string *message) {
    if (meta == nullptr || param == nullptr || result0 == nullptr || (second && result1 == nullptr)) {
        throw std::invalid_argument("DecombUCF input pointer must not be null.");
    }
    if (meta->srcw <= 0 || meta->srch <= 0 || meta->noisew <= 0 || meta->noiseh <= 0 || meta->noiseUVw <= 0 || meta->noiseUVh <= 0) {
        throw std::invalid_argument("DecombUCF source and noise sizes must be positive.");
    }

    const double pixels = static_cast<double>(meta->srcw) * meta->srch;
    const double noisepixels = static_cast<double>(meta->noisew) * meta->noiseh;
    const double noisepixelsUV = static_cast<double>(meta->noiseUVw) * meta->noiseUVh * 2.0;

    const double field_diff = (second
        ? static_cast<double>(result0[0].diff1 + result0[1].diff1)
        : static_cast<double>(result0[0].diff0 + result0[1].diff0)) / (6.0 * pixels) * 100.0;

    const double noise_t_y = (second ? result0[0].noise1 : result0[0].noise0) / noisepixels;
    const double noise_t_uv = (second ? result0[1].noise1 : result0[1].noise0) / noisepixelsUV;
    const double noise_b_y = (second ? result1[0].noise0 : result0[0].noise1) / noisepixels;
    const double noise_b_uv = (second ? result1[1].noise0 : result0[1].noise1) / noisepixelsUV;
    const double navg1_y = (noise_t_y + noise_b_y) / 2.0;
    const double navg1_uv = (noise_t_uv + noise_b_uv) / 2.0;
    const double navg2_y = (second ? result0[0].noiseR1 : result0[0].noiseR0) / noisepixels / 2.0;
    const double navg2_uv = (second ? result0[1].noiseR1 : result0[1].noiseR0) / noisepixelsUV / 2.0;
    const double diff1_y = noise_t_y - noise_b_y;
    const double diff1_uv = noise_t_uv - noise_b_uv;

    double diff1 = 0.0;
    double navg1 = 0.0;
    double navg1_d = 0.0;
    double navg2 = 0.0;
    if (param->chroma == 0) {
        diff1 = diff1_y;
        navg1_d = navg1 = navg1_y;
        navg2 = navg2_y;
    } else if (param->chroma == 1) {
        diff1 = diff1_uv;
        navg1 = -1.0;
        navg1_d = navg1_uv;
        navg2 = navg2_uv;
    } else {
        diff1 = (diff1_y + diff1_uv) / 2.0;
        navg1_d = navg1 = (navg1_y + navg1_uv) / 2.0;
        navg2 = (navg2_y + navg2_uv) / 2.0;
    }

    const double absdiff1 = std::abs(diff1);
    const double nmin1 = navg2 - absdiff1 / 2.0;
    const double nmin = (nmin1 < 7.0) ? nmin1 * 4.0 : nmin1 + 21.0;
    const double nmax = navg2 + absdiff1 * param->nrw;
    const double off_thresh = (diff1 < 0.0) ? param->off_t : param->off_b;
    const double min_thresh = (navg1 < param->namax_thresh)
        ? param->th_score.calc(nmin) + off_thresh
        : param->namax_diff + off_thresh;
    const double diff = absdiff1 < 1.8 ? diff1 * 10.0
        : absdiff1 < 5.0 ? diff1 * 5.0 + (diff1 / absdiff1) * 9.0
        : absdiff1 < 10.0 ? diff1 * 2.0 + (diff1 / absdiff1) * 24.0
        : diff1 + (diff1 / absdiff1) * 34.0;

    DECOMB_UCF_RESULT result;
    if (std::abs(diff) < min_thresh) {
        result = ((nmax < param->nrt1y) || (param->nrt2x < navg1_d && nmax < param->nrt2y))
            ? DECOMB_UCF_CLEAN_2 : DECOMB_UCF_NOISY;
    } else if (navg1 < param->namax_thresh) {
        result = (diff < 0.0) ? DECOMB_UCF_USE_0 : DECOMB_UCF_USE_1;
    } else {
        result = (diff < 0.0) ? DECOMB_UCF_USE_1 : DECOMB_UCF_USE_0;
    }

    if (message) {
        char debug1_n_t[128] = {};
        char debug1_n_b[128] = {};
        if (param->chroma == 0) {
            std::snprintf(debug1_n_t, sizeof(debug1_n_t), " [Y : %7f]", noise_t_y);
            std::snprintf(debug1_n_b, sizeof(debug1_n_b), " [Y : %7f]", noise_b_y);
        } else if (param->chroma == 1) {
            std::snprintf(debug1_n_t, sizeof(debug1_n_t), " [UV: %7f]", noise_t_uv);
            std::snprintf(debug1_n_b, sizeof(debug1_n_b), " [UV: %7f]", noise_b_uv);
        } else {
            std::snprintf(debug1_n_t, sizeof(debug1_n_t), " [Y : %7f] [UV: %7f]", noise_t_y, noise_t_uv);
            std::snprintf(debug1_n_b, sizeof(debug1_n_b), " [Y : %7f] [UV: %7f]", noise_b_y, noise_b_uv);
        }
        char reschar = '-';
        char fdeq = '>';
        char noiseeq = '<';
        const char *field = "";
        if (field_diff < param->fd_thresh) {
            reschar = 'A';
            field = "notbob";
            fdeq = '<';
        } else if (result == DECOMB_UCF_CLEAN_2 || result == DECOMB_UCF_NOISY) {
            reschar = 'B';
            field = "notbob";
            if (result == DECOMB_UCF_NOISY) {
                noiseeq = '>';
            }
        } else {
            reschar = 'C';
            field = (result == DECOMB_UCF_USE_0) ? "First" : "Second";
        }
        const char *extra = "";
        if (result == DECOMB_UCF_NOISY) {
            extra = "NR";
        } else if (field_diff < param->fd_thresh && result != DECOMB_UCF_CLEAN_2) {
            extra = "NOT CLEAN ???";
        } else if (navg1 >= param->namax_thresh) {
            extra = "Reversed";
        }
        char buf[512] = {};
        std::snprintf(buf, sizeof(buf),
            "[%c] %-6s  //  Fdiff =  %8f (FieldDiff %c %8f)\n"
            "                diff =  %8f  (NoiseDiff %c %.2f)\n"
            " Noise // First %s / Second %s\n"
            " navg1 : %.2f / nmin : %.2f / diff1 : %.3f / nrt : %.1f\n"
            "%s\n",
            reschar, field, field_diff, fdeq, param->fd_thresh,
            diff, noiseeq, min_thresh,
            debug1_n_t, debug1_n_b,
            navg1_d, nmin, diff1, nmax,
            extra);
        *message += buf;
    }

    return (field_diff < param->fd_thresh) ? DECOMB_UCF_CLEAN_1 : result;
}

KFMResult::KFMResult()
    : pattern(0)
    , is60p(0)
    , score(0.0f)
    , cost(0.0f)
    , reliability(0.0f) {
}

KFMResult::KFMResult(int pattern_, float score_, float cost_, float reliability_)
    : pattern(pattern_)
    , is60p(0)
    , score(score_)
    , cost(cost_)
    , reliability(reliability_) {
}

KFMResult::KFMResult(const FMMatch& match, int pattern_)
    : pattern(pattern_)
    , is60p(0)
    , score(match.shima[pattern_])
    , cost(match.costs[pattern_])
    , reliability(match.reliability[pattern_]) {
}

PulldownPattern::PulldownPattern(int nf0, int nf1, int nf2, int nf3)
    : fields()
    , cycle(10) {
    if (nf0 + nf1 + nf2 + nf3 != 10) {
        throw std::invalid_argument("sum of KFM pulldown fields must be 10.");
    }
    if (nf0 == nf2 && nf1 == nf3) {
        cycle = 5;
    }
    const int nfields[] = { nf0, nf1, nf2, nf3 };
    for (int c = 0, fstart = 0; c < 4; ++c) {
        for (int i = 0; i < 4; ++i) {
            const int nf = nfields[i];
            for (int f = 0; f < nf - 2; ++f) {
                fields[fstart + f].merge = true;
            }
            fields[fstart + nf - 1].split = true;

            if (nf0 == 4 && nf1 == 2 && nf2 == 2 && nf3 == 2) {
                if (i < 2) {
                    fields[fstart + nf - 1].shift = true;
                }
            }

            fstart += nf;
        }
    }
}

PulldownPattern::PulldownPattern()
    : fields()
    , cycle(2) {
    for (int c = 0, fstart = 0; c < 4; ++c) {
        for (int i = 0; i < 4; ++i) {
            const int nf = 2;
            fields[fstart + nf - 1].split = true;
            fstart += nf;
        }
    }
}

PulldownPatterns::PulldownPatterns()
    : m_p2323(2, 3, 2, 3)
    , m_p2233(2, 2, 3, 3)
    , m_p2224(4, 2, 2, 2)
    , m_p30()
    , m_patternOffsets()
    , m_allPatterns() {
    const PulldownPattern *patterns[] = { &m_p2323, &m_p2233, &m_p2224, &m_p30 };
    const int steps[] = { 1, 1, 2, 2 };

    int pi = 0;
    for (int p = 0; p < 4; ++p) {
        m_patternOffsets[p] = pi;
        for (int i = 0; i < patterns[p]->getCycleLength(); i += steps[p]) {
            m_allPatterns[pi++] = patterns[p]->getPattern(i);
        }
    }
    m_patternOffsets[4] = pi;

    if (pi != NUM_PATTERNS) {
        throw std::runtime_error("invalid KFM pulldown pattern table.");
    }
}

const PulldownPatternField *PulldownPatterns::getPattern(int patternIndex) const {
    checkPatternIndex(patternIndex);
    return m_allPatterns[patternIndex];
}

const char *PulldownPatterns::patternToString(int patternIndex, int& index) const {
    checkPatternIndex(patternIndex);
    static const char *names[] = { "2323", "2233", "2224", "30p" };
    const auto pattern = static_cast<int>(std::upper_bound(m_patternOffsets.begin(), m_patternOffsets.begin() + 4, patternIndex) - m_patternOffsets.begin() - 1);
    index = patternIndex - m_patternOffsets[pattern];
    return names[pattern];
}

Frame24Info PulldownPatterns::getFrame24(int patternIndex, int n24) const {
    checkPatternIndex(patternIndex);
    Frame24Info info = {};
    info.cycleIndex = n24 / 4;
    info.frameIndex = n24 % 4;

    int searchFrame = info.frameIndex;
    if (is30p(patternIndex)) {
        if (searchFrame >= 2) {
            ++searchFrame;
        }
    }

    const PulldownPatternField *ptn = m_allPatterns[patternIndex];
    int fldstart = 0;
    int nframes = 0;
    for (int i = 0; i < 16; ++i) {
        if (ptn[i].split) {
            if (fldstart >= 1) {
                if (nframes++ == searchFrame) {
                    const int nextfldstart = i + 1;
                    info.fieldStartIndex = fldstart - 2;
                    info.numFields = nextfldstart - fldstart;
                    return info;
                }
            }
            fldstart = i + 1;
        }
    }

    throw std::runtime_error("failed to resolve KFM 24p frame info.");
}

Frame24Info PulldownPatterns::getFrame60(int patternIndex, int n60) const {
    checkPatternIndex(patternIndex);
    Frame24Info info = {};
    info.cycleIndex = n60 / 10;

    const PulldownPatternField *ptn = m_allPatterns[patternIndex];
    int fldstart = 0;
    int nframes = -1;
    const int findex = n60 % 10;

    for (int i = 0; i < 16; ++i) {
        if (ptn[i].split) {
            if (fldstart >= 1) {
                ++nframes;
            }
            const int nextfldstart = i + 1;
            if (findex < nextfldstart - 2) {
                info.frameIndex = nframes;
                info.fieldStartIndex = fldstart - 2;
                info.numFields = nextfldstart - fldstart;
                info.fieldShift = ptn[findex + 2].shift ? 1 : 0;
                return info;
            }
            fldstart = i + 1;
        }
    }

    throw std::runtime_error("failed to resolve KFM 60p frame info.");
}

FMMatch PulldownPatterns::matching(const FMData& data, int width, int height, float costth, float adj2224, float adj30) const {
    (void)width;
    (void)height;

    FMMatch match = {};

    for (int i = 0; i < NUM_PATTERNS; ++i) {
        match.shima[i] = splitScore(m_allPatterns[i], data.mftr.data());
        match.costs[i] = splitCost(m_allPatterns[i], data.mftr.data(), data.mftcost.data(), costth);
        match.reliability[i] = splitReliability(m_allPatterns[i], data.mftr.data(), costth);
    }

    for (int i = m_patternOffsets[2]; i < m_patternOffsets[3]; ++i) {
        match.shima[i] -= adj2224;
    }
    for (int i = m_patternOffsets[3]; i < m_patternOffsets[4]; ++i) {
        match.shima[i] -= adj30;
    }

    return match;
}

KFMAnalyze::KFMAnalyze(const KFMAnalyzeParam& param)
    : m_patterns()
    , m_param(param)
    , m_results()
    , m_recentBest()
    , m_lastMatch()
    , m_hasLastMatch(false)
    , m_pattern(0) {
}

void KFMAnalyze::reset() {
    m_results.clear();
    m_recentBest.clear();
    m_lastMatch = {};
    m_hasLastMatch = false;
    m_pattern = 0;
}

FMData KFMAnalyze::makeFMData(const FMCount *counts18, int width, int height) const {
    if (counts18 == nullptr) {
        throw std::invalid_argument("KFM FMCount pointer must not be null.");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("KFM source size must be positive.");
    }

    int mft[18] = {};
    for (int i = 1; i < 17; ++i) {
        const int split = std::min(counts18[i - 1].move, counts18[i].move);
        mft[i] = split + counts18[i].shima + static_cast<int>(counts18[i].lshima * m_param.lscale);
    }

    FMData data = {};
    const int vbase = std::max(1, static_cast<int>(width * height * 0.001f) >> 4);
    for (int i = 0; i < 14; ++i) {
        data.mft[i] = static_cast<float>(mft[i + 2]);
        data.mftr[i] = (mft[i + 2] + vbase) * 2.0f / (mft[i + 1] + mft[i + 3] + vbase * 2.0f) - 1.0f;
        data.mftcost[i] = static_cast<float>(mft[i + 1] + mft[i + 3]) / vbase;
    }

    return data;
}

FMMatch KFMAnalyze::matching(const FMData& data, int width, int height) const {
    const auto costth = (m_param.costth >= 0.0f) ? m_param.costth : defaultCostth(height);
    return m_patterns.matching(data, width, height, costth, m_param.adj2224, m_param.adj30);
}

KFMResult KFMAnalyze::realtime(const FMData& data, int width, int height) const {
    const auto match = matching(data, width, height);
    return KFMResult(match, bestPattern(match));
}

KFMResult KFMAnalyze::realtimeFromCounts(const FMCount *counts18, int width, int height) const {
    return realtime(makeFMData(counts18, width, height), width, height);
}

KFMResult KFMAnalyze::analyzeMatch(const FMMatch& match) {
    m_lastMatch = match;
    m_hasLastMatch = true;
    m_results.emplace_back(match, m_pattern);
    m_recentBest.emplace_front(match, bestPattern(match));

    const auto current = static_cast<int>(m_results.size()) - 1;
    if (m_results.back().pattern != m_recentBest.front().pattern) {
        float ngScore = 0.0f;
        const int scoreCycles = std::min(m_param.cycleRange, static_cast<int>(m_recentBest.size()));
        for (int i = 0; i < scoreCycles; ++i) {
            ngScore += m_recentBest[i].score - m_results[current - i].score;
        }
        if (ngScore > m_param.NGThresh * m_param.cycleRange) {
            m_pattern = m_recentBest.front().pattern;

            for (int i = 0; i < static_cast<int>(m_recentBest.size()); ++i) {
                if (m_recentBest[i].pattern != m_pattern) {
                    break;
                }
                m_results[current - i] = m_recentBest[i];
            }
        }
    }

    if (static_cast<int>(m_recentBest.size()) > m_param.pastCycles) {
        m_recentBest.pop_back();
    }

    return m_results.back();
}

KFMResult KFMAnalyze::analyzeCycle(const FMData& data, int width, int height) {
    return analyzeMatch(matching(data, width, height));
}

KFMResult KFMAnalyze::analyzeCycleFromCounts(const FMCount *counts18, int width, int height) {
    return analyzeCycle(makeFMData(counts18, width, height), width, height);
}

void KFMAnalyze::analyzeTrailingCycles(int cycles) {
    if (!m_hasLastMatch) {
        return;
    }
    for (int i = 0; i < cycles; ++i) {
        analyzeMatch(m_lastMatch);
    }
}

void KFMAnalyze::mark60p() {
    bool is60p = true;
    for (int i = 0; i < static_cast<int>(m_results.size()); ++i) {
        auto& cur = m_results[i];
        if (is60p) {
            if (cur.cost < m_param.th24) {
                if (cur.reliability < m_param.rel24) {
                    is60p = false;
                }
            } else {
                cur.is60p = true;
            }
        } else {
            if (cur.cost >= m_param.th60) {
                is60p = true;
                for (int t = i; t >= 0; --t) {
                    auto& prev = m_results[t];
                    if (prev.cost < m_param.th24) {
                        if (prev.reliability < m_param.rel24) {
                            break;
                        }
                    } else {
                        prev.is60p = true;
                    }
                }
            }
        }
    }
}

float KFMAnalyze::defaultCostth(int height) {
    return (height >= 720) ? 1.5f : 1.0f;
}

int KFMAnalyze::bestPattern(const FMMatch& match) {
    const auto it = std::max_element(match.shima.begin(), match.shima.end());
    return static_cast<int>(it - match.shima.begin());
}

std::vector<uint8_t> serializeResults(const std::vector<KFMResult>& results) {
    std::vector<uint8_t> data(results.size() * sizeof(KFMResult));
    if (!data.empty()) {
        std::memcpy(data.data(), results.data(), data.size());
    }
    return data;
}

} // namespace RGYKFM

