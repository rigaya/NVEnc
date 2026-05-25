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

#include "NVEncFilterDegrainCommon.h"

#include <algorithm>

#include "rgy_util.h"

std::string degrainStripUtf8Bom(const std::string &source) {
    static const char bom[] = "\xef\xbb\xbf";
    if (source.size() >= 3 && source.compare(0, 3, bom, 3) == 0) {
        return source.substr(3);
    }
    return source;
}

bool degrainReplaceInclude(std::string &source, const char *includeName, const std::string &includeSource, const std::shared_ptr<RGYLog> &log) {
    const auto includePattern = std::string("#include \"") + includeName + "\"";
    if (source.find(includePattern) == std::string::npos) {
        if (log) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to search degrain CUDA include: %s\n"), char_to_tstring(includePattern).c_str());
        }
        return false;
    }
    source = str_replace(source, includePattern, includeSource);
    return true;
}

int degrainLevel1FrameCount(const int temporalDirections) {
    return std::min(temporalDirections + 1, RGY_DEGRAIN_MAX_LEVEL1_LUMA_FRAMES);
}

int degrainBinomialRowValue(const int n, const int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    const int order = std::min(k, n - k);
    int coeff = 1;
    for (int i = 1; i <= order; i++) {
        coeff = (coeff * (n - order + i)) / i;
    }
    return coeff;
}

RGYDegrainTemporalMixPriorTable degrainBuildTemporalMixPriorTable(const int temporalDirections, const bool binomial) {
    RGYDegrainTemporalMixPriorTable table;
    table.fill(1.0f);
    if (!binomial) {
        return table;
    }
    const int radius = std::max(temporalDirections / 2, 1);
    if (radius > 2) {
        return table;
    }
    const int row = radius * 2;
    table[0] = (float)degrainBinomialRowValue(row, radius);
    for (int refDirection = 0; refDirection < RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS; refDirection++) {
        const int delta = refDirection / 2 + 1;
        table[1 + refDirection] = (delta <= radius) ? (float)degrainBinomialRowValue(row, radius - delta) : 1.0f;
    }
    return table;
}

RGYDegrainRefDisableArray degrainReferenceAvailability(const RGYFilterDegrainFrameSet &frames) {
    RGYDegrainRefDisableArray disableRefs;
    disableRefs.fill(true);
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        disableRefs[rgy_degrain_ref_index(delta, false)] = !frames.backwardRefInRange(delta);
        disableRefs[rgy_degrain_ref_index(delta, true)] = !frames.forwardRefInRange(delta);
    }
    return disableRefs;
}

RGY_CSP degrainAnalysisLumaCsp(const RGYFrameInfo &frameInfo) {
    return (RGY_CSP_BIT_DEPTH[frameInfo.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
}

bool degrainRequiresAnalysisLumaCache(const VppDegrain &degrain) {
    return degrain.tr0 > 0 || degrain.searchRefine > 0 || degrain.rep0 > 0 || degrain.tvRange;
}

std::vector<cudaEvent_t> degrainWaitEventList(const std::vector<RGYCudaEvent> &waitEvents) {
    std::vector<cudaEvent_t> cudaEvents;
    cudaEvents.reserve(waitEvents.size());
    for (const auto &waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            cudaEvents.push_back(waitEvent());
        }
    }
    return cudaEvents;
}

std::shared_ptr<RGYFrameDataRtgmcSearchLuma> degrainGetAttachedSearchLuma(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataRtgmcSearchLuma>(data) != nullptr;
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    auto searchLuma = std::dynamic_pointer_cast<RGYFrameDataRtgmcSearchLuma>(*frameData);
    const auto searchFrame = searchLuma ? searchLuma->frame() : nullptr;
    if (!searchFrame || searchFrame->ptr[0] == nullptr
        || searchFrame->width != frame->width || searchFrame->height != frame->height) {
        return nullptr;
    }
    return searchLuma;
}

const RGYFrameInfo *degrainAttachedSearchLumaFrame(const RGYFrameInfo *frame) {
    const auto searchLuma = degrainGetAttachedSearchLuma(frame);
    return searchLuma ? searchLuma->frame() : nullptr;
}
