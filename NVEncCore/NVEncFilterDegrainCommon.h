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
#include <memory>
#include <string>
#include <vector>

#include "NVEncFilterDegrainMV.h"
#include "NVEncFilterRtgmcSearchPrefilter.h"

static constexpr int DEGRAIN_DEBUG_BLOCK_X = 16;
static constexpr int DEGRAIN_DEBUG_BLOCK_Y = 16;
static constexpr int DEGRAIN_ANALYZE_LOCAL_X = 1;
static constexpr int DEGRAIN_ANALYZE_LOCAL_Y = 1;

using RGYDegrainTemporalMixPriorTable = std::array<float, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS + 1>;

std::string degrainStripUtf8Bom(const std::string &source);
bool degrainReplaceInclude(std::string &source, const char *includeName, const std::string &includeSource, const std::shared_ptr<RGYLog> &log);
int degrainLevel1FrameCount(int temporalDirections);
int degrainBinomialRowValue(int n, int k);
RGYDegrainTemporalMixPriorTable degrainBuildTemporalMixPriorTable(int temporalDirections, bool binomial);
RGYDegrainRefDisableArray degrainReferenceAvailability(const RGYFilterDegrainFrameSet &frames);
RGY_CSP degrainAnalysisLumaCsp(const RGYFrameInfo &frameInfo);
bool degrainRequiresAnalysisLumaCache(const VppDegrain &degrain);
std::vector<cudaEvent_t> degrainWaitEventList(const std::vector<RGYCudaEvent> &waitEvents);
std::shared_ptr<RGYFrameDataRtgmcSearchLuma> degrainGetAttachedSearchLuma(const RGYFrameInfo *frame);
const RGYFrameInfo *degrainAttachedSearchLumaFrame(const RGYFrameInfo *frame);
