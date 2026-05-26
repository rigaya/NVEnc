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
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rgy_prm.h"

static constexpr int NNEDI_NUM_NSIZE = 7;
static constexpr int NNEDI_NUM_NNS = 5;
static constexpr VppNnediQuality NNEDI_DEFAULT_QUALITY = VPP_NNEDI_QUALITY_FAST;
static constexpr int NNEDI_DEFAULT_PRESCREEN = 2;
static constexpr VppNnediErrorType NNEDI_DEFAULT_ERRORTYPE = VPP_NNEDI_ETYPE_ABS;
static constexpr int NNEDI_DEFAULT_BITS_PER_PIXEL = 8;

struct RGYFilterNnediWeightsLayout {
    VppNnediNSize nsize = VPP_NNEDI_NSIZE_8x6;
    int nns = 16;
    int xdia = 0;
    int ydia = 0;
    int neurons = 0;
    int asize = 0;
    int legacyPrescreenerFloats = 0;
    int prescreenerNetworkFloats = 0;
    int predictorPlaneFloats = 0;
    int predictorCatalogFloats = 0;
    int predictorCatalogOffsetFloats = 0;
    size_t rawWeightFloatCount = 0;
    size_t rawPrescreenerOffsetFloats = 0;
    std::array<size_t, 2> rawPredictorOffsetFloats = {};
};

struct RGYFilterNnediWeightsParam {
    VppNnediNSize nsize = VPP_NNEDI_NSIZE_16x6;
    int nns = 32;
    VppNnediQuality quality = NNEDI_DEFAULT_QUALITY;
    int prescreen = NNEDI_DEFAULT_PRESCREEN;
    VppNnediErrorType errortype = NNEDI_DEFAULT_ERRORTYPE;
    int bitsPerPixel = NNEDI_DEFAULT_BITS_PER_PIXEL;
};

struct RGYFilterNnediTransformedWeights {
    RGYFilterNnediWeightsParam param;
    RGYFilterNnediWeightsLayout layout;
    std::vector<float> prescreenerFp32;
    std::vector<float> predictorFp32;
};

struct RGYFilterNnediFloatWeightsSample {
    size_t index = 0;
    float value = 0.0f;
};

struct RGYFilterNnediFloatBufferDigest {
    size_t floatCount = 0;
    uint64_t fnv1a64 = 0;
    std::array<RGYFilterNnediFloatWeightsSample, 8> samples = {};
};

struct RGYFilterNnediWeightsSummary {
    RGYFilterNnediWeightsParam param;
    RGYFilterNnediWeightsLayout layout;
    RGYFilterNnediFloatBufferDigest prescreenerFp32;
    RGYFilterNnediFloatBufferDigest predictorFp32;
    uint64_t combinedFnv1a64 = 0;
};

struct RGYFilterNnediWeightsDefaultSampleResult {
    bool success = false;
    std::string message;
    RGYFilterNnediWeightsSummary defaultSlower;
    RGYFilterNnediWeightsSummary chroma;
};

struct RGYFilterNnediWeightsSelfCheckResult {
    bool success = false;
    std::string message;
    RGYFilterNnediWeightsLayout defaultSlower;
    RGYFilterNnediWeightsLayout chroma;
};

bool rgy_filter_nnedi_weights_layout(RGYFilterNnediWeightsLayout& layout, VppNnediNSize nsize, int nns, int prescreen, VppNnediErrorType errortype, std::string *errorMessage = nullptr);
bool rgy_filter_nnedi_transform_weights(RGYFilterNnediTransformedWeights& dst, const float *rawWeights, size_t rawWeightFloatCount, const RGYFilterNnediWeightsParam& param, std::string *errorMessage = nullptr);
RGYFilterNnediWeightsSummary rgy_filter_nnedi_weights_summary(const RGYFilterNnediTransformedWeights& weights);
RGYFilterNnediWeightsDefaultSampleResult rgy_filter_nnedi_weights_default_samples(const float *rawWeights, size_t rawWeightFloatCount);
RGYFilterNnediWeightsSelfCheckResult rgy_filter_nnedi_weights_self_check();
