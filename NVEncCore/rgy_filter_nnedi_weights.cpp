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

#include "rgy_filter_nnedi_weights.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace {

static constexpr std::array<int, NNEDI_NUM_NSIZE> NNEDI_XDIA = { 8, 16, 32, 48, 8, 16, 32 };
static constexpr std::array<int, NNEDI_NUM_NSIZE> NNEDI_YDIA = { 6, 6, 6, 6, 4, 4, 4 };
static constexpr std::array<int, NNEDI_NUM_NNS> NNEDI_NNS = { 16, 32, 64, 128, 256 };
static constexpr int NNEDI_LEGACY_PRESCREENER_FLOATS = 252;
static constexpr int NNEDI_PRESCREENER_NETWORK_FLOATS = 280;
static constexpr int NNEDI_PRESCREENER_HIDDEN_COUNT = 4;
static constexpr int NNEDI_PRESCREENER_SAMPLE_COUNT = 64;
static constexpr int NNEDI_PRESCREENER_OUTPUT_LANES = 4;
static constexpr int NNEDI_PREDICTOR_FP32_LANES = 16;
static constexpr uint64_t FNV1A64_OFFSET_BASIS = 14695981039346656037ull;
static constexpr uint64_t FNV1A64_PRIME = 1099511628211ull;
static constexpr size_t WEIGHTBIN_FILE_SIZE = 13574928u;

void set_error(std::string *errorMessage, const std::string& message) {
    if (errorMessage) {
        *errorMessage = message;
    }
}

int nnedi_nsize_index(const VppNnediNSize nsize) {
    const auto nsizeIndex = static_cast<int>(nsize);
    return (0 <= nsizeIndex && nsizeIndex < NNEDI_NUM_NSIZE) ? nsizeIndex : -1;
}

int nnedi_nns_index(const int nns) {
    const auto it = std::find(NNEDI_NNS.begin(), NNEDI_NNS.end(), nns);
    return (it == NNEDI_NNS.end()) ? -1 : static_cast<int>(std::distance(NNEDI_NNS.begin(), it));
}

bool validate_param(const RGYFilterNnediWeightsParam& param, std::string *errorMessage) {
    if (nnedi_nsize_index(param.nsize) < 0) {
        set_error(errorMessage, "NNEDI weights: nsize must be one of 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, or 32x4.");
        return false;
    }
    if (nnedi_nns_index(param.nns) < 0) {
        set_error(errorMessage, "NNEDI weights: nns must be one of 16, 32, 64, 128, or 256.");
        return false;
    }
    if (param.quality != VPP_NNEDI_QUALITY_FAST && param.quality != VPP_NNEDI_QUALITY_SLOW) {
        set_error(errorMessage, "NNEDI weights: quality must be fast or slow.");
        return false;
    }
    if (param.prescreen < 2 || 4 < param.prescreen) {
        set_error(errorMessage, "NNEDI weights: supported prescreen values are 2, 3, and 4; prescreen=0/1 use an unsupported prescreener path.");
        return false;
    }
    if (param.errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= param.errortype) {
        set_error(errorMessage, "NNEDI weights: errortype must be abs or square.");
        return false;
    }
    if (param.bitsPerPixel <= 0 || 16 < param.bitsPerPixel) {
        set_error(errorMessage, "NNEDI weights: bitsPerPixel must be in [1,16].");
        return false;
    }
    return true;
}

struct NnediPrescreenerTensorShape {
    int neuronCount = NNEDI_PRESCREENER_HIDDEN_COUNT;
    int sampleCount = NNEDI_PRESCREENER_SAMPLE_COUNT;

    int sourcePackedSampleGroupSize = 8;
    int outputSampleStride() const { return neuronCount; }
    int sourcePackedNeuronStride() const { return sourcePackedSampleGroupSize; }
    int sourcePackedGroupStride() const { return neuronCount * sourcePackedSampleGroupSize; }
};

size_t sourcePrescreenerHiddenWeightOffset(const NnediPrescreenerTensorShape& shape, int neuron, int sample) {
    const int group = sample / shape.sourcePackedSampleGroupSize;
    const int lane = sample % shape.sourcePackedSampleGroupSize;
    return static_cast<size_t>(group * shape.sourcePackedGroupStride() + neuron * shape.sourcePackedNeuronStride() + lane);
}

size_t openclPrescreenerHiddenWeightOffset(const NnediPrescreenerTensorShape& shape, int neuron, int sample) {
    return static_cast<size_t>(sample) * shape.outputSampleStride() + neuron;
}

struct NnediOpenCLPrescreenerLayout {
    NnediPrescreenerTensorShape shape;
    size_t hiddenWeightOffset = 0;
    size_t hiddenWeightFloats = 0;
    size_t hiddenScaleOffset = 0;
    size_t hiddenBiasOffset = 0;
    size_t outputMixOffset = 0;
    size_t outputBiasOffset = 0;
    size_t totalFloats = 0;

    static NnediOpenCLPrescreenerLayout create() {
        NnediOpenCLPrescreenerLayout spec;
        spec.hiddenWeightOffset = 0;
        spec.hiddenWeightFloats = static_cast<size_t>(spec.shape.sampleCount) * spec.shape.neuronCount;
        spec.hiddenScaleOffset = spec.hiddenWeightOffset + spec.hiddenWeightFloats;
        spec.hiddenBiasOffset = spec.hiddenScaleOffset + spec.shape.neuronCount;
        spec.outputMixOffset = spec.hiddenBiasOffset + spec.shape.neuronCount;
        spec.outputBiasOffset = spec.outputMixOffset + static_cast<size_t>(NNEDI_PRESCREENER_OUTPUT_LANES) * spec.shape.neuronCount;
        spec.totalFloats = spec.outputBiasOffset + NNEDI_PRESCREENER_OUTPUT_LANES;
        return spec;
    }
};

struct NnediExternalPrescreenerTensorView {
    const float *data = nullptr;
    NnediPrescreenerTensorShape shape;

    float hiddenWeight(int sample, int neuron) const {
        return data[sourcePrescreenerHiddenWeightOffset(shape, neuron, sample)];
    }
    float hiddenBias(int neuron) const {
        return data[static_cast<size_t>(shape.sampleCount) * shape.neuronCount + neuron];
    }
    float outputMix(int lane, int neuron) const {
        const auto offset = static_cast<size_t>(shape.sampleCount) * shape.neuronCount + shape.neuronCount;
        return data[offset + static_cast<size_t>(neuron) * NNEDI_PRESCREENER_OUTPUT_LANES + lane];
    }
    float outputBias(int lane) const {
        const auto offset = static_cast<size_t>(shape.sampleCount) * shape.neuronCount
            + shape.neuronCount
            + static_cast<size_t>(shape.neuronCount) * NNEDI_PRESCREENER_OUTPUT_LANES;
        return data[offset + lane];
    }
};

struct NnediOpenCLPrescreenerLayoutWriter {
    std::vector<float> *data = nullptr;
    NnediOpenCLPrescreenerLayout layout;

    void resize() const {
        data->assign(layout.totalFloats, 0.0f);
    }
    void setHiddenWeight(int sample, int neuron, float value) const {
        const auto offset = layout.hiddenWeightOffset + openclPrescreenerHiddenWeightOffset(layout.shape, neuron, sample);
        (*data)[offset] = value;
    }
    void setHiddenScale(int neuron, float value) const {
        (*data)[layout.hiddenScaleOffset + neuron] = value;
    }
    void setHiddenBias(int neuron, float value) const {
        (*data)[layout.hiddenBiasOffset + neuron] = value;
    }
    void setOutputMix(int lane, int neuron, float value) const {
        (*data)[layout.outputMixOffset + static_cast<size_t>(lane) * layout.shape.neuronCount + neuron] = value;
    }
    void setOutputBias(int lane, float value) const {
        (*data)[layout.outputBiasOffset + lane] = value;
    }
};

struct NnediPrescreenerInputNormalization {
    std::array<double, NNEDI_PRESCREENER_HIDDEN_COUNT> hiddenMean = {};
    double inputHalf = 0.0;
};

struct NnediPredictorShape {
    int xdia = 0;
    int ydia = 0;
    int sampleCount = 0;
    int softmaxNeuronCount = 0;
    int neuronCount2 = 0;

    static NnediPredictorShape create(const RGYFilterNnediWeightsLayout& layout) {
        NnediPredictorShape shape;
        shape.xdia = layout.xdia;
        shape.ydia = layout.ydia;
        shape.sampleCount = layout.asize;
        shape.softmaxNeuronCount = layout.neurons;
        shape.neuronCount2 = shape.softmaxNeuronCount * 2;
        return shape;
    }
};

enum class NnediPredictorHead : int {
    Softmax = 0,
    Elliott = 1,
};

size_t predictor_source_weight_index(int neuron, int sample, int sampleCount);

struct NnediExternalPredictorTensorView {
    const float *data = nullptr;
    NnediPredictorShape shape;

    int neuronIndex(NnediPredictorHead head, int pair) const {
        return pair + (head == NnediPredictorHead::Elliott ? shape.softmaxNeuronCount : 0);
    }
    float weight(NnediPredictorHead head, int pair, int sample) const {
        return data[predictor_source_weight_index(neuronIndex(head, pair), sample, shape.sampleCount)];
    }
    float bias(NnediPredictorHead head, int pair) const {
        return data[static_cast<size_t>(shape.neuronCount2) * shape.sampleCount + neuronIndex(head, pair)];
    }
};

struct NnediWeightSpan {
    size_t offsetFloats = 0;
    size_t floatCount = 0;
};

struct NnediWeightFileCatalogKey {
    int nsizeIndex = 0;
    int nnsIndex = 0;
};

NnediPredictorShape nnedi_predictor_shape_for_catalog_key(const NnediWeightFileCatalogKey key) {
    NnediPredictorShape shape;
    shape.xdia = NNEDI_XDIA[key.nsizeIndex];
    shape.ydia = NNEDI_YDIA[key.nsizeIndex];
    shape.softmaxNeuronCount = NNEDI_NNS[key.nnsIndex];
    shape.neuronCount2 = shape.softmaxNeuronCount * 2;
    shape.sampleCount = shape.xdia * shape.ydia;
    return shape;
}

int nnedi_predictor_plane_float_count_for_catalog_key(const NnediWeightFileCatalogKey key) {
    const auto shape = nnedi_predictor_shape_for_catalog_key(key);
    return (shape.sampleCount + 1) * shape.neuronCount2;
}

struct NnediWeightFileSectionCursor {
    size_t cursorFloats = 0;

    NnediWeightSpan take(size_t floatCount) {
        const NnediWeightSpan span{ cursorFloats, floatCount };
        cursorFloats += floatCount;
        return span;
    }
};

struct NnediPredictorCatalogLayout {
    NnediPredictorShape selectedShape;
    int selectedPlaneFloats = 0;
    int selectedPairOffsetFloats = 0;
    int totalFloats = 0;

    static NnediPredictorCatalogLayout create(const NnediWeightFileCatalogKey selected) {
        NnediPredictorCatalogLayout catalog;
        for (int nnsIndex = 0; nnsIndex < NNEDI_NUM_NNS; nnsIndex++) {
            for (int nsizeIndex = 0; nsizeIndex < NNEDI_NUM_NSIZE; nsizeIndex++) {
                const NnediWeightFileCatalogKey key{ nsizeIndex, nnsIndex };
                const int planeFloats = nnedi_predictor_plane_float_count_for_catalog_key(key);
                if (key.nsizeIndex == selected.nsizeIndex && key.nnsIndex == selected.nnsIndex) {
                    catalog.selectedShape = nnedi_predictor_shape_for_catalog_key(key);
                    catalog.selectedPlaneFloats = planeFloats;
                    catalog.selectedPairOffsetFloats = catalog.totalFloats;
                }
                catalog.totalFloats += planeFloats * 2;
            }
        }
        return catalog;
    }
};

struct NnediWeightFileMap {
    NnediPredictorShape predictorShape;
    int legacyPrescreenerFloats = NNEDI_LEGACY_PRESCREENER_FLOATS;
    int prescreenerNetworkFloats = NNEDI_PRESCREENER_NETWORK_FLOATS;
    int predictorPlaneFloats = 0;
    int predictorCatalogFloats = 0;
    int predictorCatalogOffsetFloats = 0;
    size_t rawWeightFloatCount = 0;
    NnediWeightSpan rawPrescreener = {};
    std::array<NnediWeightSpan, 2> rawPredictor = {};

    static NnediWeightFileMap create(int nsizeIndex, int nnsIndex, int prescreen, VppNnediErrorType errortype) {
        const auto catalog = NnediPredictorCatalogLayout::create(NnediWeightFileCatalogKey{ nsizeIndex, nnsIndex });

        NnediWeightFileMap map;
        map.predictorShape = catalog.selectedShape;
        map.predictorPlaneFloats = catalog.selectedPlaneFloats;
        map.predictorCatalogFloats = catalog.totalFloats;
        map.predictorCatalogOffsetFloats = catalog.selectedPairOffsetFloats;

        NnediWeightFileSectionCursor file;
        file.take(map.legacyPrescreenerFloats);

        std::array<NnediWeightSpan, 3> prescreenerVariants = {};
        for (auto& span : prescreenerVariants) {
            span = file.take(map.prescreenerNetworkFloats);
        }
        if (2 <= prescreen && prescreen <= 4) {
            map.rawPrescreener = prescreenerVariants[prescreen - 2];
        }

        for (int errorPlane = 0; errorPlane < 2; errorPlane++) {
            const auto predictorCatalog = file.take(map.predictorCatalogFloats);
            if (errorPlane == static_cast<int>(errortype)) {
                map.rawPredictor[0] = NnediWeightSpan{
                    predictorCatalog.offsetFloats + static_cast<size_t>(map.predictorCatalogOffsetFloats),
                    static_cast<size_t>(map.predictorPlaneFloats)
                };
                map.rawPredictor[1] = NnediWeightSpan{
                    map.rawPredictor[0].offsetFloats + static_cast<size_t>(map.predictorPlaneFloats),
                    static_cast<size_t>(map.predictorPlaneFloats)
                };
            }
        }
        map.rawWeightFloatCount = file.cursorFloats;
        return map;
    }
};

struct NnediPredictorNormalizationModel {
    std::vector<double> softmaxCommonModeBySample;
    std::vector<double> perNeuronDcOffset;
    double softmaxBiasCommonMode = 0.0;
};

struct NnediPredictorTransformSpec {
    NnediPredictorShape shape;

    static NnediPredictorTransformSpec create(const RGYFilterNnediWeightsLayout& layout) {
        NnediPredictorTransformSpec spec;
        spec.shape = NnediPredictorShape::create(layout);
        return spec;
    }
};

struct NnediOpenCLPredictorLayout {
    size_t bodyOffset = 0;
    size_t biasOffset = 0;
    size_t bodyFloats = 0;
    size_t biasFloats = 0;
    size_t totalFloats = 0;

    static NnediOpenCLPredictorLayout create(const NnediPredictorShape& shape, VppNnediQuality quality) {
        NnediOpenCLPredictorLayout layout;
        layout.bodyFloats = static_cast<size_t>((int)quality) * shape.softmaxNeuronCount * shape.sampleCount * 2;
        layout.biasOffset = layout.bodyOffset + layout.bodyFloats;
        layout.biasFloats = static_cast<size_t>((int)quality) * shape.softmaxNeuronCount * 2;
        layout.totalFloats = layout.biasOffset + layout.biasFloats;
        return layout;
    }
};

size_t predictor_fp32_body_index(const NnediPredictorShape& shape, int q, int pair, int sample);
size_t predictor_fp32_bias_index(const NnediPredictorShape& shape, const NnediOpenCLPredictorLayout& layout, int q, int pair);

struct NnediOpenCLPredictorLayoutWriter {
    std::vector<float> *data = nullptr;
    NnediPredictorShape shape;
    NnediOpenCLPredictorLayout layout;

    void resize() const {
        data->assign(layout.totalFloats, 0.0f);
    }
    void setBody(int q, int pair, int sample, NnediPredictorHead head, float value) const {
        const auto offset = predictor_fp32_body_index(shape, q, pair, sample) + static_cast<int>(head);
        (*data)[offset] = value;
    }
    void setBias(int q, int pair, NnediPredictorHead head, float value) const {
        const auto offset = predictor_fp32_bias_index(shape, layout, q, pair) + static_cast<int>(head);
        (*data)[offset] = value;
    }
};

size_t predictor_source_weight_index(int neuron, int sample, int sampleCount) {
    return static_cast<size_t>(neuron) * sampleCount + sample;
}

size_t predictor_fp32_body_index(const NnediPredictorShape& shape, int q, int pair, int sample) {
    const int block = pair / NNEDI_PREDICTOR_FP32_LANES;
    const int lane = pair - block * NNEDI_PREDICTOR_FP32_LANES;
    const int blockCount = shape.softmaxNeuronCount / NNEDI_PREDICTOR_FP32_LANES;
    // fp32 predictor body is laid out as [q][neuronBlock16][sample][lane][softmax/elliott].
    // The kernel's tx=0..15 lanes then read adjacent float2 values for each sample, which is
    // the main GPU coalescing optimization for the predictor dot product.
    return (((static_cast<size_t>(q) * blockCount + block) * shape.sampleCount + sample) * NNEDI_PREDICTOR_FP32_LANES + lane) * 2;
}

size_t predictor_fp32_bias_index(const NnediPredictorShape& shape, const NnediOpenCLPredictorLayout& layout, int q, int pair) {
    return layout.biasOffset + (static_cast<size_t>(q) * shape.softmaxNeuronCount + pair) * 2;
}

uint64_t fnv1a64_update_byte(uint64_t hash, uint8_t value) {
    hash ^= value;
    hash *= FNV1A64_PRIME;
    return hash;
}

uint64_t fnv1a64_update_u64(uint64_t hash, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        hash = fnv1a64_update_byte(hash, static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
    return hash;
}

uint64_t fnv1a64_update_i32(uint64_t hash, int value) {
    const auto uvalue = static_cast<uint32_t>(value);
    for (int i = 0; i < 4; i++) {
        hash = fnv1a64_update_byte(hash, static_cast<uint8_t>((uvalue >> (i * 8)) & 0xff));
    }
    return hash;
}

uint64_t fnv1a64_float_buffer(const std::vector<float>& buffer) {
    uint64_t hash = FNV1A64_OFFSET_BASIS;
    for (const auto value : buffer) {
        std::array<uint8_t, sizeof(float)> bytes = {};
        std::memcpy(bytes.data(), &value, sizeof(value));
        for (const auto byte : bytes) {
            hash = fnv1a64_update_byte(hash, byte);
        }
    }
    return hash;
}

RGYFilterNnediFloatBufferDigest digest_float_buffer(const std::vector<float>& buffer) {
    RGYFilterNnediFloatBufferDigest digest;
    digest.floatCount = buffer.size();
    digest.fnv1a64 = fnv1a64_float_buffer(buffer);
    if (buffer.empty()) {
        return digest;
    }

    constexpr std::array<size_t, 8> sampleNumerators = { 0, 1, 2, 3, 4, 8, 16, 32 };
    const auto denom = sampleNumerators.back();
    for (size_t i = 0; i < digest.samples.size(); i++) {
        const auto index = (buffer.size() - 1) * sampleNumerators[i] / denom;
        digest.samples[i].index = index;
        digest.samples[i].value = buffer[index];
    }
    return digest;
}

uint64_t fnv1a64_summary(const RGYFilterNnediTransformedWeights& weights, const RGYFilterNnediFloatBufferDigest& prescreener, const RGYFilterNnediFloatBufferDigest& predictor) {
    uint64_t hash = FNV1A64_OFFSET_BASIS;
    hash = fnv1a64_update_i32(hash, weights.param.nsize);
    hash = fnv1a64_update_i32(hash, weights.param.nns);
    hash = fnv1a64_update_i32(hash, (int)weights.param.quality);
    hash = fnv1a64_update_i32(hash, weights.param.prescreen);
    hash = fnv1a64_update_i32(hash, (int)weights.param.errortype);
    hash = fnv1a64_update_i32(hash, weights.param.bitsPerPixel);
    hash = fnv1a64_update_i32(hash, weights.layout.xdia);
    hash = fnv1a64_update_i32(hash, weights.layout.ydia);
    hash = fnv1a64_update_i32(hash, weights.layout.neurons);
    hash = fnv1a64_update_i32(hash, weights.layout.asize);
    hash = fnv1a64_update_u64(hash, static_cast<uint64_t>(prescreener.floatCount));
    hash = fnv1a64_update_u64(hash, prescreener.fnv1a64);
    hash = fnv1a64_update_u64(hash, static_cast<uint64_t>(predictor.floatCount));
    hash = fnv1a64_update_u64(hash, predictor.fnv1a64);
    return hash;
}

NnediPrescreenerInputNormalization build_prescreener_input_normalization(const NnediExternalPrescreenerTensorView& src, int bitsPerPixel) {
    NnediPrescreenerInputNormalization normalization;
    const int prescreenerBits = std::min(bitsPerPixel, 16);
    normalization.inputHalf = (static_cast<double>((1u << prescreenerBits) - 1u)) / 2.0;

    for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
        double sum = 0.0;
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            sum += src.hiddenWeight(sample, neuron);
        }
        normalization.hiddenMean[neuron] = sum / static_cast<double>(src.shape.sampleCount);
    }
    return normalization;
}

void write_prescreener_hidden_layer(const NnediOpenCLPrescreenerLayoutWriter& dst, const NnediExternalPrescreenerTensorView& src, const NnediPrescreenerInputNormalization& normalization) {
    for (int sample = 0; sample < src.shape.sampleCount; sample++) {
        for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
            const auto value = static_cast<float>((src.hiddenWeight(sample, neuron) - normalization.hiddenMean[neuron]) / normalization.inputHalf);
            dst.setHiddenWeight(sample, neuron, value);
        }
    }
    for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
        dst.setHiddenScale(neuron, 1.0f);
        dst.setHiddenBias(neuron, src.hiddenBias(neuron));
    }
}

void write_prescreener_output_layer(const NnediOpenCLPrescreenerLayoutWriter& dst, const NnediExternalPrescreenerTensorView& src) {
    for (int lane = 0; lane < NNEDI_PRESCREENER_OUTPUT_LANES; lane++) {
        for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
            dst.setOutputMix(lane, neuron, src.outputMix(lane, neuron));
        }
        dst.setOutputBias(lane, src.outputBias(lane));
    }
}

void transform_prescreener_fp32(std::vector<float>& dst, const float *rawWeights, const RGYFilterNnediWeightsParam& param, const RGYFilterNnediWeightsLayout& layout) {
    const auto openclLayout = NnediOpenCLPrescreenerLayout::create();
    const NnediExternalPrescreenerTensorView src{ rawWeights + layout.rawPrescreenerOffsetFloats, openclLayout.shape };
    const NnediOpenCLPrescreenerLayoutWriter out{ &dst, openclLayout };
    const auto normalization = build_prescreener_input_normalization(src, param.bitsPerPixel);

    out.resize();
    write_prescreener_hidden_layer(out, src, normalization);
    write_prescreener_output_layer(out, src);
}

NnediPredictorNormalizationModel build_predictor_normalization_model(const NnediExternalPredictorTensorView& src) {
    NnediPredictorNormalizationModel model;
    model.softmaxCommonModeBySample.assign(src.shape.sampleCount, 0.0);
    model.perNeuronDcOffset.assign(src.shape.neuronCount2, 0.0);

    for (const auto head : { NnediPredictorHead::Softmax, NnediPredictorHead::Elliott }) {
        for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
            const int neuron = src.neuronIndex(head, pair);
            for (int sample = 0; sample < src.shape.sampleCount; sample++) {
                model.perNeuronDcOffset[neuron] += src.weight(head, pair, sample);
            }
            model.perNeuronDcOffset[neuron] /= static_cast<double>(src.shape.sampleCount);
        }
    }

    for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
        const int neuron = src.neuronIndex(NnediPredictorHead::Softmax, pair);
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            const double centered = src.weight(NnediPredictorHead::Softmax, pair, sample) - model.perNeuronDcOffset[neuron];
            model.softmaxCommonModeBySample[sample] += centered;
        }
        model.softmaxBiasCommonMode += src.bias(NnediPredictorHead::Softmax, pair);
    }

    for (auto& offset : model.softmaxCommonModeBySample) {
        offset /= static_cast<double>(src.shape.softmaxNeuronCount);
    }
    model.softmaxBiasCommonMode /= static_cast<double>(src.shape.softmaxNeuronCount);
    return model;
}

float predictor_normalized_weight(const NnediExternalPredictorTensorView& src, const NnediPredictorNormalizationModel& model, NnediPredictorHead head, int pair, int sample) {
    const int neuron = src.neuronIndex(head, pair);
    float centered = src.weight(head, pair, sample) - static_cast<float>(model.perNeuronDcOffset[neuron]);
    if (head == NnediPredictorHead::Softmax) {
        centered -= static_cast<float>(model.softmaxCommonModeBySample[sample]);
    }
    return centered;
}

float predictor_normalized_bias(const NnediExternalPredictorTensorView& src, const NnediPredictorNormalizationModel& model, NnediPredictorHead head, int pair) {
    const auto bias = src.bias(head, pair);
    if (head == NnediPredictorHead::Softmax) {
        return static_cast<float>(bias - model.softmaxBiasCommonMode);
    }
    return bias;
}

void write_predictor_quality_plane(const NnediOpenCLPredictorLayoutWriter& dst, int q, const NnediExternalPredictorTensorView& src, const NnediPredictorNormalizationModel& model) {
    for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            dst.setBody(q, pair, sample, NnediPredictorHead::Softmax,
                predictor_normalized_weight(src, model, NnediPredictorHead::Softmax, pair, sample));
            dst.setBody(q, pair, sample, NnediPredictorHead::Elliott,
                predictor_normalized_weight(src, model, NnediPredictorHead::Elliott, pair, sample));
        }
        dst.setBias(q, pair, NnediPredictorHead::Softmax,
            predictor_normalized_bias(src, model, NnediPredictorHead::Softmax, pair));
        dst.setBias(q, pair, NnediPredictorHead::Elliott,
            predictor_normalized_bias(src, model, NnediPredictorHead::Elliott, pair));
    }
}

bool transform_predictor_fp32(std::vector<float>& dst, const float *rawWeights, const RGYFilterNnediWeightsParam& param, const RGYFilterNnediWeightsLayout& layout, std::string *errorMessage) {
    const auto spec = NnediPredictorTransformSpec::create(layout);
    const auto fp32Layout = NnediOpenCLPredictorLayout::create(spec.shape, param.quality);
    const NnediOpenCLPredictorLayoutWriter out{ &dst, spec.shape, fp32Layout };

    out.resize();
    for (int q = 0; q < (int)param.quality; q++) {
        const NnediExternalPredictorTensorView src{ rawWeights + layout.rawPredictorOffsetFloats[q], spec.shape };
        write_predictor_quality_plane(out, q, src, build_predictor_normalization_model(src));
    }
    if (dst.size() != fp32Layout.totalFloats) {
        set_error(errorMessage, "NNEDI weights: invalid fp32 predictor layout.");
        return false;
    }
    return true;
}

void publish_weight_file_map(RGYFilterNnediWeightsLayout& layout, const NnediWeightFileMap& map, VppNnediNSize nsize, int nns) {
    layout = RGYFilterNnediWeightsLayout();
    layout.nsize = nsize;
    layout.nns = nns;
    layout.xdia = map.predictorShape.xdia;
    layout.ydia = map.predictorShape.ydia;
    layout.neurons = map.predictorShape.softmaxNeuronCount;
    layout.asize = map.predictorShape.sampleCount;
    layout.legacyPrescreenerFloats = map.legacyPrescreenerFloats;
    layout.prescreenerNetworkFloats = map.prescreenerNetworkFloats;
    layout.predictorPlaneFloats = map.predictorPlaneFloats;
    layout.predictorCatalogFloats = map.predictorCatalogFloats;
    layout.predictorCatalogOffsetFloats = map.predictorCatalogOffsetFloats;
    layout.rawWeightFloatCount = map.rawWeightFloatCount;
    layout.rawPrescreenerOffsetFloats = map.rawPrescreener.offsetFloats;
    layout.rawPredictorOffsetFloats[0] = map.rawPredictor[0].offsetFloats;
    layout.rawPredictorOffsetFloats[1] = map.rawPredictor[1].offsetFloats;
}

} // namespace

bool rgy_filter_nnedi_weights_layout(RGYFilterNnediWeightsLayout& layout, VppNnediNSize nsize, int nns, int prescreen, VppNnediErrorType errortype, std::string *errorMessage) {
    const auto nsizeIndex = nnedi_nsize_index(nsize);
    if (nsizeIndex < 0) {
        set_error(errorMessage, "NNEDI weights layout: nsize must be one of 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, or 32x4.");
        return false;
    }
    const auto nnsIndex = nnedi_nns_index(nns);
    if (nnsIndex < 0) {
        set_error(errorMessage, "NNEDI weights layout: nns must be one of 16, 32, 64, 128, or 256.");
        return false;
    }
    if (prescreen < 0 || 4 < prescreen) {
        set_error(errorMessage, "NNEDI weights layout: prescreen must be in [0,4].");
        return false;
    }
    if (errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= errortype) {
        set_error(errorMessage, "NNEDI weights layout: errortype must be abs or square.");
        return false;
    }

    const auto weightFileMap = NnediWeightFileMap::create(nsizeIndex, nnsIndex, prescreen, errortype);
    publish_weight_file_map(layout, weightFileMap, nsize, nns);
    return true;
}

bool rgy_filter_nnedi_transform_weights(RGYFilterNnediTransformedWeights& dst, const float *rawWeights, size_t rawWeightFloatCount, const RGYFilterNnediWeightsParam& param, std::string *errorMessage) {
    if (!validate_param(param, errorMessage)) {
        return false;
    }
    if (!rawWeights) {
        set_error(errorMessage, "NNEDI weights: rawWeights is null.");
        return false;
    }

    RGYFilterNnediWeightsLayout layout;
    if (!rgy_filter_nnedi_weights_layout(layout, param.nsize, param.nns, param.prescreen, param.errortype, errorMessage)) {
        return false;
    }
    if (rawWeightFloatCount < layout.rawWeightFloatCount) {
        set_error(errorMessage, "NNEDI weights: raw weights buffer is smaller than nnedi3_weights.bin layout.");
        return false;
    }

    RGYFilterNnediTransformedWeights tmp;
    tmp.param = param;
    tmp.layout = layout;
    transform_prescreener_fp32(tmp.prescreenerFp32, rawWeights, param, layout);
    if (!transform_predictor_fp32(tmp.predictorFp32, rawWeights, param, layout, errorMessage)) {
        return false;
    }

    dst = std::move(tmp);
    return true;
}

RGYFilterNnediWeightsSummary rgy_filter_nnedi_weights_summary(const RGYFilterNnediTransformedWeights& weights) {
    RGYFilterNnediWeightsSummary summary;
    summary.param = weights.param;
    summary.layout = weights.layout;
    summary.prescreenerFp32 = digest_float_buffer(weights.prescreenerFp32);
    summary.predictorFp32 = digest_float_buffer(weights.predictorFp32);
    summary.combinedFnv1a64 = fnv1a64_summary(weights, summary.prescreenerFp32, summary.predictorFp32);
    return summary;
}

RGYFilterNnediWeightsDefaultSampleResult rgy_filter_nnedi_weights_default_samples(const float *rawWeights, size_t rawWeightFloatCount) {
    RGYFilterNnediWeightsDefaultSampleResult result;
    std::string error;

    RGYFilterNnediWeightsParam defaultParam;
    defaultParam.nsize = VPP_NNEDI_NSIZE_16x6;
    defaultParam.nns = 32;
    defaultParam.quality = VPP_NNEDI_QUALITY_FAST;
    RGYFilterNnediTransformedWeights defaultWeights;
    if (!rgy_filter_nnedi_transform_weights(defaultWeights, rawWeights, rawWeightFloatCount, defaultParam, &error)) {
        result.message = error;
        return result;
    }
    result.defaultSlower = rgy_filter_nnedi_weights_summary(defaultWeights);

    RGYFilterNnediWeightsParam chromaParam;
    chromaParam.nsize = VPP_NNEDI_NSIZE_8x4;
    chromaParam.nns = 16;
    chromaParam.quality = VPP_NNEDI_QUALITY_FAST;
    RGYFilterNnediTransformedWeights chromaWeights;
    if (!rgy_filter_nnedi_transform_weights(chromaWeights, rawWeights, rawWeightFloatCount, chromaParam, &error)) {
        result.message = error;
        return result;
    }
    result.chroma = rgy_filter_nnedi_weights_summary(chromaWeights);

    result.success = true;
    result.message = "ok";
    return result;
}

RGYFilterNnediWeightsSelfCheckResult rgy_filter_nnedi_weights_self_check() {
    RGYFilterNnediWeightsSelfCheckResult result;
    std::string error;
    if (!rgy_filter_nnedi_weights_layout(result.defaultSlower, VPP_NNEDI_NSIZE_16x6, 32, NNEDI_DEFAULT_PRESCREEN, NNEDI_DEFAULT_ERRORTYPE, &error)) {
        result.message = error;
        return result;
    }
    if (!rgy_filter_nnedi_weights_layout(result.chroma, VPP_NNEDI_NSIZE_8x4, 16, NNEDI_DEFAULT_PRESCREEN, NNEDI_DEFAULT_ERRORTYPE, &error)) {
        result.message = error;
        return result;
    }

    if (result.defaultSlower.xdia != 16 || result.defaultSlower.ydia != 6 || result.defaultSlower.neurons != 32 || result.defaultSlower.asize != 96 || result.defaultSlower.predictorPlaneFloats != 6208) {
        result.message = "NNEDI weights self-check: default Slower dimensions mismatch.";
        return result;
    }
    if (result.chroma.xdia != 8 || result.chroma.ydia != 4 || result.chroma.neurons != 16 || result.chroma.asize != 32 || result.chroma.predictorPlaneFloats != 1056) {
        result.message = "NNEDI weights self-check: chroma dimensions mismatch.";
        return result;
    }
    if (result.defaultSlower.rawPrescreenerOffsetFloats != NNEDI_LEGACY_PRESCREENER_FLOATS || result.chroma.rawPrescreenerOffsetFloats != NNEDI_LEGACY_PRESCREENER_FLOATS) {
        result.message = "NNEDI weights self-check: prescreener offset mismatch.";
        return result;
    }
    if (result.defaultSlower.rawPredictorOffsetFloats[1] != result.defaultSlower.rawPredictorOffsetFloats[0] + static_cast<size_t>(result.defaultSlower.predictorPlaneFloats)) {
        result.message = "NNEDI weights self-check: default predictor plane offset mismatch.";
        return result;
    }
    if (result.chroma.rawPredictorOffsetFloats[1] != result.chroma.rawPredictorOffsetFloats[0] + static_cast<size_t>(result.chroma.predictorPlaneFloats)) {
        result.message = "NNEDI weights self-check: chroma predictor plane offset mismatch.";
        return result;
    }
    if (result.defaultSlower.rawWeightFloatCount != result.chroma.rawWeightFloatCount || result.defaultSlower.rawWeightFloatCount * sizeof(float) != WEIGHTBIN_FILE_SIZE) {
        result.message = "NNEDI weights self-check: raw nnedi3_weights.bin size mismatch.";
        return result;
    }

    result.success = true;
    result.message = "ok";
    return result;
}
