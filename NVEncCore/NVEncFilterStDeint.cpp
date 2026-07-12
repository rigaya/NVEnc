// -----------------------------------------------------------------------------------------
//     NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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

#include "NVEncFilterStDeint.h"
#include <cuda_runtime.h>
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <algorithm>

static inline uint8_t stdeint_clamp_u8(int value) {
    return (uint8_t)(value < 0 ? 0 : (value > 255 ? 255 : value));
}

static inline float stdeint_clampf(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static const TCHAR *stdeint_cx_desc_or_unknown(const CX_DESC *list, int value) {
    const auto desc = get_cx_desc(list, value);
    return (desc != nullptr) ? desc : _T("unknown");
}

static bool stdeint_matrix_to_coeff_id(CspMatrix matrix, int inputHeight, int& matrixSel) {
    if (matrix == RGY_MATRIX_AUTO || (int)matrix == COLOR_VALUE_AUTO_RESOLUTION) {
        matrixSel = (inputHeight <= 576) ? 601 : 709;
        return true;
    }
    switch (matrix) {
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:
        matrixSel = 601;
        return true;
    case RGY_MATRIX_BT709:
        matrixSel = 709;
        return true;
    case RGY_MATRIX_BT2020_NCL:
        matrixSel = 2020;
        return true;
    default:
        return false;
    }
}

static bool stdeint_supported_colorrange(CspColorRange range) {
    return range == RGY_COLORRANGE_AUTO
        || range == RGY_COLORRANGE_LIMITED
        || range == RGY_COLORRANGE_FULL;
}

static RGYOnnxRTProvider stdeint_provider(const tstring& provider) {
    const auto normalized = tolowercase(provider);
    if (normalized == _T("tensorrt")) return RGYOnnxRTProvider::TensorRT;
    if (normalized == _T("cuda")) return RGYOnnxRTProvider::Cuda;
    return RGYOnnxRTProvider::Auto;
}

NVEncFilterStDeint::NVEncFilterStDeint() :
    NVEncFilter(), m_ov(), m_width(0), m_height(0), m_mode(VppStDeintMode::Bob), m_defaultTff(true),
    m_havePrevTimestamp(false), m_prevTimestamp(0), m_prevDuration(0),
    m_yOff(0), m_yScale(1), m_yRange(255), m_cOff(128), m_cScale(1), m_cRange(255),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_inputBuf(), m_outputBuf(), m_weaveBuf(), m_inputStaging(), m_outputStaging(),
    m_inputDevice(), m_outputDevice(), m_modelPath(), m_provider(RGYOnnxRTProvider::Auto),
    m_deviceID(-1), m_cudaPathTried(false), m_cudaPath(false) {
    m_name = _T("stdeint");
}

NVEncFilterStDeint::~NVEncFilterStDeint() {
    close();
}

void NVEncFilterStDeint::close() {
    m_ov.reset();
    m_inputStaging.reset();
    m_outputStaging.clear();
    m_inputDevice.reset();
    m_outputDevice.reset();
    m_inputBuf.clear();
    m_outputBuf.clear();
    m_weaveBuf.clear();
    m_frameBuf.clear();
    m_havePrevTimestamp = false;
    m_cudaPathTried = false;
    m_cudaPath = false;
}

tstring NVEncFilterParamStDeint::print() const {
    return strsprintf(_T("stdeint: %s, mode %s, provider %s, precision %s, colormatrix %s, colorrange %s"), modelFile.c_str(),
        get_cx_desc(list_vpp_stdeint_mode, (int)mode), provider.c_str(), precision.c_str(),
        stdeint_cx_desc_or_unknown(list_colormatrix, colormatrix), stdeint_cx_desc_or_unknown(list_colorrange, colorrange));
}

void NVEncFilterStDeint::setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax) {
    float kr = 0.2126f, kb = 0.0722f;
    if (matrixSel == 601)  { kr = 0.299f;  kb = 0.114f; }
    if (matrixSel == 2020) { kr = 0.2627f; kb = 0.0593f; }
    const float kg = 1.0f - kr - kb;
    m_matVR = 2.0f * (1.0f - kr);
    m_matUG = -2.0f * kb * (1.0f - kb) / kg;
    m_matVG = -2.0f * kr * (1.0f - kr) / kg;
    m_matUB = 2.0f * (1.0f - kb);
    m_matRY = kr;                            m_matGY = kg;                            m_matBY = kb;
    m_matRU = -kr / (2.0f * (1.0f - kb));    m_matGU = -kg / (2.0f * (1.0f - kb));    m_matBU = 0.5f;
    m_matRV = 0.5f;                          m_matGV = -kg / (2.0f * (1.0f - kr));    m_matBV = -kb / (2.0f * (1.0f - kr));
    m_yOff   = rangeTV ? (16.0f  * pixMax / 255.0f) : 0.0f;
    m_yRange = rangeTV ? (219.0f * pixMax / 255.0f) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * pixMax / 255.0f) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * pixMax / 255.0f) : (float)pixMax;
    m_cScale = 1.0f / m_cRange;
}

RGY_ERR NVEncFilterStDeint::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamStDeint>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOnnxRTCUDA::available()) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: this build was compiled without ONNX Runtime CUDA support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: model= (a registered model name or ST-DeInt .onnx path) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        const auto err = registry.load(PathCombineS(prm->modelDir, _T("stdeint_ov_models.json")), m_pLog);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (!registry.find(prm->modelFile)) {
            AddMessage(RGY_LOG_ERROR, _T("stdeint: model \"%s\" not found in stdeint_ov_models.json\n"), prm->modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->modelFile = registry.resolveModelPath(prm->modelFile);
    }
    if (!rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: model file not found: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    const auto inputCsp = prm->frameIn.csp;
    if ((inputCsp != RGY_CSP_YV12 && inputCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inputCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_width = prm->frameIn.width;
    m_height = prm->frameIn.height;
    if (m_height < 4 || (m_height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: height must be an even value of at least 4 (got %d).\n"), m_height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppStDeintMode::Bob && prm->mode != VppStDeintMode::Normal) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: invalid output mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    int matrixSel = 0;
    if (!stdeint_matrix_to_coeff_id(prm->colormatrix, m_height, matrixSel)) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: unsupported colormatrix %s.\n"),
            stdeint_cx_desc_or_unknown(list_colormatrix, prm->colormatrix));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!stdeint_supported_colorrange(prm->colorrange)) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: unsupported colorrange %s.\n"),
            stdeint_cx_desc_or_unknown(list_colorrange, prm->colorrange));
        return RGY_ERR_UNSUPPORTED;
    }
    m_mode = prm->mode;
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    m_ov = std::make_unique<RGYOnnxRTCUDA>();
    m_deviceID = prm->deviceID;
    if (m_deviceID < 0) {
        cudaGetDevice(&m_deviceID);
    }
    m_modelPath = prm->modelFile;
    m_provider = stdeint_provider(prm->provider);
    tstring errorMessage;
    auto err = m_ov->init(m_modelPath, m_deviceID, m_provider, m_height, m_width, errorMessage);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to load/compile model: %s\n"), errorMessage.c_str());
        return err;
    }
    if (tolowercase(prm->provider) == _T("tensorrt") && m_ov->providerName() != _T("tensorrt")) {
        AddMessage(RGY_LOG_WARN, _T("stdeint: TensorRT provider is unavailable; falling back to CUDA: %s\n"),
            m_ov->lastError().c_str());
    }
    if (m_ov->inChannels() != 3 || m_ov->outChannels() != 6) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: invalid model (expected 3ch input / 6ch output, got %dch / %dch).\n"),
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->outHeight() == m_height && m_ov->outWidth() == m_width) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: this model contains the legacy ONNX weave output; re-export it with the current export_stdeint.py.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->outHeight() != m_height / 2 || m_ov->outWidth() != m_width) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: restoration output must be 6ch with half input height (expected %dx%d, got %dx%d).\n"),
            m_width, m_height / 2, m_ov->outWidth(), m_ov->outHeight());
        return RGY_ERR_UNSUPPORTED;
    }

    setupColorCoeffs(matrixSel, prm->colorrange != RGY_COLORRANGE_FULL, 255);

    prm->frameOut.csp = inputCsp;
    prm->frameOut.width = m_width;
    prm->frameOut.height = m_height;
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    const int outputCount = (m_mode == VppStDeintMode::Bob) ? 2 : 1;
    if (m_mode == VppStDeintMode::Bob) {
        prm->baseFps *= 2;
    }

    err = AllocFrameBuf(prm->frameOut, outputCount);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const size_t plane = (size_t)m_width * m_height;
    m_inputBuf.resize(3 * plane);
    m_outputBuf.resize(m_ov->outElemCount());
    m_weaveBuf.resize(3 * plane);
    m_cudaPathTried = false;
    m_cudaPath = false;
    m_inputDevice = std::make_unique<CUMemBuf>(m_inputBuf.size() * sizeof(float));
    m_outputDevice = std::make_unique<CUMemBuf>(m_outputBuf.size() * sizeof(float));
    if (m_inputDevice->alloc() != RGY_ERR_NONE || m_outputDevice->alloc() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("stdeint: CUDA zero-copy buffers could not be allocated; using host path.\n"));
        m_inputDevice.reset();
        m_outputDevice.reset();
        m_cudaPathTried = true;
    }
    m_inputStaging = std::make_unique<CUFrameBuf>();
    if (m_inputStaging->allocHost(m_width, m_height, inputCsp) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to allocate input staging frame buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_outputStaging.clear();
    for (int i = 0; i < outputCount; i++) {
        auto staging = std::make_unique<CUFrameBuf>();
        if (staging->allocHost(m_width, m_height, inputCsp) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to allocate output staging frame buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_outputStaging.push_back(std::move(staging));
    }

    m_havePrevTimestamp = false;
    m_param = prm;
    setFilterInfo(prm->print() + _T(", path host"));
    AddMessage(RGY_LOG_DEBUG, _T("stdeint: %s, %dx%d, mode %s, provider %s, path host.\n"),
        prm->modelFile.c_str(), m_width, m_height, get_cx_desc(list_vpp_stdeint_mode, (int)m_mode),
        m_ov->providerName().c_str());
    return RGY_ERR_NONE;
}

void NVEncFilterStDeint::yuvToRGB(const RGYFrameInfo& input, float *dst) {
    const size_t plane = (size_t)m_width * m_height;
    const bool nv12 = input.csp == RGY_CSP_NV12;
    const int chromaWidth = m_width / 2;
    const int chromaHeight = m_height / 2;
    const uint8_t *uPlane = input.ptr[1];
    const uint8_t *vPlane = nv12 ? input.ptr[1] + 1 : input.ptr[2];
    const int chromaStride = nv12 ? 2 : 1;
    const int uPitch = input.pitch[1];
    const int vPitch = nv12 ? input.pitch[1] : input.pitch[2];
    float *red = dst;
    float *green = dst + plane;
    float *blue = dst + 2 * plane;
    for (int y = 0; y < m_height; y++) {
        const uint8_t *yRow = input.ptr[0] + (size_t)y * input.pitch[0];
        const int cy = std::min(y / 2, chromaHeight - 1);
        for (int x = 0; x < m_width; x++) {
            const int cx = std::min(x / 2, chromaWidth - 1);
            const float yn = ((float)yRow[x] - m_yOff) * m_yScale;
            const float un = ((float)uPlane[(size_t)cy * uPitch + (size_t)cx * chromaStride] - m_cOff) * m_cScale;
            const float vn = ((float)vPlane[(size_t)cy * vPitch + (size_t)cx * chromaStride] - m_cOff) * m_cScale;
            const size_t index = (size_t)y * m_width + x;
            red[index] = stdeint_clampf(yn + m_matVR * vn, 0.0f, 1.0f);
            green[index] = stdeint_clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
            blue[index] = stdeint_clampf(yn + m_matUB * un, 0.0f, 1.0f);
        }
    }
}

void NVEncFilterStDeint::rgbToYUV(const RGYFrameInfo& output, const float *src) {
    const size_t plane = (size_t)m_width * m_height;
    const bool nv12 = output.csp == RGY_CSP_NV12;
    const int chromaWidth = m_width / 2;
    const int chromaHeight = m_height / 2;
    const float *red = src;
    const float *green = src + plane;
    const float *blue = src + 2 * plane;
    uint8_t *uPlane = output.ptr[1];
    uint8_t *vPlane = nv12 ? output.ptr[1] + 1 : output.ptr[2];
    const int chromaStride = nv12 ? 2 : 1;
    const int uPitch = output.pitch[1];
    const int vPitch = nv12 ? output.pitch[1] : output.pitch[2];
    for (int y = 0; y < m_height; y++) {
        uint8_t *yRow = output.ptr[0] + (size_t)y * output.pitch[0];
        for (int x = 0; x < m_width; x++) {
            const size_t index = (size_t)y * m_width + x;
            const float luma = m_matRY * red[index] + m_matGY * green[index] + m_matBY * blue[index];
            yRow[x] = stdeint_clamp_u8((int)(luma * m_yRange + m_yOff + 0.5f));
        }
    }
    for (int cy = 0; cy < chromaHeight; cy++) {
        for (int cx = 0; cx < chromaWidth; cx++) {
            float u = 0.0f, v = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const size_t index = (size_t)(cy * 2 + dy) * m_width + (cx * 2 + dx);
                    u += m_matRU * red[index] + m_matGU * green[index] + m_matBU * blue[index];
                    v += m_matRV * red[index] + m_matGV * green[index] + m_matBV * blue[index];
                }
            }
            u *= 0.25f;
            v *= 0.25f;
            uPlane[(size_t)cy * uPitch + (size_t)cx * chromaStride] = stdeint_clamp_u8((int)(u * m_cRange + m_cOff + 0.5f));
            vPlane[(size_t)cy * vPitch + (size_t)cx * chromaStride] = stdeint_clamp_u8((int)(v * m_cRange + m_cOff + 0.5f));
        }
    }
}

void NVEncFilterStDeint::setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const {
    copyFramePropWithoutRes(output, input);
    output->picstruct = RGY_PICSTRUCT_FRAME;
    output->flags = (RGY_FRAME_FLAGS)(input->flags &
        ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
}

void NVEncFilterStDeint::setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) {
    auto frameDuration = input->duration;
    if (frameDuration == 0 && m_havePrevTimestamp) {
        const auto spanDuration = input->timestamp - m_prevTimestamp;
        frameDuration = (spanDuration > 0) ? spanDuration : m_prevDuration;
    }
    outputs[0]->timestamp = input->timestamp;
    outputs[0]->duration = (frameDuration + 1) / 2;
    outputs[1]->timestamp = outputs[0]->timestamp + outputs[0]->duration;
    outputs[1]->duration = frameDuration - outputs[0]->duration;
    outputs[0]->inputFrameId = input->inputFrameId;
    outputs[1]->inputFrameId = input->inputFrameId;
    m_prevTimestamp = input->timestamp;
    m_prevDuration = frameDuration;
    m_havePrevTimestamp = true;
}

void NVEncFilterStDeint::weaveRestoration(float *dst, const float *restoration, bool frameA) const {
    const size_t plane = (size_t)m_width * m_height;
    const size_t halfPlane = plane / 2;
    for (int channel = 0; channel < 3; channel++) {
        const auto inputPlane = m_inputBuf.data() + (size_t)channel * plane;
        const auto restorePlane = restoration + (size_t)channel * halfPlane;
        auto outputPlane = dst + (size_t)channel * plane;
        for (int y = 0; y < m_height / 2; y++) {
            const auto inputRow = inputPlane + (size_t)(y * 2 + (frameA ? 0 : 1)) * m_width;
            const auto restoreRow = restorePlane + (size_t)y * m_width;
            auto upperRow = outputPlane + (size_t)(y * 2) * m_width;
            auto lowerRow = upperRow + m_width;
            std::copy_n(frameA ? inputRow : restoreRow, m_width, upperRow);
            std::copy_n(frameA ? restoreRow : inputRow, m_width, lowerRow);
        }
    }
}

NVEncStDeintColorCoeffs NVEncFilterStDeint::colorCoeffs() const {
    return NVEncStDeintColorCoeffs {
        m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange,
        m_matVR, m_matUG, m_matVG, m_matUB,
        m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU,
        m_matRV, m_matGV, m_matBV
    };
}

RGY_ERR NVEncFilterStDeint::initCudaPath(cudaStream_t stream) {
    if (m_cudaPathTried) return m_cudaPath ? RGY_ERR_NONE : RGY_ERR_UNSUPPORTED;
    m_cudaPathTried = true;
    if (!m_inputDevice || !m_outputDevice || stream == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("stdeint: CUDA zero-copy initialization is unavailable; using host path.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto deviceSession = std::make_unique<RGYOnnxRTCUDA>();
    tstring errorMessage;
    auto err = deviceSession->init(m_modelPath, m_deviceID, m_provider, m_height, m_width, errorMessage, stream);
    if (err != RGY_ERR_NONE || !deviceSession->deviceIOAvailable()
        || deviceSession->inChannels() != 3 || deviceSession->outChannels() != 6
        || deviceSession->outHeight() != m_height / 2 || deviceSession->outWidth() != m_width) {
        const auto reason = !errorMessage.empty() ? errorMessage : deviceSession->lastError();
        AddMessage(RGY_LOG_WARN, _T("stdeint: CUDA zero-copy initialization failed; using host path: %s\n"), reason.c_str());
        return (err != RGY_ERR_NONE) ? err : RGY_ERR_UNSUPPORTED;
    }
    m_ov = std::move(deviceSession);
    m_cudaPath = true;
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamStDeint>(m_param);
    if (prm) setFilterInfo(prm->print() + _T(", path cuda-zerocopy"));
    AddMessage(RGY_LOG_INFO, _T("stdeint: path cuda-zerocopy initialized on the filter stream.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterStDeint::runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int outputCount, const int sourceIndices[2], cudaStream_t stream) {
    auto err = run_stdeint_pack_rgb(input, (float *)m_inputDevice->ptr, colorCoeffs(), stream);
    if (err != RGY_ERR_NONE) return err;
    err = m_ov->inferDevice((const float *)m_inputDevice->ptr, (float *)m_outputDevice->ptr);
    if (err != RGY_ERR_NONE) return err;

    const size_t restorationElements = (size_t)3 * m_width * (m_height / 2);
    for (int i = 0; i < outputCount; i++) {
        const int frameIndex = sourceIndices[i];
        err = run_stdeint_weave_yuv(outputs[i], (const float *)m_inputDevice->ptr,
            (const float *)m_outputDevice->ptr + (size_t)frameIndex * restorationElements,
            frameIndex == 0, colorCoeffs(), stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterStDeint::runHost(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int outputCount, const int sourceIndices[2], cudaStream_t stream) {
    auto err = copyFrameAsync(&m_inputStaging->frame, input, stream);
    if (err != RGY_ERR_NONE) return err;
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;
    yuvToRGB(m_inputStaging->frame, m_inputBuf.data());

    err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) return err;
    const size_t restorationElements = (size_t)3 * m_width * (m_height / 2);
    for (int i = 0; i < outputCount; i++) {
        const int frameIndex = sourceIndices[i];
        weaveRestoration(m_weaveBuf.data(), m_outputBuf.data() + (size_t)frameIndex * restorationElements, frameIndex == 0);
        rgbToYUV(m_outputStaging[i]->frame, m_weaveBuf.data());
        err = copyFrameAsync(outputs[i], &m_outputStaging[i]->frame, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterStDeint::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    const bool bob = m_mode == VppStDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;
    if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
        for (int i = 0; i < outputCount; i++) {
            auto output = &m_frameBuf[i]->frame;
            const auto err = copyFrameAsync(output, pInputFrame, stream);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to copy progressive input: %s.\n"), get_err_mes(err));
                return err;
            }
            setOutputFrameProp(output, pInputFrame);
            ppOutputFrames[i] = output;
        }
        *pOutputFrameNum = outputCount;
        if (bob) {
            setBobTimestamp(pInputFrame, ppOutputFrames);
        }
        return RGY_ERR_NONE;
    }

    bool inputTff = m_defaultTff;
    if (pInputFrame->picstruct & RGY_PICSTRUCT_BFF) {
        inputTff = false;
    } else if (pInputFrame->picstruct & RGY_PICSTRUCT_TFF) {
        inputTff = true;
    }
    const int firstIndex = inputTff ? 0 : 1;
    const int sourceIndices[2] = { firstIndex, 1 - firstIndex };
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        setOutputFrameProp(output, pInputFrame);
        ppOutputFrames[i] = output;
    }
    if (!m_cudaPathTried) initCudaPath(stream);
    auto err = m_cudaPath
        ? runCuda(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream)
        : runHost(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream);
    if (err != RGY_ERR_NONE && m_cudaPath) {
        AddMessage(RGY_LOG_WARN, _T("stdeint: CUDA zero-copy execution failed; falling back to host path: %s\n"),
            m_ov->lastError().c_str());
        m_cudaPath = false;
        const auto prm = std::dynamic_pointer_cast<NVEncFilterParamStDeint>(m_param);
        if (prm) setFilterInfo(prm->print() + _T(", path host"));
        err = runHost(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: processing failed: %s (%s).\n"),
            get_err_mes(err), m_ov->lastError().c_str());
        return err;
    }
    *pOutputFrameNum = outputCount;
    if (bob) {
        setBobTimestamp(pInputFrame, ppOutputFrames);
    }
    return RGY_ERR_NONE;
}
