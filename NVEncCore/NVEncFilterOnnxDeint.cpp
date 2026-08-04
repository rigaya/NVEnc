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

#include "NVEncFilterOnnxDeint.h"
#include <cuda_runtime.h>
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <algorithm>
#include <cstring>

static OnnxDeintModelSpec onnxDeintModelSpec(VppOnnxDeintArchitecture architecture, int frameWidth, int frameHeight) {
    if (architecture == VppOnnxDeintArchitecture::DDD) {
        return { architecture, 9, 3, frameWidth, frameHeight / 2, frameWidth, frameHeight / 2, 1, false };
    }
    return { architecture, 3, 6, frameHeight, frameWidth, frameHeight / 2, frameWidth, 0, true };
}

static const TCHAR *onnx_deint_cx_desc_or_unknown(const CX_DESC *list, int value) {
    const auto desc = get_cx_desc(list, value);
    return (desc != nullptr) ? desc : _T("unknown");
}

static bool onnx_deint_resolve_matrix(CspMatrix matrix, int inputHeight, CspMatrix& resolved) {
    if (matrix == RGY_MATRIX_AUTO || (int)matrix == COLOR_VALUE_AUTO_RESOLUTION) {
        resolved = (inputHeight <= 576) ? RGY_MATRIX_ST170_M : RGY_MATRIX_BT709;
        return true;
    }
    switch (matrix) {
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:
        resolved = RGY_MATRIX_ST170_M;
        return true;
    case RGY_MATRIX_BT709:
        resolved = RGY_MATRIX_BT709;
        return true;
    case RGY_MATRIX_BT2020_NCL:
        resolved = RGY_MATRIX_BT2020_NCL;
        return true;
    default:
        return false;
    }
}

static bool onnx_deint_supported_colorrange(CspColorRange range) {
    return range == RGY_COLORRANGE_AUTO
        || range == RGY_COLORRANGE_LIMITED
        || range == RGY_COLORRANGE_FULL;
}

static RGYOnnxRTProvider onnx_deint_provider(const tstring& provider) {
    const auto normalized = tolowercase(provider);
    if (normalized == _T("tensorrt")) return RGYOnnxRTProvider::TensorRT;
    if (normalized == _T("cuda")) return RGYOnnxRTProvider::Cuda;
    return RGYOnnxRTProvider::Auto;
}

class OnnxDeintCudaContextRestorer {
public:
    OnnxDeintCudaContextRestorer() : m_context(nullptr), m_valid(cuCtxGetCurrent(&m_context) == CUDA_SUCCESS) {}
    ~OnnxDeintCudaContextRestorer() {
        if (m_valid) {
            cuCtxSetCurrent(m_context);
        }
    }
private:
    CUcontext m_context;
    bool m_valid;
};

NVEncFilterOnnxDeint::NVEncFilterOnnxDeint() :
    NVEncFilter(), m_ov(), m_cropToRgb(), m_cropFromRgb(), m_width(0), m_height(0), m_mode(VppOnnxDeintMode::Bob), m_defaultTff(true),
    m_havePrevTimestamp(false), m_prevTimestamp(0), m_prevDuration(0),
    m_inputBuf(), m_outputBuf(), m_inputDevice(), m_outputDevice(), m_weaveDevice(),
    m_modelName(), m_modelPath(), m_spec(), m_provider(RGYOnnxRTProvider::Auto),
    m_precision(_T("fp32")), m_cacheDir(), m_deviceID(-1), m_cudaPathTried(false), m_cudaPath(false),
    m_framesIn(0), m_frameOut(0), m_weaveBuf(), m_temporalRing() {
    m_name = _T("onnx-deint");
}

NVEncFilterOnnxDeint::~NVEncFilterOnnxDeint() {
    close();
}

void NVEncFilterOnnxDeint::close() {
    m_ov.reset();
    m_cropToRgb.reset();
    m_cropFromRgb.reset();
    m_inputDevice.reset();
    m_outputDevice.reset();
    m_weaveDevice.reset();
    m_inputBuf.clear();
    m_outputBuf.clear();
    m_weaveBuf.clear();
    m_modelName.clear();
    m_modelPath.clear();
    for (auto& slot : m_temporalRing) {
        slot.frame.reset();
        slot.rgb.clear();
    }
    m_framesIn = 0;
    m_frameOut = 0;
    m_frameBuf.clear();
    m_havePrevTimestamp = false;
    m_cudaPathTried = false;
    m_cudaPath = false;
}

tstring NVEncFilterParamOnnxDeint::print() const {
    return strsprintf(_T("onnx-deint: model=%s, mode %s, provider %s, precision %s, cache_dir %s, colormatrix %s, colorrange %s"), modelFile.c_str(),
        get_cx_desc(list_vpp_onnx_deint_mode, (int)mode), provider.c_str(), precision.c_str(),
        cacheDir.empty() ? _T("disabled") : cacheDir.c_str(),
        onnx_deint_cx_desc_or_unknown(list_colormatrix, colormatrix), onnx_deint_cx_desc_or_unknown(list_colorrange, colorrange));
}

RGY_ERR NVEncFilterOnnxDeint::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamOnnxDeint>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOnnxRTCUDA::available()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: this build was compiled without ONNX Runtime CUDA support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model= (registered model name) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\.")) != tstring::npos) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model must be a registered name, not a path: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelDir.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: --vpp-onnx-model-dir is required for model registry.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYModelRegistry registry;
    const auto registryErr = registry.load(PathCombineS(prm->modelDir, _T("onnx_deint_models.json")), m_pLog);
    if (registryErr != RGY_ERR_NONE) return registryErr;
    const auto modelEntry = registry.find(prm->modelFile);
    if (!modelEntry) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" is not registered in onnx_deint_models.json.\n"), prm->modelFile.c_str());
        return RGY_ERR_NOT_FOUND;
    }
    if (!modelEntry->onnxDeintArchitecturePresent) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" has no architecture in onnx_deint_models.json.\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (!modelEntry->onnxDeintArchitectureTypeValid || !modelEntry->onnxDeintArchitecture) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" architecture must be a string.\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    VppOnnxDeintArchitecture architecture;
    if (*modelEntry->onnxDeintArchitecture == _T("stdeint")) {
        architecture = VppOnnxDeintArchitecture::StDeint;
    } else if (*modelEntry->onnxDeintArchitecture == _T("ddd")) {
        architecture = VppOnnxDeintArchitecture::DDD;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" has unknown architecture \"%s\" (expected stdeint or ddd).\n"),
            prm->modelFile.c_str(), modelEntry->onnxDeintArchitecture->c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    m_modelName = prm->modelFile;
    m_modelPath = registry.resolveModelPath(m_modelName);
    if (!rgy_file_exists(m_modelPath)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model file not found: %s\n"), m_modelPath.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    const auto inputCsp = prm->frameIn.csp;
    if ((inputCsp != RGY_CSP_YV12 && inputCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inputCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_width = prm->frameIn.width;
    m_height = prm->frameIn.height;
    if (m_height < 4 || (m_height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: height must be an even value of at least 4 (got %d).\n"), m_height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppOnnxDeintMode::Bob && prm->mode != VppOnnxDeintMode::Normal) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: invalid output mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    CspMatrix matrix = RGY_MATRIX_UNSPECIFIED;
    if (!onnx_deint_resolve_matrix(prm->colormatrix, m_height, matrix)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: unsupported colormatrix %s.\n"),
            onnx_deint_cx_desc_or_unknown(list_colormatrix, prm->colormatrix));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_deint_supported_colorrange(prm->colorrange)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: unsupported colorrange %s.\n"),
            onnx_deint_cx_desc_or_unknown(list_colorrange, prm->colorrange));
        return RGY_ERR_UNSUPPORTED;
    }
    m_mode = prm->mode;
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    m_ov = std::make_unique<RGYOnnxRTCUDA>();
    m_deviceID = prm->deviceID;
    if (m_deviceID < 0) {
        cudaGetDevice(&m_deviceID);
    }
    m_provider = onnx_deint_provider(prm->provider);
    m_precision = prm->precision;
    m_cacheDir = prm->cacheDir;
    if (!m_cacheDir.empty() && !rgy_directory_exists(m_cacheDir) && !CreateDirectoryRecursive(m_cacheDir.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to create TensorRT engine cache directory: %s\n"), m_cacheDir.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    tstring errorMessage;
    // モデル方式はマニフェストのarchitectureで確定し、チャンネル数から推測しない。
    m_spec = onnxDeintModelSpec(architecture, m_width, m_height);
    auto err = m_ov->init(m_modelPath, m_deviceID, m_provider, m_spec.modelHeight, m_spec.modelWidth, errorMessage,
        nullptr, m_precision, m_cacheDir);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to load/compile model: %s\n"), errorMessage.c_str());
        return err;
    }
    if (tolowercase(prm->provider) == _T("tensorrt") && m_ov->providerName() != _T("tensorrt")) {
        AddMessage(RGY_LOG_WARN, _T("onnx-deint: TensorRT provider is unavailable; falling back to CUDA: %s\n"),
            m_ov->lastError().c_str());
    } else if (!m_ov->lastError().empty()) {
        AddMessage(RGY_LOG_WARN, _T("onnx-deint: %s\n"), m_ov->lastError().c_str());
    }
    if (!m_ov->cacheInfo().empty()) {
        AddMessage(RGY_LOG_INFO, _T("onnx-deint: %s\n"), m_ov->cacheInfo().c_str());
    }
    if (m_ov->inChannels() != m_spec.inputChannels || m_ov->outChannels() != m_spec.outputChannels) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: invalid %s model (expected %dch input / %dch output, got %dch / %dch).\n"),
            m_spec.architecture == VppOnnxDeintArchitecture::DDD ? _T("DDD") : _T("ST-DeInt"), m_spec.inputChannels, m_spec.outputChannels,
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->inHeight() != m_spec.modelHeight || m_ov->inWidth() != m_spec.modelWidth) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model input size mismatch (expected %dx%d, got %dx%d).\n"),
            m_spec.modelWidth, m_spec.modelHeight, m_ov->inWidth(), m_ov->inHeight());
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->outHeight() != m_spec.outputHeight || m_ov->outWidth() != m_spec.outputWidth) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model output size mismatch (expected %dx%d, got %dx%d).\n"),
            m_spec.outputWidth, m_spec.outputHeight, m_ov->outWidth(), m_ov->outHeight());
        return RGY_ERR_UNSUPPORTED;
    }

    prm->frameOut.csp = inputCsp;
    prm->frameOut.width = m_width;
    prm->frameOut.height = m_height;
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    const int outputCount = (m_mode == VppOnnxDeintMode::Bob) ? 2 : 1;
    if (m_mode == VppOnnxDeintMode::Bob) {
        prm->baseFps *= 2;
    }

    err = AllocFrameBuf(prm->frameOut, outputCount);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const size_t plane = (size_t)m_width * m_height;
    const size_t inputPlane = (size_t)m_spec.modelHeight * m_spec.modelWidth;
    const size_t outputPlane = (size_t)m_spec.outputHeight * m_spec.outputWidth;
    m_inputBuf.resize((size_t)m_spec.inputChannels * inputPlane);
    m_outputBuf.resize((size_t)m_spec.outputChannels * outputPlane);
    m_weaveBuf.resize(3 * plane);
    m_cudaPathTried = false;
    m_cudaPath = false;
    m_inputDevice = std::make_unique<CUMemBuf>(3 * plane * sizeof(float));
    m_outputDevice = std::make_unique<CUMemBuf>(m_outputBuf.size() * sizeof(float));
    m_weaveDevice = std::make_unique<CUMemBuf>(m_inputBuf.size() * sizeof(float));
    if (m_inputDevice->alloc() != RGY_ERR_NONE || m_outputDevice->alloc() != RGY_ERR_NONE || m_weaveDevice->alloc() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to allocate RGB tensor buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (m_spec.architecture == VppOnnxDeintArchitecture::DDD) {
        err = allocTemporalRing(prm->frameIn);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    auto rgbInfo = rgbFrame((float *)m_inputDevice->ptr);
    auto cropToRgbParam = std::make_shared<NVEncFilterParamCrop>();
    cropToRgbParam->frameIn = prm->frameIn;
    cropToRgbParam->frameIn.picstruct = RGY_PICSTRUCT_FRAME;
    cropToRgbParam->frameOut = rgbInfo;
    cropToRgbParam->baseFps = prm->baseFps;
    cropToRgbParam->matrix = matrix;
    cropToRgbParam->colorrange = (prm->colorrange == RGY_COLORRANGE_FULL) ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_LIMITED;
    cropToRgbParam->bOutOverwrite = false;
    m_cropToRgb = std::make_unique<NVEncFilterCspCrop>();
    err = m_cropToRgb->init(cropToRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to initialize YUV-to-RGB conversion: %s.\n"), get_err_mes(err));
        return err;
    }

    auto cropFromRgbParam = std::make_shared<NVEncFilterParamCrop>();
    cropFromRgbParam->frameIn = rgbInfo;
    cropFromRgbParam->frameOut = prm->frameOut;
    cropFromRgbParam->baseFps = prm->baseFps;
    cropFromRgbParam->matrix = matrix;
    cropFromRgbParam->colorrange = cropToRgbParam->colorrange;
    cropFromRgbParam->bOutOverwrite = false;
    m_cropFromRgb = std::make_unique<NVEncFilterCspCrop>();
    err = m_cropFromRgb->init(cropFromRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to initialize RGB-to-YUV conversion: %s.\n"), get_err_mes(err));
        return err;
    }

    m_havePrevTimestamp = false;
    m_param = prm;
    setFilterInfo(prm->print() + strsprintf(_T(", resolved=%s, architecture=%s, execution=host"),
        m_modelPath.c_str(), m_spec.architecture == VppOnnxDeintArchitecture::DDD ? _T("ddd") : _T("stdeint")));
    AddMessage(RGY_LOG_DEBUG, _T("onnx-deint: model=%s, resolved=%s, architecture=%s, %dx%d, mode %s, provider %s, execution=host, model %dch/%dch %dx%d.\n"),
        m_modelName.c_str(), m_modelPath.c_str(), m_spec.architecture == VppOnnxDeintArchitecture::DDD ? _T("ddd") : _T("stdeint"), m_width, m_height,
        get_cx_desc(list_vpp_onnx_deint_mode, (int)m_mode), m_ov->providerName().c_str(),
        m_ov->inChannels(), m_ov->outChannels(), m_spec.modelWidth, m_spec.modelHeight);
    return RGY_ERR_NONE;
}

RGYFrameInfo NVEncFilterOnnxDeint::rgbFrame(float *ptr) const {
    RGYFrameInfo frame;
    frame.width = m_width;
    frame.height = m_height;
    frame.csp = RGY_CSP_RGB_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    const size_t planeBytes = (size_t)m_width * m_height * sizeof(float);
    frame.ptr[0] = (uint8_t *)ptr;
    frame.ptr[1] = (uint8_t *)ptr + planeBytes;
    frame.ptr[2] = (uint8_t *)ptr + planeBytes * 2;
    frame.pitch[0] = m_width * sizeof(float);
    frame.pitch[1] = frame.pitch[0];
    frame.pitch[2] = frame.pitch[0];
    return frame;
}

RGY_ERR NVEncFilterOnnxDeint::convertToRgb(const RGYFrameInfo *input, cudaStream_t stream) {
    auto inputFrame = *input;
    inputFrame.picstruct = RGY_PICSTRUCT_FRAME;
    auto outputFrame = rgbFrame((float *)m_inputDevice->ptr);
    RGYFrameInfo *outputs[1] = { &outputFrame };
    int outputCount = 0;
    const auto err = m_cropToRgb->filter(&inputFrame, outputs, &outputCount, stream);
    if (err != RGY_ERR_NONE || outputCount != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: YUV-to-RGB conversion returned %d outputs (err=%s).\n"), outputCount, get_err_mes(err));
        return err != RGY_ERR_NONE ? err : RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::convertFromRgb(RGYFrameInfo *output, cudaStream_t stream) {
    auto inputFrame = rgbFrame((float *)m_weaveDevice->ptr);
    RGYFrameInfo *outputs[1] = { output };
    int outputCount = 0;
    const auto err = m_cropFromRgb->filter(&inputFrame, outputs, &outputCount, stream);
    if (err != RGY_ERR_NONE || outputCount != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: RGB-to-YUV conversion returned %d outputs (err=%s).\n"), outputCount, get_err_mes(err));
        return err != RGY_ERR_NONE ? err : RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterOnnxDeint::setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const {
    copyFramePropWithoutRes(output, input);
    output->picstruct = RGY_PICSTRUCT_FRAME;
    output->flags = (RGY_FRAME_FLAGS)(input->flags &
        ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
}

void NVEncFilterOnnxDeint::setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) {
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

RGY_ERR NVEncFilterOnnxDeint::copyProgressiveOutputs(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int *outputCount, cudaStream_t stream) {
    const bool bob = m_mode == VppOnnxDeintMode::Bob;
    const int count = bob ? 2 : 1;
    for (int i = 0; i < count; i++) {
        auto output = &m_frameBuf[i]->frame;
        const auto err = copyFrameAsync(output, input, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to copy progressive input: %s.\n"), get_err_mes(err));
            return err;
        }
        setOutputFrameProp(output, input);
        outputs[i] = output;
    }
    *outputCount = count;
    if (bob) setBobTimestamp(input, outputs);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::initCudaPath(cudaStream_t stream) {
    OnnxDeintCudaContextRestorer contextRestorer;
    if (m_cudaPathTried) return m_cudaPath ? RGY_ERR_NONE : RGY_ERR_UNSUPPORTED;
    m_cudaPathTried = true;
    if (!m_inputDevice || !m_outputDevice || stream == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("onnx-deint: CUDA zero-copy initialization is unavailable; using host path.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto deviceSession = std::make_unique<RGYOnnxRTCUDA>();
    tstring errorMessage;
    auto err = deviceSession->init(m_modelPath, m_deviceID, m_provider, m_spec.modelHeight, m_spec.modelWidth, errorMessage,
        stream, m_precision, m_cacheDir);
    if (err != RGY_ERR_NONE || !m_spec.supportsSharedCuda || !deviceSession->deviceIOAvailable()
        || deviceSession->inChannels() != m_spec.inputChannels || deviceSession->outChannels() != m_spec.outputChannels
        || deviceSession->inHeight() != m_spec.modelHeight || deviceSession->inWidth() != m_spec.modelWidth
        || deviceSession->outHeight() != m_spec.outputHeight || deviceSession->outWidth() != m_spec.outputWidth) {
        const auto reason = !errorMessage.empty() ? errorMessage : deviceSession->lastError();
        AddMessage(RGY_LOG_WARN, _T("onnx-deint: CUDA zero-copy initialization failed; using host path: %s\n"), reason.c_str());
        return (err != RGY_ERR_NONE) ? err : RGY_ERR_UNSUPPORTED;
    }
    m_ov = std::move(deviceSession);
    m_cudaPath = true;
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamOnnxDeint>(m_param);
    if (prm) setFilterInfo(prm->print() + strsprintf(_T(", resolved=%s, architecture=%s, execution=cuda-zerocopy"),
        m_modelPath.c_str(), m_spec.architecture == VppOnnxDeintArchitecture::DDD ? _T("ddd") : _T("stdeint")));
    AddMessage(RGY_LOG_INFO, _T("onnx-deint: path cuda-zerocopy initialized on the filter stream.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int outputCount, const int sourceIndices[2], cudaStream_t stream) {
    auto err = convertToRgb(input, stream);
    if (err != RGY_ERR_NONE) return err;
    err = m_ov->inferDevice((const float *)m_inputDevice->ptr, (float *)m_outputDevice->ptr);
    if (err != RGY_ERR_NONE) return err;

    const size_t restorationElements = (size_t)3 * m_width * (m_height / 2);
    for (int i = 0; i < outputCount; i++) {
        const int frameIndex = sourceIndices[i];
        err = run_onnx_deint_weave_rgb((float *)m_weaveDevice->ptr, (const float *)m_inputDevice->ptr,
            (const float *)m_outputDevice->ptr + (size_t)frameIndex * restorationElements,
            frameIndex == 0, m_width, m_height, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: CUDA weave failed: %s.\n"), get_err_mes(err));
            return err;
        }
        err = convertFromRgb(outputs[i], stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::runHost(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int outputCount, const int sourceIndices[2], cudaStream_t stream) {
    auto err = convertToRgb(input, stream);
    if (err != RGY_ERR_NONE) return err;
    auto cudaerr = cudaMemcpyAsync(m_inputBuf.data(), m_inputDevice->ptr,
        m_inputBuf.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;

    err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) return err;
    cudaerr = cudaMemcpyAsync(m_outputDevice->ptr, m_outputBuf.data(),
        m_outputBuf.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    const size_t restorationElements = (size_t)3 * m_width * (m_height / 2);
    for (int i = 0; i < outputCount; i++) {
        const int frameIndex = sourceIndices[i];
        err = run_onnx_deint_weave_rgb((float *)m_weaveDevice->ptr, (const float *)m_inputDevice->ptr,
            (const float *)m_outputDevice->ptr + (size_t)frameIndex * restorationElements,
            frameIndex == 0, m_width, m_height, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: CUDA weave failed: %s.\n"), get_err_mes(err));
            return err;
        }
        err = convertFromRgb(outputs[i], stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::allocTemporalRing(const RGYFrameInfo& frameInfo) {
    const size_t plane = (size_t)m_width * m_height;
    for (auto& slot : m_temporalRing) {
        slot.frame = std::make_unique<CUFrameBuf>(frameInfo);
        if (!slot.frame || slot.frame->alloc() != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to allocate the temporal frame ring.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        slot.rgb.assign(3 * plane, 0.0f);
        slot.tff = m_defaultTff;
        slot.interlaced = false;
    }
    m_framesIn = 0;
    m_frameOut = 0;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::addTemporalFrame(const RGYFrameInfo *input, cudaStream_t stream) {
    auto& slot = m_temporalRing[m_framesIn % m_temporalRing.size()];
    auto err = copyFrameAsync(&slot.frame->frame, input, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to copy input into the temporal frame ring: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFrameProp(&slot.frame->frame, input);
    err = convertToRgb(input, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to convert input to RGB: %s.\n"), get_err_mes(err));
        return err;
    }
    auto cudaerr = cudaMemcpyAsync(slot.rgb.data(), m_inputDevice->ptr,
        slot.rgb.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;

    slot.tff = m_defaultTff;
    if (input->picstruct & RGY_PICSTRUCT_BFF) {
        slot.tff = false;
    } else if (input->picstruct & RGY_PICSTRUCT_TFF) {
        slot.tff = true;
    }
    slot.interlaced = (input->picstruct & RGY_PICSTRUCT_INTERLACED) != 0;
    m_framesIn++;
    return RGY_ERR_NONE;
}

// フィールドはフレームごとに表示順（0が先、1が後）で数える。
// 戻り値は0がtop field（偶数行）、1がbottom field（奇数行）。
int NVEncFilterOnnxDeint::temporalFieldParity(const int fieldIndex) const {
    const auto& slot = m_temporalRing[(fieldIndex / 2) % m_temporalRing.size()];
    const int fieldPos = fieldIndex & 1;
    return slot.tff ? fieldPos : 1 - fieldPos;
}

// 出力フィールドの前・現在・次を転置し、フィールド単位の9chテンソルを組み立てる。
void NVEncFilterOnnxDeint::buildTemporalInput(const int frameIndex, const int fieldPos) {
    const int fieldHeight = m_height / 2;
    const int fieldIndex = frameIndex * 2 + fieldPos;
    const int lastField = (m_framesIn - 1) * 2 + 1;
    const bool flip = temporalFieldParity(fieldIndex) == 0;
    for (int i = 0; i < 3; i++) {
        int refIndex = fieldIndex - 1 + i;
        if (refIndex < 0) refIndex = -refIndex;
        if (refIndex > lastField) refIndex = 2 * lastField - refIndex;
        const auto& slot = m_temporalRing[(refIndex / 2) % m_temporalRing.size()];
        const int parity = temporalFieldParity(refIndex);
        for (int c = 0; c < 3; c++) {
            const float *srcPlane = slot.rgb.data() + (size_t)c * m_width * m_height;
            float *dstPlane = m_inputBuf.data() + (size_t)(i * 3 + c) * m_width * fieldHeight;
            for (int x = 0; x < m_width; x++) {
                float *dstLine = dstPlane + (size_t)x * fieldHeight;
                for (int y = 0; y < fieldHeight; y++) {
                    const int srcRow = 2 * (flip ? (fieldHeight - 1 - y) : y) + parity;
                    dstLine[y] = srcPlane[(size_t)srcRow * m_width + x];
                }
            }
        }
    }
}

// モデル出力と中央の既知フィールドを合成し、転置を戻して1枚のRGBフレームにする。
void NVEncFilterOnnxDeint::combineTemporalOutput(const int frameIndex, const int fieldPos, float *dst) const {
    const int fieldHeight = m_height / 2;
    const int fieldIndex = frameIndex * 2 + fieldPos;
    const int parity = temporalFieldParity(fieldIndex);
    const bool flip = (parity == 0);
    const auto& slot = m_temporalRing[frameIndex % m_temporalRing.size()];
    for (int c = 0; c < 3; c++) {
        const float *modelPlane = m_outputBuf.data() + (size_t)c * m_width * fieldHeight;
        const float *midPlane = slot.rgb.data() + (size_t)c * m_width * m_height;
        float *dstPlane = dst + (size_t)c * m_width * m_height;
        for (int y = 0; y < m_height; y++) {
            const int pos = flip ? (m_height - 1 - y) : y;
            float *dstLine = dstPlane + (size_t)y * m_width;
            if ((pos & 1) == 0) {
                const int row = pos / 2;
                for (int x = 0; x < m_width; x++) {
                    dstLine[x] = modelPlane[(size_t)x * fieldHeight + row];
                }
            } else {
                const int row = flip ? (fieldHeight - 1 - pos / 2) : (pos / 2);
                std::memcpy(dstLine, midPlane + (size_t)(2 * row + parity) * m_width, (size_t)m_width * sizeof(float));
            }
        }
    }
}

RGY_ERR NVEncFilterOnnxDeint::procTemporalField(const int frameIndex, const int fieldPos,
    RGYFrameInfo *output, cudaStream_t stream) {
    buildTemporalInput(frameIndex, fieldPos);
    auto err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: inference failed: %s.\n"), get_err_mes(err));
        return err;
    }
    combineTemporalOutput(frameIndex, fieldPos, m_weaveBuf.data());
    auto cudaerr = cudaMemcpyAsync(m_weaveDevice->ptr, m_weaveBuf.data(),
        m_weaveBuf.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return convertFromRgb(output, stream);
}

RGY_ERR NVEncFilterOnnxDeint::emitTemporalFrame(const int frameIndex, RGYFrameInfo **outputs,
    int *outputFrameNum, cudaStream_t stream) {
    const auto& slot = m_temporalRing[frameIndex % m_temporalRing.size()];
    const auto *source = &slot.frame->frame;
    const bool bob = m_mode == VppOnnxDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;
    if (!slot.interlaced) {
        return copyProgressiveOutputs(source, outputs, outputFrameNum, stream);
    }
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        const auto err = procTemporalField(frameIndex, i, output, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to emit temporal frame: %s.\n"), get_err_mes(err));
            return err;
        }
        setOutputFrameProp(output, source);
        outputs[i] = output;
    }
    *outputFrameNum = outputCount;
    if (bob) setBobTimestamp(source, outputs);
    return RGY_ERR_NONE;
}

// 前後フィールドを参照するため1フレーム遅延させ、drainで末尾を出力する。
RGY_ERR NVEncFilterOnnxDeint::runTemporal(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int *outputFrameNum, cudaStream_t stream) {
    if (input->ptr[0] != nullptr) {
        auto err = addTemporalFrame(input, stream);
        if (err != RGY_ERR_NONE) return err;
        if (m_frameOut + m_spec.lookaheadFrames < m_framesIn) {
            const int frameIndex = m_frameOut++;
            return emitTemporalFrame(frameIndex, outputs, outputFrameNum, stream);
        }
        return RGY_ERR_NONE;
    }
    if (m_frameOut < m_framesIn) {
        const int frameIndex = m_frameOut++;
        return emitTemporalFrame(frameIndex, outputs, outputFrameNum, stream);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnxDeint::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (!pInputFrame) {
        return RGY_ERR_NONE;
    }
    if (m_spec.architecture == VppOnnxDeintArchitecture::DDD) {
        return runTemporal(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
    }
    if (!pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
        return copyProgressiveOutputs(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
    }

    const bool bob = m_mode == VppOnnxDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;

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
        ppOutputFrames[i] = output;
    }
    if (!m_cudaPathTried) initCudaPath(stream);
    auto err = m_cudaPath
        ? runCuda(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream)
        : runHost(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream);
    if (err != RGY_ERR_NONE && m_cudaPath) {
        AddMessage(RGY_LOG_WARN, _T("onnx-deint: CUDA zero-copy execution failed; falling back to host path: %s\n"),
            m_ov->lastError().c_str());
        m_cudaPath = false;
        const auto prm = std::dynamic_pointer_cast<NVEncFilterParamOnnxDeint>(m_param);
        if (prm) setFilterInfo(prm->print() + _T(", path host"));
        err = runHost(pInputFrame, ppOutputFrames, outputCount, sourceIndices, stream);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: processing failed: %s (%s).\n"),
            get_err_mes(err), m_ov->lastError().c_str());
        return err;
    }
    for (int i = 0; i < outputCount; i++) {
        setOutputFrameProp(ppOutputFrames[i], pInputFrame);
    }
    *pOutputFrameNum = outputCount;
    if (bob) {
        setBobTimestamp(pInputFrame, ppOutputFrames);
    }
    return RGY_ERR_NONE;
}
