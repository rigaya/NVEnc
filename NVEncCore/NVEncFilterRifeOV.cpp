// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
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

#include "NVEncFilterRifeOV.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <algorithm>
#include <cmath>
#include <cstring>

static inline uint8_t clamp_u8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline float   clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

NVEncFilterRifeOV::NVEncFilterRifeOV() :
    NVEncFilter(), m_ov(), m_W(0), m_H(0), m_multi(2), m_maxval(255.0f),
    m_yOff(0), m_yScale(1), m_yRange(255), m_cOff(128), m_cScale(1), m_cRange(255),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_havePrev(false), m_prevTimestamp(0), m_prevDuration(0),
    m_prevRGB(), m_currRGB(), m_inBuf(), m_outBuf(), m_baseGrid(), m_multiplier(),
    m_inStaging(), m_outStaging(), m_inputDevice(), m_outputDevice(), m_cropToRgb(), m_cropFromRgb(),
    m_modelPath(), m_deviceID(-1), m_cudaPathTried(false), m_cudaPath(false) {
    m_name = _T("rife-ov");
}

NVEncFilterRifeOV::~NVEncFilterRifeOV() { close(); }

void NVEncFilterRifeOV::close() {
    m_ov.reset();
    m_inStaging.reset();
    m_outStaging.reset();
    m_inputDevice.reset();
    m_outputDevice.reset();
    m_cropToRgb.reset();
    m_cropFromRgb.reset();
    m_cudaPathTried = false;
    m_cudaPath = false;
    m_frameBuf.clear();
}

tstring NVEncFilterParamRifeOV::print() const {
    return strsprintf(_T("rife-ov: %s, x%d, device %s"), modelFile.c_str(), multi, device.c_str());
}

void NVEncFilterRifeOV::setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax) {
    float Kr = 0.2126f, Kb = 0.0722f; // BT.709
    if (matrixSel == 601)  { Kr = 0.299f;  Kb = 0.114f; }
    if (matrixSel == 2020) { Kr = 0.2627f; Kb = 0.0593f; }
    const float Kg = 1.0f - Kr - Kb;
    m_matVR = 2.0f * (1.0f - Kr);
    m_matUG = -2.0f * Kb * (1.0f - Kb) / Kg;
    m_matVG = -2.0f * Kr * (1.0f - Kr) / Kg;
    m_matUB = 2.0f * (1.0f - Kb);
    m_matRY = Kr;                            m_matGY = Kg;                            m_matBY = Kb;
    m_matRU = -Kr / (2.0f * (1.0f - Kb));    m_matGU = -Kg / (2.0f * (1.0f - Kb));    m_matBU = 0.5f;
    m_matRV = 0.5f;                          m_matGV = -Kg / (2.0f * (1.0f - Kr));    m_matBV = -Kb / (2.0f * (1.0f - Kr));
    m_yOff   = rangeTV ? (16.0f  * pixMax / 255.0f) : 0.0f;
    m_yRange = rangeTV ? (219.0f * pixMax / 255.0f) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * pixMax / 255.0f) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * pixMax / 255.0f) : (float)pixMax;
    m_cScale = 1.0f / m_cRange;
}

RGY_ERR NVEncFilterRifeOV::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRifeOV>(pParam);
    if (!prm) { AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n")); return RGY_ERR_INVALID_PARAM; }
    if (!RGYOnnxRTCUDA::available()) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: this build was compiled without ONNX Runtime CUDA support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: model= (a registered model name or RIFE .onnx path) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        const auto err = registry.load(PathCombineS(prm->modelDir, _T("rife_ov_models.json")), m_pLog);
        if (err != RGY_ERR_NONE) return err;
        if (!registry.find(prm->modelFile)) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: model \"%s\" not found in rife_ov_models.json\n"), prm->modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->modelFile = registry.resolveModelPath(prm->modelFile);
    }
    if (!rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: model file not found: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    if (prm->multi < 2) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: multi must be >= 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_W = prm->frameIn.width;
    m_H = prm->frameIn.height;
    if ((m_W % 32) != 0 || (m_H % 32) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: RIFE requires width/height a multiple of 32 (got %dx%d). "
            "Pad/crop the input first (e.g. --vpp-pad / --crop).\n"), m_W, m_H);
        return RGY_ERR_UNSUPPORTED;
    }
    m_multi  = prm->multi;
    m_maxval = (float)((1 << prm->frameIn.bitdepth) - 1);

    // RIFE ONNX を [1,11,H,W] 入力で読み込む。
    m_ov = std::make_unique<RGYOnnxRTCUDA>();
    int deviceID = prm->deviceID;
    if (deviceID < 0) cudaGetDevice(&deviceID);
    m_modelPath = prm->modelFile;
    m_deviceID = deviceID;
    tstring errMsg;
    RGY_ERR err = m_ov->init(prm->modelFile, deviceID, RGYOnnxRTProvider::Auto, m_H, m_W, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to load/compile model: %s\n"), errMsg.c_str());
        return err;
    }
    if (m_ov->inChannels() != 11 || m_ov->outChannels() != 3) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: not a RIFE model (expected 11ch in / 3ch out, got %dch / %dch).\n"),
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }

    // colour matrix + range (auto: BT.601 for SD, BT.709 for HD; TV range).
    int matrixSel;
    if      (prm->colormatrix == _T("bt601"))  matrixSel = 601;
    else if (prm->colormatrix == _T("bt2020")) matrixSel = 2020;
    else if (prm->colormatrix == _T("bt709"))  matrixSel = 709;
    else                                       matrixSel = (m_H <= 576) ? 601 : 709;
    const bool rangeTV = (prm->colorrange != _T("pc"));
    setupColorCoeffs(matrixSel, rangeTV, 255);

    // precompute base_grid (normalised [-1,1] mesh) and multiplier (2/(W-1), 2/(H-1)).
    const size_t plane = (size_t)m_W * m_H;
    m_baseGrid.resize(2 * plane);
    m_multiplier.resize(2 * plane);
    const float multH = 2.0f / (float)(m_W - 1);
    const float multV = 2.0f / (float)(m_H - 1);
    for (int y = 0; y < m_H; y++) {
        const float vy = (m_H > 1) ? (-1.0f + 2.0f * (float)y / (float)(m_H - 1)) : 0.0f;
        for (int x = 0; x < m_W; x++) {
            const float vx = (m_W > 1) ? (-1.0f + 2.0f * (float)x / (float)(m_W - 1)) : 0.0f;
            const size_t idx = (size_t)y * m_W + x;
            m_baseGrid[idx]           = vx;          // ch0: horizontal
            m_baseGrid[plane + idx]   = vy;          // ch1: vertical
            m_multiplier[idx]         = multH;       // ch0
            m_multiplier[plane + idx] = multV;       // ch1
        }
    }

    // host buffers
    m_prevRGB.resize(3 * plane);
    m_currRGB.resize(3 * plane);
    m_inBuf.resize(11 * plane);
    m_outBuf.resize(3 * plane);

    // output frame info: same resolution, frame rate multiplied by `multi`.
    auto frameOut = prm->frameOut;
    frameOut.csp    = inCsp;
    frameOut.width  = m_W;
    frameOut.height = m_H;
    prm->frameOut   = frameOut;

    // Multi-out filter (1-in / multi-out): the framework's auto path-through for
    // timestamp / picstruct / flags only works for 1-in/1-out, so clear those bits;
    // run_filter stamps timestamp / duration / picstruct / inputFrameId per output.
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));

    prm->baseFps   *= m_multi;   // interpolated output runs at multi x the input rate

    // pool: up to `multi` output frames per input frame.
    err = AllocFrameBuf(prm->frameOut, m_multi);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_inStaging  = std::make_unique<CUFrameBuf>();
    m_outStaging = std::make_unique<CUFrameBuf>();
    if (m_inStaging->allocHost(m_W, m_H, inCsp) != RGY_ERR_NONE
        || m_outStaging->allocHost(m_W, m_H, inCsp) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to allocate host staging frame buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    m_inputDevice = std::make_unique<CUMemBuf>(m_inBuf.size() * sizeof(float));
    m_outputDevice = std::make_unique<CUMemBuf>(m_outBuf.size() * sizeof(float));
    if (m_inputDevice->alloc() != RGY_ERR_NONE || m_outputDevice->alloc() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: CUDAテンソルバッファの確保に失敗しました。\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    const size_t planeBytes = (size_t)m_W * m_H * sizeof(float);
    auto cudaerr = cudaMemcpy((uint8_t *)m_inputDevice->ptr + 7 * planeBytes,
        m_baseGrid.data(), m_baseGrid.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaerr == cudaSuccess) {
        cudaerr = cudaMemcpy((uint8_t *)m_inputDevice->ptr + 9 * planeBytes,
            m_multiplier.data(), m_multiplier.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: 固定テンソルの転送に失敗しました: %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return err_to_rgy(cudaerr);
    }
    auto rgbCurrent = rgbFrame((float *)m_inputDevice->ptr + 3 * (size_t)m_W * m_H);
    auto toRgbParam = std::make_shared<NVEncFilterParamCrop>();
    toRgbParam->frameIn = prm->frameIn;
    toRgbParam->frameIn.picstruct = RGY_PICSTRUCT_FRAME;
    toRgbParam->frameOut = rgbCurrent;
    toRgbParam->baseFps = prm->baseFps;
    toRgbParam->matrix = (matrixSel == 601) ? RGY_MATRIX_ST170_M
        : (matrixSel == 2020) ? RGY_MATRIX_BT2020_NCL : RGY_MATRIX_BT709;
    toRgbParam->colorrange = rangeTV ? RGY_COLORRANGE_LIMITED : RGY_COLORRANGE_FULL;
    toRgbParam->bOutOverwrite = false;
    m_cropToRgb = std::make_unique<NVEncFilterCspCrop>();
    err = m_cropToRgb->init(toRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;
    auto rgbOutput = rgbFrame((float *)m_outputDevice->ptr);
    auto fromRgbParam = std::make_shared<NVEncFilterParamCrop>();
    fromRgbParam->frameIn = rgbOutput;
    fromRgbParam->frameOut = prm->frameOut;
    fromRgbParam->baseFps = prm->baseFps;
    fromRgbParam->matrix = toRgbParam->matrix;
    fromRgbParam->colorrange = toRgbParam->colorrange;
    fromRgbParam->bOutOverwrite = false;
    m_cropFromRgb = std::make_unique<NVEncFilterCspCrop>();
    err = m_cropFromRgb->init(fromRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;
    m_cudaPathTried = false;
    m_cudaPath = false;

    m_havePrev = false;
    m_param = prm;
    AddMessage(RGY_LOG_DEBUG, _T("rife-ov: %s, %dx%d, x%d, device %s.\n"),
        prm->modelFile.c_str(), m_W, m_H, m_multi, prm->device.c_str());
    return RGY_ERR_NONE;
}

// YUV (yv12/nv12 8-bit 4:2:0) -> planar RGB [0,1] CHW (3*W*H). Chroma bilinear-upsampled.
void NVEncFilterRifeOV::yuvToRGB(const RGYFrameInfo &hin, float *dst) {
    const int W = m_W, H = m_H;
    const size_t plane = (size_t)W * H;
    const bool nv12 = (hin.csp == RGY_CSP_NV12);
    const int cw = W / 2, ch = H / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + 1) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1];
    const int cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    float *R = dst, *G = dst + plane, *B = dst + 2 * plane;
    for (int y = 0; y < H; y++) {
        const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
        const int cy = std::min(y / 2, ch - 1);
        for (int x = 0; x < W; x++) {
            const int cx = std::min(x / 2, cw - 1);
            const float yn = ((float)yrow[x] - m_yOff) * m_yScale;
            const float un = ((float)pU[(size_t)cy * cPitchU + (size_t)cx * cStride] - m_cOff) * m_cScale;
            const float vn = ((float)pV[(size_t)cy * cPitchV + (size_t)cx * cStride] - m_cOff) * m_cScale;
            const size_t i = (size_t)y * W + x;
            R[i] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
            G[i] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
            B[i] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
        }
    }
}

// planar RGB [0,1] CHW (3*W*H) -> yv12/nv12 8-bit into the mapped output frame.
void NVEncFilterRifeOV::rgbToYUV(const RGYFrameInfo &hout, const float *src) {
    const int W = m_W, H = m_H;
    const size_t plane = (size_t)W * H;
    const bool nv12 = (hout.csp == RGY_CSP_NV12);
    const int cw = W / 2, chh = H / 2;
    const float *R = src, *G = src + plane, *B = src + 2 * plane;
    uint8_t *oU = hout.ptr[1];
    uint8_t *oV = nv12 ? (hout.ptr[1] + 1) : hout.ptr[2];
    const int oStride = nv12 ? 2 : 1;
    const int oPitchU = hout.pitch[1];
    const int oPitchV = nv12 ? hout.pitch[1] : hout.pitch[2];
    // luma + accumulate chroma at full res, then 4:2:0 box-average.
    for (int y = 0; y < H; y++) {
        uint8_t *yd = hout.ptr[0] + (size_t)y * hout.pitch[0];
        for (int x = 0; x < W; x++) {
            const size_t i = (size_t)y * W + x;
            const float r = R[i], g = G[i], b = B[i];
            const float Yn = m_matRY * r + m_matGY * g + m_matBY * b;
            yd[x] = clamp_u8((int)(Yn * m_yRange + m_yOff + 0.5f));
        }
    }
    for (int cy = 0; cy < chh; cy++) {
        for (int cx = 0; cx < cw; cx++) {
            float u = 0.0f, v = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const size_t i = (size_t)(cy * 2 + dy) * W + (cx * 2 + dx);
                    const float r = R[i], g = G[i], b = B[i];
                    u += m_matRU * r + m_matGU * g + m_matBU * b;
                    v += m_matRV * r + m_matGV * g + m_matBV * b;
                }
            }
            u *= 0.25f; v *= 0.25f;
            oU[(size_t)cy * oPitchU + (size_t)cx * oStride] = clamp_u8((int)(u * m_cRange + m_cOff + 0.5f));
            oV[(size_t)cy * oPitchV + (size_t)cx * oStride] = clamp_u8((int)(v * m_cRange + m_cOff + 0.5f));
        }
    }
}

// build the 11-channel input for time t and run the network -> m_outBuf.
RGY_ERR NVEncFilterRifeOV::interpolate(float t) {
    const size_t plane = (size_t)m_W * m_H;
    float *p = m_inBuf.data();
    memcpy(p + 0 * plane, m_prevRGB.data(), 3 * plane * sizeof(float)); // img0 (3)
    memcpy(p + 3 * plane, m_currRGB.data(), 3 * plane * sizeof(float)); // img1 (3)
    std::fill(p + 6 * plane, p + 7 * plane, t);                         // timestep (1)
    memcpy(p + 7 * plane, m_baseGrid.data(), 2 * plane * sizeof(float));// base_grid (2)
    memcpy(p + 9 * plane, m_multiplier.data(), 2 * plane * sizeof(float)); // multiplier (2)
    return m_ov->infer(m_inBuf.data(), m_outBuf.data());
}

RGYFrameInfo NVEncFilterRifeOV::rgbFrame(float *ptr) const {
    RGYFrameInfo frame;
    frame.width = m_W;
    frame.height = m_H;
    frame.csp = RGY_CSP_RGB_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    const size_t planeBytes = (size_t)m_W * m_H * sizeof(float);
    for (int i = 0; i < 3; i++) {
        frame.ptr[i] = (uint8_t *)ptr + planeBytes * i;
        frame.pitch[i] = m_W * sizeof(float);
    }
    return frame;
}

RGY_ERR NVEncFilterRifeOV::initCudaPath(cudaStream_t stream) {
    if (m_cudaPathTried) return m_cudaPath ? RGY_ERR_NONE : RGY_ERR_UNSUPPORTED;
    m_cudaPathTried = true;
    if (stream == nullptr || !m_inputDevice || !m_outputDevice || !m_cropToRgb || !m_cropFromRgb) {
        return RGY_ERR_UNSUPPORTED;
    }
    auto session = std::make_unique<RGYOnnxRTCUDA>();
    tstring errorMessage;
    auto err = session->init(m_modelPath, m_deviceID, RGYOnnxRTProvider::Auto,
        m_H, m_W, errorMessage, stream);
    if (err != RGY_ERR_NONE || !session->deviceIOAvailable()
        || session->inChannels() != 11 || session->outChannels() != 3
        || session->outWidth() != m_W || session->outHeight() != m_H) {
        const auto reason = !errorMessage.empty() ? errorMessage : session->lastError();
        AddMessage(RGY_LOG_WARN, _T("rife-ov: CUDAゼロコピー経路を初期化できないためホスト経路を使用します: %s\n"), reason.c_str());
        return (err != RGY_ERR_NONE) ? err : RGY_ERR_UNSUPPORTED;
    }
    m_ov = std::move(session);
    m_cudaPath = true;
    AddMessage(RGY_LOG_INFO, _T("rife-ov: path cuda-zerocopy をフィルタストリーム上で初期化しました。\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRifeOV::runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs,
    int *outputCount, cudaStream_t stream) {
    const size_t plane = (size_t)m_W * m_H;
    const size_t planeBytes = plane * sizeof(float);
    auto inputFrame = *input;
    inputFrame.picstruct = RGY_PICSTRUCT_FRAME;
    auto currentRgb = rgbFrame((float *)m_inputDevice->ptr + 3 * plane);
    RGYFrameInfo *rgbOut[1] = { &currentRgb };
    int rgbOutCount = 0;
    auto err = m_cropToRgb->filter(&inputFrame, rgbOut, &rgbOutCount, stream);
    if (err != RGY_ERR_NONE) return err;
    if (!m_havePrev) {
        outputs[0] = &m_frameBuf[0]->frame;
        err = copyFrameAsync(outputs[0], input, stream);
        if (err != RGY_ERR_NONE) return err;
        auto cudaerr = cudaMemcpyAsync(m_inputDevice->ptr,
            (uint8_t *)m_inputDevice->ptr + 3 * planeBytes, 3 * planeBytes,
            cudaMemcpyDeviceToDevice, stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        outputs[0]->timestamp = input->timestamp;
        outputs[0]->duration = input->duration;
        outputs[0]->picstruct = input->picstruct;
        outputs[0]->inputFrameId = input->inputFrameId;
        *outputCount = 1;
        m_prevTimestamp = input->timestamp;
        m_prevDuration = input->duration;
        m_havePrev = true;
        return RGY_ERR_NONE;
    }
    const int64_t spanDuration = input->timestamp - m_prevTimestamp;
    for (int k = 1; k < m_multi; k++) {
        const float t = (float)k / (float)m_multi;
        uint32_t tBits = 0;
        std::memcpy(&tBits, &t, sizeof(tBits));
        const auto cuerr = cuMemsetD32Async((CUdeviceptr)((uint8_t *)m_inputDevice->ptr + 6 * planeBytes),
            tBits, plane, (CUstream)stream);
        if (cuerr != CUDA_SUCCESS) return RGY_ERR_CUDA;
        err = m_ov->inferDevice((const float *)m_inputDevice->ptr, (float *)m_outputDevice->ptr);
        if (err != RGY_ERR_NONE) return err;
        auto rgbOutput = rgbFrame((float *)m_outputDevice->ptr);
        auto output = &m_frameBuf[k - 1]->frame;
        RGYFrameInfo *yuvOut[1] = { output };
        int yuvOutCount = 0;
        err = m_cropFromRgb->filter(&rgbOutput, yuvOut, &yuvOutCount, stream);
        if (err != RGY_ERR_NONE) return err;
        output->timestamp = m_prevTimestamp + (spanDuration > 0 ? spanDuration * (int64_t)k / (int64_t)m_multi : 0);
        output->duration = (spanDuration > 0) ? (spanDuration / m_multi) : input->duration;
        output->picstruct = input->picstruct;
        output->inputFrameId = input->inputFrameId;
        outputs[k - 1] = output;
    }
    auto passthrough = &m_frameBuf[m_multi - 1]->frame;
    err = copyFrameAsync(passthrough, input, stream);
    if (err != RGY_ERR_NONE) return err;
    passthrough->timestamp = input->timestamp;
    passthrough->duration = (spanDuration > 0) ? (spanDuration / m_multi) : input->duration;
    passthrough->picstruct = input->picstruct;
    passthrough->inputFrameId = input->inputFrameId;
    outputs[m_multi - 1] = passthrough;
    const auto cudaerr = cudaMemcpyAsync(m_inputDevice->ptr,
        (uint8_t *)m_inputDevice->ptr + 3 * planeBytes, 3 * planeBytes,
        cudaMemcpyDeviceToDevice, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    *outputCount = m_multi;
    m_prevTimestamp = input->timestamp;
    m_prevDuration = input->duration;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRifeOV::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) { *pOutputFrameNum = 0; return RGY_ERR_NONE; } // flush: drop trailing single frame

    if (initCudaPath(stream) == RGY_ERR_NONE) {
        const auto err = runCuda(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
        if (err == RGY_ERR_NONE) return err;
        AddMessage(RGY_LOG_WARN, _T("rife-ov: CUDAゼロコピー実行に失敗したためホスト経路へフォールバックします: %s.\n"), get_err_mes(err));
        m_cudaPath = false;
        m_havePrev = false;
    }

    // デバイス入力をホストステージングへコピーしてから、CPUでRGBへ変換する。
    auto err = copyFrameAsync(&m_inStaging->frame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: stream sync failed: %s.\n"), get_err_mes(err)); return err; }
    yuvToRGB(m_inStaging->frame, m_currRGB.data());

    if (!m_havePrev) {
        // first frame: emit it unchanged; it becomes the previous frame.
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
        err = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
        if (err != RGY_ERR_NONE) return err;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration  = pInputFrame->duration;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
        *pOutputFrameNum = 1;
        m_prevRGB = m_currRGB;
        m_prevTimestamp = pInputFrame->timestamp;
        m_prevDuration  = pInputFrame->duration;
        m_havePrev = true;
        return RGY_ERR_NONE;
    }

    const int64_t spanDur = pInputFrame->timestamp - m_prevTimestamp;
    // (multi-1) interpolated frames between prev and curr.
    for (int k = 1; k < m_multi; k++) {
        const float t = (float)k / (float)m_multi;
        err = interpolate(t);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: inference failed at t=%.3f.\n"), t); return err; }
        rgbToYUV(m_outStaging->frame, m_outBuf.data());
        RGYFrameInfo *out = &m_frameBuf[k - 1]->frame;
        err = copyFrameAsync(out, &m_outStaging->frame, stream);
        if (err != RGY_ERR_NONE) return err;
        out->timestamp = m_prevTimestamp + (spanDur > 0 ? spanDur * (int64_t)k / (int64_t)m_multi : 0);
        out->duration  = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
        out->picstruct = pInputFrame->picstruct;
        out->inputFrameId = pInputFrame->inputFrameId;
        ppOutputFrames[k - 1] = out;
    }
    // passthrough of the current frame (copied unchanged, no RGB round-trip).
    RGYFrameInfo *passthru = &m_frameBuf[m_multi - 1]->frame;
    err = copyFrameAsync(passthru, pInputFrame, stream);
    if (err != RGY_ERR_NONE) return err;
    passthru->timestamp = pInputFrame->timestamp;
    passthru->duration  = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
    passthru->picstruct = pInputFrame->picstruct;
    passthru->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[m_multi - 1] = passthru;

    *pOutputFrameNum = m_multi;
    m_prevRGB.swap(m_currRGB);
    m_prevTimestamp = pInputFrame->timestamp;
    m_prevDuration  = pInputFrame->duration;
    return RGY_ERR_NONE;
}
