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

#include "NVEncFilterOnnx.h"
#include "rgy_aspect_ratio.h"  // set_auto_resolution() for out_res= negative auto-aspect
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

tstring NVEncFilterParamOnnx::print() const {
    return onnx.print();
}

NVEncFilterOnnx::NVEncFilterOnnx() :
    m_ov(), m_io(OnnxIO::LumaSR), m_inC(1), m_outC(1),
    m_scale(1), m_maxval(255.0f), m_ycbcr(false), m_sigmaNorm(0.0f),
    m_yOff(0.0f), m_yScale(1.0f), m_yRange(255.0f), m_cOff(128.0f), m_cScale(1.0f), m_cRange(255.0f),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_inStaging(), m_outStaging(), m_inBuf(), m_outBuf(), m_u444(), m_v444() {
    m_name = _T("onnx");
}

NVEncFilterOnnx::~NVEncFilterOnnx() {
    close();
}

namespace {
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilinear upscale of one 8-bit channel from (sw x sh) to (sw*scale x sh*scale)
// on the CPU (host path).
static void upscale_bilinear_u8(uint8_t *dst, const int dstPitch, const int dstStride,
                                const uint8_t *src, const int srcPitch, const int srcStride,
                                const int sw, const int sh, const int scale) {
    const int dw = sw * scale;
    const int dh = sh * scale;
    const float inv = 1.0f / (float)scale;
    for (int dy = 0; dy < dh; dy++) {
        float sy = (dy + 0.5f) * inv - 0.5f;
        int y0 = (int)std::floor(sy);
        float fy = sy - (float)y0;
        const uint8_t *row0 = src + (size_t)clampi(y0,     0, sh - 1) * srcPitch;
        const uint8_t *row1 = src + (size_t)clampi(y0 + 1, 0, sh - 1) * srcPitch;
        uint8_t *drow = dst + (size_t)dy * dstPitch;
        for (int dx = 0; dx < dw; dx++) {
            float sx = (dx + 0.5f) * inv - 0.5f;
            int x0 = (int)std::floor(sx);
            float fx = sx - (float)x0;
            const int x0c = clampi(x0,     0, sw - 1) * srcStride;
            const int x1c = clampi(x0 + 1, 0, sw - 1) * srcStride;
            const float a = row0[x0c], b = row0[x1c];
            const float c = row1[x0c], d = row1[x1c];
            const float top = a + (b - a) * fx;
            const float bot = c + (d - c) * fx;
            const int v = (int)(top + (bot - top) * fy + 0.5f);
            drow[dx * dstStride] = (uint8_t)clampi(v, 0, 255);
        }
    }
}

// Bilinearly sample one 8-bit chroma channel (half-res, 4:2:0) at the location of
// luma pixel (lx, ly), upsampling x2. Returns the raw value (0..255) as a float.
static inline float sample_chroma_up2(const uint8_t *plane, const int pitch, const int stride,
                                      const int cw, const int ch, const int lx, const int ly) {
    const float cx = (lx + 0.5f) * 0.5f - 0.5f;
    const float cy = (ly + 0.5f) * 0.5f - 0.5f;
    const int x0 = (int)std::floor(cx); const float fx = cx - (float)x0;
    const int y0 = (int)std::floor(cy); const float fy = cy - (float)y0;
    const int x0c = clampi(x0,     0, cw - 1) * stride;
    const int x1c = clampi(x0 + 1, 0, cw - 1) * stride;
    const uint8_t *r0 = plane + (size_t)clampi(y0,     0, ch - 1) * pitch;
    const uint8_t *r1 = plane + (size_t)clampi(y0 + 1, 0, ch - 1) * pitch;
    const float a = r0[x0c], b = r0[x1c];
    const float c = r1[x0c], d = r1[x1c];
    const float top = a + (b - a) * fx;
    const float bot = c + (d - c) * fx;
    return top + (bot - top) * fy;
}

// 2x2 box-downsample a full-res normalised channel to a half-res 8-bit chroma
// plane, encoding each averaged value as v*encScale + encOff (rounded, clamped).
static void downsample420_encode(uint8_t *dst, const int dstPitch, const int dstStride,
                                 const float *srcFull, const int fullW, const int fullH,
                                 const float encScale, const float encOff, const int pixMax) {
    const int cw = fullW / 2;
    const int ch = fullH / 2;
    for (int cy = 0; cy < ch; cy++) {
        const float *s0 = srcFull + (size_t)(2 * cy)     * fullW;
        const float *s1 = srcFull + (size_t)(2 * cy + 1) * fullW;
        uint8_t *drow = dst + (size_t)cy * dstPitch;
        for (int cx = 0; cx < cw; cx++) {
            const int x0 = 2 * cx;
            const float avg = (s0[x0] + s0[x0 + 1] + s1[x0] + s1[x0 + 1]) * 0.25f;
            const int v = (int)(avg * encScale + encOff + 0.5f);
            drow[cx * dstStride] = (uint8_t)clampi(v, 0, pixMax);
        }
    }
}

// Copy one 8-bit plane (row-by-row, honouring pitches). width is in samples,
// srcStride/dstStride 1 for planar, 2 for nv12-interleaved.
static void copy_plane_u8(uint8_t *dst, const int dstPitch, const int dstStride,
                          const uint8_t *src, const int srcPitch, const int srcStride,
                          const int width, const int height) {
    for (int y = 0; y < height; y++) {
        const uint8_t *srow = src + (size_t)y * srcPitch;
        uint8_t *drow = dst + (size_t)y * dstPitch;
        if (srcStride == 1 && dstStride == 1) {
            memcpy(drow, srow, (size_t)width);
        } else {
            for (int x = 0; x < width; x++) drow[x * dstStride] = srow[x * srcStride];
        }
    }
}
} // namespace

void NVEncFilterOnnx::setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax) {
    float Kr = 0.2126f, Kb = 0.0722f;        // BT.709 default
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

RGY_ERR NVEncFilterOnnx::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamOnnx>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOnnxRTCUDA::available()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: this build of NVEnc was compiled without ONNX Runtime CUDA support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->onnx.modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model= (path to an .onnx model) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->onnx.modelFile.find_first_of(_T("/\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        auto jsonPath = PathCombineS(prm->modelDir, _T("models.json"));
        auto err = registry.load(jsonPath, m_pLog);
        if (err != RGY_ERR_NONE) return err;
        auto entry = registry.find(prm->onnx.modelFile);
        if (!entry) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: model \"%s\" not found in models.json\n"), prm->onnx.modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->onnx.modelFile = registry.resolveModelPath(prm->onnx.modelFile);
        if (prm->onnx.colorspace.empty() || prm->onnx.colorspace == _T("auto")) {
            prm->onnx.colorspace = entry->colorspace;
        }
        if (prm->onnx.noise == 15) {
            prm->onnx.noise = entry->noise;
        }
    }
    if (!rgy_file_exists(prm->onnx.modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model file not found: %s\n"), prm->onnx.modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }

    const int inW = prm->frameIn.width;
    const int inH = prm->frameIn.height;

    // CUDA device ordinal: prefer the one NVEnc passed (the encoder's GPU); if unset,
    // fall back to the device of the current CUDA context (the encoder's context is
    // current when filters init/run).
    int deviceID = prm->deviceID;
    if (deviceID < 0) {
        cudaGetDevice(&deviceID);
    }

    // Provider selection: auto -> CUDA (default), cuda, tensorrt.
    RGYOnnxRTProvider provider = RGYOnnxRTProvider::Auto;
    const tstring provStr = prm->onnx.provider;
    if      (provStr == _T("tensorrt") || provStr == _T("trt")) provider = RGYOnnxRTProvider::TensorRT;
    else if (provStr == _T("cuda"))                              provider = RGYOnnxRTProvider::Cuda;
    else                                                         provider = RGYOnnxRTProvider::Auto;

    m_ov = std::make_unique<RGYOnnxRTCUDA>();
    tstring errMsg;

    RGY_ERR err = m_ov->init(prm->onnx.modelFile, deviceID, provider, inH, inW, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to load/compile model: %s\n"),
            errMsg.c_str());
        return err;
    }

    // Infer the I/O convention from the compiled model's channel counts.
    m_inC  = m_ov->inChannels();
    m_outC = m_ov->outChannels();
    if      (m_inC == 1 && m_outC == 1) m_io = OnnxIO::LumaSR;
    else if (m_inC == 2 && m_outC == 1) m_io = OnnxIO::GrayNoise;
    else if (m_inC == 3 && m_outC == 2) m_io = OnnxIO::Chroma;
    else if (m_inC == 3 && m_outC == 3) m_io = OnnxIO::RGB;
    else if (m_inC == 4 && m_outC == 3) m_io = OnnxIO::RGBNoise;
    else {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported model I/O: %dch in / %dch out.\n"), m_inC, m_outC);
        return RGY_ERR_UNSUPPORTED;
    }

    const int outW = m_ov->outWidth();
    const int outH = m_ov->outHeight();
    if (outW <= 0 || outH <= 0 || (outW % inW) != 0 || (outH % inH) != 0 || (outW / inW) != (outH / inH)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model output %dx%d is not an integer upscale of input %dx%d.\n"),
            outW, outH, inW, inH);
        return RGY_ERR_UNSUPPORTED;
    }
    m_scale  = outW / inW;
    if ((m_io == OnnxIO::GrayNoise || m_io == OnnxIO::Chroma) && m_scale != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: %s model must be scale=1 (got x%d).\n"),
            (m_io == OnnxIO::Chroma) ? _T("chroma") : _T("gray+noise"), m_scale);
        return RGY_ERR_UNSUPPORTED;
    }
    m_maxval = (float)((1 << prm->frameIn.bitdepth) - 1);

    m_ycbcr = (m_io == OnnxIO::Chroma) || (m_io == OnnxIO::RGB && prm->onnx.colorspace == _T("ycbcr"));

    const int noiseClamped = std::max(0, std::min(255, prm->onnx.noise));
    m_sigmaNorm = (float)noiseClamped / 255.0f;

    int matrixSel;
    if      (prm->onnx.colormatrix == _T("bt601"))  matrixSel = 601;
    else if (prm->onnx.colormatrix == _T("bt2020")) matrixSel = 2020;
    else if (prm->onnx.colormatrix == _T("bt709"))  matrixSel = 709;
    else                                            matrixSel = (inH <= 576) ? 601 : 709; // auto
    const bool rangeTV = (prm->onnx.colorrange != _T("pc")); // auto/tv -> TV
    setupColorCoeffs(matrixSel, rangeTV, 255);

    // Output frame buffer at the (possibly upscaled) resolution.
    auto frameOut = prm->frameOut;
    frameOut.csp    = inCsp;
    frameOut.width  = outW;
    frameOut.height = outH;
    prm->frameOut   = frameOut;
    err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // host-readback scratch
    m_inBuf.resize((size_t)m_inC  * inW  * inH);
    m_outBuf.resize(m_ov->outElemCount());
    if ((m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise) && !m_ycbcr) {
        m_u444.resize((size_t)outW * outH);
        m_v444.resize((size_t)outW * outH);
    }
    m_inStaging  = std::make_unique<CUFrameBuf>();
    m_outStaging = std::make_unique<CUFrameBuf>();
    if (m_inStaging->allocHost(inW, inH, inCsp) != RGY_ERR_NONE
        || m_outStaging->allocHost(outW, outH, inCsp) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate host staging frame buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    // Opt-in end-of-chain resize (out_res=): run an internal NVEncFilterResize AFTER
    // the core, landing an arbitrary final resolution in one pass (CNN THEN resize).
    m_postResize.reset();
    if (prm->onnx.postResizeW != 0 && prm->onnx.postResizeH != 0) {
        int tgtW = prm->onnx.postResizeW;
        int tgtH = prm->onnx.postResizeH;
        if (tgtW < 0 || tgtH < 0) {
            sInputCrop nocrop;
            memset(&nocrop, 0, sizeof(nocrop));
            set_auto_resolution(tgtW, tgtH, 1, 1, outW, outH, prm->sar[0], prm->sar[1],
                2, 2, RGYResizeResMode::Normal, false, nocrop);
        }
        if (tgtW > 0 && tgtH > 0 && (tgtW != outW || tgtH != outH)) {
            auto resizeParam = std::make_shared<NVEncFilterParamResize>();
            resizeParam->interp = (prm->onnx.postResizeAlgo == RGY_VPP_RESIZE_AUTO)
                                  ? RGY_VPP_RESIZE_LANCZOS4 : prm->onnx.postResizeAlgo;
            resizeParam->frameIn  = prm->frameOut;             // network output: outW x outH, csp/pitch set above
            resizeParam->frameOut = prm->frameOut;
            resizeParam->frameOut.width  = tgtW;
            resizeParam->frameOut.height = tgtH;
            resizeParam->baseFps       = prm->baseFps;
            resizeParam->bOutOverwrite = false;
            m_postResize = std::make_unique<NVEncFilterResize>();
            auto rsts = m_postResize->init(resizeParam, m_pLog);
            if (rsts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx: failed to init end-of-chain resize: %s.\n"), get_err_mes(rsts));
                return rsts;
            }
            prm->frameOut = resizeParam->frameOut;
        }
    }

    static const TCHAR *ioName[] = { _T("luma-sr"), _T("gray+noise"), _T("chroma"), _T("rgb"), _T("rgb+noise") };
    tstring info = strsprintf(_T("onnx: %s  %dx%d -> %dx%d (x%d)  io=%s%s  backend=%s"),
        PathGetFilename(prm->onnx.modelFile).c_str(), inW, inH, outW, outH, m_scale,
        ioName[(int)m_io], (m_ycbcr && m_io == OnnxIO::RGB) ? _T("(ycbcr)") : _T(""),
        m_ov->providerName().c_str());
    if (m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise || m_io == OnnxIO::Chroma) {
        info += strsprintf(_T(" matrix=bt%d range=%s"), matrixSel, rangeTV ? _T("tv") : _T("pc"));
    }
    if (m_io == OnnxIO::GrayNoise || m_io == OnnxIO::RGBNoise) {
        info += strsprintf(_T(" noise=%d"), noiseClamped);
    }
    if (!m_ov->deviceFullName().empty()) {
        info += strsprintf(_T(" [%s]"), m_ov->deviceFullName().c_str());
    }
    if (!m_ov->inferencePrecision().empty()) {
        info += strsprintf(_T(" prec=%s"), m_ov->inferencePrecision().c_str());
    }
    if (m_postResize) {
        info += strsprintf(_T(" -> out_res %dx%d (%s)"), prm->frameOut.width, prm->frameOut.height,
            get_cx_desc(list_vpp_resize, (prm->onnx.postResizeAlgo == RGY_VPP_RESIZE_AUTO)
                ? RGY_VPP_RESIZE_LANCZOS4 : prm->onnx.postResizeAlgo));
    }
    setFilterInfo(info);
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    // The CNN core writes its (outW x outH) result into m_frameBuf.
    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *coreFrame = &pOutFrame->frame;
    copyFramePropWithoutRes(coreFrame, pInputFrame);

    auto cerr = runHost(pInputFrame, coreFrame, stream);
    if (cerr != RGY_ERR_NONE) {
        return cerr;
    }

    if (!m_postResize) {
        ppOutputFrames[0] = coreFrame;
        *pOutputFrameNum = 1;
        return RGY_ERR_NONE;
    }
    // Resize the core output to the requested resolution. bOutOverwrite=false =>
    // the sub-filter writes into its own buffer and returns it in resizeOut[0].
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    auto rerr = m_postResize->filter(coreFrame, resizeOut, &resizeNum, stream);
    if (rerr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: end-of-chain resize failed: %s.\n"), get_err_mes(rerr));
        return rerr;
    }
    ppOutputFrames[0] = resizeOut[0];
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

// Pack the host input frame into m_inBuf (inC*inW*inH floats, CHW).
void NVEncFilterOnnx::fillInputHost(const RGYFrameInfo &hin) {
    const int inW = hin.width;
    const int inH = hin.height;
    const size_t chSize = (size_t)inW * inH;
    const bool nv12 = (hin.csp == RGY_CSP_NV12);
    const int cw = inW / 2, ch = inH / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + 1) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1];
    const int cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    float *base = m_inBuf.data();

    switch (m_io) {
    case OnnxIO::LumaSR:
    case OnnxIO::GrayNoise:
        for (int y = 0; y < inH; y++) {
            const uint8_t *srow = hin.ptr[0] + (size_t)y * hin.pitch[0];
            float *drow = base + (size_t)y * inW;
            for (int x = 0; x < inW; x++) drow[x] = (float)srow[x] / m_maxval;
        }
        if (m_io == OnnxIO::GrayNoise) {
            std::fill(base + chSize, base + 2 * chSize, m_sigmaNorm);
        }
        break;
    case OnnxIO::Chroma:
        for (int y = 0; y < inH; y++) {
            const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
            float *yd = base + (size_t)y * inW;
            float *ud = base + chSize + (size_t)y * inW;
            float *vd = base + 2 * chSize + (size_t)y * inW;
            for (int x = 0; x < inW; x++) {
                yd[x] = (float)yrow[x] / m_maxval;
                ud[x] = sample_chroma_up2(pU, cPitchU, cStride, cw, ch, x, y) / m_maxval;
                vd[x] = sample_chroma_up2(pV, cPitchV, cStride, cw, ch, x, y) / m_maxval;
            }
        }
        break;
    case OnnxIO::RGB:
    case OnnxIO::RGBNoise:
        if (m_ycbcr) {
            for (int y = 0; y < inH; y++) {
                const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
                float *c0 = base + (size_t)y * inW;
                float *c1 = base + chSize + (size_t)y * inW;
                float *c2 = base + 2 * chSize + (size_t)y * inW;
                for (int x = 0; x < inW; x++) {
                    c0[x] = (float)yrow[x] / m_maxval;
                    c1[x] = sample_chroma_up2(pU, cPitchU, cStride, cw, ch, x, y) / m_maxval;
                    c2[x] = sample_chroma_up2(pV, cPitchV, cStride, cw, ch, x, y) / m_maxval;
                }
            }
        } else {
            for (int y = 0; y < inH; y++) {
                const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
                float *rd = base + (size_t)y * inW;
                float *gd = base + chSize + (size_t)y * inW;
                float *bd = base + 2 * chSize + (size_t)y * inW;
                for (int x = 0; x < inW; x++) {
                    const float yn = ((float)yrow[x] - m_yOff) * m_yScale;
                    const float un = (sample_chroma_up2(pU, cPitchU, cStride, cw, ch, x, y) - m_cOff) * m_cScale;
                    const float vn = (sample_chroma_up2(pV, cPitchV, cStride, cw, ch, x, y) - m_cOff) * m_cScale;
                    rd[x] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
                    gd[x] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
                    bd[x] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
                }
            }
        }
        if (m_io == OnnxIO::RGBNoise) {
            std::fill(base + 3 * chSize, base + 4 * chSize, m_sigmaNorm);
        }
        break;
    }
}

// Unpack m_outBuf (outC*outW*outH floats, CHW) into the host output frame.
void NVEncFilterOnnx::writeOutputHost(const RGYFrameInfo &hout, const RGYFrameInfo &hin) {
    const int outW = hout.width;
    const int outH = hout.height;
    const size_t chSize = (size_t)outW * outH;
    const bool nv12 = (hout.csp == RGY_CSP_NV12);
    const int pixMax = (int)m_maxval;
    const float *ob = m_outBuf.data();
    uint8_t *oU = hout.ptr[1];
    uint8_t *oV = nv12 ? (hout.ptr[1] + 1) : hout.ptr[2];
    const int oStride = nv12 ? 2 : 1;
    const int oPitchU = hout.pitch[1];
    const int oPitchV = nv12 ? hout.pitch[1] : hout.pitch[2];

    switch (m_io) {
    case OnnxIO::LumaSR: {
        for (int y = 0; y < outH; y++) {
            const float *srow = ob + (size_t)y * outW;
            uint8_t *drow = hout.ptr[0] + (size_t)y * hout.pitch[0];
            for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (uint8_t)clampi(v, 0, pixMax); }
        }
        const int cInW = hin.width / 2, cInH = hin.height / 2;
        if (!nv12) {
            upscale_bilinear_u8(hout.ptr[1], hout.pitch[1], 1, hin.ptr[1], hin.pitch[1], 1, cInW, cInH, m_scale);
            upscale_bilinear_u8(hout.ptr[2], hout.pitch[2], 1, hin.ptr[2], hin.pitch[2], 1, cInW, cInH, m_scale);
        } else {
            upscale_bilinear_u8(hout.ptr[1] + 0, hout.pitch[1], 2, hin.ptr[1] + 0, hin.pitch[1], 2, cInW, cInH, m_scale);
            upscale_bilinear_u8(hout.ptr[1] + 1, hout.pitch[1], 2, hin.ptr[1] + 1, hin.pitch[1], 2, cInW, cInH, m_scale);
        }
        break;
    }
    case OnnxIO::GrayNoise: {
        for (int y = 0; y < outH; y++) {
            const float *srow = ob + (size_t)y * outW;
            uint8_t *drow = hout.ptr[0] + (size_t)y * hout.pitch[0];
            for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (uint8_t)clampi(v, 0, pixMax); }
        }
        const int cw = hin.width / 2, chh = hin.height / 2;
        const uint8_t *iU = hin.ptr[1];
        const uint8_t *iV = nv12 ? (hin.ptr[1] + 1) : hin.ptr[2];
        const int iStride = nv12 ? 2 : 1;
        const int iPitchU = hin.pitch[1], iPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
        copy_plane_u8(oU, oPitchU, oStride, iU, iPitchU, iStride, cw, chh);
        copy_plane_u8(oV, oPitchV, oStride, iV, iPitchV, iStride, cw, chh);
        break;
    }
    case OnnxIO::Chroma:
        copy_plane_u8(hout.ptr[0], hout.pitch[0], 1, hin.ptr[0], hin.pitch[0], 1, outW, outH);
        downsample420_encode(oU, oPitchU, oStride, ob + 0 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        downsample420_encode(oV, oPitchV, oStride, ob + 1 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        break;
    case OnnxIO::RGB:
    case OnnxIO::RGBNoise:
        if (m_ycbcr) {
            for (int y = 0; y < outH; y++) {
                const float *srow = ob + (size_t)y * outW;
                uint8_t *drow = hout.ptr[0] + (size_t)y * hout.pitch[0];
                for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (uint8_t)clampi(v, 0, pixMax); }
            }
            downsample420_encode(oU, oPitchU, oStride, ob + 1 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
            downsample420_encode(oV, oPitchV, oStride, ob + 2 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        } else {
            for (int y = 0; y < outH; y++) {
                const float *rr = ob + 0 * chSize + (size_t)y * outW;
                const float *gg = ob + 1 * chSize + (size_t)y * outW;
                const float *bb = ob + 2 * chSize + (size_t)y * outW;
                uint8_t *yd = hout.ptr[0] + (size_t)y * hout.pitch[0];
                float *un = m_u444.data() + (size_t)y * outW;
                float *vn = m_v444.data() + (size_t)y * outW;
                for (int x = 0; x < outW; x++) {
                    const float R = rr[x], G = gg[x], B = bb[x];
                    const float Yn = m_matRY * R + m_matGY * G + m_matBY * B;
                    un[x] = m_matRU * R + m_matGU * G + m_matBU * B;
                    vn[x] = m_matRV * R + m_matGV * G + m_matBV * B;
                    const int v = (int)(Yn * m_yRange + m_yOff + 0.5f);
                    yd[x] = (uint8_t)clampi(v, 0, pixMax);
                }
            }
            downsample420_encode(oU, oPitchU, oStride, m_u444.data(), outW, outH, m_cRange, m_cOff, pixMax);
            downsample420_encode(oV, oPitchV, oStride, m_v444.data(), outW, outH, m_cRange, m_cOff, pixMax);
        }
        break;
    }
}

RGY_ERR NVEncFilterOnnx::runHost(const RGYFrameInfo *in, RGYFrameInfo *out, cudaStream_t stream) {
    // 1. device input -> host staging, then wait for the copy so the CPU can read it.
    auto err = copyFrameAsync(&m_inStaging->frame, in, stream);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: stream sync failed: %s.\n"), get_err_mes(err)); return err; }

    // 2. pack the input frame into the network tensor (per I/O mode).
    fillInputHost(m_inStaging->frame);

    // 3. inference.
    err = m_ov->infer(m_inBuf.data(), m_outBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n"));
        return err;
    }

    // 4. unpack the network output into the host output staging frame (per I/O mode).
    writeOutputHost(m_outStaging->frame, m_inStaging->frame);

    // 5. copy host staging -> device output.
    err = copyFrameAsync(out, &m_outStaging->frame, stream);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy staging to output failed: %s.\n"), get_err_mes(err)); return err; }
    return RGY_ERR_NONE;
}

void NVEncFilterOnnx::close() {
    m_postResize.reset();
    m_inStaging.reset();
    m_outStaging.reset();
    m_ov.reset();
    m_inBuf.clear();
    m_outBuf.clear();
    m_u444.clear();
    m_v444.clear();
    m_frameBuf.clear();
}
