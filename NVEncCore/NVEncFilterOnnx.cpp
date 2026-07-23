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
#include "rgy_avutil.h"
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
    m_inStaging(), m_outStaging(), m_inBuf(), m_outBuf(), m_u444(), m_v444(),
    m_temporalT(1), m_ring(), m_ringBaseIdx(0), m_recvCount(0), m_emitCount(0),
    m_maskModelW(0), m_maskModelH(0), m_imgPortIdx(0), m_mskPortIdx(1), m_maskOutScale(0.0f),
    m_maskFrame(), m_maskModel(), m_frameRGB(), m_modelIn(), m_modelOut(), m_maskMode(false) {
    m_name = _T("onnx");
}

NVEncFilterOnnx::~NVEncFilterOnnx() {
    close();
}

namespace {
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static const TCHAR *cx_desc_or_unknown(const CX_DESC *list, int value) {
    const auto desc = get_cx_desc(list, value);
    return (desc != nullptr) ? desc : _T("unknown");
}

static bool onnx_matrix_to_coeff_id(CspMatrix matrix, int inputHeight, int& matrixSel) {
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

static bool onnx_supported_colorrange(CspColorRange range) {
    return range == RGY_COLORRANGE_AUTO
        || range == RGY_COLORRANGE_LIMITED
        || range == RGY_COLORRANGE_FULL;
}

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

// マスク画像を1フレームだけ読み込み、0..1の輝度配列に変換する。
#if ENABLE_AVSW_READER
static RGY_ERR loadMaskGray(const tstring &path, std::vector<float> &gray, int &width, int &height, tstring &message) {
    const auto pathA = tchar_to_string(path, CP_UTF8);
    AVFormatContext *fmt = nullptr;
    if (avformat_open_input(&fmt, pathA.c_str(), nullptr, nullptr) != 0) {
        message = _T("マスク画像を開けません");
        return RGY_ERR_FILE_OPEN;
    }
    AVCodecContext *dec = nullptr;
    AVFrame *frame = av_frame_alloc();
    AVPacket *pkt = av_packet_alloc();
    RGY_ERR ret = RGY_ERR_INVALID_DATA_TYPE;
    do {
        if (avformat_find_stream_info(fmt, nullptr) < 0) break;
        int stream = -1;
        for (unsigned int i = 0; i < fmt->nb_streams; i++) {
            if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { stream = (int)i; break; }
        }
        if (stream < 0) break;
        const AVCodec *codec = avcodec_find_decoder(fmt->streams[stream]->codecpar->codec_id);
        if (!codec) break;
        dec = avcodec_alloc_context3(codec);
        if (!dec || avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar) < 0 || avcodec_open2(dec, codec, nullptr) < 0) break;
        bool got = false;
        while (!got && av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index == stream && avcodec_send_packet(dec, pkt) == 0 && avcodec_receive_frame(dec, frame) == 0) got = true;
            av_packet_unref(pkt);
        }
        if (!got) { avcodec_send_packet(dec, nullptr); got = (avcodec_receive_frame(dec, frame) == 0); }
        if (!got) break;
        width = frame->width; height = frame->height;
        gray.resize((size_t)width * height);
        const auto fmtpix = (AVPixelFormat)frame->format;
        int bpp = 0, coff = 0;
        if (fmtpix == AV_PIX_FMT_RGB24 || fmtpix == AV_PIX_FMT_BGR24) bpp = 3;
        else if (fmtpix == AV_PIX_FMT_RGBA || fmtpix == AV_PIX_FMT_BGRA) bpp = 4;
        else if (fmtpix == AV_PIX_FMT_ARGB || fmtpix == AV_PIX_FMT_ABGR) { bpp = 4; coff = 1; }
        if (bpp > 0) {
            for (int y = 0; y < height; y++) {
                const uint8_t *src = frame->data[0] + (size_t)y * frame->linesize[0];
                for (int x = 0; x < width; x++) {
                    const uint8_t *px = src + (size_t)x * bpp + coff;
                    gray[(size_t)y * width + x] = (px[0] + px[1] + px[2]) / (3.0f * 255.0f);
                }
            }
            ret = RGY_ERR_NONE;
        } else if (fmtpix == AV_PIX_FMT_GRAY8 || fmtpix == AV_PIX_FMT_YUV420P || fmtpix == AV_PIX_FMT_YUVJ420P || fmtpix == AV_PIX_FMT_NV12) {
            for (int y = 0; y < height; y++) {
                const uint8_t *src = frame->data[0] + (size_t)y * frame->linesize[0];
                for (int x = 0; x < width; x++) gray[(size_t)y * width + x] = src[x] / 255.0f;
            }
            ret = RGY_ERR_NONE;
        } else {
            message = _T("対応していないマスク画像形式です");
        }
    } while (false);
    av_packet_free(&pkt); av_frame_free(&frame);
    if (dec) avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return ret;
}
#else
static RGY_ERR loadMaskGray(const tstring &, std::vector<float> &, int &, int &, tstring &message) {
    message = _T("このビルドではマスク画像の読み込みに対応していません");
    return RGY_ERR_UNSUPPORTED;
}
#endif

static void resizeMaskPlane(const float *src, int sw, int sh, float *dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        const float fy = (dh > 1) ? ((y + 0.5f) * sh / dh - 0.5f) : 0.0f;
        int y0 = (int)std::floor(fy); const float wy = fy - y0;
        const int y1 = std::min(y0 + 1, sh - 1); y0 = std::max(y0, 0);
        for (int x = 0; x < dw; x++) {
            const float fx = (dw > 1) ? ((x + 0.5f) * sw / dw - 0.5f) : 0.0f;
            int x0 = (int)std::floor(fx); const float wx = fx - x0;
            const int x1 = std::min(x0 + 1, sw - 1); x0 = std::max(x0, 0);
            const float a = src[(size_t)y0 * sw + x0] * (1.0f - wx) + src[(size_t)y0 * sw + x1] * wx;
            const float b = src[(size_t)y1 * sw + x0] * (1.0f - wx) + src[(size_t)y1 * sw + x1] * wx;
            dst[(size_t)y * dw + x] = a * (1.0f - wy) + b * wy;
        }
    }
}

static inline float sampleMaskPlane(const float *src, int sw, int sh, float fx, float fy) {
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    const float wx = fx - x0, wy = fy - y0;
    const int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
    x0 = std::max(x0, 0); y0 = std::max(y0, 0);
    const float a = src[(size_t)y0 * sw + x0] * (1.0f - wx) + src[(size_t)y0 * sw + x1] * wx;
    const float b = src[(size_t)y1 * sw + x0] * (1.0f - wx) + src[(size_t)y1 * sw + x1] * wx;
    return a * (1.0f - wy) + b * wy;
}

void NVEncFilterOnnx::setupColorCoeffs(int matrixSelIn, int matrixSelOut, bool rangeTV, int pixMax) {
    float Kr = 0.2126f, Kb = 0.0722f;        // BT.709 default
    if (matrixSelIn == 601)  { Kr = 0.299f;  Kb = 0.114f; }
    if (matrixSelIn == 2020) { Kr = 0.2627f; Kb = 0.0593f; }
    const float Kg = 1.0f - Kr - Kb;
    m_matVR = 2.0f * (1.0f - Kr);
    m_matUG = -2.0f * Kb * (1.0f - Kb) / Kg;
    m_matVG = -2.0f * Kr * (1.0f - Kr) / Kg;
    m_matUB = 2.0f * (1.0f - Kb);
    float Kr2 = 0.2126f, Kb2 = 0.0722f;      // BT.709 default
    if (matrixSelOut == 601)  { Kr2 = 0.299f;  Kb2 = 0.114f; }
    if (matrixSelOut == 2020) { Kr2 = 0.2627f; Kb2 = 0.0593f; }
    const float Kg2 = 1.0f - Kr2 - Kb2;
    m_matRY = Kr2;                             m_matGY = Kg2;                             m_matBY = Kb2;
    m_matRU = -Kr2 / (2.0f * (1.0f - Kb2));    m_matGU = -Kg2 / (2.0f * (1.0f - Kb2));    m_matBU = 0.5f;
    m_matRV = 0.5f;                            m_matGV = -Kg2 / (2.0f * (1.0f - Kr2));    m_matBV = -Kb2 / (2.0f * (1.0f - Kr2));
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
        if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO && entry->colormatrixOut != RGY_MATRIX_UNSPECIFIED) {
            prm->onnx.colormatrixOut = entry->colormatrixOut;
        }
    }
    int matrixSel = 0;
    if (!onnx_matrix_to_coeff_id(prm->onnx.colormatrix, prm->frameIn.height, matrixSel)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colormatrix %s.\n"),
            cx_desc_or_unknown(list_colormatrix, prm->onnx.colormatrix));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, prm->frameIn.height, matrixSel)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colormatrix_out %s.\n"),
            cx_desc_or_unknown(list_colormatrix, prm->onnx.colormatrixOut));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_supported_colorrange(prm->onnx.colorrange)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colorrange %s.\n"),
            cx_desc_or_unknown(list_colorrange, prm->onnx.colorrange));
        return RGY_ERR_UNSUPPORTED;
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

    if (provider == RGYOnnxRTProvider::TensorRT) {
        AddMessage(RGY_LOG_INFO, prm->onnx.cacheDir.empty()
            ? _T("onnx: building TensorRT engine (this may take minutes)...\n")
            : _T("onnx: building/loading TensorRT engine (first run per model/resolution/precision may take minutes)...\n"));
    }
    RGY_ERR err = m_ov->init(prm->onnx.modelFile, deviceID, provider, inH, inW, errMsg,
        nullptr, prm->onnx.precision, prm->onnx.cacheDir);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to load/compile model: %s\n"),
            errMsg.c_str());
        return err;
    }
    if ((provStr == _T("tensorrt") || provStr == _T("trt")) && m_ov->providerName() != _T("tensorrt")) {
        AddMessage(RGY_LOG_WARN, _T("onnx: TensorRT provider is unavailable; falling back to CUDA: %s\n"),
            m_ov->lastError().c_str());
    } else if (!m_ov->lastError().empty()) {
        AddMessage(RGY_LOG_WARN, _T("onnx: %s\n"), m_ov->lastError().c_str());
    }
    if (!m_ov->cacheInfo().empty()) {
        AddMessage(RGY_LOG_INFO, _T("onnx: %s\n"), m_ov->cacheInfo().c_str());
    }

    // Infer the I/O convention from the compiled model's channel counts.
    m_inC  = m_ov->inChannels();
    m_outC = m_ov->outChannels();
    if (!prm->onnx.maskFile.empty()) {
        if (prm->onnx.frames > 1) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: mask=とframes=は同時に指定できません。\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        return initMask(prm, inW, inH, inCsp);
    }
    if (m_ov->inWidth() != inW || m_ov->inHeight() != inH) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model input %dx%d does not match the frame size %dx%d.\n"),
            m_ov->inWidth(), m_ov->inHeight(), inW, inH);
        return RGY_ERR_UNSUPPORTED;
    }
    m_temporalT = std::max(1, prm->onnx.frames);
    if (m_temporalT > 1) {
        if (m_inC != m_temporalT * 3 || m_outC != 3) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: frames=%dには%dch入力と3ch出力のRGBモデルが必要です（現在%dch/%dch）。\n"),
                m_temporalT, m_temporalT * 3, m_inC, m_outC);
            return RGY_ERR_UNSUPPORTED;
        }
        m_io = OnnxIO::RGB;
        m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
            (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    } else if (m_inC == 1 && m_outC == 1) m_io = OnnxIO::LumaSR;
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

    int matrixSelOut = 0;
    onnx_matrix_to_coeff_id(prm->onnx.colormatrix, inH, matrixSel);
    if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO) {
        matrixSelOut = matrixSel;
    } else {
        onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, inH, matrixSelOut);
    }
    const bool rangeTV = (prm->onnx.colorrange != RGY_COLORRANGE_FULL);
    setupColorCoeffs(matrixSel, matrixSelOut, rangeTV, 255);

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
        if (matrixSelOut != matrixSel) {
            info += strsprintf(_T(" matrix_out=bt%d"), matrixSelOut);
        }
    }
    if (m_io == OnnxIO::GrayNoise || m_io == OnnxIO::RGBNoise) {
        info += strsprintf(_T(" noise=%d"), noiseClamped);
    }
    if (m_temporalT > 1) {
        info += strsprintf(_T(" frames=%d"), m_temporalT);
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
    if (m_maskMode) {
        return runMask(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
    }
    if (m_temporalT > 1) {
        return runTemporal(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
    }
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

void NVEncFilterOnnx::packFrameRGB(const RGYFrameInfo &hin, float *dst) {
    const int w = hin.width, h = hin.height;
    const size_t chSize = (size_t)w * h;
    const bool nv12 = (hin.csp == RGY_CSP_NV12);
    const int cw = w / 2, ch = h / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + 1) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1], cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    if (m_ycbcr) {
        for (int y = 0; y < h; y++) {
            const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
            float *c0 = dst + (size_t)y * w;
            float *c1 = dst + chSize + (size_t)y * w;
            float *c2 = dst + 2 * chSize + (size_t)y * w;
            for (int x = 0; x < w; x++) {
                c0[x] = (float)yrow[x] / m_maxval;
                c1[x] = sample_chroma_up2(pU, cPitchU, cStride, cw, ch, x, y) / m_maxval;
                c2[x] = sample_chroma_up2(pV, cPitchV, cStride, cw, ch, x, y) / m_maxval;
            }
        }
    } else {
        for (int y = 0; y < h; y++) {
            const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
            float *rd = dst + (size_t)y * w;
            float *gd = dst + chSize + (size_t)y * w;
            float *bd = dst + 2 * chSize + (size_t)y * w;
            for (int x = 0; x < w; x++) {
                const float yn = ((float)yrow[x] - m_yOff) * m_yScale;
                const float un = (sample_chroma_up2(pU, cPitchU, cStride, cw, ch, x, y) - m_cOff) * m_cScale;
                const float vn = (sample_chroma_up2(pV, cPitchV, cStride, cw, ch, x, y) - m_cOff) * m_cScale;
                rd[x] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
                gd[x] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
                bd[x] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
            }
        }
    }
}

RGY_ERR NVEncFilterOnnx::runTemporal(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    const int k = (m_temporalT - 1) / 2;
    const size_t frameSize = (size_t)m_ov->inWidth() * m_ov->inHeight();
    if (pInputFrame->ptr[0] != nullptr) {
        auto err = copyFrameAsync(&m_inStaging->frame, pInputFrame, stream);
        if (err != RGY_ERR_NONE) return err;
        err = err_to_rgy(cudaStreamSynchronize(stream));
        if (err != RGY_ERR_NONE) return err;
        RingFrame rf;
        rf.rgb.resize(3 * frameSize);
        packFrameRGB(m_inStaging->frame, rf.rgb.data());
        rf.timestamp = pInputFrame->timestamp;
        rf.duration = pInputFrame->duration;
        rf.picstruct = pInputFrame->picstruct;
        rf.flags = pInputFrame->flags;
        rf.inputFrameId = pInputFrame->inputFrameId;
        rf.dataList = pInputFrame->dataList;
        m_ring.push_back(std::move(rf));
        m_recvCount++;
        while ((int)m_ring.size() > m_temporalT) { m_ring.pop_front(); m_ringBaseIdx++; }
        if (m_recvCount - 1 >= m_emitCount + k) {
            return emitTemporalOutput(m_emitCount++, ppOutputFrames, pOutputFrameNum, stream);
        }
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    if (m_emitCount < m_recvCount) {
        return emitTemporalOutput(m_emitCount++, ppOutputFrames, pOutputFrameNum, stream);
    }
    *pOutputFrameNum = 0;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::initMask(const std::shared_ptr<NVEncFilterParamOnnx> &prm, int inW, int inH, RGY_CSP inCsp) {
    if (prm->onnx.postResizeW != 0 || prm->onnx.postResizeH != 0) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask=とout_res=は同時に指定できません。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->inputCount() != 2 || m_ov->outputCount() != 1 || m_ov->outChannels() != 3) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask=には2入力（画像+マスク）と3ch出力のモデルが必要です。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->inputChannels(0) == 3 && m_ov->inputChannels(1) == 1) {
        m_imgPortIdx = 0; m_mskPortIdx = 1;
    } else if (m_ov->inputChannels(0) == 1 && m_ov->inputChannels(1) == 3) {
        m_imgPortIdx = 1; m_mskPortIdx = 0;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask=には3ch画像入力と1chマスク入力が必要です。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    m_maskModelW = m_ov->inputWidth(m_imgPortIdx);
    m_maskModelH = m_ov->inputHeight(m_imgPortIdx);
    if (m_maskModelW <= 0 || m_maskModelH <= 0
        || m_ov->inputWidth(m_mskPortIdx) != m_maskModelW || m_ov->inputHeight(m_mskPortIdx) != m_maskModelH
        || m_ov->inputWidth(0) <= 0 || m_ov->outWidth() != m_maskModelW || m_ov->outHeight() != m_maskModelH) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask=の入力・出力サイズが一致していません。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!rgy_file_exists(prm->onnx.maskFile)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: maskファイルが見つかりません: %s\n"), prm->onnx.maskFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    std::vector<float> native;
    int maskW = 0, maskH = 0;
    tstring errMessage;
    auto err = loadMaskGray(prm->onnx.maskFile, native, maskW, maskH, errMessage);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask画像の読み込みに失敗しました: %s\n"), errMessage.c_str());
        return err;
    }
    m_maskFrame.resize((size_t)inW * inH);
    m_maskModel.resize((size_t)m_maskModelW * m_maskModelH);
    resizeMaskPlane(native.data(), maskW, maskH, m_maskFrame.data(), inW, inH);
    resizeMaskPlane(native.data(), maskW, maskH, m_maskModel.data(), m_maskModelW, m_maskModelH);
    for (auto &v : m_maskFrame) v = (v >= 0.5f) ? 1.0f : 0.0f;
    for (auto &v : m_maskModel) v = (v >= 0.5f) ? 1.0f : 0.0f;

    m_maskMode = true;
    m_io = OnnxIO::RGB;
    m_ycbcr = false;
    m_scale = 1;
    m_maxval = 255.0f;
    m_frameRGB.resize((size_t)3 * inW * inH);
    m_modelIn.resize((size_t)3 * m_maskModelW * m_maskModelH);
    m_modelOut.resize((size_t)3 * m_maskModelW * m_maskModelH);
    m_outBuf.resize((size_t)3 * inW * inH);
    m_u444.resize((size_t)inW * inH);
    m_v444.resize((size_t)inW * inH);
    int matrixSel = 0, matrixSelOut = 0;
    onnx_matrix_to_coeff_id(prm->onnx.colormatrix, inH, matrixSel);
    if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO) matrixSelOut = matrixSel;
    else onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, inH, matrixSelOut);
    setupColorCoeffs(matrixSel, matrixSelOut, prm->onnx.colorrange != RGY_COLORRANGE_FULL, 255);

    auto frameOut = prm->frameOut;
    frameOut.csp = inCsp; frameOut.width = inW; frameOut.height = inH;
    prm->frameOut = frameOut;
    err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) return err;
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    m_inStaging = std::make_unique<CUFrameBuf>();
    m_outStaging = std::make_unique<CUFrameBuf>();
    if (m_inStaging->allocHost(inW, inH, inCsp) != RGY_ERR_NONE || m_outStaging->allocHost(inW, inH, inCsp) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask stagingバッファの確保に失敗しました。\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    setFilterInfo(strsprintf(_T("onnx: %s %dx%d io=rgb+mask (model %dx%d) backend=%s"),
        PathGetFilename(prm->onnx.modelFile).c_str(), inW, inH, m_maskModelW, m_maskModelH, m_ov->providerName().c_str()));
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::runMask(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) { *pOutputFrameNum = 0; return RGY_ERR_NONE; }
    const size_t frameSize = (size_t)pInputFrame->width * pInputFrame->height;
    const size_t modelSize = (size_t)m_maskModelW * m_maskModelH;
    auto err = copyFrameAsync(&m_inStaging->frame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) return err;
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;
    packFrameRGB(m_inStaging->frame, m_frameRGB.data());
    for (int c = 0; c < 3; c++) {
        resizeMaskPlane(m_frameRGB.data() + c * frameSize, pInputFrame->width, pInputFrame->height,
            m_modelIn.data() + c * modelSize, m_maskModelW, m_maskModelH);
    }
    std::vector<const float *> inputs(2);
    inputs[m_imgPortIdx] = m_modelIn.data(); inputs[m_mskPortIdx] = m_maskModel.data();
    std::vector<float *> outputs = { m_modelOut.data() };
    err = m_ov->inferMulti(inputs, outputs);
    if (err != RGY_ERR_NONE) return err;
    if (m_maskOutScale == 0.0f) {
        float maxv = 0.0f; for (const auto v : m_modelOut) maxv = std::max(maxv, v);
        m_maskOutScale = (maxv > 2.0f) ? (1.0f / 255.0f) : 1.0f;
    }
    for (int c = 0; c < 3; c++) {
        const float *src = m_modelOut.data() + c * modelSize;
        const float *base = m_frameRGB.data() + c * frameSize;
        float *dst = m_outBuf.data() + c * frameSize;
        for (int y = 0; y < pInputFrame->height; y++) for (int x = 0; x < pInputFrame->width; x++) {
            const size_t pos = (size_t)y * pInputFrame->width + x;
            const float fx = ((x + 0.5f) * m_maskModelW) / pInputFrame->width - 0.5f;
            const float fy = ((y + 0.5f) * m_maskModelH) / pInputFrame->height - 0.5f;
            dst[pos] = m_maskFrame[pos] > 0.5f ? clampf(sampleMaskPlane(src, m_maskModelW, m_maskModelH, fx, fy) * m_maskOutScale, 0.0f, 1.0f) : base[pos];
        }
    }
    writeOutputHost(m_outStaging->frame, m_inStaging->frame);
    auto coreFrame = &m_frameBuf[0]->frame;
    err = copyFrameAsync(coreFrame, &m_outStaging->frame, stream);
    if (err != RGY_ERR_NONE) return err;
    copyFramePropWithoutRes(coreFrame, pInputFrame);
    ppOutputFrames[0] = coreFrame; *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::emitTemporalOutput(int64_t outIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    const int k = (m_temporalT - 1) / 2;
    const int ringN = (int)m_ring.size();
    if (ringN <= 0) { *pOutputFrameNum = 0; return RGY_ERR_NONE; }
    const size_t frameSize = (size_t)m_ov->inWidth() * m_ov->inHeight();
    for (int j = 0; j < m_temporalT; j++) {
        int64_t r = outIdx - k + j - m_ringBaseIdx;
        if (r < 0) r = 0; else if (r >= ringN) r = ringN - 1;
        std::copy(m_ring[(size_t)r].rgb.begin(), m_ring[(size_t)r].rgb.end(),
            m_inBuf.begin() + (size_t)j * 3 * frameSize);
    }
    auto err = m_ov->infer(m_inBuf.data(), m_outBuf.data());
    if (err != RGY_ERR_NONE) return err;
    writeOutputHost(m_outStaging->frame, m_inStaging->frame);
    auto coreFrame = &m_frameBuf[0]->frame;
    err = copyFrameAsync(coreFrame, &m_outStaging->frame, stream);
    if (err != RGY_ERR_NONE) return err;
    int64_t cr = outIdx - m_ringBaseIdx;
    if (cr < 0) cr = 0; else if (cr >= ringN) cr = ringN - 1;
    const auto &centre = m_ring[(size_t)cr];
    coreFrame->timestamp = centre.timestamp;
    coreFrame->duration = centre.duration;
    coreFrame->picstruct = centre.picstruct;
    coreFrame->flags = centre.flags;
    coreFrame->inputFrameId = centre.inputFrameId;
    coreFrame->dataList = centre.dataList;
    if (!m_postResize) {
        ppOutputFrames[0] = coreFrame;
        *pOutputFrameNum = 1;
        return RGY_ERR_NONE;
    }
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    err = m_postResize->filter(coreFrame, resizeOut, &resizeNum, stream);
    if (err != RGY_ERR_NONE) return err;
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
        const auto lastError = m_ov->lastError();
        if (!lastError.empty()) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed: %s.\n"), lastError.c_str());
        } else {
            AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n"));
        }
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
    m_ring.clear();
    m_ringBaseIdx = 0;
    m_recvCount = 0;
    m_emitCount = 0;
    m_temporalT = 1;
    m_maskModelW = m_maskModelH = 0;
    m_maskOutScale = 0.0f;
    m_maskFrame.clear();
    m_maskModel.clear();
    m_frameRGB.clear();
    m_modelIn.clear();
    m_modelOut.clear();
    m_maskMode = false;
    m_frameBuf.clear();
}
