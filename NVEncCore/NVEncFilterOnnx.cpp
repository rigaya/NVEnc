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
#include <cuda.h>
#include <cuda_runtime.h>

tstring NVEncFilterParamOnnx::print() const {
    return onnx.print();
}

NVEncFilterOnnx::NVEncFilterOnnx() :
    m_ov(), m_io(OnnxIO::LumaSR), m_inC(1), m_outC(1),
    m_modelInW(0), m_modelInH(0), m_padL(0), m_padT(0), m_scale(1), m_maxval(255.0f), m_ycbcr(false), m_sigmaNorm(0.0f),
    m_yOff(0.0f), m_yScale(1.0f), m_yRange(255.0f), m_cOff(128.0f), m_cScale(1.0f), m_cRange(255.0f),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_inBuf(), m_outBuf(),
    m_inputDevice(), m_outputDevice(), m_yuvInputDevice(), m_cropToYuv444(), m_cropFromYuv444(), m_lumaResize(),
    m_modelPath(), m_provider(RGYOnnxRTProvider::Auto), m_precision(), m_cacheDir(), m_deviceID(-1),
    m_cudaPathTried(false), m_cudaPath(false),
    m_temporalT(1), m_ring(), m_ringBaseIdx(0), m_recvCount(0), m_emitCount(0),
    m_maskModelW(0), m_maskModelH(0), m_imgPortIdx(0), m_mskPortIdx(1), m_maskOutScale(0.0f),
    m_maskFrame(), m_maskModel(), m_maskFrameDevice(), m_maskModelDevice(), m_frameRGBDevice(), m_compositeDevice(), m_maskRangeDevice(), m_maskMode(false) {
    m_name = _T("onnx");
}

NVEncFilterOnnx::~NVEncFilterOnnx() {
    close();
}

namespace {

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

static bool onnx_supported_csp(RGY_CSP csp) {
    switch (csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_09:
    case RGY_CSP_YV12_10:
    case RGY_CSP_YV12_12:
    case RGY_CSP_YV12_14:
    case RGY_CSP_YV12_16:
        return true;
    default:
        return false;
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
    // 上位bitdepthのlimited-rangeも、元の8bitコード値を整数倍して扱う。
    const float depthScale = (float)(pixMax + 1) / 256.0f;
    m_yOff   = rangeTV ? (16.0f  * depthScale) : 0.0f;
    m_yRange = rangeTV ? (219.0f * depthScale) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * depthScale) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * depthScale) : (float)pixMax;
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
        if (prm->onnx.frames == 1 && entry->frames > 1) {
            prm->onnx.frames = entry->frames;
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
    if (!onnx_supported_csp(inCsp)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: 対応する入力形式はYUV420の8/9/10/12/14/16bitのみです: %s %dbit。\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }

    const int inW = prm->frameIn.width;
    const int inH = prm->frameIn.height;
    m_modelInW = inW; m_modelInH = inH; m_padL = m_padT = 0;

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
    m_modelPath = prm->onnx.modelFile;
    m_provider = provider;
    m_precision = prm->onnx.precision;
    m_cacheDir = prm->onnx.cacheDir;
    m_deviceID = deviceID;

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

    int outW = m_ov->outWidth();
    int outH = m_ov->outHeight();
    auto exactScale = [&]() {
        return outW > 0 && outH > 0 && outW % inW == 0 && outH % inH == 0
            && outW / inW == outH / inH;
    };
    if (!exactScale()) {
        // valid畳み込みによる対称な縮小を入力端の複製で補い、出力解像度を維持する。
        const int guessedScale = std::max(1, (int)std::round((double)outW / inW));
        const int cropW = guessedScale * inW - outW, cropH = guessedScale * inH - outH;
        if (cropW <= 0 || cropH <= 0 || cropW != cropH || cropW % guessedScale != 0) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: モデル出力%dx%dは入力%dx%dの整数倍ではありません。\n"), outW, outH, inW, inH);
            return RGY_ERR_UNSUPPORTED;
        }
        const int padTotal = cropW / guessedScale;
        m_padL = m_padT = padTotal / 2;
        m_modelInW = inW + padTotal; m_modelInH = inH + padTotal;
        err = m_ov->init(m_modelPath, m_deviceID, m_provider, m_modelInH, m_modelInW,
            errMsg, nullptr, m_precision, m_cacheDir);
        if (err != RGY_ERR_NONE) return err;
        outW = m_ov->outWidth(); outH = m_ov->outHeight();
        if (!exactScale()) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: パディング後もモデル出力%dx%dは入力%dx%dの整数倍になりません。\n"), outW, outH, inW, inH);
            return RGY_ERR_UNSUPPORTED;
        }
    }
    m_scale = outW / inW;
    if ((m_io == OnnxIO::GrayNoise || m_io == OnnxIO::Chroma) && m_scale != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: %s model must be scale=1 (got x%d).\n"),
            (m_io == OnnxIO::Chroma) ? _T("chroma") : _T("gray+noise"), m_scale);
        return RGY_ERR_UNSUPPORTED;
    }
    m_maxval = (float)((1 << RGY_CSP_BIT_DEPTH[inCsp]) - 1);

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
    setupColorCoeffs(matrixSel, matrixSelOut, rangeTV, (int)m_maxval);

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

    m_inBuf.resize((size_t)m_inC * m_modelInW * m_modelInH);
    m_outBuf.resize(m_ov->outElemCount());
    m_cudaPathTried = false; m_cudaPath = false;
    m_inputDevice = std::make_unique<CUMemBuf>(m_inBuf.size() * sizeof(float));
    m_outputDevice = std::make_unique<CUMemBuf>(m_outBuf.size() * sizeof(float));
    if (m_inputDevice->alloc() != RGY_ERR_NONE || m_outputDevice->alloc() != RGY_ERR_NONE) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (m_temporalT == 1 && m_io == OnnxIO::LumaSR && m_scale != 1) {
        auto resizeParam = std::make_shared<NVEncFilterParamResize>();
        resizeParam->interp = RGY_VPP_RESIZE_BILINEAR;
        resizeParam->frameIn = prm->frameIn;
        resizeParam->frameOut = frameOut;
        resizeParam->baseFps = prm->baseFps;
        resizeParam->bOutOverwrite = false;
        m_lumaResize = std::make_unique<NVEncFilterResize>();
        err = m_lumaResize->init(resizeParam, m_pLog);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: 輝度モデル用クロマ拡大の初期化に失敗しました: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    if (m_temporalT == 1 && (m_io == OnnxIO::Chroma || (m_io == OnnxIO::RGB && m_ycbcr))) {
        if (m_modelInW != inW || m_modelInH != inH) {
            m_yuvInputDevice = std::make_unique<CUMemBuf>((size_t)3 * inW * inH * sizeof(float));
            if (m_yuvInputDevice->alloc() != RGY_ERR_NONE) return RGY_ERR_MEMORY_ALLOC;
        }
        auto inputPtr = m_yuvInputDevice ? m_yuvInputDevice->ptr : m_inputDevice->ptr;
        const auto yuvIn = tensorFrame((float *)inputPtr, 3, inW, inH, RGY_CSP_YUV444_F32);
        auto yuvOut = tensorFrame((float *)m_outputDevice->ptr, 3, outW, outH, RGY_CSP_YUV444_F32);
        if (m_io == OnnxIO::Chroma) {
            const size_t planeBytes = (size_t)inW * inH * sizeof(float);
            yuvOut.ptr[0] = (uint8_t *)inputPtr;
            yuvOut.ptr[1] = (uint8_t *)m_outputDevice->ptr;
            yuvOut.ptr[2] = (uint8_t *)m_outputDevice->ptr + planeBytes;
        }
        auto toYuvParam = std::make_shared<NVEncFilterParamCrop>();
        toYuvParam->frameIn = prm->frameIn;
        toYuvParam->frameIn.picstruct = RGY_PICSTRUCT_FRAME;
        toYuvParam->frameOut = yuvIn;
        toYuvParam->baseFps = prm->baseFps;
        toYuvParam->matrix = RGY_MATRIX_BT709;
        toYuvParam->colorrange = RGY_COLORRANGE_FULL;
        toYuvParam->bOutOverwrite = false;
        m_cropToYuv444 = std::make_unique<NVEncFilterCspCrop>();
        err = m_cropToYuv444->init(toYuvParam, m_pLog);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: YUV444 FP32への変換初期化に失敗しました: %s.\n"), get_err_mes(err));
            return err;
        }
        auto fromYuvParam = std::make_shared<NVEncFilterParamCrop>();
        fromYuvParam->frameIn = yuvOut;
        fromYuvParam->frameOut = frameOut;
        fromYuvParam->baseFps = prm->baseFps;
        fromYuvParam->matrix = RGY_MATRIX_BT709;
        fromYuvParam->colorrange = RGY_COLORRANGE_FULL;
        fromYuvParam->bOutOverwrite = false;
        m_cropFromYuv444 = std::make_unique<NVEncFilterCspCrop>();
        err = m_cropFromYuv444->init(fromYuvParam, m_pLog);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: YUV444 FP32からの変換初期化に失敗しました: %s.\n"), get_err_mes(err));
            return err;
        }
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
            // 後段リサイズはCUDAサブフィルタのため、CUDA標準リサイズだけを許可する。
            // 外部ライブラリ系はこの経路で必要なパラメータが設定されないため明示的に拒否する。
            if (prm->onnx.postResizeAlgo != RGY_VPP_RESIZE_AUTO
             && getVppResizeType(prm->onnx.postResizeAlgo) != RGY_VPP_RESIZE_TYPE_OPENCL) {
                AddMessage(RGY_LOG_ERROR,
                    _T("resize=%s cannot be used with out_res=: only the CUDA resize algorithms are available here.\n"),
                    get_cx_desc(list_vpp_resize, prm->onnx.postResizeAlgo));
                return RGY_ERR_UNSUPPORTED;
            }
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

RGYFrameInfo NVEncFilterOnnx::tensorFrame(float *ptr, int channels, int width, int height, RGY_CSP csp) const {
    RGYFrameInfo frame;
    frame.width = width;
    frame.height = height;
    frame.csp = csp;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    const size_t planeBytes = (size_t)width * height * sizeof(float);
    const int planeCount = std::min(channels, 3);
    for (int i = 0; i < planeCount; i++) {
        frame.ptr[i] = (uint8_t *)ptr + planeBytes * i;
        frame.pitch[i] = width * sizeof(float);
    }
    return frame;
}

RGY_ERR NVEncFilterOnnx::reinitHostPath(tstring &errorMessage) {
    m_ov.reset();
    auto hostSession = std::make_unique<RGYOnnxRTCUDA>();
    auto err = hostSession->init(m_modelPath, m_deviceID, m_provider,
        m_modelInH, m_modelInW, errorMessage,
        nullptr, m_precision, m_cacheDir);
    if (err == RGY_ERR_NONE) {
        m_ov = std::move(hostSession);
    } else if (errorMessage.empty()) {
        errorMessage = hostSession->lastError();
    }
    return err;
}

RGY_ERR NVEncFilterOnnx::initCudaPath(cudaStream_t stream) {
    if (m_cudaPathTried) return m_cudaPath ? RGY_ERR_NONE : RGY_ERR_UNSUPPORTED;
    m_cudaPathTried = true;
    if (stream == nullptr || !m_inputDevice || !m_outputDevice) return RGY_ERR_UNSUPPORTED;

    // 初期化時のホスト経路用セッションを保持したまま同じモデルをウォームアップすると、
    // 重いモデルでは一時的に2セッション分のVRAMが必要になるため、先に解放する。
    m_ov.reset();
    auto session = std::make_unique<RGYOnnxRTCUDA>();
    tstring errorMessage;
    auto err = session->init(m_modelPath, m_deviceID, m_provider,
        m_modelInH, m_modelInW, errorMessage,
        stream, m_precision, m_cacheDir);
    if (err != RGY_ERR_NONE || !session->deviceIOAvailable()
        || session->inChannels() != m_inC || session->outChannels() != m_outC
        || session->outWidth() * session->outHeight() * session->outChannels() != (int)m_outBuf.size()) {
        const auto reason = !errorMessage.empty() ? errorMessage : session->lastError();
        AddMessage(RGY_LOG_WARN, _T("onnx: CUDAゼロコピー経路を初期化できないためホスト経路を使用します: %s\n"), reason.c_str());
        const auto cudaPathErr = (err != RGY_ERR_NONE) ? err : RGY_ERR_UNSUPPORTED;

        // 失敗したゼロコピー用セッションを破棄してから、ホスト経路用を再生成する。
        // 両方を同時に保持しないことで、フォールバック時にもVRAMのピークを抑える。
        session.reset();
        tstring hostErrorMessage;
        const auto hostErr = reinitHostPath(hostErrorMessage);
        if (hostErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: ホスト経路用セッションの再初期化に失敗しました: %s\n"), hostErrorMessage.c_str());
            return hostErr;
        }
        return cudaPathErr;
    }
    m_ov = std::move(session);
    m_cudaPath = true;
    AddMessage(RGY_LOG_INFO, _T("onnx: path cuda-zerocopy をフィルタストリーム上で初期化しました。\n"));
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::prepareLumaOutput(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream) {
    RGY_ERR err = RGY_ERR_NONE;
    if (m_lumaResize) {
        auto inputFrame = *input;
        RGYFrameInfo *resizeOutput[1] = { output };
        int outputCount = 0;
        err = m_lumaResize->filter(&inputFrame, resizeOutput, &outputCount, stream);
    } else {
        err = copyFrameAsync(output, input, stream);
    }
    return err;
}

OnnxColorCoeffs NVEncFilterOnnx::colorCoeffs() const {
    return { m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange,
        m_matVR, m_matUG, m_matVG, m_matUB, m_matRY, m_matGY, m_matBY,
        m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV };
}

RGY_ERR NVEncFilterOnnx::inferCuda(cudaStream_t stream) {
    if (!m_cudaPathTried) {
        auto err = initCudaPath(stream);
        if (err != RGY_ERR_NONE && !m_ov) return err;
    }
    if (m_cudaPath) {
        RGY_ERR err;
        if (m_maskMode) {
            std::vector<const float *> inputs(2);
            inputs[m_imgPortIdx] = (const float *)m_inputDevice->ptr;
            inputs[m_mskPortIdx] = (const float *)m_maskModelDevice->ptr;
            err = m_ov->inferMultiDevice(inputs, { (float *)m_outputDevice->ptr });
        } else {
            err = m_ov->inferDevice((const float *)m_inputDevice->ptr, (float *)m_outputDevice->ptr);
        }
        if (err == RGY_ERR_NONE) return err;
        AddMessage(RGY_LOG_WARN, _T("onnx: CUDA共有推論に失敗したためテンソル転送経路へ切り替えます: %s\n"), m_ov->lastError().c_str());
        // 推論途中のストリームを完了してから共有セッションを解放する。
        err = err_to_rgy(cudaStreamSynchronize(stream));
        if (err != RGY_ERR_NONE) return err;
        m_cudaPath = false;
        tstring message;
        err = reinitHostPath(message);
        if (err != RGY_ERR_NONE) return err;
    }
    // 推論I/Oだけを転送し、画素のパックと復元はCUDAの同じ処理を使う。
    auto err = err_to_rgy(cudaMemcpyAsync(m_inBuf.data(), m_inputDevice->ptr,
        m_inBuf.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    if (err != RGY_ERR_NONE) return err;
    err = err_to_rgy(cudaStreamSynchronize(stream));
    if (err != RGY_ERR_NONE) return err;
    if (m_maskMode) {
        std::vector<const float *> inputs(2);
        inputs[m_imgPortIdx] = m_inBuf.data(); inputs[m_mskPortIdx] = m_maskModel.data();
        err = m_ov->inferMulti(inputs, { m_outBuf.data() });
    } else {
        err = m_ov->infer(m_inBuf.data(), m_outBuf.data());
    }
    if (err != RGY_ERR_NONE) return err;
    return err_to_rgy(cudaMemcpyAsync(m_outputDevice->ptr, m_outBuf.data(),
        m_outBuf.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}

RGY_ERR NVEncFilterOnnx::runCudaLuma(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream) {
    auto err = prepareLumaOutput(input, output, stream);
    if (err != RGY_ERR_NONE) return err;
    err = run_onnx_pack_luma_padded((float *)m_inputDevice->ptr, input,
        m_modelInW, m_modelInH, m_padL, m_padT, stream);
    if (err != RGY_ERR_NONE) return err;
    err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    return run_onnx_unpack_luma(output, (const float *)m_outputDevice->ptr, stream);
}

RGY_ERR NVEncFilterOnnx::runCudaRGB(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream) {
    auto err = run_onnx_pack((float *)m_inputDevice->ptr, input,
        m_modelInW, m_modelInH, m_padL, m_padT, false, colorCoeffs(), stream);
    if (err != RGY_ERR_NONE) return err;
    if (m_io == OnnxIO::RGBNoise) {
        const size_t plane = (size_t)m_modelInW * m_modelInH;
        err = run_onnx_fill((float *)m_inputDevice->ptr + 3 * plane, plane, m_sigmaNorm, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    return run_onnx_unpack(output, (const float *)m_outputDevice->ptr, false, colorCoeffs(), stream);
}

RGY_ERR NVEncFilterOnnx::runCudaGrayNoise(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream) {
    auto err = copyFrameAsync(output, input, stream);
    if (err != RGY_ERR_NONE) return err;
    err = run_onnx_pack_luma_padded((float *)m_inputDevice->ptr, input,
        m_modelInW, m_modelInH, m_padL, m_padT, stream);
    if (err != RGY_ERR_NONE) return err;
    const size_t plane = (size_t)m_modelInW * m_modelInH;
    err = run_onnx_fill((float *)m_inputDevice->ptr + plane, plane, m_sigmaNorm, stream);
    if (err != RGY_ERR_NONE) return err;
    err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    return run_onnx_unpack_luma(output, (const float *)m_outputDevice->ptr, stream);
}

RGY_ERR NVEncFilterOnnx::runCudaYuv444(const RGYFrameInfo *input, RGYFrameInfo *output, cudaStream_t stream) {
    auto inputFrame = *input;
    inputFrame.picstruct = RGY_PICSTRUCT_FRAME;
    auto inputPtr = m_yuvInputDevice ? m_yuvInputDevice->ptr : m_inputDevice->ptr;
    auto yuvInput = tensorFrame((float *)inputPtr, 3, input->width, input->height, RGY_CSP_YUV444_F32);
    RGYFrameInfo *yuvInputOut[1] = { &yuvInput };
    int outputCount = 0;
    auto err = m_cropToYuv444->filter(&inputFrame, yuvInputOut, &outputCount, stream);
    if (err != RGY_ERR_NONE) return err;
    if (m_yuvInputDevice) {
        err = run_onnx_pad_planes((float *)m_inputDevice->ptr, (const float *)inputPtr,
            input->width, input->height, m_modelInW, m_modelInH, m_padL, m_padT, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    auto yuvOutput = tensorFrame((float *)m_outputDevice->ptr, 3, output->width, output->height, RGY_CSP_YUV444_F32);
    if (m_io == OnnxIO::Chroma) {
        const size_t planeBytes = (size_t)input->width * input->height * sizeof(float);
        yuvOutput.ptr[0] = (uint8_t *)inputPtr;
        yuvOutput.ptr[1] = (uint8_t *)m_outputDevice->ptr;
        yuvOutput.ptr[2] = (uint8_t *)m_outputDevice->ptr + planeBytes;
    }
    RGYFrameInfo *outputFrames[1] = { output };
    outputCount = 0;
    return m_cropFromYuv444->filter(&yuvOutput, outputFrames, &outputCount, stream);
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

    RGY_ERR cerr;
    if (m_io == OnnxIO::LumaSR) cerr = runCudaLuma(pInputFrame, coreFrame, stream);
    else if (m_io == OnnxIO::GrayNoise) cerr = runCudaGrayNoise(pInputFrame, coreFrame, stream);
    else if (m_io == OnnxIO::Chroma || m_ycbcr) cerr = runCudaYuv444(pInputFrame, coreFrame, stream);
    else cerr = runCudaRGB(pInputFrame, coreFrame, stream);
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

RGY_ERR NVEncFilterOnnx::runTemporal(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    const int k = (m_temporalT - 1) / 2;
    const size_t frameSize = (size_t)m_ov->inWidth() * m_ov->inHeight();
    if (pInputFrame->ptr[0] != nullptr) {
        RingFrame rf;
        rf.rgb = std::make_unique<CUMemBuf>(3 * frameSize * sizeof(float));
        if (rf.rgb->alloc() != RGY_ERR_NONE) return RGY_ERR_MEMORY_ALLOC;
        auto err = run_onnx_pack((float *)rf.rgb->ptr, pInputFrame,
            m_modelInW, m_modelInH, m_padL, m_padT, m_ycbcr, colorCoeffs(), stream);
        if (err != RGY_ERR_NONE) return err;
        // 出力待ちの場合も、入力が再利用される前に読み込み完了を保証する。
        err = err_to_rgy(cudaStreamSynchronize(stream));
        if (err != RGY_ERR_NONE) return err;
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
    m_modelInW = m_maskModelW; m_modelInH = m_maskModelH; m_padL = m_padT = 0;
    m_maxval = (float)((1 << RGY_CSP_BIT_DEPTH[inCsp]) - 1);
    m_inBuf.resize((size_t)3 * m_maskModelW * m_maskModelH);
    m_outBuf.resize(m_inBuf.size());
    int matrixSel = 0, matrixSelOut = 0;
    onnx_matrix_to_coeff_id(prm->onnx.colormatrix, inH, matrixSel);
    if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO) matrixSelOut = matrixSel;
    else onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, inH, matrixSelOut);
    setupColorCoeffs(matrixSel, matrixSelOut, prm->onnx.colorrange != RGY_COLORRANGE_FULL, (int)m_maxval);

    auto frameOut = prm->frameOut;
    frameOut.csp = inCsp; frameOut.width = inW; frameOut.height = inH;
    prm->frameOut = frameOut;
    err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) return err;
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    const size_t frameBytes = (size_t)inW * inH * sizeof(float);
    const size_t modelBytes = (size_t)m_maskModelW * m_maskModelH * sizeof(float);
    m_inputDevice = std::make_unique<CUMemBuf>(3 * modelBytes);
    m_outputDevice = std::make_unique<CUMemBuf>(3 * modelBytes);
    m_maskFrameDevice = std::make_unique<CUMemBuf>(frameBytes);
    m_maskModelDevice = std::make_unique<CUMemBuf>(modelBytes);
    m_frameRGBDevice = std::make_unique<CUMemBuf>(3 * frameBytes);
    m_compositeDevice = std::make_unique<CUMemBuf>(3 * frameBytes);
    m_maskRangeDevice = std::make_unique<CUMemBuf>(sizeof(int));
    for (auto buffer : { m_inputDevice.get(), m_outputDevice.get(), m_maskFrameDevice.get(),
        m_maskModelDevice.get(), m_frameRGBDevice.get(), m_compositeDevice.get(), m_maskRangeDevice.get() }) {
        if (buffer->alloc() != RGY_ERR_NONE) return RGY_ERR_MEMORY_ALLOC;
    }
    err = err_to_rgy(cudaMemcpy(m_maskFrameDevice->ptr, m_maskFrame.data(), frameBytes, cudaMemcpyHostToDevice));
    if (err != RGY_ERR_NONE) return err;
    err = err_to_rgy(cudaMemcpy(m_maskModelDevice->ptr, m_maskModel.data(), modelBytes, cudaMemcpyHostToDevice));
    if (err != RGY_ERR_NONE) return err;
    m_cudaPathTried = m_cudaPath = false;
    setFilterInfo(strsprintf(_T("onnx: %s %dx%d io=rgb+mask (model %dx%d) backend=%s"),
        PathGetFilename(prm->onnx.modelFile).c_str(), inW, inH, m_maskModelW, m_maskModelH, m_ov->providerName().c_str()));
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterOnnx::runMask(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) { *pOutputFrameNum = 0; return RGY_ERR_NONE; }
    const int fw = pInputFrame->width, fh = pInputFrame->height;
    auto err = run_onnx_pack((float *)m_frameRGBDevice->ptr, pInputFrame, fw, fh, 0, 0, false, colorCoeffs(), stream);
    if (err != RGY_ERR_NONE) return err;
    err = run_onnx_mask_resize((const float *)m_frameRGBDevice->ptr, (float *)m_inputDevice->ptr,
        fw, fh, m_maskModelW, m_maskModelH, stream);
    if (err != RGY_ERR_NONE) return err;
    err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    if (m_maskOutScale == 0.0f) {
        // 初回の出力範囲判定をGPU上に保持して後続フレームでも使用する。
        err = run_onnx_mask_detect((const float *)m_outputDevice->ptr, 3 * m_maskModelW * m_maskModelH,
            (int *)m_maskRangeDevice->ptr, stream);
        if (err != RGY_ERR_NONE) return err;
        m_maskOutScale = 1.0f;
    }
    err = run_onnx_mask_composite((const float *)m_frameRGBDevice->ptr, (const float *)m_outputDevice->ptr,
        (const float *)m_maskFrameDevice->ptr, (const int *)m_maskRangeDevice->ptr, (float *)m_compositeDevice->ptr,
        fw, fh, m_maskModelW, m_maskModelH, stream);
    if (err != RGY_ERR_NONE) return err;
    auto coreFrame = &m_frameBuf[0]->frame;
    err = run_onnx_unpack(coreFrame, (const float *)m_compositeDevice->ptr, false, colorCoeffs(), stream);
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
        auto err = err_to_rgy(cudaMemcpyAsync((float *)m_inputDevice->ptr + (size_t)j * 3 * frameSize,
            m_ring[(size_t)r].rgb->ptr, 3 * frameSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (err != RGY_ERR_NONE) return err;
    }
    auto err = inferCuda(stream);
    if (err != RGY_ERR_NONE) return err;
    auto coreFrame = &m_frameBuf[0]->frame;
    err = run_onnx_unpack(coreFrame, (const float *)m_outputDevice->ptr, m_ycbcr, colorCoeffs(), stream);
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

void NVEncFilterOnnx::close() {
    m_postResize.reset();
    m_yuvInputDevice.reset();
    m_cropToYuv444.reset();
    m_cropFromYuv444.reset();
    m_lumaResize.reset();
    m_inputDevice.reset();
    m_outputDevice.reset();
    m_ov.reset();
    m_inBuf.clear();
    m_outBuf.clear();
    m_ring.clear();
    m_ringBaseIdx = 0;
    m_recvCount = 0;
    m_emitCount = 0;
    m_temporalT = 1;
    m_maskModelW = m_maskModelH = 0;
    m_maskOutScale = 0.0f;
    m_maskFrame.clear();
    m_maskModel.clear();
    m_maskFrameDevice.reset(); m_maskModelDevice.reset(); m_frameRGBDevice.reset();
    m_compositeDevice.reset(); m_maskRangeDevice.reset();
    m_maskMode = false;
    m_cudaPathTried = false;
    m_cudaPath = false;
    m_frameBuf.clear();
}
