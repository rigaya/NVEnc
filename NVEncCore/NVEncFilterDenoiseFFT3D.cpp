// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <array>
#include <map>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "rgy_prm.h"

std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc8FP32(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc8FP16(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc16FP16(const int block_size);
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc16FP32(const int block_size);

std::unique_ptr<DenoiseFFT3DBase> getDenoiseFunc(const RGY_CSP csp, const int block_size, VppFpPrecision prec) {
    switch (RGY_CSP_DATA_TYPE[csp]) {
    case RGY_DATA_TYPE_U8:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            return getDenoiseFFT3DFunc8FP32(block_size);
        } else {
            return getDenoiseFFT3DFunc8FP16(block_size);
        }
    case RGY_DATA_TYPE_U16:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            return getDenoiseFFT3DFunc16FP32(block_size);
        } else {
            return getDenoiseFFT3DFunc16FP16(block_size);
        }
    default:
        return nullptr;
    }
}

// Resolve the effective temporal radius bt. bt overrides 'temporal' when set
// (!= 0); otherwise fall back to the legacy temporal flag (0 -> bt1 spatial,
// 1 -> bt3 prev+cur+next). bt=-1 = sharpen/degrid only (no denoising), which
// processes a single frame like bt=1. Frame layout per bt:
//   bt1 [cur]                 nPast=0 nFuture=0 curIdx=0
//   bt2 [prev,cur]            nPast=1 nFuture=0 curIdx=1
//   bt3 [prev,cur,next]       nPast=1 nFuture=1 curIdx=1
//   bt4 [prev2,prev,cur,next] nPast=2 nFuture=1 curIdx=2
static int fft3d_bt(const VppDenoiseFFT3D &f) {
    const int bt = (f.bt != 0) ? f.bt : (f.temporal ? 3 : 1);
    return (bt < -1) ? -1 : (bt > 4) ? 4 : bt;
}

// number of frames processed together (bt=-1 -> 1)
static int fft3d_bt_frames(const VppDenoiseFFT3D &f) {
    return std::max(fft3d_bt(f), 1);
}

RGY_ERR NVEncFilterDenoiseFFT3DBuffer::alloc(int width, int height, RGY_CSP csp, int frames) {
    m_bufFFT.resize(frames);
    for (auto& buf : m_bufFFT) {
        if (!buf || buf->frame.width != width || buf->frame.height != height || buf->frame.csp != csp) {
            buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
            auto sts = buf->alloc(width, height, csp);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoiseFFT3D::NVEncFilterDenoiseFFT3D() :
    m_bufIdx(0),
    m_ov1(0),
    m_ov2(0),
    m_bufFFT(),
    m_filteredBlocks(),
    m_windowBuf(),
    m_windowBufInverse(),
    m_sigmaBuf(),
    m_wsharpenBuf(),
    m_gridBuf(),
    m_gridDC(0.0f),
    m_noisePowerGain(1.0f) {
    m_name = _T("denoise-fft");
}

NVEncFilterDenoiseFFT3D::~NVEncFilterDenoiseFFT3D() {
    close();
}

RGY_ERR NVEncFilterDenoiseFFT3D::checkParam(const NVEncFilterParamDenoiseFFT3D *prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sigma < 0.0f || 100.0f < prm->fft3d.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma must be 0 - 100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (   prm->fft3d.sigma2 < 0.0f || 100.0f < prm->fft3d.sigma2
        || prm->fft3d.sigma3 < 0.0f || 100.0f < prm->fft3d.sigma3
        || prm->fft3d.sigma4 < 0.0f || 100.0f < prm->fft3d.sigma4) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma2/sigma3/sigma4 must be 0 - 100 (0 = follow sigma).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.amount < 0.0f || 1.0f < prm->fft3d.amount) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, strength must be 0 - 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fft3d_block_size, prm->fft3d.block_size) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap < 0.0f || 0.8f < prm->fft3d.overlap) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap2 < 0.0f || 0.8f < prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap2 must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (0.8f < prm->fft3d.overlap + prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sum of overlap and overlap2 must be below 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.method != 0 && prm->fft3d.method != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, method must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.temporal != 0 && prm->fft3d.temporal != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, temporal must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.bt < -1 || prm->fft3d.bt > 4) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, bt must be 0 (follow temporal), 1 - 4, or -1 (sharpen/degrid only).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.bt == -1 && prm->fft3d.sharpen == 0.0f) {
        // degrid alone with bt=-1 would be an exact no-op (subtract + add back with
        // no filtering in between), so sharpen is required.
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, bt=-1 requires sharpen.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sharpen < -10.0f || 10.0f < prm->fft3d.sharpen) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sharpen must be -10 - 10.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.scutoff <= 0.0f || 1.0f < prm->fft3d.scutoff) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, scutoff must be greater than 0, up to 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.svr < 0.0f || 10.0f < prm->fft3d.svr) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, svr must be 0 - 10.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.smin < 0.0f || prm->fft3d.smax <= 0.0f || prm->fft3d.smax < prm->fft3d.smin) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, smin/smax must satisfy 0 <= smin <= smax, smax > 0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.degrid < 0.0f || 2.0f < prm->fft3d.degrid) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, degrid must be 0 - 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fp_prec, prm->fft3d.precision) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid precision.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseFFT3D::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 && prm->compute_capability.first < 7) {
        prm->fft3d.precision = VppFpPrecision::VPP_FP_PRECISION_FP32;
    }
    if (!m_param
        || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size
        || prm->fft3d.overlap != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap
        || prm->fft3d.overlap2 != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap2
        || prm->fft3d.temporal != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.temporal
        || prm->fft3d.bt != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.bt
        || prm->fft3d.precision != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
        || prm->processChroma != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->processChroma
        || cmpFrameInfoCspResolution(&m_param->frameOut, &prm->frameOut)) {
        m_ov1 = (int)(prm->fft3d.block_size * 0.5 * prm->fft3d.overlap + 0.5);
        m_ov2 = (int)(prm->fft3d.block_size * 0.5 * (prm->fft3d.overlap + prm->fft3d.overlap2) + 0.5) - m_ov1;

        //より小さいUVに合わせてブロック数を計算し、そこから確保するメモリを決める
        auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);
        const auto blocksUV = getBlockCount(planeUV.width, planeUV.height, prm->fft3d.block_size, m_ov1, m_ov2);
        const int complexSize = (prm->fft3d.precision == VppFpPrecision::VPP_FP_PRECISION_FP32) ? 8 : 4;

        RGY_CSP fft_csp = RGY_CSP_NA;
        int blockGlobalWidth = 0, blockGlobalHeight = 0;
        if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) {
            fft_csp = RGY_CSP_YV12;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size * 2;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size * 2;
        } else if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV444) {
            fft_csp = RGY_CSP_YUV444;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
            return RGY_ERR_UNSUPPORTED;
        }

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, fft3d_bt_frames(prm->fft3d))) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
        }
        if (!prm->processChroma) {
            if ((sts = m_srcBuf.alloc(prm->frameOut.width, prm->frameOut.height, prm->frameOut.csp, fft3d_bt_frames(prm->fft3d))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for luma-only FFT3D source frames: %s.\n"), get_err_mes(sts));
                return sts;
            }
        } else {
            m_srcBuf.clear();
        }

        m_filteredBlocks = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
        if ((sts = m_filteredBlocks->alloc(blockGlobalWidth, blockGlobalHeight, prm->frameOut.csp)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for filtered blocks: %s.\n"), get_err_mes(sts));
            return sts;
        }

        sts = AllocFrameBuf(prm->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }

        if (!m_param || !m_windowBuf || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size) {
            std::vector<float> blockWindow(prm->fft3d.block_size);
            std::vector<float> blockWindowInv(prm->fft3d.block_size);
            auto winFunc = [block_size = prm->fft3d.block_size](const int x) { return 0.50f - 0.50f * std::cos(2.0f * (float)M_PI * x / (float)block_size); };
            for (int i = 0; i < prm->fft3d.block_size; i++) {
                blockWindow[i] = winFunc(i);
                blockWindowInv[i] = 1.0f / blockWindow[i];
            }

            m_windowBuf = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindow.size() * sizeof(blockWindow[0])));
            m_windowBufInverse = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindowInv.size() * sizeof(blockWindowInv[0])));

            if ((sts = m_windowBuf->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_windowBufInverse->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBuf->ptr, blockWindow.data(), blockWindow.size() * sizeof(blockWindow[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBufInverse->ptr, blockWindowInv.data(), blockWindowInv.size() * sizeof(blockWindowInv[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    // Per-frequency-bin host tables, rebuilt every init() (cheap: block_size^2
    // floats) so they always reflect the current parameters even when the
    // (gated) buffer reallocation above is skipped. Clean-room from the
    // documented FFT3DFilter parameter semantics.
    {
        const int bs = prm->fft3d.block_size;
        auto uploadTable = [&](std::unique_ptr<CUMemBuf>& buf, const std::vector<float>& table, const TCHAR *name) {
            buf = std::unique_ptr<CUMemBuf>(new CUMemBuf(table.size() * sizeof(table[0])));
            auto err = buf->alloc();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT3D %s table: %s.\n"), name, get_err_mes(err));
                return err;
            }
            if ((err = err_to_rgy(cudaMemcpy(buf->ptr, table.data(), table.size() * sizeof(table[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT3D %s table: %s.\n"), name, get_err_mes(err));
                return err;
            }
            return RGY_ERR_NONE;
        };
        // frequency coordinate of bin i, normalised to [0,1] (0 = DC, 1 = Nyquist),
        // with the negative frequencies at the upper indices mirrored down.
        auto fnorm = [bs](int i) { const int f = (i < bs - i) ? i : (bs - i); return (float)f / (float)(bs / 2); };

        // analysis window (same function the fft kernel uses) and its power sum.
        // For white noise of std s (normalised pixels), each windowed FFT bin has
        // expected power s^2 * sum(w(x)^2) * sum(w(y)^2) - the reference for
        // signorm, which lets the user give sigma as the actual noise level.
        std::vector<float> win(bs);
        auto winFunc = [bs](const int x) { return 0.50f - 0.50f * std::cos(2.0f * (float)M_PI * x / (float)bs); };
        double sw2 = 0.0;
        for (int i = 0; i < bs; i++) {
            win[i] = winFunc(i);
            sw2 += win[i] * win[i];
        }
        m_noisePowerGain = (float)(sw2 * sw2); // 2D separable window power gain

        // (1) sigma table: the 4 documented anchors (sigma = highest .. sigma4 =
        // lowest frequency) interpolated across the normalised radial frequency.
        // The same /255 scaling as the previous scalar path is baked in, so when
        // sigma2/3/4 = sigma (or left unset) every entry equals the old scalar
        // value and the output is reproduced exactly.
        {
            const float s1 = prm->fft3d.sigma;                                     // highest freq
            const float s2 = (prm->fft3d.sigma2 > 0.0f) ? prm->fft3d.sigma2 : s1;  // mid-high
            const float s3 = (prm->fft3d.sigma3 > 0.0f) ? prm->fft3d.sigma3 : s1;  // mid-low
            const float s4 = (prm->fft3d.sigma4 > 0.0f) ? prm->fft3d.sigma4 : s1;  // lowest
            const float anchors[4] = { s4, s3, s2, s1 }; // radial 0 -> 1
            std::vector<float> sigmaTable((size_t)bs * bs);
            for (int by = 0; by < bs; by++) {
                const float fy = fnorm(by);
                for (int bx = 0; bx < bs; bx++) {
                    const float fx = fnorm(bx);
                    float radial = std::sqrt(fx * fx + fy * fy) * 0.70710678f; // /sqrt(2) -> [0,1]
                    if (radial > 1.0f) radial = 1.0f;
                    const float t = radial * 3.0f; // 3 linear segments across the 4 anchors
                    int seg = (int)t; if (seg > 2) seg = 2;
                    const float frac = t - (float)seg;
                    const float sval = anchors[seg] * (1.0f - frac) + anchors[seg + 1] * frac;
                    if (prm->fft3d.signorm) {
                        // sigma given as the noise level (8-bit scale): threshold at the
                        // per-bin noise power that this noise level actually produces.
                        // The forward temporal DFT is un-normalized, so iid per-frame
                        // noise power is multiplied by btFrames in the temporal bins
                        // (cf. the original FFT3DFilter's btcur factor) - scale to match.
                        // (smin/smax must NOT get this factor: the sharpen psd is taken
                        // after the 1/N-normalized inverse temporal DFT, per-frame units.)
                        const float snorm = sval * (1.0f / ((1 << 8) - 1));
                        sigmaTable[(size_t)by * bs + bx] = snorm * snorm * m_noisePowerGain * (float)fft3d_bt_frames(prm->fft3d);
                    } else {
                        // backward compatible scale (compared against unnormalised bin power)
                        sigmaTable[(size_t)by * bs + bx] = sval * (1.0f / ((1 << 8) - 1)); // match scalar /255
                    }
                }
            }
            if ((sts = uploadTable(m_sigmaBuf, sigmaTable, _T("sigma"))) != RGY_ERR_NONE) {
                return sts;
            }
        }

        // (2) sharpen weight table: strength x gaussian high-pass frequency
        // weight, 1 - exp(-f^2 / (2*scutoff^2)), with the vertical frequency
        // contribution scaled by svr (svr = 0 -> no vertical sharpening).
        if (prm->fft3d.sharpen != 0.0f) {
            const float scutoff = std::max(prm->fft3d.scutoff, 0.01f);
            std::vector<float> wsharpenTable((size_t)bs * bs);
            for (int by = 0; by < bs; by++) {
                const float fy = fnorm(by) * prm->fft3d.svr;
                for (int bx = 0; bx < bs; bx++) {
                    const float fx = fnorm(bx);
                    const float f2 = fx * fx + fy * fy;
                    const float weight = 1.0f - std::exp(-f2 / (2.0f * scutoff * scutoff));
                    wsharpenTable[(size_t)by * bs + bx] = prm->fft3d.sharpen * weight;
                }
            }
            if ((sts = uploadTable(m_wsharpenBuf, wsharpenTable, _T("sharpen"))) != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            m_wsharpenBuf.reset();
        }

        // (3) gridsample spectrum for degrid: the 2D spectrum of the analysis
        // window itself (the spectrum a flat, featureless block produces). It is
        // separable, so it is built from the 1D DFT of the window function.
        // The kernel scales it by each block's DC / gridDC to reconstruct and
        // subtract the window bias before filtering.
        if (prm->fft3d.degrid > 0.0f) {
            std::vector<std::pair<float, float>> w1(bs); // 1D DFT of the window
            for (int k = 0; k < bs; k++) {
                double re = 0.0, im = 0.0;
                for (int x = 0; x < bs; x++) {
                    const double theta = -2.0 * M_PI * k * x / (double)bs;
                    re += win[x] * std::cos(theta);
                    im += win[x] * std::sin(theta);
                }
                w1[k] = { (float)re, (float)im };
            }
            std::vector<float> gridTable((size_t)bs * bs * 2);
            for (int by = 0; by < bs; by++) {
                for (int bx = 0; bx < bs; bx++) {
                    // complex product W1[by] * W1[bx]
                    const float re = w1[by].first * w1[bx].first - w1[by].second * w1[bx].second;
                    const float im = w1[by].first * w1[bx].second + w1[by].second * w1[bx].first;
                    gridTable[((size_t)by * bs + bx) * 2 + 0] = re;
                    gridTable[((size_t)by * bs + bx) * 2 + 1] = im;
                }
            }
            m_gridDC = w1[0].first * w1[0].first; // (sum of window)^2, DC of the 2D spectrum
            if ((sts = uploadTable(m_gridBuf, gridTable, _T("gridsample"))) != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            m_gridBuf.reset();
            m_gridDC = 0.0f;
        }
    }

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (fft3d_bt(prm->fft3d) > 1) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseFFT3D::print() const {
    return fft3d.print() + strsprintf(_T(", chroma %s"), processChroma ? _T("on") : _T("off"));
}

RGY_ERR NVEncFilterDenoiseFFT3D::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto denosieFunc = getDenoiseFunc(prm->frameIn.csp, prm->fft3d.block_size, prm->fft3d.precision);
    if (!denosieFunc) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp or block_size.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int bt = fft3d_bt(prm->fft3d);
    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (bt <= 1 || m_nFrameIdx >= m_bufIdx) {
            //終了
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
    } else {
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
        //}
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        const int curBufIdx = m_bufIdx++;
        auto fftBuf = m_bufFFT.get(curBufIdx);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (!prm->processChroma) {
            auto srcBuf = m_srcBuf.get(curBufIdx);
            if (!srcBuf || !srcBuf->frame.ptr[0]) {
                AddMessage(RGY_LOG_ERROR, _T("failed to get luma-only FFT3D source buffer.\n"));
                return RGY_ERR_NULL_PTR;
            }
            auto copyErr = copyFrameAsync(&srcBuf->frame, pInputFrame, stream);
            if (copyErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy luma-only FFT3D source frame: %s.\n"), get_err_mes(copyErr));
                return copyErr;
            }
            copyFramePropWithoutRes(&srcBuf->frame, pInputFrame);
        }
        sts = denosieFunc->fft()(&fftBuf->frame, pInputFrame, m_ov1, m_ov2, (const float *)m_windowBuf->ptr, prm->processChroma, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    const RGYFrameInfo *srcCurFrame = nullptr;
    {
        // bt frame layout (see fft3d_bt): [prev.. , cur, ..next], cur at index nPast.
        const int btFrames = std::max(bt, 1);      // bt=-1 processes a single frame
        const int nPast = btFrames / 2;
        const int nFuture = (btFrames - 1) - nPast; // bt1:0 bt2:0 bt3:1 bt4:1
        const int curIdx = nPast;                   // temporalCurrentIdx

        // Not enough future frames buffered yet to emit the next output frame.
        // (nFuture==0 for bt=1/2 -> emit immediately; nFuture==1 for bt=3/4 ->
        //  one frame of look-ahead, flushed at finalOutput.)
        if (!finalOutput && m_bufIdx < m_nFrameIdx + nFuture + 1) {
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }

        const int outFrameIdx = m_nFrameIdx; // frame to output now
        // Gather the btFrames frames [outFrameIdx-nPast .. outFrameIdx+nFuture],
        // repeating boundary frames at the start of stream and during flush by
        // clamping to the valid buffered range (reproduces the previous
        // prev=cur / next=cur edge handling for bt=3).
        CUFrameBuf *frames[4] = { nullptr, nullptr, nullptr, nullptr };
        for (int k = 0; k < btFrames; k++) {
            int idx = outFrameIdx + (k - nPast);
            if (idx < 0) idx = 0;
            if (idx > m_bufIdx - 1) idx = m_bufIdx - 1;
            frames[k] = m_bufFFT.get(idx);
        }
        auto fftCur = frames[curIdx];
        if (!prm->processChroma) {
            auto srcCur = m_srcBuf.get(outFrameIdx);
            srcCurFrame = srcCur ? &srcCur->frame : nullptr;
        }
        auto func = denosieFunc->tfft_filter_ifft(curIdx, btFrames);
        if (!func) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported fft3d bt=%d.\n"), bt);
            return RGY_ERR_UNSUPPORTED;
        }
        const float scale = (1.0f / ((1 << 8) - 1)); // same 8-bit-referenced normalisation as sigma
        const float nGain = (prm->fft3d.signorm) ? m_noisePowerGain : 1.0f; // signorm: real noise-power units
        const float sminSq = (prm->fft3d.smin * scale) * (prm->fft3d.smin * scale) * nGain;
        const float smaxSq = (prm->fft3d.smax * scale) * (prm->fft3d.smax * scale) * nGain;
        const float degridFactor = (m_gridBuf && m_gridDC > 0.0f) ? prm->fft3d.degrid / m_gridDC : 0.0f;
        sts = func(&m_filteredBlocks->frame,
            frames[0] ? &frames[0]->frame : nullptr,
            frames[1] ? &frames[1]->frame : nullptr,
            frames[2] ? &frames[2]->frame : nullptr,
            frames[3] ? &frames[3]->frame : nullptr,
            (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            (const float *)m_sigmaBuf->ptr, 1.0f - prm->fft3d.amount, (bt < 0) ? -1 : prm->fft3d.method,
            (m_wsharpenBuf) ? (const float *)m_wsharpenBuf->ptr : nullptr, sminSq, smaxSq,
            (m_gridBuf) ? (const float *)m_gridBuf->ptr : nullptr, degridFactor,
            prm->processChroma, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(%d, %d): %s.\n"), curIdx, btFrames, get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        if (btFrames > 1) {
            copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
        }
    }
    if (!prm->processChroma) {
        if (!srcCurFrame || !srcCurFrame->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("missing luma-only FFT3D source frame.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto copyErr = copyFrameAsync(ppOutputFrames[0], srcCurFrame, stream);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy luma-only FFT3D output base frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
    }
    sts = denosieFunc->merge()(ppOutputFrames[0], &m_filteredBlocks->frame, m_ov1, m_ov2, prm->processChroma, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run merge: %s.\n"), get_err_mes(sts));
        return RGY_ERR_NONE;
    }

    m_nFrameIdx++;
    return sts;
}

void NVEncFilterDenoiseFFT3D::close() {
    m_frameBuf.clear();
    m_bufFFT.clear();
    m_srcBuf.clear();
    m_windowBuf.reset();
    m_windowBufInverse.reset();
    m_sigmaBuf.reset();
    m_wsharpenBuf.reset();
    m_gridBuf.reset();
}
