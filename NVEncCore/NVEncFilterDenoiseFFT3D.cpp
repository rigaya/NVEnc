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
    m_windowBufInverse() {
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
        || prm->fft3d.precision != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
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

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, prm->fft3d.temporal ? 3 : 1)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
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

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (prm->fft3d.temporal) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseFFT3D::print() const {
    return fft3d.print();
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

    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (!prm->fft3d.temporal || m_nFrameIdx >= m_bufIdx) {
            //終了
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
    } else {
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
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
        auto fftBuf = m_bufFFT.get(m_bufIdx++);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        sts = denosieFunc->fft()(&fftBuf->frame, pInputFrame, m_ov1, m_ov2, (const float *)m_windowBuf->ptr, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    if (prm->fft3d.temporal) {
        if (m_bufIdx <= 1) {
            //出力フレームなし
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
        auto fftPrev = m_bufFFT.get(std::max(m_bufIdx - ((finalOutput) ? 2 : 3), 0));
        auto fftCur  = m_bufFFT.get(m_bufIdx - ((finalOutput) ? 1 : 2));
        auto fftNext = m_bufFFT.get(m_bufIdx - 1);
        sts = denosieFunc->tfft_filter_ifft(1, 3)(&m_filteredBlocks->frame, &fftPrev->frame, &fftCur->frame, &fftNext->frame, nullptr, (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(1, 3): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
    } else {
        auto fftCur = m_bufFFT.get(m_bufIdx - 1);
        sts = denosieFunc->tfft_filter_ifft(0, 1)(&m_filteredBlocks->frame, &fftCur->frame, nullptr, nullptr, nullptr, (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(0, 1): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
    }
    sts = denosieFunc->merge()(ppOutputFrames[0], &m_filteredBlocks->frame, m_ov1, m_ov2, stream);
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
    m_windowBuf.reset();
    m_windowBufInverse.reset();
}
