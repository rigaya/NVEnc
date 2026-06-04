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

#include <algorithm>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterDeblock.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DEBLOCK_BLOCK_X = 32;
static const int DEBLOCK_BLOCK_Y = 8;

// ITU-T Rec. H.264 (V14, 2022-08) Table 8-16 and Table 8-17.
static const int H264_ALPHA[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      4,   4,   5,   6,   7,   8,   9,  10,  12,  13,  15,  17,  20,  22,  25,  28,
     32,  36,  40,  45,  50,  56,  63,  71,  80,  90, 101, 113, 127, 144, 162, 182,
    203, 226, 255, 255
};

static const int H264_BETA[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      2,   2,   2,   3,   3,   3,   3,   4,   4,   4,   6,   6,   7,   7,   8,   8,
      9,   9,  10,  10,  11,  11,  12,  12,  13,  13,  14,  14,  15,  15,  16,  16,
     17,  17,  18,  18
};

static const int H264_TC0_BS1[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,
      2,   2,   3,   3
};

__device__ __forceinline__ int deblock_clip3(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template<int max_val>
__device__ __forceinline__ int deblock_clip_pixel(int v) {
    return (v < 0) ? 0 : ((v > max_val) ? max_val : v);
}

template<typename Type, int max_val>
__global__ void kernel_deblock_vertical(uint8_t *__restrict__ pBuf, const int bufPitch,
    const int width, const int height, const int alpha, const int beta, const int tc0, const int is_chroma) {
    const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= height) return;

    const int boundary_x = (edge_index + 1) * 4;
    if (boundary_x < 3 || boundary_x > width - 3) return;

    Type *row = (Type *)(pBuf + iy * bufPitch);

    const int p2 = (int)row[boundary_x - 3];
    const int p1 = (int)row[boundary_x - 2];
    const int p0 = (int)row[boundary_x - 1];
    const int q0 = (int)row[boundary_x    ];
    const int q1 = (int)row[boundary_x + 1];
    const int q2 = (int)row[boundary_x + 2];

    const int abs_p0q0 = abs(p0 - q0);
    const int abs_p1p0 = abs(p1 - p0);
    const int abs_q1q0 = abs(q1 - q0);
    if (abs_p0q0 >= alpha || abs_p1p0 >= beta || abs_q1q0 >= beta) {
        return;
    }

    const int ap = abs(p2 - p0);
    const int aq = abs(q2 - q0);
    const int tc = (is_chroma != 0) ? (tc0 + 1) : (tc0 + ((ap < beta) ? 1 : 0) + ((aq < beta) ? 1 : 0));

    const int delta_raw = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
    const int delta = deblock_clip3(delta_raw, -tc, tc);
    row[boundary_x - 1] = (Type)deblock_clip_pixel<max_val>(p0 + delta);
    row[boundary_x    ] = (Type)deblock_clip_pixel<max_val>(q0 - delta);

    if (is_chroma == 0) {
        const int pq_avg = (p0 + q0 + 1) >> 1;
        if (ap < beta) {
            const int p1_delta = deblock_clip3((p2 + pq_avg - (p1 << 1)) >> 1, -tc0, tc0);
            row[boundary_x - 2] = (Type)deblock_clip_pixel<max_val>(p1 + p1_delta);
        }
        if (aq < beta) {
            const int q1_delta = deblock_clip3((q2 + pq_avg - (q1 << 1)) >> 1, -tc0, tc0);
            row[boundary_x + 1] = (Type)deblock_clip_pixel<max_val>(q1 + q1_delta);
        }
    }
}

template<typename Type, int max_val>
__global__ void kernel_deblock_horizontal(uint8_t *__restrict__ pBuf, const int bufPitch,
    const int width, const int height, const int alpha, const int beta, const int tc0, const int is_chroma) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int edge_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width) return;

    const int boundary_y = (edge_index + 1) * 4;
    if (boundary_y < 3 || boundary_y > height - 3) return;

#define PIX(y) (*(Type *)(pBuf + (y) * bufPitch + ix * sizeof(Type)))
    const int p2 = (int)PIX(boundary_y - 3);
    const int p1 = (int)PIX(boundary_y - 2);
    const int p0 = (int)PIX(boundary_y - 1);
    const int q0 = (int)PIX(boundary_y    );
    const int q1 = (int)PIX(boundary_y + 1);
    const int q2 = (int)PIX(boundary_y + 2);

    const int abs_p0q0 = abs(p0 - q0);
    const int abs_p1p0 = abs(p1 - p0);
    const int abs_q1q0 = abs(q1 - q0);
    if (abs_p0q0 >= alpha || abs_p1p0 >= beta || abs_q1q0 >= beta) {
        return;
    }

    const int ap = abs(p2 - p0);
    const int aq = abs(q2 - q0);
    const int tc = (is_chroma != 0) ? (tc0 + 1) : (tc0 + ((ap < beta) ? 1 : 0) + ((aq < beta) ? 1 : 0));

    const int delta_raw = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
    const int delta = deblock_clip3(delta_raw, -tc, tc);
    PIX(boundary_y - 1) = (Type)deblock_clip_pixel<max_val>(p0 + delta);
    PIX(boundary_y    ) = (Type)deblock_clip_pixel<max_val>(q0 - delta);

    if (is_chroma == 0) {
        const int pq_avg = (p0 + q0 + 1) >> 1;
        if (ap < beta) {
            const int p1_delta = deblock_clip3((p2 + pq_avg - (p1 << 1)) >> 1, -tc0, tc0);
            PIX(boundary_y - 2) = (Type)deblock_clip_pixel<max_val>(p1 + p1_delta);
        }
        if (aq < beta) {
            const int q1_delta = deblock_clip3((q2 + pq_avg - (q1 << 1)) >> 1, -tc0, tc0);
            PIX(boundary_y + 1) = (Type)deblock_clip_pixel<max_val>(q1 + q1_delta);
        }
    }
#undef PIX
}

template<typename Type, int bit_depth>
static RGY_ERR deblock_plane(RGYFrameInfo *pOutputFrame, const int alpha, const int beta, const int tc0,
    const int is_chroma, cudaStream_t stream) {
    static constexpr int max_val = (1 << bit_depth) - 1;
    const int numVertEdges = (pOutputFrame->width / 4) - 1;
    if (numVertEdges > 0) {
        dim3 blockSize(8, 32);
        dim3 gridSize(divCeil(numVertEdges, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));
        kernel_deblock_vertical<Type, max_val><<<gridSize, blockSize, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0],
            pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height, alpha, beta, tc0, is_chroma);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    }

    const int numHorzEdges = (pOutputFrame->height / 4) - 1;
    if (numHorzEdges > 0) {
        dim3 blockSize(DEBLOCK_BLOCK_X, DEBLOCK_BLOCK_Y);
        dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(numHorzEdges, blockSize.y));
        kernel_deblock_horizontal<Type, max_val><<<gridSize, blockSize, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0],
            pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height, alpha, beta, tc0, is_chroma);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR deblock_frame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int qp, const int alphaOffset, const int betaOffset, const bool chroma, cudaStream_t stream) {
    const int indexA = std::min(51, std::max(0, qp + alphaOffset));
    const int indexB = std::min(51, std::max(0, qp + betaOffset));
    const int bdShift = std::max(0, bit_depth - 8);
    const int alpha = H264_ALPHA[indexA] << bdShift;
    const int beta  = H264_BETA[indexB] << bdShift;
    const int tc0   = H264_TC0_BS1[indexA] << bdShift;

    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int i = 0; i < planes; i++) {
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto sts = copyPlaneAsync(&planeDst, &planeSrc, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    const int planeMax = chroma ? planes : 1;
    for (int i = 0; i < planeMax; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        if (planeDst.width < 8 || planeDst.height < 8) continue;
        auto sts = deblock_plane<Type, bit_depth>(&planeDst, alpha, beta, tc0, (i == 0) ? 0 : 1, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

NVEncFilterDeblock::NVEncFilterDeblock() {
    m_name = _T("deblock");
}

NVEncFilterDeblock::~NVEncFilterDeblock() {
    close();
}

RGY_ERR NVEncFilterDeblock::checkParam(const std::shared_ptr<NVEncFilterParamDeblock> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->deblock.qp < 0 || 51 < prm->deblock.qp) {
        prm->deblock.qp = clamp(prm->deblock.qp, 0, 51);
        AddMessage(RGY_LOG_WARN, _T("qp should be in range of [0, 51] (ITU-T Rec. H.264 section 8.7); clamped.\n"));
    }
    if (prm->deblock.alpha < -6 || 6 < prm->deblock.alpha) {
        prm->deblock.alpha = clamp(prm->deblock.alpha, -6, 6);
        AddMessage(RGY_LOG_WARN, _T("alpha offset should be in range of [-6, 6]; clamped.\n"));
    }
    if (prm->deblock.beta < -6 || 6 < prm->deblock.beta) {
        prm->deblock.beta = clamp(prm->deblock.beta, -6, 6);
        AddMessage(RGY_LOG_WARN, _T("beta offset should be in range of [-6, 6]; clamped.\n"));
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp];
    if (rgy_chromafmt_is_rgb(chromaFormat)
        || (RGY_CSP_PLANES[prm->frameIn.csp] == 1 && chromaFormat != RGY_CHROMAFMT_MONOCHROME)) {
        AddMessage(RGY_LOG_ERROR, _T("deblock supports planar/semi-planar YUV or monochrome formats only: %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->deblock.chroma && RGY_CSP_PLANES[prm->frameIn.csp] < 3) {
        prm->deblock.chroma = false;
        AddMessage(RGY_LOG_WARN, _T("deblock chroma processing requires planar chroma; disabled for %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDeblock::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDeblock>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamDeblock::print() const {
    return deblock.print();
}

RGY_ERR NVEncFilterDeblock::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("deblock only supports device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("deblock does not support csp conversion.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDeblock>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    static const std::map<RGY_CSP, decltype(deblock_frame<uint8_t, 8>)*> deblock_list = {
        { RGY_CSP_YV12,      deblock_frame<uint8_t,   8> },
        { RGY_CSP_YV12_16,   deblock_frame<uint16_t, 16> },
        { RGY_CSP_YUV444,    deblock_frame<uint8_t,   8> },
        { RGY_CSP_YUV444_16, deblock_frame<uint16_t, 16> },
        { RGY_CSP_Y8,        deblock_frame<uint8_t,   8> },
        { RGY_CSP_Y16,       deblock_frame<uint16_t, 16> }
    };
    if (deblock_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    sts = deblock_list.at(pInputFrame->csp)(ppOutputFrames[0], pInputFrame,
        prm->deblock.qp, prm->deblock.alpha, prm->deblock.beta, prm->deblock.chroma, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at deblock(%s): %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterDeblock::close() {
    m_frameBuf.clear();
}
