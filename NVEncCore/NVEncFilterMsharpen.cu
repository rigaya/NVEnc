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

#include <map>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include "convert_csp.h"
#include "NVEncFilterMsharpen.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int MSHARPEN_BLOCK_X = 32;
static const int MSHARPEN_BLOCK_Y = 8;

// Read pixel with boundary clamp, returns value as float in [0,1]
template<typename Type, int bit_depth>
__device__ __inline__ float msharpen_read_f(const uint8_t *pSrc, int pitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const Type val = *(const Type *)(pSrc + y * pitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

__device__ __inline__ float msharpen_sigmoid_weight(float g, float threshold, float slope) {
    float arg = (g - threshold) * slope;
    arg = clamp(arg, -32.0f, 32.0f);
    return 1.0f / (1.0f + expf(-arg));
}

template<typename Type, int bit_depth>
__global__ void kernel_msharpen(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pSrc, const int srcPitch,
    const float strength, const float threshold,
    const float slope, const float luma_limit_norm,
    const int highq, const int mask,
    const float block_protect) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }

    float s[5][4];
    for (int dy = -1; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            s[dx + 2][dy + 1] = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix + dx, iy + dy, dstWidth, dstHeight);
        }
    }

    const float b_cc = (
          s[1][0] + s[2][0] + s[3][0]
        + s[1][1] + s[2][1] + s[3][1]
        + s[1][2] + s[2][2] + s[3][2]) * (1.0f / 9.0f);
    const float b_br = (
          s[2][1] + s[3][1] + s[4][1]
        + s[2][2] + s[3][2] + s[4][2]
        + s[2][3] + s[3][3] + s[4][3]) * (1.0f / 9.0f);
    const float b_bl = (
          s[0][1] + s[1][1] + s[2][1]
        + s[0][2] + s[1][2] + s[2][2]
        + s[0][3] + s[1][3] + s[2][3]) * (1.0f / 9.0f);

    const float src = s[2][1];
    const float g_br = fabsf(b_cc - b_br);
    const float g_bl = fabsf(b_cc - b_bl);
    float g_max = fmaxf(g_br, g_bl);
    int edge = (g_br >= threshold) || (g_bl >= threshold);

    if (highq) {
        const float b_bc = (
              s[1][1] + s[2][1] + s[3][1]
            + s[1][2] + s[2][2] + s[3][2]
            + s[1][3] + s[2][3] + s[3][3]) * (1.0f / 9.0f);
        const float b_cr = (
              s[2][0] + s[3][0] + s[4][0]
            + s[2][1] + s[3][1] + s[4][1]
            + s[2][2] + s[3][2] + s[4][2]) * (1.0f / 9.0f);
        const float g_bc = fabsf(b_cc - b_bc);
        const float g_cr = fabsf(b_cc - b_cr);
        edge = edge || (g_bc >= threshold) || (g_cr >= threshold);
        g_max = fmaxf(g_max, fmaxf(g_bc, g_cr));
    }

    const float soft_w = (slope > 0.0f) ? msharpen_sigmoid_weight(g_max, threshold, slope) : (edge ? 1.0f : 0.0f);

    float effective_mask = soft_w;
    if (block_protect > 0.0f) {
        const int nearest_v = ((ix + 4) >> 3) << 3;
        const int dx = ix - nearest_v;
        const int abs_dx = (dx < 0) ? -dx : dx;
        float v_block_w = 0.0f;
        if (abs_dx <= 1) {
            const float pl = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, nearest_v - 1, iy, dstWidth, dstHeight);
            const float pr = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, nearest_v,     iy, dstWidth, dstHeight);
            if (fabsf(pl - pr) > threshold) {
                v_block_w = (abs_dx == 0) ? 1.0f : 0.5f;
            }
        }
        const int nearest_h = ((iy + 4) >> 3) << 3;
        const int dy = iy - nearest_h;
        const int abs_dy = (dy < 0) ? -dy : dy;
        float h_block_w = 0.0f;
        if (abs_dy <= 1) {
            const float pu = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix, nearest_h - 1, dstWidth, dstHeight);
            const float pd = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix, nearest_h,     dstWidth, dstHeight);
            if (fabsf(pu - pd) > threshold) {
                h_block_w = (abs_dy == 0) ? 1.0f : 0.5f;
            }
        }
        const float block_w = fmaxf(v_block_w, h_block_w);
        effective_mask = soft_w * (1.0f - block_protect * block_w);
    }

    const float luma_w = (luma_limit_norm > 0.0f) ? fminf(src / luma_limit_norm, 1.0f) : 1.0f;

    float result;
    if (mask) {
        result = effective_mask;
    } else {
        const float sharpened = clamp(4.0f * src - 3.0f * b_cc, 0.0f, 1.0f);
        const float w = effective_mask * strength * luma_w;
        result = w * sharpened + (1.0f - w) * src;
    }

    Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(clamp(result, 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
static RGY_ERR msharpen_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    float threshold, float strength, float slope, float luma_limit_norm, bool highq, bool mask, float block_protect, cudaStream_t stream) {
    dim3 blockSize(MSHARPEN_BLOCK_X, MSHARPEN_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    const float th_norm = threshold / (float)((1 << bit_depth) - 1);

    kernel_msharpen<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        strength, th_norm, slope, luma_limit_norm, highq ? 1 : 0, mask ? 1 : 0, block_protect);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

NVEncFilterMsharpen::NVEncFilterMsharpen() {
    m_name = _T("msharpen");
}

NVEncFilterMsharpen::~NVEncFilterMsharpen() {
    close();
}

RGY_ERR NVEncFilterMsharpen::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pMsharpenParam = std::dynamic_pointer_cast<NVEncFilterParamMsharpen>(pParam);
    if (!pMsharpenParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pMsharpenParam->frameOut.height <= 0 || pMsharpenParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pMsharpenParam->msharpen.strength < 0.0f || 1.0f < pMsharpenParam->msharpen.strength) {
        pMsharpenParam->msharpen.strength = clamp(pMsharpenParam->msharpen.strength, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (pMsharpenParam->msharpen.threshold < 0.0f || 255.0f < pMsharpenParam->msharpen.threshold) {
        pMsharpenParam->msharpen.threshold = clamp(pMsharpenParam->msharpen.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (pMsharpenParam->msharpen.slope < 0.0f) {
        pMsharpenParam->msharpen.slope = 0.0f;
        AddMessage(RGY_LOG_WARN, _T("slope must be >= 0; clamped to 0 (binary gate).\n"));
    }
    if (pMsharpenParam->msharpen.luma_limit < 0.0f || 255.0f < pMsharpenParam->msharpen.luma_limit) {
        pMsharpenParam->msharpen.luma_limit = clamp(pMsharpenParam->msharpen.luma_limit, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("luma_limit should be in range of %.1f - %.1f (0 disables).\n"), 0.0f, 255.0f);
    }
    if (pMsharpenParam->msharpen.block_protect < 0.0f || 1.0f < pMsharpenParam->msharpen.block_protect) {
        pMsharpenParam->msharpen.block_protect = clamp(pMsharpenParam->msharpen.block_protect, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("block_protect should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }

    sts = AllocFrameBuf(pMsharpenParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pMsharpenParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(pParam->print());
    m_param = pMsharpenParam;
    return sts;
}

tstring NVEncFilterParamMsharpen::print() const {
    return msharpen.print();
}

RGY_ERR NVEncFilterMsharpen::procPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGY_PLANE plane, cudaStream_t stream) {
    auto pMsharpenParam = std::dynamic_pointer_cast<NVEncFilterParamMsharpen>(m_param);
    if (!pMsharpenParam) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    if (bitDepth <= 0 || 16 < bitDepth) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp/bit depth: %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    const float slope_norm = pMsharpenParam->msharpen.slope * 255.0f;
    const float luma_limit_norm = (plane == RGY_PLANE_Y && pMsharpenParam->msharpen.luma_limit > 0.0f)
        ? pMsharpenParam->msharpen.luma_limit / 255.0f
        : 0.0f;

    switch (bitDepth) {
    case 8:
        return msharpen_plane<uint8_t, 8>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    case 9:
        return msharpen_plane<uint16_t, 9>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    case 10:
        return msharpen_plane<uint16_t, 10>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    case 12:
        return msharpen_plane<uint16_t, 12>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    case 14:
        return msharpen_plane<uint16_t, 14>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    case 16:
        return msharpen_plane<uint16_t, 16>(pOutputFrame, pInputFrame,
            pMsharpenParam->msharpen.threshold, pMsharpenParam->msharpen.strength,
            slope_norm, luma_limit_norm,
            pMsharpenParam->msharpen.highq, pMsharpenParam->msharpen.mask,
            pMsharpenParam->msharpen.block_protect, stream);
    default:
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth: %d.\n"), bitDepth);
        return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterMsharpen::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pOutputFrame->csp)];
    for (int ip = 0; ip < nPlanes; ip++) {
        const auto plane = (RGY_PLANE)ip;
        auto planeInput = getPlane(pInputFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
        auto err = procPlane(&planeOutput, &planeInput, plane, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    auto err = copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) return err;

    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMsharpen::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = procFrame(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at msharpen(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterMsharpen::close() {
    m_frameBuf.clear();
}
