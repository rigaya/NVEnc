// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
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
// -----------------------------------------------------------------------------------------

#include <array>
#include "convert_csp.h"
#include "NVEncFilterSoftLight.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static constexpr int SOFTLIGHT_BLOCK_SIZE = 256;

__device__ __forceinline__ uint16_t softlight_to_u16(const float v) {
    const float x = fminf(fmaxf(v, 0.0f), 1.0f);
    return (uint16_t)(x * 65535.0f + 0.5f);
}

__device__ __forceinline__ float softlight_func(const float a, const float b, const VppSoftLightFormula formula) {
    if (formula == VppSoftLightFormula::ILLUSIONSHU) {
        return powf(a, powf(2.0f, 1.0f - 2.0f * b));
    }
    if (formula == VppSoftLightFormula::W3C) {
        if (b <= 0.5f) {
            return a - (1.0f - 2.0f * b) * a * (1.0f - a);
        }
        const float g = (a <= 0.25f) ? (((16.0f * a - 12.0f) * a + 4.0f) * a) : sqrtf(a);
        return a + (2.0f * b - 1.0f) * (g - a);
    }
    return (1.0f - 2.0f * b) * a * a + 2.0f * b * a;
}

__device__ __forceinline__ void rgb_to_hsv_value(const float r, const float g, const float b, float& h, float& s, float& v) {
    const float mx = fmaxf(r, fmaxf(g, b));
    const float mn = fminf(r, fminf(g, b));
    const float d = mx - mn;
    v = mx;
    s = (mx <= 0.0f) ? 0.0f : d / mx;
    if (d <= 0.0f) {
        h = 0.0f;
    } else if (mx == r) {
        h = fmodf((g - b) / d, 6.0f);
    } else if (mx == g) {
        h = (b - r) / d + 2.0f;
    } else {
        h = (r - g) / d + 4.0f;
    }
    if (h < 0.0f) h += 6.0f;
}

__device__ __forceinline__ void hsv_to_rgb_value(float h, const float s, const float v, float& r, float& g, float& b) {
    if (s <= 0.0f) {
        r = g = b = v;
        return;
    }
    h = fmodf(h, 6.0f);
    if (h < 0.0f) h += 6.0f;
    const float c = v * s;
    const float x = c * (1.0f - fabsf(fmodf(h, 2.0f) - 1.0f));
    const float m = v - c;
    if (h < 1.0f) {
        r = c; g = x; b = 0.0f;
    } else if (h < 2.0f) {
        r = x; g = c; b = 0.0f;
    } else if (h < 3.0f) {
        r = 0.0f; g = c; b = x;
    } else if (h < 4.0f) {
        r = 0.0f; g = x; b = c;
    } else if (h < 5.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }
    r += m;
    g += m;
    b += m;
}

__global__ void kernel_reduce_rgb_u16(
    const uint8_t *__restrict__ pR, const int pitchR,
    const uint8_t *__restrict__ pG, const int pitchG,
    const uint8_t *__restrict__ pB, const int pitchB,
    const int width, const int height, unsigned long long *__restrict__ result) {
    __shared__ unsigned long long sh[6][SOFTLIGHT_BLOCK_SIZE];
    unsigned long long sumR = 0, sumG = 0, sumB = 0;
    unsigned long long blackR = 0, blackG = 0, blackB = 0;
    const int tid = threadIdx.x;
    const int64_t total = (int64_t)width * height;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        const int x = (int)(idx % width);
        const int y = (int)(idx / width);
        const auto r = ((const uint16_t *)(pR + y * pitchR))[x];
        const auto g = ((const uint16_t *)(pG + y * pitchG))[x];
        const auto b = ((const uint16_t *)(pB + y * pitchB))[x];
        sumR += r; sumG += g; sumB += b;
        blackR += (r == 0); blackG += (g == 0); blackB += (b == 0);
    }
    sh[0][tid] = sumR; sh[1][tid] = sumG; sh[2][tid] = sumB;
    sh[3][tid] = blackR; sh[4][tid] = blackG; sh[5][tid] = blackB;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                sh[i][tid] += sh[i][tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            atomicAdd(&result[i], sh[i][0]);
        }
    }
}

__global__ void kernel_softlight_scalar_u16(
    uint8_t *__restrict__ pPlane, const int pitch, const int width, const int height,
    const float b, const VppSoftLightFormula formula) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        auto ptr = (uint16_t *)(pPlane + y * pitch);
        const float a = ptr[x] * (1.0f / 65535.0f);
        ptr[x] = softlight_to_u16(softlight_func(a, b, formula));
    }
}

__global__ void kernel_softlight_self_u16(
    uint8_t *__restrict__ pPlane, const int pitch, const int width, const int height,
    const VppSoftLightFormula formula) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        auto ptr = (uint16_t *)(pPlane + y * pitch);
        const float a = ptr[x] * (1.0f / 65535.0f);
        ptr[x] = softlight_to_u16(softlight_func(a, a, formula));
    }
}

__global__ void kernel_softlight_self_f32(float *__restrict__ pPlane, const int width, const int height, const VppSoftLightFormula formula) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        auto& a = pPlane[(int64_t)y * width + x];
        a = fminf(fmaxf(softlight_func(a, a, formula), 0.0f), 1.0f);
    }
}

__global__ void kernel_rgb_to_v_u16(
    const uint8_t *__restrict__ pR, const int pitchR,
    const uint8_t *__restrict__ pG, const int pitchG,
    const uint8_t *__restrict__ pB, const int pitchB,
    const int width, const int height, float *__restrict__ pV) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const float r = ((const uint16_t *)(pR + y * pitchR))[x] * (1.0f / 65535.0f);
        const float g = ((const uint16_t *)(pG + y * pitchG))[x] * (1.0f / 65535.0f);
        const float b = ((const uint16_t *)(pB + y * pitchB))[x] * (1.0f / 65535.0f);
        pV[(int64_t)y * width + x] = fmaxf(r, fmaxf(g, b));
    }
}

__global__ void kernel_rgb_to_hs_u16(
    const uint8_t *__restrict__ pR, const int pitchR,
    const uint8_t *__restrict__ pG, const int pitchG,
    const uint8_t *__restrict__ pB, const int pitchB,
    const int width, const int height, float *__restrict__ pH, float *__restrict__ pS) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float h, s, v;
        const float r = ((const uint16_t *)(pR + y * pitchR))[x] * (1.0f / 65535.0f);
        const float g = ((const uint16_t *)(pG + y * pitchG))[x] * (1.0f / 65535.0f);
        const float b = ((const uint16_t *)(pB + y * pitchB))[x] * (1.0f / 65535.0f);
        rgb_to_hsv_value(r, g, b, h, s, v);
        const int64_t idx = (int64_t)y * width + x;
        pH[idx] = h;
        pS[idx] = s;
    }
}

__global__ void kernel_rgb_to_hsv_u16(
    const uint8_t *__restrict__ pR, const int pitchR,
    const uint8_t *__restrict__ pG, const int pitchG,
    const uint8_t *__restrict__ pB, const int pitchB,
    const int width, const int height, float *__restrict__ pH, float *__restrict__ pS, float *__restrict__ pV) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float h, s, v;
        const float r = ((const uint16_t *)(pR + y * pitchR))[x] * (1.0f / 65535.0f);
        const float g = ((const uint16_t *)(pG + y * pitchG))[x] * (1.0f / 65535.0f);
        const float b = ((const uint16_t *)(pB + y * pitchB))[x] * (1.0f / 65535.0f);
        rgb_to_hsv_value(r, g, b, h, s, v);
        const int64_t idx = (int64_t)y * width + x;
        pH[idx] = h;
        pS[idx] = s;
        pV[idx] = v;
    }
}

__global__ void kernel_hsv_to_rgb_u16(
    uint8_t *__restrict__ pR, const int pitchR,
    uint8_t *__restrict__ pG, const int pitchG,
    uint8_t *__restrict__ pB, const int pitchB,
    const int width, const int height,
    const float *__restrict__ pH, const float *__restrict__ pS, const float *__restrict__ pV) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int64_t idx = (int64_t)y * width + x;
        float r, g, b;
        hsv_to_rgb_value(pH[idx], pS[idx], pV[idx], r, g, b);
        ((uint16_t *)(pR + y * pitchR))[x] = softlight_to_u16(r);
        ((uint16_t *)(pG + y * pitchG))[x] = softlight_to_u16(g);
        ((uint16_t *)(pB + y * pitchB))[x] = softlight_to_u16(b);
    }
}

NVEncFilterSoftLight::NVEncFilterSoftLight() :
    NVEncFilter(),
    m_convIn(),
    m_convOut(),
    m_hsvH(),
    m_hsvS(),
    m_hsvV(),
    m_reduce() {
    m_name = _T("softlight");
}

NVEncFilterSoftLight::~NVEncFilterSoftLight() {
    close();
}

tstring NVEncFilterParamSoftLight::print() const {
    return softlight.print();
}

RGY_ERR NVEncFilterSoftLight::checkParam(const std::shared_ptr<NVEncFilterParamSoftLight> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_csp_has_alpha(prm->frameOut.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("softlight is not supported on alpha csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterSoftLight::allocWork(const RGYFrameInfo& rgbFrame) {
    const auto frameSize = (size_t)rgbFrame.width * rgbFrame.height * sizeof(float);
    auto allocBuf = [&](std::unique_ptr<CUMemBuf>& buf, const size_t size, const TCHAR *name) {
        if (!buf || buf->nSize < size) {
            buf = std::make_unique<CUMemBuf>(size);
            auto sts = buf->alloc();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate %s buffer: %s.\n"), name, get_err_mes(sts));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocBuf(m_hsvH, frameSize, _T("HSV H"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocBuf(m_hsvS, frameSize, _T("HSV S"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocBuf(m_hsvV, frameSize, _T("HSV V"));
    if (sts != RGY_ERR_NONE) return sts;
    return allocBuf(m_reduce, sizeof(unsigned long long) * 6, _T("reduce"));
}

RGY_ERR NVEncFilterSoftLight::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSoftLight>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    const auto rgbCsp = RGY_CSP_RGB_16;
    if (pParam->frameIn.csp != rgbCsp) {
        VideoVUIInfo vui = prm->vuiInfo;
        if (vui.matrix == RGY_MATRIX_UNSPECIFIED) {
            vui.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
        }
        vui.apply_auto(vui, pParam->frameIn.height);
        {
            unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
            shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->frameOut.csp = rgbCsp;
            paramCrop->matrix = vui.matrix;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) return sts;
            m_convIn = std::move(filter);
        }
        {
            unique_ptr<NVEncFilterCspCrop> filter(new NVEncFilterCspCrop());
            shared_ptr<NVEncFilterParamCrop> paramCrop(new NVEncFilterParamCrop());
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameIn.csp = rgbCsp;
            paramCrop->frameOut = pParam->frameOut;
            paramCrop->matrix = vui.matrix;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) return sts;
            m_convOut = std::move(filter);
        }
    } else {
        m_convIn.reset();
        m_convOut.reset();
    }

    RGYFrameInfo rgbFrame = pParam->frameIn;
    rgbFrame.csp = rgbCsp;
    if ((sts = allocWork(rgbFrame)) != RGY_ERR_NONE) {
        return sts;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    tstring info = _T("softlight: ");
    if (m_convIn) {
        info += m_convIn->GetInputMessage() + _T("\n");
    }
    const auto extraIndent = tstring(_tcslen(_T("softlight: ")), _T(' '));
    info += tstring(INFO_INDENT) + extraIndent + pParam->print();
    if (m_convOut) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + extraIndent + m_convOut->GetInputMessage();
    }
    setFilterInfo(info);
    m_param = prm;
    return sts;
}

RGY_ERR NVEncFilterSoftLight::procFrame(RGYFrameInfo *pFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSoftLight>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pFrame->csp != RGY_CSP_RGB_16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    const auto planeR = getPlane(pFrame, RGY_PLANE_R);
    const auto planeG = getPlane(pFrame, RGY_PLANE_G);
    const auto planeB = getPlane(pFrame, RGY_PLANE_B);
    const int width = planeR.width;
    const int height = planeR.height;
    if ((int64_t)width * height <= 0) {
        return RGY_ERR_NONE;
    }
    if (auto sts = allocWork(*pFrame); sts != RGY_ERR_NONE) {
        return sts;
    }

    const auto mode = prm->softlight.mode;
    const auto formula = prm->softlight.formula;
    const bool neutralize =
        mode == VppSoftLightMode::NEUTRALIZE
        || mode == VppSoftLightMode::LIGHTNESS
        || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT
        || mode == VppSoftLightMode::NEUTRALIZE_FULL
        || mode == VppSoftLightMode::NEUTRALIZE_BOOST;
    const bool rgbBoost =
        mode == VppSoftLightMode::NEUTRALIZE_BOOST
        || mode == VppSoftLightMode::BOOST;

    dim3 block2d(32, 8);
    dim3 grid2d(divCeil(width, block2d.x), divCeil(height, block2d.y));
    auto ptrH = (float *)m_hsvH->ptr;
    auto ptrS = (float *)m_hsvS->ptr;
    auto ptrV = (float *)m_hsvV->ptr;

    if (mode == VppSoftLightMode::NEUTRALIZE || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
        kernel_rgb_to_v_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrV);
    } else if (mode == VppSoftLightMode::LIGHTNESS) {
        kernel_rgb_to_hs_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS);
    }
    if (auto sts = err_to_rgy(cudaGetLastError()); sts != RGY_ERR_NONE) return sts;

    if (neutralize) {
        auto cudaerr = cudaMemsetAsync(m_reduce->ptr, 0, sizeof(unsigned long long) * 6, stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        const auto total = (int64_t)width * height;
        const int grid = (int)std::min<int64_t>(divCeil(total, (int64_t)SOFTLIGHT_BLOCK_SIZE * 8), 65535);
        kernel_reduce_rgb_u16<<<grid, SOFTLIGHT_BLOCK_SIZE, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, (unsigned long long *)m_reduce->ptr);
        if (auto sts = err_to_rgy(cudaGetLastError()); sts != RGY_ERR_NONE) return sts;
        std::array<unsigned long long, 6> host = {};
        cudaerr = cudaMemcpyAsync(host.data(), m_reduce->ptr, sizeof(host[0]) * host.size(), cudaMemcpyDeviceToHost, stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        cudaerr = cudaStreamSynchronize(stream);
        if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
        const double totalPx = (double)total;
        std::array<float, 3> b = {};
        for (int i = 0; i < 3; i++) {
            const double denom = totalPx - (prm->softlight.skipblack ? (double)host[3 + i] : 0.0);
            const double mean = (denom > 0.0) ? ((double)host[i] / denom) / 65535.0 : 0.0;
            b[i] = (float)(1.0 - mean);
        }
        kernel_softlight_scalar_u16<<<grid2d, block2d, 0, stream>>>(planeR.ptr[0], planeR.pitch[0], width, height, b[0], formula);
        kernel_softlight_scalar_u16<<<grid2d, block2d, 0, stream>>>(planeG.ptr[0], planeG.pitch[0], width, height, b[1], formula);
        kernel_softlight_scalar_u16<<<grid2d, block2d, 0, stream>>>(planeB.ptr[0], planeB.pitch[0], width, height, b[2], formula);
        if (auto sts = err_to_rgy(cudaGetLastError()); sts != RGY_ERR_NONE) return sts;
    }

    if (rgbBoost) {
        kernel_softlight_self_u16<<<grid2d, block2d, 0, stream>>>(planeR.ptr[0], planeR.pitch[0], width, height, formula);
        kernel_softlight_self_u16<<<grid2d, block2d, 0, stream>>>(planeG.ptr[0], planeG.pitch[0], width, height, formula);
        kernel_softlight_self_u16<<<grid2d, block2d, 0, stream>>>(planeB.ptr[0], planeB.pitch[0], width, height, formula);
        if (auto sts = err_to_rgy(cudaGetLastError()); sts != RGY_ERR_NONE) return sts;
    }

    if (mode == VppSoftLightMode::NEUTRALIZE || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
        kernel_rgb_to_hs_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS);
        if (mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
            kernel_softlight_self_f32<<<grid2d, block2d, 0, stream>>>(ptrS, width, height, formula);
        }
        kernel_hsv_to_rgb_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS, ptrV);
    } else if (mode == VppSoftLightMode::LIGHTNESS) {
        kernel_rgb_to_v_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrV);
        kernel_hsv_to_rgb_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS, ptrV);
    } else if (mode == VppSoftLightMode::SATURATION) {
        kernel_rgb_to_hsv_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS, ptrV);
        kernel_softlight_self_f32<<<grid2d, block2d, 0, stream>>>(ptrS, width, height, formula);
        kernel_hsv_to_rgb_u16<<<grid2d, block2d, 0, stream>>>(
            planeR.ptr[0], planeR.pitch[0], planeG.ptr[0], planeG.pitch[0], planeB.ptr[0], planeB.pitch[0],
            width, height, ptrH, ptrS, ptrV);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterSoftLight::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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

    RGYFrameInfo targetFrame = *pInputFrame;
    if (m_convIn) {
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { nullptr };
        auto sts_filter = m_convIn->filter(&targetFrame, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, stream);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        targetFrame = *outInfo[0];
    } else {
        auto sts_copy = copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
        if (sts_copy != RGY_ERR_NONE) {
            return sts_copy;
        }
        targetFrame = *ppOutputFrames[0];
    }

    if ((sts = procFrame(&targetFrame, stream)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at softlight(%s): %s.\n"),
            RGY_CSP_NAMES[targetFrame.csp], get_err_mes(sts));
        return sts;
    }

    if (m_convOut) {
        auto sts_filter = m_convOut->filter(&targetFrame, ppOutputFrames, pOutputFrameNum, stream);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
    }
    return sts;
}

void NVEncFilterSoftLight::close() {
    m_convIn.reset();
    m_convOut.reset();
    m_hsvH.reset();
    m_hsvS.reset();
    m_hsvV.reset();
    m_reduce.reset();
    m_frameBuf.clear();
}
