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

// Kernel 1: 3x3 Box Blur
template<typename Type, int bit_depth>
__global__ void kernel_msharpen_blur(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pSrc, const int srcPitch) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy - 1, dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy - 1, dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy + 1, dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);
        sum += msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy + 1, dstWidth, dstHeight);

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(sum * (1.0f / 9.0f), 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth) - 1));
    }
}

// Kernel 2: Edge-Selective Sharpening
template<typename Type, int bit_depth>
__global__ void kernel_msharpen_sharpen(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pSrc, const int srcPitch,
    const uint8_t *pBlur, const int blurPitch,
    const float threshold, const float strength, const int highq, const int mask) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const float src = msharpen_read_f<Type, bit_depth>(pSrc, srcPitch, ix, iy, dstWidth, dstHeight);

        const float b_cc = msharpen_read_f<Type, bit_depth>(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
        const float b_br = msharpen_read_f<Type, bit_depth>(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
        const float b_bl = msharpen_read_f<Type, bit_depth>(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);

        // Diagonal edge detection
        int edge = (fabsf(b_cc - b_br) >= threshold) || (fabsf(b_cc - b_bl) >= threshold);

        // High quality: add vertical and horizontal
        if (highq) {
            const float b_bc = msharpen_read_f<Type, bit_depth>(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            const float b_cr = msharpen_read_f<Type, bit_depth>(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            edge = edge || (fabsf(b_cc - b_bc) >= threshold) || (fabsf(b_cc - b_cr) >= threshold);
        }

        float result;
        if (mask) {
            result = edge ? 1.0f : 0.0f;
        } else if (edge) {
            float sharpened = 4.0f * src - 3.0f * b_cc;
            sharpened = clamp(sharpened, 0.0f, 1.0f);
            result = strength * sharpened + (1.0f - strength) * src;
        } else {
            result = src;
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(result, 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth) - 1));
    }
}

template<typename Type, int bit_depth>
static RGY_ERR msharpen_blur_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(MSHARPEN_BLOCK_X, MSHARPEN_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_msharpen_blur<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0]);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR msharpen_sharpen_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pBlurFrame,
    float threshold, float strength, bool highq, bool mask, cudaStream_t stream) {
    dim3 blockSize(MSHARPEN_BLOCK_X, MSHARPEN_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    // Normalize threshold from [0,255] to [0,1]
    const float th_norm = threshold / (float)((1 << bit_depth) - 1);

    kernel_msharpen_sharpen<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        (const uint8_t *)pBlurFrame->ptr[0], pBlurFrame->pitch[0],
        th_norm, strength, highq ? 1 : 0, mask ? 1 : 0);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

NVEncFilterMsharpen::NVEncFilterMsharpen() : m_blur() {
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

    sts = AllocFrameBuf(pMsharpenParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pMsharpenParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Allocate blur buffers (one per plane)
    const int nPlanes = RGY_CSP_PLANES[pMsharpenParam->frameOut.csp];
    m_blur.resize(nPlanes);
    for (int ip = 0; ip < nPlanes; ip++) {
        auto plane = getPlane(&pMsharpenParam->frameOut, (RGY_PLANE)ip);
        m_blur[ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pMsharpenParam->frameOut.csp);
        m_blur[ip]->releasePtr();
        sts = m_blur[ip]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate blur buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(pParam->print());
    m_param = pMsharpenParam;
    return sts;
}

tstring NVEncFilterParamMsharpen::print() const {
    return msharpen.print();
}

RGY_ERR NVEncFilterMsharpen::procPlaneBlur(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pMsharpenParam = std::dynamic_pointer_cast<NVEncFilterParamMsharpen>(m_param);

    static const std::map<RGY_CSP, decltype(msharpen_blur_plane<uint8_t, 8>)*> blur_list = {
        { RGY_CSP_YV12,      msharpen_blur_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   msharpen_blur_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    msharpen_blur_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, msharpen_blur_plane<uint16_t, 16> }
    };
    if (blur_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return blur_list.at(pInputFrame->csp)(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterMsharpen::procPlaneSharpen(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pBlurFrame,
    float threshold, float strength, bool highq, bool mask, cudaStream_t stream) {
    static const std::map<RGY_CSP, decltype(msharpen_sharpen_plane<uint8_t, 8>)*> sharpen_list = {
        { RGY_CSP_YV12,      msharpen_sharpen_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   msharpen_sharpen_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    msharpen_sharpen_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, msharpen_sharpen_plane<uint16_t, 16> }
    };
    if (sharpen_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return sharpen_list.at(pInputFrame->csp)(pOutputFrame, pInputFrame, pBlurFrame, threshold, strength, highq, mask, stream);
}

RGY_ERR NVEncFilterMsharpen::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pMsharpenParam = std::dynamic_pointer_cast<NVEncFilterParamMsharpen>(m_param);
    if (!pMsharpenParam) {
        return RGY_ERR_INVALID_PARAM;
    }

    const int nPlanes = (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB)
        ? RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]
        : 1; // Process Y plane only for YUV

    if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB) {
        for (int ip = 0; ip < nPlanes; ip++) {
            auto plane = (RGY_PLANE)ip;
            const auto planeInput = getPlane(pInputFrame, plane);
            auto planeOutput = getPlane(pOutputFrame, plane);
            auto planeBlur = getPlane(&m_blur[ip]->frame, RGY_PLANE_Y);

            // Step 1: blur
            auto err = procPlaneBlur(&planeBlur, &planeInput, stream);
            if (err != RGY_ERR_NONE) return err;

            // Step 2: sharpen
            err = procPlaneSharpen(&planeOutput, &planeInput, &planeBlur,
                pMsharpenParam->msharpen.threshold,
                pMsharpenParam->msharpen.strength,
                pMsharpenParam->msharpen.highq,
                pMsharpenParam->msharpen.mask,
                stream);
            if (err != RGY_ERR_NONE) return err;
        }
    } else {
        // YUV: process Y plane, copy UV
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeBlurY = getPlane(&m_blur[0]->frame, RGY_PLANE_Y);

        auto err = procPlaneBlur(&planeBlurY, &planeInputY, stream);
        if (err != RGY_ERR_NONE) return err;

        err = procPlaneSharpen(&planeOutputY, &planeInputY, &planeBlurY,
            pMsharpenParam->msharpen.threshold,
            pMsharpenParam->msharpen.strength,
            pMsharpenParam->msharpen.highq,
            pMsharpenParam->msharpen.mask,
            stream);
        if (err != RGY_ERR_NONE) return err;

        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        err = copyPlaneAsync(&planeOutputU, &planeInputU, stream);
        if (err != RGY_ERR_NONE) return err;
        err = copyPlaneAsync(&planeOutputV, &planeInputV, stream);
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
    m_blur.clear();
    m_frameBuf.clear();
}
