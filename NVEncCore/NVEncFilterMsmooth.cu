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
#include "NVEncFilterMsmooth.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int MSMOOTH_BLOCK_X = 32;
static const int MSMOOTH_BLOCK_Y = 8;

// Read pixel with boundary clamp, returns value as float in [0,1]
template<typename Type, int bit_depth>
__device__ __inline__ float msmooth_read_f(const uint8_t *pSrc, int pitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const Type val = *(const Type *)(pSrc + y * pitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

// Read pixel as integer with boundary clamp
template<typename Type, int bit_depth>
__device__ __inline__ int msmooth_read_i(const uint8_t *pSrc, int pitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(pSrc + y * pitch + x * sizeof(Type)));
}

// Kernel 1: 3x3 Box Blur
template<typename Type, int bit_depth>
__global__ void kernel_msmooth_blur(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pSrc, const int srcPitch) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy - 1, dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy - 1, dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy + 1, dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);
        sum += msmooth_read_f<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy + 1, dstWidth, dstHeight);

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(sum * (1.0f / 9.0f), 0.0f, 1.0f - RGY_FLT_EPS) * ((1 << bit_depth) - 1));
    }
}

// Kernel 2: Edge Mask
template<typename Type, int bit_depth>
__global__ void kernel_msmooth_edge_mask(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *pBlur, const int blurPitch,
    const float threshold, const int highq) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const float b_cc = msmooth_read_f<Type, bit_depth>(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
        const float b_br = msmooth_read_f<Type, bit_depth>(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
        const float b_bl = msmooth_read_f<Type, bit_depth>(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);

        int edge = (fabsf(b_cc - b_br) >= threshold) || (fabsf(b_cc - b_bl) >= threshold);

        if (highq) {
            const float b_bc = msmooth_read_f<Type, bit_depth>(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            const float b_cr = msmooth_read_f<Type, bit_depth>(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            edge = edge || (fabsf(b_cc - b_bc) >= threshold) || (fabsf(b_cc - b_cr) >= threshold);
        }

        // Boundary pixels are always treated as edge
        if (ix == 0 || ix >= dstWidth - 1 || iy == 0 || iy >= dstHeight - 1) {
            edge = 1;
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = edge ? (Type)((1 << bit_depth) - 1) : (Type)0;
    }
}

// Kernel 3: Masked Smoothing Pass (single iteration)
template<typename Type, int bit_depth>
__global__ void kernel_msmooth_smooth(uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *pSrc, const int srcPitch,
    const uint8_t *pMask, const int maskPitch,
    const int width, const int height) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        const int m_cc = msmooth_read_i<Type, bit_depth>(pMask, maskPitch, ix, iy, width, height);

        int result;
        if (m_cc != 0) {
            // Edge pixel: pass through
            result = msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix, iy, width, height);
        } else {
            // Non-edge pixel: average with non-edge neighbors
            int sum = msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix, iy, width, height);
            int count = 1;

            if (iy > 0 && msmooth_read_i<Type, bit_depth>(pMask, maskPitch, ix, iy - 1, width, height) == 0) {
                sum += msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix, iy - 1, width, height);
                count++;
            }
            if (iy < height - 1 && msmooth_read_i<Type, bit_depth>(pMask, maskPitch, ix, iy + 1, width, height) == 0) {
                sum += msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix, iy + 1, width, height);
                count++;
            }
            if (ix > 0 && msmooth_read_i<Type, bit_depth>(pMask, maskPitch, ix - 1, iy, width, height) == 0) {
                sum += msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy, width, height);
                count++;
            }
            if (ix < width - 1 && msmooth_read_i<Type, bit_depth>(pMask, maskPitch, ix + 1, iy, width, height) == 0) {
                sum += msmooth_read_i<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy, width, height);
                count++;
            }
            result = (sum + count / 2) / count;
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(result, 0, (1 << bit_depth) - 1);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR msmooth_blur_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(MSMOOTH_BLOCK_X, MSMOOTH_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_msmooth_blur<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0]);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR msmooth_edge_mask_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBlurFrame,
    float threshold, bool highq, cudaStream_t stream) {
    dim3 blockSize(MSMOOTH_BLOCK_X, MSMOOTH_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    // Normalize threshold from [0,255] to [0,1]
    const float th_norm = threshold / (float)((1 << bit_depth) - 1);

    kernel_msmooth_edge_mask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pBlurFrame->ptr[0], pBlurFrame->pitch[0],
        th_norm, highq ? 1 : 0);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int bit_depth>
static RGY_ERR msmooth_smooth_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pMaskFrame, cudaStream_t stream) {
    dim3 blockSize(MSMOOTH_BLOCK_X, MSMOOTH_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_msmooth_smooth<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        (const uint8_t *)pMaskFrame->ptr[0], pMaskFrame->pitch[0],
        pOutputFrame->width, pOutputFrame->height);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

NVEncFilterMsmooth::NVEncFilterMsmooth() : m_blur(), m_mask(), m_tmp() {
    m_name = _T("msmooth");
}

NVEncFilterMsmooth::~NVEncFilterMsmooth() {
    close();
}

RGY_ERR NVEncFilterMsmooth::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pMsmoothParam = std::dynamic_pointer_cast<NVEncFilterParamMsmooth>(pParam);
    if (!pMsmoothParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pMsmoothParam->frameOut.height <= 0 || pMsmoothParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pMsmoothParam->msmooth.strength < 0 || 20 < pMsmoothParam->msmooth.strength) {
        pMsmoothParam->msmooth.strength = clamp(pMsmoothParam->msmooth.strength, 0, 20);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %d - %d.\n"), 0, 20);
    }
    if (pMsmoothParam->msmooth.threshold < 0.0f || 255.0f < pMsmoothParam->msmooth.threshold) {
        pMsmoothParam->msmooth.threshold = clamp(pMsmoothParam->msmooth.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }

    sts = AllocFrameBuf(pMsmoothParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pMsmoothParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Allocate intermediate buffers per plane
    const int nPlanes = RGY_CSP_PLANES[pMsmoothParam->frameOut.csp];
    m_blur.resize(nPlanes);
    m_mask.resize(nPlanes);
    m_tmp[0].resize(nPlanes);
    m_tmp[1].resize(nPlanes);

    for (int ip = 0; ip < nPlanes; ip++) {
        auto plane = getPlane(&pMsmoothParam->frameOut, (RGY_PLANE)ip);

        m_blur[ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pMsmoothParam->frameOut.csp);
        m_blur[ip]->releasePtr();
        sts = m_blur[ip]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate blur buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }

        m_mask[ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pMsmoothParam->frameOut.csp);
        m_mask[ip]->releasePtr();
        sts = m_mask[ip]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate mask buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }

        for (int k = 0; k < 2; k++) {
            m_tmp[k][ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pMsmoothParam->frameOut.csp);
            m_tmp[k][ip]->releasePtr();
            sts = m_tmp[k][ip]->alloc();
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate tmp buffer: %s.\n"), get_err_mes(sts));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    setFilterInfo(pParam->print());
    m_param = pMsmoothParam;
    return sts;
}

tstring NVEncFilterParamMsmooth::print() const {
    return msmooth.print();
}

RGY_ERR NVEncFilterMsmooth::procPlaneBlur(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    static const std::map<RGY_CSP, decltype(msmooth_blur_plane<uint8_t, 8>)*> blur_list = {
        { RGY_CSP_YV12,      msmooth_blur_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   msmooth_blur_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    msmooth_blur_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, msmooth_blur_plane<uint16_t, 16> }
    };
    if (blur_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return blur_list.at(pInputFrame->csp)(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterMsmooth::procPlaneEdgeMask(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBlurFrame,
    float threshold, bool highq, cudaStream_t stream) {
    static const std::map<RGY_CSP, decltype(msmooth_edge_mask_plane<uint8_t, 8>)*> mask_list = {
        { RGY_CSP_YV12,      msmooth_edge_mask_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   msmooth_edge_mask_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    msmooth_edge_mask_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, msmooth_edge_mask_plane<uint16_t, 16> }
    };
    if (mask_list.count(pBlurFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pBlurFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return mask_list.at(pBlurFrame->csp)(pOutputFrame, pBlurFrame, threshold, highq, stream);
}

RGY_ERR NVEncFilterMsmooth::procPlaneSmooth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pMaskFrame, cudaStream_t stream) {
    static const std::map<RGY_CSP, decltype(msmooth_smooth_plane<uint8_t, 8>)*> smooth_list = {
        { RGY_CSP_YV12,      msmooth_smooth_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   msmooth_smooth_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    msmooth_smooth_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, msmooth_smooth_plane<uint16_t, 16> }
    };
    if (smooth_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return smooth_list.at(pInputFrame->csp)(pOutputFrame, pInputFrame, pMaskFrame, stream);
}

RGY_ERR NVEncFilterMsmooth::procPlane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int ip, int strength, float threshold, bool highq, cudaStream_t stream) {
    auto planeBlur = getPlane(&m_blur[ip]->frame, RGY_PLANE_Y);
    auto planeMask = getPlane(&m_mask[ip]->frame, RGY_PLANE_Y);

    // Step 1: blur src → blur buf
    auto err = procPlaneBlur(&planeBlur, pInputFrame, stream);
    if (err != RGY_ERR_NONE) return err;

    // Step 2: edge mask from blur buf
    err = procPlaneEdgeMask(&planeMask, &planeBlur, threshold, highq, stream);
    if (err != RGY_ERR_NONE) return err;

    // Step 3: iterative smoothing
    if (strength == 0) {
        // No smoothing: just copy src → dst
        return copyPlaneAsync(pOutputFrame, pInputFrame, stream);
    }

    for (int i = 0; i < strength; i++) {
        RGYFrameInfo planeIn, planeOut;
        if (i == 0) {
            planeIn = *pInputFrame;
        } else {
            planeIn = getPlane(&m_tmp[(i - 1) % 2][ip]->frame, RGY_PLANE_Y);
        }
        if (i == strength - 1) {
            planeOut = *pOutputFrame;
        } else {
            planeOut = getPlane(&m_tmp[i % 2][ip]->frame, RGY_PLANE_Y);
        }
        err = procPlaneSmooth(&planeOut, &planeIn, &planeMask, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMsmooth::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pMsmoothParam = std::dynamic_pointer_cast<NVEncFilterParamMsmooth>(m_param);
    if (!pMsmoothParam) {
        return RGY_ERR_INVALID_PARAM;
    }

    if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB) {
        const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)];
        for (int ip = 0; ip < nPlanes; ip++) {
            auto plane = (RGY_PLANE)ip;
            const auto planeInput = getPlane(pInputFrame, plane);
            auto planeOutput = getPlane(pOutputFrame, plane);
            auto err = procPlane(&planeOutput, &planeInput, ip,
                pMsmoothParam->msmooth.strength,
                pMsmoothParam->msmooth.threshold,
                pMsmoothParam->msmooth.highq,
                stream);
            if (err != RGY_ERR_NONE) return err;
        }
    } else {
        // YUV: process Y plane, copy UV
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);

        auto err = procPlane(&planeOutputY, &planeInputY, 0,
            pMsmoothParam->msmooth.strength,
            pMsmoothParam->msmooth.threshold,
            pMsmoothParam->msmooth.highq,
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

RGY_ERR NVEncFilterMsmooth::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        AddMessage(RGY_LOG_ERROR, _T("error at msmooth(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterMsmooth::close() {
    m_blur.clear();
    m_mask.clear();
    m_tmp[0].clear();
    m_tmp[1].clear();
    m_frameBuf.clear();
}
