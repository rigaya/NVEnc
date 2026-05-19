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

#include <algorithm>
#include <map>
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterDetailSharpen.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DETAILSHARPEN_BLOCK_X = 32;
static const int DETAILSHARPEN_BLOCK_Y = 8;

template<typename Type>
__device__ __inline__ float detailsharpen_read(const uint8_t *pSrc, const int pitch, int x, int y, const int width, const int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (float)*(const Type *)(pSrc + y * pitch + x * sizeof(Type));
}

__device__ __inline__ float detailsharpen_median9(float v[9]) {
    #pragma unroll
    for (int i = 1; i < 9; i++) {
        const float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    return v[4];
}

template<typename Type, int mode>
__global__ void kernel_detailsharpen_blur(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const float p00 = detailsharpen_read<Type>(pSrc, srcPitch, ix - 1, iy - 1, dstWidth, dstHeight);
        const float p01 = detailsharpen_read<Type>(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        const float p02 = detailsharpen_read<Type>(pSrc, srcPitch, ix + 1, iy - 1, dstWidth, dstHeight);
        const float p10 = detailsharpen_read<Type>(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        const float p11 = detailsharpen_read<Type>(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        const float p12 = detailsharpen_read<Type>(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        const float p20 = detailsharpen_read<Type>(pSrc, srcPitch, ix - 1, iy + 1, dstWidth, dstHeight);
        const float p21 = detailsharpen_read<Type>(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);
        const float p22 = detailsharpen_read<Type>(pSrc, srcPitch, ix + 1, iy + 1, dstWidth, dstHeight);

        float blur = 0.0f;
        if (mode == 0) {
            blur = (p00 + p02 + p20 + p22 + 2.0f * (p01 + p10 + p12 + p21) + 4.0f * p11) * (1.0f / 16.0f);
        } else {
            blur = (p00 + p01 + p02 + p10 + p11 + p12 + p20 + p21 + p22) * (1.0f / 9.0f);
        }
        *(Type *)(pDst + iy * dstPitch + ix * sizeof(Type)) = (Type)(blur + 0.5f);
    }
}

template<typename Type, int med>
__global__ void kernel_detailsharpen_apply(
    uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pBlur, const int blurPitch,
    const float z, const float invPower, const float ldmp, const float strengthScaled, const float invI, const float maxv) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const float xv = detailsharpen_read<Type>(pSrc, srcPitch, ix, iy, dstWidth, dstHeight);
        float yv = 0.0f;
        if (med) {
            float v[9];
            v[0] = detailsharpen_read<Type>(pBlur, blurPitch, ix - 1, iy - 1, dstWidth, dstHeight);
            v[1] = detailsharpen_read<Type>(pBlur, blurPitch, ix,     iy - 1, dstWidth, dstHeight);
            v[2] = detailsharpen_read<Type>(pBlur, blurPitch, ix + 1, iy - 1, dstWidth, dstHeight);
            v[3] = detailsharpen_read<Type>(pBlur, blurPitch, ix - 1, iy,     dstWidth, dstHeight);
            v[4] = detailsharpen_read<Type>(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
            v[5] = detailsharpen_read<Type>(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            v[6] = detailsharpen_read<Type>(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);
            v[7] = detailsharpen_read<Type>(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            v[8] = detailsharpen_read<Type>(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
            yv = detailsharpen_median9(v);
        } else {
            yv = detailsharpen_read<Type>(pBlur, blurPitch, ix, iy, dstWidth, dstHeight);
        }

        float out = xv;
        if (xv != yv) {
            const float diff = (xv - yv) * invI;
            const float absd = fabsf(diff);
            const float amp = __powf(absd / z, invPower);
            out = xv + strengthScaled * amp * diff / (absd + ldmp);
        }
        *(Type *)(pDst + iy * dstPitch + ix * sizeof(Type)) = (Type)(clamp(out, 0.0f, maxv) + 0.5f);
    }
}

template<typename Type, int mode>
static RGY_ERR detailsharpen_blur_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(DETAILSHARPEN_BLOCK_X, DETAILSHARPEN_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_detailsharpen_blur<Type, mode><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0]);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type, int med>
static RGY_ERR detailsharpen_apply_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pBlurFrame,
    const int bitDepth, const float z, const float sstr, const float power, const float ldmp, cudaStream_t stream) {
    dim3 blockSize(DETAILSHARPEN_BLOCK_X, DETAILSHARPEN_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    const int shift = clamp(bitDepth, 8, 16) - 8;
    const float i = (float)(1 << shift);
    const float invI = 1.0f / i;
    const float maxv = (float)((1 << clamp(bitDepth, 8, 16)) - 1);
    kernel_detailsharpen_apply<Type, med><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        (const uint8_t *)pBlurFrame->ptr[0], pBlurFrame->pitch[0],
        z, 1.0f / power, ldmp, sstr * z * i, invI, maxv);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

template<typename Type>
static RGY_ERR detailsharpen_plane(RGYFrameInfo *pOutputFrame, RGYFrameInfo *pBlurFrame, const RGYFrameInfo *pInputFrame,
    const int bitDepth, const VppDetailSharpen& prm, cudaStream_t stream) {
    RGY_ERR err = RGY_ERR_NONE;
    if (prm.mode == 0) {
        err = detailsharpen_blur_plane<Type, 0>(pBlurFrame, pInputFrame, stream);
    } else {
        err = detailsharpen_blur_plane<Type, 1>(pBlurFrame, pInputFrame, stream);
    }
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return prm.med
        ? detailsharpen_apply_plane<Type, 1>(pOutputFrame, pInputFrame, pBlurFrame, bitDepth, prm.z, prm.sstr, prm.power, prm.ldmp, stream)
        : detailsharpen_apply_plane<Type, 0>(pOutputFrame, pInputFrame, pBlurFrame, bitDepth, prm.z, prm.sstr, prm.power, prm.ldmp, stream);
}

NVEncFilterDetailSharpen::NVEncFilterDetailSharpen() : m_blur() {
    m_name = _T("detailsharpen");
}

NVEncFilterDetailSharpen::~NVEncFilterDetailSharpen() {
    close();
}

RGY_ERR NVEncFilterDetailSharpen::checkParam(const std::shared_ptr<NVEncFilterParamDetailSharpen> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_RGB) {
        AddMessage(RGY_LOG_ERROR, _T("detailsharpen is not supported on RGB csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->detailsharpen.z < 0.001f || 64.0f < prm->detailsharpen.z) {
        prm->detailsharpen.z = clamp(prm->detailsharpen.z, 0.001f, 64.0f);
        AddMessage(RGY_LOG_WARN, _T("z should be in range of %.3f - %.1f.\n"), 0.001f, 64.0f);
    }
    if (prm->detailsharpen.sstr < 0.0f || 16.0f < prm->detailsharpen.sstr) {
        prm->detailsharpen.sstr = clamp(prm->detailsharpen.sstr, 0.0f, 16.0f);
        AddMessage(RGY_LOG_WARN, _T("sstr should be in range of %.1f - %.1f.\n"), 0.0f, 16.0f);
    }
    if (prm->detailsharpen.power < 1.0f || 16.0f < prm->detailsharpen.power) {
        prm->detailsharpen.power = clamp(prm->detailsharpen.power, 1.0f, 16.0f);
        AddMessage(RGY_LOG_WARN, _T("power should be in range of %.1f - %.1f.\n"), 1.0f, 16.0f);
    }
    if (prm->detailsharpen.ldmp < 0.0f || 1000.0f < prm->detailsharpen.ldmp) {
        prm->detailsharpen.ldmp = clamp(prm->detailsharpen.ldmp, 0.0f, 1000.0f);
        AddMessage(RGY_LOG_WARN, _T("ldmp should be in range of %.1f - %.1f.\n"), 0.0f, 1000.0f);
    }
    if (prm->detailsharpen.mode < 0 || 1 < prm->detailsharpen.mode) {
        AddMessage(RGY_LOG_ERROR, _T("mode should be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDetailSharpen::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDetailSharpen>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
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

    auto planeY = getPlane(&prm->frameOut, RGY_PLANE_Y);
    const auto blurCsp = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
    if (!m_blur || m_blur->frame.width != planeY.width || m_blur->frame.height != planeY.height || m_blur->frame.csp != blurCsp) {
        m_blur = std::make_unique<CUFrameBuf>(planeY.width, planeY.height, blurCsp);
        m_blur->releasePtr();
        sts = m_blur->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate blur buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(pParam->print());
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamDetailSharpen::print() const {
    return detailsharpen.print();
}

RGY_ERR NVEncFilterDetailSharpen::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDetailSharpen>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }

    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeBlurY = getPlane(&m_blur->frame, RGY_PLANE_Y);
    const auto bitDepth = pInputFrame->bitdepth > 0 ? pInputFrame->bitdepth : RGY_CSP_BIT_DEPTH[pInputFrame->csp];

    static const std::map<RGY_DATA_TYPE, decltype(detailsharpen_plane<uint8_t>)*> detailsharpen_list = {
        { RGY_DATA_TYPE_U8,  detailsharpen_plane<uint8_t> },
        { RGY_DATA_TYPE_U16, detailsharpen_plane<uint16_t> }
    };
    if (detailsharpen_list.count(RGY_CSP_DATA_TYPE[pInputFrame->csp]) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    auto err = detailsharpen_list.at(RGY_CSP_DATA_TYPE[pInputFrame->csp])(&planeOutputY, &planeBlurY, &planeInputY, bitDepth, prm->detailsharpen, stream);
    if (err != RGY_ERR_NONE) return err;

    const int copyPlanes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    for (int iplane = 1; iplane < copyPlanes; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        const auto planeInput = getPlane(pInputFrame, plane);
        auto planeOutput = getPlane(pOutputFrame, plane);
        err = copyPlaneAsync(&planeOutput, &planeInput, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    err = copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
    if (err != RGY_ERR_NONE) return err;

    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDetailSharpen::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        AddMessage(RGY_LOG_ERROR, _T("error at detailsharpen(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp],
            get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterDetailSharpen::close() {
    m_blur.reset();
    m_frameBuf.clear();
}
