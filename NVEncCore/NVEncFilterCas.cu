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
#include "convert_csp.h"
#include "NVEncFilterCas.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int CAS_BLOCK_X = 32;
static const int CAS_BLOCK_Y = 8;

template<typename Type, int bit_depth>
__device__ static inline float cas_read(const uint8_t *pSrc, int srcPitch,
    int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const Type val = *(const Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

__device__ static inline float cas_linearise(float x, int apply) {
    return apply ? (x * x) : x;
}

__device__ static inline float cas_delinearise(float x, int apply) {
    return apply ? sqrtf(x) : x;
}

template<typename Type, int bit_depth>
__global__ void kernel_cas(uint8_t *__restrict__ pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const float peak, const int apply_gamma2) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }

    float b = cas_read<Type, bit_depth>(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
    float d = cas_read<Type, bit_depth>(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
    float e = cas_read<Type, bit_depth>(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
    float f = cas_read<Type, bit_depth>(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
    float h = cas_read<Type, bit_depth>(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);

    b = cas_linearise(b, apply_gamma2);
    d = cas_linearise(d, apply_gamma2);
    e = cas_linearise(e, apply_gamma2);
    f = cas_linearise(f, apply_gamma2);
    h = cas_linearise(h, apply_gamma2);

    const float mn = fminf(fminf(fminf(fminf(b, d), e), f), h);
    const float mx = fmaxf(fmaxf(fmaxf(fmaxf(b, d), e), f), h);
    float amp = clamp(fminf(mn, 1.0f - mx) / fmaxf(mx, RGY_FLT_EPS), 0.0f, 1.0f);
    amp = sqrtf(amp);

    const float w = amp * peak;
    float result = ((b + d + f + h) * w + e) / (1.0f + 4.0f * w);
    result = clamp(result, 0.0f, 1.0f);
    result = cas_delinearise(result, apply_gamma2);
    result = clamp(result, 0.0f, 1.0f - RGY_FLT_EPS);

    Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
static RGY_ERR cas_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const float peak, const int apply_gamma2, cudaStream_t stream) {
    dim3 blockSize(CAS_BLOCK_X, CAS_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));
    kernel_cas<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        pInputPlane->ptr[0], pInputPlane->pitch[0],
        peak, apply_gamma2);
    return err_to_rgy(cudaGetLastError());
}

NVEncFilterCas::NVEncFilterCas() : NVEncFilter(), m_nFrameIdx(0) {
    m_name = _T("cas");
}

NVEncFilterCas::~NVEncFilterCas() {
    close();
}

RGY_ERR NVEncFilterCas::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamCas>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp];
    const bool supportedCsp = chromaFormat == RGY_CHROMAFMT_MONOCHROME
        || (RGY_CSP_PLANES[prm->frameOut.csp] > 1
            && (chromaFormat == RGY_CHROMAFMT_YUV420 || chromaFormat == RGY_CHROMAFMT_YUV422 || chromaFormat == RGY_CHROMAFMT_YUV444));
    if (!supportedCsp) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for cas: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->cas.sharpness < 0.0f || 1.0f < prm->cas.sharpness) {
        prm->cas.sharpness = clamp(prm->cas.sharpness, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("sharpness should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }

    sts = AllocFrameBuf(prm->frameOut, 2);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterCas::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamCas>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const float sharp = std::min(std::max(prm->cas.sharpness, 0.0f), 1.0f);
    const float lerp = 8.0f + (5.0f - 8.0f) * sharp;
    const float peak = -1.0f / lerp;
    const int apply_gamma2 = prm->cas.hdr ? 0 : 1;

    auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    RGY_ERR err = RGY_ERR_UNSUPPORTED;
    switch (RGY_CSP_DATA_TYPE[pInputFrame->csp]) {
    case RGY_DATA_TYPE_U8:
        err = cas_plane<uint8_t, 8>(&planeDstY, &planeSrcY, peak, apply_gamma2, stream);
        break;
    case RGY_DATA_TYPE_U16:
        err = cas_plane<uint16_t, 16>(&planeDstY, &planeSrcY, peak, apply_gamma2, stream);
        break;
    default:
        return RGY_ERR_UNSUPPORTED;
    }
    if (err != RGY_ERR_NONE) {
        return err;
    }
    const int nPlanes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int i = 1; i < nPlanes; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        err = copyPlaneAsync(&planeDst, &planeSrc, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("cas chroma copy (plane %d) failed: %s.\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterCas::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto sts = procFrame(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterCas::close() {
    m_frameBuf.clear();
    m_nFrameIdx = 0;
}
