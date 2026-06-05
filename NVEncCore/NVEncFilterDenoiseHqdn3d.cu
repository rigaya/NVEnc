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
#include <cmath>
#include <cstdint>
#include <vector>
#include "convert_csp.h"
#include "NVEncFilterDenoiseHqdn3d.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int HQDN3D_BLOCK_LINEAR = 32;
static const int HQDN3D_TBLOCK_X = 32;
static const int HQDN3D_TBLOCK_Y = 8;

template<typename Type, int bit_depth>
__device__ __inline__ float hqdn3d_read_pixel_f(const uint8_t *pSrc, int srcPitch, int x, int y) {
    const Type val = *(const Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

template<typename Type, int bit_depth>
__device__ __inline__ void hqdn3d_write_pixel_f(uint8_t *pDst, int dstPitch, int x, int y, float v) {
    Type *ptr = (Type *)(pDst + y * dstPitch + x * sizeof(Type));
    ptr[0] = (Type)(clamp(v, 0.0f, 1.0f) * (float)((1 << bit_depth) - 1) + 0.5f);
}

__device__ __inline__ float hqdn3d_lowpass(float prev, float cur, const float *__restrict__ coef) {
    const float delta_pix = (prev - cur) * 255.0f;
    int idx = (int)(delta_pix + (delta_pix >= 0.0f ? 0.5f : -0.5f)) + HQDN3D_LUT_RADIUS;
    idx = clamp(idx, 0, 2 * HQDN3D_LUT_RADIUS - 1);
    return cur + coef[idx];
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdn3d_h(float *__restrict__ pDst, const int dstPitchElems,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const float *__restrict__ coefSpatial) {
    const int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= height) return;

    float prev_pixel = hqdn3d_read_pixel_f<Type, bit_depth>(pSrc, srcPitch, 0, iy);
    for (int x = 0; x < width; ++x) {
        const float cur = hqdn3d_read_pixel_f<Type, bit_depth>(pSrc, srcPitch, x, iy);
        prev_pixel = hqdn3d_lowpass(prev_pixel, cur, coefSpatial);
        pDst[iy * dstPitchElems + x] = prev_pixel;
    }
}

__global__ void kernel_hqdn3d_v(float *__restrict__ pDst, const int dstPitchElems,
    const float *__restrict__ pSrc, const int srcPitchElems,
    const int width, const int height,
    const float *__restrict__ coefSpatial) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= width) return;

    float prev_pixel = pSrc[ix];
    for (int y = 0; y < height; ++y) {
        const float cur = pSrc[y * srcPitchElems + ix];
        prev_pixel = hqdn3d_lowpass(prev_pixel, cur, coefSpatial);
        pDst[y * dstPitchElems + ix] = prev_pixel;
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_hqdn3d_t(uint8_t *__restrict__ pDst, const int dstPitch,
    float *__restrict__ pFramePrev, const int prevPitchElems,
    const float *__restrict__ pSpatial, const int spatialPitchElems,
    const int width, const int height,
    const float *__restrict__ coefTemporal,
    const int first_frame) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int idxPrev = iy * prevPitchElems + ix;
    const int idxSpatial = iy * spatialPitchElems + ix;
    const float spatial = pSpatial[idxSpatial];
    float result;
    if (first_frame) {
        result = spatial;
    } else {
        const float prev = pFramePrev[idxPrev];
        result = hqdn3d_lowpass(prev, spatial, coefTemporal);
    }
    pFramePrev[idxPrev] = result;
    hqdn3d_write_pixel_f<Type, bit_depth>(pDst, dstPitch, ix, iy, result);
}

template<typename Type, int bit_depth>
static RGY_ERR hqdn3d_h_plane(RGYFrameInfo *pTmpH, const RGYFrameInfo *pInputFrame, const float *pCoefSpatial, cudaStream_t stream) {
    dim3 blockSize(HQDN3D_BLOCK_LINEAR, 1);
    dim3 gridSize(divCeil(pInputFrame->height, blockSize.x), 1);

    kernel_hqdn3d_h<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (float *)pTmpH->ptr[0], pTmpH->pitch[0],
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        pInputFrame->width, pInputFrame->height,
        pCoefSpatial);
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR hqdn3d_v_plane(RGYFrameInfo *pTmpHV, const RGYFrameInfo *pTmpH, const float *pCoefSpatial, cudaStream_t stream) {
    dim3 blockSize(HQDN3D_BLOCK_LINEAR, 1);
    dim3 gridSize(divCeil(pTmpH->width, blockSize.x), 1);

    kernel_hqdn3d_v<<<gridSize, blockSize, 0, stream>>>(
        (float *)pTmpHV->ptr[0], pTmpHV->pitch[0],
        (const float *)pTmpH->ptr[0], pTmpH->pitch[0],
        pTmpH->width, pTmpH->height,
        pCoefSpatial);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR hqdn3d_t_plane(RGYFrameInfo *pOutputFrame, RGYFrameInfo *pPrevFrame, const RGYFrameInfo *pTmpHV,
    const float *pCoefTemporal, bool firstFrame, cudaStream_t stream) {
    dim3 blockSize(HQDN3D_TBLOCK_X, HQDN3D_TBLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_hqdn3d_t<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
        (float *)pPrevFrame->ptr[0], pPrevFrame->pitch[0],
        (const float *)pTmpHV->ptr[0], pTmpHV->pitch[0],
        pOutputFrame->width, pOutputFrame->height,
        pCoefTemporal, firstFrame ? 1 : 0);
    return err_to_rgy(cudaGetLastError());
}

void NVEncFilterDenoiseHqdn3d::precalcCoefs(std::vector<float> &table, double dist25) {
    table.assign(2 * HQDN3D_LUT_RADIUS, 0.0f);
    if (dist25 <= 0.0) {
        return;
    }
    const double clamped = std::min(dist25, 253.9);
    const double sigma = -1.0 / std::log(0.25);
    const double scale = clamped * sigma + 1e-7;
    for (int i = 0; i < 2 * HQDN3D_LUT_RADIUS; ++i) {
        const double f = (double)(i - HQDN3D_LUT_RADIUS);
        const double attenuation = std::exp(-std::fabs(f) / scale);
        table[i] = (float)(attenuation * f / 256.0);
    }
}

NVEncFilterDenoiseHqdn3d::NVEncFilterDenoiseHqdn3d() :
    m_coefs(),
    m_framePrev(),
    m_framePrevPitchFloats(),
    m_tmpH(),
    m_tmpHV(),
    m_tmpPitchFloats(0),
    m_firstFrame(true) {
    m_name = _T("denoise-hqdn3d");
}

NVEncFilterDenoiseHqdn3d::~NVEncFilterDenoiseHqdn3d() {
    close();
}

RGY_ERR NVEncFilterDenoiseHqdn3d::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pHqdn3dParam = std::dynamic_pointer_cast<NVEncFilterParamDenoiseHqdn3d>(pParam);
    if (!pHqdn3dParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pHqdn3dParam->frameOut.height <= 0 || pHqdn3dParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto clampStrength = [&](float &v, const TCHAR *name) {
        if (v < 0.0f || v > 255.0f) {
            AddMessage(RGY_LOG_WARN, _T("%s must be in range 0.0 - 255.0.\n"), name);
            v = clamp(v, 0.0f, 255.0f);
        }
    };
    clampStrength(pHqdn3dParam->hqdn3d.luma_spatial,    _T("luma_spatial"));
    clampStrength(pHqdn3dParam->hqdn3d.chroma_spatial,  _T("chroma_spatial"));
    clampStrength(pHqdn3dParam->hqdn3d.luma_temporal,   _T("luma_temporal"));
    clampStrength(pHqdn3dParam->hqdn3d.chroma_temporal, _T("chroma_temporal"));

    const float strengths[4] = {
        pHqdn3dParam->hqdn3d.luma_spatial,
        pHqdn3dParam->hqdn3d.luma_temporal,
        pHqdn3dParam->hqdn3d.chroma_spatial,
        pHqdn3dParam->hqdn3d.chroma_temporal
    };
    for (int i = 0; i < 4; ++i) {
        std::vector<float> table;
        precalcCoefs(table, (double)strengths[i]);
        const auto bytes = table.size() * sizeof(table[0]);
        m_coefs[i] = std::make_unique<CUMemBuf>(bytes);
        sts = m_coefs[i]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate coefficient buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
        auto cudaerr = cudaMemcpy(m_coefs[i]->ptr, table.data(), bytes, cudaMemcpyHostToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy coefficient buffer: %s.\n"), char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }

    sts = AllocFrameBuf(pHqdn3dParam->frameOut, 2);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pHqdn3dParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pHqdn3dParam->frameOut.csp)];
    m_framePrev.resize(nPlanes);
    m_framePrevPitchFloats.resize(nPlanes);
    for (int i = 0; i < nPlanes; ++i) {
        auto planeInfo = getPlane(&m_frameBuf[0]->frame, (RGY_PLANE)i);
        m_framePrevPitchFloats[i] = planeInfo.width;
        m_framePrev[i] = std::make_unique<CUMemBuf>((size_t)planeInfo.width * planeInfo.height * sizeof(float));
        sts = m_framePrev[i]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate prev buffer plane %d: %s.\n"), i, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    {
        auto lumaInfo = getPlane(&m_frameBuf[0]->frame, RGY_PLANE_Y);
        m_tmpPitchFloats = lumaInfo.width;
        const size_t tmpBytes = (size_t)lumaInfo.width * lumaInfo.height * sizeof(float);
        m_tmpH = std::make_unique<CUMemBuf>(tmpBytes);
        sts = m_tmpH->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate hqdn3d tmpH buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_tmpHV = std::make_unique<CUMemBuf>(tmpBytes);
        sts = m_tmpHV->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate hqdn3d tmpHV buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    m_firstFrame = true;

    setFilterInfo(pParam->print());
    m_param = pHqdn3dParam;
    return sts;
}

tstring NVEncFilterParamDenoiseHqdn3d::print() const {
    return hqdn3d.print();
}

RGY_ERR NVEncFilterDenoiseHqdn3d::denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    CUMemBuf *pCoefSpatial, CUMemBuf *pCoefTemporal,
    CUMemBuf *pPrev, int prevPitchFloats, cudaStream_t stream) {
    RGYFrameInfo tmpH = *pInputPlane;
    tmpH.ptr[0] = (uint8_t *)m_tmpH->ptr;
    tmpH.pitch[0] = m_tmpPitchFloats;
    RGYFrameInfo tmpHV = *pInputPlane;
    tmpHV.ptr[0] = (uint8_t *)m_tmpHV->ptr;
    tmpHV.pitch[0] = m_tmpPitchFloats;
    RGYFrameInfo prev = *pInputPlane;
    prev.ptr[0] = (uint8_t *)pPrev->ptr;
    prev.pitch[0] = prevPitchFloats;

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputPlane->csp];
    RGY_ERR err = RGY_ERR_NONE;
    switch (bitDepth) {
    case 8:  err = hqdn3d_h_plane<uint8_t,   8>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    case 9:  err = hqdn3d_h_plane<uint16_t,  9>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    case 10: err = hqdn3d_h_plane<uint16_t, 10>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    case 12: err = hqdn3d_h_plane<uint16_t, 12>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    case 14: err = hqdn3d_h_plane<uint16_t, 14>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    case 16: err = hqdn3d_h_plane<uint16_t, 16>(&tmpH, pInputPlane, (const float *)pCoefSpatial->ptr, stream); break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth: %d.\n"), bitDepth);
        return RGY_ERR_UNSUPPORTED;
    }
    if (err != RGY_ERR_NONE) return err;

    err = hqdn3d_v_plane(&tmpHV, &tmpH, (const float *)pCoefSpatial->ptr, stream);
    if (err != RGY_ERR_NONE) return err;

    switch (bitDepth) {
    case 8:  err = hqdn3d_t_plane<uint8_t,   8>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    case 9:  err = hqdn3d_t_plane<uint16_t,  9>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    case 10: err = hqdn3d_t_plane<uint16_t, 10>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    case 12: err = hqdn3d_t_plane<uint16_t, 12>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    case 14: err = hqdn3d_t_plane<uint16_t, 14>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    case 16: err = hqdn3d_t_plane<uint16_t, 16>(pOutputPlane, &prev, &tmpHV, (const float *)pCoefTemporal->ptr, m_firstFrame, stream); break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    return err;
}

RGY_ERR NVEncFilterDenoiseHqdn3d::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pOutputFrame->csp)];
    for (int i = 0; i < nPlanes; ++i) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,  (RGY_PLANE)i);
        const bool isChroma = (i > 0);
        auto err = denoisePlane(&planeDst, &planeSrc,
            m_coefs[isChroma ? 2 : 0].get(),
            m_coefs[isChroma ? 3 : 1].get(),
            m_framePrev[i].get(), m_framePrevPitchFloats[i], stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(hqdn3d) plane %d: %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterDenoiseHqdn3d::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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

    sts = denoiseFrame(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdn3d(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    m_firstFrame = false;
    return sts;
}

void NVEncFilterDenoiseHqdn3d::close() {
    for (auto &c : m_coefs) {
        c.reset();
    }
    m_framePrev.clear();
    m_framePrevPitchFloats.clear();
    m_tmpH.reset();
    m_tmpHV.reset();
    m_tmpPitchFloats = 0;
    m_frameBuf.clear();
    m_firstFrame = true;
}
