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

#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterVinverse.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int VINVERSE_BLOCK_X = 32;
static const int VINVERSE_BLOCK_Y = 8;
static const int VINVERSE_VBLUR_BLOCK_X = 8;
static const int VINVERSE_VBLUR_BLOCK_Y = 32;

__device__ __inline__ int vinverse_abs_i(const int x) {
    return (x < 0) ? -x : x;
}

__device__ __inline__ int vinverse_vmirror_p1(const int y, const int height) {
    if (height <= 1) return 0;
    return (y == 0) ? 1 : (y - 1);
}

__device__ __inline__ int vinverse_vmirror_n1(const int y, const int height) {
    if (height <= 1) return 0;
    return (y == height - 1) ? (height - 2) : (y + 1);
}

__device__ __inline__ int vinverse_vmirror_p2(const int y, const int height) {
    if (height <= 1) return 0;
    const int yy = (y < 2) ? (y + 2) : (y - 2);
    return (height < 4) ? clamp(yy, 0, height - 1) : yy;
}

__device__ __inline__ int vinverse_vmirror_n2(const int y, const int height) {
    if (height <= 1) return 0;
    const int yy = (y > height - 3) ? (y - 2) : (y + 2);
    return (height < 4) ? clamp(yy, 0, height - 1) : yy;
}

template<typename Type>
__device__ __inline__ int vinverse_read_i(const uint8_t *pSrc, const int pitch, const int x, const int y) {
    return (int)(*(const Type *)(pSrc + y * pitch + x * sizeof(Type)));
}

template<typename Type, int bit_depth>
__device__ __inline__ int vinverse_vblur3_at(const uint8_t *pSrc, const int srcPitch, const int ix, const int row, const int height) {
    const int yp = vinverse_vmirror_p1(row, height);
    const int yn = vinverse_vmirror_n1(row, height);
    const int a = vinverse_read_i<Type>(pSrc, srcPitch, ix, yp);
    const int b = vinverse_read_i<Type>(pSrc, srcPitch, ix, row);
    const int c = vinverse_read_i<Type>(pSrc, srcPitch, ix, yn);
    return (a + (b << 1) + c + 2) >> 2;
}

template<typename Type, int bit_depth>
__global__ void kernel_vinverse_vblur3(uint8_t *__restrict__ pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const int v = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, iy, dstHeight);
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(v, 0, (1 << bit_depth) - 1);
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_vinverse_vblur35(uint8_t *__restrict__ pPb3, const int pb3Pitch,
    uint8_t *__restrict__ pPb6, const int pb6Pitch,
    const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }

    const int r_pp = vinverse_vmirror_p2(iy, dstHeight);
    const int r_p  = vinverse_vmirror_p1(iy, dstHeight);
    const int r_n  = vinverse_vmirror_n1(iy, dstHeight);
    const int r_nn = vinverse_vmirror_n2(iy, dstHeight);

    const int b3_pp = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, r_pp, dstHeight);
    const int b3_p  = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, r_p,  dstHeight);
    const int b3_c  = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, iy,   dstHeight);
    const int b3_n  = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, r_n,  dstHeight);
    const int b3_nn = vinverse_vblur3_at<Type, bit_depth>(pSrc, srcPitch, ix, r_nn, dstHeight);
    const int v5 = (b3_pp + ((b3_p + b3_n) << 2) + b3_c * 6 + b3_nn + 8) >> 4;

    Type *p3 = (Type *)(pPb3 + iy * pb3Pitch + ix * sizeof(Type));
    Type *p6 = (Type *)(pPb6 + iy * pb6Pitch + ix * sizeof(Type));
    p3[0] = (Type)clamp(b3_c, 0, (1 << bit_depth) - 1);
    p6[0] = (Type)clamp(v5, 0, (1 << bit_depth) - 1);
}

template<typename Type, int bit_depth>
__global__ void kernel_vinverse_makediff(uint8_t *__restrict__ pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pC1, const int c1Pitch,
    const uint8_t *__restrict__ pC2, const int c2Pitch,
    const int h_offset) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const int c1 = vinverse_read_i<Type>(pC1, c1Pitch, ix, iy);
        const int c2 = vinverse_read_i<Type>(pC2, c2Pitch, ix, iy);
        const int v = clamp(c1 - c2 + h_offset, 0, (1 << bit_depth) - 1);
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)v;
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_vinverse_sbr_combine(uint8_t *__restrict__ pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pDiff, const int diffPitch,
    const uint8_t *__restrict__ pBlur, const int blurPitch,
    const int h_offset) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const int s    = vinverse_read_i<Type>(pSrc,  srcPitch,  ix, iy);
        const int diff = vinverse_read_i<Type>(pDiff, diffPitch, ix, iy);
        const int blur = vinverse_read_i<Type>(pBlur, blurPitch, ix, iy);
        const int t    = diff - blur;
        const int t2   = diff - h_offset;

        int v;
        if ((t < 0 && t2 > 0) || (t > 0 && t2 < 0)) {
            v = s;
        } else if (vinverse_abs_i(t) < vinverse_abs_i(t2)) {
            v = s - t;
        } else {
            v = s - (diff - h_offset);
        }
        v = clamp(v, 0, (1 << bit_depth) - 1);
        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)v;
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_vinverse_finalize(uint8_t *__restrict__ pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pPb3, const int pb3Pitch,
    const uint8_t *__restrict__ pPb6, const int pb6Pitch,
    const float sstr, const float scl,
    const int thr_hbd, const int amnt_hbd) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < dstWidth && iy < dstHeight) {
        const int s   = vinverse_read_i<Type>(pSrc, srcPitch, ix, iy);
        const int pb3 = vinverse_read_i<Type>(pPb3, pb3Pitch, ix, iy);
        const int d1  = s - pb3;

        int result;
        if (thr_hbd > 0 && vinverse_abs_i(d1) < thr_hbd) {
            result = s;
        } else {
            const int pb6 = vinverse_read_i<Type>(pPb6, pb6Pitch, ix, iy);
            const float d1f = (float)d1;
            const float t = (float)(pb3 - pb6) * sstr;
            const float da = (fabsf(d1f) < fabsf(t)) ? d1f : t;
            const float add = ((d1f * t) < 0.0f) ? (da * scl) : da;
            const int df = pb3 + (int)add;
            result = clamp(df, s - amnt_hbd, s + amnt_hbd);
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(result, 0, (1 << bit_depth) - 1);
    }
}

template<typename Type, int bit_depth>
static RGY_ERR vinverse_vblur3_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(VINVERSE_VBLUR_BLOCK_X, VINVERSE_VBLUR_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_vinverse_vblur3<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0]);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR vinverse_vblur35_plane(RGYFrameInfo *pPb3Frame, RGYFrameInfo *pPb6Frame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    dim3 blockSize(VINVERSE_VBLUR_BLOCK_X, VINVERSE_VBLUR_BLOCK_Y);
    dim3 gridSize(divCeil(pPb3Frame->width, blockSize.x), divCeil(pPb3Frame->height, blockSize.y));

    kernel_vinverse_vblur35<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pPb3Frame->ptr[0], pPb3Frame->pitch[0],
        (uint8_t *)pPb6Frame->ptr[0], pPb6Frame->pitch[0],
        pPb3Frame->width, pPb3Frame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0]);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR vinverse_makediff_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pC1Frame, const RGYFrameInfo *pC2Frame,
    int h_offset, cudaStream_t stream) {
    dim3 blockSize(VINVERSE_BLOCK_X, VINVERSE_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_vinverse_makediff<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pC1Frame->ptr[0], pC1Frame->pitch[0],
        (const uint8_t *)pC2Frame->ptr[0], pC2Frame->pitch[0],
        h_offset);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR vinverse_sbr_combine_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSrcFrame,
    const RGYFrameInfo *pDiffFrame, const RGYFrameInfo *pBlurFrame, int h_offset, cudaStream_t stream) {
    dim3 blockSize(VINVERSE_BLOCK_X, VINVERSE_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_vinverse_sbr_combine<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pSrcFrame->ptr[0], pSrcFrame->pitch[0],
        (const uint8_t *)pDiffFrame->ptr[0], pDiffFrame->pitch[0],
        (const uint8_t *)pBlurFrame->ptr[0], pBlurFrame->pitch[0],
        h_offset);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR vinverse_finalize_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pPb3Frame, const RGYFrameInfo *pPb6Frame,
    float sstr, float scl, int thr_hbd, int amnt_hbd, cudaStream_t stream) {
    dim3 blockSize(VINVERSE_BLOCK_X, VINVERSE_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputFrame->width, blockSize.x), divCeil(pOutputFrame->height, blockSize.y));

    kernel_vinverse_finalize<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        (const uint8_t *)pPb3Frame->ptr[0], pPb3Frame->pitch[0],
        (const uint8_t *)pPb6Frame->ptr[0], pPb6Frame->pitch[0],
        sstr, scl, thr_hbd, amnt_hbd);
    return err_to_rgy(cudaGetLastError());
}

#define VINVERSE_DISPATCH(func, ...) \
    switch (RGY_CSP_BIT_DEPTH[pInputPlane->csp]) { \
    case 8:  return func<uint8_t,   8>(__VA_ARGS__); \
    case 9:  return func<uint16_t,  9>(__VA_ARGS__); \
    case 10: return func<uint16_t, 10>(__VA_ARGS__); \
    case 12: return func<uint16_t, 12>(__VA_ARGS__); \
    case 14: return func<uint16_t, 14>(__VA_ARGS__); \
    case 16: return func<uint16_t, 16>(__VA_ARGS__); \
    default: break; \
    } \
    AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth: %d.\n"), RGY_CSP_BIT_DEPTH[pInputPlane->csp]); \
    return RGY_ERR_UNSUPPORTED

NVEncFilterVinverse::NVEncFilterVinverse() : m_pb3(), m_pb6() {
    m_name = _T("vinverse");
}

NVEncFilterVinverse::~NVEncFilterVinverse() {
    close();
}

RGY_ERR NVEncFilterVinverse::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pVinverseParam = std::dynamic_pointer_cast<NVEncFilterParamVinverse>(pParam);
    if (!pVinverseParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pVinverseParam->frameOut.height <= 0 || pVinverseParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pVinverseParam->vinverse.sstr < 0.0f || 8.0f < pVinverseParam->vinverse.sstr) {
        pVinverseParam->vinverse.sstr = clamp(pVinverseParam->vinverse.sstr, 0.0f, 8.0f);
        AddMessage(RGY_LOG_WARN, _T("sstr should be in range of %.1f - %.1f.\n"), 0.0f, 8.0f);
    }
    if (pVinverseParam->vinverse.amnt < 0.0f || 255.0f < pVinverseParam->vinverse.amnt) {
        pVinverseParam->vinverse.amnt = clamp(pVinverseParam->vinverse.amnt, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("amnt should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (pVinverseParam->vinverse.scl < 0.0f || 4.0f < pVinverseParam->vinverse.scl) {
        pVinverseParam->vinverse.scl = clamp(pVinverseParam->vinverse.scl, 0.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("scl should be in range of %.1f - %.1f.\n"), 0.0f, 4.0f);
    }
    if (pVinverseParam->vinverse.thr < 0.0f || 255.0f < pVinverseParam->vinverse.thr) {
        pVinverseParam->vinverse.thr = clamp(pVinverseParam->vinverse.thr, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("thr should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }

    sts = AllocFrameBuf(pVinverseParam->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pVinverseParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pVinverseParam->frameOut.csp)];
    m_pb3.resize(nPlanes);
    m_pb6.resize(nPlanes);
    for (int ip = 0; ip < nPlanes; ip++) {
        auto plane = getPlane(&pVinverseParam->frameOut, (RGY_PLANE)ip);
        m_pb3[ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pVinverseParam->frameOut.csp);
        m_pb3[ip]->releasePtr();
        sts = m_pb3[ip]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate pb3 buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }

        m_pb6[ip] = std::make_unique<CUFrameBuf>(plane.width, plane.height, pVinverseParam->frameOut.csp);
        m_pb6[ip]->releasePtr();
        sts = m_pb6[ip]->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate pb6 buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(pParam->print());
    m_param = pVinverseParam;
    return sts;
}

tstring NVEncFilterParamVinverse::print() const {
    return vinverse.print();
}

RGY_ERR NVEncFilterVinverse::procPlaneVblur3(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, cudaStream_t stream) {
    VINVERSE_DISPATCH(vinverse_vblur3_plane, pOutputPlane, pInputPlane, stream);
}

RGY_ERR NVEncFilterVinverse::procPlaneVblur35(RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane, const RGYFrameInfo *pInputPlane, cudaStream_t stream) {
    VINVERSE_DISPATCH(vinverse_vblur35_plane, pPb3Plane, pPb6Plane, pInputPlane, stream);
}

RGY_ERR NVEncFilterVinverse::procPlaneMakediff(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pC1Plane, const RGYFrameInfo *pC2Plane, int h_offset, cudaStream_t stream) {
    const auto pInputPlane = pC1Plane;
    VINVERSE_DISPATCH(vinverse_makediff_plane, pOutputPlane, pC1Plane, pC2Plane, h_offset, stream);
}

RGY_ERR NVEncFilterVinverse::procPlaneSbrCombine(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pSrcPlane, const RGYFrameInfo *pDiffPlane, const RGYFrameInfo *pBlurPlane, int h_offset, cudaStream_t stream) {
    const auto pInputPlane = pSrcPlane;
    VINVERSE_DISPATCH(vinverse_sbr_combine_plane, pOutputPlane, pSrcPlane, pDiffPlane, pBlurPlane, h_offset, stream);
}

RGY_ERR NVEncFilterVinverse::procPlaneFinalize(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pPb3Plane, const RGYFrameInfo *pPb6Plane,
    float sstr, float scl, int thr_hbd, int amnt_hbd, cudaStream_t stream) {
    VINVERSE_DISPATCH(vinverse_finalize_plane, pOutputPlane, pInputPlane, pPb3Plane, pPb6Plane, sstr, scl, thr_hbd, amnt_hbd, stream);
}

#undef VINVERSE_DISPATCH

RGY_ERR NVEncFilterVinverse::procPlane(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane,
    VppVinverseMode mode, float sstr, float scl, int thr_hbd, int amnt_hbd, int h_offset, cudaStream_t stream) {
    RGY_ERR err = RGY_ERR_NONE;
    if (mode == VppVinverseMode::Vinverse) {
        err = procPlaneVblur35(pPb3Plane, pPb6Plane, pInputPlane, stream);
        if (err != RGY_ERR_NONE) return err;
    } else {
        if (planeIdx == 0) {
            err = procPlaneVblur3(pPb6Plane, pInputPlane, stream);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneMakediff(pPb3Plane, pInputPlane, pPb6Plane, h_offset, stream);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneVblur3(pPb6Plane, pPb3Plane, stream);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneSbrCombine(pPb3Plane, pInputPlane, pPb3Plane, pPb6Plane, h_offset, stream);
            if (err != RGY_ERR_NONE) return err;
        } else {
            err = copyPlaneAsync(pPb3Plane, pInputPlane, stream);
            if (err != RGY_ERR_NONE) return err;
        }
        err = procPlaneVblur3(pPb6Plane, pPb3Plane, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return procPlaneFinalize(pOutputPlane, pInputPlane, pPb3Plane, pPb6Plane, sstr, scl, thr_hbd, amnt_hbd, stream);
}

RGY_ERR NVEncFilterVinverse::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto pVinverseParam = std::dynamic_pointer_cast<NVEncFilterParamVinverse>(m_param);
    if (!pVinverseParam) {
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int peak = (1 << bitDepth) - 1;
    const int hbd_shift = bitDepth - 8;
    int amnt_hbd = (int)(pVinverseParam->vinverse.amnt * (float)(1 << hbd_shift));
    if (pVinverseParam->vinverse.amnt >= 255.0f || amnt_hbd > peak) amnt_hbd = peak;
    if (amnt_hbd < 0) amnt_hbd = 0;
    int thr_hbd = (int)(pVinverseParam->vinverse.thr * (float)(1 << hbd_shift));
    if (thr_hbd < 0) thr_hbd = 0;
    if (thr_hbd > peak) thr_hbd = peak;
    const int h_offset = 1 << (bitDepth - 1);

    const int nPlanes = RGY_CSP_PLANES[rgy_csp_no_alpha(pOutputFrame->csp)];
    for (int ip = 0; ip < nPlanes; ip++) {
        const auto plane = (RGY_PLANE)ip;
        auto planeDst = getPlane(pOutputFrame, plane);
        auto planeSrc = getPlane(pInputFrame, plane);
        auto planePb3 = getPlane(&m_pb3[ip]->frame, RGY_PLANE_Y);
        auto planePb6 = getPlane(&m_pb6[ip]->frame, RGY_PLANE_Y);

        const bool process = (ip == 0) || pVinverseParam->vinverse.chroma;
        if (!process) {
            auto err = copyPlaneAsync(&planeDst, &planeSrc, stream);
            if (err != RGY_ERR_NONE) return err;
            continue;
        }

        auto err = procPlane(ip, &planeDst, &planeSrc, &planePb3, &planePb6,
            pVinverseParam->vinverse.mode, pVinverseParam->vinverse.sstr, pVinverseParam->vinverse.scl,
            thr_hbd, amnt_hbd, h_offset, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterVinverse::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
        AddMessage(RGY_LOG_ERROR, _T("error at vinverse(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void NVEncFilterVinverse::close() {
    m_pb3.clear();
    m_pb6.clear();
    m_frameBuf.clear();
}
