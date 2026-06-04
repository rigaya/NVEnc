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
#include <vector>
#include "convert_csp.h"
#include "NVEncFilterColorFix.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int COLORFIX_BLOCK_X = 16;
static const int COLORFIX_BLOCK_Y = 16;
static const int COLORFIX_WG_SIZE = COLORFIX_BLOCK_X * COLORFIX_BLOCK_Y;

template<typename Type>
__global__ void kernel_colorfix_apply_rgb(uint8_t *__restrict__ pR, int pitchR,
    uint8_t *__restrict__ pG, int pitchG, uint8_t *__restrict__ pB, int pitchB,
    int width, int height, float scaleR, float scaleG, float scaleB,
    float offsetR, float offsetG, float offsetB, int maxVal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    Type *rPix = (Type *)(pR + y * pitchR + x * sizeof(Type));
    Type *gPix = (Type *)(pG + y * pitchG + x * sizeof(Type));
    Type *bPix = (Type *)(pB + y * pitchB + x * sizeof(Type));
    const int ir = min(max(__float2int_rn((float)rPix[0] * scaleR + offsetR), 0), maxVal);
    const int ig = min(max(__float2int_rn((float)gPix[0] * scaleG + offsetG), 0), maxVal);
    const int ib = min(max(__float2int_rn((float)bPix[0] * scaleB + offsetB), 0), maxVal);
    rPix[0] = (Type)ir;
    gPix[0] = (Type)ig;
    bPix[0] = (Type)ib;
}

template<typename Type>
__global__ void kernel_colorfix_reduce_uv(const uint8_t *__restrict__ pY, int pitchY, int widthY, int heightY,
    const uint8_t *__restrict__ pU, int pitchU, int widthU, int heightU,
    const uint8_t *__restrict__ pV, int pitchV, int uvInterleaved, int subX, int subY,
    long long *__restrict__ outPartials) {
    __shared__ long long sU[COLORFIX_WG_SIZE];
    __shared__ long long sV[COLORFIX_WG_SIZE];
    __shared__ long long sY[COLORFIX_WG_SIZE];
    __shared__ long long sYsq[COLORFIX_WG_SIZE];

    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    const int lid = threadIdx.y * blockDim.x + threadIdx.x;
    long long uVal = 0;
    long long vVal = 0;
    long long yAcc = 0;
    long long ysqAcc = 0;

    if (cx < widthU && cy < heightU) {
        const int uvx = uvInterleaved ? cx * 2 : cx;
        uVal = (long long)(*(const Type *)(pU + cy * pitchU + uvx * sizeof(Type)));
        vVal = (long long)(*(const Type *)(pV + cy * pitchV + (uvx + uvInterleaved) * sizeof(Type)));
        for (int dy = 0; dy < subY; dy++) {
            const int ly = cy * subY + dy;
            if (ly >= heightY) break;
            for (int dx = 0; dx < subX; dx++) {
                const int lx = cx * subX + dx;
                if (lx >= widthY) break;
                const long long yv = (long long)(*(const Type *)(pY + ly * pitchY + lx * sizeof(Type)));
                yAcc += yv;
                ysqAcc += yv * yv;
            }
        }
    }

    sU[lid] = uVal;
    sV[lid] = vVal;
    sY[lid] = yAcc;
    sYsq[lid] = ysqAcc;
    __syncthreads();
    for (int s = COLORFIX_WG_SIZE >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            sU[lid] += sU[lid + s];
            sV[lid] += sV[lid + s];
            sY[lid] += sY[lid + s];
            sYsq[lid] += sYsq[lid + s];
        }
        __syncthreads();
    }
    if (lid == 0) {
        const int groupIdx = blockIdx.y * gridDim.x + blockIdx.x;
        outPartials[groupIdx * 4 + 0] = sU[0];
        outPartials[groupIdx * 4 + 1] = sV[0];
        outPartials[groupIdx * 4 + 2] = sY[0];
        outPartials[groupIdx * 4 + 3] = sYsq[0];
    }
}

template<typename Type>
__global__ void kernel_colorfix_apply_luma(uint8_t *__restrict__ pY, int pitchY,
    int widthY, int heightY, float scaleY, float offsetY, int maxVal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthY || y >= heightY) return;

    Type *yPix = (Type *)(pY + y * pitchY + x * sizeof(Type));
    const int iv = min(max(__float2int_rn((float)yPix[0] * scaleY + offsetY), 0), maxVal);
    yPix[0] = (Type)iv;
}

template<typename Type>
__global__ void kernel_colorfix_apply_uv(uint8_t *__restrict__ pU, int pitchU,
    uint8_t *__restrict__ pV, int pitchV, int widthU, int heightU, int uvInterleaved,
    int offsetU, int offsetV, int maxVal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthU || y >= heightU) return;

    const int uvx = uvInterleaved ? x * 2 : x;
    Type *uPix = (Type *)(pU + y * pitchU + uvx * sizeof(Type));
    Type *vPix = (Type *)(pV + y * pitchV + (uvx + uvInterleaved) * sizeof(Type));
    const int u = min(max((int)uPix[0] + offsetU, 0), maxVal);
    const int v = min(max((int)vPix[0] + offsetV, 0), maxVal);
    uPix[0] = (Type)u;
    vPix[0] = (Type)v;
}

template<typename Type>
__global__ void kernel_colorfix_reduce_rgb(const uint8_t *__restrict__ pR, int pitchR,
    const uint8_t *__restrict__ pG, int pitchG, const uint8_t *__restrict__ pB, int pitchB,
    int width, int height, long long *__restrict__ outPartials) {
    __shared__ long long sR[COLORFIX_WG_SIZE];
    __shared__ long long sG[COLORFIX_WG_SIZE];
    __shared__ long long sB[COLORFIX_WG_SIZE];
    __shared__ long long sY[COLORFIX_WG_SIZE];
    __shared__ long long sYsq[COLORFIX_WG_SIZE];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lid = threadIdx.y * blockDim.x + threadIdx.x;
    long long rv = 0;
    long long gv = 0;
    long long bv = 0;
    long long yv = 0;
    long long ysq = 0;
    if (x < width && y < height) {
        rv = (long long)(*(const Type *)(pR + y * pitchR + x * sizeof(Type)));
        gv = (long long)(*(const Type *)(pG + y * pitchG + x * sizeof(Type)));
        bv = (long long)(*(const Type *)(pB + y * pitchB + x * sizeof(Type)));
        const long long yLong = (19595LL * rv + 38470LL * gv + 7471LL * bv + 32768LL) >> 16;
        yv = yLong;
        ysq = yLong * yLong;
    }

    sR[lid] = rv;
    sG[lid] = gv;
    sB[lid] = bv;
    sY[lid] = yv;
    sYsq[lid] = ysq;
    __syncthreads();
    for (int s = COLORFIX_WG_SIZE >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            sR[lid] += sR[lid + s];
            sG[lid] += sG[lid + s];
            sB[lid] += sB[lid + s];
            sY[lid] += sY[lid + s];
            sYsq[lid] += sYsq[lid + s];
        }
        __syncthreads();
    }
    if (lid == 0) {
        const int groupIdx = blockIdx.y * gridDim.x + blockIdx.x;
        outPartials[groupIdx * 5 + 0] = sR[0];
        outPartials[groupIdx * 5 + 1] = sG[0];
        outPartials[groupIdx * 5 + 2] = sB[0];
        outPartials[groupIdx * 5 + 3] = sY[0];
        outPartials[groupIdx * 5 + 4] = sYsq[0];
    }
}

NVEncFilterColorFix::NVEncFilterColorFix() :
    m_resolvedMatrix(VPP_COLORFIX_MATRIX_BT709),
    m_effectiveSpace(VPP_COLORFIX_SPACE_RGB),
    m_convToRgb(),
    m_convToYuv(),
    m_cspRgb(RGY_CSP_RGB_16),
    m_reducePartials(),
    m_numGroupsLastDispatch(0),
    m_analysisComplete(false),
    m_analysedFrames(0),
    m_skippedFrames(0),
    m_totalSeenFrames(0),
    m_sumA(0), m_sumB(0), m_sumC(0), m_sumY(0), m_sumYsq(0),
    m_rollingVarianceSum(0.0), m_rollingVarianceCount(0),
    m_offsetU(0), m_offsetV(0),
    m_scaleR(1.0f), m_scaleG(1.0f), m_scaleB(1.0f) {
    m_name = _T("colorfix");
}

NVEncFilterColorFix::~NVEncFilterColorFix() {
    close();
}

RGY_ERR NVEncFilterColorFix::checkParam(const std::shared_ptr<NVEncFilterParamColorFix> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto& c = prm->colorfix;
    if (c.mode != VPP_COLORFIX_MODE_MANUAL && c.mode != VPP_COLORFIX_MODE_AUTO && c.mode != VPP_COLORFIX_MODE_GRAY) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mode=%d: must be 0 (manual), 1 (auto) or 2 (gray).\n"), c.mode);
        return RGY_ERR_INVALID_PARAM;
    }
    if (c.mode == VPP_COLORFIX_MODE_MANUAL) {
        for (int v : { c.whiteR, c.whiteG, c.whiteB, c.blackR, c.blackG, c.blackB }) {
            if (v < 0 || v > 255) {
                AddMessage(RGY_LOG_ERROR, _T("Invalid white/black channel value %d: must be 0..255.\n"), v);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        if (c.whiteR == c.blackR || c.whiteG == c.blackG || c.whiteB == c.blackB) {
            AddMessage(RGY_LOG_ERROR, _T("white and black points must differ on every channel (got R=%d/%d, G=%d/%d, B=%d/%d).\n"),
                c.whiteR, c.blackR, c.whiteG, c.blackG, c.whiteB, c.blackB);
            return RGY_ERR_INVALID_PARAM;
        }
    } else {
        if (c.frames < 10 || c.frames > 5000) {
            AddMessage(RGY_LOG_ERROR, _T("frames=%d must be in [10, 5000].\n"), c.frames);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!(c.strength >= 0.0f && c.strength <= 1.0f)) {
            AddMessage(RGY_LOG_ERROR, _T("strength=%.2f must be in [0.0, 1.0].\n"), c.strength);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!(c.varianceThreshold > 0.0f)) {
            AddMessage(RGY_LOG_ERROR, _T("variance_threshold=%.2f must be > 0.\n"), c.varianceThreshold);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
}

int NVEncFilterColorFix::resolveMatrix(const VppColorFix& cf, const VideoVUIInfo& vui, int height) const {
    if (cf.matrix != VPP_COLORFIX_MATRIX_AUTO) {
        return cf.matrix;
    }
    switch ((int)vui.matrix) {
    case RGY_MATRIX_BT709:       return VPP_COLORFIX_MATRIX_BT709;
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:    return VPP_COLORFIX_MATRIX_BT601;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL:   return VPP_COLORFIX_MATRIX_BT2020;
    default: break;
    }
    if (height <= 576) return VPP_COLORFIX_MATRIX_BT601;
    if (height <= 1200) return VPP_COLORFIX_MATRIX_BT709;
    return VPP_COLORFIX_MATRIX_BT2020;
}

int NVEncFilterColorFix::resolveSpace(const VppColorFix& cf) const {
    if (cf.space == VPP_COLORFIX_SPACE_RGB) return VPP_COLORFIX_SPACE_RGB;
    if (cf.space == VPP_COLORFIX_SPACE_YUV) return VPP_COLORFIX_SPACE_YUV;
    return (cf.mode == VPP_COLORFIX_MODE_MANUAL) ? VPP_COLORFIX_SPACE_RGB : VPP_COLORFIX_SPACE_YUV;
}

void NVEncFilterColorFix::getMatrixCoeffs(int resolvedMatrix, float& Kr, float& Kg, float& Kb) const {
    switch (resolvedMatrix) {
    case VPP_COLORFIX_MATRIX_BT2020: Kr = 0.2627f; Kb = 0.0593f; break;
    case VPP_COLORFIX_MATRIX_BT601:  Kr = 0.299f;  Kb = 0.114f;  break;
    case VPP_COLORFIX_MATRIX_BT709:
    default:                         Kr = 0.2126f; Kb = 0.0722f; break;
    }
    Kg = 1.0f - Kr - Kb;
}

RGY_ERR NVEncFilterColorFix::setupCspConverters(const RGYFrameInfo& frameIn, RGY_CSP cspRgb, rgy_rational<int> baseFps) {
    if (m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
        m_convToRgb.reset();
        m_convToYuv.reset();
        return RGY_ERR_NONE;
    }
    auto vuiForMatrix = VideoVUIInfo();
    switch (m_resolvedMatrix) {
    case VPP_COLORFIX_MATRIX_BT601:  vuiForMatrix.matrix = RGY_MATRIX_ST170_M; break;
    case VPP_COLORFIX_MATRIX_BT709:  vuiForMatrix.matrix = RGY_MATRIX_BT709; break;
    case VPP_COLORFIX_MATRIX_BT2020: vuiForMatrix.matrix = RGY_MATRIX_BT2020_NCL; break;
    default:                         vuiForMatrix.matrix = RGY_MATRIX_BT709; break;
    }

    RGY_ERR sts = RGY_ERR_NONE;
    {
        auto filter = std::make_unique<NVEncFilterCspCrop>();
        auto param = std::make_shared<NVEncFilterParamCrop>();
        param->frameIn = frameIn;
        param->frameOut = frameIn;
        param->frameOut.csp = cspRgb;
        param->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        param->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        param->baseFps = baseFps;
        param->matrix = vuiForMatrix.matrix;
        param->bOutOverwrite = false;
        sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;
        m_convToRgb = std::move(filter);
    }
    {
        auto filter = std::make_unique<NVEncFilterCspCrop>();
        auto param = std::make_shared<NVEncFilterParamCrop>();
        param->frameIn = frameIn;
        param->frameIn.csp = cspRgb;
        param->frameOut = frameIn;
        param->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        param->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        param->baseFps = baseFps;
        param->matrix = vuiForMatrix.matrix;
        param->bOutOverwrite = false;
        sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;
        m_convToYuv = std::move(filter);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterColorFix::allocReduceBuffer(const RGYFrameInfo& frameIn) {
    const int rgbWg = divCeil(frameIn.width, COLORFIX_BLOCK_X) * divCeil(frameIn.height, COLORFIX_BLOCK_Y);
    const size_t bufBytes = (size_t)rgbWg * 5 * sizeof(long long);
    if (!m_reducePartials || m_reducePartials->nSize < bufBytes) {
        m_reducePartials = std::make_unique<CUMemBuf>(bufBytes);
        auto sts = m_reducePartials->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate colorfix reduction buffer (%zu bytes): %s.\n"), bufBytes, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterColorFix::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamColorFix>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) return sts;

    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param: colorfix is in-place; bOutOverwrite must be true.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;
    prm->frameOut.picstruct = prm->frameIn.picstruct;
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    m_resolvedMatrix = resolveMatrix(prm->colorfix, prm->vui, prm->frameIn.height);
    m_effectiveSpace = resolveSpace(prm->colorfix);
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO && m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) {
        AddMessage(RGY_LOG_WARN, _T("colorfix: mode=auto only supports space=yuv; ignoring space=rgb.\n"));
        m_effectiveSpace = VPP_COLORFIX_SPACE_YUV;
    }

    if (m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) {
        m_cspRgb = (bitDepth > 8) ? RGY_CSP_RGB_16 : RGY_CSP_RGB;
        sts = setupCspConverters(prm->frameIn, m_cspRgb, prm->baseFps);
        if (sts != RGY_ERR_NONE) return sts;
    } else {
        m_convToRgb.reset();
        m_convToYuv.reset();
    }
    if (prm->colorfix.mode != VPP_COLORFIX_MODE_MANUAL) {
        sts = allocReduceBuffer(prm->frameIn);
        if (sts != RGY_ERR_NONE) return sts;
    }

    m_analysisComplete = false;
    m_analysedFrames = 0;
    m_skippedFrames = 0;
    m_totalSeenFrames = 0;
    m_sumA = m_sumB = m_sumC = m_sumY = m_sumYsq = 0;
    m_rollingVarianceSum = 0.0;
    m_rollingVarianceCount = 0;
    m_offsetU = m_offsetV = 0;
    m_scaleR = m_scaleG = m_scaleB = 1.0f;
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_MANUAL && m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) {
        const int rgbMax = (RGY_CSP_BIT_DEPTH[m_cspRgb] > 8) ? 65535 : 255;
        const int wR = prm->colorfix.whiteR * rgbMax / 255;
        const int wG = prm->colorfix.whiteG * rgbMax / 255;
        const int wB = prm->colorfix.whiteB * rgbMax / 255;
        const int kR = prm->colorfix.blackR * rgbMax / 255;
        const int kG = prm->colorfix.blackG * rgbMax / 255;
        const int kB = prm->colorfix.blackB * rgbMax / 255;
        m_scaleR = (float)rgbMax / (float)(wR - kR);
        m_scaleG = (float)rgbMax / (float)(wG - kG);
        m_scaleB = (float)rgbMax / (float)(wB - kB);
        m_analysisComplete = true;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamColorFix::print() const {
    return colorfix.print();
}

RGY_ERR NVEncFilterColorFix::runApplyRGB(RGYFrameInfo *pTarget,
    float scaleR, float scaleG, float scaleB, float offsetR, float offsetG, float offsetB, cudaStream_t stream) {
    const auto pR = getPlane(pTarget, RGY_PLANE_R);
    const auto pG = getPlane(pTarget, RGY_PLANE_G);
    const auto pB = getPlane(pTarget, RGY_PLANE_B);
    const int maxVal = (1 << RGY_CSP_BIT_DEPTH[pTarget->csp]) - 1;
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(pTarget->width, block.x), divCeil(pTarget->height, block.y));
    if (RGY_CSP_DATA_TYPE[pTarget->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_apply_rgb<uint16_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pR.ptr[0], pR.pitch[0], (uint8_t *)pG.ptr[0], pG.pitch[0], (uint8_t *)pB.ptr[0], pB.pitch[0],
            pTarget->width, pTarget->height, scaleR, scaleG, scaleB, offsetR, offsetG, offsetB, maxVal);
    } else {
        kernel_colorfix_apply_rgb<uint8_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pR.ptr[0], pR.pitch[0], (uint8_t *)pG.ptr[0], pG.pitch[0], (uint8_t *)pB.ptr[0], pB.pitch[0],
            pTarget->width, pTarget->height, scaleR, scaleG, scaleB, offsetR, offsetG, offsetB, maxVal);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterColorFix::runReduceUV(RGYFrameInfo *pSrc, cudaStream_t stream) {
    const auto pY = getPlane(pSrc, RGY_PLANE_Y);
    const auto pU = getPlane(pSrc, RGY_PLANE_U);
    const auto pV = getPlane(pSrc, RGY_PLANE_V);
    const int uvInterleaved = (pU.ptr[0] == pV.ptr[0]) ? 1 : 0;
    const int chromaWidth = uvInterleaved ? std::max(1, pU.width / 2) : pU.width;
    const int chromaHeight = pU.height;
    if (!uvInterleaved && (pU.width != pV.width || pU.height != pV.height || pU.pitch[0] != pV.pitch[0])) {
        AddMessage(RGY_LOG_ERROR, _T("colorfix: U/V plane layout mismatch.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const int subX = std::max(1, pY.width / chromaWidth);
    const int subY = std::max(1, pY.height / pU.height);
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(chromaWidth, block.x), divCeil(chromaHeight, block.y));
    m_numGroupsLastDispatch = grid.x * grid.y;
    if (RGY_CSP_DATA_TYPE[pSrc->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_reduce_uv<uint16_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (const uint8_t *)pU.ptr[0], pU.pitch[0], chromaWidth, chromaHeight,
            (const uint8_t *)pV.ptr[0], pV.pitch[0], uvInterleaved, subX, subY, (long long *)m_reducePartials->ptr);
    } else {
        kernel_colorfix_reduce_uv<uint8_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (const uint8_t *)pU.ptr[0], pU.pitch[0], chromaWidth, chromaHeight,
            (const uint8_t *)pV.ptr[0], pV.pitch[0], uvInterleaved, subX, subY, (long long *)m_reducePartials->ptr);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterColorFix::runReduceRGB(RGYFrameInfo *pSrc, cudaStream_t stream) {
    const auto pR = getPlane(pSrc, RGY_PLANE_R);
    const auto pG = getPlane(pSrc, RGY_PLANE_G);
    const auto pB = getPlane(pSrc, RGY_PLANE_B);
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(pSrc->width, block.x), divCeil(pSrc->height, block.y));
    m_numGroupsLastDispatch = grid.x * grid.y;
    if (RGY_CSP_DATA_TYPE[pSrc->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_reduce_rgb<uint16_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pR.ptr[0], pR.pitch[0], (const uint8_t *)pG.ptr[0], pG.pitch[0], (const uint8_t *)pB.ptr[0], pB.pitch[0],
            pSrc->width, pSrc->height, (long long *)m_reducePartials->ptr);
    } else {
        kernel_colorfix_reduce_rgb<uint8_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pR.ptr[0], pR.pitch[0], (const uint8_t *)pG.ptr[0], pG.pitch[0], (const uint8_t *)pB.ptr[0], pB.pitch[0],
            pSrc->width, pSrc->height, (long long *)m_reducePartials->ptr);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterColorFix::runApplyUV(RGYFrameInfo *pTarget, int offsetU, int offsetV, cudaStream_t stream) {
    const auto pU = getPlane(pTarget, RGY_PLANE_U);
    const auto pV = getPlane(pTarget, RGY_PLANE_V);
    const int uvInterleaved = (pU.ptr[0] == pV.ptr[0]) ? 1 : 0;
    const int chromaWidth = uvInterleaved ? std::max(1, pU.width / 2) : pU.width;
    const int chromaHeight = pU.height;
    if (!uvInterleaved && (pU.width != pV.width || pU.height != pV.height || pU.pitch[0] != pV.pitch[0])) {
        AddMessage(RGY_LOG_ERROR, _T("colorfix: U/V plane layout mismatch.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const int maxVal = (1 << RGY_CSP_BIT_DEPTH[pTarget->csp]) - 1;
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(chromaWidth, block.x), divCeil(chromaHeight, block.y));
    if (RGY_CSP_DATA_TYPE[pTarget->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_apply_uv<uint16_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pU.ptr[0], pU.pitch[0], (uint8_t *)pV.ptr[0], pV.pitch[0],
            chromaWidth, chromaHeight, uvInterleaved, offsetU, offsetV, maxVal);
    } else {
        kernel_colorfix_apply_uv<uint8_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pU.ptr[0], pU.pitch[0], (uint8_t *)pV.ptr[0], pV.pitch[0],
            chromaWidth, chromaHeight, uvInterleaved, offsetU, offsetV, maxVal);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterColorFix::runApplyLuma(RGYFrameInfo *pTarget, float scaleY, float offsetY, cudaStream_t stream) {
    const auto pY = getPlane(pTarget, RGY_PLANE_Y);
    const int maxVal = (1 << RGY_CSP_BIT_DEPTH[pTarget->csp]) - 1;
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(pY.width, block.x), divCeil(pY.height, block.y));
    if (RGY_CSP_DATA_TYPE[pTarget->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_apply_luma<uint16_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height, scaleY, offsetY, maxVal);
    } else {
        kernel_colorfix_apply_luma<uint8_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height, scaleY, offsetY, maxVal);
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterColorFix::finaliseReduction(cudaStream_t stream, int numLongsPerGroup, std::vector<long long>& outTotals) {
    const size_t count = (size_t)m_numGroupsLastDispatch * numLongsPerGroup;
    std::vector<long long> host(count);
    auto cudaerr = cudaMemcpyAsync(host.data(), m_reducePartials->ptr, count * sizeof(host[0]), cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    cudaerr = cudaStreamSynchronize(stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    outTotals.assign(numLongsPerGroup, 0LL);
    for (int g = 0; g < m_numGroupsLastDispatch; g++) {
        for (int i = 0; i < numLongsPerGroup; i++) {
            outTotals[i] += host[g * numLongsPerGroup + i];
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterColorFix::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set (in-place filter).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamColorFix>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo *targetFrame = ppOutputFrames[0];

    if (prm->colorfix.mode == VPP_COLORFIX_MODE_MANUAL) {
        m_totalSeenFrames++;
        if (m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
            const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
            const int maxVal = (1 << bitDepth) - 1;
            const float neutral = (float)((maxVal + 1) / 2);
            float Kr = 0.0f, Kg = 0.0f, Kb = 0.0f;
            getMatrixCoeffs(m_resolvedMatrix, Kr, Kg, Kb);
            const float scaleRgb8ToBd = (float)maxVal / 255.0f;
            const float wY = (Kr * (float)prm->colorfix.whiteR + Kg * (float)prm->colorfix.whiteG + Kb * (float)prm->colorfix.whiteB) * scaleRgb8ToBd;
            const float kY = (Kr * (float)prm->colorfix.blackR + Kg * (float)prm->colorfix.blackG + Kb * (float)prm->colorfix.blackB) * scaleRgb8ToBd;
            const float yDen = (wY - kY != 0.0f) ? (wY - kY) : 1.0f;
            const float scaleY = (float)maxVal / yDen;
            const float offsetY = -kY * scaleY;
            const float wBbd = (float)prm->colorfix.whiteB * scaleRgb8ToBd;
            const float wRbd = (float)prm->colorfix.whiteR * scaleRgb8ToBd;
            const float Uwp = (wBbd - wY) / (2.0f * (1.0f - Kb)) + neutral;
            const float Vwp = (wRbd - wY) / (2.0f * (1.0f - Kr)) + neutral;
            const int offsetU = (int)std::lround(neutral - Uwp);
            const int offsetV = (int)std::lround(neutral - Vwp);
            auto err = runApplyLuma(targetFrame, scaleY, offsetY, stream);
            if (err != RGY_ERR_NONE) return err;
            return runApplyUV(targetFrame, offsetU, offsetV, stream);
        }

        int convOutNum = 0;
        RGYFrameInfo *convOut[1] = { nullptr };
        RGYFrameInfo inFrame = *targetFrame;
        auto err = m_convToRgb->filter(&inFrame, (RGYFrameInfo **)&convOut, &convOutNum, stream);
        if (err != RGY_ERR_NONE || convOut[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("YUV->RGB conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
        RGYFrameInfo *pRgb = convOut[0];
        const int rgbMax = (RGY_CSP_BIT_DEPTH[m_cspRgb] > 8) ? 65535 : 255;
        const int wR = prm->colorfix.whiteR * rgbMax / 255;
        const int wG = prm->colorfix.whiteG * rgbMax / 255;
        const int wB = prm->colorfix.whiteB * rgbMax / 255;
        const int kR = prm->colorfix.blackR * rgbMax / 255;
        const int kG = prm->colorfix.blackG * rgbMax / 255;
        const int kB = prm->colorfix.blackB * rgbMax / 255;
        const float scaleR = (float)rgbMax / (float)(wR - kR);
        const float scaleG = (float)rgbMax / (float)(wG - kG);
        const float scaleB = (float)rgbMax / (float)(wB - kB);
        const float offsetR = -((float)kR) * scaleR;
        const float offsetG = -((float)kG) * scaleG;
        const float offsetB = -((float)kB) * scaleB;
        err = runApplyRGB(pRgb, scaleR, scaleG, scaleB, offsetR, offsetG, offsetB, stream);
        if (err != RGY_ERR_NONE) return err;
        int convOutNum2 = 0;
        RGYFrameInfo *convOut2[1] = { (m_convToYuv->GetFilterParam()->frameOut.csp == targetFrame->csp) ? targetFrame : nullptr };
        RGYFrameInfo inFrameRgb = *pRgb;
        err = m_convToYuv->filter(&inFrameRgb, (RGYFrameInfo **)&convOut2, &convOutNum2, stream);
        if (err != RGY_ERR_NONE || convOut2[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("RGB->YUV conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }

    auto analyzeYuv = [&]() -> RGY_ERR {
        auto err = runReduceUV(targetFrame, stream);
        if (err != RGY_ERR_NONE) return err;
        std::vector<long long> totals;
        err = finaliseReduction(stream, 4, totals);
        if (err != RGY_ERR_NONE) return err;
        const long long sumU = totals[0], sumV = totals[1], sumY = totals[2], sumYsq = totals[3];
        const auto planeU = getPlane(targetFrame, RGY_PLANE_U);
        const auto planeV = getPlane(targetFrame, RGY_PLANE_V);
        const auto planeY = getPlane(targetFrame, RGY_PLANE_Y);
        const long long npxChroma = (long long)((planeU.ptr[0] == planeV.ptr[0]) ? std::max(1, planeU.width / 2) : planeU.width) * planeU.height;
        const long long npxLuma = (long long)planeY.width * planeY.height;
        const double meanY = (double)sumY / (double)npxLuma;
        const double varY = (double)sumYsq / (double)npxLuma - meanY * meanY;
        bool skip = false;
        if (m_rollingVarianceCount > 0) {
            const double rollingAvg = m_rollingVarianceSum / m_rollingVarianceCount;
            const double upper = rollingAvg * prm->colorfix.varianceThreshold;
            const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
            if (varY > upper || varY < lower) skip = true;
        }
        if (!skip) {
            m_sumA += sumU;
            m_sumB += sumV;
            m_sumY += sumY;
            m_sumYsq += sumYsq;
            m_rollingVarianceSum += varY;
            m_rollingVarianceCount++;
            m_analysedFrames++;
            m_sumC += npxChroma;
        } else {
            m_skippedFrames++;
        }
        if (m_analysedFrames >= prm->colorfix.frames) {
            const double meanU = (double)m_sumA / (double)m_sumC;
            const double meanV = (double)m_sumB / (double)m_sumC;
            const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
            const float maxVal = (float)((1 << bitDepth) - 1);
            const double neutral = (double)(1 << (bitDepth - 1));
            m_offsetU = (int)std::lround(-(meanU - neutral) * prm->colorfix.strength);
            m_offsetV = (int)std::lround(-(meanV - neutral) * prm->colorfix.strength);
            m_analysisComplete = true;
            AddMessage(RGY_LOG_INFO, _T("analysis complete -- offsetU=%+.3f, offsetV=%+.3f (skipped %d flash frames)\n"),
                m_offsetU / maxVal, m_offsetV / maxVal, m_skippedFrames);
        }
        return RGY_ERR_NONE;
    };

    if (prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO || m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
        m_totalSeenFrames++;
        if (!m_analysisComplete) {
            return analyzeYuv();
        }
        return runApplyUV(targetFrame, m_offsetU, m_offsetV, stream);
    }

    m_totalSeenFrames++;
    int convOutNum = 0;
    RGYFrameInfo *convOut[1] = { nullptr };
    RGYFrameInfo inFrame = *targetFrame;
    auto err = m_convToRgb->filter(&inFrame, (RGYFrameInfo **)&convOut, &convOutNum, stream);
    if (err != RGY_ERR_NONE || convOut[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("YUV->RGB conversion failed: %s.\n"), get_err_mes(err));
        return err;
    }
    RGYFrameInfo *pRgb = convOut[0];
    if (!m_analysisComplete) {
        err = runReduceRGB(pRgb, stream);
        if (err != RGY_ERR_NONE) return err;
        std::vector<long long> totals;
        err = finaliseReduction(stream, 5, totals);
        if (err != RGY_ERR_NONE) return err;
        const long long sumR = totals[0], sumG = totals[1], sumB = totals[2];
        const long long sumY = totals[3], sumYsq = totals[4];
        const long long npx = (long long)pRgb->width * pRgb->height;
        const double meanY = (double)sumY / (double)npx;
        const double varY = (double)sumYsq / (double)npx - meanY * meanY;
        bool skip = false;
        if (m_rollingVarianceCount > 0) {
            const double rollingAvg = m_rollingVarianceSum / m_rollingVarianceCount;
            const double upper = rollingAvg * prm->colorfix.varianceThreshold;
            const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
            if (varY > upper || varY < lower) skip = true;
        }
        if (!skip) {
            m_sumA += sumR;
            m_sumB += sumG;
            m_sumC += sumB;
            m_sumY += sumY;
            m_sumYsq += sumYsq;
            m_rollingVarianceSum += varY;
            m_rollingVarianceCount++;
            m_analysedFrames++;
        } else {
            m_skippedFrames++;
        }
        if (m_analysedFrames >= prm->colorfix.frames) {
            const long long npxTotal = (long long)pRgb->width * pRgb->height * m_analysedFrames;
            const double meanR = (double)m_sumA / (double)npxTotal;
            const double meanG = (double)m_sumB / (double)npxTotal;
            const double meanB = (double)m_sumC / (double)npxTotal;
            const double meanAll = (meanR + meanG + meanB) / 3.0;
            const float strength = prm->colorfix.strength;
            auto safeScale = [&](double mean) {
                return (mean < 1.0) ? 1.0f : (float)((meanAll / mean) * strength + (1.0 - strength));
            };
            m_scaleR = safeScale(meanR);
            m_scaleG = safeScale(meanG);
            m_scaleB = safeScale(meanB);
            m_analysisComplete = true;
            AddMessage(RGY_LOG_INFO, _T("gray analysis complete -- scaleR=%.3f, scaleG=%.3f, scaleB=%.3f (skipped %d flash frames)\n"),
                m_scaleR, m_scaleG, m_scaleB, m_skippedFrames);
        }
    } else {
        err = runApplyRGB(pRgb, m_scaleR, m_scaleG, m_scaleB, 0.0f, 0.0f, 0.0f, stream);
        if (err != RGY_ERR_NONE) return err;
    }
    int yuvOutNum = 0;
    RGYFrameInfo *yuvOut[1] = { (m_convToYuv->GetFilterParam()->frameOut.csp == targetFrame->csp) ? targetFrame : nullptr };
    RGYFrameInfo inFrameRgb = *pRgb;
    err = m_convToYuv->filter(&inFrameRgb, (RGYFrameInfo **)&yuvOut, &yuvOutNum, stream);
    if (err != RGY_ERR_NONE || yuvOut[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("RGB->YUV conversion failed: %s.\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterColorFix::close() {
    m_convToRgb.reset();
    m_convToYuv.reset();
    m_reducePartials.reset();
    m_numGroupsLastDispatch = 0;
    m_analysisComplete = false;
    m_analysedFrames = 0;
    m_skippedFrames = 0;
    m_totalSeenFrames = 0;
    m_sumA = m_sumB = m_sumC = m_sumY = m_sumYsq = 0;
    m_rollingVarianceSum = 0.0;
    m_rollingVarianceCount = 0;
    m_offsetU = m_offsetV = 0;
    m_scaleR = m_scaleG = m_scaleB = 1.0f;
    m_frameBuf.clear();
}
