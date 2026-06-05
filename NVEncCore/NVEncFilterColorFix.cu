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
#include <cstring>
#include <memory>
#include <vector>
#include "convert_csp.h"
#include "NVEncFilterColorFix.h"
#include "rgy_avutil.h"
#include "rgy_cuda_util_kernel.h"
#include "rgy_filter_input_probe.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int COLORFIX_BLOCK_X = 32;
static const int COLORFIX_BLOCK_Y = 8;
static const int COLORFIX_WG_SIZE = COLORFIX_BLOCK_X * COLORFIX_BLOCK_Y;

static bool colorfix_variance_guard(
    double varY,
    double& rollingVarSum,
    int& rollingVarCount,
    int bit_depth,
    float varianceThreshold) {
    static constexpr int K_WARMUP = 60;
    static constexpr double SCENE_CUT_FACTOR = 20.0;
    const double MIN_FLOOR = 100.0 * (double)(1LL << (2 * (bit_depth - 8)));

    if (rollingVarCount < K_WARMUP) {
        return false;
    }
    double rollingAvg = rollingVarSum / (double)rollingVarCount;
    if (rollingAvg < MIN_FLOOR) rollingAvg = MIN_FLOOR;

    const double upper = rollingAvg * (double)varianceThreshold;
    const double lower = rollingAvg * 0.1 / (double)varianceThreshold;
    if (varY > upper) {
        if (varY > rollingAvg * SCENE_CUT_FACTOR) {
            rollingVarSum = varY;
            rollingVarCount = 1;
        }
        return true;
    }
    if (varY < lower) return true;
    return false;
}

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
    const uint8_t *__restrict__ pV, int pitchV, int subX, int subY,
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
        uVal = (long long)(*(const Type *)(pU + cy * pitchU + cx * sizeof(Type)));
        vVal = (long long)(*(const Type *)(pV + cy * pitchV + cx * sizeof(Type)));
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
    uint8_t *__restrict__ pV, int pitchV, int widthU, int heightU,
    int offsetU, int offsetV, int maxVal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthU || y >= heightU) return;

    Type *uPix = (Type *)(pU + y * pitchU + x * sizeof(Type));
    Type *vPix = (Type *)(pV + y * pitchV + x * sizeof(Type));
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
    m_scaleR(1.0f), m_scaleG(1.0f), m_scaleB(1.0f),
    m_prescanUsed(false),
    m_hardCapFrames(0) {
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
    m_prescanUsed = false;
    m_hardCapFrames = prm->colorfix.frames * 3 + 10;

    const bool wantPreScan =
           prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO
        || (prm->colorfix.mode == VPP_COLORFIX_MODE_GRAY
            && m_effectiveSpace == VPP_COLORFIX_SPACE_YUV);
    if (wantPreScan) {
        const auto preErr = runPreScanLibav(prm);
        if (preErr == RGY_ERR_NONE) {
            m_prescanUsed = true;
            m_analysisComplete = true;
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("init-time pre-scan unavailable; ramp fallback at runtime.\n"));
        }
    }

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
    const int subX = std::max(1, pY.width / pU.width);
    const int subY = std::max(1, pY.height / pU.height);
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(pU.width, block.x), divCeil(pU.height, block.y));
    m_numGroupsLastDispatch = grid.x * grid.y;
    if (RGY_CSP_DATA_TYPE[pSrc->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_reduce_uv<uint16_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (const uint8_t *)pU.ptr[0], pU.pitch[0], pU.width, pU.height,
            (const uint8_t *)pV.ptr[0], pV.pitch[0], subX, subY, (long long *)m_reducePartials->ptr);
    } else {
        kernel_colorfix_reduce_uv<uint8_t><<<grid, block, 0, stream>>>(
            (const uint8_t *)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (const uint8_t *)pU.ptr[0], pU.pitch[0], pU.width, pU.height,
            (const uint8_t *)pV.ptr[0], pV.pitch[0], subX, subY, (long long *)m_reducePartials->ptr);
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
    const int maxVal = (1 << RGY_CSP_BIT_DEPTH[pTarget->csp]) - 1;
    dim3 block(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    dim3 grid(divCeil(pU.width, block.x), divCeil(pU.height, block.y));
    if (RGY_CSP_DATA_TYPE[pTarget->csp] == RGY_DATA_TYPE_U16) {
        kernel_colorfix_apply_uv<uint16_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pU.ptr[0], pU.pitch[0], (uint8_t *)pV.ptr[0], pV.pitch[0],
            pU.width, pU.height, offsetU, offsetV, maxVal);
    } else {
        kernel_colorfix_apply_uv<uint8_t><<<grid, block, 0, stream>>>(
            (uint8_t *)pU.ptr[0], pU.pitch[0], (uint8_t *)pV.ptr[0], pV.pitch[0],
            pU.width, pU.height, offsetU, offsetV, maxVal);
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

RGY_ERR NVEncFilterColorFix::runPreScanLibav(const std::shared_ptr<NVEncFilterParamColorFix>& prm) {
    if (!prm || prm->inputFilePath.empty()) {
        return RGY_ERR_UNSUPPORTED;
    }
    std::string fileUtf8;
    if (tchar_to_string(prm->inputFilePath.c_str(), fileUtf8, CP_UTF8) == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: utf-8 conversion failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (const char *proto = unsupportedProbeProtocol(fileUtf8); proto != nullptr) {
        AddMessage(RGY_LOG_DEBUG,
            _T("pre-scan: input uses %s protocol; ramp fallback at runtime.\n"),
            char_to_tstring(proto).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    const int savedAvLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    struct AvLogLevelRestorer { int prev; ~AvLogLevelRestorer() { av_log_set_level(prev); } } avGuard{ savedAvLogLevel };

    AVFormatContext *fmtCtxRaw = nullptr;
    if (avformat_open_input(&fmtCtxRaw, fileUtf8.c_str(), nullptr, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avformat_open_input failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    std::unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> fmtGuard(
        fmtCtxRaw, RGYAVDeleter<AVFormatContext>(avformat_close_input));
    AVFormatContext *fmtCtx = fmtGuard.get();

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avformat_find_stream_info failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIdx < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: no video stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVStream *vst = fmtCtx->streams[videoIdx];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: decoder unavailable for stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVCodecContext *codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecGuard(
        codecCtxRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    AVCodecContext *codecCtx = codecGuard.get();
    if (avcodec_parameters_to_context(codecCtx, vst->codecpar) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_parameters_to_context failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    codecCtx->time_base = vst->time_base;
    codecCtx->pkt_timebase = vst->time_base;
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_open2 failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    AVPacket *pktRaw = av_packet_alloc();
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> pktGuard(
        pktRaw, RGYAVDeleter<AVPacket>(av_packet_free));
    AVFrame *frameRaw = av_frame_alloc();
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameGuard(
        frameRaw, RGYAVDeleter<AVFrame>(av_frame_free));

    const int wantFrames = prm->colorfix.frames;
    const int seenCap = wantFrames * 3 + 10;
    const int targetBitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const double targetMax = (double)((1 << targetBitDepth) - 1);
    const double neutralTarget = (double)(1 << (targetBitDepth - 1));

    uint64_t sumU = 0, sumV = 0;
    uint64_t totalChromaPx = 0;
    double rollingVarSum = 0.0;
    int rollingVarCount = 0;
    int analysedFrames = 0;
    int skippedFrames = 0;
    int seenFrames = 0;
    int srcBitDepth = 0;

    auto processFrame = [&](AVFrame *f) -> RGY_ERR {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)f->format);
        if (!desc || desc->nb_components < 3
            || (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_PAL) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0) {
            return RGY_ERR_UNSUPPORTED;
        }
        const int depthLuma = desc->comp[0].depth;
        const int depthChroma = desc->comp[1].depth;
        if (depthLuma == 0 || depthChroma == 0 || depthLuma > 16 || depthChroma > 16) {
            return RGY_ERR_UNSUPPORTED;
        }
        if (srcBitDepth == 0) srcBitDepth = depthLuma;

        const int planeY = desc->comp[0].plane;
        const int planeU = desc->comp[1].plane;
        const int planeV = desc->comp[2].plane;
        const int stepY = desc->comp[0].step;
        const int stepU = desc->comp[1].step;
        const int stepV = desc->comp[2].step;
        const int offY = desc->comp[0].offset;
        const int offU = desc->comp[1].offset;
        const int offV = desc->comp[2].offset;
        const int shY = desc->comp[0].shift;
        const int shU = desc->comp[1].shift;
        const int shV = desc->comp[2].shift;
        const int chromaShiftW = desc->log2_chroma_w;
        const int chromaShiftH = desc->log2_chroma_h;
        const int chromaW = (f->width + (1 << chromaShiftW) - 1) >> chromaShiftW;
        const int chromaH = (f->height + (1 << chromaShiftH) - 1) >> chromaShiftH;
        const int lumaW = f->width;
        const int lumaH = f->height;
        const bool hbdLuma = depthLuma > 8;
        const bool hbdChroma = depthChroma > 8;
        const uint32_t maskLuma = (1U << depthLuma) - 1U;
        const uint32_t maskChroma = (1U << depthChroma) - 1U;

        if (!f->data[planeU] || !f->data[planeV] || !f->data[planeY]) {
            return RGY_ERR_UNSUPPORTED;
        }

        uint64_t frameSumU = 0, frameSumV = 0;
        for (int y = 0; y < chromaH; ++y) {
            const uint8_t *rowU = f->data[planeU] + (size_t)y * f->linesize[planeU];
            const uint8_t *rowV = f->data[planeV] + (size_t)y * f->linesize[planeV];
            for (int x = 0; x < chromaW; ++x) {
                uint32_t u, v;
                if (hbdChroma) {
                    const uint8_t *pU = rowU + (size_t)x * stepU + offU;
                    const uint8_t *pV = rowV + (size_t)x * stepV + offV;
                    const uint32_t rawU = (uint32_t)pU[0] | ((uint32_t)pU[1] << 8);
                    const uint32_t rawV = (uint32_t)pV[0] | ((uint32_t)pV[1] << 8);
                    u = (rawU >> shU) & maskChroma;
                    v = (rawV >> shV) & maskChroma;
                } else {
                    u = ((uint32_t)rowU[(size_t)x * stepU + offU] >> shU) & maskChroma;
                    v = ((uint32_t)rowV[(size_t)x * stepV + offV] >> shV) & maskChroma;
                }
                frameSumU += u;
                frameSumV += v;
            }
        }

        uint64_t frameSumY = 0, frameSumYsq = 0;
        for (int y = 0; y < lumaH; ++y) {
            const uint8_t *rowY = f->data[planeY] + (size_t)y * f->linesize[planeY];
            for (int x = 0; x < lumaW; ++x) {
                uint32_t yv;
                if (hbdLuma) {
                    const uint8_t *pY = rowY + (size_t)x * stepY + offY;
                    const uint32_t rawY = (uint32_t)pY[0] | ((uint32_t)pY[1] << 8);
                    yv = (rawY >> shY) & maskLuma;
                } else {
                    yv = ((uint32_t)rowY[(size_t)x * stepY + offY] >> shY) & maskLuma;
                }
                frameSumY += yv;
                frameSumYsq += (uint64_t)yv * (uint64_t)yv;
            }
        }
        const uint64_t npxChroma = (uint64_t)chromaW * (uint64_t)chromaH;
        const uint64_t npxLuma = (uint64_t)lumaW * (uint64_t)lumaH;
        if (npxChroma == 0 || npxLuma == 0) return RGY_ERR_UNSUPPORTED;
        const double meanY = (double)frameSumY / (double)npxLuma;
        const double varY = (double)frameSumYsq / (double)npxLuma - meanY * meanY;

        const bool skip = colorfix_variance_guard(
            varY, rollingVarSum, rollingVarCount,
            depthLuma, prm->colorfix.varianceThreshold);
        if (!skip) {
            sumU += frameSumU;
            sumV += frameSumV;
            totalChromaPx += npxChroma;
            rollingVarSum += varY;
            ++rollingVarCount;
            ++analysedFrames;
        } else {
            ++skippedFrames;
        }
        return RGY_ERR_NONE;
    };

    auto drainDecoder = [&]() -> RGY_ERR {
        while (analysedFrames < wantFrames && seenFrames < seenCap) {
            int rv = avcodec_receive_frame(codecCtx, frameGuard.get());
            if (rv == AVERROR(EAGAIN) || rv == AVERROR_EOF) return RGY_ERR_NONE;
            if (rv < 0) return RGY_ERR_UNKNOWN;
            ++seenFrames;
            auto procErr = processFrame(frameGuard.get());
            av_frame_unref(frameGuard.get());
            if (procErr != RGY_ERR_NONE) return procErr;
        }
        return RGY_ERR_NONE;
    };

    while (analysedFrames < wantFrames && seenFrames < seenCap) {
        int rd = av_read_frame(fmtCtx, pktGuard.get());
        if (rd == AVERROR_EOF) break;
        if (rd < 0) {
            AddMessage(RGY_LOG_DEBUG, _T("pre-scan: av_read_frame error.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (pktGuard.get()->stream_index != videoIdx) {
            av_packet_unref(pktGuard.get());
            continue;
        }
        const int sendErr = avcodec_send_packet(codecCtx, pktGuard.get());
        av_packet_unref(pktGuard.get());
        if (sendErr < 0 && sendErr != AVERROR(EAGAIN)) {
            AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_send_packet error.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        auto rcvErr = drainDecoder();
        if (rcvErr != RGY_ERR_NONE) return rcvErr;
    }
    if (analysedFrames < wantFrames && seenFrames < seenCap) {
        avcodec_send_packet(codecCtx, nullptr);
        auto rcvErr = drainDecoder();
        if (rcvErr != RGY_ERR_NONE) return rcvErr;
    }

    if (analysedFrames == 0 || totalChromaPx == 0 || srcBitDepth == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: no usable frames decoded.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const double srcMax = (double)((1ULL << srcBitDepth) - 1ULL);
    const double meanU_src = (double)sumU / (double)totalChromaPx;
    const double meanV_src = (double)sumV / (double)totalChromaPx;
    const double meanU = meanU_src * targetMax / srcMax;
    const double meanV = meanV_src * targetMax / srcMax;
    const double rawOffU = -(meanU - neutralTarget) * prm->colorfix.strength;
    const double rawOffV = -(meanV - neutralTarget) * prm->colorfix.strength;
    m_offsetU = (int)std::lround(rawOffU);
    m_offsetV = (int)std::lround(rawOffV);
    m_analysedFrames = analysedFrames;
    m_skippedFrames = skippedFrames;

    const float offUNorm = (float)((double)m_offsetU / targetMax);
    const float offVNorm = (float)((double)m_offsetV / targetMax);
    AddMessage(RGY_LOG_INFO,
        _T("pre-scan complete -- offsetU=%+.3f, offsetV=%+.3f ")
        _T("(analysed %d frames, skipped %d, src bit_depth=%d -> target bit_depth=%d).\n"),
        offUNorm, offVNorm, analysedFrames, skippedFrames, srcBitDepth, targetBitDepth);
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
        const auto planeY = getPlane(targetFrame, RGY_PLANE_Y);
        const long long npxChroma = (long long)planeU.width * planeU.height;
        const long long npxLuma = (long long)planeY.width * planeY.height;
        const double meanY = (double)sumY / (double)npxLuma;
        const double varY = (double)sumYsq / (double)npxLuma - meanY * meanY;
        const bool skip = colorfix_variance_guard(
            varY, m_rollingVarianceSum, m_rollingVarianceCount,
            RGY_CSP_BIT_DEPTH[targetFrame->csp], prm->colorfix.varianceThreshold);
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

        int runningOffU = 0, runningOffV = 0;
        const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
        const float maxVal = (float)((1 << bitDepth) - 1);
        if (m_analysedFrames > 0 && m_sumC > 0) {
            const double meanU = (double)m_sumA / (double)m_sumC;
            const double meanV = (double)m_sumB / (double)m_sumC;
            const double neutral = (double)(1 << (bitDepth - 1));
            runningOffU = (int)std::lround(-(meanU - neutral) * prm->colorfix.strength);
            runningOffV = (int)std::lround(-(meanV - neutral) * prm->colorfix.strength);
        }
        bool lockNow = false;
        if (m_analysedFrames >= prm->colorfix.frames) {
            lockNow = true;
        } else if (m_totalSeenFrames >= m_hardCapFrames && m_analysedFrames > 0) {
            AddMessage(RGY_LOG_WARN,
                _T("variance guard rejected too many frames after %d input ")
                _T("(only %d accepted of %d target). Locking in early offsets.\n"),
                m_totalSeenFrames, m_analysedFrames, prm->colorfix.frames);
            lockNow = true;
        }
        if (lockNow) {
            m_offsetU = runningOffU;
            m_offsetV = runningOffV;
            m_analysisComplete = true;
            AddMessage(RGY_LOG_INFO, _T("analysis complete -- offsetU=%+.3f, offsetV=%+.3f (skipped %d flash frames)\n"),
                m_offsetU / maxVal, m_offsetV / maxVal, m_skippedFrames);
        }

        const float strengthFactor = std::min((float)m_analysedFrames / (float)prm->colorfix.frames, 1.0f);
        const int applyU = (int)std::lround((double)runningOffU * (double)strengthFactor);
        const int applyV = (int)std::lround((double)runningOffV * (double)strengthFactor);
        return runApplyUV(targetFrame, applyU, applyV, stream);
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
        const bool skip = colorfix_variance_guard(
            varY, m_rollingVarianceSum, m_rollingVarianceCount,
            RGY_CSP_BIT_DEPTH[m_cspRgb], prm->colorfix.varianceThreshold);
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
        bool lockNow = false;
        if (m_analysedFrames >= prm->colorfix.frames) {
            lockNow = true;
        } else if (m_totalSeenFrames >= m_hardCapFrames && m_analysedFrames > 0) {
            AddMessage(RGY_LOG_WARN,
                _T("variance guard rejected too many frames after %d input ")
                _T("(only %d accepted of %d target). Locking in early scales.\n"),
                m_totalSeenFrames, m_analysedFrames, prm->colorfix.frames);
            lockNow = true;
        }
        if (lockNow) {
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
    m_prescanUsed = false;
    m_hardCapFrames = 0;
    m_frameBuf.clear();
}
