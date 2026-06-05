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
//
// AURORA deflicker -- van Roosmalen 1999 PhD thesis,
// "Restoration of archived film and video".

#include <algorithm>
#include <cmath>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterDeflicker.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int DEFLICKER_REDUCE_X = 32;
static const int DEFLICKER_REDUCE_Y = 8;
static const int DEFLICKER_REDUCE_THREADS = DEFLICKER_REDUCE_X * DEFLICKER_REDUCE_Y;
static const double DEFLICKER_SIGMA_EPS_8BIT = 0.5;

template<typename Type>
__global__ void kernel_deflicker_reduce(const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height, long long *__restrict__ pSum, long long *__restrict__ pSumSq) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int lidx = threadIdx.y * blockDim.x + threadIdx.x;

    long long val = 0;
    long long valSq = 0;
    if (ix < width && iy < height) {
        const int p = (int)(*(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type)));
        val = (long long)p;
        valSq = (long long)p * (long long)p;
    }

    __shared__ long long sSum[DEFLICKER_REDUCE_THREADS];
    __shared__ long long sSumSq[DEFLICKER_REDUCE_THREADS];
    sSum[lidx] = val;
    sSumSq[lidx] = valSq;
    __syncthreads();

    for (int stride = DEFLICKER_REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (lidx < stride) {
            sSum[lidx] += sSum[lidx + stride];
            sSumSq[lidx] += sSumSq[lidx + stride];
        }
        __syncthreads();
    }
    if (lidx == 0) {
        const int wgIndex = blockIdx.y * gridDim.x + blockIdx.x;
        pSum[wgIndex] = sSum[0];
        pSumSq[wgIndex] = sSumSq[0];
    }
}

template<typename Type>
__global__ void kernel_deflicker_apply(const uint8_t *__restrict__ pSrc, const int srcPitch,
    uint8_t *__restrict__ pDst, const int dstPitch, const int width, const int height,
    const float mult, const float add, const float strength, const int isChroma,
    const int bitDepth, const int maxVal) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const float srcF = (float)(*(const Type *)(pSrc + iy * srcPitch + ix * sizeof(Type)));
    float corrected;
    if (isChroma) {
        const float mid = (float)(1 << (bitDepth - 1));
        corrected = mult * (srcF - mid) + mid;
    } else {
        corrected = mult * srcF + add;
    }
    float result = strength * corrected + (1.0f - strength) * srcF;
    result = fminf(fmaxf(result, 0.0f), (float)maxVal);

    Type *dst = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = (Type)(result + 0.5f);
}

template<typename Type>
static RGY_ERR deflicker_reduce_plane(const RGYFrameInfo *pPlane,
    long long *sumBuf, long long *sumSqBuf, int64_t *sumHost, int64_t *sumSqHost,
    double& meanOut, double& stddevOut, cudaStream_t stream) {
    const int wgX = divCeil(pPlane->width, DEFLICKER_REDUCE_X);
    const int wgY = divCeil(pPlane->height, DEFLICKER_REDUCE_Y);
    const size_t wgCount = (size_t)wgX * (size_t)wgY;
    const size_t bytesUsed = wgCount * sizeof(int64_t);

    dim3 blockSize(DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y);
    dim3 gridSize(wgX, wgY);
    kernel_deflicker_reduce<Type><<<gridSize, blockSize, 0, stream>>>(
        (const uint8_t *)pPlane->ptr[0], pPlane->pitch[0], pPlane->width, pPlane->height, sumBuf, sumSqBuf);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    cudaerr = cudaMemcpyAsync(sumHost, sumBuf, bytesUsed, cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    cudaerr = cudaMemcpyAsync(sumSqHost, sumSqBuf, bytesUsed, cudaMemcpyDeviceToHost, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    cudaerr = cudaStreamSynchronize(stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    int64_t sum = 0;
    int64_t sumSq = 0;
    for (size_t i = 0; i < wgCount; i++) {
        sum += sumHost[i];
        sumSq += sumSqHost[i];
    }
    const double n = (double)((int64_t)pPlane->width * (int64_t)pPlane->height);
    const double mean = (double)sum / n;
    double variance = ((double)sumSq / n) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    meanOut = mean;
    stddevOut = std::sqrt(variance);
    return RGY_ERR_NONE;
}

template<typename Type>
static RGY_ERR deflicker_apply_plane(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcPlane,
    const float mult, const float add, const float blend, const int isChroma,
    const int bitDepth, cudaStream_t stream) {
    const int maxVal = (1 << bitDepth) - 1;
    dim3 blockSize(DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y);
    dim3 gridSize(divCeil(pDstPlane->width, blockSize.x), divCeil(pDstPlane->height, blockSize.y));
    kernel_deflicker_apply<Type><<<gridSize, blockSize, 0, stream>>>(
        (const uint8_t *)pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        (uint8_t *)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height, mult, add, blend, isChroma, bitDepth, maxVal);
    return err_to_rgy(cudaGetLastError());
}

NVEncFilterDeflicker::NVEncFilterDeflicker() :
    m_sumBuf(),
    m_sumSqBuf(),
    m_sumHost(),
    m_sumSqHost(),
    m_statsBufWGCount(0),
    m_rollingMeans(),
    m_rollingSigmas(),
    m_prevMult(1.0),
    m_prevAdd(0.0),
    m_haveDamping(false),
    m_skippedSceneFrames(0) {
    m_name = _T("deflicker");
}

NVEncFilterDeflicker::~NVEncFilterDeflicker() {
    close();
}

RGY_ERR NVEncFilterDeflicker::checkParam(const std::shared_ptr<NVEncFilterParamDeflicker> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp];
    if (rgy_chromafmt_is_rgb(chromaFormat)) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker supports YUV or monochrome formats only: %s.\n"), RGY_CSP_NAMES[prm->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->deflicker.chroma && RGY_CSP_PLANES[prm->frameIn.csp] < 2) {
        prm->deflicker.chroma = false;
        AddMessage(RGY_LOG_WARN, _T("deflicker chroma processing requires chroma planes; disabled for %s.\n"), RGY_CSP_NAMES[prm->frameIn.csp]);
    }
    if (prm->deflicker.strength < 0.0f || 1.0f < prm->deflicker.strength) {
        prm->deflicker.strength = clamp(prm->deflicker.strength, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->deflicker.damping < 0.0f || 1.0f < prm->deflicker.damping) {
        prm->deflicker.damping = clamp(prm->deflicker.damping, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("damping should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->deflicker.scene_threshold < 0.5f || 5.0f < prm->deflicker.scene_threshold) {
        prm->deflicker.scene_threshold = clamp(prm->deflicker.scene_threshold, 0.5f, 5.0f);
        AddMessage(RGY_LOG_WARN, _T("scene_threshold should be in range of %.1f - %.1f.\n"), 0.5f, 5.0f);
    }
    if (prm->deflicker.frames < 5 || 300 < prm->deflicker.frames) {
        prm->deflicker.frames = clamp(prm->deflicker.frames, 5, 300);
        AddMessage(RGY_LOG_WARN, _T("frames should be in range of %d - %d.\n"), 5, 300);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDeflicker::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDeflicker>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) return sts;

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, prm->deflicker.predictor ? 2 : 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int wgX = divCeil(prm->frameOut.width, DEFLICKER_REDUCE_X);
    const int wgY = divCeil(prm->frameOut.height, DEFLICKER_REDUCE_Y);
    const size_t wgCount = (size_t)wgX * (size_t)wgY;
    if (!m_sumBuf || m_statsBufWGCount != wgCount) {
        m_sumBuf = std::make_unique<CUMemBuf>(wgCount * sizeof(int64_t));
        m_sumSqBuf = std::make_unique<CUMemBuf>(wgCount * sizeof(int64_t));
        if (   RGY_ERR_NONE != (sts = m_sumBuf->alloc())
            || RGY_ERR_NONE != (sts = m_sumSqBuf->alloc())) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate deflicker stats buffers: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_sumHost.assign(wgCount, 0);
        m_sumSqHost.assign(wgCount, 0);
        m_statsBufWGCount = wgCount;
    }

    setFilterInfo(pParam->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamDeflicker::print() const {
    return deflicker.print();
}

RGY_ERR NVEncFilterDeflicker::computePlaneStats(const RGYFrameInfo *pPlane, double& meanOut, double& stddevOut, cudaStream_t stream) {
    static const std::map<RGY_DATA_TYPE, decltype(deflicker_reduce_plane<uint8_t>)*> reduceList = {
        { RGY_DATA_TYPE_U8,  deflicker_reduce_plane<uint8_t> },
        { RGY_DATA_TYPE_U16, deflicker_reduce_plane<uint16_t> }
    };
    if (reduceList.count(RGY_CSP_DATA_TYPE[pPlane->csp]) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pPlane->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return reduceList.at(RGY_CSP_DATA_TYPE[pPlane->csp])(pPlane,
        (long long *)m_sumBuf->ptr, (long long *)m_sumSqBuf->ptr,
        m_sumHost.data(), m_sumSqHost.data(), meanOut, stddevOut, stream);
}

RGY_ERR NVEncFilterDeflicker::runApply(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcPlane,
    float mult, float add, float blend, int is_chroma, cudaStream_t stream) {
    static const std::map<RGY_DATA_TYPE, decltype(deflicker_apply_plane<uint8_t>)*> applyList = {
        { RGY_DATA_TYPE_U8,  deflicker_apply_plane<uint8_t> },
        { RGY_DATA_TYPE_U16, deflicker_apply_plane<uint16_t> }
    };
    if (applyList.count(RGY_CSP_DATA_TYPE[pSrcPlane->csp]) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pSrcPlane->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return applyList.at(RGY_CSP_DATA_TYPE[pSrcPlane->csp])(
        pDstPlane, pSrcPlane, mult, add, blend, is_chroma, RGY_CSP_BIT_DEPTH[pSrcPlane->csp], stream);
}

RGY_ERR NVEncFilterDeflicker::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDeflicker>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, pOut->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker only supports device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker does not support csp conversion.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const double sigmaEps = DEFLICKER_SIGMA_EPS_8BIT * (double)(1 << std::max(0, bitDepth - 8));
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];

    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    double meanIn = 0.0;
    double sigmaIn = 0.0;
    sts = computePlaneStats(&planeSrcY, meanIn, sigmaIn, stream);
    if (sts != RGY_ERR_NONE) return sts;

    bool sceneChange = false;
    if (m_rollingMeans.size() >= 5) {
        double sumM = 0.0;
        for (double v : m_rollingMeans) sumM += v;
        const double rollMeanOfMeans = sumM / (double)m_rollingMeans.size();
        double sumSqDev = 0.0;
        for (double v : m_rollingMeans) {
            const double d = v - rollMeanOfMeans;
            sumSqDev += d * d;
        }
        const double rollStdOfMeans = std::sqrt(sumSqDev / (double)m_rollingMeans.size());
        const double absDiff = std::abs(meanIn - rollMeanOfMeans);
        const double absFloor = 0.10 * (double)((1 << bitDepth) - 1);
        if (rollStdOfMeans > 0.0
            && absDiff > (double)prm->deflicker.scene_threshold * rollStdOfMeans
            && absDiff > absFloor) {
            sceneChange = true;
            m_skippedSceneFrames++;
        }
    }

    if (!sceneChange) {
        m_rollingMeans.push_back(meanIn);
        m_rollingSigmas.push_back(sigmaIn);
        while ((int)m_rollingMeans.size() > prm->deflicker.frames) {
            m_rollingMeans.pop_front();
            m_rollingSigmas.pop_front();
        }
    }

    const bool haveReference = !m_rollingMeans.empty();
    double muRef = meanIn;
    double sigmaRef = sigmaIn;
    if (haveReference) {
        double sM = 0.0;
        double sS = 0.0;
        for (double v : m_rollingMeans) sM += v;
        for (double v : m_rollingSigmas) sS += v;
        muRef = sM / (double)m_rollingMeans.size();
        sigmaRef = sS / (double)m_rollingSigmas.size();
    }

    double multRaw = 1.0;
    double addRaw = 0.0;
    if (haveReference && !sceneChange) {
        const double sigmaDenom = std::max(sigmaIn, sigmaEps);
        multRaw = std::sqrt(sigmaRef / sigmaDenom);
        addRaw = muRef - multRaw * meanIn;
    }

    double multEff = multRaw;
    double addEff = addRaw;
    if (m_haveDamping && !sceneChange) {
        const double d = (double)prm->deflicker.damping;
        multEff = d * m_prevMult + (1.0 - d) * multRaw;
        addEff = d * m_prevAdd + (1.0 - d) * addRaw;
    }

    auto planeDstY = getPlane(pOut, RGY_PLANE_Y);
    if (sceneChange || !haveReference) {
        sts = copyPlaneAsync(&planeDstY, &planeSrcY, stream);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (!prm->deflicker.predictor) {
        sts = runApply(&planeDstY, &planeSrcY, (float)multEff, (float)addEff, prm->deflicker.strength, 0, stream);
        if (sts != RGY_ERR_NONE) return sts;
    } else {
        auto planeInter = getPlane(&m_frameBuf[1]->frame, RGY_PLANE_Y);
        sts = runApply(&planeInter, &planeSrcY, (float)multEff, (float)addEff, 1.0f, 0, stream);
        if (sts != RGY_ERR_NONE) return sts;

        double mu1 = 0.0;
        double sigma1 = 0.0;
        sts = computePlaneStats(&planeInter, mu1, sigma1, stream);
        if (sts != RGY_ERR_NONE) return sts;

        const double sigmaDenom2 = std::max(sigma1, sigmaEps);
        const double multRefine = std::sqrt(sigmaRef / sigmaDenom2);
        const double addRefine = muRef - multRefine * mu1;
        sts = runApply(&planeDstY, &planeInter, (float)multRefine, (float)addRefine, prm->deflicker.strength, 0, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    if (!sceneChange && haveReference) {
        m_prevMult = multEff;
        m_prevAdd = addEff;
        m_haveDamping = true;
    }

    for (int i = 1; i < planes; i++) {
        const auto planeSrcC = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeDstC = getPlane(pOut, (RGY_PLANE)i);
        if (sceneChange || !haveReference || !prm->deflicker.chroma) {
            sts = copyPlaneAsync(&planeDstC, &planeSrcC, stream);
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            sts = runApply(&planeDstC, &planeSrcC, (float)multEff, 0.0f, prm->deflicker.strength, 1, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }

    if (sceneChange) {
        AddMessage(RGY_LOG_DEBUG,
            _T("deflicker: scene change at frame %d (mean=%.1f vs rolling=%.1f); passthrough\n"),
            pInputFrame->inputFrameId, meanIn, m_rollingMeans.empty() ? 0.0 : m_rollingMeans.back());
    } else if (haveReference) {
        AddMessage(RGY_LOG_DEBUG,
            _T("deflicker: frame %d mu_in=%.2f sigma_in=%.2f mult=%.4f add=%.2f\n"),
            pInputFrame->inputFrameId, meanIn, sigmaIn, multEff, addEff);
    }

    pOut->timestamp = pInputFrame->timestamp;
    pOut->duration = pInputFrame->duration;
    pOut->inputFrameId = pInputFrame->inputFrameId;
    pOut->picstruct = pInputFrame->picstruct;
    pOut->flags = pInputFrame->flags;
    ppOutputFrames[0] = pOut;
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

void NVEncFilterDeflicker::close() {
    if (m_skippedSceneFrames > 0) {
        AddMessage(RGY_LOG_INFO, _T("deflicker: skipped %d scene-change frames during analysis.\n"), m_skippedSceneFrames);
    }
    m_sumBuf.reset();
    m_sumSqBuf.reset();
    m_sumHost.clear();
    m_sumSqHost.clear();
    m_statsBufWGCount = 0;
    m_rollingMeans.clear();
    m_rollingSigmas.clear();
    m_prevMult = 1.0;
    m_prevAdd = 0.0;
    m_haveDamping = false;
    m_skippedSceneFrames = 0;
    m_frameBuf.clear();
}
