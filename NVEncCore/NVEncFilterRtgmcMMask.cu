#include "NVEncFilterRtgmcMMask.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "NVEncFilterDegrainCommon.h"
#include "rgy_cuda_util_kernel.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

namespace {
static constexpr int RTGMC_MMASK_BLOCK_X = 32;
static constexpr int RTGMC_MMASK_BLOCK_Y = 8;

using rtgmc_mmask_sad_t = RGYDegrainSAD;

static RGY_ERR rtgmcMMaskWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto &cudaEvent : degrainWaitEventList(waitEvents)) {
        const auto sts = err_to_rgy(cudaStreamWaitEvent(stream, cudaEvent, 0));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR recordCudaEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto sts = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}

template<typename T>
__host__ __device__ inline T rtgmcMMaskClamp(const T value, const T low, const T high) {
    return (value <= high) ? ((value >= low) ? value : low) : high;
}

template<typename TypePixel>
__device__ int rtgmcMMaskReadPix(
    const uint8_t *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = rtgmcMMaskClamp(x, 0, width - 1);
    y = rtgmcMMaskClamp(y, 0, height - 1);
    return (int)(*(const TypePixel *)(src + y * pitch + x * sizeof(TypePixel)));
}

template<typename TypePixel>
__device__ void rtgmcMMaskWritePix(
    uint8_t *dst, int x, int y, const int pitch, const int value, const int maxVal
) {
    TypePixel *dstPix = (TypePixel *)(dst + y * pitch + x * sizeof(TypePixel));
    dstPix[0] = (TypePixel)rtgmcMMaskClamp(value, 0, maxVal);
}

__device__ int rtgmcMMaskBlockIndex(
    const int x,
    const int y,
    const int blocksX,
    const int blocksY,
    const int step
) {
    const int clampedStep = max(step, 1);
    const int bx = min(x / clampedStep, blocksX - 1);
    const int by = min(y / clampedStep, blocksY - 1);
    return by * blocksX + bx;
}

__device__ float rtgmcMMaskSadWeight(
    const rtgmc_mmask_sad_t *sad,
    const int block,
    const int blockSize,
    const int temporalDirections,
    const int bitdepth,
    const float ml,
    const float gamma,
    const bool usePow
) {
    const uint32_t sadValue = sad[block * temporalDirections].sad;
    const float bitScale = (float)(1 << max(bitdepth - 8, 0));
    const float denom = max(ml * (float)(blockSize * blockSize) * bitScale, 1.0f);
    const float normalized = rtgmcMMaskClamp(((float)sadValue * 4.0f) / denom, 0.0f, 1.0f);
    return usePow ? powf(normalized, gamma) : normalized;
}

template<typename TypePixel>
__global__ void kernel_rtgmc_mmask_copy(
    TypePixel *pDst, const int dstPitch,
    const TypePixel *pSrc, const int srcPitch,
    const int width,
    const int height,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) {
        return;
    }
    const int value = rtgmcMMaskReadPix<TypePixel>((const uint8_t *)pSrc, ix, iy, srcPitch, width, height);
    rtgmcMMaskWritePix<TypePixel>((uint8_t *)pDst, ix, iy, dstPitch, value, maxVal);
}

template<typename TypePixel>
__global__ void kernel_rtgmc_mmask_blend_y(
    TypePixel *pDst, const int dstPitch,
    const TypePixel *pSource, const int sourcePitch,
    const TypePixel *pEdi, const int ediPitch,
    const int width,
    const int height,
    const rtgmc_mmask_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int coveredWidth,
    const int coveredHeight,
    const int step,
    const int blockSize,
    const int temporalDirections,
    const int bitdepth,
    const float ml,
    const float gamma,
    const bool usePow,
    const int maxVal
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) {
        return;
    }

    const int sourcePix = rtgmcMMaskReadPix<TypePixel>((const uint8_t *)pSource, ix, iy, sourcePitch, width, height);
    const int ediPix = rtgmcMMaskReadPix<TypePixel>((const uint8_t *)pEdi, ix, iy, ediPitch, width, height);
    if (ix >= coveredWidth || iy >= coveredHeight || blocksX <= 0 || blocksY <= 0 || temporalDirections <= 0) {
        rtgmcMMaskWritePix<TypePixel>((uint8_t *)pDst, ix, iy, dstPitch, sourcePix, maxVal);
        return;
    }

    const int block = rtgmcMMaskBlockIndex(ix, iy, blocksX, blocksY, step);
    const float weight = rtgmcMMaskSadWeight(sad, block, blockSize, temporalDirections, bitdepth, ml, gamma, usePow);
    const int value = (int)rintf((float)sourcePix + ((float)ediPix - (float)sourcePix) * weight);
    rtgmcMMaskWritePix<TypePixel>((uint8_t *)pDst, ix, iy, dstPitch, value, maxVal);
}

template<typename TypePixel>
RGY_ERR processFrameTyped(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame,
    const RGYDegrainAnalyzeResult &analyzeResult, const NVEncFilterParamRtgmcMMask &prm,
    cudaStream_t stream, RGYCudaEvent *event) {
    const int planes = RGY_CSP_PLANES[pSourceFrame->csp];
    const int bitdepth = RGY_CSP_BIT_DEPTH[pSourceFrame->csp];
    const int maxVal = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const bool usePow = std::abs(prm.gamma - 1.0) > 1.0e-6;
    const auto *sad = reinterpret_cast<const rtgmc_mmask_sad_t *>(analyzeResult.sad->ptr);
    const dim3 blockSize(RTGMC_MMASK_BLOCK_X, RTGMC_MMASK_BLOCK_Y);

    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const auto ediPlane = getPlane(pEdiFrame, (RGY_PLANE)iplane);

        const bool processLuma = iplane == 0;
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        if (processLuma) {
            kernel_rtgmc_mmask_blend_y<TypePixel><<<gridSize, blockSize, 0, stream>>>(
                (TypePixel *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const TypePixel *)srcPlane.ptr[0], srcPlane.pitch[0],
                (const TypePixel *)ediPlane.ptr[0], ediPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                sad,
                analyzeResult.layout.blocksX,
                analyzeResult.layout.blocksY,
                analyzeResult.layout.coveredWidth,
                analyzeResult.layout.coveredHeight,
                analyzeResult.layout.step,
                analyzeResult.layout.blockSize,
                analyzeResult.layout.temporalDirections,
                bitdepth,
                (float)prm.ml,
                (float)prm.gamma,
                usePow,
                maxVal);
        } else {
            kernel_rtgmc_mmask_copy<TypePixel><<<gridSize, blockSize, 0, stream>>>(
                (TypePixel *)dstPlane.ptr[0], dstPlane.pitch[0],
                (const TypePixel *)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height, maxVal);
        }
        const auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            return err_to_rgy(cudaerr);
        }
    }

    copyFramePropWithoutRes(pOutputFrame, pEdiFrame);
    return recordCudaEvent(stream, event);
}
}

NVEncFilterParamRtgmcMMask::NVEncFilterParamRtgmcMMask() :
    kind(1),
    time(100),
    ml(1.0),
    gamma(1.0) {
}

tstring NVEncFilterParamRtgmcMMask::print() const {
    return strsprintf(_T("rtgmc-mmask: kind=%d time=%d ml=%.3f gamma=%.3f"),
        kind, time, ml, gamma);
}

NVEncFilterRtgmcMMask::NVEncFilterRtgmcMMask() :
    NVEncFilter(),
    m_buildOptions(),
    m_useKernel(false) {
    m_name = _T("rtgmc-mmask");
}

NVEncFilterRtgmcMMask::~NVEncFilterRtgmcMMask() {
    close();
}

RGY_ERR NVEncFilterRtgmcMMask::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcMMask> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->kind != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask supports only kind=1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->time != 100) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask supports only time=100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->ml <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask ml must be positive.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->gamma <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask gamma must be positive.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcMMask::buildKernel(const std::shared_ptr<NVEncFilterParamRtgmcMMask> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int usePow = (std::abs(prm->gamma - 1.0) > 1.0e-6) ? 1 : 0;
    m_buildOptions = strsprintf(
        "bit_depth=%d max_val=%d rtgmc_mmask_block_x=%d rtgmc_mmask_block_y=%d use_pow=%d",
        bitdepth, pixelMax, RTGMC_MMASK_BLOCK_X, RTGMC_MMASK_BLOCK_Y, usePow);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcMMask::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcMMask>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);
    if (m_useKernel) {
        sts = buildKernel(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to prepare rtgmc-mmask kernel options.\n"));
            return sts;
        }
    } else {
        m_buildOptions = strsprintf("bit_depth=%d fallback_copy=1", RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
    }

    sts = AllocFrameBuf(prm->frameOut, 3);
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

RGY_ERR NVEncFilterRtgmcMMask::checkAnalyzeResult(const RGYDegrainAnalyzeResult &analyzeResult, const RGYFrameInfo *pSourceFrame) {
    if (!analyzeResult.valid() || !analyzeResult.sad || !analyzeResult.sad->ptr || !analyzeResult.mv || !analyzeResult.mv->ptr) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask requires a valid degrain SAD analysis result.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (analyzeResult.layout.blockSize <= 0 || analyzeResult.layout.blockCount() <= 0 || analyzeResult.layout.temporalDirections <= 0
        || analyzeResult.layout.sadCount() != analyzeResult.layout.blockCount() * (size_t)analyzeResult.layout.temporalDirections) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask has invalid degrain SAD layout.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (analyzeResult.mv->nSize != rgy_degrain_mv_bytes(analyzeResult.layout)
        || analyzeResult.sad->nSize != rgy_degrain_sad_bytes(analyzeResult.layout)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask degrain SAD buffer size mismatch.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pSourceFrame
        && (analyzeResult.layout.coveredWidth > pSourceFrame->width || analyzeResult.layout.coveredHeight > pSourceFrame->height)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask degrain SAD layout exceeds frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcMMask::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame,
    const RGYDegrainAnalyzeResult &analyzeResult, const NVEncFilterParamRtgmcMMask &prm,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    std::vector<RGYCudaEvent> waitWithAnalysis = wait_events;
    if (analyzeResult.event() != nullptr) {
        waitWithAnalysis.push_back(analyzeResult.event);
    }
    auto err = rtgmcMMaskWaitEvents(stream, waitWithAnalysis);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const auto sourceMemcpyKind = getCudaMemcpyKind(pSourceFrame->mem_type, pOutputFrame->mem_type);
    const auto ediMemcpyKind = getCudaMemcpyKind(pEdiFrame->mem_type, pOutputFrame->mem_type);
    if (sourceMemcpyKind == cudaMemcpyDeviceToDevice && ediMemcpyKind == cudaMemcpyDeviceToDevice) {
        const int bitdepth = RGY_CSP_BIT_DEPTH[pSourceFrame->csp];
        if (bitdepth <= 8) {
            return processFrameTyped<unsigned char>(pOutputFrame, pSourceFrame, pEdiFrame, analyzeResult, prm, stream, event);
        }
        return processFrameTyped<unsigned short>(pOutputFrame, pSourceFrame, pEdiFrame, analyzeResult, prm, stream, event);
    }

    auto pSourceTmp = &m_frameBuf[1]->frame;
    auto pEdiTmp = &m_frameBuf[2]->frame;
    err = copyFrameAsync(pSourceTmp, pSourceFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-mmask source frame: %s.\n"), get_err_mes(err));
        return err;
    }
    err = copyFrameAsync(pEdiTmp, pEdiFrame, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-mmask edi frame: %s.\n"), get_err_mes(err));
        return err;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[pSourceFrame->csp];
    if (bitdepth <= 8) {
        return processFrameTyped<unsigned char>(pOutputFrame, pSourceTmp, pEdiTmp, analyzeResult, prm, stream, event);
    }
    return processFrameTyped<unsigned short>(pOutputFrame, pSourceTmp, pEdiTmp, analyzeResult, prm, stream, event);
}

RGY_ERR NVEncFilterRtgmcMMask::run_filter(const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame, const RGYDegrainAnalyzeResult &analyzeResult,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events,
    RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pSourceFrame || !pSourceFrame->ptr[0] || !pEdiFrame || !pEdiFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (pSourceFrame->csp != pEdiFrame->csp || pSourceFrame->width != pEdiFrame->width || pSourceFrame->height != pEdiFrame->height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask source and edi frames must match csp and resolution.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcMMask>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = checkAnalyzeResult(analyzeResult, pSourceFrame);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        err = processFrame(&pOutFrame->frame, pSourceFrame, pEdiFrame, analyzeResult, *prm, stream, wait_events, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    } else {
        err = rtgmcMMaskWaitEvents(stream, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = copyFrameAsync(ppOutputFrames[0], pSourceFrame, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(err));
            return err;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], pEdiFrame);
        err = recordCudaEvent(stream, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcMMask::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    RGYDegrainAnalyzeResult analyzeResult;
    return run_filter(pInputFrame, pInputFrame, analyzeResult, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

void NVEncFilterRtgmcMMask::close() {
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
}
