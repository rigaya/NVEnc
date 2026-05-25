// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "convert_csp.h"
#include "NVEncFilterRtgmcBob.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int RTGMC_BOB_BLOCK_X = 32;
static const int RTGMC_BOB_BLOCK_Y = 8;

template<typename Type>
__device__ int readPix(
    const uint8_t *__restrict__ plane, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const Type *)(plane + y * pitch + x * sizeof(Type)));
}

// Mitchell & Netravali, "Reconstruction Filters in Computer Graphics", SIGGRAPH 1988.
__device__ float mitchell_netravali_weight(const float x, const float b, const float c) {
    const float ax = fabsf(x);
    if (ax < 1.0f) {
        return ((12.0f - 9.0f * b - 6.0f * c) * ax * ax * ax
              + (-18.0f + 12.0f * b + 6.0f * c) * ax * ax
              + (6.0f - 2.0f * b)) / 6.0f;
    }
    if (ax < 2.0f) {
        return ((-b - 6.0f * c) * ax * ax * ax
              + (6.0f * b + 30.0f * c) * ax * ax
              + (-12.0f * b - 48.0f * c) * ax
              + (8.0f * b + 24.0f * c)) / 6.0f;
    }
    return 0.0f;
}

template<typename Type>
__device__ int bobInterpolate(
    const uint8_t *__restrict__ src, const int ix, const int iy,
    const int pitch, const int width, const int height,
    const int preservedParity, const int phaseQuarter
) {
    const int sourceHeight = (height + 1 - preservedParity) >> 1;
    const float cropStart = 0.25f * (float)phaseQuarter;
    const float fieldPosRaw = cropStart - 0.25f + 0.5f * (float)iy;
    const float fieldPos = clamp(fieldPosRaw, 0.0f, (float)(sourceHeight - 1));
    int endTapField = (int)(fieldPosRaw + 2.0f);
    if (endTapField > sourceHeight - 1) {
        endTapField = sourceHeight - 1;
    }
    int firstTapField = endTapField - 3;
    if (firstTapField < 0) {
        firstTapField = 0;
    }
    float weightSum = 0.0f;
    float interp = 0.0f;

    for (int tap = 0; tap < 4; tap++) {
        const int tapField = firstTapField + tap;
        const int tapY = preservedParity + tapField * 2;
        const float tapWeight = mitchell_netravali_weight((float)tapField - fieldPos, 0.0f, 0.5f);
        interp += (float)readPix<Type>(src, ix, tapY, pitch, width, height) * tapWeight;
        weightSum += tapWeight;
    }
    return (int)(interp / weightSum + 0.5f);
}

template<typename Type>
__global__ void kernel_rtgmc_bob(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width,
    const int height,
    const int max_val,
    const int preservedParity,
    const int phaseQuarter
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    const int copySource = ((iy & 1) == preservedParity);
    const int value = copySource
        ? readPix<Type>(pSrc, ix, iy, srcPitch, width, height)
        : bobInterpolate<Type>(pSrc, ix, iy, srcPitch, width, height, preservedParity, phaseQuarter);

    Type *dstPix = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

tstring NVEncFilterParamRtgmcBob::print() const {
    const TCHAR *orderStr = _T("auto");
    if (order == RGYRtgmcBobFieldOrder::TFF) {
        orderStr = _T("tff");
    } else if (order == RGYRtgmcBobFieldOrder::BFF) {
        orderStr = _T("bff");
    }
    return strsprintf(_T("rtgmc-bob: order=%s"), orderStr);
}

NVEncFilterRtgmcBob::NVEncFilterRtgmcBob() :
    NVEncFilter(),
    m_buildOptions(),
    m_defaultTff(true) {
    m_name = _T("rtgmc-bob");
}

NVEncFilterRtgmcBob::~NVEncFilterRtgmcBob() {
    close();
}

RGY_ERR NVEncFilterRtgmcBob::checkParam(const std::shared_ptr<NVEncFilterParamRtgmcBob> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmcBob::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcBob>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP);
    pParam->baseFps *= 2;

    auto prmPrev = std::dynamic_pointer_cast<NVEncFilterParamRtgmcBob>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
        const int maxVal = (1 << bitDepth) - 1;
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D rtgmc_bob_block_x=%d -D rtgmc_bob_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            RTGMC_BOB_BLOCK_X, RTGMC_BOB_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Using CUDA kernel for rtgmc-bob: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
    }

    sts = AllocFrameBuf(prm->frameOut, 2);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;
    setFilterInfo(prm->print() + _T("\n                         auto-order-fallback=") + (m_defaultTff ? _T("tff") : _T("bff")));
    m_param = prm;
    return RGY_ERR_NONE;
}

bool NVEncFilterRtgmcBob::getInputTff(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcBob>(m_param);
    if (!prm) {
        return m_defaultTff;
    }
    if (prm->order == RGYRtgmcBobFieldOrder::TFF) {
        return true;
    }
    if (prm->order == RGYRtgmcBobFieldOrder::BFF) {
        return false;
    }
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

RGY_ERR NVEncFilterRtgmcBob::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int preservedParity,
    const int phaseQuarter,
    cudaStream_t stream) {
    const char *kernelName = "kernel_rtgmc_bob";
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);

        const dim3 blockSize(RTGMC_BOB_BLOCK_X, RTGMC_BOB_BLOCK_Y);
        const dim3 gridSize(divCeil(dstPlane.width, blockSize.x), divCeil(dstPlane.height, blockSize.y));
        const int bitDepth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
        if (bitDepth <= 8) {
            kernel_rtgmc_bob<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                (1 << bitDepth) - 1,
                preservedParity,
                phaseQuarter);
        } else {
            kernel_rtgmc_bob<uint16_t><<<gridSize, blockSize, 0, stream>>>(
                dstPlane.ptr[0], dstPlane.pitch[0],
                srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                (1 << bitDepth) - 1,
                preservedParity,
                phaseQuarter);
        }
        const auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            const auto err = err_to_rgy(cudaerr);
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterRtgmcBob::setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmcBob>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR NVEncFilterRtgmcBob::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool inputTff = getInputTff(pInputFrame);
    const int firstFieldParity = inputTff ? 0 : 1;
    const int secondFieldParity = inputTff ? 1 : 0;
    const int firstFieldPhaseQuarter = (firstFieldParity == 0) ? +1 : -1;
    const int secondFieldPhaseQuarter = (secondFieldParity == 0) ? +1 : -1;

    auto err = processFrame(&m_frameBuf[0]->frame, pInputFrame, firstFieldParity, firstFieldPhaseQuarter, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = processFrame(&m_frameBuf[1]->frame, pInputFrame, secondFieldParity, secondFieldPhaseQuarter, stream);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    for (int i = 0; i < 2; i++) {
        auto pOut = &m_frameBuf[i]->frame;
        pOut->picstruct = RGY_PICSTRUCT_FRAME;
        pOut->flags = RGY_FRAME_FLAG_NONE;
        ppOutputFrames[i] = pOut;
    }
    *pOutputFrameNum = 2;
    setBobTimestamp(pInputFrame, ppOutputFrames);
    return RGY_ERR_NONE;
}

void NVEncFilterRtgmcBob::close() {
    m_buildOptions.clear();
    m_frameBuf.clear();
}
