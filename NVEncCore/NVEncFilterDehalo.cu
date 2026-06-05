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
#include "NVEncFilterDehalo.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DEHALO_BLOCK_X = 32;
static const int DEHALO_BLOCK_Y = 8;

template<typename Type>
__device__ __forceinline__ int dehalo_read_pix_clamp(const RGYFrameInfo frame, int x, int y) {
    x = clamp(x, 0, frame.width - 1);
    y = clamp(y, 0, frame.height - 1);
    const auto ptr = (const Type *)((const uint8_t *)frame.ptr[0] + y * frame.pitch[0] + x * sizeof(Type));
    return (int)ptr[0];
}

template<typename Type>
__global__ void kernel_dehalo_expand(const RGYFrameInfo src, RGYFrameInfo dst, const float rx, const float ry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;

    const int irx = (int)ceilf(rx);
    const int iry = (int)ceilf(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = dehalo_read_pix_clamp<Type>(src, x, y);
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dehalo_read_pix_clamp<Type>(src, x + dx, y + dy);
            if (v > m) m = v;
        }
    }

    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type>
__global__ void kernel_dehalo_inpand(const RGYFrameInfo src, RGYFrameInfo dst, const float rx, const float ry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;

    const int irx = (int)ceilf(rx);
    const int iry = (int)ceilf(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = dehalo_read_pix_clamp<Type>(src, x, y);
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dehalo_read_pix_clamp<Type>(src, x + dx, y + dy);
            if (v < m) m = v;
        }
    }

    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_mask(const RGYFrameInfo src, const RGYFrameInfo expanded, const RGYFrameInfo inpand,
    RGYFrameInfo mask, const int loScaled, const int hiScaled) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int s = dehalo_read_pix_clamp<Type>(src, x, y);
    const int e = dehalo_read_pix_clamp<Type>(expanded, x, y);
    const int i = dehalo_read_pix_clamp<Type>(inpand, x, y);
    const int range = e - i;

    int abs_diff = 0;
    if (range > 0) {
        long long num = (long long)(s - i) * (long long)max_val;
        int v = (int)(num / (long long)range);
        v = clamp(v, 0, max_val);
        abs_diff = v;
    }

    int m = 0;
    if (hiScaled > loScaled) {
        long long num = (long long)(abs_diff - loScaled) * (long long)max_val;
        int v = (int)(num / (long long)(hiScaled - loScaled));
        m = clamp(v, 0, max_val);
    } else {
        m = (abs_diff >= loScaled) ? max_val : 0;
    }

    auto maskPix = (Type *)((uint8_t *)mask.ptr[0] + y * mask.pitch[0] + x * sizeof(Type));
    maskPix[0] = (Type)m;
}

template<typename Type, int bit_depth>
__global__ void kernel_dehalo_apply(const RGYFrameInfo src, const RGYFrameInfo expanded, const RGYFrameInfo inpand,
    const RGYFrameInfo mask, RGYFrameInfo dst, const float darkstr, const float brightstr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const float s = (float)dehalo_read_pix_clamp<Type>(src, x, y);
    const float e = (float)dehalo_read_pix_clamp<Type>(expanded, x, y);
    const float i = (float)dehalo_read_pix_clamp<Type>(inpand, x, y);
    const float m = (float)dehalo_read_pix_clamp<Type>(mask, x, y);
    const float mn = m / (float)max_val;

    float r = s - mn * darkstr * (s - i) + mn * brightstr * (e - s);
    r = clamp(r, 0.0f, (float)max_val);

    auto dstPix = (Type *)((uint8_t *)dst.ptr[0] + y * dst.pitch[0] + x * sizeof(Type));
    dstPix[0] = (Type)(int)(r + 0.5f);
}

template<typename Type, int bit_depth>
static RGY_ERR dehalo_process_y_typed(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYFrameInfo *pExpanded, RGYFrameInfo *pInpand, RGYFrameInfo *pMask,
    const VppDehalo& prm, const int loScaled, const int hiScaled, cudaStream_t stream) {
    dim3 blockSize(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    dim3 gridSize(divCeil(pInputFrame->width, blockSize.x), divCeil(pInputFrame->height, blockSize.y));

    kernel_dehalo_expand<Type><<<gridSize, blockSize, 0, stream>>>(*pInputFrame, *pExpanded, prm.rx, prm.ry);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_inpand<Type><<<gridSize, blockSize, 0, stream>>>(*pInputFrame, *pInpand, prm.rx, prm.ry);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_mask<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pInputFrame, *pExpanded, *pInpand, *pMask, loScaled, hiScaled);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehalo_apply<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(*pInputFrame, *pExpanded, *pInpand, *pMask, *pOutputFrame, prm.darkstr, prm.brightstr);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    return RGY_ERR_NONE;
}

static RGY_ERR dehalo_process_y(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYFrameInfo *pExpanded, RGYFrameInfo *pInpand, RGYFrameInfo *pMask,
    const VppDehalo& prm, const int loScaled, const int hiScaled, cudaStream_t stream) {
    if (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) {
        return dehalo_process_y_typed<uint16_t, 16>(pOutputFrame, pInputFrame, pExpanded, pInpand, pMask, prm, loScaled, hiScaled, stream);
    } else {
        return dehalo_process_y_typed<uint8_t, 8>(pOutputFrame, pInputFrame, pExpanded, pInpand, pMask, prm, loScaled, hiScaled, stream);
    }
}

NVEncFilterDehalo::NVEncFilterDehalo() :
    m_resizeUp(),
    m_resizeDown(),
    m_supersampled(),
    m_expanded(),
    m_inpand(),
    m_mask(),
    m_corrected(),
    m_ssW(0),
    m_ssH(0),
    m_ssActive(false) {
    m_name = _T("dehalo");
}

NVEncFilterDehalo::~NVEncFilterDehalo() {
    close();
}

RGY_ERR NVEncFilterDehalo::checkParam(const std::shared_ptr<NVEncFilterParamDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.rx >= 0.5f && prm->dehalo.rx <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rx=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.rx);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ry >= 0.5f && prm->dehalo.ry <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ry=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.ry);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.darkstr >= 0.0f && prm->dehalo.darkstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.darkstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.brightstr >= 0.0f && prm->dehalo.brightstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid brightstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.brightstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.lowsens < 0 || prm->dehalo.lowsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid lowsens=%d: must be in [0, 100].\n"), prm->dehalo.lowsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.highsens < 0 || prm->dehalo.highsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid highsens=%d: must be in [0, 100].\n"), prm->dehalo.highsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ss >= 1.0f && prm->dehalo.ss <= 4.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.2f: must be in [1.0, 4.0].\n"), prm->dehalo.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDehalo::allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label) {
    if (!frame
        || frame->frame.width != frameInfo.width
        || frame->frame.height != frameInfo.height
        || frame->frame.csp != frameInfo.csp) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo %s buffer: %s.\n"), label, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDehalo::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehalo>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_ssActive = prm->dehalo.ss > 1.0f + 1e-6f;
    if (m_ssActive) {
        m_ssW = ((int)std::lround(prm->frameIn.width  * prm->dehalo.ss) + 1) & ~1;
        m_ssH = ((int)std::lround(prm->frameIn.height * prm->dehalo.ss) + 1) & ~1;
    } else {
        m_ssW = prm->frameIn.width;
        m_ssH = prm->frameIn.height;
    }

    RGYFrameInfo workInfo = prm->frameIn;
    workInfo.width = m_ssW;
    workInfo.height = m_ssH;
    sts = allocWorkFrame(m_expanded, workInfo, _T("expanded"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_inpand, workInfo, _T("inpand"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_mask, workInfo, _T("mask"));
    if (sts != RGY_ERR_NONE) return sts;

    if (m_ssActive) {
        sts = allocWorkFrame(m_supersampled, workInfo, _T("supersampled"));
        if (sts != RGY_ERR_NONE) return sts;
        sts = allocWorkFrame(m_corrected, workInfo, _T("corrected"));
        if (sts != RGY_ERR_NONE) return sts;

        const auto bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
        const auto lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;

        auto prmUp = std::make_shared<NVEncFilterParamResize>();
        prmUp->frameIn = prm->frameIn;
        prmUp->frameIn.csp = lumaCsp;
        prmUp->frameOut = prm->frameIn;
        prmUp->frameOut.csp = lumaCsp;
        prmUp->frameOut.width = m_ssW;
        prmUp->frameOut.height = m_ssH;
        prmUp->interp = RGY_VPP_RESIZE_SPLINE36;
        prmUp->baseFps = prm->baseFps;
        prmUp->bOutOverwrite = false;
        m_resizeUp = std::make_unique<NVEncFilterResize>();
        sts = m_resizeUp->init(prmUp, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo upscale sub-filter: %s.\n"), get_err_mes(sts));
            return sts;
        }

        auto prmDown = std::make_shared<NVEncFilterParamResize>();
        prmDown->frameIn = prm->frameIn;
        prmDown->frameIn.csp = lumaCsp;
        prmDown->frameIn.width = m_ssW;
        prmDown->frameIn.height = m_ssH;
        prmDown->frameOut = prm->frameOut;
        prmDown->frameOut.csp = lumaCsp;
        prmDown->interp = RGY_VPP_RESIZE_SPLINE36;
        prmDown->baseFps = prm->baseFps;
        prmDown->bOutOverwrite = false;
        m_resizeDown = std::make_unique<NVEncFilterResize>();
        sts = m_resizeDown->init(prmDown, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo downscale sub-filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        m_supersampled.reset();
        m_corrected.reset();
        m_resizeUp.reset();
        m_resizeDown.reset();
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

tstring NVEncFilterParamDehalo::print() const {
    return dehalo.print();
}

RGY_ERR NVEncFilterDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehalo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int loScaled = (int)((long long)prm->dehalo.lowsens  * maxVal / 100);
    const int hiScaled = (int)((long long)prm->dehalo.highsens * maxVal / 100);

    const RGYFrameInfo *pMorphSrc = pInputFrame;
    if (m_ssActive) {
        int resizeOutNum = 0;
        const auto lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        auto inputLuma = getPlane(pInputFrame, RGY_PLANE_Y);
        auto outputLuma = getPlane(&m_supersampled->frame, RGY_PLANE_Y);
        inputLuma.csp = lumaCsp;
        outputLuma.csp = lumaCsp;
        RGYFrameInfo *resizeOut[1] = { &outputLuma };
        sts = m_resizeUp->filter(&inputLuma, resizeOut, &resizeOutNum, stream);
        if (sts != RGY_ERR_NONE || resizeOutNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo resize-up failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
        pMorphSrc = &m_supersampled->frame;
    }

    RGYFrameInfo *pApplyDst = m_ssActive ? &m_corrected->frame : ppOutputFrames[0];
    sts = dehalo_process_y(pApplyDst, pMorphSrc, &m_expanded->frame, &m_inpand->frame, &m_mask->frame,
        prm->dehalo, loScaled, hiScaled, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo kernel failed: %s.\n"), get_err_mes(sts));
        return sts;
    }

    if (m_ssActive) {
        int resizeOutNum = 0;
        const auto lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        auto correctedLuma = getPlane(&m_corrected->frame, RGY_PLANE_Y);
        auto outputLuma = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
        correctedLuma.csp = lumaCsp;
        outputLuma.csp = lumaCsp;
        RGYFrameInfo *resizeOut[1] = { &outputLuma };
        sts = m_resizeDown->filter(&correctedLuma, resizeOut, &resizeOutNum, stream);
        if (sts != RGY_ERR_NONE || resizeOutNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo resize-down failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    const int copyPlanes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    for (int iplane = 1; iplane < copyPlanes; iplane++) {
        const auto planeInput = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(ppOutputFrames[0], (RGY_PLANE)iplane);
        sts = copyPlaneAsync(&planeOutput, &planeInput, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    sts = copyPlaneAlphaAsync(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterDehalo::close() {
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_supersampled.reset();
    m_expanded.reset();
    m_inpand.reset();
    m_mask.reset();
    m_corrected.reset();
    m_ssW = 0;
    m_ssH = 0;
    m_ssActive = false;
    m_frameBuf.clear();
}
