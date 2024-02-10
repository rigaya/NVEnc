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

#include <map>
#include <array>
#include "convert_csp.h"
#include "NVEncFilterOverlay.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

template<typename Type, int bit_depth>
__global__ void kernel_run_overlay_plane(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int width, const int height,
    const uint8_t *__restrict__ pOverlay, const int overlayPitch,
    const uint8_t *__restrict__ pAlpha, const int alphaPitch, const int overlayWidth, const int overlayHeight,
    const int overlayPosX, const int overlayPosY) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int ret = *(Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        if (   overlayPosX <= ix && ix < overlayPosX + overlayWidth
            && overlayPosY <= iy && iy < overlayPosY + overlayHeight) {
            const int overlaySrc   = *(Type    *)(pOverlay + (iy - overlayPosY) * overlayPitch + (ix - overlayPosX) * sizeof(Type)   );
            const int overlayAlpha = *(uint8_t *)(pAlpha   + (iy - overlayPosY) * alphaPitch   + (ix - overlayPosX) * sizeof(uint8_t));
            const float overlayAlphaF = overlayAlpha / 255.0f;
            float blend = overlaySrc * overlayAlphaF + ret * (1.0f - overlayAlphaF);
            ret = (int)(blend + 0.5f);
        }

        Type *ptr = (Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(ret, 0, (1 << bit_depth) - 1);
    }
}

template<typename Type, int bit_depth>
RGY_ERR run_overlay_plane(
    uint8_t *pDst, const int dstPitch,
    const uint8_t *pSrc, const int srcPitch, const int width, const int height,
    const uint8_t *pOverlay, const int overlayPitch,
    const uint8_t *pAlpha, const int alphaPitch, const int overlayWidth, const int overlayHeight,
    const int overlayPosX, const int overlayPosY,
    cudaStream_t stream) {
    dim3 blockSize(64, 8);
    dim3 gridSize(divCeil(width, blockSize.x), divCeil(height, blockSize.y));
    kernel_run_overlay_plane<Type, bit_depth> << <gridSize, blockSize, 0, stream >> > (
        pDst, dstPitch,
        pSrc, srcPitch, width, height,
        pOverlay, overlayPitch,
        pAlpha, alphaPitch, overlayWidth, overlayHeight,
        overlayPosX, overlayPosY);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR NVEncFilterOverlay::overlayFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamOverlay>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const std::map<RGY_CSP, decltype(run_overlay_plane<uint8_t, 8>)*> func_list = {
        { RGY_CSP_YV12,      run_overlay_plane<uint8_t,   8> },
        { RGY_CSP_YV12_16,   run_overlay_plane<uint16_t, 16> },
        { RGY_CSP_YUV444,    run_overlay_plane<uint8_t,   8> },
        { RGY_CSP_YUV444_16, run_overlay_plane<uint16_t, 16> },
    };
    if (func_list.count(pInputFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pInputFrame->csp]; iplane++) {
        const auto planeTarget = (RGY_PLANE)iplane;
        auto planeOut          = getPlane(pOutputFrame,     planeTarget);
        const auto planeIn     = getPlane(pInputFrame,      planeTarget);
        const auto planeFrame  = getPlane(m_frame.inputPtr, planeTarget);
        const auto planeAlpha  = getPlane(m_alpha.inputPtr, planeTarget);

        const auto posX = (planeTarget != RGY_PLANE_Y && RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420) ? prm->overlay.posX >> 1 : prm->overlay.posX;
        const auto posY = (planeTarget != RGY_PLANE_Y && RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_YUV420) ? prm->overlay.posY >> 1 : prm->overlay.posY;

        auto sts = func_list.at(pInputFrame->csp)(
            planeOut.ptr, planeOut.pitch,
            planeIn.ptr, planeIn.pitch, planeIn.width, planeIn.height,
            planeFrame.ptr, planeFrame.pitch,
            planeAlpha.ptr, planeAlpha.pitch, planeAlpha.width, planeAlpha.height,
            posX, posY, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at overlay(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                get_err_mes(sts));
            return sts;
        }
    }
    return RGY_ERR_NONE;
}
