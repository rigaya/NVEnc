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
#include "NVEncFilterCurves.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

template<typename Type, int bit_depth>
__global__ void kernel_run_curves_plane(
    uint8_t *__restrict__ pFrame, const int pitch, const int width, const int height,
    const Type *__restrict__ pLut) {
    const int PIX_PER_THREAD = 4;
    struct __align__(sizeof(Type) * 4) Type4 {
        Type x, y, z, w;
    };
    const int ix = (blockIdx.x * blockDim.x + threadIdx.x) * PIX_PER_THREAD;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        Type4 *ptr = (Type4 *)(pFrame + iy * pitch + ix * sizeof(Type));

        Type4 pix4 = ptr[0];
        pix4.x = pLut[pix4.x];
        pix4.y = pLut[pix4.y];
        pix4.z = pLut[pix4.z];
        pix4.w = pLut[pix4.w];
        ptr[0] = pix4;
    }
}

template<typename Type, int bit_depth>
void run_curves_plane(
    uint8_t *pPlane, const int pitch, const int width, const int height,
    const void *pLut, cudaStream_t stream) {
    dim3 blockSize(64, 4);
    dim3 gridSize(divCeil(width, blockSize.x * 4), divCeil(height, blockSize.y));
    kernel_run_curves_plane<Type, bit_depth> << <gridSize, blockSize, 0, stream >> > (
        pPlane, pitch, width, height, (const Type *)pLut);
}

RGY_ERR NVEncFilterCurves::procFrame(RGYFrameInfo *pFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamCurves>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const std::map<RGY_CSP, decltype(run_curves_plane<uint8_t, 8>)*> func_list = {
        { RGY_CSP_RGB,       run_curves_plane<uint8_t,   8> },
        { RGY_CSP_RGB_16,    run_curves_plane<uint16_t, 16> },
    };
    if (func_list.count(pFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pFrame->csp]; iplane++) {
        const auto planeTarget = (RGY_PLANE)iplane;
        auto plane = getPlane(pFrame, planeTarget);

        const void *lut = nullptr;
        switch (planeTarget) {
        case RGY_PLANE_R: if (m_lut.r) lut = m_lut.r->ptr; break;
        case RGY_PLANE_G: if (m_lut.g) lut = m_lut.g->ptr; break;
        case RGY_PLANE_B: if (m_lut.b) lut = m_lut.b->ptr; break;
        default:
            break;
        }
        if (lut != nullptr) {
            func_list.at(plane.csp)(
                plane.ptr, plane.pitch, plane.width, plane.height, lut, stream);
            auto cudaerr = cudaGetLastError();
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("error at curves(%s): %s.\n"),
                    RGY_CSP_NAMES[pFrame->csp],
                    char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                return err_to_rgy(cudaerr);
            }
        }
    }
    return RGY_ERR_NONE;
}
