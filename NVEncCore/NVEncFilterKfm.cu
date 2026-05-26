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

#include "NVEncFilterKfm.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

static const int KFM_PAD_BLOCK_X = 32;
static const int KFM_PAD_BLOCK_Y = 8;

__device__ int kfm_mirror_index(const int pos, const int size) {
    if (pos < 0) {
        return -pos - 1;
    }
    if (pos >= size) {
        return size - (pos - size) - 1;
    }
    return pos;
}

template<typename Type>
__global__ void kernel_kfm_pad(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int vpad) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int paddedHeight = height + vpad * 2;
    if (x >= width || y >= paddedHeight) return;

    const int srcY = kfm_mirror_index(y - vpad, height);
    const Type *pSrc = (const Type *)(src + srcY * srcPitch + x * (int)sizeof(Type));
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = pSrc[0];
}

template<typename Type>
static RGY_ERR launch_kfm_pad_plane_t(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int vpad, cudaStream_t stream) {
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
    kernel_kfm_pad<Type><<<grid, block, 0, stream>>>(
        (uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
        (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0],
        pOutputFrame->width, pInputFrame->height, vpad);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_pad_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int vpad, cudaStream_t stream) {
    if (!pOutputFrame || !pInputFrame || !pOutputFrame->ptr[0] || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    if (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] > 8) {
        return launch_kfm_pad_plane_t<uint16_t>(pOutputFrame, pInputFrame, vpad, stream);
    }
    return launch_kfm_pad_plane_t<uint8_t>(pOutputFrame, pInputFrame, vpad, stream);
}
