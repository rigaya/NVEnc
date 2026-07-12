// -----------------------------------------------------------------------------------------
//     NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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

#include "NVEncFilterStDeint.h"
#include <cuda_runtime.h>

static __device__ __forceinline__ float stdeint_clamp01(const float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __device__ __forceinline__ uint8_t stdeint_to_u8(const float value) {
    return (uint8_t)max(0, min(255, __float2int_rz(value + 0.5f)));
}

__global__ void kernel_stdeint_pack_rgb(const uint8_t *srcY, const int pitchY,
    const uint8_t *srcU, const int pitchU, const uint8_t *srcV, const int pitchV,
    const int chromaStride, float *dst, const int width, const int height,
    const NVEncStDeintColorCoeffs coeffs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int cx = min(x >> 1, (width >> 1) - 1);
    const int cy = min(y >> 1, (height >> 1) - 1);
    const float yn = ((float)srcY[(size_t)y * pitchY + x] - coeffs.yOff) * coeffs.yScale;
    const float un = ((float)srcU[(size_t)cy * pitchU + (size_t)cx * chromaStride] - coeffs.cOff) * coeffs.cScale;
    const float vn = ((float)srcV[(size_t)cy * pitchV + (size_t)cx * chromaStride] - coeffs.cOff) * coeffs.cScale;
    const size_t plane = (size_t)width * height;
    const size_t index = (size_t)y * width + x;
    dst[index] = stdeint_clamp01(yn + coeffs.matVR * vn);
    dst[plane + index] = stdeint_clamp01(yn + coeffs.matUG * un + coeffs.matVG * vn);
    dst[plane * 2 + index] = stdeint_clamp01(yn + coeffs.matUB * un);
}

static __device__ __forceinline__ float stdeint_load_rgb(const float *input,
    const float *restoration, const int channel, const int x, const int y,
    const int width, const int height, const bool frameA) {
    const size_t plane = (size_t)width * height;
    const size_t halfPlane = plane >> 1;
    const bool useInput = ((y & 1) == (frameA ? 0 : 1));
    if (useInput) {
        return input[(size_t)channel * plane + (size_t)y * width + x];
    }
    return restoration[(size_t)channel * halfPlane + (size_t)(y >> 1) * width + x];
}

__global__ void kernel_stdeint_weave_luma(uint8_t *dstY, const int pitchY,
    const float *input, const float *restoration, const int width, const int height,
    const bool frameA, const NVEncStDeintColorCoeffs coeffs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float r = stdeint_load_rgb(input, restoration, 0, x, y, width, height, frameA);
    const float g = stdeint_load_rgb(input, restoration, 1, x, y, width, height, frameA);
    const float b = stdeint_load_rgb(input, restoration, 2, x, y, width, height, frameA);
    const float luma = coeffs.matRY * r + coeffs.matGY * g + coeffs.matBY * b;
    dstY[(size_t)y * pitchY + x] = stdeint_to_u8(luma * coeffs.yRange + coeffs.yOff);
}

__global__ void kernel_stdeint_weave_chroma(uint8_t *dstU, const int pitchU,
    uint8_t *dstV, const int pitchV, const int chromaStride,
    const float *input, const float *restoration, const int width, const int height,
    const bool frameA, const NVEncStDeintColorCoeffs coeffs) {
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= width / 2 || cy >= height / 2) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;
#pragma unroll
    for (int dy = 0; dy < 2; dy++) {
#pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            const int x = cx * 2 + dx;
            const int y = cy * 2 + dy;
            r += stdeint_load_rgb(input, restoration, 0, x, y, width, height, frameA);
            g += stdeint_load_rgb(input, restoration, 1, x, y, width, height, frameA);
            b += stdeint_load_rgb(input, restoration, 2, x, y, width, height, frameA);
        }
    }
    const float u = 0.25f * (coeffs.matRU * r + coeffs.matGU * g + coeffs.matBU * b);
    const float v = 0.25f * (coeffs.matRV * r + coeffs.matGV * g + coeffs.matBV * b);
    dstU[(size_t)cy * pitchU + (size_t)cx * chromaStride] = stdeint_to_u8(u * coeffs.cRange + coeffs.cOff);
    dstV[(size_t)cy * pitchV + (size_t)cx * chromaStride] = stdeint_to_u8(v * coeffs.cRange + coeffs.cOff);
}

RGY_ERR run_stdeint_pack_rgb(const RGYFrameInfo *input, float *output,
    const NVEncStDeintColorCoeffs& coeffs, cudaStream_t stream) {
    const bool nv12 = input->csp == RGY_CSP_NV12;
    const uint8_t *srcU = input->ptr[1];
    const uint8_t *srcV = nv12 ? input->ptr[1] + 1 : input->ptr[2];
    const int pitchV = nv12 ? input->pitch[1] : input->pitch[2];
    const int chromaStride = nv12 ? 2 : 1;
    const dim3 block(32, 8);
    const dim3 grid(divCeil(input->width, (int)block.x), divCeil(input->height, (int)block.y));
    kernel_stdeint_pack_rgb<<<grid, block, 0, stream>>>(input->ptr[0], input->pitch[0],
        srcU, input->pitch[1], srcV, pitchV, chromaStride, output,
        input->width, input->height, coeffs);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_stdeint_weave_yuv(RGYFrameInfo *output, const float *input,
    const float *restoration, bool frameA, const NVEncStDeintColorCoeffs& coeffs,
    cudaStream_t stream) {
    const bool nv12 = output->csp == RGY_CSP_NV12;
    uint8_t *dstU = output->ptr[1];
    uint8_t *dstV = nv12 ? output->ptr[1] + 1 : output->ptr[2];
    const int pitchV = nv12 ? output->pitch[1] : output->pitch[2];
    const int chromaStride = nv12 ? 2 : 1;
    const dim3 block(32, 8);
    const dim3 gridY(divCeil(output->width, (int)block.x), divCeil(output->height, (int)block.y));
    kernel_stdeint_weave_luma<<<gridY, block, 0, stream>>>(output->ptr[0], output->pitch[0],
        input, restoration, output->width, output->height, frameA, coeffs);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err_to_rgy(err);
    const dim3 gridC(divCeil(output->width / 2, (int)block.x), divCeil(output->height / 2, (int)block.y));
    kernel_stdeint_weave_chroma<<<gridC, block, 0, stream>>>(dstU, output->pitch[1], dstV, pitchV,
        chromaStride, input, restoration, output->width, output->height, frameA, coeffs);
    return err_to_rgy(cudaGetLastError());
}
