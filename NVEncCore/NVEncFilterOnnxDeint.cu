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

#include "NVEncFilterOnnxDeint.h"
#include <cuda_runtime.h>

__global__ void kernel_onnx_deint_weave_rgb(float *output, const float *input,
    const float *restoration, const int width, const int height, const bool frameA) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t plane = (size_t)width * height;
    const size_t halfPlane = plane >> 1;
    const size_t index = (size_t)y * width + x;
    const bool useInput = ((y & 1) == (frameA ? 0 : 1));
    for (int channel = 0; channel < 3; channel++) {
        output[(size_t)channel * plane + index] = useInput
            ? input[(size_t)channel * plane + index]
            : restoration[(size_t)channel * halfPlane + (size_t)(y >> 1) * width + x];
    }
}

RGY_ERR run_onnx_deint_weave_rgb(float *output, const float *input,
    const float *restoration, bool frameA, int width, int height, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_onnx_deint_weave_rgb<<<grid, block, 0, stream>>>(output, input, restoration, width, height, frameA);
    return err_to_rgy(cudaGetLastError());
}

// 上段へ偶数行、下段へ奇数行を集める。復元時は逆写像を使い、余分なコピーを省く。
__global__ void kernel_onnx_deint_reorder_fields(float *output, const float *input,
    const int width, const int height, const bool stack) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int halfHeight = height / 2;
    const int sourceY = stack ? (y % halfHeight) * 2 + y / halfHeight : y / 2 + (y & 1) * halfHeight;
    const size_t plane = (size_t)width * height;
    for (int channel = 0; channel < 3; channel++) {
        const float value = input[(size_t)channel * plane + (size_t)sourceY * width + x];
        output[(size_t)channel * plane + (size_t)y * width + x] = stack ? value : fminf(fmaxf(value, 0.0f), 1.0f);
    }
}

RGY_ERR run_onnx_deint_reorder_fields(float *output, const float *input,
    int width, int height, bool stack, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_onnx_deint_reorder_fields<<<grid, block, 0, stream>>>(output, input, width, height, stack);
    return err_to_rgy(cudaGetLastError());
}
