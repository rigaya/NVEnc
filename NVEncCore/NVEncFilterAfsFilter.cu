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
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterAfs.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <sm_32_intrinsics.h>
#pragma warning (pop)

static constexpr int MAPF_WARP_SIZE = 32;
static constexpr int MAPF_COLUMNS = MAPF_WARP_SIZE - 2;
static constexpr int MAPF_ROWS = 16;
static constexpr int MAPF_WARPS = 4;

__global__ void kernel_afs_analyze_map_filter(
    uint32_t *__restrict__ ptr_dst,
    const uint32_t *__restrict__ ptr_src,
    const int si_w_type, const int pitch_type, const int height) {
    const int lane = threadIdx.x;
    const int ix = static_cast<int>(blockIdx.x) * MAPF_COLUMNS + lane - 1;
    const int y0 = (blockIdx.y * MAPF_WARPS + threadIdx.y) * MAPF_ROWS;
    const int srcx = clamp(ix, 0, si_w_type - 1);
    const bool writeColumn = lane > 0 && lane < MAPF_WARP_SIZE - 1 && ix < si_w_type;

    // 横の隣接語はワープ内で交換し、縦の中間値は直近の行だけをレジスタに保持する。
    // 両端のレーンも交換に参加する。出力レーンが参照するのは各端の内側1バイトだけ。
    // (旧実装はsharedのタイルの右ハローを1語先(+33)から読んでいたため、タイル右端の語の結果が誤っていた。本実装で解消している)
    uint32_t h1Prev2 = 0, h1Prev = 0;
    uint32_t h2Prev2 = 0, h2Prev = 0;
#pragma unroll
    for (int row = -2; row < MAPF_ROWS + 2; row++) {
        const int srcy = clamp(y0 + row, 0, height - 1);
        const uint32_t raw = ptr_src[srcy * pitch_type + srcx];
        const uint32_t rawLeft = __shfl_up_sync(0xffffffff, raw, 1);
        const uint32_t rawRight = __shfl_down_sync(0xffffffff, raw, 1);
        const uint32_t h1Left = __funnelshift_r(rawLeft, raw, 24);
        const uint32_t h1Right = __funnelshift_r(raw, rawRight, 8);
        const uint32_t h1 = raw | ((h1Left | h1Right) & 0x03030303u) | ((h1Left & h1Right) & 0x04040404u);
        if (row >= 0) {
            const uint32_t v1 = h1Prev | (h1Prev2 & h1 & 0x07070707u);
            const uint32_t v1Left = __shfl_up_sync(0xffffffff, v1, 1);
            const uint32_t v1Right = __shfl_down_sync(0xffffffff, v1, 1);
            const uint32_t h2Left = __funnelshift_r(v1Left, v1, 24);
            const uint32_t h2Right = __funnelshift_r(v1, v1Right, 8);
            const uint32_t h2 = v1 & ((h2Left & h2Right) | 0xf8f8f8f8u);
            const int y = y0 + row - 2;
            if (row >= 2 && writeColumn && y < height) {
                ptr_dst[y * pitch_type + ix] = h2Prev & ((h2Prev2 & h2) | 0xf8f8f8f8u);
            }
            h2Prev2 = h2Prev;
            h2Prev = h2;
        }
        h1Prev2 = h1Prev;
        h1Prev = h1;
    }
}

RGY_ERR run_analyze_map_filter(uint8_t *dst, uint8_t *sp,
    const int srcWidth, const int srcPitch, const int srcHeight,
    cudaStream_t stream) {
    const int widthWords = divCeil(srcWidth, sizeof(uint32_t));
    dim3 blockSize(MAPF_WARP_SIZE, MAPF_WARPS);
    dim3 gridSize(divCeil(widthWords, MAPF_COLUMNS), divCeil(srcHeight, MAPF_ROWS * MAPF_WARPS));

    kernel_afs_analyze_map_filter<<<gridSize, blockSize, 0, stream>>>(
        (uint32_t *)dst, (uint32_t *)sp,
        widthWords, divCeil(srcPitch, sizeof(uint32_t)), srcHeight);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR afsStripeCache::map_filter(AFS_STRIPE_DATA *dst, AFS_STRIPE_DATA *sp, cudaStream_t stream) {
    dst->count0 = sp->count0;
    dst->count1 = sp->count1;
    dst->frame  = sp->frame;
    dst->status = 1;
    if (sp->map.frame.pitch[0] % sizeof(uint32_t) != 0) {
        return RGY_ERR_UNSUPPORTED;
    }
    auto sts = run_analyze_map_filter(
        dst->map.frame.ptr[0], sp->map.frame.ptr[0],
        sp->map.frame.width, sp->map.frame.pitch[0], sp->map.frame.height,
        stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}
