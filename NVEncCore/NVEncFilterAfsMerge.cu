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
#include <device_functions.h>
#pragma warning (pop)

#define WARP_SIZE_2N (5)
#define WARP_SIZE    (1<<WARP_SIZE_2N)

#define MERGE_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
#define MERGE_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group
#define MERGE_BLOCK_LOOP_Y  (1) //work groupのy方向反復数

#define u8x4(x)  (uint32_t)(((uint32_t)(x)) | (((uint32_t)(x)) <<  8) | (((uint32_t)(x)) << 16) | (((uint32_t)(x)) << 24))


template<int width>
__inline__ __device__
int warp_sum(int val) {
    static_assert(width <= WARP_SIZE, "width too big for warp_sum");
    if (width >= 32) val += __shfl_xor(val, 16);
    if (width >= 16) val += __shfl_xor(val, 8);
    if (width >=  8) val += __shfl_xor(val, 4);
    if (width >=  4) val += __shfl_xor(val, 2);
    if (width >=  2) val += __shfl_xor(val, 1);
    return val;
}

__inline__ __device__
int block_sum(int val, int *shared) {
    static_assert(MERGE_BLOCK_INT_X * MERGE_BLOCK_Y <= WARP_SIZE * WARP_SIZE, "block size too big for block_sum");
    const int lid = threadIdx.y * MERGE_BLOCK_INT_X + threadIdx.x;
    const int lane    = lid & (WARP_SIZE - 1);
    const int warp_id = lid >> WARP_SIZE_2N;

    val = warp_sum<WARP_SIZE>(val);

    if (lane == 0) shared[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        val = (lid * WARP_SIZE < MERGE_BLOCK_INT_X * MERGE_BLOCK_Y) ? shared[lane] : 0;
        val = warp_sum<MERGE_BLOCK_INT_X * MERGE_BLOCK_Y / WARP_SIZE>(val);
    }
    return val;
}

template<typename Type>
__global__ void kernel_afs_merge_scan(
    Type *__restrict__ ptr_dst,
    int *__restrict__ ptr_count,
    const Type *__restrict__ src_p0,
    const Type *__restrict__ src_p1,
    const int si_w_type, const int pitch_type, const int height,
    const int tb_order,
    const uint32_t scan_left, const uint32_t scan_top, const uint32_t scan_width, const uint32_t scan_height) {
    //int lx = threadIdx.x; //スレッド数=MERGE_BLOCK_INT_X
    int ly = threadIdx.y; //スレッド数=MERGE_BLOCK_Y
    int imgx = blockIdx.x * MERGE_BLOCK_INT_X /*blockDim.x*/ + threadIdx.x;

    int stripe_count = 0;
    const int field_select = ((ly + tb_order) & 1);

    if (imgx < si_w_type) {
        const int mask_lsft = (field_select) ? 1 : 2;
        int imgy = blockIdx.y * MERGE_BLOCK_LOOP_Y * MERGE_BLOCK_Y + ly;
        src_p0  += imgx + pitch_type * imgy;
        src_p1  += imgx + pitch_type * imgy;
        ptr_dst += imgx + pitch_type * imgy;
        #pragma unroll
        for (int iloop = 0; iloop < MERGE_BLOCK_LOOP_Y; iloop++,
            imgy    += MERGE_BLOCK_Y,
            ptr_dst += MERGE_BLOCK_Y * pitch_type,
            src_p0  += MERGE_BLOCK_Y * pitch_type,
            src_p1  += MERGE_BLOCK_Y * pitch_type
            ) {
            if (imgy < height) {
                const int offsetm = (imgy == 0       ) ? 0 : -pitch_type;
                const int offsetp = (imgy >= height-1) ? 0 :  pitch_type;
                Type p0m = src_p0[offsetm];
                Type p0c = src_p0[0];
                Type p0p = src_p0[offsetp];

                Type p1m = src_p1[offsetm];
                Type p1c = src_p1[0];
                Type p1p = src_p1[offsetp];

                Type m4 = (p0m | p0p | u8x4(0xf3)) & p0c;
                Type m5 = (p1m | p1p | u8x4(0xf3)) & p1c;

                Type m6 = (m4 & m5 & u8x4(0x44)) | (~(p0c) & u8x4(0x33));

                ptr_dst[0] = m6;

                if ((((uint32_t)imgx - scan_left) < scan_width) && (((uint32_t)imgy - scan_top) < scan_height)) {
                    //(m6 & MASK) == 0 を各バイトについて加算する
                    //MASK = 0x50 or 0x60 (ly, tb_order依存)
                    //まず、0x10 or 0x20 のビットを 0x40に集約する
                    Type check = m6 | ((m6) << mask_lsft);
                    //0x40のビットが1なら0、0なら1としたいので、xorでチェック
                    //他のビットはいらないので、自分とxorして消す
                    stripe_count += __popc(check ^ (check | u8x4(0x40)));
                }
            }
        }
    }
    //motion countの総和演算
    // 32               16              0
    //  |  count_latter ||  count_first |
    static_assert(MERGE_BLOCK_INT_X * sizeof(int) * MERGE_BLOCK_Y * MERGE_BLOCK_LOOP_Y < (1<<(sizeof(short)*8-1)), "reduce block size for proper reduction in 16bit.");
    int stripe_count_01 = (int)(field_select ? (uint32_t)stripe_count << 16 : (uint32_t)stripe_count);

    __shared__ int shared[MERGE_BLOCK_INT_X * MERGE_BLOCK_Y / WARP_SIZE]; //int単位でアクセスする
    stripe_count_01 = block_sum(stripe_count_01, (int *)shared);

    const int lid = threadIdx.y * MERGE_BLOCK_INT_X + threadIdx.x;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        ptr_count[gid] = stripe_count_01;
    }
}

template<typename Type>
cudaError_t run_merge_scan(uint8_t *dst,
    uint8_t *sp0, uint8_t *sp1,
    const int srcWidth, const int srcPitch, const int srcHeight,
    CUMemBufPair *count_stripe, const VppAfs *pAfsPrm, cudaStream_t stream) {
    auto cudaerr = cudaSuccess;

    dim3 blockSize(MERGE_BLOCK_INT_X, MERGE_BLOCK_Y);
    dim3 gridSize(divCeil(srcWidth, blockSize.x * sizeof(Type)), divCeil(srcHeight, blockSize.y * MERGE_BLOCK_LOOP_Y));

    const int grid_count = gridSize.x * gridSize.y;
    if (count_stripe->nSize < grid_count) {
        count_stripe->clear();
        cudaerr = count_stripe->alloc(grid_count * sizeof(int));
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    const uint32_t scan_left   = pAfsPrm->clip.left / sizeof(Type);
    const uint32_t scan_width  = (srcWidth - pAfsPrm->clip.left - pAfsPrm->clip.right) / sizeof(Type);
    const uint32_t scan_top    = pAfsPrm->clip.top;
    const uint32_t scan_height = srcHeight - pAfsPrm->clip.top - pAfsPrm->clip.bottom;

    kernel_afs_merge_scan<<<gridSize, blockSize, 0, stream>>>(
        (Type *)dst, (int *)count_stripe->ptrDevice, (Type *)sp0, (Type *)sp1,
        divCeil(srcWidth, sizeof(Type)), divCeil(srcPitch, sizeof(Type)), srcHeight,
        pAfsPrm->tb_order ? 1 : 0,
        scan_left, scan_top, scan_width, scan_height);
    return cudaGetLastError();
}

cudaError_t NVEncFilterAfs::merge_scan(AFS_STRIPE_DATA *sp, AFS_SCAN_DATA *sp0, AFS_SCAN_DATA *sp1, CUMemBufPair *count_motion, const NVEncFilterParamAfs *pAfsParam, cudaStream_t stream) {
    auto cudaerr = run_merge_scan<uint32_t>(
        sp->map.frame.ptr, sp0->map.frame.ptr, sp1->map.frame.ptr,
        sp1->map.frame.width, sp1->map.frame.pitch, sp1->map.frame.height,
        count_motion, &pAfsParam->afs, stream);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaSuccess;
}
