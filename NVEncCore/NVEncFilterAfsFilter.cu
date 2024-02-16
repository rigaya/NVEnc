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

#define FILTER_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
#define FILTER_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group

#define u8x4(x)  (((uint32_t)x) | (((uint32_t)x) <<  8) | (((uint32_t)x) << 16) | (((uint32_t)x) << 24))

__device__ __inline__
void filter_h_1(uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2], int sx, int sy) {
    uint32_t x0 = shared[0][sy][sx-1];
    uint32_t x1 = shared[0][sy][sx+0];
    uint32_t x2 = shared[0][sy][sx+1];
#if __CUDA_ARCH__ >= 320
    uint32_t x3 = __funnelshift_r(x0, x1, 24);
    uint32_t x4 = __funnelshift_r(x1, x2,  8);
#else
    uint32_t x3 = x0 >> 24 | x1 <<  8;
    uint32_t x4 = x1 >>  8 | x2 << 24;
#endif
    shared[1][sy][sx] = x1 | ((x3 | x4) & u8x4(0x03)) | ((x3 & x4) & u8x4(0x04));
}

__device__ __inline__
void filter_h_1_edge(uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2], int sx, int sy) {
    uint32_t x0 = (sx-1 >= 0)                 ? shared[0][sy][sx-1] : 0;
    uint32_t x1 =                               shared[0][sy][sx+0];
    uint32_t x2 = (sx+1<FILTER_BLOCK_INT_X+2) ? shared[0][sy][sx+1] : 0;
#if __CUDA_ARCH__ >= 320
    uint32_t x3 = __funnelshift_r(x0, x1, 24);
    uint32_t x4 = __funnelshift_r(x1, x2, 8);
#else
    uint32_t x3 = x0 >> 24 | x1 <<  8;
    uint32_t x4 = x1 >>  8 | x2 << 24;
#endif
    shared[1][sy][sx] = x1 | ((x3 | x4) & u8x4(0x03)) | ((x3 & x4) & u8x4(0x04));
}

__device__ __inline__
void filter_v_1(uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2], int sx, int sy) {
    uint32_t x0 = shared[1][sy-1][sx];
    uint32_t x1 = shared[1][sy+0][sx];
    uint32_t x2 = shared[1][sy+1][sx];
    shared[0][sy][sx] = x1 | (x0 & x2 & u8x4(0x03 | 0x04));
}

__device__ __inline__
void filter_h_2(uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2], int sx, int sy) {
    uint32_t x0 = shared[0][sy][sx-1];
    uint32_t x1 = shared[0][sy][sx+0];
    uint32_t x2 = shared[0][sy][sx+1];
#if __CUDA_ARCH__ >= 320
    uint32_t x3 = __funnelshift_r(x0, x1, 24);
    uint32_t x4 = __funnelshift_r(x1, x2,  8);
#else
    uint32_t x3 = x0 >> 24 | x1 <<  8;
    uint32_t x4 = x1 >>  8 | x2 << 24;
#endif
    shared[1][sy][sx] = x1 & ((x3 & x4) | u8x4(0xf8));
}

__device__ __inline__
uint32_t get_filter_v_2(uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2], int sx, int sy) {
    uint32_t x0 = shared[1][sy-1][sx];
    uint32_t x1 = shared[1][sy+0][sx];
    uint32_t x2 = shared[1][sy+1][sx];
    return x1 & ((x0 & x2) | u8x4(0xf8));
}

__global__ void kernel_afs_analyze_map_filter(
    uint32_t *__restrict__ ptr_dst,
    const uint32_t *__restrict__ ptr_src,
    const int si_w_type, const int pitch_type, const int height) {
    const int lx = threadIdx.x; //スレッド数=FILTER_BLOCK_INT_X
    const int ly = threadIdx.y; //スレッド数=FILTER_BLOCK_Y
    const int imgx = blockIdx.x * FILTER_BLOCK_INT_X /*blockDim.x*/ + threadIdx.x;
    const int imgy = blockIdx.y * FILTER_BLOCK_Y     /*blockDim.y*/ + threadIdx.y;

    //左右の縁 lx(0)=0, lx(1)=FILTER_BLOCK_INT_X+1
    const int sx_edge = (lx) ? FILTER_BLOCK_INT_X+1 : 0;

    __shared__ uint32_t shared[2][FILTER_BLOCK_Y+4][FILTER_BLOCK_INT_X+2];

    //sharedメモリへのロード
#define SRCPTR(ix, iy) *(uint32_t *)(ptr_src + clamp((iy), 0, height) * pitch_type + clamp((ix), 0, si_w_type))
    //中央部分のロード
    shared[0][ly][lx+1] = SRCPTR(imgx, imgy-2);
    if (lx < 2) {
        //x方向の左(lx=0)右(lx=1)の縁をロード
        shared[0][ly][sx_edge] = SRCPTR(imgx-1+sx_edge, imgy-2);
    }
    if (ly < 4) {
        shared[0][ly + FILTER_BLOCK_Y][lx+1] = SRCPTR(imgx, imgy-2+FILTER_BLOCK_Y);
        if (lx < 2) {
            //x方向の左(lx=0)右(lx=1)の縁をロード
            shared[0][ly + FILTER_BLOCK_Y][sx_edge] = SRCPTR(imgx-1+sx_edge, imgy-2+FILTER_BLOCK_Y);
        }
    }
    __syncthreads();
#undef SRCPTR

    //filter_h(1)
    filter_h_1(shared, lx+1, ly);
    if (lx < 2) {
        //x方向の縁
        filter_h_1_edge(shared, sx_edge, ly);
    }
    if (ly < 4) {
        filter_h_1(shared, lx+1, ly+FILTER_BLOCK_Y);
        if (lx < 2) {
            //x方向の縁
            filter_h_1_edge(shared, sx_edge, ly+FILTER_BLOCK_Y);
        }
    }
    __syncthreads();

    //filter_v(1)
    filter_v_1(shared, lx, ly+1);
    if (lx < 2) {
        //x方向の縁
        filter_v_1(shared, lx+FILTER_BLOCK_INT_X, ly+1);
    }
    if (ly < 2) {
        filter_v_1(shared, lx, ly+1+FILTER_BLOCK_Y);
        if (lx < 2) {
            //x方向の縁
            filter_v_1(shared, lx+FILTER_BLOCK_INT_X, ly+1+FILTER_BLOCK_Y);
        }
    }
    __syncthreads();

    //filter_h(2)
    filter_h_2(shared, lx+1, ly+1);
    if (ly < 2) {
        filter_h_2(shared, lx+1, ly+1+FILTER_BLOCK_Y);
    }
    __syncthreads();

    //filter_v(2)
    if (imgx < si_w_type && imgy < height) {
        uint32_t ret = get_filter_v_2(shared, lx+1, ly+2);
        ptr_dst[imgy * pitch_type + imgx] = ret;
    }
}

RGY_ERR run_analyze_map_filter(uint8_t *dst, uint8_t *sp,
    const int srcWidth, const int srcPitch, const int srcHeight,
    cudaStream_t stream) {
    dim3 blockSize(FILTER_BLOCK_INT_X, FILTER_BLOCK_Y);
    dim3 gridSize(divCeil(srcWidth, blockSize.x * sizeof(uint32_t)), divCeil(srcHeight, blockSize.y));

    kernel_afs_analyze_map_filter<<<gridSize, blockSize, 0, stream>>>(
        (uint32_t *)dst, (uint32_t *)sp,
        divCeil(srcWidth, sizeof(uint32_t)), divCeil(srcPitch, sizeof(uint32_t)), srcHeight);
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
