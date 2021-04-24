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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterDecimate.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

#define DECIMATE_BLOCK_MAX (32)
#define DECIMATE_K2_THREAD_BLOCK_X (32)
#define DECIMATE_K2_THREAD_BLOCK_Y (8)

//blockxがこの値以下なら、kernel2を使用する
static const int DECIMATE_KERNEL2_BLOCK_X_THRESHOLD = 4;

__device__ __inline__
int func_diff_pix(int a, int b) {
    return abs(a - b);
}

template<typename Type, int block_half_x>
__device__ __inline__
int func_diff_block1(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy) {
    static_assert(block_half_x == 1, "block_half_x == 1");
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type pix0 = *(Type *)(p0 + (imgy + y) * p0_pitch + imgx * sizeof(Type));
            Type pix1 = *(Type *)(p1 + (imgy + y) * p1_pitch + imgx * sizeof(Type));
            diff += func_diff_pix(pix0, pix1);
        }
    }
    return diff;
}

template<typename Type2, int block_half_x>
__device__ __inline__
int func_diff_block2(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy) {
    static_assert(block_half_x == 2, "block_half_x == 2");
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type2 pix0 = *(Type2 *)(p0 + (imgy + y) * p0_pitch + imgx * sizeof(Type2::x));
            Type2 pix1 = *(Type2 *)(p1 + (imgy + y) * p1_pitch + imgx * sizeof(Type2::x));
            diff += func_diff_pix(pix0.x, pix1.x);
            if (imgx + 1 < width) diff += func_diff_pix(pix0.y, pix1.y);
        }
    }
    return diff;
}

template<typename Type4, int block_half_x>
__device__ __inline__
int func_diff_block4(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy) {
    static_assert(block_half_x <= 16, "block_half_x <= 16");
    static_assert((block_half_x & (block_half_x-1)) == 0, "(block_half_x & (block_half_x-1)) == 0");
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type4 pix0 = *(Type4 *)(p0 + (imgy + y)* p0_pitch + imgx * sizeof(Type4::x));
            Type4 pix1 = *(Type4 *)(p1 + (imgy + y)* p1_pitch + imgx * sizeof(Type4::x));
            diff += func_diff_pix(pix0.x, pix1.x);
            if (imgx + 1 < width && block_half_x >= 2) diff += func_diff_pix(pix0.y, pix1.y);
            if (imgx + 2 < width && block_half_x >= 3) diff += func_diff_pix(pix0.z, pix1.z);
            if (imgx + 3 < width && block_half_x >= 4) diff += func_diff_pix(pix0.w, pix1.w);
            if (block_half_x > 4) {
                pix0 = *(Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 4) * sizeof(Type4::x));
                pix1 = *(Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 4) * sizeof(Type4::x));
                if (imgx + 4 < width) diff += func_diff_pix(pix0.x, pix1.x);
                if (imgx + 5 < width) diff += func_diff_pix(pix0.y, pix1.y);
                if (imgx + 6 < width) diff += func_diff_pix(pix0.z, pix1.z);
                if (imgx + 7 < width) diff += func_diff_pix(pix0.w, pix1.w);
            }
            if (block_half_x > 8) {
                pix0 = *(Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 8) * sizeof(Type4::x));
                pix1 = *(Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 8) * sizeof(Type4::x));
                if (imgx +  8 < width) diff += func_diff_pix(pix0.x, pix1.x);
                if (imgx +  9 < width) diff += func_diff_pix(pix0.y, pix1.y);
                if (imgx + 10 < width) diff += func_diff_pix(pix0.z, pix1.z);
                if (imgx + 11 < width) diff += func_diff_pix(pix0.w, pix1.w);

                pix0 = *(Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 12) * sizeof(Type4::x));
                pix1 = *(Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 12) * sizeof(Type4::x));
                if (imgx + 12 < width) diff += func_diff_pix(pix0.x, pix1.x);
                if (imgx + 13 < width) diff += func_diff_pix(pix0.y, pix1.y);
                if (imgx + 14 < width) diff += func_diff_pix(pix0.z, pix1.z);
                if (imgx + 15 < width) diff += func_diff_pix(pix0.w, pix1.w);
            }
        }
    }
    return diff;
}

template<int DTB_X, int DTB_Y>
__device__ __inline__
void func_calc_sum_max(int diff[DTB_Y+1][DTB_X+1], int2 *__restrict__ pDst, const bool firstPlane) {
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    int sum = diff[ly][lx];
    int b2x2 = diff[ly+0][lx+0]
             + diff[ly+0][lx+1]
             + diff[ly+1][lx+0]
             + diff[ly+1][lx+1];
    __shared__ int tmp[DTB_X * DTB_Y / WARP_SIZE];
    sum  = block_sum<decltype(sum),  DTB_X, DTB_Y>(sum, tmp);
    b2x2 = block_max<decltype(b2x2), DTB_X, DTB_Y>(b2x2, tmp);
    const int lid = ly * DTB_X + lx;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        int2 ret = pDst[gid];
        ret.x = sum;
        ret.y = b2x2;
        if (firstPlane) {
            int2 dst = pDst[gid];
            ret.x += dst.x;
            ret.y = max(ret.y, dst.y);
        }
        pDst[gid] = ret;
    }
}

//block_half_x = 1の実装
//集計までをGPUで行う
template<typename Type, int DTB_X, int DTB_Y, int block_half_x>
__global__ void kernel_block_diff2_1(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const bool firstPlane,
    int2 *__restrict__ pDst) {
    static_assert(block_half_x == 1, "block_half_x == 1");
    const int lx = threadIdx.x; //スレッド数=DTB_X
    const int ly = threadIdx.y; //スレッド数=DTB_Y
    const int imgx = (blockIdx.x * DTB_X /*blockDim.x*/ + lx) * block_half_x;
    const int imgy = (blockIdx.y * DTB_Y /*blockDim.y*/ + ly) * block_half_y;

    __shared__ int diff[DTB_Y + 1][DTB_X + 1];
    diff[ly][lx] = func_diff_block1<Type, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy);
    if (ly == 0) {
        int loady = (blockIdx.y + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block1<Type, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (blockIdx.x + 1) * DTB_X * block_half_x;
            const int loady = (blockIdx.y * DTB_Y /*blockDim.y*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block1<Type, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady);
        }
    }
    __syncthreads();
    func_calc_sum_max<DTB_X, DTB_Y>(diff, pDst, firstPlane);
}

//block_half_x = 2の実装
//集計までをGPUで行う
template<typename Type2, int DTB_X, int DTB_Y, int block_half_x>
__global__ void kernel_block_diff2_2(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const bool firstPlane,
    int2 *__restrict__ pDst) {
    static_assert(block_half_x == 2, "block_half_x == 2");
    const int lx = threadIdx.x; //スレッド数=DTB_X
    const int ly = threadIdx.y; //スレッド数=DTB_Y
    const int imgx = (blockIdx.x * DTB_X /*blockDim.x*/ + lx) * block_half_x;
    const int imgy = (blockIdx.y * DTB_Y /*blockDim.y*/ + ly) * block_half_y;

    __shared__ int diff[DTB_Y + 1][DTB_X + 1];
    diff[ly][lx] = func_diff_block2<Type2, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy);
    if (ly == 0) {
        int loady = (blockIdx.y + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block2<Type2, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (blockIdx.x + 1) * DTB_X * block_half_x;
            const int loady = (blockIdx.y * DTB_Y /*blockDim.y*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block2<Type2, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady);
        }
    }
    __syncthreads();
    func_calc_sum_max<DTB_X, DTB_Y>(diff, pDst, firstPlane);
}

//block_half_x = 4, 8, 16の実装
//集計までをGPUで行う
template<typename Type4, int DTB_X, int DTB_Y, int block_half_x>
__global__ void kernel_block_diff2_4(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const bool firstPlane,
    int2 *__restrict__ pDst) {
    static_assert(4 <= block_half_x && block_half_x <= 16, "4 <= block_half_x && block_half_x <= 16");
    static_assert((block_half_x & (block_half_x - 1)) == 0, "(block_half_x & (block_half_x-1)) == 0");
    const int lx = threadIdx.x; //スレッド数=DTB_X
    const int ly = threadIdx.y; //スレッド数=DTB_Y
    const int imgx = (blockIdx.x * DTB_X /*blockDim.x*/ + lx) * block_half_x;
    const int imgy = (blockIdx.y * DTB_Y /*blockDim.y*/ + ly) * block_half_y;

    __shared__ int diff[DTB_Y+1][DTB_X+1];
    diff[ly][lx] = func_diff_block4<Type4, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy);
    if (ly == 0) {
        int loady = (blockIdx.y + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block4<Type4, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (blockIdx.x + 1) * DTB_X * block_half_x;
            const int loady = (blockIdx.y * DTB_Y /*blockDim.y*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block4<Type4, block_half_x>(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady);
        }
    }
    __syncthreads();
    func_calc_sum_max<DTB_X, DTB_Y>(diff, pDst, firstPlane);
}

template<typename Type4>
__global__ void kernel_block_diff(
    const uint8_t *__restrict__ p0, const int p0_pitch,
    const uint8_t *__restrict__ p1, const int p1_pitch,
    const int width, const int height, const bool firstPlane,
    int *__restrict__ pDst) {
    const int lx = threadIdx.x; //スレッド数=SSIM_BLOCK_X
    const int ly = threadIdx.y; //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = blockIdx.x * blockDim.x;
    const int blockoffset_y = blockIdx.y * blockDim.y;
    const int imgx = (blockoffset_x + lx) * 4;
    const int imgy = (blockoffset_y + ly);

    int diff = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(Type4::x);
        p1 += imgy * p1_pitch + imgx * sizeof(Type4::x);
        Type4 pix0 = *(Type4 *)p0;
        Type4 pix1 = *(Type4 *)p1;
        diff += func_diff_pix(pix0.x, pix1.x);
        if (imgx + 1 < width) diff += func_diff_pix(pix0.y, pix1.y);
        if (imgx + 2 < width) diff += func_diff_pix(pix0.z, pix1.z);
        if (imgx + 3 < width) diff += func_diff_pix(pix0.w, pix1.w);
    }

    __shared__ int tmp[DECIMATE_BLOCK_MAX * DECIMATE_BLOCK_MAX / WARP_SIZE];
    diff = block_sum<int>(diff, (int *)tmp, blockDim.x, blockDim.y);

    const int lid = threadIdx.y * blockDim.x + threadIdx.x;
    if (lid == 0) {
        const int gid = blockIdx.y * gridDim.x + blockIdx.x;
        if (firstPlane) {
            diff += pDst[gid];
        }
        pDst[gid] = diff;
    }
}

template<typename Type2, typename Type4>
cudaError calc_block_diff_plane(const bool useKernel2, const bool firstPlane, const RGYFrameInfo *p0, const RGYFrameInfo *p1, CUMemBufPair &tmp,
    const int blockHalfX, const int blockHalfY, cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer) {
    static_assert(std::is_integral<decltype(Type2::x)>::value && std::is_integral<decltype(Type4::x)>::value && sizeof(Type2::x) == sizeof(Type4::x),
        "Type2::x == Type4::x");
    const int width = p0->width;
    const int height = p0->height;
    dim3 blockSize, gridSize;
    if (useKernel2) {
        blockSize = dim3(DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y);
        gridSize = dim3(divCeil(divCeil(width, blockHalfX), blockSize.x), divCeil(divCeil(width, blockHalfY), blockSize.y));
    } else {
        blockSize = dim3(blockHalfX / 4, blockHalfY);
        gridSize = dim3(divCeil(width, blockSize.x * 4), divCeil(height, blockSize.y));
    }

    const int grid_count = gridSize.x * gridSize.y;
    const size_t bufsize = (useKernel2) ? grid_count * sizeof(int2) : grid_count * sizeof(int);
    if (tmp.nSize < bufsize) {
        tmp.clear();
        auto cudaerr = tmp.alloc(bufsize);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
        cudaerr = cudaMemset(tmp.ptrDevice, 0, tmp.nSize);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    if (useKernel2) {
        switch (blockHalfX) {
        case 1:
            kernel_block_diff2_1<decltype(Type4::x), DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y, 1> << < gridSize, blockSize, 0, streamDiff >> > (
                (const uint8_t *)p0->ptr, p0->pitch,
                (const uint8_t *)p1->ptr, p1->pitch,
                width, height,
                blockHalfX, firstPlane,
                (int2 *)tmp.ptrDevice);
            break;
        case 2:
            kernel_block_diff2_2<Type2, DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y, 2> << < gridSize, blockSize, 0, streamDiff >> > (
                (const uint8_t *)p0->ptr, p0->pitch,
                (const uint8_t *)p1->ptr, p1->pitch,
                width, height,
                blockHalfX, firstPlane,
                (int2 *)tmp.ptrDevice);
            break;
        case 4:
            kernel_block_diff2_4<Type4, DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y, 4> << < gridSize, blockSize, 0, streamDiff >> > (
                (const uint8_t *)p0->ptr, p0->pitch,
                (const uint8_t *)p1->ptr, p1->pitch,
                width, height,
                blockHalfX, firstPlane,
                (int2 *)tmp.ptrDevice);
            break;
        case 8:
            kernel_block_diff2_4<Type4, DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y, 8> << < gridSize, blockSize, 0, streamDiff >> > (
                (const uint8_t *)p0->ptr, p0->pitch,
                (const uint8_t *)p1->ptr, p1->pitch,
                width, height,
                blockHalfX, firstPlane,
                (int2 *)tmp.ptrDevice);
            break;
        case 16:
            kernel_block_diff2_4<Type4, DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y, 16> << < gridSize, blockSize, 0, streamDiff >> > (
                (const uint8_t *)p0->ptr, p0->pitch,
                (const uint8_t *)p1->ptr, p1->pitch,
                width, height,
                blockHalfX, firstPlane,
                (int2 *)tmp.ptrDevice);
            break;
        }
    } else {
        if (blockHalfX < 4 || 64 < blockHalfX) {
            return cudaErrorUnsupportedLimit;
        }
        kernel_block_diff<Type4><<< gridSize, blockSize, 0, streamDiff >>> (
            (const uint8_t *)p0->ptr, p0->pitch,
            (const uint8_t *)p1->ptr, p1->pitch,
            width, height,
            firstPlane,
            (int *)tmp.ptrDevice);
    }
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    cudaEventRecord(eventTransfer, streamDiff);
    cudaStreamWaitEvent(streamTransfer, eventTransfer, 0);
    cudaerr = tmp.copyDtoHAsync(streamTransfer);
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaGetLastError();
}

template<typename Type2, typename Type4>
cudaError_t calc_block_diff_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, CUMemBufPair &tmp,
    const int blockX, const int blockY,  const bool chroma,
    cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer) {
    if (tmp.ptrDevice) {
        //初期化
        auto cudaerr = cudaMemset(tmp.ptrDevice, 0, tmp.nSize);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    const bool useKernel2 = (blockX / 2 <= DECIMATE_KERNEL2_BLOCK_X_THRESHOLD);

    const int targetPlanes = (chroma) ? (int)(RGY_CSP_PLANES[p0->csp]) : 1;
    for (int i = 0; i < targetPlanes; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        int blockHalfX = blockX / 2;
        int blockHalfY = blockY / 2;
        if (i > 0 && RGY_CSP_CHROMA_FORMAT[p0->csp] == RGY_CHROMAFMT_YUV420) {
            blockHalfX /= 2;
            blockHalfY /= 2;
        }
        auto cudaerr = calc_block_diff_plane<Type2, Type4>(useKernel2, i==0, &plane0, &plane1, tmp, blockHalfX, blockHalfY, streamDiff, eventTransfer, streamTransfer);
        if (cudaerr != cudaSuccess) {
            return cudaerr;
        }
    }
    return cudaSuccess;
}

NVEncFilterDecimateFrameData::NVEncFilterDecimateFrameData() :
    m_inFrameId(-1),
    m_blockX(0),
    m_blockY(0),
    m_buf(),
    m_tmp(),
    m_diffMaxBlock(std::numeric_limits<int64_t>::max()),
    m_diffTotal(std::numeric_limits<int64_t>::max()) {

}

NVEncFilterDecimateFrameData::~NVEncFilterDecimateFrameData() {
    m_buf.clear();
}

cudaError_t NVEncFilterDecimateFrameData::set(const RGYFrameInfo *pInputFrame, int inputFrameId, int blockSizeX, int blockSizeY, cudaStream_t stream) {
    m_inFrameId = inputFrameId;
    m_blockX = blockSizeX;
    m_blockY = blockSizeY;
    m_diffMaxBlock = std::numeric_limits<int64_t>::max();
    m_diffTotal = std::numeric_limits<int64_t>::max();
    if (m_buf.frame.ptr == nullptr) {
        m_buf.alloc(pInputFrame->width, pInputFrame->height, pInputFrame->csp);
    }
    copyFrameProp(&m_buf.frame, pInputFrame);
    return m_buf.copyFrameAsync(pInputFrame, stream);
}

cudaError_t NVEncFilterDecimateFrameData::calcDiff(funcCalcDiff func, const NVEncFilterDecimateFrameData *target, const bool chroma,
    cudaStream_t streamDiff, cudaEvent_t eventTransfer, cudaStream_t streamTransfer) {
    func(&m_buf.frame, &target->get()->frame, m_tmp,
        m_blockX, m_blockY, chroma,
        streamDiff, eventTransfer, streamTransfer);
    return cudaGetLastError();
}

void NVEncFilterDecimateFrameData::calcDiffFromTmp() {
    if (m_inFrameId == 0) { //最初のフレームは差分をとる対象がない
        m_diffMaxBlock = std::numeric_limits<int64_t>::max();
        m_diffTotal = std::numeric_limits<int64_t>::max();
        return;
    }
    const int blockHalfX = m_blockX / 2;
    const int blockHalfY = m_blockY / 2;
    const bool useKernel2 = (m_blockX / 2 <= DECIMATE_KERNEL2_BLOCK_X_THRESHOLD);
    if (useKernel2) {
        int2 *const tmpHost = (int2 *)m_tmp.ptrHost;
        const size_t count = m_tmp.nSize / sizeof(int2);
        m_diffMaxBlock = -1;
        m_diffTotal = 0;
        for (size_t i = 0; i < count; i++) {
            m_diffTotal += tmpHost[i].x;
            m_diffMaxBlock = std::max<int64_t>(m_diffMaxBlock, tmpHost[i].y);
        }
    } else {
        const int blockXHalfCount = divCeil(m_buf.frame.width, blockHalfX);
        const int blockYHalfCount = divCeil(m_buf.frame.height, blockHalfY);
        const int blockXYHalfCount = blockXHalfCount * blockYHalfCount;

        int *const tmpHost = (int *)m_tmp.ptrHost;

        m_diffMaxBlock = -1;
        for (int i = 0; i < blockYHalfCount - 1; i++) {
            for (int j = 0; j < blockXHalfCount - 1; j++) {
                int64_t tmp = tmpHost[(i + 0) * blockXHalfCount + j + 0]
                            + tmpHost[(i + 0) * blockXHalfCount + j + 1]
                            + tmpHost[(i + 1) * blockXHalfCount + j + 0]
                            + tmpHost[(i + 1) * blockXHalfCount + j + 1];
                m_diffMaxBlock = std::max(m_diffMaxBlock, tmp);
            }
        }
        m_diffTotal = std::accumulate(tmpHost, tmpHost + blockXYHalfCount, (int64_t)0);
    }
}

NVEncFilterDecimateCache::NVEncFilterDecimateCache() : m_inputFrames(0), m_frames() {

}

NVEncFilterDecimateCache::~NVEncFilterDecimateCache() {
    m_frames.clear();
}

void NVEncFilterDecimateCache::init(int bufCount, int blockX, int blockY) {
    m_blockX = blockX;
    m_blockY = blockY;
    m_frames.clear();
    for (int i = 0; i < bufCount; i++) {
        m_frames.push_back(std::make_unique<NVEncFilterDecimateFrameData>());
    }
}

cudaError_t NVEncFilterDecimateCache::add(const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    const int id = m_inputFrames++;
    return frame(id)->set(pInputFrame, id, m_blockX, m_blockY, stream);
}

NVEncFilterDecimate::NVEncFilterDecimate() : m_flushed(false), m_frameLastDropped(-1), m_cache(), m_eventDiff(), m_streamDiff(), m_streamTransfer() {
    m_sFilterName = _T("decimate");
}

NVEncFilterDecimate::~NVEncFilterDecimate() {
    close();
}

RGY_ERR NVEncFilterDecimate::checkParam(const std::shared_ptr<NVEncFilterParamDecimate> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.cycle <= 1) {
        AddMessage(RGY_LOG_ERROR, _T("cycle must be 2 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.blockX < 4 || 64 < prm->decimate.blockX || (prm->decimate.blockX & (prm->decimate.blockX-1)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid blockX: %d.\n"), prm->decimate.blockX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.blockY < 4 || 64 < prm->decimate.blockY || (prm->decimate.blockY & (prm->decimate.blockY - 1)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid blockY: %d.\n"), prm->decimate.blockY);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDecimate::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pPrintMes = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDecimate>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    if (!m_pParam || std::dynamic_pointer_cast<NVEncFilterParamDecimate>(m_pParam)->decimate != prm->decimate) {

        m_cache.init(prm->decimate.cycle + 1, prm->decimate.blockX, prm->decimate.blockY);

        pParam->baseFps *= rgy_rational<int>(prm->decimate.cycle - 1, prm->decimate.cycle);

        auto cudaerr = cudaSuccess;

        m_eventDiff = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        if (cudaSuccess != (cudaerr = cudaEventCreateWithFlags(m_eventDiff.get(), cudaEventDisableTiming))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaEventCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreateWithFlags for m_eventDiff: Success.\n"));

        m_eventTransfer = std::unique_ptr<cudaEvent_t, cudaevent_deleter>(new cudaEvent_t(), cudaevent_deleter());
        if (cudaSuccess != (cudaerr = cudaEventCreateWithFlags(m_eventTransfer.get(), cudaEventDisableTiming))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaEventCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaEventCreateWithFlags for m_eventTransfer: Success.\n"));

        m_streamDiff = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        if (cudaSuccess != (cudaerr = cudaStreamCreateWithFlags(m_streamDiff.get(), 0/*cudaStreamNonBlocking*/))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamDiff: Success.\n"));

        m_streamTransfer = std::unique_ptr<cudaStream_t, cudastream_deleter>(new cudaStream_t(), cudastream_deleter());
        if (cudaSuccess != (cudaerr = cudaStreamCreateWithFlags(m_streamTransfer.get(), 0/*cudaStreamNonBlocking*/))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to cudaStreamCreateWithFlags: %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        AddMessage(RGY_LOG_DEBUG, _T("cudaStreamCreateWithFlags for m_streamTransfer: Success.\n"));

        prm->frameOut.pitch = prm->frameIn.pitch;

        m_fpLog.reset();
        if (prm->decimate.log) {
            const tstring logfilename = prm->outfilename + _T(".decimate.log.txt");
            m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(logfilename.c_str(), _T("w")), fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened log file: %s.\n"), logfilename.c_str());
        }

        const int max_value = (1 << RGY_CSP_BIT_DEPTH[prm->frameIn.csp]) - 1;
        m_nPathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
        m_threSceneChange = (int64_t)(((double)max_value * prm->frameIn.width * prm->frameIn.height * (double)prm->decimate.threSceneChange) / 100);
        m_threDuplicate = (int64_t)(((double)max_value * prm->decimate.blockX * prm->decimate.blockY * (double)prm->decimate.threDuplicate) / 100);
        m_frameLastDropped = -1;
        m_flushed = false;

        setFilterInfo(pParam->print());
    }
    m_pParam = pParam;
    return sts;
}

tstring NVEncFilterParamDecimate::print() const {
    return decimate.print();
}

RGY_ERR NVEncFilterDecimate::setOutputFrame(int64_t nextTimestamp, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDecimate>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int iframeStart = (int)((m_cache.inframe() + prm->decimate.cycle - 1) / prm->decimate.cycle) * prm->decimate.cycle - prm->decimate.cycle;
    //GPU->CPUの転送終了を待機
    cudaStreamSynchronize(*m_streamTransfer.get());
    //CPUに転送された情報の後処理
    for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        m_cache.frame(iframe)->calcDiffFromTmp();
    }

    //判定
    int frameDuplicate = -1;
    int frameSceneChange = -1;
    int frameLowest = iframeStart;
    for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        if (m_cache.frame(iframe)->diffTotal() > m_threSceneChange) {
            frameSceneChange = iframe;
        }
        if (m_cache.frame(iframe)->diffMaxBlock() < m_cache.frame(frameLowest)->diffMaxBlock()) {
            frameLowest = iframe;
        }
    }
    if (m_cache.frame(frameLowest)->diffMaxBlock() < m_threDuplicate) {
        frameDuplicate = frameLowest;
    }
    //ドロップするフレームの選択
    auto selectDropFrame = [&]() {
        if (m_cache.inframe() - iframeStart == prm->decimate.cycle) {
            //cycle分のフレームがそろっている場合は、必ずいずれかのフレームをドロップする
            return (frameSceneChange >= 0 && frameDuplicate < 0) ? frameSceneChange : frameLowest;
        }
        //cycle分のフレームがそろっていない(flushする)場合は、
        //dropすべきものがなければ、dropしない(-1)とする
        if (m_frameLastDropped + prm->decimate.cycle >= m_cache.inframe()) {
            return -1;
        }
        return (frameSceneChange >= 0 && frameDuplicate < 0) ? frameSceneChange : frameLowest;
    };
    const int frameDrop = selectDropFrame();

    //入力フレームのtimestamp取得
    bool ptsInvalid = false;
    std::vector<int64_t> cycleInPts;
    cycleInPts.reserve(prm->decimate.cycle+1);
    for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        auto timestamp = m_cache.frame(iframe)->get()->frame.timestamp;
        if (timestamp == AV_NOPTS_VALUE) {
            ptsInvalid = true;
        }
        cycleInPts.push_back(timestamp);
    }
    if (nextTimestamp == AV_NOPTS_VALUE && !ptsInvalid) {
        nextTimestamp = (cycleInPts.back() - cycleInPts.front()) * cycleInPts.size() / (cycleInPts.size() - 1);
    }
    cycleInPts.push_back(nextTimestamp);
    if (frameDrop < 0 && !ptsInvalid) {
        cycleInPts.push_back((cycleInPts.back() - cycleInPts.front()) * cycleInPts.size() / (cycleInPts.size() - 1));
    }

    //出力フレームのtimestampの調整
    std::vector<int64_t> cycleOutPts;
    cycleOutPts.reserve(cycleInPts.size());
    for (int i = 0; i < (int)cycleInPts.size() - 1; i++) {
        cycleOutPts.push_back((ptsInvalid) ? AV_NOPTS_VALUE : (cycleInPts[i] + (cycleInPts[i + 1] - cycleInPts[i]) * i / (prm->decimate.cycle - 1)));
    }

    //出力フレームの設定
    *pOutputFrameNum = 0;
    for (int i = 0, iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        auto iframeData = m_cache.frame(iframe);
        if (iframe != frameDrop) {
            auto frame = &iframeData->get()->frame;
            frame->timestamp = cycleOutPts[i];
            frame->duration = cycleOutPts[i + 1] - cycleOutPts[i];
            ppOutputFrames[i++] = frame;
            *pOutputFrameNum = i;
        }
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "[%s%s%s%s] %8d: diff total %10lld, max %10lld\n",
                iframe == frameSceneChange ? "S" : " ",
                iframe == frameDuplicate ? "P" : " ",
                iframe == frameLowest ? "L" : " ",
                iframe == frameDrop ? "D" : " ",
                iframe,
                (long long int)iframeData->diffTotal(),
                (long long int)iframeData->diffMaxBlock());
        }
    }
    m_frameLastDropped = frameDrop;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDecimate::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDecimate>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (pInputFrame->ptr == nullptr && m_flushed) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    const int inframeId = m_cache.inframe();
    *pOutputFrameNum = 0;
    if (m_cache.inframe() > 0 && (m_cache.inframe() % prm->decimate.cycle == 0 || pInputFrame->ptr == nullptr)) { //cycle分のフレームがそろったら
        auto ret = setOutputFrame((pInputFrame) ? pInputFrame->timestamp : AV_NOPTS_VALUE, ppOutputFrames, pOutputFrameNum);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }

        if (pInputFrame->ptr == nullptr) {
            m_flushed = true;
            return sts;
        }
    }

    auto cudaerr = m_cache.add(pInputFrame, stream);
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"),
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return RGY_ERR_CUDA;
    }

    if (inframeId > 0) {
        //前のフレームとの差分をとる
        auto frameCurrent = m_cache.frame(inframeId + 0);
        auto framePrev    = m_cache.frame(inframeId - 1);

        cudaEventRecord(*m_eventDiff.get(), stream);
        cudaStreamWaitEvent(*m_streamDiff.get(), *m_eventDiff.get(), 0);

        static const std::map<RGY_CSP, funcCalcDiff> func_list = {
            { RGY_CSP_YV12,      calc_block_diff_frame<uchar2,  uchar4>  },
            { RGY_CSP_YV12_16,   calc_block_diff_frame<ushort2, ushort4> },
            { RGY_CSP_YUV444,    calc_block_diff_frame<uchar2,  uchar4>  },
            { RGY_CSP_YUV444_16, calc_block_diff_frame<ushort2, ushort4> }
        };
        if (func_list.count(pInputFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        frameCurrent->calcDiff(func_list.at(pInputFrame->csp), framePrev,
            prm->decimate.chroma,
            *m_streamDiff.get(), *m_eventTransfer.get(), *m_streamTransfer.get());
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at calc_block_diff_frame(%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }
    return sts;
}

void NVEncFilterDecimate::close() {
    m_pFrameBuf.clear();
    m_eventDiff.reset();
    m_streamDiff.reset();
    m_streamTransfer.reset();
    m_fpLog.reset();
}
