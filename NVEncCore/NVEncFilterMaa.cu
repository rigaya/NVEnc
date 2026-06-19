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

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterMaa.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int MAA_BLOCK_X = 32;
static const int MAA_BLOCK_Y = 8;

static int alignSsDim(int srcDim, float ss) {
    const float scaled = (float)srcDim * ss / 4.0f;
    return (int)std::lround(scaled) * 4;
}

template<typename Type>
__device__ __forceinline__ int maa_load_pix_clamp(const Type *row, int x, int width) {
    x = clamp(x, 0, width - 1);
    return (int)row[x];
}

template<typename Type>
__device__ __forceinline__ int maa_read_pix_clamp(const uint8_t *frame, const int pitch, const int width, const int height, int x, int y) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)*(const Type *)(frame + y * pitch + x * sizeof(Type));
}

template<typename Type>
__device__ __forceinline__ int maa_sn3(int p1, int p2, int p3, int max_val) {
    int v = (4 * p1 + 5 * p2 - p3) >> 3;
    return clamp(v, 0, max_val);
}

template<typename Type>
__global__ void kernel_maa_fturn_left(const uint8_t *src, const int srcPitch, const int srcWidth,
    uint8_t *dst, const int dstPitch, const int dstWidth, const int dstHeight) {
    const int x_new = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_new = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_new >= dstWidth || y_new >= dstHeight) return;

    const int x_old = srcWidth - 1 - y_new;
    const int y_old = x_new;
    const Type *srcPix = (const Type *)(src + y_old * srcPitch + x_old * sizeof(Type));
    Type *dstPix = (Type *)(dst + y_new * dstPitch + x_new * sizeof(Type));
    dstPix[0] = srcPix[0];
}

template<typename Type>
__global__ void kernel_maa_fturn_right(const uint8_t *src, const int srcPitch, const int srcHeight,
    uint8_t *dst, const int dstPitch, const int dstWidth, const int dstHeight) {
    const int x_new = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_new = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_new >= dstWidth || y_new >= dstHeight) return;

    const int x_old = y_new;
    const int y_old = srcHeight - 1 - x_new;
    const Type *srcPix = (const Type *)(src + y_old * srcPitch + x_old * sizeof(Type));
    Type *dstPix = (Type *)(dst + y_new * dstPitch + x_new * sizeof(Type));
    dstPix[0] = srcPix[0];
}

template<typename Type, int bit_depth>
__global__ void kernel_maa_sangnom_prepare(const uint8_t *src, const int srcPitch, const int width, const int height,
    uint8_t *costPacked, int bufPitch, int bufSliceBytes, int bufW, int bufH) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ybuf = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bufW || ybuf >= bufH) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int yCur = 2 * ybuf;
    const int yNext = 2 * ybuf + 2;
    const Type *cur = (const Type *)(src + ((yCur < height) ? yCur : height - 1) * srcPitch);
    const Type *next = (const Type *)(src + ((yNext < height) ? yNext : height - 1) * srcPitch);

    const int cM3 = maa_load_pix_clamp(cur,  x - 3, width);
    const int cM2 = maa_load_pix_clamp(cur,  x - 2, width);
    const int cM1 = maa_load_pix_clamp(cur,  x - 1, width);
    const int cP0 = maa_load_pix_clamp(cur,  x,     width);
    const int cP1 = maa_load_pix_clamp(cur,  x + 1, width);
    const int cP2 = maa_load_pix_clamp(cur,  x + 2, width);
    const int cP3 = maa_load_pix_clamp(cur,  x + 3, width);

    const int nM3 = maa_load_pix_clamp(next, x - 3, width);
    const int nM2 = maa_load_pix_clamp(next, x - 2, width);
    const int nM1 = maa_load_pix_clamp(next, x - 1, width);
    const int nP0 = maa_load_pix_clamp(next, x,     width);
    const int nP1 = maa_load_pix_clamp(next, x + 1, width);
    const int nP2 = maa_load_pix_clamp(next, x + 2, width);
    const int nP3 = maa_load_pix_clamp(next, x + 3, width);

    const int fwdCur = maa_sn3<Type>(cM1, cP0, cP1, max_val);
    const int fwdNext = maa_sn3<Type>(nP1, nP0, nM1, max_val);
    const int bwdCur = maa_sn3<Type>(cP1, cP0, cM1, max_val);
    const int bwdNext = maa_sn3<Type>(nM1, nP0, nP1, max_val);

    const int rowOffset = ybuf * bufPitch;
    ((Type *)(costPacked + 0 * bufSliceBytes + rowOffset))[x] = (Type)abs(cM3 - nP3);
    ((Type *)(costPacked + 1 * bufSliceBytes + rowOffset))[x] = (Type)abs(cM2 - nP2);
    ((Type *)(costPacked + 2 * bufSliceBytes + rowOffset))[x] = (Type)abs(cM1 - nP1);
    ((Type *)(costPacked + 3 * bufSliceBytes + rowOffset))[x] = (Type)abs(fwdCur - fwdNext);
    ((Type *)(costPacked + 4 * bufSliceBytes + rowOffset))[x] = (Type)abs(cP0 - nP0);
    ((Type *)(costPacked + 5 * bufSliceBytes + rowOffset))[x] = (Type)abs(bwdCur - bwdNext);
    ((Type *)(costPacked + 6 * bufSliceBytes + rowOffset))[x] = (Type)abs(cP1 - nM1);
    ((Type *)(costPacked + 7 * bufSliceBytes + rowOffset))[x] = (Type)abs(cP2 - nM2);
    ((Type *)(costPacked + 8 * bufSliceBytes + rowOffset))[x] = (Type)abs(cP3 - nM3);
}

template<typename Type, int bit_depth>
__global__ void kernel_maa_sangnom_smooth_3d(const uint8_t *costPacked, uint8_t *smoothPacked,
    int bufPitch, int bufSliceBytes, int bufW, int bufH) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ybuf = blockIdx.y * blockDim.y + threadIdx.y;
    const int bufIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= bufW || ybuf >= bufH || bufIndex >= 9) return;

    static const int max_val = (1 << bit_depth) - 1;
    const uint8_t *bufIn = costPacked + bufIndex * bufSliceBytes;
    uint8_t *bufOut = smoothPacked + bufIndex * bufSliceBytes;

    int hsum = 0;
    for (int dx = -3; dx <= 3; dx++) {
        const int xc = clamp(x + dx, 0, bufW - 1);
        int vsum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            const int yc = clamp(ybuf + dy, 0, bufH - 1);
            vsum += (int)*(const Type *)(bufIn + yc * bufPitch + xc * sizeof(Type));
        }
        hsum += vsum;
    }

    int out = hsum >> 4;
    out = clamp(out, 0, max_val);
    ((Type *)(bufOut + ybuf * bufPitch))[x] = (Type)out;
}

template<typename Type, int bit_depth>
__global__ void kernel_maa_sangnom_finalize(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, const uint8_t *smoothPacked,
    int bufPitch, int bufSliceBytes, int bufW, int bufH, float aaf) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    Type *dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    if ((y & 1) == 0) {
        dstPix[0] = (Type)maa_read_pix_clamp<Type>(src, srcPitch, width, height, x, y);
        return;
    }

    const int ybuf = y >> 1;
    const uint8_t *pixBase = smoothPacked + ybuf * bufPitch + x * sizeof(Type);
    const int b0 = (int)*(const Type *)(pixBase + 0 * bufSliceBytes);
    const int b1 = (int)*(const Type *)(pixBase + 1 * bufSliceBytes);
    const int b2 = (int)*(const Type *)(pixBase + 2 * bufSliceBytes);
    const int b3 = (int)*(const Type *)(pixBase + 3 * bufSliceBytes);
    const int b4 = (int)*(const Type *)(pixBase + 4 * bufSliceBytes);
    const int b5 = (int)*(const Type *)(pixBase + 5 * bufSliceBytes);
    const int b6 = (int)*(const Type *)(pixBase + 6 * bufSliceBytes);
    const int b7 = (int)*(const Type *)(pixBase + 7 * bufSliceBytes);
    const int b8 = (int)*(const Type *)(pixBase + 8 * bufSliceBytes);
    int minCost = b0;
    minCost = min(minCost, b1);
    minCost = min(minCost, b2);
    minCost = min(minCost, b3);
    minCost = min(minCost, b4);
    minCost = min(minCost, b5);
    minCost = min(minCost, b6);
    minCost = min(minCost, b7);
    minCost = min(minCost, b8);

    const int yCur = max(y - 1, 0);
    const int yNext = min(y + 1, height - 1);
    const Type *cur = (const Type *)(src + yCur * srcPitch);
    const Type *next = (const Type *)(src + yNext * srcPitch);

    const int cM3 = maa_load_pix_clamp(cur,  x - 3, width);
    const int cM2 = maa_load_pix_clamp(cur,  x - 2, width);
    const int cM1 = maa_load_pix_clamp(cur,  x - 1, width);
    const int cP0 = maa_load_pix_clamp(cur,  x,     width);
    const int cP1 = maa_load_pix_clamp(cur,  x + 1, width);
    const int cP2 = maa_load_pix_clamp(cur,  x + 2, width);
    const int cP3 = maa_load_pix_clamp(cur,  x + 3, width);
    const int nM3 = maa_load_pix_clamp(next, x - 3, width);
    const int nM2 = maa_load_pix_clamp(next, x - 2, width);
    const int nM1 = maa_load_pix_clamp(next, x - 1, width);
    const int nP0 = maa_load_pix_clamp(next, x,     width);
    const int nP1 = maa_load_pix_clamp(next, x + 1, width);
    const int nP2 = maa_load_pix_clamp(next, x + 2, width);
    const int nP3 = maa_load_pix_clamp(next, x + 3, width);

    int result;
    if (b4 == minCost || (float)minCost > aaf) {
        result = (cP0 + nP0 + 1) >> 1;
    } else if (b5 == minCost) {
        result = (maa_sn3<Type>(cP1, cP0, cM1, max_val) + maa_sn3<Type>(nM1, nP0, nP1, max_val) + 1) >> 1;
    } else if (b3 == minCost) {
        result = (maa_sn3<Type>(cM1, cP0, cP1, max_val) + maa_sn3<Type>(nP1, nP0, nM1, max_val) + 1) >> 1;
    } else if (b6 == minCost) {
        result = (cP1 + nM1 + 1) >> 1;
    } else if (b2 == minCost) {
        result = (cM1 + nP1 + 1) >> 1;
    } else if (b7 == minCost) {
        result = (cP2 + nM2 + 1) >> 1;
    } else if (b1 == minCost) {
        result = (cM2 + nP2 + 1) >> 1;
    } else if (b8 == minCost) {
        result = (cP3 + nM3 + 1) >> 1;
    } else {
        result = (cM3 + nP3 + 1) >> 1;
    }
    dstPix[0] = (Type)clamp(result, 0, max_val);
}

template<typename Type, int bit_depth, int edgeMode>
__global__ void kernel_maa_edge(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height, int mthresh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    Type *dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        dstPix[0] = (Type)0;
        return;
    }

    const Type *rowT = (const Type *)(src + (y - 1) * srcPitch);
    const Type *rowM = (const Type *)(src +  y      * srcPitch);
    const Type *rowB = (const Type *)(src + (y + 1) * srcPitch);
    const int tl = (int)rowT[x - 1], tc = (int)rowT[x], tr = (int)rowT[x + 1];
    const int cl = (int)rowM[x - 1], cc = (int)rowM[x], cr = (int)rowM[x + 1];
    const int bl = (int)rowB[x - 1], bc = (int)rowB[x], br = (int)rowB[x + 1];
    int mag = 0;

    if (edgeMode == NVEncFilterMaa::MAA_EDGE_SOBEL) {
        mag = abs((cr + bc) - (cl + tc)) >> 1;
    } else if (edgeMode == NVEncFilterMaa::MAA_EDGE_PREWITT) {
        const int gx = (-tl + tr) + (-cl + cr) + (-bl + br);
        const int gy = (-tl - tc - tr) + (bl + bc + br);
        mag = (abs(gx) + abs(gy)) / 6;
    } else if (edgeMode == NVEncFilterMaa::MAA_EDGE_SOBEL_FULL) {
        const int gx = (-tl + tr) + 2 * (-cl + cr) + (-bl + br);
        const int gy = (-tl - 2 * tc - tr) + (bl + 2 * bc + br);
        mag = (abs(gx) + abs(gy)) / 8;
    } else if (edgeMode == NVEncFilterMaa::MAA_EDGE_SCHARR) {
        const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
        const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
        mag = (abs(gx) + abs(gy)) / 32;
    } else if (edgeMode == NVEncFilterMaa::MAA_EDGE_KIRSCH) {
        const int n  =  5 * (tl + tc + tr) - 3 * (cl + cr + bl + bc + br);
        const int ne =  5 * (tc + tr + cr) - 3 * (tl + cl + bl + bc + br);
        const int e  =  5 * (tr + cr + br) - 3 * (tl + tc + cl + bl + bc);
        const int se =  5 * (cr + br + bc) - 3 * (tl + tc + tr + cl + bl);
        const int s  =  5 * (bl + bc + br) - 3 * (tl + tc + tr + cl + cr);
        const int sw =  5 * (cl + bl + bc) - 3 * (tl + tc + tr + cr + br);
        const int w  =  5 * (tl + cl + bl) - 3 * (tc + tr + cr + bc + br);
        const int nw =  5 * (tl + tc + cl) - 3 * (tr + cr + bl + bc + br);
        mag = max(max(max(n, ne), max(e, se)), max(max(s, sw), max(w, nw)));
        mag = max(mag, 0) / 15;
    } else {
        mag = abs(4 * cc - tc - cl - cr - bc) / 4;
    }
    dstPix[0] = (Type)((mag >= mthresh) ? max_val : 0);
}

template<typename Type>
__global__ void kernel_maa_inflate(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    Type *dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        dstPix[0] = *(const Type *)(src + y * srcPitch + x * sizeof(Type));
        return;
    }

    const Type *rowAbove = (const Type *)(src + (y - 1) * srcPitch);
    const Type *rowMid = (const Type *)(src + y * srcPitch);
    const Type *rowBelow = (const Type *)(src + (y + 1) * srcPitch);
    const int mean8 = ((int)rowAbove[x - 1] + rowAbove[x] + rowAbove[x + 1]
        + rowMid[x - 1] + rowMid[x + 1]
        + rowBelow[x - 1] + rowBelow[x] + rowBelow[x + 1]) >> 3;
    dstPix[0] = (Type)max(mean8, (int)rowMid[x]);
}

template<typename Type, int bit_depth>
__global__ void kernel_maa_merge(const uint8_t *srcA, const int srcAPitch, const uint8_t *srcB, const int srcBPitch,
    const uint8_t *mask, const int maskPitch, uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int a = (int)*(const Type *)(srcA + y * srcAPitch + x * sizeof(Type));
    const int b = (int)*(const Type *)(srcB + y * srcBPitch + x * sizeof(Type));
    const int m = (int)*(const Type *)(mask + y * maskPitch + x * sizeof(Type));
    Type *dPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));

    if (m == 0) {
        dPix[0] = (Type)a;
        return;
    }
    if (m == max_val) {
        dPix[0] = (Type)b;
        return;
    }
    const long long maxvP1 = (long long)max_val + 1;
    const long long blended = ((maxvP1 - (long long)m) * (long long)a
        + (long long)m * (long long)b + (maxvP1 >> 1)) >> bit_depth;
    dPix[0] = (Type)clamp((int)blended, 0, max_val);
}

template<typename Type>
__global__ void kernel_maa_mask_subsample(const uint8_t *lumaMask, const int lumaPitch, const int lumaWidth, const int lumaHeight,
    uint8_t *chromaMask, const int chromaPitch, const int chromaWidth, const int chromaHeight, int subSampleX, int subSampleY) {
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= chromaWidth || cy >= chromaHeight) return;

    const int xBase = cx * subSampleX;
    const int yBase = cy * subSampleY;
    int sum = 0;
    int count = 0;
    for (int dy = 0; dy < subSampleY; dy++) {
        const int yi = yBase + dy;
        if (yi >= lumaHeight) continue;
        const Type *row = (const Type *)(lumaMask + yi * lumaPitch);
        for (int dx = 0; dx < subSampleX; dx++) {
            const int xi = xBase + dx;
            if (xi >= lumaWidth) continue;
            sum += (int)row[xi];
            count++;
        }
    }
    Type *dst = (Type *)(chromaMask + cy * chromaPitch + cx * sizeof(Type));
    dst[0] = (Type)((count > 0) ? (sum / count) : 0);
}

template<typename Type, int bit_depth>
__global__ void kernel_maa_show_overlay(const uint8_t *src, const int srcPitch, const uint8_t *mask, const int maskPitch,
    uint8_t *dst, const int dstPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    static const int max_val = (1 << bit_depth) - 1;
    const int s = (int)*(const Type *)(src + y * srcPitch + x * sizeof(Type));
    const int m = (int)*(const Type *)(mask + y * maskPitch + x * sizeof(Type));
    Type *dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)min((s >> 1) + (m >> 1), max_val);
}

template<typename Type>
__global__ void kernel_maa_show_darken(const uint8_t *src, const int srcPitch, uint8_t *dst, const int dstPitch,
    const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int s = (int)*(const Type *)(src + y * srcPitch + x * sizeof(Type));
    Type *dstPix = (Type *)(dst + y * dstPitch + x * sizeof(Type));
    dstPix[0] = (Type)(s >> 1);
}

template<typename Type>
static RGY_ERR maa_fturn_left_plane_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_fturn_left<Type><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0], pSrc->width,
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR maa_fturn_right_plane_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_fturn_right<Type><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0], pSrc->height,
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR maa_sangnom_plane_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    CUMemBuf *costRaw, CUMemBuf *costSmooth, int costPitch, int costSliceBytes, float aaf, cudaStream_t stream) {
    const int bufW = pSrc->width;
    const int bufH = pSrc->height / 2;
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridCost(divCeil(bufW, blockSize.x), divCeil(bufH, blockSize.y));
    dim3 gridFrame(divCeil(pSrc->width, blockSize.x), divCeil(pSrc->height, blockSize.y));

    kernel_maa_sangnom_prepare<Type, bit_depth><<<gridCost, blockSize, 0, stream>>>(
        pSrc->ptr[0], pSrc->pitch[0], pSrc->width, pSrc->height, (uint8_t *)costRaw->ptr, costPitch, costSliceBytes, bufW, bufH);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    dim3 blockSmooth(MAA_BLOCK_X, MAA_BLOCK_Y, 1);
    dim3 gridSmooth(divCeil(bufW, blockSmooth.x), divCeil(bufH, blockSmooth.y), 9);
    kernel_maa_sangnom_smooth_3d<Type, bit_depth><<<gridSmooth, blockSmooth, 0, stream>>>(
        (const uint8_t *)costRaw->ptr, (uint8_t *)costSmooth->ptr, costPitch, costSliceBytes, bufW, bufH);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_maa_sangnom_finalize<Type, bit_depth><<<gridFrame, blockSize, 0, stream>>>(
        pSrc->ptr[0], pSrc->pitch[0], pDst->ptr[0], pDst->pitch[0], pSrc->width, pSrc->height,
        (const uint8_t *)costSmooth->ptr, costPitch, costSliceBytes, bufW, bufH, aaf);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth, int edgeMode>
static RGY_ERR maa_edge_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int mthreshScaled, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_edge<Type, bit_depth, edgeMode><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0],
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height, mthreshScaled);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR maa_edge_dispatch(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    int mthreshScaled, NVEncFilterMaa::MaaEdgeMode edgeMode, cudaStream_t stream) {
    switch (edgeMode) {
    case NVEncFilterMaa::MAA_EDGE_SOBEL:      return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_SOBEL>(pDst, pSrc, mthreshScaled, stream);
    case NVEncFilterMaa::MAA_EDGE_PREWITT:    return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_PREWITT>(pDst, pSrc, mthreshScaled, stream);
    case NVEncFilterMaa::MAA_EDGE_SOBEL_FULL: return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_SOBEL_FULL>(pDst, pSrc, mthreshScaled, stream);
    case NVEncFilterMaa::MAA_EDGE_SCHARR:     return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_SCHARR>(pDst, pSrc, mthreshScaled, stream);
    case NVEncFilterMaa::MAA_EDGE_KIRSCH:     return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_KIRSCH>(pDst, pSrc, mthreshScaled, stream);
    case NVEncFilterMaa::MAA_EDGE_LAPLACIAN:  return maa_edge_typed<Type, bit_depth, NVEncFilterMaa::MAA_EDGE_LAPLACIAN>(pDst, pSrc, mthreshScaled, stream);
    default: return RGY_ERR_INVALID_PARAM;
    }
}

template<typename Type>
static RGY_ERR maa_inflate_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_inflate<Type><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0],
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR maa_merge_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA, const RGYFrameInfo *pSrcB,
    const RGYFrameInfo *pMask, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_merge<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pSrcA->ptr[0], pSrcA->pitch[0],
        pSrcB->ptr[0], pSrcB->pitch[0], pMask->ptr[0], pMask->pitch[0],
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR maa_mask_subsample_typed(RGYFrameInfo *pChromaMask, const RGYFrameInfo *pLumaMask,
    int subSampleX, int subSampleY, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pChromaMask->width, blockSize.x), divCeil(pChromaMask->height, blockSize.y));
    kernel_maa_mask_subsample<Type><<<gridSize, blockSize, 0, stream>>>(pLumaMask->ptr[0], pLumaMask->pitch[0],
        pLumaMask->width, pLumaMask->height, pChromaMask->ptr[0], pChromaMask->pitch[0],
        pChromaMask->width, pChromaMask->height, subSampleX, subSampleY);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type, int bit_depth>
static RGY_ERR maa_show_overlay_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pMask, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_show_overlay<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0],
        pMask->ptr[0], pMask->pitch[0], pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR maa_show_darken_typed(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    dim3 blockSize(MAA_BLOCK_X, MAA_BLOCK_Y);
    dim3 gridSize(divCeil(pDst->width, blockSize.x), divCeil(pDst->height, blockSize.y));
    kernel_maa_show_darken<Type><<<gridSize, blockSize, 0, stream>>>(pSrc->ptr[0], pSrc->pitch[0],
        pDst->ptr[0], pDst->pitch[0], pDst->width, pDst->height);
    return err_to_rgy(cudaGetLastError());
}

static RGY_ERR maa_sangnom_plane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    CUMemBuf *costRaw, CUMemBuf *costSmooth, int costPitch, int costSliceBytes, float aaf, cudaStream_t stream) {
    switch (RGY_CSP_BIT_DEPTH[pSrc->csp]) {
    case 8:  return maa_sangnom_plane_typed<uint8_t, 8>(pDst, pSrc, costRaw, costSmooth, costPitch, costSliceBytes, aaf, stream);
    case 10: return maa_sangnom_plane_typed<uint16_t, 10>(pDst, pSrc, costRaw, costSmooth, costPitch, costSliceBytes, aaf, stream);
    case 12: return maa_sangnom_plane_typed<uint16_t, 12>(pDst, pSrc, costRaw, costSmooth, costPitch, costSliceBytes, aaf, stream);
    case 14: return maa_sangnom_plane_typed<uint16_t, 14>(pDst, pSrc, costRaw, costSmooth, costPitch, costSliceBytes, aaf, stream);
    case 16: return maa_sangnom_plane_typed<uint16_t, 16>(pDst, pSrc, costRaw, costSmooth, costPitch, costSliceBytes, aaf, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

static RGY_ERR maa_merge_plane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA, const RGYFrameInfo *pSrcB,
    const RGYFrameInfo *pMask, cudaStream_t stream) {
    switch (RGY_CSP_BIT_DEPTH[pDst->csp]) {
    case 8:  return maa_merge_typed<uint8_t, 8>(pDst, pSrcA, pSrcB, pMask, stream);
    case 10: return maa_merge_typed<uint16_t, 10>(pDst, pSrcA, pSrcB, pMask, stream);
    case 12: return maa_merge_typed<uint16_t, 12>(pDst, pSrcA, pSrcB, pMask, stream);
    case 14: return maa_merge_typed<uint16_t, 14>(pDst, pSrcA, pSrcB, pMask, stream);
    case 16: return maa_merge_typed<uint16_t, 16>(pDst, pSrcA, pSrcB, pMask, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

static RGY_ERR maa_show_overlay_plane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pMask, cudaStream_t stream) {
    switch (RGY_CSP_BIT_DEPTH[pDst->csp]) {
    case 8:  return maa_show_overlay_typed<uint8_t, 8>(pDst, pSrc, pMask, stream);
    case 10: return maa_show_overlay_typed<uint16_t, 10>(pDst, pSrc, pMask, stream);
    case 12: return maa_show_overlay_typed<uint16_t, 12>(pDst, pSrc, pMask, stream);
    case 14: return maa_show_overlay_typed<uint16_t, 14>(pDst, pSrc, pMask, stream);
    case 16: return maa_show_overlay_typed<uint16_t, 16>(pDst, pSrc, pMask, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

NVEncFilterMaa::NVEncFilterMaa() :
    m_resizeUp(),
    m_resizeDown(),
    m_resizeUpLuma(),
    m_resizeDownLuma(),
    m_supersampled(),
    m_rotated(),
    m_rotatedAA(),
    m_unrotatedAA(),
    m_aaResult(),
    m_edgeMask(),
    m_inflatedMask(),
    m_chromaMask(),
    m_costRawPacked(),
    m_costSmoothPacked(),
    m_costPitch(0),
    m_costSliceBytes(0),
    m_costElemBytes(1),
    m_ssW(0),
    m_ssH(0),
    m_aaf(0.0f),
    m_aacf(0.0f),
    m_mthreshScaled(0),
    m_edgeMode(MAA_EDGE_SOBEL) {
    m_name = _T("maa");
}

NVEncFilterMaa::~NVEncFilterMaa() {
    close();
}

RGY_ERR NVEncFilterMaa::checkParam(const std::shared_ptr<NVEncFilterParamMaa> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("MAA requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.ss < 1.0f || prm->maa.ss > 4.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.3f: must be in [1.0, 4.0].\n"), prm->maa.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.aa < 0 || prm->maa.aa > 255 || prm->maa.aac < 0 || prm->maa.aac > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid aa/aac: must be in [0, 255].\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.mthresh < 1 || prm->maa.mthresh > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mthresh=%d: must be in [1, 255].\n"), prm->maa.mthresh);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.show < 0 || prm->maa.show > 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid show=%d: must be in [0, 2].\n"), prm->maa.show);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.edge != _T("sobel")
        && prm->maa.edge != _T("prewitt")
        && prm->maa.edge != _T("sobel_full")
        && prm->maa.edge != _T("scharr")
        && prm->maa.edge != _T("kirsch")
        && prm->maa.edge != _T("laplacian")) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid edge=%s: must be sobel|prewitt|sobel_full|scharr|kirsch|laplacian.\n"),
            prm->maa.edge.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::allocWorkFrame(std::unique_ptr<CUFrameBuf>& frame, const RGYFrameInfo& frameInfo, const TCHAR *label) {
    if (!frame || frame->frame.width != frameInfo.width || frame->frame.height != frameInfo.height || frame->frame.csp != frameInfo.csp) {
        frame = std::make_unique<CUFrameBuf>(frameInfo);
        frame->releasePtr();
        const auto sts = frame->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate MAA %s buffer: %s.\n"), label, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::allocWorkBuf(std::unique_ptr<CUMemBuf>& buf, size_t bytes, const TCHAR *label) {
    if (!buf || buf->nSize != bytes) {
        buf = std::make_unique<CUMemBuf>(bytes);
        const auto sts = buf->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate MAA %s buffer (%llu bytes): %s.\n"),
                label, (unsigned long long)bytes, get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamMaa>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    if      (prm->maa.edge == _T("sobel"))      m_edgeMode = MAA_EDGE_SOBEL;
    else if (prm->maa.edge == _T("prewitt"))    m_edgeMode = MAA_EDGE_PREWITT;
    else if (prm->maa.edge == _T("sobel_full")) m_edgeMode = MAA_EDGE_SOBEL_FULL;
    else if (prm->maa.edge == _T("scharr"))     m_edgeMode = MAA_EDGE_SCHARR;
    else if (prm->maa.edge == _T("kirsch"))     m_edgeMode = MAA_EDGE_KIRSCH;
    else if (prm->maa.edge == _T("laplacian"))  m_edgeMode = MAA_EDGE_LAPLACIAN;

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 4);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_ssW = std::max(4, alignSsDim(prm->frameIn.width, prm->maa.ss));
    m_ssH = std::max(4, alignSsDim(prm->frameIn.height, prm->maa.ss));
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int peak = (1 << bitDepth) - 1;
    m_aaf = (float)prm->maa.aa * (float)peak / 256.0f;
    m_aacf = (float)prm->maa.aac * (float)peak / 256.0f;
    m_mthreshScaled = prm->maa.mthresh * peak / 255;

    if (prm->maa.chroma) {
        auto prmUp = std::make_shared<NVEncFilterParamResize>();
        prmUp->frameIn = prm->frameIn;
        prmUp->frameOut = prm->frameIn;
        prmUp->frameOut.width = m_ssW;
        prmUp->frameOut.height = m_ssH;
        prmUp->interp = RGY_VPP_RESIZE_SPLINE36;
        prmUp->baseFps = prm->baseFps;
        prmUp->bOutOverwrite = false;
        m_resizeUp = std::make_unique<NVEncFilterResize>();
        sts = m_resizeUp->init(prmUp, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;

        auto prmDn = std::make_shared<NVEncFilterParamResize>();
        prmDn->frameIn = prm->frameIn;
        prmDn->frameIn.width = m_ssW;
        prmDn->frameIn.height = m_ssH;
        prmDn->frameOut = prm->frameOut;
        prmDn->interp = RGY_VPP_RESIZE_SPLINE36;
        prmDn->baseFps = prm->baseFps;
        prmDn->bOutOverwrite = false;
        m_resizeDown = std::make_unique<NVEncFilterResize>();
        sts = m_resizeDown->init(prmDn, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;
        m_resizeUpLuma.reset();
        m_resizeDownLuma.reset();
    } else {
        const RGY_CSP lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        auto prmUp = std::make_shared<NVEncFilterParamResize>();
        prmUp->frameIn = prm->frameIn;
        prmUp->frameIn.csp = lumaCsp;
        prmUp->frameOut = prmUp->frameIn;
        prmUp->frameOut.width = m_ssW;
        prmUp->frameOut.height = m_ssH;
        prmUp->interp = RGY_VPP_RESIZE_SPLINE36;
        prmUp->baseFps = prm->baseFps;
        prmUp->bOutOverwrite = false;
        m_resizeUpLuma = std::make_unique<NVEncFilterResize>();
        sts = m_resizeUpLuma->init(prmUp, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;

        auto prmDn = std::make_shared<NVEncFilterParamResize>();
        prmDn->frameIn = prm->frameIn;
        prmDn->frameIn.csp = lumaCsp;
        prmDn->frameIn.width = m_ssW;
        prmDn->frameIn.height = m_ssH;
        prmDn->frameOut = prm->frameOut;
        prmDn->frameOut.csp = lumaCsp;
        prmDn->interp = RGY_VPP_RESIZE_SPLINE36;
        prmDn->baseFps = prm->baseFps;
        prmDn->bOutOverwrite = false;
        m_resizeDownLuma = std::make_unique<NVEncFilterResize>();
        sts = m_resizeDownLuma->init(prmDn, m_pLog);
        if (sts != RGY_ERR_NONE) return sts;
        m_resizeUp.reset();
        m_resizeDown.reset();
    }

    RGYFrameInfo ssInfo = prm->frameIn;
    ssInfo.width = m_ssW;
    ssInfo.height = m_ssH;
    sts = allocWorkFrame(m_supersampled, ssInfo, _T("supersampled"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_unrotatedAA, ssInfo, _T("unrotatedAA"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_aaResult, ssInfo, _T("aaResult"));
    if (sts != RGY_ERR_NONE) return sts;

    RGYFrameInfo rotInfo = prm->frameIn;
    rotInfo.width = m_ssH;
    rotInfo.height = m_ssW;
    sts = allocWorkFrame(m_rotated, rotInfo, _T("rotated"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_rotatedAA, rotInfo, _T("rotatedAA"));
    if (sts != RGY_ERR_NONE) return sts;

    sts = allocWorkFrame(m_edgeMask, prm->frameIn, _T("edgeMask"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkFrame(m_inflatedMask, prm->frameIn, _T("inflatedMask"));
    if (sts != RGY_ERR_NONE) return sts;
    if (prm->maa.chroma) {
        sts = allocWorkFrame(m_chromaMask, prm->frameIn, _T("chromaMask"));
        if (sts != RGY_ERR_NONE) return sts;
    } else {
        m_chromaMask.reset();
    }

    m_costElemBytes = (bitDepth > 8) ? (int)sizeof(uint16_t) : (int)sizeof(uint8_t);
    const int costMaxW = std::max(m_ssW, m_ssH);
    const int costMaxBufRows = std::max(m_ssW, m_ssH) / 2;
    m_costPitch = costMaxW * m_costElemBytes;
    m_costSliceBytes = m_costPitch * costMaxBufRows;
    const size_t totalCostBytes = (size_t)m_costSliceBytes * (size_t)MAA_NUM_COST_BUFFERS;
    sts = allocWorkBuf(m_costRawPacked, totalCostBytes, _T("costRawPacked"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocWorkBuf(m_costSmoothPacked, totalCostBytes, _T("costSmoothPacked"));
    if (sts != RGY_ERR_NONE) return sts;

    setFilterInfo(prm->print() + strsprintf(_T(" (ssDims=%dx%d)"), m_ssW, m_ssH));
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::fturnLeftFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int planeCount, cudaStream_t stream) {
    const int srcPlanes = RGY_CSP_PLANES[pSrc->csp];
    const int planes = (planeCount < 0) ? srcPlanes : std::min(planeCount, srcPlanes);
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto srcPlane = getPlane(pSrc, (RGY_PLANE)iplane);
        auto dstPlane = getPlane(pDst, (RGY_PLANE)iplane);
        auto err = (RGY_CSP_BIT_DEPTH[pSrc->csp] > 8)
            ? maa_fturn_left_plane_typed<uint16_t>(&dstPlane, &srcPlane, stream)
            : maa_fturn_left_plane_typed<uint8_t>(&dstPlane, &srcPlane, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_fturn_left (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::fturnRightFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int planeCount, cudaStream_t stream) {
    const int srcPlanes = RGY_CSP_PLANES[pSrc->csp];
    const int planes = (planeCount < 0) ? srcPlanes : std::min(planeCount, srcPlanes);
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto srcPlane = getPlane(pSrc, (RGY_PLANE)iplane);
        auto dstPlane = getPlane(pDst, (RGY_PLANE)iplane);
        auto err = (RGY_CSP_BIT_DEPTH[pSrc->csp] > 8)
            ? maa_fturn_right_plane_typed<uint16_t>(&dstPlane, &srcPlane, stream)
            : maa_fturn_right_plane_typed<uint8_t>(&dstPlane, &srcPlane, stream);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_fturn_right (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterMaa::sangnomPassPlane(const RGYFrameInfo *pSrc, RGYFrameInfo *pDst, RGY_PLANE plane, float aaf, cudaStream_t stream) {
    const auto srcPlane = getPlane(pSrc, plane);
    auto dstPlane = getPlane(pDst, plane);
    auto err = maa_sangnom_plane(&dstPlane, &srcPlane,
        m_costRawPacked.get(), m_costSmoothPacked.get(), m_costPitch, m_costSliceBytes, aaf, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_sangnom (plane %d): %s.\n"), (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runEdge(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, int mthreshScaled, cudaStream_t stream) {
    const auto srcPlane = getPlane(pSrc, RGY_PLANE_Y);
    auto dstPlane = getPlane(pDst, RGY_PLANE_Y);
    RGY_ERR err = RGY_ERR_UNSUPPORTED;
    switch (RGY_CSP_BIT_DEPTH[srcPlane.csp]) {
    case 8:  err = maa_edge_dispatch<uint8_t, 8>(&dstPlane, &srcPlane, mthreshScaled, m_edgeMode, stream); break;
    case 10: err = maa_edge_dispatch<uint16_t, 10>(&dstPlane, &srcPlane, mthreshScaled, m_edgeMode, stream); break;
    case 12: err = maa_edge_dispatch<uint16_t, 12>(&dstPlane, &srcPlane, mthreshScaled, m_edgeMode, stream); break;
    case 14: err = maa_edge_dispatch<uint16_t, 14>(&dstPlane, &srcPlane, mthreshScaled, m_edgeMode, stream); break;
    case 16: err = maa_edge_dispatch<uint16_t, 16>(&dstPlane, &srcPlane, mthreshScaled, m_edgeMode, stream); break;
    default: break;
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_edge: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runInflate(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, cudaStream_t stream) {
    const auto srcPlane = getPlane(pSrc, RGY_PLANE_Y);
    auto dstPlane = getPlane(pDst, RGY_PLANE_Y);
    auto err = (RGY_CSP_BIT_DEPTH[srcPlane.csp] > 8)
        ? maa_inflate_typed<uint16_t>(&dstPlane, &srcPlane, stream)
        : maa_inflate_typed<uint8_t>(&dstPlane, &srcPlane, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_inflate: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runMergePlane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA, const RGYFrameInfo *pSrcB,
    const RGYFrameInfo *pMask, RGY_PLANE plane, cudaStream_t stream) {
    const auto srcAPlane = getPlane(pSrcA, plane);
    const auto srcBPlane = getPlane(pSrcB, plane);
    const auto maskPlane = getPlane(pMask, (plane == RGY_PLANE_Y) ? RGY_PLANE_Y : RGY_PLANE_U);
    auto dstPlane = getPlane(pDst, plane);
    auto err = maa_merge_plane(&dstPlane, &srcAPlane, &srcBPlane, &maskPlane, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_merge (plane %d): %s.\n"), (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runMaskSubsample(RGYFrameInfo *pChromaMaskDst, const RGYFrameInfo *pLumaMaskSrc, cudaStream_t stream) {
    const auto luma = getPlane(pLumaMaskSrc, RGY_PLANE_Y);
    auto chroma = getPlane(pChromaMaskDst, RGY_PLANE_U);
    const int subSampleX = (chroma.width > 0) ? std::max(1, luma.width / chroma.width) : 1;
    const int subSampleY = (chroma.height > 0) ? std::max(1, luma.height / chroma.height) : 1;
    auto err = (RGY_CSP_BIT_DEPTH[luma.csp] > 8)
        ? maa_mask_subsample_typed<uint16_t>(&chroma, &luma, subSampleX, subSampleY, stream)
        : maa_mask_subsample_typed<uint8_t>(&chroma, &luma, subSampleX, subSampleY, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_mask_subsample: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runShowOverlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pMask,
    RGY_PLANE plane, cudaStream_t stream) {
    const auto srcPlane = getPlane(pSrc, plane);
    const auto maskPlane = getPlane(pMask, RGY_PLANE_Y);
    auto dstPlane = getPlane(pDst, plane);
    auto err = maa_show_overlay_plane(&dstPlane, &srcPlane, &maskPlane, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_show_overlay (plane %d): %s.\n"), (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::runShowDarken(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane, cudaStream_t stream) {
    const auto srcPlane = getPlane(pSrc, plane);
    auto dstPlane = getPlane(pDst, plane);
    auto err = (RGY_CSP_BIT_DEPTH[dstPlane.csp] > 8)
        ? maa_show_darken_typed<uint16_t>(&dstPlane, &srcPlane, stream)
        : maa_show_darken_typed<uint8_t>(&dstPlane, &srcPlane, stream);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_show_darken (plane %d): %s.\n"), (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterMaa::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        *pOutputFrameNum = 1;
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamMaa>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool processChroma = prm->maa.chroma;
    const bool maskOn = prm->maa.mask;
    const int showMode = prm->maa.show;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const bool needMask = maskOn || (showMode > 0);
    const int aaPlanes = processChroma ? planes : 1;
    const RGY_CSP lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;

    RGYFrameInfo *pSupersampled = &m_supersampled->frame;
    {
        int resizeOutNum = 0;
        if (processChroma) {
            RGYFrameInfo *resizeOut[1] = { pSupersampled };
            sts = m_resizeUp->filter(const_cast<RGYFrameInfo *>(pInputFrame), resizeOut, &resizeOutNum, stream);
        } else {
            RGYFrameInfo inputLuma = getPlane(pInputFrame, RGY_PLANE_Y);
            RGYFrameInfo outputLuma = getPlane(pSupersampled, RGY_PLANE_Y);
            inputLuma.csp = lumaCsp;
            outputLuma.csp = lumaCsp;
            RGYFrameInfo *resizeOut[1] = { &outputLuma };
            sts = m_resizeUpLuma->filter(&inputLuma, resizeOut, &resizeOutNum, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("MAA resize-up failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    RGYFrameInfo *pRotated = &m_rotated->frame;
    sts = fturnLeftFrame(pRotated, pSupersampled, aaPlanes, stream);
    if (sts != RGY_ERR_NONE) return sts;

    RGYFrameInfo *pRotatedAA = &m_rotatedAA->frame;
    if (processChroma) {
        sts = copyFrameAsync(pRotatedAA, pRotated, stream);
    } else {
        auto srcPlane = getPlane(pRotated, RGY_PLANE_Y);
        auto dstPlane = getPlane(pRotatedAA, RGY_PLANE_Y);
        sts = copyPlaneAsync(&dstPlane, &srcPlane, stream);
    }
    if (sts != RGY_ERR_NONE) return sts;
    sts = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_Y, m_aaf, stream);
    if (sts != RGY_ERR_NONE) return sts;
    if (processChroma && planes >= 3) {
        sts = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_U, m_aacf, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_V, m_aacf, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    RGYFrameInfo *pUnrotatedAA = &m_unrotatedAA->frame;
    sts = fturnRightFrame(pUnrotatedAA, pRotatedAA, aaPlanes, stream);
    if (sts != RGY_ERR_NONE) return sts;

    RGYFrameInfo *pAaResult = &m_aaResult->frame;
    if (processChroma) {
        sts = copyFrameAsync(pAaResult, pUnrotatedAA, stream);
    } else {
        auto srcPlane = getPlane(pUnrotatedAA, RGY_PLANE_Y);
        auto dstPlane = getPlane(pAaResult, RGY_PLANE_Y);
        sts = copyPlaneAsync(&dstPlane, &srcPlane, stream);
    }
    if (sts != RGY_ERR_NONE) return sts;
    sts = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_Y, m_aaf, stream);
    if (sts != RGY_ERR_NONE) return sts;
    if (processChroma && planes >= 3) {
        sts = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_U, m_aacf, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_V, m_aacf, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    RGYFrameInfo *pOut = ppOutputFrames[0];
    RGYFrameInfo *pAaSrcRes = pOut;
    {
        int resizeOutNum = 0;
        if (processChroma) {
            RGYFrameInfo *resizeOut[1] = { pOut };
            sts = m_resizeDown->filter(pAaResult, resizeOut, &resizeOutNum, stream);
        } else {
            RGYFrameInfo aaLuma = getPlane(pAaResult, RGY_PLANE_Y);
            RGYFrameInfo outLuma = getPlane(pOut, RGY_PLANE_Y);
            aaLuma.csp = lumaCsp;
            outLuma.csp = lumaCsp;
            RGYFrameInfo *resizeOut[1] = { &outLuma };
            sts = m_resizeDownLuma->filter(&aaLuma, resizeOut, &resizeOutNum, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("MAA resize-down failed: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (needMask) {
        sts = runEdge(&m_edgeMask->frame, pInputFrame, m_mthreshScaled, stream);
        if (sts != RGY_ERR_NONE) return sts;
        sts = runInflate(&m_inflatedMask->frame, &m_edgeMask->frame, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }

    if (showMode == 1 || showMode == 2) {
        const RGYFrameInfo *underlayLuma = (showMode == 2) ? pAaSrcRes : pInputFrame;
        sts = runShowOverlay(&m_edgeMask->frame, underlayLuma, &m_inflatedMask->frame, RGY_PLANE_Y, stream);
        if (sts != RGY_ERR_NONE) return sts;
        {
            const auto srcP = getPlane(&m_edgeMask->frame, RGY_PLANE_Y);
            auto dstP = getPlane(pOut, RGY_PLANE_Y);
            sts = copyPlaneAsync(&dstP, &srcP, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (planes >= 3) {
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                sts = runShowDarken(&m_edgeMask->frame, pInputFrame, pl, stream);
                if (sts != RGY_ERR_NONE) return sts;
                const auto srcP = getPlane(&m_edgeMask->frame, pl);
                auto dstP = getPlane(pOut, pl);
                sts = copyPlaneAsync(&dstP, &srcP, stream);
                if (sts != RGY_ERR_NONE) return sts;
            }
        }
    } else if (maskOn) {
        sts = runMergePlane(&m_edgeMask->frame, pInputFrame, pAaSrcRes, &m_inflatedMask->frame, RGY_PLANE_Y, stream);
        if (sts != RGY_ERR_NONE) return sts;
        {
            const auto srcP = getPlane(&m_edgeMask->frame, RGY_PLANE_Y);
            auto dstP = getPlane(pOut, RGY_PLANE_Y);
            sts = copyPlaneAsync(&dstP, &srcP, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (processChroma && planes >= 3) {
            sts = runMaskSubsample(&m_chromaMask->frame, &m_inflatedMask->frame, stream);
            if (sts != RGY_ERR_NONE) return sts;
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                sts = runMergePlane(&m_edgeMask->frame, pInputFrame, pAaSrcRes, &m_chromaMask->frame, pl, stream);
                if (sts != RGY_ERR_NONE) return sts;
                const auto srcP = getPlane(&m_edgeMask->frame, pl);
                auto dstP = getPlane(pOut, pl);
                sts = copyPlaneAsync(&dstP, &srcP, stream);
                if (sts != RGY_ERR_NONE) return sts;
            }
        } else if (planes >= 3) {
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                const auto srcP = getPlane(pInputFrame, pl);
                auto dstP = getPlane(pOut, pl);
                sts = copyPlaneAsync(&dstP, &srcP, stream);
                if (sts != RGY_ERR_NONE) return sts;
            }
        }
    } else if (!processChroma && planes >= 3) {
        for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
            const auto srcP = getPlane(pInputFrame, pl);
            auto dstP = getPlane(pOut, pl);
            sts = copyPlaneAsync(&dstP, &srcP, stream);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }

    sts = copyPlaneAlphaAsync(pOut, pInputFrame, stream);
    if (sts != RGY_ERR_NONE) return sts;
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

void NVEncFilterMaa::close() {
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_resizeUpLuma.reset();
    m_resizeDownLuma.reset();
    m_supersampled.reset();
    m_rotated.reset();
    m_rotatedAA.reset();
    m_unrotatedAA.reset();
    m_aaResult.reset();
    m_edgeMask.reset();
    m_inflatedMask.reset();
    m_chromaMask.reset();
    m_costRawPacked.reset();
    m_costSmoothPacked.reset();
    m_costPitch = 0;
    m_costSliceBytes = 0;
    m_costElemBytes = 1;
    m_ssW = 0;
    m_ssH = 0;
    m_aaf = 0.0f;
    m_aacf = 0.0f;
    m_mthreshScaled = 0;
    m_edgeMode = MAA_EDGE_SOBEL;
    m_frameBuf.clear();
}
