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

template<typename Type>
__device__ int kfm_absdiff(const Type a, const Type b) {
    return abs((int)a - (int)b);
}

template<typename Type>
__device__ int kfm_calc_combe(
    const Type L0, const Type L1, const Type L2, const Type L3,
    const Type L4, const Type L5, const Type L6, const Type L7) {
    const int diff8 = kfm_absdiff(L0, L7);
    const int diffT =
        kfm_absdiff(L0, L1) + kfm_absdiff(L1, L2) + kfm_absdiff(L2, L3) + kfm_absdiff(L3, L4) +
        kfm_absdiff(L4, L5) + kfm_absdiff(L5, L6) + kfm_absdiff(L6, L7) - diff8;
    const int diffE =
        kfm_absdiff(L0, L2) + kfm_absdiff(L2, L4) + kfm_absdiff(L4, L6) + kfm_absdiff(L6, L7) - diff8;
    const int diffO =
        kfm_absdiff(L0, L1) + kfm_absdiff(L1, L3) + kfm_absdiff(L3, L5) + kfm_absdiff(L5, L7) - diff8;
    return diffT - diffE - diffO;
}

template<typename Type>
__device__ int kfm_calc_diff(
    const Type L00, const Type L10, const Type L01, const Type L11,
    const Type L02, const Type L12, const Type L03, const Type L13) {
    return kfm_absdiff(L00, L10) + kfm_absdiff(L01, L11) + kfm_absdiff(L02, L12) + kfm_absdiff(L03, L13);
}

__device__ int kfm_clamp_u8(const int v) {
    return clamp(v, 0, 255);
}

template<typename Type>
__device__ Type kfm_load_src(
    const Type *src,
    const int pitch,
    const int x,
    const int y,
    const int pixelStep,
    const int pixelOffset) {
    return src[x * pixelStep + pixelOffset + y * pitch];
}

template<typename Type>
__device__ uchar4 kfm_analyze_block(
    const uint8_t *src0,
    const uint8_t *src1,
    const int srcPitch,
    const int bitDepth,
    const int parity,
    const int pixelStep,
    const int pixelOffset,
    const int bx,
    const int by) {
    const int shift = bitDepth - 8 + 4;
    const int srcPitchT = srcPitch / (int)sizeof(Type);
    const Type *f0 = (const Type *)src0;
    const Type *f1 = (const Type *)src1;

    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    const int xBase = bx * 4;
    const int yBase = by * 4;

    for (int tx = 0; tx < 8; ++tx) {
        const int x = xBase + tx;
        const int y = yBase;

        {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            const int tmp = kfm_calc_combe(T00, B00, T01, B01, T02, B02, T03, B03);
            if (parity) {
                sum0 += tmp;
            } else {
                sum2 += tmp;
            }
        }

        if (parity) {
            const Type T10 = kfm_load_src(f1, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T11 = kfm_load_src(f1, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T12 = kfm_load_src(f1, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T13 = kfm_load_src(f1, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum2 += kfm_calc_combe(T10, B00, T11, B01, T12, B02, T13, B03);
        } else {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B10 = kfm_load_src(f1, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B11 = kfm_load_src(f1, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B12 = kfm_load_src(f1, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B13 = kfm_load_src(f1, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum0 += kfm_calc_combe(T00, B10, T01, B11, T02, B12, T03, B13);
        }

        {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type T10 = kfm_load_src(f1, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type T11 = kfm_load_src(f1, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type T12 = kfm_load_src(f1, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type T13 = kfm_load_src(f1, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            sum1 += kfm_calc_diff(T00, T10, T01, T11, T02, T12, T03, T13);
        }

        {
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type B10 = kfm_load_src(f1, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type B11 = kfm_load_src(f1, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type B12 = kfm_load_src(f1, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            const Type B13 = kfm_load_src(f1, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum3 += kfm_calc_diff(B00, B10, B01, B11, B02, B12, B03, B13);
        }
    }

    return make_uchar4(
        kfm_clamp_u8(sum0 >> shift),
        kfm_clamp_u8(sum1 >> shift),
        kfm_clamp_u8(sum2 >> shift),
        kfm_clamp_u8(sum3 >> shift));
}

__global__ void kernel_kfm_init_fmcount(RGYKFM::FMCount *dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2) return;
    dst[idx].move = 0;
    dst[idx].shima = 0;
    dst[idx].lshima = 0;
}

template<typename Type>
__global__ void kernel_kfm_analyze_count_cmflags_clean(
    RGYKFM::FMCount *dst,
    const uint8_t *prevSrc0,
    const uint8_t *prevSrc1,
    const uint8_t *curSrc0,
    const uint8_t *curSrc1,
    const int prevSrcPitch,
    const int curSrcPitch,
    const int bitDepth,
    const int width,
    const int height,
    const int prevParity,
    const int curParity,
    const int countParity,
    const int pixelStep,
    const int pixelOffset,
    const int threshM,
    const int threshS,
    const int threshLS,
    const int cleanThresh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uchar4 prevBlock = kfm_analyze_block<Type>(prevSrc0, prevSrc1, prevSrcPitch, bitDepth, prevParity, pixelStep, pixelOffset, x, y);
    const uchar4 curBlock = kfm_analyze_block<Type>(curSrc0, curSrc1, curSrcPitch, bitDepth, curParity, pixelStep, pixelOffset, x, y);

    const uchar2 prevField1 = make_uchar2(prevBlock.z, prevBlock.w);
    const uchar2 curField0 = make_uchar2(curBlock.x, curBlock.y);
    const uchar2 curField1 = make_uchar2(curBlock.z, curBlock.w);

    uchar2 vals[2];
    vals[0] = curField0;
    if (prevField1.y <= cleanThresh && vals[0].y <= cleanThresh) {
        vals[0].x = 0;
    }
    vals[1] = curField1;
    if (curField0.y <= cleanThresh && vals[1].y <= cleanThresh) {
        vals[1].x = 0;
    }

    for (int i = 0; i < 2; ++i) {
        const uchar2 v = vals[i];
        const int dstIdx = i ^ (countParity == 0);
        if (v.y >= threshM) atomicAdd(&dst[dstIdx].move, 1);
        if (v.x >= threshS) atomicAdd(&dst[dstIdx].shima, 1);
        if (v.x >= threshLS) atomicAdd(&dst[dstIdx].lshima, 1);
    }
}

RGY_ERR run_kfm_init_fmcount(RGYKFM::FMCount *dst, cudaStream_t stream) {
    if (!dst) {
        return RGY_ERR_INVALID_CALL;
    }
    kernel_kfm_init_fmcount<<<1, 2, 0, stream>>>(dst);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_analyze_count_cmflags_clean_t(
    RGYKFM::FMCount *dst,
    const RGYFrameInfo *prevSrc0,
    const RGYFrameInfo *prevSrc1,
    const RGYFrameInfo *curSrc0,
    const RGYFrameInfo *curSrc1,
    int width,
    int height,
    int prevParity,
    int curParity,
    int countParity,
    int pixelStep,
    int pixelOffset,
    int threshM,
    int threshS,
    int threshLS,
    int cleanThresh,
    cudaStream_t stream) {
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_analyze_count_cmflags_clean<Type><<<grid, block, 0, stream>>>(
        dst,
        (const uint8_t *)prevSrc0->ptr[0],
        (const uint8_t *)prevSrc1->ptr[0],
        (const uint8_t *)curSrc0->ptr[0],
        (const uint8_t *)curSrc1->ptr[0],
        prevSrc0->pitch[0],
        curSrc0->pitch[0],
        RGY_CSP_BIT_DEPTH[prevSrc0->csp],
        width,
        height,
        prevParity,
        curParity,
        countParity,
        pixelStep,
        pixelOffset,
        threshM,
        threshS,
        threshLS,
        cleanThresh);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_analyze_count_cmflags_clean(
    RGYKFM::FMCount *dst,
    const RGYFrameInfo *prevSrc0,
    const RGYFrameInfo *prevSrc1,
    const RGYFrameInfo *curSrc0,
    const RGYFrameInfo *curSrc1,
    int width,
    int height,
    int prevParity,
    int curParity,
    int countParity,
    int pixelStep,
    int pixelOffset,
    int threshM,
    int threshS,
    int threshLS,
    int cleanThresh,
    cudaStream_t stream) {
    if (!dst || !prevSrc0 || !prevSrc1 || !curSrc0 || !curSrc1) {
        return RGY_ERR_INVALID_CALL;
    }
    if (RGY_CSP_BIT_DEPTH[prevSrc0->csp] > 8) {
        return launch_kfm_analyze_count_cmflags_clean_t<uint16_t>(dst, prevSrc0, prevSrc1, curSrc0, curSrc1, width, height,
            prevParity, curParity, countParity, pixelStep, pixelOffset, threshM, threshS, threshLS, cleanThresh, stream);
    }
    return launch_kfm_analyze_count_cmflags_clean_t<uint8_t>(dst, prevSrc0, prevSrc1, curSrc0, curSrc1, width, height,
        prevParity, curParity, countParity, pixelStep, pixelOffset, threshM, threshS, threshLS, cleanThresh, stream);
}
