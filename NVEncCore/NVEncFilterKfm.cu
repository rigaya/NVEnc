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

__device__ int4 kfm_i4_add(const int4 a, const int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ int4 kfm_i4_sub(const int4 a, const int4 b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ int4 kfm_i4_mul(const int4 a, const int v) {
    return make_int4(a.x * v, a.y * v, a.z * v, a.w * v);
}

__device__ int4 kfm_i4_shr(const int4 a, const int v) {
    return make_int4(a.x >> v, a.y >> v, a.z >> v, a.w >> v);
}

__device__ int4 kfm_i4_min(const int4 a, const int4 b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

__device__ int4 kfm_i4_max(const int4 a, const int4 b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

__device__ int4 kfm_i4_abs(const int4 a) {
    return make_int4(abs(a.x), abs(a.y), abs(a.z), abs(a.w));
}

__device__ int4 kfm_i4_clamp(const int4 a, const int lo, const int hi) {
    return make_int4(clamp(a.x, lo, hi), clamp(a.y, lo, hi), clamp(a.z, lo, hi), clamp(a.w, lo, hi));
}

template<typename Type> struct KfmVec4Traits;

template<> struct KfmVec4Traits<uint8_t> {
    using Type4 = uchar4;
    __device__ static int4 to_int4(const Type4 v) {
        return make_int4(v.x, v.y, v.z, v.w);
    }
    __device__ static Type4 to_type4(const int4 v) {
        return make_uchar4(
            (uint8_t)clamp(v.x, 0, 255),
            (uint8_t)clamp(v.y, 0, 255),
            (uint8_t)clamp(v.z, 0, 255),
            (uint8_t)clamp(v.w, 0, 255));
    }
    __device__ static Type4 to_type4_float(const float4 v) {
        return make_uchar4(
            (uint8_t)clamp((int)v.x, 0, 255),
            (uint8_t)clamp((int)v.y, 0, 255),
            (uint8_t)clamp((int)v.z, 0, 255),
            (uint8_t)clamp((int)v.w, 0, 255));
    }
};

template<> struct KfmVec4Traits<uint16_t> {
    using Type4 = ushort4;
    __device__ static int4 to_int4(const Type4 v) {
        return make_int4(v.x, v.y, v.z, v.w);
    }
    __device__ static Type4 to_type4(const int4 v) {
        return make_ushort4(
            (uint16_t)clamp(v.x, 0, 65535),
            (uint16_t)clamp(v.y, 0, 65535),
            (uint16_t)clamp(v.z, 0, 65535),
            (uint16_t)clamp(v.w, 0, 65535));
    }
    __device__ static Type4 to_type4_float(const float4 v) {
        return make_ushort4(
            (uint16_t)clamp((int)v.x, 0, 65535),
            (uint16_t)clamp((int)v.y, 0, 65535),
            (uint16_t)clamp((int)v.z, 0, 65535),
            (uint16_t)clamp((int)v.w, 0, 65535));
    }
};

template<typename Type>
__device__ int4 kfm_static_calc_combe4(
    const typename KfmVec4Traits<Type>::Type4 a,
    const typename KfmVec4Traits<Type>::Type4 b,
    const typename KfmVec4Traits<Type>::Type4 c,
    const typename KfmVec4Traits<Type>::Type4 d,
    const typename KfmVec4Traits<Type>::Type4 e) {
    const int4 diff = kfm_i4_sub(
        kfm_i4_add(kfm_i4_add(KfmVec4Traits<Type>::to_int4(a), kfm_i4_mul(KfmVec4Traits<Type>::to_int4(c), 4)), KfmVec4Traits<Type>::to_int4(e)),
        kfm_i4_mul(kfm_i4_add(KfmVec4Traits<Type>::to_int4(b), KfmVec4Traits<Type>::to_int4(d)), 3));
    return kfm_i4_abs(diff);
}

template<typename Type>
__global__ void kernel_kfm_zero(
    uint8_t *dst,
    const int dstPitch,
    const int width,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = (Type)0;
}

template<typename Type>
__global__ void kernel_kfm_calc_combe(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width4,
    const int height,
    const int srcYOffset,
    const int bitDepth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    using Type4 = typename KfmVec4Traits<Type>::Type4;
    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const Type4 *s = (const Type4 *)src;
    const int yy = y + srcYOffset;
    const int4 combe = kfm_static_calc_combe4<Type>(
        s[x + (yy - 2) * srcPitch4],
        s[x + (yy - 1) * srcPitch4],
        s[x + yy * srcPitch4],
        s[x + (yy + 1) * srcPitch4],
        s[x + (yy + 2) * srcPitch4]);
    ((Type4 *)dst)[x + y * dstPitch4] = KfmVec4Traits<Type>::to_type4(kfm_i4_clamp(kfm_i4_shr(combe, 2), 0, (1 << bitDepth) - 1));
}

template<typename Type>
__global__ void kernel_kfm_temporal_min_diff5_3(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src0,
    const uint8_t *src1,
    const uint8_t *src2,
    const uint8_t *src3,
    const uint8_t *src4,
    const uint8_t *src5,
    const uint8_t *src6,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    using Type4 = typename KfmVec4Traits<Type>::Type4;
    const int off = y * (srcPitch / (int)sizeof(Type4)) + x;
    const int4 v0 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src0)[off]);
    const int4 v1 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src1)[off]);
    const int4 v2 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src2)[off]);
    const int4 v3 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src3)[off]);
    const int4 v4 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src4)[off]);
    const int4 v5 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src5)[off]);
    const int4 v6 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src6)[off]);

    const int4 min0 = kfm_i4_min(kfm_i4_min(v0, v1), kfm_i4_min(v2, kfm_i4_min(v3, v4)));
    const int4 max0 = kfm_i4_max(kfm_i4_max(v0, v1), kfm_i4_max(v2, kfm_i4_max(v3, v4)));
    const int4 diff0 = kfm_i4_sub(max0, min0);

    const int4 min1 = kfm_i4_min(kfm_i4_min(v1, v2), kfm_i4_min(v3, kfm_i4_min(v4, v5)));
    const int4 max1 = kfm_i4_max(kfm_i4_max(v1, v2), kfm_i4_max(v3, kfm_i4_max(v4, v5)));
    const int4 diff1 = kfm_i4_sub(max1, min1);

    const int4 min2 = kfm_i4_min(kfm_i4_min(v2, v3), kfm_i4_min(v4, kfm_i4_min(v5, v6)));
    const int4 max2 = kfm_i4_max(kfm_i4_max(v2, v3), kfm_i4_max(v4, kfm_i4_max(v5, v6)));
    const int4 diff2 = kfm_i4_sub(max2, min2);

    ((Type4 *)dst)[y * (dstPitch / (int)sizeof(Type4)) + x] = KfmVec4Traits<Type>::to_type4(kfm_i4_min(diff0, kfm_i4_min(diff1, diff2)));
}

template<typename Type>
__global__ void kernel_kfm_merge_uv_coefs(
    uint8_t *flagY,
    const int pitchY,
    const uint8_t *flagU,
    const uint8_t *flagV,
    const int pitchUV,
    const int width,
    const int height,
    const int logUVx,
    const int logUVy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    Type *fy = (Type *)(flagY + y * pitchY + x * (int)sizeof(Type));
    const Type *fu = (const Type *)(flagU + ((y >> logUVy) * pitchUV + (x >> logUVx) * (int)sizeof(Type)));
    const Type *fv = (const Type *)(flagV + ((y >> logUVy) * pitchUV + (x >> logUVx) * (int)sizeof(Type)));
    fy[0] = max(fy[0], max(fu[0], fv[0]));
}

template<typename Type>
__global__ void kernel_kfm_extend_coefs(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    using Type4 = typename KfmVec4Traits<Type>::Type4;
    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int y0 = max(y - 1, 0);
    const int y1 = min(y + 1, height - 1);
    const Type4 *s = (const Type4 *)src;
    const int4 v = kfm_i4_max(KfmVec4Traits<Type>::to_int4(s[x + y0 * srcPitch4]),
        kfm_i4_max(KfmVec4Traits<Type>::to_int4(s[x + y * srcPitch4]), KfmVec4Traits<Type>::to_int4(s[x + y1 * srcPitch4])));
    ((Type4 *)dst)[x + y * dstPitch4] = KfmVec4Traits<Type>::to_type4(v);
}

template<typename Type>
__global__ void kernel_kfm_and_coefs(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *diff,
    const int diffPitch,
    const int width4,
    const int height,
    const float invcombe,
    const float invdiff) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    using Type4 = typename KfmVec4Traits<Type>::Type4;
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int diffPitch4 = diffPitch / (int)sizeof(Type4);
    const int4 combeI = KfmVec4Traits<Type>::to_int4(((Type4 *)dst)[x + y * dstPitch4]);
    const int4 diffI = KfmVec4Traits<Type>::to_int4(((const Type4 *)diff)[x + y * diffPitch4]);
    const float4 outv = make_float4(
        max(clamp((float)combeI.x * invcombe - 1.0f, -0.5f, 0.5f) + clamp((float)diffI.x * (-invdiff) + 1.0f, -0.5f, 0.5f), 0.0f) * 128.0f + 0.5f,
        max(clamp((float)combeI.y * invcombe - 1.0f, -0.5f, 0.5f) + clamp((float)diffI.y * (-invdiff) + 1.0f, -0.5f, 0.5f), 0.0f) * 128.0f + 0.5f,
        max(clamp((float)combeI.z * invcombe - 1.0f, -0.5f, 0.5f) + clamp((float)diffI.z * (-invdiff) + 1.0f, -0.5f, 0.5f), 0.0f) * 128.0f + 0.5f,
        max(clamp((float)combeI.w * invcombe - 1.0f, -0.5f, 0.5f) + clamp((float)diffI.w * (-invdiff) + 1.0f, -0.5f, 0.5f), 0.0f) * 128.0f + 0.5f);
    ((Type4 *)dst)[x + y * dstPitch4] = KfmVec4Traits<Type>::to_type4_float(outv);
}

template<typename Type>
__global__ void kernel_kfm_apply_uv_coefs_420(
    const uint8_t *flagY,
    const int pitchY,
    uint8_t *flagU,
    uint8_t *flagV,
    const int pitchUV,
    const int widthUV,
    const int heightUV) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthUV || y >= heightUV) return;

    const Type *fy = (const Type *)flagY;
    const int pitchYt = pitchY / (int)sizeof(Type);
    const int pitchUVt = pitchUV / (int)sizeof(Type);
    const int v = fy[(x * 2 + 0) + (y * 2 + 0) * pitchYt]
        + fy[(x * 2 + 1) + (y * 2 + 0) * pitchYt]
        + fy[(x * 2 + 0) + (y * 2 + 1) * pitchYt]
        + fy[(x * 2 + 1) + (y * 2 + 1) * pitchYt];
    const Type outv = (Type)((v + 2) >> 2);
    ((Type *)flagU)[x + y * pitchUVt] = outv;
    ((Type *)flagV)[x + y * pitchUVt] = outv;
}

template<typename Type>
__global__ void kernel_kfm_merge_static(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src60,
    const uint8_t *src30,
    const int srcPitch,
    const uint8_t *flag,
    const int flagPitch,
    const int width4,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width4 || y >= height) return;

    using Type4 = typename KfmVec4Traits<Type>::Type4;
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int flagPitch4 = flagPitch / (int)sizeof(Type4);
    const int4 coef = KfmVec4Traits<Type>::to_int4(((const Type4 *)flag)[x + y * flagPitch4]);
    const int4 v30 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src30)[x + y * srcPitch4]);
    const int4 v60 = KfmVec4Traits<Type>::to_int4(((const Type4 *)src60)[x + y * srcPitch4]);
    const int4 outv = make_int4(
        (coef.x * v30.x + (128 - coef.x) * v60.x + 64) >> 7,
        (coef.y * v30.y + (128 - coef.y) * v60.y + 64) >> 7,
        (coef.z * v30.z + (128 - coef.z) * v60.z + 64) >> 7,
        (coef.w * v30.w + (128 - coef.w) * v60.w + 64) >> 7);
    ((Type4 *)dst)[x + y * dstPitch4] = KfmVec4Traits<Type>::to_type4(outv);
}

template<typename Func8, typename Func16>
static RGY_ERR dispatch_kfm_depth(const RGYFrameInfo *frame, Func8 func8, Func16 func16) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    if (RGY_CSP_BIT_DEPTH[frame->csp] > 8) {
        return func16();
    }
    return func8();
}

RGY_ERR run_kfm_zero_plane(RGYFrameInfo *pOutputFrame, cudaStream_t stream) {
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_zero<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_zero<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_static_calc_combe_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int srcYOffset, cudaStream_t stream) {
    if (!pOutputFrame || !pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int width4 = pOutputFrame->width >> 2;
    if (width4 <= 0 || (pOutputFrame->width & 3) != 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_calc_combe<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], width4, pOutputFrame->height, srcYOffset, RGY_CSP_BIT_DEPTH[pOutputFrame->csp]);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_calc_combe<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], width4, pOutputFrame->height, srcYOffset, RGY_CSP_BIT_DEPTH[pOutputFrame->csp]);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_temporal_min_diff5_3_plane(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *src0,
    const RGYFrameInfo *src1,
    const RGYFrameInfo *src2,
    const RGYFrameInfo *src3,
    const RGYFrameInfo *src4,
    const RGYFrameInfo *src5,
    const RGYFrameInfo *src6,
    cudaStream_t stream) {
    if (!src0 || !src1 || !src2 || !src3 || !src4 || !src5 || !src6
        || !src0->ptr[0] || !src1->ptr[0] || !src2->ptr[0] || !src3->ptr[0] || !src4->ptr[0] || !src5->ptr[0] || !src6->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int width4 = pOutputFrame->width >> 2;
    if (width4 <= 0 || (pOutputFrame->width & 3) != 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_temporal_min_diff5_3<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], (const uint8_t *)src2->ptr[0],
                (const uint8_t *)src3->ptr[0], (const uint8_t *)src4->ptr[0], (const uint8_t *)src5->ptr[0], (const uint8_t *)src6->ptr[0],
                src0->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_temporal_min_diff5_3<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], (const uint8_t *)src2->ptr[0],
                (const uint8_t *)src3->ptr[0], (const uint8_t *)src4->ptr[0], (const uint8_t *)src5->ptr[0], (const uint8_t *)src6->ptr[0],
                src0->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_merge_uv_coefs_plane(RGYFrameInfo *flagY, const RGYFrameInfo *flagU, const RGYFrameInfo *flagV, int logUVx, int logUVy, cudaStream_t stream) {
    if (!flagU || !flagV || !flagU->ptr[0] || !flagV->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(flagY,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(flagY->width, (int)block.x), divCeil(flagY->height, (int)block.y));
            kernel_kfm_merge_uv_coefs<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)flagY->ptr[0], flagY->pitch[0],
                (const uint8_t *)flagU->ptr[0], (const uint8_t *)flagV->ptr[0], flagU->pitch[0], flagY->width, flagY->height, logUVx, logUVy);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(flagY->width, (int)block.x), divCeil(flagY->height, (int)block.y));
            kernel_kfm_merge_uv_coefs<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)flagY->ptr[0], flagY->pitch[0],
                (const uint8_t *)flagU->ptr[0], (const uint8_t *)flagV->ptr[0], flagU->pitch[0], flagY->width, flagY->height, logUVx, logUVy);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_extend_coefs_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int width4 = pOutputFrame->width >> 2;
    if (width4 <= 0 || (pOutputFrame->width & 3) != 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_extend_coefs<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_extend_coefs<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pInputFrame->ptr[0], pInputFrame->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_and_coefs_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDiffFrame, float invcombe, float invdiff, cudaStream_t stream) {
    if (!pDiffFrame || !pDiffFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int width4 = pOutputFrame->width >> 2;
    if (width4 <= 0 || (pOutputFrame->width & 3) != 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_and_coefs<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pDiffFrame->ptr[0], pDiffFrame->pitch[0], width4, pOutputFrame->height, invcombe, invdiff);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_and_coefs<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pDiffFrame->ptr[0], pDiffFrame->pitch[0], width4, pOutputFrame->height, invcombe, invdiff);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_apply_uv_coefs_420_plane(RGYFrameInfo *flagU, RGYFrameInfo *flagV, const RGYFrameInfo *flagY, cudaStream_t stream) {
    if (!flagU || !flagV || !flagY || !flagU->ptr[0] || !flagV->ptr[0] || !flagY->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(flagU,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(flagU->width, (int)block.x), divCeil(flagU->height, (int)block.y));
            kernel_kfm_apply_uv_coefs_420<uint8_t><<<grid, block, 0, stream>>>((const uint8_t *)flagY->ptr[0], flagY->pitch[0],
                (uint8_t *)flagU->ptr[0], (uint8_t *)flagV->ptr[0], flagU->pitch[0], flagU->width, flagU->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(flagU->width, (int)block.x), divCeil(flagU->height, (int)block.y));
            kernel_kfm_apply_uv_coefs_420<uint16_t><<<grid, block, 0, stream>>>((const uint8_t *)flagY->ptr[0], flagY->pitch[0],
                (uint8_t *)flagU->ptr[0], (uint8_t *)flagV->ptr[0], flagU->pitch[0], flagU->width, flagU->height);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_merge_static_plane(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pFlagFrame, cudaStream_t stream) {
    if (!pDeint60Frame || !pSourceFrame || !pFlagFrame || !pDeint60Frame->ptr[0] || !pSourceFrame->ptr[0] || !pFlagFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int width4 = pOutputFrame->width >> 2;
    if (width4 <= 0 || (pOutputFrame->width & 3) != 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_merge_static<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pDeint60Frame->ptr[0], (const uint8_t *)pSourceFrame->ptr[0], pSourceFrame->pitch[0],
                (const uint8_t *)pFlagFrame->ptr[0], pFlagFrame->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width4, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_merge_static<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pDeint60Frame->ptr[0], (const uint8_t *)pSourceFrame->ptr[0], pSourceFrame->pitch[0],
                (const uint8_t *)pFlagFrame->ptr[0], pFlagFrame->pitch[0], width4, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        });
}
