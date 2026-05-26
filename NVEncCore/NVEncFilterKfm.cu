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

template<typename Type>
__device__ Type kfm_load_pixel(
    const uint8_t *src,
    const int pitch,
    const int x,
    const int y) {
    return ((const Type *)(src + y * pitch))[x];
}

template<typename Type>
__device__ void kfm_store_pixel(
    uint8_t *dst,
    const int pitch,
    const int x,
    const int y,
    const Type v) {
    ((Type *)(dst + y * pitch))[x] = v;
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

__device__ int kfm_div_floor_pow2_cuda(const int v, const int shift) {
    const int div = 1 << shift;
    const int q = v / div;
    return q - ((v < 0 && q * div != v) ? 1 : 0);
}

__device__ int kfm_temporal_avg3_cuda(const int a, const int b, const int c) {
    return (a + b + c) / 3;
}

__device__ uint8_t kfm_load_u8_max_extend_rb_cuda(const uint8_t *src, const int pitch, const int width, const int height, const int x, const int y) {
    const int hx0 = (x == 0 && width > 1) ? 1 : x;
    const int hx1 = (x > 0 && x < width - 1) ? x + 1 : hx0;
    const int hy0 = (y == 0 && height > 1) ? 1 : y;
    const int hy1 = (y > 0 && y < height - 1) ? y + 1 : hy0;
    const uint8_t v00 = src[hy0 * pitch + hx0];
    const uint8_t v10 = src[hy0 * pitch + hx1];
    const uint8_t v01 = src[hy1 * pitch + hx0];
    const uint8_t v11 = src[hy1 * pitch + hx1];
    return max(max(v00, v10), max(v01, v11));
}

template<typename Type>
__device__ Type kfm_to_type_sat(const int v);

template<>
__device__ uint8_t kfm_to_type_sat<uint8_t>(const int v) {
    return (uint8_t)clamp(v, 0, 255);
}

template<>
__device__ uint16_t kfm_to_type_sat<uint16_t>(const int v) {
    return (uint16_t)clamp(v, 0, 65535);
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

template<typename Type>
__device__ uchar2 kfm_analyze_super_pair_render_cuda(
    const uint8_t *src0,
    const uint8_t *src1,
    const int srcPitch,
    const int bitDepth,
    const int widthPairs,
    const int height,
    const int parity,
    const int pixelStep,
    const int pixelOffset,
    const int x,
    const int row) {
    if (x <= 0 || x >= widthPairs || row < 2 || row >= height * 2) {
        return make_uchar2(0, 0);
    }
    const int bx = x - 1;
    const int by = (row >> 1) - 1;
    if (bx >= widthPairs - 1 || by < 0 || by >= height - 1) {
        return make_uchar2(0, 0);
    }
    const uchar4 v = kfm_analyze_block<Type>(src0, src1, srcPitch, bitDepth, parity, pixelStep, pixelOffset, bx, by);
    return (row & 1) ? make_uchar2(v.z, v.w) : make_uchar2(v.x, v.y);
}

template<typename Type>
__device__ Type kfm_telecine_weave_pixel_cuda(
    const uint8_t *src0,
    const int src0Pitch,
    const uint8_t *src1,
    const int src1Pitch,
    const uint8_t *src2,
    const int src2Pitch,
    const int x,
    const int y,
    const int srcYOffset,
    const int fieldStart,
    const int fieldCount,
    const int parity) {
    const int srcOutY = y + srcYOffset;
    const int outField = ((srcOutY & 1) == (parity & 1)) ? 1 : 0;
    const int fieldBase = fieldStart & ~1;
    const int fieldEnd = fieldStart + fieldCount;
    Type sum = (Type)0;
    int count = 0;
    for (int field = fieldStart; field < fieldEnd; field++) {
        if ((field & 1) != outField) {
            continue;
        }
        const int frameOffset = (field - fieldBase) >> 1;
        const int srcY = (field & 1) + ((srcOutY >> 1) << 1);
        Type v = (Type)0;
        if (frameOffset == 0) {
            v = kfm_load_pixel<Type>(src0, src0Pitch, x, srcY);
        } else if (frameOffset == 1) {
            v = kfm_load_pixel<Type>(src1, src1Pitch, x, srcY);
        } else {
            v = kfm_load_pixel<Type>(src2, src2Pitch, x, srcY);
        }
        if (count == 0) {
            sum = v;
        } else {
            sum = (Type)(((int)sum + (int)v) >> 1);
        }
        count++;
    }
    return sum;
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

template<typename Type>
__global__ void kernel_kfm_telecine_weave(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src0,
    const int src0Pitch,
    const uint8_t *src1,
    const int src1Pitch,
    const uint8_t *src2,
    const int src2Pitch,
    const int width,
    const int height,
    const int srcYOffset,
    const int fieldStart,
    const int fieldCount,
    const int parity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const Type v = kfm_telecine_weave_pixel_cuda<Type>(
        src0, src0Pitch, src1, src1Pitch, src2, src2Pitch,
        x, y, srcYOffset, fieldStart, fieldCount, parity);
    kfm_store_pixel<Type>(dst, dstPitch, x, y, v);
}

template<typename Type>
__global__ void kernel_kfm_analyze(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src0,
    const uint8_t *src1,
    const int srcPitch,
    const int bitDepth,
    const int width,
    const int height,
    const int parity,
    const int pixelStep,
    const int pixelOffset) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx >= width - 1 || by >= height - 1) return;

    const int dstPitchT = dstPitch / (int)sizeof(uchar2);
    uchar2 *flag = (uchar2 *)dst;
    const uchar4 v = kfm_analyze_block<Type>(src0, src1, srcPitch, bitDepth, parity, pixelStep, pixelOffset, bx, by);
    const int dstX = bx + 1;
    const int dstY = (by + 1) * 2;
    flag[dstX + (dstY + 0) * dstPitchT] = make_uchar2(v.x, v.y);
    flag[dstX + (dstY + 1) * dstPitchT] = make_uchar2(v.z, v.w);
}

template<typename Type>
__global__ void kernel_kfm_clean_super_direct_max(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *prevSrc0,
    const uint8_t *prevSrc1,
    const int prevSrcPitch,
    const int prevParity,
    const uint8_t *curSrc0,
    const uint8_t *curSrc1,
    const int curSrcPitch,
    const int curParity,
    const int bitDepth,
    const int widthPairs,
    const int height,
    const int field,
    const int cleanThresh,
    const int maxMode,
    const int dstStep,
    const int dstOffset,
    const int pixelStep,
    const int pixelOffset) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthPairs || y >= height) return;

    const int srcField = field & 1;
    const int curRow = y * 2 + srcField;
    const int prevRow = (srcField == 0) ? (y * 2 + 1) : (y * 2);
    const uchar2 vcur = kfm_analyze_super_pair_render_cuda<Type>(
        curSrc0, curSrc1, curSrcPitch, bitDepth, widthPairs, height, curParity,
        pixelStep, pixelOffset, x, curRow);
    const uchar2 vprev = (srcField == 0)
        ? kfm_analyze_super_pair_render_cuda<Type>(prevSrc0, prevSrc1, prevSrcPitch, bitDepth, widthPairs, height, prevParity, pixelStep, pixelOffset, x, prevRow)
        : kfm_analyze_super_pair_render_cuda<Type>(curSrc0, curSrc1, curSrcPitch, bitDepth, widthPairs, height, curParity, pixelStep, pixelOffset, x, prevRow);

    uchar2 v = vcur;
    if (vprev.y <= cleanThresh && v.y <= cleanThresh) {
        v.x = 0;
    }
    uint8_t *p0 = dst + y * dstPitch + (x * 2 + 0) * dstStep + dstOffset;
    uint8_t *p1 = dst + y * dstPitch + (x * 2 + 1) * dstStep + dstOffset;
    if (maxMode) {
        p0[0] = max(p0[0], v.x);
        p1[0] = max(p1[0], v.y);
    } else {
        p0[0] = v.x;
        p1[0] = v.y;
    }
}

__global__ void kernel_kfm_clean_separated_super_max(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *prevSuper,
    const uint8_t *curSuper,
    const int superPitch,
    const int widthPairs,
    const int height,
    const int field,
    const int cleanThresh,
    const int maxMode,
    const int dstStep,
    const int dstOffset) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= widthPairs || y >= height) return;

    const int pitchT = superPitch / (int)sizeof(uchar2);
    const int srcField = field & 1;
    const int curRow = y * 2 + srcField;
    const int prevRow = (srcField == 0) ? (y * 2 + 1) : (y * 2);
    const uchar2 *prev = (const uchar2 *)((srcField == 0) ? prevSuper : curSuper);
    const uchar2 *cur = (const uchar2 *)curSuper;

    uchar2 v = cur[x + curRow * pitchT];
    const uchar2 pv = prev[x + prevRow * pitchT];
    if (pv.y <= cleanThresh && v.y <= cleanThresh) {
        v.x = 0;
    }

    uint8_t *p0 = dst + y * dstPitch + (x * 2 + 0) * dstStep + dstOffset;
    uint8_t *p1 = dst + y * dstPitch + (x * 2 + 1) * dstStep + dstOffset;
    if (maxMode) {
        p0[0] = max(p0[0], v.x);
        p1[0] = max(p1[0], v.y);
    } else {
        p0[0] = v.x;
        p1[0] = v.y;
    }
}

template<typename Type>
__global__ void kernel_kfm_remove_combe_binomial(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const uint8_t *combe,
    const int combePitch,
    const uint8_t *teleSrc0,
    const int teleSrc0Pitch,
    const uint8_t *teleSrc1,
    const int teleSrc1Pitch,
    const uint8_t *teleSrc2,
    const int teleSrc2Pitch,
    const int width,
    const int height,
    const int threshold,
    const int srcStep,
    const int srcOffset,
    const int combeStep,
    const int combeOffset,
    const int teleSrcYOffset,
    const int teleFieldStart,
    const int teleFieldCount,
    const int teleParity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int sx = x * srcStep + srcOffset;
    const int cx = (x >> 2) * 2 * combeStep + combeOffset;
    const int cy = y >> 2;
    const int score = (int)combe[cy * combePitch + cx];
    Type v = kfm_load_pixel<Type>(src, srcPitch, sx, y);
    if (score >= threshold) {
        const int prevY = max(y - 1, 0);
        const int nextY = min(y + 1, height - 1);
        const int prev = (int)((y > 0)
            ? kfm_load_pixel<Type>(src, srcPitch, sx, prevY)
            : kfm_telecine_weave_pixel_cuda<Type>(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y - 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity));
        const int cur = (int)v;
        const int next = (int)((y + 1 < height)
            ? kfm_load_pixel<Type>(src, srcPitch, sx, nextY)
            : kfm_telecine_weave_pixel_cuda<Type>(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y + 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity));
        v = (Type)((prev + 2 * cur + next + 2) >> 2);
    }
    kfm_store_pixel<Type>(dst, dstPitch, sx, y, v);
}

template<typename Type>
__global__ void kernel_kfm_patch_combe(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *base,
    const int basePitch,
    const uint8_t *patch,
    const int patchPitch,
    const uint8_t *mask,
    const int maskPitch,
    const int width,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const Type m = kfm_load_pixel<Type>(mask, maskPitch, x, y);
    const Type b = kfm_load_pixel<Type>(base, basePitch, x, y);
    const Type p = kfm_load_pixel<Type>(patch, patchPitch, x, y);
    const int coef = clamp((int)m, 0, 128);
    const int invcoef = 128 - coef;
    const Type v = (Type)((coef * (int)p + invcoef * (int)b + 64) >> 7);
    kfm_store_pixel<Type>(dst, dstPitch, x, y, v);
}

__global__ void kernel_kfm_switch_flag_combe_min(
    uint8_t *dstY,
    const int dstYPitch,
    uint8_t *dstC,
    const int dstCPitch,
    const uint8_t *superPrevY,
    const uint8_t *superY,
    const uint8_t *superNextY,
    const int superYPitch,
    const uint8_t *superPrevUV,
    const uint8_t *superUV,
    const uint8_t *superNextUV,
    const int superUVPitch,
    const uint8_t *superPrevV,
    const uint8_t *superV,
    const uint8_t *superNextV,
    const int superVPitch,
    const int combeWidth,
    const int combeHeight,
    const int combeCWidth,
    const int combeCHeight,
    const int hasUV,
    const int interleavedUV) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < combeWidth && y < combeHeight) {
        const int sx = x << 1;
        dstY[y * dstYPitch + x] = (uint8_t)kfm_temporal_avg3_cuda(
            (int)superPrevY[y * superYPitch + sx],
            (int)superY[y * superYPitch + sx],
            (int)superNextY[y * superYPitch + sx]);
    }
    if (x < combeCWidth && y < combeCHeight) {
        int cmax = 0;
        if (hasUV) {
            const int ux = interleavedUV ? (x << 2) : (x << 1);
            const int uy = y;
            const int u = kfm_temporal_avg3_cuda(
                (int)superPrevUV[uy * superUVPitch + ux],
                (int)superUV[uy * superUVPitch + ux],
                (int)superNextUV[uy * superUVPitch + ux]);
            const int v = kfm_temporal_avg3_cuda(
                (int)superPrevV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)],
                (int)superV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)],
                (int)superNextV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)]);
            cmax = max(u, v);
        }
        dstC[y * dstCPitch + x] = (uint8_t)cmax;
    }
}

__global__ void kernel_kfm_switch_flag_from_combe_min(
    uint8_t *dstY,
    const int dstYPitch,
    uint8_t *dstC,
    const int dstCPitch,
    const uint8_t *combeY,
    const int combeYPitch,
    const uint8_t *combeC,
    const int combeCPitch,
    const int flagWidth,
    const int flagHeight,
    const int innerWidth,
    const int innerHeight,
    const int combeWidth,
    const int combeHeight,
    const int combeCWidth,
    const int combeCHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= flagWidth || y >= flagHeight) return;

    const int ix = x - 4;
    const int iy = y - 2;
    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2 || ix == 0 || iy == 0) {
        dstY[y * dstYPitch + x] = 0;
        dstC[y * dstCPitch + x] = 0;
        return;
    }
    const int cx0 = (ix << 1) - 1;
    const int cx1 = ix << 1;
    const int cy0 = (iy << 1) - 1;
    const int cy1 = iy << 1;
    const int ysum =
        ((cx0 < 0 || cy0 < 0 || cx0 >= combeWidth || cy0 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb_cuda(combeY, combeYPitch, combeWidth, combeHeight, cx0, cy0))
      + ((cx1 < 0 || cy0 < 0 || cx1 >= combeWidth || cy0 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb_cuda(combeY, combeYPitch, combeWidth, combeHeight, cx1, cy0))
      + ((cx0 < 0 || cy1 < 0 || cx0 >= combeWidth || cy1 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb_cuda(combeY, combeYPitch, combeWidth, combeHeight, cx0, cy1))
      + ((cx1 < 0 || cy1 < 0 || cx1 >= combeWidth || cy1 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb_cuda(combeY, combeYPitch, combeWidth, combeHeight, cx1, cy1));
    dstY[y * dstYPitch + x] = (uint8_t)((ysum + 2) >> 2);
    dstC[y * dstCPitch + x] = (ix < combeCWidth && iy < combeCHeight) ? combeC[iy * combeCPitch + ix] : 0;
}

__global__ void kernel_kfm_switch_flag_box3x3_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = 0;
        return;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        const int sy = y + dy;
        for (int dx = -1; dx <= 1; dx++) {
            const int sx = x + dx;
            sum += (sx < 0 || sy < 0 || sx >= width || sy >= height) ? 0 : (int)src[sy * srcPitch + sx];
        }
    }
    dst[y * dstPitch + x] = (uint8_t)min(sum >> 2, 255);
}

__global__ void kernel_kfm_switch_flag_binary_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *srcY,
    const int srcYPitch,
    const uint8_t *srcC,
    const int srcCPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight,
    const int thY,
    const int thC) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = 0;
        return;
    }
    const int yv = (int)srcY[y * srcYPitch + x];
    const int cv = (int)srcC[y * srcCPitch + x];
    dst[y * dstPitch + x] = (yv >= thY || cv >= thC) ? 128 : 0;
}

__global__ void kernel_kfm_switch_flag_binary_extend_hv_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *srcY,
    const int srcYPitch,
    const uint8_t *srcC,
    const int srcCPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight,
    const int thY,
    const int thC) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = 0;
        return;
    }
    const int ix = x - 4;
    const int iy = y - 2;
    const int hx0 = (ix == 0 && innerWidth > 1) ? 1 : ix;
    const int hx1 = (ix > 0 && ix < innerWidth - 1) ? ix + 1 : hx0;
    const int hy0 = (iy == 0 && innerHeight > 1) ? 1 : iy;
    const int hy1 = (iy > 0 && iy < innerHeight - 1) ? iy + 1 : hy0;
    const int px0 = hx0 + 4;
    const int px1 = hx1 + 4;
    const int py0 = hy0 + 2;
    const int py1 = hy1 + 2;
    const int ymax = max(max((int)srcY[py0 * srcYPitch + px0], (int)srcY[py0 * srcYPitch + px1]),
        max((int)srcY[py1 * srcYPitch + px0], (int)srcY[py1 * srcYPitch + px1]));
    const int cmax = max(max((int)srcC[py0 * srcCPitch + px0], (int)srcC[py0 * srcCPitch + px1]),
        max((int)srcC[py1 * srcCPitch + px0], (int)srcC[py1 * srcCPitch + px1]));
    dst[y * dstPitch + x] = (ymax >= thY || cmax >= thC) ? 128 : 0;
}

__global__ void kernel_kfm_switch_flag_extend_h_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int offsetX,
    const int offsetY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int px = x + offsetX;
    const int py = y + offsetY;
    uint8_t v;
    if (x == width - 1) {
        v = src[py * srcPitch + px];
    } else if (x == 0) {
        v = src[py * srcPitch + px + 1];
    } else {
        v = max(src[py * srcPitch + px], src[py * srcPitch + px + 1]);
    }
    dst[py * dstPitch + px] = v;
}

__global__ void kernel_kfm_switch_flag_extend_v_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int offsetX,
    const int offsetY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int px = x + offsetX;
    const int py = y + offsetY;
    uint8_t v;
    if (y == height - 1) {
        v = src[py * srcPitch + px];
    } else if (y == 0) {
        v = src[(py + 1) * srcPitch + px];
    } else {
        v = max(src[py * srcPitch + px], src[(py + 1) * srcPitch + px]);
    }
    dst[py * dstPitch + px] = v;
}

__global__ void kernel_kfm_contains_combe_init(uint32_t *count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        count[0] = 0;
    }
}

__global__ void kernel_kfm_contains_combe_count(
    const uint8_t *mask,
    const int maskPitch,
    uint32_t *count,
    const int width,
    const int height,
    const int threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (y >= 6 && y < height - 6 && (int)mask[y * maskPitch + x] >= threshold) {
        atomicAdd(count, 1u);
    }
}

__global__ void kernel_kfm_contains_combe_mark(
    uint8_t *dst,
    const int dstPitch,
    const uint32_t *count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 4 || y >= 1) return;
    dst[y * dstPitch + x] = (count[0] != 0) ? 255 : 0;
}

template<typename Type>
__global__ void kernel_kfm_combe_mask_resize_bilinear_min(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *flag,
    const int flagPitch,
    const int width,
    const int height,
    const int srcStep,
    const int srcOffset,
    const int scaleX,
    const int shiftX,
    const int scaleY,
    const int shiftY,
    const int innerWidth,
    const int innerHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int halfX = scaleX >> 1;
    const int halfY = scaleY >> 1;
    const int x0 = kfm_div_floor_pow2_cuda(x - halfX, shiftX);
    const int y0 = kfm_div_floor_pow2_cuda(y - halfY, shiftY);
    const int c0x = ((x0 + 1) << shiftX) - (x - halfX);
    const int c1x = scaleX - c0x;
    const int c0y = ((y0 + 1) << shiftY) - (y - halfY);
    const int c1y = scaleY - c0y;
    const int fx0 = clamp(x0, 0, innerWidth - 1) + 4;
    const int fx1 = clamp(x0 + 1, 0, innerWidth - 1) + 4;
    const int fy0 = clamp(y0, 0, innerHeight - 1) + 2;
    const int fy1 = clamp(y0 + 1, 0, innerHeight - 1) + 2;
    const int h0 = ((int)flag[fy0 * flagPitch + fx0] * c0x + (int)flag[fy0 * flagPitch + fx1] * c1x + halfX) >> shiftX;
    const int h1 = ((int)flag[fy1 * flagPitch + fx0] * c0x + (int)flag[fy1 * flagPitch + fx1] * c1x + halfX) >> shiftX;
    const int v = (h0 * c0y + h1 * c1y + halfY) >> shiftY;
    kfm_store_pixel<Type>(dst, dstPitch, x * srcStep + srcOffset, y, kfm_to_type_sat<Type>(v));
}

__global__ void kernel_kfm_copy_u8_buffer_to_plane(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    dst[y * dstPitch + x] = src[y * srcPitch + x];
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

RGY_ERR run_kfm_telecine_weave_plane(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *src0,
    const RGYFrameInfo *src1,
    const RGYFrameInfo *src2,
    int srcYOffset,
    int fieldStart,
    int fieldCount,
    int parity,
    cudaStream_t stream) {
    if (!pOutputFrame || !src0 || !src1 || !src2 || !pOutputFrame->ptr[0] || !src0->ptr[0] || !src1->ptr[0] || !src2->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_telecine_weave<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)src0->ptr[0], src0->pitch[0],
                (const uint8_t *)src1->ptr[0], src1->pitch[0],
                (const uint8_t *)src2->ptr[0], src2->pitch[0],
                pOutputFrame->width, pOutputFrame->height, srcYOffset, fieldStart, fieldCount, parity);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_telecine_weave<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)src0->ptr[0], src0->pitch[0],
                (const uint8_t *)src1->ptr[0], src1->pitch[0],
                (const uint8_t *)src2->ptr[0], src2->pitch[0],
                pOutputFrame->width, pOutputFrame->height, srcYOffset, fieldStart, fieldCount, parity);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_analyze_plane(uint8_t *dst, int dstPitch,
    const RGYFrameInfo *src0, const RGYFrameInfo *src1,
    int width, int height, int parity, int pixelStep, int pixelOffset, cudaStream_t stream) {
    if (!dst || !src0 || !src1 || !src0->ptr[0] || !src1->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(src0,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
            kernel_kfm_analyze<uint8_t><<<grid, block, 0, stream>>>(dst, dstPitch,
                (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], src0->pitch[0],
                8, width, height, parity, pixelStep, pixelOffset);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const int bitDepth = RGY_CSP_BIT_DEPTH[src0->csp];
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
            kernel_kfm_analyze<uint16_t><<<grid, block, 0, stream>>>(dst, dstPitch,
                (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], src0->pitch[0],
                bitDepth, width, height, parity, pixelStep, pixelOffset);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_clean_super_direct_max_plane(RGYFrameInfo *dst,
    const RGYFrameInfo *prevSrc0, const RGYFrameInfo *prevSrc1, int prevParity,
    const RGYFrameInfo *curSrc0, const RGYFrameInfo *curSrc1, int curParity,
    int widthPairs, int height, int field, int cleanThresh, int maxMode,
    int dstStep, int dstOffset, int pixelStep, int pixelOffset, cudaStream_t stream) {
    if (!dst || !prevSrc0 || !prevSrc1 || !curSrc0 || !curSrc1
        || !dst->ptr[0] || !prevSrc0->ptr[0] || !prevSrc1->ptr[0] || !curSrc0->ptr[0] || !curSrc1->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(prevSrc0,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(widthPairs, (int)block.x), divCeil(height, (int)block.y));
            kernel_kfm_clean_super_direct_max<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
                (const uint8_t *)prevSrc0->ptr[0], (const uint8_t *)prevSrc1->ptr[0], prevSrc0->pitch[0], prevParity,
                (const uint8_t *)curSrc0->ptr[0], (const uint8_t *)curSrc1->ptr[0], curSrc0->pitch[0], curParity,
                8, widthPairs, height, field, cleanThresh, maxMode, dstStep, dstOffset, pixelStep, pixelOffset);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const int bitDepth = RGY_CSP_BIT_DEPTH[prevSrc0->csp];
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(widthPairs, (int)block.x), divCeil(height, (int)block.y));
            kernel_kfm_clean_super_direct_max<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
                (const uint8_t *)prevSrc0->ptr[0], (const uint8_t *)prevSrc1->ptr[0], prevSrc0->pitch[0], prevParity,
                (const uint8_t *)curSrc0->ptr[0], (const uint8_t *)curSrc1->ptr[0], curSrc0->pitch[0], curParity,
                bitDepth, widthPairs, height, field, cleanThresh, maxMode, dstStep, dstOffset, pixelStep, pixelOffset);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_clean_separated_super_max_plane(RGYFrameInfo *dst,
    const uint8_t *prevSuper, const uint8_t *curSuper, int superPitch,
    int widthPairs, int height, int field, int cleanThresh, int maxMode,
    int dstStep, int dstOffset, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !prevSuper || !curSuper) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(widthPairs, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_clean_separated_super_max<<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
        prevSuper, curSuper, superPitch, widthPairs, height, field, cleanThresh, maxMode, dstStep, dstOffset);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_remove_combe_binomial_plane(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pCombeFrame,
    const RGYFrameInfo *teleSrc0, const RGYFrameInfo *teleSrc1, const RGYFrameInfo *teleSrc2,
    int threshold, int srcStep, int srcOffset, int combeStep, int combeOffset,
    int teleSrcYOffset, int teleFieldStart, int teleFieldCount, int teleParity, cudaStream_t stream) {
    if (!pOutputFrame || !pSrcFrame || !pCombeFrame || !teleSrc0 || !teleSrc1 || !teleSrc2
        || !pOutputFrame->ptr[0] || !pSrcFrame->ptr[0] || !pCombeFrame->ptr[0] || !teleSrc0->ptr[0] || !teleSrc1->ptr[0] || !teleSrc2->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_remove_combe_binomial<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pSrcFrame->ptr[0], pSrcFrame->pitch[0],
                (const uint8_t *)pCombeFrame->ptr[0], pCombeFrame->pitch[0],
                (const uint8_t *)teleSrc0->ptr[0], teleSrc0->pitch[0],
                (const uint8_t *)teleSrc1->ptr[0], teleSrc1->pitch[0],
                (const uint8_t *)teleSrc2->ptr[0], teleSrc2->pitch[0],
                pOutputFrame->width, pOutputFrame->height, threshold,
                srcStep, srcOffset, combeStep, combeOffset, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_remove_combe_binomial<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pSrcFrame->ptr[0], pSrcFrame->pitch[0],
                (const uint8_t *)pCombeFrame->ptr[0], pCombeFrame->pitch[0],
                (const uint8_t *)teleSrc0->ptr[0], teleSrc0->pitch[0],
                (const uint8_t *)teleSrc1->ptr[0], teleSrc1->pitch[0],
                (const uint8_t *)teleSrc2->ptr[0], teleSrc2->pitch[0],
                pOutputFrame->width, pOutputFrame->height, threshold,
                srcStep, srcOffset, combeStep, combeOffset, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_patch_combe_plane(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame,
    int threshold, cudaStream_t stream) {
    UNREFERENCED_PARAMETER(threshold);
    if (!pOutputFrame || !pBaseFrame || !pPatchFrame || !pMaskFrame
        || !pOutputFrame->ptr[0] || !pBaseFrame->ptr[0] || !pPatchFrame->ptr[0] || !pMaskFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(pOutputFrame,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_patch_combe<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pBaseFrame->ptr[0], pBaseFrame->pitch[0],
                (const uint8_t *)pPatchFrame->ptr[0], pPatchFrame->pitch[0],
                (const uint8_t *)pMaskFrame->ptr[0], pMaskFrame->pitch[0],
                pOutputFrame->width, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(pOutputFrame->width, (int)block.x), divCeil(pOutputFrame->height, (int)block.y));
            kernel_kfm_patch_combe<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)pOutputFrame->ptr[0], pOutputFrame->pitch[0],
                (const uint8_t *)pBaseFrame->ptr[0], pBaseFrame->pitch[0],
                (const uint8_t *)pPatchFrame->ptr[0], pPatchFrame->pitch[0],
                (const uint8_t *)pMaskFrame->ptr[0], pMaskFrame->pitch[0],
                pOutputFrame->width, pOutputFrame->height);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_switch_flag_combe_min(uint8_t *dstY, int dstYPitch, uint8_t *dstC, int dstCPitch,
    const RGYFrameInfo *superPrevY, const RGYFrameInfo *superY, const RGYFrameInfo *superNextY,
    const RGYFrameInfo *superPrevUV, const RGYFrameInfo *superUV, const RGYFrameInfo *superNextUV,
    const RGYFrameInfo *superPrevV, const RGYFrameInfo *superV, const RGYFrameInfo *superNextV,
    int combeWidth, int combeHeight, int combeCWidth, int combeCHeight,
    int hasUV, int interleavedUV, cudaStream_t stream) {
    if (!dstY || !dstC || !superPrevY || !superY || !superNextY || !superPrevY->ptr[0] || !superY->ptr[0] || !superNextY->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(std::max(combeWidth, combeCWidth), (int)block.x), divCeil(std::max(combeHeight, combeCHeight), (int)block.y));
    kernel_kfm_switch_flag_combe_min<<<grid, block, 0, stream>>>(dstY, dstYPitch, dstC, dstCPitch,
        (const uint8_t *)superPrevY->ptr[0], (const uint8_t *)superY->ptr[0], (const uint8_t *)superNextY->ptr[0], superY->pitch[0],
        (const uint8_t *)(superPrevUV && superPrevUV->ptr[0] ? superPrevUV->ptr[0] : superPrevY->ptr[0]),
        (const uint8_t *)(superUV && superUV->ptr[0] ? superUV->ptr[0] : superY->ptr[0]),
        (const uint8_t *)(superNextUV && superNextUV->ptr[0] ? superNextUV->ptr[0] : superNextY->ptr[0]),
        superUV && superUV->ptr[0] ? superUV->pitch[0] : superY->pitch[0],
        (const uint8_t *)(superPrevV && superPrevV->ptr[0] ? superPrevV->ptr[0] : superPrevY->ptr[0]),
        (const uint8_t *)(superV && superV->ptr[0] ? superV->ptr[0] : superY->ptr[0]),
        (const uint8_t *)(superNextV && superNextV->ptr[0] ? superNextV->ptr[0] : superNextY->ptr[0]),
        superV && superV->ptr[0] ? superV->pitch[0] : (superUV && superUV->ptr[0] ? superUV->pitch[0] : superY->pitch[0]),
        combeWidth, combeHeight, combeCWidth, combeCHeight, hasUV, interleavedUV);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_from_combe_min(uint8_t *dstY, int dstYPitch, uint8_t *dstC, int dstCPitch,
    const uint8_t *combeY, int combeYPitch, const uint8_t *combeC, int combeCPitch,
    int flagWidth, int flagHeight, int innerWidth, int innerHeight,
    int combeWidth, int combeHeight, int combeCWidth, int combeCHeight, cudaStream_t stream) {
    if (!dstY || !dstC || !combeY || !combeC) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(flagWidth, (int)block.x), divCeil(flagHeight, (int)block.y));
    kernel_kfm_switch_flag_from_combe_min<<<grid, block, 0, stream>>>(dstY, dstYPitch, dstC, dstCPitch,
        combeY, combeYPitch, combeC, combeCPitch, flagWidth, flagHeight, innerWidth, innerHeight,
        combeWidth, combeHeight, combeCWidth, combeCHeight);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_box3x3_min(uint8_t *dst, int dstPitch, const uint8_t *src, int srcPitch,
    int width, int height, int innerWidth, int innerHeight, cudaStream_t stream) {
    if (!dst || !src) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_switch_flag_box3x3_min<<<grid, block, 0, stream>>>(dst, dstPitch, src, srcPitch, width, height, innerWidth, innerHeight);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_binary_extend_hv_min(RGYFrameInfo *dst,
    const uint8_t *srcY, int srcYPitch, const uint8_t *srcC, int srcCPitch,
    int innerWidth, int innerHeight, int thY, int thC, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !srcY || !srcC) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_switch_flag_binary_extend_hv_min<<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
        srcY, srcYPitch, srcC, srcCPitch, dst->width, dst->height, innerWidth, innerHeight, thY, thC);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_binary_min(RGYFrameInfo *dst,
    const uint8_t *srcY, int srcYPitch, const uint8_t *srcC, int srcCPitch,
    int innerWidth, int innerHeight, int thY, int thC, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !srcY || !srcC) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_switch_flag_binary_min<<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
        srcY, srcYPitch, srcC, srcCPitch, dst->width, dst->height, innerWidth, innerHeight, thY, thC);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_extend_h_min(uint8_t *dst, int dstPitch, const RGYFrameInfo *src,
    int width, int height, int offsetX, int offsetY, cudaStream_t stream) {
    if (!dst || !src || !src->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_switch_flag_extend_h_min<<<grid, block, 0, stream>>>(dst, dstPitch, (const uint8_t *)src->ptr[0], src->pitch[0],
        width, height, offsetX, offsetY);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_switch_flag_extend_v_min(RGYFrameInfo *dst, const uint8_t *src, int srcPitch,
    int width, int height, int offsetX, int offsetY, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !src) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_switch_flag_extend_v_min<<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], src, srcPitch,
        width, height, offsetX, offsetY);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_contains_combe_init(uint32_t *count, cudaStream_t stream) {
    if (!count) {
        return RGY_ERR_INVALID_CALL;
    }
    kernel_kfm_contains_combe_init<<<1, 1, 0, stream>>>(count);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_contains_combe_count(const RGYFrameInfo *mask, uint32_t *count, int threshold, cudaStream_t stream) {
    if (!mask || !mask->ptr[0] || !count) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(mask->width, (int)block.x), divCeil(mask->height, (int)block.y));
    kernel_kfm_contains_combe_count<<<grid, block, 0, stream>>>((const uint8_t *)mask->ptr[0], mask->pitch[0],
        count, mask->width, mask->height, threshold);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_contains_combe_mark(RGYFrameInfo *dst, const uint32_t *count, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !count) {
        return RGY_ERR_INVALID_CALL;
    }
    kernel_kfm_contains_combe_mark<<<1, 4, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], count);
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_kfm_combe_mask_resize_bilinear_min_plane(RGYFrameInfo *dst, const RGYFrameInfo *flag,
    int srcStep, int srcOffset, int scaleX, int shiftX, int scaleY, int shiftY,
    int innerWidth, int innerHeight, cudaStream_t stream) {
    if (!dst || !flag || !dst->ptr[0] || !flag->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(dst,
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(dst->width / std::max(1, srcStep), (int)block.x), divCeil(dst->height, (int)block.y));
            kernel_kfm_combe_mask_resize_bilinear_min<uint8_t><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
                (const uint8_t *)flag->ptr[0], flag->pitch[0], dst->width / std::max(1, srcStep), dst->height,
                srcStep, srcOffset, scaleX, shiftX, scaleY, shiftY, innerWidth, innerHeight);
            return err_to_rgy(cudaGetLastError());
        },
        [&]() {
            const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
            const dim3 grid(divCeil(dst->width / std::max(1, srcStep), (int)block.x), divCeil(dst->height, (int)block.y));
            kernel_kfm_combe_mask_resize_bilinear_min<uint16_t><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0],
                (const uint8_t *)flag->ptr[0], flag->pitch[0], dst->width / std::max(1, srcStep), dst->height,
                srcStep, srcOffset, scaleX, shiftX, scaleY, shiftY, innerWidth, innerHeight);
            return err_to_rgy(cudaGetLastError());
        });
}

RGY_ERR run_kfm_copy_u8_buffer_to_plane(RGYFrameInfo *dst, const uint8_t *src, int srcPitch,
    int width, int height, cudaStream_t stream) {
    if (!dst || !dst->ptr[0] || !src) {
        return RGY_ERR_INVALID_CALL;
    }
    const dim3 block(KFM_PAD_BLOCK_X, KFM_PAD_BLOCK_Y);
    const dim3 grid(divCeil(width, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_copy_u8_buffer_to_plane<<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], src, srcPitch, width, height);
    return err_to_rgy(cudaGetLastError());
}

static constexpr int KFM_UCF_BLOCK_X = 32;
static constexpr int KFM_UCF_BLOCK_Y = 8;
static constexpr int KFM_UCF_LOCAL_SIZE = 512;

#define KFM_UCF_COPY_KERNEL(kernel_name) \
template<typename Type> \
__global__ void kernel_name( \
    uint8_t *dst, \
    const int dstPitch, \
    const uint8_t *src, \
    const int srcPitch, \
    const int width, \
    const int height) { \
    const int x = blockIdx.x * blockDim.x + threadIdx.x; \
    const int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= width || y >= height) { \
        return; \
    } \
    const Type *pSrc = (const Type *)(src + y * srcPitch + x * (int)sizeof(Type)); \
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type)); \
    pDst[0] = pSrc[0]; \
}

KFM_UCF_COPY_KERNEL(kernel_kfm_ucf)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_noise)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_param)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_30)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_24)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_60_flag)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_60)

#undef KFM_UCF_COPY_KERNEL

template<typename Type>
__device__ __forceinline__ int kfm_ucf_read_pix(
    const uint8_t *ptr,
    const int x,
    const int y,
    const int pitch,
    const int width,
    const int height) {
    const int ix = clamp(x, 0, width - 1);
    const int iy = clamp(y, 0, height - 1);
    const Type *p = (const Type *)(ptr + iy * pitch + ix * (int)sizeof(Type));
    return (int)p[0];
}

template<typename Type>
__device__ __forceinline__ int kfm_ucf_read_field_crop_pix(
    const uint8_t *src,
    const int x,
    const int y,
    const int srcPitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep) {
    const int ix = clamp(x, 0, width - 1);
    const int iy = clamp(y, 0, height - 1);
    const int sx = ix + srcXOffset;
    const int sy = iy * srcYStep + srcYOffset;
    const Type *p = (const Type *)(src + sy * srcPitch + sx * (int)sizeof(Type));
    return (int)p[0];
}

template<typename Type>
__device__ __forceinline__ int kfm_ucf_read_pix_uv_interleaved(
    const uint8_t *ptr,
    const int chromaX,
    const int channel,
    const int y,
    const int pitch,
    const int chromaWidth,
    const int height) {
    const int ix = clamp(chromaX, 0, chromaWidth - 1);
    const int iy = clamp(y, 0, height - 1);
    const Type *p = (const Type *)(ptr + iy * pitch + ((ix << 1) + channel) * (int)sizeof(Type));
    return (int)p[0];
}

template<typename Type>
__device__ __forceinline__ void kfm_ucf_write_pix(
    uint8_t *ptr,
    const int x,
    const int y,
    const int pitch,
    const int value) {
    Type *p = (Type *)(ptr + y * pitch + x * (int)sizeof(Type));
    p[0] = (Type)value;
}

template<typename Type>
__global__ void kernel_kfm_ucf_field_crop(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int srcX = x + srcXOffset;
    const int srcY = y * srcYStep + srcYOffset;
    const Type *pSrc = (const Type *)(src + srcY * srcPitch + srcX * (int)sizeof(Type));
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = pSrc[0];
}

template<typename Type>
__global__ void kernel_kfm_ucf_gaussresize_v(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int bitDepth,
    const int *offset,
    const float *coeff,
    const int filterSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int begin = offset[y];
    const int maxValue = (1 << bitDepth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix<Type>(src, x, begin + i, srcPitch, width, height) * coeff[y * filterSize + i];
    }
    kfm_ucf_write_pix<Type>(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

template<typename Type>
__global__ void kernel_kfm_ucf_field_crop_gaussresize_v(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int bitDepth,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep,
    const int *offset,
    const float *coeff,
    const int filterSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int begin = offset[y];
    const int maxValue = (1 << bitDepth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_field_crop_pix<Type>(src, x, begin + i, srcPitch, width, height, srcXOffset, srcYOffset, srcYStep) * coeff[y * filterSize + i];
    }
    kfm_ucf_write_pix<Type>(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

template<typename Type>
__global__ void kernel_kfm_ucf_gaussresize_h(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int bitDepth,
    const int *offset,
    const float *coeff,
    const int filterSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int begin = offset[x];
    const int maxValue = (1 << bitDepth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix<Type>(src, begin + i, y, srcPitch, width, height) * coeff[x * filterSize + i];
    }
    kfm_ucf_write_pix<Type>(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

template<typename Type>
__global__ void kernel_kfm_ucf_gaussresize_h_uv_interleaved(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const int width,
    const int height,
    const int bitDepth,
    const int chromaWidth,
    const int *offset,
    const float *coeff,
    const int filterSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int chromaX = x >> 1;
    const int channel = x & 1;
    const int begin = offset[chromaX];
    const int maxValue = (1 << bitDepth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix_uv_interleaved<Type>(src, begin + i, channel, y, srcPitch, chromaWidth, height) * coeff[chromaX * filterSize + i];
    }
    kfm_ucf_write_pix<Type>(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

template<typename Type>
__device__ __forceinline__ int4 kfm_ucf_to_int4(const typename KfmVec4Traits<Type>::Type4 v) {
    return KfmVec4Traits<Type>::to_int4(v);
}

template<typename Type>
__device__ __forceinline__ uint64_t kfm_ucf_hsum4u(const int4 v) {
    return (uint64_t)(v.x + v.y + v.z + v.w);
}

template<typename Type>
__device__ __forceinline__ int4 kfm_ucf_calc_combe4(
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
__global__ void kernel_kfm_ucf_analyze_noise_partial(
    RGYKFM::NoiseResult *dst,
    const int dstOffset,
    const uint8_t *src0,
    const uint8_t *src1,
    const uint8_t *src2,
    const int srcPitch,
    const int bitDepth,
    const int width4,
    const int height) {
    const int groupLinear = blockIdx.x + gridDim.x * blockIdx.y;
    const int localLinear = threadIdx.x + blockDim.x * threadIdx.y;
    const int localSize = blockDim.x * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uint64_t lsum0[KFM_UCF_LOCAL_SIZE];
    __shared__ uint64_t lsum1[KFM_UCF_LOCAL_SIZE];
    __shared__ uint64_t lsumR0[KFM_UCF_LOCAL_SIZE];
    __shared__ uint64_t lsumR1[KFM_UCF_LOCAL_SIZE];

    uint64_t sum0 = 0;
    uint64_t sum1 = 0;
    uint64_t sumR0 = 0;
    uint64_t sumR1 = 0;
    if (x < width4 && y < height) {
        const int pitch4 = srcPitch / (int)sizeof(typename KfmVec4Traits<Type>::Type4);
        const int off = x + y * pitch4;
        const typename KfmVec4Traits<Type>::Type4 *f0 = (const typename KfmVec4Traits<Type>::Type4 *)src0;
        const typename KfmVec4Traits<Type>::Type4 *f1 = (const typename KfmVec4Traits<Type>::Type4 *)src1;
        const typename KfmVec4Traits<Type>::Type4 *f2 = (const typename KfmVec4Traits<Type>::Type4 *)src2;
        const int4 s0 = KfmVec4Traits<Type>::to_int4(f0[off]);
        const int4 s1 = KfmVec4Traits<Type>::to_int4(f1[off]);
        const int4 s2 = KfmVec4Traits<Type>::to_int4(f2[off]);
        const int neutral = 1 << max(bitDepth - 1, 0);
        const int4 neutral4 = make_int4(neutral, neutral, neutral, neutral);
        sum0 = kfm_ucf_hsum4u<Type>(kfm_i4_abs(kfm_i4_sub(s0, neutral4)));
        sum1 = kfm_ucf_hsum4u<Type>(kfm_i4_abs(kfm_i4_sub(s1, neutral4)));
        sumR0 = kfm_ucf_hsum4u<Type>(kfm_i4_abs(kfm_i4_sub(s1, s0)));
        sumR1 = kfm_ucf_hsum4u<Type>(kfm_i4_abs(kfm_i4_sub(s2, s1)));
    }

    lsum0[localLinear] = sum0;
    lsum1[localLinear] = sum1;
    lsumR0[localLinear] = sumR0;
    lsumR1[localLinear] = sumR1;
    __syncthreads();

    for (int step = localSize >> 1; step > 0; step >>= 1) {
        if (localLinear < step) {
            lsum0[localLinear] += lsum0[localLinear + step];
            lsum1[localLinear] += lsum1[localLinear + step];
            lsumR0[localLinear] += lsumR0[localLinear + step];
            lsumR1[localLinear] += lsumR1[localLinear + step];
        }
        __syncthreads();
    }

    if (localLinear == 0) {
        RGYKFM::NoiseResult result;
        result.noise0 = lsum0[0];
        result.noise1 = lsum1[0];
        result.noiseR0 = lsumR0[0];
        result.noiseR1 = lsumR1[0];
        result.diff0 = 0;
        result.diff1 = 0;
        dst[dstOffset + groupLinear] = result;
    }
}

template<typename Type>
__global__ void kernel_kfm_ucf_analyze_diff_partial(
    RGYKFM::NoiseResult *dst,
    const int dstOffset,
    const uint8_t *src0,
    const uint8_t *src1,
    const int srcPitch,
    const int bitDepth,
    const int width4,
    const int height,
    const int srcYOffset) {
    const int groupLinear = blockIdx.x + gridDim.x * blockIdx.y;
    const int localLinear = threadIdx.x + blockDim.x * threadIdx.y;
    const int localSize = blockDim.x * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uint64_t ldiff0[KFM_UCF_LOCAL_SIZE];
    __shared__ uint64_t ldiff1[KFM_UCF_LOCAL_SIZE];

    uint64_t sum0 = 0;
    uint64_t sum1 = 0;
    if (x < width4 && y < height) {
        const int pitch4 = srcPitch / (int)sizeof(typename KfmVec4Traits<Type>::Type4);
        const typename KfmVec4Traits<Type>::Type4 *f0 = (const typename KfmVec4Traits<Type>::Type4 *)src0;
        const typename KfmVec4Traits<Type>::Type4 *f1 = (const typename KfmVec4Traits<Type>::Type4 *)src1;
        const int yy = y + srcYOffset;
        const typename KfmVec4Traits<Type>::Type4 a0 = f0[x + (yy - 2) * pitch4];
        const typename KfmVec4Traits<Type>::Type4 b0 = f0[x + (yy - 1) * pitch4];
        const typename KfmVec4Traits<Type>::Type4 c0 = f0[x + yy * pitch4];
        const typename KfmVec4Traits<Type>::Type4 d0 = f0[x + (yy + 1) * pitch4];
        const typename KfmVec4Traits<Type>::Type4 e0 = f0[x + (yy + 2) * pitch4];
        sum0 = kfm_ucf_hsum4u<Type>(kfm_ucf_calc_combe4<Type>(a0, b0, c0, d0, e0));

        if (y & 1) {
            const typename KfmVec4Traits<Type>::Type4 a = f0[x + (yy - 2) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 b = f1[x + (yy - 1) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 c = f0[x + yy * pitch4];
            const typename KfmVec4Traits<Type>::Type4 d = f1[x + (yy + 1) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 e = f0[x + (yy + 2) * pitch4];
            sum1 = kfm_ucf_hsum4u<Type>(kfm_ucf_calc_combe4<Type>(a, b, c, d, e));
        } else {
            const typename KfmVec4Traits<Type>::Type4 a = f1[x + (yy - 2) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 b = f0[x + (yy - 1) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 c = f1[x + yy * pitch4];
            const typename KfmVec4Traits<Type>::Type4 d = f0[x + (yy + 1) * pitch4];
            const typename KfmVec4Traits<Type>::Type4 e = f1[x + (yy + 2) * pitch4];
            sum1 = kfm_ucf_hsum4u<Type>(kfm_ucf_calc_combe4<Type>(a, b, c, d, e));
        }
    }

    ldiff0[localLinear] = sum0;
    ldiff1[localLinear] = sum1;
    __syncthreads();

    for (int step = localSize >> 1; step > 0; step >>= 1) {
        if (localLinear < step) {
            ldiff0[localLinear] += ldiff0[localLinear + step];
            ldiff1[localLinear] += ldiff1[localLinear + step];
        }
        __syncthreads();
    }

    if (localLinear == 0) {
        RGYKFM::NoiseResult result;
        result.noise0 = 0;
        result.noise1 = 0;
        result.noiseR0 = 0;
        result.noiseR1 = 0;
        result.diff0 = ldiff0[0];
        result.diff1 = ldiff1[0];
        dst[dstOffset + groupLinear] = result;
    }
}

__device__ __forceinline__ int kfm_ucf_limiter(const int x, const int neutral, const int maxValue, const int nmin, const int range) {
    if (x == neutral) {
        return neutral;
    }
    if (x < neutral) {
        return (((neutral - 1 - range) < x) & (x < (neutral - nmin))) ? 0 : ((56 * neutral) >> 7);
    }
    return (((neutral + nmin) < x) & (x < (neutral + 1 + range))) ? maxValue : ((199 * neutral) >> 7);
}

template<typename Type>
__global__ void kernel_kfm_ucf_noise_limit(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const uint8_t *noise,
    const int noisePitch,
    const int width,
    const int height,
    const int bitDepth,
    const int nmin,
    const int range) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const Type *pSrc = (const Type *)(src + y * srcPitch + x * (int)sizeof(Type));
    const Type *pNoise = (const Type *)(noise + y * noisePitch + x * (int)sizeof(Type));
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type));

    const int neutral = 1 << max(bitDepth - 1, 0);
    const int maxValue = (1 << bitDepth) - 1;
    const int v = ((int)pSrc[0] - (int)pNoise[0] + neutral * 2) >> 1;
    const int scaledNmin = nmin << max(bitDepth - 8, 0);
    const int scaledRange = range << max(bitDepth - 8, 0);
    pDst[0] = (Type)clamp(kfm_ucf_limiter(v, neutral, maxValue, scaledNmin, scaledRange), 0, maxValue);
}

template<typename Type>
__global__ void kernel_kfm_ucf_source_crop_noise_limit(
    uint8_t *dst,
    const int dstPitch,
    const uint8_t *src,
    const int srcPitch,
    const uint8_t *noise,
    const int noisePitch,
    const int width,
    const int height,
    const int bitDepth,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep,
    const int nmin,
    const int range) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int sx = x + srcXOffset;
    const int sy = y * srcYStep + srcYOffset;
    const Type *pSrc = (const Type *)(src + sy * srcPitch + sx * (int)sizeof(Type));
    const Type *pNoise = (const Type *)(noise + y * noisePitch + x * (int)sizeof(Type));
    Type *pDst = (Type *)(dst + y * dstPitch + x * (int)sizeof(Type));

    const int neutral = 1 << max(bitDepth - 1, 0);
    const int maxValue = (1 << bitDepth) - 1;
    const int v = ((int)pSrc[0] - (int)pNoise[0] + neutral * 2) >> 1;
    const int scaledNmin = nmin << max(bitDepth - 8, 0);
    const int scaledRange = range << max(bitDepth - 8, 0);
    pDst[0] = (Type)clamp(kfm_ucf_limiter(v, neutral, maxValue, scaledNmin, scaledRange), 0, maxValue);
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_field_crop_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, int srcXOffset, int srcYOffset, int srcYStep, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_field_crop<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height, srcXOffset, srcYOffset, srcYStep);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_gaussresize_v_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, int bitDepth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_gaussresize_v<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height, bitDepth, offset, coeff, filterSize);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_field_crop_gaussresize_v_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, int bitDepth, int srcXOffset, int srcYOffset, int srcYStep, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_field_crop_gaussresize_v<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height, bitDepth, srcXOffset, srcYOffset, srcYStep, offset, coeff, filterSize);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_gaussresize_h_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, int bitDepth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_gaussresize_h<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height, bitDepth, offset, coeff, filterSize);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_gaussresize_h_uv_interleaved_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, int bitDepth, int chromaWidth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_gaussresize_h_uv_interleaved<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height, bitDepth, chromaWidth, offset, coeff, filterSize);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_analyze_noise_partial_t(RGYKFM::NoiseResult *dst, int dstOffset, const RGYFrameInfo *src0, const RGYFrameInfo *src1, const RGYFrameInfo *src2, int bitDepth, int width4, int height, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(width4, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_ucf_analyze_noise_partial<Type><<<grid, block, 0, stream>>>(dst, dstOffset, (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], (const uint8_t *)src2->ptr[0], src0->pitch[0], bitDepth, width4, height);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_analyze_diff_partial_t(RGYKFM::NoiseResult *dst, int dstOffset, const RGYFrameInfo *src0, const RGYFrameInfo *src1, int bitDepth, int width4, int height, int srcYOffset, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(width4, (int)block.x), divCeil(height, (int)block.y));
    kernel_kfm_ucf_analyze_diff_partial<Type><<<grid, block, 0, stream>>>(dst, dstOffset, (const uint8_t *)src0->ptr[0], (const uint8_t *)src1->ptr[0], src0->pitch[0], bitDepth, width4, height, srcYOffset);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_noise_limit_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int nmin, int range, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_noise_limit<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], (const uint8_t *)noise->ptr[0], noise->pitch[0], dst->width, dst->height, RGY_CSP_BIT_DEPTH[dst->csp], nmin, range);
    return err_to_rgy(cudaGetLastError());
}

template<typename Type>
static RGY_ERR launch_kfm_ucf_source_crop_noise_limit_plane_t(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int bitDepth, int srcXOffset, int srcYOffset, int srcYStep, int nmin, int range, cudaStream_t stream) {
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    kernel_kfm_ucf_source_crop_noise_limit<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], (const uint8_t *)noise->ptr[0], noise->pitch[0], dst->width, dst->height, bitDepth, srcXOffset, srcYOffset, srcYStep, nmin, range);
    return err_to_rgy(cudaGetLastError());
}

#define KFM_UCF_COPY_WRAPPER(wrapper_name, kernel_name) \
template<typename Type> \
static RGY_ERR launch_##wrapper_name##_t(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream) { \
    const dim3 block(KFM_UCF_BLOCK_X, KFM_UCF_BLOCK_Y); \
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y)); \
    kernel_name<Type><<<grid, block, 0, stream>>>((uint8_t *)dst->ptr[0], dst->pitch[0], (const uint8_t *)src->ptr[0], src->pitch[0], dst->width, dst->height); \
    return err_to_rgy(cudaGetLastError()); \
} \
RGY_ERR wrapper_name(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream) { \
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0]) { \
        return RGY_ERR_INVALID_CALL; \
    } \
    return dispatch_kfm_depth(dst, \
        [&]() { return launch_##wrapper_name##_t<uint8_t>(dst, src, stream); }, \
        [&]() { return launch_##wrapper_name##_t<uint16_t>(dst, src, stream); }); \
}

KFM_UCF_COPY_WRAPPER(run_kfm_ucf, kernel_kfm_ucf)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_noise, kernel_kfm_ucf_noise)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_param, kernel_kfm_ucf_param)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_30, kernel_kfm_ucf_30)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_24, kernel_kfm_ucf_24)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_60_flag, kernel_kfm_ucf_60_flag)
KFM_UCF_COPY_WRAPPER(run_kfm_ucf_60, kernel_kfm_ucf_60)

#undef KFM_UCF_COPY_WRAPPER

RGY_ERR run_kfm_ucf_field_crop(RGYFrameInfo *dst, const RGYFrameInfo *src, int srcXOffset, int srcYOffset, int srcYStep, cudaStream_t stream) {
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_field_crop_plane_t<uint8_t>(dst, src, srcXOffset, srcYOffset, srcYStep, stream); },
        [&]() { return launch_kfm_ucf_field_crop_plane_t<uint16_t>(dst, src, srcXOffset, srcYOffset, srcYStep, stream); });
}

RGY_ERR run_kfm_ucf_gaussresize_v(RGYFrameInfo *dst, const RGYFrameInfo *src, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0] || !offset || !coeff) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[dst->csp];
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_gaussresize_v_plane_t<uint8_t>(dst, src, bitDepth, offset, coeff, filterSize, stream); },
        [&]() { return launch_kfm_ucf_gaussresize_v_plane_t<uint16_t>(dst, src, bitDepth, offset, coeff, filterSize, stream); });
}

RGY_ERR run_kfm_ucf_field_crop_gaussresize_v(RGYFrameInfo *dst, const RGYFrameInfo *src, int srcXOffset, int srcYOffset, int srcYStep, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0] || !offset || !coeff) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[dst->csp];
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_field_crop_gaussresize_v_plane_t<uint8_t>(dst, src, bitDepth, srcXOffset, srcYOffset, srcYStep, offset, coeff, filterSize, stream); },
        [&]() { return launch_kfm_ucf_field_crop_gaussresize_v_plane_t<uint16_t>(dst, src, bitDepth, srcXOffset, srcYOffset, srcYStep, offset, coeff, filterSize, stream); });
}

RGY_ERR run_kfm_ucf_gaussresize_h(RGYFrameInfo *dst, const RGYFrameInfo *src, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0] || !offset || !coeff) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[dst->csp];
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_gaussresize_h_plane_t<uint8_t>(dst, src, bitDepth, offset, coeff, filterSize, stream); },
        [&]() { return launch_kfm_ucf_gaussresize_h_plane_t<uint16_t>(dst, src, bitDepth, offset, coeff, filterSize, stream); });
}

RGY_ERR run_kfm_ucf_gaussresize_h_uv_interleaved(RGYFrameInfo *dst, const RGYFrameInfo *src, int chromaWidth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    if (!dst || !src || !dst->ptr[0] || !src->ptr[0] || !offset || !coeff) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[dst->csp];
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_gaussresize_h_uv_interleaved_plane_t<uint8_t>(dst, src, bitDepth, chromaWidth, offset, coeff, filterSize, stream); },
        [&]() { return launch_kfm_ucf_gaussresize_h_uv_interleaved_plane_t<uint16_t>(dst, src, bitDepth, chromaWidth, offset, coeff, filterSize, stream); });
}

RGY_ERR run_kfm_ucf_analyze_noise_partial(RGYKFM::NoiseResult *dst, int dstOffset, const RGYFrameInfo *src0, const RGYFrameInfo *src1, const RGYFrameInfo *src2, int width4, int height, cudaStream_t stream) {
    if (!dst || !src0 || !src1 || !src2 || !src0->ptr[0] || !src1->ptr[0] || !src2->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[src0->csp];
    return dispatch_kfm_depth(src0,
        [&]() { return launch_kfm_ucf_analyze_noise_partial_t<uint8_t>(dst, dstOffset, src0, src1, src2, bitDepth, width4, height, stream); },
        [&]() { return launch_kfm_ucf_analyze_noise_partial_t<uint16_t>(dst, dstOffset, src0, src1, src2, bitDepth, width4, height, stream); });
}

RGY_ERR run_kfm_ucf_analyze_diff_partial(RGYKFM::NoiseResult *dst, int dstOffset, const RGYFrameInfo *src0, const RGYFrameInfo *src1, int width4, int height, int srcYOffset, cudaStream_t stream) {
    if (!dst || !src0 || !src1 || !src0->ptr[0] || !src1->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[src0->csp];
    return dispatch_kfm_depth(src0,
        [&]() { return launch_kfm_ucf_analyze_diff_partial_t<uint8_t>(dst, dstOffset, src0, src1, bitDepth, width4, height, srcYOffset, stream); },
        [&]() { return launch_kfm_ucf_analyze_diff_partial_t<uint16_t>(dst, dstOffset, src0, src1, bitDepth, width4, height, srcYOffset, stream); });
}

RGY_ERR run_kfm_ucf_noise_limit(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int nmin, int range, cudaStream_t stream) {
    if (!dst || !src || !noise || !dst->ptr[0] || !src->ptr[0] || !noise->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_noise_limit_plane_t<uint8_t>(dst, src, noise, nmin, range, stream); },
        [&]() { return launch_kfm_ucf_noise_limit_plane_t<uint16_t>(dst, src, noise, nmin, range, stream); });
}

RGY_ERR run_kfm_ucf_source_crop_noise_limit(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int srcXOffset, int srcYOffset, int srcYStep, int nmin, int range, cudaStream_t stream) {
    if (!dst || !src || !noise || !dst->ptr[0] || !src->ptr[0] || !noise->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return dispatch_kfm_depth(dst,
        [&]() { return launch_kfm_ucf_source_crop_noise_limit_plane_t<uint8_t>(dst, src, noise, RGY_CSP_BIT_DEPTH[dst->csp], srcXOffset, srcYOffset, srcYStep, nmin, range, stream); },
        [&]() { return launch_kfm_ucf_source_crop_noise_limit_plane_t<uint16_t>(dst, src, noise, RGY_CSP_BIT_DEPTH[dst->csp], srcXOffset, srcYOffset, srcYStep, nmin, range, stream); });
}

RGY_ERR run_kfm_ucf_copy_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    return run_kfm_ucf(dst, src, stream);
}

RGY_ERR run_kfm_ucf_field_crop_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, int srcXOffset, int srcYOffset, int srcYStep, cudaStream_t stream) {
    return run_kfm_ucf_field_crop(dst, src, srcXOffset, srcYOffset, srcYStep, stream);
}

RGY_ERR run_kfm_ucf_gaussresize_v_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    return run_kfm_ucf_gaussresize_v(dst, src, offset, coeff, filterSize, stream);
}

RGY_ERR run_kfm_ucf_field_crop_gaussresize_v_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, int srcXOffset, int srcYOffset, int srcYStep, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    return run_kfm_ucf_field_crop_gaussresize_v(dst, src, srcXOffset, srcYOffset, srcYStep, offset, coeff, filterSize, stream);
}

RGY_ERR run_kfm_ucf_gaussresize_h_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    return run_kfm_ucf_gaussresize_h(dst, src, offset, coeff, filterSize, stream);
}

RGY_ERR run_kfm_ucf_gaussresize_h_uv_interleaved_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, int chromaWidth, const int *offset, const float *coeff, int filterSize, cudaStream_t stream) {
    return run_kfm_ucf_gaussresize_h_uv_interleaved(dst, src, chromaWidth, offset, coeff, filterSize, stream);
}

RGY_ERR run_kfm_ucf_noise_limit_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int nmin, int range, cudaStream_t stream) {
    return run_kfm_ucf_noise_limit(dst, src, noise, nmin, range, stream);
}

RGY_ERR run_kfm_ucf_source_crop_noise_limit_plane(RGYFrameInfo *dst, const RGYFrameInfo *src, const RGYFrameInfo *noise, int srcXOffset, int srcYOffset, int srcYStep, int nmin, int range, cudaStream_t stream) {
    return run_kfm_ucf_source_crop_noise_limit(dst, src, noise, srcXOffset, srcYOffset, srcYStep, nmin, range, stream);
}
