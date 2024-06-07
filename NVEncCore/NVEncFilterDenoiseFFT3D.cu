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

#include <array>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (pop)

#define FFT_M_PI (3.14159265358979323846f)

static __device__ constexpr int log2u(int n) {
    int x = -1;
    while (n > 0) {
        x++;
        n >>= 1;
    }
    return x;
}

// intのbitを逆順に並び替える
template<int N>
static __device__ constexpr int bitreverse(int x) {
    int y = 0;
    for (int i = 0; i < N; i++) {
        y = (y << 1) + (x & 1);
        x >>= 1;
    }
    return y;
}

template<typename T, bool forward>
static __device__ const complex<T> fw(const int k, const int N) {
    // cexp<T>(complex<T>(0.0f, -2.0f * FFT_M_PI * k / (float)N));
    const float theta = ((forward) ? -2.0f : +2.0f) * FFT_M_PI * k / (float)N;
    return complex<T>(std::cos(theta), std::sin(theta));
}

template<typename T, bool forward>
static __device__ complex<T> fft_calc0(complex<T> c0, complex<T> c1, const int k, const int N) {
    return c0 + fw<T, forward>(k, N) * c1;
}
template<typename T, bool forward>
static __device__ complex<T> fft_calc1(complex<T> c0, complex<T> c1, const int k, const int N) {
    return c0 - fw<T, forward>(k, N) * c1;
}

template<typename T, int N, int step>
static __device__ void fftpermute(complex<T> *data) {
    complex<T> work[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        work[i] = data[i * step];
    }
    if (N > WARP_SIZE) {
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < N; i++) {
        data[i * step] = work[bitreverse<log2u(N)>(i)];
    }
    return;
}

template<typename T, int N, bool forward, int step>
static __device__ void fft(complex<T> *data) {
    if (N >= 4) {
        fft<T, N / 2, forward, step>(data);
        fft<T, N / 2, forward, step>(data + (N / 2) * step);
    }
    complex<T> work[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        work[i] = data[i * step];
    }
    
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
        data[(i        ) * step] = fft_calc0<T, forward>(work[i], work[i + N / 2], i, N);
        data[(i + N / 2) * step] = fft_calc1<T, forward>(work[i], work[i + N / 2], i, N);
    }
}

template<typename T, int N, int step>
static __device__ void ifft_normalize(complex<T> *data) {
    const float invN = 1.0f / (float)N;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        data[i * step] *= invN;
    }
}

template<> static __device__ void fft<float2,  1, true,   1>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, true,   1>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, true,   9>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, true,   9>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, true,  17>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, true,  17>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, true,  33>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, true,  33>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, true,  65>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, true,  65>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, false,  1>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, false,  1>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, false,  9>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, false,  9>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, false, 17>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, false, 17>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, false, 33>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, false, 33>(complex<__half2> *data) { return; }
template<> static __device__ void fft<float2,  1, false, 65>(complex<float2> *data) { return; }
template<> static __device__ void fft<__half2, 1, false, 65>(complex<__half2> *data) { return; }

template<typename T, int N, bool forward, int step>
static __device__ void dft(complex<T> *data) {
    if (N <= 2 || N == 4 || N == 8) {
        fftpermute<T, N, step>(data);
        fft<T, N, forward, step>(data);
        if (!forward) {
            ifft_normalize<T, N, step>(data);
        }
        return;
    }
    complex<T> work[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        work[i] = complex<T>(0.0f, 0.0f);
    }
    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int k = 0; k < N; k++) {
            work[k] += data[i * step] * fw<T, forward>(i*k, N);
        }
    }
    if (!forward) {
        ifft_normalize<T, N, 1>(work);
    }
    #pragma unroll
    for (int i = 0; i < N; i++) {
        data[i * step] = work[i];
    }
}

#define BLOCK_SYNC { if (BLOCK_SIZE > WARP_SIZE) { __syncthreads(); } else { __syncwarp(); } }

template<typename T, int BLOCK_SIZE>
__device__ void fftBlock(complex<T> shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], const int thWorker) {
    // x方向の変換
    fftpermute<T, BLOCK_SIZE,       1>(&shared_tmp[thWorker][0]); BLOCK_SYNC;
    fft<       T, BLOCK_SIZE, true, 1>(&shared_tmp[thWorker][0]); BLOCK_SYNC;
    // y方向の変換
    fftpermute<T, BLOCK_SIZE,       BLOCK_SIZE+1>(&shared_tmp[0][thWorker]); BLOCK_SYNC;
    fft<       T, BLOCK_SIZE, true, BLOCK_SIZE+1>(&shared_tmp[0][thWorker]);
}

template<typename T, int BLOCK_SIZE>
__device__ void ifftBlock(complex<T> shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], const int thWorker) {
    // y方向の逆変換
    fftpermute<T, BLOCK_SIZE,        BLOCK_SIZE + 1>(&shared_tmp[0][thWorker]); BLOCK_SYNC;
    fft<       T, BLOCK_SIZE, false, BLOCK_SIZE + 1>(&shared_tmp[0][thWorker]); BLOCK_SYNC;
    ifft_normalize<T, BLOCK_SIZE,    BLOCK_SIZE + 1>(&shared_tmp[0][thWorker]); BLOCK_SYNC;
    // x方向の逆変換
    fftpermute<T, BLOCK_SIZE,        1>(&shared_tmp[thWorker][0]); BLOCK_SYNC;
    fft<       T, BLOCK_SIZE, false, 1>(&shared_tmp[thWorker][0]); BLOCK_SYNC;
    ifft_normalize<T, BLOCK_SIZE,    1>(&shared_tmp[thWorker][0]);
}

template<typename TypeTmp, int BLOCK_SIZE>
__device__ void thresholdBlock(TypeTmp shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], int thWorker, const float threshold) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (y > 0 || thWorker > 0) {
            TypeTmp *ptr = &shared_tmp[y][thWorker];
            const TypeTmp val = ptr[0];
            if (fabs(val) <= threshold) {
                ptr[0] = 0.0f;
            }
        }
    }
}

template<typename TypePixel, int bit_depth, typename TypeComplex, int BLOCK_SIZE, int DENOISE_BLOCK_SIZE_X>
__global__ void kernel_fft(
    char *const __restrict__ ptrDst0,
    char *const __restrict__ ptrDst1,
    const int dstPitch,
    const char *const __restrict__ ptrSrc0,
    const char *const __restrict__ ptrSrc1,
    const int srcPitch,
    const int width, const int height,
    const int block_count_x,
    const float *const __restrict__ ptrBlockWindow,
    const int ov1, const int ov2
) {
    const int thWorker = threadIdx.x; // BLOCK_SIZE
    const int local_bx = threadIdx.y; // DENOISE_BLOCK_SIZE_X
    const int global_bx = blockIdx.x * DENOISE_BLOCK_SIZE_X + local_bx;
    const int global_by = blockIdx.y;
    const int plane_idx = blockIdx.z;

    const int block_eff = BLOCK_SIZE - ov1 - ov1 - ov2;
    const int block_x = global_bx * block_eff - ov1 - ov2;
    const int block_y = global_by * block_eff - ov1 - ov2;

    char *const __restrict__ ptrDst = selectptr2(ptrDst0, ptrDst1, plane_idx);
    const char *const __restrict__ ptrSrc = selectptr2(ptrSrc0, ptrSrc1, plane_idx);
#if 1
    __shared__ complex<TypeComplex> stmp[DENOISE_BLOCK_SIZE_X][BLOCK_SIZE][BLOCK_SIZE + 1];

    // stmpにptrSrcの該当位置からデータを読み込む
    {
        const float winFuncX = ptrBlockWindow[thWorker];
        #pragma unroll
        for (int y = 0; y < BLOCK_SIZE; y++) {
            if (global_bx < block_count_x) {
                const int src_x = wrap_idx(block_x + thWorker, 0, width - 1);
                const int src_y = wrap_idx(block_y + y, 0, height - 1);
                const TypePixel *ptr_src = (const TypePixel *)(ptrSrc + src_y * srcPitch + src_x * sizeof(TypePixel));
                stmp[local_bx][y][thWorker] = complex<TypeComplex>((float)ptr_src[0] * winFuncX * ptrBlockWindow[y] * (1.0f / (float)((1 << bit_depth) - 1)), 0.0f);
            }
        }
    }
    __syncthreads();

    fftBlock<TypeComplex, BLOCK_SIZE>(stmp[local_bx], thWorker);

    __syncthreads();

    // 計算内容をptrDstに出力
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int dst_x = global_bx * BLOCK_SIZE + thWorker;
            const int dst_y = global_by * BLOCK_SIZE + y;
            complex<TypeComplex> *ptr_dst = (complex<TypeComplex> *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(complex<TypeComplex>));
            ptr_dst[0] = stmp[local_bx][y][thWorker];
        }
    }
#else
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int dst_x = global_bx * BLOCK_SIZE + thWorker;
            const int dst_y = global_by * BLOCK_SIZE + y;
            const TypePixel *ptr_src = (const TypePixel *)(ptrSrc + dst_y * srcPitch + dst_x * sizeof(TypePixel));
            complex<TypeComplex> *ptr_dst = (complex<TypeComplex> *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(complex<TypeComplex>));
            ptr_dst[0].v.x = (float)ptr_src[0] * (1.0f / (float)((1 << bit_depth) - 1));
        }
    }
#endif
}

static std::pair<int, int> getBlockCount(const int width, const int height, const int block_size, const int ov1, const int ov2) {
    const int block_eff = block_size - ov1 * 2 - ov2;
    const int block_count_x = (width + block_eff - 1) / block_eff;
    const int block_count_y = (height + block_eff - 1) / block_eff;
    return std::make_pair(block_count_x, block_count_y);
}

template<typename Type, int bit_depth, typename TypeComplex, int BLOCK_SIZE, int DENOISE_BLOCK_SIZE_X>
RGY_ERR denoise_fft(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int ov1, const int ov2,
    const float *ptrBlockWindow, cudaStream_t stream) {
    {
        auto block_count = getBlockCount(pInputFrame->width, pInputFrame->height, BLOCK_SIZE, ov1, ov2);
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        dim3 blockSize(BLOCK_SIZE, DENOISE_BLOCK_SIZE_X);
        dim3 gridSize(divCeil(block_count.first, DENOISE_BLOCK_SIZE_X), block_count.second, 1);
        kernel_fft<Type, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputY.ptr[0], nullptr, planeOutputY.pitch[0],
            (const char *)planeInputY.ptr[0], nullptr, planeInputY.pitch[0],
            planeInputY.width, planeInputY.height, block_count.first,
            ptrBlockWindow,
            ov1, ov2
        );
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        auto block_count = getBlockCount(planeInputU.width, planeInputU.height, BLOCK_SIZE, ov1, ov2);
        dim3 blockSize(BLOCK_SIZE, DENOISE_BLOCK_SIZE_X);
        dim3 gridSize(divCeil(block_count.first, DENOISE_BLOCK_SIZE_X), block_count.second, 2);
        kernel_fft<Type, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputU.ptr[0], (char *)planeOutputV.ptr[0], planeOutputU.pitch[0],
            (const char *)planeInputU.ptr[0], (const char *)planeInputV.ptr[0], planeInputU.pitch[0],
            planeInputU.width, planeInputU.height, block_count.first,
            ptrBlockWindow,
            ov1, ov2
        );
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

template<typename TypeComplex, int temporalCurrentIdx, int temporalCount>
__device__ complex<TypeComplex> temporal_filter(
    const complex<TypeComplex> *ptrSrcA,
    const complex<TypeComplex> *ptrSrcB,
    const complex<TypeComplex> *ptrSrcC,
    const complex<TypeComplex> *ptrSrcD,
    const float sigma, const float limit, const int filterMethod) {
    static_assert(1 <= temporalCount && temporalCount <= 4, "temporalCount must be 1 to 4.");
    static_assert(0 <= temporalCurrentIdx && temporalCurrentIdx < temporalCount, "temporalCurrentIdx must be 0 to temporalCount.");
    complex<TypeComplex> work[temporalCount];
    work[0] = ptrSrcA[0];
    if (temporalCount >= 2) { work[1] = ptrSrcB[0]; }
    if (temporalCount >= 3) { work[2] = ptrSrcC[0]; }
    if (temporalCount >= 4) { work[3] = ptrSrcD[0]; }

    if (temporalCount >= 2) {
        dft<TypeComplex, temporalCount, true, 1>(work);
    }

    #pragma unroll
    for (int z = 0; z < temporalCount; z++) {
        const float power = work[z].squaref();

        float factor;
        if (filterMethod == 0) {
            factor = fmaxf(limit, (power - sigma) * __frcp_rn(power + 1e-15f));
        } else {
            factor = power < sigma ? limit : 1.0f;
        }
        work[z] *= factor;
    }

    if (temporalCount >= 2) {
        dft<TypeComplex, temporalCount, false, 1>(work);
    }

    return work[temporalCurrentIdx];
}

template<typename TypePixel, int bit_depth, typename TypeComplex, int BLOCK_SIZE, int DENOISE_BLOCK_SIZE_X,
    int temporalCurrentIdx, int temporalCount>
__global__ void kernel_tfft_filter_ifft(
    char *const __restrict__ ptrDst0,
    char *const __restrict__ ptrDst1,
    const int dstPitch,
    const char *const __restrict__ ptrSrcA0,
    const char *const __restrict__ ptrSrcA1,
    const char *const __restrict__ ptrSrcB0,
    const char *const __restrict__ ptrSrcB1,
    const char *const __restrict__ ptrSrcC0,
    const char *const __restrict__ ptrSrcC1,
    const char *const __restrict__ ptrSrcD0,
    const char *const __restrict__ ptrSrcD1,
    const int srcPitch,
    const int block_count_x,
    const float *const __restrict__ ptrBlockWindowInverse,
    const int ov1, const int ov2,
    const float sigma, const float limit, const int filterMethod
) {
    static_assert(1 <= temporalCount && temporalCount <= 4, "temporalCount must be 1 to 4.");
    const int thWorker = threadIdx.x; // BLOCK_SIZE
    const int local_bx = threadIdx.y; // DENOISE_BLOCK_SIZE_X
    const int global_bx = blockIdx.x * DENOISE_BLOCK_SIZE_X + local_bx;
    const int global_by = blockIdx.y;
    const int plane_idx = blockIdx.z;

    char *const __restrict__ ptrDst = selectptr2(ptrDst0, ptrDst1, plane_idx);
    const char *const __restrict__ ptrSrcA = selectptr2(ptrSrcA0, ptrSrcA1, plane_idx);
    const char *const __restrict__ ptrSrcB = (temporalCount >= 2) ? selectptr2(ptrSrcB0, ptrSrcB1, plane_idx) : nullptr;
    const char *const __restrict__ ptrSrcC = (temporalCount >= 3) ? selectptr2(ptrSrcC0, ptrSrcC1, plane_idx) : nullptr;
    const char *const __restrict__ ptrSrcD = (temporalCount >= 4) ? selectptr2(ptrSrcD0, ptrSrcD1, plane_idx) : nullptr;
#if 1
    __shared__ complex<TypeComplex> stmp[DENOISE_BLOCK_SIZE_X][BLOCK_SIZE][BLOCK_SIZE + 1];
#if 1
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int src_x = global_bx * BLOCK_SIZE + thWorker;
            const int src_y = global_by * BLOCK_SIZE + y;
            const int src_idx = src_y * srcPitch + src_x * sizeof(complex<TypeComplex>);
            stmp[local_bx][y][thWorker] = temporal_filter<TypeComplex, temporalCurrentIdx, temporalCount>(
                (const complex<TypeComplex> *)(ptrSrcA + src_idx),
                (const complex<TypeComplex> *)(ptrSrcB + src_idx),
                (const complex<TypeComplex> *)(ptrSrcC + src_idx),
                (const complex<TypeComplex> *)(ptrSrcD + src_idx),
                sigma, limit, filterMethod);
        }
    }
#else
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int src_x = global_bx * BLOCK_SIZE + thWorker;
            const int src_y = global_by * BLOCK_SIZE + y;
            const complex<TypeComplex> *ptr_src = (const complex<TypeComplex> *)(ptrSrcA + src_y * srcPitch + src_x * sizeof(complex<TypeComplex>));
            stmp[local_bx][y][thWorker] = ptr_src[0];
        }
    }
#endif
    __syncthreads();

    ifftBlock<TypeComplex, BLOCK_SIZE>(stmp[local_bx], thWorker);

    __syncthreads();
    {
        // 計算内容をptrDstに出力
        const float winFuncInvX = ptrBlockWindowInverse[thWorker];
        #pragma unroll
        for (int y = 0; y < BLOCK_SIZE; y++) {
            if (global_bx < block_count_x) {
                const int dst_x = global_bx * BLOCK_SIZE + thWorker;
                const int dst_y = global_by * BLOCK_SIZE + y;
                TypePixel *ptr_dst = (TypePixel *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(TypePixel));
                ptr_dst[0] = (TypePixel)clamp(stmp[local_bx][y][thWorker].realf() * winFuncInvX * ptrBlockWindowInverse[y] * ((float)((1 << bit_depth) - 1)), 0.0f, (1 << bit_depth) - 1e-6f);
            }
        }
    }
#else
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int dst_x = global_bx * BLOCK_SIZE + thWorker;
            const int dst_y = global_by * BLOCK_SIZE + y;
            const complex<TypeComplex> *ptr_src = (const complex<TypeComplex> *)(ptrSrcA + dst_y * srcPitch + dst_x * sizeof(complex<TypeComplex>));
            TypePixel *ptr_dst = (TypePixel *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(TypePixel));

            ptr_dst[0] = (TypePixel)clamp(ptr_src[0].v.x * (float)((1 << bit_depth) - 1), 0.0f, (1 << bit_depth) - 1e-6f);
        }
    }
#endif
}

template<typename TypePixel, int bit_depth, typename TypeComplex, int BLOCK_SIZE, int DENOISE_BLOCK_SIZE_X, int temporalCurrentIdx, int temporalCount>
RGY_ERR denoise_tfft_filter_ifft(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrameA, const RGYFrameInfo *pInputFrameB, const RGYFrameInfo *pInputFrameC, const RGYFrameInfo *pInputFrameD,
    const float *ptrBlockWindowInverse,
    const int widthY, const int heightY, const int widthUV, const int heightUV, const int ov1, const int ov2,
    const float sigma, const float limit, const int filterMethod, cudaStream_t stream) {
    {
        const auto block_count = getBlockCount(widthY, heightY, BLOCK_SIZE, ov1, ov2);
        const auto planeInputYA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_Y) : RGYFrameInfo();
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        dim3 blockSize(BLOCK_SIZE, DENOISE_BLOCK_SIZE_X);
        dim3 gridSize(divCeil(block_count.first, DENOISE_BLOCK_SIZE_X), block_count.second, 1);
        kernel_tfft_filter_ifft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X, temporalCurrentIdx, temporalCount> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputY.ptr[0], nullptr,
            planeOutputY.pitch[0],
            (const char *)planeInputYA.ptr[0], nullptr,
            (const char *)planeInputYB.ptr[0], nullptr,
            (const char *)planeInputYC.ptr[0], nullptr,
            (const char *)planeInputYD.ptr[0], nullptr,
            planeInputYA.pitch[0],
            block_count.first,
            ptrBlockWindowInverse,
            ov1, ov2,
            sigma * (1.0f / ((1 << 8) - 1)), limit, filterMethod
        );
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        const auto planeInputUA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_V) : RGYFrameInfo();
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        const auto block_count = getBlockCount(widthUV, heightUV, BLOCK_SIZE, ov1, ov2);
        dim3 blockSize(BLOCK_SIZE, DENOISE_BLOCK_SIZE_X);
        dim3 gridSize(divCeil(block_count.first, DENOISE_BLOCK_SIZE_X), block_count.second, 2);
        kernel_tfft_filter_ifft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X, temporalCurrentIdx, temporalCount> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputU.ptr[0], (char *)planeOutputV.ptr[0],
            planeOutputU.pitch[0],
            (const char *)planeInputUA.ptr[0], (const char *)planeInputVA.ptr[0],
            (const char *)planeInputUB.ptr[0], (const char *)planeInputVB.ptr[0],
            (const char *)planeInputUC.ptr[0], (const char *)planeInputVC.ptr[0],
            (const char *)planeInputUD.ptr[0], (const char *)planeInputVD.ptr[0],
            planeInputUA.pitch[0],
            block_count.first,
            ptrBlockWindowInverse,
            ov1, ov2,
            sigma * (1.0f / ((1 << 8) - 1)), limit, filterMethod
        );
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

template<typename TypePixel, int BLOCK_SIZE>
__global__ void kernel_merge(
    char *const __restrict__ ptrDst0,
    char *const __restrict__ ptrDst1,
    const int dstPitch,
    const char *const __restrict__ ptrSrc0,
    const char *const __restrict__ ptrSrc1,
    const int srcPitch,
    const int width, const int height,
    const int block_count_x, const int block_count_y,
    const int ov1, const int ov2
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int plane_idx = blockIdx.z;
    const int block_eff = BLOCK_SIZE - ov1 - ov1 - ov2;

    char *const __restrict__ ptrDst = selectptr2(ptrDst0, ptrDst1, plane_idx);
    const char *const __restrict__ ptrSrc = selectptr2(ptrSrc0, ptrSrc1, plane_idx);

    if (x < width && y < height) {
        const int block_x = x / block_eff;
        const int block_y = y / block_eff;
        const int block_local_pos_x = x - block_x * block_eff + ov1 + ov2;
        const int block_local_pos_y = y - block_y * block_eff + ov1 + ov2;
        int shift = 0;
#define BLOCK_VAL(x, y) (((TypePixel *)(ptrSrc + (y) * srcPitch + (x) * sizeof(TypePixel)))[0])
#if 1
        int pix = BLOCK_VAL(block_x * BLOCK_SIZE + block_local_pos_x, block_y * BLOCK_SIZE + block_local_pos_y);
        if (block_local_pos_x >= block_eff + ov1 && (block_x + 1) < block_count_x) {
            pix += BLOCK_VAL((block_x + 1) * BLOCK_SIZE + block_local_pos_x - block_eff, block_y * BLOCK_SIZE + block_local_pos_y);
            shift++;
        }
        if (block_local_pos_y >= block_eff + ov1 && (block_y + 1) < block_count_y) {
            pix += BLOCK_VAL(block_x * BLOCK_SIZE + block_local_pos_x, (block_y + 1) * BLOCK_SIZE + block_local_pos_y - block_eff);
            shift++;
            if (block_local_pos_x >= block_eff + ov1 && (block_x + 1) < block_count_x) {
                pix += BLOCK_VAL((block_x + 1) * BLOCK_SIZE + block_local_pos_x - block_eff, (block_y + 1) * BLOCK_SIZE + block_local_pos_y - block_eff);
            }
        }
#else
        int pix = BLOCK_VAL(x, y);
#endif
#undef BLOCK_VAL
        ((TypePixel *)(ptrDst + y * dstPitch + x * sizeof(TypePixel)))[0] = (TypePixel)((pix + shift) >> shift);
    }
}

template<typename TypePixel, int BLOCK_SIZE>
RGY_ERR denoise_merge(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int ov1, const int ov2, cudaStream_t stream) {
    {
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        const auto block_count = getBlockCount(planeOutputY.width, planeOutputY.height, BLOCK_SIZE, ov1, ov2);
        dim3 blockSize(32, 8);
        dim3 gridSize(divCeil(planeOutputY.width, blockSize.x), divCeil(planeOutputY.height, blockSize.y), 1);
        kernel_merge<TypePixel, BLOCK_SIZE> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputY.ptr[0], nullptr, planeOutputY.pitch[0],
            (const char *)planeInputY.ptr[0], nullptr, planeInputY.pitch[0],
            planeOutputY.width, planeOutputY.height, block_count.first, block_count.second, ov1, ov2);
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        const auto block_count = getBlockCount(planeOutputU.width, planeOutputU.height, BLOCK_SIZE, ov1, ov2);
        dim3 blockSize(32, 8);
        dim3 gridSize(divCeil(planeOutputU.width, blockSize.x), divCeil(planeOutputU.height, blockSize.y), 2);
        kernel_merge<TypePixel, BLOCK_SIZE> << <gridSize, blockSize, 0, stream >> > (
            (char *)planeOutputU.ptr[0], (char *)planeOutputV.ptr[0], planeOutputU.pitch[0],
            (const char *)planeInputU.ptr[0], (const char *)planeInputV.ptr[0], planeInputU.pitch[0],
            planeOutputU.width, planeOutputU.height, block_count.first, block_count.second, ov1, ov2);
        CUDA_DEBUG_SYNC_ERR;
        auto err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

class DenoiseFFT3DBase {
public:
    DenoiseFFT3DBase() {};
    virtual ~DenoiseFFT3DBase() {};

    virtual decltype(&denoise_fft<uint8_t, 8, float2, 8, 8>) fft() = 0;
    virtual decltype(&denoise_tfft_filter_ifft<uint8_t, 8, float2, 8, 8, 0, 1>) tfft_filter_ifft(int temporalCurrentIdx, int temporalCount) = 0;
    virtual decltype(&denoise_merge<uint8_t, 8>) merge() = 0;
};

template<typename TypePixel, int bit_depth, typename TypeComplex, int BLOCK_SIZE, int DENOISE_BLOCK_SIZE_X>
class DenoiseFFT3DFuncs : public DenoiseFFT3DBase {
public:
    DenoiseFFT3DFuncs() {};
    virtual ~DenoiseFFT3DFuncs() {};

    virtual decltype(&denoise_fft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X>) fft() override { return denoise_fft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X>; }
    virtual decltype(&denoise_tfft_filter_ifft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X, 0, 1>) tfft_filter_ifft(int temporalCurrentIdx, int temporalCount) override {
        if (temporalCount == 1) {
            return denoise_tfft_filter_ifft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X, 0, 1>;
        } else if (temporalCount == 3) {
            if (temporalCurrentIdx == 1) {
                return denoise_tfft_filter_ifft<TypePixel, bit_depth, TypeComplex, BLOCK_SIZE, DENOISE_BLOCK_SIZE_X, 1, 3>;
            }
        }
        return nullptr;
    }
    virtual decltype(&denoise_merge<uint8_t, 8>) merge() override { return denoise_merge<TypePixel, BLOCK_SIZE>; }
};

std::unique_ptr<DenoiseFFT3DBase> getDenoiseFunc(const RGY_CSP csp, const int block_size, const VppFpPrecision prec) {
    switch (csp) {
    case RGY_CSP_YV12:
    case RGY_CSP_YUV444:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            switch (block_size) {
            case  8: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2,  8, 8>());
            case 16: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 16, 4>());
            case 32: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 32, 2>());
            case 64: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 64, 1>());
            default: return nullptr;
            }
        } else {
            switch (block_size) {
            case  8: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, __half2,  8, 8>());
            case 16: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, __half2, 16, 4>());
            case 32: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, __half2, 32, 2>());
            case 64: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, __half2, 64, 1>());
            default: return nullptr;
            }
        }
    case RGY_CSP_YV12_16:
    case RGY_CSP_YUV444_16:
        if (prec == VppFpPrecision::VPP_FP_PRECISION_FP32) {
            switch (block_size) {
            case  8: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, float2,  8, 8>());
            case 16: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, float2, 16, 4>());
            case 32: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, float2, 32, 2>());
            case 64: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, float2, 64, 1>());
            default: return nullptr;
            }
        } else {
            switch (block_size) {
            case  8: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, __half2,  8, 8>());
            case 16: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, __half2, 16, 4>());
            case 32: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, __half2, 32, 2>());
            case 64: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint16_t, 16, __half2, 64, 1>());
            default: return nullptr;
            }
        }
    default:
        return nullptr;
    }
}

RGY_ERR NVEncFilterDenoiseFFT3DBuffer::alloc(int width, int height, RGY_CSP csp, int frames) {
    m_bufFFT.resize(frames);
    for (auto& buf : m_bufFFT) {
        if (!buf || buf->frame.width != width || buf->frame.height != height || buf->frame.csp != csp) {
            buf = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
            auto sts = buf->alloc(width, height, csp);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

NVEncFilterDenoiseFFT3D::NVEncFilterDenoiseFFT3D() :
    m_bufIdx(0),
    m_ov1(0),
    m_ov2(0),
    m_bufFFT(),
    m_filteredBlocks(),
    m_windowBuf(),
    m_windowBufInverse() {
    m_name = _T("denoise-fft");
}

NVEncFilterDenoiseFFT3D::~NVEncFilterDenoiseFFT3D() {
    close();
}

RGY_ERR NVEncFilterDenoiseFFT3D::checkParam(const NVEncFilterParamDenoiseFFT3D *prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sigma < 0.0f || 100.0f < prm->fft3d.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma must be 0 - 100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.amount < 0.0f || 1.0f < prm->fft3d.amount) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, strength must be 0 - 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fft3d_block_size, prm->fft3d.block_size) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap < 0.0f || 0.8f < prm->fft3d.overlap) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap2 < 0.0f || 0.8f < prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap2 must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (0.8f < prm->fft3d.overlap + prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sum of overlap and overlap2 must be below 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.method != 0 && prm->fft3d.method != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, method must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.temporal != 0 && prm->fft3d.temporal != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, temporal must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fp_prec, prm->fft3d.precision) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid precision.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseFFT3D::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 && prm->compute_capability.first < 7) {
        prm->fft3d.precision = VppFpPrecision::VPP_FP_PRECISION_FP32;
    }
    if (!m_param
        || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size
        || prm->fft3d.overlap != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap
        || prm->fft3d.overlap2 != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap2
        || prm->fft3d.temporal != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.temporal
        || prm->fft3d.precision != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
        || cmpFrameInfoCspResolution(&m_param->frameOut, &prm->frameOut)) {
        m_ov1 = (int)(prm->fft3d.block_size * 0.5 * prm->fft3d.overlap + 0.5);
        m_ov2 = (int)(prm->fft3d.block_size * 0.5 * (prm->fft3d.overlap + prm->fft3d.overlap2) + 0.5) - m_ov1;

        //より小さいUVに合わせてブロック数を計算し、そこから確保するメモリを決める
        auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);
        const auto blocksUV = getBlockCount(planeUV.width, planeUV.height, prm->fft3d.block_size, m_ov1, m_ov2);
        const int complexSize = (prm->fft3d.precision == VppFpPrecision::VPP_FP_PRECISION_FP32) ? 8 : 4;

        RGY_CSP fft_csp = RGY_CSP_NA;
        int blockGlobalWidth = 0, blockGlobalHeight = 0;
        if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) {
            fft_csp = RGY_CSP_YV12;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size * 2;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size * 2;
        } else if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV444) {
            fft_csp = RGY_CSP_YUV444;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
            return RGY_ERR_UNSUPPORTED;
        }

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, prm->fft3d.temporal ? 3 : 1)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
        }

        m_filteredBlocks = std::unique_ptr<CUFrameBuf>(new CUFrameBuf());
        if ((sts = m_filteredBlocks->alloc(blockGlobalWidth, blockGlobalHeight, prm->frameOut.csp)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for filtered blocks: %s.\n"), get_err_mes(sts));
            return sts;
        }

        sts = AllocFrameBuf(prm->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }

        if (!m_param || !m_windowBuf || prm->fft3d.block_size != std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size) {
            std::vector<float> blockWindow(prm->fft3d.block_size);
            std::vector<float> blockWindowInv(prm->fft3d.block_size);
            auto winFunc = [block_size = prm->fft3d.block_size](const int x) { return 0.50f - 0.50f * std::cos(2.0f * FFT_M_PI * x / (float)block_size); };
            for (int i = 0; i < prm->fft3d.block_size; i++) {
                blockWindow[i] = winFunc(i);
                blockWindowInv[i] = 1.0f / blockWindow[i];
            }

            m_windowBuf = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindow.size() * sizeof(blockWindow[0])));
            m_windowBufInverse = std::unique_ptr<CUMemBuf>(new CUMemBuf(blockWindowInv.size() * sizeof(blockWindowInv[0])));

            if ((sts = m_windowBuf->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_windowBufInverse->alloc()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBuf->ptr, blockWindow.data(), blockWindow.size() * sizeof(blockWindow[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = err_to_rgy(cudaMemcpy(m_windowBufInverse->ptr, blockWindowInv.data(), blockWindowInv.size() * sizeof(blockWindowInv[0]), cudaMemcpyHostToDevice))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (prm->fft3d.temporal) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring NVEncFilterParamDenoiseFFT3D::print() const {
    return fft3d.print();
}

RGY_ERR NVEncFilterDenoiseFFT3D::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto denosieFunc = getDenoiseFunc(prm->frameIn.csp, prm->fft3d.block_size, prm->fft3d.precision);
    if (!denosieFunc) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp or block_size.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (!prm->fft3d.temporal || m_nFrameIdx >= m_bufIdx) {
            //終了
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
    } else {
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
        //}
        const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != cudaMemcpyDeviceToDevice) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        auto fftBuf = m_bufFFT.get(m_bufIdx++);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        sts = denosieFunc->fft()(&fftBuf->frame, pInputFrame, m_ov1, m_ov2, (const float *)m_windowBuf->ptr, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    if (prm->fft3d.temporal) {
        if (m_bufIdx <= 1) {
            //出力フレームなし
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
        auto fftPrev = m_bufFFT.get(std::max(m_bufIdx - ((finalOutput) ? 2 : 3), 0));
        auto fftCur  = m_bufFFT.get(m_bufIdx - ((finalOutput) ? 1 : 2));
        auto fftNext = m_bufFFT.get(m_bufIdx - 1);
        sts = denosieFunc->tfft_filter_ifft(1, 3)(&m_filteredBlocks->frame, &fftPrev->frame, &fftCur->frame, &fftNext->frame, nullptr, (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            prm->fft3d.sigma, prm->fft3d.amount, prm->fft3d.method, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(1, 3): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
    } else {
        auto fftCur = m_bufFFT.get(m_bufIdx - 1);
        sts = denosieFunc->tfft_filter_ifft(0, 1)(&m_filteredBlocks->frame, &fftCur->frame, nullptr, nullptr, nullptr, (const float *)m_windowBufInverse->ptr,
            prm->frameOut.width, prm->frameOut.height, planeUV.width, planeUV.height, m_ov1, m_ov2,
            prm->fft3d.sigma, 1.0f - prm->fft3d.amount, prm->fft3d.method, stream);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(0, 1): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
    }
    sts = denosieFunc->merge()(ppOutputFrames[0], &m_filteredBlocks->frame, m_ov1, m_ov2, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run merge: %s.\n"), get_err_mes(sts));
        return RGY_ERR_NONE;
    }

    m_nFrameIdx++;
    return sts;
}

void NVEncFilterDenoiseFFT3D::close() {
    m_frameBuf.clear();
    m_bufFFT.clear();
    m_windowBuf.reset();
    m_windowBufInverse.reset();
}
